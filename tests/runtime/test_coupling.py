# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coupled qudits in their own rotating frames, against analytic results.

An exchange coupling g (a^dagger b + a b^dagger) between transmons at w_a and
w_b swaps an excitation at 2g when they are resonant, barely at all when they
are detuned by many g, and shifts |11> by the dispersive ZZ rate of
second-order perturbation theory. A residual ZZ entry adds exactly its rate.
"""

from __future__ import annotations

import cmath
import math
import re
import shutil
import subprocess
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline, evaluate_callback
from cudaq_pulse.quantum_systems import (
    Coupling,
    MicrowaveDrive,
    PhysicalQudit,
    ResidualZZ,
    SimulationModel,
)

from tests._handles import Q, rebuilt
from tests.runtime._lowering import on

_TWO_PI_GHZ = 2.0 * math.pi * 1.0e-9


def _pair(f0, f1, *, g_hz=0.0, zz_hz=0.0, alphas=(-200.0e6, -180.0e6)):
    return pulse.quantum_system(
        name="coupled-pair",
        qudits={
            q: PhysicalQudit(
                q,
                f,
                0.0,
                0.0,
                anharmonicity_hz=a,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
            for q, (f, a) in enumerate(zip((f0, f1), alphas, strict=True))
        },
        couplings=[Coupling((0, 1), strength_hz=g_hz)] if g_hz else [],
        residual_zz=[ResidualZZ((0, 1), zz_hz)] if zz_hz else [],
    )


@pulse.kernel
def _idle(q0, q1):
    d0 = get_drive_line(q0)
    get_drive_line(q1)  # simulate the idle partner
    drive(d0, square(2, 0.0))


def _lowered(target):
    return on(target, _idle, Q[0], Q[1])


def _basis(d, n0, n1):
    """Index of |n0, n1> with qudit 0 the least significant digit."""
    return n0 + d * n1


def _swapped(target, levels, times, dt_ns=0.05):
    """P(|0,1>) after starting in |1,0>, at each of *times* (ns)."""
    lowered = _lowered(target)
    psi = np.zeros(levels * levels)
    psi[_basis(levels, 1, 0)] = 1.0
    out = []
    for t in times:
        rho = pulse.simulate(
            lowered, levels=levels, initial_state=psi, t_end=t, dt_ns=dt_ns
        ).density_matrix
        out.append(rho[_basis(levels, 0, 1), _basis(levels, 0, 1)].real)
    return np.array(out)


def _ramsey_zz_hz(target, levels, t_ns, dt_ns):
    """ZZ rate and qudit 0's frequency shift (Hz), from Ramseys with qudit 1
    in |0> and |1>."""
    lowered = _lowered(target)
    phases = []
    for k in (0, 1):
        psi = np.zeros(levels * levels, dtype=complex)
        psi[_basis(levels, 0, k)] = psi[_basis(levels, 1, k)] = 1.0 / math.sqrt(2.0)
        rho = pulse.simulate(
            lowered, levels=levels, initial_state=psi, t_end=t_ns, dt_ns=dt_ns
        ).density_matrix
        phases.append(cmath.phase(rho[_basis(levels, 0, k), _basis(levels, 1, k)]))
    to_hz = 1.0 / (2.0 * math.pi * t_ns * 1.0e-9)
    return (phases[1] - phases[0]) * to_hz, phases[0] * to_hz


def _perturbative_zz_hz(f0, f1, g_hz, a0, a1):
    """zeta = 2 g^2 (a0 + a1) / ((D + a0)(D - a1)), D = f0 - f1."""
    detuning = f0 - f1
    return 2.0 * g_hz**2 * (a0 + a1) / ((detuning + a0) * (detuning - a1))


def _exact_zz_hz(f0, f1, g_hz, a0, a1, levels=4):
    """E11 - E10 - E01 + E00 of the exchange-coupled transmons, diagonalized."""
    a = np.diag(np.sqrt(np.arange(1, levels)), 1)
    n = a.T @ a
    eye = np.eye(levels)

    def mode(f, alpha):
        return f * n + 0.5 * alpha * n @ (n - eye)

    h = (
        np.kron(eye, mode(f0, a0))
        + np.kron(mode(f1, a1), eye)
        + g_hz * (np.kron(a, a.T) + np.kron(a.T, a))
    )
    energies, vectors = np.linalg.eigh(h)

    def dressed(n0, n1):
        return energies[np.argmax(np.abs(vectors[_basis(levels, n0, n1)]))]

    return dressed(1, 1) - dressed(1, 0) - dressed(0, 1) + dressed(0, 0)


@pytest.mark.parametrize("levels", [2, 3])
def test_resonant_pair_swaps_at_twice_the_coupling(levels):
    g_hz = 5.0e6
    g = g_hz * _TWO_PI_GHZ
    times = [12.5, 25.0, 50.0, 75.0]
    swapped = _swapped(_pair(5.0e9, 5.0e9, g_hz=g_hz), levels, times)
    # P(|0,1>) = sin^2(g t): a full swap at pi / (2 g) = 50 ns.
    np.testing.assert_allclose(swapped, np.sin(g * np.array(times)) ** 2, atol=1e-7)


@pytest.mark.parametrize("levels", [2, 3])
def test_detuned_pair_follows_the_generalized_rabi_formula(levels):
    g_hz, detuning_hz = 5.0e6, 40.0e6
    times = [5.0, 10.0, 17.0, 30.0]
    swapped = _swapped(_pair(5.0e9 + detuning_hz, 5.0e9, g_hz=g_hz), levels, times, dt_ns=0.01)
    # In the one-excitation subspace: P = 4g^2 / W^2 sin^2(W t / 2),
    # W = sqrt(D^2 + 4 g^2), at most 6 % here.
    g, detuning = g_hz * _TWO_PI_GHZ, detuning_hz * _TWO_PI_GHZ
    w = math.sqrt(detuning**2 + 4.0 * g**2)
    expected = 4.0 * g**2 / w**2 * np.sin(w * np.array(times) / 2.0) ** 2
    np.testing.assert_allclose(swapped, expected, atol=1e-7)


@pytest.mark.slow
@pytest.mark.parametrize("levels", [2, 3])
def test_transmons_detuned_by_gigahertz_stay_put(levels):
    # A data qubit and an ancilla of a surface-code quantum system: 2 GHz apart,
    # g = 7 MHz. At most (2 g / D)^2 = 5e-5 of the excitation moves.
    times = [50.0, 333.0, 1000.0]
    swapped = _swapped(_pair(3.9e9, 5.9e9, g_hz=7.0e6), levels, times, dt_ns=0.025)
    assert np.all(np.abs(swapped) < 1e-4)


def test_two_level_exchange_has_no_zz():
    zeta, _ = _ramsey_zz_hz(_pair(5.0e9, 5.5e9, g_hz=10.0e6), 2, 200.0, 0.02)
    assert abs(zeta) < 1.0


@pytest.mark.slow
@pytest.mark.parametrize(
    "f0, f1, g_hz, t_ns, dt_ns",
    [
        (5.0e9, 5.5e9, 10.0e6, 300.0, 0.02),
        # Krinner et al. (2022): data qubit and ancilla about 2 GHz apart,
        # J = 7 MHz; the idle ZZ is under about 10 kHz.
        (3.9e9, 5.9e9, 7.0e6, 1500.0, 0.025),
    ],
)
def test_dispersive_zz_matches_perturbation_theory(f0, f1, g_hz, t_ns, dt_ns):
    a0, a1 = -200.0e6, -180.0e6
    zeta, shift = _ramsey_zz_hz(_pair(f0, f1, g_hz=g_hz, alphas=(a0, a1)), 3, t_ns, dt_ns)
    exact = _exact_zz_hz(f0, f1, g_hz, a0, a1)
    assert zeta == pytest.approx(exact, rel=1e-2)
    assert zeta == pytest.approx(_perturbative_zz_hz(f0, f1, g_hz, a0, a1), rel=1e-2)
    # With qudit 1 in |0>, qudit 0 moves by the exchange shift g^2 / D.
    assert shift == pytest.approx(g_hz**2 / (f0 - f1), rel=1e-2)


@pytest.mark.parametrize("levels", [2, 3])
def test_residual_zz_is_the_conditional_frequency_shift(levels):
    zz_hz = 1.0e5
    zeta, shift = _ramsey_zz_hz(_pair(5.0e9, 5.5e9, zz_hz=zz_hz), levels, 1000.0, 0.05)
    assert zeta == pytest.approx(zz_hz, rel=1e-6)
    assert abs(shift) < 1e-3


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_systems_without_a_simulation_model_are_refused(backend):
    target = rebuilt(_pair(5.0e9, 5.5e9, g_hz=10.0e6), model=None)
    with pytest.raises(NotImplementedError, match=r"has no simulation model \(its model is None\)"):
        pulse.simulate(_lowered(target), backend=backend)
    target = target.replace(model=SimulationModel)
    with pytest.raises(NotImplementedError, match="SimulationModel has no simulator"):
        pulse.simulate(_lowered(target), backend=backend)


# The GPU path lowers the same pulse-to-qop operator to cuDensityMat terms:
# rebuild H(t) = i K(t) from K, the sum of the terms on the ket side, and
# compare it with the CPU reference's. The lowering may sum static terms into
# one operator on several modes, with their -i.

_CONSTANT = re.compile(r"(%\S+) = arith\.constant dense<(.*)> : tensor<(\d+)x\d+x2xf64>")
_ELEMENTARY = re.compile(r"(%\S+) = cudm\.create_elementary_op %\S+, (%\S+) \{(.*)\} :")
_PRODUCT = re.compile(r"cudm\.append_elementary_product %\S+, (%\S+), ([^{]+)\{(.*)\} :")
_APPEND = re.compile(r"cudm\.operator_append_term %\S+, %\S+, (%\S+) \{(.*)\} :")


def _attribute(attributes, name):
    found = re.search(rf"{re.escape(name)} = ([^,]+?)( : \w+)?(,|$)", attributes)
    return found.group(1) if found else None


def _integers(attributes, name):
    values = re.search(rf"{name} = array<i\d+: ([^>]*)>", attributes).group(1)
    return [int(v) for v in values.split(",")]


def _cudm_terms(text):
    """The products appended to a cuDensityMat operator: the side of the
    term, the coefficient, the callback, and the factors (matrix, modes,
    dualities) in the order they apply."""
    matrices, elementary, products = {}, {}, {}
    for line in text.splitlines():
        if found := _CONSTANT.search(line):
            data = found.group(2)
            if data.startswith('"0x'):
                # A large constant prints as the hex of its little-endian doubles.
                values = np.frombuffer(bytes.fromhex(data[3:-1]), dtype="<f8")
            else:
                values = np.array(
                    [float(v) for v in re.findall(r"-?[\d.]+(?:[eE][-+]?\d+)?", data)]
                )
            matrices[found.group(1)] = values[0::2] + 1j * values[1::2]
        elif found := _ELEMENTARY.search(line):
            extents = _integers(found.group(3), "mode_extents")
            values = matrices[found.group(2)]
            if "multidiagonal" in found.group(3):
                # One vector per diagonal, element (r, r + offset) at min(r, r + offset).
                (d,) = extents
                matrix = np.zeros((d, d), dtype=complex)
                for k, offset in enumerate(_integers(found.group(3), "diagonal_offsets")):
                    for row in range(max(0, -offset), min(d, d - offset)):
                        matrix[row, row + offset] = values[k * d + min(row, row + offset)]
            else:
                # Column-major: element [column][row].
                d = int(np.prod(extents))
                matrix = values.reshape((d, d)).T
            elementary[found.group(1)] = (matrix, len(extents))
        elif found := _PRODUCT.search(line):
            attributes = found.group(3)
            modes = _integers(attributes, "modes_acted_on")
            dualities = _integers(attributes, "duality")
            names = [name.strip() for name in found.group(2).split(",") if name.strip()]
            factors = []
            for name in names:
                matrix, count = elementary[name]
                factors.append((matrix, modes[:count], dualities[:count]))
                modes, dualities = modes[count:], dualities[count:]
            coefficient = float(_attribute(attributes, "coeff_real")) + 1j * float(
                _attribute(attributes, "coeff_imag")
            )
            params = re.search(r"cudm\.callback_params = array<f64: ([^>]*)>", attributes)
            kind = _attribute(attributes, "cudm.callback_kind")
            callback = (int(kind), [float(p) for p in params.group(1).split(",")]) if kind else None
            products.setdefault(found.group(1), []).append((coefficient, callback, factors))
        elif found := _APPEND.search(line):
            attributes = found.group(2)
            side = int(_attribute(attributes, "duality"))
            term = float(_attribute(attributes, "coeff_real")) + 1j * float(
                _attribute(attributes, "coeff_imag")
            )
            for coefficient, callback, factors in products[found.group(1)]:
                yield side, term * coefficient, callback, factors


def _cudm_hamiltonian(text, dims):
    """The terms of H = i K, K the cuDensityMat operator of a state-vector
    evolution (every term on the ket side)."""
    for side, coefficient, callback, factors in _cudm_terms(text):
        assert side == 0
        operator = np.eye(int(np.prod(dims)), dtype=complex)
        for matrix, modes, dualities in factors:
            assert set(dualities) == {0}
            operator = _embedded(dims, matrix, modes) @ operator
        yield operator, 1j * coefficient, callback


def _cudm_liouvillian(text, dims):
    """The superoperator of a constant cuDensityMat operator on the
    column-major vec(rho): its ket modes, then its bra modes. A factor's bra
    mode acts as rho -> rho M; a bra-side term as rho -> rho P, P its
    product composed as on the ket side."""
    n, size = len(dims), int(np.prod(dims))
    total = np.zeros((size * size,) * 2, dtype=complex)
    for side, coefficient, callback, factors in _cudm_terms(text):
        assert callback is None
        if side == 1:
            product = np.eye(size, dtype=complex)
            for matrix, modes, dualities in factors:
                assert set(dualities) == {0}
                product = _embedded(dims, matrix, modes) @ product
            total += coefficient * np.kron(product.T, np.eye(size))
            continue
        operator = np.eye(size * size, dtype=complex)
        for matrix, modes, dualities in factors:
            count = len(modes)
            tensor = matrix.reshape(tuple(dims[m] for m in modes) * 2, order="F")
            for position, duality in enumerate(dualities):
                if duality:
                    tensor = np.swapaxes(tensor, position, count + position)
            superoperator = tensor.reshape(matrix.shape, order="F")
            places = [m + n * d for m, d in zip(modes, dualities, strict=True)]
            operator = _embedded(tuple(dims) * 2, superoperator, places) @ operator
        total += coefficient * operator
    return total


def _embedded(dims, matrix, modes):
    """*matrix*, on *modes* (the first one's index fastest), on every mode."""
    n = len(dims)
    tensor = matrix.reshape(tuple(dims[m] for m in modes) * 2, order="F")
    rows, columns = list(range(n)), list(range(n, 2 * n))
    operands = [tensor, [rows[m] for m in modes] + [columns[m] for m in modes]]
    for mode in range(n):
        if mode not in modes:
            operands += [np.eye(dims[mode]), [rows[mode], columns[mode]]]
    full = np.einsum(*operands, rows + columns)
    return full.reshape((int(np.prod(dims)),) * 2, order="F")


def _full(dims, on_mode):
    full = np.ones((1, 1), dtype=complex)
    for mode in reversed(range(len(dims))):
        full = np.kron(full, on_mode.get(mode, np.eye(dims[mode])))
    return full


def _cpu_hamiltonian(program, dims):
    for re_, im, kind, params, factors in program[1]:
        on_mode = {}
        for mode, d, data in factors:
            values = np.asarray(data)
            on_mode[mode] = (values[0::2] + 1j * values[1::2]).reshape((d, d), order="F")
        yield _full(dims, on_mode), complex(re_, im), ((kind, list(params)) if kind else None)


def _at(terms, times):
    total = None
    for operator, coefficient, callback in terms:
        values = (
            np.asarray(evaluate_callback(callback[0], callback[1], times))
            if callback
            else np.ones(len(times))
        )
        part = coefficient * values[:, None, None] * operator[None]
        total = part if total is None else total + part
    return total


@pytest.mark.parametrize("operators", ["fused", "banded", "dense"])
@pytest.mark.parametrize("levels", [2, 3, 5])
def test_gpu_lowering_builds_the_cpu_operator(levels, operators, monkeypatch):
    monkeypatch.setenv("CUDAQ_PULSE_GPU_OPERATORS", operators)
    target = _pair(3.9e9, 5.9e9, g_hz=7.0e6, zz_hz=2.0e4)

    @pulse.kernel
    def driven(q0, q1):
        d0 = get_drive_line(q0)
        get_drive_line(q1)
        drive(d0, gaussian(40, 0.05, 10.0))

    lowered = on(target, driven, Q[0], Q[1])
    # A CPU run leaves the module with the attributes of its interval, which
    # the GPU backend lowers the same way.
    pulse.simulate(lowered, levels=levels, t_end=20.0)
    dims = (levels, levels)
    times = [0.0, 3.3, 11.7, 19.9]
    module = lowered._concrete()
    cpu_terms = list(_cpu_hamiltonian(module.operator_program([]), dims))
    pipeline = MLIRPipeline()
    text = pipeline.run_qop_to_cudm(pipeline.run_pulse_to_qop(module.print()))
    gpu_terms = list(_cudm_hamiltonian(text, dims))
    # Fused, the static terms of both qudits are one operator on both modes
    # while they fit; on five levels the number terms are stored banded.
    on_both = re.search(r"create_elementary_op .*mode_extents = array<i64: \d+, \d+>", text)
    assert bool(on_both) == (operators == "fused" and levels < 5)
    assert ("multidiagonal" in text) == (operators != "dense" and levels == 5)
    cpu = _at(cpu_terms, times)
    assert np.abs(cpu).max() > 0.04
    np.testing.assert_allclose(_at(gpu_terms, times), cpu, atol=1e-12)
    # The coupling turns at the frames' difference frequency.
    assert "coupling_0_0" in text


def _constant(name, matrix):
    flat = np.asarray(matrix, dtype=complex).ravel(order="F")
    values = np.empty(2 * flat.size)
    values[0::2], values[1::2] = flat.real, flat.imag
    d = matrix.shape[0]
    data = values.astype("<f8").tobytes().hex().upper()
    return f'%{name} = arith.constant dense<"0x{data}"> : tensor<{d}x{d}x2xf64>'


def _jaynes_cummings(fock, *, g=0.25, kappa=0.3, gamma=0.2):
    """A qop module of a qubit (mode 0) in a cavity of *fock* levels
    (mode 1), damped through sigma-minus and a, and its Lindbladian on the
    column-major vec(rho)."""
    a = np.diag(np.sqrt(np.arange(1.0, fock)), 1).astype(complex)
    s = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    leaves = {"a": (a, 1), "ad": (a.T, 1), "n": (a.T @ a, 1), "s": (s, 0), "sd": (s.T, 0)}
    leaves["sn"] = (s.T @ s, 0)
    two_pi = 2.0 * math.pi
    hamiltonian = [(two_pi, ["n"]), (two_pi * 1.1, ["sn"]), (two_pi * g, ["ad", "s"])]
    hamiltonian.append((two_pi * g, ["a", "sd"]))
    collapse = [(math.sqrt(gamma), ["s"]), (math.sqrt(kappa), ["a"])]
    lines = []
    for name, (matrix, mode) in leaves.items():
        d = matrix.shape[0]
        lines.append(_constant(f"c_{name}", matrix))
        lines.append(
            f"%{name} = qop.matrix_leaf(%c_{name}) {{dimensions = array<i64: {d}>, "
            f"targets = array<i64: {mode}>}} : tensor<{d}x{d}x2xf64> -> !qop.handler"
        )

    def product(label, coefficient, names):
        scalar = f"{{real = {coefficient!r} : f64, imag = 0.0 : f64}}"
        lines.append(f"%k_{label} = qop.const_scalar {scalar} : !qop.scalar")
        factors = ", ".join(f"%{name}" for name in names)
        lines.append(f"%p_{label} = qop.make_product(%k_{label}, {factors}) : !qop.product")
        return f"%p_{label}"

    terms = [product(f"h{i}", c, names) for i, (c, names) in enumerate(hamiltonian)]
    jumps = []
    for i, (c, names) in enumerate(collapse):
        lines.append(f"%j{i} = qop.make_sum({product(f'c{i}', c, names)}) : !qop.op")
        jumps.append(f"%j{i}")
    lines.append(f"%h = qop.make_sum({', '.join(terms)}) : !qop.op")
    lines.append(f"%l = qop.lindblad(%h, {', '.join(jumps)}) : !qop.super_op")
    lines.append("qop.evolve %l : !qop.super_op")
    text = (
        f'module attributes {{qop.integrator = "rk4", qop.mode_extents = array<i64: 2, {fock}>, '
        "qop.step_count = 10 : i64, qop.t_end = 1.0 : f64, qop.t_start = 0.0 : f64} {\n"
        "func.func @main() {\n" + "\n".join(lines) + "\nreturn\n}\n}\n"
    )

    def full(names):
        on_mode = {leaves[name][1]: leaves[name][0] for name in names}
        return np.kron(on_mode.get(1, np.eye(fock)), on_mode.get(0, np.eye(2)))

    eye = np.eye(2 * fock)
    h = sum(c * full(names) for c, names in hamiltonian)
    liouvillian = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
    for c, names in collapse:
        jump = c * full(names)
        norm = jump.conj().T @ jump
        liouvillian += np.kron(jump.conj(), jump)
        liouvillian -= 0.5 * (np.kron(eye, norm) + np.kron(norm.T, eye))
    return text, liouvillian


# An open cavity of four or more levels is stored by its diagonals unless the
# lowering is dense, and fused while qubit and cavity fit max-fused-dimension:
# every lowering is the Lindbladian, so it keeps the trace.
@pytest.mark.parametrize("operators", ["fused", "banded", "dense"])
@pytest.mark.parametrize("fock", [6, 10])
def test_gpu_lowering_of_an_open_cavity_is_its_lindbladian(fock, operators, monkeypatch):
    monkeypatch.setenv("CUDAQ_PULSE_GPU_OPERATORS", operators)
    qop, expected = _jaynes_cummings(fock)
    text = MLIRPipeline().run_qop_to_cudm(qop)
    assert ("multidiagonal" in text) == (operators != "dense")
    on_both = re.search(rf"create_elementary_op .*mode_extents = array<i64: 2, {fock}>", text)
    assert bool(on_both) == (operators == "fused" and 2 * fock <= 16)
    found = _cudm_liouvillian(text, (2, fock))
    scale = np.abs(expected).max()
    np.testing.assert_allclose(found, expected, rtol=0, atol=1e-13 * scale)
    trace = np.eye(2 * fock).ravel(order="F")
    np.testing.assert_allclose(trace @ found, 0, atol=1e-13 * scale)


# A build tree stages the package in build/<dir>/python and cudaq-pulse-opt in
# build/<dir>/bin; an installation puts the cudaq-pulse-tools wheel's on PATH.
_BUILD_DRIVER = Path(pulse.__file__).parent.parent.parent / "bin" / "cudaq-pulse-opt"
_UPSTREAM_PASSES = ["--cse", "--canonicalize", "--symbol-dce", "--sccp", "--inline"]


@pytest.mark.parametrize(
    "passes",
    [[name] for name in _UPSTREAM_PASSES] + [_UPSTREAM_PASSES],
    ids=[name[2:] for name in _UPSTREAM_PASSES] + ["all"],
)
def test_upstream_passes_keep_the_operator(passes):
    # The partner is coupled but never driven: its drive line is dead and the
    # passes erase it, so its frame is the quantum system's frequency.
    driver = str(_BUILD_DRIVER) if _BUILD_DRIVER.exists() else shutil.which("cudaq-pulse-opt")
    if driver is None:
        pytest.skip("cudaq-pulse-opt is neither in the build tree nor on PATH")

    def optimized(text):
        return subprocess.run(
            [driver, *passes], input=text, capture_output=True, text=True, timeout=60, check=True
        ).stdout

    @pulse.kernel
    def driven(q0, q1):
        d0 = get_drive_line(q0)
        get_drive_line(q1)
        drive(d0, gaussian(40, 0.05, 10.0))

    lowered = on(_pair(3.9e9, 5.9e9, g_hz=7.0e6, zz_hz=2.0e4), driven, Q[0], Q[1])
    pulse.simulate(lowered, levels=3, t_end=20.0)
    source = lowered._concrete().print()
    pipeline = MLIRPipeline()
    reference = pipeline.run_qop_to_cudm(pipeline.run_pulse_to_qop(source))
    qop = optimized(pipeline.run_pulse_to_qop(optimized(source)))
    assert "qop.evolve" in qop
    text = pipeline.run_qop_to_cudm(qop)
    dims, times = (3, 3), [0.0, 3.3, 11.7, 19.9]
    expected = _at(list(_cudm_hamiltonian(reference, dims)), times)
    assert np.abs(expected).max() > 0.04
    found = _at(list(_cudm_hamiltonian(text, dims)), times)
    np.testing.assert_allclose(found, expected, atol=1e-12)
    assert text.count("cudm.evolve") == reference.count("cudm.evolve") == 1
