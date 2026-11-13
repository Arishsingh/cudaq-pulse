# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GPU simulation backend's setup and interval logic, without a GPU.

``_GpuBackend`` runs unchanged except for ``_execute``, the one call into
cuDensityMat, which the CPU reference integrator stands in for. So these
tests cover everything the GPU path does around the device: lowering the
module before the first interval, choosing pure or mixed states, setting
each interval's measurement path and window, and batching sibling branches.
"""

from __future__ import annotations

import math
import re

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import MicrowaveDrive, PhysicalQudit
from cudaq_pulse.runtime import _jit as jit
from cudaq_pulse.runtime import simulate as simulate_module

from tests._handles import Q
from tests.runtime._lowering import evolve, on

_PI = math.pi / 20.0
_HALF_PI = math.pi / 40.0


class _CpuStandIn(simulate_module._GpuBackend):
    """The GPU backend with the cuDensityMat run replaced by the CPU one."""

    batches: list[int] = []
    path_lengths: list[int] = []

    arguments: list[float] = []
    runs: list[tuple[float, float, int]] = []

    def _execute(self, text, states, mixed):
        header = text.split("\n", 1)[0]
        assert ("qop.mixed_state" in header) == (mixed and not self.dissipative)
        _CpuStandIn.batches.append(len(states))
        # The run's path is the one the module text gives (a batch's first
        # member's).
        path = re.search(r"pulse.path = \[(.*?)\]", header).group(1)
        entries = [[int(v) for v in e.split(",")] for e in re.findall(r"array<i64: ([^>]*)>", path)]
        _CpuStandIn.path_lengths.append(len(entries))
        # The window and steps of the evolution, or of the part of one, that
        # the module lowers: what cuDensityMat would run.
        t0, t1, steps = (
            float(re.search(rf"qop\.{name} = ([^ ,}}]+)", header).group(1))
            for name in ("t_start", "t_end", "step_count")
        )
        self.runs.append((t0, t1, int(steps)))
        results = []
        for state in states:
            assert mixed == (state.ndim == 2)
            self.steps = lambda a, b: int(steps)
            try:
                rho = simulate_module._CpuBackend.evolve(self, self.density(state), entries, t0, t1)
            finally:
                del self.steps
            if mixed:
                results.append(rho)
                continue
            # Without collapse operators the state stays pure: return the vector.
            values, vectors = np.linalg.eigh(rho)
            results.append(math.sqrt(max(values[-1], 0.0)) * vectors[:, -1])
        return results


@pytest.fixture
def gpu_on_cpu(monkeypatch):
    monkeypatch.setattr(jit, "JITCompiler", lambda: None)
    monkeypatch.setattr(simulate_module, "_GpuBackend", _CpuStandIn)
    monkeypatch.setattr(_CpuStandIn, "batches", [])
    monkeypatch.setattr(_CpuStandIn, "path_lengths", [])


def _target(n, t1_us=0.0):
    return pulse.quantum_system(
        name="gpu-logic",
        qudits={
            q: PhysicalQudit(
                index=q,
                frequency_hz=5.0e9 + 0.1e9 * q,
                anharmonicity_hz=-200.0e6,
                t1_us=t1_us,
                t2_star_us=0.0,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
            for q in range(n)
        },
    )


def _active_reset(prepare):
    @pulse.kernel
    def reset(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        drive(d0, square(40, prepare))
        sync(d0, r0)
        bit = readout(r0, square(200, 0.05))
        if bit:
            drive(d0, square(40, _PI))

    return reset


@pulse.kernel
def conditional_flip(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    r0 = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI))
    sync(d0, r0)
    bit = readout(r0, square(200, 0.05))
    if bit:
        drive(d1, square(40, _PI))


@pulse.kernel
def measure(q0):
    r0 = get_readout_line(q0)
    readout(r0, square(200, 0.05))


@pulse.kernel
def repeated_reset(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI))
    for _ in range(3):
        sync(d0, r0)
        bit = readout(r0, square(200, 0.05))
        if bit:
            drive(d0, square(40, _PI))


_P1 = math.exp(-100.0 / 500.0)
_DIAG = np.diag

# (kernel, qubits, simulate() arguments, expected density matrix or None,
# expected branch weights or None)
CASES = {
    "active_reset_excited": (
        _active_reset(_PI),
        1,
        {"shots_count": 200},
        _DIAG([1, 0]),
        {(1,): 1.0},
    ),
    "active_reset_superposition": (
        _active_reset(_HALF_PI),
        1,
        {"shots_count": 400},
        _DIAG([1, 0]),
        {(0,): 0.5, (1,): 0.5},
    ),
    "conditional_flip": (
        conditional_flip,
        2,
        {"shots_count": 100},
        _DIAG([0.5, 0, 0, 0.5]),
        {(0,): 0.5, (1,): 0.5},
    ),
    "repeated_reset_loop": (
        repeated_reset,
        1,
        {"shots_count": 100},
        _DIAG([1, 0]),
        {(0, 0, 0): 0.5, (1, 0, 0): 0.5},
    ),
    "t1_readout_decay": (
        measure,
        1,
        {"target": _target(1, t1_us=0.5), "initial_state": [0.0, 1.0]},
        _DIAG([1 - _P1, _P1]),
        {(0,): 1 - _P1, (1,): _P1},
    ),
    "qutrit_measure": (
        measure,
        1,
        {"levels": 3, "initial_state": np.ones(3) / math.sqrt(3), "shots_count": 300},
        np.eye(3) / 3,
        {(0,): 1 / 3, (1,): 1 / 3, (2,): 1 / 3},
    ),
    "qutrit_active_reset": (_active_reset(_PI), 1, {"levels": 3, "shots_count": 100}, None, None),
    "mixed_dims_conditional_flip": (
        conditional_flip,
        2,
        {"levels": {0: 3, 1: 2}, "shots_count": 100},
        None,
        None,
    ),
}


def _simulate(kernel, n, backend, kwargs):
    kwargs = dict(kwargs)
    target = kwargs.pop("target", None) or _target(n)
    kwargs.setdefault("levels", 2)
    program = (kernel, *[Q[i] for i in range(n)])
    return pulse.simulate(on(target, *program), backend=backend, seed=7, **kwargs)


def _weights(result):
    return {b.levels: b.probability for b in result.branches if b.probability > 1e-9}


# The stand-in keeps pure states as vectors, as the GPU backend does, and
# recovers them from RK4 density matrices, which are rank one only to the
# integrator's accuracy; the GPU tests compare to the reference at 1e-6 too.
_ATOL = 1e-6


@pytest.mark.parametrize("name", CASES)
def test_the_gpu_backend_simulates_a_program_that_measures(name, gpu_on_cpu):
    kernel, n, kwargs, rho, branches = CASES[name]
    gpu = _simulate(kernel, n, "gpu", kwargs)
    cpu = _simulate(kernel, n, "cpu", kwargs)
    np.testing.assert_allclose(gpu.density_matrix, cpu.density_matrix, atol=_ATOL)
    gpu_weights, cpu_weights = _weights(gpu), _weights(cpu)
    assert gpu_weights.keys() == cpu_weights.keys()
    for outcome, p in cpu_weights.items():
        assert gpu_weights[outcome] == pytest.approx(p, abs=_ATOL)
    if rho is not None:
        np.testing.assert_allclose(gpu.density_matrix, rho, atol=_ATOL)
    if branches is not None:
        assert gpu_weights.keys() == branches.keys()
        for outcome, p in branches.items():
            assert gpu_weights[outcome] == pytest.approx(p, abs=_ATOL)
    if "shots_count" in kwargs:
        assert gpu.records.shape == (kwargs["shots_count"], len(gpu.measurements))


@pytest.mark.parametrize("t_start", [0.0, 130.0, 500.0])
@pytest.mark.parametrize("t1_us", [0.0, 0.5])
def test_gpu_setup_lowers_a_program_that_measures_at_any_start(t_start, t1_us, gpu_on_cpu):
    program = (_active_reset(_PI), Q[0])
    target = _target(1, t1_us)
    result = simulate_module._simulate_lowered(
        on(target, *program),
        backend="gpu",
        t_start=t_start,
        t_end=t_start + 200.0,
        levels=2,
    )
    assert np.trace(result.density_matrix).real == pytest.approx(1.0, abs=1e-9)


def test_evolve_of_a_program_that_measures_runs_the_gpu_backend(gpu_on_cpu):
    lowered = on(_target(1), _active_reset(_HALF_PI), Q[0])
    state = evolve(lowered, t_start=0.0, t_end=140.0, step_count=2800).final_state
    rho = np.outer(state, state.conj()) if state.ndim == 1 else state
    np.testing.assert_allclose(rho, np.diag([1.0, 0.0]), atol=1e-7)


@pulse.classical(max_latency_ns=0)
def _first_bit(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=0)
def _same(x: float) -> float:
    return x


@pulse.kernel
def decoded_reset(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    bit = readout(r0, square(200, 0.05))
    sync(d0, r0)
    if _first_bit.submit([bit]).result()[0]:
        drive(d0, square(40, _PI))


@pulse.kernel
def answered_wait(q0, tau: float):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    w = _same.submit(tau).result()
    drive(d0, square(40, _PI))
    wait(d0, w)
    sync(d0, r0)
    readout(r0, square(200, 0.05))


# A program with classical calls evolves in intervals between their answers,
# each lowered with the answers so far bound.
CALL_CASES = {
    "decoded_reset": ((decoded_reset,), {"initial_state": [2**-0.5, 2**-0.5]}),
    "answered_wait": ((answered_wait, 400.0), {}),
}


@pytest.mark.parametrize("name", CALL_CASES)
def test_the_gpu_backend_simulates_a_program_with_classical_calls(name, gpu_on_cpu):
    (kernel, *args), kwargs = CALL_CASES[name]
    lowered = on(_target(1, t1_us=5.0), kernel, Q[0], *args)
    gpu = pulse.simulate(lowered, backend="gpu", levels=2, **kwargs)
    cpu = pulse.simulate(lowered, backend="cpu", levels=2, **kwargs)
    np.testing.assert_allclose(gpu.density_matrix, cpu.density_matrix, atol=_ATOL)
    assert _weights(gpu) == pytest.approx(_weights(cpu), abs=_ATOL)


@pulse.kernel
def measure_then_flip(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    readout(r0, square(200, 0.05))
    sync(d0, r0)
    drive(d0, square(40, _PI))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_late_start_evolves_from_the_start_not_the_measurement(backend, gpu_on_cpu):
    # The readout ends at 100 ns and the pi pulse plays over [100, 120] ns.
    # Started at 120 ns, the evolution must not replay the pulse.
    target = _target(1)
    result = simulate_module._simulate_lowered(
        on(target, measure_then_flip, Q[0]),
        backend=backend,
        t_start=120.0,
        t_end=140.0,
        levels=2,
    )
    np.testing.assert_allclose(result.density_matrix, np.diag([1.0, 0.0]), atol=1e-12)
    assert result.times_ns == (120.0, 140.0)


@pulse.kernel
def measure_drive_measure(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    for _ in range(3):
        drive(d0, square(40, _HALF_PI))
        sync(d0, r0)
        readout(r0, square(200, 0.05))
        sync(d0, r0)
    drive(d0, square(40, _HALF_PI))


@pytest.mark.parametrize("branches", ["all", "sampled"])
def test_branches_without_feedback_evolve_as_one_batch(branches, gpu_on_cpu):
    lowered = on(_target(1, t1_us=5.0), measure_drive_measure, Q[0])
    kwargs = {"levels": 3, "shots_count": 400, "seed": 3, "branches": branches}
    gpu = pulse.simulate(lowered, backend="gpu", **kwargs)
    cpu = pulse.simulate(lowered, backend="cpu", **kwargs)
    # Every path of a depth evolves at once; sampled, the siblings of one
    # measurement that shots take do.
    if branches == "all":
        assert _CpuStandIn.batches == [1, 3, 9, 27]
    else:
        assert _CpuStandIn.batches[:2] == [1, 3] and max(_CpuStandIn.batches) == 3
    np.testing.assert_allclose(gpu.density_matrix, cpu.density_matrix, atol=_ATOL)
    assert _weights(gpu) == pytest.approx(_weights(cpu), abs=_ATOL)


@pulse.classical(max_latency_ns=200)
def _next_wait(survived: pulse.Bits, wait_ns: float) -> float:
    return min(wait_ns + 100.0, 900.0) if survived[0] else max(wait_ns - 100.0, 100.0)


@pulse.kernel
def tracking(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    wait_ns = 400.0
    for _ in range(8):
        drive(d0, square(40, _HALF_PI))
        sync(d0, r0)
        readout(r0, square(200, 0.05))
        sync(d0, r0)
        drive(d0, square(40, _HALF_PI))
        sync(d0, r0)
        m = readout(r0, square(200, 0.05))
        wait_ns = _next_wait.submit([m], wait_ns).result()
        sync(d0, r0)
        wait(d0, wait_ns * pulse.ns, bounds=(100 * pulse.ns, 1 * pulse.us))


def test_siblings_in_a_window_of_a_loop_of_calls_evolve_as_one_batch(gpu_on_cpu):
    lowered = on(_target(1, t1_us=5.0), tracking, Q[0])
    kwargs = {"levels": 3, "shots_count": 40, "seed": 3, "branches": "sampled"}
    gpu = pulse.simulate(lowered, backend="gpu", **kwargs)
    cpu = pulse.simulate(lowered, backend="cpu", **kwargs)
    # Each answer is bound in a window of the program, whose path holds only
    # the measurements in it; the outcomes of a round's first readout run to
    # its second together.
    assert max(_CpuStandIn.path_lengths) <= 3
    assert max(_CpuStandIn.batches) > 1
    np.testing.assert_allclose(gpu.density_matrix, cpu.density_matrix, atol=_ATOL)
    assert _weights(gpu) == pytest.approx(_weights(cpu), abs=_ATOL)
    assert gpu.measurements == cpu.measurements


def test_branches_that_feedback_changes_evolve_apart(gpu_on_cpu):
    gpu = _simulate(repeated_reset, 1, "gpu", {"shots_count": 100})
    cpu = _simulate(repeated_reset, 1, "cpu", {"shots_count": 100})
    # After a readout the branch that read 1 plays a pulse the other does
    # not, so the two evolve apart; after the second, the paths that last
    # read 1 play the same pulse and run together; the others play nothing
    # and idle on the host (_idle.through_gpu) with no run on the device.
    assert _CpuStandIn.batches == [1, 1, 2]
    np.testing.assert_allclose(gpu.density_matrix, cpu.density_matrix, atol=_ATOL)
    assert _weights(gpu) == pytest.approx(_weights(cpu), abs=_ATOL)
