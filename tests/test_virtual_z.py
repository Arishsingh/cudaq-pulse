# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The phase gates Z, S and T: virtual Z rotations, checked against the
simulated state. The state a simulation returns is in the frame, so the
logical state is read by tomography pulses played after the gate."""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.gates import S, T, X, Z
from cudaq_pulse.quantum_systems import DragPulse, MicrowaveDrive, PhysicalQudit

BACKENDS = ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]
PHASES = {Z: math.pi, S: math.pi / 2, T: math.pi / 4}


@pytest.fixture(scope="module")
def qs():
    # No decoherence and two levels: the rotations are exact up to the integrator.
    qubit = PhysicalQudit(
        0,
        5.0e9,
        0.0,
        0.0,
        anharmonicity_hz=-2.0e8,
        drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
        calibrations={X: DragPulse(duration_ns=40.0, sigma_ns=10.0)},
    )
    return pulse.quantum_system(name="virtual_z", qudits={0: qubit})


def _ground():
    return np.array([1.0, 0.0], dtype=complex)


def _bloch(rho):
    return np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real])


@pulse.kernel
def plus_then_gate(q, gate: str, axis: int):
    """|+> (Y90), the phase gate *gate* (or none), then the pulse that brings
    the Bloch axis *axis* (0 X, 1 Y, 2 Z) to Z."""
    d = get_drive_line(q)
    drive(d, d.get_calibration(pulse.gates.Y90))
    if gate != "":
        drive(d, d.get_calibration(gate))
    if axis == 0:
        y90 = d.get_calibration(pulse.gates.Y90)
        drive(d, y90.replace(amplitude=-y90.amplitude))
    elif axis == 1:
        drive(d, d.get_calibration(pulse.gates.X90))


def _logical_bloch(qs, gate, backend="cpu"):
    """<X>, <Y>, <Z> of the logical state after |+> and *gate*, each read as
    <Z> after the rotation that brings that axis to Z."""
    values = []
    for axis in range(3):
        result = pulse.simulate(
            plus_then_gate,
            qs.qudits[0],
            str(gate) if gate else "",
            axis,
            levels=2,
            initial_state=_ground(),
            backend=backend,
        )
        values.append(_bloch(result.density_matrix)[2])
    return np.array(values)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("gate", [Z, S, T], ids=str)
def test_a_phase_gate_turns_plus_about_z_by_its_phase(qs, gate, backend):
    # |+> -> (|0> + e^{i phi}|1>)/sqrt(2): Bloch (cos phi, sin phi, 0); T|+> is |T>.
    phi = PHASES[gate]
    found = _logical_bloch(qs, gate, backend)
    np.testing.assert_allclose(found, [math.cos(phi), math.sin(phi), 0.0], atol=2e-3)


def test_without_a_phase_gate_the_tomography_reads_plus(qs):
    np.testing.assert_allclose(_logical_bloch(qs, None), [1.0, 0.0, 0.0], atol=2e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_simulated_state_is_in_the_frame(qs, backend):
    # A virtual Z after the last pulse leaves the returned state as it was;
    # the logical state is Rz(phase) of it.
    @pulse.kernel
    def plus(q, gate: str):
        d = get_drive_line(q)
        drive(d, d.get_calibration(pulse.gates.Y90))
        if gate != "":
            drive(d, d.get_calibration(gate))

    def state(gate):
        return pulse.simulate(
            plus, qs.qudits[0], gate, levels=2, initial_state=_ground(), backend=backend
        ).density_matrix

    np.testing.assert_allclose(state("t"), state(""), atol=1e-12)
    frame = _bloch(state("t"))
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    logical = np.array([c * frame[0] - s * frame[1], s * frame[0] + c * frame[1], frame[2]])
    np.testing.assert_allclose(logical, _logical_bloch(qs, T), atol=2e-3)


def test_t_and_its_inverse_cancel_and_s_twice_is_z(qs):
    @pulse.kernel
    def products(q, axis: int, which: int):
        d = get_drive_line(q)
        drive(d, d.get_calibration(pulse.gates.Y90))
        if which == 0:
            t = d.get_calibration(pulse.gates.T)
            drive(d, t)
            drive(d, t.replace(phase_rad=-t.phase_rad))
        else:
            drive(d, d.get_calibration(pulse.gates.S))
            drive(d, d.get_calibration(pulse.gates.S))
        if axis == 0:
            y90 = d.get_calibration(pulse.gates.Y90)
            drive(d, y90.replace(amplitude=-y90.amplitude))
        elif axis == 1:
            drive(d, d.get_calibration(pulse.gates.X90))

    def bloch(which):
        return np.array(
            [
                _bloch(
                    pulse.simulate(
                        products, qs.qudits[0], axis, which, levels=2, initial_state=_ground()
                    ).density_matrix
                )[2]
                for axis in range(3)
            ]
        )

    np.testing.assert_allclose(bloch(0), [1.0, 0.0, 0.0], atol=2e-3)
    np.testing.assert_allclose(bloch(1), _logical_bloch(qs, Z), atol=1e-9)


def test_a_phase_gate_takes_no_time_and_plays_no_pulse(qs):
    @pulse.kernel
    def with_t(q):
        d = get_drive_line(q)
        drive(d, d.get_calibration(pulse.gates.X90))
        drive(d, d.get_calibration(pulse.gates.T))
        drive(d, d.get_calibration(pulse.gates.X90))

    @pulse.kernel
    def written(q):
        d = get_drive_line(q)
        drive(d, d.get_calibration(pulse.gates.X90))
        shift_phase(d.tone, math.pi / 4)
        drive(d, d.get_calibration(pulse.gates.X90))

    @pulse.kernel
    def without(q):
        d = get_drive_line(q)
        drive(d, d.get_calibration(pulse.gates.X90))
        drive(d, d.get_calibration(pulse.gates.X90))

    q = qs.qudits[0]
    lowered = with_t.lower(q)
    assert lowered.mlir().replace("with_t", "k") == written.lower(q).mlir().replace("written", "k")
    assert [(e.start, e.end) for e in lowered.schedule] == [
        (e.start, e.end) for e in without.lower(q).schedule
    ]
    drives = [e for e in lowered.export_to_controller()["events"] if e["kind"] == "drive"]
    assert [e["phase_rad"] for e in drives] == [0.0, pytest.approx(math.pi / 4)]


def test_a_phase_gate_of_another_transition_shifts_that_tone(qs):
    dev = pulse.quantum_system("transmon_surface17", layout=[0])

    @pulse.kernel
    def ef_phase(q):
        d = get_drive_line(q)
        ef = d.tones[(1, 2)]
        drive(d, d.get_calibration(pulse.gates.S, transition=(1, 2)), tone=ef)
        drive(d, gaussian(40, 0.1, 10.0), tone=ef)
        drive(d, gaussian(40, 0.1, 10.0))

    events = ef_phase.lower(dev.qudits[0]).export_to_controller()["events"]
    ef_drive, qubit_drive = [e for e in events if e["kind"] == "drive"]
    assert ef_drive["phase_rad"] == pytest.approx(math.pi / 2)
    assert qubit_drive["phase_rad"] == 0.0
