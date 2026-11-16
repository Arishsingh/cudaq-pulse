# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SimulationResult.postselect: the state, branches, records and acceptance
of the branches a test keeps."""

from __future__ import annotations

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import Link
from cudaq_pulse.gates import READOUT, X
from cudaq_pulse.quantum_systems import DragPulse, MicrowaveDrive, PhysicalQudit, ReadoutPulse
from cudaq_pulse.runtime.simulate import Branch, Measurement, SimulationResult

BACKENDS = ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]
THERMAL = 0.1


def _system(thermal_population=0.0):
    # No decoherence and two levels: a branch's state is exact up to the integrator.
    qubit = PhysicalQudit(
        0,
        5.0e9,
        0.0,
        0.0,
        anharmonicity_hz=-2.0e8,
        drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
        thermal_population=thermal_population,
        calibrations={
            X: DragPulse(duration_ns=40.0, sigma_ns=10.0),
            READOUT: ReadoutPulse(duration_ns=100.0),
        },
    )
    return pulse.quantum_system(name="postselect", qudits={0: qubit})


def _ground():
    return np.array([1.0, 0.0], dtype=complex)


@pulse.kernel
def measure_then_x90(q, measure: int):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, d.get_calibration(pulse.gates.Y90))
    sync(d, r)
    if measure == 1:
        readout(r, r.get_calibration(pulse.gates.READOUT))
    else:
        wait(r, 100 * pulse.ns)
    sync(d, r)
    drive(d, d.get_calibration(pulse.gates.X90))


@pulse.kernel
def x90_from_ground(q):
    drive(get_drive_line(q), get_drive_line(q).get_calibration(pulse.gates.X90))


@pulse.classical(min_latency_ns=20, nominal_latency_ns=30, max_latency_ns=40, link="controller")
def herald(outcome: pulse.Bits) -> bool:
    return outcome[0] == 0


@pulse.kernel
def heralded(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, r.get_calibration(pulse.gates.READOUT))
    ok = herald.submit([m]).result()
    sync(d, r)
    if not ok:
        drive(d, d.get_calibration(pulse.gates.X))
    sync(d, r)


def _heralded(qs):
    control_system = pulse.ControlSystem(
        links={"controller": Link(fixed_ns=4, feedback_ns=8, slots=2)}
    )
    return pulse.compile(heralded, quantum_system=qs, control_system=control_system).lower(
        qs.qudits[0]
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_kept_branch_holds_the_state_its_outcome_left(backend):
    qs = _system()
    result = pulse.simulate(
        measure_then_x90, qs.qudits[0], 1, levels=2, initial_state=_ground(), backend=backend
    )
    assert [b.levels for b in result.branches] == [(0,), (1,)]
    kept = result.postselect(lambda b: b.levels[0] == 0)
    # Read 0: the readout left |0>, and X90 played on it.
    expected = pulse.simulate(
        x90_from_ground, qs.qudits[0], levels=2, initial_state=_ground(), backend=backend
    ).density_matrix
    np.testing.assert_allclose(kept.density_matrix, expected, atol=1e-6)
    assert kept.acceptance == pytest.approx(0.5, abs=1e-3)
    assert kept.acceptance == pytest.approx(result.branches[0].probability)
    assert [b.levels for b in kept.branches] == [(0,)]
    assert kept.branches[0].probability == pytest.approx(1.0)
    assert np.trace(kept.density_matrix).real == pytest.approx(1.0)
    assert kept.level_populations() == pytest.approx([0.5, 0.5], abs=1e-3)
    assert (kept.dims, kept.qudits, kept.measurements) == (
        result.dims,
        result.qudits,
        result.measurements,
    )
    assert result.acceptance == 1.0


def test_keeping_every_branch_gives_the_ensemble():
    qs = _system()
    result = pulse.simulate(measure_then_x90, qs.qudits[0], 1, levels=2, initial_state=_ground())
    every = result.postselect(lambda b: True)
    np.testing.assert_allclose(every.density_matrix, result.density_matrix, atol=1e-12)
    assert every.acceptance == pytest.approx(1.0)
    assert [b.probability for b in every.branches] == pytest.approx(
        [b.probability for b in result.branches]
    )


def test_a_program_that_does_not_measure_has_one_branch_to_keep():
    qs = _system()
    result = pulse.simulate(measure_then_x90, qs.qudits[0], 0, levels=2, initial_state=_ground())
    assert result.postselect(lambda b: b.levels == ()).acceptance == pytest.approx(1.0)


def test_a_herald_keeps_the_shots_its_answer_accepts():
    qs = _system(THERMAL)
    result = pulse.simulate(_heralded(qs), levels=2, shots_count=4000, seed=5)
    kept = result.postselect(lambda b: b.answers("herald")[-1])
    # The thermal start reads 1 with probability THERMAL; those shots are dropped.
    assert kept.acceptance == pytest.approx(1 - THERMAL, abs=1e-6)
    assert kept.level_populations() == pytest.approx([1.0, 0.0], abs=1e-9)
    assert result.level_populations()[1] == pytest.approx(0.0, abs=1e-6)
    assert all(b.answers("herald") == [True] for b in kept.branches)
    assert result.records is not None and kept.records is not None
    assert set(kept.records[:, 0].tolist()) == {0}
    assert len(kept.records) == int(np.sum(result.records[:, 0] == 0))
    assert len(kept.records) / len(result.records) == pytest.approx(1 - THERMAL, abs=0.02)


def test_sampled_branches_post_select_by_their_shots():
    qs = _system(THERMAL)
    result = pulse.simulate(_heralded(qs), levels=2, shots_count=500, seed=3, branches="sampled")
    kept = result.postselect(lambda b: b.levels[0] == 0)
    assert result.records is not None and kept.records is not None
    assert kept.acceptance == pytest.approx(len(kept.records) / len(result.records))
    assert kept.level_populations()[0] == pytest.approx(1.0)


def test_post_selection_composes():
    qs = _system()
    result = pulse.simulate(measure_then_x90, qs.qudits[0], 1, levels=2, initial_state=_ground())
    once = result.postselect(lambda b: b.levels[0] == 1)
    twice = once.postselect(lambda b: True)
    np.testing.assert_allclose(twice.density_matrix, once.density_matrix, atol=1e-12)
    assert twice.acceptance == pytest.approx(once.acceptance)
    with pytest.raises(pulse.InvalidValueError, match="keeps none of the 1 branches"):
        once.postselect(lambda b: b.levels[0] == 0)


def _result(branches, states, records=None):
    return SimulationResult(
        density_matrix=sum(states),
        dims=(2,),
        qudits=(0,),
        measurements=[Measurement(0, (), 0, 0.0, 50.0)],
        branches=branches,
        records=records,
        _states=states,
    )


def test_shots_of_levels_both_kept_and_not_have_no_records():
    zero, one = np.diag([1.0, 0.0]).astype(complex), np.diag([0.0, 1.0]).astype(complex)
    records = np.array([[0], [0], [1]], dtype=np.int64)
    # Two branches read 0 (as two IQ points would) and only the first is kept.
    branches = [Branch((0,), 0.4), Branch((0,), 0.3), Branch((1,), 0.3)]
    result = _result(branches, [zero * 0.4, zero * 0.3, one * 0.3], records)
    assert result.postselect(lambda b: b.probability == 0.4).records is None
    kept = result.postselect(lambda b: b.levels == (0,))
    assert kept.records is not None and kept.records.tolist() == [[0], [0]]
    assert kept.acceptance == pytest.approx(0.7)


def test_a_result_without_branch_states_does_not_post_select():
    merged = _result([Branch((0,), 1.0)], [np.eye(2, dtype=complex)])
    merged._states = []
    with pytest.raises(pulse.UnsupportedError, match="no state per branch"):
        merged.postselect(lambda b: True)
