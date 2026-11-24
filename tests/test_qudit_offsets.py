# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A qudit without a frequency: every frequency of its system is an offset."""

from __future__ import annotations

import copy
import json
import math
import pickle

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse._errors import QuantumSystemDefinitionError
from cudaq_pulse.quantum_systems import (
    Coupling,
    MicrowaveDrive,
    Offset,
    PhysicalQudit,
    ResidualZZ,
    TransmonModel,
)

DRIVE = MicrowaveDrive(amplitude_scale_rad_per_ns=0.05)
OFFSET_HZ = 2.0e6


def _system(frequency_hz, count=1, **more):
    qudits = {i: PhysicalQudit(i, frequency_hz, 0.0, 0.0, drive=DRIVE) for i in range(count)}
    model = TransmonModel(levels={i: 2 for i in range(count)})
    return pulse.quantum_system(name="frame", qudits=qudits, model=model, clock_ghz=1.0, **more)


@pulse.kernel
def detuned_then_resonant(q, offset_hz: float):
    line = get_drive_line(q)  # noqa: F821
    shift_frequency(line.tone, offset_hz)  # noqa: F821
    drive(line, square(40 * pulse.ns, 0.5))  # noqa: F821
    shift_frequency(line.tone, -offset_hz)  # noqa: F821
    drive(line, square(24 * pulse.ns, 0.5))  # noqa: F821


def test_a_qudit_may_leave_its_frequency_out():
    qudit = PhysicalQudit(0, None, 50.0, 30.0)
    assert qudit.frequency_hz is None
    assert "frequency_hz=None" in repr(qudit)
    assert pickle.loads(pickle.dumps(qudit)) == qudit == copy.deepcopy(qudit)
    assert qudit != PhysicalQudit(0, 5.0e9, 50.0, 30.0)
    assert qudit.replace(frequency_hz=5.0e9).frequency_hz == 5.0e9
    # A qudit with a frequency computes with it as before.
    assert PhysicalQudit(0, 5.0e9, 50.0, 30.0)._frame_hz == 5.0e9


def test_a_system_of_such_qudits_reads_them_back_as_given():
    given = PhysicalQudit(0, None, 50.0, 30.0, label="a")
    qs = pulse.quantum_system(name="offsets", qudits={0: given}, model=TransmonModel(levels={0: 2}))
    held = qs.physical_qudits[0]
    assert held.frequency_hz is None
    assert held == given and given == held and hash(held) == hash(given)
    assert repr(held) == repr(given) and type(held.replace()) is PhysicalQudit
    assert held.replace(t1_us=10.0) == given.replace(t1_us=10.0)
    assert held != given.replace(frequency_hz=5.0e9)
    for back in (pickle.loads(pickle.dumps(held)), copy.deepcopy(held)):
        assert back == given and type(back) is PhysicalQudit and back.frequency_hz is None
    # Given a frequency, the held qudit is an ordinary one.
    with_one = held.replace(frequency_hz=5.0e9)
    assert type(with_one) is PhysicalQudit and with_one == given.replace(frequency_hz=5.0e9)
    assert with_one._frame_hz == 5.0e9
    # In its own frame the qudit turns at 0 Hz, and so do its tones.
    assert qs.qudit_freq[0] == 0.0 and qs.transition_freq[0, 0, 1] == 0.0
    again = qs.replace(qudits=dict(qs.physical_qudits))
    assert again.fingerprint == qs.fingerprint and again == qs
    assert pickle.loads(pickle.dumps(qs)) == qs


def test_a_system_file_holds_the_missing_frequency_as_null():
    qs = _system(None, count=2)
    data = json.loads(json.dumps(qs.as_dict()))
    assert [q["frequency_hz"] for q in data["qudits"]] == [None, None]
    back = pulse.QuantumSystem.from_dict(data)
    assert back.fingerprint == qs.fingerprint
    assert back.physical_qudits[1].frequency_hz is None
    assert _system(5.0e9).fingerprint != _system(None).fingerprint


def test_zero_is_still_not_a_frequency():
    with pytest.raises(QuantumSystemDefinitionError) as found:
        _system(0.0)
    message = str(found.value)
    assert "qudits[0].frequency_hz: 0.0 is not a positive, finite number of Hz" in message
    assert "Offset(hz) from a reference its qudits share, or None (null in a file)" in message
    for wrong in (-5.0e9, math.nan, math.inf, "5e9", True):
        with pytest.raises(QuantumSystemDefinitionError, match="frequency_hz"):
            _system(wrong)


def test_either_every_qudit_gives_its_frequency_or_none_does():
    qudits = {
        0: PhysicalQudit(0, None, 0.0, 0.0),
        1: PhysicalQudit(1, 5.0e9, 0.0, 0.0),
        2: PhysicalQudit(2, None, 0.0, 0.0),
    }
    with pytest.raises(QuantumSystemDefinitionError) as found:
        pulse.quantum_system(name="mixed", qudits=qudits)
    message = str(found.value)
    assert (
        "qudits[1].frequency_hz: the qudits give their frequencies in more than one way: "
        "as frequencies (qudits 1); not at all (None) (qudits 0, 2)"
    ) in message
    assert "or None to all of them" in message
    qudits[0] = PhysicalQudit(0, Offset(0.0), 0.0, 0.0)
    with pytest.raises(QuantumSystemDefinitionError, match="in more than one way") as found:
        pulse.quantum_system(name="mixed", qudits=qudits)
    assert "as offsets from a shared reference (qudits 0)" in str(found.value)
    # Another problem is reported as before, and alone decides.
    qudits = {0: PhysicalQudit(0, None, -1.0, 0.0)}
    with pytest.raises(QuantumSystemDefinitionError, match=r"qudits\[0\]\.t1_us"):
        pulse.quantum_system(name="bad", qudits=qudits)


def test_the_frequencies_of_its_program_are_offsets():
    qs = _system(None)
    program = (
        pulse.compile(detuned_then_resonant, quantum_system=qs)
        .lower(qs.qudits[0], OFFSET_HZ)
        .export_to_controller()
    )
    first, second = (e for e in program["events"] if e["kind"] == "drive")
    assert first["frequency_hz"] == OFFSET_HZ and second["frequency_hz"] == 0.0
    assert first["phase_rad"] == 0.0
    assert second["phase_rad"] == pytest.approx(2 * math.pi * OFFSET_HZ * 40e-9)
    # The same program on a qudit with a frequency differs by that frequency alone.
    absolute = _system(5.0e9)
    other = (
        pulse.compile(detuned_then_resonant, quantum_system=absolute)
        .lower(absolute.qudits[0], OFFSET_HZ)
        .export_to_controller()
    )
    drives = [e for e in other["events"] if e["kind"] == "drive"]
    assert [e["frequency_hz"] for e in drives] == [5.0e9 + OFFSET_HZ, 5.0e9]
    assert [e["phase_rad"] for e in drives] == [first["phase_rad"], second["phase_rad"]]
    assert [e["start_cycles"] for e in drives] == [first["start_cycles"], second["start_cycles"]]


def test_it_simulates_as_the_same_qudit_with_a_frequency():
    populations = []
    for frequency_hz in (None, 5.0e9):
        qs = _system(frequency_hz)
        result = pulse.simulate(detuned_then_resonant, qs.qudits[0], OFFSET_HZ, quantum_system=qs)
        populations.append(result.level_population(1))
    assert 0.05 < populations[0] < 0.95
    assert populations[0] == pytest.approx(populations[1], abs=1e-9)


def test_two_such_qudits_interact():
    @pulse.kernel
    def both(a, b):
        first, second = get_drive_line(a), get_drive_line(b)  # noqa: F821
        drive(first, square(40 * pulse.ns, 0.5))  # noqa: F821
        drive(second, square(40 * pulse.ns, 0.5))  # noqa: F821

    diagonals = []
    for frequency_hz in (None, 5.0e9):
        qs = _system(frequency_hz, count=2, residual_zz=[ResidualZZ((0, 1), rate_hz=3.0e6)])
        result = pulse.simulate(both, qs.qudits[0], qs.qudits[1], quantum_system=qs)
        diagonals.append(np.real(np.diag(result.density_matrix)))
    assert np.allclose(diagonals[0], diagonals[1], atol=1e-9)
    assert diagonals[0].max() < 0.9


def test_both_export_formats_and_a_kernel_of_any_qudit_give_offsets():
    qs = _system(None, count=2)

    @pulse.kernel
    def any_qudit(q: pulse.Qudit, offset_hz: float):
        line = get_drive_line(q)  # noqa: F821
        shift_frequency(line.tone, offset_hz)  # noqa: F821
        drive(line, square(40 * pulse.ns, 0.5))  # noqa: F821

    scheduled = pulse.compile(any_qudit, quantum_system=qs).lower(qs.qudits[1], OFFSET_HZ)
    flat = scheduled.export_to_controller()
    structured = scheduled.export_to_controller(version=2)
    assert [e["frequency_hz"] for e in flat["events"] if e["kind"] == "drive"] == [OFFSET_HZ]
    assert structured["events"][0]["tones"][0]["frequency_hz"] == OFFSET_HZ
    natives = [s["value"] for s in flat["symbols"] if s["kind"] == "frequency"]
    assert natives and set(natives) == {0.0}


def test_parameter_drift_needs_frequencies():
    with pytest.raises(pulse.UnsupportedError, match="no frequency to drift from"):
        _system(None).replace(parameter_drift=True, parameter_drift_seed=0)
    assert _system(5.0e9).replace(parameter_drift=True, parameter_drift_seed=0) is not None


def test_a_coupling_needs_the_frequencies_of_its_qudits():
    with pytest.raises(QuantumSystemDefinitionError) as found:
        _system(None, count=2, couplings=[Coupling((0, 1), strength_hz=5.0e6)])
    message = str(found.value)
    assert "couplings: the qudits give no frequency_hz, and a coupling ((0, 1)) acts by" in message
    assert "residual_zz=[ResidualZZ((a, b), rate_hz=...)]" in message
    # With frequencies the same system builds, and a stated ZZ needs none.
    assert _system(5.0e9, count=2, couplings=[Coupling((0, 1), strength_hz=5.0e6)]) is not None
    assert _system(None, count=2, residual_zz=[ResidualZZ((0, 1), rate_hz=3.0e6)]) is not None
    # Another problem of the entries is reported with it.
    with pytest.raises(QuantumSystemDefinitionError, match=r"2 problems") as found:
        _system(
            None,
            count=2,
            couplings=[Coupling((0, 1), strength_hz=5.0e6)],
            residual_zz=[ResidualZZ((0, 7), rate_hz=3.0e6)],
        )
    assert "residual_zz[0]" in str(found.value) and "couplings:" in str(found.value)


def test_the_checks_report_such_qudits_first_and_alone():
    from cudaq_pulse.quantum_systems import base

    without = {i: PhysicalQudit(i, None, 0.0, 0.0) for i in range(2)}
    found = base._qudits_problems(without)
    assert [type(item) for item in found] == [base._HeldQudits] and found[0].own_carrier
    assert all(type(q) is base._OwnCarrierQudit for q in found[0].qudits.values())
    # A list like any other: more problems follow it, and it is not a problem.
    assert (found + ["more"])[1:] == ["more"] and bool(found)
    # With a problem of the qudits, or with frequencies, there is none.
    wrong = {**without, 1: PhysicalQudit(1, None, -1.0, 0.0)}
    assert not any(type(item) is base._HeldQudits for item in base._qudits_problems(wrong))
    shared = base._qudits_problems({0: PhysicalQudit(0, Offset(0.0), 0.0, 0.0)})
    assert [type(item) for item in shared] == [base._HeldQudits] and not shared[0].own_carrier
    assert base._qudits_problems({0: PhysicalQudit(0, 5.0e9, 0.0, 0.0)}) == []


def test_a_drive_at_another_qudits_tone_needs_their_frequencies():
    @pulse.kernel
    def cross(a, b):
        d = get_drive_line(a)  # noqa: F821
        e = get_drive_line(b)  # noqa: F821
        drive(d, square(40 * pulse.ns, 0.5), tone=e.tone)  # noqa: F821

    @pulse.kernel
    def own_tones(a, b):
        d = get_drive_line(a)  # noqa: F821
        drive(d, square(40 * pulse.ns, 0.5), tone=d.tone)  # noqa: F821
        drive(d, square(40 * pulse.ns, 0.5), tone=d.tones[(1, 2)])  # noqa: F821

    def pair(frequency_hz):
        qudits = {
            i: PhysicalQudit(i, frequency_hz, 0.0, 0.0, anharmonicity_hz=-2.0e8, drive=DRIVE)
            for i in range(2)
        }
        return pulse.quantum_system(name="pair", qudits=qudits, clock_ghz=1.0)

    without = pair(None)
    with pytest.raises(pulse.CompilationError) as found:
        pulse.compile(cross, quantum_system=without).lower(*without.qudits[:2])
    message = str(found.value)
    assert "drive(): tone= is the tone of qudit 1, and the line is qudit 0's" in message
    assert "give no frequency_hz" in message
    # The line's own tones, of any transition, are offsets of its own qudit.
    program = (
        pulse.compile(own_tones, quantum_system=without)
        .lower(*without.qudits[:2])
        .export_to_controller()
    )
    assert [e["frequency_hz"] for e in program["events"] if e["kind"] == "drive"] == [0.0, -2.0e8]
    # With frequencies the same kernel drives qudit 0's line at qudit 1's frequency.
    absolute = pair(5.0e9)
    assert pulse.compile(cross, quantum_system=absolute).lower(*absolute.qudits[:2]) is not None


# ── Offsets from one shared reference ───────────────────────────────────────


def test_an_offset_is_a_float_that_keeps_its_kind():
    offset = Offset(1.5e8)
    assert isinstance(offset, float) and repr(offset) == "Offset(150000000.0)"
    # Arithmetic gives plain floats.
    for value in (offset + 1.0, 2 * offset, offset - offset, -offset, abs(offset), offset / 2):
        assert type(value) is float
    assert offset + 1.0 == 150000001.0 and float(offset) == 1.5e8
    # It equals only an Offset of its value, and hashes apart from the float.
    assert offset == Offset(1.5e8) and offset != Offset(1.0) and offset != 1.5e8
    assert 1.5e8 != offset and not (offset == 1.5e8) and hash(offset) != hash(1.5e8)
    assert len({offset, Offset(1.5e8), 1.5e8}) == 2
    assert Offset(0.0) == Offset(0.0) and Offset(0.0) != 0.0
    for back in (pickle.loads(pickle.dumps(offset)), copy.deepcopy(offset)):
        assert type(back) is Offset and back == offset
    qudit = PhysicalQudit(1, offset, 50.0, 30.0)
    assert qudit != PhysicalQudit(1, 1.5e8, 50.0, 30.0) and qudit == PhysicalQudit(
        1, Offset(1.5e8), 50.0, 30.0
    )
    assert "frequency_hz=Offset(150000000.0)" in repr(qudit)
    assert pickle.loads(pickle.dumps(qudit)) == qudit


def _referenced(offsets, **more):
    qudits = {
        i: PhysicalQudit(i, Offset(f), 0.0, 0.0, anharmonicity_hz=-2.0e8, drive=DRIVE)
        for i, f in enumerate(offsets)
    }
    return pulse.quantum_system(name="pair", qudits=qudits, clock_ghz=1.0, **more)


def _absolute(frequencies, **more):
    qudits = {
        i: PhysicalQudit(i, f, 0.0, 0.0, anharmonicity_hz=-2.0e8, drive=DRIVE)
        for i, f in enumerate(frequencies)
    }
    return pulse.quantum_system(name="pair", qudits=qudits, clock_ghz=1.0, **more)


def test_a_system_says_what_its_frequencies_are_measured_from():
    assert _absolute([5.0e9, 5.15e9]).frequency_frame == "absolute"
    assert _referenced([0.0, 1.5e8]).frequency_frame == "shared_reference"
    assert _system(None, count=2).frequency_frame == "own_carrier"
    assert pulse.quantum_system("transmon_surface17").frequency_frame == "absolute"


def test_a_system_of_offsets_reads_them_back_as_given():
    qs = _referenced([0.0, -1.5e8])
    held = qs.physical_qudits[1]
    given = PhysicalQudit(1, Offset(-1.5e8), 0.0, 0.0, anharmonicity_hz=-2.0e8, drive=DRIVE)
    assert type(held.frequency_hz) is Offset and held.frequency_hz == Offset(-1.5e8)
    assert held == given and given == held and hash(held) == hash(given)
    assert repr(held) == repr(given) and type(held.replace()) is PhysicalQudit
    for back in (pickle.loads(pickle.dumps(held)), copy.deepcopy(held)):
        assert back == given and type(back) is PhysicalQudit
    assert held.replace(frequency_hz=5.0e9).frequency_hz == 5.0e9
    assert qs.qudit_freq.tolist() == [0.0, -1.5e8]
    assert pickle.loads(pickle.dumps(qs)) == qs
    again = qs.replace(qudits=dict(qs.physical_qudits))
    assert again.fingerprint == qs.fingerprint
    assert _absolute([5.0e9, 5.0e9 - 1.5e8]).fingerprint != qs.fingerprint


def test_a_system_file_holds_an_offset_as_a_mapping():
    qs = _referenced([0.0, 1.5e8])
    data = json.loads(json.dumps(qs.as_dict()))
    assert [q["frequency_hz"] for q in data["qudits"]] == [
        {"offset_hz": 0.0},
        {"offset_hz": 150000000.0},
    ]
    back = pulse.QuantumSystem.from_dict(data)
    assert back.fingerprint == qs.fingerprint and back.frequency_frame == "shared_reference"
    assert back.physical_qudits[1].frequency_hz == Offset(1.5e8)
    data["qudits"][1]["frequency_hz"] = {"offset": 1.5e8}
    with pytest.raises(QuantumSystemDefinitionError, match=r"qudits\[1\]\.frequency_hz"):
        pulse.QuantumSystem.from_dict(data)


@pulse.kernel
def coupled_and_cross(a, b):
    d = get_drive_line(a)  # noqa: F821
    e = get_drive_line(b)  # noqa: F821
    drive(d, square(63, 1.0))  # noqa: F821
    drive(e, square(40, 0.5), tone=d.tone)  # noqa: F821
    drive(d, square(40, 0.5), tone=d.tones[(1, 2)])  # noqa: F821
    sync(d, e)  # noqa: F821
    wait(e, 100)  # noqa: F821


@pytest.mark.parametrize("offsets", [(0.0, 1.5e8), (-1.5e8, 0.0), (2.0e7, 0.0)])
def test_offsets_compute_as_the_same_system_with_frequencies(offsets):
    reference = 5.0e9
    coupling = [Coupling((0, 1), strength_hz=5.0e6)]
    zz = [ResidualZZ((0, 1), rate_hz=3.0e5)]
    shared = _referenced(offsets, couplings=coupling, residual_zz=zz)
    absolute = _absolute([reference + f for f in offsets], couplings=coupling, residual_zz=zz)
    programs = [
        pulse.compile(coupled_and_cross, quantum_system=qs).lower(*qs.qudits[:2])
        for qs in (shared, absolute)
    ]
    ours, theirs = (program.export_to_controller() for program in programs)
    assert [line["native_frequency_hz"] for line in ours["lines"]] == list(offsets)
    assert len(ours["events"]) == len(theirs["events"])
    for mine, other in zip(ours["events"], theirs["events"], strict=True):
        assert mine["start_cycles"] == other["start_cycles"]
        if mine["kind"] == "drive":
            assert mine["frequency_hz"] == pytest.approx(
                other["frequency_hz"] - reference, abs=1e-3
            )
            assert mine["phase_rad"] == pytest.approx(other["phase_rad"], abs=1e-9)
    # The line's own tone, the other qudit's, and its own 1-2 tone.
    drives = [e["frequency_hz"] for e in ours["events"] if e["kind"] == "drive"]
    assert drives == pytest.approx([offsets[0], offsets[0], offsets[0] - 2.0e8])
    populations = [np.real(np.diag(pulse.simulate(program).density_matrix)) for program in programs]
    assert np.allclose(populations[0], populations[1], atol=1e-6)
    assert populations[0].max() < 0.999


def test_parameter_drift_needs_absolute_frequencies():
    with pytest.raises(pulse.UnsupportedError, match="no frequency to drift from"):
        _referenced([0.0, 1.5e8]).replace(parameter_drift=True, parameter_drift_seed=0)


def test_every_shipped_systems_transitions_are_what_they_were():
    # A transition's frequency is the upper level's energy less the lower's,
    # no longer its absolute value: the same number wherever levels ascend.
    from cudaq_pulse.quantum_systems import SimulationModel, transmon_lattice, transmon_surface17

    systems = [transmon_surface17(), transmon_lattice(6)]
    checked = 0
    for system in systems:
        for model in (system.model, SimulationModel()):
            for index, qudit in system.physical_qudits.items():
                levels = model.levels_of(index)
                energies = model.energies_hz(qudit, levels)
                found = model.transitions_hz(qudit, levels)
                assert found == {
                    (lower, upper): abs(energies[upper] - energies[lower])
                    for lower in range(levels)
                    for upper in range(lower + 1, levels)
                }
                assert min(found.values()) > 0
                checked += len(found)
    assert checked == 92
