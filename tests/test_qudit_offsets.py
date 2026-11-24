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
from cudaq_pulse._errors import InvalidValueError, NotFoundError, QuantumSystemDefinitionError
from cudaq_pulse.controller import DriveLine, ReadoutLine
from cudaq_pulse.quantum_systems import (
    Coupling,
    DispersiveReadout,
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

    def diagonal(frequency_hz, **more):
        qs = _system(frequency_hz, count=2, **more)
        result = pulse.simulate(both, qs.qudits[0], qs.qudits[1], quantum_system=qs)
        return np.real(np.diag(result.density_matrix))

    zz = [ResidualZZ((0, 1), rate_hz=3.0e6)]
    without, absolute = diagonal(None, residual_zz=zz), diagonal(5.0e9, residual_zz=zz)
    assert np.allclose(without, absolute, atol=1e-9)
    assert without.max() < 0.9
    # The interaction is simulated: without it the populations differ.
    alone = diagonal(None)
    assert np.abs(without - alone).max() > 1e-3
    assert np.allclose(alone, diagonal(5.0e9), atol=1e-9)


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


def test_the_checks_tell_such_qudits_from_a_mistake_where_problems_are_read():
    from cudaq_pulse.quantum_systems import base

    # The checks of the qudits report a frequency of another form as they
    # report any value that is not a frequency in Hz.
    without = {i: PhysicalQudit(i, None, 0.0, 0.0) for i in range(2)}
    found = base._qudits_problems(without)
    assert [p.path for p in found] == ["qudits[0].frequency_hz", "qudits[1].frequency_hz"]
    shared = {0: PhysicalQudit(0, Offset(1.5e8), 0.0, 0.0)}
    assert [p.path for p in base._qudits_problems(shared)] == ["qudits[0].frequency_hz"]
    # The description, which reads the problems, holds such qudits instead.
    described = base._Description("none", without)
    assert type(described) is base._OwnCarrierDescription and described._own_carrier
    assert all(type(q) is base._OwnCarrierQudit for q in described.qudits.values())
    described = base._Description("offsets", shared)
    assert type(described) is base._SharedReferenceDescription and not described._own_carrier
    assert type(described.qudits[0]) is base._HeldQudit
    assert type(described.replace(name="again")) is base._SharedReferenceDescription
    # Beside a problem of a qudit, the qudit's form is not one more problem.
    wrong = {**without, 1: PhysicalQudit(1, None, -1.0, 0.0)}
    with pytest.raises(QuantumSystemDefinitionError) as caught:
        base._Description("wrong", wrong)
    assert [p.path for p in caught.value.problems] == ["qudits[1].t1_us"]
    # A frequency that is no number of either form stays a problem.
    for value in (Offset(math.nan), Offset(math.inf), {"offset": 1.0}, "5e9"):
        with pytest.raises(QuantumSystemDefinitionError) as caught:
            base._Description("wrong", {0: PhysicalQudit(0, value, 0.0, 0.0)})
        assert [p.path for p in caught.value.problems] == ["qudits[0].frequency_hz"]


@pytest.mark.parametrize("frequency_hz", [5.0e9, 5_000_000_000, np.float64(5.0e9)])
def test_a_system_with_frequencies_is_held_as_it_was(frequency_hz):
    # A frequency that is an int or a NumPy float is a frequency: its system
    # reports nothing, is not held as the other forms are and writes its
    # file as it did.
    from cudaq_pulse.quantum_systems import base

    given = {i: PhysicalQudit(i, frequency_hz, 0.0, 0.0, anharmonicity_hz=-2.0e8) for i in range(2)}
    assert base._qudits_problems(given) == []
    described = base._Description("plain", given)
    assert type(described) is base._Description and described._frame == "absolute"
    assert all(described.qudits[i] is given[i] for i in given)
    assert base._finite(frequency_hz)
    qs = pulse.quantum_system(name="plain", qudits=given)
    assert qs.frequency_frame == "absolute"
    written = [q["frequency_hz"] for q in qs.as_dict()["qudits"]]
    assert all(type(value) is type(frequency_hz) and value == frequency_hz for value in written)
    assert qs.qudit_freq.tolist() == [5.0e9, 5.0e9]


def test_an_offset_is_a_frequency_and_no_other_value():
    # Given where a time, a rate or another frequency belongs, an Offset is a
    # problem at its path like any other wrong value, not an error of its own.
    from cudaq_pulse.quantum_systems import FluxCZ, base

    assert not base._finite(Offset(1.0)) and base._finite(1.0) and base._finite(np.float64(1.0))
    assert not base._finite(True) and base._finite(5) and not base._finite(None)

    def problems(**system):
        with pytest.raises(QuantumSystemDefinitionError) as caught:
            pulse.quantum_system(name="wrong", **system)
        return [p.path for p in caught.value.problems]

    qudits = {i: PhysicalQudit(i, 5.0e9 + i, 0.0, 0.0) for i in range(2)}
    assert problems(qudits={0: PhysicalQudit(0, 5.0e9, Offset(50.0), 30.0)}) == ["qudits[0].t1_us"]
    assert problems(qudits=qudits, couplings=[Coupling((0, 1), strength_hz=Offset(5.0e6))]) == [
        "couplings[0].strength_hz"
    ]
    assert problems(qudits=qudits, residual_zz=[ResidualZZ((0, 1), rate_hz=Offset(1.0))]) == [
        "residual_zz[0].rate_hz"
    ]
    cz = {pulse.gates.CZ: FluxCZ(duration_ns=40.0, interaction_frequency_hz=Offset(-5.0e7))}
    shared = {i: PhysicalQudit(i, Offset(1.0e8 * i), 0.0, 0.0) for i in range(2)}
    assert problems(
        qudits=shared, couplings=[Coupling((0, 1), strength_hz=5.0e6, calibrations=cz)]
    ) == ["couplings[0].calibrations[CZ].interaction_frequency_hz"]


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
    assert "drive(): tone= is the tone of qudit 1, and the line is that of qudit 0;" in message
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
    assert not (offset != Offset(1.5e8)) and not (1.5e8 == offset)
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
    checked = expected = deepest = 0
    for system in systems:
        for model in (system.model, SimulationModel()):
            for index, qudit in system.physical_qudits.items():
                levels = model.levels_of(index)
                expected += levels * (levels - 1) // 2
                deepest = max(deepest, levels)
                energies = model.energies_hz(qudit, levels)
                found = model.transitions_hz(qudit, levels)
                assert found == {
                    (lower, upper): abs(energies[upper] - energies[lower])
                    for lower in range(levels)
                    for upper in range(lower + 1, levels)
                }
                assert min(found.values()) > 0
                checked += len(found)
    # Every transition of every qudit, a 1-2 transition among them.
    assert checked == expected >= 2 * (17 + 6) and deepest >= 3


# ── What a comparison of an Offset says ─────────────────────────────────────


def test_an_offset_compares_as_one_kind_of_number():
    offset, value = Offset(1.5e8), 1.5e8
    # With a plain number: not equal either way round, and not ordered.
    for number in (value, 150_000_000, np.float32(value), True):
        assert offset != number and not (offset == number)
        assert Offset.__eq__(offset, number) is False and Offset.__ne__(offset, number) is True
        for compare in (
            lambda a, b: a < b,
            lambda a, b: a <= b,
            lambda a, b: a > b,
            lambda a, b: a >= b,
        ):
            with pytest.raises(TypeError, match="an offset from a shared reference"):
                compare(offset, number)
    for compare in (lambda: value < offset, lambda: value >= offset, lambda: max(offset, 1.0)):
        with pytest.raises(TypeError, match="compare float\\(offset\\), or another Offset"):
            compare()
    with pytest.raises(pulse.InvalidTypeError):
        sorted([offset, 1.0, Offset(0.0)])
    # With another Offset: by value, and consistently.
    low, high = Offset(-1.0), Offset(2.0)
    assert low < high and low <= high and high > low and high >= low and low != high
    assert not (low > high) and not (high <= low) and Offset(2.0) <= high and Offset(2.0) >= high
    assert max(low, high) is high and sorted([high, low, Offset(0.0)]) == [low, Offset(0.0), high]
    # An object that is not a number decides for itself, the same either way round.
    assert Offset.__eq__(offset, "text") is NotImplemented
    assert Offset.__ne__(offset, "text") is NotImplemented
    assert Offset.__lt__(offset, "text") is NotImplemented
    assert offset == pytest.approx(value) and pytest.approx(value) == offset
    assert offset != pytest.approx(1.0) and pytest.approx(1.0) != offset
    assert offset != "text" and not (offset == None)  # noqa: E711
    with pytest.raises(TypeError):
        offset < "text"  # noqa: B015
    # Arithmetic and float() are untouched, with NumPy's numbers and arrays too.
    assert float(offset) == value and offset - value == 0.0 and abs(Offset(-2.0)) == 2.0
    assert (np.array([1.0, 2.0]) + Offset(1.0)).tolist() == [2.0, 3.0]
    assert np.float32(2.0) * Offset(1.5) == 3.0 and Offset(1.5) * np.float64(2.0) == 3.0
    # A NumPy float on the left is asked first and compares by value: the one
    # comparison Offset does not decide.
    assert Offset.__eq__(offset, np.float64(value)) is False
    assert bool(np.float64(value) == offset)


# ── Plain numbers leave the system ──────────────────────────────────────────


def test_a_system_of_offsets_gives_plain_numbers_but_for_the_qudits_own_field():
    qs = _referenced([0.0, 1.5e8])
    assert type(qs.physical_qudits[1].frequency_hz) is Offset
    assert qs.frequencies == {0: 0.0, 1: 1.5e8}
    assert all(type(value) is float for value in qs.frequencies.values())
    assert type(qs.physical_qudits[1]._frame_hz) is float

    @pulse.kernel
    def any_qudit(q: pulse.Qudit):
        line = get_drive_line(q)  # noqa: F821
        drive(line, square(40, 0.5))  # noqa: F821

    out = pulse.compile(any_qudit, quantum_system=qs).lower(qs.qudits[1]).export_to_controller()
    natives = [s["value"] for s in out["symbols"] if s["kind"] == "frequency"]
    assert natives == [1.5e8] and all(type(value) is float for value in natives)
    assert out == json.loads(json.dumps(out))
    structured = (
        pulse.compile(any_qudit, quantum_system=qs)
        .lower(qs.qudits[1])
        .export_to_controller(version=2)
    )
    assert structured == json.loads(json.dumps(structured))


# ── Simulating on another system ────────────────────────────────────────────


@pulse.kernel
def pi_pulse(q):
    line = get_drive_line(q)  # noqa: F821
    drive(line, square(63 * pulse.ns, 1.0))  # noqa: F821


_FRAME_OF = {None: "own_carrier", 5.0e9: "absolute"}


@pytest.mark.parametrize(
    ("compiled", "simulated"),
    [
        (None, 5.0e9),
        (5.0e9, None),
        (Offset(1.5e8), 5.0e9),
        (5.0e9, Offset(1.5e8)),
        (None, Offset(1.5e8)),
        (Offset(1.5e8), None),
    ],
)
def test_simulate_refuses_a_system_whose_frequencies_are_measured_from_something_else(
    compiled, simulated
):
    ours, theirs = _system(compiled), _system(simulated)
    program = pulse.compile(pi_pulse, quantum_system=ours).lower(ours.qudits[0])
    assert pulse.simulate(program).level_population(1) > 0.99
    with pytest.raises(InvalidValueError) as caught:
        pulse.simulate(program, quantum_system=theirs)
    message = str(caught.value)
    assert ours.frequency_frame != theirs.frequency_frame
    mine, other = repr(ours.frequency_frame), repr(theirs.frequency_frame)
    assert f"compiled for quantum system 'frame', whose frequency_frame is {mine} " in message
    assert f"quantum_system= is 'frame', whose frequency_frame is {other} " in message
    # From the kernel too, and for a drifting system of the other frame.
    with pytest.raises(InvalidValueError, match="frequency_frame"):
        pulse.simulate(pi_pulse, theirs.qudits[0], quantum_system=ours)


def test_simulate_takes_another_system_of_the_same_frame_as_before():
    for compiled, simulated, resonant in (
        (5.0e9, 5.0e9, True),
        (5.0e9, 5.1e9, False),
        (None, None, True),
        (Offset(1.5e8), Offset(1.5e8), True),
        (Offset(1.5e8), Offset(0.0), False),
    ):
        ours, theirs = _system(compiled), _system(simulated)
        program = pulse.compile(pi_pulse, quantum_system=ours).lower(ours.qudits[0])
        excited = pulse.simulate(program, quantum_system=theirs).level_population(1)
        assert (excited > 0.99) is resonant, (compiled, simulated, excited)
    # A system that lacks a qudit of the program is still told so first.
    two = _system(5.0e9, count=2)
    program = pulse.compile(pi_pulse, quantum_system=two).lower(two.qudits[1])
    for other in (_system(5.0e9), _system(None)):
        with pytest.raises(InvalidValueError, match=r"does not define qudits \[1\]"):
            pulse.simulate(program, quantum_system=other)
    with pytest.raises(pulse.InvalidTypeError, match="must be a QuantumSystem"):
        pulse.simulate(program, quantum_system="frame")


def test_a_coupled_system_cannot_be_reached_through_simulate():
    # The couplings an own-carrier system may not declare are not let in by
    # simulating its program on a system that has them.
    ours = _system(None, count=2)
    coupled = _system(5.0e9, count=2, couplings=[Coupling((0, 1), strength_hz=5.0e6)])
    program = pulse.compile(pi_pulse, quantum_system=ours).lower(ours.qudits[0])
    with pytest.raises(InvalidValueError, match="'own_carrier'.*'absolute'"):
        pulse.simulate(program, quantum_system=coupled)


# ── A readout's frequency ───────────────────────────────────────────────────


@pytest.mark.parametrize("frequency_hz", [None, Offset(1.5e8)])
def test_a_readout_frequency_needs_qudits_with_frequencies(frequency_hz):
    def system(readout):
        qudits = {
            i: PhysicalQudit(i, frequency_hz, 0.0, 0.0, drive=DRIVE, readout=readout)
            for i in range(2)
        }
        return pulse.quantum_system(name="read", qudits=qudits, clock_ghz=1.0)

    with pytest.raises(QuantumSystemDefinitionError) as caught:
        system(DispersiveReadout(frequency_hz=7.0e9))
    assert [p.path for p in caught.value.problems] == [
        "qudits[0].readout.frequency_hz",
        "qudits[1].readout.frequency_hz",
    ]
    message = str(caught.value)
    assert "7000000000.0 is a readout frequency, and the qudits give" in message
    assert "leave the readout's frequency_hz out" in message
    # Without one, the readout line plays at its qudit's own tone.
    qs = system(DispersiveReadout())

    @pulse.kernel
    def read(q):
        readout(get_readout_line(q), square(200, 0.05))  # noqa: F821

    out = pulse.compile(read, quantum_system=qs).lower(qs.qudits[1]).export_to_controller()
    expected = 0.0 if frequency_hz is None else float(frequency_hz)
    assert [line["native_frequency_hz"] for line in out["lines"]] == [expected]
    assert [e["frequency_hz"] for e in out["events"] if e["kind"] == "readout"] == [expected]


def test_a_readout_frequency_with_frequencies_is_what_it_was():
    qudit = PhysicalQudit(0, 5.0e9, 0.0, 0.0, readout=DispersiveReadout(frequency_hz=7.0e9))
    qs = pulse.quantum_system(name="read", qudits={0: qudit})
    assert qs._frequency_tables(1)[2][0] == 7.0e9


# ── Lines that several qudits share ─────────────────────────────────────────


@pulse.kernel
def three_readouts(a, b, c):
    ra, rb, rc = get_readout_line(a), get_readout_line(b), get_readout_line(c)  # noqa: F821
    sync(ra, rb, rc)  # noqa: F821
    readout(ra, square(200, 0.05))  # noqa: F821
    readout(rb, square(200, 0.05))  # noqa: F821
    readout(rc, square(200, 0.05))  # noqa: F821


@pulse.kernel
def three_drives(a, b, c):
    for q in (a, b, c):
        drive(get_drive_line(q), square(200, 0.05))  # noqa: F821


def _wired(frequencies, shared, **limits):
    lines = {"drive_lines": [0, 1, 2], "readout_lines": [0, 1, 2]}
    if shared == "readout":
        lines["readout_lines"] = [ReadoutLine([0, 1, 2], max_concurrent_tones=3)]
    elif shared == "drive":
        lines["drive_lines"] = [DriveLine([0, 1, 2], max_concurrent_tones=3)]
    controller = pulse.Controller(**lines, **limits)
    qudits = {i: PhysicalQudit(i, f, 0.0, 0.0, drive=DRIVE) for i, f in enumerate(frequencies)}
    return pulse.quantum_system(
        name="wired",
        qudits=qudits,
        clock_ghz=1.0,
        control_system=pulse.ControlSystem(controllers=[controller]),
    )


@pytest.mark.parametrize(
    ("kernel", "shared"), [(three_readouts, "readout"), (three_drives, "drive")]
)
def test_a_shared_line_that_counts_frequencies_needs_them(kernel, shared):
    def lowered(frequencies, shared=shared, **limits):
        qs = _wired(frequencies, shared, **limits)
        return pulse.compile(kernel, quantum_system=qs).lower(*qs.qudits[:3])

    # With frequencies or offsets the three tones are three frequencies.
    for frequencies in ([5.0e9, 5.1e9, 5.2e9], [Offset(0.0), Offset(1.0e8), Offset(2.0e8)]):
        with pytest.raises(pulse.controller.ControllerLimitError, match="3 tone frequencies"):
            lowered(frequencies, max_frequencies_per_line=2)
        assert lowered(frequencies, max_frequencies_per_line=3) is not None
    # Without, each is 0 Hz from its own carrier: the count is refused, not passed.
    with pytest.raises(pulse.CompilationError) as caught:
        lowered([None] * 3, max_frequencies_per_line=2)
    message = str(caught.value)
    assert (
        f"the {shared} line of qudits 0..2 (controllers[0].{shared}_lines[0]) is shared" in message
    )
    assert "holds 2 tone frequencies per line (max_frequencies_per_line)" in message
    assert "give no frequency_hz" in message and "count as one" in message
    # A shared line that is not counted, and lines of their own that are, compile.
    assert lowered([None] * 3) is not None
    assert lowered([None] * 3, shared=None, max_frequencies_per_line=1) is not None


# ── A Calibration ───────────────────────────────────────────────────────────


def test_a_calibration_of_couplings_needs_absolute_frequencies():
    from cudaq_pulse.quantum_systems import transmon_surface17

    published = transmon_surface17(calibration=None)
    couplings = list(published.couplings)
    # Without the readouts' frequencies, which a system of offsets does not take.
    absolute = {
        i: q.replace(readout=q.readout.replace(frequency_hz=None))
        for i, q in published.physical_qudits.items()
    }
    reference = min(q.frequency_hz for q in absolute.values())
    offsets = {
        i: q.replace(frequency_hz=Offset(q.frequency_hz - reference)) for i, q in absolute.items()
    }
    with pytest.raises(pulse.UnsupportedError) as caught:
        pulse.quantum_system(
            name="s17", qudits=offsets, couplings=couplings, calibration="transmon_surface17"
        )
    message = str(caught.value)
    assert "gives its frequency_hz as Offset(" in message
    assert "a Calibration computes a coupling's exchange" in message
    # An offset inside a circuit's tuning range is refused as well, not taken for a frequency.
    inside = {i: q.replace(frequency_hz=Offset(q.frequency_hz)) for i, q in absolute.items()}
    with pytest.raises(pulse.UnsupportedError, match="takes no Calibration of its couplings"):
        pulse.quantum_system(
            name="s17", qudits=inside, couplings=couplings, calibration="transmon_surface17"
        )
    # The same qudits with their frequencies take it, and without it the offsets build.
    calibrated = pulse.quantum_system(
        name="s17", qudits=absolute, couplings=couplings, calibration="transmon_surface17"
    )
    shipped = pulse.quantum_system("transmon_surface17")
    assert [c.strength_hz for c in calibrated.couplings] == [
        c.strength_hz for c in shipped.couplings
    ]
    assert pulse.quantum_system(name="s17", qudits=offsets, couplings=couplings) is not None


# ── Parameter drift, whatever a model returns ───────────────────────────────


class _Replacing:
    """A drift model that returns changed copies, as the protocol says."""

    def start(self, calibrated, rng):
        return calibrated.replace(t1_us=calibrated.t1_us)

    def advance(self, true, calibrated, seconds, rng):
        return true.replace()


class _LosingAtStart(_Replacing):
    def start(self, calibrated, rng):
        return calibrated.replace(frequency_hz=None)


class _LosingLater(_Replacing):
    def advance(self, true, calibrated, seconds, rng):
        return true.replace(frequency_hz=Offset(1.0))


@pytest.mark.parametrize("frequency_hz", [None, Offset(1.5e8)])
def test_parameter_drift_is_refused_whatever_class_the_model_returns(frequency_hz):
    qs = _system(frequency_hz)
    with pytest.raises(pulse.UnsupportedError, match="no frequency to drift from"):
        qs.replace(parameter_drift=True, parameter_drift_seed=0)
    with pytest.raises(InvalidValueError) as caught:
        qs.replace(parameter_drift=_Replacing(), parameter_drift_seed=0)
    message = str(caught.value)
    assert "a drift model's start() returned qudit 0 with values a quantum system cannot" in message
    assert f"its frequency_hz is {frequency_hz!r}: parameter_drift needs qudits with" in message


def test_a_drift_model_that_loses_the_frequency_is_refused_where_it_does():
    qs = _system(5.0e9)
    with pytest.raises(InvalidValueError) as caught:
        qs.replace(parameter_drift=_LosingAtStart(), parameter_drift_seed=0)
    message = str(caught.value)
    assert "a drift model's start() returned qudit 0 with values a quantum system cannot" in message
    assert "start().frequency_hz: None is not a positive, finite number of Hz" in message
    drifting = qs.replace(parameter_drift=_LosingLater(), parameter_drift_seed=0)
    with pytest.raises(InvalidValueError) as caught:
        drifting.advance(1 * pulse.s)
    assert "advance().frequency_hz: Offset(1.0) is not a positive, finite" in str(caught.value)
    # The true values and the clock are as they were.
    assert drifting.true_qudits[0].frequency_hz == 5.0e9
    assert float(drifting.parameter_drift_time) == 0.0
    # A model that keeps the frequency drifts as before.
    kept = qs.replace(parameter_drift=_Replacing(), parameter_drift_seed=0)
    kept.advance(1 * pulse.s)
    assert kept.true_qudits[0].frequency_hz == 5.0e9


# ── The tone refusal, for every way a tone reaches a drive ──────────────────


def _levels3(frequency_hz, count=2, **more):
    qudits = {
        i: PhysicalQudit(i, frequency_hz, 0.0, 0.0, anharmonicity_hz=-2.0e8, drive=DRIVE)
        for i in range(count)
    }
    return pulse.quantum_system(name="pair", qudits=qudits, clock_ghz=1.0, **more)


def _played(kernel, qs, *handles):
    out = pulse.compile(kernel, quantum_system=qs).lower(*(handles or qs.qudits[:2]))
    events = out.export_to_controller()["events"]
    return [(e["qudit"], e["frequency_hz"]) for e in events if e["kind"] in ("drive", "readout")]


def _swapped(times):
    @pulse.kernel
    def names_swapped_in_a_loop(a, b):
        d, e = get_drive_line(a), get_drive_line(b)  # noqa: F821
        x, y = d.tone, e.tone
        for _ in range(times):
            x, y = y, x
        drive(d, square(40, 0.5), tone=x)  # noqa: F821

    return names_swapped_in_a_loop


def test_a_tone_is_followed_through_the_names_that_hold_it():
    absolute = pulse.quantum_system(
        name="pair",
        qudits={
            i: PhysicalQudit(i, f, 0.0, 0.0, drive=DRIVE) for i, f in enumerate((5.0e9, 5.15e9))
        },
        clock_ghz=1.0,
    )
    without = _levels3(None)
    # After an odd number of swaps x is the other qudit's tone, after an even one its own.
    assert _played(_swapped(201), absolute) == [(0, 5.15e9)]
    assert _played(_swapped(200), absolute) == [(0, 5.0e9)]
    with pytest.raises(pulse.CompilationError, match="tone= is the tone of qudit 1, and the line"):
        _played(_swapped(201), without)
    assert _played(_swapped(200), without) == [(0, 0.0)]


class _Value:
    """A line or tone as the builder makes one; the trace records no channel for it."""

    chan = None

    def __init__(self, vid, vtype):
        self.vid, self.vtype = vid, vtype


class _Builder:
    """The part of a builder the refusal reads."""

    def __init__(self):
        self._qref_to_qudit = {0: 0, 1: 1}
        self.made = 10

    def emit(self, kind, operands=(), result_types=(), attrs=None):
        first = self.made
        self.made += len(result_types)
        return tuple(_Value(first + k, vtype) for k, vtype in enumerate(result_types))


def test_a_tone_without_a_channel_is_placed_by_what_made_it_or_refused():
    # The refusal does not pass a drive whose line or tone the trace records
    # no channel for: it knows the qudit of what the builder made from a
    # line, and refuses what it cannot place.
    import types

    from cudaq_pulse import compiler

    builder, qs = _Builder(), _levels3(None)
    machine = types.SimpleNamespace(_system=None)
    compiler._refuse_other_qudits_tones(builder, qs, machine, None, ())
    emit = builder.emit
    pair = ("drive_line", "tone")
    d, d_tone = emit("pulse.get_drive_line", (_Value(0, "qudit"),), pair)
    e, e_tone = emit("pulse.get_drive_line", (_Value(1, "qudit"),), pair)
    wf = _Value(5, "waveform")
    # Its own tone, also after ops that make a new tone or line from it.
    d, d_tone = emit("pulse.drive", (d, wf, d_tone), pair)
    (d_tone,) = emit("pulse.shift_phase", (d_tone,), ("tone",))
    d, e = emit("pulse.sync", (d, e), ("drive_line", "drive_line"))
    d, d_tone = emit("pulse.drive", (d, wf, d_tone), pair)
    # The other qudit's tone, named.
    (shifted,) = emit("pulse.shift_phase", (e_tone,), ("tone",))
    for tone in (e_tone, shifted):
        with pytest.raises(pulse.CompilationError) as caught:
            emit("pulse.drive", (d, wf, tone), pair)
        assert "tone= is the tone of qudit 1, and the line is that of qudit 0;" in str(caught.value)
    with pytest.raises(pulse.CompilationError, match=r"readout\(\): tone= is the tone of qudit 0"):
        emit("pulse.readout", (e, wf, d_tone), (*pair, "bit"))
    # A tone or line of no known qudit: a loop's result, or a value never seen.
    (looped,) = emit("scf.for_end", (d_tone,), ("tone",))
    for line, tone in ((d, looped), (d, _Value(99, "tone")), (_Value(98, "drive_line"), d_tone)):
        with pytest.raises(pulse.CompilationError) as caught:
            emit("pulse.drive", (line, wf, tone), pair)
        message = str(caught.value)
        assert "drive(): the trace cannot tell which qudit's tone tone= is" in message
        assert "give no frequency_hz" in message and "line.tones[(1, 2)]" in message
    # An id used again for another value is not placed by what it was.
    builder.made = d_tone.vid
    (reused,) = emit("scf.for_end", (), ("tone",))
    assert reused.vid == d_tone.vid
    with pytest.raises(pulse.CompilationError, match="cannot tell which qudit's tone"):
        emit("pulse.drive", (d, wf, reused), pair)


def test_another_qudits_tone_is_refused_however_it_reaches_the_drive():
    @pulse.kernel
    def after_a_loop(a, b):
        d, e = get_drive_line(a), get_drive_line(b)  # noqa: F821
        for _ in range(200):
            drive(d, square(40, 0.5))  # noqa: F821
            drive(e, square(40, 0.5))  # noqa: F821
        drive(d, square(40, 0.5), tone=e.tone)  # noqa: F821

    @pulse.kernel
    def in_a_loop(a, b):
        d, e = get_drive_line(a), get_drive_line(b)  # noqa: F821
        for _ in range(200):
            drive(d, square(40, 0.5), tone=e.tone)  # noqa: F821

    @pulse.kernel
    def a_name_through_a_loop(a, b):
        d, e = get_drive_line(a), get_drive_line(b)  # noqa: F821
        t = e.tone
        for _ in range(200):
            t = shift_phase(t, 0.1)  # noqa: F821
        drive(d, square(40, 0.5), tone=t)  # noqa: F821

    @pulse.kernel
    def a_list_through_a_loop(a, b):
        d, e = get_drive_line(a), get_drive_line(b)  # noqa: F821
        tones = [d.tone, e.tone]
        for _ in range(200):
            tones[1] = shift_phase(tones[1], 0.1)  # noqa: F821
        drive(d, square(40, 0.5), tone=tones[1])  # noqa: F821

    @pulse.kernel
    def after_a_frequency_is_set(a, b):
        d, e = get_drive_line(a), get_drive_line(b)  # noqa: F821
        drive(d, square(40, 0.5), tone=set_frequency(e.tone, 1.0e6))  # noqa: F821

    @pulse.kernel
    def another_transition(a, b):
        d, e = get_drive_line(a), get_drive_line(b)  # noqa: F821
        drive(d, square(40, 0.5), tone=e.tones[(1, 2)])  # noqa: F821

    @pulse.kernel
    def after_a_measured_branch(a, b):
        d, e = get_drive_line(a), get_drive_line(b)  # noqa: F821
        if readout(get_readout_line(a), square(200, 0.05)):  # noqa: F821
            drive(d, square(40, 0.5))  # noqa: F821
        drive(d, square(40, 0.5), tone=e.tone)  # noqa: F821

    without = _levels3(None)
    for kernel in (
        after_a_loop,
        in_a_loop,
        a_name_through_a_loop,
        a_list_through_a_loop,
        after_a_frequency_is_set,
        another_transition,
        after_a_measured_branch,
    ):
        with pytest.raises(pulse.CompilationError) as caught:
            _played(kernel, without)
        assert "tone= is the tone of qudit 1, and the line is that of qudit 0;" in str(
            caught.value
        ), kernel.__name__
        # With frequencies each of them compiles.
        assert _played(kernel, _levels3(5.0e9)), kernel.__name__


def test_a_readout_at_another_qudits_tone_is_refused():
    @pulse.kernel
    def at_its_readout_tone(a, b):
        r, s = get_readout_line(a), get_readout_line(b)  # noqa: F821
        readout(r, square(200, 0.05), tone=s.tone)  # noqa: F821

    @pulse.kernel
    def at_its_drive_tone(a, b):
        r, e = get_readout_line(a), get_drive_line(b)  # noqa: F821
        readout(r, square(200, 0.05), tone=e.tone)  # noqa: F821

    @pulse.kernel
    def at_its_own_drive_tone(a, b):
        r, d = get_readout_line(a), get_drive_line(a)  # noqa: F821
        readout(r, square(200, 0.05), tone=d.tone)  # noqa: F821
        readout(r, square(200, 0.05))  # noqa: F821

    without = _levels3(None)
    for kernel in (at_its_readout_tone, at_its_drive_tone):
        with pytest.raises(pulse.CompilationError) as caught:
            _played(kernel, without)
        message = str(caught.value)
        assert (
            "readout(): tone= is the tone of qudit 1, and the line is that of qudit 0;" in message
        )
        assert _played(kernel, _levels3(5.0e9))
    assert _played(at_its_own_drive_tone, without) == [(0, 0.0), (0, 0.0)]


def test_the_refusal_names_qudits_as_the_kernel_was_given_them():
    @pulse.kernel
    def cross(a, b):
        d, e = get_drive_line(a), get_drive_line(b)  # noqa: F821
        drive(d, square(40, 0.5), tone=e.tone)  # noqa: F821

    @pulse.kernel
    def of_any_qudits(first: pulse.Qudit, second: pulse.Qudit):
        d, e = get_drive_line(first), get_drive_line(second)  # noqa: F821
        drive(d, square(40, 0.5), tone=e.tone)  # noqa: F821

    @pulse.kernel
    def of_a_register(qudits: pulse.Register):
        d, e = get_drive_line(qudits[0]), get_drive_line(qudits[1])  # noqa: F821
        drive(e, square(40, 0.5), tone=d.tone)  # noqa: F821

    # By the user's qudit numbers, not the carriers' (carriers 2 and 0 here).
    laid_out = _levels3(None, count=3, layout=[2, 0])
    with pytest.raises(pulse.CompilationError) as caught:
        _played(cross, laid_out)
    assert "is the tone of qudit 1, and the line is that of qudit 0;" in str(caught.value)
    # Runtime qudits by the kernel's parameters, not by their slots.
    without = _levels3(None)
    with pytest.raises(pulse.CompilationError) as caught:
        _played(of_any_qudits, without)
    message = str(caught.value)
    assert (
        "is the tone of the qudit 'second', and the line is that of the qudit 'first';" in message
    )
    assert "-1" not in message and "-2" not in message
    with pytest.raises(pulse.CompilationError) as caught:
        _played(of_a_register, without, pulse.Register(without.qudits[:2]))
    message = str(caught.value)
    assert "is the tone of the qudit qudits[0], and the line is that of the qudit qudits[1];" in (
        message
    )


def test_a_lines_own_tones_stay_allowed_however_they_reach_the_drive():
    @pulse.kernel
    def own(a, b):
        d, e = get_drive_line(a), get_drive_line(b)  # noqa: F821
        lines = [d, e]
        t = d.tone
        for _ in range(200):
            t = shift_phase(t, 0.1)  # noqa: F821
            for line in lines:
                drive(line, square(40, 0.5))  # noqa: F821
        drive(d, square(40, 0.5), tone=t)  # noqa: F821
        drive(lines[1], square(40, 0.5), tone=lines[1].tone)  # noqa: F821
        drive(d, square(40, 0.5), tone=d.tones[(1, 2)])  # noqa: F821
        if readout(get_readout_line(a), square(200, 0.05)):  # noqa: F821
            drive(d, square(40, 0.5))  # noqa: F821
        drive(d, square(40, 0.5), tone=d.tone)  # noqa: F821

    @pulse.kernel
    def of_any_qudits(first: pulse.Qudit, second: pulse.Qudit):
        d, e = get_drive_line(first), get_drive_line(second)  # noqa: F821
        for _ in range(200):
            drive(d, square(40, 0.5), tone=d.tone)  # noqa: F821
            drive(e, square(40, 0.5))  # noqa: F821
        drive(e, square(40, 0.5), tone=e.tone)  # noqa: F821

    @pulse.kernel
    def through_a_measured_branch(a, b):
        d = get_drive_line(a)  # noqa: F821
        t, tones = d.tone, [d.tone, d.tones[(1, 2)]]
        for _ in range(100):
            if readout(get_readout_line(a), square(200, 0.05)):  # noqa: F821
                t = shift_phase(t, 0.2)  # noqa: F821
                tones[1] = shift_phase(tones[1], 0.2)  # noqa: F821
            drive(d, square(40, 0.5), tone=t)  # noqa: F821
        drive(d, square(40, 0.5), tone=tones[1])  # noqa: F821

    without = _levels3(None)
    played = _played(own, without)
    assert {frequency for _, frequency in played} == {0.0, -2.0e8}
    assert played[-1] == (0, 0.0)
    assert _played(of_any_qudits, without)[-1] == (1, 0.0)
    # Tones a measured branch may have moved are run-time values of qudit 0.
    played = _played(through_a_measured_branch, without)
    assert len(played) > 100 and {qudit for qudit, _ in played} == {0}


def test_a_tone_that_is_one_qudits_or_anothers_by_a_measurement_is_refused():
    @pulse.kernel
    def either(a, b):
        d, e = get_drive_line(a), get_drive_line(b)  # noqa: F821
        t = d.tone
        if readout(get_readout_line(a), square(200, 0.05)):  # noqa: F821
            t = shift_phase(e.tone, 0.2)  # noqa: F821
        drive(d, square(40, 0.5), tone=t)  # noqa: F821

    # No system compiles it; one without frequencies says so at the drive.
    with pytest.raises(pulse.CompilationError, match="its arms bind it differently"):
        _played(either, _levels3(5.0e9))
    with pytest.raises(pulse.CompilationError) as caught:
        _played(either, _levels3(None))
    message = str(caught.value)
    assert "drive(): the trace cannot tell which qudit's tone tone= is" in message
    assert "what the arms of an if on a measurement bind differently" in message


# ── What is left as it was until the owner decides ──────────────────────────


def test_transition_freq_gives_magnitudes_and_the_hint_does_not_send_there():
    # qs.transition_freq is the array it was: magnitudes. On a system of
    # offsets the line's tones keep their signs, so the hint for a runtime
    # qudit names the model's signed value instead.
    own = _levels3(None)
    assert own.transition_freq[0, 1, 2] == 2.0e8
    assert own.model.transitions_hz(own.physical_qudits[0], 3)[(1, 2)] == -2.0e8

    @pulse.kernel
    def of_any_qudit(q: pulse.Qudit):
        line = get_drive_line(q)  # noqa: F821
        drive(line, square(40, 0.5), tone=line.tones[(1, 2)])  # noqa: F821

    for qs in (own, _referenced([-1.5e8, 0.0])):
        with pytest.raises(NotFoundError) as caught:
            pulse.compile(of_any_qudit, quantum_system=qs).lower(qs.qudits[0])
        message = str(caught.value)
        assert "qs.model.transitions_hz(qs.physical_qudits[q.index]," in message
        assert "not with qs.transition_freq, which gives magnitudes" in message
    absolute = _levels3(5.0e9)
    with pytest.raises(NotFoundError) as caught:
        pulse.compile(of_any_qudit, quantum_system=absolute).lower(absolute.qudits[0])
    message = str(caught.value)
    assert "called with qs.transition_freq[q.logical, 1, 2]" in message
    assert "transitions_hz" not in message
