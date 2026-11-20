# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Update buses (``Controller.update_buses``): lines whose tone updates go
over one bus, one at a time, each holding it for ``cycles_per_update``."""

from __future__ import annotations

import pickle
from typing import Any

import cudaq_pulse as pulse
import cudaq_pulse.controller as controller_module
import pytest
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.controller import ControllerLimitError, UpdateBus, check

from tests._handles import Q, ladder_target

BUS = Controller(update_buses=[UpdateBus(drive_lines=[0, 1], cycles_per_update=8)])


def _lowered(kernel, **options):
    return pulse.compile(kernel, schedule="asap", **options).lower(Q[0], Q[1])


def _violation(event, line, at, previous, previous_line, previous_at, gap, cycles=8):
    return (
        f"update bus 0 (drive lines of qudits 0, 1): the {event} on line {line} at {at} "
        f"updates its tone {gap} cycles after the {previous} on line {previous_line} at "
        f"{previous_at}; the bus carries one update per {cycles} cycles "
        "(Controller.update_buses)"
    )


@pulse.kernel
def same_tones(q, r):
    d, e = get_drive_line(q), get_drive_line(r)
    drive(d, square(16, 0.1))
    drive(e, square(16, 0.1))
    drive(d, square(16, 0.2))
    drive(e, gaussian(16, 0.1, 4.0))


@pulse.kernel
def retuned(q, r):
    d = get_drive_line(q)
    shift_frequency(d.tone, 2.0e6)
    drive(d, square(16, 0.1))
    wait(d, 40)
    drive(d, gaussian(16, 0.1, 4.0))
    drive(d, square(16, 0.1))


def _after_a_wait(excursion: bool) -> pulse.Kernel:
    @pulse.kernel
    def after(q, r):
        d, e = get_drive_line(q), get_drive_line(r)
        drive(d, square(16, 0.1))
        drive(e, square(16, 0.1))
        if excursion:
            shift_frequency(d.tone, 2.0e6)
        wait(d, 40)
        if excursion:
            shift_frequency(d.tone, -2.0e6)
        shift_phase(e.tone, 0.5)
        sync(d, e)
        drive(d, gaussian(16, 0.1, 4.0))
        drive(e, gaussian(16, 0.1, 4.0))

    return after


def test_a_tone_back_at_its_frequency_after_an_excursion_is_an_update():
    # The frame ran 2 pi * 2 MHz * 20 ns further while retuned: a new phase.
    (found,) = check(_lowered(_after_a_wait(True)), BUS)
    assert "the drive on line 1 at 56 updates its tone 0 cycles after the drive on line 0" in found
    assert check(_lowered(_after_a_wait(False)), BUS) == []


@pulse.kernel
def turned_together(q, r):
    d, e = get_drive_line(q), get_drive_line(r)
    drive(d, square(16, 0.1))
    drive(e, square(16, 0.1))
    sync(d, e)
    shift_phase(d.tone, 0.5)
    shift_phase(e.tone, 0.5)
    drive(d, gaussian(16, 0.1, 4.0))
    drive(e, gaussian(16, 0.1, 4.0))


@pulse.kernel
def left_turned(q, r):
    d, e = get_drive_line(q), get_drive_line(r)
    drive(d, square(16, 0.1))
    drive(e, square(16, 0.1))
    sync(d, e)
    shift_phase(d.tone, 0.5)
    shift_phase(e.tone, 0.5)


def test_a_controller_has_no_update_buses_by_default():
    assert Controller().update_buses == ()
    assert "update_buses" not in Controller().as_dict()
    assert check(_lowered(turned_together), Controller()) == []


def test_pulses_that_keep_their_line_s_tone_are_not_updates():
    # Other envelopes and amplitudes at the same frequency and phase.
    assert check(_lowered(same_tones), BUS) == []
    # A retuned tone gains phase at its detuning between pulses, as it runs on.
    events = _lowered(retuned).export_to_controller()["events"]
    phases = [e["phase_rad"] for e in events if e["kind"] == "drive"]
    assert phases[0] == 0.0 and phases[1] != 0.0
    one_line = Controller(update_buses=[UpdateBus(drive_lines=[0], cycles_per_update=100)])
    assert check(_lowered(retuned), one_line) == []


def test_updates_on_one_bus_must_start_its_cycles_per_update_apart():
    lowered = _lowered(turned_together)
    assert check(lowered, BUS) == [_violation("drive", 1, 16, "drive", 0, 16, 0)]
    # A line off the bus updates as it likes.
    alone = Controller(update_buses=[UpdateBus(drive_lines=[0], cycles_per_update=8)])
    assert check(lowered, alone) == []
    machine = ControlSystem(controllers=[BUS])
    with pytest.raises(ControllerLimitError, match=r"update bus 0 .* 0 cycles after"):
        pulse.compile(turned_together, schedule="asap", control_system=machine).lower(Q[0], Q[1])


@pulse.kernel
def turned_by(q, r, angle: float):
    d, e = get_drive_line(q), get_drive_line(r)
    drive(d, square(16, 0.1))
    drive(e, square(16, 0.1))
    sync(d, e)
    shift_phase(d.tone, angle)
    shift_phase(e.tone, angle)
    drive(d, gaussian(16, 0.1, 4.0))
    drive(e, gaussian(16, 0.1, 4.0))


def test_a_phase_a_float_sets_is_checked_with_each_call_s_value():
    machine = ControlSystem(controllers=[BUS])
    compiled = pulse.compile(turned_by, schedule="asap", control_system=machine)
    compiled.lower(Q[0], Q[1], 0.0)
    with pytest.raises(ControllerLimitError, match="update bus 0"):
        compiled.lower(Q[0], Q[1], 0.5)


def test_a_frame_update_is_an_update():
    assert check(_lowered(left_turned), BUS) == [
        _violation("frame_update", 1, 16, "frame_update", 0, 16, 0)
    ]


@pulse.kernel
def apart(q, r):
    d, e = get_drive_line(q), get_drive_line(r)
    drive(d, square(16, 0.1))
    drive(e, square(24, 0.1))
    shift_phase(d.tone, 0.5)
    shift_phase(e.tone, 0.5)
    drive(d, gaussian(16, 0.1, 4.0))
    drive(e, gaussian(16, 0.1, 4.0))


def test_the_cycles_per_update_are_the_controller_s_own():
    lowered = _lowered(apart)
    buses = [UpdateBus(drive_lines=[0, 1], cycles_per_update=6)]
    # 8 cycles apart on the 2 GHz program clock, 4 on a 1 GHz controller.
    assert check(lowered, Controller(update_buses=buses)) == []
    (found,) = check(lowered, Controller(clock_ghz=1.0, update_buses=buses))
    assert "the drive on line 1 at 12 updates its tone 4 cycles after the drive on line 0 at 8" in (
        found
    )


@pulse.kernel
def in_arms(q, r):
    d, e = get_drive_line(q), get_drive_line(r)
    ro = get_readout_line(q)
    drive(d, square(16, 0.1))
    drive(e, square(16, 0.1))
    m = readout(ro, square(40, 0.05))
    sync(ro, d, e)
    if m:
        shift_phase(d.tone, 0.5)
        drive(d, gaussian(16, 0.1, 4.0))
    else:
        shift_phase(e.tone, 0.5)
        drive(e, gaussian(16, 0.1, 4.0))


def test_updates_in_the_two_arms_of_a_branch_never_meet_on_the_bus():
    assert check(_lowered(in_arms), BUS) == []


@pulse.kernel
def turning_loop(q, r):
    d, e = get_drive_line(q), get_drive_line(r)
    for _ in range(100):
        drive(d, square(16, 0.1))
        drive(e, square(16, 0.1))
        shift_phase(d.tone, 0.1)
        shift_phase(e.tone, 0.1)
        sync(d, e)


def test_a_rolled_loop_is_checked_at_every_iteration():
    rolled = _lowered(turning_loop, unroll_threshold=0)
    unrolled = _lowered(turning_loop, unroll_threshold=10**6)
    found = check(rolled, BUS)
    # 99 iterations take the turns of the one before; the last's are frame updates.
    assert len(found) == 100
    assert "the frame_update on line 1 at 1600" in found[-1]
    assert found == check(unrolled, BUS)
    assert found[0] == _violation("drive", 1, 16, "drive", 0, 16, 0)


@pulse.kernel
def three_turns(q, r):
    d, ro, e = get_drive_line(q), get_readout_line(q), get_drive_line(r)
    drive(d, square(16, 0.1))
    readout(ro, square(16, 0.05))
    drive(e, square(16, 0.1))
    shift_phase(d.tone, 0.5)
    shift_phase(ro.tone, 0.5)
    shift_phase(e.tone, 0.5)
    drive(d, gaussian(16, 0.1, 4.0))
    readout(ro, square(16, 0.05))
    drive(e, gaussian(16, 0.1, 4.0))


def test_each_controller_checks_the_buses_of_its_own_lines():
    def system(first, second):
        return ControlSystem(
            controllers=[
                Controller(drive_lines=[0], readout_lines=[0], update_buses=first),
                Controller(drive_lines=[1], readout_lines=[1], update_buses=second),
            ]
        )

    def compile_on(machine):
        return pulse.compile(three_turns, schedule="asap", control_system=machine).lower(Q[0], Q[1])

    # A bus reaches only its controller's lines: the drive of qudit 1 plays elsewhere.
    compile_on(system([{"drive_lines": [0, 1], "cycles_per_update": 8}], []))
    with pytest.raises(ControllerLimitError) as caught:
        compile_on(system([{"drive_lines": [0], "readout_lines": [0], "cycles_per_update": 8}], []))
    (violation,) = caught.value.violations
    assert violation.startswith(
        "controllers[0]: update bus 0 (drive line of qudit 0 and readout line of qudit 0): "
    )
    assert "at 16 updates its tone 0 cycles after" in violation


@pulse.kernel
def turns_on_every_line(*qs):
    drives = []
    for q in qs:
        drives.append(get_drive_line(q))
    for d in drives:
        drive(d, square(16, 0.1))
    for d in drives:
        shift_phase(d.tone, 0.5)
        drive(d, gaussian(16, 0.1, 4.0))


class _Passes(list):
    """A list that counts the passes made over it."""

    passes = 0

    def __iter__(self):
        _Passes.passes += 1
        return super().__iter__()


def _bus_reads(count, monkeypatch, listed=True):
    """Compiling and exporting every controller of *count*, two qudits each
    and a bus per line: the events read through another controller's view
    of them (``_SeenBy``), the passes over the updates per bus check, and
    the exports."""
    from cudaq_pulse import _system
    from cudaq_pulse.quantum_systems import PhysicalQudit

    def buses(lines):
        return [UpdateBus(drive_lines=[q], cycles_per_update=8) for q in lines]

    controllers = [
        Controller(drive_lines=[2 * k, 2 * k + 1], update_buses=buses([2 * k, 2 * k + 1]))
        for k in range(count)
    ]
    if not listed:
        controllers.append(Controller(update_buses=buses([0, 1])))

    qs = pulse.quantum_system(
        name="buses",
        clock_ghz=1.0,
        qudits={
            q: PhysicalQudit(q, 5.0e9 + 1.0e7 * q, 50.0, 30.0, anharmonicity_hz=-2.0e8)
            for q in range(2 * count)
        },
        control_system=ControlSystem(controllers=controllers),
    )
    counts = {"seen": 0, "checks": 0}
    seen, updates = _system._SeenBy._seen, controller_module._updates

    def counted_seen(self, e):
        counts["seen"] += 1
        return seen(self, e)

    def counted_updates(*args):
        counts["checks"] += 1
        found = updates(*args)
        return None if found is None else _Passes(found)

    monkeypatch.setattr(_system._SeenBy, "_seen", counted_seen)
    monkeypatch.setattr(controller_module, "_updates", counted_updates)
    _Passes.passes = 0
    lowered = pulse.compile(turns_on_every_line, schedule="asap", quantum_system=qs).lower(
        *(qs.qudits[q] for q in range(2 * count))
    )
    exports = [lowered.export_to_controller(controller=k) for k in range(len(controllers))]
    monkeypatch.undo()
    return counts["seen"], _Passes.passes / counts["checks"], exports


def test_each_controller_s_bus_check_reads_its_own_updates_once(monkeypatch):
    # A controller's bus check reads the updates on its lines, not every
    # event of the program through its view of them, and makes one pass over
    # them whatever its number of buses.
    for count in (4, 16):
        seen, passes, exports = _bus_reads(count, monkeypatch)
        events = sum(len(e["events"]) for e in exports)
        assert events == 4 * count
        assert seen <= events
        assert passes == 1
    # A controller with buses that lists no lines has every line on its
    # buses, another's too, and reads them through its view of every event.
    seen, passes, exports = _bus_reads(4, monkeypatch, listed=False)
    assert seen >= sum(len(e["events"]) for e in exports)
    assert passes == 1


def test_exports_list_the_buses_and_descriptions_read_them_back():
    machine = ControlSystem(controllers=[BUS])
    lowered = pulse.compile(same_tones, schedule="asap", control_system=machine).lower(Q[0], Q[1])
    # As a system file states them.
    listed: list[Any] = [{"drive_lines": [0, 1], "readout_lines": [], "cycles_per_update": 8}]
    assert lowered.export_to_controller()["limits"]["update_buses"] == listed
    assert lowered.export_to_controller(version=2)["controller"]["update_buses"] == listed
    again = Controller(**BUS.as_dict())
    assert again == BUS and hash(again) == hash(BUS)
    assert Controller(update_buses=listed) == BUS
    assert pickle.loads(pickle.dumps(BUS)) == BUS
    loaded = pulse.controller(
        "reference.controller", drive_lines=[0, 1], readout_lines=[0, 1], update_buses=listed
    )
    assert loaded.update_buses == BUS.update_buses
    wired = Controller(drive_lines=[0, 1], readout_lines=[0, 1], update_buses=listed)
    qs = ladder_target(2, control_system=ControlSystem(controllers=[wired]))
    assert pulse.QuantumSystem.from_dict(qs.as_dict()).control_system == qs.control_system


@pytest.mark.parametrize(
    ("buses", "error", "message"),
    [
        ([{"drive_lines": []}], pulse.InvalidValueError, "UpdateBus lists no lines"),
        ([{"drive_lines": [0], "cycles_per_update": 0}], pulse.InvalidValueError, "at least 1"),
        ([{"drive_lines": [0, 0]}], pulse.InvalidValueError, "lists a carrier twice"),
        ([{"readout_lines": [-1]}], pulse.InvalidValueError, "carrier indices"),
        ([{"lines": [0]}], pulse.InvalidValueError, "unexpected keyword argument 'lines'"),
        (
            [UpdateBus(drive_lines=[0]), UpdateBus(drive_lines=[0, 1])],
            pulse.InvalidValueError,
            "the drive line of qudit 0 is on another bus already",
        ),
        (UpdateBus(drive_lines=[0]), pulse.InvalidTypeError, "is a list of"),
        ([[0, 1]], pulse.InvalidTypeError, "is not a cudaq_pulse.controller.UpdateBus"),
    ],
)
def test_a_bad_bus_is_refused(buses, error, message):
    with pytest.raises(error, match=message):
        Controller(update_buses=buses)


# ── Spacing ─────────────────────────────────────────────────────────────────

SPACED = Controller(
    update_buses=[UpdateBus(drive_lines=[0, 1], cycles_per_update=8)], space_updates=True
)


def _spaced(kernel, controller=SPACED, **options):
    machine = ControlSystem(controllers=[controller])
    return pulse.compile(kernel, schedule="asap", control_system=machine, **options).lower(
        Q[0], Q[1]
    )


def _starts(lowered, kind="drive"):
    return [
        (e["line"], e["start_cycles"])
        for e in lowered.export_to_controller()["events"]
        if e["kind"] == kind
    ]


def test_spacing_is_asked_for_and_holds_the_later_update_back():
    assert SPACED.as_dict()["space_updates"] is True
    assert "space_updates" not in Controller().as_dict()
    assert Controller(**SPACED.as_dict()) == SPACED
    with pytest.raises(pulse.InvalidValueError, match="space_updates must be a bool"):
        Controller(space_updates=1)  # type: ignore[arg-type]
    with pytest.raises(ControllerLimitError, match="update bus 0"):
        _spaced(turned_together, BUS)
    lowered = _spaced(turned_together)
    assert _starts(lowered) == [(0, 0), (1, 0), (0, 16), (1, 24)]
    assert check(lowered, SPACED) == []
    assert lowered.export_to_controller()["limits"]["space_updates"] is True


@pulse.kernel
def turned_then_synced(q, r):
    d, e = get_drive_line(q), get_drive_line(r)
    drive(d, square(16, 0.1))
    drive(e, square(16, 0.1))
    sync(d, e)
    shift_phase(d.tone, 0.5)
    shift_phase(e.tone, 0.5)
    drive(d, gaussian(16, 0.1, 4.0))
    drive(e, gaussian(16, 0.1, 4.0))
    sync(d, e)
    drive(d, square(16, 0.1))


def test_a_held_update_moves_what_follows_it_through_syncs():
    # e's second drive waits 8 cycles, so the sync, and d's third drive, follow.
    assert _starts(_spaced(turned_then_synced)) == [(0, 0), (1, 0), (0, 16), (1, 24), (0, 40)]


def test_a_frame_update_is_held_back_too():
    lowered = _spaced(left_turned)
    assert _starts(lowered, "frame_update") == [(0, 16), (1, 24)]
    assert check(lowered, SPACED) == []


def test_a_rolled_loop_is_spaced_in_every_iteration():
    rolled = _spaced(turning_loop, unroll_threshold=0)
    assert check(rolled, SPACED) == []
    starts = _starts(rolled)
    # e's drive is one op, held back in every iteration: 8 cycles after d's.
    assert starts[:4] == [(0, 0), (1, 8), (0, 24), (1, 32)]
    assert starts[-1] == (1, 99 * 24 + 8)


def test_updates_in_the_two_arms_of_a_branch_need_no_spacing():
    spaced, plain = _spaced(in_arms), _lowered(in_arms)
    assert _starts(spaced) == _starts(plain)
    assert "pulse.op_id" not in spaced.mlir()


def test_the_spaced_export_and_the_simulator_agree():
    lowered = _spaced(turned_then_synced)
    simulated = [(e.qudit, e.start) for e in lowered.schedule if e.kind == "drive"]
    assert simulated == [(0, 0), (1, 0), (0, 16), (1, 24), (0, 40)]
    assert len(pulse.simulate(lowered).measurements) == 0


def test_spacing_that_does_not_settle_is_an_error(monkeypatch):
    import cudaq_pulse._bus_spacing as spacing

    monkeypatch.setattr(spacing, "MAX_ROUNDS", 1)
    with pytest.raises(pulse.CompilationError, match="did not settle after 1 reschedules"):
        _spaced(turning_loop, unroll_threshold=10**6)


# ── The fast check of rolled loops ──────────────────────────────────────────


def _loops(trips: int) -> dict[str, pulse.Kernel]:
    """Kernels of rolled loops of *trips* iterations on the two lines of BUS."""

    @pulse.kernel
    def lockstep(q, r):
        d, e = get_drive_line(q), get_drive_line(r)
        for _ in range(trips):
            drive(d, square(16, 0.1))
            drive(e, square(16, 0.1))
            shift_phase(d.tone, 0.1)
            shift_phase(e.tone, 0.1)
            sync(d, e)

    @pulse.kernel
    def apart(q, r):
        d, e = get_drive_line(q), get_drive_line(r)
        wait(e, 8)
        for _ in range(trips):
            drive(d, square(16, 0.1))
            drive(e, square(16, 0.1))
            shift_phase(d.tone, 0.1)
            shift_phase(e.tone, 0.1)

    # Entered from another tone on d, and a turned one on e: only iteration 0 clashes.
    @pulse.kernel
    def entered(q, r):
        d, e = get_drive_line(q), get_drive_line(r)
        drive(d, square(16, 0.1), tone=d.tones[(1, 2)])
        drive(e, square(16, 0.1))
        sync(d, e)
        shift_phase(e.tone, 0.2)
        for _ in range(trips):
            drive(d, square(16, 0.1))
            drive(e, square(16, 0.1))
            sync(d, e)

    # d leaves its frequency and comes back in every iteration: a new phase.
    @pulse.kernel
    def excursions(q, r):
        d, e = get_drive_line(q), get_drive_line(r)
        for _ in range(trips):
            drive(d, square(16, 0.1))
            drive(e, square(16, 0.1))
            shift_frequency(d.tone, 2.0e6)
            wait(d, 40)
            shift_frequency(d.tone, -2.0e6)
            shift_phase(e.tone, 0.1)
            sync(d, e)

    @pulse.kernel
    def excursions_alone(q, r):
        d, e = get_drive_line(q), get_drive_line(r)
        for _ in range(trips):
            drive(d, square(16, 0.1))
            drive(e, square(16, 0.1))
            shift_frequency(d.tone, 2.0e6)
            wait(d, 40)
            shift_frequency(d.tone, -2.0e6)
            sync(d, e)

    @pulse.kernel
    def nested(q, r):
        d, e = get_drive_line(q), get_drive_line(r)
        for _ in range(trips):
            drive(d, square(16, 0.1))
            shift_phase(d.tone, 0.1)
            for _ in range(9):
                drive(e, square(16, 0.1))
                shift_phase(e.tone, 0.1)
            sync(d, e)

    @pulse.kernel
    def nested_quiet(q, r):
        d, e = get_drive_line(q), get_drive_line(r)
        for _ in range(trips):
            drive(d, square(16, 0.1))
            shift_phase(d.tone, 0.1)
            for _ in range(9):
                drive(e, square(16, 0.1))
            sync(d, e)

    # Only what follows the loop clashes, against its last iteration's tone.
    @pulse.kernel
    def leaving(q, r):
        d, e = get_drive_line(q), get_drive_line(r)
        drive(e, square(16, 0.1))
        for _ in range(trips):
            drive(d, square(16, 0.1))
        sync(d, e)
        shift_phase(e.tone, 0.3)
        drive(d, square(16, 0.1), tone=d.tones[(1, 2)])
        drive(e, square(16, 0.1))

    @pulse.kernel
    def detuned(q, r):
        d, e = get_drive_line(q), get_drive_line(r)
        shift_frequency(d.tone, 1.0e6)
        for _ in range(trips):
            drive(d, square(16, 0.1))
            wait(d, 8)
            drive(e, square(16, 0.1))
            shift_phase(e.tone, 0.1)
            sync(d, e)

    return {
        k.__name__: k
        for k in (
            lockstep,
            apart,
            entered,
            excursions,
            excursions_alone,
            nested,
            nested_quiet,
            leaving,
            detuned,
        )
    }


def _every(lowered, limits):
    """The bus violations of a walk of every execution."""
    lines, events, *_ = controller_module._walk(lowered)
    return controller_module._bus_violations(lowered, events, limits, lines, None, None)


@pytest.mark.parametrize("trips", [3, 40])
@pytest.mark.parametrize("cycles", [8, 20])
@pytest.mark.parametrize(
    "name",
    [
        "lockstep",
        "apart",
        "entered",
        "excursions",
        "excursions_alone",
        "nested",
        "nested_quiet",
        "leaving",
        "detuned",
    ],
)
def test_the_fast_check_of_rolled_loops_finds_what_every_execution_has(name, cycles, trips):
    kernel = _loops(trips)[name]
    limits = Controller(update_buses=[UpdateBus(drive_lines=[0, 1], cycles_per_update=cycles)])
    rolled = _lowered(kernel, unroll_threshold=0)
    unrolled = _lowered(kernel, unroll_threshold=10**6)
    every = _every(rolled, limits)
    # A loop kept rolled walks updates at one time in another order, so which
    # of them is "the one before" can differ; how many clash cannot.
    assert len(every) == len(_every(unrolled, limits))
    fast = controller_module._bus_fast(rolled, rolled._concrete(), limits)
    assert bool(fast) == bool(every)
    assert list(check(rolled, limits)) == every
    assert list(check(unrolled, limits)) == _every(unrolled, limits)


def test_the_kernels_of_the_fast_check_cover_both_verdicts():
    limits = Controller(update_buses=[UpdateBus(drive_lines=[0, 1], cycles_per_update=8)])
    found = {
        name: bool(_every(_lowered(k, unroll_threshold=0), limits))
        for name, k in _loops(40).items()
    }
    assert found == {
        "lockstep": True,
        "apart": False,
        "entered": True,
        "excursions": True,
        "excursions_alone": False,
        "nested": True,
        "nested_quiet": False,
        "leaving": True,
        "detuned": False,
    }
