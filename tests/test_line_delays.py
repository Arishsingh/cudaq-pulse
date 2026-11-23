# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrated line delays, declared with each line (``DriveLine(5,
delay_ns=4.0)``): whole grid steps move a line's operations, the rest is its
fine delay, and a change of delays alone schedules again."""

from __future__ import annotations

import copy
import json
import pickle

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.controller import DriveLine, ReadoutLine

from tests._handles import Q


@pulse.kernel
def two_lines(q, r):
    d = get_drive_line(q)
    e = get_drive_line(r)
    drive(d, square(8, 0.1))
    drive(e, square(8, 0.1))
    sync(d, e)
    drive(d, square(8, 0.1))


@pulse.kernel
def feedback(q, r):
    ro = get_readout_line(q)
    d = get_drive_line(r)
    m = readout(ro, square(100, 0.05))
    sync(ro, d)
    if m:
        drive(d, square(20, 0.3))


@pulse.kernel
def rabi(q, amplitude: float):
    d = get_drive_line(q)
    r = get_readout_line(q)
    for _ in range(3):
        drive(d, gaussian(16, amplitude, 4.0))
        wait(d, 4)
    sync(d, r)
    readout(r, square(40, 0.05))


def _delayed(delays=None, links=None, **controller):
    """A system of one controller (of the fields *controller*) playing the
    lines of qudits 0 to 3, each delayed as *delays* gives, by (role, qudit)."""
    delays = delays or {}
    box = pulse.Controller(
        drive_lines=[DriveLine(q, delay_ns=delays.get(("drive", q), 0.0)) for q in range(4)],
        readout_lines=[ReadoutLine(q, delay_ns=delays.get(("readout", q), 0.0)) for q in range(4)],
        **controller,
    )
    return pulse.ControlSystem(controllers=[box], links=links or {})


def _starts(lowered, kind="drive"):
    return [(e.qudit, e.start) for e in lowered.schedule if e.kind == kind]


def _lines(lowered):
    return {(line["role"], line["qudit"]): line for line in lowered.export_to_controller()["lines"]}


def test_whole_cycles_of_a_delay_move_the_lines_operations():
    plain = pulse.compile(two_lines, schedule="asap").lower(Q[0], Q[1])
    machine = _delayed({("drive", 0): 1.5})
    delayed = pulse.compile(two_lines, schedule="asap", control_system=machine).lower(Q[0], Q[1])
    # 1.5 ns is 3 cycles of the 2 GHz clock; only qudit 0's line moves.
    assert _starts(plain) == [(0, 0), (1, 0), (0, 8)]
    assert _starts(delayed) == [(0, 3), (1, 0), (0, 11)]
    lines = _lines(delayed)
    assert lines[("drive", 0)]["fine_delay_ns"] == 0.0
    assert lines[("drive", 1)]["fine_delay_ns"] == 0.0
    assert "#pulse.line<drive, qudits = [0], delay_cycles = 3>" in delayed.mlir()


def test_without_delays_nothing_changes():
    lowered = pulse.compile(two_lines).lower(Q[0], Q[1])
    assert "line_delays" not in lowered.mlir()
    assert all("fine_delay_ns" not in line for line in lowered.export_to_controller()["lines"])
    # A delay of no whole cycle moves nothing and adds no attribute.
    machine = _delayed({("drive", 0): 0.25}, max_fine_delay_ns=0.5)
    lowered = pulse.compile(two_lines, control_system=machine).lower(Q[0], Q[1])
    assert "line_delays" not in lowered.mlir()
    assert _lines(lowered)[("drive", 0)]["fine_delay_ns"] == pytest.approx(0.25)


def test_the_rest_of_a_delay_is_the_lines_fine_delay():
    # A 1 GHz controller starts every 2 cycles of the 2 GHz program clock:
    # steps of 1 ns.
    machine = _delayed({("drive", 1): 2.3}, clock_ghz=1.0, max_fine_delay_ns=1.0)
    lowered = pulse.compile(two_lines, schedule="asap", control_system=machine).lower(Q[0], Q[1])
    assert _starts(lowered) == [(0, 0), (1, 4), (0, 8)]
    lines = _lines(lowered)
    assert lines[("drive", 1)]["fine_delay_ns"] == pytest.approx(0.3)
    assert lines[("drive", 0)]["fine_delay_ns"] == 0.0
    # The export counts the controller's own cycles.
    program = lowered.export_to_controller()
    starts = [e["start_cycles"] for e in program["events"] if e["kind"] == "drive"]
    assert starts == [0, 2, 4]
    json.dumps(program)


def test_a_rest_the_controller_cannot_add_fails_to_compile():
    machine = _delayed({("drive", 1): 2.3}, clock_ghz=1.0)
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.compile(two_lines, control_system=machine).lower(Q[0], Q[1])
    message = str(caught.value)
    assert message.startswith("the drive line of qudit 1 is delayed 2.3 ns (its delay_ns)")
    assert (
        "0.3 ns (0.3 of its controller's cycles) from a whole number of its 1 ns timesteps"
        in message
    )
    assert "give it max_fine_delay_ns, raise its alignment_tolerance_cycles to 0.3" in message
    assert "calibrate the delay to 2 or 3 ns" in message
    # Within the tolerance, the delay rounds to the nearest step.
    machine = _delayed({("drive", 1): 2.7}, clock_ghz=1.0, alignment_tolerance_cycles=1)
    lowered = pulse.compile(two_lines, schedule="asap", control_system=machine).lower(Q[0], Q[1])
    assert _starts(lowered) == [(0, 0), (1, 6), (0, 8)]
    assert _lines(lowered)[("drive", 1)]["fine_delay_ns"] == 0.0
    # Past the fine delay a controller has.
    machine = _delayed({("drive", 1): 2.3}, clock_ghz=1.0, max_fine_delay_ns=0.25)
    with pytest.raises(pulse.CompilationError, match="adds at most 0.25 ns of fine delay"):
        pulse.compile(two_lines, control_system=machine).lower(Q[0], Q[1])


@pytest.mark.parametrize(
    "make, error, match",
    [
        (lambda: DriveLine(0, delay_ns=-1.0), pulse.InvalidValueError, "cannot be negative"),
        (lambda: DriveLine(0, delay_ns=-1 * pulse.ns), pulse.InvalidValueError, "-1.0 ns"),
        (lambda: DriveLine(0, delay_ns=float("nan")), pulse.InvalidValueError, "a number of ns"),
        (lambda: ReadoutLine(-1), pulse.InvalidValueError, "negative carrier index"),
        (lambda: DriveLine("d0"), pulse.InvalidTypeError, "a qudit or a list of qudits"),
        (
            lambda: pulse.Controller(drive_lines=[ReadoutLine(0)]),
            pulse.InvalidTypeError,
            "holds DriveLines or carrier indices",
        ),
        (
            lambda: pulse.Controller(drive_lines=[0, DriveLine(0, delay_ns=1.0)]),
            pulse.InvalidValueError,
            "lists a carrier twice",
        ),
    ],
)
def test_line_delays_are_checked(make, error, match):
    with pytest.raises(error, match=match):
        make()


def test_a_line_is_its_qudit_and_its_delay():
    line = DriveLine(5, delay_ns=4 * pulse.ns)
    assert (line.qudit, line.delay_ns) == (5, 4.0) and line == DriveLine(5, 4.0)
    assert line != DriveLine(5) and line != ReadoutLine(5, 4.0) and DriveLine(5) != 5
    assert repr(line) == "DriveLine(5, delay_ns=4.0)" and repr(ReadoutLine(5)) == "ReadoutLine(5)"
    with pytest.raises(AttributeError):
        line.delay_ns = 1.0  # type: ignore[misc]
    box = pulse.Controller(drive_lines=[line, 6], readout_lines=[ReadoutLine(5, delay_ns=12.0)])
    assert box.drive_lines == (line, DriveLine(6))
    assert box.as_dict()["drive_lines"] == [{"qudit": 5, "delay_ns": 4.0}, 6]
    assert not hasattr(pulse.ControlSystem(controllers=[box]), "line_delays_ns")


def test_delays_are_part_of_the_system():
    a = _delayed({("drive", 0): 1.0, ("readout", 1): 2.0})
    b = _delayed({("readout", 1): 2.0, ("drive", 0): 1.0})
    assert a == b and hash(a) == hash(b)
    assert a != _delayed() and a != _delayed({("drive", 0): 1.5})


def test_moved_lines_are_checked_like_any_others():
    machine = _delayed(feedback_latency_cycles=20)
    assert pulse.compile(feedback, schedule="rcp", control_system=machine).lower(Q[0], Q[1])
    later = _delayed({("drive", 1): 5.0}, feedback_latency_cycles=20)
    assert pulse.compile(feedback, schedule="rcp", control_system=later).lower(Q[0], Q[1])
    # The readout ends 10 cycles later: every schedule waits for it.
    late_readout = _delayed({("readout", 0): 5.0}, feedback_latency_cycles=20)
    for schedule in ("rcp", "asap", "alap"):
        lowered = pulse.compile(feedback, schedule=schedule, control_system=late_readout).lower(
            Q[0], Q[1]
        )
        readout, drive = lowered.schedule
        assert (readout.end, drive.start) == (110, 130)
    # A schedule made without the latency is too close to the branch on it.
    early = pulse.compile(
        feedback, schedule="asap", control_system=_delayed({("readout", 0): 5.0})
    ).lower(Q[0], Q[1])
    [message] = pulse.controller.check(early, late_readout.controllers[0])
    assert "ending at 110; the controller needs 20" in message


# ── Scheduling again ─────────────────────────────────────────────────────────


def _fresh(kernel):
    """The same kernel, without the specializations *kernel* traced."""
    return pulse.kernel(schedule=kernel.schedule, unroll_threshold=kernel.unroll_threshold)(
        kernel.__pulse_kernel_fn__
    )


def _outputs(lowered):
    return (
        lowered.mlir(),
        json.dumps(lowered.export_to_controller(), sort_keys=True),
        json.dumps(lowered.export_to_emulator(), sort_keys=True),
    )


MIXED = {"clock_ghz": 1.0, "max_fine_delay_ns": 1.0}
# Traced with the first, then scheduled again with each of the others.
DELAYS = (
    {("drive", 0): 1.0},
    {("drive", 0): 2.5, ("readout", 0): 1.5, ("drive", 1): 4.0},
    {},
    {("readout", 0): 3.0},
)


@pytest.mark.parametrize("controller", ({}, MIXED), ids=("program_clock", "mixed"))
@pytest.mark.parametrize("schedule", ("asap", "alap", "rcp", "alap_rcp"))
@pytest.mark.parametrize(
    "kernel, args",
    [(two_lines, (Q[0], Q[1])), (feedback, (Q[0], Q[1])), (rabi, (Q[0], 0.3))],
    ids=("two_lines", "feedback", "rabi"),
)
def test_new_delays_alone_schedule_again_to_the_same_bytes(kernel, args, schedule, controller):
    kernel = _fresh(kernel)
    for k, delays in enumerate(DELAYS):
        machine = _delayed(delays, **controller)
        again = pulse.compile(kernel, schedule=schedule, control_system=machine).lower(*args)
        assert again.rescheduled_for_line_delays == (k > 0)
        full = pulse.compile(_fresh(kernel), schedule=schedule, control_system=machine).lower(*args)
        assert not full.rescheduled_for_line_delays
        assert _outputs(again) == _outputs(full)


def test_a_program_without_delays_is_traced_as_before():
    kernel = _fresh(two_lines)
    pulse.compile(kernel).lower(Q[0], Q[1])
    assert not kernel._traced
    machine = _delayed({("drive", 0): 1.0})
    pulse.compile(kernel, control_system=machine).lower(Q[0], Q[1])
    # The same delays again trace again, as without them.
    again = pulse.compile(kernel, control_system=machine).lower(Q[0], Q[1])
    assert not again.rescheduled_for_line_delays


def test_a_module_scheduled_again_for_new_delays_and_its_calls_say_so():
    kernel = _fresh(rabi)
    first = pulse.compile(kernel, control_system=_delayed({("drive", 0): 1.0})).lower(Q[0], 0.3)
    again = pulse.compile(kernel, control_system=_delayed({("drive", 0): 2.5})).lower(Q[0], 0.3)
    assert not first.rescheduled_for_line_delays
    assert not first.generic.rescheduled_for_line_delays
    assert again.rescheduled_for_line_delays and again.generic.rescheduled_for_line_delays
    assert again.generic.bind([0.5]).rescheduled_for_line_delays
    # A call bound again, with no values, reads its module's mark too.
    assert not first.bind([]).rescheduled_for_line_delays
    assert again.bind([]).rescheduled_for_line_delays
    module = pulse.compile(_fresh(two_lines)).lower(Q[0], Q[1])
    assert not module.bind([]).bind([]).rescheduled_for_line_delays
    # The amplitude moves no op: no call is scheduled again for its values.
    assert not any(p.rescheduled for p in (first, again, again.generic.bind([0.5])))
    # A copy is the program itself, and a pickle keeps how it was made.
    for program in (first, again, again.generic):
        assert copy.copy(program) is program and copy.deepcopy(program) is program
        restored = pickle.loads(pickle.dumps(program))
        assert restored.rescheduled_for_line_delays == program.rescheduled_for_line_delays
        assert restored.rescheduled == program.rescheduled
    with pytest.raises(AttributeError):
        again.rescheduled_for_line_delays = False  # type: ignore[misc]
    assert again.rescheduled_for_line_delays


@pulse.kernel
def waited(q, tau: float):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(8, 0.1))
    wait(d, tau)
    sync(d, r)
    readout(r, square(40, 0.05))


def test_a_module_checked_per_call_is_rescheduled_for_new_delays():
    # With a float wait, each call's values are checked against the
    # controller when they are bound. Each call is scheduled again with its
    # value, and the module is scheduled again only when compiled again for
    # new delays: a call of that module is both.
    kernel = _fresh(waited)
    first = pulse.compile(kernel, control_system=_delayed({("drive", 0): 1.0})).lower(Q[0], 30.0)
    again = pulse.compile(kernel, control_system=_delayed({("drive", 0): 2.5})).lower(Q[0], 30.0)
    assert first.rescheduled and not first.rescheduled_for_line_delays
    assert not first.generic.rescheduled and not first.generic.rescheduled_for_line_delays
    assert again.rescheduled and again.rescheduled_for_line_delays
    assert not again.generic.rescheduled and again.generic.rescheduled_for_line_delays


CALIBRATION = {"amplitude": 0.1}


@pulse.kernel
def calibrated(q):
    d = get_drive_line(q)
    drive(d, square(8, CALIBRATION["amplitude"]))


@pulse.kernel
def on_any_qudit(q: pulse.Qudit):
    d = get_drive_line(q)
    drive(d, square(8, 0.1))


def test_a_kernel_is_traced_again_when_its_trace_cannot_be_reused(monkeypatch):
    first, then, last = (_delayed({("drive", 0): ns}) for ns in (1.0, 2.5, 0.5))
    # A kernel with qudit parameters is traced for every compile.
    kernel = _fresh(on_any_qudit)
    pulse.compile(kernel, control_system=first).lower(Q[0])
    assert not pulse.compile(kernel, control_system=then).lower(Q[0]).rescheduled_for_line_delays
    # A change to what the kernel read traces it again, and that trace is
    # scheduled again for the next delays.
    kernel = _fresh(calibrated)
    pulse.compile(kernel, control_system=first).lower(Q[0])
    monkeypatch.setitem(CALIBRATION, "amplitude", 0.2)
    assert not pulse.compile(kernel, control_system=then).lower(Q[0]).rescheduled_for_line_delays
    assert pulse.compile(kernel, control_system=last).lower(Q[0]).rescheduled_for_line_delays


@pytest.mark.slow
def test_new_delays_schedule_the_matrix_again_to_the_same_bytes():
    from tests.classical import _schedule_matrix as matrix

    first = _delayed({("drive", 1): 1.0}, links=matrix.MACHINE.links)
    then = _delayed({("drive", 0): 2.5, ("readout", 1): 1.5}, links=matrix.MACHINE.links)
    checked = 0
    for case in matrix.all_cases()[::5]:
        if matrix.expected_error(case) is not None:
            continue
        kernel = matrix.program(case)
        matrix.compile_case(case, kernel=kernel, control_system=first).lower(
            Q[0], Q[1], matrix.LOOP, matrix.FLOAT_TAU
        )
        again = matrix.compile_case(case, kernel=kernel, control_system=then).lower(
            Q[0], Q[1], matrix.LOOP, matrix.FLOAT_TAU
        )
        full = matrix.compile_case(case, kernel=_fresh(kernel), control_system=then).lower(
            Q[0], Q[1], matrix.LOOP, matrix.FLOAT_TAU
        )
        if not again.rescheduled_for_line_delays:
            continue
        assert _outputs(again) == _outputs(full), case
        checked += 1
    assert checked > 20


def test_a_call_with_new_values_keeps_the_delays():
    machine = _delayed({("drive", 0): 1.0})
    compiled = pulse.compile(rabi, schedule="asap", control_system=machine)
    assert [s for _, s in _starts(compiled.lower(Q[0], 0.2))][0] == 2
    assert [s for _, s in _starts(compiled.lower(Q[0], 0.4))][0] == 2
