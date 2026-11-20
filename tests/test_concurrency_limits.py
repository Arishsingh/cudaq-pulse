# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``max_concurrent_drives`` and ``max_concurrent_readouts``, when a control
system with controllers is given them, hold under every schedule: ``rcp``
and ``alap_rcp`` keep to them, and a program another schedule places past
them fails to compile, naming the limit, the count and the first cycle."""

from __future__ import annotations

import pickle

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import controller
from cudaq_pulse.controller import ControllerLimitError, DriveLine, ReadoutLine

_RCP = ("rcp", "alap_rcp")


@pulse.kernel
def drives_then_readouts(qs: list):
    for q in qs:
        drive(get_drive_line(q), gaussian(80, 0.5, 20))
    for q in qs:
        readout(get_readout_line(q), square(400, 0.1))


@pulse.kernel
def drives_only(qs: list):
    for q in qs:
        drive(get_drive_line(q), gaussian(80, 0.5, 20))


@pulse.kernel
def back_to_back(q, r, s):
    a = get_drive_line(q)
    b = get_drive_line(r)
    c = get_drive_line(s)
    drive(a, square(40, 0.1))
    drive(b, square(40, 0.1))
    sync(a, b, c)
    drive(c, square(40, 0.1))


@pulse.kernel
def arms(q, r, s, t, u):
    m = readout(get_readout_line(q), square(100, 0.05))
    if m:
        drive(get_drive_line(r), square(40, 0.1))
        drive(get_drive_line(s), square(40, 0.1))
    else:
        drive(get_drive_line(t), square(40, 0.1))
        drive(get_drive_line(u), square(40, 0.1))


@pulse.kernel
def arms_and_more(q, r, s, t, u):
    ro = get_readout_line(q)
    d = get_drive_line(u)
    m = readout(ro, square(100, 0.05))
    sync(ro, d)
    if m:
        drive(get_drive_line(r), square(40, 0.1))
        drive(get_drive_line(s), square(40, 0.1))
    else:
        drive(get_drive_line(t), square(40, 0.1))
    drive(d, square(40, 0.1))


@pulse.kernel
def loop_in_an_arm(q, r, s, t):
    m = readout(get_readout_line(q), square(100, 0.05))
    if m:
        a = get_drive_line(r)
        b = get_drive_line(s)
        for _ in range(3):
            drive(a, gaussian(40, 0.1, 10))
            drive(b, gaussian(40, 0.1, 10))
            sync(a, b)
    else:
        drive(get_drive_line(t), square(200, 0.1))


@pulse.kernel
def branch_in_a_loop(q, r, s, t):
    m = readout(get_readout_line(q), square(100, 0.05))
    for _ in range(3):
        if m:
            a = get_drive_line(r)
            b = get_drive_line(s)
            for _ in range(3):
                drive(a, gaussian(40, 0.1, 10))
                drive(b, gaussian(40, 0.1, 10))
                sync(a, b)
        else:
            drive(get_drive_line(t), square(200, 0.1))


@pulse.kernel
def after_a_wait(q, r, gap: float):
    a = get_drive_line(q)
    b = get_drive_line(r)
    wait(a, gap)
    drive(a, square(40, 0.1))
    drive(b, square(40, 0.1))


@pulse.kernel
def in_a_loop(qs: list):
    for _ in range(3):
        for q in qs:
            drive(get_drive_line(q), square(40, 0.1))
        readout(get_readout_line(qs[0]), square(60, 0.05))


def _system(controllers=None, **limits):
    if controllers is None:
        controllers = [pulse.Controller(drive_lines=range(17), readout_lines=range(17))]
    control = pulse.ControlSystem(controllers=controllers, **limits)
    return pulse.quantum_system("transmon_surface17", control_system=control)


def _lower(kernel, system, schedule, *args):
    return pulse.compile(kernel, quantum_system=system, schedule=schedule).lower(*args)


def test_asap_past_the_drive_limit_fails_naming_the_limit_count_and_cycle():
    system = _system(max_concurrent_drives=6)
    with pytest.raises(ControllerLimitError) as caught:
        _lower(drives_only, system, "asap", system.qudits[:13])
    assert caught.value.violations == (
        "13 drives play at once from cycle 0; the control system plays at most 6 "
        '(max_concurrent_drives): compile with schedule="rcp" or "alap_rcp", which keep to it',
    )
    # Before this check, the same program compiled and controller.check found nothing.
    free = _system()
    unchecked = _lower(drives_only, free, "asap", free.qudits[:13])
    assert controller.check(unchecked) == []


def test_alap_past_the_limits_fails_at_the_cycle_they_are_passed():
    system = _system(max_concurrent_drives=6, max_concurrent_readouts=4)
    with pytest.raises(ControllerLimitError) as caught:
        _lower(drives_then_readouts, system, "alap", system.qudits[:13])
    assert caught.value.violations == (
        "13 drives play at once from cycle 320; the control system plays at most 6 "
        '(max_concurrent_drives): compile with schedule="rcp" or "alap_rcp", which keep to it',
        "13 readouts play at once from cycle 0; the control system plays at most 4 "
        '(max_concurrent_readouts): compile with schedule="rcp" or "alap_rcp", which keep to it',
    )


def test_readouts_past_their_limit_fail_alone():
    system = _system(max_concurrent_readouts=4)
    with pytest.raises(ControllerLimitError) as caught:
        _lower(drives_then_readouts, system, "asap", system.qudits[:5])
    assert caught.value.violations == (
        "5 readouts play at once from cycle 0; the control system plays at most 4 "
        '(max_concurrent_readouts): compile with schedule="rcp" or "alap_rcp", which keep to it',
    )


def test_the_count_names_the_most_at_once_when_it_grows():
    system = _system(max_concurrent_drives=2)

    @pulse.kernel
    def growing(q, r, s, t):
        drive(get_drive_line(q), square(400, 0.1))
        drive(get_drive_line(r), square(400, 0.1))
        drive(get_drive_line(s), square(400, 0.1))
        a = get_drive_line(t)
        wait(a, 100)
        drive(a, square(40, 0.1))

    with pytest.raises(
        ControllerLimitError, match="3 drives play at once from cycle 0, and up to 4;"
    ):
        _lower(growing, system, "asap", *system.qudits[:4])


@pytest.mark.parametrize("schedule", _RCP)
def test_the_rcp_schedules_always_fit(schedule):
    for drives, readouts, count in ((6, 4, 13), (1, 1, 5), (2, 3, 8), (17, 8, 17)):
        system = _system(max_concurrent_drives=drives, max_concurrent_readouts=readouts)
        for kernel in (drives_then_readouts, in_a_loop):
            lowered = _lower(kernel, system, schedule, system.qudits[:count])
            assert controller.check(lowered) == []
    system = _system(max_concurrent_drives=1, max_concurrent_readouts=1)
    for kernel in (arms, arms_and_more):
        assert controller.check(_lower(kernel, system, schedule, *system.qudits[:5])) == []


@pytest.mark.parametrize("schedule", _RCP)
@pytest.mark.parametrize("unroll_threshold", [None, 0, 1])
def test_the_rcp_schedules_fit_with_loops_in_branch_arms(schedule, unroll_threshold):
    # The loop's pulses and the other arm's are in one execution of the
    # branch, in different arms, whatever the loop's iteration.
    for drives in (1, 2):
        system = _system(max_concurrent_drives=drives)
        for kernel in (loop_in_an_arm, branch_in_a_loop):
            compiled = pulse.compile(
                kernel, quantum_system=system, schedule=schedule, unroll_threshold=unroll_threshold
            )
            assert controller.check(compiled.lower(*system.qudits[:4])) == []


def test_a_loop_in_an_arm_counts_with_its_own_arm_only():
    system = _system(max_concurrent_drives=2)
    _lower(loop_in_an_arm, system, "asap", *system.qudits[:4])
    with pytest.raises(ControllerLimitError, match="2 drives play at once from cycle 100;"):
        system = _system(max_concurrent_drives=1)
        _lower(loop_in_an_arm, system, "asap", *system.qudits[:4])


@pytest.mark.parametrize("schedule", _RCP)
def test_the_rcp_schedules_fit_with_line_delays(schedule):
    # A delay moves a line's pulses after they are scheduled; the limits
    # count them where the schedule placed them, as rcp does.
    lines = [DriveLine(q, delay_ns=7.0 * (q % 3)) for q in range(6)]
    readouts = [ReadoutLine(q, delay_ns=3.0 * (q % 2)) for q in range(6)]
    system = _system(
        [pulse.Controller(drive_lines=lines, readout_lines=readouts)],
        max_concurrent_drives=2,
        max_concurrent_readouts=1,
    )
    lowered = _lower(drives_then_readouts, system, schedule, system.qudits[:6])
    assert controller.check(lowered) == []


@pytest.mark.parametrize("schedule", _RCP)
def test_the_rcp_schedules_fit_on_several_controllers(schedule):
    controllers = [
        pulse.Controller(drive_lines=range(0, 8), readout_lines=range(0, 8)),
        pulse.Controller(drive_lines=range(8, 17), readout_lines=range(8, 17)),
    ]
    system = _system(controllers, max_concurrent_drives=3, max_concurrent_readouts=2)
    _lower(drives_then_readouts, system, schedule, system.qudits[:12])
    with pytest.raises(ControllerLimitError, match="12 drives play at once from cycle 0;"):
        _lower(drives_then_readouts, system, "asap", system.qudits[:12])


def test_exactly_at_the_limit_fits():
    system = _system(max_concurrent_drives=6, max_concurrent_readouts=6)
    _lower(drives_then_readouts, system, "asap", system.qudits[:6])
    # A pulse ending at a cycle frees its place for one starting then.
    system = _system(max_concurrent_drives=2)
    _lower(back_to_back, system, "asap", *system.qudits[:3])


def test_pulses_in_different_arms_of_a_branch_never_play_together():
    system = _system(max_concurrent_drives=2)
    # Two drives in each arm, at the same cycles: two at once on either path.
    _lower(arms, system, "asap", *system.qudits[:5])
    # Then one more beside them is three on the path through the first arm.
    with pytest.raises(ControllerLimitError, match="3 drives play at once from cycle 100;"):
        _lower(arms_and_more, system, "asap", *system.qudits[:5])
    system = _system(max_concurrent_drives=3)
    _lower(arms_and_more, system, "asap", *system.qudits[:5])


def test_without_limits_given_nothing_is_checked():
    # The defaults are only the rcp schedules' lanes.
    system = _system()
    assert system.control_system.max_concurrent_drives is None
    lowered = _lower(drives_then_readouts, system, "asap", system.qudits[:13])
    assert controller.check(lowered) == []
    # Nor is a control system without controllers, whatever it is given.
    bare = pulse.ControlSystem(max_concurrent_drives=2, max_concurrent_readouts=1)
    system = pulse.quantum_system("transmon_surface17", control_system=bare)
    _lower(drives_then_readouts, system, "asap", system.qudits[:13])
    # A limit given checks that kind only.
    system = _system(max_concurrent_readouts=13)
    _lower(drives_then_readouts, system, "asap", system.qudits[:13])


def test_a_call_whose_values_place_pulses_past_a_limit_fails_when_made():
    system = _system(max_concurrent_drives=1)
    compiled = pulse.compile(after_a_wait, quantum_system=system, schedule="asap")
    q, r = system.qudits[:2]
    compiled.lower(q, r, 40.0)
    with pytest.raises(ControllerLimitError, match="2 drives play at once from cycle 20;"):
        compiled.lower(q, r, 20.0)


def test_a_limit_not_given_is_none_and_stays_none_in_copies_and_files():
    controllers = [pulse.Controller(drive_lines=[0], readout_lines=[0])]
    given = pulse.ControlSystem(controllers=controllers, max_concurrent_drives=4)
    default = pulse.ControlSystem(controllers=controllers)
    assert (given.max_concurrent_drives, default.max_concurrent_drives) == (4, None)
    assert given != default and repr(given) != repr(default)
    assert given._machine != default._machine
    assert given.as_dict()["max_concurrent_drives"] == 4
    assert default.as_dict()["max_concurrent_drives"] is None
    for control in (given, default):
        assert pickle.loads(pickle.dumps(control)) == control
        system = pulse.quantum_system("transmon_surface17", control_system=control)
        assert pulse.QuantumSystem.from_dict(system.as_dict()).control_system == control
