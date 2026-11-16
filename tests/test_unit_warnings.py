# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""With unit_warnings=True, a time without units and a frequency far below
the carriers each get a UnitWarning at their kernel line; without it,
nothing is checked, and the program is the same either way."""

from __future__ import annotations

import inspect
import warnings

import cudaq_pulse as pulse
import pytest

from tests._handles import Q


def _warned(kernel, *args, **options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lowered = pulse.compile(kernel, **options).lower(*args)
    return lowered, [w for w in caught if issubclass(w.category, pulse.UnitWarning)]


def _line_of(fn, text):
    lines, first = inspect.getsourcelines(fn)
    return first + next(i for i, line in enumerate(lines) if text in line)


@pulse.kernel
def _bare(q):
    d = get_drive_line(q)
    drive(d, square(20, 0.1))
    wait(d, 100)
    drive(d, drag(16 * pulse.ns, 0.6, 2 * pulse.ns, 0.3))
    drive(d, gaussian(duration=40 * pulse.ns, amplitude=0.1, sigma=10.0))
    drive(d, gaussian_square(40 * pulse.ns, 0.1, 2 * pulse.ns, 20))
    shift_frequency(d.tone, 5)
    set_frequency(d.tone, 5000.1)


def test_each_time_without_units_and_each_tiny_frequency_is_warned_at_its_line():
    _, caught = _warned(_bare, Q[0], unit_warnings=True)
    fn = _bare.__wrapped__
    assert [(w.filename, w.lineno) for w in caught] == [
        (__file__, _line_of(fn, "square(20")),
        (__file__, _line_of(fn, "wait(d, 100)")),
        (__file__, _line_of(fn, "drag(")),
        (__file__, _line_of(fn, "sigma=10.0")),
        (__file__, _line_of(fn, "gaussian_square(")),
        (__file__, _line_of(fn, "shift_frequency(")),
        (__file__, _line_of(fn, "set_frequency(")),
    ]
    messages = [str(w.message) for w in caught]
    assert messages[0] == (
        "square(): duration 20 has no units, so it is 20 clock cycles (10 ns at the 2 GHz "
        "clock); write it with units, as 20 * pulse.ns"
    )
    assert messages[1].startswith("wait(): duration 100 has no units, so it is 100 clock cycles")
    assert messages[2].startswith("drag(): beta 0.3 has no units")
    assert messages[3].startswith("gaussian(): sigma 10.0 has no units")
    assert messages[4].startswith("gaussian_square(): width 20 has no units")
    assert messages[5] == (
        "shift_frequency(): 5 is a shift of 5 Hz, far below the quantum system's carriers "
        "(the lowest is 5 GHz), so it is probably not in Hz; frequencies are in Hz, as 5e6 "
        "for 5 MHz"
    )
    assert messages[6].startswith("set_frequency(): 5000.1 is a frequency of 5.0001 kHz")


def test_nothing_is_checked_by_default():
    _, caught = _warned(_bare, Q[0])
    assert caught == []


def test_times_with_units_zero_rounded_cycles_and_calibrations_are_not_warned():
    qs = pulse.quantum_system("transmon_surface17")

    @pulse.kernel
    def k(q, t: float):
        d = get_drive_line(q)
        drive(d, square(20 * pulse.ns, 0.1))
        wait(d, 0)
        wait(d, pulse.round_duration(d, 10.3 * pulse.ns))
        wait(d, t)
        drive(d, d.get_calibration(pulse.gates.X))
        shift_frequency(d.tone, 5e6)
        shift_frequency(d.tone, 0)
        set_frequency(d.tone, 6.0e9)

    _, caught = _warned(k, qs.qudits[0], 8.0, unit_warnings=True)
    assert caught == []


def test_a_helper_is_warned_at_its_own_line():
    _, caught = _warned(_calls_helper, Q[0], unit_warnings=True)
    assert [(w.filename, w.lineno) for w in caught] == [(__file__, _line_of(_helper, "wait(d"))]


def _helper(d):
    wait(d, 40)


@pulse.kernel
def _calls_helper(q):
    _helper(get_drive_line(q))


def test_a_compilation_with_them_is_traced_apart_from_one_without():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        wait(d, 100)

    for unit_warnings, count in ((False, 0), (True, 1), (False, 0), (True, 1)):
        _, caught = _warned(k, Q[0], unit_warnings=unit_warnings)
        assert len(caught) == count
    with warnings.catch_warnings(record=True) as called:
        warnings.simplefilter("always")
        k.lower(Q[0])
    assert called == []


def test_an_error_filter_makes_them_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error", pulse.UnitWarning)
        with pytest.raises(pulse.UnitWarning, match=r"^square\(\): duration 20 has no units"):
            pulse.compile(_bare, unit_warnings=True).lower(Q[0])


def test_the_program_is_the_same_with_and_without_them():
    plain, _ = _warned(_bare, Q[0])
    checked, _ = _warned(_bare, Q[0], unit_warnings=True)
    assert checked.mlir() == plain.mlir()
