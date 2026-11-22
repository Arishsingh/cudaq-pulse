# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a call does in Python, counted rather than timed.

The lines a call runs and the calls it makes, counted with sys.monitoring,
are the same on any machine and at any load, so a test that compares two
sizes of one program can require them equal, or in proportion, where their
times are only close. Work done in native code is not counted: a test that
bounds it as well times it, in CPU time and the best of several: all of a
call (cpu_seconds) or the native methods it calls (native_seconds); a test
that compares sizes or variants times them in turn (cpu_seconds_each,
native_seconds_each), so that a slow spell of a shared machine stretches all.
"""

from __future__ import annotations

import gc
import sys
import time
from collections.abc import Callable, Hashable

import pytest
from cudaq_pulse._native._cudaq_pulse_native import PulseModule


def python_work(fn: Callable[[], object]) -> dict[str, int]:
    """The lines run and the calls made by a call of *fn* after a first,
    uncounted one; the garbage collector is held off while counting."""
    fn()
    monitoring = sys.monitoring
    tool = next((i for i in range(6) if monitoring.get_tool(i) is None), None)
    if tool is None:
        pytest.skip("every sys.monitoring tool id is in use")
    work = {"lines": 0, "calls": 0}

    def line(*_: object) -> None:
        work["lines"] += 1

    def call(*_: object) -> None:
        work["calls"] += 1

    events = monitoring.events
    monitoring.use_tool_id(tool, "cudaq-pulse tests")
    collecting = gc.isenabled()
    try:
        monitoring.register_callback(tool, events.LINE, line)
        monitoring.register_callback(tool, events.CALL, call)
        gc.collect()
        gc.disable()
        monitoring.set_events(tool, events.LINE | events.CALL)
        fn()
    finally:
        monitoring.set_events(tool, events.NO_EVENTS)
        if collecting:
            gc.enable()
        monitoring.register_callback(tool, events.LINE, None)
        monitoring.register_callback(tool, events.CALL, None)
        monitoring.free_tool_id(tool)
    return work


def _least[K: Hashable](runs: dict[K, Callable[[], float]], repeats: int) -> dict[K, float]:
    """The least time each of *runs* reports, a run being a call that returns
    the time it spent, after a first, uncounted call of each. The runs are
    taken in turn, so that a slow spell of a shared machine falls on all of
    them alike rather than on whichever was being timed: *repeats* rounds,
    and more until they have taken half a second per run, as a short call is
    the likelier to have shared its core with other work in each of a few.
    The garbage collector is held off for each call."""
    for run in runs.values():
        run()
    best = dict.fromkeys(runs, float("inf"))
    collecting = gc.isenabled()
    deadline = time.perf_counter() + 0.5 * len(runs)
    try:
        for done in range(100):
            if done >= repeats and time.perf_counter() > deadline:
                break
            for key, run in runs.items():
                gc.collect()
                gc.disable()
                best[key] = min(best[key], run())
                if collecting:
                    gc.enable()
    finally:
        if collecting:
            gc.enable()
    assert all(0 < b < float("inf") for b in best.values()), best
    return best


def cpu_seconds_each[K: Hashable](
    fns: dict[K, Callable[[], object]], repeats: int = 5
) -> dict[K, float]:
    """The least CPU time a call of each of *fns* takes, of *repeats* or more,
    the calls taken in turn (see _least): what a test that compares sizes or
    variants of a program needs, as each is timed under the same load.

    CPU time leaves out the time a loaded machine spends running something
    else, which stretches a wall-clock time the more the longer it is."""

    def timed(fn: Callable[[], object]) -> Callable[[], float]:
        def run() -> float:
            started = time.process_time()
            fn()
            return time.process_time() - started

        return run

    return _least({key: timed(fn) for key, fn in fns.items()}, repeats)


def cpu_seconds(fn: Callable[[], object], repeats: int = 5) -> float:
    """The least CPU time a call of *fn* takes, of *repeats* or more."""
    return cpu_seconds_each({0: fn}, repeats)[0]


def native_seconds_each[K: Hashable](
    fns: dict[K, Callable[[], object]],
    methods: tuple[str, ...] = ("schedule",),
    repeats: int = 5,
) -> dict[K, float]:
    """The least CPU time, of *repeats* or more calls of each of *fns*, taken in
    turn (see _least), that its calls of the PulseModule *methods* take:
    without the tracing around them, which python_work counts, a cost in
    them that grows faster than the program is not diluted."""
    spent = [0.0]
    originals = {name: getattr(PulseModule, name) for name in methods}

    def timed(method: Callable[..., object]) -> Callable[..., object]:
        def call(*args: object, **kwargs: object) -> object:
            started = time.process_time()
            try:
                return method(*args, **kwargs)
            finally:
                spent[0] += time.process_time() - started

        return call

    def counted(fn: Callable[[], object]) -> Callable[[], float]:
        def run() -> float:
            spent[0] = 0.0
            fn()
            return spent[0]

        return run

    try:
        for name, method in originals.items():
            setattr(PulseModule, name, timed(method))
        return _least({key: counted(fn) for key, fn in fns.items()}, repeats)
    finally:
        for name, method in originals.items():
            setattr(PulseModule, name, method)


def native_seconds(
    fn: Callable[[], object], methods: tuple[str, ...] = ("schedule",), repeats: int = 5
) -> float:
    """The least CPU time, of *repeats* or more calls of *fn*, that its calls of
    the PulseModule *methods* take (see native_seconds_each)."""
    return native_seconds_each({0: fn}, methods, repeats)[0]
