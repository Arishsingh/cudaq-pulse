# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a call does in Python, counted rather than timed.

The lines a call runs and the calls it makes, counted with sys.monitoring,
are the same on any machine and at any load, so a test that compares two
sizes of one program can require them equal, or in proportion, where their
times are only close. Work done in native code is not counted: a test that
bounds it as well times it, in CPU time and the best of several: all of a
call (cpu_seconds) or the native methods it calls (native_seconds).
"""

from __future__ import annotations

import gc
import sys
import time
from collections.abc import Callable

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


def _least(fn: Callable[[], object], spent: Callable[[], float], repeats: int) -> float:
    """The least of *spent*() after each call of *fn*, which follow a first,
    uncounted one: *repeats* calls, and more until they have taken half a
    second, as a short call is the likelier to have shared its core with
    other work in each of a few. The garbage collector is held off for each
    call."""
    fn()
    best = float("inf")
    collecting = gc.isenabled()
    deadline = time.perf_counter() + 0.5
    try:
        for done in range(100):
            if done >= repeats and time.perf_counter() > deadline:
                break
            gc.collect()
            gc.disable()
            fn()
            best = min(best, spent())
            if collecting:
                gc.enable()
    finally:
        if collecting:
            gc.enable()
    assert 0 < best < float("inf"), best
    return best


def cpu_seconds(fn: Callable[[], object], repeats: int = 5) -> float:
    """The least CPU time a call of *fn* takes, of *repeats* or more.

    CPU time leaves out the time a loaded machine spends running something
    else, which stretches a wall-clock time the more the longer it is."""
    started = [0.0]

    def run() -> None:
        started[0] = time.process_time()
        fn()

    return _least(run, lambda: time.process_time() - started[0], repeats)


def native_seconds(
    fn: Callable[[], object], methods: tuple[str, ...] = ("schedule",), repeats: int = 5
) -> float:
    """The least CPU time, of *repeats* or more calls of *fn*, that its calls of the
    PulseModule *methods* take: without the tracing around them, which
    python_work counts, a cost in them that grows faster than the program
    is not diluted."""
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

    def run() -> None:
        spent[0] = 0.0
        fn()

    try:
        for name, method in originals.items():
            setattr(PulseModule, name, timed(method))
        return _least(run, lambda: spent[0], repeats)
    finally:
        for name, method in originals.items():
            setattr(PulseModule, name, method)
