# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A measured if's arms stop where they join: the trace writes that end into
its own list of the function's instructions while the arms are traced and
takes it out after, also when an arm raises or a loop being rolled is traced
again. Each list is left as the function's instructions were, and the
decoded code other traces share is never written.
"""

from __future__ import annotations

from types import SimpleNamespace

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.kernel import _bytecode_normalize as bc
from cudaq_pulse.kernel.ir_builder import CompilationError

from tests._handles import Q


@pytest.fixture
def traced(monkeypatch):
    """The tables of trace-owned lists (one per trace) and the join of each
    region written, in order."""
    seen = SimpleNamespace(tables=[], joins=[])
    real = bc.region

    def region(decoded, join, owned):
        if not any(owned is t for t in seen.tables):
            seen.tables.append(owned)
        seen.joins.append(join)
        return real(decoded, join, owned)

    monkeypatch.setattr(bc, "region", region)
    return seen


def _assert_clean(seen):
    assert seen.tables
    for owned in seen.tables:
        for instrs, own, open_regions, _ in owned.values():
            assert open_regions == 0
            assert type(instrs) is tuple
            assert tuple(own) == instrs


def _kernel(src):
    scope = dict(vars(pulse))
    exec(compile(src, "<kernel>", "exec"), scope)
    return pulse.kernel(scope["k"])


_NESTED = """\
def k(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    m = readout(get_readout_line(b), square(40, 0.05))
    if m:
        drive(da, square(16, 0.1))
        if m:
            drive(db, square({width}, 0.1))
        else:
            wait(db, 8)
        drive(da, square(16, 0.2))
    drive(db, square(16, 0.3))
"""


def test_an_arm_that_raises_leaves_the_instructions_as_they_were(traced):
    with pytest.raises(CompilationError):
        pulse.compile(_kernel(_NESTED.format(width=-1))).lower(Q[0], Q[1])
    _assert_clean(traced)


def test_nested_ifs_take_their_ends_out_in_turn(traced):
    lowered = pulse.compile(_kernel(_NESTED.format(width=8))).lower(Q[0], Q[1])
    _assert_clean(traced)
    assert len(traced.joins) == 2
    assert lowered.mlir().count("scf.if") == 2


_HELPER = """\
def h(d, m, n):
    if m:
        if n > 0:
            h(d, m, n - 1)
        drive(d, square(8, 0.1))
    else:
        wait(d, 8)
    drive(d, square(4, 0.2))


def k(a, b):
    da = get_drive_line(a)
    m = readout(get_readout_line(b), square(40, 0.05))
    h(da, m, 2)
    drive(da, square(16, 0.3))
"""


def test_a_helper_traced_again_inside_its_own_arm_gets_its_own_list(traced):
    lowered = pulse.compile(_kernel(_HELPER)).lower(Q[0], Q[1])
    _assert_clean(traced)
    # Three calls deep, each with its if: the inner calls' arms run to their
    # own joins, not to the end an outer call wrote, in lists of their own.
    assert lowered.mlir().count("scf.if") == 3
    assert len({id(entry[1]) for owned in traced.tables for entry in owned.values()}) == 3


def test_a_function_entered_again_takes_a_list_whose_ends_are_out():
    def f(x):
        return x

    d = bc.decode(f.__code__)
    owned: dict = {}
    first = bc.region(d, 0, owned)
    second = bc.region(d, 0, owned)  # entered again while the first is open
    assert second.instrs is not first.instrs
    bc.end_region(second)
    third = bc.region(d, 0, owned)  # the second list holds no end: taken again
    assert third.instrs is second.instrs
    # With the first two lists holding ends, a third is made after them.
    fourth = bc.region(d, 0, owned)
    assert fourth.instrs is not first.instrs and fourth.instrs is not third.instrs
    bc.end_region(fourth)
    bc.end_region(third)
    bc.end_region(first)
    assert len({id(entry[1]) for entry in owned.values()}) == 3
    assert all(tuple(entry[1]) == d.instrs for entry in owned.values())


def test_a_region_made_in_a_region_ends_at_both_joins():
    def f(m, n):
        a = 0
        if m:
            if n:
                a = 1
            a += 2
        return a

    d = bc.decode(f.__code__)
    branches = [i for i, (op, _) in enumerate(d.instrs) if op is bc.JUMP_IF_FALSE]
    owned: dict = {}
    outer = bc.region(d, bc.join_point(d, branches[0]), owned)
    inner = bc.region(outer, bc.join_point(outer, branches[1]), owned)
    assert inner.instrs is outer.instrs
    assert outer.ends() == {outer.join} and inner.ends() == {outer.join, inner.join}
    bc.end_region(inner)
    bc.end_region(outer)
    assert tuple(inner.instrs) == d.instrs


_LOOP = """\
def k(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    r = get_readout_line(b)
    x = da
    for i in range(6):
        m = readout(r, square(40, 0.05))
        if m:
            if m:
                drive(x, square(8, 0.1))
            wait(da, 8)
        {stmt}
        drive(db, square(8, 0.2))
"""


def test_a_loop_traced_again_after_its_arms_finds_their_ends_taken_out(traced):
    rolled = pulse.compile(_kernel(_LOOP.format(stmt="pass"))).lower(Q[0], Q[1])
    _assert_clean(traced)
    assert len(traced.joins) == 2
    assert rolled.mlir().count("scf.if") == 2
    # A body that rebinds a line its ifs read is traced to be rolled, traced
    # again with that name unbound, then once for each of its six
    # iterations: its ifs' ends are written each time, into a list the
    # earlier traces left as it was.
    del traced.joins[:]
    lowered = pulse.compile(_kernel(_LOOP.format(stmt="x = db"))).lower(Q[0], Q[1])
    _assert_clean(traced)
    assert lowered.mlir().count("scf.if") == 12
    assert len(traced.joins) == 2 + 2 + 12


_UNROLLED_IN_AN_ARM = """\
def k(a, b):
    da = get_drive_line(a)
    m = readout(get_readout_line(b), square(40, 0.05))
    if m:
        for w in [8] * 10**5:
            if m:
                drive(da, square(w, 0.1))
            else:
                wait(da, 8)
    drive(da, square(16, 0.3))
"""


def test_the_instruction_limit_names_a_loop_unrolled_in_an_arm_around_an_if():
    # The inner if's arms join at the loop's header: its end, written while
    # its arms are traced, is not the loop's for line.
    limited = ControlSystem(controllers=[Controller(max_instructions=4096)])
    with pytest.raises(CompilationError) as info:
        pulse.compile(_kernel(_UNROLLED_IN_AN_ARM), control_system=limited).lower(Q[0], Q[1])
    assert str(info.value).startswith(
        "for loop over a Python iterable unrolls to about 600,733 operations "
        "(100,000 iterations of about 6), more than the 4,096"
    )
    assert 'File "<kernel>", line 5, in k' in "\n".join(info.value.__notes__)
