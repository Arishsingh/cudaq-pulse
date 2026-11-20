# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drives and readouts a program plays at once, against the
``max_concurrent_drives`` and ``max_concurrent_readouts`` its control system
gives.

They are counted as the ``rcp`` and ``alap_rcp`` schedules count them, so a
program those schedules place fits: on every controller together, in the
cycles the schedule placed each pulse at (before its line's delay moves
it), a pulse holding its place from its start up to its end, where the next
may start. A wait takes no place. Pulses in different arms of one execution
of a branch never play together.

A long rolled loop is counted from the check walk's summary, as the other
controller checks count it (``_loops``): a loop whose copies follow each
other, with no other pulse of the kind meanwhile, from its first copy; any
other copy by copy, up to the copies a check plays one by one, past which
every iteration is walked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .compiler import ScheduledKernel

_KINDS = (("drive", "max_concurrent_drives"), ("readout", "max_concurrent_readouts"))

# A pulse being counted: (start, end, event, the copies of loops it is in).
_Played = tuple[int, int, Mapping[str, Any], tuple[Any, ...]]


def violations(lowered: ScheduledKernel) -> tuple[list[str], bool]:
    """What *lowered* plays at once past the limits its control system
    gives, and whether that is exact: ``False`` when a pulse's time depends
    on values not known yet, which each call's values then settle."""
    machine = lowered.options.machine
    limits = {
        kind: getattr(machine, name) for kind, name in _KINDS if getattr(machine, name) is not None
    }
    if not limits:
        return [], True
    from .controller import _walk

    shifts: Mapping[tuple[str, int], int] = {}
    if machine._line_delays:
        from .compiler import _line_delays

        clocks = (lowered._generic or lowered).clocks
        shifts = _line_delays(machine, lowered.quantum_system, clocks)[0]
    # Loops whose lines move by periods of their own summarized too, then
    # only loops of one period, as the other checks walk them; then none.
    passes: tuple[list[tuple[int, int, int]] | None, ...] = ([], None)
    for periods in passes:
        repeats: list[tuple[int, ...]] = []
        summary = (repeats, periods)
        lines, events, _, _, _ = _walk(lowered, values=False, waveforms=False, summary=summary)
        # Each pulse of a kind with a limit: its start and end before its line's delay.
        timed: dict[int, tuple[int, int]] = {}
        for index, event in enumerate(events):
            if event["kind"] not in limits:
                continue
            start, end = event["start_cycles"], event["end_cycles"]
            if start is None or end is None:
                return [], False
            if shifts:
                line = lines[event["line"]]
                shift = shifts.get((line["role"], line["qudit"]), 0)
                start, end = start - shift, end - shift
            timed[index] = (start, end)
        counts = {kind: _AtOnce(limit) for kind, limit in limits.items()}
        if not repeats:
            for kind, count in counts.items():
                count.sweep(
                    sorted(
                        (
                            (*timed[i], events[i], ())
                            for i in range(len(events))
                            if i in timed and events[i]["kind"] == kind
                        ),
                        key=_first,
                    )
                )
            break
        from . import _loops

        tree, chains = _loops._tree(events, repeats, periods or ())
        kinds = {i: events[i]["kind"] for i in timed}
        split = _loops._split(tree, kinds.get)
        try:
            for kind, count in counts.items():
                nodes = split.get(kind, [])
                count.enclosing = _enclosing(events, chains, nodes)
                _count(count, timed, events, nodes, 0, (), _loops._Budget())
        except _loops.UnsyncedError:
            continue
        except _PastBudgetError:
            periods = None
        else:
            break
        # Past the copies a check plays one by one: every iteration.
        lines, events, _, _, _ = _walk(lowered, values=False, waveforms=False)
        return violations_of(lowered, lines, events, limits, shifts), True
    else:
        lines, events, _, _, _ = _walk(lowered, values=False, waveforms=False)
        return violations_of(lowered, lines, events, limits, shifts), True
    return [m for kind, name in _KINDS if (m := _message(counts.get(kind), kind, name))], True


def violations_of(
    lowered: ScheduledKernel,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    limits: Mapping[str, int],
    shifts: Mapping[tuple[str, int], int],
) -> list[str]:
    """``violations`` of every iteration's *events*, all times known."""
    found = []
    for kind, name in _KINDS:
        if kind not in limits:
            continue
        played = []
        for event in events:
            if event["kind"] != kind:
                continue
            line = lines[event["line"]]
            shift = shifts.get((line["role"], line["qudit"]), 0) if shifts else 0
            played.append((event["start_cycles"] - shift, event["end_cycles"] - shift, event, ()))
        count = _AtOnce(limits[kind])
        count.sweep(sorted(played, key=_first))
        message = _message(count, kind, name)
        if message is not None:
            found.append(message)
    return found


class _PastBudgetError(Exception):
    """More copies of rolled loops to count one by one than a check plays."""


def _first(item: _Played) -> tuple[int, int]:
    return item[0], item[1]


def _message(count: _AtOnce | None, kind: str, name: str) -> str | None:
    if count is None or count.first is None:
        return None
    beyond = f", and up to {count.most}" if count.most > count.at_first else ""
    return (
        f"{count.at_first} {kind}s play at once from cycle {count.first}{beyond}; the control "
        f'system plays at most {count.limit} ({name}): compile with schedule="rcp" or '
        '"alap_rcp", which keep to it'
    )


class _AtOnce:
    """The pulses of one kind at once past *limit*: the first cycle they
    are more, how many then, and the most at once."""

    __slots__ = ("limit", "first", "at_first", "most", "enclosing")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.first: int | None = None
        self.at_first = 0
        self.most = 0
        # By branch id, the rolled loops (by id) its executions repeat in.
        self.enclosing: Mapping[int, frozenset[int]] = {}

    def sweep(self, played: Sequence[_Played]) -> None:
        """Count *played*, sorted by start; a pulse ending at a cycle frees
        its place for one starting then."""
        active: list[_Played] = []
        for item in played:
            start = item[0]
            active = [a for a in active if a[1] > start]
            active.append(item)
            if len(active) <= self.limit:
                continue
            count = self._together(active)
            if count <= self.limit:
                continue
            if self.first is None or start < self.first:
                self.first, self.at_first = start, count
            elif start == self.first:
                self.at_first = max(self.at_first, count)
            self.most = max(self.most, count)

    def _arms(self, item: _Played) -> dict[tuple[Any, ...], Any]:
        """The arm of each branch execution the pulse of *item* is in. The
        walk numbers each execution of a branch; the copies of a summarized
        loop around the branch share its numbers, told apart by the copy."""
        arms = {}
        for condition in item[2]["conditions"]:
            branch = condition["branch"]
            loops = self.enclosing.get(branch, frozenset())
            copy = tuple(c for c in item[3] if (c[0] if type(c) is tuple else c) in loops)
            arms[(branch, copy)] = condition["arm"]
        return arms

    def _together(self, active: Sequence[_Played]) -> int:
        """The most of *active* that play together: those of one path through
        the branches, no two in different arms of one branch execution."""
        count = 0
        keyed = []
        for item in active:
            if item[2]["conditions"]:
                keyed.append(self._arms(item))
            else:
                count += 1
        # Groups that share no branch execution take their best path each.
        while keyed:
            group = [keyed.pop()]
            keys = set(group[0])
            grown = True
            while grown:
                grown = False
                for arms in keyed[:]:
                    if keys.isdisjoint(arms):
                        continue
                    keyed.remove(arms)
                    group.append(arms)
                    keys.update(arms)
                    grown = True
            count += _best(group, {})
        return count


def _best(
    group: Sequence[Mapping[tuple[Any, ...], Any]], chosen: dict[tuple[Any, ...], Any]
) -> int:
    """The most of *group* (each the arms it is in) that one choice of arms,
    extending *chosen*, plays together."""
    open_key = None
    for arms in group:
        open_key = next((key for key in arms if key not in chosen), None)
        if open_key is not None:
            break
    if open_key is None:
        return len(group)
    best = 0
    for arm in {arms[open_key] for arms in group if open_key in arms}:
        chosen[open_key] = arm
        best = max(best, _best([a for a in group if a.get(open_key, arm) == arm], chosen))
        del chosen[open_key]
    return best


def _enclosing(
    events: Sequence[Mapping[str, Any]], chains: Sequence[tuple[Any, ...]], nodes: list[Any]
) -> dict[int, frozenset[int]]:
    """By branch id, the summarized loops around the branch: those around
    every pulse of *nodes* in an arm of it. A loop inside an arm repeats
    the pulses of one execution of the branch."""
    from ._loops import _events_of

    found: dict[int, frozenset[int]] = {}
    for index in _events_of(nodes):
        loops = frozenset(id(rep.orig) for rep in chains[index])
        for condition in events[index]["conditions"]:
            branch = condition["branch"]
            had = found.get(branch)
            found[branch] = loops if had is None else had & loops
    return found


def _period(node: Any, events: Sequence[Mapping[str, Any]]) -> int:
    """The cycles between copies of *node* (a ``_loops._Rep``) on the lines
    of its pulses; ``_loops.UnsyncedError`` when they move by different
    periods."""
    if node.lines is None:
        return int(node.period)
    from . import _loops

    from_lines = _loops._period(node, {events[i]["line"] for i in _loops._events_of(node.body)})
    if from_lines <= 0:
        raise _loops.UnsyncedError
    return from_lines


def _window(
    node: Any, timed: Mapping[int, tuple[int, int]], events: Sequence[Mapping[str, Any]]
) -> tuple[int, int]:
    """The first start and last end of *node*'s pulses, all its copies."""
    if type(node) is int:
        return timed[node]
    spans = [_window(child, timed, events) for child in node.body]
    lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
    return lo, hi + (node.copies - 1) * _period(node, events)


def _count(
    at_once: _AtOnce,
    timed: Mapping[int, tuple[int, int]],
    events: Sequence[Mapping[str, Any]],
    nodes: list[Any],
    shift: int,
    copy: tuple[Any, ...],
    budget: Any,
) -> None:
    """Sweep *nodes* (``_loops._tree`` nodes of one kind, shifted by *shift*
    cycles, in the *copy* of the loops around them), as the shared-line
    check sweeps its tones (``_shared_lines._count_copies``)."""
    items: list[tuple[Any, int, tuple[Any, ...]]] = [(node, shift, copy) for node in nodes]
    while True:
        windows = [_window(node, timed, events) for node, _, _ in items]
        expand = None
        for k, (node, dt, _) in enumerate(items):
            if type(node) is int:
                continue
            body = [_window(child, timed, events) for child in node.body]
            lo, hi = windows[k]
            follows = max(s[1] for s in body) - min(s[0] for s in body) <= _period(node, events)
            alone = all(
                w[1] + items[j][1] <= lo + dt or w[0] + items[j][1] >= hi + dt
                for j, w in enumerate(windows)
                if j != k
            )
            if not (follows and alone):
                expand = k
                break
        if expand is None:
            break
        node, dt, path = items[expand]
        try:
            budget.spend(node.copies)
        except ValueError:
            raise _PastBudgetError from None
        period = _period(node, events)
        items[expand : expand + 1] = [
            (child, dt + j * period, (*path, (id(node.orig), j)))
            for j in range(node.copies)
            for child in node.body
        ]
    played = []
    for node, dt, path in items:
        if type(node) is int:
            start, end = timed[node]
            played.append((start + dt, end + dt, events[node], path))
        else:
            # Its copies follow each other with no other pulse of the kind
            # meanwhile: each plays as the first does.
            _count(at_once, timed, events, node.body, dt, (*path, id(node.orig)), budget)
    at_once.sweep(sorted(played, key=_first))
