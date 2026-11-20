# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What tracing a measurement-conditioned if adds to the tracer, imported by
a trace's first such if; a trace without one runs none of it.

While an if is traced its tracer is a twin of its type (``_Journaling``
mixed in) whose ops note in the if's journal (``_Lines.journal``) each
channel they rebind, with its value entering the if: the if restores and
compares only those channels. From the first such if on, each channel has
its place in ``_Lines.held`` (``_Held.seq``; ``NumberedLines`` gives one
to each channel acquired): an if's results follow the channels' order.
Outside an if the tracer is its own type again: its ops note nothing.
"""

from __future__ import annotations

import itertools
import operator
from typing import TYPE_CHECKING, Any

from .._native._cudaq_pulse_native import identity_keys
from ._emitter import PulseIREmitter, StackVal, _Held, _Lines, _slot_holds
from .ir_builder import LINEAR_TYPES, IRValue

if TYPE_CHECKING:
    _Tracer = PulseIREmitter
else:
    _Tracer = object


class NumberedLines(_Lines):
    """``_Lines`` from a trace's first measurement-conditioned if on."""

    __slots__ = ()

    def add(self, key: tuple[Any, ...], values: tuple[IRValue, ...]) -> None:
        # Only a channel not held is added, and ``held`` only ever loses its
        # newest entries: its length is the channel's place.
        held: Any = _Held(values, key)
        held.seq = len(self.held)
        self.held[key] = held


class _Journaling(_Tracer):
    """Mixed into a tracer type for the tracer of a measurement-conditioned
    if's arms: its ops note the channels they rebind in the journal."""

    __slots__ = ()

    # Where a cell is among the channels: its channel's place, then its own.
    _CELL_ORDER = operator.attrgetter("held.seq", "index")

    def _rebind_linear(
        self,
        fname: str,
        call_args: list[StackVal],
        results: tuple[IRValue, ...],
    ) -> None:
        """``PulseIREmitter._rebind_linear``, noting each channel it moves
        in the journal."""
        slots = self._slots
        names: dict[int, str | None] | None = None
        if fname == "sync" and len(call_args) == 2 and call_args[0] is not call_args[1]:
            # Two distinct lines: their names are found in one pass.
            first, second = call_args
            if first.chan is not None and second.chan is not None and len(results) == 2:
                # Both are claimable, so each claims its own result.
                self._rebind_pair(first, results[0], second, results[1])
                return
            found = self._local_names(first, second)
            names = {id(first): found[0], id(second): found[1]}
        # The operands each result may replace, by type, the first in operand
        # order last: a sync over n lines rebinds in O(n), not O(n^2).
        unclaimed: dict[str, list[tuple[str | None, IRValue, Any, Any]]] = {}
        for a in reversed(call_args):
            if a.__class__ is IRValue and a.vtype in LINEAR_TYPES:
                slot = slots.get(id(a))
                if slot is not None and not _slot_holds(slot, a):
                    slot = None
                name = self._find_local_name(a) if names is None else names[id(a)]
                if name is not None or slot is not None or a.chan is not None:
                    if a.vtype in unclaimed:
                        unclaimed[a.vtype].append((name, a, slot, a.chan))
                    else:
                        unclaimed[a.vtype] = [(name, a, slot, a.chan)]

        for res in results:
            if res.vtype not in unclaimed or not unclaimed[res.vtype]:
                continue
            name, op_val, slot, chan = unclaimed[res.vtype].pop()
            if chan is not None and chan.value is op_val:
                journal: Any = self._lines.journal
                if chan not in journal:
                    journal[chan] = op_val
                chan.value = res
                res.chan = chan
                if chan.shared:
                    for other, value in self.locals.items():
                        if value is op_val:
                            self.locals[other] = res
            if name is not None:
                self.locals[name] = res
            if slot is not None:
                slot[0][slot[1]] = res
                slots.pop(id(op_val), None)
                slots[id(res)] = (slot[0], slot[1], res)

    def _restore_container(self, obj: Any, copy: Any) -> None:
        """``PulseIREmitter._restore_container``; a list of more than 8
        lines and tones, as an if restores, has its slots tracked in one
        step."""
        if isinstance(obj, _Held):
            for i, value in enumerate(copy):
                obj[i] = value
        elif isinstance(obj, list):
            obj[:] = copy
            if (
                len(copy) > 8
                and obj.__class__ is list
                and set(map(type, copy)) == {IRValue}
                and LINEAR_TYPES.issuperset(map(operator.attrgetter("vtype"), copy))
            ):
                # Lines or tones only, as a measurement-conditioned if restores
                # a list of them: every slot at once. A subclass is read by
                # index (its own len() and indexing run).
                entries = zip(itertools.repeat(obj), itertools.count(), copy, strict=False)
                self._slots.update(zip(map(id, copy), entries, strict=False))
            else:
                self._track_slots(obj, 0)
        elif isinstance(obj, (dict, set)):
            obj.clear()
            obj.update(copy)
        else:
            obj.clear()
            obj.extend(copy)

    def _rebind_pair(self, old: IRValue, new: IRValue, old2: IRValue, new2: IRValue) -> None:
        """``PulseIREmitter._rebind_pair``, noting the channels it moves in
        the journal."""
        chan = old.chan
        chan2 = old2.chan
        # Noted for _rebind_one too, which only this calls.
        journal: Any = self._lines.journal
        if chan is not None and chan not in journal:
            journal[chan] = chan.value
        if chan2 is not None and chan2 not in journal:
            journal[chan2] = chan2.value
        if (chan is not None and chan.shared) or (chan2 is not None and chan2.shared):
            self._rebind_one(old, new)
            self._rebind_one(old2, new2)
            return
        name, name2 = identity_keys(self.locals, old, old2)
        if chan is not None:
            chan.value = new
            new.chan = chan
        if name is not None:
            self.locals[name] = new
        self._rebind_slot(old, new)
        if chan2 is not None:
            chan2.value = new2
            new2.chan = chan2
        if name2 is not None:
            self.locals[name2] = new2
        self._rebind_slot(old2, new2)


class _Twins(dict[type[PulseIREmitter], type[PulseIREmitter]]):
    """Each tracer type's twin, made when first asked for."""

    def __missing__(self, tracer: type[PulseIREmitter]) -> type[PulseIREmitter]:
        twin = type(f"Journaling{tracer.__name__}", (_Journaling, tracer), {"__slots__": ()})
        self[tracer] = twin
        return twin


TWINS = _Twins()
