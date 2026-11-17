# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spacing the updates of update buses (``Controller.space_updates``): the
scheduled module scheduled again with the update that clashes first held
back on its line, as little as it must, until no update clashes. The
scheduler moves what follows it and keeps every other constraint; the
controller checks then run on the result. Only a compile for a controller
that asks for spacing loads this."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .controller import _AMPLITUDES, _PARAMETERS, _bus_clashes
from .kernel.ir_builder import CompilationError

if TYPE_CHECKING:
    from ._device import QuantumSystem
    from ._system import Controller

# The reschedules spacing makes at most before it gives up.
MAX_ROUNDS = 256


def spaced(
    module: Any,
    controllers: Sequence[Controller],
    target: QuantumSystem,
    schedule_args: Sequence[Any],
) -> Any:
    """*module*, scheduled, with the updates on the buses of *controllers*
    spaced: each round schedules its module before scheduling again with
    each op held back as it says (``PulseModule.with_holds``), finds the
    earliest update that starts less than its bus's ``cycles_per_update``
    after the one before it, and holds that op back by the difference, in
    program cycles; an op in a loop is held back in every iteration. A hold
    moves what follows the op on its line, through syncs as the scheduler
    places them, and can change which later pulses are updates (a retuned
    tone gains phase over the delay), so each round checks the whole
    program again. *module* itself when nothing clashes; a
    ``CompilationError`` when an update cannot move or the rounds do not
    settle."""
    clock = target.clock_ghz
    # Each bus's cycles in program cycles, rounded up.
    cycles = [
        [
            math.ceil(bus.cycles_per_update * clock / (c.clock_ghz or clock) - 1e-9)
            for bus in c.update_buses
        ]
        for c in controllers
    ]
    holds: dict[int, int] = {}
    last: tuple[int, int] | None = None
    for _ in range(MAX_ROUNDS):
        # Scheduled again from the same module, with its ops numbered.
        current = module.with_holds(holds, *schedule_args)
        ids: list[int] = []
        lines, events, _ = current.controller_program(
            _PARAMETERS, _AMPLITUDES, False, exportable=False, ids=ids
        )
        first = None
        for controller, bus_cycles in zip(controllers, cycles, strict=True):
            for index, previous, event, gap in _bus_clashes(
                target, events, controller, lines, clock, None, bus_cycles
            ):
                if first is None or event["start_cycles"] < first[1]["start_cycles"]:
                    first = (index, event, bus_cycles[index] - gap, previous)
        if first is None:
            return current if holds else module
        index, event, need, previous = first
        op = ids[event["index"]]
        if op < 0 or last == (op, event["start_cycles"]):
            raise CompilationError(
                f"Controller.space_updates: the {event['kind']} on line {event['line']} at "
                f"{event['start_cycles']} cannot be held back to space update bus {index}, "
                f"{need} cycles after the {previous['kind']} on line {previous['line']} at "
                f"{previous['start_cycles']}; space them with a wait or a sync",
                code="limit_exceeded",
            )
        last = (op, event["start_cycles"])
        holds[op] = holds.get(op, 0) + need
    raise CompilationError(
        f"Controller.space_updates: spacing the updates of the update buses did not settle "
        f"after {MAX_ROUNDS} reschedules; space some with waits or syncs",
        code="limit_exceeded",
    )
