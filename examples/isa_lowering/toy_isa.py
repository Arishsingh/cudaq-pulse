# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering compiled programs to a toy controller instruction set.

Each toy controller has one sequencer. Its channels play waveforms from
a waveform table, each at its oscillator's frequency and phase. A
channel that several qudits share has an oscillator per tone, ``ch1.0``
and ``ch1.1``. Only ``wait`` moves time::

    set_freq ch0 5002000000     set channel 0's oscillator (Hz)
    shift_phase ch0 0.502655    add to its phase (rad)
    play ch0 w0 0.25            start waveform w0 at gain 0.25
    acquire ch1.0 w1 0.05 m4    start w1 and acquire into bit m4
    wait 35                     let 35 timesteps pass
    branch !m4 end0             jump to end0 when bit m4 is 0
    branch m4 end0              jump to end0 when bit m4 is 1
    jump end0                   jump to end0
    end0:                       a label
    submit L0 m0 m1             send bits m0 and m1 over link 0
    answer L0 a3                wait for the answer, into bit a3
    poll L0 a3                  take the answer if it is there

Outcome bits travel on a bus that every controller reads. A bit is
ready on the controller that acquires it ``feedback_latency_cycles``
after the acquisition ends, and on any other ``peer_latency_cycles``
later. Bits are named for their readout's event index, which every
controller's part of a program shares, so each part lowers on its own.

An answer's bits are named for its event index: ``a3`` is its first
value, ``a3.1`` its second, and after a ``poll``, ``a3.ok`` says whether
the answer was there.

- ``lower`` turns one controller's ``export_to_controller()`` program
  into a ``ToyProgram``.
- ``events`` reads a listing back into the pulses and calls it plays.
- ``run`` executes listings on emulated toy controllers.
- ``ToyBackend`` is the lowering as a ``cudaq_pulse.backends`` backend.

The tutorials "Lowering to Your Controller's Instruction Set" and
"Getting More out of Your Controller's Instruction Set" show this file
piece by piece. Run it to see a reset on one controller and a feedback
branch across two.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cudaq_pulse as pulse
from cudaq_pulse import ControlSystem
from cudaq_pulse.backends import IncompatibleBackendError
from cudaq_pulse.controller import (
    FORMAT,
    ReadoutLine,
    sample_waveform,
    shape_key,
)
from cudaq_pulse.quantum_systems import PhysicalQudit

__all__ = [
    "DESCRIPTION",
    "EMIT",
    "Played",
    "ToyBackend",
    "ToyProgram",
    "events",
    "lower",
    "run",
    "toy_control_system",
    "toy_controller",
    "toy_qpu",
]

Event = Mapping[str, Any]

# Each channel's oscillator, as (frequency in Hz, phase in rad).
Frames = dict[str, tuple[float | None, float]]

# A branch, or an event that always plays, and its events.
Unit = tuple[tuple[str, int], list[Event]]

# What one model of toy controller can do, as its maker would ship it.
DESCRIPTION = """\
model = "toy.sequencer"
version = "1"
clock_ghz = 1.0

# Pulses start and end every 4 cycles.
cycles_per_timestep = 4

# From an acquisition's end until a branch can use it.
feedback_latency_cycles = 80

# Instruction memory.
max_pulses = 256

# Waveform memory, and oscillator frequencies, per channel.
waveform_memory_samples = 1024
max_frequencies_per_line = 2

# A poll can give up on an answer that is late.
supports_timeouts = true

supports = [
    "play",
    "readout",
    "wait",
    "sync",
    "branch",
    "forward_outcome",
    "call",
    "answer",
]
"""


def toy_controller(**wiring: Any) -> pulse.Controller:
    """A toy controller with the installation's *wiring*: its lines,
    its peer latency and any override of the description."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "toy_sequencer.toml"
        path.write_text(DESCRIPTION)
        return pulse.controller(str(path), **wiring)


# ── The lowering pass ─────────────────────────────────────────────────


def refuse(why: str) -> IncompatibleBackendError:
    message = f"the toy sequencer cannot play this: {why}"
    return IncompatibleBackendError(message)


@dataclass(frozen=True)
class Wire:
    """Where an export line plays: a physical channel, whether several
    qudits share it, and a label for the listing."""

    channel: int
    shared: bool
    label: str


class ToyProgram:
    """One controller's listing, and what lowering it knows so far."""

    def __init__(self, program: Mapping[str, Any]) -> None:
        if program.get("format") != FORMAT:
            message = f"the toy sequencer reads {FORMAT} only"
            raise IncompatibleBackendError(message)

        self.controller: int = program.get("controller", 0)
        self.limits: Mapping[str, Any] = program["limits"] or {}
        self.timestep: int = self.limits.get("cycles_per_timestep") or 1

        # The listing so far, and the cycle the sequencer has reached.
        self.code: list[str] = []
        self.now = 0

        self.frames: Frames = {}
        self.table: dict[Any, int] = {}
        self.waveforms: list[Any] = []
        self.channels: dict[str, int] = {}
        self.polls: set[int] = set()
        self.wiring = self.wire(program["lines"])

    @staticmethod
    def wire(lines: Sequence[Mapping[str, Any]]) -> dict[int, Wire]:
        """Each export line's physical channel. The lines of a shared
        line's qudits are tones of one channel."""
        channel_of: dict[tuple[str, tuple[int, ...]], int] = {}
        wiring = {}

        for line in lines:
            qudits = tuple(line.get("qudits", [line["qudit"]]))
            key = (line["role"], qudits)
            channel = channel_of.setdefault(key, len(channel_of))
            label = f"{line['role']} q{line['qudit']}"
            wiring[line["line"]] = Wire(channel, len(qudits) > 1, label)

        return wiring

    def channel(self, event: Event) -> str:
        """The channel *event* plays on: ``ch0``, or ``ch1.0`` for tone
        0 of a shared channel."""
        wire = self.wiring[event["line"]]
        if wire.shared:
            name = f"ch{wire.channel}.{event['tone_slot']}"
        else:
            name = f"ch{wire.channel}"

        self.channels[name] = event["line"]
        return name

    def wait_until(self, cycle: int) -> None:
        gap = cycle - self.now
        if gap < 0 or gap % self.timestep:
            why = f"no wait goes from cycle {self.now} to {cycle}"
            raise refuse(why)

        if gap:
            timesteps = gap // self.timestep
            self.code.append(f"wait {timesteps}")
        self.now = cycle

    def __str__(self) -> str:
        head = []
        for name, line in self.channels.items():
            head.append(f".chan {name} {self.wiring[line].label}")
        for number, samples in enumerate(self.waveforms):
            head.append(f".wave w{number} {len(samples)}")

        return "\n".join(head + self.code)


def set_frame(
    toy: ToyProgram,
    channel: str,
    frequency: float | None,
    phase: float,
) -> None:
    """Set *channel*'s oscillator to *frequency*, and shift its phase by
    the difference to *phase*."""
    old_frequency, old_phase = toy.frames.get(channel, (None, 0.0))
    if frequency != old_frequency:
        toy.code.append(f"set_freq {channel} {frequency:.0f}")

    shift = math.remainder(phase - old_phase, 2 * math.pi)
    if abs(shift) > 1e-9:
        toy.code.append(f"shift_phase {channel} {shift:.6f}")

    toy.frames[channel] = (frequency, phase)


def gain(event: Event) -> complex:
    amplitude = event["amplitude"]
    return complex(amplitude["re"], amplitude["im"])


def written(value: complex) -> str:
    """*value* as the listing writes it: real when it is real."""
    if value.imag == 0:
        return f"{value.real:g}"
    return f"{value:g}"


def refuse_run_time_values(event: Event) -> None:
    """Refuse a pulse with a value that comes from a decoder's answer."""
    values = (
        shape_key(event["waveform"]),
        event["amplitude"],
        event["frequency_hz"],
        event["phase_rad"],
    )
    if None in values or "tone" in event:
        index = event["index"]
        why = f"event {index} has a value known only at run time"
        raise refuse(why)


def table_entry(toy: ToyProgram, event: Event) -> str:
    """The waveform table's name for *event*'s envelope. The first time
    an envelope is seen, its samples are stored at gain 1."""
    key = shape_key(event["waveform"])
    if key not in toy.table:
        samples = sample_waveform(
            event["waveform"],
            event["duration_cycles"],
        )
        toy.table[key] = len(toy.waveforms)
        toy.waveforms.append(samples / gain(event))

    return f"w{toy.table[key]}"


def play(toy: ToyProgram, event: Event) -> None:
    """A drive as a ``play``. A readout as an ``acquire`` into a bit."""
    refuse_run_time_values(event)
    toy.wait_until(event["start_cycles"])
    channel = toy.channel(event)
    set_frame(
        toy,
        channel,
        frequency=event["frequency_hz"],
        phase=event["phase_rad"],
    )

    waveform = table_entry(toy, event)
    operands = f"{channel} {waveform} {written(gain(event))}"
    if event["kind"] == "readout":
        toy.code.append(f"acquire {operands} m{event['index']}")
    else:
        toy.code.append(f"play {operands}")


def submit(toy: ToyProgram, event: Event) -> None:
    """A call to the decoder, sending the outcome bits it reads."""
    bits = []
    for operand in event["operands"]:
        if set(operand) != {"readout"}:
            index = event["index"]
            why = f"event {index} sends a value computed at run time"
            raise refuse(why)
        bits.append(f"m{operand['readout']}")

    toy.wait_until(event["start_cycles"])
    toy.code.append(f"submit L{event['link']} {' '.join(bits)}")


def answer(toy: ToyProgram, event: Event) -> None:
    """The decoder's answer: an ``answer`` that waits for it, or, for a
    call with a timeout, a ``poll`` at the deadline."""
    toy.wait_until(event["start_cycles"])
    if event.get("wait") == "timeout":
        instruction = "poll"
        toy.polls.add(event["index"])
    else:
        instruction = "answer"

    toy.code.append(f"{instruction} L{event['link']} a{event['index']}")


# What the sequencer issues for each kind of event.
EMIT: dict[str, Callable[[ToyProgram, Event], None]] = {
    "drive": play,
    "readout": play,
    "submit": submit,
    "answer": answer,
}


def tested_bit(toy: ToyProgram, condition: Mapping[str, Any]) -> str:
    """The bit a branch tests: ``m4`` for readout 4's outcome, ``a3``
    or ``a3.1`` for answer 3's first or second value, and ``a3.ok``
    for whether a poll found answer 3 in time."""
    if condition["test"] == "is_set":
        return f"m{condition['readouts'][0]}"

    number = condition["answers"][0]
    result = condition["result"]
    if number in toy.polls:
        if result == 0:
            return f"a{number}.ok"
        result -= 1

    if result == 0:
        return f"a{number}"
    return f"a{number}.{result}"


def start_cycle(event: Event) -> int:
    return event["start_cycles"]


def span(events: Sequence[Event]) -> tuple[int, int]:
    """The cycles of a branch's first and last pulse, over both arms."""
    starts = [start_cycle(event) for event in events]
    return min(starts), max(starts)


def arm(
    toy: ToyProgram,
    events: Sequence[Event],
    *,
    taken: bool,
    before: Frames,
) -> list[str]:
    """One arm's code: its pulses, then a wait to the branch's end."""
    first, last = span(events)
    toy.now = first
    toy.frames = dict(before)
    start = len(toy.code)

    for event in sorted(events, key=start_cycle):
        if event["conditions"][0]["arm"] is taken:
            play(toy, event)
    toy.wait_until(last)
    restore_phases(toy, before)

    code = toy.code[start:]
    del toy.code[start:]
    return code


def restore_phases(toy: ToyProgram, before: Frames) -> None:
    """Shift each oscillator's phase back to what it was before an arm,
    so what follows the branch doesn't depend on the outcome."""
    for channel, (frequency, _) in toy.frames.items():
        _, phase = before.get(channel, (None, 0.0))
        set_frame(toy, channel, frequency=frequency, phase=phase)


def branch_code(
    number: int,
    bit: str,
    taken: list[str],
    other: list[str],
) -> list[str]:
    """Both arms behind one branch, with a jump when both play."""
    end = f"end{number}"
    if not taken:
        return [f"branch {bit} {end}", *other, f"{end}:"]
    if not other:
        return [f"branch !{bit} {end}", *taken, f"{end}:"]

    otherwise = f"else{number}"
    return [
        f"branch !{bit} {otherwise}",
        *taken,
        f"jump {end}",
        f"{otherwise}:",
        *other,
        f"{end}:",
    ]


def branch(
    toy: ToyProgram,
    number: int,
    events: Sequence[Event],
) -> None:
    """Both arms of branch *number*, ending at the same cycle."""
    condition = events[0]["conditions"][0]
    bit = tested_bit(toy, condition)
    first, _ = span(events)
    toy.wait_until(first)

    before = toy.frames
    taken = arm(toy, events, taken=True, before=before)
    other = arm(toy, events, taken=False, before=before)
    toy.code.extend(branch_code(number, bit, taken, other))

    # A later pulse sets its frequency again. Phases carry on.
    toy.frames = {}
    for channel, (_, phase) in before.items():
        toy.frames[channel] = (None, phase)


# Already in the start times, and in each pulse's frame.
FOLDED = ("wait", "sync", "frame_update")


def check(event: Event) -> None:
    """Refuse an event the toy sequencer has no instruction for."""
    index = event["index"]
    if event["kind"] not in EMIT:
        raise refuse(f"event {index} is a {event['kind']}")

    conditions = event["conditions"]
    if not conditions:
        return

    nested = len(conditions) > 1
    computed = conditions[0]["test"] not in ("is_set", "answer")
    is_call = event["kind"] in ("submit", "answer")
    if nested or computed or is_call:
        raise refuse(f"event {index} is in a nested or computed branch")


def unit_of(event: Event) -> tuple[str, int]:
    """An event that always plays is a unit of its own. The events of
    both arms of a branch are one unit."""
    conditions = event["conditions"]
    if conditions:
        return ("branch", conditions[0]["branch"])
    return ("event", event["index"])


def units(program: Mapping[str, Any]) -> list[Unit]:
    """The program's events in units, in time order."""
    grouped: dict[tuple[str, int], list[Event]] = {}
    for event in program["events"]:
        if event["kind"] in FOLDED:
            continue
        check(event)
        grouped.setdefault(unit_of(event), []).append(event)

    def first_start(unit: Unit) -> int:
        _, members = unit
        return span(members)[0]

    return sorted(grouped.items(), key=first_start)


def lower(program: Mapping[str, Any]) -> ToyProgram:
    """The toy listing of one controller's ``export_to_controller()``
    program. Raises ``IncompatibleBackendError`` for what the toy
    sequencer cannot play."""
    toy = ToyProgram(program)

    for (kind, number), members in units(program):
        if kind == "branch":
            branch(toy, number, members)
        else:
            event = members[0]
            emit = EMIT[event["kind"]]
            emit(toy, event)

    return toy


# ── Reading a listing back ────────────────────────────────────────────

# The kind of export event each instruction plays.
PULSES = {"play": "drive", "acquire": "readout"}
CALLS = {"submit": "submit", "answer": "answer", "poll": "answer"}


def pulse_event(
    toy: ToyProgram,
    op: str,
    args: list[str],
    now: int,
) -> dict[str, Any]:
    channel, waveform = args[0], args[1]
    samples = toy.waveforms[int(waveform.removeprefix("w"))]
    return {
        "kind": PULSES[op],
        "line": toy.channels[channel],
        "start_cycles": now,
        "duration_cycles": len(samples),
    }


def call_event(op: str, args: list[str], now: int) -> dict[str, Any]:
    link = int(args[0].removeprefix("L"))
    return {"kind": CALLS[op], "line": link, "start_cycles": now}


def events(toy: ToyProgram) -> list[dict[str, Any]]:
    """What *toy* plays, both arms of each branch, read back from its
    listing alone."""
    found = []
    now = 0
    branched_at = 0

    for instruction in toy.code:
        op, *args = instruction.split()
        if op == "wait":
            now += int(args[0]) * toy.timestep
        elif op == "branch":
            branched_at = now
        elif op.startswith("else"):
            now = branched_at
        elif op in PULSES:
            found.append(pulse_event(toy, op, args, now))
        elif op in CALLS:
            found.append(call_event(op, args, now))

    return found


# ── Emulated toy controllers ──────────────────────────────────────────


@dataclass(frozen=True)
class Played:
    """A pulse a toy controller played: which controller, when, and
    what."""

    controller: int
    start_cycles: int
    instruction: str


def peer_latency(program: ToyProgram, receiver: int) -> int:
    latency = program.limits.get("peer_latency_cycles") or 0
    if isinstance(latency, Mapping):
        return latency.get(str(receiver), latency.get(receiver, 0))
    return latency


def ready_cycles(
    programs: Sequence[ToyProgram],
) -> dict[str, tuple[int, int]]:
    """Each acquired bit's controller, and the cycle the bit is ready
    there: the acquisition's end plus the feedback latency."""
    ready = {}
    for program in programs:
        feedback = program.limits.get("feedback_latency_cycles") or 0
        now = 0
        branched_at = 0

        for instruction in program.code:
            op, *args = instruction.split()
            if op == "wait":
                now += int(args[0]) * program.timestep
            elif op == "branch":
                branched_at = now
            elif op.startswith("else"):
                now = branched_at
            elif op == "acquire":
                table_index = int(args[1].removeprefix("w"))
                ends = now + len(program.waveforms[table_index])
                ready[args[3]] = (program.controller, ends + feedback)

    return ready


def labels(program: ToyProgram) -> dict[str, int]:
    found = {}
    for number, instruction in enumerate(program.code):
        if instruction.endswith(":"):
            found[instruction.removesuffix(":")] = number
    return found


def run(
    programs: Sequence[ToyProgram],
    outcomes: Mapping[str, int],
) -> list[Played]:
    """The pulses toy controllers play when their acquisitions give
    *outcomes*, by bit (``{"m4": 1}``; a bit not given is 0). A branch
    on a bit before it can have reached the controller raises
    ``RuntimeError``."""
    ready = ready_cycles(programs)
    by_number = {program.controller: program for program in programs}
    played = []

    for program in programs:
        targets = labels(program)
        now = 0
        counter = 0

        while counter < len(program.code):
            instruction = program.code[counter]
            op, *args = instruction.split()
            counter += 1

            if op == "wait":
                now += int(args[0]) * program.timestep
            elif op in ("play", "acquire"):
                played.append(
                    Played(program.controller, now, instruction),
                )
            elif op == "jump":
                counter = targets[args[0]]
            elif op == "branch":
                bit = args[0].removeprefix("!")
                jump_on = 0 if args[0].startswith("!") else 1

                # An answer is taken where the compiler placed it. An
                # outcome is ready its reader's latencies after it ends.
                reader, at = ready.get(bit, (program.controller, now))
                if reader != program.controller:
                    sender = by_number[reader]
                    at += peer_latency(sender, program.controller)
                if now < at:
                    raise RuntimeError(
                        f"controller {program.controller} branches on "
                        f"{bit} at cycle {now}, before cycle {at}, "
                        "when it has it"
                    )

                if outcomes.get(bit, 0) == jump_on:
                    counter = targets[args[1]]

    played.sort(key=lambda p: (p.start_cycles, p.controller))
    return played


# ── The backend ───────────────────────────────────────────────────────


def toy_control_system() -> ControlSystem:
    """Two toy controllers and a decoder on a link. The first drives
    qudits 0 and 1 and reads both on one feedline; the second drives
    and reads qudit 2."""
    feedline = ReadoutLine([0, 1], max_concurrent_tones=2)
    first = toy_controller(
        drive_lines=[0, 1],
        readout_lines=[feedline],
        peer_latency_cycles=40,
    )
    second = toy_controller(
        drive_lines=[2],
        readout_lines=[2],
        peer_latency_cycles=40,
    )
    decoder = pulse.Link(fixed_ns=4, feedback_ns=8)

    return ControlSystem(
        controllers=[first, second],
        reference_clock_ghz=0.25,
        links={"decoder": decoder},
    )


class ToyBackend:
    """The toy controllers as a ``cudaq_pulse.backends.Backend``."""

    name = "toy"
    formats = (FORMAT,)

    def control_system(self) -> ControlSystem:
        return toy_control_system()

    def emit(self, program: Mapping[str, Any]) -> ToyProgram:
        return lower(program)

    def events(self, artifact: ToyProgram) -> list[list[Any]]:
        return [events(artifact)]

    def run(
        self,
        artifact: ToyProgram,
        *,
        shots_count: int = 1,
    ) -> list[list[Played]]:
        """Each shot's pulses. Toy controllers have no qudits, so every
        outcome is 0."""
        return [run([artifact], {}) for _ in range(shots_count)]


def toy_qpu() -> pulse.QuantumSystem:
    """Three qudits on the toy controllers, at a 1 GHz clock."""
    qudits = {}
    for number in range(3):
        qudits[number] = PhysicalQudit(
            index=number,
            frequency_hz=5.0e9 + 3.0e8 * number,
            t1_us=50.0,
            t2_star_us=30.0,
        )

    return pulse.quantum_system(
        name="toy-qpu",
        qudits=qudits,
        clock_ghz=1.0,
        control_system=toy_control_system(),
    )


if __name__ == "__main__":
    from cudaq_pulse import backends

    backend = ToyBackend()
    qpu = toy_qpu()

    @pulse.kernel
    def reset(q):
        drive_line = get_drive_line(q)
        readout_line = get_readout_line(q)
        m = readout(readout_line, square(duration=200, amplitude=0.05))
        sync(drive_line, readout_line)
        if m:
            flip = gaussian(duration=40, amplitude=0.5, sigma=10.0)
            drive(drive_line, flip)

    compiled = pulse.compile(reset, quantum_system=qpu, schedule="rcp")
    one = compiled.lower(qpu.qudits[0])
    print(backend.emit(one.export_to_controller()))

    @pulse.kernel
    def flip_other(measured, other):
        readout_line = get_readout_line(measured)
        drive_line = get_drive_line(other)
        m = readout(readout_line, square(duration=200, amplitude=0.05))
        sync(readout_line, drive_line)
        if m:
            flip = gaussian(duration=40, amplitude=0.5, sigma=10.0)
            drive(drive_line, flip)

    compiled = pulse.compile(
        flip_other,
        quantum_system=qpu,
        schedule="rcp",
    )
    split = compiled.lower(qpu.qudits[0], qpu.qudits[2])
    programs = []
    for number in range(2):
        program = lower(split.export_to_controller(controller=number))
        print(f"; controller {number}\n{program}")
        programs.append(program)

    for played in run(programs, {"m0": 1}):
        print(played)

    no_latency = toy_controller(feedback_latency_cycles=0)
    report = backends.validate(backend, controller=no_latency)
    print(str(report).splitlines()[-1])
