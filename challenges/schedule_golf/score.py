#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scorer of the challenge "Schedule golf".

    python challenges/schedule_golf/score.py my_program.py

my_program.py is an edited copy of program.py: it defines ``kernel``, which takes Z2, D5, D8, D4
and D7 of pulse.quantum_system("transmon_surface17"), and ``SCHEDULE``, the scheduling policy.

The submission never runs in the scorer's process. A second Python process (_child.py) loads the
file, compiles the kernel and writes what it plays: every pulse and readout with its start, its
samples, its frequency and its phase, and every wait on a drive line. The scorer reads only those
numbers. It compiles program.py itself and checks, on the numbers, that

- every qubit plays the operations of program.py in the same order, none starting before the one
  before it ends: the same pulses and readout, sample for sample on the same tone, and each CZ
  slot (Z2 and the data qubit both wait CZ_NS on their drive lines from the same moment) with
  the same partner;
- at most MAX_READOUTS readouts and MAX_DRIVES drive pulses play at once;
- no pulse, readout or wait depends on a measurement, and nothing else is played. A wait that
  is not half of a CZ slot only takes time.

The score is the time at which the last pulse, CZ slot or readout ends. Nothing is random: every
run prints the same lines.
"""

import importlib.util
import itertools
import sys
from pathlib import Path

import cudaq_pulse as pulse
from _qubits import ANCILLA, MAX_DRIVES, MAX_READOUTS, QUBITS, five_qubits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exchange import (  # noqa: E402  (after the path it is found on)
    InvalidSubmissionError,
    ask_child,
    driven,
    read_driven,
    run_scorer,
    said,
    same_pulse,
)

CZ_NS = 98
BRONZE_NS = 1100  # program.py takes 1312 ns
SILVER_NS = 978  # the least a valid schedule of this program can take
MAX_PROBLEMS = 8  # printed


def operations(drives, readouts, waits, others, system):
    """(each qubit's operations in time order, problems, the end of the last one in cycles)."""
    label = {system.qudit(name).index: name for name in QUBITS}
    to_ns = 1.0 / system.clock_ghz
    slot = round(CZ_NS * system.clock_ghz)
    problems = [
        f"it has {kind}; only the program's pulses, CZ slots and readouts are played"
        for kind in others
    ]
    ops = {name: [] for name in QUBITS}
    for kind, items in (("pulse", drives), ("readout", readouts)):
        for item in items:
            if item["qudit"] not in label:
                problems.append(f"a {kind} plays on a qubit that is not one of the five")
            elif item.get("depends") or item.get("in_branch"):
                problems.append(f"{label[item['qudit']]}: a {kind} depends on a measurement")
            else:
                ops[label[item["qudit"]]].append((item["start"], item["end"], kind, item))
    slots = {}
    for qudit, start, end, in_branch, on_drive_line in waits:
        if in_branch:
            problems.append(f"{label.get(qudit, 'another qubit')}: a wait depends on a measurement")
        elif end - start == slot and on_drive_line and qudit in label:
            slots.setdefault(start, []).append(label[qudit])
    for start, pair in sorted(slots.items()):
        if len(pair) != 2 or pair[0] == pair[1] or ANCILLA not in pair:
            problems.append(
                f"the {CZ_NS} ns wait at {start * to_ns:g} ns on {', '.join(pair)} is not a CZ slot: a slot is {ANCILLA} and one data qubit waiting {CZ_NS} ns from the same moment"
            )
            continue
        for name, partner in (pair, pair[::-1]):
            ops[name].append((start, start + slot, "cz", partner))
    for name, mine in ops.items():
        mine.sort(key=lambda op: op[:2])
        for before, after in itertools.pairwise(mine):
            if after[0] < before[1]:
                problems.append(
                    f"{name}: its {after[2]} at {after[0] * to_ns:g} ns starts before its {before[2]} ends"
                )
    for kind, limit in (("readout", MAX_READOUTS), ("pulse", MAX_DRIVES)):
        # +1 where one starts, -1 where one ends; an end sorts before a start at the same time.
        edges = sorted(
            edge
            for mine in ops.values()
            for op in mine
            if op[2] == kind
            for edge in ((op[0], 1), (op[1], -1))
        )
        counts = list(itertools.accumulate(step for _, step in edges))
        if counts and max(counts) > limit:
            when = edges[counts.index(max(counts))][0] * to_ns
            problems.append(
                f"{max(counts)} {kind}s play at once at {when:g} ns; the control system plays {limit}"
            )
    end = max((op[1] for mine in ops.values() for op in mine), default=0)
    return ops, problems, end


def names(mine):
    return " ".join(f"cz-{op[3]}" if op[2] == "cz" else op[2] for op in mine)


def differences(ops, reference):
    """Where a qubit does not play what it plays in program.py."""
    problems = []
    for name, wanted in reference.items():
        mine = ops[name]
        if names(mine) != names(wanted):
            problems.append(
                f"{name} plays [{names(mine)}]; in program.py it plays [{names(wanted)}]"
            )
            continue
        for place, (got, want) in enumerate(zip(mine, wanted, strict=True), start=1):
            if got[2] != "cz" and not same_pulse(got[3], want[3]):
                problems.append(
                    f"{name}: operation {place}, a {got[2]}, is not the one of program.py (its samples, frequency or phase differ)"
                )
    return problems


def the_program(system, qubits):
    """What program.py plays, compiled here; its schedule does not matter, only its operations."""
    spec = importlib.util.spec_from_file_location("_program", HERE / "program.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compiled = pulse.compile(module.kernel, quantum_system=system, schedule="rcp")
    program = compiled.lower(*qubits)
    ops, problems, _ = operations(*read_driven(driven(program)), system)
    assert not problems and not differences(ops, ops), problems
    return ops


def main(path):
    system, qubits = five_qubits()
    reference = the_program(system, qubits)
    try:
        reply = ask_child(HERE / "_child.py", [path])
        if "error" in reply:
            raise InvalidSubmissionError(f"it does not load or compile:{said(reply['error'])}")
        ops, problems, end = operations(*read_driven(reply), system)
    except InvalidSubmissionError as error:
        print(f"INVALID {error}\nBRONZE fail\nSILVER fail\nSCORE none")
        return 1
    problems += differences(ops, reference)
    for problem in problems[:MAX_PROBLEMS]:
        print(f"INVALID {problem}")
    if len(problems) > MAX_PROBLEMS:
        print(f"INVALID and {len(problems) - MAX_PROBLEMS} more")
    duration_ns = end / system.clock_ghz
    bronze = not problems and duration_ns <= BRONZE_NS
    silver = not problems and duration_ns <= SILVER_NS
    print(f"BRONZE {'pass' if bronze else 'fail'}  (valid and at most {BRONZE_NS} ns)")
    print(f"SILVER {'pass' if silver else 'fail'}  (valid and at most {SILVER_NS} ns)")
    print(f"SCORE duration_ns={duration_ns:g} valid={'no' if problems else 'yes'}")
    return 1 if problems else 0


if __name__ == "__main__":
    run_scorer(main, "usage: python challenges/schedule_golf/score.py my_program.py", seeded=False)
