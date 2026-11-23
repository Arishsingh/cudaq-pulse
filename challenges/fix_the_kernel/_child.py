# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run by score.py in a process of its own: the only code that loads the submission.

    python _child.py my_kernel.py        (the cases arrive on standard input)

Compiles the submission's ``two_turns`` for qubit D1 on the bench controller, once for each case
the scorer sends, and writes, as one line of JSON, what each plays (the pulses and the readout
with their starts, samples, frequencies and phases) or the compiler's message.
"""

import json
import sys
from pathlib import Path

import cudaq_pulse as pulse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bench import the_bench  # noqa: E402  (after the path it is found on)
from _exchange import answer, driven, explain  # noqa: E402


def compiled(submission):
    system, qubit, bench = the_bench()
    kernel = submission.two_turns
    cases = []
    for turn, gap_ns in CASES:  # read below, before the submission is loaded
        try:
            program = pulse.compile(kernel, quantum_system=system, control_system=bench)
            cases.append(driven(program.lower(qubit, turn, gap_ns)))
        except Exception as error:
            cases.append({"error": explain(error)})
    return {"cases": cases}


if __name__ == "__main__":
    CASES = json.loads(sys.stdin.readline())["cases"]
    answer(compiled)
