# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run by score.py in a process of its own: the only code that loads the submission.

    python _child.py my_program.py

Compiles the submission's ``kernel`` with its ``SCHEDULE`` for the five qubits and writes, as one
line of JSON, what it plays: each pulse and readout with its start, samples, frequency and phase,
and each wait on a drive line.
"""

import sys
from pathlib import Path

import cudaq_pulse as pulse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _exchange import answer, driven  # noqa: E402  (after the path it is found on)
from _qubits import five_qubits  # noqa: E402


def compiled(submission):
    system, qubits = five_qubits()
    program = pulse.compile(submission.kernel, quantum_system=system, schedule=submission.SCHEDULE)
    return driven(program.lower(*qubits))


if __name__ == "__main__":
    answer(compiled)
