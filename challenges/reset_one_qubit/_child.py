# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run by score.py in a process of its own: the only code that loads the submission.

    python _child.py my_reset.py

Compiles the submission's ``reset`` for qubit D1 on a controller with the feedback latency and
writes, as one line of JSON, what it plays: each pulse and readout with its start, samples,
frequency and phase, and for a pulse in a branch the readout it depends on. The states the scorer
tries are not here.
"""

import sys
from pathlib import Path

import cudaq_pulse as pulse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _exchange import answer, driven  # noqa: E402  (after the path it is found on)
from _qubit import the_qubit  # noqa: E402


def compiled(submission):
    system, qubit, controller = the_qubit()
    # "rcp" keeps a branch the feedback latency away from its readout; flatten_schedules applies
    # it to the kernels that reset calls too.
    program = pulse.compile(
        submission.reset,
        quantum_system=system,
        control_system=controller,
        schedule="rcp",
        flatten_schedules=True,
    )
    return driven(program.lower(qubit))


if __name__ == "__main__":
    answer(compiled)
