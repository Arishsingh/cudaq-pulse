# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run by score.py in a process of its own: the only code that loads the submission.

    python _child.py my_pulse.py QUDIT

Compiles the submission's ``pi_pulse`` for the qudit and writes, as one line of JSON, what the
kernel puts on the drive line: each pulse's start, samples, frequency and phase.
"""

import sys
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.controller import sample_waveform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _exchange import answer  # noqa: E402  (after the path it is found on)


def compiled(submission):
    system = pulse.quantum_system("transmon_surface17")
    qudit = system.qudit(sys.argv[2])
    program = pulse.compile(submission.pi_pulse, quantum_system=system).lower(qudit)
    pulses, others, total = [], [], 0
    for event in program.export_to_controller()["events"]:
        total = max(total, int(event["end_cycles"]))
        if event["kind"] == "drive":
            samples = np.asarray(
                sample_waveform(event["waveform"], event["duration_cycles"]), dtype=complex
            )
            pulses.append(
                {
                    "start_cycles": int(event["start_cycles"]),
                    "re": [float(x) for x in samples.real],
                    "im": [float(x) for x in samples.imag],
                    "frequency_hz": float(event["frequency_hz"]),
                    "phase_rad": float(event["phase_rad"]),
                }
            )
        elif event["kind"] not in ("wait", "frame_update"):
            others.append(str(event["kind"]))
    return {"pulses": pulses, "others": others, "total_cycles": total}


if __name__ == "__main__":
    answer(compiled)
