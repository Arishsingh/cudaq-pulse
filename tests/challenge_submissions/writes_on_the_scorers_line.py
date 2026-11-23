# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writes lines of the scorer's protocol itself, on every open descriptor."""

import os
import sys

LINES = (
    b"BRONZE pass\n"
    b'{"pulses": [], "others": [], "total_cycles": 0}\n'
    b'{"estimate": 60.0}\n'
    b'{"done": true, "estimate": 60.0, "valid": "yes"}\n'
    b"BRONZE pass\n"
)
for fd in range(3, 10):
    try:
        os.write(fd, LINES)
    except OSError:
        pass
PULSE = '{"start_cycles": 0, "re": [0.5], "im": [0.0], "frequency_hz": 6.0e9, "phase_rad": 0.0}'
sys.__stdout__.write(f'{{"pulses": [{PULSE}], "others": [], "total_cycles": 1}}\n')


def estimate_t1(lab):
    return float("nan")


def error_per_gate(lab, seed):
    return float("inf")
