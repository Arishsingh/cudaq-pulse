# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reset one qubit, bronze: measure it, and flip it back only if it was found excited.

This is the ``active_reset`` kernel of the tutorial "Measure and React" without the pulse that
prepares the qubit and without the readout that checks the result: the scorer does both.

Score it:   python challenges/reset_one_qubit/score.py challenges/reset_one_qubit/reference/bronze.py
"""

import cudaq_pulse as pulse


@pulse.kernel
def reset(qudit):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    outcome = readout(readout_line, measurement)

    sync(drive_line, readout_line)
    if outcome:
        drive(drive_line, flip)
