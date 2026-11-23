# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reset one qubit, silver: reset it twice.

One round leaves up to 1 % of an excited qubit excited. Most of it comes from the 200 ns the
controller needs between the readout and the pulse that depends on it: a qubit that read 1 and
decays while it waits is flipped back up by the pi pulse. The rest is the readout reading 0 for a
qubit in level 1.

A second round catches both. One round takes 300 ns of readout, 200 ns of feedback latency and a
40 ns pulse, so two rounds fit the 1200 ns.

Score it:   python challenges/reset_one_qubit/score.py challenges/reset_one_qubit/reference/silver.py
"""

import cudaq_pulse as pulse


@pulse.kernel
def reset(qudit):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    for _round in range(2):
        outcome = readout(readout_line, measurement)

        sync(drive_line, readout_line)
        if outcome:
            drive(drive_line, flip)

        sync(drive_line, readout_line)
