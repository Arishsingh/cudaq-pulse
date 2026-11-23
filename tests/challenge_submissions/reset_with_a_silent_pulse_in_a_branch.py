# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reset one qubit: two honest rounds, with a pulse of amplitude 0 in the else of the first."""

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
    else:
        drive(drive_line, square(40 * pulse.ns, 0.0))
    sync(drive_line, readout_line)
    outcome = readout(readout_line, measurement)
    sync(drive_line, readout_line)
    if outcome:
        drive(drive_line, flip)
    sync(drive_line, readout_line)
