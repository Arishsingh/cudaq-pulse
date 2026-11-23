# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fix the kernel: a kernel with five mistakes. Copy this file and fix the copy.

Score it:   python challenges/fix_the_kernel/score.py challenges/fix_the_kernel/broken.py
"""

import cudaq_pulse as pulse


@pulse.kernel
def two_turns(qudit, turn: float, gap_ns: float):
    """Turn the qubit, wait, turn it again by as much, then read it out.

    turn is a share of a full flip: 1 is a pi pulse, 0.5 half of one. Each turn is one Gaussian
    pulse; the second starts gap_ns after the first ends, and the readout starts when the second
    ends.
    """
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    # 0.33 for 16 ns is a pi pulse on this qubit.
    step = gaussian(
        duration=16 * pulse.ns,
        amplitude=0.33 * turn,
        sigma=4,
    )

    drve(drive_line, step)
    wait(drive_line, gap_ns)
    drive(drive_line, step)

    return readout(readout_line, square(300 * pulse.ns, 0.05))
