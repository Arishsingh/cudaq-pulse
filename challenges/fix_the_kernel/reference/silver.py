# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fix the kernel, silver: all five mistakes.

The first four are fixed as in bronze.py. The fifth takes one idea: a pulse turns the qubit by
its amplitude times its length, so half the amplitude for twice as long is the same turn. A
Gaussian's length is its sigma as well as its duration, so both double.

Score it:   python challenges/fix_the_kernel/score.py challenges/fix_the_kernel/reference/silver.py
"""

import cudaq_pulse as pulse


@pulse.kernel
def two_turns(qudit, turn: float, gap_ns: float):
    """Turn the qubit, wait, turn it again by as much, then read it out."""
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    # 5. 0.33 for 16 ns is over the bench's 0.25 once turn passes 0.75. 0.165 for 32 ns is
    #    the same pi pulse (the one of "Your First Pulse Program") and stays under it.
    step = gaussian(
        duration=32 * pulse.ns,
        amplitude=0.165 * turn,
        sigma=8 * pulse.ns,
    )

    drive(drive_line, step)
    wait(drive_line, gap_ns * pulse.ns)
    drive(drive_line, step)

    sync(drive_line, readout_line)
    return readout(readout_line, square(300 * pulse.ns, 0.05))
