# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fix the kernel, bronze: four of the five mistakes, the ones a message or the schedule shows.

The fifth is left in: a turn above 0.75 asks for more amplitude than the bench plays.

Score it:   python challenges/fix_the_kernel/score.py challenges/fix_the_kernel/reference/bronze.py
"""

import cudaq_pulse as pulse


@pulse.kernel
def two_turns(qudit, turn: float, gap_ns: float):
    """Turn the qubit, wait, turn it again by as much, then read it out."""
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    # 0.33 for 16 ns is a pi pulse on this qubit.
    step = gaussian(
        duration=16 * pulse.ns,
        amplitude=0.33 * turn,
        # 2. A bare 4 is four clock cycles, 2 ns: a Gaussian half as wide, which turns the
        #    qubit half as far.
        sigma=4 * pulse.ns,
    )

    # 1. drve: the compiler names the op it does not know, and the one it expects.
    drive(drive_line, step)
    # 3. A bare gap_ns is clock cycles too: half the wait.
    wait(drive_line, gap_ns * pulse.ns)
    drive(drive_line, step)

    # 4. Lines keep their own time. Without the sync the readout does not wait for the
    #    drive line: the qubit is read out while the pulses play, or before.
    sync(drive_line, readout_line)
    return readout(readout_line, square(300 * pulse.ns, 0.05))
