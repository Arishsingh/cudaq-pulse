# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schedule golf: the silver solution, with a wait that depends on Z2's readout after it."""

import cudaq_pulse as pulse

SCHEDULE = "rcp"

# A CZ is a flux pulse. Flux lines are not modeled yet, so its 98 ns slot idles both qubits.
CZ_SLOT = 98 * pulse.ns


@pulse.kernel
def kernel(z2, d5, d8, d4, d7):
    drives = []
    readouts = []
    for qubit in (z2, d5, d8, d4, d7):
        drives.append(get_drive_line(qubit))
        readouts.append(get_readout_line(qubit))
    half_flip = drives[0].get_calibration(pulse.gates.X90)

    drive(drives[0], half_flip)
    for step in (1, 2, 3, 4):
        flip = drives[step].get_calibration(pulse.gates.X)
        if step > 2:
            # D4 and D7 play their echo before their CZ ...
            drive(drives[step], flip)
        # The CZ slot: only Z2 and this data qubit wait for each other. The sync after it keeps
        # the two waits together under every policy.
        sync(drives[0], drives[step])
        wait(drives[0], CZ_SLOT)
        wait(drives[step], CZ_SLOT)
        sync(drives[0], drives[step])
        if step <= 2:
            # ... and D5 and D8 after it.
            drive(drives[step], flip)
    drive(drives[0], half_flip)

    # Each qubit is read out when its own drive line is done: the data qubits in the order of
    # their CZs, Z2 last.
    for i in (1, 2, 3, 4):
        sync(drives[i], readouts[i])
        readout(readouts[i], readouts[i].get_calibration(pulse.gates.READOUT))
    sync(drives[0], readouts[0])
    seen = readout(readouts[0], readouts[0].get_calibration(pulse.gates.READOUT))
    if seen:
        wait(drives[1], 10 * pulse.ns)
