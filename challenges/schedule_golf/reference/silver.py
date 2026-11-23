# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schedule golf, silver: the readouts in the order the qubits finish. 978 ns.

The bronze solution, with one change. Two readouts play at once, and the five do not take the
same time: a data qubit's lasts 300 ns, Z2's 200 ns. The data qubits are done at 178, 276, 334
and 432 ns and Z2 at 472 ns. The resource-constrained policy places the operations in the order
the kernel states them and never moves a later one in front, so the order of the readout
statements decides who gets a free lane. In the order the qubits finish, one lane plays D5, D4
and Z2 (178 to 978 ns) and the other D8 and D7 (276 to 876 ns).

978 ns is the least any schedule can reach: 1400 ns of readouts on two lanes, neither free before
178 and 276 ns, and no way to split them more evenly than 800 and 600 ns.

Score it:   python challenges/schedule_golf/score.py challenges/schedule_golf/reference/silver.py
Run it by itself to see when each readout plays.
"""

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
    for i in (1, 2, 3, 4, 0):
        sync(drives[i], readouts[i])
        readout(readouts[i], readouts[i].get_calibration(pulse.gates.READOUT))


if __name__ == "__main__":
    control = pulse.ControlSystem(max_concurrent_drives=4, max_concurrent_readouts=2)
    system = pulse.quantum_system("transmon_surface17", control_system=control)
    qubits = [system.qudit(name) for name in ("Z2", "D5", "D8", "D4", "D7")]
    program = pulse.compile(kernel, quantum_system=system, schedule=SCHEDULE).lower(*qubits)
    print(f"{SCHEDULE}: {float(program.duration):.0f} ns")
    for event in program.schedule:
        if event.kind == "readout":
            label = system.physical_qudits[event.qudit].label
            start = event.start_cycles / system.clock_ghz
            end = start + event.duration_cycles / system.clock_ghz
            print(f"  readout {label:3} {start:5.0f} to {end:5.0f} ns")
