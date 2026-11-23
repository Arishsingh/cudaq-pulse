# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schedule golf: the program to shorten. Copy this file and edit the copy.

One parity check of the surface code, then every qubit is read out. The ancilla Z2 meets its four
data qubits D5, D8, D4 and D7 one after the other (a CZ each, in that order), with an echo pi pulse
on the data qubits half way.

Run it to see the schedule:   python challenges/schedule_golf/program.py
Score it:                     python challenges/schedule_golf/score.py challenges/schedule_golf/program.py
"""

import cudaq_pulse as pulse

# The scheduling policy the scorer compiles with: "asap", "alap", "rcp" or "alap_rcp".
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
        # A barrier: every drive line waits for the latest of them.
        sync(drives)
        if step == 3:
            # The echo, between the second and the third CZ.
            for i in (1, 2, 3, 4):
                drive(drives[i], drives[i].get_calibration(pulse.gates.X))
            sync(drives)
        # The CZ slot of Z2 and this step's data qubit: both wait, together.
        wait(drives[0], CZ_SLOT)
        wait(drives[step], CZ_SLOT)
    sync(drives)
    drive(drives[0], half_flip)

    # A barrier over every line, then the five readouts.
    sync(drives)
    for i in range(5):
        sync(drives[i], readouts[i])
    for i in range(5):
        readout(readouts[i], readouts[i].get_calibration(pulse.gates.READOUT))


if __name__ == "__main__":
    # The control system of the challenge: two readouts and four drive pulses at once.
    control = pulse.ControlSystem(max_concurrent_drives=4, max_concurrent_readouts=2)
    system = pulse.quantum_system("transmon_surface17", control_system=control)
    qubits = [system.qudit(name) for name in ("Z2", "D5", "D8", "D4", "D7")]
    program = pulse.compile(kernel, quantum_system=system, schedule=SCHEDULE).lower(*qubits)
    print(f"{SCHEDULE}: {float(program.duration):.0f} ns")
    for event in program.schedule:
        if event.kind in ("drive", "readout", "wait"):
            label = system.physical_qudits[event.qudit].label
            # event.start_cycles counts clock cycles: two per ns on this 2 GHz clock.
            start = event.start_cycles / system.clock_ghz
            end = start + event.duration_cycles / system.clock_ghz
            print(f"  {event.kind:8} {label:3} {start:6.0f} to {end:6.0f} ns")
