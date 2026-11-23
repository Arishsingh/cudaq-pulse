# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The qubit of the challenge "Reset one qubit" and the controller a reset is compiled for."""

import cudaq_pulse as pulse

QUBIT = "D1"
LATENCY_NS = 200  # from the end of a readout to a pulse that depends on it


def the_qubit():
    """(the quantum system, its qubit D1, the controller with the feedback latency)."""
    system = pulse.quantum_system("transmon_surface17")
    controller = pulse.Controller(
        clock_ghz=system.clock_ghz, feedback_latency_cycles=round(LATENCY_NS * system.clock_ghz)
    )
    return system, system.qudit(QUBIT), controller


@pulse.kernel
def calibrated_readout(qudit):
    readout_line = get_readout_line(qudit)
    readout(readout_line, readout_line.get_calibration(pulse.gates.READOUT))


@pulse.kernel
def one_round(qudit):
    """What the scorer plays once before it starts the second process: a readout and a branch."""
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)
    outcome = readout(readout_line, readout_line.get_calibration(pulse.gates.READOUT))
    sync(drive_line, readout_line)
    if outcome:
        drive(drive_line, flip)
    else:
        wait(drive_line, 20 * pulse.ns)
        drive(drive_line, flip)
        drive(drive_line, flip)
