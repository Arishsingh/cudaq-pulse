# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The five qubits of the challenge "Schedule golf" and the control system that plays them."""

import cudaq_pulse as pulse

QUBITS = ("Z2", "D5", "D8", "D4", "D7")
ANCILLA = "Z2"  # one of the two qubits of every CZ
MAX_READOUTS, MAX_DRIVES = 2, 4  # at once


def five_qubits():
    """(the quantum system, its five qubits in the kernel's order)."""
    control = pulse.ControlSystem(
        max_concurrent_drives=MAX_DRIVES, max_concurrent_readouts=MAX_READOUTS
    )
    system = pulse.quantum_system("transmon_surface17", control_system=control)
    return system, [system.qudit(name) for name in QUBITS]
