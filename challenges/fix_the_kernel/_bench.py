# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The qubit and the bench controller of the challenge "Fix the kernel"."""

import cudaq_pulse as pulse

QUBIT = "D1"
MAX_AMPLITUDE = 0.25  # the most the bench plays


def the_bench():
    """(the quantum system, its qubit D1, the bench controller)."""
    system = pulse.quantum_system("transmon_surface17")
    bench = pulse.Controller(clock_ghz=system.clock_ghz, max_amplitude=MAX_AMPLITUDE)
    return system, system.qudit(QUBIT), bench


@pulse.kernel
def bench_readout(qudit):
    """The readout of broken.py by itself: the scorer takes the readout line's tone from it."""
    readout_line = get_readout_line(qudit)
    readout(readout_line, square(300 * pulse.ns, 0.05))
