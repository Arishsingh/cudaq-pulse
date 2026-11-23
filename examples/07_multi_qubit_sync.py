# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-qubit programs with sync barriers and registers of qudit handles.

Shows qudits passed one by one and as a register (a slice of the quantum system's
handles), the sync primitive for cross-line coordination, and patterns for
building multi-qubit pulse programs.
"""

import cudaq_pulse as pulse

# ── 1. Qudits passed one by one ──────────────────────────────────────────────


@pulse.kernel
def two_qudit_simultaneous(qudit_0, qudit_1, x_duration, x_sigma):
    """Drive two qubits simultaneously, then sync them."""

    drive_line_0 = get_drive_line(qudit_0)
    drive_line_1 = get_drive_line(qudit_1)

    drive(drive_line_0, gaussian(x_duration, 0.25, x_sigma))
    drive(drive_line_1, gaussian(x_duration, 0.30, x_sigma))

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(x_duration, 0.25, x_sigma))
    drive(drive_line_1, gaussian(x_duration, 0.30, x_sigma))


# ── 2. Qudits and amplitudes as parameters ───────────────────────────────────


@pulse.kernel
def parametric_pi_half(qudit_0, qudit_1, amplitude_0, amplitude_1, x_duration, x_sigma, drag_beta):
    """Calibration-friendly: pass the qudits, their amplitudes and the pulse shape in."""
    drive_line_0 = get_drive_line(qudit_0)
    drive_line_1 = get_drive_line(qudit_1)

    drive(drive_line_0, drag(x_duration, amplitude_0, x_sigma, drag_beta))
    drive(drive_line_1, drag(x_duration, amplitude_1, x_sigma, drag_beta))


# ── 3. A register of qudits ──────────────────────────────────────────────────


@pulse.kernel
def qvec_simultaneous_drive(qudit_vector, amplitude, x_duration, x_sigma):
    """Drive two qubits of a register in parallel.

    qs.qudits[0:4] is a register of four qudits; individual qudits
    are accessed by indexing.
    """
    drive_line_0 = get_drive_line(qudit_vector[0])
    drive_line_1 = get_drive_line(qudit_vector[1])

    drive(drive_line_0, gaussian(x_duration, amplitude, x_sigma))
    drive(drive_line_1, gaussian(x_duration, amplitude, x_sigma))
    sync(drive_line_0, drive_line_1)


# ── 4. Sync patterns ─────────────────────────────────────────────────────────


@pulse.kernel
def sync_after_asymmetric_ops(qudit_0, qudit_1, x_duration, x_sigma, long_duration):
    """Sync after different-duration ops to re-align timelines.

    Without sync, the two lines would diverge in time.
    """

    drive_line_0 = get_drive_line(qudit_0)
    drive_line_1 = get_drive_line(qudit_1)

    drive(drive_line_0, square(long_duration, 0.1))
    drive(drive_line_1, gaussian(x_duration, 0.25, x_sigma))

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(x_duration, 0.25, x_sigma))
    drive(drive_line_1, gaussian(x_duration, 0.25, x_sigma))


@pulse.kernel
def staggered_readout(qudit_0, qudit_1, x_duration, x_sigma, stagger_wait):
    """Stagger operations across qubits with wait + sync.

    Drive qudit_0, wait, then sync with qudit_1 before a joint operation.
    """

    drive_line_0 = get_drive_line(qudit_0)
    drive_line_1 = get_drive_line(qudit_1)

    drive(drive_line_0, gaussian(x_duration, 0.5, x_sigma))
    wait(drive_line_0, stagger_wait)

    drive(drive_line_1, gaussian(x_duration, 0.5, x_sigma))

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(x_duration, 0.25, x_sigma))
    drive(drive_line_1, gaussian(x_duration, 0.25, x_sigma))


if __name__ == "__main__":
    qs = pulse.quantum_system("transmon_surface17")
    x_duration = 20 * pulse.ns
    x_sigma = 5 * pulse.ns

    print("=== Two-qubit simultaneous drive ===")
    scheduled = pulse.compile(two_qudit_simultaneous, quantum_system=qs).lower(
        qs.qudits[0], qs.qudits[1], x_duration, x_sigma
    )
    print(scheduled.mlir())

    print("\n=== Parametric pi/2 ===")
    drag_beta = 0.25 * pulse.ns
    scheduled = pulse.compile(parametric_pi_half, quantum_system=qs).lower(
        qs.qudits[0], qs.qudits[1], 0.25, 0.28, x_duration, x_sigma, drag_beta
    )
    print(scheduled.mlir())

    print("\n=== Register simultaneous drive ===")
    scheduled = pulse.compile(qvec_simultaneous_drive, quantum_system=qs).lower(
        qs.qudits[0:4], 0.3, x_duration, x_sigma
    )
    print(scheduled.mlir())

    print("\n=== Sync after asymmetric ops ===")
    long_duration = 100 * pulse.ns
    scheduled = pulse.compile(sync_after_asymmetric_ops, quantum_system=qs).lower(
        qs.qudits[0], qs.qudits[1], x_duration, x_sigma, long_duration
    )
    print(scheduled.mlir())

    print("\n=== Staggered readout ===")
    stagger_wait = 250 * pulse.ns
    scheduled = pulse.compile(staggered_readout, quantum_system=qs).lower(
        qs.qudits[0], qs.qudits[1], x_duration, x_sigma, stagger_wait
    )
    print(scheduled.mlir())
    print(f"  Compile metrics: {scheduled.metrics}")
