# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase and frequency manipulation on tone channels.

Demonstrates shift_phase, set_phase, shift_frequency, set_frequency —
the four tone-modification primitives that control the rotating frame
of a drive or readout channel.
"""

import math

import cudaq_pulse as pulse


@pulse.kernel
def phase_rotation_demo(qudit, x_duration, x_sigma):
    """Drive X, then shift tone by pi/2 to drive Y, then by pi to drive -X.

    Illustrates how phase shifts rotate the drive axis on the Bloch sphere.
    """
    drive_line = get_drive_line(qudit)

    pi2_pulse = gaussian(x_duration, 0.25, x_sigma)

    drive(drive_line, pi2_pulse)

    shift_phase(drive_line.tone, math.pi / 2)
    drive(drive_line, pi2_pulse)

    shift_phase(drive_line.tone, math.pi / 2)
    drive(drive_line, pi2_pulse)

    shift_phase(drive_line.tone, math.pi / 2)
    drive(drive_line, pi2_pulse)


@pulse.kernel
def set_phase_vs_shift_phase(qudit, x_duration, x_sigma):
    """Contrast between set_phase (absolute) and shift_phase (relative).

    set_phase resets the tone to a fixed angle regardless of history.
    shift_phase accumulates relative to the current phase.
    """
    drive_line = get_drive_line(qudit)
    envelope = gaussian(x_duration, 0.3, x_sigma)

    shift_phase(drive_line.tone, math.pi / 4)
    drive(drive_line, envelope)

    shift_phase(drive_line.tone, math.pi / 4)
    drive(drive_line, envelope)

    set_phase(drive_line.tone, 0.0)
    drive(drive_line, envelope)


@pulse.kernel
def chirped_drive(qudit, step_count, total_shift_hz, step_duration):
    """Frequency-chirped pulse: step the drive frequency across the pulse.

    Useful for spectroscopy sweeps compiled as a single kernel.
    """
    drive_line = get_drive_line(qudit)
    step_size = total_shift_hz / step_count

    for _i in range(step_count):
        drive(drive_line, square(step_duration, 0.1))
        shift_frequency(drive_line.tone, step_size)


@pulse.kernel
def sideband_modulation(qudit, x_duration, x_sigma, park_wait):
    """Use set_frequency to park the drive tone at a sideband.

    Common in flux-tunable transmon architectures where the drive
    line needs to address different transition frequencies.
    """
    drive_line = get_drive_line(qudit)

    set_frequency(drive_line.tone, 5.0e9)
    drive(drive_line, gaussian(x_duration, 0.25, x_sigma))
    wait(drive_line, park_wait)

    set_frequency(drive_line.tone, 5.15e9)
    drive(drive_line, gaussian(x_duration, 0.25, x_sigma))


if __name__ == "__main__":
    qs = pulse.quantum_system("transmon_surface17")
    x_duration = 20 * pulse.ns
    x_sigma = 5 * pulse.ns

    print("=== Phase Rotation (4 x pi/2 = full circle) ===")
    scheduled = pulse.compile(phase_rotation_demo, quantum_system=qs).lower(
        qs.qudits[0], x_duration, x_sigma
    )
    print(scheduled.mlir())

    print("\n=== set_phase vs shift_phase ===")
    scheduled = pulse.compile(set_phase_vs_shift_phase, quantum_system=qs).lower(
        qs.qudits[0], x_duration, x_sigma
    )
    print(scheduled.mlir())

    print("\n=== Chirped Drive (10 steps, 50 MHz sweep) ===")
    step_duration = 10 * pulse.ns
    scheduled = pulse.compile(chirped_drive, quantum_system=qs).lower(
        qs.qudits[0], 10, 50e6, step_duration
    )
    print(scheduled.mlir())
    print(f"  Compile metrics: {scheduled.metrics}")

    print("\n=== Sideband Modulation ===")
    park_wait = 50 * pulse.ns
    scheduled = pulse.compile(sideband_modulation, quantum_system=qs).lower(
        qs.qudits[0], x_duration, x_sigma, park_wait
    )
    print(scheduled.mlir())
    print(f"  Compile metrics: {scheduled.metrics}")
