# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive Calibration in One Program: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/adaptive_calibration.html

Generated from docs/tutorials/adaptive_calibration.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

# %% The estimator
import cudaq_pulse as pulse
from cudaq_pulse.classical import calibration as cal


@pulse.classical(link="controller", max_latency_ns=500)
def update_t1(
    bit: pulse.Bits,
    shape: float,
    rate: float,
    wait_ns: float,
) -> tuple[float, float, float]:
    shape, rate = cal.adaptive_t1_update(
        shape,
        rate,
        bit[0],
        wait_ns,
        error_1to0=0.11,
        error_0to1=0.14,
    )
    next_wait_ns = cal.adaptive_t1_wait(shape, rate)
    next_wait_ns = min(max(next_wait_ns, 100.0), 200_000.0)
    return shape, rate, next_wait_ns


# %% The tracking kernel
@pulse.kernel
def track_t1(
    qudit,
    shots_count: int,
    shape: float,
    rate: float,
    wait_ns: float,
):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    for shot in range(shots_count):
        drive(drive_line, flip)
        wait(
            drive_line,
            wait_ns * pulse.ns,
            bounds=(100 * pulse.ns, 200 * pulse.us),
        )
        sync(drive_line, readout_line)
        outcome = readout(readout_line, measurement)
        request = update_t1.submit([outcome], shape, rate, wait_ns)
        shape, rate, wait_ns = request.result()

        sync(drive_line, readout_line)
        if outcome:
            drive(drive_line, flip)


# %% Compile the full run
system = pulse.quantum_system("transmon_surface17")
qubit = system.qudits[0]

controller_link = pulse.Link(fixed_ns=4, feedback_ns=8)
links = {"controller": controller_link}
control_system = pulse.ControlSystem(links=links)
compiled = pulse.compile(track_t1, control_system=control_system)


scheduled = compiled.lower(
    qubit,
    shots_count=50,
    shape=3.0,
    rate=450_000.0,
    wait_ns=50_000.0,
)
calls = scheduled.classical_report()
print("calls to the estimator:", len(calls))

# Prints:
#   calls to the estimator: 50


# %% Simulate a few shots
three_shots = compiled.lower(
    qubit,
    shots_count=3,
    shape=3.0,
    rate=450_000.0,
    wait_ns=50_000.0,
)
result = pulse.simulate(three_shots, levels=2)
print("branches:", len(result.branches))

# Prints:
#   branches: 8


likeliest = max(
    result.branches,
    key=lambda branch: branch.probability,
)

print(f"{'T1 estimate':>11}  {'next wait':>10}")
for shape, rate, wait_ns in likeliest.answers("update_t1"):
    t1_us = rate / shape / 1000
    wait_us = wait_ns / 1000
    print(f"{t1_us:8.1f} µs  {wait_us:7.1f} µs")

# Prints:
#   T1 estimate   next wait
#      127.7 µs     65.1 µs
#      111.0 µs     56.6 µs
#       98.1 µs     50.1 µs
