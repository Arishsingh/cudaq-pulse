# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure and React: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/measure_and_react.html

Generated from docs/tutorials/measure_and_react.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

# %% Outcomes and shots
import numpy as np

import cudaq_pulse as pulse

system = pulse.quantum_system("transmon_surface17")
qubit = system.qudits[0]


@pulse.kernel
def measure_superposition(qudit):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    half_flip = drive_line.get_calibration(pulse.gates.X90)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    drive(drive_line, half_flip)
    sync(drive_line, readout_line)
    return readout(readout_line, measurement)


result = pulse.simulate(
    measure_superposition,
    qubit,
    shots_count=1000,
    seed=7,
)
records = np.asarray(result.records)

print("first five shots:", records[:5, 0].tolist())
print("fraction that read 1:", records.mean())

# Prints:
#   first five shots: [1, 0, 0, 1, 1]
#   fraction that read 1: 0.496


# %% Branch on the outcome
@pulse.kernel
def active_reset(qudit):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)
    half_flip = drive_line.get_calibration(pulse.gates.X90)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    drive(drive_line, half_flip)
    sync(drive_line, readout_line)
    outcome = readout(readout_line, measurement)

    sync(drive_line, readout_line)
    if outcome:
        drive(drive_line, flip)

    sync(drive_line, readout_line)
    return readout(readout_line, measurement)


result = pulse.simulate(
    active_reset,
    qubit,
    shots_count=1000,
    seed=7,
)
records = np.asarray(result.records)
first, second = records.mean(axis=0)

print(f"first readout read 1:  {first:.3f}")
print(f"second readout read 1: {second:.3f}")
print(f"ground state at the end: {result.level_population(0):.3f}")

# Prints:
#   first readout read 1:  0.496
#   second readout read 1: 0.002
#   ground state at the end: 0.999


# %% How the simulator follows both outcomes
for branch in result.branches:
    if branch.probability < 0.01:
        continue
    probability = round(branch.probability, 3)
    print(f"readouts {branch.levels}: {probability}")

# Prints:
#   readouts (0, 0): 0.505
#   readouts (1, 0): 0.494


# %% See the branch in the schedule
scheduled = active_reset.lower(qubit)

print(f"{'kind':8} {'start':>6} {'length':>6}")
for event in scheduled.schedule:
    start = event.start_cycles
    length = event.duration_cycles
    print(f"{event.kind:8} {start:>6} {length:>6}")

# Prints:
#   kind      start length
#   drive         0     80
#   readout      80    600
#   drive       680     80
#   readout     760    600
