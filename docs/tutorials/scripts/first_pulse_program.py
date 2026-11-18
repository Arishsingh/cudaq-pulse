# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Your First Pulse Program: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/first_pulse_program.html

Generated from docs/tutorials/first_pulse_program.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

import importlib.util


# Blocks under ``if VIZ:`` need matplotlib (``pip install "cudaq-pulse[viz]"``).
VIZ = importlib.util.find_spec("matplotlib") is not None


# %% Pick a quantum system
import cudaq_pulse as pulse

system = pulse.quantum_system("transmon_surface17")
qubit = system.qudits[0]

frequency_ghz = system.qudit_freq[0] / 1e9
print(qubit)
print(f"0-1 transition: {frequency_ghz} GHz")
print(f"clock:          {system.clock_ghz} GHz")

# Prints:
#   <Qudit 0 -> qudit 0 of transmon_surface17>
#   0-1 transition: 3.885 GHz
#   clock:          2.0 GHz


# %% Write a pulse kernel
@pulse.kernel
def rabi(qudit, amplitude: float):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = gaussian(
        duration=32 * pulse.ns,
        amplitude=amplitude,
        sigma=8 * pulse.ns,
    )

    drive(drive_line, flip)
    # Start the readout after the drive pulse ends.
    sync(drive_line, readout_line)
    return readout(readout_line, square(300 * pulse.ns, 0.05))


# %% Simulate it
result = pulse.simulate(rabi, qubit, 0.16)
levels = result.level_populations(qubit)

for level, population in enumerate(levels):
    print(f"level {level}: {population:.3f}")

# Prints:
#   level 0: 0.036
#   level 1: 0.963
#   level 2: 0.001


# %% Sweep the amplitude
import numpy as np

amplitudes = np.linspace(0, 0.4, 41)
populations = []
for amplitude in amplitudes:
    result = pulse.simulate(rabi, qubit, amplitude)
    populations.append(result.level_population(1))

print("compilations:", rabi.cache_info().compiled)

# Prints:
#   compilations: 1


# %% Take shots
result = pulse.simulate(
    rabi,
    qubit,
    0.16,
    shots_count=1000,
    seed=7,
)
outcomes = np.asarray(result.records)

print("shape:", outcomes.shape)
print("fraction that read 1:", outcomes.mean())

# Prints:
#   shape: (1000, 1)
#   fraction that read 1: 0.956


# %% See what was compiled
scheduled = rabi.lower(qubit, 0.16)

print(f"{'kind':8} {'start':>6} {'length':>6}")
for event in scheduled.schedule:
    start = event.start_cycles
    length = event.duration_cycles
    print(f"{event.kind:8} {start:>6} {length:>6}")

# Prints:
#   kind      start length
#   drive         0     64
#   readout      64    600


# %% Draw it
if VIZ:
    figure = scheduled.timeline()
    figure.savefig("rabi_schedule.svg")
