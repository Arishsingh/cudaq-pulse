# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coherence: T1, Ramsey and Echo: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/coherence.html

Generated from docs/tutorials/coherence.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

# %% What the model knows
import numpy as np

import cudaq_pulse as pulse

system = pulse.quantum_system("transmon_surface17")
qubit = system.qudits[0]
d1 = system.physical_qudits[0]

print(f"qubit:   {d1.label}")
print(f"T1:      {d1.t1_us} µs")
print(f"T2*:     {d1.t2_star_us} µs")
print(f"T2 echo: {d1.t2_echo_us} µs")

# Prints:
#   qubit:   D1
#   T1:      29.1 µs
#   T2*:     35.0 µs
#   T2 echo: 46.2 µs


flip = qubit.get_calibration(pulse.gates.X)
print(f"{flip.duration_ns} ns DRAG, amplitude {flip.amplitude:.4f}")

# Prints:
#   40.0 ns DRAG, amplitude 0.1313


# %% Measure T1
@pulse.kernel
def t1(qudit, wait_ns: float):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    drive(drive_line, flip)
    wait(drive_line, wait_ns * pulse.ns)
    sync(drive_line, readout_line)
    return readout(readout_line, measurement)


waits = np.linspace(0, 60_000, 7)
populations = []

print("wait (µs)  excited")
for wait_ns in waits:
    excited = pulse.simulate(t1, qubit, wait_ns).level_population(1)
    populations.append(excited)
    print(f"{wait_ns / 1000:9.0f}  {excited:.3f}")
print("compilations:", t1.cache_info().compiled)

# Prints:
#   wait (µs)  excited
#           0  0.968
#          10  0.686
#          20  0.491
#          30  0.355
#          40  0.259
#          50  0.191
#          60  0.143
#   compilations: 1


# %% Fit T1
excess = np.array(populations) - d1.thermal_population
slope, intercept = np.polyfit(waits, np.log(excess), 1)
fitted_t1_us = -1 / slope / 1000
print(f"fitted T1: {fitted_t1_us:.1f} µs")

# Prints:
#   fitted T1: 29.3 µs


# %% Ramsey fringes
@pulse.kernel
def ramsey(qudit, tau_ns: float):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    half_flip = drive_line.get_calibration(pulse.gates.X90)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    shift_frequency(drive_line.tone, 1e6)
    drive(drive_line, half_flip)
    wait(drive_line, tau_ns * pulse.ns)
    drive(drive_line, half_flip)
    sync(drive_line, readout_line)
    return readout(readout_line, measurement)


taus = np.linspace(0, 4_000, 41)
fringes = []
for tau_ns in taus:
    result = pulse.simulate(ramsey, qubit, tau_ns)
    fringes.append(result.level_population(1))

# The first revival lies between the 5th and the 15th wait.
first_peak = 5 + int(np.argmax(fringes[5:15]))
print(f"first revival at {taus[first_peak]:.0f} ns")

# Prints:
#   first revival at 1000 ns


# %% Cancel the detuning with an echo
@pulse.kernel
def echo(qudit, tau_ns: float):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    half_flip = drive_line.get_calibration(pulse.gates.X90)
    flip = drive_line.get_calibration(pulse.gates.X)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    shift_frequency(drive_line.tone, 1e6)
    drive(drive_line, half_flip)
    wait(drive_line, tau_ns / 2 * pulse.ns)
    drive(drive_line, flip)
    wait(drive_line, tau_ns / 2 * pulse.ns)
    drive(drive_line, half_flip)
    sync(drive_line, readout_line)
    return readout(readout_line, measurement)


print("wait (ns)  excited")
for tau_ns in taus[::5]:
    result = pulse.simulate(echo, qubit, tau_ns)
    excited = result.level_population(1)
    print(f"{tau_ns:9.0f}  {excited:.3f}")

# Prints:
#   wait (ns)  excited
#           0  0.022
#         500  0.031
#        1000  0.035
#        1500  0.047
#        2000  0.048
#        2500  0.055
#        3000  0.061
#        3500  0.074
#        4000  0.073
