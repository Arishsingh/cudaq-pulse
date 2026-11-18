# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe Your Own Quantum System: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/own_quantum_system.html

Generated from docs/tutorials/own_quantum_system.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

# %% Qudits and a coupling
import json
from pathlib import Path

import cudaq_pulse as pulse
from cudaq_pulse.quantum_systems import (
    Coupling,
    DragPulse,
    PhysicalQudit,
)

x_pulse = {pulse.gates.X: DragPulse(duration_ns=40.0, sigma_ns=10.0)}

qudit_0 = PhysicalQudit(
    index=0,
    frequency_hz=5.1e9,
    t1_us=60.0,
    t2_star_us=40.0,
    anharmonicity_hz=-2.2e8,
    calibrations=x_pulse,
)
qudit_1 = PhysicalQudit(
    index=1,
    frequency_hz=5.3e9,
    t1_us=45.0,
    t2_star_us=30.0,
    anharmonicity_hz=-2.1e8,
    calibrations=x_pulse,
)


pair = pulse.quantum_system(
    name="my_pair",
    qudits={0: qudit_0, 1: qudit_1},
    couplings=[Coupling((0, 1), strength_hz=3.0e6)],
)
print(pair)

# Prints:
#   <QuantumSystem my_pair (2 qudits)>


# %% Calibrated pulses
x_cal = pair.qudits[0].get_calibration(pulse.gates.X)
print(f"{x_cal.duration_ns} ns DRAG, amplitude {x_cal.amplitude:.4f}")

# Prints:
#   40.0 ns DRAG, amplitude 0.1313


# %% Run a kernel on it
@pulse.kernel
def excite(qudit):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)

    drive(drive_line, flip)
    sync(drive_line, readout_line)
    return readout(readout_line, square(300 * pulse.ns, 0.05))


for qudit in pair.qudits:
    result = pulse.simulate(excite, qudit)
    excited = result.level_populations(qudit)[1]
    print(f"qudit {qudit.index} excited: {excited:.3f}")

print("compilations:", excite.cache_info().compiled)

# Prints:
#   qudit 0 excited: 0.993
#   qudit 1 excited: 0.991
#   compilations: 2


# %% One compile for every qudit
@pulse.kernel
def excite_any(qudit: pulse.Qudit, amplitude: float):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = gaussian(
        duration=40 * pulse.ns,
        amplitude=amplitude,
        sigma=10 * pulse.ns,
    )
    drive(drive_line, flip)
    sync(drive_line, readout_line)
    return readout(readout_line, square(300 * pulse.ns, 0.05))


for qudit in pair.qudits:
    amplitude = qudit.get_calibration(pulse.gates.X).amplitude
    result = pulse.simulate(excite_any, qudit, amplitude)
    excited = result.level_populations(qudit)[1]
    print(f"qudit {qudit.index} excited: {excited:.3f}")

print("compilations:", excite_any.cache_info().compiled)

# Prints:
#   qudit 0 excited: 0.993
#   qudit 1 excited: 0.991
#   compilations: 1


# %% Follow a recalibration
old_qudit = pair.physical_qudits[0]
new_qudit = old_qudit.replace(frequency_hz=5.102e9)
qudits = {**pair.physical_qudits, 0: new_qudit}
recalibrated = pair.replace(qudits=qudits)

systems = {"before": pair, "after": recalibrated}
for label, quantum_system in systems.items():
    scheduled = excite.lower(quantum_system.qudits[0])
    program = scheduled.export_to_controller()
    for event in program["events"]:
        if event["kind"] == "drive":
            frequency_ghz = event["frequency_hz"] / 1e9
            print(f"{label:6} drive at {frequency_ghz:.3f} GHz")

# Prints:
#   before drive at 5.100 GHz
#   after  drive at 5.102 GHz


# %% Save it to a file
path = Path("my_pair.json")
text = json.dumps(pair.as_dict(), indent=2)
path.write_text(text)

loaded = pulse.QuantumSystem.load(path)
print("same system:", loaded == pair)
print("format:", json.loads(text)["format"])

# Prints:
#   same system: True
#   format: cudaq-pulse.system/1


# %% Mistakes, all at once
try:
    pulse.quantum_system("transmon_surface17", layout=[0, 99, 99])
except pulse.QuantumSystemDefinitionError as error:
    for problem in error.problems:
        print(f"{problem.path}: {problem.message}")

# Prints:
#   layout[1]: carrier 99 is not in quantum system 'transmon_surface17' (it has 0..16)
#   layout[2]: carrier 99 is not in quantum system 'transmon_surface17' (it has 0..16)
