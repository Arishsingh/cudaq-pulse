# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One QPU, Several Control Systems: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/control_systems.html

Generated from docs/tutorials/control_systems.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

# %% The QPU and a reset
import cudaq_pulse as pulse

base = pulse.quantum_system("transmon_surface17")
print(base.control_system)  # None


def cycles(ns):
    """A time in ns, in cycles of the QPU's clock."""
    return round(ns * base.clock_ghz)


print(cycles(100))  # 200


@pulse.kernel
def active_reset(qudit):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    drive(drive_line, flip)
    sync(drive_line, readout_line)
    outcome = readout(readout_line, measurement)

    sync(drive_line, readout_line)
    if outcome:
        drive(drive_line, flip)


def reset_on(control_system, schedule="alap"):
    """The reset of Z4: its length in ns, and P(0) at its end."""
    system = base.replace(control_system=control_system)
    compiled = pulse.compile(
        active_reset,
        quantum_system=system,
        schedule=schedule,
    )
    scheduled = compiled.lower(system.qudit("Z4"))
    result = pulse.simulate(scheduled, readout="model")
    ground = result.level_population(0)
    return scheduled.duration / pulse.ns, ground


# %% One controller, fast or slow feedback
fast = pulse.Controller(feedback_latency_cycles=cycles(100))
slow = pulse.Controller(feedback_latency_cycles=cycles(400))

setups = {
    "one controller, 100 ns feedback": (
        pulse.ControlSystem(controllers=[fast]),
        "alap",
    ),
    "one controller, 400 ns feedback": (
        pulse.ControlSystem(controllers=[slow]),
        "alap",
    ),
}

for name, (control_system, schedule) in setups.items():
    duration_ns, ground = reset_on(control_system, schedule)
    print(f"{name}: {duration_ns:.0f} ns, P(0) = {ground:.4f}")

# Prints:
#   one controller, 100 ns feedback: 380 ns, P(0) = 0.9943
#   one controller, 400 ns feedback: 680 ns, P(0) = 0.9836


# %% A controller per feedline
import json
from importlib.resources import files

package = files("cudaq_pulse.quantum_systems")
calibration = package / "calibrations" / "transmon_surface17.json"
readout_data = json.loads(calibration.read_text())["readout"]
groups = readout_data["feedlines"]["groups"]

index = {q.label: q.index for q in base.physical_qudits.values()}
feedlines = [
    [index[label] for label in group["qudits"]]
    for group in groups.values()
]
for group in groups.values():
    print(" ".join(group["qudits"]))

# Prints:
#   D7 D4 Z1 Z2
#   D1 D2 D5 X1 X2
#   D9 D8 X3 X4
#   D3 D6 Z3 Z4


from cudaq_pulse.controller import ReadoutLine

per_feedline = [
    pulse.Controller(
        drive_lines=group,
        readout_lines=[
            ReadoutLine(group, max_concurrent_tones=len(group))
        ],
        feedback_latency_cycles=cycles(100),
        peer_latency_cycles=cycles(200),
    )
    for group in feedlines
]
setups["a controller per feedline"] = (
    pulse.ControlSystem(
        controllers=per_feedline,
        reference_clock_ghz=0.01,
    ),
    "alap",
)

control_system, schedule = setups["a controller per feedline"]
duration_ns, ground = reset_on(control_system, schedule)
print(f"{duration_ns:.0f} ns, P(0) = {ground:.4f}")

# Prints:
#   380 ns, P(0) = 0.9943


# %% Drives and readouts on separate modules
import textwrap

from cudaq_pulse.controller import ControllerLimitError

drive_module = pulse.Controller(drive_lines=range(len(index)))
readout_module = pulse.Controller(
    readout_lines=range(len(index)),
    feedback_latency_cycles=cycles(100),
    peer_latency_cycles=cycles(200),
)
unclocked = pulse.ControlSystem(
    controllers=[drive_module, readout_module],
)

try:
    reset_on(unclocked)
except ControllerLimitError as error:
    message = str(error.violations[0])
    print(textwrap.fill(message, width=64))

# Prints:
#   controllers[0]: the sync at 80 lines up the drive on line 0 with
#   the pulses of controllers[1], but they land within 1 of its
#   cycles of where they are meant to (1 without a shared
#   reference_clock_ghz), more than its alignment_tolerance_cycles
#   of 0; share a reference_clock_ghz between them, or raise its
#   alignment_tolerance_cycles to 1


setups["drive and readout modules"] = (
    pulse.ControlSystem(
        controllers=[drive_module, readout_module],
        reference_clock_ghz=0.01,
    ),
    "alap",
)

control_system, schedule = setups["drive and readout modules"]
duration_ns, ground = reset_on(control_system, schedule)
print(f"{duration_ns:.0f} ns, P(0) = {ground:.4f}")

# Prints:
#   580 ns, P(0) = 0.9872


# %% A limited number of drive lanes
setups["six drive lanes"] = (
    pulse.ControlSystem(
        controllers=[fast],
        max_concurrent_drives=6,
        max_concurrent_readouts=8,
    ),
    "rcp",
)

control_system, schedule = setups["six drive lanes"]
duration_ns, ground = reset_on(control_system, schedule)
print(f"{duration_ns:.0f} ns, P(0) = {ground:.4f}")

# Prints:
#   380 ns, P(0) = 0.9943


# %% One round of error correction
from cudaq_pulse.quantum_systems.transmon import CZ_LAYERS


@pulse.kernel
def qec_round(qudits, cz_ns: float):
    drives = [get_drive_line(q) for q in qudits]
    readouts = [get_readout_line(q) for q in qudits]
    flips, half_flips = [], []
    for line in drives:
        flips.append(line.get_calibration(pulse.gates.X))
        half_flips.append(line.get_calibration(pulse.gates.X90))
    data = sorted({d for layer in CZ_LAYERS for _, d in layer})

    for half, layers in enumerate((CZ_LAYERS[:4], CZ_LAYERS[4:])):
        auxiliaries = sorted(
            {a for layer in layers for a, _ in layer}
        )
        # The X half also turns the data qubits to the X basis.
        turned = auxiliaries + data if half == 1 else auxiliaries
        for i in turned:
            drive(drives[i], half_flips[i])
        for step, layer in enumerate(layers):
            sync(*drives)
            for a, d in layer:
                wait(drives[a], cz_ns * pulse.ns)
                wait(drives[d], cz_ns * pulse.ns)
            if step == 1:
                # An echo on the data qubits, halfway through.
                sync(*drives)
                for i in data:
                    drive(drives[i], flips[i])
        sync(*drives)
        for i in turned:
            drive(drives[i], half_flips[i])

    auxiliaries = sorted(
        {a for layer in CZ_LAYERS for a, _ in layer}
    )
    sync(*drives, *readouts)
    outcomes = []
    for i in auxiliaries:
        line = readouts[i]
        measurement = line.get_calibration(pulse.gates.READOUT)
        outcomes.append(readout(line, measurement))
    sync(*drives, *readouts)
    for i, outcome in zip(auxiliaries, outcomes):
        if outcome:
            drive(drives[i], flips[i])


cz_ns = max(
    coupling.get_calibration(pulse.gates.CZ).duration_ns
    for coupling in base.couplings
)


def round_on(control_system, schedule="alap"):
    """The length in ns of one round."""
    system = base.replace(control_system=control_system)
    compiled = pulse.compile(
        qec_round,
        quantum_system=system,
        schedule=schedule,
    )
    scheduled = compiled.lower(system.qudits[: len(index)], cz_ns)
    return scheduled.duration / pulse.ns


for name, (control_system, schedule) in setups.items():
    print(f"{name}: {round_on(control_system, schedule):.0f} ns")

# Prints:
#   one controller, 100 ns feedback: 1324 ns
#   one controller, 400 ns feedback: 1624 ns
#   a controller per feedline: 1324 ns
#   drive and readout modules: 1524 ns
#   six drive lanes: 1604 ns


# %% What a longer round costs
@pulse.kernel
def hold(qudit, half_ns: float):
    drive_line = get_drive_line(qudit)
    half_flip = drive_line.get_calibration(pulse.gates.X90)
    flip = drive_line.get_calibration(pulse.gates.X)

    drive(drive_line, half_flip)
    wait(drive_line, half_ns * pulse.ns)
    drive(drive_line, flip)
    wait(drive_line, half_ns * pulse.ns)
    drive(drive_line, half_flip)


import numpy as np

data = [
    base.qudit(q.label)
    for q in base.physical_qudits.values()
    if q.label.startswith("D")
]
ground_state = np.array([1.0, 0.0, 0.0])


def idle_error(round_ns):
    """The mean error of a data qubit held for one round."""
    errors = []
    for qudit in data:
        flip = qudit.get_calibration(pulse.gates.X)
        wait_ns = max(round_ns - 3 * flip.duration_ns, 0.0)
        result = pulse.simulate(
            hold,
            qudit,
            wait_ns / 2,
            dephasing="t2_echo",
            initial_state=ground_state,
        )
        errors.append(1 - result.level_population(0))
    return sum(errors) / len(errors)


print(f"{idle_error(0):.2%}")  # 0.10%


# %% Side by side
rows = []
for name, (control_system, schedule) in setups.items():
    reset_ns, ground = reset_on(control_system, schedule)
    round_ns = round_on(control_system, schedule)
    error = idle_error(round_ns)
    rows.append((name, reset_ns, ground, round_ns, error))

header = ("", "reset", "P(0)", "round", "idle")
print("{:31}  {:>6}  {:>6}  {:>7}  {:>5}".format(*header))
for name, reset_ns, ground, round_ns, error in rows:
    print(
        f"{name:31}  {reset_ns:3.0f} ns  {ground:.4f}  "
        f"{round_ns:4.0f} ns  {error:.2%}"
    )

# Prints:
#                                     reset    P(0)    round   idle
#   one controller, 100 ns feedback  380 ns  0.9943  1324 ns  1.23%
#   one controller, 400 ns feedback  680 ns  0.9836  1624 ns  1.51%
#   a controller per feedline        380 ns  0.9943  1324 ns  1.23%
#   drive and readout modules        580 ns  0.9872  1524 ns  1.42%
#   six drive lanes                  380 ns  0.9943  1604 ns  1.50%
