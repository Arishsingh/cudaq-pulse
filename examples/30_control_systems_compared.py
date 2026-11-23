# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One QPU under five control systems, side by side.

The QPU is the built-in transmon_surface17. Each control system is defined
here and attached with ``base.replace(control_system=...)``:

  - one controller for every line, with 100 ns of feedback latency
  - the same with 400 ns
  - a controller per readout feedline, 200 ns apart, on one reference clock
  - one module for the drives and one for the readouts, 200 ns apart, on
    one reference clock (without the clock, the compiler refuses the
    program and says why)
  - one controller that runs at most six drives at once

Two programs are compiled on each: an active reset of Z4 (X, readout, X
if it read 1) and one round of the surface code (the QPU's eight CZ
steps, then a readout and an active reset of the eight auxiliary qubits).
The table gives the length of each schedule, the ground-state population
after the reset, and the error a data qubit picks up while it waits for
one round.

That last number is not a logical error rate. Flux lines are not modeled
yet, so a CZ is a slot in which its qubits wait, and a simulated round
measures no stabilizer. The idle error is what the simulator does give:
a data qubit in a superposition, held for the round's length with one
echo, and brought back.
"""

import json
import textwrap
from importlib.resources import files

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.controller import ControllerLimitError, ReadoutLine
from cudaq_pulse.quantum_systems.transmon import CZ_LAYERS

base = pulse.quantum_system("transmon_surface17")  # the QPU; no control system
index = {q.label: q.index for q in base.physical_qudits.values()}


def cycles(ns: float) -> int:
    """A time in ns, in cycles of the QPU's clock."""
    return round(ns * base.clock_ghz)


@pulse.kernel
def active_reset(qudit):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)

    drive(drive_line, flip)
    sync(drive_line, readout_line)
    outcome = readout(readout_line, readout_line.get_calibration(pulse.gates.READOUT))
    sync(drive_line, readout_line)
    if outcome:
        drive(drive_line, flip)


@pulse.kernel
def qec_round(qudits, cz_ns: float):
    drives = [get_drive_line(q) for q in qudits]
    readouts = [get_readout_line(q) for q in qudits]
    flips = [line.get_calibration(pulse.gates.X) for line in drives]
    half_flips = [line.get_calibration(pulse.gates.X90) for line in drives]
    data = sorted({d for layer in CZ_LAYERS for _, d in layer})

    # Four CZ steps for the Z stabilizers, then four for the X stabilizers.
    for half, layers in enumerate((CZ_LAYERS[:4], CZ_LAYERS[4:])):
        auxiliaries = sorted({a for layer in layers for a, _ in layer})
        # The X half also turns the data qubits to the X basis.
        turned = auxiliaries + data if half == 1 else auxiliaries
        for i in turned:
            drive(drives[i], half_flips[i])
        for step, layer in enumerate(layers):
            sync(*drives)
            for a, d in layer:
                # A CZ is a flux pulse, which is not modeled: both qubits wait.
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

    # Read the auxiliary qubits, and flip back each one that read 1.
    auxiliaries = sorted({a for layer in CZ_LAYERS for a, _ in layer})
    sync(*drives, *readouts)
    outcomes = []
    for i in auxiliaries:
        measurement = readouts[i].get_calibration(pulse.gates.READOUT)
        outcomes.append(readout(readouts[i], measurement))
    sync(*drives, *readouts)
    for i, outcome in zip(auxiliaries, outcomes, strict=True):
        if outcome:
            drive(drives[i], flips[i])


@pulse.kernel
def hold(qudit, half_ns: float):
    drive_line = get_drive_line(qudit)
    half_flip = drive_line.get_calibration(pulse.gates.X90)

    drive(drive_line, half_flip)
    wait(drive_line, half_ns * pulse.ns)
    drive(drive_line, drive_line.get_calibration(pulse.gates.X))
    wait(drive_line, half_ns * pulse.ns)
    drive(drive_line, half_flip)


# The qudits on each readout feedline, from the QPU's calibration file.
calibration = files("cudaq_pulse.quantum_systems") / "calibrations" / "transmon_surface17.json"
groups = json.loads(calibration.read_text())["readout"]["feedlines"]["groups"]
feedlines = [[index[label] for label in group["qudits"]] for group in groups.values()]

fast = pulse.Controller(feedback_latency_cycles=cycles(100))
slow = pulse.Controller(feedback_latency_cycles=cycles(400))
per_feedline = [
    pulse.Controller(
        drive_lines=group,
        readout_lines=[ReadoutLine(group, max_concurrent_tones=len(group))],
        feedback_latency_cycles=cycles(100),
        peer_latency_cycles=cycles(200),
    )
    for group in feedlines
]
drive_module = pulse.Controller(drive_lines=range(len(index)))
readout_module = pulse.Controller(
    readout_lines=range(len(index)),
    feedback_latency_cycles=cycles(100),
    peer_latency_cycles=cycles(200),
)

# Each setup: its control system, and the schedule policy to compile with.
# "rcp" keeps to a limit on the drives and readouts that run at once.
setups = {
    "one controller, 100 ns feedback": (pulse.ControlSystem(controllers=[fast]), "alap"),
    "one controller, 400 ns feedback": (pulse.ControlSystem(controllers=[slow]), "alap"),
    "a controller per feedline": (
        pulse.ControlSystem(controllers=per_feedline, reference_clock_ghz=0.01),
        "alap",
    ),
    "drive and readout modules": (
        pulse.ControlSystem(controllers=[drive_module, readout_module], reference_clock_ghz=0.01),
        "alap",
    ),
    "six drive lanes": (
        pulse.ControlSystem(controllers=[fast], max_concurrent_drives=6, max_concurrent_readouts=8),
        "rcp",
    ),
}

cz_ns = max(c.get_calibration(pulse.gates.CZ).duration_ns for c in base.couplings)
data_qudits = [base.qudit(label) for label in index if label.startswith("D")]
ground_state = np.array([1.0, 0.0, 0.0])


def idle_error(round_ns: float) -> float:
    """The mean error of a data qubit held for *round_ns*, pulses included."""
    errors = []
    for qudit in data_qudits:
        wait_ns = max(round_ns - 3 * qudit.get_calibration(pulse.gates.X).duration_ns, 0.0)
        result = pulse.simulate(
            hold, qudit, wait_ns / 2, dephasing="t2_echo", initial_state=ground_state
        )
        errors.append(1 - result.level_population(0))
    return sum(errors) / len(errors)


print(f"{'':31}  {'reset':>6}  {'P(0)':>6}  {'round':>7}  {'idle':>5}")
for name, (control_system, schedule) in setups.items():
    system = base.replace(control_system=control_system)
    reset = pulse.compile(active_reset, quantum_system=system, schedule=schedule)
    reset = reset.lower(system.qudit("Z4"))
    ground = pulse.simulate(reset, readout="model").level_population(0)
    cycle = pulse.compile(qec_round, quantum_system=system, schedule=schedule)
    round_ns = cycle.lower(system.qudits[: len(index)], cz_ns).duration / pulse.ns
    print(
        f"{name:31}  {reset.duration / pulse.ns:3.0f} ns  {ground:.4f}  "
        f"{round_ns:4.0f} ns  {idle_error(round_ns):.2%}"
    )
print(f"the three pulses of the hold alone, with no wait: {idle_error(0):.2%}")

# Two modules without a shared reference clock: the compiler refuses the
# reset, because its sync cannot line the two boxes up, and says what to do.
unclocked = base.replace(
    control_system=pulse.ControlSystem(controllers=[drive_module, readout_module])
)
try:
    pulse.compile(active_reset, quantum_system=unclocked).lower(unclocked.qudit("Z4"))
except ControllerLimitError as error:
    print("\nwithout a reference clock, the drive and readout modules are refused:")
    print(textwrap.fill(str(error.violations[0]), width=88, initial_indent="  "))
