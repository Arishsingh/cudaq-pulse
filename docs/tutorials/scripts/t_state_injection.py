# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T-State Injection and a Heralded Repeat-Until-Success Loop: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/t_state_injection.html

Generated from docs/tutorials/t_state_injection.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

# %% The T gate is a frame change
import math

import numpy as np

import cudaq_pulse as pulse

system = pulse.quantum_system("transmon_surface17")
qubit = system.qudit("D5")

t_gate = qubit.get_calibration(pulse.gates.T)
print(t_gate)

# Prints:
#   VirtualZ(phase_rad=0.7853981633974483)


@pulse.kernel
def inject(qudit):
    drive_line = get_drive_line(qudit)
    y90 = drive_line.get_calibration(pulse.gates.Y90)
    t_gate = drive_line.get_calibration(pulse.gates.T)

    # |0> to |+>, then |+> to |T>.
    drive(drive_line, y90)
    drive(drive_line, t_gate)


print(f"{'kind':8} {'start':>6} {'length':>6}")
for event in inject.lower(qubit).schedule:
    start = event.start_cycles
    length = event.duration_cycles
    print(f"{event.kind:8} {start:>6} {length:>6}")

# Prints:
#   kind      start length
#   drive         0     80


# %% Reading the state in the frame
@pulse.kernel
def plus(qudit):
    drive_line = get_drive_line(qudit)
    y90 = drive_line.get_calibration(pulse.gates.Y90)
    drive(drive_line, y90)


ground = [1.0, 0.0, 0.0]

with_t = pulse.simulate(inject, qubit, initial_state=ground)
without_t = pulse.simulate(plus, qubit, initial_state=ground)

same = np.allclose(with_t.density_matrix, without_t.density_matrix)
print("same state with and without T:", same)

# Prints:
#   same state with and without T: True


# %% Tomography
X_AXIS, Y_AXIS, Z_AXIS = 0, 1, 2


@pulse.kernel
def t_tomography(qudit, axis: int):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    y90 = drive_line.get_calibration(pulse.gates.Y90)
    t_gate = drive_line.get_calibration(pulse.gates.T)
    x90 = drive_line.get_calibration(pulse.gates.X90)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    drive(drive_line, y90)
    drive(drive_line, t_gate)

    if axis == X_AXIS:
        drive(drive_line, y90.replace(amplitude=-y90.amplitude))
    elif axis == Y_AXIS:
        drive(drive_line, x90)

    sync(drive_line, readout_line)
    readout(readout_line, measurement)


# %% From readouts to fidelity
def expectation(result):
    """<Z> of the last readout, from its outcomes' probabilities."""
    excited = 0.0
    for branch in result.branches:
        if branch.levels[-1] > 0:
            excited += branch.probability
    return 1 - 2 * excited


T_BLOCH = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)


def fidelity(bloch):
    """<T|rho|T> for the Bloch vector, at most length 1."""
    length = max(1.0, float(np.linalg.norm(bloch)))
    unit = bloch / length
    return (1 + float(unit @ T_BLOCH)) / 2


bloch = np.zeros(3)
for axis in (X_AXIS, Y_AXIS, Z_AXIS):
    result = pulse.simulate(
        t_tomography,
        qubit,
        axis,
        initial_state=ground,
    )
    bloch[axis] = expectation(result)

print(f"fidelity with |T>: {fidelity(bloch):.2f}")
print(f"infidelity < 1e-3: {1 - fidelity(bloch) < 1e-3}")

# Prints:
#   fidelity with |T>: 1.00
#   infidelity < 1e-3: True


# %% The thermal start
excited = system.thermal_populations[qubit.index]
print(f"excited at the start: {excited:.3f}")

bloch = np.zeros(3)
for axis in (X_AXIS, Y_AXIS, Z_AXIS):
    result = pulse.simulate(t_tomography, qubit, axis)
    bloch[axis] = expectation(result)
print(f"fidelity with |T>:    {fidelity(bloch):.2f}")

# Prints:
#   excited at the start: 0.027
#   fidelity with |T>:    0.97


# %% A herald on the controller
from cudaq_pulse.classical import Link

controller_link = Link(fixed_ns=4, feedback_ns=8, slots=2)
links = {"controller": controller_link}
control_system = pulse.ControlSystem(links=links)


@pulse.classical(
    min_latency_ns=20,
    nominal_latency_ns=30,
    max_latency_ns=40,
    link="controller",
)
def herald(outcome: pulse.Bits) -> bool:
    """Accept the shot when the qubit read 0."""
    return outcome[0] == 0


# %% The kernel
@pulse.kernel
def heralded_t(qudit, attempts: int, axis: int):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    for _ in range(attempts):
        outcome = readout(readout_line, measurement)
        request = herald.submit([outcome])
        accepted = request.result()
        sync(drive_line, readout_line)
        if not accepted:
            drive(drive_line, flip)
        sync(drive_line, readout_line)

    # Prepare |T>, then turn the measured axis onto Z.
    y90 = drive_line.get_calibration(pulse.gates.Y90)
    t_gate = drive_line.get_calibration(pulse.gates.T)
    x90 = drive_line.get_calibration(pulse.gates.X90)
    drive(drive_line, y90)
    drive(drive_line, t_gate)
    if axis == X_AXIS:
        drive(drive_line, y90.replace(amplitude=-y90.amplitude))
    elif axis == Y_AXIS:
        drive(drive_line, x90)
    sync(drive_line, readout_line)
    readout(readout_line, measurement)


compiled = pulse.compile(
    heralded_t,
    quantum_system=system,
    control_system=control_system,
)


# %% Post-select on the answer
def herald_accepted(branch):
    return branch.answers("herald")[-1]


def heralded_fidelity(attempts):
    bloch = np.zeros(3)
    acceptance = 1.0
    for axis in (X_AXIS, Y_AXIS, Z_AXIS):
        program = compiled.lower(qubit, attempts, axis)
        result = pulse.simulate(program)
        kept = result.postselect(herald_accepted)
        bloch[axis] = expectation(kept)
        acceptance = kept.acceptance
    return fidelity(bloch), acceptance


heralded, acceptance = heralded_fidelity(1)
print(f"shots kept:        {acceptance:.3f}")
print(f"fidelity with |T>: {heralded:.2f}")
print(f"infidelity < 1e-3: {1 - heralded < 1e-3}")

# Prints:
#   shots kept:        0.972
#   fidelity with |T>: 1.00
#   infidelity < 1e-3: True


# %% Repeat until success
print("attempts   kept  infidelity < 1e-3")
for attempts in (1, 2, 3):
    heralded, acceptance = heralded_fidelity(attempts)
    good = 1 - heralded < 1e-3
    print(f"{attempts:8}  {acceptance:.3f}  {good}")

# Prints:
#   attempts   kept  infidelity < 1e-3
#          1  0.972  True
#          2  0.999  True
#          3  0.999  True


# %% When the answers are due
program = compiled.lower(qubit, 1, Z_AXIS)

print("function  sent (ns)  due (ns)  slack")
for call in program.classical_report():
    sent_ns = call.submit_cycles / system.clock_ghz
    due_ns = call.due_cycles / system.clock_ghz
    slack = call.slack_cycles
    print(f"{call.function:8}  {sent_ns:9}  {due_ns:8}  {slack:5}")

# Prints:
#   function  sent (ns)  due (ns)  slack
#   herald        300.0     352.0      0
