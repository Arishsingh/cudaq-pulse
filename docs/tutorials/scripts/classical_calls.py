# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calling Classical Code During Quantum Runtime: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/classical_calls.html

Generated from docs/tutorials/classical_calls.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

# %% Declare the function
import cudaq_pulse as pulse


@pulse.classical(
    link="gpu",
    min_latency_ns=2_900,
    nominal_latency_ns=3_800,
    max_latency_ns=4_500,
)
def decode(syndrome: pulse.Bits) -> bool:
    return bool(syndrome[0])


# %% Call it with a deadline
@pulse.kernel
def correct_if_needed(qudit):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    outcome = readout(readout_line, measurement)
    request = decode.submit([outcome])
    in_time, answer = request.result(timeout=4 * pulse.us)

    needs_flip = answer if in_time else outcome
    sync(drive_line, readout_line)
    if needs_flip:
        drive(drive_line, flip)


# %% Check the timing before anything runs
system = pulse.quantum_system("transmon_surface17")
qubit = system.qudits[0]

gpu_link = pulse.Link(feedback_ns=50)
control_system = pulse.ControlSystem(links={"gpu": gpu_link})

compiled = pulse.compile(
    correct_if_needed,
    control_system=control_system,
)
scheduled = compiled.lower(qubit)


call = scheduled.classical_report()[0]
submit_ns = call.submit_cycles / system.clock_ghz
due_ns = call.due_cycles / system.clock_ghz

print(f"function:      {call.function} on {call.link_name}")
print(f"request sent:  {submit_ns} ns")
print(f"answer due:    {due_ns} ns")
print(f"slack:         {call.slack_cycles} cycles")

# Prints:
#   function:      decode on gpu
#   request sent:  300.0 ns
#   answer due:    4350.5 ns
#   slack:         0 cycles


# %% Simulate the loop
equal_superposition = [2**-0.5, 2**-0.5, 0.0]
result = pulse.simulate(
    scheduled,
    initial_state=equal_superposition,
    latency_branching=True,
)

in_time = 0.0
for branch in result.branches:
    if branch.calls[0].in_time:
        in_time += branch.probability

ground = result.level_population(0)
print(f"answers in time:    {in_time:.3f}")
print(f"ground state after: {ground:.3f}")

# Prints:
#   answers in time:    0.777
#   ground state after: 0.933
