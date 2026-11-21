# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Getting More out of Your Controller's Instruction Set: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/isa_limits.html

Generated from docs/tutorials/isa_limits.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

import sys
from pathlib import Path


# The page imports examples/isa_lowering/toy_isa.py. Run this script from a
# clone of the repository, or save that file beside it.
_EXAMPLES = Path(__file__).resolve().parent.parent.parent.parent / "examples"
sys.path.insert(0, str(_EXAMPLES / "isa_lowering"))


# %% Set up
import dataclasses
import textwrap

import cudaq_pulse as pulse
from cudaq_pulse.controller import ControllerLimitError
from toy_isa import lower, toy_control_system, toy_qpu

qpu = toy_qpu()
first, second = toy_control_system().controllers
q0, q1, q2 = qpu.qudits
print("model:           ", first.model)
print("feedback latency:", first.feedback_latency_cycles, "cycles")

# Prints:
#   model:            toy.sequencer
#   feedback latency: 80 cycles


def show(error):
    message = str(error.violations[0])
    for line in textwrap.wrap(message, width=64):
        print(line)


# %% A branch on another controller's readout
@pulse.kernel
def flip_other(measured, other):
    readout_line = get_readout_line(measured)
    drive_line = get_drive_line(other)
    probe = square(duration=200, amplitude=0.05)
    flip = gaussian(duration=40, amplitude=0.5, sigma=10.0)

    outcome = readout(readout_line, probe)
    sync(readout_line, drive_line)
    if outcome:
        drive(drive_line, flip)


# %% The default schedule waits for it
compiled = pulse.compile(flip_other, quantum_system=qpu)
split = compiled.lower(q0, q2)

part = split.export_to_controller(controller=0)
for line in str(lower(part)).splitlines():
    print(line)

# Prints:
#   .chan ch0.0 readout q0
#   .wave w0 200
#   set_freq ch0.0 5000000000
#   acquire ch0.0 w0 0.05 m0


part = split.export_to_controller(controller=1)
for line in str(lower(part)).splitlines():
    print(line)

# Prints:
#   .chan ch0 drive q2
#   .wave w0 40
#   wait 80
#   branch !m0 end0
#   set_freq ch0 5600000000
#   play ch0 w0 0.5
#   end0:


# %% A controller that can't forward outcomes
local_only = dataclasses.replace(
    first,
    supports=["play", "readout", "wait", "sync", "branch"],
)
system = pulse.ControlSystem(
    controllers=[local_only, second],
    reference_clock_ghz=0.25,
)
compiled = pulse.compile(
    flip_other,
    quantum_system=qpu,
    control_system=system,
)


try:
    compiled.lower(q0, q2)
except ControllerLimitError as error:
    show(error)

# Prints:
#   controllers[0] (toy.sequencer): the readout on line 0 (qudit 0)
#   at cycle 0 needs 'forward_outcome', which the controller does
#   not support; run that part of the program on a controller that
#   supports it, or write it without 'forward_outcome'


# %% A pulse on each controller
@pulse.kernel
def joint(a, b):
    first_line = get_drive_line(a)
    second_line = get_drive_line(b)
    x90 = gaussian(duration=40, amplitude=0.25, sigma=10.0)
    flat = square(duration=40, amplitude=0.1)

    drive(first_line, x90)
    sync(first_line, second_line)
    drive(first_line, flat)
    drive(second_line, flat)


# %% Without a shared clock
unclocked = pulse.ControlSystem(controllers=[first, second])
compiled = pulse.compile(
    joint,
    quantum_system=qpu,
    control_system=unclocked,
)
try:
    compiled.lower(q0, q2)
except ControllerLimitError as error:
    show(error)

# Prints:
#   controllers[0] (toy.sequencer): the sync at 40 lines up the
#   drive on line 0 with the pulses of controllers[1]
#   (toy.sequencer), but they land within 1 of its cycles of where
#   they are meant to (1 without a shared reference_clock_ghz), more
#   than its alignment_tolerance_cycles of 0; share a
#   reference_clock_ghz between them, or raise its
#   alignment_tolerance_cycles to 1


# %% With the QPU's clock
together = pulse.compile(joint, quantum_system=qpu).lower(q0, q2)

for number in range(2):
    part = together.export_to_controller(controller=number)
    print(f"controller {number}:")
    for event in part["events"]:
        print(f"  {event['index']}  {event['kind']:<5} "
              f"at {event['start_cycles']}")

# Prints:
#   controller 0:
#     0  drive at 0
#     1  sync  at 40
#     2  drive at 40
#   controller 1:
#     1  sync  at 40
#     3  drive at 40


# %% A pulse train
@pulse.kernel
def train(qudit):
    drive_line = get_drive_line(qudit)
    x90 = gaussian(duration=40, amplitude=0.25, sigma=10.0)
    for _ in range(1000):
        drive(drive_line, x90)
        wait(drive_line, 8)


# %% On a controller without loops
try:
    pulse.compile(train, quantum_system=qpu).lower(q0)
except ControllerLimitError as error:
    show(error)

# Prints:
#   controllers[0] (toy.sequencer): 1000 drives and readouts; the
#   controller plays at most 256. It does not support 'loop', so
#   rolled loops are unrolled for it: the loop of 1000 iterations
#   from cycle 0 plays 999 more


# %% On a controller with loops
capabilities = list(first.supports or [])
capabilities.append("loop")
looping = dataclasses.replace(first, supports=capabilities)
compiled = pulse.compile(
    train,
    quantum_system=qpu,
    control_system=looping,
)
rolled = compiled.lower(q0)

flat = rolled.export_to_controller()
print(len(flat["events"]), "events")

# Prints:
#   2000 events


structured = rolled.export_to_controller(version=2)
loop = structured["events"][0]

print(loop["kind"], loop["trips"], "trips")
print(loop["period_cycles"], "cycles each")

for event in loop["body"]:
    print(" ", event["kind"])

# Prints:
#   loop 1000 trips
#   48 cycles each
#     play
#     wait


# %% The decoder
def parity(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0] ^ bits[1]]


decode = pulse.classical(
    parity,
    link="decoder",
    max_latency_ns=100,
)


# %% A round with a deadline
@pulse.kernel
def qec_round(data, ancilla):
    probe = square(duration=200, amplitude=0.05)
    flip = gaussian(duration=40, amplitude=0.5, sigma=10.0)

    syndrome_0 = readout(get_readout_line(data), probe)
    syndrome_1 = readout(get_readout_line(ancilla), probe)
    request = decode.submit([syndrome_0, syndrome_1])
    in_time, fix = request.result(timeout=60 * pulse.ns)

    if fix[0]:
        drive(get_drive_line(data), flip)
    if not in_time:
        drive(get_drive_line(ancilla), flip)


# %% The listing
compiled = pulse.compile(
    qec_round,
    quantum_system=qpu,
)
qec = compiled.lower(q0, q1)
listing = lower(qec.export_to_controller())

for line in listing.code:
    print(line)

# Prints:
#   set_freq ch0.0 5000000000
#   acquire ch0.0 w0 0.05 m0
#   set_freq ch0.1 5300000000
#   acquire ch0.1 w0 0.05 m1
#   wait 70
#   submit L0 m0 m1
#   wait 18
#   poll L0 a3
#   branch !a3 end0
#   set_freq ch1 5000000000
#   play ch1 w1 0.5
#   end0:
#   branch a3.ok end1
#   set_freq ch2 5300000000
#   play ch2 w1 0.5
#   end1:


for event in qec.export_to_controller()["events"]:
    for condition in event["conditions"]:
        line = event["line"]
        answer = condition["answers"][0]
        result = condition["result"]
        print(f"line {line}: answer {answer}, result {result}")

# Prints:
#   line 2: answer 3, result 1
#   line 3: answer 3, result 0


# %% A decoder between two controllers
@pulse.kernel
def correct_across(ancilla_0, ancilla_1, data):
    probe = square(duration=200, amplitude=0.05)
    flip = gaussian(duration=40, amplitude=0.5, sigma=10.0)

    syndrome_0 = readout(get_readout_line(ancilla_0), probe)
    syndrome_1 = readout(get_readout_line(ancilla_1), probe)
    request = decode.submit([syndrome_0, syndrome_1])
    in_time, fix = request.result(timeout=60 * pulse.ns)

    if fix[0]:
        drive(get_drive_line(data), flip)
    if not in_time:
        drive(get_drive_line(ancilla_1), flip)


compiled = pulse.compile(
    correct_across,
    quantum_system=qpu,
)
across = compiled.lower(q0, q1, q2)

for number in range(2):
    part = across.export_to_controller(controller=number)
    print(f"; controller {number}")
    for line in lower(part).code:
        print(line)

# Prints:
#   ; controller 0
#   set_freq ch0.0 5000000000
#   acquire ch0.0 w0 0.05 m0
#   set_freq ch0.1 5300000000
#   acquire ch0.1 w0 0.05 m1
#   wait 70
#   submit L0 m0 m1
#   wait 18
#   poll L0 a3
#   branch a3.ok end1
#   set_freq ch1 5300000000
#   play ch1 w1 0.5
#   end1:
#   ; controller 1
#   wait 88
#   poll L0 a3
#   branch !a3 end0
#   set_freq ch0 5600000000
#   play ch0 w0 0.5
#   end0:


# %% A sweep is one table entry
@pulse.kernel
def rabi(qudit, amplitude: float):
    shape = gaussian(
        duration=40,
        amplitude=amplitude,
        sigma=10.0,
    )
    drive(get_drive_line(qudit), shape)


sweep = pulse.compile(rabi, quantum_system=qpu)
for amplitude in (0.2, 0.4):
    program = sweep.lower(q0, amplitude).export_to_controller()
    print(lower(program).code[-1])

# Prints:
#   play ch0 w0 0.2
#   play ch0 w0 0.4


# %% New shapes and new frequencies
@pulse.kernel
def widths(qudit):
    drive_line = get_drive_line(qudit)
    for k in range(30):
        sigma = 4.0 + 0.5 * k
        shape = gaussian(duration=40, amplitude=0.25, sigma=sigma)
        drive(drive_line, shape)


@pulse.kernel
def hops(qudit):
    drive_line = get_drive_line(qudit)
    x90 = gaussian(duration=40, amplitude=0.25, sigma=10.0)
    for _ in range(3):
        drive(drive_line, x90)
        shift_frequency(drive_line.tone, 1.0e6)


for kernel in (widths, hops):
    try:
        pulse.compile(kernel, quantum_system=qpu).lower(q0)
    except ControllerLimitError as error:
        show(error)

# Prints:
#   controllers[0] (toy.sequencer): line 0: its waveforms take 1200
#   samples; the controller's waveform memory holds 1024
#   controllers[0] (toy.sequencer): line 0: 3 tone frequencies; the
#   controller holds 2 per line
