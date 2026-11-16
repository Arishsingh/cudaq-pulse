# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T-state injection and a heralded repeat-until-success loop.

A fault-tolerant computer makes its T gates by consuming magic states,
|T> = T|+> = (|0> + e^{i pi/4}|1>) / sqrt(2), which it prepares physically,
injects into a code and then distills (S. Bravyi and A. Kitaev, PRA 71,
022316 (2005); D. Litinski, Quantum 3, 205 (2019)) or cultivates (C. Gidney,
N. Shutty and C. Jones, arXiv:2409.17595 (2024)). The better the physical
state, the cheaper the rest. On a transmon the T gate itself is a virtual Z,
``pulse.gates.T``: a phase shift of the drive tone's frame, which takes no
time and adds no error. What limits a physical |T> is everything else: the
starting state, the pi/2 pulse, decoherence and the readout that checks it.

On qudit D5 of transmon_surface17 this example

1. injects |T> with a Y90 pulse and ``pulse.gates.T``, and checks that one
   X90 about the axis at 3 pi / 4 makes the same state;
2. measures it by three-axis tomography, from |0> and from the thermal
   state the model starts in, from the exact branch probabilities and from
   sampled shots through the readout model;
3. heralds the start: a readout, a ``@pulse.classical`` decision on the
   controller, and a pi pulse on reject, repeated a fixed number of times,
   then post-selected on the last answer with ``result.postselect``;
4. prints when the herald's answers are due and their slack.

Every number it prints is simulated, on a laptop CPU in a few seconds.
"""

import math
import time

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.classical import Link

qs = pulse.quantum_system("transmon_surface17")
D5 = qs.qudit("D5")
T_BLOCH = np.array([1 / math.sqrt(2), 1 / math.sqrt(2), 0.0])
# The Bloch axes tomography measures: a pulse after the state turns each onto Z.
X_AXIS, Y_AXIS, Z_AXIS = 0, 1, 2


def ground(levels=3):
    psi = np.zeros(levels, dtype=complex)
    psi[0] = 1.0
    return psi


def fidelity(bloch):
    """<T|rho|T> of a qubit state given by its Bloch vector, shortened to
    the unit ball if sampling put it outside."""
    bloch = bloch / max(1.0, float(np.linalg.norm(bloch)))
    return 0.5 * (1.0 + float(bloch @ T_BLOCH))


# ── 1. Injection ──────────────────────────────────────────────────────────


@pulse.kernel
def inject(q):
    d = get_drive_line(q)
    drive(d, d.get_calibration(pulse.gates.Y90))  # |0> to |+>
    drive(d, d.get_calibration(pulse.gates.T))  # |+> to |T>: a frame change


@pulse.kernel
def inject_one_pulse(q):
    # X90 about the axis at 3 pi / 4 lands on |T> and leaves the frame as it was.
    d = get_drive_line(q)
    shift_phase(d.tone, -3 * math.pi / 4)
    drive(d, d.get_calibration(pulse.gates.X90))
    shift_phase(d.tone, 3 * math.pi / 4)


def frame_bloch(rho, frame_phase):
    """The logical Bloch vector of a simulated qubit block: the state is in
    the frame, so turn it by the phase the tone has accumulated."""
    x, y, z = 2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real
    c, s = math.cos(frame_phase), math.sin(frame_phase)
    return np.array([c * x - s * y, s * x + c * y, z])


t = D5.get_calibration(pulse.gates.T)
print(f"T on D5: {t}, {len(inject.lower(D5).schedule)} pulse in the schedule")
virtual = frame_bloch(
    pulse.simulate(inject, D5, initial_state=ground()).density_matrix, t.phase_rad
)
one_pulse = frame_bloch(
    pulse.simulate(inject_one_pulse, D5, initial_state=ground()).density_matrix, 0.0
)
print(f"from |0>: F = {fidelity(virtual):.5f} (Y90 + T), {fidelity(one_pulse):.5f} (one X90)")
assert np.allclose(virtual, one_pulse, atol=1e-4)
assert fidelity(virtual) > 0.999


# ── 2. Tomography ─────────────────────────────────────────────────────────


@pulse.kernel
def tomography(q, axis: int):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, d.get_calibration(pulse.gates.Y90))
    drive(d, d.get_calibration(pulse.gates.T))
    if axis == X_AXIS:  # Ry(-pi/2) takes +X to +Z
        y90 = d.get_calibration(pulse.gates.Y90)
        drive(d, y90.replace(amplitude=-y90.amplitude))
    elif axis == Y_AXIS:  # Rx(+pi/2) takes +Y to +Z
        drive(d, d.get_calibration(pulse.gates.X90))
    sync(d, r)
    readout(r, r.get_calibration(pulse.gates.READOUT))


def expectation(result):
    """<Z> of the last readout, 1 - 2 P(it reads nonzero): exact from the
    branches, or from the shots when there are records."""
    if result.records is not None and len(result.records):
        return 1.0 - 2.0 * float(np.mean(result.records[:, -1] > 0))
    return 1.0 - 2.0 * sum(b.probability for b in result.branches if b.levels[-1] > 0)


compiled_tomography = pulse.compile(tomography, quantum_system=qs)
for start, initial in (("|0>", ground()), ("thermal", None)):
    for mode, shots in (("ideal", None), ("model_with_decay", 4000)):
        bloch = np.array(
            [
                expectation(
                    pulse.simulate(
                        compiled_tomography.lower(D5, axis),
                        initial_state=initial,
                        readout=mode,
                        shots_count=shots,
                        seed=11 + axis,
                    )
                )
                for axis in (X_AXIS, Y_AXIS, Z_AXIS)
            ]
        )
        how = "exact" if shots is None else f"{shots} shots"
        print(f"tomography from {start:7} ({mode}, {how}): F = {fidelity(bloch):.4f}")
        assert fidelity(bloch) > (0.99 if start == "|0>" else 0.95)


# ── 3. Heralded repeat-until-success ──────────────────────────────────────

control_system = pulse.ControlSystem(links={"controller": Link(fixed_ns=4, feedback_ns=8, slots=2)})


# [start herald]
@pulse.classical(min_latency_ns=20, nominal_latency_ns=30, max_latency_ns=40, link="controller")
def herald(outcome: pulse.Bits) -> bool:
    """Accept the shot when the qubit read 0."""
    return outcome[0] == 0


# [end herald]


# [start heralded attempts]
@pulse.kernel
def heralded_t(q, attempts: int, axis: int):
    d = get_drive_line(q)
    r = get_readout_line(q)
    x = d.get_calibration(pulse.gates.X)
    ro = r.get_calibration(pulse.gates.READOUT)
    # A fixed number of attempts: each is scheduled, and its answer gates the
    # pi pulse that returns a qubit that read 1 to |0>.
    for _ in range(attempts):
        m = readout(r, ro)
        ok = herald.submit([m]).result()
        sync(d, r)
        if not ok:
            drive(d, x)
        sync(d, r)
    # [end heralded attempts]
    # [start heralded state]
    drive(d, d.get_calibration(pulse.gates.Y90))
    drive(d, d.get_calibration(pulse.gates.T))
    if axis == X_AXIS:
        y90 = d.get_calibration(pulse.gates.Y90)
        drive(d, y90.replace(amplitude=-y90.amplitude))
    elif axis == Y_AXIS:
        drive(d, d.get_calibration(pulse.gates.X90))
    sync(d, r)
    readout(r, ro)
    # [end heralded state]


compiled_herald = pulse.compile(heralded_t, quantum_system=qs, control_system=control_system)
started = time.perf_counter()
heralded = {}
for attempts in range(4):
    bloch, acceptance = np.zeros(3), 1.0
    for axis in (X_AXIS, Y_AXIS, Z_AXIS):
        result = pulse.simulate(compiled_herald.lower(D5, attempts, axis))
        if attempts:
            result = result.postselect(lambda b: b.answers("herald")[-1])
        bloch[axis], acceptance = expectation(result), result.acceptance
    heralded[attempts] = (fidelity(bloch), acceptance)
    print(
        f"herald x{attempts}: F = {fidelity(bloch):.5f}, 1 - F = {1 - fidelity(bloch):.1e}, "
        f"acceptance {acceptance:.5f}"
    )
print(f"(3 axes x 4 loops simulated in {time.perf_counter() - started:.1f} s)")
# The herald removes the thermal excitation of the start; a qubit that
# starts in |2> reads nonzero on every attempt and is never accepted.
assert heralded[1][0] > heralded[0][0] + 0.02 and heralded[1][0] > 0.998
assert heralded[0][1] == 1.0 and 0.9 < heralded[1][1] < heralded[2][1] <= heralded[3][1] < 1.0


# ── 4. When the answers are due ───────────────────────────────────────────

program = compiled_herald.lower(D5, 2, Z_AXIS)
length_ns = max(e.end for e in program.schedule) / qs.clock_ghz
print(f"\nherald x2 program: {length_ns:.0f} ns")
for call in program.classical_report():
    exposed = (
        "unused" if call.exposed_cycles is None else f"{call.exposed_cycles / qs.clock_ghz:.0f} ns"
    )
    slack = "unused" if call.slack_cycles is None else f"{call.slack_cycles / qs.clock_ghz:.0f} ns"
    print(
        f"  {call.function}: submitted at {call.submit_cycles / qs.clock_ghz:.0f} ns, "
        f"due {call.due_cycles / qs.clock_ghz:.0f} ns, "
        f"slack {slack}, "
        f"exposed {exposed}"
    )
