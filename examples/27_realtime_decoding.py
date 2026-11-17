# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-time QEC decoding with a deadline and a fallback.

Builds on ``19_classical_calls.py``. One round of a distance-3 repetition
code sends its syndrome to a CUDA-Q QEC decoder, ``nv-qldpc-decoder`` on the
GPU (belief propagation with ordered-statistics post-processing, BP+OSD),
and asks for the corrections with a deadline. If the answer arrives in time,
the correction comes from the decoder; if not, from a lookup table on the
controller. The deadline is a kernel parameter, so trading waiting against
falling back needs no change to the program.

The decoder is configured in CUDA-Q QEC, as decoder 0, and the kernel calls
it with CUDA-Q QEC's own functions, ``cudaq_qec.qec.enqueue_syndromes`` and
``get_corrections``: on hardware the requests are the messages of CUDA-Q
QEC's decoding server, here with a latency budget of 62 us typical and 75 us
at most. Those budgets are round trips measured from a host process to the
decoding server on the same workstation's GPU, over UDP loopback; they are
not a quantum controller's latencies. Over NVQLink, FPGA-to-GPU round trips
take a few microseconds.

This example shows:

1. The compile-time report: when each call is due, for an 80 us deadline.
2. The configured decoder's correction for every syndrome, called from the
   host.
3. Simulation of the round with a 1 us deadline, from the state syndrome
   extraction leaves (no error, or one flipped data qubit). Against the
   62/75 us budget, nearly every shot misses the deadline and takes the
   table; with the same decoder given 500 ns, every shot uses the decoder.
   Both restore the data qubits equally, and far better than no correction.

What is simulated: the readouts and correction pulses on a five-qudit
transmon lattice, with decay. The configured decoder answers in this
process, through CUDA-Q QEC's simulation library, as it answers a CUDA-Q
kernel on a simulator. The simulation does not wait out the 80 us deadline,
because the simulator integrates idle time step by step.

Requires cudaq-qec (``pip install cudaq-qec``; Linux, Python 3.11 to 3.13)
and a CUDA-Q compatible NVIDIA GPU for ``nv-qldpc-decoder``.
"""

import cudaq_pulse as pulse
import cudaq_qec
import numpy as np

# [start decoder]
# Decoder 0: BP+OSD for a distance-3 repetition code.
cudaq_qec.configure_decoders_from_str("""
decoders:
  - id: 0
    type: nv-qldpc-decoder
    block_size: 3
    syndrome_size: 2
    H_sparse: [0, 1, -1, 1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1]
    decoder_custom_args:
      error_rate_vec: [0.1, 0.1, 0.1]
      use_sparsity: true
      max_iterations: 50
      use_osd: true
      osd_method: 3
      osd_order: 7
""")
# [end decoder]

# [start calls]
# Its real-time calls, sent to the decoding server over the GPU link.
enqueue_syndromes = pulse.classical(
    cudaq_qec.qec.enqueue_syndromes,
    link="gpu",
)
get_corrections = pulse.classical(
    cudaq_qec.qec.get_corrections,
    link="gpu",
    nominal_latency_ns=62_000,
    max_latency_ns=75_000,
)
reset_decoder = pulse.classical(
    cudaq_qec.qec.reset_decoder,
    link="gpu",
)
# [end calls]


# [start round]
@pulse.kernel
def qec_round(data, ancillas, deadline_ns: int):
    measurement = square(400 * pulse.ns, 0.05)
    correction = square(20 * pulse.ns, np.pi / 20)

    # Syndrome-extraction pulses on data and ancillas would go here.
    s0 = readout(get_readout_line(ancillas[0]), measurement)
    s1 = readout(get_readout_line(ancillas[1]), measurement)

    # Decoder 0, round 0.
    enqueue_syndromes.submit(0, [s0, s1], 0)
    request = get_corrections.submit(0, 3, False)
    in_time, fix = request.result(timeout=deadline_ns * pulse.ns)
    # [end round syndromes]
    # [start round corrections]

    # The decoder's answer, or the local table if it is late.
    table = [s0 and not s1, s0 and s1, s1 and not s0]
    for i in range(3):
        flip = fix[i] if in_time else table[i]
        if flip:
            drive(get_drive_line(data[i]), correction)


# [end round]


@pulse.kernel
def no_correction(data, ancillas):
    measurement = square(400 * pulse.ns, 0.05)
    readout(get_readout_line(ancillas[0]), measurement)
    readout(get_readout_line(ancillas[1]), measurement)
    for qudit in data:
        wait(get_drive_line(qudit), 1_000 * pulse.ns)


# Bits of data 0, 1, 2, then ancillas 0, 1: no error, or one flipped data
# qubit, with the ancillas holding d0^d1 and d1^d2.
STARTS = [
    (0, 0, 0, 0, 0),
    (1, 0, 0, 1, 0),
    (0, 1, 0, 1, 1),
    (0, 0, 1, 0, 1),
]


def start_state():
    state = np.zeros(2**5)
    for bits in STARTS:
        index = 0
        for position, bit in enumerate(bits):
            index += bit << position
        state[index] = 0.5
    return state


def simulate(scheduled, label):
    result = pulse.simulate(
        scheduled,
        initial_state=start_state(),
        levels=2,
        latency_branching=True,
        dt_ns=2.0,
    )
    used = 0.0
    for branch in result.branches:
        if branch.calls and branch.calls[-1].in_time:
            used += branch.probability
    restored = 0.0
    for index in range(32):
        # The data qubits are the three lowest bits.
        if index & 0b111 == 0:
            restored += result.density_matrix[index, index].real
    print(f"   {label:44} decoder answer used {used:.4f}   P(data back in 000) {restored:.4f}")
    return restored


if __name__ == "__main__":
    system = pulse.quantum_system("transmon_lattice:5")
    gpu_link = pulse.Link(feedback_ns=50)
    control_system = pulse.ControlSystem(links={"gpu": gpu_link})
    data = system.qudits[0:3]
    ancillas = system.qudits[3:5]

    print("1. Compile-time report, 80 us deadline")
    compiled = pulse.compile(
        qec_round,
        quantum_system=system,
        control_system=control_system,
    )
    scheduled = compiled.lower(data, ancillas, deadline_ns=80_000)
    for call in scheduled.classical_report():
        due_us = call.due_cycles / system.clock_ghz / 1e3
        print(f"   {call.function}: due at {due_us:.1f} us")

    print("2. Decoder corrections, asked from the host")
    expected = {
        (0, 0): [0, 0, 0],
        (1, 0): [1, 0, 0],
        (1, 1): [0, 1, 0],
        (0, 1): [0, 0, 1],
    }
    for syndrome, want in expected.items():
        reset_decoder(0)
        enqueue_syndromes(0, list(syndrome), 0)
        got = get_corrections(0, 3, False)
        print(f"   syndrome {syndrome} -> {got}")
        assert got == want, (syndrome, got)

    print("3. Simulation, 1 us deadline")
    late = simulate(
        compiled.lower(data, ancillas, deadline_ns=1_000),
        "62/75 us budget (answers late)",
    )

    get_corrections = pulse.classical(
        cudaq_qec.qec.get_corrections,
        link="gpu",
        max_latency_ns=500,
    )
    fast = pulse.compile(
        qec_round,
        quantum_system=system,
        control_system=control_system,
    )
    in_time = simulate(
        fast.lower(data, ancillas, deadline_ns=1_000),
        "500 ns budget (answers in time)",
    )

    uncorrected = pulse.compile(no_correction, quantum_system=system)
    none = simulate(uncorrected.lower(data, ancillas), "no correction")
    assert abs(late - in_time) < 1e-3
    assert late > none + 0.5
    cudaq_qec.finalize_decoders()
