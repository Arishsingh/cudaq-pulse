# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-Time Decoding with a Deadline: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/realtime_decoding.html

Generated from docs/tutorials/realtime_decoding.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

import importlib.util


# Blocks under ``if GPU:`` need the GPU runtime (Linux, an NVIDIA GPU and
# ``pip install "cudaq-pulse[cu12]"`` or ``[cu13]``).
def _gpu_runtime_loads() -> bool:
    from cudaq_pulse.runtime import require_gpu_runtime

    try:
        require_gpu_runtime()
    except Exception:
        return False
    return True


GPU = _gpu_runtime_loads()


# Blocks under ``if CUDAQ_QEC:`` need CUDA-Q QEC (``pip install cudaq-qec``)
# and the GPU runtime.
CUDAQ_QEC = GPU and importlib.util.find_spec("cudaq_qec") is not None


# %% The code and its syndrome
import numpy as np

import cudaq_pulse as pulse


def lookup(syndrome: pulse.Bits) -> pulse.Bits:
    s0, s1 = syndrome[0], syndrome[1]
    # One correction bit per data qubit.
    return [s0 & (1 - s1), s0 & s1, s1 & (1 - s0)]


decode = pulse.classical(
    lookup,
    link="gpu",
    nominal_latency_ns=62_000,
    max_latency_ns=75_000,
)


# %% One round, with a deadline
@pulse.kernel
def qec_round(data, ancillas, deadline_ns: int):
    measurement = square(400 * pulse.ns, 0.05)
    correction = square(20 * pulse.ns, np.pi / 20)

    # Syndrome-extraction pulses would go here.
    s0 = readout(get_readout_line(ancillas[0]), measurement)
    s1 = readout(get_readout_line(ancillas[1]), measurement)

    request = decode.submit([s0, s1])
    in_time, fix = request.result(timeout=deadline_ns * pulse.ns)

    # The fallback, computed on the controller.
    table = [s0 and not s1, s0 and s1, s1 and not s0]
    for i in range(3):
        flip = fix[i] if in_time else table[i]
        if flip:
            drive(get_drive_line(data[i]), correction)


# %% Check the timing
system = pulse.quantum_system("transmon_lattice:5")
data = system.qudits[0:3]
ancillas = system.qudits[3:5]

gpu_link = pulse.Link(feedback_ns=50)
control_system = pulse.ControlSystem(links={"gpu": gpu_link})
compiled = pulse.compile(
    qec_round,
    quantum_system=system,
    control_system=control_system,
)


scheduled = compiled.lower(data, ancillas, deadline_ns=80_000)
call = scheduled.classical_report()[0]
due_ns = call.due_cycles / system.clock_ghz
print(f"answer due at {due_ns} ns")

# Prints:
#   answer due at 80450.5 ns


# %% Simulate both arms
# Bits of data 0, 1, 2, then ancillas 0, 1.
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


def run(program):
    result = pulse.simulate(
        program,
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
    return round(float(used), 3), round(float(restored), 3)


short_deadline = compiled.lower(data, ancillas, deadline_ns=1_000)
used, restored = run(short_deadline)
print(f"decoder's answer used: {used}")
print(f"data back in 000:      {restored}")

# Prints:
#   decoder's answer used: 0.0
#   data back in 000:      0.883


# %% Give the decoder a budget that fits
decode = pulse.classical(lookup, link="gpu", max_latency_ns=500)
fast = pulse.compile(
    qec_round,
    quantum_system=system,
    control_system=control_system,
)

used, restored = run(fast.lower(data, ancillas, deadline_ns=1_000))
print(f"decoder's answer used: {used}")
print(f"data back in 000:      {restored}")

# Prints:
#   decoder's answer used: 1.0
#   data back in 000:      0.883


# %% With a CUDA-Q QEC decoder
if CUDAQ_QEC:
    import cudaq_qec


if CUDAQ_QEC:
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


if CUDAQ_QEC:
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


if CUDAQ_QEC:
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

        # The decoder's answer, or the local table if it is late.
        table = [s0 and not s1, s0 and s1, s1 and not s0]
        for i in range(3):
            flip = fix[i] if in_time else table[i]
            if flip:
                drive(get_drive_line(data[i]), correction)


if CUDAQ_QEC:
    decoded = pulse.compile(
        qec_round,
        quantum_system=system,
        control_system=control_system,
    )
    used, restored = run(decoded.lower(data, ancillas, deadline_ns=1_000))
    print(f"decoder's answer used: {used}")
    print(f"data back in 000:      {restored}")
    cudaq_qec.finalize_decoders()
