..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Real-Time Decoding with a Deadline
==================================

Quantum error correction measures syndromes, decodes them into corrections,
and applies them, round after round. In real time, the decoder must answer
while the qubits wait, and the program needs a plan for when it doesn't.

This tutorial runs one round of a distance-3 repetition code, and falls
back to a table on the controller when the decoder is late.

What you'll learn:

- how to call a decoder from a kernel with a deadline;
- how to fall back to a table on the controller;
- how to check both arms in simulation;
- how the same round calls a CUDA-Q QEC decoder.

The code and its syndrome
-------------------------

Three data qubits hold one logical bit, and two ancillas hold the syndrome:
the parity of data qubits 0 and 1, and of 1 and 2.

A single bit flip shows up as a unique syndrome, which says which qubit to
flip back:

.. list-table::
   :header-rows: 1

   * - Syndrome (s0, s1)
     - Flipped data qubit
   * - (0, 0)
     - none
   * - (1, 0)
     - 0
   * - (1, 1)
     - 1
   * - (0, 1)
     - 2

The decoder below answers with that table:

.. code-block:: python

   import numpy as np

   import cudaq_pulse as pulse


   def lookup(syndrome: pulse.Bits) -> pulse.Bits:
       s0, s1 = syndrome[0], syndrome[1]
       # One correction bit per data qubit.
       return [s0 & (1 - s1), s0 & s1, s1 & (1 - s0)]

.. themed-figure:: figures/repetition_syndromes.svg
   :alt: Four small diagrams of the three data qubits with the two syndrome bits between them, one per syndrome, with what the page's lookup corrects: syndrome (0, 0), no flip; syndrome (1, 0), flip data qubit 0; syndrome (1, 1), flip data qubit 1; syndrome (0, 1), flip data qubit 2.
   :width: 100%

   What ``lookup`` corrects for each syndrome. Circles are data qubits,
   squares the syndrome bits between them.

What to notice: a lit syndrome bit points at the data qubit next to it. Two
lit bits point at the one they share.

A real decoder, such as the one in the last section, does the same job for
codes whose table doesn't fit in the controller's memory.

Declare it as a classical function. Corrections come back in about 62 µs,
and 75 µs at worst:

.. code-block:: python

   decode = pulse.classical(
       lookup,
       link="gpu",
       nominal_latency_ns=62_000,
       max_latency_ns=75_000,
   )

``pulse.classical`` declares a function you already have, as well as
decorating a new one. That makes it easy to declare it again with another
budget, as the simulation below does.

.. note::

   The latencies are round trips from a host process to a decoding server
   on a GPU in the same workstation. On NVQLink, requests go from the
   controller to the GPU without the host, in a few microseconds.

One round, with a deadline
--------------------------

The round reads both ancillas, asks the decoder, and corrects the data
qubits.

First the syndrome and the call, with the deadline as a kernel argument:

.. code-block:: python
   :class: continues

   @pulse.kernel
   def qec_round(data, ancillas, deadline_ns: int):
       measurement = square(400 * pulse.ns, 0.05)
       correction = square(20 * pulse.ns, np.pi / 20)

       # Syndrome-extraction pulses would go here.
       s0 = readout(get_readout_line(ancillas[0]), measurement)
       s1 = readout(get_readout_line(ancillas[1]), measurement)

       request = decode.submit([s0, s1])
       in_time, fix = request.result(timeout=deadline_ns * pulse.ns)

Then each data qubit takes the decoder's answer, or the table's if the
answer is late:

.. code-block:: python

       # The fallback, computed on the controller.
       table = [s0 and not s1, s0 and s1, s1 and not s0]
       for i in range(3):
           flip = fix[i] if in_time else table[i]
           if flip:
               drive(get_drive_line(data[i]), correction)

What to notice:

- the correction is a π pulse;
- the deadline is an argument, so you can trade waiting against falling
  back without editing the program.

The syndrome-extraction pulses are left out. The shipped transmon models
schedule a CZ as an idle slot, since flux pulses aren't simulated yet, so
the round starts from the state extraction leaves.

Check the timing
----------------

Compiling checks the timing before anything runs. Use five transmons in a
row, three for data and two ancillas:

.. code-block:: python

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

With an 80 µs deadline, find when the answer is due:

.. code-block:: python

   scheduled = compiled.lower(data, ancillas, deadline_ns=80_000)
   call = scheduled.classical_report()[0]
   due_ns = call.due_cycles / system.clock_ghz
   print(f"answer due at {due_ns} ns")

.. code-block:: text
   :class: output

   answer due at 80450.5 ns

That is the 400 ns readouts, the 80 µs wait and the link's 50 ns. The
corrections are scheduled after it.

Simulate both arms
------------------

Now check how often the round restores the data qubits.

Start from the state syndrome extraction leaves: no error, or one flipped
data qubit, each with a quarter of the probability, with the ancillas
holding the matching syndrome:

.. code-block:: python

   # Bits of data 0, 1, 2, then ancillas 0, 1.
   STARTS = [
       (0, 0, 0, 0, 0),
       (1, 0, 0, 1, 0),
       (0, 1, 0, 1, 1),
       (0, 0, 1, 0, 1),
   ]

Each start is a basis state with amplitude 0.5:

.. code-block:: python

   def start_state():
       state = np.zeros(2**5)
       for bits in STARTS:
           index = 0
           for position, bit in enumerate(bits):
               index += bit << position
           state[index] = 0.5
       return state

``run`` simulates a program from that state, branching on whether each
answer arrives in time:

.. code-block:: python
   :class: continues

   def run(program):
       result = pulse.simulate(
           program,
           initial_state=start_state(),
           levels=2,
           latency_branching=True,
           dt_ns=2.0,
       )

It returns the share of shots that used the decoder's answer, and the
share whose data qubits end back in 000:

.. code-block:: python

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

Run the round with a 1 µs deadline:

.. code-block:: python

   short_deadline = compiled.lower(data, ancillas, deadline_ns=1_000)
   used, restored = run(short_deadline)
   print(f"decoder's answer used: {used}")
   print(f"data back in 000:      {restored}")

.. code-block:: text
   :class: output

   decoder's answer used: 0.0
   data back in 000:      0.883

The decoder's 62 µs answer is always late, so the table makes every
correction. The rest of the population decays during the round.

.. themed-figure:: figures/qec_round_timeline.svg
   :alt: Timeline of one round of the distance-3 repetition code, with a lane for each data qubit's drive line, each ancilla's readout line and the GPU link. Both ancillas are read out for 400 ns, and their outcomes go to the decoder over the link. The answer is due at 1450.5 ns, and a correction pulse on each data qubit, which plays only if the decoder's answer or the fallback table says so, starts at that moment.
   :width: 100%

   The round with the 1 µs deadline. Dashed corrections play only when
   flagged.

What to notice: the round is mostly the wait between the syndrome and the
corrections.

Up close, the end of that wait:

.. themed-figure:: figures/qec_round_deadline.svg
   :alt: The last 110 ns of the round's wait. The kernel waits for the answer with a 1 microsecond timeout, plus 50 ns of feedback on the link, so the answer is due at 1450.5 ns. The dashed correction pulses on the data qubits start at that moment, whichever arm decides them.
   :width: 100%

   The last 110 ns of the wait, and the corrections.

What to notice: every correction waits for the deadline, whichever arm
decides it.

Give the decoder a budget that fits
-----------------------------------

Declare the decoder again with a 500 ns budget, and compile again:

.. code-block:: python

   decode = pulse.classical(lookup, link="gpu", max_latency_ns=500)
   fast = pulse.compile(
       qec_round,
       quantum_system=system,
       control_system=control_system,
   )

   used, restored = run(fast.lower(data, ancillas, deadline_ns=1_000))
   print(f"decoder's answer used: {used}")
   print(f"data back in 000:      {restored}")

.. code-block:: text
   :class: output

   decoder's answer used: 1.0
   data back in 000:      0.883

A kernel reads the functions it calls when it compiles, so the new
declaration is picked up.

.. themed-figure:: figures/qec_budgets.svg
   :alt: The round with a 1 microsecond deadline, for the decoder as first declared and with a 500 ns budget. The share of shots that use the decoder's answer is 0 and 1; the share whose data qubits end back in 000 is 0.883 and 0.883.
   :width: 100%

   The two runs with a 1 µs deadline, side by side.

What to notice:

- every shot now uses the decoder's answer;
- the data qubits are restored as often as with the table, because the two
  agree on every single-qubit error;
- that agreement is what makes the fallback safe, and checking it before
  the program runs on hardware is the point of the exercise.

The trade-off
-------------

Between the two budgets lies the real choice. With the decoder as first
declared, each deadline trades waiting for the answer against decay:

.. themed-figure:: figures/qec_deadlines.svg
   :alt: Two curves against the decoding deadline, from 1 to 80 microseconds, simulated with the decoder's 62 microsecond nominal and 75 microsecond maximum latency. The share of shots that use the decoder's answer rises from 0 at 1 microsecond through 0.54 at 50 to 1 at the maximum. The share whose data qubits end back in 000 falls from 0.883 at 1 microsecond to 0.285 at 80, because the qubits decay while they wait.
   :width: 100%

   ``run`` of the first compiled round, over deadlines from 1 to 80 µs.

What to notice: waiting lets more shots use the decoder, and lets the data
qubits decay. This round's table is as good as the decoder, so here the
shortest deadline wins.

With a CUDA-Q QEC decoder
-------------------------

`CUDA-Q QEC <https://nvidia.github.io/cudaq-qec/>`_ provides GPU decoders
and their real-time functions. ``pulse.classical`` takes them as they are,
with a link and a latency budget.

This section needs:

- Linux, an NVIDIA GPU, and Python 3.12 or 3.13;
- ``pip install cudaq-qec``.

The test suite runs it only there. The code comes from
:example:`27_realtime_decoding.py`.

Import CUDA-Q QEC:

.. code-block:: python
   :class: cudaq-qec

   import cudaq_qec

Configure decoder 0, BP+OSD for the distance-3 repetition code:

.. literalinclude:: ../../examples/27_realtime_decoding.py
   :language: python
   :class: cudaq-qec
   :start-after: # [start decoder]
   :end-before: # [end decoder]

Declare its real-time functions, with the same link and budget as
``decode``:

.. literalinclude:: ../../examples/27_realtime_decoding.py
   :language: python
   :class: cudaq-qec
   :start-after: # [start calls]
   :end-before: # [end calls]

The round sends the syndrome with one call and takes the corrections with
another:

.. literalinclude:: ../../examples/27_realtime_decoding.py
   :language: python
   :class: cudaq-qec continues
   :start-after: # [start round]
   :end-before: # [end round syndromes]

The corrections use the same deadline and fallback as before:

.. literalinclude:: ../../examples/27_realtime_decoding.py
   :language: python
   :class: cudaq-qec
   :start-after: # [start round corrections]
   :end-before: # [end round]

Compile it for the same five transmons and simulate it as before:

.. code-block:: python
   :class: cudaq-qec

   decoded = pulse.compile(
       qec_round,
       quantum_system=system,
       control_system=control_system,
   )
   used, restored = run(decoded.lower(data, ancillas, deadline_ns=1_000))
   print(f"decoder's answer used: {used}")
   print(f"data back in 000:      {restored}")
   cudaq_qec.finalize_decoders()

On hardware, these requests are the messages of CUDA-Q QEC's decoding
server. In simulation, the configured decoder answers them in the same
process.

Where to go next
----------------

- :example:`27_realtime_decoding.py` runs this round end to end and
  compares both arms with no correction at all.
- :example:`19_classical_calls.py` covers decoding with feed-forward, call
  slack and hidden latency.
- :example:`18_per_kernel_schedules.py` schedules a surface-code-style
  stabilizer cycle with dynamical decoupling.
- :example:`transmon/transmon_surface17_surface_code_cycle.py` lays out one
  error-correction cycle of the 17-qubit surface code.

:doc:`../user_guide/classical_calls` describes the decoder messages,
emulation in virtual time, and how a late answer is reported.
