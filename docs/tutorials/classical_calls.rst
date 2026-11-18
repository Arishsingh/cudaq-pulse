..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Calling Classical Code During Quantum Runtime
=============================================

Some decisions need more than an ``if`` on a readout. A decoder turns
syndromes into a correction; an estimator picks the next pulse from what
was just measured.

CUDA-Q Pulse lets a program call such a function on a GPU, a host or the
controller while the qubits wait, and use the answer in the same shot.

What you'll learn:

- how to declare a classical function and its latency;
- how to call it from a kernel with a deadline and a fallback;
- how to read the timing the compiler works out;
- how to simulate the loop, late answers included.

All the code on this page is also one Python script,
:download:`classical_calls.py <scripts/classical_calls.py>`.

Declare the function
--------------------

A function the program can call is declared with ``@pulse.classical``. Its
type hints say what goes in and what comes back:

.. code-block:: python

   import cudaq_pulse as pulse


   @pulse.classical(
       link="gpu",
       min_latency_ns=2_900,
       nominal_latency_ns=3_800,
       max_latency_ns=4_500,
   )
   def decode(syndrome: pulse.Bits) -> bool:
       return bool(syndrome[0])

What the declaration says:

- ``pulse.Bits`` takes measurement results as bits;
- the ``bool`` answer is something a branch can test;
- ``link="gpu"`` names how the controller reaches the function;
- it answers in 3.8 µs typically, never sooner than 2.9 µs and never later
  than 4.5 µs, a spread like that of measured GPU round trips.

.. themed-figure:: figures/decode_declared.svg
   :alt: The latency decode declares, on a line from 0: never sooner than 2.9 microseconds, typically 3.8 and never later than 4.5.
   :width: 100%

   The latency ``decode`` declares.

What to notice: the answer can come any time in a 1.6 µs window. A
deadline inside it, as the next section sets, means some answers are late.

In simulation, the Python body gives the answer. On hardware, the control
system's service implements the same declaration.

Call it with a deadline
-----------------------

A call has two halves: ``submit`` sends the request, and ``result`` takes
the answer where the program needs it.

The kernel reads the qubit and sends the outcome. With a timeout,
``result`` also returns whether the answer came in time, and a late answer
falls back to the raw outcome:

.. code-block:: python

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

What to notice: the kernel never stalls. If the answer is late, it uses the
readout itself, and either way the π pulse plays only when the decision
says so.

Check the timing before anything runs
-------------------------------------

Before running anything, compile the kernel and read when the call happens.

Links belong to the control system, which says how the controller reaches
each service. This one adds 50 ns of feedback on the way back:

.. code-block:: python

   system = pulse.quantum_system("transmon_surface17")
   qubit = system.qudits[0]

   gpu_link = pulse.Link(feedback_ns=50)
   control_system = pulse.ControlSystem(links={"gpu": gpu_link})

   compiled = pulse.compile(
       correct_if_needed,
       control_system=control_system,
   )
   scheduled = compiled.lower(qubit)

The classical report lists each call with its times in clock cycles.
Convert them to nanoseconds:

.. code-block:: python

   call = scheduled.classical_report()[0]
   submit_ns = call.submit_cycles / system.clock_ghz
   due_ns = call.due_cycles / system.clock_ghz

   print(f"function:      {call.function} on {call.link_name}")
   print(f"request sent:  {submit_ns} ns")
   print(f"answer due:    {due_ns} ns")
   print(f"slack:         {call.slack_cycles} cycles")

.. code-block:: text
   :class: output

   function:      decode on gpu
   request sent:  300.0 ns
   answer due:    4350.5 ns
   slack:         0 cycles

Where those times come from:

- the request leaves when the 300 ns readout ends;
- the answer is due after the 4 µs timeout and the link's 50 ns;
- the conditional π pulse is placed right after the answer is due.

A slack of zero means the program uses the answer the moment it arrives.

.. themed-figure:: figures/call_timeline.svg
   :alt: Timeline of one shot of correct_if_needed on D1's readout and drive lines and the GPU link. The 300 ns readout's outcome goes to decode; the kernel waits for the answer with a 4 microsecond timeout and the link adds 50 ns of feedback, so the answer is due at 4350.5 ns, where a pi pulse that plays only if the decision says so starts.
   :width: 100%

   One shot of ``correct_if_needed``. The dashed π pulse plays only when the
   decision says so.

What to notice: almost all of the shot is the wait for the answer.

Simulate the loop
-----------------

Simulating the program runs ``decode``'s Python body where the call is
taken.

Start the qubit in an equal superposition, so both outcomes happen. With
``latency_branching``, the simulator also branches on whether each answer
arrives in time. Then add up the branches whose answer arrived in time:

.. code-block:: python

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

.. code-block:: text
   :class: output

   answers in time:    0.777
   ground state after: 0.933

.. themed-figure:: figures/call_arms.svg
   :alt: The branches of the simulation, by the readout and then by whether decode's answer arrived in time: read 0 and in time, 0.392; read 0 and late, 0.113; read 1 or more and in time, 0.385; read 1 or more and late, 0.111. Each readout outcome splits the same way.
   :width: 100%

   The branches of the simulation, by outcome and by when the answer came.

What to notice: each outcome splits into an answer in time and a late one,
in the same proportion.

With latencies from 2.9 to 4.5 µs, 77.7% of the answers beat the 4 µs
timeout. The other shots take the fallback.

.. themed-figure:: figures/call_latency.svg
   :alt: The distribution of decode's latency, triangular from 2.9 to 4.5 microseconds with its peak at the nominal 3.8, and the 4 microsecond timeout. The answers to its left, 77.7% of them in the simulation, arrive in time; the rest take the fallback.
   :width: 100%

   The latency ``decode`` declares, against the 4 µs timeout. The shaded
   share is the 77.7% of answers that arrive in time.

What to notice:

- here the fallback makes the same decision as the decoder, so late
  answers cost nothing;
- with a real decoder, the fallback is where a late answer costs fidelity;
- the qubit still ends in its ground state only 93% of the time, because
  it decays during the 4 µs it waits.

That last point is why real-time budgets are tight.

Where to go next
----------------

- :example:`19_classical_calls.py` runs decoding with feed-forward, reads
  call slack and hidden latency, and emulates the program.
- :example:`23_define_a_qpu.py` sends a readout on one controller to a
  correction on another.
- :doc:`realtime_decoding` calls a CUDA-Q QEC decoder on the GPU.
- :doc:`adaptive_calibration` lets an estimator set the next wait.

:doc:`../user_guide/classical_calls` covers argument types, dispatch keys,
emulation in virtual time and the controller's own functions.
