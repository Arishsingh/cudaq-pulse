..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Calling Classical Code During Quantum Runtime
=============================================

Some decisions need more than an ``if`` on a readout: a decoder has to turn
syndromes into a correction, or an estimator has to choose the next pulse
from what was just measured. CUDA-Q Pulse lets a pulse program call a
classical function on a GPU, a host or the controller while the qubits wait,
and use the answer later in the same shot. This tutorial declares such a
function, calls it from a kernel with a deadline and a fallback, reads the
timing the compiler works out, and simulates the whole loop.

Declare the function
--------------------

``@pulse.classical`` declares a function the program can call. Its type
hints say what goes in and what comes back, and its options say how it is
reached and how long it may take:

.. code-block:: python

   import cudaq_pulse as pulse


   @pulse.classical(
       link="gpu", min_latency_ns=2_900, nominal_latency_ns=3_800, max_latency_ns=4_500
   )
   def decode(syndrome: pulse.Bits) -> bool:
       return bool(syndrome[0])

``pulse.Bits`` takes measurement results as bits, and the ``bool`` answer
is something a branch can test. The function runs on whatever the link
named ``"gpu"`` reaches. It answers in 3.8 µs typically, never sooner than
2.9 µs and never later than 4.5 µs, a spread like that of measured GPU
round trips. In simulation, its Python body gives the answer; on hardware,
the same declaration is what the control system's service implements.

Call it with a deadline
-----------------------

A call has two halves: ``submit`` sends the request, and ``result`` takes
the answer where the program needs it. ``result(timeout=...)`` also says
how long the program will wait, and returns whether the answer came in time:

.. code-block:: python

   @pulse.kernel
   def correct_if_needed(q):
       d = get_drive_line(q)
       r = get_readout_line(q)
       outcome = readout(r, r.get_calibration(pulse.gates.READOUT))
       request = decode.submit([outcome])
       in_time, flip = request.result(timeout=4 * pulse.us)
       correct = flip if in_time else outcome   # late: fall back to the raw outcome
       sync(d, r)
       if correct:
           drive(d, d.get_calibration(pulse.gates.X))

If the answer is late, the kernel uses the readout itself instead of
stalling. Either way, a π pulse plays only when the decision says the qubit
needs flipping.

.. figure:: figures/realtime_timeline.svg
   :alt: Timeline of one shot with drive, readout and classical-function lanes. A readout's result goes to the decode function, whose answer, taken with a 4 microsecond timeout, conditions a pi pulse on the drive line. A second function, update_t1, sets a wait whose length is known only at runtime.
   :width: 95%

   One shot with real-time calls. The decoder's answer gates a conditional
   pulse, as in ``correct_if_needed``. The estimator lane is the subject of
   :doc:`adaptive_calibration`.

Check the timing before anything runs
-------------------------------------

Links belong to the control system: it says how the controller reaches each
service. Compiling against one schedules the call like any other operation:

.. code-block:: python

   qs = pulse.quantum_system("transmon_surface17")
   q = qs.qudits[0]
   control_system = pulse.ControlSystem(links={"gpu": pulse.Link(feedback_ns=50)})
   compiled = pulse.compile(correct_if_needed, control_system=control_system)
   scheduled = compiled.lower(q)

   call = scheduled.classical_report()[0]
   print(call.function, call.link_name)   # decode gpu
   print(call.submit_cycles / qs.clock_ghz)   # 300.0, when the request leaves, in ns
   print(call.due_cycles / qs.clock_ghz)   # 4350.5, when the answer is due, in ns
   print(call.slack_cycles)   # 0

The request leaves when the 300 ns readout ends. The scheduler budgets the
4 µs timeout plus the link's 50 ns of feedback, so the answer is due at
4350.5 ns, and the conditional π pulse is placed right after it. A slack of
zero means the program uses the answer the moment it is due.

Simulate the loop
-----------------

Simulating the program runs ``decode``'s Python body where the call is
taken. Start the qubit in an equal superposition so both outcomes happen,
and let the simulator branch on whether each answer arrives in time, with
the probability the latency spread gives:

.. code-block:: python

   result = pulse.simulate(scheduled, initial_state=[2**-0.5, 2**-0.5, 0.0], latency_branching=True)
   in_time = sum(b.probability for b in result.branches if b.calls[0].in_time)
   print(round(in_time, 3))   # 0.777
   print(round(result.level_population(0), 3))   # 0.933

With latencies from 2.9 to 4.5 µs, 77.7% of the answers arrive within the
4 µs timeout, and the other shots take the fallback. Here the fallback makes
the same decision as the decoder; with a real decoder, the fallback is where
a late answer costs fidelity.

Either way, the qubit ends in its ground state with 93% probability. The
rest is decay during the 4 µs it waited for its answer, a reminder of why
real-time budgets are tight.

Where to go next
----------------

- :example:`19_classical_calls.py` runs real-time decoding with
  feed-forward, reads call slack and hidden latency, and emulates the program
  against its services.
- :example:`23_define_a_qpu.py` sends a readout on one controller to a
  correction on another over a feedback link.
- :example:`27_realtime_decoding.py` calls a CUDA-Q QEC decoder on the GPU,
  which :doc:`realtime_decoding` walks through.
- :example:`20_adaptive_calibration.py` lets an estimator set the next wait,
  as in :doc:`adaptive_calibration`.

:doc:`../user_guide/classical_calls` covers argument types and lengths,
dispatch keys, emulation in virtual time and the controller's own functions.
