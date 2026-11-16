..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Adaptive Calibration in One Program
===================================

A qubit's T1 fluctuates over minutes to hours, and recalibrating by sweeping
a fixed set of waits wastes most of its shots. An adaptive measurement picks
each wait from what the earlier shots found. This tutorial implements the
Bayesian T1 tracker of Berritta et al. (Phys. Rev. X 16, 011025 (2026),
arXiv:2506.09576) as a single kernel: every shot excites the qubit, waits,
reads it out, and asks an estimator on the controller for the next wait.

The estimator
-------------

The estimator keeps a probability distribution over the decay rate 1/T1, a
gamma distribution with two parameters, ``shape`` and ``rate``. Each readout
updates it, and the updated distribution says which wait would teach the
most next. CUDA-Q Pulse ships both steps; declared as a classical function,
they run on the controller:

.. code-block:: python

   import cudaq_pulse as pulse
   from cudaq_pulse.classical import calibration as cal


   @pulse.classical(link="controller", max_latency_ns=500)
   def update_t1(bit: pulse.Bits, shape: float, rate: float, wait_ns: float) -> tuple[float, float, float]:
       shape, rate = cal.adaptive_t1_update(shape, rate, bit[0], wait_ns, error_1to0=0.11, error_0to1=0.14)
       next_wait_ns = cal.adaptive_t1_wait(shape, rate)
       next_wait_ns = min(max(next_wait_ns, 100.0), 200_000.0)   # keep it in 100 ns to 200 µs
       return shape, rate, next_wait_ns

``error_1to0`` and ``error_0to1`` are the readout's assignment errors, the
chance it reports 0 for an excited qubit and 1 for a qubit in its ground
state, as in the paper. The estimate of T1 is ``rate / shape``. The
estimator has 500 ns to answer, a budget for code that runs on the
controller itself.

The tracking kernel
-------------------

Every shot is an iteration of a loop inside one kernel, so the whole run
is one program:

.. code-block:: python

   @pulse.kernel
   def track_t1(q, shots_count: int, shape: float, rate: float, wait_ns: float):
       d = get_drive_line(q)
       r = get_readout_line(q)
       pi_pulse = d.get_calibration(pulse.gates.X)
       for shot in range(shots_count):
           drive(d, pi_pulse)   # excite, then let it decay for wait_ns
           wait(d, wait_ns * pulse.ns, bounds=(100 * pulse.ns, 200 * pulse.us))
           sync(d, r)
           outcome = readout(r, r.get_calibration(pulse.gates.READOUT))
           # The controller updates the estimate and picks the next wait.
           shape, rate, wait_ns = update_t1.submit([outcome], shape, rate, wait_ns).result()
           sync(d, r)
           if outcome:
               drive(d, pi_pulse)   # active reset

The wait's length is known only when the program runs, since it comes from
the estimator's answer. ``bounds`` tells the scheduler the range it can
take, so it can still check the timing of everything around it. The π pulse
at the end of each shot is the active reset from :doc:`measure_and_react`.

Compile the full run
--------------------

The controller link belongs to the control system. Here it adds 4 ns on the
way out and 8 ns on the way back:

.. code-block:: python

   qs = pulse.quantum_system("transmon_surface17")
   q = qs.qudits[0]
   control_system = pulse.ControlSystem(links={"controller": pulse.Link(fixed_ns=4, feedback_ns=8)})
   compiled = pulse.compile(track_t1, control_system=control_system)

   # 50 shots from a prior of shape 3 and rate 450 µs: a first guess of T1 = 150 µs.
   scheduled = compiled.lower(q, 50, 3.0, 450_000.0, 50_000.0)
   print(len(scheduled.classical_report()))   # 50

The program holds 50 calls to the estimator, one per shot, each with its
own place in the schedule. ``shots_count`` is an ``int``, so the loop's
length is fixed when it compiles; the prior and the first wait are
``float`` arguments, so a run from another prior compiles nothing new.

.. themed-figure:: figures/adaptive_shots.svg
   :alt: Timeline of the first 3 shots of track_t1 on D1's drive and readout lines and the controller link, with the long first wait cut short. Each shot plays a pi pulse, waits, and reads out; the outcome goes to update_t1, which answers within 500 ns plus the link's 4 and 8 ns, and a reset pulse plays only if the qubit read 1. The first wait is the 50 microseconds the call passes; each later wait is set by the answer, between 100 ns and 200 microseconds.
   :width: 100%

   The first three shots of ``compiled.lower(q, 3, 3.0, 450_000.0, 50_000.0)``,
   with the 50 µs first wait cut short. Each readout goes to ``update_t1``,
   whose answer sets the next wait, dashed because the schedule shows it at
   its 100 ns lower bound; the dashed reset pulses play only when the qubit
   read 1.

Simulate a few shots
--------------------

Every readout splits the simulation into two branches, and every branch
asks the estimator for its own next wait. Three shots make eight branches,
few enough to follow exactly:

.. code-block:: python

   result = pulse.simulate(compiled.lower(q, 3, 3.0, 450_000.0, 50_000.0), levels=2)
   print(len(result.branches))   # 8
   likeliest = max(result.branches, key=lambda b: b.probability)
   for shape, rate, wait_ns in likeliest.answers("update_t1"):
       print(round(rate / shape / 1000, 1), round(wait_ns / 1000, 1))
   # 127.7 65.1
   # 111.0 56.6
   # 98.1 50.1

Along the likeliest branch, each readout found the qubit decayed, so the
estimate of T1 falls from the first guess of 150 µs toward the model qubit's
29 µs, and the next wait shortens with it. ``levels=2`` keeps two levels
per qudit, enough for a T1 measurement, which makes each branch quicker to
simulate.

Over the full 50 shots, the estimate settles near the true value. On a model
qubit with a T1 of 160 µs and the paper's readout errors, 400 simulated runs
of 50 shots give a median estimate of 155 µs, where Berritta et al. report
159 µs on their processor:

.. themed-figure:: figures/adaptive_t1_estimates.svg
   :alt: Histogram of the final T1 estimates of 400 simulated runs of 50 adaptive shots each, on a model qubit with a T1 of 160 microseconds. Their median is 155 microseconds and the middle 68% lie between 123 and 195. Above them, the published estimate of 159 microseconds with its 68% interval of 131 to 202.
   :width: 100%

   The final estimates of 400 simulated runs, from
   ``benchmarks/classical/adaptive_t1_estimates.json``, beside the model's T1
   and the published estimate with its 68% interval.

Where to go next
----------------

- :example:`20_adaptive_calibration.py` tracks a drifting quantum system with
  variable waits chosen at run time.
- :example:`21_calibration_protocol.py` runs a millisecond calibration chain
  on the controller after Marciniak et al. (arXiv:2602.11912), with this
  tracker as one step: readout errors, Ramsey frequency, π and π/2
  amplitudes, then T1.
- :example:`24_pipelined_calibration.py` keeps the next batch compiled while
  the QPU runs the last one.
- :example:`26_drift_following.py` swaps programs between batches as a
  qudit drifts.

:doc:`../user_guide/classical_calls` describes the estimators that ship in
``cudaq_pulse.classical.calibration`` and what each one costs the
controller, and ``benchmarks/classical/adaptive_t1_estimates.py`` reproduces
the 400 runs.
