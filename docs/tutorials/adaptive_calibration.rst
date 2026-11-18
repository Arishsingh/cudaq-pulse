..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Adaptive Calibration in One Program
===================================

A qubit's T1 drifts over minutes to hours, and sweeping a fixed set of
waits to recalibrate it wastes most of the shots. An adaptive measurement
picks each wait from what the earlier shots found.

This tutorial builds the Bayesian T1 tracker of Berritta et al. (Phys. Rev.
X 16, 011025 (2026), arXiv:2506.09576) as a single kernel.

What you'll learn:

- how to run an estimator on the controller between shots;
- how to write a wait whose length is known only at run time;
- how to compile a whole adaptive run as one program;
- how to follow its branches in simulation.

The estimator
-------------

The estimator keeps a probability distribution over the decay rate 1/T1:
a gamma distribution with two parameters, ``shape`` and ``rate``. Its T1
estimate is rate / shape.

Each readout updates the distribution, which then says which wait would
teach the most next. CUDA-Q Pulse ships both steps. Declared as a
classical function on the controller's own link, they run between shots.
It takes the readout, the distribution and the last wait, and returns the
updated distribution and the next wait:

.. code-block:: python

   import cudaq_pulse as pulse
   from cudaq_pulse.classical import calibration as cal


   @pulse.classical(link="controller", max_latency_ns=500)
   def update_t1(
       bit: pulse.Bits,
       shape: float,
       rate: float,
       wait_ns: float,
   ) -> tuple[float, float, float]:
       shape, rate = cal.adaptive_t1_update(
           shape,
           rate,
           bit[0],
           wait_ns,
           error_1to0=0.11,
           error_0to1=0.14,
       )
       next_wait_ns = cal.adaptive_t1_wait(shape, rate)
       next_wait_ns = min(max(next_wait_ns, 100.0), 200_000.0)
       return shape, rate, next_wait_ns

What the arguments mean:

- ``error_1to0`` is the chance the readout reports 0 for an excited qubit,
  and ``error_0to1`` the chance it reports 1 for a qubit in its ground
  state, as in the paper;
- the next wait is kept between 100 ns and 200 µs;
- the estimator has 500 ns to answer, a budget for code on the controller.

.. pulse-visual:: figures/adaptive_calibration.json
   :scene: t1_belief
   :alt: One readout moves the estimate. Before any shot, the prior puts T1 at 150 µs. The next wait would be 76.5 µs. Reading 1 after the 50 µs wait: still excited, so the estimate rises to 163 µs. The next wait would be 83.1 µs. Reading 0 after the 50 µs wait: decayed, so it falls to 128 µs. The next wait would be 65.1 µs.

   The prior this page uses, and where one readout after a 50 µs wait moves
   it, from ``cal.adaptive_t1_update``, with the next wait ``update_t1``
   answers.

What to notice: a qubit still excited after the wait raises the estimate.
One that decayed lowers it.

The tracking kernel
-------------------

Every shot is one pass of a loop inside one kernel, so the whole run is one
program.

The kernel takes the number of shots, the prior, and the first wait. Each
shot excites the qubit, waits, reads it out, asks the estimator for the
next wait, and resets the qubit:

.. code-block:: python

   @pulse.kernel
   def track_t1(
       qudit,
       shots_count: int,
       shape: float,
       rate: float,
       wait_ns: float,
   ):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       flip = drive_line.get_calibration(pulse.gates.X)
       measurement = readout_line.get_calibration(pulse.gates.READOUT)

       for shot in range(shots_count):
           drive(drive_line, flip)
           wait(
               drive_line,
               wait_ns * pulse.ns,
               bounds=(100 * pulse.ns, 200 * pulse.us),
           )
           sync(drive_line, readout_line)
           outcome = readout(readout_line, measurement)
           request = update_t1.submit([outcome], shape, rate, wait_ns)
           shape, rate, wait_ns = request.result()

           sync(drive_line, readout_line)
           if outcome:
               drive(drive_line, flip)

What to notice:

- the wait's length comes from the estimator's answer, so it is known only
  when the program runs;
- ``bounds`` gives the scheduler its range, so it can still check the
  timing of everything around it;
- the last π pulse is the active reset from :doc:`measure_and_react`.

Compile the full run
--------------------

The controller link belongs to the control system. Here it adds 4 ns on the
way out and 8 ns on the way back:

.. code-block:: python

   system = pulse.quantum_system("transmon_surface17")
   qubit = system.qudits[0]

   controller_link = pulse.Link(fixed_ns=4, feedback_ns=8)
   links = {"controller": controller_link}
   control_system = pulse.ControlSystem(links=links)
   compiled = pulse.compile(track_t1, control_system=control_system)

Lower 50 shots from a prior of shape 3 and rate 450 µs, a first guess of
T1 = 150 µs:

.. code-block:: python

   scheduled = compiled.lower(
       qubit,
       shots_count=50,
       shape=3.0,
       rate=450_000.0,
       wait_ns=50_000.0,
   )
   calls = scheduled.classical_report()
   print("calls to the estimator:", len(calls))

.. code-block:: text
   :class: output

   calls to the estimator: 50

Each of the 50 calls has its own place in the schedule.

What the annotations decide:

- ``shots_count`` is an ``int``, so the loop's length is fixed when the
  kernel compiles;
- the prior and the first wait are ``float`` arguments, so a run from
  another prior compiles nothing new.

.. themed-figure:: figures/adaptive_shots.svg
   :alt: Timeline of the first 3 shots of track_t1 on D1's drive and readout lines and the controller link, with the long first wait cut short. Each shot plays a pi pulse, waits, and reads out; the outcome goes to update_t1, which answers within 500 ns plus the link's 4 and 8 ns, and a reset pulse plays only if the qubit read 1. The first wait is the 50 microseconds the call passes; each later wait is set by the answer, between 100 ns and 200 microseconds.
   :width: 100%

   The first three shots, with the 50 µs first wait cut short. Each answer
   sets the next wait, dashed because the schedule shows it at its 100 ns
   lower bound.

What to notice: the dashed reset pulses play only when the qubit read 1.

Simulate a few shots
--------------------

Every readout splits the simulation into two branches, and every branch
asks the estimator for its own next wait. Three shots make eight branches,
few enough to follow exactly.

``levels=2`` keeps two levels per qudit, enough for T1, which makes each
branch quicker to simulate:

.. code-block:: python

   three_shots = compiled.lower(
       qubit,
       shots_count=3,
       shape=3.0,
       rate=450_000.0,
       wait_ns=50_000.0,
   )
   result = pulse.simulate(three_shots, levels=2)
   print("branches:", len(result.branches))

.. code-block:: text
   :class: output

   branches: 8

Follow the likeliest branch and print what the estimator answered after
each shot:

.. code-block:: python

   likeliest = max(
       result.branches,
       key=lambda branch: branch.probability,
   )

   print(f"{'T1 estimate':>11}  {'next wait':>10}")
   for shape, rate, wait_ns in likeliest.answers("update_t1"):
       t1_us = rate / shape / 1000
       wait_us = wait_ns / 1000
       print(f"{t1_us:8.1f} µs  {wait_us:7.1f} µs")

.. code-block:: text
   :class: output

   T1 estimate   next wait
      127.7 µs     65.1 µs
      111.0 µs     56.6 µs
       98.1 µs     50.1 µs

Along this branch, each readout found the qubit decayed. The estimate falls
from the first guess of 150 µs toward the model qubit's 29 µs, and the next
wait shortens with it.

.. themed-figure:: figures/adaptive_branches.svg
   :alt: The T1 estimate after each of three shots, along each of the 8 branches of the simulation, from the first guess of 150 microseconds. After three shots the branches end between 98 and 207 microseconds. The likeliest branch, with probability 0.60, ends at 98.1.
   :width: 100%

   The estimate along all eight branches. Line widths follow the
   probabilities.

What to notice: each branch has its own readouts, so each ends at its own
estimate.

Over many shots
---------------

Over the full 50 shots, the estimate settles near the true value.

On a model qubit with a T1 of 160 µs and the paper's readout errors, 400
simulated runs of 50 shots give a median of 155 µs. Berritta et al. report
159 µs on their processor.

.. themed-figure:: figures/adaptive_t1_estimates.svg
   :alt: Histogram of the final T1 estimates of 400 simulated runs of 50 adaptive shots each, on a model qubit with a T1 of 160 microseconds. Their median is 155 microseconds and the middle 68% lie between 123 and 195. Above them, the published estimate of 159 microseconds with its 68% interval of 131 to 202.
   :width: 100%

   The final estimates of 400 simulated runs, beside the model's T1 and the
   published estimate with its 68% interval.

What to notice: the runs scatter, but their median lands within a few
microseconds of both.

Where to go next
----------------

- :example:`20_adaptive_calibration.py` tracks a drifting quantum system
  with waits chosen at run time.
- :example:`21_calibration_protocol.py` runs a millisecond calibration chain
  on the controller after Marciniak et al. (arXiv:2602.11912), with this
  tracker as one step.
- :example:`24_pipelined_calibration.py` keeps the next batch compiled while
  the QPU runs the last one.
- :example:`26_drift_following.py` swaps programs between batches as a
  qudit drifts.

:doc:`../user_guide/classical_calls` describes the estimators that ship in
the calibration module and what each costs the controller. The script
``benchmarks/classical/adaptive_t1_estimates.py`` reproduces the 400 runs.
