..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Measure and React
=================

A readout's outcome can decide what a program plays next, during the same
shot. This tutorial builds an active reset: measure the qubit, and flip it
back only if it was found excited.

What you'll learn:

- how to take shots and read their outcomes;
- how an ``if`` on a readout becomes a branch the controller takes;
- how the simulator follows every outcome at once;
- where the conditional pulse sits in the schedule.

Outcomes and shots
------------------

First, put a qubit halfway between its two states and read it out.

Set up the quantum system and pick its first qubit:

.. code-block:: python

   import numpy as np

   import cudaq_pulse as pulse

   system = pulse.quantum_system("transmon_surface17")
   qubit = system.qudits[0]

The kernel plays a calibrated π/2 pulse, then the calibrated readout:

.. code-block:: python

   @pulse.kernel
   def measure_superposition(qudit):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       half_flip = drive_line.get_calibration(pulse.gates.X90)
       measurement = readout_line.get_calibration(pulse.gates.READOUT)

       drive(drive_line, half_flip)
       sync(drive_line, readout_line)
       return readout(readout_line, measurement)

Both pulses come from the quantum system, so the kernel holds no pulse
parameters of its own.

Take 1000 shots. The ``seed`` makes them repeatable:

.. code-block:: python

   result = pulse.simulate(
       measure_superposition,
       qubit,
       shots_count=1000,
       seed=7,
   )
   records = np.asarray(result.records)

   print("first five shots:", records[:5, 0].tolist())
   print("fraction that read 1:", records.mean())

.. code-block:: text
   :class: output

   first five shots: [1, 0, 0, 1, 1]
   fraction that read 1: 0.496

What to notice: ``records`` has one row per shot and one column per readout,
and about half of the shots read 1, as a π/2 pulse should give.

.. themed-figure:: figures/superposition_shots.svg
   :alt: The 1000 shots of measure_superposition, one square per shot in the order they were taken, 100 to a row. 496 read 1 and 504 read 0, with no pattern in which do.
   :width: 100%

   The same 1000 shots, one square each. Rose squares read 1.

What to notice: each shot is a coin toss. Only the share that read 1 is
fixed by the pulse.

Branch on the outcome
---------------------

Next, use the outcome inside the same shot.

An ``if`` on a readout's result stays in the compiled program, as a branch
the controller takes while the shot runs. This kernel resets the qubit
whatever state it was in, then reads it again to check.

It starts like ``measure_superposition``:

.. code-block:: python
   :class: continues

   @pulse.kernel
   def active_reset(qudit):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       flip = drive_line.get_calibration(pulse.gates.X)
       half_flip = drive_line.get_calibration(pulse.gates.X90)
       measurement = readout_line.get_calibration(pulse.gates.READOUT)

       drive(drive_line, half_flip)
       sync(drive_line, readout_line)
       outcome = readout(readout_line, measurement)

Then it reacts to the outcome, and reads the qubit again:

.. code-block:: python

       sync(drive_line, readout_line)
       if outcome:
           drive(drive_line, flip)

       sync(drive_line, readout_line)
       return readout(readout_line, measurement)

What to notice: the π pulse inside the ``if`` plays only on the shots whose
first readout read 1.

Run it and average each readout over the shots:

.. code-block:: python

   result = pulse.simulate(
       active_reset,
       qubit,
       shots_count=1000,
       seed=7,
   )
   records = np.asarray(result.records)
   first, second = records.mean(axis=0)

   print(f"first readout read 1:  {first:.3f}")
   print(f"second readout read 1: {second:.3f}")
   print(f"ground state at the end: {result.level_population(0):.3f}")

.. code-block:: text
   :class: output

   first readout read 1:  0.496
   second readout read 1: 0.002
   ground state at the end: 0.999

.. themed-figure:: figures/reset_outcomes.svg
   :alt: The share of the active reset's 1000 shots that read 1: 0.496 at the first readout, after the pi/2 pulse, and 0.002 at the second, after the conditional pi pulse.
   :width: 100%

   The shots that read 1, before and after the reset.

What to notice:

- the first readout reads 1 in about half of the shots;
- the second reads 1 in fewer than one shot in 300, so the reset works
  whichever outcome came first;
- 99.9% of the population ends in the ground state.

How the simulator follows both outcomes
---------------------------------------

The simulator doesn't pick one outcome per shot and run the program again.
It follows every outcome of every readout as a *branch* with its
probability, and samples the shots from those branches afterwards.

Print the likely branches:

.. code-block:: python

   for branch in result.branches:
       if branch.probability < 0.01:
           continue
       probability = round(branch.probability, 3)
       print(f"readouts {branch.levels}: {probability}")

.. code-block:: text
   :class: output

   readouts (0, 0): 0.505
   readouts (1, 0): 0.494

Each branch's ``levels`` are what its readouts found. Both likely branches
end with a second readout of 0.

The branches the loop skips hold the rare paths: a qubit that decayed, or
leaked to level 2, during a readout. Because every branch is followed, the
final state is the exact average over shots, not an estimate from a sample.

.. themed-figure:: figures/reset_branches.svg
   :alt: The branches of the active reset's simulation, by the outcome of each readout: first read 0 with probability 0.505; first read 1 with probability 0.494; first read 2 with probability 7e-4. Every likely branch reads 0 the second time; 0 then 0: 0.505, 0 then 1: 1e-4, 0 then 2: 3e-8, 1 then 0: 0.494, 1 then 1: 4e-4, 1 then 2: 7e-5, 2 then 0: 1e-6, 2 then 1: 1e-5, 2 then 2: 7e-4.
   :width: 100%

   Every branch of the active reset. Line widths follow the probabilities:
   the two thick paths are the ones the loop prints.

What to notice: the faint paths are the decays and leaks, each below one
shot in a thousand.

See the branch in the schedule
------------------------------

Finally, look at where the scheduler puts the conditional pulse.

Lower the kernel and list its events, in cycles of the 2 GHz clock:

.. code-block:: python

   scheduled = active_reset.lower(qubit)

   print(f"{'kind':8} {'start':>6} {'length':>6}")
   for event in scheduled.schedule:
       start = event.start_cycles
       length = event.duration_cycles
       print(f"{event.kind:8} {start:>6} {length:>6}")

.. code-block:: text
   :class: output

   kind      start length
   drive         0     80
   readout      80    600
   drive       680     80
   readout     760    600

The drive at cycle 680 plays only on the shots whose first readout read 1.
On the other shots, its line idles for the same 80 cycles.

.. themed-figure:: figures/reset_timeline.svg
   :alt: Timeline of the active reset on D1's drive and readout lines. A pi/2 pulse, a 300 ns readout, then a pi pulse from 340 to 380 ns that plays only if the readout read 1, and a second readout from 380 ns that starts at the same time either way.
   :width: 100%

   The schedule of the active reset, in nanoseconds. The dashed π pulse
   plays only when the first readout reads 1.

What to notice: the second readout starts at cycle 760 on every shot, so
the program's timing doesn't depend on the outcome.

Where to go next
----------------

- :example:`08_readout_and_branching.py` prints the MLIR of a readout and of
  the active reset's branch.
- :example:`12_error_detection.py` shows what the verifier reports for
  malformed Pulse MLIR.
- :example:`25_precompiled_branches.py` compiles the next program of an
  adaptive experiment for each outcome ahead of time.
- :doc:`classical_calls` sends a readout to a classical function and
  branches on its answer.

For the details, :doc:`../user_guide/kernels` lists the conditions a branch
can test, and :doc:`../user_guide/compilation` what a kernel call returns.
