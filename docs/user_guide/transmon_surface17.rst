..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

The transmon_surface17 Quantum System
=====================================

``transmon_surface17`` is the quantum system that comes with CUDA-Q Pulse:
a superconducting chip of |s17-qudits| transmons, laid out for one logical
qubit of the surface code. This page shows its layout, its frequencies, its
coherence, its calibrated gates and how a program addresses it.

.. include:: transmon_surface17.inc
   :end-before: .. start of qudits

Every number, table and figure on this page is read from the system
itself by ``docs/user_guide/figures/transmon_surface17.py``, and
``tests/test_transmon_surface17_page.py`` fails when the page and the
system differ.

:doc:`../tutorials/control_systems` puts several control systems under
this chip and compares them.

What it is
----------

The chip has |s17-qudits| transmons, in these groups:

- |s17-data-count| *data* qudits, |s17-data-labels|, hold the logical
  qubit;
- |s17-x-count| *X auxiliaries*, |s17-x-labels|, each check an X
  stabilizer of the data qudits around it;
- |s17-z-count| *Z auxiliaries*, |s17-z-labels|, each check a Z
  stabilizer.

A program names a qudit by its index or by its label. The data qudits are
indices |s17-data-indices|, the X auxiliaries |s17-x-indices| and the Z
auxiliaries |s17-z-indices|. The program clock is |s17-clock|.

Every figure on this page, and in the tutorial above, shows the groups
the same way: data qudits as grey circles, X auxiliaries as orange squares
and Z auxiliaries as purple diamonds.

Layout and couplings
--------------------

.. themed-figure:: figures/surface17_layout.svg
   :alt: The 17 qudits of transmon_surface17 on a square lattice. The 9 data qudits D1 to D9, indices 0 to 8, fill a grid row by row. Each of the 8 auxiliary qudits sits between the data qudits it couples to: X1 (index 9) couples to D2, D3; X2 (index 10) couples to D1, D2, D4, D5; X3 (index 11) couples to D5, D6, D8, D9; X4 (index 12) couples to D7, D8; Z1 (index 13) couples to D1, D4; Z2 (index 14) couples to D4, D5, D7, D8; Z3 (index 15) couples to D2, D3, D5, D6; Z4 (index 16) couples to D6, D9.
   :width: 100%

   The lattice. Each marker holds a qudit's label and, under it, its
   index.

The chip has |s17-couplings| couplings. Each one joins an auxiliary to a
data qudit; no two data qudits and no two auxiliaries are coupled. An
auxiliary on the edge of the lattice couples to fewer data qudits than one
in the middle, as the figure shows.

At the idle frequencies, the exchange between a coupled pair is
|s17-exchange|. The table at the end of this page lists every coupling.

Address it in a program
-----------------------

``pulse.quantum_system`` loads the system. ``system.qudit`` takes a label
and gives the qudit's handle, which a kernel takes as an argument:

.. code-block:: python

   import cudaq_pulse as pulse

   system = pulse.quantum_system("transmon_surface17")
   z4 = system.qudit("Z4")

   print(z4.index)
   print(system.physical_qudits[z4.index].label)
   print(system.qudit_freq[z4.index] / 1e9, "GHz")

.. code-block:: text
   :class: output

   16
   Z4
   6.037 GHz

A handle gives its qudit's calibrated pulses, and a coupling gives its CZ:

.. code-block:: python

   flip = z4.get_calibration(pulse.gates.X)
   print(type(flip).__name__, flip.duration_ns, "ns")

   pair = system.coupling(system.qudit("X1"), system.qudit("D2"))
   cz = pair.get_calibration(pulse.gates.CZ)
   print(type(cz).__name__, cz.duration_ns, "ns")

.. code-block:: text
   :class: output

   DragPulse 40.0 ns
   FluxCZ 98.0 ns

Inside a kernel, ``get_drive_line`` and ``get_readout_line`` give a
qudit's two lines, and each line gives the calibrated pulses of its own
qudit. This kernel flips a qudit and reads it out, with no pulse values of
its own:

.. code-block:: python

   @pulse.kernel
   def flip_and_measure(qudit):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       flip = drive_line.get_calibration(pulse.gates.X)
       measurement = readout_line.get_calibration(pulse.gates.READOUT)

       drive(drive_line, flip)
       sync(drive_line, readout_line)
       return readout(readout_line, measurement)


   scheduled = flip_and_measure.lower(z4)
   for event in scheduled.schedule:
       print(event.kind, event.start_cycles, event.duration_cycles)

   result = pulse.simulate(flip_and_measure, z4, readout="model")
   print(f"P(1) = {result.level_population(1):.3f}")

.. code-block:: text
   :class: output

   drive 0 80
   readout 80 400
   P(1) = 0.983

The schedule counts cycles of the |s17-clock| clock. The same kernel runs
on any other qudit of the chip: pass another handle.

Frequencies
-----------

.. themed-figure:: figures/surface17_frequencies.svg
   :alt: The frequencies of the 17 qudits, one row each, on an axis in GHz: the 0-1 transition, the 1-2 transition below it by the anharmonicity, and the readout resonator. The data qudits have their 0-1 transition at 3.740 to 4.143 GHz and their resonators at 6.769 to 7.180 GHz; the X auxiliaries have their 0-1 transition at 5.885 to 6.097 GHz and their resonators at 7.258 to 7.554 GHz; the Z auxiliaries have their 0-1 transition at 5.956 to 6.328 GHz and their resonators at 7.200 to 7.502 GHz.
   :width: 100%

   Each qudit's two lowest transitions and its readout resonator.

The data qudits idle at |s17-f-data| and the auxiliaries at |s17-f-aux|.
The data qudits and the auxiliaries are at least |s17-f-gap| apart, so a
drive on one group is far from the other. The anharmonicities are
|s17-anharmonicity|: the 1-2 transition of a qudit is that far below its
0-1 transition. The readout resonators are at |s17-f-readout|.

.. include:: transmon_surface17.inc
   :start-after: .. start of qudits
   :end-before: .. end of qudits

Readout feedlines
-----------------

The qudits are read out through |s17-feedlines| feedlines of
|s17-feedline-sizes| qudits each. On one feedline the resonators are at
least |s17-feedline-spacing| apart, so one line can carry all their
readout tones at once.

.. themed-figure:: figures/surface17_feedlines.svg
   :alt: The readout resonators of the 4 feedlines, one row per feedline, on an axis in GHz. Top left: D7 at 6.910, D4 at 7.120, Z1 at 7.316, Z2 at 7.502; top right: D1 at 6.769, D2 at 6.979, D5 at 7.180, X1 at 7.372, X2 at 7.554; bottom left: D9 at 6.868, D8 at 7.075, X3 at 7.258, X4 at 7.461; bottom right: D3 at 6.880, D6 at 7.032, Z3 at 7.200, Z4 at 7.412.
   :width: 100%

   The resonators on each feedline.

The groups are recorded in the system's calibration file, under
``readout.feedlines``, with the kind |s17-feedline-kind|: they were
traced from a published picture of the chip, not taken from a table. The
system alone has no control system, so nothing ties the groups to
hardware until a program does:
:example:`transmon/transmon_surface17_multiplexed_readout.py` puts one
group on a shared readout line, and :doc:`../tutorials/control_systems`
gives each group a controller.

Coherence
---------

T1 is |s17-t1| across the chip, the Ramsey T2* |s17-t2-star| and the echo
T2 |s17-t2-echo|.

.. include:: transmon_surface17.inc
   :start-after: .. start of coherence
   :end-before: .. end of coherence

A simulation starts each qudit in its thermal mixture and relaxes it with
its T1. It dephases with T2*, or with the echo T2 under
``simulate(..., dephasing="t2_echo")``. The readout assignment error
applies under ``simulate(..., readout="model")``; it is the chance that a
readout reports the wrong one of the two lowest levels.

Calibrated gates
----------------

.. include:: transmon_surface17.inc
   :start-after: .. start of gates
   :end-before: .. end of gates

What the simulator models, and what it does not:

- **Single-qudit pulses are simulated.** ``X`` is a |s17-x-ns|
  ``DragPulse`` with a sigma of |s17-x-sigma|. Its amplitude is a rotation
  rate, and ``get_calibration`` gives the amplitude of a half turn of the
  Bloch sphere. ``X90`` is ``X`` at half the amplitude, ``Y`` and ``Y90``
  are the same pulses a quarter turn later in phase, and ``Z``, ``S`` and
  ``T`` are phase shifts of the drive tone that take no time.
- **A simulation includes the level above the qubit's own**, so a drive
  can leak to it. The |s17-x12-ns| ``X`` of the 1-2 transition brings a
  leaked qudit back.
- **A readout is simulated as a measurement at the end of its pulse.** It
  lasts |s17-readout-data-ns| on a data qudit and |s17-readout-aux-ns| on
  an auxiliary. Under ``readout="model"`` the outcome goes through the
  qudit's assignment error.
- **The couplings are simulated**, always on, at the strengths in the
  table below.
- **Flux lines are not modeled.** On the chip a CZ is a flux pulse that
  moves a qudit's frequency. Here a ``FluxCZ`` holds the gate's duration,
  |s17-cz-ns| with its buffers of |s17-cz-buffer-ns|, and sends no pulse:
  a kernel waits for that time on both qudits. So a schedule has the right
  length, and a simulation of it has no entangling gate.

One round of error correction
-----------------------------

:example:`transmon/transmon_surface17_surface_code_cycle.py` builds one
cycle of the surface code on this chip. The chip's CZs run in
|s17-cz-steps| steps of |s17-cz-per-step| CZs each, which the system gives
as ``CZ_LAYERS``:

.. include:: transmon_surface17.inc
   :start-after: .. start of steps
   :end-before: .. end of steps

Importing the example runs it. It prints the steps and compiles the cycle:

.. code-block:: python

   from transmon_surface17_surface_code_cycle import scheduled as cycle

   print(cycle.duration)

.. code-block:: text
   :class: output abridged

   CZ step 1: Z1-D1, Z2-D5, Z3-D3
   CZ step 2: Z1-D4, Z2-D8, Z3-D5
   CZ step 3: Z2-D4, Z3-D2, Z4-D6
   CZ step 4: Z2-D7, Z3-D6, Z4-D9
   CZ step 5: X2-D1, X3-D5, X4-D7
   CZ step 6: X2-D2, X3-D6, X4-D8
   CZ step 7: X1-D3, X2-D5, X3-D9
   CZ step 8: X1-D2, X2-D4, X3-D8
   Compiled and verified: 617 ops, total = 1024 ns: 8 CZ steps of
   98 ns and 6 single-qubit slots of 40 ns (the quantum system's
   cycle, with readout, takes 1.1 us)
   1.024 us

.. themed-figure:: figures/surface17_round.svg
   :alt: The schedule of one error correction cycle on the drive lines of the 17 qudits, 1024 ns long. 8 CZ steps of 98 ns follow each other, the first from 40 ns and the last from 886 ns; in each step, 3 auxiliary-data pairs wait together, each box naming its partner. Steps 1 to 4 are on the Z auxiliaries and steps 5 to 8 on the X auxiliaries. Between the steps are 52 drive pulses of 40 ns: quarter turns on the auxiliaries around their four steps, quarter turns on the data qudits around the X steps, and an echo on every data qudit in the middle of each half.
   :width: 100%

   The cycle on the drive lines. A box is a CZ slot, named for the
   partner qudit; a curve is a drive pulse.

Read the figure from the left:

- the Z auxiliaries get a quarter turn, wait through their CZ steps, and
  get a quarter turn back;
- the X auxiliaries do the same in the second half, and there the data
  qudits get a quarter turn before and after, to check them in the X
  basis;
- in the middle of each half, every data qudit gets a half turn, an echo
  that undoes slow frequency errors.

The cycle is |s17-round-ns| long. It leaves out the readout of the
auxiliaries. :doc:`../tutorials/control_systems` adds the readout and an
active reset, and shows how the control system changes the length.

Where the numbers come from
---------------------------

The system carries its source:

.. code-block:: python

   import textwrap

   print(textwrap.fill(system.attribution, width=64))
   print(system.calibration.name, system.calibration.measurement_run)

.. code-block:: text
   :class: output

   S. Krinner, N. Lacroix et al., 'Realizing repeated quantum error
   correction in a distance-three surface code', Nature 605, 669
   (2022). arXiv:2112.03708
   transmon_surface17 2022-main

The frequencies, anharmonicities, coherence times, readout parameters and
gate durations are the values that paper publishes. The paper does not
publish everything a simulation needs. The rest is in the calibration
|s17-calibration|, which the system loads by default: the coupling
strengths, the DRAG coefficient of each ``X``, the ``X`` of the 1-2
transition, the readout assignment errors and the feedline groups. Its
measurement run |s17-run| is described in the file as "|s17-run-is|".

Each calibrated value records whether it is published, read from a
figure, or fitted, and its source.
:doc:`quantum_systems` explains the calibration and how to read one value
with its source.

Couplings
---------

.. include:: transmon_surface17.inc
   :start-after: .. start of couplings
   :end-before: .. end of couplings
