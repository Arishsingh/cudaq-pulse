..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Your First Pulse Program
========================

This tutorial goes from installing CUDA-Q Pulse to a simulated Rabi
oscillation on one transmon qubit. Everything runs on a laptop CPU, with no
GPU or quantum hardware.

What you'll learn:

- how to pick a quantum system and one of its qubits;
- how to write a pulse kernel;
- how to simulate it, sweep its amplitude and take shots;
- how to see the schedule the compiler made.

Install
-------

CUDA-Q Pulse needs Python 3.12 or newer, on Linux or macOS. Install it in a
virtual environment, then run the doctor:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install cudaq-pulse
   python -m cudaq_pulse.doctor

The doctor lists what was installed and checks that it works. Without a
CUDA extra, it ends with:

.. code-block:: text

   Result: OK, compilation and CPU simulation work; GPU simulation is not set up.

That is all this tutorial needs. :doc:`../getting_started` lists the
supported platforms and the GPU extras.

Pick a quantum system
---------------------

A pulse program runs on a *quantum system*: the calibrated physics of a
machine. It holds:

- each qudit's frequencies and coherence times;
- the couplings between qudits;
- the clock that pulses are timed against.

CUDA-Q Pulse ships a model of a 17-qubit superconducting processor built
for a distance-3 surface code. Take its first qubit:

.. code-block:: python

   import cudaq_pulse as pulse

   system = pulse.quantum_system("transmon_surface17")
   qubit = system.qudits[0]

   frequency_ghz = system.qudit_freq[0] / 1e9
   print(qubit)
   print(f"0-1 transition: {frequency_ghz} GHz")
   print(f"clock:          {system.clock_ghz} GHz")

.. code-block:: text
   :class: output

   <Qudit 0 -> qudit 0 of transmon_surface17>
   0-1 transition: 3.885 GHz
   clock:          2.0 GHz

CUDA-Q Pulse calls it a qudit because a transmon has more than two levels.
The simulator keeps the lowest three, so a pulse that drives too hard shows
up as population in level 2.

Write a pulse kernel
--------------------

A pulse program is a Python function with the ``@pulse.kernel`` decorator.
Each qudit has two lines:

- a drive line, which carries the microwave pulses that rotate it;
- a readout line, which measures it.

This kernel plays one pulse on the drive line, then reads the qubit out:

.. code-block:: python

   @pulse.kernel
   def rabi(qudit, amplitude: float):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       flip = gaussian(
           duration=32 * pulse.ns,
           amplitude=amplitude,
           sigma=8 * pulse.ns,
       )

       drive(drive_line, flip)
       # Start the readout after the drive pulse ends.
       sync(drive_line, readout_line)
       return readout(readout_line, square(300 * pulse.ns, 0.05))

What the kernel does:

- the drive is a Gaussian pulse, 32 ns long and 8 ns wide, at an amplitude
  you choose for each call;
- ``sync`` lines the two lines up, so the 300 ns readout starts when the
  drive pulse ends;
- the kernel returns the readout's outcome.

Inside a kernel, the operations are used by their bare names. Editors that
use Pylance or pyright complete and check them with no setup (see
:doc:`../user_guide/editors`).

Durations carry units, so the same kernel runs unchanged on a quantum
system with another clock.

Simulate it
-----------

``pulse.simulate`` compiles the kernel for the qudit's quantum system and
evolves it on the CPU, with the qubit's relaxation and dephasing included:

.. code-block:: python

   result = pulse.simulate(rabi, qubit, 0.16)
   levels = result.level_populations(qubit)

   for level, population in enumerate(levels):
       print(f"level {level}: {population:.3f}")

.. code-block:: text
   :class: output

   level 0: 0.036
   level 1: 0.963
   level 2: 0.001

At amplitude 0.16 the pulse is close to a π pulse (the sweep below puts it
at 0.165), and 96% of the population ends up excited.

Where the rest goes:

- the qubit starts in a thermal mixture with about 2% already excited,
  which the pulse flips the other way;
- it relaxes a little during the 300 ns readout.

Sweep the amplitude
-------------------

A Rabi experiment repeats the program over a range of amplitudes:

.. code-block:: python

   import numpy as np

   amplitudes = np.linspace(0, 0.4, 41)
   populations = []
   for amplitude in amplitudes:
       result = pulse.simulate(rabi, qubit, amplitude)
       populations.append(result.level_population(1))

   print("compilations:", rabi.cache_info().compiled)

.. code-block:: text
   :class: output

   compilations: 1

Because ``amplitude`` is annotated ``float``, it stays a parameter of the
compiled program instead of being fixed into it. So the 41 simulations
share one compilation.

.. themed-figure:: figures/rabi_sweep.svg
   :alt: Simulated excited-state population after the 32 ns Gaussian pulse, for drive amplitudes from 0 to 0.4, with a cosine fit. The population rises from 0.02 to 0.96; the fit puts the pi pulse at amplitude 0.165, and the population falls back near 0.33, a full turn.
   :width: 100%

   The excited-state population after the pulse, for each amplitude, with a
   cosine fit.

What to notice: the fit puts the π pulse at 0.165, and the population
returns close to zero near 0.33, a full rotation.

Take shots
----------

A simulation gives probabilities, while an experiment gives one outcome per
shot. Ask for shots, and the simulator samples them from the same
evolution:

.. code-block:: python

   result = pulse.simulate(
       rabi,
       qubit,
       0.16,
       shots_count=1000,
       seed=7,
   )
   outcomes = np.asarray(result.records)

   print("shape:", outcomes.shape)
   print("fraction that read 1:", outcomes.mean())

.. code-block:: text
   :class: output

   shape: (1000, 1)
   fraction that read 1: 0.956

``records`` holds one row per shot and one column per readout, here the
kernel's single readout.

.. themed-figure:: figures/rabi_shots.svg
   :alt: The fraction of the first n of 1000 sampled shots that read 1, against n on a log scale. It wanders at first and settles at 0.956, inside a band of two standard errors around the excited-state population of 0.963, which narrows as the shots add up.
   :width: 100%

   The mean of the first *n* shots against *n*, with a band of two standard
   errors around the population.

What to notice: the mean wanders while there are few shots, and settles
near the population as the band narrows as 1/√n.

See what was compiled
---------------------

``lower`` compiles and schedules a call without running it, so you can see
when each pulse plays:

.. code-block:: python

   scheduled = rabi.lower(qubit, 0.16)

   print(f"{'kind':8} {'start':>6} {'length':>6}")
   for event in scheduled.schedule:
       start = event.start_cycles
       length = event.duration_cycles
       print(f"{event.kind:8} {start:>6} {length:>6}")

.. code-block:: text
   :class: output

   kind      start length
   drive         0     64
   readout      64    600

Times count cycles of the 2 GHz clock. The 32 ns drive pulse is 64
cycles, and the readout starts at cycle 64, where ``sync`` put it.

Draw it
-------

``timeline()`` draws the schedule with matplotlib. Install it with the
``[viz]`` extra:

.. code-block:: bash

   pip install "cudaq-pulse[viz]"

Then draw and save the schedule:

.. code-block:: python
   :class: viz

   figure = scheduled.timeline()
   figure.savefig("rabi_schedule.svg")

.. themed-figure:: figures/rabi_schedule.svg
   :alt: Timeline of one call of the Rabi kernel on D1: a 32 ns Gaussian drive pulse, then, where sync puts it, a 300 ns square readout pulse on the readout line.
   :width: 100%

   The schedule of one call, in nanoseconds: each pulse is its envelope as
   the controller plays it. ``timeline()`` draws the same schedule as bars,
   in clock cycles.

The same program as MLIR
------------------------

``print(scheduled.mlir())`` shows the program in the Pulse dialect, the
form that compiler passes, simulators and controller backends read.

Each drive and readout in it carries the start and duration in cycles
listed above. You don't need to read MLIR to use CUDA-Q Pulse, but it's
there when you want to check what the compiler did.
:doc:`../architecture/dialects` describes it.

Where to go next
----------------

Each of these examples takes one idea from this page further. Run them from
a clone of the repository with ``python examples/<file>``:

- :example:`01_single_qubit_rabi.py` sweeps a square pulse instead, and
  compiles the kernel for a quantum system with a 1 GHz clock.
- :example:`03_t1_t2_dissipator.py` compares a simulated T1 decay with the
  exponential the model predicts, alongside Ramsey and Hahn-echo kernels.
- :example:`08_readout_and_branching.py` branches on a measurement, as in
  an active reset.
- :example:`16_visualization.py` draws the schedules of a three-qubit
  program under two scheduling policies.
- :example:`transmon/transmon_surface17_rabi.py` drives qubit D5 with its
  calibrated π pulse and follows the program down to the operator form the
  simulator evolves.

The next tutorial, :doc:`coherence`, measures T1 and dephasing with the
quantum system's calibrated pulses.

For the details behind this page:

- :doc:`../user_guide/kernels` covers what a kernel can contain;
- :doc:`../user_guide/quantum_systems` covers how quantum systems are
  described and calibrated;
- :doc:`../user_guide/compilation` covers how calls compile and what they
  return.
