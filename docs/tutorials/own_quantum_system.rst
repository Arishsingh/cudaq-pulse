..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Describe Your Own Quantum System
================================

The tutorials so far ran on ``transmon_surface17``, a model of a published
17-qubit processor. Yours has its own frequencies, coherence times
and couplings. This tutorial describes a pair of transmons from those
numbers, calibrates their π pulses, compiles one kernel for both qudits,
follows a recalibration, and saves the description to a file.

.. themed-figure:: figures/transmon_surface17.svg
   :alt: The 17 qudits of transmon_surface17 and its 24 couplings: nine data qubits, D1 to D9, in a 3 by 3 grid, and the ancillas X1, X2, X3, X4, Z1, Z2, Z3, Z4, each coupled to the four data qubits around it or, at the edge, to two. Each qudit is labelled with its 0-1 frequency: the data qubits between 3.74 and 4.14 GHz, the ancillas between 5.88 and 6.33 GHz.
   :width: 90%

   The shipped ``transmon_surface17`` quantum system, after Krinner et al.,
   Nature 605, 669 (2022), drawn from its ``physical_qudits`` and
   ``couplings``. A quantum system of your own is described the same way,
   qudit by qudit and coupling by coupling.

Qudits and a coupling
---------------------

A quantum system is built from its qudits and the couplings between them.
Each ``PhysicalQudit`` takes its index, its 0-1 frequency in Hz, and its T1
and T2* in microseconds, then the rest by keyword:

.. code-block:: python

   import json
   from pathlib import Path

   import cudaq_pulse as pulse
   from cudaq_pulse.quantum_systems import Coupling, DragPulse, PhysicalQudit

The π pulse's shape is shared by both qudits; each gets its own amplitude:

.. code-block:: python

   x_pulse = {pulse.gates.X: DragPulse(duration_ns=40.0, sigma_ns=10.0)}
   pair = pulse.quantum_system(
       name="my_pair",
       qudits={
           0: PhysicalQudit(0, 5.1e9, 60.0, 40.0, anharmonicity_hz=-2.2e8, calibrations=x_pulse),
           1: PhysicalQudit(1, 5.3e9, 45.0, 30.0, anharmonicity_hz=-2.1e8, calibrations=x_pulse),
       },
       couplings=[Coupling((0, 1), strength_hz=3.0e6)],
   )
   print(pair)   # <QuantumSystem my_pair (2 qudits)>
   print(pair.qudits[0].get_calibration(pulse.gates.X))   # DragPulse(duration_ns=40.0, sigma_ns=10.0, amplitude=0.1313058652516393)

The anharmonicity, the shift of the 1-2 transition below the 0-1 one, is
what makes a transmon a usable qubit, and simulating its third level needs
it. The coupling's ``strength_hz`` is the exchange rate g / 2π.

.. themed-figure:: figures/pair_levels.svg
   :alt: The three lowest levels of each transmon of my_pair, qudit 0: 0-1 at 5.10 GHz and 1-2 at 4.88 GHz; qudit 1: 0-1 at 5.30 GHz and 1-2 at 5.09 GHz. The 1-2 transition sits below the 0-1 by the anharmonicity. The two are coupled with g / 2 pi = 3 MHz.
   :width: 80%

   The three lowest levels of each qudit of ``pair``, from its
   ``physical_qudits``. A drive at the 0-1 frequency stays off the 1-2 line,
   hundreds of megahertz away.

The calibration gives each qudit's π pulse a shape, a 40 ns DRAG pulse with
a 10 ns width, but no amplitude. The quantum system works it out from the
shape and the qudit's drive strength, so that the pulse turns the qubit by
π. Its π/2 pulse, ``pulse.gates.X90``, is the same pulse at half the
amplitude.

Run a kernel on it
------------------

Kernels run on whichever quantum system their qudits come from, with no
other change:

.. code-block:: python

   @pulse.kernel
   def excite(qudit):
       d = get_drive_line(qudit)
       r = get_readout_line(qudit)
       drive(d, d.get_calibration(pulse.gates.X))
       sync(d, r)
       return readout(r, square(300 * pulse.ns, 0.05))


   for q in pair.qudits:
       print(q, round(pulse.simulate(excite, q).level_populations(q)[1], 3))
   # <Qudit 0 -> qudit 0 of my_pair> 0.993
   # <Qudit 1 -> qudit 1 of my_pair> 0.991
   print(excite.cache_info().compiled)   # 2

Both π pulses land within 1% of the excited state; what's missing is the
decay during the 300 ns readout. The kernel compiled twice, once per qudit,
because it reads each line's calibration while it compiles.

One compile for every qudit
---------------------------

Annotate the qudit parameter with ``pulse.Qudit`` and the compiled program
takes the qudit as an argument, so one compile serves every qudit. A
calibration can't be read inside such a kernel, since the qudit is known only
when it is called, so the host reads it and passes the amplitude in:

.. code-block:: python

   @pulse.kernel
   def excite_any(qudit: pulse.Qudit, amplitude: float):
       d = get_drive_line(qudit)
       r = get_readout_line(qudit)
       drive(d, gaussian(40 * pulse.ns, amplitude, 10 * pulse.ns))
       sync(d, r)
       return readout(r, square(300 * pulse.ns, 0.05))


   for q in pair.qudits:
       amplitude = q.get_calibration(pulse.gates.X).amplitude
       print(round(pulse.simulate(excite_any, q, amplitude).level_populations(q)[1], 3))
   # 0.993
   # 0.991
   print(excite_any.cache_info().compiled)   # 1

The drive's frequency still follows each qudit, because a drive line plays at
its own qudit's frequency unless the kernel says otherwise.

Follow a recalibration
----------------------

Measured values drift. ``replace`` makes a new quantum system with some
values changed and the rest kept, the way ``dataclasses.replace`` does, and
the same program follows it:

.. code-block:: python

   q0 = pair.physical_qudits[0]
   recalibrated = pair.replace(
       qudits={**pair.physical_qudits, 0: q0.replace(frequency_hz=5.102e9)}
   )

   for system in (pair, recalibrated):
       program = excite.lower(system.qudits[0]).export_to_controller()
       drives = [e for e in program["events"] if e["kind"] == "drive"]
       print(drives[0]["frequency_hz"])
   # 5100000000.0
   # 5102000000.0

The program for the recalibrated system plays its drive 2 MHz higher,
without a change to the kernel. Quantum systems are read-only, so the
original ``pair`` is unchanged and programs compiled for it stay valid.

Save it to a file
-----------------

A quantum system is plain data. ``as_dict`` gives it as a dictionary that
``json`` can write, and ``QuantumSystem.load`` reads the file back:

.. code-block:: python

   path = Path("my_pair.json")
   path.write_text(json.dumps(pair.as_dict(), indent=2))
   print(pulse.QuantumSystem.load(path) == pair)   # True
   print(json.loads(path.read_text())["format"])   # cudaq-pulse.system/1

When a description has mistakes, building it reports all of them at once,
each with where it is:

.. code-block:: python

   try:
       pulse.quantum_system("transmon_surface17", layout=[0, 99, 99])
   except pulse.QuantumSystemDefinitionError as error:
       for problem in error.problems:
           print(problem.path, "|", problem.message)
   # layout[1] | carrier 99 is not in quantum system 'transmon_surface17' (it has 0..16)
   # layout[2] | carrier 99 is not in quantum system 'transmon_surface17' (it has 0..16)

A ``layout`` picks which of a quantum system's qudits a program numbers 0, 1,
2 and so on, which is how a program written for a few qubits is placed on a
larger quantum system. A file with problems reports the line and column of each one.

Where to go next
----------------

- :example:`23_define_a_qpu.py` adds the control system to a quantum system:
  two controllers with their wiring, a feedback link between them, and the
  whole QPU saved and loaded as one file.
- :example:`22_runtime_qudits.py` compiles one kernel for every qudit of a
  quantum system and passes the extra frequencies it needs as arguments.
- :example:`transmon/transmon_surface17_rabi.py` uses the shipped system's
  published π pulse on qubit D5.
- :example:`26_drift_following.py` recompiles when a drifting quantum system
  changes and swaps programs between batches.

:doc:`../user_guide/quantum_systems` covers every field of a quantum system,
residual ZZ, layouts and the shipped calibration with its sources.
