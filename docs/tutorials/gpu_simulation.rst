..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Simulating on a GPU
===================

Every simulation in the earlier tutorials ran on the CPU, with the NumPy
reference simulator. The same programs also run on an NVIDIA GPU, through
NVIDIA cuDensityMat, with one argument changed. This tutorial installs GPU
support, checks it, runs a pair of coupled transmons on both backends, and
looks at the form the program takes on its way to the GPU.

.. note::

   GPU simulation is a research preview: see :doc:`../user_guide/gpu_execution`
   for what it supports and what it doesn't yet. The GPU code on this page
   needs Linux and an NVIDIA GPU, and the repository's tests run it only
   there. Its CPU parts run anywhere.

Install and check
-----------------

GPU support comes as an extra of the same package, for CUDA 12 or CUDA 13.
The CUDA libraries come from PyPI too, so the NVIDIA driver is the only thing
the machine needs:

.. code-block:: bash

   pip install "cudaq-pulse[cu13]"   # or "cudaq-pulse[cu12]"
   python -m cudaq_pulse.doctor

The doctor reports the GPU runtime it loaded, the driver version and the
number of GPUs, and exits with an error when GPU execution doesn't work. The
same check from Python raises an error that says what is missing:

.. code-block:: python
   :class: gpu

   from cudaq_pulse.runtime import require_gpu_runtime

   require_gpu_runtime()

A coupled pair on the CPU
-------------------------

The ancilla X1 and the data qubit D2 of ``transmon_surface17`` are coupled,
about 2.1 GHz apart. Give X1 a π/2 pulse and D2 a π pulse, then let both idle
for the slot their CZ gate would take:

.. code-block:: python

   import numpy as np

   import cudaq_pulse as pulse

   qs = pulse.quantum_system("transmon_surface17")
   x1, d2 = qs.qudit("X1"), qs.qudit("D2")
   cz_slot = qs.coupling(x1, d2).get_calibration(pulse.gates.CZ).duration_ns * pulse.ns

The coupling between them carries its CZ calibration, whose duration sets the
slot. The kernel plays both pulses and idles both lines for it:

.. code-block:: python

   @pulse.kernel
   def evolve_pair(a, b):
       da, db = get_drive_line(a), get_drive_line(b)
       drive(da, da.get_calibration(pulse.gates.X90))
       drive(db, db.get_calibration(pulse.gates.X))
       sync(da, db)
       wait(da, cz_slot)   # the CZ's 98 ns slot; flux pulses aren't simulated yet
       wait(db, cz_slot)

Simulate it on the CPU, with a step fine enough for the pair's detuning:

.. code-block:: python

   cpu = pulse.simulate(evolve_pair, x1, d2, dt_ns=0.02)
   print(cpu.dims)   # (3, 3)
   print([round(cpu.level_populations(q)[1], 4) for q in (x1, d2)])   # [0.4945, 0.989]

The simulation holds only X1 and D2, each with three levels, so the state is
a 9 × 9 density matrix. X1 ends half excited and D2 nearly fully, less what
decays in the program's 138 ns. The step, ``dt_ns=0.02``, resolves the coupling, which
turns at the pair's 2.1 GHz detuning in their rotating frames.

The same call on the GPU
------------------------

``backend="gpu"`` runs the same compiled program through cuDensityMat:

.. code-block:: python
   :class: gpu

   gpu = pulse.simulate(evolve_pair, x1, d2, backend="gpu", dt_ns=0.02)
   print([round(gpu.level_populations(q)[1], 4) for q in (x1, d2)])
   print(np.abs(gpu.density_matrix - cpu.density_matrix).max())

The result has the same layout as the CPU's, so the two density matrices
can be compared entry by entry, as the last line does. ``integrator`` picks
the GPU's time-stepping scheme: fixed-step Runge-Kutta of order 1, 2 or 4
(``"rk4"`` by default), a Magnus expansion, or Crank-Nicolson for stiff
problems. Long stretches where nothing plays, such as the CZ slot here, are
propagated exactly on the host instead of stepped.

What runs on the GPU
--------------------

The compiler lowers the pulse program through an operator dialect, QOp,
which writes the Hamiltonian and the Lindblad terms of the decay, to calls
into the cuDensityMat runtime. You can see the last stage on any machine:

.. code-block:: python

   llvm = evolve_pair.lower(x1, d2).mlir("llvm")
   print(llvm.count("llvm.call"))   # 182

The pulse program you simulate on the GPU is the same one you would export
to a control system; only the last steps of compilation differ.

.. themed-figure:: figures/lowering.svg
   :alt: The page's evolve_pair kernel at each level of its lowering for GPU simulation: Pulse MLIR, 21 operations; QOp, 82 operations; cuDensityMat, 226 operations; LLVM IR, 1845 operations, of them 182 calls into the cuDensityMat runtime. From the Pulse MLIR the same program also exports for a controller as 5 events.
   :width: 100%

   ``evolve_pair`` at each level of ``mlir()``, with the operations each one
   holds, counted from its text. The same Pulse MLIR is what
   ``export_to_controller()`` turns into a controller's events.

Where to go next
----------------

- :example:`transmon/transmon_surface17_evolve_gpu.py` runs this pair on the
  GPU and prints its final state (it needs an NVIDIA GPU).
- :example:`transmon/transmon_surface17_full_pipeline.py` prints each stage of
  the lowering: the scheduled Pulse MLIR, the QOp form and the LLVM IR.

:doc:`../user_guide/gpu_execution` lists the driver requirements, the GPU
errors and what each means, and the features GPU simulation doesn't support
yet; :doc:`../architecture/dialects` describes QOp and cuDensityMat.
