..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Several Qubits: Registers, Sync and Scheduling
==============================================

Programs on more than one qubit need two things a single-qubit program
doesn't: a way to pass a group of qubits to a kernel, and a way to say which
pulses must line up in time. This tutorial drives a register of qubits in
parallel, lines up their readouts, compares two ways the compiler can place
pulses, and then lets two coupled qubits feel each other.

A register of qubits
--------------------

Slicing ``qs.qudits`` gives a *register*, which a kernel iterates like a
list:

.. code-block:: python

   import cudaq_pulse as pulse

   qs = pulse.quantum_system("transmon_surface17")
   data = qs.qudits[0:3]   # D1, D2 and D3


   @pulse.kernel
   def flip_all(qudits):
       for q in qudits:
           d = get_drive_line(q)
           drive(d, d.get_calibration(pulse.gates.X))

The loop runs while the kernel compiles, once per qubit of the register.
Lower it to see when each pulse plays:

.. code-block:: python

   scheduled = flip_all.lower(data)
   for event in scheduled.schedule:
       print(event.kind, event.qudit, event.start_cycles)
   # drive 0 0
   # drive 1 0
   # drive 2 0

Each qubit has its own drive line, so the three π pulses play at the same
time. Each line uses its own qubit's calibrated pulse, since every qubit has
its own frequency and drive strength.

.. code-block:: python

   result = pulse.simulate(flip_all, data)
   print([round(result.level_populations(q)[1], 3) for q in data])   # [0.978, 0.992, 0.984]

The simulation holds only the qubits the program uses, here a
27-dimensional state space for three three-level transmons. The other 14 qubits of the quantum system
stay out of it, in their ground state.

Line pulses up with sync
------------------------

Pulses on different lines are independent until ``sync`` ties them
together: every line it names waits for the latest of them. Here one qubit
gets three π pulses and the other one, and both are read out together
afterwards:

.. code-block:: python

   @pulse.kernel
   def uneven(a, b):
       da, db = get_drive_line(a), get_drive_line(b)
       ra, rb = get_readout_line(a), get_readout_line(b)
       x = da.get_calibration(pulse.gates.X)
       drive(da, x)
       drive(da, x)
       drive(da, x)
       drive(db, db.get_calibration(pulse.gates.X))
       sync(da, db, ra, rb)   # both readouts start after the last pulse
       readout(ra, ra.get_calibration(pulse.gates.READOUT))
       readout(rb, rb.get_calibration(pulse.gates.READOUT))

Where should the second qubit's single pulse go: at the start, or just
before the readouts? By default the compiler schedules every operation as
late as possible, which keeps a qubit's state fresh until it is measured.
``schedule="asap"`` places operations as early as possible instead:

.. code-block:: python

   for policy in ("alap", "asap"):
       program = pulse.compile(uneven, schedule=policy).lower(data[0], data[1])
       starts = [e.start_cycles for e in program.schedule if e.qudit == 1 and e.kind == "drive"]
       print(policy, starts)
   # alap [160]
   # asap [0]

.. themed-figure:: figures/alap_asap.svg
   :alt: Two timelines of the uneven kernel on D1 and D2. D1 plays three pi pulses and D2 one, and both readouts start at 120 ns. As late as possible, D2's pulse plays from 80 ns, right before the readouts; as early as possible, from 0 ns, and D2 then idles until it is measured.
   :width: 100%

   ``uneven`` under both policies, in nanoseconds of the 2 GHz clock. Only
   the second qubit's π pulse moves; the readouts start at the same time.

As late as possible, the pulse plays at cycle 160, right before the
readouts at cycle 240. As early as possible, it plays at cycle 0 and the
qubit then idles, and decays, for 160 cycles before it is measured. The
readouts start at cycle 240 either way.

Two coupled qubits
------------------

D1 isn't coupled to D2 or D3, but it is coupled to the ancilla X2, through
an exchange interaction of a few megahertz:

.. code-block:: python

   x2 = qs.qudit("X2")
   print(f"{qs.coupling(data[0], x2).strength_hz:.3g}")   # 5.69e+06, g / 2π in Hz
   print(round(qs.calibration.idle_zz_hz("X2-D1")))   # -5936, in Hz

Because of the coupling, D1's frequency depends slightly on whether X2 is
excited, by about 6 kHz. This is the residual ZZ interaction, a
two-qubit error source that matters for every gate played next to a busy
neighbour. A Ramsey sequence on D1 can see it: excite X2 first or don't, and
compare.

.. code-block:: python

   @pulse.kernel
   def ramsey_next_to(qubit, neighbour, excite: int, tau_ns: float):
       d, n = get_drive_line(qubit), get_drive_line(neighbour)
       r = get_readout_line(qubit)
       if excite:
           drive(n, n.get_calibration(pulse.gates.X))
       sync(d, n)
       x90 = d.get_calibration(pulse.gates.X90)
       drive(d, x90)
       wait(d, tau_ns * pulse.ns)
       drive(d, x90)
       sync(d, r)
       return readout(r, r.get_calibration(pulse.gates.READOUT))

The two π/2 pulses on D1 are the Ramsey sequence; the π pulse on X2 before
them sets the neighbour's state. Compare the two cases after 10 µs:

.. code-block:: python

   for excite in (0, 1):
       result = pulse.simulate(ramsey_next_to, data[0], x2, excite, 10_000.0)
       print(excite, round(result.level_populations(data[0])[1], 3))
   # 0 0.677
   # 1 0.587

After a 10 µs wait, D1's excited-state population differs by about 0.09
depending on its neighbour, from a phase difference of about 0.37 radians:
6 kHz over 10 µs. ``excite`` is annotated ``int``, so the kernel compiles
once for each value and the ``if`` is decided while compiling, not during
the shot.

.. themed-figure:: figures/zz_ramsey.svg
   :alt: D1's excited-state population against the Ramsey wait, from 0 to 20 microseconds, with its neighbour X2 in its ground state and excited. The two curves start together and separate as the wait grows, by 0.08 at 20 microseconds.
   :width: 100%

   Ramsey on D1 with its neighbour X2 in the ground state or excited. The
   curves separate as the ZZ shift accumulates phase.

For larger groups of coupled qubits, :doc:`gpu_simulation` runs the same
programs on an NVIDIA GPU.

Where to go next
----------------

- :example:`07_multi_qubit_sync.py` builds programs on registers of qudit
  handles, with ``sync`` barriers between groups of lines.
- :example:`10_scheduling_comparison.py` compares as-late-as-possible,
  as-early-as-possible and resource-constrained schedules of one four-qubit
  kernel.
- :example:`18_per_kernel_schedules.py` gives each kernel of a stabilizer
  cycle its own scheduling policy.
- :example:`02_two_qubit_cross_resonance.py` and
  :example:`17_ghz_state_prep.py` drive one qubit at another's frequency to
  build two-qubit gates and a GHZ state; their amplitudes are illustrative,
  not calibrated for this quantum system.

:doc:`../user_guide/kernels` covers registers and ``*qudits`` parameters, and
:doc:`../user_guide/compilation` what each annotation makes static or keeps a
runtime value.
