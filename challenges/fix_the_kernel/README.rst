..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Fix the kernel
==============

Level: Starter.

``broken.py`` holds one kernel, ``two_turns(qudit, turn, gap_ns)``, with five mistakes in it. The
compiler refuses some of them. The others compile and play something else. Fix all five.

What ``two_turns`` is for: turn the qubit by ``turn`` of a full flip with one pulse (1 is a pi
pulse, 0.5 half of one), wait ``gap_ns``, turn it again by as much with a second pulse, and read
it out when the second pulse ends.

Copy ``broken.py``, fix the copy and score it:

.. code-block:: console

   cp challenges/fix_the_kernel/broken.py my_kernel.py
   python challenges/fix_the_kernel/score.py my_kernel.py

The scorer compiles your ``two_turns`` for qubit D1 of ``transmon_surface17`` on a bench
controller, 12 times: for 12 pairs of ``turn`` and ``gap_ns`` that it draws anew at every run. It
checks what each one plays, not what the file says, and prints one line per case: ``ok``, or
``FAIL`` with the first thing that is wrong (the compiler's message, if the case does not
compile, on lines of its own marked ``|``). Where no case compiles, and all for one reason, it
prints that message once, as ``INVALID``. Then ``BRONZE``, ``SILVER``, ``SCORE`` (the number of cases that are ok) and
``SEED``; give the seed as a second argument to repeat a run.

What a case must do
-------------------

- It plays two drive pulses and then one readout on D1, and nothing else. None of them, and no
  wait, depends on a measurement.
- From level 0, each pulse by itself leaves ``sin(pi * turn / 2) ** 2`` in level 1, to within
  0.02, and the two with their wait ``sin(pi * turn) ** 2``, to within 0.03.
- The second pulse starts ``gap_ns`` after the first ends.
- The readout starts when the second pulse ends and lasts 300 ns or more. It is on the readout
  line's own tone, with no frame change before it, and reaches 0.01 or more: less is silent.

Limits
------

- The bench plays at most 0.25: every sample of a pulse or of the readout is at most 0.25 in
  magnitude.
- A pulse lasts at most 64 ns and is within 50 MHz of the qubit's 0-1 frequency.
- ``turn`` is between 0.1 and 1, and ``gap_ns`` a whole number of ns between 4 and 200.
- A run with a seed given repeats known cases, so it prints its rungs but no score.

The scorer checks what your file's process reports for each case. That the report is what your
``two_turns`` plays is on your honour (see "What a scorer cannot check" in
``challenges/README.rst``).

Rungs
-----

.. list-table::
   :header-rows: 1

   * - Rung
     - Pass line
     - Hint
   * - Bronze
     - Every case with a turn of at most 0.7 is ok.
     - Take the ``FAIL`` lines one at a time, and look at what your kernel plays (below). A
       number without a unit is a number of clock cycles.
   * - Silver
     - Every case is ok.
     - A large turn asks for more amplitude than the bench plays. A pulse turns the qubit by its
       amplitude times its length, and a Gaussian's length is its ``sigma`` as well as its
       ``duration``.

Getting started
---------------

To see what your kernel plays for one case, compile it as the scorer does and print its schedule.
``unit_warnings=True`` makes the compiler warn where a time has no unit. Save this next to
``my_kernel.py``, as ``look.py``, and run ``python look.py``:

.. code-block:: python

   import cudaq_pulse as pulse
   from my_kernel import two_turns

   system = pulse.quantum_system("transmon_surface17")
   qubit = system.qudit("D1")
   bench = pulse.Controller(clock_ghz=system.clock_ghz, max_amplitude=0.25)

   compiled = pulse.compile(
       two_turns, quantum_system=system, control_system=bench, unit_warnings=True
   )
   program = compiled.lower(qubit, 0.5, 30.0)
   for event in program.schedule:
       start_ns = event.start_cycles / system.clock_ghz
       length_ns = event.duration_cycles / system.clock_ghz
       print(f"{event.kind:8} starts at {start_ns:6.1f} ns and lasts {length_ns:6.1f} ns")

Expect 20 to 30 minutes for bronze after the tutorial "Your First Pulse Program".
