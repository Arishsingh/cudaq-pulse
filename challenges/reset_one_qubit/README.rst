..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Reset one qubit
===============

Level: Starter.

Qubit D1 of ``transmon_surface17`` is in a state you do not know. Bring it to level 0 within
1200 ns.

Waiting does not do it. D1 is warm: left alone for ever, it is still found outside level 0 2.1 %
of the time, and 1200 ns is far too short to get even there. Measure it, and flip it only if it
read 1.

Write one file, ``my_reset.py``, with a kernel ``reset(qudit)``, and score it:

.. code-block:: console

   python challenges/reset_one_qubit/score.py my_reset.py

The scorer prepares the state and checks the result. Your kernel does neither, and returns
nothing. The scorer tries 8 states that only it knows, each a superposition of levels 0 and 1,
and plays your reset on each with D1's decay and readout errors. It prints, state by state, what
is left outside level 0 at 1200 ns after the start, then ``BRONZE``, ``SILVER``, ``SCORE`` (the
worst state and the mean) and ``SEED``. Each run draws new states; give the seed as a second
argument to repeat a run.

Your kernel is compiled with ``schedule="rcp"`` for a controller that needs 200 ns to act on a
readout, so a pulse inside an ``if`` on a readout starts 200 ns after that readout ends. The
tutorial's listing shows it starting at once; yours will not.

Limits
------

- Only D1's drive pulses, waits and readouts.
- Every readout is D1's calibrated readout, 300 ns long, and plays in every shot: a readout
  does not go inside an ``if``. At most 2 of them: the controller starts a readout 200 ns or more
  after the one before it ends, so no more fit in 1200 ns.
- No pulse plays while D1 is read out, a pulse of amplitude 0 included.
- A pulse in a branch depends on the outcome of one readout (``if outcome:`` or its ``else``),
  and starts 200 ns or more after that readout ends.
- Pulses are at D1's frequency, with no frame change before them, and every sample is at most
  0.5 in magnitude (the calibrated X pulse peaks at 0.13). At most 16 pulses.
- Everything ends within 1200 ns: the last pulse, readout or wait.
- A run with a seed given repeats known states, so it prints its rungs but no score.

The scorer checks and plays what your file's process reports. That the report is what your
``reset`` plays is on your honour (see "What a scorer cannot check" in ``challenges/README.rst``).

Rungs
-----

.. list-table::
   :header-rows: 1

   * - Rung
     - Pass line
     - Hint
   * - Bronze
     - Every state ends with less than 0.021 outside level 0: less than waiting for ever
       leaves.
     - The ``active_reset`` kernel of "Measure and React", without the pulse that prepares the
       qubit and without the readout that checks it.
   * - Silver
     - Every state ends with less than 0.003 outside level 0.
     - Which states end worst, and why? The scorer prints when your reset is over. What could the
       rest of the 1200 ns do?

Getting started
---------------

To try a reset yourself, start D1 in level 1 and ask what is left at 1200 ns. Save this next to
``my_reset.py``, as ``try_reset.py``, and run ``python try_reset.py``:

.. code-block:: python

   import cudaq_pulse as pulse
   from my_reset import reset

   system = pulse.quantum_system("transmon_surface17")
   qubit = system.qudit("D1")
   controller = pulse.Controller(clock_ghz=system.clock_ghz, feedback_latency_cycles=400)

   compiled = pulse.compile(
       reset, quantum_system=system, control_system=controller, schedule="rcp"
   )
   program = compiled.lower(qubit)
   result = pulse.simulate(
       program, initial_state=[0, 1, 0], readout="model", t_end=1200 * pulse.ns
   )
   print("outside level 0:", 1 - result.level_population(0))

``readout="model"`` gives the readout D1's assignment errors; ``initial_state`` lists the
amplitudes of levels 0, 1 and 2.

Expect about 20 minutes for bronze after the tutorial "Measure and React".
