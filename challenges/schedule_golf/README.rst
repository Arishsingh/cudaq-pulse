..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Schedule golf
=============

Level: Starter.

``program.py`` plays one parity check of the surface code on five qubits of
``transmon_surface17`` and then reads all five out. It takes 1312 ns. Make it finish sooner
without changing what it does.

Copy ``program.py``, edit the copy and score it:

.. code-block:: console

   cp challenges/schedule_golf/program.py my_program.py
   python challenges/schedule_golf/score.py my_program.py

Your file defines ``kernel(z2, d5, d8, d4, d7)`` and ``SCHEDULE``, the scheduling policy the
scorer compiles it with: ``"asap"``, ``"alap"``, ``"rcp"`` or ``"alap_rcp"``. Change either as you
like. The scorer prints ``INVALID`` and why for each rule your program breaks, then ``BRONZE`` and
``SILVER``, pass or fail, and a ``SCORE`` line with ``duration_ns``: when the last pulse, CZ slot
or readout ends.

What must stay
--------------

The circuit: every qubit plays the operations it plays in ``program.py``, in the same order.

- Z2: a half flip, a CZ with D5, with D8, with D4 and with D7 in that order, a half flip, its
  readout.
- D5 and D8: their CZ, the echo pi pulse, their readout.
- D4 and D7: the echo pi pulse, their CZ, their readout.

A pulse or readout is its qubit's calibrated one: the scorer compares its samples, its frequency
and its phase with those ``program.py`` plays. A CZ is a flux pulse, which is not modeled yet: its slot is Z2 and the data qubit both waiting 98 ns on their
drive lines from the same moment. On one qubit, an operation starts when the one before it has
ended, or later. Nothing else ties the qubits together: a data qubit may be read out while Z2 is
still busy.

Limits
------

- The control system plays at most two readouts and at most four drive pulses at once.
- Only pulses, readouts, waits and syncs on the five qubits. No pulse, readout or wait depends
  on a measurement.
- A wait of exactly 98 ns on a drive line is half of a CZ slot; one that is not, is refused. Any
  other wait is allowed and only takes time.
- What is not played is not compared: a frame change that is put back before the next pulse
  leaves that pulse as it was, and passes.

``"asap"`` and ``"alap"`` do not look at the first limit and the compiler does not complain: the
scorer counts what plays at once. ``"rcp"`` and ``"alap_rcp"`` keep it.

Rungs
-----

.. list-table::
   :header-rows: 1

   * - Rung
     - Pass line
     - Hint
   * - Bronze
     - Valid, and at most 1100 ns.
     - A ``sync`` over all five drive lines is a barrier: every qubit waits for the slowest.
       Which lines does a CZ need lined up?
   * - Silver
     - Valid, and at most 978 ns.
     - Five readouts, two at a time; a data qubit's lasts 300 ns and Z2's 200 ns. Who should
       share a lane? ``"rcp"`` places operations in the order the kernel states them and never
       moves a later one ahead.

978 ns is the least a valid schedule of this program can take, so silver is the end.

The scorer checks the schedule your file's process reports, not how it was made: that it comes
from your kernel is on your honour (see "What a scorer cannot check" in ``challenges/README.rst``).

Getting started
---------------

.. code-block:: console

   python challenges/schedule_golf/program.py

prints the schedule: every pulse, wait and readout with its qubit and its times. Once the barriers
are gone, a CZ slot still needs its own two drive lines lined up: ``sync`` the two before their
waits, and after them too if the policy is ``"alap"`` or ``"alap_rcp"``.

The tutorials "Several Qubits: Registers, Sync and Scheduling" and "From Kernel to Controller"
(its sections "Check a limit in Python" and "Let the compiler keep the limit") cover what the
program uses. Expect 20 to 40 minutes for bronze after them.
