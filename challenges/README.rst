..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Challenges
==========

Small problems to solve with cudaq-pulse. Each folder holds the statement (``README.rst``), a
scorer (``score.py``) and reference solutions (``reference/``). You write one file, run the scorer
on it and read the result:

.. code-block:: console

   python challenges/pi_pulse/score.py my_pulse.py

.. list-table::
   :header-rows: 1

   * - Challenge
     - Level
     - What you do
   * - `Find the pi pulse <pi_pulse/README.rst>`_
     - Starter
     - Halve the length of a calibrated X pulse.
   * - `Mystery qubit <mystery_qubit/README.rst>`_
     - Starter
     - Measure the T1 of a qubit from shots alone.
   * - `Benchmark it yourself <benchmark_it_yourself/README.rst>`_
     - Starter
     - Find a pulse's error by randomized benchmarking.
   * - `Schedule golf <schedule_golf/README.rst>`_
     - Starter
     - Make a five-qubit program finish sooner without changing what it does.
   * - `Fix the kernel <fix_the_kernel/README.rst>`_
     - Starter
     - Find the five mistakes in a kernel, by what it plays.
   * - `Reset one qubit <reset_one_qubit/README.rst>`_
     - Starter
     - Bring a qubit in an unknown state to its ground state, in time.

Rungs
-----

Each challenge has a bronze and a silver rung. Bronze is reached from the tutorials; silver needs
one idea more. Starter challenges have no ranking: post your ``SCORE`` line on the challenge's
issue, and say what in the documentation slowed you down.

How scoring works
-----------------

Your file never runs in the scorer's process. The scorer starts a second Python process that loads
your file and compiles your kernels, and receives from it only numbers: the pulses your kernel
plays and when. The scorer checks those numbers against the challenge's limits and, where there is
a result to measure, plays them itself, on a qubit only it knows where the challenge has a hidden
one. Where a qubit, a state or a case is hidden, each run of the scorer draws new ones and prints
a ``SEED`` to repeat the run; a repeated run shows its rungs but is not a score. The second
process is a separate process, not a sandbox: it keeps honest work honest, and the rest is the
honour rule.

Whatever your file prints is shown with a ``|`` in front of each line, and so is any text of
yours in a scorer's message (an error your file raised, the compiler's message): at most 200
lines of what it prints and 300 characters of a message, in printable ASCII, 72 characters to a
line. A scorer puts no other text of yours into its lines. So a line without that mark, among
those the scorer itself writes, is the scorer's; see below for what your file can write there
on its own.

What a scorer reads
-------------------

Besides its statement's own limits, a scorer reads only so much from the second process, and
says so when a kernel has more.

- Find the pi pulse, Schedule golf, Fix the kernel and Reset one qubit read one report of at
  most 4,000,000 bytes, with at most 512 pulses, 512 readouts and 512 waits, none longer than
  1024 ns and none that starts or lasts beyond 100 microseconds. The second process has 120 s
  to compile and report.
- Mystery qubit reads, for each run, a kernel of at most 64 drive pulses within 2 ms, whose
  drive pulses last at most 10 microseconds in all. Benchmark it yourself reads which calibrated
  pulses a sequence has, at most 6,000 pulses for one call of ``error_per_gate``. Each has 15
  minutes for a whole run of the scorer.
- A sample is at most 1,000,000 in magnitude in its real and in its imaginary part, and a
  frequency or phase at most 1e12.
- A seed is a whole number of at most 18 digits.

What a scorer cannot check
--------------------------

These are on your honour:

- The second process is yours. A file that changes what that process reports, instead of
  writing a kernel that plays it, is scored on the report. A report that breaks a rule fails
  like any other, but a scorer cannot tell a valid report from a kernel's own.
- The reference solutions pass, copied or imported.
- A file can stop or end the scorer, which then prints no verdict, and it can change the
  scorer's files for a later run. A scorer loads nothing after it has started your file, so
  what your file writes is not run by the scorer in that run.
- A file can write lines of its own wherever the scorer's output goes: to your terminal under
  the terminal's own name, or into the file or pipe the scorer writes to. Your file's process
  has no terminal of its own, and when the scorer has its report, or the time is up, it ends
  every process of that process's session; a process that has started a session of its own is
  not found.

The reference solutions are there to read after you have tried.
