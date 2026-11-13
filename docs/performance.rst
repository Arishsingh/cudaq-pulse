..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Performance Notes
=================

Measured numbers that the documentation and announcements quote, with the
command that measures each one again. They come from one machine and one
build, not a guarantee; times in particular move with the machine and its
load.

QEC-Scale Compilation
---------------------

``benchmarks/simulation/qec_scaling.py`` builds a synthetic syndrome
extraction cycle of a rotated surface code of distance *d* on
:math:`2d^2 - 1` qudits, then traces, builds, verifies and schedules it (no
optimization passes). Each repeat compiles from scratch; the script prints
each stage's median over the repeats:

.. code-block:: bash

   python benchmarks/simulation/qec_scaling.py --distances 31 --repeats 5

.. list-table::
   :header-rows: 1
   :widths: 16 16 16 26 26

   * - Distance
     - Qudits
     - Operations
     - Compile time, median of 5
     - Peak memory of the process
   * - 31
     - 1,921
     - 20,401
     - 263 ms
     - about 53 MB

Of the 263 ms, tracing the kernel takes 224 ms, building the module 17 ms,
verifying it 4.2 ms and scheduling 18 ms. The peak memory is the maximum
resident set size of the whole Python process, imports included, as
``getrusage`` reports it.

Host: an x86-64 Linux workstation, idle apart from the benchmark, running it
in the ``quay.io/pypa/manylinux_2_28_x86_64`` image that the wheels are built
in: Python 3.12, NumPy 2.5, and cudaq-pulse built from source there (a
development build of 0.1.0, compiler and CPU simulation). In six runs of the
command above the median compile time was 256 to 265 ms, and the fastest
single compile 256 ms. The table gives the middle run.

Adaptive T1 Estimation
----------------------

``benchmarks/classical/adaptive_t1_estimates.py`` runs the adaptive T1
tracker of ``examples/21_calibration_protocol.py`` (after Berritta et al.,
arXiv:2506.09576) 400 times in simulation, on a model qubit with a T1 of
160 us, assignment errors of 0.11 and 0.14 and the paper's prior, for 50
adaptive shots each, and writes every run's estimate.
``benchmarks/classical/adaptive_t1_estimates.json`` is its output for seed 1:

.. code-block:: bash

   python benchmarks/classical/adaptive_t1_estimates.py --trajectories 400 --seed 1 \
       --output benchmarks/classical/adaptive_t1_estimates.json
   python benchmarks/classical/adaptive_t1_estimates.py \
       --check benchmarks/classical/adaptive_t1_estimates.json

The median estimate is 155 us (bootstrap 95% interval 151 to 160 us), and
68% of runs fall between 123 and 195 us. The paper reports 159 us with a
68% interval of 131 to 202 us on its processor. The file holds no timings,
so the same seed and number of runs write the same estimates; ``--check``
regenerates them and compares (one run takes about a minute on the host
above).
