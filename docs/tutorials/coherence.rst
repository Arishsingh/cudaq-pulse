..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Coherence: T1, Ramsey and Echo
==============================

A qubit forgets in two ways: it relaxes from the excited state, and its
phase drifts away from the drive's. This tutorial measures both on the
simulated qubit from :doc:`first_pulse_program`.

What you'll learn:

- how to use a quantum system's calibrated pulses;
- how to sweep a wait whose length is a kernel argument, and fit T1;
- how a Ramsey sequence records a detuning;
- how a spin echo cancels it.

What the model knows
--------------------

Each qudit carries its measured coherence times, along with the pulses its
gates are calibrated for:

.. code-block:: python

   import numpy as np

   import cudaq_pulse as pulse

   system = pulse.quantum_system("transmon_surface17")
   qubit = system.qudits[0]
   d1 = system.physical_qudits[0]

   print(f"qubit:   {d1.label}")
   print(f"T1:      {d1.t1_us} µs")
   print(f"T2*:     {d1.t2_star_us} µs")
   print(f"T2 echo: {d1.t2_echo_us} µs")

.. code-block:: text
   :class: output

   qubit:   D1
   T1:      29.1 µs
   T2*:     35.0 µs
   T2 echo: 46.2 µs

Its π pulse is a 40 ns DRAG pulse: the Gaussian shape, with a small
correction in the other quadrature that keeps a transmon out of its third
level:

.. code-block:: python

   flip = qubit.get_calibration(pulse.gates.X)
   print(f"{flip.duration_ns} ns DRAG, amplitude {flip.amplitude:.4f}")

.. code-block:: text
   :class: output

   40.0 ns DRAG, amplitude 0.1313

Simulation includes the relaxation and dephasing these times describe.

Measure T1
----------

Excite the qubit, wait, then read it out. The longer the wait, the more of
the excitation has decayed:

.. code-block:: python

   @pulse.kernel
   def t1(qudit, wait_ns: float):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       flip = drive_line.get_calibration(pulse.gates.X)
       measurement = readout_line.get_calibration(pulse.gates.READOUT)

       drive(drive_line, flip)
       wait(drive_line, wait_ns * pulse.ns)
       sync(drive_line, readout_line)
       return readout(readout_line, measurement)

.. themed-figure:: figures/t1_sequence.svg
   :alt: Timeline of the t1 kernel on D1's drive and readout lines: a pi pulse, a 10 microsecond wait, cut short here, and the readout.
   :width: 100%

   ``t1`` with a 10 µs wait, cut short.

What to notice:

- the π pulse and the readout come from the calibration, so the kernel
  holds no pulse parameters of its own;
- the wait comes from a ``float`` argument, so a sweep over it compiles
  once.

Sweep the wait from 0 to 60 µs:

.. code-block:: python

   waits = np.linspace(0, 60_000, 7)
   populations = []

   print("wait (µs)  excited")
   for wait_ns in waits:
       excited = pulse.simulate(t1, qubit, wait_ns).level_population(1)
       populations.append(excited)
       print(f"{wait_ns / 1000:9.0f}  {excited:.3f}")
   print("compilations:", t1.cache_info().compiled)

.. code-block:: text
   :class: output

   wait (µs)  excited
           0  0.968
          10  0.686
          20  0.491
          30  0.355
          40  0.259
          50  0.191
          60  0.143
   compilations: 1

The population falls toward the qubit's thermal population of 2.1%, not
toward zero, because the model's qubit sits in a slightly warm
environment.

Fit T1
------

Subtract the thermal population, and the decay is a straight line on a log
scale, whose slope gives T1:

.. code-block:: python

   excess = np.array(populations) - d1.thermal_population
   slope, intercept = np.polyfit(waits, np.log(excess), 1)
   fitted_t1_us = -1 / slope / 1000
   print(f"fitted T1: {fitted_t1_us:.1f} µs")

.. code-block:: text
   :class: output

   fitted T1: 29.3 µs

The fit gives 29.3 µs against the model's 29.1 µs. The small difference
comes from the readout pulse: the qubit keeps decaying while it is
measured.

.. themed-figure:: figures/t1_decay.svg
   :alt: Excited-state population of D1 after waits from 0 to 60 microseconds, falling from 0.97 toward the thermal population of 0.021. A straight-line fit to the log of the excess gives T1 = 29.3 microseconds; the exponential the model's 29.1 microseconds predicts runs through the points.
   :width: 100%

   Excited-state population after each wait, with the exponential the
   model's T1 predicts and the T1 the log fit gives.

Ramsey fringes
--------------

A Ramsey sequence puts the qubit on the equator with a π/2 pulse, lets it
precess, and turns it back with a second π/2 pulse.

If the drive is detuned from the qubit, the phase that builds up during the
wait shows up as fringes. ``shift_frequency`` detunes the drive's tone by
1 MHz:

.. code-block:: python

   @pulse.kernel
   def ramsey(qudit, tau_ns: float):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       half_flip = drive_line.get_calibration(pulse.gates.X90)
       measurement = readout_line.get_calibration(pulse.gates.READOUT)

       shift_frequency(drive_line.tone, 1e6)
       drive(drive_line, half_flip)
       wait(drive_line, tau_ns * pulse.ns)
       drive(drive_line, half_flip)
       sync(drive_line, readout_line)
       return readout(readout_line, measurement)

The detuning is a frame change on the drive's tone, so it costs no pulse.

.. themed-figure:: figures/ramsey_sequence.svg
   :alt: Timeline of the ramsey kernel on D1's drive and readout lines: a +1 MHz shift of the drive's tone, which takes no time, a pi/2 pulse, a 1 microsecond wait, a second pi/2 pulse and the readout.
   :width: 100%

   ``ramsey`` with a 1 µs wait, cut short.

What to notice: the +1 MHz shift is a point on the drive line, before the
first π/2 pulse.

Sweep the wait and find the first maximum after the start:

.. code-block:: python

   taus = np.linspace(0, 4_000, 41)
   fringes = []
   for tau_ns in taus:
       result = pulse.simulate(ramsey, qubit, tau_ns)
       fringes.append(result.level_population(1))

   # The first revival lies between the 5th and the 15th wait.
   first_peak = 5 + int(np.argmax(fringes[5:15]))
   print(f"first revival at {taus[first_peak]:.0f} ns")

.. code-block:: text
   :class: output

   first revival at 1000 ns

The population returns to its maximum after 1 µs, the period of the 1 MHz
detuning.

.. themed-figure:: figures/ramsey_fringes.svg
   :alt: Excited-state population after the Ramsey sequence with a 1 MHz detuned drive, at the page's 41 waits from 0 to 4 microseconds. It oscillates between 0.05 and 0.95, with the first revival at 1000 ns, once per microsecond.
   :width: 100%

   The page's Ramsey sweep, with its first revival marked.

What to notice: over a few microseconds the fringes barely shrink, since
T2* is 35 µs. Their slow loss of contrast over a longer sweep is what a
Ramsey measurement of T2* fits.

Cancel the detuning with an echo
--------------------------------

A π pulse halfway through the wait reverses the phase the qubit has
gathered, so in the second half it unwinds again. Any detuning that stays
the same during the sequence cancels.

The kernel finds the same lines and pulses as ``ramsey``, plus a π pulse:

.. code-block:: python
   :class: continues

   @pulse.kernel
   def echo(qudit, tau_ns: float):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       half_flip = drive_line.get_calibration(pulse.gates.X90)
       flip = drive_line.get_calibration(pulse.gates.X)
       measurement = readout_line.get_calibration(pulse.gates.READOUT)

The only change in the sequence is that π pulse, which splits the wait in
two:

.. code-block:: python

       shift_frequency(drive_line.tone, 1e6)
       drive(drive_line, half_flip)
       wait(drive_line, tau_ns / 2 * pulse.ns)
       drive(drive_line, flip)
       wait(drive_line, tau_ns / 2 * pulse.ns)
       drive(drive_line, half_flip)
       sync(drive_line, readout_line)
       return readout(readout_line, measurement)

.. themed-figure:: figures/echo_sequence.svg
   :alt: Timeline of the echo kernel on D1's drive and readout lines: the same +1 MHz shift and pi/2 pulse, half the 1 microsecond wait, a pi pulse, the other half, a second pi/2 pulse and the readout.
   :width: 100%

   ``echo`` with a 1 µs wait, cut short in each half.

What to notice: the π pulse sits exactly halfway, so each half of the wait
is the same length.

Run it at every fifth wait:

.. code-block:: python

   print("wait (ns)  excited")
   for tau_ns in taus[::5]:
       result = pulse.simulate(echo, qubit, tau_ns)
       excited = result.level_population(1)
       print(f"{tau_ns:9.0f}  {excited:.3f}")

.. code-block:: text
   :class: output

   wait (ns)  excited
           0  0.022
         500  0.031
        1000  0.035
        1500  0.047
        2000  0.048
        2500  0.055
        3000  0.061
        3500  0.074
        4000  0.073

The three pulses add up to a full turn, so the qubit ends close to its
ground state for every wait, with no fringes.

.. themed-figure:: figures/ramsey_echo.svg
   :alt: Excited-state population against the total wait from 0 to 4 microseconds. The Ramsey curve oscillates between 0.03 and 0.95 once per microsecond, from the 1 MHz detuning. The echo stays near the ground state, climbing slowly from 0.02 to 0.07.
   :width: 100%

   With a 1 MHz detuning, the Ramsey sequence oscillates once per
   microsecond, while the echo stays near the ground state.

What to notice: what's left of the echo is a slow rise from decay and
dephasing during the wait, which an echo measurement of T2 fits.

Where to go next
----------------

- :example:`03_t1_t2_dissipator.py` compares a T1 sweep with the decay the
  model predicts, and compiles Ramsey and echo kernels.
- :example:`04_hahn_echo.py` builds the Hahn echo and a CPMG train.
- :example:`13_dynamical_decoupling.py` spaces many refocusing pulses as
  XY4, CPMG and Uhrig sequences.
- :doc:`adaptive_calibration` picks each wait from the last measurement to
  track a drifting qubit.

For the details, :doc:`../user_guide/kernels` describes waits whose length
is a parameter, and :doc:`../user_guide/quantum_systems` how coherence
times and calibrations are stored.
