..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Pulse Operations API
====================

Channel Access
--------------

.. autofunction:: cudaq_pulse.ops.get_drive_line

.. autofunction:: cudaq_pulse.ops.get_readout_line

Scheduling
----------

.. autofunction:: cudaq_pulse.ops.drive

.. autofunction:: cudaq_pulse.ops.readout

.. autofunction:: cudaq_pulse.ops.wait

.. autofunction:: cudaq_pulse.ops.sync

Phase and Frequency
-------------------

.. autofunction:: cudaq_pulse.ops.shift_phase

.. autofunction:: cudaq_pulse.ops.set_phase

.. autofunction:: cudaq_pulse.ops.shift_frequency

.. autofunction:: cudaq_pulse.ops.set_frequency

Waveform Constructors
---------------------

.. themed-figure:: figures/waveform_spectra.svg
   :alt: Four waveforms, each 20 ns long with amplitude 0.3, above the spectrum of their samples from -60 to 0 dB within 500 MHz of the tone. A shaded band marks where 99% of each spectrum's energy up to half the clock lies: square ±427 MHz, gaussian_square ±105 MHz, gaussian ±59 MHz, cosine ±70 MHz. Outside its band the gaussian's spectrum still peaks at -41 dB between 400 and 500 MHz, and the cosine's is below -60 dB from 332 MHz on.
   :width: 100%

   Four of the envelopes as ``sample_waveform`` samples them, each 40 clock
   cycles of a 2 GHz clock, and the spectrum of those samples relative to
   its peak. The shaded band holds 99% of the spectrum's energy up to half
   the clock. It measures the main lobe, not what lies outside it: the
   gaussian has the narrowest band, but, cut off at its ends, it keeps a
   skirt that the cosine does not.

.. autofunction:: cudaq_pulse.ops.gaussian

.. autofunction:: cudaq_pulse.ops.square

.. autofunction:: cudaq_pulse.ops.drag

.. autofunction:: cudaq_pulse.ops.cosine

.. autofunction:: cudaq_pulse.ops.tanh_ramp

.. autofunction:: cudaq_pulse.ops.gaussian_square

.. autofunction:: cudaq_pulse.ops.custom

.. autofunction:: cudaq_pulse.ops.custom_samples

Waveform Arithmetic
-------------------

.. autofunction:: cudaq_pulse.ops.wf_add

.. autofunction:: cudaq_pulse.ops.wf_sub

.. autofunction:: cudaq_pulse.ops.wf_mul

.. autofunction:: cudaq_pulse.ops.wf_scale

.. autofunction:: cudaq_pulse.ops.wf_neg
