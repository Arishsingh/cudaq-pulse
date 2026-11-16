# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gates a quantum system is calibrated for, which ``get_calibration`` takes.

In a kernel, a line gives the calibrated pulse of its qudit's gates, ready
to play::

    d = get_drive_line(q)
    drive(d, d.get_calibration(pulse.gates.X))      # a DragPulse
    drive(d, d.get_calibration(pulse.gates.T))      # a VirtualZ: a frame change
    r = get_readout_line(q)
    readout(r, r.get_calibration(pulse.gates.READOUT))  # a ReadoutPulse

Outside one, a qudit handle and a coupling give the same values:
``qs.qudits[i].get_calibration(pulse.gates.X)`` and
``qs.coupling(a, b).get_calibration(pulse.gates.CZ)`` (a ``FluxCZ``).

``X``, ``X90``, ``Y``, ``Y90``, ``Z``, ``S`` and ``T`` are the gates of a
drive line, ``READOUT`` of a readout line and ``CZ`` of a coupling. What a
quantum system does not publish is derived: an ``X`` without an amplitude
gets the one that makes its area a pi rotation, ``X90`` is ``X`` at half the
amplitude, and ``Y`` and ``Y90`` are ``X`` and ``X90`` at +90 degrees.
``Z``, ``S`` and ``T`` are virtual Z rotations by pi, pi / 2 and pi / 4
(``VirtualZ``): played, each shifts the phase of the tone's frame, which
takes no time and plays no pulse. An unpublished ``CZ`` or ``READOUT`` is a
``NotFoundError`` naming the coupling or qudit. ``transition=(1, 2)`` asks
for the calibration of another transition: its pulses must be published,
and its virtual Zs are derived as on (0, 1).

There are no inverse gates, as there is no -90 degree pulse: a
calibration's ``replace`` gives one, ``t.replace(phase_rad=-t.phase_rad)``
for the inverse of ``T``.
"""

from __future__ import annotations

import enum
from typing import Final

__all__ = ["CZ", "READOUT", "S", "T", "X", "X90", "Y", "Y90", "Z", "Gate"]


class Gate(enum.StrEnum):
    """A gate a quantum system is calibrated for; see the module docstring."""

    X = "x"
    X90 = "x90"
    Y = "y"
    Y90 = "y90"
    Z = "z"
    S = "s"
    T = "t"
    CZ = "cz"
    READOUT = "readout"


X: Final = Gate.X
X90: Final = Gate.X90
Y: Final = Gate.Y
Y90: Final = Gate.Y90
Z: Final = Gate.Z
S: Final = Gate.S
T: Final = Gate.T
CZ: Final = Gate.CZ
READOUT: Final = Gate.READOUT
