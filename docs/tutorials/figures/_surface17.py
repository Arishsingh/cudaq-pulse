# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How every figure shows the qudits of ``transmon_surface17`` by group.

A qudit's group is the first letter of its label: ``D`` (data), ``X`` or ``Z``
(the auxiliaries of the X and Z stabilizers). Each group has one color per
theme and one marker, the same in every figure of the tutorials and of the
user guide, so no figure tells two groups apart by color alone. The three
colors stay apart under simulated protanopia and deuteranopia (CIE76
distance above 35 in both themes) and are at least 4.5:1 against the page.

``lattice`` places the qudits for a drawing from the couplings alone: the
data qudits on a square grid in label order, and each auxiliary in the
middle of the data qudits it couples to.
"""

from __future__ import annotations

import json
import math
from importlib.resources import files

from _style import Palette

GROUPS = ("D", "X", "Z")
NAMES = {"D": "data", "X": "X auxiliary", "Z": "Z auxiliary"}
PLURALS = {"D": "data qudits", "X": "X auxiliaries", "Z": "Z auxiliaries"}
MARKERS = {"D": "o", "X": "s", "Z": "D"}
_COLORS = {
    "light": {"D": "#4f5b66", "X": "#b85c00", "Z": "#6b46c1"},
    "dark": {"D": "#c3c9cf", "X": "#ffab40", "Z": "#b9a3f5"},
}


def group(label: str) -> str:
    """The group of the qudit named *label*: ``"D"``, ``"X"`` or ``"Z"``."""
    return label[0]


def color(p: Palette, label: str) -> str:
    """The color of *label*'s group in the theme of *p*."""
    return _COLORS[p.name][group(label)]


def labels(system) -> dict[int, str]:
    """Each qudit's label, by index."""
    return {q.index: q.label for q in system.physical_qudits.values()}


def lattice(system) -> dict[int, tuple[float, float]]:
    """Where to draw each qudit, by index, in units of the data-qudit pitch.

    Data qudits fill a square grid row by row, in label order. An auxiliary
    sits at the middle of its data qudits; one with two data qudits, on the
    edge of the grid, is moved half a pitch outwards.
    """
    names = labels(system)
    data = sorted((i for i, name in names.items() if group(name) == "D"), key=lambda i: names[i])
    side = math.isqrt(len(data))
    where = {i: (float(k % side), float(k // side)) for k, i in enumerate(data)}
    center = (side - 1) / 2
    partners: dict[int, list[int]] = {}
    for coupling in system.couplings:
        a, d = coupling.qudits
        partners.setdefault(a, []).append(d)
    for auxiliary, coupled in partners.items():
        x = sum(where[d][0] for d in coupled) / len(coupled)
        y = sum(where[d][1] for d in coupled) / len(coupled)
        if len(coupled) == 2:
            (x0, y0), (x1, y1) = (where[d] for d in coupled)
            # Across the edge the two data qudits lie on, away from the middle.
            if x0 == x1:
                x += 0.5 if x > center else -0.5
            else:
                y += 0.5 if y > center else -0.5
        where[auxiliary] = (x, y)
    return where


def feedline_record(system) -> dict:
    """What the system's shipped calibration file says about the readout
    feedlines (``readout.feedlines``): the groups, and where they come from."""
    path = files("cudaq_pulse.quantum_systems") / "calibrations" / f"{system.calibration.name}.json"
    record: dict = json.loads(path.read_text())["readout"]["feedlines"]
    return record


def feedlines(system) -> dict[str, list[int]]:
    """The qudits on each readout feedline, by the feedline's name."""
    index = {name: i for i, name in labels(system).items()}
    groups = feedline_record(system)["groups"]
    return {name: [index[label] for label in entry["qudits"]] for name, entry in groups.items()}
