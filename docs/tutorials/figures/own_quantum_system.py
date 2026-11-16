# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Describe Your Own Quantum System" tutorial.

Usage: python docs/tutorials/figures/own_quantum_system.py

Needs matplotlib (the ``[viz]`` extra). Writes ``transmon_surface17.svg`` and
``pair_levels.svg``, each with its ``-dark`` twin, next to this file: the
shipped 17-qubit system drawn from its qudits and couplings, and the levels of
the pair the page describes.
"""

import cudaq_pulse as pulse
import numpy as np
from _style import page_namespace, plot_axes, save_themed


def surface17() -> None:
    qs = pulse.quantum_system("transmon_surface17")
    qudits = qs.physical_qudits
    label = {i: pq.label for i, pq in qudits.items()}
    neighbours: dict[int, list[int]] = {}
    for c in qs.couplings:
        a, b = c.qudits
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    # Data qubits D1..D9 row by row; an ancilla sits at the middle of the data
    # qubits it couples to, pushed out past the edge when it couples to two.
    pos = {}
    for i, name in label.items():
        if name.startswith("D"):
            k = int(name[1:]) - 1
            pos[i] = 1.5 * np.array([k % 3, -(k // 3)], float)
    centre = np.mean(list(pos.values()), axis=0)
    for i, name in label.items():
        if not name.startswith("D"):
            mid = np.mean([pos[j] for j in neighbours[i]], axis=0)
            if len(neighbours[i]) == 2:
                mid = mid + 0.75 * (mid - centre) / np.linalg.norm(mid - centre)
            pos[i] = mid

    def figure(p):
        fig, ax = plot_axes(p, height_in=3.6)
        ax.axis("off")
        ax.set_aspect("equal")
        for c in qs.couplings:
            (x0, y0), (x1, y1) = pos[c.qudits[0]], pos[c.qudits[1]]
            ax.plot([x0, x1], [y0, y1], color=p.rule, lw=1.2, zorder=1)
        for i, (x, y) in pos.items():
            name = label[i]
            kind = name[0]
            color = {"D": p.text, "X": p.q, "Z": p.drive}[kind]
            ax.plot(
                [x],
                [y],
                "o" if kind == "D" else "s",
                ms=26 if kind == "D" else 23,
                mfc=p.surface,
                mec=color,
                mew=1.5,
                zorder=2,
            )
            ax.text(x, y + 0.11, name, ha="center", va="center", fontsize=8, color=color, zorder=3)
            ax.text(
                x,
                y - 0.14,
                f"{qudits[i].frequency_hz / 1e9:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=p.muted,
                zorder=3,
            )
        keys = (("o", p.text, "data qubit"), ("s", p.q, "X ancilla"), ("s", p.drive, "Z ancilla"))
        for k, (marker, color, word) in enumerate(keys):
            ax.plot([4.6], [-0.4 - 0.55 * k], marker, ms=9, mfc="none", mec=color, mew=1.4)
            ax.text(4.85, -0.4 - 0.55 * k, word, va="center", fontsize=8, color=color)
        ax.text(
            4.45,
            -2.2,
            "inside each: its 0-1\nfrequency in GHz",
            va="top",
            fontsize=8,
            color=p.muted,
        )
        ax.set_xlim(-1.4, 6.6)
        ax.set_ylim(-4.3, 1.4)
        return fig

    ancillas = sorted(n for n in label.values() if not n.startswith("D"))
    data_ghz = [qudits[i].frequency_hz / 1e9 for i in qudits if label[i][0] == "D"]
    ancilla_ghz = [qudits[i].frequency_hz / 1e9 for i in qudits if label[i][0] != "D"]
    alt = (
        f"The {len(qudits)} qudits of transmon_surface17 and its {len(qs.couplings)} couplings: "
        "nine data qubits, D1 to D9, in a 3 by 3 grid, and the ancillas "
        f"{', '.join(ancillas)}, each coupled to the four data qubits around it or, at the edge, "
        "to two. Each qudit is labelled with its 0-1 frequency: the data qubits between "
        f"{min(data_ghz):.2f} and {max(data_ghz):.2f} GHz, the ancillas between "
        f"{min(ancilla_ghz):.2f} and {max(ancilla_ghz):.2f} GHz."
    )
    save_themed(figure, "transmon_surface17", title="The 17-qubit surface-code system", desc=alt)


def pair_levels(page) -> None:
    pair = page["pair"]
    qudits = [pair.physical_qudits[i] for i in sorted(pair.physical_qudits)]
    g = pair.couplings[0].strength_hz

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.8)
        ax.axis("off")
        for k, q in enumerate(qudits):
            f01, f12 = q.frequency_hz / 1e9, (q.frequency_hz + q.anharmonicity_hz) / 1e9
            levels = [0.0, f01, f01 + f12]
            x = 2.2 * k
            for n, e in enumerate(levels):
                ax.plot([x - 0.5, x + 0.5], [e, e], color=p.text, lw=1.6)
                ax.text(x - 0.62, e, f"|{n}⟩", ha="right", va="center", fontsize=8.5, color=p.muted)
            for (lo, hi), color, word in (
                ((0, 1), p.drive, f"{f01:.2f} GHz"),
                ((1, 2), p.q, f"{f12:.2f} GHz"),
            ):
                ax.annotate(
                    "",
                    (x, levels[hi]),
                    (x, levels[lo]),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.2, shrinkA=1, shrinkB=1),
                )
                ax.text(
                    x + 0.08,
                    (levels[lo] + levels[hi]) / 2,
                    word,
                    ha="left",
                    va="center",
                    fontsize=8,
                    color=color,
                )
            ax.text(x, -1.6, f"qudit {q.index}", ha="center", va="top", fontsize=8.5, color=p.text)
            ax.text(
                x,
                -2.4,
                f"anharmonicity {q.anharmonicity_hz / 1e6:.0f} MHz",
                ha="center",
                va="top",
                fontsize=8,
                color=p.muted,
            )
        mid = qudits[0].frequency_hz / 1e9
        ax.annotate(
            "",
            (2.2 - 0.55, mid),
            (0.55, mid),
            arrowprops=dict(arrowstyle="<->", color=p.classical, lw=1.0, ls=(0, (3, 2))),
        )
        ax.text(
            1.1,
            mid + 0.25,
            f"g / 2π = {g / 1e6:g} MHz",
            ha="center",
            va="bottom",
            fontsize=8,
            color=p.classical,
        )
        ax.set_xlim(-1.4, 3.6)
        ax.set_ylim(-3.0, 11.0)
        return fig

    parts = [
        f"qudit {q.index}: 0-1 at {q.frequency_hz / 1e9:.2f} GHz and 1-2 at "
        f"{(q.frequency_hz + q.anharmonicity_hz) / 1e9:.2f} GHz"
        for q in qudits
    ]
    alt = (
        "The three lowest levels of each transmon of my_pair, "
        + "; ".join(parts)
        + f". The 1-2 transition sits below the 0-1 by the anharmonicity. The two are coupled "
        f"with g / 2 pi = {g / 1e6:g} MHz."
    )
    save_themed(figure, "pair_levels", title="The levels of the pair", desc=alt)


def main() -> None:
    surface17()
    pair_levels(page_namespace("own_quantum_system.rst"))


if __name__ == "__main__":
    main()
