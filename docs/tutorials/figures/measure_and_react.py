# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Measure and React" tutorial.

Usage: python docs/tutorials/figures/measure_and_react.py

Needs matplotlib (the ``[viz]`` extra). Writes ``reset_timeline.svg`` and
``reset_branches.svg``, each with its ``-dark`` twin, next to this file, from
the active reset the page defines and simulates.
"""

import cudaq_pulse as pulse
import numpy as np
from _style import page_namespace, plot_axes, save_themed
from _timeline import Timeline, draw


def reset_timeline(page) -> None:
    timeline = Timeline.of(page["active_reset"].lower(page["qubit"]))
    drives = [pl for pl in timeline.pulses if pl.kind == "drive"]
    readouts = [pl for pl in timeline.pulses if pl.kind == "readout"]
    words = {(pl.lane, round(pl.start_ns, 3)): "π/2" for pl in drives[:1]}
    words |= {(pl.lane, round(pl.start_ns, 3)): ("π if read 1", "right") for pl in drives[1:]}
    words |= {
        (pl.lane, round(pl.start_ns, 3)): w
        for pl, w in zip(readouts, ("measure", "check"), strict=True)
    }
    flip = drives[1]
    alt = (
        "Timeline of the active reset on D1's drive and readout lines. A pi/2 pulse, a "
        f"{readouts[0].end_ns - readouts[0].start_ns:g} ns readout, then a pi pulse from "
        f"{flip.start_ns:g} to {flip.end_ns:g} ns that plays only if the readout read 1, and a "
        f"second readout from {readouts[1].start_ns:g} ns that starts at the same time either way."
    )
    save_themed(
        lambda p: draw(timeline, p, pulse_labels=words, syncs="none"),
        "reset_timeline",
        title="The schedule of the active reset",
        desc=alt,
    )


def reset_branches(page) -> None:
    result = pulse.simulate(page["active_reset"], page["qubit"], shots_count=1000, seed=7)
    first: dict[int, float] = {}
    both: dict[tuple[int, int], float] = {}
    for b in result.branches:
        first[b.levels[0]] = first.get(b.levels[0], 0.0) + b.probability
        both[b.levels] = both.get(b.levels, 0.0) + b.probability
    outcomes = sorted(first)
    leaves = sorted(both)

    def fmt(x: float) -> str:
        return f"{x:.3f}" if x >= 0.001 else f"{x:.0e}".replace("e-0", "e-")

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.5)
        ax.axis("off")
        y_leaf = {leaf: -k for k, leaf in enumerate(leaves)}
        y_first = {a: np.mean([y_leaf[lf] for lf in leaves if lf[0] == a]) for a in outcomes}
        mid = np.mean(list(y_first.values()))

        def edge(x0, y0, x1, y1, prob):
            xs = np.linspace(0, 1, 40)
            ys = y0 + (y1 - y0) * (3 * xs**2 - 2 * xs**3)
            ax.plot(
                x0 + (x1 - x0) * xs,
                ys,
                color=p.readout,
                lw=0.6 + 7 * prob,
                alpha=0.35 + 0.65 * min(1.0, prob * 2),
                solid_capstyle="butt",
            )

        for a in outcomes:
            edge(0.0, mid, 1.0, y_first[a], first[a])
            ax.plot([1.0], [y_first[a]], "o", ms=3.5, color=p.readout, mec="none")
            ax.text(
                1.0,
                y_first[a] + 0.22,
                f"read {a}: {fmt(first[a])}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=p.text if first[a] > 0.01 else p.muted,
            )
        for leaf in leaves:
            edge(1.0, y_first[leaf[0]], 2.0, y_leaf[leaf], both[leaf])
            ax.text(
                2.06,
                y_leaf[leaf],
                f"then {leaf[1]}: {fmt(both[leaf])}",
                ha="left",
                va="center",
                fontsize=8.5,
                color=p.text if both[leaf] > 0.01 else p.muted,
            )
        ax.text(-0.04, mid, "start", ha="right", va="center", fontsize=8.5, color=p.muted)
        for x, word in ((1.0, "first readout"), (2.0, "second readout")):
            ax.text(x, 0.75, word, ha="center", va="bottom", fontsize=8, color=p.muted)
        ax.set_xlim(-0.35, 2.75)
        ax.set_ylim(-len(leaves) + 0.4, 1.1)
        return fig

    alt = (
        "The branches of the active reset's simulation, by the outcome of each readout: "
        + "; ".join(f"first read {a} with probability {fmt(first[a])}" for a in outcomes)
        + ". Every likely branch reads 0 the second time; "
        + ", ".join(f"{leaf[0]} then {leaf[1]}: {fmt(both[leaf])}" for leaf in leaves)
        + "."
    )
    save_themed(figure, "reset_branches", title="Every outcome the simulator follows", desc=alt)


def main() -> None:
    page = page_namespace("measure_and_react.rst")
    reset_timeline(page)
    reset_branches(page)


if __name__ == "__main__":
    main()
