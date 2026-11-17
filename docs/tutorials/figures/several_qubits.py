# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Several Qubits" tutorial.

Usage: python docs/tutorials/figures/several_qubits.py

Needs matplotlib (the ``[viz]`` extra). Writes ``alap_asap.svg`` and
``zz_ramsey.svg``, each with its ``-dark`` twin, next to this file: the
page's ``uneven`` kernel under both scheduling policies, and its Ramsey
kernel next to a neighbour in either state.
"""

import cudaq_pulse as pulse
import numpy as np
from _style import end_label, page_namespace, plot_axes, save_themed
from _timeline import Timeline, draw_stack


def alap_asap(page) -> None:
    a, b = page["data"][0], page["data"][1]
    timelines = [
        Timeline.of(pulse.compile(page["uneven"], schedule=policy).lower(a, b))
        for policy in ("alap", "asap")
    ]
    moved = [
        next(pl for pl in t.pulses if pl.kind == "drive" and t.lanes and pl.lane == lane)
        for t in timelines
        for lane in [next(ln.key for ln in t.lanes if ln.label == "D2 drive")]
    ]
    readout = next(pl for pl in timelines[0].pulses if pl.kind == "readout")
    titles = [
        'schedule="alap", the default: as late as possible',
        'schedule="asap": as early as possible',
    ]
    alt = (
        "Two timelines of the uneven kernel on D1 and D2. D1 plays three pi pulses and D2 one, "
        f"and both readouts start at {readout.start_ns:g} ns. As late as possible, D2's pulse "
        f"plays from {moved[0].start_ns:g} ns, right before the readouts; as early as possible, "
        f"from {moved[1].start_ns:g} ns, and D2 then idles until it is measured."
    )
    save_themed(
        lambda p: draw_stack(timelines, titles, p, lane_in=0.34, syncs="none"),
        "alap_asap",
        title="One kernel under two scheduling policies",
        desc=alt,
    )


def zz_ramsey(page) -> None:
    d1, x2, kernel = page["data"][0], page["x2"], page["ramsey_next_to"]
    taus = np.linspace(0, 20_000, 26)
    curves = [
        np.array([pulse.simulate(kernel, d1, x2, excite, t).level_populations(d1)[1] for t in taus])
        for excite in (0, 1)
    ]

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.6)
        for curve, color, word in zip(
            curves, (p.drive, p.q), ("X2 in its ground state", "X2 excited"), strict=True
        ):
            ax.plot(taus / 1e3, curve, "o-", color=color, ms=3, lw=1.3)
            end_label(ax, taus[-1] / 1e3, curve[-1], word, color)
        ax.set_xlabel("Ramsey wait (µs)")
        ax.set_ylabel("D1 excited-state population")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, taus[-1] / 1e3)
        return fig

    gap = np.abs(curves[0] - curves[1])
    alt = (
        "D1's excited-state population against the Ramsey wait, from 0 to 20 microseconds, with "
        "its neighbour X2 in its ground state and excited. The two curves start together and "
        f"separate as the wait grows, by {gap[-1]:.2f} at 20 microseconds."
    )
    save_themed(figure, "zz_ramsey", title="Residual ZZ seen by a Ramsey sequence", desc=alt)


def main() -> None:
    page = page_namespace("several_qubits.rst")
    alap_asap(page)
    zz_ramsey(page)


if __name__ == "__main__":
    main()
