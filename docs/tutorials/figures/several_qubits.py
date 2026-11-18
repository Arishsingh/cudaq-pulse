# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Several Qubits" tutorial.

Usage: python docs/tutorials/figures/several_qubits.py

Needs matplotlib (the ``[viz]`` extra). Writes, each with its ``-dark`` twin,
next to this file:

- ``register_parallel.svg``, the page's ``flip_all`` on three qubits;
- ``alap_asap.svg``, its ``uneven`` kernel under both scheduling policies;
- ``zz_shift.svg``, D1's frequency with X2 in either state;
- ``ramsey_neighbour.svg``, the schedule of its Ramsey kernel next to an
  excited neighbour;
- ``zz_ramsey.svg``, that kernel next to a neighbour in either state.
"""

import cudaq_pulse as pulse
import numpy as np
from _style import end_label, page_namespace, plot_axes, save_themed
from _timeline import Timeline, compress, draw, draw_stack


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


def register_parallel(page) -> None:
    timeline = Timeline.of(page["scheduled"])
    words = {(pl.lane, round(pl.start_ns, 3)): "π" for pl in timeline.pulses}
    lanes = [ln.label for ln in timeline.lanes]
    alt = (
        f"Timeline of flip_all on {len(lanes)} drive lines, "
        + ", ".join(lanes)
        + f". Each plays its own calibrated pi pulse, all from 0 to "
        f"{max(pl.end_ns for pl in timeline.pulses):g} ns."
    )
    save_themed(
        lambda p: draw(timeline, p, pulse_labels=words, lane_in=0.36),
        "register_parallel",
        title="A register driven in parallel",
        desc=alt,
    )


def zz_shift(page) -> None:
    zz_khz = page["idle_zz_hz"] / 1e3
    g_mhz = page["coupling"].strength_hz / 1e6

    def figure(p):
        fig, ax = plot_axes(p, height_in=1.3)
        ax.grid(False)
        ax.set_yticks([])
        for x, word, color in ((0.0, "X2 in ground", p.drive), (zz_khz, "X2 excited", p.q)):
            ax.plot([x, x], [0, 1], color=color, lw=2.2)
            end_label(ax, x, 1.05, word, color, dx=0, ha="center", va="bottom", fontsize=8)
        ax.annotate(
            "",
            (zz_khz, 0.45),
            (0, 0.45),
            arrowprops=dict(arrowstyle="-|>", color=p.text, lw=1.0),
        )
        end_label(
            ax,
            zz_khz / 2,
            0.5,
            f"{zz_khz:.2f} kHz",
            p.text,
            dx=0,
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.set_xlim(zz_khz * 1.6, -zz_khz * 0.6)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel("D1's 0-1 frequency, from its value with X2 in ground (kHz)")
        return fig

    alt = (
        f"D1's 0-1 frequency with its neighbour X2 in its ground state and excited. Through "
        f"their {g_mhz:.2f} MHz coupling, exciting X2 moves D1 by {zz_khz:.2f} kHz, the "
        "residual ZZ shift."
    )
    save_themed(figure, "zz_shift", title="A neighbour moves D1's frequency", desc=alt)


def ramsey_neighbour(page) -> None:
    d1, x2, kernel = page["data"][0], page["x2"], page["ramsey_next_to"]
    timeline, cuts = compress(Timeline.of(kernel.lower(d1, x2, 1, 10_000.0)), 300.0)
    words = {}
    drives = sorted(
        (pl for pl in timeline.pulses if pl.kind == "drive"), key=lambda pl: pl.start_ns
    )
    for pl in timeline.pulses:
        key = (pl.lane, round(pl.start_ns, 3))
        if pl.kind == "readout":
            words[key] = "readout"
        elif pl.lane != drives[-1].lane:
            words[key] = "π on X2"
        else:
            words[key] = "π/2"
    alt = (
        "Timeline of the Ramsey kernel on D1 next to X2, with X2 excited and a 10 microsecond "
        "wait cut short. X2's pi pulse plays first; then D1 plays a pi/2 pulse, waits, plays a "
        "second pi/2 pulse and is read out."
    )
    save_themed(
        lambda p: draw(timeline, p, pulse_labels=words, cuts=cuts, lane_in=0.4),
        "ramsey_neighbour",
        title="A Ramsey sequence next to an excited neighbour",
        desc=alt,
    )


def main() -> None:
    page = page_namespace("several_qubits.rst")
    register_parallel(page)
    alap_asap(page)
    zz_shift(page)
    ramsey_neighbour(page)
    zz_ramsey(page)


if __name__ == "__main__":
    main()
