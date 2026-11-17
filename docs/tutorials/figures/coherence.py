# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Coherence: T1, Ramsey and Echo" tutorial.

Usage: python docs/tutorials/figures/coherence.py

Needs matplotlib (the ``[viz]`` extra). Writes, each with its ``-dark`` twin,
next to this file, from the kernels the page defines and the T1 fit it
makes:

- ``t1_sequence.svg``, ``ramsey_sequence.svg`` and ``echo_sequence.svg``,
  the schedule of each kernel with its wait cut short;
- ``t1_decay.svg``, the T1 sweep and its fit;
- ``ramsey_fringes.svg``, the Ramsey sweep alone;
- ``ramsey_echo.svg``, the Ramsey sweep beside the echo.
"""

import cudaq_pulse as pulse
import numpy as np
from _style import end_label, page_namespace, plot_axes, save_themed
from _timeline import Timeline, compress, draw


def t1_decay(page) -> None:
    q, d1 = page["qubit"], page["d1"]
    waits = np.linspace(0, 60_000, 25)
    p1 = np.array([pulse.simulate(page["t1"], q, w).level_population(1) for w in waits])
    p_th = d1.thermal_population
    excess = p1 - p_th
    slope, intercept = np.polyfit(waits, np.log(excess), 1)
    fitted_us = -1 / slope / 1000
    model = p_th + (p1[0] - p_th) * np.exp(-waits / (d1.t1_us * 1e3))

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.7)
        ax.axhline(p_th, color=p.muted, lw=0.9, ls=(0, (1, 2)))
        ax.plot(waits / 1e3, model, color=p.muted, lw=1.0)
        ax.plot(waits / 1e3, p1, "o", color=p.drive, ms=3.6, mec="none")
        end_label(
            ax,
            waits[-1] / 1e3,
            p_th,
            f"thermal population {p_th:.3f}",
            p.muted,
            dy=7,
            ha="right",
            dx=0,
        )
        k = 6
        end_label(
            ax,
            waits[k] / 1e3,
            p1[k],
            f"simulated, fitted T1 = {fitted_us:.1f} µs",
            p.drive,
            dx=8,
            dy=6,
        )
        k = 13
        end_label(
            ax,
            waits[k] / 1e3,
            model[k],
            f"model T1 = {d1.t1_us} µs",
            p.muted,
            dx=-6,
            dy=-10,
            ha="right",
        )
        ax.set_xlabel("wait (µs)")
        ax.set_ylabel("excited-state population")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, waits[-1] / 1e3)
        return fig

    alt = (
        f"Excited-state population of D1 after waits from 0 to {waits[-1] / 1000:g} "
        f"microseconds, falling from {p1[0]:.2f} toward the thermal population of {p_th:.3f}. "
        f"A straight-line fit to the log of the excess gives T1 = {fitted_us:.1f} microseconds; "
        f"the exponential the model's {d1.t1_us} microseconds predicts runs through the points."
    )
    save_themed(figure, "t1_decay", title="T1 decay of D1", desc=alt)


def ramsey_echo(page) -> None:
    q = page["qubit"]
    taus = np.linspace(0, 4_000, 161)
    ramsey = np.array([pulse.simulate(page["ramsey"], q, t).level_population(1) for t in taus])
    echo_taus = taus[::4]
    echo = np.array([pulse.simulate(page["echo"], q, t).level_population(1) for t in echo_taus])

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.7)
        ax.plot(taus / 1e3, ramsey, color=p.drive, lw=1.5)
        ax.plot(echo_taus / 1e3, echo, "o-", color=p.q, ms=3, lw=1.2)
        peak = int(np.argmax(np.where((taus > 500) & (taus < 1500), ramsey, 0)))
        end_label(ax, taus[peak] / 1e3, ramsey[peak], "Ramsey, 1 MHz detuned", p.drive, dx=8)
        end_label(ax, echo_taus[-1] / 1e3, echo[-1], "echo", p.q, dy=9, ha="right", dx=0)
        ax.set_xlabel("total wait (µs)")
        ax.set_ylabel("excited-state population")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, taus[-1] / 1e3)
        return fig

    alt = (
        "Excited-state population against the total wait from 0 to 4 microseconds. The Ramsey "
        f"curve oscillates between {ramsey.min():.2f} and {ramsey.max():.2f} once per "
        f"microsecond, from the 1 MHz detuning. The echo stays near the ground state, climbing "
        f"slowly from {echo[0]:.2f} to {echo[-1]:.2f}."
    )
    save_themed(figure, "ramsey_echo", title="Ramsey fringes and their echo", desc=alt)


def sequence(page, kernel: str, argument: float, title: str, what: str) -> None:
    timeline, cuts = compress(Timeline.of(page[kernel].lower(page["qubit"], argument)), 250.0)
    drives = sorted(
        (pl for pl in timeline.pulses if pl.kind == "drive"), key=lambda pl: pl.start_ns
    )
    words = {}
    for pl in timeline.pulses:
        key = (pl.lane, round(pl.start_ns, 3))
        if pl.kind == "readout":
            words[key] = "readout"
        elif kernel == "t1" or (kernel == "echo" and pl is drives[1]):
            words[key] = "π"
        else:
            words[key] = "π/2"
    alt = f"Timeline of the {kernel} kernel on D1's drive and readout lines: {what}"
    save_themed(
        lambda p: draw(timeline, p, pulse_labels=words, cuts=cuts, syncs="none"),
        f"{kernel}_sequence",
        title=title,
        desc=alt,
    )


def ramsey_fringes(page) -> None:
    taus, fringes, first = np.asarray(page["taus"]), np.asarray(page["fringes"]), page["first_peak"]

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.3)
        ax.plot(taus / 1e3, fringes, "o-", color=p.drive, lw=1.3, ms=3)
        ax.axvline(taus[first] / 1e3, color=p.rule, lw=0.8, ls=(0, (1, 2)))
        end_label(
            ax,
            taus[first] / 1e3,
            fringes[first],
            f"first revival at {taus[first]:.0f} ns",
            p.text,
            dx=6,
            dy=4,
        )
        ax.set_xlabel("wait (µs)")
        ax.set_ylabel("excited-state population")
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0, taus[-1] / 1e3)
        return fig

    alt = (
        f"Excited-state population after the Ramsey sequence with a 1 MHz detuned drive, at the "
        f"page's {len(taus)} waits from 0 to {taus[-1] / 1000:g} microseconds. It oscillates "
        f"between {fringes.min():.2f} and {fringes.max():.2f}, with the first revival at "
        f"{taus[first]:.0f} ns, once per microsecond."
    )
    save_themed(figure, "ramsey_fringes", title="Ramsey fringes of a detuned drive", desc=alt)


def main() -> None:
    page = page_namespace("coherence.rst")
    sequence(
        page,
        "t1",
        10_000.0,
        "The T1 sequence",
        "a pi pulse, a 10 microsecond wait, cut short here, and the readout.",
    )
    t1_decay(page)
    sequence(
        page,
        "ramsey",
        1_000.0,
        "The Ramsey sequence",
        "a +1 MHz shift of the drive's tone, which takes no time, a pi/2 pulse, a 1 "
        "microsecond wait, a second pi/2 pulse and the readout.",
    )
    ramsey_fringes(page)
    sequence(
        page,
        "echo",
        1_000.0,
        "The echo sequence",
        "the same +1 MHz shift and pi/2 pulse, half the 1 microsecond wait, a pi pulse, the "
        "other half, a second pi/2 pulse and the readout.",
    )
    ramsey_echo(page)


if __name__ == "__main__":
    main()
