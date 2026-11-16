# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Coherence: T1, Ramsey and Echo" tutorial.

Usage: python docs/tutorials/figures/coherence.py

Needs matplotlib (the ``[viz]`` extra). Writes ``t1_decay.svg`` and
``ramsey_echo.svg``, each with its ``-dark`` twin, next to this file, from
the kernels the page defines and the T1 fit it makes.
"""

import cudaq_pulse as pulse
import numpy as np
from _style import end_label, page_namespace, plot_axes, save_themed


def t1_decay(page) -> None:
    q, d1 = page["q"], page["d1"]
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
    q = page["q"]
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


def main() -> None:
    page = page_namespace("coherence.rst")
    t1_decay(page)
    ramsey_echo(page)


if __name__ == "__main__":
    main()
