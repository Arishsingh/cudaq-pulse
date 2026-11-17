# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Your first pulse program" tutorial.

Usage: python docs/tutorials/figures/first_pulse_program.py

Needs matplotlib (the ``[viz]`` extra). Writes ``rabi_sweep.svg``,
``rabi_shots.svg`` and ``rabi_schedule.svg``, each with its ``-dark`` twin,
next to this file, from the kernel, the sweep and the shots the tutorial
runs.
"""

import cudaq_pulse as pulse
import numpy as np
from _style import end_label, page_namespace, plot_axes, save_themed
from _timeline import Timeline, draw


def fit_rabi(amplitudes: np.ndarray, p1: np.ndarray) -> tuple[float, np.ndarray]:
    """The π amplitude of ``B + A (1 - cos(π a / a_π)) / 2`` that fits *p1* best."""
    best = None
    for a_pi in np.arange(0.05, 0.4, 1e-4):
        basis = np.column_stack(
            [np.ones_like(amplitudes), (1 - np.cos(np.pi * amplitudes / a_pi)) / 2]
        )
        coef, residual, *_ = np.linalg.lstsq(basis, p1, rcond=None)
        if best is None or residual[0] < best[0]:
            best = (residual[0], a_pi, coef)
    _, a_pi, coef = best
    return float(a_pi), coef


def main() -> None:
    page = page_namespace("first_pulse_program.rst")
    rabi, q, amplitudes = page["rabi"], page["qubit"], page["amplitudes"]
    levels = np.array([pulse.simulate(rabi, q, a).level_populations(q) for a in amplitudes])
    a_pi, (b, a) = fit_rabi(amplitudes, levels[:, 1])
    fine = np.linspace(amplitudes[0], amplitudes[-1], 400)
    model = b + a * (1 - np.cos(np.pi * fine / a_pi)) / 2

    def sweep(p):
        fig, ax = plot_axes(p, height_in=2.7)
        ax.plot(fine, model, color=p.muted, lw=1.0, zorder=2)
        ax.plot(amplitudes, levels[:, 1], "o", ms=3.6, color=p.drive, mec="none", zorder=3)
        for x in (a_pi, 2 * a_pi):
            ax.axvline(x, color=p.rule, lw=0.9, ls=(0, (1, 2)), zorder=1)
        ax.annotate(
            f"π pulse at {a_pi:.3f}",
            (a_pi, 1.02),
            xytext=(5, 0),
            textcoords="offset points",
            color=p.text,
            fontsize=8.5,
            va="top",
        )
        ax.annotate(
            "full turn",
            (2 * a_pi, 1.02),
            xytext=(5, 0),
            textcoords="offset points",
            color=p.muted,
            fontsize=8.5,
            va="top",
        )
        end_label(ax, amplitudes[-1], levels[-1, 1], "simulated", p.drive, dy=7)
        end_label(ax, fine[-1], model[-1], "cosine fit", p.muted, dy=-7)
        ax.set_xlabel("drive amplitude")
        ax.set_ylabel("excited-state population")
        ax.set_ylim(0, 1.04)
        ax.set_xlim(0, amplitudes[-1])
        ax.set_yticks([0, 0.5, 1])
        return fig

    sweep_alt = (
        "Simulated excited-state population after the 32 ns Gaussian pulse, for drive "
        f"amplitudes from 0 to {amplitudes[-1]:g}, with a cosine fit. The population rises "
        f"from {levels[0, 1]:.2f} to {levels[:, 1].max():.2f}; the fit puts the pi pulse at "
        f"amplitude {a_pi:.3f}, and the population falls back near {2 * a_pi:.2f}, a full turn."
    )
    save_themed(sweep, "rabi_sweep", title="Rabi oscillation of qudit 0", desc=sweep_alt)

    shots(page, levels_at=pulse.simulate(rabi, q, 0.16).level_population(1))

    timeline = Timeline.of(rabi.lower(q, 0.16))
    words = {(pl.lane, round(pl.start_ns, 3)): pl.kind for pl in timeline.pulses}
    schedule_alt = (
        "Timeline of one call of the Rabi kernel on D1: a 32 ns Gaussian drive pulse, then, "
        "where sync puts it, a 300 ns square readout pulse on the readout line."
    )
    save_themed(
        lambda p: draw(timeline, p, pulse_labels=words),
        "rabi_schedule",
        title="The schedule of one Rabi call",
        desc=schedule_alt,
    )


def shots(page, levels_at: float) -> None:
    outcomes = np.asarray(page["outcomes"])[:, 0]
    n = np.arange(1, len(outcomes) + 1)
    running = np.cumsum(outcomes) / n
    # On a log axis, about 300 points spaced evenly in log(n) draw the same curve.
    keep = np.unique(np.geomspace(1, len(n), 300).astype(int)) - 1
    n, running = n[keep], running[keep]
    spread = 2 * np.sqrt(levels_at * (1 - levels_at) / n)

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.4)
        ax.fill_between(
            n,
            levels_at - spread,
            np.minimum(levels_at + spread, 1.0),
            color=p.muted,
            alpha=0.12,
            lw=0,
        )
        ax.axhline(levels_at, color=p.muted, lw=1.0, ls=(0, (1, 2)))
        ax.plot(n, running, color=p.drive, lw=1.4)
        end_label(
            ax,
            n[-1],
            running[-1],
            f"mean of shots {running[-1]:.3f}",
            p.drive,
            dy=-9,
            ha="right",
            dx=0,
        )
        end_label(
            ax, n[-1], levels_at, f"population {levels_at:.3f}", p.muted, dy=8, ha="right", dx=0
        )
        ax.set_xscale("log")
        ax.set_xlim(1, n[-1])
        ax.set_ylim(0.7, 1.02)
        ax.set_xlabel("shots")
        ax.set_ylabel("fraction that read 1")
        return fig

    alt = (
        f"The fraction of the first n of {len(outcomes)} sampled shots that read 1, against n on "
        f"a log scale. It wanders at first and settles at {running[-1]:.3f}, inside a band of two "
        f"standard errors around the excited-state population of {levels_at:.3f}, which narrows "
        "as the shots add up."
    )
    save_themed(figure, "rabi_shots", title="Shots converge on the population", desc=alt)


if __name__ == "__main__":
    main()
