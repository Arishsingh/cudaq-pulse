# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Your first pulse program" tutorial.

Usage: python docs/tutorials/figures/first_pulse_program.py

Needs matplotlib (the ``[viz]`` extra). Writes, each with its ``-dark`` twin,
next to this file, from the quantum system, the kernel, the sweep and the
shots the tutorial runs:

- ``qudit_levels.svg``, the three levels the simulator keeps for qudit 0;
- ``rabi_pulse.svg``, the kernel's drive pulse at amplitude 0.16;
- ``rabi_levels.svg``, the population of each level after it;
- ``rabi_sweep.svg``, ``rabi_shots.svg`` and ``rabi_schedule.svg``.
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


def qudit_levels(page) -> None:
    qudit = page["system"].physical_qudits[0]
    f01 = qudit.frequency_hz / 1e9
    f12 = f01 + qudit.anharmonicity_hz / 1e9

    def figure(p):
        fig, ax = plot_axes(p, height_in=1.9, width_in=4.2)
        ax.axis("off")
        energies = [0.0, f01, f01 + f12]
        for level, e in enumerate(energies):
            color = p.muted if level == 2 else p.text
            ax.plot([0, 1], [e, e], color=color, lw=1.8)
            ax.text(-0.08, e, f"level {level}", ha="right", va="center", fontsize=8.5, color=color)
        for k, (lo, hi, word, color) in enumerate(
            (
                (energies[0], energies[1], f"0-1: {f01:.3f} GHz", p.drive),
                (energies[1], energies[2], f"1-2: {f12:.3f} GHz", p.muted),
            )
        ):
            x = 0.35 + 0.3 * k
            ax.annotate(
                "", (x, hi), (x, lo), arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.1)
            )
            ax.text(x + 0.05, (lo + hi) / 2, word, va="center", fontsize=8, color=color)
        ax.set_xlim(-0.55, 1.45)
        ax.set_ylim(-0.6, energies[-1] + 0.6)
        return fig

    alt = (
        f"The three lowest levels of qudit 0, the levels the simulator keeps: the 0-1 "
        f"transition at {f01:.3f} GHz, the one the drive is tuned to, and the 1-2 transition "
        f"at {f12:.3f} GHz, {abs(f01 - f12) * 1000:.0f} MHz lower."
    )
    save_themed(figure, "qudit_levels", title="The three levels of a transmon", desc=alt)


def rabi_pulse(page) -> None:
    timeline = Timeline.of(page["rabi"].lower(page["qubit"], 0.16))
    drive = next(pl for pl in timeline.pulses if pl.kind == "drive")
    words = {(drive.lane, round(drive.start_ns, 3)): ("Gaussian, amplitude 0.16", "right")}

    def figure(p):
        return draw(
            timeline,
            p,
            pulse_labels=words,
            only={drive.lane},
            xlim=(-2, drive.end_ns * 2.4),
            lane_in=0.8,
        )

    alt = (
        f"The drive pulse of the rabi kernel at amplitude 0.16: a Gaussian "
        f"{drive.end_ns - drive.start_ns:g} ns long, climbing from zero to its peak in the middle "
        "and back."
    )
    save_themed(figure, "rabi_pulse", title="The kernel's drive pulse", desc=alt)


def rabi_levels(page) -> None:
    levels = pulse.simulate(page["rabi"], page["qubit"], 0.16).level_populations(page["qubit"])

    def figure(p):
        fig, ax = plot_axes(p, height_in=1.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=p.grid)
        y = -np.arange(len(levels))
        colors = [p.muted, p.drive, p.readout]
        ax.hlines(y, 1e-4, levels, color=p.grid, lw=2)
        ax.scatter(levels, y, color=colors[: len(levels)], s=26, zorder=3, lw=0)
        for x, yy in zip(levels, y, strict=True):
            end_label(ax, x, yy, f"{x:.3f}", p.text, dx=6, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 3)
        ax.set_yticks(y, [f"level {k}" for k in range(len(levels))])
        ax.set_xlabel("population after the pulse")
        return fig

    alt = (
        "The population of each level of qudit 0 after the 0.16 pulse: "
        + "; ".join(f"level {k}, {x:.3f}" for k, x in enumerate(levels))
        + "."
    )
    save_themed(figure, "rabi_levels", title="Where the population ends", desc=alt)


def main() -> None:
    page = page_namespace("first_pulse_program.rst")
    qudit_levels(page)
    rabi_pulse(page)
    rabi_levels(page)
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
