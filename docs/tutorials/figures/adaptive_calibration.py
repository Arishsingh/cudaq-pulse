# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Adaptive Calibration in One Program" tutorial.

Usage: python docs/tutorials/figures/adaptive_calibration.py

Needs matplotlib (the ``[viz]`` extra). Writes, each with its ``-dark`` twin,
next to this file:

- ``t1_belief``, the estimator's distribution moving from the prior after a
  readout of either value, from the calibration module's own update,
  animated: a scene of ``adaptive_calibration.json``, whose still
  ``tools/render-pulse-visual.js`` draws;
- ``adaptive_shots.svg``, three shots of the tracking kernel the page compiles;
- ``adaptive_branches.svg``, the estimate along every branch of the page's
  three-shot simulation;
- ``adaptive_t1_estimates.svg``, the 400 runs of
  ``benchmarks/classical/adaptive_t1_estimates.json``, which
  ``benchmarks/classical/adaptive_t1_estimates.py`` writes.
"""

import json
import math

import cudaq_pulse as pulse
import numpy as np
from _style import HERE, end_label, page_namespace, plot_axes, save_themed
from _timeline import Timeline, compress, draw
from _visual import write


def adaptive_shots(page) -> None:
    shots, first_wait_ns = 3, 50_000.0
    scheduled = page["compiled"].lower(page["qubit"], shots, 3.0, 450_000.0, first_wait_ns)
    timeline, cuts = compress(Timeline.of(scheduled), 400.0)
    words = {}
    for pl in timeline.pulses:
        key = (pl.lane, round(pl.start_ns, 3))
        words[key] = ("π" if not pl.conditional else "reset?") if pl.kind == "drive" else "read"

    def figure(p):
        return draw(
            timeline,
            p,
            pulse_labels=words,
            cuts=cuts,
            lane_in=0.62,
            call_labels={"update_t1": "update_t1"},
        )

    call = timeline.calls[0]
    variable = [w for w in timeline.waits if w.min_ns is not None]
    alt = (
        f"Timeline of the first {shots} shots of track_t1 on D1's drive and readout lines and "
        "the controller link, with the long first wait cut short. Each shot plays a pi pulse, "
        f"waits, and reads out; the outcome goes to update_t1, which answers within "
        f"{call.max_latency_ns:g} ns plus the link's {call.sent_ns - call.submit_ns:g} and "
        f"{call.link['feedback_ns']:g} ns, and a reset pulse plays only if the qubit read 1. "
        f"The first wait is the {first_wait_ns / 1000:g} microseconds the call passes; each later "
        "wait is set by the answer, between "
        f"{variable[0].min_ns:g} ns and {variable[0].max_ns / 1000:g} microseconds."
    )
    save_themed(figure, "adaptive_shots", title="Three adaptive shots in one program", desc=alt)


def adaptive_estimates() -> None:
    path = HERE.parents[2] / "benchmarks" / "classical" / "adaptive_t1_estimates.json"
    data = json.loads(path.read_text())
    estimates = np.array(data["estimates_us"])
    true_t1, median = data["quantum_system"]["t1_us"], data["median_t1_us"]
    lo, hi = data["spread_16_84_us"]
    paper = data["paper"]

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.6)
        bins = np.arange(60, 460, 10)
        ax.hist(estimates, bins=bins, color=p.drive, alpha=0.55, lw=0)
        top = np.histogram(estimates, bins=bins)[0].max()
        ax.axvspan(lo, hi, color=p.drive, alpha=0.08, lw=0)
        ax.plot([median, median], [0, top * 1.08], color=p.drive, lw=1.4)
        ax.text(
            median,
            top * 1.1,
            f"median {median:.0f} µs",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=p.drive,
        )
        ax.plot([true_t1, true_t1], [0, top * 0.82], color=p.text, lw=1.0, ls=(0, (1, 2)))
        ax.text(
            true_t1 + 4,
            top * 0.8,
            f"model T1 {true_t1:g} µs",
            ha="left",
            va="top",
            fontsize=8.5,
            color=p.text,
        )
        y = top * 1.32
        ax.plot(paper["interval_68pct_us"], [y, y], color=p.q, lw=1.4)
        ax.plot([paper["t1_us"]], [y], "o", ms=4, color=p.q, mec="none")
        ax.text(
            paper["interval_68pct_us"][1] + 6,
            y,
            f"published: {paper['t1_us']:g} µs, 68%",
            ha="left",
            va="center",
            fontsize=8.5,
            color=p.q,
        )
        ax.set_xlabel("T1 estimate after 50 shots (µs)")
        ax.set_ylabel("runs")
        ax.set_ylim(0, top * 1.45)
        ax.set_xlim(bins[0], bins[-1])
        return fig

    alt = (
        f"Histogram of the final T1 estimates of {len(estimates)} simulated runs of 50 adaptive "
        f"shots each, on a model qubit with a T1 of {true_t1:g} microseconds. Their median is "
        f"{median:.0f} microseconds and the middle 68% lie between {lo:.0f} and {hi:.0f}. Above "
        f"them, the published estimate of {paper['t1_us']:g} microseconds with its 68% interval "
        f"of {paper['interval_68pct_us'][0]:g} to {paper['interval_68pct_us'][1]:g}."
    )
    save_themed(
        figure, "adaptive_t1_estimates", title="T1 estimates of 400 adaptive runs", desc=alt
    )


def _t1_density(shape: float, rate: float, t1_us: np.ndarray) -> np.ndarray:
    """The density over T1 (µs) of a gamma distribution over 1/T1 (per ns)."""
    t1_ns = t1_us * 1000
    decay = 1 / t1_ns
    log = shape * math.log(rate) - math.lgamma(shape) + (shape - 1) * np.log(decay) - rate * decay
    return np.exp(log) / t1_ns**2 * 1000


def t1_belief(page) -> dict:
    """The estimator's distribution over T1, moving from the prior to where one
    readout after a 50 µs wait takes it, for each outcome, with the next wait
    ``update_t1`` would answer."""
    cal = page["cal"]
    shape, rate, wait_ns = 3.0, 450_000.0, 50_000.0
    steps = [("before any shot", None, shape, rate)]
    for bit in (1, 0):
        new_shape, new_rate = cal.adaptive_t1_update(
            shape, rate, bit, wait_ns, error_1to0=0.11, error_0to1=0.14
        )
        steps.append((f"after reading {bit} at {wait_ns / 1000:g} µs", bit, new_shape, new_rate))
    t1_us = np.linspace(5, 600, 120)
    peak = max(_t1_density(a, b, t1_us).max() for _, _, a, b in steps)
    snapshots = []
    for title, bit, a, b in steps:
        estimate = b / a / 1000
        next_wait = min(max(cal.adaptive_t1_wait(a, b), 100.0), 200_000.0) / 1000
        if bit is None:
            say = f"Before any shot, the prior puts T1 at {estimate:.0f} µs."
        else:
            moved = "still excited, so the estimate rises" if bit else "decayed, so it falls"
            wait_us = wait_ns / 1000
            say = f"Reading {bit} after the {wait_us:g} µs wait: {moved} to {estimate:.0f} µs."
        snapshot = {
            "label": title[0].upper() + title[1:],
            "short": title,
            "say": f"{say} The next wait would be {next_wait:.1f} µs.",
            "spans": {},
            "curve": {
                "y": [round(v, 4) for v in _t1_density(a, b, t1_us) / peak],
                "at": round(estimate, 3),
                "text": f"{estimate:.0f} µs",
            },
            "value": f"T1 {estimate:.0f} µs, next wait {next_wait:.1f} µs",
        }
        if bit is not None:
            snapshot["from"] = 0
        snapshots.append(snapshot)
    return {
        "name": "t1_belief",
        "title": "One readout moves the estimate",
        "curve": {
            "x": [round(v, 3) for v in t1_us],
            "ticks": [0, 200, 400, 600],
            "label": "T1 (µs)",
            "name": "belief",
        },
        "snapshots": snapshots,
        "share_label": "estimate",
        "panels": ["curve", "cases"],
    }


def adaptive_branches(page) -> None:
    result = pulse.simulate(page["three_shots"], levels=2)
    first = 450_000.0 / 3.0 / 1000
    paths = []
    for branch in result.branches:
        estimates = [first] + [
            rate / shape / 1000 for shape, rate, _ in branch.answers("update_t1")
        ]
        paths.append((branch.probability, branch.levels, estimates))
    likeliest = max(paths, key=lambda path: path[0])

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.4)
        shots = np.arange(len(likeliest[2]))
        for probability, levels, estimates in paths:
            if levels == likeliest[1]:
                continue
            ax.plot(shots, estimates, color=p.muted, lw=0.6 + 3 * probability, alpha=0.6)
        ax.plot(shots, likeliest[2], "o-", color=p.drive, lw=1.8, ms=4)
        end_label(
            ax,
            shots[-1],
            likeliest[2][-1],
            f"likeliest branch: {likeliest[0]:.2f}",
            p.drive,
            dx=6,
        )
        ax.set_xticks(shots, ["first guess"] + [f"after shot {k}" for k in shots[1:]])
        ax.set_xlim(-0.2, shots[-1] + 1.1)
        ax.set_ylabel("T1 estimate (µs)")
        return fig

    lows = sorted(path[2][-1] for path in paths)
    alt = (
        f"The T1 estimate after each of three shots, along each of the {len(paths)} branches "
        f"of the simulation, from the first guess of {first:g} microseconds. After three shots "
        f"the branches end between {lows[0]:.0f} and {lows[-1]:.0f} microseconds. The likeliest "
        f"branch, with probability {likeliest[0]:.2f}, ends at {likeliest[2][-1]:.1f}."
    )
    save_themed(figure, "adaptive_branches", title="Every branch learns its own T1", desc=alt)


def main() -> None:
    page = page_namespace("adaptive_calibration.rst")
    adaptive_shots(page)
    adaptive_branches(page)
    adaptive_estimates()
    write("adaptive_calibration", "docs/tutorials/adaptive_calibration.rst", [t1_belief(page)])


if __name__ == "__main__":
    main()
