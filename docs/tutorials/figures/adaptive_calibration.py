# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Adaptive Calibration in One Program" tutorial.

Usage: python docs/tutorials/figures/adaptive_calibration.py

Needs matplotlib (the ``[viz]`` extra). Writes ``adaptive_shots.svg`` and
``adaptive_t1_estimates.svg``, each with its ``-dark`` twin, next to this
file: three shots of the tracking kernel the page compiles, and the 400 runs
of ``benchmarks/classical/adaptive_t1_estimates.json``, which
``benchmarks/classical/adaptive_t1_estimates.py`` writes.
"""

import json

import numpy as np
from _style import HERE, page_namespace, plot_axes, save_themed
from _timeline import Timeline, compress, draw


def adaptive_shots(page) -> None:
    shots, first_wait_ns = 3, 50_000.0
    scheduled = page["compiled"].lower(page["q"], shots, 3.0, 450_000.0, first_wait_ns)
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


def main() -> None:
    adaptive_shots(page_namespace("adaptive_calibration.rst"))
    adaptive_estimates()


if __name__ == "__main__":
    main()
