# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Calling Classical Code During Quantum Runtime" tutorial.

Usage: python docs/tutorials/figures/classical_calls.py

Needs matplotlib (the ``[viz]`` extra). Writes ``decode_declared.svg``,
``call_timeline.svg``, ``call_arms.svg`` and ``call_latency.svg``, each with
its ``-dark`` twin, next to this file, from the function the page declares
and the call it compiles and simulates.
"""

import numpy as np
from _style import end_label, page_namespace, plot_axes, save_themed
from _timeline import Timeline, draw
from cudaq_pulse.runtime._calls import latency_in_time_probability


def call_timeline(page) -> None:
    timeline = Timeline.of(page["scheduled"])
    words = {}
    for pl in timeline.pulses:
        words[(pl.lane, round(pl.start_ns, 3))] = (
            ("π if needed", "right") if pl.kind == "drive" else "readout"
        )
    call = timeline.calls[0]
    alt = (
        "Timeline of one shot of correct_if_needed on D1's readout and drive lines and the GPU "
        f"link. The {call.source[1]:g} ns readout's outcome goes to decode; the kernel waits for "
        f"the answer with a {call.timeout_ns / 1000:g} microsecond timeout and the link adds "
        f"{call.link['feedback_ns']:g} ns of feedback, so the answer is due at "
        f"{call.due_ns:g} ns, where a pi pulse that plays only if the decision says so starts."
    )
    save_themed(
        lambda p: draw(timeline, p, pulse_labels=words, breakdown=True),
        "call_timeline",
        title="A call with a deadline in one shot",
        desc=alt,
    )


def call_latency(page) -> None:
    decode, timeout_ns, in_time = page["decode"], 4_000.0, page["in_time"]
    lo, mode, hi = decode.min_latency_ns, decode.nominal_latency_ns, decode.max_latency_ns
    x = np.linspace(lo - 300, hi + 300, 601)
    cdf = np.array([latency_in_time_probability(decode, t) for t in x])
    density = np.gradient(cdf, x)

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.3)
        us = x / 1000
        ax.fill_between(us, 0, density, where=x <= timeout_ns, color=p.classical, alpha=0.45, lw=0)
        ax.fill_between(us, 0, density, where=x >= timeout_ns, color=p.classical, alpha=0.12, lw=0)
        ax.plot(us, density, color=p.classical, lw=1.4)
        ax.axvline(timeout_ns / 1000, color=p.text, lw=1.0)
        peak = density.max()
        ax.annotate(
            f"timeout {timeout_ns / 1000:g} µs",
            (timeout_ns / 1000, peak * 1.02),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=p.text,
            va="bottom",
        )
        end_label(
            ax,
            mode / 1000 - 0.25,
            peak * 0.42,
            f"in time: {in_time:.1%}",
            p.classical,
            ha="center",
            dx=0,
        )
        end_label(ax, (timeout_ns + hi) / 2000, peak * 0.12, "late", p.muted, ha="center", dx=0)
        for value, word in ((lo, "min"), (mode, "nominal"), (hi, "max")):
            ax.annotate(
                word,
                (value / 1000, 0),
                xytext=(0, -24),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=p.muted,
            )
        ax.set_yticks([])
        ax.grid(False)
        ax.set_ylim(0, peak * 1.25)
        ax.set_xlim(us[0], us[-1])
        ax.set_xlabel("latency (µs)", labelpad=14)
        return fig

    alt = (
        f"The distribution of decode's latency, triangular from {lo / 1000:g} to "
        f"{hi / 1000:g} microseconds with its peak at the nominal {mode / 1000:g}, and the "
        f"{timeout_ns / 1000:g} microsecond timeout. The answers to its left, {in_time:.1%} of "
        "them in the simulation, arrive in time; the rest take the fallback."
    )
    save_themed(figure, "call_latency", title="How often the answer beats the timeout", desc=alt)


def decode_declared(page) -> None:
    decode = page["decode"]
    lo, mode, hi = decode.min_latency_ns, decode.nominal_latency_ns, decode.max_latency_ns

    def figure(p):
        fig, ax = plot_axes(p, height_in=1.25)
        ax.grid(False)
        ax.set_yticks([])
        ax.hlines(0, lo / 1000, hi / 1000, color=p.classical, lw=6, alpha=0.35)
        ax.plot([mode / 1000], [0], "o", color=p.classical, ms=7, mec="none")
        for value, word in ((lo, "never sooner"), (mode, "typical"), (hi, "never later")):
            end_label(
                ax,
                value / 1000,
                0,
                f"{word}\n{value / 1000:g} µs",
                p.classical if value == mode else p.muted,
                dx=0,
                dy=9,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xlim(0, (hi + 500) / 1000)
        ax.set_ylim(-0.5, 1.6)
        ax.set_xlabel("time from the request to the answer (µs)")
        return fig

    alt = (
        f"The latency decode declares, on a line from 0: never sooner than {lo / 1000:g} "
        f"microseconds, typically {mode / 1000:g} and never later than {hi / 1000:g}."
    )
    save_themed(figure, "decode_declared", title="What the declaration promises", desc=alt)


def call_arms(page) -> None:
    result = page["result"]
    read: dict[int, float] = {}
    arms: dict[tuple[int, bool], float] = {}
    for branch in result.branches:
        level, in_time = min(branch.levels[0], 1), bool(branch.calls[0].in_time)
        read[level] = read.get(level, 0.0) + branch.probability
        arms[(level, in_time)] = arms.get((level, in_time), 0.0) + branch.probability
    order = sorted(arms, key=lambda key: (key[0], not key[1]))

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.0)
        ax.axis("off")
        y_leaf = {key: -k for k, key in enumerate(order)}
        y_read = {a: np.mean([y_leaf[k] for k in order if k[0] == a]) for a in read}
        mid = float(np.mean(list(y_read.values())))

        def edge(x0, y0, x1, y1, prob, color):
            xs = np.linspace(0, 1, 40)
            ax.plot(
                x0 + (x1 - x0) * xs,
                y0 + (y1 - y0) * (3 * xs**2 - 2 * xs**3),
                color=color,
                lw=0.6 + 6 * prob,
                alpha=0.4 + 0.6 * min(1.0, prob * 2),
                solid_capstyle="butt",
            )

        for a, share in read.items():
            edge(0.0, mid, 1.0, y_read[a], share, p.readout)
            word = "read 0" if a == 0 else "read 1 or more"
            end_label(
                ax,
                1.0,
                y_read[a],
                f"{word}: {share:.3f}",
                p.text,
                dx=0,
                dy=7,
                ha="center",
                va="bottom",
            )
        for key in order:
            level, in_time = key
            color = p.classical if in_time else p.muted
            edge(1.0, y_read[level], 2.0, y_leaf[key], arms[key], color)
            word = "answer in time" if in_time else "late: fallback"
            end_label(ax, 2.0, y_leaf[key], f"{word}: {arms[key]:.3f}", color, dx=6)
        ax.text(-0.04, mid, "start", ha="right", va="center", fontsize=8.5, color=p.muted)
        for x, word in ((1.0, "readout"), (2.0, "decode's answer")):
            ax.text(x, 0.85, word, ha="center", va="bottom", fontsize=8, color=p.muted)
        ax.set_xlim(-0.3, 2.75)
        ax.set_ylim(-len(order) + 0.5, 1.15)
        return fig

    alt = (
        "The branches of the simulation, by the readout and then by whether decode's answer "
        "arrived in time: "
        + "; ".join(
            f"read {'0' if level == 0 else '1 or more'} and "
            f"{'in time' if in_time else 'late'}, {arms[(level, in_time)]:.3f}"
            for level, in_time in order
        )
        + ". Each readout outcome splits the same way."
    )
    save_themed(figure, "call_arms", title="The simulator follows late answers too", desc=alt)


def main() -> None:
    page = page_namespace("classical_calls.rst")
    decode_declared(page)
    call_timeline(page)
    call_arms(page)
    call_latency(page)


if __name__ == "__main__":
    main()
