# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Calling Classical Code During Quantum Runtime" tutorial.

Usage: python docs/tutorials/figures/classical_calls.py

Needs matplotlib (the ``[viz]`` extra). Writes ``call_timeline.svg`` and
``call_latency.svg``, each with its ``-dark`` twin, next to this file, from
the call the page compiles and simulates.
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


def main() -> None:
    page = page_namespace("classical_calls.rst")
    call_timeline(page)
    call_latency(page)


if __name__ == "__main__":
    main()
