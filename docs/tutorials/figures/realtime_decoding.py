# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "Real-Time Decoding with a Deadline" tutorial.

Usage: python docs/tutorials/figures/realtime_decoding.py

Needs matplotlib (the ``[viz]`` extra). Writes ``qec_round_timeline.svg`` and
``qec_deadlines.svg``, each with its ``-dark`` twin, next to this file: the
round the page compiles, with its 1 µs deadline, and the page's simulation of
both arms over a range of deadlines. Both use the decoder as first declared,
before the page gives it a budget that fits.
"""

import numpy as np
from _style import end_label, page_namespace, plot_axes, save_themed
from _timeline import Timeline, draw_zoomed

# The page's second declaration of decode, which the figures leave out.
_FAST = "max_latency_ns=500"


def round_timeline(page) -> None:
    data, ancillas = page["data"], page["ancillas"]
    names = {q.index: f"data {i}" for i, q in enumerate(data)}
    names |= {q.index: f"ancilla {i}" for i, q in enumerate(ancillas)}
    deadline_ns = 1_000
    timeline = Timeline.of(page["compiled"].lower(data, ancillas, deadline_ns), names=names)

    words = {}
    for pl in timeline.pulses:
        key = (pl.lane, round(pl.start_ns, 3))
        words[key] = ("π if flagged", "right") if pl.kind == "drive" else "syndrome"
    call = timeline.calls[0]
    window = (call.due_ns - 110, call.due_ns + 40)

    def figure(p):
        return draw_zoomed(timeline, p, window, pulse_labels=words, breakdown=True, lane_in=0.4)

    alt = (
        "Timeline of one round of the distance-3 repetition code, with a lane for each data "
        "qubit's drive line, each ancilla's readout line and the GPU link. Both ancillas are "
        "read out for 400 ns, and their outcomes go to the decoder over the link. The kernel "
        f"waits for the answer with a {deadline_ns / 1000:g} microsecond timeout, plus "
        f"{call.link['feedback_ns']:g} ns of feedback, so the answer is due at "
        f"{call.due_ns:g} ns. A correction pulse on each data qubit, which plays only if the "
        "decoder's answer or the fallback table says so, starts at that moment."
    )
    save_themed(
        figure, "qec_round_timeline", title="One QEC round with a decoder deadline", desc=alt
    )


def deadlines(page) -> None:
    decode = page["decode"]
    nominal, worst = decode.nominal_latency_ns, decode.max_latency_ns
    deadline_us = np.array([1, 10, 20, 30, 40, 50, 55, 60, 63, 66, 70, 75, 80])
    runs = [
        page["run"](page["compiled"].lower(page["data"], page["ancillas"], int(d * 1000)))
        for d in deadline_us
    ]
    used, restored = np.array(runs).T

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.7)
        for x, word in ((nominal, "nominal"), (worst, "max")):
            ax.axvline(x / 1000, color=p.rule, lw=0.9, ls=(0, (1, 2)))
            ax.text(
                x / 1000,
                1.07,
                f"{word} {x / 1000:g} µs",
                ha="center",
                va="bottom",
                fontsize=8,
                color=p.muted,
            )
        ax.plot(deadline_us, used, "o-", color=p.classical, ms=3.4, lw=1.4)
        ax.plot(deadline_us, restored, "o-", color=p.q, ms=3.4, lw=1.4)
        mid = int(np.searchsorted(deadline_us, 50))
        end_label(
            ax, deadline_us[mid], used[mid], "decoder's answer used", p.classical, ha="right", dx=-7
        )
        end_label(ax, deadline_us[2], restored[2], "data back in 000", p.q, dy=9, ha="left", dx=0)
        ax.set_xlabel("deadline (µs)")
        ax.set_ylabel("fraction of shots")
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, deadline_us[-1] + 2)
        return fig

    alt = (
        "Two curves against the decoding deadline, from 1 to 80 microseconds, simulated with "
        f"the decoder's {nominal / 1000:g} microsecond nominal and {worst / 1000:g} microsecond "
        "maximum latency. The share of shots that use the decoder's answer rises from "
        f"{used[0]:g} at 1 microsecond through {used[deadline_us == 50][0]:.2f} at 50 to "
        f"{used[deadline_us == 75][0]:g} at the maximum. The share whose data qubits end back in "
        f"000 falls from {restored[0]:.3f} at 1 microsecond to {restored[-1]:.3f} at 80, because "
        "the qubits decay while they wait."
    )
    save_themed(figure, "qec_deadlines", title="Waiting for the decoder against decay", desc=alt)


def main() -> None:
    page = page_namespace("realtime_decoding.rst", stop_before=_FAST)
    round_timeline(page)
    deadlines(page)


if __name__ == "__main__":
    main()
