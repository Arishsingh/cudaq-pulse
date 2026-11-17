# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "Real-Time Decoding with a Deadline" tutorial.

Usage: python docs/tutorials/figures/realtime_decoding.py

Needs matplotlib (the ``[viz]`` extra). Writes, each with its ``-dark`` twin,
next to this file:

- ``repetition_syndromes.svg``, what the page's ``lookup`` corrects for each
  syndrome;
- ``qec_round_timeline.svg`` and ``qec_round_deadline.svg``, the round the
  page compiles with its 1 µs deadline, whole and close up;
- ``qec_budgets.svg``, the page's two runs with a 1 µs deadline, before and
  after the decoder gets a budget that fits;
- ``qec_deadlines.svg``, the page's simulation of both arms over a range of
  deadlines.

All but ``qec_budgets`` use the decoder as first declared.
"""

import matplotlib.pyplot as plt
import numpy as np
from _style import end_label, page_namespace, plot_axes, save_themed
from _timeline import Timeline, busy_lanes, draw

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
        return draw(timeline, p, pulse_labels=words, lane_in=0.4)

    alt = (
        "Timeline of one round of the distance-3 repetition code, with a lane for each data "
        "qubit's drive line, each ancilla's readout line and the GPU link. Both ancillas are "
        "read out for 400 ns, and their outcomes go to the decoder over the link. The answer "
        f"is due at {call.due_ns:g} ns, and a correction pulse on each data qubit, which plays "
        "only if the decoder's answer or the fallback table says so, starts at that moment."
    )
    save_themed(
        figure, "qec_round_timeline", title="One QEC round with a decoder deadline", desc=alt
    )

    def close_up(p):
        return draw(
            timeline,
            p,
            pulse_labels=words,
            xlim=window,
            breakdown=True,
            only=busy_lanes(timeline, window),
            lane_in=0.4,
        )

    alt = (
        f"The last {call.due_ns - window[0]:g} ns of the round's wait. The kernel waits for the "
        f"answer with a {deadline_ns / 1000:g} microsecond timeout, plus "
        f"{call.link['feedback_ns']:g} ns of feedback on the link, so the answer is due at "
        f"{call.due_ns:g} ns. The dashed correction pulses on the data qubits start at that "
        "moment, whichever arm decides them."
    )
    save_themed(close_up, "qec_round_deadline", title="The end of the deadline", desc=alt)


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


def syndromes(page) -> None:
    lookup = page["lookup"]
    cases = [(0, 0), (1, 0), (1, 1), (0, 1)]
    fixes = [lookup(list(case)) for case in cases]

    def figure(p):
        fig, axes = plt.subplots(1, len(cases), figsize=(7.2, 1.35))
        for ax, case, fix in zip(axes, cases, fixes, strict=True):
            ax.axis("off")
            ax.set_xlim(-0.6, 4.6)
            ax.set_ylim(-0.9, 1.3)
            for i in range(3):
                flipped = bool(fix[i])
                ax.plot(
                    [2 * i],
                    [0],
                    "o",
                    ms=11,
                    color=p.drive if flipped else p.surface,
                    mec=p.drive if flipped else p.rule,
                    mew=1.2,
                )
                ax.text(2 * i, -0.62, f"d{i}", ha="center", va="center", fontsize=8, color=p.muted)
            for k, bit in enumerate(case):
                x = 2 * k + 1
                ax.plot(
                    [x],
                    [0],
                    "s",
                    ms=8,
                    color=p.readout if bit else p.surface,
                    mec=p.readout if bit else p.rule,
                    mew=1.2,
                )
                ax.text(
                    x, -0.62, f"s{k}={bit}", ha="center", va="center", fontsize=8, color=p.muted
                )
            flips = [f"d{i}" for i in range(3) if fix[i]]
            word = f"flip {flips[0]}" if flips else "no flip"
            ax.text(2, 1.0, word, ha="center", va="center", fontsize=8.5, color=p.text)
        fig.subplots_adjust(left=0.01, right=0.99, wspace=0.12)
        return fig

    alt = (
        "Four small diagrams of the three data qubits with the two syndrome bits between them, "
        "one per syndrome, with what the page's lookup corrects: "
        + "; ".join(
            f"syndrome {case}, " + (f"flip data qubit {fix.index(1)}" if any(fix) else "no flip")
            for case, fix in zip(cases, fixes, strict=True)
        )
        + "."
    )
    save_themed(figure, "repetition_syndromes", title="Each syndrome names one qubit", desc=alt)


def budgets(page) -> None:
    slow = page["run"](page["compiled"].lower(page["data"], page["ancillas"], 1_000))
    full = page_namespace("realtime_decoding.rst", stop_before="cudaq_qec")
    fast = full["run"](full["fast"].lower(full["data"], full["ancillas"], 1_000))
    names = ("62 µs decoder", "500 ns decoder")

    def figure(p):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.5), sharey=True)
        y = np.array([0.0, -1.0])
        panels = (
            ("decoder's answer used", 0, p.classical),
            ("data back in 000", 1, p.q),
        )
        for ax, (word, k, color) in zip(axes, panels, strict=True):
            values = [slow[k], fast[k]]
            ax.barh(y, values, height=0.45, color=color, alpha=0.85, lw=0)
            for x, yy in zip(values, y, strict=True):
                end_label(ax, x, yy, f"{x:g}", p.text, dx=5, fontsize=8)
            ax.set_xlim(0, 1.25)
            ax.set_xticks([0, 0.5, 1])
            ax.set_title(word, loc="left", fontsize=8.5, color=p.text)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=p.grid)
            ax.spines["bottom"].set_color(p.rule)
        axes[0].set_yticks(y, names)
        fig.subplots_adjust(wspace=0.12)
        return fig

    alt = (
        "The round with a 1 microsecond deadline, for the decoder as first declared and with a "
        f"500 ns budget. The share of shots that use the decoder's answer is {slow[0]:g} and "
        f"{fast[0]:g}; the share whose data qubits end back in 000 is {slow[1]:g} and "
        f"{fast[1]:g}."
    )
    save_themed(figure, "qec_budgets", title="Two arms, the same result", desc=alt)


def main() -> None:
    page = page_namespace("realtime_decoding.rst", stop_before=_FAST)
    syndromes(page)
    round_timeline(page)
    budgets(page)
    deadlines(page)


if __name__ == "__main__":
    main()
