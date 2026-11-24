# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "One QPU, Several Control Systems" tutorial.

Usage: python docs/tutorials/figures/control_systems.py

Needs matplotlib (the ``[viz]`` extra). Writes, each with its ``-dark`` twin,
next to this file, from the setups and the table the page builds:

- ``control_reset_timelines.svg``, the reset of Z4 under each control system;
- ``control_comparison.svg``, the page's table as a chart.
"""

import cudaq_pulse as pulse
import matplotlib.pyplot as plt
from _style import WIDTH_IN, _rc, page_namespace, save_themed
from _surface17 import color
from _timeline import Timeline, draw_stack


def reset_timelines(page) -> None:
    base, kernel = page["base"], page["active_reset"]
    timelines, titles, said = [], [], []
    for name, (control_system, schedule) in page["setups"].items():
        system = base.replace(control_system=control_system)
        compiled = pulse.compile(kernel, quantum_system=system, schedule=schedule)
        timeline = Timeline.of(compiled.lower(system.qudit("Z4")))
        forward = timeline.forwards[0]
        flip = max(pl.start_ns for pl in timeline.pulses if pl.kind == "drive")
        end = max(pl.end_ns for pl in timeline.pulses)
        timelines.append(timeline)
        titles.append(f"{name}: {end:g} ns")
        link = f" and {forward.peer_ns:g} ns between controllers" if forward.peer_ns else ""
        said.append(
            f"{name}: the flip starts at {flip:g} ns, after {forward.feedback_ns:g} ns of "
            f"feedback latency{link}, and the reset ends at {end:g} ns"
        )
    readout = next(pl for pl in timelines[0].pulses if pl.kind == "readout")
    alt = (
        f"Five timelines of the active reset of Z4, one per control system. In each, a pi "
        f"pulse is followed by a readout from {readout.start_ns:g} to {readout.end_ns:g} ns, "
        "and a second pi pulse that is sent only if the readout read 1. " + "; ".join(said) + "."
    )
    link_in = 0.3

    def figure(p):
        fig = draw_stack(timelines, titles, p, lane_in=0.3, link_in=link_in, syncs="none")
        # A latency too short to hold its label has it under its bar, on the
        # time axis: move it to the free space left of the bar.
        for ax, timeline in zip(fig.axes, timelines, strict=True):
            start = timeline.forwards[0].source[1]
            for text in ax.texts:
                if text.get_va() == "top" and text.get_text().startswith("feedback"):
                    text.set_position((start - 10, text.get_position()[1] + 0.03 + 0.22 * link_in))
                    text.set_ha("right")
                    text.set_va("center")
        return fig

    save_themed(
        figure,
        "control_reset_timelines",
        title="The reset of Z4 under five control systems",
        desc=alt,
    )


def comparison(page) -> None:
    rows = page["rows"]
    names = [name for name, *_ in rows]
    # Each panel: its title, its values, how a value is written, and the
    # qudit whose group gives its color (None: neither group's).
    panels = [
        ("reset of Z4\nlength (ns)", [r[1] for r in rows], "{:.0f}", "Z4"),
        ("reset of Z4\nerror, 1 − P(0) (%)", [100 * (1 - r[2]) for r in rows], "{:.2f}", "Z4"),
        ("one round\nlength (ns)", [r[3] for r in rows], "{:.0f}", None),
        ("data qubit\nidle error per round (%)", [100 * r[4] for r in rows], "{:.2f}", "D1"),
    ]

    def figure(p):
        with plt.rc_context(_rc(p)):
            fig, axes = plt.subplots(
                1,
                len(panels),
                figsize=(WIDTH_IN, 0.36 * len(rows) + 0.75),
                sharey=True,
                gridspec_kw={"left": 0.3, "right": 0.99, "top": 0.8, "bottom": 0.04, "wspace": 0.1},
            )
        y = [-k for k in range(len(rows))]
        for ax, (title, values, form, qudit) in zip(axes, panels, strict=True):
            shade = p.rule if qudit is None else color(p, qudit)
            ax.barh(y, values, height=0.5, color=shade, lw=0)
            for value, yy in zip(values, y, strict=True):
                ax.annotate(
                    form.format(value),
                    (value, yy),
                    xytext=(3, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=p.text,
                )
            ax.set_xlim(0, max(values) * 1.5)
            ax.set_title(title, loc="left", fontsize=8.5, color=p.text, pad=5)
            ax.grid(visible=False)
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
            ax.axvline(0, color=p.rule, lw=0.8)
        axes[0].set_yticks(y, names)
        axes[0].tick_params(axis="y", labelcolor=p.text)
        axes[0].set_ylim(y[-1] - 0.5, 0.5)
        return fig

    def row(r) -> str:
        return (
            f"{r[0]}: reset {r[1]:.0f} ns with error {1 - r[2]:.2%}, round {r[3]:.0f} ns "
            f"with idle error {r[4]:.2%}"
        )

    alt = (
        "Four bar charts side by side, one bar per control system: the length of the reset "
        "of Z4, the error of that reset, the length of one round of error correction, and "
        "the idle error of a data qubit over that round. " + "; ".join(map(row, rows)) + "."
    )
    save_themed(figure, "control_comparison", title="Five control systems compared", desc=alt)


def main() -> None:
    page = page_namespace("control_systems.rst")
    reset_timelines(page)
    comparison(page)


if __name__ == "__main__":
    main()
