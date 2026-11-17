# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "T-State Injection" tutorial.

Usage: python docs/tutorials/figures/t_state_injection.py

Needs matplotlib (the ``[viz]`` extra). Writes, each with its ``-dark`` twin,
next to this file and in the order the page shows them:

- ``t_frame_change.svg``, the schedule of ``inject``;
- ``tomography_axes.svg``, the three schedules of ``t_tomography``;
- ``t_bloch.svg``, the Bloch vector the tomography measures from ``|0>``;
- ``thermal_start.svg``, the levels D5 starts in;
- ``herald_loop.svg``, the heralded kernel with two attempts;
- ``herald_branches.svg``, the branches one herald keeps;
- ``herald_infidelity.svg``, the error of each start;
- ``herald_acceptance.svg``, the shots kept by each number of attempts;
- ``herald_latency.svg``, what the first answer's wait is made of.
"""

import math

import cudaq_pulse as pulse
import matplotlib.pyplot as plt
import numpy as np
from _style import end_label, page_namespace, plot_axes, save_themed
from _timeline import Timeline, busy_lanes, draw, draw_stack

AXES = ("X", "Y", "Z")


def frame_change(page) -> None:
    timeline = Timeline.of(page["inject"].lower(page["qubit"]))
    (y90,) = timeline.pulses
    (mark,) = timeline.marks
    words = {(y90.lane, round(y90.start_ns, 3)): "Y90"}

    def figure(p):
        return draw(timeline, p, pulse_labels=words, xlim=(-4, 2 * y90.end_ns))

    alt = (
        f"Schedule of the inject kernel on D5's drive line: a Y90 pulse from 0 to "
        f"{y90.end_ns:g} ns, then the T gate as a {mark.text} frame shift at {mark.t_ns:g} ns, "
        "a point with no length and no pulse."
    )
    save_themed(figure, "t_frame_change", title="The T gate is a frame shift", desc=alt)


def tomography_axes(page) -> None:
    q, kernel = page["qubit"], page["t_tomography"]
    timelines = [Timeline.of(kernel.lower(q, axis)) for axis in range(3)]
    turns = {
        "X": "−Y90 turns +X onto Z",
        "Y": "X90 turns +Y onto Z",
        "Z": "no pulse: Z is read as it is",
    }
    labels = []
    for name, timeline in zip(AXES, timelines, strict=True):
        drives = [pl for pl in timeline.pulses if pl.kind == "drive"]
        words = {(drives[0].lane, round(drives[0].start_ns, 3)): "Y90"}
        if len(drives) > 1:
            words[(drives[1].lane, round(drives[1].start_ns, 3))] = turns[name].split()[0]
        for pl in timeline.pulses:
            if pl.kind == "readout":
                words[(pl.lane, round(pl.start_ns, 3))] = "readout"
        labels.append(words)
    titles = [f"{name} axis: {turns[name]}" for name in AXES]

    def figure(p):
        return draw_stack(timelines, titles, p, labels=labels, lane_in=0.34, syncs="none")

    alt = (
        "Three schedules of the tomography kernel, one per axis, on D5's drive and readout "
        "lines. Each starts with the Y90 pulse and the T frame shift. For X, a minus Y90 pulse "
        "turns plus X onto Z; for Y, an X90 pulse turns plus Y onto Z; for Z there is no pulse. "
        "A readout then measures Z."
    )
    save_themed(figure, "tomography_axes", title="One pulse picks the axis", desc=alt)


def bloch_vector(page) -> None:
    q, kernel, expectation = page["qubit"], page["t_tomography"], page["expectation"]
    bloch = np.array(
        [
            expectation(pulse.simulate(kernel, q, axis, initial_state=page["ground"]))
            for axis in range(3)
        ]
    )
    angle = math.degrees(math.atan2(bloch[1], bloch[0]))

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.7, width_in=3.4)
        ax.axis("off")
        ax.set_aspect("equal")
        circle = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(circle), np.sin(circle), color=p.rule, lw=0.9)
        for dx, dy, word in ((1, 0, "+X"), (0, 1, "+Y"), (-1, 0, "−X"), (0, -1, "−Y")):
            ax.plot([0, dx], [0, dy], color=p.grid, lw=0.8)
            ax.text(
                1.14 * dx, 1.14 * dy, word, ha="center", va="center", fontsize=8.5, color=p.muted
            )
        ax.annotate(
            "",
            (1, 0),
            (0, 0),
            arrowprops=dict(arrowstyle="-|>", color=p.muted, lw=1.0, ls=(0, (2, 2))),
        )
        ax.text(0.55, -0.1, "|+⟩", ha="center", va="top", fontsize=8.5, color=p.muted)
        ax.annotate(
            "",
            (bloch[0], bloch[1]),
            (0, 0),
            arrowprops=dict(arrowstyle="-|>", color=p.drive, lw=1.8),
        )
        arc = np.linspace(0, math.radians(angle), 40)
        ax.plot(0.32 * np.cos(arc), 0.32 * np.sin(arc), color=p.drive, lw=1.0)
        ax.text(0.42, 0.17, f"{angle:.0f}°", fontsize=8.5, color=p.drive, va="center")
        end_label(ax, bloch[0], bloch[1], "measured |T⟩", p.drive, dx=4, dy=6, va="bottom")
        ax.set_xlim(-1.3, 1.6)
        ax.set_ylim(-1.25, 1.25)
        return fig

    alt = (
        "The Bloch vector the tomography measures from the ground state, seen from above, in "
        f"the X-Y plane: x {bloch[0]:.2f}, y {bloch[1]:.2f}, z {bloch[2]:.2f}. It points "
        f"{angle:.0f} degrees from plus X, where the plus state would point, and reaches the "
        "unit circle."
    )
    save_themed(figure, "t_bloch", title="The frame shift, seen by tomography", desc=alt)


def _first_readout(page, attempts: int) -> dict[int, float]:
    """The probability of each level the first herald readout reads."""
    result = pulse.simulate(page["compiled"].lower(page["qubit"], attempts, 2))
    first: dict[int, float] = {}
    for branch in result.branches:
        first[branch.levels[0]] = first.get(branch.levels[0], 0.0) + branch.probability
    return first


def _fmt(x: float) -> str:
    return f"{x:.1%}" if x >= 0.01 else f"{x:.1e}".replace("e-0", "e-")


def thermal_start(page) -> None:
    first = _first_readout(page, 1)
    levels = sorted(first)

    def figure(p):
        fig, ax = plot_axes(p, height_in=1.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=p.grid)
        y = -np.arange(len(levels))
        values = [first[k] for k in levels]
        colors = [p.drive if k == 0 else p.readout for k in levels]
        ax.hlines(y, 1e-4, values, color=p.grid, lw=2)
        ax.scatter(values, y, color=colors, s=26, zorder=3, lw=0)
        for x, yy in zip(values, y, strict=True):
            end_label(ax, x, yy, _fmt(x), p.text, dx=6, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 3)
        ax.set_yticks(y, [f"level {k}" for k in levels])
        ax.set_xlabel("share of shots at the start")
        return fig

    alt = (
        "The levels D5 starts in, as the first readout of the heralded kernel reads them: "
        + "; ".join(f"level {k}, {_fmt(first[k])}" for k in levels)
        + ". Everything not in level 0 is the error a herald removes."
    )
    save_themed(figure, "thermal_start", title="The thermal start of D5", desc=alt)


def _herald_words(timeline) -> dict:
    drives = [pl for pl in timeline.pulses if pl.kind == "drive"]
    words = {}
    for pl in timeline.pulses:
        key = (pl.lane, round(pl.start_ns, 3))
        if pl.kind == "drive":
            words[key] = "π if read 1" if pl.conditional else "Y90"
        else:
            words[key] = "herald" if pl.start_ns < drives[-1].start_ns else "tomography"
    return words


def herald_loop(page) -> Timeline:
    timeline = Timeline.of(page["compiled"].lower(page["qubit"], 2, 2))
    words = _herald_words(timeline)
    readouts = [pl for pl in timeline.pulses if pl.kind == "readout"]

    def figure(p):
        return draw(timeline, p, pulse_labels=words, call_labels={"herald": "herald"})

    alt = (
        "Timeline of the heralded T-state kernel with two attempts, on D5's drive and readout "
        f"lines and the controller link. Each attempt reads D5 out for "
        f"{readouts[0].end_ns - readouts[0].start_ns:g} ns, sends the outcome to the herald "
        "function, and plays a pi pulse only if the qubit read 1. After the second attempt, a "
        f"Y90 pulse and a pi/4 frame shift prepare the T state, and a last readout from "
        f"{readouts[-1].start_ns:g} ns measures it."
    )
    save_themed(figure, "herald_loop", title="Two heralded attempts, then the T state", desc=alt)
    return timeline


def herald_branches(page) -> None:
    result = pulse.simulate(page["compiled"].lower(page["qubit"], 1, 2))
    herald: dict[int, float] = {}
    tomography = {0: 0.0, 1: 0.0}
    accepted = 0.0
    for branch in result.branches:
        first = branch.levels[0]
        herald[first] = herald.get(first, 0.0) + branch.probability
        if page["herald_accepted"](branch):
            accepted += branch.probability
            tomography[min(branch.levels[-1], 1)] += branch.probability
    firsts = sorted(herald)

    def figure(p):
        fig, ax = plot_axes(p, height_in=1.9)
        ax.axis("off")
        y_first = {0: 0.0, **{a: -0.9 * k - 0.6 for k, a in enumerate(firsts[1:], start=1)}}
        y_leaf = {0: 0.45, 1: -0.45}
        mid = float(np.mean(list(y_first.values())))

        def edge(x0, y0, x1, y1, prob, color):
            xs = np.linspace(0, 1, 40)
            ys = y0 + (y1 - y0) * (3 * xs**2 - 2 * xs**3)
            ax.plot(
                x0 + (x1 - x0) * xs,
                ys,
                color=color,
                lw=0.6 + 7 * prob,
                alpha=0.35 + 0.65 * min(1.0, prob * 2),
                solid_capstyle="butt",
            )

        for a in firsts:
            kept = a == 0
            edge(0.0, mid, 1.0, y_first[a], herald[a], p.classical if kept else p.muted)
            word = "kept" if kept else "dropped"
            end_label(
                ax,
                1.0,
                y_first[a],
                f"herald read {a}: {_fmt(herald[a])}, {word}",
                p.text if kept else p.muted,
                dx=0 if kept else 6,
                dy=7 if kept else 0,
                ha="center" if kept else "left",
                va="bottom" if kept else "center",
            )
        for z, share in tomography.items():
            edge(1.0, y_first[0], 2.0, y_leaf[z], share, p.classical)
            word = "Z reads 0" if z == 0 else "Z reads 1 or 2"
            end_label(ax, 2.0, y_leaf[z], f"{word}: {_fmt(share)}", p.text, dx=6)
        ax.text(-0.04, mid, "start", ha="right", va="center", fontsize=8.5, color=p.muted)
        for x, word in ((1.0, "herald"), (2.0, "tomography, kept shots only")):
            ax.text(x, 0.95, word, ha="center", va="bottom", fontsize=8, color=p.muted)
        ax.set_xlim(-0.3, 2.75)
        ax.set_ylim(min(y_first.values()) - 0.3, 1.2)
        return fig

    alt = (
        "The branches of one heralded attempt measured on Z. "
        + "; ".join(f"The herald reads {a} with probability {_fmt(herald[a])}" for a in firsts)
        + f". Post-selection keeps only the branches whose herald read 0, {_fmt(accepted)} of "
        f"the shots: of those, Z reads 0 in {_fmt(tomography[0])} and 1 or 2 in "
        f"{_fmt(tomography[1])}, as the T state gives. The rest are dropped."
    )
    save_themed(figure, "herald_branches", title="What post-selection keeps", desc=alt)


def _rows(page):
    q, tomography, expectation, fidelity = (
        page["qubit"],
        page["t_tomography"],
        page["expectation"],
        page["fidelity"],
    )

    def unheralded(**start):
        bloch = np.array([expectation(pulse.simulate(tomography, q, a, **start)) for a in range(3)])
        return fidelity(bloch)

    starts = [
        ("start in |0⟩", unheralded(initial_state=page["ground"])),
        ("thermal start", unheralded()),
    ]
    attempts = [(n, *page["heralded_fidelity"](n)) for n in (1, 2, 3)]
    return starts, attempts


def herald_infidelity(starts, attempts) -> None:
    rows = [*starts, ("thermal, 1 herald", attempts[0][1])]
    names = [r[0] for r in rows]
    infidelity = np.array([1 - r[1] for r in rows])

    def figure(p):
        fig, ax = plot_axes(p, height_in=1.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=p.grid)
        y = -np.arange(len(rows))
        colors = [p.drive, p.readout, p.classical]
        ax.hlines(y, 1e-5, infidelity, color=p.grid, lw=2)
        ax.scatter(infidelity, y, color=colors, s=26, zorder=3, lw=0)
        for x, yy in zip(infidelity, y, strict=True):
            end_label(ax, x, yy, f"{x:.1e}", p.text, dx=6, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 0.3)
        ax.set_yticks(y, names)
        ax.set_xlabel("infidelity with |T⟩")
        return fig

    alt = (
        "Infidelity of the prepared state with the T state: "
        + "; ".join(f"{n}, {i:.1e}" for n, i in zip(names, infidelity, strict=True))
        + ". One herald brings the thermal start back to the error of the ground-state start."
    )
    save_themed(figure, "herald_infidelity", title="What one herald buys", desc=alt)


def herald_acceptance(attempts) -> None:
    counts = [n for n, _, _ in attempts]
    kept = [k for _, _, k in attempts]

    def figure(p):
        fig, ax = plot_axes(p, height_in=1.9, width_in=4.6)
        ax.plot(counts, kept, color=p.grid, lw=2, zorder=1)
        ax.scatter(counts, kept, color=p.classical, s=30, zorder=3, lw=0)
        for x, k in zip(counts, kept, strict=True):
            end_label(ax, x, k, f"{k:.1%}", p.text, dx=0, dy=-7, ha="center", va="top", fontsize=8)
        ax.set_xlim(0.6, 3.4)
        ax.set_ylim(0.96, 1.004)
        ax.set_xticks(counts, [f"{n} attempt{'s' if n > 1 else ''}" for n in counts])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_yticks([0.96, 0.98, 1.0])
        ax.set_ylabel("shots kept")
        return fig

    alt = (
        "The share of shots the herald keeps, by the number of attempts: "
        + "; ".join(f"{n}, {k:.1%}" for n, k in zip(counts, kept, strict=True))
        + ". The second attempt recovers most of the shots the first dropped; a third adds "
        "almost nothing."
    )
    save_themed(figure, "herald_acceptance", title="Shots kept by each attempt", desc=alt)


def herald_latency(timeline: Timeline) -> None:
    call = timeline.calls[0]
    source, use = call.source[1], call.use[1]
    window = (source - 0.5 * (use - source), use + 1.6 * (use - source))
    words = _herald_words(timeline)

    def figure(p):
        return draw(
            timeline,
            p,
            pulse_labels=words,
            xlim=window,
            breakdown=True,
            only=busy_lanes(timeline, window),
        )

    link, feedback = call.link["fixed_ns"], call.link["feedback_ns"]
    budget = call.due_ns - call.submit_ns
    alt = (
        "A close look at the first herald answer. The readout ends and its outcome is sent to "
        f"the herald function, whose answer is due {budget:g} ns later: {link:g} ns on the link, "
        f"at most {budget - link - feedback:g} ns in the function and {feedback:g} ns of "
        "feedback. The pi pulse that plays only if the qubit read 1 starts the moment the "
        "answer is due."
    )
    save_themed(figure, "herald_latency", title="When the first answer is due", desc=alt)


def main() -> None:
    page = page_namespace("t_state_injection.rst")
    frame_change(page)
    tomography_axes(page)
    bloch_vector(page)
    thermal_start(page)
    timeline = herald_loop(page)
    herald_branches(page)
    starts, attempts = _rows(page)
    herald_infidelity(starts, attempts)
    herald_acceptance(attempts)
    herald_latency(timeline)
    plt.close("all")


if __name__ == "__main__":
    main()
