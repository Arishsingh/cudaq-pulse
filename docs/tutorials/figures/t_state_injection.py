# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "T-State Injection" tutorial.

Usage: python docs/tutorials/figures/t_state_injection.py

Needs matplotlib (the ``[viz]`` extra). Writes, each with its ``-dark`` twin,
next to this file and in the order the page shows them:

- ``t_frame_change.svg``, the schedule of ``inject``;
- ``tomography_axes``, the three schedules of ``t_tomography``, animated;
- ``t_bloch.svg``, the Bloch vector the tomography measures from ``|0>``;
- ``thermal_start.svg``, the levels D5 starts in;
- ``herald_loop``, the heralded kernel with two attempts, animated;
- ``herald_branches``, the branches one herald keeps, animated;
- ``herald_infidelity.svg``, the error of each start;
- ``herald_acceptance.svg``, the shots kept by each number of attempts;
- ``herald_latency.svg``, what the first answer's wait is made of.

The animated figures are scenes of ``t_state_injection.json``; their stills,
``<scene>.svg``, are drawn from it by ``tools/render-pulse-visual.js``.
"""

import math

import cudaq_pulse as pulse
import matplotlib.pyplot as plt
import numpy as np
from _style import end_label, page_namespace, plot_axes, save_themed
from _timeline import Timeline, busy_lanes, draw
from _visual import Names, plays, program, write
from matplotlib.patches import Rectangle

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


def tomography_axes(page) -> dict:
    """One step per axis of ``t_tomography``: the pulse that turns the axis onto
    Z changes, and the readout reads what the page's ``expectation`` gives."""
    q, kernel, expectation = page["qubit"], page["t_tomography"], page["expectation"]
    turns = {"X": ("−Y90", "turns +X onto Z"), "Y": ("X90", "turns +Y onto Z")}
    steps, lanes, end = [], None, 0.0
    for axis, name in enumerate(AXES):
        compiled = program(kernel.lower(q, axis))
        lanes = compiled["lanes"]
        end = max(end, compiled["end"])
        drives = [s for s in compiled["spans"] if s["role"] == "drive"]
        (read,) = [s for s in compiled["spans"] if s["role"] == "readout"]
        spans = {"prep": [drives[0]["a"], drives[0]["b"]], "read": [read["a"], read["b"]]}
        spans["turn"] = [drives[1]["a"], drives[1]["b"]] if len(drives) > 1 else None
        value = expectation(pulse.simulate(kernel, q, axis, initial_state=page["ground"]))
        if name in turns:
            word, does = turns[name]
            label = f"{name} axis: {word} {does}"
            say = f"{name} axis: {word} {does}."
        else:
            word = None
            label = f"{name} axis: no pulse, Z is read as it is"
            say = f"{name} axis: no pulse. The T gate shows as a {_t_mark(compiled)} frame shift."
        steps.append(
            {
                "label": label,
                "short": f"{name} axis",
                "say": f"{say} The readout gives ⟨{name}⟩ = {value:.2f}.",
                "spans": spans,
                "labels": {"turn": word} if word else {},
                "marks": [m for m in compiled["marks"] if m["role"] == "frame"],
                "weight": round((1 + value) / 2, 4),
                "value": f"⟨{name}⟩ = {value:.2f}",
            }
        )
    drive_lane, readout_lane = lanes[0]["id"], lanes[1]["id"]
    return {
        "name": "tomography_axes",
        "title": "One pulse picks the axis",
        "end": end,
        "lanes": lanes,
        "spans": [
            {"id": "prep", "lane": drive_lane, "role": "drive", "label": "Y90"},
            {"id": "turn", "lane": drive_lane, "role": "drive"},
            {"id": "read", "lane": readout_lane, "role": "readout", "label": "readout"},
        ],
        "snapshots": steps,
        "share_label": "shots reading 0",
        "panels": ["timeline", "cases"],
    }


def _t_mark(compiled: dict) -> str:
    (mark,) = [m for m in compiled["marks"] if m["role"] == "frame"]
    return mark["label"]


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


def _yes(value: int) -> str:
    return "accepts" if value else "rejects"


def _answers_text(key: tuple[int, ...]) -> str:
    if len(set(key)) == 1:
        return f"{_yes(key[0])} both attempts"
    return " and ".join(f"{_yes(v)} attempt {k}" for k, v in enumerate(key, start=1))


def herald_loop(page) -> dict:
    """``heralded_t`` with two attempts, played once per answer the herald can
    give, with the share of shots the page's simulation gives each."""
    scheduled = page["compiled"].lower(page["qubit"], 2, 2)
    scene = program(
        scheduled,
        names=Names(readouts=["outcome", "outcome", "Z readout"], answers={"herald": "accepted"}),
    )
    flips = [s for s in scene["spans"] if s.get("when")]
    for k, span in enumerate(flips, start=1):
        span["label"] = f"π {k}"
    (y90,) = [s for s in scene["spans"] if s["role"] == "drive" and not s.get("when")]
    y90["label"] = "Y90"
    readouts = [s for s in scene["spans"] if s["role"] == "readout"]
    (mark,) = [m for m in scene["marks"] if m["role"] == "frame"]
    names = [call["answer"] for call in scene["calls"]]

    shares: dict[tuple[int, ...], float] = {}
    for branch in pulse.simulate(scheduled).branches:
        key = tuple(int(a) for a in branch.answers("herald"))
        shares[key] = shares.get(key, 0.0) + branch.probability
    cases = []
    for key, share in sorted(shares.items(), key=lambda item: -item[1]):
        given = dict(zip(names, key, strict=True))
        kept = page["herald_accepted"](_Answers(key))
        flipped = [s["label"] for s in flips if s in plays(scene, given)]
        if not flipped:
            does = "no π pulse plays"
        elif len(flipped) == 1:
            does = f"{flipped[0]} plays"
        else:
            does = "both π pulses play"
        moments = [[0.0, "Attempt 1 reads D5 out."]]
        for k, (call, value) in enumerate(zip(scene["calls"], key, strict=True), start=1):
            if k > 1:
                moments.append([readouts[k - 1]["a"], f"Attempt {k} reads D5 out."])
            moments.append([call["submit"], "The outcome goes to herald on the controller link."])
            after = "no π pulse." if value else f"π {k} turns D5 back to |0⟩."
            moments.append([call["due"], f"herald {_yes(value)} the shot: {after}"])
        moments.append([y90["a"], f"Y90 and the T gate's {mark['label']} frame shift prepare |T⟩."])
        moments.append(
            [
                readouts[-1]["a"],
                "The last readout measures Z. Post-selection "
                + ("keeps the shot." if kept else "drops the shot."),
            ]
        )
        cases.append(
            {
                "label": f"If herald {_answers_text(key)}, {does}.",
                "short": f"{', then '.join(_yes(v) for v in key)}: {'kept' if kept else 'dropped'}",
                "given": given,
                "weight": round(share, 6),
                "value": _fmt(share),
                "moments": moments,
            }
        )
    scene.update(
        name="herald_loop",
        title="Two heralded attempts, then the T state",
        cases=cases,
        panels=["timeline", "cases"],
    )
    return scene


class _Answers:
    """A branch as ``herald_accepted`` reads it: only its herald answers."""

    def __init__(self, answers: tuple[int, ...]):
        self._answers = [bool(a) for a in answers]

    def answers(self, function: str) -> list[bool]:
        assert function == "herald"
        return self._answers


def herald_branches(page) -> dict:
    """The branches of one heralded attempt measured on Z, grown one readout at
    a time: the herald's, post-selection, then the tomography's."""
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

    nodes = [{"id": "start", "col": 0, "p": 1.0, "text": "start", "step": 0}]
    for a in firsts:
        node = {"id": f"h{a}", "parent": "start", "col": 1, "p": round(herald[a], 6), "step": 1}
        node["text"] = f"herald read {a}: {_fmt(herald[a])}"
        if a != 0:
            node["drop"] = 2
        nodes.append(node)
    for z, share in tomography.items():
        word = "Z reads 0" if z == 0 else "Z reads 1 or 2"
        nodes.append(
            {
                "id": f"z{z}",
                "parent": "h0",
                "col": 2,
                "p": round(share, 6),
                "step": 3,
                "text": f"{word}: {_fmt(share)}",
            }
        )
    height = _place(nodes)
    reads = ", ".join(f"{a} on {_fmt(herald[a])}" for a in firsts)
    steps = [
        (
            "start",
            "Every shot starts in D5's thermal mixture",
            "Every shot starts in D5's thermal mixture.",
        ),
        (
            "herald readout",
            "The herald reads D5 out",
            f"The herald readout splits the shots: it reads {reads}.",
        ),
        (
            "post-select",
            "Post-selection keeps what herald accepts",
            f"Post-selection keeps the branches herald accepts, the {_fmt(accepted)} that read 0. "
            "The rest are dropped.",
        ),
        (
            "Z readout",
            "The tomography reads Z on the kept shots",
            f"The kept shots split about equally: Z reads 0 on {_fmt(tomography[0])} of all "
            f"shots and 1 or 2 on {_fmt(tomography[1])}, as |T⟩ gives.",
        ),
    ]
    return {
        "name": "herald_branches",
        "title": "What post-selection keeps",
        "tree": {
            "columns": ["", "herald readout", "tomography, kept shots only"],
            "nodes": nodes,
            "height": height,
        },
        "snapshots": [
            {"label": label, "short": short, "say": say, "spans": {}} for short, label, say in steps
        ],
        "panels": ["tree"],
    }


def _place(nodes: list[dict]) -> int:
    """Lay a tree out top to bottom: each leaf in a row of its own, in order,
    and each parent at the middle of its children. Returns the rows."""
    children: dict[str, list[dict]] = {}
    for node in nodes:
        if "parent" in node:
            children.setdefault(node["parent"], []).append(node)
    rows = 0

    def place(node: dict) -> float:
        nonlocal rows
        kids = children.get(node["id"], [])
        if not kids:
            node["y"] = float(rows)
            rows += 1
        else:
            ys = [place(kid) for kid in kids]
            node["y"] = (ys[0] + ys[-1]) / 2
        return node["y"]

    place(nodes[0])
    for node in nodes:
        node["y"] = round(node["y"] / max(1, rows - 1), 4)
    return rows


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
        fig = draw(
            timeline,
            p,
            pulse_labels=words,
            xlim=window,
            breakdown=True,
            only=busy_lanes(timeline, window),
        )
        # The link is the lowest lane here, so the function's share, which is
        # labelled under its bar, would lie on the time axis. Label it above the
        # bar instead, with no gap: the lane line above is close.
        (ax,) = fig.axes
        (label,) = [t for t in ax.texts if t.get_text().startswith(f"{call.function} ≤")]
        (bar,) = [b for b in ax.patches if isinstance(b, Rectangle) and b.get_x() == call.sent_ns]
        label.set_y(bar.get_y() + bar.get_height())
        label.set_va("bottom")
        return fig

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
    bloch_vector(page)
    thermal_start(page)
    starts, attempts = _rows(page)
    herald_infidelity(starts, attempts)
    herald_acceptance(attempts)
    herald_latency(Timeline.of(page["compiled"].lower(page["qubit"], 2, 2)))
    write(
        "t_state_injection",
        "docs/tutorials/t_state_injection.rst",
        [tomography_axes(page), herald_loop(page), herald_branches(page)],
    )
    plt.close("all")


if __name__ == "__main__":
    main()
