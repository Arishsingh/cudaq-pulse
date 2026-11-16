# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "T-State Injection" tutorial.

Usage: python docs/tutorials/figures/t_state_injection.py

Needs matplotlib (the ``[viz]`` extra). Writes ``herald_timeline.svg`` and
``herald_timeline-dark.svg`` next to this file, from the heralded kernel the
page compiles, with two attempts.
"""

from _style import page_namespace, save_themed
from _timeline import Timeline, draw_zoomed


def main() -> None:
    page = page_namespace("t_state_injection.rst")
    timeline = Timeline.of(page["compiled"].lower(page["q"], 2, 2))

    drives = [pl for pl in timeline.pulses if pl.kind == "drive"]
    words = {}
    for pl in timeline.pulses:
        key = (pl.lane, round(pl.start_ns, 3))
        if pl.kind == "drive":
            words[key] = "π if read 1" if pl.conditional else "Y90"
        else:
            words[key] = "herald" if pl.start_ns < drives[-1].start_ns else "tomography"

    call = timeline.calls[0]
    source, use = call.source[1], call.use[1]
    window = (source - 0.5 * (use - source), use + 1.6 * (use - source))

    def figure(p):
        return draw_zoomed(timeline, p, window, pulse_labels=words)

    link, feedback = call.link["fixed_ns"], call.link["feedback_ns"]
    budget = call.due_ns - call.submit_ns
    alt = (
        "Timeline of the heralded T-state kernel with two attempts, on D5's drive and readout "
        "lines and the controller link, with a closer look at the first answer below. Each "
        "attempt reads D5 out for 300 ns and sends the outcome to the herald function, whose "
        f"answer is due {budget:g} ns later: {link:g} ns on the link, at most "
        f"{budget - link - feedback:g} ns in the function and {feedback:g} ns of feedback. "
        "A pi pulse that plays only if the qubit read 1 starts the moment the answer is due. "
        "After the second attempt, a Y90 pulse and a pi/4 frame shift prepare the T state, "
        "and a last readout measures it."
    )
    save_themed(
        figure, "herald_timeline", title="Heralded T-state preparation, two attempts", desc=alt
    )
    error_budget(page)


def error_budget(page) -> None:
    import cudaq_pulse as pulse
    import matplotlib.pyplot as plt
    import numpy as np

    q, tomography, expectation, fidelity = (
        page["q"],
        page["t_tomography"],
        page["expectation"],
        page["fidelity"],
    )

    def unheralded(**start):
        bloch = np.array([expectation(pulse.simulate(tomography, q, a, **start)) for a in range(3)])
        return fidelity(bloch), 1.0

    rows = [
        ("start in |0⟩", *unheralded(initial_state=page["ground"])),
        ("thermal start", *unheralded()),
    ]
    rows += [
        (f"{n} attempt{'s' if n > 1 else ''}", *page["heralded_fidelity"](n)) for n in (1, 2, 3)
    ]
    names = [r[0] for r in rows]
    infidelity = np.array([1 - r[1] for r in rows])
    kept = np.array([r[2] for r in rows])

    def figure(p):
        fig, (left, right) = plt.subplots(
            1,
            2,
            figsize=(7.2, 2.4),
            sharey=True,
            gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.08},
        )
        y = -np.arange(len(rows))
        for ax in (left, right):
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=p.grid)
            ax.spines["bottom"].set_color(p.rule)
        left.hlines(y, 1e-5, infidelity, color=p.grid, lw=2)
        left.plot(infidelity, y, "o", color=p.drive, ms=5, mec="none")
        for x, yy in zip(infidelity, y, strict=True):
            left.annotate(
                f"{x:.1e}",
                (x, yy),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=p.text,
            )
        left.set_xscale("log")
        left.set_xlim(1e-5, 0.3)
        left.set_xlabel("infidelity with |T⟩")
        left.set_yticks(y, names)
        right.hlines(y, 0.9, kept, color=p.grid, lw=2)
        right.plot(kept, y, "o", color=p.classical, ms=5, mec="none")
        for x, yy in zip(kept, y, strict=True):
            right.annotate(
                f"{x:.1%}",
                (x, yy),
                xytext=(-6, 0),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=8,
                color=p.text,
            )
        right.set_xlim(0.9, 1.004)
        right.set_xticks([0.9, 0.95, 1.0])
        right.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        right.set_xlabel("shots kept")
        return fig

    alt = (
        "Infidelity of the prepared state with the T state, and the share of shots kept: "
        + "; ".join(
            f"{n}, {i:.1e} and {k:.1%}" for n, i, k in zip(names, infidelity, kept, strict=True)
        )
        + "."
    )
    save_themed(figure, "t_error_budget", title="What each herald buys", desc=alt)


if __name__ == "__main__":
    main()
