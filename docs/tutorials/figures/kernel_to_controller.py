# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "From Kernel to Controller" tutorial.

Usage: python docs/tutorials/figures/kernel_to_controller.py

Needs matplotlib (the ``[viz]`` extra). Writes, each with its ``-dark`` twin,
next to this file:

- ``exported_program.svg``, the export of the page's first kernel;
- ``drive_count.svg``, how many of the page's four pi pulses play at once;
- ``concurrent_drives.svg``, those pulses under two schedules;
- ``validation_corpus.svg``, what the text backend does with each program of
  the validation corpus;
- ``two_controllers.svg``, the program of ``examples/23_define_a_qpu.py``, a
  readout on one controller and the correction it decides on the other.
"""

import contextlib
import io
import runpy

import cudaq_pulse as pulse
import numpy as np
from _style import HERE, end_label, page_namespace, plot_axes, save_themed
from _timeline import Timeline, draw, draw_stack


def exported_program(page) -> None:
    timeline = Timeline.of(page["scheduled"])
    words = {
        (pl.lane, round(pl.start_ns, 3)): ("drive" if pl.kind == "drive" else "readout")
        for pl in timeline.pulses
    }
    (sync,) = timeline.syncs
    alt = (
        "Timeline of the exported drive_and_measure on qudit 0's drive and readout lines, in "
        "cycles of the 1 GHz clock: a 40-cycle drive from cycle 0, a sync at cycle "
        f"{sync.t_ns:g} that ties both lines, and a 200-cycle readout from there."
    )
    save_themed(
        lambda p: draw(timeline, p, pulse_labels=words, syncs="all"),
        "exported_program",
        title="The exported program",
        desc=alt,
    )


def drive_count(page) -> None:
    scheduled = page["flip_four"].lower(page["four"])
    edges = []
    for event in scheduled.schedule:
        if event.kind == "drive" and event.duration_cycles:
            edges += [(event.start, 1), (event.end, -1)]
    times, counts, active = [0], [0], 0
    for time, step in sorted(edges):
        active += step
        times.append(time)
        counts.append(active)
    clock = scheduled.quantum_system.clock_ghz
    end_ns = times[-1] / clock
    limit = 2

    def figure(p):
        fig, ax = plot_axes(p, height_in=1.7, width_in=5.0)
        ns = np.array(times) / clock
        ax.step(
            np.append(ns, end_ns * 1.5), np.append(counts, 0), where="post", color=p.drive, lw=1.6
        )
        ax.axhline(limit, color=p.readout, lw=1.0, ls=(0, (3, 2)))
        end_label(
            ax,
            end_ns * 1.5,
            limit,
            f"limit {limit}",
            p.readout,
            dx=-2,
            dy=6,
            ha="right",
            va="bottom",
        )
        ax.set_xlim(-2, end_ns * 1.5)
        ax.set_ylim(0, 4.6)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("drives at once")
        return fig

    alt = (
        f"The number of flip_four's drives playing at once, as late as possible: {max(counts)} "
        f"from 0 to {end_ns:g} ns, then none. The limit of {limit} is drawn across it."
    )
    save_themed(figure, "drive_count", title="Four drives against a limit of two", desc=alt)


def validation_corpus(page) -> None:
    lines = page["report"].lines
    statuses = ("PASS", "REFUSED")

    def figure(p):
        fig, ax = plot_axes(p, height_in=0.32 * len(lines) / 2 + 0.6)
        ax.axis("off")
        half = (len(lines) + 1) // 2
        for k, (status, name, _) in enumerate(lines):
            column, row = divmod(k, half)
            x, y = column * 1.0, -row
            color = p.classical if status == "PASS" else p.muted
            ax.plot([x], [y], "o" if status == "PASS" else "s", ms=6, color=color, mec="none")
            word = "passed" if status == "PASS" else "refused"
            ax.text(x + 0.04, y, f"{name}: {word}", va="center", fontsize=8, color=p.text)
        ax.set_xlim(-0.05, 2.0)
        ax.set_ylim(-half + 0.4, 0.6)
        return fig

    tally = {s: sum(1 for status, _, _ in lines if status == s) for s in statuses}
    alt = (
        f"The {len(lines)} programs of the validation corpus and what the text backend does "
        "with each: "
        + "; ".join(
            f"{name}, {'passed' if status == 'PASS' else 'refused'}" for status, name, _ in lines
        )
        + f". {tally['PASS']} pass and {tally['REFUSED']} are refused cleanly; none fail."
    )
    save_themed(figure, "validation_corpus", title="The validation corpus", desc=alt)


def concurrent_drives(page) -> None:
    four, limited = page["four"], page["limited"]
    late = Timeline.of(page["flip_four"].lower(four))
    rcp = Timeline.of(
        pulse.compile(page["flip_four"], schedule="rcp", control_system=limited).lower(four)
    )
    limit = limited.max_concurrent_drives
    titles = [
        "as late as possible: all four π pulses at once",
        f'schedule="rcp" with max_concurrent_drives={limit}: {limit} at a time',
    ]
    starts = sorted({round(pl.start_ns, 3) for pl in rcp.pulses})
    alt = (
        f"Two timelines of flip_four on the drive lines of {len(four)} qudits. As late as "
        f"possible, all {len(late.pulses)} pi pulses play at once, from 0 ns. With the "
        f"resource-constrained policy and at most {limit} drives at once, {limit} play from "
        f"{starts[0]:g} ns and the others from {starts[1]:g} ns."
    )
    save_themed(
        lambda p: draw_stack([late, rcp], titles, p, lane_in=0.36),
        "concurrent_drives",
        title="Four pi pulses under a limit of two at a time",
        desc=alt,
    )


def two_controllers() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        example = runpy.run_path(str(HERE.parents[2] / "examples" / "23_define_a_qpu.py"))
    timeline = Timeline.of(example["scheduled"])
    words = {}
    for pl in timeline.pulses:
        words[(pl.lane, round(pl.start_ns, 3))] = (
            ("flip if read 1", "left") if pl.conditional else "readout"
        )

    def figure(p):
        return draw(timeline, p, pulse_labels=words, breakdown=True, syncs="groups")

    call = timeline.calls[0]
    alt = (
        "Timeline of one program on two controllers. Controller 0 reads D1 out; the outcome "
        f"goes over the feedback link, {call.sent_ns - call.submit_ns:g} ns on the link, at "
        f"most {call.max_latency_ns:g} ns in the correction function and "
        f"{call.link['feedback_ns']:g} ns of feedback, to controller 1, which plays a pulse "
        f"on D3 only if the qubit read 1, at {call.due_ns:g} ns. A sync ties the two "
        "controllers' lines at the end of the readout."
    )
    save_themed(
        figure,
        "two_controllers",
        title="A readout on one controller, a correction on another",
        desc=alt,
    )


def main() -> None:
    page = page_namespace("kernel_to_controller.rst")
    exported_program(page)
    drive_count(page)
    concurrent_drives(page)
    validation_corpus(page)
    two_controllers()


if __name__ == "__main__":
    main()
