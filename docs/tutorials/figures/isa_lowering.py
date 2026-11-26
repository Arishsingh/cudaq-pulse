# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "Lowering to Your Controller's Instruction Set" tutorial.

Usage: python docs/tutorials/figures/isa_lowering.py

Needs matplotlib (the ``[viz]`` extra). Writes ``ramsey_reset_schedule.svg``
and its ``-dark`` twin next to this file, from the program the page compiles
and lowers for its toy QPU: every pulse, its frequency, the wait and the
feedback latency are read from the program's controller export.
"""

from dataclasses import replace

from _style import page_namespace, save_themed
from _timeline import Timeline, draw, layout


def ramsey_reset_schedule(page) -> None:
    scheduled = page["scheduled"]
    # Each pulse of the export carries its absolute frequency, so the figure labels
    # the detuned pulse with it and leaves out the frame updates between them.
    timeline = replace(Timeline.of(scheduled), marks=())
    first, second, flip = (pl for pl in timeline.pulses if pl.kind == "drive")
    (probe,) = (pl for pl in timeline.pulses if pl.kind == "readout")
    (wait,) = timeline.waits
    (forward,) = timeline.forwards
    assert flip.conditional and not (first.conditional or second.conditional)

    clock = scheduled.quantum_system.clock_ghz
    program = scheduled.export_to_controller()
    (line,) = (ln for ln in program["lines"] if ("line", ln["line"]) == first.lane)
    ghz = {
        (("line", event["line"]), round(event["start_cycles"] / clock, 3)): event["frequency_hz"]
        / 1e9
        for event in program["events"]
        if event["kind"] == "drive"
    }

    def at(pl):
        return (pl.lane, round(pl.start_ns, 3))

    def detuning(pl) -> str:
        return f"{ghz[at(pl)] * 1e3 - line['native_frequency_hz'] / 1e6:+g} MHz"

    assert detuning(first) == detuning(flip) == "+0 MHz"
    words = {
        at(first): "x90",
        at(second): f"x90, {detuning(second)}",
        at(probe): "probe",
        at(flip): ("x180 if it read 1", "left"),
    }

    def figure(p):
        fig = draw(timeline, p, pulse_labels=words, syncs="none")
        (ax,) = fig.axes
        # The latency's label goes left of its bar, clear of the time axis below it,
        # in the bar's color: the helper draws it in the page's color when it fits inside.
        (label,) = (t for t in ax.texts if t.get_text().startswith("feedback "))
        base, h = layout(timeline)[1][forward.lane]
        x0, x1 = ax.get_xlim()
        ns_per_pt = (x1 - x0) / (fig.get_figwidth() * 72 * ax.get_position().width)
        label.set_position((forward.source[1] - 5 * ns_per_pt, base + h * 0.4))
        label.set_horizontalalignment("right")
        label.set_verticalalignment("center")
        label.set_color(p.readout)
        return fig

    alt = (
        "Timeline of the scheduled program on qudit 0's drive and readout lines, with a lane "
        f"for the outcome. An x90 pulse from {first.start_ns:g} to {first.end_ns:g} ns, a "
        f"{wait.length_ns:g} ns wait, a second x90 pulse detuned by {detuning(second)} from "
        f"{second.start_ns:g} to {second.end_ns:g} ns, then the readout from "
        f"{probe.start_ns:g} to {probe.end_ns:g} ns. The outcome is ready "
        f"{forward.feedback_ns:g} ns after the readout ends, and the x180 pulse, which plays "
        f"only if the readout read 1, runs from {flip.start_ns:g} to {flip.end_ns:g} ns."
    )
    save_themed(
        figure,
        "ramsey_reset_schedule",
        title="The schedule of the Ramsey sequence and its reset",
        desc=alt,
    )


def main() -> None:
    # The page's blocks up to the export: the QPU, the kernel and its compile.
    page = page_namespace(
        "isa_lowering.rst", stop_before="program = scheduled.export_to_controller()"
    )
    ramsey_reset_schedule(page)


if __name__ == "__main__":
    main()
