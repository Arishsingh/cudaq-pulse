# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "From Kernel to Controller" tutorial.

Usage: python docs/tutorials/figures/kernel_to_controller.py

Needs matplotlib (the ``[viz]`` extra). Writes ``concurrent_drives.svg`` and
``two_controllers.svg``, each with its ``-dark`` twin, next to this file: the
page's four pi pulses under two schedules, and the program of
``examples/23_define_a_qpu.py``, a readout on one controller and the
correction it decides on the other.
"""

import contextlib
import io
import runpy

import cudaq_pulse as pulse
from _style import HERE, page_namespace, save_themed
from _timeline import Timeline, draw, draw_stack


def concurrent_drives() -> None:
    page = page_namespace("kernel_to_controller.rst")
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
    concurrent_drives()
    two_controllers()


if __name__ == "__main__":
    main()
