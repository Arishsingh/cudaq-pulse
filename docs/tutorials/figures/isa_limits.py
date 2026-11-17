# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Getting More out of Your Controller's Instruction Set" tutorial.

Usage: python docs/tutorials/figures/isa_limits.py

Needs matplotlib (the ``[viz]`` extra). Writes ``isa_feedback.svg`` and
``isa_deadline.svg``, each with its ``-dark`` twin, next to this file: the
branch on another controller's readout, as the rcp schedule places it on the
page's two toy controllers, and the page's QEC round with its decoder
deadline.
"""

from _style import page_namespace, save_themed
from _timeline import Timeline, draw

NAMES = {0: "q0", 1: "q1", 2: "q2"}


def feedback(page) -> None:
    timeline = Timeline.of(page["split"], names=NAMES)
    words = {}
    for pl in timeline.pulses:
        key = (pl.lane, round(pl.start_ns, 3))
        words[key] = ("play if m0 is 1", "left") if pl.conditional else "acquire into m0"
    forward = timeline.forwards[0]
    readout_end = forward.source[1]
    flip = forward.use[1]
    alt = (
        "Timeline of flip_other on two toy controllers, in nanoseconds of the 1 GHz clock. "
        f"Controller 0 acquires q0 from 0 to {readout_end:g} ns into bit m0. The bit is ready "
        f"on controller 0 {forward.feedback_ns:g} ns later, after its feedback latency, and "
        f"on controller 1 another {forward.peer_ns:g} ns later, after the peer latency. "
        f"Controller 1 plays the flip of q2 from {flip:g} ns, only when the bit is 1."
    )
    save_themed(
        lambda p: draw(timeline, p, pulse_labels=words, syncs="groups"),
        "isa_feedback",
        title="An outcome read on one controller and branched on by another",
        desc=alt,
    )


def deadline(page) -> None:
    timeline = Timeline.of(page["qec"], names=NAMES)
    words = {}
    for pl in timeline.pulses:
        key = (pl.lane, round(pl.start_ns, 3))
        words[key] = ("flip if needed", "right") if pl.conditional else "syndrome"
    call = timeline.calls[0]
    alt = (
        "Timeline of one QEC round on the toy QPU: both syndromes are read at once on the "
        f"feedline's two tones, and the request leaves for the decoder at {call.submit_ns:g} ns. "
        f"The kernel waits for the answer with a {call.timeout_ns:g} ns timeout, plus the "
        f"link's {call.link['feedback_ns']:g} ns back, so it is due at {call.due_ns:g} ns. "
        "Two flips that play only on the shots that need them start there: the correction "
        "and the fallback."
    )
    save_themed(
        lambda p: draw(timeline, p, pulse_labels=words, breakdown=True),
        "isa_deadline",
        title="A QEC round with a decoder deadline, on the toy QPU",
        desc=alt,
    )


def main() -> None:
    page = page_namespace("isa_limits.rst")
    feedback(page)
    deadline(page)


if __name__ == "__main__":
    main()
