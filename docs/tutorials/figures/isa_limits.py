# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Getting More out of Your Controller's Instruction Set" tutorial.

Usage: python docs/tutorials/figures/isa_limits.py

Needs matplotlib (the ``[viz]`` extra). Runs the page's code, lowers its
programs with ``examples/isa_lowering/toy_isa.py``, and writes next to this
file:

- ``toy_flow.json``: each animated program's listings, the cycle its bits are
  ready on each controller, its decoder call, and, for each outcome, every
  instruction the emulated controllers issue (``toy_isa.run``) with the
  sentence the animation shows at each moment. ``docs/_static/toy-flow.js``
  plays it.
- ``controller_memory.json``: what each of the page's programs uses of a toy
  controller's instruction memory, waveform memory and oscillators, against
  the description's limits, and whether the compiler refuses it.
  ``docs/_static/mem-gauge.js`` draws it.
- ``isa_feedback.svg`` and ``isa_feedback-dark.svg``: the feedback branch
  across two controllers, the animation's last frame for readers without
  JavaScript.
"""

import dataclasses
import json
import sys

from _style import HERE, output_dir, page_namespace, save_themed
from _timeline import Timeline, draw, layout
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(HERE.parents[2] / "examples" / "isa_lowering"))

import cudaq_pulse as pulse  # noqa: E402
from cudaq_pulse.controller import ControllerLimitError  # noqa: E402
from toy_isa import lower, peer_latency, ready_cycles, run  # noqa: E402

PAGE = "isa_limits.rst"


def write_json(name: str, data: dict) -> None:
    text = json.dumps(data, indent=1, ensure_ascii=False)
    (output_dir(HERE) / name).write_text(text + "\n")


# ── The data-flow animation ───────────────────────────────────────────


def qudit_of(program, channel: str) -> int:
    label = program.wiring[program.channels[channel]].label
    return int(label.split("q")[1])


def controllers_of(qpu, programs) -> list[dict]:
    """Each controller: the qudits it is wired to, and its listing and
    channels in this program."""
    found = []
    for number, controller in enumerate(qpu.control_system.controllers):
        wired = set()
        for line in (*controller.drive_lines, *controller.readout_lines):
            wired.update(line.qudits)

        program = programs[number]
        channels = []
        for name, line in program.channels.items():
            role = program.wiring[line].label.split()[0]
            channels.append({"name": name, "role": role, "qudit": qudit_of(program, name)})

        found.append(
            {
                "id": number,
                "model": controller.model,
                "qudits": sorted(wired),
                "listing": program.code,
                "channels": channels,
            }
        )
    return found


def branches_on(program, bit: str) -> bool:
    for instruction in program.code:
        op, *args = instruction.split()
        if op == "branch" and args[0].removeprefix("!") == bit:
            return True
    return False


def bits_of(programs) -> list[dict]:
    """Each acquired bit: its controller, when its acquisition ends, and
    when it is ready there and on each other controller that branches on it."""
    by_number = {program.controller: program for program in programs}
    found = []
    for bit, (reader, ready) in sorted(ready_cycles(programs).items()):
        sender = by_number[reader]
        feedback = sender.limits.get("feedback_latency_cycles") or 0
        peers = []
        for other, program in by_number.items():
            if other != reader and branches_on(program, bit):
                at = ready + peer_latency(sender, other)
                peers.append({"controller": other, "ready_cycles": at})
        found.append(
            {
                "name": bit,
                "controller": reader,
                "ends_cycles": ready - feedback,
                "ready_cycles": ready,
                "peers": peers,
            }
        )
    return found


def calls_of(parts, clock_ghz: float) -> list[dict]:
    """Each decoder call: who submits which bits, the deadline, and which
    controllers take the answer."""
    found = []
    first = parts[0]
    for call in first.get("calls") or []:
        link = first["links"][str(call["link"])]
        submits, answers = [], []
        for part in parts:
            for event in part["events"]:
                if event["kind"] == "submit" and event["start_cycles"] == call["submit_cycles"]:
                    submits.append((part["controller"], event))
                if event["kind"] == "answer" and event["start_cycles"] == call["taken_cycles"]:
                    answers.append((part["controller"], event))
        ((sender, submit),) = submits
        _, answer = answers[0]
        sent = call["submit_cycles"] + call["submit_duration_cycles"]
        timeout = round(answer["timeout_ns"] * clock_ghz)
        found.append(
            {
                "link": call["link"],
                "name": link["name"],
                "controller": sender,
                "bits": [f"m{m}" for m in submit["measurements"]],
                "answer": f"a{answer['index']}",
                "submit_cycles": call["submit_cycles"],
                "sent_cycles": sent,
                "deadline_cycles": sent + timeout,
                "taken_cycles": call["taken_cycles"],
                "takers": [controller for controller, _ in answers],
            }
        )
    return found


def pulses_of(programs, issued) -> list[dict]:
    """The pulses one outcome plays, from the instructions issued."""
    found = []
    for step in issued:
        if not step.plays():
            continue
        program = programs[step.controller]
        op, channel, waveform, *rest = step.instruction.split()
        samples = program.waveforms[int(waveform.removeprefix("w"))]
        pulse_ = {
            "controller": step.controller,
            "channel": channel,
            "qudit": qudit_of(program, channel),
            "kind": "readout" if op == "acquire" else "drive",
            "start_cycles": step.start_cycles,
            "duration_cycles": len(samples),
        }
        if op == "acquire":
            pulse_["bit"] = rest[-1]
        found.append(pulse_)
    return found


def drives(pulses) -> set[tuple[int, int, int]]:
    return {
        (p["controller"], p["qudit"], p["start_cycles"]) for p in pulses if p["kind"] == "drive"
    }


def effect(played, others) -> str:
    """What an outcome does that another doesn't: the drives it plays or skips."""
    words = []
    for controller, qudit, _ in sorted(played - others):
        words.append(f"controller {controller} drives q{qudit}")
    for controller, qudit, _ in sorted(others - played):
        words.append(f"controller {controller} skips its drive on q{qudit}")
    return " and ".join(words)


def moments(scene: dict, shot: dict, others: set) -> list[list]:
    """The sentences the animation shows, each from the cycle it starts at."""
    said = []
    reads = [p for p in shot["pulses"] if p["kind"] == "readout"]
    names = " and ".join(p["bit"] for p in reads)
    qudits = " and ".join(f"q{p['qudit']}" for p in reads)
    end = reads[0]["start_cycles"] + reads[0]["duration_cycles"]
    said.append([0, f"Controller {reads[0]['controller']} acquires {names} from {qudits}."])

    feedback = scene["bits"][0]["ready_cycles"] - scene["bits"][0]["ends_cycles"]
    which = "Each bit" if len(reads) > 1 else names
    said.append(
        [
            end,
            f"The acquisition ends at {end}. {which} takes {feedback} cycles of feedback latency.",
        ]
    )

    for call in scene["calls"]:
        bits = " ".join(call["bits"])
        said.append(
            [
                call["submit_cycles"],
                f"submit sends {bits} over link L{call['link']} to the {call['name']}.",
            ]
        )
        window = call["deadline_cycles"] - call["sent_cycles"]
        said.append(
            [
                call["sent_cycles"],
                f"The answer may come any time in the next {window} cycles.",
            ]
        )
        takers = " and ".join(str(c) for c in call["takers"])
        if shot["outcomes"].get(f"{call['answer']}.ok"):
            value = shot["outcomes"].get(call["answer"], 0)
            got = f"the answer is there, {call['answer']} = {value}"
        else:
            got = f"no answer, so {call['answer']}.ok = 0"
        said.append(
            [
                call["taken_cycles"],
                f"Controllers {takers} poll at {call['taken_cycles']}: {got}. "
                f"Then {effect(drives(shot['pulses']), others)}.",
            ]
        )

    if not scene["calls"]:
        for bit in scene["bits"]:
            for peer in bit["peers"]:
                hop = peer["ready_cycles"] - bit["ready_cycles"]
                said.append(
                    [
                        bit["ready_cycles"],
                        f"{bit['name']} is on the bus at {bit['ready_cycles']}. "
                        f"It reaches controller {peer['controller']} {hop} cycles later.",
                    ]
                )
                value = shot["outcomes"].get(bit["name"], 0)
                said.append(
                    [
                        peer["ready_cycles"],
                        f"Controller {peer['controller']}'s wait ends at "
                        f"{peer['ready_cycles']}. It branches on {bit['name']} = {value}, "
                        f"so {effect(drives(shot['pulses']), others)}.",
                    ]
                )
    return said


def scene(name: str, title: str, qpu, scheduled, outcomes: list[tuple[str, dict]]) -> dict:
    parts = [scheduled.export_to_controller(controller=n) for n in range(2)]
    programs = [lower(part) for part in parts]
    clock_ghz = parts[0]["clock_ghz"]
    data = {
        "name": name,
        "title": title,
        "clock_ghz": clock_ghz,
        "controllers": controllers_of(qpu, programs),
        "bits": bits_of(programs),
        "calls": calls_of(parts, clock_ghz),
        "shots": [],
    }

    shots = []
    for label, chosen in outcomes:
        issued = run(programs, chosen)
        steps = [[s.controller, s.start_cycles, s.counter] for s in issued]
        shots.append(
            {
                "label": label,
                "outcomes": chosen,
                "issued": steps,
                "pulses": pulses_of(programs, issued),
            }
        )

    ends = []
    for shot in shots:
        others = set()
        for other in shots:
            if other is not shot:
                others |= drives(other["pulses"])
        played = drives(shot["pulses"])
        shot["label"] = f"{shot['label']}, {effect(played, others)}."
        shot["moments"] = moments(data, shot, others)
        for p in shot["pulses"]:
            ends.append(p["start_cycles"] + p["duration_cycles"])
        data["shots"].append(shot)
    data["end_cycles"] = max(ends)
    return data


def flow(page: dict) -> dict:
    qpu, q0, q1, q2 = page["qpu"], page["q0"], page["q1"], page["q2"]
    feedback = scene(
        "flip_other",
        "A branch on another controller's readout",
        qpu,
        page["split"],
        [("If m0 reads 1", {"m0": 1}), ("If m0 reads 0", {"m0": 0})],
    )
    compiled = pulse.compile(page["correct_across"], quantum_system=qpu, schedule="rcp")
    decoder = scene(
        "correct_across",
        "A decoder between two controllers",
        qpu,
        compiled.lower(q0, q1, q2),
        [
            ("If the answer is in time with a3 = 1", {"a3.ok": 1, "a3": 1}),
            ("If the answer is late", {"a3.ok": 0, "a3": 0}),
        ],
    )
    return {"source": f"docs/tutorials/{PAGE}", "scenes": [feedback, decoder]}


# ── The memory gauges ─────────────────────────────────────────────────


def usage(exports) -> dict:
    """The pulses one program issues, its waveform table's entries, and the
    tone frequencies on its line, over every program a kernel lowers to
    (the points of a sweep share one table)."""
    entries: dict[int, int] = {}
    frequencies = set()
    pulses = 0
    for export in exports:
        toy = lower(export)
        for number, samples in enumerate(toy.waveforms):
            entries[number] = len(samples)
        played = [e for e in export["events"] if e["kind"] in ("drive", "readout")]
        for event in played:
            frequencies.add(event["frequency_hz"])
        pulses = max(pulses, len(played))
    sizes = [entries[k] for k in sorted(entries)]
    return {
        "pulses": pulses,
        "samples": sum(sizes),
        "entries": sizes,
        "frequencies": len(frequencies),
    }


def refused(qpu, kernel, *args, control_system=None) -> bool:
    """Whether the compiler refuses *kernel* on the toy controllers."""
    compiled = pulse.compile(kernel, quantum_system=qpu, control_system=control_system)
    try:
        compiled.lower(qpu.qudits[0], *args)
    except ControllerLimitError:
        return True
    return False


def memory(page: dict) -> dict:
    qpu, q0, first, second = page["qpu"], page["q0"], page["first"], page["second"]
    looping = page["looping"]
    roomy = dataclasses.replace(
        first,
        max_pulses=10**6,
        waveform_memory_samples=10**6,
        max_frequencies_per_line=10**3,
    )
    roomy_system = pulse.ControlSystem(
        controllers=[roomy, second],
        reference_clock_ghz=qpu.control_system.reference_clock_ghz,
    )

    def measured(kernel, *points, control_system=roomy_system):
        compiled = pulse.compile(kernel, quantum_system=qpu, control_system=control_system)
        return [compiled.lower(q0, *point).export_to_controller() for point in points]

    rows = []
    sweep = measured(page["rabi"], (0.2,), (0.4,), control_system=None)
    gains = " and ".join(lower(p).code[-1].split()[-1] for p in sweep)
    rows.append(
        {
            "name": "rabi",
            "label": f"Rabi sweep at {gains}",
            "note": f"one entry, played at {gains}",
            **usage(sweep),
            "refused": refused(qpu, page["rabi"], 0.2),
        }
    )
    rows.append(
        {
            "name": "widths",
            "label": "30 pulse widths",
            **usage(measured(page["widths"], ())),
            "refused": refused(qpu, page["widths"]),
        }
    )
    rows.append(
        {
            "name": "hops",
            "label": "3 frequencies on one line",
            **usage(measured(page["hops"], ())),
            "refused": refused(qpu, page["hops"]),
        }
    )
    train = usage(measured(page["train"], ()))
    rows.append(
        {
            "name": "train",
            "label": f"{train['pulses']:,}-pulse train, no loop",
            **train,
            "refused": refused(qpu, page["train"]),
        }
    )

    looped = measured(page["train"], (), control_system=looping)
    rolled = pulse.compile(page["train"], quantum_system=qpu, control_system=looping)
    loop = rolled.lower(q0).export_to_controller(version=2)["events"][0]
    body = [e for e in loop["body"] if e["kind"] in ("play", "readout")]
    rows.append(
        {
            "name": "train_loop",
            "label": f"{loop['trips']:,}-pulse train, with loop",
            "note": f"one body, {loop['trips']:,} trips",
            **usage(looped),
            "pulses": len(body),
            "refused": refused(qpu, page["train"], control_system=looping),
        }
    )

    limits = {
        "pulses": first.max_pulses,
        "samples": first.waveform_memory_samples,
        "frequencies": first.max_frequencies_per_line,
    }
    return {
        "source": f"docs/tutorials/{PAGE}",
        "model": first.model,
        "limits": limits,
        "rows": rows,
    }


# ── The static figure ─────────────────────────────────────────────────


def feedback_figure(page: dict, data: dict) -> None:
    """The feedback branch across two controllers, as the animation ends: the
    readout, the bit's feedback and peer latency, and the flip it gates."""
    timeline = Timeline.of(page["split"], names={0: "q0", 1: "q1", 2: "q2"})
    flip = data["scenes"][0]
    (bit,) = flip["bits"]
    (peer,) = bit["peers"]
    feedback = bit["ready_cycles"] - bit["ends_cycles"]
    hop = peer["ready_cycles"] - bit["ready_cycles"]
    to_ns = 1 / flip["clock_ghz"]
    (gated,) = [pl for pl in timeline.pulses if pl.conditional]

    def figure(p):
        fig = draw(timeline, p, syncs="none", pulse_labels=words)
        ax = fig.axes[0]
        order, ys, _ = layout(timeline)
        readout_lane = next(ln.key for ln in order if ln.role == "readout")
        drive_lane = next(ln.key for ln in order if ln.role == "drive")
        base, height = ys[readout_lane]
        y = base + height * 0.3
        ends, ready, there = (
            bit["ends_cycles"] * to_ns,
            bit["ready_cycles"] * to_ns,
            peer["ready_cycles"] * to_ns,
        )
        ax.add_patch(Rectangle((ends, y - 0.03), ready - ends, 0.06, color=p.classical, lw=0))
        ax.text(
            (ends + ready) / 2,
            y + 0.06,
            f"feedback {feedback}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=p.classical,
        )
        drive_base, drive_height = ys[drive_lane]
        arrow = FancyArrowPatch(
            (ready, y - 0.03),
            (there, drive_base + drive_height * 0.62),
            arrowstyle="-|>,head_length=3.6,head_width=2",
            color=p.classical,
            lw=1.1,
            shrinkA=0,
            shrinkB=0.5,
        )
        ax.add_patch(arrow)
        ax.text(
            there + 4,
            (y + drive_base + drive_height) / 2,
            f"peer {hop}",
            ha="left",
            va="center",
            fontsize=8,
            color=p.classical,
        )
        return fig

    words = {}
    for pl in timeline.pulses:
        key = (pl.lane, round(pl.start_ns, 3))
        if pl.kind == "readout":
            words[key] = f"acquire {bit['name']}"
        else:
            words[key] = (f"flip if {bit['name']} = 1", "left")

    desc = (
        f"Timelines of the two toy controllers. Controller {bit['controller']} acquires "
        f"qudit 0 from cycle 0 to {bit['ends_cycles']} into bit {bit['name']}. The bit "
        f"takes {feedback} cycles of feedback latency and {hop} of peer latency, so "
        f"controller {peer['controller']} branches on it at cycle {peer['ready_cycles']} "
        f"and, only when it is 1, flips qudit 2 from {peer['ready_cycles']} to "
        f"{round(gated.end_ns / to_ns)}. One cycle is {to_ns:g} ns."
    )
    save_themed(figure, "isa_feedback", title="A branch on another controller's readout", desc=desc)


def main() -> None:
    page = page_namespace(PAGE)
    data = flow(page)
    write_json("toy_flow.json", data)
    write_json("controller_memory.json", memory(page))
    feedback_figure(page, data)


if __name__ == "__main__":
    main()
