# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenes for the animated figures (``docs/_static/pulse-visual.js``), from what
the compiler exports.

A scene is plain JSON, format ``pulse-visual/1``. ``program(scheduled)`` reads
one compiled program's controller export, as ``_timeline.Timeline.of`` does,
and returns its lanes, spans, marks, windows and messages, every time in
nanoseconds:

- a span is a drive or readout event, with ``when``: the outcomes it needs,
  read from the event's ``conditions``;
- a mark is a sync, a frame update, or the time a classical answer is due;
- a window is a stretch whose length the program doesn't know: when a
  classical answer may arrive, between the function's declared latencies;
- a message is a readout's outcome on its way to a call, or the answer on
  its way back to the pulses that read it.

A page's figure script adds the cases it plays (``given``: a value per name,
with the share of shots the page's simulation gives each), the sentence for
each moment, and for a step sequence the snapshots to move between.
``write`` saves the scenes next to this file, and draws each one's still,
``<scene>.svg`` and ``<scene>-dark.svg``, with ``tools/render-pulse-visual.js``
when Node is on the path.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from _style import HERE, output_dir
from _timeline import Timeline, _frame_text

FORMAT = "pulse-visual/1"
RENDER = HERE.parents[2] / "tools" / "render-pulse-visual.js"


def _ns(cycles: float, ghz: float) -> float:
    return round(cycles / ghz, 3)


def lane_id(key: tuple[str, int]) -> str:
    return f"{'k' if key[0] == 'link' else 'l'}{key[1]}"


class Names:
    """What a scene calls each readout and each answer, as the page's code does.

    A readout is named by its order in the program (*readouts*, or ``m0``,
    ``m1``, ...). An answer is named by the variable the page's kernel keeps
    it in (*answers*, by function), or else by its function's Python name,
    with its number when the program calls the function more than once
    (``accepted 1``, ``accepted 2``)."""

    def __init__(self, readouts: list[str] | None = None, answers: dict[str, str] | None = None):
        self.readouts = list(readouts or [])
        self.answers = dict(answers or {})

    def readout(self, order: int) -> str:
        return self.readouts[order] if order < len(self.readouts) else f"m{order}"


def program(scheduled, *, names: Names | None = None) -> dict[str, Any]:
    """The lanes, spans, marks, windows and messages of *scheduled*."""
    names = names or Names()
    timeline = Timeline.of(scheduled)
    export = scheduled.export_to_controller(timeline.clock_ghz)
    ghz = export["clock_ghz"]
    events = export["events"]
    by_index = {e["index"]: e for e in events}
    readout_order = {
        e["index"]: k for k, e in enumerate(e for e in events if e["kind"] == "readout")
    }
    answers = [e for e in events if e["kind"] == "answer"]
    calls_of: dict[str, int] = {}
    for e in answers:
        calls_of[e["function"]] = calls_of.get(e["function"], 0) + 1
    answer_name: dict[int, str] = {}
    seen: dict[str, int] = {}
    for e in answers:
        function = e["function"].rsplit("__", 1)[0]
        name = names.answers.get(function, function)
        seen[e["function"]] = seen.get(e["function"], 0) + 1
        many = calls_of[e["function"]] > 1
        answer_name[e["index"]] = f"{name} {seen[e['function']]}" if many else name
    manifest = {fn["name"]: fn for fn in (export.get("manifest") or {}).get("functions", [])}
    links = export.get("links") or {}

    def when(event: dict) -> list:
        out = []
        for cond in event["conditions"]:
            value = 1 if cond["arm"] else 0
            if cond["test"] == "is_set":
                (readout,) = cond["readouts"]
                out.append([names.readout(readout_order[readout]), value])
            elif cond["test"] == "answer" and cond["result"] == 0:
                (answer,) = cond["answers"]
                out.append([answer_name[answer], value])
            else:
                raise ValueError(f"no name for a {cond['test']!r} condition")
        return out

    lanes = [
        {"id": lane_id(lane.key), "label": lane.label, "role": lane.role} for lane in timeline.lanes
    ]
    known = {lane["id"] for lane in lanes}
    spans, marks, windows, messages, calls = [], [], [], [], []
    for e in events:
        lane = lane_id(("line", e["line"])) if e["line"] is not None else None
        if e["kind"] in ("drive", "readout") and lane in known:
            span = {
                "id": f"e{e['index']}",
                "lane": lane,
                "a": _ns(e["start_cycles"], ghz),
                "b": _ns(e["end_cycles"], ghz),
                "role": e["kind"],
            }
            if e["kind"] == "readout":
                span["label"] = names.readout(readout_order[e["index"]])
            rule = when(e)
            if rule:
                span["when"] = rule
            spans.append(span)
        elif e["kind"] == "frame_update" and e["value"] and lane in known:
            t = _ns(e["start_cycles"], ghz)
            marks.append({"role": "frame", "t": t, "lanes": [lane], "label": _frame_text(e)})
        elif e["kind"] == "sync" and e["lines"]:
            ids = [lane_id(("line", n)) for n in e["lines"] if lane_id(("line", n)) in known]
            if len(ids) > 1:
                marks.append({"role": "sync", "t": _ns(e["start_cycles"], ghz), "lanes": ids})

    for e in events:
        if e["kind"] != "submit":
            continue
        answer = next(
            a for a in answers if a["function"] == e["function"] and a["index"] > e["index"]
        )
        fn = manifest[e["function"]]
        link = links[str(e["link"])]
        lane = lane_id(("link", e["link"]))
        sent = _ns(e["end_cycles"], ghz)
        due = _ns(answer["start_cycles"], ghz)
        latency = fn["latency_ns"]
        feedback = float(link.get("feedback_ns") or 0)
        earliest = sent + float(link.get("fixed_ns") or 0) + latency["min"]
        latest = sent + latency["max"]
        if answer.get("timeout_ns") is not None:
            raise ValueError("a call with a timeout needs its in-time flag named")
        calls.append(
            {
                "function": e["function"].rsplit("__", 1)[0],
                "answer": answer_name[answer["index"]],
                "link": link.get("name") or str(e["link"]),
                "lane": lane,
                "submit": _ns(e["start_cycles"], ghz),
                "sent": sent,
                "due": due,
                "feedback_ns": feedback,
                "latency_ns": latency,
            }
        )
        if e.get("measurements"):
            source = by_index[e["measurements"][-1]]
            messages.append(
                {
                    "id": f"req{e['index']}",
                    "from": lane_id(("line", source["line"])),
                    "to": lane,
                    "a": _ns(source["end_cycles"], ghz),
                    "b": sent,
                    "role": "request",
                }
            )
        windows.append(
            {
                "lane": lane,
                "a": round(earliest, 3),
                "b": round(latest, 3),
                "role": "answer",
                "label": "answer may arrive",
            }
        )
        marks.append({"role": "due", "t": due, "lanes": [lane], "label": f"due {due:g} ns"})
        users = sorted(
            {
                lane_id(("line", u["line"]))
                for u in events
                if u["kind"] == "drive"
                and any(answer["index"] in c["answers"] for c in u["conditions"])
            }
        )
        for user in users:
            messages.append(
                {
                    "id": f"ans{answer['index']}-{user}",
                    "from": lane,
                    "to": user,
                    "a": round(due - feedback, 3),
                    "b": due,
                    "role": "answer",
                }
            )
    end = max([s["b"] for s in spans] + [m["t"] for m in marks] + [0.0])
    return {
        "clock_ghz": ghz,
        "end": round(end * 1.03, 3),
        "lanes": lanes,
        "spans": spans,
        "marks": marks,
        "windows": windows,
        "messages": messages,
        "calls": calls,
    }


def applies(rule: list | None, given: dict) -> bool:
    """Whether every [name, value] of *rule* holds in *given*: the engine's test."""
    return all(name in given and given[name] == value for name, value in rule or [])


def plays(scene: dict, given: dict) -> list[dict]:
    """The spans of *scene* that play when *given* holds."""
    return [s for s in scene["spans"] if applies(s.get("when"), given)]


def write(name: str, source: str, scenes: list[dict], directory: Path = HERE) -> Path:
    """Write *scenes* to ``<name>.json``, and draw each scene's still in both themes
    when Node is on the path (tests/test_pulse_visual.py checks them either way)."""
    path = output_dir(directory) / f"{name}.json"
    data = {"format": FORMAT, "source": source, "scenes": scenes}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    node = shutil.which("node")
    if node is None:
        print(f"node is not on PATH: the stills of {path.name} were not drawn", file=sys.stderr)
    else:
        subprocess.run([node, str(RENDER), str(path), str(path.parent)], check=True)
    return path
