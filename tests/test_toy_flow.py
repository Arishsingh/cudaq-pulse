# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ISA limits tutorial's animation and gauges show what its programs compile to.

``docs/tutorials/figures/isa_limits.py`` writes ``toy_flow.json`` and
``controller_memory.json`` from the page's own code: the controller export,
``toy_isa``'s listings and the instructions its emulated controllers issue,
and the compiler's refusals. ``docs/_static/toy-flow.js`` and ``mem-gauge.js``
draw them. This test regenerates both files and compares them with the
committed ones, numbers rounded to one decimal; checks that the gauges mark
as refused exactly the programs the compiler refuses; and runs the scripts'
exported functions under Node to check that they place each instruction,
pulse, bit, call and sentence of the data at its cycle. It then draws each
figure on a stand-in page under Node and uses it as a reader would: a figure
plays by itself only while it and its controls are in the window, its buttons
pause, step and replay it, its status line speaks only when the reader asks,
and a reader who asks for reduced motion gets a still frame. Node is optional
unless ``CUDAQ_PULSE_REQUIRE_NODE=1`` (set in the docs workflow).
To update the data, run ``python docs/tutorials/figures/isa_limits.py``.
"""

from __future__ import annotations

import contextlib
import html
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests._tools import tool_env

_ROOT = Path(__file__).resolve().parents[1]
_FIGURES = _ROOT / "docs" / "tutorials" / "figures"
_PAGE = _ROOT / "docs" / "tutorials" / "isa_limits.rst"
_STATIC = _ROOT / "docs" / "_static"
_EXT = _ROOT / "docs" / "_ext"
_FLOW = _STATIC / "toy-flow.js"
_GAUGE = _STATIC / "mem-gauge.js"
_DATA = ("toy_flow.json", "controller_memory.json")

pytestmark = pytest.mark.skipif(not _PAGE.exists(), reason="needs the source tree")


def _load(name: str) -> dict:
    return json.loads((_FIGURES / name).read_text())


def rounded(value):
    """*value* with every float rounded to one decimal."""
    if isinstance(value, float):
        return round(value, 1)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="needs matplotlib")
def test_the_data_is_what_the_page_compiles_to_today(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        "PULSE_FIGURES_DIR": str(tmp_path),
        "MPLBACKEND": "Agg",
    }
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    env.pop("CUDAQ_PULSE_QUANTUM_SYSTEM", None)
    result = subprocess.run(
        [sys.executable, "isa_limits.py"], cwd=_FIGURES, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-3000:]
    for name in _DATA:
        drawn = json.loads((tmp_path / name).read_text())
        assert rounded(drawn) == rounded(_load(name)), (
            f"{name} is not what isa_limits.py writes now; "
            "run python docs/tutorials/figures/isa_limits.py"
        )


def _page_namespace(tmp_path: Path) -> dict:
    """The names the page's code defines, run as the tutorial test runs it."""
    spec = importlib.util.spec_from_file_location("_pages", _ROOT / "tests" / "test_tutorials.py")
    pages = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pages)
    sys.path.insert(0, str(_ROOT / "examples" / "isa_lowering"))
    namespace: dict = {"__name__": "isa_limits_page"}
    try:
        with contextlib.chdir(tmp_path), contextlib.redirect_stdout(io.StringIO()):
            for classes, code, _ in pages.code_blocks(_PAGE):
                if not {"fragment", "gpu", "viz", "cudaq-qec"} & classes:
                    exec(compile(code, _PAGE.name, "exec", dont_inherit=True), namespace)
    finally:
        sys.path.remove(str(_ROOT / "examples" / "isa_lowering"))
    return namespace


def test_the_gauges_mark_refused_exactly_what_the_compiler_refuses(tmp_path):
    import cudaq_pulse as pulse
    from cudaq_pulse.controller import ControllerLimitError

    page = _page_namespace(tmp_path)
    qpu, q0 = page["qpu"], page["q0"]
    programs = {
        "rabi": (page["rabi"], (0.2,), None),
        "widths": (page["widths"], (), None),
        "hops": (page["hops"], (), None),
        "train": (page["train"], (), None),
        "train_loop": (page["train"], (), page["looping"]),
    }
    refused = set()
    for name, (kernel, args, system) in programs.items():
        compiled = pulse.compile(kernel, quantum_system=qpu, control_system=system)
        try:
            compiled.lower(q0, *args)
        except ControllerLimitError:
            refused.add(name)

    data = _load("controller_memory.json")
    assert {row["name"] for row in data["rows"]} == set(programs)
    assert {row["name"] for row in data["rows"] if row["refused"]} == refused
    over = {
        row["name"]
        for row in data["rows"]
        if any(row[key] > limit for key, limit in data["limits"].items())
    }
    assert over == refused
    first = page["first"]
    assert data["limits"] == {
        "pulses": first.max_pulses,
        "samples": first.waveform_memory_samples,
        "frequencies": first.max_frequencies_per_line,
    }
    # The page's printed refusals say the same numbers.
    text = _PAGE.read_text()
    by_name = {row["name"]: row for row in data["rows"]}
    assert f"its waveforms take {by_name['widths']['samples']}" in text
    assert f"line 0: {by_name['hops']['frequencies']} tone frequencies" in text
    assert f"{by_name['train']['pulses']} drives and readouts" in text


def test_the_page_shows_scenes_and_rows_the_data_has():
    text = _PAGE.read_text()
    flow, memory = _load("toy_flow.json"), _load("controller_memory.json")
    scenes = {s["name"] for s in flow["scenes"]}
    rows = {r["name"] for r in memory["rows"]}
    shown_scenes, shown_rows = set(), set()
    for line in text.splitlines():
        option = line.strip()
        if option.startswith(":scene:"):
            shown_scenes.add(option.split()[1])
        if option.startswith(":rows:"):
            shown_rows.update(option.split()[1:])
    assert shown_scenes == scenes
    assert shown_rows == rows


def test_the_fallback_says_what_it_shows_in_both_themes():
    """Readers without JavaScript see isa_feedback.svg; its description is the
    page's alt text."""
    lines = _PAGE.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if ":fallback: figures/isa_feedback.svg" in line)
    alt = next(line for line in lines[start:] if ":alt:" in line).split(":alt:", 1)[1].strip()
    for name in ("isa_feedback.svg", "isa_feedback-dark.svg"):
        root = ET.fromstring((_FIGURES / name).read_text())
        desc = root.find("{http://www.w3.org/2000/svg}desc").text
        assert " ".join(desc.split()) == alt, name


# ── The scripts, under Node ───────────────────────────────────────────

_DUMP = """
const flow = require(process.argv[1]);
const gauge = require(process.argv[2]);
const data = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const memory = JSON.parse(require("fs").readFileSync(process.argv[4], "utf8"));
const scenes = data.scenes.map((scene) => {
  const cycles = new Set([0, scene.end_cycles]);
  for (const shot of scene.shots) {
    for (const [, cycle] of shot.issued) cycles.add(cycle);
    for (const [cycle] of shot.moments) cycles.add(cycle);
    for (const p of shot.pulses) {
      cycles.add(p.start_cycles);
      cycles.add(p.start_cycles + p.duration_cycles);
    }
  }
  for (const bit of scene.bits) {
    for (const c of [bit.ends_cycles - 1, bit.ends_cycles, bit.ready_cycles]) cycles.add(c);
    for (const peer of bit.peers) {
      cycles.add((bit.ready_cycles + peer.ready_cycles) / 2);
      cycles.add(peer.ready_cycles);
    }
  }
  for (const call of scene.calls) {
    cycles.add(call.submit_cycles);
    cycles.add(call.sent_cycles);
    cycles.add(call.deadline_cycles);
    cycles.add(call.taken_cycles);
  }
  const at = [...cycles].sort((a, b) => a - b);
  const total = scene.shots.length * (flow.SHOT_MS + flow.HOLD_MS);
  const elapsed = [0, flow.SHOT_MS / 2, flow.SHOT_MS, flow.SHOT_MS + flow.HOLD_MS, total];
  return {
    name: scene.name,
    span: flow.span(scene),
    lanes: flow.lanes(scene),
    description: flow.description(scene),
    frames: elapsed.map((e) => [e, flow.frameAt(scene, e)]),
    states: scene.shots.map((_, s) => at.map((t) => [t, flow.stateAt(scene, s, t)])),
    steps: flow.steps(scene),
    stepTexts: flow.steps(scene).map((_, k) => flow.stepText(scene, k)),
    resumes: flow.steps(scene).map(([s, cycle]) => [
      flow.frameAt(scene, flow.elapsedAt(scene, s, cycle)),
      flow.stepAt(scene, s, cycle),
      flow.stepAt(scene, s, cycle + 0.5),
    ]),
  };
});
const rows = memory.rows.map((row) => ({
  name: row.name,
  over: gauge.overLimits(row, memory.limits, Object.keys(memory.limits)),
  note: gauge.noteText(row),
  gauges: Object.keys(memory.limits).map((key) => [
    key, gauge.gauge(row[key], memory.limits[key]), gauge.valueText(row, key),
  ]),
}));
process.stdout.write(JSON.stringify({
  timing: [flow.SHOT_MS, flow.HOLD_MS],
  scenes,
  rows,
  limitAt: gauge.LIMIT_AT,
  columns: gauge.COLUMNS,
  gaugeDescription: gauge.description(memory, memory.rows, Object.keys(memory.limits)),
}));
"""


def _node(script: str, *arguments: str) -> dict:
    """What *script* prints, run under Node on the two scripts, their data and
    *arguments*."""
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CUDAQ_PULSE_REQUIRE_NODE"):
            pytest.fail("CUDAQ_PULSE_REQUIRE_NODE is set, but node is not on PATH")
        pytest.skip("needs Node.js to run the animation's scripts")
    files = [_FLOW, _GAUGE, _FIGURES / _DATA[0], _FIGURES / _DATA[1]]
    result = subprocess.run(
        [node, "-e", script, *map(str, files), *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        env=tool_env(),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def scripts():
    return _node(_DUMP)


def _scenes(scripts):
    data = _load("toy_flow.json")
    for scene, drawn in zip(data["scenes"], scripts["scenes"], strict=True):
        assert drawn["name"] == scene["name"]
        yield scene, drawn


def test_each_outcome_plays_once_from_the_first_cycle_to_the_last(scripts):
    shot_ms, hold_ms = scripts["timing"]
    for scene, drawn in _scenes(scripts):
        frames = {elapsed: frame for elapsed, frame in drawn["frames"]}
        assert frames[0] == {"shot": 0, "t": 0, "done": False}
        assert frames[shot_ms / 2]["t"] == scene["end_cycles"] / 2
        assert frames[shot_ms]["t"] == scene["end_cycles"]
        assert frames[shot_ms + hold_ms] == {"shot": 1, "t": 0, "done": False}
        last = frames[len(scene["shots"]) * (shot_ms + hold_ms)]
        assert last["done"] and last["shot"] == len(scene["shots"]) - 1
        assert drawn["span"] >= scene["end_cycles"] and drawn["span"] % 40 == 0


def test_each_listing_shows_the_instruction_issued_last_and_strikes_skipped_ones(scripts):
    for scene, drawn in _scenes(scripts):
        listings = {c["id"]: c["listing"] for c in scene["controllers"]}
        for shot, states in zip(scene["shots"], drawn["states"], strict=True):
            for t, state in states:
                for place in state["listings"]:
                    controller = place["controller"]
                    steps = [(cycle, k) for c, cycle, k in shot["issued"] if c == controller]
                    mine = [k for cycle, k in steps if cycle <= t]
                    assert place["counter"] == (mine[-1] if mine else -1)
                    listing = listings[controller]
                    done = t >= max(cycle for cycle, _ in steps)
                    passed = range(len(listing) if done else max(mine, default=-1))
                    expected = [k for k in passed if k not in mine and not listing[k].endswith(":")]
                    assert place["skipped"] == expected
            # By the end, the skipped lines are the arms this outcome didn't take.
            end = dict(states)[scene["end_cycles"]]
            for place in end["listings"]:
                listing = listings[place["controller"]]
                issued = {k for c, _, k in shot["issued"] if c == place["controller"]}
                untaken = [
                    k
                    for k, line in enumerate(listing)
                    if k not in issued and not line.endswith(":")
                ]
                assert place["skipped"] == untaken


def test_each_pulse_lights_while_it_plays_and_each_sentence_at_its_cycle(scripts):
    for scene, drawn in _scenes(scripts):
        for shot, states in zip(scene["shots"], drawn["states"], strict=True):
            by_cycle = dict(states)
            for p in shot["pulses"]:
                key = (p["controller"], p["channel"], p["start_cycles"])
                live = {
                    (q["controller"], q["channel"], q["start_cycles"])
                    for q in by_cycle[p["start_cycles"]]["live"]
                }
                assert key in live
                ended = by_cycle[p["start_cycles"] + p["duration_cycles"]]["live"]
                assert key not in {
                    (q["controller"], q["channel"], q["start_cycles"]) for q in ended
                }
            for cycle, _ in shot["moments"]:
                later = [m for m in shot["moments"] if m[0] == cycle]
                assert by_cycle[cycle]["caption"] == later[-1][1]
            assert shot["label"] in drawn["description"]


def _bit_at(by_cycle: dict, t: float, name: str) -> dict | None:
    found = [b for b in by_cycle[t]["bits"] if b["name"] == name]
    return found[0] if found else None


def _call_at(by_cycle: dict, t: float) -> dict:
    (state,) = by_cycle[t]["calls"]
    return state


def test_each_bit_waits_out_its_latencies_and_each_answer_its_window(scripts):
    for scene, drawn in _scenes(scripts):
        for shot, states in zip(scene["shots"], drawn["states"], strict=True):
            by_cycle = dict(states)
            for bit in scene["bits"]:
                name = bit["name"]
                assert _bit_at(by_cycle, bit["ends_cycles"] - 1, name) is None
                held = _bit_at(by_cycle, bit["ends_cycles"], name)
                assert held["stage"] == "held" and held["value"] is None
                ready = _bit_at(by_cycle, bit["ready_cycles"], name)
                assert ready["value"] == shot["outcomes"].get(name)
                for peer in bit["peers"]:
                    halfway = (bit["ready_cycles"] + peer["ready_cycles"]) / 2
                    crossing = _bit_at(by_cycle, halfway, name)
                    assert crossing["stage"] == "crossing" and crossing["f"] == 0.5
                    assert crossing["to"] == peer["controller"]
                    assert _bit_at(by_cycle, peer["ready_cycles"], name)["stage"] == "arrived"
            for call in scene["calls"]:
                assert _call_at(by_cycle, call["submit_cycles"])["stage"] == "sending"
                assert _call_at(by_cycle, call["sent_cycles"])["stage"] == "window"
                assert _call_at(by_cycle, call["deadline_cycles"])["stage"] == "deadline"
                taken = _call_at(by_cycle, call["taken_cycles"])
                if shot["outcomes"][f"{call['answer']}.ok"]:
                    want = f"{call['answer']} = {shot['outcomes'][call['answer']]}"
                else:
                    want = f"{call['answer']}.ok = 0"
                assert taken["stage"] == "taken" and taken["taken"] == want


def test_the_timeline_has_a_lane_per_channel_and_the_latencies_of_the_data(scripts):
    for scene, drawn in _scenes(scripts):
        lanes = drawn["lanes"]
        channels = [lane for lane in lanes if lane["kind"] == "channel"]
        expected = [
            f"C{c['id']} {ch['role']} q{ch['qudit']}"
            for c in scene["controllers"]
            for ch in c["channels"]
        ]
        assert [lane["label"] for lane in channels] == expected
        (bus,) = [lane for lane in lanes if lane["kind"] == "bus"]
        spans = {(s["a"], s["b"], s["label"]) for s in bus["spans"]}
        for bit in scene["bits"]:
            feedback = bit["ready_cycles"] - bit["ends_cycles"]
            assert (bit["ends_cycles"], bit["ready_cycles"], f"feedback {feedback}") in spans
            for peer in bit["peers"]:
                hop = peer["ready_cycles"] - bit["ready_cycles"]
                assert (bit["ready_cycles"], peer["ready_cycles"], f"peer {hop}") in spans
        links = [lane for lane in lanes if lane["kind"] == "link"]
        assert len(links) == len(scene["calls"])
        for lane, call in zip(links, scene["calls"], strict=True):
            window = [s for s in lane["spans"] if s["role"] == "window"]
            assert [(s["a"], s["b"]) for s in window] == [
                (call["sent_cycles"], call["deadline_cycles"])
            ]
        # A pulse only some outcomes play is drawn as conditional.
        for lane in channels:
            for s in lane["spans"]:
                plays = sum(
                    any(
                        p["controller"] == lane["controller"]
                        and p["channel"] == lane["channel"]
                        and p["start_cycles"] == s["a"]
                        for p in shot["pulses"]
                    )
                    for shot in scene["shots"]
                )
                assert s["conditional"] == (plays < len(scene["shots"]))


def test_the_animations_numbers_are_the_pages(scripts):
    """The latencies and cycles the page states are the ones the data plays."""
    flow = {s["name"]: s for s in _load("toy_flow.json")["scenes"]}
    (bit,) = flow["flip_other"]["bits"]
    (peer,) = bit["peers"]
    assert (bit["ends_cycles"], bit["ready_cycles"], peer["ready_cycles"]) == (200, 280, 320)
    (call,) = flow["correct_across"]["calls"]
    assert (call["submit_cycles"], call["taken_cycles"]) == (280, 352)
    assert call["deadline_cycles"] - call["sent_cycles"] == 60
    assert call["takers"] == [0, 1]
    text = " ".join(_PAGE.read_text().split())
    hop = peer["ready_cycles"] - bit["ready_cycles"]
    feedback = bit["ready_cycles"] - bit["ends_cycles"]
    assert f"ready for the second controller at cycle {peer['ready_cycles']}" in text
    assert f"The flip waits {feedback + hop} cycles: {feedback} of feedback latency" in text
    assert f"The request leaves at cycle {call['submit_cycles']}" in text
    assert f"Both controllers poll at cycle {call['taken_cycles']}" in text


def test_the_gauges_fill_to_the_limit_mark_and_say_their_numbers(scripts):
    memory = _load("controller_memory.json")
    limit_at = scripts["limitAt"]
    for row, drawn in zip(memory["rows"], scripts["rows"], strict=True):
        assert drawn["name"] == row["name"]
        assert bool(drawn["over"]) == row["refused"]
        assert (drawn["note"] == "refused when it compiles") == row["refused"]
        for key, gauge, text in drawn["gauges"]:
            limit = memory["limits"][key]
            reach = min(row[key] / limit * limit_at, 1)
            assert gauge["reach"] == pytest.approx(reach)
            assert gauge["inside"] == pytest.approx(min(reach, limit_at))
            assert gauge["cut"] == (row[key] / limit * limit_at > 1)
            assert text.startswith(f"{row[key]:,} ")
    widths = next(d for d in scripts["rows"] if d["name"] == "widths")
    assert ["samples", "1,200 samples in 30 entries"] == [
        widths["gauges"][1][0],
        widths["gauges"][1][2],
    ]
    assert set(scripts["columns"]) == set(memory["limits"])
    for row in memory["rows"]:
        assert row["label"] in scripts["gaugeDescription"]


# ── The figures, on a stand-in page ───────────────────────────────────

# Draws each figure with its script on a page that has only what the scripts
# use, and records what a reader would see and hear at each point: as the
# figure scrolls into and out of the window, as the browser draws frames, as
# buttons and keys are pressed, and as the reduced-motion setting changes.
_PLAY = r"""
const fs = require("fs");
const [flowFile, gaugeFile, flowData, memoryData, ...gauges] = process.argv.slice(1);
const FIGURES = {
  flow: { file: flowFile, cls: "pulse-toy-flow", id: "pulse-data-toy-flow", data: flowData },
  gauge: {
    file: gaugeFile,
    cls: "pulse-mem-gauge",
    id: "pulse-data-controller-memory",
    data: memoryData,
  },
};

// As much of an element as the scripts use.
class Element {
  constructor(tag) {
    Object.assign(this, { tagName: tag, attributes: {}, children: [], own: "" });
    Object.assign(this, { style: {}, dataset: {}, hidden: false, listeners: {} });
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }
  getAttribute(name) {
    return name in this.attributes ? this.attributes[name] : null;
  }
  removeAttribute(name) {
    delete this.attributes[name];
  }
  get className() {
    return this.attributes.class || "";
  }
  set className(value) {
    this.attributes.class = value;
  }
  get classList() {
    const names = () => this.className.split(" ").filter(Boolean);
    const set = (name, on) => {
      const others = names().filter((n) => n !== name);
      this.className = (on ? [...others, name] : others).join(" ");
    };
    const contains = (name) => names().includes(name);
    return { contains, add: (name) => set(name, true), toggle: set };
  }
  append(...nodes) {
    for (let node of nodes) {
      if (typeof node === "string") node = Object.assign(new Element("#text"), { own: node });
      node.parentElement = this;
      this.children.push(node);
    }
  }
  appendChild(node) {
    this.append(node);
    return node;
  }
  replaceChildren(...nodes) {
    this.textContent = "";
    this.append(...nodes);
  }
  get textContent() {
    return this.own + this.children.map((c) => c.textContent).join("");
  }
  set textContent(value) {
    this.children = [];
    this.own = String(value);
  }
  addEventListener(type, listener) {
    (this.listeners[type] = this.listeners[type] || []).push(listener);
  }
  all(test, found = []) {
    for (const child of this.children) {
      if (test(child)) found.push(child);
      child.all(test, found);
    }
    return found;
  }
  // A selector is a class or a tag name.
  querySelectorAll(selector) {
    if (selector[0] === ".") return this.all((e) => e.classList.contains(selector.slice(1)));
    return this.all((e) => e.tagName === selector);
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
}

// A page with one figure on it, drawn by its script. *calm* is whether the
// reader asks for reduced motion. The page's frames, its window and its
// settings are the test's to move.
function open(kind, dataset, calm) {
  const { file, cls, id, data } = FIGURES[kind];
  const body = new Element("body");
  const page = { calm, hidden: false, frames: new Map(), ids: 0, motion: [], tab: [] };
  const json = Object.assign(new Element("script"), { own: fs.readFileSync(data, "utf8") });
  json.setAttribute("id", id);
  const root = Object.assign(new Element("div"), { dataset: { source: id, ...dataset } });
  root.className = cls;
  const fallback = new Element("div");
  fallback.className = "pulse-fallback";
  const figure = new Element("figure");
  figure.append(root, fallback);
  body.append(json, figure);
  global.document = {
    readyState: "complete",
    get hidden() {
      return page.hidden;
    },
    createElement: (tag) => new Element(tag),
    createElementNS: (_, tag) => new Element(tag),
    getElementById: (name) => body.all((e) => e.getAttribute("id") === name)[0],
    querySelectorAll: (selector) => body.querySelectorAll(selector),
    addEventListener: (type, listener) => type === "visibilitychange" && page.tab.push(listener),
  };
  global.window = {
    matchMedia: (query) => ({
      get matches() {
        return query === "(prefers-reduced-motion: reduce)" && page.calm;
      },
      addEventListener: (type, listener) => type === "change" && page.motion.push(listener),
    }),
  };
  global.requestAnimationFrame = (callback) => {
    page.frames.set(++page.ids, callback);
    return page.ids;
  };
  global.cancelAnimationFrame = (frame) => page.frames.delete(frame);
  global.IntersectionObserver = class {
    constructor(callback, options) {
      page.observer = Object.assign(this, { callback, options, targets: [] });
    }
    observe(target) {
      this.targets.push(target);
    }
  };
  delete require.cache[require.resolve(file)];
  const api = require(file);
  const bar = root.children[root.children.length - 1];
  return Object.assign(page, { api, root, bar, fallback });
}

// The browser draws a frame at *time* ms.
function frame(page, time) {
  const due = [...page.frames.values()];
  page.frames.clear();
  for (const callback of due) callback(time);
}

// The reader scrolls: the share of the figure, and of its controls, in the window.
function scroll(page, figure, controls) {
  const entry = (target, intersectionRatio) => {
    return { target, intersectionRatio, isIntersecting: intersectionRatio > 0 };
  };
  page.observer.callback([entry(page.root, figure), entry(page.bar, controls)]);
}

function setting(page, calm) {
  page.calm = calm;
  for (const listener of page.motion) listener();
}

function tab(page, hidden) {
  page.hidden = hidden;
  for (const listener of page.tab) listener();
}

function press(page, name) {
  page.bar.children.find((b) => b.textContent === name && !b.hidden).onclick();
}

// A key pressed with focus on a control; whether the figure took it.
function key(page, name, modifier) {
  const event = { key: name, altKey: false, ctrlKey: false, metaKey: false, shiftKey: false };
  if (modifier) event[modifier] = true;
  event.preventDefault = () => (event.taken = true);
  for (const listener of page.bar.listeners.keydown || []) listener(event);
  return Boolean(event.taken);
}

function controls(page) {
  const shown = page.bar.hidden ? [] : page.bar.children.filter((b) => !b.hidden);
  const off = page.bar.children.filter((b) => b.getAttribute("aria-disabled") === "true");
  return {
    buttons: shown.map((b) => b.textContent),
    unavailable: off.map((b) => b.textContent),
    frames: page.frames.size,
    role: page.root.getAttribute("role"),
    label: page.root.getAttribute("aria-label"),
  };
}

// What a reader of a toy-flow figure sees and hears.
function flowNow(page) {
  const [svg, status] = page.root.children;
  const texts = svg.querySelectorAll("text").map((t) => t.textContent);
  const wires = svg.querySelectorAll(".tf-wire");
  return {
    ...controls(page),
    playing: page.root.classList.contains("tf-playing"),
    status: status.textContent,
    unseen: status.querySelectorAll(".visually-hidden").map((e) => e.textContent).join(""),
    live: status.getAttribute("aria-live"),
    header: texts[0],
    cycle: Number(texts[1].split(" ")[1]),
    liveWires: wires.filter((w) => w.classList.contains("tf-on")).length,
  };
}

function flowMarkup(scene) {
  const page = open("flow", { scene: scene.name }, false);
  const [svg, status, bar] = page.root.children;
  const named = (attribute) =>
    svg.all((e) => e.getAttribute("id") === svg.getAttribute(attribute))[0].textContent;
  return {
    children: page.root.children.map((c) => c.tagName),
    tabindex: page.root.getAttribute("tabindex"),
    picture: [svg.getAttribute("role"), named("aria-labelledby"), named("aria-describedby")],
    status: [status.className, status.getAttribute("role")],
    bar: bar.className,
    buttons: bar.children.map((b) => [b.tagName, b.type, b.textContent]),
    fallbackHidden: page.fallback.hidden,
    watched: [page.observer.targets.length, page.observer.options.threshold],
  };
}

// The figure plays by itself: when it starts, stops and goes on.
function flowAutoplay(scene) {
  const page = open("flow", { scene: scene.name }, false);
  const { SHOT_MS, elapsedAt } = page.api;
  const seen = { mounted: flowNow(page) };
  scroll(page, 0.4, 1);
  seen.underHalf = flowNow(page);
  scroll(page, 0.6, 0);
  seen.controlsBelow = flowNow(page);
  scroll(page, 0.6, 1);
  seen.inView = flowNow(page);
  frame(page, 1000);
  frame(page, 1000 + SHOT_MS / 2);
  seen.halfway = flowNow(page);
  scroll(page, 0.3, 0);
  seen.left = flowNow(page);
  scroll(page, 0.4, 1);
  seen.underHalfAgain = flowNow(page);
  scroll(page, 0.7, 1);
  frame(page, 50000);
  frame(page, 50000 + SHOT_MS / 4);
  seen.back = flowNow(page);
  tab(page, true);
  seen.tabHidden = flowNow(page);
  tab(page, false);
  seen.tabShown = flowNow(page);
  frame(page, 90000);
  frame(page, 90000 + elapsedAt(scene, scene.shots.length - 1, scene.end_cycles));
  seen.ended = flowNow(page);
  scroll(page, 0, 0);
  scroll(page, 0.9, 1);
  seen.seenAgain = flowNow(page);
  return seen;
}

// The reader's buttons and keys.
function flowControls(scene) {
  const page = open("flow", { scene: scene.name }, false);
  const { SHOT_MS, steps } = page.api;
  const seen = {};
  scroll(page, 1, 1);
  frame(page, 0);
  frame(page, SHOT_MS / 2);
  press(page, "Pause");
  seen.paused = flowNow(page);
  scroll(page, 0, 0);
  scroll(page, 1, 1);
  seen.pausedSeenAgain = flowNow(page);
  press(page, "Play");
  seen.resumed = flowNow(page);
  frame(page, 7000);
  frame(page, 7001);
  seen.resumedFrame = flowNow(page);
  scroll(page, 0, 0);
  seen.resumedLeft = flowNow(page);
  scroll(page, 1, 1);
  seen.resumedSeenAgain = flowNow(page);
  press(page, "Next step");
  seen.next = flowNow(page);
  for (let k = 0; k < steps(scene).length; k++) press(page, "Previous step");
  seen.first = flowNow(page);
  seen.keys = [["ArrowRight"], ["ArrowLeft", "altKey"], ["ArrowRight", "shiftKey"]]
    .concat([["ArrowRight", "ctrlKey"], ["ArrowLeft", "metaKey"], ["Home"], ["ArrowLeft"]])
    .map(([name, modifier]) => [key(page, name, modifier), flowNow(page).status]);
  seen.walk = steps(scene).slice(1).map(() => {
    press(page, "Next step");
    return flowNow(page);
  });
  press(page, "Next step");
  seen.pastLast = flowNow(page);
  press(page, "Replay");
  seen.replaying = flowNow(page);
  frame(page, 20000);
  frame(page, 20000 + SHOT_MS / 2);
  seen.replayedHalfway = flowNow(page);
  press(page, "Next step");
  seen.steppedOut = flowNow(page);
  return seen;
}

// A step taken before the figure has played, a step button pressed at its
// end, and the hook for screenshots.
function flowEarly(scene) {
  let page = open("flow", { scene: scene.name }, false);
  press(page, "Next step");
  scroll(page, 1, 1);
  const seen = { steppedThenSeen: flowNow(page) };
  page = open("flow", { scene: scene.name }, false);
  scroll(page, 1, 1);
  frame(page, 0);
  frame(page, 10);
  press(page, "Previous step");
  seen.endPressed = flowNow(page);
  page = open("flow", { scene: scene.name }, false);
  scroll(page, 1, 1);
  page.root.toyFlowShow(scene.shots.length - 1, scene.end_cycles / 2);
  seen.shown = flowNow(page);
  scroll(page, 0, 0);
  scroll(page, 1, 1);
  seen.shownSeenAgain = flowNow(page);
  return seen;
}

// Reduced motion, from the start and when the setting changes.
function flowCalm(scene) {
  let page = open("flow", { scene: scene.name }, true);
  const seen = { mounted: flowNow(page) };
  scroll(page, 1, 1);
  seen.inView = flowNow(page);
  press(page, "Next step");
  seen.next = flowNow(page);
  setting(page, false);
  seen.allowedAfterUse = flowNow(page);
  page = open("flow", { scene: scene.name }, true);
  scroll(page, 1, 1);
  setting(page, false);
  seen.allowed = flowNow(page);
  press(page, "Play");
  frame(page, 0);
  frame(page, 100);
  seen.playing = flowNow(page);
  setting(page, true);
  seen.calmAgain = flowNow(page);
  return seen;
}

// What a reader of the gauges sees: how far each bar has filled, and how
// many of the marks on the bars (entry seams, break marks) are not yet shown.
function gaugeNow(page) {
  const svg = page.root.children[0];
  const marks = [...svg.querySelectorAll(".mg-seams"), ...svg.querySelectorAll(".mg-break")];
  return {
    ...controls(page),
    fills: svg.querySelectorAll(".mg-fill").map((e) => Number(e.getAttribute("width"))),
    marks: [marks.filter((e) => e.style.opacity === 0).length, marks.length],
  };
}

function gaugePlay(rows, columns) {
  let page = open("gauge", { rows, columns }, false);
  const total = rows.split(" ").length * page.api.ROW_MS;
  const svg = page.root.children[0];
  const seen = { mounted: gaugeNow(page) };
  seen.markup = {
    children: page.root.children.map((c) => c.tagName),
    picture: [svg.getAttribute("role"), svg.getAttribute("aria-labelledby").split(" ").length],
    bar: page.bar.className,
    buttons: page.bar.children.map((b) => [b.tagName, b.type]),
    fallbackHidden: page.fallback.hidden,
    watched: [page.observer.targets.length, page.observer.options.threshold],
  };
  scroll(page, 0.4, 1);
  seen.underHalf = gaugeNow(page);
  scroll(page, 0.6, 0);
  seen.controlsBelow = gaugeNow(page);
  scroll(page, 0.6, 1);
  seen.inView = gaugeNow(page);
  frame(page, 1000);
  frame(page, 1000 + total / 2);
  seen.halfway = gaugeNow(page);
  scroll(page, 0.3, 0);
  seen.left = gaugeNow(page);
  scroll(page, 1, 1);
  frame(page, 9000);
  frame(page, 9000 + total / 4);
  seen.back = gaugeNow(page);
  tab(page, true);
  seen.tabHidden = gaugeNow(page);
  tab(page, false);
  seen.tabShown = gaugeNow(page);
  press(page, "Pause");
  seen.paused = gaugeNow(page);
  scroll(page, 0, 0);
  scroll(page, 1, 1);
  seen.pausedSeenAgain = gaugeNow(page);
  press(page, "Play");
  frame(page, 20000);
  frame(page, 20000 + total);
  seen.ended = gaugeNow(page);
  press(page, "Replay");
  seen.replaying = gaugeNow(page);
  page.root.memGaugeShow(total);
  seen.shown = gaugeNow(page);

  page = open("gauge", { rows, columns }, true);
  scroll(page, 1, 1);
  seen.calm = gaugeNow(page);
  setting(page, false);
  seen.allowed = gaugeNow(page);
  setting(page, true);
  seen.calmAgain = gaugeNow(page);
  return seen;
}

const out = { flow: {}, gauge: {} };
for (const scene of JSON.parse(fs.readFileSync(flowData, "utf8")).scenes) {
  out.flow[scene.name] = {
    markup: flowMarkup(scene),
    autoplay: flowAutoplay(scene),
    controls: flowControls(scene),
    early: flowEarly(scene),
    calm: flowCalm(scene),
  };
}
for (const figure of gauges) {
  const [rows, columns] = figure.split("/");
  out.gauge[figure] = gaugePlay(rows, columns);
}
process.stdout.write(JSON.stringify(out));
"""


def _gauges() -> list[str]:
    """The rows and columns of each gauge figure of the page, as "rows/columns"."""
    lines = _PAGE.read_text().splitlines()
    found = []
    for i, line in enumerate(lines):
        if line.startswith(".. memory-gauge::"):
            options = dict(
                option.strip()[1:].split(":", 1)
                for option in lines[i + 1 : i + 3]
                if option.strip().startswith(":")
            )
            found.append(f"{options['rows'].strip()}/{options['columns'].strip()}")
    return found


@pytest.fixture(scope="module")
def played():
    return _node(_PLAY, *_gauges())


def _steps(scene: dict) -> list[tuple[int, int]]:
    """(outcome, cycle) of each step of *scene*: each outcome's first frame,
    its moments and its last frame."""
    end = scene["end_cycles"]
    return [
        (s, cycle)
        for s, shot in enumerate(scene["shots"])
        for cycle in sorted({0, end, *(cycle for cycle, _ in shot["moments"])})
    ]


def _status(scene: dict, k: int) -> tuple[str, str]:
    """What the status line says at step *k*, and the part of it that only a
    screen reader gets: the outcome, which the picture shows in its header."""
    steps = _steps(scene)
    s, cycle = steps[k]
    shot = scene["shots"][s]
    said = [line for at, line in shot["moments"] if at <= cycle]
    sentence = said[-1] if said else ""
    return f"Step {k + 1} of {len(steps)}: {shot['label']} {sentence}", f"{shot['label']} "


def _flows(played):
    for scene in _load("toy_flow.json")["scenes"]:
        yield scene, played["flow"][scene["name"]]


def _states(seen: dict):
    """Each state recorded in *seen*, alone or in a list."""
    for value in seen.values():
        for state in value if isinstance(value, list) else [value]:
            if isinstance(state, dict) and "frames" in state:
                yield state


def test_a_reader_steps_through_each_moment_of_each_outcome(scripts):
    for scene, drawn in _scenes(scripts):
        steps = _steps(scene)
        assert [tuple(step) for step in drawn["steps"]] == steps
        for k, (s, cycle) in enumerate(steps):
            text = drawn["stepTexts"][k]
            assert f"{text['step']}: {text['outcome']} {text['says']}" == _status(scene, k)[0]
            assert text["outcome"] == scene["shots"][s]["label"]
            # Playing picks up from a step at the step's own frame, and a
            # frame belongs to the last step at or before it.
            frame, at, later = drawn["resumes"][k]
            assert frame["shot"] == s and frame["t"] == pytest.approx(cycle)
            assert at == later == k


def test_each_figure_is_a_named_group_of_a_picture_a_status_line_and_buttons(played):
    for scene, seen in _flows(played):
        markup = seen["markup"]
        assert markup["children"] == ["svg", "p", "div"]
        # The picture is named by its title; what it plays is its description.
        role, name, description = markup["picture"]
        assert (role, name) == ("img", scene["title"])
        assert description.startswith(scene["title"])
        assert all(shot["label"] in description for shot in scene["shots"])
        assert markup["status"] == ["pulse-toy-flow-status", "status"]
        assert markup["bar"] == "pulse-toy-flow-controls"
        assert markup["buttons"] == [
            ["button", "button", name] for name in ("Play", "Previous step", "Next step")
        ]
        # The group is not a Tab stop, and the fallback gives way to the drawing.
        assert markup["tabindex"] is None and markup["fallbackHidden"]
        # Half of the figure and half of its controls must be in the window.
        assert markup["watched"] == [2, 0.5]
        for part in ("autoplay", "controls", "early", "calm"):
            for now in _states(seen[part]):
                assert (now["role"], now["label"]) == ("group", scene["title"])


def test_a_figure_plays_by_itself_once_in_the_window_and_in_silence(played):
    for scene, seen in _flows(played):
        seen = seen["autoplay"]
        end = scene["end_cycles"]
        # It waits for half of it, and for its controls, to be in the window.
        for moment in ("mounted", "underHalf", "controlsBelow"):
            now = seen[moment]
            assert (now["frames"], now["buttons"][0], now["cycle"]) == (0, "Play", 0), moment
        assert (seen["inView"]["frames"], seen["inView"]["buttons"][0]) == (1, "Pause")
        assert seen["halfway"]["cycle"] == end // 2
        # With under half of it in the window, or in a hidden tab, it stops
        # where it is, and goes on from there once half of it is back.
        left = seen["left"]
        assert (left["frames"], left["buttons"][0], left["cycle"]) == (0, "Play", end // 2)
        assert seen["underHalfAgain"] == left
        assert (seen["back"]["frames"], seen["back"]["cycle"]) == (1, end * 3 // 4)
        assert (seen["tabHidden"]["frames"], seen["tabShown"]["frames"]) == (0, 1)
        # It ends on the last outcome's last frame, and does not start again.
        ended = seen["ended"]
        assert (ended["frames"], ended["buttons"][0], ended["cycle"]) == (0, "Replay", end)
        assert ended["header"] == scene["shots"][-1]["label"]
        assert ended["status"] == _status(scene, len(_steps(scene)) - 1)[0]
        assert ended["unavailable"] == ["Next step"]
        assert seen["seenAgain"] == ended
        # The status line follows the picture, and never speaks by itself.
        assert {now["live"] for now in seen.values()} == {"off"}


def test_the_buttons_and_arrow_keys_pause_step_and_replay(played):
    for scene, seen in _flows(played):
        seen = seen["controls"]
        end = scene["end_cycles"]
        steps = _steps(scene)
        said = [_status(scene, k)[0] for k in range(len(steps))]
        at = max(k for k, step in enumerate(steps) if step <= (0, end / 2))
        # Pause keeps the frame, and the status line says which step it is:
        # a status speaks when it has no aria-live of its own.
        paused = seen["paused"]
        assert (paused["frames"], paused["buttons"][0], paused["cycle"]) == (0, "Play", end // 2)
        assert (paused["status"], paused["live"]) == (said[at], None)
        # Once a control is used, the figure never plays by itself again.
        assert seen["pausedSeenAgain"] == paused
        # Play goes on from that frame, in silence. Out of the window it
        # stops, and then stays stopped.
        resumed = seen["resumed"]
        assert (resumed["frames"], resumed["buttons"][0], resumed["live"]) == (1, "Pause", "off")
        assert seen["resumedFrame"]["cycle"] == end // 2
        assert (seen["resumedLeft"]["frames"], seen["resumedSeenAgain"]["frames"]) == (0, 0)
        # A step button stops the playing and shows the step's own frame.
        after = seen["next"]
        assert (after["frames"], after["cycle"], after["live"]) == (0, steps[at + 1][1], None)
        assert after["status"] == said[at + 1]
        first = seen["first"]
        assert (first["status"], first["cycle"]) == (said[0], 0)
        assert first["unavailable"] == ["Previous step"]
        # Left and Right step; with a modifier they are left to the browser.
        assert [taken for taken, _ in seen["keys"]] == [True] + [False] * 5 + [True]
        assert [status for _, status in seen["keys"]] == [said[1]] * 6 + [said[0]]
        # Each step shows its frame and says its number, outcome and sentence;
        # the outcome is in the picture, so the line holds it out of sight.
        for k, now in enumerate(seen["walk"], start=1):
            s, cycle = steps[k]
            assert (now["status"], now["unseen"]) == _status(scene, k)
            assert (now["header"], now["cycle"]) == (scene["shots"][s]["label"], cycle)
            assert (now["frames"], now["live"]) == (0, None)
            assert now["unavailable"] == (["Next step"] if k == len(steps) - 1 else [])
        last = seen["walk"][-1]
        assert last["buttons"][0] == "Replay" and seen["pastLast"] == last
        # Replay starts over from the first outcome.
        again = seen["replaying"]
        assert (again["frames"], again["buttons"][0], again["cycle"]) == (1, "Pause", 0)
        assert again["header"] == scene["shots"][0]["label"]
        assert seen["replayedHalfway"]["cycle"] == end // 2
        assert (seen["steppedOut"]["frames"], seen["steppedOut"]["buttons"][0]) == (0, "Play")


def test_a_control_used_first_keeps_the_figure_still(played):
    for scene, seen in _flows(played):
        seen = seen["early"]
        # A step taken before the figure is in the window.
        stepped = seen["steppedThenSeen"]
        assert (stepped["frames"], stepped["status"]) == (0, _status(scene, 1)[0])
        # A step button at its end stops the playing and says nothing.
        pressed = seen["endPressed"]
        assert (pressed["frames"], pressed["buttons"][0], pressed["live"]) == (0, "Play", "off")
        assert pressed["status"] == _status(scene, 0)[0]
        # The hook that draws one moment for a screenshot.
        shown = seen["shown"]
        assert (shown["frames"], shown["cycle"]) == (0, scene["end_cycles"] // 2)
        assert shown["header"] == scene["shots"][-1]["label"]
        assert seen["shownSeenAgain"] == shown


def test_reduced_motion_shows_a_still_frame_and_follows_the_setting(played):
    for scene, seen in _flows(played):
        seen = seen["calm"]
        end = scene["end_cycles"]
        rest = _steps(scene).index((0, end))
        # The first outcome's last frame, the step buttons, and no Play.
        mounted = seen["mounted"]
        assert (mounted["frames"], mounted["cycle"]) == (0, end)
        assert mounted["buttons"] == ["Previous step", "Next step"]
        assert mounted["status"] == _status(scene, rest)[0]
        assert seen["inView"] == mounted
        after = seen["next"]
        assert (after["frames"], after["live"]) == (0, None)
        assert after["status"] == _status(scene, rest + 1)[0]
        # When motion is allowed again Play is back, and nothing moves until
        # the reader presses it: the still frame stood in for the playing.
        for moment in ("allowedAfterUse", "allowed"):
            now = seen[moment]
            assert (now["frames"], now["buttons"][0]) == (0, "Play"), moment
        assert seen["allowed"] == {**mounted, "buttons": ["Play", *mounted["buttons"]]}
        # Play goes on from the still frame. That is an outcome's last frame,
        # so the next outcome starts at once, with no hold before it.
        assert (seen["playing"]["frames"], seen["playing"]["buttons"][0]) == (1, "Pause")
        assert seen["playing"]["header"] == scene["shots"][1]["label"]
        assert 0 < seen["playing"]["cycle"] < end // 10
        # Asked for again while the figure plays, the still frame returns.
        assert seen["calmAgain"] == mounted


def _rules(css: str) -> list[tuple[str, str, str]]:
    """(media query, selector, declarations) of each rule of *css*; the query
    is "" for a rule outside a media block."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules = []

    def plain(block: str, media: str) -> None:
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", block):
            rules.append((media, " ".join(selector.split()), body))

    at = 0
    for block in re.finditer(r"@(\w+)([^{]*)\{((?:[^{}]*\{[^{}]*\})*[^{}]*)\}", css):
        plain(css[at : block.start()], "")
        if block[1] == "media":
            plain(block[3], block[2].strip())
        at = block.end()
    plain(css[at:], "")
    return rules


def test_nothing_moves_without_end_unless_the_figure_plays(played):
    """The live wires' dashes are the figures' only endless animation. The
    stylesheet runs it under ``tf-playing`` alone and never with reduced
    motion, and the script sets that class exactly while it draws frames."""
    endless = {}
    for name in ("toy-flow.css", "mem-gauge.css"):
        rules = _rules((_STATIC / name).read_text())
        for media, selector, body in rules:
            if "infinite" in body:
                assert media == "" and selector.startswith(".pulse-toy-flow.tf-playing "), selector
                calm = [
                    declarations
                    for query, same, declarations in rules
                    if "prefers-reduced-motion: reduce" in query and same == selector
                ]
                assert calm and "animation: none" in calm[0], selector
                endless[name] = selector
    assert endless == {"toy-flow.css": ".pulse-toy-flow.tf-playing .tf-wire.tf-on"}
    states = [
        now
        for _, seen in _flows(played)
        for part in ("autoplay", "controls", "early", "calm")
        for now in _states(seen[part])
    ]
    assert all(now["playing"] == (now["frames"] == 1) for now in states)
    # A wire may be live on a frame at rest: thick and colored, not moving.
    assert any(now["liveWires"] and not now["playing"] for now in states)


def test_the_gauges_fill_by_themselves_in_the_window_and_one_button_plays_them(played, scripts):
    memory = _load("controller_memory.json")
    titles = {key: column["title"] for key, column in scripts["columns"].items()}
    inside = {
        row["name"]: {key: gauge["inside"] for key, gauge, _ in row["gauges"]}
        for row in scripts["rows"]
    }
    assert list(played["gauge"]) == _gauges() and len(_gauges()) == 2
    for figure, seen in played["gauge"].items():
        rows, columns = (part.split() for part in figure.split("/"))
        assert seen["markup"] == {
            "children": ["svg", "div"],
            "picture": ["img", 2],
            "bar": "pulse-mem-gauge-controls",
            "buttons": [["button", "button"]],
            "fallbackHidden": True,
            "watched": [2, 0.5],
        }
        # Each bar ends at its share of its track, up to the limit mark.
        full = seen["ended"]["fills"]
        shares = [inside[row][column] for row in rows for column in columns]
        assert len({round(fill / share, 6) for fill, share in zip(full, shares, strict=True)}) == 1
        empty = [0] * len(full)
        for moment in ("mounted", "underHalf", "controlsBelow"):
            now = seen[moment]
            assert (now["buttons"], now["frames"], now["fills"]) == (["Play"], 0, empty), moment
        # An empty gauge shows none of its seams and break marks; a full one does.
        unshown, marks = seen["mounted"]["marks"]
        assert unshown == marks > seen["ended"]["marks"][0]
        assert (seen["inView"]["buttons"], seen["inView"]["frames"]) == (["Pause"], 1)
        # The rows fill in turn: halfway the first has begun and the last has not.
        halfway = seen["halfway"]["fills"]
        assert all(halfway[: len(columns)]) and not any(halfway[-len(columns) :])
        # With under half of the figure in the window the filling stops
        # where it is, and goes on when the figure is back.
        left = seen["left"]
        assert (left["buttons"], left["frames"], left["fills"]) == (["Play"], 0, halfway)
        back = seen["back"]
        assert back["frames"] == 1 and back["fills"] != halfway
        assert all(now >= then for now, then in zip(back["fills"], halfway, strict=True))
        assert (seen["tabHidden"]["frames"], seen["tabShown"]["frames"]) == (0, 1)
        # The button pauses, and then the gauges never fill by themselves again.
        paused = seen["paused"]
        assert (paused["buttons"], paused["frames"]) == (["Play"], 0)
        assert paused["fills"] == back["fills"]
        assert seen["pausedSeenAgain"] == paused
        assert (seen["ended"]["buttons"], seen["ended"]["frames"]) == (["Replay"], 0)
        again = seen["replaying"]
        assert (again["buttons"], again["frames"], again["fills"]) == (["Pause"], 1, empty)
        # With its button, a gauge is a group named after its columns.
        label = " and ".join(titles[column] for column in columns) + f" of a {memory['model']}"
        group = {"role": "group", "label": label.capitalize(), "unavailable": []}
        for moment, now in seen.items():
            if moment not in ("markup", "calm", "calmAgain"):
                assert now == {**now, **group}, moment
        # Reduced motion: full gauges and no button, so no group either. When
        # motion is allowed again the button is back, and the gauges stay full.
        still = {"frames": 0, "fills": full, "marks": seen["ended"]["marks"], "unavailable": []}
        for moment in ("calm", "calmAgain"):
            assert seen[moment] == {**still, "buttons": [], "role": None, "label": None}
        for moment in ("allowed", "shown"):
            assert seen[moment] == {**still, **group, "buttons": ["Replay"]}, moment


# ── The directives ────────────────────────────────────────────────────


class _Page:
    """What a directive asks of the build about the page it is on."""

    docname = "tutorials/isa_limits"

    def __init__(self):
        self.temp_data = {}

    def relfn2path(self, name: str, docname: str) -> tuple[str, str]:
        return f"tutorials/{name}", str(_PAGE.parent / name)

    def note_dependency(self, name: str) -> None:
        pass


def test_a_figure_with_no_still_holds_what_it_plays_as_text():
    """Without JavaScript the script draws nothing. A toy-flow with no
    ``:fallback:`` then shows each outcome and its moments, in order."""
    pytest.importorskip("docutils")
    from docutils.core import publish_parts
    from docutils.parsers.rst import directives
    from docutils.writers import html5_polyglot

    sys.path.insert(0, str(_EXT))
    try:
        import data_figures
    finally:
        sys.path.remove(str(_EXT))

    for scene in _load("toy_flow.json")["scenes"]:
        story = data_figures.story(scene)
        assert story.startswith('<div class="pulse-fallback tf-story">')
        assert f"<strong>{html.escape(scene['title'], quote=False)}</strong>" in story
        said = [html.unescape(line) for line in re.findall(r"<(?:p|li)>(.*?)</(?:p|li)>", story)]
        expected = []
        for shot in scene["shots"]:
            expected += [shot["label"], *(line for _, line in shot["moments"])]
        assert said[1:] == expected

    source = "\n".join(
        [
            ".. toy-flow:: figures/toy_flow.json",
            "   :scene: flip_other",
            "   :fallback: figures/isa_feedback.svg",
            "   :alt: A timeline.",
            "",
            "   A caption.",
            "",
            ".. memory-gauge:: figures/controller_memory.json",
            "   :rows: train train_loop",
            "   :columns: pulses samples",
            "",
            ".. toy-flow:: figures/toy_flow.json",
            "   :scene: correct_across",
        ]
    )
    added = {"toy-flow": data_figures.ToyFlow, "memory-gauge": data_figures.MemoryGauge}
    for name, directive in added.items():
        directives.register_directive(name, directive)
    try:
        overrides = {"env": _Page(), "report_level": 5}
        writer = html5_polyglot.Writer()
        body = publish_parts(source, writer=writer, settings_overrides=overrides)["body"]
    finally:
        for name in added:
            directives._directives.pop(name)
    with_still, gauge, with_text = re.findall(r"<figure\b.*?</figure>", body, flags=re.S)
    # Each file's data goes into the page once, before its first figure.
    assert [figure.count("<script") for figure in (with_still, gauge, with_text)] == [1, 1, 0]
    assert "isa_feedback.svg" in with_still and "tf-story" not in with_still
    assert 'class="pulse-fallback mg-table"' in gauge
    # The search index reads the table with its tags taken out: each cell's
    # words must still stand apart from the next cell's.
    table = gauge[gauge.index("<table") :]
    assert re.sub(r"<[^>]+>", "", table).split() == re.sub(r"<[^>]+>", " ", table).split()
    # The text follows the script's root, so the script and paper find it.
    scene = next(s for s in _load("toy_flow.json")["scenes"] if s["name"] == "correct_across")
    root = '<div class="pulse-toy-flow" data-source="pulse-data-toy-flow"'
    assert f'{root} data-scene="correct_across"></div>\n{data_figures.story(scene)}' in with_text
    # A figure with no caption has no caption's end tag.
    for figure in (with_still, gauge, with_text):
        assert figure.count("<figcaption") == figure.count("</figcaption>")
    assert "<figcaption" in with_still and "<figcaption" not in gauge + with_text
