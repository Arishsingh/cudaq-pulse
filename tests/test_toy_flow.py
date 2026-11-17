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
pulse, bit, call and sentence of the data at its cycle. Node is optional
unless ``CUDAQ_PULSE_REQUIRE_NODE=1`` (set in the docs workflow).
To update the data, run ``python docs/tutorials/figures/isa_limits.py``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIGURES = _ROOT / "docs" / "tutorials" / "figures"
_PAGE = _ROOT / "docs" / "tutorials" / "isa_limits.rst"
_FLOW = _ROOT / "docs" / "_static" / "toy-flow.js"
_GAUGE = _ROOT / "docs" / "_static" / "mem-gauge.js"
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


@pytest.fixture(scope="module")
def scripts():
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CUDAQ_PULSE_REQUIRE_NODE"):
            pytest.fail("CUDAQ_PULSE_REQUIRE_NODE is set, but node is not on PATH")
        pytest.skip("needs Node.js to run the animation's scripts")
    files = [_FLOW, _GAUGE, _FIGURES / _DATA[0], _FIGURES / _DATA[1]]
    result = subprocess.run(
        [node, "-e", _DUMP, *map(str, files)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout)


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
