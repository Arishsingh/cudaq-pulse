# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tutorials' animated figures play what their data says, and stay small.

The data (``docs/tutorials/figures/<page>.json``) is what each page's figure
script writes from the page's own code; tests/test_tutorial_figures.py checks
that it is still what the scripts write. Here:

- the shared script, ``docs/_static/pulse-visual.js``, stays within its
  budget of 12 KB gzipped;
- its model, run under Node on every committed scene, plays each span in
  exactly the cases its conditions allow, shows each sentence from its
  moment, plays the first two cases and ends on its last frame, and moves
  each span from one snapshot to the next;
- ``tools/render-pulse-visual.js`` draws the committed stills byte for byte;
- each scene says what its page prints.

Node is optional unless CUDAQ_PULSE_REQUIRE_NODE=1 (set in the docs workflow).
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ENGINE = _ROOT / "docs" / "_static" / "pulse-visual.js"
_RENDER = _ROOT / "tools" / "render-pulse-visual.js"
_FIGURES = _ROOT / "docs" / "tutorials" / "figures"
_BUDGET_GZ = 12 * 1024
FORMAT = "pulse-visual/1"

pytestmark = pytest.mark.skipif(not _ENGINE.exists(), reason="needs the source tree")


def _files() -> list[Path]:
    """The animated figures' data. Other in-page scripts may keep JSON of their
    own format in the same folder."""
    return sorted(
        p for p in _FIGURES.glob("*.json") if json.loads(p.read_text()).get("format") == FORMAT
    )


def _scenes():
    for path in _files():
        for scene in json.loads(path.read_text())["scenes"]:
            yield path, scene


def _scene(page: str, name: str) -> dict:
    data = json.loads((_FIGURES / f"{page}.json").read_text())
    return next(s for s in data["scenes"] if s["name"] == name)


def test_the_engine_fits_its_budget():
    size = len(gzip.compress(_ENGINE.read_bytes(), compresslevel=9))
    assert size <= _BUDGET_GZ, f"pulse-visual.js is {size} bytes gzipped, over {_BUDGET_GZ}"


def test_every_scene_has_a_still_in_both_themes():
    for path in _files():
        for scene in json.loads(path.read_text())["scenes"]:
            for suffix in ("", "-dark"):
                assert (_FIGURES / f"{scene['name']}{suffix}.svg").exists(), (
                    f"{path.name}: no still for {scene['name']}; run node {_RENDER.name}"
                )


# ── The pages' own numbers ────────────────────────────────────────────


def test_the_t_state_scenes_say_what_the_t_state_page_prints():
    tomography = _scene("t_state_injection", "tomography_axes")
    # "fidelity with |T>: 1.00": the axes the steps read make the page's Bloch vector.
    x, y, z = (2 * step["weight"] - 1 for step in tomography["snapshots"])
    assert (round(x, 2), round(y, 2), round(z, 2)) == (0.71, 0.71, 0.01)
    turns = [step["spans"]["turn"] for step in tomography["snapshots"]]
    assert [turn is None for turn in turns] == [False, False, True]

    loop = _scene("t_state_injection", "herald_loop")
    # "herald  300.0  352.0  0": the answer is due 52 ns after the outcome is
    # sent, and the pi pulse that reads it starts then.
    first = loop["calls"][0]
    assert (first["submit"], first["due"]) == (300.0, 352.0)
    flips = [s for s in loop["spans"] if s.get("when")]
    assert [s["a"] for s in flips] == [call["due"] for call in loop["calls"]]
    # "1  0.972" and "2  0.999": the shots kept after one and two attempts.
    once = sum(c["weight"] for c in loop["cases"] if c["given"]["accepted 1"])
    twice = sum(c["weight"] for c in loop["cases"] if c["given"]["accepted 2"])
    assert (round(once, 3), round(twice, 3)) == (0.972, 0.999)
    assert sum(c["weight"] for c in loop["cases"]) == pytest.approx(1, abs=1e-6)

    tree = {n["id"]: n for n in _scene("t_state_injection", "herald_branches")["tree"]["nodes"]}
    # "shots kept: 0.972", split about equally by the Z readout.
    assert round(tree["h0"]["p"], 3) == 0.972
    assert tree["z0"]["p"] + tree["z1"]["p"] == pytest.approx(tree["h0"]["p"], abs=1e-3)
    assert {n for n, node in tree.items() if "drop" in node} == {"h1", "h2"}


def test_the_estimator_scene_says_what_the_adaptive_page_prints():
    prior, read_1, read_0 = _scene("adaptive_calibration", "t1_belief")["snapshots"]
    # "a first guess of T1 = 150 µs", and the likeliest branch's first shot,
    # which reads 0 after the 50 µs wait: "127.7 µs  65.1 µs".
    assert round(prior["curve"]["at"], 1) == 150.0
    assert round(read_0["curve"]["at"], 1) == 127.7
    assert "next wait would be 65.1 µs" in read_0["say"]
    assert read_1["curve"]["at"] > prior["curve"]["at"] > read_0["curve"]["at"]
    assert read_1["from"] == read_0["from"] == 0


# ── The script, under Node ────────────────────────────────────────────

# A scene the model checks run on even before a page has one: two cases that
# branch on one readout, a call whose answer may arrive in a window, and a
# broken axis.
_SAMPLE = {
    "name": "sample",
    "title": "A readout, a call and a branch",
    "end": 2000.0,
    "cuts": [[700.0, 1500.0, 100.0]],
    "lanes": [
        {"id": "l0", "label": "q0 drive", "role": "drive"},
        {"id": "l1", "label": "q0 readout", "role": "readout"},
        {"id": "k0", "label": "link", "role": "link"},
    ],
    "spans": [
        {"id": "e0", "lane": "l1", "a": 0.0, "b": 300.0, "role": "readout", "label": "m0"},
        {"id": "e1", "lane": "l0", "a": 1600.0, "b": 1640.0, "role": "drive", "when": [["m0", 1]]},
    ],
    "marks": [{"role": "due", "t": 1600.0, "lanes": ["k0"], "label": "due 1600 ns"}],
    "windows": [{"lane": "k0", "a": 400.0, "b": 1550.0, "role": "answer", "label": "answer"}],
    "messages": [{"id": "r", "from": "l1", "to": "k0", "a": 300.0, "b": 304.0, "role": "request"}],
    "cases": [
        {
            "label": "If m0 reads 1, the pulse plays.",
            "given": {"m0": 1},
            "weight": 0.4,
            "moments": [[0.0, "The readout starts."], [1600.0, "The pulse plays."]],
        },
        {
            "label": "If m0 reads 0, nothing plays.",
            "given": {"m0": 0},
            "weight": 0.5,
            "moments": [[0.0, "The readout starts."]],
        },
        {
            "label": "If m0 reads 2, nothing plays.",
            "given": {"m0": 2},
            "weight": 0.1,
            "moments": [[0.0, "The readout starts."]],
        },
    ],
    "panels": ["timeline", "cases"],
}

_DUMP = r"""
const v = require(process.argv[1]);
const out = [];
for (const file of process.argv.slice(2)) {
  for (const scene of JSON.parse(require("fs").readFileSync(file, "utf8")).scenes) {
    const end = scene.end || 1;
    const times = new Set([0, end]);
    for (const s of scene.spans || []) { times.add(s.a); times.add(s.b); }
    for (const w of scene.windows || []) { times.add(w.a); times.add(w.b); }
    for (const kase of scene.cases || []) for (const [t] of kase.moments) times.add(t);
    const at = [...times].filter((t) => t <= end).sort((a, b) => a - b);
    const states = (scene.cases || []).map((_, c) => at.map((t) => {
      const frame = { c, t, step: 0, f: 0 };
      const live = v.inCase(scene, v.spansAt(scene, frame), c).filter((s) => s.a <= t && t < s.b);
      return [t, live.map((s) => s.id), v.captionAt(scene, frame)];
    }));
    const total = v.totalMs(scene);
    const snaps = scene.snapshots || [];
    out.push({
      name: scene.name,
      beats: v.beats(scene),
      frames: [0, total / 2, total].map((ms) => v.frameAt(scene, ms)),
      round: at.map((t) => [t, v.real(scene, v.visual(scene, t))]),
      monotone: at.every((t, i) => i === 0 || v.visual(scene, t) >= v.visual(scene, at[i - 1])),
      description: v.description(scene),
      rest: v.restFrame(scene),
      states,
      moves: snaps.slice(1).map((_, k) =>
        [0, 0.5, 1].map((f) => v.spansAt(scene, { step: k, f, c: 0, t: end }))),
      stills: snaps.map((_, k) => v.spansAt(scene, { step: k, f: 0, c: 0, t: end })),
    });
  }
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node():
    found = shutil.which("node")
    if found is None:
        if os.environ.get("CUDAQ_PULSE_REQUIRE_NODE"):
            pytest.fail("CUDAQ_PULSE_REQUIRE_NODE is set, but node is not on PATH")
        pytest.skip("needs Node.js to run the figures' script")
    return found


@pytest.fixture(scope="module")
def model(node, tmp_path_factory):
    sample = tmp_path_factory.mktemp("sample") / "sample.json"
    sample.write_text(json.dumps({"format": "pulse-visual/1", "scenes": [_SAMPLE]}))
    files = [str(sample), *map(str, _files())]
    result = subprocess.run(
        [node, "-e", _DUMP, str(_ENGINE), *files],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return {d["name"]: d for d in json.loads(result.stdout)}


def _all_scenes():
    yield _SAMPLE
    for _, scene in _scenes():
        yield scene


def test_the_first_two_cases_play_once_from_the_start_and_end_still(model):
    for scene in _all_scenes():
        drawn = model[scene["name"]]
        first, _, last = drawn["frames"]
        assert last["done"]
        assert drawn["monotone"]
        for t, back in drawn["round"]:
            assert back == pytest.approx(t, abs=1e-6)
        if scene.get("snapshots"):
            continue
        assert first["t"] == 0
        cases = [b["c"] for b in drawn["beats"] if b["kind"] == "play"]
        assert cases == list(range(min(2, len(scene["cases"]))))


def test_each_span_plays_in_exactly_the_cases_its_conditions_allow(model):
    for scene in _all_scenes():
        if scene.get("snapshots"):
            continue
        for c, states in enumerate(model[scene["name"]]["states"]):
            given = scene["cases"][c]["given"]
            for t, live, _ in states:
                want = sorted(
                    s["id"]
                    for s in scene["spans"]
                    if s["a"] <= t < s["b"] and all(given.get(k) == v for k, v in s.get("when", []))
                )
                assert sorted(live) == want, (scene["name"], c, t)


def test_each_sentence_shows_from_its_moment(model):
    for scene in _all_scenes():
        drawn = model[scene["name"]]
        for c, kase in enumerate(scene.get("cases") or []):
            by_t = {t: caption for t, _, caption in drawn["states"][c]}
            for t, _ in kase["moments"]:
                same_time = [m for m in kase["moments"] if m[0] == t]
                assert by_t[t] == same_time[-1][1]
            assert kase["label"].rstrip(".") in drawn["description"]
        for snap in scene.get("snapshots") or []:
            assert (snap.get("say") or snap["label"]) in drawn["description"]


def test_a_move_takes_each_span_from_one_snapshot_to_the_next(model):
    for scene in _all_scenes():
        snaps = scene.get("snapshots")
        if not snaps:
            continue
        drawn = model[scene["name"]]
        for k, still in enumerate(drawn["stills"]):
            shown = {s["id"]: [s["a"], s["b"]] for s in still}
            assert shown == {i: p for i, p in snaps[k]["spans"].items() if p}
        for k, (start, _, end) in enumerate(drawn["moves"]):
            to = snaps[k + 1]
            origin = snaps[to.get("from", k)]["spans"]
            for s in start:
                if origin.get(s["id"]):
                    assert [s["a"], s["b"]] == origin[s["id"]]
            for s in end:
                if to["spans"].get(s["id"]):
                    assert [s["a"], s["b"]] == to["spans"][s["id"]]


def test_the_still_is_the_first_case_or_the_last_snapshot(model):
    for scene in _all_scenes():
        rest = model[scene["name"]]["rest"]
        if scene.get("snapshots"):
            assert rest["step"] == len(scene["snapshots"]) - 1
        else:
            assert (rest["c"], rest["t"]) == (0, scene.get("end") or 1)


def test_the_stills_are_what_the_script_draws(node, tmp_path):
    for path in _files():
        subprocess.run([node, str(_RENDER), str(path), str(tmp_path)], check=True)
        for scene in json.loads(path.read_text())["scenes"]:
            for suffix in ("", "-dark"):
                file = f"{scene['name']}{suffix}.svg"
                assert (tmp_path / file).read_text() == (_FIGURES / file).read_text(), (
                    f"{file} is not what the script draws; run "
                    f"node tools/render-pulse-visual.js docs/tutorials/figures/{path.name}"
                )
