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
- its playback, run under Node on a stand-in for the page, plays a figure
  once while it is in view and says nothing meanwhile, hands over to the
  reader's Pause, Previous step and Next step and says each step they take,
  and under reduced motion shows the still frame and never moves;
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

from tests._tools import tool_env

_ROOT = Path(__file__).resolve().parents[1]
_ENGINE = _ROOT / "docs" / "_static" / "pulse-visual.js"
_STYLE = _ROOT / "docs" / "_static" / "pulse-visual.css"
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


def test_the_figures_follow_the_reduced_motion_setting_and_nothing_moves_without_end():
    script = _ENGINE.read_text()
    assert '"(prefers-reduced-motion: reduce)"' in script
    # The setting can change while the page is open.
    assert 'addEventListener("change"' in script
    assert "infinite" not in _STYLE.read_text()


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


def test_the_lowering_scene_says_what_the_gpu_page_prints():
    scene = _scene("gpu_simulation", "lowering")
    titles = [stage["title"] for stage in scene["stages"]]
    assert titles == ["Pulse MLIR", "QOp", "cuDensityMat", "LLVM IR", "controller export"]
    # "calls into cuDensityMat: 88"
    assert scene["stages"][3]["count"] == "88 llvm.call"
    # Each stage shows at its own step, and the export comes from the Pulse MLIR.
    assert [stage["step"] for stage in scene["stages"]] == list(range(len(scene["snapshots"])))
    assert scene["stages"][-1]["from"] == 0


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
      stops: v.stops(scene).map((stop) => {
        const start = v.startOf(scene, stop);
        return [v.stopAt(scene, stop), start, v.stopAt(scene, v.frameAt(scene, start))];
      }),
      total,
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


def _run(node: str, script: str, folder: Path) -> str:
    """What *script* prints for the sample scene and every committed one."""
    sample = folder / "sample.json"
    sample.write_text(json.dumps({"format": "pulse-visual/1", "scenes": [_SAMPLE]}))
    files = [str(sample), *map(str, _files())]
    result = subprocess.run(
        [node, "-e", script, str(_ENGINE), *files],
        capture_output=True,
        text=True,
        timeout=120,
        env=tool_env(),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


@pytest.fixture(scope="module")
def model(node, tmp_path_factory):
    drawn = json.loads(_run(node, _DUMP, tmp_path_factory.mktemp("sample")))
    return {d["name"]: d for d in drawn}


def _all_scenes():
    yield _SAMPLE
    for _, scene in _scenes():
        yield scene


def _steps(scene: dict) -> list[tuple[int, str]]:
    """Each step a reader can stop on: its case, and what the line under the picture says.

    A snapshot says its sentence. A case says each of its moments, then the last one again
    with the playhead at the end; a screen reader hears the case's label first.
    """
    if scene.get("snapshots"):
        return [(0, snap.get("say") or snap["label"]) for snap in scene["snapshots"]]
    out = []
    for c, kase in enumerate(scene["cases"]):
        said = [[line for u, line in kase["moments"] if u <= t][-1] for t, _ in kase["moments"]]
        out += [(c, f"{kase['label']} {line}") for line in [*said, said[-1]]]
    return out


def _status(scene: dict, k: int) -> str:
    steps = _steps(scene)
    return f"Step {k + 1} of {len(steps)}: {steps[k][1]}"


def _played(scene: dict) -> int:
    """How many steps the run plays: every snapshot, or the steps of the first two cases."""
    return sum(1 for c, _ in _steps(scene) if c < 2)


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


def test_playing_goes_on_from_the_step_a_reader_stopped_on(model):
    for scene in _all_scenes():
        drawn = model[scene["name"]]
        steps = _steps(scene)
        assert len(drawn["stops"]) == len(steps)
        for k, (at, start, resumed) in enumerate(drawn["stops"]):
            assert at == k, (scene["name"], k)
            last_of_case = k + 1 == len(steps) or steps[k + 1][0] != steps[k][0]
            if k >= _played(scene) or k + 1 == _played(scene):
                # A step the run doesn't play, and its last step: "Replay" starts over.
                assert start == drawn["total"], (scene["name"], k)
            elif last_of_case:
                # From a case's last step, the next case plays.
                assert resumed == k + 1, (scene["name"], k)
            else:
                assert resumed == k, (scene["name"], k)
        # A later step is never earlier in the run.
        starts = [start for _, start, _ in drawn["stops"]]
        assert starts == sorted(starts), scene["name"]


def test_the_stills_are_what_the_script_draws(node, tmp_path):
    for path in _files():
        subprocess.run([node, str(_RENDER), str(path), str(tmp_path)], env=tool_env(), check=True)
        for scene in json.loads(path.read_text())["scenes"]:
            for suffix in ("", "-dark"):
                file = f"{scene['name']}{suffix}.svg"
                assert (tmp_path / file).read_text() == (_FIGURES / file).read_text(), (
                    f"{file} is not what the script draws; run "
                    f"node tools/render-pulse-visual.js docs/tutorials/figures/{path.name}"
                )


# ── Playback, on a stand-in for the page ──────────────────────────────

# Mounts each scene as the directive writes it, then moves the clock, the
# window, the tab and the reduced-motion setting, presses the controls, and
# prints what the figure shows after each.
_PAGE = r"""
// Just enough of a DOM for the playback code: nodes that keep their
// attributes, children and listeners.
let heard = [];
class Node {
  constructor(tag, text = "") {
    Object.assign(this, { tag, text, attrs: new Map(), children: [], listeners: {}, parent: null });
  }
  setAttribute(k, v) { this.attrs.set(k, String(v)); }
  getAttribute(k) { return this.attrs.has(k) ? this.attrs.get(k) : null; }
  removeAttribute(k) { this.attrs.delete(k); }
  get className() { return this.getAttribute("class") || ""; }
  set className(v) { this.setAttribute("class", v); }
  get classList() { return { add: (c) => (this.className = `${this.className} ${c}`.trim()) }; }
  get hidden() { return this.attrs.has("hidden"); }
  set hidden(v) { if (v) this.setAttribute("hidden", ""); else this.removeAttribute("hidden"); }
  set tabIndex(v) { this.setAttribute("tabindex", v); }
  appendChild(c) { c.parent = this; this.children.push(c); return c; }
  append(...cs) {
    cs.forEach((c) => this.appendChild(typeof c === "string" ? new Node("#text", c) : c));
  }
  replaceChildren(...cs) {
    this.children = [];
    this.append(...cs);
    // A screen reader says a status line that is written, unless it is off.
    if (this.getAttribute("role") === "status") {
      heard.push(this.getAttribute("aria-live") !== "off");
    }
  }
  get textContent() { return this.text + this.children.map((c) => c.textContent).join(""); }
  set textContent(t) { this.text = String(t); this.children = []; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  closest(tag) { return this.tag === tag ? this : this.parent && this.parent.closest(tag); }
  all(cls, out = []) {
    if (this.className.split(" ").includes(cls)) out.push(this);
    this.children.forEach((c) => c.all(cls, out));
    return out;
  }
  querySelector(selector) { return this.all(selector.slice(1))[0] || null; }
}

let now = 1000;
let frames = new Map();
let ids = 0;
globalThis.requestAnimationFrame = (fn) => (frames.set(++ids, fn), ids);
globalThis.cancelAnimationFrame = (id) => frames.delete(id);
// Moves the clock on by up to *ms*, a frame every 25 ms, while the figure
// moves; returns how long it did.
function run(ms) {
  let moved = 0;
  for (; moved < ms && frames.size; moved += 25) {
    now += 25;
    const due = [...frames.values()];
    frames = new Map();
    due.forEach((fn) => fn(now));
  }
  return moved;
}

const observers = [];
globalThis.IntersectionObserver = class {
  constructor(seen) { this.seen = seen; observers.push(this); }
  observe() {}
  disconnect() {}
};
const queries = [];
let reduced = false;
globalThis.window = {
  matchMedia(media) {
    const query = { media, matches: reduced, addEventListener: (_, fn) => (query.changed = fn) };
    queries.push(query);
    return query;
  },
};
function reduce(on) {
  reduced = on;
  for (const query of queries) {
    query.matches = on;
    if (query.changed) query.changed();
  }
}
const page = new Node("#document");
let roots = [];
let source = "";
globalThis.document = Object.assign(page, {
  readyState: "loading",
  hidden: false,
  createElement: (tag) => new Node(tag),
  createElementNS: (_ns, tag) => new Node(tag),
  createTextNode: (text) => new Node("#text", text),
  getElementById: () => ({ textContent: source }),
  querySelectorAll: () => roots,
});
function tab(hidden) {
  page.hidden = hidden;
  page.listeners.visibilitychange.forEach((fn) => fn());
}

const [engine, ...files] = process.argv.slice(1);
const { stops } = require(engine);
const start = page.listeners.DOMContentLoaded[0];

// One figure on the page, as the directive writes it.
function mount(scene) {
  source = JSON.stringify({ scenes: [scene] });
  const figure = new Node("figure");
  const root = new Node("div");
  root.className = "pulse-visual";
  root.dataset = { source: "data", scene: scene.name };
  const fallback = new Node("div");
  fallback.className = "pulse-fallback";
  figure.append(root, fallback);
  roots = [root];
  heard = [];
  start();
  const [svg, status, bar] = root.children;
  const observer = observers[observers.length - 1];
  const named = (attr, value) =>
    bar.children.filter((b) => b.getAttribute(attr) === value).map((b) => b.textContent);
  return {
    root, svg, status, bar, fallback,
    // Whether each write of the status line since the last call was said.
    said: () => heard.splice(0),
    press(name) {
      const button = bar.children.find((b) => b.textContent === name);
      if (!button) throw new Error(`${scene.name}: no button reads "${name}"`);
      button.onclick();
    },
    // How much of the figure is in the window, and whether its controls are.
    see(ratio, controls) {
      observer.seen([
        { target: root, isIntersecting: ratio > 0, intersectionRatio: ratio },
        { target: bar, isIntersecting: controls, intersectionRatio: controls ? 1 : 0 },
      ]);
    },
    key(key, held) {
      const event = { key, [held]: true, prevented: false };
      event.preventDefault = () => (event.prevented = true);
      (bar.listeners.keydown || []).forEach((fn) => fn(event));
      return event.prevented;
    },
    state: () => ({
      status: status.textContent,
      live: status.getAttribute("aria-live"),
      toggle: bar.children[0].hidden ? null : bar.children[0].textContent,
      disabled: named("aria-disabled", "true"),
      current: named("aria-current", "step"),
      moving: frames.size > 0,
    }),
  };
}

const out = {};
for (const file of files) {
  for (const scene of JSON.parse(require("fs").readFileSync(file, "utf8")).scenes) {
    // A scene with autoplay false has no run to check.
    if (scene.autoplay === false) continue;
    const seen = (out[scene.name] = {});
    const attrs = (node) => Object.fromEntries(node.attrs);
    // No loop below outlasts a figure that never ends or never marks its ends.
    const count = stops(scene).length;

    // Mounted, below the window: what a screen reader is given.
    let fig = mount(scene);
    const [title, desc] = fig.svg.children;
    seen.markup = {
      root: attrs(fig.root),
      keysOnRoot: Object.keys(fig.root.listeners),
      svg: attrs(fig.svg),
      title: [title.getAttribute("id"), title.textContent],
      desc: [desc.getAttribute("id"), desc.textContent],
      status: attrs(fig.status),
      bar: attrs(fig.bar),
      buttons: fig.bar.children.map((b) => [b.tag, b.type, b.textContent]),
      fallbackHidden: fig.fallback.hidden,
    };
    seen.mounted = fig.state();
    seen.saidOnMount = fig.said();

    // It plays once, in view, and says nothing while it does.
    fig.see(0.5, false);
    seen.halfWithoutControls = fig.state();
    fig.see(0.4, true);
    seen.controlsWithoutHalf = fig.state();
    fig.see(0.5, true);
    seen.playing = fig.state();
    seen.run = [];
    for (let ms = 0; ms < 60000 && run(25); ms += 25) {
      const s = fig.state();
      const last = seen.run[seen.run.length - 1];
      if (!last || last[0] !== s.status) seen.run.push([s.status, s.live, s.toggle]);
    }
    seen.saidWhilePlaying = fig.said();
    seen.ended = fig.state();
    fig.see(0, false);
    fig.see(1, true);
    seen.seenAgain = fig.state();

    // Out of the window it waits, and goes on from there when it is back.
    fig = mount(scene);
    seen.total = fig.root.pulseVisual.total;
    fig.see(1, true);
    seen.ranBeforeLeaving = run(2500);
    fig.see(0, false);
    seen.left = fig.state();
    now += 60000;
    fig.see(0.6, true);
    seen.back = fig.state();
    seen.ranAfterComingBack = run(60000);
    // So it does while the tab is hidden.
    fig = mount(scene);
    fig.see(1, true);
    run(500);
    tab(true);
    seen.tabHidden = fig.state();
    tab(false);
    seen.tabShown = fig.state();
    fig.press("Pause");

    // A control stops the run for good, and the step it goes to is said.
    fig = mount(scene);
    fig.see(1, true);
    run(300);
    fig.said();
    fig.press("Next step");
    seen.saidOnStep = fig.said();
    seen.stepped = fig.state();
    fig.see(0, false);
    fig.see(1, true);
    tab(true);
    tab(false);
    seen.steppedThenSeen = fig.state();
    fig.press("Play");
    seen.played = fig.state();
    run(100);
    fig.said();
    fig.press("Pause");
    seen.saidOnPause = fig.said();
    seen.paused = fig.state();
    now += 1000;
    seen.pausedLater = fig.state();
    const ended = (name) => fig.state().disabled.includes(name);
    for (let i = 0; i < count && !ended("Previous step"); i++) fig.press("Previous step");
    seen.first = fig.state();
    fig.said();
    fig.press("Previous step");
    seen.beforeFirst = fig.state();
    seen.saidBeforeFirst = fig.said();
    seen.walk = [];
    for (let i = 0; i < count && !ended("Next step"); i++) {
      fig.press("Next step");
      seen.walk.push(fig.state().status);
    }
    seen.last = fig.state();
    seen.jumps = fig.bar.children.slice(3).map((b) => (b.onclick(), fig.state()));

    // The arrow keys step, unless a modifier is held; they count as a control.
    fig = mount(scene);
    fig.press("Next step");
    const second = fig.state().status;
    seen.keys = {};
    for (const held of ["altKey", "ctrlKey", "metaKey", "shiftKey"]) {
      seen.keys[held] = [fig.key("ArrowLeft", held), fig.state().status === second];
    }
    seen.keys.Tab = [fig.key("Tab"), fig.state().status === second];
    seen.keys.ArrowLeft = [fig.key("ArrowLeft"), fig.state().status];
    seen.keys.ArrowRight = [fig.key("ArrowRight"), fig.state().status === second];
    fig.see(1, true);
    seen.keysThenSeen = fig.state();

    // Reduced motion: the still frame and no run; the setting is followed
    // while the page is open.
    reduce(true);
    fig = mount(scene);
    fig.see(1, true);
    seen.still = fig.state();
    reduce(false);
    seen.motionAllowed = fig.state();
    fig = mount(scene);
    fig.see(1, true);
    run(300);
    reduce(true);
    seen.reducedWhilePlaying = fig.state();
    seen.saidOnSettling = fig.said();
    reduce(false);
  }
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def playback(node, tmp_path_factory):
    """Each scene that plays by itself, with what the page showed of it."""
    seen = json.loads(_run(node, _PAGE, tmp_path_factory.mktemp("page")))
    return [(scene, seen[scene["name"]]) for scene in _all_scenes() if scene["name"] in seen]


def _jumps(scene: dict) -> list[str]:
    """The buttons that name a snapshot."""
    return [snap.get("short") or snap["label"] for snap in scene.get("snapshots") or []]


def _rest(scene: dict) -> int:
    """The step a still figure shows: the last snapshot, or the end of the first case."""
    if scene.get("snapshots"):
        return len(scene["snapshots"]) - 1
    return len(scene["cases"][0]["moments"])


def test_a_figure_is_a_named_group_with_a_described_picture_and_one_bar_of_buttons(playback):
    for scene, seen in playback:
        markup = seen["markup"]
        # The group and the picture are named by the short title alone; the long text is
        # the picture's description. The group is not a Tab stop and takes no keys.
        assert markup["root"] == {
            "class": "pulse-visual",
            "role": "group",
            "aria-label": scene["title"],
        }
        assert markup["keysOnRoot"] == []
        (title_id, title), (desc_id, desc) = markup["title"], markup["desc"]
        assert title == scene["title"] and desc.startswith(f"{title}. ")
        assert markup["svg"]["role"] == "img"
        assert markup["svg"]["aria-labelledby"] == title_id
        assert markup["svg"]["aria-describedby"] == desc_id
        # The line is a status that starts off: nothing is said when the page loads.
        assert markup["status"] == {
            "class": "pulse-visual-status",
            "role": "status",
            "aria-live": "off",
        }
        assert markup["bar"] == {"class": "pulse-visual-controls"}
        names = ["Play", "Previous step", "Next step", *_jumps(scene)]
        assert markup["buttons"] == [["button", "button", name] for name in names]
        assert markup["fallbackHidden"]


def test_a_figure_plays_once_in_view_and_says_nothing_meanwhile(playback):
    for scene, seen in playback:
        first = {
            "status": _status(scene, 0),
            "live": "off",
            "toggle": "Play",
            "disabled": ["Previous step"],
            "current": _jumps(scene)[:1],
            "moving": False,
        }
        # It waits for half of it and for its controls to be in the window.
        assert seen["mounted"] == first
        assert seen["halfWithoutControls"] == first
        assert seen["controlsWithoutHalf"] == first
        assert seen["playing"] == {**first, "toggle": "Pause", "moving": True}
        # The line counts the steps up while the status is off.
        said = [status for status, _, _ in seen["run"]]
        steps = [_status(scene, k) for k in range(_played(scene))]
        assert said[0] == steps[0] and said[-1] == steps[-1], scene["name"]
        assert [steps.index(status) for status in said] == sorted({*map(steps.index, said)})
        assert {(live, toggle) for _, live, toggle in seen["run"]} == {("off", "Pause")}
        assert seen["saidOnMount"] == [False]
        assert seen["saidWhilePlaying"] and not any(seen["saidWhilePlaying"])
        # At the end it stops, and does not start again when it comes back into view.
        assert seen["ended"]["status"] == steps[-1]
        assert (seen["ended"]["live"], seen["ended"]["toggle"]) == ("off", "Replay")
        assert not seen["ended"]["moving"]
        assert seen["seenAgain"] == seen["ended"]


def test_a_figure_waits_while_it_is_out_of_the_window_or_the_tab_is_hidden(playback):
    for scene, seen in playback:
        assert seen["ranBeforeLeaving"] == 2500
        assert (seen["left"]["moving"], seen["left"]["toggle"]) == (False, "Play")
        assert (seen["back"]["moving"], seen["back"]["toggle"]) == (True, "Pause")
        assert seen["back"]["status"] == seen["left"]["status"]
        # It goes on from where it waited: the two parts make one run. Each start
        # spends a frame on reading the clock.
        over = seen["ranBeforeLeaving"] + seen["ranAfterComingBack"] - seen["total"]
        assert 0 <= over <= 100, (scene["name"], over)
        assert (seen["tabHidden"]["moving"], seen["tabHidden"]["toggle"]) == (False, "Play")
        assert (seen["tabShown"]["moving"], seen["tabShown"]["toggle"]) == (True, "Pause")


def test_a_control_stops_the_run_for_good_and_the_step_is_said(playback):
    for scene, seen in playback:
        count = len(_steps(scene))
        stepped = seen["stepped"]
        assert stepped["status"] == _status(scene, 1)
        assert (stepped["live"], stepped["toggle"], stepped["moving"]) == (None, "Play", False)
        assert seen["saidOnStep"] == [True]
        # Neither the window nor the tab starts it again.
        assert seen["steppedThenSeen"] == stepped
        # Play goes on from the step; Pause keeps the frame, and says where it is.
        assert seen["played"] == {**stepped, "live": "off", "toggle": "Pause", "moving": True}
        assert seen["paused"] == stepped
        assert seen["saidOnPause"] == [True]
        assert seen["pausedLater"] == stepped
        # The step buttons stop at the first and the last step.
        assert seen["first"]["status"] == _status(scene, 0)
        assert seen["first"]["disabled"] == ["Previous step"]
        assert seen["beforeFirst"] == seen["first"] and seen["saidBeforeFirst"] == []
        assert seen["walk"] == [_status(scene, k) for k in range(1, count)]
        assert seen["last"]["disabled"] == ["Next step"]
        assert seen["last"]["toggle"] == "Replay"
        # A button that names a step goes there, and is marked as the current one.
        assert [jump["status"] for jump in seen["jumps"]] == [
            _status(scene, k) for k in range(len(_jumps(scene)))
        ]
        assert [jump["current"] for jump in seen["jumps"]] == [[name] for name in _jumps(scene)]


def test_the_arrow_keys_step_unless_a_modifier_is_held(playback):
    for scene, seen in playback:
        # [the key was taken from the browser, the figure stayed on its step]
        for held in ("altKey", "ctrlKey", "metaKey", "shiftKey", "Tab"):
            assert seen["keys"][held] == [False, True], (scene["name"], held)
        assert seen["keys"]["ArrowLeft"] == [True, _status(scene, 0)]
        assert seen["keys"]["ArrowRight"] == [True, True]
        assert not seen["keysThenSeen"]["moving"]


def test_reduced_motion_shows_the_still_frame_and_is_followed_while_the_page_is_open(playback):
    for scene, seen in playback:
        still = seen["still"]
        assert still["status"] == _status(scene, _rest(scene))
        # No button plays it, and coming into view does not either.
        assert (still["toggle"], still["moving"], still["live"]) == (None, False, "off")
        # Motion allowed again: the button is back, and the figure stays still.
        assert seen["motionAllowed"] == {**still, "toggle": "Replay"}
        # Reduced motion turned on during the run: the still frame at once, in silence.
        assert seen["reducedWhilePlaying"] == still
        assert not any(seen["saidOnSettling"])
