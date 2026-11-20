# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The operations guide's waveform explorer shows what the library plays.

``docs/_static/waveform-explorer.js`` samples envelopes with its own copy of
the envelope formulas and prints the code that makes them. This test runs
it under Node for its cases (each shape at its defaults and with each slider
at either end, each ``wf_*`` combination, and the modulated drive), compiles
every code line it prints, and compares its samples with
``sample_waveform`` of the exported waveform. A changed formula, argument,
unit or convention then fails here, not on a reader's screen.

The explorer's controls are checked too, on a stand-in for the page that has
only what the script uses: each slider has a number field under the same
label, a typed value becomes one the slider has, a change is said once after
a pause with the code and the peaks, the rule behind "Use suggested beta" is
text on the page, and the plot is drawn at the width of its box.

It needs the source tree and Node; ``CUDAQ_PULSE_REQUIRE_NODE=1`` (set in the
docs workflow) makes a missing Node a failure instead of a skip.
"""

from __future__ import annotations

import inspect
import itertools
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import ops
from cudaq_pulse.controller import sample_waveform

from tests._tools import tool_env

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_SCRIPT = _DOCS / "_static" / "waveform-explorer.js"
_FIGURE = _DOCS / "user_guide" / "figures" / "waveform_envelopes.svg"
_OPERATIONS = _DOCS / "user_guide" / "operations.rst"

_DUMP = """
const w = require(process.argv[1]);
const cases = w.cases().map((s) => {
  const { re, im } = w.samples(s);
  return { code: w.codeLines(s), modulated: s.modulate, re, im, peaks: [w.peak(re), w.peak(im)] };
});
const defaults = Object.keys(w.SHAPES).map((shape) => {
  const s = w.defaultState();
  s.a.shape = shape;
  return w.codeLines(s)[0];
});
const beta = [-400, -340, -200, -184, -100].map((f) => [f, w.suggestedBeta(f)]);
// Typed numbers below, at, between and above each slider's steps.
const typed = (min, max, step) => [
  min - step, min, min + 0.4 * step, min + 1.6 * step,
  (min + max) / 2 + step / 3, max - 0.4 * step, max, max + 7 * step,
];
const snapped = Object.fromEntries(Object.entries(w.RANGES).map(([name, [min, max, step]]) => [
  name, typed(min, max, step).map((value) => [value, w.snap(value, min, max, step)]),
]));
process.stdout.write(JSON.stringify({
  clock_ghz: w.CLOCK_GHZ, shapes: w.SHAPES, combine: w.COMBINE, ranges: w.RANGES,
  cases, defaults, beta, snapped,
}));
"""

# The explorer on a stand-in for the page: elements that keep their attributes,
# children and listeners, timers that wait to be run, and a ResizeObserver
# that reports the widths it is given. It plays a reader's steps and prints
# what the controls, the status line and the plot hold after each.
_WIDGET = """
const timers = [];
globalThis.setTimeout = (fn, ms) => timers.push({ fn, ms, live: true });
globalThis.clearTimeout = (handle) => {
  if (timers[handle - 1]) timers[handle - 1].live = false;
};
class Element {
  constructor(tag) {
    Object.assign(this, { tag, attrs: {}, kids: [], on: {}, dataset: {}, style: {} });
    Object.assign(this, { hidden: false, value: "" });
  }
  setAttribute(name, value) {
    this.attrs[name] = String(value);
  }
  get id() {
    return this.attrs.id;
  }
  set id(value) {
    this.attrs.id = value;
  }
  get max() {
    return this.attrs.max;
  }
  set max(value) {
    this.attrs.max = String(value);
  }
  append(...kids) {
    this.kids.push(...kids);
  }
  replaceChildren(...kids) {
    this.kids = kids;
  }
  get textContent() {
    return this.kids.map((k) => (typeof k === "string" ? k : k.textContent)).join("");
  }
  set textContent(text) {
    this.kids = text === "" ? [] : [text];
  }
  addEventListener(type, fn) {
    (this.on[type] ||= []).push(fn);
  }
  // The event of a reader's action, after the change it makes to the control.
  fire(type, change = {}) {
    Object.assign(this, change);
    for (const fn of this.on[type] || []) fn();
  }
  // The elements under this one that are not hidden.
  shown() {
    const kids = this.kids.filter((k) => typeof k !== "string" && !k.hidden);
    return kids.flatMap((k) => [k, ...k.shown()]);
  }
}
const root = new Element("div");
const fallback = new Element("figure");
root.dataset.shape = "drag";
globalThis.window = {};
if (typeof navigator === "undefined") globalThis.navigator = {};
const found = { ".pulse-waveform-explorer": [root], ".pwe-fallback": [fallback] };
globalThis.document = {
  readyState: "complete",
  createElement: (tag) => new Element(tag),
  createElementNS: (_, tag) => new Element(tag),
  querySelectorAll: (selector) => found[selector],
};
let resize;
globalThis.ResizeObserver = class {
  constructor(fn) {
    resize = (width) => fn([{ contentRect: { width } }]);
  }
  observe() {}
};
require(process.argv[1]);

const all = (test) => root.shown().filter(test);
const tag = (name) => all((e) => e.tag === name)[0];
const fields = () => all((e) => e.attrs.class === "pwe-field");
const sliders = () =>
  fields()
    .filter((e) => e.kids[1].attrs.type === "range")
    .map(({ kids: [label, range, entry] }) => {
      const [number, unit] = entry.kids;
      return { label, range, number, unit };
    });
const slider = (name) => sliders().find((s) => s.label.textContent === name);
const control = (name) => fields().find((e) => e.kids[0].textContent === name).kids[1];
const spoken = () => all((e) => e.attrs.role === "status")[0].textContent;
const waiting = () => timers.filter((t) => t.live);
const code = () => tag("code").textContent;
const plot = () => ({
  viewBox: tag("svg").attrs.viewBox,
  height: tag("svg").style.height,
  name: tag("svg").attrs["aria-label"],
  text: all((e) => e.tag === "text").map((e) => [+e.attrs.x, e.textContent]),
  curve: all((e) => e.attrs.class === "pwe-i")[0]
    .attrs.points.split(" ")
    .map((point) => +point.split(",")[0]),
});
const out = {};

out.mounted = {
  tags: [...new Set(root.shown().map((e) => e.tag))],
  sliders: sliders().map((s) => ({
    label: s.label.textContent,
    labelled: { ...s.label.attrs },
    range: { ...s.range.attrs, value: s.range.value },
    number: { ...s.number.attrs, value: s.number.value },
    unit: { ...s.unit.attrs, text: s.unit.textContent },
  })),
  live: all((e) => "aria-live" in e.attrs || e.attrs.role === "status")
    .map((e) => [e.tag, e.attrs.class]),
  waiting: waiting().length,
  spoken: spoken(),
  button: { ...control("suggested beta").attrs },
  notes: all((e) => e.attrs.class === "pwe-note")
    .map((e) => [e.tag, { ...e.attrs }, e.textContent]),
  fallbackHidden: fallback.hidden,
  plot: plot(),
};

// A dragged slider: three steps, then a pause.
const amplitude = slider("amplitude");
for (const value of ["0.31", "0.32", "0.33"]) amplitude.range.fire("input", { value });
out.dragged = {
  number: amplitude.number.value,
  valuetext: amplitude.range.attrs["aria-valuetext"],
  waiting: waiting().map((t) => t.ms),
  before: spoken(),
};
waiting()[0].fn();
out.dragged.after = spoken();

// Typed values, committed: off the steps, past both ends, left empty, and
// in another notation.
const duration = slider("duration");
const controls = sliders().map((s) => s.range);
const kept = () => sliders().every((s, k) => s.range === controls[k]);
out.typed = ["41", "999", "", "-3", "1e2"].map((value) => {
  duration.number.fire("change", { value });
  return [value, duration.number.value, duration.range.value, duration.unit.textContent, code()];
});
out.kept = kept();

control("suggested beta").fire("click");
out.suggested = [slider("beta").number.value, slider("beta").range.value, code()];
out.kept = out.kept && kept();

// The flat top's width stays two cycles under the duration.
control("waveform").fire("change", { value: "gaussian_square" });
out.width = [["width", "500"], ["duration", "21"], ["width", "7"]].map(([name, value]) => {
  slider(name).number.fire("change", { value });
  const width = slider("width");
  return [width.number.attrs.max, width.range.attrs.max, width.number.value, code()];
});

// The plot at the width of a 320px screen, of a box that is not shown, of
// a desktop column, and of a box narrower than the plot is drawn.
const before = timers.length;
out.resized = [254.5, 0, 638.4, 120].map((width) => {
  resize(width);
  return plot();
});
out.resizeSpoke = timers.length !== before;

control("modulate").fire("change", { checked: true });
waiting()[0].fn();
out.modulated = { spoken: spoken(), name: plot().name };
process.stdout.write(JSON.stringify(out));
"""

# Every value the explorer draws is at most 2 in magnitude (amplitudes up to
# 1, wf_scale factors up to 2). Both sides evaluate the same expressions in
# the same order in binary64, so they differ only where V8's exp, tanh, cos
# and sin round differently from the C library's, by an ulp or two: 4.4e-16
# at magnitude 2. 1e-15 allows that and nothing a formula change could hide.
_ATOL = 1e-15

_QS = pulse.quantum_system("transmon_surface17")
_Q = _QS.qudits[0]


def _node(script: str) -> dict:
    """What *script* prints when Node runs it on the explorer's file."""
    if not _SCRIPT.exists():
        pytest.skip("needs the source tree")
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CUDAQ_PULSE_REQUIRE_NODE"):
            pytest.fail("CUDAQ_PULSE_REQUIRE_NODE is set, but node is not on PATH")
        pytest.skip("needs Node.js to run the explorer's script")
    result = subprocess.run(
        [node, "-e", script, str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
        env=tool_env(),
        check=True,
    )
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def explorer():
    return _node(_DUMP)


@pytest.fixture(scope="module")
def widget():
    return _node(_WIDGET)


def _drive(lines: list[str]) -> dict:
    """The drive event of a kernel whose body is *lines*: the explorer's
    statements, or ``drive(d, ...)`` of its one waveform expression."""
    if not lines[0].startswith("d = "):
        lines = ["d = get_drive_line(q)", f"drive(d, {lines[0]})"]
    source = "def explorer_case(q):\n" + textwrap.indent("\n".join(lines), "    ") + "\n"
    namespace: dict = {}
    exec(compile(source, "<waveform explorer>", "exec"), namespace)
    kernel = pulse.kernel(namespace["explorer_case"])
    program = pulse.compile(kernel, quantum_system=_QS).lower(_Q).export_to_controller()
    (event,) = [e for e in program["events"] if e["kind"] == "drive"]
    return event


def test_every_code_line_compiles_and_plays_the_samples_drawn(explorer):
    """The JS samples match the library's: sample_waveform of the exported
    waveform (which takes gaussian_square's ``width`` as the IR's
    ``risefall = (duration - width) / 2``), and, modulated, the drive in the
    qudit's frame, ``E(t) = envelope(t) exp(-i theta(t))``, from the
    exported frequency and phase."""
    qudit_hz = _drive(["square(8, 0.1)"])["frequency_hz"]
    assert len(explorer["cases"]) > 50
    for case in explorer["cases"]:
        code = "\n".join(case["code"])
        event = _drive(case["code"])
        expected = sample_waveform(event["waveform"], event["duration_cycles"])
        if case["modulated"]:
            k = np.arange(len(expected))
            detuning = event["frequency_hz"] - qudit_hz
            angle = event["phase_rad"] + 2 * np.pi * detuning * k / (_QS.clock_ghz * 1e9)
            expected = expected * (np.cos(angle) - 1j * np.sin(angle))
        drawn = np.asarray(case["re"]) + 1j * np.asarray(case["im"])
        assert drawn.shape == expected.shape, code
        np.testing.assert_allclose(
            drawn.real, expected.real, rtol=0, atol=_ATOL, err_msg=f"I of\n{code}"
        )
        np.testing.assert_allclose(
            drawn.imag, expected.imag, rtol=0, atol=_ATOL, err_msg=f"Q of\n{code}"
        )


def _parametric_constructors() -> dict[str, list[str]]:
    """The waveform constructors of numeric arguments, with their arguments:
    all but ``custom``'s function, ``custom_samples``' data and ``wf_*``."""
    found = {}
    for name, fn in vars(ops).items():
        if not inspect.isfunction(fn) or name.startswith("wf_"):
            continue
        if inspect.signature(fn).return_annotation not in ("Waveform", ops.Waveform):
            continue
        if name not in ("custom", "custom_samples"):
            found[name] = list(inspect.signature(fn).parameters)
    return found


def test_the_explorer_has_every_parametric_constructor_and_combinator(explorer):
    shown = {
        shape: ["duration", *(p for p in params if p != "amplitude_imag")]
        for shape, params in explorer["shapes"].items()
    }
    assert shown == _parametric_constructors()
    combinators = {name for name in vars(ops) if name.startswith("wf_")}
    assert set(explorer["combine"]) == {"none", *combinators}


def test_its_times_are_cycles_of_the_docs_quantum_system_clock(explorer):
    assert explorer["clock_ghz"] == _QS.clock_ghz


def test_its_suggested_beta_is_the_one_drag_documents(explorer):
    doc = inspect.getdoc(ops.drag)
    assert "``beta = -1 / alpha``, with ``alpha`` in radians per clock cycle" in " ".join(
        doc.split()
    )
    for anharmonicity_mhz, beta in explorer["beta"]:
        alpha = 2 * np.pi * anharmonicity_mhz * 1e6 / (_QS.clock_ghz * 1e9)
        assert beta == pytest.approx(-1 / alpha, rel=1e-15)


@pytest.mark.skipif(not _OPERATIONS.is_file(), reason="needs the source tree's docs/")
def test_the_modulated_drive_is_the_documented_frame_convention():
    text = " ".join(_OPERATIONS.read_text().split())
    assert "``E(t) = envelope(t) exp(-i theta(t))``" in text
    assert "``theta(t) = theta_0 + (w_tone - w_q) t``" in text


def test_the_static_figure_shows_the_explorer_defaults(explorer):
    """Readers without JavaScript see the figure: one panel per shape, each
    titled with the explorer's default call."""
    svg = _FIGURE.read_text()
    assert [line for line in explorer["defaults"] if line not in svg] == []


def test_the_peaks_it_reports_are_the_largest_samples_it_draws(explorer):
    """The plot's name and the status line say where I and Q peak: the first
    sample of the largest size, with its sign, and whether the next sample
    is as large (a flat top)."""
    for case in explorer["cases"]:
        for values, peak in zip((case["re"], case["im"]), case["peaks"], strict=True):
            size = np.abs(values)
            at = int(np.argmax(size))
            code = "\n".join(case["code"])
            assert (peak["cycle"], peak["value"]) == (at, values[at]), code
            assert peak["flat"] == (at + 1 < len(values) and size[at + 1] == size[at]), code


def test_a_typed_value_becomes_the_nearest_one_its_slider_has(explorer):
    """A number field takes any number; the explorer draws and prints the
    slider's nearest step within its range, written as the slider writes
    it, so the code line stays among the values the cases above cover."""
    assert set(explorer["snapped"]) == set(explorer["ranges"])
    for name, (low, high, step) in explorer["ranges"].items():
        digits = len(str(step).partition(".")[2])
        for typed, snapped in explorer["snapped"][name]:
            steps = (snapped - low) / step
            assert low <= snapped <= high, (name, typed)
            assert abs(steps - round(steps)) < 1e-9, (name, typed)
            assert abs(snapped - min(high, max(low, typed))) < 0.5 * step + 1e-9, (name, typed)
            assert snapped == round(snapped, digits), (name, typed)


def test_each_slider_has_a_number_field_under_the_same_label(widget):
    """The label is the number field's and names the slider too; the slider
    says its value with the unit, and the unit beside the field describes
    the field and is not read again as text of its own. Nothing but the
    slider speaks the value as it changes."""
    mounted = widget["mounted"]
    labels = ["duration", "amplitude", "sigma", "beta", "anharmonicity"]
    assert [s["label"] for s in mounted["sliders"]] == labels
    for s in mounted["sliders"]:
        slider, number, unit = s["range"], s["number"], s["unit"]
        assert s["labelled"] == {"id": slider["aria-labelledby"], "for": number["id"]}
        assert (slider["type"], number["type"]) == ("range", "number")
        for key in ("min", "max", "step", "value"):
            assert number[key] == slider[key], (s["label"], key)
        assert number["aria-describedby"] == unit["id"]
        assert unit.get("aria-hidden") == "true"
        assert slider["aria-valuetext"] == f"{slider['value']} {unit['text']}".strip()
    units = [s["unit"]["text"] for s in mounted["sliders"]]
    assert units == ["cycles (20 ns)", "", "cycles (5 ns)", "cycles (0.25 ns)", "MHz"]
    assert "output" not in mounted["tags"]
    assert mounted["fallbackHidden"]


def test_a_typed_value_moves_the_slider_and_the_code_and_keeps_the_controls(widget):
    assert widget["typed"] == [
        ["41", "42", "42", "cycles (21 ns)", "drag(42, 0.33, 10, 0.5)"],
        ["999", "400", "400", "cycles (200 ns)", "drag(400, 0.33, 10, 0.5)"],
        ["", "400", "400", "cycles (200 ns)", "drag(400, 0.33, 10, 0.5)"],
        ["-3", "8", "8", "cycles (4 ns)", "drag(8, 0.33, 10, 0.5)"],
        ["1e2", "100", "100", "cycles (50 ns)", "drag(100, 0.33, 10, 0.5)"],
    ]
    assert widget["suggested"] == ["1.59", "1.59", "drag(100, 0.33, 10, 1.59)"]
    # A duration committed with Tab moves the focus to the next slider, so
    # the other sliders must still be the elements they were.
    assert widget["kept"]
    # width has both controls' upper end two cycles under the duration.
    assert widget["width"] == [
        ["98", "98", "98", "gaussian_square(100, 0.3, 4, 98)"],
        ["20", "20", "20", "gaussian_square(22, 0.3, 4, 20)"],
        ["20", "20", "8", "gaussian_square(22, 0.3, 4, 8)"],
    ]
    for code in [*(row[-1] for row in widget["typed"] + widget["width"]), widget["suggested"][-1]]:
        assert _drive([code])["duration_cycles"] == int(code.split("(")[1].split(",")[0])


def test_a_change_is_said_once_after_a_pause_with_the_code_and_the_peaks(widget):
    mounted, dragged, modulated = widget["mounted"], widget["dragged"], widget["modulated"]
    # The status line and the "Copied" note are the only live text, and
    # nothing is said when the page loads.
    assert mounted["live"] == [["span", "pwe-note"], ["p", "pwe-hidden"]]
    assert (mounted["waiting"], mounted["spoken"]) == (0, "")
    # Three steps of a slider: the slider says each value; one sentence
    # follows 400 ms after the last.
    assert (dragged["number"], dragged["valuetext"]) == ("0.33", "0.33")
    assert (dragged["waiting"], dragged["before"]) == ([400], "")
    peaks = "Peak I: 0.33 at cycle 20 (10 ns). Peak Q: 0.01 at cycle 10 (5 ns)."
    assert dragged["after"] == f"drag(40, 0.33, 10, 0.5). {peaks}"
    lines = "d = get_drive_line(q); shift_frequency(d.tone, 100e6); drive(d, {})"
    flat_top = "gaussian_square(22, 0.3, 4, 8)"
    peaks = "Peak Re E(t): -0.3 at cycle 10 (5 ns). Peak Im E(t): 0.3 at cycle 15 (7.5 ns)."
    assert modulated["spoken"] == f"{lines.format(flat_top)}. {peaks}"
    assert modulated["name"] == f"Modulated drive of drive(d, {flat_top}), 22 samples. {peaks}"


def test_the_plot_is_named_with_its_peaks_which_are_also_in_sight(widget):
    mounted = widget["mounted"]
    peaks = "Peak I: 0.3 at cycle 20 (10 ns). Peak Q: 0.0091 at cycle 10 (5 ns)."
    assert mounted["plot"]["name"] == f"I and Q of drag(40, 0.3, 10, 0.5), 40 samples. {peaks}"
    # The name is an attribute of the plot: a title element would be given as
    # its description too, and the sentence read twice.
    assert "title" not in mounted["tags"]
    # The sentence under the plot repeats the name's, so only sight gets it.
    assert ["p", {"class": "pwe-note", "aria-hidden": "true"}, peaks] in mounted["notes"]


def test_the_rule_behind_use_suggested_beta_is_text_on_the_page(widget):
    """Not a tooltip, which the keyboard does not open: a note under the
    button, which is also the button's description."""
    button = widget["mounted"]["button"]
    assert "title" not in button
    ((kind, attrs, text),) = [
        note
        for note in widget["mounted"]["notes"]
        if note[1].get("id") == button["aria-describedby"]
    ]
    assert (kind, attrs["class"]) == ("p", "pwe-note")
    assert not {"aria-live", "role", "aria-hidden"} & set(attrs)
    assert "beta = -1 / alpha, for the anharmonicity alpha in radians per clock cycle" in text


def test_the_plot_is_drawn_at_the_width_of_its_box(widget):
    """Its text is set in CSS pixels, so the drawing must be as wide as the
    box, whatever the screen: all of it inside, time ticks 36px apart or
    more, and the short axis label where the long one does not fit."""
    narrow, unshown, desktop, tiny = widget["resized"]
    assert [p["viewBox"] for p in (narrow, desktop, tiny)] == [
        "0 0 254 260",
        "0 0 638 260",
        "0 0 200 260",
    ]
    assert unshown == narrow
    for plot in (narrow, desktop, tiny):
        width = int(plot["viewBox"].split()[2])
        assert plot["height"] == "260px"
        assert all(0 <= x <= width for x, _ in plot["text"])
        assert all(62 <= x <= width - 12 for x in plot["curve"])
        first = [text for _, text in plot["text"]].index("amplitude") + 1
        ticks = [x for x, _ in plot["text"][first:-1]]
        assert len(ticks) >= 2
        assert all(b - a >= 36 for a, b in itertools.pairwise(ticks))
    assert narrow["text"][-1][1] == tiny["text"][-1][1] == "time (clock cycles)"
    assert desktop["text"][-1][1] == "time (clock cycles; 1 cycle = 0.5 ns)"
    # A redraw for a new width is not a change to announce.
    assert not widget["resizeSpoke"]
