# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The operations guide's waveform explorer shows what the library plays.

``docs/_static/waveform-explorer.js`` samples envelopes with its own copy of
the envelope formulas and prints the code that makes them. This test runs
it under Node for its cases (each shape at its defaults and with each slider
at either end, each ``wf_*`` combination, and the modulated drive), compiles
every code line it prints, and compares its samples with
``sample_waveform`` of the exported waveform. A changed formula, argument,
unit or convention then fails here, not on a reader's screen. It needs the
source tree and Node; ``CUDAQ_PULSE_REQUIRE_NODE=1`` (set in the docs
workflow) makes a missing Node a failure instead of a skip.
"""

from __future__ import annotations

import inspect
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

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_SCRIPT = _DOCS / "_static" / "waveform-explorer.js"
_FIGURE = _DOCS / "user_guide" / "figures" / "waveform_envelopes.svg"
_OPERATIONS = _DOCS / "user_guide" / "operations.rst"

_DUMP = """
const w = require(process.argv[1]);
const cases = w.cases().map((s) => ({
  code: w.codeLines(s), modulated: s.modulate, ...w.samples(s),
}));
const defaults = Object.keys(w.SHAPES).map((shape) => {
  const s = w.defaultState();
  s.a.shape = shape;
  return w.codeLines(s)[0];
});
const beta = [-400, -340, -200, -184, -100].map((f) => [f, w.suggestedBeta(f)]);
process.stdout.write(JSON.stringify({
  clock_ghz: w.CLOCK_GHZ, shapes: w.SHAPES, combine: w.COMBINE, cases, defaults, beta,
}));
"""

# Every value the explorer draws is at most 2 in magnitude (amplitudes up to
# 1, wf_scale factors up to 2). Both sides evaluate the same expressions in
# the same order in binary64, so they differ only where V8's exp, tanh, cos
# and sin round differently from the C library's, by an ulp or two: 4.4e-16
# at magnitude 2. 1e-15 allows that and nothing a formula change could hide.
_ATOL = 1e-15

_QS = pulse.quantum_system("transmon_surface17")
_Q = _QS.qudits[0]


@pytest.fixture(scope="module")
def explorer():
    if not _SCRIPT.exists():
        pytest.skip("needs the source tree")
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CUDAQ_PULSE_REQUIRE_NODE"):
            pytest.fail("CUDAQ_PULSE_REQUIRE_NODE is set, but node is not on PATH")
        pytest.skip("needs Node.js to run the explorer's script")
    result = subprocess.run(
        [node, "-e", _DUMP, str(_SCRIPT)], capture_output=True, text=True, timeout=120, check=True
    )
    return json.loads(result.stdout)


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
