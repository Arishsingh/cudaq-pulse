# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""docs/user_guide/transmon_surface17.rst says what the system says.

The page's numbers and names are substitutions, and its tables are parts of
``transmon_surface17.inc``, which ``docs/user_guide/figures/
transmon_surface17.py`` writes from ``pulse.quantum_system(
"transmon_surface17")`` with the page's figures. The script writes them all
again here, and each must be the committed one: the include word for word,
a figure as ``tests/test_tutorial_figures.py`` compares figures. The page's
own text holds no number, so nothing in it can differ from the system; its
snippets run and print what their output blocks show, as a tutorial page's
do; and each figure has both themes and the page's ``:alt:`` as its
description.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest

from tests.test_tutorial_figures import _desc, _themed_figures, same_figure
from tests.test_tutorials import _abridges, code_blocks

_ROOT = Path(__file__).resolve().parents[1]
_GUIDE = _ROOT / "docs" / "user_guide"
_PAGE = _GUIDE / "transmon_surface17.rst"
_INCLUDE = _GUIDE / "transmon_surface17.inc"
_SCRIPT = _GUIDE / "figures" / "transmon_surface17.py"
_EXAMPLE_LINK = re.compile(r":example:`(?P<path>[^`<>]+\.py)`")

pytestmark = pytest.mark.skipif(not _PAGE.is_file(), reason="needs the source tree's docs/")


def _prose(text: str, substitutions: bool = False) -> str:
    """The page's own words: *text* without its comments, directives and
    their options and bodies, literals and roles, and without its
    substitutions unless *substitutions* is set."""
    kept, skipping = [], False
    for line in text.splitlines():
        if line.startswith(".."):
            skipping = True
            continue
        if skipping and (not line.strip() or line[:1] in " \t"):
            continue
        skipping = False
        kept.append(line)
    words = "\n".join(kept)
    words = re.sub(r"``[^`]*``|:\w+:`[^`]*`", " ", words)
    return words if substitutions else re.sub(r"\|[\w-]+\|", " ", words)


def test_the_page_runs_and_prints_what_it_shows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDAQ_PULSE_QUANTUM_SYSTEM", raising=False)
    # The page imports the example it links, as a tutorial page may.
    for linked in sorted({m["path"] for m in _EXAMPLE_LINK.finditer(_PAGE.read_text())}):
        monkeypatch.syspath_prepend(str((_ROOT / "examples" / linked).parent))
    namespace: dict = {"__name__": "transmon_surface17_page"}
    blocks = code_blocks(_PAGE)
    assert len(blocks) >= 5
    for number, (classes, code, shown) in enumerate(blocks, 1):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code_object = compile(code, f"{_PAGE.name}, block {number}", "exec", dont_inherit=True)
            exec(code_object, namespace)
        printed = [line.rstrip() for line in out.getvalue().splitlines()]
        assert shown is not None, f"block {number} has no output block"
        same = _abridges(shown, printed) if "abridged" in classes else printed == shown
        assert same, f"block {number}: printed {printed!r}, its output block shows {shown!r}"


def test_the_text_holds_no_number_of_its_own():
    system = pulse.quantum_system("transmon_surface17")
    labels = {q.label for q in system.physical_qudits.values()}
    # Names that hold a digit: the system, its qudits, the gates, the
    # coherence times and the two transitions.
    names = labels | {"transmon_surface17", "X90", "Y90", "T1", "T2", "0-1", "1-2"}
    words = re.findall(r"[\w.-]*\d[\w.-]*", _prose(_PAGE.read_text()))
    typed = sorted({w.strip(".-") for w in words} - names)
    assert typed == [], (
        f"the page's text gives {typed} itself; give each as a substitution that "
        f"{_SCRIPT.name} reads from the system"
    )


def test_every_substitution_and_table_of_the_page_is_in_the_include():
    page, include = _PAGE.read_text(), _INCLUDE.read_text()
    defined = set(re.findall(r"^\.\. \|([\w-]+)\| replace::", include, flags=re.M))
    used = set(re.findall(r"\|([\w-]+)\|", _prose(page, substitutions=True)))
    assert used and used <= defined, f"not in {_INCLUDE.name}: {sorted(used - defined)}"
    parts = re.findall(r"^   :start-after: \.\. start of (\w+)$", page, flags=re.M)
    assert parts, "the page includes no table"
    for part in parts:
        assert f".. start of {part}\n" in include and f".. end of {part}\n" in include, part
        assert f"   :end-before: .. end of {part}\n" in page, part


def test_each_figure_has_both_themes_and_the_page_says_what_it_shows():
    figures = list(_themed_figures(_PAGE))
    assert len(figures) >= 4
    for light, alt in figures:
        dark = light.with_name(f"{light.stem}-dark{light.suffix}")
        assert light.exists() and dark.exists(), f"{light.name} needs both files"
        for svg in (light, dark):
            title, desc = _desc(svg)
            assert title and desc, f"{svg.name} has no title or description"
            assert alt == " ".join(desc.split()), (
                f"the :alt: of {light.name} is not its description; run "
                f"python docs/user_guide/figures/{_SCRIPT.name} and copy the description"
            )
            assert "<image" not in svg.read_text(), f"{svg.name} embeds an image"


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="needs matplotlib")
def test_the_include_and_the_figures_are_what_the_script_writes(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        "PULSE_FIGURES_DIR": str(tmp_path),
        "MPLBACKEND": "Agg",
    }
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    env.pop("CUDAQ_PULSE_QUANTUM_SYSTEM", None)
    result = subprocess.run(
        [sys.executable, _SCRIPT.name],
        cwd=_SCRIPT.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    update = f"run python docs/user_guide/figures/{_SCRIPT.name}"
    written = tmp_path / _INCLUDE.name
    assert written.read_text() == _INCLUDE.read_text(), (
        f"{_INCLUDE.name} is not what the system says now; {update}"
    )
    drawn = sorted(tmp_path.glob("*.svg"))
    used = {svg.name for light, _ in _themed_figures(_PAGE) for svg in (light,)}
    assert {svg.name for svg in drawn if not svg.stem.endswith("-dark")} == used
    for svg in drawn:
        committed = _SCRIPT.parent / svg.name
        assert committed.exists(), f"{_SCRIPT.name} draws {svg.name}, which is not committed"
        assert same_figure(svg.read_text(), committed.read_text()), (
            f"{svg.name} is not what {_SCRIPT.name} draws now; {update}"
        )
