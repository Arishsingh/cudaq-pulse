# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every tutorial page runs, and prints what its comments say it prints.

A page's Python code blocks run in order in one namespace, as a reader who
pastes them would run them. A ``print(...)`` line that ends in a comment
must print that comment's first words (``# 3.885, its 0-1 transition``
checks ``3.885``); a block that prints from a loop instead lists what it
prints as comment lines after its last line of code.

Blocks with ``:class: fragment`` are not run. Blocks with ``:class: gpu``
run only where the GPU runtime loads, ``:class: viz`` only with matplotlib,
and ``:class: cudaq-qec`` only with CUDA-Q QEC; what they print is not
checked on other machines.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import textwrap
from pathlib import Path

import pytest

_TUTORIALS = Path(__file__).resolve().parents[1] / "docs" / "tutorials"
_PAGES = sorted(p for p in _TUTORIALS.glob("*.rst") if p.name != "index.rst")
_PRINT_WITH_VALUE = re.compile(r"^\s*print\(.*\)\s*#\s*(?P<value>.+)$")


def code_blocks(page: Path) -> list[tuple[frozenset[str], str]]:
    """Each ``.. code-block:: python`` of *page*: its classes and its code."""
    lines = page.read_text().splitlines()
    blocks, i = [], 0
    while i < len(lines):
        if re.match(r"\s*\.\. code-block:: python\s*$", lines[i]):
            i += 1
            body, classes = [], frozenset()
            while i < len(lines) and (not lines[i].strip() or lines[i][:1] in " \t"):
                option = lines[i].split()
                if option[:1] == [":class:"]:
                    classes = frozenset(option[1:])
                elif not (option and option[0].startswith(":") and option[0].endswith(":")):
                    body.append(lines[i])
                i += 1
            blocks.append((classes, textwrap.dedent("\n".join(body)).strip()))
        else:
            i += 1
    return blocks


def expected_output(code: str) -> list[str]:
    """What *code* says it prints, in order.

    A top-level print's trailing comment is one line of output. Comment
    lines at the left margin right after an indented body (a loop that
    prints) are that loop's output, a line each.
    """
    expected, after_body = [], False
    for line in code.splitlines():
        if not line.strip():
            after_body = False
            continue
        if line.startswith("# ") and after_body:
            expected.append(line[2:])
            continue
        indented = line[:1] in " \t"
        if not indented and (m := _PRINT_WITH_VALUE.match(line)):
            expected.append(m["value"])
        after_body = indented
    return expected


def _matches(expected: str, printed: str) -> bool:
    return expected == printed or expected.startswith((printed + ",", printed + " ", printed + ":"))


def _gpu_available() -> bool:
    from cudaq_pulse.runtime import require_gpu_runtime

    try:
        require_gpu_runtime()
    except Exception:
        return False
    return True


def _runnable(classes: frozenset[str]) -> bool:
    if "fragment" in classes:
        return False
    if "viz" in classes and importlib.util.find_spec("matplotlib") is None:
        return False
    if "cudaq-qec" in classes and importlib.util.find_spec("cudaq_qec") is None:
        return False
    return not ({"gpu", "cudaq-qec"} & classes) or _gpu_available()


def test_there_are_tutorials():
    if not _TUTORIALS.exists():
        pytest.skip("needs the source tree")
    assert len(_PAGES) >= 2


@pytest.mark.slow
@pytest.mark.parametrize("page", _PAGES, ids=lambda p: p.stem)
def test_tutorial_runs_and_prints_its_comments(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.delenv("CUDAQ_PULSE_QUANTUM_SYSTEM", raising=False)
    namespace: dict = {"__name__": f"tutorial_{page.stem}"}
    blocks = code_blocks(page)
    assert blocks
    for number, (classes, code) in enumerate(blocks, 1):
        if not _runnable(classes):
            continue
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, f"{page.name}, block {number}", "exec"), namespace)
        printed = out.getvalue().splitlines()
        for k, want in enumerate(expected_output(code)):
            got = printed[k] if k < len(printed) else "<nothing printed>"
            assert _matches(want, got), (
                f"{page.name}, block {number}: printed {got!r}, says {want!r}"
            )
