# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every tutorial page runs, and prints what it says it prints.

A page's Python code blocks run in order in one namespace, as a reader who
pastes them would run them. An output block right after a code block
(``.. code-block:: text`` with ``:class: output``) is what that block
prints, line for line. Otherwise, a ``print(...)`` line that ends in a comment
must print that comment's first words (``# 3.885, its 0-1 transition``
checks ``3.885``); a block that prints from a loop instead lists what it
prints as comment lines after its last line of code.

A ``literalinclude`` of Python, cut by ``:start-after:`` and
``:end-before:``, runs like a code block, so code a page shows from an
example file runs as the page shows it. One cut by ``:pyobject:``,
``:start-at:``, ``:end-at:`` or ``:lines:`` shows a piece of an example
module the page imports, and is not run. A block with ``:class: continues``
runs joined to the next one, as a class shown a method at a time.

Blocks with ``:class: fragment`` are not run. Blocks with ``:class: gpu``
run only where the GPU runtime loads, ``:class: viz`` only with matplotlib,
and ``:class: cudaq-qec`` only with CUDA-Q QEC; what they print is not
checked on other machines. A page may import the example modules it links
with ``:example:``, as a reader who saves them beside the page's code would.
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
_EXAMPLES = _TUTORIALS.parents[1] / "examples"
_EXAMPLE_LINK = re.compile(r":example:`(?P<path>[^`<>]+\.py)`")
_PIECE = {"pyobject", "start-at", "end-at", "lines"}


def included(page: Path, target: str, options: dict[str, str]) -> str:
    """The code a ``literalinclude`` of *target* with *options* shows, as Sphinx cuts it."""
    lines = (page.parent / target).read_text().splitlines()
    if "start-after" in options:
        lines = lines[next(i for i, ln in enumerate(lines) if options["start-after"] in ln) + 1 :]
    if "end-before" in options:
        lines = lines[: next(i for i, ln in enumerate(lines) if options["end-before"] in ln)]
    return "\n".join(lines)


def code_blocks(page: Path) -> list[tuple[frozenset[str], str, list[str] | None]]:
    """Each ``.. code-block:: python`` and Python ``literalinclude`` of *page*,
    in order: its classes, its code, and the lines of the output block that
    follows it (``.. code-block:: text`` with ``:class: output``), if any."""
    lines = page.read_text().splitlines()
    blocks: list[list] = []
    i = 0
    while i < len(lines):
        block = re.match(r"\s*\.\. code-block:: (python|text)\s*$", lines[i])
        include = re.match(r"\s*\.\. literalinclude:: (\S+)\s*$", lines[i])
        if block or include:
            i += 1
            body, classes, options = [], frozenset(), {}
            while i < len(lines) and (not lines[i].strip() or lines[i][:1] in " \t"):
                option = lines[i].split(maxsplit=1)
                if option[:1] == [":class:"]:
                    classes = frozenset(option[1].split())
                elif option and option[0].startswith(":") and option[0].endswith(":"):
                    options[option[0].strip(":")] = option[1].strip() if len(option) > 1 else ""
                else:
                    body.append(lines[i])
                i += 1
            if block and block[1] == "text":
                if "output" in classes and blocks and blocks[-1][2] is None:
                    shown = textwrap.dedent("\n".join(body)).strip("\n").splitlines()
                    blocks[-1][2] = [line.rstrip() for line in shown]
                continue
            if include:
                # Cut any other way, it shows a piece of an example module
                # the page imports; the module runs it, not the page.
                if options.get("language") != "python" or _PIECE & options.keys():
                    continue
                body = [included(page, include[1], options)]
            blocks.append([classes, "\n".join(body), None])
        else:
            i += 1
    # A piece with the class "continues" shows the start of code the next block
    # finishes, such as a class shown a method at a time: they run as one.
    joined: list[tuple[frozenset[str], str, list[str] | None]] = []
    pending = ""
    for classes, code, shown in blocks:
        if "continues" in classes:
            pending += code + "\n"
            continue
        joined.append((classes, textwrap.dedent(pending + code).strip(), shown))
        pending = ""
    return joined


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


def test_an_output_block_belongs_to_the_code_block_before_it(tmp_path):
    page = tmp_path / "page.rst"
    page.write_text(
        ".. code-block:: python\n\n   print(1)\n   print('two')\n\n"
        ".. code-block:: text\n   :class: output\n\n   1\n   two\n\n"
        ".. code-block:: text\n\n   not output\n\n"
        ".. code-block:: python\n\n   x = 3\n"
    )
    assert code_blocks(page) == [
        (frozenset(), "print(1)\nprint('two')", ["1", "two"]),
        (frozenset(), "x = 3", None),
    ]


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
    for linked in sorted({m["path"] for m in _EXAMPLE_LINK.finditer(page.read_text())}):
        monkeypatch.syspath_prepend(str((_EXAMPLES / linked).parent))
    namespace: dict = {"__name__": f"tutorial_{page.stem}"}
    blocks = code_blocks(page)
    assert blocks
    for number, (classes, code, shown) in enumerate(blocks, 1):
        if not _runnable(classes):
            continue
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            # dont_inherit: a reader's code runs without this module's
            # ``from __future__ import annotations``.
            code_object = compile(code, f"{page.name}, block {number}", "exec", dont_inherit=True)
            exec(code_object, namespace)
        printed = [line.rstrip() for line in out.getvalue().splitlines()]
        if shown is not None:
            assert printed == shown, (
                f"{page.name}, block {number}: printed {printed!r}, "
                f"its output block shows {shown!r}"
            )
            continue
        for k, want in enumerate(expected_output(code)):
            got = printed[k] if k < len(printed) else "<nothing printed>"
            assert _matches(want, got), (
                f"{page.name}, block {number}: printed {got!r}, says {want!r}"
            )
