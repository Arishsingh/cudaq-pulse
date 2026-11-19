# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write each tutorial's code as one Python script, in docs/tutorials/scripts/.

Usage: python tools/tutorial-scripts.py [--check]

A script holds the code its page runs, read the way tests/test_tutorials.py
reads it: the page's Python code blocks and literalincludes in order, those
with the class ``continues`` joined to the next, and none with ``fragment``.
Each block starts a ``# %%`` cell named for its section, and what the page
shows it printing follows it as comments. Blocks the page runs only with a
GPU (``gpu``), with matplotlib (``viz``) or with CUDA-Q QEC (``cudaq-qec``)
run under an ``if`` that is false without them, so the script runs anywhere.
A page that imports an example module it links finds it in the clone's
examples/ or beside the script.

With --check, write nothing and exit 1 if a script is missing, stale, or
has no page.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = _ROOT / "docs" / "tutorials"
SCRIPTS = TUTORIALS / "scripts"
_SITE = "https://nvidia.github.io/cudaq-pulse/tutorials"
_EXAMPLE_LINK = re.compile(r":example:`(?P<path>[^`<>]+\.py)`")
_HEADING = re.compile(r"^(=+|-+|~+|\^+)$")
_DIRECTIVE = re.compile(r"\s*\.\. (?:code-block:: (python)|literalinclude:: \S+)\s*$")

_HEADER = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
\"\"\"{title}: the code of the tutorial, as one script.

{url}

Generated from docs/tutorials/{page}.rst by tools/tutorial-scripts.py: edit
the page, not this file.
\"\"\"
"""

_NEEDS = {
    "gpu": (
        "GPU",
        "Blocks under ``if GPU:`` need the GPU runtime (Linux, an NVIDIA GPU and\n"
        '# ``pip install "cudaq-pulse[cu12]"`` or ``[cu13]``).',
        "def _gpu_runtime_loads() -> bool:\n"
        "    from cudaq_pulse.runtime import require_gpu_runtime\n\n"
        "    try:\n"
        "        require_gpu_runtime()\n"
        "    except Exception:\n"
        "        return False\n"
        "    return True\n\n\n"
        "GPU = _gpu_runtime_loads()",
    ),
    "viz": (
        "VIZ",
        'Blocks under ``if VIZ:`` need matplotlib (``pip install "cudaq-pulse[viz]"``).',
        'VIZ = importlib.util.find_spec("matplotlib") is not None',
    ),
    "cudaq-qec": (
        "CUDAQ_QEC",
        "Blocks under ``if CUDAQ_QEC:`` need CUDA-Q QEC (``pip install cudaq-qec``)\n"
        "# and the GPU runtime.",
        'CUDAQ_QEC = GPU and importlib.util.find_spec("cudaq_qec") is not None',
    ),
}


def _tutorial_test():
    test = _ROOT / "tests" / "test_tutorials.py"
    spec = importlib.util.spec_from_file_location("_tutorial_pages", test)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pages() -> list[Path]:
    return sorted(p for p in TUTORIALS.glob("*.rst") if p.name != "index.rst")


def _title_and_sections(page: Path) -> tuple[str, list[str]]:
    """The page's title, and the section each Python block that runs as code
    (each ``code_blocks`` entry) sits in, in order."""
    lines = page.read_text().splitlines()
    title, section, sections, joined = "", "", [], False
    for i, line in enumerate(lines):
        if i and _HEADING.match(line) and len(line) >= len(lines[i - 1].strip()) > 0:
            if not lines[i - 1].startswith(" "):
                section = lines[i - 1].strip()
                title = title or section
            continue
        directive = _DIRECTIVE.match(line)
        if not directive:
            continue
        options = {}
        for option in lines[i + 1 :]:
            if not option.strip().startswith(":"):
                break
            name, _, value = option.strip()[1:].partition(":")
            options[name] = value.strip()
        if not directive[1] and options.get("language") != "python":
            continue
        if not joined:
            sections.append(section)
        joined = "continues" in options.get("class", "").split()
    return title, sections


def script(page: Path) -> str:
    """The script of *page*."""
    blocks = _tutorial_test().code_blocks(page)
    title, sections = _title_and_sections(page)
    assert len(sections) == len(blocks), f"{page.name}: {len(sections)} sections, {len(blocks)}"
    text = page.read_text()

    cells: list[str] = []
    needs: set[str] = set()
    last_section = None
    for (classes, code, shown), section in zip(blocks, sections, strict=True):
        if "fragment" in classes:
            continue
        block_needs = [need for need in _NEEDS if need in classes]
        if "cudaq-qec" in block_needs:
            block_needs = ["cudaq-qec"]
        needs.update(block_needs)
        guards = [_NEEDS[need][0] for need in block_needs]
        if guards:
            code = f"if {' and '.join(guards)}:\n" + textwrap.indent(code, "    ")
        if shown:
            # An abridged block is not the printout line for line.
            says = (
                "Prints, wrapped here, with ... for text left out"
                if "abridged" in classes
                else "Prints"
            )
            code += f"\n\n# {says}:\n" + "\n".join(f"#   {line}".rstrip() for line in shown)
        if section != last_section:
            code = f"# %% {section}\n{code}"
            last_section = section
        cells.append(code)

    preamble = []
    imported = []
    for linked in sorted({m["path"] for m in _EXAMPLE_LINK.finditer(text)}):
        module = Path(linked).stem
        if re.search(rf"^\s*(?:from {module} import|import {module}\b)", text, flags=re.M):
            imported.append((module, Path(linked).parent.as_posix()))
    if imported:
        names = " and ".join(f"examples/{d}/{m}.py" for m, d in imported)
        # In a clone the script is docs/tutorials/scripts/<page>.py, so its
        # fourth parent is the clone. .parent stops at the root, so a script
        # saved higher up, as ~/<page>.py, still runs; parents[3] would raise
        # IndexError there.
        lines = [
            f"# The page imports {names}. Run this script from a",
            "# clone of the repository, or save that file beside it.",
            '_EXAMPLES = Path(__file__).resolve().parent.parent.parent.parent / "examples"',
        ]
        for directory in dict.fromkeys(d for _, d in imported):
            lines.append(f'sys.path.insert(0, str(_EXAMPLES / "{directory}"))')
        preamble.append("\n".join(lines))
    # CUDA-Q QEC's guard reads the GPU's.
    if "cudaq-qec" in needs:
        needs.add("gpu")
    for need in (n for n in _NEEDS if n in needs):
        _, note, definition = _NEEDS[need]
        preamble.append(f"# {note}\n{definition}")

    imports = []
    if any(n in needs for n in ("viz", "cudaq-qec")):
        imports.append("import importlib.util")
    if imported:
        imports += ["import sys", "from pathlib import Path"]
    head = _HEADER.format(title=title, url=f"{_SITE}/{page.stem}.html", page=page.stem)
    parts = preamble + cells
    if imports:
        parts.insert(0, "\n".join(sorted(imports, key=lambda s: (s.startswith("from"), s))))
    return head + "\n" + "\n\n\n".join(p.rstrip("\n") for p in parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only check the scripts are current")
    args = parser.parse_args()
    wanted = {SCRIPTS / f"{page.stem}.py": script(page) for page in pages()}
    problems = []
    for path, text in wanted.items():
        if path.is_file() and path.read_text() == text:
            continue
        if args.check:
            state = "stale" if path.exists() else "missing"
            problems.append(f"{path.relative_to(_ROOT)} is {state}")
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
            print(f"wrote {path.relative_to(_ROOT)}")
    for path in sorted(SCRIPTS.glob("*.py")):
        if path not in wanted:
            problems.append(f"{path.relative_to(_ROOT)} has no tutorial page")
    for problem in problems:
        print(f"FAIL {problem}: run python tools/tutorial-scripts.py")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
