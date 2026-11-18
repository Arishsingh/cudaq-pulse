# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each tutorial's code is one script, written from the page, linked from it,
and runs.

``tools/tutorial-scripts.py`` writes docs/tutorials/scripts/ from the pages;
a page edited without running it leaves its script stale, which fails here.
Each script runs as a reader runs it, ``python <script>``, in an empty
directory: its GPU, matplotlib and CUDA-Q QEC blocks are skipped where those
are missing.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "tutorial-scripts.py"

pytestmark = pytest.mark.skipif(not _TOOL.is_file(), reason="needs the source tree")


def _tool():
    spec = importlib.util.spec_from_file_location("tutorial_scripts", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pages() -> list[Path]:
    return _tool().pages() if _TOOL.is_file() else []


def test_every_script_is_written_from_its_page():
    tool = _tool()
    pages = tool.pages()
    assert pages
    stale = [p.name for p in pages if (tool.SCRIPTS / f"{p.stem}.py").read_text() != tool.script(p)]
    assert stale == [], "run python tools/tutorial-scripts.py"
    assert sorted(p.stem for p in tool.SCRIPTS.glob("*.py")) == [p.stem for p in pages]


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.stem)
def test_the_page_links_its_script(page):
    assert f":download:`{page.stem}.py <scripts/{page.stem}.py>`" in page.read_text()


@pytest.mark.slow
@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.stem)
def test_the_script_runs(page, tmp_path):
    script = _tool().SCRIPTS / f"{page.stem}.py"
    env = {k: v for k, v in os.environ.items() if k != "CUDAQ_PULSE_QUANTUM_SYSTEM"}
    env["MPLBACKEND"] = "Agg"
    # The cudaq_pulse this test imports, installed or from a build tree.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout
