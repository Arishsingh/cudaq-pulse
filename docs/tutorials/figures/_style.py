# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The look of the tutorial figures, how they are saved, and the page code they plot."""

import contextlib
import io
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE = "#4a7fb5"
GREEN = "#76b900"
GREY = "#6b6b6b"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "svg.fonttype": "none",
        "svg.hashsalt": "cudaq-pulse",
    }
)


def page_namespace(page: str) -> dict:
    """Run the CPU code blocks of tutorial *page*, as a reader would, and return their names."""
    lines = (HERE.parent / page).read_text().splitlines()
    namespace: dict = {"__name__": "tutorial"}
    i = 0
    while i < len(lines):
        if re.match(r"\s*\.\. code-block:: python\s*$", lines[i]):
            i += 1
            body, classes = [], []
            while i < len(lines) and (not lines[i].strip() or lines[i][:1] in " \t"):
                option = lines[i].split()
                if option[:1] == [":class:"]:
                    classes = option[1:]
                elif not (option and option[0].startswith(":") and option[0].endswith(":")):
                    body.append(lines[i])
                i += 1
            if not {"fragment", "gpu", "viz", "cudaq-qec"} & set(classes):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(textwrap.dedent("\n".join(body)), namespace)
        else:
            i += 1
    return namespace


def axes(figsize=(6.4, 3.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def save(fig, name: str) -> None:
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    plt.close(fig)
    # The repository's pre-commit hooks reject trailing whitespace.
    text = "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n"
    (HERE / name).write_text(text)
