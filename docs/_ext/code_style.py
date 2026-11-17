# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The colors of highlighted code, one Pygments style per theme.

The theme's own styles give most of a Python block one or two colors: every
name in the same purple, function names in the string color, builtins and
decorators in the number color, and in the dark theme comments in nearly the
text color. These two styles give each kind of token a hue of its own, the
same hue in both themes, each at least 4.5:1 (WCAG AA) on the code surface
of its theme. They go to ``_static/code-highlight.css``, which loads after
the theme's ``pygments.css`` and so replaces its colors.
"""

from __future__ import annotations

from pathlib import Path

from pygments.formatters import HtmlFormatter
from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
)

# The theme's background of a code block, which the styles' colors sit on.
SURFACES = {"light": "#f7f7f7", "dark": "#1a1a1a"}

_HIGHLIGHTED_LINES = {"light": "#fff3b0", "dark": "#33301c"}

PALETTES = {
    "light": {
        "text": "#1a1a1a",
        "comment": "#5c636b",
        "keyword": "#7b30b8",
        "function": "#0b58b8",
        "builtin": "#00707a",
        "decorator": "#875c00",
        "string": "#2e7300",
        "number": "#b03c00",
        "operator": "#a8336d",
        "error": "#b3261e",
    },
    "dark": {
        "text": "#e6e6e6",
        "comment": "#9ba3ad",
        "keyword": "#c99cff",
        "function": "#6db3ff",
        "builtin": "#4fcfcf",
        "decorator": "#e8c15a",
        "string": "#9bd35d",
        "number": "#ff9b6b",
        "operator": "#f08dbf",
        "error": "#ff8a80",
    },
}


def _style(theme: str) -> type[Style]:
    c = PALETTES[theme]
    styles = {
        Token: c["text"],
        Comment: f"italic {c['comment']}",
        Keyword: c["keyword"],
        Keyword.Constant: c["number"],
        Operator: c["operator"],
        Operator.Word: c["keyword"],
        Punctuation: c["operator"],
        Name: c["text"],
        Name.Function: c["function"],
        Name.Class: c["function"],
        Name.Exception: c["function"],
        Name.Builtin: c["builtin"],
        Name.Variable.Magic: c["builtin"],
        Name.Decorator: c["decorator"],
        Name.Attribute: c["text"],
        Name.Tag: c["keyword"],
        String: c["string"],
        String.Escape: c["operator"],
        String.Interpol: c["operator"],
        Number: c["number"],
        Generic.Deleted: c["error"],
        Generic.Error: c["error"],
        Generic.Traceback: c["error"],
        Generic.Inserted: c["string"],
        Generic.Heading: f"bold {c['function']}",
        Generic.Subheading: c["function"],
        Generic.Prompt: f"bold {c['comment']}",
        Generic.Output: c["text"],
        Generic.Emph: "italic",
        Generic.Strong: "bold",
        Error: c["error"],
    }
    # The theme paints the surface on the ``pre``; around it, and under a
    # tutorial's output blocks, the page shows through.
    return type(
        f"{theme.capitalize()}CodeStyle",
        (Style,),
        {
            "background_color": "transparent",
            "highlight_color": _HIGHLIGHTED_LINES[theme],
            "styles": styles,
        },
    )


def stylesheet() -> str:
    return "".join(
        HtmlFormatter(style=_style(theme)).get_style_defs(f'html[data-theme="{theme}"] .highlight')
        + "\n"
        for theme in PALETTES
    )


def _write_stylesheet(app):
    # Written before the pages, so that their links carry its checksum.
    if app.builder.format == "html":
        path = Path(app.builder.outdir) / "_static" / "code-highlight.css"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stylesheet())


def setup(app):
    app.add_css_file("code-highlight.css")
    app.connect("builder-inited", _write_stylesheet)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
