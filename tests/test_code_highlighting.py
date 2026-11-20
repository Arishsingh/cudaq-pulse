# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The docs' code colors are legible and tell the kinds of token apart.

``docs/_ext/code_style.py`` colors highlighted code in each theme. Every
color must reach 4.5:1 (WCAG AA) on that theme's code surface, which must
be the surface the pinned theme draws; the kinds of token must have colors
of their own; and the stylesheet must give Python's tokens those colors.
``docs/_static/a11y-print.css`` writes the light theme's colors out again,
for a page printed from the dark theme, and must keep to them.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("pygments")

_EXT = Path(__file__).resolve().parents[1] / "docs" / "_ext"
pytestmark = pytest.mark.skipif(not _EXT.exists(), reason="needs the source tree")
sys.path.insert(0, str(_EXT))
code_style = pytest.importorskip("code_style")
sys.path.remove(str(_EXT))

_PRINT_SHEET = _EXT.parent / "_static" / "a11y-print.css"

_ROLES = (
    "text",
    "comment",
    "keyword",
    "function",
    "builtin",
    "decorator",
    "string",
    "number",
    "operator",
)


def _luminance(color: str) -> float:
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_every_color_is_aa_on_the_code_surface(theme):
    surface = code_style.SURFACES[theme]
    low = {
        role: round(_contrast(color, surface), 2)
        for role, color in code_style.PALETTES[theme].items()
        if _contrast(color, surface) < 4.5
    }
    assert not low, f"below 4.5:1 on {surface}: {low}"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_each_kind_of_token_has_its_own_color(theme):
    palette = code_style.PALETTES[theme]
    assert len({palette[role] for role in _ROLES}) == len(_ROLES)


def test_the_surfaces_are_the_themes():
    spec = importlib.util.find_spec("nvidia_sphinx_theme")
    if spec is None:
        pytest.skip("needs nvidia-sphinx-theme")
    css = next(Path(spec.origin).parent.rglob("nvidia-sphinx-theme.css")).read_text()
    for theme, surface in code_style.SURFACES.items():
        block = re.search(r"html\[data-theme=" + theme + r"\]\{([^}]*)\}", css)
        assert block, theme
        assert f"--pst-color-surface:{surface}" in block.group(1), theme


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_stylesheet_colors_pythons_tokens(theme):
    css = code_style.stylesheet()
    palette = code_style.PALETTES[theme]
    expected = {
        "k": "keyword",
        "kn": "keyword",
        "ow": "keyword",
        "n": "text",
        "nn": "text",
        "nf": "function",
        "nc": "function",
        "nb": "builtin",
        "bp": "builtin",
        "nd": "decorator",
        "s2": "string",
        "sd": "string",
        "mi": "number",
        "mf": "number",
        "kc": "number",
        "c1": "comment",
        "o": "operator",
        "p": "operator",
    }
    for cls, role in expected.items():
        rule = re.search(rf'html\[data-theme="{theme}"\] \.highlight \.{cls} \{{([^}}]*)\}}', css)
        assert rule, cls
        assert f"color: {palette[role]}" in rule.group(1).lower(), (cls, rule.group(1))


def _printed() -> list[tuple[list[str], str]]:
    """The print stylesheet's rules for highlighted code: the classes of
    token each names, none for the rule of the text, and its color."""
    sheet = re.sub(r"/\*.*?\*/", "", _PRINT_SHEET.read_text(), flags=re.S)
    return [
        ([cls for cls in re.findall(r"\.([\w-]+)", selector) if cls != "highlight"], color.lower())
        for selector, body in re.findall(r"([^{}]*\.highlight(?![\w-])[^{}]*)\{([^{}]*)\}", sheet)
        for color in re.findall(r"#[0-9a-fA-F]{6}\b", body)
    ]


def test_the_print_stylesheet_has_the_light_palette():
    # Paper is white, so a page printed from the dark theme takes the light
    # theme's code colors. The stylesheet is a file of its own and repeats
    # them by hand.
    light = {*code_style.PALETTES["light"].values(), code_style._HIGHLIGHTED_LINES["light"]}
    printed = {color for _, color in _printed()}
    lacking, stale = sorted(light - printed), sorted(printed - light)
    assert not lacking, f"docs/_static/a11y-print.css lacks {lacking} of the light palette"
    assert not stale, f"docs/_static/a11y-print.css colors code {stale}, not in the light palette"


def test_the_print_stylesheet_colors_each_token_as_the_light_theme():
    light = {
        cls: color.lower()
        for cls, color in re.findall(
            r'html\[data-theme="light"\] \.highlight \.(\w+) \{[^}]*?(#[0-9a-fA-F]{6})',
            code_style.stylesheet(),
        )
    }
    assert light, "found no rule for the light theme in the stylesheet"
    rules = _printed()
    # A token that no rule names prints in the color of the rule for all code.
    text = [color for classes, color in rules if not classes]
    assert len(text) == 1, f"the print stylesheet has {len(text)} rules for the text of code"
    named = {cls: color for classes, color in rules for cls in classes}
    wrong = {
        cls: (named.get(cls, text[0]), color)
        for cls, color in light.items()
        if named.get(cls, text[0]) != color
    }
    assert not wrong, f"tokens by class, as (printed, in the light theme): {wrong}"
