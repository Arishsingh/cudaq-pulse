# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The docs' code colors are legible and tell the kinds of token apart.

``docs/_ext/code_style.py`` colors highlighted code in each theme. Every
color must reach 4.5:1 (WCAG AA) on that theme's code surface, which must
be the surface the pinned theme draws; the kinds of token must have colors
of their own; and the stylesheet must give Python's tokens those colors.
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
