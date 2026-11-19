# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration for the CUDA-Q Pulse documentation."""

import sys
from pathlib import Path

from cudaq_pulse._build_config import PACKAGE_VERSION
from pygments.lexers.special import TextLexer

project = "CUDA-Q Pulse"
copyright = "2026, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA Corporation"
release = PACKAGE_VERSION

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

extensions = [
    "code_style",
    "themed_figure",
    "pulse_visual",
    "data_figures",
    "a11y",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "myst_parser",
]

extlinks = {
    "example": ("https://github.com/NVIDIA/cudaq-pulse/blob/main/examples/%s", "examples/%s"),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["tutorial-output.css", "header.css"]
html_favicon = "_static/favicon.svg"

html_theme_options = {
    "show_nav_level": 1,
    "navigation_depth": 4,
    "show_toc_level": 2,
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True


def _own_docstrings_only(app, what, name, obj, options, lines):
    # pulse.Length is typing.Annotated: its page describes it, not typing's
    # docstring.
    if name == "cudaq_pulse.Length":
        lines.clear()


def _waveform_explorer(app, pagename, templatename, context, doctree):
    # Files added here load on this page only: Sphinx resets them per page.
    if pagename == "user_guide/operations":
        app.add_css_file("waveform-explorer.css")
        app.add_js_file("waveform-explorer.js", defer="defer")


def setup(app):
    # The generated dialect reference fences its examples as ```mlir, which
    # Pygments has no lexer for.
    app.add_lexer("mlir", TextLexer)
    app.connect("autodoc-process-docstring", _own_docstrings_only)
    app.connect("html-page-context", _waveform_explorer)
