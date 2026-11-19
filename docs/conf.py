# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration for the CUDA-Q Pulse documentation."""

import sys
from pathlib import Path

import sphinx
from cudaq_pulse._build_config import PACKAGE_VERSION
from sphinx.errors import ConfigError

project = "CUDA-Q Pulse"
copyright = "2026, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA Corporation"
release = PACKAGE_VERSION

# The API reference clean-up in _ext/api_docs.py reads what autodoc records
# about the entry it is writing, which Sphinx keeps from 8.2 on. The check is
# made here by number: needs_sphinx compares versions as text, and "10" would
# count as older than "8.2".
if sphinx.version_info < (8, 2):
    raise ConfigError("These docs need Sphinx 8.2 or later.")

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

extensions = [
    "code_style",
    "themed_figure",
    "pulse_visual",
    "data_figures",
    "a11y",
    "api_docs",
    "api_links",
    "related_pages",
    "dialect_pages",
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


def _waveform_explorer(app, pagename, templatename, context, doctree):
    # Files added here load on this page only: Sphinx resets them per page.
    if pagename == "user_guide/operations":
        app.add_css_file("waveform-explorer.css")
        app.add_js_file("waveform-explorer.js", defer="defer")


def setup(app):
    app.connect("html-page-context", _waveform_explorer)
