# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accessibility repairs for what the theme and Sphinx generate.

* The sidebar's expand/collapse buttons get a name ("Pulse Kernels
  subsections"), since the theme leaves each ``<summary>`` empty.
* Multi-line signatures, which Sphinx writes as a ``<dl>`` of ``<dd>`` with no
  ``<dt>`` (an invalid list), are written as plain blocks.
* A figure or image whose alt text is a paragraph gets a short alt text (its
  first sentence) and the full text as a "Figure description" disclosure
  under it, which also serves readers who see the figure but cannot read
  its small text.
* Every figure image that is not already a link links to the file itself, to open
  full size and zoom.
* ``404.html`` resolves its links and styles from the site root.
"""

from __future__ import annotations

import html
import re

from docutils import nodes

SHORT_ALT = 140
SITE_ROOT = "/cudaq-pulse/"

_TOGGLE = re.compile(
    r'(<a class="reference internal"[^>]*>)(.*?)(</a>\s*<details>\s*)<summary>', re.S
)
_SIG_DL = re.compile(
    r"<dl>\s*(<dd><em class=\"sig-param\".*?)</dl>(\s*<span class=\"sig-paren\">)", re.S
)
_FIGURE = re.compile(r"<figure\b.*?</figure>", re.S)
_IMG = re.compile(r"(<a\b[^>]*>\s*)?(<img\b[^>]*?\bsrc=\"([^\"]+)\"[^>]*>)")


def _sig_lines(m: re.Match[str]) -> str:
    lines = m.group(1).replace("<dd>", '<div class="sig-param-line">').replace("</dd>", "</div>")
    return f'<div class="sig-params">{lines}</div>{m.group(2)}'


def _plain(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def _name_toggles(page):
    if not isinstance(page, str):  # the theme also returns a parsed tree
        for summary in page.find_all("summary"):
            details = summary.parent
            link = details.find_previous_sibling("a") if details is not None else None
            if link is not None and not summary.get("aria-label"):
                summary["aria-label"] = _plain(link.get_text()) + " subsections"
        return page

    def add(m: re.Match[str]) -> str:
        label = html.escape(_plain(m.group(2)) + " subsections", quote=True)
        return f'{m.group(1)}{m.group(2)}{m.group(3)}<summary aria-label="{label}">'

    return _TOGGLE.sub(add, page)


def _short(alt: str) -> str:
    first = re.match(rf"(.{{30,{SHORT_ALT}}}?[.!?])(\s|$)", alt)
    if first:
        return first.group(1)
    return alt[:SHORT_ALT].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _describe(app, doctree, docname):
    for image in list(doctree.findall(nodes.image)):
        alt = image.get("alt", "")
        if len(alt) <= SHORT_ALT + 20 or image.get("a11y-described"):
            continue
        parent = image.parent
        holder = parent if isinstance(parent, nodes.figure) else None
        if holder is None and isinstance(parent, nodes.container):
            holder = parent.parent if isinstance(parent.parent, nodes.figure) else None
        siblings = [i for i in (holder or parent).findall(nodes.image) if i.get("alt") == alt]
        for i in siblings:
            i["alt"] = _short(alt)
            i["a11y-described"] = True
        details = nodes.raw(
            "",
            '<details class="figure-description"><summary>Figure description</summary>'
            f"<p>{html.escape(alt)}</p></details>",
            format="html",
        )
        if holder is not None:
            holder += details
        elif isinstance(parent, (nodes.section, nodes.document)):
            parent.insert(parent.index(image) + 1, details)


def _page_context(app, pagename, templatename, context, doctree):
    toctree = context.get("generate_toctree_html")
    if toctree is not None and not getattr(toctree, "_a11y", False):

        def generate(*args, **kwargs):
            return _name_toggles(toctree(*args, **kwargs))

        generate._a11y = True  # type: ignore[attr-defined]
        context["generate_toctree_html"] = generate
    body = context.get("body")
    if body:
        body = _SIG_DL.sub(_sig_lines, body)

        def zoom(figure: re.Match[str]) -> str:
            def wrap(img: re.Match[str]) -> str:
                if img.group(1):  # already a link; a link cannot hold a link
                    return img.group(0)
                return (
                    f'<a class="figure-zoom" href="{img.group(3)}" '
                    f'title="Open the figure full size">{img.group(2)}</a>'
                )

            return _IMG.sub(wrap, figure.group(0))

        context["body"] = _FIGURE.sub(zoom, body)
    if pagename == "404":
        context["metatags"] = context.get("metatags", "") + f'<base href="{SITE_ROOT}">'


def setup(app):
    app.add_css_file("a11y.css")
    app.add_js_file("a11y.js", defer="defer")
    app.connect("doctree-resolved", _describe)
    app.connect("html-page-context", _page_context, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
