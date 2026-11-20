# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accessibility repairs for what the theme and Sphinx generate.

* The sidebar's expand/collapse buttons get a name ("Pulse Kernels
  subsections"), since the theme leaves each ``<summary>`` empty.
* Multi-line signatures, which Sphinx writes as a ``<dl>`` of ``<dd>`` with no
  ``<dt>`` (an invalid list), are written as plain blocks.
* A figure or image whose alt text is a paragraph gets a short alt text (its
  first sentence or what stands before its first colon, or else the title
  its SVG file carries) and the full text as a "Figure description"
  disclosure under it, which also serves readers who see the figure but
  cannot read its small text. Each disclosure is named after its figure, and
  the figure's images point to it.
* The "#" link after a figure's or a table's caption is named after that
  caption, since "#" alone does not say which figure it links to.
* A figure has an anchor made of the name of its image's file,
  ``fig-rabi-pulse``, and the link after its caption leads to that. Docutils
  numbers the figures in page order, ``id2``, so a new figure would move the
  anchor of every later one. The number stays as a second anchor: a link
  made before still arrives.
* Images are not links: clicking a figure should not navigate away or open a
  new tab, so Sphinx does not link a scaled image to its file, and a
  ``:target:`` on an image is dropped.
* What a tutorial's code prints (``:class: output``) is labelled "Output" in
  the text of the page.
* A figure with no caption ends without the ``</figcaption>`` that docutils
  writes for any figure with more than one part.
* A power written as a digit and a superscript, ``2`` and ``40``, reads
  "2 to the power 40" where the superscript is not announced.
* Without JavaScript, a narrow screen shows the site navigation below the
  page, since the button that opens it needs a script, and the search page
  shows its notice without the field, which cannot search.
* The header's link to the home page is named once: the logo, then the title.
* ``404.html`` resolves its links and styles from the site root, and its
  links to its own anchors, the skip link among them, stay on that page.
"""

from __future__ import annotations

import html
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from docutils import nodes

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.config import Config

SHORT_ALT = 140
# The longest name the link after a caption takes from the caption.
SHORT_NAME = 70
SITE_ROOT = "/cudaq-pulse/"
LOGO_ALT = "NVIDIA"

# A full stop this early is more likely an abbreviation's than a sentence's.
_SENTENCE = re.compile(rf"(.{{14,{SHORT_ALT}}}?[.!?])(?:\s|$)", re.S)
# A break after a digit would cut a list of values ("qudit 0: 0-1 at 5.10").
_COLON = re.compile(r"(?<!\d): ")
_PAUSE = re.compile(r"(?<!\d)[;,] ")
_SVG_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S)

# The HTML writer drops image attributes it does not know, so an image names
# its description in a class, which the page pass turns into aria-details.
_TIE = "a11y-details-"
_TIED = re.compile(rf"<img\b[^>]*\b{_TIE}([\w-]+)[^>]*>")

_SIG_DL = re.compile(
    r'(<span class="sig-paren">[(\[]</span>\s*)<dl\b[^>]*>\s*((?:(?!<dl\b).)*?)</dl>'
    r'(?=\s*<span class="sig-paren">)',
    re.S,
)
_DD = re.compile(r"<dd\b[^>]*>")
_FIGURE = re.compile(r"<figure\b.*?</figure>", re.S)
_STRAY_CAPTION_END = re.compile(r"</figcaption>\n?")
_POWER = re.compile(r"(?<=\d)<sup>([-−]?\d+)</sup>")
_SPOKEN_POWER = r'<sup><span class="visually-hidden"> to the power </span>\1</sup>'
# The link Sphinx writes after the text of a figure's or a table's caption.
_CAPTION_LINK = re.compile(
    r'(<span class="caption-text">((?:(?!<span class="caption-text">).)*?)</span>)'
    r'<a class="headerlink" href="(#[^"]+)" title="Link to this (image|table)">',
    re.S,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
# An anchor docutils numbers in page order.
_NUMBERED = re.compile(r"id\d+")

# The buttons that open the site navigation and the page's contents on a
# narrow screen work from a script. Without one they go, and the navigation
# follows the page, as tall as its links: the share of the row that the theme
# gives it would be a share of the page's height. The search page's own field
# goes too: it cannot search without a script, as the page says above it, and
# its icon would lie over its placeholder. The field in the header's dialog,
# which every page has, is outside that box. The theme's own rules load
# later, hence !important.
_NO_SCRIPT = (
    "<noscript><style>"
    ".bd-header button.sidebar-toggle{display:none!important}"
    ".bd-search-container form.bd-search{display:none!important}"
    "@media screen and (max-width:959.98px){"
    ".bd-container .bd-container__inner{flex-direction:column}"
    ".bd-sidebar-primary{order:2;flex:0 0 auto!important;position:static!important;"
    "visibility:visible!important;margin-left:0!important;width:100%!important;"
    "max-width:none!important;height:auto!important;max-height:none!important}"
    "}</style></noscript>"
)


def _quoted(text: str) -> str:
    """*text* as the value of an attribute written in double quotes."""
    return html.escape(text, quote=False).replace('"', "&quot;")


def _name_toggles(page: Any) -> Any:
    """Name each expand/collapse button of the sidebar after the section it opens.

    The theme returns the sidebar as a parsed tree, or as text when it reuses
    the tree of an earlier page; either comes back as the kind it was, and
    text that needs no name comes back untouched.
    """
    tree = BeautifulSoup(page, "html.parser") if isinstance(page, str) else page
    named = 0
    for summary in tree.find_all("summary"):
        link = summary.parent.find_previous_sibling("a")
        if link is not None and not summary.get("aria-label"):
            summary["aria-label"] = " ".join(link.get_text().split()) + " subsections"
            named += 1
    return page if tree is page or not named else type(page)(str(tree))


def _svg_title(path: Path) -> str:
    """The title the SVG file at *path* carries, or "" if it has none or is no SVG."""
    if path.suffix.lower() != ".svg":
        return ""
    try:
        with path.open(encoding="utf-8") as file:
            head = file.read(4096)
    except (OSError, UnicodeDecodeError):
        return ""
    title = _SVG_TITLE.search(head)
    if title is None:
        return ""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", title.group(1))).split())


def _lead(alt: str) -> str:
    """The start of *alt* up to its first colon, or "" if it has none within the
    short length. What stands before a colon says what the figure is, and what
    follows gives the detail."""
    head = alt[:SHORT_ALT]
    for colon in _COLON.finditer(head):
        if colon.start() >= 30:
            return head[: colon.start()]
    return ""


def _clause(alt: str) -> str:
    """The start of *alt* up to a break between clauses, or "" if it has none:
    its first colon, else its last semicolon or comma, within the short length."""
    head = alt[:SHORT_ALT]
    pauses = [pause.start() for pause in _PAUSE.finditer(head) if pause.start() >= 30]
    return _lead(alt) or (head[: pauses[-1]] if pauses else "")


def _short(alt: str, title: str = "") -> str:
    """A short alt text for an image whose full text is *alt* and whose file is
    titled *title*: the first sentence, else what stands before the first
    colon, else the title, else the first clause. A comma may part the items
    of a list, so the title comes before a cut at one. Never cut inside a
    clause; "" if *alt* has no place to end."""
    sentence = _SENTENCE.match(alt)
    if sentence:
        return sentence.group(1)
    fitting = title if len(title) <= SHORT_ALT else ""
    return _lead(alt) or fitting or _clause(alt)


def _describe(doctree: nodes.document, srcdir: Path) -> None:
    """Give each image with a long alt text a short one, and the full text as a
    disclosure under its figure. An image with no place for the disclosure, or
    no short text to take, keeps its alt text whole."""
    count = 0
    for image in list(doctree.findall(nodes.image)):
        alt = image.get("alt", "")
        if len(alt) <= SHORT_ALT + 20 or image.get("a11y-described"):
            continue
        parent = image.parent
        figure = parent if isinstance(parent, nodes.figure) else None
        if figure is None and isinstance(parent, nodes.container):
            figure = parent.parent if isinstance(parent.parent, nodes.figure) else None
        if figure is None and not isinstance(parent, (nodes.section, nodes.document)):
            continue
        title = _svg_title(srcdir / image["uri"])
        short = _short(alt, title)
        if not short:
            continue
        count += 1
        while f"figure-description-{count}" in doctree.ids:
            count += 1
        ident = f"figure-description-{count}"
        for twin in (figure or parent).findall(nodes.image):
            if twin.get("alt") == alt:
                twin["alt"] = short
                twin["a11y-described"] = True
                twin["classes"].append(_TIE + ident)
        # The name adds the figure's title to the label on screen, which tells
        # one figure's disclosure from the next where a reader lists the
        # page's controls. The name starts with the label's words, so the
        # label is for the screen alone: a screen reader would say it again
        # after the name, as the control's description. The line break keeps
        # the label and the text apart in the search index, and an apostrophe
        # written as an entity would enter the index as a word.
        name = _quoted(f"Figure description: {title or short}")
        details = nodes.raw(
            "",
            f'<details class="figure-description" id="{ident}">'
            f'<summary aria-label="{name}"><span aria-hidden="true">Figure description</span>'
            "</summary>\n"
            f"<p>{html.escape(alt, quote=False)}</p></details>",
            format="html",
        )
        if figure is not None:
            figure += details
        else:
            parent.insert(parent.index(image) + 1, details)


def _anchor_figures(doctree: nodes.document) -> int:
    """Give each figure docutils numbered an anchor made of the name of its
    image's file, ``fig-rabi-pulse``; a dark twin's file names the same
    figure. The new anchor is its first; the number stays. A figure whose
    name the page already uses as an anchor keeps its number alone. Return
    how many got one."""
    count = 0
    for figure in doctree.findall(nodes.figure):
        image = next(iter(figure.findall(nodes.image)), None)
        if image is None or not figure["ids"] or not _NUMBERED.fullmatch(figure["ids"][0]):
            continue
        stem = PurePosixPath(image["uri"]).stem.removesuffix("-dark")
        anchor = nodes.make_id(f"fig-{stem}")
        # A file name with no letter or digit leaves only the prefix.
        if anchor != "fig" and anchor not in doctree.ids:
            figure["ids"].insert(0, anchor)
            doctree.ids[anchor] = figure
            count += 1
    return count


def _unlink_images(doctree: nodes.document) -> None:
    """Take each image out of a link that holds nothing else."""
    for reference in list(doctree.findall(nodes.reference)):
        if len(reference) == 1 and isinstance(reference[0], nodes.image):
            reference.replace_self(reference[0])


def _label_outputs(doctree: nodes.document) -> None:
    """Write "Output" before each block of printed output."""
    for block in list(doctree.findall(nodes.literal_block)):
        if "output" in block["classes"]:
            label = nodes.paragraph("", "Output", classes=["output-label"])
            block.parent.insert(block.parent.index(block), label)


def _anchor(app: Sphinx, doctree: nodes.document) -> None:
    _anchor_figures(doctree)


def _repair_doctree(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    _label_outputs(doctree)
    # A description is written as HTML, and only the site has the rule that an
    # image is no link.
    if app.builder.format == "html":
        _unlink_images(doctree)
        _describe(doctree, Path(app.srcdir))


def _sig_lines(m: re.Match[str]) -> str:
    lines = _DD.sub('<div class="sig-param-line">', m.group(2)).replace("</dd>", "</div>")
    return f'{m.group(1)}<div class="sig-params">{lines}</div>'


def _end_figure(m: re.Match[str]) -> str:
    figure = m.group(0)
    return figure if "<figcaption" in figure else _STRAY_CAPTION_END.sub("", figure)


def _tie(m: re.Match[str]) -> str:
    ident = m.group(1)
    rest = m.group(0)[len("<img") :].replace(f" {_TIE}{ident}", "").replace(_TIE + ident, "")
    return f'<img aria-details="{ident}"' + rest.replace(' class=""', "")


def _caption_start(caption: str) -> str:
    """The start of *caption* that names its link: its first sentence. One
    longer than a name ends before its last break between clauses that a
    name holds. With no such break it is cut after eight words, and an
    ellipsis marks the cut."""
    sentence = _SENTENCE_END.split(caption, maxsplit=1)[0].rstrip(".,;:")
    if len(sentence) <= SHORT_NAME:
        return sentence
    # A break in the first twenty characters would leave too little to tell
    # the figure by.
    breaks = [
        pause.start()
        for pattern in (_PAUSE, _COLON)
        for pause in pattern.finditer(sentence)
        if 20 <= pause.start() <= SHORT_NAME
    ]
    if breaks:
        return sentence[: max(breaks)]
    words = sentence.split()
    return " ".join(words[:8]).rstrip(".,;:") + ("…" if len(words) > 8 else "")


def _name_caption_link(m: re.Match[str]) -> str:
    """The link after a caption, named after the start of the caption. The
    name is its label and it has no tooltip: a tooltip of the same words
    would be read out after the name, as a description."""
    caption = " ".join(html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).split())
    start = _caption_start(caption)
    if not start:
        return m.group(0)
    kind = "figure" if m.group(4) == "image" else "table"
    name = _quoted(f"Link to {kind}: {start}")
    return f'{m.group(1)}<a class="headerlink" href="{m.group(3)}" aria-label="{name}">'


def _repair_body(body: str) -> str:
    """The page's HTML with its signatures, figures, caption links and powers
    repaired."""
    body = _SIG_DL.sub(_sig_lines, body)
    body = _FIGURE.sub(_end_figure, body)
    body = _TIED.sub(_tie, body)
    body = _CAPTION_LINK.sub(_name_caption_link, body)
    return _POWER.sub(_SPOKEN_POWER, body)


def _site_root(baseurl: str) -> str:
    """The path the site is published under: that of ``html_baseurl`` if set."""
    return urlsplit(baseurl).path.rstrip("/") + "/" if baseurl else SITE_ROOT


def _page_context(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: Any
) -> None:
    toctree = context.get("generate_toctree_html")
    if toctree is not None and not getattr(toctree, "_a11y", False):

        def generate(*args, **kwargs):
            return _name_toggles(toctree(*args, **kwargs))

        generate._a11y = True  # type: ignore[attr-defined]
        context["generate_toctree_html"] = generate
    logo = context.get("theme_logo")
    if isinstance(logo, dict) and logo.get("text"):
        # The link holds the logo and then the title as text. The theme's alt
        # text repeats the title; this says what the logo shows.
        logo["alt_text"] = LOGO_ALT
    context["metatags"] = context.get("metatags", "") + _NO_SCRIPT
    body = context.get("body")
    if body:
        context["body"] = _repair_body(body)
    if pagename == "404":
        # The page is served at whatever address was not found. Its base is
        # its own address on the site: files resolve from the site root, and
        # a link to "#anchor", the theme's skip link among them, stays on
        # this page. With the root alone as the base such a link would lead
        # to the home page.
        address = _site_root(app.config.html_baseurl) + app.builder.get_target_uri(pagename)
        context["metatags"] += f'<base href="{html.escape(address)}">'


def _images_are_not_links(app: Sphinx, config: Config) -> None:
    # Sphinx would link each image that has a width or a scale to its file.
    config.html_scaled_image_link = False


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_css_file("a11y.css")
    app.add_css_file("a11y-print.css", media="print")
    app.add_js_file("a11y.js", defer="defer")
    app.connect("config-inited", _images_are_not_links)
    # As the page is read, so that the stored page holds the anchors: Sphinx
    # writes the link after a caption to the first anchor of its figure.
    app.connect("doctree-read", _anchor)
    app.connect("doctree-resolved", _repair_doctree)
    app.connect("html-page-context", _page_context, priority=900)
    # Raise env_version when the anchors or the markup written into pages
    # change: a build directory from before then reads and writes every page
    # again.
    return {"env_version": 2, "parallel_read_safe": True, "parallel_write_safe": True}
