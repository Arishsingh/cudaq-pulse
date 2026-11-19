# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generated dialect reference pages, put right for their readers.

The build generates the pages under ``architecture/dialects/`` from the
definitions of the operations, so their text cannot be edited. What they
need is done here, as Sphinx reads and writes them:

* Each "Attributes:" table, which the generator writes as bare HTML, is
  written as the other tables are: in the box that scrolls a table wider
  than the page, with the theme's borders, and with its first row as the
  table's header.
* Each table has a name, "pulse.drive operands". The heading above it reads
  only "Operands:", "Results:" or "Attributes:", so a reader who moves from
  table to table would not hear which operation a table belongs to.
* Each of those sub-headings has an anchor of its own,
  ``pulse-drive-pulse-driveop-operands``. Docutils numbers the headings that
  share their words in page order, ``id5``, so a new operation would move
  the anchor of every later one. The anchor a heading had stays as a second
  one: a link made before still arrives.
* The entry of an operation has the operation's name as an anchor,
  ``pulse.drive``, beside the one that holds its C++ class, and the label
  ``op-pulse.drive`` for the ``ref`` role.
* A literal that is the name of an operation, ``pulse.sync``, links to its
  entry, the first time each section names it. That holds on every page: the
  pages write what a kernel calls bare, ``sync(...)``, and the links of
  ``api_links.py`` leave the name of an operation alone. An item of a list
  counts its own first mention, so a list of a dialect's operations links
  each of them.
* The MLIR examples are highlighted, by the lexer of ``mlir_lexer.py``.
"""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING, Any

from docutils import nodes
from mlir_lexer import MlirLexer
from sphinx.util.nodes import make_refnode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from sphinx.application import Sphinx

# The pages the build generates, one for each dialect.
PAGES = "architecture/dialects/"
# Before the name of an operation, the label of its entry.
LABEL = "op-"
# The theme's box around a table, which scrolls one wider than the page.
SCROLL_BOX = "pst-scrollable-table-container"

# An anchor docutils numbers in page order.
_NUMBERED = re.compile(r"id\d+")
_OPERATION = re.compile(r"[A-Za-z_]\w*(?:\.\w+)+")
_HEADINGS = (nodes.title, nodes.rubric)
# Where the mentions of an operation are counted: in the nearest of these.
_SCOPES = (nodes.list_item, nodes.section)

# A table with no attribute and no parts but its rows is the generator's own;
# its first row holds the names of the columns.
_BARE_TABLE = re.compile(r"<table>(?=\s*<tr>)(\s*<tr>\s*<th>.*?</tr>)?(.*?)</table>", re.S)
_HEADING_OR_TABLE = re.compile(
    r"<h([1-6])\b[^>]*>(.*?)</h\1>|<table\b[^>]*>(?!\s*<caption\b)", re.S
)
_HEADING_LINK = re.compile(r'<a class="headerlink".*?</a>', re.S)
_TAG = re.compile(r"<[^>]+>")
# The C++ class the generator writes after the name of an operation.
_CLASS = re.compile(r"\s*\(::[^()]*\)$")


def operation(section: nodes.section) -> str:
    """The operation whose entry *section* is, "" if it is no such entry. The
    generator heads the entry with the operation's name, as a literal, and
    then its C++ class."""
    title = section[0] if len(section) else None
    if not isinstance(title, nodes.title) or not len(title):
        return ""
    name = title[0].astext() if isinstance(title[0], nodes.literal) else ""
    return name if _OPERATION.fullmatch(name) else ""


def anchor_parts(doctree: nodes.document) -> int:
    """Give each sub-heading that ends in a colon, "Operands:", and each
    heading docutils numbered an anchor made of the anchor of the section it
    is in and its own words. The new anchor is its first; the others stay.
    Return how many got one."""
    count = 0
    # A section comes before the sections in it, so each takes the first
    # anchor its parent ends with.
    for section in doctree.findall(nodes.section):
        parent = section.parent
        if not isinstance(parent, nodes.section) or not section["ids"] or not parent["ids"]:
            continue
        # Many entries have a sub-heading of the same words. Docutils gives
        # the words to the first on the page and numbers the others, so the
        # anchor of each depends on the entries before it.
        title = section[0].astext()
        if not (title.endswith(":") or _NUMBERED.fullmatch(section["ids"][0])):
            continue
        words = nodes.make_id(title)
        anchor = f"{parent['ids'][0]}-{words}"
        if words and anchor not in doctree.ids:
            section["ids"].insert(0, anchor)
            doctree.ids[anchor] = section
            count += 1
    return count


def anchor_operations(doctree: nodes.document) -> dict[str, str]:
    """Give the entry of each operation the operation's name as one more
    anchor. Return the first anchor of each entry, by operation."""
    entries = {}
    for section in doctree.findall(nodes.section):
        name = operation(section)
        if not name or not section["ids"] or doctree.ids.get(name, section) is not section:
            continue
        if name not in section["ids"]:
            section["ids"].append(name)
            doctree.ids[name] = section
        entries[name] = section["ids"][0]
    return entries


def _above(node: nodes.Node) -> Iterator[nodes.Node]:
    while node.parent is not None:
        node = node.parent
        yield node


def link_operations(
    doctree: nodes.document, link_to: Callable[[str], nodes.reference | None]
) -> int:
    """Put each literal of running text that names an operation inside the
    link *link_to* gives for the name, the first time its section names it,
    or its item of a list where it is in one. *link_to* gives None for a name
    that is no operation's. Return how many were linked."""
    seen: set[tuple[int, str]] = set()
    count = 0
    # ``_link_first_mentions`` of ``api_links.py`` walks the same way, but
    # counts by section alone: a repair to one is due in the other.
    for literal in list(doctree.findall(nodes.literal)):
        name = literal.astext()
        # A literal written by a role has classes: its role has said what it is.
        if literal["classes"] or len(literal) != 1 or not _OPERATION.fullmatch(name):
            continue
        # Sphinx writes a link only inside text.
        if not isinstance(literal.parent, nodes.TextElement):
            continue
        above = list(_above(literal))
        if any(isinstance(node, _HEADINGS) for node in above):
            continue
        sections = [node for node in above if isinstance(node, nodes.section)]
        # The entry of an operation does not link to itself.
        if any(name in section["ids"] for section in sections):
            continue
        # A mention counts for the item of a list it is in, and else for its
        # section: a list of a dialect's operations links each of them,
        # though the paragraph above the list has named some.
        within = next((node for node in above if isinstance(node, _SCOPES)), doctree)
        mention = (id(within), name)
        # A link the page has is the mention of the operation where it stands.
        linked = any(isinstance(node, nodes.reference) for node in above)
        link = None if linked or mention in seen else link_to(name)
        if link is not None:
            literal.parent.replace(literal, link)
            link += literal
            count += 1
        if linked or link is not None:
            seen.add(mention)
    return count


def _as_the_other_tables(m: re.Match[str]) -> str:
    head = f"<thead>{m.group(1)}</thead>" if m.group(1) else ""
    return (
        f'<div class="{SCROLL_BOX}"><table class="table">{head}'
        f"<tbody>{m.group(2)}</tbody></table></div>"
    )


def wrap_tables(body: str) -> str:
    """The page's HTML with each bare table written as the theme writes one."""
    return _BARE_TABLE.sub(_as_the_other_tables, body)


def _words(heading: str) -> str:
    """The text of the HTML of a heading, without the link to itself."""
    text = html.unescape(_TAG.sub("", _HEADING_LINK.sub("", heading)))
    return " ".join(text.split())


def name_tables(body: str) -> str:
    """The page's HTML with each table under a heading that ends in a colon
    named after the entry it is in and that heading: "pulse.drive operands".
    A table that has a name or a caption keeps it."""
    headings: dict[int, str] = {}

    def named(m: re.Match[str]) -> str:
        if m.group(1):
            level = int(m.group(1))
            for deeper in [other for other in headings if other >= level]:
                del headings[deeper]
            headings[level] = _words(m.group(2))
            return m.group(0)
        tag = m.group(0)
        level = max(headings, default=0)
        part, entry = headings.get(level, ""), headings.get(level - 1, "")
        words = part[:-1].strip()
        if not part.endswith(":") or not words or not entry or "aria-label" in tag:
            return tag
        name = f"{_CLASS.sub('', entry)} {words[0].lower()}{words[1:]}"
        return f'{tag[:-1]} aria-label="{html.escape(name)}">'

    return _HEADING_OR_TABLE.sub(named, body)


def _anchor(app: Sphinx, doctree: nodes.document) -> None:
    docname = app.env.docname
    if not docname.startswith(PAGES):
        return
    anchor_parts(doctree)
    # A label is stored with the page it leads to, like one the page writes
    # itself: Sphinx drops it when it reads the page again, and takes it over
    # from a page read in parallel.
    domain = app.env.get_domain("std")
    for name, anchor in anchor_operations(doctree).items():
        domain.anonlabels[LABEL + name] = docname, anchor
        domain.labels[LABEL + name] = docname, anchor, name


def _link(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    # Only now is every page read and each entry known. The tree is the copy
    # made for writing this page, so the stored pages stay as read.
    entries = app.env.get_domain("std").anonlabels

    def link_to(name: str) -> nodes.reference | None:
        page, anchor = entries.get(LABEL + name, ("", ""))
        return make_refnode(app.builder, docname, page, anchor, []) if page else None

    link_operations(doctree, link_to)


def _page_context(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: Any
) -> None:
    body = context.get("body")
    if body and pagename.startswith(PAGES):
        context["body"] = name_tables(wrap_tables(body))


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_lexer("mlir", MlirLexer)
    # Sphinx takes the first anchor of each heading for "On this page" on the
    # same event, at the default priority of 500: the anchors are made first.
    app.connect("doctree-read", _anchor, priority=400)
    app.connect("doctree-resolved", _link)
    app.connect("html-page-context", _page_context)
    # Raise env_version when the anchors or the labels change: a build
    # directory from before then reads every page again.
    return {"env_version": 1, "parallel_read_safe": True, "parallel_write_safe": True}
