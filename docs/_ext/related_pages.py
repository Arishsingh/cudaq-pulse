# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guide, API and architecture pages name the pages that go with them.

A reader who arrives at a guide page from a search is not told that a
tutorial teaches the same subject step by step, and an API or architecture
page does not say which pages explain it. Each page with an entry in
``RELATED`` gets a "See also" note after its opening paragraph: the tutorial
to start with, where there is one, and the related pages.

* Each link is a cross-reference like one written with the ``doc`` role. Its
  text is the title of the page it leads to, and the build warns when that
  page no longer exists.
* A page that links a target in its own text, or lists it in a table of
  contents, does not name it again in the note.
* The note is no section: it adds no heading, no anchor and no entry to
  "On this page".
* The tutorials have no entry. Each ends with its own "Where to go next".

``RELATED`` is the default of the configuration value ``related_pages``, so
Sphinx reads the pages again when the map changes.
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING, Any, TypedDict

from docutils import nodes
from sphinx import addnodes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sphinx.application import Sphinx

START = "New to this topic? Start with the tutorial "
SEE_ONE = "Related page: "
SEE = "Related pages: "
# The class of a note written here, which tells it from one a page writes itself.
NOTE_CLASS = "related-pages"


class Entry(TypedDict):
    """What a page's note names: the tutorial to start with, or None, and the
    related pages, each by the name Sphinx gives the page."""

    start: str | None
    see: list[str]


# Every page of the user guide, the API reference and the architecture
# section has an entry, so that a new page is given its related pages, or an
# empty entry where its own text names them already.
# ``tests/test_docs_related_pages.py`` holds the map against the pages.
RELATED: dict[str, Entry] = {
    "user_guide/kernels": {
        "start": "tutorials/first_pulse_program",
        "see": ["user_guide/operations", "api/kernel"],
    },
    "user_guide/operations": {
        "start": None,
        "see": ["user_guide/kernels", "api/ops", "architecture/dialects/pulse"],
    },
    "user_guide/conventions": {
        "start": None,
        "see": ["user_guide/controller_backends"],
    },
    "user_guide/compilation": {
        "start": "tutorials/several_qubits",
        "see": ["api/compile", "architecture/pipeline"],
    },
    "user_guide/quantum_systems": {
        "start": "tutorials/own_quantum_system",
        "see": ["api/quantum_systems", "api/quantum_system_components"],
    },
    "user_guide/custom_passes": {
        "start": None,
        "see": ["architecture/pipeline", "architecture/mlir_passes"],
    },
    "user_guide/editors": {
        "start": None,
        "see": ["user_guide/kernels", "user_guide/operations"],
    },
    "user_guide/classical_calls": {
        "start": "tutorials/classical_calls",
        "see": ["api/classical", "architecture/dialects/classical"],
    },
    "user_guide/controller_backends": {
        "start": "tutorials/kernel_to_controller",
        "see": ["api/controller"],
    },
    "user_guide/gpu_execution": {
        "start": "tutorials/gpu_simulation",
        "see": ["api/simulation", "architecture/pipeline"],
    },
    # The index lists its pages, and these four name their guide pages in
    # their own text.
    "api/index": {"start": None, "see": []},
    "api/simulation": {"start": None, "see": []},
    "api/classical": {"start": None, "see": []},
    "api/controller": {"start": None, "see": []},
    "api/errors": {"start": None, "see": []},
    "api/compile": {
        "start": None,
        "see": ["user_guide/compilation", "user_guide/custom_passes", "architecture/mlir_passes"],
    },
    "api/quantum_systems": {"start": None, "see": ["user_guide/quantum_systems"]},
    "api/quantum_system_components": {"start": None, "see": ["user_guide/quantum_systems"]},
    "api/kernel": {"start": None, "see": ["user_guide/kernels", "user_guide/compilation"]},
    "api/ops": {"start": None, "see": ["user_guide/operations"]},
    "api/viz": {"start": None, "see": ["user_guide/compilation"]},
    "architecture/pipeline": {
        "start": None,
        "see": ["architecture/dialects", "user_guide/compilation", "user_guide/custom_passes"],
    },
    "architecture/dialects": {
        "start": None,
        "see": ["architecture/pipeline", "architecture/mlir_passes"],
    },
    "architecture/mlir_passes": {
        "start": None,
        "see": ["architecture/dialects", "user_guide/compilation", "user_guide/custom_passes"],
    },
    # The dialect reference pages, which the build generates.
    "architecture/dialects/pulse": {
        "start": None,
        "see": ["architecture/dialects", "architecture/mlir_passes", "user_guide/operations"],
    },
    "architecture/dialects/classical": {
        "start": None,
        "see": ["architecture/dialects", "user_guide/classical_calls"],
    },
    "architecture/dialects/qop": {
        "start": None,
        "see": ["architecture/dialects", "architecture/pipeline", "user_guide/gpu_execution"],
    },
    "architecture/dialects/cudm": {
        "start": None,
        "see": ["architecture/dialects", "architecture/pipeline", "user_guide/gpu_execution"],
    },
}


def linked_pages(doctree: nodes.document, docname: str) -> set[str]:
    """The pages that the page *docname* links with the ``doc`` role or lists
    in a table of contents it shows."""
    linked: set[str] = set()
    for xref in doctree.findall(addnodes.pending_xref):
        if xref.get("refdomain") == "std" and xref.get("reftype") == "doc":
            # As Sphinx reads the target: from the top of the pages after a
            # slash, else from the folder of the page that links.
            origin = "/" + xref.get("refdoc", docname)
            linked.add(posixpath.normpath(posixpath.join(origin, "..", xref["reftarget"]))[1:])
    for toctree in doctree.findall(addnodes.toctree):
        if not toctree.get("hidden"):
            linked.update(toctree.get("includefiles", ()))
    return linked


def _link(docname: str, target: str) -> addnodes.pending_xref:
    """A link from the page *docname* to the page *target*, as the ``doc``
    role writes it. Sphinx gives it the title of *target* as its text."""
    return addnodes.pending_xref(
        "",
        nodes.inline("", target, classes=["xref", "std", "std-doc"]),
        refdomain="std",
        reftype="doc",
        reftarget="/" + target,
        refdoc=docname,
        # Without a text of its own the link takes the page's title.
        refexplicit=False,
        # The build warns if the page is gone, where it would else write the
        # target as plain text.
        refwarn=True,
    )


def see_also(docname: str, start: str | None, see: Iterable[str]) -> addnodes.seealso | None:
    """The note of the page *docname*: a line for the tutorial *start* and a
    line for the pages of *see*. None if there is nothing to name."""
    note = addnodes.seealso(classes=[NOTE_CLASS])
    if start:
        note += nodes.paragraph("", START, _link(docname, start), nodes.Text("."))
    links: list[nodes.Node] = []
    for target in see:
        links += [nodes.Text(", "), _link(docname, target)]
    if links:
        label = SEE if len(links) > 2 else SEE_ONE
        note += nodes.paragraph("", label, *links[1:], nodes.Text("."))
    return note if len(note) else None


# What puts nothing on the page: comments, link targets, index entries, and
# the notices docutils leaves in a page. Sphinx takes the notices out only
# after this extension has run, and a page whose label is the name of its
# own title has one right under the title.
_UNSEEN = (nodes.Invisible, nodes.system_message)


def _next_seen(children: list[nodes.Node], index: int) -> int:
    """The place, from *index* on, of the first of *children* that shows on
    the page, or their number if none does."""
    while index < len(children) and isinstance(children[index], _UNSEEN):
        index += 1
    return index


def _opens_page(node: nodes.Node) -> bool:
    """Whether *node* is the page's title or a warning or note under it."""
    # Sphinx makes the description of an API object a kind of admonition.
    # It is the page's content, and the note goes above it.
    return isinstance(node, (nodes.title, nodes.Admonition)) and not isinstance(node, addnodes.desc)


def _place(section: nodes.section) -> int:
    """Where the note goes among the children of the page's top section:
    after the opening paragraph, or before what the page opens with instead.
    A warning or note that opens the page stays first."""
    children = section.children
    index = 0
    while True:
        head = _next_seen(children, index)
        if head == len(children) or not _opens_page(children[head]):
            break
        index = head + 1
    opening = _next_seen(children, index)
    if opening == len(children) or not isinstance(children[opening], nodes.paragraph):
        return index
    index = opening + 1
    # A paragraph that ends in a colon introduces the block after it, and
    # the two stay together. An API object's description can run to pages,
    # so the note stays above one.
    if children[opening].astext().rstrip().endswith(":"):
        block = _next_seen(children, index)
        kept = block < len(children) and not isinstance(
            children[block], (nodes.section, addnodes.desc)
        )
        if kept:
            index = block + 1
    return index


def add_see_also(doctree: nodes.document, docname: str, entry: Entry) -> bool:
    """Give the page *docname* the note of *entry*, without the pages it links
    itself. Return whether it got one."""
    section = next(iter(doctree.findall(nodes.section)), None)
    if section is None:
        return False
    linked = linked_pages(doctree, docname) | {docname}
    start = entry.get("start")
    note = see_also(
        docname,
        None if start in linked else start,
        [target for target in entry.get("see", ()) if target not in linked],
    )
    if note is None:
        return False
    section.insert(_place(section), note)
    return True


def _add_note(app: Sphinx, doctree: nodes.document) -> None:
    # The note is part of the page as read: it is stored with the page, and
    # Sphinx resolves its links with the page's own.
    docname = app.env.docname
    entry = app.config.related_pages.get(docname)
    if entry:
        add_see_also(doctree, docname, entry)


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_config_value("related_pages", RELATED, "env")
    app.connect("doctree-read", _add_note)
    # Raise env_version when the note's wording or place changes: a build
    # directory from before then reads every page again.
    return {"env_version": 1, "parallel_read_safe": True, "parallel_write_safe": True}
