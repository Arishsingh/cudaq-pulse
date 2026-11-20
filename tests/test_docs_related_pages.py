# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guide, API and architecture pages name the pages that go with them.

``docs/_ext/related_pages.py`` gives each page of its map ``RELATED`` a "See
also" note: the tutorial to start with and the related pages. None of this
needs a build: the map is held against the pages of the source tree and the
dialect reference pages the build generates, and the note is made on pages
that docutils parses here. The build itself warns when a link of a note
leads to no page.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

nodes = pytest.importorskip("docutils.nodes")
publish_doctree = pytest.importorskip("docutils.core").publish_doctree
addnodes = pytest.importorskip("sphinx.addnodes")

_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "docs"
_EXT = _DOCS / "_ext"
pytestmark = pytest.mark.skipif(not _EXT.exists(), reason="needs the source tree")
sys.path.insert(0, str(_EXT))
related_pages = pytest.importorskip("related_pages")
sys.path.remove(str(_EXT))

_MAP = "RELATED in docs/_ext/related_pages.py"
# The sections whose pages each need an entry.
_SECTIONS = ("user_guide", "api", "architecture")
_PAGE = "user_guide/kernels"

# What follows a page's title, and the parts of the page once it has its note.
_PLACES = [
    (
        "First.\n\nSecond.\n\nPart\n----\n\nText.\n",
        ["title", "paragraph", "seealso", "paragraph", "section"],
    ),
    ("Only.\n", ["title", "paragraph", "seealso"]),
    # A page that opens with no paragraph has the note under its title.
    ("::\n\n   code\n\nText.\n", ["title", "seealso", "literal_block", "paragraph"]),
    ("Part\n----\n\nText.\n", ["title", "seealso", "section"]),
    # A warning that opens the page stays first.
    (
        ".. warning::\n\n   Take care.\n\nPart\n----\n\nText.\n",
        ["title", "warning", "seealso", "section"],
    ),
    (
        ".. warning::\n\n   Take care.\n\nFirst.\n\nPart\n----\n\nText.\n",
        ["title", "warning", "paragraph", "seealso", "section"],
    ),
    # A paragraph that ends in a colon keeps the block it introduces.
    (
        "Run it::\n\n   code\n\nSecond.\n",
        ["title", "paragraph", "literal_block", "seealso", "paragraph"],
    ),
    ("These are:\n\nPart\n----\n\nText.\n", ["title", "paragraph", "seealso", "section"]),
    # A comment or a label shows nothing, and is not what the page opens with.
    (
        ".. a comment\n\nFirst.\n\nSecond.\n",
        ["title", "comment", "paragraph", "seealso", "paragraph"],
    ),
    (".. _here:\n\nPart\n----\n\nText.\n", ["title", "seealso", "target", "section"]),
    (
        "Run it:\n\n.. a comment\n\n::\n\n   code\n\nSecond.\n",
        ["title", "paragraph", "comment", "literal_block", "seealso", "paragraph"],
    ),
]


def _source_pages() -> set[str]:
    """The pages of the source tree, each by the name Sphinx gives it."""
    return {page.relative_to(_DOCS).with_suffix("").as_posix() for page in _DOCS.rglob("*.rst")}


def _generated_pages() -> set[str]:
    """The dialect reference pages, which the build writes among its sources."""
    cmake = _ROOT / "cmake"
    folder = re.search(
        r"-DOUTPUT_DIR=\$\{_cudaq_pulse_docs_source\}/(\S+)",
        (cmake / "CudaqPulseDocumentation.cmake").read_text(),
    )
    names = re.findall(r'"\w+:(\w+)"', (cmake / "CudaqPulseDialectDocs.cmake").read_text())
    assert folder and names, "the build no longer names the dialect pages as this test reads them"
    return {f"{folder.group(1)}/{name}" for name in names}


def _targets(entry) -> list[str]:
    return ([entry["start"]] if entry["start"] else []) + list(entry["see"])


def _page(body: str):
    """A page titled "Title" with *body* under it, as Sphinx reads one: the
    title stays the title of the page's top section."""
    return publish_doctree(f"Title\n=====\n\n{body}", settings_overrides={"doctitle_xform": False})


def _parts(doctree) -> list[str]:
    section = next(iter(doctree.findall(nodes.section)))
    return [type(child).__name__ for child in section.children]


def _doc_link(target: str, docname: str = _PAGE):
    """A link as the ``doc`` role writes it on the page *docname*."""
    return addnodes.pending_xref(
        "",
        nodes.inline("", target),
        refdomain="std",
        reftype="doc",
        reftarget=target,
        refdoc=docname,
        refexplicit=False,
        refwarn=True,
    )


def _note_targets(doctree) -> list[str]:
    note = next(iter(doctree.findall(addnodes.seealso)))
    return [link["reftarget"] for link in note.findall(addnodes.pending_xref)]


def test_every_guide_api_and_architecture_page_has_an_entry():
    pages = {page for page in _source_pages() if page.split("/")[0] in _SECTIONS}
    assert pages, "found no pages of the three sections under docs/"
    missing = sorted((pages | _generated_pages()) - set(related_pages.RELATED))
    assert not missing, f"give each of {missing} an entry in {_MAP} (it may be empty)"


def test_every_entry_and_every_target_is_a_page():
    pages = _source_pages() | _generated_pages()
    unknown = {
        name
        for docname, entry in related_pages.RELATED.items()
        for name in (docname, *_targets(entry))
        if name not in pages
    }
    assert not unknown, f"{_MAP} names pages that do not exist: {sorted(unknown)}"


def test_the_tutorials_have_no_entry():
    # Each tutorial ends with its own "Where to go next".
    assert not [docname for docname in related_pages.RELATED if docname.startswith("tutorials/")]


def test_a_page_starts_with_a_tutorial():
    # A page may come to link its tutorial in its own text: the note then
    # leaves the tutorial out, and the entry may stay as it is.
    started = 0
    for docname, entry in related_pages.RELATED.items():
        start = entry["start"]
        if start is None:
            continue
        started += 1
        assert start.startswith("tutorials/"), docname
    assert started


def test_no_entry_names_its_own_page_or_a_page_twice():
    for docname, entry in related_pages.RELATED.items():
        targets = _targets(entry)
        assert docname not in targets, docname
        assert len(set(targets)) == len(targets), docname


def test_the_note_names_the_tutorial_and_then_the_related_pages():
    note = related_pages.see_also(_PAGE, "tutorials/first", ["user_guide/operations", "api/kernel"])
    assert isinstance(note, addnodes.seealso)
    assert related_pages.NOTE_CLASS in note["classes"]
    # Until Sphinx resolves a link, its text is the name of the page.
    assert [line.astext() for line in note] == [
        "New to this topic? Start with the tutorial tutorials/first.",
        "Related pages: user_guide/operations, api/kernel.",
    ]
    one = related_pages.see_also("api/ops", None, ["user_guide/operations"])
    assert [line.astext() for line in one] == ["Related page: user_guide/operations."]
    assert related_pages.see_also("api/index", None, []) is None


def test_each_link_warns_when_its_page_is_gone():
    note = related_pages.see_also(_PAGE, "tutorials/first", ["api/kernel"])
    links = list(note.findall(addnodes.pending_xref))
    # From the top of the pages, whatever folder the page is in.
    assert [link["reftarget"] for link in links] == ["/tutorials/first", "/api/kernel"]
    for link in links:
        assert (link["refdomain"], link["reftype"], link["refdoc"]) == ("std", "doc", _PAGE)
        # Sphinx warns of a missing page only where the link asks for it.
        assert link["refwarn"] is True
        # Sphinx reads this key of every link to a page: without a text of
        # its own, the link takes the page's title.
        assert link["refexplicit"] is False


@pytest.mark.parametrize(("body", "parts"), _PLACES)
def test_the_note_stands_after_the_opening_paragraph_or_under_the_title(body, parts):
    doctree = _page(body)
    entry = {"start": None, "see": ["api/kernel"]}
    assert related_pages.add_see_also(doctree, _PAGE, entry)
    assert _parts(doctree) == parts


def test_a_page_labelled_by_its_title_has_the_note_after_its_opening_paragraph():
    # docutils leaves a notice under the title of such a page, and Sphinx
    # takes it out only once the note has its place.
    doctree = publish_doctree(
        ".. _title:\n\nTitle\n=====\n\nFirst.\n\nSecond.\n",
        settings_overrides={"doctitle_xform": False},
    )
    assert related_pages.add_see_also(doctree, _PAGE, {"start": None, "see": ["api/kernel"]})
    shown = [part for part in _parts(doctree) if part != "system_message"]
    assert shown == ["title", "paragraph", "seealso", "paragraph"]


def test_the_note_stands_above_the_description_of_an_api_object():
    entry = {"start": None, "see": ["api/kernel"]}
    for body, parts in (
        ("", ["title", "seealso", "index", "desc"]),
        ("These are:\n", ["title", "paragraph", "seealso", "index", "desc"]),
    ):
        doctree = _page(body)
        section = next(iter(doctree.findall(nodes.section)))
        # As Sphinx writes an object: its index entries, then its description.
        section += [addnodes.index(entries=[]), addnodes.desc()]
        assert related_pages.add_see_also(doctree, "api/ops", entry)
        assert _parts(doctree) == parts


def test_a_page_with_no_title_gets_no_note():
    doctree = publish_doctree("Only text.\n")
    assert not related_pages.add_see_also(doctree, _PAGE, {"start": None, "see": ["api/kernel"]})
    assert not list(doctree.findall(addnodes.seealso))


def test_a_page_does_not_name_again_what_its_text_links():
    doctree = _page("First.\n\nSecond.\n")
    paragraph = next(iter(doctree.findall(nodes.paragraph)))
    # From the page's folder, from the top of the pages, and beside the page.
    paragraph += [_doc_link("../tutorials/first"), _doc_link("/api/kernel"), _doc_link("editors")]
    assert related_pages.linked_pages(doctree, _PAGE) == {
        "tutorials/first",
        "api/kernel",
        "user_guide/editors",
    }
    entry = {
        "start": "tutorials/first",
        "see": ["api/kernel", "user_guide/operations", "user_guide/editors", _PAGE],
    }
    assert related_pages.add_see_also(doctree, _PAGE, entry)
    assert _note_targets(doctree) == ["/user_guide/operations"]

    linked = _page("First.\n")
    next(iter(linked.findall(nodes.paragraph))).append(_doc_link("../tutorials/first"))
    assert not related_pages.add_see_also(linked, _PAGE, {"start": "tutorials/first", "see": []})
    assert not list(linked.findall(addnodes.seealso))


def test_a_page_does_not_name_again_what_its_table_of_contents_lists():
    entry = {"start": None, "see": ["architecture/dialects/pulse", "architecture/pipeline"]}
    for hidden, named in (
        (False, ["/architecture/pipeline"]),
        (True, ["/" + t for t in entry["see"]]),
    ):
        doctree = _page("First.\n")
        section = next(iter(doctree.findall(nodes.section)))
        section += addnodes.toctree(includefiles=["architecture/dialects/pulse"], hidden=hidden)
        assert related_pages.add_see_also(doctree, "architecture/dialects", entry)
        assert _note_targets(doctree) == named


def test_the_pages_are_read_again_when_the_map_changes():
    values, handlers = {}, {}
    app = SimpleNamespace(
        add_config_value=lambda name, default, rebuild: values.update({name: (default, rebuild)}),
        connect=lambda event, handler: handlers.update({event: handler}),
    )
    meta = related_pages.setup(app)
    assert values == {"related_pages": (related_pages.RELATED, "env")}
    assert meta["parallel_read_safe"] and meta["parallel_write_safe"]

    # Only a page with an entry in the configured map gets a note.
    config = SimpleNamespace(related_pages={"api/ops": {"start": None, "see": ["api/kernel"]}})
    for docname, noted in (("api/ops", 1), ("tutorials/first", 0)):
        doctree = _page("First.\n")
        handlers["doctree-read"](
            SimpleNamespace(env=SimpleNamespace(docname=docname), config=config), doctree
        )
        assert len(list(doctree.findall(addnodes.seealso))) == noted
