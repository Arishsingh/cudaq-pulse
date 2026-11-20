# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The built site has what the docs' extensions write into its pages.

The extensions under ``docs/_ext`` repair and extend the HTML that Sphinx,
the theme, viewcode and the dialect generator write, and most of them find
that HTML by its markup. Their own tests run on markup written in the tests,
so a new version of any of the four could change what it writes and leave a
repair with nothing to match: the build would not warn, and those tests
would still pass. These tests read the pages of a built site, where the
build of the ``docs`` preset has left one, and hold what each extension
promises of them.

Without a site the tests are skipped. Where ``CUDAQ_PULSE_REQUIRE_DOCS_SITE``
is set, as a job that has just built the docs sets it, they fail instead. A
site built before a change to an extension fails too: build the docs again.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import posixpath
import re
from functools import cache
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
# Where the build of the ``docs`` preset writes the site.
_SITE = _ROOT / "build" / "docs" / "docs" / "html"
_REQUIRED = "CUDAQ_PULSE_REQUIRE_DOCS_SITE"
# Folders of the site that hold no page of the docs: the theme's files, the
# sources of the pages, the source listings and the files to download.
_NOT_PAGES = frozenset({"_static", "_sources", "_modules", "_downloads"})

_GUIDES = "user_guide/"
_TUTORIALS = "tutorials/"
_API = "api/"
_DIALECTS = "architecture/dialects/"
_HEADINGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
# The theme's box around a table, which scrolls one wider than the page.
_SCROLL_BOX = "pst-scrollable-table-container"

# A pass that adds links or notes to many pages adds next to none once it no
# longer finds its place in what Sphinx gives it. Each floor is some 60% of
# the count in the site it was measured on, so that the pages can change
# under it.
_API_LINKS_IN_GUIDES = 150  # of 244
_EXAMPLE_LINKS = 8  # of 15
_OPERATION_LINKS = 55  # of 95
# The captioned figures that hold an image and keep their numbered anchor
# beside the one the extension gives them, with room for pages to change.
_FIGURES_WITH_A_NUMBER = 35  # of 59
_GUIDES_WITH_A_NOTE = 6  # of 10

# What an entry of the API reference would show in place of a docstring of
# its own: the docstrings of ``str`` and ``tuple``, the types of the package's
# constants, and that of the descriptor a component reads its fields through.
_BORROWED = ("str(object=", "Built-in immutable sequence", "A field of a component")

# The ids of the theme and of Sphinx by which ``docs/_static/a11y.js`` finds
# the parts of a page it works on. Where one is gone, the script does nothing
# there and reports nothing.
_SCRIPT_LOOKS_UP = (
    "main-content",
    "pst-skip-link",
    "pst-search-dialog",
    "pst-primary-sidebar-modal",
    "pst-secondary-sidebar-modal",
    "pst-page-toc-nav",
    "search-results",
)

_OPERATION = re.compile(r"[A-Za-z_]\w*(?:\.\w+)+")
_WHOLE_NUMBER = re.compile(r"[-−]?\d+")
# A link to an anchor that docutils numbered in page order.
_NUMBERED_LINK = re.compile(r'href="#id\d+"')
# An anchor that docutils numbered in page order.
_NUMBERED_ANCHOR = re.compile(r"^id\d+$")


def _need(there: bool, missing: str) -> None:
    """Skip the test unless *there*, or fail it where a site is required."""
    if not there:
        if os.environ.get(_REQUIRED):
            pytest.fail(f"{_REQUIRED} is set, but {missing}")
        pytest.skip(missing)


@cache
def _pages() -> tuple[str, ...]:
    """The pages of the built site, each by its address from the site's root."""
    _need((_SITE / "index.html").is_file(), "build/docs/docs/html holds no site")
    _need(importlib.util.find_spec("bs4") is not None, "beautifulsoup4 is not installed")
    return tuple(
        sorted(
            page.relative_to(_SITE).as_posix()
            for page in _SITE.rglob("*.html")
            if not _NOT_PAGES.intersection(page.relative_to(_SITE).parts)
        )
    )


@cache
def _text(name: str) -> str:
    # Every read of the site passes the gate first: without a site a test
    # skips, or fails where one is required, instead of finding no file.
    _pages()
    return (_SITE / name).read_text(encoding="utf-8")


@cache
def _parsed(name: str) -> Any:
    """The page *name* as a tree. The tests share it, so none may change it."""
    return importlib.import_module("bs4").BeautifulSoup(_text(name), "html.parser")


@cache
def _anchors(name: str) -> frozenset[str]:
    return frozenset(tag["id"] for tag in _parsed(name).find_all(id=True))


@cache
def _articles() -> dict[str, Any]:
    """What each page says, by page: the page without the theme's header,
    sidebars and footer. The search page has none; a script writes its
    results."""
    found = {name: _parsed(name).select_one("article.bd-article") for name in _pages()}
    without = sorted(name for name, article in found.items() if article is None)
    assert without in ([], ["search.html"]), f"the theme writes no article on {without}"
    return {name: article for name, article in found.items() if article is not None}


def _dialect_pages() -> dict[str, Any]:
    """The articles of the pages the build generates, one for each dialect."""
    found = {name: article for name, article in _articles().items() if name.startswith(_DIALECTS)}
    assert found, f"the site has no page under {_DIALECTS}"
    return found


def _images() -> list[tuple[str, Any]]:
    """Each image of an article, with its page."""
    found = [
        (name, image) for name, article in _articles().items() for image in article.find_all("img")
    ]
    assert found, "found no image in any article"
    return found


def _name(element: Any) -> str:
    """The name a screen reader has for *element*: its label, or else the
    text in it that is not hidden from one."""
    label = element.get("aria-label") or "".join(
        text
        for text in element.find_all(string=True)
        if text.find_parent(attrs={"aria-hidden": "true"}) is None
    )
    return " ".join(label.split())


def _leads_to(name: str, link: Any) -> tuple[str, str]:
    """The page and the anchor that *link*, on the page *name*, leads to."""
    path, _, anchor = link.get("href", "").partition("#")
    page = posixpath.normpath(posixpath.join(posixpath.dirname(name), path)) if path else name
    return page, anchor


def _title(name: str) -> str:
    """The title of the page *name*, without its link to itself."""
    words = (
        text
        for text in _parsed(name).find("h1").find_all(string=True)
        if text.find_parent("a", class_="headerlink") is None
    )
    return " ".join("".join(words).split())


def _some(faults: list[str]) -> str:
    """How many *faults* there are, and the first of them."""
    return f"{len(faults)}, the first of them: {faults[:10]}"


def _not_running_text(tag: Any) -> bool:
    """Whether *tag* is a heading, a signature or a block of code."""
    classes = tag.get("class", [])
    return (
        tag.name in _HEADINGS
        or tag.name == "pre"
        or (tag.name == "dt" and "sig" in classes)
        or (tag.name == "p" and "rubric" in classes)
    )


def test_a_job_that_requires_the_site_does_not_skip_these_tests(monkeypatch, tmp_path):
    # A folder with no site in it stands for a build that wrote none. The
    # variable is named here as the docs workflow names it. ``_pages`` keeps
    # the pages it has found; the function under it looks again.
    look = _pages.__wrapped__
    monkeypatch.setitem(globals(), "_SITE", tmp_path)
    monkeypatch.delenv("CUDAQ_PULSE_REQUIRE_DOCS_SITE", raising=False)
    with pytest.raises(pytest.skip.Exception):
        look()
    monkeypatch.setenv("CUDAQ_PULSE_REQUIRE_DOCS_SITE", "1")
    # A skip that got out of here would end this test as skipped too.
    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as outcome:
        look()
    assert outcome.type is pytest.fail.Exception
    assert "build/docs/docs/html holds no site" in str(outcome.value)


def test_each_page_has_one_main_heading():
    counts = {name: len(_parsed(name).find_all("h1")) for name in _pages()}
    wrong = {name: count for name, count in counts.items() if count != 1}
    assert not wrong, f"pages and their number of <h1>, which must be one: {wrong}"


def test_each_toggle_of_the_sidebar_has_a_name():
    # The theme writes each as a <summary> that holds an icon and no words.
    toggles = {name: _parsed(name).select("nav summary") for name in _pages()}
    assert any(toggles.values()), "the theme writes the sidebar's toggles another way"
    unnamed = [name for name, found in toggles.items() if not all(map(_name, found))]
    assert not unnamed, f"a toggle of the sidebar has no name on {unnamed}"


def test_a_signature_on_several_lines_is_no_list():
    lines, listed = 0, []
    for name, article in _articles().items():
        lines += len(article.select("dt.sig > div.sig-params > div.sig-param-line"))
        if article.select("dt.sig dl"):
            listed.append(name)
    assert lines, "no signature of the API reference is written on several lines"
    assert not listed, f"a signature holds a list on {listed}"


def test_no_list_of_terms_is_without_a_term():
    # Sphinx writes a signature on several lines as such a list. An empty
    # list, which it leaves where an entry has no field to show, holds
    # nothing to read.
    bare = sorted(
        {
            name
            for name, article in _articles().items()
            for terms in article.find_all("dl")
            if terms.find(True, recursive=False) is not None
            and terms.find("dt", recursive=False) is None
        }
    )
    assert not bare, f"a <dl> has descriptions and no <dt> on {bare}"


def test_no_image_is_a_link():
    linked = sorted({name for name, image in _images() if image.find_parent("a") is not None})
    assert not linked, f"an image is inside a link on {linked}"


def test_each_image_has_an_alt_text_that_ends_whole():
    texts = [(name, " ".join((image.get("alt") or "").split())) for name, image in _images()]
    without = sorted({name for name, alt in texts if not alt})
    assert not without, f"an image has no alt text on {without}"
    cut = [f"{name}: {alt}" for name, alt in texts if alt.endswith(("…", "..."))]
    assert not cut, f"alt texts cut inside a clause: {_some(cut)}"


def test_each_figure_description_has_a_name_and_images_that_point_to_it():
    count, faults = 0, []
    for name, article in _articles().items():
        pointed = {image["aria-details"] for image in article.select("img[aria-details]")}
        descriptions = article.select("details.figure-description")
        count += len(descriptions)
        ids = {details.get("id", "") for details in descriptions}
        if "" in ids or ids != pointed:
            faults.append(f"{name}: descriptions {sorted(ids)}, images point to {sorted(pointed)}")
        # The words on screen are the same under every figure. The name says
        # which figure, so that a list of the page's controls tells them apart.
        summaries = [details.find("summary") for details in descriptions]
        names = ["" if summary is None else _name(summary) for summary in summaries]
        if "" in names or len(set(names)) != len(names):
            faults.append(f"{name}: descriptions named {names}")
    assert count, "found no figure description in any article"
    assert not faults, _some(faults)


def test_the_link_after_a_caption_is_named_and_leads_to_its_figure_or_table():
    count, kept, faults = 0, 0, []
    for name, article in _articles().items():
        for link in article.select("figcaption a.headerlink, caption a.headerlink"):
            count += 1
            captioned = link.find_parent(["figure", "table"])
            anchor = _leads_to(name, link)[1]
            # Sphinx links the first anchor of a figure, the one the figure
            # itself carries where it has more than one.
            if captioned is None or not anchor or captioned.get("id") != anchor:
                faults.append(f"{name}: {link.get('href')} is no anchor of its figure or table")
            # Sphinx names each link by the same tooltip. The extension labels
            # it after its caption, and by that label the stylesheet shows it
            # to keyboard and touch.
            if not (link.get("aria-label") or "").strip():
                faults.append(f"{name}: the link to {link.get('href')} has no name")
            # A figure that holds an image is anchored by the name of the
            # image's file, or by a name its page gives it: a number would
            # move when a figure is added above it. Where the extension gives
            # the name, the number stays in the figure, so a link made before
            # still arrives.
            if captioned is not None and captioned.name == "figure" and captioned.find("img"):
                if _NUMBERED_ANCHOR.match(anchor):
                    faults.append(
                        f"{name}: the figure {anchor} is anchored by a number: the name of its "
                        "image is taken on the page, as when two figures show one image"
                    )
                kept += captioned.find("span", id=_NUMBERED_ANCHOR, recursive=False) is not None
            # A tooltip of the same words would be read out after the name.
            if link.has_attr("title"):
                faults.append(f"{name}: the link to {link.get('href')} has a tooltip")
    assert count, "found no link after a caption"
    assert kept >= _FIGURES_WITH_A_NUMBER, (
        f"only {kept} figures keep their numbered anchor beside the name of their image"
    )
    assert not faults, _some(faults)


def test_figures_captions_and_descriptions_are_closed():
    # Docutils closes a caption in each figure with more than one part, also
    # in one that has no caption.
    uneven = {}
    for name in _pages():
        text = _text(name)
        for tag in ("figure", "figcaption", "details"):
            opened, closed = len(re.findall(rf"<{tag}[\s>]", text)), text.count(f"</{tag}>")
            if opened != closed:
                uneven[f"{name} <{tag}>"] = (opened, closed)
    assert not uneven, f"tags opened and closed: {uneven}"


def test_what_a_tutorial_prints_is_labelled_in_the_text():
    blocks, unlabelled = 0, set()
    for name, article in _articles().items():
        for block in article.select("div.output"):
            blocks += 1
            label = block.find_previous_sibling(True)
            labelled = (
                label is not None
                and label.name == "p"
                and "output-label" in label.get("class", [])
                and label.get_text(strip=True)
            )
            if not labelled:
                unlabelled.add(name)
    assert blocks, "found no block of printed output in any article"
    assert not unlabelled, f"printed output without its label on {sorted(unlabelled)}"


def test_a_power_of_a_number_reads_as_a_power():
    bare = []
    for name, article in _articles().items():
        for power in article.find_all("sup"):
            before = power.previous_sibling
            after_a_digit = isinstance(before, str) and before[-1:].isdigit()
            if after_a_digit and _WHOLE_NUMBER.fullmatch(power.get_text()):
                bare.append(f"{name}: {before[-12:]}^{power.get_text()}")
    assert not bare, f"powers that read as two numbers in a row: {bare}"


def test_the_404_page_resolves_its_links_from_its_own_address():
    assert "404.html" in _pages(), "the site has no 404.html"
    bases = {
        name: [base.get("href", "") for base in _parsed(name).find_all("base")] for name in _pages()
    }
    # The page is served at any address that was not found. From the site
    # root alone, its link to "#main-content" would lead to the home page.
    own = bases.pop("404.html")
    assert len(own) == 1 and own[0].endswith("/404.html"), f"the base of 404.html is {own}"
    others = sorted(name for name, found in bases.items() if found)
    assert not others, f"pages that must keep their own address as their base: {others}"


def test_the_print_stylesheet_is_linked_for_print_alone():
    wrong = {}
    for name in _pages():
        links = [
            link
            for link in _parsed(name).find_all("link")
            if "a11y-print.css" in link.get("href", "")
        ]
        if [link.get("media") for link in links] != ["print"]:
            wrong[name] = [str(link) for link in links]
    assert not wrong, f"pages and their links to a11y-print.css: {wrong}"


def test_each_top_entry_of_the_api_reference_is_a_heading_below_its_section():
    count, faults = 0, []
    for name, article in _articles().items():
        for entry in article.select("dl.py"):
            term = entry.find("dt", recursive=False)
            headings = [] if term is None else term.select('span[role="heading"]')
            levels = [heading.get("aria-level") for heading in headings]
            if entry.find_parent("dl", class_="py") is not None:
                # A member belongs to the entry of its class.
                expected = []
            else:
                count += 1
                expected = [str(min(6, 1 + len(entry.find_parents("section"))))]
            if levels != expected:
                ident = None if term is None else term.get("id")
                faults.append(f"{name}: {ident} has heading levels {levels}, not {expected}")
    assert count, "found no entry of the API reference"
    assert not faults, _some(faults)


def test_an_entry_of_the_api_reference_shows_its_own_docstring_whole():
    articles = {name: article for name, article in _articles().items() if name.startswith(_API)}
    assert articles, f"the site has no page under {_API}"
    faults = []
    for name, article in articles.items():
        # Napoleon prints what stands before a colon in the first line of a
        # docstring as a type, in a field of its own.
        fields = article.select("dl.field-list > dt")
        if any(field.get_text(strip=True).rstrip(":") == "Type" for field in fields):
            faults.append(f"{name}: a docstring is split at its first colon into a Type field")
        text = article.get_text()
        faults += [f"{name}: shows {words!r}" for words in _BORROWED if words in text]
    assert not faults, _some(faults)


def test_each_source_link_is_named_after_its_entry():
    count, faults = 0, []
    for name, article in _articles().items():
        for shown in article.select("span.viewcode-link"):
            count += 1
            link = shown.find_parent("a")
            words = shown.get_text(strip=True)
            label = "" if link is None else " ".join((link.get("aria-label") or "").split())
            entry = "" if link is None else _leads_to(name, link)[1]
            # The name starts with the words on the page and ends with the entry.
            if not (entry and label != words and label.startswith(words) and label.endswith(entry)):
                faults.append(f"{name}: the link to the source of {entry!r} is named {label!r}")
    assert count, "viewcode writes its links to the source another way"
    assert not faults, _some(faults)


def test_api_names_in_the_guides_are_links_but_not_in_headings_signatures_or_code():
    linked = {
        name: article.select("a.reference.internal > code.xref.py")
        for name, article in _articles().items()
    }
    guides = sum(len(found) for name, found in linked.items() if name.startswith(_GUIDES))
    assert guides >= _API_LINKS_IN_GUIDES, (
        f"only {guides} names of the API are links on the pages under {_GUIDES}"
    )
    misplaced = [
        f"{name}: {literal.get_text()}"
        for name, found in linked.items()
        for literal in found
        if literal.find_parent(_not_running_text) is not None
    ]
    assert not misplaced, f"names linked in a heading, a signature or code: {_some(misplaced)}"


def test_paths_of_examples_in_the_text_are_links():
    # The ``example`` role writes a link around plain text; a path written
    # as a literal is linked by ``api_links.py``.
    count = sum(len(article.select("a.extlink-example > code")) for article in _articles().values())
    assert count >= _EXAMPLE_LINKS, f"only {count} literals that name an example are links"


def test_guide_pages_name_their_related_pages_and_tutorials_do_not():
    notes = {
        name: article.select("div.seealso.related-pages") for name, article in _articles().items()
    }
    guides = [name for name, found in notes.items() if found and name.startswith(_GUIDES)]
    assert len(guides) >= _GUIDES_WITH_A_NOTE, (
        f"only {len(guides)} pages under {_GUIDES} have a note of related pages"
    )
    # Each tutorial ends with its own "Where to go next".
    tutorials = [name for name, found in notes.items() if found and name.startswith(_TUTORIALS)]
    assert not tutorials, f"tutorials with a note of related pages: {tutorials}"
    faults = []
    for name, found in notes.items():
        for note in found:
            links = note.find_all("a")
            # The note is no section: it has no heading and no anchor.
            if not links or note.get("id") or note.find(_HEADINGS) is not None:
                faults.append(f"{name}: the note has no link, or is a section")
            for link in links:
                page, anchor = _leads_to(name, link)
                text = " ".join(link.get_text().split())
                if anchor or page not in _pages() or text != _title(page):
                    faults.append(f"{name}: {text!r} is not the title of {link.get('href')}")
    assert not faults, _some(faults)


def test_each_dialect_table_scrolls_and_has_a_name():
    attributes, faults = 0, []
    for name, article in _dialect_pages().items():
        for table in article.find_all("table"):
            # The generator writes a table of attributes as bare HTML, with
            # the names of its columns in the first row.
            first = table.find("th")
            if first is not None and first.get_text(strip=True) == "Attribute":
                attributes += 1
            box = table.parent
            scrolls = box.name == "div" and _SCROLL_BOX in box.get("class", [])
            named = (table.get("aria-label") or "").strip() or table.find("caption") is not None
            if not (scrolls and named and table.find("thead") is not None):
                faults.append(f"{name}: {str(table)[:80]!r} in <{box.name}>")
    assert attributes, "found no table of attributes on the dialect pages"
    assert not faults, _some(faults)


def test_the_parts_of_a_dialect_entry_have_anchors_of_their_own():
    # Docutils numbers the sub-headings that share their words, "Operands:",
    # in page order: a new operation would move the anchor of each later one.
    numbered = {name: _NUMBERED_LINK.findall(_text(name))[:3] for name in _dialect_pages()}
    numbered = {name: links for name, links in numbered.items() if links}
    assert not numbered, f"links to a numbered anchor: {numbered}"
    count, missing = 0, []
    for name, article in _dialect_pages().items():
        for heading in article.find_all(_HEADINGS):
            first = heading.contents[0] if heading.contents else None
            operation = first.get_text() if getattr(first, "name", None) == "code" else ""
            if _OPERATION.fullmatch(operation):
                count += 1
                if operation not in _anchors(name):
                    missing.append(f"{name}: {operation}")
    assert count, "the generator heads the entry of an operation another way"
    assert not missing, f"operations whose name is no anchor: {missing}"


def test_names_of_operations_in_the_text_lead_to_their_entries():
    count, faults = 0, []
    for name, article in _articles().items():
        for literal in article.select("a.reference.internal > code:not(.xref)"):
            page, anchor = _leads_to(name, literal.parent)
            if not (page.startswith(_DIALECTS) and anchor):
                continue
            count += 1
            if page not in _pages() or anchor not in _anchors(page):
                faults.append(f"{name}: {literal.get_text()} leads to no entry")
            if literal.find_parent(_HEADINGS) is not None:
                faults.append(f"{name}: {literal.get_text()} is linked in a heading")
    assert count >= _OPERATION_LINKS, f"only {count} names of operations are links"
    assert not faults, _some(faults)


def test_the_general_index_lists_no_search_alias():
    # The words a page is found by are index entries with no target. Listed
    # in the general index, each would be a link to no anchor.
    assert "genindex.html" in _articles(), "the site has no general index"
    links = _articles()["genindex.html"].select("table a[href]")
    assert links, "Sphinx writes the entries of the general index another way"
    aliases = [link.get_text() for link in links if not _leads_to("genindex.html", link)[1]]
    assert not aliases, f"entries of the general index without an anchor: {aliases}"


def test_the_pages_have_what_the_page_script_looks_up():
    missing = [
        ident
        for ident in _SCRIPT_LOOKS_UP
        if not any(f'id="{ident}"' in _text(name) for name in _pages())
    ]
    assert not missing, f"no page has an element with the id {missing}"


def test_the_icon_of_each_kind_of_note_is_kept_from_screen_readers():
    # The theme draws the icon before the title of a note as a character of
    # its icon font, which a screen reader would be given as an unnamed
    # symbol. ``docs/_static/a11y.css`` gives the icon an empty alternative
    # text, with one rule for each kind of note the pages have.
    ruled = set(
        re.findall(
            r"div\.admonition\.([\w-]+) > \.admonition-title::after", _text("_static/a11y.css")
        )
    )
    assert ruled, "a11y.css has no rule for the icon of a note"
    titles = [
        (name, title.parent)
        for name in _pages()
        for title in _parsed(name).select(".admonition > .admonition-title")
    ]
    assert titles, "the theme writes the title of a note another way"
    bare = sorted(
        {
            f"{name}: <{box.name}> {' '.join(box.get('class', []))}"
            for name, box in titles
            if box.name != "div" or not ruled.intersection(box.get("class", []))
        }
    )
    assert not bare, f"notes of a kind without such a rule: {bare}"
