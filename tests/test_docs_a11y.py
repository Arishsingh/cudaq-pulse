# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The docs' accessibility repairs hold on the pages' own text and markup.

``docs/_ext/a11y.py`` repairs what Sphinx and the theme write. None of this
needs a build: the repairs run on the alt texts of the pages and the titles
of their figures, on HTML that docutils writes here, on the sidebar, logo and
skip-link markup the installed theme makes, and on Sphinx's markup for a
signature and for a caption's link. The focus colors of
``docs/_static/a11y.css`` are measured against the installed theme's surfaces,
and what the stylesheets and the extension take from the theme, a class, a
width and the light colors that print, is held to the installed theme.
"""

from __future__ import annotations

import html
import importlib.util
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from urllib.parse import urljoin

import pytest

nodes = pytest.importorskip("docutils.nodes")
publish = pytest.importorskip("docutils.core")
new_document = pytest.importorskip("docutils.utils").new_document
html5_polyglot = pytest.importorskip("docutils.writers.html5_polyglot")
bs4 = pytest.importorskip("bs4")
markupsafe = pytest.importorskip("markupsafe")

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_EXT = _DOCS / "_ext"
pytestmark = pytest.mark.skipif(not _EXT.exists(), reason="needs the source tree")
sys.path.insert(0, str(_EXT))
a11y = pytest.importorskip("a11y")
sys.path.remove(str(_EXT))

# An alt text longer than this is shortened.
_LONG = a11y.SHORT_ALT + 20
_ALT = "A schedule on two lines. " + "The drive line plays and the readout line waits. " * 3
_DIRECTIVE = re.compile(
    r"^\.\. (image|figure|themed-figure|toy-flow|pulse-visual):: (\S+)\n((?:[ \t]+:[\w-]+:.*\n)+)",
    re.M,
)
_THEMES = ("light", "dark")


def _figures() -> list[tuple[str, str, str, str]]:
    """(page, directive, image file, alt text) of each image of the docs that
    has an alt text, the file named from the docs folder as Sphinx names it.
    A ``toy-flow`` shows its ``:fallback:`` and a ``pulse-visual`` the still
    of its scene."""
    found = []
    for page in sorted(_DOCS.rglob("*.rst")):
        for kind, target, options in _DIRECTIVE.findall(page.read_text()):
            option = dict(re.findall(r":([\w-]+): *(.*)", options))
            if kind == "toy-flow":
                target = option.get("fallback", "")
            elif kind == "pulse-visual":
                target = str(PurePosixPath(target).with_name(option["scene"] + ".svg"))
            if target and option.get("alt"):
                file = (page.parent / target).relative_to(_DOCS).as_posix()
                found.append((page.name, kind, file, option["alt"]))
    return found


def _title(image: Path) -> str:
    """The title of the SVG file *image*, or "" if it has none or is no SVG."""
    if image.suffix != ".svg":
        return ""
    title = ET.parse(image).getroot().find("{http://www.w3.org/2000/svg}title")
    return " ".join(title.text.split()) if title is not None and title.text else ""


def _page(*parts):
    document = new_document("page")
    section = nodes.section(ids=["page"])
    section += nodes.title("", "Page")
    section += list(parts)
    document += section
    return document


def _themed(file: str, alt: str, caption: bool = True):
    """A figure as the pages' directives build it: the image and its dark
    twin in one box, then the caption."""
    light = PurePosixPath(file)
    dark = light.with_name(f"{light.stem}-dark{light.suffix}")
    box = nodes.container(classes=["themed-figure"])
    box += nodes.image(uri=str(light), alt=alt, classes=["only-light"])
    box += nodes.image(uri=str(dark), alt=alt, classes=["only-dark"])
    figure = nodes.figure("", box)
    if caption:
        figure += nodes.caption("", "What the figure shows.")
    return figure


def _written(document) -> str:
    """The HTML docutils writes for *document*."""
    writer = html5_polyglot.Writer()
    publish.publish_from_doctree(document, writer=writer, settings_overrides={"report_level": 5})
    return writer.parts["body"]


def _parsed(text: str):
    return bs4.BeautifulSoup(text, "html.parser")


def _ends_whole(short: str, alt: str) -> bool:
    """Whether *short* is *alt* up to the end of a sentence or of a clause."""
    rest = alt[len(short) :]
    if not short or not rest or not alt.startswith(short):
        return False
    sentence = short[-1] in ".!?" and rest[0].isspace()
    clause = rest[:2] in (": ", "; ", ", ") and not short[-1].isdigit()
    return sentence or clause


def _context(pagename: str, baseurl: str = "", **context):
    """The page context of *pagename* after the extension has seen it."""
    app = SimpleNamespace(
        config=SimpleNamespace(html_baseurl=baseurl),
        builder=SimpleNamespace(get_target_uri=lambda name: f"{name}.html"),
    )
    a11y._page_context(app, pagename, "page.html", context, None)
    return context


def _theme_sheet(package: str, name: str) -> str:
    spec = importlib.util.find_spec(package)
    if spec is None:
        pytest.skip(f"needs {package}")
    return next(Path(spec.origin).parent.rglob(name)).read_text()


def _theme_css() -> str:
    """The installed theme's stylesheets, in the order a page loads them."""
    return _theme_sheet("pydata_sphinx_theme", "pydata-sphinx-theme.css") + _theme_sheet(
        "nvidia_sphinx_theme", "nvidia-sphinx-theme.css"
    )


def _theme_colors(theme: str) -> dict[str, str]:
    """The theme's color variables in *theme*, a later rule over an earlier one."""
    colors = {}
    variable = re.compile(r"(--pst-color-[\w-]+):(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})\b")
    for block in re.findall(r"html\[data-theme=" + theme + r"\]\{([^}]*)\}", _theme_css()):
        colors.update(variable.findall(block))
    return colors


def _own_css() -> str:
    """``a11y.css`` without its comments."""
    return re.sub(r"/\*.*?\*/", "", (_DOCS / "_static" / "a11y.css").read_text(), flags=re.S)


def _ring(theme: str) -> str:
    """The color of the focus ring in *theme*."""
    rings = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*--a11y-focus:[^{}]*)\}", _own_css()):
        color = re.search(r"--a11y-focus:\s*(#[0-9a-fA-F]{6})", body)
        rings["dark" if "dark" in selector else "light"] = color.group(1)
    return rings[theme]


def _six_digits(color: str) -> str:
    """The hex *color* in lower case, a three-digit one written with six."""
    color = color.lower()
    return "#" + "".join(2 * c for c in color[1:]) if len(color) == 4 else color


def _luminance(color: str) -> float:
    if len(color) == 4:
        color = "#" + "".join(2 * c for c in color[1:])
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_a_long_alt_text_becomes_a_whole_phrase_and_a_description_under_the_figure():
    figures = _figures()
    assert figures, "found no image with an alt text in the docs"
    for page, kind, file, alt in figures:
        where = f"{page}: {file}"
        document = _page(nodes.image(uri=file, alt=alt) if kind == "image" else _themed(file, alt))
        a11y._describe(document, _DOCS)
        images = list(document.findall(nodes.image))
        short = images[0]["alt"]
        described = [raw.astext() for raw in document.findall(nodes.raw)]
        assert all(image["alt"] == short for image in images), where
        if len(alt) <= _LONG:
            assert short == alt and not described, where
            continue
        assert short != alt, f"{where}: the alt text has no sentence or clause to stop at"
        assert len(short) <= a11y.SHORT_ALT + 1, (where, short)
        assert "…" not in short and not short.endswith("..."), (where, short)
        assert _ends_whole(short, alt) or short == _title(_DOCS / file), (where, short)
        assert len(described) == 1, where
        # The line break keeps the label and the text apart in the search index.
        assert f"</summary>\n<p>{html.escape(alt, quote=False)}</p>" in described[0], where
        assert "&#x27;" not in described[0], where


def test_with_no_title_to_take_a_long_sentence_ends_between_clauses():
    long = [alt for *_, alt in _figures() if len(alt) > _LONG]
    assert long, "found no long alt text in the docs"
    for alt in long:
        short = a11y._short(alt)
        assert short == "" or _ends_whole(short, alt), short
        assert len(short) <= a11y.SHORT_ALT + 1 and "…" not in short, short


def test_a_long_sentence_stops_at_its_colon_and_else_gives_way_to_the_title():
    lead = "Timeline of the kernel on its drive and readout lines"
    assert a11y._short(lead + ": " + "a pulse, a wait, " * 12, "The kernel") == lead
    # A comma may part the items of a list, so the title comes before a cut at one.
    items = "Timeline of the kernel on two lines with " + "a pulse, a wait, " * 12
    assert a11y._short(items, "The kernel") == "The kernel"
    assert _ends_whole(a11y._short(items), items)
    # A colon after a number is inside a list of values.
    values = "The levels of each qudit of the pair, qudit 0: " + "0-1 at 5.10 GHz, " * 12
    assert a11y._short(values, "The levels") == "The levels"


def test_an_alt_text_with_nowhere_to_go_or_no_place_to_end_stays_whole():
    inline = nodes.paragraph("", "", nodes.image(uri="a.svg", alt=_ALT))
    item = nodes.bullet_list("", nodes.list_item("", nodes.image(uri="b.svg", alt=_ALT)))
    note = nodes.note("", nodes.image(uri="c.svg", alt=_ALT))
    unbroken = "word " * 60
    document = _page(inline, item, note, _themed("d.svg", unbroken))
    a11y._describe(document, _DOCS)
    assert [image["alt"] for image in document.findall(nodes.image)] == [_ALT] * 3 + [unbroken] * 2
    assert not list(document.findall(nodes.raw))


def test_each_image_points_to_its_own_named_description():
    themed = [(file, alt) for _, kind, file, alt in _figures() if kind != "image"]
    figures = [(file, alt) for file, alt in themed if len(alt) > _LONG][:3]
    assert len(figures) == 3, "found too few figures with a long alt text in the docs"
    document = _page(*(_themed(file, alt) for file, alt in figures))
    # An id the page already uses is left alone.
    document.ids["figure-description-2"] = document[0]
    a11y._describe(document, _DOCS)
    page = _parsed(a11y._repair_body(_written(document)))
    descriptions = page.find_all("details", class_="figure-description")
    ids = [details["id"] for details in descriptions]
    assert ids == ["figure-description-1", "figure-description-3", "figure-description-4"]
    images = page.find_all("img")
    assert [image["aria-details"] for image in images] == [i for i in ids for _ in _THEMES]
    assert not [c for image in images for c in image.get("class", []) if c.startswith(a11y._TIE)]
    names = [details.find("summary")["aria-label"] for details in descriptions]
    assert len(set(names)) == 3, names
    for name, details, (file, alt) in zip(names, descriptions, figures, strict=True):
        # The name starts with the words on screen; the figure's title follows.
        assert name == f"Figure description: {_title(_DOCS / file)}", name
        summary = details.find("summary")
        assert summary.get_text() == "Figure description"
        # The name says the label's words, so the label is for the screen
        # alone: read out as well, it would follow the name as a description.
        assert [part.get("aria-hidden") for part in summary.find_all(True)] == ["true"]
        assert summary.find(True).get_text() == summary.get_text()
        assert details.find("p").get_text() == alt


def test_a_figure_without_a_caption_closes_none():
    alone = nodes.image(uri="c.svg", alt=_ALT)
    drawn = nodes.figure(
        "",
        nodes.raw("", '<div class="drawn"></div>', format="html"),
        nodes.raw("", "<table><tr><td>1</td></tr></table>", format="html"),
    )
    document = _page(_themed("a.svg", _ALT, caption=False), drawn, _themed("b.svg", _ALT), alone)
    a11y._describe(document, _DOCS)
    page = a11y._repair_body(_written(document))
    assert page.count("<figure") == page.count("</figure>") == 3
    assert page.count("<figcaption") == page.count("</figcaption>") == 1
    parsed = _parsed(page)
    assert len(parsed.select("figure details.figure-description")) == 2
    assert len(parsed.select("figcaption details.figure-description")) == 1
    # An image outside a figure has its description right after it, and no
    # class left over from pointing to it.
    bare = parsed.find("img", src="c.svg")
    assert bare.find_next_sibling("details")["id"] == bare["aria-details"]
    assert not bare.has_attr("class")


def test_a_numbered_figure_takes_an_anchor_from_its_image_and_keeps_its_number():
    first = _themed("figures/rabi_pulse.svg", "A pulse.")
    again = _themed("figures/rabi_pulse.svg", "The same pulse.")
    dark = nodes.figure("", nodes.image(uri="figures/T1 decay-dark.svg", alt="A decay."))
    dark += nodes.caption("", "The decay.")
    taken = _themed("figures/page.svg", "A page.")
    drawn = nodes.figure("", nodes.raw("", '<div class="drawn"></div>', format="html"))
    drawn += nodes.caption("", "Drawn by a script.")
    unnamed = _themed("figures/---.svg", "No name.")
    own = _themed("figures/levels.svg", "The levels.")
    bare = _themed("figures/bare.svg", "No caption.", caption=False)
    document = _page(first, again, dark, taken, drawn, unnamed, own, bare)
    # Sphinx numbers each figure that has a caption, in page order. A figure
    # without one has no anchor.
    numbered = (first, again, dark, taken, drawn, unnamed)
    for number, figure in enumerate(numbered, 1):
        figure["ids"] = [f"id{number}"]
    own["ids"] = ["the-levels"]
    for figure in (*numbered, own):
        document.ids[figure["ids"][0]] = figure
    document.ids["fig-page"] = document[0]
    assert a11y._anchor_figures(document) == 2
    after = [figure["ids"] for figure in (*numbered, own, bare)]
    assert after == [
        ["fig-rabi-pulse", "id1"],
        # The name is taken: by the same image further up, by another part
        # of the page.
        ["id2"],
        # A dark twin's file names the figure as the light one does.
        ["fig-t1-decay", "id3"],
        ["id4"],
        # No image, or a file name with no letter in it, gives no name.
        ["id5"],
        ["id6"],
        # An anchor that is no number stays the first.
        ["the-levels"],
        [],
    ]
    for figure in (*numbered, own):
        assert all(document.ids[anchor] is figure for anchor in figure["ids"])
    assert document.ids["fig-page"] is document[0]
    # The figure is written under its new anchor, which is where Sphinx
    # points the link after the caption, and the number still leads to it.
    page = _parsed(_written(document))
    old = page.find(id="id1")
    assert old.name == "span" and not old.get_text()
    assert old.parent.name == "figure" and old.parent["id"] == "fig-rabi-pulse"
    anchors = [tag["id"] for tag in page.find_all(id=True)]
    assert len(set(anchors)) == len(anchors)
    assert {"id1", "id2", "id3", "fig-t1-decay"} < set(anchors)
    assert a11y._anchor_figures(document) == 0
    assert [figure["ids"] for figure in (*numbered, own, bare)] == after


def test_no_image_is_a_link():
    document = publish.publish_doctree(
        ".. image:: a.svg\n   :target: https://example.org\n   :width: 50%\n\n"
        ".. figure:: b.svg\n   :target: https://example.org\n\n   The caption.\n\n"
        "A `link in words <https://example.org>`_ stays.\n",
        settings_overrides={"report_level": 5},
    )
    a11y._unlink_images(document)
    page = _parsed(_written(document))
    assert len(page.find_all("img")) == 2
    assert not [image for image in page.find_all("img") if image.find_parent("a")]
    assert [link.get_text() for link in page.find_all("a")] == ["link in words"]
    # Sphinx would link a scaled image to its file.
    config = SimpleNamespace(html_scaled_image_link=True)
    a11y._images_are_not_links(None, config)
    assert config.html_scaled_image_link is False


def test_printed_output_is_labelled_in_the_text_of_the_page():
    code = nodes.literal_block("", "print(1)")
    output = nodes.literal_block("", "1", classes=["output"])
    document = _page(code, output, nodes.literal_block("", "print(2)"))
    a11y._label_outputs(document)
    section = document[0]
    label = section[section.index(output) - 1]
    assert isinstance(label, nodes.paragraph) and label.astext() == "Output"
    assert len(list(document.findall(nodes.paragraph))) == 1
    assert '<p class="output-label">Output</p>' in _written(document)
    # The stylesheet styles the label and draws none of its own.
    css = (_DOCS / "_static" / "tutorial-output.css").read_text()
    assert re.search(r"^p\.output-label \{", css, re.M) and "content:" not in css


def test_each_sidebar_toggle_is_named_after_its_section():
    toctree = pytest.importorskip("pydata_sphinx_theme.toctree")
    # The page's own section, a section holding the page, and another one.
    entries = (
        '<ul><li class="toctree-l1 current"><a class="current reference internal" href="#">'
        'Tutorials</a><ul><li class="toctree-l2"><a href="a.html">First</a></li></ul></li>'
        '<li class="toctree-l1 current"><a class="reference internal" href="index.html">User '
        ' <em>Guide</em></a><ul><li class="toctree-l2 current"><a class="current reference '
        'internal" href="#">Operations</a></li></ul></li>'
        '<li class="toctree-l1"><a class="reference internal" href="../api/index.html">API '
        'Reference</a><ul><li class="toctree-l2"><a href="../api/k.html">Kernels</a></li></ul>'
        '</li><li class="toctree-l1"><a class="reference internal" href="faq.html">FAQ</a></li>'
        "</ul>"
    )

    def sidebar():
        tree = _parsed(entries)
        toctree.add_collapse_checkboxes(tree)
        return tree

    expected = ["Tutorials subsections", "User Guide subsections", "API Reference subsections"]
    for kind in (lambda tree: tree, str, lambda tree: markupsafe.Markup(str(tree))):
        page = kind(sidebar())
        named = a11y._name_toggles(page)
        assert type(named) is type(page)
        assert [s.get("aria-label") for s in _parsed(str(named)).find_all("summary")] == expected
    once = a11y._name_toggles(str(sidebar()))
    assert a11y._name_toggles(once) == once


def _signature(*lines: str, parens: str = "()") -> str:
    """A parameter list on several lines as Sphinx writes it: a ``<dl>`` with
    a ``<dd>`` for each line and no ``<dt>``."""
    rows = "".join(f"<dd>{line}</dd>\n" for line in lines)
    return (
        f'<span class="sig-paren">{parens[0]}</span>\n\n<dl>\n{rows}</dl>\n\n'
        f'<span class="sig-paren">{parens[1]}</span>'
    )


_PARAM = '<em class="sig-param"><span class="n"><span class="pre">{}</span></span></em>'
_SIGNATURES = {
    "a return annotation": (
        _signature(_PARAM.format("kernel") + ",", _PARAM.format("schedule"))
        + ' <span class="sig-return"><span class="sig-return-icon">&#x2192;</span> '
        '<span class="sig-return-typehint"><span class="pre">Program</span></span></span>',
        2,
    ),
    "a leading optional parameter": (
        _signature(
            '<span class="optional">[</span>' + _PARAM.format("start") + ",",
            _PARAM.format("step") + '<span class="optional">]</span>,',
            _PARAM.format("stop"),
        ),
        3,
    ),
    "parameters with no emphasis, as C++ has": (
        _signature(
            '<span class="kt"><span class="pre">int</span></span><span class="w"> </span>'
            '<span class="n"><span class="pre">count</span></span>,',
            '<span class="kt"><span class="pre">double</span></span><span class="w"> </span>'
            '<span class="n"><span class="pre">scale</span></span>',
        ),
        2,
    ),
    "type parameters": (
        _signature(_PARAM.format("T") + ",", _PARAM.format("U"), parens="[]")
        + _signature(_PARAM.format("value")),
        3,
    ),
}


@pytest.mark.parametrize("case", _SIGNATURES)
def test_a_signature_on_several_lines_is_no_list_without_terms(case):
    signature, lines = _SIGNATURES[case]
    page = (
        '<dl class="py function"><dt class="sig sig-object py" id="f">'
        f'<span class="sig-name descname"><span class="pre">f</span></span>{signature}</dt>'
        "<dd><p>What it does.</p></dd></dl>"
    )
    repaired = _parsed(a11y._repair_body(page))
    # The object's own list of a term and its description stays.
    assert [bool(dl.find("dt", recursive=False)) for dl in repaired.find_all("dl")] == [True]
    assert len(repaired.select("dt > div.sig-params > div.sig-param-line")) == lines
    assert len(repaired.find_all("dd")) == 1
    assert repaired.get_text().split() == _parsed(page).get_text().split()


def test_a_power_of_a_number_reads_as_a_power():
    body = "from 1 to 2<sup>40</sup>; −2<sup>63</sup> wraps and 10<sup>-3</sup> is small."
    kept = " x<sup>2</sup>, 2<sup>n</sup> and a note<sup>1</sup>"
    repaired = a11y._repair_body(body + kept)
    assert repaired.endswith(kept)
    assert _parsed(repaired).get_text() == (
        "from 1 to 2 to the power 40; −2 to the power 63 wraps and 10 to the power -3 is small."
        " x2, 2n and a note1"
    )
    assert repaired.count('<sup><span class="visually-hidden"> to the power </span>') == 3


def test_the_link_after_a_caption_is_named_after_the_caption():
    link = '<a class="headerlink" href="#{}" title="Link to this {}">#</a>'
    page = (
        '<p class="caption" role="heading"><span class="caption-text">User Guide</span></p>'
        "<h2>Install" + link.format("install", "heading") + "</h2>"
        '<figcaption>\n<p><span class="caption-text">The schedule of <code class="docutils '
        'literal notranslate"><span class="pre">inject</span></code>. The diamond is\nthe T '
        "gate.</span>" + link.format("id1", "image") + "</p>\n</figcaption>"
        '<figcaption>\n<p><span class="caption-text">The mean of the first n shots against n, '
        "with a band\nof two standard errors.</span>" + link.format("id2", "image") + "</p>\n"
        '</figcaption><table><caption><span class="caption-text">Drivers &amp; &quot;extra'
        "&quot; libraries</span>" + link.format("id3", "table") + "</caption></table>"
    )
    repaired = a11y._repair_body(page)
    links = _parsed(repaired).find_all("a", class_="headerlink")
    assert [link["href"] for link in links] == ["#install", "#id1", "#id2", "#id3"]
    # A heading's link is not shown apart from its heading and keeps its name.
    assert links[0]["title"] == "Link to this heading" and not links[0].has_attr("aria-label")
    assert [link["aria-label"] for link in links[1:]] == [
        "Link to figure: The schedule of inject",
        "Link to figure: The mean of the first n shots against n",
        'Link to table: Drivers & "extra" libraries',
    ]
    # A tooltip of the same words would follow the name as a description.
    assert not [link for link in links[1:] if link.has_attr("title")]
    assert _parsed(repaired).get_text() == _parsed(page).get_text()
    assert a11y._repair_body(repaired) == repaired


def test_the_name_of_a_captions_link_ends_with_a_sentence_or_a_clause():
    start = a11y._caption_start
    # A first sentence that fits is the name, without its full stop.
    short = "t1 with a 10 µs wait, cut short. The dashed line marks the cut."
    assert start(short) == "t1 with a 10 µs wait, cut short"
    assert start("Supported platforms") == "Supported platforms"
    # A longer one ends before the last break between clauses that fits.
    long = (
        "The schedule of one call, in nanoseconds: each pulse is its envelope as the "
        "controller plays it, bar by bar."
    )
    assert len(long) > a11y.SHORT_NAME
    assert start(long) == "The schedule of one call, in nanoseconds"
    # A comma in a list of numbers is no break between clauses.
    values = "The levels of the pair, as the simulator keeps them at 5.10, 5.25 and 5.40 GHz."
    assert len(values) > a11y.SHORT_NAME and start(values) == "The levels of the pair"
    # A break in the first words would leave too little. With no other, the
    # sentence is cut after eight words and the cut is marked.
    listed = "Here, the levels 0, 1 and 2 of each qudit of the pair after the last pulse plays."
    assert start(listed) == "Here, the levels 0, 1 and 2 of…"
    # Eight long words are the whole sentence, so nothing is cut.
    whole = "Simultaneous randomized benchmarking of neighbouring superconducting transmon qudits."
    assert len(whole) > a11y.SHORT_NAME and start(whole) == whole[:-1]


def test_the_404_page_resolves_from_the_site_root_and_keeps_its_own_anchors():
    body = (
        '<h1>Page not found<a class="headerlink" href="#page-not-found">#</a></h1>'
        '<a class="reference internal" href="index.html">Home</a>'
    )
    context = _context("404", body=body)
    assert context["body"] == body
    # The page is served at any address that was not found, and its links
    # resolve from its own address on the site.
    site = "https://example.org" + a11y.SITE_ROOT
    base = "https://example.org" + _parsed(context["metatags"]).find("base")["href"]
    assert urljoin(base, "_static/a11y.css") == site + "_static/a11y.css"
    assert urljoin(base, "index.html") == site + "index.html"
    assert urljoin(base, "#page-not-found") == site + "404.html#page-not-found"
    addresses = {
        "https://example.org/docs/": "/docs/404.html",
        "https://example.org/docs": "/docs/404.html",
        "https://example.org/": "/404.html",
        "https://example.org": "/404.html",
    }
    for baseurl, address in addresses.items():
        metatags = _context("404", baseurl, body=body)["metatags"]
        assert metatags.endswith(f'<base href="{address}">'), baseurl
    # Any other page keeps its own address as the base of its links.
    other = _context("index", "https://example.org/docs/", body=body)
    assert "<base" not in other["metatags"] and other["body"] == body


def test_the_themes_skip_link_stays_on_the_404_page():
    spec = importlib.util.find_spec("pydata_sphinx_theme")
    if spec is None:
        pytest.skip("needs pydata-sphinx-theme")
    layout = Path(spec.origin).parent / "theme" / "pydata_sphinx_theme" / "layout.html"
    if not layout.is_file():
        pytest.skip("the theme lays its pages out from another template")
    skip = re.search(r'<div id="pst-skip-link"[^>]*><a href="([^"]+)">', layout.read_text())
    assert skip, "the theme no longer writes its skip link this way"
    base = _parsed(_context("404", body="")["metatags"]).find("base")["href"]
    # From the site root alone the link would lead to the home page.
    there = urljoin("https://example.org" + base, skip.group(1))
    assert there == f"https://example.org{a11y.SITE_ROOT}404.html#main-content"


def test_the_404_page_is_in_neither_the_sidebar_nor_the_search():
    document = publish.publish_doctree(
        (_DOCS / "404.rst").read_text(), settings_overrides={"report_level": 5}
    )
    info = next(iter(document.findall(nodes.docinfo)))
    assert {"orphan", "nosearch"} <= {field[0].astext() for field in info.findall(nodes.field)}


def test_without_a_script_the_navigation_follows_the_page():
    metatags = _context("index", metatags="<meta>", body="")["metatags"]
    assert metatags.startswith("<meta><noscript><style>") and metatags.endswith("</noscript>")
    rules = _parsed(metatags).find("style").get_text()
    # The rules stand in for the theme's, so they name what the theme styles.
    theme = _theme_css()
    assert "@media (max-width:959.98px){.bd-sidebar-primary{" in theme
    assert "@media screen and (max-width:959.98px){" in rules
    for selector in (
        ".bd-header button.sidebar-toggle{",
        ".bd-container .bd-container__inner{",
        ".bd-sidebar-primary{",
    ):
        assert selector in rules and selector in theme, selector
    # In the column the navigation is as tall as its links, whatever share of
    # the row the theme gives it.
    assert re.search(r"\.bd-sidebar-primary\{[^}]*\bflex:0 0 auto!important", rules)


def test_without_a_script_the_search_page_shows_no_field():
    jinja2 = pytest.importorskip("jinja2")
    spec = importlib.util.find_spec("pydata_sphinx_theme")
    if spec is None:
        pytest.skip("needs pydata-sphinx-theme")
    templates = Path(spec.origin).parent / "theme" / "pydata_sphinx_theme"
    page, field = templates / "search.html", templates / "components" / "search-field.html"
    if not (page.is_file() and field.is_file()):
        pytest.skip("the theme lays its search page out from other templates")
    selector = ".bd-search-container form.bd-search"
    rules = _parsed(_context("search", body="")["metatags"]).find("style").get_text()
    assert selector + "{display:none!important}" in rules
    # The search page holds its field in a box of its own. Every page has the
    # same field once more, in the dialog that the header's button opens.
    box = r'<div class="bd-search-container">.*?\{% include "[./]*components/search-field\.html"'
    assert re.search(box, page.read_text(), re.S), "the theme writes its search page another way"
    form = jinja2.Template(field.read_text()).render(
        pathto=lambda *args: "search.html", theme_search_bar_text="Search"
    )
    written = _parsed(
        f'<dialog id="pst-search-dialog">{form}</dialog>'
        f'<div class="bd-search-container">{form}</div>'
    )
    assert len(written.select("form.bd-search")) == 2
    (hidden,) = written.select(selector)
    assert hidden.find_parent("dialog") is None


def test_the_theme_has_the_class_for_words_that_only_a_screen_reader_says():
    # The extension writes such words under the theme's class and has no
    # rule of its own for them.
    assert 'class="visually-hidden"' in a11y._SPOKEN_POWER
    assert re.search(r"(?<![\w-])\.visually-hidden(?![\w-])[^{}]*\{", _theme_css())


def test_the_second_skip_link_goes_where_on_this_page_becomes_a_drawer():
    link = r"#pst-skip-link a\.a11y-skip-toc \{\s*display: none;"
    hidden = re.search(r"@media \(max-width: ([\d.]+px)\) \{\s*" + link, _own_css())
    assert hidden, "a11y.css no longer hides the second skip link below a width"
    # Below that width the theme takes the list off the page until a button
    # opens it.
    width = hidden.group(1)
    rule = r"@media \(max-width:" + re.escape(width) + r"\)\{\.bd-sidebar-secondary\{([^}]*)\}"
    drawer = re.search(rule, _theme_css())
    assert drawer, f"the theme no longer makes a drawer of the list at {width}"
    assert "visibility:hidden" in drawer.group(1)


def test_a_page_printed_from_the_dark_theme_takes_the_light_themes_colors():
    sheet = (_DOCS / "_static" / "a11y-print.css").read_text()
    sheet = re.sub(r"/\*.*?\*/", "", sheet, flags=re.S)
    blocks = re.findall(r'html\[data-theme="dark"\]:root\s*\{([^{}]*)\}', sheet)
    printed = dict(re.findall(r"(--pst-color-[\w-]+):\s*(#[0-9a-fA-F]{3,6})\b", "".join(blocks)))
    assert printed, "the print stylesheet sets no theme color for a page in the dark theme"
    light = _theme_colors("light")
    other = {
        name: (color, light.get(name))
        for name, color in printed.items()
        if name not in light or _six_digits(color) != _six_digits(light[name])
    }
    assert not other, f"not the light theme's: {other}"


def test_the_home_link_names_the_logo_and_the_title_once_each():
    jinja2 = pytest.importorskip("jinja2")
    spec = importlib.util.find_spec("pydata_sphinx_theme")
    if spec is None:
        pytest.skip("needs pydata-sphinx-theme")
    template = next(Path(spec.origin).parent.rglob("components/navbar-logo.html"), None)
    if template is None:
        pytest.skip("the theme draws its logo from another template")
    logo = {
        "image_relative": {"light": "light.svg", "dark": "dark.svg"},
        "text": "The Title",
        "alt_text": "The Title - Home",
    }
    context = _context("index", theme_logo=logo, body="")
    written = jinja2.Template(template.read_text()).render(
        context, pathto=lambda *args: "index.html", root_doc="index", docstitle="The Title"
    )
    link = _parsed(written).find("a")
    assert {image["alt"] for image in link.find_all("img")} == {a11y.LOGO_ALT}
    assert link.get_text(strip=True) == "The Title"


def test_the_extension_loads_files_that_exist():
    loaded = []
    app = SimpleNamespace(
        add_css_file=lambda name, **attributes: loaded.append((name, attributes)),
        add_js_file=lambda name, **attributes: loaded.append((name, attributes)),
        connect=lambda *args, **kwargs: None,
    )
    meta = a11y.setup(app)
    assert ("a11y-print.css", {"media": "print"}) in loaded
    assert not [name for name, _ in loaded if not (_DOCS / "_static" / name).is_file()]
    assert meta["parallel_read_safe"] and meta["parallel_write_safe"]


def test_a_figure_gets_its_anchor_as_its_page_is_read():
    heard = {}
    app = SimpleNamespace(
        add_css_file=lambda *args, **kwargs: None,
        add_js_file=lambda *args, **kwargs: None,
        connect=lambda event, handler, **kwargs: heard.setdefault(event, handler),
    )
    a11y.setup(app)
    figure = _themed("figures/rabi_pulse.svg", "A pulse.")
    figure["ids"] = ["id1"]
    # The stored page then holds the anchor, and Sphinx writes the link after
    # the caption to it.
    heard["doctree-read"](None, _page(figure))
    assert figure["ids"] == ["fig-rabi-pulse", "id1"]


@pytest.mark.parametrize("theme", _THEMES)
def test_the_focus_ring_shows_on_the_page_and_on_its_surfaces(theme):
    colors = _theme_colors(theme)
    low = {
        surface: round(_contrast(_ring(theme), colors[surface]), 2)
        for surface in ("--pst-color-background", "--pst-color-surface")
        if _contrast(_ring(theme), colors[surface]) < 3
    }
    assert not low, f"{_ring(theme)} is below 3:1 on {low}"


@pytest.mark.parametrize("theme", _THEMES)
def test_the_skip_links_ring_shows_on_its_strip(theme):
    colors = _theme_colors(theme)
    strip = re.search(r"\.skip-link\{background-color:var\((--[\w-]+)\)", _theme_css())
    assert strip, "the theme no longer colors the skip link's strip from a variable"
    # The ring is the general one unless a rule gives the skip link its own.
    ring = _ring(theme)
    own = re.search(r"\.skip-link a:focus-visible\s*\{[^}]*outline-color:\s*([^;!}]+)", _own_css())
    if own:
        variable = re.match(r"var\((--[\w-]+)(?:,\s*(#[0-9a-fA-F]{3,6}))?\)", own.group(1).strip())
        ring = (
            colors.get(variable.group(1), variable.group(2)) if variable else own.group(1).strip()
        )
    assert _contrast(ring, colors[strip.group(1)]) >= 3, (ring, colors[strip.group(1)])


def test_an_icon_kept_from_screen_readers_is_the_one_the_theme_draws():
    # a11y.css repeats the theme's rule for the icon before the title of each
    # kind of note, with an empty alternative text. Under another selector
    # the theme's rule would win. With a variable the theme does not set, or
    # in a browser that does not know the form, the note would lose its icon,
    # so the rules apply only where the browser knows the form.
    own, theme = _own_css(), _theme_css()
    known = re.search(r'@supports \(content: "x" / ""\) \{\n(.*?)\n\}', own, re.S)
    assert known, "a11y.css no longer asks whether the browser knows the form"
    rule = (
        r"(div\.admonition\.[\w-]+) > (\.admonition-title)::after \{\s*"
        r'content: var\((--[\w-]+)\) / "";\s*\}'
    )
    hidden = re.findall(rule, known.group(1))
    assert hidden, "a11y.css gives no note icon an empty alternative text"
    assert len(hidden) == own.count(".admonition-title::after"), (
        "a11y.css has a rule for a note's icon in another form, or outside the question"
    )
    for box, title, variable in hidden:
        drawn = re.escape(f"{box}>{title}:after") + r"\{[^}]*content:var\(" + variable + r"\)"
        assert re.search(drawn, theme), f"the theme does not draw {box} from {variable}"
        assert variable + ":" in theme, f"the theme does not set {variable}"
    # The arrow before a download link and the divider between breadcrumbs
    # are kept from screen readers the same way, by the theme's own rules.
    others = re.findall(
        r'^  ([^\n{]+)::before \{\s*content: var\((--[\w-]+)\) / "";\s*\}', known.group(1), re.M
    )
    assert len(hidden) + len(others) == known.group(1).count(' / "";'), (
        "a11y.css keeps an icon from screen readers by a rule of another form"
    )
    for selector, variable in others:
        drawn = re.escape(f"{selector}:before") + r"\{[^}]*content:var\(" + variable + r"\)"
        assert re.search(drawn, theme), f"the theme does not draw {selector} from {variable}"
        assert variable + ":" in theme, f"the theme does not set {variable}"


def test_a_color_the_print_stylesheet_takes_from_a11y_css_is_set_there():
    sheet = (_DOCS / "_static" / "a11y-print.css").read_text()
    used = set(re.findall(r"var\((--a11y-[\w-]+)\)", sheet))
    assert used, "the print stylesheet no longer takes a color from a11y.css"
    # What a11y.css sets on :root reaches paper: it is linked for all media.
    root = re.search(r":root\s*\{([^{}]*)\}", _own_css()).group(1)
    unset = sorted(name for name in used if not re.search(name + r":\s*#[0-9a-fA-F]{6};", root))
    assert not unset, f"a11y.css sets no color for {unset}"
