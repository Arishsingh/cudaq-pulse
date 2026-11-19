# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dialect reference pages get their tables, anchors, links and colors.

``docs/_ext/dialect_pages.py`` repairs the pages the build generates under
``architecture/dialects/``, and ``docs/_ext/mlir_lexer.py`` colors their MLIR
examples. None of this needs a build: the repairs run on a page that docutils
parses here in the generator's layout and on the HTML the generator and the
theme write for a table, and the lexer reads the examples of the operation
definitions, which the generator copies into the pages.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

nodes = pytest.importorskip("docutils.nodes")
publish = pytest.importorskip("docutils.core")
html5_polyglot = pytest.importorskip("docutils.writers.html5_polyglot")
bs4 = pytest.importorskip("bs4")
token = pytest.importorskip("pygments.token")
pytest.importorskip("sphinx")

_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "docs"
_EXT = _DOCS / "_ext"
pytestmark = pytest.mark.skipif(not _EXT.exists(), reason="needs the source tree")
sys.path.insert(0, str(_EXT))
dialect_pages = pytest.importorskip("dialect_pages")
mlir_lexer = pytest.importorskip("mlir_lexer")
code_style = pytest.importorskip("code_style")
sys.path.remove(str(_EXT))

_PAGE = dialect_pages.PAGES + "pulse"

# A page in the generator's layout: the dialect, its groups, an entry for
# each operation or attribute, and under an entry the sub-headings that many
# entries share.
_SOURCE = """\
'pulse' Dialect
===============

The role's :code:`pulse.wait` holds a line. See ``pulse.drive``, ``pulse.drive``
again, the attribute ``pulse.quantum_system`` and ``pulse.wait``.

Operations
----------

``pulse.drive`` (::pulse::DriveOp)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plays a waveform: ``pulse.drive`` follows a ``pulse.wait``.

Attributes:
^^^^^^^^^^^

.. raw:: html

   <table>
   <tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
   <tr><td><code>start</code></td><td>::mlir::IntegerAttr</td><td>64-bit integer</td></tr>
   </table>

Operands:
^^^^^^^^^

A drive line.

``pulse.wait`` (::pulse::WaitOp)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operands:
^^^^^^^^^

A line.

Results:
^^^^^^^^

The line.

Attributes
----------

ControlSystemAttr
~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

A clock.
"""

# A section that names two operations and then lists them, as the overview
# of the dialects lists the operations of each.
_LISTED = """\
Pulse
=====

A ``pulse.wait`` follows a ``pulse.drive``.

- Ops: ``pulse.drive``, ``pulse.wait`` and ``pulse.drive`` again.
- Frames: a ``pulse.wait`` holds one.

  The ``pulse.wait`` of the same item.

  - Deeper: ``pulse.wait`` and ``pulse.sync``.

After the list, ``pulse.wait`` and ``pulse.drive`` stay as written, and
``pulse.sync`` is new to the section.
"""

# The same table as the generator writes it and as the theme writes one.
_BARE = """\
<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>start</code></td><td>::mlir::IntegerAttr</td><td>64-bit integer</td></tr>
<tr><td><code>length</code></td><td>::mlir::IntegerAttr</td><td>64-bit integer</td></tr>
</table>
"""
_THEMED = """\
<div class="pst-scrollable-table-container"><table class="table">
<thead>
<tr class="row-odd"><th class="head text-center"><p>Operand</p></th>
<th class="head"><p>Description</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td class="text-center"><p><code>line</code></p></td>
<td><p>A drive line</p></td>
</tr>
</tbody>
</table>
</div>
"""

_EXAMPLE = """\
// One pulse on a drive line.
%d1, %t1 = pulse.drive %d0, %wf, %t0 {start = 40 : i64, scale = -2.5e-01}
    : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 500000>} {
  func.func @main() { ^bb0: return }
  %fix = classical.wait %answer, @decode {mode = "bounded", lazy = false} : i1
}
"""


def _page():
    """The page of ``_SOURCE`` as Sphinx reads one: the title stays a section,
    and a heading whose words are taken is numbered ``id1``."""
    settings = {"doctitle_xform": False, "auto_id_prefix": "id", "report_level": 5}
    return publish.publish_doctree(_SOURCE, settings_overrides=settings)


def _anchors(document) -> dict[str, list[str]]:
    """The anchors of each section of *document*, by its heading and the
    heading of the section it is in."""
    found = {}
    for section in document.findall(nodes.section):
        above = section.parent[0].astext() if isinstance(section.parent, nodes.section) else ""
        found[f"{above} > {section[0].astext()}"] = list(section["ids"])
    return found


def _written(document) -> str:
    """The HTML docutils writes for *document*."""
    writer = html5_polyglot.Writer()
    publish.publish_from_doctree(document, writer=writer, settings_overrides={"report_level": 5})
    return writer.parts["body"]


def _parsed(text: str):
    return bs4.BeautifulSoup(text, "html.parser")


def _app(docname: str = _PAGE):
    """An application that is reading or writing the page *docname*, with
    the labels its pages have."""
    domain = SimpleNamespace(labels={}, anonlabels={})
    return SimpleNamespace(
        env=SimpleNamespace(docname=docname, get_domain=lambda name: domain),
        builder=SimpleNamespace(get_relative_uri=lambda page, to: f"../{to}.html"),
        labels=domain.anonlabels,
    )


def _links(document) -> list[tuple[str, str]]:
    """(text, target) of each link of *document*."""
    return [
        (link.astext(), link.get("refuri") or "#" + link["refid"])
        for link in document.findall(nodes.reference)
    ]


def _theme_file(name: str) -> str:
    spec = importlib.util.find_spec("pydata_sphinx_theme")
    if spec is None:
        pytest.skip("needs pydata-sphinx-theme")
    return next(Path(spec.origin).parent.rglob(name)).read_text()


def _examples() -> list[tuple[str, str]]:
    """(where, text) of each MLIR example of the operation definitions and of
    the pages."""
    found = []
    for file in sorted((_ROOT / "core" / "mlir" / "include").rglob("*.td")):
        fences = re.findall(r"```mlir\n(.*?)```", file.read_text(), re.S)
        found += [(file.name, text) for text in fences]
    for page in sorted(_DOCS.rglob("*.rst")):
        blocks = re.findall(r"\.\. code-block:: mlir\n\n((?:[ \t]+.*\n|\n)+)", page.read_text())
        found += [(page.name, text) for text in blocks]
    return found


def test_a_bare_table_is_written_as_the_theme_writes_one():
    body = f"<p>Before.</p>\n{_BARE}{_THEMED}<p>After.</p>"
    wrapped = dialect_pages.wrap_tables(body)
    page = _parsed(wrapped)
    tables = page.find_all("table")
    assert len(tables) == 2
    for table in tables:
        assert table["class"] == ["table"]
        assert table.parent.name == "div" and table.parent["class"] == [dialect_pages.SCROLL_BOX]
        assert [child.name for child in table.find_all(recursive=False)] == ["thead", "tbody"]
        assert len(table.thead.find_all("tr")) == 1 and not table.thead.find("td")
        assert not table.tbody.find("th")
    assert [len(table.tbody.find_all("tr")) for table in tables] == [2, 1]
    assert page.get_text() == _parsed(body).get_text()
    # A table the theme wrote stays as it is, and so does one written twice.
    assert wrapped.endswith(f"{_THEMED}<p>After.</p>")
    assert dialect_pages.wrap_tables(wrapped) == wrapped


def test_a_bare_table_without_a_header_row_is_boxed_too():
    table = _parsed(dialect_pages.wrap_tables("<table>\n<tr><td>1</td></tr>\n</table>")).table
    assert table["class"] == ["table"] and table.parent["class"] == [dialect_pages.SCROLL_BOX]
    assert [child.name for child in table.find_all(recursive=False)] == ["tbody"]
    # A table that has its parts is not the generator's, and stays as written.
    parted = "<table>\n<thead><tr><th>1</th></tr></thead><tbody><tr><td>2</td></tr></tbody></table>"
    assert dialect_pages.wrap_tables(parted) == parted


def test_the_box_and_the_table_class_are_the_themes():
    css = _theme_file("pydata-sphinx-theme.css")
    assert f".{dialect_pages.SCROLL_BOX}{{overflow-x:auto}}" in css
    # The header row and the stripes are drawn for these parts of a table.
    assert ".table thead tr{" in css and ".table tbody tr:nth-child(odd){" in css
    assert f'<div class="{dialect_pages.SCROLL_BOX}">' in _theme_file("translator.py")


def test_each_table_is_named_after_its_entry_and_its_sub_heading():
    link = '<a class="headerlink" href="#{}" title="Link to this heading">#</a>'
    code = '<code class="docutils literal notranslate"><span class="pre">{}</span></code>'
    body = (
        f"<h1>‘pulse’ Dialect{link.format('pulse-dialect')}</h1>"
        f"<h2>Operations{link.format('operations')}</h2>"
        f"<h3>{code.format('pulse.drive')} (::pulse::DriveOp){link.format('a')}</h3>"
        '<table class="table"><tr><td>under no sub-heading</td></tr></table>'
        f"<h4>Attributes:{link.format('b')}</h4>{_BARE}"
        f"<h4>Operands:{link.format('c')}</h4>{_THEMED}"
        f"<h3>{code.format('pulse.wait')} (::pulse::WaitOp){link.format('d')}</h3>"
        f"<h4>Results:{link.format('e')}</h4>{_THEMED}"
        '<table class="table" aria-label="Its own name"><tr><td>1</td></tr></table>'
        '<table class="table">\n<caption>Its own caption</caption><tr><td>1</td></tr></table>'
        f"<h2>Attributes{link.format('f')}</h2>"
        f"<h3>Control&lt;System&gt;Attr{link.format('g')}</h3>"
        f"<h4>Parameters:{link.format('h')}</h4>{_THEMED}"
        f"<h3>Another{link.format('i')}</h3>{_THEMED}"
    )
    named = dialect_pages.name_tables(dialect_pages.wrap_tables(body))
    assert [table.get("aria-label") for table in _parsed(named).find_all("table")] == [
        None,
        "pulse.drive attributes",
        "pulse.drive operands",
        "pulse.wait results",
        "Its own name",
        None,
        "Control<System>Attr parameters",
        None,
    ]
    assert _parsed(named).get_text() == _parsed(body).get_text()
    assert dialect_pages.name_tables(named) == named


def test_only_the_dialect_pages_are_rewritten():
    body = f"<h3>An entry</h3><h4>Operands:</h4>{_BARE}"
    for pagename, rewritten in ((_PAGE, True), ("architecture/dialects", False), ("index", False)):
        context = {"body": body}
        dialect_pages._page_context(None, pagename, "page.html", context, None)
        assert (context["body"] != body) is rewritten, pagename
    # A page with no text of its own, the search page, has no body.
    dialect_pages._page_context(None, _PAGE, "search.html", {}, None)


def test_each_sub_heading_has_an_anchor_of_its_own_and_keeps_the_one_it_had():
    document = _page()
    before = _anchors(document)
    # Docutils gives the shared words to the first heading and numbers the rest.
    assert before["pulse.drive (::pulse::DriveOp) > Operands:"] == ["operands"]
    assert before["pulse.wait (::pulse::WaitOp) > Operands:"] == ["id1"]
    assert before["'pulse' Dialect > Attributes"] == ["id2"]
    assert dialect_pages.anchor_parts(document) == 6
    after = _anchors(document)
    expected = {
        "pulse.drive (::pulse::DriveOp) > Attributes:": "pulse-drive-pulse-driveop-attributes",
        "pulse.drive (::pulse::DriveOp) > Operands:": "pulse-drive-pulse-driveop-operands",
        "pulse.wait (::pulse::WaitOp) > Operands:": "pulse-wait-pulse-waitop-operands",
        "pulse.wait (::pulse::WaitOp) > Results:": "pulse-wait-pulse-waitop-results",
        "'pulse' Dialect > Attributes": "pulse-dialect-attributes",
        "ControlSystemAttr > Parameters:": "controlsystemattr-parameters",
    }
    for heading, anchors in before.items():
        if heading in expected:
            assert after[heading] == [expected[heading], *anchors], heading
        else:
            assert after[heading] == anchors, heading
    for section in document.findall(nodes.section):
        assert all(document.ids[anchor] is section for anchor in section["ids"])
    # Each heading is written under its new anchor, and the old one still
    # leads to its section.
    page = _parsed(_written(document))
    old = page.find(id="id1")
    assert old.name == "span" and not old.get_text()
    assert old.parent.name == "section" and old.parent["id"] == "pulse-wait-pulse-waitop-operands"
    assert old.find_next_sibling().get_text() == "Operands:"
    assert dialect_pages.anchor_parts(document) == 0 and _anchors(document) == after


def test_an_anchor_that_is_taken_is_left_alone():
    document = _page()
    document.ids["pulse-wait-pulse-waitop-results"] = document
    assert dialect_pages.anchor_parts(document) == 5
    assert _anchors(document)["pulse.wait (::pulse::WaitOp) > Results:"] == ["results"]


def test_the_entry_of_an_operation_has_its_name_as_an_anchor():
    document = _page()
    entries = dialect_pages.anchor_operations(document)
    assert entries == {
        "pulse.drive": "pulse-drive-pulse-driveop",
        "pulse.wait": "pulse-wait-pulse-waitop",
    }
    anchors = _anchors(document)
    assert anchors["Operations > pulse.drive (::pulse::DriveOp)"] == [
        "pulse-drive-pulse-driveop",
        "pulse.drive",
    ]
    # An entry that is no operation's has the anchor it had.
    assert anchors["Attributes > ControlSystemAttr"] == ["controlsystemattr"]
    assert _parsed(_written(document)).find(id="pulse.wait").parent["id"] == entries["pulse.wait"]
    assert dialect_pages.anchor_operations(document) == entries and _anchors(document) == anchors


def test_reading_a_dialect_page_labels_its_operations():
    app, document = _app(), _page()
    dialect_pages._anchor(app, document)
    domain = app.env.get_domain("std")
    drive, wait = (dialect_pages.LABEL + name for name in ("pulse.drive", "pulse.wait"))
    assert sorted(domain.labels) == sorted(domain.anonlabels) == [drive, wait]
    assert domain.anonlabels[drive] == (_PAGE, "pulse-drive-pulse-driveop")
    # The ``ref`` role shows the operation's name.
    assert domain.labels[drive] == (_PAGE, "pulse-drive-pulse-driveop", "pulse.drive")
    assert _anchors(document)["pulse.wait (::pulse::WaitOp) > Operands:"][1] == "id1"
    # Any other page is read as it is.
    other_app, other = _app("architecture/dialects"), _page()
    dialect_pages._anchor(other_app, other)
    assert _anchors(other) == _anchors(_page()) and not other_app.labels


def test_the_first_mention_of_an_operation_in_a_section_links_to_its_entry():
    document = _page()
    entries = dialect_pages.anchor_operations(document)
    asked = []

    def link_to(name):
        asked.append(name)
        return nodes.reference("", "", refuri="#" + entries[name]) if name in entries else None

    assert dialect_pages.link_operations(document, link_to) == 3
    # The page's opening names both operations, one of them twice, after a
    # role has written the name of the other; the entry of ``pulse.drive``
    # names itself and ``pulse.wait``.
    assert _links(document) == [
        ("pulse.drive", "#pulse-drive-pulse-driveop"),
        ("pulse.wait", "#pulse-wait-pulse-waitop"),
        ("pulse.wait", "#pulse-wait-pulse-waitop"),
    ]
    assert "pulse.quantum_system" in asked
    for link in document.findall(nodes.reference):
        assert isinstance(link[0], nodes.literal) and isinstance(link.parent, nodes.paragraph)
    # The headings name the operations and stay as they are.
    titles = list(document.findall(nodes.title))
    assert not [link for title in titles for link in title.findall(nodes.reference)]
    assert dialect_pages.link_operations(document, link_to) == 0


def test_a_heading_that_names_another_operation_stays_as_it_is():
    source = "After ``pulse.wait``\n====================\n\nA ``pulse.wait`` ends.\n"
    document = publish.publish_doctree(source, settings_overrides={"doctitle_xform": False})
    assert dialect_pages.link_operations(document, lambda name: nodes.reference("", "")) == 1
    assert [type(link.parent) for link in document.findall(nodes.reference)] == [nodes.paragraph]


def test_a_link_the_page_has_is_its_sections_mention():
    document = _page()
    entries = dialect_pages.anchor_operations(document)
    opening = next(iter(document.findall(nodes.paragraph)))
    own = nodes.reference("", "", nodes.literal("", "pulse.drive"), refuri="https://example.org")
    opening.insert(0, own)

    def link_to(name):
        return nodes.reference("", "", refuri="#" + entries[name]) if name in entries else None

    assert dialect_pages.link_operations(document, link_to) == 2
    assert _links(document)[:2] == [
        ("pulse.drive", "https://example.org"),
        ("pulse.wait", "#pulse-wait-pulse-waitop"),
    ]


def test_an_item_of_a_list_links_its_own_first_mention():
    document = publish.publish_doctree(_LISTED, settings_overrides={"doctitle_xform": False})
    made = dialect_pages.link_operations(document, lambda name: nodes.reference("", ""))
    linked = [
        [link.astext() for link in paragraph.findall(nodes.reference)]
        for paragraph in document.findall(nodes.paragraph)
    ]
    assert linked == [
        ["pulse.wait", "pulse.drive"],
        # Each item links what it names, once, though the opening named both.
        ["pulse.drive", "pulse.wait"],
        ["pulse.wait"],
        [],
        # An item inside an item counts for itself too.
        ["pulse.wait", "pulse.sync"],
        # After the list the text is the section's again, and an operation
        # that only an item has named is new to it.
        ["pulse.sync"],
    ]
    assert made == 8
    assert dialect_pages.link_operations(document, lambda name: nodes.reference("", "")) == 0


def test_every_page_links_the_operations_it_names():
    app, page = _app(), _page()
    dialect_pages._anchor(app, page)
    dialect_pages._link(app, page, _PAGE)
    assert {target for _, target in _links(page)} == {
        "#pulse-wait-pulse-waitop",
        "#pulse-drive-pulse-driveop",
    }
    # Another page names an operation, something that is none, and a call.
    entry = ("pulse.wait", f"../{_PAGE}.html#pulse-wait-pulse-waitop")
    for docname in ("architecture/pipeline", "user_guide/kernels", "api/controller"):
        other = publish.publish_doctree("A ``pulse.wait``, a ``pulse.modes``, a ``pulse.wait(4)``.")
        dialect_pages._link(app, other, docname)
        assert _links(other) == [entry], docname


def test_the_pages_are_where_the_build_generates_them():
    cmake = (_ROOT / "cmake" / "CudaqPulseDocumentation.cmake").read_text()
    output = re.search(r"-DOUTPUT_DIR=\$\{\w+\}/(\S+)", cmake)
    assert output and output.group(1) + "/" == dialect_pages.PAGES


def test_the_extension_registers_its_lexer_and_reads_before_the_contents_are_taken():
    lexers, handlers = {}, {}
    app = SimpleNamespace(
        add_lexer=lexers.__setitem__,
        connect=lambda event, handler, priority=500: handlers.update({event: priority}),
    )
    meta = dialect_pages.setup(app)
    assert lexers == {"mlir": mlir_lexer.MlirLexer}
    # Sphinx collects a page's contents on this event at 500.
    assert handlers["doctree-read"] < 500
    assert {"doctree-resolved", "html-page-context"} <= set(handlers)
    assert meta["parallel_read_safe"] and meta["parallel_write_safe"]


def test_the_lexer_reads_every_example():
    if not (_ROOT / "core" / "mlir").exists():
        pytest.skip("needs the operation definitions")
    examples = _examples()
    assert examples, "found no MLIR example in the operation definitions"
    lexer = mlir_lexer.MlirLexer()
    for where, text in [*examples, ("this file", _EXAMPLE)]:
        tokens = list(lexer.get_tokens(text))
        assert not [value for kind, value in tokens if kind in token.Error], (where, text)
        assert "".join(value for _, value in tokens) == text.strip("\n") + "\n", where
        # An example shows more than plain text: an operation, a type or an
        # attribute.
        colored = {token.Keyword, token.Name.Builtin, token.Name.Decorator}
        assert {kind for kind, _ in tokens} & colored, (where, text)


def test_the_lexer_reads_any_text_without_an_error():
    text = "".join(chr(code) for code in range(1, 0x250)) + '\n"open %% ^^ @@ !! ## -> .. ....\n'
    tokens = list(mlir_lexer.MlirLexer().get_tokens(text))
    assert not [value for kind, value in tokens if kind in token.Error]


def test_the_parts_of_an_example_are_told_apart():
    tokens = list(mlir_lexer.MlirLexer().get_tokens(_EXAMPLE))
    kinds = {value: kind for kind, value in tokens}
    expected = {
        "// One pulse on a drive line.": token.Comment.Single,
        "%d1": token.Name.Variable,
        "pulse.drive": token.Keyword,
        "classical.wait": token.Keyword,
        "func.func": token.Keyword,
        "module": token.Keyword,
        "attributes": token.Keyword,
        "return": token.Keyword,
        "false": token.Keyword.Constant,
        "!pulse.drive_line": token.Name.Builtin,
        "i64": token.Name.Builtin,
        "i1": token.Name.Builtin,
        "#pulse.control_system": token.Name.Decorator,
        "@main": token.Name.Function,
        "@decode": token.Name.Function,
        "^bb0": token.Name.Label,
        # The key of an attribute is no operation, though it has a dot.
        "pulse.control_system": token.Name.Attribute,
        "start": token.Name.Attribute,
        "40": token.Number.Integer,
        "500000": token.Number.Integer,
        "-2.5e-01": token.Number.Float,
        '"bounded"': token.String.Double,
        "->": token.Operator,
        "=": token.Operator,
        "{": token.Punctuation,
        "<": token.Punctuation,
    }
    assert {value: kinds.get(value) for value in expected} == expected


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_every_kind_of_token_has_a_color_of_the_palette(theme):
    css = code_style.stylesheet()
    colors = set(code_style.PALETTES[theme].values())
    kinds = {rule[1] for rule in mlir_lexer.MlirLexer.tokens["root"]}
    assert len(kinds) > 10
    for kind in kinds:
        cls = token.STANDARD_TYPES[kind]
        if not cls:
            # Plain text has no class: it has the color of the block.
            assert kind is token.Text
            continue
        rule = re.search(rf'html\[data-theme="{theme}"\] \.highlight \.{cls} \{{([^}}]*)\}}', css)
        assert rule, kind
        color = re.search(r"(?<![-\w])color: (#[0-9a-fA-F]{6})", rule.group(1))
        assert color and color.group(1).lower() in colors, (kind, rule.group(1))
