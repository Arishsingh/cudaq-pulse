# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API names and example paths in the docs' running text become links.

``docs/_ext/api_links.py`` links an inline literal that names exactly one
documented object to its entry in the API reference, and a literal that
names an example to the file. None of this needs a build: the matching runs
on a small inventory in the shape of Sphinx's, and the linking on pages that
docutils parses here. The build itself checks the names of ``MUST_RESOLVE``
against the real inventory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import pytest

nodes = pytest.importorskip("docutils.nodes")
publish_doctree = pytest.importorskip("docutils.core").publish_doctree
addnodes = pytest.importorskip("sphinx.addnodes")

_EXT = Path(__file__).resolve().parents[1] / "docs" / "_ext"
pytestmark = pytest.mark.skipif(not _EXT.exists(), reason="needs the source tree")
sys.path.insert(0, str(_EXT))
api_links = pytest.importorskip("api_links")
sys.path.remove(str(_EXT))

_ADDRESS = "https://example.org/blob/main/examples/%s"


class _Entry(NamedTuple):
    docname: str
    node_id: str
    objtype: str
    aliased: bool = False


def _inventory(*objects: tuple[str, str, str]) -> dict[str, _Entry]:
    return {name: _Entry(docname, name, objtype) for name, objtype, docname in objects}


_OBJECTS = _inventory(
    ("cudaq_pulse.simulate", "function", "api/simulation"),
    ("cudaq_pulse.compile", "function", "api/compile"),
    ("cudaq_pulse.kernel", "function", "api/kernel"),
    ("cudaq_pulse.Kernel", "class", "api/kernel"),
    ("cudaq_pulse.Kernel.lower", "method", "api/kernel"),
    ("cudaq_pulse.Kernel.cache_info", "method", "api/kernel"),
    ("cudaq_pulse.CompiledKernel", "class", "api/compile"),
    ("cudaq_pulse.CompiledKernel.lower", "method", "api/compile"),
    ("cudaq_pulse.CompiledKernel.cache_info", "method", "api/compile"),
    ("cudaq_pulse.ScheduledKernel", "class", "api/compile"),
    ("cudaq_pulse.ScheduledKernel.bind", "method", "api/compile"),
    ("cudaq_pulse.ScheduledKernel.duration", "property", "api/compile"),
    ("cudaq_pulse.ScheduledKernel.export_to_controller", "method", "api/compile"),
    ("cudaq_pulse.ops.gaussian", "function", "api/ops"),
    ("cudaq_pulse.ops.wait", "function", "api/ops"),
    # The operations guide describes the kernel calls a second time.
    ("wait", "function", "user_guide/operations"),
    ("cudaq_pulse.Controller", "class", "api/quantum_systems"),
    ("cudaq_pulse.Controller.clock_ghz", "attribute", "api/quantum_systems"),
    ("cudaq_pulse.controller.FORMAT", "data", "api/controller"),
    ("cudaq_pulse.controller.ControllerLimitError", "exception", "api/errors"),
    ("cudaq_pulse.backends.Backend", "class", "api/controller"),
    ("cudaq_pulse.backends.Backend.run", "method", "api/controller"),
    ("cudaq_pulse.backends.get", "function", "api/controller"),
    ("cudaq_pulse.classical.Bits", "class", "api/classical"),
    ("cudaq_pulse.classical.ClassicalDecorator.Bits", "attribute", "api/classical"),
    ("cudaq_pulse.classical.emulation.emulate", "function", "api/classical"),
)
# A module's anchor is not its name, and Sphinx lists a class once more under
# the private path it is imported from, with the anchor of the public entry.
_OBJECTS["cudaq_pulse.classical.wire"] = _Entry(
    "api/classical", "module-cudaq_pulse.classical.wire", "module"
)
_OBJECTS["cudaq_pulse.compiler.ScheduledKernel"] = _Entry(
    "api/compile", "cudaq_pulse.ScheduledKernel", "class", aliased=True
)
_NAMES = api_links.Names(_OBJECTS)

_LINKED = [
    ("cudaq_pulse.controller.FORMAT", "cudaq_pulse.controller.FORMAT"),
    ("cudaq_pulse.classical.wire", "cudaq_pulse.classical.wire"),
    ("pulse.simulate", "cudaq_pulse.simulate"),
    ("backends.get", "cudaq_pulse.backends.get"),
    ("pulse.Bits", "cudaq_pulse.classical.Bits"),
    ("pulse.gaussian(20, 0.3, 5)", "cudaq_pulse.ops.gaussian"),
    ("pulse.wait(line, 20)", "cudaq_pulse.ops.wait"),
    ("ScheduledKernel.bind", "cudaq_pulse.ScheduledKernel.bind"),
    ("ScheduledKernel.duration", "cudaq_pulse.ScheduledKernel.duration"),
    ("Kernel.lower", "cudaq_pulse.Kernel.lower"),
    ("compiler.ScheduledKernel", "cudaq_pulse.ScheduledKernel"),
    ("ScheduledKernel", "cudaq_pulse.ScheduledKernel"),
    ("Controller", "cudaq_pulse.Controller"),
    ("ControllerLimitError", "cudaq_pulse.controller.ControllerLimitError"),
    ("Controller(max_instructions=n)", "cudaq_pulse.Controller"),
    ("@pulse.kernel", "cudaq_pulse.kernel"),
    ('@pulse.kernel(schedule="asap")', "cudaq_pulse.kernel"),
    ("pulse.compile(kernel)", "cudaq_pulse.compile"),
    ("simulate(lower(k), shots_count=10)", "cudaq_pulse.simulate"),
    ("gaussian(duration,\namplitude, sigma)", "cudaq_pulse.ops.gaussian"),
    ("wait()", "cudaq_pulse.ops.wait"),
    ("export_to_controller()", "cudaq_pulse.ScheduledKernel.export_to_controller"),
    ("export_to_controller(version=2)", "cudaq_pulse.ScheduledKernel.export_to_controller"),
]

_PLAIN = [
    # A bare lowercase word, though one object alone has the name.
    "simulate",
    "wait",
    "bind",
    "export_to_controller",
    # No bare name leads to an attribute, a property or a constant.
    "clock_ghz",
    "duration",
    "FORMAT",
    # A bare call of one word that is not one of the package's own functions,
    # though one method or function alone has the name.
    "run(artifact)",
    "emulate()",
    "bind(values)",
    # What a kernel calls, under the alias and not called: so written, the
    # name of an MLIR operation.
    "pulse.gaussian",
    "pulse.wait",
    # Two methods have the name.
    "cache_info()",
    # No full name ends so.
    "ScheduledKernel.lower",
    "Bits.clock_ghz",
    # More than a name, a decorator or one call.
    "pulse.compile(kernel).mlir()",
    "pulse.simulate(k).counts",
    "20 * pulse.ns",
    "pulse.compile(",
    ".bind()",
    "()",
    "",
    # Not this package's.
    "ValueError",
    "pulse.undocumented",
    "numpy.ndarray",
    "None",
]


@pytest.mark.parametrize(("text", "name"), _LINKED)
def test_a_name_leads_to_its_one_object(text, name):
    target = api_links.resolve(text, _NAMES)
    assert target is not None, text
    assert target.name == name
    assert (target.docname, target.objtype) == (_OBJECTS[name].docname, _OBJECTS[name].objtype)
    assert target.anchor == _OBJECTS[name].node_id


@pytest.mark.parametrize("text", _PLAIN)
def test_a_name_of_no_one_object_stays_plain(text):
    assert api_links.resolve(text, _NAMES) is None, text


def test_a_dotted_end_of_two_names_stays_plain():
    names = api_links.Names(
        _inventory(
            ("package.first.Report.total", "attribute", "api/first"),
            ("package.second.Report.total", "attribute", "api/second"),
        )
    )
    assert api_links.resolve("Report.total", names) is None
    assert api_links.resolve("first.Report.total", names).docname == "api/first"
    assert api_links.resolve("second.Report.total", names).docname == "api/second"


def test_the_guarded_names_are_checked_against_the_inventory():
    assert api_links.missing_names(_NAMES) == []
    renamed = {name: entry for name, entry in _OBJECTS.items() if "ScheduledKernel" not in name}
    assert api_links.missing_names(api_links.Names(renamed)) == ["ScheduledKernel"]
    assert api_links.missing_names(api_links.Names({})) == list(api_links.MUST_RESOLVE)


def _page(text: str):
    # Without the title transform a page keeps each of its sections.
    return publish_doctree(text, settings_overrides={"doctitle_xform": False, "report_level": 5})


def _link_names(page, docname: str = "user_guide/kernels") -> int:
    return api_links.link_api_names(page, docname, _NAMES, lambda to: f"../{to}.html")


def _links(page) -> list[tuple[str, str]]:
    """(text, address) of each link of the page that holds a literal."""
    return [
        (link.astext(), link["refuri"])
        for link in page.findall(nodes.reference)
        if isinstance(link[0], nodes.literal)
    ]


_TWO_SECTIONS = """\
Running ``pulse.simulate``
==========================

``pulse.simulate`` runs a kernel, and ``pulse.simulate(k)`` returns its
result. A ``ScheduledKernel`` runs too, and so does a ``duration``.

Compiling
=========

``pulse.compile`` gives a ``ScheduledKernel`` to ``pulse.simulate``.
"""


def test_the_first_mention_in_each_section_is_linked():
    page = _page(_TWO_SECTIONS)
    assert _link_names(page) == 5
    assert _links(page) == [
        ("pulse.simulate", "../api/simulation.html#cudaq_pulse.simulate"),
        ("ScheduledKernel", "../api/compile.html#cudaq_pulse.ScheduledKernel"),
        ("pulse.compile", "../api/compile.html#cudaq_pulse.compile"),
        ("ScheduledKernel", "../api/compile.html#cudaq_pulse.ScheduledKernel"),
        ("pulse.simulate", "../api/simulation.html#cudaq_pulse.simulate"),
    ]
    # The heading keeps its literal, and the later mentions stay plain.
    assert not list(page.next_node(nodes.title).findall(nodes.reference))
    plain = [code.astext() for code in page.findall(nodes.literal) if not code["classes"]]
    assert plain == ["pulse.simulate", "pulse.simulate(k)", "duration"]


def test_a_link_is_marked_as_sphinx_marks_its_own():
    page = _page("``pulse.simulate`` and ``Controller`` and ``cudaq_pulse.controller.FORMAT``.")
    assert _link_names(page) == 3
    simulate, controller, constant = page.findall(nodes.reference)
    assert simulate["internal"] is True
    assert simulate["reftitle"] == "cudaq_pulse.simulate"
    assert simulate["classes"] == []
    assert simulate[0]["classes"] == ["xref", "py", "py-func"]
    assert controller[0]["classes"] == ["xref", "py", "py-class"]
    assert constant[0]["classes"] == ["xref", "py", "py-data"]


def test_a_page_without_sections_is_one_section():
    page = _page("``pulse.simulate`` first.\n\n``pulse.simulate`` again.")
    assert _link_names(page) == 1
    assert isinstance(page[0][0], nodes.reference)
    assert isinstance(page[1][0], nodes.literal)


def test_linking_a_page_twice_changes_nothing():
    page = _page(_TWO_SECTIONS)
    _link_names(page)
    once = page.pformat()
    assert _link_names(page) == 0
    assert page.pformat() == once


def test_a_page_does_not_link_to_itself():
    page = _page("``pulse.simulate`` and ``pulse.compile``.")
    assert _link_names(page, "api/simulation") == 1
    assert _links(page) == [("pulse.compile", "../api/compile.html#cudaq_pulse.compile")]


def test_the_architecture_pages_are_left():
    page = _page("The ``pulse.wait`` operation and ``pulse.simulate``.")
    assert _link_names(page, "architecture/pipeline") == 0
    assert not list(page.findall(nodes.reference))


def test_what_is_not_running_text_is_left():
    page = _page(".. rubric:: About ``pulse.simulate``\n\nText.")
    signature = addnodes.desc_signature("", "", nodes.literal("", "pulse.simulate"))
    role = nodes.paragraph("", "", nodes.literal("", "pulse.simulate", classes=["file"]))
    # Sphinx's HTML writer accepts a link outside a paragraph only around an image.
    outside_text = nodes.container("", nodes.literal("", "pulse.simulate"))
    page += [signature, role, outside_text]
    assert _link_names(page) == 0
    assert not list(page.findall(nodes.reference))


def test_a_link_already_there_is_the_sections_mention():
    page = _page("``pulse.simulate`` again, and ``pulse.compile``.")
    paragraph = page[0]
    own = nodes.reference("", "", internal=True, refuri="x", reftitle="cudaq_pulse.simulate")
    own += nodes.literal("", "simulate", classes=["xref", "py", "py-func"])
    paragraph.insert(0, own)
    assert _link_names(page) == 1
    assert _links(page) == [
        ("simulate", "x"),
        ("pulse.compile", "../api/compile.html#cudaq_pulse.compile"),
    ]


def _examples(folder: Path) -> Path:
    examples = folder / "examples"
    (examples / "custom_passes").mkdir(parents=True)
    (examples / "custom_passes" / "python_pass.py").write_text("")
    # A file beside the examples, which no path may reach.
    (folder / "secret.py").write_text("")
    return examples


_EXAMPLE_PAGE = """\
``examples/custom_passes/python_pass.py`` is one of ``examples/custom_passes/``
in ``examples/``. ``examples/custom_passes/python_pass.py`` again, then
``examples/missing.py``, ``examples/../secret.py``, ``custom_passes/python_pass.py``
and ``python examples/custom_passes/python_pass.py``.
"""


def test_an_example_path_links_to_the_file_or_directory(tmp_path):
    page = _page(_EXAMPLE_PAGE)
    examples = _examples(tmp_path)
    assert api_links.link_example_paths(page, examples, _ADDRESS) == 3
    assert _links(page) == [
        (
            "examples/custom_passes/python_pass.py",
            "https://example.org/blob/main/examples/custom_passes/python_pass.py",
        ),
        ("examples/custom_passes/", "https://example.org/tree/main/examples/custom_passes"),
        ("examples/", "https://example.org/tree/main/examples/"),
    ]
    for link in page.findall(nodes.reference):
        assert link["internal"] is False
        assert link["classes"] == ["extlink-example"]
        assert link[0]["classes"] == []
    # A second pass finds each already linked.
    assert api_links.link_example_paths(page, examples, _ADDRESS) == 0


def test_without_the_examples_a_path_stays_plain(tmp_path):
    page = _page(_EXAMPLE_PAGE)
    assert api_links.link_example_paths(page, tmp_path / "examples", _ADDRESS) == 0
    assert not list(page.findall(nodes.reference))


def test_api_names_and_example_paths_link_on_one_page(tmp_path):
    page = _page("``pulse.simulate`` runs ``examples/custom_passes/python_pass.py``.")
    assert api_links.link_example_paths(page, _examples(tmp_path), _ADDRESS) == 1
    assert _link_names(page) == 1
    assert [address for _, address in _links(page)] == [
        "../api/simulation.html#cudaq_pulse.simulate",
        "https://example.org/blob/main/examples/custom_passes/python_pass.py",
    ]
