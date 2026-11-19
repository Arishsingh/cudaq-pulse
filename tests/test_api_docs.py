# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every public name is in the API reference, and its entries read as their
docstrings say (source tree only).

``docs/_ext/api_docs.py`` puts right what autodoc writes into the reference.
Most of this needs no build: the docstrings of the documented constants,
attributes and properties go through the extension and through napoleon here,
in the order a build calls them, the signatures of the documented dataclasses
are printed and read by Sphinx's own functions, the names in types are looked
up by Sphinx's own search, the headings are made on entries built here in the
shape Sphinx gives them, and the links are named in the markup Sphinx writes
for them.

The last tests read the site that the ``docs`` preset builds, for what a
build alone shows: that the markup Sphinx writes is still the markup the
extension looks for. Without a site they are skipped, and with
``CUDAQ_PULSE_REQUIRE_DOCS_SITE=1`` they fail instead. A site built before a
change to the extension fails them too: build the docs again, and without the
site's ``_modules`` folder, as Sphinx writes a page of source again only when
its module has changed.
"""

import dataclasses
import importlib
import importlib.util
import inspect
import os
import re
import sys
import types
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_API = _ROOT / "docs" / "api"
_EXT = _ROOT / "docs" / "_ext"
# Where the build of the ``docs`` preset writes the site.
_SITE = _ROOT / "build" / "docs" / "docs" / "html"
pytestmark = pytest.mark.skipif(not _API.is_dir(), reason="needs the source tree's docs/")

# The documented submodules whose __all__ is the public API (CONTRIBUTING.md,
# "Versioning and stability").
PUBLIC_MODULES = [
    "cudaq_pulse",
    "cudaq_pulse.ops",
    "cudaq_pulse.compiler",
    "cudaq_pulse.quantum_systems",
    "cudaq_pulse.quantum_systems.transmon_model",
    "cudaq_pulse.runtime",
    "cudaq_pulse.runtime.simulate",
    "cudaq_pulse.viz",
    "cudaq_pulse.controller",
    "cudaq_pulse.backends",
    "cudaq_pulse.classical",
    "cudaq_pulse.classical.wire",
    "cudaq_pulse.classical.emulation",
    "cudaq_pulse.classical.calibration",
    "cudaq_pulse.classical.loop",
    "cudaq_pulse.quantum_systems.drift",
]

_DIRECTIVE = re.compile(r"^\.\. auto(\w+):: (\S+)", re.M)
_MEMBERS = re.compile(r"^\.\. automodule:: (\S+)\n((?:   :.*\n)+)", re.M)
# A constant's directive, its options, and the text the page writes under it.
_CONSTANT = re.compile(r"^\.\. autodata:: (\S+)\n((?:   :.*\n)*)((?:\n|   .*\n)*)", re.M)
_PARAMETER = re.compile(r":param ([^:]*):")
# A link to an entry's source as Sphinx writes it into a signature.
_SOURCE_LINK = (
    '<a class="reference internal" href="../_modules/package/kernel.html#{}">'
    '<span class="viewcode-link"><span class="pre">[source]</span></span></a>'
)
# The link back to an entry as Sphinx writes it above a definition, on the
# page that lists a module's source.
_DOCS_LINK = '<a class="viewcode-back" href="../../api/kernel.html#package.{}">[docs]</a>'
# An entry of a built page: its name, then its markup up to the next entry.
_BUILT_ENTRY = re.compile(
    r'<dt class="sig sig-object py" id="([^"]+)">(.*?)(?=<dt class="sig sig-object py"|\Z)', re.S
)
# Text that Python and Sphinx make and the extension keeps out of the
# reference: the docstring of the descriptor a component reads its fields
# through, those of ``str`` and ``tuple``, and the mark of a factory.
_MACHINE_MADE = (
    "A field of a component",
    "str(object=",
    "Built-in immutable sequence",
    "&lt;factory&gt;",
)


def _extension():
    """``docs/_ext/api_docs.py``, or None where Sphinx is not installed, the
    docs' tools being no dependency of the library, or where the tests run
    without the source tree's ``docs/``."""
    if importlib.util.find_spec("sphinx") is None or not (_EXT / "api_docs.py").is_file():
        return None
    sys.path.insert(0, str(_EXT))
    try:
        return importlib.import_module("api_docs")
    finally:
        sys.path.remove(str(_EXT))


api_docs = _extension()
needs_sphinx = pytest.mark.skipif(api_docs is None, reason="needs Sphinx")
if api_docs is not None:
    from docutils import nodes
    from sphinx import addnodes
    from sphinx.domains.python import ObjectEntry, PythonDomain
    from sphinx.errors import PycodeError
    from sphinx.ext import napoleon
    from sphinx.pycode import ModuleAnalyzer
    from sphinx.transforms.post_transforms import ReferencesResolver
    from sphinx.util.docstrings import prepare_docstring
    from sphinx.util.inspect import (
        getdoc,
        isattributedescriptor,
        signature_from_str,
        stringify_signature,
    )


def _resolve(path):
    """The object *path* names, by attribute access from the package, as
    Sphinx resolves it."""
    parts = path.split(".")
    obj = importlib.import_module(parts[0])
    for i, part in enumerate(parts[1:], start=1):
        if not hasattr(obj, part):
            importlib.import_module(".".join(parts[: i + 1]))
        obj = getattr(obj, part)
    return obj


def _named():
    """(kind, dotted name, object) of each object the pages document, and the
    modules they document. The kind is the directive's; a module's member is
    a class, a function or data by what it is."""
    objects, modules = [], set()
    for page in sorted(_API.glob("*.rst")):
        text = page.read_text()
        for kind, path in _DIRECTIVE.findall(text):
            if kind == "module":
                modules.add(path)
            else:
                objects.append((kind, path, _resolve(path)))
        for module, options in _MEMBERS.findall(text):
            members = re.search(r":members:(.*)", options)
            if members is None:
                continue
            names = [m.strip() for m in members.group(1).split(",") if m.strip()]
            # A bare :members: documents the module's __all__.
            names = names or list(importlib.import_module(module).__all__)
            for name in names:
                obj = _resolve(f"{module}.{name}")
                kind = "class" if inspect.isclass(obj) else "function" if callable(obj) else "data"
                objects.append((kind, f"{module}.{name}", obj))
    return objects, modules


def _documented():
    objects, modules = _named()
    return [obj for _, _, obj in objects], modules


def test_every_public_name_is_documented():
    objects, modules = _documented()
    ids = {id(o) for o in objects}
    pages = "".join(p.read_text() for p in _API.glob("*.rst"))
    missing = []
    for module in PUBLIC_MODULES:
        mod = importlib.import_module(module)
        for name in mod.__all__:
            value = getattr(mod, name)
            if isinstance(value, types.ModuleType):
                found = value.__name__ in modules or f"{value.__name__}." in pages
            else:
                found = id(value) in ids
            if not found:
                missing.append(f"{module}.{name}")
    assert missing == []


def test_contributing_names_the_public_modules():
    text = (_ROOT / "CONTRIBUTING.md").read_text()
    assert [m for m in PUBLIC_MODULES if f"`{m}`" not in text] == []


def _text(lines):
    return " ".join(" ".join(lines).split())


def _first(lines):
    return next((line for line in lines if line.strip()), "")


def _comments(module):
    """The documentation comments and attribute docstrings of *module*, by
    (class, name), as autodoc reads them from its source."""
    try:
        analyzer = ModuleAnalyzer.for_module(module)
        analyzer.analyze()
    except PycodeError:
        return {}
    return analyzer.attr_docs


def _prose():
    """(kind, dotted name, value, docstring lines) of each constant the pages
    document and of each attribute and property of the classes they document
    that has a docstring, the one autodoc finds for it."""
    found = []
    for kind, path, obj in _named()[0]:
        if kind == "data":
            module, _, name = path.rpartition(".")
            lines = _comments(module).get(("", name)) or prepare_docstring(getdoc(obj) or "")
            found.append(("data", path, obj, list(lines)))
        elif inspect.isclass(obj):
            comments = {
                name: lines
                for (scope, name), lines in _comments(obj.__module__).items()
                if scope == obj.__qualname__
            }
            for name in sorted({*comments, *vars(obj)}):
                value = vars(obj).get(name)
                member = "property" if isinstance(value, property) else "attribute"
                lines = comments.get(name)
                if not lines and (member == "property" or isattributedescriptor(value)):
                    lines = prepare_docstring(getdoc(value) or "")
                if lines and not name.startswith("_"):
                    found.append((member, f"{path}.{name}", value, list(lines)))
    return found


def _connected(event):
    """(priority, listener) of each listener the extension connects to
    *event*."""
    found = []

    def connect(name, listener, priority=500):
        if name == event:
            found.append((priority, listener))

    app = SimpleNamespace(
        connect=connect,
        setup_extension=lambda name: None,
        add_post_transform=lambda transform: None,
    )
    api_docs.setup(app)
    return found


def _listeners():
    """The docstring listeners of a build, in the order it calls them: those
    the extension connects, and napoleon's at the default priority."""
    found = [(500, napoleon._process_docstring), *_connected("autodoc-process-docstring")]
    return [listener for _, listener in sorted(found, key=lambda entry: entry[0])]


def _printed(kind, name, value, lines):
    """What the reference prints of the docstring *lines* of an entry: the
    lines after every listener, with napoleon set as ``docs/conf.py`` sets
    it."""
    config = napoleon.Config(napoleon_google_docstring=True, napoleon_numpy_docstring=True)
    lines = list(lines)
    for listener in _listeners():
        listener(SimpleNamespace(config=config), kind, name, value, None, lines)
    return lines


def _is_its_types(value, lines):
    """Whether *lines* is the docstring of the type of *value*: what a value
    without a docstring of its own answers with."""
    return _text(lines) == _text((type(value).__doc__ or "").splitlines())


@needs_sphinx
def test_the_extension_reads_a_docstring_before_and_after_napoleon():
    listeners = _listeners()
    assert len(listeners) == 3
    assert listeners[1] is napoleon._process_docstring


@needs_sphinx
def test_a_docstring_whose_first_line_has_a_colon_reads_whole():
    sentence = "How far the due time is inside the deadline (negative: missed)."
    # A method's docstring was never read as "type: description".
    for kind in (*api_docs.PROSE, "method"):
        printed = _printed(kind, "package.Report.slack", None, [sentence, ""])
        assert [line for line in printed if line] == [sentence], kind
    sections = [
        "How long it runs: the latest end.",
        "",
        "Returns",
        "-------",
        "int",
        "    A number of cycles.",
        "",
    ]
    printed = _printed("property", "package.Report.duration", None, sections)
    assert printed[0] == sections[0]
    assert ":returns: A number of cycles." in printed
    assert not [line for line in printed if line.startswith(":type:")]


@needs_sphinx
def test_each_documented_docstring_reads_whole():
    """No attribute, property or constant of the reference has its first
    clause printed as its type."""
    with_colon = []
    for kind, name, value, lines in _prose():
        printed = _printed(kind, name, value, lines)
        assert not [line for line in printed if line.startswith(":type:")], name
        assert api_docs.SHIELD not in printed, name
        if ":" in _first(lines) and not _is_its_types(value, lines):
            with_colon.append(name)
            assert _first(printed) == _first(lines), name
    assert with_colon


@needs_sphinx
def test_no_entry_shows_the_docstring_of_its_values_type():
    borrowed = []
    for kind, name, value, lines in _prose():
        if _is_its_types(value, lines):
            borrowed.append(name)
            assert not _is_its_types(value, _printed(kind, name, value, lines)), name
    assert "cudaq_pulse.controller.FORMAT" in borrowed
    assert "cudaq_pulse.compiler.DEFAULT_PASSES" in borrowed
    for name in ("cudaq_pulse.controller.FORMAT", "cudaq_pulse.ns", "cudaq_pulse.Length"):
        value = _resolve(name)
        assert _printed("data", name, value, prepare_docstring(getdoc(value) or "")) == [], name
    # A constant that the source describes keeps its description.
    described = "The layers of one cycle: each is three gates."
    assert _text(_printed("data", "package.LAYERS", (1, 2), [described, ""])) == described


@needs_sphinx
def test_a_component_field_reads_as_its_class_declares_it():
    """A field read through a descriptor shows the docstring its class gives
    it for type checkers, not the descriptor's."""
    from cudaq_pulse.quantum_systems import DispersiveReadout, DragPulse

    for component in (DragPulse, DispersiveReadout):
        for field in inspect.signature(component).parameters:
            value = vars(component)[field]
            name = f"cudaq_pulse.quantum_systems.{component.__name__}.{field}"
            stub = prepare_docstring(getdoc(value))
            printed = _printed("attribute", name, value, stub)
            assert _text(printed), name
            assert _text(printed) != _text(stub), name
            assert not [line for line in printed if line.startswith(":type:")], name


@needs_sphinx
def test_each_constant_has_a_description():
    """A constant shows its own description from the source, or the text its
    page writes under it."""
    written = {
        path: text
        for page in sorted(_API.glob("*.rst"))
        for path, _, text in _CONSTANT.findall(page.read_text())
    }
    constants = [entry for entry in _prose() if entry[0] == "data"]
    bare = [
        name
        for kind, name, value, lines in constants
        if not written.get(name, "").strip() and not _text(_printed(kind, name, value, lines))
    ]
    assert constants
    assert bare == []


@needs_sphinx
def test_each_documented_parameter_is_in_the_signature():
    """A name in a docstring's parameter list that the signature does not
    have would print as a parameter of its own."""
    callables = []
    for kind, path, obj in _named()[0]:
        if inspect.isclass(obj):
            callables.append((path, obj))
            callables += [
                (f"{path}.{name}", getattr(obj, name))
                for name, value in vars(obj).items()
                if not name.startswith("_")
                and (inspect.isfunction(value) or isinstance(value, classmethod | staticmethod))
            ]
        elif kind != "data" and callable(obj):
            callables.append((path, obj))
    unknown, checked = {}, 0
    for path, obj in callables:
        kind = "class" if inspect.isclass(obj) else "function"
        printed = _printed(kind, path, obj, prepare_docstring(getdoc(obj) or ""))
        named = [match.group(1) for line in printed if (match := _PARAMETER.match(line))]
        if not named:
            continue
        checked += 1
        parameters = inspect.signature(obj).parameters
        wrong = [name for name in named if name.replace("\\", "").lstrip("*") not in parameters]
        if wrong:
            unknown[path] = wrong
    assert checked
    assert unknown == {}


def _signed(name, obj):
    """Of the class *obj*, documented as *name*: the argument list as autodoc
    prints it under ``docs/conf.py``, without types, the one the reference
    shows, and the types Sphinx then lists under "Parameters". Sphinx notes
    the types, then the extension's listener runs, as in a build."""
    # Sphinx 9 keeps this listener for its older documenters alone; it notes
    # the types where the newer ones do.
    from sphinx.ext.autodoc.typehints import record_typehints

    config = SimpleNamespace(
        autodoc_typehints_format="short",
        autodoc_type_aliases={},
        python_display_short_literal_types=False,
    )
    document = SimpleNamespace(autodoc_annotations={})
    app = SimpleNamespace(config=config, env=SimpleNamespace(current_document=document))
    printed = stringify_signature(
        inspect.signature(obj), show_annotation=False, show_return_annotation=False
    )
    listeners = [(500, record_typehints), *_connected("autodoc-process-signature")]
    answers = [
        listener(app, "class", name, obj, None, printed, None)
        for _, listener in sorted(listeners, key=lambda entry: entry[0])
    ]
    # A build takes the first answer there is.
    shown = next((answer[0] for answer in answers if answer is not None), printed)
    return printed, shown, document.autodoc_annotations.get(name, {})


@needs_sphinx
def test_a_dataclass_signature_reads_as_python():
    @dataclasses.dataclass
    class Shots:
        levels: list[int]
        span_ns: tuple[float, float] = (0.0, 0.0)
        calls: list[str] = dataclasses.field(default_factory=list)
        _states: list[object] = dataclasses.field(default_factory=list, repr=False)

    printed, shown, listed = _signed("package.Shots", Shots)
    # Python's mark of a factory is no Python: Sphinx cannot read the list,
    # and cuts it at each comma.
    assert api_docs.FACTORY in printed
    with pytest.raises(SyntaxError):
        signature_from_str(printed)
    assert shown == "(levels, span_ns=(0.0, 0.0), calls=...)"
    # Sphinx reads it: three parameters, the default with a comma in it whole.
    read = signature_from_str(shown).parameters
    assert list(read) == ["levels", "span_ns", "calls"]
    assert str(read["span_ns"].default) == "(0.0, 0.0)"
    # The private field is not under "Parameters" either.
    assert list(listed) == ["levels", "span_ns", "calls", "return"]
    # A signature that shows its types keeps them.
    typed = "(levels: list[int], span_ns: tuple[float, float] = (0.0, 0.0)"
    private = ", _states: list[object] = <factory>)"
    assert api_docs.readable_signature(typed + private, {"_states"}) == typed + ")"


@needs_sphinx
def test_a_signature_with_nothing_to_repair_is_left_as_it_is():
    @dataclasses.dataclass
    class Plain:
        shots: int = 1
        span_ns: tuple[float, float] = (0.0, 0.0)

    class Written:
        # Its author wrote the constructor, the private parameter too.
        def __init__(self, shots: int, _variant: bool = False) -> None:
            self.shots = shots

    unset = object()

    @dataclasses.dataclass
    class Opaque:
        # Its first default prints as no Python, with or without the mark.
        marker: object = unset
        _states: list[object] = dataclasses.field(default_factory=list)

    for cls in (Plain, Written, Opaque):
        printed, shown, listed = _signed("package.Class", cls)
        assert shown == printed, cls
        assert list(listed) == [*inspect.signature(cls).parameters, "return"], cls


@needs_sphinx
def test_each_documented_dataclass_signature_reads_as_python():
    repaired = []
    for _, path, obj in _named()[0]:
        if not (inspect.isclass(obj) and dataclasses.is_dataclass(obj)):
            continue
        printed, shown, listed = _signed(path, obj)
        assert api_docs.FACTORY not in shown, path
        if shown == printed:
            continue
        repaired.append(path)
        names = [*signature_from_str(shown).parameters, *listed]
        assert not [name for name in names if name.startswith("_")], path
    # The two whose lists Sphinx cut inside a default.
    assert "cudaq_pulse.SimulationResult" in repaired
    assert "cudaq_pulse.Processor" in repaired


def _type_name(target, module):
    """A name in a type as Sphinx reads it from the signature of an entry of
    *module*: a reference to the class of that name."""
    scope = {"py:module": module, "py:class": None}
    return addnodes.pending_xref(
        "",
        nodes.Text(target),
        refdomain="py",
        reftype="class",
        reftarget=target,
        refspecific=False,
        **scope,
    )


@needs_sphinx
def test_a_name_in_a_type_is_no_link_to_an_attribute_that_ends_with_it():
    """A type variable ``T`` is not the attribute ``Gate.T``. Sphinx's own
    search and resolver run here, on a small set of entries."""
    domain = PythonDomain(SimpleNamespace(domaindata={}))
    entries = {
        "package.gates.Gate": "class",
        "package.gates.Gate.T": "attribute",
        "package.calls.Function": "class",
        "package.Length": "data",
    }
    for name, kind in entries.items():
        domain.objects[name] = ObjectEntry("api/page", name, kind, False)

    find = partial(domain.find_obj, None)
    # No class has the name, and the search by the end of names finds the gate.
    assert find("package.calls", None, "T", "class", 1) == []
    gate = find("package.calls", None, "T", "attr", 1)
    assert [name for name, _ in gate] == ["package.gates.Gate.T"]
    variable, cls = _type_name("T", "package.calls"), _type_name("Function", "package.calls")
    # A type alias is a constant: named in its module, or in full.
    alias, full = _type_name("Length", "package"), _type_name("package.Length", "package.calls")
    line = nodes.paragraph("", "", variable, cls, alias, full)
    api_docs.unlink_guessed_types(line, find)
    assert list(line.children) == [nodes.Text("T"), cls, alias, full]
    # What Sphinx's resolver then links: the class, and nothing to the gate.
    linked = []
    for xref in line.findall(addnodes.pending_xref):
        target = xref["reftarget"]
        made = domain.resolve_xref(None, "api/page", None, "class", target, xref, xref[0])
        linked += [] if made is None else [made["reftitle"]]
    assert linked[0] == "package.calls.Function"
    assert "package.gates.Gate.T" not in linked
    # The names are unlinked before Sphinx resolves the references.
    transforms = []
    app = SimpleNamespace(
        connect=lambda *args, **options: None,
        setup_extension=lambda name: None,
        add_post_transform=transforms.append,
    )
    api_docs.setup(app)
    assert transforms
    for transform in transforms:
        assert transform.default_priority < ReferencesResolver.default_priority


def _entry(name, *members, signatures=1):
    """An entry of the reference as Sphinx builds it: its signatures, each a
    prefix and the name, then its content, the entries of its *members*."""
    entry = addnodes.desc(domain="py", objtype="function")
    for _ in range(signatures):
        signature = addnodes.desc_signature("", "")
        signature += addnodes.desc_addname("", "package.")
        signature += addnodes.desc_name("", name)
        entry += signature
    entry += addnodes.desc_content("", *members)
    return entry


def _section(title, *children):
    return nodes.section("", nodes.title("", title), *children)


def _heading(entry, signature=0):
    """The level of the heading the name of *entry* is, and the name as it
    reads; level 0 if it is no heading."""
    name = next(entry[signature].findall(addnodes.desc_name))
    raw = [child for child in name.children if isinstance(child, nodes.raw)]
    text = "".join(child.astext() for child in name.children if child not in raw)
    if not raw:
        return 0, text
    # The role is written around the name, for the HTML writer alone.
    start, end = raw
    assert (name[0], name[-1]) == (start, end)
    assert (start["format"], end["format"], end.astext()) == ("html", "html", "</span>")
    return int(re.fullmatch(r'<span role="heading" aria-level="(\d)">', start.astext())[1]), text


@needs_sphinx
def test_each_entry_of_a_page_is_a_heading_below_its_section():
    method = _entry("lower")
    function, cls, grouped = _entry("compile"), _entry("Kernel", method), _entry("check")
    page = _section("Compilation API", function, cls, _section("Limits", grouped))
    api_docs.head_entries(page)
    assert _heading(function) == (2, "compile")
    assert _heading(cls) == (2, "Kernel")
    assert _heading(grouped) == (3, "check")
    # A member is part of its class's entry.
    assert _heading(method) == (0, "lower")


@needs_sphinx
def test_an_entry_is_one_heading_of_level_six_at_most():
    overloaded = _entry("kernel", signatures=2)
    deep = overloaded
    for depth in range(7):
        deep = _section(f"Section {depth}", deep)
    api_docs.head_entries(deep)
    assert _heading(overloaded) == (6, "kernel")
    assert _heading(overloaded, signature=1) == (0, "kernel")


@needs_sphinx
def test_each_source_link_is_named_after_its_entry():
    elsewhere = '<a class="reference internal" href="guide.html#use">the guide</a>'
    page = (
        f"<dt>Kernel{_SOURCE_LINK.format('Kernel')}</dt><dd><p>See {elsewhere}.</p>"
        f"<dt>lower{_SOURCE_LINK.format('Kernel.lower')}</dt></dd>"
    )
    # A page of source: each documented definition under a link to its entry.
    source = (
        f'<div class="viewcode-block" id="Kernel">\n{_DOCS_LINK.format("Kernel")}\n'
        '<span class="k">class</span> <span class="nc">Kernel</span><span class="p">:</span>\n'
        f'<div class="viewcode-block" id="Kernel.lower">\n{_DOCS_LINK.format("Kernel.lower")}\n'
        '    <span class="k">def</span> <span class="nf">lower</span></div>\n</div>\n'
    )
    expected = {
        page: ["[source] of Kernel", "[source] of Kernel.lower"],
        source: ["[docs] of package.Kernel", "[docs] of package.Kernel.lower"],
    }
    for text, names in expected.items():
        named = api_docs.name_source_links(text)
        # The words on the page come first, so the name still starts with them.
        assert re.findall(r'<a [^>]*aria-label="([^"]*)">', named) == names
        # Nothing else changes, and a page that is named stays as it is.
        assert re.sub(r' aria-label="[^"]*"', "", named) == text
        assert api_docs.name_source_links(named) == named


@needs_sphinx
def test_a_page_goes_through_the_extension_as_html():
    """Sphinx hands the extension each page as HTML, a page of source too,
    which has no doctree."""
    [(_, listener)] = _connected("html-page-context")
    # What Sphinx writes for a class whose constructor takes nothing.
    empty = '<dl class="field-list simple">\n</dl>\n'
    fields = (
        '<dl class="field-list simple">\n'
        '<dt class="field-odd">Parameters<span class="colon">:</span></dt>\n'
        '<dd class="field-odd"><p><strong>shots</strong> (<em>int</em>)</p>\n</dd>\n</dl>\n'
    )
    entry = f"<dt>Log{_SOURCE_LINK.format('Log')}</dt><dd><p>Every estimate.</p>\n{empty}"
    context = {"title": "Classical API", "body": f"{entry}{fields}</dd>"}
    listener(None, "api/classical", "page.html", context, None)
    assert context["body"] == api_docs.name_source_links(f"{entry}{fields}</dd>".replace(empty, ""))
    assert 'aria-label="[source] of Log"' in context["body"]
    assert fields in context["body"]
    listing = {"title": "package.kernel", "body": f"<pre>{_DOCS_LINK.format('Kernel')}\n</pre>"}
    listener(None, "_modules/package/kernel", "page.html", listing, None)
    assert 'aria-label="[docs] of package.Kernel"' in listing["body"]
    # A page without text of its own, the search page, stays as it is.
    search = {"title": "Search"}
    listener(None, "search", "search.html", search, None)
    assert search == {"title": "Search"}


def _built(folder):
    """The pages of the built site under *folder*, by their paths in the
    site, each as its HTML."""
    pages = sorted((_SITE / folder).rglob("*.html")) if (_SITE / folder).is_dir() else []
    if not pages:
        if os.environ.get("CUDAQ_PULSE_REQUIRE_DOCS_SITE"):
            pytest.fail(
                "CUDAQ_PULSE_REQUIRE_DOCS_SITE is set, but build/docs/docs/html holds no site"
            )
        pytest.skip("needs the site that the docs preset builds")
    return {page.relative_to(_SITE).as_posix(): page.read_text(encoding="utf-8") for page in pages}


def _unnamed(text, links, word):
    """Of the links that the pattern *links* finds in the page *text*, each
    by its attributes: those not named *word* and then their entry, and how
    many links there are."""
    found = re.findall(links, text)
    unnamed = []
    for attributes in found:
        entry = re.search(r'href="[^"]*#([^"]+)"', attributes)
        if entry is None or f'aria-label="{word} of {entry[1]}"' not in attributes:
            unnamed.append(attributes)
    return unnamed, len(found)


def test_the_built_reference_names_each_source_link():
    links = 0
    for page, text in _built("api").items():
        unnamed, found = _unnamed(text, r'<a\b([^>]*)>\s*<span class="viewcode-link">', "[source]")
        assert unnamed == [], page
        # A link that Sphinx came to write in another shape would not be found.
        assert found == text.count('class="viewcode-link"'), page
        links += found
    assert links


def test_the_built_pages_of_source_name_each_link_back():
    links = 0
    for page, text in _built("_modules").items():
        unnamed, found = _unnamed(text, r'<a\b([^>]*\bclass="viewcode-back"[^>]*)>', "[docs]")
        assert unnamed == [], page
        assert found == text.count("viewcode-back"), page
        links += found
    assert links


def test_the_built_reference_has_no_text_that_a_machine_made():
    for page, text in _built("api").items():
        # The first clause of a docstring, read as the type of its entry.
        assert not re.search(r'<dt class="field-(?:odd|even)">Type<span', text), page
        assert [made for made in _MACHINE_MADE if made in text] == [], page
        assert not re.search(r'<dl class="field-list[^"]*">\s*</dl>', text), page


def test_the_built_reference_shows_no_private_field_of_a_dataclass():
    entries = 0
    for page, text in _built("api").items():
        for name, markup in _BUILT_ENTRY.findall(text):
            entries += 1
            signature, _, described = markup.partition("</dt>")
            private = re.findall(
                r'<em class="sig-param"><span class="n"><span class="pre">(_\w+)</span>', signature
            )
            private += re.findall(r"<p><strong>(_\w+)</strong>", described)
            if private:
                assert not dataclasses.is_dataclass(_resolve(name)), (page, name, private)
    assert entries


def test_the_built_reference_heads_its_entries_at_levels_two_to_six():
    for page, text in _built("api").items():
        signatures = [markup.partition("</dt>")[0] for _, markup in _BUILT_ENTRY.findall(text)]
        headings = [
            tag
            for signature in signatures
            for tag in re.findall(r'<[^>]*\brole="heading"[^>]*>', signature)
        ]
        assert [tag for tag in headings if not re.search(r'\baria-level="[2-6]"', tag)] == [], page
        # A page with entries has one at its top level, and that is a heading.
        assert headings or not signatures, page
