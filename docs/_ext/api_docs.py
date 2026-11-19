# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What autodoc writes into the API reference, put right for its readers.

* An attribute, a property or a constant shows its docstring whole. Napoleon
  reads the first line of such a docstring as ``type: description``, so
  "The frequency it plays at (``None``: the qudit's)." would print "the
  qudit's)." and then, as the type, "The frequency it plays at (``None``".
  The docstrings of the package are prose, so one written ``int: the
  count`` prints as written.
* An entry shows no docstring that is not its own. A constant without one
  has that of the type of its value, ``str`` or ``tuple``, and a name
  re-exported from another package has that package's (``pulse.Length`` is
  ``typing.Annotated``): the page describes these. A field that a class
  reads through a descriptor shows what the class says of it to type
  checkers, under ``if TYPE_CHECKING:``.
* The signature of a dataclass reads as Python. Where a factory makes the
  default of a field, Python prints ``<factory>``, which Sphinx cannot read:
  it then cuts the whole argument list at each comma, the one inside a
  default such as ``(0.0, 0.0)`` too. Such a default shows as ``...``. A
  field with a private name, one that starts with an underscore, is shown
  neither in the signature nor under "Parameters".
* No list of fields is empty. Sphinx writes one for a class whose
  constructor takes nothing.
* A name in a type links to the class of that name, or it is plain text.
  Sphinx 9 also links a name that no class has to a constant or an
  attribute, as a type alias may be one, and finds it by the end of its
  name alone: the type variable ``T`` of a signature led to the gate
  ``Gate.T``. A constant is still linked by its own name.
* Each entry at the top level of a page, a function, a class, an exception
  or a constant, is a heading one level below the section it is in, so that
  a reader who moves by heading stops at each and can list them. The
  members of a class are not headings: they belong to its entry, and "On
  this page" lists them. A guide that describes its functions as entries
  gets the same headings.
* Each ``[source]`` link is named after its entry, "[source] of simulate":
  a list of a page's links would otherwise hold one name many times. The
  name starts with the words on the page, and none are added to the page's
  text, where a search of the page or a copied signature would meet them.
  The ``[docs]`` links of a page of source, each back to the entry of the
  definition under it, are named the same way.
"""

from __future__ import annotations

import ast
import dataclasses
import html
import inspect
import re
import sys
import textwrap
from functools import cache, partial, reduce
from typing import TYPE_CHECKING, Any

from docutils import nodes
from sphinx import addnodes
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.inspect import safe_getattr, signature_from_str, stringify_signature
from sphinx.util.nodes import find_pending_xref_condition

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator

    from sphinx.application import Sphinx

# The kinds of entry whose first line napoleon reads as "type: description".
PROSE = ("attribute", "property", "data")
# An empty comment: napoleon finds no type in it, and it prints nothing.
SHIELD = ".."
# What Python prints as the default of a field whose default a factory makes.
FACTORY = "<factory>"

# A [source] link as Sphinx writes it: its attributes, then what it shows.
_SOURCE_LINK = re.compile(r'<a\b([^>]*)>(\s*<span class="viewcode-link">.*?)</a>', re.S)
# The [docs] link above a definition on a page of source, in the same parts.
_DOCS_LINK = re.compile(r'<a\b([^>]*\bclass="viewcode-back"[^>]*)>([^<]*)</a>')
_FRAGMENT = re.compile(r'\bhref="[^"]*#([^"]+)"')
_TAG = re.compile(r"<[^>]+>")
# A list of fields that holds no field.
_NO_FIELDS = re.compile(r'<dl class="field-list[^"]*">\s*</dl>\n?')


def _words(text: object) -> str:
    """*text* with each run of white space as one space, "" if it is no text."""
    return " ".join(text.split()) if isinstance(text, str) else ""


def _borrowed(what: str, name: str, obj: Any, lines: list[str]) -> bool:
    """Whether *lines*, the docstring autodoc found for the entry *name* of
    value *obj*, is not the entry's own: that of the value's type, or the
    value's own when the value comes from another package."""
    text = _words(" ".join(lines))
    if not text or what not in ("data", "attribute"):
        return False
    if text == _words(type(obj).__doc__):
        return True
    home = safe_getattr(obj, "__module__", None)
    foreign = isinstance(home, str) and home.partition(".")[0] != name.partition(".")[0]
    return foreign and text == _words(inspect.getdoc(obj))


def _owner(name: str) -> Any:
    """What holds the entry *name*, a class or a module, or None if it is not
    loaded."""
    path = name.split(".")[:-1]
    for cut in range(len(path), 0, -1):
        module = sys.modules.get(".".join(path[:cut]))
        if module is not None:
            try:
                return reduce(getattr, path[cut:], module)
            except AttributeError:
                return None
    return None


@cache
def _declared(owner: type) -> dict[str, str]:
    """The docstrings that the source of the class *owner* gives type checkers
    alone, by name: those of the definitions under ``if TYPE_CHECKING:`` in
    its body. Empty if the source cannot be read."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(owner)))
    except (OSError, TypeError, SyntaxError):
        return {}
    body = tree.body[0].body if tree.body and isinstance(tree.body[0], ast.ClassDef) else []
    found = {}
    for block in body:
        if not isinstance(block, ast.If):
            continue
        if getattr(block.test, "id", getattr(block.test, "attr", "")) != "TYPE_CHECKING":
            continue
        for definition in block.body:
            if isinstance(definition, ast.FunctionDef):
                found[definition.name] = ast.get_docstring(definition) or ""
    return found


def own_docstring(what: str, name: str, obj: Any, lines: list[str]) -> None:
    """Make *lines*, the docstring of the entry *name*, its own, and keep its
    first line from being read as a type. Runs before napoleon."""
    if _borrowed(what, name, obj, lines):
        owner = _owner(name)
        declared = _declared(owner) if inspect.isclass(owner) else {}
        lines[:] = declared.get(name.rpartition(".")[2], "").splitlines()
    first = next((line for line in lines if line.strip()), "")
    if what in PROSE and ":" in first:
        lines[:0] = [SHIELD, ""]


def unshield(lines: list[str]) -> None:
    """Take from *lines* the comment that kept the first line whole. Runs
    after napoleon."""
    if lines[:1] == [SHIELD]:
        del lines[: 2 if lines[1:2] == [""] else 1]


def _before_napoleon(
    app: Sphinx, what: str, name: str, obj: Any, options: Any, lines: list[str]
) -> None:
    own_docstring(what, name, obj, lines)


def _after_napoleon(
    app: Sphinx, what: str, name: str, obj: Any, options: Any, lines: list[str]
) -> None:
    unshield(lines)


def readable_signature(signature: str, private: Collection[str] = ()) -> str | None:
    """The argument list *signature* as Sphinx can read it: ``...`` where a
    factory makes a default, and without the parameters named in *private*.
    None if it is no Python even then."""
    try:
        read = signature_from_str(signature.replace(FACTORY, "..."))
        public = [value for name, value in read.parameters.items() if name not in private]
        return stringify_signature(read.replace(parameters=public))
    except (SyntaxError, NotImplementedError, ValueError):
        return None


def _signature(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str, str | None] | None:
    """The signature of the dataclass *obj* without what its generated
    constructor alone brings: the mark of a factory and the private fields.
    None, which leaves the signature as it is, for anything else."""
    if not signature or not (inspect.isclass(obj) and dataclasses.is_dataclass(obj)):
        return None
    private = {field.name for field in dataclasses.fields(obj) if field.name.startswith("_")}
    if not private and FACTORY not in signature:
        return None
    shown = readable_signature(signature, private)
    if shown is None:
        return None
    # Sphinx lists under "Parameters" each parameter whose type it has noted
    # for the entry, whatever the signature shows.
    noted = app.env.current_document.autodoc_annotations.get(name, {})
    for field in private:
        noted.pop(field, None)
    return shown, return_annotation


def unlink_guessed_types(doctree: nodes.Node, find: Callable[..., list[tuple[str, Any]]]) -> None:
    """Leave as text each name in a type that no class has and that Sphinx
    would link by a guess: to a constant or an attribute whose name only ends
    with it. *find* is the search of the Python domain: for a name in the
    module and the class of its entry, as a role, in a mode."""
    for xref in list(doctree.findall(addnodes.pending_xref)):
        if (xref.get("refdomain"), xref.get("reftype")) != ("py", "class"):
            continue
        # Sphinx searches by the end of names, mode 1, for a reference that
        # has this attribute, whatever its value: every name in a type has.
        if not xref.hasattr("refspecific"):
            continue
        module, target = xref.get("py:module"), xref.get("reftarget", "")
        scope = (module, xref.get("py:class"), target)
        if find(*scope, "class", 1):
            continue
        found = {name for name, _ in find(*scope, "data", 1) or find(*scope, "attr", 1)}
        # Its own name, in full or within the module of the entry, is no guess.
        if found and not found & {target, f"{module}.{target}"}:
            shown = find_pending_xref_condition(xref, "*")
            xref.replace_self(list((xref if shown is None else shown).children))


class _NoGuessedTypes(SphinxPostTransform):
    """Runs ``unlink_guessed_types`` on a page before its references are
    resolved, which Sphinx does at priority 10."""

    default_priority = 5

    def run(self, **kwargs: Any) -> None:
        find = partial(self.env.get_domain("py").find_obj, self.env)
        unlink_guessed_types(self.document, find)


def _above(node: nodes.Node) -> Iterator[nodes.Node]:
    while node.parent is not None:
        node = node.parent
        yield node


def head_entries(doctree: nodes.document) -> None:
    """Make the name of each entry that is in no other entry a heading, one
    level below the section the entry is in and of level 6 at most."""
    for entry in doctree.findall(addnodes.desc):
        above = list(_above(entry))
        if any(isinstance(node, addnodes.desc) for node in above):
            continue
        level = min(6, 1 + sum(isinstance(node, nodes.section) for node in above))
        # An entry with several signatures, the overloads of a function, is
        # one stop: its first.
        first = entry.first_child_matching_class(addnodes.desc_signature)
        name = None if first is None else next(entry[first].findall(addnodes.desc_name), None)
        if name is not None:
            # The HTML writer drops attributes it does not know, so the role
            # is written as it will stand in the page.
            name[:] = [
                nodes.raw("", f'<span role="heading" aria-level="{level}">', format="html"),
                *name.children,
                nodes.raw("", "</span>", format="html"),
            ]


def _named(link: re.Match[str]) -> str:
    attributes, shown = link.groups()
    entry = _FRAGMENT.search(attributes)
    if entry is None or "aria-label=" in attributes:
        return link.group(0)
    name = html.unescape(f"{_TAG.sub('', shown).strip()} of {entry.group(1)}")
    return f'<a{attributes} aria-label="{html.escape(name)}">{shown}</a>'


def name_source_links(body: str) -> str:
    """The page's HTML with each ``[source]`` link named after the entry whose
    source it shows, and each ``[docs]`` link of a page of source after the
    entry it leads to: the words the link shows, then the entry's name."""
    return _DOCS_LINK.sub(_named, _SOURCE_LINK.sub(_named, body))


def drop_empty_field_lists(body: str) -> str:
    """The page's HTML without the lists of fields that hold no field."""
    return _NO_FIELDS.sub("", body)


def _resolved(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    head_entries(doctree)


def _page_context(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: Any
) -> None:
    # Every page comes here with its text as HTML, the pages of source too,
    # which have no doctree. Sphinx writes a page of source again only when
    # its module has changed: a build directory from before a change to the
    # names of the links keeps its old pages of source.
    body = context.get("body")
    if body:
        context["body"] = name_source_links(drop_empty_field_lists(body))


def setup(app: Sphinx) -> dict[str, Any]:
    app.setup_extension("sphinx.ext.autodoc")
    # Napoleon reads each docstring at the default priority, 500.
    app.connect("autodoc-process-docstring", _before_napoleon, priority=400)
    app.connect("autodoc-process-docstring", _after_napoleon, priority=600)
    # Sphinx must have noted the types of an entry's parameters before this
    # listener takes those of the private fields away. Sphinx 9 notes them
    # before the event, Sphinx 8 in a listener of its own at the default
    # priority.
    app.connect("autodoc-process-signature", _signature, priority=600)
    app.add_post_transform(_NoGuessedTypes)
    app.connect("doctree-resolved", _resolved)
    app.connect("html-page-context", _page_context)
    # Raise env_version when what is written into the entries changes: a build
    # directory from before then reads every page again.
    return {"env_version": 2, "parallel_read_safe": True, "parallel_write_safe": True}
