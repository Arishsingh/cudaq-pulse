# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API names and example paths in running text link to what they name.

The pages write the names of functions and classes as inline literals,
``pulse.simulate`` or ``ScheduledKernel``, and a reader would have to find
each in the API reference by hand. A literal becomes a link to the object's
entry there when it names exactly one documented object:

* a full name, ``cudaq_pulse.controller.FORMAT``;
* a name under an import alias the pages use: ``pulse.compile`` is
  ``cudaq_pulse.compile`` and ``backends.get`` is
  ``cudaq_pulse.backends.get``;
* the dotted end of one full name only, ``ScheduledKernel.bind``;
* a bare class or exception name in CamelCase, ``PhysicalQudit``;
* a bare call of one of the package's own functions, ``simulate(...)``, or
  of what a kernel calls, ``gaussian(...)``, or of the one function or
  method with a compound name, ``export_to_controller()``.

An alias may also stand before a name that is enough without it:
``pulse.Bits`` is the class ``Bits`` and ``pulse.drive(...)`` the call
``drive(...)``. A decorator's ``@`` and a call's arguments are not part of
the name.

A bare lowercase word is never linked, and no bare name leads to an
attribute, a property or a constant: most such words on the pages are
argument names or keys of a file format. A bare call of one word, ``run()``
or ``emulate()``, is not linked either: such a word is as often a method of
the reader's own object as the one function the reference documents. Nor is
``pulse.gaussian`` without a call: so written, in a docstring as on the
architecture pages, it is the name of an MLIR operation.

Only the first mention of an object in each section becomes a link, so a
page gains a few links and not one for every mention; a link the page
already has counts as that mention. Headings, signatures and the page that
documents the object stay as they are. So do the architecture pages, where
``pulse.sync`` and its like name MLIR operations and not Python calls.

A literal that starts with ``examples/`` and names a file or a directory of
the repository's examples links to it at the address the ``example`` role
uses, on every page.

The build warns when a name in ``MUST_RESOLVE`` no longer leads to an object,
so that renaming part of the API cannot drop these links unnoticed.
"""

from __future__ import annotations

import re
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from docutils import nodes
from sphinx import addnodes
from sphinx.util import logging

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment

logger = logging.getLogger(__name__)

# Names the pages lean on. Each must lead to an object in every build.
MUST_RESOLVE = (
    "pulse.simulate",
    "pulse.compile",
    "ScheduledKernel",
    "cudaq_pulse.controller.FORMAT",
    "cudaq_pulse.backends.Backend",
    "cudaq_pulse.Controller",
)

# The pages import the package as ``pulse``. What a kernel calls is written
# bare, ``gaussian(...)``, and documented under ``cudaq_pulse.ops``. The alias
# does not lead there: ``pulse.gaussian`` is the name of an MLIR operation.
PACKAGE = ("cudaq_pulse.", "cudaq_pulse.ops.")
ALIASES = {"pulse.": ("cudaq_pulse.",), "backends.": ("cudaq_pulse.backends.",)}

# Pages where a literal of these shapes names something else.
UNLINKED_PAGES = "architecture/"

EXAMPLE_ROLE = "example"
EXAMPLES = "examples"

_CLASSES = frozenset({"class", "exception"})
_CALLABLES = frozenset({"function", "method", "classmethod", "staticmethod"})
# The role that names each kind of object in the classes of Sphinx's own links.
_ROLES = {
    "function": "func",
    "method": "meth",
    "classmethod": "meth",
    "staticmethod": "meth",
    "class": "class",
    "exception": "exc",
    "attribute": "attr",
    "property": "attr",
    "data": "data",
    "module": "mod",
}

_NAME = re.compile(r"(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*")
# ``Controller`` and ``FluxCZ``, but not the constants ``X90`` and ``FORMAT``.
_CAMEL_CASE = re.compile(r"[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*")
_EXAMPLE_PATH = re.compile(rf"{EXAMPLES}/[\w./-]*")
_NOT_RUNNING_TEXT = (nodes.title, nodes.rubric, addnodes.desc_signature)

# A link and the classes the literal inside it takes.
_Link = tuple[nodes.reference, list[str]]


class Target(NamedTuple):
    """Where a name leads: the object's public name, its page and its anchor there."""

    name: str
    docname: str
    anchor: str
    objtype: str


class Names:
    """The documented objects, under each way a page may write their names.

    *objects* is the Python domain's inventory: each full name with an entry
    that has ``docname``, ``node_id``, ``objtype`` and ``aliased``.
    """

    def __init__(self, objects: Mapping[str, Any]) -> None:
        # Sphinx lists an object once more under each private path it is
        # imported from, with the anchor of its public entry.
        public = {
            (entry.docname, entry.node_id): Target(
                name, entry.docname, entry.node_id, entry.objtype
            )
            for name, entry in objects.items()
            if not entry.aliased
        }
        self.full: dict[str, Target] = {}
        self.ends: dict[str, set[Target]] = {}
        self.classes: dict[str, set[Target]] = {}
        self.calls: dict[str, set[Target]] = {}
        for name, entry in objects.items():
            target = public.get((entry.docname, entry.node_id))
            if target is None:
                continue
            self.full[name] = target
            parts = name.split(".")
            for start in range(1, len(parts) - 1):
                self.ends.setdefault(".".join(parts[start:]), set()).add(target)
            if target.objtype in _CLASSES:
                self.classes.setdefault(parts[-1], set()).add(target)
            elif target.objtype in _CALLABLES:
                self.calls.setdefault(parts[-1], set()).add(target)

    def find(self, name: str, called: bool) -> Target | None:
        """The one object that *name* names, written as a call or not."""
        if "." not in name:
            return self._bare(name, called)
        alias = next((alias for alias in ALIASES if name.startswith(alias)), "")
        spellings = [name] + [package + name[len(alias) :] for package in ALIASES.get(alias, ())]
        for spelling in spellings:
            if spelling in self.full:
                return self.full[spelling]
        target = _only(self.ends.get(name))
        if target is None and alias:
            # ``pulse.Bits`` is documented as ``cudaq_pulse.classical.Bits``:
            # the alias may stand before a name that is enough without it.
            return self.find(name[len(alias) :], called)
        return target

    def _bare(self, name: str, called: bool) -> Target | None:
        if _CAMEL_CASE.fullmatch(name):
            return _only(self.classes.get(name))
        if not called:
            return None
        for package in PACKAGE:
            target = self.full.get(package + name)
            if target is not None and target.objtype in _CALLABLES:
                return target
        return _only(self.calls.get(name)) if "_" in name else None


def _only(targets: set[Target] | None) -> Target | None:
    return next(iter(targets)) if targets and len(targets) == 1 else None


def written_name(text: str) -> tuple[str, bool] | None:
    """The name a literal writes and whether it writes a call of it:
    ``("pulse.compile", True)`` for ``pulse.compile(kernel)``. None if *text*
    is more than a name, a decorator or one call."""
    name, paren, arguments = text.strip().removeprefix("@").partition("(")
    if paren:
        depths = list(accumulate((char == "(") - (char == ")") for char in paren + arguments))
        # The call closes once, at the end: ``compile(k).mlir()`` names no one object.
        if depths[-1] != 0 or 0 in depths[:-1]:
            return None
    return (name, bool(paren)) if _NAME.fullmatch(name) else None


def resolve(text: str, names: Names) -> Target | None:
    """The one object the literal *text* names, or None if it names none or several."""
    written = written_name(text)
    return names.find(*written) if written else None


def missing_names(names: Names) -> list[str]:
    """The names of ``MUST_RESOLVE`` that lead to no object."""
    return [name for name in MUST_RESOLVE if resolve(name, names) is None]


def _link_first_mentions(doctree: nodes.document, link_for: Callable[[str], _Link | None]) -> int:
    """Put each plain literal of running text inside the link that *link_for*
    gives its text, where its section does not lead there yet. Return how
    many were linked."""
    seen: set[tuple[int, str | None]] = set()
    count = 0
    # ``link_operations`` of ``dialect_pages.py`` walks the same way: repair the two together.
    for literal in list(doctree.findall(nodes.literal)):
        if len(literal) != 1 or not isinstance(literal[0], nodes.Text):
            continue
        # Sphinx writes a link only inside text: one elsewhere must hold an image.
        if not isinstance(literal.parent, nodes.TextElement):
            continue
        section = link = None
        node = literal.parent
        while node is not None and not isinstance(node, _NOT_RUNNING_TEXT):
            if link is None and isinstance(node, nodes.reference):
                link = node
            if section is None and (node.parent is None or isinstance(node, nodes.section)):
                section = node
            node = node.parent
        if node is not None:
            continue
        linked = link is not None
        if not linked:
            # A literal written by a role (a file, a key, highlighted code)
            # has classes: its role has said what it is.
            made = None if literal["classes"] else link_for(literal.astext())
            if made is None:
                continue
            link, classes = made
        # A link that is already there is its section's mention of where it leads.
        mention = (id(section), link.get("reftitle") or link.get("refuri"))
        if not linked and mention not in seen:
            literal["classes"] += classes
            literal.parent.replace(literal, link)
            link += literal
            count += 1
        seen.add(mention)
    return count


def link_api_names(
    doctree: nodes.document, docname: str, names: Names, uri_of: Callable[[str], str]
) -> int:
    """Link the API names on the page *docname* and return how many were linked.

    *uri_of* gives the address of another page as seen from this one.
    """
    if docname.startswith(UNLINKED_PAGES):
        return 0

    def link_for(text: str) -> _Link | None:
        target = resolve(text, names)
        if target is None or target.docname == docname:
            return None
        # As Sphinx's own link to the object: the full name as the link's
        # title, the kind of object in the literal's classes.
        uri = f"{uri_of(target.docname)}#{target.anchor}"
        link = nodes.reference("", "", internal=True, refuri=uri, reftitle=target.name)
        return link, ["xref", "py", "py-" + _ROLES.get(target.objtype, "obj")]

    return _link_first_mentions(doctree, link_for)


def link_example_paths(doctree: nodes.document, examples: Path, address: str) -> int:
    """Link the literals that name a file or a directory under *examples* and
    return how many were linked. *address* is the ``example`` role's, with
    ``%s`` for the path below the examples."""

    def link_for(text: str) -> _Link | None:
        if not _EXAMPLE_PATH.fullmatch(text) or ".." in text.split("/"):
            return None
        path = text[len(EXAMPLES) + 1 :].rstrip("/")
        if (examples / path).is_file():
            uri = address % path
        elif (examples / path).is_dir():
            # The hosting site shows a file under blob/ and a directory under tree/.
            uri = address.replace("/blob/", "/tree/", 1) % path
        else:
            return None
        classes = [f"extlink-{EXAMPLE_ROLE}"]
        return nodes.reference("", "", internal=False, refuri=uri, classes=classes), []

    return _link_first_mentions(doctree, link_for)


def _link_examples(app: Sphinx, doctree: nodes.document) -> None:
    # The examples lie beside the pages, in the repository and in the copy
    # the build reads. Without them, or without the role, the literals stay
    # as written.
    role = getattr(app.config, "extlinks", {}).get(EXAMPLE_ROLE)
    examples = Path(app.srcdir).parent / EXAMPLES
    if role and "%s" in role[0] and examples.is_dir():
        link_example_paths(doctree, examples, role[0])


def _link_names(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    # Only now is every page read and the inventory whole. The tree is the
    # copy made for writing this page, so the stored pages stay as read.
    names = Names(app.env.get_domain("py").objects)
    link_api_names(doctree, docname, names, lambda to: app.builder.get_relative_uri(docname, to))


def _check_names(app: Sphinx, env: BuildEnvironment) -> None:
    for name in missing_names(Names(env.get_domain("py").objects)):
        logger.warning(
            "the pages link %r to the API reference, which no longer documents one object "
            "of that name: restore it, or update MUST_RESOLVE in docs/_ext/api_links.py",
            name,
            type="api_links",
            subtype="must_resolve",
        )


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("doctree-read", _link_examples)
    app.connect("doctree-resolved", _link_names)
    app.connect("env-check-consistency", _check_names)
    # Raise env_version when the links to examples change: they are stored
    # with the pages as read, and a build directory from before keeps those.
    return {"env_version": 1, "parallel_read_safe": True, "parallel_write_safe": True}
