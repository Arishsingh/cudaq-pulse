# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The search finds a page by a word it uses often but has in no heading.

Sphinx's search scores a page by where it has the query: 15 points for a
heading, 5 for the text, however often the word is there. The coherence
tutorial is the page about T2 and the first tutorial simulates a Rabi
oscillation, and neither has the word in a heading, so a search for "T2" or
"Rabi" listed them among the pages that mention the word once.

Each page of ``ALIASES`` carries its words as index entries, and the search
lists a page first when the query is one of its entries.

* An alias is a word the page itself uses, so the search promises nothing
  the page does not have. ``tests/test_docs_search.py`` holds the map
  against the pages.
* The search also matches a query that is part of an entry and at least half
  as long: an alias "reset" would list its page first for "set". The same
  test refuses an alias with a part that is another word of the pages.
* The entries are for the search only. Sphinx has collected the general
  index by the time they are added, so it does not list them, and they put
  nothing on the page.

``ALIASES`` is the default of the configuration value ``search_aliases``, so
Sphinx reads the pages again when the map changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sphinx import addnodes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from docutils import nodes
    from sphinx.application import Sphinx

# The words each page is found by, by the name Sphinx gives the page.
ALIASES: dict[str, tuple[str, ...]] = {
    "tutorials/coherence": ("T2",),
    "tutorials/first_pulse_program": ("Rabi", "Rabi oscillation"),
}


def alias_entries(words: Iterable[str]) -> addnodes.index:
    """The index entries that make a page the first result for each of *words*.

    The entries have no target, so the result leads to the top of the page,
    and the search lists it once, in place of the page's own result.
    """
    # The search lists a "main" entry with the pages; any other entry comes
    # after every page.
    node = addnodes.index(entries=[("single", word, "", "main", None) for word in words])
    node["inline"] = False
    return node


def _add_aliases(app: Sphinx, doctree: nodes.document) -> None:
    # The entries are part of the page as read, where the search index finds
    # them. They go last, clear of the title and of the fields that open a page.
    words = app.config.search_aliases.get(app.env.docname)
    if words:
        doctree.append(alias_entries(words))


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_config_value("search_aliases", ALIASES, "env")
    app.connect("doctree-read", _add_aliases)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
