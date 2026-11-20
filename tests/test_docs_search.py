# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The search page lists first the result a reader looks for.

``docs/_ext/search_scorer.js`` orders the results of the search page, and
``docs/_ext/search_aliases.py`` gives a page the words it is found by. Most
of this needs no build: the scorer scores results written here, under Node,
and the aliases are held against the pages of the source tree and added to a
page that docutils parses here. One test searches a built site with the
site's own scripts, where the build of the ``docs`` preset has left one, for
the searches that must keep their first result. Without a site it is
skipped, and with ``CUDAQ_PULSE_REQUIRE_DOCS_SITE=1`` it fails instead. Node
is optional unless ``CUDAQ_PULSE_REQUIRE_NODE=1``. The docs workflow sets
both.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._tools import tool_env

publish_doctree = pytest.importorskip("docutils.core").publish_doctree
addnodes = pytest.importorskip("sphinx.addnodes")

_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "docs"
_EXT = _DOCS / "_ext"
_SCORER = _EXT / "search_scorer.js"
# Where the build of the ``docs`` preset writes the site.
_SITE = _ROOT / "build" / "docs" / "docs" / "html"
# A job that has just built the docs sets this, so that it cannot pass by
# skipping the search of the site it built.
_REQUIRE_SITE = "CUDAQ_PULSE_REQUIRE_DOCS_SITE"
pytestmark = pytest.mark.skipif(not _EXT.exists(), reason="needs the source tree")
sys.path.insert(0, str(_EXT))
search_aliases = pytest.importorskip("search_aliases")
sys.path.remove(str(_EXT))

# The weights of Sphinx's search script. A scorer replaces the script's own,
# so one that leaves a weight out breaks the search.
_WEIGHTS = {
    "objNameMatch": 11,
    "objPartialMatch": 6,
    "objPrio": {"0": 15, "1": 5, "2": -5},
    "objPrioDefault": 0,
    "title": 15,
    "partialTitle": 7,
    "term": 5,
    "partialTerm": 2,
}

# Results as the search script scores them for "controller", and the score
# each is listed by: (page, title, description of an object, score, new score).
_RESULTS = [
    (
        "api/quantum_systems",
        "cudaq_pulse.Controller",
        "Python class, in Quantum system API",
        16,
        16,
    ),
    ("api/kernel", "cudaq_pulse.Kernel.lower", "Python method, in Kernel API", 16, 16),
    (
        "api/classical",
        "cudaq_pulse.compiler.ClassicalCall.controller",
        "Python attribute, in Classical Calls API",
        16,
        10,
    ),
    (
        "api/compile",
        "cudaq_pulse.CompiledKernel.schedule",
        "Python property, in Compilation API",
        16,
        10,
    ),
    ("user_guide/controller_backends", "Writing a Controller Backend", None, 15, 15.8),
    ("tutorials/kernel_to_controller", "From Kernel to Controller", None, 15, 15.7),
    ("api/controller", "Controller Backends API", None, 15, 15.5),
    ("user_guide/classical_calls", "Classical Calls at QPU Runtime", None, 15, 15.3),
    ("getting_started", "Getting Started", None, 15, 15.3),
    ("tutorials/coherence", "Coherence: T1, Ramsey and Echo", None, 15, 15.2),
    ("api/classical", "Classical Calls API", None, 15, 15),
    ("architecture/pipeline", "Compilation Pipeline", None, 15, 15),
    ("tutorials/index", "Tutorials", None, 15, 14.7),
    ("index", "CUDA-Q Pulse", None, 15, 14.7),
    ("architecture/dialects/pulse", "‘pulse’ Dialect", None, 15, 14.6),
    # A section of a page named after the query.
    ("user_guide/controller_backends", "Writing a Controller Backend > Errors", None, 8, 8.8),
]

# A stemmer as the search script calls one, which only drops a plural's "s".
_STEMMER = 'var Stemmer = function () { this.stemWord = (word) => word.replace(/s$/, ""); };'

# Searches of the built site: (query, result, how near the top it must be).
# The documented names stay first, and so does the page to install from.
_WANTED = [
    ("emulate", "api/classical.html#cudaq_pulse.classical.emulation.emulate", 1),
    ("simulate", "api/simulation.html#cudaq_pulse.simulate", 1),
    ("compile", "api/compile.html#cudaq_pulse.compile", 1),
    ("plot", "api/viz.html#cudaq_pulse.viz.plot_schedule", 1),
    ("lower", "api/compile.html#cudaq_pulse.CompiledKernel.lower", 1),
    ("readout", "api/ops.html#cudaq_pulse.ops.readout", 1),
    ("install", "getting_started.html", 1),
    ("T1", "tutorials/coherence.html", 3),
    ("controller", "api/quantum_systems.html#cudaq_pulse.Controller", 3),
]

# Prints the scorer's weights and its score of each result, for a query in
# the page's address. What the page defined before the scorer is run first.
# JSON leaves out the scoring function, so the weights are what is printed.
_SCORE = r"""
const fs = require("fs");
const vm = require("vm");
const [file, given] = process.argv.slice(1);
const { query, before, results } = JSON.parse(given);
const search = query === null ? "" : "?" + new URLSearchParams({ q: query });
const page = { URLSearchParams, location: { search } };
page.window = page;
vm.createContext(page);
vm.runInContext(before, page);
vm.runInContext(fs.readFileSync(file, "utf8"), page, { filename: file });
const scorer = vm.runInContext("Scorer", page);
const scores = results.map((result) => scorer.score(result));
console.log(JSON.stringify({ weights: scorer, scores }));
"""

# Prints the first six results of each query, best first, as the search
# script of a built site orders them, and whether the site has a scorer.
_SEARCH = r"""
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const [site, queries] = process.argv.slice(1);
const page = {
  URLSearchParams,
  location: { search: "" },
  document: { getElementById: () => null },
  DOCUMENTATION_OPTIONS: {},
  SPHINX_HIGHLIGHT_ENABLED: false,
  _: (text) => text,
  _ready: () => {},
};
page.window = page;
vm.createContext(page);
for (const file of ["_static/searchtools.js", "_static/language_data.js", "searchindex.js"]) {
  vm.runInContext(fs.readFileSync(path.join(site, file), "utf8"), page, { filename: file });
}
const [search, scorer] = vm.runInContext("[Search, Scorer]", page);
const found = {};
for (const query of JSON.parse(queries)) {
  page.location.search = "?" + new URLSearchParams({ q: query });
  // The search script keeps its results worst first.
  found[query] = search
    ._performSearch(...search._parseQuery(query))
    .reverse()
    .slice(0, 6)
    .map(([docname, , anchor]) => docname + ".html" + anchor);
}
console.log(JSON.stringify({ scored: typeof scorer.score === "function", found }));
"""


def _node(script: str, *arguments: str) -> dict:
    """What *script* prints, run under Node with *arguments*."""
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CUDAQ_PULSE_REQUIRE_NODE"):
            pytest.fail("CUDAQ_PULSE_REQUIRE_NODE is set, but node is not on PATH")
        pytest.skip("needs Node.js to run the search scripts")
    result = subprocess.run(
        [node, "-e", script, *arguments],
        capture_output=True,
        text=True,
        timeout=60,
        env=tool_env(),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout)


def _scored(query: str | None, results: list[tuple], before: str = "") -> dict:
    """The scorer's weights and its scores of *results*, each a (page, title,
    description, score), on a search page asked for *query*."""
    rows = [
        [page, title, "", description, score, f"{page}.rst", "object" if description else "text"]
        for page, title, description, score in results
    ]
    given = json.dumps({"query": query, "before": before, "results": rows})
    return _node(_SCORE, str(_SCORER), given)


def _score(query: str | None, page: str, title: str, score: int, before: str = "") -> float:
    return _scored(query, [(page, title, None, score)], before)["scores"][0]


def _page(text: str = "Title\n=====\n\nText.\n"):
    return publish_doctree(text, settings_overrides={"doctitle_xform": False})


def _app(docname: str, aliases: dict[str, tuple[str, ...]]):
    """As much of Sphinx as the extension asks for while it reads *docname*."""
    return SimpleNamespace(
        config=SimpleNamespace(search_aliases=aliases), env=SimpleNamespace(docname=docname)
    )


def _words(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.lower()))


def test_the_scorer_declares_every_weight():
    source = _SCORER.read_text()
    missing = [name for name in _WEIGHTS if not re.search(rf"^\s*{name}:", source, re.M)]
    assert not missing, f"docs/_ext/search_scorer.js leaves out {missing}"


def test_the_scorer_is_not_a_file_of_the_site():
    # Sphinx copies the scorer into language_data.js, and every file under
    # docs/_static to the site as it is. A scorer kept there is published a
    # second time, as a file that no page loads.
    assert _SCORER.is_file()
    copies = sorted(
        path.relative_to(_ROOT).as_posix() for path in (_DOCS / "_static").rglob(_SCORER.name)
    )
    assert not copies, f"the site would also publish {copies}: keep the scorer in docs/_ext"


def test_the_scorer_has_every_weight_when_it_loads_first():
    assert _scored("controller", [])["weights"] == _WEIGHTS


def test_the_scorer_keeps_the_weights_of_the_search_script():
    # A weight the search script sets, or one a later Sphinx adds, stays.
    weights = _scored("controller", [], before="var Scorer = { title: 20, later: 3 };")["weights"]
    assert weights == {**_WEIGHTS, "title": 20, "later": 3}


def test_attributes_come_after_pages_and_pages_are_ordered():
    scores = _scored("controller", [result[:4] for result in _RESULTS])["scores"]
    assert scores == pytest.approx([result[4] for result in _RESULTS])
    # A page with the query in a heading stays between an attribute named
    # like the query and a function, class or method named like it.
    pages = [
        new
        for (_, _, description, old, _), new in zip(_RESULTS, scores, strict=True)
        if description is None and old == 15
    ]
    assert 10 < min(pages) and max(pages) < 16


def test_a_page_is_named_after_the_whole_query():
    title = "Calling Classical Code During Quantum Runtime"
    assert _score("error code", "tutorials/classical_calls", title, 15) == pytest.approx(15.2)
    named = _score("Measure and React", "tutorials/measure_and_react", "Measure and React", 16)
    assert named == pytest.approx(16.7)


def test_a_page_is_named_by_the_stem_of_a_word():
    assert _score("kernel", "user_guide/kernels", "Pulse Kernels", 15) == pytest.approx(15.3)
    stemmed = _score("kernel", "user_guide/kernels", "Pulse Kernels", 15, before=_STEMMER)
    assert stemmed == pytest.approx(15.8)


def test_a_search_with_no_query_in_the_address_is_scored():
    assert _score(None, "user_guide/kernels", "Pulse Kernels", 15) == pytest.approx(15.3)


def test_every_alias_is_a_word_its_page_uses():
    assert search_aliases.ALIASES
    for docname, aliases in search_aliases.ALIASES.items():
        source = _DOCS / f"{docname}.rst"
        assert source.exists(), f"{docname} in ALIASES is no page"
        text = " ".join(source.read_text().split())
        for alias in aliases:
            used = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.I)
            assert used, f"{docname} does not use its alias {alias!r}"


def test_no_part_of_an_alias_is_another_word():
    # The search answers a query that is part of an entry and at least half
    # as long, so the page of an alias "reset" would come first for "set".
    # A word of the alias itself may find the page. Parts under three letters
    # are not words a reader searches for.
    used = _words(" ".join(page.read_text() for page in _DOCS.rglob("*.rst")))
    for docname, aliases in search_aliases.ALIASES.items():
        for alias in map(str.lower, aliases):
            shortest = max(3, math.ceil(len(alias) / 2))
            parts = {
                alias[start : start + length]
                for length in range(shortest, len(alias))
                for start in range(len(alias) - length + 1)
            }
            also = sorted((parts & used) - _words(alias))
            assert not also, f"the alias {alias!r} of {docname} also answers a search for {also}"


def test_a_page_carries_its_aliases_as_main_index_entries():
    doctree = _page()
    before = list(doctree.children)
    search_aliases._add_aliases(_app("a/page", {"a/page": ("T2", "spin echo")}), doctree)
    *rest, entries = doctree.children
    assert rest == before
    assert isinstance(entries, addnodes.index)
    assert entries["entries"] == [
        ("single", "T2", "", "main", None),
        ("single", "spin echo", "", "main", None),
    ]


def test_a_page_with_no_alias_is_left_as_it_is():
    doctree = _page()
    before = doctree.pformat()
    search_aliases._add_aliases(_app("another/page", {"a/page": ("T2",)}), doctree)
    assert doctree.pformat() == before


def test_sphinx_reads_the_pages_again_when_the_aliases_change():
    values, handlers = [], []
    app = SimpleNamespace(
        add_config_value=lambda *value: values.append(value),
        connect=lambda *handler: handlers.append(handler),
    )
    assert search_aliases.setup(app) == {"parallel_read_safe": True, "parallel_write_safe": True}
    assert values == [("search_aliases", search_aliases.ALIASES, "env")]
    assert handlers == [("doctree-read", search_aliases._add_aliases)]


def test_the_built_site_lists_the_wanted_result_first():
    if not (_SITE / "searchindex.js").exists():
        if os.environ.get(_REQUIRE_SITE):
            pytest.fail(f"{_REQUIRE_SITE} is set, but build/docs/docs/html holds no site")
        pytest.skip("needs the site that the docs preset builds")
    aliases = [
        (alias, f"{docname}.html", 1)
        for docname, words in search_aliases.ALIASES.items()
        for alias in words
    ]
    wanted = _WANTED + aliases
    site = _node(_SEARCH, str(_SITE), json.dumps([query for query, _, _ in wanted]))
    # A site built before the scorer was added has none either.
    assert site["scored"], (
        "the site's language_data.js has no scorer: build the docs again, "
        "and if it still has none, see html_search_scorer in docs/conf.py"
    )
    late = {
        query: site["found"][query]
        for query, result, within in wanted
        if result not in site["found"][query][:within]
    }
    assert not late, f"the wanted result is not at the top of: {late}"


def test_a_job_that_requires_the_site_does_not_skip_its_search(monkeypatch, tmp_path):
    # A folder with no site in it stands for a build that wrote none. The
    # variable is named here as the docs workflow names it.
    monkeypatch.setitem(globals(), "_SITE", tmp_path)
    monkeypatch.delenv("CUDAQ_PULSE_REQUIRE_DOCS_SITE", raising=False)
    with pytest.raises(pytest.skip.Exception):
        test_the_built_site_lists_the_wanted_result_first()
    monkeypatch.setenv("CUDAQ_PULSE_REQUIRE_DOCS_SITE", "1")
    # A skip that got out of here would end this test as skipped too.
    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as outcome:
        test_the_built_site_lists_the_wanted_result_first()
    assert outcome.type is pytest.fail.Exception
    assert "build/docs/docs/html holds no site" in str(outcome.value)
