/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The order of the search page's results.
 *
 * Sphinx copies this file into language_data.js (html_search_scorer). With
 * Sphinx's own scoring an attribute named like the query (16 points) comes
 * before a page with the query in a heading (15), and results with the same
 * score are listed by the alphabet. Half of the documented objects are
 * attributes, so four of the first five results for "controller" were fields
 * of data classes, and the reference pages, whose titles come early in the
 * alphabet, came before the guides.
 *
 * Sphinx's weights stay as they are. score() moves attributes and properties
 * below the pages with the query in a heading, and orders pages of the same
 * score: a page named after the query first, then the user guide and the
 * pages at the top of the site, the tutorials, the reference, and last the
 * pages that list other pages and the generated dialect reference. A page
 * moves by less than one point, so one with the query in a heading stays
 * below a function, class or method named like the query.
 */
var Scorer = Object.assign(
  // Sphinx's weights, for a page that loads this file before searchtools.js.
  {
    objNameMatch: 11,
    objPartialMatch: 6,
    objPrio: { 0: 15, 1: 5, 2: -5 },
    objPrioDefault: 0,
    title: 15,
    partialTitle: 7,
    term: 5,
    partialTerm: 2,
  },
  // The weights of the search script itself, when it loaded first.
  typeof Scorer === "undefined" ? {} : Scorer,
  (function () {
    "use strict";

    var stemmer = null;

    // The words of a title or a query, as the stems the search index holds.
    function words(text) {
      if (!stemmer && typeof Stemmer === "function") {
        stemmer = new Stemmer();
      }
      return String(text || "")
        .toLowerCase()
        .split(/[^\p{Letter}\p{Number}_]+/u)
        .filter(Boolean)
        .map(function (word) {
          return stemmer ? stemmer.stemWord(word) : word;
        });
    }

    // The query, from the address the search script reads it from.
    var asked = { search: null, words: [] };
    function queryWords() {
      var search = window.location.search;
      if (search !== asked.search) {
        asked = { search: search, words: words(new URLSearchParams(search).get("q")) };
      }
      return asked.words;
    }

    // Whether every word of the query is in the title: "T1" names "Coherence:
    // T1, Ramsey and Echo". One word of a longer query does not, or "code" in
    // "error code" would name every page with "Code" in its title.
    function namedAfterQuery(title) {
      var query = queryWords();
      var named = words(title);
      return (
        query.length > 0 &&
        query.every(function (word) {
          return named.indexOf(word) >= 0;
        })
      );
    }

    return {
      // A result is [docname, title, anchor, description, score, filename, kind].
      score: function (result) {
        var docname = String(result[0]);
        var description = result[3];
        var score = result[4];

        // Only an object has a description: "Python attribute, in <page>".
        if (typeof description === "string") {
          return /^Python (attribute|property)\b/.test(description) ? score - 6 : score;
        }

        // A section is titled "<page> > <section>".
        if (namedAfterQuery(String(result[1] || "").split(" > ")[0])) {
          score += 0.5;
        }
        if (docname === "index" || /\/index$/.test(docname)) {
          score -= 0.3;
        } else if (/^architecture\/dialects\//.test(docname)) {
          // The generated dialect reference has a heading for every operation.
          score -= 0.4;
        } else if (/^tutorials\//.test(docname)) {
          score += 0.2;
        } else if (!/^(api|architecture)\//.test(docname)) {
          // The user guide and the pages at the top of the site.
          score += 0.3;
        }
        return score;
      },
    };
  })(),
);
