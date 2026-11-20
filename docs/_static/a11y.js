/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Small accessibility repairs the theme's markup needs at run time.
 *
 * The script is deferred and runs before the theme's own script. Each repair
 * looks for the elements it works on and leaves the page alone when they are
 * missing. Without JavaScript the page is what the theme wrote.
 */
(function () {
  "use strict";

  var root = document.documentElement;

  function each(selector, fn, scope) {
    (scope || document).querySelectorAll(selector).forEach(fn);
  }

  // Which part of the site a search result is from, by the start of its path.
  // Most API page titles end in "API", so those results say "Reference".
  var SECTIONS = [
    ["tutorials/", "Tutorial"],
    ["user_guide/", "User Guide"],
    ["api/", "Reference"],
    ["architecture/", "Architecture"],
  ];

  // Where else to look when a search finds nothing. A place is offered only if
  // this page already links to it, so a page that moves cannot leave a dead link.
  var ELSEWHERE = [
    ['link[rel="index"]', "Index"],
    ['.bd-docs-nav a[href$="api/index.html"]', "API Reference"],
    ['.bd-docs-nav a[href$="tutorials/index.html"]', "Tutorials"],
  ];

  var NO_MATCH = "No page matches. Try one word, or a function or class name.";

  // A search form is a search landmark, so it can be reached as one.
  function searchForms() {
    each("form.bd-search", function (form) {
      form.setAttribute("role", "search");
    });
  }

  // The search page fills #search-results after it loads, one result at a time.
  // A screen reader is told the outcome once, in one sentence, and not every
  // result as it arrives.
  function searchResults() {
    var results = document.getElementById("search-results");
    if (!results || !results.parentNode) {
      return;
    }
    results.setAttribute("role", "region");
    results.setAttribute("aria-label", "Search results");

    var status = document.createElement("p");
    status.className = "visually-hidden";
    status.setAttribute("role", "status");
    results.parentNode.insertBefore(status, results);

    // Titles that look alike sit together in the list ("Classical Calls API",
    // "Classical Calls at QPU Runtime"), so each result says where it is from.
    function label(link) {
      var base = root.dataset.content_root || "";
      var path = link.getAttribute("href") || "";
      // Looked at once, whether or not it gets a label.
      link.setAttribute("data-labelled", "");
      if (base && path.indexOf(base) === 0) {
        path = path.slice(base.length);
      }
      SECTIONS.some(function (section) {
        if (path.indexOf(section[0]) !== 0) {
          return false;
        }
        // A title that says it already ("Tutorials", "API Reference") is not
        // told twice.
        if (!new RegExp("\\b" + section[1] + "s?\\b", "i").test(link.textContent)) {
          var tag = document.createElement("span");
          // "nohighlight" keeps the search-term highlighter out of the label.
          tag.className = "a11y-result-kind nohighlight";
          tag.textContent = section[1];
          // Read aloud, the title would run into the label. The link's name
          // puts a comma between them; what is shown stays the same.
          link.setAttribute("aria-label", link.textContent.trim() + ", " + section[1]);
          link.append(" ", tag);
        }
        return true;
      });
    }

    // The stock sentence for no results speaks of categories, which this site
    // does not have. Say what to try instead, and where else to look.
    function noMatch(summary) {
      summary.setAttribute("data-no-match", "");
      summary.textContent = NO_MATCH;
      var places = [];
      ELSEWHERE.forEach(function (place) {
        var known = document.querySelector(place[0]);
        if (known) {
          var link = document.createElement("a");
          link.setAttribute("href", known.getAttribute("href"));
          link.textContent = place[1];
          places.push(link);
        }
      });
      places.forEach(function (link, i) {
        summary.append(i ? ", " : " Or browse: ", link);
      });
      if (places.length) {
        summary.append(".");
      }
    }

    // A short search ends while the page is still loading, and a status that
    // is written then is not reliably spoken. The sentence waits until a
    // moment after the page has loaded.
    var outcome = "";
    var loaded = false;

    function say() {
      if (loaded && outcome && status.textContent !== outcome) {
        status.textContent = outcome;
      }
    }

    function update() {
      each("ul.search > li > a:not([data-labelled])", label, results);
      // The search script writes its summary sentence last. Until then there
      // is nothing to say.
      var summary = results.querySelector("p.search-summary");
      if (!summary || !summary.textContent) {
        return;
      }
      if (!summary.hasAttribute("data-no-match") && !results.querySelector("ul.search li")) {
        noMatch(summary);
      }
      // Only the sentence is announced, not the links after it.
      outcome = summary.hasAttribute("data-no-match") ? NO_MATCH : summary.textContent;
      say();
    }

    function settled() {
      window.setTimeout(function () {
        loaded = true;
        say();
      }, 500);
    }

    new MutationObserver(update).observe(results, { childList: true, subtree: true });
    // Results that are already there, should the search ever start first.
    update();
    if (document.readyState === "complete") {
      settled();
    } else {
      window.addEventListener("load", settled);
    }
  }

  // The generated index and module-index tables only lay out lists.
  function indexTables() {
    each("table.indextable", function (table) {
      table.setAttribute("role", "presentation");
    });
  }

  // "Skip to main content" moves the reading position and keyboard focus. The
  // link itself is not followed: on a page that sets a base address, as the
  // "page not found" page does, its address is on another page.
  function skipToMain() {
    each('a.skip-link, a[href="#main-content"]', function (link) {
      link.addEventListener("click", function (event) {
        var main = document.getElementById("main-content");
        if (main) {
          if (!main.hasAttribute("tabindex")) {
            main.setAttribute("tabindex", "-1");
          }
          event.preventDefault();
          main.focus({ preventScroll: false });
          main.scrollIntoView();
        }
      });
    });
  }

  // At desktop width "On this page" comes after the whole article in Tab
  // order. A second skip link goes straight to it. The stylesheet hides the
  // link at the widths where the list is a drawer.
  function skipToPageContents() {
    var toc = document.getElementById("pst-page-toc-nav");
    var strip = document.getElementById("pst-skip-link");
    if (!toc || !strip || !toc.querySelector("a[href]")) {
      return;
    }
    var jump = document.createElement("a");
    jump.setAttribute("href", "#pst-page-toc-nav");
    jump.className = "a11y-skip-toc";
    jump.textContent = "Skip to On this page";
    jump.addEventListener("click", function (event) {
      var first = toc.querySelector("a[href]");
      if (first) {
        // Move focus only: no new address, no history entry, no page scroll.
        event.preventDefault();
        first.focus();
      }
    });
    strip.appendChild(jump);
  }

  // The sidebar's title and captions are headings of level 1 and 2 that come
  // before the page's own title. One level down each, the page title is the
  // only level 1.
  function sidebarHeadings() {
    each('.bd-docs-nav .bd-links__title[aria-level="1"]', function (title) {
      title.setAttribute("aria-level", "2");
    });
    each('.bd-docs-nav p.caption[aria-level="2"]', function (caption) {
      caption.setAttribute("aria-level", "3");
    });
  }

  // The icon-only header controls answer to the word people look for first
  // ("theme", "menu"). The rest of each name is the theme's own.
  function controlNames() {
    each(".theme-switch-button", function (button) {
      button.setAttribute("aria-label", "Theme: color mode");
    });
    each(".primary-toggle", function (button) {
      button.setAttribute("aria-label", "Menu: site navigation");
    });
  }

  // The color-mode switch shows its state only as an icon and a background
  // color. Its options say which one is on, and the button says that it
  // opens something.
  function colorMode() {
    if (!document.querySelector(".theme-change-button")) {
      return;
    }
    function mark() {
      // The theme treats an unset mode as "auto".
      var mode = root.dataset.mode || "auto";
      each(".theme-change-button", function (option) {
        option.setAttribute("aria-pressed", String(option.dataset.mode === mode));
      });
    }
    mark();
    new MutationObserver(mark).observe(root, { attributes: true, attributeFilter: ["data-mode"] });
    // The dropdown script keeps aria-expanded up to date once it has opened.
    each(".theme-switch-button", function (button) {
      if (!button.hasAttribute("aria-expanded")) {
        button.setAttribute("aria-expanded", "false");
      }
    });
  }

  // The theme's dialogs have no name and no way out that shows: they close on
  // Escape or on a press outside them. Each gets a name and a Close button,
  // and the buttons that open them say that a dialog follows.
  function dialogs() {
    // On the search page the Search button moves focus to the field in the
    // page and opens nothing.
    var ownField = document.querySelector(":not(#pst-search-dialog) > form.bd-search");
    [
      ["pst-search-dialog", "Search", ownField ? "" : ".search-button__button"],
      ["pst-primary-sidebar-modal", "Site navigation", ".primary-toggle"],
      ["pst-secondary-sidebar-modal", "On this page", ".secondary-toggle"],
    ].forEach(function (entry) {
      var dialog = document.getElementById(entry[0]);
      if (!dialog) {
        return;
      }
      if (!dialog.hasAttribute("aria-label") && !dialog.hasAttribute("aria-labelledby")) {
        dialog.setAttribute("aria-label", entry[1]);
      }
      var close = document.createElement("button");
      close.setAttribute("type", "button");
      close.className = "a11y-dialog-close";
      close.textContent = "Close";
      close.addEventListener("click", function () {
        dialog.close();
      });
      // In the search dialog the button follows the form, so the field keeps
      // first focus. The two drawers are empty until they open and the theme
      // adds the sidebar after the button, which is then the first stop. The
      // theme carries the button back into the sidebar with the rest when a
      // drawer closes; the stylesheet shows it only inside an open dialog.
      dialog.appendChild(close);
      if (entry[2]) {
        each(entry[2], function (opener) {
          opener.setAttribute("aria-haspopup", "dialog");
        });
      }
    });
  }

  // The navigation drawer opens at its top, which on a small or zoomed screen
  // can be a screen or more above the current page. When that page's link is
  // out of view, bring it to the middle of the drawer. Focus stays where the
  // theme puts it.
  function drawerPosition() {
    var drawer = document.getElementById("pst-primary-sidebar-modal");
    if (!drawer) {
      return;
    }
    each(".primary-toggle", function (toggle) {
      toggle.addEventListener("click", function () {
        // The theme fills and opens the drawer in its own handler for this
        // click, so measure on the next frame.
        window.requestAnimationFrame(function () {
          var here = drawer.open && drawer.querySelector(".bd-docs-nav a.current");
          if (!here) {
            return;
          }
          var box = drawer.getBoundingClientRect();
          var at = here.getBoundingClientRect();
          // The Close button stays at the top of the drawer, over the list.
          var close = drawer.querySelector(".a11y-dialog-close");
          var top = Math.max(box.top, close ? close.getBoundingClientRect().bottom : 0);
          if (at.top < top || at.bottom > box.bottom) {
            drawer.scrollTop += at.top - box.top - (box.height - at.height) / 2;
          }
        });
      });
    });
  }

  // On a long page "On this page" is taller than its box, and the theme moves
  // only the highlight. Bring the highlighted entry back to the middle when it
  // leaves the box, unless the reader is using the list.
  function pageContentsFollow() {
    var toc = document.getElementById("pst-page-toc-nav");
    if (!toc) {
      return;
    }
    new MutationObserver(function () {
      var box = toc.closest(".bd-sidebar-secondary");
      var entry = toc.querySelector("a[aria-current]");
      if (!box || !entry || box.matches(":hover") || box.contains(document.activeElement)) {
        return;
      }
      var frame = box.getBoundingClientRect();
      var at = entry.getBoundingClientRect();
      if (at.top < frame.top || at.bottom > frame.bottom) {
        box.scrollTop += at.top - frame.top - (frame.height - at.height) / 2;
      }
    }).observe(toc, { attributes: true, attributeFilter: ["aria-current"], subtree: true });
  }

  // On paper a closed figure description is only its label, so each one is
  // opened for printing. The ones opened here are closed again afterwards.
  function printDescriptions() {
    var opened = [];
    window.addEventListener("beforeprint", function () {
      each("details.figure-description:not([open])", function (details) {
        details.open = true;
        opened.push(details);
      });
    });
    window.addEventListener("afterprint", function () {
      opened.splice(0).forEach(function (details) {
        details.open = false;
      });
    });
  }

  // One repair that fails must not stop the others.
  [
    searchForms,
    searchResults,
    indexTables,
    skipToMain,
    skipToPageContents,
    sidebarHeadings,
    controlNames,
    colorMode,
    dialogs,
    drawerPosition,
    pageContentsFollow,
    printDescriptions,
  ].forEach(function (repair) {
    try {
      repair();
    } catch (error) {
      console.error("a11y.js: " + repair.name + " did not run.", error);
    }
  });
})();
