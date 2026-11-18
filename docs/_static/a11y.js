/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Small accessibility repairs the theme's markup needs at run time.
 */
(function () {
  "use strict";

  // The search page fills #search-results after it loads; announce it.
  var results = document.getElementById("search-results");
  if (results) {
    results.setAttribute("role", "region");
    results.setAttribute("aria-label", "Search results");
    results.setAttribute("aria-live", "polite");
  }

  // The generated index and module-index tables only lay out lists.
  document.querySelectorAll("table.indextable").forEach(function (table) {
    table.setAttribute("role", "presentation");
  });

  // "Skip to main content" moves the reading position, and now keyboard focus.
  document.querySelectorAll('a.skip-link, a[href="#main-content"]').forEach(function (link) {
    link.addEventListener("click", function () {
      var main = document.getElementById("main-content");
      if (main) {
        if (!main.hasAttribute("tabindex")) {
          main.setAttribute("tabindex", "-1");
        }
        main.focus({ preventScroll: false });
      }
    });
  });
})();
