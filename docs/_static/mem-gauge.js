// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// What programs of the ISA tutorials use of a toy controller: instruction
// memory, waveform memory and oscillators, each against the limit the
// controller's description states. docs/tutorials/figures/isa_limits.py
// measures each from the export and the toy listing, and records whether the
// compiler refuses the program. tests/test_toy_flow.py runs the functions this
// file exports under Node and checks them against the data and the compiler.
"use strict";

(function () {
  const COLUMNS = {
    pulses: { title: "instruction memory", one: "pulse", many: "pulses" },
    samples: { title: "waveform memory", one: "sample", many: "samples" },
    frequencies: { title: "oscillators", one: "frequency", many: "frequencies" },
  };
  // The limit sits at this fraction of a track. A use past the end of the
  // track is cut there, with a break mark, and its number says how far.
  const LIMIT_AT = 0.7;
  const ROW_MS = 650;

  function number(n) {
    return n.toLocaleString("en-US");
  }

  // A use of *limit* as fractions of the track: up to the limit, past it,
  // and whether it runs off the end.
  function gauge(value, limit) {
    const scaled = (value / limit) * LIMIT_AT;
    const reach = Math.min(scaled, 1);
    return { reach, inside: Math.min(reach, LIMIT_AT), over: Math.max(0, reach - LIMIT_AT), cut: scaled > 1 };
  }

  // The columns where *row* uses more than the controller holds.
  function overLimits(row, limits, columns) {
    return columns.filter((key) => row[key] > limits[key]);
  }

  function valueText(row, key) {
    const unit = row[key] === 1 ? COLUMNS[key].one : COLUMNS[key].many;
    let text = `${number(row[key])} ${unit}`;
    if (key === "samples")
      text += ` in ${row.entries.length} ${row.entries.length === 1 ? "entry" : "entries"}`;
    return text;
  }

  function noteText(row) {
    return row.refused ? "refused when it compiles" : row.note || "";
  }

  // What a screen reader hears: the limits, then each row's use.
  function description(data, rows, columns) {
    const limits = columns.map((key) => `${number(data.limits[key])} ${COLUMNS[key].many}`);
    const parts = [`A ${data.model} holds ${limits.join(", ")}.`];
    for (const row of rows) {
      const uses = columns.map((key) => valueText(row, key)).join(", ");
      parts.push(`${row.label}: ${uses}${row.refused ? ", refused when it compiles" : ""}.`);
    }
    return parts.join(" ");
  }

  const api = { COLUMNS, LIMIT_AT, ROW_MS, gauge, overLimits, valueText, noteText, description };
  if (typeof module === "object" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  // ── Drawing ────────────────────────────────────────────────────────────

  const SVG_NS = "http://www.w3.org/2000/svg";
  const W = 760;
  const LABEL_W = 236;
  const GAP = 28;
  const ROW_H = 46;
  const TOP = 36;
  const TRACK_H = 12;

  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  let count = 0;

  function build(svg, data, rows, columns) {
    const id = `pmg${count++}`;
    const width = Math.min(400, (W - LABEL_W - GAP * (columns.length - 1)) / columns.length);
    const height = TOP + rows.length * ROW_H;
    svg.setAttribute("viewBox", `0 0 ${W} ${height}`);
    el("title", { id: `${id}-title` }, svg, `What each program uses of a ${data.model}`);
    el("desc", { id: `${id}-desc` }, svg, description(data, rows, columns));
    svg.setAttribute("aria-labelledby", `${id}-title ${id}-desc`);
    const hatch = el(
      "pattern",
      {
        id: `${id}-hatch`,
        width: 5,
        height: 5,
        patternUnits: "userSpaceOnUse",
        patternTransform: "rotate(45)",
      },
      el("defs", {}, svg)
    );
    el("rect", { width: 2.2, height: 5, class: "mg-hatch" }, hatch);

    columns.forEach((key, c) => {
      const x = LABEL_W + c * (width + GAP);
      el("text", { x, y: 13, class: "mg-head" }, svg, COLUMNS[key].title);
      el("text", { x, y: 27, class: "mg-small" }, svg, `holds ${number(data.limits[key])}`);
    });

    const bars = [];
    rows.forEach((row, r) => {
      const y = TOP + r * ROW_H + 8;
      el("text", { x: 0, y: y + 10, class: "mg-label" }, svg, row.label);
      el(
        "text",
        { x: 0, y: y + 26, class: `mg-small${row.refused ? " mg-refused" : ""}` },
        svg,
        noteText(row)
      );
      columns.forEach((key, c) => {
        const x = LABEL_W + c * (width + GAP);
        const limit = data.limits[key];
        const g = gauge(row[key], limit);
        el("rect", { x, y, width, height: TRACK_H, rx: 2, class: "mg-track" }, svg);
        const fill = el("rect", { x, y, width: 0, height: TRACK_H, rx: 2, class: "mg-fill" }, svg);
        const spill = el(
          "rect",
          { x: x + LIMIT_AT * width, y, width: 0, height: TRACK_H, fill: `url(#${id}-hatch)` },
          svg
        );
        // Waveform memory shows each table entry as a block of its own.
        const seams = el("g", { class: "mg-seams" }, svg);
        if (key === "samples") {
          let at = 0;
          for (const entry of row.entries.slice(0, -1)) {
            at += entry;
            const sx = x + gauge(at, limit).reach * width;
            el("path", { d: `M ${sx} ${y} V ${y + TRACK_H}`, class: "mg-seam" }, seams);
          }
        }
        const lx = x + LIMIT_AT * width;
        el("path", { d: `M ${lx} ${y - 3} V ${y + TRACK_H + 3}`, class: "mg-limit" }, svg);
        const cut = g.cut
          ? el(
              "path",
              { d: `M ${x + width - 6} ${y + TRACK_H + 3} L ${x + width - 2} ${y - 3}`, class: "mg-break" },
              svg
            )
          : null;
        const over = row[key] > limit;
        const label = el("text", { x, y: y + 26, class: `mg-small${over ? " mg-over" : ""}` }, svg, "");
        bars.push({
          r,
          fill,
          spill,
          seams,
          cut,
          label,
          g,
          width,
          text: valueText(row, key),
          value: row[key],
        });
      });
    });
    return bars;
  }

  // Row r of the bars at progress f, from 0 to 1, eased out.
  function show(bars, r, f) {
    const ease = 1 - Math.pow(1 - Math.max(0, Math.min(1, f)), 3);
    for (const bar of bars) {
      if (bar.r !== r) continue;
      const reach = ease * bar.g.reach;
      bar.fill.setAttribute("width", Math.min(reach, LIMIT_AT) * bar.width);
      bar.spill.setAttribute("width", Math.max(0, reach - LIMIT_AT) * bar.width);
      bar.seams.style.opacity = ease > 0 ? 1 : 0;
      if (bar.cut) bar.cut.style.opacity = ease >= 1 ? 1 : 0;
      bar.label.textContent = ease >= 1 ? bar.text : f > 0 ? number(Math.round(bar.value * ease)) : "";
    }
  }

  function mount(root) {
    const data = JSON.parse(document.getElementById(root.dataset.source).textContent);
    const byName = new Map(data.rows.map((row) => [row.name, row]));
    const rows = root.dataset.rows.split(" ").map((name) => byName.get(name));
    const columns = root.dataset.columns.split(" ");
    const svg = el("svg", { role: "img", class: "mg-svg" });
    const button = document.createElement("button");
    button.type = "button";
    button.className = "mg-button";
    button.textContent = "Replay";
    root.replaceChildren(svg, button);
    const bars = build(svg, data, rows, columns);
    const fallback = root.parentElement.querySelector(".pulse-fallback");
    if (fallback) fallback.hidden = true;

    const all = (f) => rows.forEach((_, r) => show(bars, r, f));
    // Each row fills in turn, ROW_MS after the one above it.
    const at = (elapsed) => rows.forEach((_, r) => show(bars, r, (elapsed - r * ROW_MS) / ROW_MS));
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      all(1);
      button.hidden = true;
      return;
    }

    let raf = 0;
    const play = () => {
      cancelAnimationFrame(raf);
      all(0);
      const begin = performance.now();
      const frame = (now) => {
        const elapsed = now - begin;
        at(elapsed);
        raf = elapsed < (rows.length + 1) * ROW_MS ? requestAnimationFrame(frame) : 0;
      };
      raf = requestAnimationFrame(frame);
    };
    button.onclick = play;
    all(0);
    // Fills once, the first time most of it is in view.
    const seen = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          seen.disconnect();
          play();
        }
      },
      { threshold: 0.6 }
    );
    seen.observe(root);
    // For screenshots: stop, and draw the moment *elapsed* ms into playing.
    root.memGaugeShow = (elapsed) => {
      seen.disconnect();
      cancelAnimationFrame(raf);
      at(elapsed);
    };
  }

  function start() {
    document.querySelectorAll(".pulse-mem-gauge").forEach(mount);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
