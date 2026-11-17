// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Draws each scene of a pulse-visual JSON file as two static SVGs, light and
// dark, with the drawing code the page runs: the figure's still frame, which
// readers without JavaScript or with reduced motion see. The stylesheet goes
// inside each file, so the SVG shows the same through <img>.
//
//   node tools/render-pulse-visual.js DATA.json [OUT_DIR]
//
// OUT_DIR defaults to the JSON file's folder; each scene writes
// <scene>.svg and <scene>-dark.svg.
"use strict";

const fs = require("fs");
const path = require("path");

const STATIC = path.join(__dirname, "..", "docs", "_static");
const visual = require(path.join(STATIC, "pulse-visual.js"));

// Just enough of a DOM for pulse-visual.js's drawing code.
class Element {
  constructor(tag) {
    this.tag = tag;
    this.attrs = new Map();
    this.children = [];
    this.text = null;
  }
  // Coordinates to two decimals: far finer than a pixel, and the same on
  // every machine.
  setAttribute(k, v) {
    this.attrs.set(k, String(v).replace(/(\d+\.\d\d)\d+/g, "$1"));
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  replaceChildren(...children) {
    this.children = children;
  }
  set textContent(text) {
    this.text = String(text);
    this.children = [];
  }
  get textContent() {
    return this.text || "";
  }
  serialize(indent = "") {
    const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    const attrs = [...this.attrs].map(([k, v]) => ` ${k}="${esc(v)}"`).join("");
    if (this.text !== null) return `${indent}<${this.tag}${attrs}>${esc(this.text)}</${this.tag}>`;
    if (!this.children.length) return `${indent}<${this.tag}${attrs}/>`;
    const inner = this.children.map((c) => c.serialize(indent + " ")).join("\n");
    return `${indent}<${this.tag}${attrs}>\n${inner}\n${indent}</${this.tag}>`;
  }
}

const doc = { createElementNS: (_ns, tag) => new Element(tag) };

function render(scene, css, dark) {
  const svg = new Element("svg");
  svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  svg.setAttribute("class", dark ? "pv-svg pv-dark" : "pv-svg");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-labelledby", "title desc");
  const title = svg.appendChild(new Element("title"));
  title.setAttribute("id", "title");
  title.textContent = scene.title;
  const desc = svg.appendChild(new Element("desc"));
  desc.setAttribute("id", "desc");
  desc.textContent = visual.description(scene);
  const style = svg.appendChild(new Element("style"));
  const draw = visual.build(svg, scene, doc);
  draw(visual.restFrame(scene));
  style.textContent = used(css, svg);
  const header =
    "<!--\n  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n" +
    "  SPDX-License-Identifier: Apache-2.0\n  Drawn by tools/render-pulse-visual.js from the figure's JSON; do not edit.\n-->\n";
  return header + svg.serialize() + "\n";
}

// The rules that style the SVG itself; the page-only rules stay out.
function svgRules(css) {
  return css
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/html\[data-theme="dark"\] \.pv-svg,\s*/g, "")
    .split("}")
    .map((rule) => rule.trim())
    .filter((rule) => rule.startsWith(".pv-"))
    .map((rule) => `${rule.replace(/\s+/g, " ")}}`)
    .join("\n");
}

// The rules of *css* that style something in *svg*, so a still changes only
// when the rules it uses do.
function used(css, svg) {
  const classes = new Set();
  const walk = (e) => {
    for (const name of (e.attrs.get("class") || "").split(/\s+/)) classes.add(name);
    e.children.forEach(walk);
  };
  walk(svg);
  return css
    .split("\n")
    .filter((rule) =>
      rule
        .split("{")[0]
        .split(",")
        .some((selector) => {
          const names = (selector.match(/\.pv-[\w-]+/g) || []).map((n) => n.slice(1));
          return names.every((n) => n === "pv-svg" || n === "pv-dark" || classes.has(n));
        })
    )
    .join("\n");
}

if (require.main === module) {
  const [data, out] = process.argv.slice(2);
  const folder = out || path.dirname(data);
  const css = svgRules(fs.readFileSync(path.join(STATIC, "pulse-visual.css"), "utf8"));
  fs.mkdirSync(folder, { recursive: true });
  for (const scene of JSON.parse(fs.readFileSync(data, "utf8")).scenes) {
    for (const dark of [false, true]) {
      const file = path.join(folder, `${scene.name}${dark ? "-dark" : ""}.svg`);
      fs.writeFileSync(file, render(scene, css, dark));
    }
  }
}

module.exports = { render, svgRules, Element };
