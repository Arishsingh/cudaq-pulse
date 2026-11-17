// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The tutorials' animated figures. Each figure is one scene of a JSON file
// (format pulse-visual/1) that a figure script in docs/tutorials/figures
// writes from what the page compiles or simulates: lanes and the spans that
// play on them, windows whose length the program doesn't know, messages
// between lanes, the cases a branch can take with the share of shots each
// gets, and snapshots to move between. This file computes no schedule. It
// places what the data says at the time the playhead is on, plays once when
// the figure scrolls into view, and shows its still frame to readers who ask
// for reduced motion. The model functions at the top are pure;
// tests/test_pulse_visual.py runs them under Node, and
// tools/render-pulse-visual.js draws the no-script fallback with the same
// drawing code.
"use strict";

(function () {
  const CASE_MS = 4800;
  const HOLD_MS = 1400;
  const STILL_MS = 1600;
  const MORPH_MS = 1100;
  // Cases after the first PLAYED wait in the small multiples, and for the
  // reader's steps.
  const PLAYED = 2;

  // ── Model ──────────────────────────────────────────────────────────────

  function clamp01(f) {
    return Math.max(0, Math.min(1, f));
  }

  function ease(f) {
    const x = clamp01(f);
    return x < 0.5 ? 2 * x * x : 1 - 2 * (1 - x) * (1 - x);
  }

  function pct(w) {
    return `${(w * 100).toFixed(1)}%`;
  }

  // Whether every [name, value] of *rule* holds in *given*.
  function applies(rule, given) {
    return (rule || []).every(([name, value]) => name in given && given[name] === value);
  }

  function givenOf(scene, c) {
    return scene.cases && scene.cases.length ? scene.cases[c].given : {};
  }

  // The items of a list that play in case *c*.
  function inCase(scene, items, c) {
    const given = givenOf(scene, c);
    return (items || []).filter((x) => applies(x.when, given));
  }

  // A broken axis: each cut [a, b, w] draws the idle stretch from a to b as
  // w ns. visual(t) is where t sits on the drawn axis, real(v) its inverse,
  // and the playhead moves at a steady pace along the drawn axis.
  function visual(scene, t) {
    let v = 0;
    let last = 0;
    for (const [a, b, w] of scene.cuts || []) {
      if (t <= a) break;
      if (t < b) return v + (a - last) + ((t - a) / (b - a)) * w;
      v += a - last + w;
      last = b;
    }
    return v + (t - last);
  }

  function real(scene, v) {
    let at = 0;
    let last = 0;
    for (const [a, b, w] of scene.cuts || []) {
      const plain = a - last;
      if (v <= at + plain) return last + (v - at);
      if (v < at + plain + w) return a + ((v - at - plain) / w) * (b - a);
      at += plain + w;
      last = b;
    }
    return last + (v - at);
  }

  function endOf(scene) {
    return scene.end || 1;
  }

  function drawnSpan(scene) {
    return visual(scene, endOf(scene));
  }

  function played(scene) {
    return Math.min(PLAYED, scene.cases && scene.cases.length ? scene.cases.length : 1);
  }

  // What plays over time: the first cases for CASE_MS each, then a hold; or
  // each snapshot held still, then a move to the next.
  function beats(scene) {
    const out = [];
    if (scene.snapshots) {
      scene.snapshots.forEach((_, k) => {
        out.push({ kind: "still", step: k, ms: STILL_MS });
        if (k + 1 < scene.snapshots.length) out.push({ kind: "move", step: k, ms: MORPH_MS });
      });
      return out;
    }
    for (let c = 0; c < played(scene); c++) {
      out.push({ kind: "play", c, ms: CASE_MS });
      out.push({ kind: "hold", c, ms: HOLD_MS });
    }
    return out;
  }

  function totalMs(scene) {
    return beats(scene).reduce((sum, b) => sum + b.ms, 0);
  }

  // The frame *elapsed* ms after the start: a case and a time, or a
  // snapshot and how far the move to the next one has gone.
  function frameAt(scene, elapsed) {
    let left = Math.max(0, elapsed);
    for (const b of beats(scene)) {
      if (left < b.ms) {
        const f = left / b.ms;
        const end = endOf(scene);
        if (b.kind === "play") return { c: b.c, t: real(scene, f * drawnSpan(scene)), step: 0, f: 0, done: false };
        if (b.kind === "hold") return { c: b.c, t: end, step: 0, f: 0, done: false };
        return { c: 0, t: end, step: b.step, f: b.kind === "move" ? f : 0, done: false };
      }
      left -= b.ms;
    }
    return lastFrame(scene);
  }

  function lastFrame(scene) {
    const end = endOf(scene);
    if (scene.snapshots) return { c: 0, t: end, step: scene.snapshots.length - 1, f: 0, done: true };
    return { c: played(scene) - 1, t: end, step: 0, f: 0, done: true };
  }

  // The frame a still figure shows: the first case played to its end, with
  // every case in the small multiples; or the last snapshot.
  function restFrame(scene) {
    if (scene.snapshots) return lastFrame(scene);
    return { c: 0, t: endOf(scene), step: 0, f: 0, done: true };
  }

  // The frames a reader steps through: each moment of each case, or each
  // snapshot.
  function stops(scene) {
    const end = endOf(scene);
    if (scene.snapshots) return scene.snapshots.map((_, k) => ({ c: 0, t: end, step: k, f: 0 }));
    const out = [];
    (scene.cases || [{ moments: [] }]).forEach((kase, c) => {
      for (const [t] of kase.moments || []) out.push({ c, t, step: 0, f: 0 });
      out.push({ c, t: end, step: 0, f: 0 });
    });
    return out;
  }

  // The snapshot a move into step + 1 starts from: its "from", or the one
  // before it.
  function source(scene, frame) {
    const to = scene.snapshots[Math.min(frame.step + 1, scene.snapshots.length - 1)];
    const from = frame.f > 0 && to.from !== undefined ? to.from : frame.step;
    return [scene.snapshots[from], to];
  }

  // Where each span is at a frame: its own place, or between two
  // snapshots. A span a snapshot leaves out (null) grows from, or shrinks
  // to, the start of where the other one has it.
  function spansAt(scene, frame) {
    if (!scene.snapshots) return (scene.spans || []).map((s) => ({ ...s }));
    const [from, to] = frame.f > 0 ? source(scene, frame) : [scene.snapshots[frame.step], scene.snapshots[frame.step]];
    const f = ease(frame.f);
    const out = [];
    for (const s of scene.spans || []) {
      let p0 = from.spans[s.id];
      let p1 = to.spans[s.id];
      if (!p0 && !p1) continue;
      p0 = p0 || [p1[0], p1[0]];
      p1 = p1 || [p0[0], p0[0]];
      out.push({ ...s, a: p0[0] + (p1[0] - p0[0]) * f, b: p0[1] + (p1[1] - p0[1]) * f });
    }
    return out;
  }

  function snapshotIndex(scene, frame) {
    const k = frame.f >= 0.5 ? frame.step + 1 : frame.step;
    return Math.min(k, scene.snapshots.length - 1);
  }

  function snapshotAt(scene, frame) {
    return scene.snapshots ? scene.snapshots[snapshotIndex(scene, frame)] : null;
  }

  // How much of [a, b] the playhead has covered.
  function covered(a, b, t) {
    return b > a ? clamp01((t - a) / (b - a)) : t >= a ? 1 : 0;
  }

  function live(spans, t) {
    return spans.filter((s) => s.a <= t && t < s.b);
  }

  function captionAt(scene, frame) {
    const snap = snapshotAt(scene, frame);
    if (snap) return snap.say || snap.label;
    const kase = (scene.cases || [])[frame.c];
    if (!kase) return "";
    let text = "";
    for (const [t, line] of kase.moments || []) if (t <= frame.t) text = line;
    return text;
  }

  function headerAt(scene, frame) {
    const snap = snapshotAt(scene, frame);
    if (snap) return snap.label;
    const kase = (scene.cases || [])[frame.c];
    return kase ? kase.label : scene.title;
  }

  // What a screen reader hears, and the fallback's alt text: the title,
  // then the first case's moments and every case with its share of shots,
  // or each snapshot.
  function description(scene) {
    const parts = [`${scene.title}.`];
    if (scene.snapshots) {
      for (const s of scene.snapshots) parts.push(s.say || s.label);
    } else if (scene.cases && scene.cases.length) {
      parts.push(...(scene.cases[0].moments || []).map((m) => m[1]));
      for (const kase of scene.cases) {
        const share = kase.weight !== undefined ? ` (${kase.value || pct(kase.weight)} of shots)` : "";
        parts.push(`${kase.label.replace(/\.$/, "")}${share}.`);
      }
    }
    return parts.join(" ");
  }

  // Axis ticks: about six along the drawn axis, at 1, 2 or 5 times a power
  // of ten, none inside a cut.
  function ticks(scene) {
    const end = endOf(scene);
    const raw = drawnSpan(scene) / 6;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw);
    const out = [];
    const cut = (t) => (scene.cuts || []).some(([a, b]) => t > a && t < b);
    for (let t = 0; t <= end + 1e-9; t += step) {
      const v = Math.round(t * 1e6) / 1e6;
      if (!cut(v)) out.push(v);
    }
    return { step, values: out, unit: end >= 5000 ? "µs" : "ns" };
  }

  const api = {
    CASE_MS, HOLD_MS, STILL_MS, MORPH_MS, PLAYED,
    applies, inCase, visual, real, beats, totalMs, frameAt, lastFrame, restFrame, stops, spansAt,
    snapshotAt, snapshotIndex, covered, live, captionAt, headerAt, description, ticks, ease,
  };

  // ── Drawing ────────────────────────────────────────────────────────────

  const SVG_NS = "http://www.w3.org/2000/svg";
  const W = 760;
  const LABEL_W = 132;
  const PAD = 14;
  // A lane: a band for labels, then its bars.
  const LANE = 32;
  const BAR = 12;
  const BAR_H = 16;

  function maker(doc) {
    return function el(tag, attrs, parent, text) {
      const e = doc.createElementNS(SVG_NS, tag);
      for (const [k, v] of Object.entries(attrs || {})) if (v !== undefined && v !== null) e.setAttribute(k, String(v));
      if (text !== undefined) e.textContent = text;
      if (parent) parent.appendChild(e);
      return e;
    };
  }

  function pill(el, parent, x, y, text, cls) {
    const w = 10 + text.length * 6.4;
    const g = el("g", { class: `pv-pill ${cls || ""}` }, parent);
    el("rect", { x: x - w / 2, y: y - 8, width: w, height: 16, rx: 8 }, g);
    el("text", { x, y: y + 4, "text-anchor": "middle" }, g, text);
    return g;
  }

  function fmtTick(t, unit) {
    return unit === "µs" ? String(Math.round((t / 1000) * 1000) / 1000) : String(t);
  }

  // Each panel lays itself out below *y* and returns its height; draw() then
  // updates it for a frame. A panel draws into its own group, so draw()
  // replaces what moves and keeps what doesn't.
  const PANELS = {
    timeline: {
      build(ctx, y) {
        const { el, scene, x } = ctx;
        const g = el("g", { class: "pv-timeline" }, ctx.svg);
        const tk = ticks(scene);
        const axisY = y + 14;
        const rows = {};
        scene.lanes.forEach((lane, i) => (rows[lane.id] = axisY + 2 + i * LANE));
        const marks = [...(scene.marks || []), ...(scene.snapshots || []).flatMap((s) => s.marks || [])];
        const dueRoom = marks.some((m) => m.role === "due") ? 14 : 0;
        const bottom = axisY + 2 + scene.lanes.length * LANE;
        for (const t of tk.values) {
          el("path", { d: `M ${x(t)} ${axisY + 4} V ${bottom}`, class: "pv-grid" }, g);
          el("text", { x: x(t), y: axisY, class: "pv-small", "text-anchor": "middle" }, g, fmtTick(t, tk.unit));
        }
        el("text", { x: LABEL_W - 8, y: axisY, class: "pv-small", "text-anchor": "end" }, g, tk.unit);
        for (const [a, b] of scene.cuts || []) {
          const mid = (x(a) + x(b)) / 2;
          el("path", { d: `M ${mid - 5} ${axisY + 9} l 4 -8 M ${mid} ${axisY + 9} l 4 -8`, class: "pv-break" }, g);
          el("rect", { x: x(a), y: axisY + 10, width: x(b) - x(a), height: bottom - axisY - 10, class: "pv-cut" }, g);
        }
        for (const lane of scene.lanes) {
          const cls = lane.role === "link" ? "pv-lane pv-classical-text" : "pv-lane";
          el("text", { x: LABEL_W - 8, y: rows[lane.id] + BAR + 12, class: cls, "text-anchor": "end" }, g, lane.label);
        }
        ctx.tl = { rows, axisY, bottom, layer: el("g", {}, g), head: el("path", { class: "pv-head" }, g) };
        return bottom - y + 6 + dueRoom;
      },
      draw(ctx, frame, spans) {
        const { el, scene, tl, x } = ctx;
        const { rows } = tl;
        const t = frame.t;
        const layer = tl.layer;
        layer.replaceChildren();
        const end = endOf(scene);
        const snap = snapshotAt(scene, frame);
        // A label that doesn't fit in its bar sits above it, right of the
        // last label above that lane.
        const taken = {};
        const above = el("g", {});
        const label = (lane, x0, x1, text, cls, over = "") => {
          const w = text.length * 5.6;
          if (w < x1 - x0 - 8) {
            el("text", { x: (x0 + x1) / 2, y: rows[lane] + BAR + 12, "text-anchor": "middle", class: `pv-small ${cls}` }, layer, text);
            return;
          }
          const at = Math.min(Math.max((x0 + x1) / 2 - w / 2, taken[lane] || 0), W - PAD - w);
          el("text", { x: at, y: rows[lane] + BAR - 3, class: `pv-small pv-halo ${over}` }, above, text);
          taken[lane] = at + w + 6;
        };
        const windows = snap ? (frame.f === 0 ? snap.windows || [] : []) : inCase(scene, scene.windows, frame.c);
        for (const w of windows) {
          const y = rows[w.lane] + BAR;
          const x0 = x(w.a);
          const x1 = x(Math.min(w.b, end));
          el("rect", { x: x0, y, width: Math.max(1, x1 - x0), height: BAR_H, class: `pv-window pv-window-${w.role}`, fill: w.role === "late" ? ctx.hatch : null }, layer);
          // The part of the window the playhead has passed.
          if (!snap && w.role === "answer") {
            const reach = Math.max(0, Math.min(x1, x(t)) - x0);
            el("rect", { x: x0, y, width: reach, height: BAR_H, class: "pv-window-passed" }, layer);
          }
          label(w.lane, x0, x1, w.label, "pv-window-text", "pv-window-text");
        }
        const plays = new Set(inCase(scene, spans, frame.c).map((s) => s.id));
        // A still snapshot keeps where the one before it had each span that
        // moved or left, as a dotted ghost.
        const past = snap && frame.step > 0 && frame.f === 0 ? scene.snapshots[snap.from !== undefined ? snap.from : frame.step - 1] : null;
        const labels = (snap && snap.labels) || {};
        const bar = (lane, a, b) => ({ x: x(a), y: rows[lane] + BAR, width: Math.max(2.5, x(b) - x(a)), height: BAR_H, rx: 1.5 });
        if (past) {
          for (const s of scene.spans) {
            const was = past.spans[s.id];
            const now = snap.spans[s.id];
            if (was && (!now || was[0] !== now[0] || was[1] !== now[1])) el("rect", { ...bar(s.lane, ...was), class: "pv-ghost pv-was" }, layer);
          }
        }
        for (const s of spans) {
          if (s.b - s.a < 1e-9) continue;
          const shape = bar(s.lane, s.a, s.b);
          el("rect", { ...shape, class: `pv-ghost${s.when ? " pv-cond" : ""}` }, layer);
          if (!plays.has(s.id)) continue;
          const fill = snap ? shape.width : Math.max(0, Math.min(shape.width, x(t) - shape.x));
          if (fill > 0) el("rect", { ...shape, width: fill, class: `pv-fill pv-${s.role}` }, layer);
          const text = labels[s.id] || (s.arm ? (s.arm === "answer" ? "from the answer" : "fallback") : s.label);
          // A label inside the bar shows once the fill has passed it.
          const inside = text && text.length * 5.6 < shape.width - 8;
          const ready = inside ? fill >= (shape.width + text.length * 5.6) / 2 : t >= s.a;
          if (text && (snap || ready)) label(s.lane, shape.x, shape.x + shape.width, text, "pv-on-bar");
        }
        // A snapshot's own marks show while it stands still.
        const marks = snap && snap.marks ? (frame.f === 0 ? snap.marks : []) : scene.marks || [];
        for (const m of marks) {
          const y = rows[m.lanes[0]];
          if (m.role === "sync") {
            const ys = m.lanes.map((id) => rows[id]);
            el("path", { d: `M ${x(m.t)} ${Math.min(...ys) + BAR - 2} V ${Math.max(...ys) + BAR + BAR_H + 2}`, class: "pv-sync" }, layer);
          } else if (m.role === "due") {
            el("path", { d: `M ${x(m.t)} ${y + BAR - 3} V ${y + LANE + 1}`, class: "pv-due" }, layer);
            const near = x(m.t) > W - PAD - m.label.length * 3;
            el("text", { x: near ? W - PAD : x(m.t), y: y + LANE + 11, "text-anchor": near ? "end" : "middle", class: "pv-small pv-classical-text" }, layer, m.label);
          } else if (m.role === "frame" && (snap || t >= m.t)) {
            // A frame update takes no time: a diamond on the lane's baseline.
            const cx = x(m.t);
            const cy = y + BAR + BAR_H;
            el("path", { d: `M ${cx} ${cy - 5} l 5 5 l -5 5 l -5 -5 z`, class: "pv-frame" }, layer);
            el("text", { x: cx + 7, y: cy - 3, class: "pv-small pv-drive-text pv-halo" }, layer, m.label);
          }
        }
        // Messages: an arrow from the sending lane to the receiving one once
        // it has left.
        for (const m of inCase(scene, scene.messages, frame.c)) {
          if (snap || t < m.a) continue;
          const y0 = rows[m.from] + BAR + BAR_H / 2;
          const y1 = rows[m.to] + BAR + BAR_H / 2;
          const reach = covered(m.a, m.b, t);
          const xb = x(m.a + (m.b - m.a) * reach);
          el("path", { d: `M ${x(m.a)} ${y0} L ${xb} ${y0 + (y1 - y0) * reach}`, class: `pv-arrow pv-arrow-${m.role}` }, layer);
        }
        layer.appendChild(above);
        tl.head.setAttribute("d", snap || frame.done ? "" : `M ${x(t)} ${tl.axisY + 4} V ${tl.bottom}`);
      },
    },

    // Small multiples: every case, or every snapshot, in a row of its own,
    // with its share of shots and the conditional spans it plays.
    cases: {
      build(ctx, y) {
        const { el, scene } = ctx;
        const g = el("g", { class: "pv-cases" }, ctx.svg);
        const rows = scene.cases || scene.snapshots;
        const cond = (scene.spans || []).filter((s) => s.when);
        const ROW = 20;
        el("text", { x: PAD, y: y + 14, class: "pv-lane" }, g, scene.cases ? "every case" : "every step");
        const cellX = LABEL_W + 210;
        const barX = cellX + cond.length * 52 + 8;
        const barW = W - PAD - barX - 60;
        const bands = [];
        rows.forEach((row, c) => {
          const yy = y + 24 + c * ROW;
          bands.push(el("rect", { x: LABEL_W - 4, y: yy - 2, width: W - PAD - LABEL_W + 4, height: ROW - 2, rx: 3, class: "pv-case-band" }, g));
          el("text", { x: LABEL_W, y: yy + 11, class: "pv-small" }, g, row.short || row.label);
          const plays = scene.cases ? inCase(scene, cond, c) : [];
          cond.forEach((s, i) => {
            const on = plays.includes(s);
            const box = { x: cellX + i * 52, y: yy + 2, width: 44, height: ROW - 8, rx: 2 };
            el("rect", { ...box, class: on ? `pv-fill pv-${s.role}` : "pv-ghost pv-cond" }, g);
            el("text", { x: box.x + 22, y: yy + 11, "text-anchor": "middle", class: `pv-tiny${on ? " pv-on-text" : ""}` }, g, s.label || s.role);
          });
          if (row.weight === undefined) return;
          el("rect", { x: barX, y: yy + 3, width: barW, height: ROW - 10, class: "pv-track" }, g);
          el("rect", { x: barX, y: yy + 3, width: Math.max(1, barW * row.weight), height: ROW - 10, class: "pv-share" }, g);
          el("text", { x: barX + barW + 6, y: yy + 11, class: "pv-small" }, g, row.value || pct(row.weight));
        });
        el("text", { x: barX, y: y + 14, class: "pv-small" }, g, scene.share_label || "share of shots");
        ctx.caseRows = bands;
        return 24 + rows.length * ROW + 4;
      },
      draw(ctx, frame) {
        const current = ctx.scene.cases ? frame.c : snapshotIndex(ctx.scene, frame);
        ctx.caseRows.forEach((band, c) => band.setAttribute("class", `pv-case-band${c === current ? " pv-current" : ""}`));
      },
    },

    // A tree of branches that grows one readout at a time: a node shows from
    // the snapshot its "step" names, and greys out from its "drop". Its edge
    // is as wide as its probability.
    tree: {
      build(ctx, y) {
        const { el, scene } = ctx;
        const tree = scene.tree;
        const g = el("g", { class: "pv-tree" }, ctx.svg);
        const h = 44 + tree.height * 30;
        const cols = Math.max(...tree.nodes.map((n) => n.col));
        const colW = (W - LABEL_W - PAD - 200) / cols;
        const at = {};
        for (const n of tree.nodes) at[n.id] = [LABEL_W + n.col * colW, y + 40 + n.y * (h - 54)];
        tree.columns.forEach((text, c) => {
          el("text", { x: LABEL_W + c * colW, y: y + 12, class: "pv-small", "text-anchor": c ? "middle" : "end" }, g, text);
        });
        ctx.tree = { at, layer: el("g", {}, g) };
        return h;
      },
      draw(ctx, frame) {
        const { el, scene } = ctx;
        const { at, layer } = ctx.tree;
        layer.replaceChildren();
        const base = frame.step;
        const shown = snapshotIndex(scene, frame);
        for (const n of scene.tree.nodes) {
          const grow = n.step <= base ? 1 : n.step === base + 1 ? ease(frame.f) : 0;
          if (grow <= 0) continue;
          const dropped = n.drop !== undefined && n.drop <= shown;
          const cls = dropped ? " pv-dropped" : "";
          const [x1, y1] = at[n.id];
          if (n.parent !== undefined) {
            const [x0, y0] = at[n.parent];
            const mid = (x0 + x1) / 2;
            el("path", {
              d: `M ${x0} ${y0} C ${mid} ${y0} ${mid} ${y1} ${x1} ${y1}`,
              class: `pv-branch${cls}`,
              "stroke-width": 1 + 10 * n.p,
              pathLength: grow < 1 ? 1 : null,
              "stroke-dasharray": grow < 1 ? `${grow} 1` : null,
            }, layer);
          }
          if (grow < 1) continue;
          el("circle", { cx: x1, cy: y1, r: 3.5, class: `pv-dot${cls}` }, layer);
          // The root's label sits left of it, a node with branches of its
          // own above it, and a leaf's right of it.
          const root = n.parent === undefined;
          const inner = !root && scene.tree.nodes.some((m) => m.parent === n.id);
          const text = dropped ? `${n.text}, dropped` : n.text;
          const where = root ? [x1 - 8, y1 + 4, "end"] : inner ? [x1, y1 - 12 - 5 * n.p, "middle"] : [x1 + 8, y1 + 4, "start"];
          el("text", { x: where[0], y: where[1], "text-anchor": where[2], class: `${dropped ? "pv-small" : "pv-node-text"} pv-halo` }, layer, text);
        }
      },
    },
  };

  let serial = 0;

  // Builds the figure into *svg*; returns draw(frame), which updates it.
  function build(svg, scene, doc) {
    const el = maker(doc);
    // One time scale for every panel that has a time axis.
    const span = drawnSpan(scene);
    const x = (t) => LABEL_W + (visual(scene, t) / span) * (W - LABEL_W - PAD);
    const ctx = { el, svg, scene, x, W, PAD, LABEL_W, pill, pct };
    const hatch = `pvh${serial++}`;
    const pattern = el("pattern", { id: hatch, width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, el("defs", {}, svg));
    el("rect", { width: 2, height: 5, class: "pv-hatch" }, pattern);
    ctx.hatch = `url(#${hatch})`;
    ctx.header = el("text", { x: PAD, y: 18, class: "pv-title" }, svg, "");
    const panels = scene.panels || ["timeline"];
    let y = 28;
    for (const p of panels) y += PANELS[p].build(ctx, y);
    svg.setAttribute("viewBox", `0 0 ${W} ${y + 4}`);
    return (frame) => {
      ctx.header.textContent = headerAt(scene, frame);
      const spans = spansAt(scene, frame);
      for (const p of panels) PANELS[p].draw(ctx, frame, spans);
      return captionAt(scene, frame);
    };
  }

  api.build = build;
  api.PANELS = PANELS;
  if (typeof module === "object" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  // ── Playback ───────────────────────────────────────────────────────────

  let count = 0;

  function button(text, onclick) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "pv-button";
    b.textContent = text;
    b.onclick = onclick;
    return b;
  }

  function mount(root) {
    const data = JSON.parse(document.getElementById(root.dataset.source).textContent);
    const scene = data.scenes.find((s) => s.name === root.dataset.scene);
    const id = `pv${count++}`;
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("role", "img");
    svg.setAttribute("class", "pv-svg");
    svg.setAttribute("aria-labelledby", `${id}-title ${id}-desc`);
    const el = maker(document);
    el("title", { id: `${id}-title` }, svg, scene.title);
    el("desc", { id: `${id}-desc` }, svg, description(scene));
    const caption = document.createElement("p");
    caption.className = "pv-caption";
    const bar = document.createElement("div");
    bar.className = "pv-bar";
    root.replaceChildren(svg, caption, bar);
    const draw = build(svg, scene, document);
    const figure = root.closest("figure");
    const fallback = figure && figure.querySelector(".pulse-fallback");
    if (fallback) fallback.hidden = true;

    const show = (frame) => {
      caption.textContent = draw(frame);
    };
    const list = stops(scene);
    let at = list.length - 1;
    let raf = 0;
    const stop = () => {
      cancelAnimationFrame(raf);
      raf = 0;
    };
    const step = (by) => {
      stop();
      at = (at + by + list.length) % list.length;
      show(list[at]);
    };
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let started = 0;
    const tick = (now) => {
      if (!started) started = now;
      const frame = frameAt(scene, now - started);
      show(frame);
      raf = frame.done ? 0 : requestAnimationFrame(tick);
    };
    const play = () => {
      stop();
      started = 0;
      raf = requestAnimationFrame(tick);
    };

    if (scene.snapshots) {
      // One button per snapshot; a click jumps there without moving.
      scene.snapshots.forEach((s, k) => bar.append(button(s.short || s.label, () => {
        at = k;
        stop();
        show(list[k]);
      })));
    } else {
      bar.append(button("Previous", () => step(-1)), button("Next", () => step(1)));
    }
    // A scene with autoplay false waits for the reader.
    const autoplay = !still && scene.autoplay !== false;
    if (autoplay) bar.prepend(button("Replay", play));
    root.tabIndex = 0;
    root.setAttribute("aria-label", `${scene.title}. Arrow keys step through it.`);
    root.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") step(1);
      else if (e.key === "ArrowLeft") step(-1);
      else return;
      e.preventDefault();
    });

    // A step-through starts on its first snapshot; reduced motion shows the
    // still frame, and the buttons step; otherwise it plays once when in view.
    const rest = restFrame(scene);
    if (!autoplay && !still) {
      at = 0;
      show(list[0]);
    } else if (still) {
      at = list.findIndex((f) => f.c === rest.c && f.t === rest.t && f.step === rest.step);
      show(rest);
    } else {
      show(frameAt(scene, 0));
      const seen = new IntersectionObserver(
        (entries) => {
          if (entries.some((e) => e.isIntersecting)) {
            seen.disconnect();
            play();
          }
        },
        { threshold: 0.5 }
      );
      seen.observe(root);
      root.pulseVisualSeen = seen;
    }
    document.addEventListener("visibilitychange", () => {
      if (document.hidden && raf) {
        stop();
        show(rest);
      }
    });
    // For screenshots and tests: stop, and draw one frame.
    root.pulseVisual = {
      show(frame) {
        if (root.pulseVisualSeen) root.pulseVisualSeen.disconnect();
        stop();
        show({ c: 0, t: endOf(scene), step: 0, f: 0, ...frame });
      },
      at(ms) {
        this.show(frameAt(scene, ms));
      },
      total: totalMs(scene),
      scene,
    };
  }

  function start() {
    document.querySelectorAll(".pulse-visual").forEach(mount);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
