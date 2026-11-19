// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The toy controllers of the ISA tutorials, playing one compiled program once
// per outcome. docs/tutorials/figures/isa_limits.py writes the data from the
// controller export and toy_isa: each listing, when each bit is ready, each
// decoder call, every instruction the emulated controllers issue, and the
// sentence for each moment. This file computes no schedule; it places what the
// data says at the cycle the playhead is on. The figure plays once by itself
// while it is in the window; its buttons pause it and step through its
// moments, and a status line says which step is on screen. Readers who ask
// for reduced motion get a still frame and the step buttons.
// tests/test_toy_flow.py runs the functions this file exports under Node,
// checks them against the data, and presses the buttons on a stand-in page.
"use strict";

(function () {
  // Each outcome plays for SHOT_MS, holds its last frame for HOLD_MS, then the
  // next one starts.
  const SHOT_MS = 5200;
  const HOLD_MS = 1600;

  // The cycles the timeline shows: the program, rounded up to a tick.
  function span(scene) {
    return Math.ceil((scene.end_cycles + 8) / 40) * 40;
  }

  // Which outcome is playing, and at what cycle, *elapsed* ms after the start:
  // the playhead runs from cycle 0 to the program's last.
  function frameAt(scene, elapsed) {
    const each = SHOT_MS + HOLD_MS;
    const shot = Math.min(scene.shots.length - 1, Math.floor(Math.max(0, elapsed) / each));
    const local = Math.max(0, elapsed) - shot * each;
    const t = Math.min(1, local / SHOT_MS) * scene.end_cycles;
    return { shot, t, done: elapsed >= scene.shots.length * each };
  }

  // The ms after the start at which outcome *shot* is at cycle t: where
  // playing picks up from a frame.
  function elapsedAt(scene, shot, t) {
    return shot * (SHOT_MS + HOLD_MS) + (t / scene.end_cycles) * SHOT_MS;
  }

  // The frames a reader steps through, as [outcome, cycle]: each outcome's
  // first frame, each of its moments, then its last frame.
  function steps(scene) {
    const out = [];
    scene.shots.forEach((shot, s) => {
      const cycles = new Set([0, ...shot.moments.map((m) => m[0]), scene.end_cycles]);
      for (const cycle of [...cycles].sort((a, b) => a - b)) out.push([s, cycle]);
    });
    return out;
  }

  // The step a frame belongs to: the last one at or before it.
  function stepAt(scene, shot, t) {
    let at = 0;
    steps(scene).forEach(([s, cycle], k) => {
      if (s < shot || (s === shot && cycle <= t)) at = k;
    });
    return at;
  }

  function clamp01(f) {
    return Math.max(0, Math.min(1, f));
  }

  // Each controller's place in its listing at cycle t: the last instruction it
  // has issued, and the ones it went past without issuing (a skipped arm).
  // Once it has issued its last instruction, every one it never issued is skipped.
  function listingAt(scene, shot, t) {
    return scene.controllers.map((c) => {
      let counter = -1;
      let last = 0;
      const issued = new Set();
      for (const [controller, cycle, k] of shot.issued) {
        if (controller !== c.id) continue;
        last = Math.max(last, cycle);
        if (cycle > t) continue;
        counter = k;
        issued.add(k);
      }
      const passed = t >= last ? c.listing.length : counter;
      const skipped = [];
      for (let k = 0; k < passed; k++) {
        if (!issued.has(k) && !c.listing[k].endsWith(":")) skipped.push(k);
      }
      return { controller: c.id, counter, skipped };
    });
  }

  function livePulses(shot, t) {
    return shot.pulses.filter((p) => p.start_cycles <= t && t < p.start_cycles + p.duration_cycles);
  }

  function valueOf(shot, bit) {
    return bit in shot.outcomes ? shot.outcomes[bit] : null;
  }

  // Each bit at cycle t: held for its feedback latency, ready on its own
  // controller, crossing the bus to a peer that branches on it, or arrived.
  function bitsAt(scene, shot, t) {
    const out = [];
    for (const bit of scene.bits) {
      if (t < bit.ends_cycles) continue;
      const value = t >= bit.ready_cycles ? valueOf(shot, bit.name) : null;
      const entry = { name: bit.name, value, stage: "held", f: 0 };
      if (t < bit.ready_cycles) {
        entry.f = clamp01((t - bit.ends_cycles) / (bit.ready_cycles - bit.ends_cycles));
      } else if (bit.peers.length) {
        const peer = bit.peers[0];
        entry.to = peer.controller;
        entry.f = clamp01((t - bit.ready_cycles) / (peer.ready_cycles - bit.ready_cycles));
        entry.stage = t < peer.ready_cycles ? "crossing" : "arrived";
      } else {
        entry.stage = "ready";
      }
      out.push(entry);
    }
    return out;
  }

  // Each decoder call at cycle t. The answer may come back any time between
  // the request leaving and the deadline; the poll takes what has come.
  function callsAt(scene, shot, t) {
    return scene.calls.map((call) => {
      const inTime = valueOf(shot, `${call.answer}.ok`) === 1;
      const state = { call, inTime, stage: "idle", f: 0, taken: null };
      if (t >= call.taken_cycles) {
        state.stage = "taken";
        state.taken = inTime ? `${call.answer} = ${valueOf(shot, call.answer)}` : `${call.answer}.ok = 0`;
      } else if (t >= call.deadline_cycles) {
        state.stage = "deadline";
      } else if (t >= call.sent_cycles) {
        state.stage = "window";
        state.f = clamp01((t - call.sent_cycles) / (call.deadline_cycles - call.sent_cycles));
      } else if (t >= call.submit_cycles) {
        state.stage = "sending";
      }
      return state;
    });
  }

  function captionAt(shot, t) {
    let text = "";
    for (const [cycle, line] of shot.moments) if (cycle <= t) text = line;
    return text;
  }

  // What the status line says at step k: where it is, the outcome the step
  // belongs to, and the sentence of its moment. The picture shows the outcome
  // in its header, so the line shows the other two and speaks all three.
  function stepText(scene, k) {
    const list = steps(scene);
    const [s, cycle] = list[k];
    const shot = scene.shots[s];
    return {
      step: `Step ${k + 1} of ${list.length}`,
      outcome: shot.label,
      says: captionAt(shot, cycle),
    };
  }

  function stateAt(scene, shotIndex, t) {
    const shot = scene.shots[shotIndex];
    return {
      listings: listingAt(scene, shot, t),
      live: livePulses(shot, t),
      bits: bitsAt(scene, shot, t),
      calls: callsAt(scene, shot, t),
      caption: captionAt(shot, t),
    };
  }

  // The timeline's lanes: each controller's channels, then the outcome bus and
  // each link after the first controller's. A lane's spans are what any outcome
  // plays on it; a pulse only some outcomes play is conditional.
  function lanes(scene) {
    const out = [];
    for (const c of scene.controllers) {
      for (const ch of c.channels) {
        const seen = new Map();
        for (const shot of scene.shots) {
          for (const p of shot.pulses) {
            if (p.controller !== c.id || p.channel !== ch.name) continue;
            const key = p.start_cycles;
            const s = seen.get(key) || {
              a: p.start_cycles,
              b: p.start_cycles + p.duration_cycles,
              role: p.kind,
              shots: 0,
            };
            s.shots += 1;
            seen.set(key, s);
          }
        }
        const spans = [...seen.values()];
        for (const s of spans) s.conditional = s.shots < scene.shots.length;
        out.push({
          kind: "channel",
          controller: c.id,
          channel: ch.name,
          label: `C${c.id} ${ch.role} q${ch.qudit}`,
          spans,
        });
      }
      if (c.id !== scene.controllers[0].id) continue;
      const bus = [];
      const said = new Set();
      const add = (a, b, label) => {
        if (said.has(`${a}/${b}`)) return;
        said.add(`${a}/${b}`);
        bus.push({ a, b, role: "classical", label });
      };
      for (const bit of scene.bits) {
        add(bit.ends_cycles, bit.ready_cycles, `feedback ${bit.ready_cycles - bit.ends_cycles}`);
        for (const peer of bit.peers)
          add(bit.ready_cycles, peer.ready_cycles, `peer ${peer.ready_cycles - bit.ready_cycles}`);
      }
      out.push({ kind: "bus", label: `bus ${scene.bits.map((b) => b.name).join(" ")}`, spans: bus });
      for (const call of scene.calls) {
        out.push({
          kind: "link",
          call,
          label: `link L${call.link} ${call.name}`,
          spans: [
            { a: call.submit_cycles, b: call.sent_cycles, role: "classical", label: "submit", side: "left" },
            {
              a: call.sent_cycles,
              b: call.deadline_cycles,
              role: "window",
              label: `answer window ${call.deadline_cycles - call.sent_cycles}`,
            },
            {
              a: call.deadline_cycles,
              b: call.taken_cycles,
              role: "classical",
              label: `back ${call.taken_cycles - call.deadline_cycles}`,
              side: "right",
            },
          ],
        });
      }
    }
    return out;
  }

  // What a screen reader hears: each outcome, then each moment of it.
  function description(scene) {
    const parts = [`${scene.title}, once per outcome.`];
    for (const shot of scene.shots) parts.push(shot.label, ...shot.moments.map((m) => m[1]));
    return parts.join(" ");
  }

  const api = {
    SHOT_MS,
    HOLD_MS,
    span,
    frameAt,
    elapsedAt,
    steps,
    stepAt,
    stepText,
    stateAt,
    lanes,
    description,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  // ── Drawing ────────────────────────────────────────────────────────────

  const SVG_NS = "http://www.w3.org/2000/svg";
  const W = 760;
  const BOX_X = [16, 388];
  const BOX_W = 356;
  // A listing row holds a 12-unit line and leaves room for the outline of the
  // current row's band.
  const ROW = 16;
  const LANE = 22;
  const LABEL_W = 136;

  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  // A pill as wide as its text, which is monospace at 12 units.
  function pill(parent, x, y, text, cls) {
    const w = 10 + text.length * 7.5;
    const g = el("g", { class: `tf-pill ${cls || ""}` }, parent);
    el("rect", { x: x - w / 2, y: y - 8, width: w, height: 16, rx: 8 }, g);
    el("text", { x, y: y + 4, "text-anchor": "middle" }, g, text);
  }

  function build(svg, scene) {
    const hasCalls = scene.calls.length > 0;
    const rows = Math.min(8, Math.max(...scene.controllers.map((c) => c.listing.length)));
    const top = hasCalls ? 100 : 38;
    const bottom = top + 34 + rows * ROW;
    const busY = bottom + 22;
    const quditY = bottom + 58;
    const view = { rows, top, bottom, busY, quditY, boxes: {}, wires: [], qudits: [] };

    const header = el("text", { x: 16, y: 18, class: "tf-title" }, svg, "");
    const clock = el("text", { x: W - 16, y: 18, class: "tf-muted", "text-anchor": "end" }, svg, "");
    view.header = header;
    view.clock = clock;

    // The controllers, each with a window on its listing and wires to its qudits.
    for (const c of scene.controllers) {
      const x = BOX_X[c.id];
      const g = el("g", {}, svg);
      el("rect", { x, y: top, width: BOX_W, height: bottom - top, rx: 6, class: "tf-box" }, g);
      el("text", { x: x + 12, y: top + 18, class: "tf-title" }, g, `controller ${c.id}`);
      el("text", { x: x + BOX_W - 12, y: top + 18, class: "tf-muted", "text-anchor": "end" }, g, c.model);
      const mark = el("rect", { x: x + 8, width: BOX_W - 16, height: ROW, rx: 3, class: "tf-pc" }, g);
      const lines = [];
      for (let i = 0; i < rows; i++)
        lines.push(el("text", { x: x + 16, y: top + 41 + i * ROW, class: "tf-code" }, g, ""));
      view.boxes[c.id] = { x, mark, lines, first: 0 };

      const at = (q) => x + (BOX_W * (c.qudits.indexOf(q) + 1)) / (c.qudits.length + 1);
      for (const q of c.qudits) {
        const circle = el("circle", { cx: at(q), cy: quditY, r: 13, class: "tf-qudit" }, svg);
        el("text", { x: at(q), y: quditY + 4, class: "tf-small", "text-anchor": "middle" }, svg, `q${q}`);
        view.qudits.push({ controller: c.id, qudit: q, el: circle });
      }
      for (const ch of c.channels) {
        const wx = at(ch.qudit) + (ch.role === "readout" ? -6 : 6);
        const wire = el("path", { d: `M ${wx} ${bottom} V ${quditY - 13}`, class: "tf-wire" }, svg);
        const anchor = ch.role === "readout" ? "end" : "start";
        el(
          "text",
          {
            x: wx + (ch.role === "readout" ? -4 : 4),
            y: bottom + 12,
            class: "tf-small",
            "text-anchor": anchor,
          },
          svg,
          ch.name
        );
        view.wires.push({ controller: c.id, channel: ch.name, x: wx, el: wire });
      }
    }
    el("path", { d: `M 16 ${busY} H ${W - 16}`, class: "tf-bus" }, svg);
    el("text", { x: W - 16, y: busY - 5, class: "tf-small", "text-anchor": "end" }, svg, "outcome bus");

    // The decoder, and its link to each controller that sends to it or polls it.
    if (hasCalls) {
      const call = scene.calls[0];
      const dx = W / 2 - 100;
      const box = el("rect", { x: dx, y: 30, width: 200, height: 44, rx: 6, class: "tf-box" }, svg);
      el("text", { x: dx + 12, y: 47, class: "tf-title" }, svg, call.name);
      const status = el("text", { x: dx + 12, y: 64, class: "tf-muted" }, svg, "");
      const windowBar = el("rect", { x: dx + 1, y: 70, width: 0, height: 3, class: "tf-window" }, svg);
      const ends = new Set([call.controller, ...call.takers]);
      const links = [];
      for (const id of ends) {
        const cx = BOX_X[id] + BOX_W / 2;
        const from = id === scene.controllers[0].id ? dx + 30 : dx + 170;
        links.push(el("path", { d: `M ${from} 74 V 87 H ${cx} V ${top}`, class: "tf-link" }, svg));
      }
      el("text", { x: dx + 206, y: 56, class: "tf-small" }, svg, `link L${call.link}`);
      view.decoder = { box, status, windowBar, links, x: dx, w: 200 };
    }
    view.tokens = el("g", {}, svg);

    // The timeline: a lane per channel, the bus and each link.
    const total = span(scene);
    const x = (t) => LABEL_W + (t / total) * (W - LABEL_W - 16);
    const axisY = quditY + 40;
    const laneList = lanes(scene);
    const height = axisY + 8 + laneList.length * LANE + 6;
    for (let t = 0; t <= total; t += total > 400 ? 80 : 40) {
      el("path", { d: `M ${x(t)} ${axisY + 4} V ${height - 4}`, class: "tf-grid" }, svg);
      el("text", { x: x(t), y: axisY, class: "tf-small", "text-anchor": "middle" }, svg, String(t));
    }
    el("text", { x: LABEL_W - 8, y: axisY, class: "tf-small", "text-anchor": "end" }, svg, "cycle");
    view.fills = [];
    laneList.forEach((lane, i) => {
      const y = axisY + 8 + i * LANE;
      el("text", { x: LABEL_W - 8, y: y + 15, class: "tf-lane", "text-anchor": "end" }, svg, lane.label);
      for (const s of lane.spans) {
        const x0 = x(s.a);
        const w = Math.max(1.5, x(s.b) - x0);
        const pulse = s.role === "drive" || s.role === "readout";
        const box = pulse ? { y: y + 4, height: LANE - 6 } : { y: y + 15, height: 4 };
        let ghost = "tf-ghost";
        if (s.conditional) ghost += " tf-cond";
        if (s.role === "window") ghost += " tf-window-ghost";
        el("rect", { x: x0, width: w, rx: 1.5, class: ghost, ...box }, svg);
        const fill = el("rect", { x: x0, width: 0, rx: 1.5, class: `tf-fill tf-${s.role}`, ...box }, svg);
        if (s.label) {
          // A label too long for its span sits beside it. The one before its
          // span stands off far enough to read apart from the label of the
          // span after it, which is centered and reaches back toward it.
          let where = [x0 + w / 2, "middle"];
          if (s.side === "left") where = [x0 - 8, "end"];
          if (s.side === "right") where = [x0 + w + 3, "start"];
          const attrs = { x: where[0], y: y + 11, class: "tf-small tf-span", "text-anchor": where[1] };
          el("text", attrs, svg, s.label);
        }
        view.fills.push({ el: fill, lane, s, x0, w });
      }
    });
    view.head = el("path", { d: "", class: "tf-head" }, svg);
    view.x = x;
    view.axisY = axisY;
    view.height = height;
    svg.setAttribute("viewBox", `0 0 ${W} ${height}`);
    return view;
  }

  function draw(view, scene, shotIndex, t) {
    const shot = scene.shots[shotIndex];
    const now = stateAt(scene, shotIndex, t);
    view.header.textContent = shot.label;
    view.clock.textContent = `cycle ${Math.floor(t)} \u00b7 ${Math.floor(t / scene.clock_ghz)} ns`;

    for (const place of now.listings) {
      const c = scene.controllers.find((k) => k.id === place.controller);
      const box = view.boxes[c.id];
      const k = place.counter;
      // The window keeps two lines after the current one in view.
      const ahead = k + 2;
      if (k < box.first || t === 0) box.first = 0;
      if (ahead >= box.first + view.rows) {
        box.first = Math.max(0, Math.min(c.listing.length - view.rows, ahead - view.rows + 1));
      }
      const skipped = new Set(place.skipped);
      box.lines.forEach((line, i) => {
        const n = box.first + i;
        line.textContent = c.listing[n] || "";
        line.setAttribute("class", `tf-code${skipped.has(n) ? " tf-skipped" : ""}`);
      });
      const row = k - box.first;
      box.mark.setAttribute("y", view.top + 30 + row * ROW);
      box.mark.style.display = k >= 0 && row >= 0 && row < view.rows ? "" : "none";
    }

    for (const wire of view.wires) {
      const p = now.live.find((q) => q.controller === wire.controller && q.channel === wire.channel);
      wire.el.setAttribute("class", `tf-wire${p ? ` tf-on tf-${p.kind}` : ""}`);
    }
    for (const q of view.qudits) {
      const p = now.live.find((r) => r.controller === q.controller && r.qudit === q.qudit);
      q.el.setAttribute("class", `tf-qudit${p ? ` tf-on tf-${p.kind}` : ""}`);
    }

    // Bits leave from the wire of the readout that acquires them.
    view.tokens.replaceChildren();
    const calls = now.calls;
    const submitted = new Set();
    for (const c of calls) if (c.stage !== "idle") for (const b of c.call.bits) submitted.add(b);
    for (const bit of now.bits) {
      if (submitted.has(bit.name)) continue;
      const read = shot.pulses.find((p) => p.bit === bit.name);
      const wire = view.wires.find((w) => w.controller === read.controller && w.channel === read.channel);
      const text = bit.value === null ? bit.name : `${bit.name} = ${bit.value}`;
      if (bit.stage === "held") {
        const y = view.bottom + 6 + (view.busY - view.bottom - 6) * bit.f;
        pill(view.tokens, wire.x - 22, y, bit.name, "tf-pending");
        continue;
      }
      let x = wire.x - 22;
      if (bit.stage === "crossing" || bit.stage === "arrived") {
        x += (BOX_X[bit.to] + 40 - x) * bit.f;
      }
      pill(view.tokens, x, view.busY, text);
    }

    if (view.decoder) {
      const dec = view.decoder;
      const state = calls[0];
      const call = state.call;
      const words = {
        idle: "",
        sending: `receiving ${call.bits.join(" ")}`,
        window: `may answer until cycle ${call.deadline_cycles}`,
        deadline: state.inTime ? "answered in time" : `no answer by ${call.deadline_cycles}`,
        taken: state.inTime ? "answered in time" : `no answer by ${call.deadline_cycles}`,
      };
      dec.status.textContent = words[state.stage];
      const busy = state.stage === "sending" || state.stage === "window";
      dec.box.setAttribute("class", `tf-box${busy ? " tf-busy" : ""}`);
      let filled = 1;
      if (state.stage === "idle" || state.stage === "sending") filled = 0;
      if (state.stage === "window") filled = state.f;
      dec.windowBar.setAttribute("width", (dec.w - 2) * filled);
      for (const link of dec.links) link.setAttribute("class", `tf-link${busy ? " tf-on" : ""}`);
      if (state.stage === "sending" || state.stage === "window") {
        pill(view.tokens, dec.x + dec.w - 44, 47, call.bits.join(" "));
      }
      if (state.taken) {
        for (const id of call.takers)
          pill(view.tokens, BOX_X[id] + BOX_W / 2 + 46, view.top - 7, state.taken);
      }
    }

    // The timeline fills up to the playhead with what this outcome plays.
    for (const f of view.fills) {
      let played = true;
      if (f.lane.kind === "channel") {
        played = shot.pulses.some(
          (p) =>
            p.controller === f.lane.controller && p.channel === f.lane.channel && p.start_cycles === f.s.a
        );
      }
      const width = Math.max(0, Math.min(f.w, view.x(t) - f.x0));
      f.el.setAttribute("width", played ? width : 0);
    }
    view.head.setAttribute("d", `M ${view.x(t)} ${view.axisY + 4} V ${view.height - 4}`);
  }

  // ── Playback ───────────────────────────────────────────────────────────

  let count = 0;

  function button(text) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tf-button";
    b.textContent = text;
    return b;
  }

  function mount(root) {
    const data = JSON.parse(document.getElementById(root.dataset.source).textContent);
    const scene = data.scenes.find((s) => s.name === root.dataset.scene);
    const id = `ptf${count++}`;
    // The picture is named by its title alone. What it plays, moment by
    // moment, is its description: as part of the name it would be read out
    // whole wherever the picture is named.
    const svg = el("svg", {
      role: "img",
      "aria-labelledby": `${id}-title`,
      "aria-describedby": `${id}-desc`,
      class: "tf-svg",
    });
    el("title", { id: `${id}-title` }, svg, scene.title);
    el("desc", { id: `${id}-desc` }, svg, description(scene));
    const status = document.createElement("p");
    status.className = "pulse-toy-flow-status";
    status.setAttribute("role", "status");
    const toggle = button("Play");
    toggle.classList.add("tf-toggle");
    const back = button("Previous step");
    const forth = button("Next step");
    const bar = document.createElement("div");
    bar.className = "pulse-toy-flow-controls";
    bar.append(toggle, back, forth);
    root.setAttribute("role", "group");
    root.setAttribute("aria-label", scene.title);
    root.replaceChildren(svg, status, bar);
    const view = build(svg, scene);
    const fallback = root.parentElement.querySelector(".pulse-fallback");
    if (fallback) fallback.hidden = true;

    const list = steps(scene);
    const end = scene.end_cycles;
    const last = scene.shots.length - 1;
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)");
    // The frame on screen, and the step it belongs to.
    let at = { shot: 0, t: 0 };
    let step = -1;
    let raf = 0;
    let started = null;
    // It still plays by itself: until it has played to its end, the reader
    // has used a control, or a still frame has stood in for the playing.
    let owed = true;
    // Half of the figure, and half of its controls, are in the window.
    let half = false;
    let reach = false;

    // Draws one frame. The status line follows it, and speaks only when
    // *say* is true: when the reader asked for the frame. While the figure
    // plays, the line changes in silence.
    const show = (shot, t, say) => {
      at = { shot, t };
      draw(view, scene, shot, t);
      const k = stepAt(scene, shot, t);
      if (k === step && !say) return;
      step = k;
      const text = stepText(scene, k);
      const outcome = document.createElement("span");
      outcome.className = "visually-hidden";
      outcome.textContent = `${text.outcome} `;
      // A status is polite unless it is switched off.
      if (say) status.removeAttribute("aria-live");
      else status.setAttribute("aria-live", "off");
      status.replaceChildren(`${text.step}: `, outcome, text.says);
      // At either end a step button says it is unavailable, and keeps the
      // focus it may have: a disabled button would drop it.
      for (const [b, ended] of [[back, k === 0], [forth, k === list.length - 1]]) {
        if (ended) b.setAttribute("aria-disabled", "true");
        else b.removeAttribute("aria-disabled");
      }
    };
    const over = () => at.shot === last && at.t >= end;
    // The live wires' dashes run under tf-playing only, so only with frames.
    const sync = () => {
      root.classList.toggle("tf-playing", raf !== 0);
      toggle.textContent = raf ? "Pause" : over() ? "Replay" : "Play";
      toggle.hidden = calm.matches;
    };
    const stop = () => {
      cancelAnimationFrame(raf);
      raf = 0;
    };
    const frame = (time) => {
      // The first frame sets the clock: the frame to play from is on screen.
      // From an outcome's last frame the next outcome starts at once. The
      // hold between the two is for a figure that has just played up to it;
      // after a press of Play it would look as if nothing had happened.
      if (started === null) {
        const held = at.t >= end && at.shot < last;
        started = time - (held ? elapsedAt(scene, at.shot + 1, 0) : elapsedAt(scene, at.shot, at.t));
      } else {
        const next = frameAt(scene, time - started);
        show(next.shot, next.t, false);
      }
      raf = over() ? 0 : requestAnimationFrame(frame);
      if (!raf) {
        owed = false;
        sync();
      }
    };
    const play = () => {
      stop();
      started = null;
      status.setAttribute("aria-live", "off");
      raf = requestAnimationFrame(frame);
      sync();
    };
    const pause = (say) => {
      stop();
      if (say) show(at.shot, at.t, true);
      sync();
    };

    // The reader's controls. Using one ends the playing by itself for good;
    // a step button at its end does that and no more.
    const move = (by) => {
      const k = Math.max(0, Math.min(list.length - 1, step + by));
      owed = false;
      stop();
      if (k !== step) show(list[k][0], list[k][1], true);
      sync();
    };
    toggle.onclick = () => {
      owed = false;
      if (raf) {
        pause(true);
      } else {
        if (over()) show(0, 0, false);
        play();
      }
    };
    back.onclick = () => move(-1);
    forth.onclick = () => move(1);
    // Left and Right step while focus is on a control. With a modifier the
    // keys are the browser's: Alt+Left goes back a page.
    bar.addEventListener("keydown", (e) => {
      const by = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!by || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      e.preventDefault();
      move(by);
    });

    // It plays by itself once. It starts with half of it and half of its
    // controls in the window, so Pause is in reach. Any playing stops when
    // the figure leaves that half, or the tab is hidden; playing by itself
    // then goes on when the figure is back.
    const auto = (away) => {
      if (away) {
        if (raf) pause(false);
      } else if (owed && !raf && half && reach && !document.hidden && !calm.matches) {
        play();
      }
    };
    const watch = new IntersectionObserver(
      (entries) => {
        let left = false;
        for (const e of entries) {
          const most = e.intersectionRatio >= 0.5;
          if (e.target === bar) {
            reach = most;
          } else {
            left = half && !most;
            half = most;
          }
        }
        auto(left);
      },
      { threshold: 0.5 }
    );
    watch.observe(root);
    watch.observe(bar);
    document.addEventListener("visibilitychange", () => auto(document.hidden));

    // Reduced motion: no playing and no Play button. The figure rests on the
    // first outcome's last frame, which stands in for the playing, and the
    // step buttons show each moment without moving between them. The setting
    // may change while the page is open: turned on, it stills a figure that
    // plays or has yet to.
    const settle = () => {
      if (calm.matches && (raf || owed)) {
        stop();
        owed = false;
        show(0, end, false);
      }
      sync();
      auto(false);
    };
    if (calm.addEventListener) calm.addEventListener("change", settle);
    show(0, 0, false);
    settle();

    // For screenshots: stop, and draw one moment.
    root.toyFlowShow = (shotIndex, t) => {
      owed = false;
      stop();
      show(shotIndex, t, false);
      sync();
    };
  }

  function start() {
    document.querySelectorAll(".pulse-toy-flow").forEach(mount);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
