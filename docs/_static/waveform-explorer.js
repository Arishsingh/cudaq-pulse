// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The waveform explorer of the operations guide: a slider and a number field
// for each argument of each built-in envelope, its I and Q drawn as SVG, and
// the call that makes it.
// The envelopes are evaluateEnvelope's (core/mlir/include/cudaq-pulse/Runtime/
// WaveformCallback.h), sampled at t = 0, 1, 2, ... clock cycles as
// cudaq_pulse.controller.sample_waveform samples them. The modulated drive is
// E(t) = envelope(t) exp(-i theta(t)) of the Conventions page.
// tests/test_waveform_explorer.py runs this file under Node, then compiles and
// samples every code line it prints with the library and compares.
"use strict";

(function () {
  // The program clock of transmon_surface17, which the docs compile for.
  const CLOCK_GHZ = 2;

  // Each constructor's arguments after the duration; square's amplitude_imag
  // is the imaginary part of its one amplitude argument.
  const SHAPES = {
    gaussian: ["amplitude", "sigma"],
    drag: ["amplitude", "sigma", "beta"],
    cosine: ["amplitude"],
    tanh_ramp: ["amplitude", "sigma"],
    gaussian_square: ["amplitude", "sigma", "width"],
    square: ["amplitude", "amplitude_imag"],
  };
  const DEFAULTS = {
    gaussian: { amplitude: 0.3, sigma: 10 },
    drag: { amplitude: 0.3, sigma: 10, beta: 0.5 },
    cosine: { amplitude: 0.3 },
    tanh_ramp: { amplitude: 0.3, sigma: 4 },
    gaussian_square: { amplitude: 0.3, sigma: 4, width: 20 },
    square: { amplitude: 0.3, amplitude_imag: 0 },
  };
  // [min, max, step]; times in clock cycles. width also stays below the
  // duration by a positive even number of cycles.
  const RANGES = {
    duration: [8, 400, 2],
    amplitude: [-1, 1, 0.01],
    amplitude_imag: [-1, 1, 0.01],
    sigma: [1, 50, 0.5],
    beta: [-4, 4, 0.01],
    width: [0, 398, 2],
    factor: [-2, 2, 0.05],
    detuning_mhz: [-250, 250, 5],
    phase: [-3.14, 3.14, 0.01],
    anharmonicity_mhz: [-400, -100, 1],
  };
  const TIMES = new Set(["duration", "sigma", "beta", "width"]);
  const COMBINE = ["none", "wf_scale", "wf_neg", "wf_add", "wf_sub", "wf_mul"];
  const BINARY = new Set(["wf_add", "wf_sub", "wf_mul"]);

  function defaultState() {
    const copy = () => JSON.parse(JSON.stringify(DEFAULTS));
    return {
      duration: 40,
      a: { shape: "drag", params: copy() },
      combine: "none",
      factor: 0.5,
      b: { shape: "gaussian", params: copy() },
      modulate: false,
      detuning_mhz: 100,
      phase: 0,
      anharmonicity_mhz: -200,
    };
  }

  // The arguments a waveform plays with: width at most duration - 2.
  function args(state, which) {
    const { shape, params } = state[which];
    const p = { duration: state.duration, ...params[shape] };
    if (shape === "gaussian_square") p.width = Math.max(0, Math.min(p.width, p.duration - 2));
    return p;
  }

  // evaluateEnvelope at localTime t (cycles): [re, im], zero outside [0, d).
  function envelope(shape, p, t) {
    const d = p.duration;
    if (!(t >= 0 && t < d && d > 0)) return [0, 0];
    const centered = t - d / 2;
    const a = p.amplitude;
    switch (shape) {
      case "square":
        return [a, p.amplitude_imag];
      case "gaussian":
        return [a * Math.exp((-0.5 * centered * centered) / (p.sigma * p.sigma)), 0];
      case "drag": {
        const re = a * Math.exp((-0.5 * centered * centered) / (p.sigma * p.sigma));
        return [re, (-p.beta * centered * re) / (p.sigma * p.sigma)];
      }
      case "cosine":
        return [a * 0.5 * (1 - Math.cos((2 * Math.PI * t) / d)), 0];
      case "tanh_ramp":
        return [a * 0.5 * (1 + Math.tanh((t - d / 2) / p.sigma)), 0];
      case "gaussian_square": {
        // The IR's risefall, which the frontend computes from width.
        const edge = (d - p.width) / 2;
        let x = 0;
        if (t < edge) x = t - edge;
        else if (t >= d - edge) x = t - (d - edge);
        else return [a, 0];
        return [a * Math.exp((-0.5 * x * x) / (p.sigma * p.sigma)), 0];
      }
    }
    throw new Error(`unknown shape ${shape}`);
  }

  function sampleOne(state, which) {
    const p = args(state, which);
    const re = [];
    const im = [];
    for (let k = 0; k < p.duration; k++) {
      const [r, i] = envelope(state[which].shape, p, k);
      re.push(r);
      im.push(i);
    }
    return { re, im };
  }

  // The envelope, after wf_* if any: sample_waveform's arithmetic on samples.
  function envelopeSamples(state) {
    const a = sampleOne(state, "a");
    const n = a.re.length;
    if (state.combine === "none") return a;
    if (state.combine === "wf_scale") {
      const f = state.factor;
      return { re: a.re.map((x) => x * f), im: a.im.map((y) => y * f) };
    }
    if (state.combine === "wf_neg") return { re: a.re.map((x) => -x), im: a.im.map((y) => -y) };
    const b = sampleOne(state, "b");
    const re = new Array(n);
    const im = new Array(n);
    for (let k = 0; k < n; k++) {
      if (state.combine === "wf_add") {
        re[k] = a.re[k] + b.re[k];
        im[k] = a.im[k] + b.im[k];
      } else if (state.combine === "wf_sub") {
        re[k] = a.re[k] - b.re[k];
        im[k] = a.im[k] - b.im[k];
      } else {
        re[k] = a.re[k] * b.re[k] - a.im[k] * b.im[k];
        im[k] = a.re[k] * b.im[k] + a.im[k] * b.re[k];
      }
    }
    return { re, im };
  }

  function detuningHz(state) {
    return Number(`${fmt(state.detuning_mhz)}e6`);
  }

  // What the qudit sees in its frame: E(t) = envelope(t) exp(-i theta(t)),
  // theta(t) = phase + 2 pi (f_tone - f_qudit) t, t = k cycles from the start.
  function samples(state) {
    const env = envelopeSamples(state);
    if (!state.modulate) return env;
    const hz = detuningHz(state);
    const re = [];
    const im = [];
    env.re.forEach((x, k) => {
      const angle = state.phase + (2 * Math.PI * hz * k) / (CLOCK_GHZ * 1e9);
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      re.push(x * c + env.im[k] * s);
      im.push(env.im[k] * c - x * s);
    });
    return { re, im };
  }

  // A number as Python reads it back: the shortest string that round-trips.
  function fmt(x) {
    return String(x);
  }

  function call(state, which) {
    const shape = state[which].shape;
    const p = args(state, which);
    if (shape === "square") {
      const im = p.amplitude_imag;
      const amplitude =
        im === 0 ? fmt(p.amplitude) : `${fmt(p.amplitude)}${im < 0 ? "-" : "+"}${fmt(Math.abs(im))}j`;
      return `square(${fmt(p.duration)}, ${amplitude})`;
    }
    return `${shape}(${["duration", ...SHAPES[shape]].map((n) => fmt(p[n])).join(", ")})`;
  }

  function waveformCode(state) {
    const a = call(state, "a");
    if (state.combine === "none") return a;
    if (state.combine === "wf_scale") return `wf_scale(${a}, ${fmt(state.factor)})`;
    if (state.combine === "wf_neg") return `wf_neg(${a})`;
    return `${state.combine}(${a}, ${call(state, "b")})`;
  }

  // The code the plot shows: the waveform, or with modulation the kernel
  // lines that shift the tone and play it.
  function codeLines(state) {
    const wf = waveformCode(state);
    if (!state.modulate) return [wf];
    const lines = ["d = get_drive_line(q)"];
    if (state.phase !== 0) lines.push(`shift_phase(d.tone, ${fmt(state.phase)})`);
    lines.push(`shift_frequency(d.tone, ${fmt(state.detuning_mhz)}e6)`);
    lines.push(`drive(d, ${wf})`);
    return lines;
  }

  // drag's documented optimum: beta = -1 / alpha, alpha the anharmonicity
  // in radians per clock cycle (2 pi f_alpha / f_clock).
  function suggestedBeta(anharmonicityMhz) {
    return -1 / ((2 * Math.PI * anharmonicityMhz * 1e6) / (CLOCK_GHZ * 1e9));
  }

  // A typed value as its slider would give it: within [min, max], at the
  // nearest step from min, with the step's decimals. The code line then
  // prints only values the slider can reach.
  function snap(value, min, max, step) {
    const digits = (String(step).split(".")[1] || "").length;
    const steps = Math.round((Math.min(max, Math.max(min, value)) - min) / step);
    return +(min + steps * step).toFixed(digits);
  }

  // Where a curve is largest in size: that sample with its sign, the first
  // cycle that has it, and whether the next cycle has it too (a flat top).
  function peak(values) {
    let at = 0;
    values.forEach((v, k) => {
      if (Math.abs(v) > Math.abs(values[at])) at = k;
    });
    return { value: values[at], cycle: at, flat: Math.abs(values[at + 1]) === Math.abs(values[at]) };
  }

  // Every shape at its defaults and with each slider at both ends, each
  // combination, and the modulated drive: what the drift test checks.
  function cases() {
    const out = [];
    const make = (edit) => {
      const s = defaultState();
      edit(s);
      out.push(s);
    };
    for (const shape of Object.keys(SHAPES)) {
      make((s) => (s.a.shape = shape));
      for (const end of [0, 1]) {
        make((s) => ((s.a.shape = shape), (s.duration = RANGES.duration[end])));
        for (const name of SHAPES[shape]) {
          make((s) => ((s.a.shape = shape), (s.a.params[shape][name] = RANGES[name][end])));
        }
      }
    }
    make((s) => ((s.a.shape = "gaussian_square"), (s.duration = 8), (s.a.params.gaussian_square.width = 0)));
    for (const op of COMBINE.slice(1)) {
      const shapes = BINARY.has(op) ? Object.keys(SHAPES) : ["drag"];
      for (const shape of shapes) make((s) => ((s.combine = op), (s.b.shape = shape)));
    }
    for (const end of [0, 1]) make((s) => ((s.combine = "wf_scale"), (s.factor = RANGES.factor[end])));
    const square = (w) => ((w.shape = "square"), (w.params.square.amplitude_imag = -0.4));
    make((s) => ((s.combine = "wf_mul"), square(s.b)));
    make((s) => (s.modulate = true));
    for (const end of [0, 1]) {
      make((s) => ((s.modulate = true), (s.detuning_mhz = RANGES.detuning_mhz[end])));
      make((s) => ((s.modulate = true), (s.phase = RANGES.phase[end])));
      make((s) => ((s.modulate = true), (s.duration = RANGES.duration[end]), (s.phase = 0.5)));
    }
    make((s) => ((s.modulate = true), (s.detuning_mhz = 0), square(s.a)));
    make((s) => ((s.modulate = true), (s.combine = "wf_add"), (s.phase = -1.2)));
    return out;
  }

  const api = { CLOCK_GHZ, SHAPES, RANGES, COMBINE, defaultState, cases, codeLines, samples, suggestedBeta, snap, peak };
  if (typeof module === "object" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  // ── The widget ─────────────────────────────────────────────────────────

  const SVG_NS = "http://www.w3.org/2000/svg";
  // The plot is drawn in CSS pixels, as wide as its box, so that its 12px
  // text is 12px on a phone too. W is its width until the box is measured,
  // and MIN_W the narrowest it is drawn before it scales down instead.
  const W = 640;
  const MIN_W = 200;
  const H = 260;
  const PAD = { left: 62, right: 12, top: 10, bottom: 34 };
  const LABELS = {
    duration: "duration",
    amplitude: "amplitude",
    amplitude_imag: "amplitude (imag)",
    sigma: "sigma",
    beta: "beta",
    width: "width",
    factor: "factor",
    detuning_mhz: "detuning (tone \u2212 qudit)",
    phase: "phase",
    anharmonicity_mhz: "anharmonicity",
  };

  function el(tag, attrs, ...kids) {
    const svg = ["svg", "g", "line", "polyline", "text"].includes(tag);
    const e = svg ? document.createElementNS(SVG_NS, tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v !== false) e.setAttribute(k, v);
    e.append(...kids);
    return e;
  }

  function ns(cycles) {
    return +(cycles / CLOCK_GHZ).toPrecision(6);
  }

  // The unit after a value; a time also in ns at the program clock.
  function unit(name, v) {
    if (TIMES.has(name)) return `cycles (${ns(v)} ns)`;
    if (name.endsWith("_mhz")) return "MHz";
    if (name === "phase") return "rad";
    return "";
  }

  function shown(name, v) {
    return `${fmt(v)} ${unit(name, v)}`.trim();
  }

  // A curve's peak() in words: "Peak I: 0.3 at cycle 20 (10 ns)."
  function peakText(name, p) {
    if (p.value === 0) return `${name} is 0.`;
    return `Peak ${name}: ${+p.value.toPrecision(3)} ${p.flat ? "from" : "at"} cycle ${p.cycle} (${ns(p.cycle)} ns).`;
  }

  // The y range: the curve's peak with 10% headroom, rounded up to a 1-1.2-
  // 1.5-2-2.5-3-4-5-6-8 step, and at least 0.01 so a flat zero still has an
  // axis. I and Q share it, so a small Q stays small beside I.
  function niceMax(peak) {
    const target = Math.max(peak * 1.1, 0.01);
    const decade = 10 ** Math.floor(Math.log10(target));
    for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (target <= m * decade * (1 + 1e-9)) return +(m * decade).toPrecision(3);
    return 10 * decade;
  }

  // The tick step along the time axis: at most `most` steps across the span.
  function niceStep(span, most) {
    for (const s of [2, 5, 10, 20, 25, 50, 100]) if (span / s <= most) return s;
    return 200;
  }

  let count = 0;

  function mount(root) {
    const id = `pwe${count++}`;
    const state = defaultState();
    if (SHAPES[root.dataset.shape]) state.a.shape = root.dataset.shape;

    // One control per grid cell: its label, the control, and its value.
    function field(label, control, value) {
      return el("div", { class: "pwe-field" }, label, control, value || el("span", {}));
    }

    // A slider and a number field for one value, under one label. The slider
    // says the value with its unit; the unit beside the number field
    // describes the field, and is hidden as text of its own so that a row is
    // not read with its unit three times. A typed number counts when Enter
    // or leaving the field commits it, so a half-typed one changes nothing.
    function slider(name, get, set, after, prefix = "") {
      const [min, max, step] = RANGES[name];
      const base = `${id}-${name}-${Math.random().toString(36).slice(2, 7)}`;
      const label = el("label", { id: `${base}-label`, for: `${base}-number` }, prefix + LABELS[name]);
      const range = el("input", { type: "range", min, max, step, "aria-labelledby": `${base}-label` });
      const number = el("input", { type: "number", id: `${base}-number`, min, max, step, "aria-describedby": `${base}-unit` });
      const suffix = el("span", { id: `${base}-unit`, "aria-hidden": "true" });
      const sync = () => {
        const v = get();
        range.value = number.value = fmt(v);
        suffix.textContent = unit(name, v);
        range.setAttribute("aria-valuetext", shown(name, v));
      };
      const changed = (v) => {
        set(v);
        sync();
        (after || draw)();
      };
      range.addEventListener("input", () => changed(+range.value));
      number.addEventListener("change", () => {
        if (number.value === "") sync();
        else changed(snap(+number.value, min, +number.max, step));
      });
      sync();
      return { node: field(label, range, el("span", { class: "pwe-entry" }, number, suffix)), range, number, sync };
    }

    function select(label, options, get, set) {
      const selectId = `${id}-${label.replace(/\W+/g, "-")}`;
      const s = el("select", { id: selectId }, ...options.map((o) => el("option", { value: o }, o)));
      s.value = get();
      s.addEventListener("change", () => {
        set(s.value);
        layout();
      });
      return field(el("label", { for: selectId }, label), s);
    }

    // Selects and the toggle stay in place, so they keep the keyboard focus.
    // The sliders of a shape are built when the shape is chosen and updated
    // in place after that: a value typed into one field and committed with
    // Tab must not remove the control the focus moves to. Groups lay their
    // controls out in the one grid of the explorer.
    const group = () => el("div", { class: "pwe-group" });
    const paramsA = group();
    const paramsB = group();
    const betaHelp = group();
    const factor = group();
    const modSliders = group();
    const built = {};

    function shapeSliders(which, box) {
      const w = state[which];
      const params = w.params[w.shape];
      if (!built[which] || built[which].shape !== w.shape) {
        const prefix = which === "b" ? "second " : "";
        const sliders = SHAPES[w.shape].map((name) => [
          name,
          slider(name, () => params[name], (v) => (params[name] = v), undefined, prefix),
        ]);
        box.replaceChildren(...sliders.map(([, s]) => s.node));
        built[which] = { shape: w.shape, sliders };
      }
      for (const [name, s] of built[which].sliders) {
        if (name === "width") {
          params.width = args(state, which).width;
          s.range.max = s.number.max = state.duration - 2;
        }
        s.sync();
      }
    }

    // The rule behind the button is on the page for every reader, under the
    // button, and is the button's description.
    const suggestion = el("span", { class: "pwe-value" });
    const hint =
      "beta = -1 / alpha, for the anharmonicity alpha in radians per clock cycle, " +
      "minimizes leakage to the second excited level; it matters most for short pulses.";
    const hintId = `${id}-beta-hint`;
    const useBeta = el("button", { type: "button", class: "pwe-button", "aria-describedby": hintId }, "Use suggested beta");
    useBeta.addEventListener("click", () => {
      const beta = +suggestedBeta(state.anharmonicity_mhz).toFixed(2);
      state.a.params.drag.beta = Math.max(RANGES.beta[0], Math.min(RANGES.beta[1], beta));
      layout();
    });
    const updateSuggestion = () => {
      const beta = suggestedBeta(state.anharmonicity_mhz);
      suggestion.textContent = `${beta.toFixed(2)} cycles (${ns(beta).toFixed(2)} ns)`;
    };
    betaHelp.append(
      slider("anharmonicity_mhz", () => state.anharmonicity_mhz, (v) => (state.anharmonicity_mhz = v), updateSuggestion).node,
      field(el("span", { class: "pwe-label" }, "suggested beta"), useBeta, suggestion),
      el("p", { id: hintId, class: "pwe-note" }, hint)
    );
    factor.append(slider("factor", () => state.factor, (v) => (state.factor = v)).node);
    modSliders.append(
      slider("detuning_mhz", () => state.detuning_mhz, (v) => (state.detuning_mhz = v)).node,
      slider("phase", () => state.phase, (v) => (state.phase = v)).node
    );
    const second = select("second waveform", Object.keys(SHAPES), () => state.b.shape, (v) => (state.b.shape = v));
    const toggleId = `${id}-modulate`;
    const toggle = el("input", { type: "checkbox", id: toggleId });
    toggle.addEventListener("change", () => {
      state.modulate = toggle.checked;
      layout();
    });

    function layout() {
      shapeSliders("a", paramsA);
      const binary = BINARY.has(state.combine);
      second.hidden = paramsB.hidden = !binary;
      if (binary) shapeSliders("b", paramsB);
      factor.hidden = state.combine !== "wf_scale";
      betaHelp.hidden = state.a.shape !== "drag";
      updateSuggestion();
      modSliders.hidden = !state.modulate;
      draw();
    }

    const iLine = el("polyline", { class: "pwe-i" });
    const qLine = el("polyline", { class: "pwe-q" });
    const upper = el("polyline", { class: "pwe-env" });
    const lower = el("polyline", { class: "pwe-env" });
    const axes = el("g", { class: "pwe-axes" });
    // The plot is named by an attribute: a <title> element is also given as
    // the plot's description, so its sentence would be read twice.
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, axes, upper, lower, iLine, qLine);
    let width = W;
    const code = el("code", {});
    // Where the curves peak, in sight under the code. The plot's name ends
    // with the same sentence, so a screen reader is given it there only.
    const facts = el("p", { class: "pwe-note", "aria-hidden": "true" });
    const units = el("p", { class: "pwe-note" });
    const status = el("span", { class: "pwe-note", "aria-live": "polite" });
    const spoken = el("p", { class: "pwe-hidden", role: "status" });
    const copy = el("button", { type: "button", class: "pwe-button" }, "Copy");
    copy.addEventListener("click", () => {
      navigator.clipboard.writeText(code.textContent).then(() => (status.textContent = "Copied"));
    });
    if (!(navigator.clipboard && window.isSecureContext)) copy.hidden = true;
    const keyI = el("span", { class: "pwe-key pwe-key-i" }, state.modulate ? "Re E" : "I (real)");
    const keyQ = el("span", { class: "pwe-key pwe-key-q" }, "Q (imag)");
    const keyEnv = el("span", { class: "pwe-key pwe-key-env" }, "|envelope|");

    // Draws the state and returns it in a sentence: the code, then where
    // each curve peaks.
    function render() {
      const lines = codeLines(state);
      code.textContent = lines.join("\n");
      const times = [["duration", state.duration]];
      for (const which of BINARY.has(state.combine) ? ["a", "b"] : ["a"]) {
        const p = args(state, which);
        for (const n of SHAPES[state[which].shape]) if (TIMES.has(n)) times.push([n, p[n]]);
      }
      units.textContent =
        `Times are clock cycles at the ${CLOCK_GHZ} GHz program clock: ` +
        times.map(([n, v]) => `${n} ${fmt(v)} = ${ns(v)} ns`).join(", ") +
        ". One sample per cycle, at t = 0, 1, 2, ... cycles.";

      const { re, im } = samples(state);
      const env = envelopeSamples(state);
      const mag = env.re.map((x, k) => Math.hypot(x, env.im[k]));
      const yMax = niceMax(Math.max(1e-9, ...re.map(Math.abs), ...im.map(Math.abs), ...mag));
      const n = re.length;
      const right = width - PAD.right;
      const x = (t) => PAD.left + (t / state.duration) * (right - PAD.left);
      const y = (v) => PAD.top + ((yMax - v) / (2 * yMax)) * (H - PAD.top - PAD.bottom);
      const pts = (vs) => vs.map((v, k) => `${x(k).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      svg.setAttribute("viewBox", `0 0 ${width} ${H}`);
      iLine.setAttribute("points", pts(re));
      qLine.setAttribute("points", pts(im));
      upper.setAttribute("points", state.modulate ? pts(mag) : "");
      lower.setAttribute("points", state.modulate ? pts(mag.map((v) => -v)) : "");
      keyI.textContent = state.modulate ? "Re E(t)" : "I (real)";
      keyQ.textContent = state.modulate ? "Im E(t)" : "Q (imag)";
      keyEnv.hidden = !state.modulate;
      const peaks = `${peakText(state.modulate ? "Re E(t)" : "I", peak(re))} ${peakText(state.modulate ? "Im E(t)" : "Q", peak(im))}`;
      facts.textContent = peaks;
      svg.setAttribute("aria-label", `${state.modulate ? "Modulated drive" : "I and Q"} of ${lines[lines.length - 1]}, ${n} samples. ${peaks}`);

      const parts = [
        el("line", { x1: PAD.left, x2: right, y1: y(0), y2: y(0), class: "pwe-zero" }),
        el("line", { x1: PAD.left, x2: PAD.left, y1: y(yMax), y2: y(-yMax) }),
        el("line", { x1: PAD.left, x2: right, y1: y(-yMax), y2: y(-yMax) }),
      ];
      for (const v of [-yMax, -yMax / 2, 0, yMax / 2, yMax]) {
        if (v !== 0) parts.push(el("line", { x1: PAD.left, x2: right, y1: y(v), y2: y(v), class: "pwe-gridline" }));
        parts.push(el("line", { x1: PAD.left - 4, x2: PAD.left, y1: y(v), y2: y(v) }));
        parts.push(el("text", { x: PAD.left - 7, y: y(v) + 4, "text-anchor": "end" }, String(+v.toPrecision(3))));
      }
      const mid = (y(yMax) + y(-yMax)) / 2;
      parts.push(el("text", { x: 12, y: mid, "text-anchor": "middle", transform: `rotate(-90 12 ${mid})` }, "amplitude"));
      // A narrow plot has fewer time ticks, 36px apart or more, and the
      // short axis label; the last note of the explorer gives the times in ns.
      const span = right - PAD.left;
      const step = niceStep(state.duration, Math.min(8, Math.floor(span / 36)));
      for (let t = 0; t <= state.duration; t += step) {
        parts.push(el("line", { x1: x(t), x2: x(t), y1: y(-yMax), y2: y(-yMax) + 4 }));
        parts.push(el("text", { x: x(t), y: y(-yMax) + 16, "text-anchor": "middle" }, String(t)));
      }
      const cycle = span >= 210 ? `; 1 cycle = ${ns(1)} ns` : "";
      parts.push(el("text", { x: (PAD.left + right) / 2, y: H - 4, "text-anchor": "middle" }, `time (clock cycles${cycle})`));
      axes.replaceChildren(...parts);
      return `${lines.join("; ")}. ${peaks}`;
    }

    // A change is said once, 400 ms after the last one: the slider and the
    // number field say their own value at each step, and the code and the
    // peaks follow when the reader pauses.
    let pending = 0;
    function draw() {
      const sentence = render();
      status.textContent = "";
      spoken.textContent = "";
      clearTimeout(pending);
      pending = setTimeout(() => (spoken.textContent = sentence), 400);
    }

    root.setAttribute("role", "group");
    root.setAttribute("aria-label", "Waveform explorer");
    root.replaceChildren(
      el(
        "div",
        { class: "pwe-controls" },
        select("waveform", Object.keys(SHAPES), () => state.a.shape, (v) => (state.a.shape = v)),
        slider("duration", () => state.duration, (v) => (state.duration = v), layout).node,
        paramsA,
        betaHelp,
        select("combine", COMBINE, () => state.combine, (v) => (state.combine = v)),
        second,
        factor,
        paramsB,
        field(el("label", { for: toggleId }, "modulate"), toggle, el("span", { class: "pwe-value" }, "the drive in the qudit's frame")),
        modSliders
      ),
      svg,
      el("div", { class: "pwe-legend" }, keyI, keyQ, keyEnv),
      el("div", { class: "pwe-code" }, el("pre", {}, code), el("div", {}, copy, status)),
      facts,
      units,
      spoken
    );
    layout();
    // Nothing is said until the reader changes something.
    clearTimeout(pending);

    // The plot follows the width of its box. Its height is fixed, so that a
    // redraw never changes the box it was measured from.
    if (typeof ResizeObserver === "function") {
      svg.style.height = `${H}px`;
      new ResizeObserver(([entry]) => {
        const measured = Math.max(MIN_W, Math.floor(entry.contentRect.width));
        if (entry.contentRect.width > 0 && measured !== width) {
          width = measured;
          render();
        }
      }).observe(svg);
    }
  }

  function start() {
    const roots = document.querySelectorAll(".pulse-waveform-explorer");
    roots.forEach(mount);
    if (roots.length) document.querySelectorAll(".pwe-fallback").forEach((f) => (f.hidden = true));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
