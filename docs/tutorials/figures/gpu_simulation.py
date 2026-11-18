# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "Simulating on a GPU" tutorial.

Usage: python docs/tutorials/figures/gpu_simulation.py

Needs matplotlib (the ``[viz]`` extra). Writes, each with its ``-dark`` twin,
next to this file:

- ``pair_schedule.svg``, the schedule of the page's ``evolve_pair``;
- ``pair_density.svg``, the density matrix of its CPU simulation;
- ``lowering``, the kernel at each level the compiler lowers it to, with the
  operations of each, and its export for a controller, one stage at a time:
  a scene of ``gpu_simulation.json``, whose still ``tools/render-pulse-visual.js``
  draws.

Runs on any machine; nothing here needs a GPU.
"""

import re
from collections import Counter

import numpy as np
from _style import page_namespace, save_themed
from _timeline import Timeline, draw
from _visual import write

_OP = re.compile(r"^\s*(?:%[^=]+=\s*)?\"?([a-z_]+)\.([a-z_.]+)", re.M)
_LEVELS = (("pulse", "Pulse MLIR"), ("qop", "QOp"), ("cudm", "cuDensityMat"), ("llvm", "LLVM IR"))


def pair_schedule(page) -> None:
    names = {page["x1"].index: "X1", page["d2"].index: "D2"}
    timeline = Timeline.of(page["evolve_pair"].lower(page["x1"], page["d2"]), names=names)
    drives = sorted(timeline.pulses, key=lambda pl: pl.end_ns - pl.start_ns)
    words = {(drives[0].lane, round(drives[0].start_ns, 3)): "π/2"}
    words |= {(pl.lane, round(pl.start_ns, 3)): "π" for pl in drives[1:]}
    slot = next(w for w in timeline.waits)
    alt = (
        "Timeline of evolve_pair on the drive lines of X1 and D2: a pi/2 pulse on X1 and a pi "
        f"pulse on D2, a sync, then the CZ's {slot.length_ns:g} ns slot as a wait on both lines, "
        f"to {slot.end_ns:g} ns."
    )
    save_themed(
        lambda p: draw(timeline, p, pulse_labels=words, syncs="all", xlim=(0, slot.end_ns * 1.02)),
        "pair_schedule",
        title="The coupled pair's program",
        desc=alt,
    )


def pair_density(page) -> None:
    rho = np.abs(np.asarray(page["cpu"].density_matrix))
    dims = page["cpu"].dims
    labels = [f"{a}{b}" for a in range(dims[0]) for b in range(dims[1])]
    big = [(labels[k], rho[k, k]) for k in range(len(labels)) if rho[k, k] > 0.05]

    def figure(p):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        cmap = LinearSegmentedColormap.from_list("theme", [p.surface, p.drive], N=64)
        # Cells and colorbar as vector paths: an embedded PNG's bytes differ
        # between platforms, so the figure would not compare equal on all.
        edges = np.arange(len(labels) + 1) - 0.5
        image = ax.pcolormesh(
            edges, edges, rho, cmap=cmap, vmin=0, vmax=max(0.5, float(rho.max())), edgecolors="face"
        )
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(edges[-1], edges[0])
        ax.set_aspect("equal")
        ax.set_xticks(range(len(labels)), labels, fontsize=7.5)
        ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("X1 D2 levels")
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7.5, length=0)
        bar.set_label("|ρ|", color=p.muted)
        return fig

    alt = (
        f"The magnitudes of the {len(labels)} by {len(labels)} density matrix of the pair after "
        "the CPU simulation, rows and columns labelled by the levels of X1 then D2. Almost all "
        "of it sits in "
        + " and ".join(f"{name} ({value:.2f})" for name, value in big)
        + ", with the coherence between them off the diagonal; levels 2 stay empty."
    )
    save_themed(figure, "pair_density", title="The pair's 9 × 9 density matrix", desc=alt)


_SAY = {
    "pulse": "The kernel as written: pulses, waits and syncs on its lines.",
    "qop": "QOp writes the pair's Hamiltonian and the Lindblad terms of its decay.",
    "cudm": "The cuDensityMat dialect is the cuDensityMat API as IR.",
    "llvm": "LLVM IR turns each of its operations into a call into the cuDensityMat runtime.",
}


def lowering(page) -> dict:
    """``evolve_pair`` at each level of ``mlir()``, one stage at a time, with
    the operations each holds counted from its text, then its controller export."""
    scheduled = page["evolve_pair"].lower(page["x1"], page["d2"])
    stages, snapshots = [], []
    for k, (level, name) in enumerate(_LEVELS):
        ops = Counter(f"{d}.{o}" for d, o in _OP.findall(scheduled.mlir(level)))
        dialects = Counter()
        for op, n in ops.items():
            dialects[op.split(".")[0]] += n
        total = sum(ops.values())
        if level == "llvm":
            count = f"{ops['llvm.call']} llvm.call"
            detail = f"{total} operations"
            said = f"{total} operations, {ops['llvm.call']} of them calls into cuDensityMat"
        else:
            count = f"{total} operations"
            detail = "  ".join(f"{d} {n}" for d, n in dialects.most_common(2))
            said = f"{total} operations"
        stage = {
            "title": name,
            "sub": f'mlir("{level}")',
            "count": count,
            "detail": detail,
            "role": "drive" if k == 0 else "q",
            "step": k,
        }
        if k:
            stage["from"] = k - 1
        stages.append(stage)
        snapshots.append(
            {
                "label": f"{name}: {said}",
                "short": name,
                "say": f"{_SAY[level]} It holds {said}.",
                "spans": {},
            }
        )
    events = Counter(e["kind"] for e in scheduled.export_to_controller()["events"])
    total = sum(events.values())
    stages.append(
        {
            "title": "controller export",
            "sub": "export_to_controller()",
            "count": f"{total} events",
            "detail": "  ".join(f"{kind} {n}" for kind, n in events.most_common(3)),
            "role": "link",
            "step": len(_LEVELS),
            "from": 0,
            "row": 1,
            "verb": "export",
            "note": "the program a control system plays, from the same Pulse MLIR",
        }
    )
    snapshots.append(
        {
            "label": f"Controller export: {total} events",
            "short": "export",
            "say": f"From the same Pulse MLIR, the program exports for a controller as {total} "
            "events. Only the last steps differ.",
            "spans": {},
        }
    )
    return {
        "name": "lowering",
        "title": "One program, lowered for the GPU and exported",
        "stages": stages,
        "snapshots": snapshots,
        "panels": ["stages"],
    }


def main() -> None:
    page = page_namespace("gpu_simulation.rst")
    pair_schedule(page)
    pair_density(page)
    write("gpu_simulation", "docs/tutorials/gpu_simulation.rst", [lowering(page)])


if __name__ == "__main__":
    main()
