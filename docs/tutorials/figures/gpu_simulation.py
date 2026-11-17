# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "Simulating on a GPU" tutorial.

Usage: python docs/tutorials/figures/gpu_simulation.py

Needs matplotlib (the ``[viz]`` extra). Writes ``lowering.svg`` and
``lowering-dark.svg`` next to this file: the page's ``evolve_pair`` at each
level the compiler lowers it to, with the operations of each, and its export
for a controller. Runs on any machine; nothing here needs a GPU.
"""

import re
from collections import Counter

from _style import page_namespace, plot_axes, save_themed
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_OP = re.compile(r"^\s*(?:%[^=]+=\s*)?\"?([a-z_]+)\.([a-z_.]+)", re.M)
_LEVELS = (("pulse", "Pulse MLIR"), ("qop", "QOp"), ("cudm", "cuDensityMat"), ("llvm", "LLVM IR"))


def main() -> None:
    page = page_namespace("gpu_simulation.rst")
    scheduled = page["evolve_pair"].lower(page["x1"], page["d2"])
    stages = []
    for level, name in _LEVELS:
        ops = Counter(f"{d}.{o}" for d, o in _OP.findall(scheduled.mlir(level)))
        dialects = Counter()
        for op, n in ops.items():
            dialects[op.split(".")[0]] += n
        stages.append((name, sum(ops.values()), dialects, ops))
    events = Counter(e["kind"] for e in scheduled.export_to_controller()["events"])

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.9)
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4.4)
        width, gap = 2.15, 0.45

        def box(x, top, title, count, lines, color):
            ax.add_patch(
                FancyBboxPatch(
                    (x, top - 1.25),
                    width,
                    1.25,
                    boxstyle="round,pad=0,rounding_size=0.08",
                    fc=p.surface,
                    ec=color,
                    lw=1.3,
                )
            )
            ax.text(
                x + 0.12,
                top - 0.18,
                title,
                ha="left",
                va="top",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )
            ax.text(x + 0.12, top - 0.5, count, ha="left", va="top", fontsize=8.5, color=p.text)
            ax.text(
                x + 0.12,
                top - 0.8,
                lines,
                ha="left",
                va="top",
                fontsize=7.5,
                color=p.muted,
                linespacing=1.3,
            )

        def arrow(a, b):
            ax.add_patch(
                FancyArrowPatch(
                    a,
                    b,
                    arrowstyle="-|>,head_length=4,head_width=2.4",
                    color=p.rule,
                    lw=1.1,
                    shrinkA=2,
                    shrinkB=2,
                )
            )

        for k, (name, total, dialects, ops) in enumerate(stages):
            x = 0.1 + k * (width + gap) * 0.98
            top = dialects.most_common(2)
            detail = "  ".join(f"{d} {n}" for d, n in top)
            if name == "LLVM IR":
                count = f"{ops['llvm.call']} llvm.call"
                detail = f"{total} operations"
            else:
                count = f"{total} operations"
            box(x, 4.3, name, count, detail, p.drive if k == 0 else p.q)
            if k:
                arrow((x - gap * 0.98 + 0.02, 3.68), (x - 0.02, 3.68))
        export = "  ".join(f"{kind} {n}" for kind, n in events.most_common(3))
        box(0.1, 1.45, "controller export", f"{sum(events.values())} events", export, p.classical)
        arrow((0.1 + width / 2, 3.05), (0.1 + width / 2, 1.5))
        ax.text(
            0.1 + width + 0.3,
            0.9,
            "the program a control system plays,\nfrom the same Pulse MLIR",
            ha="left",
            va="center",
            fontsize=8,
            color=p.muted,
        )
        ax.text(
            10,
            2.65,
            "what cuDensityMat runs on the GPU",
            ha="right",
            va="top",
            fontsize=8,
            color=p.muted,
        )
        return fig

    parts = [f"{name}, {total} operations" for name, total, *_ in stages]
    alt = (
        "The page's evolve_pair kernel at each level of its lowering for GPU simulation: "
        + "; ".join(parts)
        + f", of them {stages[-1][3]['llvm.call']} calls into the cuDensityMat runtime. From the "
        f"Pulse MLIR the same program also exports for a controller as "
        f"{sum(events.values())} events."
    )
    save_themed(figure, "lowering", title="One program, lowered for the GPU and exported", desc=alt)


if __name__ == "__main__":
    main()
