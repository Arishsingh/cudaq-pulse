# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Coherence: T1, Ramsey and Echo" tutorial.

Usage: python docs/tutorials/figures/coherence.py

Needs matplotlib (the ``[viz]`` extra). Writes ``t1_decay.svg`` and
``ramsey_echo.svg`` next to this file, from the kernels the page defines.
"""

import cudaq_pulse as pulse
import numpy as np
from _style import BLUE, GREEN, GREY, axes, page_namespace, save


def main() -> None:
    page = page_namespace("coherence.rst")
    q, d1 = page["q"], page["d1"]

    waits = np.linspace(0, 60_000, 25)
    p1 = [pulse.simulate(page["t1"], q, w).level_population(1) for w in waits]
    p_th = d1.thermal_population
    model = p_th + (p1[0] - p_th) * np.exp(-waits / (d1.t1_us * 1e3))
    fig, ax = axes()
    ax.plot(waits / 1e3, p1, "o", color=BLUE, markersize=4, label="simulated")
    ax.plot(waits / 1e3, model, "-", color=GREEN, linewidth=1.3, label=f"T1 = {d1.t1_us} µs")
    ax.axhline(p_th, color=GREY, linestyle=":", linewidth=1)
    ax.annotate(
        "thermal population", (45, p_th), xytext=(0, 4), textcoords="offset points", color=GREY
    )
    ax.set_xlabel("wait (µs)")
    ax.set_ylabel("excited-state population")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    save(fig, "t1_decay.svg")

    taus = np.linspace(0, 4_000, 161)
    ramsey = [pulse.simulate(page["ramsey"], q, t).level_population(1) for t in taus]
    echo = [pulse.simulate(page["echo"], q, t).level_population(1) for t in taus[::4]]
    fig, ax = axes()
    ax.plot(taus / 1e3, ramsey, "-", color=BLUE, linewidth=1.5, label="Ramsey")
    ax.plot(taus[::4] / 1e3, echo, "o-", color=GREEN, markersize=3, linewidth=1.2, label="echo")
    ax.set_xlabel("total wait (µs)")
    ax.set_ylabel("excited-state population")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "ramsey_echo.svg")


if __name__ == "__main__":
    main()
