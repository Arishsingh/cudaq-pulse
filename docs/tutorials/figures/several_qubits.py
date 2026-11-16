# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "Several Qubits" tutorial.

Usage: python docs/tutorials/figures/several_qubits.py

Needs matplotlib (the ``[viz]`` extra). Writes ``zz_ramsey.svg`` next to
this file, from the Ramsey kernel the page defines.
"""

import cudaq_pulse as pulse
import numpy as np
from _style import BLUE, GREEN, axes, page_namespace, save


def main() -> None:
    page = page_namespace("several_qubits.rst")
    d1, x2, kernel = page["data"][0], page["x2"], page["ramsey_next_to"]
    taus = np.linspace(0, 20_000, 26)
    fig, ax = axes()
    for excite, color, label in ((0, BLUE, "X2 in its ground state"), (1, GREEN, "X2 excited")):
        p1 = [pulse.simulate(kernel, d1, x2, excite, t).level_populations(d1)[1] for t in taus]
        ax.plot(taus / 1e3, p1, "o-", color=color, markersize=3, linewidth=1.3, label=label)
    ax.set_xlabel("Ramsey wait (µs)")
    ax.set_ylabel("D1 excited-state population")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    save(fig, "zz_ramsey.svg")


if __name__ == "__main__":
    main()
