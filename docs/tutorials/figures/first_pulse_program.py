# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Your first pulse program" tutorial.

Usage: python docs/tutorials/figures/first_pulse_program.py

Needs matplotlib (the ``[viz]`` extra). Writes ``rabi_sweep.svg`` and
``rabi_schedule.svg`` next to this file, from the kernel and the sweep the
tutorial runs.
"""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cudaq_pulse as pulse
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_BLUE = "#4a7fb5"
_GREEN = "#76b900"
_GREY = "#6b6b6b"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "svg.fonttype": "none",
        "svg.hashsalt": "cudaq-pulse",
    }
)


@pulse.kernel
def rabi(qudit, amplitude: float):
    d = get_drive_line(qudit)
    r = get_readout_line(qudit)
    x = gaussian(duration=32 * pulse.ns, amplitude=amplitude, sigma=8 * pulse.ns)
    drive(d, x)
    sync(d, r)
    return readout(r, square(300 * pulse.ns, 0.05))


def _save(fig, name: str) -> None:
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    plt.close(fig)
    # The repository's pre-commit hooks reject trailing whitespace.
    text = "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n"
    (_HERE / name).write_text(text)


def main() -> None:
    qs = pulse.quantum_system("transmon_surface17")
    q = qs.qudits[0]

    amplitudes = np.linspace(0, 0.4, 41)
    populations = [pulse.simulate(rabi, q, a).level_population(1) for a in amplitudes]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(amplitudes, populations, "o-", color=_BLUE, markersize=4, linewidth=1.5)
    ax.axvline(0.16, color=_GREEN, linestyle="--", linewidth=1.2)
    ax.annotate("π pulse", (0.16, 0.5), xytext=(6, 0), textcoords="offset points", color=_GREY)
    ax.set_xlabel("drive amplitude")
    ax.set_ylabel("excited-state population")
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "rabi_sweep.svg")

    fig = rabi.lower(q, 0.16).timeline(figsize=(8, 2.2))
    _save(fig, "rabi_schedule.svg")


if __name__ == "__main__":
    main()
