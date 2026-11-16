# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the waveform figure of the operations guide.

Usage: python docs/user_guide/figures/waveform_envelopes.py

Needs matplotlib (the ``[viz]`` extra). Writes ``waveform_envelopes.svg``
next to this file: the I and Q of each parametric waveform at the waveform
explorer's defaults, compiled and sampled with
``cudaq_pulse.controller.sample_waveform``. The page shows it where the
explorer does not run; tests/test_waveform_explorer.py checks that its
calls are the explorer's defaults.
"""

import sys
from pathlib import Path

import cudaq_pulse as pulse
import matplotlib.pyplot as plt
import numpy as np
from cudaq_pulse.controller import sample_waveform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tutorials" / "figures"))
from _style import BLUE, GREEN, save  # noqa: E402

CALLS = [
    "gaussian(40, 0.3, 10)",
    "drag(40, 0.3, 10, 0.5)",
    "cosine(40, 0.3)",
    "tanh_ramp(40, 0.3, 4)",
    "gaussian_square(40, 0.3, 4, 20)",
    "square(40, 0.3)",
]


def envelope(call: str, qs: pulse.QuantumSystem) -> np.ndarray:
    """The samples ``drive(d, <call>)`` plays, one per clock cycle."""
    namespace: dict = {}
    source = f"def play(q):\n    drive(get_drive_line(q), {call})\n"
    exec(compile(source, "<figure>", "exec"), namespace)
    program = pulse.compile(pulse.kernel(namespace["play"]), quantum_system=qs)
    (event,) = program.lower(qs.qudits[0]).export_to_controller()["events"]
    return sample_waveform(event["waveform"], event["duration_cycles"])


def main() -> None:
    qs = pulse.quantum_system("transmon_surface17")
    fig, grid = plt.subplots(2, 3, figsize=(9.6, 4.6), sharex=True, sharey=True)
    for ax, call in zip(grid.flat, CALLS, strict=True):
        samples = envelope(call, qs)
        t = np.arange(len(samples))
        ax.plot(t, samples.real, "-", color=GREEN, linewidth=1.6, label="I (real)")
        ax.plot(t, samples.imag, "--", color=BLUE, linewidth=1.4, label="Q (imag)")
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.set_title(call, fontsize=10, family="monospace")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in grid[1]:
        ax.set_xlabel(f"time (clock cycles at {qs.clock_ghz:g} GHz)")
    for ax in grid[:, 0]:
        ax.set_ylabel("amplitude")
    grid[1, 2].legend(frameon=False, loc="center", fontsize=9)
    fig.tight_layout()
    save(fig, "waveform_envelopes.svg", Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
