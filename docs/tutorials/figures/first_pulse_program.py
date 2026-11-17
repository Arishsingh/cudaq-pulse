# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the "Your first pulse program" tutorial.

Usage: python docs/tutorials/figures/first_pulse_program.py

Needs matplotlib (the ``[viz]`` extra). Writes ``rabi_sweep.svg`` and
``rabi_schedule.svg`` next to this file, from the kernel and the sweep the
tutorial runs.
"""

import cudaq_pulse as pulse
import matplotlib.pyplot as plt
import numpy as np
from _style import BLUE, GREEN, GREY, save


@pulse.kernel
def rabi(qudit, amplitude: float):
    d = get_drive_line(qudit)
    r = get_readout_line(qudit)
    x = gaussian(duration=32 * pulse.ns, amplitude=amplitude, sigma=8 * pulse.ns)
    drive(d, x)
    sync(d, r)
    return readout(r, square(300 * pulse.ns, 0.05))


def main() -> None:
    qs = pulse.quantum_system("transmon_surface17")
    q = qs.qudits[0]

    amplitudes = np.linspace(0, 0.4, 41)
    populations = [pulse.simulate(rabi, q, a).level_population(1) for a in amplitudes]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(amplitudes, populations, "o-", color=BLUE, markersize=4, linewidth=1.5)
    ax.axvline(0.16, color=GREEN, linestyle="--", linewidth=1.2)
    ax.annotate("π pulse", (0.16, 0.5), xytext=(6, 0), textcoords="offset points", color=GREY)
    ax.set_xlabel("drive amplitude")
    ax.set_ylabel("excited-state population")
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "rabi_sweep.svg")

    fig = rabi.lower(q, 0.16).timeline(figsize=(8, 2.2))
    save(fig, "rabi_schedule.svg")


if __name__ == "__main__":
    main()
