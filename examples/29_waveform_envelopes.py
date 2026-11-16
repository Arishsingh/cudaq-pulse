# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Waveform envelopes: the samples each built-in waveform plays.

Compiles one kernel that plays every built-in envelope, exports it for a
controller, and samples each drive's waveform with
``cudaq_pulse.controller.sample_waveform``: the complex envelope, one sample
per clock cycle, that the simulator and a controller play. It prints the
peak I and Q of each, and plots them all with matplotlib, from the viz
extra: pip install "cudaq-pulse[viz]".

The documentation's operations guide has an interactive version of the
plot, the waveform explorer.
"""

import math

import cudaq_pulse as pulse
from cudaq_pulse.controller import sample_waveform

cal = {
    "duration": 20 * pulse.ns,
    "sigma": 5 * pulse.ns,
    "drag_beta": 0.25 * pulse.ns,
    "tanh_sigma": 2 * pulse.ns,
    "flat_top_duration": 60 * pulse.ns,
    "flat_top_width": 40 * pulse.ns,
}


def half_sine(t):
    """Half a sine period over 40 clock cycles (20 ns at 2 GHz), for custom():
    *t* is in clock cycles from the start of the pulse."""
    return 0.3 * math.sin(math.pi * t / 40)


NAMES = [
    "gaussian",
    "square",
    "drag",
    "cosine",
    "tanh_ramp",
    "gaussian_square",
    "custom",
    "custom_samples",
]


@pulse.kernel
def every_envelope(qudit):
    """Play each built-in envelope once, in the order of NAMES."""
    d = get_drive_line(qudit)
    duration, sigma = cal["duration"], cal["sigma"]
    drive(d, gaussian(duration, 0.3, sigma))
    drive(d, square(duration, 0.2 + 0.1j))
    drive(d, drag(duration, 0.3, sigma, cal["drag_beta"]))
    drive(d, cosine(duration, 0.3))
    drive(d, tanh_ramp(duration, 0.3, cal["tanh_sigma"]))
    drive(d, gaussian_square(cal["flat_top_duration"], 0.3, sigma, cal["flat_top_width"]))
    drive(d, custom(duration, half_sine))
    drive(d, custom_samples([0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0, -0.1]))


def main():
    qs = pulse.quantum_system("transmon_surface17")
    program = pulse.compile(every_envelope, quantum_system=qs).lower(qs.qudits[0])
    drives = [e for e in program.export_to_controller()["events"] if e["kind"] == "drive"]
    envelopes = {}
    print(f"{'waveform':<16} {'cycles':>6} {'ns':>6} {'peak |I|':>9} {'peak |Q|':>9}")
    for name, event in zip(NAMES, drives, strict=True):
        samples = sample_waveform(event["waveform"], event["duration_cycles"])
        envelopes[name] = samples
        cycles = event["duration_cycles"]
        print(
            f"{name:<16} {cycles:>6} {cycles / qs.clock_ghz:>6g} "
            f"{abs(samples.real).max():>9.4f} {abs(samples.imag).max():>9.4f}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed. Install it with the viz extra: cudaq-pulse[viz]")
        return
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharey=True)
    for ax, (name, samples) in zip(axes.flat, envelopes.items(), strict=True):
        ax.plot(samples.real, label="I (real)")
        ax.plot(samples.imag, "--", label="Q (imag)")
        ax.set_title(name)
        ax.set_xlabel(f"clock cycles ({qs.clock_ghz:g} GHz)")
    axes[0, 0].legend()
    fig.tight_layout()
    print("\n  matplotlib figure created. Save it with fig.savefig('envelopes.png').")


if __name__ == "__main__":
    main()
