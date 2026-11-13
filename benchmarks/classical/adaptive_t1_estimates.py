# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every estimate of the adaptive T1 benchmark, as a reproducible JSON file.

Runs the adaptive T1 step of ``calibration_protocol.py`` (the tracker of
examples/21 on a quantum system like that of Berritta et al., arXiv:2506.09576:
T1 160 us, assignment errors 0.11 and 0.14, prior (3, 450 us), 50 shots)
and writes each trajectory's T1 estimate with their median, a bootstrap
interval for the median and the 16-84% spread. Nothing in the file depends
on timing, so the same seed and trajectory count give the same file.

    python benchmarks/classical/adaptive_t1_estimates.py --trajectories 400 --seed 1 \\
        --output benchmarks/classical/adaptive_t1_estimates.json

``--check FILE`` regenerates the estimates and compares them with FILE.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np

_BENCH = runpy.run_path(
    str(Path(__file__).with_name("calibration_protocol.py")), run_name="adaptive_t1_estimates"
)
CLOCK = _BENCH["CLOCK"]


def estimates(trajectories: int, seed: int) -> list[float]:
    """The T1 estimate (us) of every trajectory, in branch order."""
    qs = _BENCH["_att_target"]()
    scheduled = pulse.compile(
        _BENCH["att_paper"], quantum_system=qs, control_system=_BENCH["P"]["control_system"]
    ).lower(qs.qudits[0])
    result = pulse.simulate(
        scheduled,
        shots_count=trajectories,
        seed=seed,
        branches="sampled",
        readout="model",
        levels=2,
    )
    out = []
    for calls, count in _BENCH["_shares"](result, trajectories):
        shape, rate, _ = calls[-1].answer
        out += [rate / shape / CLOCK / 1e3] * count
    return out


def report(trajectories: int, seed: int) -> dict:
    values = estimates(trajectories, seed)
    rng = np.random.default_rng(seed)
    median, lo, hi = _BENCH["_bootstrap"](values, np.median, rng)
    return {
        "command": (
            "python benchmarks/classical/adaptive_t1_estimates.py"
            f" --trajectories {trajectories} --seed {seed}"
        ),
        "trajectories": trajectories,
        "seed": seed,
        "quantum_system": {
            "t1_us": 160.0,
            "assignment_errors": [0.11, 0.14],
            "prior_shape_and_mean_t1_us": [3.0, 450.0],
            "adaptive_shots": _BENCH["P"]["tracker_shots"],
        },
        "median_t1_us": median,
        "median_t1_bootstrap_95pct_us": [lo, hi],
        "spread_16_84_us": [float(x) for x in np.quantile(values, [0.16, 0.84])],
        "paper": {
            "reference": "F. Berritta et al., arXiv:2506.09576 (2025)",
            "t1_us": 159.0,
            "interval_68pct_us": [131.0, 202.0],
        },
        "estimates_us": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectories", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, help="write the JSON here (default: stdout)")
    parser.add_argument("--check", type=Path, help="compare with this JSON file instead")
    args = parser.parse_args()
    if args.check is not None:
        saved = json.loads(args.check.read_text())
        fresh = estimates(saved["trajectories"], saved["seed"])
        same = len(fresh) == len(saved["estimates_us"]) and np.allclose(
            fresh, saved["estimates_us"], rtol=1e-9, atol=0.0
        )
        print(f"{args.check}: {'reproduced' if same else 'DIFFERS'}")
        return 0 if same else 1
    text = json.dumps(report(args.trajectories, args.seed), indent=1) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
