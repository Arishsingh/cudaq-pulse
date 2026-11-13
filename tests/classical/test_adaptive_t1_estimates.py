# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The committed adaptive T1 estimates: their summary matches the estimates,
and benchmarks/classical/adaptive_t1_estimates.py writes them again."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

_BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks" / "classical"
_JSON = _BENCHMARKS / "adaptive_t1_estimates.json"
if not _JSON.exists():
    pytest.skip("needs the source tree's benchmarks/", allow_module_level=True)
SAVED = json.loads(_JSON.read_text())


def test_the_summary_is_that_of_the_estimates():
    values = SAVED["estimates_us"]
    assert len(values) == SAVED["trajectories"]
    assert SAVED["median_t1_us"] == pytest.approx(np.median(values), rel=1e-12)
    assert SAVED["spread_16_84_us"] == pytest.approx(np.quantile(values, [0.16, 0.84]), rel=1e-12)
    lo, hi = SAVED["median_t1_bootstrap_95pct_us"]
    assert lo < SAVED["median_t1_us"] < hi
    assert SAVED["command"].endswith(f"--trajectories {len(values)} --seed {SAVED['seed']}")


def test_the_median_is_within_the_papers_interval():
    lo, hi = SAVED["paper"]["interval_68pct_us"]
    assert lo < SAVED["median_t1_us"] < hi


@pytest.fixture(scope="module")
def bench():
    return runpy.run_path(str(_BENCHMARKS / "adaptive_t1_estimates.py"), run_name="bench")


@pytest.mark.slow
def test_the_benchmark_writes_the_report(bench):
    report = bench["report"](4, 1)
    assert len(report["estimates_us"]) == 4
    assert report["command"].endswith("--trajectories 4 --seed 1")
    assert set(report) == set(SAVED)


@pytest.mark.nightly
def test_the_committed_estimates_are_reproduced(bench):
    fresh = bench["estimates"](SAVED["trajectories"], SAVED["seed"])
    assert fresh == pytest.approx(SAVED["estimates_us"], rel=1e-9)
