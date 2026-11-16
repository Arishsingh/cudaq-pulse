# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""benchmarks/simulation/qec_scaling.py: the gross code's checks are those of
a [[144,12]] code, and both cycles compile to the sizes the performance notes
quote."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "simulation" / "qec_scaling.py"
if not _SCRIPT.exists():
    pytest.skip("needs the source tree's benchmarks/", allow_module_level=True)


@pytest.fixture(scope="module")
def bench():
    return runpy.run_path(str(_SCRIPT), run_name="bench")


def _gf2_rank(rows: list[list[int]]) -> int:
    pivots: dict[int, int] = {}
    for row in rows:
        v = sum(1 << i for i in row)
        while v:
            top = v.bit_length() - 1
            if top not in pivots:
                pivots[top] = v
                break
            v ^= pivots[top]
    return len(pivots)


def test_the_gross_code_checks(bench):
    x_checks, z_checks = bench["gross_code_checks"]()
    assert len(x_checks) == len(z_checks) == 72
    assert {len(set(row)) for row in x_checks + z_checks} == {6}
    for checks in (x_checks, z_checks):
        degree = [0] * 144
        for row in checks:
            for q in row:
                degree[q] += 1
        assert set(degree) == {3}
    assert all(len(set(x) & set(z)) % 2 == 0 for x in x_checks for z in z_checks)
    assert 144 - _gf2_rank(x_checks) - _gf2_rank(z_checks) == 12


def test_the_cycles_compile_to_the_quoted_sizes(bench):
    qudit_count, op_count, stages = bench["bench_gross"](1)
    assert (qudit_count, op_count) == (288, 4320)
    assert set(stages) == set(bench["_STAGES"])
    assert bench["bench"](31, 1)[:2] == (1921, 20401)


def test_the_gross_code_takes_no_distances(bench, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qec_scaling.py", "--code", "gross", "--distances", "5"])
    with pytest.raises(SystemExit) as raised:
        bench["main"]()
    assert raised.value.code == 2
