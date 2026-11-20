# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each challenge's scorer passes its reference solutions and cannot be changed by a submission."""

import subprocess
import sys
from pathlib import Path

import pytest

_CHALLENGES = Path(__file__).resolve().parents[1] / "challenges"
_NAMES = ("pi_pulse", "mystery_qubit", "benchmark_it_yourself")
_SEEDED = {"mystery_qubit", "benchmark_it_yourself"}  # scorers that draw hidden qubits

pytestmark = pytest.mark.skipif(not _CHALLENGES.exists(), reason="needs the source tree")


def _score(name, submission, *args):
    done = subprocess.run(
        [sys.executable, str(_CHALLENGES / name / "score.py"), str(submission), *args],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return done.stdout


def _rungs(output):
    return {line.split()[0]: line.split()[1].rstrip(":") for line in output.splitlines() if line}


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("rung", ["bronze", "silver"])
def test_a_reference_solution_reaches_its_rung(name, rung):
    seed = ["2026"] if name in _SEEDED else []
    output = _score(name, _CHALLENGES / name / "reference" / f"{rung}.py", *seed)
    rungs = _rungs(output)
    assert rungs["BRONZE"] == "pass", output
    if rung == "silver":
        assert rungs["SILVER"] == "pass", output


def test_the_bronze_references_do_not_reach_silver():
    # Silver must need more than bronze does, or the rung means nothing.
    for name in _NAMES:
        seed = ["2026"] if name in _SEEDED else []
        output = _score(name, _CHALLENGES / name / "reference" / "bronze.py", *seed)
        assert _rungs(output)["SILVER"] == "fail", output


def test_a_submission_cannot_change_the_scorer(tmp_path):
    # It replaces the simulator, lowers the pass lines and prints a result of its own.
    cheat = tmp_path / "cheat.py"
    cheat.write_text(
        "import sys\n"
        "import cudaq_pulse as pulse\n"
        "class Perfect:\n"
        "    def level_populations(self, qudit):\n"
        "        return (0.0, 1.0, 0.0)\n"
        "pulse.simulate = lambda *args, **kwargs: Perfect()\n"
        "main = sys.modules['__main__']\n"
        "main.BRONZE_P1 = main.SILVER_P1 = 0.0\n"
        "print('BRONZE pass')\n"
        "@pulse.kernel\n"
        "def pi_pulse(qudit):\n"
        "    drive(get_drive_line(qudit), gaussian(20 * pulse.ns, 0.05, 5 * pulse.ns))\n"
    )
    output = _score("pi_pulse", cheat)
    assert "BRONZE fail" in output and "BRONZE pass" not in output, output
    assert "valid=yes" in output, output


def test_a_file_that_does_not_load_is_invalid(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('no kernel here')\n")
    for name in _NAMES:
        output = _score(name, broken)
        assert "INVALID" in output and "BRONZE fail" in output, output
