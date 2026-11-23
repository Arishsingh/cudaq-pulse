# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each challenge's scorer passes its reference solutions and cannot be changed by a submission."""

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

_CHALLENGES = Path(__file__).resolve().parents[1] / "challenges"
_SUBMISSIONS = Path(__file__).resolve().parent / "challenge_submissions"
_NAMES = (
    "pi_pulse",
    "mystery_qubit",
    "benchmark_it_yourself",
    "schedule_golf",
    "fix_the_kernel",
    "reset_one_qubit",
)
# Scorers that draw something hidden: qubits, cases or states.
_SEEDED = {"mystery_qubit", "benchmark_it_yourself", "fix_the_kernel", "reset_one_qubit"}
_LABS = ("benchmark_it_yourself", "mystery_qubit")  # a submission plays through lab.run

# Every test starts a scorer, which starts a process of its own for the submission.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _CHALLENGES.exists(), reason="needs the source tree"),
]


def _run(name, submission, *args, timeout=600, env=None):
    return subprocess.run(
        [sys.executable, str(_CHALLENGES / name / "score.py"), str(submission), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _score(name, submission, *args):
    """What the scorer prints, with what it shows of the submission's text put back on one line."""
    return _run(name, submission, *args).stdout.replace("\n  | ", " ")


def _rungs(output):
    """The verdict of each rung: the word after BRONZE or SILVER on every such line."""
    rungs = {"BRONZE": [], "SILVER": []}
    for line in output.splitlines():
        words = line.split()
        if len(words) > 1 and words[0] in rungs:
            rungs[words[0]].append(words[1].rstrip(":"))
    return rungs


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("rung", ["bronze", "silver"])
def test_a_reference_solution_reaches_its_rung(name, rung):
    seed = ["2026"] if name in _SEEDED else []
    output = _score(name, _CHALLENGES / name / "reference" / f"{rung}.py", *seed)
    rungs = _rungs(output)
    assert rungs["BRONZE"] == ["pass"], output
    if rung == "silver":
        assert rungs["SILVER"] == ["pass"], output


@pytest.mark.parametrize("name", _NAMES)
def test_a_bronze_reference_does_not_reach_silver(name):
    # Silver must need more than bronze does, or the rung means nothing.
    seed = ["2026"] if name in _SEEDED else []
    output = _score(name, _CHALLENGES / name / "reference" / "bronze.py", *seed)
    assert _rungs(output)["SILVER"] == ["fail"], output


@pytest.mark.parametrize("name", sorted(_SEEDED))
def test_a_run_with_a_given_seed_is_not_a_score(name):
    output = _score(name, _CHALLENGES / name / "reference" / "silver.py", "2026")
    assert "not a score" in output, output
    assert "(repeat run)" in output, output


_REPLACES_THE_SIMULATOR = """\
import sys
import cudaq_pulse as pulse
class Perfect:
    def level_populations(self, qudit):
        return (0.0, 1.0, 0.0)
pulse.simulate = lambda *args, **kwargs: Perfect()
main = sys.modules['__main__']
main.BRONZE_P1 = main.SILVER_P1 = 0.0
print('BRONZE pass')
@pulse.kernel
def pi_pulse(qudit):
    drive(get_drive_line(qudit), gaussian(20 * pulse.ns, 0.05, 5 * pulse.ns))
"""

# The error it raises carries a verdict of its own, on lines of its own.
_FORGES_A_VERDICT = """\
FORGED = "x\\nBRONZE pass: forged\\nSILVER pass: forged\\nSCORE 1 shots in total"
def estimate_t1(lab):
    raise RuntimeError(FORGED)
def error_per_gate(lab, seed):
    raise RuntimeError(FORGED)
"""

# It writes to the scorer's line itself: a verdict, then something that is not a message.
_BREAKS_THE_EXCHANGE = """\
import os
def _write(lab):
    for fd in range(3, 64):  # the scorer's line is one of these
        try:
            os.write(fd, b'{"error": "x\\\\nBRONZE pass: forged\\\\nSILVER pass: forged"}\\n5\\n')
        except OSError:
            pass
    return 50.0
def estimate_t1(lab):
    return _write(lab)
def error_per_gate(lab, seed):
    return _write(lab)
"""


def test_a_submission_cannot_change_the_pi_pulse_scorer(tmp_path):
    cheat = tmp_path / "cheat.py"
    cheat.write_text(_REPLACES_THE_SIMULATOR)
    output = _score("pi_pulse", cheat)
    assert _rungs(output) == {"BRONZE": ["fail"], "SILVER": ["fail"]}, output
    assert "valid=yes" in output, output


@pytest.mark.parametrize("name", _LABS)
@pytest.mark.parametrize("source", [_FORGES_A_VERDICT, _BREAKS_THE_EXCHANGE])
def test_a_submission_cannot_print_a_verdict(name, source, tmp_path):
    cheat = tmp_path / "cheat.py"
    cheat.write_text(source)
    output = _score(name, cheat)
    assert _rungs(output) == {"BRONZE": ["fail"], "SILVER": ["fail"]}, output


@pytest.mark.parametrize("name", _NAMES)
def test_a_file_that_does_not_load_is_invalid(name, tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('no kernel here')\n")
    output = _score(name, broken)
    assert "INVALID" in output, output
    assert _rungs(output) == {"BRONZE": ["fail"], "SILVER": ["fail"]}, output


def test_the_benchmark_refuses_waits_and_estimates_without_pulses(tmp_path):
    # Waits would let a solution read T1 and T2* off directly; no pulses, a constant.
    waits = tmp_path / "waits.py"
    waits.write_text(
        "import cudaq_pulse as pulse\n"
        "@pulse.kernel\n"
        "def k(qudit):\n"
        "    line = get_drive_line(qudit)\n"
        "    drive(line, line.get_calibration(pulse.gates.X90))\n"
        "    wait(line, 1000 * pulse.ns)\n"
        "    drive(line, line.get_calibration(pulse.gates.X90))\n"
        "def error_per_gate(lab, seed):\n"
        "    lab.run(k, lab.qubit)\n"
        "    return 6.5e-4\n"
    )
    assert "back to back" in _score("benchmark_it_yourself", waits)
    constant = tmp_path / "constant.py"
    constant.write_text(
        "import cudaq_pulse as pulse\n"
        "@pulse.kernel\n"
        "def k(qudit):\n"
        "    get_drive_line(qudit)\n"
        "def error_per_gate(lab, seed):\n"
        "    lab.run(k, lab.qubit)\n"
        "    lab.run(k, lab.qubit)\n"
        "    return 6.5e-4\n"
    )
    output = _score("benchmark_it_yourself", constant)
    assert "INVALID" in output and "at least 2 sequences and 20 pulses" in output, output


# The three challenges below score what a kernel plays, as their second process reports it. A
# cheat is a reference solution with one edit, or a file that goes around the kernel.


def _reference(name, rung, *edits):
    """The source of a reference solution, with each (old, new) pair replaced in it."""
    source = (_CHALLENGES / name / "reference" / f"{rung}.py").read_text()
    for old, new in edits:
        assert old in source, old
        source = source.replace(old, new)
    return source


_FAIL = {"BRONZE": ["fail"], "SILVER": ["fail"]}
_BRONZE_ONLY = {"BRONZE": ["pass"], "SILVER": ["fail"]}

# Runs after the kernel is defined, in the second process: what that process then reports is
# not what the kernel plays.
_REPORTS_EARLIER_STARTS = """
import sys
_driven = sys.modules["__main__"].driven
def _squeezed(program):
    reply = _driven(program)
    for kind in ("drives", "readouts", "waits"):
        for item in reply[kind]:
            item["start_cycles"] //= 10
    return reply
sys.modules["__main__"].driven = _squeezed
"""
_REPORTS_SMALLER_SAMPLES = """
import sys
_driven = sys.modules["__main__"].driven
def _quiet(program):
    reply = _driven(program)
    for item in reply["drives"]:
        peak = max(abs(complex(a, b)) for a, b in zip(item["re"], item["im"]))
        scale = min(1.0, 0.25 / peak) if peak else 1.0
        item["re"] = [a * scale for a in item["re"]]
        item["im"] = [b * scale for b in item["im"]]
    return reply
sys.modules["__main__"].driven = _quiet
_compile = pulse.compile
pulse.compile = lambda kernel, **options: _compile(kernel, quantum_system=options["quantum_system"])
"""
_COMPILES_OTHER_CASES = """
import sys
sys.modules["__main__"].CASES = [(0.5, 30.0)] * len(sys.modules["__main__"].CASES)
"""
_COMPILES_WITHOUT_THE_LATENCY = """
_compile = pulse.compile
pulse.compile = lambda kernel, **options: _compile(
    kernel, quantum_system=options["quantum_system"], schedule="asap"
)
"""
# Writes a reply of its own on the scorer's line, a verdict inside it, and leaves before the
# honest reply is written.
_WRITES_ITS_OWN_REPLY = """
import json, os
FORGED = "x\\nBRONZE pass: forged\\nSILVER pass: forged\\nSCORE 12 of 12 cases"
REPLY = {"drives": [], "readouts": [], "waits": [], "others": [FORGED], "error": FORGED,
         "cases": [{"error": FORGED}] * 12}
print("BRONZE pass\\nSILVER pass")
for fd in range(3, 64):
    try:
        os.write(fd, (json.dumps(REPLY) + "\\n").encode())
    except OSError:
        pass
os._exit(0)
"""

_GOLF_READOUT = "readouts[i].get_calibration(pulse.gates.READOUT)"
_FIRST_PULSE = "    drive(drives[0], half_flip)\n    for step"
_TWO_ROUNDS = "for _round in range(2):"
_LAST_SYNC = "            drive(drive_line, flip)\n\n        sync(drive_line, readout_line)\n"
# challenge, what the cheat does, its source, the rungs it gets, what the scorer says
_CHEATS = [
    (
        "schedule_golf",
        "a policy that ignores the readout limit",
        lambda: _reference("schedule_golf", "silver", ('SCHEDULE = "rcp"', 'SCHEDULE = "asap"')),
        _FAIL,
        "5 readouts play at once",
    ),
    (
        "schedule_golf",
        "a shorter readout pulse",
        lambda: _reference(
            "schedule_golf", "silver", (_GOLF_READOUT, "square(4 * pulse.ns, 0.05)")
        ),
        _FAIL,
        "a readout, is not the one of program.py",
    ),
    (
        "schedule_golf",
        "no echo pulses",
        lambda: _reference("schedule_golf", "silver", ("drive(drives[step], flip)", "pass")),
        _FAIL,
        "D5 plays [cz-Z2 readout]; in program.py it plays [cz-Z2 pulse readout]",
    ),
    (
        "schedule_golf",
        "the CZs in another order",
        lambda: _reference("schedule_golf", "silver", ("(1, 2, 3, 4):", "(1, 2, 4, 3):")),
        _FAIL,
        "Z2 plays [pulse cz-D5 cz-D8 cz-D7 cz-D4 pulse readout]",
    ),
    (
        "schedule_golf",
        "a CZ slot in which only Z2 waits",
        lambda: _reference("schedule_golf", "silver", ("wait(drives[step], CZ_SLOT)", "pass")),
        _FAIL,
        "is not a CZ slot",
    ),
    (
        "schedule_golf",
        "a phase change before a pulse",
        lambda: _reference(
            "schedule_golf",
            "silver",
            (_FIRST_PULSE, "    shift_phase(drives[0].tone, 1.0)\n" + _FIRST_PULSE),
        ),
        _FAIL,
        "its samples, frequency or phase differ",
    ),
    (
        "schedule_golf",
        "a report with every start divided by ten",
        lambda: _reference("schedule_golf", "silver") + _REPORTS_EARLIER_STARTS,
        _FAIL,
        "starts before its",
    ),
    (
        "fix_the_kernel",
        "the amplitude clipped at the bench's limit",
        lambda: _reference(
            "fix_the_kernel",
            "bronze",
            ("turn: float", "turn"),
            ("amplitude=0.33 * turn", "amplitude=min(0.33 * turn, 0.25)"),
        ),
        _BRONZE_ONLY,
        "from level 0 the first pulse leaves",
    ),
    (
        "fix_the_kernel",
        "a longer pulse with the old sigma",
        lambda: _reference(
            "fix_the_kernel", "silver", ("sigma=8 * pulse.ns", "sigma=4 * pulse.ns")
        ),
        _FAIL,
        "from level 0 the first pulse leaves",
    ),
    (
        "fix_the_kernel",
        "a wait that ignores gap_ns",
        lambda: _reference("fix_the_kernel", "silver", ("gap_ns * pulse.ns", "30 * pulse.ns")),
        _FAIL,
        "the second pulse starts 30 ns after the first ends; gap_ns is",
    ),
    (
        "fix_the_kernel",
        "a readout that does not wait for the pulses",
        lambda: _reference("fix_the_kernel", "silver", ("sync(drive_line, readout_line)", "pass")),
        _FAIL,
        "the readout starts at",
    ),
    (
        "fix_the_kernel",
        "a report with the samples scaled under the limit",
        lambda: _reference("fix_the_kernel", "bronze") + _REPORTS_SMALLER_SAMPLES,
        _BRONZE_ONLY,
        "from level 0 the first pulse leaves",
    ),
    (
        "fix_the_kernel",
        "every case compiled as one that passes",
        lambda: _reference("fix_the_kernel", "silver") + _COMPILES_OTHER_CASES,
        _FAIL,
        "gap_ns is",
    ),
    (
        "reset_one_qubit",
        "a reset that plays nothing",
        lambda: _reference("reset_one_qubit", "bronze", ("if outcome:", "if False:")).replace(
            "    outcome = readout(readout_line, measurement)\n", ""
        ),
        _FAIL,
        "your reset: 0 readouts and 0 pulses",
    ),
    (
        "reset_one_qubit",
        "a flip without a measurement",
        lambda: _reference("reset_one_qubit", "bronze", ("if outcome:", "if True:")).replace(
            "    outcome = readout(readout_line, measurement)\n", ""
        ),
        _FAIL,
        "your reset: 0 readouts and 1 pulse,",
    ),
    (
        "reset_one_qubit",
        "a shorter readout pulse",
        lambda: _reference(
            "reset_one_qubit",
            "silver",
            ("readout_line.get_calibration(pulse.gates.READOUT)", "square(4 * pulse.ns, 0.05)"),
        ),
        _FAIL,
        "a readout is not D1's calibrated readout",
    ),
    (
        "reset_one_qubit",
        "three rounds",
        lambda: _reference("reset_one_qubit", "silver", (_TWO_ROUNDS, "for _round in range(3):")),
        _FAIL,
        "it has 3 pulses and 3 readouts; the limits are 16 and 2",
    ),
    (
        "reset_one_qubit",
        "two rounds compiled without the feedback latency",
        lambda: _reference("reset_one_qubit", "silver") + _COMPILES_WITHOUT_THE_LATENCY,
        _FAIL,
        "the controller needs 200 ns",
    ),
    (
        "reset_one_qubit",
        "a pulse during the next readout",
        lambda: _reference(
            "reset_one_qubit",
            "silver",
            (_LAST_SYNC, "            drive(drive_line, flip)\n"),
        ),
        _FAIL,
        "plays while D1 is read out",
    ),
    (
        "reset_one_qubit",
        "two rounds that start 300 ns late",
        lambda: _reference(
            "reset_one_qubit",
            "silver",
            ("    for _round", "    wait(readout_line, 300 * pulse.ns)\n    for _round"),
        ),
        _FAIL,
        "it ends at 1380 ns, its last pulse, readout or wait; the limit is 1200 ns",
    ),
    (
        "reset_one_qubit",
        "a pulse at the 1-2 frequency",
        lambda: (
            _reference("reset_one_qubit", "bronze")
            + "    shift_frequency(drive_line.tone, -184e6)\n    drive(drive_line, flip)\n"
        ),
        _FAIL,
        "is not at D1's frequency with phase 0",
    ),
]
_GOES_AROUND = [
    ("its own reply and a verdict", _WRITES_ITS_OWN_REPLY, "INVALID"),
    (
        "an early exit with code 0",
        "import os\nprint('BRONZE pass')\nos._exit(0)\n",
        "ended without a result",
    ),
]


@pytest.mark.parametrize(
    ("name", "source", "rungs", "message"),
    [pytest.param(*cheat[:1], *cheat[2:], id=f"{cheat[0]}: {cheat[1]}") for cheat in _CHEATS],
)
def test_a_cheat_fails_with_the_rule_it_breaks(name, source, rungs, message, tmp_path):
    cheat = tmp_path / "cheat.py"
    cheat.write_text(source())
    output = _score(name, cheat)
    assert _rungs(output) == rungs, output
    assert message in output, output


@pytest.mark.parametrize("name", ["schedule_golf", "fix_the_kernel", "reset_one_qubit"])
@pytest.mark.parametrize(
    ("source", "message"), [pytest.param(*cheat[1:], id=cheat[0]) for cheat in _GOES_AROUND]
)
def test_a_reply_that_is_not_a_valid_report_gets_no_rung(name, source, message, tmp_path):
    cheat = tmp_path / "cheat.py"
    cheat.write_text(source)
    output = _score(name, cheat)
    assert _rungs(output) == _FAIL, output
    assert message in output, output
    assert "SCORE none" in output or "valid=no" in output, output


def test_the_program_to_shorten_is_valid_and_reaches_no_rung():
    output = _score("schedule_golf", _CHALLENGES / "schedule_golf" / "program.py")
    assert _rungs(output) == _FAIL, output
    assert "SCORE duration_ns=1312 valid=yes" in output, output


def test_the_kernel_to_fix_is_refused_with_the_compilers_message():
    output = _score("fix_the_kernel", _CHALLENGES / "fix_the_kernel" / "broken.py")
    assert _rungs(output) == _FAIL, output
    assert "INVALID it does not compile" in output and "did you mean drive()?" in output, output


# What an independent attack on the scorers found. Each test runs the submission that showed
# it, kept in challenge_submissions/.

_NEW = ("schedule_golf", "fix_the_kernel", "reset_one_qubit")
_FORGED = ("written by the submission", "written to the terminal", "BRONZE pass  (", "SCORE forged")


def _seed(name):
    return ["2026"] if name in _SEEDED else []


def _package_copy(folder):
    """A copy of the cudaq_pulse package: its Python files copied, the rest linked."""
    origin = importlib.util.find_spec("cudaq_pulse").origin
    source = Path(origin).parent
    for path, _, files in os.walk(source, followlinks=True):
        if "__pycache__" in path:
            continue
        target = folder / "cudaq_pulse" / Path(path).relative_to(source)
        target.mkdir(parents=True, exist_ok=True)
        for file in files:
            if file.endswith(".py"):
                shutil.copyfile(Path(path) / file, target / file)
            else:
                os.symlink(os.path.realpath(Path(path) / file), target / file)
    return folder / "cudaq_pulse"


@pytest.mark.skipif(os.name != "posix", reason="links the package's other files")
@pytest.mark.parametrize("name", _NAMES)
def test_code_planted_in_the_package_during_a_run_is_not_run_by_the_scorer(name, tmp_path):
    # The scorer has loaded and used all it needs before it starts the submission's process.
    package = _package_copy(tmp_path)
    paths = [str(tmp_path), os.environ.get("PYTHONPATH", "")]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(path for path in paths if path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CHALLENGE_TEST_PACKAGE_COPY": str(package),
    }
    done = _run(name, _SUBMISSIONS / "plants_code_in_the_package.py", *_seed(name), env=env)
    assert "written by the submission" in (package / "__init__.py").read_text(), done.stderr
    assert "written by the submission" not in done.stdout, done.stdout
    assert _rungs(done.stdout) == _BRONZE_ONLY, done.stdout + done.stderr


@pytest.mark.skipif(os.name != "posix", reason="needs a pseudo-terminal")
@pytest.mark.parametrize("name", _NAMES)
def test_a_submission_has_no_terminal_to_write_a_verdict_to(name):
    # The scorer runs on a pseudo-terminal; what reaches that terminal is all that is read here.
    scorer = [sys.executable, str(_CHALLENGES / name / "score.py")]
    command = [*scorer, str(_SUBMISSIONS / "writes_to_the_terminal.py")]
    wait = [sys.executable, "-c", "import time; time.sleep(5)"]  # for the process left behind
    code = f"import pty\npty.spawn({command!r})\npty.spawn({wait!r})\n"
    done = subprocess.run(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert "BRONZE fail" in done.stdout, done.stdout + done.stderr
    assert not any(forged in done.stdout for forged in _FORGED), done.stdout


@pytest.mark.skipif(os.name != "posix", reason="passes descriptors to a process it leaves behind")
@pytest.mark.parametrize("name", _NEW)
def test_a_process_left_behind_does_not_hold_the_scorer_up(name):
    started = time.monotonic()
    done = _run(name, _SUBMISSIONS / "leaves_a_process_holding_the_line.py", timeout=100)
    assert _rungs(done.stdout) == _BRONZE_ONLY, done.stdout + done.stderr
    assert time.monotonic() - started < 60


@pytest.mark.parametrize("name", sorted(_SEEDED))
@pytest.mark.parametrize("seed", ["abc", "1.5", "", "0x10", "-1"])
def test_a_seed_that_is_not_a_whole_number_gets_the_usage_line(name, seed):
    done = _run(name, _CHALLENGES / name / "reference" / "bronze.py", seed)
    assert done.returncode != 0 and done.stdout == "", done.stdout
    assert done.stderr.startswith("usage: python challenges/"), done.stderr
    assert "Traceback" not in done.stderr, done.stderr


@pytest.mark.parametrize("name", _NAMES)
def test_text_from_a_submission_never_starts_or_wraps_onto_a_line(name):
    # An error text padded so that, printed after the scorer's words, a verdict would start
    # each row of an 80-column terminal.
    output = _run(name, _SUBMISSIONS / "raises_a_verdict_padded_to_wrap.py").stdout
    assert _rungs(output) == _FAIL, output
    for line in output.splitlines():
        if "pass" in line or "duration_ns" in line:
            assert line.startswith("  | ") and len(line) <= 76, output
        assert len(line) <= 120, output


@pytest.mark.parametrize("name", _NAMES)
def test_what_a_submission_prints_is_marked_as_its_own(name):
    done = _run(name, _SUBMISSIONS / "prints_a_verdict_everywhere.py")
    assert _rungs(done.stdout) == _FAIL, done.stdout
    assert "SILVER pass" in done.stderr, done.stderr
    assert all(line.startswith("  | ") for line in done.stderr.splitlines()), done.stderr


@pytest.mark.parametrize("name", _LABS)
def test_junk_on_the_scorers_line_is_named_as_that(name):
    output = _score(name, _SUBMISSIONS / "writes_on_the_scorers_line.py")
    assert _rungs(output) == _FAIL, output
    assert "INVALID its process sent something that is not a message" in output, output


# challenge, submission, the rungs it gets with seed 2026, what the scorer says
_TOO_LONG = "lasts 2200 clock cycles; none longer than 2048 (1024 ns) is read"
_BOTH = {"BRONZE": ["pass"], "SILVER": ["pass"]}
_FOUND = [
    ("reset_one_qubit", "reset_with_a_silent_pulse_in_a_branch", _BOTH, "2 readouts and 3 pulses"),
    ("reset_one_qubit", "reset_with_a_pulse_too_long_to_read", _FAIL, _TOO_LONG),
    ("reset_one_qubit", "reset_with_a_wait_too_long_to_read", _FAIL, "beyond 200000 (100 us)"),
    ("fix_the_kernel", "two_turns_with_a_pulse_too_long_to_read", _FAIL, _TOO_LONG),
    ("reset_one_qubit", "reset_with_a_readout_in_a_branch", _FAIL, "inside an if on an outcome"),
    ("reset_one_qubit", "reset_with_three_readouts", _FAIL, "3 readouts; the limits are 16 and 2"),
    ("reset_one_qubit", "reset_with_a_pulse_just_over_the_amplitude", _FAIL, "reaches 0.5000001;"),
    ("schedule_golf", "golf_with_a_wait_in_a_branch", _FAIL, "a wait depends on a measurement"),
    ("fix_the_kernel", "two_turns_with_a_detuned_readout", _FAIL, "the readout line's own tone"),
    ("fix_the_kernel", "two_turns_with_a_silent_readout", _FAIL, "less than 0.01 is silent"),
]


@pytest.mark.parametrize(
    ("name", "submission", "rungs", "message"),
    [pytest.param(*found, id=found[1]) for found in _FOUND],
)
def test_a_submission_the_attack_found_gets_the_right_verdict(name, submission, rungs, message):
    output = _score(name, _SUBMISSIONS / f"{submission}.py", *_seed(name))
    assert _rungs(output) == rungs, output
    assert message in output, output


# What the scorers do not check, as their statements say.
_TAKEN_AT_ITS_WORD = [
    # Only what is played is compared: a wait that is no CZ slot, or a phase put back, changes
    # nothing that is played.
    ("schedule_golf", "golf_with_an_extra_wait"),
    ("schedule_golf", "golf_with_a_phase_there_and_back"),
    # The second process is the submission's: one that reports a valid schedule, or the times
    # of a fixed kernel, without a kernel that plays them, is not told apart. On honour.
    ("schedule_golf", "golf_reply_copied_from_the_reference"),
    ("fix_the_kernel", "two_turns_reported_with_other_times"),
]


@pytest.mark.parametrize(("name", "submission"), _TAKEN_AT_ITS_WORD)
def test_what_a_scorer_cannot_tell_apart_passes(name, submission):
    output = _score(name, _SUBMISSIONS / f"{submission}.py", *_seed(name))
    assert _rungs(output) == _BOTH, output


_ANSWER_ONCE = ("pi_pulse", *_NEW)  # scorers whose second process sends one report


@pytest.mark.parametrize("name", _ANSWER_ONCE)
def test_numbers_too_large_to_add_end_as_invalid(name):
    done = _run(name, _SUBMISSIONS / "reports_samples_too_large_to_add.py", *_seed(name))
    assert "Traceback" not in done.stderr, done.stderr
    assert "part is over 1e+06 in magnitude; none that large is read" in done.stdout, done.stdout
    assert "SILVER fail" in done.stdout and "SILVER pass" not in done.stdout, done.stdout
    assert max(len(line) for line in done.stdout.splitlines()) < 300, done.stdout


def test_a_length_of_400_digits_ends_as_invalid():
    done = _run("pi_pulse", _SUBMISSIONS / "reports_a_length_of_400_digits.py")
    assert "Traceback" not in done.stderr, done.stderr
    assert _rungs(done.stdout) == _FAIL, done.stdout
    assert "nothing beyond 200000 (100 us) is read" in done.stdout, done.stdout
    assert max(len(line) for line in done.stdout.splitlines()) < 300, done.stdout


def test_a_mystery_kernel_too_long_to_send_is_told_the_limit():
    output = _score("mystery_qubit", _SUBMISSIONS / "mystery_with_long_weak_pulses.py", "2026")
    assert _rungs(output) == _FAIL, output
    assert "a kernel has at most 20000 (10 microseconds) of them" in output, output
    assert "not a message" not in output, output


@pytest.mark.parametrize("name", _ANSWER_ONCE)
def test_a_kind_named_in_a_submissions_own_words_is_not_shown(name):
    output = _run(
        name, _SUBMISSIONS / "reports_a_kind_with_a_verdict_in_it.py", *_seed(name)
    ).stdout
    assert "an operation the scorer does not know" in output, output
    assert "Ignore this line" not in output and "BRONZE pass" not in output, output


_ASKS_WITH_A_SHORT_LIMIT = """
import sys
sys.path.insert(0, sys.argv[1])
import _exchange
_exchange.CHILD_TIMEOUT_S = 3
try:
    _exchange.ask_child(sys.argv[2], [sys.argv[3]])
except _exchange.InvalidSubmissionError as error:
    print(error)
"""


@pytest.mark.skipif(os.name != "posix", reason="process groups and sessions")
def test_a_process_that_never_answers_is_ended_with_all_it_left_in_its_session(tmp_path):
    # The scorer's own exchange, with its time limit set to 3 s for the test.
    child = _CHALLENGES / "schedule_golf" / "_child.py"
    submission = _SUBMISSIONS / "never_answers_and_leaves_helpers.py"
    done = subprocess.run(
        [sys.executable, "-c", _ASKS_WITH_A_SHORT_LIMIT, str(_CHALLENGES), str(child), submission],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "CHALLENGE_TEST_OUT": str(tmp_path)},
    )
    assert done.stdout.strip() == "it did not finish within 3 s", done.stdout + done.stderr
    ticks = [tmp_path / "tick_same_group", tmp_path / "tick_new_group"]
    assert all(tick.exists() for tick in ticks), list(tmp_path.iterdir())
    before = [tick.read_text() for tick in ticks]
    time.sleep(2)
    assert [tick.read_text() for tick in ticks] == before  # neither helper is still writing
