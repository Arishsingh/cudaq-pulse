# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tools/perf-gate.py: reading Callgrind's dumps, the verdicts, and the
program sizes of its cases, which are the same on every platform."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import cudaq_pulse
import pytest

_ROOT = Path(__file__).resolve().parents[1]

# The wheels' tests run on a copy of tests/ without the tools.
if not (_ROOT / "tools" / "perf-gate.py").exists():
    pytest.skip("tools/perf-gate.py is not in this tree", allow_module_level=True)


def _gate():
    spec = importlib.util.spec_from_file_location("perf_gate", _ROOT / "tools" / "perf-gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


gate = _gate()

# A dump as Callgrind 3.22 writes it with --cache-sim=yes (the format of
# its manual, "Callgrind Format Specification"): the header's summary:, then
# cost lines under ob=/fl=/fn= with positions relative to the last ("+4",
# "-700", "*"), trailing zero events left out, file names given once and
# then referred to by number, a call (calls= and its inclusive cost, which
# is not self cost) and an inlined file (fi=, back with fe=). The totals:
# are the sum of the self-cost lines, added up by hand.
_SUMMARY = "941000 380000 185000 401 1810 600 40 160 60"
_DUMP = f"""\
# callgrind format
version: 1
creator: callgrind-3.22.0
pid: 4242
cmd:  python3 benchmarks/perf_gate/cases.py --measure libperf_gate_marks.so
part: 3


desc: I1 cache: 32768 B, 64 B, 8-way associative
desc: D1 cache: 32768 B, 64 B, 8-way associative
desc: LL cache: 8388608 B, 64 B, 16-way associative
desc: Timerange: Basic block 18263114 - 18402551
desc: Trigger: Client Request: compile_bell

positions: line
events: Ir Dr Dw I1mr D1mr D1mw ILmr DLmr DLmw
summary: {_SUMMARY}


ob=(1) /usr/bin/python3.12
fl=(1) ./Python/ceval.c
fn=(1) _PyEval_EvalFrameDefault
812 600000 250000 120000 300 900 200 40 60 10
+4 200000 80000 30000 100 300
cob=(2) /usr/lib/x86_64-linux-gnu/libc.so.6
cfi=(2) ./string/../sysdeps/x86_64/multiarch/memmove-vec-unaligned-erms.S
cfn=(2) __memmove_avx_unaligned_erms
calls=1500 0
* 90000 30000 30000 0 600 400 0 100 50
fi=(3) ./Include/object.h
-700 50000 20000 5000 0 10
fe=(1)
+700 1000 0 0 1

ob=(2)
fl=(2)
fn=(2)
0 90000 30000 30000 0 600 400 0 100 50

totals: {_SUMMARY}
"""


def test_a_dump_gives_its_case_and_counts():
    dump = gate.parse_dump(_DUMP)
    assert dump.name == "compile_bell"
    assert dump.events["Ir"] == 941_000
    assert dump.events["DLmw"] == 60
    assert dump.summary == dump.events


def test_the_dump_at_exit_is_not_a_case():
    assert (
        gate.parse_dump(_DUMP.replace("Client Request: compile_bell", "Program termination"))
        is None
    )


def test_a_dump_after_instrumentation_stopped_has_a_zero_summary_and_its_totals_are_used():
    # Switching instrumentation off zeroes the running total the summary
    # is taken from, not the cost lines: the first CI run's dumps.
    dump = gate.parse_dump(_DUMP.replace(f"summary: {_SUMMARY}", "summary: 0"))
    assert dump.events["Ir"] == 941_000
    assert dump.summary["Ir"] == 0


def test_a_dump_whose_totals_do_not_add_up_or_are_missing_is_an_error():
    with pytest.raises(gate.GateError, match="differ from the sum"):
        gate.parse_dump(_DUMP.replace("+700 1000 0 0 1", "+700 1001 0 0 1"))
    with pytest.raises(gate.GateError, match="totals"):
        gate.parse_dump(_DUMP.replace(f"totals: {_SUMMARY}\n", ""))


def test_a_dump_may_give_any_number_of_positions():
    instr_line = """\
desc: Trigger: Client Request: submit
positions: instr line
events: Ir
summary: 27
fn=(1) f
0x401000 12 20
+3 * 7
jump=2 +9 *
+9 *
calls=1 0x402000 40
* * 500
totals: 27
"""
    assert gate.parse_dump(instr_line).events == {"Ir": 27}
    no_positions = "desc: Trigger: Client Request: submit\npositions:\nevents: Ir Dr\n"
    no_positions += "fn=(1) f\n20 4\n7\ntotals: 27 4\n"
    assert gate.parse_dump(no_positions).events == {"Ir": 27, "Dr": 4}


def test_the_cycle_estimate_weights_cache_misses():
    events = gate.parse_dump(_DUMP).events
    # 1,506,000 accesses, 2,811 L1 misses of which 260 miss the last level too.
    assert gate.estimated_cycles(events) == (1_506_000 - 2811) + 5 * (2811 - 260) + 35 * 260
    assert gate.estimated_cycles({"Ir": 10}) is None


def _runs(**instructions):
    return {n: gate.Measured([float(x)] * 2, [0.0] * 2) for n, x in instructions.items()}


_PLAUSIBLE = {
    "first_compile": 90e6,
    "compile_bell": 1.2e6,
    "compile_surface_d5": 11e6,
    "compile_calls_small": 1.1e6,
    "compile_calls_large": 24e6,
    "submit": 1500,
}


def test_plausible_counts_pass_the_checks():
    assert gate.implausible(_runs(**_PLAUSIBLE), set(_PLAUSIBLE)) == []


def test_zero_missing_or_too_small_counts_are_errors():
    problems = gate.implausible(_runs(**{**_PLAUSIBLE, "submit": 0}), set(_PLAUSIBLE) | {"x"})
    assert any(p.startswith("submit: 0, 0 instructions") for p in problems)
    assert "x: no instruction count measured" in problems
    small = gate.implausible(_runs(**{**_PLAUSIBLE, "compile_bell": 9e4}), set(_PLAUSIBLE))
    assert any(p.startswith("compile_bell:") for p in small)
    nan = gate.Measured([float("nan")], [0.0])
    assert gate.implausible({"submit": nan}, {"submit"})


def test_a_heavier_case_must_measure_well_above_a_lighter_one():
    problems = gate.implausible(
        _runs(**{**_PLAUSIBLE, "compile_calls_large": 1.5e6}), set(_PLAUSIBLE)
    )
    assert problems == [
        "compile_calls_large measured 1,500,000 instructions per repetition, not more than "
        "twice compile_calls_small's 1,100,000: the counts do not follow the work"
    ]
    names = set(BASELINE["cases"])
    assert {name for pair in gate.HEAVIER for name in pair} <= names


def _baseline(instructions=1_000_000, **sizes):
    return {"cases": {"c": {"instructions": instructions, "op_count": 15, **sizes}}}


def _verdict(measured, baseline=None, sizes=None):
    runs = gate.Measured(instructions=list(measured), estimated_cycles=[0.0] * len(measured))
    rows = gate.compare(baseline or _baseline(), {"c": sizes or {"op_count": 15}}, {"c": runs})
    return rows[0]


@pytest.mark.parametrize(
    ("measured", "verdict"),
    [
        (1_000_000, "ok"),
        (1_003_999, "ok"),
        (1_004_001, "slower"),
        (1_005_000, "slower"),
        (996_001, "ok"),
        (995_000, "faster"),
        (989_000, "improved"),
    ],
)
def test_the_verdict_follows_the_change_in_instructions(measured, verdict):
    assert _verdict([measured]).verdict == verdict


def test_the_lowest_of_identical_runs_is_compared_and_their_spread_reported():
    row = _verdict([1_000_100, 1_000_000])
    assert row.measured == 1_000_000
    assert row.spread == pytest.approx(1e-4)


def test_a_case_may_carry_its_own_tolerance():
    baseline = _baseline()
    baseline["cases"]["c"]["tolerance"] = 0.02
    assert _verdict([1_010_000], baseline).verdict == "ok"


def test_any_change_in_program_size_fails():
    row = _verdict([1_000_000], sizes={"op_count": 16})
    assert row.verdict == "sizes"
    assert row.sizes == "op_count 15 -> 16"


def test_new_removed_and_unmeasured_cases_fail():
    rows = gate.compare(_baseline(), {"d": {}}, {"d": gate.Measured([1.0], [1.0])})
    assert {r.name: r.verdict for r in rows} == {"c": "removed", "d": "new"}
    row = _verdict([1_000_000], {"cases": {"c": {"op_count": 15}}})
    assert row.verdict == "no baseline"
    assert set(gate.FAILING) >= {"slower", "improved", "sizes", "new", "removed", "no baseline"}


def test_without_instruction_counts_only_the_sizes_are_compared():
    rows = gate.compare(_baseline(), {"c": {"op_count": 15}}, None)
    assert rows[0].verdict == "ok"


_SOURCE = {"repository": "o/r", "run": 7, "commit": "a" * 40, "processes": 5}


def test_a_baseline_without_instruction_counts_keeps_the_previous_ones():
    previous = {"environment": {"python": "3.12.15"}, "source": _SOURCE, **_baseline()}
    kept = gate.as_baseline(
        {"python": "other"}, {"c": {"op_count": 16}}, None, previous, {"c": 0.0045}, {"run": 8}
    )
    assert kept["environment"] == {"python": "3.12.15"}
    assert kept["source"] == _SOURCE
    assert kept["cases"]["c"] == {"instructions": 1_000_000, "op_count": 16, "tolerance": 0.0045}
    runs = {"c": gate.Measured([2_000_000.4, 2_000_100.0], [3_000_000.0, 3_000_001.0])}
    taken = gate.as_baseline({"python": "other"}, {"c": {"op_count": 16}}, runs, previous, {})
    assert taken["environment"] == {"python": "other"}
    assert "source" not in taken
    assert taken["cases"]["c"]["instructions"] == 2_000_000
    assert taken["cases"]["c"]["estimated_cycles"] == 3_000_000


def test_a_measured_baseline_records_its_run_and_the_declared_tolerances(monkeypatch):
    for name, value in {
        "GITHUB_RUN_ID": "41000000001",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_REPOSITORY": "o/r",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/perf-baseline",
        "GITHUB_SHA": "b" * 40,
    }.items():
        monkeypatch.setenv(name, value)
    source = gate._source(5)
    assert source == {
        "repository": "o/r",
        "run": 41000000001,
        "attempt": 2,
        "event": "workflow_dispatch",
        "ref": "refs/heads/perf-baseline",
        "commit": "b" * 40,
        "processes": 5,
    }
    previous = {"cases": {"c": {"op_count": 15, "tolerance": 0.02}}}
    runs = {"c": gate.Measured([1_000_000.0] * 5, [2.0] * 5)}
    taken = gate.as_baseline(
        {"python": "3.12.15"}, {"c": {"op_count": 15}}, runs, previous, {}, source
    )
    assert taken["source"] == source
    assert "tolerance" not in taken["cases"]["c"]
    monkeypatch.delenv("GITHUB_RUN_ID")
    assert gate._source(2) is None


def test_the_baseline_must_record_the_tolerances_the_cases_declare():
    baseline = _baseline(tolerance=0.0045)
    assert gate.tolerance_error(baseline, {"c": 0.0045}) is None
    assert "declares" in gate.tolerance_error(baseline, {"c": 0.005})
    assert "declares" in gate.tolerance_error(baseline, {})
    assert "declares" in gate.tolerance_error(_baseline(), {"c": 0.0045})


def test_counts_from_another_environment_are_named():
    baseline = {"environment": {"mlir_toolchain": "mlir-toolchain-22.1.8", "python": "3.12.15"}}
    same = {"mlir_toolchain": "mlir-toolchain-22.1.8", "python": "3.12.15"}
    assert gate.environment_error(baseline, same) is None
    error = gate.environment_error(baseline, {**same, "mlir_toolchain": "mlir-toolchain-23.1.4"})
    assert "'mlir-toolchain-22.1.8' -> 'mlir-toolchain-23.1.4'" in error
    assert "python" not in error
    assert "valgrind None ->" in gate.environment_error(baseline, {**same, "valgrind": "3.22"})


def _differences(*values):
    return [(f"case{i} (run0/callgrind.out.{i})", v) for i, v in enumerate(values)]


def test_a_summary_above_every_total_by_the_same_few_instructions_is_the_marks():
    # The difference CI's dumps showed: the same 4 instructions in every one.
    same = {"Ir": 4, "Dr": 0, "Dw": 0}
    warnings, note = gate.dump_notes(_differences(same, same, dict(same)))
    assert warnings == []
    assert note.startswith("every dump's summary: is Ir +4") and "marks' own blocks" in note
    assert gate.dump_notes(_differences({"Ir": -2}, {"Ir": -2}))[0] == []
    assert gate.dump_notes(_differences({"Ir": 0, "Dr": 0})) == ([], "")
    assert gate.dump_notes([]) == ([], "")


def test_a_summary_that_differs_from_case_to_case_or_by_much_is_a_warning():
    warnings, note = gate.dump_notes(_differences({"Ir": 4}, {"Ir": 5}, {"Ir": 4}))
    assert warnings == [
        "case1 (run0/callgrind.out.1): the dump's summary: differs from its totals: by "
        "Ir +5, not Ir +4 like the other dumps"
    ]
    assert note.startswith("every other dump's summary: is Ir +4")
    assert gate.dump_notes(_differences({"Ir": 0}, {"Ir": 3}, {"Ir": 0}))[1] == ""
    many = {"Ir": gate.MARKS_INSTRUCTIONS + 1}
    assert gate.dump_notes(_differences(many, many)) == (
        [
            "case0 (run0/callgrind.out.0): the dump's summary: differs from its totals: by Ir +65",
            "case1 (run0/callgrind.out.1): the dump's summary: differs from its totals: by Ir +65",
        ],
        "",
    )
    # The first CI run's summaries were 0: below each total by the case.
    zeros = gate.dump_notes(_differences({"Ir": -941_000}, {"Ir": -27}))
    assert len(zeros[0]) == 2 and zeros[1] == ""


_WORK = {"bytecodes": 449.6, "python_calls": 16.0, "c_calls": 17.1}


def _with_work(work=_WORK, measured=1_005_000):
    baseline = _baseline()
    baseline["cases"]["c"]["python_work"] = dict(_WORK)
    runs = {"c": gate.Measured([measured], [0.0])}
    return gate.compare(baseline, {"c": {"op_count": 15}}, runs, {"c": dict(work)})[0]


def test_a_moved_count_with_the_same_python_work_still_fails_and_says_so():
    row = _with_work()
    assert row.verdict == "slower" and row.work == "same"
    assert gate.failure_kind(row) == "layout only"
    assert gate.failure_reason(row) == (
        f"c: layout only, +0.50% instructions (limit +0.4%); {gate.LAYOUT_ONLY}"
    )
    assert "judge before re-baselining" in gate.LAYOUT_ONLY
    # A shift the other way is layout only too, not a gain to keep.
    down = _with_work(measured=985_000)
    assert down.verdict == "improved" and gate.failure_kind(down) == "layout only"
    assert gate.failure_reason(down) == (
        f"c: layout only, -1.50% instructions (limit -1%); {gate.LAYOUT_ONLY}"
    )


def test_a_moved_count_with_other_python_work_shows_each_counter():
    row = _with_work({**_WORK, "bytecodes": 452.6, "c_calls": 16.1})
    assert row.work == "changed" and gate.failure_kind(row) == "regression"
    assert gate.failure_reason(row) == (
        "c: regression, +0.50% instructions (limit +0.4%); Python work changed per "
        "repetition: bytecodes 449.6 -> 452.6 (+3), python_calls same, c_calls 17.1 -> 16.1 (-1)"
    )
    assert gate.work_changes({"a": 1.0}, {"a": 1.0, "b": 2.5}) == ["a same", "b none -> 2.5"]


def test_a_gain_is_reported_as_a_gain_to_keep_not_as_a_regression_or_layout():
    # compile_surface_d5 on 8f67cc3: -19.37%, its passes doing less.
    row = _with_work({**_WORK, "bytecodes": 400.0}, measured=806_300)
    assert gate.failure_kind(row) == "gain"
    assert gate.failure_reason(row) == (
        "c: gain, -19.37% instructions (limit -1%): update the baseline to keep the gain; "
        "Python work changed per repetition: bytecodes 449.6 -> 400 (-49.6), "
        "python_calls same, c_calls same"
    )
    assert gate.failure_reason(_verdict([806_300])) == (
        "c: gain, -19.37% instructions (limit -1%): update the baseline to keep the gain"
    )


def test_the_same_python_work_beyond_a_layout_shift_is_a_change_in_native_code():
    gain = _with_work(measured=806_300)
    assert gate.failure_kind(gain) == "gain"
    assert gate.failure_reason(gain) == (
        "c: gain, -19.37% instructions (limit -1%): update the baseline to keep the gain; "
        f"{gate.NATIVE_ONLY}"
    )
    worse = _with_work(measured=1_000_000 * (1 + gate.LAYOUT_SHIFT) + 1)
    assert gate.failure_kind(worse) == "regression"
    assert gate.failure_reason(worse).endswith(gate.NATIVE_ONLY)
    within = _with_work(measured=1_019_900)
    assert gate.failure_kind(within) == "layout only"


def test_the_failures_are_counted_by_kind():
    rows = [
        _with_work(),
        _with_work(measured=806_300),
        _verdict([806_300]),
        _verdict([1_005_000]),
        _verdict([1_000_000], sizes={"op_count": 16}),
    ]
    assert gate.failure_counts(rows) == "1 regression, 2 gains, 1 layout only"
    assert gate.failure_counts(rows[-1:]) == ""


def test_python_work_labels_only_a_moved_count_and_never_gates():
    assert _with_work(measured=1_000_000).verdict == "ok"
    assert _with_work({**_WORK, "bytecodes": 1.0}, measured=1_000_000).verdict == "ok"
    sizes = gate.compare(
        {"cases": {"c": {"op_count": 15, "python_work": _WORK}}}, {"c": {"op_count": 16}}, None
    )[0]
    assert gate.failure_reason(sizes) == "c: program sizes changed: op_count 15 -> 16"


def test_without_python_work_on_either_side_a_moved_count_is_not_labelled():
    row = _verdict([1_005_000])
    assert row.work_changes is None and row.work == ""
    assert gate.failure_reason(row) == "c: regression, +0.50% instructions (limit +0.4%)"
    baseline = _baseline()
    baseline["cases"]["c"]["python_work"] = _WORK
    runs = {"c": gate.Measured([1_005_000.0], [0.0])}
    unmeasured = gate.compare(baseline, {"c": {"op_count": 15}}, runs, None)[0]
    assert gate.failure_reason(unmeasured) == "c: regression, +0.50% instructions (limit +0.4%)"


def test_a_baseline_records_the_python_work_measured_with_its_counts():
    runs = {"c": gate.Measured([1_000_000.0], [2.0])}
    taken = gate.as_baseline({}, {"c": {"op_count": 15}}, runs, {}, {}, None, {"c": _WORK})
    assert taken["format"] == 2
    assert taken["cases"]["c"]["python_work"] == _WORK
    kept = gate.as_baseline({}, {"c": {"op_count": 15}}, None, taken, {})
    assert kept["cases"]["c"]["python_work"] == _WORK
    unlabelled = gate.as_baseline({}, {"c": {"op_count": 15}}, runs, taken, {}, None, None)
    assert "python_work" not in unlabelled["cases"]["c"]


def test_formats_1_and_2_are_read_and_others_are_errors():
    assert gate.baseline_error({"format": 1, **_baseline()}) is None
    assert gate.baseline_error({"format": 2, **_baseline()}) is None
    assert "format 3" in gate.baseline_error({"format": 3, **_baseline()})


# The fake-Valgrind harness: the gate's main() as CI runs it, with a fake
# valgrind (it writes one dump per case of instructions.json, of one
# repetition), a fake compiler for the marks, and a fake cases.py (its
# sizes, and the Python work of python_work.json, which varies from run to
# run while a file "varies" exists).
_FAKE_VALGRIND = """\
import json, pathlib, sys
here = pathlib.Path(__file__).parent
if sys.argv[1:] == ["--version"]:
    print("valgrind-3.22.0 (fake)")
    raise SystemExit
out = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--callgrind-out-file="))
counts = json.loads((here / "instructions.json").read_text())
for k, (name, ir) in enumerate(counts.items(), 1):
    pathlib.Path(f"{out}.{k}").write_text(
        "# callgrind format\\nversion: 1\\ncreator: callgrind-3.22.0\\n"
        f"desc: Trigger: Client Request: {name}\\n\\npositions: line\\nevents: Ir\\n"
        f"summary: {ir}\\n\\nfl=(1) a.c\\nfn=(1) f\\n1 {ir}\\n\\ntotals: {ir}\\n"
    )
print(json.dumps({name: 1 for name in counts}))
"""
_FAKE_CC = """\
import pathlib, sys
pathlib.Path(sys.argv[sys.argv.index("-o") + 1]).touch()
"""
_FAKE_CASES = """\
import json, os, pathlib, sys
here = pathlib.Path(__file__).parent
if "--structural" in sys.argv:
    print(json.dumps({"sizes": {"c": {"op_count": 15}}, "tolerances": {}}))
elif "--python-work" in sys.argv:
    work = json.loads((here / "python_work.json").read_text())
    if (here / "varies").exists():
        work["c"]["bytecodes"] += os.getpid()
    print(json.dumps(work))
"""


@pytest.fixture
def fake_gate(tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("valgrind", _FAKE_VALGRIND), ("cc", _FAKE_CC)):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n{text}")
        path.chmod(0o755)
    (tmp_path / "cases.py").write_text(_FAKE_CASES)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CC", str(bin_dir / "cc"))
    for name in ("GITHUB_ACTIONS", "GITHUB_RUN_ID", "GITHUB_REPOSITORY", "GITHUB_REF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gate, "CASES", tmp_path / "cases.py")
    monkeypatch.setattr(gate, "_byte_compile", lambda *args: None)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gate.platform, "machine", lambda: "x86_64")
    environment = gate._environment(sys.executable)

    def run(instructions, work, baseline_work=_WORK, form=2, source=None, layout=None):
        case = {"op_count": 15, "instructions": 1_000_000}
        if baseline_work is not None:
            case["python_work"] = baseline_work
        baseline = tmp_path / "perf_gate.json"
        taken = {"format": form, "environment": environment, "cases": {"c": case}}
        if source is not None:
            taken["source"] = source
        baseline.write_text(json.dumps(taken))
        draws = tmp_path / "perf_gate_layout.json"
        draws.unlink(missing_ok=True)
        if layout is not None:
            draws.write_text(json.dumps(layout))
        (bin_dir / "instructions.json").write_text(json.dumps({"c": instructions}))
        (tmp_path / "python_work.json").write_text(json.dumps({"c": work}))
        summary, out = tmp_path / "summary.md", tmp_path / "out.json"
        summary.unlink(missing_ok=True)
        argv = ["perf-gate.py", "--baseline", str(baseline), "--layout", str(draws)]
        argv += ["--python", sys.executable]
        argv += ["--pythonpath", str(tmp_path), "--summary", str(summary), "--out", str(out)]
        monkeypatch.setattr(sys, "argv", argv)
        code = gate.main()
        return code, capsys.readouterr().out, summary.read_text(), json.loads(out.read_text())

    run.varies = tmp_path / "varies"
    return run


def test_the_gate_labels_a_failure_whose_python_work_is_the_same(fake_gate):
    code, log, summary, _ = fake_gate(1_005_000, _WORK)
    assert code == 1
    line = f"c: layout only, +0.50% instructions (limit +0.4%); {gate.LAYOUT_ONLY}"
    assert f"  {line}\n" in log
    assert f"- {line}\n" in summary
    assert "| same | slower |" in summary
    assert "Perf gate failed (1 layout only):" in log
    assert "**Failed (1 layout only):**" in summary


def test_the_gate_shows_where_the_python_work_of_a_failure_changed(fake_gate):
    code, log, summary, _ = fake_gate(1_005_000, {**_WORK, "bytecodes": 452.6})
    assert code == 1
    line = (
        "c: regression, +0.50% instructions (limit +0.4%); Python work changed per "
        "repetition: bytecodes 449.6 -> 452.6 (+3), python_calls same, c_calls same"
    )
    assert f"  {line}\n" in log and f"- {line}\n" in summary
    assert gate.LAYOUT_ONLY not in log + summary
    assert "Perf gate failed (1 regression):" in log


def test_the_gate_reports_a_large_gain_as_a_gain_to_keep(fake_gate):
    code, log, summary, _ = fake_gate(806_300, {**_WORK, "bytecodes": 400.0})
    assert code == 1
    line = (
        "c: gain, -19.37% instructions (limit -1%): update the baseline to keep the gain; "
        "Python work changed per repetition: bytecodes 449.6 -> 400 (-49.6), "
        "python_calls same, c_calls same"
    )
    assert f"  {line}\n" in log and f"- {line}\n" in summary
    assert "Perf gate failed (1 gain):" in log and "**Failed (1 gain):**" in summary
    assert gate.LAYOUT_ONLY not in log + summary and "regression" not in log


def test_the_gate_fails_unlabelled_against_a_baseline_without_python_work(fake_gate):
    code, log, summary, taken = fake_gate(1_005_000, _WORK, baseline_work=None, form=1)
    assert code == 1
    assert "  c: regression, +0.50% instructions (limit +0.4%)\n" in log
    assert "- c: regression, +0.50% instructions (limit +0.4%)\n" in summary
    assert gate.LAYOUT_ONLY not in log + summary and "Python work changed" not in log
    assert "the baseline has no Python work" in log
    # The run taken as the next baseline brings it.
    assert taken["format"] == 2 and taken["cases"]["c"]["python_work"] == _WORK


def test_the_gate_passes_and_records_the_python_work_it_measured(fake_gate):
    code, log, summary, taken = fake_gate(1_000_000, _WORK)
    assert code == 0 and "Perf gate passed." in log
    assert taken["cases"]["c"]["python_work"] == _WORK


def test_python_work_that_differs_between_identical_runs_labels_nothing(fake_gate):
    fake_gate.varies.touch()
    code, log, summary, taken = fake_gate(1_005_000, _WORK)
    assert code == 1
    assert "two identical runs counted it differently (c)" in log
    assert "  c: regression, +0.50% instructions (limit +0.4%)\n" in log
    assert "python_work" not in taken["cases"]["c"]


def test_in_bootstrap_mode_a_case_without_a_count_is_reported_and_passes():
    baseline = {"bootstrap": True, "cases": {"c": {"op_count": 15}}}
    row = _verdict([1_000_000, 1_000_000], baseline)
    assert row.verdict == "bootstrap"
    assert row.measured == 1_000_000 and row.spread == 0
    assert "bootstrap" not in gate.FAILING


def test_in_bootstrap_mode_sizes_still_block():
    baseline = {"bootstrap": True, "cases": {"c": {"op_count": 15}}}
    assert _verdict([1_000_000], baseline, sizes={"op_count": 16}).verdict == "sizes"


def test_bootstrap_mode_is_allowed_only_while_no_case_has_a_count():
    assert gate.baseline_error({"bootstrap": True, "cases": {"c": {"op_count": 1}}}) is None
    assert gate.baseline_error(_baseline()) is None
    error = gate.baseline_error({"bootstrap": True, **_baseline()})
    assert error is not None and "remove" in error
    assert gate.baseline_error({"bootstrap": "yes", "cases": {}}) is not None


def test_bootstrap_mode_is_an_error_on_main():
    bootstrapping = {"bootstrap": True, "cases": {"c": {"op_count": 1}}}
    assert "main" in gate.baseline_error(bootstrapping, frozen=True)
    assert gate.baseline_error({"bootstrap": False, **_baseline()}, frozen=True) is not None
    assert gate.baseline_error(_baseline(), frozen=True) is None


def test_a_baseline_count_of_zero_is_an_error():
    assert "positive" in gate.baseline_error(_baseline(instructions=0))
    assert "positive" in gate.baseline_error(_baseline(instructions=None))


def test_taking_a_measured_run_as_the_baseline_ends_bootstrap_mode():
    previous = {"bootstrap": True, "environment": {}, "cases": {"c": {"op_count": 15}}}
    runs = {"c": gate.Measured([1_000_000.0, 1_000_000.0], [2.0, 2.0])}
    taken = gate.as_baseline({"python": "3.12.15"}, {"c": {"op_count": 15}}, runs, previous, {})
    assert "bootstrap" not in taken
    assert gate.baseline_error(taken) is None
    sizes_only = gate.as_baseline({}, {"c": {"op_count": 16}}, None, previous, {})
    assert sizes_only["bootstrap"] is True


BASELINE = json.loads(gate.BASELINE.read_text())


def test_the_checked_in_baseline_is_bootstrapping_or_has_every_count():
    assert gate.baseline_error(BASELINE) is None
    if BASELINE.get("bootstrap") is not True:
        missing = [n for n, c in BASELINE["cases"].items() if "instructions" not in c]
        assert not missing, f"no instruction count for {missing}"


def test_a_measured_baseline_is_of_the_locked_toolchain_and_names_its_run():
    # A new MLIR toolchain moves the counts: its pull request takes a new
    # baseline, and the old one stays in the file's history.
    if BASELINE.get("bootstrap") is True:
        pytest.skip("the baseline has no instruction counts yet")
    lock = json.loads((_ROOT / "tools" / "mlir-toolchain.lock.json").read_text())
    assert BASELINE["environment"]["mlir_toolchain"] == lock["release_tag"]
    assert {"repository", "run", "commit", "processes"} <= set(BASELINE["source"])


def test_a_case_of_its_own_tolerance_still_fails_a_half_percent_regression():
    # The baseline's tolerances are the cases' own (the size test below).
    own = {n: c["tolerance"] for n, c in BASELINE["cases"].items() if "tolerance" in c}
    assert "simulate_setup" in own
    for name, tolerance in own.items():
        assert gate.REGRESSION < tolerance < 0.005, name


def test_every_case_gates_at_least_one_program_size():
    sized = {
        name: [k for k in case if k not in gate.NOT_SIZES]
        for name, case in BASELINE["cases"].items()
    }
    assert not [name for name, keys in sized.items() if not keys]


def test_the_program_sizes_match_the_baseline(tmp_path):
    env = gate._run_env(str(Path(cudaq_pulse.__file__).parents[1]), str(tmp_path))
    # Under the sanitizers the extension needs their runtime preloaded.
    for name in ("LD_PRELOAD", "ASAN_OPTIONS", "UBSAN_OPTIONS", "TSAN_OPTIONS", "LSAN_OPTIONS"):
        if name in os.environ:
            env[name] = os.environ[name]
    sizes, tolerances = gate._structural(sys.executable, env)
    assert gate.tolerance_error(BASELINE, tolerances) is None
    expected = {
        name: {k: v for k, v in case.items() if k not in gate.NOT_SIZES}
        for name, case in BASELINE["cases"].items()
    }
    assert sizes == expected, "update with: python tools/perf-gate.py --update"


def test_the_workload_op_counts_agree_with_compile_time_json():
    workloads = json.loads((_ROOT / "benchmarks" / "baselines" / "compile_time.json").read_text())
    for name, entry in workloads.items():
        assert BASELINE["cases"][f"compile_{name}"]["op_count"] == entry["op_count"], name


# Layout bands. The recorded rows of CI runs against the baseline of run
# 44179125553 (commit 933e24f5), for the cases they concern, and the distinct
# counts its era's runs on its measured code gave each case (52 runs).
_B0_RUN, _B0_COMMIT = 44179125553, "933e24f5528663f07b305d41e6762fe5cdab700b"
_AMP = {"bytecodes": 439.6, "python_calls": 16.0, "c_calls": 16.1}
_WAIT = {"bytecodes": 640.6, "python_calls": 22.0, "c_calls": 37.1}
_CTRL = {"bytecodes": 1339.0, "python_calls": 29.0, "c_calls": 74.667}
_SAMPLED = {"bytecodes": 5266.0, "python_calls": 40.0, "c_calls": 161.667}
_B0 = {
    "format": 2,
    "source": {"run": _B0_RUN, "commit": _B0_COMMIT},
    "cases": {
        "specialize_amplitude": {"mlir_bytes": 1303, "instructions": 409188, "python_work": _AMP},
        "specialize_wait": {
            "schedule_cycles": 145,
            "schedule_events": 3,
            "instructions": 360385,
            "python_work": _WAIT,
        },
        "export_to_controller": {
            "export_bytes": 87794,
            "instructions": 4725299,
            "python_work": _CTRL,
        },
        "export_to_controller_sampled": {
            "export_bytes": 49449,
            "instructions": 1519377,
            "python_work": _SAMPLED,
        },
    },
}
# fmt: off
_B0_DRAWS = {
    "specialize_amplitude": [
        406094, 407132, 407753, 407754, 407837, 407873, 407951, 407976, 408019, 408115,
        408243, 408506, 408919, 409188, 409420, 410053, 410644, 411144, 411731,
    ],
    "specialize_wait": [
        356647, 356688, 357523, 357589, 357616, 357703, 357780, 357800, 357836, 357856,
        358144, 358310, 358430, 358906, 358922, 359026, 359228, 360001, 360385,
    ],
    "export_to_controller": [4725299, 4728927, 4729579, 4732543, 4733822, 4734665, 4735729],
    "export_to_controller_sampled": [1516389, 1517349, 1517430, 1517835, 1518095, 1518182, 1519377],
}
# fmt: on
_CTRL_107 = {"bytecodes": 1358.0, "python_calls": 29.0, "c_calls": 75.667}
_SAMPLED_107 = {"bytecodes": 5285.0, "python_calls": 40.0, "c_calls": 162.667}
# PR: (run, its counts of the four cases; the export cases' Python work
# is #107's, which merged before #98, #89 and #94 ran).
_RECORDED = {
    98: (44488536412, (411144, 357800, 4734947, 1519008), True),
    63: (44258997277, (411731, 356647, 4734665, 1517430), False),
    89: (44402542327, (410644, 358430, 4734348, 1519899), True),
    94: (44410403238, (406094, 356688, 4731508, 1519372), True),
    106: (44446239417, (408019, 357410, 4698206, 1512776), True),
    107: (44446244619, (407971, 357571, 4788958, 1529702), True),
    109: (44456030769, (409555, 357687, 4739186, 1520159), True),
}
# What each native change's tree differs from the baseline's in.
_MEASURED = {
    106: "core/mlir/include/cudaq-pulse/Analysis/LineTrace.h",
    107: "core/mlir/bindings/ControllerProgram.h, core/mlir/bindings/bindings.cpp",
    109: "core/mlir/bindings/ControllerProgram.h, core/mlir/bindings/bindings.cpp, "
    "core/mlir/conversions/PulseToQOp/PulseToQOp.cpp and 10 more",
}


def _layout(draws=_B0_DRAWS, run=_B0_RUN, commit=_B0_COMMIT, **extra):
    runs = [{"run": i, "counts": {c: v}} for c, vs in draws.items() for i, v in enumerate(vs)]
    return {"format": 1, "baseline": {"run": run, "commit": commit}, "runs": runs, **extra}


def _recorded(pr, scope="", amplitude=None, wait=None, amp_work=_AMP, layout=None):
    run, counts, export_moved = _RECORDED[pr]
    counts = (amplitude or counts[0], wait or counts[1], *counts[2:])
    names = list(_B0["cases"])
    work = [amp_work, _WAIT, *((_CTRL_107, _SAMPLED_107) if export_moved else (_CTRL, _SAMPLED))]
    sizes = {
        n: {k: v for k, v in c.items() if k not in gate.NOT_SIZES} for n, c in _B0["cases"].items()
    }
    runs = {n: gate.Measured([float(v)], [0.0]) for n, v in zip(names, counts, strict=True)}
    bands, why = gate.layout_bands(_layout() if layout is None else layout, _B0)
    rows = gate.compare(_B0, sizes, runs, dict(zip(names, work, strict=True)), bands, scope)
    failures = [gate.failure_reason(r) for r in rows if r.verdict in gate.FAILING]
    return failures, gate.layout_notes(rows), why


def test_the_bands_of_the_recorded_draws():
    bands, why = gate.layout_bands(_layout(), _B0)
    assert why == "" and set(bands) == {"specialize_amplitude", "specialize_wait"}
    shown = {n: (gate._band(b), gate._band(b[2:])) for n, b in bands.items()}
    assert shown == {
        "specialize_amplitude": ("-1.10% to +0.83%", "-0.76% to +0.62%"),
        "specialize_wait": ("-1.49% to +0.32%", "-1.04% to +0.00%"),
    }


@pytest.mark.parametrize(
    ("pr", "notes"),
    [
        (98, ["specialize_amplitude: layout, +0.48% instructions (limit +0.4%; layout band "
              "-1.10% to +0.83%, draws -0.76% to +0.62%)"]),
        (63, ["specialize_amplitude: layout, +0.62% instructions (limit +0.4%; layout band "
              "-1.10% to +0.83%, draws -0.76% to +0.62%)",
              "specialize_wait: layout, -1.04% instructions (limit -1%; layout band "
              "-1.49% to +0.32%, draws -1.04% to +0.00%)"]),
        (94, ["specialize_wait: layout, -1.03% instructions (limit -1%; layout band "
              "-1.49% to +0.32%, draws -1.04% to +0.00%)"]),
        (89, []),
    ],
)  # fmt: skip
def test_the_recorded_layout_shifts_pass_with_a_notice(pr, notes):
    failures, shown, _ = _recorded(pr)
    assert failures == [] and shown == notes
    # Today's limits fail every one with a notice.
    today, _, _ = _recorded(pr, layout={})
    assert len(today) == len(notes)
    assert all("layout only" in line for line in today)


@pytest.mark.parametrize("pr", [98, 63, 94])
@pytest.mark.parametrize(
    "scope",
    [
        "the measured code differs from the baseline's: core/mlir/transforms/PulseVerify.cpp",
        "git cannot compare this tree with the baseline's commit 933e24f5",
    ],
)
def test_a_recorded_layout_shift_fails_on_other_measured_code(pr, scope):
    failures, notes, _ = _recorded(pr, scope)
    assert failures and not notes
    for line in failures:
        assert ", within its layout band " in line and f"but {scope}: the limits apply;" in line


def test_off_the_baselines_measured_code_a_row_outside_its_band_fails_as_today():
    scope = "the measured code differs from the baseline's: core/mlir/lib/Foo.cpp"
    amplitude = round(409188 * 1.010)
    failures, notes, _ = _recorded(63, scope, amplitude=amplitude)
    today, _, _ = _recorded(63, layout={}, amplitude=amplitude)
    # Outside its band (not applied): today's text; within it: why it fails.
    assert not notes and len(failures) == 2 and failures[0] == today[0]
    assert failures[0].startswith("specialize_amplitude: layout only, +1.00% instructions")
    assert ", within its layout band -1.49% to +0.32%, but " in failures[1]


@pytest.mark.parametrize("pr", [106, 107, 109])
def test_native_changes_are_judged_by_the_limits(pr):
    scope = f"the measured code differs from the baseline's: {_MEASURED[pr]}"
    failures, notes, _ = _recorded(pr, scope)
    assert failures == _recorded(pr, layout={})[0] and not notes
    if pr == 107:
        assert failures == [
            "export_to_controller: regression, +1.35% instructions (limit +0.4%); Python work "
            "changed per repetition: bytecodes 1,339 -> 1,358 (+19), python_calls same, "
            "c_calls 74.667 -> 75.667 (+1)",
            "export_to_controller_sampled: regression, +0.68% instructions (limit +0.4%); "
            "Python work changed per repetition: bytecodes 5,266 -> 5,285 (+19), "
            "python_calls same, c_calls 161.667 -> 162.667 (+1)",
        ]
    else:
        assert failures == []


def test_a_shift_beyond_every_draw_but_in_the_band_says_so():
    # The hole: native work behind the same Python work passes up to the edge.
    failures, notes, _ = _recorded(98, amplitude=round(409188 * 1.008))
    assert failures == []
    assert notes == [
        "specialize_amplitude: layout, +0.80% instructions (limit +0.4%; layout band "
        "-1.10% to +0.83%, draws -0.76% to +0.62%, beyond every draw so far)"
    ]


@pytest.mark.parametrize(("amplitude", "wait"), [(1.010, None), (None, 0.981)])
def test_a_shift_outside_the_band_fails_on_either_side(amplitude, wait):
    failures, notes, _ = _recorded(
        63,
        amplitude=amplitude and round(409188 * amplitude),
        wait=wait and round(360385 * wait),
    )
    assert len(failures) == 1 and len(notes) == 1
    band = "-1.10% to +0.83%" if amplitude else "-1.49% to +0.32%"
    assert (
        f"(limit {'+0.4%' if amplitude else '-1%'}), outside its layout band {band};"
        in (failures[0])
    )


def test_changed_python_work_keeps_the_limits():
    failures, notes, _ = _recorded(98, amp_work={**_AMP, "bytecodes": 442.6})
    assert not notes
    assert failures == [
        "specialize_amplitude: regression, +0.48% instructions (limit +0.4%); Python work "
        "changed per repetition: bytecodes 439.6 -> 442.6 (+3), python_calls same, c_calls same"
    ]


def test_without_draws_for_the_baseline_the_limits_apply():
    for layout, why in [
        ({}, "there are no layout draws"),
        (_layout(run=1), "the layout draws are for another baseline"),
        (_layout(commit="b" * 40), "the layout draws are for another baseline"),
        (
            _layout(baseline={"run": _B0_RUN, "attempt": 2, "commit": _B0_COMMIT}),
            "the layout draws are for another baseline",
        ),
        ({**_layout(), "format": 2}, "the layout draws are of format 2, not 1"),
    ]:
        failures, notes, shown = _recorded(63, layout=layout)
        assert len(failures) == 2 and not notes and shown.startswith(why)


def test_the_fences_are_tukeys_on_distinct_draws_within_the_layout_shift():
    assert gate.FENCE == 2 and gate.MIN_DRAWS == 8
    assert gate.fences([1000, 1001, 1002, 1003, 1004, 1005, 1006] * 3, 1000) is None
    # Quartiles 1001.5 and 1005.5: fences 993.5 and 1013.5.
    band = gate.fences([1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007], 1000)
    assert band == pytest.approx((-0.0065, 0.0135, 0.0, 0.007))
    wide = gate.fences([960, 970, 980, 990, 1010, 1020, 1030, 1040], 1000)
    assert wide[:2] == (-gate.LAYOUT_SHIFT, gate.LAYOUT_SHIFT)


def test_a_carried_draw_is_shifted_by_its_change_and_a_change_not_carried_drops_it():
    layout = {
        "carried": [{"commit": "c1", "moves": {"a": -10, "b": None}}],
        "runs": [
            {"run": 1, "counts": {"a": 100, "b": 200}},
            {"run": 2, "across": ["c1"], "counts": {"a": 110, "b": 210, "c": 5}},
            {"run": 3, "across": ["unknown"], "counts": {"a": 120}},
        ],
    }
    assert gate.layout_draws(layout) == {"a": [100, 100], "b": [200]}


def _git(repo, *args):
    import subprocess

    command = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    command += ["-c", "commit.gpgsign=false", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


_PYPROJECT = """\
[build-system]
requires = ["scikit-build-core"]
[dependency-groups]
build = ["numpy"]
test = ["pytest"]
"""
_UV_LOCK = """\
version = 1
[[package]]
name = "cudaq-pulse"
source = {{ virtual = "." }}
dependencies = [{{ name = "numpy" }}]
[package.dev-dependencies]
build = [{{ name = "numpy" }}]
test = [{{ name = "pytest" }}]
[[package]]
name = "numpy"
version = "{numpy}"
[[package]]
name = "pytest"
version = "{pytest}"
"""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository whose first commit is the baseline's; commit(**files)
    adds a commit (None removes a file) and returns its id."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")

    def commit(**files):
        for name, text in files.items():
            path = root / name.replace("__", "/")
            if text is None:
                _git(root, "rm", "-q", str(path))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "--allow-empty", "-m", "c")
        return _git(root, "rev-parse", "HEAD")

    commit.root = root
    commit.base = commit(
        **{
            "core__frontend__cudaq_pulse__k.py": "x = 1\n",
            "core__mlir__lib__Foo.cpp": "int f();\n",
            "pyproject.toml": _PYPROJECT,
            "uv.lock": _UV_LOCK.format(numpy="2.0", pytest="8.0"),
            "benchmarks__baselines__perf_gate.json": "{}\n",
        }
    )
    # The gate's git runs in it; the environment is still this checkout's.
    environment = gate._environment(sys.executable)
    monkeypatch.setattr(gate, "_environment", lambda python: environment)
    monkeypatch.setattr(gate, "ROOT", root)
    return commit


def test_python_sources_and_files_the_job_never_reads_keep_the_measured_code(repo):
    repo(
        **{
            "core__frontend__cudaq_pulse__k.py": "x = 2\n",
            "tests__test_k.py": "",
            "docs__k.rst": "",
            "CHANGELOG.md": "",
            ".github__workflows__docs.yml": "",
            "tools__check-k.py": "",
        }
    )
    assert gate.measured_scope(repo.base) == ""
    # A baseline commit, and one of its draws, leave the measured code.
    repo(**{"benchmarks__baselines__perf_gate.json": '{"a": 1}\n'})
    repo(**{"benchmarks__baselines__perf_gate_layout.json": "{}\n"})
    assert gate.measured_scope(repo.base) == ""
    # Dev dependencies and tool settings are not built.
    repo(**{"uv.lock": _UV_LOCK.format(numpy="2.0", pytest="9.0")})
    repo(**{"pyproject.toml": _PYPROJECT + "[tool.ruff]\nline-length = 99\n"})
    assert gate.measured_scope(repo.base) == ""


@pytest.mark.parametrize(
    ("files", "paths"),
    [
        ({"core__mlir__lib__Foo.cpp": "int g();\n"}, "core/mlir/lib/Foo.cpp"),
        ({".github__workflows__ci.yml": ""}, ".github/workflows/ci.yml"),
        ({"tools__perf-gate.py": ""}, "tools/perf-gate.py"),
        ({"benchmarks__perf_gate__cases.py": ""}, "benchmarks/perf_gate/cases.py"),
        ({"core__frontend__cudaq_pulse__data.json": ""}, "core/frontend/cudaq_pulse/data.json"),
        ({"uv.lock": _UV_LOCK.format(numpy="2.1", pytest="8.0")}, "uv.lock"),
        ({"pyproject.toml": _PYPROJECT.replace('"numpy"', '"numpy", "cmake"')}, "pyproject.toml"),
        # A rename into tests/ lists both paths.
        (
            {"core__mlir__lib__Foo.cpp": None, "tests__Foo.cpp": "int f();\n"},
            "core/mlir/lib/Foo.cpp",
        ),
    ],
)
def test_any_other_path_is_measured_code(repo, files, paths):
    repo(**files)
    assert gate.measured_scope(repo.base) == (
        f"the measured code differs from the baseline's: {paths}"
    )


def test_every_entry_of_a_package_locked_twice_is_measured(repo):
    # A lock split by Python version locks numpy twice; either entry counts.
    def lock(old):
        entry = f'[[package]]\nname = "numpy"\nversion = "{old}"\n'
        entry += "resolution-markers = [\"python_full_version < '3.11'\"]\n"
        text = _UV_LOCK.format(numpy="2.0", pytest="8.0")
        return text.replace('[[package]]\nname = "numpy"', entry + '[[package]]\nname = "numpy"')

    split = repo(**{"uv.lock": lock("1.26")})
    repo(**{"uv.lock": lock("1.25")})
    assert gate.measured_scope(split) == "the measured code differs from the baseline's: uv.lock"


def test_the_measured_code_fails_closed(repo, tmp_path, monkeypatch):
    assert gate.measured_scope(None) == "the baseline names no commit"
    missing = "0123456789abcdef0123456789abcdef01234567"
    assert gate.measured_scope(missing) == (
        "git cannot compare this tree with the baseline's commit 01234567"
    )
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    assert gate.measured_scope(repo.base).startswith("git cannot compare")


def _layout_for_c(commit, run=_B0_RUN):
    draws = [990_000, 994_000, 997_000, 999_000, 1_000_000]
    draws += [1_002_000, 1_004_000, 1_006_000, 1_008_000]
    return _layout({"c": draws}, run, commit)


def test_the_gate_passes_a_layout_shift_on_the_baselines_measured_code(fake_gate, repo):
    repo(**{"core__frontend__cudaq_pulse__k.py": "x = 2\n"})
    source = {"run": _B0_RUN, "commit": repo.base}
    code, log, summary, _ = fake_gate(
        1_005_000, _WORK, source=source, layout=_layout_for_c(repo.base)
    )
    note = (
        "c: layout, +0.50% instructions (limit +0.4%; layout band -2.00% to +2.00%, draws "
        "-1.00% to +0.80%)"
    )
    assert code == 0 and log.endswith("Perf gate passed (1 layout, within the band).\n")
    assert f"Layout, within the band:\n  {note}\n  ({gate.IN_BAND})" in log
    assert f"**Layout, within the band (1).** {note}. In each, {gate.IN_BAND}" in summary
    assert "| same | layout |" in summary
    assert "passes within its case's layout band (verdict layout)" in summary


def test_the_gate_judges_a_layout_shift_on_other_measured_code_by_the_limits(fake_gate, repo):
    repo(**{"core__mlir__lib__Foo.cpp": "int g();\n"})
    source = {"run": _B0_RUN, "commit": repo.base}
    code, log, summary, _ = fake_gate(
        1_005_000, _WORK, source=source, layout=_layout_for_c(repo.base)
    )
    scope = "the measured code differs from the baseline's: core/mlir/lib/Foo.cpp"
    assert code == 1 and "Perf gate failed (1 layout only):" in log
    assert f"within its layout band -2.00% to +2.00%, but {scope}: the limits apply" in log
    assert f"Layout bands not applied: {scope}; every row is judged by the limits." in summary


def test_the_gate_without_git_judges_by_the_limits(fake_gate, repo, tmp_path, monkeypatch):
    (tmp_path / "plain").mkdir()
    monkeypatch.setattr(gate, "ROOT", tmp_path / "plain")
    source = {"run": _B0_RUN, "commit": repo.base}
    code, log, _, _ = fake_gate(1_005_000, _WORK, source=source, layout=_layout_for_c(repo.base))
    assert code == 1 and "Perf gate failed (1 layout only):" in log
    assert f"git cannot compare this tree with the baseline's commit {repo.base[:8]}" in log
