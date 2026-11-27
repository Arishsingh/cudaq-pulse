# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which CI jobs a change runs: the classifier of tools/ci-changes.py and the
conditions the workflows put on their jobs (source tree only)."""

import importlib.util
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "ci-changes.py"
_WORKFLOWS = _ROOT / ".github" / "workflows"
pytestmark = pytest.mark.skipif(not _TOOL.exists(), reason="needs the source tree's tools/")


def _classify(paths):
    spec = importlib.util.spec_from_file_location("ci_changes", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.classify(paths)


META, DOCS, PYTHON, NATIVE = (
    {"native": False, "python": False, "docs": False},
    {"native": False, "python": False, "docs": True},
    {"native": False, "python": True, "docs": True},
    {"native": True, "python": True, "docs": True},
)


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        # The repository's own files: lint alone.
        (["CITATION.cff"], META),
        (["CODE_OF_CONDUCT.md", "SUPPORT.md", ".github/ISSUE_TEMPLATE/bug.yml"], META),
        # The pages, and the two files the pages or their tests read.
        (["docs/tutorials/coherence.rst"], DOCS),
        (["README.md"], DOCS),
        (["CHANGELOG.md", "CITATION.cff"], DOCS),
        # Files that tests read: the contributing guide, the license files.
        (["CONTRIBUTING.md"], DOCS),
        (["LICENSE", "LICENSES/Apache-2.0.txt", "NOTICE", "THIRD_PARTY_NOTICES"], DOCS),
        (["SECURITY.md"], DOCS),
        (["docs/_ext/a11y.py", "docs/_static/a11y.css"], DOCS),
        # The Python package, its tests and what runs them: a docstring too.
        (["core/frontend/cudaq_pulse/ops/__init__.py"], PYTHON),
        (["tests/test_units.py"], PYTHON),
        (["docs/index.rst", "tests/test_docs_a11y.py"], PYTHON),
        (["uv.lock"], PYTHON),
        (["tools/perf-gate.py", "benchmarks/perf_gate/cases.py"], PYTHON),
        # The compiler, the runtime, the build and the workflows: everything.
        (["core/mlir/transforms/ScheduleAlap.cpp"], NATIVE),
        (["CMakeLists.txt"], NATIVE),
        (["test/Dialect/Pulse/ops.mlir"], NATIVE),
        ([".github/workflows/ci.yml"], NATIVE),
        ([".github/actions/changes/action.yml"], NATIVE),
        (["README.md", "core/mlir/bindings/bindings.cpp"], NATIVE),
        # What the tool cannot place runs everything; so does no list at all.
        (["a_new_directory/file.txt"], NATIVE),
        (["docs.txt"], NATIVE),
        (["README.md.orig"], NATIVE),
        ([], NATIVE),
        (["", "  "], NATIVE),
    ],
)
def test_a_change_is_classed_by_the_widest_thing_it_touches(paths, expected):
    assert _classify(paths) == expected


def test_every_top_level_entry_of_the_repository_is_placed_on_purpose():
    """A new top-level file or directory is native until someone places it;
    this test names the ones that are, so that a move is a decision."""
    classes = {
        name: _classify([name + "/x" if (_ROOT / name).is_dir() else name]) for name in _tracked()
    }
    assert {name for name, c in classes.items() if c == NATIVE} == {
        *(".clang-format", ".github", "CMakeLists.txt", "CMakePresets.json", "cmake", "core"),
        *("packaging", "test"),
    }
    assert _classify(["core/frontend/cudaq_pulse/compiler.py"]) == PYTHON
    assert _classify([".github/ISSUE_TEMPLATE/bug.yml"]) == META


def _tracked():
    import subprocess

    out = subprocess.run(
        ["git", "ls-tree", "--name-only", "HEAD"], cwd=_ROOT, capture_output=True, text=True
    )
    if out.returncode:
        pytest.skip("not a git checkout")
    return out.stdout.split()


def _jobs(workflow):
    """Each job's ``needs`` and ``if`` as text, read without a YAML parser:
    a job starts at a two-space key and holds its lines up to the next."""
    text = (_WORKFLOWS / workflow).read_text(encoding="utf-8")
    body = text[text.index("\njobs:\n") + len("\njobs:\n") :]
    jobs = {}
    starts = [m.start() for m in re.finditer(r"(?m)^  [A-Za-z][\w-]*:\n", body)]
    for start, end in zip(starts, [*starts[1:], len(body)], strict=True):
        part = body[start:end]
        name = part.split(":", 1)[0].strip()
        needs = re.search(r"(?m)^    needs: (.*)$", part)
        condition = re.search(r"(?ms)^    if: (.*?)\n    (?=[a-z-]+:)", part)
        jobs[name] = (
            needs.group(1) if needs else "",
            " ".join(condition.group(1).split()) if condition else "",
        )
    return jobs


# The class a job waits for; lint and the summary wait for none.
_CI = {
    "test": "python == 'true'",
    "perf-gate": "python == 'true'",
    "editor-types": "python == 'true'",
    "test-free-threaded": "python == 'true'",
    "old-compilers": "native == 'true'",
    "test-docs": "docs == 'true'",
}


def test_each_ci_job_runs_for_the_class_that_can_break_it():
    jobs = _jobs("ci.yml")
    assert set(_CI) | {"changes", "lint", "ci-ok", "gpu"} == set(jobs), (
        "a new job needs a class here"
    )
    for job, wanted in _CI.items():
        needs, condition = jobs[job]
        assert "changes" in needs, job
        assert f"needs.changes.outputs.{wanted}" in condition, (job, condition)
    # The pages' tests run alone only when no test job runs them anyway.
    assert "needs.changes.outputs.python != 'true'" in jobs["test-docs"][1]
    # Lint runs for every change.
    assert "needs.changes" not in jobs["lint"][1] and "changes" not in jobs["lint"][0]


def test_the_perf_gate_runs_for_every_change_that_can_move_it():
    """The gate measures the compiler and the Python frontend: a change of
    either class runs it, and python is true for both."""
    assert _classify(["core/mlir/bindings/bindings.cpp"])["python"]
    assert _classify(["core/frontend/cudaq_pulse/kernel/_kernel.py"])["python"]
    assert _classify(["benchmarks/baselines/perf_gate.json"])["python"]
    assert _classify(["tools/perf-gate.py"])["python"]
    assert "needs.changes.outputs.python == 'true'" in _jobs("ci.yml")["perf-gate"][1]


def test_the_summary_waits_for_every_job_and_always_runs():
    jobs = _jobs("ci.yml")
    needs, condition = jobs["ci-ok"]
    waited = set(re.findall(r"[\w-]+", needs))
    # gpu runs elsewhere.
    assert waited == set(jobs) - {"ci-ok", "gpu"}
    assert condition.startswith("${{ always()")


def _summary(results):
    """The summary job's own script, run on *results*."""
    text = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    script = text[
        text.index("python3 - <<'PY'\n") + len("python3 - <<'PY'\n") : text.index(
            "\n          PY\n"
        )
    ]
    script = "\n".join(line[10:] for line in script.splitlines())
    import json
    import subprocess
    import sys

    needs = json.dumps({job: {"result": result} for job, result in results.items()})
    return subprocess.run(
        [sys.executable, "-c", script], env={"NEEDS": needs}, capture_output=True, text=True
    )


_RAN = {"changes": "success", "lint": "success"}


@pytest.mark.parametrize(
    ("results", "passes"),
    [
        # A docs-only change: the gated jobs are skipped.
        ({**_RAN, "test": "skipped", "test-docs": "success", "perf-gate": "skipped"}, True),
        ({**_RAN, "test": "success", "test-docs": "skipped", "old-compilers": "success"}, True),
        ({**_RAN, "test": "failure", "test-docs": "skipped"}, False),
        ({**_RAN, "test": "success", "perf-gate": "cancelled"}, False),
        ({**_RAN, "lint": "failure", "test": "skipped"}, False),
        # Without the classifier every job is skipped, which must not pass.
        ({"changes": "failure", "lint": "success", "test": "skipped"}, False),
        ({"changes": "skipped", "lint": "success", "test": "skipped"}, False),
    ],
)
def test_the_summary_fails_when_a_job_that_ran_failed_and_passes_on_skips(results, passes):
    done = _summary(results)
    assert (done.returncode == 0) == passes, done.stderr


@pytest.mark.parametrize(
    ("workflow", "job", "wanted"),
    [("docs.yml", "build", "docs"), ("sanitizers.yml", "sanitizers", "python")],
)
def test_the_other_workflows_gate_their_job_on_the_same_classes(workflow, job, wanted):
    needs, condition = _jobs(workflow)[job]
    assert "changes" in needs
    assert f"needs.changes.outputs.{wanted} == 'true'" in condition


def test_the_nightly_hash_seeds_are_drawn_each_run_logged_and_counted():
    """Two seeds are drawn in the run, not written in the workflow; each has
    a job of its own that runs the suite under it and names it; and the
    nightly issue waits for both and gives the seeds."""
    text = (_WORKFLOWS / "property-nightly.yml").read_text(encoding="utf-8")
    jobs = _jobs("property-nightly.yml")
    # Every job but the report-only coverage counts toward the issue.
    waited = set(re.findall(r"[\w-]+", jobs["nightly-issue"][0]))
    assert waited == set(jobs) - {"nightly-issue", "coverage"}
    assert {"seeds", "hash-seeds"} <= waited
    assert jobs["hash-seeds"][0] == "seeds" and jobs["seeds"][0] == ""

    def job(name):
        return re.split(r"\n\n  (?=[#A-Za-z])", text[text.index(f"\n  {name}:\n") + 1 :])[0]

    draw = job("seeds")
    assert "shuf -i 1-4294967295 -n 2" in draw
    assert 'echo "first=${seeds[0]}"' in draw and 'echo "second=${seeds[1]}"' in draw
    assert "::notice title=PYTHONHASHSEED::" in draw and "$GITHUB_STEP_SUMMARY" in draw
    assert not re.search(r"PYTHONHASHSEED[=:] *\d", text), "a seed written in the workflow"
    suite = job("hash-seeds")
    for output in ("first", "second"):
        assert f"          - ${{{{ needs.seeds.outputs.{output} }}}}\n" in suite
    assert suite.split("\n    needs:")[0].endswith("PYTHONHASHSEED=${{ matrix.seed }})")
    assert "          PYTHONHASHSEED: ${{ matrix.seed }}\n" in suite
    assert 'if [ -z "$PYTHONHASHSEED" ]; then' in suite
    assert 'pytest tests -q -m "not gpu" -p no:cacheprovider' in suite
    assert "continue-on-error" not in draw + suite
    assert "fail-fast: false" in suite and re.search(r"(?m)^    timeout-minutes: \d+$", suite)
    # The test legs no longer run the suite a second and third time.
    assert "hash-seeds: true" not in text and "for seed in" not in text
    issue = job("nightly-issue")
    for name in ("first", "second"):
        assert f"{name.upper()}_SEED: ${{{{ needs.seeds.outputs.{name} }}}}\n" in issue
    assert "were given the hash seeds $FIRST_SEED and $SECOND_SEED" in issue
    assert issue.count('"$seeds" \\\n') == 1


def _module():
    spec = importlib.util.spec_from_file_location("ci_changes", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_test_reads_a_file_that_runs_lint_alone():
    """A file of META runs no test, so no test may read one."""
    names = [name.rstrip("/") for name in _module().META if not name.startswith(".github/")]
    names.remove(".gitignore")
    readers = {
        (path.name, name)
        for path in (_ROOT / "tests").rglob("*.py")
        if path.name != "test_ci_changes.py"
        for name in names
        if re.search(rf"[\"'/]{re.escape(name)}[\"']", path.read_text(encoding="utf-8"))
    }
    assert not readers


def test_the_pages_job_runs_every_test_that_reads_a_file_of_the_docs_class():
    """test-docs runs alone for a change of the docs class: each test file
    that names one of those files is in its list, or docs.yml runs it."""
    ci = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    listed = set(
        re.findall(r"tests/[\w/]+\.py", ci[ci.index("  test-docs:") : ci.index("  ci-ok:")])
    )
    listed |= set(
        re.findall(r"tests/[\w/]+\.py", (_WORKFLOWS / "docs.yml").read_text(encoding="utf-8"))
    )
    names = [name for name in _module().DOCS if name != "docs/"]
    pattern = "|".join(re.escape(name.rstrip("/")) for name in names)
    reads = re.compile(rf"[\"'/]({pattern})[\"'/]|\"docs\"|'docs'|/ \"docs\"|docs/[\w/]+\.rst")
    missing = {
        path.relative_to(_ROOT).as_posix()
        for path in (_ROOT / "tests").glob("*.py")
        if reads.search(path.read_text(encoding="utf-8"))
    } - listed
    assert not missing


def test_only_a_pull_request_is_classed_and_a_rename_counts_twice():
    action = (_ROOT / ".github/actions/changes/action.yml").read_text(encoding="utf-8")
    assert '"$EVENT" != pull_request ]] ||' in action
    assert "git diff --name-only --no-renames" in action
    assert action.count("ci-changes.py --all") == 1


def test_the_summary_has_the_required_name_only_where_it_is_the_whole_run():
    ci = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    job = ci[ci.index("  ci-ok:\n") :]
    assert "github.event_name == 'pull_request' || github.ref == 'refs/heads/main'" in job
    assert "!inputs.perf_baseline && 'ci-ok' ||" in job
    done = _summary({"changes": "skipped", "lint": "skipped", "test": "skipped"})
    assert done.returncode != 0
