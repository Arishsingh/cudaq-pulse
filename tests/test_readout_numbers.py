# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each readout keeps the number and the loop iteration the kernel gave it
under every schedule policy.

asap and rcp peel a loop's first iterations off, a loop may lose its last
ones or run several iterations in each, and simulation unrolls loops that
make calls: each copies the loop's readouts. A copy reports the kernel's
number and iteration, so ``SimulationResult.measurements`` reads the same
under asap, alap, rcp and alap_rcp, and a measurement a kernel returns
after a loop of readouts is found. The exports describe the scheduled
program, loop by loop as it is scheduled.
"""

from __future__ import annotations

import functools
from typing import Any

import cudaq_pulse as pulse
import pytest

from tests._handles import Q

POLICIES = ("asap", "alap", "rcp", "alap_rcp")
ROLLED, UNROLLED = 0, 10**6


@functools.cache
def _qudits() -> tuple[Any, Any]:
    system = pulse.quantum_system("transmon_surface17")
    return system.qudits[0], system.qudits[1]


@pulse.kernel
def after_loop(data, ancilla):
    # Three X on the data qudit, an ancilla readout in each round; the data
    # qudit's readout, after the loop, is returned.
    d = get_drive_line(data)
    a = get_drive_line(ancilla)
    r = get_readout_line(ancilla)
    rd = get_readout_line(data)
    x = d.get_calibration(pulse.gates.X)
    m = r.get_calibration(pulse.gates.READOUT)
    md = rd.get_calibration(pulse.gates.READOUT)
    for _ in range(3):
        drive(d, x)
        sync(d, a, r)
        readout(r, m)
    sync(d, rd)
    return readout(rd, md)


@pulse.kernel
def around_loop(data, ancilla):
    # The data qudit is read before and after a loop that flips the ancilla
    # and reads it each round; it is flipped once in between.
    d = get_drive_line(data)
    a = get_drive_line(ancilla)
    r = get_readout_line(ancilla)
    rd = get_readout_line(data)
    xd = d.get_calibration(pulse.gates.X)
    xa = a.get_calibration(pulse.gates.X)
    m = r.get_calibration(pulse.gates.READOUT)
    md = rd.get_calibration(pulse.gates.READOUT)
    first = readout(rd, md)
    sync(d, a, r, rd)
    for _ in range(3):
        drive(a, xa)
        sync(a, r)
        readout(r, m)
    sync(d, a, r, rd)
    drive(d, xd)
    sync(d, rd)
    return first, readout(rd, md)


@pulse.kernel
def nested_loops(data, ancilla):
    d = get_drive_line(data)
    a = get_drive_line(ancilla)
    r = get_readout_line(ancilla)
    rd = get_readout_line(data)
    x = d.get_calibration(pulse.gates.X)
    m = r.get_calibration(pulse.gates.READOUT)
    md = rd.get_calibration(pulse.gates.READOUT)
    for _ in range(2):
        for _ in range(3):
            drive(d, x)
            sync(d, a, r)
            readout(r, m)
        sync(d, a, r)
    sync(d, rd)
    return readout(rd, md)


@pulse.kernel
def feedback_loop(data, ancilla):
    # Each round flips both qudits and reads the ancilla; a 1 flips the
    # ancilla back, so every round reads 1 if the branch takes the outcome
    # of its own round's readout (1, 0, 1 if it took none).
    d = get_drive_line(data)
    a = get_drive_line(ancilla)
    r = get_readout_line(ancilla)
    rd = get_readout_line(data)
    xd = d.get_calibration(pulse.gates.X)
    xa = a.get_calibration(pulse.gates.X)
    m = r.get_calibration(pulse.gates.READOUT)
    md = rd.get_calibration(pulse.gates.READOUT)
    for _ in range(3):
        drive(d, xd)
        drive(a, xa)
        sync(d, a, r)
        if readout(r, m):
            drive(a, xa)
    sync(d, a, r, rd)
    return readout(rd, md)


@pulse.kernel
def selected(data, ancilla):
    # The waveform of the readout `last` depends on the data qudit's
    # readout: it runs in both arms of an if on it, two copies made before
    # scheduling. `tail` follows it.
    d = get_drive_line(data)
    a = get_drive_line(ancilla)
    r = get_readout_line(ancilla)
    rd = get_readout_line(data)
    x = d.get_calibration(pulse.gates.X)
    m = r.get_calibration(pulse.gates.READOUT)
    md = rd.get_calibration(pulse.gates.READOUT)
    for _ in range(3):
        drive(d, x)
        sync(d, a, r)
        readout(r, m)
    sync(d, rd)
    m1 = readout(rd, md)
    amp = 0.05
    if m1:
        amp = 0.06
    sync(d, a, r, rd)
    last = readout(r, square(400, amp))
    sync(d, a, r, rd)
    tail = readout(rd, md)
    return last, tail


def _lowered(kernel: Any, policy: str, threshold: int = ROLLED) -> Any:
    return pulse.compile(kernel, schedule=policy, unroll_threshold=threshold).lower(*_qudits())


def _measured(lowered: Any) -> list[tuple[int, tuple[int, ...], int]]:
    result = pulse.simulate(lowered, shots_count=1, seed=7)
    return [(m.readout, m.iteration, m.qudit) for m in result.measurements]


def _levels(lowered: Any) -> dict[tuple[int, tuple[int, ...]], int]:
    """The level of each measurement, by readout and iteration, on the path
    the program most likely takes (imperfect gates leave a few percent to
    the others)."""
    result = pulse.simulate(lowered)
    (branch,) = [b for b in result.branches if b.probability > 0.5]
    return {
        (m.readout, m.iteration): k for m, k in zip(result.measurements, branch.levels, strict=True)
    }


def _returned(
    compiled: Any, monkeypatch: Any
) -> tuple[Any, dict[tuple[int, tuple[int, ...]], int]]:
    """What a call of *compiled* returns, and the record of the shot it ran,
    by readout and iteration."""
    from cudaq_pulse.runtime import simulate

    results = []
    run = simulate._simulate_lowered

    def recorded(*args: Any, **kwargs: Any) -> Any:
        results.append(run(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(simulate, "_simulate_lowered", recorded)
    value = compiled(*_qudits())
    monkeypatch.undo()
    (result,) = results
    (row,) = result.records
    return value, {
        (m.readout, m.iteration): int(k) for m, k in zip(result.measurements, row, strict=True)
    }


AFTER = [(0, (0,), 1), (0, (1,), 1), (0, (2,), 1), (1, (), 0)]


@pytest.mark.parametrize("threshold", [ROLLED, UNROLLED])
@pytest.mark.parametrize("policy", POLICIES)
def test_a_measurement_returned_after_a_loop_of_readouts_is_found(policy, threshold, monkeypatch):
    # One shot of an imperfect X: the shot's level of the data qudit's
    # readout, where the lookup found none (-1); 1 on the likely path.
    compiled = pulse.compile(after_loop, schedule=policy, unroll_threshold=threshold)
    value, record = _returned(compiled, monkeypatch)
    returned = 1 if threshold == ROLLED else 3
    assert value == record[(returned, ())] != -1
    assert _levels(compiled.lower(*_qudits()))[(returned, ())] == 1


@pytest.mark.parametrize("policy", POLICIES)
def test_the_readouts_of_a_rolled_loop_keep_the_kernels_numbers(policy):
    assert _measured(_lowered(after_loop, policy)) == AFTER


@pytest.mark.parametrize("policy", POLICIES)
def test_the_readouts_of_an_unrolled_loop_keep_the_kernels_numbers(policy):
    # Unrolled while tracing, each round's readout is one of its own.
    expected = [(0, (), 1), (1, (), 1), (2, (), 1), (3, (), 0)]
    assert _measured(_lowered(after_loop, policy, UNROLLED)) == expected


@pytest.mark.parametrize("policy", POLICIES)
def test_measurements_returned_before_and_after_a_loop_are_found(policy, monkeypatch):
    compiled = pulse.compile(around_loop, schedule=policy, unroll_threshold=ROLLED)
    (first, last), record = _returned(compiled, monkeypatch)
    assert (first, last) == (record[(0, ())], record[(2, ())])
    assert -1 not in (first, last)
    levels = _levels(compiled.lower(*_qudits()))
    assert (levels[(0, ())], levels[(2, ())]) == (0, 1)
    assert _measured(compiled.lower(*_qudits())) == [
        (0, (), 0),
        (1, (0,), 1),
        (1, (1,), 1),
        (1, (2,), 1),
        (2, (), 0),
    ]


@pytest.mark.parametrize("policy", POLICIES)
def test_the_readouts_of_nested_loops_keep_the_kernels_numbers(policy, monkeypatch):
    lowered = _lowered(nested_loops, policy)
    inner = [(0, (i, j), 1) for i in range(2) for j in range(3)]
    assert _measured(lowered) == [*inner, (1, (), 0)]
    compiled = pulse.compile(nested_loops, schedule=policy, unroll_threshold=ROLLED)
    value, record = _returned(compiled, monkeypatch)
    assert value == record[(1, ())] != -1
    # Six X on the data qudit.
    assert _levels(lowered)[(1, ())] == 0


@pytest.mark.parametrize("policy", POLICIES)
def test_branches_on_peeled_readouts_take_the_outcomes_of_the_kernels_readouts(policy, monkeypatch):
    lowered = _lowered(feedback_loop, policy)
    peeled = lowered.mlir("pulse").count("pulse.readout ")
    assert peeled == (3 if policy in ("asap", "rcp") else 2)
    assert _levels(lowered) == {(0, (0,)): 1, (0, (1,)): 1, (0, (2,)): 1, (1, ()): 1}
    compiled = pulse.compile(feedback_loop, schedule=policy, unroll_threshold=ROLLED)
    value, record = _returned(compiled, monkeypatch)
    assert value == record[(1, ())] != -1


@pytest.mark.parametrize("policy", POLICIES)
def test_a_readout_copied_into_both_arms_of_an_if_is_one_readout(policy, monkeypatch):
    # `last` is copied into each arm before scheduling (asap and rcp also
    # peel the loop): each shot runs one copy, reported as the kernel's
    # readout 2, and `tail` stays readout 3.
    lowered = _lowered(selected, policy)
    copies = lowered.mlir("pulse").count("pulse.readout ")
    assert copies == (6 if policy in ("asap", "rcp") else 5)
    result = pulse.simulate(lowered)
    assert [(m.readout, m.iteration, m.qudit) for m in result.measurements] == [
        *AFTER[:3],
        (1, (), 0),
        (2, (), 1),
        (3, (), 0),
    ]
    # Each branch measured every readout: the copy its arm ran.
    assert all(-1 not in b.levels for b in result.branches)
    assert _levels(lowered) == {
        (0, (0,)): 0,
        (0, (1,)): 0,
        (0, (2,)): 0,
        (1, ()): 1,
        (2, ()): 0,
        (3, ()): 1,
    }
    compiled = pulse.compile(selected, schedule=policy, unroll_threshold=ROLLED)
    (last, tail), record = _returned(compiled, monkeypatch)
    assert (last, tail) == (record[(2, ())], record[(3, ())])
    assert -1 not in (last, tail)


@pytest.mark.parametrize("policy", POLICIES)
def test_a_loop_peeled_at_its_end_keeps_the_kernels_numbers(policy):
    # The conditional drive moves by its own length while the wait before
    # the loop holds its line, then by the readout's: asap and rcp peel the
    # last eleven iterations off. (The loops above lose their first.)
    @pulse.kernel
    def drifts_then_locks(q, r):
        b = get_drive_line(r)
        p = get_readout_line(q)
        wait(b, 600)
        for _ in range(30):
            m = readout(p, square(40, 0.05))
            if m:
                drive(b, square(10, 0.1))

    lowered = pulse.compile(drifts_then_locks, schedule=policy, unroll_threshold=ROLLED).lower(
        Q[0], Q[1]
    )
    peeled = lowered.mlir("pulse").count("pulse.readout ")
    assert peeled == (12 if policy in ("asap", "rcp") else 1)
    result = pulse.simulate(lowered)
    assert [(m.readout, m.iteration) for m in result.measurements] == [(0, (k,)) for k in range(30)]


@pulse.classical(max_latency_ns=400)
def decide(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pytest.mark.parametrize("policy", POLICIES)
def test_a_loop_of_calls_run_several_iterations_at_a_time_keeps_the_kernels_numbers(policy):
    # rcp and alap_rcp run four iterations in each and the two left over
    # after the loop; simulation unrolls the loop, which makes calls.
    @pulse.kernel
    def answered(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(r)
        for _ in range(26):
            m = readout(probe, square(40, 0.05))
            if decide.submit([m]).result()[0]:
                drive(line, square(8, 0.1))
            else:
                drive(line, square(4, 0.2))

    lowered = pulse.compile(answered, schedule=policy, unroll_threshold=ROLLED).lower(Q[0], Q[1])
    copies = lowered.mlir("pulse").count("pulse.readout ")
    assert copies == (6 if policy.endswith("rcp") else 1)
    result = pulse.simulate(lowered)
    assert [(m.readout, m.iteration) for m in result.measurements] == [(0, (k,)) for k in range(26)]


def test_measurements_read_the_same_under_every_policy():
    for kernel in (after_loop, around_loop, nested_loops, feedback_loop, selected):
        measured = {policy: _measured(_lowered(kernel, policy)) for policy in POLICIES}
        assert len({tuple(m) for m in measured.values()}) == 1, (kernel.__name__, measured)


def test_the_exports_describe_the_loops_as_scheduled():
    # rcp peels the loop's first iteration: format 1 lists that readout at
    # the top level and the loop's two others at its iterations 0 and 1;
    # format 2 keeps the loop rolled after the peeled readout.
    lowered = _lowered(after_loop, "rcp")
    every = lowered.export_to_controller(version=1)["events"]
    assert [e["iteration"] for e in every if e["kind"] == "readout"] == [[], [0], [1], []]
    rolled = lowered.export_to_controller(version=2)["events"]
    assert [e["kind"] for e in rolled if e["kind"] in ("readout", "loop")] == [
        "readout",
        "loop",
        "readout",
    ]
