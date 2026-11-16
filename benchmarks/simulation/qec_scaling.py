#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QEC scaling benchmark: compile time vs distance for surface-code-like programs.

Compiles a synthetic surface-code syndrome extraction cycle at several code
distances (trace, module build, verify, ALAP schedule; no optimization
passes) and prints each stage's time and its growth relative to the smallest
distance, to compare against the growth in qubit and op count.

``--code gross`` compiles the same kind of cycle for the [[144,12,12]]
bivariate bicycle ("gross") code instead, and prints its one row.

Usage:
  python benchmarks/simulation/qec_scaling.py [--distances D ...] [--repeats N]
  python benchmarks/simulation/qec_scaling.py --code gross [--repeats N]
"""

from __future__ import annotations

import argparse
import statistics
from typing import Any

import cudaq_pulse as pulse


def _device(frequencies: dict[int, float]) -> pulse.QuantumSystem:
    """A quantum system with these drive frequencies, indexed by qudit."""
    from cudaq_pulse.quantum_systems import PhysicalQudit

    qudits = {
        q: PhysicalQudit(q, f, 50.0, 30.0, anharmonicity_hz=-300e6) for q, f in frequencies.items()
    }
    return pulse.quantum_system(name="bench", qudits=qudits)


_STAGES = ("trace_ms", "ffi_ms", "passes_ms", "schedule_ms", "total_ms")


def _extraction_cycle(data_count: int, checks: list[tuple[bool, list[int]]]) -> pulse.Kernel[Any]:
    """One syndrome extraction cycle on data_count data qudits, then one
    ancilla qudit per check, in order.

    Each check is (wrapped, data qudits): the ancilla interacts with each of
    its data qudits in turn via echoed CR, and a wrapped check's ancilla gets
    a Hadamard (phase, SX, phase) before and after.
    """
    check_count = len(checks)
    total_count = data_count + check_count
    wrapped = [w for w, _ in checks]
    neighbors = [row for _, row in checks]

    @pulse.kernel
    def cycle(qudits):
        lines = []
        tones = []
        for q in range(total_count):
            dl = get_drive_line(qudits[q])
            lines.append(dl)
            tones.append(dl.tone)
        for check in range(check_count):
            anc = data_count + check
            if wrapped[check]:
                shift_phase(tones[anc], 1.5708)
                drive(lines[anc], drag(20, 0.44, 5.0, 0.5), tone=tones[anc])
                shift_phase(tones[anc], 1.5708)
            row = neighbors[check]
            for j in range(len(row)):
                data_q = row[j]
                drive(lines[data_q], gaussian(98, 0.32, 24.5), tone=tones[data_q])
                drive(lines[anc], drag(20, 0.44, 5.0, 0.5), tone=tones[anc])
            if wrapped[check]:
                shift_phase(tones[anc], 1.5708)
                drive(lines[anc], drag(20, 0.44, 5.0, 0.5), tone=tones[anc])
                shift_phase(tones[anc], 1.5708)

    return cycle


def surface_code_cycle(d: int) -> tuple[pulse.Kernel[Any], int]:
    """A d-distance surface-code-like syndrome extraction kernel.

    Layout:
      d^2 data qubits + (d^2 - 1) ancilla qubits = 2d^2 - 1 total
      Each ancilla interacts with up to 4 data neighbors via echoed CR, and
      Z-type ancillas are wrapped in Hadamards (phase, SX, phase).
    """
    data_count = d * d
    ancilla_count = d * d - 1
    checks = []
    for anc_idx in range(ancilla_count):
        row, col = anc_idx // (d - 1), anc_idx % (d - 1)
        checks.append(
            (
                (row + col) % 2 == 0,
                [
                    (row + dr) * d + (col + dc)
                    for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1))
                    if row + dr < d and col + dc < d
                ],
            )
        )
    return _extraction_cycle(data_count, checks), data_count + ancilla_count


# The gross code: l = 12, m = 6, A = x^3 + y + y^2, B = y^3 + x + x^2, as
# (power of x, power of y) for x = S_l (x) I_m and y = I_l (x) S_m.
_GROSS_L, _GROSS_M = 12, 6
_GROSS_A = ((3, 0), (0, 1), (0, 2))
_GROSS_B = ((0, 3), (1, 0), (2, 0))


def gross_code_checks() -> tuple[list[list[int]], list[list[int]]]:
    """The data qudits of each X check (rows of H_X = [A|B]) and each Z check
    (rows of H_Z = [B^T|A^T]) of the [[144,12,12]] bivariate bicycle code.

    Data qudit (block, a, b) is block * 72 + a * 6 + b: 144 data qudits,
    72 X and 72 Z checks of weight 6.
    """
    lx, my = _GROSS_L, _GROSS_M

    def support(
        a: int, b: int, terms: tuple[tuple[int, int], ...], sign: int, block: int
    ) -> list[int]:
        return [
            block * lx * my + ((a + sign * p) % lx) * my + (b + sign * q) % my for p, q in terms
        ]

    x_checks = []
    z_checks = []
    for a in range(lx):
        for b in range(my):
            x_checks.append(sorted(support(a, b, _GROSS_A, 1, 0) + support(a, b, _GROSS_B, 1, 1)))
            z_checks.append(sorted(support(a, b, _GROSS_B, -1, 0) + support(a, b, _GROSS_A, -1, 1)))
    return x_checks, z_checks


def gross_code_cycle() -> tuple[pulse.Kernel[Any], int]:
    """A syndrome extraction kernel for the [[144,12,12]] bivariate bicycle code.

    Layout:
      144 data qubits + 144 check qubits (72 X, then 72 Z) = 288 total
      Each check qubit interacts with its 6 data qubits via echoed CR, and
      Z-type check qubits are wrapped in Hadamards (phase, SX, phase), as in
      surface_code_cycle. Any two qubits may couple.
    """
    x_checks, z_checks = gross_code_checks()
    data_count = 2 * _GROSS_L * _GROSS_M
    checks = [(False, row) for row in x_checks] + [(True, row) for row in z_checks]
    return _extraction_cycle(data_count, checks), data_count + len(checks)


def _bench(
    kernel: pulse.Kernel[Any], qudit_count: int, repeats: int
) -> tuple[int, int, dict[str, float]]:
    quantum_system = _device({q: 5.0e9 + 0.05e9 * (q % 8) for q in range(qudit_count)})

    def compile_once() -> pulse.ScheduledKernel:
        # A fresh compiled kernel holds no specializations: this compiles.
        return pulse.compile(kernel, passes=("verify",)).lower(quantum_system.qudits[:qudit_count])

    op_count = compile_once().metrics.op_count
    samples: dict[str, list[float]] = {stage: [] for stage in _STAGES}
    for _ in range(repeats):
        metrics = compile_once().metrics
        for stage in _STAGES:
            samples[stage].append(getattr(metrics, stage))
    return qudit_count, op_count, {stage: statistics.median(xs) for stage, xs in samples.items()}


def bench(d: int, repeats: int) -> tuple[int, int, dict[str, float]]:
    return _bench(*surface_code_cycle(d), repeats)


def bench_gross(repeats: int) -> tuple[int, int, dict[str, float]]:
    return _bench(*gross_code_cycle(), repeats)


def _print_header() -> None:
    labels = ("Trace", "Build", "Verify", "Sched", "Total")
    print(
        f"  {'d':>3s}  {'Qubits':>7s}  {'Ops':>7s}"
        + "".join(f"  {s + ' (ms)':>11s}" for s in labels)
    )
    print(f"  {'---':>3s}  {'-' * 7:>7s}  {'-' * 7:>7s}" + f"  {'-' * 11}" * len(labels))


def _print_row(d: int, qudit_count: int, op_count: int, stages: dict[str, float]) -> None:
    print(
        f"  {d:3d}  {qudit_count:7d}  {op_count:7d}"
        + "".join(f"  {stages[s]:11.2f}" for s in _STAGES)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QEC compile-time scaling benchmark")
    parser.add_argument("--code", choices=("surface", "gross"), default="surface")
    parser.add_argument("--distances", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.code == "gross":
        if args.distances is not None:
            parser.error("--distances is for the surface code; the gross code has distance 12")
        print("  [[144,12,12]] bivariate bicycle (gross) code")
        _print_header()
        _print_row(12, *bench_gross(args.repeats))
        return

    distances = args.distances or [3, 5, 7, 9, 11, 15, 21]
    if any(d < 2 for d in distances):
        parser.error("distances must be at least 2")

    _print_header()
    rows = []
    for d in sorted(distances):
        qudit_count, op_count, stages = bench(d, args.repeats)
        rows.append((d, qudit_count, op_count, stages["total_ms"]))
        _print_row(d, qudit_count, op_count, stages)

    base_d, base_q, base_ops, base_t = rows[0]
    print(f"\n  Scaling relative to d={base_d} ({base_q} qubits, {base_ops} ops, {base_t:.2f} ms):")
    print(
        f"  {'d':>3s}  {'Qubits':>7s}  {'Q ratio':>8s}  {'Ops':>7s}  {'Op ratio':>9s}  {'Time ratio':>10s}"
    )
    print(
        f"  {'---':>3s}  {'-------':>7s}  {'--------':>8s}  {'-------':>7s}  {'---------':>9s}  {'----------':>10s}"
    )
    for d, nq, nops, t in rows:
        print(
            f"  {d:3d}  {nq:7d}  {nq / base_q:8.1f}x  {nops:7d}  {nops / base_ops:9.1f}x  {t / base_t:10.1f}x"
        )


if __name__ == "__main__":
    main()
