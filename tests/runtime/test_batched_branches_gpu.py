# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sibling branches that evolve as one batched GPU run end where each one's
own run ends, pure or mixed, on one qudit or two: bit for bit when every
product is its own term, and to rounding when the lowering sums them.
"""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import MicrowaveDrive, PhysicalQudit
from cudaq_pulse.runtime import simulate as simulate_module

from tests._handles import Q
from tests.runtime._lowering import on

pytestmark = pytest.mark.gpu

_HALF_PI = math.pi / 40.0
# A few ulps of a density matrix's entries, which are at most 1.
_ROUNDING = 1e-15


def _target(t1_us):
    return pulse.quantum_system(
        name="batched-branches",
        qudits={
            q: PhysicalQudit(
                index=q,
                frequency_hz=5.0e9 + 0.1e9 * q,
                anharmonicity_hz=-200.0e6,
                t1_us=t1_us,
                t2_star_us=0.0,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
            for q in range(2)
        },
    )


@pulse.kernel
def _rounds(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    r0 = get_readout_line(q0)
    for _ in range(3):
        drive(d0, square(40, _HALF_PI))
        drive(d1, gaussian(40, _HALF_PI, 10))
        sync(d0, d1)
        sync(d0, r0)
        readout(r0, square(200, 0.05))
        sync(d0, r0)
        sync(d0, d1)
    drive(d1, square(40, _HALF_PI))


def _one_at_a_time(self, states, paths, t0, t1):
    return [self.evolve(s, e, t0, t1) for s, e in zip(states, paths, strict=True)]


@pytest.mark.parametrize("operators", ["dense", "fused"])
@pytest.mark.parametrize("t1_us", [0.0, 5.0])
@pytest.mark.parametrize("levels", [2, 3])
def test_batched_branches_are_their_own_runs(t1_us, levels, operators, monkeypatch):
    monkeypatch.setenv("CUDAQ_PULSE_GPU_OPERATORS", operators)
    lowered = on(_target(t1_us), _rounds, Q[0], Q[1])
    kwargs = {"backend": "gpu", "levels": levels, "shots_count": 100, "seed": 4}
    runs = []
    execute = simulate_module._GpuBackend._execute

    def counted(self, text, states, mixed):
        runs.append(len(states))
        return execute(self, text, states, mixed)

    with monkeypatch.context() as m:
        m.setattr(simulate_module._GpuBackend, "_execute", counted)
        batched = pulse.simulate(lowered, **kwargs)
    # The paths after the third readout run as one batch.
    assert max(runs) == levels**3
    with monkeypatch.context() as m:
        m.setattr(simulate_module._GpuBackend, "evolve_together", _one_at_a_time)
        single = pulse.simulate(lowered, **kwargs)
    probabilities = [[b.probability for b in r.branches] for r in (batched, single)]
    if operators == "dense":
        assert batched.density_matrix.tobytes() == single.density_matrix.tobytes()
        assert probabilities[0] == probabilities[1]
    else:
        # cuDensityMat may contract an operator on both qudits in another order
        # for a batch than for one state; any other difference is far larger.
        np.testing.assert_allclose(
            batched.density_matrix, single.density_matrix, rtol=0, atol=_ROUNDING
        )
        np.testing.assert_allclose(*probabilities, rtol=0, atol=_ROUNDING)
    np.testing.assert_array_equal(batched.records, single.records)
