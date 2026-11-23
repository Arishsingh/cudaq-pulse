# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reset one qubit: three readouts and nothing else."""

import cudaq_pulse as pulse


@pulse.kernel
def reset(qudit):
    readout_line = get_readout_line(qudit)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)
    for _ in range(3):
        readout(readout_line, measurement)
