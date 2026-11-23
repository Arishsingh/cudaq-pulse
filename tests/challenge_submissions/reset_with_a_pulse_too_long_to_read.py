# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reset one qubit: nothing but a pulse of 1100 ns."""

import cudaq_pulse as pulse


@pulse.kernel
def reset(qudit):
    drive_line = get_drive_line(qudit)
    drive(drive_line, square(1100 * pulse.ns, 0.0001))
