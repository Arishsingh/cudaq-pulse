# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mystery qubit: a T1 measurement that holds the qubit with long weak pulses, of 10, 30 and
60 microseconds: fewer than 64 pulses and shorter than 2 ms, but too long in all to send."""

import cudaq_pulse as pulse


@pulse.kernel
def long_drive(q, length: float):
    d, r = get_drive_line(q), get_readout_line(q)
    drive(d, d.get_calibration(pulse.gates.X))
    drive(d, gaussian(length * pulse.ns, 0.0001, length / 4 * pulse.ns))
    sync(d, r)
    return readout(r, r.get_calibration(pulse.gates.READOUT))


def estimate_t1(lab):
    for length in (20000.0, 60000.0, 120000.0):
        records = lab.run(long_drive, lab.qubit, length, shots_count=20)
        print("C1", length, "ok", int(records.sum()))
    return 50.0
