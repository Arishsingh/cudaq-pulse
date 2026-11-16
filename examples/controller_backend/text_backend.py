# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A controller backend in thirty lines, which plays a program as text.

It plays each drive and readout of an exported program as one line of text,
skips waits and syncs, and refuses anything else, such as a pulse inside a
branch, with ``IncompatibleBackendError``. Registered in the running process,
it emits a program and passes ``backends.validate``. The "From Kernel to
Controller" tutorial shows it a method at a time.
"""

# [start imports]
from collections.abc import Mapping
from typing import Any

from cudaq_pulse import Controller, backends
from cudaq_pulse.backends import IncompatibleBackendError
from cudaq_pulse.controller import FORMAT

# [end imports]


# [start class]
class TextBackend:
    name = "text"
    formats = (FORMAT,)

    def control_system(self) -> Controller:
        return Controller(clock_ghz=1.0, instruction_cycles=2, min_idle_cycles=2)

    # [end class]
    # [start emit]
    def emit(self, program: Mapping[str, Any]) -> list[str]:
        if program["format"] != FORMAT:
            raise IncompatibleBackendError(f"cannot read {program['format']}")
        played = []
        for event in program["events"]:
            if event["kind"] in ("wait", "sync"):
                continue
            if event["kind"] not in ("drive", "readout") or event["conditions"]:
                raise IncompatibleBackendError(f"cannot play event {event['index']}")
            played.append(
                f"{event['start_cycles']} {event['kind']} line {event['line']} "
                f"{event['waveform']['shape']} at {event['frequency_hz']:.6g} Hz"
            )
        return played

    # [end emit]
    # [start run]
    def run(self, artifact: Any, *, shots_count: int = 1) -> list[list[str]]:
        return [list(artifact) for _ in range(shots_count)]

    # [end run]


if __name__ == "__main__":
    import cudaq_pulse as pulse
    from cudaq_pulse.quantum_systems import PhysicalQudit

    qs = pulse.quantum_system(
        name="one-qubit",
        qudits={0: PhysicalQudit(0, 5.0e9, 50.0, 30.0, anharmonicity_hz=-2.0e8)},
        clock_ghz=1.0,
    )

    @pulse.kernel
    def drive_and_measure(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, gaussian(40 * pulse.ns, 0.5, 10 * pulse.ns))
        sync(d, r)
        readout(r, square(200 * pulse.ns, 0.05))

    backends.register("text", TextBackend)
    backend = backends.get("text")
    compiled = pulse.compile(
        drive_and_measure, quantum_system=qs, control_system=backend.control_system()
    )
    for line in backend.run(backend.emit(compiled.lower(qs.qudits[0]).export_to_controller()))[0]:
        print(line)
    print(str(backends.validate("text", controller=Controller(clock_ghz=1.0))).splitlines()[-1])
    backends.unregister("text")
