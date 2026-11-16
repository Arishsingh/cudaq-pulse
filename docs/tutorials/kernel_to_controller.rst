..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

From Kernel to Controller
=========================

A compiled pulse program doesn't depend on any one control system. A
*controller backend* turns it into what a particular controller plays: its
sequencer instructions and waveform memory. This tutorial is for people who
build or program control systems. It exports a scheduled program, checks a
hardware limit on it in plain Python, writes a small backend, and runs the
backend through the checks every backend should pass.

The exported program
--------------------

``export_to_controller()`` gives a scheduled program as JSON-ready data:
every drive, readout, wait and sync with its line and its time in clock
cycles.

.. code-block:: python

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


   program = drive_and_measure.lower(qs.qudits[0]).export_to_controller()
   print(program["format"])   # cudaq-pulse.controller/1
   for event in program["events"]:
       print(event["kind"], event["line"], event["start_cycles"], event["duration_cycles"])
   # drive 0 0 40
   # sync None 40 0
   # readout 1 40 200

At a 1 GHz clock, the 40 ns drive pulse is 40 cycles, and the readout starts
where ``sync`` put it. The format's name carries a version: within
``/1``, keys and event kinds may be added but never change meaning, so a
backend written today keeps reading programs from later releases.

Check a limit in Python
-----------------------

``schedule`` lists every operation with its line and its start and end, so a
check the compiler doesn't know about is a short Python function. Suppose a
controller can play at most two drive lines at once:

.. code-block:: python

   def concurrent_drives(scheduled, limit):
       edges = []
       for event in scheduled.schedule:
           if event.kind == "drive" and event.duration_cycles:
               edges += [(event.start, 1), (event.end, -1)]
       active, violations = 0, []
       for time, step in sorted(edges):   # an end sorts before a start at the same time
           active += step
           if step > 0 and active > limit:
               violations.append((time, active))
       return violations


   @pulse.kernel
   def flip_four(qudits):
       for q in qudits:
           d = get_drive_line(q)
           drive(d, d.get_calibration(pulse.gates.X))


   surface = pulse.quantum_system("transmon_surface17")
   four = surface.qudits[0:4]
   print(concurrent_drives(flip_four.lower(four), 2))   # [(0, 3), (0, 4)]

All four π pulses start at cycle 0, two more than the limit. The compiler can
schedule around such a limit itself when the control system states it, with
the resource-constrained policy:

.. code-block:: python

   limited = pulse.ControlSystem(max_concurrent_drives=2)
   program = pulse.compile(flip_four, schedule="rcp", control_system=limited).lower(four)
   print(concurrent_drives(program, 2))   # []
   print(sorted(e.start for e in program.schedule))   # [0, 0, 80, 80]

Two pulses play first and two follow, so the limit holds.

A backend in thirty lines
-------------------------

A backend is an object with a ``name`` and three methods: ``control_system``
says what its controller can play, ``emit`` turns an exported program into
the controller's own form, and ``run`` plays it. This one emits a line of
text per drive and readout:

.. code-block:: python

   from collections.abc import Mapping
   from typing import Any

   from cudaq_pulse import Controller, backends
   from cudaq_pulse.backends import IncompatibleBackendError
   from cudaq_pulse.controller import FORMAT


   class TextBackend:
       name = "text"
       formats = (FORMAT,)

       def control_system(self) -> Controller:
           return Controller(clock_ghz=1.0, instruction_cycles=2, min_idle_cycles=2)

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

       def run(self, artifact: Any, *, shots_count: int = 1) -> list[list[str]]:
           return [list(artifact) for _ in range(shots_count)]

``Controller`` states the controller's limits: a 1 GHz clock, one
instruction every 2 cycles, and idle gaps of at least 2 cycles. Compiling
against it checks the program against those limits. The backend refuses
what it can't play, such as a pulse inside a branch, with
``IncompatibleBackendError``, instead of playing something else.

Register it and run a program
-----------------------------

An installed backend is found by its entry point; ``register`` adds one to
the running process instead:

.. code-block:: python

   backends.register("text", TextBackend)
   backend = backends.get("text")
   compiled = pulse.compile(drive_and_measure, quantum_system=qs, control_system=backend.control_system())
   artifact = backend.emit(compiled.lower(qs.qudits[0]).export_to_controller())
   for line in backend.run(artifact)[0]:
       print(line)
   # 0 drive line 0 gaussian at 5e+09 Hz
   # 40 readout line 1 square at 5e+09 Hz

``backends.validate`` runs a corpus of test programs through a backend, one
for each kind of event, and checks that each is either emitted or refused
cleanly:

.. code-block:: python

   report = backends.validate("text", controller=Controller(clock_ghz=1.0))
   print(report.ok)   # True
   print(str(report).splitlines()[-1])   # backend 'text': 10 passed, 8 refused
   backends.unregister("text")

The text backend plays drives, readouts, waits, syncs and loops, and refuses
classical calls, frame updates, branches and values known only at run time.
Refusing is a pass: what fails validation is a backend that emits a program
it can't play faithfully, or reads a format it doesn't know.

Where to go next
----------------

- :example:`controller_backend/toy_controller/__init__.py` is a toy backend as
  an installable distribution, with its entry point in
  :example:`controller_backend/pyproject.toml`.
- :example:`custom_passes/python_pass.py` is the concurrent-drive check above,
  as a script.
- :example:`23_define_a_qpu.py` exports one program for each of two
  controllers joined by a feedback link.
- :example:`transmon/transmon_surface17_multiplexed_readout.py` reads four
  qudits through one feedline and shows each readout's tone in the export.

:doc:`../user_guide/controller_backends` documents every key of the export
format, clocks, shared lines and waveform memory, and
:doc:`../user_guide/custom_passes` how to load your own MLIR passes into
``cudaq-pulse-opt``.
