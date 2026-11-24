..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

One QPU, Several Control Systems
================================

The same chip can be wired to very different electronics. This tutorial
keeps one QPU, the built-in ``transmon_surface17``, and puts five control
systems under it. Two programs are compiled on each: an active reset of
one qubit, which is also simulated, and one round of error correction.

What you'll learn:

- how to describe a control system and attach it to a QPU;
- how feedback latency and a hop between boxes change a schedule;
- what the compiler tells you when two boxes can't be lined up;
- how a limit on drives at once stretches a round;
- what each choice costs, in nanoseconds and in errors.

All the code on this page is also one Python script,
:download:`control_systems.py <scripts/control_systems.py>`.
:doc:`../user_guide/transmon_surface17` describes the QPU itself.

The QPU and a reset
-------------------

First, load the QPU. On its own it has no control system:

.. code-block:: python

   import cudaq_pulse as pulse

   base = pulse.quantum_system("transmon_surface17")
   print(base.control_system)  # None

A controller counts time in cycles of its clock. This helper turns
nanoseconds into cycles of the QPU's 2 GHz clock:

.. code-block:: python

   def cycles(ns):
       """A time in ns, in cycles of the QPU's clock."""
       return round(ns * base.clock_ghz)


   print(cycles(100))  # 200

The first program is an active reset. It excites the qubit, reads it out,
and flips it back only if it read 1:

.. code-block:: python

   @pulse.kernel
   def active_reset(qudit):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       flip = drive_line.get_calibration(pulse.gates.X)
       measurement = readout_line.get_calibration(pulse.gates.READOUT)

       drive(drive_line, flip)
       sync(drive_line, readout_line)
       outcome = readout(readout_line, measurement)

       sync(drive_line, readout_line)
       if outcome:
           drive(drive_line, flip)

Both pulses come from the QPU's calibration, so the kernel holds no pulse
values of its own.

``base.replace(control_system=...)`` gives the same QPU with a control
system. This function compiles the reset of qubit Z4 on it, simulates it,
and returns two numbers: how long the schedule is, and how much of the
population ends in the ground state:

.. code-block:: python

   def reset_on(control_system, schedule="alap"):
       """The reset of Z4: its length in ns, and P(0) at its end."""
       system = base.replace(control_system=control_system)
       compiled = pulse.compile(
           active_reset,
           quantum_system=system,
           schedule=schedule,
       )
       scheduled = compiled.lower(system.qudit("Z4"))
       result = pulse.simulate(scheduled, readout="model")
       ground = result.level_population(0)
       return scheduled.duration / pulse.ns, ground

``readout="model"`` reports each outcome through the qubit's calibrated
readout errors, so a wrong outcome sometimes flips the qubit the wrong way.

One controller, fast or slow feedback
-------------------------------------

The simplest control system is one controller that drives and reads every
line. Its feedback latency is the time from the end of a readout to the
first pulse that can depend on the outcome.

Make two of them, with 100 ns and with 400 ns of feedback latency, and
keep every setup with its schedule policy in one dictionary:

.. code-block:: python

   fast = pulse.Controller(feedback_latency_cycles=cycles(100))
   slow = pulse.Controller(feedback_latency_cycles=cycles(400))

   setups = {
       "one controller, 100 ns feedback": (
           pulse.ControlSystem(controllers=[fast]),
           "alap",
       ),
       "one controller, 400 ns feedback": (
           pulse.ControlSystem(controllers=[slow]),
           "alap",
       ),
   }

   for name, (control_system, schedule) in setups.items():
       duration_ns, ground = reset_on(control_system, schedule)
       print(f"{name}: {duration_ns:.0f} ns, P(0) = {ground:.4f}")

.. code-block:: text
   :class: output

   one controller, 100 ns feedback: 380 ns, P(0) = 0.9943
   one controller, 400 ns feedback: 680 ns, P(0) = 0.9836

What to notice:

- the schedule grows by exactly the extra 300 ns of latency;
- the qubit sits excited while it waits, and decays, so the slower
  controller resets it less well.

A controller per feedline
-------------------------

Next, split the electronics the way the chip is wired. The QPU's
calibration file groups the qubits by the readout feedline they share.
Read the groups as qudit indices:

.. code-block:: python

   import json
   from importlib.resources import files

   package = files("cudaq_pulse.quantum_systems")
   calibration = package / "calibrations" / "transmon_surface17.json"
   readout_data = json.loads(calibration.read_text())["readout"]
   groups = readout_data["feedlines"]["groups"]

   index = {q.label: q.index for q in base.physical_qudits.values()}
   feedlines = [
       [index[label] for label in group["qudits"]]
       for group in groups.values()
   ]
   for group in groups.values():
       print(" ".join(group["qudits"]))

.. code-block:: text
   :class: output

   D7 D4 Z1 Z2
   D1 D2 D5 X1 X2
   D9 D8 X3 X4
   D3 D6 Z3 Z4

Give each feedline its own controller. It drives the feedline's qubits and
reads them all through one shared line. An outcome needs 200 ns more to
reach another controller, and all four lock to one 10 MHz reference clock:

.. code-block:: python

   from cudaq_pulse.controller import ReadoutLine

   per_feedline = [
       pulse.Controller(
           drive_lines=group,
           readout_lines=[
               ReadoutLine(group, max_concurrent_tones=len(group))
           ],
           feedback_latency_cycles=cycles(100),
           peer_latency_cycles=cycles(200),
       )
       for group in feedlines
   ]
   setups["a controller per feedline"] = (
       pulse.ControlSystem(
           controllers=per_feedline,
           reference_clock_ghz=0.01,
       ),
       "alap",
   )

   control_system, schedule = setups["a controller per feedline"]
   duration_ns, ground = reset_on(control_system, schedule)
   print(f"{duration_ns:.0f} ns, P(0) = {ground:.4f}")

.. code-block:: text
   :class: output

   380 ns, P(0) = 0.9943

What to notice: nothing changed. Z4's drive line and its feedline are on
the same controller, so the outcome never has to travel the 200 ns to
another one.

Drives and readouts on separate modules
---------------------------------------

Another common rack has one module for every drive and another for every
readout. Now each outcome must cross from one box to the other.

Describe the two modules, and try them without a shared clock first:

.. code-block:: python

   import textwrap

   from cudaq_pulse.controller import ControllerLimitError

   drive_module = pulse.Controller(drive_lines=range(len(index)))
   readout_module = pulse.Controller(
       readout_lines=range(len(index)),
       feedback_latency_cycles=cycles(100),
       peer_latency_cycles=cycles(200),
   )
   unclocked = pulse.ControlSystem(
       controllers=[drive_module, readout_module],
   )

   try:
       reset_on(unclocked)
   except ControllerLimitError as error:
       message = str(error.violations[0])
       print(textwrap.fill(message, width=64))

.. code-block:: text
   :class: output

   controllers[0]: the sync at 80 lines up the drive on line 0 with
   the pulses of controllers[1], but they land within 1 of its
   cycles of where they are meant to (1 without a shared
   reference_clock_ghz), more than its alignment_tolerance_cycles
   of 0; share a reference_clock_ghz between them, or raise its
   alignment_tolerance_cycles to 1

The compiler refuses the program, and says why. The kernel's ``sync`` asks
for the readout to start right when the drive ends. Two boxes without a
shared clock each start within one cycle of where they should, and that is
more than these controllers allow. The message gives both ways out.

Take the first: lock both modules to the 10 MHz reference:

.. code-block:: python

   setups["drive and readout modules"] = (
       pulse.ControlSystem(
           controllers=[drive_module, readout_module],
           reference_clock_ghz=0.01,
       ),
       "alap",
   )

   control_system, schedule = setups["drive and readout modules"]
   duration_ns, ground = reset_on(control_system, schedule)
   print(f"{duration_ns:.0f} ns, P(0) = {ground:.4f}")

.. code-block:: text
   :class: output

   580 ns, P(0) = 0.9872

What to notice: the flip now waits 300 ns: 100 ns of feedback latency in
the readout module, then 200 ns on the way to the drive module.

A limited number of drive lanes
-------------------------------

The last setup is one fast controller again, with a limit: it can run at
most six drives at the same time. ``max_concurrent_drives`` states it, and
the ``"rcp"`` schedule policy keeps to it:

.. code-block:: python

   setups["six drive lanes"] = (
       pulse.ControlSystem(
           controllers=[fast],
           max_concurrent_drives=6,
           max_concurrent_readouts=8,
       ),
       "rcp",
   )

   control_system, schedule = setups["six drive lanes"]
   duration_ns, ground = reset_on(control_system, schedule)
   print(f"{duration_ns:.0f} ns, P(0) = {ground:.4f}")

.. code-block:: text
   :class: output

   380 ns, P(0) = 0.9943

The reset uses one drive at a time, so the limit costs it nothing. The
next program is where it shows.

.. themed-figure:: figures/control_reset_timelines.svg
   :alt: Five timelines of the active reset of Z4, one per control system. In each, a pi pulse is followed by a readout from 40 to 240 ns, and a second pi pulse that is sent only if the readout read 1. one controller, 100 ns feedback: the flip starts at 340 ns, after 100 ns of feedback latency, and the reset ends at 380 ns; one controller, 400 ns feedback: the flip starts at 640 ns, after 400 ns of feedback latency, and the reset ends at 680 ns; a controller per feedline: the flip starts at 340 ns, after 100 ns of feedback latency, and the reset ends at 380 ns; drive and readout modules: the flip starts at 540 ns, after 100 ns of feedback latency and 200 ns between controllers, and the reset ends at 580 ns; six drive lanes: the flip starts at 340 ns, after 100 ns of feedback latency, and the reset ends at 380 ns.
   :width: 100%

   The reset of Z4 under each control system, on one time axis. The
   bar on the outcome bus is the time the outcome takes to reach the
   flip.

One round of error correction
-----------------------------

The second program uses all 17 qubits: one round of the surface code. It
follows the QPU's CZ order, ``CZ_LAYERS``: four CZ steps for the Z
stabilizers, then four for the X stabilizers. Then it reads the eight
auxiliary qubits and resets each one that read 1.

The simulator does not model flux lines yet, so a CZ is a slot in which
its two qubits wait. The slot's length comes from the QPU and enters the
kernel as an argument:

.. code-block:: python

   from cudaq_pulse.quantum_systems.transmon import CZ_LAYERS


   @pulse.kernel
   def qec_round(qudits, cz_ns: float):
       drives = [get_drive_line(q) for q in qudits]
       readouts = [get_readout_line(q) for q in qudits]
       flips, half_flips = [], []
       for line in drives:
           flips.append(line.get_calibration(pulse.gates.X))
           half_flips.append(line.get_calibration(pulse.gates.X90))
       data = sorted({d for layer in CZ_LAYERS for _, d in layer})

       for half, layers in enumerate((CZ_LAYERS[:4], CZ_LAYERS[4:])):
           auxiliaries = sorted(
               {a for layer in layers for a, _ in layer}
           )
           # The X half also turns the data qubits to the X basis.
           turned = auxiliaries + data if half == 1 else auxiliaries
           for i in turned:
               drive(drives[i], half_flips[i])
           for step, layer in enumerate(layers):
               sync(*drives)
               for a, d in layer:
                   wait(drives[a], cz_ns * pulse.ns)
                   wait(drives[d], cz_ns * pulse.ns)
               if step == 1:
                   # An echo on the data qubits, halfway through.
                   sync(*drives)
                   for i in data:
                       drive(drives[i], flips[i])
           sync(*drives)
           for i in turned:
               drive(drives[i], half_flips[i])

       auxiliaries = sorted(
           {a for layer in CZ_LAYERS for a, _ in layer}
       )
       sync(*drives, *readouts)
       outcomes = []
       for i in auxiliaries:
           line = readouts[i]
           measurement = line.get_calibration(pulse.gates.READOUT)
           outcomes.append(readout(line, measurement))
       sync(*drives, *readouts)
       for i, outcome in zip(auxiliaries, outcomes):
           if outcome:
               drive(drives[i], flips[i])

Compile the round on each control system and print its length:

.. code-block:: python

   cz_ns = max(
       coupling.get_calibration(pulse.gates.CZ).duration_ns
       for coupling in base.couplings
   )


   def round_on(control_system, schedule="alap"):
       """The length in ns of one round."""
       system = base.replace(control_system=control_system)
       compiled = pulse.compile(
           qec_round,
           quantum_system=system,
           schedule=schedule,
       )
       scheduled = compiled.lower(system.qudits[: len(index)], cz_ns)
       return scheduled.duration / pulse.ns


   for name, (control_system, schedule) in setups.items():
       print(f"{name}: {round_on(control_system, schedule):.0f} ns")

.. code-block:: text
   :class: output

   one controller, 100 ns feedback: 1324 ns
   one controller, 400 ns feedback: 1624 ns
   a controller per feedline: 1324 ns
   drive and readout modules: 1524 ns
   six drive lanes: 1604 ns

What to notice:

- slow feedback and the hop between modules add to the round what they
  added to the reset;
- a controller per feedline again costs nothing: every auxiliary qubit is
  driven by the controller that reads it;
- six drive lanes stretch the round. Wherever more than six pulses are
  due at once, they now go out in batches: the 17 pulses between the two
  halves take three batches instead of one.

What a longer round costs
-------------------------

A logical error rate would be the natural thing to compare. It can't be
computed honestly here: without modeled flux lines there is no simulated
CZ, so a simulated round measures no stabilizer.

This page uses the closest quantity the simulator does give: the error a
data qubit picks up while it waits for one round. The kernel puts a data
qubit in a superposition, holds it for a time with one echo pulse in the
middle, and brings it back:

.. code-block:: python

   @pulse.kernel
   def hold(qudit, half_ns: float):
       drive_line = get_drive_line(qudit)
       half_flip = drive_line.get_calibration(pulse.gates.X90)
       flip = drive_line.get_calibration(pulse.gates.X)

       drive(drive_line, half_flip)
       wait(drive_line, half_ns * pulse.ns)
       drive(drive_line, flip)
       wait(drive_line, half_ns * pulse.ns)
       drive(drive_line, half_flip)

Without errors the qubit ends in the ground state. The idle error is the
population found elsewhere, averaged over the nine data qubits. The three
pulses are part of the round's time, so the wait is what is left of it:

.. code-block:: python

   import numpy as np

   data = [
       base.qudit(q.label)
       for q in base.physical_qudits.values()
       if q.label.startswith("D")
   ]
   ground_state = np.array([1.0, 0.0, 0.0])


   def idle_error(round_ns):
       """The mean error of a data qubit held for one round."""
       errors = []
       for qudit in data:
           flip = qudit.get_calibration(pulse.gates.X)
           wait_ns = max(round_ns - 3 * flip.duration_ns, 0.0)
           result = pulse.simulate(
               hold,
               qudit,
               wait_ns / 2,
               dephasing="t2_echo",
               initial_state=ground_state,
           )
           errors.append(1 - result.level_population(0))
       return sum(errors) / len(errors)


   print(f"{idle_error(0):.2%}")  # 0.10%

``dephasing="t2_echo"`` uses each qubit's echo T2, as the hold has an
echo. The three pulses alone, with no time between them, already cost the
error printed above. It is in every number below.

This is not a logical error rate. It leaves out the CZ errors, crosstalk
and wrong readouts. It shows one thing only: what the extra length of a
round costs the data qubits.

Side by side
------------

Finally, run both programs on every setup and print the table:

.. code-block:: python

   rows = []
   for name, (control_system, schedule) in setups.items():
       reset_ns, ground = reset_on(control_system, schedule)
       round_ns = round_on(control_system, schedule)
       error = idle_error(round_ns)
       rows.append((name, reset_ns, ground, round_ns, error))

   header = ("", "reset", "P(0)", "round", "idle")
   print("{:31}  {:>6}  {:>6}  {:>7}  {:>5}".format(*header))
   for name, reset_ns, ground, round_ns, error in rows:
       print(
           f"{name:31}  {reset_ns:3.0f} ns  {ground:.4f}  "
           f"{round_ns:4.0f} ns  {error:.2%}"
       )

.. code-block:: text
   :class: output

                                     reset    P(0)    round   idle
   one controller, 100 ns feedback  380 ns  0.9943  1324 ns  1.23%
   one controller, 400 ns feedback  680 ns  0.9836  1624 ns  1.51%
   a controller per feedline        380 ns  0.9943  1324 ns  1.23%
   drive and readout modules        580 ns  0.9872  1524 ns  1.42%
   six drive lanes                  380 ns  0.9943  1604 ns  1.50%

The columns are the length of the reset, the ground-state population
after it, the length of one round, and the idle error of a data qubit
over that round.

.. themed-figure:: figures/control_comparison.svg
   :alt: Four bar charts side by side, one bar per control system: the length of the reset of Z4, the error of that reset, the length of one round of error correction, and the idle error of a data qubit over that round. one controller, 100 ns feedback: reset 380 ns with error 0.57%, round 1324 ns with idle error 1.23%; one controller, 400 ns feedback: reset 680 ns with error 1.64%, round 1624 ns with idle error 1.51%; a controller per feedline: reset 380 ns with error 0.57%, round 1324 ns with idle error 1.23%; drive and readout modules: reset 580 ns with error 1.28%, round 1524 ns with idle error 1.42%; six drive lanes: reset 380 ns with error 0.57%, round 1604 ns with idle error 1.50%.
   :width: 100%

   The table as a chart. Each panel has its own scale.

What to notice:

- feedback latency is paid twice: the reset is worse, and the round is
  longer, so the data qubits lose more while they wait;
- where the boxes are split matters more than how many there are. Four
  controllers that each keep a qubit's drive and readout together cost
  nothing; two that separate them pay for the hop on every branch;
- a limit on drives at once does not touch a single-qubit reset, yet it
  stretches the round almost as much as the slow feedback does.

Where to go next
----------------

- :example:`30_control_systems_compared.py` is this comparison as one
  example file.
- :example:`23_define_a_qpu.py` loads controllers from a description file
  and saves the QPU with its control system as one system file.
- :example:`transmon/transmon_surface17_multiplexed_readout.py` shows a
  shared feedline in the controller export.
- :doc:`isa_limits` shows more of what the compiler checks: clocks,
  instruction memory and decoder calls.

For the details, :doc:`../user_guide/transmon_surface17` describes this
QPU, and :doc:`../user_guide/controller_backends` every field of a
``Controller`` and a ``ControlSystem``.
