..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

T-State Injection and a Heralded Repeat-Until-Success Loop
==========================================================

This tutorial prepares the magic state ``|T>`` on one transmon, measures how
good it is, and makes it better with a herald: a readout, a decision on the
controller and a correcting pulse, repeated until the qubit starts where it
should.

Its infidelity drops from about 3e-2 to below 1e-3.

What you'll learn:

- why the T gate costs no pulse on a transmon;
- how to measure a state with tomography;
- how to herald on the controller and post-select in simulation;
- where the time and the error go.

Every number here is simulated on the 17-qubit transmon model, in a few
seconds on a laptop CPU.

The page builds on :doc:`measure_and_react` and :doc:`classical_calls`.

Why T states
------------

A fault-tolerant quantum computer gets its Clifford gates (H, S, CNOT)
cheaply from its error-correcting code. It also needs one non-Clifford
gate, usually :math:`T = \mathrm{diag}(1, e^{i\pi/4})`.

It makes T gates by consuming magic states:

.. math::

   |T\rangle = T|+\rangle = \frac{|0\rangle + e^{i\pi/4}|1\rangle}{\sqrt 2}

The state is prepared on a physical qubit, injected into the code, and then
purified:

- by distillation (Bravyi and Kitaev, PRA 71, 022316 (2005); Litinski,
  Quantum 3, 205 (2019));
- or by cultivation (Gidney, Shutty and Jones, arXiv:2409.17595 (2024)).

The cost of those steps grows quickly with the error of the starting state,
so a better physical ``|T>`` makes everything after it cheaper.

The T gate is a frame change
----------------------------

On a transmon, a Z rotation needs no pulse. Shifting the phase of the drive
tone's frame turns every later pulse about a rotated axis, which is the
same as rotating the qubit about Z.

CUDA-Q Pulse gives the phase gates Z, S and T as these virtual Z rotations.
This tutorial uses D5, the middle data qubit of the 17-qubit model:

.. code-block:: python

   import math

   import numpy as np

   import cudaq_pulse as pulse

   system = pulse.quantum_system("transmon_surface17")
   qubit = system.qudit("D5")

   t_gate = qubit.get_calibration(pulse.gates.T)
   print(t_gate)

.. code-block:: text
   :class: output

   VirtualZ(phase_rad=0.7853981633974483)

Preparing ``|T>`` takes a π/2 pulse about Y, which turns the ground state
into ``|+>``, and then the T gate:

.. code-block:: python

   @pulse.kernel
   def inject(qudit):
       drive_line = get_drive_line(qudit)
       y90 = drive_line.get_calibration(pulse.gates.Y90)
       t_gate = drive_line.get_calibration(pulse.gates.T)

       # |0> to |+>, then |+> to |T>.
       drive(drive_line, y90)
       drive(drive_line, t_gate)

List what the schedule plays:

.. code-block:: python

   print(f"{'kind':8} {'start':>6} {'length':>6}")
   for event in inject.lower(qubit).schedule:
       start = event.start_cycles
       length = event.duration_cycles
       print(f"{event.kind:8} {start:>6} {length:>6}")

.. code-block:: text
   :class: output

   kind      start length
   drive         0     80

The schedule holds one pulse, the 40 ns Y90. The T gate played no pulse: it
advanced the phase of the drive line's tone by π/4.

What to notice: the T gate takes no time and adds no error. What limits
this ``|T>`` is the Y90 pulse, the qubit's decoherence, and the state it
starts in.

Reading the state in the frame
------------------------------

The frame lives on the control side. It changes what later pulses do,
never the qubit's state itself.

So the state that ``pulse.simulate`` returns is the same with and without
the T gate. Compare it with a kernel that stops at ``|+>``:

.. code-block:: python

   @pulse.kernel
   def plus(qudit):
       drive_line = get_drive_line(qudit)
       y90 = drive_line.get_calibration(pulse.gates.Y90)
       drive(drive_line, y90)

Simulate both from ``|0>``, in the three levels simulated per transmon:

.. code-block:: python

   ground = [1.0, 0.0, 0.0]

   with_t = pulse.simulate(inject, qubit, initial_state=ground)
   without_t = pulse.simulate(plus, qubit, initial_state=ground)

   same = np.allclose(with_t.density_matrix, without_t.density_matrix)
   print("same state with and without T:", same)

.. code-block:: text
   :class: output

   same state with and without T: True

Tomography
----------

A readout right after the T gate can't see it either, since a Z rotation
doesn't change the populations a readout measures. What does see it is the
next pulse.

State tomography uses that: a pulse turns the X, Y or Z axis onto Z, and a
readout measures it. This kernel prepares ``|T>`` and measures one axis:

.. code-block:: python
   :class: continues

   X_AXIS, Y_AXIS, Z_AXIS = 0, 1, 2


   @pulse.kernel
   def t_tomography(qudit, axis: int):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       y90 = drive_line.get_calibration(pulse.gates.Y90)
       t_gate = drive_line.get_calibration(pulse.gates.T)
       x90 = drive_line.get_calibration(pulse.gates.X90)
       measurement = readout_line.get_calibration(pulse.gates.READOUT)

After the state, one pulse picks the axis:

- a -90 degree Y rotation turns +X onto +Z;
- a +90 degree X rotation turns +Y onto +Z;
- Z needs no pulse.

.. code-block:: python

       drive(drive_line, y90)
       drive(drive_line, t_gate)

       if axis == X_AXIS:
           drive(drive_line, y90.replace(amplitude=-y90.amplitude))
       elif axis == Y_AXIS:
           drive(drive_line, x90)

       sync(drive_line, readout_line)
       readout(readout_line, measurement)

From readouts to fidelity
-------------------------

Each simulation is exact: its branches hold every outcome of the readout
with its probability. An axis reads ⟨Z⟩ = 1 - 2 P(nonzero):

.. code-block:: python

   def expectation(result):
       """<Z> of the last readout, from its outcomes' probabilities."""
       excited = 0.0
       for branch in result.branches:
           if branch.levels[-1] > 0:
               excited += branch.probability
       return 1 - 2 * excited

The three axes make the Bloch vector, and its overlap with ``|T>``'s gives
the fidelity:

.. code-block:: python

   T_BLOCH = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)


   def fidelity(bloch):
       """<T|rho|T> for the Bloch vector, at most length 1."""
       length = max(1.0, float(np.linalg.norm(bloch)))
       unit = bloch / length
       return (1 + float(unit @ T_BLOCH)) / 2

Measure all three axes from ``|0>``:

.. code-block:: python

   bloch = np.zeros(3)
   for axis in (X_AXIS, Y_AXIS, Z_AXIS):
       result = pulse.simulate(
           t_tomography,
           qubit,
           axis,
           initial_state=ground,
       )
       bloch[axis] = expectation(result)

   print(f"fidelity with |T>: {fidelity(bloch):.2f}")
   print(f"infidelity < 1e-3: {1 - fidelity(bloch) < 1e-3}")

.. code-block:: text
   :class: output

   fidelity with |T>: 1.00
   infidelity < 1e-3: True

What to notice: from ``|0>``, the prepared state is within 1e-3 of
``|T>``, and that includes the tomography pulses' own errors.

The thermal start
-----------------

A real transmon doesn't start in ``|0>``. It sits in a thermal mixture,
and the model starts D5 with part of its population already excited.

Leave out ``initial_state`` and the simulation starts there:

.. code-block:: python

   excited = system.thermal_populations[qubit.index]
   print(f"excited at the start: {excited:.3f}")

   bloch = np.zeros(3)
   for axis in (X_AXIS, Y_AXIS, Z_AXIS):
       result = pulse.simulate(t_tomography, qubit, axis)
       bloch[axis] = expectation(result)
   print(f"fidelity with |T>:    {fidelity(bloch):.2f}")

.. code-block:: text
   :class: output

   excited at the start: 0.027
   fidelity with |T>:    0.97

The excited part of the start ends up near the opposite state. So the
infidelity is about the thermal population: 2.6e-2, about a thousand times
the error of the pulses.

That is the error to remove.

A herald on the controller
--------------------------

A herald measures the qubit before the state is prepared and keeps only the
shots that read 0.

Here the decision runs on the controller, as a classical function the
kernel calls while it runs. Its link is the controller's own logic, with a
few nanoseconds of fixed and feedback cost:

.. code-block:: python

   from cudaq_pulse.classical import Link

   controller_link = Link(fixed_ns=4, feedback_ns=8, slots=2)
   links = {"controller": controller_link}
   control_system = pulse.ControlSystem(links=links)

The herald itself is one comparison, with a budget of tens of nanoseconds.
The code from here on comes from :example:`28_t_state_injection.py`:

.. literalinclude:: ../../examples/28_t_state_injection.py
   :language: python
   :start-after: # [start herald]
   :end-before: # [end herald]

The kernel
----------

Each attempt measures the qubit, submits the outcome, and plays a π pulse
when the answer says the qubit read 1. That returns it to ``|0>`` for the
next attempt.

The number of attempts is fixed when the kernel compiles, so every attempt
is scheduled:

.. literalinclude:: ../../examples/28_t_state_injection.py
   :language: python
   :class: continues
   :start-after: # [start heralded attempts]
   :end-before: # [end heralded attempts]

After the attempts, the kernel prepares ``|T>`` and measures one axis, as
``t_tomography`` does:

.. literalinclude:: ../../examples/28_t_state_injection.py
   :language: python
   :start-after: # [start heralded state]
   :end-before: # [end heralded state]

Compile it against the control system with the controller link:

.. code-block:: python

   compiled = pulse.compile(
       heralded_t,
       quantum_system=system,
       control_system=control_system,
   )

Post-select on the answer
-------------------------

Each branch of a simulation records the classical calls it made.
``postselect`` keeps the branches a test accepts, here those whose last
herald answer was true:

.. code-block:: python

   def herald_accepted(branch):
       return branch.answers("herald")[-1]

The kept result holds their state, normalized, and ``acceptance``, the
share of shots they are:

.. code-block:: python

   def heralded_fidelity(attempts):
       bloch = np.zeros(3)
       acceptance = 1.0
       for axis in (X_AXIS, Y_AXIS, Z_AXIS):
           program = compiled.lower(qubit, attempts, axis)
           result = pulse.simulate(program)
           kept = result.postselect(herald_accepted)
           bloch[axis] = expectation(kept)
           acceptance = kept.acceptance
       return fidelity(bloch), acceptance

Try one attempt:

.. code-block:: python

   heralded, acceptance = heralded_fidelity(1)
   print(f"shots kept:        {acceptance:.3f}")
   print(f"fidelity with |T>: {heralded:.2f}")
   print(f"infidelity < 1e-3: {1 - heralded < 1e-3}")

.. code-block:: text
   :class: output

   shots kept:        0.972
   fidelity with |T>: 1.00
   infidelity < 1e-3: True

What to notice: one herald keeps the 97.2% of shots that didn't start
excited, and their infidelity falls from 2.6e-2 to below 1e-3, close to the
result from ``|0>``.

Repeat until success
--------------------

The 2.8% of shots that didn't read 0 need not all be lost. The kernel
already flipped them back to ``|0>``, so a second attempt reads most of
them as 0 and keeps them:

.. code-block:: python

   print("attempts   kept  infidelity < 1e-3")
   for attempts in (1, 2, 3):
       heralded, acceptance = heralded_fidelity(attempts)
       good = 1 - heralded < 1e-3
       print(f"{attempts:8}  {acceptance:.3f}  {good}")

.. code-block:: text
   :class: output

   attempts   kept  infidelity < 1e-3
          1  0.972  True
          2  0.999  True
          3  0.999  True

.. themed-figure:: figures/t_error_budget.svg
   :alt: Infidelity of the prepared state with the T state, and the share of shots kept: start in |0⟩, 2.4e-05 and 100.0%; thermal start, 2.6e-02 and 100.0%; 1 attempt, 2.4e-05 and 97.2%; 2 attempts, 2.4e-05 and 99.9%; 3 attempts, 2.4e-05 and 99.9%.
   :width: 100%

   The infidelity and the shots kept for each start, from this page's
   tomography.

What to notice:

- two attempts keep 99.9% of the shots at the same fidelity;
- a third gains almost nothing: about 1e-3 of the shots are never
  accepted.

Most of those start in level 2, about 8e-4 of the thermal start from the
model's Boltzmann ratio. They read nonzero on every attempt, because the
loop's π pulse drives only the 0-1 transition. A π pulse at the 1-2 tone
after a readout of 2 would bring them back.

When the answers are due
------------------------

Every attempt waits for its answer before its pulse can play, and a qubit
that waits decoheres. The compiler places each call and reports when its
answer is due, here in nanoseconds:

.. code-block:: python

   program = compiled.lower(qubit, 1, Z_AXIS)

   print("function  sent (ns)  due (ns)  slack")
   for call in program.classical_report():
       sent_ns = call.submit_cycles / system.clock_ghz
       due_ns = call.due_cycles / system.clock_ghz
       slack = call.slack_cycles
       print(f"{call.function:8}  {sent_ns:9}  {due_ns:8}  {slack:5}")

.. code-block:: text
   :class: output

   function  sent (ns)  due (ns)  slack
   herald        300.0     352.0      0

The outcome is sent when the 300 ns readout ends. The answer is due 52 ns
later:

- 4 ns on the link;
- at most 40 ns in the function;
- 8 ns of feedback.

The slack is 0 because the π pulse that depends on the answer is scheduled
the moment it is due.

.. themed-figure:: figures/herald_timeline.svg
   :alt: Timeline of the heralded T-state kernel with two attempts, on D5's drive and readout lines and the controller link, with a closer look at the first answer below. Each attempt reads D5 out for 300 ns and sends the outcome to the herald function, whose answer is due 52 ns later: 4 ns on the link, at most 40 ns in the function and 8 ns of feedback. A pi pulse that plays only if the qubit read 1 starts the moment the answer is due. After the second attempt, a Y90 pulse and a pi/4 frame shift prepare the T state, and a last readout measures it.
   :width: 100%

   Two attempts of ``heralded_t``. Dashed pulses play only on shots whose
   herald said no; the blue dashes are each pulse's Q quadrature. Below,
   the first answer's 52 ns.

What to notice: waiting is where the time goes. On D5, each microsecond
spent waiting costs about 1e-2 of fidelity (from its 46 µs T2*), far more
than the pulses.

A herald whose answer comes late, or a decoder that answers in
microseconds instead of nanoseconds, costs more than every other effect on
this page.

Where to go next
----------------

- :example:`28_t_state_injection.py` is this page as one script. It also
  makes the same ``|T>`` with one X90 about the axis at 3π/4, and measures
  the tomography with sampled shots.
- :example:`19_classical_calls.py` sends readouts to classical functions
  and branches on their answers, the mechanism the herald uses.
- :example:`08_readout_and_branching.py` shows the active reset each
  attempt's correcting pulse is built from.

For the details:

- :ref:`virtual-z` gives the phase gates' sign convention;
- :doc:`../user_guide/classical_calls` covers classical functions, links
  and their timing reports;
- :doc:`../user_guide/gpu_execution` covers how simulations follow
  measurement branches and post-select on them.
