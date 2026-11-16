..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

T-State Injection and a Heralded Repeat-Until-Success Loop
==========================================================

This tutorial prepares the magic state ``|T>`` on one transmon qubit, measures
how good it is, and then makes it better with a herald: a readout, a decision
on the controller and a correcting pulse, repeated until the qubit starts
where it should. At the end you have a T state whose infidelity drops from
about 3e-2 to below 1e-3, and you know where the rest of the error comes from.

Every number on this page is simulated with ``pulse.simulate`` on the
``transmon_surface17`` model, except the run times, which are measured on a
laptop CPU. The whole page runs in a few seconds (measured). It builds on
:doc:`measure_and_react` and :doc:`classical_calls`.

Why T states
------------

A fault-tolerant quantum computer gets its Clifford gates (H, S, CNOT) cheaply
from its error-correcting code, but it also needs one non-Clifford gate, and
the usual choice is :math:`T = \mathrm{diag}(1, e^{i\pi/4})`. It makes T
gates by consuming magic states,

.. math::

   |T\rangle = T|+\rangle = \frac{|0\rangle + e^{i\pi/4}|1\rangle}{\sqrt 2},

which it prepares on a physical qubit, injects into the code, and then purifies
by distillation (Bravyi and Kitaev, PRA 71, 022316 (2005); Litinski,
Quantum 3, 205 (2019)) or cultivation (Gidney, Shutty and Jones,
arXiv:2409.17595 (2024)). The cost of those steps grows quickly with the
error of the state they start from, so a better physical ``|T>`` makes
everything after it cheaper.

The T gate is a frame change
----------------------------

On a transmon, a Z rotation needs no pulse. Shifting the phase of the drive
tone's frame turns every later pulse about a rotated axis, which is the same
as rotating the qubit about Z. CUDA-Q Pulse gives the phase gates ``Z``,
``S`` and ``T`` as these virtual Z rotations. This tutorial uses qubit D5,
the middle data qubit of the 17-qubit model:

.. code-block:: python

   import math

   import numpy as np

   import cudaq_pulse as pulse

   qs = pulse.quantum_system("transmon_surface17")
   q = qs.qudit("D5")

   t = q.get_calibration(pulse.gates.T)
   print(t)   # VirtualZ(phase_rad=0.7853981633974483)

Preparing ``|T>`` takes a π/2 pulse about Y, which turns ``|0>`` into
``|+>``, and then the T gate:

.. code-block:: python

   @pulse.kernel
   def inject(qudit):
       d = get_drive_line(qudit)
       drive(d, d.get_calibration(pulse.gates.Y90))   # |0> to |+>
       drive(d, d.get_calibration(pulse.gates.T))     # |+> to |T>

   for event in inject.lower(q).schedule:
       print(event.kind, event.start_cycles, event.duration_cycles)
   # drive 0 80

The schedule holds one pulse, the 40 ns Y90 (80 cycles of the 2 GHz clock).
``drive(d, t)`` played no pulse: it advanced the phase of ``d.tone`` by π/4.
The T gate takes no time and adds no error, so everything that limits this
``|T>`` comes from the Y90 pulse, the qubit's decoherence, and the state the
qubit starts in.

Reading the state in the frame
------------------------------

The frame lives on the control side. It changes what later pulses do, never
the qubit's state itself, so the state that ``pulse.simulate`` returns is the
same with and without the T gate:

.. code-block:: python

   @pulse.kernel
   def plus(qudit):
       d = get_drive_line(qudit)
       drive(d, d.get_calibration(pulse.gates.Y90))

   ground = [1.0, 0.0, 0.0]   # |0>, in the three levels simulated per transmon
   with_t = pulse.simulate(inject, q, initial_state=ground).density_matrix
   without_t = pulse.simulate(plus, q, initial_state=ground).density_matrix
   print(np.allclose(with_t, without_t))   # True

A readout straight after the T gate cannot see it either, since a Z rotation
does not change the populations a readout measures. What does see it is the
next pulse. State tomography uses that: a pulse after the state turns the X,
Y or Z axis onto Z, and a readout measures it. This kernel prepares ``|T>``
and measures one axis:

.. code-block:: python

   X_AXIS, Y_AXIS, Z_AXIS = 0, 1, 2

   @pulse.kernel
   def t_tomography(qudit, axis: int):
       d = get_drive_line(qudit)
       r = get_readout_line(qudit)
       drive(d, d.get_calibration(pulse.gates.Y90))
       drive(d, d.get_calibration(pulse.gates.T))
       if axis == X_AXIS:     # a -90 degree Y rotation turns +X onto +Z
           y90 = d.get_calibration(pulse.gates.Y90)
           drive(d, y90.replace(amplitude=-y90.amplitude))
       elif axis == Y_AXIS:   # a +90 degree X rotation turns +Y onto +Z
           drive(d, d.get_calibration(pulse.gates.X90))
       sync(d, r)
       readout(r, r.get_calibration(pulse.gates.READOUT))

Each simulation is exact: its ``branches`` hold every outcome of the readout
with its probability. The axis reads ``<Z> = 1 - 2 P(nonzero)``, and the
fidelity with ``|T>`` follows from the three axes, the Bloch vector:

.. code-block:: python

   T_BLOCH = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)

   def expectation(result):
       """<Z> of the last readout, from the probabilities of its outcomes."""
       return 1 - 2 * sum(b.probability for b in result.branches if b.levels[-1] > 0)

   def fidelity(bloch):
       """<T|rho|T> for the Bloch vector, shortened to length 1 if it is longer."""
       bloch = bloch / max(1.0, float(np.linalg.norm(bloch)))
       return (1 + float(bloch @ T_BLOCH)) / 2

``expectation`` reads one axis from a simulation and ``fidelity`` combines the
three. Measure all three axes from ``|0>``:

.. code-block:: python

   bloch = np.array([
       expectation(pulse.simulate(t_tomography, q, axis, initial_state=ground))
       for axis in range(3)
   ])
   print(f"{fidelity(bloch):.2f}")    # 1.00, simulated
   print(1 - fidelity(bloch) < 1e-3)   # True

From ``|0>``, the prepared state is within 1e-3 of ``|T>`` (simulated), and
the tomography pulses' own errors are part of that.

The thermal start
-----------------

A real transmon does not start in ``|0>``. It sits in a thermal mixture, and
the model starts D5 with 2.7% of its population already excited, the value
its quantum system gives. Leave out ``initial_state`` and the simulation
starts there:

.. code-block:: python

   print(f"{qs.thermal_populations[q.index]:.3f}")   # 0.027

   bloch = np.array([expectation(pulse.simulate(t_tomography, q, axis)) for axis in range(3)])
   print(f"{fidelity(bloch):.2f}")   # 0.97, simulated

The excited part of the start ends up near the opposite state, so the
infidelity is about the thermal population: 2.6e-2 (simulated), about a
thousand times the error of the pulses. That is the error to remove.

A herald on the controller
--------------------------

A herald measures the qubit before the state is prepared and keeps only the
shots that read 0. Here the decision runs on the controller as a classical
function, a ``@pulse.classical`` function called from the kernel while it
runs. The function needs a link to the hardware it runs on; this one is the
controller's own logic, with a few nanoseconds of fixed and feedback cost:

.. code-block:: python

   from cudaq_pulse.classical import Link

   control_system = pulse.ControlSystem(
       links={"controller": Link(fixed_ns=4, feedback_ns=8, slots=2)}
   )

The herald itself is one comparison, with a latency budget of tens of
nanoseconds, as :example:`28_t_state_injection.py` declares it:

.. literalinclude:: ../../examples/28_t_state_injection.py
   :language: python
   :start-after: # [start herald]
   :end-before: # [end herald]

The kernel measures, submits the outcome, and plays a π pulse when the answer
says the qubit read 1, which returns it to ``|0>`` for the next attempt. The
number of attempts is fixed when the kernel compiles, so every attempt is
scheduled, and each answer gates only its own pulse:

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

Compile it against the control system that has the controller link:

.. code-block:: python

   compiled = pulse.compile(heralded_t, quantum_system=qs, control_system=control_system)

Post-select on the answer
-------------------------

A simulation follows every outcome of every readout. Each of its branches
records the classical calls it made, and ``result.postselect`` keeps the
branches a test accepts: here, those whose last herald answer was true. The
kept result holds their state, normalized, and ``acceptance``, the fraction
of shots they are:

.. code-block:: python

   def heralded_fidelity(attempts):
       bloch, acceptance = np.zeros(3), 1.0
       for axis in range(3):
           result = pulse.simulate(compiled.lower(q, attempts, axis))
           kept = result.postselect(lambda b: b.answers("herald")[-1])
           bloch[axis], acceptance = expectation(kept), kept.acceptance
       return fidelity(bloch), acceptance

   f, acceptance = heralded_fidelity(1)
   print(f"{acceptance:.3f}")   # 0.972, simulated
   print(f"{f:.2f}")            # 1.00, simulated
   print(1 - f < 1e-3)          # True

One herald keeps 97.2% of the shots (simulated), the ones that did not start
excited, and the infidelity of the kept shots falls from 2.6e-2 to below
1e-3 (simulated), close to the result from ``|0>``.

Repeat until success
--------------------

The 2.8% of shots that did not read 0 need not all be lost. The kernel
already flipped them back to ``|0>``, so a second attempt reads most of them
as 0 and keeps them:

.. code-block:: python

   for attempts in (1, 2, 3):
       f, acceptance = heralded_fidelity(attempts)
       print(attempts, f"{acceptance:.3f}", 1 - f < 1e-3)
   # 1 0.972 True
   # 2 0.999 True
   # 3 0.999 True

.. themed-figure:: figures/t_error_budget.svg
   :alt: Infidelity of the prepared state with the T state, and the share of shots kept: start in |0⟩, 2.4e-05 and 100.0%; thermal start, 2.6e-02 and 100.0%; 1 attempt, 2.4e-05 and 97.2%; 2 attempts, 2.4e-05 and 99.9%; 3 attempts, 2.4e-05 and 99.9%.
   :width: 100%

   The infidelity and the shots kept for each start, from the page's own
   tomography: the thermal start costs 2.6e-2, and one herald brings the
   kept shots back to the ``|0>`` result.

Two attempts keep 99.9% of the shots (simulated) at the same fidelity. A third
gains almost nothing: about 1e-3 of the shots are never accepted, and most of
those start in level 2 (about 8e-4 of the thermal start, from the model's
Boltzmann ratio), which reads nonzero on every attempt because the loop's π
pulse drives only the 0-1 transition. A π pulse at the 1-2 tone after a
readout of 2 would bring them back.

When the answers are due
------------------------

Every attempt waits for its answer before its pulse can play, and a qubit
that waits decoheres. The compiler places each call and reports when its
answer is due:

.. code-block:: python

   program = compiled.lower(q, 1, 2)
   for call in program.classical_report():
       print(call.function, call.submit_cycles / qs.clock_ghz,
             call.due_cycles / qs.clock_ghz, call.slack_cycles)
   # herald 300.0 352.0 0

Times are in nanoseconds from the start of the program, from the compiled
schedule. The outcome is submitted when the 300 ns readout ends, and the
answer is due 52 ns later: the link's 4 ns, the function's 40 ns worst case
and the 8 ns feedback. ``slack_cycles`` is 0 because the π pulse that depends
on the answer is scheduled at the moment it is due.

.. themed-figure:: figures/herald_timeline.svg
   :alt: Timeline of the heralded T-state kernel with two attempts, on D5's drive and readout lines and the controller link, with a closer look at the first answer below. Each attempt reads D5 out for 300 ns and sends the outcome to the herald function, whose answer is due 52 ns later: 4 ns on the link, at most 40 ns in the function and 8 ns of feedback. A pi pulse that plays only if the qubit read 1 starts the moment the answer is due. After the second attempt, a Y90 pulse and a pi/4 frame shift prepare the T state, and a last readout measures it.
   :width: 100%

   Two attempts of ``heralded_t``, from ``compiled.lower(q, 2, 2)``. Dashed
   pulses play only on shots whose herald said no; the blue dashes are each
   pulse's Q quadrature. Below, the first answer's 52 ns, split into the
   link, the function's budget and the feedback.

Waiting is where the time goes: on D5 each microsecond the state spends
waiting costs about 1e-2 of fidelity (simulated, from its 46 us T2*), far more
than the pulses. A herald whose answer comes late, or a decoder that answers
in microseconds instead of nanoseconds, costs more than every other effect on
this page.

Where to go next
----------------

- :example:`28_t_state_injection.py` is this page as one script. It also
  checks that one X90 about the axis at 3π/4 makes the same ``|T>`` as Y90
  and T, measures the tomography with sampled shots through the readout
  model, and prints the herald's timing for two attempts.
- :example:`19_classical_calls.py` sends readouts to classical functions and
  branches on their answers, the mechanism the herald uses.
- :example:`08_readout_and_branching.py` shows the active reset that each
  attempt's correcting pulse is built from.

:ref:`virtual-z` gives the phase gates' sign convention and how a simulated
state relates to its frame, :doc:`../user_guide/classical_calls` covers
classical functions, links and their timing reports, and
:doc:`../user_guide/gpu_execution` how simulations follow measurement
branches and post-select on them.
