// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A classical call inside a schedule block, its future leaving the block.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-classical-check --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-classical-check --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-inline-schedule-blocks --pulse-schedule-asap %s | FileCheck %s --check-prefix=FLAT
// RUN: cudaq-pulse-opt --pulse-inline-schedule-blocks --pulse-schedule-alap %s | FileCheck %s --check-prefix=FLAT

module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>} {

classical.callable @decode_wide {dispatch_key = 1, device = 1 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 100.0,
    latency_nominal_ns = 500.0, latency_max_ns = 1000.0, mode = "bounded",
    submit = "eager", dispatch_key_override}

// A schedule block measures and submits, and returns the future without
// waiting: the parent waits after the block, and the block's shape makes the
// future available at its local due time (300 + 1 + 1000 = 1301).
// CHECK-LABEL: func.func @measure_and_submit
// CHECK: classical.submit {{.*}}start_cycles = 300 : i64}
// CHECK: classical.wait {{.*}}start_cycles = 1301 : i64}
func.func @measure_and_submit() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %l0 = classical.get_link {device = 1 : i64} : !classical.link
  %r1, %rt1, %l1, %f = pulse.schedule_block alap (%r0, %rt0, %l0)
      : (!pulse.readout_line, !pulse.tone, !classical.link)
      -> (!pulse.readout_line, !pulse.tone, !classical.link, !classical.future) {
  ^bb0(%r: !pulse.readout_line, %rt: !pulse.tone, %l: !classical.link):
    %r2, %rt2, %m = pulse.readout %r, %ro, %rt, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %l2, %g = classical.submit %l, @decode_wide(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
    pulse.yield %r2, %rt2, %l2, %g : !pulse.readout_line, !pulse.tone, !classical.link, !classical.future
  }
  %fix = classical.wait %f, @decode_wide : i1
  return
}

// A block given a future waits for the answer and changes a tone's phase on
// it: in a kernel that makes classical calls, what a branch on an answer
// yields is there once the answer is, as in the flat schedule, so the drive
// after the block waits for it. Before, the tone was taken to pass through
// the block unchanged and the drive started at 0 (asap) or 1261 (alap).
// CHECK-LABEL: func.func @answer_changes_a_tone
// CHECK: classical.wait {{.*}}start_cycles = 1301 : i64}
// CHECK: pulse.drive {{.*}}start_cycles = 1301 : i64}
// FLAT-LABEL: func.func @answer_changes_a_tone
// FLAT: classical.wait {{.*}}start_cycles = 1301 : i64}
// FLAT: pulse.drive {{.*}}start_cycles = 1301 : i64}
func.func @answer_changes_a_tone() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %d0, %t0 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %phi = arith.constant 3.14 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %l0 = classical.get_link {device = 1 : i64} : !classical.link
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %l1, %f = classical.submit %l0, @decode_wide(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %t1 = pulse.schedule_block asap (%f, %t0) : (!classical.future, !pulse.tone) -> (!pulse.tone) {
  ^bb0(%g: !classical.future, %t: !pulse.tone):
    %b = classical.wait %g, @decode_wide : i1
    %t2 = scf.if %b -> (!pulse.tone) {
      %t3 = pulse.shift_phase %t, %phi : !pulse.tone, f64 -> !pulse.tone
      scf.yield %t3 : !pulse.tone
    } else {
      scf.yield %t : !pulse.tone
    }
    pulse.yield %t2 : !pulse.tone
  }
  %d2, %t4 = pulse.drive %d0, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

}
