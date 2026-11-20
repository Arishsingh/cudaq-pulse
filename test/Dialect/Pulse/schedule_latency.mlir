// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Every policy makes a readout's outcome usable readout-latency cycles
// after the readout ends on its line, the line's fine delay rounded up
// (fine_cycles) included; only rcp holds the readout line for it. Qudit
// 0's readout line ends its readouts 1 cycle late; qudit 1's does not.
// RUN: cudaq-pulse-opt '--pulse-schedule-asap=readout-latency=400' %s | FileCheck %s --check-prefixes=CHECK,FREE
// RUN: cudaq-pulse-opt '--pulse-schedule-alap=readout-latency=400' %s | FileCheck %s --check-prefixes=CHECK,FREE
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=readout-latency=400' %s | FileCheck %s --check-prefixes=CHECK,HELD
// Without the option, asap waits for the readout's end only.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefix=NONE

!d = !pulse.drive_line
!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

module attributes {pulse.qudit_count = 2 : i64,
    pulse.control_system = #pulse.control_system<clock_period_fs = 500000,
      lines = [#pulse.line<readout, qudits = [0], fine_cycles = 1>]>} {

// CHECK-LABEL: func.func @direct
// CHECK: pulse.readout {{.*}}start_cycles = 0 : i64
// CHECK: pulse.drive {{.*}}start_cycles = 800 : i64
// NONE-LABEL: func.func @direct
// NONE: pulse.drive {{.*}}start_cycles = 400 : i64
func.func @direct() {
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 5.000000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %ro = pulse.square %c400, %a, %z : i64, f64, f64 -> !w
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q1 : (!pulse.qudit) -> (!r, !t)
  %d0, %dt0 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, discriminated : !r, !w, !t -> !r, !t, !m
  %bit = pulse.is_set %m : !m -> i1
  %d2, %dt2 = scf.if %bit -> (!d, !t) {
    %d, %t = pulse.drive %d0, %x, %dt0 : !d, !w, !t -> !d, !t
    scf.yield %d, %t : !d, !t
  } else {
    scf.yield %d0, %dt0 : !d, !t
  }
  return
}

// The fine delay of qudit 0's readout line: 1 cycle more.
// CHECK-LABEL: func.func @fine
// CHECK: pulse.drive {{.*}}start_cycles = 801 : i64
// NONE-LABEL: func.func @fine
// NONE: pulse.drive {{.*}}start_cycles = 401 : i64
func.func @fine() {
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 5.000000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %ro = pulse.square %c400, %a, %z : i64, f64, f64 -> !w
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %d0, %dt0 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, discriminated : !r, !w, !t -> !r, !t, !m
  %bit = pulse.is_set %m : !m -> i1
  %d2, %dt2 = scf.if %bit -> (!d, !t) {
    %d, %t = pulse.drive %d0, %x, %dt0 : !d, !w, !t -> !d, !t
    scf.yield %d, %t : !d, !t
  } else {
    scf.yield %d0, %dt0 : !d, !t
  }
  return
}

// Nothing depends on the first readout: asap and alap start the second
// where it ends; rcp holds the line for the latency.
// CHECK-LABEL: func.func @two_readouts
// CHECK: pulse.readout {{.*}}start_cycles = 0 : i64
// FREE: pulse.readout {{.*}}start_cycles = 400 : i64
// HELD: pulse.readout {{.*}}start_cycles = 800 : i64
func.func @two_readouts() {
  %c400 = arith.constant 400 : i64
  %a = arith.constant 5.000000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %ro = pulse.square %c400, %a, %z : i64, f64, f64 -> !w
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q1 : (!pulse.qudit) -> (!r, !t)
  %r1, %rt1, %m1 = pulse.readout %r0, %ro, %rt0, discriminated : !r, !w, !t -> !r, !t, !m
  %r2, %rt2, %m2 = pulse.readout %r1, %ro, %rt1, discriminated : !r, !w, !t -> !r, !t, !m
  return
}
}
