// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A measurement carried to the next iteration of a loop: under every policy
// a branch on it starts once its readout has ended, the readout before the
// loop included.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,EARLY
// RUN: cudaq-pulse-opt --pulse-schedule-rcp %s | FileCheck %s --check-prefixes=CHECK,EARLY
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,LATE
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap' %s | FileCheck %s --check-prefixes=CHECK,LATE
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap readout-latency=400' %s | FileCheck %s --check-prefix=LATENCY
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-rcp --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap' --pulse-verify %s > /dev/null

!d = !pulse.drive_line
!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

// Each round branches on the previous round's readout. Before, alap slid
// the readout before the loop to 240..640, past the first branch at 400.
// CHECK-LABEL: func.func @carried_round
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 640>, start_cycles = 400 : i64}>
// EARLY: pulse.readout {{.*}}<{duration_cycles = 400 : i64, period_cycles = array<i64: 400>, start_cycles = 400 : i64}>
// LATE: pulse.readout {{.*}}<{duration_cycles = 400 : i64, period_cycles = array<i64: 640>, start_cycles = 640 : i64}>
func.func @carried_round() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %r0, %tr0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %c600 = arith.constant 600 : i64
  %w600 = pulse.duration_from_int %c600 : (i64) -> !pulse.duration
  %lb = arith.constant 0 : index
  %ub = arith.constant 40 : index
  %step = arith.constant 1 : index
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %m0 = pulse.readout %r0, %ro, %tr0, discriminated : !r, !w, !t -> !r, !t, !m
  %res:5 = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0, %m = %m0, %r = %r1, %tr = %tr1) -> (!d, !t, !m, !r, !t) {
    %bit = pulse.is_set %m : !m -> i1
    %d2, %t2 = scf.if %bit -> (!d, !t) {
      %d3, %t3 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
      scf.yield %d3, %t3 : !d, !t
    } else {
      scf.yield %d, %t : !d, !t
    }
    %d4 = pulse.wait %d2, %w600 : (!d, !pulse.duration) -> !d
    %r2, %tr2, %m2 = pulse.readout %r, %ro, %tr, discriminated : !r, !w, !t -> !r, !t, !m
    scf.yield %d4, %t2, %m2, %r2, %tr2 : !d, !t, !m, !r, !t
  }
  return
}

// Repeat until success: the arm measures again. Before, alap and alap_rcp
// put the first readout at 360..760 and the first drive at 400; with a
// latency of 400, alap_rcp put the readout at 360 and the drive at 800.
// CHECK-LABEL: func.func @rus_long_arm
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 360 : i64, period_cycles = array<i64: 760>, start_cycles = 400 : i64}>
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, period_cycles = array<i64: 760>, start_cycles = 760 : i64}>
// LATENCY-LABEL: func.func @rus_long_arm
// LATENCY: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// LATENCY: pulse.drive {{.*}}<{duration_cycles = 360 : i64, period_cycles = array<i64: 1160>, start_cycles = 800 : i64}>
// LATENCY: pulse.readout {{.*}}<{duration_cycles = 400 : i64, period_cycles = array<i64: 1160>, start_cycles = 1160 : i64}>
func.func @rus_long_arm() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!d, !t)
  %r0, %tr0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c400 = arith.constant 400 : i64
  %c360 = arith.constant 360 : i64
  %lb = arith.constant 0 : index
  %ub = arith.constant 20 : index
  %step = arith.constant 1 : index
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x = pulse.square %c360, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %m0 = pulse.readout %r0, %ro, %tr0, discriminated : !r, !w, !t -> !r, !t, !m
  %res:5 = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0, %m = %m0, %r = %r1, %tr = %tr1) -> (!d, !t, !m, !r, !t) {
    %bit = pulse.is_set %m : !m -> i1
    %out:5 = scf.if %bit -> (!d, !t, !m, !r, !t) {
      %d3, %t3 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
      %ds, %rs = pulse.sync %d3, %r : !d, !r -> !d, !r
      %r2, %tr2, %m2 = pulse.readout %rs, %ro, %tr, discriminated : !r, !w, !t -> !r, !t, !m
      scf.yield %ds, %t3, %m2, %r2, %tr2 : !d, !t, !m, !r, !t
    } else {
      scf.yield %d, %t, %m, %r, %tr : !d, !t, !m, !r, !t
    }
    scf.yield %out#0, %out#1, %out#2, %out#3, %out#4 : !d, !t, !m, !r, !t
  }
  return
}

// The drive is longer than the readout. Before, alap put the readout
// before the loop at 360..400, after the first drive started at 40.
// CHECK-LABEL: func.func @carried_longdrive
// CHECK: pulse.readout {{.*}}<{duration_cycles = 40 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 400 : i64, period_cycles = array<i64: 400>, start_cycles = 40 : i64}>
// EARLY: pulse.readout {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 40>, start_cycles = 40 : i64}>
// LATE: pulse.readout {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 400>, start_cycles = 400 : i64}>
func.func @carried_longdrive() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %r0, %tr0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %lb = arith.constant 0 : index
  %ub = arith.constant 40 : index
  %step = arith.constant 1 : index
  %ro = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %x = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %m0 = pulse.readout %r0, %ro, %tr0, discriminated : !r, !w, !t -> !r, !t, !m
  %res:5 = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0, %m = %m0, %r = %r1, %tr = %tr1) -> (!d, !t, !m, !r, !t) {
    %bit = pulse.is_set %m : !m -> i1
    %d2, %t2 = scf.if %bit -> (!d, !t) {
      %d3, %t3 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
      scf.yield %d3, %t3 : !d, !t
    } else {
      scf.yield %d, %t : !d, !t
    }
    %r2, %tr2, %m2 = pulse.readout %r, %ro, %tr, discriminated : !r, !w, !t -> !r, !t, !m
    scf.yield %d2, %t2, %m2, %r2, %tr2 : !d, !t, !m, !r, !t
  }
  return
}

// The readout comes before the branch on the previous one (unchanged).
// CHECK-LABEL: func.func @p4
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, period_cycles = array<i64: 400>, start_cycles = 400 : i64}>
// EARLY: pulse.drive {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 400>, start_cycles = 400 : i64}>
// LATE: pulse.drive {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 40>, start_cycles = 7600 : i64}>
func.func @p4() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %r0, %tr0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %lb = arith.constant 0 : index
  %ub = arith.constant 20 : index
  %step = arith.constant 1 : index
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %m0 = pulse.readout %r0, %ro, %tr0, discriminated : !r, !w, !t -> !r, !t, !m
  %res:5 = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0, %m = %m0, %r = %r1, %tr = %tr1) -> (!d, !t, !m, !r, !t) {
    %r2, %tr2, %m2 = pulse.readout %r, %ro, %tr, discriminated : !r, !w, !t -> !r, !t, !m
    %bit = pulse.is_set %m : !m -> i1
    %d2, %t2 = scf.if %bit -> (!d, !t) {
      %d3, %t3 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
      scf.yield %d3, %t3 : !d, !t
    } else {
      scf.yield %d, %t : !d, !t
    }
    scf.yield %d2, %t2, %m2, %r2, %tr2 : !d, !t, !m, !r, !t
  }
  return
}

// The measurement is carried through both loops of a nest. Before, alap
// put the readout before the nest at 360..400, after the first drive.
// CHECK-LABEL: func.func @nested
// CHECK: pulse.readout {{.*}}<{duration_cycles = 40 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 400 : i64, period_cycles = array<i64: 1600, 400>, start_cycles = 40 : i64}>
// EARLY: pulse.readout {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 160, 40>, start_cycles = 40 : i64}>
// LATE: pulse.readout {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 1600, 400>, start_cycles = 400 : i64}>
func.func @nested() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %r0, %tr0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %lb = arith.constant 0 : index
  %u3 = arith.constant 3 : index
  %u4 = arith.constant 4 : index
  %step = arith.constant 1 : index
  %ro = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %x = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %m0 = pulse.readout %r0, %ro, %tr0, discriminated : !r, !w, !t -> !r, !t, !m
  %outer:5 = scf.for %i = %lb to %u3 step %step iter_args(%od = %d0, %ot = %t0, %om = %m0, %or = %r1, %otr = %tr1) -> (!d, !t, !m, !r, !t) {
    %inner:5 = scf.for %j = %lb to %u4 step %step iter_args(%d = %od, %t = %ot, %m = %om, %r = %or, %tr = %otr) -> (!d, !t, !m, !r, !t) {
      %bit = pulse.is_set %m : !m -> i1
      %d2, %t2 = scf.if %bit -> (!d, !t) {
        %d3, %t3 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
        scf.yield %d3, %t3 : !d, !t
      } else {
        scf.yield %d, %t : !d, !t
      }
      %r2, %tr2, %m2 = pulse.readout %r, %ro, %tr, discriminated : !r, !w, !t -> !r, !t, !m
      scf.yield %d2, %t2, %m2, %r2, %tr2 : !d, !t, !m, !r, !t
    }
    scf.yield %inner#0, %inner#1, %inner#2, %inner#3, %inner#4 : !d, !t, !m, !r, !t
  }
  return
}
