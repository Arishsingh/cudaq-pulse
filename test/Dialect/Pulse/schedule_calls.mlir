// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --pulse-schedule-asap | FileCheck %s
// RUN: cudaq-pulse-opt %s --pulse-schedule-alap | FileCheck %s
// RUN: cudaq-pulse-opt %s --pulse-schedule-rcp | FileCheck %s

// What the schedule passes do with a function call and a function they
// cannot look into.

// A call of numbers only takes no qudit, line, tone or measurement, so the
// passes schedule around it (a call that takes one is refused:
// schedule_invalid.mlir).
func.func private @doubled(%n: i64) -> i64 {
  %m = arith.addi %n, %n : i64
  return %m : i64
}

// CHECK-LABEL: func.func @call_of_numbers
// CHECK-SAME: attributes {pulse.schedule_state = #pulse.schedule_state<timed>}
// CHECK: call @doubled
// CHECK: pulse.drive {{.*}}start_cycles = 0 : i64}>
func.func @call_of_numbers(%line: !pulse.drive_line, %tone: !pulse.tone, %n: i64) -> (!pulse.drive_line, !pulse.tone) {
  %twice = call @doubled(%n) : (i64) -> i64
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %line, %x, %tone : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return %d1, %t1 : !pulse.drive_line, !pulse.tone
}

// A declaration has no body. The passes crashed on one; they leave it as it
// is and schedule the functions beside it.
// CHECK-LABEL: func.func private @declared(i64) -> i64
// CHECK-NOT: pulse.schedule_state
// CHECK-LABEL: func.func @beside_a_declaration
// CHECK-SAME: attributes {pulse.schedule_state = #pulse.schedule_state<timed>}
// CHECK: call @declared
func.func private @declared(%n: i64) -> i64

func.func @beside_a_declaration(%n: i64) -> i64 {
  %m = call @declared(%n) : (i64) -> i64
  return %m : i64
}
