// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --pulse-schedule-asap | FileCheck %s
// RUN: cudaq-pulse-opt %s --pulse-schedule-alap | FileCheck %s
// RUN: cudaq-pulse-opt %s --pulse-inline-schedule-blocks | FileCheck %s
// RUN: cudaq-pulse-opt %s --pulse-simplify-control-flow | FileCheck %s --check-prefix=SIMPLIFY
// RUN: not cudaq-pulse-opt %s --pulse-schedule-rcp 2>&1 | FileCheck %s --check-prefix=RCP

// A schedule block that yields one of its own arguments passes that input
// through. Inlining the block destroys its arguments, and the passes that
// inline one once replaced its result by the destroyed argument: a use after
// free. The result is the input. The rcp pass refuses schedule blocks.

// CHECK-LABEL: func.func @through_a_schedule_block
// CHECK-SAME: (%[[M:.*]]: !pulse.measurement, %{{.*}}: !pulse.drive_line)
// CHECK-NOT: pulse.schedule_block
// CHECK: pulse.is_set %[[M]]
// CHECK: %[[RES:.*]] = pulse.is_set %[[M]]
// CHECK-NEXT: return %[[RES]]
// SIMPLIFY-LABEL: func.func @through_a_schedule_block
// SIMPLIFY-SAME: (%[[M:.*]]: !pulse.measurement, %{{.*}}: !pulse.drive_line)
// SIMPLIFY-NOT: pulse.schedule_block
// SIMPLIFY: %[[RES:.*]] = pulse.is_set %[[M]]
// SIMPLIFY-NEXT: return %[[RES]]
// RCP: error: resource-constrained scheduling (rcp, alap_rcp) does not support kernels with their own schedule yet
func.func @through_a_schedule_block(%m: !pulse.measurement, %d: !pulse.drive_line) -> i1 {
  %d1, %m1 = pulse.schedule_block asap (%d, %m) : (!pulse.drive_line, !pulse.measurement) -> (!pulse.drive_line, !pulse.measurement) {
  ^bb0(%db: !pulse.drive_line, %mb: !pulse.measurement):
    %bit = pulse.is_set %mb : !pulse.measurement -> i1
    pulse.yield %db, %mb : !pulse.drive_line, !pulse.measurement
  }
  %res = pulse.is_set %m1 : !pulse.measurement -> i1
  return %res : i1
}

// The block of test/Dialect/Pulse/timing_properties.mlir: both results are
// arguments, and the measurement input is not among them.

// CHECK-LABEL: func.func @yields_two_of_three
// CHECK-SAME: (%{{.*}}: !pulse.measurement, %{{.*}}: !pulse.drive_line, %[[T:.*]]: !pulse.tone, %{{.*}}: f64)
// CHECK-NOT: pulse.schedule_block
// CHECK: pulse.shift_phase %[[T]]
// SIMPLIFY-LABEL: func.func @yields_two_of_three
// SIMPLIFY-NOT: pulse.schedule_block
// SIMPLIFY: pulse.shift_phase %arg2
func.func @yields_two_of_three(%m0: !pulse.measurement, %d0: !pulse.drive_line, %t0: !pulse.tone, %x: f64) {
  %d1, %t1 = pulse.schedule_block alap (%m0, %d0, %t0) : (!pulse.measurement, !pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {
  ^bb0(%m: !pulse.measurement, %d: !pulse.drive_line, %t: !pulse.tone):
    pulse.yield %d, %t : !pulse.drive_line, !pulse.tone
  }
  %t3 = pulse.shift_phase %t1, %x : !pulse.tone, f64 -> !pulse.tone
  return
}
