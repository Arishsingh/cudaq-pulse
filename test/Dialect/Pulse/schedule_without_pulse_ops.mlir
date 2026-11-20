// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --pulse-schedule-asap | FileCheck %s
// RUN: cudaq-pulse-opt %s --pulse-schedule-alap | FileCheck %s
// RUN: cudaq-pulse-opt %s --pulse-schedule-rcp | FileCheck %s

// A schedule pass marks every function with its schedule state, an attribute
// of the pulse dialect. It did so without loading that dialect, and crashed
// on a module that holds no pulse operation or type.

// CHECK-LABEL: func.func @no_pulse_operation
// CHECK-SAME: attributes {pulse.schedule_state = #pulse.schedule_state<timed>}
// CHECK-NEXT: arith.addf
func.func @no_pulse_operation(%x: f64) -> f64 {
  %y = arith.addf %x, %x : f64
  return %y : f64
}

// CHECK-LABEL: func.func @only_a_classical_link
// CHECK-SAME: attributes {pulse.schedule_state = #pulse.schedule_state<timed>}
// CHECK-NEXT: classical.get_link
func.func @only_a_classical_link() {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  return
}
