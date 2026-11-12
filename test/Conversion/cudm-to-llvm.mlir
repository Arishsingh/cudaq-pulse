// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --cudm-to-llvm %s | FileCheck %s

// CHECK-LABEL: func @main
func.func @main() {
  // CHECK: llvm.call @cudm_init
  %h = cudm.init_handle : !cudm.handle

  // CHECK: llvm.call @cudm_state_alloc
  %s_in = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state
  %s_out = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state

  // CHECK: llvm.call @cudm_workspace_create
  %ws = cudm.create_workspace %h : (!cudm.handle) -> !cudm.workspace

  // CHECK: llvm.call @cudm_operator_create
  %op = cudm.create_operator %h {mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.operator

  // CHECK: llvm.call @cudm_evolve
  // CHECK: llvm.call @cudm_state_capture
  %result = cudm.evolve %h, %op, %s_in, %s_out, %ws {integrator = #cudm<integrator rk4>, t_start = 0.0 : f64, t_end = 10.0 : f64, step_count = 10 : i64} : !cudm.handle, !cudm.operator, !cudm.state, !cudm.state, !cudm.workspace -> !cudm.state

  // CHECK: llvm.call @cudm_operator_destroy
  cudm.destroy_operator %op : !cudm.operator
  // CHECK: llvm.call @cudm_workspace_destroy
  cudm.destroy_workspace %ws : !cudm.workspace
  // CHECK: llvm.call @cudm_state_destroy
  cudm.destroy_state %s_out : !cudm.state
  cudm.destroy_state %s_in : !cudm.state
  // CHECK: llvm.call @cudm_destroy
  cudm.destroy_handle %h : !cudm.handle
  return
}

// CHECK-LABEL: func @main_magnus
func.func @main_magnus() {
  %h = cudm.init_handle : !cudm.handle
  %s_in = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state
  %s_out = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state
  %ws = cudm.create_workspace %h : (!cudm.handle) -> !cudm.workspace
  %op = cudm.create_operator %h {mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.operator
  // The magnus integrator lowers to the dialect enum value 5.
  // CHECK: llvm.mlir.constant(5 : i32)
  // CHECK: llvm.call @cudm_evolve
  %result = cudm.evolve %h, %op, %s_in, %s_out, %ws {integrator = #cudm<integrator magnus>, t_start = 0.0 : f64, t_end = 10.0 : f64, step_count = 10 : i64} : !cudm.handle, !cudm.operator, !cudm.state, !cudm.state, !cudm.workspace -> !cudm.state
  cudm.destroy_operator %op : !cudm.operator
  cudm.destroy_workspace %ws : !cudm.workspace
  cudm.destroy_state %s_out : !cudm.state
  cudm.destroy_state %s_in : !cudm.state
  cudm.destroy_handle %h : !cudm.handle
  return
}

// CHECK-LABEL: func @main_crank_nicolson
func.func @main_crank_nicolson() {
  %h = cudm.init_handle : !cudm.handle
  %s_in = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state
  %s_out = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state
  %ws = cudm.create_workspace %h : (!cudm.handle) -> !cudm.workspace
  %op = cudm.create_operator %h {mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.operator
  // The crank_nicolson integrator lowers to the dialect enum value 6.
  // CHECK: llvm.mlir.constant(6 : i32)
  // CHECK: llvm.call @cudm_evolve
  %result = cudm.evolve %h, %op, %s_in, %s_out, %ws {integrator = #cudm<integrator crank_nicolson>, t_start = 0.0 : f64, t_end = 10.0 : f64, step_count = 10 : i64} : !cudm.handle, !cudm.operator, !cudm.state, !cudm.state, !cudm.workspace -> !cudm.state
  cudm.destroy_operator %op : !cudm.operator
  cudm.destroy_workspace %ws : !cudm.workspace
  cudm.destroy_state %s_out : !cudm.state
  cudm.destroy_state %s_in : !cudm.state
  cudm.destroy_handle %h : !cudm.handle
  return
}

// A superoperator on one qubit: one elementary operator on its ket and bra
// sides takes an entry per mode in the mode and duality arrays.
// CHECK-LABEL: func @superoperator_product
func.func @superoperator_product() {
  %h = cudm.init_handle : !cudm.handle
  %data = arith.constant dense<0.0> : tensor<4x4x2xf64>
  // CHECK: llvm.call @cudm_elementary_op_create
  %e = cudm.create_elementary_op %h, %data {sparsity = #cudm<sparsity none>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2, 2>} : !cudm.handle, tensor<4x4x2xf64> -> !cudm.elementary_op
  %term = cudm.create_op_term %h {mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.op_term
  // CHECK: llvm.call @cudm_op_term_create
  // CHECK: llvm.alloca %{{.*}} x f64
  // CHECK: %[[OPS:.*]] = llvm.alloca %{{.*}} x !llvm.ptr
  // CHECK-NEXT: %[[COUNT:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK-NEXT: %[[MODES:.*]] = llvm.alloca %[[COUNT]] x i32
  // CHECK-NEXT: %[[DUALITY:.*]] = llvm.alloca %[[COUNT]] x i32
  // CHECK: %[[SECOND:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: llvm.getelementptr %[[MODES]][%[[SECOND]]]
  // CHECK-NOT: llvm.getelementptr %[[OPS]][%[[SECOND]]]
  // CHECK: %[[BRA:.*]] = llvm.getelementptr %[[DUALITY]][%[[SECOND]]]
  // CHECK-NEXT: %[[ONE_I32:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.store %[[ONE_I32]], %[[BRA]]
  // CHECK-NEXT: llvm.call @cudm_op_term_append(%{{.*}}, %{{.*}}, %[[OPS]], %[[MODES]], %[[DUALITY]],
  cudm.append_elementary_product %h, %term, %e {modes_acted_on = array<i32: 0, 0>, duality = array<i32: 0, 1>, coeff_real = 1.0 : f64, coeff_imag = 0.0 : f64} : !cudm.handle, !cudm.op_term, !cudm.elementary_op
  cudm.destroy_op_term %term : !cudm.op_term
  cudm.destroy_elementary_op %e : !cudm.elementary_op
  cudm.destroy_handle %h : !cudm.handle
  return
}
