// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The constant Hamiltonian terms and the dissipators on few enough levels
// become a few dense blocks; time-dependent terms stay one term each.
// RUN: cudaq-pulse-opt --split-input-file --qop-to-cudm %s | FileCheck %s
// RUN: cudaq-pulse-opt --split-input-file --qop-to-cudm=max-fused-dimension=64 %s | FileCheck %s --check-prefix=WIDE
// RUN: cudaq-pulse-opt --split-input-file --qop-to-cudm=max-fused-dimension=0 %s | FileCheck %s --check-prefix=NONE
// RUN: not cudaq-pulse-opt --qop-to-cudm=operators=sparse %s 2>&1 | FileCheck %s --check-prefix=BAD

// BAD: qop-to-cudm operators must be 'fused' or 'dense' (CUDAQ_PULSE_GPU_OPERATORS), not 'sparse'

// A driven qubit that decays. Its static Z term and its dissipator become
// one superoperator: a 4 x 4 operator on the qubit's ket and bra sides. The
// drive is time-dependent and acts on each side on its own.
// CHECK-LABEL: module @driven_qubit
// CHECK: %[[S:.*]] = cudm.create_elementary_op %{{.*}}, %{{.*}} {{.*}}mode_extents = array<i64: 2, 2>{{.*}} tensor<4x4x2xf64>
// CHECK: cudm.append_elementary_product %{{.*}}, %[[ST:.*]], %[[S]] {{{.*}}duality = array<i32: 0, 1>, modes_acted_on = array<i32: 0, 0>}
// CHECK: cudm.operator_append_term %{{.*}}, %{{.*}}, %[[ST]] {coeff_imag = 0.000000e+00 : f64, coeff_real = 1.000000e+00 : f64, duality = 0 : i32}
// CHECK: %[[X:.*]] = cudm.create_elementary_op {{.*}}mode_extents = array<i64: 2>
// CHECK: cudm.append_elementary_product %{{.*}}, %[[XT:.*]], %[[X]] {callback = @drive_envelope_0_x
// CHECK: cudm.operator_append_term %{{.*}}, %{{.*}}, %[[XT]] {coeff_imag = -1.000000e+00 : f64, coeff_real = 0.000000e+00 : f64, duality = 0 : i32}
// CHECK: cudm.operator_append_term %{{.*}}, %{{.*}}, %[[XT]] {coeff_imag = 1.000000e+00 : f64, coeff_real = 0.000000e+00 : f64, duality = 1 : i32}
// CHECK-NOT: cudm.operator_append_term
// CHECK: cudm.evolve
// With nothing fused, each product is a term: the jump term acts on the
// ket and the bra side of the qubit.
// NONE-LABEL: module @driven_qubit
// NONE-NOT: mode_extents = array<i64: 2, 2>
// NONE: duality = array<i32: 0, 1>, modes_acted_on = array<i32: 0, 0>
// NONE-COUNT-3: cudm.operator_append_term
module @driven_qubit attributes {qop.qudit_count = 1 : i64, qop.t_start = 0.0 : f64, qop.t_end = 10.0 : f64, qop.step_count = 10 : i64} {
func.func @main() {
  %c0 = arith.constant 0 : i64
  %sz = qop.spin(%c0) {kind = #qop<handler_kind spin_z>} : !qop.handler
  %cz = qop.const_scalar {real = 1.5 : f64, imag = 0.0 : f64} : !qop.scalar
  %z = qop.make_product(%cz, %sz) : !qop.product
  %sx = qop.spin(%c0) {kind = #qop<handler_kind spin_x>} : !qop.handler
  %cx = qop.callback_scalar @drive_envelope_0_x : !qop.scalar
  %x = qop.make_product(%cx, %sx) : !qop.product
  %sm = qop.spin(%c0) {kind = #qop<handler_kind spin_lowering>} : !qop.handler
  %cm = qop.const_scalar {real = 0.1 : f64, imag = 0.0 : f64} : !qop.scalar
  %m = qop.make_product(%cm, %sm) : !qop.product
  %c = qop.make_sum(%m) : !qop.op
  %h = qop.make_sum(%z, %x) : !qop.op
  %l = qop.lindblad(%h, %c) : !qop.super_op
  qop.evolve %l : !qop.super_op
  return
}
}

// -----

// A decaying chain of three qubits with nearest-neighbour ZZ. The default
// limit (16) fuses neighbouring pairs, a 16 x 16 superoperator each: the
// single-qubit parts join the first pair that holds their qubit. A limit of
// 64 makes the whole chain one 64 x 64 superoperator.
// CHECK-LABEL: module @chain
// CHECK: cudm.create_elementary_op {{.*}}mode_extents = array<i64: 2, 2, 2, 2>{{.*}} tensor<16x16x2xf64>
// CHECK: duality = array<i32: 0, 0, 1, 1>, modes_acted_on = array<i32: 0, 1, 0, 1>
// CHECK: cudm.operator_append_term
// CHECK: cudm.create_elementary_op {{.*}}mode_extents = array<i64: 2, 2, 2, 2>{{.*}} tensor<16x16x2xf64>
// CHECK: duality = array<i32: 0, 0, 1, 1>, modes_acted_on = array<i32: 1, 2, 1, 2>
// CHECK: cudm.operator_append_term
// CHECK-NOT: cudm.operator_append_term
// CHECK: cudm.evolve
// WIDE-LABEL: module @chain
// WIDE: cudm.create_elementary_op {{.*}}mode_extents = array<i64: 2, 2, 2, 2, 2, 2>{{.*}} tensor<64x64x2xf64>
// WIDE: duality = array<i32: 0, 0, 0, 1, 1, 1>, modes_acted_on = array<i32: 0, 1, 2, 0, 1, 2>
// WIDE: cudm.operator_append_term
// WIDE-NOT: cudm.operator_append_term
// WIDE: cudm.evolve
module @chain attributes {qop.qudit_count = 3 : i64, qop.t_start = 0.0 : f64, qop.t_end = 10.0 : f64, qop.step_count = 10 : i64} {
func.func @main() {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c2 = arith.constant 2 : i64
  %one = qop.const_scalar {real = 1.0 : f64, imag = 0.0 : f64} : !qop.scalar
  %j = qop.const_scalar {real = 0.25 : f64, imag = 0.0 : f64} : !qop.scalar
  %g = qop.const_scalar {real = 0.05 : f64, imag = 0.0 : f64} : !qop.scalar
  %z0 = qop.spin(%c0) {kind = #qop<handler_kind spin_z>} : !qop.handler
  %z1 = qop.spin(%c1) {kind = #qop<handler_kind spin_z>} : !qop.handler
  %z2 = qop.spin(%c2) {kind = #qop<handler_kind spin_z>} : !qop.handler
  %h0 = qop.make_product(%one, %z0) : !qop.product
  %h1 = qop.make_product(%one, %z1) : !qop.product
  %h2 = qop.make_product(%one, %z2) : !qop.product
  %h01 = qop.make_product(%j, %z0, %z1) : !qop.product
  %h12 = qop.make_product(%j, %z1, %z2) : !qop.product
  %m0 = qop.spin(%c0) {kind = #qop<handler_kind spin_lowering>} : !qop.handler
  %m1 = qop.spin(%c1) {kind = #qop<handler_kind spin_lowering>} : !qop.handler
  %m2 = qop.spin(%c2) {kind = #qop<handler_kind spin_lowering>} : !qop.handler
  %l0 = qop.make_product(%g, %m0) : !qop.product
  %l1 = qop.make_product(%g, %m1) : !qop.product
  %l2 = qop.make_product(%g, %m2) : !qop.product
  %s0 = qop.make_sum(%l0) : !qop.op
  %s1 = qop.make_sum(%l1) : !qop.op
  %s2 = qop.make_sum(%l2) : !qop.op
  %h = qop.make_sum(%h0, %h1, %h2, %h01, %h12) : !qop.op
  %l = qop.lindblad(%h, %s0, %s1, %s2) : !qop.super_op
  qop.evolve %l : !qop.super_op
  return
}
}

// -----

// Two coupled qutrits without dissipation, evolved as a state vector: their
// static terms sum to one 9 x 9 operator on both qutrits, on the ket side.
// CHECK-LABEL: module @qutrit_pair
// CHECK: cudm.create_elementary_op {{.*}}mode_extents = array<i64: 3, 3>{{.*}} tensor<9x9x2xf64>
// CHECK: duality = array<i32: 0, 0>, modes_acted_on = array<i32: 0, 1>
// CHECK: cudm.operator_append_term {{.*}}coeff_real = 1.000000e+00 : f64, duality = 0 : i32}
// CHECK-NOT: cudm.operator_append_term
// CHECK: cudm.evolve
module @qutrit_pair attributes {qop.mode_extents = array<i64: 3, 3>, qop.t_start = 0.0 : f64, qop.t_end = 10.0 : f64, qop.step_count = 10 : i64} {
func.func @main() {
  %n = arith.constant dense<[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]]]> : tensor<3x3x2xf64>
  %n0 = qop.matrix_leaf(%n) {dimensions = array<i64: 3>, targets = array<i64: 0>} : tensor<3x3x2xf64> -> !qop.handler
  %n1 = qop.matrix_leaf(%n) {dimensions = array<i64: 3>, targets = array<i64: 1>} : tensor<3x3x2xf64> -> !qop.handler
  %w0 = qop.const_scalar {real = 0.3 : f64, imag = 0.0 : f64} : !qop.scalar
  %w1 = qop.const_scalar {real = 0.4 : f64, imag = 0.0 : f64} : !qop.scalar
  %k = qop.const_scalar {real = 0.01 : f64, imag = 0.0 : f64} : !qop.scalar
  %p0 = qop.make_product(%w0, %n0) : !qop.product
  %p1 = qop.make_product(%w1, %n1) : !qop.product
  %p01 = qop.make_product(%k, %n0, %n1) : !qop.product
  %h = qop.make_sum(%p0, %p1, %p01) : !qop.op
  %l = qop.lindblad(%h, ) : !qop.super_op
  qop.evolve %l : !qop.super_op
  return
}
}
