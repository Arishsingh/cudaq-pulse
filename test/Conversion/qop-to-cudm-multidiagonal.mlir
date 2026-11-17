// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// On a mode of four or more levels, an operator with one nonzero diagonal is
// stored by that diagonal. A drive's a + a^dagger has two and stays dense.
// RUN: cudaq-pulse-opt --qop-to-cudm=operators=banded %s | FileCheck %s --check-prefix=BANDED
// RUN: cudaq-pulse-opt --qop-to-cudm %s | FileCheck %s --check-prefix=FUSED
// RUN: cudaq-pulse-opt --qop-to-cudm=operators=dense %s | FileCheck %s --check-prefix=DENSE
// RUN: cudaq-pulse-opt --qop-to-cudm %s --cudm-to-llvm | FileCheck %s --check-prefix=LLVM

// The number operator, the drive, and the lowering operator and its adjoint
// in the jump term; the anticommutator a^dagger a is one diagonal operator.
// BANDED: %[[N:.*]] = cudm.create_elementary_op {{.*}}diagonal_offsets = array<i32: 0>{{.*}}sparsity = #cudm<sparsity multidiagonal>{{.*}}tensor<1x5x2xf64>
// BANDED: %[[X:.*]] = cudm.create_elementary_op {{.*}}sparsity = #cudm<sparsity none>{{.*}}tensor<5x5x2xf64>
// BANDED: %[[A:.*]] = cudm.create_elementary_op {{.*}}diagonal_offsets = array<i32: 1>{{.*}}tensor<1x5x2xf64>
// BANDED: %[[AD:.*]] = cudm.create_elementary_op {{.*}}diagonal_offsets = array<i32: -1>{{.*}}tensor<1x5x2xf64>
// BANDED: cudm.append_elementary_product %{{.*}}, %{{.*}}, %[[A]], %[[AD]] {{{.*}}duality = array<i32: 0, 1>
// BANDED: %[[NORM:.*]] = cudm.create_elementary_op {{.*}}diagonal_offsets = array<i32: 0>{{.*}}tensor<1x5x2xf64>
// BANDED: cudm.append_elementary_product %{{.*}}, %[[NT:.*]], %[[NORM]] {{{.*}}duality = array<i32: 0>, modes_acted_on = array<i32: 0>}
// BANDED-COUNT-2: cudm.operator_append_term %{{.*}}, %{{.*}}, %[[NT]] {coeff_imag = 0.000000e+00 : f64, coeff_real = -0.0050000
// Fused, the static term and the anticommutator sum to one diagonal per side.
// FUSED: cudm.create_elementary_op {{.*}}diagonal_offsets = array<i32: 0>{{.*}}tensor<1x5x2xf64>
// FUSED: cudm.operator_append_term {{.*}}coeff_real = 1.000000e+00 : f64, duality = 0 : i32}
// FUSED: cudm.create_elementary_op {{.*}}diagonal_offsets = array<i32: 0>{{.*}}tensor<1x5x2xf64>
// FUSED: cudm.operator_append_term {{.*}}coeff_real = 1.000000e+00 : f64, duality = 1 : i32}
// FUSED: diagonal_offsets = array<i32: 1>
// FUSED: diagonal_offsets = array<i32: -1>
// DENSE-NOT: multidiagonal
// LLVM: llvm.call @cudm_elementary_op_create_multidiagonal
// LLVM: llvm.call @cudm_elementary_op_create(
module @ladder attributes {qop.mode_extents = array<i64: 5>, qop.t_start = 0.0 : f64, qop.t_end = 10.0 : f64, qop.step_count = 10 : i64} {
func.func @main() {
  %n = arith.constant dense<[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [4.0, 0.0]]]> : tensor<5x5x2xf64>
  %x = arith.constant dense<[[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0], [1.4142135623730951, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.4142135623730951, 0.0], [0.0, 0.0], [1.7320508075688772, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [1.7320508075688772, 0.0], [0.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]]> : tensor<5x5x2xf64>
  %a = arith.constant dense<[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.4142135623730951, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [1.7320508075688772, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]]> : tensor<5x5x2xf64>
  %ln = qop.matrix_leaf(%n) {dimensions = array<i64: 5>, targets = array<i64: 0>} : tensor<5x5x2xf64> -> !qop.handler
  %lx = qop.matrix_leaf(%x) {dimensions = array<i64: 5>, targets = array<i64: 0>} : tensor<5x5x2xf64> -> !qop.handler
  %la = qop.matrix_leaf(%a) {dimensions = array<i64: 5>, targets = array<i64: 0>} : tensor<5x5x2xf64> -> !qop.handler
  %w = qop.const_scalar {real = 0.5 : f64, imag = 0.0 : f64} : !qop.scalar
  %pn = qop.make_product(%w, %ln) : !qop.product
  %cx = qop.callback_scalar @drive_envelope_0_x : !qop.scalar
  %px = qop.make_product(%cx, %lx) : !qop.product
  %g = qop.const_scalar {real = 0.1 : f64, imag = 0.0 : f64} : !qop.scalar
  %pa = qop.make_product(%g, %la) : !qop.product
  %c = qop.make_sum(%pa) : !qop.op
  %h = qop.make_sum(%pn, %px) : !qop.op
  %l = qop.lindblad(%h, %c) : !qop.super_op
  qop.evolve %l : !qop.super_op
  return
}
}
