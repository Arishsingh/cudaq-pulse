// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A held op (pulse.hold_cycles, which spacing for an update bus sets) keeps
// its line idle that many cycles before it: qudit 0's second drive starts 5
// cycles after its first ends, and the sync carries the move to qudit 1.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP

// CHECK-LABEL: func.func @held
// CHECK: pulse.drive {{.*}}<{duration_cycles = 8 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 4 : i64, start_cycles = 13 : i64}> {pulse.hold_cycles = 5 : i64}
// ASAP: pulse.drive {{.*}}<{duration_cycles = 4 : i64, start_cycles = 0 : i64}>
// ALAP: pulse.drive {{.*}}<{duration_cycles = 4 : i64, start_cycles = 13 : i64}>
// CHECK: pulse.sync
// CHECK: pulse.drive {{.*}}<{duration_cycles = 4 : i64, start_cycles = 17 : i64}>
func.func @held() {
  %qa = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %qb = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %a0, %ta = pulse.get_drive_line %qa : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %b0, %tb = pulse.get_drive_line %qb : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c8 = arith.constant 8 : i64
  %c4 = arith.constant 4 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w8 = pulse.square %c8, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %w4 = pulse.square %c4, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %a1, %ta1 = pulse.drive %a0, %w8, %ta : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %a2, %ta2 = pulse.drive %a1, %w4, %ta1 {pulse.hold_cycles = 5 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %b1, %tb1 = pulse.drive %b0, %w4, %tb : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %a3, %b2 = pulse.sync %a2, %b1 : !pulse.drive_line, !pulse.drive_line -> !pulse.drive_line, !pulse.drive_line
  %b3, %tb2 = pulse.drive %b2, %w4, %tb1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
