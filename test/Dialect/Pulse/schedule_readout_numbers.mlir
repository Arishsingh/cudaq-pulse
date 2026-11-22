// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A readout copied by peeling or unrolling a loop keeps the kernel's number
// (pulse.readout_number) and maps its iterations to the kernel's
// (pulse.readout_iteration: depth, scale, offset per kernel loop around it,
// outermost first; depth -1 is a constant iteration).
// RUN: cudaq-pulse-opt --split-input-file --pulse-schedule-asap %s | FileCheck %s --check-prefix=ASAP
// RUN: cudaq-pulse-opt --split-input-file "--pulse-schedule-rcp=max-drives=4 max-readouts=2" %s | FileCheck %s --check-prefix=RCP

// Nested peel: line 1 is busy until 500, so asap peels the first iteration
// off the outer loop and, in both copies of it, off the inner loop. The
// readouts carry no number: they are numbered in walk order before the
// first copy. Inner loop of the peeled outer iteration: its depth moves
// from 1 to 0, iteration k + 1.
// ASAP-LABEL: module @nested_loops
// ASAP: pulse.readout {{.*}}{pulse.readout_iteration = array<i64: -1, 0, 0, -1, 0, 0>, pulse.readout_number = 0 : i64}
// ASAP: scf.for %{{.*}} = %c1{{.*}} to %c3
// ASAP: pulse.readout {{.*}}{pulse.readout_iteration = array<i64: -1, 0, 0, 0, 1, 1>, pulse.readout_number = 0 : i64}
// ASAP: scf.for %{{.*}} = %c1{{.*}} to %c2
// ASAP: pulse.readout {{.*}}{pulse.readout_iteration = array<i64: 0, 1, 1, -1, 0, 0>, pulse.readout_number = 0 : i64}
// ASAP: scf.for %{{.*}} = %c1{{.*}} to %c3
// ASAP: pulse.readout {{.*}}{pulse.readout_iteration = array<i64: 0, 1, 1, 1, 1, 1>, pulse.readout_number = 0 : i64}
// ASAP: pulse.readout {{.*}}{pulse.readout_number = 1 : i64}
// ASAP-NOT: pulse.readout_iteration
// ASAP-LABEL: module @answered
module @nested_loops attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 500000>, pulse.quantum_system = #pulse.quantum_system<qudit_frequency_hz = [3.885000e+09, 3.994000e+09]>, pulse.qudit_count = 2 : i64} {
  func.func @main() {
    %c40_i64 = arith.constant 40 : i64
    %cst = arith.constant 1.000000e+00 : f64
    %c600_i64 = arith.constant 600 : i64
    %cst_0 = arith.constant 1.0277299890703711 : f64
    %cst_1 = arith.constant 2.000000e+01 : f64
    %cst_2 = arith.constant 0.13130586525163929 : f64
    %c80_i64 = arith.constant 80 : i64
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %line, %tone = pulse.get_drive_line %0 {frequency_hz = 3.885000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
    %line_3, %tone_4 = pulse.get_drive_line %1 {frequency_hz = 3.994000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %line_5, %tone_6 = pulse.get_readout_line %1 {frequency_hz = 6.979000e+09 : f64} : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %line_7, %tone_8 = pulse.get_readout_line %0 {frequency_hz = 6.769000e+09 : f64} : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %c_busy = arith.constant 500 : i64
    %busy = pulse.duration_from_int %c_busy : (i64) -> !pulse.duration
    %line_busy = pulse.wait %line_3, %busy : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %2:5 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %line_busy, %arg2 = %line, %arg3 = %tone, %arg4 = %line_5, %arg5 = %tone_6) -> (!pulse.drive_line, !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone) {
      %5:5 = scf.for %arg6 = %c0 to %c3 step %c1 iter_args(%arg7 = %arg1, %arg8 = %arg2, %arg9 = %arg3, %arg10 = %arg4, %arg11 = %arg5) -> (!pulse.drive_line, !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone) {
        %7 = pulse.drag %c80_i64, %cst_2, %cst_1, %cst_0 : i64, f64, f64, f64 -> !pulse.waveform
        %updated_line_9, %updated_tone_10 = pulse.drive %arg8, %7, %arg9 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
        %8:3 = pulse.sync %updated_line_9, %arg7, %arg10 : !pulse.drive_line, !pulse.drive_line, !pulse.readout_line -> !pulse.drive_line, !pulse.drive_line, !pulse.readout_line
        %9 = pulse.gaussian_square %c600_i64, %cst, %cst_1, %c40_i64 : i64, f64, f64, i64 -> !pulse.waveform
        %updated_line_11, %updated_tone_12, %result_13 = pulse.readout %8#2, %9, %arg11, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
        scf.yield %8#1, %8#0, %updated_tone_10, %updated_line_11, %updated_tone_12 : !pulse.drive_line, !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone
      }
      %6:3 = pulse.sync %5#1, %5#0, %5#3 : !pulse.drive_line, !pulse.drive_line, !pulse.readout_line -> !pulse.drive_line, !pulse.drive_line, !pulse.readout_line
      scf.yield %6#1, %6#0, %5#2, %6#2, %5#4 : !pulse.drive_line, !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone
    }
    %3:2 = pulse.sync %2#1, %line_7 : !pulse.drive_line, !pulse.readout_line -> !pulse.drive_line, !pulse.readout_line
    %4 = pulse.gaussian_square %c600_i64, %cst, %cst_1, %c40_i64 : i64, f64, f64, i64 -> !pulse.waveform
    %updated_line, %updated_tone, %result = pulse.readout %3#1, %4, %tone_8, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    return
  }
}

// -----

// unrollBy and peelBack: rcp runs four of the 26 iterations in each trip
// (iteration 4k + c in copy c) and peels the two left over off the back.
// The readout's number, given (here 7), is kept by every copy.
// RCP-LABEL: module @answered
// RCP: scf.for %{{.*}} = %c0 to %c24 step %c4
// RCP: pulse.readout {{.*}}{pulse.readout_iteration = array<i64: 0, 4, 0>, pulse.readout_number = 7 : i64}
// RCP: pulse.readout {{.*}}{pulse.readout_iteration = array<i64: 0, 4, 1>, pulse.readout_number = 7 : i64}
// RCP: pulse.readout {{.*}}{pulse.readout_iteration = array<i64: 0, 4, 2>, pulse.readout_number = 7 : i64}
// RCP: pulse.readout {{.*}}{pulse.readout_iteration = array<i64: 0, 4, 3>, pulse.readout_number = 7 : i64}
// RCP: scf.yield
// RCP: pulse.readout {{.*}}{pulse.readout_iteration = array<i64: -1, 0, 24>, pulse.readout_number = 7 : i64}
// RCP: pulse.readout {{.*}}{pulse.readout_iteration = array<i64: -1, 0, 25>, pulse.readout_number = 7 : i64}
// RCP-NOT: pulse.readout {{.*}}iq
module @answered attributes {classical.links = {"0" = {bytes_per_ns = 0.000000e+00 : f64, feedback_ns = 0.000000e+00 : f64, fixed_ns = 0.000000e+00 : f64, slot_bytes = 1024 : i64, slots = 64 : i64}}, pulse.control_system = #pulse.control_system<clock_period_fs = 500000>, pulse.quantum_system = #pulse.quantum_system<qudit_frequency_hz = [5.000000e+09, 5.100000e+09]>, pulse.qudit_count = 2 : i64} {
  classical.callable @decide__1 {args = ["bits:1"], device = 0 : i64, dispatch_key = 1522877098 : i64, latency_max_ns = 4.000000e+02 : f64, latency_min_ns = 0.000000e+00 : f64, latency_nominal_ns = 4.000000e+02 : f64, mode = "bounded", results = ["bits:1"], submit = "eager"}
  func.func @main() {
    %cst = arith.constant 2.000000e-01 : f64
    %c4_i64 = arith.constant 4 : i64
    %cst_0 = arith.constant 1.000000e-01 : f64
    %c8_i64 = arith.constant 8 : i64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 5.000000e-02 : f64
    %c40_i64 = arith.constant 40 : i64
    %c1 = arith.constant 1 : index
    %c26 = arith.constant 26 : index
    %c0 = arith.constant 0 : index
    %0 = ub.poison : !pulse.measurement
    %1 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %line, %tone = pulse.get_drive_line %1 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %2 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
    %line_3, %tone_4 = pulse.get_readout_line %2 {frequency_hz = 5.100000e+09 : f64} : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %3 = classical.get_link {device = 0 : i64} : !classical.link
    %4:6 = scf.for %arg0 = %c0 to %c26 step %c1 iter_args(%arg1 = %line, %arg2 = %tone, %arg3 = %0, %arg4 = %line_3, %arg5 = %tone_4, %arg6 = %3) -> (!pulse.drive_line, !pulse.tone, !pulse.measurement, !pulse.readout_line, !pulse.tone, !classical.link) {
      %5 = pulse.square %c40_i64, %cst_2, %cst_1 : i64, f64, f64 -> !pulse.waveform
      %updated_line, %updated_tone, %result = pulse.readout %arg4, %5, %arg5, iq {pulse.readout_number = 7 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
      %updated_link, %future = classical.submit %arg6, @decide__1(%result) : (!pulse.measurement) -> (!classical.link, !classical.future)
      %6 = classical.wait %future, @decide__1 : i1
      %7:2 = scf.if %6 -> (!pulse.drive_line, !pulse.tone) {
        %8 = pulse.square %c8_i64, %cst_0, %cst_1 : i64, f64, f64 -> !pulse.waveform
        %updated_line_5, %updated_tone_6 = pulse.drive %arg1, %8, %arg2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
        scf.yield %updated_line_5, %updated_tone_6 : !pulse.drive_line, !pulse.tone
      } else {
        %8 = pulse.square %c4_i64, %cst, %cst_1 : i64, f64, f64 -> !pulse.waveform
        %updated_line_5, %updated_tone_6 = pulse.drive %arg1, %8, %arg2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
        scf.yield %updated_line_5, %updated_tone_6 : !pulse.drive_line, !pulse.tone
      }
      scf.yield %7#0, %7#1, %result, %updated_line, %updated_tone, %updated_link : !pulse.drive_line, !pulse.tone, !pulse.measurement, !pulse.readout_line, !pulse.tone, !classical.link
    }
    return
  }
}
