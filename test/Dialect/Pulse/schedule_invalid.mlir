// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-asap --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-alap --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-rcp --verify-diagnostics

// One case per diagnostic of the schedule passes that fails a program before
// it is placed. A pulse or wait whose length is not known cannot be placed:
// it is an error, not a placement that assumes some length.

// A waveform that is an argument has no known duration.
func.func @argument_waveform(%w: !pulse.waveform) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{cannot determine the duration of this pulse; specialize the waveform to a constant duration first}}
  %d1, %t1 = pulse.drive %d0, %w, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d2, %t2 = pulse.drive %d1, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// Nor does a built-in waveform whose duration is an argument.
func.func @argument_duration(%dur: i64) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %amp = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %c40 = arith.constant 40 : i64
  %w = pulse.square %dur, %amp, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %amp, %z : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{cannot determine the duration of this pulse}}
  %d1, %t1 = pulse.drive %d0, %w, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d2, %t2 = pulse.drive %d1, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// Waveform arithmetic has the duration of its operands.
func.func @scaled_argument_waveform(%w: !pulse.waveform, %d0: !pulse.drive_line, %t0: !pulse.tone) {
  %s = arith.constant 0.5 : f64
  %half = pulse.scale %w, %s : !pulse.waveform, f64 -> !pulse.waveform
  // expected-error @+1 {{cannot determine the duration of this pulse}}
  %d1, %t1 = pulse.drive %d0, %half, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// A wait whose length is an argument is only placed when it is variable,
// with its bounds.
func.func @argument_wait(%n: i64, %d0: !pulse.drive_line) {
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // expected-error @+1 {{cannot determine the length of this wait; give it a constant length, or make it a pulse.variable_wait with its bounds}}
  %d1 = pulse.wait %d0, %len : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

func.func @negative_wait(%d0: !pulse.drive_line) {
  %n = arith.constant -5 : i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // expected-error @+1 {{a wait cannot last -5 cycles}}
  %d1 = pulse.wait %d0, %len : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

func.func @loop_of_argument_trips(%n: index, %d0: !pulse.drive_line, %t0: !pulse.tone) {
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  // expected-error @+1 {{cannot schedule a loop with non-constant bounds}}
  %dn, %tn = scf.for %i = %lb to %n step %step iter_args(%d = %d0, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %d1, %t1 = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d1, %t1 : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// A qudit has one drive line however often it is acquired, so the two
// versions below would give one line two timelines.
func.func @drive_line_acquired_twice() {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  // expected-note @+1 {{acquired first here}}
  %a, %ta = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  // expected-error @+1 {{this qudit's drive line is already acquired; use the line acquired first}}
  %b, %tb = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a1, %ta1 = pulse.drive %a, %x, %ta : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %b1, %tb1 = pulse.drive %b, %x, %tb : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// A call that takes a line would be timed as taking no time, so the pulse
// after it would start while the called function's pulse still plays. The
// called function itself is scheduled.
func.func private @plays(%line: !pulse.drive_line, %tone: !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %line, %x, %tone : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return %d1, %t1 : !pulse.drive_line, !pulse.tone
}

func.func @call_that_takes_a_line(%line: !pulse.drive_line, %tone: !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {
  %c8 = arith.constant 8 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c8, %am, %z : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %line, %x, %tone : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // expected-error @+1 {{'func.call' takes or gives a qudit, a line, a tone, a measurement or a classical handle, and the schedule passes cannot tell what it plays or how long it takes; write the ops it stands for in its place before scheduling}}
  %out:2 = call @plays(%d1, %t1) : (!pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone)
  %d2, %t2 = pulse.drive %out#0, %x, %out#1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return %d2, %t2 : !pulse.drive_line, !pulse.tone
}

// -----

// So would one inside a loop that only gives a measurement.
func.func private @reads() -> !pulse.measurement {
  %none = ub.poison : !pulse.measurement
  return %none : !pulse.measurement
}

func.func @call_that_gives_a_measurement() {
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    // expected-error @+1 {{'func.call' takes or gives a qudit, a line, a tone, a measurement or a classical handle, and the schedule passes cannot tell what it plays or how long it takes; write the ops it stands for in its place before scheduling}}
    %m = func.call @reads() : () -> !pulse.measurement
  }
  return
}

// -----

// Nor can a call that takes a qudit be timed: the called function acquires
// the qudit's line itself and plays on it.
func.func private @plays_on(%q: !pulse.qudit) {
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

func.func @call_that_takes_a_qudit(%q: !pulse.qudit) {
  // expected-error @+1 {{'func.call' takes or gives a qudit, a line, a tone, a measurement or a classical handle, and the schedule passes cannot tell what it plays or how long it takes; write the ops it stands for in its place before scheduling}}
  call @plays_on(%q) : (!pulse.qudit) -> ()
  return
}

// -----

// Nor is a call timed from inside a region the passes do not go into: of
// scf they go into scf.for and scf.if.
func.func private @plays_too(%line: !pulse.drive_line, %tone: !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %line, %x, %tone : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return %d1, %t1 : !pulse.drive_line, !pulse.tone
}

func.func @call_in_a_region_that_gives_a_line(%line: !pulse.drive_line, %tone: !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {
  // expected-error @+1 {{'scf.execute_region' takes or gives a qudit, a line, a tone, a measurement or a classical handle, and the schedule passes cannot tell what it plays or how long it takes; write the ops it stands for in its place before scheduling}}
  %out:2 = scf.execute_region -> (!pulse.drive_line, !pulse.tone) {
    %inner:2 = func.call @plays_too(%line, %tone) : (!pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone)
    scf.yield %inner#0, %inner#1 : !pulse.drive_line, !pulse.tone
  }
  return %out#0, %out#1 : !pulse.drive_line, !pulse.tone
}

// -----

// A region that gives nothing still holds ops the passes would not time.
func.func @region_that_gives_nothing(%n: i64) {
  // expected-error @+1 {{'scf.execute_region' holds ops the schedule passes do not go into; write them in its place before scheduling}}
  scf.execute_region {
    %m = arith.addi %n, %n : i64
    scf.yield
  }
  return
}

// -----

// A function of several blocks: only the first would be timed.
// expected-error @+1 {{the schedule passes time a function of one block; this one branches between several}}
func.func @two_blocks(%n: i64) {
  llvm.br ^next
^next:
  return
}
