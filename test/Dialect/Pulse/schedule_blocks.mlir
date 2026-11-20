// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A pulse.schedule_block is scheduled on its own with its policy, then placed
// whole by the enclosing schedule and inlined.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-verify %s > /dev/null
// Without the blocks, the same ops are scheduled as one flat program.
// RUN: cudaq-pulse-opt --pulse-inline-schedule-blocks --pulse-schedule-asap %s | FileCheck %s --check-prefix=FLAT-ASAP
// RUN: cudaq-pulse-opt --pulse-inline-schedule-blocks --pulse-schedule-alap %s | FileCheck %s --check-prefix=FLAT-ALAP
// Scheduling inlines every block.
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefix=INLINED
// INLINED-NOT: pulse.schedule_block
// INLINED-NOT: pulse.yield

!d = !pulse.drive_line
!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

// An ASAP block on lines b and c: inside it both start together. The ALAP
// parent places the block as late as it can, so it ends with line a; line
// c keeps its place at the start of the block instead of moving to the end.
// CHECK-LABEL: func.func @asap_in_alap
// CHECK: pulse.drive {{.*}}<{duration_cycles = 200 : i64, start_cycles = 0 : i64}>
// ASAP: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 0 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 100 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 100 : i64}
// FLAT-ALAP-LABEL: func.func @asap_in_alap
// FLAT-ALAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 160 : i64}
func.func @asap_in_alap() {
  %qa = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %qb = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %qc = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %a0, %ta0 = pulse.get_drive_line %qa : (!pulse.qudit) -> (!d, !t)
  %b0, %tb0 = pulse.get_drive_line %qb : (!pulse.qudit) -> (!d, !t)
  %c0, %tc0 = pulse.get_drive_line %qc : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c200 = arith.constant 200 : i64
  %c100 = arith.constant 100 : i64
  %c40 = arith.constant 40 : i64
  %x200 = pulse.square %c200, %amp, %zero : i64, f64, f64 -> !w
  %x100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %a1, %ta1 = pulse.drive %a0, %x200, %ta0 : !d, !w, !t -> !d, !t
  %b1, %tb1, %c1, %tc1 = pulse.schedule_block asap (%b0, %tb0, %c0, %tc0) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
  ^bb0(%b: !d, %tb: !t, %c: !d, %tc: !t):
    %b2, %tb2 = pulse.drive %b, %x100, %tb : !d, !w, !t -> !d, !t
    %c2, %tc2 = pulse.drive %c, %x40, %tc : !d, !w, !t -> !d, !t
    pulse.yield %b2, %tb2, %c2, %tc2 : !d, !t, !d, !t
  }
  return
}

// An ALAP block in an ASAP parent: the short line c is right-aligned inside
// the block, and the drive after the block waits for the block's end on c.
// CHECK-LABEL: func.func @alap_in_asap
// CHECK: pulse.drive {{.*}}<{duration_cycles = 100 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 60 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 100 : i64}>
// FLAT-ASAP-LABEL: func.func @alap_in_asap
// FLAT-ASAP: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 0 : i64}
// FLAT-ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 0 : i64}
// FLAT-ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 40 : i64}
func.func @alap_in_asap() {
  %qb = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %qc = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %b0, %tb0 = pulse.get_drive_line %qb : (!pulse.qudit) -> (!d, !t)
  %c0, %tc0 = pulse.get_drive_line %qc : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c100 = arith.constant 100 : i64
  %c40 = arith.constant 40 : i64
  %x100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %b1, %tb1, %c1, %tc1 = pulse.schedule_block alap (%b0, %tb0, %c0, %tc0) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
  ^bb0(%b: !d, %tb: !t, %c: !d, %tc: !t):
    %b2, %tb2 = pulse.drive %b, %x100, %tb : !d, !w, !t -> !d, !t
    %c2, %tc2 = pulse.drive %c, %x40, %tc : !d, !w, !t -> !d, !t
    pulse.yield %b2, %tb2, %c2, %tc2 : !d, !t, !d, !t
  }
  %c3, %tc3 = pulse.drive %c1, %x40, %tc1 : !d, !w, !t -> !d, !t
  return
}

// Three levels: an ASAP block holding an ALAP block holding an ASAP block.
// Each level is placed whole in the one above, so start times add up.
// Level 3 runs x for 40 and y for 20 from its start. Level 2 drives x for
// 100 first and places level 3 at 100. Level 1 drives y for 60, which level
// 2 does not wait for: its first op on y is at 100. The ALAP root places
// level 1 (length 140) to end with the 300-long drive on z.
// CHECK-LABEL: func.func @three_levels
// CHECK: pulse.drive {{.*}}<{duration_cycles = 300 : i64, start_cycles = 0 : i64}>
// ASAP: pulse.drive {{.*}}{duration_cycles = 60 : i64, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 100 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 20 : i64, start_cycles = 100 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 60 : i64, start_cycles = 160 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 160 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 260 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 20 : i64, start_cycles = 260 : i64}
func.func @three_levels() {
  %qx = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %qy = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %qz = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %x0, %tx0 = pulse.get_drive_line %qx : (!pulse.qudit) -> (!d, !t)
  %y0, %ty0 = pulse.get_drive_line %qy : (!pulse.qudit) -> (!d, !t)
  %z0, %tz0 = pulse.get_drive_line %qz : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c300 = arith.constant 300 : i64
  %c100 = arith.constant 100 : i64
  %c60 = arith.constant 60 : i64
  %c40 = arith.constant 40 : i64
  %c20 = arith.constant 20 : i64
  %w300 = pulse.square %c300, %amp, %zero : i64, f64, f64 -> !w
  %w100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %w60 = pulse.square %c60, %amp, %zero : i64, f64, f64 -> !w
  %w40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %w20 = pulse.square %c20, %amp, %zero : i64, f64, f64 -> !w
  %z1, %tz1 = pulse.drive %z0, %w300, %tz0 : !d, !w, !t -> !d, !t
  %x1, %tx1, %y1, %ty1 = pulse.schedule_block asap (%x0, %tx0, %y0, %ty0) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
  ^bb0(%x: !d, %tx: !t, %y: !d, %ty: !t):
    %y2, %ty2 = pulse.drive %y, %w60, %ty : !d, !w, !t -> !d, !t
    %x3, %tx3, %y3, %ty3 = pulse.schedule_block alap (%x, %tx, %y2, %ty2) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
    ^bb0(%xx: !d, %txx: !t, %yy: !d, %tyy: !t):
      %x4, %tx4 = pulse.drive %xx, %w100, %txx : !d, !w, !t -> !d, !t
      %x5, %tx5, %y5, %ty5 = pulse.schedule_block asap (%x4, %tx4, %yy, %tyy) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
      ^bb0(%xxx: !d, %txxx: !t, %yyy: !d, %tyyy: !t):
        %x6, %tx6 = pulse.drive %xxx, %w40, %txxx : !d, !w, !t -> !d, !t
        %y6, %ty6 = pulse.drive %yyy, %w20, %tyyy : !d, !w, !t -> !d, !t
        pulse.yield %x6, %tx6, %y6, %ty6 : !d, !t, !d, !t
      }
      pulse.yield %x5, %tx5, %y5, %ty5 : !d, !t, !d, !t
    }
    pulse.yield %x3, %tx3, %y3, %ty3 : !d, !t, !d, !t
  }
  return
}

// A block in a loop has the same shape in every iteration, so the loop stays
// affine: the block's ops take the loop's period, followed by the period of
// the loop inside the block.
// CHECK-LABEL: func.func @block_in_loop
// CHECK: pulse.drive {{.*}}<{duration_cycles = 100 : i64, start_cycles = 0 : i64}>
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 100, 40>, start_cycles = 100 : i64}>
// CHECK: pulse.wait {{.*}}<{duration_cycles = 20 : i64, period_cycles = array<i64: 100>, start_cycles = 180 : i64}>
func.func @block_in_loop() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c100 = arith.constant 100 : i64
  %c40 = arith.constant 40 : i64
  %c20 = arith.constant 20 : i64
  %x100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %w20 = pulse.duration_from_int %c20 : (i64) -> !pulse.duration
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %two = arith.constant 2 : index
  %step = arith.constant 1 : index
  %d1, %t1 = pulse.drive %d0, %x100, %t0 : !d, !w, !t -> !d, !t
  %d2, %t2 = scf.for %i = %lb to %ub step %step iter_args(%d = %d1, %t = %t1) -> (!d, !t) {
    %d3, %t3 = pulse.schedule_block asap (%d, %t) : (!d, !t) -> (!d, !t) {
    ^bb0(%bd: !d, %bt: !t):
      %d4, %t4 = scf.for %j = %lb to %two step %step iter_args(%e = %bd, %u = %bt) -> (!d, !t) {
        %e1, %u1 = pulse.drive %e, %x40, %u : !d, !w, !t -> !d, !t
        scf.yield %e1, %u1 : !d, !t
      }
      %d5 = pulse.wait %d4, %w20 : (!d, !pulse.duration) -> !d
      pulse.yield %d5, %t4 : !d, !t
    }
    scf.yield %d3, %t3 : !d, !t
  }
  return
}

// A block in a measurement-conditioned arm starts once the measurement is
// available; inside it, the ALAP policy right-aligns the short line.
// CHECK-LABEL: func.func @block_in_if
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: scf.if
// CHECK: pulse.drive {{.*}}<{duration_cycles = 100 : i64, start_cycles = 400 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 460 : i64}>
func.func @block_in_if() {
  %qa = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %qb = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %tr0 = pulse.get_readout_line %qa : (!pulse.qudit) -> (!r, !t)
  %a0, %ta0 = pulse.get_drive_line %qa : (!pulse.qudit) -> (!d, !t)
  %b0, %tb0 = pulse.get_drive_line %qb : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c400 = arith.constant 400 : i64
  %c100 = arith.constant 100 : i64
  %c40 = arith.constant 40 : i64
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %m = pulse.readout %r0, %ro, %tr0, discriminated : !r, !w, !t -> !r, !t, !m
  %bit = pulse.is_set %m : !m -> i1
  %a1, %ta1, %b1, %tb1 = scf.if %bit -> (!d, !t, !d, !t) {
    %a2, %ta2, %b2, %tb2 = pulse.schedule_block alap (%a0, %ta0, %b0, %tb0) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
    ^bb0(%a: !d, %ta: !t, %b: !d, %tb: !t):
      %a3, %ta3 = pulse.drive %a, %x100, %ta : !d, !w, !t -> !d, !t
      %b3, %tb3 = pulse.drive %b, %x40, %tb : !d, !w, !t -> !d, !t
      pulse.yield %a3, %ta3, %b3, %tb3 : !d, !t, !d, !t
    }
    scf.yield %a2, %ta2, %b2, %tb2 : !d, !t, !d, !t
  } else {
    scf.yield %a0, %ta0, %b0, %tb0 : !d, !t, !d, !t
  }
  return
}

// A measurement-conditioned branch inside a block: the measurement is an
// operand that only the branch, 40 into the block, waits for. The block
// starts so that the branch begins when the readout is done.
// CHECK-LABEL: func.func @if_in_block
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 360 : i64}>
// CHECK: scf.if
// CHECK: pulse.drive {{.*}}<{duration_cycles = 100 : i64, start_cycles = 400 : i64}>
func.func @if_in_block() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %tr0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!r, !t)
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c400 = arith.constant 400 : i64
  %c100 = arith.constant 100 : i64
  %c40 = arith.constant 40 : i64
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %m = pulse.readout %r0, %ro, %tr0, discriminated : !r, !w, !t -> !r, !t, !m
  %d1, %t1 = pulse.schedule_block asap (%m, %d0, %t0) : (!m, !d, !t) -> (!d, !t) {
  ^bb0(%bm: !m, %d: !d, %t: !t):
    %d2, %t2 = pulse.drive %d, %x40, %t : !d, !w, !t -> !d, !t
    %bit = pulse.is_set %bm : !m -> i1
    %d3, %t3 = scf.if %bit -> (!d, !t) {
      %d4, %t4 = pulse.drive %d2, %x100, %t2 : !d, !w, !t -> !d, !t
      scf.yield %d4, %t4 : !d, !t
    } else {
      scf.yield %d2, %t2 : !d, !t
    }
    pulse.yield %d3, %t3 : !d, !t
  }
  return
}

// Tones crossing the boundary. Tone b is busy with the cross-resonance pulse
// on line a until 300, so the block that plays on it starts then. Tone c
// only changes phase in the block, so it passes through: the drive on line c
// after the block does not wait for the block.
// CHECK-LABEL: func.func @tones_across_blocks
// CHECK: pulse.drive {{.*}}<{duration_cycles = 300 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 300 : i64}>
// CHECK: pulse.shift_phase
// ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 0 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 300 : i64}
func.func @tones_across_blocks() {
  %qa = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %qb = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %qc = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %a0, %ta0 = pulse.get_drive_line %qa : (!pulse.qudit) -> (!d, !t)
  %b0, %tb0 = pulse.get_drive_line %qb : (!pulse.qudit) -> (!d, !t)
  %c0, %tc0 = pulse.get_drive_line %qc : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %phi = arith.constant 0.5 : f64
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %cr = pulse.square %c300, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %a1, %tb1 = pulse.drive %a0, %cr, %tb0 : !d, !w, !t -> !d, !t
  %b1, %tb2, %tc1 = pulse.schedule_block asap (%b0, %tb1, %tc0) : (!d, !t, !t) -> (!d, !t, !t) {
  ^bb0(%b: !d, %tb: !t, %tc: !t):
    %b2, %tb3 = pulse.drive %b, %x40, %tb : !d, !w, !t -> !d, !t
    %tc2 = pulse.shift_phase %tc, %phi : !t, f64 -> !t
    pulse.yield %b2, %tb3, %tc2 : !d, !t, !t
  }
  %c1, %tc3 = pulse.drive %c0, %x40, %tc1 : !d, !w, !t -> !d, !t
  return
}

// A sync inside a block aligns the lines it names; the block waits for both.
// Line a is busy until 200, so the block, whose readout follows the sync,
// starts at 200 even though line r is free.
// CHECK-LABEL: func.func @sync_in_block
// CHECK: pulse.drive {{.*}}<{duration_cycles = 200 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 200 : i64}>
func.func @sync_in_block() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!d, !t)
  %r0, %tr0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!r, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c200 = arith.constant 200 : i64
  %c400 = arith.constant 400 : i64
  %x200 = pulse.square %c200, %amp, %zero : i64, f64, f64 -> !w
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %d1, %t1 = pulse.drive %d0, %x200, %t0 : !d, !w, !t -> !d, !t
  %d2, %r2, %tr2, %m = pulse.schedule_block alap (%d1, %r0, %tr0) : (!d, !r, !t) -> (!d, !r, !t, !m) {
  ^bb0(%d: !d, %r: !r, %tr: !t):
    %ds, %rs = pulse.sync %d, %r : !d, !r -> !d, !r
    %r3, %tr3, %mm = pulse.readout %rs, %ro, %tr, discriminated : !r, !w, !t -> !r, !t, !m
    pulse.yield %ds, %r3, %tr3, %mm : !d, !r, !t, !m
  }
  return
}

// A block that reads a bit and changes a tone's phase on it hands the
// tone out once the readout has ended, as the flat schedule does: the drive
// after the block plays the chosen phase. Before, the tone was taken to pass
// through the block unchanged and the drive started at 0 (asap) or 360
// (alap), before the readout ended at 400.
// CHECK-LABEL: func.func @bit_read_in_block
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 400 : i64}>
// FLAT-ASAP-LABEL: func.func @bit_read_in_block
// FLAT-ASAP: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// FLAT-ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 400 : i64}>
// FLAT-ALAP-LABEL: func.func @bit_read_in_block
// FLAT-ALAP: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// FLAT-ALAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 400 : i64}>
func.func @bit_read_in_block() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %tr0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!r, !t)
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %phi = arith.constant 3.14 : f64
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %t1 = pulse.schedule_block asap (%r0, %tr0, %t0) : (!r, !t, !t) -> (!r, !t, !t) {
  ^bb0(%r: !r, %tr: !t, %t: !t):
    %r2, %tr2, %m = pulse.readout %r, %ro, %tr, discriminated : !r, !w, !t -> !r, !t, !m
    %bit = pulse.is_set %m : !m -> i1
    %t2 = scf.if %bit -> (!t) {
      %t3 = pulse.shift_phase %t, %phi : !t, f64 -> !t
      scf.yield %t3 : !t
    } else {
      scf.yield %t : !t
    }
    pulse.yield %r2, %tr2, %t2 : !r, !t, !t
  }
  %d1, %t4 = pulse.drive %d0, %x40, %t1 : !d, !w, !t -> !d, !t
  return
}

// A measured bit passed into a block that changes a phase on it: the
// branch's result counts as a use of the bit, so the block waits for the
// readout. Before, the drive started at 0 (asap) or 360 (alap).
// CHECK-LABEL: func.func @bit_passed_in
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 400 : i64}>
// FLAT-ASAP-LABEL: func.func @bit_passed_in
// FLAT-ASAP: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// FLAT-ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 400 : i64}>
// FLAT-ALAP-LABEL: func.func @bit_passed_in
// FLAT-ALAP: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// FLAT-ALAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 400 : i64}>
func.func @bit_passed_in() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %tr0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %d0, %t0 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %phi = arith.constant 3.14 : f64
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %m = pulse.readout %r0, %ro, %tr0, discriminated : !r, !w, !t -> !r, !t, !m
  %d1, %t1 = pulse.schedule_block asap (%m, %d0, %t0) : (!m, !d, !t) -> (!d, !t) {
  ^bb0(%bm: !m, %d: !d, %t: !t):
    %bit = pulse.is_set %bm : !m -> i1
    %t2 = scf.if %bit -> (!t) {
      %t3 = pulse.shift_phase %t, %phi : !t, f64 -> !t
      scf.yield %t3 : !t
    } else {
      scf.yield %t : !t
    }
    %d2, %t4 = pulse.drive %d, %x40, %t2 : !d, !w, !t -> !d, !t
    pulse.yield %d2, %t4 : !d, !t
  }
  return
}

// The same bit is branched on again at local 504. The first use is still
// the drive at local 0 on the changed tone: the block, placed whole, waits
// for the readout. Before, 504 was taken as the first use and the drive
// started at 0, before the readout ended at 400 (asap) or 504 (alap).
// CHECK-LABEL: func.func @bit_passed_in_used_later
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 400 : i64}>
// CHECK: pulse.wait {{.*}}<{duration_cycles = 504 : i64, start_cycles = 400 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 904 : i64}>
// FLAT-ASAP-LABEL: func.func @bit_passed_in_used_later
// FLAT-ASAP: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// FLAT-ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 400 : i64}>
// FLAT-ASAP: pulse.wait {{.*}}<{duration_cycles = 504 : i64, start_cycles = 0 : i64}>
// FLAT-ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 504 : i64}>
// FLAT-ALAP-LABEL: func.func @bit_passed_in_used_later
// FLAT-ALAP: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 104 : i64}>
// FLAT-ALAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 504 : i64}>
// FLAT-ALAP: pulse.wait {{.*}}<{duration_cycles = 504 : i64, start_cycles = 0 : i64}>
// FLAT-ALAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 504 : i64}>
func.func @bit_passed_in_used_later() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %tr0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %d0, %t0 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %e0, %te0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %phi = arith.constant 3.14 : f64
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %c504 = arith.constant 504 : i64
  %w504 = pulse.duration_from_int %c504 : (i64) -> !pulse.duration
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %m = pulse.readout %r0, %ro, %tr0, discriminated : !r, !w, !t -> !r, !t, !m
  %d1, %t1, %e1, %te1 = pulse.schedule_block asap (%m, %d0, %t0, %e0, %te0) : (!m, !d, !t, !d, !t) -> (!d, !t, !d, !t) {
  ^bb0(%bm: !m, %d: !d, %t: !t, %e: !d, %te: !t):
    %bit = pulse.is_set %bm : !m -> i1
    %t2 = scf.if %bit -> (!t) {
      %t3 = pulse.shift_phase %t, %phi : !t, f64 -> !t
      scf.yield %t3 : !t
    } else {
      scf.yield %t : !t
    }
    %d2, %t4 = pulse.drive %d, %x40, %t2 : !d, !w, !t -> !d, !t
    %e2 = pulse.wait %e, %w504 : (!d, !pulse.duration) -> !d
    %e3, %te3 = scf.if %bit -> (!d, !t) {
      %e4, %te4 = pulse.drive %e2, %x40, %te : !d, !w, !t -> !d, !t
      scf.yield %e4, %te4 : !d, !t
    } else {
      scf.yield %e2, %te : !d, !t
    }
    pulse.yield %d2, %t4, %e3, %te3 : !d, !t, !d, !t
  }
  return
}

// A loop in a block branches on a measurement carried two trips deep
// (a = b; b = readout): the last trip branches on the readout of trip 5,
// which ends at 2480. The block hands the tone out after it, as the flat
// schedule does. Before, the shape took the last trip's condition from the
// first trips and the drive started at 0 (asap), before that readout ended.
// ASAP-LABEL: func.func @carried_two_deep
// ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 2480 : i64}>
// ALAP-LABEL: func.func @carried_two_deep
// ALAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 3240 : i64}>
// FLAT-ASAP-LABEL: func.func @carried_two_deep
// FLAT-ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 2480 : i64}>
// FLAT-ALAP-LABEL: func.func @carried_two_deep
// FLAT-ALAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 3240 : i64}>
func.func @carried_two_deep() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %tr0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %d0, %t0 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %phi = arith.constant 3.14 : f64
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %st = arith.constant 1 : index
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %t1 = pulse.schedule_block asap (%r0, %tr0, %t0) : (!r, !t, !t) -> (!r, !t, !t) {
  ^bb0(%r: !r, %tr: !t, %t: !t):
    %ra, %tra, %a = pulse.readout %r, %x40, %tr, discriminated : !r, !w, !t -> !r, !t, !m
    %rb, %trb, %b = pulse.readout %ra, %x40, %tra, discriminated : !r, !w, !t -> !r, !t, !m
    %rl, %trl, %tl, %al, %bl = scf.for %i = %lb to %ub step %st iter_args(%ri = %rb, %tri = %trb, %ti = %t, %ai = %a, %bi = %b) -> (!r, !t, !t, !m, !m) {
      %bit = pulse.is_set %ai : !m -> i1
      %t2 = scf.if %bit -> (!t) {
        %t3 = pulse.shift_phase %ti, %phi : !t, f64 -> !t
        scf.yield %t3 : !t
      } else {
        scf.yield %ti : !t
      }
      %rn, %trn, %mn = pulse.readout %ri, %ro, %tri, discriminated : !r, !w, !t -> !r, !t, !m
      scf.yield %rn, %trn, %t2, %bi, %mn : !r, !t, !t, !m, !m
    }
    pulse.yield %rl, %trl, %tl : !r, !t, !t
  }
  %d1, %t4 = pulse.drive %d0, %x40, %t1 : !d, !w, !t -> !d, !t
  return
}

// Three trips deep: the last trip branches on the readout of trip 4, which
// ends at 2120. Before, the drive started at 0 (asap).
// ASAP-LABEL: func.func @carried_three_deep
// ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 2120 : i64}>
// ALAP-LABEL: func.func @carried_three_deep
// ALAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 3280 : i64}>
// FLAT-ASAP-LABEL: func.func @carried_three_deep
// FLAT-ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 2120 : i64}>
// FLAT-ALAP-LABEL: func.func @carried_three_deep
// FLAT-ALAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 3280 : i64}>
func.func @carried_three_deep() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %tr0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %d0, %t0 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %phi = arith.constant 3.14 : f64
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %st = arith.constant 1 : index
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %t1 = pulse.schedule_block asap (%r0, %tr0, %t0) : (!r, !t, !t) -> (!r, !t, !t) {
  ^bb0(%r: !r, %tr: !t, %t: !t):
    %ra, %tra, %a = pulse.readout %r, %x40, %tr, discriminated : !r, !w, !t -> !r, !t, !m
    %rb, %trb, %b = pulse.readout %ra, %x40, %tra, discriminated : !r, !w, !t -> !r, !t, !m
    %rc, %trc, %c = pulse.readout %rb, %x40, %trb, discriminated : !r, !w, !t -> !r, !t, !m
    %rl, %trl, %tl, %al, %bl, %cl = scf.for %i = %lb to %ub step %st iter_args(%ri = %rc, %tri = %trc, %ti = %t, %ai = %a, %bi = %b, %ci = %c) -> (!r, !t, !t, !m, !m, !m) {
      %bit = pulse.is_set %ai : !m -> i1
      %t2 = scf.if %bit -> (!t) {
        %t3 = pulse.shift_phase %ti, %phi : !t, f64 -> !t
        scf.yield %t3 : !t
      } else {
        scf.yield %ti : !t
      }
      %rn, %trn, %mn = pulse.readout %ri, %ro, %tri, discriminated : !r, !w, !t -> !r, !t, !m
      scf.yield %rn, %trn, %t2, %bi, %ci, %mn : !r, !t, !t, !m, !m, !m
    }
    pulse.yield %rl, %trl, %tl : !r, !t, !t
  }
  %d1, %t4 = pulse.drive %d0, %x40, %t1 : !d, !w, !t -> !d, !t
  return
}

// No readout: a loop in a block plays on the first of three tones and
// rotates them, 7 trips. The block hands out tone b (last played in trip 4,
// ending at 200), tone c (trip 5, 240) and tone a (trip 6, 280). Before,
// the drive on tone b started at 0 and the one on tone c at 40, while the
// block still played them.
// ASAP-LABEL: func.func @tones_rotated_in_block
// ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 200 : i64}>
// ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 240 : i64}>
// ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
func.func @tones_rotated_in_block() {
  %qa = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %qb = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %qc = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %a0, %ta0 = pulse.get_drive_line %qa : (!pulse.qudit) -> (!d, !t)
  %b0, %tb0 = pulse.get_drive_line %qb : (!pulse.qudit) -> (!d, !t)
  %c0, %tc0 = pulse.get_drive_line %qc : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c40 = arith.constant 40 : i64
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %lb = arith.constant 0 : index
  %ub = arith.constant 7 : index
  %st = arith.constant 1 : index
  %a5, %x5, %y5, %z5 = pulse.schedule_block asap (%a0, %ta0, %tb0, %tc0) : (!d, !t, !t, !t) -> (!d, !t, !t, !t) {
  ^bb0(%a: !d, %x: !t, %y: !t, %z: !t):
    %l, %p, %q, %r = scf.for %i = %lb to %ub step %st iter_args(%l1 = %a, %p1 = %x, %q1 = %y, %r1 = %z) -> (!d, !t, !t, !t) {
      %l2, %p2 = pulse.drive %l1, %x40, %p1 : !d, !w, !t -> !d, !t
      scf.yield %l2, %q1, %r1, %p2 : !d, !t, !t, !t
    }
    pulse.yield %l, %p, %q, %r : !d, !t, !t, !t
  }
  %b1, %t1 = pulse.drive %b0, %x40, %x5 : !d, !w, !t -> !d, !t
  %c1, %t2 = pulse.drive %c0, %x40, %y5 : !d, !w, !t -> !d, !t
  %b2, %t3 = pulse.drive %b1, %x40, %z5 : !d, !w, !t -> !d, !t
  return
}

// No readout: a loop in a block passes two tones to each other, 9 trips,
// so the second result is the first operand, busy until 200. The shape
// skips trips; when they are not a whole number of swaps it takes the
// later of the two tones, so the block also waits for both (alap places it
// at 200, its drive at 520, later than the flat 320). Before, the drive on
// the second result started at 0 (asap).
// ASAP-LABEL: func.func @tones_swapped_in_block
// ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 200 : i64}>
// ALAP-LABEL: func.func @tones_swapped_in_block
// ALAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 520 : i64}>
// FLAT-ASAP-LABEL: func.func @tones_swapped_in_block
// FLAT-ASAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 200 : i64}>
// FLAT-ALAP-LABEL: func.func @tones_swapped_in_block
// FLAT-ALAP: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 320 : i64}>
func.func @tones_swapped_in_block() {
  %qa = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %qb = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %qc = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %a0, %ta0 = pulse.get_drive_line %qa : (!pulse.qudit) -> (!d, !t)
  %b0, %tb0 = pulse.get_drive_line %qb : (!pulse.qudit) -> (!d, !t)
  %c0, %tc0 = pulse.get_drive_line %qc : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c40 = arith.constant 40 : i64
  %c200 = arith.constant 200 : i64
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %x200 = pulse.square %c200, %amp, %zero : i64, f64, f64 -> !w
  %lb = arith.constant 0 : index
  %ub = arith.constant 9 : index
  %st = arith.constant 1 : index
  %a1, %ta1 = pulse.drive %a0, %x200, %ta0 : !d, !w, !t -> !d, !t
  %c5, %tc5, %x5, %y5 = pulse.schedule_block asap (%c0, %tc0, %ta1, %tb0) : (!d, !t, !t, !t) -> (!d, !t, !t, !t) {
  ^bb0(%c: !d, %tc: !t, %x: !t, %y: !t):
    %l, %w, %p, %q = scf.for %i = %lb to %ub step %st iter_args(%l1 = %c, %w1 = %tc, %p1 = %x, %q1 = %y) -> (!d, !t, !t, !t) {
      %l2, %w2 = pulse.drive %l1, %x40, %w1 : !d, !w, !t -> !d, !t
      scf.yield %l2, %w2, %q1, %p1 : !d, !t, !t, !t
    }
    pulse.yield %l, %w, %p, %q : !d, !t, !t, !t
  }
  %b1, %t1 = pulse.drive %b0, %x40, %y5 : !d, !w, !t -> !d, !t
  return
}
