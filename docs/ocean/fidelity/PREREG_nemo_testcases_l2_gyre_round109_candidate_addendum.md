# Round 109 frozen candidate addendum: vector stage-one source handoff

Date: 2026-09-18. Frozen at clean instrument tip
`50522d1164b502155599016cc187c9fe61939634`, after the registered boundary
measurements and before any numerical edit.

This addendum does not change the predictions or falsifiers in
`PREREG_nemo_testcases_l2_gyre_round109.md`. It records that their P1--P3
preconditions closed and fixes the exact P4 implementation before measuring
it.

## Measurements that admit the candidate

Given NEMO's recorded kt=1 stage entry, the complete production JIT and the
complete production closure with JIT disabled both report:

- full pre-external accumulator U/V: BIT, 0 unequal cells;
- projected accumulator U/V: 17,400 / 17,100 unequal cells, maximum
  `2.0121494123449567e-8` on both faces;
- post-transport, post-W/ZAD, and final RK-input rows: the same counts and
  maximum as the projected accumulator;
- raw RK U/V: 17,400 / 17,100 unequal cells, maximum
  `9.658317179255792e-5`;
- corrected output: production-JIT 7,620 / 8,460 unequal and production-eager
  7,920 / 8,640 unequal, all with maximum `5.421010862427522e-20`.

The isolated JIT of the shared RK assignment and barotropic correction, fed
NEMO's recorded full RHS, before velocity, clock, masks, admitted
post-external target, and reference geometry, is BIT for raw and corrected U/V.
The initial isolated correction run that used the stage writer's pre-external
zero Kaa slot was a harness defect and is retained as
`handoff_corrected_output_isolated_jit.json`; the corrected admitted arm is
`handoff_corrected_output_isolated_jit_v2.json`.

Thus the first non-bit statement in the model program is the construction of
the projected accumulator from the full accumulator. Production eager and JIT
name the same statement. The later eager/JIT count difference belongs to the
barotropic-correction fusion and is not promoted ahead of that first boundary.

## Frozen numerical hunk

In the one shared `rk3_ws` stage-1 implementation:

1. when the already-resolved shared predicate
   `_vector_velocity_stage_update` is true, initialize the stage-1 U/V RHS
   from `du_dt`/`dv_dt`, the full pre-external accumulator;
2. in that vector arm, do not add the post-external transport-difference
   momentum recomputation, the post-external W/ZAD momentum reassociation, or
   `_stage_vertical_up3` to the stage-1 RHS;
3. preserve construction of `_g0`, W, and metric transports for tracers and
   diagnostics;
4. preserve the existing projected/reassociated program byte-for-byte for
   the non-vector arm, and do not alter stages 2/3, the external solve, tracer
   stepping, or the barotropic correction.

No selector or new configuration is introduced. The branch predicate is the
same vector predicate already consumed by the shared RK assignment.

This is the compiled program at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:141-176,202-213`,
`stprk3.f90:188-202`, and
`stprk3_stg.f90:326-347,363-374,661-675`: `stp_2D` completes and carries
`Krhs`; the vector stage-1 branch adds no momentum operator before consuming
that slot. The non-vector `dyn_adv` call remains where the compiled source
places it.

## Frozen candidate gates

The candidate must make all seven selected production-JIT and
production-eager boundaries BIT, and isolated-JIT raw/corrected U/V must stay
BIT. The nonzero one-ULP production plant will target candidate `raw_rk` U and
must change exactly one previously equal, finite, nonzero cell, print
`STATUS PLANT-FIRED`, and exit nonzero.

Only after those conditions close may the 954-row Rule-12 ladder and 30-day
score run. The immutable predictions remain: zero Rule-12 violations, no
headline movement, no AT-BAR row leaving the bar, first-over-bar no earlier,
and absolute day-30 T-RMS movement below `1e-10 K`. Failure restores the
candidate and records it HELD.
