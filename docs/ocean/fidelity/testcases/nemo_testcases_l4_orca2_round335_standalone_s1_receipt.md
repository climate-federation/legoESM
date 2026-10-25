# ORCA2 round 335 — standalone design and S1 model preparation

Date: 2026-10-25.  Base `072e4aa1b1c8`; preregistration/design commit
`9194a71e6`; model commit `12c0e9ac6`.  Disposition: **LANDED under Decisions
122 and 139, with S1 still DEBT rather than certified**.  No configuration
changed and no NEMO executable ran.

## Result

The standalone route is now specified as four exact-input rungs: S1 bulk
ocean, S2 remaining surface composition, S3 observed-cover ice-if, and S4 a
composed card against a purpose-built NEMO reference without SI3.  The S2/S3
delivery order is not confused with execution order: NEMO calls bulk, ice,
runoff, restoring/budget, halo assembly, then T-to-U/V stress rotation in
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90:428-570`.

S1 now uses the existing shared bulk-flux module rather than another copy.
The literal path covers humidity/temperature preprocessing and the two ocean
calls in
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk.f90:640-678`,
the executing `blk_oce_1` statements at `:791-949` and `:976`, and all of
`blk_oce_2` at `:1019-1046`.  The exact deck selects the NCAR call at
`:835-838`; no dead algorithm arm is claimed.

The NCAR coefficient transcription follows the initialization, five-iteration
loop, neutral-wind exp/log round-trip, coefficients, and literal arithmetic
stability blend in
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk_algo_ncar.f90:124-238`,
with its coefficient and stability helpers at `:241-381`.  The shared physical
helpers transcribe the active pressure/Exner/density statements in
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbc_phy.f90:250-409`,
latent heat and moist heat capacity at `:443-506`, Obukhov inverse at
`:548-581`, Goff saturation and humidity at `:645-766`, bulk assembly at
`:898-940`, net longwave at `:1020-1042`, roughness reconstruction at
`:1062-1080`, and neutral wind at `:1198-1213`.

Assignments pass through the existing no-FMA materialization helper.  The
precision-policy module now supplies scalar glibc `pow` and `atan`, alongside
its prior exp/log/log10 arm, with custom JVPs.  Disassembly of the producing
`sbcblk_algo_ncar.o` and `sbc_phy.o` confirms scalar `MAXSD`/`MINSD`; the loaded
first Fortran value is the destination and the second source operand wins an
equal `+0/-0` tie.  A dedicated test pins both operand orders.  The producing
flags are `-O3 -funroll-all-loops -fno-tree-vectorize`, with no fast-math or
FMA option.

## Registered offline score

The committed preregistration's parent replay reproduced all seven counts:
`theta_air 117`, `ssq 3964`, and `utau/vtau/sensible/latent/evap 8794` each,
all out of 8,794 wet cells.  That is 48,051 unequal words across the seven
registered fields.

The literal current replay improves every common field: `theta_air 1`, `ssq
474`, `utau 3641`, `vtau 3669`, `sensible 3075`, `latent 5316`, and `evap
5476`, totalling 21,652 unequal words.  No exact common field was lost, the
first field moved sharply toward the bar, and no earlier row worsened.  The
largest common-field residual is `2.0463630789890885e-12 W m-2` in latent
heat, compared with `11.997116616015276 W m-2` before.

The expanded 18-defined-field census finds seven exact fields (`q_air`,
`precip`, `sst`, `ssu`, `ssv`, `tsk`, `qsr`) and eleven DEBT fields.  Additional
unequal counts are `wndm 774`, `taum 3896`, `qns 3928`, and `emp 4519`; the
18-field total is 34,769 unequal words.  Therefore the exact S1 bar remains
**11/18 fields over bar before/after classification: 7/7 registered fields
were over bar before, 7/7 common fields remain over bar after**.  This round
lands the source-derived first rung preparation under the improvement rule;
it does not call S1 certified or standalone-ready.

Every output group has a one-ULP binding plant through the actual cell scorer.
The measurement is CPU, fp64, production JIT, scalar glibc transcendental
policy, on the immutable round-5 O1 record.  The record mapping boundary stays
exact and is not credited to this change.

## Caller controls, tests, and review

A real parent-source/current-source control used identical three-cell inputs.
All five legacy `ly09_2coeff` outputs were bit-identical (`0,0,0,0,0` unequal).
The production NCAR absolute-wind output changed `(2,2,3,3,3)` words and its
current-feedback arm changed `(3,3,3,3,3)`, as required by the literal repair.
The only production callers are the OMIP2 applicator/coupler; grep finds no SI3
surface-module caller of this shared function.  The applicator, phase-E, and
relative-wind caller suite passes 67/67.

Focused source tests pass 22/22: independent source transcription, inexact
binary64 pins, code-path plants, signed-zero tie order, eager equals JIT,
finite-difference gradient, explicit physical-input signature, legacy
dispatch, and supersaturated/unphysical source behavior.  Shared saturation
plus caller tests pass 67/67.  The O1 gate's mutation plant fires on every one
of the 18 defined outputs.

independent review: operator (Claude)

## Choices and open work

- ASKED: design S1--S4, prepare literal S1 in shared code, score the existing
  phase-2h/round-5 record, and preregister but not build the standalone S1
  acquisition.  All are complete.
- UNASKED: configure/build/run NEMO, change the comparison/training card,
  relax a bar, create an acquisition tool, remove replay modes, or select an
  ice-cover file.  None occurred.
- OPEN: locate the remaining source-rounding owners in the 11 S1 debt fields,
  starting with `theta_air`'s one word; do not advance S1's exact bar on this
  receipt.
- DECISION_NEEDED: choose ice-cover provenance: (1) monthly native-grid
  climatology from a pinned NEMO ORCA2_ICE run (pick, controlled fidelity but
  reference-derived), (2) a named/versioned external observed climatology, or
  (3) a user-supplied existing file admitted with schema/provenance/digest.
  No option has been selected.
- The future acquisition remains preregistered only; after that decision it
  records S1 entry, post-`blk_oce_1`, and post-`blk_oce_2` on kt 1--10,
  ranks 0--1, as independent byte-identical twins.
