# Preregistration — NEMO testcase L2 GYRE round 153

Date: 2026-09-22

Incoming lane tip: `2374207bfd8ddbf5708207c3839daaff603cef91`.
Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round153/`.  This document is
frozen before constructing or parsing the developed-state FCT record.

## Fixed question and source card

Round 152 measured the first active process boundary at day 180: stage-3 FCT
advection differs from NEMO by `1.0974591404090626e-8 K` RMS in its isolated
one-step temperature contribution.  Its admitted record begins only at the
completed `Krhs`, so no internal statement is yet attributable.

This round reuses the Round-111 self-describing writer layout on NEMO's own
developed step 1081.  The exact source card is the admitted Round-148 build and
run (`GYRE_OMIP_L2_P3_SM_R148LDF`, binary SHA-256
`9d758bf51d85a27b7692858697ddbc5fbd9b19d724f957c983a605085c89e250`).
Its namelist ends at step 1081 and retains `nn_stock=180`, `nn_write=2160`,
and seed zero.  A new target name is mandatory.  The patch is additive against
the canonical `traadv_fct.F90`; the existing source-card `EXP00`, `MY_SRC`,
and preprocessor file are copied file by file.

The compiled program calls the two-step upstream routine before constructing
the high-order-minus-upstream faces at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:164-199`.
That routine writes the first upwind faces, first divergence, midpoint tracer,
averaged faces, second divergence, and direct upstream `Krhs` at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:495-610`.
The parent routine then exchanges the anti-diffusive faces, invokes `nonosc`,
forms the final divergence, divides it by the live Kmm thickness, and adds it
to `Krhs` at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:306-330`.
Inside `nonosc`, NEMO forms the beta operands at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:849-883` and applies
the sign-selected U/V/W coefficients at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:886-936`.

## Frozen record

The writer activates only for `kt=1081`, the stage-3 slot tuple recorded by
Round 123 (`Kbb=1`, `Kmm=2`, `Krhs=3`, `Kaa=3`), `cdtype='TRA'`, and the
ordered temperature/salinity pair.  It refuses tiling and non-fp64 execution.
It writes exactly one stream file with magic `NEMO_L2_R153FCT`, a versioned
header, and 61 ordered self-describing fields.  Every field carries name, rank,
extent, origin, and binary64 payload so byte size is derived by the reader,
not duplicated in shell arithmetic.

The 11 common fields are `p2dt`, all three stage-3 transports, `e3t_3d`, the
Kbb/Kmm/Kaa T-point free-surface ratios, `tmask`, `wmask`, and `r1_e1e2t`.
For each of T and S, the 25 compiled-order fields are:

1. Kbb tracer, Kmm tracer, and entry `Krhs`;
2. first U/V/W upwind faces, first divergence, and midpoint tracer;
3. averaged U/V/W faces, upstream final divergence, and `Krhs` after the
   direct upstream write;
4. exchanged anti-diffusive U/V/W faces before `nonosc`;
5. the actual sign-selected U/V/W limiter coefficients and the limited U/V/W
   faces after `nonosc`;
6. final divergence, a separately observed value of the live Kmm divisor
   expression, and `Krhs` after the final direct write.

The coefficient arrays are initialized to one and filled immediately after
each compiled `zcoef` assignment.  Limiter activity is defined mechanically as
the exact coefficient word differing from binary64 one; no flux-ratio division
is used to infer it.  All capture calls occur after the production statement
whose value they observe.  No recorded value is read back into NEMO state.

## Admission and falsifiers

The new run is admitted only if all of these hold:

1. the new record has the exact header, 61-field order, extents, origins,
   finite payloads, step/stage tuple, `p2dt=14400 s`, and exact EOF;
2. at least one directly recorded limiter coefficient differs from one, and
   the report gives the active coefficient count separately for T and S;
3. every `oracle*.bin`, restart, and mesh file inherited from the admitted
   Round-148 source run is present and byte-identical; the file registry is
   source-derived rather than hand-written;
4. the record digest, producing clean commit, source-card hash, and binary hash
   agree; and
5. stamp, truncation, missing-field, all-coefficients-one, and inherited-byte
   plants each print `STATUS PLANT-FIRED` and exit nonzero.

The passive-instrument prediction is **CONFIRMED** only if every inherited
file is bit-identical.  One changed inherited byte **REFUTES** passivity; the
record is rejected and a corrected writer requires a new target name.  In
particular, an optimizer/materialisation explanation is not a waiver.

The activity prediction is that both tracers have at least one coefficient
different from one, consistent with Round 152's model-side 502-cell active
limiter map.  Zero active NEMO coefficients **REFUTES** that prediction and
stops the internal comparison rather than relabeling the model map as NEMO
activity.

## Round stopping rule and next measurement

No physics, configuration, default, carried state, restart schema, stabilizer,
year harness, reconciliation gate, freshwater pair, or #1484 guard changes in
this acquisition round.  In-sandbox `mpirun` is prohibited by the campaign's
recorded PMIx socket refusal, so a syntax-proven operator-run `run.sh` is the
expected deliverable and status is `STOPPED_FOR_RECORD`.

After admission, Round 154 extends the existing Round-111/112 production-step
walk, not a second FCT closure.  Given NEMO's exact day-180 entry it compares
the 61 rows in compiled order under production JIT, production eager, and
isolated JIT, with a production-path ULP plant.  The first production-JIT
non-bit row owns the next statement.  Only a source-exact candidate may then
enter the full Decision-43/45 ladder, month, year, shared-card, and DINO gates.

The inherited campaign headlines remain unmeasured this round: kt2 T/S stay
AT-BAR; kt2 U/V remain the first-over-bar rows; kt3 T/S are
`8.659371033559182e-7` / `7.027288972949464e-8`; day-30/day-240/day-360 T3D
RMS remain `6.888193513796918e-5`, `1.644671864406711e-2`, and
`1.1223450861560211e-2 K`.  ORCA2 is `UNMEASURED-WITH-SPEC`: repeat the same
developed-entry FCT record and production walk on its ocean-only card before
transferring any statement verdict.
