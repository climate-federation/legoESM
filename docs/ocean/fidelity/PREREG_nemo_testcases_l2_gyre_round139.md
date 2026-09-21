# Preregistration — NEMO testcase L2 GYRE round 139

Date: 2026-09-21

Incoming lane tip: `b096d2f2e895f67b9375c140ec686365cf415f7c`

This document is frozen before adding or reading a Round-139 operand record and
before running a new developed-state production comparison. Evidence will live
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round139/`.

Round 138 drove one production-JIT step from NEMO's admitted completed-step-
1080 state. Every registered first-substep boundary was BIT through pressure,
Coriolis, and drag. The first non-bit boundary was the frozen slow-forcing
result: all 580 wet U faces differed by at most
`4.2854247978022983e-13`, and all 570 wet V faces differed by at most
`4.4333086294645174e-13`. The completed external SSH then differed in all 600
wet columns. The downstream QCO multiplication was BIT under exact model and
NEMO inputs and is not reopened here.

## P0 — compiled boundary and minimum passive record

The executing Round-137 compiled program builds the two-dimensional momentum
RHS in `GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stp2d.f90:130-265`.
It copies `Ue_rhs` and `Ve_rhs` into the frozen external forcing, initializes
the barotropic Coriolis coefficients, evaluates `dyn_cor_2D`, and subtracts
that result with the native face masks at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:289-325`.
The called four-point Coriolis statements are
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:1336-1358`.

No admitted developed record separates the two operands of the line-323/324
subtraction. Extend the existing Round-137 source card under a new target name
with one WRITE-only stream at `kt=1081`. Immediately around that one existing
call and loop, record the native owned extents of:

1. incoming `Ue_rhs` and `Ve_rhs`;
2. returned `zu_trd` and `zv_trd`; and
3. final `zu_frc` and `zv_frc`.

The mask operands remain in the same run's admitted Round-137 external stream;
they are not duplicated. The new record has a 16-byte magic, eleven int32
header values, and exactly six 32-by-22 fp64 arrays: 33,852 bytes. Its parser
must reject the wrong magic, version, step, time level, grid, scalar kind,
owned extent, field count, truncation, trailing bytes, non-finite values, and
any replay that does not reproduce final forcing bit for bit. Header,
truncation, replay-ULP, and commit-stamp plants must exit nonzero and print a
named `STATUS PLANT-FIRED` marker.

The new run must reproduce the admitted Round-137 step-1080 restart,
step-1081 process stream, external stream, and QCO stream byte for byte. Any
failed identity makes the writer non-passive and stops the round. The
passive-admission plant changes one byte in one inherited control and must be
refused. Every shell failure prints `REFUSE:` before exit. The source card is
syntax-proved before `makenemo`; no canonical NEMO source is modified.

## P1 — production-JIT operand split

Extend the existing Round-83 slow-forcing walk; do not add another model
stepper. Reuse Round 138's complete step-1080 restart bridge and certified
step-1081 forcing. Run `LatLonCGridOceanModel.step -> self._step_jitted` with
the existing live-stage operand hook and compare, in compiled order:

1. incoming U and V forcing;
2. initializing Coriolis U and V;
3. native U and V masks; and
4. final U and V forcing.

The traced step and an independently compiled ordinary step must return
bit-identical pytrees. Production JIT is claim-bearing; production eager and
an isolated replay, if reported, are controls only. Signed zero is non-bit.
A registry plant removing any one operand must fail, and a one-ULP change to a
consumed wet incoming-U value must move the registered final-U row and exit
nonzero.

Frozen scientific prediction: the first non-bit operand is incoming
`Ue_rhs`, followed by incoming `Ve_rhs`; both initializing `dyn_cor_2D`
outputs and both masks are BIT. Substituting only NEMO's incoming pair into
the production closure makes the final frozen forcing BIT on all 580 U and
570 V wet faces. This is **CONFIRMED** only if the ordinary trace reproduces
Round 138's final row, the record replay is BIT, all earlier registered
operands are BIT, and the directed arm is exact. It is **REFUTED** by a
non-bit Coriolis or mask operand before incoming forcing, an exact incoming
pair, a non-exact final replay, or a non-exact NEMO-incoming directed arm.
The record decides the operand; the final subtraction alone does not.

If incoming forcing is first, the receipt names that boundary and hands its
existing compiled-order RHS decomposition forward. If Coriolis is first, the
receipt names the first non-bit Coriolis input or statement reached by the
existing registry. No production change or landing is expected this round.

## P2 — campaign gates and stop condition

No Decision-43/45 candidate exists unless one recorded operand is exact and a
single shared statement is independently proven wrong. In the expected
diagnostic outcome, the ladder, month, year, DINO, LOCK_EXCHANGE, OVERFLOW,
tanks, and ORCA2 are not rerun because production is unchanged. ORCA2 remains
`UNMEASURED-WITH-SPEC`: its ocean-only card must independently register this
incoming/Coriolis/mask/final split on native U/V extents before any identity
claim.

A separate read-only Codex pass must try to refute source ancestry, insertion
order, passivity, byte layout, restart timing, production-JIT status, returned-
state identity, registry completeness, directed substitution, and plants. A
`DO NOT SHIP` verdict blocks the receipt. Every compiled-source citation is
mapped by the receipt citation gate, whose shifted-citation plant must exit
nonzero.

The current sandbox reports the NEMO `cfgs` directory read-only. The run is
still attempted only through the committed fail-closed script. If it cannot
create the new target, status is `STOPPED_FOR_RECORD`, the frozen predictions
remain `UNMEASURED`, and `ACQUISITION_NEEDED` names that script. There is no
configuration, carried-state, threshold, scheme, stabilizer, or source choice;
`DECISION_NEEDED` is `NONE`.
