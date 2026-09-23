# Preregistration — NEMO testcase L2 GYRE round 156

Date: 2026-09-22

Incoming lane tip: `43068109f14d181a74f35a3b0975322a3e00a160`.
Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round156/`.  This document is
frozen before any stage-2 measurement is run.

Round 155 established, with its calibration arm, that legoESM's stage-3
transport statements rebuild NEMO's transport bit for bit from NEMO's own
recorded operands, and that the magnitude owner of the 1.4247 transport
difference is the stage-2 velocity `uu(Kmm)` (95.21% by rms, 99.33% by
maximum), not the compiled-order first input `un_adv` (0.69% by rms).  Round
156 walks the compiled stage-2 program that writes that velocity.

## The compiled stage-2 program, in execution order

Cited from the record's own build,
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90`.  Stage 2 binds
`Kbb = N`, `Kmm = N+1/3`, `Kaa = N+1/2`, and stage 3 then reads that `Kaa`
field as its `Kmm`.

1. `:217-259` — stage 2 sets the external-mode fields at `Kaa`: under the
   active `np_HYB` arm `ssh(Kaa)` is the half-sum, `uu_b(Kaa) = ua_b` is the
   final external-mode velocity, and `r3u(Kaa)` is the half-sum of `r3u(Kbb)`
   and `r3ua`.
2. `:300-309` and `:313-315` — the stage transports, already walked and
   proven bit given NEMO's operands in round 155.
3. `:400` `CASE ( 2 , 3 )` — the stage RHS: `dyn_hpg` at `:475` (which
   OVERWRITES `Krhs`), `dyn_vor` at `:485`, `dyn_adv` at `:497`.
4. `:687-703` — the stage 1 and 2 time-stepping assignment, thickness
   weighted:
   `uu(Kaa) = ((1+r3u(Kbb))*uu(Kbb) + rDt*(1+r3u(Kmm))*uu(Krhs))
   / (1+r3u(Kaa)) * umask`.
5. `:753` and `:772` — the stage barotropic correction
   `zub = uu_b(Kaa) - SUM(e3u_3d*uu(Kaa))*r1_hu_0` and
   `uu(Kaa) = uu(Kaa) + zub*umask`.

Stage 2 carries NO lateral diffusion and NO implicit vertical part: `dyn_ldf`
at `:732` and `dyn_zdf` at `:745` are stage-3 only.  That is read off the
compiled `CASE ( 3 )` guard, not inferred.

## What the admitted records can and cannot measure at day 180

The admitted Round-154 record carries, at step 1081, only the operands stage 3
reads: `uu(Kmm)` (the stage-2 output), `uu_b(Kmm)` (= stage 2's `uu_b(Kaa)`),
`r3u(Kmm)` (= stage 2's `r3u(Kaa)`), `e3u_0`, `r1_hu_0`, `umask`, `e2u`,
`un_adv`, `zub` and `zFu`.  Every writer of a stage-2 INTERNAL quantity in
every existing build is gated on `kstp == nit000`, so NEMO's stage-2 RHS
snapshots, its `uu(N+1/3)` and its `r3u(N+1/3)` do NOT exist at day 180.  This
round therefore splits the measured `uu(Kmm)` difference between the two
halves of the stage-2 program that the record CAN separate, and requests the
missing record for the half that owns it.

## The measurement

One production step from NEMO's admitted day-180 entry through
`LatLonCGridOceanModel.step` under production JIT, reusing the existing
round-155 transport observer; no second implementation.  The split calls the
model's OWN shared transcription of the compiled correction statement,
`rk3_stage_barotropic_correction`
(`packages/ocean/legoesm/ocean/dynamics/barotropic_common.py`), driven with
NEMO's own recorded `uu_b(Kmm)`, `e3u_0`, `r1_hu_0` and `umask`.  Scoring is
over the 17,400 active U faces; rms ranks and the maximum is reported beside
it, as in round 155.

## Frozen predictions and falsifiers

P1. CALIBRATION.  Re-applying the shared correction statement to NEMO's own
    already-corrected `uu(Kmm)`, with NEMO's own `uu_b(Kmm)`, `e3u_0`,
    `r1_hu_0` and `umask`, is a no-op whose active maximum is below
    `1.0e-15` m/s, i.e. at least nine orders below the measured `uu(Kmm)`
    difference.  A larger residual means the decomposition's own floor is not
    negligible and every share below is withdrawn.
P2. ANCHOR.  The production `uu(Kmm)` row reproduces round 155 exactly:
    17,400 of 17,400 active faces unequal, active maximum
    `1.30926020461275e-6`.  A different value refutes the reproduction and the
    round stops on the instrument.
P3. OWNERSHIP.  Installing NEMO's own depth mean into legoESM's `uu(Kmm)` —
    the only half of the stage-2 program the record can substitute — removes
    LESS THAN 1% of the active rms of the `uu(Kmm)` difference.  The external
    barotropic half is therefore NOT the magnitude owner and the owner is the
    stage-2 internal 3-D program at `:400-497` and `:687-703`.
    FALSIFIER: if it removes 1% or more, P3 is REFUTED, the external step is a
    co-owner, and round 157 returns to the `un_adv`/`dynspg_ts` walk instead of
    requesting a stage-2 record.
P4. KNOWN ANSWER.  The measured depth-mean error, legoESM's own reference
    weighted column mean of `uu(Kmm)` minus NEMO's recorded `uu_b(Kmm)`,
    agrees with the independently recorded round-155 `uu_b(Kmm)` row (580 of
    580 active columns, maximum `6.134203041352482e-10`) to within the P1
    floor.  A disagreement means the decomposition is not measuring the
    barotropic half and the split is withdrawn.
P5. PLANT.  A one-ULP perturbation of NEMO's recorded `uu_b(Kmm)` operand
    changes the reprojected velocity row, prints `STATUS PLANT-FIRED` and
    exits nonzero.

## Landing rule

No statement inside stage 2 can be named a candidate this round unless the
record can calibrate it — round 155's discipline, kept: before naming a
statement, legoESM's transcription of it must rebuild NEMO's output bit for
bit from NEMO's own recorded operands.  The stage-2 RHS operands do not exist
at day 180, so no candidate can arise from them and nothing lands from them.
If a single source-exact production-JIT statement did become a candidate it
would still have to pass the full Decision 43/45 gate — month, day 240, day
360 not worse, kt2 T/S at the bar, first-over-bar not earlier, every moved row
registered, DINO measured if shared, the generic NEMO-GYRE card, the
LOCK_EXCHANGE and OVERFLOW tanks, and the six-file push gate — before landing.

No physics, configuration, default, carried state, restart schema, stabilizer,
year harness, reconciliation gate, freshwater pair or #1484 guard changes in
this round.  ORCA2 remains **UNMEASURED-WITH-SPEC**: repeat this stage-2
velocity split on the ocean-only ORCA2 card before transferring any verdict.
