# Preregistration: GYRE live stage operands and barotropic-memory substitution, round 51

Date: 2026-09-11. Frozen at legoESM `7674552beb52` before any round-51
model-path measurement or production edit. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round51/`. CPU/fp64/libm only.
The round-46 stage record and round-48 barotropic-memory record are the admitted
oracles. No NEMO executable, build, source, TKE implementation, or GYRE year
harness may be changed or run.

## 1. Live handed-operand ladder

The compiled program calls stage 1 with `Kbb=Kmm=N`, stage 2 with
`Kbb=N,Kmm=N+1/3`, and stage 3 with `Kbb=N,Kmm=N+1/2`
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:200-215`). Stage 1
calls HPG, LDF, VOR, WZV, KEG, and ZAD in that order
(`BLD/ppsrc/nemo/stp2d.f90:141-175`). Stages 2/3 form Kmm transports and ww
before HPG, VOR, and momentum advection (`BLD/ppsrc/nemo/stprk3_stg.f90:
276-345,431-480`); stage 3 then calls LDF and ZDF (`:690-725`).

A private WRITE-only trace will return operands already materialized inside the
one production-jitted WS-RK3 stage program, after the ordinary step completes.
It will not reconstruct them in the scorer. For every stage/operator the gate
will bit-score the velocity level, T/S/ssh level, live/reference thickness,
r3 fields, coefficient, ww, and metric transports against the matching
round-46 stage-entry arrays. Each row reports cells unequal and max absolute
difference. Static metrics/masks are admitted by round 46 and need not be
duplicated. Unknown trace layouts, missing fields, wrong dtype, dirty producer,
and wrong commit fail closed. A one-bit plant in the first unequal field must
exit nonzero and change that row.

**Frozen prediction.** The first unequal handed operand is kt=2 stage-1 HPG
`ssh(Kmm)` (and only afterward its r3/thickness descendants), because stage 1
uses the step entry while round 50 found 597--600 numeric SSH-history/current
differences at the kt=1-to-2 boundary. CONFIRM requires Kbb/Kmm U, V, T, and S
to be bit-exact before a nonzero SSH row. REFUTE if any earlier field differs
or if stage-1 SSH is exact. If confirmed, the owning boundary is NEMO's raw
history rotation/filter versus legoESM's deviation reconstruction, and no
stage-program or carried-state fix lands pending decision 33.

## 2. Barotropic-memory decision packet

NEMO extrapolates raw `now/b/bb` arrays using AB3-AM4 coefficients
(`dynspg_ts.f90:456-489`), then rotates them after every substep
(`:749-761`). legoESM instead reconstructs raw histories from deviations
against the next window's current state (`barotropic_latlon_cgrid.py:
2053-2084`) and stores new deviations at the window end (`:2871-2883`).

The gate will run the ordinary kt=1 step, then the ordinary kt=2 step twice:
baseline and with only the six raw NEMO kt=1-end histories substituted through
the same production barotropic path. It will report kt=2 U/V/SSH unequal counts
and maxima against round-46 NEMO before and after substitution. CONFIRM for
history ownership requires at least one improved row and no upstream input
change besides the six histories. REFUTE otherwise. The proposed raw-history
transcription is written under `manifests/` as an unapplied patch, including
state-shape and restart consequences plus predicted kt=1..10 effects. It is
not applied without decision 33.

## 3. Rule 12 and trajectory

No production transcription is eligible if the first mismatch is the pending
carried-state choice. Otherwise only the shared WS-RK3 statement may change,
and it must be exact on GYRE kt=1..10 before/after, preserve measured LOCK and
OVERFLOW rows, mark ORCA2 UNMEASURED-WITH-SPEC, and mark DINO shared-statement
risk. ZAD is retested only after an eligible item-1 fix lands.

## Decisions

| status | item | disposition |
|---|---|---|
| ASKED | live kt=2 handed-operand table; conditional shared transcription and Rule-12 score; raw-history substitution and decision-33 packet | this round |
| UNASKED | carried-state/configuration/default/threshold change; NEMO run/build/source edit; TKE/year-harness edit; ZAD landing before items 1--2 | forbidden |
