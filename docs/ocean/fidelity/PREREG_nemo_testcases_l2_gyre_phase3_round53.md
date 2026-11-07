# Round 53 preregistration — unconditional WS-RK3 ZAD retest

Date: 2026-09-11. Frozen base: `8ecdfd459234`. CPU, production JIT,
fp64/scalar-libm only. NEMO source, executables, and records are read-only.
The sole physics candidate is the byte-preserved
`nemo_testcase_l2_gyre_round47_rejected_stage_zad.patch`; no configuration,
threshold, state, or oracle choice is introduced.

## Executed source and candidate

GYRE resolves vector momentum (`namelist_cfg:159-168`). Its compiled stage-1
program executes HPG, LDF, VOR, WZV, KEG, then ZAD
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-175`). Stages 2/3
construct `ww` from `(Kbb,Kmm,Kaa)` and Kmm velocity
(`stprk3_stg.f90:324-337`) before their vector ADV call
(`stprk3_stg.f90:472-480`). `dyn_zad` consumes that `ww`, Kmm velocity, and
Kmm `e3u/e3v` in the literal recurrence (`dynzad.f90:100-138`).

The candidate passes each stage program's already-materialized `(ww,e3u,e3v)`
to the one shared ZAD kernel. Stage 1 replaces only ZAD by the difference of
the same tendency call with and without those operands. Stages 2/3 pass their
own materialized triplets directly. The existing given-NEMO-input replay must
remain bit-exact.

## Frozen prediction and falsifier

With ENE and LDF now exact given NEMO inputs, round 44's 57 worsened trajectory
rows should disappear, or be limited to kt>=3 rows whose first handed operand
identifies a kt=3 owner. Confirm: GYRE first-over-bar moves later than kt=2 and
no row worsens by more than the two-row-scale-ULP Rule-12 allowance. Refute:
any row's oracle residual worsens by more than the kt=2 improvement buys; report
every worsened row with kt, field, absolute worsening, row-scale ULP, and ULP
count. If refuted, walk the earliest-kt worsened row (largest worsening within
that kt) through the existing live-stage operand table against the independent
per-step entry/stage records and name the first operator and compiled statement.

## Measurements and landing contract

Controlled baseline and candidate runs differ only by the preserved patch.
Both score GYRE kt=1..10 against
`round19_oracle_v2_external`, kt=2 stages against
`round46/oracle_kt2_stage`, and steps 1..60 against
`year_owners/nemo_seed0`. Plants and producer commit stamps must exit nonzero
on mutation/mismatch. The candidate lands only if the Rule-12 comparison is
clean and first-over-bar is later than kt=2.

If landed, measure the shared program on GYRE, LOCK_EXCHANGE, and OVERFLOW.
Record the executed vector/flux arm and every moved row. ORCA2 remains
UNMEASURED-WITH-SPEC pending a native record; DINO uses the separate leap-frog
program but retains shared-statement regression risk. Run the identical
one-day-cadence, days-1..30 member and score both baseline and candidate against
`year_owners/nemo_seed0` with the existing day-gap instrument.

## Default-root correction

Change `ROOT`, `STAGE2_ROOT`, and `STAGE3_ROOT` to the certified scalar-math
`round19_oracle_v2_external` root. Before any scoring, each supplied root's
kt=2 entry must be byte-identical to that canonical root; a one-bit/root plant
must fail. Re-run the trajectory gate with no root flags and require its
kt=1..10 rows to reproduce the explicit-root result exactly.

ASKED: all work above. UNASKED: none. Forbidden: NEMO run/build/edit,
configuration/default physics choices, threshold relaxation, carried-state
change, per-card guard, GPU use, deletion, merge, or push.
