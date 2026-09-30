# Preregistration: GYRE kt=2 LDF/ENE statement alignment, round 49

Date: 2026-09-11. Frozen at legoESM `665dc7fa6472` before any round-49
counterfactual, production edit, or trajectory measurement. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round49/`. CPU/fp64 only.
The admitted input is round 46's six-stage record; no NEMO executable will be
built or run.

## Compiled-arm correction and source contract

The task's EEN/ENS premise is retracted for this compiled card. The compiled
namelist selects `ln_dynvor_ene=.true.`
(`cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/EXP00/namelist_cfg:165-168`). The measured
operator is therefore `vor_ene`, and no EEN/ENS result may be printed as a
round-49 verdict.

Stage 1 calls HPG, LDF, VOR, KEG, ZAD in that order and calls
`dyn_ldf(Kbb,Kbb)`
(`cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-175`). Stages 2
and 3 call HPG, VOR, ADV on Kmm
(`.../BLD/ppsrc/nemo/stprk3_stg.f90:431-480`), while stage 3 later calls
`dyn_ldf(Kbb,Kmm)` (`.../BLD/ppsrc/nemo/stprk3_stg.f90:690-712`). These are
the only admitted execution arms.

| operator | compiled statement | required operand family | current legoESM substitution ladder |
|---|---|---|---|
| LDF curl | `dynldf_lev.f90:121-125` | Kbb velocity; live `e3f`; `ahmf` | velocity -> coefficient -> live F thickness |
| LDF divergence | `dynldf_lev.f90:127-129` | Kbb `e3t/e3u/e3v`; Kbb velocity; `ahmt` | velocity -> coefficient -> complete Kbb thickness |
| LDF output | `dynldf_lev.f90:132-140` | Kmm `e3u/e3v` only on curl divisions | Kmm output thickness last |
| ENE PV | `dynvor.f90:536-557` | Kmm velocity, `ff_f`, live `e3f` | velocity -> F thickness |
| ENE transport | `dynvor.f90:559-562` | `e2u*e3u(Kmm)*u`, `e1v*e3v(Kmm)*v` | recorded face thickness -> recorded velocity |
| ENE trend | `dynvor.f90:567-574` | pair sums, then `r1_4*r1_e1u/e2v*(...)` | recorded reciprocal and literal association last |

The pre-implementation search found the reusable production seams
`_nemo_ws_qco_stage_faces`, `nemo_qco_live_vorticity_e3f_cgrid`, and the
round-46 self-describing reader. No second geometry formula or record reader
is permitted.

## Frozen discriminator

Run a JIT model-path factorial on every kt=2 stage. Each arm replaces exactly
one named family while all other inputs remain those of the baseline. Exact is
zero unequal wet values; no tolerance verdict exists.

* **LDF prediction:** Kbb velocity and the compiled constant/masked coefficient
  arms do not close the row. Replacing the complete thickness contract — Kbb
  `e3t/e3u/e3v`, live `e3f`, and Kmm output `e3u/e3v` — makes stages 1 and 3
  exactly equal. Otherwise the candidate is REFUTED and is not landed.
* **ENE prediction:** recorded Kmm velocity, `e3f`, and metric-complete
  transports do not individually close the row because production already
  receives them. Replacing live `/e1u,/e2v` with the recorded
  `r1_e1u/r1_e2v` and evaluating the line-572/573 product association makes
  every stage exactly equal. Otherwise the candidate is REFUTED and is not
  landed.

The identity arm must pass. One-nextafter plants in each causal family, a
header/truncation plant, and a false commit stamp must exit nonzero. The gate
must emit raw counts and maximum absolute errors for every arm and face.

## Landing, ZAD, trajectory, and Rule 12

Only CONFIRMED statements enter the shared implementation. The LDF API must
retain its historical single-thickness default for non-WS callers; the NEMO
identity path supplies the full family without a card-name guard. The ENE
implementation must use the compiled reciprocal association for the existing
`een_metric_weighting="nemo"` route; its off route remains pinned.

After both candidates land, reapply the preserved round-47 stage-3 explicit
ZAD patch and rerun its unchanged two-sided comparison. **CONFIRM** means its
57 worsened kt>=3 rows become zero and it improves or preserves every scored
row; only then may it land. Any worsened row is **REFUTE**, and the patch stays
rejected.

Run the unchanged GYRE production-JIT kt=1..10 gate before and after and report
the first normalized-bar failure. For each landed statement report
GYRE-zco, LOCK_EXCHANGE-zco, OVERFLOW-zps, DINO, and ORCA2 as tested,
value-inert, unsupported/unmeasured, or worsened; no card may be silently
waived. Targeted unit tests must exercise eager, JIT, gradient, default-pin,
unknown-selector, and red-operand behavior.

## Decisions

| status | item | disposition |
|---|---|---|
| ASKED | transcribe compiled LDF/ENE statements, re-test ZAD, rescore kt=1..10 and all named cards | this round, gated as above |
| UNASKED | choose EEN/ENS, relax a bar, create a GYRE-only guard, build/run NEMO, or infer absent ORCA2 evidence | forbidden / stop and report |
