# Lane 1 closure receipt — NEMO OVERFLOW and LOCK_EXCHANGE

Date: 2026-09-01

Session: `01a0591f-a335-7060-9257-6c47bf2149ee`

## PR verdict

Lane 1 certifies two NEMO 5.0.2 oracle configurations and their legoESM
geometry/initial-condition/first-step harnesses:

- `LOCK_EXCHANGE-zco`: `key_qco + key_RK3`, flat z-coordinate tank;
- `OVERFLOW-zps`: `key_qco + key_RK3`, priority partial-cell slope case.

The oracle geometry and run inventory are certified fail-closed, and both
legoESM cards match geometry and kt=1 initial conditions to the registered
roundoff/bit bars.  This PR does **not** claim trajectory parity.  Both exact
prefixes end at kt=1; the first evolved state, kt=2, remains DEBT.  The
owner-exhausted continuations through kt=60 show polynomial accumulation or
bounded oscillation, with no measured amplifying mode.

Post-PR round 4 supersedes the OVERFLOW external-mode rows below.  A new
19-frame NEMO/legoESM substep walk confirms that the executed NEMO flux-form
face-transport update was missing.  The canonical correction removes the
internal substep-2--4 amplification by `7.2e5--9.5e7x` and improves kt=2 SSH
from `1.23723132e-7` to `1.04916076e-14`, but kt=2 instantaneous U regresses
`7.42%` to `3.31108168e-6` and T remains `2.40034479e-8`.  Accordingly the
update is **CONFIRMED** as owner of the downstream external recurrence and
**REFUTED** as owner of the kt=2 root initiator.  That root is **UNMEASURED
outside the exhausted external-substep register**.  The same-revision
full-duration verdict is four `OUTSIDE`, two `WITHIN-SCHEME-SPREAD`, and zero
`INDISTINGUISHABLE-AT-FLOOR`; see the stability receipt's round-4 section and
machine gate `b46f7390...`.

## What is certified

- Phase 1: source dossiers, resolved cpp/namelist state, CPU builds, completed
  oracle runs, trajectory instruments, and a geometry/coverage registry that
  hard-fails on unaccounted mesh/restart/namelist arrays.
- Phase 2: fp64 legoESM cards, geometry and IC parity, explicit time-level
  registry, and exact kt=1 entry alignment.  OVERFLOW's T-point partial-cell
  bottom rule and neighboring-min U-face thickness retire D2.5
  (`tools/DOMAINcfg/src/domzgr.F90:1163-1169` and
  `tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:157-186`).
- Phase 3: source-bound canonical options, non-vacuous planted controls,
  same-frame instantaneous velocity scoring, separate time-mean transport
  scoring, first-divergence ownership ledger, and explicit continuation to
  kt=60.  Every candidate floating array is `float64` under
  `PrecisionPolicy.fp64()`.

At kt=60 the normalized wet-point L-infinity debts are:

| case | T | S | instantaneous u | SSH |
|---|---:|---:|---:|---:|
| LOCK | `3.95492e-6` | `2.84217e-15` | `7.41114e-6` | `3.84974e-6` |
| OVERFLOW-zps | `7.96523e-4` | `3.65422e-15` | `4.13339e-3` | `1.98816e-4` |

These are measurements from an already-divergent prefix, not match claims.

## Fix inventory and oracle sources

| canonical implementation | reference exercised by the cards |
|---|---|
| NEMO/Roquet TEOS-10 density and geometric depth | `src/OCE/TRA/eosbn2.F90:253-288,1920-2108` |
| terrain-following `hpg_sco` pressure gradient | `src/OCE/DYN/dynhpg.F90:117-123,340-390` |
| LOCK Demange filter 3 and OVERFLOW forward boxcar filter 1 | `src/OCE/DYN/dynspg_ts.F90:1058-1102,1223-1273` |
| literal continuity and secondary time-mean transport accumulation | `dynspg_ts.F90:603-643,999-1000` |
| flux-form primary transport average and post-average depth division | `dynspg_ts.F90:823-834,956-979` |
| active horizontal/vertical UP3 momentum flux | `dynadv.F90:87-89`; `dynadv_up3.F90:141-365` |
| NEMO WS-RK3 scheme identity | `stprk3.F90:184-207`; `stprk3_stg.F90:112-303,433-446,456-519` |
| Kmm tracer transports and distinct `un_adv/H` reconcile | `stprk3_stg.F90:250-303,456-519` |
| two-step RK3 FCT low-order predictor | `src/OCE/TRA/traadv_fct.F90:153-161,470-641` |
| Campin-Goosse BBL option 2 | `src/OCE/TRA/trabbl.F90:243-284,342-353,415-454`; stage calls `stprk3_stg.F90:468,588` |
| adaptive vertical advection and implicit vertical mixing placement | `stprk3_stg.F90:281-303,428-430` |

The standing user rule is recorded and enforced: **no Frankenstein
compositions**.  Every selectable arm names either its exercised NEMO
source/namelist or “legoESM legacy, pre-existing behavior.”  NEMO exposes no
micro-switches inside its RK3 program, so selecting the NEMO scheme implies
Kmm transports, the two-step FCT predictor, per-stage primary correction,
transport reconciliation, and flux-form primary transport averaging as one
unbranched package.  A/B omissions exist only as private test/harness hooks.
Validation rejects the unattested public combinations inventoried in the
phase-3 receipt, and the certified cards disable legoESM eta diffusion because
the oracle has no corresponding stabilizer.

## Ownership ledger

- **CONFIRMED_EXONERATED:** NEMO TEOS-10/`hpg_sco` at the first active LOCK
  RHS; full u RHS is at the registered bar.
- **CONFIRMED_REQUIRED:** Kmm tracer transports, the two-step FCT program,
  per-stage primary velocity correction, and distinct advecting-transport
  reconciliation.  Their legacy/omission arms worsen direct source-level or
  trajectory measurements.
- **CONFIRMED_FRAME_ARTIFACT:** the former `2.43e-2 m/s` OVERFLOW and
  `1.135e-3 m/s` LOCK velocity debts compared NEMO time-mean transport with
  legoESM instantaneous velocity.  Same-frame scoring restored OVERFLOW to
  `4.121e-6 m/s`, about `5901x` better, before the final primary-transport
  correction improved it to `3.956e-6 m/s`.
- **PLAUSIBLE_CONTRIBUTOR_NOT_OWNER:** omitting OVERFLOW's required stage
  primary correction moves `24.7836%` of the kt=2 T residual, but leaves
  `3.6109e-7 K` and destroys stage-u alignment.
- **REFUTED_AS_PRIMARY_OWNER:** OVERFLOW flux-form primary transport averaging
  moves kt=2 T by only `9.45e-13 K`; one-step FCT, frozen-final transports,
  transport-reconcile omission, wrong prognostic transport frame, and BBL at
  the shipped kt=1 geometry likewise do not own the remaining T residual.
- **UNMEASURED_AFTER_TWO_ARMS:** LOCK kt=2 instantaneous-u tail
  `1.7121e-10 m/s`.
- **UNMEASURED_AFTER_REGISTERED_ARMS:** the remaining roughly 75% of
  OVERFLOW kt=2 T (`4.8007e-7 K` absolute), and term-level ownership of the
  later T/u/SSH growth.
- **ONE-SIDED:** legoESM BBL is confirmed inactive at OVERFLOW-zps kt=2;
  NEMO BBL is only geometrically plausible inactive because `utr_bbl` was not
  dumped.

## Growth and the sco finding

The gate records every successive-step ratio and compares tail log-log and
semilog fits.  LOCK T/u/SSH prefer polynomial fits with
`p=3.545/3.306/5.079`; OVERFLOW T/u prefer polynomial fits with
`p=1.707/1.697`; OVERFLOW SSH is bounded/oscillatory with a negative fitted
rate.  Full-history ratios are not monotonically decreasing in every series,
so that reviewer wording is retracted; no series selects the exponential
open-mode classification through kt=60.

OVERFLOW-sco remains an oracle finding, not a shipped trajectory oracle.  With
BBL enabled it fails at step 4772; the one-variable `ln_trabbl=.false.` control
still fails at step 4773 with the same `|U|max≈10.01 m/s` location.  BBL is
therefore refuted as sole owner even though the every-column downslope mask is
a real mechanism.  TEOS-10 sigma-coordinate pressure-gradient truncation is
the next registered suspect; no chained second arm was run.  The sco run has
no final restart and remains loudly incomplete.

## Artifacts and review state

The quota-heavy runs and JSON reports live under
`/data/abyssal/dbalwada/nemo-testcases-l1/phase3/`.  Their hashes are pinned in
`nemo_testcases_l1_phase3_artifacts.sha256`.  The two kt=60 oracle reruns each
contain 60 entry and 60 barotropic-frame dumps; kt=1--10 entry dumps are
byte-identical to the shorter certified runs.  Independent review shipped the
geometry/kt=1 scope and both preceding phase-3 rounds.  Review coverage:
Claude adversarial review across seven rounds (codex-authored work); the
mandatory GLM second-reviewer pass has NOT run (reviewer unreachable) and
remains OUTSTANDING — this lane is one reviewer short of the dual-review rule.  This closure preserves
all remaining DEBT/UNMEASURED labels rather than promoting the continuation to
trajectory parity.
