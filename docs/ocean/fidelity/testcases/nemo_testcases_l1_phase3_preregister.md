# NEMO testcase lane 1 — phase-3 trajectory preregistration

Date: 2026-08-31

Scope: LOCK_EXCHANGE-zco first, then OVERFLOW-zps.  This file is committed
before any phase-3 EOS score, selector measurement, finer-cadence NEMO run, or
legoESM trajectory is produced.

## Dependency order and stop rule

The lane advances only in this order:

1. NEMO TEOS-10 density polynomial on registered dump states.
2. The three previously waived programs, separately per case: barotropic time
   filter, active UP3 vertical momentum flux, and tracer RK3 stage program.
3. Full-state step-entry trajectory, LOCK before OVERFLOW.

At trajectory step `kt`, compare the registered **before/Nbb** state first,
then walk the executed term/stage registry in oracle call order.  Stop at the
first measured row over its bar.  That row must be owned—source-aligned,
implemented as a selectable canonical option, directly tested with a planted
violation, and remeasured—before `kt+1` or a later term is considered.  No
aggregate later-step score may hide an earlier divergence.

## Oracle source resolution

Both resolved namelists select TEOS-10 (`LOCK output.namelist.dyn:45-46`,
`OVERFLOW :46-47`).  `src/OCE/TRA/eosbn2.F90:1890-1916` reads and resolves the
EOS flags; `:1920-1929` selects Conservative Temperature/Absolute Salinity and
its normalization.  `eos_insitu` evaluates the density polynomial at live
`gdept(Knn)` in `:240-288`.

The pre-implementation search found the canonical Roquet polynomial and the
full `_ROQUET_TEOS10` coefficient table already in `ocean/eos.py`, plus NEMO
alpha/beta/BN2 consumers.  It found no dispatchable density scheme for that
coefficient set: `make_eos_fn` exposes `nemo_eos80` but not `nemo_teos10`.
Therefore a confirmed Veros-GSW difference will be closed by adding only a
selectable canonical dispatch arm and selecting it on these cards; the default
EOS remains unchanged.

For split-explicit time stepping, both cases resolve `ln_dynspg_ts=T`,
`ln_bt_fw=T`, `nn_e=30`.  LOCK resolves `nn_bt_flt=3`, `rn_bt_alpha=.07`
(`LOCK output.namelist.dyn:439-447`); OVERFLOW resolves `nn_bt_flt=1`, alpha
zero (`OVERFLOW :440-448`).  NEMO defines filter 1 as the `nn_e`-wide boxcar
and filter 3 as Demange in `src/OCE/DYN/dynspg_ts.F90:1068-1094,1265-1273`.
The search found canonical `nemo_ab3am4` for filter 3 but no shipped filter-1
program; filter 1 must be a per-case selectable canonical option if measurement
confirms the gap.

The active momentum arm is `ln_dynadv_up3=T` (LOCK `:414-420`, OVERFLOW
`:415-421`).  `src/OCE/DYN/dynadv.F90:78-90` dispatches it to
`dyn_adv_up3`; its vertical flux is `dynadv_up3.F90:250-358`.  The existing
`vertical_momentum_scheme="nemo_advective"` is a `dynzad.F90` transcription and
does not certify this active arm.

`src/OCE/stprk3.F90:194-207` calls stages 1, 2, 3 with the documented level
swaps.  Within a stage, `src/OCE/stprk3_stg.F90:519,586,598` calls tracer
advection, lateral diffusion, and vertical diffusion/update.  The phase-3
time-level registry must cite these sites and refuse unregistered stage dumps.

## Pre-registered measurements and bars

### E1 — EOS on actual registered states

For every wet, non-dummy T point in the available `kt=1`, midpoint, and final
step-entry dumps, produce:

- `veros_gsw_vs_nemo_teos10.normalized_max_abs`;
- `nemo_teos10_vs_independent_nemo_literal.normalized_max_abs`;
- dtype and wet-cell count per case/dump.

The pointwise bar is `1e-15`, with scale `max(max(abs(oracle density)),1)`.
`<=1e-15` is AT-BAR; otherwise DEBT.  A coefficient perturbation must turn the
direct gate red.  If the existing canonical table itself misses the independent
literal evaluator, stop and repair it before dispatch/card work.

### S1–S3 — selector programs

Each selector gets an ordered internal table and a direct synthetic control:

- barotropic filter: coefficient arrays, window weights, loop length, carried
  histories, and returned state/transport convention;
- UP3 vertical momentum: W transport, face interpolation/masks, reconstruction,
  vertical flux, divergence, and bottom level;
- tracer RK3: stage input level, stage coefficients, tracer advective flux,
  diffusion/update, and committed after level.

Pointwise rows use `1e-15`; reductions/recurrences use `1e-12`.  A row without
an executed oracle value is UNMEASURED, never inferred from a later trajectory.

### T1 — state trajectory

The NEMO entry dump remains **before/Nbb**.  Existing sparse dumps are
insufficient for first-divergence walking, so the pinned NEMO configs may be
rerun on CPU with a dump at every early step.  The run must retain cpp keys,
namelist, binary, geometry, and initial state; only stop length and dump cadence
may change, and both committed/run hashes are recorded.

For each `kt=1,2,3,...`, score T, S, u, v, SSH on registered masks.  Exact
identity is reported for the prefix where it holds.  After the first nonexact
row, the fixed pointwise bar is `1e-15`; record the first over-bar step and stop.
The planted control changes one wet T value by 1 C and must fail at `kt=1`.

## Coverage and withheld claims

The phase-3 gate extends the mesh/namelist/restart registry with the executed
RK3 step call graph one level through dispatch.  Every call is VERIFIED,
WAIVED with a source reason, or UNMEASURED.  TEOS-10 density, each selector
program, and each trajectory advance only when their predecessor is AT-BAR.
Until then, all later rows remain loudly UNMEASURED.  No statistical or
phenomenological trajectory claim is in scope; this is a numerical alignment
campaign only.
