# NEMO testcase lane 2 GYRE — causal-column and barotropic preregistration

Date: 2026-09-01  
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`  
Parent evidence commit: `fbeef5d4d3e0513f4b1b8bedbd2e8bc7c1a74e91`

This file freezes the next measurements before either oracle instrumentation
or a candidate run is changed.  It also dispositions the two minor findings
from Claude's SHIP review of the resolved-operator closure.

## Review-note preregistration

1. The existing test named
   `test_resolved_program_coverage_is_oracle_block_driven` is a source-string
   tripwire, not a parser of `output.namelist.dyn`.  It will be replaced by a
   runtime enumeration of every resolved block whose name begins `NAMDYN`,
   `NAMZDF`, or `NAMTRA`; the gate will fail if that set and the disposition
   set differ.  A synthetic fourteenth block must make the test fail.  Until
   that change lands, the historical 13-row inventory is a static checklist.
2. The `momentum_scheme_identity` and `momentum_level_laplacian` scaling arms
   are preregistered **NEAR-NULL AT kt=2**: their approximately `8e-9` movement
   is far below the approximately `1e-4` stage-entry velocity residual.  They
   have no discriminating power at this rung; this is not exoneration.

## Oracle program read before hypotheses

The resolved run prints `ln_dynspg_ts=T`, `ln_bt_auto=T`, `nn_e=50`,
`nn_bt_flt=3`, `rn_bt_alpha=.07`, `ln_bt_fw=T`, and a 288 s external step in
`phase3/gyre_kt1_10/ocean.output:845-861`.  `stp2d.F90:126-202` builds the
complete depth-mean momentum forcing and `stp2d.F90:247-281` builds the water
flux forcing and calls `dyn_spg_ts`.  The RK3 call is once per whole step at
`stprk3.F90:202-230`.

The concrete external-mode recurrence is:

- RK3-specific forcing setup and removal of the entering barotropic Coriolis:
  `dynspg_ts.F90:270-301`;
- forward seed and 50-substep loop: `dynspg_ts.F90:468-520`;
- AB3 predictor and QCO mid-step depths: `dynspg_ts.F90:532-609`;
- continuity and transport accumulation: `dynspg_ts.F90:623-643`;
- AM4/Demange pressure level, live Coriolis, drag, and velocity update:
  `dynspg_ts.F90:671-768`;
- history rotation and final-substep selection for filter 3:
  `dynspg_ts.F90:804-865`.

The DINO MLF campaign's literal `dynspg_ts` machinery therefore covers the
substep recurrence itself.  RK3 reuses it, but composes it differently:
`stprk3_stg.F90:121-243` uses the module-default `n_baro_upd=np_HYB`, and
`:449-462` imposes the completed external-mode mean at every RK3 stage.  The
MLF-only post-loop barotropic-trend update and Kmm transport reconciliation at
`dynspg_ts.F90:910-1002` are excluded by `key_RK3` and must not be imported
into the GYRE identity.

Stage-entry diffusivities are computed once before `stp_2D` by
`stprk3.F90:174-204`; `zdfphy.F90:268-344` composes TKE, EVD, and background
floors into public W-point `avm`/`avt`.  RK3 two-band shortwave is applied only
at stage 3 by `stprk3_stg.F90:584-615`; its literal absorption recurrence is
`traqsr.F90:627-712`.

## Frozen measurements and verdicts

All comparisons use CPU, fp64 state and fp64 geometry.  The bar remains
normalized max error `<=1e-15`.  Scaling is written before any owner label.

### C1 — TKE/EVD/background causal arm

WRITE-only NEMO instrumentation will dump the stage-entry W-point `avm` and
`avt` immediately after `zdf_phy`, before `stp_2D`.  legoESM will diagnose the
profiles through its production `diagnose_vertical_K` path.

- Direct-profile result: AT-BAR only if every wet interior W point is within
  the exact bar after a declared W-level mapping; otherwise DEBT.
- Causal arm: replace only legoESM's vertical `A_v/K_v` solve operands with
  the dumped oracle profiles for the kt=1→2 step.  Record the movement and
  residual for T/S/u/v/SSH.
- CONFIRMED owner only if that one replacement clears the affected whole-step
  rows.  A residual-scale movement without clearance is
  `CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER`.  Movement below 0.1 residual is
  `NEAR-NULL_NO_DISCRIMINATING_POWER`, never exoneration.

### C2 — two-band shortwave causal arm

WRITE-only NEMO instrumentation will dump `qsr` and the exact stage-3
temperature-RHS increment produced by `tra_qsr`, without assigning a
prognostic field.  legoESM will evaluate its production Jerlov kernel on the
same stage-entry ladder.

- Direct-profile result: AT-BAR only at normalized max `<=1e-15` on wet cells.
- Causal arm: replace only legoESM's shortwave temperature-rate contribution
  by the dumped oracle increment for the kt=1→2 step.
- Labels follow the same clear / residual-scale / near-null rule as C1.

### B1 — within-substep barotropic first divergence

A WRITE-only `dynspg_ts` extension will dump, for each of the 50 substeps at
`kt=1`, the entry `(ssh,u,v)`, AB3 mid-step `(ssh,u,v)`, continuity output,
AM4 pressure-level SSH, component tendencies `(spg,coriolis+drag,slow)`, and
exit `(ssh,u,v)`.  A private legoESM trace will expose those same executed
operands without creating a public configuration switch.

For each registered frame, score SSH/u/v on their native masks and record the
first field and first substep over the bar.  If substep entry is exact and the
first mismatch appears in continuity, the boundary is continuity/transport;
if continuity is exact and the velocity exit is not, compare SPG, live
Coriolis+drag, and slow forcing in that order.  No owner is assigned from an
end-of-loop frame alone.

### Instrument bit-identity and planted controls

The instrumented NEMO rerun must reproduce the existing uninstrumented
`kt=1/2` step-entry, stage 1/2/3, RHS, and post-`stp_2D` frame payloads
bit-for-bit.  Each new reader must reject a corrupted magic/header, and each
new numerical gate must ship a planted single-value violation.  Failure of
any control is exit 2 and invalidates all scientific rows.

## Stop rule

This round stops after the first within-substep boundary and the two causal
arms.  Any surviving owner remains `UNMEASURED` unless its preregistered
single-variable replacement clears the relevant exact-bar row.  No trajectory
claim beyond kt=2 is authorized by these measurements.
