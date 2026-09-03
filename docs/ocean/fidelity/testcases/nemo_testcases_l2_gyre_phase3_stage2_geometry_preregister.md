# NEMO testcase lane 2 GYRE — stage-2 HPG geometry preregistration

Date: 2026-09-01

Session: `ea650f83-28b8-4c68-b0cc-809a9fd417de`

Parent term split: `e0079d6acdcd7c7917f32970e96e14d0e58a7ff3`

The accepted first term is stage-2 HPG at `1.441923755423408e-11` U and
`3.430804646663045e-11` V.  Oracle stage-1 T/S injection moved only
`8.8301%` of the corrected-stage residual and is
`NEAR_NULL_NO_DISCRIMINATING_POWER`.  This preregisters the remaining Kmm
thermodynamic-geometry operands before extending the causal arm.

## Executing geometry

In the HYB RK3 program, stage 1 constructs the N+1/3 SSH and QCO ratios at
`stprk3_stg.F90:130-175`.  After the time-level swap, stage 2 consumes that
state as Kmm (`stprk3.F90:215-225`).  `eos(ts,Kmm)` and `dyn_hpg(Kmm)` are
called at `stprk3_stg.F90:338-344`.  The selected `hpg_sco` source reads the
Kmm `e3w` and `gdept_z0` ladders in its surface recurrence
(`dynhpg.F90:343-360`) and at every interior level (`:367-388`); under key_RK3
it assigns HPG as the first Krhs term rather than adding to an old accumulator.

The legoESM search found that `compute_frozen_geom_density` is evaluated once
from step-entry T/S/eta at `ocean_model_latlon_cgrid.py:3834-3849` and reused
by `_mom_pert_ws` at `:4161-4195`.  The T/S-only arm recomputed density but
still retained step-entry eta/thickness.  The existing oracle stage artifact
already contains `ssh(Kaa)` (`stprk3.F90:357-374`); the reader previously
discarded that payload.  No new NEMO state-writing or dump format is required.

## Frozen comparisons and arms

The gate will first expose and score stage-1 SSH itself.  The candidate value
is the literal HYB interpolation
`eta_Kbb + (eta_N+1 - eta_Kbb)/3`, using the already accepted post-`stp_2D`
SSH operand.  It then runs two private, source-local HPG arms:

1. `stage1_ssh_only`: replace only eta/derived thickness in stage-2 HPG while
   retaining step-entry T/S;
2. `stage1_thermodynamic_bundle`: replace T, S, and eta together with the
   oracle stage-1 Kmm state.

Each arm changes only the stage-2 EOS/HPG input; velocity, external target,
vorticity, advection, returned tracer state, and later production state remain
untouched.  The direct HPG term and corrected stage-2 U/V are scored for each
arm.  Scaling precedes labels.

- If SSH-only clears direct HPG and downstream Kaa with movement matching the
  residual, label `CONFIRMED_CAUSAL_OWNER_OF_STAGE2_HPG_GEOMETRY`.
- If only the full bundle clears, label
  `CONFIRMED_CAUSAL_OWNER_OF_STAGE2_THERMODYNAMIC_BUNDLE`.
- Movement below one tenth of the residual remains
  `NEAR_NULL_NO_DISCRIMINATING_POWER`; partial clearance remains
  `CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER`.
- A production landing must fold live Kmm T/S/geometry into the complete WS-RK3
  identity.  NEMO has no frozen-density stage switch; no public micro-selector
  or mixed-stage Frankenstein program is allowed.

The stage reader must reject a missing/truncated SSH payload.  A planted `+1`
stage-1 SSH violation must fire at its registered magnitude.  Existing
artifact hashes remain bit-identity controls.  A supported landing triggers a
fresh CPU/fp64 kt=2-10 sweep; otherwise the walk continues through the HPG
operands and then the measured tracer-stage debt with honest UNMEASURED labels.
