# NEMO testcase lane 2 GYRE — bottom-drag and ordered substep preregistration

Date: 2026-09-01

Session: `ea650f83-28b8-4c68-b0cc-809a9fd417de`

Parent evidence commit: `b2f7c298411f03848b54f0df64c0cf617a710665`

This freezes the next measurements before oracle instrumentation, card
selection, or trajectory reruns change.  The accepted starting boundary is
substep-2 combined `trd_u`: `6.731208045740773e-14` absolute,
`0.012444%` of the `5.409032045634095e-10` oracle magnitude.  The isolated
ENE term is already AT-BAR; bottom drag is the first unmeasured addend.

## Resolved GYRE drag program

The runtime-resolved `output.namelist.dyn:328-343` selects
`ln_non_lin=T`, `ln_drgimp=T`, `rn_Cd0=1e-3`, `rn_ke0=2.5e-3`,
`rn_Cdmax=.1`, `rn_z0=.003`, and no regional boost.  It also prints
`ln_drgice_imp=T`, but that is not the executing value: `zdfdrg.F90:320`
forces it false unless `nn_ice=2`, while this GYRE run records `nn_ice=0` in
`ocean.output:525` and prints effective `ln_drgice_imp=F` at
`ocean.output:625-630`.  The live program is therefore bottom-only nonlinear
drag with no top/ice contribution.  NEMO 5.0.2 has no `nn_drg` selector in
this block; `zdfdrg.F90:311,335-341` resolves the four logical scheme flags to
one `ndrg` identity.

`zdfdrg.F90:183-190` constructs the Nbb T-point coefficient

`rCdU_bot = -Cd0 * sqrt(0.25*((u_e+u_w)^2+(v_n+v_s)^2) + ke0)`.

For RK3, `stprk3.F90:182-205` diagnoses vertical physics at Nbb and calls
`stp_2D` before all three RK stages.  `stp2d.F90:195-196` calls
`dyn_drg_init(Kbb,Kbb,...)`; `dynspg_ts.F90:289-290` freezes its face
coefficients for the complete external-mode window.  The bottom-only face
average is `dynspg_ts.F90:1671-1675`.  Inside the substep loop,
`dynspg_ts.F90:729-750` first evaluates live ENE and then adds
`(zCdU_u*un_e)*hur_e` and `(zCdU_v*vn_e)*hvr_e`.  GYRE has no wetting/drying,
so the explicit branch executes; the implicit division at `:808-814` is dead.

## Pre-implementation search

The repository search found the canonical DINO machinery already on the card
ancestry: `nemo_bottom_drag_rate_faces` transcribes the T-point nonlinear law
and face average; `barotropic_drag_substep` freezes that rate and applies it
inside the shared split-explicit loop; the `nemo_quadratic` drag identity is
already selected by GYRE.  Existing unit coverage includes the nonlinear
floor, face mapping, implicit-3D composition, substep drag, differentiability,
and single-owner guards.  GYRE selects the quadratic law but leaves both
`zdf_drag_in_matrix=False` and `barotropic_drag_substep=False`.  The existing
constructor guard proves these are one indivisible NEMO composition:
`ocean_model_latlon_cgrid.py:2579-2625` requires the substep owner when the
matrix owner removes the old explicit 3-D kick, and `:2633-2671` rejects the
substep owner without that matrix removal.  GYRE already selects
`nemo_stage_mean_imposition=True`, the RK3 sibling of the required
baroclinic-only solve.  No new drag kernel, coefficient knob, or Frankenstein
scheme is authorized.  If the measured arm clears, the canonical pair is
folded into the existing collapsed `gyre_vector_ene_c2` identity; defaults and
all other cards remain unchanged.

## Frozen operand walk and controls

CPU fp64 state and geometry are pinned.  The pointwise normalized maximum bar
remains `1e-15` on native owned masks.  A WRITE-only trace extension will add:

1. frozen `zCdU_u/zCdU_v` face coefficients;
2. substep-2 `un_e/vn_e` and `hur_e/hvr_e` inputs;
3. each source-ordered product, the pure drag term, and the post-drag
   `zu_trd/zv_trd` total;
4. the velocity update operands and exit state.

Every operand records absolute error, oracle magnitude, relative error, and
dtype before an owner label.  The old trace payload, step-entry dumps, stages,
RHS, and post-`stp_2D` frame must remain bit-identical.  New readers reject
corrupt magic/header/payload.  A planted nonzero coefficient violation must
fail by its registered magnitude; a `ke0=0` synthetic arm must materially
change the at-rest coefficient, proving that the background floor is live.

The controlled arm changes one collapsed scheme identity on the GYRE card:
`(zdf_drag_in_matrix,barotropic_drag_substep)=(False,False) -> (True,True)`.
Those internal switches are not separable under NEMO's resolved
`ln_drgimp + ln_dynspg_ts` program; either mixed pair is rejected rather than
run as a Frankenstein arm.  If its movement matches the historical residual
and the direct coefficient/products are AT-BAR, the label is
`CONFIRMED_CAUSAL_OWNER_OF_BOTTOM_DRAG_COMPONENT`.  It is
`CONFIRMED_OWNER_OF_COMBINED_TRD` only if post-drag `trd_u/v` also clear the
exact bar.  Movement under one tenth of the residual is
`NEAR_NULL_NO_DISCRIMINATING_POWER`; partial clearance is
`CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER`.

## Ordered continuation register

After any supported landing, the gate restarts at substep 1 and selects the
first over-bar row in this fixed order:

1. entry and AB3 predictor state;
2. continuity exit and pressure-level SSH;
3. pressure-gradient term;
4. pure ENE term;
5. bottom-drag term and combined trend;
6. frozen slow forcing;
7. source-ordered velocity update and history rotation;
8. weighted velocity/SSH and tracer-transport accumulators;
9. final post-`stp_2D` barotropic frame;
10. RK3 stage 1, stage 2, stage 3, and whole-step kt=2 state.

For a new boundary, direct operands are dumped or reconstructed before any
owner arm.  One-variable scaling precedes labels.  A causally supported
canonical correction triggers a fresh kt=2-10 CPU/fp64 sweep; an unsupported
or uninstrumentable operand stays UNMEASURED.  The loop stops only when every
kt=2 field is AT-BAR or the register is exhausted with each survivor named.
