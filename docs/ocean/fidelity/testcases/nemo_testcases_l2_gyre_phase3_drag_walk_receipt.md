# NEMO testcase lane 2 — GYRE bottom-drag landing receipt

Date: 2026-09-01

Session: `ea650f83-28b8-4c68-b0cc-809a9fd417de`

Preregistration commits: `0638308b06a31d01157a32dfad23757d587e8ae2`,
`0398f39041e78b2571745019a39d0fd6c8bd8f98`

Parent ENE commit: `b2f7c298411f03848b54f0df64c0cf617a710665`

## Verdict

**DEBT, with the complete 50-substep external-mode program and kt=2 SSH now
AT-BAR.**  The bottom-drag component is a confirmed causal owner of the prior
`6.731208045740773e-14` combined-trend residual: the collapsed canonical drag
identity moves that residual by exactly the same magnitude and leaves
`trd_u/trd_v` at `2.0679515313825692e-25` absolute.  The first surviving
momentum boundary is now RK stage 2; T/S stage numerics remain UNMEASURED.
Whole-step kt=2 T, S, U, and V and the kt=3-10 trajectory remain DEBT.  No
trajectory is called matched.

## Resolved source program and canonical search

The resolved `output.namelist.dyn:328-343` selects `ln_non_lin=T`,
`ln_drgimp=T`, `rn_Cd0=1e-3`, `rn_ke0=2.5e-3`, `rn_Cdmax=.1`, and
`rn_z0=.003`.  Although that block prints `ln_drgice_imp=T`,
`zdfdrg.F90:320` disables it unless `nn_ice=2`; GYRE records `nn_ice=0` at
`ocean.output:525` and the effective false value at `ocean.output:625-630`.
This is therefore bottom-only nonlinear drag.  NEMO has no `nn_drg` selector
here: `zdfdrg.F90:311,335-341` collapses its logicals into the `ndrg` identity.

`zdfdrg.F90:183-190` constructs the Nbb nonlinear T-point coefficient.
`stprk3.F90:182-205` diagnoses it before the RK stages, and
`stp2d.F90:195-196` passes Kbb/Kbb to `dyn_drg_init`.
`dynspg_ts.F90:289-290` freezes the face coefficient,
`dynspg_ts.F90:1671-1675` performs the bottom-only face average, and the live
no-WAD branch at `dynspg_ts.F90:729-750` adds
`(zCdU_u*un_e)*hur_e`/`(zCdU_v*vn_e)*hvr_e` after ENE.  The implicit division
at `:808-814` is dead for this GYRE arm.

The pre-implementation search found all required kernels on origin/main
ancestry: `nemo_bottom_drag_rate_faces`, `zdf_drag_in_matrix`, and
`barotropic_drag_substep`.  The constructor's single-owner guard makes the
last two indivisible under this resolved RK3 program.  The GYRE card therefore
changes one collapsed identity from `(False,False)` to `(True,True)`; mixed
pairs are rejected.  `nemo_stage_mean_imposition=True` remains its canonical
RK3 baroclinic-only sibling.  No new drag kernel or public selector was added,
and defaults and other cards are unchanged.

## WRITE-only instrument and controls

The separate `NEMO_L2_BTDRG_1` record stores the frozen coefficient and, for
all 50 substeps, entry velocity, literal inverse depth, coefficient-times-
velocity, isolated drag, Coriolis, and combined trend.  It never assigns model
state.  The pre-existing step-entry, stage, RHS, barotropic-frame, v2 substep,
and ENE-coefficient artifacts retain their registered hashes bit-for-bit.

The reader rejects corrupt magic, header, and payload.  A planted `+1` in a
wet U coefficient was detected at exactly `1.0` and classified DEBT.  The
independent `rn_ke0=0` red control moves the at-rest coefficient by `5e-5`, so
the floor control is nonvacuous.

## First-divergence and scaling evidence

All values are fp64 pointwise maxima on native owned masks.  Scaling was
recorded before the owner label.

| registered row | absolute max | oracle max | relative | status |
|---|---:|---:|---:|---|
| frozen drag coefficient U/V | `0` | `5e-5` | `0` | AT-BAR |
| substep-2 U entry | `2.541098841762901e-21` | `5.789794768264535e-6` | `4.38893e-16` | AT-BAR |
| substep-2 V entry | `3.3881317890172014e-21` | `5.810693254979859e-6` | `5.83086e-16` | AT-BAR |
| literal inverse depth U/V | `5.421010862427522e-20` | `2.325197460624262e-4` | `2.33142e-16` | AT-BAR |
| coefficient-times-velocity | `1.550963648536927e-25` | `2.9053466274899296e-10` | `5.36e-16` | AT-BAR |
| reconstructed drag | `5.048709793414476e-29` | `6.755504590904853e-14` | `7.51e-16` | AT-BAR |
| production drag | `3.7865323450608567e-29` | `6.755504590904853e-14` | `5.63e-16` | AT-BAR |
| post-drag `trd_u/trd_v` | `2.0679515313825692e-25` | `5.409032045634095e-10` | `3.83e-16` | AT-BAR |

The legacy no-composition residual and causal movement are both
`6.731208045740773e-14`, a ratio of `1.0`.  Direct operands, isolated terms,
and combined trends all clear the bar, supporting the honest labels
**CONFIRMED_CAUSAL_OWNER_OF_BOTTOM_DRAG_COMPONENT** and
**CONFIRMED_OWNER_OF_COMBINED_TRD**.

Every registered boundary of all 50 barotropic substeps is AT-BAR; the
substep first-over-bar field is null.  The final post-`stp_2D` state also
propagates to kt=2 SSH at `5.204170427930421e-18`, AT-BAR.  Stage 1 U/V remain
AT-BAR.  The next ordered boundary is stage 2 U/V at
`1.0381851039053744e-7`/`2.4701793455947666e-7`; this is not yet an owner
finding.

## kt sweep after the drag landing

| kt | T max error | S max error | U max error | V max error | SSH max error |
|---:|---:|---:|---:|---:|---:|
| 2 | `5.162449518e-2` | `4.035174225e-3` | `2.525607457e-2` | `2.525682546e-2` | `5.204170428e-18` |
| 10 | `5.563805194e-1` | `9.806306601e-3` | `4.217945430e-2` | `9.093125938e-2` | `6.559165432e-4` |

The sweep is a debt characterization, not certification.  Its first whole-
step over-bar fields are T, S, U, and V at kt=2.  The ordered register now
continues beyond the fully cleared external-mode loop into the RK stage-2
composition operands, followed by stage 3 and T/S stage numerics.

## Review and artifacts

Claude accepted the preceding ENE round.  No independent Claude or GLM claim
review is available for this new drag landing, so it is explicitly
**UNREVIEWED**, not dual-reviewed.

| artifact | SHA256 |
|---|---|
| kt=1-10 drag gate | `10e3ad27df36a461be75a91cef089dd7e9fe79c0378e9fd6472b3a7e2e1b6d87` |
| planted coefficient control | `a83956e31c15e9ece263cb213965f3478cb758def97cee006900f629c0ef315a` |
| drag operand dump | `138ec8c117e0a0b26ae0f894a504f12e71c5f4406e6079fc55b2fe2c469101c1` |
| instrumented `nemo.exe` | `49698e289f5817d146c5e4c57aca0b5afbee3b32f85d5e71edc985b7b5fd8b9b` |
| `MY_SRC/dynspg_ts.F90` | `6558511b9b7387f260f32d0a0583383031993620576a430b0ab2c6dd2c0bb17c` |

Run root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10_drag_walk`.
The outstanding stage boundaries and T/S stage numerics remain UNMEASURED.
