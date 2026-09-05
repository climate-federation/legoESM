# SI3 lane 3b round 17 preregistration — kt2 momentum owner

Date: 2026-09-04

Tracker: `climate-federation/legoESM#1699`

Parent: `4c3550e411456d6d348b372250bf1bed8eae7020`

State: **PREREGISTERED; NEW KT2 OPERANDS UNMEASURED**

## Frozen observation and stopping rule

Round 16 established all six `kt=2 PRE_SSM` state rows bit for bit.  Advancing
that exact state one more hour produces the first red row at
`kt=3 PRE_SSM.u`: absolute and normalized error
`1.7912434032934096e-3`; `v` behind it is
`1.8350774187287796e-3`.  T, S, SSH, and e3t remain bit-identical.  Those
numbers locate the interval from the kt2 `sbc_ssm` entry through the second
coupled ocean step; they do not identify an operator.

A fresh config-local scalar-math copy will dump the kt2 path only.  No shipped
NEMO file and no retained oracle root will be changed.  The target after a
source-backed correction is `kt=3 PRE_SSM.u/v` **1/1 bits each**.  Only after
that target is reached will the continuous walk resume, stopping at the first
later row above the fixed `1e-15` pointwise bar.

## Rule 0: executing source order

The resolved oracle prints `nn_fsbc=4`, `ln_dynadv_OFF=T`,
`ln_dynadv_vec=F`, `ln_dynldf_OFF=T`, `ln_dynvor_ene=T`, and
`ln_drgice_imp=T` (`ocean.output:540,1181,1315-1319,1325-1338,1125`).
Thus this is **linear momentum with no momentum advection**, not a
vector-invariant advection arm.  The live vorticity selection is ENE applied
to planetary Coriolis only.

At kt2, `sbc_ssm` first resets its accumulator because
`MOD(kt-2,nn_fsbc)=0`, then adds Kbb velocity and Kmm tracer/SSH/thickness
(`sbcssm.F90:128-161`).  It does not form a four-step mean until
`MOD(kt-1,nn_fsbc)=0` (`:164-173`), so SI3 is not stepped again at kt2.

The second ocean step then follows:

1. `stp2d` constructs its Kbb barotropic RHS in HPG, LDF, ENE-Coriolis,
   advection-dispatch, vertical-mean, drag, then wind order
   (`stp2d.F90:118-201`).  The OFF LDF and OFF/linear advection dispatches must
   be inert.  The energy-conserving Coriolis statements form face transports
   and add `r1_4*r1_e1u*(zwz*zy1+zwz*zy2)` / its signed v analogue
   (`dynvor.F90:406-490,518-532`).
2. `dynspg_ts` removes the Kmm two-dimensional Coriolis tendency from the
   slow RHS (`dynspg_ts.F90:352-369`) and reapplies live Coriolis in every
   split-explicit substep (`:671-705`).  Its barotropic drag coefficient is one
   source statement,
   `r1_2*((bot_east+bot)+(top_east+top))`
   (`:1608-1612`).
3. `stp_RK3_stg` stage 1 calls the flux-form `dyn_adv` dispatch only; under
   `ln_dynadv_OFF` it leaves Krhs zero (`stprk3_stg.F90:309-316`).  Stages 2
   and 3 call HPG, ENE-Coriolis, and the OFF advection dispatch in that order
   (`:317-336`).  Stage 3 then calls the OFF LDF dispatch before `dyn_zdf`
   (`:394-430`).  HYB imposes the final barotropic velocity at stages 1 and 2
   (`:137-145,194-212`) and all stages receive the literal depth-mean
   correction at `:433-445`.
4. `dyn_zdf` removes the barotropic mode, builds its implicit diagonal, and
   adds the surface stress.  Its ice-top diagonal is
   `zDt_2*(rCdU_top(east)+rCdU_top)/e3u` at `dynzdf.F90:302` (v at `:478`),
   not two separately halved drag terms.

The frame registry will carry explicit `(kt,kstg,Kbb,Kmm,Krhs,Kaa)` levels.
It will register the kt2 SSM reset/add rows; stp2d RHS after each ordered
operator; the frozen/live barotropic Coriolis and combined drag operands; each
RK3 stage's incoming velocity, post-operator Krhs, raw Kaa, and imposed Kaa;
the stage-3 dyn-ZDF matrix/RHS/solution; and final post-LBC velocity.  Every
header count is derived from its write array and the schema gate must reject a
planted false count.

## Pre-implementation reuse search

Search covered the existing rung-3.6 gate/writers, the shared
`barotropic_substeps_latlon_cgrid`, `nemo_bottom_drag_rate_faces`,
`nemo_top_drag_rate_face_sums`, `_nemo_ws_rk3` stage program, ENE vorticity
operators, and the shared implicit ZDF solver.  These are the only numerical
implementations used.  The round-17 code may extend their private diagnostic
hooks and the existing gate; it will not add a slab solver, Coriolis routine,
drag routine, or second time integrator.

## Ranked hypotheses and discriminators

| rank | hypothesis | confirm | refute |
|---:|---|---|---|
| 1 | The first split is the kt2 barotropic slow-forcing/substep program when its Coriolis and drag first consume nonzero velocity. | Exact entry state; first red stp2d/dynspg operand or statement before RK3 stage 1. | Final barotropic targets and substep operands are bit-identical. |
| 2 | The first split is stage-2 ENE Coriolis on the nonzero stage-1 velocity. | Stage-1 imposed velocity exact; first red `dyn_vor` operand or post-statement row in stage 2. | Stage-2 post-vorticity Krhs is exact. |
| 3 | The first split is the nonzero ice-top contribution in the stage-3 implicit ZDF diagonal/RHS. | Pre-ZDF velocity exact; first red combined top-drag statement, matrix row, or solve output. | Complete post-ZDF velocity is exact. |
| 4 | The SSM reset/add cadence changes an input used at kt2. | First red row occurs in the registered SSM reset/add chain and reaches a live forcing operand. | SSM rows are exact or remain observational because the card consumes the already-certified oracle exchange stream. |
| 5 | OFF advection/LDF is not structurally inert in legoESM. | NEMO post-dispatch equals pre-dispatch while legoESM changes the corresponding RHS. | Both paths preserve structural zero. |

Scaling is measured before assigning an owner.  The first red statement gets
one private scalar/bypass hook: production value 1, causal arm 0.  It must move
the named kt3 row without changing earlier exact rows and cannot appear in a
public config.  A `+1e-8` row-level plant at the same register must exit
nonzero.  If a source-identity fix touches shared code, its operator row must
be exact on NEMO inputs; every runnable cross-card register is rechecked and
any movement is entered in the Rule-12 register rather than reverted.

## Carry-over and Rule-12 preregistration

The current branch already contains the corrected round-13 score
`448,950/448,950`, and the drag docstring already cites
`dynspg_ts.F90:1611-1612` and `dynzdf.F90:302,478`; this round will re-score
them rather than silently claim they were absent.  The shared drag probe will
evaluate both the former separate-half association and the literal combined
statement at the first nonzero top-drag step and report 0/n bit counts.

The OVERFLOW/LOCK construction blocker is preregistered as a configuration
contradiction introduced on this branch: shared `_model_config` still selects
`momentum_advection='flux_form'`/`momentum_flux_scheme='upwind3'`, while the
round-13 slab commit changed its shared `vertical_momentum_scheme` to `off`.
The strict constructor at `ocean_model_latlon_cgrid.py:2435-2440` correctly
rejects that impossible half-OFF `ln_dynadv` program.  The validation will not
be weakened and those card choices will not be changed without a separate user
decision.  The round-15 shared ZDF input-ordering change is registered for
merge verification at `PRE_DYN_ZDF_SOLVE`; GYRE is owner-of-record for the
general tra-ZDF/TKE path.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| fresh kt2 config-local WRITE-only operand run | ASKED | preregistered |
| ranked bisection, private one-variable arm, row-level plant | ASKED | preregistered |
| shared source-identity correction and ordered continuation | ASKED | conditional on the measured owner |
| re-score stale FZP/citations and literal nonzero drag | ASKED | preregistered |
| weaken validation or silently change OVERFLOW/LOCK/GYRE selectors | UNASKED | forbidden; not planned |
| modify shipped NEMO, delete roots, GPU, `mpirun`, runtime artifacts in git, push | UNASKED | forbidden; not planned |

## CONFIRMED / PLAUSIBLE

**CONFIRMED before the new run:** the kt2 bit-identical entry, kt3 stopping
values, resolved selectors and source order, existing carry-over corrections,
and exact cross-card constructor contradiction.  **PLAUSIBLE, UNMEASURED:**
ranked hypotheses 1-5 and any causal owner.  No new operand result or fix is
claimed before the fresh oracle and controls bind.
