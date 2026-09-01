# Lane 1 OVERFLOW-zps stability investigation: preregistration

Status: **frozen before computing any approach-window legoESM metric or running
any causal arm.**  The already-recorded full runs are prior facts: the certified
fp64 card first becomes non-finite after completed step 2,877, the identical
fp32 scheme after step 2,879, and NEMO completes 6,120 steps.

## Claim and prohibitions

The primary finding is already fixed by Rule 9: legoESM has a numerical mismatch
because the oracle stably completes the same configured case.  No damping,
clipping, diffusion, or limiter absent from the executed oracle may be proposed
as a correction.  A diagnostic ablation is not a selectable model and cannot be
shipped as one.

The fp32/fp64 failure separation is only two steps out of about 2,878.  The
registered interpretation is **PLAUSIBLE deterministic structural divergence**,
not roundoff accumulation.  It becomes **CONFIRMED** only if the approach-window
field, location, and normalized growth signature agree through the common
finite prefix and a source-attested operator arm moves the failure coherently.

## Executed oracle path, read before ranking

The resolved OVERFLOW namelist selects `key_qco + key_RK3`, flux-form UP3
momentum, FCT2 tracers, `ln_zad_Aimp=.true.`, constant vertical mixing,
`rn_avm0=1e-4 m2/s`, and `rn_avt0=0`.  It disables Richardson, TKE, GLS,
OSMOSIS, mass-flux convection, enhanced-diffusion convection,
non-penetrative convection, double diffusion, and wave mixing
(`overflow_zps/namelist_cfg:123-140`; executed `ocean.output:630-660`).
`zdfphy.F90:193-213` therefore dispatches “no specific convection scheme” and
the constant closure.  There is no unrepresented NEMO convective stabilizer to
add.

At RK3 stage 3 NEMO computes `ww/wi` twice, once from velocity and once from
transport (`stprk3_stg.F90:281-303`).  The live Wicker--Skamarock partition uses
horizontal Courant inflow, vertical Courant number, the bottom-to-top running
maximum, and thresholds 0.8/1.1 (`sshwzv.F90:696-707,710-873`).  The explicit
part enters FCT and the implicit part enters both the FCT two-step predictor and
the tracer tridiagonal solve (`traadv_fct.F90:141-167,286-330`;
`trazdf.F90:207-225`).

The pre-implementation search found legoESM's only active implementation in
`vertical.py:1716-1909` and its only testcase call in
`ocean_model_latlon_cgrid.py:4996-5054`.  It uses the older local vertical-only
fraction, applies it to momentum after the completed RK3 program, and passes
unpartitioned vertical transport to all tracer FCT stages.  No second faithful
Wicker partition or implicit tracer-advection implementation exists.  Existing
private WS-RK3 hooks are the sanctioned location for causal ablations; no
public micro-selector is introduced.

## Frozen approach-window measurement

The exact matched interval is NEMO step-entry `kt=2601..2878` against legoESM
completed states `2600..2877`; both denote the state before NEMO step `kt`.
NEMO is rerun from the pinned initial configuration only to increase write-only
dump cadence.  Its namelist physics and binary source are unchanged except for
a new `tests/OVERFLOW_OMIP_L1_STABILITY/MY_SRC/stprk3.F90` dump condition.
legoESM runs CPU fp64 with `PrecisionPolicy.fp64()` and records all state and
geometry dtypes.

At every matched state the committed probe records, without `nan*` reductions:

- wet-intersection L-infinity T/S and instantaneous C-grid U/V/SSH differences,
  with both models' staggering, reduction, value, array index, x position, and
  T-level depth at the common argmax;
- each model's wet extrema and the signed margin by which legoESM lies outside
  NEMO's same-step global wet range, both strictly and after a `64 eps` field-
  scale roundoff pad;
- the actual finite/non-finite inventory and the first failing field;
- volume, heat-content proxy `sum(area*h*T)`, salt-content proxy
  `sum(area*h*S)`, and their drift from the initial legoESM value; these closed-
  box budgets classify source versus redistribution before mechanism ranking;
- maximum speed, maximum one-step field increment, its location, and successive
  increment/error ratios.  A decreasing ratio is polynomial accumulation; a
  sustained ratio above one is an amplifying mode.  No fit may replace the raw
  ratios.

The first envelope departure is the earliest matched state whose legoESM wet
range exceeds NEMO's wet range after the fixed `64 eps` pad.  If it predates
`kt=2601`, the same every-step protocol expands backward by powers of two until
the first transition is bracketed; this changes neither metric nor threshold.

Non-vacuity controls plant (1) a finite 50 C wet T cell, which must trip the
envelope and gross-excursion gates, (2) one NaN in each field family, which must
be fatal, and (3) a shifted candidate step number, which must fail time
alignment.  The scorer must also reproduce the certified kt=1 exact state.

## Candidate ranking before arms

1. **Adaptive-implicit tracer and momentum program — source-confirmed mismatch,
   highest rank.**  The active NEMO stage program described above is absent;
   thin partial cells make its Courant trigger directly scale-compatible with a
   slope-front runaway.  This is a mismatch finding, not yet the failure owner.
2. **Earlier unowned OVERFLOW T program — UNMEASURED.**  Roughly 75% of the
   kt=2 temperature residual remained after the registered primary-transport
   arms.  It is real but presently has no late-time scaling link.
3. **Barotropic substep composition on the slope — UNMEASURED.**  Early U/SSH
   debts and the sloping depth field make it plausible, but the corrected
   instantaneous frame removed the prior large artifact and no approach-window
   growth evidence exists yet.
4. **BBL transport — NEMO-side PLAUSIBLE, legoESM-side confirmed live.**  The
   stage-3 Campin--Goosse path is source-attested, but phase 3 did not dump
   NEMO's transport and its kt=2 ablation did not move T materially.
5. **Vertical mixing/convection — source-exonerated as a missing stabilizer.**
   Both cards select the same constant momentum viscosity and zero tracer
   diffusivity; NEMO runs no convection scheme.  A later arithmetic/time-level
   mismatch in the constant solve remains measurable but ranks below the live
   adaptive-advection mismatch.

## One-variable arms and verdict rules

Arm A is the private `disable_tracer_vertical_transport` hook: it sets only the
three stage vertical tracer transports to zero and retains horizontal FCT,
momentum, BBL, geometry, timestep, precision, and all card selectors.  It is an
experimental operator-class ablation, not a reference configuration or public
model.  Before an owner label, the baseline vertical-transport scale at the
eventual hot cell must be at least 0.1 times the observed one-step T increment.
Arm A **supports** vertical tracer transport as necessary if the first non-
finite step moves by at least 100 steps or beyond 3,200 and the pre-failure T
increment falls by at least 2x.  It **refutes it as primary** if failure remains
within steps 2,870--2,884 and both failure-step and increment move under 10%.
Anything else is `PLAUSIBLE`, never `CONFIRMED`.

Only after Arm A and the source-exact scale diagnostic may Arm B replace the
legacy partition with NEMO's unbranched RK3 adaptive-implicit package.  It must
include the Wicker criterion, stage-3 explicit/implicit split, tracer FCT
predictor, tracer tridiagonal, and momentum tridiagonal together because NEMO
exposes no switches among them.  A partial composition is prohibited by the
standing no-Frankenstein rule.  Arm B confirms ownership only if it completes
6,120 finite steps and reduces the matched approach-window growth at the same
location; mere delay is `PLAUSIBLE`.  If Arm A refutes, Arm B is not armed in
this round and the next candidate is ranked from the measured ledger.

## Frozen addendum after Arm A, before the barotropic arm

Arm A did not enter either preregistered terminal corridor: zeroing tracer
vertical transport from initialization failed at completed step 240, 2,637
steps earlier than baseline.  The same-input completed-2,875 scale diagnostic
then found that this hook removes 99.961% of the next T increment at its x=41.5
km, z=970 m maximum but changes U and SSH by exactly zero in that step.  The
early exact-frame score independently fixes the sequence: U and SSH first leave
the roundoff-padded NEMO range at kt=2 and become gross at kt=3; T remains
inside the NEMO wet range through kt=60.  Thus explicit vertical tracer
transport is a terminal T amplifier on an already-divergent velocity/free-
surface state, not evidence that it initiates that state.

The evidence ranking is therefore updated before any second arm:

1. barotropic substep/transport composition on the slope -- **UNMEASURED,
   highest rank**;
2. NEMO's complete adaptive-implicit RK3 package -- **PLAUSIBLE terminal T
   owner, root ownership UNMEASURED**;
3. BBL stage transport -- **NEMO-side PLAUSIBLE**;
4. earlier unowned T residual -- **UNMEASURED**;
5. missing mixing/convection stabilizer -- **SOURCE-EXONERATED**.

Arm C is the existing private `primary_transport_average=False` hook.  It
changes only the `un_adv` primary-transport time average in the barotropic
substep; NEMO always retains that average (`dynspg_ts.F90:509,641,843`), so this
is localization instrumentation, not a selectable correction or reference
configuration.  It supports the primary-transport composition as causal only
if failure moves at least 100 steps later (or beyond 3,200) **and** both U and
SSH matched-window L-infinity at kt=2601 fall by at least 2x.  It refutes that
term as primary if failure remains in 2,870--2,884 and both errors move under
10%.  Earlier failure or mixed movement is `PLAUSIBLE`, never `CONFIRMED`.
No production selector or stabilizer is armed.

## Frozen resolved-coverage addendum before the next arm

The new file-driven gate inventories every one of the **108** resolved keys in
the executed `namdyn*`, `namzdf`, and `namtra*` blocks.  Its registry is exact:
a planted file-side key fails as missing, a removed key fails as stale, and a
card mutation from flux to vector momentum fails before integration.  The
coverage result is 67 VERIFIED, 40 WAIVED inactive-family operands, and one
loud UNMEASURED row: `namzdf.ln_zad_aimp`.  WAIVED means inventory stability
only, not review of a dead algorithm.

This pass refutes the proposed missing-convection mechanism.  The resolved
oracle has `ln_zdfevd=F`, `ln_zdfnpc=F`, and all other specific convection
closures off; `zdfphy.F90:181-197` therefore prints “no specific scheme used.”
It selects only constant vertical mixing with `rn_avm0=1e-4` and `rn_avt0=0`
(`zdfphy.F90:151-179,207-215`).  Momentum is flux-form UP3
(`dynadv.F90:78-90`) and lateral momentum diffusion returns without allocating
viscosity arrays (`ldfdyn.F90:220-239`).  Enabling EVD, convection, or an LDF
would consequently be a Rule-9 stabilizer that the reference does not run; no
such arm is permitted.

The per-case barotropic composition is VERIFIED, not inherited from a lane
default.  The resolved values are `ln_bt_fw=T`, `ln_bt_auto=T`,
`nn_bt_flt=1`, `rn_bt_alpha=0`, and the executed Courant calculation resolves
`nn_e=3` (`dynspg_ts.F90:1200-1240`; matched `ocean.output:848-878`).  For
three substeps, NEMO's exact filter rule (`dynspg_ts.F90:1041-1108`) gives
primary weights `[0,1,1,1]/3`, raw secondary weights `[3,3,2,1]`, divisor 9,
and four loop iterations.  The certified card produces those arrays exactly.
The `rn_bt_alpha=0` back interpolation uses the live AB3-AM4 coefficients
`0.614/0.285/0.088/0.013` (`dynspg_ts.F90:1676-1711`).  No barotropic
selector mismatch is exposed by this coverage round.

The remaining active gap is narrower and source-bound.  NEMO partitions
`ww/wi` at RK3 stage 3 (`traadv.F90:220-227`); with the resolved optimized
`nn_fct_imp=1`, the implicit transport is removed from both halves of FCT's
two-step low-order predictor (`traadv_fct.F90:140-145,526-536`) while the
explicit transport remains in the flux.  legoESM instead applies its older
local adaptive momentum solve after the complete stage program and supplies
unpartitioned vertical transport to tracer FCT.  The boolean therefore remains
UNMEASURED until its arithmetic and time levels are compared; it is not marked
VERIFIED merely because both configurations spell “adaptive implicit.”

### Arm D: current legoESM adaptive-momentum localization

Arm D is a private harness ablation, not a public selector or reference model:
it suppresses only legoESM's current post-program adaptive vertical-momentum
rewrite while retaining the certified card, tracer transport, FCT, BBL,
barotropic solver, mixing, geometry, timestep, and fp64 policy.  Before a full
arm, a same-input completed-step-2,875 measurement must show that the arm's
one-step U effect is at least 0.1 times the baseline one-step U increment at
the baseline increment maximum; otherwise the term is scale-incompatible and
the full arm is not run.  If scale-compatible, the current implementation is
**CONFIRMED** as an instability owner only if the arm completes 6,120 finite
steps and reduces both the matched U growth and the terminal T growth at their
registered slope-front loci.  A delay of at least 100 steps with both growth
measures reduced by at least 2x is **PLAUSIBLE**.  Failure within 2,870--2,884
and movement below 10% is **REFUTED_PRIMARY**.  All other outcomes remain
UNMEASURED.  Even a confirmation localizes wrong legoESM arithmetic; it does
not authorize shipping the no-adaptive arm, because NEMO has no such switch.
