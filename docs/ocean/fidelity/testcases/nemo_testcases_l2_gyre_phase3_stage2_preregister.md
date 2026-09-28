# NEMO testcase lane 2 GYRE — stage-2 boundary preregistration

Date: 2026-09-03  
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`  
Reconciled baseline: `57429ecf5f377ce2bf220f36bc05313cd29e0dfd`  
Bottom-drag preregistration commit: `239511e3216fe69f905e6b1a40156cdc6421d013`

This freezes the next ordered measurement after the bottom-drag landing.  It
is derived again from the NEMO source and the reconciled implementation; no
round-7 WIP receipt or its invalid preregistration SHA is evidence here.

## Re-pinned boundary after the drag landing

The CPU/fp64 production run clears every one of the 800 registered
barotropic-substep rows and clears stage 1 (`u` and `v` each
`2.710505431213761e-19 m/s`).  The first remaining internal momentum debt is
therefore stage 2: `u=1.057858031242796e-5 m/s` and
`v=1.0572800989805292e-5 m/s`.  Dividing by the stage interval
`rn_Dt/2=7200 s` gives a residual tendency scale of about `1.47e-9 m/s2`.
The whole-step first debt remains kt=2; SSH alone now clears kt=2 at
`6.5052130349130266e-18 m`.

## Source order and first operands

GYRE uses the module-default hybrid external-mode update (`n_baro_upd=np_HYB`;
`stprk3_stg.F90:40-44`).  Stage 2 sets `Kbb=N`, `Kmm=N+1/3`,
`Kaa=N+1/2`, and `rDt=rn_Dt/2` (`:174-215`).  It then constructs the Kmm
transport (`:257-304`), and accumulates the momentum RHS in the fixed order
EOS/HPG, vorticity, advection (`:317-336`) before restarting the velocity from
Kbb and applying the stage interval (`:360-388`).  Finally it replaces the
reference-thickness depth mean by the hybrid barotropic Kaa target
(`:433-446`).

The critical interleave is source-mandated: after each momentum Kaa update,
NEMO advances tracers on the same Kmm transport (`stprk3_stg.F90:452-521`).
Stages 1 and 2 start their tracer RHS at zero, apply advection and
`tra_sbc_RK3`, and restart from Kbb with the QCO weights (`:508-565`).  For a
nonlinear free surface, `tra_sbc_RK3` adds the water-exchange heat and salt
terms at stages 1 and 2 using the Kbb surface tracer and Kmm top-cell
thickness (`trasbc.F90:224-292`).  The stage-2 EOS/HPG therefore consumes the
stage-1 T/S/SSH bundle, not a full post-physics Euler tracer state.

The reconciled legoESM identity already interleaves tracer and momentum
stages, but `_stage_tracers` currently seeds its helper with
`state_new.T/S` (the full pre-applied physics Euler state) and supplies no
stage-local `tra_sbc_RK3` source.  GYRE has live TKE/EVD, lateral tracer
diffusion, penetrative shortwave, and seasonal EMP, so the old comment that
this ordering is inert does not apply to this case.

## Frozen measurements and decisions

The gate will first score, in source order:

1. stage-2 Kbb velocity, stage-1 Kmm velocity, and hybrid barotropic target;
2. the stage-1 T, S, and SSH values actually handed to stage-2 EOS/HPG;
3. the NEMO source-ordered stage-2 HPG, vorticity, and advection increments;
4. the corrected stage-2 Kaa velocity.

The first over-bar row in that order owns the next boundary.  Raw Kaa and its
barotropic correction are gauge-dependent because legoESM removes the depth
mean before integration while NEMO removes it afterward; only the
baroclinic RHS and corrected sum are owner-capable.

A private one-variable arm may replace only the stage-2 EOS/HPG
thermodynamic bundle with the oracle stage-1 T/S/SSH.  It is
`CONFIRMED_CAUSAL_OWNER` only if the direct HPG rows and corrected stage-2
rows all clear `1e-15`.  Movement at least 0.9 of the faithful stage-2
residual without clearance is `CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER`; movement
below 0.1 residual is `NEAR_NULL_NO_DISCRIMINATING_POWER`.  Scaling is printed
before any label.  If this bundle is causal, the production fix must fold the
source-exact tracer interleave into the single WS-RK3 identity, not expose a
public mix-and-match switch.

## Controls and honesty

The stage-2 operand and term dumps are accepted only with their exact headers,
payload sizes, fp64 dtype, finite owned values, and WRITE-only MY_SRC
placement.  Their uninstrumented state output must retain its pinned SHA256.
A planted nonzero stage-2 HPG/RHS violation must become DEBT at the registered
row.  Internal tracer-stage agreement and all downstream owner labels remain
UNMEASURED until these checks actually run.  No external adversarial review
has occurred for this round.

## Vector-invariant stage-update continuation

The valid post-`tra_adv_trp` stage-3 transport arm traces the complete
advection-content residual upstream to the stage-2 momentum state.  Every
source-ordered stage-2 tendency operand is already AT-BAR, so the next
boundary is the literal stage update.  NEMO selects the velocity branch when
`ln_dynadv_vec=.true.`: `stprk3_stg.F90:365-369` computes
`Kaa = (Kbb + rDt*Krhs)*mask`.  The candidate instead applies the QCO
thickness weights from the flux-form branch at `:370-386` even though that
branch is dead for resolved GYRE.

The faithful correction will select the literal velocity recurrence for
stages 1 and 2 inside the existing WS-RK3 identity.  A private one-variable
legacy arm will restore the old QCO-weighted recurrence.  **CONFIRM** iff the
faithful corrected stage-2 velocity and post-`tra_adv_trp` transport clear
`1e-15`, while the legacy arm reproduces the former stage-2/transport debt at
its measured scale.  Otherwise report the surviving operand without an owner
label.  Scaling precedes ownership, and no public Frankenstein switch is
permitted.

### Raw-RHS gauge continuation

The source-correct vector recurrence changes the post-`tra_adv_trp` transport
by only `O(1e-10)` absolute against its `O(1e-5)` horizontal residual, so it
is retained as a fidelity correction but is **NEAR-NULL / NOT AN OWNER** of
the open boundary.  The committed source-term gates already show HPG,
vorticity, and advection are AT-BAR after removing their reference-depth
means, while their raw depth-uniform gauge differs.  Because NEMO forms raw
`Kaa` from the unprojected `Krhs` and only then replaces the mean
(`stprk3_stg.F90:365-369,433-446`), that gauge can survive through fp64
association even though it cancels algebraically.

A WRITE-only candidate seam will expose the complete stage-2 `Krhs`, and a
one-variable arm will replace only it with the existing oracle
`after_advection` arrays from `oracle_rkstage2_terms_kt00000001.bin`.  The gate
will score raw RHS, raw Kaa, corrected Kaa, and the post-`tra_adv_trp`
transport in that order.  **CONFIRM raw-RHS gauge ownership** only if the
oracle-RHS arm clears the downstream stage-2 and transport boundary with
movement at the faithful residual scale; otherwise the first surviving row
remains the boundary.  Dtype, header, registry, and planted controls remain
fail-closed.

### ENE Kmm face-thickness continuation

The raw-RHS arm clears corrected stage-2 u/v to `3.25e-19`/`2.71e-19` and
improves the downstream triplet by about two million-fold, confirming the raw
stage-2 RHS as causal.  Its first direct divergence is the raw vorticity
component (`5.03e-10` u, `5.93e-10` v); the reference-depth projection is
AT-BAR, which explains the much smaller corrected-state residual.

Source inspection identifies the next operand: NEMO `vor_ene` forms face mass
fluxes with `e3u/e3v(Kmm)` (`dynvor.F90`, ENE transport products), whereas the
candidate's vector branch did not consume the stage's canonical QCO face
thickness supplied to the momentum kernel.  The faithful arm will thread that
already-built `e3u_0*(1+r3u(Kmm))` / v-face pair into the single ENE routine;
a private legacy arm will restore the min-of-stretched-T face pair.  **CONFIRM**
iff raw stage-2 RHS and corrected Kaa clear `1e-15`, and the legacy arm restores
the prior residual at its scale.  Otherwise no ENE-thickness owner label is
allowed.  This is folded into the existing vector/ENE WS-RK3 identity and the
isomorphism register, not exposed as a public selector.

### ENE F-point `e3f_vor(Kmm)` continuation

The Kmm U/V face-thickness correction is near-null at the raw-RHS boundary.
The next ENE operand is the F-point divisor.  Under `key_qco`, NEMO uses
`e3f_vor = e3f_0vor*(1+r3f*fe3mask)`
(`DOM/domzgr_substitute.h90:125-130`), and the RK3 `r3f` is the explicitly
parenthesized four-cell surface-weighted SSH average
(`DOM/domqco.F90:233-246`; stage-2 interpolation at
`stprk3_stg.F90:202-203`).  GYRE's uniform reference geometry makes this the
existing canonical `een_e3f_scheme="nemo_avg"`; the card currently inherits
the generic vertex minimum.

The one-variable arm selects `nemo_avg` without changing ENE, velocity, or
face transports.  **CONFIRM** iff raw stage-2 RHS and corrected Kaa clear
`1e-15`, the inherited-min arm restores the old `5.03e-10`/`5.93e-10` raw
residual at its scale, and the downstream post-`tra_adv_trp` triplet moves at
that scale.  Otherwise the selector is not assigned ownership.  This uses the
pre-existing canonical option; no second ENE implementation is permitted.

### ENE coastal q and `nn_e3f_typ=0` continuation

The four-cell `/sum(tmask)` selector is near-null and is not the owner.  The
resolved oracle output pins `ln_dynvor_msk=.false.` and `nn_e3f_typ=0`
(`ocean.output:813,815`).  NEMO therefore leaves coastal relative vorticity
live (`dynvor.F90:469-477`) and constructs reference `e3f_0vor` from the
masked four-cell sum divided by four, not by the wet count
(`dynvor.F90`, `nn_e3f_typ` case 0).  The candidate ENE routine currently
Neumann-fills q unconditionally, and its available `nemo_avg` selector is the
case-1 `/sum(tmask)` sibling.  The raw-RHS maxima lie on the first wet row or
column, making this the next source-ordered discriminator.

The faithful arm will (1) reuse the existing `nemo_live` coastal-q behavior
inside the one ENE implementation and (2) add the missing literal
`nemo_avg4` F-thickness selector beside `nemo_avg`, then select both as an
inseparable resolved-GYRE ENE identity.  A private legacy arm restores
Neumann fill plus the prior vertex rule.  **CONFIRM** only if raw stage-2 RHS,
corrected Kaa, and downstream transport clear the bar and the legacy arm
restores the former residual at scale.  Synthetic coastal-q and `/4` versus
`/sum(tmask)` violation tests are required; the isomorphism tripwire must keep
one `pv_flux_ene` implementation.

### Direct `vor_ene` operand stream

Neither coastal-q nor the available F-thickness selector moves the raw ENE
residual at a discriminating scale.  Before another implementation change, a
config-local WRITE-only `dynvor.F90` extension will record, for the stage-2
`Kmm=3` call only, each level's post-division `zwz=(f+curl)/e3f_vor` and the
already-materialized `zwx=e2u*e3u*u`, `zwy=e1v*e3v*v` operands
(`dynvor.F90:469-490,518-522`).  It will not alter shipped NEMO source.

The candidate gate will obtain the same intermediates from the sole
`pv_flux_ene` implementation through a WRITE-only return option.  Header,
dimensions, fp64, finite owned values, central registry, and a planted q
violation must pass; ordinary stage, kt=2, and restart hashes must remain
bit-identical.  The first operand over `1e-15` owns the next boundary.  Only
after that direct comparison may a one-variable arm be constructed, with
movement printed relative to the faithful residual before any label.

### Post-ENE HPG operand continuation

The native `ff_f` landing makes the stage-2 ENE and raw advection increments
AT-BAR; the surviving raw-RHS maxima, `9.69e-17` U and `1.10e-16` V, are now
identical to the HPG increment errors and amplify through `rDt/2` into the
stage-2 Kaa/transport debt. Before changing HPG arithmetic, a config-local
WRITE-only `dynhpg.F90` record will dump the executing stage-2 `rhd`,
`e3w(Kmm)`, and `gdept_z0(Kmm)` operands. Existing stage, kt=2, and restart
hashes must remain bit-identical, and a planted nonzero `rhd` violation must
fail closed.

The gate will compare those operands before the source recurrence. If they
are AT-BAR, the next arm may select a literal `hpg_sco` recurrence in the
collapsed GYRE identity; it is causal only if HPG becomes exact enough that
the stage-2 Kaa clears `1e-15`, while a private algebraic-reassociation
ablation restores at least 99% of the former Kaa residual. Otherwise the
first differing operand remains the owner and no recurrence change is
authorized.
