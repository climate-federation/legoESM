# SI3 lane 3b round 16 preregistration — coupled tracer owner hunt

Date: 2026-09-04  
Tracker: `climate-federation/legoESM#1699`  
Parent: `5cd6203aaad2c65eba82ec2de8b6ed86bb3e0991`  
State: **PREREGISTERED; UNMEASURED**

## Frozen observation and ordered stop

Round 15 closed the momentum/SSH chain at `kt=2 PRE_SSM` bit for bit.  Its
next row is temperature, with absolute error `2.267231727692831e-5 K` and
normalised error `1.308615514721144e-5`; salinity behind it differs by
`7.624427225110253e-4` (`2.2424454879980685e-5` normalised).  Those values
locate an interval, not an owner.  No downstream trajectory row is used to
choose a fix.

A fresh config-local copy of the accepted scalar-math oracle will write the
`kt=1` active-tracer chain only.  It will not modify the shipped NEMO tree or
the retained R15C run.  Every binary header count is derived from the write
list and checked by the schema gate.  The accepted physical streams,
`ocean.output` numerics, and restarts must remain byte-identical to R15C.

## Rule 0: executing NEMO path and time levels

The resolved run prints `ln_shuman=T`, `ln_traadv_OFF=T`,
`ln_traldf_OFF=T`, `ln_traqsr=T`, `ln_trabbc=F`, `ln_trabbl=F`,
`ln_tradmp=F`, `ln_zad_Aimp=F`, `ln_zdfmfc=F`, `ln_zdfosm=F`, and
`ln_zdfnpc=F`.  Thus calls that dispatch to OFF routines are registered but
must be inert; bottom heat, BBL, damping, adaptive vertical advection, MFC,
OSMOSIS, and NPC do not execute.

For each Wicker--Skamarock stage, `stprk3_stg.F90:112-249` sets `rDt` and
the QCO `r3t` time levels.  `:510-522` clears `Krhs`, calls `tra_adv`, then
`tra_sbc_RK3`.  Stage 1/2 QCO integration is the literal three-statement
content expression at `:549-554`.  Stage 3 adds `tra_qsr`, dispatches the OFF
`tra_ldf`, and calls `tra_zdf` at `:572-599`.

The active surface terms are:

1. stages 1 and 2: `z1_rho0_e3t=r1_rho0/e3t(Kmm)`, followed separately by
   `Krhs_T -= emp*T(Kbb)*z1_rho0_e3t` and
   `Krhs_S -= emp*S(Kbb)*z1_rho0_e3t`
   (`trasbc.F90:282-290`);
2. stage 3: the non-solar heat and ice salt statements
   `Krhs_T += r1_rcp*qns*z1_rho0_e3t` and
   `Krhs_S += sfx*z1_rho0_e3t` (`trasbc.F90:306-313`);
3. the selected RGB-with-data arm, dispatched at `traqsr.F90:172-176`; its
   one-wet-layer heat deposit is the active `key_RK3` assignment in
   `qsr_RGBc` and is already covered at `POST_TRA_QSR` by the round-13 gate.

Finally `tra_zdf` calls `tra_zdf_imp` (`trazdf.F90:88`).  With one wet layer
there is no live interior-diffusion recurrence.  The temperature and salinity
content RHS is nevertheless formed in NEMO statement order from
`e3t(Kbb)*T(Kbb) + rDt*e3t(Kmm)*Krhs` (`trazdf.F90:271-278`) and the terminal
solve divides by the after-level diagonal (`:281-286`).  The resulting Kaa
tracers receive their T-point halo update in `stprk3_stg.F90:625-637` before
the next `sbc_ssm` entry reads them at `kt=2`.

The new frame registry records `(kt,kstg,Kbb,Kmm,Krhs,Kaa)` and, in the order
above, the Kbb tracer, all three `r3t` operands, `e3t(Kbb/Kmm/Kaa)`, zeroed
Krhs, post-advection Krhs, the `z1_rho0_e3t` scalar, post-`tra_sbc` Krhs,
stage-1/2 numerator terms and Kaa result, stage-3 pre/post-RGB Krhs,
`tra_zdf` `zwi/zws/zwd`, content RHS, solve output, and post-LBC/final T/S.

## Reuse search and hypotheses

The pre-implementation search found one shared tracer program,
`_nemo_ws_rk3_tracer_pair_step`, one shared QCO content pair
(`thickness_weighted_tracer_content` / `thickness_weighted_tracer_combine`),
and one NEMO-ordered tracer solve
(`implicit_vertical_diffusion_nemo_tracer_pair`).  It found no slab tracer
solver.  Any accepted correction must extend these shared owners; no new
numerical method or slab fork is permitted.

H16-A predicts that the first split is before `tra_zdf`, in the QCO/surface
content construction: legoESM currently composes its surface and volume
sources outside NEMO's three stage-local source statements.  It is confirmed
only if the NEMO and shared inputs agree bitwise but the first post-statement
operand does not.  It is refuted if the complete stage-3 Krhs is bit-identical.

H16-B predicts that, if stage-3 Krhs is exact, the first split is the
one-layer `tra_zdf` content RHS or terminal division association.  It is
confirmed only if the matrix inputs are exact and one of
`trazdf.F90:271-286` is the first non-bit statement.  It is refuted if the
post-solve tracer is bit-identical.

H16-C predicts that, if the solve output is exact, the split is a time-level
or halo/carry operation after `tra_zdf`.  It is confirmed only by an exact
pre-operation operand and a red post-operation row.

The one-variable causal arm will bypass or scale exactly the first measured
non-bit statement through a private `_NEMOWSRK3TestHooks` field.  Scale 1 is
the production identity; the altered scale must move the named boundary and
must not be constructible from public configuration.  A `+1e-8` row-level
plant for each new register must exit nonzero.  After a source-backed fix,
the target is `kt=2 PRE_SSM.temperature` and `.salinity` **1/1 bits each**.
Only then does the ordered walk continue through `kt=2..8760`, stopping at
the first later over-bar boundary.

## Carried drag-association hypothesis

NEMO forms the barotropic coefficient as
`r1_2*((bot_east+bot)+(top_east+top))` at
`dynspg_ts.F90:1611-1612`; the 3-D top term is
`zDt_2*(top_east+top)/e3u` at `dynzdf.F90:302`.  The current shared helper
halves bottom and top separately.  H16-D predicts a source-literal regrouping
is inert at `kt=1` because the ice top coefficient is zero, hence it cannot
move the already exact `kt=2 PRE_SSM` rows.  At the first later step with
nonzero top drag, the committed probe will report the before/literal bits and
whether the barotropic coefficient changes.  This carry-over is corrected
independently of the tracer owner.

## Cross-card and stopping contract

If the tracer owner is a shared statement association, the retained
OVERFLOW, LOCK_EXCHANGE, and GYRE registers are run before acceptance.  Any
movement invokes Rule 12 and is reported rather than hidden.  No claim extends
beyond the first continuous over-bar boundary.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| fresh config-local `kt=1` tracer operand writer | ASKED | preregistered above |
| source-order bisection, private one-variable arm, shared-path fix | ASKED | conditional on measured owner |
| literal shared drag association and later nonzero-drag measurement | ASKED | preregistered as H16-D |
| continue the year walk after T/S close | ASKED | stop at first later debt |
| edit shipped NEMO, delete retained roots, add slab numerics, GPU, `mpirun`, push | UNASKED | forbidden / not planned |
| select an unrequested tracer, mixing, or forcing arm | UNASKED | not done |

## CONFIRMED / PLAUSIBLE

**CONFIRMED:** the round-15 stopping values; the active namelist selectors;
the source execution order; and the existing shared implementations named by
the reuse search.  **PLAUSIBLE, UNMEASURED:** H16-A through H16-D.  No owner,
fix, or trajectory claim is made before the new stream and causal arm bind.
