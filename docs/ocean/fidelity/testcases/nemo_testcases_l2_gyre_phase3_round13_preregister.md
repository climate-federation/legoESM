# NEMO testcase lane 2 GYRE — phase-3 round-13 preregistration

Date: 2026-09-04
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`
Reviewed baseline: `4f10d623923453f1bf985cced9693c5d5eb86167`

## Frozen boundary and predicted owner

Round 12 made the shared TEOS-10 statement chain bit-exact for identical
inputs.  On the live production-JIT path, however, the stage-2-entry T/S
operands still differ from NEMO by one ulp at nine wet cells.  Two resulting
`prd` cells differ by `2.220446049250313e-16`, and the stage-2 corrected Kaa
remains DEBT near `2.0e-14`.  This round walks the propagated stage-1 tracer
state and does not enter the stage-3 transport or ZDF boundaries unless that
state and Kaa clear.

The user-supplied candidate ranking is retained, then resolved against the
executed source before measurement:

1. **Transcendentals.**  This is the first prior candidate, but source search
   predicts it is dead at stage 1.  The resolved deck enables two-band QSR
   (`EXP00/namelist_cfg:73,76-79`), yet `stprk3_stg.F90:556-600` calls
   `tra_qsr`, `tra_ldf`, and `tra_zdf` only at stage 3.  The live stage-1
   CEN2 and nonlinear-SBC statements contain no EXP/TANH/SQRT/LOG
   (`traadv_cen.F90:137-149,196-216`; `trasbc.F90:282-292`).  CONFIRM a
   transcendental owner only if the oracle record exposes a live call missed
   by this static trace and NEMO's dumped result matches glibc/NumPy but not
   production-JIT JAX for the exact same argument.  If confirmed, record all
   four values and STOP without implementing a remedy.
2. **FCT limiter ties.**  Also predicted dead at stage 1:
   `traadv.F90:280-283` forces `ll_dofct=.FALSE.` for RK stage 1, and the
   resolved `ln_traadv_fct=.true., nn_fct_h=nn_fct_v=2`
   (`EXP00/namelist_cfg:129-133`) therefore dispatches the plain CEN2 path at
   `traadv.F90:359-365`.  CONFIRM only if a MIN/MAX/SIGN limiter value is
   nevertheless present in the executed stage-1 call; otherwise label this
   source-REFUTED, not exonerated for stage 3.
3. **CEN2 arithmetic and RHS accumulation.**  This is the first live
   prediction.  NEMO forms metric face transports, differences their rounded
   products, multiplies by `r1_e1e2t`, and divides by `e3t(Kmm)` before storing
   `ts(Krhs)` (`traadv_cen.F90:137-149,196-216`).  CONFIRM if the zeroed RHS is
   bit-identical but post-`tra_adv` is the first one-ulp departure at any of
   the nine cells, and a one-variable literal-association arm accounts for at
   least 0.9 of the final stage-state move.  REFUTE if post-`tra_adv` is
   bit-identical at all nine cells.
4. **Nonlinear-SBC accumulation.**  At stages 1–2 GYRE adds only EMP-carried
   surface T/S, with `z1_rho0_e3t=r1_rho0/e3t(Kmm)` and source-order subtraction
   into Krhs (`trasbc.F90:282-292`); runoff is off.  CONFIRM if post-advection
   matches and post-`tra_sbc_RK3` is the first departure, at surface cells,
   with the isolated association arm moving the final result at residual
   scale.  REFUTE if the SBC checkpoint is bit-identical at every listed cell.
5. **QCO tracer-update association.**  Stage 1 uses `rDt=rn_Dt/3=4800 s`
   (`stprk3_stg.F90:118-123`) and evaluates
   `((1+r3t(Kbb))*T(Kbb) + rDt*(1+r3t(Kmm))*Krhs*tmask) /
   (1+r3t(Kaa))` in that association (`:540-554`).  CONFIRM if both live RHS
   checkpoints match and the post-update Kaa is the first departure, and if
   reproducing only that statement makes the nine cells bit-identical without
   worsening another wet cell.  This outranks halo handling when every listed
   cell lies outside the halo.
6. **Boundary/halo handling or compiled contraction.**  The stage update is
   followed by tracer `lbc_lnk` at `stprk3_stg.F90:622-638`.  A halo owner
   requires a first departure confined to cells changed at that call.  A
   compiled-contraction owner requires the same-input NumPy transcription to
   match NEMO while the production-JIT statement does not; any fix must use
   the existing shared IEEE-identity mechanism and remain card-independent.

The initial `kt=1` entry must be bit-identical.  A difference already present
there invalidates this preregistration and stops the stage-1 attribution.

## Source-ordered record and comparison

Search before implementation found the existing config-local WRITE-only
`MY_SRC/stprk3_stg.F90` stream
`oracle_rktracer_operands_kt00000001_s1.bin`.  It already records the zeroed
T/S Krhs, the live stage transport, Krhs after `tra_adv`, Krhs after
`tra_sbc_RK3`, Kbb/Kmm/Kaa T/S, and Kbb/Kmm/Kaa `r3t`; it is therefore extended
only if validation proves a missing boundary.  No shipped NEMO source is
modified.  The legoESM side will extend the existing private
`_NEMOWSRK3TestHooks`/tracer helper and existing Round-11/12 diagnostic surface,
not add an operator or public selector.

The accepted record must be a newly rerun stage 1 of kt=1, non-tiled, fp64,
with the registered Kbb/Kmm/Krhs/Kaa indices, finite arrays, and ordinary
stage/restart hashes bit-identical to the Round-12 control.  Comparisons use
the production-JIT step and report at each of the nine cells the NEMO and
legoesm uint64 bit patterns, ulp distance, surface/bottom/coast relation, and
the first source-ordered boundary that differs.  A planted one-bit wet-cell
mutation must exit nonzero.

If association/order/contraction is confirmed, the one shared WS-RK3 path is
fixed, a failing-first behavioural test is retained, and GYRE stage-2 Kaa,
kt=1…10, plus OVERFLOW/LOCK_EXCHANGE oracle-relative stage/trajectory tables
are rerun.  Any cross-card row moving away from its oracle by more than two ulp
or crossing the bar downward is reported without tuning.  The compare-to gate
and criterion remain unchanged.  No external review of this preregistration
has occurred.

## Registered redirect after the tracer-routine checkpoints

The accepted production-JIT checkpoint run finds nine differing T cells and
nine differing S cells; the coordinate sets are disjoint (18 unique wet
locations), correcting the briefing's ambiguous “nine wet cells” wording.
All 18 first differ after `tra_adv`.  Replacing only stage 1's complete
`zFu/zFv/zFw` operand bundle with the oracle bundle moves the final T and S
residuals by exactly 1.0× and makes both stage updates bit-identical.  This
confirms the transport *bundle* as causal but does not yet identify its first
bad operand.

Before changing arithmetic, a second config-local WRITE-only record will dump
stage-1 `e2u/e1v`, `e3u/e3v(Kmm)`, `uu/vv(Kmm)`, `zub/zvb`, `umask/vmask`, and
the stored `zFu/zFv` immediately after `stprk3_stg.F90:257-278`.  The ranked
prediction is: all inputs match, while the constructed metric transport first
differs because legoESM presently stores `e3*(u+zub)` and later rematerializes
`e2*transport`, whereas NEMO stores the source-associated
`(e2*e3)*(u+zub*mask)` once.  CONFIRM only if a shared native-metric transport
materialization makes `zFu/zFv`, the stage-1 T/S update, and stage-2 Kaa clear
with residual-scale movement.  REFUTE if any dumped input is first non-bit-
exact or native materialization moves less than 0.1 of the tracer residual.
The previously measured stored-barotropic-mean arm is retained as a separate
near-null alternative, not silently folded into this association test.

## Registered redirect after the horizontal-operand split

The accepted split refutes the “all inputs match” part of that prediction.
The horizontal metric, Kmm face thickness, Kmm velocity, and mask are each
bit-exact on every live face.  NEMO's literal final product is also reproduced
bit-exactly by production JIT when it receives NEMO's operands; the former
regrouping differs by at most two ulp (5,937 U and 5,870 V cells).  But the
corrected velocity is already the first non-bit-exact operand: maximum
`2.799952110443815e-17` U and `1.2766480581016815e-17` V, propagating to
`8.922143024392426e-10`/`4.069988790433854e-10` in native zF.

The next record therefore splits `zub = un_adv*r1_hu(Kmm)-uu_b(Kmm)` and its
V counterpart at `stprk3_stg.F90:265-278`.  The preregistered ranking is:
(1) external-mode `un_adv/vn_adv`, because this is the only time-integrated
input and the reconciled kt=2 register still carries barotropic debt;
(2) `r1_hu/r1_hv(Kmm)` association; (3) stored `uu_b/vv_b(Kmm)`.  The first
non-bit-exact input is the owner of the corrected-velocity boundary.  The
native-product change remains a separately confirmed literal-association fix,
but it will not be credited with clearing the tracer boundary unless the
post-fix causal gate actually clears.
