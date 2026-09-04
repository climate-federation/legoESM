# NEMO testcase lane 2 GYRE — Phase 3 rounds 8–13 boundary receipt

**Verdict: STOP / DEBT at the external transport-average input to stage-1
tracer advection.  Round 13 finds the first propagated tracer departure after
`tra_adv`, confirms the complete stage-1 transport as causal, and then finds
`un_adv/vn_adv` already non-bit-exact before `zFu/zFv` are formed.  The shared
literal metric-product association is corrected, but does not clear the
boundary: stage-2 corrected Kaa remains `2.00e-14`, above the immutable
`1e-15` bar.  The pending D-gate criterion remains an open user decision and
is unchanged.  The stage-3 transport and ZDF walks are not entered.**

Date: 2026-09-04

Round-12 implementation commit: `afd6a2082047bbafcddf9ede4acf1ae44d0ec028`

Round-12 preregistration commit: `9ff84260941160940954660c0ae5d347e34506c3`

Round-12 starting tip: `bef8033900877e1e1eefb704a69bc05080c21064`

Round-11 implementation commit: `f446d1820`

Round-11 starting tip: `d124c77c94c9307cec8a74b564b0ad2c0829b990`

Round-10 implementation commit: `7a38ef6e4efd7bd94b1743cae1525c59802f7511`

Original reconciled tip: `57429ecf5f377ce2bf220f36bc05313cd29e0dfd`

Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`

Round-13 starting tip: `4f10d623923453f1bf985cced9693c5d5eb86167`

Round-13 preregistration/redirect commits: `8988ef294`, `c26b49cce`,
`fc26db757`

Round-13 literal-product implementation commit: `09bef32cb`

## Round-13 propagated tracer input walk

### Executed stage-1 sequence and cell census

The kt=1 entry is bit-exact for both T and S.  The executing RK3 source zeros
Krhs at `stprk3_stg.F90:510-513`, calls
`tra_adv(kstp,Kbb,Kmm,Kaa,ts,Krhs,zFu,zFv,zFw,kstg)` at `:519`, then calls
`tra_sbc_RK3` at `:521`.  With `rDt=rn_Dt/3=4800 s` (`:118-123`), the QCO
update at `:540-554` evaluates
`((1+r3t(Kbb))*T(Kbb)+rDt*(1+r3t(Kmm))*Krhs*tmask)/(1+r3t(Kaa))`.
`tra_qsr`, `tra_ldf`, `tra_bbc`, and `tra_zdf` are stage-3-only at `:556-600`;
`tra_atf` belongs to MLF, not `stprk3_stg`.  FCT is also absent here:
`traadv.F90:280-283` disables it before stage 3 and `:359-365` dispatches the
resolved `nn_fct_h=nn_fct_v=2` to CEN2.  Thus the preregistered transcendental
and limiter candidates are **REFUTED_BY_EXECUTED_SOURCE**, not numerically
exonerated for stage 3.

The briefing's “nine wet cells” resolves to nine T cells and nine S cells at
disjoint coordinates: 18 locations total.  None differs at the kt=1 entry;
all first differ after `tra_adv`.  Three T cells are at the surface, no cell is
a bottom cell, and five locations are coast-adjacent.  These are the baseline
production-JIT fp64 bits before the round-13 product correction:

| NEMO (i,j,k) | field | oracle bits | legoESM bits | surface | bottom | coast-adjacent |
|---|---|---|---|---:|---:|---:|
| (18,4,1) | T | `0x403776004ca041ef` | `0x403776004ca041f0` | yes | no | yes |
| (4,7,4) | S | `0x4042683fb2c45753` | `0x4042683fb2c45752` | no | no | yes |
| (20,8,1) | T | `0x40377600509b85aa` | `0x40377600509b85a9` | yes | no | no |
| (12,12,12) | S | `0x4042450cc12a23ae` | `0x4042450cc12a23ad` | no | no | no |
| (23,14,24) | S | `0x40418f655f17d9c8` | `0x40418f655f17d9c9` | no | no | no |
| (33,14,11) | T | `0x40333ba91a6880dc` | `0x40333ba91a6880dd` | no | no | yes |
| (22,15,15) | S | `0x4042021ca033fa34` | `0x4042021ca033fa35` | no | no | no |
| (13,16,4) | S | `0x4042683fb2ba947d` | `0x4042683fb2ba947c` | no | no | no |
| (12,18,3) | T | `0x403702145123a02c` | `0x403702145123a02b` | no | no | no |
| (14,18,16) | T | `0x402ad90432576abd` | `0x402ad90432576abc` | no | no | no |
| (32,18,16) | S | `0x4041df371eeb6530` | `0x4041df371eeb6531` | no | no | no |
| (5,19,12) | S | `0x4042450cc12a23ab` | `0x4042450cc12a23aa` | no | no | no |
| (7,19,20) | S | `0x404193150212bac8` | `0x404193150212bac7` | no | no | no |
| (12,22,1) | T | `0x4037760058e7171d` | `0x4037760058e7171c` | yes | no | no |
| (13,22,17) | S | `0x4041bd6716945702` | `0x4041bd6716945701` | no | no | no |
| (31,22,5) | T | `0x403675fc485de71f` | `0x403675fc485de720` | no | no | no |
| (33,22,14) | T | `0x403056d60958a27e` | `0x403056d60958a27f` | no | no | yes |
| (32,23,17) | T | `0x4026771c23458c03` | `0x4026771c23458c04` | no | no | yes |

At all wet cells, the post-advection maximum tendency gaps are
`1.7835203078773822e-20` T and `2.812308100164311e-20` S.  Substituting only
the oracle `zFu/zFv/zFw` moves each final stage-state residual by exactly 1.0×
and makes both T and S bit-identical.  That is a **CONFIRMED** complete-
transport causal arm.  The stored-barotropic-mean arm leaves the T residual
unchanged and merely relocates an S mismatch, so it is **REFUTED as sole
owner**.

### Operand split and owner

The config-local WRITE-only record is taken immediately after NEMO forms
`zub/zvb` and `zFu/zFv` at `stprk3_stg.F90:265-280`.  On every live face:

| operand/output | U max abs / differing | V max abs / differing | label |
|---|---:|---:|---|
| horizontal metric | `0 / 0` | `0 / 0` | bit-exact |
| Kmm face thickness | `0 / 0` | `0 / 0` | bit-exact |
| Kmm velocity | `0 / 0` | `0 / 0` | bit-exact |
| mask | `0 / 0` | `0 / 0` | bit-exact |
| corrected velocity | `2.799952110443815e-17 / 17400` | `1.2766480581016815e-17 / 17100` | first non-bit-exact compound operand |
| `un_adv` / `vn_adv` | `1.2045919817182948e-13 / 580` | `5.495603971894525e-14 / 570` | **CONFIRMED upstream owner** |
| native `zF` | `8.922143024392426e-10 / 17400` | `4.069988790433854e-10 / 17092` | propagated DEBT |

With the exact NEMO operands, NumPy and production JIT both reproduce
`(e2u*e3u)*(uu+zub*umask)` and its V counterpart bit-for-bit.  The prior
`e2u*(e3u*corrected_u)` regrouping differs on 5,937 U and 5,870 V cells, by at
most two ulp.  The shared S-21 identity now retains the native metric-bearing
product and feeds it directly to CEN2 and WZV.  Its behavioural test failed
first with `AttributeError` because the source-order materializer was absent;
after landing it passes and proves the old regrouping differs on its awkward
fp64 input.  The operand gate's planted one-bit live-U mutation exits `1`.

This literal correction is not over-credited.  On the actual post-fix path,
upstream `un_adv/vn_adv` remains the first bad input.  The final stage-1 update
has nine one-ulp T cells and ten one-ulp S cells; the added S location is
NEMO `(7,21,1)`, oracle bits `0x40426b4578b1f897`, candidate bits
`0x40426b4578b1f896`.  Corrected stage-2 Kaa is unchanged at
`2.0033670902752654e-14` U and `2.0003665607629117e-14` V.  Hence the owner is
**CONFIRMED_EXTERNAL_TRANSPORT_AVERAGE_INPUT**, while its first divergence
inside the external-mode integration is **UNMEASURED** this round.

### Re-pinned trajectory and Rule-8 disclosure

The post-fix production-JIT sweep keeps kt=1 exact (u/v/SSH remain the same
at-rest **UNINFORMATIVE** rows).  At kt=2 the normalized maxima are T
`1.3614736849003888e-12`, S `2.2181101297999213e-14`, u
`9.484089954572578e-7`, v `8.987992592336029e-7`, and SSH
`2.3724511938327808e-15`; first-over-bar remains kt=2.  At kt=10 they are T
`5.636638587568748e-3`, S `1.5483121735304233e-4`, u
`5.624987258870561e-2`, v `1.1173366024871394e-2`, and SSH
`2.1255475816557047e-4`.

Against the Round-12 sweep there are 27 identical, 6 improved, and 17
worsened rows.  Every worsened row is enumerated here; ratios are current /
Round-12 normalized maxima:

| field | kt | Round 12 | Round 13 | ratio |
|---|---:|---:|---:|---:|
| u | 4 | `1.4182404011403506e-2` | `1.418240401140368e-2` | `1.0000000000000122` |
| v | 4 | `1.594784253330965e-2` | `1.5947842533309664e-2` | `1.0000000000000009` |
| u | 5 | `2.073389053895669e-2` | `2.0733890538956697e-2` | `1.0000000000000004` |
| SSH | 5 | `6.701856627955131e-5` | `6.701856628047159e-5` | `1.0000000000137317` |
| u | 6 | `3.113473033866225e-2` | `3.11347303386626e-2` | `1.0000000000000113` |
| v | 6 | `6.0219593254002834e-2` | `6.021959325400314e-2` | `1.000000000000005` |
| S | 7 | `1.224404428342701e-4` | `1.2244044283754916e-4` | `1.0000000000267808` |
| u | 7 | `4.004140560219323e-2` | `4.004140560220132e-2` | `1.000000000000202` |
| v | 7 | `6.631577474100295e-2` | `6.631577474102054e-2` | `1.0000000000002653` |
| u | 8 | `4.71800603612995e-2` | `4.7180060361312434e-2` | `1.0000000000002742` |
| v | 8 | `2.452716279874642e-2` | `2.452716279882225e-2` | `1.0000000000030915` |
| SSH | 8 | `2.510825763594551e-4` | `2.510825764118834e-4` | `1.0000000002088092` |
| u | 9 | `5.256039963431678e-2` | `5.2560399634324524e-2` | `1.0000000000001474` |
| SSH | 9 | `2.1202997652220953e-4` | `2.1202997711175444e-4` | `1.000000002780479` |
| T | 10 | `5.636638587568597e-3` | `5.636638587568748e-3` | `1.0000000000000269` |
| u | 10 | `5.624987258870016e-2` | `5.624987258870561e-2` | `1.000000000000097` |
| SSH | 10 | `2.1255475784966258e-4` | `2.1255475816557047e-4` | `1.0000000014862425` |

The lane-1 oracle-relative compatibility runs have zero changed normalized
rows versus Round 12.  OVERFLOW stage/trajectory stay DEBT with
`first_over_bar=kt2 {T,u}`; LOCK_EXCHANGE stage remains AT-BAR and its
trajectory remains DEBT with `first_over_bar=kt4 {u}`.  Tracer rows are
bit-identical to their Round-12 candidate outputs, so no row moves away from
the oracle by more than two ulp and no status crosses downward.

## Round-12 preregistration and EOS operand walk

The committed preregistration ranked live QCO depth association first,
salinity normalization second, constants third, EOS selection fourth, and
compiled association fifth.  The config-local WRITE-only record was accepted
only after tracing the executed overload: `stprk3_stg.F90:321-324` calls the
four-argument `eos(ts,Kmm,rhd,rhop)`, which resolves to
`eos_insitu_pot_New_t`.  The first attempted record had instrumented the
three-argument overload and was rejected because its `prd` did not equal the
HPG `rhd`; it is not science evidence.  The corrected record's `prd` equals
the stage-2 HPG input bit-for-bit, and ordinary stage/restart outputs remain
bit-identical.

NEMO evaluates `zh`, `zt`, `zs`, `ztm`, `zn3`, `zn2`, `zn1`, `zn0`, `zn`,
and `prd` in that written order at `eosbn2.F90:260-288`.  The normalizers are
set at `:1926-1929`, the TEOS-10 coefficients begin at `:1931`, `rho0` is
`1026._wp` at `:1898`, and `r1_rho0=1._wp/rho0` at `:2331-2334`.  The
resolved deck selects `ln_TEOS10=.true.` and the other EOS flags false
(`round12_oracle_eos_v2/namelist_cfg:124-129`), which `eos_init` maps to
`np_teos10` at `eosbn2.F90:1911-1924`.  Under QCO, `domqco.F90:159-161`
first forms `r3t=ssh*r1_ht_0`; `domzgr_substitute.h90:50,56,75,139` expands
the used depth as `gdept_0*(1+r3t)`.

| source-ordered discriminator (oracle inputs, fp64 production JIT) | before | after | disposition |
|---|---:|---:|---|
| `pts(T)`, `pts(S)`, `pdep`, `zh`, `zt`, `zs`, `ztm` | all bit-exact | all bit-exact | preregistered depth/salinity candidates REFUTED |
| first departure: `zn0` | `9.094947017729282e-13`, 4 ulp, 12914 cells | `0`, 0 ulp | source-operation association owner CONFIRMED |
| `zn` | `9.094947017729282e-13`, 4 ulp | `0`, 0 ulp | cleared |
| `prd` | `9.378348791999613e-16`, 18000 cells | `0`, 0 ulp | cleared for identical oracle operands |
| pure NumPy literal transcription | `0` through `prd` | `0` through `prd` | confirms source statements/constants, isolates compiled association |

All 52 coefficients and the six scalar constants were bit-identical.  JIT and
eager already agreed for the same shared callable, so this was not the
round-10 route split.  NEMO's compiler flags include `-fdefault-real-8 -O3`
and `-funroll-all-loops`, but no `-ffast-math`, `-march`, or `-mfma`; the
Fortran source association is therefore the executable target.  The one
shared EOS now places a finite, differentiable IEEE identity after each
source-level operation as well as the existing optimization barrier.  That
prevents CPU LLVM from contracting or reassociating adjacent operations;
there is no card guard, GYRE arm, callback, or second EOS implementation.

The diagnostic's planted wet-cell mutation exits `1`.  The strengthened
parity test checks every EOS intermediate; restoring the pre-fix barrier-only
helper makes it fail, while the fixed suite passes.  The accepted operand
record is fp64, stage 2, `kt=1` (the state producing whole-step `kt=2`), with
`Knn=Kmm=3`, `Krhs=2`, and `neos=-1` (TEOS-10).

## Round-10 preregistration (before diagnosis)

Round 10 reuses the existing phase-3 readers, card, private WRITE-only hooks,
fp64 policy, masks, and scoring code.  Repository search found no unified
same-input eager/JIT operator ladder, so the existing GYRE completion gate is
extended rather than creating parallel operator implementations.

The ordered parity output is fixed before measurement: EOS/prd, HPG,
vorticity, momentum advection, stage composition to Kaa, tracer advection,
tracer ZDF, momentum ZDF, stage transports, WZV, external-mode/drag, and final
state.  Each row compares arrays returned by the same callable with identical
entry state and fp64 operands.  Four ulps is the preregistered diagnostic owner
threshold; certification remains the campaign's stricter bit-equality/2-ulp
rule.  A planted one-bit output mutation must exit nonzero.

The independent mask discriminator compares the exact 3-D array handed to the
Roquet anomaly function with `tmask(1,:,:,:)` from OVERFLOW's `mesh_mask.nc`,
after the bridge's documented halo/axis transform.  Exact equality requires
zero differing cells and zero NEMO-wet cells zeroed by the applied mask.  Any
nonzero count confirms a bridge/mask owner and records the first `(i,j,k)`;
exact masks with a changed wet HPG trend instead send ownership to the shared
HPG consumer.  Success remains the predeclared two-ulp lane-1 comparison bar,
bit-identical T/S, unchanged first-over-bar, and no card guard.

Before measurement I reread the GYRE reconciliation receipt, the requested
lane-1 stage/SSH/phantom/census/face-thickness receipts, and the branch-
isomorphism map.  The unmerged round-7 WIP was not trusted: every live result
below was rederived from NEMO 5.0.2 source and a newly built config-local
instrument.  The run remained CPU-only, production-JIT fp64, and used the lane-1 binary
record format, central time-level registry, immutable pointwise `1e-15` bar,
one-variable arms, planted controls, and shared log-log growth instrument.

## Round-10 JIT/eager diagnosis

The defect was a route split.  Calling the public step inside
`jax.disable_jit()` bypassed its cached `_step_jitted` production kernel, so
the supposed eager comparison was not evaluating the production route.
The shared wrapper now enters `_step_jitted` under an explicit
`jax.disable_jit(False)` boundary; the direct EOS and HPG diagnostic wrappers
likewise compare the same callable.  This is a shared execution-path fix, not
a numerical tuning or card-specific branch.  An attempted nested `lax.scan`
Horner evaluator was discarded because it made the full GYRE compile stall;
the final code retains NEMO's explicit parenthesized source statements.

On `f8295e426c3`, the preregistered ladder failed first at EOS/prd: `prd`
`8.436584764126565e-13` (256833 ulp) and pressure anomaly
`5.558831617236137e-9` (135018 ulp).  It then exposed stage-2 Kaa u/v gaps of
`4.674608275337735e-13`/`5.187571292669668e-13` and raw stage-transport gaps
`zFu=1.4172543842505547e-5`, `zFv=1.3654937902174424e-5`,
`zFw=1.7636114772301426e-4`.  The strengthened test therefore demonstrably
fails on the reviewed tip (`2 failed`).

At `7a38ef6e4`, all eleven production-fp64 rows—EOS/prd, HPG, vorticity,
advection, stage-2 Kaa, tra_adv, tra_zdf, dyn_zdf, stage transport,
external-mode/drag, and final update—report `verdict=PASS`.  The planted EOS
mutation reports `verdict=FAIL` and exits nonzero.  The focused eager/JIT
regression passes both the direct EOS/HPG/transport path and an outer
`disable_jit` call around the public production step (`2 passed`).

## Round-8 card-choice disclosure and one-variable audit

Round 9 searched the existing DINO/lane-1 option surface before changing the
card.  All six round-8 additions already existed as canonical shared options;
none is a GYRE implementation.  Each is selected by the resolved GYRE program,
so no flip is reverted.  The TKE/EVD selectors share an Nbb input but retain
independent consumers; `stprk3.F90:154-165` makes the RK3 whole-step entry Nbb
for both.

| choice | resolved namelist and selecting source | one-variable JIT move at kt=2 (T/S/u/v/SSH) | worst JIT move at kt=10 | round-8 / round-9 disposition |
|---|---|---|---:|---|
| `tke_n2_time_level=nemo_before` | `EXP00/namelist_cfg:204,215-217`; `stprk3.F90:154-165` | `0/0/0/0/0` | `0` | UNASKED addition; ASKED audit, retained (RK3-equivalent timing) |
| `evd_n2_time_level=nemo_now_before` | `EXP00/namelist_cfg:205-207`; `stprk3.F90:154-165` | `1.77e-3/9.34e-5/2.53e-2/2.53e-2/0` | `8.26e-3` | UNASKED addition; ASKED audit, retained |
| `een_e3f_scheme=nemo_avg4` | `EXPREF/namelist_ref:1072-1073`; `dynvor.F90:918-950` | `1.51e-16/1.93e-16/5.12e-9/2.07e-9/0` | `1.20e-2` | UNASKED addition; ASKED audit, retained |
| `een_metric_weighting=nemo` | `EXP00/namelist_cfg:163-165`; `dynvor.F90:518-531` | `1.51e-16/1.93e-16/6.94e-18/6.94e-18/0` | `3.97e-11` | UNASKED addition; ASKED audit, retained |
| `een_q_boundary=nemo_live` | `EXPREF/namelist_ref:1069`; `dynvor.F90:450-490` | `0/0/0/0/0` | `0` | UNASKED addition; ASKED audit, retained (uninformative through kt=10) |
| `shortwave_penetration.scheme=nemo_qsr_2bd` | `EXP00/namelist_cfg:73,76-79`; `traqsr.F90:55-56,627-712,1239,1266-1276` | `2.20e-3/1.97e-14/0/0/0` | `9.01e-2` | UNASKED addition; ASKED audit, retained under NEMO's public selector name |

The earlier “142× kt=2 T improvement” is attributed to this named selector
set, not to “round 8”: the direct ablations show the two live tracer-scale
members are full two-band shortwave and EVD N2 timing, while `nemo_avg4`
becomes trajectory-scale by kt=10.  Their effects are nonlinear and partly
cancelling, so the audit does not invent an additive per-arm share.  The
production-JIT arm artifact is `selector_arms.json`, SHA-256
`487879f62bb95fa4f5f22d8eab9edd4fc70eb27ab632e9eab07a24d4b41aee3f`.
The `--plant` two-choice mutation exits 2, proving the one-variable manifest
fails closed.

## Ordered walk

### Bottom drag closes the external-mode register

GYRE resolves `nn_drg=np_non_lin`, `ln_drgimp=.true.`, and `rn_Cd0=1.0e-3`
(`rn_Cdmax=0.1`, `rn_ke0=2.5e-3`, `rn_z0=3.0e-3`).  NEMO constructs one Kmm
quadratic coefficient in `zdfdrg.F90:138-190`, freezes that coefficient and
the Kmm baroclinic residual in `dynspg_ts.F90:1584-1643`, applies drag to the
entry external velocity inside every substep at `dynspg_ts.F90:699-705`, and
uses the same coefficient in the bottom-cell implicit diagonal at
`dynzdf.F90:148-160,293-305`.

The canonical DINO bottom-drag implementation is now consumed as one
inseparable WS-RK3 identity; no GYRE-only operator was added.  At substep 2
the direct drag operands agree to `4.08e-27` absolute and combined
`trd_u/trd_v` agree to `3.83e-23`.  Omitting only in-substep drag produces
`6.73e-14` and `6.76e-14`, the exact residual scale, so the boundary label is
**CONFIRMED_CAUSAL_OWNER_AT_SUBSTEP2**.  All 800 external-mode rows (50
substeps × 16 boundaries) are AT-BAR.

For continuity with the accepted review redirect: before the literal ENE
operand landing, substep-2 `trd_u` was `8.60e-12`, which is **1.59%** of the
`5.41e-10` oracle magnitude—not tiny in relative terms.  Literal ENE reduced
it 128× to `6.73e-14` (`0.0124%`) and bottom drag then reduced the combined row
to `2.07e-25`.  This scaling sequence precedes every owner label.

The freshwater discriminator carried from the reviewed reversal remains
source- and runtime-backed.  NEMO's first SSH increment selects the
`8.50091456831154e-6` owned-domain EMP mean; the naive 704-cell value
`9.166209883944753e-6` would cause `1.86749562283012e-7` error.
`lib_fortran_generic.h90:92,144-148` multiplies each 2-D reduction operand by
`smask0_i`, and `dommsk.F90:200-205` constructs that unique interior mask.
Thus the masked numerator is faithful even though `usrdef_sbc.F90:122-140`
passes an unmasked array expression to `glob_2Dsum`.

### Stage-2 momentum improves but does not close under production JIT

The round-8 eager values remain withdrawn.  Round 11 recorded the actual
stage-2 accumulator before the update: `Krhs == Kaa == 2`, so the older
post-update operand stream had aliased and overwritten the quantity it called
`Krhs`.  The new config-local `MY_SRC/stprk3_stg.F90:404-416` stream is emitted
before that overwrite.  Its ordinary stage-2 state and restart remain
bit-identical (`55e780b8…`, `3271da17…`).

The source execution order is literal and resolved.  `stprk3_stg.F90:321-334`
sets `Krhs` with EOS/HPG first, then accumulates `dyn_vor`, then `dyn_adv`;
vector `dyn_adv.F90:78-89` executes C2 `dyn_keg` then `dyn_zad`.  Stage 1 was
seeded earlier by `stp2d.F90:126-165`, where HPG precedes LDF, vorticity, KEG,
and ZAD, and its depth mean is passed to the external solver.
`stprk3.F90:183-207` runs that `stp_2D`/`dyn_spg_ts` solve once before all
three stages, so
`dyn_spg` supplies `uu_b/vv_b(Kaa)` rather than silently adding another stage-2
3-D `Krhs` term.  Stage 3 alone appends `dyn_ldf` at
`stprk3_stg.F90:395-409`.  Thus no live stage-2 source was omitted from the
record.

GYRE's resolved `ln_dynadv_vec=.true.` selects the direct velocity update
`Kaa = (Kbb + rDt*Krhs)*mask` at `stprk3_stg.F90:365-369`, with
`rDt=rn_Dt/2=7200 s`; the QCO e3-weighted alternative at `:370-387` is dead
for this configuration.  Reference-depth mean replacement then computes and
adds `zub/zvb` at `:433-446`.

Round 12 changes only the shared EOS association.  The same production-JIT
composition instruments give:

| production-JIT fp64 boundary | u max | v max | round-11 baseline | scaling-first disposition |
|---|---:|---:|---|---|
| complete pre-update `Krhs` | `3.460773136050383e-18` | `3.3977899498493705e-18` | `1.0739067765248428e-16` / `1.2178020157343528e-16` | improved 31×/36×, but DEBT against effective `1e-15/7200` bar |
| raw `Kaa`, before mean replacement | `2.4917568008930857e-14` | `2.4464073345234483e-14` | `7.732128807125434e-13` / `8.768174513816736e-13` | improved 31×/36×; DEBT |
| corrected `Kaa` | `2.0033670902752654e-14` | `2.0003665607629117e-14` | `4.674608410863007e-13` / `5.187571089381761e-13` | improved 23×/26×; DEBT |

The round-11 one-variable composition arms remain valid controls: exact
oracle `Krhs` made raw Kaa bit-identical and corrected Kaa AT-BAR; NEMO-order
RHS accumulation and live-thickness mean replacement were zero-move, and the
dead QCO/e3 update was NEAR-NULL.  Round 12 therefore does not relabel those
operations as owners.  It removes almost exactly the entering-RHS scale, but
the remaining `rDt*|δKrhs|` still predicts the raw-Kaa debt.

The follow-on WRITE-only `MY_SRC/dynhpg.F90:343-419` record separates the
cumulative `zhpi`, local terrain correction `zuap`, stored sum, and metric
reciprocals.  `dynhpg.F90:340-390` is the cited production recurrence.  The
effective tendency bar is `1.388888888888889e-19`:

| HPG operand/term | u max | v max | round-11 baseline | disposition |
|---|---:|---:|---|---|
| `r1_e1u/r1_e2v` | `0` | `0` | `0` | bit-identical |
| production `rhd` input | `2.220446049250313e-16` | same 3-D operand | `9.378348791999613e-16` | two cells remain after propagating the changed stage |
| production `e3w` input | `0` | same | `0` | bit-identical |
| production `gdept_z0` input | `4.547473508864641e-13` absolute, `1.0957024380274693e-16` normalized | same | `5.684341886080802e-14` absolute | AT normalized bar; not residual-scale ownership evidence |
| `zhpi` from production operands | `3.3286426446217166e-18` | `3.3322417282676192e-18` | `1.0750480746694731e-16` / `1.2202333356727058e-16` | improved 32×/37×; DEBT at effective bar |
| `zuap` from production operands | `7.18162964712906e-20` | `7.18162965722648e-20` | `2.550418053222173e-21` / `2.550594455142355e-21` | AT effective bar despite moving away |
| full HPG from exact oracle operands | `2.3959792121371355e-20` | `2.3959792929164922e-20` | same | AT effective bar |

The isolated EOS evaluator fed the oracle's recorded T/S/depth and reproduces
every intermediate and `prd` bit-for-bit.  After that shared fix is propagated
through the live RK3 step, however, stage-entry T and S differ from NEMO in
nine wet cells by one ulp; the derived live `prd` differs in two cells by
`2.220446049250313e-16`.  HPG integrates that to the remaining
`3.33e-18` tendency and corrected Kaa remains DEBT.  The EOS source-association
owner is **CONFIRMED_FIXED_FOR_IDENTICAL_OPERANDS**; the next upstream owner of
the propagated tracer input is **UNMEASURED**.  It would violate the ordered
gate to claim EOS itself fully closed or proceed downstream.  Both inherited
composition/HPG plants and the new EOS plant exit nonzero.

### Stage-3 transport and WZV

Round 12 does **not** remeasure this boundary because the ordered stage-2 gate
above remains DEBT.  The values below are retained round-10 production-JIT
context, not promoted round-12 evidence.  Stage-3 `zFu/zFv/zFw/zub/zvb`
instrumentation and the e3u/e3v(Kmm) walk remain the next gated boundary.

NEMO forms `zFu/zFv` from the retained Kmm stage velocity and barotropic mean
at `stprk3_stg.F90:257-278`, then passes those already materialized arrays to
`wzv(...,np_transport)` at `traadv.F90:220-226`.  Production now reuses that
same pair rather than recomputing an algebraically equivalent transport.

| boundary | normalized max | disposition |
|---|---:|---|
| `zFu` | `2.268967008217296e-9` | DEBT; production-JIT |
| `zFv` | `2.4393178638950564e-9` | DEBT; production-JIT |
| `zFw` / shared-transport production | `4.367606308796769e-7` | DEBT; production-JIT |
| `zFw` / legacy rederived transport | `4.367606359827147e-7` | DEBT; production-JIT |

The private stored-barotropic-mean association arm moves only
`2.1235675846249628e-13` against the JIT `4.367606308796769e-7` residual; it is
**NOT_SOLE_OWNER**.  Sharing the materialized transport moves only
`1.6533842413934496e-13`, also **NOT_SOLE_OWNER** under JIT.  The eager owner
labels in the previous version of this receipt are withdrawn.

A new config-local WRITE-only `MY_SRC/traadv.F90` records `ww` after WZV,
after the adaptive partition, and final `pFw`.  GYRE resolves
`ln_zad_Aimp=.false.`, so `wi` is a zero-extent NEMO array and the parser
records that logical boundary as exact zero.  Under production JIT the WZV
velocity is already DEBT at `1.569607877847196e-14`; `pFw` is
`4.3676062860970496e-7`.  The eager-only `8.44e-21`/`2.3488e-13` statement is
withdrawn.

Scaling-first arms assign no false sole owner:

- replacing only `pFu/pFv` moves `4.367606243337112e-7` and leaves
  `9.221013305395866e-13`;
- replacing only the `Kbb/Kmm/Kaa` SSH operand triplet moves
  `9.221013305395866e-13` and leaves `4.367606243337112e-7`;
- the planted SSH-arm control writes exactly `1.0` into a wet `ww` cell and
  fails closed.

The horizontal transport is the dominant upstream contributor but does not
clear the bar; SSH is near-null at this scale.  The literal WZV recurrence is
not assigned a sole-owner label.

### First unmeasured boundary: implicit ZDF

Round 12 does not enter this matrix walk: its prerequisite stage-2 and
stage-3-transport boundaries have not cleared.

The eager-only claim that the explicit stage-3 accumulator cleared is
withdrawn.  Its production-JIT rerun is DEBT before ZDF: T
`7.325244772979124e-14`, S `5.786348021897527e-15`.  Injecting the oracle
pre-ZDF tracer does not move the compiled final state, so the honest label is
**PRE_ZDF_ACCUMULATOR_DEBT_NO_ZDF_OWNER**.  The JIT whole-step kt=2 tracer rows
are T `1.3614736849003888e-12` and S `2.2181101297999213e-14`.
`tra_zdf`/`dyn_zdf` internals remain **UNMEASURED** and this round does not
start their matrix walk.

## Re-pinned kt=1 and kt=2…10 sweep

Round 12 reruns the entire sweep from `afd6a2082` under production JIT on CPU.
At kt=1 T/S are bit-exact.  At-rest u/v/SSH are still **UNINFORMATIVE** because
their oracle and candidate values are identically zero.  The first whole-step
DEBT remains kt=2, but SSH now joins it:

| field | kt=2 normalized max | status |
|---|---:|---|
| T | `1.3614736849003888e-12` | DEBT |
| S | `2.2181101297999213e-14` | DEBT |
| u | `9.484089954572578e-7` | DEBT |
| v | `8.987992592336029e-7` | DEBT |
| SSH | `2.3724511938327808e-15` | DEBT; downward crossing from round-11 `4.77048955893622e-16` |

Like-for-like production-JIT comparison against the immediate round-11
baseline gives 20 improved, 23 worsened, and seven identical rows.  The seven
identical rows include all five exact kt=1 rows.  Rule 8 requires every
worsened row, even when it was already far over bar:

| field | kt | round-11 normalized max | round-12 normalized max | ratio |
|---|---:|---:|---:|---:|
| T | 2 | `1.3608682491316726e-12` | `1.3614736849003888e-12` | `1.00044488933378` |
| u | 2 | `9.484089544036715e-7` | `9.484089954572578e-7` | `1.0000000432868` |
| SSH | 2 | `4.77048955893622e-16` | `2.3724511938327808e-15` | `4.97318181818182` |
| T | 3 | `3.7223422533452327e-4` | `3.7223422533633866e-4` | `1.00000000000488` |
| S | 3 | `3.683246743167771e-5` | `3.683246743187058e-5` | `1.00000000000524` |
| u | 3 | `9.311678912623694e-3` | `9.311678913140425e-3` | `1.00000000005549` |
| v | 3 | `4.719077591028548e-3` | `4.719077591308032e-3` | `1.00000000005922` |
| SSH | 3 | `7.074668139339354e-7` | `7.074669648514083e-7` | `1.00000021332092` |
| u | 4 | `1.418240401102859e-2` | `1.4182404011403506e-2` | `1.00000000002644` |
| SSH | 4 | `5.060118542839914e-7` | `5.06011882079032e-7` | `1.00000005492962` |
| u | 5 | `2.0733890538774537e-2` | `2.073389053895669e-2` | `1.00000000000879` |
| v | 5 | `4.379766615426355e-2` | `4.3797666154580514e-2` | `1.00000000000724` |
| SSH | 5 | `6.701856586948003e-5` | `6.701856627955131e-5` | `1.00000000611877` |
| S | 6 | `1.287536211062039e-4` | `1.287536215345963e-4` | `1.00000000332723` |
| u | 6 | `3.1134730338205307e-2` | `3.113473033866225e-2` | `1.00000000001468` |
| v | 6 | `6.021959325392878e-2` | `6.0219593254002834e-2` | `1.00000000000123` |
| T | 7 | `4.546668332780763e-3` | `4.546668332781217e-3` | `1.00000000000010` |
| S | 7 | `1.2244044280379423e-4` | `1.224404428342701e-4` | `1.00000000024890` |
| v | 7 | `6.631577474091133e-2` | `6.631577474100295e-2` | `1.00000000000138` |
| u | 9 | `5.256039963425046e-2` | `5.256039963431678e-2` | `1.00000000000126` |
| u | 10 | `5.624987258844358e-2` | `5.624987258870016e-2` | `1.00000000000456` |
| v | 10 | `1.1173366024295098e-2` | `1.1173366024902619e-2` | `1.00000000005437` |
| SSH | 10 | `2.12554757616743e-4` | `2.1255475784966258e-4` | `1.00000000109581` |

The baseline and current production-JIT sweep artifacts have SHA-256
`17b1d103bd6e6871ddeecbfe59d9ccc9b7a5551041c4dd13e2c1f1699cc66489`
and `3979363314af89460dceb73c3544a3502040c406d133271dbea9181ffa02256f`,
respectively.  At kt=10 the current normalized maxima are T
`5.636638587568597e-3`, S `1.5483121735304233e-4`, u
`5.624987258870016e-2`, v `1.1173366024902619e-2`, and SSH
`2.1255475784966258e-4`.

The lane-1 log-log instrument labels the short tail for T and u
POLYNOMIAL_FIT_PREFERRED (exponents `0.60445` and `0.95631`) and SSH
BOUNDED_OR_DECAYING_NO_AMPLIFYING_MODE.  These are characterization labels,
not acceptance evidence.

## Pending D boundary and round-12 cross-card oracle scoring

Independent re-review returned **SHIP** for round 10 and clarified that this
gate compares against the previous legoESM output, not directly against NEMO.
Round-11 independent review was still running when round 12 was dispatched.
The retained D criterion is therefore an **OPEN USER DECISION**.  Round 12
does not modify that gate or criterion; it reports both its direction-blind
compare-to result and direct oracle-relative residuals.

The shared Roquet EOS remains one implementation.  Its NEMO-literal
`zn3/zn2/zn1/zn0` nesting is unconditional for EOS-80 and TEOS-10, and the
literal terminal mask remains
`prd = (zn*r1_rho0 - 1)*ztm` (`src/OCE/TRA/eosbn2.F90:288`).  No EOS identity
guard was added.

The required discriminator found that the mask is not the owner.  The exact
3-D array applied by OVERFLOW has shape `(3,202,100)`, 17000 wet cells, and is
bit-identical to `tmask(1,:,:,:)` from its `mesh_mask.nc`: differing cells `0`,
NEMO-wet cells zeroed `0`, and therefore no first differing `(i,j,k)`.  The
planted flip at `(0,0,0)` exits nonzero.  This agrees with the HPG source:
`dynhpg.F90:340-390` pairs same-level T points and masks the resulting U/V
trend.

The round-9 `3.051e12`-ulp result was primarily an invalid comparison with
missing raw W-grid geometry.  The shared PE path now raises when `e3w0` is
absent, like its sibling, and every reachable lane-1 card passes its real
oracle `e3w_0`; midpoint reconstruction was removed.  That reduces the stage
move to ulps, but not to the required two ulps:

| round-12 production-JIT compare-to gate | certified rows | largest move | verdict |
|---|---:|---:|---|
| OVERFLOW stage sweep | 9 | `4.0625` ulp | FAIL |
| LOCK_EXCHANGE stage sweep | 9 | `0.125` ulp | PASS |
| OVERFLOW kt=1…10 trajectory | 50 | `146.9375` ulp | FAIL |
| LOCK_EXCHANGE kt=1…10 trajectory | 50 | `4.72357177734375` ulp | FAIL |

The direct stage rows below are normalized maxima against NEMO, before
(round-10/11 shared EOS) and after round 12.  They are the complete nine-row
stage register for each card:

| card / row | before | after | status after |
|---|---:|---:|---|
| OVERFLOW instantaneous stage-1 u | `2.914335439641036e-16` | `1.3877787807814457e-17` | AT-BAR, better |
| OVERFLOW instantaneous stage-2 u | `1.5709517020567887e-12` | `1.5711182355104825e-12` | DEBT, worse <2 ulp |
| OVERFLOW instantaneous stage-3 u | `7.068304275215098e-12` | `7.069206331422606e-12` | DEBT, worse `4.0625` ulp |
| OVERFLOW instantaneous kt=2 u | `7.068304275215098e-12` | `7.069206331422606e-12` | DEBT, worse `4.0625` ulp |
| OVERFLOW kt=2 T | `7.815970093361103e-15` | same | DEBT, identical |
| OVERFLOW baroclinic stage-1 u | `1.1102230246251565e-16` | `6.938893903907228e-18` | AT-BAR, better |
| OVERFLOW baroclinic stage-2 u | `1.5711251744043864e-12` | `1.5711182355104825e-12` | DEBT, better |
| OVERFLOW baroclinic stage-3 u | `7.068463869774888e-12` | `7.069192453634798e-12` | DEBT, worse `3.28125` ulp |
| OVERFLOW baroclinic kt=2 u | `7.068463869774888e-12` | `7.069192453634798e-12` | DEBT, worse `3.28125` ulp |
| LOCK instantaneous stage-1 u | `2.5153490401663703e-17` | `6.505213034913027e-19` | AT-BAR, better |
| LOCK instantaneous stage-2 u | `2.8189256484623115e-17` | `4.336808689942018e-19` | AT-BAR, better |
| LOCK instantaneous stage-3 u | `2.4015078120553923e-17` | `8.673617379884035e-19` | AT-BAR, better |
| LOCK instantaneous kt=2 u | `2.4015078120553923e-17` | `8.673617379884035e-19` | AT-BAR, better |
| LOCK kt=2 T | `0` | `0` | bit-identical |
| LOCK baroclinic stage-1 u | `6.396792817664476e-18` | `4.336808689942018e-19` | AT-BAR, better |
| LOCK baroclinic stage-2 u | `9.540979117872439e-18` | `2.168404344971009e-19` | AT-BAR, better |
| LOCK baroclinic stage-3 u | `1.8955811279475375e-17` | `4.336808689942018e-19` | AT-BAR, better |
| LOCK baroclinic kt=2 u | `1.8955811279475375e-17` | `4.336808689942018e-19` | AT-BAR, better |

Every trajectory row is recorded below as `before → after`; all are
production-JIT fp64 normalized maxima against NEMO.  `U` means
UNINFORMATIVE and `M` means UNMEASURED, matching the gate rather than silently
promoting a structural zero.

| OVERFLOW kt | T | S | u | v | SSH |
|---:|---:|---:|---:|---:|---:|
| 1 | `0→0` | `0→0` | `0→0` | `0→0 M` | `0→0` |
| 2 | `7.815970093361103e-15→same` | `2.0301221021717148e-16→same U` | `7.068304275215098e-12→7.069206331422606e-12` | `0→0 M` | `3.3306690738754696e-16→2.7755575615628914e-17` |
| 3 | `5.725553364754887e-12→same` | `2.0301221021717145e-16→same` | `4.181440946965376e-9→4.181444881318219e-9` | `0→0 M` | `1.4938744685721872e-13→1.4924173008523667e-13` |
| 4 | `4.110836115955862e-11→4.1108272341716646e-11` | `2.0301221021717145e-16→same` | `2.3323167099020825e-8→2.3323169319466874e-8` | `0→0 M` | `2.896962253418067e-10→2.897001111223929e-10` |
| 5 | `1.1836380764407291e-10→same` | `4.060244204343429e-16→same` | `4.6243884971319815e-8→4.6243887726060695e-8` | `0→0 M` | `1.1794371007622928e-8→1.1794364623840536e-8` |
| 6 | `2.3705917229221987e-10→same` | `6.090366306515144e-16→same` | `3.184360849589618e-7→3.1843605377210316e-7` | `0→0 M` | `1.0532435790189254e-7→1.05324343940838e-7` |
| 7 | `3.725337371918156e-10→3.7253382600965757e-10` | `4.060244204343429e-16→same` | `1.262631212875509e-6→1.2626311811717028e-6` | `0→0 M` | `3.89638112441304e-7→3.896381028378748e-7` |
| 8 | `5.484132259425675e-10→5.484131371247256e-10` | `4.060244204343428e-16→same` | `2.8378625916772315e-6→2.8378625590505524e-6` | `0→0 M` | `7.518269373174569e-7→7.518269323769644e-7` |
| 9 | `9.63236068685091e-10→9.63236335138617e-10` | `6.090366306515141e-16→same` | `4.344091409307083e-6→4.344091389274496e-6` | `0→0 M` | `8.261133408460353e-7→8.261133412346133e-7` |
| 10 | `1.5225928073903094e-9→1.5225922744832577e-9` | `8.120488408686856e-16→same` | `5.422695732899829e-6→5.422695711472525e-6` | `0→0 M` | `6.265018272777478e-7→6.265018344109308e-7` |

| LOCK kt | T | S | u | v | SSH |
|---:|---:|---:|---:|---:|---:|
| 1 | `0→0` | `0→0` | `0→0` | `0→0 M` | `0→0` |
| 2 | `0→0` | `0→0 U` | `2.4015078120553923e-17→8.673617379884035e-19` | `0→0 M` | `0→0 U` |
| 3 | `1.1842378929335003e-16→same` | `2.0301221021717148e-16→same U` | `5.445828927782073e-17→1.3010426069826053e-18` | `0→0 M` | `1.3552527156068805e-18→2.710505431213761e-20` |
| 4 | `3.552713678800501e-16→same` | `2.0301221021717148e-16→same` | `2.5781139274480298e-14→2.5789380058024135e-14` | `0→0 M` | `1.870248747537495e-18→2.710505431213761e-20` |
| 5 | `2.368475785867e-16→same` | `2.0301221021717145e-16→same` | `1.4704223624296964e-13→1.4690369892224851e-13` | `0→0 M` | `3.686287386450715e-18→5.421010862427522e-20` |
| 6 | `5.921189464667501e-16→same` | `4.060244204343429e-16→same` | `5.101191787504258e-13→5.099759963010219e-13` | `0→0 M` | `5.637851296924623e-18→1.0842021724855044e-19` |
| 7 | `1.5395092608135503e-15→same` | `4.060244204343429e-16→same` | `1.362963215868155e-12→1.3626691734627133e-12` | `0→0 M` | `9.64939933512099e-18→1.0842021724855044e-19` |
| 8 | `3.5527136788005005e-15→same` | `6.090366306515144e-16→same` | `3.0797251955681088e-12→3.0792796291327987e-12` | `0→0 M` | `1.3877787807814457e-17→4.336808689942018e-19` |
| 9 | `8.171241461241152e-15→same` | `6.090366306515144e-16→same` | `6.184440329640162e-12→6.1836913356743545e-12` | `0→0 M` | `2.3418766925686896e-17→1.3552527156068805e-17` |
| 10 | `1.68161780796557e-14→same` | `6.090366306515144e-16→same` | `1.1372521780089566e-11→1.137147293646043e-11` | `0→0 M` | `3.664603343001005e-17→3.2873348682939396e-17` |

Direct oracle scoring exposes the required adverse moves instead of hiding
them behind the direction-blind comparison.  OVERFLOW moves away by more than
two ulps at instantaneous/baroclinic stage 3 and kt=2; in the trajectory, kt=2
u, kt=3 u, kt=4 u/SSH, kt=5 u, and kt=10 SSH do so.  Its tracer bits also move
at kt=4/7/8/9/10, although several moves are toward NEMO.  LOCK stage rows all
improve or stay AT-BAR; its compare-to trajectory failures at kt=8–10 u are
improvements versus NEMO, while kt=4 u moves away by less than two ulps.  No
lane-1 row crosses the fixed `1e-15` bar in the adverse direction.  The
`first_over_bar` registers are unchanged: OVERFLOW `kt=2 {T,u}` and LOCK
`kt=4 {u}`.

These mixed oracle-relative moves are findings, not tuning invitations.  The
two-ulp comparison rule and its interpretation remain pending the user's
decision.  Independently, the GYRE ordered gate already requires **STOP** at
stage 2.  No card guard, GYRE-only arm, stage-3 transport rerun, or
tra_zdf/dyn_zdf matrix walk was started.

## Provenance, controls, and review

The oracle executable was rebuilt only from config-local `MY_SRC`; no shipped
NEMO source was modified.  The round-12 executable is `6c2d5862…`, the EOS
operand record `f8ab6767…`, and the matching HPG input record `2c2c5eef…`.
The config-local sources, pre/post/plant diagnostics, stage-2 measurements,
immediate GYRE baseline, all four lane-1 baselines, current oracle-relative
gates, and compare-to logs are pinned in
`nemo_testcases_l2_gyre_phase3_round12_artifacts.sha256`.

The monolithic GYRE completion process still retains too many static-hook
executables, so the fail-closed low-memory route runs the production kt=1…10
sweep once, then each private hook in a fresh process.  The round-12 sweep is
CPU-only production JIT at SHA-256
`3979363314af89460dceb73c3544a3502040c406d133271dbea9181ffa02256f`.
Every certification gate rejects `JAX_DISABLE_JIT`; no eager result is
promoted.

### Round-9 E/F/G safeguards

The three source-string assertions were replaced with four behavioral tests:
the returned stage-3 content is the array actually advanced; stage-3 QSR
removes Kbb and adds Kmm; `nemo_qsr_2bd` deposits the full QSR; and the live
Kmm optical ladder changes the vertical profile.  Each was mutated once and
observed to fail; the corrected live-ladder mutation disabled both interface
depth and thickness and failed at
`test_two_band_live_kmm_ladder_changes_the_stage3_profile` before restoration.
The restored 18-test file passes.

The PE dynamics path no longer synthesizes midpoint `e3w` when `e3w0` is
missing.  A caller census found the lane-1 recipes could reach it; those cards
now pass the real oracle `e3w_0` and missing geometry raises in both PE and its
sibling path.  The no-scheme-duplication census still treats these as shared
RoutineRows S-49/S-50, not exceptions.

The required shortwave search (`qsr|shortwave|rgb|two_band|penetrat`) found the
existing `ShortwavePenetrationConfig.scheme`.  It was extended with NEMO-named
values `nemo_qsr_2bd` and `nemo_qsr_rgb`; the temporary public boolean was
removed.  `OceanExperimentConfig.validate_strict`, its membership coverage,
and a bogus-value footgun test cover the one selector.  NEMO's corresponding
switches and branches are `ln_qsr_2bd`/`ln_qsr_rgb` at
`traqsr.F90:55-56,627-712,1239,1266-1276`.

These E/F/G safeguard commits landed before the final D adjudication, contrary
to the requested ordering; this receipt flags that process error rather than
hiding it.  They do not tune a trajectory.  After D failed, no further physics
work was performed.

### Lane-1 certification-regime audit

Search of every testcase gate and the requested lane-1 receipts found two
explicit eager-only instruments.  All numbers produced by
`nemo_testcase_overflow_barotropic_gate.py`'s traced 19-frame/reseeded route
(`jax.disable_jit`, lines 276/350) are **UNVERIFIED as production-JIT
certification**.  Phantom-velocity derivative/census numbers produced through
`nemo_testcase_phantom_velocity_probe.py:202` are likewise **UNVERIFIED**.
The phase-3 stage-sweep and trajectory gates call the public `.step()` whose
production kernel is JIT; their numbers are production-JIT.  No receipt number
was upgraded merely because its documentation omitted a regime stamp.

### ASKED / UNASKED register

| choice or action | origin | disposition |
|---|---|---|
| same-input 11-operator JIT/eager ladder and public-step route fix | ASKED round 10 | landed shared; all rows PASS, planted row FAIL |
| NEMO terminal `tmask` and OVERFLOW discriminator | ASKED round 10 | retained; exact mask, zero differing/zero wet-zeroed |
| literal shared EOS/HPG association | UNASKED round 9 correction, ASKED round 10 diagnosis | one implementation retained; cross-card gate FAIL, STOP |
| raw oracle `e3w_0`, no midpoint fallback | ASKED F | landed shared fail-close; no synthetic domain geometry |
| four behavioral replacements | ASKED E | landed; each path shown non-vacuous |
| source-named `nemo_qsr_2bd`/`nemo_qsr_rgb` selector | ASKED G | landed in existing selector; private boolean removed |
| six GYRE card choices in the earlier table | UNASKED round 8, ASKED round 9 audit | independently ablated and retained from resolved namelist |
| nested `lax.scan` Horner experiment | UNASKED diagnostic | reverted after compile stall; absent from final tree |
| immediate pre-update `Krhs`, raw/corrected `Kaa`, and HPG literal WRITE-only streams | ASKED round 11 | config-local only; ordinary outputs bit-identical |
| NEMO-order RHS, update/e3, and mean-replacement one-variable arms | ASKED round 11 | accumulation and mean arms zero-move; update arm NEAR-NULL; exact RHS clears |
| source-ordered EOS operand/intermediate dump | ASKED round 12 | config-local WRITE-only; executed overload identified; invalid first routing rejected |
| ranked QCO-depth / salinity / constants / EOS-choice arms | ASKED round 12 | first two refuted; constants and selection exact; no card choice changed |
| shared EOS source-operation association | forbidden round 11, explicitly ASKED round 12 | one shared implementation landed; oracle-input chain bit-exact; propagated stage `prd` still two-cell DEBT |
| finite IEEE identity between source operations | UNASKED implementation mechanism | retained only after NumPy/NEMO exact and XLA discriminator; shared, differentiable, no callback |
| compare-to gate or criterion change | explicitly forbidden round 12 | NOT MODIFIED; open user decision retained |
| stage-3 transport remeasurement | ASKED only if EOS and stage 2 clear | NOT ENTERED; corrected Kaa remains DEBT |
| GYRE tra_zdf/dyn_zdf matrix walk | ASKED only if stages 2 and 3 clear | NOT ENTERED |
| source-ordered stage-1 tracer routine checkpoints and 9-T/9-S cell census | ASKED round 13 | config-local WRITE-only; all first differ after `tra_adv` |
| transcendental and FCT candidates | ASKED round 13 | REFUTED_BY_EXECUTED_SOURCE at stage 1; no transcendental remedy proposed |
| complete oracle stage-1 transport arm | ASKED round 13 | CONFIRMED causal; makes T/S stage update bit-exact |
| native metric-bearing zF product association | ASKED round 13 after operand split | landed shared S-21 identity; literal statement confirmed at 0 ulp, but not the complete owner |
| stage-1 transport metric/e3/velocity/mask/corrected-velocity split | ASKED owner walk | first bad compound operand is corrected velocity; `un_adv/vn_adv` first bad primitive input |
| new public selector/card guard/GYRE-only physics arm | UNASKED and forbidden | none added |
| stage-3 transport and ZDF matrix walks | conditional on stage-2 clearing | NOT ENTERED; stage-2 corrected Kaa remains DEBT |

### Test and artifact closure

Focused results are: JIT parity `11 PASS` plus planted `FAIL`; public-step/EOS
parity `2 passed`; and the consolidated GYRE behavioral, recipe, config,
shortwave, and no-scheme-duplication suite `152 passed`.  The broad
strict-coverage file's new shortwave validation/footgun checks pass, but the
full file retains a pre-existing `grid_type` AST-classification inconsistency
and is not claimed green.  Larger combined pytest invocations and the
monolithic stage gate hit the documented compiler/resource wall; the
scientific gates above ran as separate CPU processes.

Round 11 additionally passed all 20 WS-RK3 tests plus all 35
single-implementation tripwires (`55 passed`).  Round 12 passes the expanded
WS-RK3, Roquet EOS, TEOS-10 RAB/BN2, time-level-registry, and
single-implementation suite: `108 passed` in `645.85 s`, CPU fp64.  The new EOS
plant exits `1`; restoring the old barrier-only operation makes the expanded
intermediate parity test fail.  The GYRE sweep gate exits `1` for scientific
DEBT, not a harness error; kt=1 is exact and its full artifact is retained.

Independent Claude re-review shipped round 10 and retracted the older JIT
blocker.  Round-11 review was in progress at this dispatch.  This round-12
result is a new Codex-internal measurement; independent review of it remains
outstanding and no dual-review claim is made.
Per the accepted process note, the earlier EMP reversal returned through
review with both source and runtime evidence before it landed.

Round 13 additionally passes 26 time-level tests, all 35 no-scheme-duplication
tripwires, and the complete WS-RK3 tracer file (`23 passed` in `571.85 s`).
That file's failing-first run was `22 passed, 1 failed`: the failure was the
legacy six-array test geometry leaving a newly optional native-product local
uninitialized.  After setting that legacy local to `None`, the full rerun is
green.  The broader private-cross-import suite reports three pre-existing
unrelated violations and no Round-13 import; it is not claimed green.
Round-12 independent review is running in parallel; Round-13 measurements
have no independent review yet and make no dual-review claim.
