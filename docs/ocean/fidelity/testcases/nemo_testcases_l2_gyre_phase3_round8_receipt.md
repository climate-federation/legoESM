# NEMO testcase lane 2 GYRE — Phase 3 rounds 8–25 boundary receipt

**Verdict: HOLD.**  Round 25 retracts the ill-posed stage-2/kt1 carried-Kbb
owner claim, makes every NEMO-identity recipe carry the approved prognostic
`uu_b/vv_b` pair, and lands the four required negative guards.  The ordered
stage-2 walk then reaches NEMO's hydrostatic-pressure-gradient insertion into
the momentum RHS and finds two source-association defects there: the live QCO
depth `gdept_z0` and a lossy acceleration-to-pressure round trip at the Krhs
boundary.  With the shared fix the HPG operator is bit-identical given NEMO's
own inputs (`0 / 17400` U, `0 / 17100` V) and GYRE's completed stage-2 Kaa moves
from DEBT `2.1986806906376666e-15` U / `2.2380914396075147e-15` V to AT-BAR
`1.0842021724855044e-19` on both components.  Every figure in this round was
re-run independently at the landed tip and reproduces at zero row-scale ulp.
HOLD stands: GYRE's trajectory Rule-12 comparison exposes compensating debt on
56 rows, OVERFLOW's first external departure is localized to a statement but has
no surviving one-variable owner, and LOCK needs the supplied WRITE-only substep
twin — which only the user can run — before its one U bit can be localized at
all.  The C1D result remains struck; lane 3b must remeasure with its identity
card opted in.  Stage-3/ZDF/TKE work is not entered, and nothing is merged into
the reconciled/integration line.

Updated: 2026-09-05

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

Round-14 starting tip: `b947a570e3abb01252e57eed7c34db882dec49b3`

Round-14 preregistration commit: `4802e424a00cb2701c8a659ba98f7d936d470818`

Round-14 oracle-relative criterion commit: `6cde4f2d7451d87aa862a41931c3a1aa9225ef68`

Round-14 EOS follow-up commit: `d1ec2eb3b9823e37497a7bd17148cf4f5ca39532`

Round-14 transport-boundary commit: `165ada6d5073309532bee50ec54b426168143c24`

Round-15 starting tip: `2b322c42f1c5c6d82ad6b930436b6f74295c486c`

Round-15 preregistration commit: `ca613d0d366b089f61ee03e4a34d2e3daec5161b`

Round-15 oracle-V2/census commit: `72020136f`

Round-15 scalar-libm implementation commit: `61180a677`

Round-15 eligibility-gate commit: `ed62a96ce`

Round-16 starting tip: `b9311e65de2`

Round-16 preregistration commits: `1e4a8fe97`, `3dee9024a`

Round-16 discriminator/implementation commits: `f3dc65745` through
`7da8b5169`

Round-16 attribution/ULP commits: `0ab248075`, `ae1a21357`

Round-16 slow-forcing implementation commit: `ea6ac7f0a`

Round-16 shared source-round implementation tip: `97d40d24e`

## Round-14 oracle-relative cross-card criterion

The user made this an **ASKED decision on 2026-09-04**.  The one shared
`--compare-to` implementation now records, for every scored row, the NEMO
field, the candidate field, and `abs(candidate-NEMO)` in a compressed NPZ
sidecar.  It compares cells, not row reductions.  One ulp for every cell in a
row is the row-scale value
`numpy.spacing(max(max(abs(float64(NEMO_row))), 1.0))`, the same floor used by
the campaign normalization.  The gate fails if any cell's absolute residual
grows by more than two such ulps, an AT-BAR row
becomes DEBT, or `first_over_bar` moves earlier.  Movement toward NEMO is free.
Movement against the previous legoESM output is retained in every
`field_moves` row for Rule 8, but is no longer itself a failure criterion.

The gate docstring and oracle-fidelity skill now carry Rule 12, the
compensating-error clause: a change is eligible to land only when the changed
operator is bit-exact for NEMO's own inputs on every card it touches.  If that
change makes a card worse against NEMO, it has exposed a second error: the fix
stays, the worsened row enters that card's register naming the boundary, and a
later round walks it.  It is never silently waived, reverted, or hidden behind
a per-card switch.

The three synthetic controls are behavioral and fail closed:
`test_three_row_scale_oracle_ulp_cell_worsening_fails`,
`test_one_cell_improvement_passes`, and
`test_at_bar_to_debt_flip_fails_without_field_movement`; the explicit exit-
contract test proves the 3-ulp and status plants return `1`, while the
improvement plant returns `0`.  All 15 shared-gate tests pass.

The baseline reports were generated from model commit
`57429ecf5f377ce2bf220f36bc05313cd29e0dfd` in a clean temporary worktree.
They stamp `be7f3ea1338cb69f0aed9707f6d922246c5968a9`, which adds only the new
gate plus a faithful-only harness needed to emit the cell fields; it changes
no model routine.  The current reports stamp
`165ada6d5073309532bee50ec54b426168143c24`, run production JIT on CPU with
fp64/x64, and retain all eight residual NPZs and their SHA-256 values.

Round 16 corrected the unfloored cell-local ulp used in the initial Round-14
implementation.  Re-scoring the unchanged residual sidecars gives these exact
verdicts:

```text
ORACLE_RELATIVE_COMPARE FAIL: rows=9 max_worsening_ulps=24.4375 first_over_bar='<absent>'->'<absent>' plant=None
ORACLE_RELATIVE_COMPARE PASS: rows=9 max_worsening_ulps=0.0009765625 first_over_bar='<absent>'->'<absent>' plant=None
ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=513408.375 first_over_bar={'fields': ['T', 'u', 'ssh'], 'kt': 2}->{'kt': 2, 'fields': ['T', 'u']} plant=None
ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=3.0 first_over_bar={'fields': ['u'], 'kt': 4}->{'kt': 4, 'fields': ['u']} plant=None
```

These are, respectively, OVERFLOW stage, LOCK_EXCHANGE stage, OVERFLOW
trajectory, and LOCK_EXCHANGE trajectory.  The initially reported four FAIL
verdicts do **not** all survive the requested row-scale formula: LOCK stage is
PASS because its largest absolute worsening, `2.168404344971009e-19`, is only
`0.0009765625` row ulp.  This is a correction, not a waiver.  No AT-BAR row
crosses to DEBT and neither `first_over_bar` moves earlier; OVERFLOW instead
drops SSH from its kt=2 first-over set.

### Cross-card Rule-8 and compensating-error rows

The four comparisons still contain 83 rows whose legoESM field changed.  The
complete prior-output movement, including improved-only rows, remains in the
four hashed comparison JSONs.  With the corrected row-scale definition, 32
rows have at least one cell whose NEMO residual worsens by more than two ulps.
This is the complete Rule-8/Rule-12 adverse-row enumeration.

| row | max worsening (absolute) | row ulp | max worsening (row ulp) | worsened / improved cells |
|---|---:|---:|---:|---:|
| `LOCK_EXCHANGE-zco.kt10.before.S` | `2.1316282072803006e-14` | `7.105427357601002e-15` | `3` | 100 / 0 |
| `LOCK_EXCHANGE-zco.kt8.before.S` | `2.1316282072803006e-14` | `7.105427357601002e-15` | `3` | 80 / 20 |
| `LOCK_EXCHANGE-zco.kt8.before.T` | `1.0658141036401503e-14` | `3.552713678800501e-15` | `3` | 19 / 7 |
| `LOCK_EXCHANGE-zco.kt9.before.S` | `2.1316282072803006e-14` | `7.105427357601002e-15` | `3` | 100 / 0 |
| `OVERFLOW-zps.kt1.stage2.faithful.instantaneous_u` | `4.829470157119431e-15` | `2.220446049250313e-16` | `21.75` | 44 / 158 |
| `OVERFLOW-zps.kt1.stage3.faithful.baroclinic_u` | `1.8218933306446417e-15` | `2.220446049250313e-16` | `8.205078125` | 73 / 180 |
| `OVERFLOW-zps.kt1.stage3.faithful.instantaneous_u` | `5.426215032855453e-15` | `2.220446049250313e-16` | `24.4375` | 88 / 165 |
| `OVERFLOW-zps.kt2.faithful.baroclinic_u` | `1.8218933306446417e-15` | `2.220446049250313e-16` | `8.205078125` | 73 / 180 |
| `OVERFLOW-zps.kt2.faithful.instantaneous_u` | `5.426215032855453e-15` | `2.220446049250313e-16` | `24.4375` | 88 / 165 |
| `OVERFLOW-zps.kt10.before.S` | `2.1316282072803006e-14` | `7.105427357601002e-15` | `3` | 292 / 85 |
| `OVERFLOW-zps.kt10.before.T` | `1.2434497875801753e-13` | `3.552713678800501e-15` | `35` | 370 / 192 |
| `OVERFLOW-zps.kt10.before.ssh` | `4.417577414983498e-13` | `2.220446049250313e-16` | `1989.5` | 25 / 34 |
| `OVERFLOW-zps.kt10.before.u` | `1.1399955979207732e-10` | `2.220446049250313e-16` | `513408.375` | 952 / 1948 |
| `OVERFLOW-zps.kt2.before.u` | `5.426215032855453e-15` | `2.220446049250313e-16` | `24.4375` | 88 / 165 |
| `OVERFLOW-zps.kt3.before.u` | `1.4510094514808003e-14` | `2.220446049250313e-16` | `65.34765625` | 123 / 363 |
| `OVERFLOW-zps.kt4.before.ssh` | `2.853273173286652e-14` | `2.220446049250313e-16` | `128.5` | 6 / 20 |
| `OVERFLOW-zps.kt4.before.u` | `1.405629085349247e-13` | `2.220446049250313e-16` | `633.0390625` | 168 / 639 |
| `OVERFLOW-zps.kt5.before.T` | `1.0658141036401503e-14` | `3.552713678800501e-15` | `3` | 122 / 71 |
| `OVERFLOW-zps.kt5.before.ssh` | `2.758904216193514e-14` | `2.220446049250313e-16` | `124.25` | 10 / 27 |
| `OVERFLOW-zps.kt5.before.u` | `2.591154721343081e-12` | `2.220446049250313e-16` | `11669.5234375` | 306 / 792 |
| `OVERFLOW-zps.kt6.before.T` | `2.842170943040401e-14` | `3.552713678800501e-15` | `8` | 145 / 51 |
| `OVERFLOW-zps.kt6.before.ssh` | `2.425837308805967e-14` | `2.220446049250313e-16` | `109.25` | 7 / 36 |
| `OVERFLOW-zps.kt6.before.u` | `1.1866722882114544e-11` | `2.220446049250313e-16` | `53442.96875` | 252 / 1141 |
| `OVERFLOW-zps.kt7.before.T` | `3.552713678800501e-14` | `3.552713678800501e-15` | `10` | 146 / 79 |
| `OVERFLOW-zps.kt7.before.ssh` | `1.2112533198660458e-13` | `2.220446049250313e-16` | `545.5` | 13 / 34 |
| `OVERFLOW-zps.kt7.before.u` | `2.8834767906715797e-11` | `2.220446049250313e-16` | `129860.25` | 336 / 1398 |
| `OVERFLOW-zps.kt8.before.T` | `6.394884621840902e-14` | `3.552713678800501e-15` | `18` | 215 / 157 |
| `OVERFLOW-zps.kt8.before.ssh` | `2.7161606297454455e-13` | `2.220446049250313e-16` | `1223.25` | 15 / 36 |
| `OVERFLOW-zps.kt8.before.u` | `5.060207114437887e-11` | `2.220446049250313e-16` | `227891.46875` | 564 / 1547 |
| `OVERFLOW-zps.kt9.before.T` | `1.0835776720341528e-13` | `3.552713678800501e-15` | `30.5` | 311 / 216 |
| `OVERFLOW-zps.kt9.before.ssh` | `3.743116927523715e-13` | `2.220446049250313e-16` | `1685.75` | 23 / 32 |
| `OVERFLOW-zps.kt9.before.u` | `7.7805727138891e-11` | `2.220446049250313e-16` | `350405.84375` | 777 / 1726 |

The DINO developed-state single-step HPG consumer exists and ran in 21 s,
well below the 20-minute limit.  It is a **consistency probe, not a match**.
At the exact halo alignment both U
(`n=335145`) and V (`n=340271`) report correlation `1.000000000` and
`rms(lego)/rms(NEMO)=1.000000000`; its transcript is retained and hashed.

## Round-14 EOS follow-ups

The new hermetic CI test fixes five TEOS-10 `prd` uint64 patterns from the
accepted oracle record and uses only embedded T/S/depth literals—no `/data`
or oracle executable.  It passes under production CPU JIT/x64 and will fail if
a future XLA peephole changes those bits.  The EOS docstring now says exactly
what the saved optimized-HLO census showed: 105 source barriers become zero
optimized `optimization_barrier` operations; the finite/select IEEE identity,
not the barrier, is what survives.  The isomorphism map adds a **PLAUSIBLE**
debt list, without changing physics, for barrier-only QCO/e3f recurrences in
`vertical.py`, `barotropic_common.py`, `barotropic_latlon_cgrid.py`, the PE HPG
literal, `implicit_solver.py`, and `gm_redi_latlon_cgrid.py`.

## Round-14 `un_adv/vn_adv` boundary walk

NEMO calls `stp_2D(kstp,Nbb,Nbb,Naa,Nrhs)` before RK stage 1
(`stprk3.F90:186`); that dispatches `dyn_spg_ts` with `Kbb` for both before
and now (`stp2d.F90:281`).  The RK3 arm seeds external SSH, U, V, and live
depths from `Kmm=Kbb` (`dynspg_ts.F90:485-493`), zeros `un_adv/vn_adv` at
`:506-510`, forms the resolved vector-invariant slow forcing from the
reference-depth mean of `Krhs` at `:310-345`, and applies the 2-D Coriolis,
drag, and surface terms at `:355-401`.  The resolved card has
`ln_dynadv_vec=.true.` (`round14_oracle_advmean_v1/namelist_cfg:161`).

Inside each of 50 substeps, NEMO forms the metric transport at
`dynspg_ts.F90:605-611`, then uses `za2=wgtbtp2(jn)` and accumulates
`za2*zhU*r1_e2u` / `za2*zhV*r1_e1v` in that written association at `:639-642`.
It divides by `SUM(wgtbtp2)` at `:843-844` and applies LBC at `:852-854`.
Stage 1 consumes those NOW-level means in `zub/zvb` and then the native
transports at `stprk3_stg.F90:262-274`.  The time-level registry names the new
record `now`; no literal `Nbb/before` label is emitted.

The config-local `MY_SRC/dynspg_ts.F90` extension wrote the divisor, every
weight, reciprocal face metric, and for each substep the accumulator entry,
midpoint velocity, face depth, metric transport, and accumulator exit, plus
pre/post-LBC means.  It changed no shipped NEMO source.  A serial diagnostic
build temporarily needed `key_mpi_off` because the sandbox denied MPI socket
creation; the config was restored to `key_qco key_vco_1d3d key_RK3` after the
run.  The ordinary stage-1 record hash is
`35e6892b799aeaf8d06d4affcd71b5ba0c71dc41bc0e8970c033459c46cd1402`,
exactly the Round-13 control, so the WRITE-only instrumentation is bit-
identity preserving.

| ordered boundary | U result | V result | disposition |
|---|---:|---:|---|
| 50 weights and divisor 50 | 0 ulp | 0 ulp | exact |
| substep 1 sum entry/velocity/depth/transport/sum exit | all exact | all exact | averaging path exact |
| substep 2 face depth | exact | exact | earlier `eta_exit` difference is not live here |
| substep 2 midpoint velocity | `1.728794245346027e-18`, 580 cells | `1.7296412782932813e-18`, 570 cells | **first live non-bit-exact operand** |
| substep 2 metric transport | `7.885319064371288e-10` | `7.885319064371288e-10` | propagated at the expected metric scale |
| final mean before/after LBC | `1.2045919817182948e-13`, 580 cells | `5.495603971894525e-14`, 570 cells | LBC zero-move; original boundary reproduced |
| literal accumulation fed NEMO operands | exact | exact | association/order/weights/division exonerated |

The external frame census sees an earlier numerical difference at substep-1
`eta_exit` (`3.3881317890172014e-21`, 600 cells), but substep-2 face depth is
bit-exact, so that departure does not enter this transport boundary.  At the
live velocity update, the slow-forcing operand differs by
`6.002022524684769e-21`; over the 288-second substep that scales to
`1.7286e-18`, matching the observed `1.7288e-18` velocity residual.

The one-variable arm reconstructs the production vector-form update exactly,
then substitutes only NEMO's dumped slow forcing.  The faithful reconstruction
is bit-identical to production.  The U residual moves by 1.0× and falls from
`1.728794245346027e-18` to `1.0587911840678754e-22` (1 ulp); V moves by
0.999510× and falls from `1.7296412782932813e-18` to
`8.470329472543003e-22` (1 ulp).  This is
**CONFIRMED_FIRST_DIVERGENCE_UPSTREAM_SLOW_FORCING**.  It is not a claim that
slow forcing is the sole substep error, and it does not exonerate the later
external-mode operands.  The averaging hypothesis and weight hypothesis are
**REFUTED**; the mean-to-stage handoff is exact.  No shared physics change is
made, so the Round-13 GYRE kt=1…10 register remains numerically current and no
post-landing sweep is claimed.  The diagnostic exits `1` for honest DEBT; its
one-bit planted accumulator mutation also exits `1` and moves the first
reported boundary to substep-1 U `sum_exit`.


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
| horizontal metric | `0 / 17,400` | `0 / 17,100` | bit-exact |
| Kmm face thickness | `0 / 17,400` | `0 / 17,100` | bit-exact |
| Kmm velocity | `0 / 17,400` | `0 / 17,100` | bit-exact |
| mask | `0 / 17,400` | `0 / 17,100` | bit-exact |
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

That was the Round-13 **reduction-level** statement under the then-current
gate.  It is retained as historical provenance but is superseded for landing
decisions by Round 14's cellwise oracle-relative result above, which exposes
worsened cells inside rows whose maximum residual improved.

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

## Historical pending D boundary and round-12 cross-card oracle scoring

Independent re-review returned **SHIP** for round 10 and clarified that this
gate compares against the previous legoESM output, not directly against NEMO.
Round-11 independent review was still running when round 12 was dispatched.
The retained D criterion is therefore an **OPEN USER DECISION**.  Round 12
does not modify that gate or criterion; it reports both its direction-blind
compare-to result and direct oracle-relative residuals.

This describes the Round-12 state only.  The user resolved the decision on
2026-09-04; the Round-14 criterion and results at the start of this receipt
are authoritative.

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
| compare-to gate or criterion change | forbidden round 12; explicitly ASKED round 14 | changed once, shared, to cellwise oracle-relative; prior-output motion disclosed only |
| compressed per-cell NEMO/candidate/residual sidecars | ASKED round 14 | landed for shared stage and trajectory gates; all eight before/after NPZs hashed |
| 3-ulp-worse / improve / AT-BAR-to-DEBT controls | ASKED round 14 | behavioral exit contracts pass: FAIL / PASS / FAIL |
| compensating-error clause in gate and oracle-fidelity skill | ASKED round 14 | landed as Rule 12; no per-card waiver or switch |
| OVERFLOW and LOCK stage plus kt=1…10 cellwise comparisons | ASKED round 14 | all four run; FAIL on exposed cells, no downward status crossing or earlier first-over |
| DINO developed-state single-step HPG third consumer | ASKED if under 20 minutes | CONSISTENCY PROBE, not a match; RUN in 21 s wall, U/V corr and ratio both 1.000000000 |
| hermetic five-sample `prd` hex test | ASKED round-12 follow-up | landed and passes without `/data`; pins production-JIT XLA bits |
| optimized-HLO EOS docstring correction | ASKED round-12 follow-up | corrected: barriers disappear, finite/select identity survives |
| barrier-only recurrence debt census | ASKED round-12 follow-up | six requested shared surfaces listed PLAUSIBLE; no recurrence changed |
| stage-3 transport remeasurement | ASKED only if EOS and stage 2 clear | NOT ENTERED; corrected Kaa remains DEBT |
| GYRE tra_zdf/dyn_zdf matrix walk | ASKED only if stages 2 and 3 clear | NOT ENTERED |
| source-ordered stage-1 tracer routine checkpoints and 9-T/9-S cell census | ASKED round 13 | config-local WRITE-only; all first differ after `tra_adv` |
| transcendental and FCT candidates | ASKED round 13 | REFUTED_BY_EXECUTED_SOURCE at stage 1; no transcendental remedy proposed |
| complete oracle stage-1 transport arm | ASKED round 13 | CONFIRMED causal; makes T/S stage update bit-exact |
| native metric-bearing zF product association | ASKED round 13 after operand split | landed shared S-21 identity; literal statement confirmed at 0 ulp, but not the complete owner |
| stage-1 transport metric/e3/velocity/mask/corrected-velocity split | ASKED owner walk | first bad compound operand is corrected velocity; `un_adv/vn_adv` first bad primitive input |
| config-local per-substep `un_adv/vn_adv` accumulator dump | ASKED round 14 | WRITE-only, ordinary stage hash unchanged; weights/association/division/LBC exact |
| weighted-mean association and weight hypotheses | ASKED round 14 | REFUTED by exact operands, literal partial sums, and final oracle-input replay |
| one-variable oracle slow-forcing arm | ASKED round 14 boundary bisection | CONFIRMED causal first divergence; U/V velocity residuals fall to 1 ulp |
| shared slow-forcing physics fix | conditional on literal owner | NOT LANDED; exact source operand inside the slow-forcing construction remains the next boundary |
| temporary serial `key_mpi_off` diagnostic build | UNASKED operational necessity | used only because sandbox MPI sockets failed; config restored; ordinary output identity checked |
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

Round 14 passes all 15 oracle-relative move-gate tests, including the three
asked plants, and the hermetic five-sample `prd` bit-pattern test.  The full
barotropic-continuity/drag plus WS-RK3 tracer run passes `50 passed` in
`742.73 s` on CPU; the time-level and single-implementation guards add
`61 passed` in `4.39 s`.  The faithful GYRE advective-mean diagnostic exits `1`
because the scientific boundary remains DEBT, and its planted run also exits
`1` after moving the first failure to one deliberately changed cell.  Fresh
OVERFLOW/LOCK stage and trajectory gates likewise exit `1` for the disclosed
cellwise compensating-error debts, not a harness failure.  The DINO probe
exits `0`.

The Round-14 artifact manifest hashes the baseline sweep artifacts, all eight
cell-residual sidecars, four comparison reports, the DINO transcript, the
config-local oracle record/source/deck, and faithful/planted walk reports.
Round-13 independent review is running in parallel; every Round-14 result is
a new Codex-internal measurement.  Independent review remains outstanding and
no dual-review claim is made.

## Round 15 — scalar-math oracle and eligibility STOP

User Decision 4 is **ASKED (2026-09-04)**.  NEMO's `-O3` executable called
glibc's vector-math entry points, whose EXP/TANH last bits are not the scalar
library's last bits.  This round therefore rebuilt GYRE without tree
vectorization and added one shared precision-policy implementation for scalar
system-libm EXP/TANH.  It did not change the oracle-relative cross-card gate,
merge a branch, or add a GYRE/card-specific physics arm.

### Oracle V2 build and corrected preregistration result

`GYRE_OMIP_L2_P3_SM` was created from shipped `GYRE_PISCES` with
`arch-conda-scalarmath.fcm`, then received SHA-verified copies of the existing
instrumented configuration's `EXP00`, `MY_SRC`, and CPP deck.  The deck remains
`key_qco key_vco_1d3d key_RK3`.  The arch SHA-256 is
`132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561`
and its flags are `-fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer
-ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`.
The pre-instrument V2 executable is `e1b16b90…`; `nm -D` reports zero `_ZGV*`
symbols.  The exact vector-control executable is `16cd9192…` and retains five
such linked symbols.  Both completed ten CPU steps.

The old oracle is retained at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round15_oracle_v1_vectorized_exact`.
V2 is retained at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round15_oracle_v2_scalarmath`;
the SBC-instrumented V2 is separately retained with suffix `_sbc`.  V1 is
flagged non-certifying rather than deleted.

The preregistration correctly predicted that `kt=1` entry, EOS/HPG, and
constant ENE coefficients need no changed transcendental.  Round 16 retracts
the inference that disabling tree vectorization changed ordinary loops and
reductions.  The committed source-ancestry discriminator assigns every one of
the 41 DIFFERENT records to an executed per-step transcendental ancestor and
finds zero unattributed records.  The five pre-forcing/source-invariant controls
(`ENE coefficients`, `rhs`, both HPG records, and `kt=1` step entry) are
byte-identical.  The change is therefore not described as compiler-wide.
The briefing also suspected the `zdftke` EXP selected by `nn_etau=1`; the
resolved GYRE namelist actually has `nn_etau=0`
(`EXP00/namelist_cfg:220`, `output.namelist.dyn:311`), so that arm is dead.

The committed byte census names the first changed source-stream payload:

| record | V1→V2 | first differing field |
|---|---|---|
| `oracle_bt_advmean_operands_kt00000001.bin` | DIFFERENT | `substep[2].metric_u` |
| `oracle_bt_drag_operands_kt00000001.bin` | DIFFERENT | `substep[2].un_e` |
| `oracle_bt_ene_coeff_kt00000001.bin` | IDENTICAL | — |
| `oracle_bt_frames_kt00000001.bin` | DIFFERENT | `uu_b[111]` |
| `oracle_bt_frames_kt00000002.bin` | DIFFERENT | `uu_b[111]` |
| `oracle_bt_frames_kt00000003.bin` | DIFFERENT | `uu_b[111]` |
| `oracle_bt_frames_kt00000004.bin` | DIFFERENT | `uu_b[111]` |
| `oracle_bt_frames_kt00000005.bin` | DIFFERENT | `uu_b[111]` |
| `oracle_bt_frames_kt00000006.bin` | DIFFERENT | `uu_b[111]` |
| `oracle_bt_frames_kt00000007.bin` | DIFFERENT | `uu_b[111]` |
| `oracle_bt_frames_kt00000008.bin` | DIFFERENT | `uu_b[111]` |
| `oracle_bt_frames_kt00000009.bin` | DIFFERENT | `uu_b[111]` |
| `oracle_bt_frames_kt00000010.bin` | DIFFERENT | `uu_b[111]` |
| `oracle_bt_substeps_kt00000001.bin` | DIFFERENT | `substep[1].slow_u` |
| `oracle_qsr_stage3_kt00000001.bin` | DIFFERENT | `qsr` |
| `oracle_rhs_kt00000001.bin` | IDENTICAL | — |
| `oracle_rkstage1_transport_operands_kt00000001.bin` | DIFFERENT | `zub` |
| `oracle_rkstage2_ene_operands_kt00000001.bin` | DIFFERENT | `zwx[k=1]` |
| `oracle_rkstage2_eos_operands_kt00000001.bin` | DIFFERENT | `T` |
| `oracle_rkstage2_hpg_literal_kt00000001.bin` | IDENTICAL | — |
| `oracle_rkstage2_hpg_operands_kt00000001.bin` | IDENTICAL | — |
| `oracle_rkstage2_operands_kt00000001.bin` | DIFFERENT | `Kmm_u` |
| `oracle_rkstage2_preupdate_kt00000001.bin` | DIFFERENT | `Krhs_u` |
| `oracle_rkstage2_terms_kt00000001.bin` | DIFFERENT | `after_vorticity_u` |
| `oracle_rkstage3_wzv_kt00000001.bin` | DIFFERENT | `ww_pre_aimp` |
| `oracle_rktracer_operands_kt00000001_s1.bin` | DIFFERENT | `zFu` |
| `oracle_rktracer_operands_kt00000001_s2.bin` | DIFFERENT | `zFu` |
| `oracle_rktracer_stage3_kt00000001.bin` | DIFFERENT | `after_advection_T` |
| `oracle_stage_kt00000001_s1.bin` | DIFFERENT | `T` |
| `oracle_stage_kt00000001_s2.bin` | DIFFERENT | `T` |
| `oracle_stage_kt00000001_s3.bin` | DIFFERENT | `T` |
| `oracle_step_entry_kt00000001.bin` | IDENTICAL | — |
| `oracle_step_entry_kt00000002.bin` | DIFFERENT | `T` |
| `oracle_step_entry_kt00000003.bin` | DIFFERENT | `T` |
| `oracle_step_entry_kt00000004.bin` | DIFFERENT | `T` |
| `oracle_step_entry_kt00000005.bin` | DIFFERENT | `T` |
| `oracle_step_entry_kt00000006.bin` | DIFFERENT | `T` |
| `oracle_step_entry_kt00000007.bin` | DIFFERENT | `T` |
| `oracle_step_entry_kt00000008.bin` | DIFFERENT | `T` |
| `oracle_step_entry_kt00000009.bin` | DIFFERENT | `T` |
| `oracle_step_entry_kt00000010.bin` | DIFFERENT | `T` |
| `oracle_tracer_transport_kt00000001_s3.bin` | DIFFERENT | `zFu` |
| `oracle_transport_kt00000001_s1.bin` | DIFFERENT | `zFu` |
| `oracle_transport_kt00000001_s2.bin` | DIFFERENT | `zFu` |
| `oracle_transport_kt00000001_s3.bin` | DIFFERENT | `zFu` |
| `oracle_zdf_entry_kt00000001.bin` | DIFFERENT | `avm` |

The exhaustive ancestry register partitions those 41 table rows as follows.
The 27 `usrdef_sbc`-only rows are both advmean/drag records, all ten
`oracle_bt_frames`, the barotropic-substep record, stage-1 transport operands,
stage-2 ENE/EOS/general/preupdate/terms records, stage-3 WZV, both tracer
operand records, all four transport records, and the ZDF-entry record.  Their
first transcendental ancestor is seasonal SIN/COS, through stress to slow
forcing/external mode (or, for ZDF, through stress magnitude to TKE).  The one
QSR record has `usrdef_sbc` COS and `traqsr` 2BD EXP as ancestors.  The other
13 are tracer-stage3, all three stage records, and step-entry records kt=2…10;
their ancestors are seasonal SBC SIN/COS plus 2BD EXP through the propagated
tracer state.  Thus each DIFFERENT row above has a named ancestor; none lacks
one.  The discriminator's unregistered-record plant exits `1`.

The census plant changes one payload byte and exits `1`.  The legacy
momentum-side zFw record is written before vector-invariant `tra_adv_trp`
initializes zFw (`MY_SRC/stprk3_stg.F90:312-345,616-624`).  The gate formerly
labelled that row UNINFORMATIVE only after first trying to score its accidental
contents.  It now finite-checks and scores zFu/zFv while recording the raw zFw
sentinel count and never treating it as science.

### Production-JIT register against V2

The current model at `72020136f`, before selecting scalar libm, was rerun in
production JIT on CPU/fp64 against V2.  `kt=1` T/S remain bit-exact; at-rest
u/v/SSH remain UNINFORMATIVE.  `first_over_bar` remains `kt=2` for all five
fields.  The re-pinned normalized maximum residuals are:

| kt | T | S | u | v | SSH |
|---:|---:|---:|---:|---:|---:|
| 1 | `0` | `0` | `0` | `0` | `0` |
| 2 | `1.3614736849003888e-12` | `2.2181101297999213e-14` | `9.4840899545731204e-7` | `8.9879925923354873e-7` | `2.3733185555707692e-15` |
| 3 | `3.7223422534148244e-4` | `3.683246743167771e-5` | `9.3116789131404247e-3` | `4.7190775913080185e-3` | `7.0746696494725179e-7` |
| 4 | `1.0594736858280278e-3` | `8.2257404918755151e-5` | `1.4182404011401410e-2` | `1.5947842533309442e-2` | `5.0601188622632215e-7` |
| 5 | `3.0537433514018029e-3` | `8.611826496841798e-5` | `2.0733890538957842e-2` | `4.3797666154583352e-2` | `6.7018566281600023e-5` |
| 6 | `3.8176772033418527e-3` | `1.2875362101381311e-4` | `3.1134730338654347e-2` | `6.0219593254000642e-2` | `1.6310220843274655e-4` |
| 7 | `4.5466683327810668e-3` | `1.2244044313748583e-4` | `4.0041405602208285e-2` | `6.6315774741028066e-2` | `2.0871415929327739e-4` |
| 8 | `5.0983786157443181e-3` | `1.3609844417961678e-4` | `4.7180060361300936e-2` | `2.4527162798603175e-2` | `2.5108257580629123e-4` |
| 9 | `5.4450653406584275e-3` | `1.4711821963103337e-4` | `5.2560399634331935e-2` | `1.4533611363184411e-2` | `2.1202997678594127e-4` |
| 10 | `5.6366385875687508e-3` | `1.5483121735342806e-4` | `5.6249872588641961e-2` | `1.1173366024903889e-2` | `2.125547585165961e-4` |

V1→V2 changes 20 maximum-residual rows for the worse, 23 for the better, and
leaves seven identical.  This is oracle-toolchain movement; it is still Rule-8
disclosure, not a new operator owner.  All worsened rows are:

| field | kt | V1 absolute max | V2 absolute max | ratio |
|---|---:|---:|---:|---:|
| u | 2 | `9.4840899545725783e-7` | `9.4840899545731204e-7` | `1.00000000000006` |
| SSH | 2 | `2.3724511938327808e-15` | `2.3733185555707692e-15` | `1.00036559729458` |
| T | 3 | `8.7413117393815298e-3` | `8.7413117395023221e-3` | `1.00000000001382` |
| SSH | 3 | `7.0746696485140832e-7` | `7.0746696494725179e-7` | `1.00000000013547` |
| SSH | 4 | `5.0601188203002606e-7` | `5.0601188622632215e-7` | `1.00000000829288` |
| T | 5 | `7.1740734435106646e-2` | `7.1740734435113751e-2` | `1.00000000000010` |
| u | 5 | `2.0733890538956697e-2` | `2.0733890538957842e-2` | `1.00000000000006` |
| v | 5 | `4.3797666154580465e-2` | `4.3797666154583352e-2` | `1.00000000000007` |
| SSH | 5 | `6.7018566280471585e-5` | `6.7018566281600023e-5` | `1.00000000001684` |
| SSH | 6 | `1.6310220830087775e-4` | `1.6310220843274655e-4` | `1.00000000080850` |
| S | 7 | `4.5104093018011326e-3` | `4.5104093128500722e-3` | `1.00000000244965` |
| u | 7 | `4.0041405602201319e-2` | `4.0041405602208285e-2` | `1.00000000000017` |
| v | 7 | `6.6315774741020544e-2` | `6.6315774741028066e-2` | `1.00000000000011` |
| T | 8 | `1.1973658446773072e-1` | `1.1973658446773427e-1` | `1.00000000000003` |
| S | 8 | `5.0135224231979691e-3` | `5.01352242321218e-3` | `1.00000000000283` |
| S | 9 | `5.4194104725056036e-3` | `5.419410472512709e-3` | `1.00000000000131` |
| u | 9 | `5.2560399634324524e-2` | `5.2560399634331935e-2` | `1.00000000000014` |
| S | 10 | `5.7034808159812656e-3` | `5.7034808159954764e-3` | `1.00000000000249` |
| v | 10 | `1.1173366024871394e-2` | `1.1173366024903889e-2` | `1.00000000000291` |
| SSH | 10 | `2.1255475816557047e-4` | `2.125547585165961e-4` | `1.00000000165146` |

### One shared scalar-libm policy

Pre-implementation search found no `pure_callback`, `ctypes`, or `CDLL` under
`packages/` or `src/`.  `PrecisionPolicy` now has one `native|libm`
transcendental selector, default `native`; every NEMO testcase card explicitly
requires `libm`.  `legoesm.core.transcendentals` loads the same `libm.so.6`
soname as NEMO (`/lib64/libm.so.6`, glibc 2.34), calls scalar C `exp`/`tanh`
once per array element via `jax.pure_callback`, rejects non-CPU execution, and
defines custom JVPs whose reverse rules are obtained by JAX transposition.
The two-band NEMO path uses the shared EXP.  The reusable GYRE profile helper
uses the shared TANH; the actual testcase card already constructs its static
IC through scalar Python `math.tanh`.  Executed GYRE SIN/COS remain JAX-native
because the machine discriminator found them bit-identical; the resolved TKE
EXP is inactive (`nn_etau=0`).

For each of EXP and TANH, the 100,000-input test requires scalar-libm output
bits to equal an independent ctypes call and requires the native JIT path to
differ somewhere.  Separate tests require libm eager/JIT bit identity and
finite-difference-consistent forward plus reverse AD.  The core precision,
transcendental, and testcase-card run is `94 passed` in `22.02 s`, CPU/x64.
The final focused rerun, extended with the one-routine-per-reference
isomorphism tripwire, is `129 passed` in `24.78 s`, CPU/x64.
Rule 1c in the oracle-fidelity skill now permits a platform/precision selector
only with compiler/library/binary provenance, one shared implementation, no
per-card guard, and preserved JIT/AD.

### Eligibility measurement and mandatory STOP

The config-local `usrdef_sbc.F90` instrument writes qsr, qns, emp, utau, and
vtau only after their source assignments
(`src/OCE/USR/usrdef_sbc.F90:109-145,161-184`).  Its final executable still has
zero `_ZGV*`.  Both production restart/history NetCDF files have bit-identical
data variables before and after instrumentation.  Forty of 46 raw diagnostic
streams are also byte-identical; the six raw differences are retained and
listed by the committed gate as uninitialized/halo diagnostic records rather
than being hidden or used as physics evidence.

The eligibility gate evaluates the same stage-3 Kmm stretch carried in NEMO's
tracer record and compares binary64 bits over wet/active cells.  Counts are
production-JIT CPU/fp64.  “Land-only” is separately disclosed and never used
to erase a wet mismatch.

| boundary / field | native differing wet cells | scalar-libm differing wet cells | scalar-libm max abs | verdict |
|---|---:|---:|---:|---|
| `qsr_2BD` increment | 16,082 | 15,891 | `1.9058241313221758e-21` | DEBT |
| `usrdef_sbc.qsr` | 0 | 0 | 0 | BIT-EXACT |
| `usrdef_sbc.qns` | 115 (+104 land-only) | 115 (+104 land-only) | `1.1368683772161603e-13` | DEBT |
| `usrdef_sbc.emp` | 490 | 490 | `6.776263578034403e-21` | DEBT |
| `usrdef_sbc.utau` | 0 | 0 | 0 | BIT-EXACT |
| `usrdef_sbc.vtau` | 0 | 0 | 0 | BIT-EXACT |

Scalar libm removes 191 wet QSR bit mismatches but leaves 15,891; qns and EMP
do not traverse EXP/TANH and do not move.  This refutes eligibility.  The gate
prints `ROUND15_ELIGIBILITY DEBT stop_required=True` and exits `1`.  Its clean
oracle-copy control has zero mismatches; a one-ulp qsr plant changes one cell
and exits `1`.

No association was tuned after seeing this result.  In particular,
`_nemo_source_round` was **not** renamed or applied to corrected-velocity/zub,
the `2.799952110443815e-17` operand was not remeasured, the slow-forcing dump
was not added, and no kt sweep was run under the ineligible scalar-libm policy.
Those are conditional work, not silently completed work.

### Round-15 ASKED / UNASKED register

| choice or action | origin | disposition |
|---|---|---|
| rebuild GYRE oracle with math-call vectorization off | ASKED Decision 4 | complete; V2 retained, zero `_ZGV*` |
| preserve and flag V1 | ASKED | complete; exact vector control retained |
| scalar `libm.so.6` EXP/TANH precision policy | ASKED | landed shared, default native, NEMO cards explicit |
| custom AD, production JIT, CPU-only enforcement | ASKED | complete; focused tests pass |
| V1/V2 byte census and V2 kt=1…10 register | ASKED | complete; 46 rows and production-JIT sweep retained |
| config-local SBC WRITE-only dump | UNASKED enabling measurement | retained only in oracle config; production fields bit-identical |
| pre-consumer zFw UNINFORMATIVE gate correction | UNASKED gate correctness | landed; no uninitialized value is scored |
| scalar-libm eligibility | ASKED | REFUTED; fail-closed STOP |
| public `nemo_source_round` and corrected-velocity application | conditional ASKED | NOT ENTERED after STOP |
| slow-forcing boundary walk/fix | conditional ASKED | NOT ENTERED after STOP |
| stage-3 transport and ZDF walks | conditional ASKED | NOT ENTERED |
| GYRE-only association, card guard, tuning after eligibility | forbidden | none added |

Round-14 independent review was still running when this dispatch began.
Every Round-15 result is a Codex-internal measurement; independent review is
outstanding and no dual-review claim is made.

## Round 16 — scalar-math discriminator and ordered slow-forcing walk

Round 16 started at `b9311e65de2`.  The discriminator was preregistered at
`1e4a8fe97` before any implementation change.  Every measurement in this
section is production JIT on CPU with x64/fp64 and the explicit scalar-libm
policy unless a row says otherwise.  Oracle V1 remains retained and flagged;
nothing was deleted or overwritten.  The config-local `MY_SRC` extensions are
WRITE-only, and no shipped NEMO source was modified.

### Discriminator: H1 confirmed, H2 refuted

The pre-fix NumPy transcription preserves NEMO's statement boundaries for the
two-band constants, exponential arguments, cutoff, and accumulation
(`traqsr.F90:621,629-630,642` in the executed compilation listing; the same
statements are `src/OCE/TRA/traqsr.F90:665-683`).  The nine seasonal
trigonometric sites and the qsr/qns/emp/stress assemblies follow
`cfgs/GYRE_OMIP_L2_P3_SM/MY_SRC/usrdef_sbc.F90:86-146,163-187`.

The QSR diagnostic is an accumulated delta, not a bare tendency.  The probe
therefore replays the candidate increment into the dumped pre-QSR `Krhs` and
compares that result with NEMO's dumped post-QSR `Krhs`.  Before the fix,
NEMO-literal replay was bit-exact in all 18,000 active cells while the then-
current shared path differed in 5,324.  The exponential arguments and scalar-
libm results themselves were bit-exact in all 2,112 first-band and 12,672
second-band sites.  NEMO-literal SBC replay was bit-exact for all five 600-cell
fields, while the current path differed in 115 wet qns cells and 490 emp
cells.  This **CONFIRMS H1** (statement association) and **REFUTES H2** for
EXP (callback delivery).

The preregistered recovered-delta row itself—`Krhs_after-Krhs_before` from the
two dumps—is **NOT bit-exact**: 4,191 of 18,000 active cells differ, with
maximum absolute difference `1.6940658945086007e-21`.  This is subtractive
cancellation noise from recovering a small increment from two rounded
accumulators, not an exact observation of the increment.  It is explicitly
unscored; the bit-exact accumulation-boundary replay above replaces it as the
certification quantity.

The route plant advances every scalar-libm return by one binary64 step.  Before
the fix it counted 43,648 EXP calls and changed the QSR result.  After extending
the same policy to the executed SIN/COS sites, it counted 43,648 EXP, 2,112
SIN, and 1,412 COS calls, and changed qsr, qns, emp, utau, vtau, and the QSR
increment.  Thus the card demonstrably traverses each callback; the plant is
behavioral and not a source-string assertion.

The post-fix discriminator now labels H1 `SUPERSEDED_BY_FIX`, rather than
misleadingly printing `NOT_CONFIRMED` after the two implementations converge.
Its pre-fix evidence remains the retained `round16/discriminator.json`,
SHA-256
`3c44294d3dd5051958931baa42a16d670bc6226723f7061522201b88bf163405`.
The four RGB-path EXP source sites in `_morel_berthon_chl_column` and
`shortwave_penetration_rgb_tendency` also route through the precision-policy
EXP, so ORCA1's RGB selection honours scalar libm.  The behavioral route test
observes all seven executions (three profile sites plus four band calls);
reverting one site reduces the count to six and fails with exit `1`.

The shared fix at `7da8b5169` applies `nemo_source_round` at the NEMO statement
boundaries, uses the same precomputed reciprocals and multiplication forms,
and sends the seasonal SIN/COS sites through the same scalar-library policy.
This is shared NEMO identity code, not a GYRE switch.  The post-fix eligibility
measurement is:

| boundary / field | pre-fix scalar-libm wet mismatches | post-fix scalar-libm wet mismatches | post-fix max abs | verdict |
|---|---:|---:|---:|---|
| `qsr_2BD` accumulated increment | 15,891 | 0 / 18,000 | `0` | BIT-EXACT |
| `usrdef_sbc.qsr` | 0 | 0 / 600 | `0` | BIT-EXACT |
| `usrdef_sbc.qns` | 115 | 0 / 600 | `0` | BIT-EXACT |
| `usrdef_sbc.emp` | 490 | 0 / 600 | `0` | BIT-EXACT |
| `usrdef_sbc.utau` | 0 | 0 / 600 | `0` | BIT-EXACT |
| `usrdef_sbc.vtau` | 0 | 0 / 600 | `0` | BIT-EXACT |

The remaining native-policy QSR difference is 1,363 active cells with maximum
`1.6940658945086007e-21` (128 ulp); native is not the certification policy.
The clean scalar-libm eligibility gate is `PASS`.  A one-ulp QSR plant changes
one certified cell, prints DEBT, and exits `1`.  This satisfies the Round-15
eligibility falsifier and releases the ordered walk.

The Oracle V1/V2 compiler-wide claim is retracted.  The exhaustive committed
ancestry probe reports `different=41 controls=5 unattributed=0`: all 41 changed
records have an executed seasonal SIN/COS or two-band EXP ancestor, while ENE
coefficients, kt=1 RHS, HPG literal/operands, and kt=1 step entry are byte-
identical controls.  Its unregistered-record plant exits `1`.  This attribution
is causal ancestry, not an assertion that every descendant's first differing
field is itself transcendental.

### Row-scale cross-card correction

Commit `ae1a21357` corrects the shared cellwise ULP definition to
`spacing(max(max(abs(oracle_row)), 1.0))`.  The in-place 32-row Rule-12 table
above is the rescore under that definition: OVERFLOW stage FAIL (5/9 rows),
LOCK stage PASS (0/9), OVERFLOW trajectory FAIL (23/50), and LOCK trajectory
FAIL (4/50).  The LOCK stage result is intentionally reported as measured,
not forced to match an expectation.  DINO remains only a consistency probe,
not a match claim.

### Public source-round helper and corrected-velocity composition

Commit `97d40d24e` promotes `_nemo_source_round` to the single public
`nemo_source_round` spelling with no compatibility alias.  It applies the
helper at each operation of the stage corrected-velocity/zub/zvb composition,
where a bare XLA optimization barrier was insufficient.  The stage-1 U
corrected-velocity maximum moves from `2.799952110443815e-17` to
`2.7985968577282083e-17`, a reduction of `1.3552527156067e-20` (0.0484%).
The V maximum remains `1.2766480581016815e-17`.  The downstream zF residuals
remain DEBT (`8.917595550883561e-10` U and `4.069988790433854e-10` V), so this
is a measured recurrence correction, not an owner claim.

The oracle-relative cross-card gates for this shared correction all pass:

```text
OVERFLOW stage: ORACLE_RELATIVE_COMPARE PASS: rows=9 max_worsening_ulps=0 first_over_bar='<absent>'->'<absent>'
LOCK stage: ORACLE_RELATIVE_COMPARE PASS: rows=9 max_worsening_ulps=0 first_over_bar='<absent>'->'<absent>'
OVERFLOW trajectory: ORACLE_RELATIVE_COMPARE PASS: rows=50 max_worsening_ulps=0 first_over_bar={'fields':['T','u'],'kt':2}->{'fields':['T','u'],'kt':2}
LOCK trajectory: ORACLE_RELATIVE_COMPARE PASS: rows=50 max_worsening_ulps=0 first_over_bar={'fields':['u'],'kt':4}->{'fields':['u'],'kt':4}
```

Each gate persisted and hashed its cellwise residual NPZ.  Movement versus the
previous legoESM output remains disclosed; it does not determine eligibility.

### Ordered slow-forcing operand walk

The second preregistration landed at `3dee9024a`.  The new V2 oracle root is
`round16_oracle_v2_slow_v2`; an earlier size-incomplete diagnostic root remains
retained and flagged.  Its ten-step history and restart data variables are
bit-identical to the Round-15 V2 oracle.  The scalar-math executable contains
zero `_ZGV*` symbols.  The config-local record dumps `Krhs`, e3 face thickness,
mask, inverse resting depth, the depth mean, post-drag forcing, and post-wind
forcing at kt=1/stage 1.

NEMO accumulates the three-dimensional RHS in routine call order at
`stp2d.F90:126-171`, forms its depth mean at `stp2d.F90:177-181`, applies the
implicit-drag term at `stp2d.F90:196`, and adds surface stress at
`stp2d.F90:198-202`.  `dynspg_ts.F90:280-300` copies the slow forcing into the
external mode and removes the separately treated Coriolis term.  The ordered
comparison is:

| boundary / arm | U max abs | V max abs | differing U / V | disposition |
|---|---:|---:|---:|---|
| initial-rest oracle `Krhs` | `0` | `0` | `0 / 0` | BIT-EXACT control |
| legoESM `Krhs` vs oracle | `1.3869160759180077e-20` | `1.3869160759180077e-20` | `17,400 / 17,100` | first non-bit-exact primitive |
| e3, mask, inverse depth | `0` | `0` | `0 / 0` | BIT-EXACT inputs |
| production depth mean | `5.9922313054079714e-21` | `5.9922313054079714e-21` | `580 / 570` | DEBT |
| NEMO-literal reduction arm | `5.022485483447278e-21` | `5.022485483447278e-21` | `580 / 570` | 16.18% causal reduction |
| oracle `Krhs` substituted alone | `0` | `0` | `0 / 0` | owner discriminator |
| post wind / slow forcing | `5.998713802234557e-21` | `5.998713802234557e-21` | `580 / 570` | DEBT |

Scaling precedes the owner label: changing only reduction association moves
`9.697458219606931e-22`, just 16.18% of the production depth-mean residual.
It is therefore **CONFIRMED_CAUSAL_CONTRIBUTOR_NOT_OWNER**.  Substituting only
NEMO's `Krhs` makes the replay bit-exact, so the first boundary is honestly
**CONFIRMED_UPSTREAM_KRHS**.  The resulting external-mode slow-forcing residual
is about `2.98e-13` relative and remains AT-BAR.  No compensating depth-mean or
slow-forcing physics change landed.  The plant forces e3 to become the first
boundary and exits `1`.

### Final production-JIT GYRE sweep

The post-fix, post-source-round V2 sweep remains DEBT.  kt=1 T/S are bit-exact;
the at-rest kt=1 u/v/SSH rows are UNINFORMATIVE.  `first_over_bar=kt=2` for all
five live fields.

| kt | T | S | u | v | SSH |
|---:|---:|---:|---:|---:|---:|
| 1 | `0` | `0` | `0` | `0` | `0` |
| 2 | `1.3614736849003888e-12` | `2.2181101297999213e-14` | `9.484089954573663e-7` | `8.987992592337385e-7` | `2.372668034267278e-15` |
| 3 | `3.7223441372679294e-4` | `3.683246743167771e-5` | `9.311678913140426e-3` | `4.7190775913080185e-3` | `7.074669649582022e-7` |
| 4 | `1.059473805182152e-3` | `8.225740462674584e-5` | `1.418240401195741e-2` | `1.594784230453221e-2` | `5.060423358578536e-7` |
| 5 | `3.0537424781430155e-3` | `8.611826431262424e-5` | `2.0733890435774838e-2` | `4.3797665645115294e-2` | `6.701851023208603e-5` |
| 6 | `3.817675809918862e-3` | `1.2875362386037564e-4` | `3.113472973279577e-2` | `6.021959208697472e-2` | `1.6310239042308892e-4` |
| 7 | `4.546666553978738e-3` | `1.2244044900930025e-4` | `4.004140471797118e-2` | `6.631577323125762e-2` | `2.08714182109976e-4` |
| 8 | `5.098376094806285e-3` | `1.3609844305104157e-4` | `4.718005909062654e-2` | `2.4527161865936703e-2` | `2.5108247682647427e-4` |
| 9 | `5.445062142555661e-3` | `1.471182174585388e-4` | `5.2560397604513136e-2` | `1.4533612266675985e-2` | `2.1203002216882037e-4` |
| 10 | `5.636634494144881e-3` | `1.548312135446306e-4` | `5.6249869570834554e-2` | `1.1173365767158869e-2` | `2.1255435596181406e-4` |

Against the Round-15 pre-libm model evaluated on the same V2 oracle, 27 of 50
maximum rows improve, nine are unchanged, and 14 worsen.  Rule 8 requires each
worsened row in place:

| field | kt | Round-15 value | Round-16 value | ratio |
|---|---:|---:|---:|---:|
| u | 2 | `9.48408995457312e-7` | `9.484089954573663e-7` | `1.000000000000057` |
| v | 2 | `8.987992592335487e-7` | `8.987992592337385e-7` | `1.0000000000002112` |
| T | 3 | `3.7223422534148244e-4` | `3.7223441372679294e-4` | `1.0000005060934694` |
| u | 3 | `9.311678913140425e-3` | `9.311678913140426e-3` | `1.0000000000000002` |
| SSH | 3 | `7.074669649472518e-7` | `7.074669649582022e-7` | `1.0000000000154783` |
| T | 4 | `1.0594736858280278e-3` | `1.059473805182152e-3` | `1.0000001126541658` |
| u | 4 | `1.418240401140141e-2` | `1.418240401195741e-2` | `1.0000000000392035` |
| SSH | 4 | `5.060118862263221e-7` | `5.060423358578536e-7` | `1.000060175723852` |
| S | 6 | `1.287536210138131e-4` | `1.2875362386037564e-4` | `1.0000000221086018` |
| SSH | 6 | `1.6310220843274655e-4` | `1.6310239042308892e-4` | `1.0000011158055069` |
| S | 7 | `1.2244044313748583e-4` | `1.2244044900930025e-4` | `1.0000000479564943` |
| SSH | 7 | `2.087141592932774e-4` | `2.08714182109976e-4` | `1.0000001093203197` |
| v | 9 | `1.4533611363184411e-2` | `1.4533612266675985e-2` | `1.0000000621656622` |
| SSH | 9 | `2.1202997678594127e-4` | `2.1203002216882037e-4` | `1.0000002140399191` |

The shared source-round correction has no measurable effect on this full sweep:
the before/after JSON artifacts are byte-identical.  The changes above are the
disclosed scalar-forcing/libm move against the same oracle, not evidence that
the slow-forcing reduction owns trajectory debt.

### Round-16 artifact and verification manifest

The committed manifest is
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round16.json`.
Key artifact hashes are:

| artifact | SHA-256 |
|---|---|
| pre-fix discriminator | `3c44294d3dd5051958931baa42a16d670bc6226723f7061522201b88bf163405` |
| post-fix discriminator | `106a778e18d3899982c7acf1e547b98858cf954ca7c53071f0aab1f58a89ba83` |
| eligibility | `336cd2026e4d51bec48de7ada71c449bdb0ecffbc0b1bac2867f3f226e4c5978` |
| V1/V2 attribution | `ee29acca0b381abacb68d24642590b4f9cbcfaaddd5e522aafe7776b23177cac` |
| row-scale cross-card rescore | `c6157e655c0757e5715dd3966326b648f89b7437231e7562eabdf894ad8550c6` |
| corrected velocity | `dd675a15d0b7b0cd756a8f6c383c73209fb6a402707dfbf4fa5979092d080bf3` |
| slow-forcing walk | `8b418d220147a604d9ca4a32796979c033c3ad2d40352b40767ad03ecbf5b02c` |
| final production-JIT sweep | `83f411ca736e6870c6b62a3a3c8e73e6751931c7eae8f4a1ef4ee685d4c76b4c` |

The final focused scalar-forcing/transcendental/gate suite is `60 passed` in
`8.13 s`.  The source-round follow-up run of the WS-RK3 tracer and GYRE phase-3
tests is `42 passed` in `648.56 s`.  A broader recipe invocation encountered
five existing paired-integrator validation failures and is not claimed as a
pass.

### Round-16 ASKED / UNASKED register

| choice or action | origin | disposition |
|---|---|---|
| V2 literal NumPy discriminator before code change | ASKED | complete; H1 confirmed, EXP-route H2 refuted |
| callback route plant for EXP/SIN/COS | ASKED | complete; every executed route counted and poisoned |
| shared NEMO-literal QSR/SBC fix after H1 | conditional ASKED | landed; all six eligibility boundaries bit-exact |
| extend scalar-libm policy to executed SIN/COS | UNASKED required mechanism | landed shared; no card guard |
| attribute all 41 V1/V2 changed records | ASKED | complete; zero unattributed, compiler-wide claim retracted |
| row-scale oracle ULP correction | ASKED | landed; 32 Rule-12 rows rewritten in place |
| DINO correlation wording | ASKED | relabelled consistency probe, not a match |
| public `nemo_source_round`, no alias | carried ASKED from Round 15 | complete |
| source-round corrected velocity composition | carried ASKED from Round 15 | landed shared; cross-card gates pass |
| slow-forcing operand walk after eligibility | conditional ASKED | complete; upstream `Krhs` is first boundary |
| literal depth-reduction change | UNASKED candidate arm | measured 16.18%; not landed because it is not owner |
| stage-3 transport continuation | conditional ASKED | not entered; upstream `Krhs` register is open |
| ZDF matrix walk | explicitly deferred | not entered |
| GYRE-only guard, tuning, or shipped NEMO edit | forbidden | none |

Round-15 independent review was still running during this dispatch.  Every
Round-16 result is a Codex-internal measurement; independent review remains
outstanding and no dual-review claim is made.

## Round 17 — shared rounding convergence and stage-1 HPG owner

Round 17 started at `8b6ec4efe31e08d0f8db9c1e1378f7049ffc2409`.
The HPG prediction was committed before measurement at `f7409dddb`; all rows
below use Oracle V2 with production JIT, CPU, x64/fp64, and the card's explicit
`transcendentals="libm"` policy.

### One shared per-statement rounding owner

The user corrected the source reference during the convergence check.  The
canonical core module was introduced by ice-lane commit `86a8eb21d18`, not the
earlier `e2ad2c30629` scalar-libm import commit.  Commit `2a7b1f7ae` checks out
`packages/core/legoesm/core/source_rounding.py` verbatim from `86a8eb21d18`,
removes the ocean definition, and imports
`legoesm.core.source_rounding.nemo_source_round` everywhere.  There is one
executable implementation.  The former ocean and canonical core bodies were
computationally identical; only annotations, import spelling, formatting, and
the docstring differed.

The new direct core test covers normal values, both signed zeros, positive and
negative binary64 subnormals, infinities, production JIT parity, and the
identity gradient on finite inputs.  The hermetic GYRE prd bit-pattern pin is
unchanged.  The production EOS parity gate reports:

```text
jacobian          0 ulp  BIT-EXACT
thickness         0 ulp  BIT-EXACT
prd               0 ulp  BIT-EXACT
pressure_anomaly  0 ulp  BIT-EXACT
```

The convergence/unit subset is `4 passed` (`2` direct core tests, prd pin, and
EOS/HPG/transport parity).  No scientific choice or card switch changed.

### Stage-1 source order and zero terms

At kt=1 from rest, NEMO calls EOS and HPG first
(`stp2d.F90:126-128`), then lateral diffusion (`:130-131`), vorticity/Coriolis
(`:144-146`), WZV (`:155`), and vector-invariant KE-gradient plus vertical
advection (`:159-165`).  The dumped `uu/vv(:,:,:,Krhs)` entry velocity is
identically zero.  Consequently LDF, planetary/relative-vorticity, KE-gradient,
and vertical-advection contributions are identically zero: the complete
stage-1 3-D `Krhs` is HPG alone.

The shared operator now preserves each written NEMO operation in
`dynhpg.F90:340-390`: `zcoef0=-grav*0.5`; surface `zhpi/zhpj`; surface
`zuap/zvap`; `Krhs=zhpi+zuap`; the top-down `zhpi/zhpj` recurrence; each
level-local terrain correction; and the final sum.  It uses NEMO's multiply-
by-reciprocal metric forms and materializes each add, subtraction, and
multiplication with the shared helper.  The operator remains the single
registered implementation of `hpg_sco`; `_source_round=False` is a private
test-only ablation, not a card or scheme choice.

The one-variable result is decisive at the predicted boundary:

| boundary | pre-fix max abs | source-literal max abs | source-literal differing cells | disposition |
|---|---:|---:|---:|---|
| stage-1 U/V `Krhs` | `1.3869160759180077e-20` | `0` | `0 / 17,400 U; 0 / 17,100 V` | CONFIRMED HPG association owner |
| e3, mask, reciprocal depth | `0` | `0` | `0 / 17,400 U; 0 / 17,100 V` | BIT-EXACT inputs |
| reference-depth reduction | `5.9922313054079714e-21` | `0` | `0 / 580 U; 0 / 570 V` | BIT-EXACT after HPG fix |
| post-drag forcing | `5.9922313054079714e-21` | `0` | `0 / 580 U; 0 / 570 V` | BIT-EXACT |
| post-wind U | `5.998713802234557e-21` | `6.617444900424222e-24` | `415` | DEBT, 3 ULP |
| post-wind V | `5.998713802234557e-21` | `6.617444900424222e-24` | `409` | DEBT, 4 ULP |

The pre-fix recurrence arm is non-vacuous on deterministic fp64 inputs and
reproduces bit movement; removing the source materialization makes its unit
test fail.  HPG eager/JIT parity is bit-exact for both U and V (0 ulp).  The
planted live e3 operand becomes the first failing boundary and exits `1`.

The already proposed literal depth reduction is now a zero-movement arm: both
the production and independent NEMO-order replay are bit-exact.  It was not
landed.  The first remaining boundary is therefore honestly
**CONFIRMED_POST_WIND_FIRST_DIVERGENCE**; its owner is **UNMEASURED**.  Since the
complete stage-1 chain is not bit-exact, the brief's condition for a new
kt=1…10 sweep, OVERFLOW/LOCK Rule-12 gates, stage-2 Kaa, and stage-3 transport
was not met.  Those current-tip results are explicitly UNMEASURED, not carried
forward from Round 16.

The HPG/isomorphism subset is `41 passed` in `9.46 s`; the corrected census
schema has `4 passed` in `0.05 s`.  A broader 62-test invocation produced
`57 passed, 5 failed`: all five failures occur before this operator because
synthetic `nemo_sco` fixtures omit the now-required oracle `e3w_0`.  They are
not claimed as HPG regressions or as a passing suite.  The final combined
focused rerun is `45 passed` in `10.94 s`.

### V1/V2 record-schema correction

The compiler-wide V2 claim was already retracted in Round 16 and DINO was
already labelled a consistency probe, not a match.  The remaining schema bug
is fixed by `c89f6b65c`: `oracle_bt_frames` contains, in order, `uu_b`, `vv_b`,
`un_adv`, and `vn_adv`—not SSH fields.  All ten records first differ at byte
928, `40 + 111*8`, hence `uu_b[111]`; the earlier table above is rewritten in
place.  A four-field behavioral test plants byte 111 in each payload and
requires the correct field name.  The corrected census and attribution retain
the substantive result: 41 different records, five identical controls, zero
unattributed.  Both census and unregistered-record plants exit `1`.

### Round-17 artifacts

| artifact | SHA-256 |
|---|---|
| EOS source-rounding parity | `559c427944358a82715a087ecbf6e5c68bc12deb3185905db7f66c395ad8421b` |
| HPG JIT parity | `bcfacdae524861242c9a42b63281612d5ab31e67016bf1f058a64b8b8bae1898` |
| HPG/slow-forcing walk | `1ebdbade910d1620f1078735f90ea12c950aebedb62c448c1e31c10f5804ad0b` |
| HPG/slow-forcing plant | `8d37b6e98a69c9ebee33291cf427d1c5fc9771b39f13ae10cd0d1b0d92bb36db` |
| corrected V1/V2 census | `21f8416f51c87fefc2e48fe89dbaa1f143ce8e6d785f6f2154d48a739b904694` |
| corrected attribution | `774aace672c4c136284e6e758e3ab59ad4e44be59a55cf9661660733536c270e` |

The complete machine-readable register is
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round17.json`.

### Round-17 ASKED / UNASKED register

| choice or action | origin | disposition |
|---|---|---|
| canonical helper from `86a8eb21d18` | ASKED correction | copied verbatim and recorded |
| one shared core owner; remove ocean definition | ASKED | complete; prd/EOS parity 0 ulp |
| direct core module test | ASKED | complete |
| shared source-literal `hpg_sco` | ASKED | landed; stage-1 Krhs bit-exact |
| private pre-fix association arm | ASKED one-variable control | test-only; non-vacuous |
| literal depth-reduction arm | conditional ASKED | not landed; production already bit-exact |
| kt=1…10 sweep and OVERFLOW/LOCK gates | conditional ASKED | NOT RUN; stage-1 post-wind remains DEBT |
| stage-2 Kaa and stage-3 transports | conditional ASKED | NOT ENTERED |
| compiler-wide retraction | ASKED if open | already complete; retained |
| `oracle_bt_frames` schema correction | ASKED | complete; first difference `uu_b[111]` |
| DINO relabel | ASKED if open | already complete; consistency only |
| post-wind owner/fix | UNASKED | none selected; owner remains UNMEASURED |
| per-card HPG guard or shipped NEMO edit | forbidden | none |

All controls above are Codex-internal measurements.  Round-16 independent
review was still running during this dispatch; Round-17 independent review is
outstanding and no dual-review claim is made.

## Round 18 — surface-wind boundary (preregistered)

Round 18 starts from `dfc6eed322c`; the independent Round-17 review is still
running.  Before measuring or changing the wind path, the predicted owner is
the source association of the barotropic wind increment, ranked as follows:

1. `r1_rho0 * utauU * r1_hu(Kbb)` (and V) plus its left-to-right addition to
   the already exact slow RHS;
2. the upstream T-to-U/V face-stress interpolation;
3. accumulation order into `Ue_rhs` / `Ve_rhs`.

The source contract is explicit.  `sbcmod.F90:539-547` forms `utauU` as
`0.5*(utau(i,j)+utau(i+1,j))*(2-umask)*MAX(tmask,tmask_east)` and analogously
forms `vtauV`.  `domain.F90:159` forms resting `r1_hu_0` by masked division;
under QCO, `domzgr_substitute.h90:51,125-138` expands live `r1_hu(Kbb)` to
`r1_hu_0/(1+r3u(Kbb))`.  Finally, `stp2d.F90:198-202` performs the wind update
as `(r1_rho0*utauU)*r1_hu` followed by addition.  The retained config-local
V2 slow-forcing record is already the requested WRITE-only operand dump: it
contains `r1_rho0`, `utauU/vtauV`, live `r1_hu/r1_hv`, the pre-wind RHS, and
the post-wind RHS.

The preregistered discriminator is: operand bit identity plus a bit-exact
NEMO-literal replay confirms candidate 1; substituting oracle face stress alone
confirms candidate 2; exact operands and exact increment with only the final
sum non-exact confirms candidate 3.  Any residual not removed by the selected
one-variable replay refutes that candidate.  A planted one-ulp wind operand
must become the first boundary and exit nonzero.  If post-wind clears, the walk
continues in source order through atmospheric pressure (expected dead because
resolved `ln_apr_dyn=.false.` from `namelist_ref:211`), external substeps,
`un_adv/vn_adv`, and stage-1 Kaa; the first non-bit-exact boundary stops the
walk.  The kt=1…10 and cross-card gates remain conditional on that entire
stage-1 chain becoming bit-exact.

### Surface-stress and wind-product results

The baseline discriminator overturned the preregistered ranking at the first
operand.  `r1_rho0` and live `r1_hu/r1_hv(Kbb)` were bit-exact, but the face
stress already differed: U had 414 cells and V 418 cells different, both with
maximum absolute error `2.7755575615628914e-17` and maximum 3 ULP.  The raw
geographic midpoint is intentionally not a NEMO quantity: the harness had
inverse-rotated the already-native `usrdef_sbc` stress into the public
east/north fields, and the shared face helper rotated it back.  That algebraic
round trip is lossy on the 45-degree grid.

The shared forcing container now optionally carries a paired
`tau_i_native/tau_j_native` input, fail-closed if only one is present.  The
single shared face helper consumes that pair directly and applies the
source-associated midpoint; ordinary geographic coupling remains the default
when the pair is absent.  This is a forcing-coordinate representation, not a
GYRE physics switch.  Supplying GYRE's source-native T-point fields makes both
`utauU` and `vtauV` bit-exact.  This is
**CONFIRMED_NATIVE_STRESS_HANDOFF_OWNER**.  Its private legacy geographic arm
restores 414/418 differing cells and exits `1`.

With the face operands exact, the old collapsed `tau/(rho0*H)` expression left
179 U cells and 156 V cells different after wind, maximum
`3.308722450212111e-24` (1 ULP).  Materializing NEMO's written
`(r1_rho0*tau)*r1_h` product and then the addition clears every cell.  This is
**CONFIRMED_WIND_PRODUCT_ASSOCIATION_SECOND_OWNER**.  Its private legacy arm
reproduces the 1-ULP residual and exits `1`; the independent e3 plant perturbs
one nonzero cell, becomes the first boundary, and exits `1`.

| ordered boundary | pre-fix U / V | final U / V | disposition |
|---|---:|---:|---|
| `Krhs`, depth mean, post-drag | `0 / 17,400 U; 0 / 17,100 V` | `0 / 17,400 U; 0 / 17,100 V` | retained BIT-EXACT; 2-D reductions are `0 / 580 U; 0 / 570 V` |
| face stress differing cells | `414 / 580 U; 418 / 570 V` | `0 / 580 U; 0 / 570 V` | BIT-EXACT |
| post-wind max abs | `6.617444900424222e-24 / 6.617444900424222e-24` | `0 / 580 U; 0 / 570 V` | BIT-EXACT |
| pre-external max abs | `6.617444900424222e-24 / 6.617444900424222e-24` | `0 / 580 U; 0 / 570 V` | BIT-EXACT |

Atmospheric pressure is dead exactly as preregistered: resolved
`ln_apr_dyn=.false.` (`namelist_ref:211`), so post-wind is the external-mode
slow forcing.  The Round-12 compensating-error rule applies: the corrected
operators are bit-exact on NEMO's own operands, so both fixes stay even though
later external frames expose a second error.

### Next ordered boundary and Rule 8/12 disclosure

All 16 named external-mode frames at substep 1 are bit-exact.  At substep 2,
`eta_entry`, `eta_mid`, `u/v_entry`, `u/v_mid`, and `slow_u/slow_v` remain
bit-exact.  The first non-bit-exact source boundary is `eta_exit` at substep 2:
114 cells differ, maximum absolute error `1.3552527156068805e-20`, maximum
8 ULP.  The same substep's later `trd_u/trd_v` are NEAR-NULL (maximum 2 ULP and
`2.0679515313825692e-25`) and have no owner-discriminating power.

Movement versus the two-legacy-arm legoESM baseline is disclosed, but does not
decide eligibility.  Among the 800 substep-frame maximum-residual rows, 449
improve, 153 are unchanged, and 198 worsen after the exact wind fix.  The first
debt moves later, from `slow_u` at substep 1 (`6.617444900424222e-24`, 3 ULP)
to `eta_exit` at substep 2.  The 198 later maximum-residual movements are fully
enumerated, cell counts included, by the paired `advmean_pre_wind.json` and
`advmean_final.json` artifacts; they are downstream diagnostic frames, not a
newly run whole-step or cross-card Rule-12 score.  The final weighted means are
U `4.440892098500626e-16` and V `3.3306690738754696e-16` from their oracle
values; they are downstream of the first debt and carry no owner label.

Because the complete stage-1 chain is not bit-exact, the requested conditional
work is honestly **UNMEASURED**: no new kt=1…10 sweep, OVERFLOW/LOCK stage or
trajectory gate, stage-1 Kaa certification, stage-2 Kaa measurement, or
stage-3 transport measurement was run.  There are therefore no new whole-step
Rule-8 rows or cross-card Rule-12 rows to enumerate.  The newly exposed debt is
registered as `external_mode.substep2.eta_exit`; it is never used to revert the
bit-exact upstream corrections.  The nonzero advmean plant selects substep 2,
U cell `(j=1,i=1)`, becomes the first accumulated-transport mismatch, and exits
`1`.

### Round-18 artifacts and verification

| artifact | SHA-256 |
|---|---|
| pre-fix wind operand walk | `dbd0c0ef7bf77979d5cc4fba3b4b89d6be4766acfe830abc0e4227910639bcd9` |
| final wind operand walk | `cd5c61c294de1524355e0710391c6934d0860cc74a9d86b163946f72d9698a92` |
| legacy geographic-stress arm | `e2ab3a7dc0a0a4e1ac0783be35557d94633035d480b0cf4ba9329fa8e0cb029a` |
| legacy wind-product arm | `b00fbd31ac9c98c27e81ccbc43174d008122fb7b2ba56b290c024ddacaa7529b` |
| wind planted control | `aabc5b521f35f7028b2cefb1387a40e6ab630fecc392d5c1b3f6f597fdf68fa2` |
| pre-fix external/advmean trace | `2f4f9874c1a52de88a76f38d20c62944418165e3984f5fe76754cf055d4921ba` |
| final external/advmean trace | `69229ad6f6b0eee2d8ce1cf28e01c126bcd05ed72db5a836ce7b43d0133401aa` |
| nonzero advmean plant | `cce7101f436b949df010fb9d8bbb20341e95ffe63f79c2e4c625ab77721bd087` |
| superseded-H1 discriminator | `9febb8f3d28e7b34ae24f6eb6102eabc5fa744cf0d86439479f2678c67a33918` |

The source-rounding, isomorphism, surface-forcing, RGB, GYRE gate, and full
WS-RK3 focused invocation is `98 passed in 1826.42 s`.  The RGB routing control
fails `6 == 7` with exit `1` when one of its four source sites is reverted.

### Round-18 ASKED / UNASKED register

| choice or action | origin | disposition |
|---|---|---|
| source-first surface-wind operand walk | ASKED | complete; two ordered owners confirmed |
| paired native stress representation | UNASKED implementation mechanism | shared forcing interface; no card physics switch |
| source-associated wind product/add | ASKED | landed in shared external-mode path |
| geographic and product legacy arms | ASKED one-variable controls | private only; each exits `1` |
| APR boundary | ASKED | VERIFIED-ABSENT, resolved `ln_apr_dyn=.false.` |
| external substeps / advmean continuation | ASKED | stopped at substep-2 `eta_exit` |
| QSR recovered-delta disclosure | ASKED review fix | corrected in place; unscored 4,191-cell row |
| H1 post-fix wording and pre-fix hash | ASKED review fix | `SUPERSEDED_BY_FIX`; evidence retained |
| RGB precision-policy EXP routing | ASKED review fix | all four sites routed; behavioral plant fails |
| kt=1…10 and OVERFLOW/LOCK gates | conditional ASKED | UNMEASURED; complete stage-1 chain did not clear |
| stage-1/2 Kaa and stage-3 transports | conditional ASKED | UNMEASURED / NOT ENTERED |
| ZDF matrix walk | explicitly out of scope | not entered |
| GYRE card guard, tuning, or shipped NEMO edit | forbidden | none |

All Round-18 measurements are Codex-internal.  Round-17 independent review was
still running during this dispatch; Round-18 independent review is outstanding
and no dual-review claim is made.

## Round 19 — external-mode substep 1→2 preregistration

Round 19 starts from `a07338be5bed45a615b41bc491db5369372ce867`; review
minor commit `2f395a91e0539e8c51f831fe13c2c9a381c6de0e` precedes this
measurement.  Round-17 independent review returned SHIP; Round-18 independent
review is still running.  The shipped NEMO source read before this
preregistration is `dynspg_ts.F90` SHA-256
`49f367fe8dc7a47771cdabc39a4eea7b8c36001f623f57f132beb909b835ea60`.

The predicted owner is the source association of the substep-1 velocity
update, with the ordered candidates below.  The ranking follows the first
reported live residual: substep-2 midpoint U/V differ by
`1.728794245346027e-18` / `1.7296412782932813e-18` in 580 / 570 wet faces,
whereas the substep-2 `eta_exit` residual is only
`1.3552527156068805e-20` in 114 wet T cells.

1. The vector update at `dynspg_ts.F90:719-731`: NEMO forms
   `spg + trd + frc`, multiplies once by `rDt_e`, adds `un_e/vn_e`, then
   masks.  Exact inputs with a non-exact result confirm association/order;
   substituting the literal replay alone must clear the exit and next
   midpoint or this candidate is refuted.
2. The continuity chain at `:603-609,627-630`: NEMO forms
   `(e2u*ua_e)*zhup2_e`, the two face differences, their sum, the
   `r1_e1e2t` product, then `sshn_e-rDt_e*(ssh_frc+zhdiv)`.  Its first
   non-exact intermediate confirms this candidate only for the SSH branch;
   it owns the velocity branch only if replacing that one operand clears the
   later PGF and velocity result.
3. The startup midpoint and SSH blends at `:535-562`: the first two substeps
   set `(za1,za2,za3)=(1,0,0)` before the three-term expressions.  A mismatch
   before continuity confirms this association; exact replay refutes it.
4. The exit face depths and reciprocals at `:653-667,771-778`: the QCO build
   uses the surface-weighted two-point SSH average, adds `hu_0/hv_0`, then
   divides the mask by `depth + 1 - mask`.  These operands must be exact before
   any drag attribution.
5. The half-step-back SSH blend and pressure gradient at `:671-685`, followed
   by the same-time ENE Coriolis at `:688-689`, explicit drag at `:699-705`,
   and forcing add at `:719-731`.  This candidate is downstream and cannot be
   labelled until all earlier operands are exact.

The config-local WRITE-only extension will record, for substeps 1 and 2, every
term above immediately before and after its source statement: history fields
and interpolation weights; metric transports and their two differences;
`zhdiv`, `ssh_frc`, and SSH entry/exit; face-depth sum operands, depth, and
reciprocal; backward-interpolation weights/terms/result; PGF difference,
metric reciprocal, and result; Coriolis, drag, and slow forcing; velocity-sum
partials and exit.  Its control is the ordinary stage record hash, which must
remain unchanged.  The committed diagnostic must exit nonzero for a planted
one-ulp mutation of the first exact nonzero operand.

Each arm changes one replay boundary only and is evaluated on identical V2
operands under production JIT/CPU/fp64/libm.  A boundary is an owner only when
its magnitude scales to the child residual and its substitution clears that
child; otherwise it is labelled a contributor or refuted.  No shared
external-mode arithmetic changes before this discriminator reports.

### Round-19 result — ordered external-mode recurrence

The preregistered first-boundary premise was refuted before attribution.  The
carried `1.7288e-18` midpoint value came from a superseded pre-wind artifact;
on the Round-18 tip both substep-1 and substep-2 midpoint velocities are
bit-exact.  A separate diagnostic bug had reconstructed face SSH by subtracting
the roughly 4,450 m reference depth from live depth, losing low bits through
catastrophic cancellation.  The production trace now returns the actual face
SSH without changing the solver.  After that correction, the first real
Round-18-tip boundary is substep-1 face SSH: U differs in `147 / 580` wet
faces and V in `158 / 570`, each by at most `1.6940658945086007e-21`
(2 ulp).

The retained scalar-math executable and ordinary stage record establish the
WRITE-only contract.  The executable SHA-256 is
`207e701f740b2fc4ee5f234a22508d774c2f517646e8b9d2f1e0a2a2d58d7ea9`,
contains zero `_ZGV*` symbols, and the stage-1 record remains
`ce25b004e7e8289b6e803263f895576981ce22516ccddfbd85d7be5ce5bcaedc`.
The config-local `MY_SRC/dynspg_ts.F90` extension is WRITE-only, SHA-256
`62d51638def7a811264d5f0eb53c8ef1c27b34b601110e7b9993ff62dfa4b4e6`;
its ordered record is
`8aa0836c11824ffcaddaafca4aefd53f62692481fc8d10d74665ac508d1eb054`.
No shipped NEMO source was edited.  The first diagnostic root without the ENE
coefficient extension remains retained and flagged; the scored root is
`round19_oracle_v2_external_coeff`.

The source-order walk follows NEMO exactly: startup AB3 midpoints at
`dynspg_ts.F90:535-562`; metric transports and continuity at `:603-629`;
surface-weighted face SSH, the backward SSH blend, and pressure gradient at
`:653-685`; ENE Coriolis, drag, and the vector update at `:688-731`; face
depth and reciprocal refresh at `:771-778`; and history rotation at
`:804-816`.  GYRE's live ENE coefficient recurrence is `:1383-1410`:
NEMO first sums `e3u*e3v*mask/e3f_vor` over levels and only then multiplies
by `ff_f` and the horizontal/depth metrics.

Each row below is one source-local change.  The previous first boundary is
shown before the change; the after column is the newly exposed boundary.
This sequential replacement is the scaling check: every landed expression
clears its child at the same residual scale before an owner label is assigned.

| one-variable boundary | before | after / next boundary | disposition |
|---|---|---|---|
| surface-weighted face SSH and direct inverse depth (`:653-667,771-778`) | U `1.6941e-21`, 2 ulp, `147 / 580`; V `1.6941e-21`, 2 ulp, `158 / 570` | face values exact; substep-2 `eta_exit` exposed | **CONFIRMED_SOURCE_ASSOCIATION_OWNER** |
| continuity update (`:627-629`) | `eta_exit=1.3552527156068805e-20`, 8 ulp, `114 / 600` | `eta_exit` exact; `eta_pgf` exposed | **CONFIRMED_CONTINUITY_ASSOCIATION_OWNER** |
| four-term backward SSH blend (`:671-679`) | `eta_pgf=1.3552527156068805e-20`, 1 ulp, `173 / 600` | blend and PGF exact; ENE trend exposed | **CONFIRMED_BACKWARD_BLEND_OWNER** |
| ENE coefficient recurrence (`:1383-1410`) | each of eight coefficients `6.776263578034403e-21`, at most 3 ulp, `222` cells | all coefficients and Coriolis exact | **CONFIRMED_ENE_COEFFICIENT_OWNER** |
| vector velocity update (`:719-731`) | U `1.6940658945086007e-21`, 1 ulp, `126 / 580`; V same maximum, `111 / 570` | substep-2 exit exact; substep-3 midpoint exposed | **CONFIRMED_VECTOR_UPDATE_ASSOCIATION_OWNER** |
| AB3 midpoint (`:549-562`) | first later midpoint residual `1.6940658945086007e-21` | all state frames exact through the next live trend boundary | **CONFIRMED_MIDPOINT_ASSOCIATION_OWNER** |

The ENE discriminator is particularly direct: all eight dumped coefficients
were three ulp off under the former algebraic `ff_f/e3f` folding, while a
pure-NumPy statement transcription using NEMO's dumped inputs reproduced the
oracle Coriolis term bit-for-bit.  The shared builder now uses the one core
`nemo_source_round` helper and NEMO's post-sum `ff_f` order.  S-16/S-17 in the
isomorphism map record these as refinements of the existing shared routines;
the private legacy-continuity hook is not a card selector.  Its run recreates
the 8-ulp, `114 / 600` substep-2 failure and exits `1`.  The independent
one-ulp metric plant becomes the first ordered failure and also exits `1`.

The final production-JIT trace is exact for all 110 explicitly dumped
substep-1/2 operands.  Across the full 50-substep, 16-frame stream, `792 / 800`
rows are bit-exact.  The first remaining non-bit-exact row is substep 7
`trd_u`: one of 580 wet faces differs by `5.048709793414476e-29` (1 ulp),
only `1.4695666785339487e-20` of the oracle term magnitude
`3.435509165498437e-9`.  Seven later trend rows each contain one differing
cell; the largest is 16 ulp but remains at most `2.45e-18` relative to its
oracle term.  These are **NEAR_NULL_BIT_DEBT_NO_OWNER_POWER**, not
exoneration.  The requested all-50 bit-exact condition therefore does not
hold.  Nevertheless both final advective transports are exact (`0 / 580` U,
`0 / 570` V), and stage-1 Kaa is exact (`0 / 17,400` U,
`0 / 17,100` V).  The ordered register stops at the substep-7 trend boundary.

### Stage-2 and stage-3 boundary measurements

The condition for measuring the later stages was met at the stage-1 Kaa
boundary, but the measurement refutes the expectation that stage 2 is now
exact.  Corrected stage-2 Kaa remains DEBT: U
`2.1986806906376666e-15` over `17,400` wet values and V
`2.2380914396075147e-15` over `17,100`.  The directly measured stage-1 tracer
operands remain AT-BAR but non-bit-exact (T `3.552713678800501e-15`, S
`7.105427357601002e-15` absolute), so the honest first capable upstream
boundary is still the tracer accumulation/propagation register, not the now
exact external-mode chain.  The stage-2 plant changes U by approximately one
and exits `1`.

Stage-3 transports were measured, but no downstream owner is assigned while
stage 2 is open.  Against Oracle V2, zFu has absolute/normalized residual
`7.00832742950297e-8 / 1.122004923897255e-11`, zFv
`7.133894541766495e-8 / 1.2743986985222504e-11`, and zFw
`3.0141292484131554e-7 / 7.464529752188664e-10`; all are **DEBT,
OWNER_UNMEASURED_UPSTREAM_STAGE2**.  Their dedicated plant exits `1`.  No ZDF
matrix walk was entered.

### Re-pinned GYRE whole-step sweep

The low-memory trajectory-only route executes the same production-jitted step
and returns before compiling unrelated private hooks.  Two full-hook attempts
did not produce an artifact; this is harness resource containment, not an
eager or alternate dynamics route.  CPU, x64/fp64, Oracle V2, and explicit
`transcendentals="libm"` remain pinned.  `first_over_bar` is unchanged at
kt=2 for T/S/u/v; SSH now remains AT-BAR at kt=2.

| kt | T | S | u | v | SSH |
|---:|---:|---:|---:|---:|---:|
| 1 | `0` | `0` | `0` | `0` | `0` |
| 2 | `1.3614736849003888e-12` | `2.2181101297999213e-14` | `9.484089954776408e-7` | `8.987992592542841e-7` | `4.336808689942018e-19` |
| 3 | `3.7223441372709554e-4` | `3.683246743167771e-5` | `9.311678913140553e-3` | `4.719077591308102e-3` | `7.074669630242024e-7` |
| 4 | `1.0594738051818496e-3` | `8.225740462674584e-5` | `1.4182404011957604e-2` | `1.5947842304474213e-2` | `5.0604233772311505e-7` |
| 5 | `3.0537424781427133e-3` | `8.611826431262424e-5` | `2.0733890435953528e-2` | `4.379766564509835e-2` | `6.701851022690615e-5` |
| 6 | `3.817675809918862e-3` | `1.287536238418589e-4` | `3.113472973295961e-2` | `6.021959208711861e-2` | `1.631023907114217e-4` |
| 7 | `4.546666553978738e-3` | `1.2244044884708371e-4` | `4.004140471842379e-2` | `6.631577323114793e-2` | `2.087141815218642e-4` |
| 8 | `5.098376094806436e-3` | `1.360984430508487e-4` | `4.718005909052282e-2` | `2.4527161866119307e-2` | `2.5108247588168423e-4` |
| 9 | `5.445062142555964e-3` | `1.471182174583459e-4` | `5.256039760377703e-2` | `1.4533612267114887e-2` | `2.1203002269254564e-4` |
| 10 | `5.636634494144276e-3` | `1.5483121354443772e-4` | `5.6249869570541774e-2` | `1.1173365766925972e-2` | `2.12554355858912e-4` |

The final source-round hardening of the ENE recurrence is scientifically
identical to the immediately preceding Round-19 sweep.  Against the clean
Round-18 tip `a07338be5`, the full 50-row register has 15 improvements,
21 unchanged rows, and the following 14 worsened maxima (Rule 8 disclosure):

| field | kt | Round-18 tip | Round-19 | ratio |
|---|---:|---:|---:|---:|
| v | 2 | `8.9879925925417567e-07` | `8.987992592542841e-07` | `1.0000000000001206` |
| u | 3 | `0.0093116789131405323` | `0.0093116789131405531` | `1.0000000000000022` |
| SSH | 4 | `5.0604233771834456e-07` | `5.0604233772311505e-07` | `1.0000000000094271` |
| u | 5 | `0.020733890435948865` | `0.020733890435953528` | `1.0000000000002249` |
| v | 5 | `0.043797665645095019` | `0.043797665645098349` | `1.0000000000000759` |
| u | 6 | `0.031134729732959374` | `0.03113472973295961` | `1.0000000000000075` |
| v | 6 | `0.060219592087115387` | `0.060219592087118606` | `1.0000000000000535` |
| u | 7 | `0.040041404718418105` | `0.040041404718423788` | `1.0000000000001419` |
| v | 8 | `0.024527161866106789` | `0.024527161866119307` | `1.0000000000005103` |
| u | 9 | `0.0525603976037241` | `0.05256039760377703` | `1.000000000001007` |
| v | 9 | `0.014533612266942067` | `0.014533612267114887` | `1.0000000000118912` |
| SSH | 9 | `0.0002120300216775443` | `0.00021203002269254564` | `1.0000000047870643` |
| u | 10 | `0.056249869570499197` | `0.056249869570541774` | `1.000000000000757` |
| SSH | 10 | `0.0002125543557739409` | `0.000212554355858912` | `1.0000000003997618` |

### Oracle-relative cross-card result and Rule-12 register

The four cellwise gates compare this round with clean baseline
`a07338be5` against each card's NEMO oracle.  All eight compressed residual
fields (before/after for each gate) are retained and hashed in the Round-19
manifest.  The exact verdict lines are:

```text
ORACLE_RELATIVE_COMPARE PASS: rows=9 max_worsening_ulps=6.103515625e-05 first_over_bar='<absent>'->'<absent>' plant=None
ORACLE_RELATIVE_COMPARE PASS: rows=9 max_worsening_ulps=0 first_over_bar='<absent>'->'<absent>' plant=None
ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=3.5625 first_over_bar={'fields': ['T', 'u'], 'kt': 2}->{'kt': 2, 'fields': ['T', 'u']} plant=None
ORACLE_RELATIVE_COMPARE PASS: rows=50 max_worsening_ulps=0.0009765625 first_over_bar={'fields': ['u'], 'kt': 4}->{'kt': 4, 'fields': ['u']} plant=None
```

The order is OVERFLOW stage, LOCK stage, OVERFLOW trajectory, LOCK
trajectory.  No AT-BAR row crosses to DEBT and neither first-over-bar moves
earlier.  The OVERFLOW trajectory gate nevertheless fails correctly: U at
kt=9 and kt=10 has cellwise worsening beyond two row-scale ulp (maximum
`2.625` and `3.5625` ulp respectively; the gate's first printed violating
cells are 2.5 and 3.5625 ulp).  Under the compensating-error clause the
source-exact fixes stay, and this becomes new named debt:
**OVERFLOW_TRAJECTORY_LATE_U_DOWNSTREAM_OF_S16, OWNER_UNMEASURED**.  It is
neither waived nor hidden behind a card switch.

Every Rule-12 row with at least one cell moving away from its oracle is
enumerated below.  `worse` is the maximum increase in absolute residual;
the ulp column uses the campaign row-scale definition, and the final column
is worsened cells / scored cells.  Rows absent from this table have zero
worsened cells.  Movement versus the prior legoESM output remains in each
comparison artifact as required by Rule 8.

| gate row | worse | row-scale ulp | cells |
|---|---:|---:|---:|
| OVERFLOW s2 baroclinic u | `1.3553e-20` | `6.1035e-5` | `39 / 16,900` |
| OVERFLOW s2 instantaneous u | `1.3553e-20` | `6.1035e-5` | `38 / 16,900` |
| OVERFLOW s3 baroclinic u | `1.3553e-20` | `6.1035e-5` | `20 / 16,900` |
| OVERFLOW s3 instantaneous u | `1.3553e-20` | `6.1035e-5` | `20 / 16,900` |
| OVERFLOW kt2 baroclinic u | `1.3553e-20` | `6.1035e-5` | `20 / 16,900` |
| OVERFLOW kt2 instantaneous u | `1.3553e-20` | `6.1035e-5` | `20 / 16,900` |
| OVERFLOW kt2 SSH | `1.3878e-17` | `0.0625` | `2 / 200` |
| OVERFLOW kt2 u | `1.3553e-20` | `6.1035e-5` | `20 / 16,900` |
| OVERFLOW kt3 SSH | `6.9389e-18` | `0.03125` | `6 / 200` |
| OVERFLOW kt3 u | `3.4694e-18` | `0.015625` | `68 / 16,900` |
| OVERFLOW kt4 SSH | `1.1102e-16` | `0.5` | `5 / 200` |
| OVERFLOW kt4 u | `2.7756e-17` | `0.125` | `217 / 16,900` |
| OVERFLOW kt5 SSH | `5.5511e-17` | `0.25` | `9 / 200` |
| OVERFLOW kt5 u | `5.5511e-17` | `0.25` | `263 / 16,900` |
| OVERFLOW kt6 SSH | `5.5511e-17` | `0.25` | `11 / 200` |
| OVERFLOW kt6 u | `5.5511e-17` | `0.25` | `363 / 16,900` |
| OVERFLOW kt7 T | `3.5527e-15` | `1` | `1 / 17,000` |
| OVERFLOW kt7 SSH | `1.1102e-16` | `0.5` | `20 / 200` |
| OVERFLOW kt7 u | `5.5511e-17` | `0.25` | `509 / 16,900` |
| OVERFLOW kt8 S | `7.1054e-15` | `1` | `3 / 17,000` |
| OVERFLOW kt8 T | `3.5527e-15` | `1` | `2 / 17,000` |
| OVERFLOW kt8 SSH | `2.2204e-16` | `1` | `15 / 200` |
| OVERFLOW kt8 u | `8.3267e-17` | `0.375` | `523 / 16,900` |
| OVERFLOW kt9 T | `7.1054e-15` | `2` | `3 / 17,000` |
| OVERFLOW kt9 SSH | `1.6653e-16` | `0.75` | `21 / 200` |
| OVERFLOW kt9 u | `5.8287e-16` | `2.625` | `662 / 16,900` |
| OVERFLOW kt10 S | `7.1054e-15` | `1` | `2 / 17,000` |
| OVERFLOW kt10 T | `7.1054e-15` | `2` | `8 / 17,000` |
| OVERFLOW kt10 SSH | `1.6653e-16` | `0.75` | `16 / 200` |
| OVERFLOW kt10 u | `7.9103e-16` | `3.5625` | `471 / 16,900` |
| LOCK kt4 SSH | `1.3235e-23` | `5.9605e-8` | `2 / 128` |
| LOCK kt4 u | `2.0680e-25` | `9.3132e-10` | `38 / 2,540` |
| LOCK kt5 SSH | `2.5849e-26` | `1.1642e-10` | `3 / 128` |
| LOCK kt5 u | `8.2718e-25` | `3.7253e-9` | `62 / 2,540` |
| LOCK kt6 SSH | `3.7865e-29` | `1.7053e-13` | `4 / 128` |
| LOCK kt6 u | `3.3087e-24` | `1.4901e-8` | `106 / 2,540` |
| LOCK kt7 SSH | `4.9304e-32` | `2.2204e-16` | `4 / 128` |
| LOCK kt7 u | `1.3553e-20` | `6.1035e-5` | `133 / 2,540` |
| LOCK kt8 SSH | `8.4703e-22` | `3.8147e-6` | `7 / 128` |
| LOCK kt8 u | `2.7105e-20` | `0.00012207` | `137 / 2,540` |
| LOCK kt9 SSH | `1.6544e-24` | `7.4506e-9` | `6 / 128` |
| LOCK kt9 u | `1.3553e-20` | `6.1035e-5` | `116 / 2,540` |
| LOCK kt10 SSH | `2.1684e-19` | `0.0009765625` | `7 / 128` |
| LOCK kt10 u | `1.3553e-20` | `6.1035e-5` | `160 / 2,540` |

The current cross-card reports were launched from the origin-visible checkout
before the identical file content was committed through the required local
Git directory, so their internal provenance field reads `a07338be5-dirty`.
The local-Git content commit is `ec6af6c61`; its package/script diff against
the measured tree is empty.  The artifact hashes, baseline SHA, and final
branch SHA in the manifest remove the ambiguity rather than rewriting the
reports.

### Round-19 artifact manifest

The complete machine-readable register, including all eight cellwise
cross-card residual NPZ hashes, is
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round19.json`.
Key hashes are:

| artifact | SHA-256 |
|---|---|
| pre-fix ordered trace | `b6e017400e264114993cae126bfe5f79f20eb2ae33851265303162eb8a0e1623` |
| final ordered trace | `a72055a7d97f008d8bbac25777ac3d651d8481eca319bf7028e0a720bd743472` |
| legacy continuity arm | `815337c35adf87c2e7f5831212b73ad4a1af9933bf0f5211cce64d25a091e806` |
| ordered metric plant | `d5b424ad9d159582594b316e9d7e2b441988a3edea171dbd626d601073c9c236` |
| clean Round-18-tip whole-step baseline | `330aaf2625f45bcd7b596e2c943b8dc3eb1a3428bd1bda465748212a1f38bca6` |
| final whole-step sweep | `9ad9b3fc36f07aa0bb38516f07ea366be20a9a1d23f0317244bd1f2c38cdf393` |
| stage-2 Kaa | `a471c4d419d00706ce5ef4cac403be56f75bdb43345fc343c9d6ec4283c0a3d8` |
| stage-3 transport | `b6a28c99ba79d811ed429a465dcec73041171b8555859352c56d3d7770dd9858` |
| OVERFLOW stage comparison | `3b98c41ae26f66e17fd61934fe1904bdd8ebe1c916234931ccd21776c7e6858b` |
| LOCK stage comparison | `9ec7319bf3e736460472a725fc49a6ba3d8e668e1269e87045a1a8fc256391b3` |
| OVERFLOW trajectory comparison | `c0c41c862fa7f7868763ea7bcc2ccfd385f666e6889baf072be7102adbe0b242` |
| LOCK trajectory comparison | `d8ee86328071f2d47b73817ed0fb298f6e5c01ea74bdd196550b4bdb7b7b622e` |

Verification is `85 passed in 16.42 s` for the combined core helper, HPG
consumer pin, ENE literal recurrence, time-level registry, scheme-isomorphism,
and GYRE gate suite.  A separate barotropic behavioral selection is
`29 passed in 29.31 s`.  The continuity, ordered-metric, stage-2, and stage-3
plants each exit `1`.

### Round-19 ASKED / UNASKED register

| choice or action | origin | disposition |
|---|---|---|
| correct source-rounding docstring | ASKED review minor | complete; barriers are explicitly documented as stripped |
| HPG consumer hex bit-pin | ASKED review minor | complete; direct production-JIT consumer pin added |
| print exact rows as `0 / wet` | ASKED review minor | complete in ordered receipt rows |
| substep-1→2 WRITE-only operand dump | ASKED | complete; ordinary stage hash unchanged |
| return actual face SSH in the private trace | UNASKED measurement-integrity fix | removes catastrophic-cancellation reconstruction; production state untouched |
| source-literal face SSH/inverse depth, continuity, backward blend, vector update, and midpoint | ASKED owner walk | landed in the shared S-16 identity |
| live ENE vertical coefficient association | ASKED owner walk | landed in shared S-17; no per-card arm |
| private legacy-continuity and one-ulp metric controls | ASKED one-variable arms | both exit `1` |
| whole-step kt=1…10 repin | conditional ASKED | complete; first_over_bar remains kt=2 T/S/u/v |
| trajectory-only low-memory return | UNASKED harness containment | same production step and scoring; private hooks run separately |
| OVERFLOW/LOCK cellwise cross-card gates | conditional ASKED | complete; three PASS, OVERFLOW trajectory FAIL registered as second error |
| stage-2 Kaa | conditional ASKED | measured DEBT; propagated tracer boundary remains open |
| stage-3 transports | conditional ASKED | measured DEBT; no downstream owner label |
| substep-7 near-null trend continuation | ordered stopping boundary | registered as bit debt; owner remains UNMEASURED |
| tra_zdf/dyn_zdf matrix walk | explicitly deferred | not entered |
| GYRE-only guard, tuning, gate change, shipped NEMO edit | forbidden | none |

Round-17 independent review returned SHIP.  All Round-19 measurements are
Codex-internal; Round-18 and Round-19 independent review remain outstanding,
and no dual-review claim is made.

## Round 20 — merged baseline, gate repair, stress handoff, and stage transport

Round 20 started from reviewed merge `03c6e8d96ff7f69207abdee43ec28e223d083f4e`
(Round-19 parent `c83f73c23ff8`, integration/census parent `b46e617d02a5`).
The local Git directory was fetched and its branch ref was advanced to that
merge before any edit; the mixed worktree status was empty.  Every number
below is CPU, production JIT, binary64, Oracle V2, and explicit
`PrecisionPolicy.fp64(transcendentals="libm")` unless a baseline is named.

The merged-tree starting facts remain exactly as supplied: GYRE's register is
bit-identical to Round 19; LOCK stage and trajectory gates PASS at zero ulp;
OVERFLOW stage has zero violations.  OVERFLOW trajectory must be quoted
against both baselines: versus Round 19 it FAILS in four rows at at most 5.5
ulp with zero status changes, while versus the integration census table 49 / 50
rows are identical and kt2 SSH improves from DEBT to AT-BAR.

### 20.0 — production-JIT 19-frame gate repair

The failure belonged to the gate, not the model.  The Round-10 route fix forces
`LatLonCGridOceanModel.step` through `_step_jitted`, while the census gate had
two outer `jax.disable_jit` contexts solely so its Python closure could call
`np.asarray` on `_nemo_substep_trace_test_hook` operands.  The gate now uses the
existing WRITE-only returned-pytree trace seam and materialises operands only
after the production step returns.  Both eager wrappers are gone.

The repaired tests are **13 passed in 287.60 s**.  The planted pre-entry
control exits 1.  The 19-frame artifact retains the integration census verdict
`DEBT`; no physics row or owner label moved.  Its SHA-256 is
`ac2728dc4b9e1b4cec322dbf4cb3ef743417c68f0f3c56a30b9e9d95e360369f`.
The first stored `pgf_v` row remains inventory-only because OVERFLOW's
one-wet-row tank has no active V face; it is not presented as an alignment
claim.

### 20.1 — scalar-libm provenance convergence

All nine NEMO-identity census-probe commands now set scalar-libm explicitly.
Against a clean detached `03c6e8d96ff7` native-math worktree (retained and
flagged, not deleted), the BBL, Aimp, and ZDF scaling payloads are identical,
and all 15 `run-arm` arrays are bit-identical.  Both native and scalar-libm
state NPZs hash to
`1b4a98f2af4ae81815bbaf47775aa82408fad7d2efbb1b0a8f6b12c390cd0ac7`.
This confirms the preregistered zero move for OVERFLOW, whose executed probe
path has no per-step transcendental.  The full term-budget command was stopped
at the 12-minute CPU budget before emitting a row and is **UNMEASURED**, not
silently inferred from the other probes.

### 20.2 — coastal U/V wind-stress handoff

The shared handoff now transcribes `sbcmod.F90:539-546`, in particular the
source statements at `:543-544`: the midpoint stress is multiplied by
`(2-umask)` / `(2-vmask)` and by the maximum of the two adjacent T masks.
Every written product uses the shared `nemo_source_round`; no public selector
was added.  S-51 in the branch-isomorphism map records the one implementation.

With GYRE's V2 operands the faithful face stresses are bit-exact: 0 / 704 U
and 0 / 704 V cells differ.  The legacy ablation differs on 124 U and 102 V
dry coastal cells (maximum `8.87866658e-2`), but on the dynamically active
faces both variants are unchanged: 0 / 580 U and 0 / 570 V.  The apparently
live coastal-owner prediction is therefore **REFUTED FOR GYRE DYNAMICS**;
the faithful source identity still lands.  The one-ulp synthetic U-face plant
exits 1.  Focused verification is 41 passed.

The required cellwise cross-card gates completed after this shared change:

```text
ORACLE_RELATIVE_COMPARE PASS: rows=9 max_worsening_ulps=0 first_over_bar='<absent>'->'<absent>' plant=None
ORACLE_RELATIVE_COMPARE PASS: rows=50 max_worsening_ulps=0 first_over_bar={'fields': ['T', 'u'], 'kt': 2}->{'fields': ['T', 'u'], 'kt': 2} plant=None
ORACLE_RELATIVE_COMPARE PASS: rows=9 max_worsening_ulps=0 first_over_bar='<absent>'->'<absent>' plant=None
ORACLE_RELATIVE_COMPARE PASS: rows=50 max_worsening_ulps=0 first_over_bar={'fields': ['u'], 'kt': 4}->{'fields': ['u'], 'kt': 4} plant=None
```

The order is OVERFLOW stage, OVERFLOW trajectory, LOCK stage, LOCK trajectory.
Rule 8 and Rule 12 have zero moved rows for this change.

### 20.3 — stage-1 tracer transport and stored barotropic mean

The V2 ordered tracer record first proves the child boundary: after the stage-1
QCO update T is 0 / 17,600 and S is 0 / 17,600 differing wet cells.  The raw
V2 record contains two non-finite `zFw` values only in discarded NEMO halos;
all transformed scored interiors are finite.  The earlier V1 comparison that
showed one T/S cell is retained but flagged non-authoritative for this
scalar-math round.

The first primitive transport departure was source association at
`stprk3_stg.F90:257-280`.  NEMO forms
`zub=un_adv*r1_hu-uu_b(Kmm)` / its V analogue and then
`zFu=(e2u*e3u(Kmm))*(uu(Kmm)+zub*umask)` (`:265-278`).  Before the fix,
metric, masks, Kmm velocity, Kmm face thickness, and `un_adv/vn_adv` were
bit-exact, but the corrected velocity differed in 4,650 U and 5,250 V cells
(at most 1 ulp), and zF differed in 4,190 U and 4,705 V cells (at most 2 ulp).
The production code was incorrectly re-reducing the 3-D Kmm velocity instead
of consuming NEMO's separately stored `uu_b/vv_b(Kmm)`.

The shared S-21 identity now carries that stored mean through all three RK3
stage transports.  The old 3-D reduction is a private one-variable legacy
ablation.  With the fix, corrected U/V and zFu/zFv are 0 / 17,400 and
0 / 17,100 differing wet cells respectively; the legacy arm reproduces the
old departures and makes one T and one S prognostic cell differ by one ulp.
The owner is **CONFIRMED_STORED_BAROTROPIC_MEAN_ASSOCIATION**.  A fixed
oracle-bit JIT consumer pin was added; three focused tests pass, all 35
isomorphism tripwires pass, and the planted zFu cell exits 1.

The child stage-2 composition does **not** clear.  Its source and scale table is:

| boundary | U | V | disposition |
|---|---:|---:|---|
| stage-2 HPG primitive inputs (`rhd`,`e3w`,`gdept_z0`) | `0 / 21,120` each | `0 / 21,120` each | BIT-EXACT |
| direct source-literal HPG recurrence on NEMO inputs | `0 / 17,400` | `0 / 17,100` | BIT-EXACT |
| accumulated Krhs | `2.8731622268661884e-19` | `2.8731291396416863e-19` | AT-BAR, not exact |
| raw Kaa | `2.0686848501566546e-15` | `2.0686441925751864e-15` | DEBT |
| post-mean Kaa | `2.1986806906376666e-15` | `2.2380914396075147e-15` | DEBT |

Multiplication by the stage-2 `rDt=7200 s` accounts for the raw-Kaa scale.
The previously printed HPG term residual was reconstructed as
`after_hpg-before`; like the QSR recovered-delta row, it contains subtraction
cancellation and is not a direct operator mismatch.  Experiments that retained
the direct anomaly ratio, added more statement guards, or changed the shared
rounding helper did not move the live child and were all reverted.  No owner is
claimed for the remaining Krhs accumulation debt.

Because stage-2 Kaa is still the first unresolved boundary, stage-3
`zFu/zFv/zFw`, `tra_zdf`/`dyn_zdf` matrix coefficients and solutions, and TKE
are **UNMEASURED_UPSTREAM_STAGE2_DEBT** in this round.  This follows the
preregistered ordering; none was tuned or exonerated.

### Whole-step repin and Rule 8

The production-JIT V2 sweep remains `first_over_bar=kt2` in T/S/u/v.  kt1 T/S
are bit-exact; the at-rest kt1 u/v/SSH rows remain UNINFORMATIVE.  Every kt2
maximum is unchanged from the merge (`T=1.3614736849003888e-12`,
`S=2.2181101297999213e-14`, `u=9.484089954776408e-7`,
`v=8.987992592542841e-7`, `SSH=4.336808689942018e-19`).  Across all 50
rows, 20 improve, 20 worsen, and 10 are identical.  The 20 worsened Rule-8
maxima are enumerated here; no status changes.

| field | kt | merge `03c6e8d96` | Round 20 | ratio |
|---|---:|---:|---:|---:|
| S | 3 | `3.683246743167771e-5` | `3.6833248025036244e-5` | `1.0000211930780902` |
| u | 3 | `9.311678913140553e-3` | `9.311918024754345e-3` | `1.0000256786790032` |
| SSH | 3 | `7.074669630242024e-7` | `7.074669630674621e-7` | `1.0000000000611473` |
| T | 4 | `1.0594738051818496e-3` | `1.0594757154853101e-3` | `1.0000018030681375` |
| v | 4 | `1.5947842304474213e-2` | `1.594803608623198e-2` | `1.0000121509702733` |
| SSH | 4 | `5.06042337723115e-7` | `5.299775777930019e-7` | `1.0472988884241998` |
| u | 5 | `2.0733890435953528e-2` | `2.073403487719125e-2` | `1.0000069664319953` |
| S | 6 | `1.287536238418589e-4` | `1.2875696502347874e-4` | `1.0000259501947995` |
| v | 6 | `6.0219592087118606e-2` | `6.022050081599162e-2` | `1.0000150902528815` |
| SSH | 6 | `1.631023907114217e-4` | `1.6312992706670428e-4` | `1.0001688286429309` |
| S | 7 | `1.2244044884708371e-4` | `1.224430327208792e-4` | `1.0000211031062023` |
| u | 7 | `4.004140471842379e-2` | `4.004166157574929e-2` | `1.00000641479307` |
| v | 7 | `6.631577323114793e-2` | `6.631670463083134e-2` | `1.000014044919301` |
| SSH | 7 | `2.087141815218642e-4` | `2.087569352508757e-4` | `1.0002048434308572` |
| u | 8 | `4.718005909052282e-2` | `4.718072375983288e-2` | `1.0000140879287325` |
| v | 8 | `2.4527161866119307e-2` | `2.4527890741688735e-2` | `1.0000297170774755` |
| T | 9 | `5.445062142555964e-3` | `5.445067667597903e-3` | `1.0000010146885001` |
| u | 9 | `5.256039760377703e-2` | `5.2561503561392935e-2` | `1.000021041652391` |
| T | 10 | `5.636634494144276e-3` | `5.636643309707159e-3` | `1.0000015639763218` |
| u | 10 | `5.6249869570541774e-2` | `5.6250884173481036e-2` | `1.000018037427411` |

The mandatory post-S-21 cross-card rerun completed with a retained CPU JAX
cache after the original 20-minute attempts.  A fresh empty-cache trial emitted
the same XLA AOT `prefer-no-gather` / `prefer-no-scatter` host-feature warnings;
the processes completed without `SIGILL`, and the cache path
`/tmp/gyre-r20-jaxcache.Ccdm2p` is retained as run provenance.  The verdicts are:

```text
ORACLE_RELATIVE_COMPARE PASS: rows=9 max_worsening_ulps=0 first_over_bar='<absent>'->'<absent>' plant=None
ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=3563707596 first_over_bar={'fields': ['T', 'u'], 'kt': 2}->{'kt': 2, 'fields': ['T', 'u']} plant=None
ORACLE_RELATIVE_COMPARE PASS: rows=9 max_worsening_ulps=0 first_over_bar='<absent>'->'<absent>' plant=None
ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=1549621 first_over_bar={'fields': ['u'], 'kt': 4}->{'kt': 3, 'fields': ['T', 'u']} plant=None
```

The order is OVERFLOW stage, OVERFLOW trajectory, LOCK stage, LOCK trajectory.
Thus S-21 is unchanged at the single-step stage boundaries on both earlier
cards, but is not yet eligible to merge under Rule 12: after kt1 legoESM must
reconstruct NEMO's persistent `uu_b/vv_b(Kbb)` from the committed 3-D state.
The Round-20 implementation supplies zero for the stage-1 Kbb mean on every
step, which is exact only for the at-rest first step.  This is the first
unmatched operand at the new trajectory boundary and is registered as
**PERSISTENT_KBB_BAROTROPIC_MEAN_UNMEASURED**.  The source-exact S-21 operator
stays; it is not waived, reverted, or hidden behind a card switch.

The OVERFLOW trajectory has four AT-BAR-to-DEBT status changes (`S` at kt7,
kt8, kt9, and kt10) but its first-over-bar remains kt2 T/u.  LOCK has ten
AT-BAR-to-DEBT changes (`T,u` at kt3; `T` at kt4-6; `ssh` at kt6-10), and its
first-over-bar moves earlier from kt4 u to kt3 T/u.  Those are gate failures,
not rounding disclosures.

The following combined Rule-8/Rule-12 ledger enumerates every trajectory row
whose candidate differs from the preceding post-coastal legoESM baseline.
`move` is max |after-before| in row-scale oracle ulp; `worse` is max increase
in |candidate-NEMO| in the same ulp; `cells` is worsened / scored and `over`
counts cells beyond the two-ulp gate.  Stage rows are omitted because all 18
are exactly unchanged: for each card, six momentum rows are `0 / 16900`
(OVERFLOW) or `0 / 2540` (LOCK), the kt2 tracer row is `0 / 17000` or
`0 / 2560`, and the two kt2 momentum rows repeat the corresponding momentum
denominator.

| card | kt | field | move ulp | worse ulp | cells | over |
|---|---:|---|---:|---:|---:|---:|
| OVERFLOW | 3 | S | `2` | `2` | `9 / 17000` | 0 |
| OVERFLOW | 3 | T | `2.07127374e9` | `2.07127374e9` | `77 / 17000` | 74 |
| OVERFLOW | 3 | u | `3.05172106e8` | `2.67508999e8` | `283 / 16900` | 223 |
| OVERFLOW | 4 | S | `3` | `3` | `84 / 17000` | 4 |
| OVERFLOW | 4 | T | `3.56372962e9` | `3.56370760e9` | `127 / 17000` | 76 |
| OVERFLOW | 4 | ssh | `1.18449991e8` | `1.18430118e8` | `13 / 200` | 9 |
| OVERFLOW | 4 | u | `7.24995883e8` | `5.14919450e8` | `385 / 16900` | 270 |
| OVERFLOW | 5 | S | `4` | `4` | `155 / 17000` | 14 |
| OVERFLOW | 5 | T | `3.16022534e9` | `3.15967334e9` | `176 / 17000` | 88 |
| OVERFLOW | 5 | ssh | `5.50941930e8` | `5.50850572e8` | `19 / 200` | 13 |
| OVERFLOW | 5 | u | `7.01726734e8` | `2.85198822e8` | `424 / 16900` | 283 |
| OVERFLOW | 6 | S | `5` | `2` | `107 / 17000` | 0 |
| OVERFLOW | 6 | T | `2.16441218e9` | `2.16291456e9` | `165 / 17000` | 101 |
| OVERFLOW | 6 | ssh | `1.01200833e9` | `1.01200833e9` | `22 / 200` | 16 |
| OVERFLOW | 6 | u | `3.64492564e8` | `1.36508168e8` | `483 / 16900` | 327 |
| OVERFLOW | 7 | S | `6` | `4` | `178 / 17000` | 27 |
| OVERFLOW | 7 | T | `2.13179039e9` | `2.12905125e9` | `264 / 17000` | 102 |
| OVERFLOW | 7 | ssh | `9.37935692e8` | `9.37935692e8` | `24 / 200` | 18 |
| OVERFLOW | 7 | u | `4.27374910e8` | `1.87725884e8` | `627 / 16900` | 409 |
| OVERFLOW | 8 | S | `6` | `6` | `204 / 17000` | 10 |
| OVERFLOW | 8 | T | `2.81168887e9` | `2.80769391e9` | `302 / 17000` | 111 |
| OVERFLOW | 8 | ssh | `5.19056397e8` | `5.19056397e8` | `31 / 200` | 21 |
| OVERFLOW | 8 | u | `9.40292522e8` | `5.58641418e8` | `857 / 16900` | 457 |
| OVERFLOW | 9 | S | `7` | `7` | `228 / 17000` | 20 |
| OVERFLOW | 9 | T | `3.04286165e9` | `3.03821590e9` | `318 / 17000` | 124 |
| OVERFLOW | 9 | ssh | `4.63559930e8` | `4.63559930e8` | `37 / 200` | 23 |
| OVERFLOW | 9 | u | `1.22600098e9` | `9.18648236e8` | `1184 / 16900` | 479 |
| OVERFLOW | 10 | S | `8` | `5` | `221 / 17000` | 8 |
| OVERFLOW | 10 | T | `2.62242110e9` | `2.61828096e9` | `337 / 17000` | 134 |
| OVERFLOW | 10 | ssh | `8.66497437e8` | `8.66497437e8` | `41 / 200` | 24 |
| OVERFLOW | 10 | u | `1.01872543e9` | `9.34932649e8` | `1572 / 16900` | 531 |
| LOCK | 3 | T | `43668` | `43668` | `25 / 2560` | 23 |
| LOCK | 3 | u | `66.239666` | `66.238554` | `140 / 2540` | 60 |
| LOCK | 4 | T | `130878` | `130878` | `23 / 2560` | 21 |
| LOCK | 4 | ssh | `0.000244141` | `6.93889e-18` | `2 / 128` | 0 |
| LOCK | 4 | u | `276.813080` | `92.432362` | `179 / 2540` | 53 |
| LOCK | 5 | S | `1` | `0` | `0 / 2560` | 0 |
| LOCK | 5 | T | `261409` | `261409` | `24 / 2560` | 21 |
| LOCK | 5 | ssh | `1.320801` | `1.320313` | `10 / 128` | 0 |
| LOCK | 5 | u | `706.697968` | `274.002228` | `177 / 2540` | 45 |
| LOCK | 6 | S | `1` | `1` | `32 / 2560` | 0 |
| LOCK | 6 | T | `434936` | `434936` | `40 / 2560` | 24 |
| LOCK | 6 | ssh | `7.540527` | `7.540527` | `12 / 128` | 3 |
| LOCK | 6 | u | `1439.22144` | `606.521561` | `194 / 2540` | 45 |
| LOCK | 7 | S | `2` | `2` | `29 / 2560` | 0 |
| LOCK | 7 | T | `651176` | `651176` | `36 / 2560` | 27 |
| LOCK | 7 | ssh | `21.926758` | `21.926758` | `13 / 128` | 3 |
| LOCK | 7 | u | `2551` | `1130.95316` | `223 / 2540` | 54 |
| LOCK | 8 | S | `2` | `2` | `18 / 2560` | 0 |
| LOCK | 8 | T | `909420` | `909420` | `49 / 2560` | 28 |
| LOCK | 8 | ssh | `49.251953` | `49.251953` | `13 / 128` | 3 |
| LOCK | 8 | u | `4083.79208` | `1924.28589` | `216 / 2540` | 58 |
| LOCK | 9 | S | `1` | `1` | `3 / 2560` | 0 |
| LOCK | 9 | T | `1209134` | `1209134` | `50 / 2560` | 28 |
| LOCK | 9 | ssh | `100.823242` | `100.770508` | `14 / 128` | 4 |
| LOCK | 9 | u | `6130.55237` | `3006.86243` | `236 / 2540` | 61 |
| LOCK | 10 | S | `2` | `2` | `26 / 2560` | 0 |
| LOCK | 10 | T | `1549621` | `1549621` | `44 / 2560` | 34 |
| LOCK | 10 | ssh | `182.201172` | `182.078125` | `13 / 128` | 4 |
| LOCK | 10 | u | `8760.48010` | `4420.52179` | `220 / 2540` | 58 |

### Round-20 ASKED / UNASKED register

| choice or action | origin | disposition |
|---|---|---|
| sync local Git to reviewed merge `03c6e8d96` | ASKED | complete before edits |
| production-JIT returned-pytree 19-frame trace | ASKED item 0 | complete; 13 tests, plant exit 1 |
| explicit scalar-libm on every census probe | ASKED item 1 | complete; measured zero payload move |
| full census term-budget rerun | ASKED item 1 | UNMEASURED after 12-minute CPU budget |
| NEMO coastal stress factors | ASKED item 2 | source-exact shared change; dynamically inert on GYRE active faces |
| stored `uu_b/vv_b` stage-transport operand | ASKED ordered item 3 | CONFIRMED at GYRE kt1; committed but HOLD for landing pending persistent Kbb operand |
| V2 halo-only non-finite tolerance after owned-cell transform | UNASKED measurement-integrity fix | two discarded halo values disclosed; scored interiors remain fail-closed |
| direct-ratio / extra-rounding HPG experiments | UNASKED discriminators | refuted and reverted; no shipped change |
| kt1–10 repin | ASKED conditional | complete; first_over_bar kt2 unchanged |
| post-S-21 OVERFLOW/LOCK cross-card gates | ASKED process rule | complete; both stage gates PASS 0 ulp, both trajectory gates FAIL and second-error rows are registered |
| persistent `uu_b/vv_b(Kbb)` across legoESM steps | UNASKED boundary exposed by Rule 12 | next owner; UNMEASURED, no persistence-field implementation attempted |
| stage-3 transports, ZDF matrices/solutions, TKE | ASKED conditional | not entered because stage-2 Kaa remains DEBT |
| GYRE-only arm, tuning, gate criterion change, shipped NEMO edit | forbidden | none |

The machine-readable artifact ledger is
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round20.json`.
All Round-20 review remains **independent review outstanding**; these are
Codex-internal measurements and no dual-review claim is made.

## Round 21 — vertical-velocity coverage preregistration

Starting tip: `fa14627631d8a7babd4701b939b57cad360ca2ed` (Round 20, imported
only to `fidelity/nemo-testcases-l2-gyre-codex2`; integration remains at
`03c6e8d96ff7`).  Regime remains production JIT, CPU, binary64, Oracle V2,
and explicit scalar-libm.  No Round-17 Kmm seed from lane 3b is adopted.

### 21.1 preregistered stage-W clock discriminator

Source coverage found a real blind spot.  GYRE's existing
`oracle_transport_kt00000001_s{1,2,3}.bin` is emitted at
`stprk3_stg.F90:257-304`, before vector-invariant tracer advection calls
`tra_adv_trp`.  In the executed vector-invariant arm `zFw` is filled only by
`traadv.F90:220-235`, after its `wzv(...,np_transport)` call.  Therefore the
existing stage-1 `zFw` row is explicitly UNINFORMATIVE and no existing V2
record can score the tracer-consumed `ww` for all three stages.

NEMO sets the stage-local clock before those calls: `rn_Dt/3` at
`stprk3_stg.F90:118-124`, `rn_Dt/2` for stage 2, and `rn_Dt` for stage 3;
`sshwzv.F90:334-335` consumes its reciprocal in the QCO thickness-change
term.  legoESM builds one `_stage_transport_kw` with the full `dt` and reuses
it for all three `_nemo_ws_stage_transport` calls.  The preregistered rows are
the actual post-`tra_adv_trp` `ww` at stages 1, 2, and 3, over every wet
T/W-owned cell, printed as differing / scored.

Prediction: full-step production `dt` is DEBT at stages 1 and 2 and AT-BAR at
stage 3.  A private one-variable arm supplying `(rn_Dt/3,rn_Dt/2,rn_Dt)` must
make every over-bar stage row `0 / n`; a one-ulp wet-cell plant must exit 1.
The prediction is REFUTED if production is already AT-BAR in all three rows,
in which case the ORCA2 finding belongs to its gate/operand pairing and no
production fix lands.  The apparent contradiction with bit-exact stage-1 T/S
is not explained in advance: after `ww` is measured, the tracer tendency at
the affected cells must either be identically insensitive or the prior T/S
row must be retracted.

Because the required GYRE record is absent, this round first adds a
config-local WRITE-only `traadv.F90` extension and a non-overwriting `run.sh`.
The ordinary stage/restart files must hash identically to the pinned V2 record
before the new record is admitted.  Per the dispatch, Codex does not execute
NEMO from the sandbox.  No WZV clock change is authorized until the returned
record scores production over bar.

### 21.2 ordered stops

The ORCA2 phase-2i/2j handoff was searched at imported source tip
`3e425e24ded84e9bd1cbba7af492aaec43e99170`.  It identifies the first EEN
external-mode operand at `dynspg_ts.F90:1514-1570` and its consumer at
`:1685-1694`; all eight coefficient arrays are DEBT there.  This walk remains
ordered after the stage-W discriminator and is not entered while the GYRE
oracle acquisition is outstanding.  Likewise, lane 3b has not yet handed off
a dry-face NaN producer statement; the OVERFLOW NaN remains registered as
**UNMEASURED_PENDING_LANE3B_PRODUCER** and will never be masked at the
consumer.

The two pre-existing HOLD items remain explicit:

| boundary | status | required decision / next measurement |
|---|---|---|
| prognostic `uu_b/vv_b(Kbb)` | HOLD — design decision pending | NEMO declares and restarts it (`oce.F90:39,99`; `restart.F90:181,313`), writes it in `dynspg_ts.F90:862,879,890`, and reads it at `stprk3_stg.F90:262-263`; do not rederive |
| stage-2 Kaa | DEBT: U `2.1986806906376666e-15`, V `2.2380914396075147e-15` | resume the Krhs/update walk only after the stage-W coverage result |

### 21.3 acquisition handoff and stop

The V2 inventory confirms that no existing record can answer the registered
question.  `oracle_transport_kt00000001_s{1,2,3}.bin` carries an uninitialized
`zFw` in the vector-invariant branch, and the later
`oracle_rkstage3_wzv_kt00000001.bin` covers stage 3 only.  The new acquisition
therefore writes `rDt`, `ww`, and `pFw` immediately after
`traadv.F90:220-235` for each stage with magic `NEMO_L2_STGWW_1`; it never
reads a diagnostic value back into NEMO.

The user-executed, non-overwriting recipe is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round21_oracle/run.sh`.
It creates `GYRE_OMIP_L2_P3_SM_R21W` and
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round21_oracle_v2_stage_ww`,
copies the reviewed V2 deck and MY_SRC byte-for-byte, applies only
`traadv_round21.patch`, rebuilds with scalar math, refuses `_ZGV*`, and admits
the three new files only after every old `oracle_*.bin` plus the final restart
is byte-identical.  Codex did not run it.  Patch dry-run succeeded against the
reviewed V2 `traadv.F90` (`4807fa59...`); the resulting retained patch-check
copy hashes `a2ed0cf0e581a920fc9df01a56131ffa07d4c2d3c244d3d78542415068ae25df`.

The focused production-JIT gate is
`nemo_testcase_l2_gyre_stage_ww_gate.py`.  It refuses a missing record before
building a model:

```text
FAIL: missing /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round21_oracle_v2_stage_ww/oracle_rkstage_ww_kt00000001_s1.bin
```

That fail-closed transcript hashes
`4a142241914e29a7f3363bcac0e024b37efe7b2c1cfa402d597de3add3e98918`.
The shared model change is diagnostic-only: production still receives the
full step clock, while `_NEMOWSRK3TestHooks.source_stage_wzv_clock_arm`
changes only the WZV clock and
`expose_tracer_transport_as_ww` only changes the returned diagnostic pytree.
No constructible card selects either hook.  S-21 in the isomorphism map records
the arm and source references.

Verification is `80 passed in 11.46 s` for the GYRE gate, time-level registry,
and one-implementation tripwires.  The record reader's synthetic exact control
is `0 / 21,824`; its one-cell plant is `1 / 21,824`, DEBT.  Shell syntax,
Python compilation, and patch dry-run pass.  The science gate currently exits
2 on the deliberately absent oracle input; this is acquisition pending, not a
physics verdict.

Round-20's independent review verdict is **FIX-THEN-SHIP the delta, HOLD the
line**.  The review confirmed that `uu_b/vv_b` is prognostic and restartable
NEMO state, so re-derivation cannot certify S-21 beyond kt1.  Round 21 makes no
design choice and does not merge the Round-20 tip.  EEN is not entered out of
order.  The lane-3b NaN producer did not arrive during this round, so its
registered status remains UNMEASURED rather than being hidden by a mask.

### Round-21 ASKED / UNASKED register

| choice or action | origin | disposition |
|---|---|---|
| score tracer-consumed `ww` at stages 1-3 | ASKED item 1 | complete; production AT-BAR at all three stages |
| WRITE-only MY_SRC extension and `run.sh` | ASKED conditional | complete; non-overwriting, scalar-math, consumed-field admission PASS |
| private per-stage WZV-clock arm | ASKED one-variable discriminator | DEBT at all stages; refutes isolated clock replacement |
| fix the shared WZV clock | conditional ASKED | condition false; no fix landed |
| LOCK/OVERFLOW/ORCA2 cross-card WZV gates | conditional ASKED | not applicable because no shared change landed; ORCA2 ablation is consistency-only |
| EEN external-mode coefficient walk | ASKED item 2 | live-divisor arm improves all eight ORCA2 rows but leaves all DEBT; no change landed; HOLD at missing primitive oracle operands |
| OVERFLOW dry-face NaN producer | ASKED item 3 | UNMEASURED_PENDING_LANE3B_PRODUCER; no consumer mask added |
| prognostic `uu_b/vv_b` state | open USER DECISION | not implemented or rederived |
| lane-3b Round-17 Kmm seed | explicitly forbidden | not adopted |
| NEMO execution, shipped-source edit, merge, push | forbidden | none |

### 21.4 returned-record admission preregistration

The user-executed acquisition completed at
`round21_oracle_v2_stage_ww`, but the bytewise admission stopped on ten of the
49 inherited records.  Before inspecting payload values, the source/schema
prediction is: `BTORD_1 -> BTORD_2` is an intentional append-only instrument
schema change, while every same-tag difference is confined to a whole-array
halo or workspace slot which NEMO does not initialize before that particular
WRITE and which no gate scores.  The known example is `zFw` in the three
momentum-side transport records, written at `stprk3_stg.F90:343-348` before the
vector-invariant arm initializes it in `traadv.F90:220-235`.

The committed admission classifier will name every changed field and cell,
the config-local writer statement, and every repo parser that consumes it.
The prediction is CONFIRMED only if all common, parser-scored consumed fields
and the final restart are bit-identical.  One changed value in such a field
REFUTES WRITE-only admission and stops the round before the W-clock score.
If confirmed, the raw files stay unchanged and `run.sh` will compare the
schema-aware consumed fields rather than zero-filling NEMO workspaces; this is
the smaller authorized alternative and retains the evidence.  A planted
change to one consumed value must make the admission command exit nonzero.

The classifier confirms that prediction.  Search-before-build found no shared
schema-aware admission utility; the existing readers were reused as the
projection contract.  It reports `PASS: exact=39/49 changed=10
restart_equal=True`; the consumed-bit plant reports `FAIL` and exits 1.  The
ordinary kt10 restart is byte-identical (`6245d06d...`) and the Round-19
coefficient twin agrees on every consumed projection.  Raw evidence is retained
unchanged; no dump was zero-filled.

| record | raw bytes / first byte (1-based) | changed slot(s) | writer and consumption finding |
|---|---:|---|---|
| `bt_ordered_operands` | `256507` / `15`, plus `90112` appended | schema `BTORD_1 -> BTORD_2`; eight EEN arrays | all 43 common fields and weights bit-exact; current `MY_SRC/dynspg_ts.F90:598-599,934-942` appends the coefficients; the BTORD_2 twin is raw-identical |
| `rkstage1_transport_operands` | `826` / `472073` | `zub` 57 elements, `zvb` 54 | all changes outside the reader projection; the allocatables are assigned only on the bounded loops at `MY_SRC/stprk3_stg.F90:261-280` before the whole-array WRITE at `:297-302`; twin raw-identical |
| `rkstage3_wzv` | `5445` / `51` | `ww_pre` 7, `ww_post` 7, `pFw` 690 | all changes are discarded halos: WZV assigns its bounded domain at `sshwzv.F90:308-336`, and `pFw` at `MY_SRC/traadv.F90:234-235`, before the whole-array writes at `:230-236`; twin consumed projection exact |
| `rktracer_operands_s1` | `5371` / `1112885` | `zFw` 688 | discarded halo only; whole-array WRITE at `MY_SRC/stprk3_stg.F90:657-670`, after bounded `traadv.F90:229-235`; twin raw-identical |
| `rktracer_operands_s2` | `5377` / `928575` | `zFw` 690 | same bounded-writer halo; twin consumed projection exact |
| `slow_forcing` | `1660` / `1445335` | `utauU` 120, `vtauV` 120 | discarded halo only; `sbcmod.F90:542-547` assigns the bounded face arrays and `MY_SRC/stp2d.F90:219-230` writes the whole arrays; twin consumed projection exact |
| `tracer_transport_s3` | `5377` / `464319` | `zFw` 690 | discarded halo only; writer `MY_SRC/stprk3_stg.F90:622-626`; twin consumed projection exact |
| `transport_s1` | `42182` / `494890` | `zFw` 5456 | the entire slot is pre-consumer workspace (4034 changes even lie in the later parser projection): vector-invariant stage 1 skips WZV before the WRITE at `MY_SRC/stprk3_stg.F90:343-348`, and `tra_adv_trp` overwrites it later; the gate labels this row UNINFORMATIVE and never scores it; twin consumed fields exact |
| `transport_s2` | `5377` / `464307` | `zFw` 690 | same pre-`tra_adv_trp` slot; changed bytes are halo; twin consumed fields exact |
| `transport_s3` | `5377` / `464307` | `zFw` 690 | same pre-`tra_adv_trp` slot; changed bytes are halo; twin consumed fields exact |

Thus none of the ten contains a changed byte read by a scoring parser as a
consumed operand.  The varying bytes are proven uninitialized/stale at this
WRITE by the bounded producer statements, their variation across the two
independent binaries, exact consumed projections, and the exact final restart.
The diagnostic `zFw_nonfinite_count` is derived from those admitted changed
bytes; it describes an uninitialized workspace sample and is not comparable
between the Round-19 and Round-21 oracle binaries.
The acquisition is **ADMITTED_WRITE_ONLY** under Rules 1 and 8.  `run.sh` now
invokes the classifier; its original `EXP00` correction at `9019dc8e5ab` is
preserved unchanged.

For the size-changing `oracle_bt_ordered_operands` record specifically, the
admission compared every one of the 43 common scalar/array operands and all
barotropic weights in `BTORD_1` against their same-named `BTORD_2` fields;
each owned consumed value was bit-identical.  The growth from 636,712 to
726,824 bytes consists of eight newly appended EEN coefficient slots, not
altered pre-existing operands.

### 21.5 production-JIT stage-W result

The preregistered defect is **REFUTED**.  Against the admitted V2 records,
production is AT-BAR at every stage:

| stage | production max abs | unequal / wet | private stage-clock arm max abs | arm status |
|---:|---:|---:|---:|---|
| 1 | `1.6543612251060553e-23` | `10548 / 18000` | `3.9555664454351783e-7` | DEBT, `18000 / 18000` |
| 2 | `6.107901643091556e-21` | `17267 / 18000` | `1.9777832227175958e-7` | DEBT, `18000 / 18000` |
| 3 | `2.68242819152769e-17` | `18000 / 18000` | `3.1361945593164585e-8` | DEBT, `18000 / 18000` |

All figures are production JIT, CPU, binary64, scalar-libm.  The apparent
stage-1 tracer contradiction is resolved: legoESM's literal WZV call receives
the full-step `eta_after-eta_before` pair at
`ocean_model_latlon_cgrid.py:6103-6123`, so its full-step thickness delta and
full `dt` yield the same stretching rate as NEMO's stage-local delta and
stage-local `rDt` (`stprk3_stg.F90:123-124,177-178,221-222`;
`sshwzv.F90:334-335`).  The private arm changes only the denominator while
retaining the full-step delta, so it fails.  Its stage-3 row also inherits the
altered stage-1/2 tracer path because the arm is selected for the whole step.
The ORCA2 production-clock ablation therefore paired different operands and is
a **CONSISTENCY_PROBE**, not evidence for a shared production change.  No WZV
clock fix lands and no cross-card physics gate is required for this refuted
arm.

### 21.6 EEN external-mode coefficient preregistration

Search-before-build found the canonical shared source program
`_nemo_literal_een_coefficients` and the existing
`nemo_qco_live_vorticity_e3f_cgrid` implementation of
`e3f_0vor*(1+r3f)`.  The former currently constructs EEN's `q` from raw
`e3f_0`, bypassing the latter.  The imported ORCA2 production-JIT gate at
`3e425e24d` reproduces its handoff exactly: all eight fields are DEBT, first
`ffu_nw=8308 / 8568`, with maxima `6.983696920301593e-5` through
`8.320420480597231e-5`.

Preregistered one-variable arm: replace only EEN's divisor with NEMO's live
`e3f_vor` program (`dynvor.F90:918-950`; `dynspg_ts.F90:1514-1569`), holding
`ff_f`, face thicknesses, masks, metrics, association, and inputs fixed.  It is
CONFIRMED only if the ORCA2 eight-field gate becomes AT-BAR while GYRE's ENE
eight-field gate remains exact; otherwise it is refuted and no shared change
lands.  The tripolar north-fold application is part of the same source
operand, not a per-card physics switch.

### 21.7 EEN external-mode coefficient result

The preregistered complete-owner prediction is **REFUTED**.  The imported
ORCA2 gate ran production JIT on CPU in binary64 with explicit scalar-libm at
tip `3e425e24ded84e9bd1cbba7af492aaec43e99170`.  Its frozen coefficient record
is `oracle_bt_ene_coeff_kt00000001.bin`, SHA-256 `e7282ddc...`; it is written
after `dyn_cor_2D_init(Kmm)` and before the first coefficient consumer.  The
baseline and the one-variable live-divisor arm are:

| coefficient | baseline unequal / wet | baseline max abs | live-divisor unequal / wet | live-divisor max abs |
|---|---:|---:|---:|---:|
| `ffu_nw` | `8308 / 8568` | `8.056177274927652e-5` | `3219 / 8568` | `3.492369915092338e-5` |
| `ffu_ne` | `8304 / 8568` | `7.727286259909532e-5` | `3199 / 8568` | `3.9504115575330624e-5` |
| `ffu_sw` | `8331 / 8568` | `6.983696920301593e-5` | `3100 / 8568` | `1.2534393282444756e-5` |
| `ffu_se` | `8323 / 8568` | `7.253667168386182e-5` | `3134 / 8568` | `1.2479779528873167e-5` |
| `ffv_nw` | `8285 / 8589` | `8.320420480597231e-5` | `3113 / 8589` | `4.944551560061315e-5` |
| `ffv_ne` | `8299 / 8589` | `7.900097225433845e-5` | `3066 / 8589` | `4.8189779586498704e-5` |
| `ffv_sw` | `8310 / 8589` | `7.257335949618652e-5` | `3227 / 8589` | `4.0477968809950055e-5` |
| `ffv_se` | `8320 / 8589` | `7.788661883968118e-5` | `3231 / 8589` | `3.493630122319228e-5` |

Thus the raw-`e3f_0` divisor is a
**CONFIRMED_CONTRIBUTOR_NOT_COMPLETE_OWNER**: every maximum and mismatch count
moves toward NEMO, but no row reaches the bar.  Two further private arms—NEMO
source-rounding/order in the four-T-cell `e3f_0vor` average and precomputed
reciprocal multiplication in `r3f`—produce the exact same eight rows and are
refuted as the remaining owner.  They are measurements only; none is present
in the committed production tree.

The source boundary is now sharper.  NEMO forms `e3f_0vor` from the four
source-associated masked `e3t_0` terms at `dynvor.F90:918-941`, applies the
F-point lateral boundary/fold exchange at `:943`, falls back to `e3f_0` only
where zero at `:944-950`, and then `dynspg_ts.F90:1520-1569` consumes live
`e3f_vor` in the four U/V `zpvo` triads.  The handed-over record contains only
the eight final coefficients: it does not contain `e3f_0vor`, live `e3f_vor`,
or the individual `q=ff_f/e3f_vor` / `zpvo` operands.  Because the remaining
residual is widespread away from the north fold, selecting another recurrence
without those oracle inputs would be post-hoc and underdetermined.  The next
required acquisition is a config-local WRITE-only dump of those primitives at
`Kmm=1`, followed by a cellwise first-divergence comparison.  That acquisition
belongs to the ORCA2 configuration and is not fabricated from the final
coefficient record here.

GYRE's eight ENE coefficient fields remain bit-exact: its Round-21
`oracle_bt_ene_coeff` is one of the 39 raw-identical V2 records.  Rule 8 and
Rule 12 have zero production rows because the live-divisor arm did not meet
its landing condition and was reverted.  The ORCA2 baseline transcript hashes
`6f3981c13aec91269b413b0180786703370489a843bf31e576971e59079326d0`; the
live-divisor transcript hashes
`e881d2f7617a4d6aa017d38d39af6b89d3f179c5878c6bf0f9575a660cd5a43c`.
The source-rounding and reciprocal arms have that same latter hash, confirming
zero additional move.  Final focused verification is `21 passed in 5.31 s`
for the GYRE Phase-3 fidelity test module under CPU/x64.

All Round-21 measurements and harness results were Codex-internal when first
recorded.  The independent Round-21 review subsequently returned SHIP; Round
22 records that verdict without relabelling the original internal passes.

## Round 22 — ORCA2 shared-operator handoff

Starting tip: `e2378f057ca814bbdc0b6a8c8ec0e768c78cbdb2`.  Independent Round-21 review verdict:
**SHIP**.  It independently reproduced the consumed-field admission and WZV
mechanism.  The ORCA2 lane has retracted its invalid mixed-clock claim under
Rule 11 and is separately acquiring the EEN primitive operands requested in
Section 21.7.  The unresolved prognostic `uu_b/vv_b` design remains an open
user decision and is not implemented here.

### 22.1 source-first FCT precursor preregistration

A fresh detached probe worktree is pinned to ORCA2 handoff tip
`ce0f353e25a282b80ce4d134d1f937af24d1ed5e`; the dirty Round-21 EEN probe is
not read.  The ORCA2 production gate resolves `ln_traadv_fct=T`,
`nn_fct_h=2`, `nn_fct_v=2`, `nn_fct_imp=1`, and `ln_zad_Aimp=F`, matching the
GYRE card's collapsed FCT identity.  At RK3 stages 1 and 2,
`traadv.F90:280-283,355-365` disables the limiter and dispatches the FCT card
to the executed CEN2 precursor in `traadv_cen.F90:137-149,191-216`; the
two-step limiter in `traadv_fct.F90:153-189` is stage 3 only.  This round
therefore walks the actual CEN2 statements first rather than attributing a
stage-1 row to the unexecuted limiter.

The handed-off gate supplies NEMO's exact `zFu/zFv/zFw` and scores the stored
stage-1 tracer accumulator: T is `180882 / 228641`, maximum
`9.952637130238029e-21`; S is `190802 / 228641`, maximum
`1.4558378780933287e-20`.  The ORCA2 review found approximately 79% mismatch
at every level, including `6778 / 8613` bottom partial cells and
`174104 / 220028` non-bottom cells, with the worst row fractions in interior
latitudes 38–49.  This uniform localization refutes fold/coast/bottom-only
ownership and makes source association on non-uniform metrics the ranked
candidate.

The one-variable arm will replace only the shared CEN2 accumulator with the
written NEMO sequence: direct T-neighbour sums; left-associated
`(0.5*pU)*sum`; separate U and V face differences; their parenthesized sum;
multiplication by the precomputed `r1_e1e2t`; division by live `e3t(Kmm)`;
then the written vertical-flux recurrence and subtraction.  Each Fortran
assignment is guarded by `nemo_source_round`.  The arm holds the oracle
transports, tracer input, masks, metrics and stage-update formula fixed.

**CONFIRM:** ORCA2 reaches T/S `0 / 228641` against the supplied-input record;
the one-ULP tracer plant still exits nonzero.  **REFUTE:** either row remains
non-bit-exact, in which case the first differing statement/operand is named
and no shared change lands.  Only after confirmation may the shared identity
change be copied to this branch and subjected to GYRE kt=1–10 plus
LOCK/OVERFLOW cellwise Rule-12 gates.  Movement is tabulated before any owner
label.

### 22.2 subsequent ordered boundary

Only after the tracer precursor closes, walk
`GYRE_OWNER_SHARED_STAGE_TRANSPORT_SOURCE_ASSOCIATION`: NEMO's
`stprk3_stg.F90:259-275` `zub/zvb` and metric/e3/corrected-velocity products
against `_nemo_ws_stage_transport`.  The ORCA2 production handoff is up to
four ULP.  Its discriminator and Rule-12 gates are separate from the tracer
change.  Stage-3 FCT, BBL, ZDF and TKE remain outside this round until both
earlier boundaries clear.

### Round-22 ASKED / UNASKED register

| choice or action | origin | disposition at preregistration |
|---|---|---|
| fold two Round-21 admission clarifications | ASKED review minors | complete in Sections 21.4 and 21.7 |
| fresh ORCA2 probe at `ce0f353e25a2` | ASKED | created; old dirty probe ignored |
| source-literal stage-1 CEN2 precursor | ASKED item i | first ordered measurement |
| shared stage-transport association | ASKED item ii | conditional second measurement |
| ORCA2 per-level/bottom/row localization | ASKED if gate owned | recorded here as handoff; gate file remains ORCA2-owned unless the shared fix lands there |
| prognostic `uu_b/vv_b` state | open USER DECISION | not implemented |
| EEN primitive acquisition | ORCA2 lane in progress | not duplicated |
| shipped NEMO edit, NEMO run, card fork, push, merge | forbidden | none |

### 22.3 CEN2 precursor result — confirmed shared owner

The source-first prediction is **CONFIRMED**.  The fresh ORCA2 probe uses the
committed phase-2l gate at `ce0f353e25a2`, with NEMO's recorded transports,
tracer state, masks and metrics held fixed.  All figures below are
production-JIT, CPU, binary64, scalar-libm:

| one-variable boundary | T unequal / wet | S unequal / wet | T / S maximum absolute residual |
|---|---:|---:|---:|
| handoff baseline | `180882 / 228641` | `190802 / 228641` | `9.952637130238029e-21` / `1.4558378780933287e-20` |
| source association and per-statement rounding only | `80034 / 228641` | `81277 / 228641` | `6.776263578034403e-21` / `6.776263578034403e-21` |
| live `e3t(Kmm)` construction only | `176929 / 228641` | `187752 / 228641` | `9.952637130238029e-21` / `1.4558378780933287e-20` |
| complete literal CEN2 statement | **`0 / 228641`** | **`0 / 228641`** | **`0` / `0`** |

This is one shared operator identity, not a card arm: both individually
required source operands were ablated separately, and neither alone closed
the record.  The landed statement follows `traadv_cen.F90:137-149,191-216`
and the live T-cell thickness follows `domain.F90:158`,
`domqco.F90:159-161`, and `domzgr_substitute.h90:45-51,126`.  The code is
`vertical.py:43-82` plus
`ocean_model_latlon_cgrid.py:1437-1484`; the RK3 dispatch evidence remains
`traadv.F90:280-283,355-365`.  The one-cell scored-T plant exits 1.  The
baseline, final, and plant transcripts hash `3ea7c6d2...`, `7e1f1768...`, and
`a3cbfa30...`, respectively.  Classification:
**CONFIRMED_COMPLETE_OWNER_GIVEN_NEMO_TRANSPORTS**.

The ORCA2 reviewer localization is handed back to that lane rather than
duplicating its owned gate: approximately 79% mismatch at every level,
`6778 / 8613` bottom partial cells versus `174104 / 220028` non-bottom cells,
and the largest row fractions at interior `j=38..49`.  Those fifteen lines
belong in the ORCA2 phase-2l tracer gate/receipt on its next owner round.

The first cross-card run after this change is clean: LOCK stage `PASS 0 ulp`
(`9` rows), LOCK trajectory `PASS 0 ulp` (`50` rows, first-over-bar kt3 T/u),
OVERFLOW stage `PASS 0 ulp` (`9` rows), and OVERFLOW trajectory `PASS 0 ulp`
(`50` rows, first-over-bar kt2 T/u).  Their comparison artifacts are
`c540a87f...`, `0966173e...`, `3225a716...`, and `cc119217...`.

Rule 8 still requires disclosure of the GYRE trajectory's last-bit movement.
The following are every changed scored maximum from the Round-20 production
sweep to the post-CEN2 sweep; all unlisted rows have identical maxima, every
listed row remains DEBT, and first-over-bar remains kt2 T/S/u/v:

| row | before | after |
|---|---:|---:|
| kt5 u | `0.020734034877191249` | `0.020734034877191242` |
| kt5 ssh | `6.7010559036960113e-05` | `6.7010559036954909e-05` |
| kt6 u | `0.031134675691760226` | `0.031134675691760198` |
| kt6 ssh | `0.00016312992706670428` | `0.00016312992706689415` |
| kt7 u | `0.040041661575749288` | `0.040041661575749413` |
| kt7 v | `0.066316704630831336` | `0.066316704630831114` |
| kt7 ssh | `0.00020875693525087570` | `0.00020875693525002396` |
| kt8 u | `0.047180723759832878` | `0.047180723759818792` |
| kt8 v | `0.024527890741688735` | `0.024527890741687930` |
| kt8 ssh | `0.00025106275789497663` | `0.00025106275789159391` |
| kt9 S | `0.0054193833238400657` | `0.0054193833238329603` |
| kt9 u | `0.052561503561392935` | `0.052561503561377627` |
| kt9 v | `0.014532741725458046` | `0.014532741725461922` |
| kt9 ssh | `0.00021202403721123301` | `0.00021202403721025419` |
| kt10 T | `0.13238632692936037` | `0.13238632692936392` |
| kt10 S | `0.0057034654585876865` | `0.0057034654585805811` |
| kt10 u | `0.056250884173481036` | `0.056250884173407428` |
| kt10 v | `0.011171896160492037` | `0.011171896160381577` |
| kt10 ssh | `0.00021254041670566888` | `0.00021254041676015568` |

### 22.4 stage-transport source result — confirmed shared owner

The second ordered boundary is also **CONFIRMED**, specifically for the U/V
stage transports.  NEMO constructs stored QCO inverse depths at
`domqco.F90:175-181,219-222`, then evaluates
`zub=un_adv*r1_hu(Kmm)-uu_b` and
`zFu=(e2u*e3u)*(uu+zub*umask)` (V analogously) at
`stprk3_stg.F90:257-278`.  Reconstructing those statements directly with
NumPy and the dumped operands matches the oracle transport record bit for
bit; the alternative right-associated product differs in `78865` U and
`79690` V cells.

| one-variable boundary | zFu unequal / wet (max, ulp) | zFv unequal / wet (max, ulp) |
|---|---:|---:|
| post-CEN2 baseline | `102431 / 228641` (`4.656612873077393e-10`, 4) | `108307 / 228641` (`9.313225746154785e-10`, 4) |
| corrected-velocity/source association only | `85175 / 228641` (4 ulp) | `93486 / 228641` (5 ulp) |
| carry NEMO's stored QCO reciprocal | `14 / 228641` (`3.637978807091713e-12`, 1) | **`0 / 228641`** (`0`, 0) |
| reciprocal plus source-rounded live-face recurrence | **`0 / 228641`** (`0`, 0) | **`0 / 228641`** (`0`, 0) |

The source-rounded recurrence alone leaves the baseline transport rows
unchanged; it is the final one-variable discriminator after the reciprocal
handoff, closing the remaining fourteen U cells.  The implementation is
shared in `vertical.py:185-238,482-552` and
`ocean_model_latlon_cgrid.py:1288-1405,1428-1478`; S-21 in the branch
isomorphism map records the one owner and its NEMO references.  The exact
probe transcript hashes `7e2bf16f...`; its built-in vector/C2 selector plant
is rejected.  Classification:
**CONFIRMED_COMPLETE_OWNER_GIVEN_NEMO_STAGE_STATE**.

This does not close W.  The independently scored stage-W row remains DEBT at
`189969 / 228641`, maximum `3.48342299558331e-20`; it is retained as
`UNASSIGNED_PRODUCTION_WZV_DEBT`, not attributed to the now-exact U/V source.

The GYRE production-JIT sweep after this landing hashes `c5b231fb...`.
At kt1 it remains exactly `0 / 18000` T/S and `0 / 17400`, `0 / 17100`,
`0 / 600` for the three at-rest UNINFORMATIVE u/v/ssh rows.  At kt2 its
scored maxima are unchanged (`3.1956659540810506e-11` T,
`8.171241461241152e-13` S, `9.484089954776408e-7` u,
`8.987992592542841e-7` v, `4.336808689942018e-19` ssh); only T's unequal
count moves `12015 -> 12016`.  First-over-bar remains kt2 T/S/u/v.

Every changed maximum relative to the immediately preceding CEN2 sweep is
enumerated here for Rule 8; all statuses remain DEBT:

| row | before | after |
|---|---:|---:|
| kt3 u | `0.009311918024754345` | `0.0093119180247543468` |
| kt3 ssh | `7.0746696306746206e-7` | `7.0746696305163271e-7` |
| kt4 u | `0.014181918360118145` | `0.014181918360118131` |
| kt4 v | `0.015948036086231979` | `0.015948036086231965` |
| kt4 ssh | `5.2997757779300186e-7` | `5.2997757738924497e-7` |
| kt5 u | `0.020734034877191242` | `0.020734034877194454` |
| kt5 v | `0.043797500736911535` | `0.043797500736911521` |
| kt5 ssh | `6.7010559036954909e-5` | `6.7010559038582079e-5` |
| kt6 u | `0.031134675691760198` | `0.031134675691760726` |
| kt6 v | `0.060220500815991618` | `0.060220500815993450` |
| kt6 ssh | `0.00016312992706689415` | `0.00016312992706647071` |
| kt7 S | `0.0045105047068290105` | `0.0045105046992617304` |
| kt7 u | `0.040041661575749413` | `0.040041661575748205` |
| kt7 v | `0.066316704630831114` | `0.066316704630794060` |
| kt7 ssh | `0.00020875693525002396` | `0.00020875693634198446` |
| kt8 u | `0.047180723759818792` | `0.047180723759933117` |
| kt8 v | `0.024527890741687930` | `0.024527890741623953` |
| kt8 ssh | `0.00025106275789159391` | `0.00025106275951282181` |
| kt9 S | `0.0054193833238329603` | `0.0054193833238400657` |
| kt9 u | `0.052561503561377627` | `0.052561503561447530` |
| kt9 v | `0.014532741725461922` | `0.014532741725734780` |
| kt9 ssh | `0.00021202403721025419` | `0.00021202403923572025` |
| kt10 S | `0.0057034654585805811` | `0.0057034654585876865` |
| kt10 u | `0.056250884173407428` | `0.056250884172887441` |
| kt10 v | `0.011171896160381577` | `0.011171896159946744` |
| kt10 ssh | `0.00021254041676015568` | `0.00021254041667595740` |

### 22.5 cellwise cross-card result and Rule-12 debt

After the stage-transport landing, LOCK again passes stage (`9` rows) and
trajectory (`50` rows) at `0` oracle-relative worsening ulp and `0` movement
against the previous legoESM output; first-over-bar remains kt3 T/u.
OVERFLOW's stage gate likewise passes `9` rows at `0` ulp.  OVERFLOW's
trajectory gate is **FAIL**, with no status change and unchanged
first-over-bar kt2 T/u, but seven cells exceed the two-row-scale-ulp limit:

| newly registered boundary | first failing cell | oracle-residual worsening | row-scale ulp |
|---|---:|---:|---:|
| kt6 u | 500 | `4.51028103753969845e-16` | `2.031` |
| kt7 u | 501 | `4.51028103753969845e-16` | `2.031` |
| kt9 ssh | 19 | `1.38777878078144568e-15` | `6.250` |
| kt9 u | 480 | `6.66133814775093924e-16` | `3.000` |
| kt10 T | 606 | `1.42108547152020037e-14` | `4.000` |
| kt10 ssh | 18 | `8.32667268468867405e-16` | `3.750` |
| kt10 u | 478 | `4.99600361081320443e-16` | `2.250` |

The full comparison's largest cellwise worsening is `17.25` row-scale ulp
and its largest movement against the prior legoESM output is `41.875` ulp;
those aggregates occur at cells other than the first failing cell printed for
each row.  The comparison artifact enumerates all 22 rows with any movement,
including sub-threshold and improving cells, plus improved/worsened counts.
Its SHA-256 is `f6eec112...`; the residual NPZ is `9adeee2c...`.

Rule 12 applies: CEN2 and the U/V stage-transport statements are bit-exact
given NEMO's own ORCA2 inputs, so neither shared fix is reverted, hidden, or
guarded by card.  These seven OVERFLOW rows are a second downstream error
exposed at the trajectory boundary and are the next cross-card register.
The other verdict hashes are LOCK stage `ee5ad37b...`, LOCK trajectory
`76998204...`, and OVERFLOW stage `4f82b919...`.

### 22.6 verification, artifacts, and disposition

Focused tests pass: `44 passed in 347.46 s` across shared QCO faces,
WS-RK3 face use, and the no-scheme-duplication tripwire; the two direct CEN2
bit tests pass in `1.90 s`; and all 21 GYRE Phase-3 fidelity tests pass in
`4.65 s`.  The ORCA2 exact rows use the committed `ce0f353e25a2` gate in a
fresh probe worktree plus the two landed shared commits; diagnostic parser
extensions affect only additional printed operands, not the committed zF/FCT
scorers.  No NEMO executable was run and no shipped NEMO source was edited.

Round-22 commits are preregistration `820374d2dde2`, shared CEN2
`2ec14108995e`, and shared QCO reciprocal/source recurrence `8d679dee176e`.
The external evidence root is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round22/`; no multi-megabyte
artifact is committed.  The ORCA2 probe JSONs are retained under
`round22/orca2_probe`, the two GYRE sweeps under `round22/fct` and
`round22/stage_transport`, and all cross-card residual NPZs sit beside their
reports.  The manifest records full SHA-256 values rather than the abbreviated
receipt forms.

### Round-22 final ASKED / UNASKED register

| choice or action | origin | final disposition |
|---|---|---|
| source-literal stage-1/2 CEN2 precursor | ASKED item i | CONFIRMED and landed shared; ORCA2 T/S `0 / 228641` |
| source-literal U/V stage transport | ASKED item ii | CONFIRMED and landed shared; ORCA2 zFu/zFv `0 / 228641` |
| ORCA2 localization rows | ASKED conditional | recorded as an explicit handoff; ORCA2 gate remains lane-owned |
| cross-card gates after each shared change | ASKED / Rule 12 | run; CEN2 all PASS, final OVERFLOW trajectory FAIL and seven rows registered |
| prognostic `uu_b/vv_b` state | open USER DECISION | HOLD; not implemented |
| separate WZV recurrence | measured boundary | still DEBT; not mislabelled as U/V closure |
| EEN primitive acquisition | ORCA2 lane in progress | not duplicated |
| stage-3 FCT, BBL, ZDF, TKE | downstream | not entered across open boundaries |
| shipped NEMO edit, NEMO run, card fork, push, merge | forbidden | none |

Round-22 measurements and labels are Codex-internal.  Independent review of
this round remains **OUTSTANDING**.

## Round 23 — CEN2 `wmask` closure and OVERFLOW Rule-12 walk

### 23.1 preregistration (before measurement or implementation)

Starting point: `b2702b318f7e`.  The production regime remains CPU,
production JIT, fp64, scalar-libm, and GYRE oracle V2.  The open
`uu_b/vv_b` carried-state design is a user decision and is not implemented in
this round.  No state was read from the flagged ORCA2 probe worktrees; the
discriminating ORCA2 probe starts clean at `b6a6189c9357`.

The pre-implementation search found the single shared stage-1/2 CEN2 owner
`_nemo_cen2_tracer_rhs` and the existing ORCA2 supplied-transport gate; no
second advection implementation or scorer is introduced.  NEMO forms the
interior vertical flux as
`0.5*pW(k+1)*(pt(k+1)+pt(k))*wmask(k+1)` in
`traadv_cen.F90:201-210`, while `dommsk.F90:176,180` defines the surface and
interior W masks, with interior
`wmask(k)=tmask(k)*tmask(k-1)`.  The shared helper presently omits that final
factor.

Preregistered `wmask` outcomes:

- **CONFIRM_INERT_ON_OWNED_CELLS** if every ORCA2 and GYRE scored vertical
  face with `wmask=0` already has an exactly zero supplied `pW`, the literal
  factor moves zero owned/scored cells, and the ORCA2 supplied-input result
  remains T/S `0 / 228641`.
- **CONFIRM_LATENT_OMISSION** if any `wmask=0` face has nonzero `pW` or the
  literal factor moves a scored cell.  The factor then lands in the one shared
  CEN2 owner and must retain ORCA2 T/S `0 / 228641`; otherwise the first
  departing flux statement is reported and work stops rather than tuning.
- The direct behavioural plant removes or flips one W-mask cell and must fail.

The GYRE sweep comparison is preregistered cellwise against a clean
round-21b (`e2378f057ca8`) production run using the same diagnostic gate code.
The expected first movement is no earlier than the stage-1 CEN2 result feeding
kt2 T/S; kt1 initial/at-rest rows must remain `0 / n`, and first-over-bar must
remain kt2 T/S/u/v.  Every moved row, including improvements, will be listed
under Rules 8 and 12 with row-scale ulp provenance.  Any kt1 movement or an
earlier first-over-bar refutes that expectation.

The seven OVERFLOW trajectory violations are preregistered as a boundary
localization, not an owner claim.  Ranked candidates are: (a) partial-cell
bottom/adjacent-face use of the newly source-literal stored QCO reciprocal;
(b) its rounded live face-thickness recurrence; (c) a later trajectory
composition boundary, including the already open missing prognostic
`uu_b/vv_b` state.  Candidate (a) is confirmed only if the failing and
maximum-worsening cells cluster at zps bottom levels/columns and the first
departure given NEMO inputs is the reciprocal statement.  A uniform
interior-level distribution refutes it.  If the first departure is
OVERFLOW-specific geometry, it is registered and the walk stops as directed;
if it is the pending carried-state design, it remains HOLD and is not built.

No lane-3b dry-face producer has been handed over at preregistration time; the
NaN remains registered and no mask workaround is authorized.

### 23.2 literal `wmask` result — inert but retained

The shared CEN2 vertical-flux statement now carries NEMO's explicit
`wmask(k+1)` factor.  This is a source-completeness correction, not a new
option: the same `_nemo_cen2_tracer_rhs` remains the sole stage-1/2 owner.
The direct JIT bit test includes a dry lower T cell and fails if the factor is
removed.

The clean ORCA2 probe at `b6a6189c9357`, plus the already landed round-22
shared commits and this one-variable factor, counts `161733` dry inner W
faces.  Oracle `pW` is bit-zero on all `161733`, so the factor is
**CONFIRMED_INERT_ON_OWNED_CELLS**.  The supplied-input gate remains T
`0 / 228641` and S `0 / 228641`; its ordinary planted tracer cell exits 1 at
`1 / 228641`.  The report hash is `4fa0bfc6...` and the plant transcript hash
is `3b06f0d4...`.

Cross-card production-JIT comparisons against the round-22 tip are all exact
zero-move: LOCK stage `9 / 9` and trajectory `50 / 50` PASS, OVERFLOW stage
`9 / 9` and trajectory `50 / 50` PASS, with `0` row-scale ulp worsening,
`0` prior-legoESM movement, no status changes, and unchanged first-over-bar
(LOCK kt3 T/u; OVERFLOW kt2 T/u).  Their comparison hashes are respectively
`5992b7db...`, `524858b8...`, `c19b273f...`, and `ef076ec9...`; the residual
NPZ hashes are unchanged from round 22, which independently proves bit
identity.  Rules 8 and 12 therefore contain zero moved rows for this factor.
The isolated GYRE b2702b3-to-current production comparison also PASSes all
`50 / 50` rows with zero moved cells, zero worsening, no status changes, and
unchanged first-over-bar kt2 T/S/u/v.  Its before and after residual sidecars
are byte-identical (`459e6321...`); comparison SHA-256 is `c4831e1a...`.

### 23.3 GYRE production-JIT kt1--10 cellwise register

The GYRE gate now uses the shared residual recorder and `ulp_move_gate` for
its own trajectory, persisting oracle, candidate, and absolute-residual cells
in a compressed hashed NPZ.  Reduction-only metadata (`n_unequal` and the
relative/max summaries) remains disclosed but is not a second admission
criterion; structural metadata remains fail-closed.  A direct helper test
pins that rule.  The round-21b baseline was reproduced from clean model code
at `e2378f057ca8` while executing this exact diagnostic gate, so scorer
evolution is not confounded with physics evolution.

Here is the complete current register.  Each cell reads
`unequal / owned ; normalized L-inf residual ; status`; every number is CPU,
production-JIT, fp64, scalar-libm, against GYRE oracle V2:

| kt | T | S | u | v | ssh |
|---:|---|---|---|---|---|
| 1 | `0/18000;0;AT-BAR` | `0/18000;0;AT-BAR` | `0/17400;0;UNINFORMATIVE` | `0/17100;0;UNINFORMATIVE` | `0/600;0;UNINFORMATIVE` |
| 2 | `12016/18000;1.36147e-12;DEBT` | `11238/18000;2.21811e-14;DEBT` | `17400/17400;9.48409e-07;DEBT` | `17100/17100;8.98799e-07;DEBT` | `600/600;4.33681e-19;AT-BAR` |
| 3 | `18000/18000;0.000372234;DEBT` | `17938/18000;3.68332e-05;DEBT` | `17400/17400;0.00931192;DEBT` | `17100/17100;0.00471784;DEBT` | `600/600;7.07467e-07;DEBT` |
| 4 | `18000/18000;0.00105948;DEBT` | `17996/18000;8.22573e-05;DEBT` | `17400/17400;0.0141819;DEBT` | `17100/17100;0.015948;DEBT` | `600/600;5.29978e-07;DEBT` |
| 5 | `18000/18000;0.00305374;DEBT` | `17999/18000;8.61133e-05;DEBT` | `17400/17400;0.020734;DEBT` | `17100/17100;0.0437975;DEBT` | `600/600;6.70106e-05;DEBT` |
| 6 | `18000/18000;0.00381767;DEBT` | `17999/18000;0.000128757;DEBT` | `17400/17400;0.0311347;DEBT` | `17100/17100;0.0602205;DEBT` | `600/600;0.00016313;DEBT` |
| 7 | `18000/18000;0.00454667;DEBT` | `18000/18000;0.000122443;DEBT` | `17400/17400;0.0400417;DEBT` | `17100/17100;0.0663167;DEBT` | `600/600;0.000208757;DEBT` |
| 8 | `18000/18000;0.00509837;DEBT` | `18000/18000;0.000136098;DEBT` | `17400/17400;0.0471807;DEBT` | `17100/17100;0.0245279;DEBT` | `600/600;0.000251063;DEBT` |
| 9 | `18000/18000;0.00544507;DEBT` | `18000/18000;0.000147117;DEBT` | `17400/17400;0.0525615;DEBT` | `17100/17100;0.0145327;DEBT` | `600/600;0.000212024;DEBT` |
| 10 | `18000/18000;0.00563664;DEBT` | `18000/18000;0.000154831;DEBT` | `17400/17400;0.0562509;DEBT` | `17100/17100;0.0111719;DEBT` | `600/600;0.00021254;DEBT` |

First-over-bar remains kt2 T/S/u/v.  The cellwise round-21b-to-current
comparison is **FAIL**: 42 rows move, 39 contain at least one cell worsening
by more than two row-scale ulp, no row changes status, and first-over-bar does
not move.  The combined largest worsening is `140135.640625` row-scale ulp.
The controlled intermediate at `2ec14108995e` attributes 34 moved / 29
Rule-12 rows to source-literal CEN2 and the subsequent stored-QCO-reciprocal
landing contributes movement in 42 / Rule-12 debt in 39 rows.  Both operators
are bit-exact given NEMO's own ORCA2 inputs, so Rule 12 keeps both shared fixes
and registers the exposed downstream GYRE debt.

Every moved row is enumerated below.  `I/W` is the count of cells whose oracle
residual improved/worsened; `>2` is the number violating Rule 12; the last
column classifies each one-variable intermediate as `0`, sub-threshold
`moved`, or `DEBT`:

| kt | field | round-21b norm | current norm | I / W | >2 | max worsening ulp | CEN2 / stored QCO reciprocal |
|---:|---|---:|---:|---:|---:|---:|---|
| 2 | S | `2.21811013e-14` | `2.21811013e-14` | 1 / 3 | 0 | `1` | 0 / moved |
| 2 | T | `1.36147368e-12` | `1.36147368e-12` | 1 / 2 | 0 | `1` | 0 / moved |
| 3 | S | `3.6833248e-05` | `3.6833248e-05` | 12 / 17 | 2 | `3` | 0 / DEBT |
| 3 | T | `0.000372234099` | `0.000372234099` | 21 / 27 | 2 | `5` | 0 / DEBT |
| 3 | ssh | `7.07466963e-07` | `7.07466963e-07` | 317 / 281 | 0 | `0.508789` | 0 / moved |
| 3 | u | `0.00931191802` | `0.00931191802` | 8349 / 8163 | 4 | `3.63281` | 0 / DEBT |
| 3 | v | `0.00471783625` | `0.00471783625` | 7983 / 7717 | 6 | `4.0625` | 0 / DEBT |
| 4 | S | `8.22572821e-05` | `8.22572821e-05` | 158 / 211 | 19 | `448` | moved / DEBT |
| 4 | T | `0.00105947572` | `0.00105947572` | 305 / 318 | 79 | `559` | moved / DEBT |
| 4 | ssh | `5.29977578e-07` | `5.29977577e-07` | 357 / 243 | 23 | `19.9434` | 0 / DEBT |
| 4 | u | `0.0141819184` | `0.0141819184` | 6336 / 10872 | 348 | `329.401` | moved / DEBT |
| 4 | v | `0.0159480361` | `0.0159480361` | 9371 / 7352 | 356 | `329.45` | moved / DEBT |
| 5 | S | `8.61133488e-05` | `8.61133488e-05` | 630 / 655 | 276 | `4873` | DEBT / DEBT |
| 5 | T | `0.00305373707` | `0.00305373707` | 1450 / 1329 | 611 | `5551` | DEBT / DEBT |
| 5 | ssh | `6.7010559e-05` | `6.7010559e-05` | 276 / 324 | 278 | `352.645` | moved / DEBT |
| 5 | u | `0.0207340349` | `0.0207340349` | 8130 / 9256 | 1938 | `1643.76` | DEBT / DEBT |
| 5 | v | `0.0437975007` | `0.0437975007` | 9040 / 8039 | 1813 | `1643.87` | DEBT / DEBT |
| 6 | S | `0.000128756965` | `0.000128756965` | 1860 / 1750 | 627 | `11554` | DEBT / DEBT |
| 6 | T | `0.00381766932` | `0.00381766932` | 4685 / 4499 | 2638 | `14377` | DEBT / DEBT |
| 6 | ssh | `0.000163129927` | `0.000163129927` | 292 / 308 | 295 | `2083.49` | DEBT / DEBT |
| 6 | u | `0.0311346757` | `0.0311346757` | 8525 / 8866 | 6603 | `2033.95` | DEBT / DEBT |
| 6 | v | `0.0602205008` | `0.0602205008` | 8533 / 8567 | 6520 | `2500.37` | DEBT / DEBT |
| 7 | S | `0.000122443033` | `0.000122443033` | 3558 / 3559 | 1124 | `11052` | DEBT / DEBT |
| 7 | T | `0.00454666567` | `0.00454666567` | 6747 / 7150 | 4949 | `15637` | DEBT / DEBT |
| 7 | ssh | `0.000208756935` | `0.000208756936` | 333 / 267 | 263 | `4913.92` | DEBT / DEBT |
| 7 | u | `0.0400416616` | `0.0400416616` | 8540 / 8860 | 8633 | `4724.72` | DEBT / DEBT |
| 7 | v | `0.0663167046` | `0.0663167046` | 8616 / 8484 | 8262 | `17666.8` | DEBT / DEBT |
| 8 | S | `0.000136097664` | `0.000136097664` | 5307 / 5373 | 2240 | `18292` | DEBT / DEBT |
| 8 | T | `0.00509837336` | `0.00509837336` | 7744 / 8231 | 6477 | `47285` | DEBT / DEBT |
| 8 | ssh | `0.000251062758` | `0.00025106276` | 310 / 290 | 289 | `7286.13` | DEBT / DEBT |
| 8 | u | `0.0471807238` | `0.0471807238` | 8637 / 8763 | 8727 | `41486.2` | DEBT / DEBT |
| 8 | v | `0.0245278907` | `0.0245278907` | 8604 / 8496 | 8456 | `140136` | DEBT / DEBT |
| 9 | S | `0.000147117483` | `0.000147117483` | 6473 / 6324 | 3143 | `22484` | DEBT / DEBT |
| 9 | T | `0.00544506767` | `0.00544506767` | 8392 / 8293 | 6807 | `33180` | DEBT / DEBT |
| 9 | ssh | `0.000212024037` | `0.000212024039` | 305 / 295 | 295 | `10138.1` | DEBT / DEBT |
| 9 | u | `0.0525615036` | `0.0525615036` | 8480 / 8920 | 8900 | `12681.9` | DEBT / DEBT |
| 9 | v | `0.0145327417` | `0.0145327417` | 8656 / 8444 | 8420 | `43943.4` | DEBT / DEBT |
| 10 | S | `0.0001548308` | `0.0001548308` | 7076 / 6897 | 3569 | `20561` | DEBT / DEBT |
| 10 | T | `0.00563664331` | `0.00563664331` | 8696 / 8317 | 6759 | `47791` | DEBT / DEBT |
| 10 | ssh | `0.000212540417` | `0.000212540417` | 308 / 292 | 290 | `17811.4` | DEBT / DEBT |
| 10 | u | `0.0562508842` | `0.0562508842` | 8631 / 8769 | 8749 | `14499.6` | DEBT / DEBT |
| 10 | v | `0.0111718962` | `0.0111718962` | 8511 / 8589 | 8572 | `91789.3` | DEBT / DEBT |

Artifacts: baseline JSON/NPZ `f63c7be5...` / `aca765c0...`, post-CEN2
JSON/NPZ `c14a3541...` / `4d6c6341...`, current JSON/NPZ `3cd8332f...` /
`459e6321...`, combined comparison `c55dcfc7...`, and the isolated stored-QCO
comparison `aaa7b220...`.  The baseline and both intermediates are retained
outside git under the round-23 root.

### 23.4 OVERFLOW seven-row localization and source boundary

The committed localization probe reconstructs the persisted active-cell order
from the exact OVERFLOW masks and fails closed if payload cardinality changes.
Its planted one-cell mask shift exits 1 with `active-cell map does not match
persisted payload`; transcript SHA-256 is
`4292123bc457c8ed5e5ca4385b144090b732f9aa415dfea03e8cb87ee3793fa1`.
Coordinates below are legoESM-owned zero-based `(j,i,k)`; the probe also emits
the corresponding two-halo, Fortran-one-based NEMO coordinates.  All cells are
on the only wet interior row, `j=1`.

| row | `>2` cells | max worsening, row-scale ULP | first / maximum cell | levels, zero based | bottom / non-bottom | columns `(j,i):count` |
|---|---:|---:|---|---|---:|---|
| kt6 u | 8 | `2.0625` | `(1,21,0)` / `(1,21,1)` | `0:7` once each | `0 / 8` | `(1,21):8` |
| kt7 u | 2 | `2.03125` | `(1,21,1)` / `(1,21,1)` | `1,5` once each | `0 / 2` | `(1,21):2` |
| kt9 ssh | 2 | `6.75` | `(1,20)` / `(1,22)` | surface | n/a | `(1,20):1; (1,22):1` |
| kt9 u | 22 | `4.84375` | `(1,20,5)` / `(1,21,1)` | `0,1,5:24` once each | `1 / 21` | `(1,20):20; (1,21):2` |
| kt10 T | 1 | `4` | `(1,25,4)` / same | `4` | `0 / 1` | `(1,25):1` |
| kt10 ssh | 4 | `17.25` | `(1,19)` / `(1,20)` | surface | n/a | `(1,19/20/22/26):1` each |
| kt10 u | 38 | `4.28125` | `(1,20,3)` / `(1,21,2)` | `0:2` once; `3` twice; `4:12` once; `13:24` twice | `2 / 36` | `(1,20):22; (1,21):4; (1,22):12` |

Across the 71 three-dimensional violations, only 3 are at the face/column
bottom and 68 are above it.  The levels span the surface through level 24 and
the columns form an advancing-front cluster at `i=19:26`; this is
**REFUTED_PARTIAL_CELL_BOTTOM_CLUSTER**, not an OVERFLOW-zps bottom-geometry
owner.  The exact machine-readable census is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round23/overflow_rule12/localization.json`,
SHA-256
`1e8a70a2b6df820a31ab6432e902334692afea36a04ce5f1953241b4c0b13422`.

**Rule-11 retraction (Round 25):** the ordered-walk conclusion immediately
above was **REFUTED AND ILL-POSED** for the stage-2 and kt1 rows.  Although
`uu_b/vv_b` really is prognostic (`oce.F90:39,99`) and is read by S-21,
`stp_2D` first writes Naa at `stprk3.F90:186`; the stage-1 swap at :197 makes
that same-step result Nnn before stages 2 and 3 at :200-207.  Those stages
never lacked their `uu_b/vv_b(Nnn)` operand.  The carried Kbb pair has live
scope only at a subsequent step's seed and stage-1 S-21.  This dead attribution
is retained next to the source-order fact that killed it and the localization
probe now emits the same retraction instead of the obsolete boundary label.

No lane-3b dry-face NaN producer was handed over during Round 23.  It remains
registered; no downstream mask workaround was made.

### Round-23 ASKED / UNASKED register

| choice or action | origin | final disposition |
|---|---|---|
| literal CEN2 `wmask` factor | ASKED item 1 | landed shared; inert because supplied `pW=0` on all 161733 dry W faces; ORCA2 T/S `0 / 228641` |
| ORCA2 and cross-card reruns for `wmask` | ASKED item 1 / Rule 12 | ORCA2 exact with failing plant; LOCK/OVERFLOW stage and trajectory all 0-ULP PASS |
| GYRE kt1--10 cellwise sweep | ASKED item 2 | complete; kt1 `0 / n`, first-over-bar kt2 unchanged; 42 moved and 39 Rule-12 rows fully enumerated |
| OVERFLOW seven-row localization | ASKED item 3 | complete; partial-cell-bottom candidate refuted, source walk stops at absent prognostic `uu_b/vv_b(Kbb)` operand |
| prognostic `uu_b/vv_b` carried state | open USER DECISION | HOLD; not implemented, reconstructed, waived, or hidden behind a card switch |
| lane-3b dry-face NaN producer | conditional ASKED item 4 | no handoff received; remains registered, no mask workaround |
| stage-3/ZDF/TKE continuation | downstream | not entered across the carried-state HOLD and existing stage-2 debt |
| shipped NEMO edit, NEMO run, multi-MB git artifact, push, merge | forbidden | none |

Round-23 measurements and labels are Codex-internal.  Independent review of
this round remains **OUTSTANDING**.

The focused CPU verification is `74 passed in 8.63s`: the literal CEN2 bit
test, `ulp_move_gate`, the GYRE phase-3 gate tests, and the one-implementation
tripwire.  The committed Round-23 manifest is
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round23.json`,
SHA-256
`74936c85585699e215b5cade81bb9669024c2de059a3a0eb07ddf5a273f381b3`.

## Round 24 preregistration — Decision 8 prognostic depth-mean state

User Decision 8 (ASKED, 2026-09-05) authorizes exactly one U-face/V-face pair
on NEMO-identity states: NEMO's prognostic `uu_b/vv_b`.  The source contract is
`oce.F90:39,99` (declaration/allocation), `istate.F90:143-167` (rest-start
depth mean and Kbb-to-Kmm copy), `dynspg_ts.F90:484-500` (Kmm/Kbb external-mode
seed), `dynspg_ts.F90:857-897` (Kaa write),
`stprk3_stg.F90:257-274` (Kmm S-21 transport correction),
`stprk3_stg.F90:115-228` (stage Kaa ownership), and
`stprk3.F90:195-213` (stage rotations and final Naa/Nbb swap).  Restart writes
Kbb at `restart.F90:175-182`, reads it at `restart.F90:304-314`, and permits
the depth-mean reconstruction only when `uu_n/vv_n` is missing at :316-330.
The shared state will follow exactly that contract; reconstruction is a loud
legacy-restart fallback, never the identity live-step route.

The existing Rule-1d registry entry for
`oracle_bt_frames_ktNNNNNNNN.bin` is `after`: the instrumented
`stprk3.F90:344-352` writes explicit Kaa `uu_b/vv_b` plus `un_adv/vn_adv`
after `stp_2D`.  The final `stprk3.F90:213` swap makes that Kaa pair the next
step's Kbb.  Thus the existing ten oracle records are sufficient; no NEMO run
is requested.

The one-variable arm replaces live re-derivation by the carried pair.  A
rest-start run has a strict causality control: kt1 consumes the same exact
zero pair as the current implementation and merely produces Kaa; kt2 consumes
that pair; therefore kt3 entry is the first ordinary prognostic frame allowed
to move.  Consequently the preregistered kt1 stage-2 Kaa residuals
(`2.1986806906376666e-15` U, `2.2380914396075147e-15` V) and the kt2
first-over-bar T/S/u/v rows must remain unchanged.  A change in either is a
failure of isolation, not an improvement.  The carried Kaa itself must be
`0 / n` against every available oracle frame.  GYRE kt3--10 and the seven
OVERFLOW Rule-12 rows are predicted to move toward NEMO, without presuming
that unrelated open operators clear.  LOCK and any multi-step ORCA2 rows may
first move at kt3; their rest-start single-stage rows must remain 0 ulp.  The
C1D coupled-slab kt2 `PRE_SSM.u=1.1172865e-7` row is likewise predicted
unchanged because it precedes the first carried-state effect.

Non-NEMO constructors must retain `None/None`, adding no array pytree leaf.
The paired-field validation, one-cell three-ulp plant, restart missing-field
plant, and full-state restart continuation are the preregistered controls.
The machine-readable preregistration is
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round24_preregister.json`.

## Round 24 result — Decision 8 carried barotropic state

Starting tip: `498b10e5aa01ec88c77f176289a1e1da5890664c`.
Preregistration: `e2f2246f7`.  Shared implementation: `b6a0d6e9f`.
The reporting-only equal-input/downstream distinction is `1ee449046`; the
generic cross-card state gate and its absent-face correction are
`15eef39f0`, `9c4324ce5`, and `e1397b5a0`.

### Source contract and implementation

| NEMO contract | executed source | legoESM disposition |
|---|---|---|
| separately allocated prognostic pair | `oce.F90:39,99` | optional U-face/V-face `uu_b/vv_b` fields on `LatLonCGridOceanState` |
| rest initial depth mean and Kbb-to-Kmm copy | `istate.F90:143-167` | exact zero pair on the shipped rest-start cards; explicit opt-in constructor argument |
| external-window seed | `dynspg_ts.F90:484-500` | the standard and wide-halo split-explicit solvers read the pair directly; a partial pair raises |
| external Kaa write | `dynspg_ts.F90:857-897` | the solver returns its prognostic `U_bar_avg/V_bar_avg` into the pair |
| S-21 Kmm correction | `stprk3_stg.F90:257-274` | `_nemo_ws_stage_transport` reads entry Kbb at stage 1 and the live external Kaa target thereafter |
| stage rotation/final swap | `stprk3.F90:195-213` | the returned whole-step pair is next-step Kbb; the Rule-1d entry names this swap |
| restart write/read | `restart.F90:175-182,304-314` | the pair is in the prognostic restart inventory and round-trips bitwise |
| old-restart compatibility | `restart.F90:315-330` | missing pair alone may invoke a logged mesh-weighted reconstruction; never used by the live identity step |

The state API contains one pair only.  A NEMO identity recipe opts in; all
other recipes retain `None/None`, so JAX sees no additional array leaves.
Every direct constructor, `_replace`, bridge, restart pack/unpack and test
constructor found by the repository-wide census was audited: central NEMO
constructors were opted in, while direct non-NEMO keyword constructors retain
the pair's `None` defaults without edits.
The generic NEMO restart reader maps `uu_n/vv_n` directly.  The bridge rejects
a half-pair, and the legacy missing-field reconstruction requires the exact
`e3u_0/e3v_0`, masks and `hu_0/hv_0` operands and emits a warning.  The C1D
and ORCA2 fresh-probe cards each needed only the same constructor opt-in,
demonstrating that the API does not assume a multi-column domain.

The merged state had four existing TKE carry fields not yet classified by the
restart policy (`tke_avm`, `tke_avt`, `tke_dissl`, `tke_avm_surface`).  The
required exhaustive restart inventory now classifies them as prognostic.  No
TKE arithmetic changed; this is a necessary fail-closed policy closure exposed
by adding the new pair.

### Preregistered causality correction — Rule-11 retraction

The preregistration above and Round 23's “first unavailable operand =
`uu_b/vv_b(Kbb)`” claim for stage-2 and kt1 rows are **REFUTED AND
ILL-POSED**.  `stp_2D` runs once before the stages and writes Naa
(`stprk3.F90:186`); stage 1 is called with Kmm=Nbb at :195, then the :197 swap
promotes the just-solved Naa slot to Nnn, which stages 2 and 3 read at :200-207.
Stage 2 therefore consumes `uu_b/vv_b(Nnn)`, a same-step value that was never
unavailable.  The carried Kbb pair can move only the next step's external-mode
seed (`dynspg_ts.F90:484-500`) and stage-1 S-21 correction
(`stprk3_stg.F90:257-274`).  The historical preregistration is retained next
to this dead finding so it cannot be rediscovered.

### GYRE production-JIT register against oracle V2

Regime for every row below: CPU production JIT, fp64/x64,
`transcendentals="libm"`, oracle V2.  Values are normalized maximum absolute
residuals.  The ordinary first-over-bar remains kt2 T/S/u/v.

| kt | T | S | u | v | ssh | over-bar |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 0 | 0 | 0 | 0 | none |
| 2 | 1.3614736849003888e-12 | 2.2181101297999213e-14 | 9.4840899547758662e-7 | 8.9879925925425699e-7 | 4.3368086899420177e-19 | T, S, u, v |
| 3 | 3.7223441372709554e-4 | 3.683246743167771e-5 | 9.3116789131405323e-3 | 4.7190775913081018e-3 | 7.0746696306789574e-7 | T, S, u, v, ssh |
| 4 | 1.0594738051818496e-3 | 8.2257404626745841e-5 | 1.4182404011958687e-2 | 1.5947842304474213e-2 | 5.0604233765936396e-7 | T, S, u, v, ssh |
| 5 | 3.0537424781427133e-3 | 8.6118264313010001e-5 | 2.0733890435948851e-2 | 4.3797665645095095e-2 | 6.7018510229067615e-5 | T, S, u, v, ssh |
| 6 | 3.817675809918862e-3 | 1.287536238418589e-4 | 3.1134729732957417e-2 | 6.0219592087125712e-2 | 1.631023908310134e-4 | T, S, u, v, ssh |
| 7 | 4.5466665539785865e-3 | 1.2244044906195795e-4 | 4.0041404718354365e-2 | 6.631577323112417e-2 | 2.0871418176652678e-4 | T, S, u, v, ssh |
| 8 | 5.098376094806587e-3 | 1.3609844305084869e-4 | 4.7180059090506474e-2 | 2.452716186617937e-2 | 2.5108247651913873e-4 | T, S, u, v, ssh |
| 9 | 5.4450621425559641e-3 | 1.4711821745815303e-4 | 5.2560397603943147e-2 | 1.4533612266875308e-2 | 2.1203002233072732e-4 | T, S, u, v, ssh |
| 10 | 5.6366344941442756e-3 | 1.5483121354424481e-4 | 5.6249869570708808e-2 | 1.1173365767016698e-2 | 2.125543565721557e-4 | T, S, u, v, ssh |

The new pair is a separately scored boundary.  Only kt1 has equal inputs and
therefore supports an operator-identity label; kt2--10 are honest downstream
consistency rows after the full prognostic states differ.

| kt | uu_b unequal / wet n | max abs | vv_b unequal / wet n | max abs | label |
|---:|---:|---:|---:|---:|---|
| 1 | 0 / 580 | 0 | 0 / 570 | 0 | equal-input identity AT-BAR |
| 2 | 580 / 580 | 5.0535482543209373e-8 | 570 / 570 | 4.8148447874866128e-8 | downstream DEBT |
| 3 | 580 / 580 | 3.0619862773272746e-7 | 570 / 570 | 3.2257736902414685e-7 | downstream DEBT |
| 4 | 580 / 580 | 4.9641838429596158e-7 | 570 / 570 | 9.8709409025884119e-7 | downstream DEBT |
| 5 | 580 / 580 | 7.3468773024445381e-7 | 570 / 570 | 1.8549364711026219e-6 | downstream DEBT |
| 6 | 580 / 580 | 1.2136151661190707e-6 | 570 / 570 | 2.165814754655172e-6 | downstream DEBT |
| 7 | 580 / 580 | 1.7080553144873939e-6 | 570 / 570 | 2.314422217679986e-6 | downstream DEBT |
| 8 | 580 / 580 | 1.0942452698521171e-6 | 570 / 570 | 1.7587407682504569e-6 | downstream DEBT |
| 9 | 580 / 580 | 1.2129572566003363e-6 | 570 / 570 | 1.5503292519388612e-6 | downstream DEBT |
| 10 | 580 / 580 | 1.127403774763263e-6 | 570 / 570 | 1.050722699821535e-6 | downstream DEBT |

The independently compiled stage-2 gate is unchanged at
`2.1986806906376666e-15` U and `2.2380914396075147e-15` V: Decision 8 does
not own that existing kt1 stage-2 Kaa debt.  The dedicated three-ulp Kaa plant
exits `1`.

The complete LOCK trajectory Rule-8 movement census follows.  The seven rows
with a nonzero “worse >2” count are the Rule-12 debt register.

| row | move ULP | worse ULP | cells worse >2 ULP / n | improved / worsened |
|---|---:|---:|---:|---:|
| `LOCK_EXCHANGE-zco.kt2.before.u` | 0.001953125 | 2.32830644e-9 | 0 / 2540 | 59 / 34 |
| `LOCK_EXCHANGE-zco.kt3.before.T` | 43668 | 0 | 0 / 2560 | 25 / 0 |
| `LOCK_EXCHANGE-zco.kt3.before.ssh` | 6.10351562e-5 | 0 | 0 / 128 | 6 / 0 |
| `LOCK_EXCHANGE-zco.kt3.before.u` | 66.2396669 | 4.16043625e-50 | 0 / 2540 | 203 / 17 |
| `LOCK_EXCHANGE-zco.kt4.before.T` | 130878 | 0 | 0 / 2560 | 23 / 0 |
| `LOCK_EXCHANGE-zco.kt4.before.ssh` | 0.000122070312 | 0 | 0 / 128 | 10 / 0 |
| `LOCK_EXCHANGE-zco.kt4.before.u` | 276.81308 | 77.5860901 | 23 / 2540 | 257 / 23 |
| `LOCK_EXCHANGE-zco.kt5.before.S` | 1 | 1 | 0 / 2560 | 0 / 28 |
| `LOCK_EXCHANGE-zco.kt5.before.T` | 261409 | 0 | 0 / 2560 | 24 / 0 |
| `LOCK_EXCHANGE-zco.kt5.before.ssh` | 1.3203125 | 0 | 0 / 128 | 13 / 0 |
| `LOCK_EXCHANGE-zco.kt5.before.u` | 706.69796 | 616.49276 | 34 / 2540 | 245 / 55 |
| `LOCK_EXCHANGE-zco.kt6.before.S` | 1 | 1 | 0 / 2560 | 32 / 20 |
| `LOCK_EXCHANGE-zco.kt6.before.T` | 434936 | 1 | 0 / 2560 | 40 / 1 |
| `LOCK_EXCHANGE-zco.kt6.before.ssh` | 7.54003906 | 0 | 0 / 128 | 15 / 0 |
| `LOCK_EXCHANGE-zco.kt6.before.u` | 1439.22144 | 1439.22144 | 36 / 2540 | 294 / 86 |
| `LOCK_EXCHANGE-zco.kt7.before.S` | 2 | 0 | 0 / 2560 | 29 / 0 |
| `LOCK_EXCHANGE-zco.kt7.before.T` | 651176 | 1 | 0 / 2560 | 36 / 1 |
| `LOCK_EXCHANGE-zco.kt7.before.ssh` | 21.9262695 | 1.27211933e-7 | 0 / 128 | 18 / 1 |
| `LOCK_EXCHANGE-zco.kt7.before.u` | 2550.99997 | 2550.99997 | 37 / 2540 | 337 / 77 |
| `LOCK_EXCHANGE-zco.kt8.before.S` | 2 | 1 | 0 / 2560 | 18 / 10 |
| `LOCK_EXCHANGE-zco.kt8.before.T` | 909420 | 1 | 0 / 2560 | 49 / 2 |
| `LOCK_EXCHANGE-zco.kt8.before.ssh` | 49.2519531 | 7.82905772e-7 | 0 / 128 | 18 / 3 |
| `LOCK_EXCHANGE-zco.kt8.before.u` | 4083.79208 | 4083.79208 | 37 / 2540 | 336 / 124 |
| `LOCK_EXCHANGE-zco.kt9.before.S` | 1 | 1 | 0 / 2560 | 3 / 14 |
| `LOCK_EXCHANGE-zco.kt9.before.T` | 1209134 | 1 | 0 / 2560 | 50 / 1 |
| `LOCK_EXCHANGE-zco.kt9.before.ssh` | 100.824219 | 6.98791247e-9 | 0 / 128 | 19 / 4 |
| `LOCK_EXCHANGE-zco.kt9.before.u` | 6130.55237 | 6130.55237 | 36 / 2540 | 336 / 164 |
| `LOCK_EXCHANGE-zco.kt10.before.S` | 2 | 2 | 0 / 2560 | 26 / 26 |
| `LOCK_EXCHANGE-zco.kt10.before.T` | 1549621 | 0.25 | 0 / 2560 | 44 / 1 |
| `LOCK_EXCHANGE-zco.kt10.before.ssh` | 182.203125 | 9.0120302e-6 | 0 / 128 | 18 / 8 |
| `LOCK_EXCHANGE-zco.kt10.before.u` | 8760.48041 | 8760.48041 | 40 / 2540 | 340 / 180 |

The complete OVERFLOW trajectory Rule-8 movement census follows.  The 23 rows
with a nonzero “worse >2” count are the Rule-12 debt register.

| row | move ULP | worse ULP | cells worse >2 ULP / n | improved / worsened |
|---|---:|---:|---:|---:|
| `OVERFLOW-zps.kt2.before.u` | 0.0625 | 0.0625 | 0 / 16900 | 102 / 98 |
| `OVERFLOW-zps.kt3.before.S` | 2 | 1 | 0 / 17000 | 9 / 3 |
| `OVERFLOW-zps.kt3.before.T` | 2.07127374e9 | 1 | 0 / 17000 | 77 / 2 |
| `OVERFLOW-zps.kt3.before.ssh` | 0.00390625 | 0.000122070312 | 0 / 200 | 7 / 3 |
| `OVERFLOW-zps.kt3.before.u` | 305172106 | 10068188.1 | 3 / 16900 | 369 / 176 |
| `OVERFLOW-zps.kt4.before.S` | 3 | 1 | 0 / 17000 | 84 / 13 |
| `OVERFLOW-zps.kt4.before.T` | 3.56372962e9 | 2 | 0 / 17000 | 127 / 9 |
| `OVERFLOW-zps.kt4.before.ssh` | 118449990 | 5.25265932e-7 | 0 / 200 | 17 / 8 |
| `OVERFLOW-zps.kt4.before.u` | 724995881 | 47893474.2 | 16 / 16900 | 569 / 210 |
| `OVERFLOW-zps.kt5.before.S` | 4 | 2 | 0 / 17000 | 155 / 42 |
| `OVERFLOW-zps.kt5.before.T` | 3.16022534e9 | 2 | 0 / 17000 | 176 / 58 |
| `OVERFLOW-zps.kt5.before.ssh` | 550941929 | 31002650.2 | 1 / 200 | 26 / 10 |
| `OVERFLOW-zps.kt5.before.u` | 701726732 | 93154917.6 | 53 / 16900 | 781 / 316 |
| `OVERFLOW-zps.kt6.before.S` | 5 | 3 | 5 / 17000 | 100 / 164 |
| `OVERFLOW-zps.kt6.before.T` | 2.16441218e9 | 4 | 1 / 17000 | 165 / 148 |
| `OVERFLOW-zps.kt6.before.ssh` | 1.01200833e9 | 116223612 | 1 / 200 | 28 / 14 |
| `OVERFLOW-zps.kt6.before.u` | 364492566 | 265447033 | 105 / 16900 | 911 / 480 |
| `OVERFLOW-zps.kt7.before.S` | 8 | 3 | 5 / 17000 | 186 / 85 |
| `OVERFLOW-zps.kt7.before.T` | 2.13179039e9 | 5 | 2 / 17000 | 238 / 80 |
| `OVERFLOW-zps.kt7.before.ssh` | 937935687 | 185074563 | 2 / 200 | 29 / 18 |
| `OVERFLOW-zps.kt7.before.u` | 427374937 | 427374937 | 77 / 16900 | 1109 / 625 |
| `OVERFLOW-zps.kt8.before.S` | 7 | 2 | 0 / 17000 | 216 / 136 |
| `OVERFLOW-zps.kt8.before.T` | 2.81168887e9 | 4 | 3 / 17000 | 272 / 82 |
| `OVERFLOW-zps.kt8.before.ssh` | 519056479 | 136314453 | 1 / 200 | 36 / 15 |
| `OVERFLOW-zps.kt8.before.u` | 940292705 | 747171909 | 83 / 16900 | 1351 / 760 |
| `OVERFLOW-zps.kt9.before.S` | 8 | 3 | 8 / 17000 | 226 / 137 |
| `OVERFLOW-zps.kt9.before.T` | 3.04286165e9 | 16859 | 7 / 17000 | 305 / 105 |
| `OVERFLOW-zps.kt9.before.ssh` | 463559881 | 994147.5 | 1 / 200 | 41 / 14 |
| `OVERFLOW-zps.kt9.before.u` | 1.22600132e9 | 982110113 | 115 / 16900 | 1612 / 891 |
| `OVERFLOW-zps.kt10.before.S` | 7 | 3 | 9 / 17000 | 276 / 155 |
| `OVERFLOW-zps.kt10.before.T` | 2.62242109e9 | 4 | 5 / 17000 | 354 / 133 |
| `OVERFLOW-zps.kt10.before.ssh` | 866496525 | 360690850 | 2 / 200 | 45 / 14 |
| `OVERFLOW-zps.kt10.before.u` | 1.01872603e9 | 1.01872603e9 | 120 / 16900 | 1984 / 916 |

### Round-24 HOLD register and scope

Decision 8 itself is implemented and the previous “state design pending” HOLD
is closed.  It does not make the branch merge-ready.  The ordered open register
is now:

| boundary | result | honest owner/disposition |
|---|---|---|
| GYRE equal-input kt1 Kaa `uu_b/vv_b` | `0 / 580`, `0 / 570` | CONFIRMED exact state write/read contract |
| LOCK/OVERFLOW equal-input Kaa | U `1 / 127` and `7 / 199`; absent V `0 / 390`, `0 / 606` | CONFIRMED upstream external-mode debt exposed at the new state boundary |
| GYRE kt1 stage-2 Kaa | U `2.1986806906376666e-15`, V `2.2380914396075147e-15` | existing owner remains open; Decision 8 is near-null there |
| GYRE kt2--10 trajectory | first-over-bar kt2 T/S/u/v; 40 Rule-12 rows | downstream external-mode/stage composition debt; walk before merge |
| LOCK kt2--10 trajectory | first-over-bar improves kt3 T/u to kt4 u; seven Rule-12 u rows | compensating error exposed; faithful state stays |
| OVERFLOW kt2--10 trajectory | first-over-bar remains kt2 T/u; 23 Rule-12 rows | compensating error exposed; faithful state stays |
| ORCA2 production W | still DEBT, cellwise Decision-8 comparison PASS | existing ORCA2 stage-transport/W owner remains with lane 4 |
| ~~C1D kt2 PRE_SSM~~ | ~~six rows unchanged~~ | **RETRACTED invalid control**: the branch card never enabled `nemo_prognostic_barotropic_velocity`; the probe-only number perturbed nothing and is not evidence about Decision 8 |
| OVERFLOW dry-face NaN producer | no lane-3b handoff received | registered only; no mask workaround |

Before the Round-20 HOLD on a reconciled/integration merge can lift, the GYRE
stage-2 Kaa and newly exposed external-mode/stage-composition Rule-12 rows must
be walked to their first primitive operands, the GYRE production trajectory
must clear the bar, and this round must receive independent review.  Stage-3
transport and ZDF/TKE work was not entered because those prerequisites do not
hold.  No merge or push was performed.

### Round-24 ASKED / UNASKED register

| choice or action | origin | final disposition |
|---|---|---|
| one prognostic U/V depth-mean pair on NEMO identities | ASKED Decision 8 | implemented shared; no per-card physics fork |
| exact NEMO time levels, seed, Kaa write, S-21 read, final swap | ASKED items 1/3 | implemented and source-cited |
| rest initialization and legacy restart fallback | ASKED item 1 | rest zero; fallback only for missing fields, logged and tested |
| non-NEMO no-array-leaf guarantee | ASKED item 2 | `None/None`; negative pytree test passes |
| every constructor/restart/bridge site | ASKED item 2 | repository census updated; half-pair rejected |
| GYRE kt1--10 and stage-2 Kaa | ASKED item 4 | measured; full tables above, prediction partly refuted |
| LOCK and OVERFLOW stage/trajectory cross-card gates | ASKED item 4 / Rule 12 | stage PASS; trajectory FAIL without earlier first-over-bar; every moved row enumerated |
| ORCA2 production-W and entry-stage gates | ASKED item 4 | fresh probe measured; cellwise Rule 12 PASS, production rows remain DEBT |
| ~~C1D kt2 PRE_SSM~~ | ASKED item 4 | **RETRACTED invalid control**; no carried-state flag on this branch, so no lane-3 number is claimed |
| kt5 restart then kt6--10 | ASKED item 4 | every returned pytree leaf bit-identical to unbroken run |
| new NEMO acquisition | conditional | not needed; existing BTFRM records contain Kaa pair |
| rederive live state from 3-D velocity | forbidden | removed from identity path; legacy missing-restart fallback only |
| stage-3/ZDF/TKE continuation | downstream | not entered across open stage-2 and Rule-12 boundaries |
| shipped NEMO edit, NEMO run, multi-MB git artifact, push, merge | forbidden | none |

All Round-24 labels and measurements are Codex-internal.  Independent review
of this round remains **OUTSTANDING**.

Focused CPU verification is `168 passed, 1 warning in 78.92s`; the warning is
the pre-existing negative precision-policy warning in the bridge test.  The
selection covers both phase-3 gates, the new state gate, NEMO IO/bridges,
restart persistence, and the no-scheme-duplication tripwire.  The machine
manifest is
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round24.json`,
SHA-256
`25e7cd11a82b6c87c6120be1aba36b0ecae76a876466a9037570d282669882b4`.
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`.

### Restart and negative-state controls

The production restart test advances LOCK to kt5, saves every prognostic
leaf, reloads, and advances kt6--10.  Every leaf, including `uu_b/vv_b`, is
bit-identical to the unbroken run at every continuation step.  A direct
round-trip test also pins nontrivial pair bits.  The missing-pair bridge test
proves the legacy reconstruction is logged; the half-pair test raises.  The
non-NEMO negative test sees exactly two fewer JAX array leaves than the
otherwise identical opted-in state.

### Rule 8 / Rule 12: GYRE

The changed boundary is the same-step external Kaa target handed to S-21 and
then the carried Kbb seed on the next step; the owner of the movement is
Decision 8.  The operator itself is exact at the equal-input kt1 Kaa write.
The cellwise comparison against the Round-23 report is nevertheless **FAIL**:

```text
ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=19206928396.31543 first_over_bar={'fields': ['T', 'S', 'u', 'v'], 'kt': 2}->{'kt': 2, 'fields': ['T', 'S', 'u', 'v']} plant=None
```

No row changes status and first-over-bar does not move.  This is the complete
44-row Rule-8 movement census; the 40 kt3--10 rows have cells exceeding the
two-ulp Rule-12 allowance.  “Move” remains movement against the previous
legoesm output; “worse” is the cellwise increase in oracle residual.  Both are
in the row-scale oracle ulp.

| row | move ULP | worse ULP | cells worse >2 ULP / n | improved / worsened |
|---|---:|---:|---:|---:|
| `GYRE-zco.kt2.before.S` | 1 | 1 | 0 / 18000 | 3 / 1 |
| `GYRE-zco.kt2.before.T` | 1 | 0 | 0 / 18000 | 3 / 0 |
| `GYRE-zco.kt2.before.u` | 0.03125 | 0.03125 | 0 / 17400 | 4037 / 5569 |
| `GYRE-zco.kt2.before.v` | 0.03125 | 0.03125 | 0 / 17100 | 4467 / 5662 |
| `GYRE-zco.kt3.before.S` | 74226637 | 31314675 | 4815 / 18000 | 11808 / 5150 |
| `GYRE-zco.kt3.before.T` | 120896622 | 80711651 | 5640 / 18000 | 12163 / 5665 |
| `GYRE-zco.kt3.before.ssh` | 30.1328125 | 30.1328125 | 2 / 600 | 281 / 318 |
| `GYRE-zco.kt3.before.u` | 1.4869927e10 | 1.17903339e10 | 8604 / 17400 | 8796 / 8604 |
| `GYRE-zco.kt3.before.v` | 1.08455746e10 | 9.98338518e9 | 8320 / 17100 | 8780 / 8320 |
| `GYRE-zco.kt4.before.S` | 86172021 | 86172021 | 7904 / 18000 | 8984 / 8467 |
| `GYRE-zco.kt4.before.T` | 198784977 | 198784977 | 8617 / 18000 | 9214 / 8699 |
| `GYRE-zco.kt4.before.ssh` | 187527570 | 153965014 | 279 / 600 | 321 / 279 |
| `GYRE-zco.kt4.before.u` | 1.35151818e10 | 1.35151818e10 | 8555 / 17400 | 8845 / 8555 |
| `GYRE-zco.kt4.before.v` | 1.27469482e10 | 1.27469482e10 | 8351 / 17100 | 8749 / 8351 |
| `GYRE-zco.kt5.before.S` | 78270650 | 49824166 | 7923 / 18000 | 9149 / 8461 |
| `GYRE-zco.kt5.before.T` | 186633216 | 186633216 | 8583 / 18000 | 9242 / 8700 |
| `GYRE-zco.kt5.before.ssh` | 321355221 | 321355221 | 269 / 600 | 331 / 269 |
| `GYRE-zco.kt5.before.u` | 1.34490835e10 | 1.24238231e10 | 8199 / 17400 | 9201 / 8199 |
| `GYRE-zco.kt5.before.v` | 1.04738843e10 | 1.04738843e10 | 8019 / 17100 | 9081 / 8019 |
| `GYRE-zco.kt6.before.S` | 103631970 | 103631970 | 7775 / 18000 | 9465 / 8205 |
| `GYRE-zco.kt6.before.T` | 209927188 | 209927188 | 8263 / 18000 | 9572 / 8383 |
| `GYRE-zco.kt6.before.ssh` | 430849891 | 430849891 | 287 / 600 | 313 / 287 |
| `GYRE-zco.kt6.before.u` | 1.92069284e10 | 1.92069284e10 | 8646 / 17400 | 8754 / 8646 |
| `GYRE-zco.kt6.before.v` | 1.43816347e10 | 1.43816347e10 | 8745 / 17100 | 8355 / 8745 |
| `GYRE-zco.kt7.before.S` | 84038532 | 78409574 | 7306 / 18000 | 10069 / 7705 |
| `GYRE-zco.kt7.before.T` | 482954233 | 482954233 | 7692 / 18000 | 10208 / 7779 |
| `GYRE-zco.kt7.before.ssh` | 402174312 | 370411895 | 320 / 600 | 280 / 320 |
| `GYRE-zco.kt7.before.u` | 1.65534627e10 | 1.49570594e10 | 8738 / 17400 | 8662 / 8738 |
| `GYRE-zco.kt7.before.v` | 1.64281573e10 | 1.64281573e10 | 8433 / 17100 | 8667 / 8433 |
| `GYRE-zco.kt8.before.S` | 88623795 | 87727965 | 8086 / 18000 | 9266 / 8480 |
| `GYRE-zco.kt8.before.T` | 346969126 | 346969126 | 8450 / 18000 | 9450 / 8543 |
| `GYRE-zco.kt8.before.ssh` | 520126523 | 406146731 | 278 / 600 | 322 / 278 |
| `GYRE-zco.kt8.before.u` | 1.63082063e10 | 1.63082063e10 | 8950 / 17400 | 8450 / 8950 |
| `GYRE-zco.kt8.before.v` | 1.90039058e10 | 1.54288965e10 | 8522 / 17100 | 8578 / 8522 |
| `GYRE-zco.kt9.before.S` | 76347236 | 76347236 | 8898 / 18000 | 8466 / 9308 |
| `GYRE-zco.kt9.before.T` | 311833598 | 311833598 | 9302 / 18000 | 8576 / 9412 |
| `GYRE-zco.kt9.before.ssh` | 746319096 | 746319096 | 300 / 600 | 300 / 300 |
| `GYRE-zco.kt9.before.u` | 1.48462441e10 | 1.48335787e10 | 9073 / 17400 | 8327 / 9073 |
| `GYRE-zco.kt9.before.v` | 1.64024528e10 | 1.33172267e10 | 8249 / 17100 | 8851 / 8249 |
| `GYRE-zco.kt10.before.S` | 69267551 | 69267551 | 8478 / 18000 | 8887 / 8871 |
| `GYRE-zco.kt10.before.T` | 319248770 | 319248770 | 8884 / 18000 | 9000 / 8991 |
| `GYRE-zco.kt10.before.ssh` | 656161889 | 590236844 | 305 / 600 | 295 / 305 |
| `GYRE-zco.kt10.before.u` | 1.37100843e10 | 1.17432804e10 | 8566 / 17400 | 8834 / 8566 |
| `GYRE-zco.kt10.before.v` | 1.06070325e10 | 1.00824836e10 | 8612 / 17100 | 8488 / 8612 |

The comparison correctly reports the 20 new Kaa rows as candidate-only and
does not pretend that a missing Round-23 row is comparable.  Their absolute
values are instead listed in the dedicated pair table above.

### Cross-card outcomes

All earlier-card rows use the same production-JIT CPU fp64/scalar-libm regime
and the cellwise oracle-relative, row-scale-ulp criterion.  The verdict lines
are:

```text
ORACLE_RELATIVE_COMPARE PASS: rows=9 max_worsening_ulps=0.0009765625 first_over_bar='<absent>'->'<absent>' plant=None
ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=8760.480407714844 first_over_bar={'fields': ['T', 'u'], 'kt': 3}->{'kt': 4, 'fields': ['u']} plant=None
ORACLE_RELATIVE_COMPARE PASS: rows=9 max_worsening_ulps=0.0625 first_over_bar='<absent>'->'<absent>' plant=None
ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=1018726033.65625 first_over_bar={'fields': ['T', 'u'], 'kt': 2}->{'kt': 2, 'fields': ['T', 'u']} plant=None
```

These are LOCK stage, LOCK trajectory, OVERFLOW stage, and OVERFLOW
trajectory, respectively.  Neither failing trajectory moves first-over-bar
earlier and neither creates an AT-BAR-to-DEBT status change.  LOCK instead
moves ten rows DEBT-to-AT-BAR and delays first-over-bar; OVERFLOW moves kt7--10
S DEBT-to-AT-BAR.  Rule 12 still fails on seven LOCK-u rows and 23 OVERFLOW
rows because individual oracle residuals worsen by more than two row-scale
ulp.  The faithful carried state stays and the external-mode/stage-composition
boundary is registered as the exposed second error.

The complete stage-row movement census is:

| row | move ULP | worse ULP | cells worse >2 ULP / n | improved / worsened |
|---|---:|---:|---:|---:|
| `LOCK_EXCHANGE-zco.kt1.stage1.faithful.baroclinic_u` | 0.001953125 | 0.0009765625 | 0 / 2540 | 16 / 1 |
| `LOCK_EXCHANGE-zco.kt1.stage1.faithful.instantaneous_u` | 0.001953125 | 0 | 0 / 2540 | 19 / 0 |
| `LOCK_EXCHANGE-zco.kt1.stage2.faithful.baroclinic_u` | 0.0009765625 | 0.0009765625 | 0 / 2540 | 9 / 25 |
| `LOCK_EXCHANGE-zco.kt1.stage2.faithful.instantaneous_u` | 0.001953125 | 8.14907253e-10 | 0 / 2540 | 27 / 16 |
| `LOCK_EXCHANGE-zco.kt1.stage3.faithful.baroclinic_u` | 0.0009765625 | 0.0009765625 | 0 / 2540 | 24 / 43 |
| `LOCK_EXCHANGE-zco.kt1.stage3.faithful.instantaneous_u` | 0.001953125 | 2.32830644e-9 | 0 / 2540 | 59 / 34 |
| `LOCK_EXCHANGE-zco.kt2.faithful.baroclinic_u` | 0.0009765625 | 0.0009765625 | 0 / 2540 | 24 / 43 |
| `LOCK_EXCHANGE-zco.kt2.faithful.instantaneous_u` | 0.001953125 | 2.32830644e-9 | 0 / 2540 | 59 / 34 |
| `OVERFLOW-zps.kt1.stage1.faithful.baroclinic_u` | 0.03125 | 0.03125 | 0 / 16900 | 52 / 1 |
| `OVERFLOW-zps.kt1.stage1.faithful.instantaneous_u` | 0.03125 | 0.001953125 | 0 / 16900 | 48 / 75 |
| `OVERFLOW-zps.kt1.stage2.faithful.baroclinic_u` | 0.0625 | 0.0625 | 0 / 16900 | 45 / 49 |
| `OVERFLOW-zps.kt1.stage2.faithful.instantaneous_u` | 0.0625 | 0.0625 | 0 / 16900 | 80 / 90 |
| `OVERFLOW-zps.kt1.stage3.faithful.baroclinic_u` | 0.0625 | 0.03125 | 0 / 16900 | 84 / 63 |
| `OVERFLOW-zps.kt1.stage3.faithful.instantaneous_u` | 0.0625 | 0.0625 | 0 / 16900 | 102 / 98 |
| `OVERFLOW-zps.kt2.faithful.baroclinic_u` | 0.0625 | 0.03125 | 0 / 16900 | 84 / 63 |
| `OVERFLOW-zps.kt2.faithful.instantaneous_u` | 0.0625 | 0.0625 | 0 / 16900 | 102 / 98 |

The dedicated equal-input Kaa gate gives GYRE `0 / 580` and `0 / 570`.
LOCK has one U-face bit different (`1 / 127`, max
`2.168404344971009e-19`); its V row is `0 / 390` but UNINFORMATIVE because
the three-row tank has no active V face.  OVERFLOW has `7 / 199` U faces
different (max `6.938893903907228e-18`); its structurally absent V row is
`0 / 606`.  The state write is a direct copy, so these bits are labeled
**CONFIRMED_UPSTREAM_EXTERNAL_MODE_DEBT**, not state-storage errors.

The fresh ORCA2 probe is based on `b6a6189c9357`, with the Decision-8 probe at
`845201229a4` and WRITE-only residual capture at `49026566a53`; its clean-tip
baseline capture is `c0faa2540`.  The production gate retains exact `e2u/e1v`.
Against the previous ORCA2 result, zFu unequal cells improve
`102431 -> 85175 / 228641`, zFv improve `108307 -> 93486 / 228641`, and W
improves `220589 -> 218667 / 228641`; all remain DEBT.  The W maximum absolute
residual moves `4.6872685718684845e-19 -> 5.007089683905987e-19`.
The independent cellwise Rule-12 replay nevertheless passes:

| ORCA2 row | move ULP | worse ULP | cells worse >2 ULP / n | improved / worsened |
|---|---:|---:|---:|---:|
| production W | 0.00147694349 | 0.00147694349 | 0 / 228641 | 75727 / 65934 |
| zFu | 1 | 0.5 | 0 / 228641 | 30620 / 19471 |
| zFv | 1 | 1 | 0 / 228641 | 28933 / 19593 |

The ORCA2 entry stage gate is still exact, and its one-cell production-W plant
exits `1`.  The former C1D paragraph is struck: its conflict-marked probe-only
opt-in never existed on this branch, so the six-row no-move output was a
control that perturbed nothing.  No C1D numerical result is claimed here;
lane 3b must remeasure after opting its NEMO-identity card into the shared
state field.

## Round 25 — retraction, negative guards, stage-2 composition, independent re-measurement

Round 25 starts from `e80dd94b8276` and runs in the production-JIT CPU fp64/x64,
oracle-V2 scalar-libm regime.  The preregistration
(`manifests/nemo_testcase_l2_gyre_round25_preregister.json`) is committed before
the measurements.  `/tmp/codex-c1d-r24` and `/tmp/codex-orca2-r24` remain
flagged; neither was read.  No NEMO executable was run from the sandbox.

Every GYRE number below is scored against the **V2 scalar-math oracle root**
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round19_oracle_v2_external`,
passed explicitly to `--oracle-root`, `--stage2-oracle-root` and
`--stage3-oracle-root`.  This is recorded because it is a live trap: the gate's
compiled-in `ROOT`/`STAGE2_ROOT`/`STAGE3_ROOT` defaults still point at
`gyre_kt1_10` and `gyre_kt1_10_stage2_terms`, which hold the superseded
**V1 vector-math** oracle (`oracle_stage_kt00000001_s1.bin` =
`35e6892b799aeaf8d06d4affcd71b5ba0c71dc41bc0e8970c033459c46cd1402`, against the
registered V2 digest
`ce25b004e7e8289b6e803263f895576981ce22516ccddfbd85d7be5ce5bcaedc`).  Running
the gate on its defaults aborts fail-closed on the digest check rather than
scoring silently, which is the correct behaviour, but no round-25 figure may be
reproduced without naming the V2 root.

### Rule-11 records

Three round-25 claims died.  Each is written here next to what killed it.

1. **Round-23 owner claim: REFUTED AND ILL-POSED.**  The statement that carried
   `uu_b/vv_b(Kbb)` was the first unavailable operand for the stage-2 and kt1
   rows cannot be posed.  `stprk3.F90:186` calls
   `stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )`, which writes `ssh`, `(uu_b,vv_b)` at
   **Naa** and `(un_adv,vn_adv)` *before* any stage; `stprk3.F90:195` runs stage
   1 with `Kmm=Nbb`; `:197` swaps `Nnn <==> Naa`; `:200` and `:207` run stages 2
   and 3 with `Kmm=Nnn`.  Stage 2 therefore consumes same-step
   `uu_b/vv_b(Nnn)`, an operand that was never unavailable.  The carried Kbb
   pair reaches only a later step's external-mode seed
   (`stp2d.F90:281` -> `dynspg_ts.F90:487`, `un_e(:,:) = puu_b(:,:,Kmm)` under
   `ln_bt_fw`) and the stage-1 S-21 operand (`stprk3_stg.F90:262,267`,
   `zub = un_adv*r1_hu(Kmm) - uu_b(Kmm)`, read at `Kmm` and hence last-step at
   stage 1 only).  Landed as `0dbb4bc574b3`.
2. **"Linear surface update owner": REFUTED.**  A one-variable arm routing the
   non-selected `lk_linssh` external-mode formula moved neither the OVERFLOW
   `u_exit` frame nor the `7 / 199` Kaa row.  Landed then reverted
   (`15aa818f4e6c` -> `8d104f61c450`).
3. **"NEMO call-order / source-round accumulation" arm: REFUTED.**  It changed
   only unequal-cell counts, did not reduce the `9.926167350636332e-24`
   cumulative-vorticity maximum, and made the corrected V maximum
   `1.3552527156068805e-19` instead of `1.0842021724855044e-19`.  Reverted; no
   owner is assigned to the AT-BAR tail.

The C1D kt2 row remains struck: that probe never enabled the identity field, so
it perturbed nothing.  Lane 3b must remeasure with its card opted in.

### Acceptance guards, and the break that proves each can fail

The identity is now uniform: generic rest, Eady, GYRE, lane-1 tanks and every
other recipe selecting the NEMO WS-RK3 identity initialize the optional
`uu_b/vv_b` pair (`oce.F90:39,99`; the zero value follows
`DOM/istate.F90:149-155`, the depth-weighted mean of `uu(Kbb)`); ordinary
recipes retain `None` and gain no array pytree leaves.  Each guard below was
shown non-vacuous by breaking the code it guards, running the test, and
restoring the file (`git checkout --` plus `git status --porcelain` empty).

| guard | test | planted break | observed failure |
|---|---|---|---|
| stage-1 reads last-step Kbb; stages 2-3 read this-step Nnn | `test_stage_time_levels_keep_kbb_out_of_same_step_nnn` | stage-1 branch returns `nnn_velocity` | `ACTUAL array([[101.]]) DESIRED array([[1.]])`, 1 failed |
| external mode is the only live dynamics writer | `test_identity_path_has_one_external_mode_writer_for_carried_pair` | added a second `dict(uu_b=..., vv_b=...)` call in `ocean_pe_latlon_cgrid.py` | inventory mismatch, 1 failed |
| non-NEMO recipes keep `None` leaves | `test_non_nemo_state_has_no_depth_mean_array_leaves`, `test_generic_nemo_recipes_all_carry_pair_without_changing_base_eady` | `rest_state_latlon_cgrid_ocean` builds the pair unconditionally | 2 failed |
| restart re-derives only on an ABSENT pair, never on a parse failure | `test_restart_malformed_depth_mean_does_not_use_missing_field_fallback` | `read_nemo_restart` swallows `TypeError/ValueError` and returns `None` | `DID NOT RAISE`, 1 failed |
| every stage call site routes through that selector | `test_every_stage_barotropic_operand_routes_through_the_time_level_selector` | stage-1 call site passes `(target_u, target_v)` directly | selector bypass detected, 1 failed |

The fifth row is added by this round.  The four landed guards test the
selector helper in isolation, and that helper test keeps passing when a stage
call site skips the helper entirely — which is the defect the review asked
about.  The new guard walks the AST of the module that runs and requires all
three stage operands to be selector calls carrying the literal stage number.

### Stage-2 ordered walk

NEMO stage 2 accumulates the momentum RHS in written source order:
`dyn_hpg` at `stprk3_stg.F90:324`, `dyn_vor` at `:327`, `dyn_adv` at `:331`
(vector form) or `:333` (flux form).  GYRE runs the vector-invariant arm, so the
stage update is `uu(Kaa) = ( uu(Kbb) + rDt * uu(Krhs) ) * umask` at
`stprk3_stg.F90:367-368`; the `key_qco` thickness-weighted alternative sits at
`:373-378` and is not selected here.  The all-stage barotropic replacement is
`stprk3_stg.F90:437-446`.

Injecting NEMO's own Krhs made the raw update exact (`0 / 17400` U,
`0 / 17100` V, probe mode `oracle_rhs_raw`), which refutes the update and the
barotropic replacement as the first owner and puts the boundary inside the
accumulation.

The landed shared fix has two parts, both source-association, no card arm:

- **QCO live depth.**  `domzgr_substitute.h90:145` defines
  `gdept_z0(i,j,k,t) = (gdept(i,j,k,t) - ssh(i,j,t))` with `gdept` expanded at
  `:139`, i.e. two statements' operations.  `_nemo_qco_gdept_z0` now
  materialises the multiply before the subtraction instead of leaving the chain
  for XLA to reassociate.
- **HPG interface.**  `dyn_hpg_sco` produces an acceleration and writes
  `zhpi + zuap` straight into the momentum RHS (`dynhpg.F90:359` for `k=1`,
  `:383` for `k>1`, the "RK3 case: dyn_hpg always called first" branch).  The
  shared model previously multiplied that by `-rho0` to fit a
  pressure-gradient interface and divided it back at the consumer;
  `_nemo_hpg_tendency_from_pressure_or_direct` now carries `nemo_sco`'s native
  acceleration through.  Other PGF schemes keep their pressure-gradient route.

All rows below are `nemo_testcase_l2_gyre_round11_composition.py` against the V2
root, re-measured at `4b6c421df5d0` on a clean tree.  Values are the wet-face
absolute maximum; parentheses are `unequal / n`.

| stage-2 boundary | U max / unequal | V max / unequal | verdict |
|---|---:|---:|---|
| round-24 completed Kaa (before the fix) | `2.1986806906376666e-15`, `17400 / 17400` | `2.2380914396075147e-15`, `17077 / 17100` | DEBT |
| HPG given NEMO inputs (`oracle_input_hpg`) | `0.0`, `0 / 17400` | `0.0`, `0 / 17100` | bit-exact |
| cumulative through vorticity | `9.926167350636332e-24`, `8699 / 17400` | `9.926167350636332e-24`, `8360 / 17100` | AT-BAR |
| cumulative through advection | `0.0`, `0 / 17400` | `0.0`, `0 / 17100` | bit-exact |
| oracle Krhs into the raw update | `0.0`, `0 / 17400` | `0.0`, `0 / 17100` | bit-exact |
| landed full Krhs (`rhs`) | `9.926167350636332e-24`, `10765 / 17400` | `9.926167350636332e-24`, `10626 / 17100` | AT-BAR |
| landed raw Kaa (`raw`) | `8.131516293641283e-20`, `10054 / 17400` | `8.131516293641283e-20`, `9888 / 17100` | AT-BAR |
| landed corrected Kaa (`corrected`) | `1.0842021724855044e-19`, `7306 / 17400` | `1.0842021724855044e-19`, `7405 / 17100` | AT-BAR |

The preregistered stage-2 prediction is **CONFIRMED**: the HPG insertion into
Krhs was the first non-bit statement, and every stage-2 operand ahead of it is
bit-exact.  The completed stage-2 Kaa moves from round-24's DEBT
`2.1986806906376666e-15` U / `2.2380914396075147e-15` V to AT-BAR
`1.0842021724855044e-19` on both components.  The first residual after exact
HPG is the vorticity accumulation, nine orders below the bar; it is left
unlabelled.  The planted Krhs control changes one U cell by `1.0` and exits `1`.

Each half of the shared fix was ablated on its own, one variable at a time, and
the file restored afterwards (`git status --porcelain` empty each time):

| arm | HPG given NEMO inputs, U / V | completed stage-2 Kaa, U / V | verdict |
|---|---|---|---|
| landed | `0.0` (`0 / 17400`) / `0.0` (`0 / 17100`) | `1.0842021724855044e-19` (`7306 / 17400`) / `1.0842021724855044e-19` (`7405 / 17100`) | AT-BAR |
| A: fused QCO `gdept_z0` | `2.8730908583045487e-19` (`7205 / 17400`) / `2.8730908663824844e-19` (`7028 / 17100`) | `2.1986806906376666e-15` (`17400 / 17400`) / `2.2380914396075147e-15` (`17077 / 17100`) | DEBT |
| B: pressure-gradient round trip | `1.6155871338926322e-27` (`6248 / 17400`) / `3.2311742677852644e-27` (`6242 / 17100`) | `1.0842021724855044e-19` (`7307 / 17400`) / `1.0842021724855044e-19` (`7473 / 17100`) | AT-BAR |

Arm A alone restores round 24's DEBT row to the digit on both components — its
report file hashes to
`72b949660ed979635158987f601de6fe152eff23b73ae2dcdb71d8900653f0cf`, which is
byte-for-byte the round-24 manifest's `gyre_stage2` artifact — so the QCO
`gdept_z0` source association is the **CONFIRMED owner** of the stage-2 Kaa
DEBT.  Arm B moves the HPG frame off exact zero but leaves the completed Kaa
maximum unchanged, so the acceleration-to-pressure round trip is the
**CONFIRMED owner of the HPG frame's residual only** — it is required for the
bit-exact HPG operand the preregistered prediction is checked against, and it is
not what cleared the Kaa row.  Both halves are source-faithful and both stay;
the joint attribution stated in the ending session's draft is corrected here.

### Production GYRE trajectory and the Rule-12 compensating-error register

The production sweep is exact at kt1 on every field.  `first_over_bar` stays at
kt2 `T/S/u/v`.  Values are the normalized wet-cell maximum, parentheses
`unequal / n`; exact rows print `0 / n`.

| kt | T | S | u | v | ssh |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 / 18000 | 0 / 18000 | 0 / 17400 | 0 / 17100 | 0 / 600 |
| 2 | 1.36147368e-12 (11840 / 18000) | 2.21811013e-14 (11229 / 18000) | 9.48408994e-7 (17400 / 17400) | 8.98799261e-7 (17100 / 17100) | 4.33680869e-19 (600 / 600) |
| 3 | 3.72234414e-4 (18000 / 18000) | 3.68324674e-5 (17901 / 18000) | 9.31167891e-3 (17400 / 17400) | 4.71907759e-3 (17100 / 17100) | 7.07466963e-7 (600 / 600) |
| 4 | 1.05947381e-3 (18000 / 18000) | 8.22574046e-5 (17995 / 18000) | 1.41824040e-2 (17400 / 17400) | 1.59478423e-2 (17100 / 17100) | 5.06042337e-7 (600 / 600) |
| 5 | 3.05374248e-3 (18000 / 18000) | 8.61182643e-5 (18000 / 18000) | 2.07338904e-2 (17400 / 17400) | 4.37976656e-2 (17100 / 17100) | 6.70185102e-5 (600 / 600) |
| 6 | 3.81767581e-3 (18000 / 18000) | 1.28753624e-4 (18000 / 18000) | 3.11347297e-2 (17400 / 17400) | 6.02195921e-2 (17100 / 17100) | 1.63102390e-4 (600 / 600) |
| 7 | 4.54666655e-3 (18000 / 18000) | 1.22440449e-4 (18000 / 18000) | 4.00414047e-2 (17400 / 17400) | 6.63157732e-2 (17100 / 17100) | 2.08714182e-4 (600 / 600) |
| 8 | 5.09837609e-3 (18000 / 18000) | 1.36098443e-4 (18000 / 18000) | 4.71800591e-2 (17400 / 17400) | 2.45271619e-2 (17100 / 17100) | 2.51082477e-4 (600 / 600) |
| 9 | 5.44506214e-3 (18000 / 18000) | 1.47118217e-4 (18000 / 18000) | 5.25603976e-2 (17400 / 17400) | 1.45336123e-2 (17100 / 17100) | 2.12030022e-4 (600 / 600) |
| 10 | 5.63663449e-3 (18000 / 18000) | 1.54831214e-4 (18000 / 18000) | 5.62498696e-2 (17400 / 17400) | 1.11733658e-2 (17100 / 17100) | 2.12554356e-4 (600 / 600) |

Against round 24 the cellwise oracle-relative gate reports:

```text
ORACLE_RELATIVE_COMPARE FAIL: rows=70 max_worsening_ulps=185097.1640625 first_over_bar={'fields': ['T', 'S', 'u', 'v'], 'kt': 2}->{'kt': 2, 'fields': ['T', 'S', 'u', 'v']} plant=None
```

62 of the 70 certified rows moved and 56 exceed the two-row-scale-ulp
threshold.  The changed HPG operator is bit-exact given NEMO's own inputs, so
Rule 12 applies: the fix stays, `first_over_bar` did not come earlier, and the
56 rows enter the register as named debt.  Every moved row is disclosed below.
| row | move ULP | worse ULP | cells >2 ULP / n | improved / worsened |
|---|---:|---:|---:|---:|
| `GYRE-zco.kt10.after.uu_b` | 51.05078125 | 51.05078125 | 305 / 580 | 227 / 353 |
| `GYRE-zco.kt10.after.vv_b` | 51.37939453125 | 51.37939453125 | 274 / 570 | 272 / 298 |
| `GYRE-zco.kt10.before.S` | 32631.0 | 25813.0 | 4164 / 18000 | 7424 / 7410 |
| `GYRE-zco.kt10.before.T` | 42227.0 | 42227.0 | 7242 / 18000 | 8493 / 8739 |
| `GYRE-zco.kt10.before.ssh` | 15503.8515625 | 15503.8515625 | 292 / 600 | 308 / 292 |
| `GYRE-zco.kt10.before.u` | 67625.8046875 | 67625.8046875 | 8688 / 17400 | 8697 / 8703 |
| `GYRE-zco.kt10.before.v` | 185097.1640625 | 185097.1640625 | 8738 / 17100 | 8345 / 8755 |
| `GYRE-zco.kt2.after.uu_b` | 0.16461181640625 | 0.16461181640625 | 0 / 580 | 280 / 299 |
| `GYRE-zco.kt2.after.vv_b` | 0.111083984375 | 0.08937835693359375 | 0 / 570 | 330 / 238 |
| `GYRE-zco.kt2.before.S` | 1.0 | 1.0 | 0 / 18000 | 20 / 10 |
| `GYRE-zco.kt2.before.T` | 2.0 | 2.0 | 0 / 18000 | 756 / 451 |
| `GYRE-zco.kt2.before.u` | 24.48193359375 | 24.48193359375 | 1955 / 17400 | 8263 / 9124 |
| `GYRE-zco.kt2.before.v` | 26.783203125 | 26.783203125 | 1911 / 17100 | 8592 / 8496 |
| `GYRE-zco.kt3.after.uu_b` | 1.295166015625 | 1.295166015625 | 0 / 580 | 279 / 299 |
| `GYRE-zco.kt3.after.vv_b` | 1.1357421875 | 1.04052734375 | 0 / 570 | 298 / 272 |
| `GYRE-zco.kt3.before.S` | 7.0 | 6.0 | 20 / 18000 | 319 / 321 |
| `GYRE-zco.kt3.before.T` | 102.0 | 99.5 | 533 / 18000 | 2260 / 2197 |
| `GYRE-zco.kt3.before.ssh` | 163.916015625 | 59.4765625 | 41 / 600 | 320 / 280 |
| `GYRE-zco.kt3.before.u` | 946.771484375 | 564.234375 | 3687 / 17400 | 8737 / 8654 |
| `GYRE-zco.kt3.before.v` | 551.77880859375 | 532.5 | 3827 / 17100 | 8567 / 8517 |
| `GYRE-zco.kt4.after.uu_b` | 4.36058235168457 | 4.36058235168457 | 30 / 580 | 255 / 325 |
| `GYRE-zco.kt4.after.vv_b` | 2.5517578125 | 2.5517578125 | 5 / 570 | 314 / 256 |
| `GYRE-zco.kt4.before.S` | 6085.0 | 5820.0 | 199 / 18000 | 1228 / 1138 |
| `GYRE-zco.kt4.before.T` | 7622.0 | 7291.0 | 1796 / 18000 | 4505 / 4251 |
| `GYRE-zco.kt4.before.ssh` | 771.0390625 | 771.0390625 | 197 / 600 | 363 / 237 |
| `GYRE-zco.kt4.before.u` | 1109.068359375 | 1106.62451171875 | 6493 / 17400 | 8572 / 8828 |
| `GYRE-zco.kt4.before.v` | 1090.8865966796875 | 1090.8865966796875 | 6401 / 17100 | 8548 / 8547 |
| `GYRE-zco.kt5.after.uu_b` | 7.9461669921875 | 7.9461669921875 | 151 / 580 | 306 / 274 |
| `GYRE-zco.kt5.after.vv_b` | 7.80419921875 | 7.80419921875 | 151 / 570 | 271 / 299 |
| `GYRE-zco.kt5.before.S` | 8493.0 | 5947.0 | 806 / 18000 | 2742 / 2688 |
| `GYRE-zco.kt5.before.T` | 18347.0 | 14829.0 | 4124 / 18000 | 6588 / 6384 |
| `GYRE-zco.kt5.before.ssh` | 3065.5703125 | 3065.5703125 | 289 / 600 | 295 / 305 |
| `GYRE-zco.kt5.before.u` | 1993.7607421875 | 1993.7607421875 | 8351 / 17400 | 8547 / 8853 |
| `GYRE-zco.kt5.before.v` | 2090.6571044921875 | 2066.5078125 | 8002 / 17100 | 8600 / 8500 |
| `GYRE-zco.kt6.after.uu_b` | 14.0263671875 | 12.884765625 | 169 / 580 | 333 / 247 |
| `GYRE-zco.kt6.after.vv_b` | 15.8095703125 | 15.8095703125 | 246 / 570 | 270 / 300 |
| `GYRE-zco.kt6.before.S` | 11738.0 | 11738.0 | 1683 / 18000 | 4356 / 4537 |
| `GYRE-zco.kt6.before.T` | 19816.0 | 14635.0 | 5948 / 18000 | 7486 / 7797 |
| `GYRE-zco.kt6.before.ssh` | 4380.75 | 4380.75 | 293 / 600 | 305 / 295 |
| `GYRE-zco.kt6.before.u` | 3004.1015625 | 2823.2806396484375 | 8476 / 17400 | 8876 / 8524 |
| `GYRE-zco.kt6.before.v` | 3072.5 | 2834.6484375 | 8309 / 17100 | 8726 / 8374 |
| `GYRE-zco.kt7.after.uu_b` | 22.859375 | 22.859375 | 198 / 580 | 316 / 264 |
| `GYRE-zco.kt7.after.vv_b` | 22.35986328125 | 18.99090576171875 | 268 / 570 | 255 / 315 |
| `GYRE-zco.kt7.before.S` | 20364.0 | 20364.0 | 2529 / 18000 | 5746 / 5709 |
| `GYRE-zco.kt7.before.T` | 25207.0 | 25207.0 | 6491 / 18000 | 8158 / 8133 |
| `GYRE-zco.kt7.before.ssh` | 6386.53125 | 5995.265625 | 305 / 600 | 294 / 306 |
| `GYRE-zco.kt7.before.u` | 6209.64453125 | 4957.15234375 | 8745 / 17400 | 8625 / 8775 |
| `GYRE-zco.kt7.before.v` | 10752.484375 | 9629.609375 | 8556 / 17100 | 8516 / 8584 |
| `GYRE-zco.kt8.after.uu_b` | 34.525390625 | 34.525390625 | 208 / 580 | 310 / 270 |
| `GYRE-zco.kt8.after.vv_b` | 31.423828125 | 31.423828125 | 181 / 570 | 336 / 234 |
| `GYRE-zco.kt8.before.S` | 28922.0 | 22662.0 | 3160 / 18000 | 6546 / 6524 |
| `GYRE-zco.kt8.before.T` | 34704.0 | 34704.0 | 6890 / 18000 | 8435 / 8423 |
| `GYRE-zco.kt8.before.ssh` | 11221.5458984375 | 9706.609375 | 295 / 600 | 304 / 296 |
| `GYRE-zco.kt8.before.u` | 22170.09375 | 22170.09375 | 8694 / 17400 | 8681 / 8719 |
| `GYRE-zco.kt8.before.v` | 71113.0625 | 71113.0625 | 8683 / 17100 | 8399 / 8701 |
| `GYRE-zco.kt9.after.uu_b` | 30.4052734375 | 30.4052734375 | 297 / 580 | 243 / 337 |
| `GYRE-zco.kt9.after.vv_b` | 30.94140625 | 30.94140625 | 298 / 570 | 234 / 336 |
| `GYRE-zco.kt9.before.S` | 28565.0 | 28565.0 | 3801 / 18000 | 7126 / 7115 |
| `GYRE-zco.kt9.before.T` | 38204.0 | 38204.0 | 7115 / 18000 | 8464 / 8654 |
| `GYRE-zco.kt9.before.ssh` | 15552.1689453125 | 10692.7578125 | 293 / 600 | 307 / 293 |
| `GYRE-zco.kt9.before.u` | 16985.4296875 | 12630.5 | 8824 / 17400 | 8559 / 8841 |
| `GYRE-zco.kt9.before.v` | 97644.15625 | 97644.15625 | 8671 / 17100 | 8401 / 8699 |

### External-mode bits reached by the carried field

Equal-input kt1 Kaa, re-measured at `4b6c421df5d0`:

| card | U | V |
|---|---|---|
| GYRE-zco | `0 / 580` bit-identical | `0 / 570` bit-identical |
| LOCK_EXCHANGE-zco | `1 / 127`, max `2.168404344971009e-19` DEBT | `0 / 390` UNINFORMATIVE |
| OVERFLOW-zps | `7 / 199`, max `6.938893903907228e-18` DEBT | `0 / 606` UNINFORMATIVE |

The V rows are structurally uninformative on both lane-1 tanks, and both U rows
are upstream external-mode debt, not state-storage error.  The GYRE planted
control exits `1`.

For OVERFLOW the 19-frame substep-1 record is exact through entry, midpoint,
transport, continuity, PGF, slow forcing and drag.  The first unequal frame is
`u_exit` at `3.469446951953614e-18` (AT-BAR), produced by the flux-form external
update `dynspg_ts.F90:735-761` (the `ua_e`/`va_e` statement is `:752-761`); the
later weighted accumulation and division sit at `:825-847` and the Kaa velocity
conversion at `:870-895`.  The preregistered "only the final Kaa differs"
prediction is therefore **REFUTED**; the statement boundary is **CONFIRMED** and
its arithmetic owner remains **UNMEASURED_AFTER_REFUTED_ARMS** (source-rounding
the flux update and routing the non-selected `lk_linssh` formula were separate
one-variable arms; neither moved `u_exit` nor the `7 / 199` row, and both were
reverted).  This record was re-run at `4b6c421df5d0` on a clean tree; the
round-24-era copy carried an `-dirty` stamp and is superseded.  Its planted
exit control exits `1`.

LOCK has no 19-frame substep oracle, so assigning OVERFLOW's statement to LOCK's
one U bit would be self-agreement.  The committed user-run acquisition
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round25_lock_external_oracle/run.sh`
builds a distinct scalar-math LOCK configuration, refuses unless LOCK and
OVERFLOW compile the same shipped `dynspg_ts.F90` body, refuses a pre-existing
target, requires every pre-existing identity record and the final restart to
stay byte-for-byte, rejects a binary carrying vector-math symbols, and emits the
existing 19-frame format.  **It must be executed by the user**: it runs
`makenemo` and `mpirun`, which are outside the agent sandbox.  Until it is run,
LOCK's first producing statement is honestly **UNMEASURED**.

### Independent re-measurement of every round-25 figure

Round 25's measurements were taken by a session that ended mid-round.  Every
figure above was re-run from scratch at `4b6c421df5d0` on a clean tree and
compared cell by cell against that session's own reports through the shared
oracle-relative move gate.  Nothing moved.

| gate | rows | max worsening (row-scale ulp) | first_over_bar |
|---|---:|---:|---|
| GYRE trajectory kt1-10 | 70 | 0 | unchanged, kt2 `T/S/u/v` |
| LOCK stage sweep (`--faithful-only`) | 9 | 0 | absent |
| LOCK trajectory (`--continue-after-first`) | 50 | 0 | unchanged, kt4 `u` |
| OVERFLOW stage sweep (`--faithful-only`) | 9 | 0 | absent |
| OVERFLOW trajectory (`--continue-after-first`) | 50 | 0 | unchanged, kt2 `T/u` |

All eight stage-2 composition modes reproduce the LANDED values bit for bit,
as do the three equal-input Kaa rows and the OVERFLOW 19-frame record.  The
ending session left several same-mode reports under distinct names as it
walked the arms (`*_literal`, `*_direct`, `*_gdept`, `*_ordered`); only the
`*_gdept` set is the landed tree, and it is that set the re-run matches.  The
cross-card Rule-12 verdicts against round 24 are:

```text
ORACLE_RELATIVE_COMPARE PASS: rows=9  max_worsening_ulps=0         first_over_bar='<absent>'->'<absent>'                                  plant=None
ORACLE_RELATIVE_COMPARE PASS: rows=50 max_worsening_ulps=0.0234375 first_over_bar={'fields': ['u'], 'kt': 4}->{'kt': 4, 'fields': ['u']}   plant=None
ORACLE_RELATIVE_COMPARE PASS: rows=9  max_worsening_ulps=0.125     first_over_bar='<absent>'->'<absent>'                                  plant=None
ORACLE_RELATIVE_COMPARE PASS: rows=50 max_worsening_ulps=2         first_over_bar={'fields': ['T', 'u'], 'kt': 2}->{'kt': 2, 'fields': ['T', 'u']} plant=None
```

in the order LOCK stage, LOCK trajectory, OVERFLOW stage, OVERFLOW trajectory.
No first-over-bar comes earlier on any card.

Two claims from the ending session are **struck for want of a committed probe**
(a throwaway probe's number is unmeasured): the "4,579 differing `gdept_z0`
cells, maximum `9.094947017729282e-13` m, first live example
`(i,j,k)=(1,1,5)`" census, and the `0x404e5dbfcaf8a971` / `...972` bit-pattern
pair.  Neither is reproducible from anything in the tree.  What replaces them
is the ablation table above plus the committed `oracle_input_hpg` row, which is
exactly `0.0` on both components.

### Focused tests, and a PRE-EXISTING failure set that is not round 25's

```text
6 failed, 110 passed in 3158.98s (0:52:38)
```

(`tests/ocean/fidelity/test_nemo_testcase_l2_gyre_phase3_gate.py`,
`tests/ocean/fidelity/test_nemo_prognostic_barotropic_state_gate.py`,
`tests/ocean/unit/test_nemo_prognostic_barotropic_state.py`,
`tests/ocean/unit/test_nemo_ws_*.py`, `tests/ocean/unit/test_nemo_io.py`,
`tests/ocean/unit/test_nemo_recipe.py`.)

The six failures are **NOT round-25 regressions.**  Reverting the five files
round 25 touched to their `e80dd94b8276` content and rerunning the two affected
modules gives the identical set:

```text
6 failed, 19 passed in 504.98s (0:08:24)
```

Five of them are `tests/ocean/unit/test_nemo_recipe.py::test_nemo_gyre_*` plus
`test_surface_stress_implicit_wiring`, all raising

```text
ValueError: NEMO rk3_ws is one coupled momentum/tracer stage program; select rk3_ws for both integrators or for neither
```

from `LatLonCGridOceanModel._validate_config`; that guard landed in
`36d4a2f72ffe` (2026-08-31), an ancestor of the round-24 tip, and the GYRE
native card still selects `rk3_ws` for one integrator only.  The sixth,
`test_nemo_ws_tracer_rk3.py::test_overflow_bbl_is_live_inside_real_rk3_stage3`,
asserts `movement > 1.0e-12` and measures exactly `0.0`.

Deciding which integrator that card selects, or whether the BBL expectation or
the code is wrong, is a configuration choice and is **NOT made here**.  It is
raised as an open question.  No round-25 gate consumes these two cards.

### What is still open

- **LOCK's first producing external-mode statement — UNMEASURED.** Blocked on
  the user running the committed acquisition script; no substitute claim is
  made from OVERFLOW.
- **OVERFLOW's `u_exit` arithmetic owner — UNMEASURED after two refuted arms.**
  The statement boundary is confirmed; the owner is not.
- **The AT-BAR vorticity tail (`9.926167350636332e-24`) — deliberately
  unlabelled.** Nine orders below the bar; one arm was tried and refuted.
- **GYRE kt2 `T/S/u/v` — DEBT, unchanged owner.** The stage-2 fix did not move
  `first_over_bar`.
- **56 GYRE trajectory rows carry Rule-12 compensating-error debt.**
- **Six PRE-EXISTING focused-test failures on this branch.**  Five GYRE
  native-card tests and one OVERFLOW BBL liveness test fail at the round-24
  tip as well.  They need a configuration decision that is not made here.
- **C1D — struck, delegated to lane 3b.**
- **Stage-3 transport and the ZDF/TKE matrix walk — NOT ENTERED** across the
  open register.
- **Independent adversarial review of round 25 — OUTSTANDING.**

Verdict stays **HOLD**.  Nothing is merged into the reconciled/integration line.

### Asked / unasked disposition

| choice | disposition |
|---|---|
| Rule-11 correction of the stage-2/kt1 Kbb owner | ASKED; retracted as REFUTED AND ILL-POSED |
| enable the carried pair on every NEMO identity recipe | ASKED; implemented; non-NEMO leaves stay `None` |
| four negative acceptance guards | ASKED; implemented, each shown to fail under a planted break |
| stage-2 ordered walk | ASKED; HPG exact, completed Kaa AT-BAR, vorticity tail not over-labelled |
| LOCK/OVERFLOW external-bit walk | ASKED; OVERFLOW localized with two refuted arms; LOCK acquisition supplied, not run |
| re-measure every round-25 figure at the landed tip | ASKED (finish the round honestly); all reproduce at 0 ulp |
| correct wrong NEMO line citations in the round-25 text | not a scientific choice; corrected against the shipped source |
| strike two throwaway-probe numbers | not a scientific choice; no committed probe stands behind them |
| C1D numerical claim | FORBIDDEN here; struck and delegated |
| per-card HPG or external switch | UNASKED/FORBIDDEN; none added |
| stage-3/ZDF/TKE walk | UNASKED across this boundary; not entered |
| merge or push | FORBIDDEN; neither performed |

UNASKED list: empty.

Every artifact hash, oracle root, commit and test line quoted in this section
is pinned in `manifests/nemo_testcase_l2_gyre_round25.json`; the reports
themselves live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round25/verify/`.

## Round 26 — Rule-12 eligibility on the other cards, and three retired claims

Round 26 starts from `991f9f95047d` on a clean tree and runs in the same
regime: CPU production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.
The preregistration
(`manifests/nemo_testcase_l2_gyre_round26_preregister.json`, committed at
`2fcb991fa373`) precedes every measurement below except three that were taken
before it was written; those three are named as such inside it and carry no
prediction.  No NEMO executable was run from the sandbox.  `/tmp/codex-c1d-r24`
and `/tmp/codex-orca2-r24` remain flagged and unread.

Oracle roots are named explicitly for every figure: GYRE
`round19_oracle_v2_external`, LOCK `nemo-testcases-l1/phase3/lock_kt1_10`,
OVERFLOW `nemo-testcases-l1/phase3/overflow_kt1_10`, and the user-run LOCK
external-mode acquisition
`nemo-testcases-l2/phase3/round25_lock_external_oracle`.

### Rule-11 record — the round-25 admission story is REFUTED

**Dead claim.**  Round 25 reported that the three
`oracle_transport_kt00000001_s{1,2,3}.bin` records in the LOCK acquisition
"differ ONLY in the unconsumed pre-`tra_adv_trp` `zFw` workspace slot".

**What killed it.**  A byte-range decomposition against `lock_kt1_10` puts
almost none of the difference in `zFw`.  Of the 266 differing bytes in the
stage-1 record, 266 are in `zFv`; the stage-2 and stage-3 records add 6 bytes
in `zFw`.  Within `zFv`, 264 of the 265 differing elements are signed-zero bit
flips that compare numerically equal, and the 265th is an uninitialised
subnormal `2.7594e-320` where the baseline holds `0.0`.  The two `zFw`
elements are likewise uninitialised subnormals near `6.9e-310`.

**What replaces it, and why the records are still admitted.**  Every differing
element lies in the HALO.  The admission gate reports
`changed_in_parser_projection: 0` for all five changed fields across the three
records: the owned `[2:-2, 2:-2, :]` interior that every reader consumes is
BIT-identical, including `zFv`, which the gate marks `slot_consumed: True`.
The admission is therefore stronger than the round-25 story claimed, and it
rests on a different fact.

### The LOCK external-mode acquisition is ADMITTED

```text
ROUND21_WRITE_ONLY_ADMISSION PASS: exact=24/27 changed=3 restart_equal=True plant=False
ROUND21_WRITE_ONLY_ADMISSION FAIL: exact=24/27 changed=3 restart_equal=True plant=True
```

24 of the 27 pre-existing records and the final restart are byte-identical to
`lock_kt1_10`; the three changed records are admitted by consumed-field
identity above; the four new
`oracle_overflow_bt_substeps_kt0000000{1,2,3,4}_call1.bin` are the requested
addition, keeping the historical OVERFLOW filename on purpose.  The
`--plant-consumed` control exits `1` on LOCK and on GYRE.

| record | raw differing bytes | field | changed elements | in parser projection | consumed |
|---|---:|---|---:|---:|---|
| `oracle_transport_kt00000001_s1.bin` | 266 | `zFv` | 265 | 0 | yes |
| `oracle_transport_kt00000001_s2.bin` | 272 | `zFv` | 265 | 0 | yes |
| `oracle_transport_kt00000001_s2.bin` |  | `zFw` | 2 | 0 | no |
| `oracle_transport_kt00000001_s3.bin` | 272 | `zFv` | 265 | 0 | yes |
| `oracle_transport_kt00000001_s3.bin` |  | `zFw` | 2 | 0 | no |

The consumed-field admission RULE is card-independent, so the round-21 GYRE
probe was generalised (`--dims/--restart/--allowed-new/--writer`) rather than
copied.  GYRE's default run is unchanged at `PASS exact=39/49`.

**Its provenance check was vacuous and is now real.**  `run.sh` asserted that
LOCK and OVERFLOW compile the same shipped `dynspg_ts` body by `cmp`-ing two
`WORK/dynspg_ts.F90` entries.  Both are symlinks to
`src/OCE/DYN/dynspg_ts.F90`, so it compared a file with itself.  The premise
happens to be true, verified out of band on the already-run acquisition:
neither `LOCK_EXCHANGE_OMIP_L1_P3` nor `OVERFLOW_OMIP_L1_P3` overrides
`dynspg_ts.F90` in `MY_SRC`, and the transplanted instrument
(`tests/OVERFLOW_OMIP_L1_BTWALK4/MY_SRC/dynspg_ts.F90`) is a PURE ADDITION
over the shipped body — 53 added lines, **0** deleted or changed.  The script
now checks exactly those three things and refuses otherwise.

The substep reader also stopped being OVERFLOW-only: `--case`/`--expected`
replace the hardcoded `CASE="OVERFLOW-zps"` and `EXPECTED=(206,7,4,19,64)`.
LOCK's own record header, read off the file, is `(134, 7, 1, 19, 64)` — ONE
barotropic substep against OVERFLOW's four.

### Rule-12 eligibility, discharged on the two lane-1 tanks

> **RETRACTED IN PART — read "Round-26 independent review" below before using this subsection.**  The measured rows stand exactly as printed, but the frame is NOT `dyn_hpg` alone and the citation of `stprk3_stg.F90:309-334` is wrong; the dump is at `MY_SRC/stprk3.F90:206`, before stage 1 and after `dyn_spg_ts`.  The row is renamed `kt1.stp2d.momentum_rhs`.  The paragraph that follows is kept unedited as the dead claim, next to what killed it.

Round 25 landed two source associations inside NEMO's hydrostatic pressure
gradient and showed the changed operator bit-exact given NEMO's inputs on GYRE
only.  Rule 12 asks for that row on every card that executes it, and the
testcase recipe pins `pgf_scheme="nemo_sco"` on all of them
(`nemo_testcase_recipe.py:375,630,3368`).

The lane-1 tanks have no HPG-literal dump and do not need one.  NEMO
accumulates `dyn_hpg`, `dyn_vor` and `dyn_adv` into a zeroed `puu(Krhs)`
(`stprk3_stg.F90:309-334`); `dyn_vor` and `dyn_adv` are bilinear in the
velocity; both tanks start from rest.  So the dumped `oracle_rhs_kt00000001`
frame IS the hydrostatic pressure gradient alone.  All three preconditions are
asserted rather than assumed.

| card | equal inputs (T/S/u/ssh unequal) | NEMO entry rest check | changed functions called (`gdept_z0` / HPG interface) | U row | verdict |
|---|---|---|---|---|---|
| GYRE-zco (round 25) | `oracle_input_hpg` route | n/a | n/a | `0.0`, `0 / 17400` | bit-exact |
| LOCK_EXCHANGE-zco | `0/2560`, `0/2560`, `0/2540`, `0/128` | max abs u and v both `0.0` | 1 / 2 | `0.0`, `0 / 2540` | **bit-exact** |
| OVERFLOW-zps | `0/17000`, `0/17000`, `0/16900`, `0/200` | max abs u and v both `0.0` | 1 / 2 | `0.0`, `0 / 16900` | **bit-exact** |

The preregistered prediction is **CONFIRMED** on both cards.  The planted
control exits `1` on both.  The V component is WAIVED in writing with its
measured count: both tanks are single-wet-row channels with **0** wet V faces,
so a V row would compare masked zeros with masked zeros.

Rule-12 eligibility for the round-25 change is therefore discharged on GYRE,
LOCK and OVERFLOW.  ORCA2 is **UNMEASURED**, for the reason below.

### ORCA2 as the fourth card — BLOCKED, in both directions

ORCA2 could not be scored against this tip and the failure is a branch
divergence, not a numerical one.

| direction | blocker |
|---|---|
| ORCA2 gates against the round-25 model | `ImportError` on `build_orca2_zps_card`, `validate_nemo_testcase_card_for_execution`, `nemo_qco_wzv_recurrence`, `nemo_transport_wzv_divergence_level` — none exist at `991f9f95047d` |
| ORCA2 gates on their own branch | that tip still carries the PRE-fix fused `gdept_z0` (its `_bc_ke_and_pressure_gradients` builds it as one expression, with no `_nemo_qco_gdept_z0`; line numbers on another branch are not citable here), i.e. byte-for-byte round-25 ablation arm A; `d5a7f8169507` is not its ancestor |

The two branches diverged at `03c6e8d96ff7` with 49 commits on this side and
170 on the ORCA2 side, 263 changed lines in `ocean_pe_latlon_cgrid.py` and 550
in `nemo_testcase_recipe.py`.  Closing this is a branch-integration decision,
which is an OPEN QUESTION and not a choice made here.  The ORCA2 oracle data
itself is intact (101 `.bin` records, WZV record present).  The ORCA2 HPG gate
was deliberately NOT run: it calls `nemo_hpg_sco_literal_cgrid` on dumped
operands rather than the model's own HPG path, so it is blind to exactly the
interface round 25 changed.  Evidence
`round26/orca2/orca2_fourth_card_blocked_round26.json`, SHA-256
`35e80a445bed1f1c0cf8bfe4459acf261b517d05c044f2eac9a74a1dc876a3de`.

### The stage-operand guard had a hole, and it is closed

Round 25's fifth guard walked the AST of the routine that runs and required
each `barotropic_velocity=` operand to be a selector call, unwrapping the
`None if <legacy arm> else <selector>` conditional through `.orelse` ONLY.  A
bypass placed in the BODY branch is live whenever the
`legacy_reduced_stage_transport_mean_arm` hook is set, and the guard never
looked at it.

Planted exactly that at the stage-1 call site —
`(u0, v0) if _legacy_reduced_stage_transport_mean_arm else
_nemo_ws_stage_barotropic_velocity(1, ...)`:

| guard | result on the planted `.body` bypass |
|---|---|
| round-25 form (`.orelse` only), run verbatim over the planted source | **PASSES** — the hole was real |
| round-26 form | fails: `the guarded arm must disable the operand, not supply a second one: Tuple(elts=[Name(id='u0'...` |

The plant was reverted with `git checkout --` and `git status --porcelain`
confirmed empty.

### The FMA fact, and one struck number restored

Round 25 struck two throwaway-probe numbers.  One of them had a committed
probe all along and is **restored**: the `0x404E5DBFCAF8A971` /
`0x404E5DBFCAF8A972` pair is pinned by
`test_nemo_qco_gdept_z0_oracle_bit_pattern`, landed in `d5a7f8169507`, which
`git merge-base --is-ancestor` confirms is an ancestor of this tip.  It passes
(`1 passed in 0.65s`).  The other, the "4,579 differing cells" census, has no
probe and **stays struck** — and the code comment that still quoted it is
rewritten.

Why the two-statement association is the FAITHFUL one and not merely a
different one is now cited at `_nemo_qco_gdept_z0` itself:

| fact | evidence |
|---|---|
| the oracle's build enables no FMA | `arch/arch-conda-scalarmath.fcm` contains no `-march`, `-mfma`, `-mavx2` or `-ffast-math` (grep count 0) |
| the oracle's binary contains none | `objdump -d` of the scalar-math GYRE `nemo.exe` (SHA-256 `207e701f740b2fc4ee5f234a22508d774c2f517646e8b9d2f1e0a2a2d58d7ea9`) finds **0** `vfmadd`/`vfmsub`/`vfnmadd`/`vfnmsub` in 1868261 disassembled instructions |
| XLA does contract it | the committed bit-pattern probe's two outcomes differ in the last bit |

### Rule-12 compensating-error register, with boundary and owner

The round-24 comparison row for the completed stage-2 Kaa carried a
mis-transcribed V unequal count.  Corrected in place from `17100 / 17100` to
**`17077 / 17100`**, read off `round25/verify/stage2/ablationA_corrected.json`
(`n_unequal: 17077`); the maximum `2.2380914396075147e-15` is unchanged.

The 62 moved GYRE rows are re-emitted below with the two columns round 23 had.
56 exceed the two-row-scale-ULP threshold and are the register; the other 6
moved by less and are named but not registered.  Every registered row shares
one boundary and one owner, because one ablation established both:

- **B1** — the stage-2 `dyn_hpg` insertion into `Krhs`
  (`stprk3_stg.F90:324`), whose QCO depth operand is
  `domzgr_substitute.h90:139,145`.
- **O1** — **CONFIRMED**.  Round-25 ablation A (fused `gdept_z0`, one
  variable, file restored) reproduces round-24's DEBT to the digit on both
  components and its report hashes to the round-24 manifest's `gyre_stage2`
  artifact.

| row | move ULP | worse ULP | cells >2 ULP / n | boundary | owner |
|---|---:|---:|---:|---|---|
| `GYRE-zco.kt10.after.uu_b` | 51.05078125 | 51.05078125 | 305 / 580 | B1 | O1 |
| `GYRE-zco.kt10.after.vv_b` | 51.37939453 | 51.37939453 | 274 / 570 | B1 | O1 |
| `GYRE-zco.kt10.before.S` | 32631 | 25813 | 4164 / 18000 | B1 | O1 |
| `GYRE-zco.kt10.before.T` | 42227 | 42227 | 7242 / 18000 | B1 | O1 |
| `GYRE-zco.kt10.before.ssh` | 15503.85156 | 15503.85156 | 292 / 600 | B1 | O1 |
| `GYRE-zco.kt10.before.u` | 67625.80469 | 67625.80469 | 8688 / 17400 | B1 | O1 |
| `GYRE-zco.kt10.before.v` | 185097.1641 | 185097.1641 | 8738 / 17100 | B1 | O1 |
| `GYRE-zco.kt2.after.uu_b` | 0.1646118164 | 0.1646118164 | 0 / 580 | below 2 ULP | not registered |
| `GYRE-zco.kt2.after.vv_b` | 0.1110839844 | 0.08937835693 | 0 / 570 | below 2 ULP | not registered |
| `GYRE-zco.kt2.before.S` | 1 | 1 | 0 / 18000 | below 2 ULP | not registered |
| `GYRE-zco.kt2.before.T` | 2 | 2 | 0 / 18000 | below 2 ULP | not registered |
| `GYRE-zco.kt2.before.u` | 24.48193359 | 24.48193359 | 1955 / 17400 | B1 | O1 |
| `GYRE-zco.kt2.before.v` | 26.78320312 | 26.78320312 | 1911 / 17100 | B1 | O1 |
| `GYRE-zco.kt3.after.uu_b` | 1.295166016 | 1.295166016 | 0 / 580 | below 2 ULP | not registered |
| `GYRE-zco.kt3.after.vv_b` | 1.135742188 | 1.040527344 | 0 / 570 | below 2 ULP | not registered |
| `GYRE-zco.kt3.before.S` | 7 | 6 | 20 / 18000 | B1 | O1 |
| `GYRE-zco.kt3.before.T` | 102 | 99.5 | 533 / 18000 | B1 | O1 |
| `GYRE-zco.kt3.before.ssh` | 163.9160156 | 59.4765625 | 41 / 600 | B1 | O1 |
| `GYRE-zco.kt3.before.u` | 946.7714844 | 564.234375 | 3687 / 17400 | B1 | O1 |
| `GYRE-zco.kt3.before.v` | 551.7788086 | 532.5 | 3827 / 17100 | B1 | O1 |
| `GYRE-zco.kt4.after.uu_b` | 4.360582352 | 4.360582352 | 30 / 580 | B1 | O1 |
| `GYRE-zco.kt4.after.vv_b` | 2.551757812 | 2.551757812 | 5 / 570 | B1 | O1 |
| `GYRE-zco.kt4.before.S` | 6085 | 5820 | 199 / 18000 | B1 | O1 |
| `GYRE-zco.kt4.before.T` | 7622 | 7291 | 1796 / 18000 | B1 | O1 |
| `GYRE-zco.kt4.before.ssh` | 771.0390625 | 771.0390625 | 197 / 600 | B1 | O1 |
| `GYRE-zco.kt4.before.u` | 1109.068359 | 1106.624512 | 6493 / 17400 | B1 | O1 |
| `GYRE-zco.kt4.before.v` | 1090.886597 | 1090.886597 | 6401 / 17100 | B1 | O1 |
| `GYRE-zco.kt5.after.uu_b` | 7.946166992 | 7.946166992 | 151 / 580 | B1 | O1 |
| `GYRE-zco.kt5.after.vv_b` | 7.804199219 | 7.804199219 | 151 / 570 | B1 | O1 |
| `GYRE-zco.kt5.before.S` | 8493 | 5947 | 806 / 18000 | B1 | O1 |
| `GYRE-zco.kt5.before.T` | 18347 | 14829 | 4124 / 18000 | B1 | O1 |
| `GYRE-zco.kt5.before.ssh` | 3065.570312 | 3065.570312 | 289 / 600 | B1 | O1 |
| `GYRE-zco.kt5.before.u` | 1993.760742 | 1993.760742 | 8351 / 17400 | B1 | O1 |
| `GYRE-zco.kt5.before.v` | 2090.657104 | 2066.507812 | 8002 / 17100 | B1 | O1 |
| `GYRE-zco.kt6.after.uu_b` | 14.02636719 | 12.88476562 | 169 / 580 | B1 | O1 |
| `GYRE-zco.kt6.after.vv_b` | 15.80957031 | 15.80957031 | 246 / 570 | B1 | O1 |
| `GYRE-zco.kt6.before.S` | 11738 | 11738 | 1683 / 18000 | B1 | O1 |
| `GYRE-zco.kt6.before.T` | 19816 | 14635 | 5948 / 18000 | B1 | O1 |
| `GYRE-zco.kt6.before.ssh` | 4380.75 | 4380.75 | 293 / 600 | B1 | O1 |
| `GYRE-zco.kt6.before.u` | 3004.101562 | 2823.28064 | 8476 / 17400 | B1 | O1 |
| `GYRE-zco.kt6.before.v` | 3072.5 | 2834.648438 | 8309 / 17100 | B1 | O1 |
| `GYRE-zco.kt7.after.uu_b` | 22.859375 | 22.859375 | 198 / 580 | B1 | O1 |
| `GYRE-zco.kt7.after.vv_b` | 22.35986328 | 18.99090576 | 268 / 570 | B1 | O1 |
| `GYRE-zco.kt7.before.S` | 20364 | 20364 | 2529 / 18000 | B1 | O1 |
| `GYRE-zco.kt7.before.T` | 25207 | 25207 | 6491 / 18000 | B1 | O1 |
| `GYRE-zco.kt7.before.ssh` | 6386.53125 | 5995.265625 | 305 / 600 | B1 | O1 |
| `GYRE-zco.kt7.before.u` | 6209.644531 | 4957.152344 | 8745 / 17400 | B1 | O1 |
| `GYRE-zco.kt7.before.v` | 10752.48438 | 9629.609375 | 8556 / 17100 | B1 | O1 |
| `GYRE-zco.kt8.after.uu_b` | 34.52539062 | 34.52539062 | 208 / 580 | B1 | O1 |
| `GYRE-zco.kt8.after.vv_b` | 31.42382812 | 31.42382812 | 181 / 570 | B1 | O1 |
| `GYRE-zco.kt8.before.S` | 28922 | 22662 | 3160 / 18000 | B1 | O1 |
| `GYRE-zco.kt8.before.T` | 34704 | 34704 | 6890 / 18000 | B1 | O1 |
| `GYRE-zco.kt8.before.ssh` | 11221.5459 | 9706.609375 | 295 / 600 | B1 | O1 |
| `GYRE-zco.kt8.before.u` | 22170.09375 | 22170.09375 | 8694 / 17400 | B1 | O1 |
| `GYRE-zco.kt8.before.v` | 71113.0625 | 71113.0625 | 8683 / 17100 | B1 | O1 |
| `GYRE-zco.kt9.after.uu_b` | 30.40527344 | 30.40527344 | 297 / 580 | B1 | O1 |
| `GYRE-zco.kt9.after.vv_b` | 30.94140625 | 30.94140625 | 298 / 570 | B1 | O1 |
| `GYRE-zco.kt9.before.S` | 28565 | 28565 | 3801 / 18000 | B1 | O1 |
| `GYRE-zco.kt9.before.T` | 38204 | 38204 | 7115 / 18000 | B1 | O1 |
| `GYRE-zco.kt9.before.ssh` | 15552.16895 | 10692.75781 | 293 / 600 | B1 | O1 |
| `GYRE-zco.kt9.before.u` | 16985.42969 | 12630.5 | 8824 / 17400 | B1 | O1 |
| `GYRE-zco.kt9.before.v` | 97644.15625 | 97644.15625 | 8671 / 17100 | B1 | O1 |

Legend: **B1** and **O1** as above; rows marked "below 2 ULP" moved but do not
enter the register.

**Named footgun on the `nemo_sco` arm.**  `_bc_ke_and_pressure_gradients`
returns `dp_dx`/`dp_dy` as literal ZEROS when `pgf_scheme="nemo_sco"`
(the `pgf_scheme="nemo_sco"` arm assigns `dp_dx_sco = jnp.zeros_like(hpg_u)`); the acceleration is carried in
`direct_hpg_u/v` instead, because `dynhpg.F90:359,383` writes acceleration
straight into `Krhs` and leaving a synthetic `-rho0*hpg` in the graph lets XLA
rediscover the algebraic pressure interface and lose the source bit.  Any
probe or consumer that reads `dp_dx`/`dp_dy` on this arm reads zeros, not a
pressure gradient.  Registered here so the next round does not measure them.

### Decision 15A — the native demo GYRE card's tracer integrator

NEMO's `key_RK3` advances momentum AND tracers inside ONE stage routine:
`stp_RK3_stg` opens at `stprk3_stg.F90:65`, calls `dyn_hpg`/`dyn_vor`/
`dyn_adv`/`dyn_zdf` at `:315-430` and `tra_adv`/`tra_sbc_RK3`/`tra_ldf`/
`tra_zdf` at `:463-599`, and closes at `:655`.  The tracer integrator is
therefore not a free choice beside `momentum_time_integrator="rk3_ws"`; it is
the same program, which is what the coupled-integrator guard added in
`36d4a2f72ffe` says.  Per user decision 15A the card now selects `rk3_ws` for
both.

The change advances the failure one guard deeper rather than clearing it, and
that is the honest result:

| guard | before decision 15A | after |
|---|---|---|
| coupled momentum/tracer integrator | RAISES | passes |
| complete rk3_ws momentum program | not reached | RAISES |

The card resolves `vertical_momentum_scheme="upwind_perturbation"` and
`adaptive_implicit_vertadv=True`, both of which disagree with NEMO: the vector
form is "keg + zad + vor" (`dynadv.F90:144`), i.e. the advective `dyn_zad`,
and `ln_zad_Aimp` defaults to `.false.` (`namelist_ref:1177`) with no override
in `GYRE_PISCES`'s `namelist_cfg`.  Both are reported as findings and left as
an OPEN QUESTION; changing them is a second configuration choice this round
was not given.  `test_surface_stress_implicit_wiring` reveals nothing new: it
never reaches its own `nemo_stage_mean_imposition` assertion, because
construction now raises the momentum-program message instead — it is masked
one guard deeper, not unmasked.

### LOCK's external-mode bit, walked — preregistered prediction REFUTED

The user-run acquisition supplies the 19-frame `NEMO_L1_OVBT_1` record LOCK
never had.  Re-measured on a clean tree at `66f9145c951e`, production
`nemo_flux_form_update` arm, substep 1 of 1:

| frame | verdict | max abs | unequal / n |
|---|---|---:|---:|
| `eta_entry` | AT-BAR | 0 | 0 / 128 |
| `u_entry` | AT-BAR | 0 | 0 / 127 |
| `eta_mid` | AT-BAR | 0 | 0 / 128 |
| `u_mid` | AT-BAR | 0 | 0 / 127 |
| `transport_u` | AT-BAR | 0 | 0 / 127 |
| `eta_continuity` | AT-BAR | 0 | 0 / 128 |
| `eta_pgf` | AT-BAR | 0 | 0 / 128 |
| `pgf_u` | AT-BAR | 0 | 0 / 127 |
| **`slow_u`** | AT-BAR | **`2.1684043449710089e-19`** | **1 / 127** |
| `drag_u` | AT-BAR | 0 | 0 / 127 |
| `u_exit` | AT-BAR | `2.1684043449710089e-19` | 1 / 127 |
| `eta_exit` | AT-BAR | 0 | 0 / 128 |

Every V frame is UNMEASURED: LOCK has no active meridional face.  Both plants
(`--plant-entry`, `--plant-exit`) exit `1`.

**The preregistered prediction is REFUTED.**  It named the flux-form external
velocity update as the first departure.  The first departure is `slow_u`, one
frame earlier and OUTSIDE the substep loop; `u_exit` carries the identical
value in the identical cell `[1, 64]` and therefore inherits it rather than
producing it.  This is round 25's unlocalized LOCK bit (`1 / 127`,
`2.168404344971009e-19`), now localized.

The same shape was already recorded for OVERFLOW inside its own gate — its
`u_exit` tail "is inherited from the slow forcing, not produced by the update"
— so round 25's receipt text naming `dynspg_ts.F90:752-761` as OVERFLOW's
producer is narrowed here to a boundary the update INHERITS.

**Producing statement, read from the source, not assumed.**  LOCK compiles
`key_qco key_vco_1d key_RK3`
(`tests/LOCK_EXCHANGE_OMIP_L1_P3/cpp_LOCK_EXCHANGE_OMIP_L1_P3.fcm`), so the
RK3 branch runs and the MLF branch is dead.  Under RK3 the slow forcing is one
statement, `zu_frc(:,:) = Ue_rhs(:,:)` at `dynspg_ts.F90:282`; the
`dyn_cor_2D` subtraction at `:296` contributes exactly zero because
`usrdef_hgr.F90:103-104` sets `pff_f = pff_t = 0`.  So the bit is INHERITED
and produced upstream in `stp2d.F90`.  LOCK resolves `ln_dynadv_up3 = .true.`
(`namelist_cfg:85`) and therefore `np_FLX_up3`, which writes
`Ue_rhs` twice: `dyn_adv_up3(..., pUe=Ue_rhs)` at `stp2d.F90:172` and the
cumulated depth mean `Ue_rhs = Ue_rhs + SUM(e3u_0*uu(Krhs)*umask)*r1_hu_0` at
`:185`.  Wind (`:200`) and drag (`:196`) contribute nothing on this card
(`usrdef_sbc.F90` utau = 0, `ln_drg_OFF = .true.`).  **Which of those two
statements owns the bit is PLAUSIBLE, not CONFIRMED — no dumped frame
separates them**, and no arm was run to guess.

Three report blocks would have become false records on LOCK and are now
withheld by name (`NOT_APPLICABLE_ON_THIS_CARD`): `resolved_program` hardcodes
OVERFLOW's `nn_bt_flt=1 / rn_bt_alpha=0.0 / nn_e=3 / actual_icycle=4` where
LOCK's `namelist_cfg:105-106` says `nn_bt_flt=3, rn_bt_alpha=0.07`;
`whole_step_kt2_causal_arm` scores against a hardcoded OVERFLOW prior triple;
`ownership` is an OVERFLOW owner-label set.  The card's own resolved program
was instantiated and printed rather than assumed: OVERFLOW
`("nemo_boxcar1_ab3", 3)`, LOCK `("nemo_ab3am4", 1)`.

The generalization is a bit-for-bit no-op on OVERFLOW.  A one-variable control
ran the same gate on its defaults from a detached worktree at the round-25 tip
`991f9f95047d`:

| tip | status | `first_over_bar` (both arms) |
|---|---|---|
| `991f9f95047d` (before) | DEBT | `pgf_v`, substep 2, `3.1252342425882175e-4` |
| `66f9145c951e` (after) | DEBT | `pgf_v`, substep 2, `3.1252342425882175e-4` |

All 76 substep rows, the 19 substep-1 frames, `resolved_program` and the
scaling block are identical.  (The much older
`nemo-testcases-l1/barotropic_walk/overflow_barotropic_frame_gate.json`, which
reports `u_exit` at `1.8735013540549517e-15`, is a stale-format report with no
`legoesm_git_sha` and no substep-1 frame list; it is NOT a valid baseline and
was not used as one.)

### Round-26 independent review — two reviewers, four retractions

Two independent reviews ran on the round-26 diff and on its claims.  They
CONVERGED on the same defect, which is the strongest signal available, and
they were right.  Every finding below was re-measured before being accepted or
rejected.

**RETRACTION 1 — "the dumped frame is the HPG alone": REFUTED.**  The frame is
not a stage-1 frame at all.  `l1_dump_rhs` runs at `MY_SRC/stprk3.F90:206`,
immediately after `stp_2D` at `:204` and BEFORE stage 1 at `:215`.  The cited
`stprk3_stg.F90:309-334` is the `CASE(2,3)` block; `CASE(1)` calls only
`dyn_adv`.  What the frame holds, in `stp2d.F90` order: `dyn_hpg` at `:128`
(which ASSIGNS over its loop range — "a zeroed Krhs" was also wrong),
`dyn_ldf` at `:131` called UNCONDITIONALLY, `dyn_vor` at `:146`, then
`dyn_spg_ts` at `:281`, which -- in the SHIPPED source, but NOT in the code
these cards compile -- REMOVES the vertical mean at
`dynspg_ts.F90:344-345` and ADDS the barotropic acceleration back at
`:938-975`.  Measured, so this is not a hypothetical: the dumped `uu_b(Kaa)`
maximum is `1.135367194865404e-3` on LOCK and `4.5029698607113644e-2` on
OVERFLOW.  The 3-D momentum advection is absent because both cards are flux
form (`ln_dynadv_up3 = .true.`) and it lands at `stprk3_stg.F90:315`, after
the dump.

`dyn_ldf` is inert only because both cards set `ln_dynldf_OFF = .true.` — a
NAMELIST fact, not a rest-state fact.  The row was right for a reason that was
not stated, so the gate now asserts the legoESM equivalent (all eight lateral
viscosity coefficients zero and `lateral_friction_scheme='none'`) instead of
relying on the coincidence.

**What survives.**  The measured row is unchanged — `0.0`, exact, on all 2540
LOCK and 16900 OVERFLOW faces — and the changed operator is still counted
executing inside it.  What changes is the STRENGTH of the claim, in both
directions: it is a stronger statement about how much of NEMO's step is
reproduced bitwise (the depth-mean removal and the barotropic add-back are
inside it), and a weaker ISOLATION, because a bit-level compensating error
between the pressure gradient and the mode split is not excluded by this row.
The row is renamed `kt1.stp2d.momentum_rhs`; a row named for an operator it
does not isolate is the false label this campaign exists to catch.

**RETRACTION 2 — the `zFv` consumed-identity argument is VACUOUS.**  `zFv` is
identically zero over the whole array on this card (maximum `2.759e-320`, the
single uninitialised subnormal).  "Consumed-equal on a consumed slot" is
therefore no evidence at all, and round 26's "stronger than round 25 claimed"
is withdrawn.  What actually admits the acquisition is the byte-identical
final restart plus 24 of 27 byte-identical records — trajectory identity, for
which no projection argument is needed.  "Differs only outside our parser's
projection" certifies OUR READER, not NEMO's run, and is not shipped as the
rule.

**RETRACTION 3 — LOCK's `u_exit` inheritance is downgraded to PLAUSIBLE.**  The
receipt omitted a number from the run's own report that cuts against it:
`scaling_check_before_owner_label` gives slow error times dt = `7.228e-19`
against an exit error of `2.168e-19`, ratio **0.3**.  The flux-form update
(`dynspg_ts.F90:750-753`) carries `zu_frc` with coefficient about `rDt_e`, so
pure inheritance predicts roughly 3 ULP at exit and 1 was measured.
Same-value/same-cell is weak evidence in any case: `2.168404344971009e-19` is
exactly `2**-62`, one ULP for any value in `[2**-10, 2**-9)`.  The
discriminating measurement is named and NOT run: transplant NEMO's dumped
`zu_frc` into legoESM's substep and recompute `u_exit` — bit-exact means
inherited, still-off means the update makes its own bit.  It needs no NEMO
run.  Until then the boundary stays CONFIRMED and the inheritance PLAUSIBLE.

**RETRACTION 4 — the BBL comment's stated reason was false.**  It said the
test's two `.set()` calls hit the same cell; `shelf_i` and `deep_i` are
different COLUMNS, so they do not collide.  The face gives `0.0` because the
three-leg exchange degenerates when `ku_s == ku_d`.  On a zps card that is a
partial-cell face, not a flat bottom, so **why it is marked BBL-active at all
is an open model question** the filter does not settle — recorded rather than
buried by the test fix.

**One reviewer finding REFUTED, with its evidence.**  A reviewer reported that
the gate's `resolved_program` block (`nn_bt_flt=1, rn_bt_alpha=0.0`) is false
because both shipped decks carry `nn_bt_flt=3, rn_bt_alpha=0.07`.  That reads
the SHIPPED decks, not what the oracle runs resolved.  The OVERFLOW oracle
run's own `namelist_cfg` has no `nn_bt_flt` line at all (grep count 0), so it
takes `namelist_ref:1092`'s default, and its `ocean.output:875` prints
`Barotropic time filter => nn_bt_flt = 1`.  LOCK's oracle `namelist_cfg:105-106`
does set `3` and `0.07`.  The per-card map is therefore correct as landed.

**Two more defects fixed.**  The `run.sh` replacement provenance check was
vacuous in the same way the check it replaced was: under `set -o pipefail`,
`diff … | grep -q '^<'` returns diff's own exit `1` whenever the files differ,
so the refusal could never fire — reproduced directly, a deliberately
deleted-line instrument passed.  It now counts the `<` lines.  And the
admission report published `plant_applied: true` on runs that planted nothing,
because it printed a one-shot sentinel instead of the fact.

**Both reviewer NITs turned out to be real holes and are FIXED.**  The
stage-operand AST guard checked only `args[0]`, so
`_nemo_ws_stage_barotropic_velocity(1, (target_u, target_v), (target_u,
target_v))` routes through the selector, carries the literal stage number, and
still hands stage 1 the same-step pair the selector exists to keep out.
Planted exactly that: the guard as landed earlier this round PASSES on it, the
tightened guard fails.  The Kbb operand must now differ from the Nnn operand
and must read the carried `uu_b/vv_b`.  That is the SECOND hole found in this
one guard this round, which is itself the finding — an AST guard is only as
good as the specific bypasses someone has tried against it.

Round-21's `--writer` now fails closed when more than one record kind changed,
so one instrument's `file:line` can never be stamped over records written by
another.  Shown non-vacuous: GYRE's default run, whose ten changed records
span several kinds, PASSES without `--writer` and FAILS with it, while the
LOCK run that legitimately names one instrument still passes
(`exact=24/27`, `restart_equal=True`).

The guard still walks the whole module rather than the routine its docstring
names; that remains recorded and unfixed, since narrowing it would need the
enclosing-function walk the docstring promises and no bypass has been shown
against it.

### The six pre-existing failures — one fixed, five remain

| failure | round-25 state | round-26 state |
|---|---|---|
| `test_nemo_gyre_native_builds_valid_model_and_dispatch` | fails: integrators not both `rk3_ws` | still fails, one guard deeper: incomplete `rk3_ws` momentum program |
| `test_nemo_gyre_coordinate_is_consistent_clean_w_bc` | same | same |
| `test_nemo_gyre_forced_trajectory_is_finite_and_stable` | same | same |
| `test_nemo_gyre_wind_forcing_sign_chain_end_to_end` | same | same |
| `test_surface_stress_implicit_wiring` | same | same; it still never reaches its own assertion |
| `test_overflow_bbl_is_live_inside_real_rk3_stage3` | fails: movement measured `0.0` | **PASSES** (`1 passed in 428.34s`) |

`tests/ocean/unit/test_nemo_recipe.py` after decision 15A: `5 failed, 19 passed
in 49.39s`.  The five remaining need the configuration decision in open
question 2 and are NOT round-26 regressions.

The whole module the BBL fix touches was then re-run at the round-26 tip on a
clean tree, so the fix is confirmed against its 30 neighbours and not only
against itself:

```text
tests/ocean/unit/test_nemo_ws_tracer_rk3.py
======================= 31 passed in 1468.36s (0:24:28) ========================
```

That module also carries `test_nemo_qco_gdept_z0_oracle_bit_pattern`, so the
restored bit-pattern pair is green in the same run.

The wide focused suite was also run, and it reproduces round 25's failure set
exactly:

```text
6 failed, 111 passed in 2963.24s (0:49:23)
```

Its BBL row is still red, and that is a STALE-COLLECTION artifact, not a
contradiction: that session was launched before the BBL fix was written and
pytest imports every selected module at collection, so it ran the pre-fix
module for its whole 49 minutes.  It is therefore the BEFORE snapshot.  The
AFTER result is the dedicated module run above, taken at the round-26 tip on
a clean tree, where the same test passes inside `31 passed`.  The five
`test_nemo_recipe` rows are red in both and need open question 2.  The new eligibility gate's
reader guards are `7 passed in 0.06s`, and they were shown non-vacuous by
deleting the reader's time-level guard, which turned two of them red with
`DID NOT RAISE` before the file was restored.

### OVERFLOW's seven Rule-12 rows, against the same boundary

The LOCK result generalizes what OVERFLOW's own gate had already recorded and
round 25's receipt text had over-stated.  OVERFLOW's `u_exit` tail is
"inherited from the slow forcing, not produced by the update" — its `ownership`
block says so and prints the ratio (`u_exit` error equals `dt` times the
`5.63e-16` `slow_u` input residual, ratio 0.998).  Round 25's sentence naming
`dynspg_ts.F90:752-761` as the producer is therefore narrowed to a statement
that INHERITS.  Re-run at this tip, OVERFLOW is unchanged in every substep row
and its first over-bar frame is `pgf_v` at substep 2
(`3.1252342425882175e-4`), a V frame LOCK cannot see because it has no active
meridional face.  The seven OVERFLOW trajectory rows therefore stay registered
behind the same slow-forcing boundary, and their arithmetic owner remains
**UNMEASURED_AFTER_REFUTED_ARMS** from round 25's two refuted arms.  No new arm
was run to guess it.

### The OVERFLOW BBL liveness test aimed at a degenerate face

`test_overflow_bbl_is_live_inside_real_rk3_stage3` plants a dense shelf cell
over a light deep cell across an active BBL face and requires the step to
move.  It measured exactly `0.0`.  The cause is the face it picked, not the
BBL code: `bbl_static_geometry` marks flat-bottom faces active as well, and
there `ku_s == ku_d`, so the test's two `.set()` calls address the SAME cell
and the second overwrites the first.  No density contrast is planted at all.

| face | `ku_s`, `ku_d` | movement |
|---|---|---:|
| `(1, 94)` — the median active face the test picked | 99, 99 | `0.0` |
| `(1, 165)` — also flat | 99, 99 | `0.0` |
| `(1, 40)` | 59, 64 | `0.17292126063998048` |
| `(1, 22)` | 24, 25 | `0.3200974770662164` |

The test now selects among active faces with `ku_s != ku_d`.  This is a TEST
correction, not a model change; nothing in the BBL path was touched.

### Open questions

1. **ORCA2 cannot be scored without a branch decision.**  Do we back-port the
   four symbols the ORCA2 gates need onto this branch, merge the two branches,
   or forward-port the round-25 HPG/QCO fix onto the ORCA2 branch and score
   there?  Until one is chosen, Rule-12 eligibility on ORCA2 stays UNMEASURED.
3. **The native demo GYRE card's momentum program.**  It resolves
   `vertical_momentum_scheme="upwind_perturbation"` and
   `adaptive_implicit_vertadv=True`; NEMO's vector form is "keg + zad + vor"
   (`dynadv.F90:144`) and `ln_zad_Aimp` is `.false.`
   (`namelist_ref:1177`, not overridden by `GYRE_PISCES`).  Move both to NEMO's
   values, or leave the card as it is and mark the five tests expected-fail?
3. **Which `stp2d.F90` statement owns LOCK's external bit** — `:172` or `:185`.
   Separating them needs one more instrumented frame; that is a NEMO run
   request, not something to guess.

### ASKED / UNASKED

| choice | disposition |
|---|---|
| native GYRE card tracer integrator `euler` -> `rk3_ws` | ASKED (user decision 15A); landed; the deeper guard it exposes is reported, not silenced |
| extend the Rule-12 exact-input row to LOCK and OVERFLOW | ASKED; landed as one committed gate; CONFIRMED bit-exact on both |
| score ORCA2 as the fourth card | ASKED; BLOCKED; reported with the blocking symbols, no workaround invented |
| close guard 5's `.body` hole | ASKED; landed; shown non-vacuous against a planted bypass |
| cite the FMA fact, un-strike the bit-pattern pair, keep the census struck | ASKED; done; the stale comment quoting the census is rewritten |
| admit the LOCK acquisition and make its reader consume it | ASKED; admitted by consumed-field identity; three reader/provenance defects fixed |
| walk LOCK's external bit | ASKED; prediction REFUTED, boundary CONFIRMED, arithmetic owner left PLAUSIBLE |
| Rule-12 register boundary/owner columns, V-count correction, struck label, footgun | ASKED; done |
| OVERFLOW BBL liveness test target | ASKED; test corrected, model untouched |
| change `vertical_momentum_scheme` / `adaptive_implicit_vertadv` on the demo card | FORBIDDEN this round; raised as open question 2 |
| choose an ORCA2 branch strategy | FORBIDDEN this round; raised as open question 1 |
| per-card HPG, external or BBL switch | UNASKED/FORBIDDEN; none added |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed |

UNASKED list: empty.

Two probe worktrees were created under `/tmp` and are FLAGGED, not deleted:
`/tmp/codex-orca2-r26` (detached, ORCA2 branch, read-only) and
`/tmp/codex-gyre-r25tip` (detached at `991f9f95047d`, the OVERFLOW control).

Verdict stays **HOLD**.  Nothing is merged into the reconciled/integration
line.  Independent adversarial review of round 26 is **OUTSTANDING**, as is
round 25's.

## Round 27 — one retraction retracted, one open question retired, the fourth card scored

Round 27 starts from `1af6ba2976dc` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is `manifests/nemo_testcase_l2_gyre_round27_preregister.json`,
committed at `9aba55fc9304` before any measurement below.  No NEMO executable
was run.  Oracle roots are named per figure: GYRE `round19_oracle_v2_external`,
LOCK `nemo-testcases-l1/phase3/lock_kt1_10`, OVERFLOW
`nemo-testcases-l1/phase3/overflow_kt1_10`, the LOCK external-mode acquisition
`nemo-testcases-l2/phase3/round25_lock_external_oracle`, and ORCA2
`nemo-testcases-l4/runs/variant_icebergs_off_phase2v_tke_a_10step_np2`.

### Rule-11 record — round 26's Retraction 1 is ITSELF RETRACTED

**Dead claim (round 26).**  "The dumped kt=1 momentum-RHS frame on LOCK and
OVERFLOW is a COMPOSITE: baroclinic HPG with its vertical mean removed, plus
the barotropic acceleration added back."

**What killed it.**  That read the SHIPPED `dynspg_ts.F90`.  Both cards compile
`key_RK3`, and neither statement it names is in the object code.  The
vertical-mean removal (`dynspg_ts.F90:344-345`) is inside the `#else` at
`:303`; the barotropic add-back (`:938-975`) is inside a DIFFERENT `#else`, at
`:910` — round 26 attributed both to the one at `:303`, which is also wrong.
Both are the MLF arm.  In each card's `BLD/ppsrc/nemo/dynspg_ts.f90` there are
**zero** assignments matching `^\s*(puu|pvv)\(`.

At a rest start the frame is `dyn_hpg` ALONE, and the eligibility gate now
proves it mechanically instead of arguing it, reporting every violated fact at
once rather than the first:

| fact | where it is checked, in the code the card COMPILES |
|---|---|
| the card compiles `key_RK3` | its `cpp_*.fcm` |
| `dyn_spg_ts` writes nothing to the 3-D RHS | 0 `puu`/`pvv` assignments in ppsrc `dynspg_ts.f90` |
| `dyn_drg_init` only reads it | `BLD/ppsrc/nemo/dynspg_ts.f90:1224` declares `puu, pvv` `INTENT(in)` |
| `dyn_hpg` ASSIGNS, so nothing before `stp2d.F90:128` survives | `BLD/ppsrc/nemo/dynhpg.f90:393,412` |
| `dyn_adv_up3` with `pUe` writes only the 2-D RHS | every 3-D write at `BLD/ppsrc/nemo/dynadv_up3.f90:213,336,357` sits in the `ELSE` at `:211,317,355` |
| `dyn_ldf` is inert | `ln_dynldf_OFF = T`, `lock_kt1_10/ocean.output:615` and `overflow_kt1_10/ocean.output:727`; the gate asserts the legoESM equivalent, not the namelist |
| `dyn_vor` adds exactly `0.0` | `BLD/ppsrc/nemo/dynvor.f90:655-656` builds `zwx`/`zwy` as products with the entry velocity and `:665` adds `zuav * (zwz + zwz)`; the gate asserts NEMO's dumped entry velocity is identically zero |

Round 26's OTHER correction stands and is not disturbed: the dump is a
`stp_2D` frame (`MY_SRC/stprk3.F90:206`, after `:204` and before `:215`), not a
stage-1 frame, and its citation of `stprk3_stg.F90:309-334` was wrong.  So the
row KEEPS the name `kt1.stp2d.momentum_rhs`; what is restored is the
ISOLATION.

**Measured rows are unchanged**, which is the preregistered prediction
CONFIRMED:

| card | U row | verdict |
|---|---|---|
| LOCK_EXCHANGE-zco | `0.0`, `0 / 2540` unequal | VALUE-exact |
| OVERFLOW-zps | `0.0`, `0 / 16900` unequal | VALUE-exact |

### Rule-11 record — this round's OWN un-retraction over-claimed, twice

Independent review caught both, and both are measured, not argued.

**Dead claim 1: "bit-exact".**  The shared scorer reports `exact` from
`np.array_equal`, which is VALUE equality — `-0.0 == 0.0`.  At the bit level
the row is NOT exact:

| card | scored | bit-unequal | of which signed-zero only | after adding NEMO's zero vorticity term |
|---|---:|---:|---:|---:|
| LOCK_EXCHANGE-zco | 2540 | **1260** | 1260 | **0** |
| OVERFLOW-zps | 16900 | **16400** | 16400 | **0** |

**Dead claim 2: "dyn_hpg ALONE".**  The last column above is the mechanism,
CONFIRMED by direct test rather than inferred: `dyn_vor` accumulates
`pu_rhs + zuav*(zwz + zwz)` (`BLD/ppsrc/nemo/dynvor.f90:665`) and at a rest
start that added term is exactly `0.0`.  IEEE 754 gives `-0.0 + 0.0 = +0.0`, so
NEMO carries **0** negative zeros in this frame and legoESM carries 1260 /
16400.  `dyn_vor` is therefore VALUE-invisible and BIT-VISIBLE, and the frame
is `dyn_hpg` PLUS a zero-valued `dyn_vor` accumulation.  Round 26 was wrong
that the depth-mean removal and the barotropic add-back are in this frame —
they do not compile — and round 27's first wording was wrong in the other
direction.

The gate now publishes `value_exact_given_nemo_inputs` and
`bit_exact_given_nemo_inputs` separately, with the full bit census, so the
headline can no longer drift to the stronger of the two.  The signed-zero
population is **DEBT with a CONFIRMED owner and a named minimal fix** (carry
NEMO's zero-valued vorticity accumulation, or normalise the zero); it is NOT
landed, because it is a numerics change that moves every card and was not
asked for.

One more scope correction: `hpg_sco` writes only `ntsi..ntei` and
`jk = 1..jpkm1` (`BLD/ppsrc/nemo/dynhpg.f90:378,397`), not the whole array.
That happens to be EXACTLY the scored set — `nn_hls = 2` on both cards, so the
reader's `[2:-2]` trim is the owned interior, and it takes levels `:jpkm1` —
so the number stands, but "ASSIGNS" no longer implies a whole-array overwrite.

V stays WAIVED with its measured count (0 wet V faces on both one-wet-row
tanks).  Controls: the numeric plant exits `1` on both cards, and
`--plant-ppsrc cfgs/DINO` runs the same assertion against an MLF-compiled
configuration and fires on three checks at once — including the contrast the
whole argument rests on, that on MLF `hpg_sco` ACCUMULATES
(`puu = puu + zhpi + zuap`) where under `key_RK3` it ASSIGNS.

### Wrong citations, for the third round running — now gated

In `stp2d.F90`, line 126 is a comment, 129 is blank, 190 is a banner and 279
is blank; the calls are `:128` `dyn_hpg`, `:131` `dyn_ldf`, `:146` `dyn_vor`,
`:281` `dyn_spg_ts`.  Separately, a code comment called
`stprk3_stg.F90:168-243` "tracers"; that range is the `r3t/r3u/r3v` block and
the "Dynamic : RHS" banner, and the tracer program is `stprk3_stg.F90:453-598`.

Hand-checking has now failed three rounds in a row, so it is replaced by
`nemo_testcase_receipt_citation_gate.py`.  It walks this receipt's inline code
spans in order from a named heading, binds bare `:NNN` continuations to the
last named file (mis-binding those was itself the round-26 defect — the
`stp2d.F90` continuations were being read against `stprk3_stg.F90`), requires
every citation to appear in a committed citation-to-symbol map, and greps the
cited line range of the real file for that symbol.  An UNMAPPED citation
fails, so a new claim cannot enter the receipt uninspected.

On its first run it found four more things, all corrected: the four `stp2d`
rows above; the map's own first draft naming the wrong symbol for
`dynspg_ts.F90:870-895`; and two `ocean_pe_latlon_cgrid.py:NNNN` citations that
had rotted with the tree, one of them pointing into a DIFFERENT branch.  Repo
line numbers are not stable enough to cite, so both now name the symbol.

Its blind spot is written into its docstring: it proves the symbol is AT the
cited line, not that the prose's claim about that line is true.  Non-vacuity
runs both ways — shifting one citation by two lines fails it (exit `1`), and
so does appending an unmapped citation to a copy of the receipt.

### Rule-11 record — the BBL "open model question" is RETRACTED

**Dead claim (round 26).**  "On a zps card `ku_s == ku_d` is a partial-cell
face, so why it is marked BBL-active at all is an open model question."

**What killed it.**  The test called `bbl_static_geometry`.  The model calls
`nemo_bbl_static_geometry`, which follows `trabbl.F90:519-527`: it signs
`gdept_0(...,mbkt)` differences and leaves `mgrhu = 0` when they are equal, so
an equal-`mbkt` face is never active.  Measured on OVERFLOW-zps:

| builder | active U faces | of which `ku_s == ku_d` | stepped |
|---|---:|---:|---:|
| `bbl_static_geometry` (what the test called) | 141 | 112 | 29 |
| `nemo_bbl_static_geometry` (what the model runs) | 29 | **0** | 29 |

The 29 stepped faces are the same set in both.  The degenerate face the test
drew never existed in the model, so there was no model question; and the
`ku_s != ku_d` filter round 26 added to work around it was compensation, not a
fix — it would have hidden exactly this disagreement.  The test now builds the
NEMO geometry from the card's own three operands, asserts no active face is
degenerate, and picks the median active face with NO filter.  It passes
(`1 passed in 398.72s`).

Non-vacuity, by planting and restoring: disabling BBL in the test's ON arm as
well gives `assert np.float64(0.0) > 1e-12`, `1 failed in 381.78s`.  Restored
with `git checkout --`; `git status --porcelain` empty.

Recorded honestly rather than left to be re-discovered: the new
"no active face is degenerate" assertion CANNOT FAIL on this card.  NEMO signs
a 3-D `gdept_0`, so on a real zps mesh two columns with equal `mbkt` but
different partial-cell centroids WOULD be active; this card feeds a 1-D ladder
and `usrdef_zgr` keeps `pdept` uniform.  The assertion is a theorem here, kept
as a regression pin against a future 3-D `gdept` and as documentation of the
census.  It would bind on ORCA2.

### LOCK's external-mode bit, walked to the producing operation

Round 26 left the owner PLAUSIBLE between `stp2d.F90:172` and `:185` and said
"no dumped frame separates them".  One does not have to.

`stp2d.F90:172` is **REFUTED** from source plus one asserted precondition:
`dyn_adv_up3` zeroes `pUe`/`pVe` on entry
(`BLD/ppsrc/nemo/dynadv_up3.f90:138-141`) and every later write subtracts a
term built from `zFu`/`zFv`/`puu(:,:,:,Kbb)`, each a product with the entry
velocity; NEMO's dumped kt=1 entry velocity is exactly `0.0` on this card.  So
`Ue_rhs` after `stp2d.F90:172` is exactly `0.0`.  `dyn_drg_init`
(`stp2d.F90:196`) and wind (`stp2d.F90:199-202`) add exact zeros here.  `dynspg_ts.F90:282` then copies `Ue_rhs`
into `zu_frc` verbatim, and `:296` subtracts exactly zero because
`usrdef_hgr.F90:103-104` sets `f = 0`.

`stp2d.F90:185`'s counterpart is **CONFIRMED**, in three arms differing in one
variable,
scored against the round-25 acquisition's 19-frame record:

| arm | `slow_u` unequal / n | absolute max |
|---|---:|---:|
| `model_captured` (the production frame) | `1 / 127` | `2.168404344971009e-19` |
| `statement_model_rhs` | `1 / 127` | `2.168404344971009e-19` |
| `statement_nemo_rhs` (NEMO's own `uu(:,:,:,Krhs)` transplanted) | `1 / 127` | `2.168404344971009e-19` |
| `statement_nemo_rhs_reciprocal` | **`0 / 127`** | **`0.0`** |

The instrument is calibrated before its number is used: the reproduced
statement equals the production frame bit for bit, and NEMO's dumped
`uu(:,:,:,Krhs)` equals legoESM's momentum tendency on all wet faces, so the
transplant changes nothing — the reduction makes the bit from NEMO's OWN 3-D
RHS.

**And it walks one operation deeper.**  NEMO ends `stp2d.F90:185` with a
multiply by `r1_hu_0`, a PRECOMPUTED reciprocal; legoESM divides by the live
column sum.  `x*(1/H)` reproduces NEMO EXACTLY; `x/H` is the whole 1-ULP
residual of THIS frame.

**Scoped, after review, and the general form is REFUTED.**  Two things that
differ in general coincide at this frame, and both were measured:

- `r1_hu_0` is not `1/hu_0`.  `domain.F90:159` builds it as
  `ssumask/(hu_0 + 1 - ssumask)`.  LOCK has exactly ONE distinct wet column
  depth, `20.0`, for which `H + 1 - 1 == H` and `1/(H+1-1)` is bit-identical
  to `1/H`.  On a depth where that shift rounds, the arm as written is not
  NEMO's expression.
- NEMO weights with the REFERENCE `e3u_0` and divides by the REFERENCE
  `hu_0`; legoESM weights with LIVE thicknesses.  LOCK's kt=1 entry `ssh` is
  exactly `0.0` on every cell (measured), so they coincide bitwise here.  At
  kt >= 2 reference-versus-live is a SEPARATE and much larger difference,
  O(eta/H), which this probe does not measure.

So the association is CONFIRMED as the whole residual OF THIS FRAME and
REFUTED as a general statement about the statement's fidelity.  Nothing is
landed; the named next check is to re-run these arms at kt=2 with a fifth arm
weighted by `e3u_0` and `r1_hu_0`.

Controls: the plant adds `1.0` to the largest wet operand and the reciprocal
arm goes from bit-exact to `1 / 127` (exit `1`).  Two weaker plants were tried
first and both are recorded in the probe, because each is a lesson — the first
wet face carries an RHS of exactly `0.0` at kt=1, so a control there perturbs a
zero; and one ULP even on the largest operand (`|RHS| ~ 2.2e-3`) moves the mean
by `~2e-20`, below one ULP of the `~1.1e-3` mean.  That second number is this
probe's operand resolution and it bounds what the probe can claim.

### ORCA2 is the fourth Rule-12 card, and it is AT BAR

Round 26 called ORCA2 BLOCKED on a branch decision.  It was not blocked
numerically.  `/tmp/codex-orca2-r27-probe` is a DISPOSABLE detached checkout of
`origin/fidelity/nemo-testcases-l4-orca2-codex` with `d5a7f8169507`
cherry-picked; the ORCA2 branch itself is untouched and the worktree is FLAGGED,
not deleted.  Zero NEMO runs.

The existing ORCA2 HPG gate was deliberately not reused for scoring: it calls
`nemo_hpg_sco_literal_cgrid` directly and is therefore blind to the half of the
round-25 change that lives in the CONSUMER.  This probe imports that gate for
its record readers (reuse, not a second copy) and routes the same dumped
operands through the model's `nemo_sco` composition.  ORCA2 is not at rest, so
the LOCK/OVERFLOW frame isolation does not transfer and the operands route is
the only valid one.

| component | unequal / n | verdict |
|---|---:|---|
| `sum_u`, `zhpi_u`, `zuap_u` | `0 / 221640` | AT-BAR |
| `sum_v`, `zhpi_v`, `zuap_v` | `0 / 222048` | AT-BAR |

**Rule-12 eligibility for the round-25 change is discharged on all four cards
that execute it**: GYRE (round 25), LOCK and OVERFLOW (rounds 26 and 27), ORCA2
(here).

Two controls, because a bit-exact row is worthless if it cannot fail.  A
one-variable ablation restores the `-rho0` pressure round trip on the SAME
operands and `sum_u`/`sum_v` go to `79793` / `80268` unequal — the change is
load-bearing on ORCA2, not inert.  A planted one-ULP `gdept_z0` perturbation on
a wet, scored face moves 2 U and 4 V cells (exit `1`); the first draft planted
an arbitrary index that turned out to be masked land, so the plant target is now
derived from the mask.  The cherry-pick gives the fix a different sha, so
ancestry cannot be tested; the probe checks the two changed functions BEHAVE
instead — the pinned `gdept_z0` bit pattern and the acceleration pass-through —
which a reverted or half-applied pick would fail.

Three scope limits, two of them raised by review and now recorded next to the
row.  (i) NEMO supplies `gdept_z0`, so this row exercises the CONSUMER and the
operator, not the `_nemo_qco_gdept_z0` builder.  (ii) The records are rank 0 of
a two-rank run and the window is `[3:-3, 3:-3, :30]`, so the ORCA2 NORTH FOLD,
the cyclic seam and level 31 are all OUTSIDE it — the scored interior is, in
that sense, a larger GYRE, and the topology ORCA2 alone could test is not in
this number.  (iii) The horizontal metric is RECONSTRUCTED, `1/r1_e1u`, which
the operator then inverts back — the same association hazard the LOCK walk
turned on, so it was measured rather than assumed: the round trip is bit-exact
on all 2035 and 2070 distinct nonzero values, so it does not bite here.  That builder is
card-independent arithmetic pinned by `test_nemo_qco_gdept_z0_oracle_bit_pattern`,
and no ORCA2 record carries the `(gdept_0, 1+r3t, ssh)` triple at the stage-2
time level.

### Two report defects, and one premise of the round-26 review REFUTED

`exit_over_slow_dt_prediction` printed `1.559250241824e+290` on OVERFLOW.  Its
denominator is exactly `0.0` there and the code divided by
`np.finfo(float).tiny` rather than declining to answer.  It is `null` now with
a field naming why the ratio does not exist.  LOCK's `0.3`, which has a real
denominator, is unchanged.

The round-26 review also reported that the LOCK walk names
`overflow_kt1_10_flux_gate.json` as its `trajectory_gate`.  **That premise is
REFUTED**: the round-26 LOCK record names LOCK's own
`lock_trajectory_gate_kt10.json`, because the path was passed explicitly.  What
is real is the LATENT hole behind it — the default was OVERFLOW's report for
every case, so a LOCK run on gate defaults WOULD have stamped it.  Defaults are
per card now, and the reader refuses a report whose `case` is not this run's, so
an explicit wrong path cannot get through either.

Controlled: both cards were re-run and their reports are identical to round
26's, field by field, apart from the git sha and the two ratio keys.  Both LOCK
plants still exit `1`.

### Round-27 independent review — two reviewers, two blockers, six retractions

Two independent reviewers ran on this round: one on the MECHANISM and the
claims, one on the DIFF.  Every finding was re-measured before being accepted,
and both reviewers were right on every count that survived measurement.

**BLOCKER 1 — the citation gate could not catch the defect it exists for.**
It joined every cited line into one string and asked `symbol in text`, so a
RANGE passed whenever any line inside it held the symbol.  Measured over the
whole map: **36 of 78** citations still passed with a WRONG line number within
±6, several at every shift tried.  "Hand-checking is replaced" was false as
shipped.

Both endpoints are pinned now — the first and last cited line each carry their
own symbol — and `audit_shift_sensitivity()` runs on every invocation and
fails the gate for any entry that survives a ±1 or ±2 shift.  That is
self-enforcing: a symbol too generic to identify its line cannot remain in the
map.  The audit drove the count 36 → 1 → 0.  The last holdout was
`domzgr_substitute.h90:145`, whose text also appears at `:143`, the
`key_linssh` variant.

**BLOCKER 2 — the ORCA2 probe re-derived the model's arm rather than running
its shape.**  It called the operator with `return_components=True` where the
model does not, and skipped the model's `astype` cast.  The scored pair now
comes from the production call shape; the components call is a diagnostic
whose first two elements must equal the production pair BIT for bit, which is
the calibration the LOCK walk had and this one lacked.  And the ablation is
now REQUIRED, not merely reported — a round trip that changed nothing would
have left an AT-BAR row standing as evidence for a change that is inert.

**Four claim-strength retractions**, in addition to the two already recorded
above:

| claim as written | what measurement said |
|---|---|
| "the isolation is proved mechanically, none is prose" | it omitted `stp2d.F90`'s other advection arms — `dyn_keg` `:163` + `dyn_zad` `:165` and `dyn_adv_cen2` `:169` DO write the 3-D RHS. `ln_dynadv_up3` and `ln_hpg_sco` are now asserted from the card's namelist |
| the DINO plant "fires on every fact at once" | it fires on THREE of five; `dyn_drg_init`'s INTENT and the `dyn_adv_up3` guard pass on that build and have no plant |
| "four unconditional `Ue_rhs` writers" | there are SIX assignments; `:207`/`:223`/`:235` are guarded by `ln_apr_dyn` / `ln_ice_embd` / `ln_bern_srfc` and `:180` is the unselected vector-form arm. All are now asserted off from the run's own `ocean.output` and namelists |
| the +1.0 plant is the LOCK walk's control | it proves the probe READS its operand, not that it resolves the `2.168e-19` effect — one ULP moves the mean by `~2e-20`, below its resolution. The ARM SEPARATION is what demonstrates that, and is reported as such |

**Two reviewer findings not adopted, with the reason.**  The `hpg_sco` ASSIGNS
regex inspects only the first right-hand-side token, and the `IF`/`ELSE`
tracker models neither `ELSEIF` nor a one-line `IF` carrying a write.  Both
are real gaps; neither construct occurs in the routines walked (`grep -E
"ELSE ?IF" dynadv_up3.f90` is empty).  Recorded as known blind spots rather
than fixed on a hypothetical.

Two defects this round found in its OWN work while acting on the review.  A
new test enabled float64 only if the caller had exported `JAX_ENABLE_X64`, so
it passed standalone and failed inside a combined run — Rule 1c's exact
failure mode; it sets x64 itself now.  And the strengthened citation gate
immediately caught a wrong citation introduced by this very edit: the review
quoted `dyn_keg`/`dyn_zad`/`dyn_adv_cen2` at LOCK's PREPROCESSED line numbers,
which are offset by two from the shipped source, and those numbers went
straight into the receipt.  The gate refused them on the first run.  That is
the fourth wrong-citation event in three rounds and the first one caught by a
machine instead of a person.

### Tests, and the ratchets

The focused suite over the two modules this round touches plus its three new
test files:

```text
5 failed, 70 passed in 1477.04s (0:24:37)
```

The five are `tests/ocean/unit/test_nemo_recipe.py::test_nemo_gyre_*` plus
`test_surface_stress_implicit_wiring` — the SAME pre-existing set, needing the
configuration decision in open question 2.  The OVERFLOW BBL liveness test is
green, and so are the three new test files.

The constant / dispatch / private-import / inline-coefficient ratchets give the
IDENTICAL 9 failures at the round-27 starting tip `1af6ba2976dc` as at this tip,
so none of them is round 27's; this tree passes 6 more tests than that one and
fails none extra.

### Merge readiness

`fidelity/nemo-testcases-l2-gyre-codex2` is **78 commits ahead of
`03c6e8d96ff7`**, which is the tip of BOTH `origin/fidelity/nemo-gyre-integration-merge`
and `origin/fidelity/nemo-testcases-l2-gyre-reconciled`, and is an ancestor of
this branch — so the integration is a FAST-FORWARD with zero conflicts by
construction, not a merge to be resolved.

Nothing in round 27 blocks that fast-forward.  What is stated honestly next to
it: this round landed no model numerics at all — the only change under
`packages/` is a one-line comment, verified by `git diff` — and the
pre-existing focused-test failures are unchanged in kind.  Independent
adversarial review of round 27 HAS now run — two reviewers, two blockers and
six retractions, all folded in above and all re-measured before acceptance.

*(Corrected in round 28: the two sentences that stood here said rounds 25 and
26 were unreviewed.  They were not.  Round 25 was independently reviewed and
round 26 acted on that review; round 26 was reviewed in turn — its own section
records two reviewers and four retractions — and so was round 27.  The commit
count "74" above was also wrong; measured at this tip it is 78.)*  The merge
decision itself is not made here.

### Open questions

1. **The barotropic slow-forcing weighting and association.**  Two nested
   questions, and review showed the smaller one cannot be answered alone.  The
   `x/H` versus `x*(1/H)` association is the whole residual of LOCK's kt=1
   frame; but NEMO weights that sum with REFERENCE `e3u_0`/`hu_0` where
   legoESM uses LIVE thicknesses, a difference that is exactly zero at kt=1
   (eta ≡ 0) and O(eta/H) afterwards.  Landing only the re-association would
   leave the larger defect.  Measure the kt=2 arms first, or land both
   together, or leave the row as named debt?
2. **The signed-zero population in the kt=1 momentum-RHS frame.**  1260 / 2540
   on LOCK and 16400 / 16900 on OVERFLOW, DEBT, CONFIRMED owner (NEMO's
   zero-valued `dyn_vor` accumulation normalises `-0.0` to `+0.0`; legoESM has
   no such addition).  Carry the zero, normalise it, or accept value-exactness
   as the bar for this frame?
2. **The native demo GYRE card's momentum program.**  Unchanged this round per
   the standing decision; the five `test_nemo_recipe` failures stay attributed
   to it.
4. **The ORCA2 branch strategy.**  The probe answers the NUMERICAL question
   from a disposable worktree, so round 26's open question 1 is no longer
   blocking a measurement — but the two branches are still divergent and that
   is a decision, not a finding.
5. ~~**Independent adversarial review of rounds 25 and 26** — OUTSTANDING~~
   — RETIRED in round 28: rounds 25, 26 and 27 have each been independently
   reviewed, and each review is recorded in its own section.

### ASKED / UNASKED

| choice | disposition |
|---|---|
| un-retract the `dyn_hpg`-alone frame, and prove it from the ppsrc | ASKED; landed; rows unchanged, plant fires on an MLF build |
| keep round 26's row NAME while restoring its isolation | ASKED; the dump really is a `stp_2D` frame, so the name stays |
| correct the `stp2d` and `stprk3_stg` citations | not a scientific choice; corrected against the shipped source and gated |
| add a citation gate with a committed symbol map | ASKED; landed, non-vacuous both ways |
| point the BBL liveness test at `nemo_bbl_static_geometry`, drop the filter | ASKED; landed; the "open model question" is retracted |
| separate LOCK's `stp2d.F90:172` / `stp2d.F90:185` owner offline | ASKED; `:172` REFUTED, `:185` CONFIRMED for this frame, walked to the divide-vs-reciprocal operation and then SCOPED |
| **land** the reciprocal association | NOT DONE — open question 1; it moves every card and was not asked for |
| score ORCA2 from a disposable probe worktree | ASKED; AT-BAR on all six components; ORCA2 branch untouched |
| null the undefined ratio; per-card trajectory provenance | ASKED; landed; both cards otherwise byte-identical to round 26 |
| change the demo GYRE card's `vertical_momentum_scheme` / `adaptive_implicit_vertadv` | FORBIDDEN this round; untouched |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed |

UNASKED list: empty.

Two probe worktrees are FLAGGED, not deleted: `/tmp/codex-orca2-r27-probe`
(this round's, detached, cherry-picked) and round 26's
`/tmp/codex-orca2-r26` and `/tmp/codex-gyre-r25tip`.

Every figure above is pinned in `manifests/nemo_testcase_l2_gyre_round27.json`;
the reports live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round27/` with SHA-256 in
`artifacts.sha256`.

Verdict stays **HOLD** on merging.  Nothing was merged or pushed.

## Round 28 — the GYRE merge blocker is REFUTED, and the gates learn to fail

Round 28 starts from `359c33c40ecc` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round28_preregister.json`, committed at
`2f292b24e110` before any measurement below.  No NEMO executable was run and
no configuration choice was made.  Oracle roots are the round-27 ones,
unchanged.

### Rule 0 first — GYRE does not execute the statement round 27 named

`stp2d.F90:177` opens a `SELECT CASE( n_dynadv )` with two arms.  `:178` is
`CASE( np_VEC_c2, np_LIN_dyn )` and its body `:180` ASSIGNS the depth mean;
`:183` is `CASE ( np_FLX_c2, np_FLX_up3 )` and its body `:185` CUMULATES it
onto the 2-D advective RHS.  GYRE's own deck sets `ln_dynadv_vec = .true.`
(`GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:161`), so GYRE executes
`stp2d.F90:180`; LOCK and OVERFLOW set `ln_dynadv_up3 = .true.` and execute
`stp2d.F90:185`.  **Corrected in round 29:** this sentence originally named
GYRE's deck by an unqualified basename, which the citation gate resolves to
LOCK's namelist -- where that switch is `.false.`, the opposite of the claim.
The premise was right and the citation was not auditable; GYRE's deck is now
a separate key in the gate's file map.  Round 27's open question named
`stp2d.F90:185` for GYRE.  Both arms weight with the reference `e3u_0` and multiply by
the reference `r1_hu_0` (`domain.F90:159`), so the question survives the
correction and only the line moves.

Two more readings bound what this round can claim, and both were taken before
the manifest was written.  `stprk3.F90:186` calls `stp_2D` exactly ONCE per
timestep, from the `Nbb` entry, so the state entering kt=2 is produced by a
step whose slow forcing used the kt=1 entry state.  And the GYRE oracle V2
record set holds `oracle_bt_substeps_kt00000001.bin` only, with a reader that
requires `kt == 1` — there is no dumped GYRE slow-forcing frame at kt=2, so a
kt=2 arm cannot be SCORED against the oracle.  What is measurable at kt=2 is
an arm-to-arm difference, and it is labelled as one everywhere below.

### The blocker, measured

| what | measured |
|---|---|
| GYRE kt=1 entry `ssh` / `u` / `v` | `0.0` / `0.0` / `0.0` — at rest |
| GYRE kt=1 `slow_u` vs NEMO | `0 / 580` unequal, **`0` BIT-unequal** |
| GYRE kt=1 `slow_v` vs NEMO | `0 / 570` unequal, **`0` BIT-unequal** |
| GYRE kt=1 external mode, 18 frames x 50 substeps | `900` rows, first-over-bar **NONE** |
| first BIT-unequal frame of those 900 | substep 7 `trd_u`, `1 / 580` cells, `5.048709793414476e-29` |
| kt=2 reference minus live, relative | `8.307570867102084e-16` (U), `5.993770496385322e-16` (V) |
| kt=2 max \|eta\| / H | `6.622180591839445e-07` |

So the reference-versus-live weighting is **roundoff — about 4 ulp — and NOT
`O(eta/H)`**.  Round 27 asserted the `O(eta/H)` sizing without measuring it;
that assertion is RETRACTED and replaced by the two numbers above, which sit
nine decades apart.

It cannot own GYRE's kt2 rows for a second, independent reason that needs no
sizing argument at all: `stp_2D` runs once per step from the kt=1 entry, where
`ssh` is exactly `0.0`, so during the step that PRODUCES the kt=2 state the
live and reference thicknesses are the same numbers.  The difference is
identically zero there.

**The GYRE merge blocker as round 27 posed it is REFUTED.**  Nothing is
landed: the reference weighting is bitwise identical to the live one on every
frame the GYRE records can score, so there is no measured basis for a model
change.

The reference arm is not merely a reconstruction.  NEMO dumps its own
operands in `oracle_slow_forcing_kt00000001.bin`, and this probe's mesh-built
`e3u_0`, `umask` and `r1_hu_0` are **`0` bit-unequal** against them on both
faces — which is also the check that NEMO's LIVE `e3u` equals its own
reference `e3u_0` bitwise at this zero-`ssh` entry.

### The next operand, named and sized — MIN is not MEAN

The reference-versus-live question is not the only thickness question, and the
other one is NOT roundoff.  NEMO's live face stretch is an AREA-WEIGHTED MEAN
of the two neighbouring `ssh` (`domqco.F90:166-169`), and `stp2d.F90:200`
divides the wind stress by that mean-rule depth; legoESM builds the face
thickness with a MIN rule and sums it.

| where | max relative \| MIN − NEMO's MEAN \| | bit-unequal |
|---|---|---|
| kt=1, `eta == 0` | `0.0` | `0 / 580` |
| kt=2, `eta != 0` | `3.058607589676341e-08` | `580 / 580` |

That is `0.046 x (eta/H)` — first order, and eight decades above the
reference-versus-live residual.

~~It is also exactly zero during step 1, so it does not own GYRE's kt2 rows
either; it enters from kt=3.~~ **STRUCK IN ROUND 29, and struck in place
rather than edited away.**  Two things are wrong with it.  The stage face
ratio is NOT zero during step 1: `r3u(:,:,Kaa)` is assigned at STAGE 1 from
`ssha` (`stprk3_stg.F90:156`, the assignment at `:163`), so a stage-3
thickness already carries a nonzero free-surface ratio inside the step that
produces the kt=2 state.  And the MIN rule is not what the production stage
builder uses: `nemo_qco_live_face_geometry_cgrid` (`vertical.py:772`)
transcribes NEMO's area-weighted MEAN (`domqco.F90:166-169` for the MLF
entry, `:219-222` for the RK3 one) and both time-stepping lanes call it, so
MIN survives only as an ablation arm.  What the measured `3.06e-08` actually
sizes is the SLOW-FORCING depth average, where `min_cell_to_uface` is still
the rule.  That is the debt, it is registered with this size and this source
line, and it is not a statement about the RK3 stages.

### Where GYRE's kt2 divergence is, and where it is not

One step from NEMO's own kt=1 entry reproduces the round-23 register rather
than a private number — two instruments converging:

| field | this round | round-23 register (normalized) |
|---|---|---|
| u | `17400/17400` at `9.484089938411461e-07` | `9.48409e-07` |
| v | `17100/17100` at `8.987992610401277e-07` | `8.98799e-07` |
| T | `11840/18000` at `3.1956659540810506e-11` | `1.36147e-12` |
| S | `11229/18000` at `8.171241461241152e-13` | `2.21811e-14` |
| ssh | `0/600` at `0.0` — bit-exact | `4.33681e-19` |

`score` normalizes by `max(|oracle|, 1.0)`, so the T and S columns agree once
divided by their own scales (`23.47` and `36.84`); u and v need no
normalization and match to ten figures.  The ssh row differs because the two
paths seed differently — this round seeds NEMO's kt=1 entry, the register
steps the card's own trajectory — and both are at the bar.  The u residual is
`1.586e-05` RELATIVE to that frame's own `max|u| = 5.98e-02`.

Everything NEMO computes before the RK3 stages is at the bar on GYRE, ssh
after one step is bit-exact, and yet every U and V cell is wrong at ~1e-5
relative.  The divergence is produced AFTER the external mode, inside
`stp_RK3_stg`.  Recorded honestly, because a reviewer raised it: "inside the
stages" and "a thickness convention" are NOT alternatives, since NEMO
recomputes `r3u` per stage — the MIN-versus-MEAN row above is a live candidate
INSIDE the stages, not a competitor to them.

### Rule-11 records

**Dead claim 1 (round 27).** "Reference-versus-live is a SEPARATE and much
larger difference at kt >= 2, `O(eta/H)`."  Measured: `8.31e-16` relative
against `eta/H = 6.62e-07`.  It is roundoff.

**Dead claim 2 (this round's own, retracted twice).**  A draft published
`5.256e-4` as NEMO's kt=1 momentum-RHS maximum and used it to retract the
statement that the frame is zero.  That number is UNMASKED: measured on the
wet faces the frame is exactly `0.0` on all 17400 U and all 17100 V faces, and
its 1200 nonzero cells are all on LAND.  So the original statement was right,
the retraction was the error, and the retraction is retracted.  The
consequence is structural: every kt=1 arm of the depth-mean statement is
`0 == 0`, so the kt=1 "0 of 580 bit-unequal" that a draft reported as a
control was PERTURBING A ZERO.  That block is deleted, not relabelled, and the
arms run at kt=2 only.

**Dead claim 3 (this round's own).** "The whole kt=1 external-mode trace, 800
rows, none over bar."  The record is format 2 and carries 20 frames; the draft
copied the phase-3 gate's 16-name list and silently omitted `cor_u`/`cor_v` —
the separated barotropic Coriolis — and `transport_metric_u`/`_v`.  Scored
now: `cor_u`/`cor_v` are IN and at the bar, which is a real gain; the two
`transport_metric` frames are OUT, carrying the phase-3 gate's existing
UNMEASURED waiver ("oracle stores e2u/e1v metric transport; no independent
metric operand was dumped").  A draft scored them anyway and reported
`transport_metric_u` as the first frame over bar at `2639.4` — a Rule-2
cross-quantity comparison, not a defect.  The corrected figure is 900 rows
over 18 comparable frames, first-over-bar NONE.

**Dead claim 4 (this round's own).** The docstring claimed an arm-equality
check against the captured production frame, copied from the LOCK probe.  No
such check exists here and none can: legoESM's `F_slow` is the depth mean PLUS
wind, drag and biharmonic increments, so the sub-statement is not that frame.
The preregistered `statement_model_rhs` calibration arm is therefore DROPPED,
with that reason, rather than reported.

**Scope correction, not a retraction.** The cancellation mechanism is written
for the `dz_ref * (1 + eta/H)` form.  GYRE's card resolves an
`OceanPartialCellCoordinate`, whose thickness builder takes a partial-cell
branch; the factorisation survives only while the min-rule's argmin does not
switch with depth.  On GYRE it does not — `h_u(k)/e3u_0(k)` is constant to one
ulp — but on stepped bathymetry it can, and ORCA2 is zps.  UNMEASURED there.

### Both gates learn to fail

**The citation gate was still defeatable, and the defeats were re-run first.**
Round 27 pinned both endpoints of a range but audited them by shifting every
line TOGETHER, so a changed range EXTENT was never tested; terminal tokens
recur so densely that both ends of a widened range still matched.  Measured on
the shipped gate: `stprk3_stg.F90:309-334` still passed with its end moved to
line 533, because line 334 and line 533 are both `ENDIF`.  And a range written
backwards, 309 down to 300, returned an empty list, after which `check` raised
`IndexError` — a malformed citation had no verdict at all.

Three guards replace one.  An endpoint anchor must now IDENTIFY its line — a
symbol unique in the file, or `(symbol, nth)` naming which occurrence is meant
— so no line-number error of any size passes; every multi-line citation states
its LENGTH a second time; and a reversed or empty range raises cleanly.  31 of
the 86 map entries were anchored on a symbol occurring 2 to 59 times and are
now pinned by occurrence.

| defeat re-tried | verdict |
|---|---|
| `stprk3_stg.F90:309-334` widened to end at line 533 | EXTENT-MISMATCH |
| same, with the pinned extent widened to match | SYMBOL-NOT-AT-LINE |
| the same citation reversed, 309 down to 300 | BAD-CITATION |
| five terminal-token ranges widened | EXTENT-MISMATCH, all five |
| uniform `+2` shift, and each endpoint shifted alone | SYMBOL-NOT-AT-LINE |
| a bare `ENDIF` as an anchor | AMBIGUOUS-ANCHOR |
| an unmapped citation appended to a copy of the receipt | FAIL, exit `1` |

The shift audit is RETIRED rather than kept: once an anchor resolves to one
line, no shift can pass, so a shift sweep could never report anything and
would have been decoration.  What replaces it is an audit over the WHOLE map —
entries this round's prose did not cite are checked too — plus a self-test
that plants five defects (bare terminal anchor, widened extent with the extent
widened to match, reversed range, and each endpoint shifted on its OWN) and
fails the gate unless every one fires AND the unplanted baseline still passes.

**A hole this round's own gate still has, measured not guessed.**  A citation
written as a comma list pins only its OUTER two members, so every INTERIOR one
is unchecked.  Measured against the shipped map entry for
`BLD/ppsrc/nemo/dynadv_up3.f90:211,317,355`: moving the middle member to 320
returns `OK`, and so does moving it to 999.  That is exactly the defect class
this gate exists to catch, one level down from the range-extent case it now
closes.  It is NOT fixed here — a concurrent session is editing this file to
group and pin every endpoint, and colliding with an in-flight edit would be
worse than the hole.  Recorded so the next round inherits the measurement
rather than the surprise.

CI now runs it.  Round 27 shipped the gate but nothing in the suite called
`run` on the receipt, so a bad citation only failed if somebody remembered to
run the script.  Two tests do it now — one asserts the real receipt is clean,
one plants a shifted citation and asserts the same call turns red.  Coverage
is stated instead of implied: **rounds 25 onward are audited; rounds 1-24 are
NOT**, and the gate's own report carries that in a field.  16 tests pass.

**The bit bar is in an exit code.**  `nemo_testcase_rule12_hpg_eligibility.py`
published `bit_exact_given_nemo_inputs: false` while exiting `0`, so a caller
that checked only the exit status read a bit-DEBT row as at the bar.  The
value-exact-but-bit-unequal case now has its own status, and only bit equality
exits `0`.

| card | status | bit-unequal | exit |
|---|---|---:|---:|
| LOCK_EXCHANGE-zco | VALUE-AT-BAR | `1260 / 2540` | `1` |
| OVERFLOW-zps | VALUE-AT-BAR | `16400 / 16900` | `1` |
| LOCK, planted | DEBT | — | `1` |

The receipt sentence, restated: the round-25 HPG change is **value-exact on
LOCK and OVERFLOW, bit-exact on ORCA2, and on GYRE its eligibility frame is
UNINFORMATIVE** — NEMO's kt=1 momentum RHS is exactly zero on every wet face
there, so that row compares zeros with zeros and cannot discriminate.  The
signed-zero population itself is untouched: landing it is user decision 16.

### One shared implementation, and a controlled re-run

The GYRE probe first carried its own copy of the depth-mean statement.  That
is duplicated numerics, which this repo forbids, so the LOCK probe's
`depth_mean_statement` became the one implementation — extended with a face
selector and the grid the V-face helper needs — and GYRE imports it.  LOCK was
re-run afterwards and every row is unchanged: `1 / 127` at
`2.168404344971009e-19` on three arms, `0 / 127` at `0.0` on the reciprocal
arm, instrument-reproduces-model `true`.

### Tests, and which failure was actually this round's

The focused suite over the two gate modules this round touches: **23 passed**.
The whole `tests/ocean/fidelity/` directory was then run, because a signature
change to a SHARED helper is a change to every caller:

```text
3 failed, 764 passed, 7 skipped, 18 deselected in 2417.17s (0:40:17)
```

One of the three was this round's, and it is the interesting one.  Making
`depth_mean_statement` the single shared implementation added a `grid`
argument between the config and the face mask, and the LOCK probe's
dispatch-guard test called it POSITIONALLY.  The unknown-association
`ValueError` it asserts was therefore never reached — the call died earlier
with a `TypeError`, so a dispatch-hardening test had silently stopped testing
dispatch hardening while still being counted as one.  The call goes through a
named helper now, so the next signature change fails loudly in one place; and
the face selector this round introduced gets the same guard and its own test,
because a silent fall-through there would score a V face with the U operator,
which is a wrong ANSWER rather than an error.  Plant-verified: replacing the
face guard with `pass` turns the new test red, and `git checkout --` restores
it with `git status --porcelain` empty.

The other two are NOT called pre-existing on inspection — they were re-run at
round 28's starting tip `359c33c40ecc` in a disposable worktree, which is what
this campaign requires of any new-looking failure.  Both reproduce there:

```text
2 failed in 507.03s (0:08:27)      # at 359c33c40ecc, this round's changes absent
FAILED tests/ocean/fidelity/test_recipe_case_board.py::test_every_oracle_comparison_has_a_row
FAILED tests/ocean/fidelity/test_nemo_testcase_phase3_stage_sweep_gate.py::test_planted_stage_control_exits_nonzero_end_to_end
```

| failure | disposition |
|---|---|
| `test_recipe_case_board::test_every_oracle_comparison_has_a_row` | **PRE-EXISTING**, reproduced failing at the starting tip.  It wants board rows for `advection_nemo`, `grids_tripole_mpas`, `tendencies_nemo` and `three_way_nemo`; this round adds no `compare_*.py` driver at all |
| `test_nemo_testcase_phase3_stage_sweep_gate::test_planted_stage_control_exits_nonzero_end_to_end` | **PRE-EXISTING**, reproduced failing at the starting tip.  Corroborating and independent: this round modifies eight files and that gate is not one of them — the probes only import `GateError`/`git_sha`/`require`/`sha256` from it |

So round 28's own count is **one failure, found and fixed**, and the tree ends
the round failing exactly the tests it started it failing.  The disposable
worktree `/tmp/codex-gyre-r28-basecheck` (detached at `359c33c40ecc`) is
FLAGGED, not deleted.

### Provenance repair, and a concurrent session in this worktree

Every round-28 figure was first produced with `--allow-dirty`, because the
probes were being iterated while they ran.  `git_sha` fails closed on tracked
dirt for exactly this reason, and passing that flag defeats it — so each
report stamped `<sha>-dirty` and none of the numbers was tied to a committed
tree.  That is a provenance hole whatever the numbers turn out to be.

All four gates were therefore re-run from a clean detached checkout of
`aab15e1fb7c5` with the flag REMOVED, so `git_sha` had to stamp a clean
revision.  Every report is byte-identical to its dirty-run predecessor apart
from the stamp:

| report | clean stamp | identical modulo the sha |
|---|---|---|
| `rule12_eligibility_LOCK_EXCHANGE_zco` | `aab15e1fb7c5` | yes |
| `rule12_eligibility_OVERFLOW_zps` | `aab15e1fb7c5` | yes |
| `lock_slow_forcing_owner_reshared` | `aab15e1fb7c5` | yes |
| `gyre_slow_forcing_weighting` | `aab15e1fb7c5` | yes |

The clean copies live under `round28/clean_sha_rerun/` and are the citable
ones; the dirty originals are kept beside them rather than deleted, so the
comparison stays checkable.  Exit codes on the clean run are unchanged too —
`1`, `1` for the two eligibility cards and `0`, `0` for the two probes.

**A concurrent session is working in this same worktree and on this same
branch**, and that is recorded here because it bounds what a later reader may
assume.  It committed `7d7cbfb97319` (a round-29 preregistration) between two
round-28 commits, and it has UNCOMMITTED edits to
`packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py` and
`tests/ocean/unit/test_nemo_recipe.py` plus an untracked round-29 script.
None of those is round 28's and none was touched or staged — every commit here
used explicit pathspecs.  Two consequences: `/tmp/codex-gyre` is no longer a
clean tree, so any gate re-run there will stamp dirty or refuse; and the
statement that this round landed no model numerics is a statement about the
COMMIT RANGE (`git diff --name-only 359c33c40ecc..HEAD` names zero files under
`packages/` or `src/`, verified), not about the working tree as a later reader
may find it.

### Card debt registered, card NOT changed

The LOCK and OVERFLOW cards resolve `vorticity_scheme = "al81"` from a module
default.  NEMO's LOCK card resolves `ln_dynvor_ens = .true.`
(`namelist_cfg:91`, printed at `lock_kt1_10/ocean.output:715`) — the
enstrophy-conserving scheme, which is not the same operator.  Measured inert
at the kt=1 rest start: the vorticity term adds exactly `0.0` there, and the
eligibility gate asserts it.  Whether it is inert at kt >= 2 is **UNMEASURED**,
and LOCK's first-over-bar is kt4 `u`, so it is a live candidate there.  The
card is NOT changed — that is a user decision, and it is open question 3.

### Merge readiness

`fidelity/nemo-testcases-l2-gyre-codex2` was **78 commits ahead of**
`03c6e8d96ff7` at round 28's starting tip `359c33c40ecc`, and round 28 adds
its own commits on top of that — stated this way because a count "at HEAD"
changes the moment the sentence recording it is committed, which is how round
27's "74" went stale.  `03c6e8d96ff7` is the tip of BOTH
`origin/fidelity/nemo-gyre-integration-merge` and
`origin/fidelity/nemo-testcases-l2-gyre-reconciled`, and is an ANCESTOR of
this branch (`git merge-base --is-ancestor` returns true).  The integration is
therefore a FAST-FORWARD with zero conflicts by construction.

What blocks that fast-forward after this round: **nothing mechanical**.  Round
28's OWN eleven commits landed no model numerics — verified by listing the
files every one of them touches, nine in total, all under `scripts/validate/`,
`tests/` and `docs/`, zero under `packages/` or `src/` — and the round removed
one merge blocker by measurement rather than by decision.

**The branch range is no longer round 28's alone, and the earlier wording of
this paragraph was wrong about it.**  A concurrent session committed three
further changes onto this same branch while round 28 was running:
`7d7cbfb97319` (a round-29 preregistration), `1a019b2fe85e`, which sets the
demo GYRE card's `vertical_momentum_scheme` and `adaptive_implicit_vertadv`
and cites "User decision 15C (2026-09-05)" for doing so, and `8091542726b3`
(round-29 `dyn_zdf` instrumentation).  So `359c33c40ecc..HEAD` now DOES touch
`packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py`.  Round 28 neither made
nor verified that decision — it was FORBIDDEN for this round and is untouched
by its commits — and it is flagged here rather than reverted, because
reverting another session's committed work on a cited user decision is not
round 28's call.  A reader diffing this branch should attribute those three
commits to round 29, not to this section.  Two gates that previously exited `0`
on a defect now exit non-zero, which is a strictly tighter tree, and the
pre-existing focused-test failures are unchanged in kind.

What is honestly still open next to it, with numbers:

| card | first-over-bar at HEAD | detail |
|---|---|---|
| GYRE-zco | kt2 T/S/u/v | u `17400/17400` at `9.484e-07` (`1.586e-05` relative); v `17100/17100` at `8.988e-07`; T `1.361e-12` and S `2.218e-14` normalized; ssh AT-BAR |
| LOCK_EXCHANGE-zco | kt4 `u` | later than lane 1's certified kt2; seven Rule-12 rows registered in round 24 |
| OVERFLOW-zps | seven Rule-12 rows | localized in round 23 |
| ORCA2 | AT-BAR on all six HPG components | round 27, from a disposable probe worktree |

Both remaining thickness conventions are now measured and neither owns GYRE's
kt2 rows.  The merge decision itself is not made here.

### Open questions

1. **The MIN-versus-MEAN face-thickness rule, restated in round 29.**
   `3.06e-08` relative at kt=2, first order in `eta/H`, source-cited.  The
   question is narrower than round 28 posed it: the RK3 stage builder ALREADY
   uses NEMO's mean rule, so what is left is the barotropic slow forcing's
   depth average.  Land the mean rule there too, or leave it as named debt?
2. **The signed-zero population in the kt=1 momentum-RHS frame** — unchanged
   from round 27: `1260 / 2540` on LOCK, `16400 / 16900` on OVERFLOW, DEBT
   with a CONFIRMED owner.  User decision 16, still pending; not landed.
3. **The LOCK/OVERFLOW cards' `vorticity_scheme`** — `al81` by module default
   against NEMO's `ln_dynvor_ens`.  Inert at the rest start, UNMEASURED at
   kt >= 2, and LOCK's first-over-bar is kt4 `u`.  Measure it, or change the
   card?  Card untouched this round.
4. **The native demo GYRE card's momentum program** — unchanged per the
   standing decision (15C); the five `test_nemo_recipe` failures stay
   attributed to it.
5. **The partial-cell scope limit** — the depth-mean cancellation is measured
   on GYRE (flat-bottom, argmin constant to one ulp) and UNMEASURED on a
   stepped-bathymetry card.  Re-run the kt=2 sizing on ORCA2?
6. **Rounds 1-24 of this receipt are UNAUDITED by the citation gate.**  Extend
   the map backwards, or leave the statement standing?
7. **A defect found in round 16's probe, reported and NOT fixed here.**  Its
   "literal left-to-right transcription of the scalar-math Fortran SUM" loops
   `range(1, product.shape[-1] - 1)` over an array the reader has already
   trimmed to `jpkm1` levels.  Measured by calling it on a ones-array of 30
   levels: it returns `29.0`, so it accumulates 29 of the 30 and DROPS the
   deepest one.  NEMO's `stp2d.F90:180` sums `1:jpkm1`, all 30.  Any
   round-16 figure that rests on that helper is suspect.  Fixing it means
   re-running and re-pinning round-16's recorded numbers, which is a decision,
   not a patch — so it is registered here and left alone.

### ASKED / UNASKED

| choice | disposition |
|---|---|
| correct GYRE's executed statement from `stp2d.F90:185` to `:180` | not a scientific choice; corrected against the shipped source and gated |
| measure the reference-versus-live weighting instead of asserting its size | ASKED; REFUTED as `O(eta/H)`, measured at `8.31e-16` |
| **land** the reference weighting | NOT DONE — bitwise identical to the live arm on every scorable GYRE frame, so there is no measured basis |
| score the barotropic Coriolis frames that were being skipped | ASKED; landed; both are at the bar |
| carry the phase-3 gate's `transport_metric` waiver rather than re-judge it | ASKED; the draft's `2639.4` "first over bar" is retracted as a cross-quantity comparison |
| make `depth_mean_statement` one shared implementation | ASKED; landed; LOCK re-run, every row unchanged |
| give the eligibility gate a status the exit code reflects | ASKED; landed; both cards now exit `1` |
| re-anchor 31 citation-map entries by occurrence and pin every range length | not a scientific choice; every named defeat re-tried and closed |
| state rounds 1-24 as unaudited rather than extend the map | ASKED; stated in the gate's report and in this section |
| land the signed-zero `dyn_vor` accumulation | FORBIDDEN this round; untouched |
| change the demo card's `vertical_momentum_scheme` / `adaptive_implicit_vertadv` | FORBIDDEN this round; untouched |
| change the LOCK card's `vorticity_scheme` | FORBIDDEN; registered as debt instead |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed |

UNASKED list: empty.

Independent adversarial review of round 28 ran on the mechanism and on the
diff, two reviewers.  Both found real defects, every one was re-measured
before acceptance, and the four Rule-11 records above are theirs: the unmasked
precondition, the zero-perturbing kt=1 control, the 16-of-20 trace, and the
docstring's phantom calibration check.  Three further findings are folded in
as scope limits (partial cells, the normalized-versus-bit bar, and stage-wise
`r3u`), and one — that NEMO had DUMPED the operands this probe was
reconstructing — became the calibration that now backs the reference arm.

Every figure above is pinned in `manifests/nemo_testcase_l2_gyre_round28.json`;
the reports live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round28/` with SHA-256 in
`artifacts.sha256`.

Verdict stays **HOLD** on merging.  Nothing was merged or pushed.

## Round 29 — the stage-3 owner, an instrument for it, and a corrected premise

Round 29 starts from `509aed683857` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round29_preregister.json`, committed at
`7d7cbfb97319` before any measurement below.  No NEMO executable was run.

### Rule 0 first — the premise this round was handed is wrong

Round 29 was asked on the premise that `dyn_zdf` is the only stage-3-only
momentum operator touching every wet U face.  It is not.  The stage-3 arm of
the stage `SELECT CASE` calls `dyn_ldf` first
(`stprk3_stg.F90:400`, compiled
`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3_stg.f90:485`) and `dyn_zdf` after
(`stprk3_stg.F90:430`, compiled
`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3_stg.f90:498`), and neither runs at
stage 1 or 2.
GYRE resolves an iso-level laplacian viscosity for the first of them —
`ln_dynldf_lap` and `ln_dynldf_lev` are both true
(`GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:187-188`) with a constant
coefficient built from `nn_ahm_ijk_t = 0`
(`GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:189-191`), printed as
`iso-level laplacian operator` at
`round19_oracle_v2_external/ocean.output:706`.  So the correction changes
what the instrument has to dump: a matrix record alone could never separate
the two, and a frame at the boundary between them can.

Four more readings, all from the oracle's own printed output rather than
inferred, bound what `dyn_zdf` even does on this deck:

| contribution | resolved | consequence |
|---|---|---|
| `ln_zad_Aimp` | `F` (`round19_oracle_v2_external/ocean.output:553`) | the Courant-dependent implicit vertical advection never enters the matrix |
| lateral operator | iso-level, not iso-neutral (`:706`) | the rotated lateral-mixing term never enters the matrix |
| `ln_isfcav` | `F` (`:338`) | every top-friction block is dead |
| `ln_drgice_imp` | `F` (`:630`) | so is the ice one |
| `ln_drgimp` | `T` (`:629`) | implicit BOTTOM friction is live, in both of its places |

With those four dead, what `dyn_zdf` runs on GYRE is small enough to score
term by term: a velocity-form explicit update (`dynzdf.F90:121-122`, taken
because `ln_dynadv_vec = .true.`,
`GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:161`), a barotropic removal and a
bottom-stress boundary addition (`dynzdf.F90:156-159`), the ZDF matrix
(`:182-195`), one implicit-drag diagonal term (`:296`), the wind entering
mid-solve through the `key_RK3` arm (`:329-330`), and three recurrences.

### How far the existing records localise the stage-3 owner: not far

The per-stage localisation the campaign already owns, recorded here because
round 28 did not carry it (**F4**):

| stage | `u` at the stage's own `Kaa` | status |
|---|---|---|
| 1 | `2.710505431213761e-19` | AT-BAR |
| 2 | `4.740083109008864e-13` | DEBT |
| 3 | `9.481924730527598e-07` | DEBT |

Stage 3 produces essentially all of it.  Two things INSIDE stage 3 are
already exonerated, and they are the only two: the vertical velocity the
stage consumes is at the bar at all three stages (round 21), and the mixing
coefficients the matrix reads are at the bar at the ZDF entry — `avm`
`3.469446951953614e-17` and `avt` `3.469446951953614e-18`, both AT-BAR.

Everything else is unmeasured, and a census says why.  All 51 `oracle_*.bin`
records in the V2 set were read for their magic: 21 distinct instrument
tags, and every momentum-operand tag among them — the stage-2 term record,
the HPG operands, the HPG literal, the EOS operands, the ENE operands, the
pre-update frame and the stage-2 operand frame — is written under
`kstg == 2`.  The only stage-3 records are tracer and vertical-velocity
ones, and the three `oracle_stage_kt00000001_s*.bin` files hold stage
OUTPUTS only, five fields each, no RHS and no matrix.

So the honest answer to "does a pre-`dyn_zdf` frame exist, and is it exact"
is **neither**.  There is no stage-3 momentum frame of any kind.  Six
operators run between the last scored frame and the first divergent one —
`eos` + `dyn_hpg`, `dyn_vor`, `dyn_adv`, `dyn_ldf`, `dyn_zdf`, and the
barotropic correction — with zero records among them.  Closing that is what
the instrument is for, and no localisation claim is made here.

### The instrument

Two WRITE-only records, at
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round29_zdf/`.
The user-executed recipe is `run.sh` in that directory; it is not run here.

The first is one frame: the momentum RHS entering `dyn_ldf` at stage 3.  It
exists only to answer the question the corrected premise raised, and one
bit-unequal cell in it moves the owner from `dyn_zdf` to `dyn_ldf` or
earlier.

The second is `dyn_zdf`'s own record: the RHS as received, the pre-solve
column vector after the explicit update and the two boundary corrections,
the matrix as built before the LU recurrence overwrites the diagonal, the
operands as consumed, and the solved velocity.  It is self-describing — a
magic, sixteen header integers including the writer's own tile bounds, then
one named group per array until EOF — so the reader parses to EOF without
knowing the list, and a short record is a hard failure rather than a quietly
smaller arm.  `e3u` and `e3uw` are preprocessor macros under `key_qco`, not
arrays, so each is materialised elementwise into a buffer already emitted.
`zwi`/`zwd`/`zws` are a single i-k slice reused inside the j loop, so they
are copied per slice into global buffers, which is also why the writer
refuses domain tiling rather than dumping one tile and labelling it the
field.

### The reader was calibrated against the writer, not against a mock

The gate scores two things on the record alone, and both are the reader's
calibration in the Rule-1e sense: no number from this record may be quoted
until they are at the bar.  `matrix` rebuilds `zwi`/`zwd`/`zws` for both
faces from the record's own operands by `dynzdf.F90:182-195` and `:296`.
`solve` replays NEMO's three recurrences on the dumped matrix and the dumped
pre-solve vector, with the wind term at `dynzdf.F90:329-330`.

Because the record does not exist yet, that calibration was taken against
the actual Fortran writer instead of postponed.  The inserted code was
preprocessed with the deck's own keys, compiled with gfortran, and run on a
small domain, with NEMO's own matrix-build and solve statements lifted
verbatim from the preprocessed source so the fixture is what its operands
imply rather than what the gate says they imply.  The python transcription
reproduces gfortran's evaluation of the same statements **bit for bit**:

| arm | bit-unequal | max abs |
|---|---:|---:|
| `zwi_u` / `zwd_u` / `zws_u` | `0 / 8` each | `0` |
| `zwi_v` / `zwd_v` / `zws_v` | `0 / 8` each | `0` |
| `zdf_solve.u` | `0 / 8` | `0` |
| `zdf_solve.v` | `0 / 8` | `0` |

Two independent evaluators agreeing is the strongest evidence available
before the record exists; it is not a claim about NEMO's real GYRE numbers,
which is what `run.sh` acquires.  Moving one operand by a single ulp turns
six of the eight rows red and the gate exits non-zero, so the green arm is
not vacuous.  The record, the compiled fixture generator and both gate
reports are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round29/` with SHA-256 in
`artifacts.sha256`; the record hashes
`d88b74e2a34c071697b712ce6ae01f0cfa7ef88384b266b65de02158a4b17e59`.

Writing the tests found a defect in the reader: a corrupt array name raised a
decoder traceback instead of returning a verdict.  A malformed record with no
verdict is exactly how a defeat hides — the same shape as round 28's reversed
range — so it fails closed now.  Nine tests pass.

### Decision 15C — the demo card takes NEMO's vertical momentum advection

The native demo GYRE card selected an upwind perturbation for vertical
momentum advection and ran the adaptive-implicit vertical advection.  NEMO's
GYRE does neither: it resolves the vector form, whose printed program is
`keg + zad + vor` (`dynadv.F90:144`), and `ln_zad_Aimp` is `.false.` by
reference default (`namelist_ref:1177`) and prints `F` on this deck
(`round19_oracle_v2_external/ocean.output:553`).  The two move together
because the adaptive-implicit path replaces the explicit vertical advection
entirely; the model already refuses the mixed pair.

That was the last field of the WS-RK3 vector-invariant momentum program the
card was missing, so it had been rejecting itself at construction and five
tests could not reach a model at all.

**What it was masking.**  With the card constructible, three of those tests
step, and all three stop in the same place: NEMO's literal QCO thickness
statement refuses the card because the demo card's z-star coordinate carries
no active-cell mask at all — `OceanZStarCoordinate` has no such attribute,
and the reference thickness is supplied by the caller, so the mask is the
missing operand.  Every oracle-bridge card gets one from NEMO's mesh; the
native demo card never did, and nothing on it had reached a statement that
asks.  Deciding what that mask IS on a card with no mesh file is a geometry
choice, so it is registered, not guessed.

The fourth test was asserting a guard this card is exempt from — its
split-explicit solver is named in the guard's own exemption clause, so the
assertion could never have fired.  It now runs against the solver the guard
covers, with the card's exemption asserted next to it.

Five failures with two causes become four with one, and the one is named.
`20 passed, 4 failed`.

### Bit debt registered (F5)

The reference-versus-live weighting round 28 refuted as an OWNER is still
not bitwise equal, and that is registered here rather than left implied:

| face | relative difference | bit-unequal | arm scale |
|---|---|---:|---|
| `u` | `8.307570867102084e-16` | `488 / 580` | `2.3896678125115675e-08` |
| `v` | `5.993770496385322e-16` | `488 / 570` | `3.0361478617076356e-08` |

About four ulp on roughly five sixths of the wet faces.  It is DEBT at the
campaign's exact bar, it is an arm-to-arm difference and not an oracle score,
and it owns nothing measured so far.

### Rule-11 records

**Corrected, not retracted.** The line number this round was handed for
`dyn_zdf` (`stprk3_stg.F90:430`) is the SHIPPED source's and is right.  A
draft of the preregistration cited line 523 instead, which is where the same
statement sits in the instrumented per-config copy of that file — a copy the
citation gate deliberately does not resolve, because a basename that can mean
either file is how a MY_SRC override passes as shipped source.  Both files
exist; only the shipped one is citable.

**Wrong premise, corrected before measuring.** `dyn_zdf` is not the only
stage-3-only momentum operator; `dyn_ldf` is one too, and it is active on
GYRE.  Recorded before any number was taken, which is why the instrument
dumps two boundaries instead of one.

**A gate defect found by using it.** A filename in backticks anywhere in the
prose rebinds every following bare `:N` citation to that file, and the
round-28 correction written this round did exactly that twelve lines later.
Caught by the gate.

### Open questions

1. **The `dyn_zdf` walk itself is UNMEASURED** until `run.sh` is executed.
   The prediction, its falsifier and the alternative owner are preregistered.
2. **The demo card's vertical coordinate has no active-cell mask.**  Four
   tests stop there.  What should that mask be on a card with no mesh file?
3. **The slow forcing's depth average** still uses the min rule where NEMO
   uses the area-weighted mean (`domqco.F90:166-169`, `:219-222`, transcribed
   in `vertical.py:772`).  `3.06e-08` at kt=2.  Land it, or leave it as debt?
4. **Decision 16 is preregistered and NOT landed** — the unconditional
   vorticity call in the pre-stage 2-D momentum RHS.  Its predictions are in
   the round-29 manifest; landing it needs cross-card runs this round did not
   have room for.
5. **Rounds 1-24 of this receipt remain UNAUDITED** by the citation gate.

### ASKED / UNASKED

| choice | disposition |
|---|---|
| correct the round-29 premise from one stage-3-only momentum operator to two | not a scientific choice; corrected against the shipped source before measuring, and gated |
| dump the pre-`dyn_ldf` momentum RHS as well as `dyn_zdf`'s operands | ASKED by the correction; it is the only frame that separates the two |
| `vertical_momentum_scheme` `upwind_perturbation` -> `nemo_advective`, `adaptive_implicit_vertadv` `True` -> `False` on the demo GYRE card | ASKED; user decision 15C; landed |
| give the demo card's z-coordinate an active-cell mask | NOT DONE — a geometry choice on a card with no mesh file; registered |
| require an anchor per cited endpoint in the citation gate | not a scientific choice; the interior of a comma citation was unvalidated and the plant now fires |
| add GYRE's deck, run log and compiled branch as separate citation keys | not a scientific choice; a GYRE premise was auditing LOCK's namelist |
| strike round 28's MIN-versus-MEAN disposition in place | ASKED; both halves were wrong and the debt is restated where it belongs |
| land decision 16's unconditional vorticity call | NOT DONE this round; preregistered only |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed |

UNASKED list: empty.

Every figure above is pinned in
`manifests/nemo_testcase_l2_gyre_round29.json`.

## Round 30 — the stage-3 owner is `dyn_ldf`, and it is not the kt2 blocker

Round 30 starts from `ce34ae0f16c6` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round30_preregister.json`, committed at
`d84262b0a0d4` before the pre-`dyn_ldf` and pre-`dyn_zdf` scores below.  No
NEMO executable was run.  Every artifact is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round30/` with SHA-256 in
`artifacts.sha256`.

### Rule 0 first — two corrections, both before any measurement

**The record this round was handed does not have the name it was given.**  The
task named the new pre-`dyn_zdf` frame `oracle_rhs_kt00000001.bin`.  That
basename is a PRE-EXISTING round-19 record (magic `NEMO_L1_RHS___1`) and is
byte-identical between the two roots.  The round-29 frame is
`oracle_rkstage3_preldf_kt00000001.bin` (magic `NEMO_L2_RKPLD_1`), and its
patch inserts it BEFORE the `dyn_ldf` call, not after it — so it is the
pre-`dyn_ldf` boundary, and the post-`dyn_ldf` one is `uu_Krhs_in` inside the
matrix record.

**Exactly ONE momentum call sits inside the stage-3 `CASE` block on this
deck.**  `stprk3_stg.F90:400` is `dyn_ldf`; the other three calls in that block
are gated on `ln_zdfosm`, `ln_bdy` and `ln_dyndmp .AND. ln_c1d`, and the run's
own printed output resolves all three false —
`round19_oracle_v2_external/ocean.output:559`,
`round19_oracle_v2_external/ocean.output:546` and
`round19_oracle_v2_external/ocean.output:214` respectively.  `dyn_zdf` is NOT in the block at all: it is called under
`IF( kstg == 3 )` at `stprk3_stg.F90:430`, after the block closes.  So the two
frames bracket exactly one operator, which is what makes the walk a one-number
question.

### Item 1 — the round-29 root is ADMITTED

`nemo_testcase_l2_gyre_round21_admission.py` against
`round19_oracle_v2_external`: **39 of 49** pre-existing `oracle_*.bin` records
byte-identical, the final restart **byte-identical**, verdict `PASS`, zero
violations.  The 10 that differ are the same 10 round 21 classified, and every
one has `consumed_equal: true`:

| record | changed fields | disposition |
|---|---|---|
| `oracle_bt_ordered_operands` | 8 appended `ff*` fields | schema `NEMO_L2_BTORD_1 -> _2`, instrument version |
| `oracle_rkstage1_transport_operands` | `zub`, `zvb` | unconsumed slot |
| `oracle_rkstage3_wzv` | `ww_pre_aimp`, `ww_post_aimp`, `pFw` | unconsumed slot |
| `oracle_rktracer_operands` s1, s2 | `zFw` | unconsumed slot |
| `oracle_slow_forcing` | `utau`, `vtau` | unconsumed slot |
| `oracle_tracer_transport` s3 | `zFw` | unconsumed slot |
| `oracle_transport` s1, s2, s3 | `zFw` | unconsumed slot |

Both new records parse to EOF from their own headers.  The pre-`dyn_ldf` frame
is `464316` bytes: magic, eleven integers
`(1, 1, 3, 1, 2, 3, 3, 36, 26, 31, 64)` — version, `kt`, `kstg`, `Kbb`, `Kmm`,
`Krhs`, `Kaa`, `jpi`, `jpj`, `jpk`, bits — then exactly `2 * 36 * 26 * 31`
doubles.  The matrix record's header carries `jpkm1 = 30` and the writer's own
tile bounds `ntsi..ntei = 3..34`, `ntsj..ntej = 3..24`, and its 32 named arrays
are all present.

### Item 2 — the reader's calibration, now on the REAL record

Round 29 could only calibrate against a compiled Fortran fixture.  Re-run on
NEMO's own GYRE record, every row is at the bit bar:

| arm | bit-unequal | max abs |
|---|---:|---:|
| `zwi_u` / `zwd_u` / `zws_u` | `0 / 21120` each | `0` |
| `zwi_v` / `zwd_v` / `zws_v` | `0 / 21120` each | `0` |
| `zdf_solve.u` | `0 / 21120` | `0` |
| `zdf_solve.v` | `0 / 21120` | `0` |

`STATUS AT-BAR`, exit `0`; the one-ulp plant on `avm` turns six rows red and
exits `1`.  So NEMO's `dyn_zdf` matrix (`dynzdf.F90:182-195` plus the implicit
bottom-drag diagonal at `dynzdf.F90:296`) and its three recurrences with the
`key_RK3` wind term (`dynzdf.F90:329-330`) are reproducible bit for bit from
the record's own operands, and any number quoted from this record is admissible
(Rule 1e).

### Item 3 — the pre-`dyn_zdf` boundary, and the owner

Scored through the model's own path — production JIT, fp64, the phase-3 gate's
masks and scorer — by
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round30_stage3_owner.py`.
The model side is a WRITE-only exposure of the stage-3 momentum RHS; the
ordinary step completes before the arrays are substituted.

| boundary | u bit-unequal | u max abs | u relative | v bit-unequal | v max abs |
|---|---:|---:|---:|---:|---:|
| pre-`dyn_ldf` (`stprk3_stg.F90:400`) | `17382 / 17400` | `2.0614443633579772e-16` | `7.83e-09` | `17096 / 17100` | `2.4827278704178648e-16` |
| pre-`dyn_zdf` (`stprk3_stg.F90:430`), before | `17400 / 17400` | `8.256225965606334e-10` | `3.14e-02` | `17100 / 17100` | `8.290054935129283e-10` |

**P1 of the round-30 preregistration is REFUTED, and P2's alternative is
CONFIRMED.**  The frame entering `dyn_zdf` is not bit-exact, so the owner is at
or before `dyn_ldf`; and the pre-`dyn_ldf` frame is four million times closer
than the post-`dyn_ldf` one, so the owner is `dyn_ldf` itself.  (The pre-`ldf`
row is AT-BAR on the campaign's normalized scorer only because that scorer
divides by `max(|oracle|, 1)` and this field's maximum is `2.63e-08`; the
relative figure is quoted next to it for that reason.  It is not bit-exact and
it is not claimed to be.)

**What `dyn_ldf` does at kt=1: exactly nothing.**  `dynldf.F90:70` dispatches
the resolved `np_lap` operator to `dynldf_lev_lap`, whose vorticity-divergence
scheme builds its curl from `pv_in(...,Kbb)` and `pu_in(...,Kbb)`
(`dynldf_lev_rot_scheme.h90:24-25`) and its divergence from the same BEFORE
velocity (`dynldf_lev_rot_scheme.h90:28-29`).  GYRE starts kt=1 from rest, and
the matrix record proves it rather than assuming it: `uu_Kbb_in` and
`vv_Kbb_in` are identically zero, `0` nonzero elements each.  So NEMO's
`dyn_ldf` adds a term that is exactly `0.0`, and the two oracle frames confirm
it directly — `max |pre - post|` is `0.0` on BOTH faces, with `0` bit-unequal
cells on `u` and `2220` signed-zero flips on `v` (IEEE `-0.0 + 0.0 = +0.0`).

legoESM was adding `8.256e-10` there because it evaluated the same operator on
the STAGE velocity.  The shared seam for the BEFORE-level operand already
existed and the leap-frog path already used it
(`ocean_pe_latlon_cgrid.py:5631-5632`); the WS-RK3 stage RHS simply did not
pass it.  Landed at `7521513a54c3` in that one shared place, no card switch,
no new knob.  Stage 1 hands the helper the step-entry velocity already, so only
the stage-3 call moves, and stage 2 never calls the operator.

| boundary | u bit-unequal | u max abs | v bit-unequal | v max abs |
|---|---:|---:|---:|---:|
| pre-`dyn_zdf`, after the fix | `17382 / 17400` | `2.0614443633579772e-16` | `17098 / 17100` | `2.4827278704178648e-16` |

That is the pre-`dyn_ldf` residual, unchanged, which is the correct answer: the
operator now contributes what NEMO's contributes.  The pre-`dyn_ldf` arm's
report file is BYTE-IDENTICAL before and after
(`61a98efccdcf7db62b6e8e40392b15d018a44f6cad9a75ade606c74aacd7f850`), which is
the one-variable control — the fix could not have reached the arm it was not
supposed to touch.

### Item 4 — the fix is NOT the kt2 blocker, and the walk must continue

Both trajectory arms were run at the same protocol, all three oracle roots
pinned to the V2 root, `--trajectory-only`.
**WITHDRAWN by round 31 — the AFTER arm was NOT on a committed clean tree.**
Its report was written at 22:13:10 and the BEFORE arm's at 22:16:54; the fix
commit is timestamped 22:20:56, seven minutes after the later of the two.  So
both arms ran on the SAME uncommitted tree, the AFTER one carrying the fix as
a working-tree edit and the BEFORE one carrying a one-line temporary revert of
it.  That is still a controlled pair — one line differs — but the sentence
claimed a provenance the timestamps refute, and neither artifact carries a
revision stamp that could have shown it (round 31, item 2).  The restoration
was verified by file SHA-256 at the time; the hash is recorded in round 31.

| row | before | after |
|---|---:|---:|
| `kt2.before.T` | `1.3614736849003888e-12` | `1.3614736849003888e-12` |
| `kt2.before.S` | `2.2181101297999213e-14` | `2.2181101297999213e-14` |
| `kt2.before.u` | `9.4840899384114608e-07` | `9.48236979236058e-07` |
| `kt2.before.v` | `8.987992610401277e-07` | `8.9486629205594802e-07` |
| `kt2.before.ssh` | `4.3368086899420177e-19` | `4.3368086899420177e-19` |
| `kt3.before.u` | `9.3116789131403085e-03` | `7.1930784111826074e-04` |
| `kt3.before.v` | `4.7190775913077548e-03` | `8.6064784770509273e-04` |
| `kt4.before.u` | `1.4182404011957986e-02` | `7.078541518550233e-03` |
| `kt5.before.u` | `2.0733890435921859e-02` | `9.2090289296850921e-03` |
| `kt5.before.ssh` | `6.7018510230519579e-05` | `9.3606152350264767e-07` |
| `kt6.before.ssh` | `1.6310239045095183e-04` | `3.2278369542912064e-06` |
| `kt7.before.ssh` | `2.0871418234338955e-04` | `6.646098587755056e-06` |
| `kt8.before.ssh` | `2.5108247679344643e-04` | `1.1722279529014011e-05` |
| `kt9.before.ssh` | `2.1203002219557815e-04` | `2.167864546419253e-05` |
| `kt10.before.u` | `5.6249869570505276e-02` | `5.1403631510096456e-02` |
| `kt10.before.ssh` | `2.1255435580060100e-04` | `3.3284883532392094e-05` |

**CORRECTED by round 31 — there are EIGHT worsened rows, not four**, and this
paragraph listed half of them with no threshold stated.  The four named here
were `kt4.before.T`, `kt5.before.v`, `kt6.before.v` and `kt8.before.v`; the
four omitted were `kt4.before.ssh` (+4.63 per cent, LARGER than two of the
four that were listed), `kt5.before.T` (+0.20 per cent), `kt4.before.S`
(+0.15 per cent) and `kt3.before.T` (+1.8e-06 per cent).  The complete
register, with the threshold that separates them and the boundary and owner
stated once for all eight, is in round 31, item 1.  Their boundary is the
stage-3 momentum RHS; nothing in round 30 attributes them.

**The merge blocker does NOT clear.**  First-over-bar stays `kt2` on
`T`/`S`/`u`/`v`, and the u row moves by 0.02 per cent.  So the stage-3 momentum
RHS is EXONERATED as the owner of GYRE's kt2 divergence: with it at the bit bar
the blocker is essentially unchanged.  What remains between an (almost) exact
RHS and a `9.48e-07` stage-3 velocity is `dyn_zdf` itself — its explicit
update, its barotropic removal and bottom-stress addition, its matrix, its
solve — and the barotropic correction that follows it at
`stprk3_stg.F90:437-446` (corrected by round 31; `stprk3_stg.F90:430` is the
`CALL dyn_zdf` itself, not the correction).  That is the next boundary, and this round does not name
a statement inside it: legoESM exposes none of those internals, and inventing
a number for them would be exactly the failure this campaign's rules exist to
stop.  The oracle side of that walk is now fully instrumented and bit-verified
(item 2), so the remaining work is entirely on the model side.

### Item 5 — cross-card, Rule 12

The changed operator's coefficients, instantiated and printed rather than read
off a comment:

| card | `A_h` | `B_h` | `C_smag` | `C_leith` | operator |
|---|---:|---:|---:|---:|---|
| GYRE-zco | `100000.0` | `0.0` | `0.0` | `0.0` | `nemo_div_curl` |
| LOCK_EXCHANGE-zco | `0.0` | `0.0` | `0.0` | `0.0` | `vector_laplacian` |
| OVERFLOW-zps | `0.0` | `0.0` | `0.0` | `0.0` | `vector_laplacian` |

GYRE's `1.0e5` is `0.5 * rn_Uv * rn_Lv` from
`GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:189-191` with the operator selected at
`GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:187-188` and printed as `iso-level
laplacian operator` at `round19_oracle_v2_external/ocean.output:706`.  The two
tanks resolve `ln_dynldf_OFF` (`lock_kt1_10/ocean.output:615`,
`overflow_kt1_10/ocean.output:727`), and legoESM's cards carry zero
coefficients to match, so the operator is identically zero there whichever
velocity it reads.

| card | first-over-bar before | first-over-bar after | source of the before |
|---|---|---|---|
| LOCK_EXCHANGE-zco | kt4 `u` | kt4 `u` | round 28's merge-readiness table |
| OVERFLOW-zps | kt2 `T`/`u` | kt2 `T`/`u` | round 23's trajectory row |
| GYRE-zco | kt2 `T`/`S`/`u`/`v` | kt2 `T`/`S`/`u`/`v` | measured both arms, table above |

ORCA2 is **UNMEASURED** and the reason is not that it was skipped: the card
does not exist in this worktree's registry, which offers exactly
`GYRE-zco`, `LOCK_EXCHANGE-zco` and `OVERFLOW-zps`.  Round 27's ORCA2 rows came
from a disposable probe worktree, and reproducing them needs that lane.

### Item 6 — decision 16 is NOT landed, because its premise is REFUTED

The preregistered decision was to make the pre-stage 2-D vorticity call
unconditional, as `stp2d.F90:146` (`dyn_vor` after `dyn_hpg` at
`stp2d.F90:128` and `dyn_ldf` at `stp2d.F90:131`), on the stated ground that
"on LOCK and OVERFLOW f = 0, so it adds `+0.0` through `vor_ens`".  The
measured baseline is reproduced exactly — LOCK `1260 / 2540` bit-unequal, all
of them signed-zero-only, `0` remaining after adding NEMO's zero; OVERFLOW
`16400 / 16900`, likewise all signed-zero and all healed.

But the wiring is not a skipped call.  The three cards' resolved momentum
programs, instantiated and printed:

| card | momentum advection | flux scheme | vorticity | Coriolis |
|---|---|---|---|---|
| GYRE-zco | `vector_invariant` | `upwind` | `ene_total` | `explicit_ab2` |
| LOCK_EXCHANGE-zco | `flux_form` | `nemo_up3` | `al81` | `matsuno_split` |
| OVERFLOW-zps | `flux_form` | `nemo_up3` | `al81` | `matsuno_split` |

On GYRE the planetary term is INSIDE the vorticity flux in the momentum RHS,
i.e. at NEMO's own position.  On the two tanks the flux-form branch does not
call the vorticity operator at all, and Coriolis is applied as a separate
velocity rotation OUTSIDE the RHS — a different operator splitting, not a
conditional that can be made unconditional.  Adding an unconditional vorticity
accumulation to the flux-form RHS would therefore DOUBLE-COUNT Coriolis on
every `f != 0` flux-form card, which is the default legoESM ocean
configuration, and it is value-inert on LOCK and OVERFLOW only because `f = 0`
there.

So the honest statement of the residual is: it is an operator-splitting
difference whose only observable is the sign of a zero, and closing it means
moving Coriolis out of the Matsuno rotation into the RHS for the flux-form
path — a scheme selection this round is forbidden to make.  Adding a literal
`+ 0.0` to normalise the signed zeros would pass the gate and transcribe
nothing, so it was not done.  Decision 16 is returned to the user with this
measurement attached.

### Rule-11 records

**Round 30's own P1 is REFUTED by its own first measurement.**  The
preregistration predicted the pre-`dyn_ldf` frame would be bit-exact.  It is
not: `17382 / 17400` cells differ, at `2.06e-16`.  The prediction was written
from the stage-2 row's size and was wrong about the residual the stage-3 RHS
inherits.

**The round-29 task premise, corrected for the second round running.**  Round
29 corrected "`dyn_zdf` is the only stage-3-only momentum operator" to "there
are two".  Round 30 narrows it again: `dyn_zdf` is not even inside the stage-3
`CASE` block, and `dyn_ldf` is the only momentum call in it that runs on this
deck.

**No retraction of a landed number.**  Every figure round 28 and 29 recorded
for the stage boundaries is reproduced here at the same protocol.

### Round-30 independent review — two reviewers, one converging finding

Codex is unavailable on this account, so the mandatory DUAL adversarial review
ran as two fresh independent agents, one on the DIFF and one on the CLAIM, each
briefed separately and neither shown the other's answer.  They converged on the
same defect from opposite directions, which is the strongest signal this
process produces.

**FINDING (both reviewers, independently): the fix is HALF of NEMO's operand,
and the other half is invisible at kt=1.**  NEMO's scheme reads THREE distinct
thicknesses — `e3t`/`e3u`/`e3v` at `Kbb` inside the divergence
(`dynldf_lev_rot_scheme.h90:28-29`), `e3f` carrying no time index at all
(`dynldf_lev_rot_scheme.h90:24-25`), and `e3u`/`e3v` at `Kmm` in the final
division.  legoESM's lateral operator receives ONE thickness: a single `h_k`
argument at `ocean_pe_latlon_cgrid.py:5654`, built from the stage's own live
`eta`, which at stage 3 is the `Kmm` sea level.  So after this round the
velocity operand is NEMO's and the thickness operand is not.  **CONFIRMED by
reading, UNMEASURED in size**: at kt=1 the whole term is exactly zero, so no
existing record can size it, and the discriminating measurement is a kt=2
pre-`dyn_ldf` / post-`dyn_ldf` frame pair, which does not exist yet.  It is
registered here as the next candidate owner rather than asserted to be one.

**The reviewer also sharpened WHY the kt=1 term is zero, and my reason was the
weaker one.**  Both `zcur` and `zdiv` are purely multiplicative in velocity —
coefficient times a velocity difference, with no additive term — so zero
velocity gives exactly zero whatever time level any thickness carries.  The
`Kbb` read is not what makes it zero; it is what makes it zero at kt=1 for
NEMO and NOT zero for legoESM, which is a different statement and the one that
belongs above.  With `nn_ahm_ijk_t = 0` the coefficient is the constant
`0.5 * rn_Uv * rn_Lv = 1.0e5`, masked, and no operand can make the term
nonzero.

**The arithmetic gap in item 4 is CLOSED, and it is not a contradiction.**
`14400 * 8.256e-10 = 1.19e-05` is twelve times LARGER than the `9.48e-07` the
fix did not remove, which looked wrong.  It is not: after `dyn_zdf` the stage
resets the reference-weighted depth mean of `uu(Kaa)` to `uu_b(Kaa)`, which the
barotropic solver produced without ever seeing `dyn_ldf`, so only the
BAROCLINIC part of a stage-3 RHS error survives.  Measured from the round-29
dump: stage-2 velocity max `4.255e-04` m/s against a baroclinic deviation of
`1.04e-07`, a ratio of `2.4e-04`, because GYRE's initial T/S are horizontally
uniform.  Predicted surviving error `1.19e-05 * 2.4e-04` is about `3e-09`;
the observed kt2 move is `1.7e-10`.  **RELABELLED by round 31: this BOUNDS the
move, it does not confirm it.**  The prediction is 17 times larger than what
was observed, and a bound satisfied by a factor of 17 discriminates against
almost nothing — a mechanism that produced ten times less would satisfy it
too.  What it does do is rule out the arithmetic reading that first looked
wrong, that `14400 * 8.256e-10 = 1.19e-05` should have shown up whole.  It
supports the exoneration in item 4 as a consistency check, and the exoneration
still rests on the before/after pair.

**Provenance, noted not changed.**  The round-29 acquisition executed
`cfgs/GYRE_OMIP_L2_P3_SM_R29ZDF/MY_SRC/stprk3_stg.F90`, whose `dyn_ldf` sits at
its own line 509, not the shipped `src/OCE` line.  Round 29 settled the
convention for exactly this case — a basename that can mean either file is how
a `MY_SRC` override passes as shipped source, so the citation gate resolves the
SHIPPED file only — and the reviewer's diff of the two confirms the copy is
instrument-only.  The citations above are therefore the shipped ones, and this
paragraph is the record that the executed file is a copy of them.

**Two diff defects, both fixed in this round.**  Setting two momentum exposures
at once used to hand the caller the later frame under the earlier one's name,
because they share the returned `u`/`v` slots; that is now refused at
construction, with three tests plus a companion asserting the production
default still constructs.  And the six fail-closed reader cases asserted only
the exception TYPE, so all six could have been firing on one shared guard;
each now names the message it must produce.  13 tests pass.

**Two latent items registered, neither this round's to fix.**  The stage face
mask handed to the lateral operator is the 2-D `u_mask` broadcast over levels
where NEMO's `umask` is 3-D — inert on GYRE, which has one constant depth and
therefore a flat bottom, and live on any topography card.  And legoESM builds
its face thickness by a min-rule where NEMO uses `e3u_0 * (1 + r3u)`; the same
class as the finding above and equally invisible at kt=1.

### A concurrent session wrote to this branch again

Two commits landed on `fidelity/nemo-testcases-l2-gyre-codex2` between round
30's preregistration and its fix: `1829f219987e`, which stops a
card-constructibility test asserting that the legacy native GYRE card raises,
and `74306d92e05a`, which extends the round-29 manifest.  Neither touches
`packages/` or `src/` — verified by listing every file in
`ce34ae0f16c6..74306d92e05a` — so no number in this section is affected, and
the before/after trajectory pair straddles them with identical model code.
They are round 29's, not round 30's, and they are flagged rather than reverted.

### Tests

Focused: the citation gate, the round-29 matrix gate, the round-30 boundary
gate and the LOCK slow-forcing probe together, **37 passed**.  The nine
round-30 tests cover the reader's interior layout against a known element and
six fail-closed arms — wrong magic, a future version, the stage-2 frame, a
32-bit payload, a short payload and a non-finite one — plus the fail-closed
registry.  `tests/ocean/unit/test_nemo_recipe.py` is `20 passed, 4 failed`,
the same four round 29 recorded and for the same reason.

Whole directory, run without `-x` so nothing hides behind the first failure:

```text
4 failed, 783 passed, 7 skipped, 18 deselected in 2294.60s (0:38:14)
```

Every one of the four is dispositioned, and **round 30's own count is zero**:

| failure | disposition |
|---|---|
| `test_nemo_testcase_phase3_stage_sweep_gate::test_planted_stage_control_exits_nonzero_end_to_end` | **PRE-EXISTING**; round 28 reproduced it failing at ITS starting tip in a disposable worktree.  It is a LOCK arm, and this round's change is identically zero on LOCK |
| `test_recipe_case_board::test_every_oracle_comparison_has_a_row` | **PRE-EXISTING**; round 28 recorded the same four missing board rows, `advection_nemo`, `grids_tripole_mpas`, `tendencies_nemo`, `three_way_nemo`.  This round adds no comparison driver |
| `test_nemo_testcase_phase3_stage_sweep_gate::test_prediction_plant_is_fail_closed` | **NOT A FAILURE — a duration artifact**, and this was measured rather than assumed.  The traceback is `Timeout (>900.0s) from pytest-timeout` inside JAX lowering, not an assertion.  Re-run alone at `--timeout=3000` it is `1 passed in 996.95s`, so the test needs about 997 s and the suite's 900 s per-test cap cut it |
| `test_nemo_testcase_receipt_citation_gate::test_the_gate_runs_clean_on_the_real_receipt` | **AN ARTIFACT OF THIS ROUND'S OWN EDITING**, not of its code.  That 38-minute run read the receipt while this section was being appended, before the round-30 citations were in the map.  It passes now, twice, `16 passed` |

### Merge readiness

`03c6e8d96ff7` remains an ancestor of this branch, so the integration is still
a FAST-FORWARD with zero conflicts by construction.  What blocks it after this
round:

1. **GYRE's kt2 `T`/`S`/`u`/`v` rows still fail.**  `u` at `9.482e-07` on
   `17400 / 17400` faces, `v` at `8.949e-07`, `T` at `1.361e-12`, `S` at
   `2.218e-14`; `ssh` is AT-BAR.  This is the same blocker round 28 named,
   and this round removed one candidate owner for it rather than the blocker.
2. **Round 30 DOES land model numerics**, unlike round 28.  One shared
   change, in one function, measured on all three cards this tree can build.
   A reader integrating this branch inherits a stage-3 momentum RHS that is at
   the bit bar where it was `3.1` per cent off, four worsened rows registered
   as debt above, and no card switch.
3. **ORCA2 is unmeasured against this change** and needs its lane's worktree.

### Open questions

0. **The lateral operator's THICKNESS operand**, named by both reviewers and
   unmeasurable from any existing record because the term is zero at kt=1.  A
   kt=2 pre-`dyn_ldf` / post-`dyn_ldf` frame pair discriminates it, and that is
   a cheaper acquisition than instrumenting `dyn_zdf`'s internals.
1. **Where inside `dyn_zdf` GYRE's kt2 divergence enters.**  The oracle side is
   instrumented and bit-verified; the model side exposes no pre-solve vector,
   no matrix and no solved column, so the next round's first job is that
   exposure, not another oracle run.
2. **The four worsened trajectory rows** (`kt4` T, `kt5`/`kt6`/`kt8` v) have a
   boundary but no owner.
3. **Decision 16**, returned to the user above with its measurement.
4. **Decision 17**, the demo card's vertical coordinate, untouched here.
5. **The slow forcing's depth average** still uses the min rule where NEMO uses
   the area-weighted mean; `3.06e-08` at kt=2, unchanged.
6. **Rounds 1-24 of this receipt remain UNAUDITED** by the citation gate.

### ASKED / UNASKED

| choice | disposition |
|---|---|
| correct the round-30 record name and the stage-3 `CASE` block's contents | not a scientific choice; corrected against the shipped source before measuring, and gated |
| route the WS-RK3 stage lateral viscosity to the step-entry velocity | ASKED by the oracle: `dynldf_lev_rot_scheme.h90:24-25` and `dynldf_lev_rot_scheme.h90:28-29` read `Kbb`, and NEMO's kt=1 contribution is measured to be exactly zero. One shared implementation, no card switch, no knob |
| land decision 16's unconditional vorticity call | NOT DONE — premise REFUTED, measurement returned to the user |
| change the demo card's vertical coordinate | NOT DONE — decision 17 pending |
| refuse two simultaneous momentum exposures at construction | not a scientific choice; a reviewer finding, and a gate that scores the wrong frame under the right name is the defeat these gates exist to stop |
| do NOT act on the thickness-operand finding this round | ASKED by the evidence: it is CONFIRMED structurally and UNMEASURED in size, and this campaign does not land a fix whose size no record can show |
| add four citation-map entries for `dynldf.F90`, `dynldf_lev_rot_scheme.h90` and the shared `ldf_state` seam | not a scientific choice; without them the gate is fail-closed on this round's citations |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed |

UNASKED list: empty.

## Round 31 — the kt2 owner is the ORDER of the barotropic correction

Round 31 starts from `ce38f20c89a0` on a clean re-attached tree, same regime:
CPU production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round31_preregister.json`, committed at
`3a36c07fc335` before any measurement, with three addenda each committed
before the run it governs.  No NEMO executable was run.  Every artifact is
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round31/` with SHA-256
in `artifacts.sha256`, and — for the first round in this campaign — every one
of them names the tree that produced it: the five walk arms are stamped
`835e536808ae`, clean, on this branch, and the trajectory re-run
`da93c090cf73`, clean, on this branch.  The two commits between them touch
only the citation map and a docstring.

### Rule 0 first — the task's own citation is off by one at both ends

The round-31 task names the barotropic correction after `dyn_zdf` as lines
434 to 445 of `stprk3_stg.F90`.  Read against the shipped source that is wrong
at both ends: line 434 is a bare comment, the block opens at the PSYCLONE guard
on `stprk3_stg.F90:435` and its last statement is `END_3D` on
`stprk3_stg.F90:446`.  The two
statements that add the column shift are `stprk3_stg.F90:444-445`.  The
citation gate already carried the correct range, `stprk3_stg.F90:437-446`,
anchored on `#endif` and `END_3D`.  Corrected before anything was scored.

Round 30's receipt cited `stprk3_stg.F90:430` for the same block, which is the
`CALL dyn_zdf` itself.  That is struck in place, above.

### Item 1 — the round-30 receipt, corrected in place

The merge artifact is the receipt, so all six corrections land in the round-30
text as well as here.

**EIGHT rows moved the wrong way, not four.**  The threshold, stated because
round 30 stated none: a row is WORSENED when the trajectory gate's
`normalized_max_abs` is larger after than before, and it is ABOVE ROUNDOFF
when that increase exceeds `1e-6` relative.  Seven of the eight are above it.

| row | before | after | relative |
|---|---:|---:|---:|
| `kt8.before.v` | `2.4527161866164299e-02` | `6.2034193135222182e-02` | `+152.92 %` |
| `kt4.before.ssh` | `5.0604233711813020e-07` | `5.2949464552374050e-07` | `+4.6345 %` |
| `kt5.before.v` | `4.3797665645157385e-02` | `4.5721416735565040e-02` | `+4.3924 %` |
| `kt6.before.v` | `6.0219592086987961e-02` | `6.2092404213714036e-02` | `+3.1100 %` |
| `kt4.before.T` | `1.0594738051818496e-03` | `1.0662127787936730e-03` | `+0.6361 %` |
| `kt5.before.T` | `3.0537424781428640e-03` | `3.0599261436702885e-03` | `+0.2025 %` |
| `kt4.before.S` | `8.2257404626745840e-05` | `8.2383027017548940e-05` | `+0.1527 %` |
| `kt3.before.T` | `3.7223441372709554e-04` | `3.7223442048188220e-04` | `+1.8e-06 %` |

Round 30 listed rows one, three, four and five.  It omitted `kt4.ssh`, which
is LARGER than two of the four it listed.  34 rows improved and 8 were
unchanged, out of 50.

**BOUNDARY and OWNER, stated once for all eight.**  The boundary is the
stage-3 momentum RHS: the only model change between the arms is the velocity
level the lateral viscosity reads.  The owner is UNKNOWN and nothing here or
in round 30 attributes it.  What round 31 adds is that the owner is not the
stage-3 velocity error itself, because that error is now attributed (item 4)
to a term the lateral-viscosity change does not touch.

**WITHDRAWN: "the AFTER arm on the committed clean tree".**  The AFTER report
was written at 22:13:10 and the BEFORE one at 22:16:54; the fix commit
`7521513a54c3` is timestamped 22:20:56, seven minutes after the later of the
two.  Both arms ran on the same uncommitted tree — the AFTER one carrying the
fix as a working-tree edit, the BEFORE one carrying a one-line revert of it.
That is still a controlled pair, but not the provenance the sentence claimed,
and no artifact could have shown it: none of the five round-30 JSONs carries a
revision stamp of any kind.  Item 2 closes that.

**RESTORATION HASH.**  The file the temporary revert touched is
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` — not the
`ocean_pe_latlon_cgrid.py` the round-30 text cites for the shared seam, which
the fix does not modify.  Its SHA-256 at `7521513a54c3`, the state the revert
was restored to, is
`1878abd84548f115b31cf9972c644351bea65af67bbdb995819d3fad904f0c89`.  The same
file at this round's tip hashes differently, because four later commits touch
it; that is expected and is why the hash is pinned to the commit.

**RULE-12 ELIGIBILITY FOR THE `dyn_ldf` CHANGE IS DISCHARGED ON NO CARD.**
Stated plainly because round 30 did not.  Rule 12 requires the changed
operator to be bit-exact given NEMO's own inputs on every card that executes
it.  On GYRE the operator ran on zero non-zero-input cells: the run starts
from rest, `uu_Kbb_in` and `vv_Kbb_in` are identically zero in the round-29
record, and both `zcur` and `zdiv` are purely multiplicative in velocity, so
the term is exactly `0.0`.  Both tanks resolve `ln_dynldf_OFF`.  ORCA2 is
covered in item 5 and cannot discharge it either.  So the fix is landed on a
reading of `dynldf_lev_rot_scheme.h90:24-25` and `:28-29` plus a measured zero,
and its eligibility is OPEN.

**COMMIT MESSAGE versus RECEIPT.**  The round-30 commit message says the
worsened rows are "a second error exposed"; the receipt says their owner is
UNKNOWN.  Both are as intended and they are not the same claim.  Rule 12's
phrase "a second error exposed" names the CATEGORY a worsened row enters when
a faithful fix removes a compensating error — it does not name the error.  The
receipt's "owner UNKNOWN" is the attribution, and there is none.  This note is
the reconciliation; no wording changes.

**THE ROUND-29 INSTRUMENT REFORMATTED ONE PHYSICS LINE.**  Diffing
`cfgs/GYRE_OMIP_L2_P3_SM_R29ZDF/MY_SRC/stprk3_stg.F90` against the shipped
`src/OCE/stprk3_stg.F90`, every added line is instrument except one, where the
single-line `IF( ln_traqsr ) CALL tra_qsr( kstp, Kmm, ts, Krhs )` became a
block `IF( ln_traqsr ) THEN ... CALL tra_qsr( kstp, Kmm, ts, Krhs ) ... ENDIF`
so the instrument could bracket the call.  The call and its four arguments are
identical; only interior whitespace changed.  `dynzdf.F90`'s instrumented copy
removes no physics line at all.  Recorded because a reformatted physics line
in an oracle-acquisition copy is exactly the kind of thing that should never
be found later.

### Item 2 — a gate report that cannot name its own tree is not evidence

Round 30's worktree was left DETACHED by an earlier session and every gate run
in it for most of a day emitted numbers under a `HEAD` that named different
source.  Measured, not assumed: `legoesm_git_sha` is absent from all five of
`round30_gyre_trajectory_{before,after}.json`, `round30_pre_ldf.json`,
`round30_post_ldf_after.json` and `round30_zdf_calibration.json`.

The repo already had the right helper and it was reused rather than rebuilt:
`provenance.py:106`'s `git_sha` fails closed on tracked dirt for exactly this
reason.  What it could not answer is whether `HEAD` was describing this code at
all.  `worktree_stamp` adds the branch (or `DETACHED`), whether the tree is
clean, and, when it is not, the SHA-256 of the full `git diff HEAD`.  Detached
is RECORDED, never refused — the cross-card probe lanes are deliberately
detached worktrees — and dirt refuses, with one named escape that suppresses
nothing.

Wired into all 62 report dicts across all 37 report-emitting gates in that
directory (36 by the AST pass, plus this round's own gate),
mechanically by AST so none was missed, and held by a grow-only ratchet.  It
fired for real three times during this round: each time an edit landed while an
arm was running, that arm refused at report time rather than stamping a commit
that no longer described its code.  That cost three runs and is the whole
point.

**The round-30 trajectory arm, re-run at this tip with the stamp.**  All ten
steps, same protocol, all three oracle roots pinned to the V2 root,
`--trajectory-only`, on a clean tree on the branch: **all 50 rows reproduce
EXACTLY**, zero differing, first-over-bar `kt2` on `T`/`S`/`u`/`v`, `u` at
`9.48236979236058e-07` and `v` at `8.94866292055948e-07`.  So round 30's AFTER
numbers ARE the committed tree's numbers — which is what the withdrawn sentence
claimed and could not show.  It is shown now, with a stamp naming
`da93c090cf73` on branch `fidelity/nemo-testcases-l2-gyre-codex2`, clean.  The
BEFORE arm is not re-run: reproducing it needs the fix temporarily reverted,
and its numbers are already recorded.

### Item 3 — the depth profile, and what it does and does not settle

GYRE's kt=1 stage-3 velocity error, binned by model level over the 580 wet u
columns and 570 wet v columns.  All 580 columns are 30 levels deep — GYRE has
a flat bottom — so level 30 is the deepest wet level everywhere.

| level | `max abs` (u) | signed mean (u) |
|---|---:|---:|
| 1 | `7.131145e-08` | `-1.019240e-09` |
| 15 | `7.131128e-08` | `-1.019234e-09` |
| 29 | `7.097569e-08` | `-1.014435e-09` |
| **30** | **`9.482370e-07`** | **`+1.355289e-08`** |

Levels 1 to 29 sit on top of each other to five digits; level 30 is 13.3 times
larger and of the opposite sign.  Zero of 580 u columns and zero of 570 v
columns are column-uniform to 1 per cent, and the median per-column span over
peak is `1.0752`.

**P3a is CONFIRMED and P3b REFUTED**, so an error in `zub` ITSELF —
the barotropic correction's own operand — is out.  That is what this
measurement settles, and the independent CLAIM review is right that it is less
than round 31 first claimed: it does not separate the two statements that act
at the deepest level only, and it never could.

**The review's reading, adopted, and it is sharper than mine.**  NEMO's
correction is not an additive constant.  `stprk3_stg.F90:440` builds
`zub = uu_b(Kaa) − SUM(e3u_0·uu(:,Kaa))·r1_hu_0` from the SOLVE OUTPUT, so it
is a rank-1 operator: a single-level error `d` at the deepest level appears as
`(1−f)·d` there and `−f·d` at every level above, with `f = e3(deepest)/hu_0`.
GYRE's ladder gives `f = 0.069921016765114`, and three observed statistics
match that one constant:

| statistic | predicted from `f` | observed |
|---|---:|---:|
| levels 1-29 max over level 30 | `0.075178` | `0.075206` |
| signed-mean ratio, opposite sign | `-0.075178` | `-0.075201` |
| median span over peak | `1.075177` | `1.075204` |

So the measured field is **one bad number per column, at the deepest wet
level**, and every level above it is that number's barotropic image.  The flat
1-29 rows were the tell and round 31 first read them as unremarkable.

### Item 4 — the pre-solve vector, and the owner

**The transcription is calibrated (Rule 1e).**  Walking the round-29 record
forward through `dynzdf.F90:121-122`, then `dynzdf.F90:149-150`, then
`dynzdf.F90:156-159` reproduces the record's own `uu_Kaa_pre`/`vv_Kaa_pre` with **0 of 17400** and
**0 of 17100** cells unequal, max difference exactly `0`.  Labelled as the
review insisted: this certifies the gate's TRANSCRIPTION of those statements
and nothing downstream.  It is entirely internal to the oracle — legoESM never
appears in it — so it does not establish index or halo alignment with the
model, and it touches none of the solve's own inputs (`avm`, `e3uw(Kmm)`,
`e3u(Kaa)`, `rCdU_bot`, `utauU`, `rDt`).  The one-ulp plant turns it red.

**Round 30 was wrong that legoESM exposes no pre-solve vector.**
`expose_pre_implicit_state` publishes `state_new` immediately before the
implicit solver (`ocean_model_latlon_cgrid.py:9872-9874`, struck in place from lines 7731 to
7733, which round 32 moved) and it carries u and v.  **P4b is REFUTED**: that vector is not NEMO's explicit stage update, and
not by a little — `4.269765124169735e-04` on u, which is the size of the
FIELD, not of a residual.

**P4f is CONFIRMED: it is none of `dyn_zdf`'s three entry boundaries.**

| boundary | NEMO statement | u max abs |
|---|---|---:|
| A, the explicit stage update | `dynzdf.F90:121-122` | `4.269765e-04` |
| B, A minus the barotropic velocity | `dynzdf.F90:149-150` | `8.524937e-04` |
| C, B plus the barotropic bottom stress | `dynzdf.F90:156-159` | `8.535126e-04` |

The model's own field peaks at `4.255609e-04` where the oracle's `uu_b_Kaa`
peaks at `4.2551722767418954e-04` on the same mask.  The model's pre-implicit
vector IS the barotropic velocity plus a deviation of about `4e-07`; NEMO's
has that velocity REMOVED.  Removing it a second time (arm B) doubles the
residual, which is the check.

**THE OWNER: the ORDER of the barotropic correction relative to the solve.**
Read on both sides before it was measured.  legoESM applies NEMO's correction
to the stage-3 velocity BEFORE the implicit solve
(`ocean_model_latlon_cgrid.py:8230-8233` — the line numbers ROUND 32 MOVED,
struck in place from the range round 31 cited, lines 6300 to 6321, which is
rendered without backticks here because a struck citation is not a claim about
current code and the gate is right to refuse it as one; the code at the new
site no longer corrects at all, it defers the closure) and again after it, under
`nemo_stage_mean_imposition`, which GYRE resolves `True`.  NEMO applies it
ONCE, after (`stprk3_stg.F90:437-446`); before the solve it instead SUBTRACTS
`uu_b` (`dynzdf.F90:149-150`).  So the two solves receive vectors differing by
the column constant `mean(A) − uu_b`, and a constant does not pass a solve
whose deepest diagonal carries the drag (`dynzdf.F90:296`).

**P4g CONFIRMED, and then P4i CONFIRMED column by column.**  The constant
peaks at `4.2697651206621503e-04`; the drag factor
`zDt_2·|rCdU_bot(i+1,j)+rCdU_bot(i,j)|/e3u(iku,Kaa)` peaks at
`2.3943348583166887e-03`; their product is `1.022324152002595e-06` against a
back-derived single-cell error of `1.0195230688237005e-06` — 0.3 per cent.
Regressing the two on every column, not at the maxima:

| face | columns | slope through origin | R-squared |
|---|---:|---:|---:|
| u | `580` | `0.99726006226993791` | `0.999999999999996` |
| v | `570` | `0.99726003050683432` | `0.99999999999996492` |

One slope, two independent fields, and a residual of four parts in `1e15`.
The `f` used to invert the rank-1 correction is read from the model's own
geometry and printed: `0.069921016765114`, matching what the reviewer read off
NEMO's ladder parameters (`usrdef_zgr.F90:133-137`).

That is a mechanism with a scaling test, on every column, and it names GYRE's
kt2 blocker.  **NOTHING LANDS ON IT THIS ROUND (P4k, preregistered).**  A
regression is not an ablation, and the change it points at is an ordering
change in the shared WS-RK3 momentum program that needs its own Rule-12
discharge on every card that runs it.  The 0.27 per cent the slope sits from
1.0 is itself structured and unexplained, and it is the first thing round 32
should look at.

**The first non-bit statement in `dynzdf.F90` is: NONE OF THEM.**  That was
the question this item was handed, and the answer is that `dyn_zdf`'s
statements are never reached with NEMO's inputs, because legoESM enters the
routine with a different vector.  Naming a statement inside it would have been
the failure this campaign's rules exist to stop.

### Item 5 — ORCA2 cannot discharge round 30's Rule-12 row, and here is what would

Worked in a disposable detached worktree of the round-27 ORCA2 probe under
`/tmp` (`/tmp/r31-orca2-scratch` at `c6f3e33f063b`, nothing committed there,
flagged for removal).  Round 30's fix commit applies cleanly there —
`git apply --check` exits 0 — and it was NOT applied, because the answer makes
it moot.

**legoESM cannot evaluate the changed operator on ORCA2 at all.**  The card
selects `nemo_div_curl` with `A_h = 1e5`, and that branch refuses a tripolar
grid: `ocean_pe_latlon_cgrid.py:3551-3554` raises unless the grid carries a
positive scalar `dlon`, and ORCA2's is `0.0`.  That is why the existing ORCA2
phase-2 gate zeroes `A_h` in its discarded tendency.  A second gap sits behind
it: ORCA2 resolves `nn_ahm_ijk_t = -30`, i.e. `ahmt_3d`/`ahmf_3d` read from
`eddy_viscosity_3D.nc`, where legoESM synthesises a 1-D profile.

**And no ORCA2 record could score it even if it ran.**  All 101 `oracle_*`
streams in the phase-2 root were enumerated.  Every momentum-`Krhs` record is
stage-2-gated or pre-stage: `oracle_rhs_kt00000001.bin` is written before the
three stage calls, the `rkstage2_*` family is inside `IF( kstg == 2 )`, and
`oracle_stage_kt*_s3.bin` is the post-`dyn_zdf`, post-correction Kaa STATE, an
endpoint rather than an operand.  ORCA2's `MY_SRC` has no `dynzdf.F90`, so
there is no twin of GYRE's round-29 pair.

So the row is **UNMEASURED**, and the review's suggestion — discharge it on
GYRE at kt>=2, where the velocity is nonzero and the operator does run — does
not work either: it needs an oracle value to compare against, and the only
pre-/post-`dyn_ldf` frames that exist are at kt=1, where the term is zero.
That is round 30's open question 0, unchanged.

**What would discharge it**, in order: tripolar support in `nemo_div_curl`; a
file-read `ahmt_3d`/`ahmf_3d`; and one new WRITE-only frame.  The frame's
specification, from the scratch lane: insert immediately BEFORE `CALL dyn_ldf`
in `cfgs/ORCA2_OMIP_L4/MY_SRC/stprk3_stg.F90`, guarded
`IF( lwp .AND. kstp == nit000 )`, writing `oracle_rkstage3_preldf_*.bin` with
magic `NEMO_L2_RKPLD_1` and header `(1, kstp, kstg, Kbb, Kmm, Krhs, Kaa, jpi,
jpj, jpk, STORAGE_SIZE(1._wp))`, payload `uu/vv(:,:,:,Krhs)` through this
lane's canonicalising wrapper rather than raw, because its halo and tripolar
conventions differ from GYRE's.  A second frame after `dyn_ldf` must also
carry the operands: `uu/vv(:,:,:,Kbb)`, `e3t/e3u/e3v(:,:,:,Kbb)`,
`e3f(:,:,:)` and `e3u/e3v(:,:,:,Kmm)` — the three distinct thicknesses
`dynldf_lev_rot_scheme.h90:24-25` and
`dynldf_lev_rot_scheme.h90:28-29` read plus the `/e3u(...,Kmm)` divisor.
Under `key_qco` those are macros, so each must be materialised into a buffer
before the `WRITE`.

**The three-level thickness operand (round 30's open question 0) is
measurable on ORCA2 and NOT on GYRE**, and that is now quantified rather than
asserted: ORCA2's horizontal `e3` spread within a level is nonzero on 27 of 30
levels, from about 1.3 m partial cells up to the full 500 m reference; GYRE's
is nonzero on 0 of 30.  So legoESM's single-thickness argument is invisible on
GYRE by construction, exactly as both round-30 reviewers said.

### Item 6 — decision 16, re-filed as an open card-identity gap

CLOSED as "re-file", with round 30's wording struck.  NEMO calls `dyn_vor`
UNCONDITIONALLY on the tanks — `stp2d.F90:146` has no `IF`, and the banner
above it at `:144-145` names the two things the one call serves, `COR + MET`
in flux form and `VOR` in vector-invariant form.  On LOCK and OVERFLOW the
call adds nothing because `f = 0` and the grid is uniform, so the metric term
vanishes too; that is a property of those decks, not "by construction", and
round 30's "premise REFUTED" overstated it.

legoESM's flux-form cards apply Coriolis as a Matsuno-split velocity rotation
outside the RHS, with no metric term at all.  No target card runs flux form
WITH rotation — GYRE is vector-invariant, and so are ORCA2 and the ORCA1 deck
— so nothing is landed and nothing needs to be.  Registered as an open
card-identity gap with its boundary named: **legoESM's flux-form momentum
program has no metric term and applies Coriolis outside the RHS; it is
value-inert on every card the campaign scores today, and it would be wrong on
the first flux-form card with rotation.**

### Item 7 — decision 17 is CONFIRMED, and it is a guard-ordering defect

The claim was that the four remaining `test_nemo_recipe` failures have one
cause.  Measured: all four —
`test_nemo_gyre_coordinate_is_consistent_clean_w_bc`,
`test_nemo_gyre_forced_trajectory_is_finite_and_stable`,
`test_nemo_gyre_wind_forcing_sign_chain_end_to_end` and
`test_surface_stress_implicit_wiring` — reach the SAME raise site with the
same stack, `vertical.py:66-71`: "literal NEMO QCO e3t requires
explicit/reference nemo_e3t_0 and is_active".  **One cause, CONFIRMED.**
(ROUND-34 NOTE: the user answered this as decision 17 and the guard's mask half
now sits BELOW the early return, so that combined message no longer exists and
the two halves say their own thing.  The citation is re-anchored to the lines
that carry the surviving `e3t_0` half; the sentence above is left as the round
wrote it, because it is what was measured then.  All four tests pass now.)

Which operand is missing was measured rather than read off: `e3t_0` IS
supplied by the caller; only `is_active` is absent, and `OceanZStarCoordinate`
carries no such field where `OceanPartialCellCoordinate` does.

**But the cause is not the geometry choice round 29 registered.**  The guard
sits ABOVE the function's own linear-free-surface early return
(`vertical.py:76-77`), which returns `e3t_0` and never reads the mask.  The
demo card resolves `linear_free_surface=True`, so it is refused for an operand
its own arm does not consume.  And that resolution is the FAITHFUL one: the
demo card reproduces NEMO's `GYRE_BARE`, whose compile keys are
`key_linssh key_vco_1d key_RK3` (`cpp_GYRE_BARE.fcm:1`) — no `key_qco`, so
NEMO's own `GYRE_BARE` never executes `dom_qco_r3c` either.  The oracle deck
is the other one, `key_qco key_vco_1d3d key_RK3`
(`cpp_GYRE_OMIP_L2_P3_SM.fcm:1`).  `ln_linssh` is not a live namelist variable
in NEMO 5.0.2 at all; every occurrence in the source is commented out.

**The demo card was NOT changed, and neither was the guard.**  Moving a guard
below an early return changes which cards a shared physics path accepts, which
is on the stop-and-ask list.  The one-line change is named here and is round
32's to ask about.

### Rule-11 records

**P4b is REFUTED by its own first measurement.**  It predicted the model's
pre-implicit vector would match NEMO's explicit stage update to `5e-12`.  It is
off by `4.269765124169735e-04`.  The prediction was written from an alignment
read that was wrong, and the size of the miss — the size of the FIELD, not of a
residual — is what said so.

**P3a's REASONING is retracted; its conclusion is not.**  Round 31 argued that
the barotropic correction adds a column-uniform shift, so a non-uniform error
exonerates it.  The correction is rank-1, not additive: `stprk3_stg.F90:440`
builds `zub` from the solve output, so it MAPS a single-level error into a
column-uniform one.  The measurement still rules out an error in `zub` itself,
which is what the conclusion needs, but the stated mechanism was wrong and the
test cannot separate the two deepest-level statements.  Found by the
independent CLAIM review, not by me.

**Round 30's "legoESM exposes no pre-solve vector" is RETRACTED.**  It does;
`expose_pre_implicit_state` carries u and v.  What is true is that the vector
it exposes is not any of `dyn_zdf`'s entry boundaries, which is a different
and more useful statement.

**Round 30's "the AFTER arm on the committed clean tree" is WITHDRAWN**, and
its four-row worsened register is CORRECTED to eight.  Both struck in place
above.

**The 17x prediction gap is relabelled a second time.**  Round 31 first called
it "bounds, does not confirm"; the review is right that even that is too
strong, because the prediction was a point estimate and not an upper bound by
construction.  It is a REFUTATION PENDING EXPLANATION, and a `1.7e-10` move
against a `9.5e-07` error confirms nothing in either direction.

### Round-31 independent review — two reviewers, both productive

Codex is unavailable on this account, so the mandatory DUAL adversarial review
ran as two fresh independent agents, one on the DIFF and one on the CLAIM,
briefed separately and neither shown the other's answer.  Both found real
defects and neither found the other's.

**The CLAIM reviewer refuted item 3's mechanism** and replaced it with the
rank-1 reading above, matching three observed statistics to four significant
figures on one geometric constant read off NEMO's own ladder.  It also
relabelled the calibration ("this certifies a transcription, not an
alignment") and the 17x gap, and both relabellings are adopted verbatim.  It
proposed discharging Rule 12 for the `dyn_ldf` fix on GYRE at kt>=2; that is
answered in item 5 — there is no oracle frame at kt>=2 to compare against.

**The DIFF reviewer found three wrong-number defects in the new gate**, each
fixed with the control that catches it: a column whose error is exactly zero
was counted COLUMN-UNIFORM, so on a nearly exact field every already-correct
column voted for the barotropic owner; a NaN column read as uniform too while
`nanmax` hid it, so a blow-up would have become evidence; and the
`depth_profile` plant perturbed one face while the verdict ANDs over two, so
the control's own assertion depended on the hypothesis under test.  It also
found that the stamp named the tree owning `provenance.py` rather than the
tree the CALLER's source came from — the round-30 failure wearing a stamp —
and that wiring the stamp in had silently killed the `--allow-dirty` flag on
eight gates that emit their report AFTER a model run.  Two of the new tests
were source-text greps that stayed green when the verdict ladder was inverted;
the ladder is a function now and the test exercises it.  It independently
verified the reconstruction arithmetic against the real record statement by
statement, including that dropping `dynzdf.F90:149-150` leaves 17400 of 17400
cells unequal and dropping `dynzdf.F90:156-159` leaves exactly 580 — the deepest wet cells.

Two latent assumptions it verified safe and this round then GUARDED rather
than relied on: the explicit update writes all `jpk` levels where NEMO writes
`1..jpkm1`, and `np.roll` wraps where NEMO reads a real neighbour.  Both now
refuse a record that would make them bite.

### Tests

`tests/ocean/fidelity/test_nemo_testcase_worktree_stamp.py` (10) and
`tests/ocean/fidelity/test_nemo_testcase_l2_gyre_round31_zdf_walk.py` (20):
**30 passed**.  Fifteen of the thirty are synthetic-violation arms — a report
dict with the stamp deleted, a dirty tree, the escape, a detached tree, a
directory git cannot describe, a caller outside the stamped tree, a mask wet
above `jpkm1`, a tile reaching the array edge, each of the three walked
statements zeroed in turn, a bottom stress moved off `mbku`, a zero-error
column, a NaN column, and both ends of the verdict ladder.

Every mode's plant control fires and exits non-zero: `calibrate` on a one-ulp
move (`1/17400` unequal), `depth_profile` on a column-uniform plant applied to
BOTH faces (the verdict flips to `BAROTROPIC-CORRECTION CANDIDATE`, which is
the proof that the discriminator can say the other thing), and
`ordering_regression` on a zeroed slope (`NO OWNER NAMED`).

The citation gate is `PASS` over 131 citations with 0 failures and 0 map
entries failing audit.  Seven of this round's own new map entries were wrong
when first written — three off by one to four lines, one anchored on a symbol
occurring seven times, one pointing at a `MY_SRC` copy that does not exist, and
two whose extent disagreed with the range — and the map audit caught every
one before the receipt was written.  It also caught the round-29 rebinding
defect twice more: a bare `:N` after a different backticked filename bound to
that filename, so a `dynzdf` statement was auditing `provenance.py`.

Focused run across the stamp, this round's walk, the citation gate and the
round-29 and round-30 gates: **68 passed**.
`tests/ocean/unit/test_nemo_recipe.py` is `4 failed, 20 passed`, the same four
round 29 recorded and for the same reason, now measured (item 7).

### Merge readiness

`03c6e8d96ff7` remains an ancestor of this branch, so the integration is still
a FAST-FORWARD with zero conflicts by construction.  What blocks it after this
round:

1. **GYRE's kt2 `T`/`S`/`u`/`v` rows still fail**, unchanged — `u` at
   `9.482e-07` on `17400 / 17400` faces, `v` at `8.949e-07`.  Round 31 lands
   NO model numerics, so nothing about the blocker moved.  What changed is that
   the blocker now has a named, sized, column-wise-confirmed owner.
2. **Round 30's `dyn_ldf` fix carries an OPEN Rule-12 eligibility.**  It is
   discharged on no card and cannot be discharged on any card that exists
   today (item 5).  It stays, per Rule 12 — the fix is not reverted for a gap
   in its evidence — but the gap is now stated in the register instead of
   implied.
3. **Eight worsened trajectory rows** from round 30, seven above roundoff, with
   a boundary and no owner.

### Open questions

0. **The 0.27 per cent the ordering regression's slope sits from 1.0.**  It is
   the same to nine digits on both faces, so it is structure, not scatter.
   Cheapest first look: whether it is the `e3u(Kaa)` versus `e3u_0` difference
   in the weighting, or a second-order term of the tridiagonal.
1. **The ordering fix itself.**  Move legoESM's stage-3 barotropic correction
   from before the implicit solve to after it, and subtract `uu_b` before it as
   `dynzdf.F90:149-150` does.  It is one place in the shared WS-RK3 program and
   it needs its own Rule-12 discharge on GYRE, both tanks and ORCA2 — the tanks
   resolve the flags `False`, so the change must be shown inert there.
2. **Round 30's `dyn_ldf` Rule-12 row**, and the three pieces of work item 5
   names as its price.
3. **The lateral operator's THICKNESS operand**, still unmeasurable on GYRE and
   now quantified as measurable on ORCA2 (27 of 30 levels have horizontal `e3`
   spread there, 0 of 30 on GYRE).
4. **The eight worsened trajectory rows.**
5. **Decision 16**, re-filed above as an open card-identity gap.
6. **Decision 17's guard**, which refuses a card for an operand its own arm
   never consumes.  One line; asks first.
7. **The slow forcing's depth average** still uses the min rule where NEMO uses
   the area-weighted mean; `3.06e-08` at kt=2, unchanged.
8. **Rounds 1-24 of this receipt remain UNAUDITED** by the citation gate.

### ASKED / UNASKED

| choice | disposition |
|---|---|
| correct the task's line range 434-445 to `stprk3_stg.F90:437-446` | not a scientific choice; corrected against the shipped source before measuring, and the citation gate already carried the right range |
| stamp every gate report with its worktree, fail-closed on dirt | not a scientific choice; it reuses the existing `git_sha` rather than adding a second stamp, and its one escape records rather than suppresses |
| record DETACHED without refusing it | ASKED by the lanes: the cross-card probe worktrees are deliberately detached, and a clean detached tree hides no `HEAD`/code disagreement |
| refuse a stamp whose caller is outside the stamped tree | not a scientific choice; a reviewer finding, and it is the round-30 failure wearing a stamp |
| bridge eight gates' own `--allow-dirty` flag to the shared stamp | not a scientific choice; wiring the stamp in had silently killed the flag |
| withdraw round 30's arm-provenance sentence rather than restate it | ASKED by the timestamps; the pair is still controlled and that is said |
| do NOT land the ordering fix this round | ASKED, and preregistered as P4k before the number that would have tempted it: a regression is not an ablation |
| do NOT change the demo card or its guard (decision 17) | ASKED; moving a guard below an early return changes which cards a shared physics path accepts |
| do NOT land decision 16 | ASKED; no target card runs flux form with rotation, so there is nothing to land |
| apply round 30's fix in the ORCA2 scratch lane | NOT DONE — `git apply --check` passes, but legoESM cannot evaluate the operator on a tripolar grid at all, so applying it would have measured nothing |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed |

UNASKED list: empty.

Every figure above is pinned in
`manifests/nemo_testcase_l2_gyre_round31.json`.

## Round 32 — the ordering fix lands, and the 0.27 per cent is explained

Round 32 starts from `7fe79083eaef` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round32_preregister.json`, committed at
`4a9b7d732471` before any measurement and before a line of model code moved.
No NEMO executable was run and no NEMO source was edited.  Every artifact is
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round32/` with SHA-256
in `artifacts.sha256`, and every one carries the worktree stamp round 31
built.  One detached probe worktree was used and is flagged:
`/tmp/codex-gyre-r32-before` at `4a9b7d732471`, which carries the BEFORE arm
of the two tank cards on a clean committed tree rather than on a temporarily
reverted one.

### What changed, and why it is one change

NEMO's RK3 stage-3 momentum program is `dyn_zdf` and THEN the barotropic
correction: the call sits at `stprk3_stg.F90:430` and the correction block at
`stprk3_stg.F90:437-446`, below it.  ROUND-33 CORRECTION to that range: line 437 is
the `#endif` of the PSYCLONE allocate guard, not a statement.  The block's
STATEMENTS are `stprk3_stg.F90:439-446` and the two that do the work are
`:440-441` and `:444-445`; the citation map's anchor stays `:437-446` because it
anchors on `#endif` and `END_3D`, which is a different thing from a claim about
which lines execute.  Inside `dyn_zdf` the barotropic mode is
removed first (`dynzdf.F90:150-151`, under the guard at `dynzdf.F90:148`) and
its bottom stress added back explicitly at the deepest wet level
(`dynzdf.F90:156-159`), so the implicit friction acts on the baroclinic
residual only.

legoESM applied that correction TWICE — once to the stage-3 velocity before
the solve and once after — and its own barotropic removal subtracted a depth
mean rebuilt from the solve input instead of the prognostic `uu_b`.  Composed:
legoESM handed the solve `A − mean(A)` where NEMO hands it `A − uu_b`, with
`A` the explicit stage update.  Both sides then force the post-solve depth
mean back to `uu_b`, so the whole difference was the column constant
`uu_b − mean(A)` sitting inside a solve whose deepest diagonal carries the
bottom drag (`dynzdf.F90:296`).

Three statements changed and they are one change in one shared program, with
no knob, no card-specific branch and no default that keeps the old behaviour:

1. stage 3 no longer corrects before the solve; it hands the solve the masked
   explicit update and the ladder carries its own correction closure down to
   the post-solve site.  Stages 1 and 2 are untouched — `stprk3_stg.F90:433`
   says "All stages", and no solve runs between a stage update and its
   correction there.
2. the barotropic removal takes the prognostic `uu_b`/`vv_b` whenever the
   state carries the pair, the same rule the stage correction already used for
   its target.
3. the post-solve site applies the deferred correction UNCONDITIONALLY.

### The measured result

`kt=1` stage-3 velocity against NEMO's own stage record, by model level, over
the 580 wet u columns:

| level | before | after |
|---|---:|---:|
| 1 | `7.131145e-08` | `1.764547e-12` |
| 15 | `7.131128e-08` | `1.866456e-13` |
| 29 | `7.097569e-08` | `2.266544e-13` |
| **30** | **`9.482370e-07`** | **`2.249741e-13`** |

The rank-1 bottom-cell signature is gone: level 30 falls by a factor of
`4.2e6`, the whole field peaks at `2.7922718390943624e-12` on u and
`3.3877006018132039e-12` on v, and the residual now peaks at the SURFACE where
the wind enters (`dynzdf.F90:329-330`) instead of at the bottom.  The signed
column mean falls from `1.355e-08` to `-7.13e-15`.

### Preregistered predictions, scored

| prediction | registered threshold | measured | verdict |
|---|---|---:|---|
| P1a level 30 | `< 1.0e-08` | `2.249741e-13` | CONFIRMED |
| P1a levels 1-29 | `< 7.6e-10` | `1.821848e-12` peak, `2.27e-13` mid-column | **PARTLY REFUTED** |
| P1b level 30 stays above `1.0e-07` | — | no | not triggered |
| P1d pre-implicit vector IS `dynzdf.F90:121-122` | `< 1.0e-08` | `4.5974974126122489e-11` (u), `1.2570099171618776e-10` (v) | CONFIRMED |
| P1e the key_qco residual | "near 1e-13" | `4.6e-11` to `1.3e-10` | **REFUTED as to SIZE** |
| P3a registered slope estimator | shortfall within 5 per cent of `2.73994e-03` | `2.3886142509000896e-03`, `12.82` per cent low | NO VERDICT, and **refuted by construction** |
| P3d no `2.56e-09` rank-1 residual survives | — | `2.79e-12` | CONFIRMED |

**P1a is only PARTLY confirmed and that is worth saying.**  The registered
threshold was written for the bottom cell and the bottom cell beat it by five
orders; the levels-1-29 threshold was derived from it by the same `f` ratio
and the surface rows land `2.4x` above it, because after the fix those rows
are no longer that ratio's image at all — they are the surface-stress
residual, a different quantity.  The prediction's arithmetic was right and its
premise expired the moment the fix worked.

### The 0.27 per cent: EXPLAINED, and the registered estimator was wrong

Round 31 left an unexplained shortfall: the ordering regression's slope was
`0.99726006226993791` on u and `0.99726003050683432` on v, identical to nine
digits, where the mechanism predicted `1.0`.

The registered estimator approximated the solve's response by `1/(1+c)` with
`c` the bottom drag factor.  **It could never have confirmed.**  Its value is
a convex combination of `1/(1+c)`, so its shortfall is bounded above by
`c_max = 2.3943348583166887e-03`, and the observed shortfall is
`2.7399377300620900e-03` — larger.  Measured, it lands at
`2.3886142509000896e-03`, `12.82` per cent low, which is NO VERDICT and one
careless step from chasing a "second residual" through the drag operands.
The independent CLAIM review found this before the gate was written.

The EXACT estimator solves NEMO's own tridiagonal, which the round-29 record
carries: the deepest diagonal is not `1+c` but `1+c+η` with
`η = |zwi(bottom)| = 3.284935433818586e-04`, the coupling to the cell above,
`13.7` per cent of `c`.

| face | observed shortfall | exact model | relative miss |
|---|---:|---:|---:|
| u | `2.7399377300620900e-03` | `2.7399455958878605e-03` | `2.87e-06` |
| v | `2.7399694931656800e-03` | `2.7399457330931076e-03` | `8.67e-06` |

Seven significant figures on u.  **The 0.27 per cent was the double
application's own image — the solve's response to the wrongly ordered column
constant — and there is no second residual.**  The independent arm agrees:
after the fix, a surviving rank-1 residual of `2.56e-09` would have said
otherwise, and what survives is `2.79e-12`.

The solver is CALIBRATED before either number is used, and the gate raises
rather than reporting if it is not: reconstructing NEMO's own `uu_Kaa_out`
from NEMO's own `uu_Kaa_pre` plus the surface stress `dynzdf.F90:329-330` adds
inside the recurrence agrees to `5.551115123125783e-17` absolute,
`9.28e-16` relative — one ulp.

### Rule-12 discharge, per card

| card | executes what | discharge |
|---|---|---|
| GYRE-zco | all three statements | **AT-BAR, and NOT bit for bit** (round-33 correction, below).  legoESM's own `rk3_stage_barotropic_correction`, given NEMO's `uu_Kaa_out`, NEMO's `uu_b(Kaa)` and NEMO's `e3u_0` from `mesh_mask.nc` with the column depth REBUILT from it, reproduces NEMO's own stage-3 velocity to `2.168404344971009e-19` (u) and `6.938893903907228e-18` (v) against the `1e-15` bar.  `exact` is **false** on both rows: `1730` of `17400` u cells and `3586` of `17100` v cells are bit-unequal |
| LOCK_EXCHANGE-zco | executes the moved operator (`ln_drgimp = T` at `lock_kt1_10/ocean.output:560`, `ln_dynspg_ts = T` at `:752`) | **UNDISCHARGED** (round-33 correction).  What ran was the TRAJECTORY MOVE gate, not a given-NEMO-inputs discharge: PASS, largest worsening `0.046875` of a 2-ulp bar, first-over-bar unchanged at `kt4`/`u`, 7 of 50 rows differ and all at `1e-19`..`1e-11`.  A passing move gate is not a Rule-12 discharge |
| OVERFLOW-zps | executes the moved operator (`ln_drgimp = T` at `overflow_kt1_10/ocean.output:672`, `ln_dynspg_ts = T` at `:869`) | **UNDISCHARGED, and the move gate FAILS** (round-33 correction).  What ran was the TRAJECTORY MOVE gate: 11 of 50 rows improved, 2 worsened, 37 unchanged, and it FAILS on its CELL criterion -- 9 cells worsen, the largest by `4.443181933488916e-13` absolute at `kt10`/`u`/cell `550`, which is `2001` row-scale ulps of a 2-ulp bar.  At `kt2` on `u`, `133` cells worsened and `81` improved.  A first-over-bar that did not move is NOT a Rule-12 argument -- Rule 12 does not mention it |
| ORCA2-zps | NEMO executes the pair (`ln_drgimp = T` at `variant_icebergs_off_phase2v_tke_a_10step_np2/ocean.output:1097`, `ln_dynspg_ts = T` at `:1363`) | **UNMEASURED**, and it cannot be measured today: this branch has no ORCA2 card at all, and the probe lineage where ORCA2 lives predates the prognostic `uu_b` there is nothing to swap to |
| DINO — not a campaign card, but it executes the operand swap | `zdf_baroclinic_only` with the prognostic pair | **INERT, measured**: an operand change of `1.249e-01` m/s at DINO scale moves the solve's output by `3.331e-16` m/s, `2.8e-16` of the field, against a control of `3.505e-02` when the compensation is removed |

The two tanks resolve `zdf_baroclinic_only` and `nemo_stage_mean_imposition`
FALSE, so statements 2 and 3 do not run there and only the placement does.
Their inertness could not be ARGUED — both resolve `ln_zad_Aimp = T`, so
legoESM feeds `implicit_w` into the same momentum tridiagonal, and that
operator does not annihilate a column constant — so it was measured on both,
BEFORE and AFTER, on clean committed trees.

A pre-existing gap the tank work exposed and this round does NOT close: NEMO
runs `dynzdf.F90:150-151` on both tanks, because both resolve `ln_drgimp` and
`ln_dynspg_ts` true, and legoESM does not, because both cards resolve
`zdf_baroclinic_only` FALSE.  With `rCdU_bot` identically zero there the
statement removes a constant and adds nothing back, so it is not obviously
inert under `ln_zad_Aimp`.  Turning that flag on for two certified cards is a
configuration choice and it is in the ASKED table, not taken.

### The trajectory, and the worsened rows

GYRE `kt=1..10`, all three oracle roots pinned to the V2 root,
`--trajectory-only`, both arms on clean committed trees on this branch:

| row | before | after |
|---|---:|---:|
| `kt2.before.u` | `9.48236979236058e-07` | ``2.792271839094362e-12`` |
| `kt2.before.v` | `8.94866292055948e-07` | ``3.387700601813204e-12`` |
| `kt2.before.T` | `1.3614736849003888e-12` | unchanged |
| `kt2.before.S` | `2.2181101297999213e-14` | unchanged |
| `kt2.before.ssh` | `4.336808689942018e-19` | unchanged |

**`kt2` does NOT clear the bar and the first-over-bar does not move.**  It is
still `kt=2` on `T`, `S`, `u` and `v`.  What changed is which quantity binds:
`v` at ``3.387700601813204e-12`` is now the largest `kt2` row, `u` next, then `T` at
`1.36e-12` — and `T` and `S` did not move by a single bit, so their owner is
untouched by this round and is now within a factor of `2.5` of the velocity's.

**The shared oracle-relative move gate FAILS on GYRE**, and that is said here
rather than left in the artifact: ``FAIL`, 70 certified rows compared, largest cell worsening `1.063192104e+10` row-scale float64 ulps against a 2-ulp bar; the first-over-bar step did NOT move earlier, which is the gate's other criterion`.  Under Rule 12's
compensating-error clause the fix stays, the worsened rows enter the register
as debt naming their boundary, and the next round walks them.

**22 of the 50 rows worsened, 17 of them above the 1e-6 relative threshold round 31 fixed; 20 improved and 8 are unchanged.**

| row | before | after | relative |
|---|---:|---:|---:|
| `kt7.before.ssh` | `6.646099e-06` | `6.749100e-06` | `+1.55 %` |
| `kt5.before.ssh` | `9.360615e-07` | `9.475137e-07` | `+1.223 %` |
| `kt6.before.ssh` | `3.227837e-06` | `3.265152e-06` | `+1.156 %` |
| `kt8.before.ssh` | `1.172228e-05` | `1.181515e-05` | `+0.7922 %` |
| `kt4.before.ssh` | `5.294946e-07` | `5.299464e-07` | `+0.08532 %` |
| `kt10.before.ssh` | `3.328488e-05` | `3.330666e-05` | `+0.06541 %` |
| `kt4.before.u` | `7.078542e-03` | `7.078770e-03` | `+0.003235 %` |
| `kt10.before.v` | `8.868411e-03` | `8.868672e-03` | `+0.002946 %` |
| `kt3.before.u` | `7.193078e-04` | `7.193256e-04` | `+0.002463 %` |
| `kt3.before.v` | `8.606478e-04` | `8.606654e-04` | `+0.00204 %` |
| `kt6.before.u` | `2.776558e-02` | `2.776573e-02` | `+0.0005441 %` |
| `kt7.before.u` | `3.604430e-02` | `3.604447e-02` | `+0.0004901 %` |
| `kt6.before.v` | `6.209240e-02` | `6.209267e-02` | `+0.0004236 %` |
| `kt5.before.u` | `9.209029e-03` | `9.209065e-03` | `+0.0003946 %` |
| `kt5.before.v` | `4.572142e-02` | `4.572158e-02` | `+0.0003645 %` |
| `kt8.before.u` | `4.300540e-02` | `4.300556e-02` | `+0.0003621 %` |
| `kt9.before.u` | `4.806947e-02` | `4.806954e-02` | `+0.0001359 %` |
| `kt7.before.v` | `6.143942e-02` | `6.143945e-02` | `+6.044e-05 %` |
| `kt4.before.T` | `1.066213e-03` | `1.066213e-03` | `+1.648e-05 %` |
| `kt4.before.S` | `8.238303e-05` | `8.238303e-05` | `+5.065e-06 %` |
| `kt3.before.T` | `3.722344e-04` | `3.722344e-04` | `+9.336e-07 %` |
| `kt5.before.T` | `3.059926e-03` | `3.059926e-03` | `+7.775e-07 %` |

**BOUNDARY, stated once for all of them.**  Every worsened row is at `kt >= 3`,
downstream of the first divergence, where NEITHER arm has an exact entering
prefix — the gate's own `exact_prefix_entering` is true only at `kt1` and
`kt2`.  No row at or before the first divergence worsened.  **OWNER**: the
`kt2` residual itself, whose size fell by five orders but whose STRUCTURE
changed completely (bottom-peaked rank-1 before, surface-peaked after), so the
downstream trajectories are not comparable term by term.  That is a stronger
statement than round 30's "owner UNKNOWN" and a weaker one than an
attribution.

**Round 30's eight worsened rows.**  `This round moved 8 of them; 1 improved and 7 worsened further.  `kt8.before.v` `6.203419e-02` -> `6.203416e-02`; `kt4.before.ssh` `5.294946e-07` -> `5.299464e-07`; `kt5.before.v` `4.572142e-02` -> `4.572158e-02`; `kt6.before.v` `6.209240e-02` -> `6.209267e-02`; `kt4.before.T` `1.066213e-03` -> `1.066213e-03`; `kt5.before.T` `3.059926e-03` -> `3.059926e-03`; `kt4.before.S` `8.238303e-05` -> `8.238303e-05`; `kt3.before.T` `3.722344e-04` -> `3.722344e-04`.  None of them is at or before the first divergence, so this round's boundary statement covers them too.`

### Rule-11 records

**P3a's registered estimator is REFUTED BY CONSTRUCTION, not merely
unconfirmed.**  Its shortfall is bounded by `c_max = 2.394e-03` and the target
is `2.740e-03`, so no data could have made it confirm.  It was written from
the diagonal alone and the matrix's bottom off-diagonal is `13.7` per cent of
the diagonal's drag term.  Found by the independent CLAIM review BEFORE the
gate existed; the gate reports it against its registered window anyway,
because a preregistration only reported when it passes is not one.

**P1e's SIZE estimate is REFUTED.**  It said the key_qco arm difference would
sit "near `1e-13`" of the field.  GYRE is flat-bottomed, so the column-to-column
spread in the free-surface factor is `1.233e-06` relative — an ssh structure of
about `5.3e-03` m, not the `4e-07` m that estimate needed.  Measured after the
fix, the dyn_zdf entry residual is `4.6e-11` to `1.3e-10`.  The residual
itself is registered, not fixed, and it is now the sized candidate for the
next round.

**A gated post-solve site would have DELETED the correction on both tanks.**
The first implementation of this round applied the deferred correction under
`nemo_stage_mean_imposition`, which the two tank cards resolve False.  An
independent review measured what that costs before it reached a receipt:
`8.37e-03` on OVERFLOW's `u`, `8.7` per cent of the field, and `1.93e-08` on
LOCK's.  The site is unconditional for that reason.

**The first placement test caught only a verbatim restore.**  The independent
DIFF review broke it four ways that all passed — re-gating the apply site,
restoring the pre-solve correction through a renamed local, swapping the u and
v targets, and reverting the `uu_b` operand.  The checker now reads five
structural facts and each of those four mutations is a test that must go red.

**Round 31's open question 0 is CLOSED**, and its answer is the one round 31
guessed at but could not show: the 0.27 per cent was the solve's own response.

**A sign slip, in the test and not in the code.**  The compensated-pair
invariant was first written with the bottom term subtracted where it must be
added for a positive shift; its own non-vacuity arm caught it.  Recorded
because a test that agrees with the code for the wrong reason is the thing
these arms exist to stop.

### Independent review — two fresh agents, both productive, neither shown the other

Codex is unavailable on this account, so the mandatory DUAL adversarial review
ran as two fresh independent agents: one on the CLAIM, before the code
existed, and one on the DIFF, after.

**The CLAIM reviewer refuted the registered estimator by arithmetic** and
supplied the correction that closed the question, computed the true corner
response from the record's own matrix, and independently reconstructed the
defect offline to `0.017` per cent.  It also called P1a's threshold too loose
(it sits above P3d's own tripwire — true, and the measurement clears the
stricter one by four orders), and P1d circular (partly true: P1d localises the
boundary and sizes the next owner, but it is not independent evidence for
P1a).  One of its findings is NOT adopted: it argued statements 2 and 3 are
both exact no-ops on this card.  Statement 2 is, to roundoff, and that is now
measured on DINO as well.  Statement 3 is not: with it reverted the final
stage-3 depth mean would be `mean(A)` rather than `uu_b`, and `|mean(A) − uu_b|`
is the ordering constant already measured at `4.2697651206621503e-04` — four
orders ABOVE the defect this round fixed.  That is arithmetic, not a
measurement, and it is labelled as such.

**The DIFF reviewer found four holes in the placement test and two latent
defects in the diff**: the cyclic wrap column, which used to travel with the
correction and was left applied to the uncorrected vector, and the missing
shape validation on a function exported specifically for gates to drive with
oracle arrays.  Both fixed.  It also verified independently that the
extraction is bit-identical, that `rule12_correction` is not vacuous (it
measured the no-correction residual at `7.12e-04` on u against `2.17e-19` with
it), that the layouts and masks match on both sides, and that the correction
is applied exactly once on every path.

### Tests

`tests/ocean/fidelity/test_nemo_testcase_l2_gyre_round32_ordering.py` (20):
**20 passed**.  Eleven of the twenty are synthetic-violation arms — an
add-only "correction" that does not install the target mean, each of the four
mutations the DIFF review used to defeat the first placement checker, a bound
but never applied correction, an inlined copy that stops delegating, a
tridiagonal with one off-diagonal bumped, a transposed layout, both ends of
the verdict ladder, a broadcastable-but-wrong target shape, and the
compensated pair with its bottom term dropped.

Every gate mode's plant control fires and exits non-zero: `slope_model` on an
identity-response solver (both estimators drop to `NO VERDICT`) and
`rule12_correction` on a unit offset (`AT-BAR` -> `DEBT`).  The solver
calibration is not a plant but a precondition: the gate RAISES rather than
reporting if it cannot reproduce NEMO's own solve.

The worktree-stamp ratchet floor moves `62` -> `64` for the two new report
dicts; the scan now sees 38 report-emitting gates.

The full ocean-fidelity suite plus the recipe and duplication units:
**884 passed, 10 failed, 7 skipped**.  Every one of the ten was run again to
decide whether it is this round's:

* four `test_nemo_recipe` failures — the same four round 29 recorded and round
  31 measured to one cause, decision 17's guard, untouched by instruction;
* `test_recipe_case_board.py::test_every_oracle_comparison_has_a_row` and
  `test_nemo_testcase_phase3_stage_sweep_gate.py::test_planted_stage_control_exits_nonzero_end_to_end`
  — both reproduced on the round-32 BEFORE probe worktree at `4a9b7d732471`,
  so both are PRE-EXISTING and neither is this round's;
* four `test_nemo_testcase_receipt_citation_gate` and worktree-stamp failures
  that were artefacts of running a 43-minute suite across a tree that was
  being edited: on a clean tree the stamp tests pass, and the citation gate's
  three failures were REAL and are fixed here.

**The citation gate caught this round's own line shifts**, which is what it is
for: moving the stage-3 correction moved four map anchors in
`ocean_model_latlon_cgrid.py`, and the receipt's round-31 citation of the
pre-solve site no longer resolved.  Re-anchored, and the round-31 sentence
that named it is struck in place below rather than silently repointed.  Five
new map entries were added for this round's citations, and one of them was
WRONG when first written — line 118 of `dynzdf.F90` is a bare comment and the
arm selector is `dynzdf.F90:119` — caught before the receipt was committed.

### Merge readiness

`03c6e8d96ff7` remains an ancestor of this branch, so the integration is still
a FAST-FORWARD with zero conflicts by construction.  What blocks it after this
round:

1. **GYRE's `kt2` `T`/`S`/`u`/`v` rows still fail.**  The velocity rows
   improved by five orders and the first-over-bar did not move; `T` and `S`
   did not move at all.
2. **Round 30's `dyn_ldf` fix still carries an OPEN Rule-12 eligibility**,
   unchanged: discharged on no card and not dischargeable on any card that
   exists today.
3. **ROUND-33 CORRECTION — this round's fix is discharged on GYRE only.**
   The sentence that stood here said "discharged on GYRE, LOCK and DINO"; that
   is withdrawn.  What ran on the two tanks was the oracle-relative TRAJECTORY
   MOVE gate, which is not a given-NEMO-inputs discharge, and it FAILED on
   OVERFLOW (9 cells worsen, largest `4.443181933488916e-13` at `kt10`/`u`/cell
   `550`, `2001` row-scale ulps of a 2-ulp bar).  Both tanks EXECUTE the moved
   operator — `ln_drgimp` and `ln_dynspg_ts` are both true on each — so both
   are Rule-12 relevant and both are UNDISCHARGED.  DINO is not a campaign card
   and its measurement is an inertness measurement, not a discharge.  ORCA2
   stays UNMEASURED.  Under Rule 12 the fix stays; the gap is now stated at its
   real size.  Round 33 writes the acquisition that closes it (item 3 below).
4. **22 worsened trajectory rows**, all at `kt >= 3`, with a boundary and
   a named owner, plus round 30's eight.  The 22 are tabulated above, under
   "The trajectory, and the worsened rows"; 17 of them are above the `1e-6`
   relative threshold round 31 fixed, 20 rows improved and 8 are unchanged.
   (ROUND-33 CORRECTION: this item was a broken template — a whole table had
   been substituted into the middle of the sentence and a literal `_COUNT` left
   in the text.  No number changed; the same 22 rows are meant, and they are
   listed once, above, instead of twice.)

### Open questions

0. **The `dyn_zdf` entry residual, `4.6e-11` (u) / `1.3e-10` (v)**, newly
   exposed and newly sized.  The leading candidate is named and was registered
   before the fix ran: legoESM's stage-3 explicit update carries the key_qco
   `(1 + r3u)` ratios, while GYRE's deck resolves `ln_dynadv_vec = T`
   (`round19_oracle_v2_external/ocean.output:798`) so `dynzdf.F90:119` takes the velocity arm at
   `dynzdf.F90:121-122`, which carries none.  Not fixed this round — it is a
   scheme-arm question, and changing which arm the stage takes is a
   configuration choice.
1. **`kt2`'s `T` residual, `1.36e-12`, and `S`'s `2.22e-14`**, both bit-for-bit
   unchanged by this round and therefore owned by something else entirely.
2. **Round 30's `dyn_ldf` Rule-12 row** and the three pieces of work round 31
   priced for it.
3. **ORCA2 has no card on this branch.**  Round 31 priced a WRITE-only frame
   for the `dyn_ldf` question; this round's operator needs the same lane, and
   the two-frame specification is recorded in the round-32 manifest.
4. **The worsened trajectory rows**, this round's and round 30's.
5. **Decision 16**, still an open card-identity gap.
6. **Decision 17's guard**, untouched by instruction — it refuses a card for
   an operand its own arm never consumes.  One line; asks first.
7. **The slow forcing's depth average** still uses the min rule where NEMO
   uses the area-weighted mean; `3.06e-08` at `kt=2`, unchanged.
8. **Rounds 1-24 of this receipt remain UNAUDITED** by the citation gate.

### ASKED / UNASKED

| choice | disposition |
|---|---|
| move the stage-3 barotropic correction from before the solve to after it | not a scientific choice; it is what `stprk3_stg.F90:430` and `:437-446` say, and it is the round's assignment |
| subtract the prognostic `uu_b` rather than a rebuilt depth mean | not a scientific choice; `dynzdf.F90:150-151` names the variable |
| apply the deferred correction UNCONDITIONALLY rather than under `nemo_stage_mean_imposition` | not a scientific choice; the alternative deletes the correction on two cards, measured at `8.4e-03` on OVERFLOW |
| extract the correction into the shared barotropic home as a third sibling | not a scientific choice; Rule 12 cannot be discharged on a closure, and the two existing siblings already encode the same association distinction |
| re-apply the cyclic wrap column after the deferred correction | not a scientific choice; it restores an invariant the deferral had dropped, and it is inert on all four cards |
| report the preregistered slope estimator against its window even though it cannot pass | ASKED and answered by the preregistration itself |
| a detached BEFORE probe worktree for the tank arms instead of a temporary revert | ASKED by round 31's own withdrawn provenance sentence |
| do NOT change which `dyn_zdf` arm the stage-3 explicit update takes | ASKED; the key_qco ratio question is a scheme-arm choice and it is open question 0 |
| do NOT enable `zdf_baroclinic_only` on the tanks, where NEMO runs `dynzdf.F90:150-151` and legoESM does not | ASKED; a config-flag change on two certified cards, registered as a finding and left for the user |
| do NOT touch the decision-17 guard or the demo card | ASKED; instructed, and unchanged |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed |

UNASKED list: empty.

Every figure above is pinned in `manifests/nemo_testcase_l2_gyre_round32.json`.

## Round 33 — the stage-3 arm, and what round 32 claimed too strongly

Round 33 starts from `bb6fece7e21a` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round33_preregister.json`, committed at
`956dffbefa9e` before any measurement and before a line of model code moved;
its addendum is `..._round33_preregister_addendum1.json` at `33e3fd02db7b`,
committed after the independent CLAIM review and before the AFTER arm ran.  No
NEMO executable was run and no NEMO source was edited.  Every artifact is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round33/` with SHA-256 in
`artifacts.sha256`, and every one names the tree that produced it.  One
detached probe worktree was used and is flagged: `/tmp/codex-gyre-r33-before`
at `33e3fd02db7b`, which carries the OVERFLOW BEFORE arm on a clean committed
tree rather than on a temporarily reverted one.

### Item 1 — eight corrections the independent review found, and two more

All eight are in `1fb6b57f3f47` and land in the round-32 text above, in place.
The two that came out of this round's own measurements:

**A RECEIPT DEFECT: round 32's `kt2` table mixes two quantities in one
column.**  The trajectory gate's `normalized_max_abs` is
`absolute / max(reference, 1.0)`, so for `u`, `v` and `ssh` -- whose reference
scale is below 1 -- it IS the absolute residual, and for `T` and `S` -- whose
reference scale is `23.47` K and `36.84` psu -- it is the RELATIVE one.  Round
32 therefore compared a dimensionless number with one in m/s when it wrote
that `T` is "within a factor of 2.5 of the velocity's".  The ABSOLUTE `kt2`
residuals are `T` `3.1956659540810506e-11` K and `S`
`8.1712414612411521e-13` psu, so `T` is an order of magnitude LARGER than the
velocity rows, not smaller.  Both quantities are named in this round's tables.

**The round-32 Rule-12 gate re-runs identical with its claim corrected**:
`2.168404344971009e-19` (u) and `6.938893903907228e-18` (v), `exact` false,
`1730` of `17400` and `3586` of `17100` cells bit-unequal, AT-BAR at the
`1e-15` bar.  Its report now carries `divisor_provenance` and an
`inputs_reconstructed_not_nemo` list, because the divisor is rebuilt.

### Item 2 — Rule 0 first: NEMO selects the stage update ONCE, for all three stages

`stprk3_stg.F90:365` opens `IF( ln_dynadv_vec .OR. lk_linssh )`.  Its TRUE arm
at `:366-369` advances the velocity directly, with no thickness ratios; the
ELSE arm at `:370-388` is the thickness-weighted one, `:373-378` under
key_qco.  **Stage 3 is not in that SELECT at all**: `stprk3_stg.F90:395`
CASE(3) adds the leftover RHS terms and does no time stepping, because the
stage-3 time step lives inside `dyn_zdf` -- where `dynzdf.F90:119` carries the
IDENTICAL predicate, vector arm at `dynzdf.F90:121-122` and key_qco arm at
`dynzdf.F90:127-132`.

Which arm GYRE compiles, read from the build rather than argued: its keys are
`key_qco key_vco_1d3d key_RK3` (`cpp_GYRE_OMIP_L2_P3_SM.fcm`), so the ELSE arm
is the key_qco form, and the deck resolves `ln_dynadv_vec = T`
(`round19_oracle_v2_external/ocean.output:798`) with `lk_linssh` `.FALSE.` in
that build's own `dom_oce.f90`.  NEMO takes the VECTOR arm at all three stages
on GYRE.  legoESM honoured that at stages 1 and 2
(`ocean_model_latlon_cgrid.py:7467` and its two branch sites) and hardcoded the
key_qco ratios at stage 3.  A transcription defect, and it had a comment
asserting the opposite.

### The measured result

| prediction | registered window | measured | verdict |
|---|---|---:|---|
| P1a `zdf_entry_A` u | within `1e-6` of `rDt` x the RHS residual | `2.9684798833066247e-12`, `2.396e-11` relative | **CONFIRMED** |
| P1a `zdf_entry_A` v | same | `3.5751281340072215e-12`, `1.694e-10` relative | **CONFIRMED** |
| P1f index sets agree | `n = 17400` / `17100` on both gates | identical | **CONFIRMED** |
| P1g legoESM's kt=1 entry `u0` is identically zero | — | `0.0` on u and v | **CONFIRMED** |
| P2a the stage-3 velocity does NOT fall an order of magnitude | u in `[3.0e-13, 4.5e-12]` | `2.7478405293344943e-12` | **CONFIRMED** |
| P2d the level-1 error stays within `3x` of `1.764547e-12` | `[5.9e-13, 5.3e-12]` | `1.709431e-12` | **CONFIRMED** |
| P2e the model's own field DID move | nonzero and at most `4.6e-11` | `1.013143e-13` on u, `3.001220e-13` on v | **CONFIRMED** |
| P3a the first-over-bar does not move | — | still `kt=2` on `T`, `S`, `u`, `v` | **CONFIRMED** |
| P3b `kt2` `T` and `S` do not move by a bit | — | field move exactly `0.0` over all `18000` cells, both | **CONFIRMED, uninformative by construction** |
| P4a the operator is BIT FOR BIT given NEMO's inputs | `exact`, `n_unequal = 0` | see the Rule-12 table; the discharge was REBUILT after the DIFF review found the first version circular | **CONFIRMED, on the rebuilt discharge** |
| P4b LOCK's 10-step trajectory is bit-identical | every cell | `PASS`, every one of 20 rows' field move exactly `0.0` | **CONFIRMED** |
| P4b OVERFLOW's 10-step trajectory is bit-identical | every cell | `PASS`, every one of 10 rows' field move exactly `0.0` | **CONFIRMED** |

`kt=1` stage-3 velocity against NEMO's own stage record, by model level, over
the 580 wet u columns:

| level | before | after |
|---|---:|---:|
| 1 | `1.764547e-12` | `1.709431e-12` |
| 15 | `1.866456e-13` | `1.682309e-13` |
| 29 | `2.266544e-13` | `2.199583e-13` |
| 30 | `2.249741e-13` | `2.199770e-13` |

**The entry residual falls by `15.5x` on u and `35.2x` on v and the stage-3
OUTPUT barely moves.**  That was the round's counter-intuitive prediction and
it is the interesting number: `4.5974974126122489e-11` to
`2.9684798833066247e-12` at the entry, `2.7922718390943624e-12` to
`2.7478405293344943e-12` at the exit.  What is left at the entry is now
EXACTLY `rDt` times the stage-3 RHS residual -- `14400` x
`2.0614443633579772e-16` = `2.9684798832354876e-12`, measured
`2.9684798833066247e-12` -- and that RHS residual is bit-identical before and
after, so the two arms of the comparison differ in one variable.

### Rule-12 discharge, per card

| card | executes the changed arm | discharge |
|---|---|---|
| GYRE-zco | YES, at stage 3 (`ln_dynadv_vec = T`, `lk_linssh` false) | **BIT FOR BIT against a NEMO OUTPUT ARRAY** -- see "the discharge was circular" below.  legoESM's own `rk3_stage_velocity_update`, given NEMO's `uu(Kbb)`, `uu(Krhs)`, `rDt` and `umask` from the round-29 record, carried forward through NEMO's own `dynzdf.F90:149-150` and `:156-159`, reproduces NEMO's `uu_Kaa_pre` with `exact` true and `0` of `17400` u and `0` of `17100` v cells unequal.  The composition is calibrated inside the gate, which RAISES rather than reporting if the calibration fails |
| LOCK_EXCHANGE-zco | NO.  `ln_dynadv_vec = F` (`lock_kt1_10/ocean.output:705`), `lk_linssh` false in its own build, and legoESM resolves `momentum_advection = flux_form` | **NOT REACHED, and measured**: the `kt=1..10` trajectory is bit-identical, every cell of every one of 20 rows moving exactly `0.0` |
| OVERFLOW-zps | NO, same two reasons (`overflow_kt1_10/ocean.output:822`) | **NOT REACHED, and measured**: the `kt=1..10` trajectory is bit-identical, every cell of every one of 10 rows moving exactly `0.0`, and the move gate PASSES at `0` ulps |
| ORCA2-zps | unknown; no ORCA2 card exists on this branch | **UNMEASURED**, frame spec unchanged |
| DINO | not a campaign card; its deck resolves the vector arm, so it would execute the changed statement | **OPEN**, not claimed; no DINO gate ran |

**THE FIRST VERSION OF THIS DISCHARGE WAS CIRCULAR, and the independent DIFF
review caught it before a receipt quoted it.**  The gate built NEMO's side as
`(uu_Kbb_in + rDt*uu_Krhs_in)*umask` from the record and the candidate as
legoESM's helper evaluating that same expression on the same operands in the
same association order.  Nothing on the "oracle" side was a NEMO OUTPUT ARRAY,
so `exact` was guaranteed by construction and only the artificial plant could
turn it red.  NEMO holds no array at that boundary at all.  The discharge is
now the COMPOSITION: legoESM's helper output carried forward through NEMO's
own `dynzdf.F90:149-150` and `:156-159` and scored against `uu_Kaa_pre`, which
IS a NEMO output.  The gate re-runs `--mode calibrate` inside itself and
RAISES rather than reporting if those two follow-on statements do not
reproduce `uu_Kaa_pre` bit for bit on NEMO's own operands, so the composition
can never be quoted uncalibrated.  The bare row stays, relabelled a
TRANSCRIPTION IDENTITY with `is_a_discharge: false` and a `why_not` string.

**Its remaining blind spot, stated in the report rather than discovered
later**: the record's `uu_Kbb_in` is identically `0.0` -- GYRE is at rest at
kt=1 -- so BOTH rows drive only `(0 + rDt*rhs)*umask`, and a wrong
`velocity_before` operand would pass either.  The independent CLAIM review
found that before the gate was written.  The non-zero before-velocity arm is a
unit test with two plants on it.  The mask row the addendum opened is CLOSED
by the same report: NEMO's `umask` in the scored layout and the mask every
GYRE gate scores on agree on `17400` and `17100` cells with `0`
disagreements.

### The trajectory, and the worsened rows

GYRE `kt=1..10`, all three oracle roots pinned to the V2 root,
`--trajectory-only`, both arms on clean committed trees on this branch.
**`kt2` does NOT clear the bar and the first-over-bar does not move**; it is
still `kt=2` on `T`, `S`, `u` and `v`.

| row | before (abs) | after (abs) | before (norm) | after (norm) |
|---|---:|---:|---:|---:|
| `kt2.before.T` | `3.1956659540810506e-11` | unchanged, bit for bit | `1.361474e-12` | unchanged |
| `kt2.before.S` | `8.1712414612411521e-13` | unchanged, bit for bit | `2.218110e-14` | unchanged |
| `kt2.before.u` | `2.7922718390943624e-12` | `2.7478405293344943e-12` | same | same |
| `kt2.before.v` | `3.3877006018132039e-12` | `3.3055602526033123e-12` | same | same |
| `kt2.before.ssh` | `4.336808689942018e-19` | unchanged, bit for bit | same | same |

**`T` and `S` did not move by a single bit and that is now measured cell by
cell, not inferred**: the shared oracle-relative comparison records a
`max_previous_legoesm_field_move` of exactly `0.0` over all `18000` cells on
each.  Round 32 moved `kt=1`'s FINAL `u` by `9.482369518886303e-07` and left
the same two rows at exactly `0.0` as well.  So **`kt=2`'s `T` and `S` carry no
path from `kt=1`'s final velocity at these sizes**, which is what the code
says: the stage-3 tracer step consumes the STAGE-2 transport `_g2`, built from
`(u2_corr, v2_corr)`.  Their next candidate, registered before this round
measured anything: score `kt=1`'s `T` and `S` at stages 1, 2 and 3 against the
round-29 root's `oracle_rktracer_stage3_kt00000001.bin` and the round-21 stage
records and report the FIRST stage over bar.  A stage-1 or stage-2 failure
exonerates everything downstream of it.  Nothing this round or round 32 touched
can reach them.

**The shared oracle-relative move gate FAILS on GYRE**, and that is said here
rather than left in the artifact: `FAIL`, 70 certified rows compared, largest
cell worsening `6.667640e-08` absolute at `kt10`/`u` = `3.003e+08` row-scale
float64 ulps against a 2-ulp bar; no row changed status and the first-over-bar
did not move.  Round 32's figure was `1.063192104e+10` ulps, so the same
criterion is `35x` less badly violated, which is a direction and not a pass.
Under Rule 12 the fix stays -- it is discharged BIT FOR BIT on the only card
that executes it -- and the worsened rows enter the register as debt.

**24 of the 50 rows worsened, 5 of them above the `1e-6` relative threshold;
17 improved and 9 are unchanged.**  Round 32's counts were 22, 17 and 20/8.

| row | before | after | relative |
|---|---:|---:|---:|
| `kt3.before.v` | `8.606654e-04` | `8.606873e-04` | `+0.002545 %` |
| `kt3.before.u` | `7.193256e-04` | `7.193412e-04` | `+0.00218 %` |
| `kt9.before.ssh` | `2.167026e-05` | `2.167050e-05` | `+0.001123 %` |
| `kt8.before.ssh` | `1.181515e-05` | `1.181524e-05` | `+0.0007811 %` |
| `kt10.before.ssh` | `3.330666e-05` | `3.330685e-05` | `+0.0005915 %` |

The other 19 worsened rows are all below `1e-4` per cent and are listed in
`round33/after_trajectory.json` against `before_trajectory.json`.

**BOUNDARY, stated once for all of them.**  Every worsened row is at
`kt >= 3`, downstream of the first divergence, where neither arm has an exact
entering prefix -- the gate's own `exact_prefix_entering` is true only at `kt1`
and `kt2`.  No row at or before the first divergence worsened; `kt2`'s two
velocity rows IMPROVED and its three others did not move.  **OWNER**: the
`kt2` residual itself, which is now `2.75e-12` on `u` against a `T` residual of
`3.20e-11` -- so the tracer rows are an order of magnitude larger than the
velocity rows in absolute terms and are untouched by this round.

**Round 32's 22 worsened rows.**  This round moved all 22: 14 worsened further
and 8 improved, none unchanged.

### Item 3 — the tanks' acquisition, written and NOT run

Round 32's ordering fix is UNDISCHARGED on both tanks (item 1(a)).  The
discharge needs `puu(:,:,:,Kaa)` as `dyn_zdf` leaves it and `uu_b(:,:,Kaa)`,
which NEMO holds only inside `dyn_zdf` and never writes.  `run.sh` builds a
config COPY carrying GYRE's round-29 instrument -- the SAME patch file, because
both tanks compile the same shipped `dynzdf.F90` body, checked rather than
assumed -- plus ONE round-33 addition that dumps NEMO's own `e3u_0`, `hu_0` and
`r1_hu_0`.

That addition is not tidiness.  GYRE's discharge REBUILDS the column divisor
because `mesh_mask.nc` has no `hu_0`; on LOCK it could not even rebuild it,
because LOCK's `mesh_mask.nc` carries no `e3*_0` at all -- under `key_vco_1d`
the vertical coordinate IS the 1-D ladder, and `domzgr_substitute.h90:89`
DEFINES `e3u_0(i,j,k)` to be `e3t_1d(k)` (`:98` defines it as `e3u_3d(i,j,k)`
under `key_vco_3d`).  So the tank discharge reconstructs nothing, which makes
it a stronger discharge than GYRE's is today.

**A retraction while writing its test.**  The first version asserted NEMO's
`r1_hu_0` must differ from `1/hu_0`, because `domain.F90:159` builds it as
`ssumask/(hu_0 + 1 - ssumask)`.  Measured on both tanks that difference is
EXACTLY zero -- their column depths are small exactly-representable values.
The row that survives is the one that matters and it is measured on the right
quantity: NEMO forms `SUM(e3u_0*uu(Kaa)) * r1_hu_0` where the operator forms
the same sum DIVIDED by `hu_0`, and `x/h` is not `x*(1/h)` even when `1/h` is
correctly rounded.  On the synthetic LOCK fixture that difference is
`3.42e-49`.  It stays OPEN.

The reader FAILS CLOSED: until the acquisition has run it exits non-zero naming
the command that creates the record, and it refuses a round-29-only record
rather than quietly rebuilding the divisor.  It states what it cannot see: the
tanks' stage record carries `T`, `S`, `u` and `ssh` and no `v`, so only the u
face is dischargeable there.

**The exact commands, to be run by the user.**  The agent wrote them and
stopped; no `makenemo`, no `mpirun`, no NEMO executable was run this round.

```
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tanks_round33_zdf/run.sh LOCK_EXCHANGE
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tanks_round33_zdf/run.sh OVERFLOW
```

Each builds `tests/<CARD>_OMIP_L1_P3_R33ZDF`, runs it into
`round33_<card>_zdf_matrix`, checks the twin byte-identity, and then runs the
discharge and its plant itself.  Afterwards the discharge alone is one command
per card:

```
python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tanks_round33_zdf_rule12.py --card LOCK_EXCHANGE
python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tanks_round33_zdf_rule12.py --card OVERFLOW
```

### Independent review — two fresh agents, both productive, neither shown the other

Codex is unavailable on this account, so the mandatory DUAL adversarial review
ran as two fresh independent agents: one on the CLAIM, before a line of model
code moved, and one on the DIFF, after.

**The CLAIM reviewer verified every citation line for line** -- the selector
at `stprk3_stg.F90:365`, CASE(3) at `:395` doing no stepping, the same
selector at `dynzdf.F90:119`, and `lk_linssh = .FALSE.` in all three builds'
own `dom_oce.f90` -- and it verified there is no omitted operator between the
RHS boundary and the stage update on GYRE (`zdf_drg_exp` is skipped under
`ln_drgimp = T`, `ln_zdfosm` and `ln_bdy` are both false).  Then it broke
three things.  It REFUTED P2a's mechanism by measuring that today's stage-3
peak is already `0.9406` of `rDt` times the RHS residual on u and `0.9476` on
v -- one mechanism explains both, and the registered "6.1 per cent baroclinic
fraction" is a back-fit that gives `6.07` per cent on u and `2.70` on v.  It
showed P2a could then not fail, which bought the P2e companion.  It showed
P4a was half vacuous because `uu_Kbb_in` is identically zero.  And it withdrew
P1's `+/-50` per cent window as unjustified, which is why the confirmation
above is quoted at `2.4e-11` relative instead of "within the window".

**The DIFF reviewer found two things that would have shipped.**  It measured
that the Rule-12 discharge was CIRCULAR -- both sides the same expression on
the same operands -- and that is why the discharge is now the composition
against `uu_Kaa_pre`.  And it defeated the structural checker with five
mutations, one of which (`_vector_velocity_stage_update = False`, one inserted
line) replants the entire defect this round fixed.  All five are tests now and
all five go red.  It also confirmed, by measurement, that all five refactored
arms are bitwise identical to the old inline expressions and that association
genuinely matters here (`((dt*q)*r)` differs from `(dt*(q*r))` by `5.55e-17`),
that no private hook regresses on the new shape validation, and that the
Fortran patch's six new labels are all exactly sixteen characters -- a short
one desynchronises the self-describing stream.

Its four smaller findings, adopted: one tautological assertion dropped; the
stale-comment test now says out loud that it would pass with the code fully
reverted; and two `run.sh` weaknesses are recorded rather than silently kept
-- a `sha256sum -c` that cannot fail because it checks a manifest generated
from the same files one line earlier, and an add-only check whose `^-[^-]`
pattern misses the deletion of a BLANK line.  Both are fixed.

NOT adopted, and why: it argued the tank reader's byte layout is only READ
against a Python re-implementation of the patch and never measured.  True, and
it cannot be measured until the acquisition runs -- that is exactly why the
reader fails closed and why the sixteen-character check exists.  Recorded as a
limitation of item 3, not as a defect in it.

### Rule-11 records

**P2a's registered mechanism is RETRACTED**, and it was retracted in an
addendum committed BEFORE the AFTER arm ran, not afterwards.  The surviving
kt=1 stage-3 velocity error is not "the qco term's baroclinic remainder"; it
is the `dt`-times-RHS roundoff, and the qco term contributes essentially
nothing to the OUTPUT because `r3u` carries no level index and
`stprk3_stg.F90:440,444-445` removes its column mean.  The prediction that
followed from it -- that the output would barely move -- was right for the
corrected reason.

**P1's `+/-50` per cent window is WITHDRAWN as unjustified.**  With
`uu_Kbb` identically zero the residual is cell-by-cell and there is no sum
anywhere, so the max is `rDt` times the RHS max to a few ulp.  Measured at
`2.4e-11` relative on u.

**The first Rule-12 discharge was CIRCULAR and is RETRACTED**, replaced by the
composition against `uu_Kaa_pre`.  Its remaining hole, stated: the composed
row can be defeated only by compensating errors ACROSS the three transcribed
statements, because the calibration pins the chain rather than each statement
on its own.

**A test claim retracted mid-writing**: that NEMO's `r1_hu_0` must differ from
`1/hu_0`.  On both tanks it is exactly equal.

**Round 32's `kt2` table mixed absolute and normalized quantities**, and its
sentence comparing `T` to the velocity rows is withdrawn.

### Tests

`tests/ocean/fidelity/test_nemo_testcase_l2_gyre_round33_stage_arm.py` (22):
14 are synthetic-violation arms -- stage 3 reverted to the qco form, a stage
pinned to one arm, a swapped stage time level, a wrong timestep, a wrong RHS,
the selector rebound to a constant, an in-place reweighting after the call, a
local shadow of the helper, a dead stage-3 call, swapped entry velocities, a
one-ulp bump and a dropped before-velocity on the operator, a collapsed
selector, and a swapped Kmm/Kaa ratio pair.

`tests/ocean/fidelity/test_nemo_testcase_l1_tanks_round33_zdf_rule12.py` (14):
the tank reader driven END TO END on a synthetic record in the instrument's
exact binary layout, built by INVERTING the operator so the answer must come
back equal to its input; a planted unit offset and a planted 1 nm/s barotropic
target that both turn it red; the round-29-only refusal; the fail-closed path;
the geometry round trip against an independent construction; and a check that
both Fortran patches only ADD, stack cleanly, and label every array with
exactly sixteen characters, plus a non-vacuity arm proving the add-only
pattern can fire on a deleted BLANK line, which is the case the pattern the
review defeated could not see.

These tests stub the worktree stamp.  The stamper REFUSES a dirty tree, which
is right for a gate producing an artifact and wrong for a unit test -- it made
every arithmetic assertion fail for a reason that has nothing to do with the
arithmetic, which is the same accident that cost round 32 four gate results.
The stamp is exercised when the gate really runs; here only its presence in
the report is asserted.

`tests/ocean/fidelity/test_nemo_testcase_l2_gyre_round32_ordering.py` (21):
one new arm for the mask shape validation item 1(h) added.

The three round-32/33 files together: **57 passed** on a clean tree.

**The full `tests/ocean/fidelity/` suite: 910 passed, 3 failed, 7 skipped in
39 min 16 s** -- and the run is CONTAMINATED, which is the more important fact.
Every one of the three was run again in isolation to decide whether it is this
round's:

* `test_recipe_case_board.py::test_every_oracle_comparison_has_a_row` --
  PRE-EXISTING.  Round 32 recorded it; it reproduces on the clean tree, and the
  four drivers it names have nothing to do with this round.
* `test_nemo_testcase_phase3_stage_sweep_gate.py::test_planted_stage_control_exits_nonzero_end_to_end`
  -- PRE-EXISTING.  Round 32 recorded it, reproduced on its own BEFORE probe
  worktree.
* `test_nemo_testcase_worktree_stamp.py::test_a_dirty_tree_refuses` -- NOT a
  failure of this round's code.  It PASSES on the clean tree, twice.

**WHY THE RUN IS CONTAMINATED, recorded because the round-32 receipt logged the
same accident and it happened again.**  A 39-minute suite spanned a moment when
a tracked file in this worktree -- `nemo_testcase_l2_gyre_round21_admission.py`
-- was rewritten by SOMETHING OTHER THAN THIS ROUND, 391 lines added and 169
removed, at 05:20.  A worktree this round's provenance sentences call
single-writer is not one.  The stamper refuses a dirty tree, which is why two
stamp tests read as failures; both pass on the restored tree.  Nothing else
this round measured is affected: every other artifact was produced before
05:20, and the round's headline numbers were taken between 03:56 and 04:47.
A re-run on the clean tree is logged as `round33_fidelity_suite_rerun.log`.

What is measured beyond that: the 57 above, the stamp ratchet's 10, and the
citation gate, each run directly on a clean tree.

**Gates, each with the line that decided it.**  The receipt citation gate
PASSES on 155 citations with 0 unmapped and 0 failing, and its planted
two-line shift exits non-zero.  The worktree-stamp ratchet floor moves `64` ->
`66` for this round's two new report dicts, and its 10 tests pass including the
non-vacuity arm.  The round-33 Rule-12 discharge exits 0 and its plant exits
non-zero.  The round-32 Rule-12 gate re-runs unchanged with its corrected
claim string.

**The citation gate caught this round's own line shifts**, which is what it is
for: inserting `rk3_stage_velocity_update` moved five anchors in
`ocean_model_latlon_cgrid.py` by exactly 99 lines each.  Re-anchored in the map
and repointed in the prose that names them, because the SITE did not move, only
its line number did.  Eleven new entries were added and two of them were WRONG
when first written -- `DO_3D( 0, 0, 0, 0, 1, jpkm1 )` occurs five times in
`stprk3_stg.F90` and a bare `ENDIF` twelve, and both first-guess occurrence
indices resolved to the wrong line.  Caught before the receipt was committed.

### Merge readiness

`03c6e8d96ff7` remains an ancestor of this branch, so the integration is still
a FAST-FORWARD with zero conflicts by construction.  What blocks it after this
round:

1. **GYRE's `kt2` `T`/`S`/`u`/`v` rows still fail.**  The velocity rows improved
   by `1.6` and `2.4` per cent and the first-over-bar did not move.  `T` and
   `S` did not move by a single bit, measured cell by cell, and they are now
   the LARGEST `kt2` rows in absolute terms -- `3.20e-11` K against `2.75e-12`
   m/s -- which round 32's mixed-units table had hidden.
2. **Round 30's `dyn_ldf` fix still carries an OPEN Rule-12 eligibility**,
   unchanged: discharged on no card and not dischargeable on any card that
   exists today.
3. **Round 32's ordering fix is discharged on GYRE ONLY**, and the acquisition
   that would discharge it on the two tanks is written and NOT run.  Two
   commands, above.
4. **This round's fix is discharged BIT FOR BIT on GYRE, the only card that
   executes it**, and measured NOT REACHED on both tanks by a bit-identical
   10-step trajectory.  ORCA2 stays UNMEASURED.
5. **24 worsened trajectory rows**, 5 of them above the `1e-6` relative
   threshold, all at `kt >= 3`, with a boundary and a named owner; plus round
   32's 22, of which 14 worsened further and 8 improved.
6. **The shared oracle-relative move gate still FAILS on GYRE** on its cell
   criterion, at `3.003e+08` row-scale ulps against a 2-ulp bar -- `35x` less
   than round 32's `1.063e+10`, which is a direction and not a pass.

### Open questions

0. **`kt2`'s `T` residual, `3.1956659540810506e-11` K absolute, and `S`'s
   `8.1712414612411521e-13` psu**, both bit-for-bit unchanged by this round AND
   by round 32, and now the largest `kt2` rows.  Their owner is not the
   velocity: measured, a `9.5e-07` change in `kt=1`'s final `u` moved neither
   by a single cell.  The discriminating measurement is registered: score
   `kt=1`'s `T` and `S` at stages 1, 2 and 3 and report the FIRST stage over
   bar.
1. **The `dyn_zdf` entry residual is now `rDt` times the stage-3 RHS
   residual**, `2.9684798833066247e-12` on u.  Closing it means closing the
   RHS residual itself, which is AT-BAR at `2.06e-16` -- i.e. there is no
   entry-side work left at this boundary.
2. **The tanks' Rule-12 discharge for round 32's operator**, one command each
   once the acquisition has run.
3. **Round 30's `dyn_ldf` Rule-12 row** and the three pieces of work round 31
   priced for it.
4. **ORCA2 has no card on this branch.**  The two-frame specification is
   recorded in the round-32 manifest and unchanged.
5. **The worsened trajectory rows**, this round's 24 and round 32's 22.
6. **Decision 16**, still an open card-identity gap.
7. **Decision 17's guard**, untouched by instruction.
8. **The slow forcing's depth average** still uses the min rule where NEMO
   uses the area-weighted mean; `3.06e-08` at `kt=2`, unchanged.
9. **Two association rows on the barotropic correction's divisor**: the
   operator DIVIDES by a column depth where NEMO MULTIPLIES by the precomputed
   `r1_hu_0` (`domain.F90:159`), and on GYRE that depth is REBUILT because
   `mesh_mask.nc` carries no `hu_0`.  The tank records will carry NEMO's own
   `hu_0` and `r1_hu_0`, so the tank discharge reconstructs nothing; GYRE's
   still does.  Changing the shared operator to multiply is a separate change
   with its own discharge and is not taken here.
10. **Rounds 1-24 of this receipt remain UNAUDITED** by the citation gate.

### ASKED / UNASKED

| choice | disposition |
|---|---|
| route the stage-3 momentum update through the same `ln_dynadv_vec .OR. lk_linssh` selector stages 1 and 2 already use | not a scientific choice; it is what `dynzdf.F90:119` says, and it is the round's assignment |
| extract one module-level helper carrying both arms rather than a second inline branch | not a scientific choice; Rule 12 cannot be discharged on a closure, and the sibling `rk3_stage_barotropic_correction` set the pattern |
| do NOT add `umask` to the key_qco arm although NEMO's has one | ASKED; legoESM masks after the barotropic correction instead and that is bit-identical, so adding it would be a second change in one commit.  Named, not taken |
| replace the circular Rule-12 discharge with the composition against `uu_Kaa_pre` | not a scientific choice; the first version could not fail, which a review measured |
| dump NEMO's own `e3u_0`/`hu_0`/`r1_hu_0` in the tank instrument rather than rebuild them | not a scientific choice; LOCK's `mesh_mask.nc` carries no `e3*_0` at all, so the rebuild is impossible there |
| do NOT change the shared operator to MULTIPLY by `r1_hu_0` | ASKED; it is a change to a shared operator on all four cards with its own discharge, and it would move GYRE's numbers.  Open question 9 |
| do NOT touch the decision-17 guard, the demo card, or the tanks' `zdf_baroclinic_only` flag | ASKED; instructed, and unchanged |
| a detached BEFORE probe worktree for the OVERFLOW arm instead of a temporary revert | ASKED; the same disposition round 32 recorded |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed |

UNASKED list: empty.

Every figure above is pinned in `manifests/nemo_testcase_l2_gyre_round33.json`.

## Round 34 — the tanks are admitted, round 32 is discharged on them, and two ASKED items are answered

Round 34 starts from `2c83a59bcfbe` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round34_preregister.json`, committed at
`a6a96b933bfb` before any measurement; addendum 1
(`..._round34_preregister_addendum1.json`, `6ea69c28df01`) records what the
independent CLAIM review broke, and addendum 2
(`..._round34_preregister_addendum2.json`, `9ba367885ba8`) registers the two
configuration changes the user authorised mid-round, before either was made.
No NEMO executable was run and no NEMO source was edited.  Every artifact is
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round34/` with SHA-256
in `artifacts.sha256`.

### A concurrency incident, recorded because it moved this round's files

A lingering round-33 agent was still writing into the same worktree when this
round began.  It committed `a7aeda155abf` at 05:24 -- a receipt-only change,
between this round's `a6a96b933bfb` and after `6ea69c28df01` -- and at about
05:21 it REVERTED this round's then-uncommitted rewrite of the admission gate,
preserving a byte copy and a diff under
`round33/UNEXPECTED_uncommitted_round21_admission_rewrite.{py,diff}`.  The
rewrite was re-applied and diffed against that preserved copy: the only
differences are formatting, two dropped type annotations, one deleted unused
constant, and two real improvements made on the second pass.  Nothing was
lost.

Two consequences are recorded rather than absorbed.  First, every measurement
taken between 05:14 and 05:25 is treated as CONTAMINATED and none is quoted:
the GYRE stage measurement launched in that window was killed and its log kept
as `round34/DISCARDED_contaminated_stage_ts.log`.  Second, that agent had also
left a full fidelity suite running against the tree this round was editing; it
was stopped, and no count from it is quoted either.

### Item 1 — the two tank records are ADMITTED, and the gate that admits them now works on any card

The user ran the round-33 acquisition on both tanks.  Attempt 1 died inside
FCM because the system perl has no `Text::Balanced`; attempt 2, with the conda
build environment first on `PATH`, built and ran both.  That PATH is now in
`run.sh`'s preamble so the next operator does not rediscover it.

`run.sh` then died anyway, at its RAW twin comparison under `set -e`, and took
the Rule-12 discharge, its plant and the outputs manifest down with it.  Raw
byte identity is the right thing to TRY first and the wrong thing to gate on:
NEMO's stream dumps write whole work arrays including the `nn_hls = 2` halo,
which NEMO neither owns nor initialises.  `run.sh` now logs every record's raw
result and falls through to the shared CONSUMED-FIELD ADMISSION.

That admission was written around GYRE.  It is card-general now and it is
still ONE gate: each record's dimensions come from its own header, each
record's kind from its own 16-byte magic (so the GYRE file-name map is
deleted), the inventory is discovered by globbing the source run, and the
files that must stay byte-identical are a list on the command line, so the
mesh travels with the restart.  A declared layout that does not account for
the payload exactly RAISES rather than comparing a misaligned buffer.

| card | inherited records | byte-identical | changed | admitted differences | verdict |
|---|---:|---:|---:|---:|---|
| LOCK_EXCHANGE-zco | 27 | 25 | 2 | 4 | **PASS** |
| OVERFLOW-zps | 27 | 24 | 3 | 16 | **PASS** |
| GYRE-zco (regression, round 19 vs 21) | 49 | 39 | 10 | 201 | **PASS**, unchanged |

**Every admitted difference is a HALO cell holding uninitialised memory, and
each is printed with its index and its two values.**  On LOCK the two changed
records are `oracle_transport_kt00000001_s2` and `_s3`, each differing in
`zFw` at `(0,0,0)` and `(1,0,0)`, values near `6.94e-310`.  On OVERFLOW the
three changed records are `_s1`, `_s2` and `_s3`, differing in `zFu` at
`(1,6,11)`, `(3,6,12)`, `(5,6,13)` and `(205,1,16)`, values near `4.2e-318`,
and in `s2`/`s3` also in `zFw` at `(0..1,0,0)` near `6.9e-310`.  **`0` of every
record's owned cells differ on either card**, and `mesh_mask.nc` and each
card's `kt=10` restart are byte-identical.  The plant -- one bit flipped in one
owned cell -- turns the gate red on both cards and exits non-zero.

### Item 2 — round 32's ordering fix is AT-BAR-NOT-EXACT on both tanks

**RELABELLED IN ROUND 38.**  This section said DISCHARGED, and its own table
below says `exact` FALSE on both cards -- 20 of 2540 cells on LOCK and 30 of
16900 on OVERFLOW are bit-unequal, AT-BAR by the `1e-15` tolerance and not by
the bar Rule 12 sets, which is bit exactness given NEMO's own inputs.  The
word DISCHARGED is now reserved for a bit-exact row.

| card | u face | v face |
|---|---|---|
| LOCK_EXCHANGE-zco | **AT-BAR**, `3.0814879110195774e-33` absolute against the `1e-15` bar; `exact` false, `20` of `2540` cells bit-unequal | **NOT APPLICABLE** |
| OVERFLOW-zps | **AT-BAR**, `3.8518598887744717e-34`; `exact` false, `30` of `16900` cells bit-unequal | **NOT APPLICABLE** |

legoESM's own `rk3_stage_barotropic_correction`, imported from the production
module the step function calls, is driven with NEMO's `dyn_zdf` output, NEMO's
`uu_b(Kaa)` and NEMO's OWN `e3u_0`, `umask`, `hu_0` and `r1_hu_0` read from the
record, and scored against NEMO's own stage-3 velocity.  **Nothing is
rebuilt** -- which makes this a stronger discharge than GYRE's, where the
column divisor is still reconstructed because `mesh_mask.nc` carries no
`hu_0`.  Both plants exit non-zero.

The two association rows are measured, not argued: the rebuilt column sum
minus NEMO's own `hu_0` is `0.0` on both cards, `1/hu_0` minus NEMO's
`r1_hu_0` is `0.0` on both, and the row that actually rides the discharge --
NEMO MULTIPLIES by `r1_hu_0` where the operator DIVIDES by the depth -- is
`1.6155871338926322e-27` on LOCK and `4.336808689942018e-19` on OVERFLOW.  It
stays OPEN; changing the shared operator is a separate change with its own
discharge.

### Item 3 — the diff review broke the admission gate, and it was right to

An independent DIFF reviewer, given the two commits and the oracle's source
and nothing else, **planted 1 m/s in an owned interior cell of a tank's `zFw`
and the gate said PASS.**  Four real defects came out of that review and each
now has an arm that goes red on it.

**THE WAIVER WAS ARM-BLIND.**  The gate's one admission ground that is not the
halo -- a slot the writer has not defined at the write point -- was hardcoded
for `zFw`.  That is true only under `ln_dynadv_vec = T`, where the record is written before
the `tra_adv_trp` call that creates `zFw` (`stprk3_stg.F90:463`), which the
source says in as many words at `stprk3_stg.F90:287`.  Under
`ln_dynadv_vec = F` the flux-form branch at `stprk3_stg.F90:295-301` fills
`zFw = e1e2t*ww` BEFORE that same point, so the slot is a real
state-carrying field.  **GYRE resolves T; both tanks resolve F.**
The waiver is now resolved per run out of the run's own `ocean.output` and
reported with the line it was read from --
`round19_oracle_v2_external/ocean.output:798` on GYRE,
`lock_kt1_10/ocean.output:705` on LOCK,
`overflow_kt1_10/ocean.output:822` on OVERFLOW -- and a run whose log is missing
or silent FAILS CLOSED to bit-testing.  **Both tanks still PASS, on a strictly
stronger claim: nothing is waived there at all.**

**THE PLANT COULD SILENTLY DO NOTHING.**  It was reachable only through a
record that already differed raw, so a PERFECT twin -- the strongest possible
outcome -- produced a plant that landed nowhere, a PASS, and a `run.sh` that
then refused the round.  A pending plant now forces the comparison open on the
first record whether or not its bytes differ, and a plant that still fails to
land is itself a violation.

**THE ORDERED BAROTROPIC STREAM never compared its eight appended fields.**
It compared a hardcoded name list, so once both sides carried the grown
schema a state change in those fields was invisible.  It compares the
intersection of the two field sets now, the legitimate growth still passes,
and it is inside the plant's reach.

**SIX OF TEN WRITER PROVENANCE STRINGS WERE STALE**, and two named the same
line for different records.  All ten are re-read from the `MY_SRC` files that
are compiled.

Not adopted as stated: the reviewer read "an undefined slot with no registered
reason is itself a test failure" as unenforced.  The test existed; what did
not exist was a RUNTIME refusal, and that is added -- an unreasoned waiver is
now a violation as well as a red test.

### Item 4 — decision 19: the tanks remove the barotropic mode, as NEMO does

ASKED and answered by the user, in the user's words: *"on BOTH tank cards the
barotropic removal before the implicit vertical solve goes OFF -> ON
(`zdf_baroclinic_only` False -> True), because NEMO executes
`dynzdf.F90:148` and the block below it there.  Do as NEMO does."*

One field moves.  `zdf_drag_in_matrix`, `barotropic_drag_substep` and
`nemo_stage_mean_imposition` stay OFF on these cards, and GYRE's resolution of
all four is unchanged -- printed from the built cards, not read off the source,
and pinned by a test.

**The companion statement is not transcribed, and the reason is MEASURED.**
`dynzdf.F90:156-159` adds the barotropic bottom stress back at the deepest wet
level.  `rCdU_bot` is EXACTLY zero on every owned cell of both tanks -- `0` of
`390` on LOCK, `0` of `606` on OVERFLOW -- so that statement contributes
exactly `0.0` and legoESM not carrying it is bit-exact rather than merely
small.  A whole-array maximum would report a denormal near `6.9e-310`; that is
uninitialised halo memory, and the test arm requires it to be present so the
zero is not measured on a field that is trivially zero everywhere.  GYRE's is
`5.0e-05` on `600` of `704` owned cells, and there the statement is already
carried by `zdf_drag_in_matrix`.  **This retires the round-32 receipt's open
worry that the missing removal was "not obviously inert" on these cards.**

**Rule 12 on the changed operator has a RECORD GAP, and it is named rather
than papered over.**  The changed operator maps the explicit stage update `A`,
`uu_b(Kaa)` and `umask` to the solve input.  NEMO's OUTPUT of it is
`uu_Kaa_pre` and it is in the record; its INPUT `A` is not, and the tanks take
the key_qco arm, which needs `r3u` at three time levels of which only `Kaa` is
recoverable.  Reconstructing `A` as `uu_Kaa_pre + uu_b` and feeding it back is
the operator's own inverse -- exactly the circularity round 33's review
caught -- so it is not done.  The WRITE-only frame that closes it is specified
and NOT run: one array pair, `puu(:,:,:,Kaa)` and `pvv(:,:,:,Kaa)` as the arm
leaves them and BEFORE the removal, in the same self-describing stream.

### Item 5 — decision 17: a caller is no longer refused for a mask it never reads

ASKED and answered by the user, in the user's words: *"move ONLY the
`active is None` half of the mask guard below the linear_free_surface early
return, so linear-free-surface callers are no longer refused for a mask that
path never reads; the other half of the guard and the moving-thickness (qco)
path stay exactly as they are."*  The earlier proposal to switch the demo card
to qco instead is WITHDRAWN.

The mask conversion moves with the guard, because it would otherwise raise a
`TypeError` before the guard could give its own message.  The `nemo_e3t_0` half
stays above, because the early return needs it.

**The four `test_nemo_recipe` failures go green and nothing else moves**: that
file was `4 failed, 20 passed` and is now `24 passed`.  All four failed in the same
guard, whose surviving `e3t_0` half is now `vertical.py:66-71` and whose mask
half is `vertical.py:85-89`.

A behavioural test cannot see the ORDER -- a linear-free-surface caller that
DOES supply a mask is served either way -- so a fourth arm reads the source and
pins it.  Moving the guard back above the early return turns three of the four
red, shown by mutation.

### ASKED / UNASKED

| choice | disposition |
|---|---|
| turn the tanks' barotropic removal ON (`zdf_baroclinic_only` False -> True) | ASKED and ANSWERED by the user: *"Do as NEMO does."*  Decision 19 |
| move the mask half of the thickness guard below the linear-free-surface early return | ASKED and ANSWERED by the user: *"Yes."*  Decision 17 |
| do NOT transcribe `dynzdf.F90:156-159` on the tanks | not a scientific choice; `rCdU_bot` is exactly zero on every owned cell of both, measured, so the statement contributes exactly `0.0` |
| do NOT turn on `zdf_drag_in_matrix` or `barotropic_drag_substep` with it | not a choice this round made; decision 19 named one field and only that field moved |
| resolve the admission's undefined-slot waiver from each run's own `ocean.output` rather than hardcoding it | not a scientific choice; the hardcoded version admitted a planted 1 m/s change on both tanks, which a review measured |
| a gate whose `ocean.output` is missing FAILS CLOSED to bit-testing | not a scientific choice; it is the strict reading, and the alternative is a silent waiver |
| open the tanks' v face and then report it NOT APPLICABLE | not a scientific choice; the record carried `v` all along and NEMO's own `vmask` there is identically zero |
| detached probe worktrees for the BEFORE and A2 arms | ASKED; the same disposition rounds 32 and 33 recorded.  Both flagged: `/tmp/codex-gyre-r34-before` at `947aa1e43a8e`, `/tmp/codex-gyre-r34-a2` at `049653f6a40f` |
| do NOT change the shared correction operator to MULTIPLY by `r1_hu_0` | ASKED; unchanged from round 33, open question 9 |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed.  The two tank runs were executed by the user, not by the agent |

UNASKED list: empty.

### The trajectories, before and after each decision

Nine arms, three cards by three trees, every one on a clean committed tree.
The before arm is `947aa1e43a8e` in the flagged probe worktree
`/tmp/codex-gyre-r34-before`, the decision-19 arm is `049653f6a40f` in
`/tmp/codex-gyre-r34-a2`, and the decision-17 arm is `8da3475fbc3d` in the
canonical worktree.  Every pair is scored by the SHARED oracle-relative move
gate, driven offline from the saved reports and their hashed per-cell
sidecars rather than by re-running each arm a second time.

| card | change | move gate | worsening, row-scale ulps (bar 2) | rows whose field moved | largest field move | first-over-bar |
|---|---|---|---:|---:|---:|---|
| GYRE-zco | decision 19 | **PASS** | `0` | `0` of `70` | `0.0` | `kt2` on T,S,u,v -> unchanged |
| LOCK_EXCHANGE-zco | decision 19 | **PASS** | `0.03125` | `10` of `50` | `6.93889e-18` | `kt4` on u -> unchanged |
| OVERFLOW-zps | decision 19 | **PASS** | `1` | `13` of `50` | `7.10543e-15` | `kt2` on T,u -> unchanged |
| GYRE-zco | decision 17 | **PASS** | `0` | `0` of `70` | `0.0` | unchanged |
| LOCK_EXCHANGE-zco | decision 17 | **PASS** | `0` | `0` of `50` | `0.0` | unchanged |
| OVERFLOW-zps | decision 17 | **PASS** | `0` | `0` of `50` | `0.0` | unchanged |

**Decision 17 is BIT-IDENTICAL on all three certified cards, measured cell by
cell**: no row's field moved by a single bit.  That is what a guard move must
be, and it is the answer to "show by measurement that no certified card's
trajectory moves".

**Decision 19 is inert on GYRE, also cell by cell** -- `0` of `70` rows moved
-- so the tank flag reaches no other card.  On the tanks it moves what it
should: five rows on LOCK, all `u`, three worsened and two improved, the
largest relative change `+3.284e-06` per cent at `kt4`; six rows on OVERFLOW,
all `u`, **all six improved**, the largest `-2.975e-08` per cent at `kt4`.
No row changed status on any card and the first-over-bar step never moved.

**This is the first landed change in this campaign that PASSES the shared move
gate.**  Rounds 32 and 33 both failed it on GYRE, at `1.063e+10` and
`3.003e+08` row-scale ulps.  Here the worst of six comparisons is `1` ulp
against a `2`-ulp bar.

**BOUNDARY for the eleven moved rows.**  Every one is at `kt >= 4` on LOCK and
`kt >= 4` on OVERFLOW, downstream of where each card's entering prefix stops
being exact -- the gate's own `exact_prefix_entering` is true only at `kt1` and
`kt2` on both.  No row at or before the first divergence moved.  **OWNER**: the
removal itself, whose whole ten-step effect is `6.9e-18` (LOCK) and `7.1e-15`
(OVERFLOW) -- five and eleven orders below the `uu_b` it subtracts, `1.135e-03`
and `4.503e-02` m/s.

### PART B — the kt=2 T/S owner is inside the STAGE-3 IMPLICIT TRACER SOLVE

kt=1's T and S at the end of each RK3 stage, scored through the model's own
compiled step against NEMO's own per-stage record, on `8da3475fbc3d`:

| stage | T absolute | T normalized | T bit-exact | S absolute | S normalized | S bit-exact | verdict |
|---|---:|---:|---|---:|---:|---|---|
| 1 | `0.0` | `0.0` | **yes** | `0.0` | `0.0` | **yes** | AT-BAR |
| 2 | `3.5527136788005009e-15` K | `1.514310e-16` | no, `3` of `18000` cells | `0.0` | `0.0` | **yes** | AT-BAR |
| 3 | `3.1956659540810506e-11` K | `1.361474e-12` | no, `11840` cells | `8.1712414612411521e-13` psu | `2.218110e-14` | no, `11229` cells | **DEBT** |

**THE FIRST STAGE OVER BAR IS STAGE 3.**  Stage 1 is BIT-EXACT on both
tracers -- not "at bar", identical -- and stage 2 is three cells of T off by
exactly one ulp of `23.460943` K with S still bit-exact.  Stage 3's rows are
the same arrays the trajectory scores as `kt2.before`, so the two agree by
construction and the walk closes.

**INSIDE STAGE 3, the divergence enters at the IMPLICIT VERTICAL SOLVE.**  The
accumulator that `tra_zdf` receives -- advection, then `tra_sbc_RK3`, then
`tra_qsr`, then `tra_ldf`, combined through the qco step -- is scored against
NEMO's own:

| boundary | T | S | verdict |
|---|---:|---:|---|
| `kt1.stage3.pre_zdf` | `1.4210854715202004e-14` K, `6.053921e-16` | `2.1316282072803006e-14` psu, `5.786350e-16` | **AT-BAR** |
| `kt1.stage3` output | `3.1956659540810506e-11` K, `1.361474e-12` | `8.1712414612411521e-13` psu, `2.218110e-14` | **DEBT** |

Everything `tra_zdf` is HANDED is at bar; what it RETURNS is over bar by three
orders on T.  **The owner is the stage-3 implicit vertical tracer solve
itself**, and every operator upstream of it in that stage is exonerated.

NEMO's own operator increments at this stage, from its own record, say which
of them could have carried a T-only signature at all:
`tra_ldf` contributes **exactly `0`** to both tracers, `tra_sbc_RK3`
`2.41485e-05` to T and `0` to S, `tra_qsr` `4.15401e-06` to T and `0` to S,
advection `8.38855e-08` on T and `1.32313e-07` on S.  All four are inside the
AT-BAR accumulator, so none of them owns the residual -- **`tra_ldf` is
exonerated twice over, by its own zero and by the accumulator's verdict.**

**THE CAUSAL INJECTION ARM IS UNINFORMATIVE AND IS NOT QUOTED AS EVIDENCE.**
The gate also feeds a pre-ZDF seed RECONSTRUCTED from NEMO's dumped `Kbb`
tracers, its post-`tra_ldf` tracers and its three `r3t` stretches
(`nemo_testcase_l2_gyre_stage3_completion_gate.py:147-156`,
`((1+r3t_Kbb)*T_Kbb + dt*(1+r3t_Kmm)*T_after_ldf)/(1+r3t_Kaa)`) into legoESM's
solve and scores the result -- **it is not a dumped array of NEMO's, and round
38 renamed it**; calling it "NEMO's own accumulator" credited a
reconstruction with the standing of an oracle dump.  It reads: it reads `3.1956659540810506e-11`, identical to the
faithful run, and its `causal_movement` is **exactly `0.0`** against a
faithful residual of `1.361474e-12`.  A control that moves the output by
nothing has not been applied, so its `ZDF_SOLVER_FIRST_OWNER_CAPABLE_BOUNDARY`
label proves nothing and the attribution above rests on the two MEASURED rows
instead.  Repairing that arm is registered work.

### Rule-11 records

**PB1's VALUE is REFUTED.**  It predicted stage 1's T at
`3.552713678800501e-15` K and S at `7.105427357601002e-15`, from a round-16/19
measurement it labelled STALE.  Measured on this tree, stage 1 is BIT-EXACT on
both: `0.0`, `0` of `18000` cells unequal.  The `3.55e-15` figure is what
stage TWO reads today, so quoting it as stage 1's was wrong by one stage.  The
prediction's DIRECTION -- stage 1 at bar -- held.

**PB2's MECHANISM was already retracted in addendum 1 and its PREDICTION
holds.**  Stage 3 is the first stage over bar.  The reason the addendum kept
it -- that a rounding residual cannot be amplified `95x` per stage by a
near-identity operator, so the size demands a new term -- is what the
measurement shows: the new term is the implicit vertical solve.

**PB3 is REFUTED.**  It predicted the stage-3 pre-zdf accumulator would
ALREADY be over bar, putting the owner at or before `tra_ldf`.  It is AT-BAR
at `6.05e-16` on T and `5.79e-16` on S, and the owner is downstream of it.
The falsifier the preregistration wrote for it is the outcome that occurred.

**PB6 is NOT SCORED, and `tra_qsr` is exonerated anyway.**  Addendum 1 bought
a qsr-only discriminator on the grounds that `tra_qsr` is the only stage-3
operator touching temperature alone, and that the residual is `61x`
T-favoured.  It is not needed: `tra_qsr`'s increment sits inside an
accumulator that is AT-BAR, so no operator before `tra_zdf` can own the
residual.  The T-favoured ratio is explained without it -- `tra_sbc_RK3` and
`tra_qsr` are both T-only in NEMO's own record, and GYRE's initial T carries
`19.460410` K of vertical contrast against S's `1.718057` psu.

**A COMMIT MESSAGE OVERSTATES ITS OWN MUTATION COUNT.**  `8da3475fbc3d` says
moving the guard back above the early return "turns three of the four red".
An independent reviewer built the exact pre-decision-17 form and measured
**two of four**; the three-of-four figure came from a different mutation --
an inserted duplicate guard -- run while the fix was being written.  The fix
is still non-vacuous and the order arm still catches the real regression; the
number in that commit message is wrong and cannot be edited, so it is
corrected here.

**A ROUND-33 CLAIM IS RETRACTED.**  The tank Rule-12 gate said those cards'
stage record "carries T, S, u and ssh and no v".  It carries `v`; the reader
skipped the third block.  The v face is still not scored, for a MEASURED
reason instead: NEMO's own `vmask` there is identically zero and its
`vv_Kaa_out` and `vv_b_Kaa` are zero with it, because both tanks are 2-D x-z
boxes.  NOT APPLICABLE, not UNMEASURED.

**THE ROUND-33 TANK DISCHARGE COULD NEVER HAVE PRINTED ITS OWN SUMMARY.**  It
asked each row for `n_unequal`, which the shared scoring helper did not
return, so a real record crashed AFTER the arithmetic had already run.  Round
33's review could not have caught it: the record did not exist yet.

**A MEASUREMENT WAS LOST TO MY OWN EDIT, AND THE STAMP WAS RIGHT TO REFUSE
IT.**  A fifteen-minute OVERFLOW arm was killed at its worktree stamp because
I edited a tracked file while it was in flight.  The fail-closed stamp did
exactly its job; the discipline was mine, and every arm quoted above was
launched only on a tree that then stayed untouched.

### Tests and gates

`tests/ocean/fidelity/test_nemo_testcase_round34_admission.py` (19): the
card-general admission, including the reviewer's own attack -- 1 m/s planted
in an owned tank `zFw` cell must FAIL under the flux-form arm and be admitted
only under the vector arm -- a run with no `ocean.output` failing closed, a
plant on a perfectly identical twin, a waiver with an empty reason, and the
appended barotropic fields being compared once both sides carry them.

`tests/ocean/fidelity/test_nemo_testcase_round34_tank_zdf_removal.py` (9):
the flag resolution on all three cards, that no other flag moved, `ln_drg_OFF`
resolved true on both tanks, `rCdU_bot` zero on every owned cell with a
non-vacuity arm on the halo denormal, and NEMO's own super-diagonal at the
deepest wet level being zero on every wet column of both tanks.

`tests/ocean/unit/test_nemo_qco_thickness_guard.py` (4): decision 17's four
arms, one of which reads the source and refuses both a moved guard and a
duplicate one.

`tests/ocean/fidelity/test_nemo_testcase_offline_compare.py` (5): the offline
comparison is the shared gate -- every plant it refuses, an improving plant
passing, a self-comparison moving nothing, and a drifted sidecar refused.

`tests/ocean/fidelity/test_nemo_testcase_l1_tanks_round33_zdf_rule12.py` (15)
and `..._l2_gyre_phase3_gate.py` (22): green, including the new v-face arm and
its plant.

The three offline-comparison plants: `worsen-3ulp` and `at-bar-to-debt` both
exit non-zero, `improve` passes.  Both admission plants and both tank Rule-12
plants exit non-zero.

### Merge readiness

`03c6e8d96ff7` remains an ancestor of this branch, so the integration is still
a FAST-FORWARD with zero conflicts by construction.  What blocks it after this
round:

1. **GYRE's `kt2` `T`/`S`/`u`/`v` rows still fail**, unchanged to the bit by
   everything this round did.  Their owner is now LOCALISED: the stage-3
   implicit vertical tracer solve, with every operator upstream of it AT-BAR
   and stages 1 and 2 at bar (stage 1 bit-exact).
2. **Round 32's ordering fix is AT-BAR-NOT-EXACT on LOCK_EXCHANGE and
   OVERFLOW and DISCHARGED bit for bit on GYRE** (relabelled in round 38: the
   two tank rows carry `exact` false, 20 of 2540 and 30 of 16900 cells
   bit-unequal, which is inside the `1e-15` tolerance and outside Rule 12's
   bar).  ORCA2 stays UNMEASURED.  Round 33's item 1(a) is CLOSED.
3. **Round 33's stage-arm fix** stays discharged bit for bit on GYRE and
   measured NOT REACHED on both tanks.
4. **Round 30's `dyn_ldf` fix still carries an OPEN Rule-12 eligibility**,
   unchanged.
5. **Decision 19 has no operator-level Rule-12 discharge**, because the
   removal's INPUT is not dumped.  The WRITE-only frame that closes it is
   specified and not run.  What IS measured is the whole-trajectory effect,
   which passes the shared move gate on all three cards.
6. **Eleven worsened-or-moved trajectory rows** from decision 19, all at
   `kt >= 4`, with a boundary and an owner; plus round 33's 24 and round 32's
   22, unchanged.
7. **ORCA2 has no card on this branch.**

### Open questions

0. **The stage-3 implicit vertical TRACER solve owns `kt2`'s T and S.**  Its
   internals are not instrumented: the round-29 record carries the MOMENTUM
   tridiagonal, not the tracer one.  The next step is the WRITE-only frame for
   `tra_zdf`'s own `zwi`/`zwd`/`zws` and its entering and leaving T/S, which is
   the exact analogue of what round 29 built for `dyn_zdf`.
1. **The causal injection arm at that boundary is broken**: it moves the
   output by exactly `0.0`.  Repair it before any attribution rests on it.
2. **The tanks' operator-level Rule-12 for decision 19**, one WRITE-only array
   pair away.
3. **Round 30's `dyn_ldf` Rule-12 row**, unchanged.
4. **ORCA2 has no card on this branch**, unchanged.
5. **The moved trajectory rows**, this round's eleven and the earlier rounds'.
6. **Decision 16**, still an open card-identity gap.
7. **The slow forcing's depth average** still uses the min rule where NEMO
   uses the area-weighted mean; `3.06e-08` at `kt=2`, unchanged.
8. **Two association rows on the barotropic correction's divisor**, unchanged
   on GYRE; on the tanks the reconstruction rows are now `0.0` and only the
   divide-versus-multiply row survives, at `1.6155871338926322e-27` (LOCK) and
   `4.336808689942018e-19` (OVERFLOW).
9. **Rounds 1-24 of this receipt remain UNAUDITED** by the citation gate.

Every figure above is pinned in `manifests/nemo_testcase_l2_gyre_round34.json`.

## Round 35 — the tracer solve gets an instrument, and its first suspect changes before the record exists

Round 35 starts from `23c46232ea60` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round35_preregister.json`, committed at
`0e78666589ae` before any measurement; addendum 1
(`..._round35_preregister_addendum1.json`, `033dbf9c4d1e`) records what the
independent CLAIM review broke.  **No NEMO executable was run, no `makenemo`
or `mpirun` was invoked, and no NEMO source was edited.**  The acquisition is
written and handed over; the record does not exist yet.

One detached probe worktree is flagged: `/tmp/codex-gyre-r35-suite` at
`0e78666589ae`, used for the suite diagnosis so that a long test run could not
collide with edits in the canonical tree.

### What was read before anything was measured

The subject is the boundary round 34 named: at kt=1 stage 3 the accumulator
handed to `tra_zdf` is AT-BAR on both tracers while `tra_zdf`'s output is
`1.36e-12` on T.  Four readings of the oracle shaped the instrument, and each
of them changes what a reader has to transcribe.

**The matrix is built ONCE and salinity is solved with temperature's.**  The
build is guarded by `trazdf.F90:159-160`, and `ln_zdfddm` is `F`
(`round29_oracle_v2_zdf_matrix/ocean.output:568`), so `avs` never enters.

**The matrix diffusivity is `avt` PLUS the isoneutral a33 fold `ah_wslp2`**
(`trazdf.F90:172-174`).  `l_ldfslp` is true because `ldftra.F90:249` sets it
for the rotated laplacian, which this run resolves (`round29_oracle_v2_zdf_matrix/ocean.output:667`), and
`ln_traldf_msc` is `F` (`round29_oracle_v2_zdf_matrix/ocean.output:657`), so it is the `ah_wslp2` arm and not the `akz`
one.  **Round 34's exoneration of `tra_ldf` does not cover this**: that fold
enters the ZDF matrix and never touches `ts(Krhs)`, so "`tra_ldf` contributes
exactly `0`" says nothing about it.

**The thickness macros resolve differently at T and W points.**
`domzgr_substitute.h90:126` gives `e3t` a `tmask` factor;
`domzgr_substitute.h90:131` gives `e3w` none, a one-dimensional reference
ladder, and `r3t` at the T point rather than any `r3w`.  Both are verbatim in
the compiled `ppsrc/nemo/trazdf.f90:231-233`.

**NEMO's negative-salinity clamp RUNS on this card.**  `trazdf.F90:89-91` is
guarded by `.NOT.(ln_SEOS .AND. rn_b0==0)` and `ln_SEOS` is `F`
(`round29_oracle_v2_zdf_matrix/ocean.output:158`), so the existing stage-3 record holds a POST-clamp
column and any pre-clamp comparison needed a new dump.

### Item 1 — the instrument, and the exact commands

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round35_trazdf/`
holds `trazdf_round35.patch` and `run.sh`.  The patch adds **258 lines and
removes none** to a config copy of the shipped `trazdf.F90`; `run.sh` refuses
if the removal count ever exceeds the diff header, refuses a source card that
overrides `trazdf.F90` or lacks the round-29 `dynzdf` instrument, refuses a
build whose compiled `ppsrc` does not carry each writer, refuses a
vector-math symbol in the binary, and refuses a mount under two gigabytes
free.

The record, one file, `oracle_trazdf_matrix_kt00000001.bin`, magic
`NEMO_L2_TRAZD_1`, written once at `kt = nit000` from `tra_zdf` AFTER the
clamp so both columns are in it:

| group | arrays |
|---|---|
| the arm, as payload | `ln_zdfddm`, `ln_zad_Aimp`, `ln_zdfmfc`, `ln_traldf_msc`, `l_ldfslp`, `ln_SEOS`, `a33_allocated`, `rDt`, `rn_b0`, `jp_tem`, `jp_sal` |
| as `tra_zdf` RECEIVED them | `T_Kbb_in`, `S_Kbb_in`, `T_Kmm_in`, `S_Kmm_in`, `T_Krhs_in`, `S_Krhs_in` |
| the matrix as built | `zwt_mix` (= `avt` + `ah_wslp2`), `zwi`, `zwd`, `zws`, `zwt_lu` |
| the solve | `rhs_T`, `rhs_S`, `fwd_T`, `fwd_S`, `sol_T_pre_clamp`, `sol_S_pre_clamp`, `sol_T_post_clamp`, `sol_S_post_clamp` |
| the operands | `avt`, `avs`, `ah_wslp2`, `akz`, `tmask`, `e3t_Kbb`, `e3t_Kmm`, `e3t_Kaa`, `e3w_Kmm`, `e3t_0`, `e3w_0`, `r3t_Kbb`, `r3t_Kmm`, `r3t_Kaa` |

The right-hand side is NEMO's own `zrhs` captured in the loop that computes
it, not a rebuild; the solved column is an independent Fortran copy taken
after the third recurrence and therefore before the clamp, so the clamp arm
is not circular.

`run.sh` also stacks the round-33 reference-geometry writer — the same file
the two tanks were acquired with, not a copy — so the one run also gives GYRE
the `r1_hu_0` that `stprk3_stg.F90:440,444-445` multiplies by and that GYRE's
barotropic discharge currently has to rebuild.  **That transcription change is
NOT made here.**

**THE COMMANDS, to be run by the operator, not the agent:**

```
PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH \
  /tmp/codex-gyre/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round35_trazdf/run.sh
```

One command.  It builds `GYRE_OMIP_L2_P3_SM_R35TRAZDF` from
`GYRE_OMIP_L2_P3_SM_R29ZDF`, runs it into
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round35_oracle_trazdf_matrix`,
tries raw twin identity first and logs every record's result, falls through to
the shared consumed-field admission, re-runs the round-29 gate on the grown
momentum record, then runs the round-35 gate and its seven plants.  It prints
`ROUND35_GYRE_TRAZDF_MATRIX_ORACLE_READY` on success.

### Item 2 — the reader and the gate, proven on a synthetic record

`nemo_testcase_l2_gyre_round35_trazdf_matrix.py` was written before the record
exists and is exercised against a **synthetic record written by a Python twin
of the Fortran writer** — same magic, same sixteen header integers, same
groups, same column-major payload.  The twin's payload is built by a plain
per-column triple loop written straight off the source; the reader's rebuild
is a vectorised one.  They are different code on purpose, and a test asserts
it: perturbing an operand moves the rebuild and not the dump.

| arm | what it decides |
|---|---|
| `calibration` | NEMO rebuilt from NEMO's own operands; zero cells unequal or no number may be quoted |
| `assembly` | legoESM's own tracer diagonals, given NEMO's operands, scored per diagonal |
| `sweep` | legoESM's own ordered solve, given NEMO's matrix and NEMO's RHS |
| `rhs_content` | legoESM's own content-RHS builder, given NEMO's operands |

The sweep arm drives the function object the trajectory runs, not a copy:
`nemo_ordered_tridiagonal_solve` IS the private solve, and a test asserts the
identity by name.  The assembly arm drives `nemo_tracer_tridiagonal`,
extracted this round from the pair solve that still calls it, so the assembly
can be scored separately from the sweep; before the extraction it had no name
and could only be observed through a solved column, which conflates the two.

Seven plants, one per arm, each turning the gate red and exiting non-zero,
end to end through the CLI.  Fourteen malformed records each get a VERDICT
rather than a traceback: bad magic, bad version, 32-bit payload, a duplicated
array, an extent past EOF, a tail shorter than one group header, a non-finite
payload, a record short of an expected array, `rDt` of zero, `jp_tem` equal to
`jp_sal`, and five flag settings naming an arm this reader does not
transcribe.  **42 tests pass.**

**The gate names what it cannot see.**  Four deviations are reported with the
measured condition that makes each inert here, rather than passed over: two
maskings legoESM does that NEMO does not, a bottom row that agrees only
because the diffusivity at `jk = jpk` is zero, and the signed-zero one below.
A synthetic record with a dry bottom flips two of them LIVE and turns the
assembly rows red, so the report is not decoration.

**A REAL TRANSCRIPTION DIFFERENCE FELL OUT OF RUNNING THE GATE ON THE TWIN.**
NEMO's `zwi` at the surface row and its `zws` at the bottom row are NEGATIVE
zeros — `-p2dt*0.0/e3w` with `zwt(:,1) = 0` at `trazdf.F90:204` and
`trazdf.F90:219-220` — where legoESM writes POSITIVE zeros into both slots.
The bits differ; no arithmetic result does, because `-0.0 + x == x` exactly
and neither slot is ever divided by.  It gets its own status,
`AT-BAR-SIGNED-ZERO`, with its own count.  Folding it into AT-BAR would hide a
real difference and calling it DEBT would claim a numerical one that is not
there.

**The shared admission gate learns self-describing records.**  It used to
raise on any magic outside its fixed schema table, and the GYRE baseline
carries eighteen such magics; this round grows one of them on purpose.  The
comparison is now over the intersection of the two field sets, so the
legitimate growth passes while a state change in a shared field does not, and
its plant targets an OWNED cell — at flat index zero it landed in the halo and
was admitted, the same defect round 34 removed from the record-level plant.

### The diff review broke the gate seven times, and the seventh produced a result

An independent DIFF reviewer, given the commits and the oracle and told to
make the reader accept a wrong record and the gate pass on a wrong solve,
succeeded seven times.  Each defect now has an arm that goes red on it.

**THE VACUOUS PASS, and the one change the reviewer insisted on.**  The reader
took the scored box as a header claim: any sub-box parsed, five per cent of
the cells were scored, the verdict was the same green, and **every plant
stayed red**, so the controls could not catch it.  The box is now CHECKED — it
must be the whole computed domain minus one symmetric halo — and the cell
count is printed next to the verdict.  A record from any step but the first is
refused rather than scored under a label naming one.

**A MANUFACTURED ZERO.**  The writer emits zeros for `ah_wslp2` when the array
is unallocated, so a record claiming the isoneutral slopes are on while saying
the array is absent would have made "the fold is exactly zero" true by
construction.  The record's flags must now agree with each other.  The same
check refuses a record whose salinity clamp never ran, whose pre/post rows
would otherwise compare a column with itself.

**A TRACEBACK INSTEAD OF A VERDICT.**  A rank-3 array with a plausible but
wrong third extent passed structural validation and died in a broadcast three
functions later — exactly what the reader's own docstring forbids.  Every
array now has an expected rank and depth.  And `run.sh` no longer reads a
non-zero exit as proof a plant landed: a crash exits non-zero too, so it
requires the gate's own `STATUS DEBT`, and it takes the arm list from the gate
rather than a copy that could fall behind.

**THE ADMISSION GATE** admitted a candidate that DROPPED a field — only growth
is legitimate — and its plant landed in a rank-0 scalar, which has no halo, so
it never exercised the owned/halo selector it exists to control.  It also
CRASHED on this baseline: a pending plant forced open the first record
whatever its magic, and eighteen of GYRE's magics are unregistered, so the
plant run ended in `GATE-ERROR` **at exit 0**, which `run.sh` would have read
as "the plant turned it red".  Measured before and after on the round-29
records: `GATE-ERROR` exit 0, now `FAIL exact=50/51 changed=1` exit 1, with
the unplanted run `PASS exact=51/51 changed=0 admitted=0` exit 0.

**AND THE PRINTED VERDICT** now carries the blind spot, so a human reading
`STATUS` cannot miss that the model's own right-hand side was never compared.

**THE SEVENTH FIX PRODUCED THE ROUND'S SECOND MEASURED RESULT.**  The assembly
arm was taking its working dtype from the thickness where its production
caller takes it from the right-hand side — assuming exactly what the new
parameter exists to stop assuming.  With that corrected, the `rhs_content` arm
reads **VALUE-AT-BAR, not exact**, on the synthetic twin: `1` cell of `125`,
`2.842170943040401e-14` absolute, `5.17e-17` normalized.

| arm, on the synthetic twin | status |
|---|---|
| `calibration` (all thirteen rows) | **AT-BAR**, exact |
| `assembly.zwd`, `sweep.T`, `sweep.S` | **AT-BAR**, exact |
| `assembly.zwi`, `assembly.zws` | **AT-BAR-SIGNED-ZERO**, `25` of `125` cells, absolute `0.0` |
| `rhs_content.T` | **VALUE-AT-BAR**, `1` of `125` cells, normalized `5.17e-17` |

That is the association PR8 predicted, and it is the answer round 36 needs
before it can act on the `tracer_combine` question: legoESM's content builder
groups the update as `h*(p2dt*T)` where NEMO writes `p2dt*h*T`, so **flipping
the switch alone would not be bit-exact** — the association has to move with
it.  The number is from the SYNTHETIC twin and is labelled so; what transfers
is the statement that the two groupings are different floating-point
expressions, not the cell count.  The gate calls the row DEBT, because the bar
is exact.

**Three attacks FAILED, and are recorded because a survived attack is a
result.**  The signed-zero status could not be made to swallow a real
difference.  The `implicit_solver` extraction was verified bit-identical
against `23c46232ea60` across dry masks, the implicit-`w` arm and `nlev = 1`.
And `nemo_rebuild` is provably not the code it checks — NumPy against JAX
against the test's scalar loops — with the face index mapping matching
`trazdf.F90:219-220` exactly.

**AND THE FINDING BROKE THIS ROUND'S OWN CONTROLS, which is recorded rather
than quietly repaired.**  Once the RHS association turned the unplanted
verdict to DEBT, every plant check — in `run.sh` and in the tests — was
asking a question that answers itself: with a baseline already red, "the
planted run exited non-zero" is true with the plant DELETED.  Seven controls
that proved nothing, and the same defect round 34 removed from the admission
gate's plant, reintroduced here by a finding that arrived after the controls
were written.  A planted run now scores the record twice and names the rows
that MOVED; a plant that moved nothing exits `3` and says so.  The `a33` plant
lands only in the condition row, because `nextafter(0)` is a denormal that
vanishes when added to `avt` — which is why the condition rows had to join the
comparison, and why that plant read as landing nowhere while still exiting
non-zero.

### The preregistered prediction, and the part of it already REFUTED

PR1 through PR7 are in the preregistration.  The headline, PR4, predicted that
legoESM's assembly and sweep would both reproduce NEMO given NEMO's own
inputs, putting the owner in an OPERAND — ranked `avt` first, the right-hand
side second.

**PR4's RANKING IS REFUTED, before the record exists, by measurement rather
than argument.**  An independent claim review found that neither arm can see
the right-hand side at all: the production solve takes it as an argument.
Building the three NEMO cards then printed the reason it matters —

| card | `tracer_combine` | `zdf_implicit_solver_evaluation` |
|---|---|---|
| GYRE-zco | `concentration` | `nemo_literal` |
| LOCK_EXCHANGE-zco | `concentration` | `nemo_literal` |
| OVERFLOW-zps | `concentration` | `nemo_literal` |

— so `_nemo_tracer_content_rhs` stays `None` and the literal `trazdf` matrix
is fed `T_solve_in * dz_cell`, which is `e3t(Kaa)*T_expl`, where NEMO writes
`e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*T(Krhs)`.  **The builder whose own docstring
says it exists so the literal matrix can consume NEMO's content RHS is
selected on no NEMO card.**  The first suspect is therefore the right-hand
side, and not as "the same expression with a different association" — on this
card it is a different expression.  PR4's claim SHAPE stands: the two arms
still decide whether the owner is inside the transcription.  Its ranking does
not, and the new arm PR8 measures whether flipping the switch would even be
bit-exact, so that round 36 does not decide it on evidence nobody has.

PR5's threshold survived the review and its MECHANISM did not: GYRE's initial
density is not horizontally uniform, because `usrdef_istate.F90` multiplies
both tracers by `ptmask`, so land columns carry `T = 0`.  `ah_wslp2` is still
predicted exactly zero, for the masked-difference reason instead, and that
reason does not transfer to a card with interior topography.

### Item 3 — the three suite failures

| failure | verdict | status |
|---|---|---|
| `test_every_oracle_comparison_has_a_row` | the ratchet's DATA is stale; neither the test nor the model is wrong | FIXED |
| `test_planted_stage_control_exits_nonzero_end_to_end` | the TEST EXPECTATION is stale | FIXED |
| `test_a_dirty_tree_refuses` | the CODE is wrong: a real hole in the provenance gate | FIXED at the source |

**The first** refuses four `compare_*` drivers that landed between 2026-07-27
and 2026-08-12 with no disposition, which is exactly the omission the ratchet
exists to force.  The decisive line: `oracle comparison drivers missing from
the case board ... ['advection_nemo', 'grids_tripole_mpas', 'tendencies_nemo',
'three_way_nemo']`.  Each is classified as a diagnostic with the reason quoted
from its own docstring: two are per-process matches at a fixed state with no
time integration, one says in as many words that it "is NOT a NEMO-fidelity
statement", and the fourth re-scores arms an existing board row already
covers.  `9 passed`.

**The second** froze three control keys where the gate emits four; the fourth,
`faithful_only`, was already present before this branch, which is why it also
failed on the pre-fix tree.  The decisive line: `Left contains 1 more item:
{'faithful_only': False}`.

**THE WHOLE SUITE IS GREEN, in collection order, on the final tree**:
`969 passed, 7 skipped, 18 deselected in 2360.12s`, against a baseline of
`3 failed, 910 passed, 7 skipped, 18 deselected` measured independently on a
clean tree.  Three failures fixed, fifty-nine tests added, nothing else moved.
Nine repo-wide ratchet failures remain and are NOT this round's: the identical
nine appear on `23c46232ea60`, in files this round did not touch.

**The third is not a test quirk and it was fixed at the source.**
`allow_dirty_stamps` set a PROCESS-GLOBAL latch and nothing reset it, so once
any driver's `main` armed it every later `worktree_stamp` in that process
accepted a dirty tree SILENTLY — a harness running two gates back to back
would stamp the second clean while it was not.  The polluter is
`test_planted_entry_control_exits_nonzero_end_to_end`, which calls a gate's
`main` in-process with `--allow-dirty`; four other tests leak identically.

The fix is three pieces and none of them is a conftest fixture, because the
hole is in real callers too: `worktree_stamp` takes an explicit `allow_dirty`;
`allow_dirty_stamps` still arms immediately but returns a context manager that
restores; and every driver `main` that arms it wears `@scoped_allow_dirty`.
Per Rule 2, the enforcement is a GATE and not the prose: a test audits every
driver in the directory and goes red on one that arms the escape without
scoping its `main`, with a synthetic undecorated driver proving the audit can
fail.

Red then green, measured: the pair that reproduced it — the overflow
barotropic gate followed by the stamp tests — was `1 failed, 23 passed` and is
now `24 passed in 250.67s`.  A test reproduces the pre-fix shape in-process
and asserts the latch leaks, so the fix cannot become vacuous.

### ASKED / UNASKED

| choice | disposition |
|---|---|
| flip `tracer_combine` from `concentration` to `thickness_weighted` on the NEMO cards | **ASKED, NOT ANSWERED, NOT CHANGED.**  It is a scientific choice about what the model computes and it is not this round's to make.  The new `rhs_content` arm measures whether it would be bit-exact |
| land the `r1_hu_0` multiply-form transcription | ASKED; explicitly deferred to round 36 by the coordinator.  Only the RECORD was made to carry it |
| extract `nemo_tracer_tridiagonal` from the pair solve and publish the ordered sweep under a name | not a scientific choice; a behaviour-preserving extraction with the dtype threaded so it is byte-exact, plus a test that the composition reproduces the pair solve bit for bit |
| give `-0.0` versus `+0.0` its own status rather than AT-BAR or DEBT | not a scientific choice; it is the only classification that neither hides a bit difference nor claims a numerical one.  No bar constant moved |
| classify the four unclassified comparison drivers as diagnostics | not a scientific choice; each reason is quoted from the driver's own docstring, and none of them adds a case the board lacks |
| scope the allow-dirty escape rather than reset it in a conftest | not a scientific choice; the conftest form fixes pytest only, and the defect is in real callers |
| source card `GYRE_OMIP_L2_P3_SM_R29ZDF` rather than the base GYRE card | not a scientific choice; it is the card the round-29 and round-34 GYRE records were written by, so the twin identity check compares against a run of the SAME `MY_SRC` set |
| detached probe worktree for the suite diagnosis | ASKED; the same disposition rounds 32-34 recorded.  Flagged: `/tmp/codex-gyre-r35-suite` at `0e78666589ae` |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed |

UNASKED list: empty.

### Merge readiness

Unchanged from round 34 except where noted.  `03c6e8d96ff7` remains an
ancestor, so the integration is still a fast-forward.

1. **GYRE's `kt2` `T`/`S`/`u`/`v` rows still fail.**  Nothing this round
   changed a model number: the only production edits are a behaviour-preserving
   extraction and a provenance scope.
2. **The tracer solve now has an instrument and a gate**, both unmeasured
   until the operator runs the acquisition.
3. **A first suspect that was not on round 34's list**: the right-hand side,
   because no NEMO card selects the content form.
4. Rounds 30, 32, 33 and 34's open rows are unchanged.

### Open questions

0. **Run the acquisition.**  Every number in this round's gate is UNMEASURED
   until it exists.
1. **`tracer_combine` on the three NEMO cards.**  Asked above; a decision is
   needed before round 36 can act.
2. **The causal injection arm at the stage-3 boundary is still broken** — it
   moves the output by exactly `0.0`.  Unchanged from round 34.
3. **The tanks' operator-level Rule-12 for decision 19**, unchanged.
4. **Round 30's `dyn_ldf` Rule-12 row**, unchanged.
5. **ORCA2 has no card on this branch**, unchanged.
6. **The moved trajectory rows** from rounds 32, 33 and 34, unchanged.
7. **The slow forcing's depth average**, unchanged.
8. **The barotropic correction's divisor**: GYRE's discharge still rebuilds it,
   and after this acquisition it will not have to.
9. **Rounds 1-24 of this receipt remain UNAUDITED** by the citation gate.

## Round 36 — the record was mislabelled, the RHS residual was ours, and the divisor becomes a multiply

Round 36 starts from `dcc4b23787cd` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  **No NEMO
executable was run, no `makenemo` or `mpirun` was invoked, and no NEMO source
was edited.**  No detached probe worktree was created.

The round-35 acquisition had already been executed by the operator (log
`round35/gyre_trazdf_run.log`); this round finishes it and then walks two
items off it.

### Item 1 — the acquisition finished, and why it had not

The run built, ran ten steps, wrote the record and decided the twin admission
`PASS` — and then died in the loop that PRINTS the admitted differences:

```
TypeError: list indices must be integers or slices, not str
```

Under `set -e` that took down everything after it: the round-29 regression,
the round-35 gate, its seven plants and the outputs manifest never ran.

The cause was ONE list with TWO shapes.  `compare_record` appends a dict per
admitted difference; `_compare_self_describing` appended a five-element list
with no `record` key.  `run` concatenates both and `main` formats every entry
through the dict format string, so any run in which a self-describing record
differs in a halo cell — which the round-33 reference-geometry growth
guarantees — crashes at the very end.  Fixed in the shared shape, not at the
print site, with a test that drives `main` end to end and is red on the parent
commit with the exact `TypeError`.

Re-run on the real record: **admission PASS**, 43 of 51 records byte-identical,
8 changed, **49 admitted differences** — 44 halo cells across six records plus
5 undefined-slot cells in `oracle_transport_kt00000001_s1.bin:zFw`, every one
printed with its index and both values.  The restart and `mesh_mask.nc` are
byte-identical, so the twin is real.  The `--plant-consumed` control turns it
red.  The round-29 regression on the GROWN momentum record is `AT-BAR` on all
eight rows, 0/21120 each.  `round35_outputs.sha256` written over 57 files.

### Item 2 — the record did not decode, and NEMO's own ALLOCATE says why

The gate refused the new record: *array name at 4621256 is not ASCII*.  The
refusal was right and the diagnosis was missing.

`zdf_oce.f90:85-86` of this round's own compiled ppsrc allocates

```fortran
ALLOCATE( avm (jpi,jpj,jpk), ..., avs(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk) ,
   &      avt (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk) , ...
```

`avm`, on the same statement, IS full-domain; `avt` and `avs` are the INTERIOR
box.  The round-35 instrument writes `WRITE(il2_unit) avt` (`trazdf.f90:247`,
`:249`) after a header declaring `jpi, jpj, jpk`, so each of those payloads is
`32*22*31` values where its header claims `36*26*31` — 57536 bytes short
each — and every array after `avs` lands at the wrong offset.

**The instrument needs fixing and that is an acquisition.**  The two lines
become the same staging every neighbouring array already uses:

```fortran
zl2_tmp(:,:,:) = 0._wp
zl2_tmp(ntsi:ntei,ntsj:ntej,:) = avt(:,:,:)
WRITE(il2_unit) 'avt             '   ;   WRITE(il2_unit) 3, jpi, jpj, jpk
WRITE(il2_unit) zl2_tmp
```

Meanwhile the payloads are COMPLETE over the interior and the interior is
exactly the box this gate scores, so the record is mislabelled rather than
short.  The reader recovers those two arrays — but only where the STREAM
proves it: the declared extent is used unless it fails to leave a recognised
array name behind and the interior extent succeeds, EOF counts as an ending
for the declared extent and NOT for the shorter one, and the duplicate-name
and shape refusals every other array gets run first.

Both gaps in that guard were found by an independent claim review and closed
before this was recorded.

### The `tra_zdf` discharge, given NEMO's inputs

Commit `e6ef14d5d4f0`, record
`round35_oracle_trazdf_matrix/oracle_trazdf_matrix_kt00000001.bin`, 21120
scored cells (i 3–34, j 3–24, halo 2, 30 levels) on the 36×26×31 domain.

| arm | row | cells unequal | max abs |
|---|---|---|---|
| calibration | all 13 rows | **0 / 21120** | **0** |
| assembly | `zwi` | 704 (signed zero only) | 0 |
| assembly | `zwd` | **3120 / 21120** | **299.71** |
| assembly | `zws` | 704 (signed zero only) | 0 |
| sweep | T | 133 / 21120 | 7.105e-15 |
| sweep | S | 111 / 21120 | 7.105e-15 |
| rhs_content | T | **0 / 21120** | **0** |
| rhs_content | S | **0 / 21120** | **0** |
| clamp | T, S | 0 / 21824 | 0 |

The calibration arm is 0 cells unequal on every row.  That is the reader's
licence to quote anything else, and it is also the interior-extent recovery's
second, independent check: it rebuilds `zwt_mix`, `zwi`, `zwd`, `zws`, the LU
diagonal, both right-hand sides, both forward sweeps and both solved columns
FROM the embedded `avt`, and a misaligned embedding cannot reproduce them.
An independent review measured that directly: C-order 12743 cells unequal,
origin ±1 in i 2663 or 2099, origin (0,0) 5033, correct 0.

**The first non-bit statement — CORRECTED after review.**  The independent
claim review BROKE the first version of this claim, and it was right.  In
NEMO's own execution order:

* `trazdf.f90:443-444` is the first, and it is a SIGN OF ZERO.  NEMO's
  `zwi(ji,1) = -p2dt*zwt(ji,1)/e3w` with `zwt(:,1)=0` is `-0.0`, where
  legoESM writes `+0.0`; 704 cells on each of `zwi` and `zws`, and **600 of
  the 704 are WET**.  Every arithmetic result downstream is identical, which
  is why the gate gives it its own status and its own count instead of
  folding it into AT-BAR.  Calling it "at bar" and then claiming :445 is
  "first" would have been a tolerance smuggled into an ordering claim.
* `trazdf.f90:445` is the first with a NON-ZERO value difference:

```fortran
zwd(ji,jk) = (e3t_3d(ji,jj,jk)*(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) - ( zwi(ji,jk) + zws(ji,jk) )
```

  NEMO writes it unconditionally; `nemo_tracer_tridiagonal` ends with
  `diagonal = jnp.where(wet > 0, diagonal, 1.0)`.  The 3120 differing cells
  are EXACTLY the 3120 dry cells of the scored box — zero wet cells differ —
  and GYRE has no dry cell above a wet one, so the dry diagonal cannot reach
  a wet answer through the LU recurrence on this card.

**Which arm carries the 1.36e-12: NEITHER — the round-35 preregistration is
REFUTED at the value level.**  It predicted the RHS.  Measured, the RHS arm
is bit-exact and the largest WET residual in any arm is 7.11e-15 in the
sweep, on NEMO's own dumped matrix and NEMO's own dumped right-hand side.
The 1.36e-12 was measured on the model's own trajectory, where the RHS is
NOT the content form — that is the gate's declared blind spot, and no arm
here can reach it.

**RETRACTION — the content-RHS residual was the harness's.**  Round 35
recorded `rhs_content.T` as VALUE-AT-BAR, 1/125 cells at 5.2e-17, and read it
as a property of `thickness_weighted_tracer_content`.  NEMO writes, left to
right (`trazdf.f90:528-529`),

```fortran
zrhs =       (e3t_3d*(1._wp+r3t(Kbb)*tmask)) * pt(Kbb)   &
   & + p2dt * (e3t_3d*(1._wp+r3t(Kmm)*tmask)) * pt(Krhs)
```

i.e. `(p2dt*e3t_Kmm)*T_Krhs`.  The arm fed the builder a PRE-MULTIPLIED
`t_expl = p2dt*T_Krhs`, forming `e3t_Kmm*(p2dt*T_Krhs)` — an association NEMO
never writes.  Under NEMO's own grouping the builder is bit-exact: 0/21120 on
both tracers here, 0/125 on the twin.  The old grouping is kept as a
REPORTED, never scored, association-sensitivity row (17/21120, 2.84e-14 on T;
0 on S) so the retraction keeps its evidence and a harness choice cannot
decide the verdict.

Confirmed directly rather than argued: `e3t_Kbb*T_Kbb + (p2dt*e3t_Kmm)*T_Krhs`
reproduces NEMO's dumped `rhs_T`/`rhs_S` at 0 cells unequal, max 0.

### Item 3 — decision 20 is NOT executed this round, and the reason is measured

The user answered decision 20 "I would go with the NEMO form": NEMO-identity
cards switch `tracer_combine` from `concentration` to the two-term CONTENT
form.  The order set for it was (a) make the content builder bit-exact given
NEMO's operands, (b) prove the whole `tra_zdf` program bit-exact given NEMO's
inputs on GYRE, (c) only then flip the switch — and, explicitly, do not flip
if (a) or (b) cannot be reached.

* **(a) REACHED.**  0/21120 on both tracers, above.
* **(b) NOT REACHED.**  `tra_zdf` is not bit-exact given NEMO's inputs:
  `assembly.zwd` differs on 3120 dry cells (absmax 299.71) and the sweep
  differs on 133 (T) and 111 (S) WET cells at 7.105e-15, on NEMO's own dumped
  matrix and right-hand side.  Both are inside legoESM, not the harness.
* **(c) THEREFORE NOT DONE.**  `tracer_combine` stays `concentration` on
  every card.  Rule 10, printed by the gate rather than assumed:

| card | tracer_combine | zdf_implicit_solver_evaluation |
|---|---|---|
| GYRE-zco | concentration | nemo_literal |
| LOCK_EXCHANGE-zco | concentration | nemo_literal |
| OVERFLOW-zps | concentration | nemo_literal |

No before/after per card and no Rule-12 table are owed, because no switch
moved.  The ASKED record stands with the user's words; the flip is round 37's,
behind the two rows above.

### Item 4 — the divisor becomes NEMO's multiply

ASKED-by-directive under the user's standing "do as NEMO does" instruction.
Preregistered at `e6ef14d5d4f0` before any measurement
(`round36/round36_item4_preregistration.md`).

`stprk3_stg.f90:522-523` writes

```fortran
zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)
```

a MULTIPLY by the reciprocal `domain.f90:213` builds ONCE as
`ssumask/(hu_0 + 1 - ssumask)`, with the dry-column zero INSIDE the reciprocal
and NO mask inside the `SUM`.  `rk3_stage_barotropic_correction` divided by
`hu_0` and multiplied by a separate wet mask.  It now multiplies by a supplied
reciprocal; the `face_mask` argument is gone, because NEMO has no mask there.
One shared implementation, both production call sites and all three gates.

| card | row | before | after |
|---|---|---|---|
| GYRE-zco | rule12_correction.u | exact=False, 2.168e-19 | **exact=True, 0.0** (17400 cells) |
| GYRE-zco | rule12_correction.v | exact=False, 6.939e-18 | **exact=True, 0.0** (17100 cells) |
| LOCK_EXCHANGE-zco | rule12_correction.u | exact=False, 3.081e-33 | **exact=True, 0.0** (2540 cells) |
| OVERFLOW-zps | rule12_correction.u | exact=False, 3.8519e-34 | exact=False, **3.8519e-34** (16900) |

GYRE is discharged on both faces with NEMO's OWN `r1_hu_0`/`r1_hv_0` read from
this round's record — nothing rebuilt, no floor.  It is also bit-exact with the
reciprocal REBUILT from `SUM(e3u_0*umask)`, which is the stronger statement of
the two and is reported per face in `reciprocal_provenance`.

**OVERFLOW: PREDICTION REFUTED.**  Predicted 0/16900 at absmax 0.0; measured
BYTE-IDENTICAL to before.  The reciprocal is therefore proven NOT to be its
owner, and the owner is named rather than guessed: OVERFLOW's column has 101
levels and **XLA's reduction of that sum differs from NumPy's — and from
NEMO's ascending-k `SUM` — on 5 of 606 columns by 4.441e-16**, which is
8.882e-19 through the reciprocal and 3.852e-34 in the corrected velocity.
LOCK_EXCHANGE's 21-level column is 0 of 390.  With the same arithmetic in
NumPy the candidate is bit-identical to NEMO on all 61206 cells, which
reconciles the earlier review's "0/16900": it did not go through XLA.
Registered OPEN, not reverted (Rule 12).

**Second Rule-12 row, raised by the claim review and MEASURED.**  NEMO writes
`uu(jk) + zub*umask(jk)` (`stprk3_stg.f90:541-542`); the operator writes
`(uu + zub)*stage_mask`.  On both tanks 0 of 8190 and 0 of 61206 bits differ,
because every dry face already carries `+0.0`.  On GYRE it is 349 (u) and 212
(v) cells — **ALL of them signed zero, max abs difference exactly 0.0**.
Registered OPEN with its boundary; transcribing NEMO's per-level add is not
round 36's ask.

**ORCA2 is UNMEASURED**, with a spec: no ORCA2 record carries `r1_hu_0` and
no ORCA2 kt=1..10 oracle is in this campaign's data tree.  Closing it needs an
acquisition that appends `r1_hu_0`/`r1_hv_0` to an ORCA2 `dyn_zdf` record,
then the same three rows.

### What two independent reviews broke

codex is unavailable on this account, so both reviews are fresh Claude agents
with no shared context.  The claim review ran on items 2–4's claims; the gate
review attacked the gates directly.  Between them they landed five defects
that are fixed here and three that are registered.

**FIXED, because they undermined this round's own numbers.**

1. **A cropped domain scored 30 cells and read green.**  The box guard
   constrained the halo to be symmetric and never constrained the DOMAIN, and
   `jpi`/`jpj`/`jpk` come from the record's own header.  A record cropped to
   5×5×31 with NEMO's halo of 2 returned exit 0 and `STATUS AT-BAR` out of ONE
   scored column — 704× fewer cells — and **all seven plants still reported
   `landed=True`**, so no control could tell.  The gate now pins the card's
   domain (36×26×31, from the run's own resolved namelist output) and refuses
   any other.  Reproduced against the fix: refused.
2. **The salvage skipped two refusals** every other array gets — the
   duplicate-name check and the rank/extent check.  Both now run first.
3. **EOF counted as proof for the SHORTER extent**, so a final array truncated
   at the tile boundary would have "proven" a salvage that was data loss.  EOF
   now ends the declared extent only.
4. **The "first non-bit statement" was wrong**, above.
5. Two synthetic fixtures inverted the OLD divide and went red on OVERFLOW's
   5 reduction-split columns; they now invert the multiply, so they test the
   plumbing they exist to test.

**REGISTERED, with owners, not fixed here.**

6. **The calibration arm cannot see two slices it never reads.**  Measured,
   not reasoned: setting `avt`'s whole surface plane to 999.0, or the whole
   `jk=jpk` level of nine arrays (11232 values) to 12345.0, leaves all 13
   calibration rows at 0 cells unequal.  Both follow from NEMO's structure
   (`zwt(:,1)=0` at `trazdf.f90:419`; the matrix has `jpkm1` rows), so
   "calibration = 0" certifies the slices the SOLVE reads and nothing else.
   Now declared in the report as `slices_no_arm_reads`.
7. **The admission gate resolves the `zFw` waiver from the BASELINE's
   `ocean.output` only.**  A candidate run with `ln_dynadv_vec = F`, where
   `zFw` IS defined, had an owned-cell change admitted.  The run.sh's
   byte-identical namelist check stands between that and a real acquisition,
   but the gate should not depend on it.
8. **The admission gate cannot read the round-35 record at all** — it has no
   interior-extent recovery, and confirmed here it raises
   `AdmissionError: array '...' rank 1896932436`.  Harmless this round (the
   admission only parses records the BASELINE carries, and this one is new),
   and a HARD BLOCKER the moment a later round inherits round 35 as its
   baseline.  Fixing the instrument removes it at the source, which is why the
   corrected `WRITE` is quoted above.

### ASKED / UNASKED

| choice | status |
|---|---|
| `tracer_combine` stays `concentration` on every card | ASKED — decision 20 answered "the NEMO form", and its own stated precondition (b) is not met, so the flip is not made |
| the divisor becomes NEMO's `* r1_hu_0` | ASKED-by-directive — the user's standing "do as NEMO does" |
| `face_mask` removed from the correction's signature | follows from the above: NEMO has no mask inside the `SUM` |
| the gate pins GYRE's domain and refuses another | a previously-tolerated condition becomes a hard error — taken because the tolerated condition was a measured green-on-wrong-record defeat, and named here rather than left silent |
| the reader recovers `avt`/`avs` at the interior extent | a decoding correction, proven by the stream and by the calibration arm; the instrument fix is still owed |

UNASKED list: **empty**.

### kt = 1..10, before and after item 4

Same gate, same oracle roots, same `--max-step 10` on both sides.  BEFORE was
taken at `e6ef14d5d4f0` in a detached probe worktree
(`/tmp/codex-gyre-r36-before`, flagged), so the two arms differ only in the
operator.

| card | first over bar BEFORE | first over bar AFTER |
|---|---|---|
| GYRE-zco | kt=2, T S u v | kt=2, T S u v |
| LOCK_EXCHANGE-zco | kt=4, u | kt=4, u |
| OVERFLOW-zps | kt=2, T u | kt=2, T u |

BEFORE is stamped `e6ef14d5d4f0`, AFTER `95ebc9a91b67` (LOCK) and
`89630cb22155` (GYRE, OVERFLOW).  The commits between those and the operator
change are the citation map and this receipt; no numerics moved in them.
The stamp ratchet refused every AFTER run taken on a dirty tree, which is why
they are stamped at the commit and not before it.

The first-over-bar step does not move on any card, which is what a change of
this size should do: it removes a handful of ULPs from ONE stage-3 statement
on a trajectory whose kt=2 divergence is owned by `tra_zdf`, measured above.
Reporting it as an improvement would have been a confound.

### Evidence

Under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round36/`,
`round36_evidence.sha256` over its 20 files, itself
`60e3b5f65b3046c6c6e8e2b86a2def9320b138939734fbf5b21d0248e7d025fd`; and
`round35_oracle_trazdf_matrix/round35_outputs.sha256` over the acquisition's
57 files, written as `run.sh` would have.

The full ocean fidelity suite is 972 passed, 7 skipped, 0 failed
(`tests/ocean/fidelity/` plus `tests/ocean/unit/test_zdf_implicit_literal.py`).
Six of those went red mid-round for one reason, and it is worth recording
because it looked like six defects: the stamp ratchet refuses to stamp a
dirty tree, so every gate that stamps fails while an edit is uncommitted.
Four of them were the citation gate, whose real complaint arrived only after
the commit.

One detached probe worktree is flagged: `/tmp/codex-gyre-r36-before` at
`e6ef14d5d4f0`, used to take the BEFORE trajectories against the pre-change
operator.

### What is open

1. `assembly.zwd` — 3120 dry cells, 299.71.  legoESM replaces the dry
   diagonal by 1.0 where `trazdf.f90:445` leaves `e3t`.  Owner: legoESM.
2. `sweep.T`/`sweep.S` — 133 and 111 WET cells at 7.105e-15, on NEMO's own
   dumped matrix and right-hand side, so the difference is inside legoESM's
   three recurrences and nowhere else.
3. `trazdf.f90:443-444` — 704 signed zeros per diagonal, 600 of them wet.
4. The model's OWN right-hand side is still UNMEASURED: the production solve
   takes the RHS as an argument and this card resolves
   `tracer_combine="concentration"`, so no arm here drives it.  Closing it
   needs the `pre_implicit_tracer_content_override` hook on a run.
5. OVERFLOW's column-sum reduction, and the correction's mask placement —
   both registered in that gate with boundary and owner.
6. The round-35 INSTRUMENT still writes `avt`/`avs` short.  The admission
   gate cannot read that record and fails closed; harmless this round, a hard
   blocker the moment a later round inherits round 35 as its baseline.
7. The rounds-33/34 bookkeeping list is untouched except where it blocked:
   nothing on it was needed this round.

## Round 37 — the solve was fused, the dry diagonal was substituted, and the owner is neither

Round 37 starts from `e49063768799` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round37_preregister.json`, committed at
`4569156f07a6`, and it records as MEASUREMENTS -- not as predictions -- the
three localisation results that were taken before it was written.  **No NEMO
executable was run by the agent, no `makenemo` or `mpirun` was invoked, and no
NEMO source was edited.**  The operator ran the round's one acquisition.

Three detached probe worktrees are flagged: `/tmp/codex-gyre-r37-start` at
`e49063768799`, `/tmp/codex-gyre-r37-before` at `4569156f07a6` and
`/tmp/codex-gyre-r37-presum` at `4ba1ffd6b428`, each used to take a BEFORE
against the pre-change model with the same gate source.

### Item 1 — the instrument says what it writes, and the record proves it

`zdf_oce.f90:85-86` allocates `avt` and `avs` over the INTERIOR box while
`avm`, on the same `ALLOCATE`, is full-domain.  Round 35 wrote both raw under
a header declaring `jpi, jpj, jpk`, so each payload was 57536 bytes short of
its own header and every array after `avs` landed at the wrong offset.  The
two `WRITE`s become the staging idiom four lines below them already uses for
`ah_wslp2` and `akz`; `ln_tile` is `F` on this card
(`round29_oracle_v2_zdf_matrix/ocean.output:247`), so `ntsi:ntei,ntsj:ntej` IS
`Nis0:Nie0,Njs0:Nje0` and the copy conforms exactly.

**The staged-temporary form was chosen over declaring the true interior
extents in the header**, because it leaves the record uniform: no reader needs
a special case, and that is what closes round 36's open item 8 at the source
-- the shared admission gate has no interior-extent recovery and raises on the
round-35 record.

The delta patches the ROUND-35 CARD's own `MY_SRC`, so round 35's "the patch
may only add lines" guard no longer states the invariant it stood for.
`run.sh` checks the stronger form on the BUILT file: `MY_SRC/trazdf.F90` must
differ from NEMO's shipped `trazdf.F90` by additions only.  Measured before
the commit and re-checked by `run.sh` before building: **0 removed lines**.

The acquisition ran.  The new record is 6965416 bytes, exactly 115072 = 2 x
57536 longer than round 35's.  **The field-by-field twin, through the gate's
own reader -- the only one that can decode both -- is PASS: all 44 arrays bit
for bit what round 35 wrote, `avt` and `avs` included, 0 failures; round 35's
record still needs the salvage and round 37's does not** (`tile_shaped_salvage`
is `{}`).  The admission is PASS with an empty owned-field difference list,
its plant turns it red, the round-29 momentum regression is AT-BAR, and
`round37_outputs.sha256` is written.

### Item 2 — THE OWNER IS NEITHER THE SOLVE NOR ITS RIGHT-HAND SIDE

Round 34 attributed GYRE's kt=1 stage-3 tracer residual,
`3.1956659540810506e-11` K on T, to "the stage-3 implicit vertical tracer
solve itself".  Three measurements this round, in NEMO's execution order:

| input to the tracer implicit solve | scored against | result |
|---|---|---|
| the whole matrix + RHS, given NEMO's own | NEMO's dumped solved column | **0 / 21120**, both tracers, after this round's two fixes |
| the model's OWN content right-hand side, captured LIVE through `expose_pre_implicit_content` in the model's own stage-3 path | NEMO's own `e3t(Kbb)T(Kbb) + p2dt e3t(Kmm)T(Krhs)` | **6.821210263296962e-13** K.m on T, **3.637978807091713e-12** psu.m on S, 18000 cells |
| NEMO's own content RHS SUBSTITUTED into the model | the model's own faithful run | the output moves by **5.786374251651969e-16** K |

**The stage-3 output residual is UNCHANGED by both of this round's solve
fixes**: `3.1956659540810506e-11` K before and after, to every digit.  And
substituting NEMO's own right-hand side moves the output by `5.79e-16`, which
is `4.3e-4` of the residual.  So the right-hand side differs, and it is
CAUSALLY INERT at this size: it cannot own the residual either.

**PR3 IS REFUTED, twice, and by two independent routes.**  It predicted the
right-hand-side FORM would reproduce the residual to within a factor of a few.
The first route: built from NEMO's own operands, legoESM's supposed
concentration form `e3t(Kaa)*(T(Kbb) + rDt*T(Krhs))` through NEMO's own matrix
gives `1.5458154965841686e-05` K -- **484000x too large**, which by itself
proves the model does not evaluate that expression.  The second route is the
substitution arm above.  The falsifier the preregistration wrote (a residual
three orders under the trajectory's) is what occurred.

**A ROUND-35/36 CLAIM IS RETRACTED.**  Both rounds recorded that the NEMO
cards resolve `tracer_combine="concentration"`, therefore
`_nemo_tracer_content_rhs` stays `None`, therefore the literal `trazdf` matrix
is fed `T_solve_in * dz_cell`.  The first clause is true and the conclusion is
false for the path these cards run.  The WS-RK3 stage ladder sets its own
`_nemo_ws_tracer_content_rhs` from the tracer stage program's
`return_final_content`, independently of `tracer_combine`, and that is what
the solve consumes.  Measured, not read: the `expose_pre_implicit_content`
hook RAISES when that value is `None`, and the content-mode gate ran and
produced two rows -- so it is not `None` on GYRE.  **The model has been
feeding the literal matrix a CONTENT right-hand side all along.**

**So the owner is inside the MATRIX legoESM builds, or downstream of the
solve, and this round did not close it.**  The frame spec for the next round
is the one arm nothing has: the model's own `K_v_cell`, `dz_cell`,
`dz_half_cell` and wet mask at the moment the literal solve is called, scored
against the record's `zwt_mix`, `e3t_Kaa`, `e3w_Kmm` and `tmask`.  There is no
hook for them; `expose_pre_implicit_state` publishes the tracers only.  On
GYRE the matrix diffusivity is a LIVE TKE closure output, which is the one
operand two independent codes are least likely to agree on to the last bit.

### Item 3 — the sweep, and why it looked like the back substitution

Given NEMO's own dumped matrix and right-hand side, legoESM's ordered solve
differed on 133 (T) and 111 (S) WET cells at `7.105e-15`.  **Every one of
those cells sits in a column with NO dry cell**, so neither the dry diagonal
nor the `tmask` placement could have owned them.

The cause, identified two-sidedly rather than argued: a plain NumPy sequential
transcription of `trazdf.f90:493`/`:496`, `:515-516`/`:528-529`/`:532` and
`:543`/`:546-547` reproduces NEMO's dumped LU diagonal, forward sweep and
solved column at **0 of 21120**; the SAME transcription with `math.fma` in the
second AND third recurrences reproduces legoESM's JAX answer at **0 of
21120**, both tracers; with the fusion in only one of the two it reproduces
neither (63 and 80 cells left over on T).  **XLA on CPU contracts `a - b*c`
into a fused multiply-add and gfortran does not.**  The first recurrence ends
in a division and cannot be contracted, which is why its row was never over
the bar and the residual looked like it lived in the back substitution.

`jax.lax.optimization_barrier` does NOT stop it.  On 4096 random triples:
plain, a barrier on the product, a barrier on the `(product, source)` tuple,
`lax.reduce_precision(.., 11, 52)` and a bitcast round trip all reproduce the
FUSED answer on all 812 cells where fused and separate differ.  Inside a
`lax.scan`, an extra unused output and an unused carry slot are both
eliminated, so the usual "give the multiply a second use" trick does not
survive.  `XLA_FLAGS=--xla_allow_excess_precision=false` and
`--xla_cpu_enable_fast_math=false` change nothing.

What works is making the subtraction's operand an ADD, since a fused
multiply-add can only absorb a multiply: `x + copysign(0.0, x)` is exactly `x`
for every finite input and for BOTH signed zeros -- which matters, because
NEMO's own `zwi` and `zws` carry negative zeros -- and unlike `x*1.0` or
`x-0.0` the compiler cannot fold it away.

The dry diagonal is the second half.  `trazdf.f90:445` is written
unconditionally, so a dry row reduces to its own `e3t`; legoESM substituted
`1.0`.  The substitution is gone, with the precondition it rests on stated at
the return: the layer thickness is strictly positive below the seafloor as
well as above it, exactly as it is in NEMO.  And the two slots the recurrences
never read -- `zwi` at the surface row, `zws` at the bottom -- are NEGATIVE
zeros in NEMO (`:419` sets `zwt(:,1) = 0`, `:443-444` divide it), where
legoESM wrote positive zeros into both, 704 cells each, 600 of them wet.

**THE DRY-DIAGONAL HALF WAS LANDED AND THEN TAKEN BACK OUT, AND THAT IS THIS
ROUND'S MOST IMPORTANT FINDING.**  With the substitution removed, OVERFLOW's
`kt=2` tracers went **NON-FINITE**.  The mechanism, measured on all three
cards rather than reasoned: NEMO's `e3t_3d` is the positive REFERENCE
thickness below the seafloor as well as above it, so a dry row of its matrix
reduces to a nonzero `e3t`; legoESM's `h_partial` is **EXACTLY `0.0` at every
dry cell** -- 3120 of GYRE's 21120, 5240 of LOCK_EXCHANGE's 7800, 43600 of
OVERFLOW's 60600.  Without the `1.0` the diagonal is zero, the solve divides
by it, and the resulting infinity multiplies the `-0.0` off-diagonal above it
into a NaN.

**THE BLIND SPOT IS WORTH WRITING DOWN.**  The arm that certified the change
used NEMO'S OWN `e3t_Kaa`, which satisfies the precondition.  "Bit-exact given
NEMO's inputs" is structurally incapable of seeing a defect whose entire
content is that legoESM's inputs are not NEMO's.  Rule 12's discharge is
necessary and, for a change that reads geometry, not sufficient; the tanks'
trajectories are what caught it, which is why the round's own instruction
required them.

So the substitution is back, REGISTERED as a deviation rather than presented
as a transcription, with the measurement in the code beside it.  Closing it
means giving legoESM NEMO's reference thickness below the seafloor -- a
geometry change, not this function's to make.

Given NEMO's inputs, on the round-37 record, 21120 scored cells, at the
round's final tip:

| arm | row | before | after |
|---|---|---|---|
| calibration | all 13 rows | 0 / 21120 | 0 / 21120 |
| assembly | `zwi` | 704 signed zeros | **0 / 21120** |
| assembly | `zwd` | 3120 at 299.71 | 3120 at 299.71 — **REGISTERED**, owner: legoESM's zero dry-cell thickness |
| assembly | `zws` | 704 signed zeros | **0 / 21120** |
| sweep | T | 133 at 7.105e-15 | **0 / 21120** |
| sweep | S | 111 at 7.105e-15 | **0 / 21120** |
| rhs_content | T, S | 0 / 21120 | 0 / 21120 |
| clamp | T, S | 0 / 21824 | 0 / 21824 |

**RULE 12, discharged on every card that executes the changed sweep.**  The
same `_nemo_ordered_solve` is called by the MOMENTUM solve, so the round-29
gate grows an arm that drives it on NEMO's own dumped momentum matrix and
right-hand side -- the same arm the two tanks' round-33 records can be scored
with, so no card is left UNMEASURED.  The surface-stress statement is factored
out of `nemo_solve` so the new arm shares it rather than growing a copy.

| card | row | before | after |
|---|---|---|---|
| GYRE-zco | `zdf_solve_lego.u` | 582 / 21120 at 2.082e-17 | **0 / 21120** |
| GYRE-zco | `zdf_solve_lego.v` | 623 / 21120 at 1.388e-17 | **0 / 21120** |
| LOCK_EXCHANGE-zco | `zdf_solve_lego.u`, `.v` | 0 / 7800 | 0 / 7800 |
| OVERFLOW-zps | `zdf_solve_lego.u`, `.v` | 0 / 60600 | 0 / 60600 |

The dry-diagonal change reaches only the tracer pair solve, and **the tanks
have NO tracer-matrix record** — which is exactly why the round's instruction
required their trajectories before landing on them, and exactly what refused
the change.  The frame spec for closing that gap properly is a `trazdf`
instrument on each tank card writing `zwi`/`zwd`/`zws`/`rhs`/`sol` at
`kt = nit000`, as round 35 did for GYRE; but it would not have caught this
one either, for the reason above.

The new test builds a matrix on which the fused and unfused answers differ in
2709 of 12288 cells and asserts the solve matches the unfused one.  It fails
with exactly that count on the parent commit.

### Item 4 — the column sum accumulates the way Fortran's SUM does

`stprk3_stg.f90:522-523` writes `SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) )`.
**There is no `DO jk` loop for this sum in NEMO** -- the correction's `DO jk`
loop at `:540-543` applies `zub`, it does not form it -- so what is
transcribed is the intrinsic's emitted order, and the standard does not fix
it.  The evidence for the order is the measurement, not the citation.
ASKED-by-directive under the user's standing "do as NEMO does".

| card | row | before | after |
|---|---|---|---|
| GYRE-zco | `rule12_correction.u` | exact=True, 0.0 (17400) | exact=True, 0.0 |
| GYRE-zco | `rule12_correction.v` | exact=True, 0.0 (17100) | exact=True, 0.0 |
| LOCK_EXCHANGE-zco | `rule12_correction.u` | exact=True, 0.0 (2540) | exact=True, 0.0 |
| OVERFLOW-zps | `rule12_correction.u` | exact=False, **3.8519e-34** (16900) | **exact=True, 0.0** |

**PR5 IS CONFIRMED and round 36's registered OPEN row is DISCHARGED.**
OVERFLOW's 101-level column was the only one long enough for XLA's tree to
disagree with an ascending accumulation; LOCK_EXCHANGE's 21-level column and
GYRE's never did, and neither moved.

### Item 3's last question — decision 20 is NOT executed, and the reason changed

The order set for decision 20 was (a) make the content builder bit-exact given
NEMO's operands, (b) prove the whole `tra_zdf` program bit-exact given NEMO's
inputs, (c) only then flip `tracer_combine`.  (a) was reached in round 36.
**(b) is NOT reached** -- the dry-diagonal row above is still 3120 cells --
and, independently, **the premise of the flip is refuted.**  On the WS-RK3 path
these three cards run, the literal matrix already consumes a CONTENT
right-hand side, built by the tracer stage program and not by
`tracer_combine`; the flip would change a field the stage ladder does not
read.  Flipping it would be an unasked change of behaviour on paths nobody in
this campaign measures, in exchange for nothing on the paths that matter.

The ASKED record stands with the user's words, "I would go with the NEMO
form", and what the model does is already the NEMO form.  What is owed instead
is the row that says so per card, and the disposition of a config field that
these cards do not read.

### What two independent reviews broke

codex is unavailable on this account, so both reviews are fresh Claude agents
with no shared context: one attacked the CLAIMS, one attacked the DIFF and was
told to make the sweep discharge pass on a wrong solve and the flip land
without its per-card rows.  Between them they landed five defects that are
fixed here and three that are registered.  **Both reviews independently
reproduced the dry-diagonal defect, with the same counts, one of them before
the revert was pushed and with a two-sided control the round did not run:
removing the substitution gives 2800 NaN of 60600 on OVERFLOW step 1,
restoring it gives 0.**

**FIXED, because they undermined this round's own numbers.**

1. **THE SWEEP DISCHARGE WAS DEFEATED.**  `AT-BAR-SIGNED-ZERO` was a PASSING
   status.  The reviewer made the solve return `out*3 + 12345` in every column
   whose off-diagonals are all zero; the gate printed `sweep.T 5/21120 max 0`,
   classified it as a signed-zero row, and **exited 0 with STATUS AT-BAR**.
   The exemption existed because legoESM wrote `+0.0` where NEMO writes
   `-0.0`, and that transcription landed this round, so no row needs it.  It
   is no longer a passing status; the classification stays, so a `+0.0`
   regression is still named rather than folded into AT-BAR.
2. **AND THE SWEEP ARM HAS A BLIND SPOT IT CANNOT CLOSE**, now named in its
   docstring: NEMO's back substitution masks at every level, so both sides are
   zero at all 3120 dry cells of the scored box and a solve arbitrarily wrong
   there scores as equal.  What covers them is the round-29 momentum arm,
   whose oracle is NEMO's UNMASKED `uu(Kaa)` -- the same corruption moved 3720
   of its 21120 cells by 12345.
3. **THE NEW MOMENTUM ARM HAD NO NON-VACUITY CONTROL.**  The round-29 plant
   moved `avm`, which the rebuilt matrix reads and the solve arms do not: with
   `--plant` every `zdf_matrix` row moved and both `zdf_solve_lego` rows
   stayed at `0/21120`.  The plant now also moves the dumped diagonal.
   Measured after the fix: unplanted `0`/`0`, planted **1748 / 1758**.
4. **THE ANTI-FUSION HELPER'S DOCSTRING PROMISED MORE THAN IT DELIVERS.**
   Under `jit`, `x + copysign(0.0, x)` FLUSHES a subnormal to a signed zero
   (5e-324, 1e-308, 1.5e-310 all become 0.0) where `x + 0.0` and `x * 1.0`
   preserve them.  Inert in this solve on these cards -- the products are of
   order 1e-4 to 1e4, and XLA's CPU scan already flushes a computed subnormal
   product either way -- but the sentence was false and is now the measured
   statement.
5. **A TANK FIXTURE RE-IMPLEMENTED THE REDUCTION IT EXISTS TO HOLD FIXED.**
   It built its inverted target with `jnp.sum`; after item 4 that would have
   reintroduced exactly the 5-column OVERFLOW difference the change removed.
   It imports the operator's own accumulation instead.

**REGISTERED, with owners, not fixed here.**

6. **`tracer_combine` IS INERT ON ALL THREE CARDS**, measured by the diff
   review rather than read: each card built twice, one step, every float leaf
   compared -- **0 of 198956 (GYRE), 0 of 47262 (LOCK), 0 of 328152
   (OVERFLOW)** bits differ.  The field is read only by `_leapfrog_step` and
   `_nemo_mlf_step`; the NEMO cards run the WS-RK3 ladder, which never reads
   it.  That is the measurement behind this round's retraction, and it makes
   the pending flip a decision with no effect to decide.  A lever nothing
   selects is a defect; its disposition is a question for the coordinator, not
   this round's to take.
7. **The round-29 gate is RED on both tank records for a pre-existing reason**
   -- `zwi_u`/`zwi_v` are VALUE-AT-BAR signed zeros, 4997/7800 and
   43293/60600 at max 0, reproduced at the round's parent commit.  The
   `zdf_solve_lego` rows quoted above are real measurements taken from that
   run; the run's own exit code is 1 and that is said here rather than left
   for a reader to discover.
8. **The instrument's write-only guard inspects `trazdf.F90` only.**  It is
   not defeatable as written -- `diff` emits `^<` for a changed line too -- but
   the card's `stprk3_stg.F90` removes one shipped line (a single-line
   `IF( ln_traqsr ) CALL tra_qsr` rewritten as a block), and that is the file
   item 4 cites.  The guard's scope is the round-37 delta; widening it to
   every `MY_SRC` file is owed.
9. **The claim review weakened one framing**: "0 cells unequal both ways"
   identifies the ARITHMETIC legoESM performed, not the compiler mechanism.
   "XLA contracts" is read off the code; what is MEASURED is that legoESM
   computed a single-rounded product in exactly those two spots.  It also
   corrected the reason the first recurrence is immune: not "it ends in a
   division" but that the subtrahend is a QUOTIENT, which an FMA cannot
   absorb.  And it noted the new unit test would pass vacuously on a target
   where XLA does not contract -- its control proves the reference pair
   differ, not that the compiler would fuse.
10. **The claim review settled item 2's arithmetic independently.**  The
    matrix's row sums are exactly `e3t`, so `||M^-1||_inf` over every wet
    column is `0.09996`; a `6.821e-13` K.m right-hand-side difference can move
    the output by at most **6.82e-14 K, 469x short of `3.196e-11` K**.  The
    residual demands a RELATIVE MATRIX difference of about `1.4e-12`, i.e.
    roughly `1e-13` m2/s of `avt`.  That is the next round's arm.

### kt = 1..10, before and after

Same gates, same oracle roots, same `--max-step 10` on both sides.  BEFORE was
taken at `e49063768799` in a detached probe worktree (flagged), so the two
arms differ only in the model.

| card | first over bar BEFORE | first over bar AFTER |
|---|---|---|
| GYRE-zco | kt=2, T S u v | kt=2, T S u v |
| LOCK_EXCHANGE-zco | kt=4, u | kt=4, u |
| OVERFLOW-zps | kt=2, T u | kt=2, T u |

BEFORE is stamped `e49063768799`; GYRE's AFTER is stamped `ba0752aec5b1` and
the two tanks' `1fdd853a19b7`.  The commits between them are tests and this
receipt; no model numbers moved in them.

**No card's first-over-bar step moves**, which is what a change of this size
should do: it removes a handful of ULPs from operators whose kt=2 divergence
is owned by something else -- measured directly this round, since the stage-3
tracer residual is `3.1956659540810506e-11` K before AND after, to every
digit.  Reporting any of it as an improvement would be a confound.  OVERFLOW's
AFTER is FINITE, which is the trajectory-level confirmation of the revert.

### Gates and evidence

The round-35 gate on the round-37 record, on a CLEAN tree at the round's final
tip: every row AT-BAR with 0 bits unequal except `assembly.zwd`, which is the
registered 3120-dry-cell deviation and is DEBT by design.  `SCORED 21120
cells, i [3, 34] j [3, 24] halo 2, on a [36, 26, 31] domain`, and no array
needed the interior-extent salvage.

The receipt citation gate is `PASS` over **190 citations**, none unmapped, no
failures.  Eight citations were added this round and each is pinned to its
source text; two were caught by the gate before this was written -- one named
a run whose `ocean.output` the gate cannot resolve, and one anchored on an
`END DO` that occurs eight times in the file.

The full ocean-fidelity suite at the final tip, on a detached worktree:
**`980 passed, 7 skipped, 18 deselected in 2566.82s`**, zero failures.  An
earlier run mid-round had three failures and all three were this round's own
stale expectations -- the round-29 row count, the tank fixture's reduction,
and the citation map -- each fixed at the source rather than by relaxing the
assertion.

Evidence under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round37/`,
`round37_evidence.sha256` over its 15 files, itself
`a0fd35bb84a9ce1adeacba3085962a7454efa3426f2d1b494d279aa12aa3573b`; and
`round37_oracle_trazdf_matrix/round37_outputs.sha256` over the acquisition's
own outputs, written by `run.sh`.

### ASKED / UNASKED

| choice | status |
|---|---|
| `tracer_combine` stays `concentration` on every card | ASKED — decision 20 answered "the NEMO form", precondition (b) is not met, and the flip is measured INERT on all three cards (0 of 198956 / 47262 / 328152 bits).  Not made |
| the ordered solve stops being fused into an FMA | ASKED-by-directive — the user's standing "do as NEMO does"; it makes legoESM round NEMO's written statement the way gfortran rounds it |
| the two boundary slots become NEGATIVE zeros | same directive; inert by construction, and it removes a bit difference a gate at the exact bar can see |
| the barotropic column sum accumulates in ascending k | ASKED-by-directive, same standing instruction |
| the dry-diagonal substitution was removed and PUT BACK | the removal was the directive's transcription; putting it back is not a preference but a refusal to ship non-finite state on a card the change cannot be measured on.  Registered as a deviation with its owner |
| `AT-BAR-SIGNED-ZERO` stops being a passing status | a previously-tolerated condition becomes a hard error — taken because the tolerated condition was a measured green-on-wrong-solve defeat, and named here rather than left silent |
| the instrument stages `avt`/`avs` into a full-domain temporary rather than declaring their true interior extents | a serialisation choice, not a scientific one: it leaves the record uniform so no reader needs a special case, which is what closes round 36's admission blocker at the source |
| the admission baseline stays round 29 rather than round 35 | a harness choice, and the reason is round 36's open item 8: the admission cannot parse the round-35 record |
| detached probe worktrees | ASKED; the same disposition rounds 32-36 recorded.  Flagged: `/tmp/codex-gyre-r37-start`, `/tmp/codex-gyre-r37-before`, `/tmp/codex-gyre-r37-presum`, `/tmp/codex-gyre-r37-fma`, `/tmp/codex-gyre-r37-suite` |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed by the agent.  The operator ran the acquisition |

UNASKED list: **empty**.

### Merge readiness

`03c6e8d96ff7` remains an ancestor, so the integration is still a
fast-forward.  **HOLD.**

1. **GYRE's `kt2` `T`/`S`/`u`/`v` rows still fail**, unchanged, and this round
   measured that the tracer solve does not own them.
2. **`tra_zdf` is bit-exact given NEMO's inputs on every arm except the dry
   diagonal**, which is registered with a named owner and a named cure.
3. **The owner of the stage-3 residual is inside the matrix legoESM builds, or
   downstream of the solve.**  Its arm does not exist yet and the frame spec
   is written above.
4. OVERFLOW's barotropic-correction row is DISCHARGED; its mask-placement row
   and round 30's `dyn_ldf` row are unchanged.
5. ORCA2 has no card on this branch, unchanged.
6. Rounds 1-24 of this receipt remain UNAUDITED by the citation gate.

### What is open

1. `assembly.zwd` — 3120 dry cells at 299.71.  Owner: legoESM's layer
   thickness is exactly `0.0` below the seafloor where NEMO's `e3t_3d` is the
   positive reference thickness.  Cure: give legoESM NEMO's reference
   thickness there; that is a geometry change.
2. **The model's own matrix operands are UNMEASURED** against the record.
   Frame spec above; the claim review's `||M^-1||` bound says the residual
   needs about `1e-13` m2/s of `avt`, so that is where to look first.
3. `tracer_combine` is a field these cards do not read.  Disposition owed.
4. The instrument's write-only guard covers one file of eleven.
5. The round-29 gate is red on both tank records for a pre-existing
   signed-zero reason.
6. The causal injection arm at the stage-3 boundary, the tanks' operator-level
   Rule-12 for decision 19, round 30's `dyn_ldf` row, the moved trajectory
   rows from rounds 32-34, the slow forcing's depth average: unchanged.
7. The rounds-33/34 bookkeeping list is untouched except where it blocked;
   nothing on it was needed this round.

## Round 38 — the matrix's operands, and the owner is the isoneutral fold

Round 38 starts from `09fafdb0fd42` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round38_preregister.json` at `5f5859165c97`,
and its AMENDMENT — written after an independent claim review broke six of its
predictions and BEFORE the arm existed — is
`manifests/nemo_testcase_l2_gyre_round38_preregister_amendment.json` at
`e222ee969a3d`.  **No NEMO executable was run, no `makenemo` or `mpirun` was
invoked, and no NEMO source was edited.**  One detached probe worktree is
flagged: `/tmp/codex-gyre-r38-suite`, used for the suite.

### Item 1 — THE OWNER IS THE ISONEUTRAL FOLD IN THE MATRIX legoESM BUILDS

Round 37 exonerated both halves of GYRE's kt=1 stage-3 tracer residual,
`3.1956659540810506e-11` K on T: given NEMO's own dumped matrix and right-hand
side the ordered solve is bit-exact, and substituting NEMO's own right-hand
side moves the output `5.786374251651969e-16` K.  What was left was the MATRIX
legoESM BUILDS.  This round captured its operands THROUGH THE MODEL'S OWN PATH
and scored them, and then asked the causal question directly.

**The operand table**, on the round-37 record, WET cells only, at the exact
bar.  The face rows are the 17400 wet interior faces, the cell rows the 18000
wet cells of the same 21120-cell box.

| operand | oracle | wet cells unequal | max abs | max relative | argmax cell |
|---|---|---:|---:|---:|---|
| `K` (the matrix diffusivity) | `zwt_mix` | **17383 / 17400** | **9.66209e-13** | 8.05174e-08 | `[19, 29, 1]` |
| `K33` (the isoneutral fold) | `ah_wslp2` | **17400 / 17400** | **9.66209e-13** | — | `[19, 29, 1]` |
| `dz_after` | `e3t(Kaa)` | 5207 / 18000 | 1.13687e-13 | 4.27045e-16 | `[1, 18, 25]` |
| `e3w_now` | `e3w(Kmm)` | **0 / 17400** | **0** | 0 | — |
| `wet` | `tmask` | **0 / 18000** | **0** | 0 | — |
| `dt` | `rDt` | **0 / 1** | **0** | 0 | — |

Every row above was REPRODUCED at the round's final tip, from the unplanted
rows of the plant runs: the same counts, the same maxima and the same
attribution figures as the report stamped at `e222ee969a3d`.

The first differing operand in NEMO's own statement order is `K`.  Its entire
difference is the FOLD, and that is arithmetic, not a story: the largest wet
difference between legoESM's `K` and NEMO's `zwt_mix` is `9.66209e-13`, and
with legoESM's own fold subtracted back out the largest difference against
NEMO's `avt` is `1.73472e-18`.  **NEMO's `ah_wslp2` is IDENTICALLY 0.0 on this
record** (absolute maximum 0, and its companion `akz` likewise), where
legoESM's `K33` reaches `9.6620886711883531e-13` m²/s.

**THE CAUSAL ARM SETTLES IT.**  Substituting one array at legoESM's own tracer
solve call site and re-running the stage-3 completion gate:

| arm | stage-3 T (K) | stage-3 S (psu) |
|---|---:|---:|
| faithful | `3.1956659540810506e-11` | `8.1712414612411521e-13` |
| NULL substitution — legoESM's OWN captured `K`, as a host array | `3.1956659540810506e-11` | `8.1712414612411521e-13` |
| `K` := NEMO's own `zwt_mix` | **`1.4210854715202004e-14`** | **`2.8421709430404007e-14`** |
| `K` := NEMO's `avt` + legoESM's own fold | `3.1956659540810506e-11` | `8.2422957348171622e-13` |

The matrix diffusivity owns **99.96 per cent** of the T residual — strictly,
it owns it TO WITHIN `1.4210854715202004e-14` K, and that bound covers the
second operand that also differs: `dz_after`'s 5207 wet cells at `1.13687e-13`
are inside it and are never substituted separately.  That `1.42e-14` is the
same figure round 34 recorded for `kt1.stage3.pre_zdf`, the residual `tra_zdf`
is HANDED, to every digit — a CROSS-ROUND comparison, not an arm this round
ran.  The CLOSURE owns exactly `0` on T and `7.105427357601002e-15` on S; the
FOLD owns `3.19424e-11` of the `3.19567e-11`.

The null substitution is the noise floor that makes those two readable, and it
is `0` on both tracers — **in the max-abs metric, which is the one thing it can
say**.  The same substitution moves 4174 T bits inside the step, and a max-abs
residual cannot see that; what the floor establishes is that constant lowering
does not move the number this table reports, not that it moves nothing.

**THE UPSTREAM MODULE, AND THE FIRST NON-BIT STATEMENT.**  It is not a
transcription of the isoneutral formula — it is the CALL SITE.

* NEMO computes the neutral slopes ONCE PER STEP, on the BEFORE state, OUTSIDE
  the RK3 stage loop: `stprk3.F90:174` `CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )`.
  `traldf_iso.F90:135` then fills `ah_wslp2` from those slopes, and
  `trazdf.F90:173` reads `zwt = avt + ah_wslp2` at every stage.
* legoESM recomputes its `K33` INSIDE each stage from that stage's own
tracers: `ocean_model_latlon_cgrid.py:8795` sets `_T_gm_in = T_mid` on this
path — `_ldf_state` is `None` outside `_nemo_mlf_step` — and `ocean_model_latlon_cgrid.py:9162` hands it
  to `compute_isoneutral_K33_latlon`.

At `kt = nit000` GYRE has `ln_rstart = F` and its analytic initial T and S are
horizontally UNIFORM on wet cells.  MEASURED, not assumed, on both sides:
NEMO's dumped `T_Kbb_in` and `S_Kbb_in` have a per-level wet range of exactly
`0.0` on all 31 levels, and so do legoESM's own initial T and S.  So NEMO's
slopes are exactly zero and its `ah_wslp2` is identically zero at every stage
of step 1.

**THE DISCRIMINATOR, because two mechanisms could produce a non-zero fold and
they must not be collapsed.**  legoESM's own `compute_isoneutral_K33_latlon`,
asked for its value on that same step-entry state, returns **EXACTLY 0.0 on
all 20416 faces**.  So legoESM's slope arithmetic leaves no rounding residue on
a horizontally uniform field, and the difference is the STATE the function is
handed, not the function.  This arm calls the function directly rather than
through the model and is a property of that function, not of the model's call:
it omits ten arguments `ocean_model_latlon_cgrid.py:9162` passes.  Two of them
were measured inert by the diff review (`u_mask`/`v_mask` threaded: 0 bits
changed), and the zero is not degenerate (a `1e-12` tracer bump gives
`4.69e-31`).

**WHICH INPUT OF THAT CALL DOMINATES IS UNMEASURED, AND THE FIRST VERSION OF
THIS SECTION OVERCLAIMED IT.**  It read "the owner is the CALL SITE" as
"the owner is the tracer TIME LEVEL".  The diff review ran the scaling test
nobody had: injecting NEMO's own stage (`Kmm`) per-level tracer anomaly, whose
absolute maximum is `7.25e-6` K, into the step-entry state gives a K33 of
`7.84e-17` — **1e-4 of the live `9.66e-13`** — and it scales as the square of
that anomaly (`x0.5` -> `1.96e-17`, `x2` -> `3.13e-16`), which is the right
direction and the wrong magnitude.  A separate `eta` perturbation of `1e-3` m
gives `8.09e-16`, so the thickness/`eta` time level is UNEXCLUDED.  What is
MEASURED is that the call reads the STAGE state where NEMO's reads the BEFORE
state, and that the before state produces exactly zero; which component of the
stage state carries the remaining four orders of magnitude is the next round's
question, not this one's answer.

**THE PREREGISTRATION'S FIRST PR2 IS REFUTED.**  It named the TKE closure and
the Prandtl/avt derivation as the culprit, on the reasoning that a live
closure is the operand two codes are least likely to agree on.  Measured, the
closure owns exactly `0` of the residual and its `avt` agrees with NEMO's to
`1.73e-18`.  The AMENDED PR2 — a ranked hypothesis naming the fold — is
CONFIRMED, and so are the amended PR1, PR3 and PR4.

**A RULE-10 ERROR, SELF-CAUGHT AND RECORDED.**  The first reading of this
card's GM/Redi configuration printed `physics.lateral_mixing.gm_redi` and got
`implicit_K33 = False`, `slope_positions = "mode_b"` — the library defaults on
an object the model never reads.  The RESOLVED config the step uses is
`_cfg_b.gm_redi`: `implicit_K33 = True`, `slope_positions = "nemo_native"`,
`slope_scheme = "nemo_iso_lap"`.  Acting on the first print would have
concluded the fold was not even active.  The gate prints the resolved values
now.

**THE INSTRUMENT, AND THE THREE CONTROLS IT NEEDED.**  The capture wraps the
two functions the compiled step calls — the tracer-pair dispatch, reached
through a function-scope import so a module-attribute patch is picked up at
trace time, and `_apply_implicit_vertical_mixing`, which receives the fold —
and emits `jax.debug.callback` before delegating.

1. **The call count.**  The step's jit cache key is the MODEL INSTANCE, so a
   patch shadowed by an already-compiled executable would capture nothing and
   read as measuring zero.  Each arm builds a fresh card and model, and the
   gate refuses a run in which either point did not fire exactly once.
2. **Inertness.**  The model's T and S are bit-identical with and without the
   capture, `0` of `21120` on each.
3. **The IN-GRAPH IDENTITY CHECK, which replaced a control that failed.**  The
   first version substituted the captured host arrays and compared the step's
   OUTPUT.  It moved **4174 T bits and 3925 S bits** — with values that are
   bit-identical, which the in-graph check proves at `0` bits unequal on every
   operand.  A host array lowers as a CONSTANT and changes fusion downstream.
   That number is reported beside the control rather than discarded, because
   it is a warning for every substitution arm this campaign runs; the causal
   arm's own null substitution shows the effect does not reach the stage-3
   residual maximum at all.

`implicit_w` is a matrix operand and was off the preregistration's list.  It is
captured and a non-`None` value is a HARD FAILURE, not a row: GYRE's record
pins `ln_zad_Aimp = F`.

**THE PLANT CONTROL WAS DEFEATED TWICE, AND THE SECOND TIME IT DEFEATED
ITSELF.**  Its first form asked only whether the gate exited non-zero — which
three DEBT rows already guarantee, so a plant that did nothing would have read
as passing; the diff review demonstrated that by neutering the plant.  Its
second form asked whether the planted operand's OWN ROW moved, scored both
ways, and then reported `False` on `--plant K`: one ulp on the largest `|K|`
cell is `1.7e-18` against a row whose absolute maximum is `9.66e-13` and whose
17383 of 17400 cells already differ, so the row's count, maximum and status
are all untouched.  A control that cannot see its own plant proves nothing,
and it said so and exited 3.  Each row now carries a SHA-256 of its scored
candidate bytes.  Measured at the round's tip: `--plant K`, `--plant e3w` and
`--plant wet` each report `moved_its_own_row True` and exit 1.

**THE RECORD GAP, with its frame spec and `run.sh`.**  A record whose
isoneutral term is identically zero cannot discriminate ANY transcription of
it — it can only say whether the candidate's fold is also exactly zero.  So
`nemo_testcase_l2_gyre_round38_trazdf_kt2/` widens the existing instrument
from one step to two: `ll_l2_tra = ( lwp .AND. kt <= nit000 + 1 )` and a
per-`kt` file name, nothing else.  `kt = nit000 + 1` is the first step whose
BEFORE state carries the horizontal structure step 1 created.  The delta
removes none of NEMO's own lines (measured: `0`), and the acquisition REFUSES
the run unless the `kt = nit000` record comes back BYTE-IDENTICAL to round
37's.  The agent did not run it.

### Item 2 — THE FMA POLICY: a global flag exists, and a NAME SEARCH MISSED IT

Round 37 fixed XLA's fused multiply-add contraction PER SITE.  The first
version of this round's answer was that no global flag exists: the installed
`jaxlib 0.10.0` advertises **404** `xla_*` flags and none of them NAMES `fma`,
`contraction` or `fp-contract`, and LLVM's `--fp-contract` is not a registered
option in this build (`--xla_backend_extra_options=--fp-contract=off` is
refused with *Unknown command line argument*).

**THAT ANSWER WAS WRONG, AND AN INDEPENDENT DIFF REVIEW FOUND THE FLAG.**  A
fused multiply-add is an AVX2/FMA3 instruction, so CAPPING THE ISA below AVX2
removes it — at full optimisation.  Nothing in the flag's name says `fma`,
which is exactly how a search over flag NAMES missed it.  Measured, 4096
triples drawn so the two roundings differ on all of them:

| setting | reproduces the SEPARATELY rounded answer | reproduces the FUSED answer |
|---|---:|---:|
| baseline (no flag) | 0 / 4096 | 4096 / 4096 |
| `--xla_allow_excess_precision=false` | 0 / 4096 | 4096 / 4096 |
| `--xla_cpu_enable_fast_math=false` | 0 / 4096 | 4096 / 4096 |
| `--xla_backend_optimization_level=0` | **4096 / 4096** | 0 / 4096 |
| `--xla_cpu_max_isa=AVX` | **4096 / 4096** | 0 / 4096 |
| `--xla_cpu_max_isa=SSE4_2` | **4096 / 4096** | 0 / 4096 |
| `--xla_cpu_max_isa=AVX2` | 0 / 4096 | 4096 / 4096 |
| `--xla_cpu_max_isa=AVX512` | 0 / 4096 | 4096 / 4096 |

**AND THE SURVEY WAS SHALLOWER THAN IT LOOKED.**  Five of the first version's
nine candidates never executed — `--fp-contract=off`, `--fp-contract=on`,
`--xla_cpu_disable_platform_dependent_math`,
`--xla_cpu_disable_new_fusion_emitters` and the two combined all abort with
*Unknown flag* — and the summary still printed a one-line verdict as though
nine flags had been tried.  A rejected arm is now REPORTED and the probe exits
non-zero, and the four flags this build refuses are named in its source rather
than left in a candidate list they cannot occupy.

On the round-37 gate's sweep rows, given NEMO's own matrix and right-hand side:

| arm | sweep `T` | sweep `S` |
|---|---:|---:|
| per-site rounding ON, no flag | 0 / 21120 | 0 / 21120 |
| per-site rounding REMOVED, no flag | 133 / 21120 | 111 / 21120 |
| per-site rounding REMOVED, `--xla_backend_optimization_level=0` | **0 / 21120** | **0 / 21120** |
| per-site rounding REMOVED, `--xla_cpu_max_isa=AVX` | **0 / 21120** | **0 / 21120** |

So EITHER flag is an equivalent cure for that row, and the ISA cap is the
better of the two — it leaves optimisation at full strength and removes only
the instruction.  **The default is NOT changed.**  Capping the ISA is a
change to every operator's code generation and to the model's cost on every
machine it runs on; it is in the ASKED table with its measured effect.

### Item 3 — the dry diagonal is an UNCONSUMED SLOT, for FINITE values only

| card | dry cells | dry ABOVE wet | finite plant moved | NaN plant non-finite | negative control moved |
|---|---:|---:|---:|---:|---:|
| GYRE-zco | 3120 | 0 | 0 | 0 | **0 — VACUOUS** |
| LOCK_EXCHANGE-zco | 5240 | 0 | 0 | 0 | **0 — VACUOUS** |
| OVERFLOW-zps | 43600 | 0 | **0** | 2800 | 204 |
| synthetic dry-above-wet column | 1 | 1 | **0** | 5 | 5 |

A dry diagonal's FINITE value cannot reach a wet cell: the off-diagonal
coupling across a dry interface is exactly `0.0`, and `0.0 * finite` is `0.0`.
It is NOT NaN-safe — `0.0 * NaN` is NaN — which is why the plant carries a NaN
arm and a synthetic column with the topology the real cards do not provide.

**AND THE GATE SAYS WHERE IT IS VACUOUS.**  On a `zco` card every dry cell is a
whole LAND COLUMN with no wet vertical neighbour, so GYRE's and LOCK's clean
plants prove nothing — their own negative control moves nothing either, and
the gate reports them as VACUOUS instead of counting them as evidence.  The
identity rests on OVERFLOW, where the control fires on 204 cells, and on the
synthetic column.

So `assembly.zwd`'s 3120-dry-cell row is registered as an UNCONSUMED-SLOT
IDENTITY rather than DEBT, on the finite-value reading, with the NaN
restriction named.

**THE DRY-CELL THICKNESS CONVENTION stays an OPEN CARD-IDENTITY GAP, and is
NOT changed here.**  legoESM's layer thickness is exactly `0.0` at every dry
cell where NEMO's `e3t_3d` is the positive reference thickness — measured this
round through the live capture as well: the model's own `dz_after` has a
dry-cell maximum of `0.0` against NEMO's `301.09862643921588`.  The BOUNDARY:
closing it is a GEOMETRY change, not the solver's, and round 37 measured that
removing the dry-diagonal substitution that compensates for it makes
OVERFLOW's kt=2 tracers non-finite.

### Item 4 — `tracer_combine`: round 37's registered finding is RETRACTED

Round 37 registered `tracer_combine` as "a lever nothing selects", measured
inert on all three cards, and left its disposition — delete or wire — as a
question.  **The premise is FALSE and is retracted here.**  The grep:

```
packages/ocean/legoesm/ocean/state.py:2164       tracer_combine: str = "concentration"
packages/ocean/legoesm/ocean/experiments/dino.py:1012    tracer_combine: str = "concentration"
packages/ocean/legoesm/ocean/experiments/dino.py:1764    "tracer_combine": "thickness_weighted",
packages/ocean/legoesm/ocean/experiments/dino.py:3983        tracer_combine=cfg.tracer_combine,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:12650  _combine = getattr(_cfg_b, "tracer_combine", "concentration")
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:13036  _combine = getattr(_cfg_b, "tracer_combine", "concentration")
```

The two readers are `_leapfrog_step` and `_nemo_mlf_step`.  The three NEMO
test-case cards run the WS-RK3 stage ladder, which reads neither — that is why
it is inert on them.  But `dino.py:1764` is inside `DINO_RECIPES`, and it
SELECTS `"thickness_weighted"`; `dino.py:3983` routes it into the config those
steps read.  So something does select it, on another card family, and neither
"delete" nor "wire" is the right disposition.  The correct row is: the field is
LIVE on the leapfrog and modified-leapfrog paths and simply not on the path
these three cards execute.  Nothing is deleted, nothing is wired, and decision
20 stays where round 37 left it.

### Item 5 — the rounds-33/34 bookkeeping, seven items

1. **The `zFw` waiver's reason string published ONE card's line numbers** as
   the provenance of a waiver resolved on three.  It cites the STATEMENTS now,
   which are portable, and labels the per-card numbers as per-card; every
   `SOURCES` value names the card it was read from, and a test refuses one
   that does not.
2. **The association rows were outside the plant signature**, so a plant that
   moved only an association row read as landing nowhere — the same defect the
   condition rows had before them.  They are in it now, scored on the same
   21120 cells as every other row.
3. **A pending plant could not land past a record the reader cannot parse.**
   The plant forces the first comparable record open; the admission reader has
   no interior-extent recovery, so a BYTE-IDENTICAL record it cannot decode
   raised before the plant landed and `run.sh` would have read the crash as the
   plant working.  Reproduced on the parent commit: `AdmissionError: array
   '...' rank -1`.  Such a record is now skipped LOUDLY and the plant goes on;
   a record that genuinely DIFFERS still raises.  Both `SELF_DESCRIBING`
   magics also gained the writer and parser rows they had never had.
4. **"NEMO's OWN pre-zdf accumulator" was not NEMO's.**  It is a seed
   RECONSTRUCTED from NEMO's dumped `Kbb` tracers, its post-`tra_ldf` tracers
   and its three `r3t` stretches.  Renamed in round 34's Part B, where it
   credited a reconstruction with the standing of an oracle dump.
5. **Six comparison reports were unstamped**, in the module every driver routes
   through, and the stamp ratchet's scope was the scripts directory only.  The
   comparison result carries `worktree_stamp()` now and the ratchet sees the
   shared fidelity package.
6. **The `nn_hls` docstring** gave GYRE a file halo of 1.  This campaign's GYRE
   is NEMO 5.0.2 with a RUNTIME `nn_hls = 2` and files that carry NO halo, so
   the value to pass is `0`; `nn_hls = 1` is the NEMO ≤ 4.0 era's and is
   labelled as such.
7. **"DISCHARGED" is now reserved for bit-exact.**  Two receipt rows said it
   beside their own tables reading `exact` FALSE — 20 of 2540 cells on LOCK and
   30 of 16900 on OVERFLOW, AT-BAR by the `1e-15` tolerance and not by Rule
   12's bar.  Both are relabelled AT-BAR-NOT-EXACT, and the reservation is
   written into Rule 12 of the skill, where it did not appear at all.

### What two independent reviews broke

codex is unavailable on this account, so both reviews are fresh Claude agents
with no shared context: one attacked the round's PREREGISTRATION before a line
of the arm existed, one attacked the DIFF afterwards.  Between them they landed
nine defects that are fixed here and four that are registered.

**THE CLAIM REVIEW, before the code.**  It broke six of the six predictions.
The capture's jit cache hazard (the step's key is the model INSTANCE) became a
call-count control.  The inertness control — "T and S bit-identical with and
without the capture" — was shown structurally blind to a capture that reads a
differently-lowered copy, and became the in-graph identity check.  A
decomposition claim was read off the wrong lines and withdrawn.  PR1's
falsifier required a row that a permanent DEBT entry makes unreachable, so
every row became wet-cells-only.  `implicit_w` was named as a matrix operand
that was off the list.  And PR3's bracket was shown to be on the wrong
quantity: perturbing EVERY wet `avt` by one ulp moves T by at most `1.16e-14`
K, `2760x` short of the residual, so the prediction became output-side.  It
also independently confirmed the index mapping and the `||M^-1||` bound.

**THE DIFF REVIEW, after the code.  It found the flag a name search cannot
find**, above, and it broke four more things that would have shipped.

1. **The `--plant` control could not fail.**  The reviewer neutered the plant
   and ran `--plant K`: exit 1, no failure line, rows byte-identical to the
   unplanted run.  The gate's own baseline is DEBT, so "exited non-zero" was
   satisfied by a plant that did nothing.  The verdict is now the planted ROW,
   scored both ways, and the plant map is one dict so a renamed row cannot
   leave a plant pointing at nothing.
2. **`run.sh` could not run**: `SOURCE_RUN` named a directory that does not
   exist, and under `set -e` it would have died at a bare test with no message.
3. **`run.sh` carried round-37 prose** saying the twin may legitimately differ
   by 115072 bytes, immediately above a gate that REFUSES any difference.
4. **One test asserted a tuple against itself** and passed on any code.

**REGISTERED, with owners.**

5. **The fold's MAGNITUDE is unattributed**, above, and it is this round's
   strongest remaining doubt.
6. **The time-level discriminator is a different invocation**: it omits ten
   arguments the model's call passes.  Two are measured inert and the zero is
   measured non-degenerate; it is labelled a property of the function.
7. **The substitution noise floor is a max-abs number** and cannot see the
   4174 bits the same substitution moves.
8. **A second operand differs and is never substituted separately**:
   `dz_after`, 5207 wet cells at `1.13687e-13`, bounded by the K arm's
   `1.42e-14`.

### kt = 1..10, before and after

**No model numerics changed this round.**  Every arm is READ-ONLY: it
captures, scores and substitutes inside a gate.  So before and after are the
same run, and reporting them as a pair would be theatre; what is reported is
the trajectory at this round's tip, against round 37's.

| card | round 37 | round 38 | under `--xla_backend_optimization_level=0` |
|---|---|---|---|
| GYRE-zco | kt=2, T S u v | kt=2, T S u v | kt=2, T S u v |
| LOCK_EXCHANGE-zco | kt=4, u | kt=4, u | kt=4, u |
| OVERFLOW-zps | kt=2, T u | kt=2, T u | kt=2, T u |

**No card's first-over-bar step moves under the flag — but the flag is NOT
inert, and that is the reason not to take it silently.**  With the per-site
rounding still in place, `--xla_backend_optimization_level=0` moves **60 of
GYRE's 70 scored trajectory rows**, the largest by `4.68248e-12` on
`GYRE-zco.kt7.before.S`, while leaving both tanks bit-identical on every scored
row they carry (0 of 20 on LOCK, 0 of 10 on OVERFLOW).  A setting that changes
the model's answers at `1e-12` over ten steps, on top of disabling backend
optimisation everywhere, is a user decision and not a harness one.

### Gates and evidence

The round-38 matrix-operand gate on the round-37 record, at the round's final
tip: `operand.e3w_now`, `operand.wet` and `operand.dt` AT-BAR at 0 bits,
`operand.dz_after` VALUE-AT-BAR, and `operand.K`/`operand.K33_fold` DEBT — the
finding, not a defect in the gate.  Its three controls: each capture point
fires exactly once, the capture is inert (0 of 21120 on each tracer), and the
in-graph identity check is 0 bits unequal on every operand.  `--plant K`,
`--plant e3w` and `--plant wet` each report `moved_its_own_row True` and exit
non-zero.

The dry-slot plant: `AT-BAR`, with GYRE and LOCK_EXCHANGE declared VACUOUS in
its own output rather than counted as evidence.

The FMA-flag probe: 8 of 8 arms ran, none rejected, three disablers.

The receipt citation gate is `PASS` over **198 citations**, none unmapped, no
failures, no map entry failing its own audit — EIGHT more than round 37's 190,
which is exactly the number this round added.  Three of them named the COMPILED
ppsrc's line numbers as though they were the shipped source's and one anchored
on a brace; the gate refused all four before this was written.

The full ocean-fidelity suite on a detached worktree
(`/tmp/codex-gyre-r38-suite`), stamped at `b82d01c4e26b`:
**`999 passed, 7 skipped, 18 deselected in 2489.84s`**, exit 0, zero failures.
ONE code commit lands after that stamp — the plant's per-row checksum,
`af1a46e9202c`, in this round's own gate — and it is covered instead by that
gate's own test file (17 passed at the final tip) and by the three plant runs,
which are themselves stamped at the tip.  Everything else after it is this
receipt.  The citation gate above was re-run at the final tip and exits 0.

**A RETRACTION ABOUT THAT RUN.**  Mid-round it was reported as anomalously
slow — apparently stalled at 63 per cent for over an hour of CPU while a
diagnosis was written about which test might be hanging.  It was not stalled:
pytest's progress output is BLOCK-BUFFERED when redirected to a file, so the
percentage a reader sees is arbitrarily stale.  The run took 41 min 29 s, in
line with round 37's 42 min 47 s.  Nothing was wrong, and the instrument being
misread was the log.

Evidence under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round38/`,
`round38_evidence.sha256` over its 19 files, itself
`a83c67531d7bff6a20945719a84bf663c4ee91e8b13feaaf0152415065547a78`.

### ASKED / UNASKED

| choice | status |
|---|---|
| legoESM's `K33` keeps its per-stage placement | ASKED, NOT TAKEN.  The measurement says NEMO computes the slopes once per step on the BEFORE state and legoESM recomputes them per stage; moving legoESM's is a scheme change on every card that runs GM/Redi, so it is named here and left for the user |
| `--xla_backend_optimization_level=0` is NOT the default | ASKED.  It is the only setting that disables the contraction and it disables backend optimisation entirely; measured effect recorded above |
| `tracer_combine` is neither deleted nor wired | ASKED.  Round 37's premise is retracted: DINO's recipes select it and the leapfrog steps read it |
| the dry-cell thickness convention is unchanged | ASKED.  Registered as an open card-identity gap with its boundary named |
| the admission gate skips an unreadable BYTE-IDENTICAL record instead of raising | a previously-tolerated condition changes — taken because the tolerated condition made a control unable to prove anything, and a record that DIFFERS still raises |
| the capture instrument lives in the GATE, not the model | a harness choice: no model edit, no new hook, and the wrapped functions are the ones the compiled step calls |
| detached probe worktree | ASKED; the same disposition rounds 32-37 recorded.  Flagged: `/tmp/codex-gyre-r38-suite` |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed |

UNASKED list: **empty**.

### Rule-11 records

* **The preregistration's PR2 is REFUTED**: it named the TKE closure; the
  closure owns exactly `0` and the isoneutral fold owns the residual.
* **Round 37's "a lever nothing selects" is RETRACTED**: `tracer_combine` is
  selected by DINO's recipes and read by two step functions.
* **The output-level round trip is RETRACTED as a control**: it moves 4174 T
  bits on values that are bit-identical, because a host array lowers as a
  constant.  The in-graph identity check replaces it.
* **A Rule-10 error, self-caught**: the first print of this card's GM/Redi
  config read a different object and said `implicit_K33 = False`.

### Merge readiness

`03c6e8d96ff7` remains an ancestor, so the integration is still a
fast-forward.  **HOLD.**

1. **GYRE's `kt2` `T`/`S`/`u`/`v` rows still fail**, unchanged.
2. The owner of the kt=1 stage-3 tracer residual is NAMED and MEASURED, and
   the fix for it is an ASKED scheme choice that has not been made.
3. `tra_zdf` is bit-exact given NEMO's inputs on every arm except the dry
   diagonal, which is now an unconsumed-slot identity for finite values.
4. ORCA2 has no card on this branch, unchanged.
5. Rounds 1-24 of this receipt remain UNAUDITED by the citation gate.

### What is open

1. **The isoneutral fold's placement**, ASKED above.  Verifying any change to
   it needs the `kt = nit000 + 1` record, whose frame spec and `run.sh` this
   round wrote and did not run.
2. **The dry-cell thickness convention**, registered with its boundary.
3. The 5207 cells at `1.13687e-13` on `dz_after` — inside the bar, reported,
   and not the owner of anything measured here.
4. OVERFLOW's mask-placement row, round 30's `dyn_ldf` row, the moved
   trajectory rows from rounds 32-34, the slow forcing's depth average, the
   causal injection arm at the stage-3 boundary: unchanged.

## Round 39 — the slopes move to the before state, and GYRE's kt=2 tracers clear

Round 39 starts from `3dceb7b8231b` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round39_preregister.json` at `24014cacfc11`,
committed BEFORE the arm existed; its AMENDMENT, written after an independent
claim review returned VERDICT NO on two blockers and before the post-change
numbers were taken, is
`manifests/nemo_testcase_l2_gyre_round39_preregister_amendment.json`.  **No
NEMO executable was run by the agent, no `makenemo` or `mpirun` was invoked,
and no NEMO source was edited.**  The operator ran the round's one
acquisition.  Four detached probe worktrees are flagged:
`/tmp/codex-gyre-r39-before` at `24014cacfc11`, `/tmp/codex-gyre-r39-fma` at
`194b4094b5b7` with the round-37 per-site rounding removed,
`/tmp/codex-gyre-r39-defeat` at `367c0930c18b` for one attack, and
`/tmp/codex-gyre-r39-suite`.

### Item 0 — the kt=2 acquisition was refused by a rule that should not exist

The operator's acquisition ran and `run.sh` exited 71 at "REFUSE: widening the
arm moved a round-37 record".  **It was a good run.**

The rule demanded RAW byte identity of every round-37 record.  NEMO's stream
dumps write whole work arrays INCLUDING the `nn_hls` halo, which NEMO neither
owns nor initialises, and the momentum-side transport record carries a `zFw`
slot the vector-invariant branch has not defined at the write point.  Those
bytes are uninitialised memory: they differ between two runs of the SAME
executable, so raw identity is not attainable and a rule that demands it
refuses runs that changed no model state.  That is the entire reason this
campaign has a consumed-field admission gate — and `run.sh` already ran it four
lines below, while the log line it printed said the differing record "falls
through to the consumed-field admission".  The prose and the code disagreed.

**The generalised admission, base round 37 vs candidate round 38, verdict
PASS, 0 violations.**  All 8 differing record kinds were PARSED, none skipped:

| record | kind | raw bytes differing | changed elements | in OWNED cells |
|---|---|---:|---:|---:|
| `oracle_rkstage3_wzv_kt00000001.bin` | `NEMO_L2_WZVOP_1` | 52 | 16 | 0 |
| `oracle_rktracer_operands_kt00000001_s2.bin` | `NEMO_L2_RKTRA_1` | 4 | 2 | 0 |
| `oracle_slow_forcing_kt00000001.bin` | `NEMO_L2_SLOW_2` | 24 | 8 | 0 |
| `oracle_tracer_transport_kt00000001_s3.bin` | `NEMO_L2_TRTRP_1` | 4 | 2 | 0 |
| `oracle_transport_kt00000001_s1.bin` | `NEMO_L1_TRANSP_1` | 6 | 2 | 2, `zFw`, WAIVED |
| `oracle_transport_kt00000001_s2.bin` | `NEMO_L1_TRANSP_1` | 4 | 2 | 0 |
| `oracle_transport_kt00000001_s3.bin` | `NEMO_L1_TRANSP_1` | 4 | 2 | 0 |
| `oracle_zdf_matrix_kt00000001.bin` | `NEMO_L2_ZDFMX_1` | — | 8 | 0 |

44 of 52 records raw identical, 46 admitted differences each printed with its
index and both values, **every one halo or the registered undefined `zFw` slot
and every value subnormal garbage of order `1e-310`** (for example
`6.927945e-310` vs `6.944252e-310`).  The `zFw` waiver resolved against the
run's own `round38_oracle_trazdf_kt2/ocean.output:798`, `ln_dynadv_vec = T`.  The kt=1 trazdf record, the
final restart and `mesh_mask.nc` are byte-identical.  With `--plant-consumed`
the same call is FAIL with `plant_applied` true, so the admission is not
vacuous here.

So the raw comparison stays a REPORT; the refusal becomes raw identity of the
ONE record every round-35/37/38 arm scores against, plus a consumed-field
admission against the round-37 run itself with its own plant.  Four tests
EXECUTE the shipped guard rather than matching its text; on the old rule the
halo-only case exits 71, on this one 0.

### Item 1 — THE SLOPES ARE BUILT ON THE BEFORE STATE, AND THE RESIDUAL FALLS 1799x

NEMO's WS-RK3 program computes the neutral slopes ONCE PER STEP, on the BEFORE
state, OUTSIDE the stage loop: `stprk3.F90:173` `CALL eos ( ts, Nbb, rhd )`
then `:174` `CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )`, compiled at
`GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2/BLD/ppsrc/nemo/stprk3.f90:178`; the three
`CALL stp_RK3_stg` are at `stprk3.F90:195`, `stprk3.F90:200` and
`stprk3.F90:207`.  `Kbb = Kmm = Nbb` inside
`ldf_slp`, so every geometry it reads -- `gdept`, `gdepw`, `e3u`, `e3v`, `e3w` and
the ssh behind them -- is at the before level too (`LDF/ldfslp.F90:80` is the
routine, and both its time-level arguments are `Nbb` at the call site).  `traldf_iso.F90:135` squares those
once-per-step slopes into `ah_wslp2` (`:296-297`) at each stage and
`trazdf.F90:173` adds it to `avt`.  **Nothing in `ah_wslp2` carries a stage
time level**: `ahtu`/`ahtv` have no time index (`traldf_iso.F90:291-294`) and
the only `Kmm` in that routine is `akz`'s `e3w`, which `ln_traldf_msc = F`
does not reach.  The claim review corrected an earlier
sentence here that said otherwise, and it independently established that
`grep` for `ldf_slp`, `zdf_mxl` and `zdf_phy` in `stprk3_stg.F90` returns ZERO
hits, so nothing recomputes the slopes between stages.

legoESM built them from `T_mid`/`S_mid` — the forward-Euler PREDICTOR of the
whole baroclinic tendency (`ocean_model_latlon_cgrid.py:5929`) — and from the
after-ssh.

**THE STAGE-STATE COMPONENT TABLE, which closes round 38's strongest remaining
doubt.**  Round 38 could account for only `1e-4` of the fold: injecting NEMO's
own stage-minus-before tracer anomaly, absolute maximum `7.25e-6` K, gave
`7.84e-17` against a live `9.66e-13`.  The reason is that legoESM's operand is
not a stage tracer at all.  Measured through the model's own call, then
re-called outside the graph with one component swapped to step entry:

| arm | K33 absolute maximum (m²/s) | share of live |
|---|---:|---:|
| live, as the model calls it | `9.6620886711883531e-13` | 1.000000 |
| CONTROL: same operands, re-called outside the graph | `9.6620886715999986e-13` | 1.000000 |
| tracers := step-entry (Kbb) T, S | `1.3071545538245700e-14` | 0.013529 |
| ssh := step-entry (Kbb) | `9.6625031189531020e-13` | 1.000043 |
| BOTH := step-entry (Kbb) | **`0.0`** | 0.000000 |

The state the call was handed differs from step entry by
**`0.3372313792464041` K on 10199 of 21120 cells**, `0.0` on S, and
`0.002848007840713329` m of ssh on 600 of 704.  **That 0.337 K is the four
orders of magnitude round 38 could not find.**  Reading, stated as the skill's
Rule 5 requires rather than as two independent shares: the TRACER time level
removes 98.65 per cent, and the remaining 1.35 per cent vanishes only when the
ssh moves too — ssh ALONE moves the fold by +0.004 per cent, so the two are
not separable and neither owns that remainder.  The control reproduces the
live value to a relative `4.3e-10`, not bit for bit, so "BOTH gives exactly
0.0" is exactly 0.0 ON THIS INSTRUMENT; the in-graph operand gate below
measures it bit-exactly.

**THE BLAST RADIUS, measured before the change was made.**  On this lane the
GM/Redi TENDENCY is causally dead: zeroing `gm_redi_tracer_tendency_latlon`'s
whole output moves GYRE's kt=1 T by **0 of 21120 bits**, while zeroing K33
moves it by `3.196021e-11` K on 7045 cells.  The diff review confirmed the
mechanism independently — `dT_gm` reaches only `T_mid`, the AB2 branch is
unreachable on a forward-Euler card, the rk3_ws tracer base is `state.T.data`,
and `T_mid` has no consumer afterwards — and closed the one escape by
instantiating the card: `gm_bolus_advection = "centred"`, so the through-FCT
bolus is not requested.

So on the WS-RK3 tracer lane the slope operand set — the tracers and the ssh
handed to the shared density/Jacobian, the GM/Redi tendency and K33 — is the
step-entry state.  **No knob, and no default that preserves the old
behaviour.**  `_ldf_state` still wins where the modified-leapfrog lane supplies
it; every other integrator has no stage loop for this statement to be about.

**Prediction against measurement:**

| row | before | predicted | after |
|---|---:|---:|---:|
| kt=1 stage-3 T residual (K) | `3.1956659540810506e-11` | `<= 2.6e-14`, and the `lego_avt_no_fold` arm's value | **`1.7763568394002505e-14`** |
| the `lego_avt_no_fold` arm, on the PRE-change model | — | — | `1.7763568394002505e-14` |
| kt=1 stage-3 S residual (psu) | `8.1712414612411521e-13` | the same arm's value | **`3.5527136788005009e-14`** |
| the arm's S, on the PRE-change model | — | — | `3.5527136788005009e-14` |
| `operand.K33_fold`, wet faces unequal | 17400 / 17400 at `9.66209e-13` | 0, EXACT | **0 / 17400 at `0`, AT-BAR** |
| `operand.K`, wet faces unequal | 17383 / 17400 at `9.66209e-13` | `<= 1.8e-18` | 4905 / 17400 at **`1.73472e-18`** |

**PR1, PR2, PR3 and PR4 are CONFIRMED, and PR1/PR4 to every digit.**  The
`lego_avt_no_fold` arm — added because the claim review measured that round
38's substitution ceiling was the wrong bound for a change that removes only
the fold — predicted the post-change residual exactly, on both tracers, from
the pre-change model.

### Item 1, Rule 12 per card

| card | does the changed statement execute? | how established | result |
|---|---|---|---|
| GYRE-zco | YES | `gm_redi` is not None, `tracer_time_integrator = "rk3_ws"`, `implicit_K33 = True`, printed from `_cfg_b.gm_redi` | given NEMO's inputs at kt=1: `K33` vs `ah_wslp2` **0 / 17400 unequal, absolute maximum 0, EXACT** — both are identically zero because GYRE's before state is horizontally uniform, measured on both sides |
| GYRE-zco, kt=2 | YES | the round-38 record, `ah_wslp2` absolute maximum `3.3898494597440722e-08` | given NEMO's OWN before state: **DEBT**, 17400 / 17400 wet faces, max `3.3884e-08`, max relative `2.63`; legoESM's fold reaches `1.1728064666279615e-10`, 289x smaller.  Owner: the isoneutral SLOPE TRANSCRIPTION, which this round did not touch |
| LOCK_EXCHANGE-zco | NO | legoESM resolves `gm_redi = None`; NEMO resolves `ln_traldf_OFF = T`, `ln_traldf_iso = F` (`lock_kt1_10/ocean.output:578`, `lock_kt1_10/ocean.output:584`), so `l_ldfslp = F` and `ldf_slp` is never called | kt=1..10 **BIT-IDENTICAL** before and after: the residual artifact's SHA-256 is the same on both arms, all 20 scored rows unmoved |
| OVERFLOW-zps | NO | same, `overflow_kt1_10/ocean.output:690`, `overflow_kt1_10/ocean.output:696` | kt=1..10 **BIT-IDENTICAL**, same residual SHA-256, all 10 scored rows unmoved |
| `build_nemo_gyre_recipe` (`fidelity/nemo_recipe.py:988`) | YES | the diff review's scope sweep: `gm_redi` on with `kappa_GM = 600`, `rk3_ws` on both integrators | **UNMEASURED WITH SPEC** — it has no trajectory gate.  Spec: the same operand gate driven off a record this configuration does not have.  Its committed tests are in the suite below |
| ORCA2 | UNKNOWN | no card on this branch | **UNMEASURED WITH SPEC**, unchanged |
| DINO / every other integrator | NO | `_ldf_state` wins, and no other lane sets `tracer_time_integrator = "rk3_ws"` | untouched by construction; a test asserts the guard keeps both conditions |

**"DISCHARGED" is not used of GYRE's kt=1 row here even though it is bit-exact
given NEMO's inputs, because the CARD is not: its kt=2 given-inputs row is
DEBT and its trajectory still fails at kt=2 on `u` and `v`.**

### Item 1, kt = 1..10 before and after

BEFORE was taken at `24014cacfc11` in a detached probe worktree, same gates,
same oracle roots, same `--max-step 10`, so the two arms differ only in the
model.

| card | first over bar BEFORE | first over bar AFTER |
|---|---|---|
| GYRE-zco | kt=2, **T S u v** | kt=2, **u v** |
| LOCK_EXCHANGE-zco | kt=4, u | kt=4, u |
| OVERFLOW-zps | kt=2, T u | kt=2, T u |

**No card's first-over-bar step moves earlier, and GYRE loses two of its four
failing fields.**  At kt=2:

| row | before | after |
|---|---:|---:|
| `GYRE-zco.kt2.before.T` | `1.3614736849003888e-12` DEBT | **`7.567947108951577e-16` AT-BAR** |
| `GYRE-zco.kt2.before.S` | `2.2181101297999213e-14` DEBT | **`9.643957086086614e-16` AT-BAR** |
| `GYRE-zco.kt2.before.u` | `2.7478406377547115e-12` DEBT | unchanged |
| `GYRE-zco.kt2.before.v` | `3.305560306813421e-12` DEBT | unchanged |
| `GYRE-zco.kt2.before.ssh` | `4.336808689942018e-19` AT-BAR | unchanged |

Both cleared rows are **AT-BAR-NOT-EXACT**: their own `exact` field is false,
so the word DISCHARGED does not apply to them.

**Every moved row is registered.**  42 of GYRE's 50 scored rows moved.  Two are
the improvements above.  **39 moved only in their last digits — the largest
relative movement over all of them is `-1.69e-07`, on `kt10.before.ssh` — and
every one of them is still DEBT**; they are downstream of the kt=2 `u`/`v`
divergence this change does not touch.  One row is LARGER, `kt7.before.ssh` by
`+1.18e-07` relative, inside that same band and still DEBT.  Eight rows did not
move.  Both tanks: 0 rows moved.

### Item 1 — the instrument, and one control that DEGRADED

The operand gate's three controls after the change: each capture point fires
exactly once; the in-graph identity check is **0 bits unequal on every
operand**; and the capture-inertness control **went from 0 to 1 moved T cell**.

That is reported, not hidden.  Its magnitude is now printed beside its count,
because a bit count without one is not a reportable number: the move is
`3.55271e-15` at cell `[3, 21, 2]`, **exactly 1 ulp**.  A debug callback gives
an operand a second consumer, which is precisely what stops XLA contracting a
product into a multiply-add (round 37), so one ulp is the expected size; the
gate's status is `PERTURBED` and it exits 1.  The operand rows above are
unaffected — the in-graph check proves the captured arrays ARE the arrays that
run's solve consumed — but the model the capture run integrates differs from
the production one by that ulp, and that is now a printed row rather than a
silent zero.

`--plant K`, `--plant e3w` and `--plant wet` each report
`moved_its_own_row True` and exit non-zero.

### Item 2 — what remains, and who owns it

At kt=1 stage 3, and at kt=2:

| quantity | before | after | disposition |
|---|---:|---:|---|
| kt=1 stage-3 T (K) | `3.1956659540810506e-11` | `1.7763568394002505e-14` | the closure's own `avt`, `3.55271e-15` of it measured by the gate's CLOSURE-ONLY arm |
| kt=1 stage-3 S (psu) | `8.1712414612411521e-13` | `3.5527136788005009e-14` | same |
| kt=2 whole-step `u` | `2.7478406377547115e-12` | unchanged | round 30's `dyn_ldf` row, unchanged |
| kt=2 whole-step `v` | `3.305560306813421e-12` | unchanged | same |
| `operand.dz_after` | 5207 / 18000 at `1.13687e-13` | unchanged | named below |

**THE `dz_after` OWNER, with the given-inputs discipline.**
`e3t(i,j,k,Kaa) = e3t_0(i,j,k)·(1 + r3t(i,j,Kaa))`
(`domzgr_substitute.h90:139`) and `r3t = ssh/ht_0` (`domqco.F90:160`,
`dom_qco_r3c`), so the residual is owned by the reference thickness or by the
stretch and by nothing else.  **The record's own consistency was checked
first, not assumed**: NEMO's `e3t_Kaa` equals `e3t_0·(1+r3t_Kaa)` at **0 cells
unequal** on every level this campaign scores (it differs only at `jk = jpk`,
which is outside the scored box).  The record that carries NEMO's stage-3
stretch is `r3t_Kaa` in the trazdf record.  Measured:

| row | wet cells unequal | max abs | status |
|---|---:|---:|---|
| `dz_owner.reference_thickness` (legoESM vs NEMO `e3t_0`) | **0 / 18000** | **0** | AT-BAR |
| `dz_owner.r3t` (legoESM's `ssh(Kaa)/H` vs NEMO's `r3t_Kaa`) | 566 / 600 | `2.11758e-22` | VALUE-AT-BAR |
| `dz_owner.dz_after` | 5207 / 18000 | `1.13687e-13` | VALUE-AT-BAR |

**The reference thickness is bit-shared, so the owner is legoESM's own
`ssh(Kaa)` out of the stage-3 update, and the FIRST NON-BIT STATEMENT is `r3t`
itself** — 566 of 600 wet columns, `2.11758e-22`, which is `3.2e-16` relative
on an `r3t` whose own maximum is `6.62e-07`.  `h_partial` remains the
registered dry-cell convention gap: it differs from `e3t_0` only at the 3120
DRY cells, by up to `300.71` m.

### Item 3 — the FMA flag, measured for decision 22.  THE DEFAULT IS NOT CHANGED

Two settings were compared: the current tip (round 37's per-site rounding in
place, no flag) and a scratch worktree with that rounding REMOVED under
`XLA_FLAGS=--xla_cpu_max_isa=AVX`.

| measurement | tip, no flag | ISA cap + per-site rounding removed |
|---|---|---|
| per-operator `trazdf` gate, rows moved by the flag | **0 of 24** | 2: `sweep.T` 133→0, `sweep.S` 111→0 |
| per-operator `zdf` gate, rows moved by the flag | **0 of 10** | 2: `zdf_solve_lego.u` 582→0, `zdf_solve_lego.v` 623→0 |
| distinct FMA SITES silenced | — | **2** — the second and third Thomas recurrences of the ONE shared ordered solve, reached by 2 operators |
| GYRE first over bar | kt=2, u v | kt=2, u v — **42 of 50 rows moved**, largest `-7.33816e-13` on `kt4.before.v`, NO status change |
| LOCK_EXCHANGE first over bar | kt=4, u | kt=4, u — **1 of 20 rows moved, by `-8.47033e-22`** |
| OVERFLOW first over bar | kt=2, T u | kt=2, T u — **0 of 10 rows moved** |
| wall time, one 10-step LOCK trajectory, n=1 each | 48.41 s | 47.12 s |

**RECOMMENDATION: do NOT take the flag.**  It silences exactly the two sites
round 37's per-site rounding already covers — no operator's given-inputs result
changes that the rounding does not already fix — while capping the instruction
set for every operator in the model on every machine it runs on; and round 38
measured that a global disabler moves 60 of GYRE's 70 scored trajectory rows by
up to `4.68e-12`, so other FMA sites exist that neither approach has
identified.  Cost is not the argument: one trajectory took 48.41 s without the
flag and 47.12 s with it, a single run each, i.e. no measurable cost.  The
argument is that it is a global change of code generation bought for two sites
that are already fixed locally.

### What two independent reviews broke

codex is unavailable on this account, so both reviews are fresh Claude agents
with no shared context: one attacked the PREREGISTRATION before a line of the
arm existed, one attacked the DIFF afterwards.  Both returned NO.

**THE CLAIM REVIEW, before the code.**  Two blockers and ten findings, all
dispositioned in the amendment.  It measured that round 38's substitution
ceiling was the WRONG BOUND for a change that removes only the fold — one ulp
on every wet `avt` is worth up to `1.16e-14` K on its own — so PR1's falsifier
would have fired on a change that is exactly correct; the `lego_avt_no_fold`
arm exists because of that finding and it predicted the answer to every digit.
It found the `slope_prd_geometry_stage` knob the preregistration never
mentioned, which is now MEASURED redundant (below) rather than argued.  It
corrected a wrong sentence about `ahtu`/`ahtv` carrying a stage time level, it
showed PR3's falsifier band was unfalsifiable, and it verified PR5 on both
sides independently.

**THE DIFF REVIEW, after the code.  Its blocker made the round's central claim
untested**, and it is the most important thing either review did.  The two
model tests wrapped only the K33 routine, so restoring the predictor state at
the OTHER TWO call sites — the shared density/Jacobian and the GM/Redi tendency
— left every test green.  Reproduced in a detached worktree with exactly that
edit: **2 of 8 tests now fail**, at `jac.T` 10199 of 21120 cells and at a fold
of `9.6931758357204761e-13` m²/s.  That reverted state is the disagreement this
file's own comment calls a CONFIRMED P1 defect.

It also found **one operand still on the after-ssh**: `native_bolus_slope_eta`,
while the comment above it claimed the whole slope operand set had moved.  It
is moved, and measured inert on GYRE.  It found the new source-admission plant
**could not fail** — the gate exits non-zero when a plant NEVER LANDS, so
"exited non-zero" was satisfied by a control that proved nothing; both plants
in that script now require `"plant_applied": true`.  And it found the
substitution arm injected a **negative diffusivity** at masked interfaces,
because the model builds `K_v_cell = (K_v + K33)·mask` and a plain `K - K33` is
`-K33` there.

**REGISTERED, with owners.**

1. **`run.sh`'s new block has never executed.**  The round-38 records come from
   the run the old rule refused; the only evidence the new rule works is this
   round's own read-only re-runs of the same gate.
2. **`round38_source_admission.json` is not in that script's final `sha256sum`
   manifest**, and its plant JSON is written with `2>&1`, so it is not
   parseable JSON.  Pre-existing pattern.
3. **`build_nemo_gyre_recipe` is a fourth configuration whose numbers move**,
   with no trajectory gate.  Its committed tests are in the suite below.
4. **The GM/Redi TENDENCY leg of this change is unmeasurable on all three
   cards**, because its output is discarded on this lane.  Boundary: it becomes
   measurable on the first card that pairs `rk3_ws` with `through_fct`.
5. **The capture-inertness control degraded to 1 ulp**, above.

### Item 1's other answered question — the knob is REDUNDANT, measured

With the fix in place, the card built with
`slope_prd_geometry_stage="before_step"` and the card built with the default
`"current_step"` give **BIT-IDENTICAL T and S, 0 of 21120 on each**.  PR9
CONFIRMED: the two mechanisms are redundant on this lane, not competing.  The
knob keeps its meaning on the lanes the fix does not touch, and it is NOT the
ship vehicle — a knob whose default preserves the old behaviour is exactly what
this campaign forbids.

### Rule-11 records

* **Round 38's caveat is CLOSED, and its framing was too weak.**  It said the
  stage TRACER anomaly explained only `1e-4` of the fold and left the
  responsible component UNMEASURED.  The component is the tracer time level
  after all; what round 38 lacked was that legoESM's operand is not a stage
  tracer but a forward-Euler PREDICTOR `0.337` K from step entry, where NEMO's
  own stage anomaly is `7.25e-6` K.
* **The kt=1 record's inability to discriminate is now demonstrated, not
  argued.**  legoESM's fold at kt=1 is exactly 0 and so is NEMO's; at kt=2,
  given NEMO's own before state, they differ by 289x.  A round that had only
  the kt=1 record could have called the transcription matched.
* **An earlier sentence in the model's own comment claimed `ahtu`/`ahtv` carry
  the stage's `Kmm`.**  They carry no time index at all; corrected in the code
  and here.
* **The capture-inertness control is no longer 0.**  Reported with its
  magnitude, 1 ulp, rather than left as a bare count.

### ASKED / UNASKED

| choice | status |
|---|---|
| the isoneutral slopes / K33 are built ONCE PER STEP on the BEFORE state on the WS-RK3 lane | ASKED-BY-DIRECTIVE — the user's standing "do as NEMO does".  It makes legoESM read the state `stprk3.F90:174` reads.  No knob, and no default that preserves the old behaviour |
| `native_bolus_slope_eta` moves with the rest of the slope operand set | same directive: NEMO's ONE `ldf_slp` call per step feeds the bolus and the Redi tensor the same before-state slopes.  Measured inert on GYRE |
| `slope_prd_geometry_stage` keeps its default and its meaning | ASKED — it is MEASURED redundant with the fix on this lane (0 of 21120 bits), and it is not the ship vehicle |
| the acquisition's admission becomes consumed-field identity plus raw identity of ONE record | a previously-tolerated condition changes, in the loosening direction — taken because the tolerated condition REFUSED A GOOD RUN, and because the instrument that replaces it is strictly stronger per element.  Named here rather than left silent |
| both plants in that script now require `"plant_applied": true` | a previously-tolerated condition becomes a hard error — taken because the diff review demonstrated the old form passes a control that proved nothing |
| the XLA ISA cap is NOT taken | ASKED — decision 22, measured above, default unchanged, recommendation given |
| the dry-cell thickness convention is unchanged | ASKED — registered open card-identity gap, unchanged from round 38 |
| `tracer_combine` is neither deleted nor wired | ASKED — unchanged from round 38 |
| detached probe worktrees | ASKED; the same disposition rounds 32-38 recorded.  Flagged: `/tmp/codex-gyre-r39-before`, `/tmp/codex-gyre-r39-fma`, `/tmp/codex-gyre-r39-defeat`, `/tmp/codex-gyre-r39-suite` |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed by the agent.  The operator ran the acquisition |

UNASKED list: **empty**.

### Merge readiness

`03c6e8d96ff7` remains an ancestor, so the integration is still a
fast-forward.  **HOLD.**

1. **GYRE's kt=2 `u` and `v` still fail**, unchanged and untouched by this
   round; `T` and `S` are now AT-BAR-NOT-EXACT.
2. **The kt=2 isoneutral fold, given NEMO's own inputs, is DEBT at 289x.**  Its
   owner is the slope transcription, and it is the first thing the next round
   should walk.
3. `tra_zdf`'s matrix diffusivity is now bit-exact given NEMO's inputs at kt=1;
   `operand.K` is `1.73e-18` and belongs to the closure.
4. The `dz_after` owner is named — legoESM's own `ssh(Kaa)` — and its first
   non-bit statement is `r3t`.
5. ORCA2 has no card on this branch, unchanged; `build_nemo_gyre_recipe` joins
   it as UNMEASURED-WITH-SPEC.
6. Rounds 1-24 of this receipt remain UNAUDITED by the citation gate.

### What is open

1. **The isoneutral SLOPE TRANSCRIPTION.**  Given NEMO's own before state at
   kt=2, legoESM's fold is `1.17e-10` where NEMO's `ah_wslp2` is `3.39e-08`.
   The argmax sits at the first interior face, which is where NEMO's
   mixed-layer slope ramp acts, so the MLD/ramp and the `N²` denominator are
   the first two suspects — neither is measured.
2. **The `dz_after` residual**, owned by the stage-3 `ssh(Kaa)` update, first
   non-bit at `r3t`, 566 of 600 wet columns at `2.11758e-22`.
3. **The GM/Redi tendency leg is unmeasurable on all three cards** and becomes
   measurable on the first `rk3_ws` + `through_fct` card.
4. The dry-cell thickness convention, `tracer_combine`, OVERFLOW's
   mask-placement row, round 30's `dyn_ldf` row, the moved trajectory rows from
   rounds 32-34, the slow forcing's depth average, the causal injection arm at
   the stage-3 boundary: unchanged.
5. `run.sh`'s new admission block has never executed on a real acquisition.

### Gates and evidence

The round-38 matrix-operand gate on the round-37 record, at the round's final
tip: `operand.K33_fold`, `operand.e3w_now`, `operand.wet` and `operand.dt`
AT-BAR at 0 bits; `operand.K` VALUE-AT-BAR at `1.73472e-18`;
`operand.dz_after` VALUE-AT-BAR at `1.13687e-13`.  Its controls: each capture
point fires exactly once, the in-graph identity check is 0 bits unequal on
every operand, and the capture-inertness control is 1 T cell at `3.55271e-15`,
1 ulp — so the gate's status is `PERTURBED` and it exits 1, and that is stated
here rather than left for a reader to discover.  `--plant K`, `--plant e3w`
and `--plant wet` each report `moved_its_own_row True` and exit non-zero.

The `dz_owner` arm: the record is self-consistent at 0 cells,
`reference_thickness` AT-BAR at 0 of 18000, `r3t` VALUE-AT-BAR at
`2.11758e-22`.  The `knob-redundancy` arm: `REDUNDANT True`, 0 of 21120 bits
on each tracer.  The `kt2-given-inputs` arm: record discriminating,
`KT2-GIVEN-INPUTS DEBT unequal 17400/17400 max 3.3884e-08 rel 2.62947`, and it
exits 1.

The consumed-field admission on (round 37, round 38): `PASS`, 0 violations, 44
of 52 raw identical, 46 admitted differences all halo or the registered
undefined slot; with `--plant-consumed` it is `FAIL` with `plant_applied` true.

The receipt citation gate is `PASS` over **211 citations**, 0 failures, 0
unmapped, 0 map entries failing their own audit, and all 9 of its self-test
plants fired.  Thirteen citations were added this round.  **Five map entries
had to be RE-ANCHORED**, because this round's before-state block added 57 lines
to `ocean_model_latlon_cgrid.py` and five keys were line numbers below it; the
statements are unchanged and the numbers were not, which is exactly what that
gate exists to catch.  The stamp-scope ratchet is 18 passed.

The full ocean-fidelity suite at the round's FINAL tip, `365c15b9013f`, on a
detached worktree (`/tmp/codex-gyre-r39-suite2`):
**`997 passed, 7 skipped, 18 deselected in 2461.27s (0:41:01)`**, exit 0, zero
failures.  An earlier run at `ba3e5f1ff271` — the last CODE commit, before the
citation map was re-anchored — had **3 failures, all three in the citation
gate's own test file and all three this round's stale line numbers**; they are
fixed at the source in the receipt commit and that file is 16 passed at the
final tip.  Nothing else changed between the two runs.

Evidence under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round39/`,
`round39_evidence.sha256` over its 30 files, itself
`c99c874c7451abee9a83efdec2d1cfbbf3d197ca278a8f6f197f85abb503498b`.

## Round 40 — the stage-3 owner is not the state, the stretch is discharged, and the mixed-layer index moves the fold 232x

Round 40 starts from `c37c05ee8951` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round40_preregister.json` at `9da65cf73f9a`,
committed BEFORE any arm of this round existed and BEFORE the item-3 edit.
**No NEMO executable was run by the agent, no `makenemo` or `mpirun` was
invoked, and no NEMO source was edited.**  Two detached probe worktrees are
flagged: `/tmp/codex-gyre-r40-nonvac` at `9da65cf73f9a` for the non-vacuity
check, and `/tmp/codex-gyre-r40-before` at `c37c05ee8951` for the BEFORE arms.

### Item 1 — THE STAGE-3 MOMENTUM RHS: the operator table, and where the residual is NOT

**THE TABLE, in NEMO's own call order**, with the deck's resolved switches
read from its own log and the compiled branch read from the ppsrc this
configuration built.  Every row is the statement, its time levels, and
legoESM's corresponding statement.

| # | NEMO statement | operands / time level | legoESM |
|---|---|---|---|
| 1 | `stprk3_stg.F90:266-268` `zub = un_adv*r1_hu(Kmm) - uu_b(Kmm)` | `n_baro_upd = np_HYB` (`stprk3_stg.F90:44`); the compiled form divides by `1+r3u(Kmm)` rather than multiplying a stored reciprocal | `_mom_pert_ws`'s `transport_velocity`, `ocean_model_latlon_cgrid.py:6471-6475` |
| 2 | `stprk3_stg.F90:273-274` `zFu = e2u*e3u(Kmm)*(uu(Kmm)+zub*umask)` | `e3u(Kmm) = e3u_3d*(1+r3u(Kmm)*umask)` | `_nemo_ws_stage_transport`, `ocean_model_latlon_cgrid.py:8156` |
| 3 | `stprk3_stg.F90:290` `CALL wzv(... uu(Kmm), vv(Kmm), ww, np_velocity)` | velocity form, because `ln_dynadv_vec = T` (`round38_oracle_trazdf_kt2/ocean.output:798`) | `literal_wzv` inside the same helper |
| 4 | `stprk3_stg.F90:322` `CALL eos( ts, Kmm, rhd, rhop )` | stage tracers | `_stage_hpg_operands`, `ocean_model_latlon_cgrid.py:8175` |
| 5 | `stprk3_stg.F90:324` `dyn_hpg` -> `hpg_sco` | `ln_hpg_sco = T` (`round38_oracle_trazdf_kt2/ocean.output:834`); under `key_RK3` it **OVERWRITES** `Krhs` (`dynhpg.F90:359-363`, `dynhpg.F90:383-387`), so nothing before it survives | the shared tendency accumulator's `hpg` bucket |
| 6 | `stprk3_stg.F90:327` `dyn_vor` -> `vor_ene` | `ln_dynvor_ene = T` (`round38_oracle_trazdf_kt2/ocean.output:810`), `nn_e3f_typ = 0` (`round38_oracle_trazdf_kt2/ocean.output:813`) | `pv_flux_ene` with `nemo_qco_live_vorticity_e3f_cgrid` |
| 7 | `stprk3_stg.F90:331` `dyn_adv` -> `dyn_keg` + `dyn_zad` | `nn_dynkeg = 0` (`round38_oracle_trazdf_kt2/ocean.output:799`); `dynkeg.F90:120-121` and `dynzad.F90:104-107`, both reading `e3u(Kmm)` | the same accumulator's `advection` bucket |
| 8 | `stprk3_stg.F90:400` `dyn_ldf( kstp, Kbb, Kmm, ...)` | BEFORE velocity, fixed in round 30 | `ldf_state=(..., u0*u_mask_3d, ...)` |
| 9 | `stprk3_stg.F90:430` `dyn_zdf` | the implicit solve, bit-exact given NEMO's inputs since round 33 | `rk3_stage_velocity_update` + the ordered solve |

**`dyn_spg` IS NOT IN THE STAGE LOOP.**  `ln_dynspg_ts = T`
(`round38_oracle_trazdf_kt2/ocean.output:846`), but the only `dyn_spg_ts`
call is `stp2d.F90:281`, which runs once per step BEFORE the three stages.
A stage-3 momentum walk that looked for it would be looking in the wrong
routine.

**AND THE `nn_e3f_typ` BRANCH IS DEAD CODE ON THIS BUILD.**  The
configuration's own keys are `key_qco key_vco_1d3d key_RK3`
(`cpp_GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2.fcm:1`), so `vor_ene` takes the
`key_qco` arm and divides the potential vorticity by
`e3f_0vor*(1+r3f*fe3mask)` —
`GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2/BLD/ppsrc/nemo/dynvor.f90:556` — while the
`nn_e3f_typ` SELECT at `dynvor.F90:493-516` is never compiled.  `e3f_0vor`
is frozen ONCE at init from the REFERENCE thicknesses, and `r3f` is the only
live part; it is also the one quantity in the whole stage program whose
FORMULA differs between stages, `r2_3*r3fb + r1_3*r3fa` at stage 2
(`stprk3_stg.F90:202`) against `r1_2*(r3fb + r3fa)` at stage 3
(`stprk3_stg.F90:234`).

**THE RESIDUAL, RE-MEASURED AT THIS TIP.**  The frame `dyn_ldf` receives:

| row | wet cells unequal | max abs |
|---|---:|---:|
| `kt1.stage3.pre_ldf_rhs.u` | 17383 / 17400 | `2.0614443638749651e-16` |
| `kt1.stage3.pre_ldf_rhs.v` | 17097 / 17100 | `2.4827278704178648e-16` |

Round 30 recorded `17382` and `2.0614443633579772e-16`; the last digits moved
with rounds 32 and 39 and the boundary did not.

**PR3 IS CONFIRMED: THE RESIDUAL IS NOT INHERITED.**  The state stage 3 is
handed, scored against NEMO's own `oracle_stage_kt00000001_s2.bin` through the
model's own path:

| operand | what NEMO's stage 3 reads it as | wet cells unequal | max abs | status |
|---|---|---:|---:|---|
| `u` | `uu(Kmm)` for `dyn_vor`/`dyn_adv` | 7453 / 17400 | `1.0842021724855044e-19` | AT-BAR |
| `v` | `vv(Kmm)` | 7287 / 17100 | `1.0842021724855044e-19` | AT-BAR |
| `T` | `ts(Kmm)` for the `eos` at `stprk3_stg.F90:322` | **2 / 18000** | `3.5527136788005009e-15` | AT-BAR |
| `S` | same | **0 / 18000** | **0** | AT-BAR |
| `ssh` | `ssh(Kmm)` behind `e3t/e3u/e3v(Kmm)` | **0 / 600** | **0** | AT-BAR |

So the operators are handed a state good to `1.08e-19` on the velocity and
bit-exact on salinity and ssh, and they produce a right-hand side that is
`2.06e-16` off on 17383 of 17400 cells.  **The 2.06e-16 is GENERATED at stage
3**, inside the operator arithmetic or the geometry derived from that state,
and it is not carried in.  The two T cells are one ulp of a ~16 K field and
are two cells against 17383; they are reported rather than waved away, and
they cannot produce a difference that is everywhere.

**THE PER-OPERATOR SPLIT, AND WHY IT IS A SCALE AND NOT A SCORE.**  legoESM's
own stage-3 buckets, exposed WRITE-only through the seam this round extended
to stage 3:

| bucket | legoESM absolute max, u | the relative error it would need |
|---|---:|---:|
| `vorticity` | `2.631168e-08` | `7.83e-09` |
| `hpg` | `1.382735e-11` | `1.49e-05` |
| `advection` | `1.790666e-13` | `1.15e-03` |

**NO OPERATOR IS NAMED THE OWNER, and the preregistration forbids it.**  NEMO
has no stage-3 split to score these against: `oracle_rkstage2_terms` is
header-locked to `kstg = 2`
(`nemo_testcase_l2_gyre_phase3_gate.py:323`).  What the table buys is a
RANKING by the relative error each operator would have to carry, and even that
ranking is WEAK IN ONE DIRECTION: `hpg` and `advection` are both differences
of much larger intermediates — `hpg_sco` accumulates `zhpi` down the column
(`dynhpg.F90:368-375`) and `dyn_zad` forms `e1e2t*ww` products of order `1e4`
before scaling (`dynzad.F90:89-96`) — so either could carry a large ABSOLUTE
error at a small relative error on its own intermediates.  The ranking is
reported as what it is.

**PR4 IS REFUTED AS STATED.**  It predicted the three buckets would sum to the
exposed total BIT-EXACTLY.  Measured: 7648 of 17400 cells on u and 6868 of
17100 on v differ, at `6.6174449004242214e-24`.  The cause is that the gate
sums three host arrays left to right where the model accumulates them inside
one fused graph — a different association, not a different value.  That
residual is `3.2e-08` of the `2.06e-16` the buckets must account for, so they
remain admissible as the SCALES they are used as, and the prediction's
wording, not its purpose, was wrong.

**THE RECORD THAT WOULD SCORE THEM IS SPECIFIED AND COMMITTED, NOT RUN.**
`nemo_testcase_l2_gyre_round40_stage3_terms/` carries the frame spec, the
Fortran delta and `run.sh` for `NEMO_L2_RKTS3_1` — the stage-3 sibling of the
stage-2 terms record, plus `r3f`, `e3f_0vor`, the live `e3f_vor(Kmm)`,
`fe3mask`, `rhd` and `ww`, none of which any existing record carries.  Its
delta removes zero lines of NEMO's own source and its `run.sh` refuses the run
if the patched result removes any.  The agent did not run it.

### Item 2 — THE SLOPE TRANSCRIPTION: the mixed-layer index moves the fold 232x

**WHAT NEMO DOES.**  `ldf_slp` (`LDF/ldfslp.F90:80`) reads a mixed-layer index
`nmln` and depth `hmlp` that `zdf_mxl` produced on the before state, and uses
them twice: `zhmlpt = gdept(nmln-1,Kmm)*ssmask` (`LDF/ldfslp.F90:143`) sets the
u/v ramp's length scale, and `r1_hmlw = 1/MAX(hmlp - gdepw(mikt,Kmm), 10.)`
(`LDF/ldfslp.F90:161`) sets the w one.  `zdf_mxl`'s criterion is an
N-SQUARED INTEGRAL, not a density difference: it converts the density
criterion once, `zN2_c = grav*rho_c*r1_rho0` (`ZDF/zdfmxl.F90:95`) with
`rho_c = 0.01` (`ZDF/zdfmxl.F90:34`), integrates `MAX(rn2b,0)*e3w(Kmm)` down
the column (`ZDF/zdfmxl.F90:98`), takes `nmln = MIN(jk,mbkt)+1` while the
integral is under threshold (`ZDF/zdfmxl.F90:99`), and sets
`hmlp = gdepw(nmln,Kmm)*ssmask` (`ZDF/zdfmxl.F90:104`) — a W LEVEL, live
geometry, with a bottom cap and a positive-only clamp.

legoESM's GYRE card resolves `mld_criterion = "rho_c"`, a
potential-density-DIFFERENCE test referenced to the nearest CELL CENTRE at or
below 10 m, with no `MAX(N2,0)` clamp, no `mbkt` cap and an unstretched
`hmlp`.  NEMO's own criterion already exists in the same file behind
`mld_criterion = "n2_integral"`, wired for one DINO card and for nothing else.

**THE ARM, ONE VARIABLE, ON NEMO'S OWN BEFORE STATE AT kt=2.**  Only that
field is swapped; grid, coordinate, masks, diffusivities, timestep and the
before state are the card's and NEMO's respectively.

| mixed-layer criterion | wet faces unequal | max abs | max relative | legoESM's fold, absolute max |
|---|---:|---:|---:|---:|
| `rho_c`, the card's default | 17400 / 17400 | `3.3884e-08` | `2.62947` | `1.17281e-10` |
| `n2_integral`, NEMO's own | 17400 / 17400 | **`1.45837e-10`** | **`0.0516838`** | **`3.40443e-08`** |

NEMO's own `ah_wslp2` has absolute maximum `3.38985e-08`.  So the card's
default puts legoESM's isoneutral fold **289x too SMALL**, and NEMO's own
criterion puts it within **0.43 per cent** of NEMO's, cutting the absolute
residual by **232x** and the relative by **51x**.  The preregistered falsifier
was a move of less than 10 per cent; the measured move is 98.0 per cent.
**PR6 IS CONFIRMED.**

**IT IS NOT LANDED, AND THAT IS AN ASKED QUESTION, NOT AN OVERSIGHT.**
Selecting a different criterion on a card is a scheme selection, the row is
still DEBT at 5.2 per cent relative, and Rule 12 cannot discharge it without a
record that scores a STATEMENT of `ldf_slp`.  **PR7 IS CONFIRMED**: the
round-38 kt=2 record carries none of `prd`, `rn2b`, `nmln`, `hmlp`, `uslp`,
`vslp`, `wslpi`, `wslpj`, `ahtu`, `ahtv`, `e3u_3d`, `e3v_3d`, `umask`,
`vmask`, `wmask` or `ssmask` — it can score the finished `ah_wslp2` and
nothing else.  `nemo_testcase_l2_gyre_round40_ldfslp/` specifies
`NEMO_L2_LDFSL_1`, which captures every one of them plus the 33 per-`jk`
intermediates the descending loop destroys, and it too removes zero lines of
NEMO's source.  The agent did not run it.

**FIVE MORE OF THIS CARD'S SLOPE DEFAULTS SELECT THE NON-NEMO ARM**, each with
a `nemo_*` sibling wired for DINO only: `slope_n2` (NEMO reads `rn2b`;
legoESM recomputes), `slope_prd_evaluation`, `slope_metric_evaluation` (NEMO
multiplies a stored `r1_e1u`, `LDF/ldfslp.F90:206`; legoESM divides),
`slope_face_thickness_evaluation` (NEMO's limiter reads the LIVE `e3u(Kmm)`,
`LDF/ldfslp.F90:212-213`) and `slope_depth_evaluation`.  They are registered
here, unmeasured, and ranked below the mixed-layer index because none of them
can move a whole level of `nmln`.  One LATENT difference is recorded and is
inert on this card: NEMO's w-point bound uses a HARDCODED `100.`
(`LDF/ldfslp.F90:281-282`) where its u/v bound uses `1/rn_slpmax`, and legoESM
uses `1/S_max` at both; with `rn_slpmax = 1.0e-2`
(`round38_oracle_trazdf_kt2/ocean.output:658`) the two are the same number
here and only here.

### Item 3 — THE STRETCH IS DISCHARGED, BIT-EXACT, IN THE ONE SHARED HELPER

**WHAT NEMO DOES, and it is one statement.**  `dom_qco_r3c_RK3` forms the
RATIO first and adds one: `r3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)`
(`domqco.F90:189-209`), where `r1_ht_0 = ssmask/(ht_0 + 1 - ssmask)`
(`domain.F90:158`) is exactly `1/ht_0` on a wet column, and the thickness
macro is `e3t(i,j,k,t) = e3t_0(i,j,k)*(1 + r3t(i,j,t))`
(`domzgr_substitute.h90:139`).  Stage 3 recovers the stage-1 value:
`stprk3_stg.F90:156` computes `r3ta` from `ssha`, and `stprk3_stg.F90:231`
assigns it to `r3t(:,:,Kaa)`.  NEMO never forms `(ssh + ht_0)/ht_0`.

legoESM's shared `compute_ocean_jacobian` (`vertical.py:1914`) formed the SUM
first, so the low bits of the small ratio were lost to cancellation.

**THE INSTRUMENT WAS VALIDATED BEFORE THE CLAIM.**  On host arrays, from
NEMO's own dumped `r3t_Kaa` and `e3t_Kaa` and its own `ssha`:
`ssha * r1_ht_0` reproduces `r3t_Kaa` at **0 of 600** wet columns and
`e3t_0*(1+r3t_Kaa)` reproduces `e3t_Kaa` at **0 of 18000** wet cells, both at
absolute maximum 0.  Only then was legoESM's form scored against the same
arrays: `(ssha+ht_0)/ht_0` differs on 221 of 600 columns at `2.220446e-16`,
and `e3t_0` times it differs on **5207 of 18000** cells at
**`1.136868e-13`** — reproducing round 38 and 39's `operand.dz_after` row
exactly, from host arrays alone.  A pure divide, `ssha/ht_0`, differs on 168
of 600 at `1.058791e-22`, so the divide and the sum are two separate defects
and only the sum is large.

**AFTER, through the model's own path:**

| row | before | after |
|---|---|---|
| `dz_owner.reference_thickness` | 0 / 18000, AT-BAR | 0 / 18000, **EXACT** |
| `dz_owner.stretch_given_nemo_ssh` | (row did not exist) | **0 / 600, EXACT** |
| `dz_owner.stretch_model_path` | (row did not exist) | **0 / 600, EXACT** |
| `dz_owner.dz_after` | 5207 / 18000 at `1.136868e-13` | **0 / 18000, EXACT** |

**PR1 IS CONFIRMED on `dz_after` and REFUTED as worded on `r3t`.**  The old
`dz_owner.r3t` row computed `eta/H` INSIDE THE GATE — a division the model
does not perform — and compared it against NEMO's `r3t`, so it scored the
gate's own arithmetic on top of legoESM's ssh and could never have gone to
zero however the model changed.  That is a Rule-10 defect in round 39's own
instrument, and it is fixed here rather than explained: two rows replace it,
one feeding NEMO's own `ssh(Kaa)` through the shared helper and one feeding
legoESM's.  Both are now bit-exact, which also says that legoESM's own
`ssh(Kaa)` agrees with NEMO's to better than one ulp of the stretch.

**THE CHANGE IS ONE STATEMENT IN ONE SHARED HELPER, no knob and no default
that preserves the old behaviour.**  legoESM's own `min_water_column_m` floor
— NEMO has none — moves onto the Jacobian instead of the water column, which
leaves every clipped cell at exactly `min_col/H` and every unclipped cell
bit-identical to the unclipped call.  The z-star branch and the `key_linssh`
branch are untouched.

### Item 3, Rule 12 per card

| card | does the changed statement execute? | how established | result |
|---|---|---|---|
| GYRE-zco | YES | its coordinate is the partial-cell one and every live thickness is `dz_ref * J` | **bit-exact GIVEN NEMO'S OWN ssh: `stretch_given_nemo_ssh` 0 / 600 and `dz_after` 0 / 18000, absolute maximum 0** |
| LOCK_EXCHANGE-zco | YES | same coordinate, same helper; the trajectory MOVED, which is itself the proof it executes | kt=1..10 measured: 21 of 50 scored rows moved, **first-over-bar unchanged at kt=4 on u**, 0 status changes |
| OVERFLOW-zps | YES | same, and this card has a real staircase so `H_bathy` varies | kt=1..10 measured; see the table below |
| ORCA2 | UNKNOWN | no card on this branch | **UNMEASURED WITH SPEC**, unchanged from rounds 38-39 |
| `build_nemo_gyre_recipe` and every other z-star / partial-cell card | YES | 52 call sites share this helper | their answers move in the last bits; no trajectory gate exists for them, so UNMEASURED WITH SPEC |

**The word DISCHARGED is used of the STATEMENT, not of the CARD.**  GYRE's
kt=2 `u` and `v` still fail and its kt=2 isoneutral fold is still DEBT.

### Item 3, kt = 1..10 before and after

BEFORE was taken at `c37c05ee8951` in a detached probe worktree, same gates,
same oracle roots, same `--max-step 10`, so the two arms differ only in the
model.

## Round 40 — the stage-3 owner is `dyn_adv`, the mixed-layer index is the fold's, and the stretch is discharged

Round 40 starts from `c37c05ee8951` on a clean tree, same regime: CPU
production JIT, fp64/x64, `transcendentals="libm"`, oracle V2.  The
preregistration is
`manifests/nemo_testcase_l2_gyre_round40_preregister.json` at `9da65cf73f9a`,
committed BEFORE any arm of this round existed and BEFORE the one numerics
change it makes.  **The agent ran no NEMO executable, invoked no `makenemo`
and no `mpirun`, and edited no NEMO source.**  It wrote two WRITE-only
instruments and sent the operator one command each; **the OPERATOR ran both**,
and both records are scored below.  Three detached probe worktrees are
flagged: `/tmp/codex-gyre-r40-nonvac` at `9da65cf73f9a`,
`/tmp/codex-gyre-r40-before` at `c37c05ee8951`, and `/tmp/codex-gyre-r40-suite`.

### Item 1 — THE STAGE-3 MOMENTUM RHS: the table, and the operator that owns it

**THE TABLE, in NEMO's own call order**, with the deck's resolved switches read
from its own log and the compiled branch read from the ppsrc this
configuration built.

| # | NEMO statement | operands / time level | legoESM |
|---|---|---|---|
| 1 | `stprk3_stg.F90:266-268` `zub = un_adv*r1_hu(Kmm) - uu_b(Kmm)` | `n_baro_upd = np_HYB` (`stprk3_stg.F90:44`) | `ocean_model_latlon_cgrid.py:6471-6475` |
| 2 | `stprk3_stg.F90:273-274` `zFu = e2u*e3u(Kmm)*(uu(Kmm)+zub*umask)` | `e3u(Kmm) = e3u_0*(1+r3u(Kmm)*umask)` | `ocean_model_latlon_cgrid.py:8156` |
| 3 | `stprk3_stg.F90:290` `wzv(... uu(Kmm), vv(Kmm), ww, np_velocity)` | velocity form: `ln_dynadv_vec = T` (`round38_oracle_trazdf_kt2/ocean.output:798`) | the same helper's `literal_wzv` |
| 4 | `stprk3_stg.F90:322` `eos( ts, Kmm, rhd, rhop )` | stage tracers | `ocean_model_latlon_cgrid.py:8175` |
| 5 | `stprk3_stg.F90:324` `dyn_hpg` -> `hpg_sco` | `ln_hpg_sco = T` (`round38_oracle_trazdf_kt2/ocean.output:834`); under `key_RK3` it OVERWRITES `Krhs` (`dynhpg.F90:359-363`, `dynhpg.F90:383-387`) | the shared accumulator's `hpg` bucket |
| 6 | `stprk3_stg.F90:327` `dyn_vor` -> `vor_ene` | `ln_dynvor_ene = T` (`round38_oracle_trazdf_kt2/ocean.output:810`), `nn_e3f_typ = 0` (`round38_oracle_trazdf_kt2/ocean.output:813`) | `pv_flux_ene` with the literal `e3f_vor` |
| 7 | `stprk3_stg.F90:331` `dyn_adv` -> `dyn_keg` + `dyn_zad` | `nn_dynkeg = 0` (`round38_oracle_trazdf_kt2/ocean.output:799`); `dynkeg.F90:120-121` and `dynzad.F90:104-107`, both on `e3u(Kmm)` | the `advection` bucket |
| 8 | `stprk3_stg.F90:400` `dyn_ldf( kstp, Kbb, Kmm, ...)` | BEFORE velocity, fixed in round 30 | `ldf_state=(..., u0*u_mask_3d, ...)` |
| 9 | `stprk3_stg.F90:430` `dyn_zdf` | bit-exact given NEMO's inputs since round 33 | the ordered solve |

**`dyn_spg` IS NOT IN THE STAGE LOOP.**  `ln_dynspg_ts = T`
(`round38_oracle_trazdf_kt2/ocean.output:846`), but the only `dyn_spg_ts` call
is `stp2d.F90:281`, once per step, before the stages.  A stage-3 momentum walk
that looked for it would be reading the wrong routine.

**AND THE `nn_e3f_typ` SELECT INSIDE `vor_ene` IS NOT COMPILED.**  This
configuration's keys are `key_qco key_vco_1d3d key_RK3`
(`cpp_GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2.fcm:1`), so `vor_ene` takes the
`key_qco` arm and divides the potential vorticity by
`e3f_0vor*(1+r3f*fe3mask)` —
`GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2/BLD/ppsrc/nemo/dynvor.f90:556` — while the
SELECT at `dynvor.F90:493-516` is dead.  A SECOND `nn_e3f_typ` SELECT is very
much alive at initialisation and builds `e3f_0vor` itself, so the switch is
live geometry even though that one branch is not; an independent review caught
the first version of this sentence overstating it.

**THE RESIDUAL, RE-MEASURED AT THIS TIP.**  The frame `dyn_ldf` receives:
`kt1.stage3.pre_ldf_rhs.u` 17383 / 17400 at `2.0614443638749651e-16`, and
`.v` 17097 / 17100 at `2.4827278704178648e-16`.  Round 30 recorded `17382` and
`2.0614443633579772e-16`; the last digits moved with rounds 32 and 39 and the
boundary did not.  **That absolute figure is `7.83e-09` RELATIVE to the field's
own maximum of `2.63e-08`**, and it is quoted that way from here on: the
campaign's shared scorer normalises by `max(|reference|, 1)`, so AT-BAR is
VACUOUS for any field whose maximum is below one, and an independent review
was right to call that out.

**PR3 IS CONFIRMED: THE RESIDUAL IS NOT INHERITED.**  The state stage 3 is
handed, scored against NEMO's own `oracle_stage_kt00000001_s2.bin` through the
model's own path:

| operand | what stage 3 reads it as | cells unequal | max abs | relative |
|---|---|---:|---:|---:|
| `u` | `uu(Kmm)` for `dyn_vor`/`dyn_adv` | 7453 / 17400 | `1.0842021724855044e-19` | `2.55e-16` |
| `v` | `vv(Kmm)` | 7287 / 17100 | `1.0842021724855044e-19` | `2.55e-16` |
| `T` | `ts(Kmm)` for the `eos` at `stprk3_stg.F90:322` | **2 / 18000** | `3.5527136788005009e-15` | `1.78e-16` |
| `S` | same | **0 / 18000** | **0** | 0 |
| `ssh` | `ssh(Kmm)` behind `e3t/e3u/e3v(Kmm)` | **0 / 600** | **0** | 0 |

**THE SPLIT, AGAINST NEMO'S OWN FRAMES.**  The operator ran
`nemo_testcase_l2_gyre_round40_stage3_terms/run.sh`; its record admission and
its source admission both report `PASS` with zero violations.  The record's
OWN closure control is checked before anything is read off it: its last frame
reproduces the pre-`dyn_ldf` record every round-30..40 arm already scores at
**0 cells unequal, absolute maximum 0**, on both faces.  Two instruments, one
quantity.

| operator | NEMO frame | cells unequal | max abs | RELATIVE to the operator | status |
|---|---|---:|---:|---:|---|
| `dyn_hpg` | `after_hpg` | **0 / 17400** | **0** | **0** | **EXACT** |
| `dyn_vor` | `after_vor - after_hpg` | 9186 / 17400 | `9.926167350636332e-24` | `3.77e-16` | AT-BAR |
| `dyn_adv` | `after_adv - after_vor` | **17400 / 17400** | **`2.0614443630503727e-16`** | **`1.15e-03`** | **DEBT** |

The `v` face says the same: `dyn_hpg` 0 / 17100, `dyn_vor` 9214 / 17100 at
`9.926167350636332e-24`, `dyn_adv` 17100 / 17100 at
`2.4827278704178648e-16`, relative `9.88e-04`.

**THE FIRST DIFFERING OPERATOR IS `dyn_adv`, AND IT OWNS THE WHOLE
RESIDUAL.**  Its own contribution's maximum is `1.79e-13` on `u`, so a
`2.06e-16` difference is **0.115 per cent of the operator**, on every one of
17400 cells — not roundoff by four orders of magnitude.  `dyn_hpg` is
bit-exact and `dyn_vor`'s `9.926167350636332e-24` is, to every digit, the
figure round 25 measured for the stage-2 vorticity arm: the same operator,
the same residual, two stages apart.

**AND THE VORTICITY DIVISOR IS EXONERATED, MEASURED.**  Given NEMO's own
`ssh(Kmm)`, legoESM's literal `e3f_vor` reproduces NEMO's own dumped
`e3f_vor(Kmm)` at **0 of 16530 wet f-faces, absolute maximum 0**, against a
field whose maximum is `300.71` m.  That was this round's leading suspect
before the record existed — the divisor is the one quantity in the stage
program whose FORMULA differs between stages (`stprk3_stg.F90:202` against
`stprk3_stg.F90:234`) — and it is now refuted rather than argued about.  PR5
forbade naming an owner from magnitudes, and the magnitudes would have named
the wrong one.

**WHAT IS NOT SEPARATED, AND WHY.**  `dyn_keg` and `dyn_zad` are ONE NEMO call
(`stprk3_stg.F90:331`), so this record cannot split them and neither can a
finer one without a second delta inside `dyn_adv`.  The discriminator is named
rather than guessed: `dyn_keg` reads only `uu/vv(Kmm)`, which the table above
shows exact to `1.08e-19`, while `dyn_zad` reads `ww` — which THIS record
carries and legoESM does not expose at stage 3.  One WRITE-only seam scores
it, and that is the next round's first move.

### Item 2 — THE FOLD'S OWNER IS THE MIXED-LAYER INDEX, scored at the statement

**WHAT NEMO DOES.**  `ldf_slp` reads a mixed-layer index and depth twice:
`zhmlpt = gdept(nmln-1,Kmm)*ssmask` (`LDF/ldfslp.F90:143`) sets the u/v ramp's
length scale and `r1_hmlw = 1/MAX(hmlp - gdepw(mikt,Kmm), 10.)`
(`LDF/ldfslp.F90:161`) sets the w one.  Both come from `zdf_mxl`, whose
criterion is an N-SQUARED INTEGRAL and not a density difference: it converts
the criterion once, `zN2_c = grav*rho_c*r1_rho0` (`ZDF/zdfmxl.F90:95`) with
`rho_c = 0.01` (`ZDF/zdfmxl.F90:34`), integrates `MAX(rn2b,0)*e3w(Kmm)` down
the column (`ZDF/zdfmxl.F90:98`), takes `nmln = MIN(jk,mbkt)+1` while the
integral is under threshold (`ZDF/zdfmxl.F90:99`) and sets
`hmlp = gdepw(nmln,Kmm)*ssmask` (`ZDF/zdfmxl.F90:104`) — a W LEVEL, live
geometry, with a bottom cap and a positive-only clamp.  legoESM's GYRE card
resolves `mld_criterion = "rho_c"`: a potential-density DIFFERENCE referenced
to the nearest cell CENTRE at or below 10 m, with no clamp, no cap and an
unstretched depth.

**THE STATEMENT-LEVEL SCORE, on NEMO's own before state at kt = 2**, out of
`oracle_ldfslp_kt00000002.bin` — the second acquisition the operator ran.  The
arm refuses the record unless NEMO's own `uslp` is non-zero, so a record that
could not discriminate cannot read as agreement.

| statement | criterion | wet columns unequal | max abs |
|---|---|---:|---:|
| `nmln` (`ZDF/zdfmxl.F90:99`) | `rho_c`, the card's default | **290 / 600** | **1 whole level** |
| `nmln` | `n2_integral`, NEMO's own | **0 / 600** | **0** |
| `hmlp` (`ZDF/zdfmxl.F90:104`) | `rho_c` | 600 / 600 | `10.2647` m |
| `hmlp` | `n2_integral` | 600 / 600 | `1.3422e-05` m |

**NEMO's mixed-layer LEVEL is reproduced EXACTLY by the criterion legoESM
already has and this card does not select, and missed by a whole level on
almost half the domain by the one it does.**  The depth follows it 765000x
closer.  This is the owner named where it lives, not inferred from the product
it moves.

**AND THE PRODUCT MOVES WITH IT.**  Round 39's `kt2-given-inputs` fold arm,
re-run with only that field swapped on a copy of the resolved config:

| criterion | wet faces unequal | max abs | max relative | legoESM's fold, absolute max |
|---|---:|---:|---:|---:|
| `rho_c`, the card's default | 17400 / 17400 | `3.3884e-08` | `2.62947` | `1.17281e-10` |
| `n2_integral`, NEMO's own | 17400 / 17400 | **`1.45837e-10`** | **`0.0516838`** | **`3.40443e-08`** |

NEMO's own `ah_wslp2` has absolute maximum `3.38985e-08`.  The card's default
puts legoESM's isoneutral fold **289x too SMALL**; NEMO's own criterion puts it
within **0.43 per cent**, cutting the absolute residual **232x** and the
relative **51x**.  The preregistered falsifier was a move under 10 per cent;
the measured move is 98.0 per cent.  **PR6 IS CONFIRMED**, and the arm exits
non-zero on its own refutation now rather than returning 0 whatever it
measures.

**PR7 IS CONFIRMED**: the round-38 kt=2 record carries none of `prd`, `rn2b`,
`nmln`, `hmlp`, `uslp`, `vslp`, `wslpi`, `wslpj`, `ahtu`, `ahtv`, `e3u_3d`,
`e3v_3d`, `umask`, `vmask`, `wmask` or `ssmask`.  It could score the finished
`ah_wslp2` and nothing else, which is why round 39 could report a 289x fold
with no owner inside the routine.

**THE CRITERION IS NOT CHANGED HERE.**  Selecting a different one on a card is
a scheme selection; it goes in the ASKED table with the numbers above and a
recommendation, and Rule 12 cannot discharge it while `hmlp` still carries
`1.34e-05` m under NEMO's own criterion — a residual now registered with its
own likely owner, the live `(1+r3t)` stretch on `gdepw`.

**FIVE MORE OF THIS CARD'S SLOPE DEFAULTS SELECT THE NON-NEMO ARM**, each with
a `nemo_*` sibling wired for one DINO card: `slope_n2` (NEMO reads `rn2b`;
legoESM recomputes), `slope_prd_evaluation`, `slope_metric_evaluation` (NEMO
multiplies a stored `r1_e1u`, `LDF/ldfslp.F90:206`; legoESM divides),
`slope_face_thickness_evaluation` (NEMO's limiter reads the LIVE `e3u(Kmm)`,
`LDF/ldfslp.F90:212-213`) and `slope_depth_evaluation`.  Registered, unmeasured,
and ranked below the index because none of them moves a whole level.  One
LATENT difference is inert here and recorded anyway: NEMO's w-point bound uses
a hardcoded `100.` (`LDF/ldfslp.F90:281-282`) where its u/v bound uses
`1/rn_slpmax`, and legoESM uses `1/S_max` at both; with
`rn_slpmax = 1.0e-2` (`round38_oracle_trazdf_kt2/ocean.output:658`) the two
are the same number on this card and only on this card.

### Item 3 — THE STRETCH IS DISCHARGED, BIT-EXACT, IN THE ONE SHARED HELPER

**WHAT NEMO DOES, and it is one statement.**  `dom_qco_r3c_RK3` forms the RATIO
first and adds one: `domqco.F90:189-209`, where
`r1_ht_0 = ssmask/(ht_0 + 1 - ssmask)` (`domain.F90:158`) is exactly `1/ht_0`
on a wet column, and the thickness macro is
`e3t = e3t_0*(1 + r3t)` (`domzgr_substitute.h90:139`).  Stage 3 recovers the
stage-1 value at `stprk3_stg.F90:231`.  NEMO never forms `(ssh + ht_0)/ht_0`,
and it never DIVIDES by `ht_0` — it multiplies a reciprocal built once.

**THE INSTRUMENT WAS VALIDATED BEFORE THE CLAIM.**  On host arrays, from
NEMO's own dumped `r3t_Kaa` and `e3t_Kaa` and its own `ssha`:
`ssha * r1_ht_0` reproduces `r3t_Kaa` at **0 of 600** wet columns and
`e3t_0*(1+r3t_Kaa)` reproduces `e3t_Kaa` at **0 of 18000** wet cells.  Only
then was legoESM's form scored against the same arrays: `(ssha+ht_0)/ht_0`
differs on 221 of 600 columns at `2.220446e-16` and its `e3t` on **5207 of
18000** cells at **`1.136868e-13`** — reproducing rounds 38 and 39's
`operand.dz_after` row exactly, from host arrays alone.

**AFTER, through the model's own path:** `dz_owner.reference_thickness`
0 / 18000, `dz_owner.stretch_given_nemo_ssh` **0 / 600**,
`dz_owner.stretch_model_path` **0 / 600** and `dz_owner.dz_after`
**0 / 18000**, every one at absolute maximum 0.

**PR1 IS CONFIRMED on `dz_after` and REFUTED AS WORDED on `r3t`.**  The old
`dz_owner.r3t` row computed `eta/H` INSIDE THE GATE — a division the model does
not perform — and compared it to NEMO's `r3t`, so it scored the gate's own
arithmetic on top of legoESM's ssh and could never have reached zero however
the model changed.  That is a Rule-10 defect in round 39's own instrument, and
it is replaced rather than explained: one row feeds NEMO's own `ssh(Kaa)`
through the shared helper, one feeds legoESM's, and the arm's verdict sentence
is now READ OFF those rows instead of asserted beside them.

**THE CHANGE IS ONE STATEMENT IN ONE SHARED HELPER**, no knob and no default
that preserves the old behaviour.  `compute_ocean_jacobian` (`vertical.py:1914`)
forms `1 + eta*r1_h` with the reciprocal taken in the PROMOTED dtype of ssh and
bathymetry — an existing test caught the first version taking it in the
bathymetry's storage dtype, at a relative `3.0e-10` that is f32 eps times the
stretch and not roundoff, which is Rule 1c exactly.  legoESM's own
`min_water_column_m` floor, which NEMO does not have, moves onto the Jacobian
rather than the column, leaving every clipped cell at its pre-round-40 value.
The z* branch and the `key_linssh` branch are untouched, and the unreachable
`key_linssh` arms inside the partial-cell branch are deleted rather than left
implying geometry that cannot occur.

**AND A SECOND COPY OF THE SAME STATEMENT WAS FOUND AND REMOVED.**
`compute_layer_thickness` still formed `(eta+H)/H` inline, so for part of this
round the model carried TWO different roundings of one NEMO statement — and an
independent review instrumented a GYRE step and found that branch called
FIFTEEN times, including from the momentum right-hand side.  It routes through
the shared helper now, and a test pins that it does.

### Item 3, Rule 12 per card

| card | does the changed statement execute? | how established | result |
|---|---|---|---|
| GYRE-zco | YES | every live thickness on the partial-cell coordinate is `dz_ref * J` | **bit-exact GIVEN NEMO'S OWN ssh: `stretch_given_nemo_ssh` 0 / 600 and `dz_after` 0 / 18000, absolute maximum 0** |
| LOCK_EXCHANGE-zco | YES | same coordinate and helper; the trajectory MOVED, which is the proof it executes | kt=1..10 measured: 21 of 50 rows moved, 0 status changes, **first-over-bar unchanged at kt=4 on u** |
| OVERFLOW-zps | YES | same, and this card carries a real staircase so `H_bathy` varies per column | kt=1..10 measured: 20 of 50 rows moved, **ONE status change**, first-over-bar unchanged at kt=2 on T and u |
| ORCA2 | UNKNOWN | no card on this branch | **UNMEASURED WITH SPEC**, unchanged from rounds 38-39 |
| every other partial-cell card | YES | 52 call sites share this helper | last-bit movement; no trajectory gate exists for them, so UNMEASURED WITH SPEC |

**The word DISCHARGED is used of the STATEMENT, not of the CARD.**  GYRE's
kt=2 `u` and `v` still fail, and its kt=2 isoneutral fold is still DEBT.

**THE ONE WORSENED ROW, REGISTERED.**  `OVERFLOW-zps.kt10.before.S` goes from
`8.120488e-16` AT-BAR to `1.015061e-15` DEBT — one ulp of a salinity near 35,
eight steps after that card's own first-over-bar at kt=2, and it crosses the
bar rather than moving within it.  The fix stays; the row enters OVERFLOW's
register as debt naming its boundary, which is the stretch's last bit and not
a physics change.  LOCK_EXCHANGE's largest move is
`kt10.before.u` `1.137147e-11` -> `1.137243e-11`, still DEBT on both sides.

### kt = 1..10, before and after

BEFORE was taken at `c37c05ee8951` in a detached probe worktree, same gates,
same oracle roots, same `--max-step 10`, so the two arms differ only in the
model.

| card | first over bar BEFORE | first over bar AFTER | rows moved | status changes |
|---|---|---|---:|---:|
| GYRE-zco | kt=2, **u v** | kt=2, **u v** | 39 / 50 | 0 |
| LOCK_EXCHANGE-zco | kt=4, u | kt=4, u | 21 / 50 | 0 |
| OVERFLOW-zps | kt=2, T u | kt=2, T u | 20 / 50 | **1** |

**No card's first-over-bar step moves earlier and no card's failing fields
change.**  GYRE's kt=2 `T` and `S` stay AT-BAR-NOT-EXACT and both IMPROVE —
`T` from `7.567947e-16` to `6.054358e-16` and `S` from `9.643957e-16` to
`5.786374e-16` — while `u` and `v` stay DEBT at `2.747840e-12` and
`3.305560e-12`.  This round did not touch the operator that owns those two; it
NAMED it.

Every moved row is registered.  GYRE's 39 are last-digit movement on rows that
are already DEBT downstream of the kt=2 `u`/`v` divergence, the largest
relative being `-1.28e-07` on `kt3.before.ssh`.  LOCK_EXCHANGE's largest is
`kt10.before.u`, `1.137147e-11` to `1.137243e-11`, DEBT on both sides.
OVERFLOW's one status change is the worsened row named above.

### What two independent reviews broke

codex is unavailable on this account, so both reviews are fresh Claude agents
with no shared context: one attacked the round's CLAIMS, one attacked the
DIFF.  **Both returned NO**, and between them they landed nine defects that
are fixed here and four that are registered.  Their most valuable finding was
one neither was asked for: that the operator had already run the acquisitions,
so two questions this round was going to leave as record specs could be
answered instead.

**THE CLAIM REVIEW.**  It found that the campaign's shared scorer normalises
by `max(|reference|, 1)`, so every AT-BAR verdict on a sub-unit field is
vacuous — the stage-3 residual reads AT-BAR at `2.06e-16` on a field whose own
maximum is `2.63e-08`.  Every operator row in this round carries its RELATIVE
figure now and the per-operator gate classifies on it, which is what turned
`dyn_adv` from AT-BAR into DEBT.  It found that the landed stretch DIVIDES
where NEMO multiplies a stored reciprocal, and measured the two differing on
56656 of 200000 GYRE-magnitude values.  It found the amplification argument in
PR3 asserted and not measured — superseded by the record, which scores the
operators directly instead of reasoning about what the inputs could do.  And
it corrected the "dead code" sentence about `nn_e3f_typ`.

**THE DIFF REVIEW.**  It instrumented a GYRE step and found
`compute_layer_thickness` still forming `(eta+H)/H` on the partial-cell
branch, called FIFTEEN times including from the momentum right-hand side —
two roundings of one statement, and the unfixed one feeding the very
right-hand side item 1 was walking.  It showed the `split` mode's closure
could never be bit-exact and its plant could not fail against a baseline that
was already red; that mode is DELETED, superseded by NEMO's own frames.  It
found the mixed-layer arm returning 0 whatever it measured despite carrying an
explicit falsifier, `run_dz_owner` asserting its verdict in a constant string
beside rows that might contradict it, and the new gate shipping without the
companion test every prior round's gate has.  It verified, independently, the
dtype promotion, the broadcasting, `jax.grad` at `H_bathy == 0`, eager/JIT
parity, a 2000-point sweep across the clip boundary, and the non-vacuity of
the primary new test against the parent commit.

**REGISTERED, with owners.**

1. **`worktree_stamp()` reads HEAD when the report is WRITTEN, not when the
   process started.**  A long arm that straddles a commit is stamped with a
   commit whose code it predates — observed this round: a `terms` run reported
   under `605048f4ff51` while executing the code of its parent, and the tell
   was a key the newer code adds being absent from the report.  Every number
   quoted here was re-run to completion after its last code commit and checked
   for that key.  The stamp itself is unchanged; the defect is named.
2. **`dyn_keg` and `dyn_zad` are not separated**, above, with the discriminator
   named: `ww`, which the record carries and legoESM does not expose.
3. **`hmlp` is `1.34e-05` m off under NEMO's own criterion**, with the live
   `(1+r3t)` stretch on `gdepw` as its likely and unmeasured owner.
4. **Five slope defaults and one latent limiter difference**, above.
5. **`build_nemo_gyre_recipe` and every other partial-cell card** move in their
   last bits with no trajectory gate, unchanged from round 39.

### ASKED / UNASKED

| choice | status |
|---|---|
| the quasi-Eulerian stretch forms NEMO's ratio and multiplies its reciprocal | ASKED-BY-DIRECTIVE — the user's standing "do as NEMO does".  One statement, one shared helper, no knob and no default that preserves the old behaviour |
| `compute_layer_thickness` routes through that helper instead of repeating it | same directive: it is the SAME NEMO statement, and two roundings of one statement is not a choice anyone made |
| **the GYRE card's mixed-layer criterion is NOT changed** | **ASKED, AND THIS IS THE ROUND'S ONE OPEN DECISION.**  NEMO's own N-squared-integral criterion reproduces NEMO's `nmln` at 0 of 600 columns where the card's default misses 290 by a whole level, and it cuts the isoneutral fold's residual 232x.  Default stays at `rho_c` (289x too small), or moves to `n2_integral` (0.43 per cent)?  My recommendation: MOVE IT — on the card whose purpose is to be NEMO, and paired with the `hmlp` stretch so Rule 12 can then discharge it |
| the per-operator gate classifies on the RELATIVE bar as well as the normalised one | a previously-tolerated condition becomes a hard error — taken because the normalised bar is vacuous for a sub-unit field and would have passed an operator carrying 0.115 per cent |
| the `split` mode is deleted rather than repaired | its closure could not be bit-exact and its plant could not fail; NEMO's own frames supersede it entirely |
| an unknown momentum-operator name is refused at CONSTRUCTION | a previously-tolerated condition becomes a hard error — the inner guard raised only once the stage was reached |
| the XLA ISA cap is NOT taken | ASKED — decision 22, unchanged, recommendation on file is not to take it |
| the dry-cell thickness convention, `tracer_combine` | ASKED — unchanged from rounds 38-39 |
| detached probe worktrees | ASKED; the disposition rounds 32-39 recorded.  Flagged: `/tmp/codex-gyre-r40-nonvac`, `/tmp/codex-gyre-r40-before`, `/tmp/codex-gyre-r40-suite` |
| shipped NEMO edit, `makenemo`, `mpirun`, push, merge, deletion | forbidden; none performed by the agent.  The OPERATOR ran both acquisitions, on the agent's written request |

UNASKED list: **empty**.

### Rule-11 records

* **This round's own leading suspect is REFUTED.**  Before the record existed,
  the vorticity divisor `e3f_vor` was the ranked candidate — it is the one
  quantity in the stage program whose formula differs between stages.  Given
  NEMO's own ssh it is bit-exact, 0 of 16530.  The magnitudes would have named
  the wrong operator, which is why the preregistration forbade naming one from
  them.
* **PR1 is refuted as worded on `r3t`**, above: the row it named was measuring
  the gate's own division.
* **PR4 is refuted and its mode deleted**: three host buckets summed in the
  gate cannot reproduce a fused accumulation bit for bit, and its plant could
  not fail.
* **The first version of item 3's fix was incomplete and its second was
  imprecise**: one copy of the statement remained, and the reciprocal was taken
  in the wrong dtype.  Both were found by review and by an existing test, not
  by the author.
* **Round 38's "the `nn_e3f_typ` branch is dead code" is narrowed**: the SELECT
  inside `vor_ene` is dead; the one that builds `e3f_0vor` at initialisation is
  live.

### Merge readiness

`03c6e8d96ff7` remains an ancestor, so the integration is still a
fast-forward.  **HOLD.**

1. **GYRE's kt=2 `u` and `v` still fail**, and their owner is now NAMED:
   `dyn_adv` at 0.115 per cent of its own magnitude, with `dyn_hpg` bit-exact
   and `dyn_vor` at roundoff.
2. **The kt=2 isoneutral fold's owner is NAMED at the statement**: NEMO's
   mixed-layer level, reproduced exactly by a criterion this card does not
   select.  Selecting it is the ASKED decision above.
3. The quasi-Eulerian stretch is bit-exact given NEMO's own ssh, in one shared
   helper, on every card that executes it and is measurable.
4. ORCA2 has no card on this branch, unchanged.
5. Rounds 1-24 of this receipt remain UNAUDITED by the citation gate.

### What is open

1. **`dyn_keg` versus `dyn_zad`**, and the `ww` seam that separates them.
2. **The mixed-layer criterion**, the ASKED decision, and the `hmlp` residual
   that has to move with it.
3. The five slope defaults, the latent w-point limiter constant, the dry-cell
   thickness convention, `tracer_combine`, OVERFLOW's mask-placement row, round
   30's `dyn_ldf` row, the moved trajectory rows from rounds 32-34 and this
   one, the slow forcing's depth average: unchanged.
4. ORCA2, and every partial-cell card without a trajectory gate.

### Gates and evidence

The stage-3 operator gate at this round's tip: the record's own closure
against the pre-dyn_ldf frame is 0 cells unequal on both faces; dyn_hpg is
EXACT at 0 of 17400 and 0 of 17100; dyn_vor is AT-BAR at 9.926167350636332e-24
on both; dyn_adv is DEBT at 2.0614443630503727e-16 and 2.4827278704178648e-16,
relative 1.15e-03 and 9.88e-04, and the gate exits non-zero naming advection as
the first operator over the bar.  The e3f_vor operand row is 0 of 16530 at
absolute maximum 0.  The stage-3 input arm is AT-BAR on all five operands.

The slope arm: NEMO's nmln is reproduced at 0 of 600 under the n2_integral
criterion and missed on 290 of 600 under the card's rho_c default; hmlp is
1.3422e-05 m and 10.2647 m respectively.  The arm refuses a record whose own
uslp is identically zero.  The mixed-layer fold arm exits non-zero on its own
refutation and did not fire: the measured move is 98.0 per cent against a
10 per cent falsifier.

The dz_owner arm: reference_thickness 0 of 18000, stretch_given_nemo_ssh 0 of
600, stretch_model_path 0 of 600, dz_after 0 of 18000, every one at absolute
maximum 0, and its verdict sentence is derived from those rows rather than
asserted beside them.

The ldf_slp acquisition's own gates, re-run from a clean tree at this tip
because the operator's run hit the fail-closed stamp on a dirty worktree: the
round-29 matrix regression is AT-BAR at 0 of 21120 on every row, its plant
turns the solve rows red and exits non-zero, and round40_outputs.sha256 is
written over 65 files.  The stage3_terms acquisition's own admission and
source admission both report PASS with zero violations, and its header verdict
is PASS.

The receipt citation gate is PASS over 251 citations, 0 unmapped, 0 failures,
0 map entries failing their own audit, and all 9 self-test plants fired.

The full ocean-fidelity suite at the round's FINAL tip, on a detached
worktree (/tmp/codex-gyre-r40-suite), same selection as round 39's:
**1011 passed, 7 skipped, 18 deselected in 2504.45s (0:41:44)**, exit 0, zero
failures.  Round 39's was 997 passed; the 14 added are this round's own tests.
Nothing lands after that stamp.

A first attempt ran the WHOLE of tests/ocean (7674 selected) rather than round
39's selection and was stopped and re-run at the matching protocol -- a
protocol change is a confound even when it is a widening, and the four test
files this round touches were run directly besides (15 passed on the
coordinate pair, 27 on the two gate files).

Evidence under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round40/`,
`round40_evidence.sha256` over its files, plus the two acquisitions' own
manifests under `round40_oracle_stage3_terms/` and `round40_oracle_ldfslp/`.

## Round 41 — `dyn_adv` split preregistered; stopped at the acquisition boundary

Round 41 starts at `c9feb3205c9f`.  Its prior-to-measurement contract is
`manifests/nemo_testcase_l2_gyre_round41_preregister.json`; the adversarial
pre-code review is beside it.  The acquisition refuses a dirty tree, so these
files must be committed before the operator runs it.  No NEMO executable,
`makenemo`, or `mpirun` was run by the agent, and no NEMO source was edited.

### Stage-3 vector-advection alignment

| statement | NEMO's compiled GYRE program | legoESM statement | finding before measurement |
|---|---|---|---|
| dispatch and time level | `GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynadv.f90:134-138` calls KEG then ZAD with `Kmm`; the stage diagnoses `ww` from `uu/vv(Kmm)` immediately beforehand at `GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/stprk3_stg.f90:326-332` | stage-3 vector-invariant C2 KE plus NEMO-advective ZAD, on the stage-2 state | branch/time level ALIGNED |
| C2 KEG | `GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynkeg.f90:117-130` rounds `zu`, rounds `zv`, forms `0.25*(zv+zu)`, then differences it and multiplies stored reciprocal metrics | `ocean_pe_latlon_cgrid.py:2033-2067` now materializes the four products and source-ordered pair sums; round 41's pre-measurement one-expression observation is RETRACTED | exact association landed in round 42 |
| ZAD transport/thickness | `GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynzad.f90:105-137` sums adjacent `e1e2t*ww`, multiplies a Kmm velocity difference, and scales with live Kmm face thickness | `ocean_pe_latlon_cgrid.py:3259-3285` area-weights/interpolates `ww` and calls the one shared NEMO-advective helper with Kmm thickness | algebra ALIGNED; reciprocal and accumulation association UNMEASURED |
| `ww` / `wsd` | `GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sshwzv.f90:271-298` builds `ww` bottom-up from Kmm velocity and the Kaa-Kbb QCO stretch.  Resolved `ln_wave=F` takes `GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sbcwave.f90:408-423`, which returns before allocating `wsd` at `:469-477`; ZAD therefore takes its `ww`-only arm | the coupled QCO seam provides stage-consistent `ww`; no Stokes vertical velocity is added | branch ALIGNED; numerical identity UNMEASURED |

**PREREGISTERED PREDICTION.**  KEG carries the first non-bit statement and the
round-40 `dyn_adv` debt: at least one KEG face is DEBT relative to KEG's own
magnitude, while both ZAD faces are AT-BAR.  A green KEG or any red ZAD
**REFUTES** that exclusive prediction; if both are red, both first unequal
statements are named rather than assigning the total by subtraction.

### Acquisition and controls

The additions-only writer records the `Krhs` frames before KEG, after KEG and
after ZAD; Kmm velocities; the exact `ww`; live/reference T/U/V/W thicknesses;
T/U/V areas and reciprocal metrics; and T/U/V/W masks.  It never references
the unallocated `wsd`; it records `ln_vortex_force=0` and a writer-defined
all-zero `wsd_effective`.  The reader first requires a literal source-order
replay of both sequential accumulator updates at 0 unequal cells, then closure
of `after_zad` against round 40's `after_adv` at 0 unequal cells.  Header,
calibration, closure, stamp, KEG and ZAD each have a named nonzero-exit plant.

The operator command is:

```bash
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round41_dynadv_split/run.sh
```

The script writes only under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round41/oracle_dynadv_split/`,
runs the raw twin report with consumed-field admission fall-through, hashes
every oracle record and gate report, and prints READY only after validation.

### Receipt hygiene: exactness vocabulary and one ORCA2 specification table

The round-38--40 `DISCHARGED` audit found no remaining at-bar-but-not-exact
mislabel: the tank ordering rows and GYRE kt=2 T/S rows explicitly say
AT-BAR-NOT-EXACT; round 40 uses DISCHARGED only for the four zero-unequal
stretch statements.  No historical wording needed alteration.

| open shared lane | ORCA2 status | exact acquisition/specification required to close it |
|---|---|---|
| round-39 before-state isoneutral fold | UNMEASURED WITH SPEC | native ORCA2 before-state T/S/ssh, `nmln`, `hmlp`, slope operands and folded K33 at the first nonzero-slope step; score the same model operand path and trajectory |
| round-40 QCO stretch | UNMEASURED WITH SPEC | native ORCA2 ssh, stored reference reciprocal thicknesses and resulting live T/U/V/W thicknesses on every topology-owned point, plus a native trajectory gate |
| round-41 KEG/ZAD split | UNMEASURED WITH SPEC | first resolve ORCA2's vector/flux and implicit-ZAD switches; if this arm executes, acquire the same pre-KEG/post-KEG/post-ZAD frames, `ww`/effective-`wsd`, live thicknesses, metrics and masks, then score through its native card |
| all shared changes' trajectories | UNMEASURED WITH SPEC | construct the missing native ORCA2 card on this branch and run its mechanically identical kt=1..10 trajectory contract; no lat-lon surrogate can discharge tripolar topology |

### ASKED / UNASKED and stop

| choice | status |
|---|---|
| write the split acquisition and stop | ASKED; complete in the repository, record UNMEASURED pending operator |
| decision 23, mixed-layer criterion | OPEN, NOT AUTHORISED; unchanged |
| decision 22, XLA ISA/FMA flag | CLOSED NOT ADOPTED; unchanged |
| model arithmetic, scheme/default choice, stabiliser | UNASKED; none changed |
| NEMO run/build, push, merge, deletion | forbidden to the agent; none performed |

UNASKED list: **empty**.  The merge verdict remains **HOLD** until the record
is returned and kt=2 U/V clear the exact bar.

## Round 42 — admitted `dyn_adv`, exact C2/`ldf_slp` statements, decision 23

### Round-41 blocker retraction and calibration

The real stream header decodes to
`(version,kt,kstg,Kbb,Kmm,Krhs,Kaa,nn_dynkeg)=(1,1,3,2,2,3,3,0)`.
The former Kbb=1 admission rule is **RETRACTED**: this build calls
`dyn_adv(kstp,Kmm,Kmm,...)` at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:466-472`, and the
writer records those local arguments at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynadv.f90:176-185`.
The model-side halo contract is last-owned-column substitution in x and zero
in y.

The independent source-order calibration is 0 unequal for KEG U/V, ZAD U/V,
and round-40 closure U/V.  The wrong four-square KEG association and wrong
reference-thickness ZAD divisor both fail; header, calibration, closure,
stamp, KEG and ZAD plants all exit nonzero.  Given NEMO operands, before the
fix KEG is DEBT on U (2/17400, 5.169878828456423e-26, relative
2.887126798214743e-13) and V (2/17100, 2.5849394142282115e-26, relative
1.028551927420229e-13); ZAD is exact on both.  The preregistered verdict at the
pre-fix commit is **CONFIRMED**.  The final fixed gate prints
**PREREGISTERED-VERDICT REFUTED** because P2 mechanically requires KEG to stay
DEBT; making KEG exact intentionally falsifies that pre-intervention
conjunction.  NEMO separately evaluates the four square products, then the
two sums and `0.25*(zv+zu)` at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynkeg.f90:117-130`.
Materializing that source association in the one shared C2 path makes all
four KEG/ZAD rows 0 unequal.

### Decision 23 and compiled `ldf_slp` walk

The shared NEMO identity now selects `n2_integral`; DINO already resolves the
same value and is unchanged.  On NEMO's kt=2 state, `nmln` and live `hmlp`
are each 0/600 unequal.  This is the compiled positive-N2 integral and live
depth path at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdfmxl.f90:109-123`.

After that decision the first non-bit statement is the stored-reciprocal
metric multiplication at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/ldfslp.f90:222-232`: division
misses 6413 U and 6236 V values; multiplication is exact.  With the compiled
live face/depth and source-parenthesized W/Shapiro associations at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/ldfslp.f90:235-275` and
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/ldfslp.f90:284-333`, the model's
own path gives 0 unequal for zau, zav, uslp, vslp, wslpi and wslpj.  Feeding
those four recorded slopes through the shared A33 fold also gives 0/17400;
the coefficient/slope association is compiled at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/traldf_iso.f90:793-794`.
The card's end-to-end K33 residual remains 1.4583733349836258e-10
(17400/17400): the remaining 0.43%-of-NEMO-maximum boundary is upstream
model-side prd/rn2 production, not `ldf_slp` or A33.  It is **DEBT**, never
DISCHARGED.

### Rule 12 and gates

GYRE kt=1 is exact.  Its first-over-bar remains kt=2 U/V before and after:
2.7478404751243857e-12 and 3.305560306813421e-12.  The exact criterion first
moves the kt=2 tracer update, visible at the kt=3 before-state boundary; all
70 trajectory rows and 53 worsening rows are registered in
`round42_before_to_final_GYRE_compare.json`.  That comparison gate is
**FAIL**, not PASS: 53 rows violate the no-worsening contract, led by
`kt10.before.T` at 1.0063482509536925e-05.  The KEG-to-slopes comparison is
also **FAIL** with the same 53-row boundary, so the slope/criterion leg owns
the regression; the scope correction below is remeasured before any final
claim.  The stage-3 advection U maximum
moves 2.0614443630503727e-16 to 2.0614443630503688e-16; V remains
2.482727869951749e-16.  LOCK_EXCHANGE and OVERFLOW resolve flux-form momentum
and no GM/Redi, so neither changed statement executes; their independent
kt=1..10 comparisons are bit-identical.  ORCA2 remains **UNMEASURED WITH
SPEC**: acquire native tripolar prd/rn2/slopes/A33 and vector-advection records,
then its kt=1..10 trajectory.

Round-42 gate reports and SHA-256 evidence are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round42/`, but every record
stamped `a75339c05277...` is **INVALID AS FINAL EVIDENCE**: that commit does
not exist in this repository.  Those final-status claims are retracted until
every gate is rerun and fail-closed at the shipped review-fix commit.  ASKED:
decision 23 and exact compiled statements.  UNASKED: empty.  No NEMO source,
build, or executable was touched.

### Independent-review corrections carried into round 43

The preceding `UNASKED: empty` statement is **RETRACTED**.  Decision 23 was
scoped to GYRE, but its five selectors were applied to every GM/Redi NEMO
lat-lon recipe; the source association was also widened to every native-slope
caller.  The selector bundle is now confined to the resolved GYRE path, and
the association change is confined to the existing DINO oracle and GYRE
identity paths pending the user's scope decision.

| NEMO lat-lon recipe | current slope path | open exactness row |
|---|---|---|
| GYRE | decision-23 live geometry and literal associations | measured here |
| rest | pre-round-42 defaults | raw NEMO `gdept_0`/`gdepw_0` absent; UNMEASURED |
| Eady | pre-round-42 defaults | raw NEMO `gdept_0`/`gdepw_0` absent; UNMEASURED |

| choice | historical status | corrected disposition |
|---|---|---|
| apply decision-23 selectors to every GM/Redi NEMO recipe | UNASKED round 42 | reverted outside GYRE |
| apply the literal slope association to every native-slope caller | UNASKED round 42 | limited to NEMO identities; wider scope ASKED and pending |
| keep the association limited to NEMO identities or widen it globally | ASKED round 43 | pending; no non-NEMO card moves |

Current UNASKED list: **empty**.  Historical round-42 UNASKED list: the two
rows above.

## Round 46 — kt=2 momentum stage acquisition, stopped before execution

Round 46 starts from `14df593e10fe`.  No NEMO build or integration has been
run by the agent.  The preregistered WRITE-only instrument records kt 1 and 2,
stages 1–3, following the compiled source order and exact Kbb/Kmm/Kaa call
tuples at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:201-215`.
Stage 1 spans the executing HPG/LDF/VOR/WZV/KEG/ZAD calls at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stp2d.f90:141-166`; stages 2–3
span WZV and HPG/VOR/ADV at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:327-333` and
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:447-469`, with
stage-3-only LDF/ZDF at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:704-717` and
all-stage barotropic replacement at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:724-743`.

The record includes named/ranked per-field extents, all RHS frames, stage
states, live/reference e3 and r3 families, Kbb/Kmm/Kaa velocity and tracer/SSH
levels, barotropic memory, masks, WZV fields, and TKE closure memory.  The gate
parses through EOF, requires exact WZV/KEG/ZAD source replays, runs NEMO-given
inputs through legoESM's own KEG/ZAD paths, and can start a kt=2 legoESM step
from the recorded complete kt=1 endpoint.  Header, truncation, calibration,
given-input, trajectory, legacy-twin, consumed-field-admission, and commit
plants are fail-closed.  Scientific status remains **UNMEASURED** until the
operator runs the acquisition.

### Round-46 ASKED / UNASKED register

| choice | status | disposition |
|---|---|---|
| literal slope-association scope | **ASKED-and-answered, 2026-09-12** | NEMO-identity cards only, exactly the scope implemented by `92c00497f48f`; the round-43 pending row is closed |
| acquire kt=2 stage records and stop | ASKED round 46 | instrument and gate written; awaiting operator run |
| configuration/physics/default change | UNASKED | none |

The preregistered first differing statement is kt=2 stage-1 after ZAD, with
exact earlier frames and exact WZV replay required.  Any earlier difference,
any WZV replay difference, or exact after-ZAD refutes it.  Merge readiness:
**HOLD — acquisition pending**.
