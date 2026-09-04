# NEMO testcase lane 2 GYRE — Phase 3 rounds 8–17 boundary receipt

**Verdict: DEBT.  Round 17 converged per-statement rounding to one core owner
and made stage-1 HPG/Krhs, the depth reduction, and post-drag slow forcing
bit-exact against Oracle V2 under production JIT/CPU/fp64/scalar-libm.  The
first remaining boundary is the surface-wind assembly (`6.62e-24`, 3 ULP U / 4
ULP V), so the conditional trajectory, cross-card, stage-2, and stage-3 work
was not entered.**

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
| stage-1 U/V `Krhs` | `1.3869160759180077e-20` | `0` | `0 / 0` | CONFIRMED HPG association owner |
| e3, mask, reciprocal depth | `0` | `0` | `0 / 0` | BIT-EXACT inputs |
| reference-depth reduction | `5.9922313054079714e-21` | `0` | `0 / 0` | BIT-EXACT after HPG fix |
| post-drag forcing | `5.9922313054079714e-21` | `0` | `0 / 0` | BIT-EXACT |
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
| `Krhs`, depth mean, post-drag | `0 / 0` | `0 / 0` | retained BIT-EXACT |
| face stress differing cells | `414 / 418` | `0 / 0` | BIT-EXACT |
| post-wind max abs | `6.617444900424222e-24 / 6.617444900424222e-24` | `0 / 0` | BIT-EXACT |
| pre-external max abs | `6.617444900424222e-24 / 6.617444900424222e-24` | `0 / 0` | BIT-EXACT |

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
