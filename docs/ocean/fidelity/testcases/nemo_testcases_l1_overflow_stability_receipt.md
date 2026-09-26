# Lane 1 OVERFLOW-zps non-finite investigation receipt

Status: **NON-FINITE FAILURE OWNER: CONFIRMED (completion in both precisions);
ROOT DIVERGENCE OWNER: PLAUSIBLE.**  The
source-exact NEMO 5.0.2 `ln_zad_Aimp` RK3 package makes the certified legoESM
OVERFLOW-zps card complete all 6,120 steps in both fp64 and fp32.  The prior
failures at completed steps 2,877/2,879 were therefore caused by legoESM's
approximate adaptive-implicit composition.  No damping, clipping, diffusion,
limiter, or oracle-absent selector was added.  The frozen statistical scorer
now returns a fully classified `OUTSIDE` verdict.  After the round-4
barotropic source correction and same-revision fp32 floor rerun, four metrics
were `OUTSIDE` and two `WITHIN-SCHEME-SPREAD`; after the round-5 RK3
stage-composition fixes (kt=2 U owner MEASURED and fixed, `3.31e-6 ->
2.60e-7`) three metrics are `OUTSIDE` and three `WITHIN-SCHEME-SPREAD`; no
metric is `INDISTINGUISHABLE-AT-FLOOR`.

## Round 7: the WS momentum ladder is written once (S-30 / S-12 collapse)

Full ledger, before/after tables, the ulp bar and its planted control in
`nemo_testcases_l1_phase3_receipt.md` ("S-30 / S-12 collapse").

The Wicker-Skamarock RK3 momentum recurrence was written TWICE inside one
function: once before the barotropic solve, purely to build that solve's
velocity seed, and once after it, corrected and kept.  NEMO has no ladder at
the first site — `stp_2D` evaluates the Kbb RHS once, depth-means it
(`stp2d.F90:177-186`) and hands `dyn_spg_ts` the BEFORE velocity
(`stp2d.F90:280-281`) — so the first copy is deleted and the solve is seeded
with `u0`/`v0`.  This also removes two `tendencies()` evaluations per step and
closes the failure mode that put the stage vertical-UP3 and live-HPG fixes in
one copy only.

NOTHING in this receipt's numbers changes beyond roundoff.  The round-6 rows
above are re-pinned at the collapse commit
`28df515a84d572b28a7a5c6afb1ca905b31bfda4`, fp64, CPU:

| round-6 row (OVERFLOW-zps, kt=2 absolute L-inf) | round 6 | after the collapse |
|---|---|---|
| stage-1 u | `6.501744e-15` | `6.5225603e-15` |
| stage-2 u | `9.433404e-11` | `9.4334038e-11` |
| stage-3 u | `2.598798e-07` | `2.5987979e-07` |
| kt=2 u | `2.598798e-07` | `2.5987979e-07` |
| kt=2 SSH (normalized) | `1.0491608e-14` | `1.0505485e-14` |

Every one of those is a move of at most 0.094 float64 ulp of the row's own
scale; the largest move anywhere in the kt=1..10 trajectory of either card is
0.5 ulp (OVERFLOW kt=10 SSH).  T and S are BIT-IDENTICAL at every step of
both cards.  `first_over_bar` is unchanged (LOCK `kt=2 {T,u}`, OVERFLOW
`kt=2 {T,u,ssh}`).  The kt=2 velocity debt is therefore unchanged and stays
UNOWNED — this collapse is exonerated for it.

The 6120-step statistical scorer was NOT re-run, and after adversarial review
the 3 OUTSIDE / 3 WITHIN-SCHEME-SPREAD split is recorded as **UNMEASURED**,
not as "expected to hold".  The gate's own growth block for OVERFLOW gives tail
exponential rates of `0.203/step` (ssh), `0.269/step` (u) and `0.083/step` (T)
and cannot discriminate exponential from polynomial on a four-point tail, so a
`1e-16` seed either reaches order one in a few hundred steps or never matters,
and 10 steps do not tell us which.  Discriminator: ~300 steps on both
revisions, `max|delta u|` per step on a semilog axis.

Two further corrections from the same review, both detailed in the phase-3
receipt: the tracer FIELDS are not bit-identical even though the tracer ROWS
are -- OVERFLOW T departs from kt=7 at five cells, reaching `1.07e-14 K` by
kt=11 -- and the "pure re-association" claim is certified for **f = 0 only**,
because both L1 cards are non-rotating and the Matsuno rotation that sits
between the deleted ladder and the barotropic seed is the identity there.  On a
rotating rk3_ws card (GYRE) it is not, and that card has no gate here.

The bar is mechanical, not a judgement: `legoesm.ocean.fidelity.ulp_move_gate`,
reached as `--compare-to` on both phase-3 gates, with a `--compare-plant-ulps 3`
control that must (and does) turn it red.

## Round 6: qco stage face thickness and stage weighting

Full ledger, source citations, reach table and controls in
`nemo_testcases_l1_phase3_receipt.md` ("Face-thickness and stage-qco round").
Preregistered in
`nemo_testcases_l1_overflow_face_thickness_preregister.md` before either arm
existed.

The kt=2 TEMPERATURE residual that round 5 left as its registered next owner
is CLOSED.  Two unbranched corrections to the WS-RK3 identity: the stage
transport now uses NEMO's `e3u(Kmm) = e3u_0*(1 + r3u(Kmm))`
(`domqco.F90:219-222`, `domzgr_substitute.h90:127`) instead of the min of the
two stretched T thicknesses, and every stage velocity update now carries the
qco weighting `(1+r3u(Kbb)) / (1+r3u(Kmm)) / (1+r3u(Kaa))`
(`stprk3_stg.F90:373-378`).  The tracer stage already carried its analogue.

kt=2, OVERFLOW-zps, absolute L-infinity on the wet mask:
T `2.548493e-07 -> 2.238210e-13 K` (1.14e6x); u `2.598440e-07 ->
2.598798e-07 m/s`; SSH unchanged at `1.0491608e-14` normalized.  Stage-1 u
`3.757022e-12 -> 6.501744e-15`, stage-2 `6.908155e-11 -> 9.433404e-11`,
stage-3 `2.598440e-07 -> 2.598798e-07`.  kt=10 T `5.55186262e-06 ->
7.71178526e-08` normalized; kt=60 before-entry T `1.027641e-04 ->
1.184430e-05`, u `4.193114e-04 -> 2.507515e-04`.

Six of seven frozen predictions were MET, including the two that
preregistered these hypotheses as REFUTED for the u residual.  The miss is
P3b: the stage-2 u residual does not collapse — both corrections make it
slightly WORSE than the pre-fix `6.908e-11` (`9.433e-11`).  Rule 8:
disclosed, not reverted, four orders below the stage-3 residual.

The 6120-step statistics keep 3 OUTSIDE / 3 WITHIN-SCHEME-SPREAD.  Every row
improves or holds (water-mass census 5.05x -> 3.88x the NEMO scheme spread,
instantaneous-u L-infinity 1.49x -> 1.24x) and none crosses, so the "3 -> 4
OUTSIDE" regression of round 4 remains at 3 and the long-run statistics stay
owned by something this round did not touch.

LOCK_EXCHANGE-zco is bit-identical at every kt=1 stage and at kt=2, which the
source predicts: its oracle `ssh` is identically zero at all three kt=1 stages
so both face rules and both velocity updates coincide.  Its kt>=3 T improves
1092x by kt=10; its u and SSH move a few percent in the worse direction.

## Round 5: RK3 stage composition (takeover of the codex round)

The round-4 HOLD is closed in commit `3a68e43338a4`: the 19 frames are
registered to their exact `dynspg_ts.F90` assignments and time levels, the
gate JSON stamps the legoESM git SHA and the pytest-log sha256, the planted
entry control exits nonzero end to end, and the literal flux-form external
update is disclosed as PRODUCTION-ACTIVE and KEPT (Rule 8).  Decision
record: with the update, the 19 substep-1 frames are 11 AT-BAR, 7 UNMEASURED
(no active V face) and one DEBT — `u_exit` at `1.87350135e-15`, which is
identical in the legacy arm and equals `dt x` the `5.62917768e-16` slow-U
input residual (ratio `0.998`), i.e. inherited from the slow forcing, not
produced by the update; substeps 2-4 go from `2.6e-9 / 1.3e-7 / 5.6e-7` to
`3.5e-15 / 4.9e-15 / 5.9e-15`.  The kt=2 pair
(U `3.08238867e-06 -> 3.31108168e-06`, SSH `1.23723132e-07 ->
1.04916076e-14`, T unchanged) stays disclosed side by side; the
"3 -> 4 OUTSIDE" regression of round 4 is re-scored below on the same
literal-update arm after the stage-composition fixes.

The kt=2 initiator the round-4 gate left UNMEASURED ("post-external RK3
stage composition") is now MEASURED and owned by three composition defects
inside the WS-RK3 identity, fixed in `614bed818bb4` (full ledger in
`nemo_testcases_l1_phase3_receipt.md`): FCT at tracer stages 1-2 where NEMO
runs `tra_adv_cen` (`traadv.F90:281-282,361-364`); stage-2/3 `eos+dyn_hpg`
on Kbb where NEMO uses the stage Kmm (`stprk3_stg.F90:317-320`); and the
`dyn_adv_up3` vertical flux applied once per step where NEMO carries it in
every stage RHS (`:315,331-334`).  kt=2 instantaneous U falls
`3.31108168e-06 -> 2.59844026e-07`, T `2.40034479e-08 -> 1.27424627e-08`,
SSH unchanged at `1.04916076e-14`; kt=10 U `1.40474350e-04 ->
2.64522041e-05`, T `1.35356455e-05 -> 5.55186262e-06`, SSH unchanged.

Both 6,120-step arms complete on CPU with every-step finite checks at
commit `c8f506a69545` (fp64 states `0599482dd41d...`, fp32 states
`c73306a060b8...`); the kt=1--60 bridge is re-pinned to
`stage_composition/overflow_trajectory_gate_kt60.json` (`086dbd6fc7a6...`),
replacing the round-4 bridge.  The reissued verdict on the SAME scorer,
floor protocol and NEMO spread:

| metric | before | after | fp32 floor before | fp32 floor after | NEMO spread | verdict before | verdict after |
|---|---:|---:|---:|---:|---:|---|---|
| final_temperature_histogram_tv | `0.0537796` | `0.0423046` | `0.0237045` | `0.0194058` | `0.044231` | OUTSIDE | WITHIN-SCHEME-SPREAD |
| final_water_mass_census | `0.0150395` | `0.0169753` | `0.00349422` | `0.00157409` | `0.00336209` | OUTSIDE | OUTSIDE |
| instantaneous_u_linf | `0.969889` | `0.999211` | `0.456889` | `0.402734` | `0.672694` | OUTSIDE | OUTSIDE |
| plume_descent_m | `0.904647` | `16.9554` | `0.100003` | `0.0087228` | `1499.62` | WITHIN-SCHEME-SPREAD | WITHIN-SCHEME-SPREAD |
| plume_front_km | `1.00947` | `4.04936` | `0.561019` | `0.0837761` | `121.931` | WITHIN-SCHEME-SPREAD | WITHIN-SCHEME-SPREAD |
| temperature_linf | `0.394129` | `0.379492` | `0.0809865` | `0.140318` | `0.356744` | OUTSIDE | OUTSIDE |

Round-5 artifacts under `/data/abyssal/dbalwada/nemo-testcases-l1/`:

| artifact (under `/data/abyssal/dbalwada/nemo-testcases-l1/`) | SHA256 |
|---|---|
| `barotropic_walk/review_round/frame_gate_hold_round_ab6ca17b6.json` | `59c0a4e056e87d74af04937a8638dcc279840c6f7ae7e78abf4499e6c7d43e42` |
| `barotropic_walk/review_round/pytest_barotropic_gate_hold_round.log` | `64a815b7a0608d37ab43bcd3a7377d8a559ecd7692fd45baaf9818f0ce66b62a` |
| `stage_composition/legoesm/overflow_zps/fp32/metadata.json` | `dfaa8b66f927bd3e6a75bc9aae79771ad1390f29f00fbe91ad213c9fedc28207` |
| `stage_composition/legoesm/overflow_zps/fp32/states.npz` | `c73306a060b8cd74bdabc56465eefa4b6414d403ea0cc1d9187bf6b2d20de3f7` |
| `stage_composition/legoesm/overflow_zps/fp64/metadata.json` | `6470d575cca23987c94b73696bc2f811e3f65f099d9086931ace43247733150a` |
| `stage_composition/legoesm/overflow_zps/fp64/states.npz` | `0599482dd41dba259494cc277c4ad36dabbb2c7bdd370177d5430226038ccdff` |
| `stage_composition/lock_stage_sweep_gate_kt2.json` | `a3716cb258e33da4143d4da3fc5bd0fde2ebedc94872a4606a51c766809f5a4c` |
| `stage_composition/lock_trajectory_gate_kt10.json` | `4332b47664820b6a67769f17ea4de5c1d2efc449db7070d8a3f3564d350788c6` |
| `stage_composition/overflow_stage_sweep_gate_kt2.json` | `5bdcd995da66758199f3ad2db5cc44b97f42e093962476c2b86624d676c5dc9c` |
| `stage_composition/overflow_statistics.json` | `a9aca2d78dae62e8de4da4edd61233b101160637ed769a476254841f3c05df26` |
| `stage_composition/overflow_trajectory_gate_kt10.json` | `bcc8a68d26c918954329173fe955aec82155502db5b64743cac9ab963ab59377` |
| `stage_composition/overflow_trajectory_gate_kt60.json` | `086dbd6fc7a61ea328ecda692d7ef3491092f9fc6127db5a661d5f81f5967712` |

## Round 4: OVERFLOW 19-frame external-mode walk

Commit `c5882cb8aa18` froze the comparison before reading any new frame.  A
separate NEMO configuration, `OVERFLOW_OMIP_L1_BTWALK`, added WRITE-only
instrumentation to the executed `nn_bt_flt=1`, `rn_bt_alpha=0`, `nn_e=3`
branch.  The resolved `nn_e=3` produces four cold-start boxcar substeps.
The run uses `ln_bt_fw=T`, flux-form UP3 momentum, `ln_drg_OFF=T`, and `f=0`.
The instrumented run's kt=1 step-entry file is byte-identical to the certified
oracle (`cf0183e5...`), so instrumentation did not perturb the overlap state.

The 19-frame gate compares NEMO and legoESM at the same instantaneous native
T/U/V staggering after NEMO's two-cell halo strip, on the common certified wet
mask, using an elementwise L-infinity reduction with no depth or substep-time
average.  The time-level header is `Kbb=1, Kmm=1, Kaa=3`; the full registry is
entry T/U/V, midpoint T/U/V, U/V transport, continuity SSH, PGF SSH and U/V,
slow U/V, drag U/V, exit U/V, and exit SSH.  The reader hard-fails on magic,
version, dimensions, time levels, substep count, field count, truncation, or
trailing bytes.  Independent planted entry and exit controls make the gate
red.

Substep 1 is exact through midpoint transport, continuity, and PGF.  The U
slow-forcing row differs by `5.62917768e-16`, still AT-BAR.  Multiplication by
the `10/3 s` external step predicts `1.87639256e-15`; the measured first strict
DEBT is the U exit at `1.87350135e-15`, ratio `0.99846`.  Drag is exact zero in
both dumped operands, confirming the resolved OFF arm rather than inferring it
from geometry.  PGF, continuity, slope face depth, and partial-cell metrics are
therefore **CONFIRMED EXONERATED through the first strict boundary**.  The
slow arithmetic tail is the **CONFIRMED first-boundary contributor** but is
about nine orders below the kt=2 instantaneous-U debt and is
**REFUTED as its root owner by scaling**.

Source reading exposed a separate executed mismatch.  NEMO advances face
transport, not velocity, in `dynspg_ts.F90:731-761`:

`(hu_e*un_e + dt*(zhu_bck*spg + zhup2*trd + hu(Kmm)*frc)) / hu_a`.

Commit `0ff51eca2bfd` transcribes that expression as an unbranched part of the
NEMO WS-RK3 + flux-form identity; NEMO has no switch, so legoESM gets no public
switch.  The legacy velocity update is accessible only through a private gate
hook.  A synthetic test changes the midpoint-depth operand and proves the
literal test is non-vacuous.  The arm is scale-causal inside the loop:

| substep | legacy U-exit error | NEMO-form error | improvement |
|---:|---:|---:|---:|
| 1 | `1.87350135e-15` | `1.87350135e-15` | `1.0x` |
| 2 | `2.55383398e-9` | `3.53189700e-15` | `7.23077e5x` |
| 3 | `1.28549120e-7` | `4.94743135e-15` | `2.59830e7x` |
| 4 | `5.60239321e-7` | `5.87030424e-15` | `9.54362e7x` |

This confirms the literal update as the **structural owner of the downstream
external-loop recurrence debt**.  It does not own the whole-step initiator.
The frozen criterion required kt=2 instantaneous U to fall by at least 10x
without a greater-than-10x T/SSH regression.  Instead U moves
`3.08238867e-6 -> 3.31108168e-6` (a plain `7.42%` regression), T is unchanged
to six significant figures, and SSH improves
`1.23723132e-7 -> 1.04916076e-14`.  The root-initiator ownership claim is
therefore **REFUTED BY THE FROZEN CAUSAL PREDICATE**.  The external substep
register is exhausted: the remaining kt=2 U/T root operand is
**UNMEASURED outside this register**, in the post-external RK3 stage
composition.

The required kt=2--10 regression is below.  Values are normalized common-wet
L-infinity errors; U is instantaneous C-grid U-face Nbb on both sides.

| kt | T before | T after | U before | U after | SSH before | SSH after |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | `2.40035055e-08` | `2.40034479e-08` | `3.08238867e-06` | `3.31108168e-06` | `1.23723132e-07` | `1.04916076e-14` |
| 3 | `2.86533772e-07` | `2.86529500e-07` | `6.46235078e-06` | `9.22615336e-06` | `5.12880845e-06` | `3.66567551e-09` |
| 4 | `9.70053135e-07` | `9.70021745e-07` | `1.16440779e-05` | `1.50603834e-05` | `2.64530059e-05` | `1.07010798e-06` |
| 5 | `1.99089401e-06` | `1.99077587e-06` | `2.80038578e-05` | `2.93540699e-05` | `4.74630495e-05` | `1.48167577e-05` |
| 6 | `3.25857142e-06` | `3.25827731e-06` | `5.97568478e-05` | `5.59331066e-05` | `4.72344006e-05` | `4.00067365e-05` |
| 7 | `4.93304987e-06` | `4.93252050e-06` | `8.16437683e-05` | `7.99314090e-05` | `6.97196664e-05` | `4.55653806e-05` |
| 8 | `7.23683063e-06` | `7.23598252e-06` | `8.92508130e-05` | `9.16463001e-05` | `8.65979411e-05` | `7.97549653e-05` |
| 9 | `1.01541688e-05` | `1.01527667e-05` | `1.05130255e-04` | `1.07033146e-04` | `8.39242799e-05` | `7.29954633e-05` |
| 10 | `1.35379486e-05` | `1.35356455e-05` | `1.41894266e-04` | `1.40474350e-04` | `8.66359613e-05` | `9.24576527e-05` |

LOCK's kt=2 regression is neutral: T stays `1.58214182e-13` and U changes
`1.71208684e-10 -> 1.71208363e-10`.  At kt=10 U changes
`4.35050760e-8 -> 4.47830601e-8` while SSH improves
`1.06780006e-10 -> 2.71415052e-13`.

Both same-revision full-duration arms complete on CPU with every-step finite
checks: fp64 in `404.5752 s`, fp32 in `268.5355 s`.  The current kt=1--60 gate
(`197a8959...`) replaces the stale pre-arm bridge in the statistical scorer.
The reissued verdict is:

| metric | candidate | same-revision fp32 floor | NEMO scheme spread | verdict |
|---|---:|---:|---:|---|
| plume descent | `0.904647 m` | `0.100003 m` | `1499.623281 m` | `WITHIN-SCHEME-SPREAD` |
| plume front | `1.009466 km` | `0.561019 km` | `121.930713 km` | `WITHIN-SCHEME-SPREAD` |
| final T histogram TV | `0.0537796` | `0.0237045` | `0.0442310` | `OUTSIDE` |
| final water-mass census | `0.0150395` | `0.00349422` | `0.00336209` | `OUTSIDE` |
| instantaneous U L-inf | `0.969889` | `0.456889` | `0.672694` | `OUTSIDE` |
| T L-inf | `0.394129` | `0.0809865` | `0.356744` | `OUTSIDE` |

Round-4 artifacts under
`/data/abyssal/dbalwada/nemo-testcases-l1/barotropic_walk/`:

| artifact | SHA256 |
|---|---|
| NEMO WRITE-only `dynspg_ts.F90` | `f1ebfe150001de7c08e378749f3386e65374f83ca459af7fad8fa35ce43a73c0` |
| NEMO binary | `84d2fb40eb6864921206335c4e1f2520d2120df1b2527945e23cbbe939b85284` |
| NEMO resolved run namelist | `17cdebf82b03d9db69c32a5b6e54c7d089958e73daaff0ab60add4d3afdd6d5e` |
| NEMO frame stream | `02b7e53362da1a69e695bdb9dab86ee6a8362b6906fef8d93f977372c26b035e` |
| 19-frame gate | `b46f7390ef83702b0595fc267121008d170d7e36143eb7a03e2bd9638209a65b` |
| current kt=1--60 gate | `197a8959f9c72814c6c3a29fe92452ceaf87aafca5bd900536bc479a2ae83be2` |
| LOCK kt=1--10 regression | `2a9221853bb07a314d37de299ea82768a6a17753a5dc4c99bbe965786ba256c1` |
| fp64 metadata / states | `ac148830e3b099e3d1a94566f0faec2a1dc1a7325fae2f147d0e00c35bd862c1` / `7ea95685241a631ebbea1b8ef554793f68ce4625fdcc4024b9c2649f2e607ea8` |
| fp32 metadata / states | `014b21ac1aece9fcf0bf0263b900e53a9e2761b15cb375174977a242a20a2998` / `53cdc440425c03ed992abee044184014e1b2333de1fa5c6d5e7c35309595550e` |
| statistical report | `a99ebd4007426ebd35a8dbe30e1fa4c0a17bb3e3176af66add3b0963b99eab58` |

Focused CPU/fp64 validation is **56 passed**: 5 barotropic-gate controls,
15 full-statistics controls, 9 real WS-RK3 tests, 15 testcase-card tests, and
12 OVERFLOW stability-probe controls.  This is the collected five-file
breakdown, not a cumulative campaign-test count.

## Round 3 resolution: source-exact adaptive-implicit package

Commit `7876b3ea869f` froze the terminal prediction before any new integration:
root ownership required both completion of all 6,120 fp64 steps and reduced
matched approach-window growth at the same location; any later failure would
classify the package only as a contributor.  Commit `5b59e923e0bc`
then replaced the public NEMO WS-RK3 post-step approximation with one
unbranched source identity:

- The stage-3 transport-form Courant criterion uses horizontal inflow,
  sign-selected upstream-cell `Cu_h`, `Cu_min_v=0.8`, `Cu_max_v=1.1`, and
  `Cu_max_h=1.1` (`sshwzv.F90:773-843`).
- Stage-3 momentum sends explicit `ww` through UP3 and folds the
  e1e2t-area-weighted `wi` into the same vertical-viscosity tridiagonal
  (`stprk3_stg.F90:257-304,309-336,419-446`;
  `dynzdf.F90:181-195,233-270`).
- Tracers recompute the corrected stage-3 transport, use explicit `ww` in
  both FCT predictor steps, subtract the optimized `nn_fct_imp=1`
  zero-order `wi*T(Kbb)` divergence in both predictors, and fold the same
  `wi` into the one tracer ZDF matrix (`traadv.F90:220-227`;
  `traadv_fct.F90:140-145,470-607`; `trazdf.F90:207-216,271-285`).

The exact package completed 6,120 fp64 steps on CPU in `467.8001 s`, with
every-step finite checks, and completed the fp32 companion in `288.5956 s`.
Completion in both precisions confirms ownership of the **non-finite failure**.
No matched approach-window growth row was computed, and U/SSH already leave the
NEMO range at kt=2 while the package acts downstream.  The full frozen root
predicate is therefore unsatisfied: **root divergence ownership remains
PLAUSIBLE**.  The former near-identical fp64/fp32 failure pair was a
deterministic structural mismatch at the terminal event, not evidence that the
package initiated the earlier trajectory divergence.

The short gates did not regress their certified classification.  kt=1 remains
bit-exact for T/S/U/SSH.  At kt=2 the normalized L-infinity rows are T
`2.40035055e-8`, U `3.08238867e-6`, and SSH `1.23723132e-7`; the U row improves
from the prior approximately `4.12e-6` class, while T and SSH remain at their
recorded values/classes.  Resolved coverage is now 68 VERIFIED, 40 WAIVED,
and **0 UNMEASURED**; `namzdf.ln_zad_aimp` is no longer a boolean-only claim.

The required kt=3--10 regression was scored against the same hash-pinned kt60
oracle root as the certified predecessor (`98856a4f...`), resolving the
otherwise disagreeing `4.121e-6` and `3.956e-6` historical kt=2 U values before
recording a table.  Values are normalized wet-intersection L-infinity errors;
U is instantaneous C-grid U-face `Nbb` on both sides.

| kt | T before | T after | U before | U after | SSH before | SSH after |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | `2.86506559e-7` | `2.86533772e-7` | `1.48996998e-5` | `6.46235078e-6` | `5.12822130e-6` | `5.12880845e-6` |
| 4 | `9.69751797e-7` | `9.70053135e-7` | `3.12223642e-5` | `1.16440779e-5` | `2.64646724e-5` | `2.64530059e-5` |
| 5 | `1.98970837e-6` | `1.99089401e-6` | `5.70663548e-5` | `2.80038578e-5` | `4.75242327e-5` | `4.74630495e-5` |
| 6 | `3.25521608e-6` | `3.25857142e-6` | `8.64905185e-5` | `5.97568478e-5` | `4.72451204e-5` | `4.72344006e-5` |
| 7 | `4.92499227e-6` | `4.93304987e-6` | `1.08568566e-4` | `8.16437683e-5` | `6.96928483e-5` | `6.97196664e-5` |
| 8 | `7.22006930e-6` | `7.23683063e-6` | `1.27503714e-4` | `8.92508130e-5` | `8.54599094e-5` | `8.65979411e-5` |
| 9 | `1.01233435e-5` | `1.01541688e-5` | `1.56823138e-4` | `1.05130255e-4` | `8.39364577e-5` | `8.39242799e-5` |
| 10 | `1.34856007e-5` | `1.35379486e-5` | `1.97738709e-4` | `1.41894266e-4` | `8.66446322e-5` | `8.66359613e-5` |

The package changes every post-initial step.  It reduces U by `2.30x` at kt=3
and `1.39x` at kt=10, while T moves upward by at most `0.39%` and SSH remains
within `1.33%` of the predecessor across this interval.  This is a regression
table, not a claim that any row reaches the `1e-15` bar.

The full-duration scorer first exposed and then repaired a latent mismatch
between its code and the frozen preregistration: the preregistered front is the
rightmost connected ascending crossing, whereas the code rejected any state
with more than one crossing.  Commit `a9fa35365478` applies the registered
rightmost reduction and adds multiple/missing-crossing controls.  It does not
relax a scientific bar.  The rerun produces a machine-readable `OUTSIDE`
report when L32 ended at `20.002197265625 C`: its `0.002197265625 K`,
`1.0986328125e-4` relative excess exceeded the original fixed `1e-6` gross
guard.  The reviewer-provided ratio-of-ratios `0.477` and approximately
`0.19 ULP/step` made precision accumulation **PLAUSIBLE-strong**, not proven.
Commit `b3a813c8a2e3` then froze a discriminating 3,060-step trace before it
ran.  The trace is finite and classifies `PRECISION_ACCUMULATION`: peak excess
`0.0027885437 K`, largest positive one-step jump `5.7220459e-6 K`, jump/peak
`0.002052`, and peak `0.3822 ULP/step`.  A limiter-like jump would have required
`jump/peak >= 0.5`.

Only after that result, the scorer admitted L32 as a floor arm using the frozen
relative eligibility guard `max(1e-6, N_steps*eps(dtype)) = 7.2956085e-4`.
The fp64 guard remains exactly `1e-6`.  L32's range row remains loudly
`UNMEASURED` against the stricter `sqrt(N)*eps` endpoint floor; the new guard
establishes floor-arm eligibility, not FCT monotonicity.  The reissued fp64
statistical verdict is:

| metric | candidate | fp32 floor | NEMO scheme spread | verdict |
|---|---:|---:|---:|---|
| plume descent | `0.904742 m` | `0.042421 m` | `1499.623281 m` | `WITHIN-SCHEME-SPREAD` |
| plume front | `1.009612 km` | `0.536798 km` | `121.930713 km` | `WITHIN-SCHEME-SPREAD` |
| final T histogram TV | `0.0529614` | `0.0181573` | `0.0442310` | `OUTSIDE` |
| final water-mass census | `0.0148908` | `0.00282271` | `0.00336209` | `OUTSIDE` |
| instantaneous U L-inf | `0.972586` | `1.071714` | `0.672694` | `INDISTINGUISHABLE-AT-FLOOR` |
| T L-inf | `0.394134` | `0.153583` | `0.356744` | `OUTSIDE` |

Thus OVERFLOW fp64 is still `OUTSIDE` statistically, now for measured metric
distances rather than an invalid floor arm.

Round-3 artifacts (all under
`/data/abyssal/dbalwada/nemo-testcases-l1/round3_aimp/`):

| artifact | SHA256 |
|---|---|
| resolved coverage | `376a53b4d893a831fee98c70fcde142bf5b424437d36c307f796b1fab0698401` |
| kt=1/2 gate | `683c175f8ea942c9ab9cba19f02fb3c35f96ba22024760581ca64260f9027e3f` |
| fp64 metadata | `fbd02221ddab494a11f7d1110694072b241360d7a060756a2a27fb0d44511a51` |
| fp64 states | `6a8520bb2cf82d74dc0c9658989cdda36f032fbc9bd4c4abc515fff07acc3373` |
| fp32 metadata | `7cc10f9f7dc171356a2df682096cef96e5d568a89be8f434071afa73c96b4084` |
| fp32 states | `4d24f03c79a4b25622d13f2b8226dde831ef021a613d86ce9c04ef5ebea06166` |
| fp32 3,060-step range trace | `a093cf1127da5a4572010d8f49fa7fda752b99b9d847ba1a47e0c58a79907dfa` |
| kt=1--10 gate, kt60 oracle frame | `5854acc4680f1a2c572b813331525b16b59959160874ca565fba18a332ee5670` |
| statistical report | `2fab37b6c8fadb35074efc403c5abe6716d6fe634070c2fb299b2e65ff886c49` |

## Pre-fix executive finding (retained as the first-divergence record)

The new every-step comparison separates initiation from terminal amplification.
The common state is exact at kt=1.  Instantaneous U and SSH first leave NEMO's
roundoff-padded wet range at kt=2 and become gross at kt=3, at the developing
front near x=21 km and z=490 m (U).  T remains inside NEMO's global wet range
through kt=60.  By kt=2,601 the largest instantaneous U error is 17.9033 m/s at
x=41 km, z=1,250 m and the SSH error is 3.82650 m near x=54.5 km; the dynamics
are already far outside NEMO before the terminal tracer event.

At completed steps 2,865--2,876, maximum one-step T increments grow from
1.88257 K to 314,761 K with raw successive ratios
`1.04, 1.11, 1.13, 1.43, 1.89, 1.12, 2.32, 1.72, 3.10, 2.37, 2.63, 574`.
The maxima migrate along the slope around x=41.5--43 km and z=850--1,250 m.
This is a late amplifying mode, not the earlier polynomial trajectory-debt
signature.  T/S/volume content remain closed to `5.84e-16`, `3.29e-16`, and
`1.80e-16` relative respectively, classifying the event as redistribution,
not an unbalanced source.

The same-input scale arm at completed step 2,875 is decisive but limited:
zeroing only tracer vertical transport removes 99.961% of the next T increment
at x=41.5 km, z=970 m, while changing U and SSH by exactly zero in that step.
Thus the current explicit vertical tracer flux is **PLAUSIBLE terminal T
amplifier**, but cannot be the initiating U/SSH owner.  NEMO's complete
adaptive-implicit RK3 package remains the highest source-attested mismatch; a
faithful unbranched implementation is required before it can receive a
`CONFIRMED` root-owner label.

## Rule 0: executed source and configuration

The matched NEMO rerun changes only `cn_exp`, `nn_itend/nn_stock`, and the
write-only dump condition.  Its `mesh_mask.nc` byte-matches the certified mesh
(`4692b893...`).  The resolved run selects adaptive implicit vertical
advection, constant mixing, `rn_avm0=1e-4`, and `rn_avt0=0`; Richardson, TKE,
GLS, enhanced-diffusion convection, and all other specific convection closures
are off (`overflow_zps/namelist_cfg:123-140`; matched `ocean.output:630-660`).
`zdfphy.F90:193-213` consequently dispatches no specific convection scheme.
Missing stabilizing mixing/convection is **SOURCE-EXONERATED**.

NEMO evaluates its adaptive Wicker--Skamarock partition twice in RK3 stage 3,
for velocity and transport (`src/OCE/stprk3_stg.F90:281-303`).  The routine
combines horizontal and vertical Courant numbers with the 0.8/1.1 thresholds
for the live coefficient and separately accumulates a bottom-up maximum for
diagnostic output; that maximum does not enter the split coefficient
(`src/OCE/DYN/sshwzv.F90:696-873`).  Its explicit and implicit tracer pieces
enter FCT and the vertical tridiagonal solve
(`src/OCE/TRA/traadv_fct.F90:141-167,286-330`;
`src/OCE/TRA/trazdf.F90:207-225`).

Before round 3, legoESM's only implementation was the older local vertical-only momentum form
in `packages/ocean/legoesm/ocean/vertical.py:1716-1909`; it is applied after
the completed RK3 program in
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4996-5054`.
The tracer WS-RK3 stages received unpartitioned vertical transport and had no
matching implicit tracer solve.  This was a **SOURCE-CONFIRMED MISMATCH**;
round 3's completed discriminating run promotes it to the confirmed stability
owner.

## Comparator frame, inventory, and dtypes

The committed probe compares NEMO's instantaneous Nbb step-entry fields at
`kt` against legoESM's instantaneous prognostic fields after `kt-1` completed
steps.  T/S/SSH use T centres, U uses C-grid U faces, and V has no active
meridional face in this three-row closed tank.  Every reduction is elementwise
L-infinity on the phase-3 gate's certified common wet mask; no depth or
substep-time averaging is applied.  The exact kt=1 control checks 17,000 T/S
cells, 16,900 U faces, and 200 SSH cells bit-for-bit.

The legoESM baseline ran on CPU with storage, compute, accumulation, control,
all five state arrays, and all five coordinate arrays recorded as fp64.  The
NEMO rerun also used CPU and was invoked directly, without `mpirun`.  Its 278
new dumps cover kt=2,601--2,878.  The scorer never calls a `nan*` reduction:
non-finite rows are inventoried explicitly and make the JSON verdict red.

## First-divergence and approach ledger

| matched state | measurement | disposition |
|---|---|---|
| kt=1 | T/S/U/SSH exact in the certified frame | VERIFIED |
| kt=2 | U range excess `8.45e-7 m/s`; SSH low-range excess `1.24e-7 m` | FIRST OUTSIDE |
| kt=3 | U and SSH relative range excursions exceed `1e-6` | FIRST GROSS |
| kt=60 | T still within NEMO's global wet range | VERIFIED BOUND |
| kt=2,601 | T/U/SSH normalized L-inf `0.3933/3.1491/2.2768` | OUTSIDE |
| completed 2,870 | legoESM T range first exceeds 20 C in the approach window (`20.8693 C`) | OUTSIDE |
| completed 2,871 | T range expands to `4.3978..25.8267 C` | AMPLIFYING |
| completed 2,876 | T increment `314,761 K`; U/SSH increments `3,392.87 m/s` / `7,992.10 m` | AMPLIFYING |
| completed 2,877 | T/S/U/V non-finite; SSH finite but `1.66e86 m` | CONFIRMED OUTSIDE |

The exact first T-only global-range transition lies between kt=61 and kt=2,601
and is **UNMEASURED**.  This does not weaken the initiating-order result: the
state as a whole first leaves NEMO at kt=2 in U/SSH, while the every-step
approach window establishes the slope-front terminal locus.

## Frozen causal arms and reconciliation

All arms are private harness hooks with reference `experimental harness
ablation; no reference model`; none is a public selector or constructible
Frankenstein configuration.

| arm | prestated discriminator | movement | label |
|---|---|---|---|
| zero tracer vertical transport from initialization | support only if failure is >=100 steps later or >3,200 and T increment halves | failure moves 2,877 -> 240 (2,637 earlier) | PLAUSIBLE; current vertical transport is necessary, full adaptive-package ownership unresolved |
| same-input vertical-transport scale at completed 2,875 | effect must be >=0.1 of observed T increment before any owner label | effect/increment `0.999609`; U and SSH effect exactly zero | PLAUSIBLE terminal T amplifier; initiating owner REFUTED |
| omit `un_adv` primary-transport time average | support only if failure moves >=100 later and kt=2,601 U/SSH errors both halve | failure unchanged at 2,877; U/SSH move `8.62e-5` / `9.54e-6` relatively | REFUTED_PRIMARY |

The primary-transport arm cites NEMO's always-live time mean at
`src/OCE/DYN/dynspg_ts.F90:509,641,843`; disabling it is localization only and
cannot be shipped.  The tracer ablation's earlier failure falls outside both
frozen terminal corridors, so it is not post-hoc promoted to confirmation.

Reconciled suspect order after the arms:

1. complete adaptive-implicit RK3 tracer **and momentum** composition --
   source-confirmed mismatch, root owner PLAUSIBLE;
2. remaining barotropic stage/substep arithmetic on partial-cell slope --
   UNMEASURED after the primary-average refutation;
3. BBL stage transport -- legoESM-side CONFIRMED live, NEMO-side PLAUSIBLE;
4. prior ~75% unowned kt=2 T term -- UNMEASURED but not scale-compatible with
   the initiating late U/SSH growth on current evidence;
5. vertical mixing/convection stabilizer -- SOURCE-EXONERATED.

The near-identical fp64/fp32 failure steps (2,877/2,879) remain **PLAUSIBLE
DETERMINISTIC STRUCTURAL DIVERGENCE**, strongly inconsistent with slow roundoff
accumulation but not promoted to `CONFIRMED`: no matched fp32 approach-window
signature was recorded in this round.

## Provenance and controls

Run root: `/data/abyssal/dbalwada/nemo-testcases-l1/stability/`.

| artifact | SHA-256 |
|---|---|
| NEMO matched `namelist_cfg` | `a98d58159ca792b49da51606eafe79f0fa32e8a284f696b11eaebeb959157e50` |
| NEMO dump binary | `4337874f60ba9c3a7c2bdcf6b7121bfff04cda2cc88ddc89e5089137ecdbccb9` |
| NEMO dump-only `stprk3.F90` | `48ae371ee0bb8730cf1c84cad501f6dc1c57715e790ecc8c1f3ae507182f6700` |
| NEMO `ocean.output` | `9edf2b400c74dd95b499071ce0f199f972f0728a278dbfa3f5e905b906516381` |
| baseline `run.json` | `4098b4b1725d361d0a66d8c7dd72802235576be95729550272133c141c14a098` |
| baseline `matched_score.json` | `0dff011511dc2c1370678168f39dd75c3b34d412d19834ce85bb640ea61ce4e0` |
| baseline `run_summary.json` | `7b1ead3d21c0107eff420415405e2815830c544bbbbd709b5204b0e533eb419d` |
| paired-step scale | `f4de0f9f13987bb9c2dfdbbfed083a82679fe7a874d78ad5d25a8f9fd64f3e54` |
| tracer-arm comparison | `0fe2732288b7c10180ec2e18b1e718af6564e643ca8378c7fa91fc3314aeea21` |
| primary-average comparison | `e623d0a0c1665590922cd36abc26cae3ed15ae96ada6e1f7900a06353ec7a52f` |

All machine artifacts stamp probe commit `c658e8d3139d`.  The baseline's model
semantics are the certified f7e044e card; the intervening model change adds only
an inert-by-default private test hook, proven by direct hook-isolation tests.
The committed controls plant a 50 C wet cell, a non-finite wet value, and a
shifted step; the gross/non-finite and time-alignment paths all go red.  The
focused CPU/fp64 run passed **13 tests**: 5 stability-probe controls plus 8
real WS-RK3 tracer-path tests:

```text
PYTHONPATH=src:packages/core:packages/grids:packages/ocean \
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q \
  tests/ocean/unit/test_nemo_overflow_stability_probe.py \
  tests/ocean/unit/test_nemo_ws_tracer_rk3.py
```

`py_compile` and `jq empty` also passed for the probe and all six pinned JSON
artifacts.

Independent adversarial review status is **UNMEASURED** for this round; no
shipping claim is made beyond the measured OUTSIDE finding and labeled owner
ledger.

## Pre-round-3 resolved-physics coverage continuation (historical)

**Historical outcome, superseded by the round-3 confirmation above:** the
then-tested register was exhausted without a confirmed initiating owner.  The
file-driven coverage pass eliminated a missing NEMO convection,
momentum-advection-form, momentum-LDF, or barotropic-selector mechanism.  The
current legoESM adaptive-momentum approximation is a **PLAUSIBLE late
amplifier**: removing only that rewrite delays failure by 1,051 steps and
suppresses the baseline terminal growth, but the arm still becomes non-finite
at step 3,928 rather than completing the oracle's 6,120 steps.  BBL is
**REFUTED_PRIMARY** at the terminal event's scale.  The kt=2 U/SSH initiating
arithmetic owner therefore remains **UNMEASURED**.

### Pre-implementation search and fail-closed coverage

Searched before adding any arm: the existing resolved-namelist parser and
file-side planted-control gate in
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_oracle_gate.py`, the
certified card builder and validator in
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py`, the canonical
barotropic filter helpers in `barotropic_common.py`, the adaptive vertical
operator in `vertical.py`, the WS-RK3 call path in
`ocean_model_latlon_cgrid.py`, and the existing private BBL hook.  Found and
reused all of them; no duplicate parser, solver, public selector, or
Frankenstein card was added.

The committed gate inventories all **108** file-side keys in the executed
`namdyn*`, `namzdf`, and `namtra*` blocks.  Exact ledger equality is required;
the planted extra file key fails as missing, and a flux-to-vector card mutation
fails before integration.  The disposition is:

| status | rows | meaning |
|---|---:|---|
| VERIFIED | 67 | live card binding or source-bound executed-off selector |
| WAIVED | 40 | dead-family operand; inventory stability only, not review |
| UNMEASURED | 1 | active `namzdf.ln_zad_aimp` arithmetic and time levels |

The matched resolved namelist SHA-256 is
`2ed353704f87a1a9fa0cb09ec33452f238cd442ccf2afa7db591acd4898d84d0`.
It has `ln_zdfevd=F`, `ln_zdfnpc=F`, and every other specific convective
closure off.  NEMO's executed dispatcher consequently selects “no specific
scheme used” (`src/OCE/ZDF/zdfphy.F90:181-197`) and the constant closure with
`rn_avm0=1e-4`, `rn_avt0=0` (`:151-179,207-215`).  Enabling EVD, convection,
or tracer diffusion would be an oracle-absent Rule-9 stabilizer and was not
armed.  Momentum is flux-form UP3 (`src/OCE/DYN/dynadv.F90:78-90`), while
`ln_dynldf_off=T` returns before viscosity arrays are allocated
(`src/OCE/LDF/ldfdyn.F90:220-239`).

The per-case external-mode composition is **VERIFIED** against the resolved
values, not a lane default: `ln_bt_fw=T`, `ln_bt_auto=T`, `nn_bt_flt=1`,
`rn_bt_alpha=0`, runtime `nn_e=3`.  NEMO's executed weight builder
(`src/OCE/DYN/dynspg_ts.F90:1041-1108,1223-1240`) and legoESM both produce
primary `[0,1,1,1]/3`, raw secondary `[3,3,2,1]`, divisor 9, and four loop
iterations.  The live zero-alpha back interpolation is the same
`0.614/0.285/0.088/0.013` arm (`:1676-1711`).  This verifies the registered
substep count/filter composition; it does not convert the still-unowned kt=2
U/SSH arithmetic residual into a match.

### Scaling-first arms and reconciliation

Both arms are private experimental harness ablations with **no reference
model**.  Public cards still run NEMO's named configuration.

| arm | frozen scale result | trajectory movement | label |
|---|---|---|---|
| suppress legoESM's current post-program adaptive momentum rewrite | same-input step-2,875 U effect `2,766.778 m/s`, ratio `0.815468` of the baseline increment | non-finite `2,877 -> 3,928`; kt=2,877 U error `621.703 -> 3.90185` (`159.336x` reduction); step-2,876 T increment `314,761 -> 0.0967489 K` (`3.25338e6x`) | **PLAUSIBLE amplifier**, not CONFIRMED |
| suppress Campin--Goosse BBL transport | effect at the registered T and U increment maxima exactly zero; global T effect only `5.733e-8` of the increment | full arm forbidden below the frozen 0.1 scale floor | **REFUTED_PRIMARY** |

Arm D's step-3,928 failure is again a slope-front redistribution event: its
last finite large increments localize near x=46--51 km and z about 530--630 m,
and volume remains closed to `1.80e-16` relative.  It cannot receive a
CONFIRMED label because it does not complete 6,120 steps.  The movement instead
shows that legoESM's current post-stage adaptive rewrite is a powerful
late-time amplifier.  NEMO does not expose a no-adaptive switch, so the arm is
not a correction and cannot ship.

The sole coverage UNMEASURED row is source-specific.  NEMO constructs its
Wicker horizontal/vertical Courant partition bottom-up at RK3 stage 3
(`src/OCE/DYN/sshwzv.F90:710-847`), then uses the explicit and implicit
transports in the resolved optimized `nn_fct_imp=1` two-step predictor
(`src/OCE/TRA/traadv.F90:220-227`;
`src/OCE/TRA/traadv_fct.F90:140-145,526-536`) and the momentum/tracer implicit
applications at their NEMO time levels.  legoESM's current local vertical-only,
post-program approximation is not that package.  The next implementation debt
is therefore the **source-exact, unbranched NEMO RK3 `ln_zad_Aimp` package**;
it must be implemented as one scheme identity, not as public micro-selectors.
Until that package runs the oracle duration and the kt=2 U/SSH term is owned,
the initiating owner remains **UNMEASURED**.

The fp64/fp32 failure steps 2,877/2,879 still support a deterministic structural
interpretation, not roundoff accumulation.  This remains labeled PLAUSIBLE,
not CONFIRMED, because no matched fp32 approach-window diagnostic was added.

### Continuation artifacts

Run root remains
`/data/abyssal/dbalwada/nemo-testcases-l1/stability/`; no new NEMO run was
needed after the coverage table.

| artifact | SHA-256 |
|---|---|
| resolved coverage JSON | `f843679436f9c174ef3809e8ade31b32118e694916b643c3fc53c1517f1d216a` |
| adaptive-momentum same-input scale | `d7ef743ae95d5f7553f5763673d53f7ec04fa0662040587872e9c9b73762e2b1` |
| adaptive-momentum arm `run.json` | `3d3a51b3a15ea96150bfa187cdcc807ee4a758e11c049b431fd0c11e113d580d` |
| adaptive-momentum matched score | `8d642cbd0385bcde27ae9baad880757e61cc92318cb03958500f30a516f60679` |
| adaptive-momentum run summary | `3162b99bf49bc155e01b1c0a110f8846bd0533b8892d49870d8415a2a6e4279f` |
| BBL same-input scale | `a87ec2fdfb5aa8b8c8805a76d6edc614bf216eabd3a56d3fbcf6804f55cfc6a0` |
| machine verdict | `6b0143b5b30ee65a87b50ca771c7aef2af33f78a1ba4eac5926d0f1bba8b5cba` |

The coverage JSON and machine verdict stamp scorer commit `b8fb77fec3ef`;
the full adaptive arm stamps runner commit `5e68669f5e8f`.  The **32 tests**
reported here were the historical two-file Round-2 subset: 12 stability-probe/
coverage/label controls plus 20 adaptive-implicit vertical-advection tests.
Rule 1e reconciliation of the later counts is exact: the reviewer's **66** was
the five pre-existing files (20 adaptive + 9 ZDF literal + 15 recipe + 12
stability probe + 10 statistical); the author's **70** added the 4 new Wicker
package literal tests.  This round adds 5 statistical controls, so the current
six-file collection is **75 = 4 + 20 + 9 + 15 + 12 + 15**.  All ran CPU/fp64 with
`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`; every state and geometry dtype in the
arm receipt is `float64`.  Direct CLI controls make the planted state and census
return `OUTSIDE`; the planted unregistered metric exits nonzero.  The state and
census plants are applied to both L64 and L32 so their signal cannot cancel into
the precision floor.

Independent adversarial review is **UNMEASURED**.  A read-only Codex review was
attempted after the commits, but the sandbox denied both WebSocket and HTTPS
network transport before a reviewer response was produced.  The subsequent
local fail-closed audit found and fixed a verdict-path vacuity: a successful
run records `first_nonfinite_completed_step=null`, which the first loader could
not classify as CONFIRMED.  The committed control now proves a finite 6,120-
step arm reaches CONFIRMED while a short null-failure run goes red.  This
self-audit is not represented as independent review.
