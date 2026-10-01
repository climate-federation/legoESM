# OVERFLOW-zps below-seabed ("phantom") velocity: receipt for the stage face-mask-rank arm

Preregistration: `nemo_testcases_l1_phantom_velocity_preregister.md` (commit
`8b4bcdb4a`, frozen before the arm; the probe was committed first, at
`75017dc58`).  Base tree for every "before" row: `c5275ae48` (the S-46
face-thickness tip), whose committed gate JSONs under
`stage3_remainder/after/gates/` are the byte-identical-protocol baseline;
"after" = the same tree plus the fix (`7d66b7f37`) and the isomorphism row
(`b23fb3cd8`).  fp64 (`PrecisionPolicy.fp64()` set BEFORE the card is built +
`JAX_ENABLE_X64=1`; every array dtype printed `float64`), CPU.  The
one-variable arm is the private `_NEMOWSRK3TestHooks.legacy_2d_stage_face_mask`
(harness only; NEMO has no such switch).

## Verdict

**CONFIRMED largest contributor to the kt>=4 `u` walk and the kt>=3 SSH walk
(S-47)** -- specifically, it OWNS the `k=25` shelf-break injection family that
becomes the L-infinity maximum from kt~4, and leaves the `k=24` shoreward
family (which carries kt=2 and kt=3) untouched at exactly `1.00x`:
NEMO multiplies every WS-RK3 stage velocity by `umask(ji,jj,jk)`
(`stprk3_stg.F90:367` for `ln_dynadv_vec .OR. lk_linssh`, `:375` for the
compiled `key_qco` branch, `:382` for the `#else`), adds the barotropic
correction as `zub(ji,jj)*umask(ji,jj,jk)` (`:444`), and masks the SAME
correction inside the advective transport `dyn_adv_up3` consumes
(`:273-274`), so both its `uu` and that transport are EXACTLY zero below the
seabed -- and `dyn_adv_up3` READS that zero rather than skipping a dry
neighbour (`dynadv_up3.F90:142-143` `zlu_uu`, `:160` `zFu`, `:166-176`
`zFu_t`).  legoESM masked the stage update, and the stage velocities the
tracer transports consume, with the 2-D `state.u_mask` broadcast over every
level (`_replace_stage_mean` and `_transport_stage` in
`ocean_model_latlon_cgrid.py`), so on a staircase a face wet at ANY level kept
the barotropic depth-mean increment at EVERY level below its own seabed --
0.228 m/s by kt=6, 78% of the card's own wet maximum -- and the deeper
neighbour's wet bottom level read it as an UP3 stencil neighbour.

With NEMO's rule enforced, the OVERFLOW `u` walk drops **4.9x at kt=10**
(`2.632503e-05 -> 5.422602e-06`), **7.7x at kt=60** (`2.506199e-04 ->
3.238532e-05`) and 5-21x at kt=4..9; the SSH walk drops **2.6x at kt=10** and
2.75x at kt=60.  At 6120 steps five of the six registered statistics move into
`WITHIN-SCHEME-SPREAD` (section 7).  46 of 50 kt<=10
rows improve or are unchanged; the four that worsen are disclosed below.
LOCK_EXCHANGE is BIT-IDENTICAL on every row, by construction: its 3-D live
u-face mask EQUALS the 2-D broadcast (measured, 0 of its points differ).

RETRACTIONS, first-class (Rule 11).  **The kt=2 rows do NOT move at all**,
which REFUTES prediction P3 as written:
the arm moves the kt=2 field by `1.73e-12` at the shelf-break bottom level,
but the row's scored MAXIMUM (`7.064252e-12`) sits at a different face and is
bit-identical.  P4 (kt=3 `u` drops `>= 2x`) is also REFUTED: kt=3 `u` is
`4.181282e-09 -> 4.181444e-09`, 0.004% WORSE.  Both disclosed, not reverted
(Rule 8).

## Rule 0 -- the two rules, quoted

```fortran
! stprk3_stg.F90:272-274   the Kmm advective transport dyn_adv_up3 consumes
zFu(ji,jj,jk) = e2u(ji,jj)*e3u(ji,jj,jk,Kmm) * ( uu(ji,jj,jk,Kmm) + zub(ji,jj)*umask(ji,jj,jk) )  ! :273

! stprk3_stg.F90:365-385   stage 1 & 2 time stepping   (CASE(1,2); stage 3's own
!                          update is in dyn_zdf and is masked there)
uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * uu(ji,jj,jk,Krhs) ) * umask(ji,jj,jk)   ! :367
uu(ji,jj,jk,Kaa) = (         ( 1._wp + r3u(ji,jj,Kbb) ) * uu(ji,jj,jk,Kbb )  &        ! :373  key_qco
   &                 + rDt * ( 1._wp + r3u(ji,jj,Kmm) ) * uu(ji,jj,jk,Krhs)  )   &
   &             /           ( 1._wp + r3u(ji,jj,Kaa) ) * umask(ji,jj,jk)             ! :375

! stprk3_stg.F90:439-446   barotropic velocity correction
zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj) ! :440
uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)                      ! :444

! dynadv_up3.F90:142-176   the k-slab stencil READS the dry neighbour's zero
zlu_uu(ji,jj) = (  ( puu(ji+1,jj,jk,Kbb) - puu(ji,jj,jk,Kbb) )  &
   &             + ( puu(ji-1,jj,jk,Kbb) - puu(ji,jj,jk,Kbb) )  ) * umask(ji,jj,jk)   ! :142-143
zFu(ji,jj)     = e2u(ji,jj) * e3u(ji,jj,jk,Kmm) * puu(ji,jj,jk,Kmm)                   ! :160
zui            = ( puu(ji,jj,jk,Kmm) + puu(ji+1,jj,jk,Kmm) )                          ! :166
IF( zui > 0 ) THEN ; zl_u = zlu_uu(ji,jj) ; ELSE ; zl_u = zlu_uu(ji+1,jj) ; ENDIF     ! :169-170
zFu_t(ji+1,jj) = (  zFu(ji,jj) + zFu(ji+1,jj)  ) * ( zui - gamma1 * zl_u )            ! :176
```

**(i) the operator that writes a nonzero `u` into a masked cell**:
`ocean_model_latlon_cgrid.py:4999-5012` (`_replace_stage_mean`, the ONE stage
ladder, called at `:5103,:5120,:5139`) applied `u_mask_3d =
state.u_mask.data[..., jnp.newaxis]` (`:4066`) -- the 2-D face mask broadcast
over levels -- to `u_in + (target_u - mean_u)[..., jnp.newaxis]`; and
`:5155-5162` (`_transport_stage`) did the same for the stage velocities the
tracer transports consume.  MEASURED fingerprint: the below-seabed column is
EXACTLY constant -- 1 unique value over all 75 dry levels, `ptp = 0.0` -- i.e.
a depth-mean broadcast, not an operator residue.  Both halves of
`_replace_stage_mean` write there and partly cancel (kt=1: stage-1 velocity
`2.737944e-02`; with the barotropic replacement ABLATED, `5.295233e-02`).

NOTE, corrected by review: NEMO has ONE rule -- *the state is masked* -- and
zero flux through a masked face is its CONSEQUENCE, not a second rule.  The
3-D RHS branch of `dyn_adv_up3` (`:205-211`) carries no `umask` at all; only
the vertically-cumulated `pUe` branch (`:193-199`) does.  The preregistration
called both "halves of NEMO's rule"; that phrasing is RETRACTED.

**(ii) the stencil that reads a masked neighbour**:
`ocean_pe_latlon_cgrid.py:4255-4258` feeds `u_core = u[:, :-1, :]` UNMASKED
into `_up3_reconstruct` (`:4078-4097`, consumed at `:4170`).  With (i) present
that reads the phantom.  Separately, `_up3_reconstruct` is algebraically
`0.5*(adv_pos + adv_neg) - (1/6)*curvature`, i.e. NEMO's `zlu_uu` with
`umask = 1`, so it also omits NEMO's `* umask` factor at the curvature's own
point.  (ii) is REAL and MEASURED but 17-2500x smaller than (i) on every row
(section 2); it is NOT landed in this round -- see "What remains open".

## 1. Census -- `phantom_velocity/{before,after}/census.json`

Below-live-seabed `max |u|`, free run.  The card's initial state is `0.0`
there and NEMO's own kt=2..5 entries are asserted `0.0` (probe controls).

| kt | before | at | wet max abs | below-seabed column | after |
|---:|---|---|---|---|---|
| 2 | `7.920607e-03` | face 21, k=25 | `9.652900e-02` | 1 unique value, ptp `0.0` | `0.0` |
| 3 | `5.294269e-02` | face 21, k=25 | `1.559111e-01` | 1 unique value, ptp `0.0` | `0.0` |
| 4 | `1.223887e-01` | face 21, k=25 | `1.907901e-01` | -- | `0.0` |
| 5 | `1.836966e-01` | face 21, k=25 | `2.336829e-01` | -- | `0.0` |
| 6 | `2.283002e-01` | face 21, k=25 | `2.930978e-01` | -- | `0.0` |

REGENERATED every step, not inherited: one step from NEMO's EXACT (clean)
entry gave `4.502208e-02 / 6.944601e-02 / 6.130782e-02 / 4.460362e-02` at
kt=2/3/4/5 before, and `0.0` at every one after.  Origin arms at kt=1 (from
rest, so every below-seabed value was written during the step): stage-1
velocity `2.737944e-02 -> 0.0`, stage-2 `1.855487e-02 -> 0.0`, barotropic
replacement ablated `5.295233e-02 -> 0.0`.  No later operator (the implicit
ZDF, the dissipation increment, the post-solve barotropic re-pin) reintroduces
it: P1 CONFIRMED, EXACTLY `0.0` at every kt and in every arm.

## 2. Scaling -- `phantom_velocity/before/scaling.json`

Model u-face index `f` == trajectory-gate face index `f-1`.  Shelf staircase,
row 1: T-cells 0..22 carry 25 wet levels and 23..26 carry 26, so u-faces 16..23
are dry at `k=25` while faces 24..27 are wet there (3000 points where the 3-D
live mask differs from the 2-D broadcast).  `dt * d(du/dt)` from legoESM's OWN
`tendencies()` called twice on the SAME stage-2 state, one variable:

| step kt | stage-2 phantom | (i) phantom, face 24 (gate 23) k=25 | (i) face 25 (gate 24) k=25 | (ii) curvature umask, face 24 k=25 | measured injection at kt+1 |
|---:|---|---|---|---|---|
| 1 | `1.8555e-02` | `-3.2086e-12` | `-9.2243e-14` | `+1.2663e-15` | `7.064e-12` (gate face 20, k=24) |
| 2 | `1.9588e-02` | `-4.1696e-10` | `-3.0844e-10` | `+2.4291e-11` | `4.181e-09` (gate face 21, k=24) |
| 3 | `4.2979e-02` | `+8.0218e-08` | `-1.5846e-08` | `+3.3445e-09` | `9.816e-08` (gate face 22, k=24) |

Ratios (i)/injection `0.45 / 0.10 / 0.82`; (ii)/(i) `4e-4 / 5.8e-2 / 4.2e-2`.
Instrument controls, all run before any number above was written: the two arms
differ only in `u` and only on dry faces (asserted); the arm is EXACTLY `0.0`
on a phantom-free state (NEMO's kt=3 entry, 16900 active u points); both
`tendencies()` calls reproducible to `0.0`; NEMO's entries asserted `0.0`
below the seabed; LOCK's 3-D live mask asserted EQUAL to its 2-D broadcast.

The preregistration read this honestly as PLAUSIBLE, not CONFIRMED: the
leverage is one level BELOW the injection maximum before kt=6.  The free-run
arm settled it -- the direct one-variable step measurement puts the movement
at model face 23, `k=24`, exactly where the injection maximum sits (stage-2
velocity `8.640469e-13`, kt=2 exit `1.729661e-12`).

## 3. The review round -- a second site, found by measurement, and INERT

An independent adversarial review (below) found, with its own measurement,
that the first commit fixed the prognostic STATE but not the array that
actually reaches `dyn_adv_up3`: `_mom_pert_ws` builds a separately
barotropically-reconciled transport velocity and hands it to `tendencies()`
as `momentum_flux_transport_velocity`, still masked by the 2-D rule.  NEMO
masks that one too, at `stprk3_stg.F90:273-274`.

CONFIRMED by measurement (probe census, `transport_operand`): the transport
handed to the horizontal UP3 at the three stages carried
`2.0709e-02 / 3.0591e-03 / 5.7655e-03 m/s` below the live seabed, and is now
EXACTLY `0.0`.  Landed (`52e7ea9f9`).

**But its numerical leverage on these cards is EXACTLY ZERO, measured.**  The
transport velocity enters the operator ONLY through
`Q_u = h_u * transport_u * ...` (`ocean_pe_latlon_cgrid.py:4244`); the
advected value is `u`, not `transport_u`.  And `h_u` -- the stage
`e3u(Kmm)` -- is EXACTLY `0.0` below the live seabed at every stage, in BOTH
arms (measured directly), so `h_u * transport_u` is `0.0` in both.  The
OVERFLOW kt<=10 trajectory rows are unchanged by this commit to every printed
digit.  So: a real faithfulness gap, correctly closed, but NOT a second
numerical owner.  Reported as such rather than as a second win (a reviewer's
finding is a hypothesis until measured).

## 4. Predictions vs outcomes

| # | prediction | outcome | verdict |
|---|---|---|---|
| P1 | below-seabed `max abs u` EXACTLY `0.0` at every kt and after every exact-entry step | `0.0` at kt=2..6 of the free run, at every exact-entry step kt=2..5, in all three origin arms -- AND, after the review round, `0.0` in the transport operand the UP3 consumes | CONFIRMED |
| P2 | kt=1 stage-1 `u` BIT-IDENTICAL | wet stage-1 field differs by EXACTLY `0.0` | CONFIRMED |
| P3 | kt=1 stage-2 and stage-3/kt=2 `u` both MOVE by `>= 3e-13`; kt=2 in `[1e-13, 1.5e-11]` | the FIELDS move (`8.640469e-13` at stage 2, `1.729661e-12` at the kt=2 exit, both at model face 23 `k=24`) but the scored MAXIMA are BIT-IDENTICAL (kt=2 `u` `7.064252e-12`), because the maximum sits at model face 21 | **REFUTED as written.**  RETRACTED: the prediction confused the field's movement with the scored row's L-infinity, at a cell the preregistration's own scaling table had already located elsewhere.  Bad prediction design, not a failed mechanism |
| P4 | kt=3 `u` `4.181282e-09` drops `>= 2x` | `4.181444e-09`, 0.004% WORSE | **REFUTED** |
| P5 | kt=4 and kt=5 `u` each drop `>= 1.3x` | `1.261475e-07 -> 2.332316e-08` (5.41x); `9.858747e-07 -> 4.624387e-08` (21.32x) | CONFIRMED |
| P6 | kt=10 `u` `>= 2x`; kt=10 SSH `>= 1.5x` | `2.632503e-05 -> 5.422602e-06` (4.85x); `1.624474e-06 -> 6.266335e-07` (2.59x) | CONFIRMED |
| P7 | kt=60 `u` drops `>= 1.3x`; T and SSH reported | `2.506199e-04 -> 3.238532e-05` (**7.74x**); SSH `3.394740e-05 -> 1.234092e-05` (2.75x); T `1.183929e-05 -> 1.183988e-05` (1.00x, marginally worse) | CONFIRMED |
| P8 | kt=2 `S` and `SSH` BIT-IDENTICAL; `T` moves `<= 3e-14` | all three unchanged (`T` `7.815970e-15`, `SSH` `1.050549e-14`), i.e. `T` moved by `0.0` | CONFIRMED |
| P9 | LOCK every row BIT-IDENTICAL | trajectory kt=1..60: **300 of 300 rows identical**; kt=1..10 50/50; stage sweep 63/63 common rows, AT-BAR | CONFIRMED |
| P10 | the `legacy_2d_stage_face_mask` arm reproduces the pre-fix rows BIT-FOR-BIT | trajectory kt=1..10: **50/50 rows equal the committed pre-fix run on BOTH cards** | CONFIRMED |
| P11 | kt=1 19-frame barotropic gate: 0 rows differ | 1014 of 1014 numeric fields identical.  Also, at kt=2/3/4 the reseeded-from-oracle-entry arm is 471/471 fields identical (only the inherited-entry arm moves, as it must) | CONFIRMED |
| P12 | DINO reach NIL | every DINO recipe resolves `momentum_time_integrator='euler'` (7 of 7) and all 11 diff hunks are inside the `rk3_ws` branch; the 5-day `nemo_dino_kamm_mlf` twin differs by `0.000000e+00` in all 33 arrays, only `producer_git_sha` differs | CONFIRMED |
| P13 | 6120-step statistics reported, no direction | five of six rows improve into `WITHIN-SCHEME-SPREAD`; see section 7 | reported |
| P14 | the new unit test FAILS on the reverted code | on a worktree at HEAD with ONLY the mask symbols reverted (hook kept, so the arm still constructs), the staircase test FAILS with `assert 0.007920606888375661 == 0.0` -- the exact pre-fix phantom; the flat-bottom control passes, as a control should | CONFIRMED |

## 5. The discriminating measurement the lead asked for

The previous round named the discriminator: *"mask the stage velocities with
the 3-D live face mask and re-read the injection table (predicted: the kt>=3
injection at faces 22-24 k=25 collapses; faces 19-21 unchanged)."*  It was not
in P1-P14; the independent review flagged that omission, and it was run.

One step from NEMO's EXACT kt-1 entry (`growth.json`), with the argmax face
and level:

| kt | before | after | ratio |
|---:|---|---|---:|
| 2 | `7.0643e-12` (gate f20, k24) | `7.0643e-12` (f20, k24) | 1.00x |
| 3 | `4.1752e-09` (f21, k24) | `4.1754e-09` (f21, k24) | 1.00x |
| 4 | `9.8164e-08` (f22, k24) | `1.9146e-08` (**f21, k24**) | 5.13x |
| 5 | `4.8675e-07` (f22, k24) | `3.5158e-08` (f23, k25) | 13.84x |
| 6 | `8.9388e-07` (f23, k25) | `2.8383e-07` (f23, k25) | 3.15x |
| 7 | `9.4709e-07` (f23, k25) | `9.5816e-07` (f23, k25) | 0.99x |
| 8 | `3.9739e-06` (f23, k25) | `1.6142e-06` (f23, k25) | 2.46x |
| 9 | `2.8347e-06` (f23, k25) | `1.5525e-06` (f23, k25) | 1.83x |
| 10 | `2.8644e-07` (f24, k25) | `1.0937e-06` (f23, k25) | 0.26x |

This is what the prediction said: the `k=25` shelf-break family collapses
(kt=4 5.1x, kt=5 13.8x, and at kt=4 the maximum MOVES OFF `k=25` back onto the
`k=24` shoreward family), while `k=24` at gate faces 20/21 is untouched
(kt=2 and kt=3 exactly 1.00x).  The kt=10 injection is 3.8x WORSE -- from an
exact kt=9 reseed the corrected trajectory is farther from the free run, so
that row is not comparable across the arms; stated, not explained away.

## 6. Trajectory (`phantom_velocity/after/gates/`)

OVERFLOW-zps `normalized_max_abs`, before -> after (ratio):

| kt | T | u | SSH |
|---:|---|---|---|
| 2 | `7.815970e-15` -> same | `7.064252e-12` -> same | `1.050549e-14` -> same |
| 3 | `5.725731e-12` -> same | `4.181282e-09` -> `4.181444e-09` (1.00x, WORSE) | `2.713463e-13` -> `1.509461e-13` (1.80x) |
| 4 | `4.110801e-11` -> same | `1.261475e-07` -> `2.332316e-08` (5.41x) | `3.228340e-10` -> `2.897012e-10` (1.11x) |
| 5 | `1.183638e-10` -> same | `9.858747e-07` -> `4.624387e-08` (21.32x) | `1.176729e-08` -> `1.179419e-08` (1.00x, WORSE) |
| 10 | `1.614452e-09` -> `1.522637e-09` (1.06x) | `2.632503e-05` -> `5.422602e-06` (4.85x) | `1.624474e-06` -> `6.266335e-07` (2.59x) |
| 20 | `1.544406e-08` -> `1.112020e-08` (1.39x) | `9.111165e-05` -> `1.671404e-05` (5.45x) | `3.945735e-06` -> `1.556077e-06` (2.54x) |
| 30 | `4.894869e-08` -> `3.294603e-08` (1.49x) | `1.155192e-04` -> `2.005403e-05` (5.76x) | `1.589129e-05` -> `6.091947e-06` (2.61x) |
| 40 | `1.418169e-07` -> `1.418330e-07` (1.00x, WORSE) | `1.438921e-04` -> `2.307055e-05` (6.24x) | `1.945216e-05` -> `7.042411e-06` (2.76x) |
| 50 | `1.633793e-06` -> `1.633901e-06` (1.00x, WORSE) | `1.879061e-04` -> `2.713034e-05` (6.93x) | `2.640717e-05` -> `9.730033e-06` (2.71x) |
| 60 | `1.183929e-05` -> `1.183988e-05` (1.00x, WORSE) | `2.506199e-04` -> `3.238532e-05` (**7.74x**) | `3.394740e-05` -> `1.234092e-05` (2.75x) |

kt<=60, 300 rows: **180 better, 82 unchanged, 38 worse**.  Every `u` row from
kt=4 and every `SSH` row from kt=3 improves; the `u` improvement GROWS with kt
(4.9x -> 7.7x from kt=10 to kt=60).  Faithful-but-worse, DISCLOSED, not
reverted (Rule 8): kt=3 `u` (1.00004x), kt=5 `ssh` (1.0023x), and the `T`/`S`
rows from kt=39 (all `<= 1.0001x`).  Reading, PLAUSIBLE: the surviving kt=2/3
injection sits at gate faces 20/21 `k=24`, where this arm has no leverage at
all (section 5 measures exactly `1.00x` there), so those rows are a different
term whose trajectory this fix reshuffles in the fifth digit.

LOCK_EXCHANGE-zco: **300 of 300 rows BIT-IDENTICAL** through kt=60.

## 7. Statistics (6120 steps, fp64 candidate + fp32 precision floor)

Before = `stage3_remainder/after/stats` (`bd0861319`); after = this tree.  Same
scorer, same registered NEMO N2 run, byte-identical protocol.  Overall status
`OUTSIDE -> OUTSIDE`; verdict counts
`{IND-AT-FLOOR 1, OUTSIDE 4, WITHIN-SPREAD 1}` ->
`{IND-AT-FLOOR 0, OUTSIDE 1, WITHIN-SPREAD 5}`.

| metric | before | after | fp32 floor before | fp32 floor after | NEMO spread | verdict before | verdict after |
|---|---:|---:|---:|---:|---:|---|---|
| `final_temperature_histogram_tv` | `0.0638199` | `0.0330051` | `0.0275579` | `0.0062277` | `0.0442310` | OUTSIDE | WITHIN-SCHEME-SPREAD |
| `final_water_mass_census` | `0.0167508` | `0.0094380` | `0.0016361` | `0.0005206` | `0.0033621` | OUTSIDE | OUTSIDE |
| `instantaneous_u_linf` | `1.71839` | `0.657229` | `2.29226` | `0.0865875` | `0.672694` | INDISTINGUISHABLE-AT-FLOOR | WITHIN-SCHEME-SPREAD |
| `plume_descent_m` | `1499.86` | `1.43044` | `0.0291712` | `0.0036066` | `1499.62` | OUTSIDE | WITHIN-SCHEME-SPREAD |
| `plume_front_km` | `117.139` | `2.96245` | `0.0112836` | `0.135531` | `121.931` | WITHIN-SCHEME-SPREAD | WITHIN-SCHEME-SPREAD |
| `temperature_linf` | `0.379253` | `0.319981` | `0.168227` | `0.0537680` | `0.356744` | OUTSIDE | WITHIN-SCHEME-SPREAD |

P13 asked for no direction, and none is claimed as a prediction -- but the
rows are not ambiguous: the plume descent distance moves from `1499.9 m`
(the whole water column: a completely different plume) to `1.43 m`, the plume
front from `117 km` to `3.0 km`, and the fp32 precision floor collapses with
them (`u_linf` floor `2.29 -> 0.087`), i.e. the two legoESM precisions no
longer decorrelate.  The 6120-step state was chaotic BECAUSE of the phantom.
Only the water-mass census remains OUTSIDE.

## 8. Stage sweep and the `slow_u` frames

OVERFLOW kt=1..2 stage sweep: every FAITHFUL row is unchanged from the
committed pre-fix run (the mask rank moves the field at model face 23 `k=24`
by `1.73e-12`, below the scored maximum at face 21).  Predicate
`S4_stage_velocity_carries_NEMOs_3d_umask` reads **MET**: below-seabed
`max abs u` at stages 1/2/3 `[0.0, 0.0, 0.0]` faithful versus
`[2.737944e-02, 1.855487e-02, 7.920607e-03]` on the 2-D arm, with wet stage-3
movement `1.7296608e-12`.  The older predicates S1 and S3 stay MET; the
pre-existing `P3_h2` NOT-MET is unchanged.  The committed baseline's ARM rows
for `legacy_stage_min_face_thickness` are stale relative to `c5275ae48` (the
previous round's own receipt records the post-review value `8.557e-11`, which
this run reproduces), so only the faithful rows are a valid before/after on
that gate.

`slow_u` from NEMO's exact entries is UNCHANGED at `8.1105e-15 / 1.5448e-11 /
5.9460e-10` (kt=2/3/4) -- as it must be: it is the step-entry `F_slow`, formed
before the stage ladder runs, and a masked level carries `h = 0` so no depth
mean can move.  The stage3-remainder round's open item 2 survives untouched.

## 9. What remains open, ranked

1. **The `k=24` shoreward family** -- the kt=2 remainder `7.064252e-12` at
   gate face 20 and the kt=3 injection `4.1754e-09` at gate face 21, both
   measured at EXACTLY `1.00x` across this arm.  It is the seed of everything
   downstream and is now the largest unexplained term.
2. **(ii), the UP3 curvature `umask`** (`dynadv_up3.F90:142-143`): MEASURED,
   `dt*d(du/dt)` `1.2663e-15 / 2.4291e-11 / 3.3444e-09` at kt=1/2/3 at model
   face 24 `k=25`, unchanged by this round (it was always measured on the
   masked state).  It is now the leading measured deviation at that cell.
   NOT landed here: it is a second variable, and NEMO's F-point form is an
   `fmask` PAIR (`:146-149`), not a centre mask, so the shared
   `_up3_reconstruct` cannot express both without a scoped change.
3. **`_replace_stage_mean`'s weights**: legoESM depth-means with the live
   `h_u_pre`/`H_u_pre`; NEMO's `zub` uses the REFERENCE `e3u_0`/`r1_hu_0`
   (`stprk3_stg.F90:440`).  The previous round measured that difference on
   `F_slow` (`<= 8.7e-14`) but NOT at this site.  UNMEASURED.
4. The non-partial-cell path keeps the 2-D rule because an
   `OceanZStarCoordinate` carries no per-level active flag.  Latent; no
   certified card takes it.  Now stated in the code.
5. The `v`-half of the change is exercised by nothing measurable: the 3-D and
   2-D `v`-masks differ at 0 points on both cards (three-row channel).
6. The reference-mesh literal seed (DINO) `e3u_0 == e3t_0` assumption
   (unchanged from the SSH-walk receipt).

## 10. Controls run

- probe instrument controls BEFORE any number was quoted: the two scaling arms
  differ only in `u` and only on dry faces (asserted); the arm is EXACTLY
  `0.0` on a phantom-free state (NEMO's kt=3 entry, 16900 active u points);
  both `tendencies()` calls reproducible to `0.0`; NEMO's own entries asserted
  `0.0` below the seabed; LOCK's 3-D live u-face mask asserted EQUAL to its
  2-D broadcast; the transport-operand section asserts BOTH that the faithful
  operand is `0.0` and that the 2-D arm still exceeds `1e-3`, so it cannot
  pass vacuously; every dtype asserted `float64`;
- `legacy_2d_stage_face_mask` arm bit-identical to the committed pre-fix run
  (trajectory kt=1..10, 50/50 rows, both cards);
- kt=1 19-frame gate 1014/1014 numeric fields identical; kt=2/3/4 frame walks
  471/471 fields identical in the reseeded-from-oracle-entry arm;
- LOCK bit-identical on 300 trajectory rows and 63 stage-sweep rows;
- `tests/ocean/unit/test_nemo_ws_stage_face_mask_rank.py` fails on the reverted
  tree with the exact pre-fix phantom and passes at HEAD;
- 9 WS-RK3 unit tests and the 35 isomorphism/duplication tests pass at HEAD;
- 2 pre-existing failures in `tests/ocean/unit/test_rk3_ws_and_mxl3.py`
  reproduce IDENTICALLY at `c5275ae48` and are not from this round.

## 11. Reviews

Codex CLI and the GLM tool are unavailable on this account, so the DUAL review
ran as two independent reviewer subagents on the `c5275ae48..b23fb3cd8` range,
one on the diff and one on the mechanism, before the receipt was written.

**Reviewer A (diff): REQUEST CHANGES -> addressed.**
1. (BLOCKING) the fix missed `_mom_pert_ws`'s transport construction, which is
   handed to the horizontal UP3 as `momentum_flux_transport_velocity` and
   still carried a below-seabed velocity.  **CONFIRMED by my own measurement**
   (`2.0709e-02 / 3.0591e-03 / 5.7655e-03 m/s`) and **LANDED** (`52e7ea9f9`),
   with NEMO's line for it (`stprk3_stg.F90:273-274`).  Its numerical leverage
   is then measured as EXACTLY ZERO (section 3) -- real gap, correctly closed,
   not a second owner.  The probe now guards the operand, not only the state.
2. (IMPORTANT) the `:377` citation is wrong in six places -- the `key_qco`
   u-branch mask is `:375`; `:377` is the v-branch's unmasked line.
   **CONFIRMED against the Fortran and corrected everywhere.**
3. (IMPORTANT) the non-partial-cell `else` keeps the 2-D rule.  CONFIRMED and
   now stated in the code; an `OceanZStarCoordinate` has no per-level active
   flag, so the 2-D mask is the only mask available there, and no certified
   card takes that branch.  Recorded as open item 4, not silently guarded.
4. (IMPORTANT) the `v`-half is exercised by nothing (0 differing points on
   both cards).  CONFIRMED; open item 5.
5. (MINOR) `3-D subset of 2-D` is assumed, never asserted.  Agreed; adding a
   hard error is a behaviour choice, so it is listed under ASK, not taken.
6. (MINOR) a stale self-referential line citation.  Fixed by moving the mask
   pair to one place.
7. (MINOR) S-47 should list `_mom_pert_ws`.  Done, under its own NEMO arm
   (`:273-274`), distinct from the state-update arm (`:444`).
8. (NOTE) binding, shapes, dtype, AD/JIT/pytree, hook one-variable-ness and
   test non-vacuity all check out.

**Reviewer B (mechanism): SHIP-WITH-NOTES.**
1. NEMO reading CONFIRMED line by line, and strengthened: `domain.F90:145`
   builds `hu_0 = SUM(e3u_0*umask)` (masked) while `stprk3_stg.F90:440` sums
   `e3u_0*uu` UNMASKED and divides by it -- NEMO's own barotropic diagnostic
   is only self-consistent if `uu` is exactly zero below the floor.  Added
   here; it was not in the preregistration.
2. `:367/:375` are `CASE(1,2)`; stage 3's own update is masked in `dyn_zdf`.
   The "EVERY stage velocity" phrasing over-cited.  CORRECTED.
3. "both halves are in the source" REFUTED: NEMO has ONE rule (mask the
   state); zero flux is a consequence, and the 3-D RHS branch carries no
   `umask`.  RETRACTED in section 0.
4. Ownership overstated: the evidence supports "owns the `k=25` seaward-face
   family, which becomes the maximum from kt~4", not "owns the walk".
   ADOPTED -- the verdict is rewritten and section 5 shows the `k=24` family
   at exactly `1.00x`.
5. The discriminator named by the previous round was missing from P1-P14.
   CORRECT, and it was run: section 5.  It confirms the predicted collapse.
6. kt=2 not moving and kt=3 moving `+1.6e-13` are CONSISTENT, not falsifying
   (the kt=3 move is only possible if the kt=2 state changed).  Agreed;
   recorded as a prediction-design retraction, not a mechanism failure.
7. P9 (LOCK) is near-vacuous as a mechanism test.  Agreed and stated: it tests
   inertness where the geometry makes the two rules coincide.
8. Ranked next steps adopted verbatim as open items 1-3, including the
   previously UNMEASURED `_replace_stage_mean` weight question.

The two reviewers agreed on the citation corrections and on the completeness
gap; where A called the missed site BLOCKING for impact, the measurement in
section 3 settles it as inert.  Not averaged -- measured.

## 12. Artifacts (`/data/abyssal/dbalwada/nemo-testcases-l1/phantom_velocity/`)

| artifact | sha256 |
|---|---|
| `after/gates/frame_gate_kt1.json` | `b9878e1a4503cfefa3748ecebae7213dbe0f55bdbf25df8bb63598b965eec161` |
| `after/gates/frame_walk_kt2.json` | `d312b4f9e3d4e73625825667927005e030bd22a2cdb8dfb741e0d20f21fe7423` |
| `after/gates/frame_walk_kt3.json` | `aa419f01ea4bda00857f36e3c5acb5b8692602b788ee04529110c0b2a7d6c019` |
| `after/gates/frame_walk_kt4.json` | `08c6c4b7a1e259da8829cb21f78c7029c9f7bd8a40fbfe5434573036024f8d51` |
| `after/gates/lock_stage_sweep_kt2.json` | `ba2cedced7fdf99c61206ef3485a940bc2ead504bbd446c1d6bec69633afa976` |
| `after/gates/lock_trajectory_kt10.json` | `bcc2c33bf195146d44d457daf816abbfbdf92bdeb5837dde6e49e7d7c6ff221b` |
| `after/gates/lock_trajectory_kt10_legacy_2d_mask_arm.json` | `023f9a87babfd0b307a6830a7d9577600524d7d496b98de7221292d329670c95` |
| `after/gates/lock_trajectory_kt60.json` | `67520758594c4d37136cf2de296668f6cd252d4dae6a18b6a5f9a7de37154848` |
| `after/gates/overflow_stage_sweep_kt2.json` | `e185e49cd71e3fc893c4c005c14962e8006d7f1ccbc5ab9ad9e781cf194f26ad` |
| `after/gates/overflow_trajectory_kt10.json` | `89f69a11ffeaa4261f9d3cce6b92db0d9cc646dd3c35b4434071465743d1bf4b` |
| `after/gates/overflow_trajectory_kt10_legacy_2d_mask_arm.json` | `7aa43dea3dffd44f5960374a196c76c54d5f25432b6f97e3d84fa59ae6531e40` |
| `after/gates/overflow_trajectory_kt60.json` | `35dfb2b0f33d82bf79a4e8484faa331eba2ec44cd7c80156a7313f9740d06da3` |
| `after/growth/growth.json` | `bc49172527a0321e7139579ae264a9b98e5b3d0036317af85c093f6e338bc79b` |
| `after/growth/slow.json` | `3a8dba73f313808b7fb9d07fdae2e6538b6e360cd71680e47042a012178d545b` |
| `after/probe/census.json` | `b4bd891d8778b8d191a8834a34d50b2c9f7829e072123488dfd9fe0fcc62e666` |
| `after/probe/scaling.json` | `03542fa22aee414e937b1efde40bf24722751d5c513b3a6dcb70ac4024a51324` |
| `after/stats/legoesm/overflow_zps/fp32/metadata.json` | `3e4e27dfa80a3fe1ed7e7f8cbc47a42774db1438741b4b6ba7a3a9f9320c29ca` |
| `after/stats/legoesm/overflow_zps/fp32/states.npz` | `aca1da3bdc9b2478895e5fb1c4767a9ef90188a104d895dab1b5fb02b4e0d229` |
| `after/stats/legoesm/overflow_zps/fp64/metadata.json` | `b0140811034a44e0568b28064bbea2c2d7d24fd345e37180d268909ec6cc1125` |
| `after/stats/legoesm/overflow_zps/fp64/states.npz` | `b0a20735446d1c03621984ae824d601c71471e76b1f2c81dd8c1cc25b8e73cf4` |
| `after/stats/overflow_statistics.json` | `92f2a0e7ff51f370cfdce4687b8860b09256e9e09246e50f0e83676974a4496e` |
| `before/census.json` | `6b83e28f2071436cb84c411d806b8fc6a987dad79e3599cad68dd0427cdfd505` |
| `before/scaling.json` | `bdd5df02f05ca663581e943f31a441a0b299b53f096f1485c8d1be10d832e2fd` |
| `dino_reach/twin_BASE_d5.npz` | `405a7f33427f4e730078b2b565c10f3e0dadc6dddb8151275bf61204cb4434ea` |
| `dino_reach/twin_HEAD_d5.npz` | `88d212312f09d9e1302473b52c2b45982e3b952127691f7e2da153ac9cf4fac4` |

legoESM commits on `fidelity/overflow-phantom-below-seabed`, cut from
`c5275ae48`: `75017dc58` (probe), `8b4bcdb4a` (preregistration, frozen before
the arm), `7d66b7f37` (fix), `b23fb3cd8` (isomorphism row S-47), `52e7ea9f9`
(review round: the advective-transport site, the citation corrections, the
operand guard), plus this receipt.  Not pushed.

## 13. Choices

ASKED: none needed -- every landed change is NEMO's own rule, transcribed
(Rule 9: masking is the oracle's behaviour, not a stabilizer).

UNASKED, each offered for revert in the same breath:
* none that change model behaviour.  The only judgement calls were (a) NOT
  landing (ii), the UP3 curvature `umask`, in the same round -- it is a second
  variable, 17-2500x smaller, and its NEMO F-point form is an `fmask` pair;
  and (b) NOT adding a hard error asserting the 3-D face mask is a subset of
  the 2-D one (reviewer A-5) -- turning a tolerated condition into a hard
  error is exactly the kind of choice that must be asked first.
Both are listed as open items rather than taken.
