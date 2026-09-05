# OVERFLOW-zps kt=2 stage-3 `u` remainder: receipt for the face-thickness arm

Preregistration: `nemo_testcases_l1_stage3_remainder_preregister.md` (commit
`f8247f9a3`, frozen before the arm).  Base tree for every "before" row:
`032bedd3a` (branch tip: UP3 selector fix, RK3 ladder collapse, S-21 stage-1
transport, barotropic seed fix, face thickness, stage composition, DINO
divisor), re-measured by the committed probes at `8e75304ea`/`fcd0d6057`
(probe-only commits on the same numerics); "after" = the same tree plus the
fix (`60d0c5420`) and the map row (`bd0861319`).  fp64
(`PrecisionPolicy.fp64()` set before the card is built + `JAX_ENABLE_X64=1`;
every array dtype printed `float64`), CPU.  The one-variable arm is the
private `_NEMOWSRK3TestHooks.legacy_hadv_min_face_thickness` (harness only;
NEMO has no such switch); its legacy side reproduces the pre-fix tree
BIT-FOR-BIT (stage sweep kt=1..2 both cards, trajectory kt=1..10 both cards).

## Verdict

**(A) The kt=2 remainder does NOT own the kt>=3 `u` walk.**  From NEMO's
EXACT kt=2 entry, legoESM reproduces its own free trajectory to 1.6% at
every kt>=3 (same face pattern, corr >= 0.9994); adding the measured kt=2
remainder field changes nothing beyond kt=3.  The per-step injection from an
exact entry GROWS with the flow (`4.6e-10, 8.4e-9, 9.9e-8, 4.9e-7, 8.9e-7`
at kt=1..5) and walks with the front onto the shelf-break faces (21 -> 22 ->
23), always at the BOTTOM level (k=24/25): a state-dependent operator error,
not a propagated seed.

**(B) CONFIRMED owner of the kt=2 stage-3 remainder, the stage-2 residual and
the front-face `slow_u` debt (S-47; renumbered from S-46 when this round merged with the UP3 SELECTOR round, which took S-46):** the flux-form momentum advection
inside `tendencies()` built its own face thickness as the MIN of the two
STRETCHED T thicknesses (`ocean_pe_latlon_cgrid.py:1221,1364`, consumed at
`:4504`), where NEMO's `dyn_adv_up3` consumes `e3u(Kmm) = e3u_0*(1+r3u(Kmm))`
in `zFu` and as the divisor (`dynadv_up3.F90:160,205-207`;
`domzgr_substitute.h90:127`; `domqco.F90:219-220`; `stprk3_stg.F90:273`).
The difference, `-0.5*|ssh_W - ssh_E|/hu_0`, cancels at the front face in the
T-point flux divergence and survives at the flanking faces 19/21 -- where
every one of the three symptoms sat.  Landed as the fourth consumer of the
one shared kernel (`_nemo_ws_qco_stage_faces` ->
`vertical.nemo_qco_live_face_geometry_cgrid`); no public selector.

**REFUTED (P6, P7): the kt>=4 `u` walk and the kt>=8 SSH walk have ANOTHER
owner.**  OVERFLOW kt=10 `u` `2.629597e-05 -> 2.632503e-05` (1.00x, 0.1%
worse), SSH `1.473509e-06 -> 1.624474e-06` (0.91x); the remaining per-step
injection sits at the shelf-break faces 22-24, bottom levels.  Disclosed,
not reverted (Rule 8).  Lead measured, not owned: see "What remains open".

## Predictions vs outcomes

| # | prediction | outcome | verdict |
|---|---|---|---|
| P1 | OVERFLOW kt=2 `u` `4.551736e-10 -> <= 1.0e-11` | `7.064252e-12` (64x) | CONFIRMED |
| P2 | stage-2 `u` `9.433404e-11 -> <= 3.0e-12`; stage 1 bit-identical | `1.566344e-12` (60x); stage 1 `6.522560e-15` -> same | CONFIRMED |
| P3 | kt=2 `S`, `SSH` bit-identical; `T` moves `<= 3e-14` | `SSH` `1.050549e-14` -> same, `S` same; `T` `1.119105e-14 -> 7.815970e-15` | CONFIRMED |
| P4 | legacy arm reproduces the pre-fix rows bit-for-bit | stage sweep: all legacy rows equal the before faithful rows (both cards); trajectory kt=1..10: `legacy == before` on every row, both cards | CONFIRMED |
| P5 | `slow_u` exact-reseed kt=2/3/4 `1.296e-10 / 9.63e-10 / 1.90e-9 -> <= 1e-13 / <= 3e-11 / <= 7e-10` | `8.111e-15 / 1.545e-11 / 5.946e-10` (16000x / 62x / 3.2x); the kt=3/4 leftovers sit at face 22 (`1.5e-11`, `5.9e-10`), faces 19-21 at `<= 3e-16` | CONFIRMED |
| P6 | kt=3 `u` drops `>= 5x`; kt=10 `u` `>= 10x` | kt=3 `8.843722e-09 -> 4.181282e-09` (2.1x); kt=10 `2.629597e-05 -> 2.632503e-05` (1.00x) | **REFUTED** (kt=10 `< 2x`) |
| P7 | kt=10 `SSH`, `T` drop `>= 3x` | SSH `1.473509e-06 -> 1.624474e-06` (0.91x); T `2.128213e-09 -> 1.614452e-09` (1.32x).  kt=3/4/5 SSH: `2.440693e-09 -> 2.713463e-13` (8995x), `2.262702e-08 -> 3.228340e-10` (70x), `8.313770e-08 -> 1.176729e-08` (7.1x) | **REFUTED** at kt=10 (the front-face injection was this operand; the kt>=6 walk is not) |
| P8 | LOCK every row bit-identical | stage sweep kt=1..2: 54/54 rows bit-identical; trajectory rows MOVE from kt=3, ALL improving, 0 rows worse (u kt=3 `7.267792e-15 -> 4.423545e-17`, now AT-BAR; SSH kt=10 `3.574534e-13 -> 2.233456e-17`; kt=60 T `8.755466e-10 -> 7.022415e-10`, u `7.496604e-08 -> 6.096440e-08`, SSH `1.609001e-08 -> 9.877738e-12`) | REFUTED in the good direction: the premise "`1 + r3u == 1.0`" was wrong for legoESM's OWN eta (`~1e-18` from kt=3), whose min-vs-mean differs at the last bit |
| P9 | 6120-step statistics, no direction | see "Statistics" | reported |
| P10 | unit test fails on the reverted code | `tests/ocean/unit/test_nemo_ws_hadv_face_thickness.py`: 2 passed at HEAD; on a worktree at `f8247f9a3` (pre-fix) both FAIL | CONFIRMED |

## (A) Growth attribution (before -> after, `stage3_remainder/{before,after}/growth.json`)

Normalized L-inf of `u` on the entering state; "injection" = one step from
NEMO's EXACT kt-1 entry:

| kt | free before -> after | injection before -> after (after: argmax face, k) | exact kt=2 entry, stepped (after; ratio free/arm) |
|---:|---|---|---|
| 2 | `4.552e-10 -> 7.064e-12` | `4.552e-10 -> 7.064e-12` (20, 24) | -- |
| 3 | `8.844e-09 -> 4.181e-09` | `8.388e-09 -> 4.175e-09` (21, 24) | `4.175e-09` (1.001) |
| 4 | `1.271e-07 -> 1.261e-07` | `9.873e-08 -> 9.816e-08` (22, 24) | `1.250e-07` (1.009) |
| 5 | `9.918e-07 -> 9.859e-07` | `4.890e-07 -> 4.868e-07` (22, 24) | `9.816e-07` (1.004) |
| 6 | `3.427e-06 -> 3.430e-06` | `8.938e-07 -> 8.939e-07` (23, 25) | `3.482e-06` (0.985) |
| 10 | `2.630e-05 -> 2.633e-05` | `4.408e-07 -> 2.864e-07` (24, 25) | `2.676e-05` (0.984) |

Before the fix the exact-entry arm and the exact-entry-plus-remainder arm
agree with each other to 5 digits from kt=4 and with the free run to 1.6%.
That 1.6% is a reseed artifact, MEASURED: the reseed keeps the card's zeros on
gate-INACTIVE faces, while legoESM's own kt=2 state carries `u = -7.92e-3 m/s`
at the front face (lego face 21) at level k=25 -- BELOW the shelf floor
(25 wet levels), where NEMO's `umask` zeroes it (max NEMO `u` on those faces
`0.0`).  The WS stage program masks its stage velocities with the 2-D
`u_mask`, not the 3-D live face mask.  On the active faces the two states
differ by `7.1e-12` only.

## (B) Candidate table (before, `stage3_remainder/before/candidates.json`)

Target `R` = e3u_0-baroclinic `u3_lego - u3_nemo` at kt=1 stage 3:
`4.5517e-10` at face 21 k=24, depth-mean part `4.8e-15`, faces 19/21
anticorrelated (`-0.9996`), face 20 `1.0e-11`, near-linear in `gdept`
(r^2 0.97/0.96).  Transcription controls: `hpg_sco` vs NEMO's rest RHS
`1.1e-15`; `wzv` vs the dumped `zFw` `9.1e-13`; NEMO's own stage-3 closure
`4.6e-16`.  `E = dt*(1+r3u(Kmm))/(1+r3u(Kaa))*bc(RHS_lego - RHS_nemo)`,
regressed on `R`:

| candidate | max abs E | faces 19/20/21 | corr | slope | max abs R - E |
|---|---:|---|---:|---:|---:|
| **X1 momentum-advection thickness (legoESM's own operator, h_u the only variable)** | `4.612e-10` | `2.04e-10 / 3.1e-12 / 4.61e-10` | **`0.99979`** | **`0.9891`** | **`7.1e-12`** |
| X1t (the `dynadv_up3` transcription with `h_min`) | `4.552e-10` | `2.06e-10 / 1.0e-11 / 4.55e-10` | `1.00000` | `1.00004` | `1.7e-12` |
| X1 at stage 2 vs the stage-2 residual `9.433e-11` | `9.568e-11` | -- | `0.99978` | `0.9888` | `1.6e-12` |
| X2 HPG operator | `9.0e-16` | | | | unchanged |
| X3 vertical UP3 operator | `5.0e-21` | | | | unchanged |
| X4 qco stage ratios | `1.4e-21` | | | | unchanged |
| X5 stage transport triplet | `2.0e-20` | | | | unchanged |
| X6 implicit ZDF solver | `5.4e-18` | | | | unchanged |
| X7 stage-mean weights | `0.0` | | | | -- |

Same operator at the step entry vs the exact-entry `slow_u` residual
(`before/slow.json`): kt=2 corr `1.00000` slope `1.0000` (residual `8e-15`),
kt=3 `0.99991 / 0.9997` (`1.5e-11`), kt=4 `0.9735 / 0.9989` (`5.9e-10`, off
the front faces).

After the fix (`after/candidates.json`): `R` `7.064e-12` at face 20 k=24
(faces 19/20/21/22: `2.0e-12 / 7.1e-12 / 6.1e-12 / 2.7e-12`), the RHS
closure remainder `D_l` `4.552e-11 -> 7.070e-13 m/s^2`; X1 no longer fits
(corr `-0.47`).

## `slow_u` and the external-solve frames (substep 1; `frames/`, before = `ssh_walk/seed_fix/`)

| kt | arm | u_entry | transport_u | eta_continuity | pgf_u | slow_u | u_exit | eta_exit |
|---:|---|---|---|---|---|---|---|---|
| 2 | reseeded exact | `6.9e-18` -> same | `1.6e-16` -> same | `1.7e-18` -> same | `1.4e-20` -> same | `1.296e-10 -> 8.1e-15` | `4.32e-10 -> 2.7e-14` | `1.7e-18` -> same |
| 2 | inherited | `4.77e-15 -> 4.78e-15` | `1.06e-13 -> 1.06e-13` | `1.78e-14` | `3.5e-16` | `1.296e-10 -> 8.1e-15` | `4.32e-10 -> 2.7e-14` | `1.78e-14` |
| 3 | reseeded exact | `3.5e-18` -> same | `1.3e-16` -> same | `1.4e-17` -> same | `1.1e-19` -> same | `9.63e-10 -> 1.55e-11` | `3.21e-9 -> 5.15e-11` | `1.4e-17` -> same |
| 3 | inherited | `1.10e-9 -> 6.4e-14` | `2.08e-8 -> 1.2e-12` | `4.13e-9 -> 4.6e-13` | `8.1e-11 -> 6.9e-15` | `9.63e-10 -> 1.54e-11` | `4.04e-9 -> 5.2e-11` | `4.13e-9 -> 4.6e-13` |
| 4 | reseeded exact | `2.1e-17` -> same | `4.8e-16` -> same | `5.6e-17` -> same | `8.7e-19` -> same | `1.90e-9 -> 5.95e-10` | `6.34e-9 -> 1.98e-9` | `5.6e-17` -> same |
| 4 | inherited | `8.42e-9 -> 1.30e-10` | `1.88e-7 -> 2.9e-9` | `3.46e-8 -> 5.5e-10` | `6.7e-10 -> 1.0e-11` | `1.90e-9 -> 5.95e-10` | `1.25e-8 -> 2.1e-9` | `3.46e-8 -> 5.5e-10` |

`slow_u` stays the first DEBT frame at kt=2..4 from an exact entry, now at
face 22 (kt=3 `1.5e-11`, kt=4 `5.9e-10`; faces 19-21 `<= 3e-16`).  The kt=1
19-frame gate: 0/152 production+legacy rows differ from the before run
(`ssh_walk/head_kt1`), as predicted (eta(Kbb) = 0).

## Trajectory before -> after

OVERFLOW-zps (`gates/overflow_trajectory_kt{10,60}.json`):

| kt | T | u | SSH |
|---:|---|---|---|
| 2 | `1.119105e-14 -> 7.815970e-15` | `4.551736e-10 -> 7.064252e-12` | `1.050549e-14` -> same |
| 3 | `6.832934e-12 -> 5.725731e-12` | `8.843722e-09 -> 4.181282e-09` | `2.440693e-09 -> 2.713463e-13` |
| 4 | `5.452812e-11 -> 4.110801e-11` | `1.270969e-07 -> 1.261475e-07` | `2.262702e-08 -> 3.228340e-10` |
| 5 | `1.787193e-10 -> 1.183638e-10` | `9.918410e-07 -> 9.858747e-07` | `8.313770e-08 -> 1.176729e-08` |
| 10 | `2.128213e-09 -> 1.614452e-09` | `2.629597e-05 -> 2.632503e-05` | `1.473509e-06 -> 1.624474e-06` |
| 20 | `2.476749e-08 -> 1.544406e-08` | `9.101258e-05 -> 9.111165e-05` | `3.798106e-06 -> 3.945735e-06` |
| 40 | `5.525444e-07 -> 1.418169e-07` | `1.437992e-04 -> 1.438921e-04` | `1.930713e-05 -> 1.945216e-05` |
| 60 | `1.184427e-05 -> 1.183929e-05` | `2.505926e-04 -> 2.506199e-04` | `3.315149e-05 -> 3.394740e-05` |

LOCK_EXCHANGE-zco (`gates/lock_trajectory_kt{10,60}.json`): kt=2
`0.0 / 2.276825e-17 / 0.0` unchanged; kt=3 u `7.267792e-15 -> 4.423545e-17`;
kt=10 `1.693460e-14 / 1.314215e-11 -> 1.137195e-11 / 3.574534e-13 ->
2.233456e-17`; kt=60 `8.755466e-10 -> 7.022415e-10 / 7.496604e-08 ->
6.096440e-08 / 1.609001e-08 -> 9.877738e-12`.  0 of 180 LOCK rows worse.

Faithful-but-worse, disclosed (Rule 8): 94 of 180 OVERFLOW rows are worse
after the fix -- the `u` rows from kt=6 by `<= 0.3%` (kt=6 `3.427e-06 ->
3.430e-06`, kt=10 `2.630e-05 -> 2.633e-05`, kt=60 `2.505926e-04 ->
2.506199e-04`) and the SSH rows from kt=8 by `<= 10%` (kt=8 `9.945e-07 ->
1.052e-06`, kt=10 `1.474e-06 -> 1.624e-06`, kt=60 `3.315e-05 -> 3.395e-05`);
T improves at every kt <= 40 (kt=30 `2.9x`, kt=40 `3.9x`).  The kt=2..5 rows
improve 1.0-9000x, so no operand cancelled the fix; the reading is trajectory
reshuffling of the separately-owned bottom-level injection (open item 1).
NOT reverted.

## Stage sweep kt=1..2 (both cards, `gates/*_stage_sweep_kt2.json`)

OVERFLOW: stage 1 `6.522560e-15` -> same; stage 2 `9.433404e-11 ->
1.566344e-12`; stage 3 / kt=2 `4.551736e-10 -> 7.064252e-12`; kt=2 T
`1.119105e-14 -> 7.815970e-15`; the `legacy_hadv_min_face_thickness` arm
equals every before row; predicates S1 (selector) and S3 (this round) MET.
LOCK: 54/54 rows bit-identical, AT-BAR.

## Statistics (6120 steps, OVERFLOW-zps, fp64 candidate + fp32 floor, `stats/overflow_statistics.json`)

Before = `ssh_walk/stats_after` (the S-21 + seed tree, `f9f027525`); after =
`stage3_remainder/after/stats` at `bd0861319`; both `nemo_testcase_full_statistics.py`
(fp64 candidate + fp32 precision floor) scored against the registered NEMO N2 run;
overall status `OUTSIDE -> OUTSIDE` (verdict counts after: {'INDISTINGUISHABLE-AT-FLOOR': 1, 'OUTSIDE': 4, 'WITHIN-SCHEME-SPREAD': 1}).

| metric | before | after | fp32 floor before | fp32 floor after | NEMO spread | verdict before | verdict after |
|---|---:|---:|---:|---:|---:|---|---|
| final_temperature_histogram_tv | `0.0612665` | `0.0638199` | `0.0205036` | `0.0275579` | `0.044231` | OUTSIDE | OUTSIDE |
| final_water_mass_census | `0.0168986` | `0.0167508` | `0.00490892` | `0.00163606` | `0.00336209` | OUTSIDE | OUTSIDE |
| instantaneous_u_linf | `2.0436` | `1.71839` | `2.50075` | `2.29226` | `0.672694` | INDISTINGUISHABLE-AT-FLOOR | INDISTINGUISHABLE-AT-FLOOR |
| plume_descent_m | `1499.79` | `1499.86` | `0.175234` | `0.0291712` | `1499.62` | OUTSIDE | OUTSIDE |
| plume_front_km | `117.125` | `117.139` | `0.0412242` | `0.0112836` | `121.931` | WITHIN-SCHEME-SPREAD | WITHIN-SCHEME-SPREAD |
| temperature_linf | `0.379252` | `0.379253` | `0.0894013` | `0.168227` | `0.356744` | OUTSIDE | OUTSIDE |

Direction-free, as preregistered (P9): histogram `+4%` (worse), census `-1%`,
`u_linf` `2.04 -> 1.72` (both arms INDISTINGUISHABLE-AT-FLOOR; the fp32 floor itself
`2.50 -> 2.29`), plume rows and `T_linf` within their two-valued / floor readings.
The 6120-step state is chaotic (the two legoESM precisions decorrelate to `2.3 m/s`
in `u_linf`), so no row here can confirm or refute a `1e-11` fix; reported, not
interpreted.

## What remains open, ranked

1. **The kt>=4 `u` walk and the kt>=8 SSH walk (OVERFLOW kt=10 `u` `2.63e-5`,
   SSH `1.62e-6`)**: a per-step injection that grows with the flow and lives at
   the BOTTOM level of the front / shelf-break faces (after the fix: kt=3
   `4.2e-9` at face 21 k=24; kt=4 `9.8e-8` at face 22 k=24; kt>=6 at face 23
   k=25 -- a partial-cell bottom level).  MEASURED lead: legoESM's own kt=2
   state carries `u = -7.92e-3 m/s` at the front face one level BELOW the
   shelf floor (k=25, gate-inactive, NEMO `0.0`), because the WS stage program
   masks with the 2-D `u_mask` (`omlc` `u_mask_3d = state.u_mask.data[...,
   None]`) rather than the 3-D live face mask; and the reseeded arms, which
   zero it, differ from the free run by 1.6% from kt=4, so it is NOT inert.
   Mechanism CONFIRMED by reading (review round, both reviewers + author):
   NOT the vertical operators (the implicit ZDF zeroes the below-seabed
   interfaces through `_act_u3`, the vertical UP3 is `face_active`-masked,
   every thickness-weighted sum sees `h_partial = 0`) but the HORIZONTAL UP3
   stencil -- `_bc_horizontal_momentum_advection_flux_form` reconstructs the
   T-point face value from `u_core = u[:, :-1, :]` UNMASKED, so the wet bottom
   level of the shelf-break faces reads the dry front face's phantom as a
   stencil neighbour with a nonzero transport from its wet side, where NEMO's
   `uu(Kaa)` is `umask`'d at `stprk3_stg.F90:367,375,444`.  Discriminating
   one-variable arm: mask the stage velocities with the 3-D live face mask
   (equivalently zero `u` below the live seabed after every stage) and re-read
   the injection table (predicted: the kt>=3 injection at faces 22-24 k=24/25
   collapses; faces 19-21 unchanged).  UNMEASURED; needs its own
   preregistration.  ASK before landing: it changes carried state on masked
   cells.
2. Remaining kt=2 `R` `7.06e-12` at face 20 k=24 and the exact-entry `slow_u`
   `1.5e-11` (kt=3) / `5.9e-10` (kt=4) at face 22: same bottom-level family as
   item 1 (the kt=4 replay left exactly `5.9e-10` off the front).
3. The reference-mesh literal seed (DINO) `e3u_0 == e3t_0` assumption
   (unchanged from the SSH-walk receipt, item 3).

## Controls run

- transcription controls before any candidate row (`hpg_sco` rest `1.1e-15`,
  `wzv` `9.1e-13`, NEMO stage-3 closure `4.6e-16`);
- legacy arm bit-identical to the pre-fix tree (stage sweep both cards,
  trajectory kt=1..10 both cards);
- kt=1 19-frame gate 0/152 rows differ from `ssh_walk/head_kt1`;
- `tests/ocean/unit/test_nemo_ws_hadv_face_thickness.py` fails on the pre-fix
  tree (both tests) and passes at HEAD; the probe's own unit tests
  (`tests/ocean/fidelity/test_nemo_testcase_stage3_remainder_probe.py`, 5)
  exercise the transcriptions on synthetic rows with planted moves;
- isomorphism tripwire (`test_no_scheme_duplication.py -k isomorphism`) green
  with S-47 in the map and the registry;
- 71 tests across the WS-RK3 unit files and the three gates' test files pass
  at HEAD (`test_nemo_ws_qco_stage_faces`, `test_nemo_ws_stage1_transport`,
  `test_nemo_ws_tracer_rk3`, `test_flux_form_momentum`, the stage-sweep /
  trajectory / barotropic gate tests, the probe tests, the new test).

## Reviews

Codex CLI and the GLM tool are unavailable on this account, so the dual
review ran as two independent reviewer subagents on the full
`032bedd3a..bd0861319` range (one on the diff contract, one on the
mechanism), after the arm.  Both verdicts and the disposition of every
finding:

**Reviewer A (diff): REQUEST CHANGES -> addressed.**
1. (Important) `F_slow` still depth-means with the min-rule `h_u_pre` while
   the advection now divides by `e3u(Kbb)`; claimed a `4e-4`/level weighting
   error.  **REFUTED by measurement** (the per-face stretch factor cancels in
   a normalised mean; only a partial-cell face whose min switches columns
   between levels can differ): on NEMO's exact kt=2/3/4 entries
   `max|F_slow[h_u_pre] - F_slow[e3u_0]|` = `8.7e-19 / 2.4e-16 / 8.7e-14`
   (face 23; identical against `e3u(Kbb)` weights), while the remaining
   exact-entry `slow_u` residual is `8.1e-15 / 1.5e-11 / 5.9e-10` at face
   22 -- four to seven orders apart and on a different face.  Not changed;
   unifying the weights onto the kernel's pair for isomorphism's sake is a
   choice, listed under ASK.
2. (Important) the kwarg is consumed only under `momentum_advection ==
   "flux_form"`, so a vector-invariant `rk3_ws` card would silently get no
   effect.  **REFUTED**: the constructor refuses `rk3_ws` without the coupled
   `flux_form/upwind3/nemo_up3` program (`ocean_model_latlon_cgrid.py`
   `_validate_config`, "NEMO rk3_ws requires the coupled flux_form/upwind3/
   nemo_up3 momentum program"), so the path is fail-closed already.
3. (Minor) with `adaptive_implicit_vertadv=False` (a legal namelist) the
   kernel's explicit vertical advection still divides by the min-rule `h_u`
   while the horizontal uses `e3u(Kmm)`.  CONFIRMED by reading, dormant on
   every certified card (`nemo_testcase_recipe.py` sets Aimp true); recorded
   as debt, not changed (an unmeasurable path; ASK).
4. (Minor) `_stage_face_thickness` read the pair off the stage transport's
   `geom[4]`, which the OLDER `legacy_stage_min_face_thickness` transport arm
   also flips, so that arm had silently become two-variable.  **FIXED** in
   this receipt's commit: the pair is now built from the stage ssh through
   `_nemo_ws_qco_stage_faces` at every site; the faithful kt=2 rows on both
   cards reproduce the after-gate numbers bit-for-bit (OVERFLOW `u`
   `7.064251961175216e-12`, `T` `7.815970093361103e-15`, `ssh`
   `1.050548537051554e-14`; LOCK `2.276824562219559e-17 / 0 / 0`), 7 WS unit
   tests pass, and the older transport arm now reads kt=2 `u` `8.557e-11`,
   `T` `1.274e-08` (one variable again).  Also noted: the S1 selector
   predicate's frozen `2.598798e-07` is now measured on top of the new
   thickness (`2.598859e-07`, inside its 1% band) -- MET by tolerance, stated.
5. (Note) the "rest identical" assertion of the unit test is a smoke control
   (at eta=0 the two rules coincide by construction), not evidence; the live
   tilted-entry assertion is the non-vacuous half.  Agreed.

**Reviewer B (mechanism): SHIP-WITH-NOTES.**
1. NEMO reading CONFIRMED line by line (`dynadv_up3.F90:160,207`;
   `stprk3_stg.F90:273`; `stp2d.F90:172`; `domzgr_substitute.h90:46,127`;
   `domqco.F90:219-220` is the compiled branch, `:227` the `#else`), including
   the `*umask` factor (`vertical.py:190`).
2. Time levels CONFIRMED at all four sites; NEMO blends `r3u`
   (`stprk3_stg.F90:166-168`) where legoESM blends `eta` and then forms
   `r3u` -- algebraically identical, `~1e-21` measured.
3. `h_u_pre` (live min-rule) vs NEMO's `e3u_0/r1_hu_0` weights for `Ue_rhs`:
   same hypothesis as A-1, same discriminator -- **REFUTED by the
   measurement above** (`<= 8.7e-14`, face 23, vs `5.9e-10` at face 22).
4. Rule 8 reading agreed: the per-step injection is NOT worse after the fix
   (kt=6 `8.938e-7 -> 8.939e-7`, kt=10 `4.41e-7 -> 2.86e-7`), so the worse
   free rows compound a separately-owned term, not a wrong site.
5. Open item 1's mechanism corrected: the implicit ZDF path is CLOSED to the
   below-seabed phantom (`A_v_u = A_v_u * _act_u3[..., 1:]` zeroes the
   interfaces; every thickness-weighted sum sees `h_partial = 0`), and the
   vertical UP3 is masked by `face_active`.  The LIVE path is HORIZONTAL:
   `_bc_horizontal_momentum_advection_flux_form` feeds `u_core = u[:, :-1, :]`
   into `_up3_reconstruct` UNMASKED, so the T-point flux at a wet bottom level
   next to the dry front face reads its `-7.9e-3 m/s` phantom as a stencil
   neighbour (NEMO's `uu` is `umask`'d at `stprk3_stg.F90:367,375,444`), with
   a nonzero transport from the wet side -- exactly faces 22/23 at the bottom
   level, the observed injection.  CONFIRMED by reading (both reviewers and
   the author traced it independently); the discriminating arm is the 3-D
   live face mask on the stage velocities.  Open item 1 is rewritten below.

Reviewer disagreement (A-1/B-3 vs the receipt) was settled by the
measurement, not averaged.

## Artifacts (`/data/abyssal/dbalwada/nemo-testcases-l1/`)

| artifact | sha256 |
|---|---|
| `stage3_remainder/after/frames/frame_gate_kt1.json` | `a15a3858b604e97e7f8769f26a1cd30c28c1328b9be44e3482eadb681360fee9` |
| `stage3_remainder/after/frames/frame_walk_kt2.json` | `f1c5c4b98a5db048c25df423ddc98b81c2cbbe17605177de5d53137c9cb96f17` |
| `stage3_remainder/after/frames/frame_walk_kt3.json` | `36326e511c6e09fa77e12c0214880aaf57d670a587591761f6b74a973434278d` |
| `stage3_remainder/after/frames/frame_walk_kt4.json` | `7c74e93ac74fa4232e6b5394f3c48a8e800fd6c73e38dc951d3ca3fa7dec3392` |
| `stage3_remainder/after/gates/lock_exchange_stage_sweep_kt2.json` | `ac81a6dd95c780abf14a240e72b438b31b9cc892b7c86d6bf624ed3ada46b1f0` |
| `stage3_remainder/after/gates/lock_stage_sweep_kt2.json` | `fe6a61fab6afb2a25b43301a7a44f94fc2e0c8a7119c7cde1fa0e8a96be2faab` |
| `stage3_remainder/after/gates/lock_trajectory_kt10.json` | `77a3d76c931a088ff43b426f6b778a6d0be61ddc66cc8206ef0c00a848cd68ff` |
| `stage3_remainder/after/gates/lock_trajectory_kt10_legacy_hadv_arm.json` | `58b98062feba1decdb9b69a785e3c1a42e9f26f1dc431ca9c243de1b1fc20786` |
| `stage3_remainder/after/gates/lock_trajectory_kt60.json` | `5df429259e79ae4b9b0e7fa797580cc340433194109e355b5bcc19c17495efa1` |
| `stage3_remainder/after/gates/overflow_stage_sweep_kt2.json` | `7f4b391eca4dce81cd69403f023fa687e7c67024ac1aa9e5a4788c1589642d09` |
| `stage3_remainder/after/gates/overflow_trajectory_kt10.json` | `54f359245b20d2541cb64d07022913f4c1943c29ec7cb106fcea7b412170e9a7` |
| `stage3_remainder/after/gates/overflow_trajectory_kt10_legacy_hadv_arm.json` | `0ce4b195f5f9a075661d60045e41c27e0813ed5269e0270bc18cb84327a203f5` |
| `stage3_remainder/after/gates/overflow_trajectory_kt60.json` | `9d42350f7d9aadfe77c213015234b6e68c9125b5574184e379c316037701f58d` |
| `stage3_remainder/after/probe/candidates.json` | `bfb97b57451c718224a5660bafdb70c98bbf45a2045b603ad0f735655aab2e9d` |
| `stage3_remainder/after/probe/growth.json` | `68ee63ef71049c44ca73a06e3b4608195833970cbaffeedc5e199436034ad008` |
| `stage3_remainder/after/probe/slow.json` | `ffff416c2a8ff50f9a1a41f2a3598dfa1bb32966425f883c294f5abc8d748324` |
| `stage3_remainder/before/candidates.json` | `e48f480c1d608da428ebd28785f88c9e1d52c7313eb6664ffa5df3588c0293fd` |
| `stage3_remainder/before/growth.json` | `edebbcf8b6a19643947d647e4bf6cd140225b6ccc4465b720242852a4d95f714` |
| `stage3_remainder/before/slow.json` | `bd6ca11a48f36b0960afdc9ea3abe811b0d0a32f7c3cfc1e48aa1f15fc7e6597` |
| `stage3_remainder/after/stats/overflow_statistics.json` | `1049adf32a3defd30c31361e5be688d3e757a8959a883bb065bc02632d6d7399` |
| `stage3_remainder/after/stats/legoesm/overflow_zps/fp64/metadata.json` | `be057804e76895f04446fa1463cb4feee031dbbc54b737c0aec08e35617e0913` |
| `stage3_remainder/after/stats/legoesm/overflow_zps/fp64/states.npz` | `9dde36f43049d2cedae83ec5c6877c8617bc47db7bf88c2a3f0b57058109447e` |
| `stage3_remainder/after/stats/legoesm/overflow_zps/fp32/metadata.json` | `92bc3b2ffa9663319e655bc8544049d2a5ebb1505b85dce99aa5a4852603f44e` |
| `stage3_remainder/after/stats/legoesm/overflow_zps/fp32/states.npz` | `7aac0f389d6c009d2464a01c4d168a262dfca7a8d61583a62ec2eb548a296d9d` |

legoESM commits on `fidelity/overflow-stage3-remainder`: `8e75304ea`,
`fcd0d6057` (probe), `f8247f9a3` (preregistration), `60d0c5420` (fix),
`bd0861319` (map/registry), `e0910abee` (receipt), plus the review-round
commit (one-variable hygiene of the stage pair; this section).
