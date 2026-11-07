# PREREG — the NEMO-faithful DINO card moves onto NEMO's own vertical ladder

Round 61, #1728.  Written BEFORE the edit and BEFORE any measurement.
Decision 27 (user, "Ok go for it") authorises the flip; nothing else here is a
choice.

## 0. What is wrong, in one line, with its citation

The DINO binary is compiled `key_qco key_vco_3d` (`cfgs/DINO/cpp_DINO.fcm:1`).
Under `key_vco_3d` every reference scale factor in NEMO's source is the 3-D
array — `#define E3t_0(i,j,k) e3t_3d(i,j,k)`, `#define DEPt_0(i,j,k)
gdept_3d(i,j,k)`, `#define E3w_0(i,j,k) e3w_3d(i,j,k)`
(`src/OCE/DOM/domzgr_substitute.h90:102-119`).  `e3t_1d` is reachable ONLY
under `key_vco_1d` (`:72-87`), which DINO does not define.  The standalone card
builds its staircase from `e3t_1d`.  The two ladders are not the same ladder
(they part company below k=25), so the card is integrating in boxes NEMO does
not have.

## 1. The consumer table — NEMO statement ↔ legoESM statement

Every row is a consumer of the REFERENCE geometry on the leap-frog qco path.
"after" is what the row reads once the card's `z_coord` is built from NEMO's
3-D ladder.

| # | consumer | NEMO statement (cited) | legoESM statement | before | after |
|---|---|---|---|---|---|
| C1 | live layer thickness | `e3t(i,j,k,t) = E3t_0(i,j,k)*(1+r3t(i,j,t)*tmask)` — `domzgr_substitute.h90:45` (`Tmsk`) with `E3t_0 = e3t_3d` (`:102`); `r3t = ssh*r1_ht_0` `domqco.F90:160` | `compute_layer_thickness(eta, H_bathy, z_coord)` off `z_coord.h_partial`/`dz_ref` | `e3t_1d` | `e3t_0` |
| C2 | column reference depth | `ht_0 = Σ_k e3t_0*tmask` `domain.F90:144` | `H_bathy` = `cumsum(dz_ref)[k_bot-1]` in `bridge_nemo_to_legoesm_topo` | `Σ e3t_1d` | `Σ e3t_0` |
| C3 | u/v face reference depth | `hu_0 = Σ_k e3u_0*umask` `domain.F90:145`; `r1_hu_0 = ssumask/(hu_0+1-ssumask)` `:159` | `min_cell_to_uface(h)` summed; raw `z_coord.nemo_hu_0` on the qco arm | face depth from `e3t_1d` | from `e3t_0` |
| C4 | barotropic seed / reconcile | `pr3u = ½(e1e2t·ssh + e1e2t·ssh)·r1_hu_0·r1_e1e2u` `domqco.F90:167-168`; `dyn_spg_ts` depth-mean of `puu(Kbb)` over `e3u(Kbb)` | `nemo_qco_resolved_mesh_operands` (raw `nemo_hu_0`) + live `h` from C1 | mixed: raw hu_0 exact, live h wrong | both `e3t_0` |
| C5 | S-EOS depth | `zh = gdept(ji,jj,jk,Knn)` `eosbn2.F90:297`, `gdept = DEPt_0·(1+r3t)` `domzgr_substitute.h90:49,105` | `z_coord.t_depth_ref` (`eos.py:823`) | `gdept_1d` | `gdept_0` |
| C6 | sco pressure-gradient depth | `gdept_z0(i,j,k,Kmm)` and `e3w(i,j,k,Kmm)` `dynhpg.F90:345-380` | `ocean_pe_latlon_cgrid.py:1900-1906` `t_depth_ref`; raw `nemo_gdept_0`/`nemo_gdepw_0`/`nemo_e3w_0` on the fidelity arms | `gdept_1d` on the `t_depth_ref` site | `gdept_0` |
| C7 | vertical-mixing `dz`/`dz_half` | `e3w(Kmm)` divisor, `trazdf.F90:219-221` with `domzgr_substitute.h90:108` | `nemo_e3w_kmm` arm 1 = raw `z_coord.nemo_e3w_0`·stretch (`implicit_solver.py:829-846`) | ALREADY NEMO's `e3w_0` | unchanged |
| C8 | wzv / div_hor qco operands | `domain.F90:139-159` | `nemo_qco_resolved_mesh_operands` raw branch (`vertical.py:308-326`) | ALREADY NEMO's | unchanged |

C7 and C8 are already on NEMO's arrays; the gap is C1-C6, all of which hang off
the ONE `dz_ref`/`t_depth_ref` pair the bridge hands
`create_z_star_from_thicknesses`.

## 2. The change (shortest diff that is actually correct)

One shared path, no second implementation: the standalone card already builds
its domain by calling `bridge_nemo_to_legoesm_topo` — the SAME function, with
the same arguments, that the certified twin calls.  The twin resolves the
ladder to NEMO's (`kamm_twin_90d.resolve_ladder_mode` → `"both"`); the
standalone card passed nothing and got the bridge-wide default `"off"`.

1. `_nemo_faithful_dino_domain` passes `e3t_mode="both"` — a literal, not a
   knob: `LEGOESM_NEMO_E3T` can no longer put this card back on `e3t_1d`.
2. `effective_vertical_scale_factors` takes the level's EXACT value where the
   level is horizontally uniform instead of `np.mean` over identical cells.
   MEASURED, before the edit: the mean lands 1 ulp off the value on 3 of 35
   `e3t_0` levels and 18 of 35 `gdept_0` levels.  A gate whose bar is "0 cells
   unequal" refuses that, and it is the oracle's own number being rounded by
   our reduction.  Non-uniform levels keep the mean they always had, so no
   other mesh moves.
3. `nemo_dino_mesh_gate.py` section 3(a) compares the analytic domain against
   the FILE-read domain through the same bridge; that call gets the same
   `e3t_mode="both"`, or it would be comparing two different ladders.

## 3. The gate (preregistered before it is written)

`nemo_dino_mesh_gate.py` gains SECTION 4, MODEL-CONSUMED GEOMETRY: every field
below taken from the domain `run_dino.py`'s own path CONSTRUCTS
(`dino_lat_lon_grid` → `dino_lat_lon_vertical` → `dino_lat_lon_state`), scored
against `mesh_mask.nc`.

Bar: **0 cells unequal**, per field, over NEMO's own wet mask.

| field | model side | oracle side |
|---|---|---|
| e3t_0 | `z_coord.h_partial` (live-thickness reference) | `e3t_0`·tmask |
| e3t_0 (1-D) | `z_coord.dz_ref` | `e3t_0` at any wet cell of the level |
| gdept_0 | `z_coord.t_depth_ref` | `gdept_0` |
| gdepw_0 | `z_coord.nemo_gdepw_0` | `gdepw_0` |
| e3w_0 | `z_coord.nemo_e3w_0` | `e3w_0` |
| e3u_0 / e3v_0 | `min_cell_to_uface/vface(h_partial)` | `e3u_0` / `e3v_0` |
| e3uw_0 / e3vw_0 | `z_coord.nemo_e3w_0` (uniform ladder ⇒ equal, `zgr_lib.F90:206-264`) | `e3uw_0` / `e3vw_0` |
| ht_0 | `state.H_bathy` | `Σ e3t_0·tmask` |
| hu_0 / hv_0 | `z_coord.nemo_hu_0` / `nemo_hv_0` AND the live face sum | `Σ e3u_0·umask` / `Σ e3v_0·vmask` |
| r1_hu_0 / r1_hv_0 | `nemo_qco_resolved_mesh_operands` → the reciprocal the model divides by | `ssumask/(hu_0+1-ssumask)` `domain.F90:159` |

`--plant-ladder` perturbs the CONSTRUCTED ladder by 1 ulp and every row above
that depends on it must flip to FAIL; a plant that leaves the section green is
a vacuous gate.  The pytest wrapper SKIPS when `mesh_mask.nc` is absent.

## 4. Predictions (falsifiable, written before the run)

| row | now | predicted after | refuted if |
|---|---|---|---|
| geometry gate, every field | DEBT (94134 / 2503 cells) | **0 cells unequal** | any field is non-zero |
| R2 barotropic seed η/U/V | 0 / 0 / 0 cells | **still 0 / 0 / 0, bit-exact** | any becomes non-zero — from rest ssh, ub, vb are identically zero and no ladder multiplies zero into something else |
| R0a live thickness / R0b u-face depth | 94134 / 2503 cells | **0 / 0** | non-zero |
| step-1 η rms 2.639e-4 m | — | **falls below 1.3e-4** (>2x) | it does not fall, or rises |
| step-1 u 7.601e-6 / v 1.863e-5 | — | both fall | either rises |
| step-1 T 4.996e-6 K / S 2.337e-7 | — | move by less than 2x in either direction | a >2x move (the ladder change lives below 1000 m; one Euler step from rest is surface-forced) |
| R4a/R4b barotropic exit | 1.75% / 0.65% / 1.32% | fall, but **not to zero** | they reach zero — two source-literal diffs (the η continuity assembly and the raw-`wgtbtp1` primary average) are still unlanded, so an exact exit would mean one of them is not real |
| year-360 T3D rms 2.75e-2 K | — | **no direction claimed.** The 360-day curve is a chaotic integral; Rule 8 says a faithful fix may worsen it | — |

If the geometry gate reaches 0 on every field and the step-1 η row does NOT
improve, the vertical ladder does not own the barotropic residual.  That is a
finding, not a reason to revert (Rule 8).

## 5. Retraction watch

`nemo_state_bridge` carries an unreproduced note that NEMO's true thicknesses
destabilise the model (max|u| 0.66 → 2.2 m/s over 20 days).  Four 90-day arms
failed to reproduce it but never refuted it.  The 360-day from-rest run in this
round is the re-check at this commit: `max|u|` is reported next to the curve.

---

## 6. HOW IT RESOLVED (written after the runs; nothing above was edited)

### The claim review changed two things before any code was written

* **RETRACTED, section 4's R2 row.** "the barotropic seed stays bit-exact
  zero" is UNFALSIFIABLE from rest: the seed is `0 * x`, already 0/0/0, and no
  ladder can move it. It is not evidence for anything and is recorded here as
  a prediction that should not have been written. It becomes a real test only
  on the restart step-walk arms, where `ssh != 0` and both `r1_hu_0` and the
  `e3u(Kbb)` depth-mean divisor carry the ladder.
* **CORRECTED, claim 1.** `e3t_1d`/`gdept_1d` are not "unreachable"; they sit
  UPSTREAM of the 3-D ladder. `zgr_lib.F90:167` picks the re-anchor level from
  `pdepw_1d`, and `usrdef_zgr.F90:120` passes `pdept_1d` to
  `zgr_msk_top_bot`, so **NEMO's wet/dry bottom index comes from the 1-D
  ladder**. That is safe here and it is MEASURED, not argued: legoESM takes
  `k_bot` from the mesh's `tmask`, never from `z_coord`, and the mesh gate
  reports 0 of 342134 wet cells differing after the change.
* **DECLARED BLIND SPOT.** NEMO builds `e3u_0 = 0.5*(e3t(i) + e3t(i+1))`
  (`zgr_lib.F90:230`), not the minimum legoESM takes. On a horizontally
  uniform ladder the two are identical, so the gate row goes green WITHOUT
  testing the operator. Not changed here (it is a card-wide change with its
  own Rule-12 table); written down so the next non-uniform card does not
  inherit a green row as evidence.
* **COVERAGE GAP, added.** `ln_dynvor_een = .true.` (`namelist_cfg:331`), so
  `e3f_0vor` divides relative vorticity at every f-point and is built from the
  3-D `e3t_0` (`dynvor.F90`). The EEN barotropic operand bundle carries NEMO's
  own `e3f_0`; section 4 now scores it.

### THE DEFECT THE FLIP EXPOSED: a silent fallback keyed on the wrong ladder

Putting the card on NEMO's thicknesses made step 1 **3000x WORSE**, not better
(T rms 5.0e-6 -> 1.6e-2 K, i.e. 127% of NEMO's own first step). Rule 8 says
find what the wrong version was compensating for. It was this:

`_nemo_dino_domain_for` decides whether a run gets NEMO's OWN initial state or
the analytic paper profile, and it decided by comparing the run's `dz_ref`
against `e3t_1d` alone. The moment the card moved onto `e3t_0` that test
failed, the function returned `None`, and the run quietly swapped NEMO's
initial condition for a different one. Both ladders are NEMO's; the predicate
now accepts either, and rejects anything else exactly as before.

This is the "silent fallback that substitutes physics" shape: no error, no
warning, a plausible run, and a 3000x wrong answer. It was invisible while the
card stood on the ladder the predicate happened to name.

### The measured table

Day 0 is unchanged and bit-exact on both arms (`nemo_dino_fromrest_gate`: T
and S 0 cells unequal), so the step-1 movement below is the STEP, not the
initial state. Both arms ran the same command on the same config; the run
metadata differs only in `output_dir`.

| row | predicted | measured | verdict |
|---|---|---|---|
| geometry gate, 18 fields | 0 cells unequal | 0 cells unequal, all 18 | CONFIRMED |
| R0a live thickness | 0 | 94134 -> **0** | CONFIRMED |
| R0b u-face column depth | 0 | 2503 -> **0** | CONFIRMED |
| step-1 eta rms | < 1.3e-4 m | 2.639e-4 -> **6.82e-7** (387x) | CONFIRMED |
| step-1 u rms | falls | 7.601e-6 -> **1.08e-8** (703x) | CONFIRMED |
| step-1 v rms | falls | 1.863e-5 -> **6.46e-8** (288x) | CONFIRMED |
| step-1 T rms | < 2x either way | 4.996e-6 -> 4.959e-6 (1.007x) | CONFIRMED |
| step-1 S rms | < 2x either way | 2.337e-7 -> 2.314e-7 (1.010x) | CONFIRMED |
| R4a/R4b barotropic exit | fall, not to zero | 1.75/0.65/1.32% -> 0.0018/0.0025/0.0034%, still DEBT | CONFIRMED |

The barotropic loop is EXONERATED as the owner of the kt=1 sea-level and
velocity residual: it was the boxes, not the loop. What remains at step 1 is
the TRACER path (T/S unmoved at ~5e-6 K), whose largest named component is the
oracle-side `qns_b` solar-term DEBT already recorded by
`nemo_dino_fromrest_gate`.

## 7. WHAT THE DIFF REVIEW BROKE, AND WHAT IT COST

Eleven findings; the first two would have made this round's evidence worthless.

1. **THE GATE PASSED ON A WRONG LADDER.** The "e3t_0 reference ladder" row
   fell back to the MODEL's own `dz[k]` as the oracle value on a level with no
   wet cell — it compared the model against itself and printed EXACT. The
   reviewer planted 300 m at every dry level and all 18 rows still read EXACT
   with `GATE PASS`. The oracle side is now taken from `e3t_0` itself (it is
   horizontally uniform over the WHOLE array, dry cells included, and the gate
   now CHECKS that before using it). DINO's level 36 is the one dry level and
   it is now an explicit WAIVER with its measurement printed (e3t 506.375 vs
   617.462, gdept 4253.187 vs 4308.731), a precondition the gate verifies on
   the run (0 active cells, 0 thickness, 0 wet mask), and a pin: the waived
   value must be NEMO's OTHER ladder, not an arbitrary number, so the
   reviewer's plant now exits non-zero.
   NOT CHANGED, and on the ASKED table instead: moving the model's dry level
   onto `e3t_0` as well. Measured, it alters `dz_ref[36]`, `t_depth_ref[36]`,
   `z_full_ref[36]`, `z_half_ref[37]`, `dz_half_ref[35]` and `H_max` and
   leaves `h_partial`, `H_bathy`, `is_active` and the initial T bit-identical
   — but `H_max` and `dz_half_ref` are carried state and this is the campaign
   owner's call, not a silent edit.
2. **THE IN-PLACE ORACLE EDIT WAS STILL COMMITTED AND REACHABLE.**
   `spg_kt1_barotropic_ladder.py --emit-runsh` wrote a script that edited
   `cfgs/DINO/MY_SRC/dynspg_ts.F90` in place and ran `makenemo -r DINO -n
   DINO`, overwriting the oracle's own `BLD` tree and `bin/nemo.exe` — which
   no trap restores. The config-copy acquisition was supposed to replace it
   and did not remove it. Deleted, with a test that asserts on the source
   (the flag is gone, so no invocation can exercise it).
3. **`run.sh` advertised a handoff that cannot work.** It told the operator to
   re-score with `spg_kt1_barotropic_ladder.py`, which opens the OLD fixed
   filenames — every one of them rank-tagged by the patch. It now says so and
   names extending the ladder gate as the next change.
4. **The record could be written into `cfgs/SHARED`.** The guard named only
   `cfgs/DINO` and `src/`; `OUT=$NEMO/cfgs/SHARED` walked through and would
   have clobbered the `namelist_ref` every configuration includes. The record
   may now not land anywhere inside the checkout. A second hole found while
   fixing it: `readlink -f` returns EMPTY for a path whose parent does not
   exist, and `local real=$(...)` hides that from `set -e`, so every
   not-yet-created path passed silently. Now `readlink -m`, with a
   parametrised test over five locations plus a symlinked checkout.
5. **One ppsrc token proved nothing.** `ACTION='WRITE'` already occurs once in
   the pristine source. The build check now REFUSES if any token it greps for
   is also present unpatched.
6. **Three lanes flip their initial condition** at
   `LEGOESM_NEMO_E3T=both` — `kamm_run5y_v3.py`, `box_budget_run.py`,
   `dino_year_screen_fullframe.py` (which REQUIRES an explicit mode) — from
   the paper profile to NEMO's own. That is the fix, and they are named.
7. **Two of the four screens the docstring claims to cover cannot reach NEMO's
   initial condition at all**: `dino_90d_screen` and `dino_year_screen` read
   at `nn_hls=2`, so their mask is 203x56 against the mesh's 199x52 and the
   predicate returns `None` at either mode. Pre-existing; named, not fixed.
8. The plant banner said "level 1" while planting every level. Fixed.
9. `r1_hu_0`/`r1_hv_0` re-derive the formula rather than calling it, so they
   test the PREDICATE and the OPERAND, not the arithmetic. Written at the row.
10. `e3uw_0`/`e3vw_0` catch the carried array being absent, zeroed, truncated
    or swapped; they do not test the face interpolation, which on a uniform
    ladder is the identity. Written at the row.
11. The stitcher was only tested on a 2-rank i-split. It is now tested on the
    REAL 16-rank decomposition, and the decomposition constant is itself
    checked against the oracle's own `layout.dat`.
