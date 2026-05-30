# FV3-faithful cubed-sphere — change log

Goal: make the legoESM cubed-sphere finite-volume core **faithful to GFDL FV3**
(`atmos_cubed_sphere`), so that across the full atmosphere + ocean test matrix
(shallow water → AMIP/OMIP) the cube results match the **MPAS / icosahedral**
and **lat-lon FV** cores, with **zero edge/corner artifacts** on the cube
panels. Verification is **visual** (native + regridded PNGs) plus quantitative
cross-grid norms, with `/codex:adversarial-review` on code + visuals.

Run on **CPU** (`JAX_PLATFORMS=cpu`); Metal backend broken in this env
([[metal-backend-broken-use-cpu]]).

---

## Environment constraint — Fortran oracle is NOT readable here
`../Code/FV3/atmos_cubed_sphere-symmetryclean` resolves to
`/Users/pierregentine/Documents/Code/FV3/...`. macOS TCC blocks the whole
`~/Documents` tree *except* the `legoESM` cwd — `ls` returns
`Operation not permitted` even with the sandbox disabled. No vendored copy in
the repo. So the Fortran cannot be diffed line-by-line in this session.
Faithfulness is therefore audited against **known FV3 algorithms** (Lin–Rood /
Putman-Lin D-grid FV; d2a2c; PPM `fv_tp_2d`; d_sw5 corner div-damp; d_sw6
del-6 vorticity damp; a2b corner ops) already partially ported in-repo, plus
the **physical benchmarks** (Williamson, cosine-bell advection) cross-grid.

## User-flagged concrete targets (2026-05-30)
1. **Cosine bell (PL07) — no grid distortion.** Day-1 regridded cube ≈ ico5 ≈
   latlon (good). Scattered squares at high-lat in the *native* plot are
   native-cube scatter rendering (cosmetic). OPEN: verify full **12-day** run
   for panel-edge imprint as the bell crosses cube seams.
2. **Williamson-5 — waves must propagate on the cube like latlon/ico.**
   Baseline: cube **over-concentrates** amplitude at the mountain (~25 m/s blob
   at −75,15) and **under-propagates** the downstream Rossby wave train vs
   latlon/ico. Primary suspect: stacked heavy damping (see below).
3. **Cross-grid longitude alignment.** All regridded/native PNGs must share the
   same longitude centering (continents/features at the same lon). VERIFY on
   AMIP/OMIP (SW W5 mountain is aligned at ~−90 across grids — looks OK there).
4. **No edge/corner artifacts** anywhere on the cube (visual, native plots).

## Root-cause hypothesis (central)
The cube SW config stacks **div_damp = 8× `_div_damp_cube(n)`** +
**hyperdiff = 2× `_hyperdiff_cube(n)`** (W2/W5/W6) + **del-6 vorticity damp
`damp_v=0.030, nord_v=2`** — all calibrated over iters 761→71 to pass the
**W2 steady-state** sentinel `v_ll_Linf ≤ 0.119`
(`tests/test_iter1002_w2_target_met.py`, helper
`iter1009_dual_target_config`). W2 is a *steady* balanced flow, so heavy
damping is "free" there, but it **suppresses real W5 wave propagation**. FV3
keeps W2 clean through **correct edge/corner discretization**, not brute-force
damping. ⇒ "faithful to FV3", "no edge artifacts", and "waves propagate" are
the **same** root cause: residual cube edge/corner imprint masked by damping.

## Done this run (iter 1)
- Mapped the cube FV3 stack (call chain matrix-runner → `FV3EdgeShallowWaterModel`
  → `fv3_sw_tendencies` → `_d2a2c_vect` / `fv_tp_2d` PPM / `cdgrid_momentum_tendencies`
  → halos `pad_halo_dgrid_vector_4d` + corner fills). 55 file:line anchors.
- **Fixed blank native cube velocity plots** (`scripts/run_atmosphere_test_matrix.py`
  `_save_native_snapshot_plots`): cube stores `u/v/wind_speed` regridded to
  (181,360) but the native scatter uses native (lon,lat) of size 6·n·n, so the
  size mismatch silently `continue`d → blank `snapshots_{u,v,wind_speed}_native.png`
  for **every** cube velocity case (W2/W5/...). Now routes the native
  cell-centre `u_cc_east`/`v_cc_north` (and derives native `wind_speed`).
  Verified: cube W5 `snapshots_u_native.png` renders. NOTE follow-up: ico/spectral
  native velocity plots have the same blank issue but store no native cc winds.

## Baseline matrix (quick, CPU) — `results/fv3_baseline/atmosphere`
SW all grids PASS (mass drift O(1e-15) cube). W5 cube finite (u 0.6→25.5 m/s),
NOT a NaN/blowup — purely over-damped propagation + the (now-fixed) plot bug.

## iter 2 — W5 damping-sensitivity probe (full 15-day, `results/probe_w5`)
Added reusable env knobs `LEGOESM_SW_DIV_DAMP_FACTOR` (def 8.0) +
`LEGOESM_SW_HYPERDIFF_FACTOR` (def 2.0) to the cube SW W2/5/6 config (unset ==
unchanged). Metric tool `scripts/probe_w5_metrics.py`.

Day-15 eddy/propagation metrics (cube vs latlon vs MPAS/ico):
| run | eddy_rms | eddy_down | reach_E° | spd_max |
|-----|---------|-----------|----------|---------|
| cube dd8/hd2 (prod) | 149.5 | 119.1 | 18.3 | 41.7 |
| cube dd4/hd1 | 151.7 | 120.0 | 18.3 | 42.7 |
| cube dd2/hd0.5 | 196.4 | 138.6 | 96.5 | **127.9 BLOWUP** |
| latlon (ref) | 147.4 | 103.0 | 20.0 | 35.5 |
| MPAS/ico (ref) | 167.2 | 148.5 | 22.3 | 45.8 |

Findings:
- At **day 15** the cube W5 is comparable to latlon/MPAS — it DOES propagate.
- **Lowering damping does NOT fix it — it BLOWS UP** (dd2/hd0.5 → 127 m/s, noise
  everywhere). So damping is *masking* a real cube edge-imprint instability;
  it is not a free knob. The fix must be a better edge discretization, not
  less damping.
- **Day ~1.5** is where the user-visible deficit lives: cube downstream
  h'rms = 64.8 vs latlon 74.0 / ico 74.8 (~12% under-propagation), near/down
  ratio 8.71 vs ~8.1 (cube concentrates more at the mountain). Modest but real
  and matches the day-1 images. spd_max ~30 m/s all (no early blowup).

CONCLUSION: cube W5 is ~12% over-damped on freshly-launched downstream waves;
root cause = residual edge/corner imprint that forces the heavy damping crutch.
`FV3EdgeShallowWaterModel` is self-described "research path, NOT a faithful FV3
port" (A-L 4-pt grad vs FV3 2-pt c_sw; RK3 vs forward-backward; edge-midpoint
stagger vs tile-edge coupling; aggregated div_damp vs structured d_sw5). The
faithful path is `FV3FBShallowWaterModel`/`fv3_fb_sw_step` + `use_fv3_dsw5_corner_damping`.

## Cross-grid data note (ask D)
Regridded `snapshots_latlon.npz` uses canonical lon[-180,180]/lat[-90,90] for
cube+ico, BUT the **latlon** grid writes fields at NATIVE (72,144) while still
labelling lat/lon as (181,360) — a harmless-but-confusing inconsistency; the
cross-grid comparison plotter renders each on its own extent so continents
align (verified day-1 SW comparison). Re-verify on AMIP/OMIP.

## Plan (next iters)
- iter 2: damping-sensitivity probe — cube W5 downstream wave amplitude vs
  div_damp/hyperdiff factor; quantify vs latlon/ico. Confirm the lever.
- then: tighten edge/corner discretization (d2a2c non-orthogonal metric, a2b
  corner ops, PPM cube-edge formulas) to cut the W2 imprint *without* damping,
  then back off damping → W5 propagates. Re-pin the W2 sentinel.
- run full (non-quick) cosine-bell 12-day cube for panel-crossing imprint.
- ocean matrix (rest/barotropic/baroclinic) cube vs MPAS vs latlon.
- `/codex:adversarial-review` on each code change + on the visuals.

## Status: NOT done. `DONE` withheld until cube matches latlon/MPAS across the
matrix with no edge artifacts and waves propagating, all adversarially reviewed.
