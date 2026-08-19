# DCMIP TC2 / TC3 cube failures — measured diagnosis (2026-08-05/06)

Both NH cube cases have failed since they were added. This records what they
actually are, because the repository's own prose had them wrong.

## Retractions first

- **TC2 is DRY.** `tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2.py:226`
  sets `tracers_data = jnp.zeros((*shape_3d, 0))`, and
  `scripts/matrix/run_atmosphere_test_matrix.py:6244` gates Kessler on
  `test_case == "tc3"` alone — so TC2 runs with ZERO tracers and no Kessler.
  CORRECTION (codex): the "moist" label is NOT merely a display string.
  `run_atmosphere_test_matrix.py:211` puts `dcmip_tc2` in
  `frozenset({"nh", "moist", "hughes"})`, so `--family moist` SELECTS it.
  The defensible claim is narrower: TC2 carries no tracers and no Kessler
  at run time, so a moisture/Kessler ATTRIBUTION is void — but the family
  taxonomy still calls it moist, and
  `docs/validation/dycore_validation_catalog.md:73` still says so too.
  Whoever settles this should change the taxonomy or the catalogue, not
  just this note.
- **The Kessler-coupling comment belongs to TC3, not TC2**, and is not TC3's
  cause either (below).
- **There is no lat-lon NH row.** `run_atmosphere_test_matrix.py:376-378`:
  `nh_grids = [cubed_sphere, icosahedral, spectral]`. Cross-grid TC
  comparisons against lat-lon are not available.

## TC3 — CONFIRMED: the initial condition is singular at the poles

`test_case_3.py:109` builds a **latitude-independent** `u_east`. A constant
zonal wind is multivalued at a pole, so the IC injects an unresolvable polar
shear.

Controlled pair — same commit, same `--days 0.02` (**7776** steps at dt = 0.5*16/36 = 0.22222 s;
the runner computes int(days*86400/dt) -- an earlier revision said
7854, which does not reproduce), 3.5x past the
shipped step-2250 trip), same C36 config, **only the IC wind differs**:

| arm | `max\|u\|` trace | result |
|---|---|---|
| shipped | 30.2 flat, then 70.65 (face 5), 170.2 (face 4), 569.6 (face 5), nan | **FAIL** |
| `u_east * cos(lat)` | 30.21 -> 30.35, bounded, argmax on face 2 (equatorial) | **PASS** (`\|w\|max` 0.2752, mass drift 1.72e-16) |

Faces 4 and 5 are the polar caps. The fix arm never visits them.

`cos(lat)` is consistent with (PLAUSIBLE, single-arm) the **cause**; it is not automatically the **specified**
profile. Check the DCMIP-2025 TC3 document for the intended wind before
changing the case, then re-run at the full 2/24-day duration.

## TC2 — CONFIRMED finite-time singularity, cause still open

From `mean_timeseries.csv` (no new compute): `1/max|u|` is **linear in t** over
steps 8100-12960, slope -4.945e-6 (m/s)^-1 s^-1. Extrapolating to zero predicts
blow-up at step 13597; the run tripped at 13500 — 0.7% error. So
`u ~ C/(t_c - t)`: an advective/quadratic runaway, **not** an exponential
eigenmode, **not** CFL, **not** a checkerboard.

Implied scale `L = 202 km = 14.6 dx` — **resolved**. This argues against (PLAUSIBLE — a linear fit to 1/max|u| cannot by itself exclude CFL, checkerboard or dissipation mechanisms) the
grid-scale-dissipation family offline: damping a 202 km mode needs
~4.2e14 m^4/s, about 530x the shipped hyperdiffusion.

`TC1` is a valid control: extended 3h -> 6h with no code change it **passes**,
with `max|u|` saturating (increments 0.44 -> 0.30 -> 0.14 -> 0.04) and
`d(1/u)/dt` flattening 27x toward zero. A finite-time singularity requires that
derivative to stay constant and negative.

Live hypotheses, in order:

1. **Sponge relaxes to rest, not to the initial state.**
   `compressible_euler_cdgrid.py:1010-1041` damps `u`, `v`, `theta'` toward 0
   and ignores `rho'`; FV3 relaxes to `u00/v00` and never touches `pt`
   (`fv_dynamics.F90:1064-1081`). All three cases put `max|u|` at their own
   sponge-base level. Against: sigma at that level is 1.79e-7 s^-1 (inactive),
   Ri = 53.7 argues against (PLAUSIBLE — a linear fit to 1/max|u| cannot by itself exclude CFL, checkerboard or dissipation mechanisms) KH, and spectral shares the sponge and passes.
2. **Incomplete terrain-following transform.** No `G^13` or `dz_s/dx` anywhere;
   `w_surface` frozen at 0 (`:1034-1037`). TC2's mountain is 3.6 cells,
   TC1's 0.26 (invisible).
3. Unbalanced-IC transient — 35 m/s excursions at `k=0` face-edge columns,
   period ~2160 s. Against: spectral shares the IC and passes.

Do not "fix" this with damping or clipping the oracle does not have.

## Operational notes

- `short` caps at 12 h; the 200-day Held-Suarez case and the whole `nh` tier
  exceed it. Full-tier matrix runs belong on `glab1`.
- Measured cost for sizing: TC3 at `--days 0.02` is 4.8 h on a healthy node,
  6.0 h on `g055` (slow — exclude alongside `g161`).
- 2D is clean: shallow-water matrix **24/24 PASS** on all four grids.


## Provenance status (codex review of this note)

Labelled honestly rather than left implicit:

- The `cos(lat)` intervention is a SINGLE ARM. It changes the wind field, the
  shear and the KE distribution globally, so it is **consistent with** a polar
  origin, not a proof of one. No run artifact, SHA or full command is retained
  for it.
- The `1/max|u|` linear fit is **PLAUSIBLE** evidence of finite-time blow-up.
  It cannot discriminate against CFL, a checkerboard mode, or a dissipation
  defect. The CSV behind it is not in the repo.
- "Kessler is not TC3's cause" has **no Kessler-on/off control** behind it; the
  A/B varied the wind only. It also conflicts with
  `docs/science/specs/new_test_dycores.md:19`.
- The sponge/FV3 comparison numbers (max-location, sigma, Ri, spectral pass)
  and the timing/node figures have no retained configuration or artifact, and
  the FV3 file referenced is not in this repo. The "24/24 PASS on all four
  grids" line appears to conflate a cube-only sentinel count
  (`docs/dev-notes/fv3_fortran_fidelity_review.md:705`) with a four-grid matrix
  run.

Per this repo's rule that prose is a pointer and never a citable fact, treat
every quantitative line above as needing re-measurement before it is repeated.
