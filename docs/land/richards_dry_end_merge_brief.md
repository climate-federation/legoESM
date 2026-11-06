# Merge brief: Richards dry-end fix (`jisu/richards-dry-end-fix`) onto main

Written 2026-11-06 for the agent doing the merge.  Branch base: `f8143622` (main
as of 2026-08-31).  Two commits on top:

| commit | files |
|---|---|
| `b741b16` land: Richards solver dry-end fix for the global LMIP runoff spikes | `packages/land/legoesm/land/richards.py`, `packages/land/legoesm/land/soil_hydraulics.py`, `scripts/data/cap_land_restart.py` (new), `tests/land/unit/test_richards_dry_wetting.py` (new), `tests/unit/test_corrections.py`, `tests/unit/test_diff_land_ice_schemes.py` |
| `bdbb4dd` land: applying consistent theta floor across richards.py, multilayer_land.py, cap_land_restart.py | `richards.py`, `multilayer_land.py` (9 lines), `cap_land_restart.py`, `tests/land/unit/test_multilayer_water_balance.py`, `test_richards_dry_wetting.py` |

Main (`ee54a853`, 2026-11-06) has since rewritten the same solver (commits
`3ca6d56`, `06f9b7a`, `24487ca`, `20b3216`, `00820e6`, `6bb8b13`, `2a97cb9`,
`a11aab5`, plus `cd583d2`, `a5ab190`, `bc7bd99`).  A cherry-pick of either
commit onto main conflicts in `richards.py`; the second also in
`multilayer_land.py`.  This brief says which side to keep, hunk by hunk, and
how to verify the result.

## 1. Decision

Keep THIS BRANCH's Picard core and retention curve; keep MAIN's additions
around it.  Reason (section 4): on the dry cell that produced the global
runoff spikes, main's hourly solve does not converge and places the rain
through the whole 3 m column; this branch converges and keeps the wetting
front at the surface, matching a 60-second reference.

## 2. Mechanism map

| aspect | main | this branch | keep |
|---|---|---|---|
| Picard update variable | psi: `psi_new = psi_m + dpsi` (damped), `theta = theta(psi)` | theta: `theta_lin = theta_m + C_m*dpsi`, then `psi = psi_from_theta(theta)` | **branch** — the linearised balance solved for the mass change; dividing it by a capacity that collapses at the dry end (C/dt ~ 1e-16 1/s) is the spike |
| iterate control | damping `lam = min(1, max_dse_per_iter/dse_try)`, 30 iterations, `converged` flag, unconverged columns keep the last iterate | fixed 10 iterations, no damping | **branch core, main's bookkeeping**: keep `max_iter`, `theta_tol`, `max_dse_per_iter`, the `conv`/`done_m` freeze and the `converged`/`n_iter` outputs; apply `lam` to `dpsi` BEFORE the theta update (untested combination, see 5) |
| retention above saturation | elastic branch from psi = 0; Clapp–Hornberger flat between air entry and 0 with the Se = 1 capacity (~0.36 1/m vs elastic 4e-5) | elastic branch from air entry (`soil_hydraulics.air_entry_psi`), `theta_from_psi` strictly increasing, `psi_from_theta` its exact inverse through saturation, `moisture_capacity` consistent | **branch** (`soil_hydraulics.py` hunks: `air_entry_psi`, `theta_from_psi`, `psi_from_theta`, `moisture_capacity`) |
| upper bound per layer | none (over-saturation stored elastically) | `theta <= theta(z_k + h_s)` (hydrostatic head of the pond); excess cascades upward into room, then onto the surface store; `theta_new = clip(theta_lin, _theta_floor, theta_cap)` | **branch** |
| surface link | Robin flux capped at supply (`q01_robin > _avail_rate`) | same plus the mirror cap `q01 >= min(0, _avail_rate)` (no exfiltration) | **branch** `_surface_terms(h_s, psi0, theta0)` |
| water budget | post-solve give-back/take-back (`refill`, `water_created`) | conserved at every iterate | **branch core; keep main's output fields** — `water_created` should come out ~0; keep `refill` (the dry-floor refill of an unmet demand is still real) |
| dry floor | `psi_dry_floor(hydro_config)` (Se-based), clamp per iterate | `dry_floor(hydro_config) -> (psi_floor, theta_floor)` | either; the branch's returns both values, `multilayer_land.py` needs the theta one |
| only on main | `log_impedance` argument (frozen-soil infiltration/drainage impedance), `k_sat_decay_m`, `_f_drain` field-capacity limiter, PDI/Lu psi shift (`_shift_psi`), `_bottom_drain_K`, CFL cap on `_Ksat`, dtype promotion of the carry | — | **main** |

## 3. Hunk-by-hunk resolution

`packages/land/legoesm/land/richards.py`

1. Module docstring: main's (fixed trip count + damping + `converged`), add one
   sentence: the update is applied in theta and psi recovered by the inverse
   retention curve.
2. `__param_spec__`, `RichardsConfig`, `RichardsOutput`: **main** verbatim.
3. `psi_dry_floor` (main) vs `dry_floor` (branch): keep **main's**
   `psi_dry_floor` and add the branch's theta companion so callers get
   `(psi_floor, theta_floor)`; or keep both names.  `multilayer_land.py` (hunk
   below) imports `dry_floor`.
4. `solve_richards` signature: **main** (has `log_impedance`).
5. `_surface_terms`: **branch** body (exfiltration cap), inside main's
   surrounding `_Ksat` block (CFL cap, `k_sat_decay`, `log_impedance`).
6. Picard body, after `dpsi = thomas_solve(...)`:
   - main's `dh_s`, damping `lam`, `dpsi = dpsi*lam`, `dh_s = dh_s*lam`,
     `h_s_new = h_s_m + dh_s` — keep;
   - then the **branch's** theta update block (from `theta_lin = theta_m +
     C_m * dpsi` through `psi_new = jnp.maximum(psi_from_theta(theta_new,
     hydro_config), _psi_dry_floor)`), replacing main's `psi_new = max(psi_m +
     dpsi, floor)` / `theta_new = theta_from_psi(psi_new)` pair;
   - main's `_bottom_drain_K`, `_k_bot_diag`, `conv`, `done_m` freezes,
     `n_it_new` — keep.
7. Post-solve: main's pond/runoff split, `refill`, `water_created`, PDI/Lu
   `_shift_psi` — keep.  Note: with the branch's inversion applied every
   iterate, PDI/Lu (approximate inverses) are not covered by the branch's
   tests; keep main's shift and run main's PDI/Lu tests.

`packages/land/legoesm/land/soil_hydraulics.py`: take the **branch's** four
hunks (`air_entry_psi`, elastic branch from air entry in `theta_from_psi` and
`moisture_capacity`, exact inverse in `psi_from_theta`); keep main's
`slice_layer` deletion and ice-impedance code.

`packages/land/legoesm/land/multilayer_land.py`: one hunk — the water the soil
can give is measured above the solver's theta floor, not above `theta_r`
(`jnp.maximum(theta_avail - _theta_floor, 0.0)`), with
`_, _theta_floor = dry_floor(config.hydraulics)`.  Main's line is
`jnp.maximum(theta_avail - theta_r, 0.0)` (main:1336).  Apply the branch's
floor there.

`scripts/data/cap_land_restart.py`: **add** (new file).  Raises a pre-fix
restart's over-pressured / sub-floor layers to the capped state once;
rebuilds the per-cell hydraulics from surfdata as the driver does.  Needed for
any run restarting from a file written before the fix.  Has no test of its own.

`scripts/run/submit_lmip_ginsburg.sh`: the branch's edit is dropped (file no
longer exists on main).

Tests: add `tests/land/unit/test_richards_dry_wetting.py` (13 tests); apply the
branch's edits to `test_multilayer_water_balance.py`, `tests/unit/test_corrections.py`,
`tests/unit/test_diff_land_ice_schemes.py` (they pin the new retention/inverse
behaviour).  Then run main's own Richards tests too.

## 4. Evidence (spike cell: 15S 32E Cosby Clapp–Hornberger clay, theta_sat 0.426, psi_sat −0.169 m, b 7.1, K_sat 5.36e-6 m/s, theta_r 0; 10 layers over 3 m, growth 2, top layer 2.93 mm; dry profile floored at psi = −1e6 m; rain for one hour; default `RichardsConfig()`)

| rain | main, one 3600 s step | this branch, one 3600 s step | 60 s × 60 reference (same solver) |
|---|---|---|---|
| 1 mm/h | top theta/theta_sat 0.11, **not converged** (30 it.), water in layers 3–5 | top 0.53, converged (10 it.) | top 0.60 (both solvers) |
| 10 mm/h | all 10 mm placed through 3 m (4.75 mm in the bottom 1.5 m), top 0.12, not converged | top two layers saturated, 3.8 mm stored, 6.2 mm ponded, converged | top 0.81–0.87, 10 mm stored in the top 9 cm, no pond |
| 50 mm/h | 50 mm through 3 m (23.7 mm in the bottom layer), no pond, not converged | 8 mm stored in the top 2 cm, 42 mm ponded | 26–33 mm stored, 10–16 mm ponded |

Pre-fix base (`f8143622`) on the same case: runoff up to 1e10 mm/day, top layer
at 970 × porosity — the global-run spikes.  Main removes the spike but keeps an
unconverged iterate that moves rain to depths it cannot reach in an hour
(K ~ 1e-5 mm/h at theta = 0.2); its budget closes through the take-back term.
This branch converges and errs on the conservative side (ponds water that
infiltrates in later steps).

## 5. Verification after the merge

1. `JAX_ENABLE_X64=1 python -m pytest tests/land/unit/test_richards_dry_wetting.py tests/land/unit/test_multilayer_water_balance.py tests/unit/test_corrections.py tests/unit/test_diff_land_ice_schemes.py` — on this branch: 141 passed, 1 failed (`test_surface_resistance_throttles_dry_soil_evaporation`, fails identically on the pre-fix base; unrelated), 1 skipped.
2. Main's Richards tests (whatever `tests/land` collects for `richards`), especially the PDI/Lu curves and the damping/convergence tests of `3ca6d56`.
3. The probe below on the merged code: expect converged = True at 1 and 10 mm/h, top layer wet (theta/theta_sat ≥ 0.5 at 1 mm/h), zero runoff, budget error ≤ 1e-9 mm, no water below 10 cm after one hour at 1 mm/h.
4. Combination not yet tested: main's damping `lam` applied before the branch's theta update.  If the probe no longer converges in ≤ 30 iterations with damping on, set `max_dse_per_iter` large (damping off) and report.
5. Dual adversarial review of the conflict resolution before the PR (repo rule).

```python
# richards_spike_probe.py — rain on the bone-dry clay column, one hour, default config
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from legoesm import constants
from legoesm.land.richards import RichardsConfig, solve_richards
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta, theta_from_psi
DT = 3600.0; RHO = float(constants.rho_water)
grid = make_soil_grid(SoilGridConfig(n_layers=10, total_depth=3.0, growth_factor=2.0)); dz = np.asarray(grid.dz)
ch = SoilHydraulicsConfig(retention_curve="clapp_hornberger", theta_sat=0.426, psi_sat=-0.169,
                          b_ch=7.1, K_sat=5.36e-6, theta_r=0.0)
theta0 = jnp.array([[0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.20]])
theta0 = jnp.maximum(theta0, float(theta_from_psi(jnp.asarray(-1.0e6), ch)))   # driest admitted state
psi0 = psi_from_theta(theta0, ch)
for rain in (0.01, 0.1, 1.0, 10.0, 50.0):
    out = solve_richards(psi0, theta0, grid, ch, RichardsConfig(), jnp.array([rain / 1000 / DT]),
                         jnp.zeros_like(theta0), DT, surface_water=jnp.array([0.0]))
    th = np.asarray(out.theta_new[0]); st = (th - np.asarray(theta0[0])) * dz * 1000
    ro = float(out.runoff_surface[0]) / RHO * DT * 1000; dr = float(out.runoff_subsurface[0]) / RHO * DT * 1000
    pond = float(out.surface_water[0]) * 1000; budget = st.sum() + ro + dr + pond - rain
    conv = bool(out.converged[0]) if hasattr(out, "converged") else None
    print(f"rain {rain:6.2f} mm/h: runoff {ro:.3e} mm/h, budget err {budget:.1e} mm, "
          f"top theta/theta_sat {th[0]/0.426:.3f}, stored by layer {np.round(st, 2)}, "
          f"n_iter {int(out.n_iter[0])}, converged {conv}")
```

Reference numbers for this probe (this branch): zero runoff at every rate,
budget error ≤ 5e-9 mm, top layer 0.469/0.469/0.526/1.000/1.000 of saturation
for 0.01/0.1/1/10/50 mm/h, converged at 10 iterations.

## 6. Not included

The three canopy/P-model changes on `jisu/land-p-model` (unfloored net An,
`rjv25` kernel argument, C4 P-model) are deliberately NOT part of this branch;
`canopy/photosynthesis.py` is untouched here.
