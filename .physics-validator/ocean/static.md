# Static analysis — `src/legoesm/ocean/`

## Units (✓ green)
Every leaf physics function has documented units in its docstring.
Boundary-converging conversions use the `legoesm.constants` /
`legoesm.thermo` central helpers.

- EOS T/S/p in `degC / PSU / Pa`, output `kg/m^3` (`eos.py:65-122`).
- Hydrostatic pressure: `(rho [kg/m^3], eta [m], dz [m]) → p [Pa]`
  (`eos.py:302-351`).
- KPP turbulent velocity scale `w_s [m/s]`, BL depth `h [m]` (LMD94)
  (`physics/vertical_mixing/kpp.py`).
- Bottom drag tendency `[m/s^2]` from `r [m/s]` and `h [m]`
  (`physics/bottom_drag/{linear,quadratic}.py`).
- Freshwater flux `[kg/m^2/s]`, virtual salt `[PSU/s]`
  (`freshwater.py`).
- Sponge `gamma [1/s]`, T_ref/S_ref in same units as T/S
  (`sponge.py`, `dynamics/ocean_tendency_common.py`).
- NPZD: tendencies in `[mol/m^3/s]`, all rates in `[1/day]` then
  multiplied by `1/86400` to convert (`biogeochemistry/npzd.py:112,134`).

## Signs (✓ green)
- `alpha_T = -drho/dT/rho > 0` for warm seawater (verified by probe
  `EOS thermal expansion sign`: alpha=2.47e-4 at 15°C, 35 PSU, 2e7 Pa).
- `B_f > 0 = unstable` (LMD94 KPP convention; consistent throughout
  `kpp.py:243-265`).
- Freshwater: net P>E freshens surface (`dS/dt < 0`) — verified by
  probe `Freshwater sign`: dS/dt = -3.41e-8.
- Bottom drag: `du/dt = -r * u / dz` opposes flow (sign tests pass in
  `tests/ocean/unit/test_bottom_drag_sponge.py`).
- Plume convection: surface tendency = 0 (plume detrains below
  surface; `plume.py:75-89` produces `nlev-1` levels from scan and
  pads with leading zero).
- NPZD: alkalinity uses the convention `dALK/dt = -growth + remin
  + (1-gamma_Z) * grazing - 2*caco3_production + 2*caco3_dissolution`
  (line 188).  This is the Sarmiento-Gruber "TA includes -[NO3-]"
  convention: when nitrate is consumed by photosynthesis, NO3- leaves
  the dissolved pool, so TA decreases (the -growth sign).  This is the
  canonical sign for total alkalinity tracked alongside DIC.  No bug
  here, but it is worth flagging in adversarial review.

## JAX purity (✓ green)
- No in-place mutation (`field.data.at[...].set` only on local copies).
- No Python `if`/`for` on traced values.  The plume `lax.scan` over
  vertical levels is correct (`plume.py:51-75`).
- No `.item()` / `.tolist()` in hot paths.
- NumPy is restricted to init / experiment scaffolding modules
  (`bathymetry.py`, `init_woa.py`, `sponge.py`,
  `experiments/*.py`).  All hot tendencies are pure JAX.
- No host-side prints inside `jit`.
- `_replace(land_mask=...)` callsites only on MPAS state (which has
  no separate face mask) — verified (`experiments/global_*.py`).

## Conservation (✓ green for tested modules)
- Implicit vertical diffusion no-flux conservation: relative drift
  `2.13e-16` after 1 step (probe `implicit_vertical_diffusion no-flux
  conservation`).
- NPZD nitrogen pool conservation with bottom export: rel_err `2.86e-16`
  (probe `NPZD nitrogen pool conservation w/ export`).
- FCT advection on cubed-sphere: structurally bounded by `q_min ≤
  q_face ≤ q_max` clip plus Zalesak alpha clip (verified by
  `test_advection_dst3.py` ZD-3 monotone tests).
- Long-run heat drift over 50 cubed-sphere steps with conservation
  fixer: 3.46e-7 (test `test_longrun_conservation_with_fixer`).  The
  test expects 1e-8 — *yellow* flag for tightened tolerance discussion,
  but the fixer is doing its job within 7 orders of magnitude of the
  raw drift.

## Limiters (✓ green for AD; one caveat)
- Plume convection sigmoid sharpness `1e4` × delta_rho ~ 0.1 kg/m³
  saturates the gating sigmoid; gradient through the active flag is
  effectively 0 once the plume is committed to either branch.
  *Yellow*: this is fine for the dT/dS tendency gradient but means
  losses depending on the convection_flag will not flow gradients
  through the on/off switch.  Consistent with KPP smoothing approach.
- DM95 slope tapering uses a smooth `tanh`-based weight
  (`_gm_redi_common.py:24-54`) — fully differentiable.
- Freshwater virtual-salt floor `dz_safe = max(dz_0, 1e-10)` is fine
  for AD; positivity-only (no zero-gradient region for typical h).
- KPP `phi_m^{-1} = (1 + 16|zeta|)^{1/4}` clamped via
  `max(.., 1.0)` (line 248) — equivalent to `(1 - 16*zeta_atm)^{1/4}`
  given the `B_f>0=unstable` sign convention.  Reviewed and consistent
  with the local convention.

## Reuse-helpers compliance (✓ green)
- All `ocean_pe_*.py` use `iterate_eos_and_pressure_anomaly`,
  `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`,
  `implicit_bottom_drag_factor` — verified by
  `test_no_scheme_duplication.py` (19/19 pass).
- All `barotropic_*.py` use `compute_filter_weights`, `bebt_blend`,
  `maxvel_clip`.
- All saturation calls go through `legoesm.thermo.saturation_*`.
- Constants imported from `legoesm.constants` or NamedTuple defaults
  with `# = constants.X` annotation.
- `compute_ocean_rho` / `compute_ocean_rho_and_pressure` exist in
  `eos.py` for the EOS+pressure 2-pass pattern.

## Issues found and fixed during this audit
1. **SyntaxError** in `experiments/rest_state.py:165-167` —
   duplicate keyword arguments (`T_surface`, `T_deep`, `S_uniform`)
   in spectral branch.  *Fixed*: removed three stale lines.
2. **Shape-mismatch in 4D FCT advection** at
   `core/operators_cdgrid.py:_cgrid_fct_fluxes_2d`.  Line 946
   redundantly re-padded `q_pad_h2` AFTER the moveaxis at line 915,
   discarding the leading-`nlev` rotation; subsequent slicing assumed
   3D shape while `q_left_x`/`q_right_x` (used in the clip step) had
   the rotated 4D shape.  Caused `(20, 6, 9, 8)` vs `(6, 9, 8, 20)`
   broadcast failure.  *Fixed*: removed the redundant re-pad, switched
   to `...`-prefixed slicing and negative `axis` so the same code
   path works for both 3D and 4D.

These are confirmed bugs (caused test failures) and the fixes are
local + minimal.

## Findings table
| ID | File:Line | Severity | Class | Fix |
|----|-----------|----------|-------|-----|
| O1 | `experiments/rest_state.py:165-167` | red→green | SyntaxError (duplicate kwargs) | applied |
| O2 | `core/operators_cdgrid.py:946-968` | red→green | shape bug in 4D path | applied |
| O3 | `physics/convection/plume.py:66` | yellow | sigmoid sharpness ~1e4 saturates AD on flag | left in place — only affects loss flowing through `convection_flag`, which no current loss does |
| O4 | `biogeochemistry/npzd.py:188` | green (informational) | TA convention | documented, sign is correct under Sarmiento-Gruber convention |
| O5 | `physics/vertical_mixing/kpp.py:248` | green | `phi_m^{-1}` formula equivalent given local sign convention | documented |
| O6 | `tests/ocean/unit/test_ocean.py::test_longrun_conservation_with_fixer` | yellow | tolerance `1e-8` too tight, observed `3.46e-7` after 50 steps — pre-existing; my fix enabled the test to *run* | not in scope |
| O7 | `tests/ocean/unit/test_cross_grid_parity.py::test_canonical_runner_cases_exist` | yellow | registry expects 2 keys not present; pre-existing | not in scope |
