# Adversarial review request — Volcanic LW aerosol fix (LegoESM)

You are an independent adversarial physics reviewer. Find every bug, sign
error, unit inconsistency, broken-gradient pattern, conservation violation, and
test-case discrepancy in this implementation. Cite file:line. If you believe
there are no bugs, say so and explain why each candidate concern is not one.
The full working tree is available (cd is the repo root); read any file you
want. The unified diff of this change is at
`.physics-validator/volcanic_lw_fix/fix.diff` and pasted at the end.

## Context — what this fixes

A prior review confirmed 3 sub-defects + 1 gate defect in how the CMIP6
volcanic stratospheric LW aerosol file
`bc_aeropt_cmip6_volc_lw_b16_sw_b14_*.nc` is applied on the standalone
(MPAS/spectral) radiation path in `packages/coupler/.../model_driver.py`
(`_precompute_external_forcing`), which feeds RRTMGP's LW **absorption**
optical-depth slot (`solve_lw(..., aerosol_absorption_optical_depth=...)`,
`rrtmgp.py:599-605`, `rte/two_stream.py:420-426`).

The physical file (verified via xarray) is:
- dims: `terrestrial_bands=16`, `solar_bands=14`, `latitude=36`,
  `altitude=70` (5.0–39.5 km, 0.5 km steps), `month=12`.
- vars: `ext_earth(band,lat,alt,month)` [1/km] extinction, `omega_earth`
  (LW single-scattering albedo; min 1.8e-5, **max 0.40**, mean 0.022),
  `g_earth`, `wl1_earth`/`wl2_earth` (band bounds; the 16 bands span 40–3250
  cm⁻¹ in increasing-wavenumber order = the RRTMGP LW band set).

### The 4 defects and how each is addressed
- **B1 extinction-as-absorption.** The old loader integrated raw `ext_earth`
  and fed it to the LW **absorption** slot (which docstring-forbids
  extinction). Fix: the new loader multiplies by `(1 - omega_earth)` per band
  → true absorption. The file provides `omega_earth`, so this is exact, not
  the ω≈0 approximation. (If a file lacks `omega_earth`, we fall back to
  ω=0 = ext≡abs, documented.)
- **B2 vertical mis-placement.** File IS height-resolved. Old code collapsed
  altitude to a single column AOD then spread by `dp/Σdp` (full-column
  pressure mass) → ~90 % landed in the troposphere. Fix: the loader retains
  the per-file-layer profile, tags each layer with a US-Std-Atm-1976
  pressure edge, and a new conservative pressure-overlap remap
  (`place_stratospheric_aod_profile_to_layers`) bins it onto model layers.
- **B3 spectral collapse.** Old code `nanmean`'d the 16 bands to one gray AOD
  applied to every g-point. We KEEP gray (the LW slot is per-layer, not
  per-band; threading per-band τ through the g-point scan for a
  background-magnitude term with band-order-matching risk was judged
  disproportionate), but replace the flat `nanmean` with a
  **Planck-emission-weighted** band mean at 220 K — the correct gray
  equivalent. Verified: normalized band weights concentrate on far-IR bands
  0–2 (~63 %) and give ~0 to near-IR bands 12–15 (a flat mean gives 6.25 %
  each). Residual per-band spread is the documented approximation.
- **Gate.** `model_driver.py` `_ext_forcing` (two sites: ~6170, ~7126) gated
  on ozone/aerosol/ghg/experiment but omitted `_aerosol_lw_active`, so a
  volcanic-LW-only run never built the forcing dict → LW aerosol never
  reached radiation. Fix: added `or self._aerosol_lw_active` at both sites.

`distribute_column_aod_to_layers` (the old helper) is UNCHANGED and still used
by the SW path (line ~2401) — only the LW path switched to the new helper.
Confirmed only caller of `get_aerosol_lw_at_time` is model_driver.py:2415, so
its return-contract change (now `(profile, p_edges)`) is safe.

## Static analysis summary

- **Units.** `ext_earth` [1/km] · `(1-ω)` [-] · `dz` [km] = OD [-]. Pressure
  everywhere [Pa]. `_ussa1976_pressure` uses `constants.g`, `constants.R_d`.
  Planck weights use `constants.h_planck/c_light/k_B`; band bounds µm→m→1/m.
  No hardcoded physical constants in function bodies (US-Std-Atm table and
  strat temp are module-level `_UPPER_SNAKE` with provenance comments).
- **Signs.** OD ≥ 0 (clipped). Placement conserves column OD within the model
  pressure range. `_ussa1976_pressure` monotone-decreasing in height
  (asserted). Test case: stratospheric LW-absorbing aerosol → ΔOLR(TOA) < 0
  (LW trapping / warming) — correct sign.
- **JAX purity / diff.** The loader is host-side (numpy, outside JIT, per-day,
  lru_cached). The placement helper is pure `jnp` (min/max/clip/cumsum/
  searchsorted/take_along_axis/einsum-free), no Python control flow on traced
  values, static shapes → differentiable in both `aod_profile` and
  `p_half_col` and JIT/vmap-safe (tested).
- **Conservation.** Placement = difference of a piecewise-linear cumulative-OD
  CDF sampled at model half levels = exact fractional-pressure-overlap remap;
  Σ_layers = total source OD when the model column spans the source range
  (tested to rtol 1e-6). OD outside the model pressure range is dropped
  (cannot be represented) — intentional.
- **Limiters.** `clip(...,0,None)` on OD (positivity), `clip(p1-p0,1e-12,None)`
  and `clip(dp,1e-12,None)` (divide-by-zero floors), `clip(frac,0,1)`
  (interp clamp outside source range). None create dead-gradient regions in
  the physically active band.

## Differentiability + tests (all pass; CPU, JAX_ENABLE_X64=1)

`tests/unit/test_volcanic_lw_aerosol.py` — 14 passed:
- USSA z→p known values (101325/22632/5475/868 Pa) + monotonic.
- Planck weights: far-IR ≫ near-IR at 220 K; not a flat mean.
- B1 synthetic-violation: synthetic file with ω=0 vs ω=0.3 → absorption column
  = 0.7×extinction column (fails if `(1-ω)` dropped).
- p_edges ascending & aligned to profile.
- B2 placement: >80 % OD above 100 hPa on a standard column + conservation
  to rtol 1e-6; contrast: old helper leaves <15 % stratospheric.
- Differentiability: `jax.grad` w.r.t. `p_half_col` matches centered FD
  (delta 1e-4), no NaN; jit and vmap match eager.
- Real-file end-to-end (1979): deep-tropical column >80 % above 100 hPa;
  all-latitude OD-weighted centroid < 150 hPa (stratospheric). The loader is
  verified faithful to the raw file — the extratropical aerosol genuinely
  reaches 14 km (141 hPa) at 30N because the tropopause is lower there, so a
  fixed 100 hPa line (a tropical proxy) correctly lets some 30N OD sit at
  100–180 hPa.

Collateral (no regressions): `test_convection_config_for.py` 6/6;
`test_bechtold_column_conservation.py` 3/3.

## Test case — OLR impact on a standard tropical column (1979)

RRTMGP `solve_columns` LW, 50-level tropical column (300 K surface):
- baseline OLR(TOA) = 271.308 W/m²
- FIX (correctly-placed absorption, col OD 1.34e-4): ΔOLR = −0.040 W/m²
- OLD (raw extinction spread by pressure mass, col OD 2.57e-4): ΔOLR = −0.029 W/m²
- fix − old = −0.011 W/m²

1979 is a background year → the fix is **immaterial** in absolute terms
(< 0.05 W/m²). Note the fix yields a *larger* |ΔOLR| per unit OD despite
smaller OD, because it places absorption in the cold (220 K) stratosphere
where the emission-vs-absorption temperature contrast is large — the correct
behavior that becomes materially important (~1–4 W/m²) for a major eruption
(Pinatubo AOD ~100× larger).

## Specific things to attack
1. Is the pressure-overlap remap (`place_stratospheric_aod_profile_to_layers`)
   truly conservative and correctly oriented for BOTH ascending model
   half-levels and the ascending source `p_edges`? Any off-by-one in the CDF
   alignment (`cum` has nsrc+1 entries aligned to `pe`)?
2. Is the US-Std-Atm z→p (`_ussa1976_pressure`) geopotential-vs-geometric or
   g/R choice a real error at 5–40 km, or negligible for placement?
3. Is Planck-weighted gray defensible, or is per-band mandatory? (I have the
   per-band spread as the documented error; the LW slot is per-layer gray.)
4. Any differentiability trap in `searchsorted`/`take_along_axis`?
5. Did I miss any `_ext_forcing`-style gate or any other caller/shape contract?
6. SW volcanic path: it shares the same collapse+spread defect but mixes
   volcanic+tropospheric AOD into one column before distribution — I filed it
   as a residual (see report), not fixed. Agree it's out of cheap scope?

## Full unified diff

See `.physics-validator/volcanic_lw_fix/fix.diff` (810 lines). Key new symbols:
`external.py::_ussa1976_pressure`, `_planck_band_weights`,
`_load_volcanic_lw_absorption_profile`, reworked `get_aerosol_lw_at_time`;
`surface_utils.py::place_stratospheric_aod_profile_to_layers`; driver wiring +
gate at `model_driver.py`.
