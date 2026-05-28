# RRTMGP Faithfulness Audit — Tracking

Goal: ensure `src/legoesm/atmosphere/physics/radiation/rrtmgp/` matches
upstream `earth-system-radiation/rte-rrtmgp` semantics — especially in
the upper troposphere/stratosphere — while remaining JAX-differentiable
and supporting float32/float64/mixed precision and MPI+GPU scaling.

Upstream reference: https://github.com/earth-system-radiation/rte-rrtmgp
(commit hashes recorded per iteration below).

---

## Iter 1 — initial audit (2026-05-28)

### Files reviewed
- `optics/gas_optics.py` (major/minor optical depth, Rayleigh, Planck)
- `optics/optics_utils.py` (linear interpolant, lookup_values)
- `optics/optics.py` (`RRTMOptics.compute_planck_sources`)
- `optics/lookup_gas_optics_base.py` (table loading, `kmajor`, `vmr_ref`)
- `optics/lookup_gas_optics_longwave.py` (`t_planck`, `totplnk`)
- `rte/two_stream.py` (`solve_lw`, `solve_sw`, source combining)
- `rte/monochromatic_two_stream.py` (`lw_combine_sources`,
  `lw_cell_source_and_properties`)
- `rrtmgp.py` (`RRTMGP.solve_columns` legoESM entry point)
- Upstream: `rrtmgp/kernels/mo_gas_optics_rrtmgp_kernels.F90`,
  `rrtmgp/frontend/mo_gas_optics_rrtmgp.F90` (lines 300/303/519/1030/1354)

### Findings — passing match to upstream
- Pressure interpolation: log(p) into log(p_ref) with `+itropo-1` shift
  applied as `offset` arg of `create_linear_interpolant`. `kmajor` shape
  (n_t_ref, n_p_ref+1=60, n_eta, n_gpt) sized for the +1 shift. ✓
- Tropopause boundary `tropo_idx = (p <= p_ref_trop)` ↔ Fortran
  `merge(1,2,p>p_ref_trop)`. Indices 0=lower, 1=upper. ✓
- `col_mix` ≡ `combined_vmr * molecules_dry` ≡ `col_gas(igas_1) +
  ratio_eta_half * col_gas(igas_2)`. ✓
- Eta interpolation with `eta = vmr_1/combined_vmr` fallback 0.5. ✓
- Rayleigh: `(1 + vmr_h2o) * molecules_dry / 1e4 = col_dry + col_h2o`. ✓
- Planck-source combination at faces: `sqrt(pfrac[k]*pfrac[k+1]) *
  totplnk(T_face)` via `lw_combine_sources`. ✓
- `t_planck` linspace(temp_ref_min, temp_ref_max, n_t_plnk) matches
  upstream `totplnk_delta = (T_max-T_min)/(N-1)` and
  `temp_ref_min = temp_ref(1)` (frontend line 1354/1030). ✓
- Pressure halo extrapolation clamped to ≥1 Pa via
  `solve_columns:p_3d = jnp.clip(p_3d, 1.0, None)`. ✓

### Findings — bugs

**BUG-1 (critical, stratosphere)**: `compute_planck_sources` evaluates
`totplnk(T)` via `create_linear_interpolant(T, t_planck)`. When `T <
t_planck[0]=160K` the interpolant returns
`weight1*table[0] + weight2*table[1]` with `weight1 = 1 - weight2`,
`weight2 = |T - 160|/Δ` so `weight2` ≫ 1 and `weight1` ≪ 0. Linear
extrapolation of a function `~T^4` is catastrophic. Upstream raises an
error if any tlay/tlev/tsfc lies outside [temp_ref_min, temp_ref_max]
(frontend lines 300/303/519). Halo cells can be T<160K via
`top_halo = 2*T[-1] - T[-2]` with e.g. T[-1]=180, T[-2]=220 →
top_halo=140K.

Mitigation must remain differentiable (no error stop). Plan: clip the
temperature passed to `totplnk` interpolation into `[t_planck[0],
t_planck[-1]]`. The Planck source from a face outside the table range
will be evaluated at the boundary; acceptable since the halo cells are
stripped post-solve and the rare physical case of T<160K outside the
mesosphere is unlikely for tropospheric-stratospheric Earth models.

**BUG-2 (moderate, stratosphere)**: same out-of-range extrapolation
risk for `kmajor`, `kminor_*`, `krayl_*`, `planck_fraction` tables. Less
severe than `totplnk` (linear vs T^4) but still wrong. Apply the same
clip on temperature passed to those interpolants.

**BUG-3 (AD)**: `_compute_relative_abundance_interpolant` uses
`jnp.where(combined_vmr > 0, vmr_for_interp[0]/combined_vmr, 0.5)`.
`combined_vmr=0` causes NaN forward (passed through where) and NaN
backward through the dead branch. Upstream uses `if (col_mix > 2*tiny)
else 0.5` which is non-differentiable. Use safe-division:
`vmr/jnp.maximum(combined_vmr, eps)` then where-pick 0.5 fallback.

**BUG-4 (AD/numerical)**: `_compute_minor_optical_depth` does
`jax.lax.fori_loop(0, minor_absorber_intervals, body_fn, ...)` then
outer `jnp.where(p > p_ref_tropo, lower, upper)`. Both branches always
run. AD propagates through both — wasteful but functionally OK.
Inefficient: O(2*n_minor_absrb) per g-point. Defer optimisation;
correctness more important right now.

**BUG-5 (climatology)**: `_standard_o3_profile` =
`8e-6 * exp(-0.5*((log10(p/10hPa))/1.5)^2)` peaks at 8 ppm at p=10 hPa
with σ_log=1.5. Observed mid-latitude stratospheric ozone peaks at
~9-10 ppm at 10 hPa with FWHM≈0.6 dex (σ≈0.25). Current σ is too wide
→ too much O3 in upper stratosphere/lower mesosphere, too little near
the peak. Affects SW heating in the stratosphere. Driver should
provide an external O3 climatology; the fallback should still produce
a reasonable profile. Tighten the climatology and bias the peak
slightly higher. Out of scope for iter 1 — flagged for iter 2.

### Findings — non-bugs (verified OK on close reading)
- `_compute_minor_optical_depth` evaluates the relative-abundance
  interpolant with the cell's actual `tropo_idx`, not the per-call
  `is_lower_atmosphere` flag. In the dead branch (lower computed in
  stratosphere) this looks up `key_species[..., 1]` (upper) but the
  whole branch is masked by the outer `where(p > p_ref_tropo, ...)`.
  Gradient is masked the same way. ✓
- `_shift_up` / `_shift_down` use `jnp.roll` (wraps at boundaries) but
  the wrap hits halo cells which are stripped before the recurrence
  and after the solve. Interior values unaffected. ✓
- `molecules` in `_air_molecules_per_area` is dry-air molecules per m²
  (matches upstream `col_dry`). Confirmed by mass-balance derivation
  with mixed dry+water and consistent usage in major/minor/Rayleigh.
  ✓

### Plan for iter 1 fix
1. `gas_optics.py`: add `_clip_to_table_range(t, t_ref)` helper, apply
   in `compute_major_optical_depth`, `_compute_minor_optical_depth`,
   `compute_rayleigh_optical_depth`, `compute_planck_fraction`,
   `compute_planck_sources`.
2. `gas_optics.py`: safe-div in `_compute_relative_abundance_interpolant`.
3. Add `tests/atmosphere/.../test_rrtmgp_stratosphere_fidelity.py`
   exercising: (a) cold mesosphere column (T=150K aloft) does not
   produce NaN or absurd LW flux, (b) gradient w.r.t. T at TOA is
   finite, (c) gradient w.r.t. T at surface is finite, (d) mixed
   x32/x64 produce consistent fluxes within tolerance.

### Iter 1 status (2026-05-28)
- ✅ All 4 fixes landed in `src/.../rrtmgp/optics/gas_optics.py`.
- ✅ 14 new tests in `tests/atmosphere/hydrostatic/unit/test_rrtmgp_stratosphere.py` pass.
- ✅ 61 existing radiation tests in `test_radiation.py` continue to pass (no regression).
- ⚠️ Codex adversarial review hung after ~13 min in reasoning phase
  (last command at 18:46:14, no subsequent activity).  Cancelled and
  rolled into iter-2 review.

---

## Iter 2 — optimal LW diffusivity (2026-05-28)

### Findings carried from iter 1
- **BUG-5 (climatology)**: standard O3 profile σ_log=1.5 too wide, peak
  8 ppm at 10 hPa overestimates tropopause O3 by 5-10×.  Deferred.
- **Optimal angle missing**: upstream RRTMGP computes per-band,
  per-column LW diffusivity secant via `compute_optimal_angles` using
  the `optimal_angle_fit` polynomial coefficients shipped in the
  gas-optics netCDF.  legoESM hard-codes `_LW_DIFFUSIVE_FACTOR = 1.66`.

### Iter 2 changes — optimal LW diffusivity-angle plumbing
- `lookup_gas_optics_longwave.py`: load `optimal_angle_fit` ``(n_bnd, 2)``
  from netCDF; tolerate older files via ``tables.get(...)`` → ``None``.
- `monochromatic_two_stream.py:lw_cell_source_and_properties`: accept
  optional ``lw_diffusive_factor`` kwarg (Array or scalar) replacing the
  hard-coded ``_LW_DIFFUSIVE_FACTOR``.
- `two_stream.py`: new `_compute_optimal_lw_secant(tau, band_idx,
  optimal_angle_fit, halo_width=1)` matching upstream formula
  ``c0 * exp(-sum_z tau) + c1``.  Halo-aware sum.
- `two_stream.py:solve_lw`: opt-in via ``use_optimal_angle: bool = False``
  kwarg; pulls table from ``optics_lib.gas_optics_lw.optimal_angle_fit``.
- `config.py:RRTMGPConfig`: add ``use_optimal_angle: bool = False`` field.
- `rrtmgp.py:RRTMGP.solve_columns``: forward
  ``config.use_optimal_angle`` to ``solve_lw``.
- `rrtmgp.py:_instance_cache_key``: include ``use_optimal_angle`` so
  flipping the flag does not silently reuse the wrong cached solver
  (same defensive pattern as ``use_scan`` in iter-1090).

### Iter 2 tests
(``test_rrtmgp_stratosphere.py::TestOptimalLwSecant``)
- ``test_optimal_angle_fit_loaded``: shipped netCDF has the field.
- ``test_secant_formula_matches_upstream``: verifies
  ``c0*exp(-tau_total)+c1`` to 1e-12 rel tol.
- ``test_secant_grad_finite``: jax.grad propagates without NaN.
- ``test_solve_lw_optimal_angle_smoke``: end-to-end ``solve_columns``
  with ``use_optimal_angle=True`` differs from baseline by > 1e-3,
  guarding against silently-no-op plumbing.

### Iter 2 status (initial)
- ✅ All 4 new optimal-angle tests pass.
- ✅ All 23 RRTMGP-relevant tests pass (iter-1 + iter-2 combined).
- ✅ Opt-in: existing config consumers see no behavior change.

### Iter 2.5 — codex adversarial review (2026-05-28)
Codex fresh-thread review (after iter-1 review hung) flagged:
- **MEDIUM**: ``solve_lw(use_optimal_angle=True)`` silently degraded to
  fixed 1.66 when the netCDF file lacked ``optimal_angle_fit``,
  contradicting the config contract.
- **LOW**: ``_compute_optimal_lw_secant`` clamps ``tau`` to ``>= 0``
  before summing — a deliberate guard but undocumented.
- Coverage gaps: no multi-column secant test, no end-to-end AD test
  through ``solve_columns(use_optimal_angle=True)``, no
  ``optimal_angle_fit=None`` error-path test.

Fixes landed same iter:
- `two_stream.py:solve_lw`: raise ``ValueError`` when
  ``use_optimal_angle=True`` and ``optimal_angle_fit is None`` with a
  clear "upgrade the data file or set use_optimal_angle=False" message.
- `_compute_optimal_lw_secant`: explicit docstring paragraph on the
  ``jnp.maximum(tau, 0.0)`` clamp and how it differs from upstream.
- New tests:
  - ``test_raises_when_optimal_angle_fit_missing`` (covers FIX-1).
  - ``test_optimal_angle_per_column_secants_differ`` (multi-column
    broadcasting against ``(ncol, 1, nlev+2)``).
  - ``test_solve_columns_use_optimal_angle_differentiable``
    (reverse-mode AD through the full solver).

Verdict: VERDICT: FIX → addressed; ready to ship iter-2.

### Final iter-2 status
- ✅ 21 RRTMGP-stratosphere tests pass.
- ✅ 82 radiation tests pass (iter-1 + iter-2 + codex fixes; 2 skipped
  are multidevice MPI tests needing >1 visible JAX device).
- ✅ Zero regressions vs. iter-1 baseline.

### Deferred to iter 3+
- Enable ``use_optimal_angle=True`` by default after validating against
  upstream RFMIP reference fluxes.
- Tighten ``_standard_o3_profile`` Gaussian width (σ_log 1.5 → ~0.9 in
  natural log) and document recommended external profile.
- Replicate-boundary halo for T/q_v in ``solve_columns`` to remove the
  out-of-range temperature path entirely.

---
