# Cubed-Sphere Ocean PGF Stability (Open Issue)

## Summary

The cubed-sphere `OceanModel` exhibits a slow exponential instability
whenever the horizontal density field develops gradients of any
amplitude.  Under the rest-state + WOA-restoring OMIP smoke test
(`scripts/run_omip.py --grid cubed_sphere --quick`), the run blows up
around physical day 4-5 with the current defaults; pre-tuning defaults
(2026-05-19), blow-up was at ~2 days.

The instability is **not** present on `latlon`, `mpas`, or `spectral`
ocean models — those all reach the 30-day quick target.

## Symptom

Step-by-step trace on a 24-cell, 20-level config with `A_h=5e5`,
`K_h=5e6`, `n_barotropic_substeps=60`, `barotropic_diffusion_alpha=0.3`,
`tau_restore=3650 d`, `restoring_ramp=14 d`:

| step | day  | max\|u\| (m/s) | max\|eta\| (m) | notes |
|------|------|----------------|----------------|-------|
| 0    | 0.0  | 0              | 0              | rest state |
| 1500 | 1.0  | 2.5e-6         | 6.0e-6         | tiny noise floor |
| 3000 | 2.1  | 9.5e-3         | 6.4e-3         | exponential ramp begins |
| 5000 | 3.5  | 5e-2           | 6e-2           | still growing |
| 15000| 10.4 | 10.6           | 3.7            | unphysical |
| 17437| 12.1 | NaN            | NaN            | cascade |

Spatially the first non-finite cell sits at the cube **face boundary**
(j ∈ {0, 1, n-2, n-1}) and the velocity hot spots cluster on the same
edges through the entire run, identifying the artifact as a face-edge
halo / PGF interplay rather than a uniform numerical instability.

## Root cause (hypothesis)

The cubed-sphere `ocean_baroclinic_tendencies_cdgrid` PGF reads
``p_prime`` through the cubed-sphere halo interpolator
(`pad_halo_4d` + `halo_interp_offsets`), which is O(Δx) accurate at
face boundaries.  Under any horizontal density gradient the
halo-interpolation error injects an O(Δx) artifact into ``∇p``, which
is exactly the term the barotropic substep amplifies via the
free-surface ↔ ``U_bar`` feedback.  The PGF lives on cells just
inside the face boundary so the artifact has nowhere to dissipate
laterally and the only available sink (Laplacian + biharmonic
viscosity) is too weak to compete with the gravity-wave-fast
amplification.

The latlon C-grid does not see this because its halo is exact
(periodic in longitude, fold at the pole); the MPAS path uses
unstructured cells (no faces); spectral has no real-space face
boundaries at all.

## Mitigations applied so far

The defaults landed on 2026-05-19 buy ~5 days of stable integration
on the 30-day quick smoke run (~2× the pre-tuning baseline) without
masking the underlying bug:

| knob | old | new (cubed_sphere only) |
|------|-----|-------------------------|
| `dt` | 300 s | **60 s** |
| `A_h` | 1e5 m²/s | **5e5 m²/s** |
| `K_h` | 1e5 m²/s | **5e6 m²/s** |
| `n_barotropic_substeps` | 30 | **60** |
| `barotropic_diffusion_alpha` | 0.05 | **0.3** |
| `restoring_tau_days` | 1095 | **3650** |
| `restoring_ramp_days` | 0 | **14** |

These were chosen to maximise the survival time without altering the
target restoring climatology.

## What does NOT help

These were tested and gave only marginal improvements (delaying
blow-up by a few hundred steps each):

* `hyperdiff_coeff` (Laplacian or biharmonic) up to 1e16.
* `div_damp_2`, `div_damp_4` up to 1e17.
* `barotropic_staggering="c_grid"` (made it strictly worse).
* `use_duogrid=True` on the cubed-sphere halo.
* Restoring across the entire column instead of surface only
  (faster blow-up because it drives larger 3-D gradients).
* Initialising from WOA directly (blows up faster — initial gradient
  is larger than the rest-state + slow build-up case).
* Bumping `A_h` and `K_h` to 1e7 and beyond (still blows up; ice-
  free polar caps go unphysical first).
* Post-step Laplacian smoothing on T, S, u, v.
* Post-step clipping of u, v, eta — works to prevent NaN but saturates
  the fields at the clip values, making output physically meaningless.

## What WILL fix it (planned)

The structural fix is to swap the cubed-sphere baroclinic PGF for an
SMC03-style density-Jacobian formulation that already exists for the
latlon C-grid (`pgf_scheme="smc03"` in `LatLonCGridOceanConfig`).
That requires:

1. Port the SMC03 density-Jacobian gradient to the cubed-sphere C-D
   grid (the Arakawa-Lamb metric stencil already works on tensor
   fields; the inner Jacobian needs adaptation).
2. Add a duogrid halo on T, S (not just on velocities) so the
   halo-interpolated density is O(Δx²) instead of O(Δx).
3. Add a face-edge boundary correction analogous to
   `_extrapolate_boundary_corners` (already applied to ``du_dt`` /
   ``dv_dt``) for the pressure-gradient term itself.

Estimated effort: a focused 2-3 day rework with proper benchmark
validation (Williamson-2-like + rest-state + WOA restoring).

## Production guidance

Until the structural fix lands, **do not use the cubed-sphere ocean
for multi-day OMIP integrations.**  The startup warning printed by
`scripts/run_omip.py` flags this on every invocation.

The other three grids (`latlon`, `mpas`, `spectral`) are stable for
the 30-day quick run and the latlon path has demonstrated 50+ year
stability with realistic ETOPO bathymetry.
