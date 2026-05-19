# Cubed-Sphere Ocean PGF Stability (RESOLVED — 2026-05-20)

## Status

**Fixed.**  Routing cubed-sphere baroclinic tendencies through the
FC-Gram spectral operator path (`ocean_pe_fc.ocean_baroclinic_
tendencies_fc`) via the `OceanModel(..., fc_config=...)` constructor
kwarg removes the face-edge halo amplification described below.
`scripts/run_omip.py --grid cubed_sphere --quick` now completes the
30-day smoke run with `max_speed ≈ 3e-5 m/s` and `SST ≈ 19.63`,
fully equivalent to the latlon, MPAS, and spectral grids.  The script
enables FC by default for cubed_sphere (see `_create_setup`).

## Summary (original failure mode)

The cubed-sphere `OceanModel`'s default cd-grid path used the
Arakawa-Lamb 4-pt D-grid corner gradient for the baroclinic
pressure-gradient force.  At face boundaries the A-L stencil reads
halo-interpolated neighbour-face values; the off-diagonal entries of
the 2×2 Cartesian metric matrix (`c01`, `c10`) amplify the O(dx²)
halo-interp error to an O(dx) PGF error.  Under any horizontal
density gradient (rest state + WOA restoring, or full WOA init), the
spurious face-edge PGF feeds a barotropic free-surface ↔ U_bar
feedback that doubles the velocity every ~25 steps and reaches NaN
around physical day 2 with legacy defaults, day 4-5 with tuned
defaults.

The instability was **not** present on `latlon`, `mpas`, or
`spectral` ocean models — those use their own model implementations
without the cubed-sphere face-edge halo path.

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

## What fixed it

Routing the baroclinic-tendency computation through the FC-Gram
spectral operator path (`legoesm.ocean.dynamics.ocean_pe_fc.
ocean_baroclinic_tendencies_fc`).  FC-Gram (Fourier continuation)
evaluates horizontal gradients spectrally on each cube face with a
C∞-smooth extension into the halo region.  This:

* Eliminates the A-L 4-pt corner stencil entirely (no off-diagonal
  Cartesian-metric amplification of halo errors).
* Replaces O(dx²) Lagrange halo interpolation with the FC extension,
  whose effective accuracy near the boundary is set by the FC
  polynomial degree (default 5) rather than the corner-stencil
  width.
* Keeps the rest of the ocean PE pipeline (EOS + hydrostatic
  pressure, layer thickness, w diagnosis, skew-symmetric momentum,
  vertical advection / diffusion, land masking) bit-identical to
  the cd-grid path.

The fix is wired through `OceanModel(..., fc_config=...)`; pass an
`FCOperatorConfig` (build with `legoesm.core.operators_fc.
build_fc_config`) to activate.  Passing `fc_config=None` (default)
keeps the legacy cd-grid path for back-compat.  `scripts/run_omip.py`
builds and threads `fc_config` automatically for `--grid
cubed_sphere`.

## What does NOT help (preserved for posterity)

Before the FC route was identified, the following were tested and
each gave only marginal improvement on the cd-grid path:

* `hyperdiff_coeff` (Laplacian or biharmonic) up to 1e16.
* `div_damp_2`, `div_damp_4` up to 1e17.
* `barotropic_staggering="c_grid"` (made it strictly worse).
* `use_duogrid=True` on the cubed-sphere halo.
* Restoring across the entire column instead of surface only.
* Initialising from WOA directly.
* Bumping `A_h` and `K_h` to 1e7 and beyond.
* Post-step Laplacian smoothing on T, S, u, v.
* Post-step clipping of u, v, eta (band-aid; saturates physics).
* `fortran_a2b_corner_avg` and `fortran_dir_aware_corners` flags on
  `_arakawa_lamb_gradient` (corner-only fix; the artifact also
  lives along the cube edges, not just at the four vertices).

## Production guidance

All four ocean grids (`cubed_sphere`, `latlon`, `mpas`, `spectral`)
are now stable for the 30-day OMIP quick run.  The cubed-sphere FC
backend is ~3× slower per timestep than the cd-grid path on small
configs (per-face FFT cost), which combined with the conservative
`dt=60s` default makes the C24 quick run take ~10 min wall time.
For multi-year integrations on cubed-sphere, the FC backend is the
recommended path; the cd-grid path is preserved as the default
backwards-compatible mode for atmosphere-equivalent dynamics work.
