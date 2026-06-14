# Overflow on cubed-sphere: spurious reference-PE gain (non-conservation)

## Symptom
`scripts/matrix/run_ocean_test_matrix.py --only overflow --grid cubed_sphere`
fails the documented `pe_rel_final < 0` gate: the reference potential energy
(RPE) *increases* instead of decreasing as the dense plume descends.

The drift grows monotonically with integration time (not noise):

| duration | PE_rel_final | T range (init [0, 20] °C) |
|----------|--------------|---------------------------|
| 0.10 d (quick) | +6.47e-7 | — |
| 0.50 d (full)  | +4.48e-6 | [0.00, 20.04] |
| 1.00 d         | +1.04e-5 | [0.00, 20.10] |

`latlon` overflow passes (RPE decreases, ~−5.7e-8) and shows **no** tracer
overshoot. The cube **overshoots** the initial temperature maximum
(T_max → 20.04 → 20.10 °C), which is the tell-tale of a non-monotone /
non-conservative tracer transport: spurious extrema raise the sorted-density
RPE.

## Root cause (confirmed)
Cube-edge / partial-cell tracer-transport conservation, not a physics-scheme
bug. Two compounding mechanisms, both on the sloped bathymetry the overflow
case uses:

1. **Partial-cell spurious cross-face flux** — documented limitation in
   `packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py:262-275`: zeroing
   the A-cell velocity does not *strictly* close the C-grid face between an
   active cell and a below-seafloor (or coastline) cell; the dgrid→cgrid
   averaging leaves a small nonzero face velocity, so
   `cgrid_mass_flux_divergence` / `cgrid_tracer_advection_fct` carry a small
   spurious flux across the seafloor step. On a slope this injects a steady
   spurious tracer flux → growing RPE.
2. **Vertical tracer advection** (`_vertical_advection_ocean`) of the
   descending plume contributes over/undershoots where the slope forces strong
   vertical motion.

This is squarely "assume edge+metric errors first" for the cubed sphere.

## Fix path (deferred — substantial, needs visual + MPI validation)
The conservation upgrade is already named in
`docs/md_files/ocean_faithfulness_nemo.md`: a **strict C-face wet/rock mask**
(a face is active iff BOTH adjacent A-cells are active, with a cross-seam
`is_active` halo) that closes coastline + seafloor faces together. This is the
correct fix and would remove the spurious cross-face flux.

It is intentionally NOT applied here because:
- It changes tracer transport for **every** cube ocean case with bathymetry
  (broad blast radius; all currently pass).
- Per CLAUDE.md, cube-edge transport changes MUST be visually verified
  (cube-imprint / edge artifacts are not caught by norms alone) and validated
  across serial → MPI-sharded with the cross-seam `is_active` halo.

Until then the `overflow/cubed_sphere` matrix case is a **known, tracked**
non-conservation, distinct from a regression. The `latlon` overflow remains
the conservation-faithful reference.

## Reproduce
```
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
  --only overflow --grid cubed_sphere --days 1.0
```
Watch `PE_rel` grow positive and `T_max` exceed 20 °C.
