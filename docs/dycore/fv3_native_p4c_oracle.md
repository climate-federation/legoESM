# Phase-4c oracle: authoritative Mouallem/Xi-Chen duo-grid FV3 (Zenodo 8327578)

Phases 4a (c_sw) and 4b (d_sw) certified legoESM's loop-faithful reference
field-by-field against a VERBATIM Fortran extraction from
`GFDL_atmos_cubed_sphere @ 6f658bd0` — the **plain** (non-duo) FV3.  Phase 4c
re-targets the oracle to the **authoritative duo-grid FV3** the campaign
actually implements, and adds the cross-grid grid-imprinting battery.

## The oracle (Zenodo 8327578)

Mouallem 2023 (JAMES), *"Implementation of the novel Duo-Grid in GFDL's
FV3 — Code and simulations files"* (3.6 GB), fetched to
`/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo/` (job 9033356).
Contents:

- `atmos_cubed_sphere-symmetryclean.zip` → the **definitive duo-grid FV3
  source** (extracted to `Code/FV3/duogrid_symmetryclean/`).  This is the
  `symmetryclean` tree memory flagged as the authoritative provenance
  (Xi Chen, Princeton — the FV3 duo-grid author).
- Idealized simulation reference outputs, one dir per case, each with
  `rundir/atmos_daily.nc` (reference solution) + `input.nml` (config):
  - `C{48,96,192,384,768}.sw.case2.alpha{0,45}[.duo].hord{5,6,8,10}` —
    **Williamson-2** (solid-body rotation), duo vs non-duo, the primary
    grid-imprinting case.
  - `.sw.case{5,6,8}.*` — further shallow-water cases.
  - `.nh.case-13.*` — 3-D baroclinic wave.

## Re-pin findings (authoritative vs what phases 3-4 pinned)

Established this phase (`reconcile_production_csw.py`, sbatch on `glab`):

1. **Production grid geometry is BIT-EXACT to the certified builder.** The
   JAX `create_cubed_sphere_cdgrid(create_fv3_native_cubed_sphere(...))`
   tile-1 `area` matches `build_fv3_native_gridstruct` to 8.6e-14 (face 3 /
   rot90=1); `sin_sg`/`sina_cell` agree to 2.7e-8 (cdgrid double vs the
   certified longdouble `compute_fv3_native_angles` — sub-1e-7, not
   bit-exact; follow-up: adopt the longdouble angle build in the cdgrid).
   `cosa_cell` differs only by the rot90-induced sign flip of the signed
   non-orthogonality angle (a comparison artifact, not a grid difference).

2. **The duo-grid c_sw/d_sw DIFFER from the plain-FV3 6f658bd0 extraction.**
   The authoritative c_sw adds `flagstruct%duogrid` branches: it calls
   `divergence_corner_duo` (not `divergence_corner`), **skips**
   `fill2/fill_4corners` (the duo halos carry real cross-face data), and
   takes the `bounded_domain .or. grid_type>=3 .or. duogrid` KE/vorticity
   branch — i.e. **no `sin_sg` panel-edge special-casing**, because the
   duo-grid supplies genuine cross-face edge winds.  legoESM's PRODUCTION
   `fv3_sw_core._divergence_corner_duo` already ports this
   (sw_core.F90:2345-2447), so the production solver is duo-grid; the
   phase-4a/4b certified *reference* implements the plain branch.  Full
   duo-grid fidelity requires certifying the duo branches.

3. **The authoritative `fv_duogrid.F90` / `global_grid_gen_k2e.F90` differ
   from the luanfs mirror phase-3 pinned against** (fv_duogrid 212 lines —
   largely the mirror inlining geometry helpers the authoritative imports;
   `global_grid_gen_k2e.F90` (775 ln, Xi Chen) is a purpose-built k2e
   generator vs the mirror's monolithic `global_grid.F90` (2461 ln), same
   six-stagger structure `get_loc_{x,y}[_b,_c_x,_c_y,_d_x,_d_y]`).  Same
   author, same method — phase-3's k2e tables are expected numerically
   faithful, to be re-verified against the definitive generator.

## Remaining P4c work (loop, each gated by codex review)

- **Re-verify phase-3 k2e** against `global_grid_gen_k2e.F90` (compile,
  dump tables, compare to `compute_fv3_native_k2e`).
- **Certify the duo branches** of c_sw/d_sw (`divergence_corner_duo`, the
  no-corner-fill + no-edge-special-case paths) — the branch the production
  solver actually runs — as a duo-grid one-step oracle.
- **Grid-imprinting battery**: run legoESM's FV3-native duo-grid SW dycore
  on Williamson-2 (C48, alpha0/45, hord6/8) and compare to Mouallem's
  `atmos_daily.nc`; check the cube imprint is absent (the whole point of
  the duo grid) and that it stays close to lat-lon / MPAS.  Then W5/modon
  and Held-Suarez.
