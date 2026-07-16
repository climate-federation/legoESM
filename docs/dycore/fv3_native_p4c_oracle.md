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

## P4c battery results — C36 Williamson-2 imprint ladder (2026-07-15)

Run by `scripts/cluster/fv3_native/sw_imprint_battery.sbatch` (job
9035783).  Williamson-2 has analytic `v == 0` everywhere, so the
lat-lon-regridded `v_ll_Linf` is a pure grid-imprint metric.  Wiring
landed via `--fv3-native-grid` / `--fv3-native-angles` (matrix runner) +
the FB-lane ED build (`_fb_cube_sw_model`) + `FV3FBShallowWaterModel(...,
fv3_native_angles=)`.

| row | solver | grid            | v_ll_Linf | L2       |
|-----|--------|-----------------|-----------|----------|
| A   | A-L    | equiangular,noduo | **0.54**  | 4.68e-04 |
| B   | A-L    | ED + duo        | 22.5      | 4.25e-02 |
| C   | FB     | equiangular,duo | 43.8      | 1.54e-02 |
| D   | FB     | ED + duo        | 58.9      | 1.48e-02 |
| E   | FB     | ED + duo + native angles | **NaN (blew up)** | nan |
| —   | ref    | lat-lon         | —         | 1.41e-03 |
| —   | ref    | MPAS (ico5)     | —         | 1.25e-04 |

Single-factor transitions (only these isolate one variable): C→D = ED
metrics (imprint 43.8→58.9, WORSE); D→E = native seam angles (→ blow-up).

**Findings (largely negative for the native-grid FB path as assembled):**

1. legoESM's **tuned production A-L solver on the equiangular grid (row A)
   is already the best cube and is close to lat-lon / MPAS** (L2 4.68e-4 vs
   lat-lon 1.41e-3, MPAS 1.25e-4).  The science goal — cube W2 close to
   other grids — is met on the *production* configuration.
2. The FV3-native **ED grid does NOT reduce imprint under either solver**
   at C36 with these configs: A-L 0.54→22.5, FB 43.8→58.9.  The A-L config
   is tuned for the equiangular discretisation and does not transfer
   (`cubed_sphere_cdgrid.py:639`).
3. The **FB core is ~80× worse than the tuned A-L solver** on W2 imprint
   even on the equiangular grid (43.8 vs 0.54) — the M1 preset is a
   coarse, un-tuned damping, not a calibrated production config.
4. **Native seam angles blow the FB core up (row E → NaN).**  This is the
   matched-pair problem: FV3 uses the native cross-face seam angles
   *together with* the faithful d_sw5 cross-face halo.  legoESM has the
   native angles but the FB `d_sw5` still runs the *stable zero-ring*
   approximation (the faithful attenuated ghost itself destabilises the
   modon run; `fv3_sw_core.py:3205`).  Native angles without the matching
   faithful halo is inconsistent → instability.

**Consequence for "full FV3 faithful portability":** the *kernels* are
certified bit-exact (c_sw / d_sw / divergence_corner_duo), and the
*production* cube already meets the science goal on the equiangular grid.
But the *assembled native-grid FB path* (ED + native angles + faithful
cross-face halo) is NOT yet consistent — row E blow-up localises the gap
to the native-angle ⇄ faithful-d_sw5-halo pairing.  The next brick is to
port that matched pair together and stabilise it, rather than swapping the
grid under the A-L solver (which is not FV3's scheme and is tuned for
equiangular).
