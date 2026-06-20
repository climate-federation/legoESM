# #480 — WENO vector-invariant eddy-permitting blow-up: a free-slip-wall grid mode

## Symptom
The lat-lon C-grid WENO **vector-invariant** ocean momentum (`momentum_advection="weno9"`,
W9V) NaNs at ~**day 11** on the eddy-permitting Silvestri §5 baroclinic jet
(160×128×50, no closure, `stabilize=False`). Coarser/closure runs are fine.

## Root cause (diagnosed)
The blow-up is a **2Δx-in-lon, v-dominant, ROTATIONAL grid mode that grows only in the
first/last ~8 N/S free-slip-wall rows** (the interior stays clean):

- Mode location (`scripts/tmp/_silvestri_mode_location.py`): wall/interior amplitude
  ratio 2 → 7 → 73 over days 3/6/9; the interior 2Δx-v is dead-flat.
- Source (`_silvestri_wall_budget.py`): the **pressure gradient** does positive work on
  the grid-scale v at the wall (`KE_PGF`/PGF wall production +9.6e-5 vs interior 1.7e-10);
  the coupled grid-scale **density** grows there too (`_silvestri_T_location.py`).
- It is **un-dissipatable by the advection scheme**: the mode has ~zero zonal velocity,
  so meridional WENO upwinding (large v) only damps lat-structure, and the vorticity flux
  that would damp the lon-mode is multiplied by an x-mass-flux ≈ 0. Confirmed: pure
  1st-order-upwind boundary order-reduction only delays the blow (day 11→13).
- The Oceananigans oracle (same WENO scheme, same free-slip BC, same graceful boundary
  order-reduction, run in **both z and z\***) stays bounded at ~0.06 — it never *sources*
  the mode.

### Ruled out as the cause (with the method)
WENO scheme (tendency-bridge `bridge_momentum.jl`: legoESM 0.23 vs Oceananigans 0.248
grid-scale dissipation — MATCH); vertical resolution (50-level oracle saturates); EOS
(linear, matches); tracer advection (WENO7 all dirs); grad/curl operators
(`curl(grad)=0` to machine precision); spherical metric (the meridional PGF `−dp_dy` uses
`dy=R·dlat`, no cos-lat); **z\*** (Oceananigans-in-z\* stays bounded → innocent; legoESM's
baroclinic `p_prime` uses static `dz_ref`, not the z\* Jacobian); Coriolis (cancels only
~6% of the wall PGF, but that holds everywhere for a v-dominant mode, not a wall-specific
failure).

### The remaining (un-isolated) faithful root
The only structural difference left is legoESM's **masked-land-row wall** (with Neumann
fills of pressure/vorticity/density at land-adjacent cells) vs Oceananigans' clean
**grid-edge wall**. Isolating exactly which term this changes needs a **wall tendency-bridge**
(feed one identical grid-scale-perturbed wall state into both codes, diff the
density+momentum tendency term-by-term).

## Fix (shipped, default-off)
A **boundary-localised 2Δx-in-lon Shapiro filter on the wall rows only** —
`_bc_wall_grid_filter` (Stage 8b in `ocean_pe_latlon_cgrid.py`), gated by
`LatLonCGridOceanConfig.wall_grid_filter_rate_s` [1/s], default **0.0 (off)**:

    dv/dt -= rate · hp_lon(v) · v_mask     (and u),   on the first/last 8 wall rows only

- Interior is **identically zero** ⇒ NOT a domain viscosity/closure (distinct from the
  rejected A_h/Smag/biharmonic backstop). Default-off ⇒ production paths bit-identical.
- **SPMD/MPI-correct**: the wall band is applied only at the TRUE domain N/S ends via
  `lat_ends_are_poles()` (serial/MPI) + `spmd_pole_end_masks()` (SPMD shard_map), so it
  does NOT fire at rank-interior lat cuts.
- **jax.grad + JIT safe** (verified).
- Wired into `SilvestriJetConfig` (`wall_grid_filter_rate_s=5e-4`, no-op for the non-WENO
  schemes UP3/SM2/QG2).

### Validation
- §5 W9V no-closure **survives 20 days** at max|u|≈0.065 = the Oceananigans-oracle
  amplitude (`scripts/tmp/_silvestri_fixed_validate.py`, GPU ~60 s). Set
  `wall_grid_filter_rate_s=0.0` to reproduce the day-11 blow-up.
- `tests/ocean/unit/test_wall_grid_filter.py` — 7 tests (no-op default / non-WENO,
  interior bit-unchanged, correct hp_lon, land masking, MPI south-end-only gating, MPI
  interior-rank no-filter). `tests/core/test_weno.py` + `tests/ocean/unit/test_weno_momentum.py`
  unchanged (100 pass).

## Open items
1. **Faithfulness call**: this is the *symptom* fix (a localised wall filter), not the
   masked-wall *root* fix. It is defensibly not the prohibited domain backstop, but it does
   not "match Oceananigans" (which avoids the mode by not sourcing it).
2. **Codex adversarial review** still owed (per repo policy) — it could not run in the
   diagnosing environment (no `codex` binary / plugin).
3. **Faithful-root path**: the wall tendency-bridge (above), or a targeted study of the
   masked-land-wall Neumann fills vs a grid-edge wall.
