# Phase B.1 — Cube ocean bottom drag

## What changed

* `src/legoesm/ocean/state.py`: cubed-sphere `OceanConfig` gains three
  fields mirroring `LatLonCGridOceanConfig`:
  * `bottom_drag_r: float = 0.0` — linear drag rate [m/s].
  * `bottom_drag_bg_velocity: float = 0.0` — MOM6-style background
    floor velocity; `> 0` lifts the drag to the quadratic-with-floor
    formula `tau = (r/u_bg) sqrt(u² + u_bg²) u`.
  * `bottom_drag_bbl_thickness: float = 0.0` — when `> 0`, drag is
    distributed over an Ekman thickness `H_BBL` near the seafloor
    (Killworth & Edwards 1999 / MOM6 `BBL_thick_min`).
* `src/legoesm/ocean/dynamics/ocean_pe_fc.py:
  ocean_baroclinic_tendencies_fc(...)`: adds the bottom-drag block
  mirroring the lat-lon C-grid path at line 1649 in
  `ocean_pe_latlon_cgrid.py`. Two regimes:
  - `H_BBL == 0` — single-cell drag applied via a one-hot mask at the
    bottom level: `du/dt |_drag = -r_eff * u / max(h_bot, 1e-10)`.
  - `H_BBL > 0` — overlap-based BBL distribution over `H_BBL` near
    the seafloor.
* `scripts/run_ocean_test_matrix.py`: removes the
  `NotImplementedError` gate at the previous lines 2155-2173 — cube
  now accepts `bottom_drag_r` straight through.

## Verification

* `pytest tests/ocean/unit/test_ocean_compatibility.py
  tests/ocean/unit/test_latlon_cgrid_ocean.py`: **53 passed**, 2 skipped.
* `scripts/run_ocean_test_matrix.py --only stommel_gyre_tracer
  --grid cubed_sphere --quick --days 0.1`: PASS (previously failed
  with `NotImplementedError` because the Stommel runner passes
  `bottom_drag_r = 1e-4`).
* Cube rest-state (4 variants) all PASS with the new bottom-drag
  fields (defaults are zero, so prior runs are bit-identical).

## Known cube limitations still open

Independent of bottom drag, the cube ocean has a structural blowup on
the **global lock-exchange matrix entry**:

* `lock_exchange/cubed_sphere/C24` and `C72` both diverge to NaN at
  step 10 (model time t = 3000 s), with field amplitudes ~ 1e15. The
  blowup is driven by the sharp 15 K T-front initialized at lat 50°
  combined with the shallow (`H_max=20 m`) Petersen geometry running
  on a coarse global cube. Pre-existing — not introduced by B.1.
* Mitigation attempts that did NOT help (each ran to NaN at step 10):
  * Bump `A_h` from 1 × 10⁴ to 1 × 10⁵.
  * Bump `barotropic_diffusion_alpha` from 0.05 to 0.3.
  * Bump `n_barotropic_substeps` from 30 to 60.

The growth pattern (1e15 within 10 steps) indicates an exponentially
unstable mode in the cube C-D grid pressure-gradient / momentum
operator at face-seam discontinuities under sharp horizontal density
gradients. A proper fix would require:
* Higher-order halo interpolation at cube face seams
  (`legoesm.parallel.halo_exchange_cube`).
* Or a pressure-gradient taper over 1-2 cells across face seams.
* Or a different IC for cube lock-exchange that avoids placing the
  front near a face seam.

This is deferred as a dedicated cube-dycore work item; Phase B.2 will
NOT attempt the fix in this session. The cube ocean validation suite
remains focused on tier-0 to tier-2 cases (rest state, barotropic
gravity wave, geostrophic adjustment, baroclinic adjustment) where the
cube produces results comparable to lat-lon and MPAS.
