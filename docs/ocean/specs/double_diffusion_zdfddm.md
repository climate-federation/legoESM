# HIGH-1 spec — Double-diffusive mixing (NEMO zdfddm)

## Gap
No salt-fingering / diffusive-convection vertical mixing. NEMO ORCA1 runs `ln_zdfddm=.true.`
(Merryfield et al. 1999). Sets tropical/subtropical thermocline structure, Mediterranean &
Red-Sea outflow spreading, and Arctic staircases. Absent on both grids.

## Physics (Merryfield, Holloway & Gargett 1999; NEMO 4/5 `zdfddm.F90`)
Density ratio at each W-interface (z up, flux positive up):
```
R_rho = (alpha * dT/dz) / (beta * dS/dz)        # both gradients on the interface
```
using the SAME `alpha`, `beta` as the model EOS (reuse `eos.eos_density_derivatives`,
NOT a re-derived linear pair). Two regimes ADD to the closure diffusivities `avt` (heat) /
`avs` (salt); momentum `avm` untouched (NEMO does not modify avm for ddm):

**Salt fingering** — warm salty over cold fresh, `1 < R_rho`:
```
avs_ddm = rn_avts * (1 - ((R_rho - 1) / (rn_hsbfr - 1))**2)**3      for 1 < R_rho < rn_hsbfr
        = 0                                                          for R_rho >= rn_hsbfr
avt_ddm = 0.7 * avs_ddm / R_rho
```
Defaults (NEMO namzdf_ddm): `rn_avts = 1e-4 m^2/s` (max fingering salt diffusivity),
`rn_hsbfr = 1.6` (fingering cutoff density ratio R_c).

**Diffusive convection** — cold fresh over warm salty, `0 < R_rho < 1` with `N^2 > 0`:
```
avt_ddm = 1.5e-6 * 0.909 * exp( 4.6 * exp( -0.54 * (1/R_rho - 1) ) )
avs_ddm = avt_ddm * (1.85 - 0.85/R_rho) * R_rho       for 0.5 <= R_rho < 1
        = avt_ddm * 0.15 * R_rho                        for 0    < R_rho < 0.5
```
(Federov 1988 / Kelley 1984 fit; the two branches are continuous at R_rho=0.5.)

Outside both regimes (R_rho <= 0, or fingering above cutoff) the ddm contribution is 0.
Guard `N^2 <= 0` columns to 0 (convection scheme handles those). Floor `|beta*dS/dz|` by
`_EPS` before the ratio (traced-safe, no branch).

## Sign / conservation
- avt_ddm, avs_ddm >= 0 by construction (the polynomials are clipped to their valid R_rho
  windows via `jnp.where`; the fingering cubic is >=0 on (1, R_c)).
- Additive to avt/avs on the implicit path exactly like `iwm_K_profile` — conserves column
  heat/salt (flux-form implicit diffusion, no-flux interior BC).
- Idealized test (acceptance, write FIRST): a two-column analytic check —
  (a) warm-salty-over-cold-fresh column → avs_ddm > avt_ddm > 0, R_rho in (1, R_c);
  (b) cold-fresh-over-warm-salty → avt_ddm > avs_ddm > 0;
  (c) statically-stable single-sign T,S (R_rho<0 or >R_c) → both 0.

## API (mirror zdfiwm)
New module `ocean/physics/vertical_mixing/double_diffusion.py`:
```python
__physics_contract__ = {units, signs: "avt/avs>=0, decrease outside DDM windows",
    conserves: [], differentiable: True, reference: "Merryfield 1999 / NEMO zdfddm",
    idealized_test: "tests/ocean/unit/test_double_diffusion.py"}
class DoubleDiffusionConfig(NamedTuple):
    enabled: bool = False
    rn_avts: float = 1e-4     # max salt-fingering salt diffusivity [m^2/s]
    rn_hsbfr: float = 1.6     # fingering cutoff density ratio R_c [1]
    k_max: float = 1e-2       # cap [m^2/s]
__param_spec__ = {rn_avts: tier1 bounds(1e-5,5e-4), rn_hsbfr: tier2 bounds(1.2,2.0)}
def compute_ddm_diffusivity(N2, alpha_dTdz, beta_dSdz, cfg) -> (avt_ddm, avs_ddm)
```
Add `ddm: DoubleDiffusionConfig = DoubleDiffusionConfig()` field to `VerticalMixingConfig`
(mirror `iwm`). Fingering/convection formulas as `_UPPER_SNAKE` fitted constants w/ provenance.

## Wiring
- `k_profiles.py`: after the closure K profile + iwm add, add a `ddm_K_profile(...)` that
  returns (avt_ddm, avs_ddm) and add to `K_v_cell` (heat) and a SEPARATE salt diffusivity.
  ⚠️ NOTE: current implicit solver uses one shared K for T+S (`_pair`). Double diffusion
  REQUIRES distinct avt vs avs. Check `implicit_solver.py::_pair` — if it forces equal K,
  extend to accept (K_T, K_S) separately (this is the one real structural change; zdfiwm
  also adds to both but equally, so it didn't need the split). Reuse the existing
  `_batched` tridiagonal per-tracer.
- Gate on `cfg.ddm.enabled` (Python static bool, feature-gate exception — not jnp.where).
- Grids: latlon/tripole via k_profiles; MPAS via mpas_integration (same additive pattern).
- Fail-fast: `implicit_vertical_mixing=True` required (like iwm) — the explicit path can't
  carry a separate salt diffusivity.

## CLI (run_omip_core2 + run_omip)
`--double-diffusion` (enable), `--ddm-avts`, `--ddm-rc` → build DoubleDiffusionConfig →
thread into VerticalMixingConfig. Round-trip test in test_run_omip*_cli.

## Ponytail
Reuse: `eos.eos_density_derivatives` (alpha,beta), the iwm additive-K wiring, the
`_batched` tridiagonal, the feature-gate pattern. NO new solver, NO re-derived EOS.
New code ≈ one module (~120 LOC) + the (K_T,K_S) solver split + config field + CLI.

## Codex adversarial review targets
- R_rho sign at the fingering/convection boundary; continuity at R_rho=0.5 and R_rho=1.
- avs>=0 across the whole fingering cubic on (1, R_c); no negative K from the convection fit.
- (K_T,K_S) split doesn't break the shared-K fast path for schemes that DON'T use ddm
  (byte-identical when ddm.enabled=False).
- alpha,beta from in-situ vs neutral — match what the closure/KPP uses (consistency).
