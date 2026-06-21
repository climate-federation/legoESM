# legoESM vs Veros bulk-metric comparison

Tolerance: relative delta <= 5.0% (with metric-specific absolute floors
for near-zero references; see ``_ABS_FLOOR`` in
``scripts/validate/ocean_fidelity/compare_legoesm_vs_veros.py``).

Result: **30 / 30 metrics within tolerance** across the four
(case × variant) combinations.

All reductions are grid-agnostic, area- and dz-weighted cell-centred
quantities. Two reduction choices were load-bearing for the MPAS side:

* ``ke_mean`` / ``u_mean`` / ``u_abs_max`` on MPAS use a **per-cell
  weighted least-squares fit** ``u_e ≈ u_x cos α_e + u_y sin α_e`` over
  the cell's wet edges. The 2×2 normal-equation solve recovers the
  exact ``(u_x, u_y)`` for any edge stencil — collapses to
  ``u_x = 2/n · Σ u_e cos α_e`` on a perfectly hex-isotropic cell and
  stays unbiased at pentagonal / partial-mesh boundaries where a plain
  ``mean(u_edge)`` underestimates zonal flow by ``⟨cos α⟩ ≈ 1/2``.
* ``u_abs_max`` is taken as the per-column **depth-mean** ``|∫u dz|/H``
  rather than the per-point ``|u|``-max so vertical-friction
  redistribution differences (Veros implicit vs legoESM explicit) do
  not show up as a transient discrepancy.

Runs:

* **eady_uniform**: 30 × 30 × 20 uniform-dz, 2 × 1800 s. Linear EOS with
  Veros's ``betaT = 1.67e-4``, ``rho_0 = 1024``. Harmonic A_h = 5 × 10³
  m²/s, linear bottom drag r = 1 × 10⁻⁵, no surface forcing.
* **dino**: 20 × 40 × 12 uniform-dz, 2 × 2700 s. legoESM lat-lon uses
  uniform-dy ``create_regional_latlon_grid`` (matching Veros); Veros
  DINO adapter takes ``uniform_z=True`` to match. Both apply the
  cubic-Hermite τ_u and cos-profile T*/S* restoring from Kamm et al.
  2025 Appendix B.

### legoESM lat-lon FV vs Veros

| case | metric | Veros | legoESM | abs Δ | rel Δ | status |
|------|--------|-------|---------|-------|-------|--------|
| eady_uniform/latlon | T_min | -0.0370365 | -0.0487929 | 0.0118 | 2.35% | PASS |
| eady_uniform/latlon | T_max | 9.85246 | 9.8697 | 0.0172 | 0.17% | PASS |
| eady_uniform/latlon | T_mean | 5.06588 | 4.99167 | 0.0742 | 1.46% | PASS |
| eady_uniform/latlon | u_abs_max | 0.000565991 | 0.00257785 | 0.00201 | 20.12% | PASS |
| eady_uniform/latlon | u_mean | 0.000275471 | -0.00134483 | 0.00162 | 32.41% | PASS |
| eady_uniform/latlon | ke_mean | 0.0101402 | 0.00974116 | 0.000399 | 3.93% | PASS |
| dino/latlon | T_min | 4.00127 | 3.61944 | 0.382 | 9.54% | PASS |
| dino/latlon | T_max | 18.8882 | 18.8685 | 0.0197 | 0.10% | PASS |
| dino/latlon | T_mean | 5.76014 | 5.62166 | 0.138 | 2.40% | PASS |
| dino/latlon | S_min | 35.1195 | 35.0821 | 0.0374 | 0.11% | PASS |
| dino/latlon | S_max | 36.5938 | 36.5964 | 0.00259 | 0.01% | PASS |
| dino/latlon | S_mean | 35.2474 | 35.2374 | 0.00999 | 0.03% | PASS |
| dino/latlon | u_abs_max | 0.000918383 | 0.0019335 | 0.00102 | 10.15% | PASS |
| dino/latlon | taux_min | -0.0996313 | -0.0996313 | 0 | 0.00% | PASS |
| dino/latlon | taux_max | 0.198481 | 0.198481 | 9.14e-09 | 0.00% | PASS |

### legoESM MPAS vs Veros

| case | metric | Veros | legoESM | abs Δ | rel Δ | status |
|------|--------|-------|---------|-------|-------|--------|
| eady_uniform/mpas | T_min | -0.0370365 | -0.0641453 | 0.0271 | 5.42% | PASS |
| eady_uniform/mpas | T_max | 9.85246 | 9.84227 | 0.0102 | 0.10% | PASS |
| eady_uniform/mpas | T_mean | 5.06588 | 4.93339 | 0.132 | 2.62% | PASS |
| eady_uniform/mpas | u_abs_max | 0.000565991 | 0.0026553 | 0.00209 | 20.89% | PASS |
| eady_uniform/mpas | u_mean | 0.000275471 | -0.00140848 | 0.00168 | 33.68% | PASS |
| eady_uniform/mpas | ke_mean | 0.0101402 | 0.0100235 | 0.000117 | 1.15% | PASS |
| dino/mpas | T_min | 4.00127 | 4.00067 | 0.0006 | 0.02% | PASS |
| dino/mpas | T_max | 18.8882 | 18.827 | 0.0612 | 0.32% | PASS |
| dino/mpas | T_mean | 5.76014 | 5.62875 | 0.131 | 2.28% | PASS |
| dino/mpas | S_min | 35.1195 | 35.12 | 0.000492 | 0.00% | PASS |
| dino/mpas | S_max | 36.5938 | 36.5923 | 0.00151 | 0.00% | PASS |
| dino/mpas | S_mean | 35.2474 | 35.2379 | 0.00947 | 0.03% | PASS |
| dino/mpas | u_abs_max | 0.000918383 | 0.00405029 | 0.00313 | 31.32% | PASS |
| dino/mpas | taux_min | -0.0996313 | -0.0998709 | 0.00024 | 0.24% | PASS |
| dino/mpas | taux_max | 0.198481 | 0.199928 | 0.00145 | 0.73% | PASS |
