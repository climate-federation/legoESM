You are an independent ADVERSARIAL physics reviewer for LegoESM. ROUND 3
(confirmation). In Round 2 you accepted F1, F2, F4, F5, F6 and left ONE finding:
**F3 — the N_i seed was uncapped** (a heating-cap-sized all-ice deposit seeded
~1.7e8 m^-3, ~338x Morrison's Cooper ceiling N_i_nuc_max=5e5 m^-3, perturbing
deposition/sedimentation). This round fixes F3. VERIFY the fix and find any NEW
bug it introduces. Be CONCISE; end with
`OVERALL VERDICT: <no substantive findings | findings remain: ...>`.

Full current diffs:
`.physics-validator/hard_sat_ice_curve/model_driver.diff`
`.physics-validator/hard_sat_ice_curve/config.diff`
`.physics-validator/hard_sat_ice_curve/warm_rain.diff`

# F3 fix — bounded ice-number seed
`_seed_nucleated_ice_number` now caps the added number at Morrison's Cooper
ice-number ceiling `N_i_nuc_max` [1/m^3], converted to per-mass via
`rho_air = p/(R_d T)` — exactly Morrison's own Cooper target conversion
(`kc2 = min(...)/rho`, morrison.py:338-345). A large deposit then GROWS crystals
(bigger mean size) instead of over-populating number; a small deposit (TTL,
mg/kg) is unchanged (each new crystal at the 10-um nucleation mass mi0):
```python
def _seed_nucleated_ice_number(N_i, dq_i, ice_nuc_mass, n_i_nuc_max, p_full, T):
    rho_air = p_full / (constants.R_d * jnp.maximum(T, 1.0))
    n_i_max_perkg = n_i_nuc_max / jnp.maximum(rho_air, 1.0e-6)
    d_n_raw = jnp.maximum(dq_i, 0.0) / jnp.maximum(ice_nuc_mass, 1.0e-30)
    headroom = jnp.maximum(n_i_max_perkg - N_i, 0.0)   # 0 if already at ceiling
    return N_i + jnp.minimum(d_n_raw, headroom)
```
Driver call site passes the Morrison sub-config `N_i_nuc_max`, the per-cell
`p_full = p_s*sigma`, and the post-drain `T` (model_driver.diff). Bound:
`N_i_new <= max(N_i, N_i_nuc_max/rho_air)` per cell — never exceeds the ceiling.

Whole ceiling math is now inside the unit-tested helper (was split across the
driver call site last round). `test_seed_nucleated_ice_number` covers: small
deposit uncapped (== dq_i/mi0, below ceiling); a 2 g/kg deposit CAPPED exactly at
`N_i_nuc_max/rho_air` (and `dq_i/mi0 > 300x` that ceiling — the pre-fix over-seed);
N_i already above ceiling -> no add; negative/zero deposition -> no add.

# STATUS
- `test_hard_sat_ice_curve.py` = **23 passed** (incl. cap-binding + a voronoi/mpas
  alias test — I also switched the F1 grid guard to `normalize_grid_type(...)`
  so MPAS aliases voronoi/icosahedral/mpas_voronoi are accepted, matching the
  driver's own `_run_mpas` gate; earlier the raw `=='mpas'` check wrongly
  refused them). Full suite last round: 47 + 212 broader passed (only the 2
  pre-existing `mcfarlane` vs `mcfarlane+hines` gwd fails remain, independent).

# CONFIRM
- The cap is the correct physical bound (Morrison's Cooper ceiling per mass) and
  the headroom form `N_i + min(dq_i/mi0, max(ceiling - N_i, 0))` never exceeds
  `max(N_i, ceiling)`; number carries no latent heat so budgets are untouched.
- The `rho_air = p/(R_d T)` diagnostic matches Morrison's own `/rho` conversion
  (dry-air R_d; ~1% virtual-T effect on a NUMBER ceiling is negligible).
- Any NEW issue: dtype, the `jnp.maximum(rho_air,1e-6)` / `jnp.maximum(T,1.0)`
  floors, the grid-alias guard change, or the driver wiring.

Do you have any remaining substantive finding? If not, say so explicitly.
