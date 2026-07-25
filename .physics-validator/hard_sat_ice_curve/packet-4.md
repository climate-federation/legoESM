You are an independent ADVERSARIAL physics reviewer for LegoESM. ROUND 4
(confirmation). Round 3 left ONE finding: the F3 ice-number-seed cap used a
density floor `1e-6` while Morrison floors `rho` at `_RHO_FLOOR=0.1 kg/m^3` in
its Cooper number/mass conversion (`/clip(rho,0.1)`, morrison.py:346), so at low
density my ceiling loosened ~4x (15 hPa) to far more (near top). This round makes
the floor Morrison-exact. VERIFY and find any NEW issue. Be CONCISE; end with
`OVERALL VERDICT: <no substantive findings | findings remain: ...>`.

Current diff: `.physics-validator/hard_sat_ice_curve/model_driver.diff`

# Round-3 finding fix — Morrison-consistent density floor
```python
# Morrison _RHO_FLOOR (module_mp_graupel density floor in divisions), so the
# seed ceiling is Morrison-consistent at low density.
_ICE_SEED_RHO_FLOOR = 0.1        # [kg/m^3]

def _seed_nucleated_ice_number(N_i, dq_i, ice_nuc_mass, n_i_nuc_max, p_full, T):
    rho_air = p_full / (constants.R_d * jnp.maximum(T, 1.0))
    n_i_max_perkg = n_i_nuc_max / jnp.maximum(rho_air, _ICE_SEED_RHO_FLOOR)
    d_n_raw = jnp.maximum(dq_i, 0.0) / jnp.maximum(ice_nuc_mass, 1.0e-30)
    headroom = jnp.maximum(n_i_max_perkg - N_i, 0.0)
    return N_i + jnp.minimum(d_n_raw, headroom)
```
This is `jnp.maximum(rho_air, 0.1)` == Morrison's `jnp.clip(rho, 0.1)` (floor
only), so the per-mass Cooper ceiling `N_i_nuc_max / clip(rho, 0.1)` now matches
Morrison at every level, incl. the TTL/stratosphere. `0.1` is a named module
constant with a provenance comment (not an inline literal; passes the inline-
coeff and hardcoded-constant ratchets). The residual dry-vs-moist rho difference
is ~1% on a NUMBER ceiling (negligible); Morrison itself receives a diagnostic
rho of the same p/(R_d T) form.

# Test fix
`test_seed_nucleated_ice_number` now:
- mid-tropo (500 hPa/260 K, rho~0.67>>floor): cap uses true rho;
- cap-binding 2 g/kg -> capped exactly at `N_i_nuc_max/rho`, and `dq/mi0 > 300x`;
- **GENUINE low density (1500 Pa/205 K, rho~0.025 << 0.1 floor)**: asserts the
  ceiling == `N_i_nuc_max / 0.1` (floored), and `< 0.5 *` the unfloored
  `N_i_nuc_max/rho` (proves the floor bit — the Round-3 mislabelled 1.5e4 Pa /
  "15 hPa" case was actually 150 hPa and missed this regime);
- N_i already above ceiling -> no add; negative/zero deposit -> no add.

# STATUS
`test_hard_sat_ice_curve.py` (23) + inline-coeff + hardcoded-constant ratchets:
**3783 passed, 2 skipped**. Prior rounds: 47 + 233 broader passed (only the 2
pre-existing `mcfarlane` vs `mcfarlane+hines` gwd fails remain, independent).

Accepted earlier (unchanged): F1 (MPAS-grid guard, normalize_grid_type so
voronoi/icosahedral aliases pass), F2 (Morrison-only + energy-exact liquid
degradation when no q_i), F4 (L_s deposition matches Morrison; -700 J/kg is a
pre-existing constants inconsistency), F5 (w(T0) routing required for enthalpy;
Morrison melts any above-freezing q_i), F6 (fp32 subprocess + T/p FD tests).

# CONFIRM
Is the F3 cap now fully Morrison-consistent? Any remaining substantive finding
across the whole change (units, signs, differentiability, conservation, MPAS
wiring)? If none, say so explicitly.
