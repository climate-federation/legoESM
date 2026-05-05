# Turbulence — static analysis

## A. Confirmed correctness

1. **Sign of surface stress** (`surface_layer.py:93-94`): `tau_x = -ρ Cd |V| u`. Applied at `rhs.at[:, -1].add(dt * tau_x / (ρ dz))` ⇒ decelerates u for u>0. ✓
2. **Sign of sensible/latent heat fluxes**: `shflx = ρ c_pd Ch |V| (T_sfc - T)` ⇒ positive upward when surface warmer. ✓
3. **Implicit vertical diffusion**: `b = 1 + a + c`, all coefficients positive, Thomas algorithm via scan. Diagonally dominant ⇒ stable.
4. **TKE equation** (`tke.py`): `tke_new = (tke_diff + dt(P+B)) / (1 + dt diss_coeff)`. Semi-implicit dissipation, unconditionally stable. ✓
5. **Counter-gradient correction** (Holtslag-Boville `holtslag_boville.py:181-192`): documented as enhanced surface flux (a simplification of the textbook explicit `Kh γ` interior term). Acceptable for first-order approx.
6. **PBL height (sigmoid method)**: `weights = σ(1-σ) + 1e-20`, peaks at the Ri_crit crossing — gives a smooth fractional height that converges to the right limit. ✓

## B. Potential issues (RANKED)

### B1. **Hardcoded `0.61` virtual-T factor everywhere** [LOW STYLE]
Every scheme uses `1.0 + 0.61 * q_v`. Standard but inconsistent with `_shared.py:478` which writes `1 + (R_v/R_d - 1)*q_v ≈ 1 + 0.608*q_v`. The numerical drift is `0.16%`. Should call a shared `virtual_temperature(T, q_v)` helper. Not a bug, just code-discipline.

### B2. **`edmf.py:168` `theta_u_init = (theta[:, -1] + 0.5)`** [TUNABLE-IN-BODY]
The +0.5 K parcel perturbation is hardcoded; should be a config parameter `EDMFConfig.parcel_dT`. Same pattern as convection schemes which do this correctly via config.

### B3. **Holtslag-Boville `b_louis = 5.0` and `5.0 * Ri` literals** [TUNABLE-IN-BODY]
`holtslag_boville.py:141`: `b_louis = 5.0` and line 143: `1.0 + 5.0 * Ri_pos`. These are scheme-defining constants and should be in `HoltslagBovilleConfig`.

### B4. **`smooth_tk_e_min` in dissipation** [STABILITY]
`tke.py:163-165`: `tke_new = (tke_diffused + dt*(P+B)) / (1 + dt * diss_coeff)`. If `P + B < 0` (e.g. strongly stable, `B < 0` and shear is small), the numerator can go negative. Then `jnp.maximum(tke_new, config.tke_min)` floors it (good). But the gradient w.r.t. tke vanishes in the floor region — known and intentional.

### B5. **`edmf.py` updraft scan: smooth deactivation may keep dead updrafts alive** [LOW]
`edmf.py:204-207`:
```python
active = jax.nn.sigmoid(20.0 * w_u_new / config.w_updraft_min)
w_u_new = w_u_new * active
```
For `w_u_new = 0`, `active = 0.5`, then `w_u_new = 0`. But for small positive `w_u_new`, `active < 1` and `w_u_new` is reduced but not driven to zero. The recursive multiplication across scan steps will eventually decay the updraft, but slowly. Acceptable.

### B6. **`edmf.py:249-252`: one-sided FD at top/bottom** [INFO]
`_mf_tendency` uses centered FD interior with one-sided FD at edges. Standard; the truncation error is O(dz) at edges vs O(dz²) interior. Acceptable.

### B7. **Stability function blend `sigmoid(100 * Ri)` in louis/HB** [SHARP]
`louis.py:134`, `holtslag_boville.py:149`: `blend = jax.nn.sigmoid(100.0 * Ri)`. Sharpness 100 over Ri ~ O(0.1) → very steep. Numerically fine but practically a hard switch. Documented.

### B8. **`vertical_diffusion.py:108` `_TINY = float(jnp.finfo(jnp.float32).tiny)`** [INFO]
`b_prev_mod` clipped to `≥ TINY` (~1.18e-38). Sufficient to prevent division by zero. ✓

### B9. **`Holtslag-Boville` mixing length uses `kappa_vk * z / (1 + kappa_vk * z / l_max)`** [VERIFIED]
This is Blackadar's asymptotic form. Standard. ✓

### B10. **YSU and CLUBB-lite not deeply audited** — flagged for follow-up.
File sizes: YSU 213 LOC, CLUBB-lite 295 LOC. Initial scan showed similar patterns (sigmoid blends, virtual theta, implicit diffusion). No obvious red flags but not a deep audit.

## C. Top issues

* **B2** (hardcoded EDMF parcel perturbation), **B3** (HB Louis constants in body) — both violate CLAUDE.md "no hardcoded tunables in body". Real but minor.
* No confirmed numerical bugs.
