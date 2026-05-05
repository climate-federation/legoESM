# Convection inventory

Scope: `src/legoesm/atmosphere/physics/convection/`. Eleven leaf modules + integration. ~3700 LOC. Recently refactored (commits 0f0a405, 8786140). Most active surface in the audit.

## Public schemes (`integration.py` factory branches)

| Scheme | File | LOC | Carry | CMT | Inputs | Output |
|--------|------|-----|-------|-----|--------|--------|
| sbm | sbm.py | 165 | none | no | T, q_v, p_full, p_half, dt | dT/dt, dq_v/dt, dq_c_conv/dt, cape, mask |
| dca | dca.py | 277 | none | no | T, q_v, p_full, p_half, dt | dT/dt, dq_v/dt, dq_c_conv/dt, cape, mask |
| kuo | kuo.py | 217 | none | no | T, q_v, p_full, p_half, dt | dT/dt, dq_v/dt, dq_c_conv/dt, cape, mask |
| mass_flux (Arakawa-Wu) | mass_flux.py | 495 | M_c (scalar) | no | T, q_v, p_full, p_half, M_c, dt | dT/dt, dq_v/dt, dq_c_conv/dt, cape, mask + M_c_new |
| edmf | mass_flux.py | (in mass_flux) | a_u (scalar) | no | T, q_v, p_full, p_half, a_u, dt | same + a_u_new |
| zhang_mcfarlane | zhang_mcfarlane.py | 220 | M_b @ [:,-1] | yes (Gregory) | T, q_v, p_full, p_half, u, v, conv_prog_profile, dt | + du/dt, dv/dt + carry |
| kain_fritsch | kain_fritsch.py | 241 | diagnostic M_b @ [:,-1] | no | T, q_v, p_full, p_half, w_grid, conv_prog_profile, dt | dT/dt, dq_v/dt, dq_c_conv/dt + carry |
| emanuel | emanuel.py | 258 | diagnostic M_b @ [:,-1] | no | T, q_v, p_full, p_half, conv_prog_profile, dt | dT/dt, dq_v/dt, dq_c_conv/dt + carry |
| tiedtke | tiedtke.py | 316 | M_u(k) profile | yes | + u, v, moisture_convergence | + du/dt, dv/dt + profile |
| bechtold | bechtold.py | 322 | M_u(k) + AR1 stoch | yes | + prng_key, conv_stoch_state | + du/dt, dv/dt + profile + stoch state |

## Shared primitives

* `_triggers.py` (363 LOC): smooth_step, smooth_max, smooth_min, smooth_positive_part, smooth_level_indicator, smooth_lowest_crossing_index, cape_trigger.
* `_plume.py` (673 LOC): compute_lcl (Bolton 1980), compute_lfc_lnb, compute_cin, entraining_detraining_plume (jax.lax.scan), cmt_gregory_1997.
* `mass_flux.py` shared kernels: `_compute_column_geometry`, `_compute_cape_diagnostics`, `_compute_centered_gradients`, `stratosphere_mass_flux_gate`, `_apply_mass_flux_kernel`.

## Units (column convention)

* `T` [K], `q_v` [kg/kg], `p_full`, `p_half` [Pa].
* Tendencies: `dT/dt` [K/s], `dq_v/dt` [kg/kg/s], `dq_c_conv/dt` [kg/kg/s] (≥0).
* CAPE [J/kg], M_b / M_u [kg/m²/s].
* All schemes use `(ncol, nlev)`, surface at `[:, -1]`.

## Sign conventions (as documented)

* Buoyancy `B = g (T_v_u − T_v_env) / T_v_env` — positive upward.
* CMT `du/dt = −(1/ρ) dF/dz` (Gregory bulk closure).
* `omega = dp/dt > 0` ⇒ sinking; `w = −ω/(ρg)`.
* `dq_c_conv_dt ≥ 0` everywhere (enforced by `jnp.maximum(..., 0)` at ConvectionOutput emission for tiedtke / bechtold / emanuel).

## Physics state usage

* `tiedtke`, `bechtold` carry `conv_prog_profile = M_u(k)` — relaxed via implicit Euler over `tau_M_u_relax`.
* `mass_flux` packs scalar `M_c` at `[:, -1]`.
* `edmf` packs scalar `a_u` at `[:, -1]`.
* `zhang_mcfarlane` packs scalar `M_b` at `[:, -1]` for visibility.
* `kain_fritsch` / `emanuel` are fully diagnostic — emit fresh profile every step.
* `bechtold` ALSO carries `conv_stoch_state` (AR1 noise) and consumes/refreshes `prng_key`.

## Recent activity (last 3 months)

* `0f0a405` (2026-05-01): Convection corrections and test runs.
* `8786140` (2026-05-01): Improvement convection (current HEAD).
* `34652ff` (2026-04-30): `Convection/pr1 spectral pe tracers`.
* `954f369` (2026-04-28): `Corrects Kuo scheme`.
* `bae73a8` (2026-03-25): `corrected physics`.

This subsystem has had the most churn — rationale for being the first/highest-priority target of static + codex review.
