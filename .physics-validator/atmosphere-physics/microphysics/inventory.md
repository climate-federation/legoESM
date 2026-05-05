# Microphysics inventory

Scope: `src/legoesm/atmosphere/physics/microphysics/`. 6 schemes + integration. ~2200 LOC total.

## Schemes

| Scheme | File | LOC | Tracers | Notes |
|--------|------|-----|---------|-------|
| kessler | kessler.py | 136 | q_v, q_c, q_r | warm-rain, single moment |
| sundqvist | sundqvist.py | 170 | q_v, q_c, q_r | diagnostic condensation, no autoconversion threshold |
| seifert_beheng | seifert_beheng.py | 130 | q_v, q_c, q_r, N_c, N_r | two-moment warm-rain |
| morrison | morrison.py | 189 | + q_i, q_s, N_i | adds mixed-phase (Cooper, Bergeron, riming) |
| thompson | thompson.py | 213 | + q_g | adds graupel |
| ml_emulator | ml_emulator.py | ~? | learned | NN replaces process rates |

## Shared helpers (`_warm_rain.py`, 176 LOC)

* `saturation_adjustment(T, q_v, p_full, dt, sharpness)` — smooth condensation tendency [kg/kg/s].
* `effective_Nc(N_c, Nc_0)` — replace zero N_c with default.
* `autoconversion_sb(q_c, N_c, ρ, k_au, x_star, sharpness, gamma_norm)` — SB mass-dependent autoconversion.
* `accretion(q_c, q_r, ρ, k_ac, gamma_norm)` — bulk accretion.
* `self_collection_breakup(N_r, q_r, ρ, k_sc, sharpness, D_eq)` — N_r evolution.
* `rain_evaporation(q_v, q_r, q_sat, evap_coeff)`.

All use `legoesm.thermo.saturation_mixing_ratio` (not re-derived). ✓

## Output shape (`MicrophysicsOutput`)

* `dT_dt` [K/s], shape `(ncol, nlev)`.
* `dq_v_dt`, `dq_c_dt`, `dq_r_dt`, `dq_i_dt`, `dq_s_dt`, `dq_g_dt` — all [kg/kg/s].
* `dN_c_dt`, `dN_r_dt`, `dN_i_dt` — [1/kg/s].
* `precipitation` — [kg/m²/s] (positive, surface flux).

## Convention

* Surface at `[:, -1]`.
* Sedimentation: positive `V_t` is downward fall speed [m/s].
* `q` are *specific* mixing ratios (per total air mass).

## Constant usage

All schemes use `from legoesm import constants` — `constants.L_v`, `constants.L_s`, `constants.L_f`, `constants.c_pd`, `constants.T_freeze`. No hardcoded literals found.

## Latent heat balance (per scheme, conceptual)

* kessler: `dT_dt = L_v * (cond - evap) / c_pd`. Pure warm-rain, no ice.
* sundqvist: `dT_dt = L_v * (cond - evap) / c_pd`. Same.
* seifert_beheng: `dT_dt = L_v * (cond - evap) / c_pd`. Two-moment but warm only.
* morrison: `+ L_s * dq_i_dep + (terms missing)`.
* thompson: `+ L_s * dq_i_dep + (terms missing)` — see static analysis.
