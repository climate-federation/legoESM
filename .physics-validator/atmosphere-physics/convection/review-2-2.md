Reading prompt from stdin...
OpenAI Codex v0.125.0 (research preview)
--------
workdir: /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
model: gpt-5.5
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019de5d0-74b2-7b60-ae2c-6093b346146e
--------
user
# Codex Adversarial Review — Convection & Microphysics (Cycle 2 — Iteration 2)

You are an independent adversarial physics-parameterization reviewer for legoESM, a JAX-native differentiable Earth System Model. Your job is to verify that fixes applied in response to your previous review (cycle 2 iteration 1) are correct and to find any new issues.

## Context: previous review (cycle 2 iteration 1) findings

You returned 6 findings:
1. **P1 (CONFIRMED)**: plume entrainment uses unstable explicit Euler `T_u_ent = T_u_prev + eps*dz*(T_e - T_u_prev)`. With Bechtold defaults `epsilon_shallow=3e-3` × layer thickness 500-1000 m, `eps*dz > 1` causes `q_u_ent` to overshoot past `q_e` (go negative).
2. **P1 (DEFERRED)**: Hines drag dimensional inconsistency `rho * (sigma_grown - sigma_new)` is `kg/(m²·s)` not Pa. Deferred because fixing requires re-tuning.
3. **P2**: plume termination is local (no buoyancy-death memory in carry). Documented as deferred design issue.
4. **P2**: Tiedtke MC proxy has zero dry-side gradient via `jnp.maximum`. Documented as P2.
5. **P2**: warm_rain `x_c = q_c * rho / N_c` and `dN_r_au = dq_c_au * rho / ...` use per-volume convention while N is documented per-kg. Documented as P3 (likely doc bug).
6. **P1 (CONFIRMED)**: Thompson rime_to_graupel uses fixed 1.0 / 0.5 / 1.5 split on total_riming, violating mass conservation when only one species is rimed (e.g. q_i=0, q_s>0).

## Fixes applied to your findings

### Fix for #1: Plume entrainment exponential relaxation

`src/legoesm/atmosphere/physics/convection/_plume.py:534-548`:
```python
# T_u_ent and q_u_ent: exponential relaxation toward env, exact for
# linear ODE dX/dz = -eps*(X - X_e)
decay_eps = jnp.exp(-eps * dz)
T_u_ent = T_e + (T_u_prev - T_e) * decay_eps
q_u_ent = q_e + (q_u_prev - q_e) * decay_eps
```

Also changed `q_c_u_ent = q_c_u_prev * decay_eps` (was `max(q_c_u_prev * (1 - eps*dz), 0)`).

### Fix for #6: Thompson rime_to_graupel donor split

`src/legoesm/atmosphere/physics/microphysics/thompson.py`:
```python
# Compute conversion per donor:
rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s

# Apply qc_scale to each donor (same scale, since both come from q_c):
rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s

# Apply per-donor in tendency assembly:
dq_i_dt = (... + riming_i - aggregation - melt_ice - rime_to_graupel_from_i + sed_i)
dq_s_dt = (aggregation + riming_s - melt_snow - rime_to_graupel_from_s + sed_s)
dq_g_dt = rime_to_graupel - melt_graupel + sed_g  # all of it goes to graupel
```

Mass conservation: `dq_i loss + dq_s loss + dq_g gain = -rime_to_graupel_from_i - rime_to_graupel_from_s + (from_i + from_s) = 0` ✓.

### Tests added

- `tests/unit/test_convection_plume.py::test_plume_T_q_entrainment_does_not_overshoot_environment` — passes on fixed code, fails on buggy code.
- `tests/unit/test_convection_plume.py::test_plume_grad_through_epsilon_at_strong_entrainment` — verifies non-zero grad at eps·dz > 1.
- `tests/unit/test_physics_microphysics.py::test_thompson_rime_to_graupel_donor_split` — pathological column q_i=0, q_s>0 — verifies q_i stays non-negative.

## Your task (iteration 2)

1. Verify the two fixes above are mathematically correct and complete.
2. Look for **NEW issues** introduced by these fixes (regressions).
3. Re-read the code below and try to find any **OTHER bugs** the cycle-1 review missed.
4. Specifically: check the `q_u_ent = q_e + (q_u_prev - q_e) * decay_eps` formulation — does it correctly preserve total water (q_u_ent + q_c_u_ent) when entrainment dilutes both? Or is there a hidden mass leak?

## Source: convection/_plume.py (post-fix)

```python
def step(carry, layer_inputs):
    T_u_prev, q_u_prev, q_c_u_prev, M_u_raw_prev, z_prev = carry
    T_e, q_e, p_e, z_e, eps, dlt, abv = layer_inputs

    dz = jnp.maximum(z_e - z_prev, 1.0)

    # Exact integration of dM/dz = (eps - dlt) M
    M_u_raw = M_u_raw_prev * jnp.exp((eps - dlt) * dz)

    # Exponential relaxation entrainment (NEW FIX)
    decay_eps = jnp.exp(-eps * dz)
    T_u_ent = T_e + (T_u_prev - T_e) * decay_eps
    q_u_ent = q_e + (q_u_prev - q_e) * decay_eps

    rho_u_ent = p_e / (constants.R_d * jnp.maximum(T_u_ent, 100.0))
    Gamma_moist_per_pa = moist_adiabat_lapse_rate(T_u_ent, p_e)
    dT_dz = Gamma_moist_per_pa * (-rho_u_ent * g)
    T_u = T_u_ent + dT_dz * dz

    q_sat_new = saturation_mixing_ratio(T_u, p_e).astype(_dtype)
    condensate = jnp.maximum(q_u_ent - q_sat_new, 0.0).astype(_dtype)
    q_u = (q_u_ent - condensate).astype(_dtype)

    # Exponential cloud-water dilution (NEW FIX)
    q_c_u_ent = q_c_u_prev * decay_eps
    q_c_u = (q_c_u_ent + condensate).astype(_dtype)

    T_u = T_u.astype(_dtype)
    B_u = T_u - T_e

    plume_alive = jax.nn.sigmoid(buoyancy_sharpness * B_u)
    M_u_reported = M_u_raw * plume_alive * abv

    new_carry = (T_u, q_u, q_c_u, M_u_raw, z_e)
    outputs = (T_u, q_u, q_c_u, M_u_reported, B_u)
    return new_carry, outputs
```

## Source: microphysics/thompson.py (post-fix, key sections)

```python
# === GRAUPEL (Thompson extension) — donor split ===
graupel_frac = jax.nn.sigmoid(
    config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
)
rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
melt_graupel = jnp.minimum(
    config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac,
    jnp.clip(q_g, 0.0) / dt_safe,
)

# === DONOR CLAMP for q_c sinks ===
qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s
qc_avail = jnp.clip(q_c, 0.0)
qc_scale = jnp.minimum(
    1.0,
    qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
)
dq_c_au = dq_c_au * qc_scale
dq_c_ac = dq_c_ac * qc_scale
bergeron = bergeron * qc_scale
riming_i = riming_i * qc_scale
riming_s = riming_s * qc_scale
total_riming = riming_i + riming_s
rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
dN_r_au = dN_r_au * qc_scale

# === COMBINE TENDENCIES ===
dq_v_dt = -condensation + evaporation - dq_i_dep
dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel + sed_r
dq_i_dt = (
    dq_i_dep + bergeron + riming_i - aggregation - melt_ice
    - rime_to_graupel_from_i + sed_i
)
dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel_from_s + sed_s
dq_g_dt = rime_to_graupel - melt_graupel + sed_g
```

## Specific verification questions

1. **Plume mass conservation under exponential entrainment**: with `decay_eps = exp(-eps*dz)`, the env air entrained into the plume is `q_e * (1 - decay_eps) + q_c_e * (1 - decay_eps)`. Since `q_c_e = 0` (no env cloud water), the plume's total water `q_u + q_c_u` after entrainment is `q_e * (1 - decay_eps) + q_u_prev * decay_eps + q_c_u_prev * decay_eps`. Is this equal to `q_u_prev + q_c_u_prev` (closed plume) plus an entrained source? YES if we add the env water in proportion to entrainment — checking that this is indeed what happens physically.

2. **q_u_ent monotonicity**: with exponential relaxation, is `q_u_ent` always between `min(q_e, q_u_prev)` and `max(q_e, q_u_prev)`? (Should be, since `decay_eps ∈ (0, 1]`.)

3. **Thompson conservation**: in the new tendency assembly, verify ∑(dq_v + dq_c + dq_r + dq_i + dq_s + dq_g) = sed terms only (no internal phase changes accumulate).

4. **Did anything else break?**: re-read `_plume.py` and `thompson.py` from a fresh perspective. Look for new bugs introduced by these specific changes. Pay attention to AD: does `decay_eps = exp(-eps*dz)` have well-defined gradients w.r.t. `eps` and `dz` everywhere?

5. Any **OTHER** issues you see in the cycle-1 source that you didn't flag the first time? (E.g. in microphysics — how about the `dN_i_dt = dN_i_nuc - aggregation * N_i / q_i` formula at q_i → 0?)

Provide a concise list of any new findings; if your previous findings are all addressed, say so explicitly.

codex
I’ll verify the actual post-fix code and tests rather than relying only on the pasted snippets, then I’ll do the review in the usual findings-first format.
exec
/bin/zsh -lc "sed -n '1,280p' src/legoesm/atmosphere/physics/microphysics/thompson.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "sed -n '500,575p' src/legoesm/atmosphere/physics/convection/_plume.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
        jnp.moveaxis(z_full_rev, 1, 0),
        jnp.moveaxis(eps_rev, 1, 0),
        jnp.moveaxis(del_rev, 1, 0),
        jnp.moveaxis(above_base_weight, 1, 0),
    )

    g = constants.g
    c_pd = constants.c_pd
    L_v = constants.L_v

    def step(carry, layer_inputs):
        T_u_prev, q_u_prev, q_c_u_prev, M_u_raw_prev, z_prev = carry
        T_e, q_e, p_e, z_e, eps, dlt, abv = layer_inputs

        dz = jnp.maximum(z_e - z_prev, 1.0)  # ascending; floor to avoid div-by-zero

        # Raw plume mass flux: dM/dz = (epsilon - delta) * M.  Use the
        # exact integration ``M(z+dz) = M(z) * exp((eps - dlt) * dz)``
        # for this linear ODE — always positive, AD-safe everywhere,
        # and exact when ``(eps - dlt)`` is constant over the layer.
        # An earlier explicit-Euler form ``M * (1 + (eps - dlt) * dz)``
        # could go negative for strong detrainment + thick layers
        # (e.g. ``dlt = 5e-3 /m``, ``dz = 2000 m`` ⇒ multiplier =
        # ``-7``); the subsequent ``jnp.maximum(..., 0)`` clipped the
        # mass flux to 0 AND *zeroed the gradient* w.r.t. ``dlt`` /
        # ``eps``, breaking AD-based sensitivity studies through the
        # plume integrator (audit Codex finding: "plume mass flux uses
        # explicit Euler plus a hard nonnegative clip ... after
        # which jnp.maximum kills both mass flux and gradients").
        # We intentionally do NOT bake the buoyancy / sub-cloud masks
        # into the carry — those are reporting filters, not dynamics.
        # Folding them into the carry would compound across levels and
        # destroy the cloud-base-to-LNB profile that consumers expect.
        M_u_raw = M_u_raw_prev * jnp.exp((eps - dlt) * dz)

        # Entrainment of environmental T, q via the analytic relaxation
        # ``X(z+dz) = X_e + (X_prev - X_e) * exp(-eps · dz)`` — exact for
        # the linear ODE ``dX/dz = -eps · (X - X_e)`` and always bounded
        # between ``X_prev`` and ``X_e``.  An earlier explicit-Euler form
        # ``X_prev + eps · dz · (X_e - X_prev)`` overshoots past ``X_e``
        # for ``eps · dz > 1`` (e.g. Bechtold ``epsilon_shallow=3e-3``
        # with a 500–1000 m layer gives ``eps·dz ∈ [1.5, 3]``) — making
        # ``q_u_ent`` go negative and zeroing the gradient w.r.t. ``eps``.
        # The exponential form is dimensionally identical and AD-safe
        # everywhere (Codex stop-time review cycle 2: "plume entrainment
        # still uses unstable explicit Euler").
        decay_eps = jnp.exp(-eps * dz)
        T_u_ent = T_e + (T_u_prev - T_e) * decay_eps
        q_u_ent = q_e + (q_u_prev - q_e) * decay_eps

        # Use the analytic moist-adiabatic lapse rate from
        # ``moist_adiabat_lapse_rate`` (Iribarne–Godson) at the
        # entrained parcel state.  The function evaluates dT/dp on
        # the moist adiabat assuming the parcel is saturated; for
        # *unsaturated* parcels this is approximate but matches the
        # behavior of the existing ``compute_moist_adiabat`` helper
        # that we're benchmarked against.  Latent heating is already
        # baked into the lapse rate, so the post-hoc condensation
        # step below does NOT add an additional ``L_v/c_pd *
        # condensate`` correction — that would double-count.
        rho_u_ent = p_e / (constants.R_d * jnp.maximum(T_u_ent, 100.0))
        # ``moist_adiabat_lapse_rate`` returns dT/dp [K/Pa]; convert
        # to dT/dz [K/m] via dp/dz = -rho*g.
        Gamma_moist_per_pa = moist_adiabat_lapse_rate(T_u_ent, p_e)
        dT_dz = Gamma_moist_per_pa * (-rho_u_ent * g)

        T_u = T_u_ent + dT_dz * dz

        # Condense any super-saturation into cloud water.  This is the
        # diagnostic that resolves the q-budget; the temperature
        # already incorporates the latent heat from condensation via
        # the moist lapse rate.
        q_sat_new = saturation_mixing_ratio(T_u, p_e).astype(_dtype)
        condensate = jnp.maximum(q_u_ent - q_sat_new, 0.0).astype(_dtype)
        q_u = (q_u_ent - condensate).astype(_dtype)
        # Dilute plume cloud water by entrainment.  The continuity

 succeeded in 0ms:
"""Thompson hybrid-moment microphysics.

Extends Morrison with graupel formation from intense riming and
gamma distribution shape corrections for autoconversion/accretion.

All operations use smooth (differentiable) approximations.

References
----------
- Thompson, G., Field, P. R., Rasmussen, R. M., & Hall, W. D. (2008).
  Explicit forecasts of winter precipitation using an improved bulk
  microphysics scheme. Part II: Implementation of a new snow
  parameterization. Mon. Wea. Rev., 136, 5095-5115.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    saturation_adjustment,
    effective_Nc,
    autoconversion_sb,
    accretion,
    self_collection_breakup,
    rain_evaporation,
    safe_pow,
)
from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


def _gamma_ratio(mu):
    """Gamma(mu+4)/Gamma(mu+1) = (mu+3)(mu+2)(mu+1) for integer-like mu."""
    return (mu + 3.0) * (mu + 2.0) * (mu + 1.0)


def thompson_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: ThompsonConfig = ThompsonConfig(),
) -> MicrophysicsOutput:
    """Compute Thompson hybrid-moment microphysics tendencies.

    Parameters
    ----------
    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
        Same interface as all microphysics backends.

    Returns
    -------
    MicrophysicsOutput
    """
    ncol, nlev = T.shape
    q_c = hydrometeors.q_c
    q_r = hydrometeors.q_r
    q_i = hydrometeors.q_i
    q_s = hydrometeors.q_s
    q_g = hydrometeors.q_g
    N_c = hydrometeors.N_c
    N_r = hydrometeors.N_r
    N_i = hydrometeors.N_i
    sharpness = config.saturation_sharpness

    N_c_eff = effective_Nc(N_c, config.Nc_0)

    # === WARM RAIN ===
    # Saturation adjustment — convert increment [kg/kg] to tendency [kg/kg/s]
    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)

    # Gamma distribution corrections
    gamma_c = _gamma_ratio(config.mu_c)
    gamma_r = _gamma_ratio(config.mu_r)
    gamma_c_norm = gamma_c / _gamma_ratio(0.0)  # normalize to mu=0 baseline (=24)
    gamma_r_norm = gamma_r / _gamma_ratio(0.0)

    # Autoconversion (gamma-corrected)
    dq_c_au, dN_r_au, x_c = autoconversion_sb(
        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness, gamma_norm=gamma_c_norm,
    )

    # Accretion (gamma-corrected)
    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac, gamma_norm=gamma_r_norm)

    # Self-collection / breakup
    dN_r_sc, dN_r_br = self_collection_breakup(
        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    )

    # Rain evaporation
    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)

    # === ICE PHASE (Morrison processes) ===
    T_freeze = constants.T_freeze
    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))

    # Ice nucleation
    N_i_target = config.N_i0 * jnp.exp(
        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
    ) / jnp.clip(rho, 0.1)
    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)

    # Depositional growth
    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
    dq_i_dep = (
        config.dep_coeff
        * jnp.maximum(S_i, 0.0)
        * jnp.clip(q_i, 0.0)
        * safe_pow(N_i, 1.0 / 3.0)
        * f_ice
    )

    # Bergeron
    berg_window = (
        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
    )
    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window

    # Riming
    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
    total_riming = riming_i + riming_s

    # Aggregation
    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice

    # Melting (clamp to available mass so an explicit Euler step cannot
    # drive q_i / q_s / q_g negative — same pattern Morrison already uses).
    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
    dt_safe = jnp.maximum(dt, 1e-10)
    melt_ice = jnp.minimum(
        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
        jnp.clip(q_i, 0.0) / dt_safe,
    )
    melt_snow = jnp.minimum(
        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
        jnp.clip(q_s, 0.0) / dt_safe,
    )

    # === GRAUPEL (Thompson extension) ===
    # Split the rime → graupel conversion by donor: the fraction of
    # rime_to_graupel that comes from q_i scales with riming_i, and
    # the fraction from q_s scales with riming_s.  This avoids a
    # mass-leak corner case where ``q_i = 0`` and ``riming_s > 0``:
    # the previous form set ``rime_to_graupel ∝ total_riming``, then
    # subtracted the FULL value from ``q_i`` (driving it negative)
    # while only subtracting half from ``q_s`` and adding 150% to
    # ``q_g`` — a non-conservative split that depended on the
    # ad-hoc 1.0 / 0.5 / 1.5 coefficients.  (Codex audit cycle 2:
    # "Thompson graupel conversion can draw from the wrong donor".)
    graupel_frac = jax.nn.sigmoid(
        config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
    )
    rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
    rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
    melt_graupel = jnp.minimum(
        config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac,
        jnp.clip(q_g, 0.0) / dt_safe,
    )

    # === DONOR CLAMP for q_c sinks (see morrison.py for rationale) ===
    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s
    qc_avail = jnp.clip(q_c, 0.0)
    qc_scale = jnp.minimum(
        1.0,
        qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
    )
    dq_c_au = dq_c_au * qc_scale
    dq_c_ac = dq_c_ac * qc_scale
    bergeron = bergeron * qc_scale
    riming_i = riming_i * qc_scale
    riming_s = riming_s * qc_scale
    total_riming = riming_i + riming_s
    # Each rime-to-graupel donor scales with its parent riming term —
    # which has already been scaled by qc_scale above.  Re-scaling
    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``
    # by ``qc_scale`` once preserves both per-donor proportionality and
    # mass conservation.
    rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
    rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
    dN_r_au = dN_r_au * qc_scale

    # === SEDIMENTATION ===
    # Marshall-Palmer fall speeds use fractional exponents (b_v_x in
    # [0.25, 0.5]); guard the AD path with safe_pow.
    rho_sfc = rho[:, -1:]
    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
    V_t_g = config.a_v_g * safe_pow(jnp.clip(q_g, 0.0) * rho_ratio, config.b_v_g)
    V_t_g = jnp.clip(V_t_g, 0.0, 30.0)

    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
    sed_i = sedimentation_tendency(q_i, rho, V_t_i, dz)
    sed_s = sedimentation_tendency(q_s, rho, V_t_s, dz)
    sed_g = sedimentation_tendency(q_g, rho, V_t_g, dz)

    # === LATENT HEATING ===
    L_v = constants.L_v
    L_s = constants.L_s
    L_f = constants.L_f
    c_pd = constants.c_pd
    dT_dt = (
        L_v * condensation / c_pd
        - L_v * evaporation / c_pd
        + L_s * dq_i_dep / c_pd
        # Cloud water → ice/snow freezing releases L_f (Bergeron, riming).
        # See morrison.py for the moist-enthalpy rationale; Thompson
        # mirrors Morrison's ice-phase latent heating.
        + L_f * (bergeron + riming_i + riming_s) / c_pd
        - L_f * (melt_ice + melt_snow + melt_graupel) / c_pd
    )

    # === COMBINE TENDENCIES ===
    # Conservation: each rime-to-graupel donor leaves its parent
    # species and arrives in q_g.  The total mass moved is
    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``.
    # An earlier form used 1.0 / 0.5 / 1.5 splits on ``rime_to_graupel``
    # which (a) drove ``q_i`` negative when only snow was being rimed
    # (``q_i = 0`` but ``riming_s > 0``), and (b) created mass
    # apparently from nothing in the same regime.  See the
    # ``=== GRAUPEL ===`` block above for the donor-split rationale.
    dq_v_dt = -condensation + evaporation - dq_i_dep
    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel + sed_r
    dq_i_dt = (
        dq_i_dep + bergeron + riming_i - aggregation - melt_ice
        - rime_to_graupel_from_i + sed_i
    )
    dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel_from_s + sed_s
    dq_g_dt = rime_to_graupel - melt_graupel + sed_g

    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)

    # Precipitation
    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)
    precip_s = jnp.clip(q_s[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_s[:, -1], 0.0)
    precip_g = jnp.clip(q_g[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_g[:, -1], 0.0)
    precipitation = precip_r + precip_i + precip_s + precip_g

    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=dq_i_dt,
        dq_s_dt=dq_s_dt,
        dq_g_dt=dq_g_dt,
        dN_c_dt=dN_c_dt,
        dN_r_dt=dN_r_dt,
        dN_i_dt=dN_i_dt,
        precipitation=precipitation,
    )

exec
/bin/zsh -lc "rg -n \"decay_eps|rime_to_graupel|dN_i_dt|qc_sink_total|def step\\(\" src/legoesm/atmosphere/physics/convection/_plume.py src/legoesm/atmosphere/physics/microphysics/thompson.py tests/unit/test_convection_plume.py tests/unit/test_physics_microphysics.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/atmosphere/physics/convection/_plume.py:510:    def step(carry, layer_inputs):
src/legoesm/atmosphere/physics/convection/_plume.py:546:        decay_eps = jnp.exp(-eps * dz)
src/legoesm/atmosphere/physics/convection/_plume.py:547:        T_u_ent = T_e + (T_u_prev - T_e) * decay_eps
src/legoesm/atmosphere/physics/convection/_plume.py:548:        q_u_ent = q_e + (q_u_prev - q_e) * decay_eps
src/legoesm/atmosphere/physics/convection/_plume.py:595:        q_c_u_ent = q_c_u_prev * decay_eps
src/legoesm/atmosphere/physics/microphysics/thompson.py:157:    # rime_to_graupel that comes from q_i scales with riming_i, and
src/legoesm/atmosphere/physics/microphysics/thompson.py:160:    # the previous form set ``rime_to_graupel ∝ total_riming``, then
src/legoesm/atmosphere/physics/microphysics/thompson.py:167:        config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
src/legoesm/atmosphere/physics/microphysics/thompson.py:169:    rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
src/legoesm/atmosphere/physics/microphysics/thompson.py:170:    rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
src/legoesm/atmosphere/physics/microphysics/thompson.py:171:    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
src/legoesm/atmosphere/physics/microphysics/thompson.py:178:    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s
src/legoesm/atmosphere/physics/microphysics/thompson.py:182:        qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
src/legoesm/atmosphere/physics/microphysics/thompson.py:192:    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``
src/legoesm/atmosphere/physics/microphysics/thompson.py:195:    rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:196:    rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:197:    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
src/legoesm/atmosphere/physics/microphysics/thompson.py:238:    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``.
src/legoesm/atmosphere/physics/microphysics/thompson.py:239:    # An earlier form used 1.0 / 0.5 / 1.5 splits on ``rime_to_graupel``
src/legoesm/atmosphere/physics/microphysics/thompson.py:249:        - rime_to_graupel_from_i + sed_i
src/legoesm/atmosphere/physics/microphysics/thompson.py:251:    dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel_from_s + sed_s
src/legoesm/atmosphere/physics/microphysics/thompson.py:252:    dq_g_dt = rime_to_graupel - melt_graupel + sed_g
src/legoesm/atmosphere/physics/microphysics/thompson.py:256:    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
src/legoesm/atmosphere/physics/microphysics/thompson.py:275:        dN_i_dt=dN_i_dt,
tests/unit/test_physics_microphysics.py:341:def test_thompson_rime_to_graupel_donor_split():
tests/unit/test_physics_microphysics.py:352:        rime_to_graupel = rate * (riming_i + riming_s) > 0
tests/unit/test_physics_microphysics.py:353:        dq_i_dt -= rime_to_graupel              # full subtraction!
tests/unit/test_physics_microphysics.py:354:        dq_s_dt -= 0.5 * rime_to_graupel
tests/unit/test_physics_microphysics.py:355:        dq_g_dt += 1.5 * rime_to_graupel
tests/unit/test_physics_microphysics.py:359:    drained.  Total mass moved (``rime_to_graupel_from_i +
tests/unit/test_physics_microphysics.py:360:    rime_to_graupel_from_s``) goes 1:1 to graupel.
tests/unit/test_physics_microphysics.py:391:        "q_i=0 with q_s>0 and active riming.  Pre-fix the rime_to_graupel "
tests/unit/test_physics_microphysics.py:403:    # rime_to_graupel.  q_g production must be matched by q_s loss

codex
The actual files match the pasted fixes. I’m now checking whether the math identities hold after all the later clamps and whether the number tendencies have analogous donor-limit problems.
exec
/bin/zsh -lc "sed -n '575,635p' src/legoesm/atmosphere/physics/convection/_plume.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "sed -n '1,240p' src/legoesm/atmosphere/physics/microphysics/morrison.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "sed -n '320,430p' tests/unit/test_physics_microphysics.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "sed -n '250,380p' tests/unit/test_convection_plume.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
        # Dilute plume cloud water by entrainment.  The continuity
        # equation for an intensive quantity in an entraining-
        # detraining plume is ``dq_c/dz = -eps · q_c + cond/M`` —
        # environmental air carries q_c=0 so entrainment uniformly
        # decreases ``q_c_u`` while detrainment is intensively
        # neutral (it removes mass but not the per-kg amount).  An
        # earlier formulation ``q_c_u = q_c_u_prev + condensate``
        # carried ``q_c_u_prev`` forward unchanged and the plume's
        # total water grew unphysically aloft (audit Codex finding:
        # "plume cloud water is accumulated but not diluted by
        # entrainment").
        # Exponential dilution: ``q_c_u_ent = q_c_u_prev * exp(-eps·dz)``
        # — exact analytic solution to ``dq_c/dz = -eps · q_c`` for an
        # entraining plume with environment q_c=0.  Always non-negative,
        # AD-safe everywhere.  An earlier explicit-Euler form
        # ``max(q_c_u_prev * (1 - eps·dz), 0)`` zeroed the gradient
        # whenever ``eps·dz > 1`` (corner case at Bechtold's
        # ``epsilon_shallow=3e-3`` × dz=500 m and thicker — the clip
        # branch dominated and made the test case ineffective for AD-
        # tuning of ``eps`` in shallow convection).
        q_c_u_ent = q_c_u_prev * decay_eps
        q_c_u = (q_c_u_ent + condensate).astype(_dtype)

        T_u = T_u.astype(_dtype)

        # Buoyancy at this level.
        B_u = T_u - T_e

        # Reporting filters: smoothly suppress the mass flux below
        # cloud base (``abv``) and where the plume has lost buoyancy
        # (``plume_alive``).  These do NOT enter the carry.
        plume_alive = jax.nn.sigmoid(buoyancy_sharpness * B_u)
        M_u_reported = M_u_raw * plume_alive * abv

        new_carry = (T_u, q_u, q_c_u, M_u_raw, z_e)
        outputs = (T_u, q_u, q_c_u, M_u_reported, B_u)
        return new_carry, outputs

    _, scan_out = jax.lax.scan(step, init_carry, inputs)

    T_u_rev, q_u_rev, q_c_u_rev, M_u_rev, B_u_rev = scan_out  # (nlev, ncol)

    # Move axis back and reverse to surface-last.
    T_u = jnp.moveaxis(T_u_rev, 0, 1)[:, ::-1]
    q_u = jnp.moveaxis(q_u_rev, 0, 1)[:, ::-1]
    q_c_u = jnp.moveaxis(q_c_u_rev, 0, 1)[:, ::-1]
    M_u = jnp.moveaxis(M_u_rev, 0, 1)[:, ::-1]
    B_u = jnp.moveaxis(B_u_rev, 0, 1)[:, ::-1]

    return Plume(M_u=M_u, T_u=T_u, q_u=q_u, q_c_u=q_c_u, B_u=B_u)


# ---------------------------------------------------------------------------
# Convective momentum transport (Gregory et al. 1997)
# ---------------------------------------------------------------------------

def cmt_gregory_1997(
    u_env: jax.Array,
    v_env: jax.Array,
    M_u: jax.Array,
    M_d: jax.Array | None,

 succeeded in 0ms:
"""Morrison double-moment ice+liquid microphysics.

Extends Seifert-Beheng warm-rain with ice-phase processes: nucleation
(Cooper 1986), depositional growth, Bergeron process, riming, snow
aggregation, and melting. Tracks cloud water, rain, ice, and snow.

All operations use smooth (differentiable) approximations.

References
----------
- Morrison, H., Curry, J. A., & Khvorostyanov, V. I. (2005). A new
  double-moment microphysics parameterization. Part I: Description.
  J. Atmos. Sci., 62, 1665-1677.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    saturation_adjustment,
    effective_Nc,
    autoconversion_sb,
    accretion,
    self_collection_breakup,
    rain_evaporation,
    safe_pow,
)
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


def morrison_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: MorrisonConfig = MorrisonConfig(),
) -> MicrophysicsOutput:
    """Compute Morrison double-moment microphysics tendencies.

    Parameters
    ----------
    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
        Same interface as all microphysics backends.

    Returns
    -------
    MicrophysicsOutput
    """
    ncol, nlev = T.shape
    q_c = hydrometeors.q_c
    q_r = hydrometeors.q_r
    q_i = hydrometeors.q_i
    q_s = hydrometeors.q_s
    N_c = hydrometeors.N_c
    N_r = hydrometeors.N_r
    N_i = hydrometeors.N_i
    sharpness = config.saturation_sharpness

    N_c_eff = effective_Nc(N_c, config.Nc_0)

    # === WARM RAIN (shared Seifert-Beheng helpers) ===
    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
    dq_c_au, dN_r_au, x_c = autoconversion_sb(
        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness,
    )
    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac)
    dN_r_sc, dN_r_br = self_collection_breakup(
        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    )
    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)

    # === ICE PHASE ===
    T_freeze = constants.T_freeze
    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))

    # 1. Ice nucleation (Cooper 1986, smoothed)
    N_i_target = config.N_i0 * jnp.exp(
        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
    ) / jnp.clip(rho, 0.1)
    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)

    # 2. Depositional growth
    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
    dq_i_dep = (
        config.dep_coeff
        * jnp.maximum(S_i, 0.0)
        * jnp.clip(q_i, 0.0)
        * safe_pow(N_i, 1.0 / 3.0)
        * f_ice
    )

    # 3. Bergeron process: cloud water -> ice in mixed-phase zone
    berg_window = (
        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
    )
    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window

    # 4. Riming: ice/snow collect cloud water
    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice

    # 5. Snow aggregation: ice -> snow
    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice

    # 6. Melting near T_freeze: ice/snow -> rain (clipped to available mass)
    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
    melt_ice = jnp.minimum(
        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
        jnp.clip(q_i, 0.0) / jnp.maximum(dt, 1e-10),
    )
    melt_snow = jnp.minimum(
        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
        jnp.clip(q_s, 0.0) / jnp.maximum(dt, 1e-10),
    )

    # === DONOR CLAMP for q_c sinks ===
    # Scale q_c-consuming processes (autoconversion, accretion, Bergeron,
    # riming) by a common factor so the total loss per timestep does not
    # exceed available q_c.  Without this clamp, default rates at dt = 1200s
    # in a mixed-phase column drive q_c negative on a single explicit step
    # (bergeron alone gives bergeron_rate * q_c * dt = 1.2 * q_c).  Mass is
    # conserved because each process's matching source term in dq_r/dq_i/dq_s
    # gets the same scale factor (the rates appear once as sinks in dq_c and
    # once as sources elsewhere, so a uniform rescale preserves the budget).
    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s
    qc_avail = jnp.clip(q_c, 0.0)
    qc_scale = jnp.minimum(
        1.0,
        qc_avail / jnp.maximum(qc_sink_total * jnp.maximum(dt, 1e-10), 1e-30),
    )
    dq_c_au = dq_c_au * qc_scale
    dq_c_ac = dq_c_ac * qc_scale
    bergeron = bergeron * qc_scale
    riming_i = riming_i * qc_scale
    riming_s = riming_s * qc_scale
    # Number tendency for autoconverted droplets must scale identically.
    dN_r_au = dN_r_au * qc_scale

    # === SEDIMENTATION ===
    # Marshall-Palmer fall speeds V_t = a_v * (q * rho / rho_sfc)^b_v use
    # fractional exponents (b_v_r=0.5, b_v_i=0.25, b_v_s=0.3); guard the
    # AD path with safe_pow so cold-start columns (q=0) don't NaN gradients.
    rho_sfc = rho[:, -1:]
    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)

    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
    sed_i = sedimentation_tendency(q_i, rho, V_t_i, dz)
    sed_s = sedimentation_tendency(q_s, rho, V_t_s, dz)

    # === LATENT HEATING ===
    L_v = constants.L_v
    L_s = constants.L_s
    L_f = constants.L_f
    c_pd = constants.c_pd

    dT_dt = (
        L_v * condensation / c_pd
        - L_v * evaporation / c_pd
        + L_s * dq_i_dep / c_pd
        # Cloud water → ice/snow freezing releases latent heat of fusion
        # (~333 kJ/kg).  Bergeron is liquid → ice via the WBF mechanism,
        # riming is supercooled-droplet capture by ice/snow.  Both are
        # phase changes that release L_f; the moist-enthalpy invariant
        # ``h = c_pd T + L_v q_v - L_f q_ice`` requires this term for
        # column conservation.  Magnitude estimate: ~2 K/day at default
        # rates in mixed-phase clouds.
        + L_f * (bergeron + riming_i + riming_s) / c_pd
        - L_f * (melt_ice + melt_snow) / c_pd
    )

    # === COMBINE TENDENCIES ===
    dq_v_dt = -condensation + evaporation - dq_i_dep
    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + sed_r
    dq_i_dt = dq_i_dep + bergeron + riming_i - aggregation - melt_ice + sed_i
    dq_s_dt = aggregation + riming_s - melt_snow + sed_s

    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)

    # Precipitation (rain + ice + snow at surface)
    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)
    precip_s = jnp.clip(q_s[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_s[:, -1], 0.0)
    precipitation = precip_r + precip_i + precip_s

    # Pin dtype to the input precision so we never silently promote
    # the unused-species placeholders to f64 under x64 mode.
    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=dq_i_dt,
        dq_s_dt=dq_s_dt,
        dq_g_dt=z,
        dN_c_dt=dN_c_dt,
        dN_r_dt=dN_r_dt,
        dN_i_dt=dN_i_dt,
        precipitation=precipitation,
    )

 succeeded in 0ms:
        N_r=jnp.zeros_like(q_c),
        N_i=1e4 * jnp.ones_like(q_c),
    )
    q_v = 0.8 * saturation_mixing_ratio(T, p_full)
    dt = 1200.0
    out = thompson_microphysics(
        T, q_v, hydro, p_full, p_half, rho, dz, dt, ThompsonConfig(),
    )
    q_i_after = q_i + out.dq_i_dt * dt
    # Skip the top level: with q_i uniform throughout the column the top
    # level always loses sedimentation flux without compensating inflow,
    # which is a sedimentation-CFL concern rather than a melt-clamp one.
    # The melt-clamp bug shows up uniformly in interior levels.
    min_qi_interior = float(jnp.min(q_i_after[:, 1:]))
    assert min_qi_interior >= -1e-9, (
        f"Thompson: interior q_i went negative ({min_qi_interior:.3e}) at "
        f"T=280 K with dt=1200s — melt rate × q_i × melt_frac × dt exceeded "
        "q_i without a donor clamp."
    )


def test_thompson_rime_to_graupel_donor_split():
    """Audit cycle 2 (Codex): the Thompson rime-to-graupel conversion
    must subtract from the SOURCE species (q_i for ``riming_i``,
    q_s for ``riming_s``), not split via fixed 1.0 / 0.5 / 1.5
    coefficients on the total.

    Pathological column: ``q_c > 0`` (cloud water source for riming),
    ``q_i = 0`` (no ice to be rimed), ``q_s > 0`` (snow that gets
    rimed by cloud water), ``T < T_freeze`` (active ice phase).  In
    this column ``riming_i = 0`` (no q_i to rime) and ``riming_s > 0``
    (q_c × q_s × f_ice).  Pre-fix:
        rime_to_graupel = rate * (riming_i + riming_s) > 0
        dq_i_dt -= rime_to_graupel              # full subtraction!
        dq_s_dt -= 0.5 * rime_to_graupel
        dq_g_dt += 1.5 * rime_to_graupel
    drives ``q_i`` negative and creates 1.5× extra mass — both
    conservation violations.  Post-fix the donor split scales by
    ``riming_i / riming_s`` and only the actually-rimed species is
    drained.  Total mass moved (``rime_to_graupel_from_i +
    rime_to_graupel_from_s``) goes 1:1 to graupel.
    """
    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 250.0)  # below freezing — ice active
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 10_000.0)
    # The pathological mix: cloud water + snow, NO ice.
    q_c = jnp.full((ncol, nlev), 5e-3)
    q_i = jnp.zeros((ncol, nlev))                # zero ice — would be drained negative pre-fix
    q_s = jnp.full((ncol, nlev), 1e-3)            # snow gets rimed
    hydro = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
        q_g=jnp.zeros_like(q_c),
        N_c=1e8 * jnp.ones_like(q_c),
        N_r=jnp.zeros_like(q_c),
        N_i=jnp.zeros_like(q_c),
    )
    q_v = 0.5 * saturation_mixing_ratio(T, p_full)
    dt = 1200.0
    out = thompson_microphysics(
        T, q_v, hydro, p_full, p_half, rho, dz, dt, ThompsonConfig(),
    )
    q_i_after = q_i + out.dq_i_dt * dt
    # q_i must remain non-negative (donor split: only riming_i drains q_i).
    min_qi = float(jnp.min(q_i_after))
    assert min_qi >= -1e-9, (
        f"Thompson: q_i went negative ({min_qi:.3e}) when starting at "
        "q_i=0 with q_s>0 and active riming.  Pre-fix the rime_to_graupel "
        "subtracted the full conversion from q_i regardless of which "
        "species was actually rimed.  Audit cycle 2 Codex finding "
        "'Thompson graupel conversion can draw from the wrong donor' "
        "has regressed."
    )
    # Mass conservation: ∑ dq_i + dq_s + dq_g (pure ice phase) should
    # equal sed_i + sed_s + sed_g (sedimentation only escapes the column);
    # internal phase changes cancel in the sum.  We can verify rime->graupel
    # specifically by checking that any q_g gain matches a q_s loss.
    dq_g_total = float(jnp.sum(out.dq_g_dt))
    # Without melting (T well below freeze), all q_g must come from
    # rime_to_graupel.  q_g production must be matched by q_s loss
    # (donor split: only riming_s active here).
    assert dq_g_total >= 0.0, (
        f"Thompson: dq_g_dt total ({dq_g_total:.3e}) is negative "
        "without graupel sedimentation source — graupel mass conservation "
        "violation."
    )


# ============================================================================
# Morrison/Thompson moist-enthalpy conservation (latent heat of fusion)
# ============================================================================

@pytest.mark.parametrize(
    "scheme,call",
    [("morrison", morrison_microphysics), ("thompson", thompson_microphysics)],
)
def test_freezing_releases_latent_heat_of_fusion(scheme, call):
    """Bergeron + riming (cloud water → ice/snow) must release ``L_f``.

    Moist enthalpy ``h = c_pd T + L_v q_v - L_f * q_ice`` is conserved
    by phase transitions in a closed column, so per-level the residual
        c_pd * dT_dt + L_v * dq_v_dt - L_f * (dq_i + dq_s + dq_g)
    is the divergence of sedimentation flux (surface boundary effect),
    not a phase-change heating/cooling.

    Pre-fix the dT_dt assembly omitted the ``+ L_f * (bergeron +
    riming_i + riming_s)`` term, so in a mixed-phase column where

 succeeded in 0ms:
    T_env = jnp.full((ncol, nlev), 280.0)
    T_parcel = jnp.full((ncol, nlev), 280.0).at[:, 6:8].set(278.0)
    p_half = jnp.broadcast_to(
        jnp.linspace(1e4, 1e5, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    k_lcl = jnp.array([8.0])
    k_lfc = jnp.array([5.0])
    cin = P.compute_cin(
        T_env, T_parcel, p_full, p_half, k_lcl, k_lfc,
        indicator_sharpness=5.0,
    )
    # Analytical CIN over k=6,7 only
    dp = p_half[:, 1:] - p_half[:, :-1]
    cin_analytical = constants.R_d * jnp.sum(
        jnp.maximum(0.0, T_env - T_parcel) * dp / p_full, axis=-1,
    )
    rel_err = float(jnp.max(jnp.abs(cin - cin_analytical) / cin_analytical))
    # Pre-fix: rel_err ~ 0.93 (CIN ≈ 7% of analytical).
    # Post-fix: should match within ~30% (smooth window has soft edges).
    assert rel_err < 0.3, (
        f"CIN = {float(cin[0]):.3e} vs analytical = "
        f"{float(cin_analytical[0]):.3e}; rel_err = {rel_err:.2f}. "
        "Pre-fix the CIN window factors selected the WRONG sides "
        "(`above_LFC * below_LCL` is empty for k_lnb<k_lfc<k_lcl)."
    )


def test_compute_cin_zero_for_unstable_parcel_above_LCL():
    """When the parcel is positively buoyant immediately above the
    LCL (no cap), CIN ≈ 0."""
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
        T_sfc=302.0, q_sfc=18.0e-3, lapse_rate_K_per_km=8.0,
    )
    T_ma = compute_moist_adiabat(T_env[:, -1], p_full)
    lcl = P.compute_lcl(T_env[:, -1], q_v_env[:, -1], p_full[:, -1], p_full)
    k_lfc, _ = P.compute_lfc_lnb(T_env, T_ma, sharpness=1.0)
    cin = P.compute_cin(T_env, T_ma, p_full, p_half, lcl.k_lcl_smooth, k_lfc)
    assert float(cin[0]) < 5.0   # J/kg


# ---------------------------------------------------------------------------
# entraining_detraining_plume
# ---------------------------------------------------------------------------

def test_plume_outputs_finite_and_correct_shape():
    """Smoke: every output array is finite with the expected
    ``(ncol, nlev)`` shape."""
    ncol, nlev = 3, 14
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]
    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
    eps = jnp.full((ncol, nlev), 5.0e-4)
    dlt = jnp.full((ncol, nlev), 5.0e-4)
    M_b = jnp.full((ncol,), 0.05)

    plume = P.entraining_detraining_plume(
        T_env, q_v_env, p_full, p_half, z_full,
        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
    )
    for arr in (plume.M_u, plume.T_u, plume.q_u, plume.q_c_u, plume.B_u):
        assert arr.shape == (ncol, nlev)
        assert jnp.all(jnp.isfinite(arr))


def test_plume_no_entrainment_limit_matches_moist_adiabat():
    """With ``epsilon = delta = 0`` and a strongly buoyant parcel,
    the in-cloud plume temperature follows the moist adiabat (within
    a tolerance set by the discretization)."""
    ncol, nlev = 1, 16
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
        ncol, nlev, T_sfc=300.0, q_sfc=18.0e-3, lapse_rate_K_per_km=8.0,
    )
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]

    # Cloud base at surface.  With unsaturated air the plume condenses
    # immediately above the surface; the moist adiabat starts from
    # T_base.
    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
    T_ma = compute_moist_adiabat(T_base, p_full)

    eps = jnp.zeros((ncol, nlev))
    dlt = jnp.zeros((ncol, nlev))
    M_b = jnp.full((ncol,), 0.05)

    plume = P.entraining_detraining_plume(
        T_env, q_v_env, p_full, p_half, z_full,
        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
    )
    # Sample three mid-troposphere levels and compare to the moist
    # adiabat.  The first-order Euler ascent in the plume integrator
    # accumulates discretization error, so the tolerance is generous.
    sampled = jnp.array([nlev // 2, nlev // 2 + 1, nlev // 2 + 2])
    plume_T = plume.T_u[0, sampled]
    ma_T = T_ma[0, sampled]
    diff = jnp.abs(plume_T - ma_T)
    assert jnp.all(diff < 5.0), (
        f"No-entrainment plume should track moist adiabat within ~5 K, "
        f"got differences {np.asarray(diff)} K"
    )


def test_plume_sub_cloud_mass_flux_is_suppressed():
    """Levels strictly below the cloud base have ``M_u ≈ 0`` because
    the ``above_base_weight`` mask suppresses them."""
    ncol, nlev = 1, 16
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]
    # Force a cloud-base index well above the surface by passing a
    # synthetic ``k_base_smooth`` (mid-column) so we have several
    # sub-cloud levels to inspect.
    k_base_synthetic = jnp.full((ncol,), float(nlev) - 4.0)

    eps = jnp.full((ncol, nlev), 5.0e-4)
    dlt = jnp.full((ncol, nlev), 5.0e-4)
    M_b = jnp.full((ncol,), 0.05)

    plume = P.entraining_detraining_plume(
        T_env, q_v_env, p_full, p_half, z_full,
        T_base, q_base, k_base_synthetic, eps, dlt, M_b,
        buoyancy_sharpness=2.0,
    )
    # Surface-last: levels with index > k_base are below the cloud
    # base (lower altitude).
    sub_cloud_mass = plume.M_u[:, -1]   # surface-most level
    base_mass = plume.M_u[:, int(float(nlev) - 4.0)]
    # Sub-cloud value should be much smaller than the cloud-base
    # value.  Tolerance reflects the smoothness of the sigmoid mask.

exec
/bin/zsh -lc "sed -n '1,260p' src/legoesm/atmosphere/physics/microphysics/config.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "sed -n '1,220p' src/legoesm/atmosphere/physics/microphysics/output.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "sed -n '1,260p' src/legoesm/atmosphere/physics/microphysics/_warm_rain.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
"""Microphysics output containers.

HydrometeorState holds the prognostic hydrometeor fields passed to backends.
MicrophysicsOutput is the common interface returned by all backends.

All backends accept and return the same containers so that integration
code can be backend-agnostic.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class HydrometeorState(NamedTuple):
    """Hydrometeor state for backends. All fields shape (ncol, nlev)."""
    q_c: jax.Array    # cloud water [kg/kg]
    q_r: jax.Array    # rain water [kg/kg]
    q_i: jax.Array    # cloud ice [kg/kg]
    q_s: jax.Array    # snow [kg/kg]
    q_g: jax.Array    # graupel [kg/kg]
    N_c: jax.Array    # cloud droplet number [1/kg]
    N_r: jax.Array    # rain drop number [1/kg]
    N_i: jax.Array    # ice crystal number [1/kg]


class MicrophysicsOutput(NamedTuple):
    """Backend-agnostic output. All (ncol, nlev) except precipitation (ncol,)."""
    dT_dt: jax.Array          # latent heating [K/s]
    dq_v_dt: jax.Array        # vapor tendency [kg/kg/s]
    dq_c_dt: jax.Array        # cloud water tendency
    dq_r_dt: jax.Array        # rain tendency
    dq_i_dt: jax.Array        # ice tendency
    dq_s_dt: jax.Array        # snow tendency
    dq_g_dt: jax.Array        # graupel tendency
    dN_c_dt: jax.Array        # cloud number tendency [1/kg/s]
    dN_r_dt: jax.Array        # rain number tendency
    dN_i_dt: jax.Array        # ice number tendency
    precipitation: jax.Array  # surface precip [kg/m^2/s]


def make_zero_hydrometeors(
    ncol: int, nlev: int, dtype=None,
) -> HydrometeorState:
    """Create a zero-initialized HydrometeorState.

    ``dtype`` defaults to the JAX default float (``float64`` under x64,
    ``float32`` otherwise).  Callers integrating with the column physics
    pipeline should pass the upstream state dtype explicitly so this
    fallback never silently promotes a float32 column path to float64.
    """
    z = jnp.zeros((ncol, nlev), dtype=dtype)
    return HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=z,
    )


def make_zero_output(
    ncol: int, nlev: int, dtype=None,
) -> MicrophysicsOutput:
    """Create a zero-initialized MicrophysicsOutput.

    ``dtype`` is forwarded to ``jnp.zeros`` for the same reason as
    ``make_zero_hydrometeors``: defaulting allows x64 mode to silently
    promote the precip path.
    """
    z2 = jnp.zeros((ncol, nlev), dtype=dtype)
    z1 = jnp.zeros((ncol,), dtype=dtype)
    return MicrophysicsOutput(
        dT_dt=z2, dq_v_dt=z2, dq_c_dt=z2, dq_r_dt=z2,
        dq_i_dt=z2, dq_s_dt=z2, dq_g_dt=z2,
        dN_c_dt=z2, dN_r_dt=z2, dN_i_dt=z2,
        precipitation=z1,
    )


def sedimentation_tendency(
    q: jax.Array,
    rho: jax.Array,
    V_t: jax.Array,
    dz: jax.Array,
) -> jax.Array:
    """Compute sedimentation tendency from vertical flux divergence.

    Parameters
    ----------
    q : jax.Array
        Hydrometeor mixing ratio [kg/kg], shape (ncol, nlev).
    rho : jax.Array
        Air density [kg/m^3], shape (ncol, nlev).
    V_t : jax.Array
        Terminal velocity [m/s], shape (ncol, nlev).
    dz : jax.Array
        Layer thickness [m], shape (ncol, nlev).

    Returns
    -------
    jax.Array
        Sedimentation tendency [kg/kg/s], shape (ncol, nlev).
    """
    q_pos = jnp.clip(q, 0.0, None)
    flux = V_t * q_pos * rho  # (ncol, nlev)

    # Flux from above: zero at top, flux[k-1] enters level k.  Use
    # ``jnp.pad`` (single Pad HLO) instead of allocating a fresh
    # zero buffer + concatenate.
    flux_in = jnp.pad(flux[:, :-1], ((0, 0), (1, 0)))
    dz_safe = jnp.clip(dz, 1.0, None)
    return (flux_in - flux) / (rho * dz_safe)

 succeeded in 0ms:
"""Configuration for atmospheric microphysics schemes.

Provides configuration NamedTuples for:
1. Kessler — warm-rain one-moment (refactored from physics/kessler.py)
2. Sundqvist — large-scale diagnostic condensation
3. Seifert-Beheng — two-moment warm rain
4. Morrison — double-moment ice+liquid
5. Thompson — hybrid moment with graupel
6. ML Emulator — Equinox MLP surrogate
7. Top-level MicrophysicsConfig that selects the active scheme.

References
----------
- Kessler (1969): On the Distribution and Continuity of Water Substance.
- Sundqvist et al. (1989): Condensation and cloud parameterization studies.
- Seifert & Beheng (2001): A two-moment cloud microphysics scheme.
- Morrison et al. (2005): A new double-moment microphysics scheme.
- Thompson et al. (2008): Explicit forecasts of winter precipitation.
"""

from __future__ import annotations

from typing import NamedTuple


class KesslerConfig(NamedTuple):
    """Configuration for Kessler warm-rain microphysics."""
    autoconversion_threshold: float = 1.0e-3   # q_c threshold [kg/kg]
    autoconversion_rate: float = 1.0e-3         # Rate [1/s]
    accretion_coeff: float = 2.2                # Collection coefficient
    evaporation_coeff: float = 1.0              # Evaporation coefficient
    rain_fall_speed: float = 5.0                # Terminal velocity [m/s]
    saturation_sharpness: float = 100.0         # Smooth switch sharpness


class SundqvistConfig(NamedTuple):
    """Configuration for Sundqvist large-scale condensation."""
    RH_crit: float = 0.8              # Critical relative humidity
    sigmoid_sharpness: float = 20.0   # Sharpness for smooth activation
    auto_rate: float = 1e-3           # Autoconversion rate [1/s]
    evap_coeff: float = 5e-4          # Sub-cloud evaporation coefficient


class SeifertBehengConfig(NamedTuple):
    """Configuration for Seifert-Beheng two-moment warm rain."""
    k_au: float = 6e2                # Autoconversion rate [1/(kg*s)]
    x_star: float = 2.6e-10          # Separation mass [kg]
    Nc_0: float = 1e8                # Initial cloud droplet number [1/kg]
    k_ac: float = 5.25               # Accretion rate [m^3/(kg*s)]
    k_sc: float = 1e-3               # Self-collection rate [m^3/(kg*s)]
    D_eq: float = 1.1e-3             # Equilibrium breakup diameter [m]
    breakup_sharpness: float = 1e4   # Sigmoid sharpness for breakup
    a_v_r: float = 130.0             # Rain fall speed coefficient a [m^(1-b)/s]
    b_v_r: float = 0.5               # Rain fall speed exponent b
    evap_coeff: float = 1.0          # Evaporation coefficient
    saturation_sharpness: float = 100.0  # Sigmoid sharpness for saturation


class MorrisonConfig(NamedTuple):
    """Configuration for Morrison double-moment (ice+liquid)."""
    # Warm rain (same as SB)
    k_au: float = 6e2
    x_star: float = 2.6e-10
    Nc_0: float = 1e8
    k_ac: float = 5.25
    k_sc: float = 1e-3
    D_eq: float = 1.1e-3
    breakup_sharpness: float = 1e4
    a_v_r: float = 130.0
    b_v_r: float = 0.5
    evap_coeff: float = 1.0
    saturation_sharpness: float = 100.0
    # Ice nucleation (Cooper 1986)
    N_i0: float = 5e3               # Base ice crystal number [1/m^3]
    cooper_a: float = 0.304          # Cooper exponent
    cooper_T_act: float = 265.0      # Activation temperature [K]
    ice_sigmoid_sharpness: float = 5.0  # Sharpness for ice-liquid partition
    # Depositional growth
    dep_coeff: float = 1e-3          # Deposition growth coefficient
    # Bergeron
    bergeron_rate: float = 1e-3      # Bergeron conversion rate [1/s]
    T_center: float = 258.0          # Bergeron T window center [K]
    T_width: float = 10.0            # Bergeron T window width [K]
    # Riming
    rime_coeff: float = 1.0          # Riming collection efficiency
    # Aggregation
    agg_coeff: float = 1e-3          # Ice-to-snow aggregation rate [1/s]
    # Melting
    melt_rate: float = 5e-3          # Melting rate [1/s]
    melt_sharpness: float = 2.0      # Sigmoid sharpness near T_freeze
    # Ice sedimentation
    a_v_i: float = 50.0              # Ice fall speed coefficient [m^(1-b)/s]
    b_v_i: float = 0.25              # Ice fall speed exponent
    # Snow sedimentation
    a_v_s: float = 30.0              # Snow fall speed coefficient
    b_v_s: float = 0.3               # Snow fall speed exponent


class ThompsonConfig(NamedTuple):
    """Configuration for Thompson hybrid-moment microphysics."""
    # All Morrison params
    k_au: float = 6e2
    x_star: float = 2.6e-10
    Nc_0: float = 1e8
    k_ac: float = 5.25
    k_sc: float = 1e-3
    D_eq: float = 1.1e-3
    breakup_sharpness: float = 1e4
    a_v_r: float = 130.0
    b_v_r: float = 0.5
    evap_coeff: float = 1.0
    saturation_sharpness: float = 100.0
    N_i0: float = 5e3
    cooper_a: float = 0.304
    cooper_T_act: float = 265.0
    ice_sigmoid_sharpness: float = 5.0
    dep_coeff: float = 1e-3
    bergeron_rate: float = 1e-3
    T_center: float = 258.0
    T_width: float = 10.0
    rime_coeff: float = 1.0
    agg_coeff: float = 1e-3
    melt_rate: float = 5e-3
    melt_sharpness: float = 2.0
    a_v_i: float = 50.0
    b_v_i: float = 0.25
    a_v_s: float = 30.0
    b_v_s: float = 0.3
    # Graupel
    rime_to_graupel_threshold: float = 1e-4  # Riming threshold for graupel [kg/kg/s]
    rime_to_graupel_rate: float = 0.5        # Fraction converted to graupel
    graupel_sharpness: float = 1e4           # Sigmoid sharpness
    a_v_g: float = 80.0                      # Graupel fall speed coefficient
    b_v_g: float = 0.4                       # Graupel fall speed exponent
    # Gamma distribution shape
    mu_c: float = 3.0                        # Cloud droplet shape parameter
    mu_r: float = 1.0                        # Rain drop shape parameter


class MLEmulatorConfig(NamedTuple):
    """Configuration for ML microphysics emulator (Equinox MLP)."""
    n_input: int = 9
    n_hidden: int = 128
    n_layers: int = 3
    n_output: int = 7
    seed: int = 0
    use_residual: bool = True
    norm_T: float = 300.0       # Temperature scale [K] for input normalization
    norm_q_factor: float = 1e3  # q_v / q_c / q_r / q_i scale
    norm_rho: float = 1.2       # Air density scale [kg/m^3]
    norm_dz: float = 1000.0     # Layer thickness scale [m]
    norm_dt: float = 3600.0     # Time-step scale [s]


class MicrophysicsConfig(NamedTuple):
    """Top-level microphysics configuration.

    Selects the active scheme and holds sub-configurations.

    Fields
    ------
    scheme : str
        Active scheme: "kessler", "sundqvist", "seifert_beheng",
        "morrison", "thompson", "ml_emulator", or "none".
    kessler : KesslerConfig
    sundqvist : SundqvistConfig
    seifert_beheng : SeifertBehengConfig
    morrison : MorrisonConfig
    thompson : ThompsonConfig
    ml_emulator : MLEmulatorConfig
    """
    scheme: str = "none"
    kessler: KesslerConfig = KesslerConfig()
    sundqvist: SundqvistConfig = SundqvistConfig()
    seifert_beheng: SeifertBehengConfig = SeifertBehengConfig()
    morrison: MorrisonConfig = MorrisonConfig()
    thompson: ThompsonConfig = ThompsonConfig()
    ml_emulator: MLEmulatorConfig = MLEmulatorConfig()

 succeeded in 0ms:
"""Shared warm-rain microphysics helpers.

Functions here are used by multiple microphysics backends (Seifert-Beheng,
Morrison, Thompson, Kessler) to avoid duplicating identical physics code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio


def safe_pow(x, p):
    """Differentiable ``x ** p`` with grad=0 wherever ``x <= 0``.

    Microphysics has many Marshall-Palmer-style fractional powers of
    hydrometeor mixing ratios (``q_r``, ``q_i``, ``q_s``, ``q_g``,
    ``N_i``, …) with exponents in (0, 1) — typically 0.5 for fall
    speeds, 1/3 for diameters, 0.525/0.875 for ventilation/accretion.
    Their analytic derivative ``p * x**(p-1)`` is unbounded at ``x=0``
    and complex for ``x<0``.  ``jnp.clip(x, 0.0) ** p`` therefore
    returns ``inf`` (at zero) or ``nan`` (at negatives) under
    ``jax.grad``, breaking AD on cold-start (no-precip) initial
    conditions.

    The double-where pattern below routes the AD graph through a
    placeholder of 1.0 in the inactive branch so the gradient never
    sees ``0**(p-1)``.

    Parameters
    ----------
    x : array
        Argument of the power.  May be zero or negative.
    p : float or array
        Exponent.  Intended for ``0 < p < 1`` where the bug applies;
        also safe for ``p >= 1``.

    Returns
    -------
    array
        ``x ** p`` for ``x > 0``, else 0; gradient is finite
        everywhere.
    """
    positive = x > 0.0
    safe_x = jnp.where(positive, x, 1.0)
    return jnp.where(positive, safe_x ** p, 0.0)


def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0):
    """Compute smooth saturation adjustment (condensation tendency).

    Parameters
    ----------
    T : array (ncol, nlev)
        Temperature [K].
    q_v : array (ncol, nlev)
        Water vapor mixing ratio [kg/kg].
    p_full : array (ncol, nlev)
        Pressure [Pa].
    dt : float
        Time step [s].
    sharpness : float
        Sigmoid sharpness for smooth condensation switch.

    Returns
    -------
    condensation : array (ncol, nlev)
        Condensation tendency [kg/kg/s].
    q_sat : array (ncol, nlev)
        Saturation mixing ratio [kg/kg].
    """
    q_sat = saturation_mixing_ratio(T, p_full)
    excess = q_v - q_sat
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    condensation = cond_frac * excess / dt
    return condensation, q_sat


def effective_Nc(N_c, Nc_0):
    """Use config default cloud droplet number where N_c is zero.

    Parameters
    ----------
    N_c : array
        Cloud droplet number concentration [1/kg].
    Nc_0 : float
        Default cloud droplet number.

    Returns
    -------
    array : Effective N_c.
    """
    return jnp.where(N_c > 1.0, N_c, Nc_0 * jnp.ones_like(N_c))


def autoconversion_sb(q_c, N_c_eff, rho, k_au, x_star, sharpness=50.0, gamma_norm=1.0):
    """Seifert-Beheng mass-dependent autoconversion.

    Parameters
    ----------
    q_c : array
        Cloud water mixing ratio [kg/kg].
    N_c_eff : array
        Effective cloud droplet number [1/kg].
    rho : array
        Air density [kg/m3].
    k_au : float
        Autoconversion rate constant.
    x_star : float
        Mean droplet mass threshold [kg].
    sharpness : float
        Sigmoid sharpness.
    gamma_norm : float
        Gamma distribution correction (1.0 for SB/Morrison, != 1.0 for Thompson).

    Returns
    -------
    dq_c_au : array
        Cloud water autoconversion rate [kg/kg/s].
    dN_r_au : array
        Rain number formation rate [1/kg/s].
    x_c : array
        Mean cloud droplet mass [kg].
    """
    q_c_pos = jnp.clip(q_c, 0.0)
    x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)
    onset = jax.nn.sigmoid(sharpness * (x_c - x_star))
    dq_c_au = k_au * q_c_pos ** 2 * onset * gamma_norm * rho
    dN_r_au = dq_c_au * rho / (x_star * 20.0)
    return dq_c_au, dN_r_au, x_c


def accretion(q_c, q_r, rho, k_ac, gamma_norm=1.0):
    """Rain collecting cloud water (accretion).

    Parameters
    ----------
    q_c, q_r : array
        Cloud water and rain mixing ratios [kg/kg].
    rho : array
        Air density [kg/m3].
    k_ac : float
        Accretion rate constant.
    gamma_norm : float
        Gamma distribution correction.

    Returns
    -------
    array : Accretion rate [kg/kg/s].
    """
    return k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho * gamma_norm


def self_collection_breakup(N_r, q_r, rho, k_sc, breakup_sharpness, D_eq):
    """Self-collection and breakup of rain drops.

    Parameters
    ----------
    N_r : array
        Rain drop number concentration [1/kg].
    q_r : array
        Rain mixing ratio [kg/kg].
    rho : array
        Air density [kg/m3].
    k_sc : float
        Self-collection rate constant.
    breakup_sharpness : float
        Sigmoid sharpness for breakup onset.
    D_eq : float
        Equilibrium drop diameter [m].

    Returns
    -------
    dN_r_sc : array
        Self-collection tendency [1/kg/s].
    dN_r_br : array
        Breakup tendency [1/kg/s].
    """
    dN_r_sc = -k_sc * jnp.clip(N_r, 0.0) * jnp.clip(q_r, 0.0) * rho
    # Mean drop diameter D ~ (q_r * rho / N_r / (pi/6 * rho_water))^(1/3).
    # Cube-root has unbounded derivative at zero — guard with safe_pow.
    D_r_arg = (
        jnp.clip(q_r, 0.0) * rho
        / jnp.clip(N_r, 1.0)
        / (jnp.pi / 6.0 * constants.rho_water)
    )
    D_r = safe_pow(D_r_arg, 1.0 / 3.0)
    breakup_frac = jax.nn.sigmoid(breakup_sharpness * (D_r - D_eq))
    dN_r_br = -dN_r_sc * breakup_frac
    return dN_r_sc, dN_r_br


def rain_evaporation(q_v, q_r, q_sat, evap_coeff):
    """Compute rain evaporation in subsaturated air.

    Parameters
    ----------
    q_v : array
        Water vapor mixing ratio [kg/kg].
    q_r : array
        Rain mixing ratio [kg/kg].
    q_sat : array
        Saturation mixing ratio [kg/kg].
    evap_coeff : float
        Evaporation rate coefficient.

    Returns
    -------
    array : Evaporation rate [kg/kg/s].
    """
    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
    # Marshall-Palmer ventilation factor q_r^0.525 — fractional power has
    # an unbounded derivative at q_r=0; safe_pow handles the AD guard.
    return evap_coeff * subsaturation * safe_pow(q_r, 0.525)

exec
/bin/zsh -lc 'rg -n "thompson|graupel|dN_i|nucleation|total water|mass conservation|conserv" tests src/legoesm/atmosphere/physics/microphysics' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/atmosphere/physics/microphysics/output.py:24:    q_g: jax.Array    # graupel [kg/kg]
src/legoesm/atmosphere/physics/microphysics/output.py:38:    dq_g_dt: jax.Array        # graupel tendency
src/legoesm/atmosphere/physics/microphysics/output.py:41:    dN_i_dt: jax.Array        # ice number tendency
src/legoesm/atmosphere/physics/microphysics/output.py:76:        dN_c_dt=z2, dN_r_dt=z2, dN_i_dt=z2,
src/legoesm/atmosphere/physics/microphysics/sundqvist.py:195:        dN_i_dt=z,
tests/core/test_vertical_remap.py:78:    def test_conservation(self):
tests/core/test_vertical_remap.py:79:        """∫ q dp should be conserved after remapping."""
src/legoesm/atmosphere/physics/microphysics/kessler.py:135:        dN_i_dt=z,
src/legoesm/atmosphere/physics/microphysics/thompson.py:3:Extends Morrison with graupel formation from intense riming and
src/legoesm/atmosphere/physics/microphysics/thompson.py:45:def thompson_microphysics(
src/legoesm/atmosphere/physics/microphysics/thompson.py:110:    # Ice nucleation
src/legoesm/atmosphere/physics/microphysics/thompson.py:114:    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
src/legoesm/atmosphere/physics/microphysics/thompson.py:156:    # Split the rime → graupel conversion by donor: the fraction of
src/legoesm/atmosphere/physics/microphysics/thompson.py:157:    # rime_to_graupel that comes from q_i scales with riming_i, and
src/legoesm/atmosphere/physics/microphysics/thompson.py:160:    # the previous form set ``rime_to_graupel ∝ total_riming``, then
src/legoesm/atmosphere/physics/microphysics/thompson.py:163:    # ``q_g`` — a non-conservative split that depended on the
src/legoesm/atmosphere/physics/microphysics/thompson.py:165:    # "Thompson graupel conversion can draw from the wrong donor".)
src/legoesm/atmosphere/physics/microphysics/thompson.py:166:    graupel_frac = jax.nn.sigmoid(
src/legoesm/atmosphere/physics/microphysics/thompson.py:167:        config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
src/legoesm/atmosphere/physics/microphysics/thompson.py:169:    rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
src/legoesm/atmosphere/physics/microphysics/thompson.py:170:    rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
src/legoesm/atmosphere/physics/microphysics/thompson.py:171:    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
src/legoesm/atmosphere/physics/microphysics/thompson.py:172:    melt_graupel = jnp.minimum(
src/legoesm/atmosphere/physics/microphysics/thompson.py:190:    # Each rime-to-graupel donor scales with its parent riming term —
src/legoesm/atmosphere/physics/microphysics/thompson.py:192:    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``
src/legoesm/atmosphere/physics/microphysics/thompson.py:194:    # mass conservation.
src/legoesm/atmosphere/physics/microphysics/thompson.py:195:    rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:196:    rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:197:    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
src/legoesm/atmosphere/physics/microphysics/thompson.py:232:        - L_f * (melt_ice + melt_snow + melt_graupel) / c_pd
src/legoesm/atmosphere/physics/microphysics/thompson.py:236:    # Conservation: each rime-to-graupel donor leaves its parent
src/legoesm/atmosphere/physics/microphysics/thompson.py:238:    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``.
src/legoesm/atmosphere/physics/microphysics/thompson.py:239:    # An earlier form used 1.0 / 0.5 / 1.5 splits on ``rime_to_graupel``
src/legoesm/atmosphere/physics/microphysics/thompson.py:246:    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel + sed_r
src/legoesm/atmosphere/physics/microphysics/thompson.py:249:        - rime_to_graupel_from_i + sed_i
src/legoesm/atmosphere/physics/microphysics/thompson.py:251:    dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel_from_s + sed_s
src/legoesm/atmosphere/physics/microphysics/thompson.py:252:    dq_g_dt = rime_to_graupel - melt_graupel + sed_g
src/legoesm/atmosphere/physics/microphysics/thompson.py:256:    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
src/legoesm/atmosphere/physics/microphysics/thompson.py:275:        dN_i_dt=dN_i_dt,
src/legoesm/atmosphere/physics/microphysics/ml_emulator.py:121:        dN_i_dt=z,
src/legoesm/atmosphere/physics/microphysics/__init__.py:8:5. **Thompson**: Hybrid moment with graupel (Thompson et al. 2008)
src/legoesm/atmosphere/physics/microphysics/__init__.py:45:from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:124:        dN_i_dt=z,
src/legoesm/atmosphere/physics/microphysics/config.py:8:5. Thompson — hybrid moment with graupel
src/legoesm/atmosphere/physics/microphysics/config.py:73:    # Ice nucleation (Cooper 1986)
src/legoesm/atmosphere/physics/microphysics/config.py:130:    rime_to_graupel_threshold: float = 1e-4  # Riming threshold for graupel [kg/kg/s]
src/legoesm/atmosphere/physics/microphysics/config.py:131:    rime_to_graupel_rate: float = 0.5        # Fraction converted to graupel
src/legoesm/atmosphere/physics/microphysics/config.py:132:    graupel_sharpness: float = 1e4           # Sigmoid sharpness
src/legoesm/atmosphere/physics/microphysics/config.py:164:        "morrison", "thompson", "ml_emulator", or "none".
src/legoesm/atmosphere/physics/microphysics/config.py:169:    thompson : ThompsonConfig
src/legoesm/atmosphere/physics/microphysics/config.py:177:    thompson: ThompsonConfig = ThompsonConfig()
src/legoesm/atmosphere/physics/microphysics/integration.py:41:from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
src/legoesm/atmosphere/physics/microphysics/integration.py:70:    elif config.scheme == "thompson":
src/legoesm/atmosphere/physics/microphysics/integration.py:71:        return "thompson", thompson_microphysics, config.thompson
src/legoesm/atmosphere/physics/microphysics/integration.py:412:            micro_out.dN_c_dt, micro_out.dN_r_dt, micro_out.dN_i_dt,
src/legoesm/atmosphere/physics/microphysics/integration.py:473:        "N_i": "dN_i_dt",
tests/parallel/test_scaling_operators.py:349:    def test_pv_flux_energy_conserving_3d(self, mesh_and_state):
tests/parallel/test_scaling_operators.py:353:            pv_flux_energy_conserving,
tests/parallel/test_scaling_operators.py:354:            pv_flux_energy_conserving_3d,
tests/parallel/test_scaling_operators.py:363:        result = pv_flux_energy_conserving_3d(u_3d, h_3d, q_v_3d, mesh)
tests/parallel/test_scaling_operators.py:371:                pv_flux_energy_conserving(
src/legoesm/atmosphere/physics/microphysics/morrison.py:3:Extends Seifert-Beheng warm-rain with ice-phase processes: nucleation
src/legoesm/atmosphere/physics/microphysics/morrison.py:89:    # 1. Ice nucleation (Cooper 1986, smoothed)
src/legoesm/atmosphere/physics/microphysics/morrison.py:93:    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
src/legoesm/atmosphere/physics/microphysics/morrison.py:137:    # conserved because each process's matching source term in dq_r/dq_i/dq_s
src/legoesm/atmosphere/physics/microphysics/morrison.py:186:        # column conservation.  Magnitude estimate: ~2 K/day at default
src/legoesm/atmosphere/physics/microphysics/morrison.py:201:    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
src/legoesm/atmosphere/physics/microphysics/morrison.py:222:        dN_i_dt=dN_i_dt,
tests/sea_ice/validation/test_sea_ice_validation.py:4:  - Transport: conservation, non-negativity, temperature bounds
tests/sea_ice/validation/test_sea_ice_validation.py:5:  - ITD remap: category-bound preservation, volume conservation,
tests/sea_ice/validation/test_sea_ice_validation.py:162:    def test_volume_approximately_conserved(self):
tests/sea_ice/validation/test_sea_ice_validation.py:163:        """Total volume should be approximately conserved."""
tests/sea_ice/validation/test_sea_ice_validation.py:173:        # Approximate conservation (clamping can break exact conservation)
tests/conftest.py:61:    """A constant state for testing conservation."""
tests/ocean/distributed/test_ocean_mpi_conservation.py:1:"""Long-run MPI ocean conservation regression.
tests/ocean/distributed/test_ocean_mpi_conservation.py:5:    mpirun -np 6 python -m pytest tests/ocean/distributed/test_ocean_mpi_conservation.py -v
tests/ocean/distributed/test_ocean_mpi_conservation.py:50:from legoesm.ocean.conservation import ocean_conservation_fixer
tests/ocean/distributed/test_ocean_mpi_conservation.py:102:    """Nightly long-run MPI conservation regression."""
tests/ocean/distributed/test_ocean_mpi_conservation.py:104:    def test_longrun_mpi_ocean_conservation(self, topology):
tests/ocean/distributed/test_ocean_mpi_conservation.py:136:            use_conservation_fixer=True,
tests/ocean/distributed/test_ocean_mpi_conservation.py:179:            state = ocean_conservation_fixer(
tests/ocean/validation/test_differentiability_ocean.py:39:        use_conservation_fixer=False,
tests/ocean/validation/test_differentiability_ocean.py:72:        use_conservation_fixer=False,
tests/ocean/unit/test_ocean_fc.py:111:        use_conservation_fixer=False,
tests/ocean/unit/test_ocean_fc.py:129:        use_conservation_fixer=False,
tests/ocean/unit/test_ocean_fc.py:144:    config = OceanConfig(use_conservation_fixer=False)
tests/stress/test_phase1_sea_ice.py:149:    # 1A.3  Energy conservation (approximate check)
tests/stress/test_phase1_sea_ice.py:151:    def test_energy_conservation(self):
tests/stress/test_phase1_sea_ice.py:252:    # 1A.5  ITD linear remap conserves volume
tests/stress/test_phase1_sea_ice.py:254:    def test_itd_volume_conservation(self):
tests/stress/test_phase1_sea_ice.py:255:        """Linear remapping should conserve total ice volume across categories."""
tests/stress/test_phase1_sea_ice.py:285:            f"Volume not conserved: relative error = {rel_err.item():.2e}"
tests/stress/test_phase1_sea_ice.py:289:    # 1A.6  Transport conserves volume (approximate)
tests/stress/test_phase1_sea_ice.py:291:    def test_transport_volume_conservation(self):
tests/stress/test_phase1_sea_ice.py:293:        conserve area-weighted volume."""
tests/williamson_diagnostic.py:8:Test Case 5: Isolated Mountain (stability & conservation)
tests/williamson_diagnostic.py:11:Reports error norms, mass/energy conservation, cube-edge artifacts, and
tests/williamson_diagnostic.py:124:    All arrays promoted to float64 for conservation accuracy.
tests/williamson_diagnostic.py:170:        use_conservation_fixer=True,
tests/williamson_diagnostic.py:234:    # Mass conservation
tests/williamson_diagnostic.py:237:    print(f"  Mass conservation: relative error = {mass_rel_err:.2e}")
tests/williamson_diagnostic.py:239:    # Energy conservation
tests/williamson_diagnostic.py:280:    print(f"  [{'PASS' if c3 else 'FAIL'}] Mass conservation < 1e-8: {mass_rel_err:.2e}")
tests/williamson_diagnostic.py:300:    All arrays promoted to float64 for conservation accuracy.
tests/williamson_diagnostic.py:350:        use_conservation_fixer=True,
tests/williamson_diagnostic.py:387:    print(f"  Mass conservation: relative error = {mass_rel_err:.2e}")
tests/williamson_diagnostic.py:423:    print(f"  [{'PASS' if c2 else 'FAIL'}] Mass conservation < 1e-6: {mass_rel_err:.2e}")
tests/williamson_diagnostic.py:448:    All arrays promoted to float64 for conservation accuracy.
tests/williamson_diagnostic.py:520:        use_conservation_fixer=True,
tests/williamson_diagnostic.py:561:    print(f"  Mass conservation: relative error = {mass_rel_err:.2e}")
tests/williamson_diagnostic.py:604:    print(f"  [{'PASS' if c2 else 'FAIL'}] Mass conservation < 1e-6: {mass_rel_err:.2e}")
tests/validation/test_ec_eigenvalues2.py:58:    """Energy-conserving flux form: (sigma*sd_upper - sigma*sd_lower)/dsigma"""
tests/validation/test_ec_eigenvalues2.py:142:# For the D-lnps coupling to conserve energy:
tests/validation/test_ec_eigenvalues2.py:145:# These should be "paired" in energy conservation.
tests/validation/test_ec_eigenvalues2.py:150:# Actually, let me compute the FULL conserved energy quadratic form and see
tests/validation/test_ec_eigenvalues2.py:165:print("Case 6: Testing quadratic form conservation")
tests/ocean/unit/test_barotropic_noise_invariant.py:245:def test_implicit_solver_conserves_mass():
tests/ocean/unit/test_barotropic_noise_invariant.py:246:    """Per-step mass conservation: ``Σ η·area`` drift over 100 steps
tests/unit/test_bechtold.py:142:def test_bechtold_downdraft_evap_conserves_water_locally():
tests/unit/test_bechtold.py:146:    test_tiedtke_downdraft_evap_conserves_water_locally`` for the
tests/unit/test_bechtold.py:150:      2. Column water conservation: column-integrated
tests/unit/test_bechtold.py:191:    # (2) Column water conservation
tests/unit/test_bechtold.py:578:# MSE conservation regression guard (currently expected to fail)
tests/unit/test_bechtold.py:586:        "Currently ~92% non-conservation residual; flagged as xfail so "
tests/unit/test_bechtold.py:591:def test_bechtold_mse_conservation_within_tolerance():
tests/stress/test_phase2_coupled.py:5:boundedness, conservation, and diagnostic population.
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:7:Covers: shape/finiteness, conservation, variance reduction, land masks,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:181:# 3. Tracer conservation
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:186:    def test_tracer_integral_conserved(self):
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:272:        """With land, conservation should still hold."""
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:533:    def test_triad_tracer_integral_conserved(self):
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:549:            assert relative < 1e-10, f"Triad conservation: rel={relative:.2e}"
tests/validation/bench_spectral_pe.py:11:  - conservation.png      — mass (dry), energy time series
tests/validation/bench_spectral_pe.py:330:    plt.savefig(os.path.join(OUT_DIR, "conservation.png"), dpi=150, bbox_inches="tight")
tests/validation/bench_spectral_pe.py:332:    print(f"  Saved conservation.png")
tests/unit/test_land_ice_slab_land.py:8:  - bucket hydrology water conservation
tests/unit/test_land_ice_slab_land.py:255:# 1e  Bucket hydrology water conservation
tests/stress/test_phase1_land_carbon.py:45:    # 1B.2 -- Carbon pool conservation
tests/stress/test_phase1_land_carbon.py:47:    def test_carbon_pool_conservation(self):
tests/stress/test_phase1_land_carbon.py:48:        """Total carbon (pools + cumulative NEE) should be approximately conserved."""
tests/stress/test_phase1_land_carbon.py:88:            f"Pool conservation violated: relative error {float(rel_err):.6e}"
tests/validation/test_ensemble_correctness.py:67:    "dN_i_dt",
tests/validation/test_ensemble_correctness.py:245:    from legoesm.core.conservation import fix_ps_mass_target
tests/validation/test_ec_eigenvalues.py:1:"""Test eigenvalues of linearized PE with energy-conserving vs current discretization.
tests/validation/test_ec_eigenvalues.py:96:# ---- Build M matrix: THEORETICAL energy-conserving (from antisymmetry with N) ----
tests/stress/test_phase1_ocean_slab.py:90:    # 1D.2 -- Two-layer conservation and relaxation
tests/stress/test_phase1_ocean_slab.py:92:    def test_two_layer_conservation(self):
tests/stress/test_phase1_ocean_slab.py:153:        # should be approximately conserved.
tests/validation/validation_differentiability_all.py:153:            use_conservation_fixer=False,
tests/validation/validation_differentiability_all.py:168:            hyperdiff_coeff=0.0, use_conservation_fixer=False,
tests/validation/README_DYCORE_PROGRESSION.md:27:- `conservation_timeseries.csv`
tests/validation/README_DYCORE_PROGRESSION.md:28:- `conservation_timeseries.png`
tests/test_mpas_conservation.py:3:Tests mass/energy/enstrophy conservation for shallow water
tests/test_mpas_conservation.py:4:and volume/heat/salt conservation for the ocean model.
tests/test_mpas_conservation.py:15:from legoesm.core.conservation import global_integral_voronoi
tests/test_mpas_conservation.py:46:    """Compute shallow water conservation diagnostics."""
tests/test_mpas_conservation.py:77:    """Compute ocean conservation diagnostics."""
tests/test_mpas_conservation.py:132:def check_sw_conservation():
tests/test_mpas_conservation.py:133:    """Check conservation for shallow water on icosahedral grid."""
tests/test_mpas_conservation.py:198:def check_ocean_conservation():
tests/test_mpas_conservation.py:199:    """Check conservation for ocean PE on icosahedral grid."""
tests/test_mpas_conservation.py:226:            use_conservation_fixer=(fix_vol or fix_heat or fix_salt),
tests/test_mpas_conservation.py:309:    check_sw_conservation()
tests/test_mpas_conservation.py:310:    check_ocean_conservation()
tests/ocean/unit/test_sfno_ocean.py:23:from legoesm.ml.conservation import (
tests/ocean/unit/test_sfno_ocean.py:253:    def test_with_conservation(self, grid_t10, z_coord, ocean_state, sfno_config):
tests/land/unit/test_multilayer_land.py:6:- 8C: Richards equation solver (mass conservation, boundary conditions)
tests/land/unit/test_multilayer_land.py:7:- 8D: Soil thermal diffusion (energy conservation, convergence)
tests/land/unit/test_multilayer_land.py:567:    def test_energy_conservation(self):
tests/distributed/test_mpi_driver.py:98:            use_conservation_fixer=False,
tests/land/unit/test_carbon_cycle.py:340:    def test_total_carbon_conservation_tendency(self):
tests/land/unit/test_carbon_cycle.py:367:                            err_msg="Carbon not approximately conserved")
tests/land/unit/test_land_audit_fixes.py:224:    def test_energy_conservation_x64(self):
tests/land/unit/test_land_audit_fixes.py:247:                            err_msg="Energy conservation violated under x64")
tests/validation/run_dycore_progression_suite.py:14:- conservation_timeseries.csv / conservation_timeseries.png
tests/validation/run_dycore_progression_suite.py:96:def _derive_conservation_from_mean(case_dir: Path) -> None:
tests/validation/run_dycore_progression_suite.py:141:    out_csv = case_dir / "conservation_timeseries.csv"
tests/validation/run_dycore_progression_suite.py:156:    axes[0].set_title(f"{mass_key or 'mass'} proxy conservation")
tests/validation/run_dycore_progression_suite.py:163:    axes[1].set_title(f"{energy_key or 'energy'} proxy conservation")
tests/validation/run_dycore_progression_suite.py:166:    fig.savefig(case_dir / "conservation_timeseries.png", dpi=150, bbox_inches="tight")
tests/validation/run_dycore_progression_suite.py:176:        "conservation_timeseries.csv": False,
tests/validation/run_dycore_progression_suite.py:177:        "conservation_timeseries.png": False,
tests/validation/run_dycore_progression_suite.py:191:    copied["conservation_timeseries.csv"] = _copy_if_exists(
tests/validation/run_dycore_progression_suite.py:192:        case_dir / "conservation_timeseries.csv",
tests/validation/run_dycore_progression_suite.py:193:        case_dir / "conservation_timeseries.csv",
tests/validation/run_dycore_progression_suite.py:195:    copied["conservation_timeseries.png"] = _copy_if_exists(
tests/validation/run_dycore_progression_suite.py:196:        case_dir / "conservation_timeseries.png",
tests/validation/run_dycore_progression_suite.py:197:        case_dir / "conservation_timeseries.png",
tests/validation/run_dycore_progression_suite.py:199:    if not copied["conservation_timeseries.csv"] or not copied["conservation_timeseries.png"]:
tests/validation/run_dycore_progression_suite.py:200:        _derive_conservation_from_mean(case_dir)
tests/validation/run_dycore_progression_suite.py:201:        copied["conservation_timeseries.csv"] = (case_dir / "conservation_timeseries.csv").exists()
tests/validation/run_dycore_progression_suite.py:202:        copied["conservation_timeseries.png"] = (case_dir / "conservation_timeseries.png").exists()
tests/validation/run_dycore_progression_suite.py:479:            "- `conservation_timeseries.csv` / `conservation_timeseries.png`",
tests/land/unit/test_land_water_budget.py:126:# 2. Slab land: bucket overflow conservation / runoff
tests/land/unit/test_land_water_budget.py:150:    def test_water_conservation_with_runoff(self):
tests/land/unit/test_land_water_budget.py:377:    def test_mass_conservation(self):
tests/ocean/unit/test_ocean_compatibility.py:6:- MPI-aware conservation fixers and operator dispatch
tests/ocean/unit/test_ocean_compatibility.py:30:from legoesm.ocean.conservation import (
tests/ocean/unit/test_ocean_compatibility.py:219:    def test_conservation_uses_is_distributed(self):
tests/ocean/unit/test_ocean_compatibility.py:222:        from legoesm.ocean.conservation import _ocean_area_sum, _ocean_global_sum
tests/ocean/unit/test_ocean_compatibility.py:263:    def test_conservation_fixer_produces_correct_output(self, grid, z_coord, state):
tests/ocean/unit/test_ocean_compatibility.py:273:    def test_heat_conservation_fixer(self, grid, z_coord, state):
tests/ocean/unit/test_ocean_compatibility.py:283:    def test_spectral_conservation_uses_mpi_aware_reductions(self):
tests/ocean/unit/test_ocean_compatibility.py:284:        """Spectral conservation fixer should route totals through MPI-aware sums."""
tests/ocean/unit/test_ocean_compatibility.py:288:            _spectral_conservation_fixer,
tests/ocean/unit/test_ocean_compatibility.py:295:        fixer_source = inspect.getsource(_spectral_conservation_fixer)
tests/ocean/unit/test_ocean_compatibility.py:362:            use_conservation_fixer=False,
tests/ocean/unit/test_ocean_compatibility.py:390:            use_conservation_fixer=False,
tests/validation/test_differentiability_regression.py:186:            use_conservation_fixer=False,
tests/validation/test_differentiability_regression.py:288:            use_conservation_fixer=False,
tests/validation/test_differentiability_regression.py:313:            use_conservation_fixer=False,
tests/unit/test_physics_grid_adapters.py:237:        expected = {"kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"}
tests/validation/test_restart_reproducibility.py:59:            conservation_fixer=True,
tests/validation/bench_spectral_sw.py:11:  - conservation_tc2.png   — mass, energy, enstrophy time series (TC2)
tests/validation/bench_spectral_sw.py:12:  - conservation_tc5.png   — mass, energy, enstrophy time series (TC5)
tests/validation/bench_spectral_sw.py:124:def plot_conservation(result, name, filename):
tests/validation/bench_spectral_sw.py:125:    """Plot relative conservation errors."""
tests/validation/bench_spectral_sw.py:289:    plot_conservation(res2, "TC2 — Geostrophic Balance", "conservation_tc2.png")
tests/validation/bench_spectral_sw.py:290:    plot_conservation(res5, "TC5 — Mountain Flow", "conservation_tc5.png")
tests/unit/test_spectral_dycores_comprehensive.py:435:    def test_4a_mass_conservation_sw(self, grid_t21):
tests/unit/test_spectral_dycores_comprehensive.py:436:        """SW mass conservation in TC5 for 500 steps."""
tests/unit/test_spectral_dycores_comprehensive.py:458:        assert dM < 1e-12, f"SW mass conservation: |dM/M| = {dM:.3e}, expected < 1e-12"
tests/unit/test_spectral_dycores_comprehensive.py:461:    def test_4b_energy_conservation_sw(self, grid_t21):
tests/unit/test_spectral_dycores_comprehensive.py:462:        """SW total energy conservation in TC2 (no diffusion) for 500 steps."""
tests/unit/test_spectral_dycores_comprehensive.py:484:        assert dE < 1e-6, f"SW energy conservation: |dE/E| = {dE:.3e}, expected < 1e-6"
tests/unit/test_spectral_dycores_comprehensive.py:487:    def test_4c_enstrophy_conservation_sw(self, grid_t21):
tests/unit/test_spectral_dycores_comprehensive.py:488:        """SW potential enstrophy conservation in TC2 (no diffusion) for 500 steps."""
tests/unit/test_spectral_dycores_comprehensive.py:510:        assert dZ < 1e-6, f"SW enstrophy conservation: |dZ/Z| = {dZ:.3e}, expected < 1e-6"
tests/unit/test_spectral_dycores_comprehensive.py:512:    def test_4d_mass_conservation_pe(self, grid_t10, sigma_5lev):
tests/unit/test_spectral_dycores_comprehensive.py:513:        """PE mass (surface pressure integral) conservation in rest state, 100 steps."""
tests/unit/test_spectral_dycores_comprehensive.py:539:        assert dM < 1e-10, f"PE mass conservation: |dM/M| = {dM:.3e}, expected < 1e-10"
tests/unit/test_spectral_dycores_comprehensive.py:542:        """Angular momentum conservation for inviscid PE (rest state stays at rest)."""
tests/unit/test_spectral_dycores_comprehensive.py:1012:        """TC2 at T21 for 5 days: L2(h) < 1e-9, mass conserved to machine eps.
tests/unit/test_spectral_dycores_comprehensive.py:1087:    def test_tc5_15day_mass_conservation(self):
tests/unit/test_spectral_dycores_comprehensive.py:1088:        """TC5 (mountain) at T21 for 15 days: mass conserved to machine eps.
tests/unit/test_spectral_dycores_comprehensive.py:1090:        Mass must be conserved exactly in a spectral shallow water model
tests/unit/test_spectral_dycores_comprehensive.py:1122:        # Mass MUST be conserved to machine precision
tests/unit/test_spectral_dycores_comprehensive.py:1177:    def test_pe_mass_conservation_200steps(self):
tests/unit/test_spectral_dycores_comprehensive.py:1178:        """PE surface pressure integral should be conserved over 200 steps.
tests/unit/test_spectral_dycores_comprehensive.py:1181:        mass (integral of p_s) must be conserved because:
tests/unit/test_spectral_dycores_comprehensive.py:1210:            f"PE mass conservation 200 steps: |dM/M| = {dM:.3e}, expected < 1e-12"
tests/distributed/test_mpi_differentiability.py:166:    """Gradient through conservation fixers with MPI reductions."""
tests/distributed/test_mpi_differentiability.py:173:        from legoesm.core.conservation import fix_ps_mass_target
tests/ocean/unit/test_weno_momentum.py:6:2. WENO vertical momentum advection: shapes, constant field, conservation
tests/ocean/unit/test_weno_momentum.py:263:        """Uniform velocity ⟹ advection can't change it (conservation)."""
tests/unit/test_conservation.py:1:"""Unit tests for conservation fixers."""
tests/unit/test_conservation.py:9:from legoesm.core.conservation import (
tests/unit/test_conservation.py:12:    apply_conservation_fixer,
tests/unit/test_conservation.py:13:    compute_conservation_diagnostics,
tests/unit/test_conservation.py:20:    """Tests for mass and energy conservation fixers."""
tests/unit/test_conservation.py:136:    def test_conservation_diagnostics(self, grid, state):
tests/unit/test_conservation.py:138:        diag = compute_conservation_diagnostics(state, grid)
tests/atmosphere/shallow_water/integration/test_shallow_water.py:53:            use_conservation_fixer=True,
tests/atmosphere/shallow_water/integration/test_shallow_water.py:89:    def test_mass_conservation(self, model, grid, cdgrid):
tests/atmosphere/shallow_water/integration/test_shallow_water.py:90:        """Mass should be conserved with fixer."""
tests/atmosphere/shallow_water/integration/test_shallow_water.py:148:            use_conservation_fixer=False
tests/atmosphere/shallow_water/integration/test_shallow_water.py:165:            use_conservation_fixer=False
tests/atmosphere/shallow_water/integration/test_shallow_water.py:181:        config = CDGridShallowWaterConfig(use_conservation_fixer=False)
tests/unit/test_vector_calculus_identities.py:383:    This is critical for energy conservation and 4D-Var correctness.
tests/validation/test_conservation_baseline.py:6:scripts/run_amip.py --conservation-audit.
tests/validation/test_conservation_baseline.py:9:1. Dry mass conservation: relative drift < 1e-10
tests/validation/test_conservation_baseline.py:33:        out = tmp_path_factory.mktemp("conservation")
tests/validation/test_conservation_baseline.py:61:    def test_dry_mass_conserved(self, run_result):
tests/unit/test_kain_fritsch.py:290:# MSE conservation regression guard (currently expected to fail)
tests/unit/test_kain_fritsch.py:295:        "Standard mass-flux kernel does not conserve column MSE on a "
tests/unit/test_kain_fritsch.py:296:        "closed (no-surface-flux) probe.  Currently ~95% non-conservation "
tests/unit/test_kain_fritsch.py:302:def test_kf_mse_conservation_within_tolerance():
tests/unit/test_physical_balances.py:67:            use_conservation_fixer=True,
tests/unit/test_physical_balances.py:142:            use_conservation_fixer=True,
tests/unit/test_moisture_budget.py:5:- Budget residual is ~0 for a conserving system
tests/unit/test_moisture_budget.py:107:    def test_conserving_system_small_residual(self):
tests/validation/test_scaling_readiness.py:40:    "dN_i_dt",
tests/ocean/unit/test_advection_weno.py:188:    def test_conservation_closed_column(self):
tests/ocean/unit/test_advection_weno.py:329:    def test_conservation_closed_column(self):
tests/ocean/unit/test_advection_weno.py:409:# Horizontal conservation
tests/ocean/unit/test_advection_weno.py:416:    def test_conservation(self, scheme):
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:56:            use_conservation_fixer=True,
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:62:        # Anchor conservation fixer to initial mass for drift-free long runs
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:98:    def test_mass_conservation(self, model_and_state, grid_sw):
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:99:        """Mass should be conserved to near machine precision over 50 steps."""
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:108:        # The conservation fixer anchors to the CDGrid state mass.
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:109:        # Machine-precision conservation (< 1e-10) requires a flux-form
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:121:            use_conservation_fixer=True,
tests/distributed/test_halo_mpi.py:30:from legoesm.ocean.conservation import ocean_conservation_fixer
tests/distributed/test_halo_mpi.py:318:    """MPI conservation-fixer behavior on partitioned ocean state."""
tests/distributed/test_halo_mpi.py:320:    def test_ocean_conservation_fixer_matches_local(self, topology):
tests/distributed/test_halo_mpi.py:321:        """MPI conservation fixers should match local-global reference."""
tests/distributed/test_halo_mpi.py:352:        ref_state = ocean_conservation_fixer(
tests/distributed/test_halo_mpi.py:360:        # For MPI conservation: use _zero_non_owned on state arrays
tests/distributed/test_halo_mpi.py:361:        # but keep grid global (conservation fixer needs global areas
tests/distributed/test_halo_mpi.py:376:        fixed_part = ocean_conservation_fixer(
tests/ocean/unit/test_gm_redi_eady_physics.py:188:    def test_gm_only_conserves_tracer(self):
tests/ocean/unit/test_gm_redi_eady_physics.py:214:            f"GM should conserve tracer, relative error = {relative:.2e}"
tests/ocean/unit/test_gm_redi_eady_physics.py:398:    def test_triad_gm_only_conserves_tracer(self):
tests/ocean/unit/test_gm_redi_eady_physics.py:412:        assert rel < 1e-10, f"Triad GM must conserve tracer; rel={rel:.2e}"
tests/unit/test_voronoi_trisk_weights.py:160:        used for energy conservation of the PV flux (Ringler 2010 Eq. 49).
tests/unit/test_voronoi_trisk_weights.py:173:            f"rel={rel:.3e} — breaks energy conservation of PV flux"
tests/unit/test_voronoi_trisk_weights.py:259:        """Energy-conserving PV flux does near-zero discrete work.
tests/unit/test_voronoi_trisk_weights.py:267:            pv_flux_energy_conserving,
tests/unit/test_voronoi_trisk_weights.py:278:        Fq = pv_flux_energy_conserving(u, h, q, mesh)
tests/unit/test_voronoi_trisk_weights.py:293:            "stencil operator energy conservation broken"
tests/unit/test_component_factory.py:310:# 7. Lat-lon C-grid conservation_fixer=False
tests/unit/test_component_factory.py:314:    """conservation_fixer=False must disable the mass fixer."""
tests/unit/test_component_factory.py:316:    def test_conservation_fixer_false_disables_fix_mass(self):
tests/unit/test_component_factory.py:317:        """conservation_fixer=False should produce fix_mass=False."""
tests/unit/test_component_factory.py:326:                conservation_fixer=False,
tests/unit/test_component_factory.py:332:            "conservation_fixer=False must override fix_mass to False"
tests/unit/test_component_factory.py:335:    def test_conservation_fixer_true_preserves_fix_mass(self):
tests/unit/test_component_factory.py:336:        """conservation_fixer=True (default) should keep fix_mass=True."""
tests/unit/test_component_factory.py:351:    def test_conservation_fixer_false_propagates_to_driver_config(self):
tests/unit/test_component_factory.py:352:        """conservation_fixer=False must also set fix_mass=False in the
tests/unit/test_component_factory.py:364:                conservation_fixer=False,
tests/unit/test_component_factory.py:376:            "conservation_fixer=False must propagate to "
tests/unit/test_cdgrid.py:6:3. Shallow water solver: stability and conservation
tests/unit/test_cdgrid.py:282:    def test_mass_conservation(self):
tests/unit/test_cdgrid.py:283:        """Mass should be conserved after time stepping."""
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:29:    """Set fp64 precision policy for this module (conservation tests need it).
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:157:    """Mass conservation with conservation fixer."""
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:159:    def test_mass_conservation_precision(self, mesh):
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:160:        """Mass should be conserved to near machine precision with fixer."""
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:186:    """Energy conservation with energy-conserving PV scheme."""
tests/distributed/test_voronoi_mpi.py:12:- Mass conservation with global MPI fixer
tests/distributed/test_voronoi_mpi.py:215:    """Verify global mass is conserved under MPI."""
tests/distributed/test_voronoi_mpi.py:217:    def test_mass_conserved(self, mesh, sigma, config):
tests/distributed/test_voronoi_mpi.py:218:        """Global dry mass should be conserved after MPI stepping."""
tests/distributed/test_voronoi_mpi.py:257:                err_msg="Mass not conserved under MPI")
tests/unit/test_precision.py:693:        from legoesm.ocean.conservation import _ocean_area_sum
tests/unit/test_precision.py:707:        from legoesm.ocean.conservation import _ocean_area_sum
tests/unit/test_precision.py:718:        from legoesm.ocean.conservation import _ocean_volume_sum
tests/ocean/unit/test_barotropic_implicit_mpas.py:182:def test_implicit_solver_conserves_mass(state, mesh, z_coord):
tests/unit/test_grid_dycore_fixes.py:69:        # float32 precision (~1e-7 relative).  Use a conservative
tests/unit/test_grid_dycore_fixes.py:136:        discrete energy-conservation identity for the stencil operator.
tests/unit/test_grid_dycore_fixes.py:155:        """Energy-conserving PV flux must do near-zero discrete work."""
tests/unit/test_grid_dycore_fixes.py:157:            pv_flux_energy_conserving,
tests/unit/test_grid_dycore_fixes.py:168:        Fq = pv_flux_energy_conserving(u, h, q, mesh)
tests/atmosphere/shallow_water/integration/test_boundary_fix.py:127:    def test_mass_conservation(self, grid_and_cdgrid):
tests/unit/test_operators.py:88:        from legoesm.core.conservation import zero_mean_tendency, _global_area_sum
tests/unit/test_operators.py:99:        from legoesm.core.conservation import zero_mean_tendency, _global_area_sum
tests/unit/test_operators.py:111:        from legoesm.core.conservation import zero_mean_tendency
tests/unit/test_operators.py:121:    """Tests for conservation fixer enhancements."""
tests/unit/test_operators.py:125:        from legoesm.core.conservation import fix_mass_hydrostatic_target
tests/unit/test_operators.py:157:    """Tests that conservation code is differentiable with jax.grad."""
tests/unit/test_operators.py:161:        from legoesm.core.conservation import zero_mean_tendency
tests/unit/test_operators.py:173:        from legoesm.core.conservation import zero_mean_tendency
tests/unit/test_operators.py:185:        from legoesm.core.conservation import fix_mass_hydrostatic_target
tests/unit/test_precision_modes.py:166:        config = CDGridShallowWaterConfig(use_conservation_fixer=False)
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:1:"""Convergence and conservation tests for FV3 shallow water on cubed sphere.
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:4:to verify bounded error and mass conservation precision.
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:52:        use_conservation_fixer=True,
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:102:    """Mass conservation precision for C-D grid shallow water."""
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:104:    def test_mass_conservation_precision(self):
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:105:        """Mass should be conserved to high precision over 100 steps."""
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:120:            use_conservation_fixer=True,
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:141:        # conservation fixer this is corrected, but rounding accumulates.
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:151:        signals non-conservative face fluxes.
tests/atmosphere/shallow_water/test_cases/williamson.py:3:These test cases validate the numerical accuracy and conservation properties
tests/unit/test_land_ice_multilayer.py:139:# 2d  Soil thermal energy conservation
tests/unit/test_land_ice_multilayer.py:144:    def test_energy_conserved(self):
tests/unit/test_land_ice_multilayer.py:280:# 2i  Richards equation -- water conservation
tests/unit/test_backend_precision.py:153:            hyperdiff_coeff=0.0, use_conservation_fixer=False,
tests/unit/test_backend_precision.py:209:    def test_conservation_float64(self, grid):
tests/unit/test_backend_precision.py:244:            hyperdiff_coeff=0.0, use_conservation_fixer=False,
tests/unit/test_zhang_mcfarlane.py:6:* energy-conservation budget over one step;
tests/unit/test_zhang_mcfarlane.py:7:* moisture-conservation budget (vapor + cloud-water source vs precip);
tests/unit/test_zhang_mcfarlane.py:254:    # Environment is heated above LFC and roughly conserved below;
tests/unit/test_zhang_mcfarlane.py:513:# MSE conservation regression guard (currently expected to fail)
tests/unit/test_zhang_mcfarlane.py:519:        "moist-adiabat T_u) does not conserve column MSE on a closed "
tests/unit/test_zhang_mcfarlane.py:520:        "(no-surface-flux) probe.  Currently ~98% non-conservation "
tests/unit/test_zhang_mcfarlane.py:526:def test_zm_mse_conservation_within_tolerance():
tests/ocean/unit/test_gm_redi_mpas.py:330:def test_centered_conserves_mass_globally(mesh, z_coord, cfg):
tests/unit/test_emanuel.py:9:* the unsaturated-downdraft toggle (column conservation under both
tests/unit/test_emanuel.py:254:# MSE conservation regression guard (currently expected to fail)
tests/unit/test_emanuel.py:259:        "Standard mass-flux kernel does not conserve column MSE on a "
tests/unit/test_emanuel.py:260:        "closed (no-surface-flux) probe.  Currently ~99% non-conservation "
tests/unit/test_emanuel.py:266:def test_emanuel_mse_conservation_within_tolerance():
tests/unit/test_cdgrid_fv3_regression.py:271:    def test_mass_approximately_conserved_one_step(self):
tests/unit/test_cdgrid_fv3_regression.py:272:        """Mass should be approximately conserved for 1 step."""
tests/unit/test_duogrid.py:482:    def test_duogrid_mass_conservation_one_step(self):
tests/unit/test_duogrid.py:483:        """One RK3 step with duogrid should conserve mass."""
tests/unit/test_duogrid.py:499:        assert rel_err < 1e-8, f"Mass conservation violated: rel_err={rel_err:.2e}"
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:485:    """Tests for mass conservation options in CE model."""
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:489:        from legoesm.core.conservation import compute_nh_dry_mass
tests/unit/test_land_ice_carbon.py:110:# 6e  Carbon pool conservation
tests/ocean/unit/test_advection_som.py:5:2. X-sweep: conservation, uniform tracer, periodic wrapping
tests/ocean/unit/test_advection_som.py:6:3. Y-sweep: conservation, solid-wall BCs
tests/ocean/unit/test_advection_som.py:7:4. Z-sweep: conservation, closed column
tests/ocean/unit/test_advection_som.py:8:5. Full 3D: conservation, differentiability
tests/ocean/unit/test_advection_som.py:113:    def test_conservation(self):
tests/ocean/unit/test_advection_som.py:114:        """Total sm_o must be conserved (periodic domain)."""
tests/ocean/unit/test_advection_som.py:126:        # Global sum conserved (periodic: everything stays in domain)
tests/ocean/unit/test_advection_som.py:169:    def test_conservation_channel(self):
tests/ocean/unit/test_advection_som.py:170:        """Total sm_o conserved with solid walls."""
tests/ocean/unit/test_advection_som.py:195:        # Total should be conserved (wall blocks southward leakage from cell 0)
tests/ocean/unit/test_advection_som.py:222:    def test_conservation_closed_column(self):
tests/ocean/unit/test_advection_som.py:223:        """Total column sm_o conserved with closed top/bottom."""
tests/ocean/unit/test_advection_som.py:323:    def test_conservation_with_flow(self):
tests/ocean/unit/test_advection_som.py:324:        """Global volume-weighted tracer integral is conserved."""
tests/unit/test_sea_ice_dynamics.py:407:    def test_volume_conservation(self):
tests/unit/test_sea_ice_dynamics.py:408:        """Total volume h*a should be conserved in aggregation."""
tests/unit/test_sea_ice_dynamics.py:459:    def test_volume_conserved(self):
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:164:# Mass conservation — uses explicit target_mass API
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:169:    def test_mass_conserved_100_steps(self, grid):
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:183:        assert rel_err < 1e-6, f"Mass conservation error: {rel_err}"
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:260:# Total energy conservation
tests/unit/test_conservation_laws.py:4:  4e) Energy conservation (shallow water, C-D grid cubed-sphere)
tests/unit/test_conservation_laws.py:6:  4k) MPAS shallow water energy conservation
tests/unit/test_conservation_laws.py:22:# 4e) Energy conservation (shallow water, C-D grid cubed-sphere)
tests/unit/test_conservation_laws.py:26:    """Total energy (KE + PE) should be approximately conserved in inviscid
tests/unit/test_conservation_laws.py:63:            use_conservation_fixer=True,
tests/unit/test_conservation_laws.py:85:    def test_energy_conserved(self):
tests/unit/test_conservation_laws.py:111:# 4i) Enstrophy budget (MPAS, energy-conserving vs enstrophy-conserving)
tests/unit/test_conservation_laws.py:115:    """Compare enstrophy behaviour under energy-conserving vs
tests/unit/test_conservation_laws.py:116:    enstrophy-conserving PV flux options in MPAS shallow water.
tests/unit/test_conservation_laws.py:118:    For enstrophy-conserving flux, potential enstrophy Z should be
tests/unit/test_conservation_laws.py:119:    much better conserved than for energy-conserving flux.
tests/unit/test_conservation_laws.py:174:    def test_enstrophy_conserving_better(self):
tests/unit/test_conservation_laws.py:175:        """Enstrophy-conserving PV flux should preserve Z better than
tests/unit/test_conservation_laws.py:176:        energy-conserving flux."""
tests/unit/test_conservation_laws.py:183:        print(f"  Enstrophy-conserving: Z0={Z0_en:.6e}, Zf={Zf_en:.6e}, "
tests/unit/test_conservation_laws.py:185:        print(f"  Energy-conserving: Z0={Z0_ec:.6e}, Zf={Zf_ec:.6e}, "
tests/unit/test_conservation_laws.py:189:        assert rel_en < 0.20, f"Enstrophy-conserving Z drift = {rel_en}"
tests/unit/test_conservation_laws.py:190:        assert rel_ec < 0.50, f"Energy-conserving Z drift = {rel_ec}"
tests/unit/test_conservation_laws.py:194:# 4k) MPAS shallow water energy conservation
tests/unit/test_conservation_laws.py:198:    """Total energy (KE + PE) should be approximately conserved in
tests/unit/test_conservation_laws.py:247:    def test_energy_conserved(self):
tests/ocean/unit/test_cross_grid_parity.py:243:    def test_cube_volume_conservation(self, cube_model, cube_state, cube_grid):
tests/ocean/unit/test_cross_grid_parity.py:244:        """Volume (eta integral) should be conserved on cubed-sphere."""
tests/unit/test_physics_microphysics.py:3:Tests total water conservation, temperature-moisture coupling (Clausius-
tests/unit/test_physics_microphysics.py:19:from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
tests/unit/test_physics_microphysics.py:89:    elif name == "thompson":
tests/unit/test_physics_microphysics.py:90:        return thompson_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:96:ALL_SCHEMES = ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"]
tests/unit/test_physics_microphysics.py:177:@pytest.mark.parametrize("scheme", ["morrison", "thompson"])
tests/unit/test_physics_microphysics.py:243:    [("morrison", morrison_microphysics), ("thompson", thompson_microphysics)],
tests/unit/test_physics_microphysics.py:252:    loss per step is bounded by q_c (mass conservation preserved by
tests/unit/test_physics_microphysics.py:294:def test_thompson_qi_does_not_go_negative_warm():
tests/unit/test_physics_microphysics.py:325:    out = thompson_microphysics(
tests/unit/test_physics_microphysics.py:341:def test_thompson_rime_to_graupel_donor_split():
tests/unit/test_physics_microphysics.py:342:    """Audit cycle 2 (Codex): the Thompson rime-to-graupel conversion
tests/unit/test_physics_microphysics.py:352:        rime_to_graupel = rate * (riming_i + riming_s) > 0
tests/unit/test_physics_microphysics.py:353:        dq_i_dt -= rime_to_graupel              # full subtraction!
tests/unit/test_physics_microphysics.py:354:        dq_s_dt -= 0.5 * rime_to_graupel
tests/unit/test_physics_microphysics.py:355:        dq_g_dt += 1.5 * rime_to_graupel
tests/unit/test_physics_microphysics.py:357:    conservation violations.  Post-fix the donor split scales by
tests/unit/test_physics_microphysics.py:359:    drained.  Total mass moved (``rime_to_graupel_from_i +
tests/unit/test_physics_microphysics.py:360:    rime_to_graupel_from_s``) goes 1:1 to graupel.
tests/unit/test_physics_microphysics.py:383:    out = thompson_microphysics(
tests/unit/test_physics_microphysics.py:391:        "q_i=0 with q_s>0 and active riming.  Pre-fix the rime_to_graupel "
tests/unit/test_physics_microphysics.py:394:        "'Thompson graupel conversion can draw from the wrong donor' "
tests/unit/test_physics_microphysics.py:397:    # Mass conservation: ∑ dq_i + dq_s + dq_g (pure ice phase) should
tests/unit/test_physics_microphysics.py:399:    # internal phase changes cancel in the sum.  We can verify rime->graupel
tests/unit/test_physics_microphysics.py:403:    # rime_to_graupel.  q_g production must be matched by q_s loss
tests/unit/test_physics_microphysics.py:407:        "without graupel sedimentation source — graupel mass conservation "
tests/unit/test_physics_microphysics.py:413:# Morrison/Thompson moist-enthalpy conservation (latent heat of fusion)
tests/unit/test_physics_microphysics.py:418:    [("morrison", morrison_microphysics), ("thompson", thompson_microphysics)],
tests/unit/test_physics_microphysics.py:423:    Moist enthalpy ``h = c_pd T + L_v q_v - L_f * q_ice`` is conserved
tests/unit/test_physics_microphysics.py:495:    Column total water tendency must balance surface precipitation:
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:255:- EPV conservation quality
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:426:    ! Cl + Cl₂ = const (conservation constraint: cly = cl + 2*cl2)
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:439:- Terminator tracer total `Cly = Cl + 2*Cl₂` should be conserved
tests/unit/test_gradient_checkpointing.py:38:    "dN_i_dt",
tests/unit/test_training_modules.py:34:    "dN_i_dt",
tests/unit/test_coupled_esm.py:3:Tests wiring, SST feedback, energy conservation, and configuration
tests/unit/test_convergence_rates.py:161:            use_conservation_fixer=False,
tests/unit/test_scale_global_reductions.py:1:"""Category 4: Global reductions & conservation under sharding.
tests/unit/test_scale_global_reductions.py:4:conservation fixer, and consistency with Gaussian-grid total area.
tests/unit/test_scale_global_reductions.py:17:from legoesm.core.conservation import _global_area_sum, _total_area
tests/unit/test_scale_global_reductions.py:79:# _global_area_sum (conservation module)
tests/unit/test_williamson2_cdgrid.py:10:- Mass conservation (relative error)
tests/unit/test_williamson2_cdgrid.py:11:- Energy conservation (relative error)
tests/unit/test_williamson2_cdgrid.py:99:    def test_mass_conservation_5day(self):
tests/unit/test_williamson2_cdgrid.py:100:        """Mass should be conserved to machine precision over 5 days."""
tests/unit/test_williamson2_cdgrid.py:114:        print(f"  Mass conservation: rel_err = {rel_err:.2e}")
tests/unit/test_williamson2_cdgrid.py:116:                        f"Mass conservation failed: rel_err = {rel_err:.2e}")
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:9:the long-roll suite catches conservation drift, slow blow-ups, and
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:293:# Tracer mass conservation under filter (no physics tendency)
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:410:    integral must be approximately conserved (drift < 1 %) when the
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:471:        # Global mean conserved to ~1 % (the spectral filter has
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:472:        # eigenvalue 1 at n=0 → exact conservation in spectral space;
tests/unit/test_diff_atmosphere_physics.py:266:        ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"],
tests/unit/test_diff_atmosphere_physics.py:287:        ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"],
tests/unit/test_diff_atmosphere_physics.py:328:    @pytest.mark.parametrize("scheme", ["morrison", "thompson"])
tests/unit/test_diff_atmosphere_physics.py:331:        warm rain: ``N_i**(1/3)`` (depositional growth) and ice/snow/graupel
tests/unit/test_physics_smoke.py:120:                                     "morrison", "thompson"])
tests/unit/test_fv3_audit_harness.py:7:  4. One-step mass conservation
tests/unit/test_fv3_audit_harness.py:285:    """Mass should be conserved to machine precision with conservation fixer."""
tests/unit/test_fv3_audit_harness.py:287:    def test_one_step_mass_conservation(self):
tests/unit/test_fv3_audit_harness.py:290:        The conservation fixer operates in float32, so precision is ~1e-7.
tests/unit/test_fv3_audit_harness.py:302:            fix_mass=True, use_conservation_fixer=True,
tests/unit/test_fv3_audit_harness.py:323:        """Without conservation fixer, mass error should still be small."""
tests/unit/test_fv3_audit_harness.py:334:            fix_mass=False, use_conservation_fixer=False,
tests/unit/test_fv3_audit_harness.py:462:    def test_cosine_bell_mass_conservation(self):
tests/unit/test_fv3_audit_harness.py:463:        """Mass should be conserved during cosine bell transport."""
tests/unit/test_fv3_audit_harness.py:648:    def test_tc5_mass_conservation(self):
tests/unit/test_fv3_audit_harness.py:649:        """TC5 mass should be conserved with fixer."""
tests/unit/test_equation_fixes.py:238:        ("thompson", "thompson_microphysics"),
tests/unit/test_greens_function.py:339:            use_conservation_fixer=True,
tests/unit/test_hardware_runtime_config.py:13:from legoesm.core.conservation import _accumulation_dtype
tests/unit/test_hardware_runtime_config.py:29:        conservation=None,
tests/unit/test_hardware_runtime_config.py:36:        conservation=None,
tests/unit/test_hardware_runtime_config.py:125:                        "conservation": "float32",
tests/unit/test_hardware_runtime_config.py:144:        assert get_runtime_precision_dtype("conservation") == jnp.float32
tests/unit/test_physics_convection.py:3:Tests moisture conservation, energy conservation, CAPE reduction, stable
tests/unit/test_physics_convection.py:91:# 3a  Moisture conservation (SBM, DCA only -- see docstring)
tests/unit/test_physics_convection.py:95:def test_moisture_conservation(scheme):
tests/unit/test_physics_convection.py:109:    machine-precision column water conservation regardless of the
tests/unit/test_physics_convection.py:116:    is non-conservative (it imports a ``(1 - alpha_heat) * MC /
tests/unit/test_physics_convection.py:126:    design — water conservation is not the right invariant to test
tests/unit/test_physics_convection.py:136:    # With column-scaling, conservation is exact in floating point.
tests/unit/test_physics_convection.py:148:                f"{scheme} col {i}: moisture conservation rel_err = "
tests/unit/test_physics_convection.py:154:# 3b  Energy conservation (moist static energy) - SBM only
tests/unit/test_physics_convection.py:157:def test_sbm_energy_conservation():
tests/unit/test_physics_convection.py:433:# 3b extended -- closed energy budget across all conservative schemes
tests/unit/test_physics_convection.py:450:#   * Kuo is non-conservative by design (alpha_heat fraction sourced
tests/unit/test_physics_convection.py:461:def test_dca_extended_mse_conservation():
tests/unit/test_physics_convection.py:464:    DCA preserves layer-mean T per pair (col_dT ~ 0) and conserves total
tests/unit/test_operators_fv.py:79:    """Tests for the Lin-Rood conservative flux-form transport."""
tests/unit/test_operators_fv.py:91:    def test_conservation(self, grid):
tests/unit/test_operators_fv.py:152:    """Tests for the non-conservative advection operator."""
tests/unit/test_mpas_atmosphere.py:7:4. Mass conservation (hydrostatic)
tests/unit/test_mpas_atmosphere.py:191:    def test_mass_conservation(self):
tests/unit/test_mpas_atmosphere.py:192:        """Mass (sum of p_s * area) is conserved after one step."""
tests/unit/test_tiedtke.py:227:def test_tiedtke_downdraft_evap_conserves_water_locally():
tests/unit/test_tiedtke.py:233:          conserving');
tests/unit/test_tiedtke.py:244:      2. Column water conservation: column-integrated
tests/unit/test_tiedtke.py:280:    # (2) Column water conservation — net column water source is zero
tests/unit/test_tiedtke.py:374:# MSE conservation guard
tests/unit/test_tiedtke.py:380:        "moist-adiabat T_u) does not conserve column MSE on a closed "
tests/unit/test_tiedtke.py:381:        "(no-surface-flux) probe.  Currently ~96% non-conservation "
tests/unit/test_tiedtke.py:387:def test_tiedtke_mse_conservation_within_tolerance():
tests/atmosphere/hydrostatic/integration/test_amip_smoke.py:77:        use_conservation_fixer=True,
tests/ocean/unit/test_advection_dst3.py:5:2. DST-3 vertical advection: conservation, monotonicity, accuracy
tests/ocean/unit/test_advection_dst3.py:6:3. DST-3 horizontal advection: conservation, periodicity, accuracy
tests/ocean/unit/test_advection_dst3.py:7:4. Multi-dimensional advection: conservation
tests/ocean/unit/test_advection_dst3.py:107:    def test_conservation_closed_column(self):
tests/ocean/unit/test_advection_dst3.py:282:    def test_conservation_horizontal_flux(self):
tests/ocean/unit/test_advection_dst3.py:329:    def test_conservation(self):
tests/ocean/unit/test_advection_dst3.py:330:        """Total tracer (sum h*T*area) is conserved by multi-dim advection."""
tests/ocean/unit/test_advection_dst3.py:362:        # Total tendency should conserve tracer: sum((div_h + vert_div) * area) = 0
tests/ocean/unit/test_advection_dst3.py:366:            f"Multi-dim advection not conservative: {total_tendency}")
tests/ocean/unit/test_barotropic_cgrid.py:5:- Mass (volume) conservation
tests/ocean/unit/test_barotropic_cgrid.py:83:    """Volume should be approximately conserved."""
tests/ocean/unit/test_barotropic_cgrid.py:85:    def test_volume_conservation(self, grid, z_coord, igw_state):
tests/ocean/unit/test_advection_fct_zalesak.py:3:Replaces the old conservation-preserving ``alpha_face = min(alpha_left,
tests/ocean/unit/test_advection_fct_zalesak.py:64:    """The limited fluxes must keep mass conserved bit-for-bit."""
tests/ocean/unit/test_advection_fct_zalesak.py:70:        This is the conservation property the original face-min heuristic
tests/ocean/unit/test_mpas_ocean.py:34:from legoesm.ocean.conservation_mpas import (
tests/ocean/unit/test_mpas_ocean.py:38:    mpas_ocean_conservation_fixer,
tests/ocean/unit/test_mpas_ocean.py:443:        conservation property that prevents the biharmonic from leaking
tests/ocean/unit/test_mpas_ocean.py:821:    """Test conservation fixers."""
tests/ocean/unit/test_mpas_ocean.py:823:    def test_volume_conservation(self, state, mesh, z_coord):
tests/ocean/unit/test_mpas_ocean.py:847:    def test_heat_conservation(self, state, mesh, z_coord):
tests/ocean/unit/test_mpas_ocean.py:880:        """Full conservation fixer chain works."""
tests/ocean/unit/test_mpas_ocean.py:886:        state_fixed = mpas_ocean_conservation_fixer(
tests/ocean/unit/test_mpas_ocean.py:939:            state_fixed = mpas_ocean_conservation_fixer(
tests/ocean/unit/test_inertia_gravity_wave.py:11:- Mass (volume) conservation
tests/ocean/unit/test_inertia_gravity_wave.py:123:    def test_volume_conservation_10_steps(self, igw_grid, igw_z_coord, igw_state, ocean_config):
tests/ocean/unit/test_inertia_gravity_wave.py:124:        """Volume (mass) should be approximately conserved."""
tests/ocean/unit/test_inertia_gravity_wave.py:140:        # Volume conservation: relative to initial amplitude
tests/unit/test_compiled_segments.py:122:        dN_i_dt=jnp.zeros(shape_3d),
tests/ocean/unit/test_eta_floor.py:1:"""Smoke test for `clamp_and_redistribute` (mass-conserving eta floor).
tests/ocean/unit/test_eta_floor.py:57:def test_mass_conservation_serial():
tests/ocean/unit/test_ocean.py:50:from legoesm.ocean.conservation import (
tests/ocean/unit/test_ocean.py:54:    ocean_conservation_fixer,
tests/ocean/unit/test_ocean.py:633:    def test_column_momentum_conserved_in_closed_column(self):
tests/ocean/unit/test_ocean.py:708:        vertically integrated momentum is conserved.
tests/ocean/unit/test_ocean.py:801:            use_conservation_fixer=False,
tests/ocean/unit/test_ocean.py:822:            use_conservation_fixer=False,
tests/ocean/unit/test_ocean.py:941:    """Tests for ocean conservation fixers."""
tests/ocean/unit/test_ocean.py:943:    def test_volume_conservation(self, ocean_grid, ocean_z_coord, ocean_state):
tests/ocean/unit/test_ocean.py:1091:    """Tests for spectral ocean model wiring and conservation."""
tests/ocean/unit/test_ocean.py:1120:    def test_conservation_fixer_restores_volume_heat_salt(self):
tests/ocean/unit/test_ocean.py:1130:            _spectral_conservation_fixer,
tests/ocean/unit/test_ocean.py:1139:        cfg = SpectralOceanConfig(use_conservation_fixer=True, min_water_column_m=0.5)
tests/ocean/unit/test_ocean.py:1152:        fixed = _spectral_conservation_fixer(
tests/ocean/unit/test_ocean.py:1199:            grid, z_coord, SpectralOceanConfig(use_conservation_fixer=False),
tests/ocean/unit/test_ocean.py:1299:    """Tests for mass, heat, and salt conservation over many timesteps.
tests/ocean/unit/test_ocean.py:1302:    spurious drift, and that the conservation fixer restores invariants.
tests/ocean/unit/test_ocean.py:1323:    def test_longrun_conservation_with_fixer(self):
tests/ocean/unit/test_ocean.py:1324:        """50-step integration with conservation fixer: drift should be small."""
tests/ocean/unit/test_ocean.py:1334:            use_conservation_fixer=True,
tests/ocean/unit/test_ocean.py:1345:        # With the conservation fixer, mean eta drift should be very small.
tests/ocean/unit/test_ocean.py:1376:            use_conservation_fixer=False,
tests/ocean/unit/test_implicit_solver.py:47:    # Mass-like quantity should be conserved (zero-flux BCs)
tests/ocean/unit/test_freshwater.py:4:mass terms, coupler integration, and conservation with freshwater.
tests/ocean/unit/test_freshwater.py:439:            use_conservation_fixer=False,
tests/ocean/unit/test_freshwater.py:451:            use_conservation_fixer=False,
tests/ocean/unit/test_freshwater.py:465:            use_conservation_fixer=False,
tests/ocean/unit/test_freshwater.py:491:            use_conservation_fixer=False,
tests/ocean/unit/test_freshwater.py:516:            use_conservation_fixer=False,
tests/ocean/unit/test_freshwater.py:540:            use_conservation_fixer=True,
tests/ocean/unit/test_freshwater.py:566:            use_conservation_fixer=False,
tests/ocean/unit/test_freshwater.py:619:            use_conservation_fixer=False,
tests/ocean/unit/test_freshwater.py:631:            use_conservation_fixer=False,
tests/ocean/unit/test_freshwater.py:650:            use_conservation_fixer=False,
tests/ocean/unit/test_freshwater.py:675:            use_conservation_fixer=False,
tests/ocean/unit/test_freshwater.py:700:            use_conservation_fixer=False,
tests/ocean/unit/test_freshwater.py:725:            use_conservation_fixer=True,
tests/ocean/unit/test_freshwater.py:753:            use_conservation_fixer=False,
tests/unit/test_production_blockers.py:427:            # (dynamics, conservation) → expected mode
tests/unit/test_production_blockers.py:436:                "hardware.precision.conservation": cons,
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:84:        use_conservation_fixer=True,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:6:  are conserved by advection on the sphere).
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:95:        so the conservative advection ``-∇·(q v) + q · ∇·v`` is exactly
tests/ocean/unit/test_implicit_vertical_solver.py:108:    def test_mass_conserved_uniform_K(self):
tests/ocean/unit/test_implicit_vertical_solver.py:115:        # Volume-weighted total is what the no-flux BC conserves.
tests/ocean/unit/test_implicit_vertical_solver.py:120:    def test_mass_conserved_nonuniform_K(self):
tests/ocean/unit/test_implicit_vertical_solver.py:132:    def test_mass_conserved_many_steps(self):
tests/ocean/unit/test_implicit_vertical_solver.py:240:        # Total conserved
tests/unit/test_runtime_bootstrap.py:345:            "hardware.precision.conservation": None,
tests/unit/test_runtime_bootstrap.py:371:    def test_yaml_mixed_via_conservation(self):
tests/unit/test_runtime_bootstrap.py:375:            "hardware.precision.conservation": "float64",
tests/unit/test_runtime_bootstrap.py:416:        set_runtime_precision_policy(dynamics="float32", conservation=None)
tests/unit/test_runtime_bootstrap.py:419:        assert policy["conservation"] is None
tests/unit/test_runtime_bootstrap.py:434:            "hardware.precision.conservation": None,
tests/atmosphere/hydrostatic/test_cases/dcmip_transport.py:199:    # q4: conservation tracer
tests/atmosphere/hydrostatic/test_cases/dcmip_transport.py:771:      q4: 1 - 0.3*(q1 + q2 + q3) (conservation check)
tests/atmosphere/hydrostatic/test_cases/dcmip_transport.py:822:    # q4: conservation tracer
tests/unit/test_timestepping.py:282:        """Energy should be nearly conserved for a harmonic oscillator."""
tests/unit/test_symmetry_invariance.py:61:            use_conservation_fixer=True,
tests/unit/test_operators_fc.py:65:def test_flux_divergence_conserves(grid_and_config):
tests/unit/test_coupler.py:269:    """Invalid static masks are clipped/rescaled to a conservative partition."""
tests/unit/test_coupler.py:831:def test_lake_freezing_energy_conservation():
tests/unit/test_coupler.py:900:        use_conservation_fixer=False,
tests/unit/test_coupler.py:952:        use_conservation_fixer=False,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:10:from legoesm.core.conservation import fix_mass_hydrostatic
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:458:            use_conservation_fixer=False,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:474:            use_conservation_fixer=False,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:486:    def test_mass_conservation_with_fixer(self, grid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:491:            use_conservation_fixer=True,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:503:            f"Mass not conserved: before={mass_before:.6e}, after={mass_after:.6e}"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:572:            use_conservation_fixer=False,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:715:                use_conservation_fixer=True,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:746:                use_conservation_fixer=True,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:770:    """Tests for mass conservation options in PE model."""
tests/atmosphere/hydrostatic/unit/test_microphysics.py:34:from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
tests/atmosphere/hydrostatic/unit/test_microphysics.py:205:        # With zero moisture, all ice/snow/graupel/number tendencies are exact zero
tests/atmosphere/hydrostatic/unit/test_microphysics.py:343:        assert out.dN_i_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:401:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:407:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:412:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:417:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:421:    def test_graupel_from_riming(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:422:        """With strong riming, graupel tendencies should be nonzero."""
tests/atmosphere/hydrostatic/unit/test_microphysics.py:432:            rime_to_graupel_threshold=1e-6,
tests/atmosphere/hydrostatic/unit/test_microphysics.py:434:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:441:            out = thompson_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:450:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:635:                       "morrison", "thompson", "ml_emulator", "none"]:
tests/unit/test_diff_ocean.py:45:            use_conservation_fixer=False,
tests/unit/test_diff_ocean.py:146:            use_conservation_fixer=False,
tests/unit/test_diff_ocean.py:200:            use_conservation_fixer=False,
tests/unit/test_dcmip_transport.py:80:    def test_init_q4_conservation(self, grid, sigma_coord):
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:239:    def test_thompson(self):
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:241:        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="thompson"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:243:        _check_tendencies(tend, "microphysics/thompson")
tests/unit/test_issue_fixes.py:10:C5: Global dry-mass conservation (ps correction restores mass exactly)
tests/unit/test_issue_fixes.py:22:from legoesm.core.conservation import fix_mass_hydrostatic
tests/unit/test_issue_fixes.py:155:        # With zero surface flux and zero-flux top, total phi*rho*dz is conserved
tests/unit/test_issue_fixes.py:159:            f"Heat not conserved: old={total_old:.4f}, new={total_new:.4f}")
tests/unit/test_issue_fixes.py:335:# C5: Global dry-mass conservation (ps correction)
tests/unit/test_issue_fixes.py:347:        # Perturb p_s (simulating a timestep that doesn't conserve mass)
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:237:# Mass conservation — uses explicit target_mass API
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:242:    def test_mass_conservation_50_steps(self, grid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:258:        assert rel_err < 1e-5, f"Mass conservation error: {rel_err}"
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:292:# Energy conservation (KE + enthalpy)
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:989:# Tracer mass conservation (mass-weighted)
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:993:    """Verify that ∫ q·dp·dA (tracer mass) is conserved by flux-form transport."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:1003:    def test_tracer_mass_conserved_50_steps(self, grid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:1004:        """A Gaussian tracer bump should conserve ∫ q·dp·dA over 50 steps."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:1031:            f"Tracer mass conservation error: {rel_err:.4e} "
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:1035:    def test_tracer_mass_conserved_hybrid_20_steps(self, grid):
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:1036:        """Tracer mass ∫ q·dp·dA must be conserved on hybrid coords too.
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:1072:            f"Hybrid tracer mass conservation error: {rel_err:.4e} "
tests/atmosphere/hydrostatic/unit/test_convection.py:5:- SBM: shapes, enthalpy conservation, precipitation, trigger, differentiability
tests/atmosphere/hydrostatic/unit/test_convection.py:218:    def test_enthalpy_conservation(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:219:        """Column enthalpy tendency should be approximately conserved.
tests/atmosphere/hydrostatic/unit/test_convection.py:221:        The linearized enthalpy correction ensures approximate conservation.
tests/atmosphere/hydrostatic/unit/test_convection.py:240:        # Relative conservation: enthalpy residual should be small fraction of total
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:26:from legoesm.core.conservation import compute_conservation_diagnostics
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:82:            use_conservation_fixer=False,
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:127:                use_conservation_fixer=False,
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:132:        # Use CDGrid diagnostics for conservation
tests/atmosphere/hydrostatic/unit/test_radiation.py:4:- Gray radiation: energy conservation, heating rate bounds, moisture feedback
tests/atmosphere/hydrostatic/unit/test_radiation.py:267:    def test_energy_conservation(self):
tests/atmosphere/hydrostatic/unit/test_radiation.py:268:        """Column energy should be approximately conserved.
tests/atmosphere/hydrostatic/unit/test_radiation.py:288:        # Energy conservation: F_net(TOA) - F_net(sfc) ≈ -col_heating * (c_p/g not needed since hr already in K/s)
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:101:# Per-scheme configs (one per scheme — keep defaults conservative)
tests/unit/test_physics_units.py:134:def test_sbm_column_mse_conservation():
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:252:        # Kessler doesn't predict graupel) which IS in the output map
tests/atmosphere/hydrostatic/unit/test_turbulence.py:4:- Vertical diffusion: shape, conservation, smoothing, differentiability
tests/atmosphere/hydrostatic/unit/test_turbulence.py:131:    def test_conserves_column_integral(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:132:        """With zero surface flux, column integral should be conserved."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:144:        # Column integral: sum(phi * rho * dz) should be conserved

codex
One potential issue is not in the two fixed mass budgets but in the number budget: the schemes can create ice number without an accompanying ice-mass seed. I’m checking the integration layer now to see whether a later positivity/update step masks that or whether it reaches prognostic state unchanged.
exec
/bin/zsh -lc "sed -n '360,500p' src/legoesm/atmosphere/physics/microphysics/integration.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
        # Reshape to columns
        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        rho_col = rho_total.reshape(ncol, nlev)

        # Map tracers -> HydrometeorState
        # [0]=q_v, [1]=q_c, [2]=q_r, [3]=q_i, [4]=q_s, [5]=q_g, [6]=N_c, [7]=N_r, [8]=N_i
        def _get_tracer(idx):
            if n_tracers > idx:
                return tracers[..., idx].reshape(ncol, nlev)
            return jnp.zeros((ncol, nlev), dtype=_state_dtype)

        q_v_col = _get_tracer(0)
        hydrometeors = HydrometeorState(
            q_c=_get_tracer(1),
            q_r=_get_tracer(2),
            q_i=_get_tracer(3),
            q_s=_get_tracer(4),
            q_g=_get_tracer(5),
            N_c=_get_tracer(6),
            N_r=_get_tracer(7),
            N_i=_get_tracer(8),
        )

        if is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = MicrophysicsEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output, key=key,
                )
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half, rho_col, dz, dt,
                scheme_config, _ml_model_cache[0],
            )
        else:
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half, rho_col, dz, dt, scheme_config,
            )

        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner).
        dT_dt = micro_out.dT_dt.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        # Map output fields -> dtracers_dt
        dtracers = jnp.zeros_like(tracers)
        # Tracer mapping: 0=q_v, 1=q_c, 2=q_r, 3=q_i, 4=q_s, 5=q_g, 6=N_c, 7=N_r, 8=N_i
        tend_fields = [
            micro_out.dq_v_dt, micro_out.dq_c_dt, micro_out.dq_r_dt,
            micro_out.dq_i_dt, micro_out.dq_s_dt, micro_out.dq_g_dt,
            micro_out.dN_c_dt, micro_out.dN_r_dt, micro_out.dN_i_dt,
        ]
        for idx, field in enumerate(tend_fields):
            if n_tracers > idx:
                dtracers = dtracers.at[..., idx].set(field.reshape(shape_3d))

        return NonHydrostaticTendencies(
            du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_micro", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_micro", dims=dims_3d, units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_micro", dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_micro", dims=dims_3d, units="K/s"),
            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_micro", dims=dims_3d, units="kg/m^3/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
            dtracers_dt=Field(data=dtracers, name="dtracers_dt_micro", dims=dims_tr, units="1/s"),
        )

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_microphysics(
    microphysics_config: MicrophysicsConfig,
    dt: float,
) -> Callable:
    """Create microphysics physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord, grid_fields=None) -> SpectralHydrostaticState

    The bridge pulls ``q_v`` and the full hydrometeor state out of
    ``state.tracers`` (when present), runs the column microphysics
    backend, and returns a ``SpectralHydrostaticState`` whose ``T_hat``
    carries the spectral latent-heating tendency *and* whose ``tracers``
    dict carries grid-space ``dq_v_dt`` / ``dq_c_dt`` / ``dq_r_dt`` /
    etc.  The dycore RHS (``spectral_pe_tendencies``) adds these tracer
    tendencies to its own advective tendencies during the SSP-RK stages.
    """
    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]

    # Tracer key → MicrophysicsOutput attribute name.  Mirrors the
    # ``HydrometeorState`` field layout in ``microphysics/output.py``
    # plus ``q_v``.  The dycore RHS only flows tendencies for keys that
    # exist on the input ``state.tracers``; missing keys are silently
    # dropped (no carry to write into).
    _TRACER_TEND_MAP = {
        "q_v": "dq_v_dt",
        "q_c": "dq_c_dt",
        "q_r": "dq_r_dt",
        "q_i": "dq_i_dt",
        "q_s": "dq_s_dt",
        "q_g": "dq_g_dt",
        "N_c": "dN_c_dt",
        "N_r": "dN_r_dt",
        "N_i": "dN_i_dt",
    }

    def physics_fn(state, grid, sigma_coord, grid_fields=None):
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralHydrostaticState,
            spectral_pe_to_grid,
        )
        from legoesm.atmosphere.physics._shared import zero_like_tracers
        from legoesm.grids.gaussian import sh_analysis_3d

        # Transform spectral state to grid space
        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        T = fields['T']
        p_s = fields['p_s']

        nlev = sigma_coord.n_levels
        n_lat, n_lon = p_s.shape

        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)
        # Pin the column-physics dtype to the gridded state precision so
        # we do not silently flow x64 zeros into the column path.
        _state_dtype = T.dtype

        if micro_fn is None:

exec
/bin/zsh -lc "sed -n '380,460p' tests/atmosphere/hydrostatic/unit/test_microphysics.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "sed -n '180,230p' tests/atmosphere/hydrostatic/unit/test_microphysics.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))

    def test_evaporation_enthalpy_balance(self):
        """Warm-rain evaporation cooling should close latent energy tendency."""
        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
        assert float(jnp.max(jnp.abs(residual))) < 5e-3


# ======================================================================
# Thompson tests
# ======================================================================

class TestThompson:

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert out.dT_dt.shape == T.shape
        assert out.dq_g_dt.shape == T.shape

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            assert jnp.all(jnp.isfinite(field))

    def test_graupel_from_riming(self):
        """With strong riming, graupel tendencies should be nonzero."""
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        # Boost cloud water and ice to promote riming
        h = h._replace(
            q_c=jnp.full_like(h.q_c, 5e-3),
            q_i=jnp.full_like(h.q_i, 5e-3),
            q_s=jnp.full_like(h.q_s, 5e-3),
        )
        config = ThompsonConfig(
            rime_coeff=10.0,
            rime_to_graupel_threshold=1e-6,
        )
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
        assert float(jnp.max(jnp.abs(out.dq_g_dt))) > 0

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()

        def loss(T_in):
            out = thompson_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))

    def test_evaporation_enthalpy_balance(self):
        """Warm-rain evaporation cooling should close latent energy tendency."""
        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
        assert float(jnp.max(jnp.abs(residual))) < 5e-3


# ======================================================================
# ML Emulator tests
# ======================================================================

class TestMLEmulator:


 succeeded in 0ms:
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            assert jnp.all(jnp.isfinite(field)), f"Non-finite in {field}"

    def test_dry_air_zero_tendency(self):
        ncol, nlev = 4, 10
        T = jnp.full((ncol, nlev), 280.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        q_v = jnp.zeros((ncol, nlev))
        rho = p_full / (constants.R_d * T)
        dz = jnp.full((ncol, nlev), 500.0)
        h = make_zero_hydrometeors(ncol, nlev)
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        # With zero moisture, all ice/snow/graupel/number tendencies are exact zero
        assert float(jnp.max(jnp.abs(out.dq_i_dt))) == 0.0
        assert float(jnp.max(jnp.abs(out.dq_s_dt))) == 0.0
        assert float(jnp.max(jnp.abs(out.dq_g_dt))) == 0.0

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()

        def loss(T_in):
            out = kessler_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))

    def test_evaporation_enthalpy_balance(self):
        """Evaporation cooling should balance vapor tendency latent energy."""
        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
        assert float(jnp.max(jnp.abs(residual))) < 1e-6


# ======================================================================
# Sundqvist tests
# ======================================================================

exec
/bin/zsh -lc "rg -n \"q_i|N_i|HydrometeorState|dN_i_dt|clip\\(.*N_i|clip\\(.*q_i\" src/legoesm/atmosphere/physics/microphysics/integration.py tests/atmosphere/hydrostatic/unit/test_microphysics.py tests/unit/test_physics_microphysics.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
tests/atmosphere/hydrostatic/unit/test_microphysics.py:24:    HydrometeorState,
tests/atmosphere/hydrostatic/unit/test_microphysics.py:72:    hydrometeors = HydrometeorState(
tests/atmosphere/hydrostatic/unit/test_microphysics.py:74:        q_i=jnp.zeros((ncol, nlev)),
tests/atmosphere/hydrostatic/unit/test_microphysics.py:79:        N_i=jnp.zeros((ncol, nlev)),
tests/atmosphere/hydrostatic/unit/test_microphysics.py:96:    q_i = jnp.zeros((ncol, nlev)).at[:, 2:6].set(1e-4)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:104:    hydrometeors = HydrometeorState(
tests/atmosphere/hydrostatic/unit/test_microphysics.py:105:        q_c=q_c, q_r=q_r, q_i=q_i, q_s=q_s, q_g=q_g,
tests/atmosphere/hydrostatic/unit/test_microphysics.py:108:        N_i=jnp.full((ncol, nlev), 1e3),
tests/atmosphere/hydrostatic/unit/test_microphysics.py:125:    hydrometeors = HydrometeorState(
tests/atmosphere/hydrostatic/unit/test_microphysics.py:128:        q_i=z,
tests/atmosphere/hydrostatic/unit/test_microphysics.py:133:        N_i=jnp.full((ncol, nlev), 1e3),
tests/atmosphere/hydrostatic/unit/test_microphysics.py:206:        assert float(jnp.max(jnp.abs(out.dq_i_dt))) == 0.0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:342:        assert out.dq_i_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:343:        assert out.dN_i_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:373:        assert float(jnp.max(jnp.abs(out.dq_i_dt))) < 1e-6
tests/atmosphere/hydrostatic/unit/test_microphysics.py:427:            q_i=jnp.full_like(h.q_i, 5e-3),
src/legoesm/atmosphere/physics/microphysics/integration.py:36:from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
src/legoesm/atmosphere/physics/microphysics/integration.py:219:        hydrometeors = HydrometeorState(
src/legoesm/atmosphere/physics/microphysics/integration.py:222:            q_i=_get_tracer("q_i"),
src/legoesm/atmosphere/physics/microphysics/integration.py:227:            N_i=_get_tracer("N_i"),
src/legoesm/atmosphere/physics/microphysics/integration.py:258:            "q_i": Field(data=micro_out.dq_i_dt.reshape(shape_3d),
src/legoesm/atmosphere/physics/microphysics/integration.py:259:                         name="dq_i_dt_micro", dims=dims_3d, units="kg/kg/s"),
src/legoesm/atmosphere/physics/microphysics/integration.py:365:        # Map tracers -> HydrometeorState
src/legoesm/atmosphere/physics/microphysics/integration.py:366:        # [0]=q_v, [1]=q_c, [2]=q_r, [3]=q_i, [4]=q_s, [5]=q_g, [6]=N_c, [7]=N_r, [8]=N_i
src/legoesm/atmosphere/physics/microphysics/integration.py:373:        hydrometeors = HydrometeorState(
src/legoesm/atmosphere/physics/microphysics/integration.py:376:            q_i=_get_tracer(3),
src/legoesm/atmosphere/physics/microphysics/integration.py:381:            N_i=_get_tracer(8),
src/legoesm/atmosphere/physics/microphysics/integration.py:408:        # Tracer mapping: 0=q_v, 1=q_c, 2=q_r, 3=q_i, 4=q_s, 5=q_g, 6=N_c, 7=N_r, 8=N_i
src/legoesm/atmosphere/physics/microphysics/integration.py:411:            micro_out.dq_i_dt, micro_out.dq_s_dt, micro_out.dq_g_dt,
src/legoesm/atmosphere/physics/microphysics/integration.py:412:            micro_out.dN_c_dt, micro_out.dN_r_dt, micro_out.dN_i_dt,
src/legoesm/atmosphere/physics/microphysics/integration.py:460:    # ``HydrometeorState`` field layout in ``microphysics/output.py``
src/legoesm/atmosphere/physics/microphysics/integration.py:468:        "q_i": "dq_i_dt",
src/legoesm/atmosphere/physics/microphysics/integration.py:473:        "N_i": "dN_i_dt",
src/legoesm/atmosphere/physics/microphysics/integration.py:542:        hydrometeors = HydrometeorState(
src/legoesm/atmosphere/physics/microphysics/integration.py:545:            q_i=_get_tracer("q_i"),
src/legoesm/atmosphere/physics/microphysics/integration.py:550:            N_i=_get_tracer("N_i"),
tests/unit/test_physics_microphysics.py:25:    HydrometeorState, make_zero_hydrometeors,
tests/unit/test_physics_microphysics.py:62:    hydro = HydrometeorState(
tests/unit/test_physics_microphysics.py:65:        q_i=jnp.zeros((ncol, nlev)),
tests/unit/test_physics_microphysics.py:70:        N_i=jnp.zeros((ncol, nlev)),
tests/unit/test_physics_microphysics.py:186:        ice_formation = out.dq_i_dt[warm_mask]
tests/unit/test_physics_microphysics.py:190:            f"{scheme}: ice forms above freezing, max dq_i_dt = {max_ice_form:.2e}"
tests/unit/test_physics_microphysics.py:238:# Donor clamps: Morrison/Thompson q_c, q_i sinks must not over-extract
tests/unit/test_physics_microphysics.py:253:    scaling source terms in dq_r/dq_i/dq_s by the same factor).
tests/unit/test_physics_microphysics.py:269:    q_i = jnp.full((ncol, nlev), 1e-4)
tests/unit/test_physics_microphysics.py:271:    hydro = HydrometeorState(
tests/unit/test_physics_microphysics.py:272:        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
tests/unit/test_physics_microphysics.py:276:        N_i=1e4 * jnp.ones_like(q_c),
tests/unit/test_physics_microphysics.py:295:    """Thompson melt processes must not over-extract q_i above freezing.
tests/unit/test_physics_microphysics.py:299:    default rates, q_i was driven below zero.  After fix, melt is clamped
tests/unit/test_physics_microphysics.py:314:    q_i = jnp.full((ncol, nlev), 1e-4)
tests/unit/test_physics_microphysics.py:316:    hydro = HydrometeorState(
tests/unit/test_physics_microphysics.py:317:        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
tests/unit/test_physics_microphysics.py:321:        N_i=1e4 * jnp.ones_like(q_c),
tests/unit/test_physics_microphysics.py:328:    q_i_after = q_i + out.dq_i_dt * dt
tests/unit/test_physics_microphysics.py:329:    # Skip the top level: with q_i uniform throughout the column the top
tests/unit/test_physics_microphysics.py:333:    min_qi_interior = float(jnp.min(q_i_after[:, 1:]))
tests/unit/test_physics_microphysics.py:335:        f"Thompson: interior q_i went negative ({min_qi_interior:.3e}) at "
tests/unit/test_physics_microphysics.py:336:        f"T=280 K with dt=1200s — melt rate × q_i × melt_frac × dt exceeded "
tests/unit/test_physics_microphysics.py:337:        "q_i without a donor clamp."
tests/unit/test_physics_microphysics.py:343:    must subtract from the SOURCE species (q_i for ``riming_i``,
tests/unit/test_physics_microphysics.py:348:    ``q_i = 0`` (no ice to be rimed), ``q_s > 0`` (snow that gets
tests/unit/test_physics_microphysics.py:350:    this column ``riming_i = 0`` (no q_i to rime) and ``riming_s > 0``
tests/unit/test_physics_microphysics.py:353:        dq_i_dt -= rime_to_graupel              # full subtraction!
tests/unit/test_physics_microphysics.py:356:    drives ``q_i`` negative and creates 1.5× extra mass — both
tests/unit/test_physics_microphysics.py:372:    q_i = jnp.zeros((ncol, nlev))                # zero ice — would be drained negative pre-fix
tests/unit/test_physics_microphysics.py:374:    hydro = HydrometeorState(
tests/unit/test_physics_microphysics.py:375:        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
tests/unit/test_physics_microphysics.py:379:        N_i=jnp.zeros_like(q_c),
tests/unit/test_physics_microphysics.py:386:    q_i_after = q_i + out.dq_i_dt * dt
tests/unit/test_physics_microphysics.py:387:    # q_i must remain non-negative (donor split: only riming_i drains q_i).
tests/unit/test_physics_microphysics.py:388:    min_qi = float(jnp.min(q_i_after))
tests/unit/test_physics_microphysics.py:390:        f"Thompson: q_i went negative ({min_qi:.3e}) when starting at "
tests/unit/test_physics_microphysics.py:391:        "q_i=0 with q_s>0 and active riming.  Pre-fix the rime_to_graupel "
tests/unit/test_physics_microphysics.py:392:        "subtracted the full conversion from q_i regardless of which "
tests/unit/test_physics_microphysics.py:397:    # Mass conservation: ∑ dq_i + dq_s + dq_g (pure ice phase) should
tests/unit/test_physics_microphysics.py:423:    Moist enthalpy ``h = c_pd T + L_v q_v - L_f * q_ice`` is conserved
tests/unit/test_physics_microphysics.py:425:        c_pd * dT_dt + L_v * dq_v_dt - L_f * (dq_i + dq_s + dq_g)
tests/unit/test_physics_microphysics.py:452:    q_i = jnp.full((ncol, nlev), 5e-4)
tests/unit/test_physics_microphysics.py:454:    hydro = HydrometeorState(
tests/unit/test_physics_microphysics.py:455:        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
tests/unit/test_physics_microphysics.py:459:        N_i=1e4 * jnp.ones_like(q_c),
tests/unit/test_physics_microphysics.py:466:    # is approximately zero (uniform q_i, q_s ⇒ flux_in ≈ flux_out).
tests/unit/test_physics_microphysics.py:471:        - constants.L_f * (out.dq_i_dt[0, k] + out.dq_s_dt[0, k] + out.dq_g_dt[0, k])
tests/unit/test_physics_microphysics.py:475:        constants.L_f * jnp.abs(out.dq_i_dt[0, k] + out.dq_s_dt[0, k] + out.dq_g_dt[0, k])

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/_plume.py | sed -n '534,602p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/thompson.py | sed -n '105,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   534	
   535	        # Entrainment of environmental T, q via the analytic relaxation
   536	        # ``X(z+dz) = X_e + (X_prev - X_e) * exp(-eps · dz)`` — exact for
   537	        # the linear ODE ``dX/dz = -eps · (X - X_e)`` and always bounded
   538	        # between ``X_prev`` and ``X_e``.  An earlier explicit-Euler form
   539	        # ``X_prev + eps · dz · (X_e - X_prev)`` overshoots past ``X_e``
   540	        # for ``eps · dz > 1`` (e.g. Bechtold ``epsilon_shallow=3e-3``
   541	        # with a 500–1000 m layer gives ``eps·dz ∈ [1.5, 3]``) — making
   542	        # ``q_u_ent`` go negative and zeroing the gradient w.r.t. ``eps``.
   543	        # The exponential form is dimensionally identical and AD-safe
   544	        # everywhere (Codex stop-time review cycle 2: "plume entrainment
   545	        # still uses unstable explicit Euler").
   546	        decay_eps = jnp.exp(-eps * dz)
   547	        T_u_ent = T_e + (T_u_prev - T_e) * decay_eps
   548	        q_u_ent = q_e + (q_u_prev - q_e) * decay_eps
   549	
   550	        # Use the analytic moist-adiabatic lapse rate from
   551	        # ``moist_adiabat_lapse_rate`` (Iribarne–Godson) at the
   552	        # entrained parcel state.  The function evaluates dT/dp on
   553	        # the moist adiabat assuming the parcel is saturated; for
   554	        # *unsaturated* parcels this is approximate but matches the
   555	        # behavior of the existing ``compute_moist_adiabat`` helper
   556	        # that we're benchmarked against.  Latent heating is already
   557	        # baked into the lapse rate, so the post-hoc condensation
   558	        # step below does NOT add an additional ``L_v/c_pd *
   559	        # condensate`` correction — that would double-count.
   560	        rho_u_ent = p_e / (constants.R_d * jnp.maximum(T_u_ent, 100.0))
   561	        # ``moist_adiabat_lapse_rate`` returns dT/dp [K/Pa]; convert
   562	        # to dT/dz [K/m] via dp/dz = -rho*g.
   563	        Gamma_moist_per_pa = moist_adiabat_lapse_rate(T_u_ent, p_e)
   564	        dT_dz = Gamma_moist_per_pa * (-rho_u_ent * g)
   565	
   566	        T_u = T_u_ent + dT_dz * dz
   567	
   568	        # Condense any super-saturation into cloud water.  This is the
   569	        # diagnostic that resolves the q-budget; the temperature
   570	        # already incorporates the latent heat from condensation via
   571	        # the moist lapse rate.
   572	        q_sat_new = saturation_mixing_ratio(T_u, p_e).astype(_dtype)
   573	        condensate = jnp.maximum(q_u_ent - q_sat_new, 0.0).astype(_dtype)
   574	        q_u = (q_u_ent - condensate).astype(_dtype)
   575	        # Dilute plume cloud water by entrainment.  The continuity
   576	        # equation for an intensive quantity in an entraining-
   577	        # detraining plume is ``dq_c/dz = -eps · q_c + cond/M`` —
   578	        # environmental air carries q_c=0 so entrainment uniformly
   579	        # decreases ``q_c_u`` while detrainment is intensively
   580	        # neutral (it removes mass but not the per-kg amount).  An
   581	        # earlier formulation ``q_c_u = q_c_u_prev + condensate``
   582	        # carried ``q_c_u_prev`` forward unchanged and the plume's
   583	        # total water grew unphysically aloft (audit Codex finding:
   584	        # "plume cloud water is accumulated but not diluted by
   585	        # entrainment").
   586	        # Exponential dilution: ``q_c_u_ent = q_c_u_prev * exp(-eps·dz)``
   587	        # — exact analytic solution to ``dq_c/dz = -eps · q_c`` for an
   588	        # entraining plume with environment q_c=0.  Always non-negative,
   589	        # AD-safe everywhere.  An earlier explicit-Euler form
   590	        # ``max(q_c_u_prev * (1 - eps·dz), 0)`` zeroed the gradient
   591	        # whenever ``eps·dz > 1`` (corner case at Bechtold's
   592	        # ``epsilon_shallow=3e-3`` × dz=500 m and thicker — the clip
   593	        # branch dominated and made the test case ineffective for AD-
   594	        # tuning of ``eps`` in shallow convection).
   595	        q_c_u_ent = q_c_u_prev * decay_eps
   596	        q_c_u = (q_c_u_ent + condensate).astype(_dtype)
   597	
   598	        T_u = T_u.astype(_dtype)
   599	
   600	        # Buoyancy at this level.
   601	        B_u = T_u - T_e
   602	

 succeeded in 0ms:
   105	
   106	    # === ICE PHASE (Morrison processes) ===
   107	    T_freeze = constants.T_freeze
   108	    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))
   109	
   110	    # Ice nucleation
   111	    N_i_target = config.N_i0 * jnp.exp(
   112	        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
   113	    ) / jnp.clip(rho, 0.1)
   114	    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
   115	
   116	    # Depositional growth
   117	    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
   118	    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
   119	    dq_i_dep = (
   120	        config.dep_coeff
   121	        * jnp.maximum(S_i, 0.0)
   122	        * jnp.clip(q_i, 0.0)
   123	        * safe_pow(N_i, 1.0 / 3.0)
   124	        * f_ice
   125	    )
   126	
   127	    # Bergeron
   128	    berg_window = (
   129	        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
   130	        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
   131	    )
   132	    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window
   133	
   134	    # Riming
   135	    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   136	    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   137	    total_riming = riming_i + riming_s
   138	
   139	    # Aggregation
   140	    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
   141	
   142	    # Melting (clamp to available mass so an explicit Euler step cannot
   143	    # drive q_i / q_s / q_g negative — same pattern Morrison already uses).
   144	    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
   145	    dt_safe = jnp.maximum(dt, 1e-10)
   146	    melt_ice = jnp.minimum(
   147	        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
   148	        jnp.clip(q_i, 0.0) / dt_safe,
   149	    )
   150	    melt_snow = jnp.minimum(
   151	        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
   152	        jnp.clip(q_s, 0.0) / dt_safe,
   153	    )
   154	
   155	    # === GRAUPEL (Thompson extension) ===
   156	    # Split the rime → graupel conversion by donor: the fraction of
   157	    # rime_to_graupel that comes from q_i scales with riming_i, and
   158	    # the fraction from q_s scales with riming_s.  This avoids a
   159	    # mass-leak corner case where ``q_i = 0`` and ``riming_s > 0``:
   160	    # the previous form set ``rime_to_graupel ∝ total_riming``, then
   161	    # subtracted the FULL value from ``q_i`` (driving it negative)
   162	    # while only subtracting half from ``q_s`` and adding 150% to
   163	    # ``q_g`` — a non-conservative split that depended on the
   164	    # ad-hoc 1.0 / 0.5 / 1.5 coefficients.  (Codex audit cycle 2:
   165	    # "Thompson graupel conversion can draw from the wrong donor".)
   166	    graupel_frac = jax.nn.sigmoid(
   167	        config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
   168	    )
   169	    rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
   170	    rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
   171	    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
   172	    melt_graupel = jnp.minimum(
   173	        config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac,
   174	        jnp.clip(q_g, 0.0) / dt_safe,
   175	    )
   176	
   177	    # === DONOR CLAMP for q_c sinks (see morrison.py for rationale) ===
   178	    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s
   179	    qc_avail = jnp.clip(q_c, 0.0)
   180	    qc_scale = jnp.minimum(
   181	        1.0,
   182	        qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
   183	    )
   184	    dq_c_au = dq_c_au * qc_scale
   185	    dq_c_ac = dq_c_ac * qc_scale
   186	    bergeron = bergeron * qc_scale
   187	    riming_i = riming_i * qc_scale
   188	    riming_s = riming_s * qc_scale
   189	    total_riming = riming_i + riming_s
   190	    # Each rime-to-graupel donor scales with its parent riming term —
   191	    # which has already been scaled by qc_scale above.  Re-scaling
   192	    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``
   193	    # by ``qc_scale`` once preserves both per-donor proportionality and
   194	    # mass conservation.
   195	    rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
   196	    rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
   197	    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
   198	    dN_r_au = dN_r_au * qc_scale
   199	
   200	    # === SEDIMENTATION ===
   201	    # Marshall-Palmer fall speeds use fractional exponents (b_v_x in
   202	    # [0.25, 0.5]); guard the AD path with safe_pow.
   203	    rho_sfc = rho[:, -1:]
   204	    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
   205	    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
   206	    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
   207	    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
   208	    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
   209	    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
   210	    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
   211	    V_t_g = config.a_v_g * safe_pow(jnp.clip(q_g, 0.0) * rho_ratio, config.b_v_g)
   212	    V_t_g = jnp.clip(V_t_g, 0.0, 30.0)
   213	
   214	    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
   215	    sed_i = sedimentation_tendency(q_i, rho, V_t_i, dz)
   216	    sed_s = sedimentation_tendency(q_s, rho, V_t_s, dz)
   217	    sed_g = sedimentation_tendency(q_g, rho, V_t_g, dz)
   218	
   219	    # === LATENT HEATING ===
   220	    L_v = constants.L_v
   221	    L_s = constants.L_s
   222	    L_f = constants.L_f
   223	    c_pd = constants.c_pd
   224	    dT_dt = (
   225	        L_v * condensation / c_pd
   226	        - L_v * evaporation / c_pd
   227	        + L_s * dq_i_dep / c_pd
   228	        # Cloud water → ice/snow freezing releases L_f (Bergeron, riming).
   229	        # See morrison.py for the moist-enthalpy rationale; Thompson
   230	        # mirrors Morrison's ice-phase latent heating.
   231	        + L_f * (bergeron + riming_i + riming_s) / c_pd
   232	        - L_f * (melt_ice + melt_snow + melt_graupel) / c_pd
   233	    )
   234	
   235	    # === COMBINE TENDENCIES ===
   236	    # Conservation: each rime-to-graupel donor leaves its parent
   237	    # species and arrives in q_g.  The total mass moved is
   238	    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``.
   239	    # An earlier form used 1.0 / 0.5 / 1.5 splits on ``rime_to_graupel``
   240	    # which (a) drove ``q_i`` negative when only snow was being rimed
   241	    # (``q_i = 0`` but ``riming_s > 0``), and (b) created mass
   242	    # apparently from nothing in the same regime.  See the
   243	    # ``=== GRAUPEL ===`` block above for the donor-split rationale.
   244	    dq_v_dt = -condensation + evaporation - dq_i_dep
   245	    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
   246	    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel + sed_r
   247	    dq_i_dt = (
   248	        dq_i_dep + bergeron + riming_i - aggregation - melt_ice
   249	        - rime_to_graupel_from_i + sed_i
   250	    )
   251	    dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel_from_s + sed_s
   252	    dq_g_dt = rime_to_graupel - melt_graupel + sed_g
   253	
   254	    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
   255	    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
   256	    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
   257	
   258	    # Precipitation
   259	    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
   260	    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/morrison.py | sed -n '88,205p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
    88	
    89	    # 1. Ice nucleation (Cooper 1986, smoothed)
    90	    N_i_target = config.N_i0 * jnp.exp(
    91	        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
    92	    ) / jnp.clip(rho, 0.1)
    93	    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
    94	
    95	    # 2. Depositional growth
    96	    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
    97	    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
    98	    dq_i_dep = (
    99	        config.dep_coeff
   100	        * jnp.maximum(S_i, 0.0)
   101	        * jnp.clip(q_i, 0.0)
   102	        * safe_pow(N_i, 1.0 / 3.0)
   103	        * f_ice
   104	    )
   105	
   106	    # 3. Bergeron process: cloud water -> ice in mixed-phase zone
   107	    berg_window = (
   108	        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
   109	        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
   110	    )
   111	    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window
   112	
   113	    # 4. Riming: ice/snow collect cloud water
   114	    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   115	    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   116	
   117	    # 5. Snow aggregation: ice -> snow
   118	    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
   119	
   120	    # 6. Melting near T_freeze: ice/snow -> rain (clipped to available mass)
   121	    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
   122	    melt_ice = jnp.minimum(
   123	        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
   124	        jnp.clip(q_i, 0.0) / jnp.maximum(dt, 1e-10),
   125	    )
   126	    melt_snow = jnp.minimum(
   127	        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
   128	        jnp.clip(q_s, 0.0) / jnp.maximum(dt, 1e-10),
   129	    )
   130	
   131	    # === DONOR CLAMP for q_c sinks ===
   132	    # Scale q_c-consuming processes (autoconversion, accretion, Bergeron,
   133	    # riming) by a common factor so the total loss per timestep does not
   134	    # exceed available q_c.  Without this clamp, default rates at dt = 1200s
   135	    # in a mixed-phase column drive q_c negative on a single explicit step
   136	    # (bergeron alone gives bergeron_rate * q_c * dt = 1.2 * q_c).  Mass is
   137	    # conserved because each process's matching source term in dq_r/dq_i/dq_s
   138	    # gets the same scale factor (the rates appear once as sinks in dq_c and
   139	    # once as sources elsewhere, so a uniform rescale preserves the budget).
   140	    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s
   141	    qc_avail = jnp.clip(q_c, 0.0)
   142	    qc_scale = jnp.minimum(
   143	        1.0,
   144	        qc_avail / jnp.maximum(qc_sink_total * jnp.maximum(dt, 1e-10), 1e-30),
   145	    )
   146	    dq_c_au = dq_c_au * qc_scale
   147	    dq_c_ac = dq_c_ac * qc_scale
   148	    bergeron = bergeron * qc_scale
   149	    riming_i = riming_i * qc_scale
   150	    riming_s = riming_s * qc_scale
   151	    # Number tendency for autoconverted droplets must scale identically.
   152	    dN_r_au = dN_r_au * qc_scale
   153	
   154	    # === SEDIMENTATION ===
   155	    # Marshall-Palmer fall speeds V_t = a_v * (q * rho / rho_sfc)^b_v use
   156	    # fractional exponents (b_v_r=0.5, b_v_i=0.25, b_v_s=0.3); guard the
   157	    # AD path with safe_pow so cold-start columns (q=0) don't NaN gradients.
   158	    rho_sfc = rho[:, -1:]
   159	    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
   160	    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
   161	    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
   162	    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
   163	    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
   164	    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
   165	    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
   166	
   167	    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
   168	    sed_i = sedimentation_tendency(q_i, rho, V_t_i, dz)
   169	    sed_s = sedimentation_tendency(q_s, rho, V_t_s, dz)
   170	
   171	    # === LATENT HEATING ===
   172	    L_v = constants.L_v
   173	    L_s = constants.L_s
   174	    L_f = constants.L_f
   175	    c_pd = constants.c_pd
   176	
   177	    dT_dt = (
   178	        L_v * condensation / c_pd
   179	        - L_v * evaporation / c_pd
   180	        + L_s * dq_i_dep / c_pd
   181	        # Cloud water → ice/snow freezing releases latent heat of fusion
   182	        # (~333 kJ/kg).  Bergeron is liquid → ice via the WBF mechanism,
   183	        # riming is supercooled-droplet capture by ice/snow.  Both are
   184	        # phase changes that release L_f; the moist-enthalpy invariant
   185	        # ``h = c_pd T + L_v q_v - L_f q_ice`` requires this term for
   186	        # column conservation.  Magnitude estimate: ~2 K/day at default
   187	        # rates in mixed-phase clouds.
   188	        + L_f * (bergeron + riming_i + riming_s) / c_pd
   189	        - L_f * (melt_ice + melt_snow) / c_pd
   190	    )
   191	
   192	    # === COMBINE TENDENCIES ===
   193	    dq_v_dt = -condensation + evaporation - dq_i_dep
   194	    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
   195	    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + sed_r
   196	    dq_i_dt = dq_i_dep + bergeron + riming_i - aggregation - melt_ice + sed_i
   197	    dq_s_dt = aggregation + riming_s - melt_snow + sed_s
   198	
   199	    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
   200	    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
   201	    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
   202	
   203	    # Precipitation (rain + ice + snow at surface)
   204	    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
   205	    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)

codex
The key line references are clear now. I’m going to run the three targeted regression tests with pytest cache disabled, then I’ll base the review on the code plus those results.
exec
/bin/zsh -lc 'PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/unit/test_convection_plume.py::test_plume_T_q_entrainment_does_not_overshoot_environment tests/unit/test_convection_plume.py::test_plume_grad_through_epsilon_at_strong_entrainment tests/unit/test_physics_microphysics.py::test_thompson_rime_to_graupel_donor_split' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 127 in 0ms:
zsh:1: command not found: pytest

codex
`pytest` is not on this shell’s PATH, so I can’t execute the added tests from the current environment. I’ll still complete the review from static inspection and small direct algebra/probe checks where possible.
exec
/bin/zsh -lc 'rg -n "pytest|uv|poetry|pixi|hatch|tox|nox|requirements|jax" pyproject.toml setup.cfg setup.py tox.ini noxfile.py requirements*.txt environment*.yml .python-version' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 0ms:
zsh:1: no matches found: requirements*.txt

exec
/bin/zsh -lc "rg --files -g 'pyproject.toml' -g 'requirements*.txt' -g 'uv.lock' -g 'poetry.lock' -g 'pixi.toml' -g 'environment*.yml' -g 'tox.ini' -g 'noxfile.py'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
pyproject.toml

exec
/bin/zsh -lc "python -c \"import sys; print(sys.executable);\\ntry:\\n import jax; print('jax', jax.__version__)\\nexcept Exception as e: print('no jax', type(e).__name__, e)\\ntry:\\n import pytest; print('pytest', pytest.__version__)\\nexcept Exception as e: print('no pytest', type(e).__name__, e)\"" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 127 in 0ms:
zsh:1: command not found: python

exec
/bin/zsh -lc "sed -n '1,220p' pyproject.toml" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "legoesm"
version = "0.1.0"
description = "legoESM: A Differentiable Earth System Model in JAX"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "Pierre Gentine"},
]
keywords = ["earth-system-model", "jax", "differentiable", "weather", "climate"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]
dependencies = [
    "jax>=0.4.35",              # install minimum; tested with 0.8–0.9
    "jaxlib>=0.4.35",           # must match jax
    "equinox>=0.11",
    "optax>=0.2",
    "xarray>=2024.0",
    "zarr>=2.18",
    "matplotlib>=3.9",
    "cartopy>=0.23",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "netCDF4>=1.6",
    "dataclasses-json>=0.6",
    "scipy>=1.10",
    "pandas>=2.0",
]

[project.optional-dependencies]
ml = ["imageio>=2.30", "equinox>=0.11", "optax>=0.2"]
mesh = ["pymetis>=1.2", "scipy>=1.10"]
viz = ["matplotlib>=3.9", "cartopy>=0.23"]
data = [
    "gcsfs>=2024.2",
    "fsspec>=2024.2",
]
metal = ["jax-metal"]
mpi = [
    "mpi4py>=4.1,<5",
    "mpi4jax>=0.8,<0.9",
]
dev = [
    "pytest>=8.0",
    "pytest-xdist",
    "pytest-timeout",
    "ruff",
    "mypy",
    "pre-commit",
]
docs = ["sphinx", "myst-parser"]
all = [
    "legoesm[ml,mesh,viz,data,dev,docs]",
]

[project.scripts]
legoesm = "legoesm.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/legoesm"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short -m 'not slow'"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]

exec
/bin/zsh -lc 'command -v python; command -v python3; command -v uv; command -v pytest; command -v pip; command -v pip3' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
/opt/homebrew/bin/python3
/opt/homebrew/bin/pip3

exec
/bin/zsh -lc "python3 -c \"import sys; print(sys.executable);\\ntry:\\n import jax; print('jax', jax.__version__)\\nexcept Exception as e: print('no jax', type(e).__name__, e)\\ntry:\\n import pytest; print('pytest', pytest.__version__)\\nexcept Exception as e: print('no pytest', type(e).__name__, e)\"" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 0ms:
  File "<string>", line 1
    import sys; print(sys.executable);\ntry:\n import jax; print('jax', jax.__version__)\nexcept Exception as e: print('no jax', type(e).__name__, e)\ntry:\n import pytest; print('pytest', pytest.__version__)\nexcept Exception as e: print('no pytest', type(e).__name__, e)
                                       ^
SyntaxError: unexpected character after line continuation character

exec
/bin/zsh -lc "python3 -c \"import sys, importlib.util; print(sys.executable); print('jax_spec', importlib.util.find_spec('jax')); print('pytest_spec', importlib.util.find_spec('pytest'))\"" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
/opt/homebrew/opt/python@3.14/bin/python3.14
jax_spec ModuleSpec(name='jax', loader=<_frozen_importlib_external.SourceFileLoader object at 0x102d07e30>, origin='/opt/homebrew/lib/python3.14/site-packages/jax/__init__.py', submodule_search_locations=['/opt/homebrew/lib/python3.14/site-packages/jax'])
pytest_spec None

exec
/bin/zsh -lc 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -c $'"'import jax.numpy as jnp\\nfrom legoesm import constants\\nfrom legoesm.thermo import saturation_mixing_ratio_ice, saturation_mixing_ratio\\nfrom legoesm.atmosphere.physics.microphysics.output import make_zero_hydrometeors\\nfrom legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics\\nncol=nlev=1\\nT=jnp.array([[240.0]])\\np_full=jnp.array([[5e4]])\\np_half=jnp.array([[4.5e4,5.5e4]])\\nrho=p_full/(constants.R_d*T)\\ndz=jnp.array([[500.0]])\\nh=make_zero_hydrometeors(1,1)\\nqsi=saturation_mixing_ratio_ice(T,p_full)\\nqsl=saturation_mixing_ratio(T,p_full)\\nq_v=1.2*qsi\\nout=thompson_microphysics(T,q_v,h,p_full,p_half,rho,dz,dt=60.0)\\nprint(\"qsi\", float(qsi[0,0]), \"qsl\", float(qsl[0,0]), \"qv\", float(q_v[0,0]))\\nprint(\"dq_i_dt\", float(out.dq_i_dt[0,0]), \"dN_i_dt\", float(out.dN_i_dt[0,0]), \"dq_v_dt\", float(out.dq_v_dt[0,0]), \"dT_dt\", float(out.dT_dt[0,0]))'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 715ms:
Traceback (most recent call last):
  File "<string>", line 4, in <module>
    from legoesm.atmosphere.physics.microphysics.output import make_zero_hydrometeors
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/__init__.py", line 3, in <module>
    from legoesm.atmosphere.physics.radiation import (
    ...<2 lines>...
    )
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/__init__.py", line 40, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import rrtmgp_radiation
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp_radiation.py", line 14, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py", line 26, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp_common
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp_common.py", line 17, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp.config import radiative_transfer
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp/config/radiative_transfer.py", line 18, in <module>
    import dataclasses_json  # Used for JSON serialization.
    ^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'dataclasses_json'

codex
A direct runtime probe is also blocked by an incomplete Python environment: importing the microphysics module walks through the package radiation imports and fails on missing `dataclasses_json`. I’ll avoid overstating runtime verification; the remaining checks are static/algebraic.
exec
/bin/zsh -lc 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -c "import legoesm.constants as c; print(c.T_freeze)"' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 342ms:
273.15

exec
/bin/zsh -lc 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -c $'"'import importlib.util, sys, types, pathlib\\nroot=pathlib.Path(\"src\")\\n# Stub parent packages to avoid executing legoesm.atmosphere.physics.__init__\\nfor name in [\"legoesm.atmosphere\", \"legoesm.atmosphere.physics\", \"legoesm.atmosphere.physics.microphysics\"]:\\n    m=types.ModuleType(name); m.__path__=[]; sys.modules[name]=m\\ndef load(name, rel):\\n    spec=importlib.util.spec_from_file_location(name, root/rel)\\n    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod\\nload(\"legoesm.atmosphere.physics.microphysics.output\", \"legoesm/atmosphere/physics/microphysics/output.py\")\\nload(\"legoesm.atmosphere.physics.microphysics.config\", \"legoesm/atmosphere/physics/microphysics/config.py\")\\nload(\"legoesm.atmosphere.physics.microphysics._warm_rain\", \"legoesm/atmosphere/physics/microphysics/_warm_rain.py\")\\nth=load(\"legoesm.atmosphere.physics.microphysics.thompson\", \"legoesm/atmosphere/physics/microphysics/thompson.py\")\\nimport jax.numpy as jnp\\nfrom legoesm import constants\\nfrom legoesm.thermo import saturation_mixing_ratio_ice, saturation_mixing_ratio\\noutmod=sys.modules[\"legoesm.atmosphere.physics.microphysics.output\"]\\nT=jnp.array([[240.0]]); p_full=jnp.array([[5e4]]); p_half=jnp.array([[4.5e4,5.5e4]])\\nrho=p_full/(constants.R_d*T); dz=jnp.array([[500.0]])\\nh=outmod.make_zero_hydrometeors(1,1)\\nqsi=saturation_mixing_ratio_ice(T,p_full); qsl=saturation_mixing_ratio(T,p_full); q_v=1.2*qsi\\nout=th.thompson_microphysics(T,q_v,h,p_full,p_half,rho,dz,dt=60.0)\\nprint(\"qsi\", float(qsi[0,0]), \"qsl\", float(qsl[0,0]), \"qv\", float(q_v[0,0]))\\nprint(\"dq_i_dt\", float(out.dq_i_dt[0,0]), \"dN_i_dt\", float(out.dN_i_dt[0,0]), \"dq_v_dt\", float(out.dq_v_dt[0,0]), \"dT_dt\", float(out.dT_dt[0,0]))'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 1172ms:
qsi 0.0003409385681152344 qsl 0.0004698038101196289 qv 0.0004091263108421117
dq_i_dt 0.0 dN_i_dt 2733144.25 dq_v_dt 5.04111710597499e-07 dT_dt -0.0012549604289233685

codex
The static review exposed a broader microphysics problem outside the two fixes: the shared saturation adjustment returns a negative “condensation” tendency in subsaturated clear air, so it can create negative cloud water without a cloud-water donor. I’m checking whether that is intentionally guarded elsewhere.
exec
/bin/zsh -lc "sed -n '1,90p' tests/atmosphere/hydrostatic/unit/test_microphysics.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/_warm_rain.py | sed -n '52,92p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
"""Unit tests for the microphysics module.

Tests all 6 backends (Kessler, Sundqvist, Seifert-Beheng, Morrison,
Thompson, ML emulator) and the integration bridge for hydrostatic
and non-hydrostatic dycores.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
    KesslerConfig,
    SundqvistConfig,
    SeifertBehengConfig,
    MorrisonConfig,
    ThompsonConfig,
    MLEmulatorConfig,
)
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    make_zero_hydrometeors,
    make_zero_output,
    sedimentation_tendency,
)
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.ml_emulator import (
    ml_microphysics,
    MicrophysicsEmulator,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.thermo import saturation_mixing_ratio
from legoesm import constants


# ======================================================================
# Test helpers
# ======================================================================

def _make_warm_columns(ncol=4, nlev=10):
    """Create warm, near-saturated columns for testing warm-rain schemes."""
    # Temperature profile: 290K at surface, 220K at top
    T = jnp.linspace(220.0, 290.0, nlev)[None, :].repeat(ncol, axis=0)
    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    # Near-saturated vapor
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = 0.95 * q_sat

    # Cloud and rain water at mid levels
    q_c = jnp.zeros((ncol, nlev))
    q_c = q_c.at[:, 3:7].set(1e-3)
    q_r = jnp.zeros((ncol, nlev))
    q_r = q_r.at[:, 5:9].set(1e-4)

    rho = p_full / (constants.R_d * T)
    dp = p_half[:, 1:] - p_half[:, :-1]
    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0))
    dz = jnp.abs(dz)

    hydrometeors = HydrometeorState(
        q_c=q_c, q_r=q_r,
        q_i=jnp.zeros((ncol, nlev)),
        q_s=jnp.zeros((ncol, nlev)),
        q_g=jnp.zeros((ncol, nlev)),
        N_c=jnp.full((ncol, nlev), 1e8),
        N_r=jnp.full((ncol, nlev), 1e4),
        N_i=jnp.zeros((ncol, nlev)),
    )

    return T, q_v, hydrometeors, p_full, p_half, rho, dz


def _make_cold_columns(ncol=4, nlev=10):
    """Create cold columns with ice and snow for testing mixed-phase schemes."""
    T = jnp.linspace(200.0, 265.0, nlev)[None, :].repeat(ncol, axis=0)
    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])


 succeeded in 0ms:
    52	def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0):
    53	    """Compute smooth saturation adjustment (condensation tendency).
    54	
    55	    Parameters
    56	    ----------
    57	    T : array (ncol, nlev)
    58	        Temperature [K].
    59	    q_v : array (ncol, nlev)
    60	        Water vapor mixing ratio [kg/kg].
    61	    p_full : array (ncol, nlev)
    62	        Pressure [Pa].
    63	    dt : float
    64	        Time step [s].
    65	    sharpness : float
    66	        Sigmoid sharpness for smooth condensation switch.
    67	
    68	    Returns
    69	    -------
    70	    condensation : array (ncol, nlev)
    71	        Condensation tendency [kg/kg/s].
    72	    q_sat : array (ncol, nlev)
    73	        Saturation mixing ratio [kg/kg].
    74	    """
    75	    q_sat = saturation_mixing_ratio(T, p_full)
    76	    excess = q_v - q_sat
    77	    cond_frac = jax.nn.sigmoid(sharpness * excess)
    78	    condensation = cond_frac * excess / dt
    79	    return condensation, q_sat
    80	
    81	
    82	def effective_Nc(N_c, Nc_0):
    83	    """Use config default cloud droplet number where N_c is zero.
    84	
    85	    Parameters
    86	    ----------
    87	    N_c : array
    88	        Cloud droplet number concentration [1/kg].
    89	    Nc_0 : float
    90	        Default cloud droplet number.
    91	
    92	    Returns

exec
/bin/zsh -lc "rg -n \"saturation_adjustment\\(|condensation =|dq_c_dt = condensation|jnp.maximum\\(.*condensation|clip\\(.*dq_c|q_c.*clip|positive|nonnegative|tracer.*clip|dtracers_dt\" src/legoesm tests/unit/test_physics_microphysics.py tests/atmosphere/hydrostatic/unit/test_microphysics.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
tests/atmosphere/hydrostatic/unit/test_microphysics.py:616:        assert tend.dtracers_dt.data.shape == state.tracers.data.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:627:        assert jnp.all(jnp.isfinite(tend.dtracers_dt.data))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:702:        assert tend.dtracers_dt.data.shape == (*shape_3d, 3)
tests/unit/test_physics_microphysics.py:624:    # (positive evap_coeff, positive subsaturation, p<1) ⇒ positive grad.
tests/unit/test_physics_microphysics.py:628:    assert float(jnp.max(g_evap_pos)) > 0.0, "rain_evaporation grad should be positive for subsaturated column"
src/legoesm/coupler/coupling_fields.py:49:    shflx: jax.Array             # [step-averaged] Sensible heat flux [W/m2] (positive up)
src/legoesm/coupler/coupling_fields.py:50:    lhflx: jax.Array             # [step-averaged] Latent heat flux [W/m2] (positive up)
src/legoesm/coupler/coupling_fields.py:56:    co2_flux: jax.Array          # CO2 flux [kg/m2/s] (positive up)
src/legoesm/diagnostics/energy_budget.py:9:where R_TOA is the net TOA radiation (positive downward) and dE/dt
src/legoesm/diagnostics/energy_budget.py:189:    """Net TOA radiation (positive downward).
src/legoesm/diagnostics/energy_budget.py:219:    """Net surface radiation (positive into surface).
src/legoesm/diagnostics/energy_budget.py:232:    """Net surface energy flux into the atmosphere (positive upward).
src/legoesm/diagnostics/energy_budget.py:235:    (Radiation into surface is positive down, fluxes into atm are positive up.)
src/legoesm/diagnostics/energy_budget.py:254:    toa_net: float          # R_TOA: net TOA radiation (positive down)
src/legoesm/coupler/surface_energy.py:42:        Net longwave flux at surface [W/m2] (positive = warming surface).
src/legoesm/core/state.py:209:    dtracers_dt: Field
src/legoesm/core/state.py:273:    dtracers_dt: Field
src/legoesm/core/state.py:361:    dtracers_dt: Field
src/legoesm/io/cmor_output.py:596:            "positive": "down",
src/legoesm/io/cmor_output.py:612:            "positive": "down",
src/legoesm/atmosphere/physics/radiation/gray.py:276:    the layer has gained energy (positive heating).
src/legoesm/grids/gaussian.py:751:    for any positive order.
src/legoesm/training/trainable_params.py:4:with constraint transforms (softplus for positive, sigmoid for bounded).
src/legoesm/training/trainable_params.py:24:    transform: str  # "softplus" (positive), "sigmoid" (bounded), "none"
src/legoesm/atmosphere/physics/combined.py:325:                dtracers_dt=Field(data=jnp.zeros_like(state.tracers.data), name="dtracers_dt_phys", dims=dims_tr, units="1/s"),
src/legoesm/atmosphere/physics/combined.py:350:        dtracers_dt = first.dtracers_dt.data
src/legoesm/atmosphere/physics/combined.py:373:            dtracers_dt = dtracers_dt + t.dtracers_dt.data
src/legoesm/atmosphere/physics/combined.py:382:            dtracers_dt=first.dtracers_dt.replace(data=dtracers_dt),
src/legoesm/grids/vertical.py:1106:        dz[k] = z_half[k] - z_half[k+1] (positive since top-to-bottom).
src/legoesm/grids/vertical.py:1279:    dz = z_half[:-1] - z_half[1:]  # (nlev,) positive
src/legoesm/grids/vertical.py:1280:    dz_half = z_full[:-1] - z_full[1:]  # (nlev-1,) positive
src/legoesm/coupler/bulk_flux.py:144:        Sensible heat flux [W/m²] (positive upward = surface warmer).
src/legoesm/coupler/bulk_flux.py:146:        Latent heat flux [W/m²] (positive upward = surface moister).
src/legoesm/coupler/bulk_flux.py:332:        Sensible heat flux [W/m2] (positive upward = surface warmer).
src/legoesm/coupler/bulk_flux.py:334:        Latent heat flux [W/m2] (positive upward = surface moister).
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:125:        # negative ⇒ accel positive ⇒ wave accelerates the resolved flow).
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:149:    # Column dissipation (positive-definite: KE lost by the mean flow)
src/legoesm/forcing/external.py:1457:                    f"Interpolated spectral forcing has non-positive sum at day={day}",
src/legoesm/grids/voronoi.py:855:    edgeSignOnVertex(k, v) = +1 if the edge contributes positively to
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:85:    # Column dissipation (positive-definite: KE lost by the mean flow)
src/legoesm/grids/duogrid.py:465:                    # Sign: positive if si < i_tgt (source is to the left)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:341:                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_gwd", dims=dims_tr, units="1/s"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:410:            dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_gwd", dims=dims_tr, units="1/s"),
src/legoesm/core/operators_voronoi.py:827:    returns ``∇²(∇²f)`` (positive-definite eigenvalues), so the physical
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:151:    # Column dissipation (positive-definite: KE lost by the mean flow)
src/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:213:    # inherently nonnegative.
src/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:217:    q_c = jnp.clip(q_c, 0.0, None)
src/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:640:      # dp = p_half[k+1] - p_half[k] is positive (p increases toward surface).
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:166:    # Column dissipation (positive-definite: KE lost by the mean flow)
src/legoesm/core/fv_tp_2d.py:48:    """FV3 pert_ppm iv=0: positive definite constraint (tp_core.F90:1169-1192).
src/legoesm/core/fv_tp_2d.py:51:    cell mean ``q`` is positive.  When ``q <= 0``, zeroes the reconstruction.
src/legoesm/core/fv_tp_2d.py:75:    both_positive = (br > 0.0) & (bl > 0.0)
src/legoesm/core/fv_tp_2d.py:76:    da1_positive = da1 > 0.0
src/legoesm/core/fv_tp_2d.py:78:    bl_fix = jnp.where(both_positive, zero,
src/legoesm/core/fv_tp_2d.py:79:                       jnp.where(da1_positive, bl, -2.0 * br))
src/legoesm/core/fv_tp_2d.py:80:    br_fix = jnp.where(both_positive, zero,
src/legoesm/core/fv_tp_2d.py:81:                       jnp.where(da1_positive, -2.0 * bl, br))
src/legoesm/core/fv_tp_2d.py:207:    # Simple PPM reconstruction (bl = al - q, br = al - q) with positive-
src/legoesm/core/fv_tp_2d.py:218:    # pert_ppm(iv=0): positive definite constraint (tp_core.F90:610)
src/legoesm/core/fv_tp_2d.py:498:    # positive values to conserve total mass.  This compensates for
src/legoesm/core/fv_tp_2d.py:504:        # Step 2: scale positive values to match target mass
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:119:    # Column dissipation (positive-definite: KE lost by the mean flow)
src/legoesm/atmosphere/physics/radiation/rrtmgp/interpolation.py:201:  positive velocities) and the right-biased interpolation is labeled '-' (e.g.,
src/legoesm/atmosphere/physics/radiation/rrtmgp/interpolation.py:324:  positive velocities) and the right-biased interpolation is labeled '-' (e.g.,
src/legoesm/da/gen_be.py:70:        Vertical EOF eigenvalues (positive; covariance units).
src/legoesm/ml/s2s/sfno_slab/preparation.py:227:        raise ValueError(f"chunk_days must be positive, got {chunk_days}")
src/legoesm/atmosphere/physics/radiation/rrtmgp/rte/monochromatic_two_stream.py:353:  # Constrain reflectance and transmittance to be positive and to not go above
src/legoesm/atmosphere/physics/thermodynamics.py:69:    """Clip thermodynamic state to physically positive ranges.
src/legoesm/atmosphere/physics/thermodynamics.py:75:        positive finite floors for robust EOS/exner evaluations.
src/legoesm/atmosphere/physics/thermodynamics.py:139:    # Layer thicknesses are positive with top-to-bottom level indexing.
src/legoesm/atmosphere/physics/turbulence/vertical_diffusion.py:64:    # Build tridiagonal coefficients (all positive).
src/legoesm/atmosphere/physics/turbulence/surface_layer.py:96:    # Sensible heat flux (positive upward = surface warmer than air)
src/legoesm/atmosphere/physics/turbulence/surface_layer.py:99:    # Latent heat flux (positive upward = surface moister than air)
src/legoesm/core/operators_cdgrid.py:885:    """Monotone tracer advection using PPM with local-bounds clipping.
src/legoesm/atmosphere/physics/turbulence/ysu.py:156:    # w_star only meaningful for unstable (positive wtheta)
src/legoesm/atmosphere/physics/microphysics/sundqvist.py:62:    condensation = (
src/legoesm/atmosphere/physics/microphysics/sundqvist.py:68:    P_auto = config.auto_rate * jnp.maximum(q_c + condensation * dt, 0.0)
src/legoesm/core/tracers.py:31:    positive_definite: bool = True
src/legoesm/core/tracers.py:151:def clip_positive_definite(
src/legoesm/core/tracers.py:155:    """Clip positive-definite tracers to zero floor.
src/legoesm/core/tracers.py:157:    Preserves non-positive-definite tracers unchanged.
src/legoesm/core/tracers.py:162:        if info.positive_definite:
src/legoesm/atmosphere/physics/radiation/integration.py:633:            q_cloud_col = jnp.clip(
src/legoesm/atmosphere/physics/radiation/integration.py:696:            dtracers_dt=Field(
src/legoesm/atmosphere/physics/radiation/integration.py:698:                name="dtracers_dt_rad",
src/legoesm/atmosphere/physics/microphysics/thompson.py:82:    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
src/legoesm/atmosphere/physics/microphysics/thompson.py:245:    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
src/legoesm/atmosphere/physics/microphysics/thompson.py:254:    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
src/legoesm/atmosphere/physics/clouds/cloud_fraction.py:146:        Computed as ``p_half[..., 1:] - p_half[..., :-1]`` (positive).
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:47:    positive = x > 0.0
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:48:    safe_x = jnp.where(positive, x, 1.0)
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:49:    return jnp.where(positive, safe_x ** p, 0.0)
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:52:def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0):
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:78:    condensation = cond_frac * excess / dt
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:128:    q_c_pos = jnp.clip(q_c, 0.0)
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:129:    x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:154:    return k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho * gamma_norm
src/legoesm/atmosphere/physics/turbulence/integration.py:340:                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_turb", dims=dims_tr, units="1/s"),
src/legoesm/atmosphere/physics/turbulence/integration.py:405:            dtracers_dt=Field(data=dtracers, name="dtracers_dt_turb", dims=dims_tr, units="1/s"),
src/legoesm/ml/s2s/neuralgcm_slab/ensemble.py:127:        positive = np.concatenate([np.abs(diffs), np.asarray([abs(wrap)], dtype=float)])
src/legoesm/ml/s2s/neuralgcm_slab/ensemble.py:129:        positive = np.abs(diffs)
src/legoesm/ml/s2s/neuralgcm_slab/ensemble.py:130:    return float(np.mean(positive)) if positive.size else 1.0
src/legoesm/atmosphere/physics/convection/tiedtke.py:51:    smooth_positive_part,
src/legoesm/atmosphere/physics/convection/tiedtke.py:175:    # (positive in moist columns, vanishing in dry ones).
src/legoesm/atmosphere/physics/convection/tiedtke.py:203:    mc_gate = smooth_positive_part(
src/legoesm/atmosphere/physics/convection/tiedtke.py:221:        * smooth_positive_part(cape - config.cape_threshold, config.cape_sharpness)
src/legoesm/atmosphere/physics/convection/tiedtke.py:328:        # [kg/(m²·s)] (positive part — cloud water is generated where
src/legoesm/atmosphere/physics/convection/tiedtke.py:350:        # over levels where it is positive).  Net column ∫(dq_v + dq_c)
src/legoesm/atmosphere/physics/microphysics/integration.py:346:                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_micro", dims=dims_tr, units="1/s"),
src/legoesm/atmosphere/physics/microphysics/integration.py:406:        # Map output fields -> dtracers_dt
src/legoesm/atmosphere/physics/microphysics/integration.py:425:            dtracers_dt=Field(data=dtracers, name="dtracers_dt_micro", dims=dims_tr, units="1/s"),
src/legoesm/atmosphere/physics/convection/integration.py:685:                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_conv", dims=dims_tr, units="1/s"),
src/legoesm/atmosphere/physics/convection/integration.py:871:            dtracers_dt=Field(data=dtracers, name="dtracers_dt_conv", dims=dims_tr, units="1/s"),
src/legoesm/atmosphere/physics/microphysics/morrison.py:75:    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
src/legoesm/atmosphere/physics/microphysics/morrison.py:194:    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
src/legoesm/atmosphere/physics/microphysics/morrison.py:199:    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
src/legoesm/atmosphere/physics/turbulence/smagorinsky.py:112:    # For u diffusion: surface_flux = tau_x (positive = upward flux of u)
src/legoesm/core/fv3_sw_core.py:1569:    # are SIGNED, so the positive-definite constraint would incorrectly zero
src/legoesm/core/fv3_sw_core.py:1609:    # cfl = c * rdy[j-1] (positive) or c * rdy[j] (negative)
src/legoesm/atmosphere/physics/convection/bechtold.py:54:    smooth_positive_part,
src/legoesm/atmosphere/physics/convection/bechtold.py:208:        * smooth_positive_part(cape_pbl - config.cape_threshold, config.cape_sharpness)
src/legoesm/atmosphere/physics/convection/emanuel.py:8:contributes to the upward mass flux (positive buoyancy) or detrains
src/legoesm/atmosphere/physics/convection/emanuel.py:53:    smooth_positive_part,
src/legoesm/atmosphere/physics/convection/emanuel.py:132:        * smooth_positive_part(cape - config.cape_threshold, config.cape_sharpness)
src/legoesm/atmosphere/physics/convection/emanuel.py:235:        # ``evap_rate`` from ``dq_c_conv_dt`` *at the BL*, then clipped
src/legoesm/atmosphere/physics/convection/emanuel.py:238:        # nothing, flipping the sign of column ``Q_v`` on CAPE-positive
src/legoesm/atmosphere/physics/convection/kuo.py:94:    # 2. Column moisture excess: positive part only (zero when subsaturated)
src/legoesm/atmosphere/physics/convection/kuo.py:123:    # ``implied_condensation = 0``, ``dq_v_dt = 0``, and
src/legoesm/atmosphere/physics/convection/kuo.py:156:    implied_condensation = dT_dt * constants.c_pd / constants.L_v  # (ncol, nlev)
src/legoesm/atmosphere/physics/convection/kuo.py:206:    local_cond = jnp.maximum(implied_condensation, 0.0)
src/legoesm/atmosphere/dynamics/tracer_transport_mpas.py:89:        Tendency pytree: dtracers_dt and dtime_dt = 1.0.
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:70:    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:102:    dq_c_dt = condensation - dq_c_au - dq_c_ac
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:104:    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:18:* ``(CAPE - threshold)+`` uses ``smooth_positive_part``.
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:59:    smooth_positive_part,
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:143:        * smooth_positive_part(
src/legoesm/atmosphere/physics/microphysics/kessler.py:80:    condensation = cond_frac * excess / dt  # [kg/kg/s]
src/legoesm/atmosphere/physics/_shared.py:304:        Tracer array (for shape of dtracers_dt).
src/legoesm/atmosphere/physics/_shared.py:327:        dtracers_dt=Field(data=jnp.zeros_like(tracers), name=f"dtracers_dt{sfx}", dims=dims_tr, units="1/s"),
src/legoesm/atmosphere/dynamics/tracer_transport.py:85:        Tendency pytree: dtracers_dt and dtime_dt = 1.0.
src/legoesm/atmosphere/physics/convection/_triggers.py:14:  approximations of step / max / positive-part / lowest-crossing-index
src/legoesm/atmosphere/physics/convection/_triggers.py:47:    "smooth_positive_part",
src/legoesm/atmosphere/physics/convection/_triggers.py:125:def smooth_positive_part(
src/legoesm/atmosphere/physics/convection/_triggers.py:150:        Strictly positive everywhere; tends to ``max(x, 0)`` as
src/legoesm/atmosphere/physics/convection/config.py:269:    fractions with positive buoyancy continue to ascend while the
src/legoesm/atmosphere/physics/convection/config.py:333:    # of column heating from a CAPE-positive sounding.
src/legoesm/atmosphere/physics/convection/config.py:339:    # produces a column-net moistening on CAPE-positive soundings —
src/legoesm/atmosphere/physics/convection/config.py:364:    qualitative behavior (positive in moist columns, zero in dry
src/legoesm/atmosphere/physics/convection/config.py:447:    # vapor in excess of ``RH_crit * q_sat``: positive in moist columns,
src/legoesm/atmosphere/physics/convection/_plume.py:205:    The buoyancy proxy is ``T_parcel_ma - T_env`` (positive where the
src/legoesm/atmosphere/physics/convection/_plume.py:207:    level where this turns from negative to positive going upward,
src/legoesm/atmosphere/physics/convection/_plume.py:209:    turns back from positive to negative.
src/legoesm/atmosphere/physics/convection/_plume.py:240:    # positive to negative buoyancy.  Implementation: apply the same
src/legoesm/atmosphere/physics/convection/_plume.py:300:    Convention check: for a positively-CAPE column where the parcel
src/legoesm/atmosphere/physics/convection/_plume.py:518:        # for this linear ODE — always positive, AD-safe everywhere,
src/legoesm/atmosphere/physics/convection/_plume.py:527:        # explicit Euler plus a hard nonnegative clip ... after
src/legoesm/atmosphere/physics/convection/mass_flux.py:240:    dq_c_conv_dt = delta_0 * M_profile * jnp.clip(q_c_u, 0.0, None) / rho_safe
src/legoesm/atmosphere/physics/convection/mass_flux.py:336:    q_c_u_undiluted = jnp.clip(q_v_sfc - q_sat_moist, 0.0, None)
src/legoesm/atmosphere/physics/convection/mass_flux.py:455:    q_c_u_undiluted = jnp.clip(q_v_sfc - q_sat_moist, 0.0, None)
src/legoesm/parallel/device_config.py:561:                "must be provided (positive integers)."
src/legoesm/atmosphere/physics/convection/dca.py:97:        dp_pair = p_below - p_upper  # pressure difference (positive)
src/legoesm/atmosphere/physics/convection/dca.py:109:        # Dimensionless instability: positive means superadiabatic
src/legoesm/core/weno.py:492:        Left-biased (positive velocity) reconstruction.
src/legoesm/atmosphere/dynamics/tracer_transport_latlon.py:126:    TracerState  — tendency pytree (dtracers_dt, dtime_dt=1).
src/legoesm/atmosphere/dynamics/compressible_euler_mpas.py:300:        dtracers_dt = _tracer_tendencies(
src/legoesm/atmosphere/dynamics/compressible_euler_mpas.py:304:        dtracers_dt = jnp.zeros_like(tracers)
src/legoesm/atmosphere/dynamics/compressible_euler_mpas.py:320:        dtracers_dt = dtracers_dt + physics_tendency.dtracers_dt.data
src/legoesm/atmosphere/dynamics/compressible_euler_mpas.py:333:        dtracers_dt=Field(data=dtracers_dt, name="dtracers_dt",
src/legoesm/atmosphere/dynamics/compressible_euler_mpas.py:592:                tracers=s.tracers.replace(data=tend.dtracers_dt.data),
src/legoesm/land/soil_thermal.py:159:    # Bottom BC: geothermal heat flux (Neumann, positive into soil)
src/legoesm/land/multilayer_land.py:220:    # G = SW_net + LW_net - SH - LH  (positive into soil)
src/legoesm/land/multilayer_land.py:240:    evap_rate_demand = lhflx / L_eff  # kg/m2/s, positive up
src/legoesm/land/multilayer_land.py:280:    flux_top = (precip_rain + melt_rate - evap_bare) / rho_w  # m/s, positive down
src/legoesm/land/slab_land.py:161:    # --- Net surface energy flux (positive = energy into soil) ---
src/legoesm/land/slab_land.py:185:    evap_rate = lhflx / L_eff  # kg/m2/s (positive = upward)
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:366:        dtracers_dt = horiz + vert
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:368:        dtracers_dt = jnp.zeros_like(tracers)
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:473:        dtracers_dt = dtracers_dt + physics_tendency.dtracers_dt.data
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:494:        dtracers_dt=Field(
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:495:            data=dtracers_dt, name="dtracers_dt", dims=dims_tr, units="1/s",
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:607:                tracers=s.tracers.replace(data=tend.dtracers_dt.data),
src/legoesm/ocean/bathymetry.py:195:        Ocean depth [m], positive downward. Land ≤ 0.
src/legoesm/ocean/bathymetry.py:248:        Ocean depth [m], positive downward. Land ≤ 0.
src/legoesm/ocean/bathymetry.py:411:        Ocean depth [m], positive downward (any shape).
src/legoesm/ocean/bathymetry.py:546:        Ocean depth [m], positive downward. 0 on land. Same shape as lat_deg.
src/legoesm/ocean/bathymetry.py:589:    # Convert elevation to depth (positive downward for ocean)
src/legoesm/ocean/bathymetry.py:594:        # Already positive-downward depth
src/legoesm/ocean/advection_som.py:104:        +1.0 for extraction from the RIGHT/NORTH/TOP edge (positive flow),
src/legoesm/ocean/advection_som.py:169:    opposite to the physical z-direction (positive = upward).  The flux
src/legoesm/ocean/advection_som.py:170:    sign is flipped in ``_som_z_sweep`` so that positive = downward
src/legoesm/ocean/advection_som.py:173:    helpers see a consistent "positive = rightward = downward" convention.
src/legoesm/ocean/advection_som.py:337:        SIGNED volume transport at interior x-faces, positive = eastward.
src/legoesm/ocean/advection_som.py:353:    # Donor for positive flow: cell j-1 (left of face)
src/legoesm/ocean/advection_som.py:413:    # Incoming from LEFT (positive flux at left face)
src/legoesm/ocean/advection_som.py:451:        positive = northward.  Face j sits between cell j (south) and cell j+1.
src/legoesm/ocean/advection_som.py:494:    # Donor for positive flow (northward): cell j-1 (south of face)
src/legoesm/ocean/advection_som.py:584:        SIGNED volume transport at interior z-faces, positive = upward.
src/legoesm/ocean/advection_som.py:599:    # For z, "positive" = upward.  Level 0 = surface (top), level nlev-1 = bottom.
src/legoesm/ocean/advection_som.py:604:    #   positive vf → flow from right to left → donor = below
src/legoesm/ocean/advection_som.py:606:    # But our generic helpers assume positive = rightward (south→north for y, etc).
src/legoesm/ocean/advection_som.py:607:    # For z: we want positive = "downward" (increasing index) to match x-convention.
src/legoesm/ocean/advection_som.py:608:    # FLIP the sign so positive = downward:
src/legoesm/ocean/advection_som.py:609:    vf_int = -vol_flux_z  # (n_lat, n_lon, nlev-1), positive = downward
src/legoesm/ocean/advection_som.py:623:    # Donor for positive (downward) flow: level k-1 (above)
src/legoesm/ocean/advection_som.py:632:    is_pos = vf_int >= 0  # positive = downward in flipped convention
src/legoesm/ocean/advection_som.py:726:        Vertical velocity at half-levels [m/s], positive = upward.
src/legoesm/ocean/advection_som.py:786:    vol_flux_z = w_half[:, :, 1:-1] * area * dt  # (n_lat, n_lon, nlev-1), positive = upward
src/legoesm/land/snow_budget.py:49:        Net surface energy flux available for melting [W/m2], positive
src/legoesm/land/carbon/carbon_cycle.py:237:    co2_flux  : Net CO2 flux [kgCO2/m2/s], positive up.
src/legoesm/land/carbon/carbon_cycle.py:326:    # --- NEE: positive = source to atmosphere ------------------------------
src/legoesm/land/carbon/carbon_cycle.py:349:    co2_flux : kgCO2/m2/s, positive up (source to atmosphere).
src/legoesm/land/carbon/carbon_cycle.py:393:    co2_flux     : kgCO2/m2/s, positive up.
src/legoesm/ocean/state.py:38:        Bathymetry depth [m]. Static (positive downward). Shape (6, n, n).
src/legoesm/ocean/state.py:79:        Net surface heat flux (positive into ocean) [W/m²].
src/legoesm/ocean/state.py:216:        Bathymetry depth [m]. Static (positive downward). Shape (n_lat, n_lon).
src/legoesm/ocean/state.py:299:        Bathymetry depth [m]. Static (positive downward). Shape (n_lat, n_lon).
src/legoesm/ocean/simple_ocean.py:109:    # Bulk fluxes (positive upward)
src/legoesm/ocean/simple_ocean.py:158:    # Bulk fluxes (positive upward)
src/legoesm/ocean/simple_ocean.py:168:    # Vertical mixing flux (positive downward = heat from surface to deep)
src/legoesm/ocean/init_woa.py:57:        WOA depth levels [m] (positive downward).
src/legoesm/ocean/init_woa.py:65:    # Model full-level depths (positive), z_full_ref is negative
src/legoesm/land/soil_hydraulics.py:267:    h = jnp.abs(psi)  # suction head (positive)
src/legoesm/ocean/vertical.py:34:        Maximum ocean depth [m] (positive).
src/legoesm/ocean/vertical.py:112:    # Layer thicknesses (positive)
src/legoesm/ocean/vertical.py:113:    dz_ref = z_half_ref[:-1] - z_half_ref[1:]  # positive since z[k] > z[k+1]
src/legoesm/ocean/vertical.py:116:    dz_half_ref = z_full_ref[:-1] - z_full_ref[1:]  # positive
src/legoesm/ocean/vertical.py:439:    w positive = upward.
src/legoesm/ocean/advection.py:11:The DST-3 face value for positive flow at face j+1/2 (donor = cell j):
src/legoesm/ocean/advection.py:129:    # For face j: donor for positive flow is cell j-1, receiver is cell j
src/legoesm/ocean/advection.py:527:    # - positive flow → use right edge of cell j-1 = a_R[j-1] (in padded coords: a_R[j])
src/legoesm/ocean/advection.py:553:    #   positive flow → right-edge of left cell
src/legoesm/ocean/advection.py:1197:        vertical interface flux (positive = upward).
src/legoesm/ocean/advection.py:1215:    # Local positive / negative parts of the anti-diffusive face fluxes.
src/legoesm/ocean/experiments/baroclinic_gyre.py:168:    actual_depths = -np.asarray(z_coord.z_full_ref)  # positive-down depth [m]
src/legoesm/ocean/dynamics/advection_mpas.py:61:        Upwind-of-upwind cell index for positive (c1→c2) flow.
src/legoesm/ocean/dynamics/advection_mpas.py:131:        Upwind-of-upwind cell for positive flow (from compute_upup_cells).
src/legoesm/ocean/dynamics/ocean_model.py:133:        nonnegative = {
src/legoesm/ocean/dynamics/ocean_model.py:141:        for name, value in nonnegative.items():
src/legoesm/ocean/freshwater.py:11:- Evaporation is positive upward in the coupler, so E enters here
src/legoesm/ocean/freshwater.py:12:  as a positive value that *removes* freshwater from the ocean.
src/legoesm/ocean/freshwater.py:32:    positive upward (i.e., freshwater leaving ocean).
src/legoesm/ocean/freshwater.py:39:        Evaporation rate [kg/m2/s], positive upward.
src/legoesm/ocean/freshwater.py:43:        Ice melt/freeze freshwater [kg/m2/s], positive = melt.
src/legoesm/ocean/freshwater.py:75:    where P=precip, E=evaporation (positive up), R=runoff, M=ice melt.
src/legoesm/ocean/freshwater.py:84:        Net freshwater flux [kg/m2/s], positive into ocean.
src/legoesm/ocean/freshwater.py:160:        Latent heat flux [W/m2], positive upward.
src/legoesm/ocean/freshwater.py:201:    # Melting (mass decrease) puts freshwater into ocean (positive fw).
src/legoesm/ocean/init_mpas.py:37:        Bathymetry depth, positive downward.
src/legoesm/ocean/simple_ocean_mpas.py:79:    # Bulk fluxes (positive upward)
src/legoesm/ocean/dynamics/spectral_ocean_pe.py:711:        nonnegative = {
src/legoesm/ocean/dynamics/spectral_ocean_pe.py:719:        for name, value in nonnegative.items():
src/legoesm/ocean/dynamics/latlon_cgrid_operators.py:1201:        positive floor (1e-30) to avoid division by zero.
src/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:114:    product, positive-definite for ``coeff ≥ 0`` and ``H ≥ 0``.
src/legoesm/ocean/physics/vertical_mixing/kpp.py:62:    Returns shape (...) boundary layer depth [m, positive downward].
src/legoesm/ocean/physics/vertical_mixing/kpp.py:68:    # Depth of cell centers below surface (positive downward)
src/legoesm/ocean/physics/vertical_mixing/kpp.py:172:        Surface buoyancy flux [m^2/s^3], positive = destabilizing (convective).
src/legoesm/ocean/physics/vertical_mixing/kpp.py:180:        BL depth from the previous time step [m, positive downward].
src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:157:        Sign convention: positive = flow in +j (eastward) direction.
src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:190:        Sign convention: positive = flow in +i (northward) direction.
src/legoesm/ocean/physics/lateral_mixing/backscatter.py:247:    # explicit positive sign.
src/legoesm/ocean/physics/lateral_mixing/backscatter.py:270:    Sign convention: positive ``tend`` means the backscatter ADDS
src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py:69:        # Fluxes are positive upward; stress opposes wind
src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py:70:        tau_x = -tau_x  # flip to positive eastward
src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py:82:    # Net heat flux (positive into ocean)
src/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:136:    positive-definite (for ``coeff ≥ 0`` and ``H ≥ 0``).
src/legoesm/ocean/biogeochemistry/config.py:164:    flux_co2: jax.Array     # Air-sea CO2 flux [mol C/m^2/s], positive into ocean
src/legoesm/ocean/physics/lateral_mixing/gm_redi_mpas.py:504:    H_bathy : (nCells,) — bottom depth (positive).
src/legoesm/ocean/physics/vertical_mixing/implicit_solver.py:87:        Time step [s].  Must be positive.
src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:698:        Bottom depth (positive).
src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:215:        nonnegative = {
src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:224:        for name, value in nonnegative.items():

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/integration.py | sed -n '400,416p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   400	            )
   401	
   402	        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner).
   403	        dT_dt = micro_out.dT_dt.reshape(shape_3d)
   404	        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)
   405	
   406	        # Map output fields -> dtracers_dt
   407	        dtracers = jnp.zeros_like(tracers)
   408	        # Tracer mapping: 0=q_v, 1=q_c, 2=q_r, 3=q_i, 4=q_s, 5=q_g, 6=N_c, 7=N_r, 8=N_i
   409	        tend_fields = [
   410	            micro_out.dq_v_dt, micro_out.dq_c_dt, micro_out.dq_r_dt,
   411	            micro_out.dq_i_dt, micro_out.dq_s_dt, micro_out.dq_g_dt,
   412	            micro_out.dN_c_dt, micro_out.dN_r_dt, micro_out.dN_i_dt,
   413	        ]
   414	        for idx, field in enumerate(tend_fields):
   415	            if n_tracers > idx:
   416	                dtracers = dtracers.at[..., idx].set(field.reshape(shape_3d))

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/kessler.py | sed -n '65,93p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
    65	    Returns
    66	    -------
    67	    MicrophysicsOutput
    68	    """
    69	    ncol, nlev = T.shape
    70	    q_c = hydrometeors.q_c
    71	    q_r = hydrometeors.q_r
    72	    sharpness = config.saturation_sharpness
    73	
    74	    # Saturation mixing ratio
    75	    q_sat = saturation_mixing_ratio(T, p_full)
    76	
    77	    # 1. Saturation adjustment — convert from increment [kg/kg] to tendency [kg/kg/s]
    78	    excess = q_v - q_sat
    79	    cond_frac = jax.nn.sigmoid(sharpness * excess)
    80	    condensation = cond_frac * excess / dt  # [kg/kg/s]
    81	
    82	    dq_v_sat = -condensation
    83	    dq_c_sat = condensation
    84	
    85	    # 2. Autoconversion: cloud -> rain (threshold excess)
    86	    # dq_c_sat is a tendency [kg/kg/s]; multiply by dt to get increment [kg/kg]
    87	    q_c_updated = q_c + dq_c_sat * dt
    88	    autoconv = config.autoconversion_rate * jnp.maximum(
    89	        q_c_updated - config.autoconversion_threshold, 0.0
    90	    )
    91	
    92	    # 3. Accretion: cloud collected by rain.  Fractional powers of q_r
    93	    # have unbounded derivative at q_r=0 — safe_pow handles the AD guard.

codex
**Findings**

- **P1: subsaturated clear air can create negative cloud water.**  
  [\_warm_rain.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:75>) computes `condensation = sigmoid(...) * excess / dt`, so `q_v < q_sat` gives negative condensation even when `q_c = 0`. Thompson then adds this directly in [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:244>) / `dq_c_dt`, with no cloud-water donor clamp for evaporation. Kessler duplicates the same pattern in [kessler.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/kessler.py:78>). This is not introduced by the two fixes, but it is a missed microphysics bug.

- **P2: ice nucleation creates `N_i` without any ice-mass seed, so pure deposition can never start from `q_i = 0`.**  
  In [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:110>), `dN_i_nuc > 0`, but `dq_i_dep` is proportional to `q_i` at [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:119>). With `q_i = 0`, the scheme can produce positive `N_i` and zero `q_i` indefinitely. Morrison has the same structure at [morrison.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/morrison.py:89>). The `dN_i_dt = ... / clip(q_i, 1e-15)` form avoids NaNs, but it does not enforce number-mass consistency near zero ice mass.

- **P3: Thompson graupel activation still uses pre-clamp riming.**  
  `graupel_frac` is computed from raw `total_riming` before `qc_scale` at [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:166>), while the actual returned riming tendencies are reduced later at [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:184>). This is mass-conservative, but the threshold decision can say “strong riming” based on a rate the limiter no longer returns.

**Fix Verification**

The plume entrainment fix is mathematically correct. With `d = exp(-eps dz)`, total plume water before condensation is  
`q_t_ent = d * (q_u_prev + q_c_u_prev) + (1 - d) * q_e`  
assuming environmental cloud water is zero. The later condensation step only transfers `q_u_ent` to `q_c_u`, so `q_u + q_c_u = q_u_ent + q_c_u_ent`. I do not see a hidden mass leak in [\_plume.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:546>).

`q_u_ent` is bounded between `q_u_prev` and `q_e` for physical `eps >= 0` and `dz >= 0`. The exponential has well-defined gradients w.r.t. `eps` and `dz`; only extreme `eps*dz` can numerically underflow to a saturated zero-gradient regime. The existing `dz = maximum(..., 1.0)` kink is pre-existing.

The Thompson donor split is conservative: summing `dq_v + dq_c + dq_r + dq_i + dq_s + dq_g` cancels condensation, evaporation, deposition, Bergeron, riming, aggregation, melting, and `rime_to_graupel`; only `sed_r + sed_i + sed_s + sed_g` remains. The previous `q_i=0, q_s>0` graupel mass leak is addressed.

I could not run the pytest regressions because `pytest` is not installed in this shell.
2026-05-01T23:17:10.166119Z ERROR codex_core::session: failed to record rollout items: thread 019de5d0-74b2-7b60-ae2c-6093b346146e not found
tokens used
92,904
**Findings**

- **P1: subsaturated clear air can create negative cloud water.**  
  [\_warm_rain.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:75>) computes `condensation = sigmoid(...) * excess / dt`, so `q_v < q_sat` gives negative condensation even when `q_c = 0`. Thompson then adds this directly in [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:244>) / `dq_c_dt`, with no cloud-water donor clamp for evaporation. Kessler duplicates the same pattern in [kessler.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/kessler.py:78>). This is not introduced by the two fixes, but it is a missed microphysics bug.

- **P2: ice nucleation creates `N_i` without any ice-mass seed, so pure deposition can never start from `q_i = 0`.**  
  In [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:110>), `dN_i_nuc > 0`, but `dq_i_dep` is proportional to `q_i` at [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:119>). With `q_i = 0`, the scheme can produce positive `N_i` and zero `q_i` indefinitely. Morrison has the same structure at [morrison.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/morrison.py:89>). The `dN_i_dt = ... / clip(q_i, 1e-15)` form avoids NaNs, but it does not enforce number-mass consistency near zero ice mass.

- **P3: Thompson graupel activation still uses pre-clamp riming.**  
  `graupel_frac` is computed from raw `total_riming` before `qc_scale` at [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:166>), while the actual returned riming tendencies are reduced later at [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:184>). This is mass-conservative, but the threshold decision can say “strong riming” based on a rate the limiter no longer returns.

**Fix Verification**

The plume entrainment fix is mathematically correct. With `d = exp(-eps dz)`, total plume water before condensation is  
`q_t_ent = d * (q_u_prev + q_c_u_prev) + (1 - d) * q_e`  
assuming environmental cloud water is zero. The later condensation step only transfers `q_u_ent` to `q_c_u`, so `q_u + q_c_u = q_u_ent + q_c_u_ent`. I do not see a hidden mass leak in [\_plume.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/convection/_plume.py:546>).

`q_u_ent` is bounded between `q_u_prev` and `q_e` for physical `eps >= 0` and `dz >= 0`. The exponential has well-defined gradients w.r.t. `eps` and `dz`; only extreme `eps*dz` can numerically underflow to a saturated zero-gradient regime. The existing `dz = maximum(..., 1.0)` kink is pre-existing.

The Thompson donor split is conservative: summing `dq_v + dq_c + dq_r + dq_i + dq_s + dq_g` cancels condensation, evaporation, deposition, Bergeron, riming, aggregation, melting, and `rime_to_graupel`; only `sed_r + sed_i + sed_s + sed_g` remains. The previous `q_i=0, q_s>0` graupel mass leak is addressed.

I could not run the pytest regressions because `pytest` is not installed in this shell.
