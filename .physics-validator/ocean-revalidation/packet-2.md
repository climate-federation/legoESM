# Ocean physics-validator REVALIDATION packet (round 2)

This is round 2 of the adversarial review. Round 1 (codex review-1.md)
identified 8 findings. Below are the fixes applied + remaining open items
for re-confirmation.

You are an independent adversarial physics reviewer. Confirm or rebut the
fixes below, and if rebutting, give a numerical or textual counter-example.
Find any NEW bugs introduced by the fixes. Do not re-flag the items that
were correctly accepted in round 1.

## Findings classified

| # | Finding | Status | Resolution |
|---|---------|--------|-----------|
| 1 | KPP non-local masking non-conservative (in_bl_full mask) | FIXED | Drop the in_bl_full mask; keep is_unstable_col gate. F_T already vanishes outside BL via G(sigma)=0. Column drift goes to 1.4e-18 (machine eps). Probe + regression test added. |
| 2 | Plume convection units [K/m] not [K/s] | FIXED | Multiply tendency by `cfg.w_plume_min` to give units [m/s × 1/m × K] = [K/s]. Existing plume tests retuned for new magnitude. |
| 3 | Visbeck NaN gradient through sqrt(max(N²,0)) | FIXED | Use `sqrt(max(N², 1e-30))` in `_gm_redi_common.py:154` and analogous fixes in `kpp.py:85`, `gm_redi_mpas.py:464`, `diagnostics.py:61`. |
| 4 | KPP MO stability sign error (`max(zeta, 0)` zero in stable) | FIXED | Stable suppression now uses `max(-zeta_kpp, 0)` since the local convention has `zeta_kpp < 0` for stable (`L_MO < 0` when `B_f < 0`). |
| 5 | KPP B_f=None proxy sign | FIXED | Removed the leading minus: `B_f = +g/rho_0 * K_bg * drho_dz_sfc`. Stable column now produces `B_f < 0` correctly. |
| 6 | KPP A_v interior uses K_bg instead of A_bg | FIXED | Build separate `A_interior = Ri_shear_curve + cfg.A_bg + K_conv` for momentum. |
| 7 | Surface forcing/KPP architectural inconsistency | NOT FIXED | Out of scope (multi-file architectural refactor required). Documented as residual risk. |
| 8 | `implicit_bottom_drag_factor` is explicit Euler | NOT FIXED | Documented as latent risk only when `dt*r/H ≥ 1`. Not exercised in any current test config. |

## Plus (this round): the original B_salt sign bug from packet-1

| # | Finding | Status | Resolution |
|---|---------|--------|-----------|
| O1 | `B_salt = -g*beta*Q_S` sign inverted in integration.py:144 | FIXED | Now `B_salt = +g*beta*Q_S`. Probe + regression test added. |

## Probes confirming each fix

```
diff_probe_kpp_bsalt.py        : B_salt sign for freshening / brine
diff_probe_kpp_proxy_sharp.py  : B_f proxy sign for stable column
diff_probe_visbeck_grad.py     : NaN gradient absent in unstable column
diff_probe_kpp_vt_grad.py      : NaN gradient absent at N²=0 layer
diff_probe_kpp_nonlocal_conservation.py: column drift < 1e-12
```

All probes pass after fixes.

## Tests added (regression guards)

```
tests/unit/test_corrections.py::TestKPP::test_b_salt_sign_freshening_is_stabilizing
tests/unit/test_corrections.py::TestKPP::test_b_f_proxy_sign_for_stable_column
tests/unit/test_corrections.py::TestKPP::test_w_s_suppressed_in_stable_conditions
tests/unit/test_corrections.py::TestKPP::test_a_v_uses_a_bg_not_k_bg
tests/unit/test_corrections.py::TestKPP::test_nonlocal_transport_conserves_column_tracer
tests/ocean/unit/test_visbeck_gm.py::TestGMRediWithVisbeck::test_visbeck_kappa_grad_finite_in_unstable_column
```

Each test has been verified to FAIL under its corresponding buggy form
and PASS under the fix.

## Diff summary (production source)

### `src/legoesm/ocean/physics/vertical_mixing/integration.py`
```diff
-            B_salt = -constants.g * beta * Q_sfc_S
+            B_salt = constants.g * beta * Q_sfc_S
```

### `src/legoesm/ocean/physics/vertical_mixing/kpp.py`
```diff
 (line 85) - V_t² sqrt(0)→NaN gradient floor
-    V_t2 = (cfg.Cv * jnp.sqrt(jnp.maximum(jnp.abs(N2_full), 0.0)) ...
+    V_t2 = (cfg.Cv * jnp.sqrt(jnp.maximum(jnp.abs(N2_full), 1e-30)) ...

 (line 207) - B_f=None proxy sign
-        B_f = -g / rho_0_ref * cfg.K_bg * drho_dz_sfc
+        B_f = g / rho_0_ref * cfg.K_bg * drho_dz_sfc

 (line 276) - MO stable suppression sign
-                  / jnp.maximum(1.0 + 5.0 * jnp.maximum(zeta_kpp, 0.0), 1.0))
+                  / jnp.maximum(1.0 + 5.0 * jnp.maximum(-zeta_kpp, 0.0), 1.0))

 (lines 300-321) - Separate K_interior / A_interior, drop in_bl_full mask on non-local
-    K_interior = cfg.K_0_shear * (1.0 - Ri_ratio**2) ** 3 + cfg.K_bg
-    K_interior = K_interior + K_conv
+    Ri_shear_curve = cfg.K_0_shear * (1.0 - Ri_ratio**2) ** 3
+    K_interior = Ri_shear_curve + cfg.K_bg + K_conv
+    A_interior = Ri_shear_curve + cfg.A_bg + K_conv
-    A_v = jnp.where(in_bl, K_bl_half + cfg.A_bg, K_interior)
+    A_v = jnp.where(in_bl, K_bl_half + cfg.A_bg, A_interior)

 (lines ~395-414) - Drop in_bl_full mask on non-local tendency
-    dT_nonlocal = jnp.where(in_bl_full & is_unstable_col[...], dT_nonlocal, 0.0)
+    dT_nonlocal = jnp.where(is_unstable_col[..., jnp.newaxis], dT_nonlocal, 0.0)
 (same for dS_nonlocal)
```

### `src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py`
```diff
-    N = jnp.sqrt(jnp.maximum(N2, 0.0))
+    N = jnp.sqrt(jnp.maximum(N2, 1e-30))
```

### `src/legoesm/ocean/physics/lateral_mixing/gm_redi_mpas.py`
```diff
-    S_mag_cell = jnp.sqrt(jnp.maximum(S_sq_cell, 0.0))
+    S_mag_cell = jnp.sqrt(jnp.maximum(S_sq_cell, 1e-30))
```

### `src/legoesm/ocean/diagnostics.py`
```diff
-    N = jnp.sqrt(jnp.maximum(N2, 0.0))
+    N = jnp.sqrt(jnp.maximum(N2, 1e-30))
```

### `src/legoesm/ocean/physics/convection/plume.py`
```diff
-        dT_k = cfg.alpha_plume * cfg.epsilon * (T_plume - T[..., k]) * active
-        dS_k = cfg.alpha_plume * cfg.epsilon * (S_plume - S[..., k]) * active
+        dT_k = (cfg.w_plume_min * cfg.alpha_plume * cfg.epsilon
+                * (T_plume - T[..., k]) * active)
+        dS_k = (cfg.w_plume_min * cfg.alpha_plume * cfg.epsilon
+                * (S_plume - S[..., k]) * active)
```

## Open / out-of-scope items

* Finding #7 (surface forcing/KPP architecture): the diagnostic outputs
  of `bulk_formula_surface_forcing` (tau_x, Q_net, ...) are discarded
  by the wrapper, then KPP independently reads `surface_forcing.tau_x`
  from the externally-supplied OceanSurfaceForcing struct.  When config
  uses the bulk_formulas / prescribed scheme, the OceanSurfaceForcing
  struct is typically still None or empty, so KPP falls back to its
  proxies — physically inconsistent.  Fix requires changes to the
  physics pipeline driver to feed the cfg-derived diagnostics into
  the surface_forcing pipeline.

* Finding #8 (`implicit_bottom_drag_factor` explicit Euler): only
  matters at `dt*r/H ≥ 1`.  Not exercised in any current test or
  production config.  Yellow flag.

* Plume convection: still has no compensating entrainment source
  tendency, so although units are now correct, mass/heat conservation
  in the plume interaction with environment isn't yet enforced.
  Codex flagged this in finding #2, but a full conservation-correct
  plume rewrite is out of scope for a unit fix.

* Pre-existing `test_longrun_conservation_with_fixer` failure (heat
  drift 3.46e-7 vs tolerance 1e-8): unaffected by this round, still
  outstanding.

## Source files attached (re-attach for review)

(Same set as packet-1 with the fixes applied.)
"""Factory for ocean vertical mixing physics."""

from __future__ import annotations

from typing import Callable

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import compute_ocean_rho as _compute_rho
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig


def make_vertical_mixing_physics(
    config: VerticalMixingConfig,
) -> Callable:
    """Create a vertical mixing physics function.

    Parameters
    ----------
    config : VerticalMixingConfig

    Returns
    -------
    Callable : physics_fn(state, grid, z_coord) -> OceanTendencies
    """
    scheme = config.scheme

    if scheme == "none":
        return _make_none()
    elif scheme == "constant":
        return _make_constant(config)
    elif scheme == "richardson":
        return _make_richardson(config)
    elif scheme == "kpp":
        return _make_kpp(config)
    else:
        raise ValueError(f"Unknown vertical mixing scheme: {scheme!r}")


def _make_none() -> Callable:
    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        return _zero_tendencies(state)
    return physics_fn


def _make_constant(config: VerticalMixingConfig) -> Callable:
    from legoesm.ocean.physics.vertical_mixing.constant import constant_vertical_mixing
    cfg = config.constant

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        out = constant_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            z_coord, J, cfg,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_richardson(config: VerticalMixingConfig) -> Callable:
    from legoesm.ocean.physics.vertical_mixing.richardson import richardson_vertical_mixing
    cfg = config.richardson

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        rho = _compute_rho(state, z_coord, J)
        out = richardson_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, z_coord, J, cfg,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_kpp(config: VerticalMixingConfig) -> Callable:
    from legoesm import constants
    from legoesm.ocean.eos import (
        rho_0 as _RHO_0,
        c_sw as _C_SW,
        thermal_expansion_coeff,
        haline_contraction_coeff,
    )
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    import jax.numpy as jnp

    cfg = config.kpp

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        rho = _compute_rho(state, z_coord, J)

        # Forward surface forcing into KPP.  KPP needs:
        #   tau_x, tau_y [Pa] for the friction velocity u_star
        #   B_f [m^2/s^3, +ve = unstable] from net heat + freshwater fluxes
        #   Q_sfc_T [K m/s] kinematic heat flux for non-local T transport
        #   Q_sfc_S [PSU m/s] kinematic salt flux for non-local S transport
        # All are derived from the OceanSurfaceForcing struct when
        # available; otherwise we fall through to the proxies inside
        # ``kpp_vertical_mixing`` so KPP still runs unforced.
        tau_x = getattr(surface_forcing, "tau_x", None) if surface_forcing else None
        tau_y = getattr(surface_forcing, "tau_y", None) if surface_forcing else None
        q_net = getattr(surface_forcing, "q_net", None) if surface_forcing else None
        fw    = getattr(surface_forcing, "freshwater", None) if surface_forcing else None

        # Surface kinematic heat flux: Q_T = q_net / (rho_0 * c_sw)  [K m/s]
        # KPP convention: positive Q_T heats the ocean.
        Q_sfc_T = None
        B_f = None
        if q_net is not None:
            Q_sfc_T = q_net / (_RHO_0 * _C_SW)
            # Surface thermal expansion at the top layer.
            T_sfc = state.T.data[..., 0]
            S_sfc = state.S.data[..., 0]
            p_sfc = jnp.zeros_like(T_sfc)
            alpha = thermal_expansion_coeff(T_sfc, S_sfc, p_sfc)
            # Buoyancy flux from heat: B_heat = g * alpha * Q_T  (positive
            # Q_T = warming = lighter water at top = stabilizing).  KPP
            # convention is B_f > 0 = unstable (cooling-driven), so we
            # keep the *negative* of the heat-driven contribution.
            B_f = -constants.g * alpha * Q_sfc_T

        # Surface kinematic salt flux from freshwater: Q_S = -S_sfc * F_fw
        # / rho_0  [PSU m/s].  Net P-E entering ocean (F_fw > 0) freshens
        # the surface, hence the negative sign.
        Q_sfc_S = None
        if fw is not None:
            S_sfc = state.S.data[..., 0]
            T_sfc = state.T.data[..., 0]
            p_sfc = jnp.zeros_like(T_sfc)
            beta = haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
            Q_sfc_S = -S_sfc * fw / _RHO_0
            # Salt-driven surface buoyancy flux (KPP convention,
            # B_f > 0 = unstable):
            #   B_f = -g*(alpha*Q_T - beta*Q_S) = -g*alpha*Q_T + g*beta*Q_S
            # so the salt contribution is +g*beta*Q_S, NOT -g*beta*Q_S.
            # Sanity check: freshening (fw>0) gives Q_sfc_S<0 (salt flux
            # INTO ocean is negative) → B_salt = +g*beta*(neg) < 0
            # (stabilizing, lighter water on top).  Brine rejection
            # (fw<0) gives Q_sfc_S>0 → B_salt > 0 (destabilizing).
            B_salt = constants.g * beta * Q_sfc_S
            B_f = B_salt if B_f is None else (B_f + B_salt)

        out = kpp_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, state.eta.data, z_coord, J, cfg,
            tau_x=tau_x, tau_y=tau_y, B_f=B_f,
            Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn



def _zero_tendencies(state):
    from legoesm.ocean.physics.combined import zero_ocean_tendencies
    return zero_ocean_tendencies(state)


def _wrap_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state):
    from legoesm.ocean.physics.combined import wrap_ocean_tendencies
    return wrap_ocean_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state)
"""LMD94-style K-Profile Parameterization (KPP).

Boundary-layer parameterization following Large, McWilliams & Doney (1994)
with:

- Bulk Richardson number BL-depth diagnosis with linear interpolation
  of the crossing depth between model levels.
- Turbulent velocity scales w_s(sigma) from surface forcing (u_star, B_f).
- Cubic shape function G(sigma) = sigma * (1 - sigma)^2.
- Non-local tracer transport for unstable (convective) conditions only.
- Interior mixing: Richardson-number dependent + convective instability
  enhancement for statically unstable layers below the BL.

The caller should provide surface wind stress and buoyancy flux when
available.  If tau_x/tau_y are None, a simplified u_star proxy from
surface speed is used.

References
----------
- Large, W. G., McWilliams, J. C., & Doney, S. C. (1994). Oceanic
  vertical mixing: A review and a model with a nonlocal boundary layer
  parameterization. Rev. Geophys., 32, 363-403.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import (
    wright_eos,
    compute_buoyancy_frequency,
    compute_hydrostatic_pressure,
    rho_0 as rho_0_ref,
)
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def _boundary_layer_depth(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    u_star: jnp.ndarray,
    B_f: jnp.ndarray,
    cfg: KPPConfig,
    g: float = constants.g,
    h_bl_prev: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Estimate boundary layer depth h via bulk Richardson number.

    Uses linear interpolation to find the depth where Ri_b crosses
    Ri_crit, rather than snapping to the nearest model level.

    Returns shape (...) boundary layer depth [m, positive downward].
    """
    eps = _EPS
    nlev = rho.shape[-1]

    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    # Depth of cell centers below surface (positive downward)
    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual

    # Density and velocity differences from surface
    delta_rho = rho - rho[..., :1]
    delta_u = u - u[..., :1]
    delta_v = v - v[..., :1]
    delta_V2 = delta_u**2 + delta_v**2

    # LMD94 Eq. 23: V_t^2 = Cv * sqrt(|N2|) / sqrt(c_s * epsilon) *
    #   max(Ri_crit * h - d, 0) * d / h
    # Uses h_bl from the previous time step to break the coupling.
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    N2_full = jnp.concatenate([N2[..., :1], N2], axis=-1)
    max_depth = z_depth[..., -1]
    h_est = max_depth if h_bl_prev is None else h_bl_prev
    h_safe = jnp.maximum(h_est[..., jnp.newaxis], eps)
    # ``sqrt(0)`` has an infinite backward derivative; combined with
    # ``maximum(., 0)`` whose VJP is zero on the masked side, JAX
    # produces ``0 * inf = NaN``.  Use a tiny positive floor instead
    # so the gradient is finite (very large) and gets multiplied by
    # zero through ``maximum`` to a well-defined zero.  Forward
    # error is at most ``sqrt(1e-30) ≈ 1e-15``, negligible.
    V_t2 = (cfg.Cv * jnp.sqrt(jnp.maximum(jnp.abs(N2_full), 1e-30))
            / jnp.sqrt(jnp.maximum(cfg.c_s * cfg.epsilon_lmd, eps))
            * jnp.maximum(cfg.Ri_crit * h_safe - z_depth, 0.0)
            * z_depth / h_safe)

    # Bulk Richardson number
    Ri_b = (g * delta_rho * z_depth) / (
        rho_0_ref * jnp.maximum(delta_V2 + V_t2, eps)
    )

    # --- Differentiable soft interpolation of crossing depth ---
    # Instead of argmax (non-differentiable), use a sigmoid-weighted
    # average over all levels.  Each level contributes a weight
    # proportional to how much Ri_b crosses Ri_crit there.
    #
    # Weight at level k = sigmoid(sharpness * (Ri_b[k] - Ri_crit))
    #                    - sigmoid(sharpness * (Ri_b[k-1] - Ri_crit))
    # This is ~1 at the crossing level and ~0 elsewhere.
    sharpness = cfg.crossing_sharpness
    sig = jax.nn.sigmoid(sharpness * (Ri_b - cfg.Ri_crit))  # (..., nlev)

    # Crossing weight: difference of adjacent sigmoid values.  ``jnp.pad``
    # along the trailing axis is one HLO op; the previous
    # ``concatenate([zeros_like(sig[..., :1]), sig[..., :-1]])`` allocated
    # a fresh zero buffer and concatenated.
    pad_axes = ((0, 0),) * (sig.ndim - 1)
    sig_prev = jnp.pad(sig[..., :-1], (*pad_axes, (1, 0)))
    w_cross = sig - sig_prev  # (..., nlev), peaks at crossing level
    w_cross = jnp.maximum(w_cross, 0.0)
    w_sum = jnp.sum(w_cross, axis=-1, keepdims=True)
    w_norm = w_cross / jnp.maximum(w_sum, eps)

    # Crossing-based depth estimate
    h_crossing = jnp.sum(w_norm * z_depth, axis=-1)  # (...)

    # Fallback for columns where Ri_b never crosses Ri_crit:
    # - If column is mostly unstable (sig ≈ 0): BL extends to full depth
    # - If column is mostly stable (sig ≈ 1): BL is one layer
    column_stability = jnp.mean(sig, axis=-1)  # 0 = all unstable, 1 = all stable
    max_depth = z_depth[..., -1]
    min_depth = dz_actual[..., 0]
    h_fallback = (1.0 - column_stability) * max_depth + column_stability * min_depth

    # Blend: use crossing depth when crossing signal is strong, fallback otherwise
    crossing_strength = w_sum[..., 0]
    blend = jax.nn.sigmoid(cfg.crossing_sharpness * (crossing_strength - cfg.crossing_threshold))
    h = blend * h_crossing + (1.0 - blend) * h_fallback

    # At least one layer thick
    h = jnp.maximum(h, dz_actual[..., 0])

    return h


def kpp_vertical_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    eta: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: KPPConfig,
    g: float = constants.g,
    tau_x: jnp.ndarray | None = None,
    tau_y: jnp.ndarray | None = None,
    B_f: jnp.ndarray | None = None,
    Q_sfc_T: jnp.ndarray | None = None,
    Q_sfc_S: jnp.ndarray | None = None,
    h_bl_prev: jnp.ndarray | None = None,
) -> VerticalMixingOutput:
    """Apply LMD94-style KPP vertical mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    eta : array (6, n, n)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : KPPConfig
    g : float
    tau_x, tau_y : array (6, n, n) or None
        Surface wind stress [Pa]. If None, a proxy from surface speed is used.
    B_f : array (6, n, n) or None
        Surface buoyancy flux [m^2/s^3], positive = destabilizing (convective).
        If None, estimated from surface density gradient.
    Q_sfc_T : array (6, n, n) or None
        Surface kinematic heat flux [K*m/s] for non-local transport (LMD94
        Eq. 19).  If None, falls back to diagnosed K_sfc * dT/dz proxy.
    Q_sfc_S : array (6, n, n) or None
        Surface kinematic salt flux [PSU*m/s]. Same convention as Q_sfc_T.
    h_bl_prev : array (6, n, n) or None
        BL depth from the previous time step [m, positive downward].
        Used to break the implicit V_t-h_bl coupling in the Ri_b diagnosis
        (LMD94 Eq. 23).  If None, uses the full column depth as estimate.

    Returns
    -------
    VerticalMixingOutput
    """
    eps = _EPS
    nlev = u.shape[-1]

    # --- Friction velocity ---
    if tau_x is not None and tau_y is not None:
        # Proper u_star from wind stress: u_star = sqrt(|tau| / rho_0)
        tau_mag = jnp.sqrt(tau_x**2 + tau_y**2 + eps)
        u_star = jnp.sqrt(tau_mag / rho_0_ref)
    else:
        # Simplified proxy: u_star ~ 0.01 * |U_surface|
        speed_sfc = jnp.sqrt(u[..., 0]**2 + v[..., 0]**2 + eps)
        u_star = jnp.maximum(speed_sfc * 0.01, 1e-4)

    # --- Surface buoyancy flux ---
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    dz_half0 = 0.5 * (dz_actual[..., 0] + dz_actual[..., 1])
    if B_f is None:
        # Diffusive proxy from near-surface density gradient.  The KPP
        # convention here is ``B_f > 0 = unstable``, so the proxy must
        # be POSITIVE when the surface layer is statically unstable
        # (i.e. ``rho[0] > rho[1]`` ⇒ ``drho_dz_sfc > 0`` ⇒
        # ``B_f > 0``).  An earlier version had a leading minus sign
        # which inverted the sign and made stable columns spuriously
        # trigger non-local transport.
        drho_dz_sfc = (rho[..., 0] - rho[..., 1]) / jnp.maximum(dz_half0, eps)
        B_f = g / rho_0_ref * cfg.K_bg * drho_dz_sfc  # simplified proxy

    # --- Boundary layer depth ---
    h_bl = _boundary_layer_depth(
        rho, u, v, z_coord, jacobian, u_star, B_f, cfg, g,
        h_bl_prev=h_bl_prev,
    )

    # --- Depth coordinate ---
    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual
    sigma = z_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)

    # --- Shape function G(sigma) = sigma * (1 - sigma)^2 ---
    sigma_clip = jnp.clip(sigma, 0.0, 1.0)
    G = sigma_clip * (1.0 - sigma_clip) ** 2

    # --- Turbulent velocity scale w_s(sigma) (LMD94 Appendix B) ---
    # w_s depends on stability (B_f) and depth d = sigma * h_bl
    d = sigma_clip * h_bl[..., jnp.newaxis]
    # Monin-Obukhov length: L_MO = u_star^3 / (kappa * B_f)
    # Use copysign(eps, B_f) to preserve the sign of B_f near zero,
    # preventing a stability classification flip (issue #168 bug 1).
    B_f_safe = jnp.where(
        jnp.abs(B_f[..., jnp.newaxis]) > eps,
        B_f[..., jnp.newaxis],
        jnp.copysign(eps, B_f[..., jnp.newaxis]),
    )
    L_MO = u_star[..., jnp.newaxis]**3 / (cfg.kappa_vk * B_f_safe)
    zeta_kpp = d / L_MO

    # LMD94 Appendix B turbulent velocity scales:
    # Stable (B_f <= 0): w_s = kappa * u_star / (1 + 5*zeta)
    # Unstable, weakly (epsilon*d < |L|): w_s = kappa * u_star * phi_m^{-1}
    #   where phi_m^{-1} = (1 - 16*zeta)^{1/4}
    # Unstable, strongly convective (epsilon*d > |L|):
    #   w_s = (kappa * (u_star^3 + c_b * kappa * (-B_f) * d))^{1/3}
    is_unstable = B_f[..., jnp.newaxis] > 0.0
    epsilon_lmd = cfg.epsilon_lmd

    # Weakly unstable: phi_m^{-1} formulation
    w_s_weak = (cfg.kappa_vk * u_star[..., jnp.newaxis]
                * jnp.power(jnp.maximum(1.0 + 16.0 * jnp.abs(zeta_kpp), 1.0), 0.25))

    # Strongly convective: includes convective velocity scale
    Bf_pos = jnp.maximum(B_f[..., jnp.newaxis], 0.0)
    w_s_conv = jnp.power(
        cfg.kappa_vk * (u_star[..., jnp.newaxis]**3
                        + cfg.c_b * cfg.kappa_vk * Bf_pos * d),
        1.0 / 3.0,
    )

    # Transition: use convective scale when epsilon*d > |L_MO|
    is_strongly_convective = epsilon_lmd * d > jnp.abs(L_MO)
    w_s_unstable = jnp.where(is_strongly_convective, w_s_conv, w_s_weak)

    # Stable suppression: ``phi_m = 1 + 5*|zeta|`` for |zeta| > 0 in the
    # classical Monin-Obukhov convention.  Under the sign convention
    # used here (B_f > 0 = unstable), ``L_MO = u*^3 / (kappa * B_f)``
    # is NEGATIVE for stable forcing, so ``zeta_kpp = d / L_MO < 0`` for
    # stable.  The previous form ``max(zeta_kpp, 0)`` always returned
    # zero in stable conditions and disabled the suppression entirely.
    # Use ``max(-zeta_kpp, 0)`` so the magnitude of zeta drives the
    # stable suppression (codex adversarial review iter-1, finding #4).
    w_s_stable = (cfg.kappa_vk * u_star[..., jnp.newaxis]
                  / jnp.maximum(1.0 + 5.0 * jnp.maximum(-zeta_kpp, 0.0), 1.0))
    w_s = jnp.where(is_unstable, w_s_unstable, w_s_stable)
    w_s = jnp.maximum(w_s, 1e-10)

    # --- BL diffusivity at full levels ---
    K_bl_full = h_bl[..., jnp.newaxis] * w_s * G
    K_bl_full = jnp.minimum(K_bl_full, cfg.K_max)

    # --- Interior mixing: Richardson-number dependent ---
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    du = u[..., :-1] - u[..., 1:]
    dv = v[..., :-1] - v[..., 1:]
    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)
    Ri_int = N2 / jnp.maximum(S2, eps)
    # LMD94 interior shear instability: K = K_0 * (1 - (Ri/Ri_0)^2)^3
    # for Ri < Ri_0, zero above.
    Ri_ratio = jnp.clip(Ri_int / cfg.Ri_0, 0.0, 1.0)
    Ri_shear_curve = cfg.K_0_shear * (1.0 - Ri_ratio**2) ** 3

    # Interior static instability: enhanced mixing where N2 < 0
    K_conv = jnp.where(N2 < cfg.Ri_conv, cfg.K_conv, 0.0)

    # Build separate interior floors for tracer (K_v) and momentum (A_v).
    # The shear-instability + convective enhancement is shared, but the
    # background floors differ (K_bg = 1e-5 for tracers, A_bg = 1e-4 for
    # momentum).  Previously both branches used cfg.K_bg, which dropped
    # the momentum interior viscosity by an order of magnitude (codex
    # adversarial review iter-1, finding #6).
    K_interior = Ri_shear_curve + cfg.K_bg + K_conv
    A_interior = Ri_shear_curve + cfg.A_bg + K_conv

    # --- K at interfaces (average of full level K_bl) ---
    K_bl_half = 0.5 * (K_bl_full[..., :-1] + K_bl_full[..., 1:])

    # sigma at interfaces
    z_half_depth = 0.5 * (z_depth[..., :-1] + z_depth[..., 1:])
    sigma_half = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
    in_bl = sigma_half < 1.0

    # Combine BL and interior.  Each branch uses its own background
    # floor (K_bg for tracers, A_bg for momentum) so the merged field
    # honors the configured background levels in BOTH the BL and the
    # interior.
    K_v = jnp.where(in_bl, K_bl_half + cfg.K_bg, K_interior)
    A_v = jnp.where(in_bl, K_bl_half + cfg.A_bg, A_interior)
    K_v = jnp.minimum(K_v, cfg.K_max)
    A_v = jnp.minimum(A_v, cfg.K_max)

    # --- Apply diffusion ---
    vel = jnp.stack([u, v], axis=0)
    vel_tend = jax.vmap(
        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, A_v),
        in_axes=0, out_axes=0,
    )(vel)

    tracers = jnp.stack([T, S], axis=0)
    tr_tend = jax.vmap(
        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, K_v),
        in_axes=0, out_axes=0,
    )(tracers)

    # --- Non-local flux for T, S (LMD94 Eq. 19) ---
    #
    # LMD94 defines a counter-gradient term:
    #   gamma_T(sigma) = C_s * Q_0 / (w_s(sigma) * h)   [K/m]
    # where Q_0 is the surface kinematic heat flux [K*m/s].
    #
    # The non-local tendency is  -d/dz(K_bl * gamma_T).
    # Substituting K_bl = h * w_s * G(sigma):
    #   K_bl * gamma_T = h * w_s * G * C_s * Q_0 / (w_s * h) = C_s * Q_0 * G(sigma)
    #
    # So the non-local tendency reduces to:
    #   dT/dt_nonlocal = -d/dz[ C_s * Q_0 * G(sigma) ]            [K/s]
    #
    # We discretize this as the vertical divergence of the non-local
    # flux F_nl = C_s * Q_0 * G(sigma) evaluated at interfaces.

    # Surface kinematic heat/salt flux for non-local transport (LMD94 Eq. 19).
    # Use the IMPOSED surface flux when available (from bulk formulas or
    # prescribed forcing).  Fall back to diagnosed K_sfc * dT/dz proxy
    # only when no external flux is provided (issue #168 bug 2).
    if Q_sfc_T is not None:
        Q_T = Q_sfc_T  # [K*m/s]
    else:
        dT_dz_sfc = (T[..., 0] - T[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
        K_sfc = K_bl_full[..., 0]
        Q_T = K_sfc * dT_dz_sfc

    if Q_sfc_S is not None:
        Q_S = Q_sfc_S  # [PSU*m/s]
    else:
        dS_dz_sfc = (S[..., 0] - S[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
        K_sfc = K_bl_full[..., 0]
        Q_S = K_sfc * dS_dz_sfc

    # Only apply non-local transport for unstable (convective) columns.
    is_unstable_col = B_f > 0.0

    # G(sigma) at interior interfaces (half levels between full levels)
    sigma_half_full = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
    sigma_half_clip = jnp.clip(sigma_half_full, 0.0, 1.0)
    G_half = sigma_half_clip * (1.0 - sigma_half_clip) ** 2  # (..., nlev-1)

    in_bl_full = sigma < 1.0

    # --- Temperature non-local tendency ---
    # Non-local flux at interfaces: F_nl = C_s * Q_T * G_half  [K*m/s]
    # G_half = 0 for sigma_half >= 1 (outside BL) so F_T automatically
    # vanishes below the BL — the divergence ``-dF/dz`` is naturally
    # restricted to the BL.  The previous in-BL mask
    # ``in_bl_full & is_unstable_col`` zeroed the compensating
    # tendency in the layer whose CENTER sigma >= 1 but whose TOP
    # interface sigma_half < 1, breaking column conservation when h_bl
    # cut through a grid cell (codex adversarial review iter-1,
    # finding #1).  Keep only the column-level ``is_unstable_col``
    # gate.
    F_T = cfg.gamma_T * Q_T[..., jnp.newaxis] * G_half  # (..., nlev-1)
    # Tendency = -dF/dz at full levels (zero-flux BCs at surface and bottom)
    dT_nonlocal_top = -F_T[..., :1] / dz_actual[..., :1]
    dT_nonlocal_int = (F_T[..., :-1] - F_T[..., 1:]) / dz_actual[..., 1:-1]
    dT_nonlocal_bot = F_T[..., -1:] / dz_actual[..., -1:]
    dT_nonlocal = jnp.concatenate(
        [dT_nonlocal_top, dT_nonlocal_int, dT_nonlocal_bot], axis=-1
    )  # (..., nlev)  [K/s]
    dT_nonlocal = jnp.where(
        is_unstable_col[..., jnp.newaxis], dT_nonlocal, 0.0
    )

    # --- Salinity non-local tendency ---
    F_S = cfg.gamma_S * Q_S[..., jnp.newaxis] * G_half  # (..., nlev-1)
    dS_nonlocal_top = -F_S[..., :1] / dz_actual[..., :1]
    dS_nonlocal_int = (F_S[..., :-1] - F_S[..., 1:]) / dz_actual[..., 1:-1]
    dS_nonlocal_bot = F_S[..., -1:] / dz_actual[..., -1:]
    dS_nonlocal = jnp.concatenate(
        [dS_nonlocal_top, dS_nonlocal_int, dS_nonlocal_bot], axis=-1
    )  # (..., nlev)  [psu/s]
    dS_nonlocal = jnp.where(
        is_unstable_col[..., jnp.newaxis], dS_nonlocal, 0.0
    )

    return VerticalMixingOutput(
        du_dt=vel_tend[0],
        dv_dt=vel_tend[1],
        dT_dt=tr_tend[0] + dT_nonlocal,
        dS_dt=tr_tend[1] + dS_nonlocal,
        K_v=K_v,
        A_v=A_v,
    )
"""Grid-agnostic helpers shared by cubed-sphere and lat-lon GM/Redi.

Functions in this module operate on ``(..., nlev)`` arrays and make no
reference to a specific grid type.  Both ``gm_redi.py`` (cubed-sphere)
and ``gm_redi_latlon_cgrid.py`` (lat-lon C-grid) import from here.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import compute_buoyancy_frequency, rho_0 as _RHO_0_DEFAULT
from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)  # ~1.19e-7


# ---------------------------------------------------------------------------
# DM95 slope tapering
# ---------------------------------------------------------------------------

def dm95_taper(
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    S_max: float,
    eps: float = _EPS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply Danabasoglu & McWilliams (1995) smooth slope tapering.

    Returns tapered ``(S_x, S_y, taper)`` where *taper* is a smooth
    factor in [0, 1] computed as::

        taper = 0.5 * (1 + tanh((S_max - |S|) / (0.1 * S_max)))

    Parameters
    ----------
    S_x, S_y : array (..., nlev-1)
        Raw (clipped but un-tapered) isopycnal slopes at interfaces.
    S_max : float
        Maximum slope for tapering.
    eps : float
        Small constant for sqrt regularisation.

    Returns
    -------
    S_x_tapered, S_y_tapered, taper : same shapes as inputs.
    """
    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + eps)
    taper = 0.5 * (1.0 + jnp.tanh(
        (S_max - S_mag) / (0.1 * S_max + eps)
    ))
    return S_x * taper, S_y * taper, taper


def dm95_taper_scalar(
    S: jnp.ndarray,
    S_max: float,
    eps: float = _EPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-component variant of :func:`dm95_taper`.

    Used by grids that carry a scalar slope along each face's own normal
    (e.g. MPAS/Voronoi edges).  Identical functional form, with ``|S|``
    replaced by ``|S_n|``.

    Returns
    -------
    S_tapered, taper : same shape as ``S``.
    """
    taper = 0.5 * (1.0 + jnp.tanh(
        (S_max - jnp.abs(S)) / (0.1 * S_max + eps)
    ))
    return S * taper, taper


# ---------------------------------------------------------------------------
# Vertical flux divergence with zero-flux BCs
# ---------------------------------------------------------------------------

def vertical_flux_divergence(
    F_z: jnp.ndarray,
    dz_actual: jnp.ndarray,
    eps: float = _EPS,
) -> jnp.ndarray:
    """Compute vertical flux divergence at full levels.

    ``dq/dt[k] = (F_z[k-1/2] - F_z[k+1/2]) / dz[k]``

    with F_z = 0 at the surface and bottom boundaries (zero-flux BCs).

    Parameters
    ----------
    F_z : array (..., nlev-1)
        Vertical flux at interior interfaces.
    dz_actual : array (..., nlev)
        Layer thicknesses at full levels.

    Returns
    -------
    tendency : array (..., nlev)
    """
    # Single Pad HLO op (replaces alloc-zeros + concatenate of three).
    pad_axes = ((0, 0),) * (F_z.ndim - 1)
    F_z_ext = jnp.pad(F_z, (*pad_axes, (1, 1)))
    return (F_z_ext[..., :-1] - F_z_ext[..., 1:]) / jnp.maximum(dz_actual, eps)


# ---------------------------------------------------------------------------
# Visbeck (1997) adaptive GM coefficient
# ---------------------------------------------------------------------------

def compute_visbeck_kappa_gm(
    rho: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    cfg: VisbeckConfig,
    rho_ref: float = _RHO_0_DEFAULT,
) -> jnp.ndarray:
    """Visbeck (1997) adaptive GM coefficient.

    ``kappa(x, y) = alpha * L^2 * <N * |S|>_z`` with the depth-average
    weighted by the local interface thickness, optionally using the
    local first-baroclinic Rossby radius as the mixing length.

    Parameters
    ----------
    rho : (..., nlev) in-situ density.
    S_x, S_y : (..., nlev-1) tapered isopycnal slopes at interfaces.
    z_coord : OceanZStarCoordinate.
    jacobian : (...,) z* Jacobian at cell centres.
    f_coriolis : (...,) Coriolis parameter.
    cfg : VisbeckConfig.
    rho_ref : float
        Boussinesq reference density [kg/m^3].

    Returns
    -------
    kappa : (...,) horizontally-varying kappa_GM [m^2/s], clamped to
        the configured bounds.
    """
    eps = _EPS
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])

    # Local growth rate sigma_Eady ~ N * |S| at each interior interface.
    N2 = compute_buoyancy_frequency(
        rho, z_coord.dz_ref, jacobian, rho_ref=rho_ref, g=constants.g,
    )
    # Use a small positive floor on N² before sqrt, NOT a hard zero.
    # ``sqrt(0)`` has an infinite gradient in JAX; combined with the
    # ``maximum(N²,0)`` mask whose gradient is zero on the unstable
    # side, the backward pass evaluates ``inf * 0`` and produces NaN.
    # ``maximum(N², 1e-30)`` keeps N tiny but positive in unstable
    # layers, so ``sqrt`` has a finite (but very large) derivative
    # which is then multiplied by zero from ``maximum``'s VJP — a
    # well-defined zero rather than NaN.  Forward effect is at most
    # ``sqrt(1e-30) ≈ 1e-15``, negligible.
    N = jnp.sqrt(jnp.maximum(N2, 1e-30))
    # Regularise sqrt at zero slope — 1e-30 avoids spurious |S| ~ 3e-4
    # that the float32 eps (~1.19e-7) would produce.
    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + 1e-30)
    sigma = N * S_mag

    # Depth-weighted average of sigma_Eady.
    w_total = jnp.sum(dz_half, axis=-1)
    sigma_bar = jnp.sum(sigma * dz_half, axis=-1) / jnp.maximum(w_total, eps)

    # Mixing length L.
    if cfg.use_rossby_radius:
        N_bar = jnp.sum(N * dz_half, axis=-1) / jnp.maximum(w_total, eps)
        H_col = jnp.sum(dz_actual, axis=-1)
        f_safe = jnp.maximum(jnp.abs(f_coriolis), cfg.f_min)
        L = jnp.clip(N_bar * H_col / f_safe, cfg.L_min, cfg.L_max)
    else:
        L = jnp.full_like(sigma_bar, cfg.L_fixed)

    kappa = cfg.alpha * L ** 2 * sigma_bar
    return jnp.clip(kappa, cfg.kappa_min, cfg.kappa_max)
"""Entraining mass-flux convective plume parameterization."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import wright_eos
from legoesm.ocean.physics.convection.config import PlumeConfig
from legoesm.ocean.physics.convection.output import OceanConvectionOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def plume_convection(
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    p_hydro: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: PlumeConfig,
) -> OceanConvectionOutput:
    """Apply entraining mass-flux plume convection.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    p_hydro : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : PlumeConfig

    Returns
    -------
    OceanConvectionOutput
    """
    nlev = T.shape[-1]
    shape_3d = T.shape
    dtype = T.dtype
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]

    # Detect unstable surface: rho(k=0) > rho(k=1)
    surface_unstable = rho[..., 0] > rho[..., 1]  # (6, n, n)

    # Initialize plume properties at surface
    T_plume_init = T[..., 0] + cfg.T_excess
    S_plume_init = S[..., 0]

    # Descend plume using scan over levels (starting from level 1)
    def scan_fn(carry, k):
        T_plume, S_plume, active = carry
        dz_k = dz_actual[..., k]

        # Entrain environment.  ``1 - exp(-epsilon*dz)`` is the exact
        # solution of dT_plume/dz = -epsilon*(T_plume - T_env) over a
        # layer of thickness ``dz``.  The first-order linearization
        # ``epsilon*dz`` exceeds 1 and goes negative for thick layers
        # (e.g. epsilon=1e-3 m^-1, dz>1000 m), which would produce an
        # unphysical sign-flip on the plume properties.  ``-expm1(-x)``
        # is monotone in [0, 1) for x>=0 and gradient-friendly.
        entrain = -jnp.expm1(-cfg.epsilon * dz_k)
        T_plume = (1.0 - entrain) * T_plume + entrain * T[..., k]
        S_plume = (1.0 - entrain) * S_plume + entrain * S[..., k]

        # Buoyancy check
        rho_plume = wright_eos(T_plume, S_plume, p_hydro[..., k])
        delta_rho = rho_plume - rho[..., k]

        # Plume is active where it's denser than environment (sinking):
        # delta_rho > 0 means rho_plume > rho_env → plume sinks → stay active
        active = active * jax.nn.sigmoid(delta_rho * 1e4)

        # Detrainment tendency at this level [K/s], [PSU/s].
        #
        # The mass-flux plume formulation gives a tendency of the form
        #   dT/dt_env = w_p * alpha_plume * epsilon * (T_plume - T_env)
        # where ``w_p [m/s]`` is the plume vertical velocity, ``epsilon
        # [1/m]`` is the entrainment rate, and ``alpha_plume`` is a
        # dimensionless detrainment efficiency.  Without the ``w_p``
        # factor, the units would be [K/m] instead of [K/s] (codex
        # adversarial review iter-1, finding #2).  ``cfg.w_plume_min``
        # is used as the constant plume velocity (the minimum-floor
        # interpretation of an unresolved plume's effective speed).
        dT_k = (cfg.w_plume_min * cfg.alpha_plume * cfg.epsilon
                * (T_plume - T[..., k]) * active)
        dS_k = (cfg.w_plume_min * cfg.alpha_plume * cfg.epsilon
                * (S_plume - S[..., k]) * active)

        return (T_plume, S_plume, active), (dT_k, dS_k, active)

    init_active = surface_unstable.astype(dtype)
    (_, _, _), (dT_levels, dS_levels, active_levels) = jax.lax.scan(
        scan_fn,
        (T_plume_init, S_plume_init, init_active),
        jnp.arange(1, nlev),
    )

    # dT_levels shape: (nlev-1, 6, n, n) — move level axis to last,
    # then ``jnp.pad`` along the trailing axis instead of
    # ``zeros + .at[..., 1:].set(...)`` which materialises a fresh
    # zero buffer + scatter.  Single Pad HLO op each.
    dT_levels_t = jnp.moveaxis(dT_levels, 0, -1)  # (6, n, n, nlev-1)
    dS_levels_t = jnp.moveaxis(dS_levels, 0, -1)
    pad_axes = ((0, 0),) * (dT_levels_t.ndim - 1)
    dT_dt = jnp.pad(dT_levels_t, (*pad_axes, (1, 0)))
    dS_dt = jnp.pad(dS_levels_t, (*pad_axes, (1, 0)))

    # Convection flag at interfaces (average of adjacent levels' activity)
    active_t = jnp.moveaxis(active_levels, 0, -1)  # (6, n, n, nlev-1)
    flag = active_t

    return OceanConvectionOutput(
        dT_dt=dT_dt,
        dS_dt=dS_dt,
        convection_flag=flag,
    )
