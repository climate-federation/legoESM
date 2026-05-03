# Ocean physics-validator REVALIDATION packet (round 1)

This is an INDEPENDENT revalidation of the LegoESM ocean physics modules.
A previous physics-validator pass (commits `c798fb43`, `aaae5700`, `1e9e6b8c`)
fixed 8 bugs (5 surfaced by Codex). The current pass started with a fresh
audit of the recently-touched ocean physics-relevant files and the shared
helpers in `src/legoesm/ocean/dynamics/`.

You are an independent adversarial physics reviewer. Find every remaining
bug — sign error, unit inconsistency, broken-gradient pattern, conservation
violation, dtype hazard, MPI hazard, or physics-vs-implementation
disagreement — in the source attached below. Cite line numbers. If you
believe there are no further bugs, say so explicitly and explain why each
candidate concern is not one. Be rigorous: this is a Boussinesq, free-surface
ocean GCM with a Wright-1997 EOS, KPP boundary layer, GM/Redi lateral mixing,
plume + enhanced-diffusion convection, two-band shortwave penetration,
COARE/L-Y bulk flux, virtual-salt freshwater coupling, and a uniform-additive
conservation fixer.

## What this round changed

**Confirmed sign bug found and fixed**: KPP integration `B_salt` sign in
`src/legoesm/ocean/physics/vertical_mixing/integration.py:144`.

* Convention in this codebase: `B_f > 0 = unstable`.
* Standard MOM6/POP/NEMO formula:
  `B_f = -g*alpha*Q_T + g*beta*Q_S`
  where `Q_T`, `Q_S` are kinematic surface fluxes INTO the ocean.
* Code sets `Q_sfc_S = -S_sfc * fw / rho_0` with `fw > 0 = freshening`,
  so `Q_sfc_S < 0` for freshening (salt is being diluted out of the
  surface — kinematic salt flux INTO ocean is NEGATIVE).
* Old code: `B_salt = -constants.g * beta * Q_sfc_S`
  → For freshening: `B_salt > 0` → KPP marks the column UNSTABLE.
  But freshening is STABILIZING (lighter water on top).
* New code: `B_salt = +constants.g * beta * Q_sfc_S`
  → For freshening: `B_salt < 0` → STABILIZING ✓
  → For brine rejection: `B_salt > 0` → DESTABILIZING ✓

The author's stated intent in the comment ("freshening = stabilizing →
negative contribution to B_f") was correct, but the formula's sign was
inverted relative to that intent.

**Numerical confirmation** (probe in
`.physics-validator/ocean-revalidation/diff_probe_kpp_bsalt.py`):

```
beta             = 7.439420e-04  [1/PSU]
Q_S              = -3.414634e-07  [PSU m/s]   (< 0 for freshening)
B_salt (current) = 2.491049e-09  [m^2/s^3]   (should be < 0 = stabilizing)
B_salt (correct) = -2.491049e-09  [m^2/s^3]
```

**Regression test added**: `tests/unit/test_corrections.py::TestKPP::test_b_salt_sign_freshening_is_stabilizing`
exercises the full vertical-mixing factory and asserts brine rejection
mixes at least as hard as freshening. Verified to FAIL under the buggy
sign and PASS under the fixed sign.

## Other items to scrutinise (please confirm or rebut)

1. **`implicit_bottom_drag_factor`** at
   `src/legoesm/ocean/dynamics/ocean_tendency_common.py:250-279` returns
   `1 - dt*r/H`, which is forward-Euler, not backward-Euler. The function
   name says "implicit" but the math is explicit. For typical open-ocean
   parameters (`r=1.1e-3`, `dt=30 s`, `H=4000 m`), `dt*r/H ≈ 8e-6`
   so the difference vs `1/(1+dt*r/H)` is negligible. Worst case
   (`H=10 m`, `dt=300 s`): `dt*r/H = 0.033`; even there the difference
   is small. But if `dt*r/H >= 1` (very shallow + strong drag) the
   factor goes negative — a velocity sign-flip without warning. I left
   this as a documented yellow flag rather than fixing it (a drive-by
   change to `1 / (1 + dt*r/H)` would not be a measurable behavioural
   change in any current test). Disagree?

2. **KPP `cfg.Ri_conv`** is named like a Richardson-number threshold but
   compared against `N²` at `kpp.py:285`. Default value `0.0` makes
   the comparison `N² < 0` (statically unstable) → enhanced K. Function
   correct, name misleading. Not fixed (API change).

3. **KPP convective branch double-coverage**: when `N² < 0`, the interior
   shear-instability branch already gives `K = K_0_shear*(1-0)^3 + K_bg
   = 5e-3 + 1e-5`, then `K_conv = 1.0` is ADDED on top. The merged
   `K_v` is then `min(., K_max=1.0)`. This is functional (capped at
   `K_max`) but `K_conv` essentially overrides everything when
   convective.

4. **`B_salt = constants.g * beta * Q_sfc_S`** — fix verified by sign
   probe and by regression test. Are there other equivalent forms in
   the integration that I should re-check (e.g. inside `kpp.py:204-207`,
   the proxy when `B_f=None`)? That proxy `B_f = -g/rho_0 * K_bg *
   drho_dz_sfc` uses density gradient, with `drho_dz_sfc < 0` for stable
   stratification → `-g/rho_0 * K_bg * neg = +`. So `B_f > 0` for
   stable, which is wrong (should be < 0 for stable). The proxy
   ALWAYS labels stable columns as unstable when `B_f=None` — this is
   an additional latent bug. Confirm or rebut.

5. **`compute_visbeck_kappa_gm`** at `_gm_redi_common.py:114-174`: the
   Visbeck adaptive κ uses `N` (not `N²`) per the original Visbeck (1997)
   formula `κ = α*L²*<N|S|>`. Code uses `N = sqrt(max(N²,0))` at line
   154 — clamping to zero in unstable layers (`N²<0`). This is a
   reasonable choice (Eady growth rate is undefined for unstable),
   but it produces a zero-gradient region in unstable columns. AD-safe
   because `sqrt(max(N²,0))` is differentiable except exactly at
   `N²=0`. Comment?

6. **Wright EOS sign tests pass** (`alpha=2.47e-4 > 0` at 15°C/35 PSU,
   `beta>0`). Differentiability tests pass.

7. **`compute_filter_weights`** with `n_substeps=1` uses the box-filter
   fallback (was bug #4 in previous round, fixed). Does the formula
   `(1 + cos(2π*(i - n/2)/n))` for `n>=2` correctly normalise to a
   Hanning bell with peak at `i = n/2`? Yes: at `i = n/2`,
   `cos(0) = 1`, so weight = 2 (peak); at `i = 0` or `i = n`,
   `cos(±π) = -1`, weight = 0. ✓

## Source files attached

(see body below — full source for the modules touched + the shared
helpers).

## Pre-existing tech debt (NOT introduced by this round)

* `tests/ocean/unit/test_ocean.py::TestLongRunConservation::test_longrun_conservation_with_fixer`
  fails: heat drift `3.46e-7` vs tolerance `1e-8`. Pre-existing, conservation
  fixer can't deliver the tightened tolerance.
* `shortwave_penetration.py:64-65` and `biogeochemistry/{gas_exchange,carbonate}.py`
  carry function-default `rho_0=1025.0` / `c_sw=3994.0` literals — CLAUDE.md
  forbids function-default constants, but these are pre-existing.
* `ocean_tendency_common.implicit_bottom_drag_factor` is named
  "implicit" but uses an explicit form. Pre-existing (factored out of
  inline code in #214). Not changed.

# Source

```python
# === src/legoesm/ocean/physics/vertical_mixing/integration.py ===
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

```

```python
# === src/legoesm/ocean/physics/vertical_mixing/kpp.py ===
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
    V_t2 = (cfg.Cv * jnp.sqrt(jnp.maximum(jnp.abs(N2_full), 0.0))
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
        # Estimate from near-surface density gradient
        drho_dz_sfc = (rho[..., 0] - rho[..., 1]) / jnp.maximum(dz_half0, eps)
        B_f = -g / rho_0_ref * cfg.K_bg * drho_dz_sfc  # simplified proxy

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

    # Stable: standard suppression
    w_s_stable = (cfg.kappa_vk * u_star[..., jnp.newaxis]
                  / jnp.maximum(1.0 + 5.0 * jnp.maximum(zeta_kpp, 0.0), 1.0))
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
    K_interior = cfg.K_0_shear * (1.0 - Ri_ratio**2) ** 3 + cfg.K_bg

    # Interior static instability: enhanced mixing where N2 < 0
    K_conv = jnp.where(N2 < cfg.Ri_conv, cfg.K_conv, 0.0)
    K_interior = K_interior + K_conv

    # --- K at interfaces (average of full level K_bl) ---
    K_bl_half = 0.5 * (K_bl_full[..., :-1] + K_bl_full[..., 1:])

    # sigma at interfaces
    z_half_depth = 0.5 * (z_depth[..., :-1] + z_depth[..., 1:])
    sigma_half = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
    in_bl = sigma_half < 1.0

    # Combine BL and interior.
    # K_interior already includes ``+ cfg.K_bg`` at line ~282; the BL
    # branch needs the floor added explicitly so the merged field is
    # consistent.  Adding ``+ cfg.K_bg`` *after* the where would
    # double-count the floor on the interior branch.
    K_v = jnp.where(in_bl, K_bl_half + cfg.K_bg, K_interior)
    A_v = jnp.where(in_bl, K_bl_half + cfg.A_bg, K_interior)
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
    F_T = cfg.gamma_T * Q_T[..., jnp.newaxis] * G_half  # (..., nlev-1)
    # Tendency = -dF/dz at full levels (zero-flux BCs at surface and bottom)
    dT_nonlocal_top = -F_T[..., :1] / dz_actual[..., :1]
    dT_nonlocal_int = (F_T[..., :-1] - F_T[..., 1:]) / dz_actual[..., 1:-1]
    dT_nonlocal_bot = F_T[..., -1:] / dz_actual[..., -1:]
    dT_nonlocal = jnp.concatenate(
        [dT_nonlocal_top, dT_nonlocal_int, dT_nonlocal_bot], axis=-1
    )  # (..., nlev)  [K/s]
    dT_nonlocal = jnp.where(
        in_bl_full & is_unstable_col[..., jnp.newaxis], dT_nonlocal, 0.0
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
        in_bl_full & is_unstable_col[..., jnp.newaxis], dS_nonlocal, 0.0
    )

    return VerticalMixingOutput(
        du_dt=vel_tend[0],
        dv_dt=vel_tend[1],
        dT_dt=tr_tend[0] + dT_nonlocal,
        dS_dt=tr_tend[1] + dS_nonlocal,
        K_v=K_v,
        A_v=A_v,
    )

```

```python
# === src/legoesm/ocean/eos.py ===
"""Equation of state for seawater: Wright (1997) and linear.

Provides:
- ``wright_eos`` — nonlinear Wright (1997) EOS (MOM6 implementation)
- ``linear_eos`` — configurable linear EOS: ρ = ρ₀[1 - αT(T-Tref) + βS(S-Sref)]
- ``make_eos_fn`` — dispatcher returning an EOS callable based on config

Pure JAX functions, compatible with jit/grad/vmap.

Reference
---------
Wright, D. G. (1997): An Equation of State for Use in Ocean Models:
Ockham's Razor Revisited. J. Atmos. Oceanic Tech., 14(3), 735-740.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.precision import _resolve_dtype

# ==============================================================================
# Ocean constants
# ==============================================================================
rho_0 = 1025.0          # Reference seawater density [kg/m^3]
c_sw = 3994.0           # Specific heat of seawater [J/(kg*K)]
T_freeze_ocean = constants.T_freeze_ocean  # re-export from central constants
scale_depth = 1000.0     # Reference e-folding depth for stratification [m]

# ==============================================================================
# Wright (1997) EOS coefficients — from MOM6 (MOM_EOS_Wright.F90)
# Pressure units: Pa. Temperature: degC. Salinity: PSU.
#
# Formula: rho = (p + p0) / (lambda + al0 * (p + p0))
#   al0(T, S) = a0 + a1*T + a2*S
#   p0(T, S)  = (b0 + b4*S) + T*(b1 + T*(b2 + b3*T) + b5*S)
#   lambda(T, S) = (c0 + c4*S) + T*(c1 + T*(c2 + c3*T) + c5*S)
# ==============================================================================

# Specific volume coefficients al0(T, S)
_a0 = 7.057924e-4
_a1 = 3.480336e-7
_a2 = -1.112733e-7

# Pressure offset p0(T, S) [Pa]
_b0 = 5.790749e8
_b1 = 3.516535e6
_b2 = -4.002714e4
_b3 = 2.084372e2
_b4 = 5.944068e5
_b5 = -9.643486e3

# Lambda(T, S) [m^2/s^2]
_c0 = 1.704853e5
_c1 = 7.904722e2
_c2 = -7.984422
_c3 = 5.140652e-2
_c4 = -2.302158e2
_c5 = -3.079464


def wright_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    """Compute in-situ density from Wright (1997) EOS.

    Parameters
    ----------
    T : array
        Potential temperature [degC].
    S : array
        Salinity [PSU].
    p : array
        Pressure [Pa]. Use 0 for surface.

    Returns
    -------
    array : In-situ density [kg/m^3].

    Notes
    -----
    Intermediate computation is promoted to float64 to avoid precision
    loss from large polynomial coefficients (e.g., _b0 ~ 5.79e8).
    If ``JAX_ENABLE_X64=1`` is not set, the astype calls are no-ops
    (safe but no precision improvement).  ``jnp.astype`` is
    differentiable in JAX.

    The Wright (1997) polynomial is nominally valid for T in [-2, 40] degC
    and S in [0, 42] PSU, but extrapolates smoothly outside that box.
    Inputs are not clipped: silent clipping would zero gradients at the
    boundary and mask unphysical state from advection overshoots or
    coupler bugs. See issue #165.
    """
    orig_dtype = T.dtype

    # Promote to the EOS compute dtype (float64 in mixed mode) for
    # intermediate polynomial evaluation.  On backends that lack float64
    # (e.g. Metal), _resolve_dtype silently returns float32.
    hi = _resolve_dtype("equation_of_state", "compute")
    T = T.astype(hi)
    S = S.astype(hi)
    p = p.astype(hi)

    # Specific volume parameter
    al0 = _a0 + _a1 * T + _a2 * S

    # Pressure offset
    p0 = (_b0 + _b4 * S) + T * (_b1 + T * (_b2 + _b3 * T) + _b5 * S)

    # Lambda
    lam = (_c0 + _c4 * S) + T * (_c1 + T * (_c2 + _c3 * T) + _c5 * S)

    # Density: rho = (p + p0) / (lambda + al0 * (p + p0))
    p_plus_p0 = p + p0
    rho = p_plus_p0 / (lam + al0 * p_plus_p0)

    return rho.astype(orig_dtype)


def _wright_eos_scalar(T: float, S: float, p: float) -> float:
    """Scalar Wright EOS for JAX grad (no dtype promotion).

    Used internally by ``thermal_expansion_coeff`` and
    ``haline_contraction_coeff`` via ``jax.grad``.
    """
    al0 = _a0 + _a1 * T + _a2 * S
    p0 = (_b0 + _b4 * S) + T * (_b1 + T * (_b2 + _b3 * T) + _b5 * S)
    lam = (_c0 + _c4 * S) + T * (_c1 + T * (_c2 + _c3 * T) + _c5 * S)
    p_plus_p0 = p + p0
    return p_plus_p0 / (lam + al0 * p_plus_p0)


# Partial derivatives via JAX autodiff (scalar → vmap for arrays).
import jax
_drho_dT_scalar = jax.grad(_wright_eos_scalar, argnums=0)
_drho_dS_scalar = jax.grad(_wright_eos_scalar, argnums=1)


def thermal_expansion_coeff(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    r"""Thermal expansion coefficient α = -(1/ρ) ∂ρ/∂T.

    Parameters
    ----------
    T : array — Potential temperature [degC].
    S : array — Salinity [PSU].
    p : array — Pressure [Pa].

    Returns
    -------
    array : α [1/K], same shape as inputs.
    """
    hi = _resolve_dtype("equation_of_state", "compute")
    T64 = T.astype(hi)
    S64 = S.astype(hi)
    p64 = p.astype(hi)
    flat_T = T64.ravel()
    flat_S = S64.ravel()
    flat_p = p64.ravel()
    drho_dT = jax.vmap(_drho_dT_scalar)(flat_T, flat_S, flat_p).reshape(T.shape)
    rho = wright_eos(T, S, p)
    return (-drho_dT / rho).astype(T.dtype)


def haline_contraction_coeff(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    r"""Haline contraction coefficient β = (1/ρ) ∂ρ/∂S.

    Parameters
    ----------
    T : array — Potential temperature [degC].
    S : array — Salinity [PSU].
    p : array — Pressure [Pa].

    Returns
    -------
    array : β [1/PSU], same shape as inputs.
    """
    hi = _resolve_dtype("equation_of_state", "compute")
    T64 = T.astype(hi)
    S64 = S.astype(hi)
    p64 = p.astype(hi)
    flat_T = T64.ravel()
    flat_S = S64.ravel()
    flat_p = p64.ravel()
    drho_dS = jax.vmap(_drho_dS_scalar)(flat_T, flat_S, flat_p).reshape(T.shape)
    rho = wright_eos(T, S, p)
    return (drho_dS / rho).astype(T.dtype)


# ==============================================================================
# Linear equation of state
# ==============================================================================

class LinearEOSConfig(NamedTuple):
    """Configuration for the linear equation of state.

    ρ = rho_ref * [1 - alpha_T * (T - T_ref) + beta_S * (S - S_ref)]
    """
    rho_ref: float = 1025.0    # Reference density [kg/m³]
    alpha_T: float = 2.0e-4    # Thermal expansion coefficient [1/K]
    beta_S: float = 7.4e-4     # Haline contraction coefficient [1/PSU]
    T_ref: float = 10.0        # Reference temperature [°C]
    S_ref: float = 35.0        # Reference salinity [PSU]


def linear_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    rho_ref: float = 1025.0,
    alpha_T: float = 2.0e-4,
    beta_S: float = 7.4e-4,
    T_ref: float = 10.0,
    S_ref: float = 35.0,
) -> jnp.ndarray:
    """Compute density from a linear equation of state.

    ρ = rho_ref * [1 - alpha_T * (T - T_ref) + beta_S * (S - S_ref)]

    Parameters
    ----------
    T : array — Potential temperature [°C].
    S : array — Salinity [PSU].
    p : array — Pressure [Pa] (unused, accepted for API compatibility).
    rho_ref : float — Reference density [kg/m³].
    alpha_T : float — Thermal expansion coefficient [1/K].
    beta_S : float — Haline contraction coefficient [1/PSU].
    T_ref : float — Reference temperature [°C].
    S_ref : float — Reference salinity [PSU].

    Returns
    -------
    array : In-situ density [kg/m³].
    """
    return rho_ref * (1.0 - alpha_T * (T - T_ref) + beta_S * (S - S_ref))


def make_eos_fn(eos="wright", eos_linear=None):
    """Return an EOS callable ``fn(T, S, p) -> rho``.

    Parameters
    ----------
    eos : str
        ``"wright"`` (default) or ``"linear"``.
    eos_linear : LinearEOSConfig or None
        Parameters for linear EOS.  Ignored when *eos* is ``"wright"``.
        If ``None`` and *eos* is ``"linear"``, default parameters are used.

    Returns
    -------
    Callable[[array, array, array], array]
    """
    if eos == "wright":
        return wright_eos
    elif eos == "linear":
        cfg = eos_linear if eos_linear is not None else LinearEOSConfig()
        def _linear(T, S, p):
            return linear_eos(
                T, S, p,
                rho_ref=cfg.rho_ref, alpha_T=cfg.alpha_T,
                beta_S=cfg.beta_S, T_ref=cfg.T_ref, S_ref=cfg.S_ref,
            )
        return _linear
    else:
        raise ValueError(f"Unknown EOS scheme: {eos!r}")


def density_perturbation(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    rho_ref: float = rho_0,
) -> jnp.ndarray:
    """Compute density perturbation rho' = rho(T,S,p) - rho_ref.

    Parameters
    ----------
    T, S, p : array
        Temperature [degC], salinity [PSU], pressure [Pa].
    rho_ref : float
        Reference density [kg/m^3].

    Returns
    -------
    array : Density perturbation [kg/m^3].
    """
    return wright_eos(T, S, p) - rho_ref


def compute_hydrostatic_pressure(
    rho: jnp.ndarray,
    eta: jnp.ndarray,
    dz: jnp.ndarray,
    jacobian: jnp.ndarray,
    rho_ref: float = rho_0,
    g: float = constants.g,
) -> jnp.ndarray:
    """Compute hydrostatic pressure at full levels.

    p(z) = rho_ref * g * eta + integral_{z}^{0} rho * g dz'

    Integrated top-to-bottom (k=0 is surface, k=nlev-1 is deepest).
    Pressure at cell center is the cumulative integral from surface
    down to the midpoint of each layer.

    Parameters
    ----------
    rho : array
        In-situ density, shape (..., nlev).
    eta : array
        Sea surface height [m], shape (...).
    dz : array
        Reference layer thickness [m], shape (nlev,).
    jacobian : array
        Dynamic Jacobian (eta + H) / H, shape (...).
    rho_ref : float
        Reference density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    array : Hydrostatic pressure at full levels [Pa], shape (..., nlev).
    """
    # Surface pressure from free surface
    p_surface = rho_ref * g * eta  # (...,)

    # Actual layer thickness
    dz_actual = dz * jacobian[..., jnp.newaxis]  # (..., nlev)

    # Pressure increment per layer: rho * g * dz
    dp = rho * g * dz_actual  # (..., nlev)

    # Pressure at layer top = cumulative sum from surface
    # p_top[k] = p_surface + sum(dp[0:k])
    p_top = p_surface[..., jnp.newaxis] + jnp.cumsum(dp, axis=-1) - dp

    # Pressure at cell center = p_top + 0.5 * dp
    return p_top + 0.5 * dp


def compute_buoyancy_frequency(
    rho: jnp.ndarray,
    dz: jnp.ndarray,
    jacobian: jnp.ndarray,
    rho_ref: float = rho_0,
    g: float = constants.g,
) -> jnp.ndarray:
    """Compute Brunt-Vaisala frequency N^2.

    N^2 = -(g / rho_ref) * d(rho) / dz

    Computed at interior interfaces (nlev-1 values).

    Parameters
    ----------
    rho : array
        In-situ density, shape (..., nlev).
    dz : array
        Reference layer thickness [m], shape (nlev,).
    jacobian : array
        Dynamic Jacobian, shape (...).
    rho_ref : float
        Reference density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    array : N^2 at interior interfaces [1/s^2], shape (..., nlev-1).
    """
    dz_actual = dz * jacobian[..., jnp.newaxis]
    dz_interface = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])

    # drho/dz: rho[k] is shallower than rho[k+1]
    # N^2 = -(g/rho_0) * (rho[k] - rho[k+1]) / dz_interface
    drho_dz = (rho[..., :-1] - rho[..., 1:]) / dz_interface

    return -(g / rho_ref) * drho_dz


# ==============================================================================
# Shared helpers for ocean physics integration modules
# ==============================================================================

def compute_ocean_rho(state, z_coord, jacobian, eos_fn=None):
    """Compute in-situ density from ocean state.

    Used by vertical mixing, lateral mixing, and convection integration
    bridges. Avoids triplicating the same hydrostatic pressure + EOS call.

    Parameters
    ----------
    state : OceanState
        Must have .T, .S, .eta fields.
    z_coord : OceanZStarCoordinate
        Vertical coordinate with .dz_ref.
    jacobian : array
        Dynamic Jacobian (eta + H) / H.
    eos_fn : callable or None
        EOS function ``fn(T, S, p) -> rho``.  If None, uses ``wright_eos``.

    Returns
    -------
    array : In-situ density [kg/m^3].
    """
    if eos_fn is None:
        eos_fn = wright_eos
    # Two EOS iterations for density-pressure consistency, matching the
    # dynamical core (ocean_pe_cdgrid.py).
    rho = eos_fn(state.T.data, state.S.data, jnp.zeros_like(state.T.data))
    for _ in range(2):
        p_hydro = compute_hydrostatic_pressure(
            rho, state.eta.data, z_coord.dz_ref, jacobian, rho_0,
        )
        rho = eos_fn(state.T.data, state.S.data, p_hydro)
    return rho


def compute_ocean_rho_and_pressure(state, z_coord, jacobian, eos_fn=None):
    """Compute in-situ density and hydrostatic pressure from ocean state.

    Delegates to ``compute_ocean_rho`` for the 2-iteration EOS-pressure
    coupling, then computes a final hydrostatic pressure consistent with
    the converged density.

    Parameters
    ----------
    state, z_coord, jacobian : same as ``compute_ocean_rho``.
    eos_fn : callable or None
        EOS function. If None, uses ``wright_eos``.

    Returns
    -------
    rho : array — in-situ density [kg/m^3].
    p_hydro : array — hydrostatic pressure [Pa].
    """
    rho = compute_ocean_rho(state, z_coord, jacobian, eos_fn=eos_fn)
    p_hydro = compute_hydrostatic_pressure(
        rho, state.eta.data, z_coord.dz_ref, jacobian, rho_0,
    )
    return rho, p_hydro

```

```python
# === src/legoesm/ocean/dynamics/ocean_tendency_common.py ===
"""Grid-agnostic baroclinic tendency helpers shared by ocean_pe_*.py files.

This module factors logic that was previously duplicated across the
A-grid (``ocean_pe_cdgrid.py``), lat-lon C-grid
(``ocean_pe_latlon_cgrid.py``), and MPAS Voronoi (``ocean_pe_mpas.py``)
baroclinic tendency entry points.  Grid-specific operators (gradient,
divergence, vorticity, fill, ...) are passed in as callables so this
module never imports from a particular grid package.

Closes #214 (Phase 1).

Functions
---------
``iterate_eos_and_pressure_anomaly``
    EOS iteration (T_filled, S_filled, p) → ρ, ρ', p' using a
    reference-thickness hydrostatic integral.  The cubed-sphere caller
    can request high-precision arithmetic for the cumsum so that the
    halo-exchanged p' gradient stays clean.

``apply_sponge_tracer_relaxation``
    Linear restoring of T, S towards reference fields with rate
    ``γ``.  Used identically by the lat-lon C-grid and MPAS callers.

``apply_freshwater_virtual_salt_top``
    Top-layer salinity tendency from the freshwater volume flux.

``implicit_bottom_drag_factor``
    Returns ``1 - dt * r / max(H, eps)`` — the per-substep multiplicative
    factor used by both C-grid lat-lon and MPAS barotropic substeps.

These helpers are pure and pytree-friendly: they accept and return
``jax.Array`` values and never mutate inputs.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax.numpy as jnp

from legoesm.ocean.eos import compute_hydrostatic_pressure
from legoesm.ocean.freshwater import virtual_salt_flux


def iterate_eos_and_pressure_anomaly(
    T: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    fill_fn: Callable[[jnp.ndarray], jnp.ndarray],
    eos_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    dz_ref: jnp.ndarray,
    rho_0: float,
    g: float,
    *,
    n_iter: int = 2,
    hi_precision_pressure: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the standard 2-pass EOS iteration and form ``p_prime``.

    Replicates the identical iteration that previously lived inline in
    every ``ocean_pe_*.py`` file:

    1. Fill land-cell ``T``, ``S`` with ocean-neighbour values via
       ``fill_fn`` so the EOS does not produce spurious ``ρ'`` values
       on land that contaminate the gradient at coastlines.
    2. Iterate ``ρ ← EOS(T, S, p_hydro(ρ))`` ``n_iter`` times against
       the **reference** thickness profile ``dz_ref`` (i.e. ``J=1``,
       ``η=0``).  Using the actual Jacobian here would double-count
       the ``-g·∇η`` forcing handled by the barotropic solver.
    3. Build the layer-centred baroclinic pressure anomaly

       ``p'(k) = g · Σ_{j<k} ρ'(j) · dz_ref(j) + 0.5 · g · ρ'(k) · dz_ref(k)``

       which equals the half-trapezoidal cumulative integral of
       ``g·ρ'`` from the surface to the layer mid-point.

    Parameters
    ----------
    T, S : jax.Array
        Tracer fields with a trailing vertical axis (``..., nlev``).  All
        upstream callers store T and S with identical shapes.
    mask : jax.Array
        Land mask with the same horizontal shape as ``T[..., 0]`` (used
        only by ``fill_fn``; passed back to the caller for any post-
        processing it needs).
    fill_fn : Callable[[jax.Array], jax.Array]
        Grid-specific land-cell filler.  Must accept and return arrays
        with the same shape as ``T``.  Typical implementations:
        ``jax.vmap(fill_land_cells_cubed_sphere, in_axes=-1)``,
        ``_neumann_fill_cgrid`` (lat-lon C-grid), or
        ``fill_land_cells_mpas`` (MPAS).
    eos_fn : Callable
        Equation of state ``(T, S, p) → ρ``.  Same signature used by
        every grid (built via ``make_eos_fn``).
    dz_ref : jax.Array
        Reference layer thickness (``z_coord.dz_ref``), shape
        ``(nlev,)``.
    rho_0, g : float
    n_iter : int, default 2
        Number of EOS iterations *before* the final pressure update.
        All current callers use 2.
    hi_precision_pressure : bool, default False
        If True, perform the pressure cumsum in float64 so that
        halo-exchange interpolation errors do not contaminate the
        downstream gradient.  Used by the cubed-sphere C-D path; on
        lat-lon and MPAS the compact 2-cell stencils are well-behaved
        enough that the working precision is sufficient.

    Returns
    -------
    rho : jax.Array
        In-situ density (same shape as ``T``).
    rho_prime : jax.Array
        ``rho - rho_0`` (same shape as ``T``).
    p_prime : jax.Array
        Baroclinic pressure anomaly (same shape as ``T``).  Returned in
        whatever precision was used for the cumulative sum.
    """
    del mask  # currently unused (passed to fill_fn by the caller); kept
              # in signature for clarity at call sites.

    T_filled = fill_fn(T)
    S_filled = fill_fn(S)

    # Reference Jacobian (J=1, η=0).  Both have the horizontal shape of
    # T (i.e. no vertical axis).  Match dtype to the working state so we
    # never accidentally promote the EOS iteration to float64.
    horiz_shape = T.shape[:-1]
    J_ref = jnp.ones(horiz_shape, dtype=T.dtype)
    eta_ref = jnp.zeros(horiz_shape, dtype=T.dtype)

    rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T))
    for _ in range(n_iter):
        p_hydro = compute_hydrostatic_pressure(
            rho, eta_ref, dz_ref, J_ref, rho_0, g,
        )
        rho = eos_fn(T_filled, S_filled, p_hydro)

    rho_prime = rho - rho_0

    if hi_precision_pressure:
        rho_prime_hi = rho_prime.astype(jnp.float64)
        dz_hi = dz_ref.astype(jnp.float64)
        dp_layer = rho_prime_hi * g * dz_hi
    else:
        dp_layer = rho_prime * g * dz_ref

    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer

    return rho, rho_prime, p_prime


def apply_sponge_tracer_relaxation(
    dT_dt: jnp.ndarray,
    dS_dt: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    sponge,
    mask: Optional[jnp.ndarray] = None,
    *,
    expand_gamma_axis: int = -1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply tracer sponge relaxation ``+γ·(ref - q)``.

    Casts ``sponge`` arrays to ``T.dtype`` so the precision policy that
    holds the state in float32 is not silently promoted to float64
    (which previously crashed the barotropic scan; see latlon C-grid).

    Parameters
    ----------
    dT_dt, dS_dt : jax.Array
        Existing tendency arrays (modified by addition).
    T, S : jax.Array
        Current tracer state.
    sponge : SpongeForcing
        Must expose ``gamma``, ``T_ref``, ``S_ref``.  ``gamma`` is the
        relaxation rate per cell (1 / s), broadcast along the vertical
        axis via ``expand_gamma_axis``.
    mask : jax.Array, optional
        Ocean mask.  When provided, the relaxation tendency is
        multiplied by ``mask`` (with the same axis expansion as
        ``gamma``) so land cells stay quiescent.  When ``None`` no
        masking is applied (the caller masks downstream).
    expand_gamma_axis : int, default -1
        Axis on which to insert a singleton in ``sponge.gamma`` so that
        it broadcasts against the (..., nlev) tracer arrays.  Use ``-1``
        for both lat-lon C-grid (axis after lat/lon) and MPAS (axis
        after nCells).

    Returns
    -------
    (dT_dt_new, dS_dt_new) : tuple of jax.Array
    """
    dtype = T.dtype
    gamma = sponge.gamma.astype(dtype)
    gamma_b = jnp.expand_dims(gamma, expand_gamma_axis)
    dT = gamma_b * (sponge.T_ref.astype(dtype) - T)
    dS = gamma_b * (sponge.S_ref.astype(dtype) - S)
    if mask is not None:
        mask_b = jnp.expand_dims(mask, expand_gamma_axis)
        dT = dT * mask_b
        dS = dS * mask_b
    return dT_dt + dT, dS_dt + dS


def apply_freshwater_virtual_salt_top(
    dS_dt: jnp.ndarray,
    freshwater,
    S_ref: float,
    h_top: jnp.ndarray,
    rho_0: float,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Add the surface virtual-salt tendency to the top tracer level.

    Wraps ``freshwater.virtual_salt_flux`` and assigns the result to
    ``dS_dt[..., 0]`` (or the equivalent leading-axis slice for MPAS).
    All callers operate on cell-centred salinity, so the trailing
    ``nlev`` axis is the vertical axis.

    Parameters
    ----------
    dS_dt : jax.Array
        Salinity tendency, modified at index ``[..., 0]``.
    freshwater : FreshwaterForcing
    S_ref : float
        Reference salinity used for the virtual-flux closure.
    h_top : jax.Array
        Top-layer thickness with the same horizontal shape as ``mask``.
    rho_0 : float
    mask : jax.Array
        Ocean mask (1 = ocean) with the same horizontal shape as the
        leading axes of ``dS_dt``.

    Returns
    -------
    jax.Array
        ``dS_dt`` with the virtual-salt flux added to the top layer.
    """
    dS_top = virtual_salt_flux(freshwater, S_ref, h_top, rho_0)
    # Cast the freshwater contribution to dS_dt's dtype so the scatter
    # add does not silently widen on x64 mode (the freshwater struct
    # is built at JAX-default precision in init helpers, which can be
    # f64 while the salinity tendency runs at the storage policy's
    # f32).
    return dS_dt.at[..., 0].add((dS_top * mask).astype(dS_dt.dtype))


def implicit_bottom_drag_factor(
    dt: jnp.ndarray | float,
    drag_r: jnp.ndarray | float,
    H: jnp.ndarray,
    *,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """Per-substep bottom-drag multiplier ``1 - dt · r / max(H, eps)``.

    Used by both the lat-lon C-grid and MPAS barotropic substeps to
    apply a linear bottom drag on the depth-averaged velocity.  The
    floor on ``H`` prevents the drag from blowing up over very thin
    water columns (≈ inundation).

    Parameters
    ----------
    dt : float or jax.Array
        Substep size [s].
    drag_r : float or jax.Array
        Linear drag coefficient [m/s].
    H : jax.Array
        Total water column depth at the velocity location [m].
    eps : float
        Floor on ``H`` for numerical safety.

    Returns
    -------
    jax.Array, same shape as ``H``.
    """
    return 1.0 - dt * drag_r / jnp.maximum(H, eps)

```

```python
# === src/legoesm/ocean/dynamics/barotropic_common.py ===
"""Grid-agnostic helpers shared by barotropic_*.py substep solvers.

Factors small repeated patterns out of the four grid-specific
barotropic solvers (cubed-sphere A-grid, cubed-sphere C-grid, lat-lon
C-grid, MPAS).  Closes #214 (Phase 2).

The helpers are intentionally thin: each replaces a few lines that were
character-for-character identical across solvers, so future updates to
the BEBT scheme, the cosine time filter, or MAXVEL clipping touch one
file instead of three.

This module performs no halo exchange and does not depend on any
grid-specific operator package.  Callers are responsible for staging
field shapes correctly before invoking these helpers.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp


def compute_filter_weights(
    n_substeps: int,
    dtype: jnp.dtype,
    *,
    use_cosine: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return per-substep accumulator weights for time-averaging.

    The cosine bell (Hanning window) suppresses the side lobes of the
    plain box filter that alias barotropic modes into the baroclinic
    coupling.  Both the lat-lon C-grid and MPAS solvers compute the
    weights with the same formula:

        w_i = 1 + cos(2π · (i - n/2) / n)         (cosine)
        w_i = 1                                   (box)

    Parameters
    ----------
    n_substeps : int
        Number of barotropic substeps.
    dtype : jnp.dtype
        Working precision for the weight array.
    use_cosine : bool
        ``config.barotropic_time_filter == "cosine"``.

    Returns
    -------
    w_filter : jax.Array, shape (n_substeps,)
        Per-substep weight passed as ``xs`` to ``lax.scan`` (or
        indexed inside ``fori_loop``).
    w_total : jax.Array, scalar
        ``sum(w_filter)`` — used to normalise the eta / velocity
        accumulators.  Transport accumulators (``Hu``) keep using
        ``n_substeps`` for exact volume conservation.
    """
    i = jnp.arange(n_substeps, dtype=dtype)
    if use_cosine:
        # Hanning-window weights.  At ``n_substeps == 1`` (rare, only
        # used by tests / 1-substep spin-ups) the formula
        # ``1 + cos(2 pi (0 - 0.5)/1) = 1 + cos(-pi) = 0`` collapses to
        # zero, which then divides by zero in
        # ``eta_sum / w_total`` downstream.  Fall back to the box
        # filter when the cosine bell would degenerate (codex
        # adversarial review iter-1, bug #4).
        if n_substeps < 2:
            w_filter = jnp.ones(n_substeps, dtype=dtype)
        else:
            w_filter = 1.0 + jnp.cos(
                2.0 * jnp.pi * (i - 0.5 * n_substeps) / n_substeps,
            )
    else:
        w_filter = jnp.ones(n_substeps, dtype=dtype)
    return w_filter, jnp.sum(w_filter)


def bebt_blend(
    eta_new: jnp.ndarray,
    eta_old: jnp.ndarray,
    bebt: float | jnp.ndarray,
) -> jnp.ndarray:
    """Backward-Euler/Backward-time blend of new and old eta for the PGF.

    ``bebt = 0`` recovers the standard forward-backward scheme; the
    MOM6 default ``bebt = 0.2`` introduces semi-implicit damping of
    the fastest barotropic gravity waves (#205).
    """
    return (1.0 - bebt) * eta_new + bebt * eta_old


def maxvel_clip(field: jnp.ndarray, maxvel: float | jnp.ndarray) -> jnp.ndarray:
    """Symmetric clip of barotropic velocity components.

    Used to suppress runaway velocities at single grid points that
    would otherwise crash the solver before the substep finishes.
    """
    return jnp.clip(field, -maxvel, maxvel)

```

```python
# === src/legoesm/ocean/freshwater.py ===
"""Freshwater forcing for the MPAS ocean model.

Handles precipitation, evaporation, land runoff, and ice melt/freeze
freshwater fluxes. Applies virtual salt flux to salinity and real
freshwater mass flux to the free surface.

Conventions
-----------
- All fluxes in kg/m2/s (mass flux per unit area).
- Positive = freshwater entering ocean (precip, runoff, ice melt).
- Evaporation is positive upward in the coupler, so E enters here
  as a positive value that *removes* freshwater from the ocean.

References
----------
- Griffies, S. M. (2004). Fundamentals of Ocean Climate Models, Ch. 12.
- Large, W. G. et al. (1997). J. Phys. Oceanogr., 27(11), 2418-2447.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class FreshwaterForcing(NamedTuple):
    """Freshwater fluxes applied to the ocean surface.

    All fields have shape (nCells,) and units kg/m2/s.
    Positive = freshwater into ocean, except evaporation which is
    positive upward (i.e., freshwater leaving ocean).

    Fields
    ------
    precip : jax.Array
        Precipitation rate [kg/m2/s].
    evap : jax.Array
        Evaporation rate [kg/m2/s], positive upward.
    runoff : jax.Array
        Land runoff rate [kg/m2/s].
    ice_fw : jax.Array
        Ice melt/freeze freshwater [kg/m2/s], positive = melt.
    """
    precip: jnp.ndarray
    evap: jnp.ndarray
    runoff: jnp.ndarray
    ice_fw: jnp.ndarray


def zero_freshwater(nCells: int) -> FreshwaterForcing:
    """Create zero freshwater forcing.

    Parameters
    ----------
    nCells : int
        Number of Voronoi cells.

    Returns
    -------
    FreshwaterForcing
    """
    # Init helper: keep at the JAX default float dtype.  Callers running
    # under a non-default precision policy can ``cast_pytree`` the
    # result to match their state.
    z = jnp.zeros(nCells)
    return FreshwaterForcing(precip=z, evap=z, runoff=z, ice_fw=z)


def net_freshwater_flux(fw: FreshwaterForcing) -> jnp.ndarray:
    """Compute net freshwater flux into ocean [kg/m2/s].

    F_fw = P - E + R + M

    where P=precip, E=evaporation (positive up), R=runoff, M=ice melt.

    Parameters
    ----------
    fw : FreshwaterForcing

    Returns
    -------
    jax.Array, shape (nCells,)
        Net freshwater flux [kg/m2/s], positive into ocean.
    """
    return fw.precip - fw.evap + fw.runoff + fw.ice_fw


def freshwater_eta_tendency(fw: FreshwaterForcing, rho_0: float) -> jnp.ndarray:
    """Compute free-surface tendency from freshwater flux.

    deta/dt = F_fw / rho_0

    Parameters
    ----------
    fw : FreshwaterForcing
    rho_0 : float
        Reference seawater density [kg/m3].

    Returns
    -------
    jax.Array, shape (nCells,)
        Free-surface tendency [m/s].
    """
    return net_freshwater_flux(fw) / rho_0


def virtual_salt_flux(
    fw: FreshwaterForcing,
    S_ref: float,
    dz_0: jnp.ndarray,
    rho_0: float,
) -> jnp.ndarray:
    """Compute virtual salt flux for the top ocean layer.

    dS/dt = -S_ref * F_fw / (rho_0 * dz_0)

    This approximation maintains volume while adjusting salinity
    to account for freshwater dilution/concentration.

    Parameters
    ----------
    fw : FreshwaterForcing
    S_ref : float
        Reference salinity [PSU].
    dz_0 : jax.Array, shape (nCells,)
        Top layer thickness [m].
    rho_0 : float
        Reference seawater density [kg/m3].

    Returns
    -------
    jax.Array, shape (nCells,)
        Salinity tendency [PSU/s] for top layer.
    """
    F_fw = net_freshwater_flux(fw)
    dz_safe = jnp.maximum(dz_0, 1e-10)
    return -S_ref * F_fw / (rho_0 * dz_safe)


def freshwater_from_coupler(
    precip_total: jnp.ndarray,
    lhflx: jnp.ndarray,
    L_v: float,
    runoff_surface: jnp.ndarray | None = None,
    runoff_subsurface: jnp.ndarray | None = None,
    ice_state_old=None,
    ice_state_new=None,
    ice_config=None,
    ocean_mask: jnp.ndarray | None = None,
    dt: float = 1.0,
) -> FreshwaterForcing:
    """Compute freshwater forcing from coupler fields.

    Parameters
    ----------
    precip_total : jax.Array, shape (nCells,)
        Total precipitation [kg/m2/s].
    lhflx : jax.Array, shape (nCells,)
        Latent heat flux [W/m2], positive upward.
    L_v : float
        Latent heat of vaporization [J/kg].
    runoff_surface : jax.Array or None, shape (nCells,)
        Surface runoff from land [kg/m2/s].
    runoff_subsurface : jax.Array or None, shape (nCells,)
        Subsurface runoff from land [kg/m2/s].
    ice_state_old, ice_state_new : SeaIceState or None
        Ice state before/after ice step. Used to compute ice freshwater.
    ice_config : SeaIceConfig or None
        Ice config with rho_ice.
    ocean_mask : jax.Array or None, shape (nCells,)
        Ocean mask (1=ocean). Used to restrict fluxes to ocean cells.
    dt : float
        Timestep [s]. Used for ice thickness change rate.

    Returns
    -------
    FreshwaterForcing
    """
    nCells = precip_total.shape[0]

    # Precipitation over ocean
    precip = precip_total

    # Evaporation from latent heat flux: E = lhflx / L_v
    evap = lhflx / L_v

    # Land runoff (sum surface + subsurface).  Pin the zero-fallback
    # dtype to the precip path so a missing runoff input does not
    # silently widen the freshwater forcing struct to f64 under x64.
    # Both fields are summed independently (codex adversarial review,
    # iter-1, bug #3) — previously, ``runoff_subsurface`` was silently
    # dropped whenever ``runoff_surface`` was ``None``.
    runoff = jnp.zeros(nCells, dtype=precip.dtype)
    if runoff_surface is not None:
        runoff = runoff + runoff_surface
    if runoff_subsurface is not None:
        runoff = runoff + runoff_subsurface

    # Ice freshwater: based on areal ice mass change.
    # ice_mass = rho_ice * h * A  (per unit area of grid cell)
    # ice_fw = -(ice_mass_new - ice_mass_old) / dt
    # Melting (mass decrease) puts freshwater into ocean (positive fw).
    # Supports both single-category and multi-category ice.
    if ice_state_old is not None and ice_state_new is not None and ice_config is not None:
        h_old = ice_state_old.h_ice.data
        h_new = ice_state_new.h_ice.data
        A_old = ice_state_old.concentration.data
        A_new = ice_state_new.concentration.data
        rho_ice = ice_config.rho_ice
        ice_mass_old = rho_ice * h_old * A_old
        ice_mass_new = rho_ice * h_new * A_new
        # Multi-category: h has more dims than precip_total; sum categories.
        n_extra = ice_mass_old.ndim - precip_total.ndim
        for _ in range(n_extra):
            ice_mass_old = jnp.sum(ice_mass_old, axis=-1)
            ice_mass_new = jnp.sum(ice_mass_new, axis=-1)
        ice_fw = -(ice_mass_new - ice_mass_old) / jnp.maximum(dt, 1e-10)
    else:
        ice_fw = jnp.zeros(nCells, dtype=precip.dtype)

    # Mask to ocean cells
    if ocean_mask is not None:
        precip = precip * ocean_mask
        evap = evap * ocean_mask
        runoff = runoff * ocean_mask
        ice_fw = ice_fw * ocean_mask

    return FreshwaterForcing(
        precip=precip,
        evap=evap,
        runoff=runoff,
        ice_fw=ice_fw,
    )

```

```python
# === src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py ===
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
    N = jnp.sqrt(jnp.maximum(N2, 0.0))
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

```

```python
# === src/legoesm/ocean/physics/convection/plume.py ===
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

        # Detrainment tendency at this level
        dT_k = cfg.alpha_plume * cfg.epsilon * (T_plume - T[..., k]) * active
        dS_k = cfg.alpha_plume * cfg.epsilon * (S_plume - S[..., k]) * active

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

```

```python
# === src/legoesm/ocean/physics/convection/enhanced_diffusion.py ===
"""Enhanced diffusion for convective adjustment.

Applies large vertical diffusivity where N^2 < 0 (statically unstable).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import compute_buoyancy_frequency
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.physics.convection.output import OceanConvectionOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def enhanced_diffusion_convection(
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: EnhancedDiffusionConfig,
) -> OceanConvectionOutput:
    """Apply enhanced diffusion where the water column is unstable.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : EnhancedDiffusionConfig

    Returns
    -------
    OceanConvectionOutput
    """
    # N^2 at interfaces
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)

    # Diffusivity: large where N^2 < 0
    if cfg.smooth_transition:
        # Smooth sigmoid transition
        K = cfg.K_bg + (cfg.K_conv - cfg.K_bg) * jax.nn.sigmoid(
            -N2 * cfg.sigmoid_sharpness
        )
    else:
        K = jnp.where(N2 < 0.0, cfg.K_conv, cfg.K_bg)

    # Convection flag
    if cfg.smooth_transition:
        flag = jax.nn.sigmoid(-N2 * cfg.sigmoid_sharpness)
    else:
        flag = jnp.where(N2 < 0.0, 1.0, 0.0)

    # Apply variable-K vertical diffusion to T and S
    tracers = jnp.stack([T, S], axis=0)
    tr_tend = jax.vmap(
        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, K),
        in_axes=0, out_axes=0,
    )(tracers)

    return OceanConvectionOutput(
        dT_dt=tr_tend[0],
        dS_dt=tr_tend[1],
        convection_flag=flag,
    )

```

```python
# === src/legoesm/ocean/physics/shortwave_penetration.py ===
"""Subsurface shortwave penetration heating.

Distributes downwelling shortwave radiation through the water column
using a two-band exponential absorption profile (Paulson & Simpson 1977,
Jerlov water types).  Without this, all SW heating is applied to the
surface layer, producing unrealistically warm SST and shallow mixed layers.

References
----------
Paulson, C. A. & Simpson, J. J. (1977): Irradiance measurements in the
    upper ocean. J. Phys. Oceanogr., 7(6), 952-956.
Jerlov, N. G. (1976): Marine Optics. Elsevier, 231 pp.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


# ==============================================================================
# Jerlov water type parameters
# ==============================================================================
# Two-band model: I(z) = Q_sw * [R * exp(z/zeta1) + (1-R) * exp(z/zeta2)]
# where z is negative (depth below surface), zeta1/zeta2 are e-folding depths.

class JerlovParams(NamedTuple):
    """Two-band parameters for a Jerlov water type."""
    R: float        # Fraction in short-wavelength band (red/IR)
    zeta1: float    # e-folding depth of band 1 [m] (short, ~red/IR)
    zeta2: float    # e-folding depth of band 2 [m] (long, ~blue/green)


# Standard Jerlov water types (Paulson & Simpson 1977, Table 1).
JERLOV_TYPES: dict[str, JerlovParams] = {
    "I":   JerlovParams(R=0.58, zeta1=0.35, zeta2=23.0),
    "IA":  JerlovParams(R=0.62, zeta1=0.60, zeta2=20.0),
    "IB":  JerlovParams(R=0.67, zeta1=1.00, zeta2=17.0),
    "II":  JerlovParams(R=0.77, zeta1=1.50, zeta2=14.0),
    "III": JerlovParams(R=0.78, zeta1=1.40, zeta2=7.9),
}


class ShortwavePenetrationConfig(NamedTuple):
    """Configuration for subsurface SW penetration.

    Parameters
    ----------
    water_type : str
        Jerlov water type ("I", "IA", "IB", "II", "III").
        Type I = clearest open ocean, Type III = coastal/turbid.
        Default "II" is a reasonable global average.
    """
    water_type: str = "II"


def shortwave_penetration_tendency(
    sw_down: jnp.ndarray,
    z_coord_dz_ref: jnp.ndarray,
    z_coord_z_half_ref: jnp.ndarray,
    jacobian: jnp.ndarray,
    config: ShortwavePenetrationConfig = ShortwavePenetrationConfig(),
    rho_0: float = 1025.0,  # = eos.rho_0
    c_sw: float = 3994.0,   # = eos.c_sw
) -> jnp.ndarray:
    """Compute 3D temperature tendency from subsurface SW absorption.

    Parameters
    ----------
    sw_down : array, shape (...,)
        Downwelling shortwave at the sea surface [W/m²].
    z_coord_dz_ref : array, shape (nlev,)
        Reference layer thicknesses [m].
    z_coord_z_half_ref : array, shape (nlev+1,)
        Reference interface depths [m] (negative, z_half_ref[0]=0).
    jacobian : array, shape (...,)
        Dynamic z-star Jacobian (eta + H) / H.
    config : ShortwavePenetrationConfig
    rho_0 : float
        Reference seawater density [kg/m³].
    c_sw : float
        Specific heat of seawater [J/(kg·K)].

    Returns
    -------
    array, shape (..., nlev)
        Temperature tendency dT/dt [K/s] from SW absorption.
    """
    params = JERLOV_TYPES[config.water_type]
    R = params.R
    zeta1 = params.zeta1
    zeta2 = params.zeta2

    # Interface depths (negative), shape (nlev+1,).
    # Uses reference z (not dynamic z*J) for the absorption profile.
    # Error is O(eta/H) ~ O(1e-4), negligible vs Jerlov parameter
    # uncertainty.  Standard practice in MOM6, NEMO, and POP.
    z_half = z_coord_z_half_ref

    # SW flux at each interface: I(z) = Q_sw * [R*exp(z/zeta1) + (1-R)*exp(z/zeta2)]
    # z_half[0] = 0 (surface), z_half[-1] = -H_max (bottom)
    I_half = R * jnp.exp(z_half / zeta1) + (1.0 - R) * jnp.exp(z_half / zeta2)
    # Shape: (nlev+1,)

    # Fraction absorbed in each layer = I_half[k] - I_half[k+1]
    frac_absorbed = I_half[:-1] - I_half[1:]  # (nlev,)

    # Actual layer thickness
    dz_actual = z_coord_dz_ref * jacobian[..., jnp.newaxis]  # (..., nlev)

    # Temperature tendency: dT/dt = Q_sw * frac / (rho_0 * c_sw * dz)
    dT_dt = (
        sw_down[..., jnp.newaxis] * frac_absorbed / (rho_0 * c_sw * dz_actual)
    )

    return dT_dt

```

```python
# === src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py ===
"""COARE-like bulk air-sea flux formulation."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_P_ATM = 101325.0  # Standard atmosphere [Pa]


def _saturation_specific_humidity(T_K: jnp.ndarray) -> jnp.ndarray:
    """Saturation specific humidity at standard atmosphere pressure."""
    return saturation_mixing_ratio(T_K, jnp.full_like(T_K, _P_ATM))


def bulk_formula_surface_forcing(
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: BulkFormulaConfig,
) -> SurfaceForcingOutput:
    """Apply bulk formulas for air-sea fluxes.

    Supports constant coefficients or stability-dependent MOST algorithms
    (COARE 3.0 or Large & Yeager 2004) selected via ``cfg.bulk_scheme``.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : BulkFormulaConfig

    Returns
    -------
    SurfaceForcingOutput
    """
    shape_3d = T.shape
    dtype = T.dtype

    T_s = T[..., 0] + constants.T_freeze  # (6, n, n)
    q_sat = _saturation_specific_humidity(T_s)

    Q_lw_up = cfg.emissivity * constants.sigma_sb * T_s ** 4

    if cfg.bulk_scheme in ("coare3", "large_yeager"):
        from legoesm.coupler.bulk_flux import compute_most_fluxes
        # Wind is zonal only (prescribed), zero meridional
        u_a = jnp.full_like(T_s, cfg.U_a, dtype=dtype)
        v_a = jnp.zeros_like(T_s)
        T_a = jnp.full_like(T_s, cfg.T_a, dtype=dtype)
        q_a = jnp.full_like(T_s, cfg.q_a, dtype=dtype)
        rho_a = jnp.full_like(T_s, cfg.rho_a, dtype=dtype)

        tau_x, tau_y, Q_sh, Q_lh, _ = compute_most_fluxes(
            u_a, v_a, T_a, q_a, T_s, q_sat, rho_a,
            z_ref=cfg.z_ref,
            z0_init=cfg.z0,
            scheme=cfg.bulk_scheme,
            n_iter=cfg.bulk_n_iter,
        )
        # Fluxes are positive upward; stress opposes wind
        tau_x = -tau_x  # flip to positive eastward
    else:
        # Constant coefficients: wind is zonal-only (u_a = U_a, v_a = 0)
        # to match the directional convention used in the MOST path.
        Q_sh = cfg.rho_a * cfg.c_pa * cfg.C_H * cfg.U_a * (T_s - cfg.T_a)
        Q_lh = cfg.rho_a * cfg.L_v * cfg.C_E * cfg.U_a * (q_sat - cfg.q_a)

        # tau = rho_a * C_D * |U_a| * (u_a, v_a)  — directional stress
        U_a_speed = jnp.abs(cfg.U_a)
        tau_x = jnp.full_like(T_s, cfg.rho_a * cfg.C_D * U_a_speed * cfg.U_a, dtype=dtype)
        tau_y = jnp.zeros_like(T_s)

    # Net heat flux (positive into ocean)
    Q_net = cfg.SW_down - Q_lw_up + cfg.LW_down - Q_sh - Q_lh

    # Convert to top-layer tendencies
    dz_0 = z_coord.dz_ref[0] * jacobian
    inv_rho_dz = 1.0 / (rho_0_ref * jnp.maximum(dz_0, 1e-10))
    inv_rho_csw_dz = 1.0 / (rho_0_ref * c_sw * jnp.maximum(dz_0, 1e-10))

    # Pad with zero on trailing axis instead of alloc-zeros +
    # scatter — single Pad HLO op per field.  Same pattern as the
    # ``prescribed.py`` and ``restoring.py`` rewrites.
    nlev = shape_3d[-1]
    pad_axes = ((0, 0),) * (len(shape_3d) - 1)
    du_dt = jnp.pad(
        (tau_x * inv_rho_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )
    dv_dt = jnp.pad(
        (tau_y * inv_rho_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )
    dT_dt = jnp.pad(
        (Q_net * inv_rho_csw_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )
    # No freshwater forcing in basic bulk formulation
    dS_dt = jnp.zeros(shape_3d, dtype=dtype)

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=tau_x, tau_y=tau_y,
    )

```

```python
# === src/legoesm/ocean/physics/surface_forcing/prescribed.py ===
"""Prescribed surface forcing: fixed wind stress and heat/freshwater fluxes."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
from legoesm.ocean.physics.surface_forcing.config import PrescribedForcingConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput
from legoesm.ocean.physics.surface_forcing.wind_profiles import compute_wind_stress
from legoesm.ocean.vertical import OceanZStarCoordinate


def prescribed_surface_forcing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid,  # Any grid with grid_lat property (GridProtocol)
    cfg: PrescribedForcingConfig,
) -> SurfaceForcingOutput:
    """Apply prescribed surface forcing to the top ocean layer.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    grid : CubedSphereGrid
    cfg : PrescribedForcingConfig

    Returns
    -------
    SurfaceForcingOutput
    """
    nlev = u.shape[-1]
    shape_3d = u.shape
    dtype = u.dtype

    # Top layer thickness
    dz_0 = z_coord.dz_ref[0] * jacobian  # (6, n, n)
    inv_rho_dz = 1.0 / (rho_0_ref * jnp.maximum(dz_0, 1e-10))

    # Wind stress from shared grid-agnostic computation
    tau_x, tau_y = compute_wind_stress(grid.grid_lat, cfg)

    # Build top-layer-only tendencies via ``jnp.pad`` along the
    # trailing axis instead of allocating a fresh full ``(*, nlev)``
    # zero buffer per field and scattering the surface row.  Single
    # Pad HLO op each — this physics fires every ocean step in
    # configurations that use the prescribed surface forcing.
    nlev = shape_3d[-1]
    pad_axes = ((0, 0),) * (len(shape_3d) - 1)
    du_dt = jnp.pad(
        (tau_x * inv_rho_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )
    dv_dt = jnp.pad(
        (tau_y * inv_rho_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )

    # Heat flux: dT/dt = Q_net / (rho_0 * c_sw * dz_0)
    Q_net = jnp.full_like(dz_0, cfg.Q_net, dtype=dtype)
    inv_rho_csw_dz = 1.0 / (rho_0_ref * c_sw * jnp.maximum(dz_0, 1e-10))
    dT_dt = jnp.pad(
        (Q_net * inv_rho_csw_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )

    # Freshwater (virtual salt flux): dS/dt = +S * E_minus_P / dz_0
    # Positive E-P means net evaporation → water leaves → salt concentrates → dS/dt > 0
    if cfg.E_minus_P != 0.0:
        inv_dz = 1.0 / jnp.maximum(dz_0, 1e-10)
        dS_dt = jnp.pad(
            (S[..., 0] * cfg.E_minus_P * inv_dz)[..., None],
            (*pad_axes, (0, nlev - 1)),
        )
    else:
        dS_dt = jnp.zeros(shape_3d, dtype=dtype)

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=tau_x, tau_y=tau_y,
    )

```

```python
# === src/legoesm/ocean/physics/bottom_drag/linear.py ===
"""Linear bottom drag: du/dt = -r * u / dz_bottom.

The coefficient r has units [m/s] so that the bottom stress
tau = rho_0 * r * u is independent of vertical resolution.
This matches MITgcm's ``bottomDragLinear`` convention and is
consistent with the quadratic drag implementation.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.bottom_drag.config import LinearDragConfig
from legoesm.ocean.physics.bottom_drag.output import BottomDragOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)


def linear_bottom_drag(
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: LinearDragConfig,
) -> BottomDragOutput:
    """Apply linear bottom drag at the deepest level.

    du/dt = -r * u / dz_bottom   [m/s^2]

    where dz_bottom = dz_ref[-1] * jacobian is the physical thickness
    of the bottom layer.

    Parameters
    ----------
    u, v : array (..., nlev)
        Velocity components.
    z_coord : OceanZStarCoordinate
        Vertical coordinate (provides reference layer thicknesses).
    jacobian : array (...)
        z-star Jacobian (eta + H_bathy) / H_max at cell centers.
    cfg : LinearDragConfig
        Configuration with r in [m/s].

    Returns
    -------
    BottomDragOutput
    """
    # Bottom layer thickness
    dz_bottom = z_coord.dz_ref[-1] * jacobian  # (...)
    inv_dz = 1.0 / jnp.maximum(dz_bottom, _EPS)

    # Pad with zero on top instead of allocating ``zeros_like`` and
    # scattering only the bottom row.  Single Pad HLO op vs alloc +
    # dynamic_update_slice.
    nlev = u.shape[-1]
    drag_u = -cfg.r * u[..., -1] * inv_dz
    drag_v = -cfg.r * v[..., -1] * inv_dz
    pad_axes = ((0, 0),) * (drag_u.ndim)
    du_dt = jnp.pad(drag_u[..., None], (*pad_axes, (nlev - 1, 0)))
    dv_dt = jnp.pad(drag_v[..., None], (*pad_axes, (nlev - 1, 0)))
    return BottomDragOutput(du_dt=du_dt, dv_dt=dv_dt)

```

```python
# === src/legoesm/ocean/physics/bottom_drag/quadratic.py ===
"""Quadratic bottom drag: tau = -C_d * |u| * u."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.bottom_drag.config import QuadraticDragConfig
from legoesm.ocean.physics.bottom_drag.output import BottomDragOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def quadratic_bottom_drag(
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: QuadraticDragConfig,
) -> BottomDragOutput:
    """Apply quadratic bottom drag at the deepest level.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : QuadraticDragConfig

    Returns
    -------
    BottomDragOutput
    """
    eps = _EPS

    # Bottom layer thickness
    dz_bottom = z_coord.dz_ref[-1] * jacobian  # (6, n, n)

    # Speed at bottom level
    u_bot = u[..., -1]
    v_bot = v[..., -1]
    speed = jnp.sqrt(u_bot**2 + v_bot**2 + eps)

    # Drag: -C_d * |u| * u / dz_bottom
    inv_dz = 1.0 / jnp.maximum(dz_bottom, eps)
    drag_u = -cfg.C_d * speed * u_bot * inv_dz
    drag_v = -cfg.C_d * speed * v_bot * inv_dz

    # Pad with zero on top instead of allocating ``zeros_like`` and
    # scattering only the bottom row.  Single Pad HLO op vs alloc +
    # dynamic_update_slice.  The bottom row is ``drag_u``/``drag_v``;
    # the top ``nlev-1`` rows are zero by construction.
    nlev = u.shape[-1]
    pad_axes = ((0, 0),) * (drag_u.ndim)
    du_dt = jnp.pad(drag_u[..., None], (*pad_axes, (nlev - 1, 0)))
    dv_dt = jnp.pad(drag_v[..., None], (*pad_axes, (nlev - 1, 0)))

    return BottomDragOutput(du_dt=du_dt, dv_dt=dv_dt)

```
