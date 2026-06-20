"""Factory for ocean vertical mixing physics."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import (
    compute_ocean_rho as _compute_rho,
    thermal_expansion_coeff,
    haline_contraction_coeff,
)
from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.constant import constant_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.richardson import richardson_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.physics.tendencies import make_none_physics_fn, wrap_ocean_tendencies


def make_vertical_mixing_physics(
    config: VerticalMixingConfig,
    apply_diffusion: bool = True,
    constants_config: ConstantsConfig = ConstantsConfig(),
) -> Callable:
    """Create a vertical mixing physics function.

    Parameters
    ----------
    config : VerticalMixingConfig
    apply_diffusion : bool
        If False, the scheme returns zero local-diffusion tendency for
        momentum and tracers but still computes the K_v/A_v profiles and
        (for KPP) the non-local counter-gradient flux.  Use this with the
        implicit backward-Euler vertical diffusion solver in the
        dynamics step.

    Returns
    -------
    Callable : physics_fn(state, grid, z_coord) -> OceanTendencies
    """
    scheme = config.scheme

    # Tidal mixing rides on VerticalMixingConfig but is NOT applied by this
    # factory (nor by make_ocean_physics) — it is a separate caller-applied
    # additive step: precompute K_tidal via
    # vertical_mixing.tidal.compute_tidal_diffusivity and apply it with
    # ocean.coupler.tidal_mixing_apply.apply_tidal_mixing_step. Reject
    # enabled=True here so it cannot SILENTLY no-op inside the physics
    # composition (per the no-silent-default dispatch rule).
    if config.tidal.enabled:
        raise NotImplementedError(
            "VerticalMixingConfig.tidal.enabled=True is not consumed by "
            "make_vertical_mixing_physics / make_ocean_physics. Tidal mixing is "
            "applied as a separate additive step via "
            "ocean.coupler.tidal_mixing_apply.apply_tidal_mixing_step (with a "
            "K_tidal field from vertical_mixing.tidal.compute_tidal_diffusivity); "
            "enable it there, not in the physics-composition config."
        )

    if scheme == "none":
        return make_none_physics_fn()
    elif scheme == "constant":
        return _make_constant(config, apply_diffusion=apply_diffusion)
    elif scheme == "richardson":
        return _make_richardson(config, apply_diffusion=apply_diffusion)
    elif scheme == "kpp":
        return _make_kpp(config, apply_diffusion=apply_diffusion,
                         constants_config=constants_config)
    elif scheme == "tke":
        return _make_tke(config, apply_diffusion=apply_diffusion)
    elif scheme == "catke":
        return _make_catke(config, apply_diffusion=apply_diffusion)
    else:
        raise ValueError(f"Unknown vertical mixing scheme: {scheme!r}")


def _make_constant(config: VerticalMixingConfig,
                   apply_diffusion: bool = True) -> Callable:
    cfg = config.constant

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        out = constant_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            z_coord, J, cfg,
            apply_diffusion=apply_diffusion,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state,
                                K_v=out.K_v if not apply_diffusion else None,
                                A_v=out.A_v if not apply_diffusion else None)
    return physics_fn


def _make_richardson(config: VerticalMixingConfig,
                     apply_diffusion: bool = True) -> Callable:
    cfg = config.richardson

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        rho = _compute_rho(state, z_coord, J)
        out = richardson_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, z_coord, J, cfg,
            apply_diffusion=apply_diffusion,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state,
                                K_v=out.K_v if not apply_diffusion else None,
                                A_v=out.A_v if not apply_diffusion else None)
    return physics_fn


def _make_kpp(config: VerticalMixingConfig,
              apply_diffusion: bool = True,
              constants_config: ConstantsConfig = ConstantsConfig()) -> Callable:
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
        salt  = getattr(surface_forcing, "salt_flux", None) if surface_forcing else None

        # Surface kinematic heat flux: Q_T = q_net / (rho_0 * c_sw)  [K m/s]
        # KPP convention: positive Q_T heats the ocean.
        Q_sfc_T = None
        B_f = None
        if q_net is not None:
            Q_sfc_T = q_net / (constants_config.rho_0 * constants_config.c_sw)
            # Surface thermal expansion at the top layer.
            T_sfc = state.T.data[..., 0]
            S_sfc = state.S.data[..., 0]
            p_sfc = jnp.zeros_like(T_sfc)
            alpha = thermal_expansion_coeff(T_sfc, S_sfc, p_sfc)
            # Buoyancy flux from heat: B_heat = g * alpha * Q_T  (positive
            # Q_T = warming = lighter water at top = stabilizing).  KPP
            # convention is B_f > 0 = unstable (cooling-driven), so we
            # keep the *negative* of the heat-driven contribution.
            B_f = -constants_config.g * alpha * Q_sfc_T

        # Surface kinematic salt flux from freshwater: Q_S = -S_sfc * F_fw
        # / rho_0  [PSU m/s].  Net P-E entering ocean (F_fw > 0) freshens
        # the surface, hence the negative sign.
        Q_sfc_S = None
        if fw is not None or salt is not None:
            S_sfc = state.S.data[..., 0]
            T_sfc = state.T.data[..., 0]
            p_sfc = jnp.zeros_like(T_sfc)
            beta = haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
            # Kinematic surface salt flux [PSU·m/s] = freshwater virtual-salt
            # (-S*fw/rho) PLUS the REAL salt-mass flux (+salt*1e3/rho).
            Q_sfc_S = jnp.zeros_like(S_sfc)
            if fw is not None:
                Q_sfc_S = Q_sfc_S - S_sfc * fw / constants_config.rho_0
            if salt is not None:
                Q_sfc_S = Q_sfc_S + salt * 1.0e3 / constants_config.rho_0
            # Salt-driven surface buoyancy flux (KPP convention,
            # B_f > 0 = unstable):
            #   B_f = -g*(alpha*Q_T - beta*Q_S) = -g*alpha*Q_T + g*beta*Q_S
            # so the salt contribution is +g*beta*Q_S, NOT -g*beta*Q_S.
            # Sanity check: freshening (fw>0) gives Q_sfc_S<0 (salt flux
            # INTO ocean is negative) → B_salt = +g*beta*(neg) < 0
            # (stabilizing, lighter water on top).  Brine rejection / a
            # positive real salt flux gives Q_sfc_S>0 → B_salt > 0
            # (destabilizing).
            B_salt = constants_config.g * beta * Q_sfc_S
            B_f = B_salt if B_f is None else (B_f + B_salt)

        # KPP expects u, v at cell centers (same shape as T).
        # On C-grids, u is (n_lat, n_lon+1, nlev) and v is
        # (n_lat+1, n_lon, nlev) — average to cell centers.
        u_data = state.u.data
        v_data = state.v.data
        if u_data.shape[1] != state.T.data.shape[1]:
            # C-grid: u at lon+1, v at lat+1 faces → cell centers
            u_data = 0.5 * (u_data[:, :-1, :] + u_data[:, 1:, :])
            v_data = 0.5 * (v_data[:-1, :, :] + v_data[1:, :, :])
        out = kpp_vertical_mixing(
            u_data, v_data, state.T.data, state.S.data,
            rho, state.eta.data, z_coord, J, cfg,
            tau_x=tau_x, tau_y=tau_y, B_f=B_f,
            Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
            apply_diffusion=apply_diffusion,
        )
        # When apply_diffusion is False, KPP returns zero du/dv at
        # cell-center shape (from the C-grid u/v interpolation above).
        # Pass None so _wrap_tendencies uses the face-shaped zero
        # template — avoids shape mismatch with C-grid state.
        _du = None if not apply_diffusion else out.du_dt
        _dv = None if not apply_diffusion else out.dv_dt
        return _wrap_tendencies(_du, _dv, out.dT_dt, out.dS_dt, state,
                                K_v=out.K_v if not apply_diffusion else None,
                                A_v=out.A_v if not apply_diffusion else None)
    return physics_fn



def _make_tke(config: VerticalMixingConfig,
              apply_diffusion: bool = True) -> Callable:
    """Factory for the Gaspar 1990 / Burchard 2002 TKE closure
    (Veros's canonical vertical mixing scheme).

    The TKE closure requires the implicit vertical-mixing path
    (``LatLonCGridOceanConfig.implicit_vertical_mixing=True``). The
    actual K_M / K_H computation runs inside
    :func:`legoesm.ocean.physics.vertical_mixing.k_profiles._vmix_K_profiles`
    where the model timestep is available; this factory simply returns
    a no-op physics tendency (zero everywhere, ``K_v=None``,
    ``A_v=None``) so that the model's implicit-mixing path triggers
    the ``compute_vertical_K_profiles`` fallback and uses the TKE
    branch wired there.

    When ``apply_diffusion`` is ``True`` (explicit-vertical-mixing
    mode) we raise — TKE is implicit-only in legoESM v1 to match
    Veros's ``enable_implicit_vert_friction=True`` recipe convention.
    """
    if apply_diffusion:
        raise ValueError(
            "vertical_mixing=\"tke\" requires implicit_vertical_mixing=True. "
            "Set LatLonCGridOceanConfig.implicit_vertical_mixing=True so the "
            "implicit vertical solver can consume the K profiles computed "
            "by the TKE closure."
        )

    def physics_fn(state, grid, z_coord, surface_forcing=None):
        # No-op: TKE K-profiles are computed by the implicit solver via
        # _vmix_K_profiles (k_profiles.py). Returning K_v=None /
        # A_v=None triggers the fallback path in
        # _apply_implicit_vertical_mixing → compute_vertical_K_profiles.
        return _wrap_tendencies(None, None, None, None, state,
                                 K_v=None, A_v=None)

    return physics_fn


def _make_catke(config: VerticalMixingConfig,
                apply_diffusion: bool = True) -> Callable:
    """Factory for the CATKE closure (Wagner et al. 2025).

    Like the TKE closure, CATKE is implicit-only and prognostic: the actual
    K_M / K_H computation + the backward-Euler TKE step run inside
    :func:`...k_profiles._vmix_K_profiles` (the ``"catke"`` branch) where the
    model timestep + carried ``OceanState.tke`` are available.  This factory
    returns a no-op physics tendency so the model's implicit-mixing path
    triggers the ``compute_vertical_K_profiles`` fallback and uses that branch.
    Requires ``implicit_vertical_mixing=True`` (raises otherwise).
    """
    if apply_diffusion:
        raise ValueError(
            "vertical_mixing=\"catke\" requires implicit_vertical_mixing=True. "
            "Set LatLonCGridOceanConfig.implicit_vertical_mixing=True so the "
            "implicit vertical solver can consume the K profiles computed by "
            "the CATKE closure."
        )

    def physics_fn(state, grid, z_coord, surface_forcing=None):
        # No-op: CATKE K-profiles + the prognostic TKE step are computed by the
        # implicit solver via _vmix_K_profiles (catke branch).
        return _wrap_tendencies(None, None, None, None, state,
                                 K_v=None, A_v=None)

    return physics_fn


def _wrap_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state,
                     K_v=None, A_v=None):
    t = wrap_ocean_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state)
    if K_v is not None or A_v is not None:
        t = t._replace(K_v=K_v, A_v=A_v)
    return t
