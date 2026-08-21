"""Factory for ocean vertical mixing physics."""

from __future__ import annotations

from typing import Callable

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import (
    compute_ocean_rho as _compute_rho,
    compute_ocean_rho_and_pressure as _compute_rho_and_pressure,
)
from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
from legoesm.ocean.physics.vertical_mixing.config import (
    VALID_VERTICAL_MIXING_SCHEMES,
    VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.constant import constant_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.richardson import richardson_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.physics.vertical_mixing._shared import surface_buoyancy_flux
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
    # Internal wave-driven mixing (zdfiwm) is NOT applied by the
    # explicit-tendency composition: it is added inside
    # k_profiles.compute_vertical_K_profiles (the implicit vertical-mixing
    # path), the only place its avm contribution can enter the
    # backward-Euler momentum solve.  With apply_diffusion=True (the
    # explicit route) nothing downstream consumes it → reject so it cannot
    # silently no-op; with apply_diffusion=False (the implicit route) the
    # host model's K-profile solve is the consumer — legitimate.
    if (apply_diffusion
            and getattr(config, "iwm", None) is not None
            and config.iwm.enabled):
        raise NotImplementedError(
            "VerticalMixingConfig.iwm.enabled=True is not consumed by the "
            "EXPLICIT vertical-mixing composition.  Internal wave-driven "
            "mixing is applied inside compute_vertical_K_profiles and "
            "requires implicit_vertical_mixing=True on the host model "
            "config."
        )
    if (apply_diffusion
            and getattr(config, "ddm", None) is not None
            and config.ddm.enabled):
        raise NotImplementedError(
            "VerticalMixingConfig.ddm.enabled=True is not consumed by the "
            "EXPLICIT vertical-mixing composition.  Double-diffusive mixing "
            "is applied inside compute_vertical_K_profiles (separate salt "
            "diffusivity) and requires implicit_vertical_mixing=True on the "
            "host model config."
        )

    if scheme == "none":
        return make_none_physics_fn()
    elif scheme == "constant":
        return _make_constant(config, apply_diffusion=apply_diffusion)
    elif scheme == "richardson":
        return _make_richardson(config, apply_diffusion=apply_diffusion,
                                constants_config=constants_config)
    elif scheme == "kpp":
        return _make_kpp(config, apply_diffusion=apply_diffusion,
                         constants_config=constants_config)
    elif scheme == "tke":
        return _make_tke(config, apply_diffusion=apply_diffusion)
    elif scheme == "catke":
        return _make_catke(config, apply_diffusion=apply_diffusion)
    else:
        # Same canonical source as the implicit K-profile dispatcher, so the
        # explicit-composition factory and _vmix_K_profiles never drift apart on
        # which schemes are selectable.
        raise ValueError(
            f"unknown vertical_mixing.scheme={scheme!r}; expected one of "
            f"{sorted(VALID_VERTICAL_MIXING_SCHEMES)}"
        )


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
                     apply_diffusion: bool = True,
                     constants_config: ConstantsConfig = ConstantsConfig()) -> Callable:
    cfg = config.richardson

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        # The adiabatic PP81 N² trigger needs the cell-centre hydrostatic
        # pressure; compute it (with rho) only when opted in so the default
        # in-situ path stays bit-identical.  eos_fn is None here (this factory
        # does not thread a recipe EOS) → Wright, matching the density path.
        if cfg.n2_mode == "adiabatic":
            rho, p_cell = _compute_rho_and_pressure(
                state, z_coord, J, g=constants_config.g,
                rho0=constants_config.rho_0)
        else:
            rho = _compute_rho(state, z_coord, J, g=constants_config.g,
                               rho0=constants_config.rho_0)
            p_cell = None
        out = richardson_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, z_coord, J, cfg,
            apply_diffusion=apply_diffusion,
            p_cell=p_cell,
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
        # The run's constants, not the library's: this density is what
        # every downstream pressure integrates (#1627).
        rho = _compute_rho(state, z_coord, J,
                           rho0=constants_config.rho_0,
                           g=constants_config.g)

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

        # Surface buoyancy + kinematic T/S fluxes (shared grid-agnostic kernel,
        # #518 item 1).  Lat-lon convention: the real salt-mass flux feeds BOTH
        # the surface buoyancy and the non-local Q_sfc_S (real_salt_in_qs=True).
        B_f, Q_sfc_T, Q_sfc_S = surface_buoyancy_flux(
            q_net, fw, salt,
            state.T.data[..., 0], state.S.data[..., 0],
            g=constants_config.g,
            rho_0=constants_config.rho_0,
            c_sw=constants_config.c_sw,
            real_salt_in_qs=True,
        )

        # KPP expects u, v at cell centers (same shape as T).
        # On C-grids, u is (n_lat, n_lon+1, nlev) and v is
        # (n_lat+1, n_lon, nlev) — average to cell centers.
        u_data = state.u.data
        v_data = state.v.data
        if u_data.shape[1] != state.T.data.shape[1]:
            # C-grid: u at lon+1, v at lat+1 faces → cell centers
            u_data = 0.5 * (u_data[:, :-1, :] + u_data[:, 1:, :])
            v_data = 0.5 * (v_data[:-1, :, :] + v_data[1:, :, :])
        # Under-ice velocity-scale attenuation (KPPConfig.eice; NEMO nn_eice) —
        # same static-config gate as the implicit k_profiles path so the
        # explicit pipeline honours eice instead of silently no-oping it
        # (dispatch discipline).  eice=0 -> ice_frac=None -> bit-identical.
        _kpp_eice = int(getattr(cfg, "eice", 0))
        if _kpp_eice not in (0, 1, 3):
            raise ValueError(
                f"Unknown KPPConfig.eice={_kpp_eice!r}; expected 0, 1 or 3.")
        _kpp_ice_fr = (getattr(surface_forcing, "ice_concentration", None)
                       if (_kpp_eice != 0 and surface_forcing is not None)
                       else None)
        out = kpp_vertical_mixing(
            u_data, v_data, state.T.data, state.S.data,
            rho, state.eta.data, z_coord, J, cfg,
            tau_x=tau_x, tau_y=tau_y, B_f=B_f,
            Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
            apply_diffusion=apply_diffusion, ice_frac=_kpp_ice_fr,
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
