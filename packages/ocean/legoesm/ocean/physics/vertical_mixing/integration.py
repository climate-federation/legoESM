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
from legoesm.ocean.physics.vertical_mixing.epbl import (
    epbl_vertical_mixing,
    surface_tke_forcing,
)
from legoesm.ocean.physics.vertical_mixing._shared import (
    eos_salinity,
    surface_buoyancy_flux,
)
from legoesm.ocean.physics.tendencies import make_none_physics_fn, wrap_ocean_tendencies


def make_vertical_mixing_physics(
    config: VerticalMixingConfig,
    apply_diffusion: bool = True,
    constants_config: ConstantsConfig = ConstantsConfig(),
    eos_fn=None,
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
    eos_fn : callable or None
        The run's EOS ``fn(T, S, p) -> rho`` for the Richardson density and
        ePBL's specific-volume derivatives and surface buoyancy (#2177).
        ``None`` = Wright 1997 (unchanged). KPP does not take it yet.

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
                                constants_config=constants_config,
                                eos_fn=eos_fn)
    elif scheme == "kpp":
        return _make_kpp(config, apply_diffusion=apply_diffusion,
                         constants_config=constants_config)
    elif scheme == "epbl":
        return _make_epbl(config, apply_diffusion=apply_diffusion,
                          constants_config=constants_config, eos_fn=eos_fn)
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
                     constants_config: ConstantsConfig = ConstantsConfig(),
                     eos_fn=None) -> Callable:
    cfg = config.richardson

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        # The adiabatic PP81 N² trigger needs the cell-centre hydrostatic
        # pressure; compute it (with rho) only when opted in so the default
        # in-situ path stays bit-identical.  eos_fn is the run's EOS (#2177);
        # None → Wright.
        if cfg.n2_mode == "adiabatic":
            rho, p_cell = _compute_rho_and_pressure(
                state, z_coord, J, eos_fn=eos_fn, g=constants_config.g,
                rho0=constants_config.rho_0)
        else:
            rho = _compute_rho(state, z_coord, J, eos_fn=eos_fn,
                               g=constants_config.g,
                               rho0=constants_config.rho_0)
            p_cell = None
        out = richardson_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, z_coord, J, cfg,
            apply_diffusion=apply_diffusion,
            p_cell=p_cell, eos_fn=eos_fn,
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
        if _kpp_eice not in (0, 1, 2, 3):
            raise ValueError(
                f"Unknown KPPConfig.eice={_kpp_eice!r}; expected 0, 1, 2 or 3.")
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



def _make_epbl(config: VerticalMixingConfig,
               apply_diffusion: bool = True,
               constants_config: ConstantsConfig = ConstantsConfig(),
               eos_fn=None) -> Callable:
    """Factory for the energetic PBL closure (Reichl & Hallberg 2018).

    ePBL is implicit-only: it returns the interface diffusivity/viscosity for
    the backward-Euler solver and no explicit tendency, so the host model
    must run with ``implicit_vertical_mixing=True``.

    ``cfg.dt_s`` is used as given: this factory does not know the caller's
    timestep and does not check it.  The OMIP driver, the ocean SCM and the
    lat-lon tendency probe check (or set) it against their dt; any other
    caller of this factory must set ``EPBLConfig.dt_s`` to the dt its
    implicit solve uses.
    """
    import jax.numpy as jnp

    from legoesm.ocean.eos import eos_density_derivatives, make_eos_fn
    from legoesm.ocean.physics.vertical_mixing.epbl import validate_epbl_config

    cfg = config.epbl
    validate_epbl_config(cfg)
    if apply_diffusion:
        raise ValueError(
            "vertical_mixing=\"epbl\" requires implicit_vertical_mixing=True. "
            "ePBL prices mixing against the surface energy budget and hands "
            "back K_v/A_v for the unconditionally-stable implicit solve; it "
            "has no explicit-diffusion form."
        )

    if cfg.k_after_surface_flux:
        # The MOM6 step order is a driver-level operator split, not part of
        # the closure.  Only the ocean SCM implements it, and it clears this
        # flag before building physics; reaching here with it set means a
        # driver that would silently ignore it.
        raise NotImplementedError(
            "EPBLConfig.k_after_surface_flux=True (MOM6 step order) is "
            "implemented only in the ocean single-column model; this driver "
            "diagnoses K before the surface flux. Set it to False."
        )

    # The run's EOS (#2177); None -> Wright, as before.
    run_eos = eos_fn
    eos_fn = make_eos_fn() if eos_fn is None else eos_fn

    def physics_fn(state: OceanState, grid, z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        h = z_coord.dz_ref * J[..., jnp.newaxis]          # layer thickness [m]
        T = state.T.data
        S = state.S.data
        g = constants_config.g
        rho_0 = constants_config.rho_0

        # Specific-volume derivatives at MID-LAYER pressure, as MOM6's
        # applyBoundaryFluxesInOut computes them (MOM_diabatic_aux.F90:909).
        d_pres = (g * rho_0) * h
        p_lay = jnp.cumsum(d_pres, axis=-1) - 0.5 * d_pres
        # sqrt(S) EOSes have an infinite dRho/dS at S = 0 (land / fill cells):
        # floor S for them as the surface-buoyancy helper does; Wright as is.
        S_eos = eos_salinity(S, eos_fn)
        rho = eos_fn(T, S_eos, p_lay)
        drho_dT, drho_dS = eos_density_derivatives(eos_fn, T, S_eos, p_lay)
        inv_rho2 = 1.0 / (rho * rho)
        dSV_dT = -drho_dT * inv_rho2
        dSV_dS = -drho_dS * inv_rho2

        tau_x = getattr(surface_forcing, "tau_x", None) if surface_forcing else None
        tau_y = getattr(surface_forcing, "tau_y", None) if surface_forcing else None
        q_net = getattr(surface_forcing, "q_net", None) if surface_forcing else None
        fw = getattr(surface_forcing, "freshwater", None) if surface_forcing else None
        salt = getattr(surface_forcing, "salt_flux", None) if surface_forcing else None
        # Missing forcing is an error, as in MOM6: a zero would run the
        # boundary layer unforced and no downstream guard could see it.
        missing = [n for n, x in (("tau_x", tau_x), ("tau_y", tau_y),
                                  ("q_net", q_net), ("freshwater", fw))
                   if x is None]
        if missing:
            raise ValueError(
                "ePBL needs wind stress, net heat and freshwater forcing; "
                f"surface_forcing supplies no {', '.join(missing)}. Pass "
                "explicit zeros for an unforced column.")

        # Surface buoyancy flux through the shared helper (#2178), MOM6 form:
        # derivatives at surface pressure divided by rho_0 (GoRho = g/Rho0),
        # brine salt included with the freshwater dilution (MOM6 netSalt).
        # The helper's sign is POSITIVE = destabilizing; ePBL's (MOM6's
        # buoyancyFlux) is POSITIVE = stabilizing, hence the minus.
        c_sw = constants_config.c_sw
        B_f, _, Q_sfc_S = surface_buoyancy_flux(
            q_net, fw, salt, T[..., 0], S[..., 0],
            g=g, rho_0=rho_0, c_sw=c_sw, real_salt_in_qs=True,
            rho_0_norm=True, eos_fn=run_eos)
        B_flux = -B_f        # never None: q_net and fw are required above

        # Energy released (or required) by mixing the surface fluxes through
        # the top layer, with the same net salt flux [psu kg/m^2/s] as the
        # buoyancy.  Only the top layer is forced: no penetrating shortwave
        # profile is supplied on this path.
        tke_top = surface_tke_forcing(
            q_net, Q_sfc_S * rho_0,
            h[..., 0], dSV_dT[..., 0], dSV_dS[..., 0],
            g=g, rho_0=rho_0, c_sw=c_sw, dt=cfg.dt_s)
        tke_forcing = jnp.zeros_like(T).at[..., 0].set(tke_top)

        # Coriolis from the grid latitude (ePBL's rotational length scale).
        lat = getattr(grid, "grid_lat", None)
        if lat is None:
            raise ValueError(
                "ePBL needs the column latitude (grid.grid_lat, radians) for "
                "its Coriolis length scale; the grid supplied none.")
        absf = jnp.abs(2.0 * constants_config.Omega * jnp.sin(
            jnp.broadcast_to(lat, T.shape[:-1])))

        # ePBL needs cell-centre velocities; average C-grid faces if needed.
        u_data, v_data = state.u.data, state.v.data
        if u_data.shape[1] != T.shape[1]:
            u_data = 0.5 * (u_data[:, :-1, :] + u_data[:, 1:, :])
            v_data = 0.5 * (v_data[:-1, :, :] + v_data[1:, :, :])

        out = epbl_vertical_mixing(
            u_data, v_data, T, S, dSV_dT, dSV_dS, h, cfg,
            tau_x=tau_x, tau_y=tau_y, B_flux=B_flux,
            tke_forcing=tke_forcing, absf=absf,
            g=g, rho_0=rho_0, apply_diffusion=False,
        )
        return _wrap_tendencies(None, None, out.dT_dt, out.dS_dt, state,
                                K_v=out.K_v, A_v=out.A_v)

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
