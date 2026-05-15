"""MPAS-specific ocean physics: surface forcing and bottom drag.

MPAS uses edge-normal velocity on a TRiSK C-grid, so cell-centered
wind stress (tau_x, tau_y) must be projected onto edge normals.  This
module mirrors the ``make_ocean_physics`` pipeline but returns
``MPASOceanTendencies`` compatible with the MPAS tendency function.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASOceanState, MPASOceanTendencies
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.ocean.eos import rho_0 as rho_0_ref
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig


def make_mpas_ocean_physics(
    config,
    implicit_vertical_mixing: bool = False,
) -> Callable:
    """Build a combined physics function for MPAS ocean.

    Parameters
    ----------
    config : OceanPhysicsConfig
    implicit_vertical_mixing : bool
        When True, KPP and convective-adjustment tendencies are skipped
        here — their K profiles are routed through the backward-Euler
        implicit solver in ``MPASOceanModel.step()`` instead.  Wind,
        restoring, and other physics are still applied as explicit
        tendencies.

    Returns
    -------
    Callable
        ``physics_fn(state, mesh, z_coord, surface_forcing=None)``
        returning ``MPASOceanTendencies``.
    """
    sf_config = config.surface_forcing
    bd_config = config.bottom_drag
    vm_config = getattr(config, "vertical_mixing", None)

    # Vertical mixing dispatch (currently only KPP is wired into MPAS).
    vm_scheme = (vm_config.scheme
                 if vm_config is not None else "none")
    apply_kpp = vm_scheme == "kpp"
    if apply_kpp:
        from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
            make_kpp_physics_mpas,
        )
        _kpp_fn = make_kpp_physics_mpas(vm_config)
    else:
        _kpp_fn = None

    # Warn about unsupported physics schemes that would be silently ignored.
    # ``convection`` is handled explicitly below (supports "enhanced_diffusion").
    # ``vertical_mixing="kpp"`` is now supported (above); other schemes
    # (constant, richardson) are not yet wired in.
    import warnings
    _unsupported = []
    if vm_config is not None and vm_scheme not in ("none", "kpp"):
        _unsupported.append(f"vertical_mixing={vm_scheme!r}")
    for attr in ("lateral_mixing", "shortwave_penetration"):
        sub = getattr(config, attr, None)
        if sub is not None and getattr(sub, "scheme", "none") != "none":
            _unsupported.append(f"{attr}={getattr(sub, 'scheme', '?')!r}")
    if _unsupported:
        warnings.warn(
            f"MPAS ocean physics: ignoring unsupported schemes: "
            + ", ".join(_unsupported),
            RuntimeWarning,
            stacklevel=2,
        )

    sf_scheme = (sf_config.scheme
                 if isinstance(sf_config, SurfaceForcingConfig)
                 else "none")
    # Bail loudly on schemes the MPAS factory does not implement, rather
    # than silently producing zero tendencies.
    _supported_sf = ("none", "prescribed", "restoring", "combined")
    if sf_scheme not in _supported_sf:
        raise NotImplementedError(
            f"MPAS ocean physics does not support surface_forcing scheme "
            f"{sf_scheme!r}. Supported: {_supported_sf}."
        )
    apply_wind_block = sf_scheme in ("prescribed", "combined")
    apply_restoring = sf_scheme in ("restoring", "combined")

    conv_config = getattr(config, "convection", None)
    conv_scheme = (conv_config.scheme
                   if conv_config is not None else "none")
    _supported_conv = ("none", "enhanced_diffusion")
    if conv_scheme not in _supported_conv:
        raise NotImplementedError(
            f"MPAS ocean physics does not support convection scheme "
            f"{conv_scheme!r}. Supported: {_supported_conv}."
        )
    apply_convection = conv_scheme == "enhanced_diffusion"

    # Physics-level bottom drag is deprecated — use the dynamics-level
    # ``bottom_drag_r`` field on ``MPASOceanConfig`` instead.  The
    # dynamics path applies drag in both the baroclinic PE and the
    # barotropic substeps, which is physically correct (MOM6 convention).
    if (isinstance(bd_config, BottomDragConfig)
            and bd_config.scheme != "none"):
        raise ValueError(
            f"Physics-level bottom drag (scheme={bd_config.scheme!r}) is "
            "deprecated. Use MPASOceanConfig(bottom_drag_r=...) instead, "
            "which applies drag in both the baroclinic PE and the "
            "barotropic substeps (matching MOM6). Set "
            "BottomDragConfig(scheme='none') in your OceanPhysicsConfig."
        )

    def physics_fn(
        state: MPASOceanState,
        mesh: VoronoiMesh,
        z_coord: OceanZStarCoordinate,
        surface_forcing=None,
    ) -> MPASOceanTendencies:
        u_3d = state.u.data        # (nEdges, nlev)
        T_3d = state.T.data        # (nCells, nlev)
        eta = state.eta.data        # (nCells,)
        H_bathy = state.H_bathy.data
        mask = state.land_mask.data  # (nCells,)
        dtype = u_3d.dtype

        du_dt = jnp.zeros_like(u_3d)
        dT_dt = jnp.zeros_like(T_3d)
        dS_dt = jnp.zeros_like(T_3d)
        deta_dt = jnp.zeros_like(eta)

        # Jacobian — needed by surface forcing (top-layer thickness) and
        # bottom drag (bottom-layer thickness).  Compute once.
        jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
        c1 = mesh.cellsOnEdge[0]  # (nEdges,)
        c2 = mesh.cellsOnEdge[1]  # (nEdges,)

        # --- Prescribed wind / Q_net / E-P (also reused under "combined") ---
        if apply_wind_block:
            cfg = sf_config.prescribed

            dz_0_cell = z_coord.dz_ref[0] * jacobian  # (nCells,)

            # Compute cell-centered wind stress from latitude
            from legoesm.ocean.physics.surface_forcing.wind_profiles import compute_wind_stress
            tau_x, tau_y = compute_wind_stress(mesh.grid_lat, cfg)

            # Project cell-centered wind stress onto edge normals.
            # Average tau from the two cells sharing each edge, then dot
            # with the edge-normal direction (angleEdge).
            tau_x_e = 0.5 * (tau_x[c1] + tau_x[c2])
            tau_y_e = 0.5 * (tau_y[c1] + tau_y[c2])
            tau_n = (tau_x_e * jnp.cos(mesh.angleEdge)
                     + tau_y_e * jnp.sin(mesh.angleEdge))

            # Edge top-layer thickness
            dz_0_e = 0.5 * (dz_0_cell[c1] + dz_0_cell[c2])
            inv_rho_dz_e = 1.0 / (rho_0_ref * jnp.maximum(dz_0_e, 1e-10))

            # Apply wind stress to top layer only
            du_dt = du_dt.at[:, 0].add(tau_n * inv_rho_dz_e)

            # Heat flux: dT/dt = Q_net / (rho_0 * c_sw * dz_0)
            if cfg.Q_net != 0.0:
                from legoesm.ocean.eos import c_sw
                inv_rho_csw_dz = 1.0 / (
                    rho_0_ref * c_sw * jnp.maximum(dz_0_cell, 1e-10))
                dT_dt = dT_dt.at[:, 0].add(cfg.Q_net * inv_rho_csw_dz * mask)

            # E-P virtual salt flux
            if cfg.E_minus_P != 0.0:
                inv_dz = 1.0 / jnp.maximum(dz_0_cell, 1e-10)
                dS_dt = dS_dt.at[:, 0].add(
                    state.S.data[:, 0] * cfg.E_minus_P * inv_dz * mask)

        # --- External surface forcing (e.g. from JRA55 bulk fluxes) ---
        # When sf_scheme is "none", the prescribed wind block above is
        # skipped.  If the caller passes an OceanSurfaceForcing with
        # tau_x / tau_y / q_net, apply them here.  This is the path
        # used by run_omip.py --forcing-mode jra55_do_tropical on MPAS.
        if not apply_wind_block and surface_forcing is not None:
            _sf_tau_x = getattr(surface_forcing, "tau_x", None)
            _sf_tau_y = getattr(surface_forcing, "tau_y", None)
            _sf_q_net = getattr(surface_forcing, "q_net", None)

            if _sf_tau_x is not None and _sf_tau_y is not None:
                dz_0_cell = z_coord.dz_ref[0] * jacobian  # (nCells,)
                tau_x_e = 0.5 * (_sf_tau_x[c1] + _sf_tau_x[c2])
                tau_y_e = 0.5 * (_sf_tau_y[c1] + _sf_tau_y[c2])
                tau_n = (tau_x_e * jnp.cos(mesh.angleEdge)
                         + tau_y_e * jnp.sin(mesh.angleEdge))
                dz_0_e = 0.5 * (dz_0_cell[c1] + dz_0_cell[c2])
                inv_rho_dz_e = 1.0 / (
                    rho_0_ref * jnp.maximum(dz_0_e, 1e-10))
                du_dt = du_dt.at[:, 0].add(tau_n * inv_rho_dz_e)

            if _sf_q_net is not None:
                from legoesm.ocean.eos import c_sw
                dz_0_cell_q = z_coord.dz_ref[0] * jacobian
                inv_rho_csw_dz = 1.0 / (
                    rho_0_ref * c_sw * jnp.maximum(dz_0_cell_q, 1e-10))
                dT_dt = dT_dt.at[:, 0].add(
                    _sf_q_net * inv_rho_csw_dz * mask)

        # --- T/S restoring (under "restoring" or "combined") ---
        if apply_restoring:
            from legoesm.ocean.physics.surface_forcing.restoring import (
                restoring_surface_forcing,
            )
            cfg_r = sf_config.restoring
            r_out = restoring_surface_forcing(
                state.T.data, state.S.data, mesh, cfg_r,
            )
            # restoring_surface_forcing does not mask land; do it here so
            # land-cell tracer values are not driven by the restoring term.
            dT_dt = dT_dt + r_out.dT_dt * mask[:, None]
            dS_dt = dS_dt + r_out.dS_dt * mask[:, None]

        # --- Convective adjustment (enhanced diffusion where N²<0) ---
        # When implicit_vertical_mixing is True, convection K profiles are
        # routed through the implicit solver in step() — skip the explicit
        # tendency here to avoid double-counting and CFL violations on
        # thin partial cells.
        if apply_convection and not implicit_vertical_mixing:
            from legoesm.ocean.physics.convection.enhanced_diffusion import (
                enhanced_diffusion_convection,
            )
            from legoesm.ocean.eos import compute_ocean_rho
            cfg_c = conv_config.enhanced_diffusion
            # Match the lat-lon convection integration: use the default
            # (Wright) EOS for the ρ used in the static-stability check,
            # even when the dycore is configured with linear EOS.  This
            # is a known approximation — the EOS choice only affects the
            # static-stability ranking, not the dycore tendencies.
            rho = compute_ocean_rho(state, z_coord, jacobian)
            c_out = enhanced_diffusion_convection(
                state.T.data, state.S.data, rho, z_coord, jacobian, cfg_c,
            )
            dT_dt = dT_dt + c_out.dT_dt * mask[:, None]
            dS_dt = dS_dt + c_out.dS_dt * mask[:, None]

        # ---- KPP vertical mixing ----
        # Returns tendencies for u (edge-normal) and T, S (cell-centered).
        # Mask land cells out of tracer tendencies; for edges, the model's
        # edge_mask already zeros out land-touching contributions.
        # When implicit_vertical_mixing is True, KPP K profiles are routed
        # through the implicit solver — skip the explicit tendency here.
        if _kpp_fn is not None and not implicit_vertical_mixing:
            kpp_du, kpp_dT, kpp_dS = _kpp_fn(
                state, mesh, z_coord, surface_forcing,
            )
            du_dt = du_dt + kpp_du
            dT_dt = dT_dt + kpp_dT * mask[:, None]
            dS_dt = dS_dt + kpp_dS * mask[:, None]

        return MPASOceanTendencies(
            du_dt=Field(data=du_dt, name="du_dt",
                        dims=("nEdges", "nlev"), units="m/s²"),
            dT_dt=Field(data=dT_dt, name="dT_dt",
                        dims=("nCells", "nlev"), units="degC/s"),
            dS_dt=Field(data=dS_dt, name="dS_dt",
                        dims=("nCells", "nlev"), units="PSU/s"),
            deta_dt=Field(data=deta_dt, name="deta_dt",
                          dims=("nCells",), units="m/s"),
        )

    return physics_fn
