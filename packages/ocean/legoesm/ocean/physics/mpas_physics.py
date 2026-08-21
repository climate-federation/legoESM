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


# Fraction of net surface SW that penetrates below the skin (Paulson-Simpson).
_SW_PENETRATION_FRACTION = 0.94

def make_mpas_ocean_physics(
    config,
    implicit_vertical_mixing: bool = False,
    eos_fn=None,
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
    eos_fn : callable or None
        Model-selected EOS ``(T, S, p) -> rho`` for the explicit KPP /
        convective-adjustment density diagnostics.  ``None`` ⇒ Wright
        default (bit-identical legacy).  Threaded so a non-Wright EOS
        (e.g. ``nemo_seos``) drives the mixing decision consistently with
        the baroclinic dycore instead of silently via Wright.

    Returns
    -------
    Callable
        ``physics_fn(state, mesh, z_coord, surface_forcing=None)``
        returning ``MPASOceanTendencies``.
    """
    sf_config = config.surface_forcing
    bd_config = config.bottom_drag
    vm_config = getattr(config, "vertical_mixing", None)

    # Vertical mixing dispatch.  KPP has an explicit-tendency path (built here)
    # AND an implicit-profile path (make_kpp_profiles_mpas in the model step);
    # TKE is implicit-only (make_tke_profiles_mpas in the model step) — this
    # factory only validates it here (guarded below) and builds no explicit fn.
    vm_scheme = (vm_config.scheme
                 if vm_config is not None else "none")
    apply_kpp = vm_scheme == "kpp"
    if apply_kpp:
        from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
            make_kpp_physics_mpas,
        )
        _kpp_fn = make_kpp_physics_mpas(vm_config, eos_fn=eos_fn,
                                        constants_config=config.constants)
    else:
        _kpp_fn = None

    # CATKE is a prognostic-TKE closure wired ONLY for the lat-lon C-grid
    # dynamics step (it carries OceanState.tke + needs the implicit solve);
    # on MPAS it would silently become a no-op (kpp_fn=None below).  Fail
    # closed rather than silently mis-run (dispatch discipline; codex review).
    if vm_config is not None and vm_scheme == "catke":
        raise ValueError(
            "vertical_mixing.scheme='catke' is an unsupported vertical-mixing "
            "scheme on the MPAS ocean (CATKE is wired for the lat-lon C-grid "
            "only) and would silently no-op here. Use 'kpp', or run CATKE on "
            "the lat-lon C-grid."
        )

    # TKE (Gaspar/Burchard) on MPAS runs through the implicit vertical solver
    # ONLY (like the lat-lon TKE path): its diagnostic K-profiles are routed
    # through the backward-Euler solve in MPASOceanModel.step (via
    # ``make_tke_profiles_mpas``), and there is NO explicit-tendency TKE
    # operator on the edge-normal C-grid.  Fail-fast when implicit vmix is off
    # rather than silently producing no TKE mixing.  ``vm_scheme`` +
    # ``implicit_vertical_mixing`` are static ⇒ raising at build time is jit-safe.
    if (vm_config is not None and vm_scheme == "tke"
            and not implicit_vertical_mixing):
        raise ValueError(
            "vertical_mixing.scheme='tke' on the MPAS ocean requires "
            "implicit_vertical_mixing=True (the TKE K-profiles are applied by "
            "the backward-Euler implicit solver in MPASOceanModel.step; there "
            "is no explicit-tendency TKE path on the edge-normal C-grid). Set "
            "implicit_vertical_mixing=True, or use scheme='kpp'."
        )
    # NOTE: the prognostic TKE carry (tke.prognostic=True) IS wired on MPAS:
    # MPASOceanModel carries MPASOceanState.tke (seeded via model.seed_tke)
    # and make_tke_profiles_mpas runs the Mode-A one-step en integration —
    # no factory reject here (mirrors the lat-lon path).

    # Bail loudly on a vertical_mixing scheme whose K-PROFILE the MPAS factory
    # does not wire in and would SILENTLY DROP (finding #4).  Supported here:
    #   - "kpp"      : K-profile applied (above);
    #   - "none"     : no scheme K-profile;
    #   - "constant" : the DEFAULT — MPAS gets constant background viscosity /
    #                  diffusivity from ``MPASOceanConfig.A_v``/``K_v`` through the
    #                  implicit vertical solver (ocean_model_mpas.step), NOT through
    #                  this physics K-profile, so accepting it is correct (no silent
    #                  drop of a scheme-specific profile).
    #   - "tke"      : diagnostic Gaspar/Burchard K-profile applied through the
    #                  implicit solver (make_tke_profiles_mpas); requires
    #                  implicit_vertical_mixing=True (guarded above).
    # "richardson" DOES compute a scheme-specific K-profile that MPAS would
    # silently ignore, and "catke" is rejected above — so reject those (and any
    # typo) rather than warn-and-drop, matching the sibling surface_forcing
    # (NotImplementedError) and convection guards (dispatch discipline; CLAUDE.md
    # "Dispatch").  ``vm_scheme`` is the static config value -> raising at factory
    # build time is jit-safe.
    if vm_config is not None and vm_scheme not in (
            "none", "kpp", "constant", "tke"):
        raise NotImplementedError(
            f"MPAS ocean physics does not implement vertical_mixing scheme "
            f"{vm_scheme!r} (its K-profile would be silently ignored). Supported "
            "on MPAS: {'none', 'kpp', 'constant', 'tke'} ('constant' via the "
            "MPASOceanConfig.A_v/K_v background + implicit solver; 'tke' via the "
            "diagnostic quasi-steady closure + implicit solver).  'catke' is "
            "rejected separately; 'richardson' is wired for the lat-lon "
            "C-grid only."
        )

    # ``constant`` on MPAS is honoured via ``MPASOceanConfig.A_v``/``K_v`` (the
    # implicit solver background), NOT via ``VerticalMixingConfig.constant``.  The
    # DEFAULT ConstantVerticalMixingConfig (A_v=1e-3, K_v=1e-4) MATCHES the MPAS
    # background defaults, so the default path is exact.  But a user who sets a
    # NON-DEFAULT ``constant.A_v``/``constant.K_v`` here would have it SILENTLY
    # ignored on MPAS (footgun; codex review #3) — raise so they set the values
    # on ``MPASOceanConfig`` instead (static config value -> jit-safe at build).
    if vm_config is not None and vm_scheme == "constant":
        _const = getattr(vm_config, "constant", None)
        if _const is not None:
            _default_const = type(_const)()
            if (_const.A_v != _default_const.A_v
                    or _const.K_v != _default_const.K_v):
                raise NotImplementedError(
                    "MPAS ocean honours constant vertical mixing through "
                    "MPASOceanConfig.A_v/K_v (the implicit-solver background), "
                    "not VerticalMixingConfig.constant.  A non-default "
                    f"constant.A_v={_const.A_v!r}/K_v={_const.K_v!r} would be "
                    "silently ignored on MPAS — set MPASOceanConfig.A_v/K_v "
                    "instead (the default constant config IS consumed and need "
                    "not be changed)."
                )

    # Warn about the remaining physics schemes that would be silently ignored on
    # MPAS (lateral mixing / shortwave penetration are not yet wired in here).
    # ``convection`` is handled explicitly below (supports "enhanced_diffusion").
    # NB: ``vertical_mixing.scheme="constant"`` is accepted above — MPAS applies a
    # constant background A_v/K_v via ``MPASOceanConfig.A_v``/``K_v`` (its own
    # field), not via ``vertical_mixing.constant``; that subtlety belongs to the
    # MPAS step, not this factory (which has no MPASOceanConfig to compare).
    import warnings
    _unsupported = []
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
    # "external" enables the coupler-provided surface-forcing block below
    # (tau / q_net / real salt_flux from a passed OceanSurfaceForcing) — the
    # MPAS analogue of the cubed-sphere 'external' scheme.  It behaves like
    # "none" plus a required OceanSurfaceForcing; freshwater (eta + virtual
    # salt) is delivered separately through the step(freshwater=) arg.
    _supported_sf = ("none", "prescribed", "restoring", "combined", "external")
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
    # MPAS convective adjustment is tracer-only: edge-normal velocity needs a
    # TRiSK cell->edge reconstruction that the cell-centred enhanced_diffusion
    # momentum operator does not provide, and neither the explicit nor the
    # implicit MPAS path applies a convective edge viscosity.  Reject a
    # nonzero convective momentum viscosity rather than silently dropping it
    # (mirrors the C-grid explicit guard in convection/integration.py).
    if apply_convection:
        _ed = conv_config.enhanced_diffusion
        if _ed.nu_conv != 0.0 or _ed.nu_bg != 0.0:
            raise ValueError(
                "EnhancedDiffusionConfig convective momentum viscosity "
                "(nu_conv/nu_bg) is unsupported on MPAS: the convective "
                "adjustment mixes tracers only (edge-normal momentum would "
                "need a TRiSK cell->edge reconstruction). Set "
                "EnhancedDiffusionConfig(nu_conv=0.0, nu_bg=0.0)."
            )

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
        # Fail CLOSED: 'external' exists solely to apply a coupler-provided
        # OceanSurfaceForcing.  Silently dropping tau/q_net/salt because the
        # struct was forgotten is a whole-run coupling failure, so require it.
        if sf_scheme == "external" and surface_forcing is None:
            raise ValueError(
                "surface_forcing.scheme='external' requires an "
                "OceanSurfaceForcing to be passed to the ocean step "
                "(got surface_forcing=None) — otherwise tau/q_net/salt_flux "
                "are silently dropped. Pass surface_forcing=, or use "
                "scheme='none' for an unforced run.",
            )
        u_3d = state.u.data        # (nEdges, nlev)
        T_3d = state.T.data        # (nCells, nlev)
        eta = state.eta.data        # (nCells,)
        H_bathy = state.H_bathy.data
        mask = state.land_mask.data  # (nCells,)

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
                # Negate: the bulk-flux solver returns tau in the
                # atmosphere convention (opposing the wind).  The ocean
                # needs the reaction force (in the direction of the
                # wind).  The prescribed-wind path (wind_profiles.py)
                # already uses the ocean convention, so this negation
                # only applies to externally-provided bulk-flux tau.
                tau_x_e = -0.5 * (_sf_tau_x[c1] + _sf_tau_x[c2])
                tau_y_e = -0.5 * (_sf_tau_y[c1] + _sf_tau_y[c2])
                tau_n = (tau_x_e * jnp.cos(mesh.angleEdge)
                         + tau_y_e * jnp.sin(mesh.angleEdge))
                dz_0_e = 0.5 * (dz_0_cell[c1] + dz_0_cell[c2])
                inv_rho_dz_e = 1.0 / (
                    rho_0_ref * jnp.maximum(dz_0_e, 1e-10))
                du_dt = du_dt.at[:, 0].add(tau_n * inv_rho_dz_e)

            if _sf_q_net is not None:
                from legoesm.ocean.eos import c_sw
                _sf_sw = getattr(surface_forcing, "sw_down", None)

                if _sf_sw is not None:
                    # VERTICAL split of the surface heat flux (NOT an albedo):
                    # ``q_net`` carries 100% of the incident sw_down; here 94% is
                    # routed through the Jerlov penetration profile and the
                    # remaining 6% is retained as non-solar heating in the surface
                    # skin cell (q_nonsolar = q_net - 0.94*sw).  Total column
                    # heating = q_nonsolar + integral(penetration) = q_net, so NO
                    # SW is reflected/lost here.  Any SURFACE ALBEDO is applied
                    # UPSTREAM in compute_omip2_surface_forcing (--ice-albedo),
                    # which reduces sw_down -> sw_net in BOTH q_net and this field.
                    sw_absorbed = _sf_sw * _SW_PENETRATION_FRACTION  # 94% penetrates; 6% surface skin
                    q_nonsolar = _sf_q_net - sw_absorbed

                    # Non-solar part: surface cell only
                    dz_0_cell_q = z_coord.dz_ref[0] * jacobian
                    inv_rho_csw_dz = 1.0 / (
                        rho_0_ref * c_sw * jnp.maximum(dz_0_cell_q, 1e-10))
                    dT_dt = dT_dt.at[:, 0].add(
                        q_nonsolar * inv_rho_csw_dz * mask)

                    # Solar part: Jerlov penetration through column
                    from legoesm.ocean.physics.shortwave_penetration import (
                        shortwave_penetration_tendency,
                    )
                    sw_tend = shortwave_penetration_tendency(
                        sw_absorbed, z_coord.dz_ref, z_coord.z_half_ref,
                        jacobian, rho_0=rho_0_ref, c_sw=c_sw,
                    )
                    dT_dt = dT_dt + sw_tend * mask[:, None]
                else:
                    # No SW field — all heat into surface (legacy)
                    dz_0_cell_q = z_coord.dz_ref[0] * jacobian
                    inv_rho_csw_dz = 1.0 / (
                        rho_0_ref * c_sw * jnp.maximum(dz_0_cell_q, 1e-10))
                    dT_dt = dT_dt.at[:, 0].add(
                        _sf_q_net * inv_rho_csw_dz * mask)

            # NOTE: the real brine salt_flux is NOT applied here.  It is a
            # top-layer salinity SOURCE applied alongside the freshwater
            # virtual-salt closure in ``mpas_ocean_baroclinic_tendencies``
            # (ocean_pe_mpas.py), so both use the SAME canonical floored top-
            # layer thickness ``h_k[:, 0]`` the tracer update integrates mass
            # against — guaranteeing the injected salt MASS equals salt_flux.
            # (KPP separately reads salt_flux for its surface buoyancy.)

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
            # Static-stability ρ for the convective-adjustment check uses the
            # model-selected EOS (``eos_fn``; ``None`` ⇒ Wright default),
            # matching the lat-lon convection integration which threads its
            # ``_vmix_eos_fn`` (ocean_model_latlon_cgrid.py).  This matters for
            # a depth-dependent (thermobaric) EOS such as ``nemo_seos``, where
            # the stability ranking — not just the dycore tendencies — depends
            # on the EOS; using Wright there would mis-rank N²<0 convection.
            # The run's gravity, not the library's. The pressure this density
            # is built on is linear in it, and the NEMO comparison cards pin a
            # value that differs from the module default.
            rho = compute_ocean_rho(state, z_coord, jacobian, eos_fn=eos_fn,
                                    g=config.constants.g)
            # Tracer-only on MPAS: the convective **momentum** viscosity
            # (cfg_c.nu_conv / convective_νz) is intentionally NOT applied
            # here.  MPAS carries edge-normal velocity (nEdges) whose
            # vertical mixing needs a TRiSK cell->edge reconstruction; the
            # cell-centred enhanced_diffusion momentum operator does not map
            # onto it.  Omitting u/v takes the kernel's tracer-only branch
            # (du/dv = None).  Convective momentum on MPAS is a separate
            # follow-up (like its KPP edge-momentum path); nu_conv only
            # affects the lat-lon / cubed-sphere cell-centred grids.
            # ``n2_mode='adiabatic'`` computes the true static-stability
            # trigger from T/S/p_cell via the EOS, so it needs cell-centre
            # pressure; compute it ONLY on that path (default 'insitu' path
            # stays byte-identical — no extra pressure solve).
            if getattr(cfg_c, "n2_mode", "insitu") == "adiabatic":
                from legoesm.ocean.eos import compute_ocean_rho_and_pressure
                _, p_cell = compute_ocean_rho_and_pressure(
                    state, z_coord, jacobian, eos_fn=eos_fn,
                    g=config.constants.g,
                )
                c_out = enhanced_diffusion_convection(
                    state.T.data, state.S.data, rho, z_coord, jacobian, cfg_c,
                    p_cell=p_cell, eos_fn=eos_fn,
                    eta=state.eta.data, H_bathy=state.H_bathy.data,
                )
            else:
                c_out = enhanced_diffusion_convection(
                    state.T.data, state.S.data, rho, z_coord, jacobian, cfg_c,
                    eta=state.eta.data, H_bathy=state.H_bathy.data,
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
