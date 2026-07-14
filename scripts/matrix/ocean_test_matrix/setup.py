"""Ocean grid setup and model creation for the ocean test matrix."""

from __future__ import annotations

import numpy as np

from ocean_test_matrix import config


def _parse_resolution(tc):
    """Parse resolution string and return grid-appropriate parameters."""
    if tc.grid_type == "cubed_sphere":
        return {"n": int(tc.resolution[1:])}
    elif tc.grid_type == "latlon":
        parts = tc.resolution.split("x")
        return {"n_lat": int(parts[0]), "n_lon": int(parts[1])}
    elif tc.grid_type == "mpas":
        return {"level": int(tc.resolution.replace("ico", ""))}
    elif tc.grid_type == "mpas_regional":
        return {"resolution_km": int(tc.resolution.replace("km", ""))}
    elif tc.grid_type == "latlon_regional":
        parts = tc.resolution.split("x")
        return {"n_lat": int(parts[0]), "n_lon": int(parts[1])}
    elif tc.grid_type == "cs_regional":
        return {"n": int(tc.resolution[1:])}
    elif tc.grid_type == "latlon_channel":
        parts = tc.resolution.split("x")
        return {"n_lat": int(parts[0]), "n_lon": int(parts[1])}
    elif tc.grid_type == "mpas_channel":
        return {"resolution_km": int(tc.resolution.replace("km", ""))}
    elif tc.grid_type == "spectral":
        return {"truncation": int(tc.resolution[1:])}
    raise ValueError(f"Unknown grid type: {tc.grid_type}")


# ---------------------------------------------------------------------------
# Cube ocean cold-start stabilization (shared by BOTH matrix drivers)
# ---------------------------------------------------------------------------
# The cd-grid Arakawa-Lamb corner stencil's face-edge PGF amplification under
# horizontal density gradients is the documented cube cold-start gate
# (lock_exchange, phillips_two_layer, overflow, geostrophic_adjustment,
# stommel_gyre_tracer).  The DOMINANT stabilizer is the barotropic substep
# count: overflow blows up to NaN at 30 substeps but is stable at 60 — a
# panel-edge barotropic gravity-wave CFL limit (raising A_h/K_h alone or the
# conservation fixer alone does NOT rescue it; measured 2026-06-14).  Raised
# A_h/K_h additionally damp the slow panel-edge tracer overshoot for the
# matrix smoke.  ``cube_light_diffusion`` keeps lateral diffusion light for
# wave tests (barotropic_wave / inertia_gravity_wave carry no density gradient;
# K_h=5e6 would decay a 0.1 m wave to 0.04 m).  The a_grid barotropic is
# forbidden by the never-A-grid / FV3-faithfulness directive — use fv3sw
# (vector-invariant absolute-vorticity flux + RK3 + div-damp/hyperdiff).
#
# This block is the single source of truth: ``run_ocean_test_matrix.py``'s
# local cube ``_create_ocean_setup`` imports ``cube_matrix_ocean_config_kwargs``
# so the monolithic and modular drivers cannot drift apart again.
_CUBE_MATRIX_N_BAROTROPIC_SUBSTEPS = 60
_CUBE_MATRIX_BAROTROPIC_DIFFUSION_ALPHA = 0.3
_CUBE_MATRIX_A_H = 5.0e5    # [m^2/s] raised horizontal viscosity
_CUBE_MATRIX_K_H = 5.0e6    # [m^2/s] raised horizontal tracer diffusivity


def cube_matrix_ocean_config_kwargs(*, physics, A_h=None, A_v=None, K_h=None,
                                    bottom_drag_r=None,
                                    cube_light_diffusion=False):
    """Build the cube ``OceanConfig`` kwargs for the test matrix.

    Heavy mode (default): 60 barotropic substeps, raised A_h/K_h, conservation
    fixer on.  ``cube_light_diffusion=True``: same substeps/fixer but lateral
    diffusion only when explicitly requested (wave tests).  An explicit ``K_h``
    overrides the raised default (e.g. a GM-handled experiment passing K_h=0).
    """
    kw = dict(
        n_barotropic_substeps=_CUBE_MATRIX_N_BAROTROPIC_SUBSTEPS,
        barotropic_diffusion_alpha=_CUBE_MATRIX_BAROTROPIC_DIFFUSION_ALPHA,
        use_conservation_fixer=True,
        physics=physics,
        barotropic_staggering="fv3sw",
    )
    if cube_light_diffusion:
        if A_h is not None:
            kw["A_h"] = A_h
        if K_h is not None:
            kw["K_h"] = K_h
    elif A_h is None:
        kw["A_h"] = _CUBE_MATRIX_A_H
        kw["K_h"] = K_h if K_h is not None else _CUBE_MATRIX_K_H
    else:
        kw["A_h"] = max(A_h, _CUBE_MATRIX_A_H)
        kw["K_h"] = K_h if K_h is not None else _CUBE_MATRIX_K_H
    if A_v is not None:
        kw["A_v"] = A_v
    if bottom_drag_r is not None:
        kw["bottom_drag_r"] = bottom_drag_r
    return kw


def _create_ocean_setup(tc, nlev: int | None = None,
                        H_max: float | None = None, physics=None,
                        A_h: float | None = None,
                        B_h: float | None = None,
                        C_smag: float | None = None,
                        A_v: float | None = None,
                        K_h: float | None = None,
                        K_v: float | None = None,
                        K_bih: float | None = None,
                        bottom_drag_r: float | None = None,
                        eos: str | None = None,
                        eos_linear=None,
                        barotropic_diffusion_alpha: float | None = None,
                        barotropic_div_damp: float | None = None,
                        tracer_advection: str | None = None,
                        gm_redi=None,
                        pv_scheme: str | None = None,
                        apvm_dt: float | None = None,
                        pv_alpha: float | None = None,
                        K_zeta_bih: float | None = None,
                        C_leith: float | None = None,
                        C_leith_modified: bool | None = None,
                        momentum_advection: str | None = None,
                        weno_d_term: bool | None = None,
                        barotropic_solver: str | None = None,
                        cube_light_diffusion: bool = False,
                        model_config=None):
    """Create grid, z_coord, and rest-state for any grid type.

    Parameters
    ----------
    tc : TestCase
    nlev : int
    H_max : float
    physics : OceanPhysicsConfig or None
        If provided, passed to the model config to enable physics
        (e.g. prescribed surface forcing for wind-driven experiments).
    A_h : float or None
        Override horizontal viscosity [m^2/s]. If None, uses config default.
    model_config : *OceanConfig or None
        Pre-built model config to use VERBATIM instead of assembling one from
        the scraped scalar kwargs above. This is how an experiment that exposes
        a recipe factory (``EXPERIMENT_CONFIG["create_model_config"]``) makes the
        matrix test the SAME recipe its production driver runs — the field-by-
        field scrape above silently drops K_v / bottom_drag / eos / gm_redi on
        the lat-lon path, so the factory path is strictly more faithful. Only
        the ``latlon``/``mpas``/``latlon_channel`` branches honour it (raises
        otherwise).

    Returns (grid, z_coord, config, model, coord_kind, lon_deg, lat_deg).
    """
    _MODEL_CONFIG_GRIDS = ("latlon", "mpas", "latlon_channel")
    if model_config is not None and tc.grid_type not in _MODEL_CONFIG_GRIDS:
        raise NotImplementedError(
            "model_config injection is only wired for grid_type in "
            f"{set(_MODEL_CONFIG_GRIDS)}, got {tc.grid_type!r}")
    if nlev is None:
        nlev = config.DEFAULT_NLEV
    if H_max is None:
        H_max = config.DEFAULT_H_MAX
    from legoesm.ocean.vertical import create_ocean_z_star

    z_coord = create_ocean_z_star(n_levels=nlev, H_max=H_max)
    params = _parse_resolution(tc)

    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.state import OceanConfig
        # Register the FV3 SW barotropic provider (fv3sw) — the cube uses the
        # FV3-faithful C-D barotropic, not the forbidden a_grid (never-A-grid).
        import legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid  # noqa: F401

        # Phase B.1 parity with the monolithic runner: cubed_sphere
        # ``OceanConfig`` now exposes ``bottom_drag_r`` and the cd-grid backend
        # (``ocean_baroclinic_tendencies_cdgrid``) applies linear / quadratic /
        # BBL bottom drag the same way the lat-lon C-grid does. The earlier
        # NotImplementedError gate (modular iter-175 / monolithic iter-174) is
        # removed; the kwarg is plumbed straight through, so the modular and
        # monolithic paths agree (no silent drop, no spurious SKIP).
        n = params["n"]
        grid = create_cubed_sphere(n)
        # Cube cold-start stabilization (barotropic substeps=60 etc.) lives in
        # the shared ``cube_matrix_ocean_config_kwargs`` so this modular driver
        # and the monolithic ``run_ocean_test_matrix.py`` cannot diverge: the
        # previous inline ``n_barotropic_substeps=30`` here blew overflow up to
        # NaN in ~6 steps while the monolithic ran stably at 60.
        kw = cube_matrix_ocean_config_kwargs(
            physics=physics, A_h=A_h, A_v=A_v, K_h=K_h,
            bottom_drag_r=bottom_drag_r,
            cube_light_diffusion=cube_light_diffusion)
        cfg = OceanConfig(**kw)
        model = OceanModel(grid, z_coord, cfg)
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, cfg, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        from legoesm.ocean.state import LatLonCGridOceanConfig

        grid = create_latlon_grid(params["n_lat"], params["n_lon"])
        if model_config is not None:
            cfg = model_config
        else:
            kw = dict(n_barotropic_substeps=30, physics=physics)
            if A_h is not None:
                kw["A_h"] = A_h
            if A_v is not None:
                kw["A_v"] = A_v
            cfg = LatLonCGridOceanConfig.from_flat(**kw)
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, cfg, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "mpas":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig

        mesh = create_voronoi_mesh(params["level"])
        if model_config is not None:
            cfg = model_config
            model = MPASOceanModel(mesh, z_coord, cfg)
            coord_kind = "mpas"
            lon_deg = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
            lat_deg = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi
            return mesh, z_coord, cfg, model, coord_kind, lon_deg, lat_deg
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if B_h is not None:
            kw["B_h"] = B_h
        if C_smag is not None:
            kw["C_smag"] = C_smag
        if A_v is not None:
            kw["A_v"] = A_v
        if K_v is not None:
            kw["K_v"] = K_v
        if K_h is not None:
            kw["K_h"] = K_h
        if K_bih is not None:
            kw["K_bih"] = K_bih
        if bottom_drag_r is not None:
            kw["bottom_drag_r"] = bottom_drag_r
        if eos is not None:
            kw["eos"] = eos
        if eos_linear is not None:
            kw["eos_linear"] = eos_linear
        if barotropic_diffusion_alpha is not None:
            kw["barotropic_diffusion_alpha"] = barotropic_diffusion_alpha
        if barotropic_div_damp is not None:
            kw["barotropic_div_damp"] = barotropic_div_damp
        if tracer_advection is not None:
            kw["tracer_advection"] = tracer_advection
        if pv_scheme is not None:
            kw["pv_scheme"] = pv_scheme
        if apvm_dt is not None:
            kw["apvm_dt"] = apvm_dt
        if pv_alpha is not None:
            kw["pv_alpha"] = pv_alpha
        if K_zeta_bih is not None:
            kw["K_zeta_bih"] = K_zeta_bih
        if C_leith is not None:
            kw["C_leith"] = C_leith
        if C_leith_modified is not None:
            kw["C_leith_modified"] = C_leith_modified
        cfg = MPASOceanConfig(**kw)
        model = MPASOceanModel(mesh, z_coord, cfg)
        coord_kind = "mpas"
        lon_deg = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi
        return mesh, z_coord, cfg, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "mpas_regional":
        from legoesm.grids.voronoi import create_regional_voronoi_mesh
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig

        res_km = params["resolution_km"]
        # Default gyre basin bounds
        lon_w = tc.run_kwargs.get("lon_west", 0.0)
        lon_e = tc.run_kwargs.get("lon_east", 120.0)
        lat_s = tc.run_kwargs.get("lat_south", 15.0)
        lat_n = tc.run_kwargs.get("lat_north", 75.0)
        mesh = create_regional_voronoi_mesh(
            (lon_w, lon_e), (lat_s, lat_n), resolution_km=res_km)
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if B_h is not None:
            kw["B_h"] = B_h
        if C_smag is not None:
            kw["C_smag"] = C_smag
        if A_v is not None:
            kw["A_v"] = A_v
        if K_v is not None:
            kw["K_v"] = K_v
        if K_h is not None:
            kw["K_h"] = K_h
        if K_bih is not None:
            kw["K_bih"] = K_bih
        if bottom_drag_r is not None:
            kw["bottom_drag_r"] = bottom_drag_r
        if eos is not None:
            kw["eos"] = eos
        if eos_linear is not None:
            kw["eos_linear"] = eos_linear
        if barotropic_diffusion_alpha is not None:
            kw["barotropic_diffusion_alpha"] = barotropic_diffusion_alpha
        if barotropic_div_damp is not None:
            kw["barotropic_div_damp"] = barotropic_div_damp
        if tracer_advection is not None:
            kw["tracer_advection"] = tracer_advection
        if pv_scheme is not None:
            kw["pv_scheme"] = pv_scheme
        if apvm_dt is not None:
            kw["apvm_dt"] = apvm_dt
        if pv_alpha is not None:
            kw["pv_alpha"] = pv_alpha
        if K_zeta_bih is not None:
            kw["K_zeta_bih"] = K_zeta_bih
        if C_leith is not None:
            kw["C_leith"] = C_leith
        if C_leith_modified is not None:
            kw["C_leith_modified"] = C_leith_modified
        # pv_scheme defaults to "enstrophy" in MPASOceanConfig.
        # apvm_dt left at 0 (disabled); see mpas_channel branch notes.
        cfg = MPASOceanConfig(**kw)
        model = MPASOceanModel(mesh, z_coord, cfg)
        coord_kind = "mpas"
        lon_deg = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi
        return mesh, z_coord, cfg, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "latlon_regional":
        from legoesm.grids.latlon import create_regional_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        from legoesm.ocean.state import LatLonCGridOceanConfig

        n_lat, n_lon = params["n_lat"], params["n_lon"]
        lon_w = tc.run_kwargs.get("lon_west", 0.0)
        lon_e = tc.run_kwargs.get("lon_east", 120.0)
        lat_s = tc.run_kwargs.get("lat_south", 15.0)
        lat_n = tc.run_kwargs.get("lat_north", 75.0)
        grid, wall_mask = create_regional_latlon_grid(
            n_lat, n_lon, lat_s, lat_n, lon_w, lon_e)
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if A_v is not None:
            kw["A_v"] = A_v
        cfg = LatLonCGridOceanConfig.from_flat(**kw)
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, cfg, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "latlon_channel":
        from legoesm.grids.latlon import create_regional_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        from legoesm.ocean.state import LatLonCGridOceanConfig

        n_lat, n_lon = params["n_lat"], params["n_lon"]
        lat_s = tc.run_kwargs.get("lat_south", 25.0)
        lat_n = tc.run_kwargs.get("lat_north", 65.0)
        lon_w = tc.run_kwargs.get("lon_west", 0.0)
        lon_e = tc.run_kwargs.get("lon_east", 360.0)
        grid, wall_mask = create_regional_latlon_grid(
            n_lat, n_lon, lat_s, lat_n,
            lon_west=lon_w, lon_east=lon_e, periodic_x=True)
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if B_h is not None:
            kw["B_h"] = B_h
        if C_smag is not None:
            kw["C_smag"] = C_smag
        if A_v is not None:
            kw["A_v"] = A_v
        if K_v is not None:
            kw["K_v"] = K_v
        if K_h is not None:
            kw["K_h"] = K_h
        if K_bih is not None:
            kw["K_bih"] = K_bih
        if bottom_drag_r is not None:
            kw["bottom_drag_r"] = bottom_drag_r
        if eos is not None:
            kw["eos"] = eos
        if eos_linear is not None:
            kw["eos_linear"] = eos_linear
        if barotropic_diffusion_alpha is not None:
            kw["barotropic_diffusion_alpha"] = barotropic_diffusion_alpha
        if barotropic_div_damp is not None:
            kw["barotropic_div_damp"] = barotropic_div_damp
        if tracer_advection is not None:
            kw["tracer_advection"] = tracer_advection
        if gm_redi is not None:
            kw["gm_redi"] = gm_redi
        if momentum_advection is not None:
            kw["momentum_advection"] = momentum_advection
        if weno_d_term is not None:
            kw["weno_d_term"] = weno_d_term
        if barotropic_solver is not None:
            kw["barotropic_solver"] = barotropic_solver
        cfg = model_config if model_config is not None \
            else LatLonCGridOceanConfig.from_flat(**kw)
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, cfg, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "mpas_channel":
        from legoesm.grids.voronoi import create_regional_voronoi_mesh
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig

        res_km = params["resolution_km"]
        lat_s = tc.run_kwargs.get("lat_south", 25.0)
        lat_n = tc.run_kwargs.get("lat_north", 65.0)
        # Periodic-x zonal extent: defaults to the full 360° (legacy) but
        # can be overridden via run_kwargs to match a lat-lon channel
        # (e.g. the Eady lon_west/lon_east bounds).
        lon_w = tc.run_kwargs.get("lon_west", 0.0)
        lon_e = tc.run_kwargs.get("lon_east", 360.0)
        mesh = create_regional_voronoi_mesh(
            (lon_w, lon_e), (lat_s, lat_n), resolution_km=res_km,
            periodic_x=True)
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if B_h is not None:
            kw["B_h"] = B_h
        if C_smag is not None:
            kw["C_smag"] = C_smag
        if A_v is not None:
            kw["A_v"] = A_v
        if K_v is not None:
            kw["K_v"] = K_v
        if K_h is not None:
            kw["K_h"] = K_h
        if K_bih is not None:
            kw["K_bih"] = K_bih
        if bottom_drag_r is not None:
            kw["bottom_drag_r"] = bottom_drag_r
        if eos is not None:
            kw["eos"] = eos
        if eos_linear is not None:
            kw["eos_linear"] = eos_linear
        if barotropic_diffusion_alpha is not None:
            kw["barotropic_diffusion_alpha"] = barotropic_diffusion_alpha
        if barotropic_div_damp is not None:
            kw["barotropic_div_damp"] = barotropic_div_damp
        if tracer_advection is not None:
            kw["tracer_advection"] = tracer_advection
        if pv_scheme is not None:
            kw["pv_scheme"] = pv_scheme
        if apvm_dt is not None:
            kw["apvm_dt"] = apvm_dt
        if pv_alpha is not None:
            kw["pv_alpha"] = pv_alpha
        if K_zeta_bih is not None:
            kw["K_zeta_bih"] = K_zeta_bih
        if C_leith is not None:
            kw["C_leith"] = C_leith
        if C_leith_modified is not None:
            kw["C_leith_modified"] = C_leith_modified
        if gm_redi is not None:
            kw["gm_redi"] = gm_redi
        # pv_scheme defaults to "enstrophy" in MPASOceanConfig — suppresses
        # the ζ-checkerboard null mode of the energy-conserving scheme.
        # APVM is left disabled (``apvm_dt=0``); enabling it on top of
        # enstrophy was found to *destabilise* Eady channel simulations.
        cfg = MPASOceanConfig(**kw)
        model = MPASOceanModel(mesh, z_coord, cfg)
        coord_kind = "mpas"
        lon_deg = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi
        return mesh, z_coord, cfg, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "cs_regional":
        from legoesm.grids.cubed_sphere import create_cubed_sphere_panel
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.state import OceanConfig

        n = params["n"]
        grid = create_cubed_sphere_panel(n, face_id=0, return_cdgrid=False)
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if A_v is not None:
            kw["A_v"] = A_v
        cfg = OceanConfig(**kw)
        model = OceanModel(grid, z_coord, cfg)
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, cfg, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "spectral":
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel
        from legoesm.ocean.state import SpectralOceanConfig

        trunc = params["truncation"]
        grid = create_gaussian_grid(trunc)
        cfg = SpectralOceanConfig()
        model = SpectralOceanModel(grid, z_coord, cfg)
        coord_kind = "gaussian"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, cfg, model, coord_kind, lon_deg, lat_deg

    raise ValueError(f"Unknown grid type: {tc.grid_type}")
