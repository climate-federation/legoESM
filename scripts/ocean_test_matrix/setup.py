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


def _create_ocean_setup(tc, nlev: int | None = None,
                        H_max: float | None = None, physics=None,
                        A_h: float | None = None,
                        B_h: float | None = None,
                        A_v: float | None = None,
                        eos: str | None = None,
                        eos_linear=None,
                        barotropic_diffusion_alpha: float | None = None):
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

    Returns (grid, z_coord, config, model, coord_kind, lon_deg, lat_deg).
    """
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

        n = params["n"]
        grid = create_cubed_sphere(n)
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

    elif tc.grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        from legoesm.ocean.state import LatLonCGridOceanConfig

        grid = create_latlon_grid(params["n_lat"], params["n_lon"])
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if A_v is not None:
            kw["A_v"] = A_v
        cfg = LatLonCGridOceanConfig(**kw)
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
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if A_v is not None:
            kw["A_v"] = A_v
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
        if A_v is not None:
            kw["A_v"] = A_v
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
        cfg = LatLonCGridOceanConfig(**kw)
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
        grid, wall_mask = create_regional_latlon_grid(
            n_lat, n_lon, lat_s, lat_n, periodic_x=True)
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if B_h is not None:
            kw["B_h"] = B_h
        if A_v is not None:
            kw["A_v"] = A_v
        if eos is not None:
            kw["eos"] = eos
        if eos_linear is not None:
            kw["eos_linear"] = eos_linear
        if barotropic_diffusion_alpha is not None:
            kw["barotropic_diffusion_alpha"] = barotropic_diffusion_alpha
        cfg = LatLonCGridOceanConfig(**kw)
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
        mesh = create_regional_voronoi_mesh(
            (0, 360), (lat_s, lat_n), resolution_km=res_km,
            periodic_x=True)
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if A_v is not None:
            kw["A_v"] = A_v
        if eos is not None:
            kw["eos"] = eos
        if eos_linear is not None:
            kw["eos_linear"] = eos_linear
        if barotropic_diffusion_alpha is not None:
            kw["barotropic_diffusion_alpha"] = barotropic_diffusion_alpha
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
