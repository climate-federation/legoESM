"""Field extraction, check, and scalar diagnostic helpers for the ocean test matrix."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ocean_test_matrix.timeloop import check_finite
from ocean_test_matrix.regridding import _bin_to_latlon


# ===========================================================================
# Field extraction helpers
# ===========================================================================

def _extract_fv_ocean(state, grid_type: str, include_velocity_3d: bool = False):
    """Extract snapshot fields for cubed-sphere or lat-lon ocean state."""
    eta = np.asarray(state.eta.data, dtype=np.float64)
    T_3d = np.asarray(state.T.data, dtype=np.float64)
    S_3d = np.asarray(state.S.data, dtype=np.float64)
    u_sfc = np.asarray(state.u.data[..., 0], dtype=np.float64)
    result = {
        "eta": eta,
        "SST": np.asarray(state.T.data[..., 0], dtype=np.float64),
        "SSS": np.asarray(state.S.data[..., 0], dtype=np.float64),
        "u_sfc": u_sfc,
        "T_3d": T_3d,
        "S_3d": S_3d,
        "land_mask": np.asarray(state.land_mask.data, dtype=np.float64),
    }
    if hasattr(state, "v"):
        v_sfc = np.asarray(state.v.data[..., 0], dtype=np.float64)
        result["v_sfc"] = v_sfc
        # C-grid: u and v have different shapes; skip speed_sfc
        if u_sfc.shape == v_sfc.shape:
            result["speed_sfc"] = np.sqrt(u_sfc ** 2 + v_sfc ** 2)
    if include_velocity_3d:
        u_raw = np.asarray(state.u.data, dtype=np.float64)
        if hasattr(state, "v"):
            v_raw = np.asarray(state.v.data, dtype=np.float64)
            # C-grid: interpolate staggered u/v to cell centers
            if u_raw.shape[:-1] != v_raw.shape[:-1]:
                u_cc = 0.5 * (u_raw[:, :-1, :] + u_raw[:, 1:, :])
                v_cc = 0.5 * (v_raw[:-1, :, :] + v_raw[1:, :, :])
            else:
                u_cc = u_raw
                v_cc = v_raw
            result["u_3d"] = u_cc
            result["v_3d"] = v_cc
            result["speed_3d"] = np.sqrt(u_cc**2 + v_cc**2)
        else:
            result["u_3d"] = u_raw
        # Add vertical velocity if available
        if hasattr(state, "w"):
            w_3d = np.asarray(state.w.data, dtype=np.float64)
            result["w_3d"] = w_3d
            # Extract vertical velocity below Ekman layer (level 1 = 133.9m depth)
            result["w_133m"] = w_3d[..., 1]
            # Also keep surface for comparison if needed
            result["w_sfc"] = w_3d[..., 0]

    # Create surface speed field by interpolating u,v to common grid
    if "u_sfc" in result and "v_sfc" in result:
        u_sfc = result["u_sfc"]
        v_sfc = result["v_sfc"]
        # For C-grid: interpolate staggered velocities to cell centers for speed
        if u_sfc.shape != v_sfc.shape:
            # u is on east-west faces, v on north-south faces
            # Interpolate both to cell centers
            u_cc = 0.5 * (u_sfc[:, :-1] + u_sfc[:, 1:])  # avg in lon direction
            v_cc = 0.5 * (v_sfc[:-1, :] + v_sfc[1:, :])  # avg in lat direction
            # Make sure they have same shape (min of both)
            ny_min = min(u_cc.shape[0], v_cc.shape[0])
            nx_min = min(u_cc.shape[1], v_cc.shape[1])
            u_final = u_cc[:ny_min, :nx_min]
            v_final = v_cc[:ny_min, :nx_min]
            result["speed_sfc"] = np.sqrt(u_final**2 + v_final**2)
            # Replace staggered u/v with cell-center values for quiver plots
            result["u_sfc"] = u_final
            result["v_sfc"] = v_final
    return result


def _extract_mpas_ocean(state, lon_deg, lat_deg, mesh=None,
                        include_velocity_3d: bool = False):
    """Extract snapshot fields for MPAS ocean state.

    Returns raw cell-center arrays; regridding to lat-lon is done by the
    plotting functions via _regrid_2d / _regrid_3d_level.

    When *include_velocity_3d* is True and *mesh* is provided, reconstructs
    cell-center (u_east, v_north) at all levels via Perot reconstruction.
    """
    eta = np.asarray(state.eta.data, dtype=np.float64)
    T_3d = np.asarray(state.T.data, dtype=np.float64)
    S_3d = np.asarray(state.S.data, dtype=np.float64)
    SST = T_3d[..., 0] if T_3d.ndim >= 2 else T_3d

    result = {
        "eta": eta,
        "SST": SST,
        "T_3d": T_3d,
        "S_3d": S_3d,
        "land_mask": np.asarray(state.land_mask.data, dtype=np.float64),
    }
    if mesh is not None:
        # SURFACE SPEED, ALWAYS -- see the identical block in the monolithic
        # runner. Without it this arm writes no cell-shaped velocity at all
        # and the cross-grid velocity row reads as agreement rather than as
        # "never compared". Level 0 only; the per-level loop below stays
        # behind include_velocity_3d because that is what costs.
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity as _rcv
        _ue0, _vn0 = _rcv(state.u.data[:, 0], mesh)
        result["speed_sfc"] = np.sqrt(
            np.asarray(_ue0, dtype=np.float64) ** 2
            + np.asarray(_vn0, dtype=np.float64) ** 2)
    if include_velocity_3d and mesh is not None:
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity
        u_edge = np.asarray(state.u.data, dtype=np.float64)  # (nEdges, nlev)
        nlev = u_edge.shape[-1]
        nCells = eta.shape[0]
        u_cc = np.zeros((nCells, nlev), dtype=np.float64)
        v_cc = np.zeros((nCells, nlev), dtype=np.float64)
        for k in range(nlev):
            ue, vn = reconstruct_cell_velocity(state.u.data[:, k], mesh)
            u_cc[:, k] = np.asarray(ue, dtype=np.float64)
            v_cc[:, k] = np.asarray(vn, dtype=np.float64)
        result["u_3d"] = u_cc
        result["v_3d"] = v_cc
        result["speed_3d"] = np.sqrt(u_cc**2 + v_cc**2)
        # Reuse level 0 rather than reconstructing it a second time: the
        # surface block above already did this solve, and the loop has now
        # redone it at k=0 (codex 2026-08-13). Same value, one less Perot
        # reconstruction per 3-D snapshot.
        result["speed_sfc"] = result["speed_3d"][..., 0]

        # Add vertical velocity if available (for future MPAS implementation)
        if hasattr(state, "w"):
            w_3d = np.asarray(state.w.data, dtype=np.float64)
            result["w_3d"] = w_3d
            # Extract vertical velocity below Ekman layer (level 1 = 133.9m depth)
            result["w_133m"] = w_3d[..., 1]
            # Also keep surface for comparison if needed
            result["w_sfc"] = w_3d[..., 0]
    return result


def _extract_spectral_ocean(state, grid):
    """Extract snapshot fields for spectral ocean state."""
    from legoesm.grids.gaussian import sh_synthesis, sh_synthesis_3d

    eta_grid = np.asarray(
        sh_synthesis(grid, state.eta_hat.data), dtype=np.float64)
    T_grid = np.asarray(
        sh_synthesis_3d(grid, state.T_hat.data), dtype=np.float64)
    S_grid = np.asarray(
        sh_synthesis_3d(grid, state.S_hat.data), dtype=np.float64)
    SST = T_grid[..., 0] if T_grid.ndim >= 3 else T_grid
    SSS = S_grid[..., 0] if S_grid.ndim >= 3 else S_grid

    # For spectral grids, synthesize land mask if available (typically all ocean for spectral)
    if hasattr(state, 'land_mask') and hasattr(state.land_mask, 'data'):
        land_mask_grid = np.asarray(
            sh_synthesis(grid, state.land_mask.data), dtype=np.float64)
    else:
        # Default to all ocean for spectral grids (consistent with rest_state config)
        land_mask_grid = np.ones_like(eta_grid, dtype=np.float64)

    return {
        "eta": eta_grid,
        "SST": SST,
        "SSS": SSS,
        "T_3d": T_grid,
        "S_3d": S_grid,
        "land_mask": land_mask_grid,
    }


# ===========================================================================
# Scalar / check functions per grid type
# ===========================================================================

def _make_check_fn(grid_type: str):
    """Return a check_fn(state) -> (is_finite, metric)."""
    if grid_type == "spectral":
        def check_fn(s):
            eta_max = float(jnp.max(jnp.abs(s.eta_hat.data)))
            fin = (bool(jnp.all(jnp.isfinite(s.eta_hat.data))) and
                   bool(jnp.all(jnp.isfinite(s.T_hat.data))))
            return fin, eta_max
        return check_fn
    elif grid_type in ("mpas", "mpas_regional", "mpas_channel"):
        def check_fn(s):
            fin = check_finite({"eta": s.eta.data, "T": s.T.data,
                                "u": s.u.data})
            metric = float(jnp.max(jnp.abs(s.eta.data)))
            return fin, metric
        return check_fn
    else:
        def check_fn(s):
            fin = check_finite({"eta": s.eta.data, "T": s.T.data,
                                "u": s.u.data})
            metric = float(jnp.max(jnp.abs(s.eta.data)))
            return fin, metric
        return check_fn


def _make_scalar_fn(grid_type: str, grid=None, z_coord=None):
    """Return a scalar_fn(state) -> dict for diagnostics.

    When *z_coord* is provided, mean_T and mean_S are volume-weighted
    (sum(T * dz * area * mask) / sum(dz * area * mask)).  Without it
    the diagnostic falls back to an unweighted nanmean, which drifts
    spuriously when vertical diffusion redistributes heat across layers
    of different thickness.
    """
    if grid_type == "spectral":
        if grid is None:
            raise ValueError("Grid object required for spectral scalar function")

        def scalar_fn(s):
            from legoesm.grids.gaussian import sh_synthesis, sh_synthesis_3d

            # Convert spectral coefficients to physical fields
            eta_phys = sh_synthesis(grid, s.eta_hat.data)        # meters
            T_phys = sh_synthesis_3d(grid, s.T_hat.data)         # °C
            S_phys = sh_synthesis_3d(grid, s.S_hat.data)         # PSU

            # Apply ocean masking if available
            if hasattr(s, 'land_mask') and hasattr(s.land_mask, 'data'):
                land_mask_phys = sh_synthesis(grid, s.land_mask.data)
                ocean_mask = land_mask_phys > 0.5

                # Compute ocean-only statistics
                eta_ocean = jnp.where(ocean_mask, eta_phys, jnp.nan)
                T_ocean = jnp.where(ocean_mask, T_phys, jnp.nan)
                S_ocean = jnp.where(ocean_mask, S_phys, jnp.nan)

                return {
                    "mean_eta": float(jnp.nanmean(eta_ocean)),           # meters
                    "max_abs_eta": float(jnp.nanmax(jnp.abs(eta_ocean))), # meters
                    "mean_T": float(jnp.nanmean(T_ocean)),               # °C
                    "mean_S": float(jnp.nanmean(S_ocean)),               # PSU
                }
            else:
                # Fallback for spectral (typically all ocean anyway)
                return {
                    "mean_eta": float(jnp.mean(eta_phys)),           # meters
                    "max_abs_eta": float(jnp.max(jnp.abs(eta_phys))), # meters
                    "mean_T": float(jnp.mean(T_phys)),               # °C
                    "mean_S": float(jnp.mean(S_phys)),               # PSU
                }
        return scalar_fn
    elif grid_type in ("mpas", "mpas_regional", "mpas_channel"):
        # Capture z_coord layer thicknesses and cell areas for
        # volume-weighted diagnostics.
        # Use actual h_k (which depends on eta) rather than reference dz_ref,
        # so that the diagnostic tracks the true conserved quantity h*T.
        _area = grid.areaCell if grid is not None else None
        _z_coord = z_coord
        _min_wc = 0.5  # default min_water_column_m

        def scalar_fn(s):
            if hasattr(s, 'land_mask') and hasattr(s.land_mask, 'data'):
                mask = s.land_mask.data > 0.5

                # Area-weighted eta mean (nanmean is biased on non-uniform grids)
                if _area is not None:
                    area_ocean = _area * mask.astype(_area.dtype)
                    mean_eta = float(jnp.sum(s.eta.data * area_ocean) / jnp.maximum(jnp.sum(area_ocean), 1e-30))
                else:
                    eta_ocean = jnp.where(mask, s.eta.data, jnp.nan)
                    mean_eta = float(jnp.nanmean(eta_ocean))
                eta_ocean = jnp.where(mask, s.eta.data, jnp.nan)

                # Volume-weighted T and S using actual layer thickness h_k
                if _area is not None and _z_coord is not None:
                    from legoesm.ocean.vertical import compute_layer_thickness
                    h_k = compute_layer_thickness(
                        s.eta.data, s.H_bathy.data, _z_coord,
                        min_water_column_m=_min_wc)
                    vol = _area[:, None] * h_k * mask[:, None]
                    vol_sum = jnp.sum(vol)
                    mean_T = float(jnp.sum(s.T.data * vol) / vol_sum)
                    mean_S = float(jnp.sum(s.S.data * vol) / vol_sum)
                else:
                    T_ocean = jnp.where(mask[:, None], s.T.data, jnp.nan)
                    S_ocean = jnp.where(mask[:, None], s.S.data, jnp.nan)
                    mean_T = float(jnp.nanmean(T_ocean))
                    mean_S = float(jnp.nanmean(S_ocean))

                max_abs_u = float(jnp.max(jnp.abs(s.u.data)))

                # Reconstruct cell-center velocity for comparable speed diagnostic
                from legoesm.ocean.init_mpas import reconstruct_cell_velocity
                u_east, v_north = reconstruct_cell_velocity(s.u.data[:, 0], grid)
                speed_cell = jnp.sqrt(u_east**2 + v_north**2)
                max_speed = float(jnp.max(jnp.where(mask, speed_cell, 0.0)))

                # Volume-weighted KE: 0.5 * sum(speed^2 * h_k * area) / sum(h_k * area)
                if _area is not None and _z_coord is not None:
                    from legoesm.ocean.vertical import compute_layer_thickness
                    h_k = compute_layer_thickness(
                        s.eta.data, s.H_bathy.data, _z_coord,
                        min_water_column_m=_min_wc)
                    nlev = h_k.shape[-1]
                    # Reconstruct cell speed at all levels
                    ke_sum = jnp.zeros(())
                    vol_ke = jnp.zeros(())
                    for lev in range(nlev):
                        ue, vn = reconstruct_cell_velocity(s.u.data[:, lev], grid)
                        spd2 = ue**2 + vn**2
                        cell_vol = _area * h_k[:, lev] * mask
                        ke_sum = ke_sum + jnp.sum(0.5 * spd2 * cell_vol)
                        vol_ke = vol_ke + jnp.sum(cell_vol)
                    mean_ke = float(ke_sum / jnp.maximum(vol_ke, 1e-10))
                else:
                    mean_ke = 0.0

                return {
                    "mean_eta": mean_eta,
                    "max_abs_eta": float(jnp.nanmax(jnp.abs(eta_ocean))),
                    "mean_T": mean_T,
                    "mean_S": mean_S,
                    "max_abs_u": max_abs_u,
                    "max_speed": max_speed,
                    "mean_ke": mean_ke,
                }
            else:
                max_abs_u = float(jnp.max(jnp.abs(s.u.data)))
                from legoesm.ocean.init_mpas import reconstruct_cell_velocity
                u_east, v_north = reconstruct_cell_velocity(s.u.data[:, 0], grid)
                max_speed = float(jnp.max(jnp.sqrt(u_east**2 + v_north**2)))
                return {
                    "mean_eta": float(jnp.mean(s.eta.data)),
                    "max_abs_eta": float(jnp.max(jnp.abs(s.eta.data))),
                    "mean_T": float(jnp.mean(s.T.data)),
                    "mean_S": float(jnp.mean(s.S.data)),
                    "max_abs_u": max_abs_u,
                    "max_speed": max_speed,
                }
        return scalar_fn
    else:
        # Cubed-sphere and lat-lon grids.
        # Capture grid area and z_coord for volume-weighted diagnostics.
        # Use actual h_k (which depends on eta) rather than reference dz_ref,
        # so that the diagnostic tracks the true conserved quantity h*T.
        _area = grid.area if (grid is not None and hasattr(grid, 'area')) else None
        _z_coord = z_coord
        _min_wc = 0.5  # default min_water_column_m

        def scalar_fn(s):
            if hasattr(s, 'land_mask') and hasattr(s.land_mask, 'data'):
                ocean_mask = s.land_mask.data > 0.5

                # Area-weighted eta mean (nanmean is biased on non-uniform grids)
                if _area is not None:
                    area_ocean = _area * ocean_mask.astype(_area.dtype)
                    mean_eta = float(jnp.sum(s.eta.data * area_ocean) / jnp.maximum(jnp.sum(area_ocean), 1e-30))
                else:
                    mean_eta = float(jnp.nanmean(jnp.where(ocean_mask, s.eta.data, jnp.nan)))
                eta_ocean = jnp.where(ocean_mask, s.eta.data, jnp.nan)

                # Volume-weighted T and S using actual layer thickness h_k
                if _area is not None and _z_coord is not None:
                    from legoesm.ocean.vertical import compute_layer_thickness
                    h_k = compute_layer_thickness(
                        s.eta.data, s.H_bathy.data, _z_coord,
                        min_water_column_m=_min_wc)
                    vol = _area[..., None] * h_k * ocean_mask[..., None]
                    vol_sum = jnp.sum(vol)
                    mean_T = float(jnp.sum(s.T.data * vol) / vol_sum)
                    mean_S = float(jnp.sum(s.S.data * vol) / vol_sum)
                else:
                    ocean_mask_3d = ocean_mask[..., jnp.newaxis]
                    T_ocean = jnp.where(ocean_mask_3d, s.T.data, jnp.nan)
                    S_ocean = jnp.where(ocean_mask_3d, s.S.data, jnp.nan)
                    mean_T = float(jnp.nanmean(T_ocean))
                    mean_S = float(jnp.nanmean(S_ocean))

                if hasattr(s, 'u') and hasattr(s, 'v'):
                    u_sfc = s.u.data[..., 0]
                    v_sfc = s.v.data[..., 0]
                    # C-grid: u (n_lat, n_lon+1) and v (n_lat+1, n_lon)
                    # can't compute sqrt(u^2+v^2) directly. Use max(|u|, |v|).
                    if u_sfc.shape != v_sfc.shape:
                        max_speed = float(jnp.maximum(
                            jnp.max(jnp.abs(u_sfc)),
                            jnp.max(jnp.abs(v_sfc))))
                    else:
                        speed_sfc = jnp.sqrt(u_sfc ** 2 + v_sfc ** 2)
                        speed_ocean = jnp.where(ocean_mask, speed_sfc, jnp.nan)
                        max_speed = float(jnp.nanmax(speed_ocean))
                else:
                    max_speed = 0.0

                # Volume-weighted KE
                mean_ke = 0.0
                if hasattr(s, 'u') and hasattr(s, 'v') and _area is not None and _z_coord is not None:
                    u = s.u.data; v = s.v.data
                    # Interpolate to cell centers for KE
                    if u.shape[:-1] != v.shape[:-1]:
                        # C-grid: u (n_lat, n_lon+1, nlev), v (n_lat+1, n_lon, nlev)
                        u_cc = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
                        v_cc = 0.5 * (v[:-1, :, :] + v[1:, :, :])
                    else:
                        u_cc = u; v_cc = v
                    spd2 = u_cc**2 + v_cc**2
                    ke_vol = 0.5 * spd2 * _area[..., None] * h_k * ocean_mask[..., None]
                    vol_total = jnp.sum(_area[..., None] * h_k * ocean_mask[..., None])
                    mean_ke = float(jnp.sum(ke_vol) / jnp.maximum(vol_total, 1e-10))

                return {
                    "mean_eta": mean_eta,
                    "max_abs_eta": float(jnp.nanmax(jnp.abs(eta_ocean))),
                    "mean_T": mean_T,
                    "mean_S": mean_S,
                    "max_speed": max_speed,
                    "mean_ke": mean_ke,
                }
            else:
                # Fallback without masking
                return {
                    "mean_eta": float(jnp.mean(s.eta.data)),
                    "max_abs_eta": float(jnp.max(jnp.abs(s.eta.data))),
                    "mean_T": float(jnp.mean(s.T.data)),
                    "mean_S": float(jnp.mean(s.S.data)),
                    "max_speed": float(jnp.maximum(
                        jnp.max(jnp.abs(s.u.data)),
                        jnp.max(jnp.abs(s.v.data)))),
                    "mean_ke": 0.0,
                }
        return scalar_fn


def _make_extract_fn(grid_type: str, grid, lon_deg, lat_deg,
                     include_velocity_3d: bool = False):
    """Return an extract_fn(state) -> dict for snapshots."""
    if grid_type == "spectral":
        def extract_fn(s):
            return _extract_spectral_ocean(s, grid)
        return extract_fn
    elif grid_type == "fesom":
        # FESOM state is a 1-D NODE vector, structurally like MPAS's 1-D cell
        # vector, so the MPAS extractor applies -- but only on the
        # include_velocity_3d=False path. The True path calls
        # reconstruct_cell_velocity(state.u, mesh), which reads MPAS edge
        # connectivity that a FesomOceanGrid does not have.
        if include_velocity_3d:
            raise NotImplementedError(
                "include_velocity_3d=True is not supported for grid_type "
                "'fesom': the MPAS Perot reconstruction needs edge "
                "connectivity the FESOM mesh does not expose. A dedicated "
                "FESOM extractor is required."
            )

        def extract_fn(s):
            return _extract_mpas_ocean(s, lon_deg, lat_deg, mesh=None,
                                       include_velocity_3d=False)
        return extract_fn
    elif grid_type in ("mpas", "mpas_regional", "mpas_channel"):
        # Always, so the cheap surface speed can be produced -- the same
        # fix as the monolithic runner, which this module had not received.
        _mesh = grid
        def extract_fn(s):
            return _extract_mpas_ocean(s, lon_deg, lat_deg, mesh=_mesh,
                                       include_velocity_3d=include_velocity_3d)
        return extract_fn
    else:
        def extract_fn(s):
            return _extract_fv_ocean(s, grid_type,
                                     include_velocity_3d=include_velocity_3d)
        return extract_fn


def _key_array_fn(state, grid_type: str):
    """Return a key array for jax.block_until_ready."""
    if grid_type == "spectral":
        return state.eta_hat.data
    return state.eta.data


# ===========================================================================
# Baroclinic gyre diagnostic helper
# ===========================================================================

def _make_baroclinic_scalar_fn(grid_type: str, grid=None, z_coord=None, config=None):
    """Enhanced scalar function for baroclinic gyre with N-S temperature gradient diagnostics."""
    import jax.numpy as jnp

    # Get base scalar function
    base_scalar_fn = _make_scalar_fn(grid_type, grid, z_coord)

    # Domain bounds for North-South analysis
    lat_south = config.lat_south if config else 15.0
    lat_north = config.lat_north if config else 75.0
    lat_center = (lat_south + lat_north) / 2.0

    if grid_type == "latlon_regional":
        # Get latitude coordinates
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

        # Find indices for north/south split
        center_idx = np.argmin(np.abs(lat_deg - lat_center))

        def scalar_fn(s):
            base_diag = base_scalar_fn(s)

            if hasattr(s, 'land_mask') and hasattr(s.land_mask, 'data'):
                ocean_mask = s.land_mask.data > 0.5

                # Compute North-South temperature differences at key levels
                T_data = s.T.data

                # Surface level (0) and thermocline level (4, ~681m depth)
                for level, level_name in [(0, 'surface'), (4, 'thermocline')]:
                    if level < T_data.shape[-1]:
                        T_level = T_data[..., level]
                        T_level_ocean = jnp.where(ocean_mask, T_level, jnp.nan)

                        # Split domain at center latitude
                        T_north = T_level_ocean[center_idx:, :]
                        T_south = T_level_ocean[:center_idx, :]

                        # Compute mean temperatures in each region
                        T_north_mean = jnp.nanmean(T_north)
                        T_south_mean = jnp.nanmean(T_south)

                        # North-South temperature difference (positive = north warmer)
                        dT_ns = T_north_mean - T_south_mean

                        # Add to diagnostics
                        base_diag[f"dT_ns_{level_name}"] = float(dT_ns)
                        base_diag[f"T_north_{level_name}"] = float(T_north_mean)
                        base_diag[f"T_south_{level_name}"] = float(T_south_mean)

                        # Spatial standard deviation (measure of baroclinic development)
                        spatial_std = jnp.nanstd(T_level_ocean)
                        base_diag[f"T_spatial_std_{level_name}"] = float(spatial_std)

            return base_diag

        return scalar_fn

    elif grid_type in ("mpas_regional", "mpas_channel"):
        # For MPAS, use cell latitude coordinates
        lat_deg = np.asarray(grid.latCell, dtype=np.float64) * 180 / np.pi

        # Find cells in north vs south regions
        north_mask = lat_deg >= lat_center
        south_mask = lat_deg < lat_center

        def scalar_fn(s):
            base_diag = base_scalar_fn(s)

            if hasattr(s, 'land_mask') and hasattr(s.land_mask, 'data'):
                ocean_mask = s.land_mask.data > 0.5

                # Compute North-South temperature differences at key levels
                T_data = s.T.data

                for level, level_name in [(0, 'surface'), (4, 'thermocline')]:
                    if level < T_data.shape[-1]:
                        T_level = T_data[..., level]
                        T_level_ocean = jnp.where(ocean_mask, T_level, jnp.nan)

                        # Extract north and south regions
                        T_north = jnp.where(north_mask & ocean_mask, T_level, jnp.nan)
                        T_south = jnp.where(south_mask & ocean_mask, T_level, jnp.nan)

                        # Compute mean temperatures in each region
                        T_north_mean = jnp.nanmean(T_north)
                        T_south_mean = jnp.nanmean(T_south)

                        # North-South temperature difference
                        dT_ns = T_north_mean - T_south_mean

                        # Add to diagnostics
                        base_diag[f"dT_ns_{level_name}"] = float(dT_ns)
                        base_diag[f"T_north_{level_name}"] = float(T_north_mean)
                        base_diag[f"T_south_{level_name}"] = float(T_south_mean)

                        # Spatial standard deviation
                        spatial_std = jnp.nanstd(T_level_ocean)
                        base_diag[f"T_spatial_std_{level_name}"] = float(spatial_std)

            return base_diag

        return scalar_fn

    else:
        # For other grids, just return the base function
        return base_scalar_fn
