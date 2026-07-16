"""Lat-lon C-grid FV ocean model with split-explicit barotropic/baroclinic stepping.

Arakawa C-grid staggering eliminates the 2*dx checkerboard null space
that plagues collocated A-grid formulations at moderate resolutions.

  u -> lon interfaces  (n_lat, n_lon+1, nlev)
  v -> lat interfaces  (n_lat+1, n_lon, nlev)
  eta, T, S -> cell centers (n_lat, n_lon [, nlev])

Uses the same split-explicit scheme as the A-grid LatLonOceanModel:
1. Compute 3D baroclinic tendencies (slow mode)
2. Update tracers (T, S) and 3D velocity with slow tendency
3. Run barotropic substeps (forward-backward) for free surface
4. Apply conservation fixers

Public API: state_new = model.step(state, dt)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field
from legoesm.core.precision import cast_pytree
from legoesm.grids.latlon import LatLonGrid, ensure_geometry
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    OceanPartialCellCoordinate,
    compute_layer_thickness,
    diagnose_w_from_flux_div,
    flux_form_vertical_tracer_advection,
    flux_form_vertical_tracer_advection_tvd,
)
from legoesm.ocean.state import (
    LatLonCGridOceanState,
    LatLonCGridOceanConfig,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
    _interp_to_v_points,
    _upwind_to_u_points,
    _upwind_to_v_points,
    _tvd_to_u_points,
    _tvd_to_v_points,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks,
    compute_face_masks_3d,
    divergence_cgrid,
    min_cell_to_uface,
    min_cell_to_vface,
)
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
)
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    barotropic_implicit_latlon_cgrid,
)
from legoesm.ocean.freshwater import freshwater_eta_tendency, virtual_salt_flux
from legoesm.ocean.physics.combined import make_ocean_physics
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    gm_redi_tracer_tendency_latlon,
)
from legoesm.ocean.advection_som import som_advect_tracers
from legoesm.ocean.advection import (
    fct_tracer_advection,
    ppm_to_u_points,
    ppm_to_v_points,
    flux_form_vertical_tracer_advection_ppm,
    dst3_to_u_points,
    dst3_to_v_points,
    flux_form_vertical_tracer_advection_dst3,
    multidim_tracer_advection,
    weno5_to_u_points,
    weno5_to_v_points,
    weno7_to_u_points,
    weno7_to_v_points,
    flux_form_vertical_tracer_advection_weno5,
    flux_form_vertical_tracer_advection_weno7,
)
from legoesm.ocean.conservation import ocean_conservation_fixer


# ---------------------------------------------------------------------------
# Advection flux-divergence helpers (extracted for AB2/RK3 reuse)
# ---------------------------------------------------------------------------

def _compute_advection_flux_div(
    tr: jnp.ndarray,
    tracer_advection: str,
    mass_flux_u: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    w_baro: jnp.ndarray,
    h_k_old: jnp.ndarray,
    h_u_old: jnp.ndarray,
    h_v_old: jnp.ndarray,
    grid,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute advection flux divergence for a single tracer field.

    Returns ``(div_hut, vert_flux_div)`` — horizontal and vertical
    components of the total flux divergence.  Units: [tracer · m/s]
    (thickness-weighted: ``div(h·u·T_face)``).  The caller combines
    them into a flux-form tracer update:

        hT_new = h_old * tr - dt * (div_hut + vert_flux_div)

    Notes
    -----
    When used with AB2: the extrapolated flux divergence
    ``(3/2+eps)*F^n - (1/2+eps)*F^{n-1}`` uses thickness-weighted
    divergences from different time levels.  In variable-thickness
    (z-star) runs this introduces an O(dt) accuracy degradation
    because ``F^{n-1}`` was computed with ``h^{n-1}`` mass fluxes
    but is combined with the current ``h_k_old``.  For constant-
    thickness convergence tests this is exact.

    When used with AB2 + FCT/PPM_FCT: the individual ``F^n`` and
    ``F^{n-1}`` are separately monotone (Zalesak-limited), but their
    AB2 linear combination is NOT guaranteed monotone.  This is a
    known limitation shared with MITgcm.
    """

    if tracer_advection == "ppm_fct":
        from legoesm.ocean.advection import fct_tracer_advection
        div_hut, vert_flux_div = fct_tracer_advection(
            tr, mass_flux_u, mass_flux_v, w_baro, h_k_old, grid, dt,
        )
    elif tracer_advection == "ppm":
        from legoesm.ocean.advection import (
            ppm_to_u_points, ppm_to_v_points,
            flux_form_vertical_tracer_advection_ppm,
        )
        tr_u = ppm_to_u_points(tr, mass_flux_u)
        tr_v = ppm_to_v_points(tr, mass_flux_v)
        tracer_flux_u = mass_flux_u * tr_u
        tracer_flux_v = mass_flux_v * tr_v
        div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, grid)
        vert_flux_div = flux_form_vertical_tracer_advection_ppm(
            tr, w_baro, h_k_old, dt)
    elif tracer_advection == "dst3":
        from legoesm.ocean.advection import (
            dst3_to_u_points, dst3_to_v_points,
            flux_form_vertical_tracer_advection_dst3,
        )
        tr_u = dst3_to_u_points(tr, mass_flux_u, h_u_old, grid, dt)
        tr_v = dst3_to_v_points(tr, mass_flux_v, h_v_old, grid, dt)
        tracer_flux_u = mass_flux_u * tr_u
        tracer_flux_v = mass_flux_v * tr_v
        div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, grid)
        vert_flux_div = flux_form_vertical_tracer_advection_dst3(
            tr, w_baro, h_k_old, dt)
    elif tracer_advection == "dst3_multidim":
        from legoesm.ocean.advection import multidim_tracer_advection
        div_hut, vert_flux_div = multidim_tracer_advection(
            tr, mass_flux_u, mass_flux_v, w_baro,
            h_k_old, h_u_old, h_v_old, grid, dt,
        )
    elif tracer_advection in ("weno5", "weno7"):
        from legoesm.ocean.advection import (
            weno5_to_u_points, weno5_to_v_points,
            weno7_to_u_points, weno7_to_v_points,
            flux_form_vertical_tracer_advection_weno5,
            flux_form_vertical_tracer_advection_weno7,
        )
        _u_fn, _v_fn, _vert_fn = {
            "weno5": (weno5_to_u_points, weno5_to_v_points,
                      flux_form_vertical_tracer_advection_weno5),
            "weno7": (weno7_to_u_points, weno7_to_v_points,
                      flux_form_vertical_tracer_advection_weno7),
        }[tracer_advection]
        tr_u = _u_fn(tr, mass_flux_u)
        tr_v = _v_fn(tr, mass_flux_v)
        tracer_flux_u = mass_flux_u * tr_u
        tracer_flux_v = mass_flux_v * tr_v
        div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, grid)
        vert_flux_div = _vert_fn(tr, w_baro, h_k_old, dt)
    else:
        # upwind or tvd
        if tracer_advection == "tvd":
            tr_u = _tvd_to_u_points(tr, mass_flux_u)
            tr_v = _tvd_to_v_points(tr, mass_flux_v, grid=grid)
        else:
            tr_u = _upwind_to_u_points(tr, mass_flux_u)
            tr_v = _upwind_to_v_points(tr, mass_flux_v, grid=grid)
        tracer_flux_u = mass_flux_u * tr_u
        tracer_flux_v = mass_flux_v * tr_v
        div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, grid)

        if tracer_advection == "tvd":
            vert_flux_div = flux_form_vertical_tracer_advection_tvd(
                tr, w_baro, h_k_old, dt)
        else:
            vert_flux_div = flux_form_vertical_tracer_advection(tr, w_baro)

    return div_hut, vert_flux_div


def _ssp_rk3_tracer_step(
    tr: jnp.ndarray,
    tracer_advection: str,
    mass_flux_u: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    w_baro: jnp.ndarray,
    h_k_old: jnp.ndarray,
    h_k_new: jnp.ndarray,
    h_u_old: jnp.ndarray,
    h_v_old: jnp.ndarray,
    grid,
    dt: float,
    active_3d: jnp.ndarray,
) -> jnp.ndarray:
    """RK3 flux-form tracer advection step (Butcher-tableau form).

    Three-stage Runge-Kutta with the same coefficients as SSP-RK3
    (Shu-Osher 1988), but applied via the effective tendency

        F_eff = F0/6 + F1/6 + 2*F2/3

    in flux form for **exact conservation**:

        h_new * T_new = h_old * T - dt * F_eff

    Trade-off: the Shu-Osher convex-combination form preserves the
    strong stability (monotonicity/TVD) property of the forward Euler
    operator, but does not conserve mass exactly when h changes.
    This Butcher form conserves mass exactly but does NOT preserve
    SSP — with nonlinear limiters (TVD, WENO, FCT) new extrema may
    appear that would not appear under the true Shu-Osher form.

    Intermediate stages use ``h_k_old`` (exact for constant-thickness
    convergence tests, O(dt) approximation for the full model where
    thickness changes are from the barotropic step).  The velocity
    field is frozen across all three stages (correct for the
    operator-split advection sub-problem where velocity comes from
    the barotropic solver).
    """

    def _flux_div(tr_val):
        dh, dv = _compute_advection_flux_div(
            tr_val, tracer_advection, mass_flux_u, mass_flux_v, w_baro,
            h_k_old, h_u_old, h_v_old, grid, dt,
        )
        return dh + dv

    h_safe = jnp.maximum(h_k_old, 1e-10)

    # Stage 1
    fd0 = _flux_div(tr)
    tr1 = (h_k_old * tr - dt * fd0) / h_safe
    tr1 = jnp.where(active_3d > 0.5, tr1, tr)

    # Stage 2
    fd1 = _flux_div(tr1)
    tr1_adv = (h_k_old * tr1 - dt * fd1) / h_safe
    tr1_adv = jnp.where(active_3d > 0.5, tr1_adv, tr)
    tr2 = 0.75 * tr + 0.25 * tr1_adv

    # Stage 3 — conservative final update via effective flux
    fd2 = _flux_div(tr2)
    F_eff = (1.0 / 6.0) * fd0 + (1.0 / 6.0) * fd1 + (2.0 / 3.0) * fd2
    hT_new = h_k_old * tr - dt * F_eff
    tr_new = hT_new / jnp.maximum(h_k_new, 1e-10)
    tr_new = jnp.where(active_3d > 0.5, tr_new, tr)

    return tr_new


def _forward_backward_coriolis_3d(
    u: jnp.ndarray,
    v: jnp.ndarray,
    dt: float,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    land_mask: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply forward-backward (Matsuno) Coriolis to baroclinic perturbation velocity.

    The forward-backward scheme is unconditionally neutral for the
    inertial oscillation: the amplification factor is exactly 1,
    unlike forward Euler which amplifies by sqrt(1 + (f*dt)^2).

    Steps:
      1. Compute perturbation velocity u' = u - U_bar, v' = v - V_bar
      2. Forward:  u'_new = u' + dt * f_u * avg(v'  -> u-points)
      3. Backward: v'_new = v' - dt * f_v * avg(u'_new -> v-points)
      4. Reconstruct: u_new = u'_new + U_bar, v_new = v'_new + V_bar

    Parameters
    ----------
    u : (n_lat, n_lon+1, nlev) full 3D u velocity
    v : (n_lat+1, n_lon, nlev) full 3D v velocity
    dt : time step [s]
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    config : LatLonCGridOceanConfig
    u_mask : (n_lat, n_lon+1)
    v_mask : (n_lat+1, n_lon)
    land_mask : (n_lat, n_lon)
    eta, H_bathy : (n_lat, n_lon) for layer thickness computation

    Returns
    -------
    u_new, v_new : updated velocities with Coriolis applied
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = u.shape[-1]

    u_mask_3d = u_mask[..., jnp.newaxis]
    v_mask_3d = v_mask[..., jnp.newaxis]

    # --- Layer thickness at face points for depth averaging ---
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * land_mask

    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )

    # h at u/v-faces — min-rule (MOM6/MITgcm hFacW convention).
    # Must match the PE tendency and slow-forcing depth-average which
    # both use min_cell_to_uface/min_cell_to_vface.  Arithmetic mean
    # overestimates face depth at topographic steps, creating a
    # barotropic-baroclinic residual that drives spurious currents.
    h_u = min_cell_to_uface(h_k)
    h_v = min_cell_to_vface(h_k, grid)

    # --- Depth-averaged velocity (barotropic component) ---
    # Per-face thickness + barotropic-mean column reductions share the
    # h_u/h_v weight on the level axis — fuse into one stack each.
    _u_pair = jnp.sum(jnp.stack([h_u, u * h_u], axis=-1), axis=-2)
    H_u = jnp.maximum(_u_pair[..., 0], min_water_col)
    U_bar = _u_pair[..., 1] / H_u * u_mask

    _v_pair = jnp.sum(jnp.stack([h_v, v * h_v], axis=-1), axis=-2)
    H_v = jnp.maximum(_v_pair[..., 0], min_water_col)
    V_bar = _v_pair[..., 1] / H_v * v_mask

    # --- Perturbation velocity ---
    u_prime = (u - U_bar[..., jnp.newaxis]) * u_mask_3d
    v_prime = (v - V_bar[..., jnp.newaxis]) * v_mask_3d

    # --- Coriolis parameter at face points ---
    if hasattr(grid, "f_u") and hasattr(grid, "f_v"):
        f_u = grid.f_u.astype(u.dtype)  # (n_lat, n_lon+1)
        f_v = grid.f_v.astype(u.dtype)  # (n_lat+1, n_lon)
    else:
        f_cell = grid.f.astype(u.dtype)
        f_u = 0.5 * (jnp.roll(f_cell, 1, axis=1) + f_cell)
        f_u = jnp.concatenate([f_u, f_u[:, 0:1]], axis=1)
        f_v_interior = 0.5 * (f_cell[:-1] + f_cell[1:])
        f_v = jnp.concatenate(
            [f_cell[0:1], f_v_interior, f_cell[-1:]], axis=0,
        )

    # --- Forward step: update u' using old v' ---
    # Average v' to u-points (Sadourny 4-point average)
    v_west = jnp.roll(v_prime, 1, axis=1)
    v_at_u = 0.25 * (v_prime[:-1] + v_prime[1:]
                      + v_west[:-1] + v_west[1:])
    v_at_u = jnp.concatenate([v_at_u, v_at_u[:, 0:1, :]], axis=1)

    u_prime_new = (u_prime + dt * f_u[:, :, jnp.newaxis] * v_at_u) * u_mask_3d

    # --- Backward step: update v' using NEW u' ---
    # Average u'_new to v-points (Sadourny 4-point average).
    # Boundary: wall BC on regular lat-lon; fold halo on tripolar.
    from legoesm.ocean.dynamics.latlon_cgrid_operators import pad_ns_vector_u
    u_at_v_interior = 0.25 * (
        u_prime_new[:-1, :-1] + u_prime_new[:-1, 1:]
        + u_prime_new[1:, :-1] + u_prime_new[1:, 1:]
    )
    u_at_v = pad_ns_vector_u(u_at_v_interior, grid)

    v_prime_new = (v_prime - dt * f_v[:, :, jnp.newaxis] * u_at_v) * v_mask_3d

    # --- Reconstruct full velocity ---
    u_new = u_prime_new + U_bar[..., jnp.newaxis]
    v_new = v_prime_new + V_bar[..., jnp.newaxis]

    return u_new, v_new


class LatLonCGridOceanModel:
    """Boussinesq hydrostatic ocean model on a C-grid latitude-longitude grid.

    Uses split-explicit time stepping with compact-stencil C-grid operators
    for pressure gradient and divergence.

    Parameters
    ----------
    grid : LatLonGrid
        Horizontal grid.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    config : LatLonCGridOceanConfig, optional
        Model configuration.

    Example
    -------
    >>> grid = create_latlon_grid(90, 180)
    >>> z_coord = create_ocean_z_star(50)
    >>> model = LatLonCGridOceanModel(grid, z_coord)
    >>> state = rest_state_latlon_cgrid_ocean(grid, z_coord)
    >>> state_new = model.step(state, dt=3600.0)
    """

    def __init__(
        self,
        grid: LatLonGrid,
        z_coord: OceanZStarCoordinate,
        config: LatLonCGridOceanConfig | None = None,
        bgc_cfg=None,
    ):
        # Convert LatLonGrid -> LatLonCGridGeometry once at construction.
        # All downstream operators see the enriched geometry with per-cell
        # metric arrays.  For a plain LatLonGrid this is a no-op on field
        # access (legacy fields are identical); for a tripolar grid the
        # geometry carries fold descriptor and rotation angles.
        self.grid = ensure_geometry(grid)
        self.z_coord = z_coord
        self.config   = config
        self._bgc_cfg = bgc_cfg or LatLonCGridOceanConfig()
        self._validate_config(self.config)
        self._cfl_checked = False

        if self.config.physics is not None:
            self._physics_fn = make_ocean_physics(
                self.config.physics,
                apply_vertical_diffusion=not self.config.implicit_vertical_mixing,
            )
        else:
            self._physics_fn = None

    @staticmethod
    def _validate_config(config: LatLonCGridOceanConfig) -> None:
        """Validate configuration ranges."""
        nonnegative = {
            "A_h": config.A_h,
            "B_h": config.B_h,
            "K_h": config.K_h,
            "A_v": config.A_v,
            "K_v": config.K_v,
            "hyperdiff_coeff": config.hyperdiff_coeff,
            "barotropic_diffusion_alpha": config.barotropic_diffusion_alpha,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")

        if config.n_barotropic_substeps < 1:
            raise ValueError(
                f"n_barotropic_substeps must be >= 1, got "
                f"{config.n_barotropic_substeps!r}",
            )
        if config.barotropic_diffusion_dt_ref <= 0.0:
            raise ValueError(
                f"barotropic_diffusion_dt_ref must be > 0, got "
                f"{config.barotropic_diffusion_dt_ref!r}",
            )
        if config.min_water_column_m <= 0.0:
            raise ValueError(
                f"min_water_column_m must be > 0, got "
                f"{config.min_water_column_m!r}",
            )
        _valid_fw = {"none", "virtual_salt_flux"}
        if config.freshwater_closure not in _valid_fw:
            raise ValueError(
                f"freshwater_closure must be one of {_valid_fw}, "
                f"got {config.freshwater_closure!r}",
            )
        if config.max_abs_eta_m <= 0.0:
            raise ValueError(
                f"max_abs_eta_m must be > 0, got {config.max_abs_eta_m!r}")
        if config.temperature_min_c >= config.temperature_max_c:
            raise ValueError(
                f"temperature_min_c ({config.temperature_min_c}) must be "
                f"< temperature_max_c ({config.temperature_max_c})")
        if config.salinity_min_psu >= config.salinity_max_psu:
            raise ValueError(
                f"salinity_min_psu ({config.salinity_min_psu}) must be "
                f"< salinity_max_psu ({config.salinity_max_psu})")
        _valid_solvers = {"explicit_substep", "implicit_cn"}
        if config.barotropic_solver not in _valid_solvers:
            raise ValueError(
                f"barotropic_solver must be one of {_valid_solvers}, "
                f"got {config.barotropic_solver!r}")
        for fld in ("barotropic_implicit_theta_eta",
                    "barotropic_implicit_theta_pgf"):
            v = getattr(config, fld)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{fld} must be in [0, 1], got {v!r}")
        if config.barotropic_implicit_pcg_tol <= 0.0:
            raise ValueError(
                f"barotropic_implicit_pcg_tol must be > 0, "
                f"got {config.barotropic_implicit_pcg_tol!r}")
        if config.barotropic_implicit_pcg_maxiter < 1:
            raise ValueError(
                f"barotropic_implicit_pcg_maxiter must be >= 1, "
                f"got {config.barotropic_implicit_pcg_maxiter!r}")
        _valid_pgf = {"adcroft", "smc03"}
        pgf_scheme = getattr(config, "pgf_scheme", "adcroft")
        if pgf_scheme not in _valid_pgf:
            raise ValueError(
                f"pgf_scheme must be one of {_valid_pgf}, got {pgf_scheme!r}")
        _valid_time_int = {"euler", "ab2", "rk3"}
        if config.tracer_time_integrator not in _valid_time_int:
            raise ValueError(
                f"tracer_time_integrator must be one of {_valid_time_int}, "
                f"got {config.tracer_time_integrator!r}")
        if config.ab2_epsilon < 0.0:
            raise ValueError(
                f"ab2_epsilon must be >= 0, got {config.ab2_epsilon!r}")

    def check_barotropic_cfl(self, dt: float) -> float:
        """Check barotropic CFL and warn if marginal or unstable.

        Parameters
        ----------
        dt : float
            Baroclinic timestep [s].

        Returns
        -------
        cfl : float
            Barotropic CFL number.
        """
        import math
        import warnings

        g = self.config.g
        H_max = self.z_coord.H_max
        n_sub = self.config.n_barotropic_substeps
        # grid.dx and grid.dy are "distance over 2 cells", so cell width = dx/2.
        # grid.dy is (n_lat,) — take the global min (Mercator-safe).
        dx_min = min(
            float(jnp.min(self.grid.dx)) / 2.0,
            float(jnp.min(self.grid.dy)) / 2.0,
        )

        c_baro = math.sqrt(g * H_max)
        dt_baro = dt / n_sub
        cfl = c_baro * dt_baro / dx_min

        if cfl > 0.8:
            n_min = math.ceil(c_baro * dt / (0.8 * dx_min))
            warnings.warn(
                f"Barotropic CFL = {cfl:.2f} (> 0.8) — may be unstable. "
                f"c_baro={c_baro:.1f} m/s, dx_min={dx_min:.0f} m, "
                f"dt_baro={dt_baro:.1f} s. "
                f"Suggest n_barotropic_substeps >= {n_min} "
                f"(currently {n_sub}).",
                stacklevel=2,
            )

        # AB2 has a tighter advective CFL limit (~0.72 with eps=0.1)
        # than forward Euler (~1.0).  Warn if the advective CFL is
        # close to the AB2 stability boundary.
        if self.config.tracer_time_integrator == "ab2":
            # Rough advective CFL estimate using max barotropic velocity
            adv_cfl = c_baro * dt / dx_min  # upper bound (uses gravity wave speed)
            if adv_cfl > 0.7:
                warnings.warn(
                    f"Advective CFL estimate ~{adv_cfl:.2f} is near the "
                    f"AB2 stability limit (~0.72 with eps={self.config.ab2_epsilon}). "
                    f"Consider reducing dt or using tracer_time_integrator='euler'.",
                    stacklevel=2,
                )

        return cfl

    def tendencies(self, state: LatLonCGridOceanState, surface_forcing=None,
                   sponge=None, dt=300.0):
        """Compute baroclinic tendencies."""
        return latlon_cgrid_ocean_baroclinic_tendencies(
            state, self.grid, self.z_coord, self.config,
            physics_fn=self._physics_fn,
            surface_forcing=surface_forcing,
            sponge=sponge,
            dt=dt,
        )

    def tendencies_with_diagnostics(
        self, state: LatLonCGridOceanState, surface_forcing=None,
        sponge=None, dt=300.0,
    ):
        """Compute baroclinic tendencies + per-term momentum-tendency
        breakdown.

        Returns
        -------
        (LatLonCGridOceanTendencies, MomentumTendencyDiagnostics)
            The diagnostics satisfy
            ``Σ components == du_dt`` to machine precision (verified by
            ``tests/ocean/unit/test_momentum_diagnostics_closure.py``).

        Use the returned tendencies as the start-of-step approximation
        of what the model integrates internally; for the
        time-mean budget this converges to the actually-applied
        tendency at O(dt) accuracy.
        """
        return latlon_cgrid_ocean_baroclinic_tendencies(
            state, self.grid, self.z_coord, self.config,
            physics_fn=self._physics_fn,
            surface_forcing=surface_forcing,
            sponge=sponge,
            dt=dt,
            diagnose_momentum=True,
        )

    def _step_impl(self, state: LatLonCGridOceanState, dt: float,
                   freshwater=None, surface_forcing=None,
                   sponge=None) -> LatLonCGridOceanState:
        """Core step logic — no JIT wrapper.

        Use this directly inside an outer ``@jax.jit`` context (e.g.
        ``lax.scan`` block functions) to avoid nested JIT boundaries
        that can cause numerical divergence with partial-cell
        coordinates.  For standalone calls, use ``step()`` which wraps
        this in ``@jax.jit``.
        """
        state = cast_pytree(state, None, "compute")

        u_mask_3d = state.u_mask.data[..., jnp.newaxis]
        v_mask_3d = state.v_mask.data[..., jnp.newaxis]
        mask_3d = state.land_mask.data[..., jnp.newaxis]

        # 1. Baroclinic tendencies (non-Coriolis)
        tend = self.tendencies(state, surface_forcing, sponge=sponge, dt=dt)

        # 2. Update tracers
        T_new = state.T.data + dt * tend.dT_dt.data
        S_new = state.S.data + dt * tend.dS_dt.data

        # 3. Slow-forcing coupling (#205): split baroclinic tendency into
        # depth-averaged (slow forcing for barotropic solver) and
        # perturbation (applied to 3D velocity before barotropic step).
        #
        # Current approach (without slow-forcing): apply full tendency
        # to u_star, then barotropic solver sees initial U_bar that
        # already includes the depth-averaged tendency.  The problem:
        # the barotropic solver doesn't know about this forcing, so
        # the baroclinic-barotropic coupling is only through the initial
        # velocity — not updated as eta evolves during substeps.
        #
        # MOM6 approach: pass depth-averaged tendency as F_slow_u to the
        # barotropic solver, which applies it at each substep.  This
        # couples the slow forcing to the evolving barotropic state.
        du_dt = tend.du_dt.data
        dv_dt = tend.dv_dt.data

        # Compute layer thickness at u/v faces for depth-averaging.
        # Min-rule: the face's effective wet thickness is the shallower
        # side's thickness (MOM6/MITgcm hFacW convention).  The same
        # convention is used in the barotropic Helmholtz solver and the
        # tracer mass flux below — consistency is required for the
        # Hallberg-Adcroft 2009 column-sum invariant
        # ``sum_k(h_u * u_corrected) == Hu_avg`` to hold to machine
        # precision.  For full cells this reduces to the cell value
        # (bit-exact backwards-compat).
        h_k_pre = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m,
        )
        # h at u-faces — min-rule (MOM6/MITgcm hFacW convention).
        # Must match the PE tendency which uses min_cell_to_uface, so
        # that F_slow = depth_avg(du_dt, h_u) is consistent with the
        # 3D tendency.  Arithmetic mean overestimates face depth at
        # topographic steps, creating a barotropic-baroclinic residual.
        h_u_pre = min_cell_to_uface(h_k_pre)
        # h at v-faces — same min-rule for meridional direction.
        h_v_pre = min_cell_to_vface(h_k_pre, self.grid)

        # H + F_slow share the per-face h weight on the level axis —
        # fuse the two reductions per face into one stacked sum.
        _u_pair = jnp.sum(jnp.stack([h_u_pre, du_dt * h_u_pre], axis=-1), axis=-2)
        H_u_pre = jnp.maximum(_u_pair[..., 0], 1e-10)
        F_slow_u = _u_pair[..., 1] / H_u_pre * state.u_mask.data
        _v_pair = jnp.sum(jnp.stack([h_v_pre, dv_dt * h_v_pre], axis=-1), axis=-2)
        H_v_pre = jnp.maximum(_v_pair[..., 0], 1e-10)
        F_slow_v = _v_pair[..., 1] / H_v_pre * state.v_mask.data

        # Perturbation tendency (depth-mean removed) → applied to 3D.
        # MUST be computed from the *baroclinic-only* F_slow (before A2 is
        # added below) so that the depth-mean biharmonic damping acts only
        # on U_bar, not on the perturbation u' = u - U_bar.
        du_dt_pert = du_dt - F_slow_u[..., jnp.newaxis]
        dv_dt_pert = dv_dt - F_slow_v[..., jnp.newaxis]

        # A2 — depth-mean biharmonic hyperviscosity on (U_bar, V_bar).
        # Damps the barotropic standing mode at deep cells next to steep
        # slopes (Rhines 1969 bottom-trapped wave with f≈0) without
        # touching the baroclinic perturbation u' (already finalized
        # above as du_dt_pert / dv_dt_pert).  Applied as an additional
        # slow forcing on the implicit-CN barotropic solver:
        #   ∂U_bar/∂t |_diss = -ν₄ · ∇⁴ U_bar.
        # MOM6/HIM BIHARMONIC_BAROTROPIC analog.  No-op at default
        # ``B_h_barotropic = 0`` (bit-exact backward compat).
        if getattr(self.config, "B_h_barotropic", 0.0) > 0.0:
            from legoesm.ocean.dynamics.latlon_cgrid_operators import (
                vector_bilaplacian_cgrid, biharmonic_scaling_factor,
            )
            U_bar = jnp.sum(state.u.data * h_u_pre, axis=-1) / H_u_pre
            V_bar = jnp.sum(state.v.data * h_v_pre, axis=-1) / H_v_pre
            U_bar = U_bar * state.u_mask.data
            V_bar = V_bar * state.v_mask.data
            bilap_U, bilap_V = vector_bilaplacian_cgrid(
                U_bar, V_bar, self.grid,
                mask=state.land_mask.data,
                u_mask=state.u_mask.data,
                v_mask=state.v_mask.data,
            )
            # cos^4(lat) scaling: lat-lon grid spacing shrinks as
            # cos(lat) near the poles, so a constant ν₄ would violate
            # biharmonic CFL there.  Same convention as the layered B_h.
            scale_u, scale_v = biharmonic_scaling_factor(self.grid)
            nu4 = jnp.asarray(
                self.config.B_h_barotropic, dtype=F_slow_u.dtype,
            )
            scale_u_b = scale_u.astype(F_slow_u.dtype)[:, None]
            scale_v_b = scale_v.astype(F_slow_v.dtype)[:, None]
            F_slow_u = F_slow_u - nu4 * scale_u_b * bilap_U.astype(F_slow_u.dtype)
            F_slow_v = F_slow_v - nu4 * scale_v_b * bilap_V.astype(F_slow_v.dtype)
            F_slow_u = F_slow_u * state.u_mask.data
            F_slow_v = F_slow_v * state.v_mask.data

        u_star = state.u.data + dt * du_dt_pert
        v_star = state.v.data + dt * dv_dt_pert

        # 4. Forward-backward Coriolis on perturbation velocity
        #
        # Forward Euler Coriolis amplifies inertial oscillations by
        # sqrt(1 + (f*dt)^2) per step, causing blowup at high latitudes.
        # Forward-backward (Matsuno) stepping is unconditionally neutral:
        #   u' += dt * f * v'_at_u          (forward: old v')
        #   v' -= dt * f * u'_new_at_v      (backward: new u')
        # This matches the barotropic solver's Coriolis treatment.
        u_star, v_star = _forward_backward_coriolis_3d(
            u_star, v_star, dt, self.grid, self.z_coord, self.config,
            state.u_mask.data, state.v_mask.data, state.land_mask.data,
            state.eta.data, state.H_bathy.data,
        )

        # Enforce periodic wrap column: u[:,n_lon] must equal u[:,0].
        u_star = u_star.at[:, -1].set(u_star[:, 0])

        state_mid = state._replace(
            u=state.u.replace(data=u_star * u_mask_3d),
            v=state.v.replace(data=v_star * v_mask_3d),
            T=state.T.replace(data=T_new * mask_3d),
            S=state.S.replace(data=S_new * mask_3d),
        )

        # 5. Save pre-barotropic layer thickness
        h_k_old = compute_layer_thickness(
            state_mid.eta.data, state_mid.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m,
        )

        # 6. Barotropic step.  Two paths:
        #    - explicit_substep: split-explicit forward-backward substepping
        #      with cosine/box time filter.
        #    - implicit_cn: single-step Crank-Nicolson free surface (PCG).
        #      Eliminates the chequerboard mode by construction; no
        #      substepping or time filter needed.

        # Freshwater mass flux for barotropic continuity equation
        F_slow_eta = None
        if freshwater is not None and self.config.freshwater_closure != "none":
            F_slow_eta = freshwater_eta_tendency(
                freshwater, self.config.rho_0,
            ) * state.land_mask.data

        if self.config.barotropic_solver == "implicit_cn":
            state_new, (Hu_avg, Hv_avg) = barotropic_implicit_latlon_cgrid(
                state_mid, dt,
                self.grid, self.z_coord, self.config,
                F_slow_eta=F_slow_eta,
                F_slow_u=F_slow_u,
                F_slow_v=F_slow_v,
            )
        else:
            dt_s = dt / self.config.n_barotropic_substeps
            state_new, (Hu_avg, Hv_avg) = barotropic_substeps_latlon_cgrid(
                state_mid, dt_s, self.config.n_barotropic_substeps,
                self.grid, self.z_coord, self.config,
                F_slow_eta=F_slow_eta,
                F_slow_u=F_slow_u,
                F_slow_v=F_slow_v,
            )

        # 7. Flux-form tracer update using full 3D velocity
        #
        # The barotropic solver returns Hu_avg (time-averaged depth-
        # integrated transport) consistent with the continuity equation
        # that produced h_new.  For tracer advection we need per-layer
        # mass fluxes that:
        #   (a) preserve the baroclinic velocity shear (needed for
        #       Ekman pumping and vertical tracer transport), and
        #   (b) have depth-integrated transport matching Hu_avg (needed
        #       for consistency with the barotropic continuity).
        #
        # We take the full 3D velocity from state_new (which preserves
        # baroclinic structure) and apply a uniform barotropic correction
        # so that sum_k(h_k * u_corrected_k) = Hu_avg exactly.
        # (Hallberg & Adcroft 2009, Shchepetkin & McWilliams 2005).
        _min_uface_op = min_cell_to_uface
        _min_vface_op = lambda f: min_cell_to_vface(f, self.grid)
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            divergence_cgrid, interp_cell_to_uface,
        )

        mask = state.land_mask.data

        # Layer thickness at face points (min-rule, partial-cell aware
        # and consistent with the barotropic solver and slow forcing).
        h_u_old = _min_uface_op(h_k_old)        # (n_lat, n_lon+1, nlev)
        h_v_old = _min_vface_op(h_k_old)        # (n_lat+1, n_lon, nlev)
        H_u_old = jnp.sum(h_u_old, axis=-1)     # (n_lat, n_lon+1)
        H_v_old = jnp.sum(h_v_old, axis=-1)     # (n_lat+1, n_lon)

        # 3D face masks for the tracer mass flux.  For partial cells the
        # 2D u_mask/v_mask are non-zero at the topographic-step face
        # (both surface columns are wet) but the face must be closed
        # below the shallower seafloor.  Using compute_face_masks_3d on
        # the partial coord's is_active gives the correct per-level
        # closed-wall faces.  For pure z\\* the 3D mask collapses to the
        # 2D mask broadcast across all levels — bit-exact backwards-compat.
        # Likewise, ``active_3d`` gates inactive cells (below the
        # partial seafloor) where h_k_old = h_k_new = 0; without this
        # gate, the floor in ``tr_new = hT_new / max(h_k_new, 1e-10)``
        # amplifies tiny float-precision residuals into huge spurious
        # tracer values inside the ground.
        if isinstance(self.z_coord, OceanPartialCellCoordinate):
            u_mask_3d_tracer, v_mask_3d_tracer = compute_face_masks_3d(
                self.z_coord.is_active, self.grid,
            )
            u_mask_3d_tracer = u_mask_3d_tracer.astype(h_u_old.dtype)
            v_mask_3d_tracer = v_mask_3d_tracer.astype(h_v_old.dtype)
            active_3d = self.z_coord.is_active.astype(h_u_old.dtype)
        else:
            u_mask_3d_tracer = state.u_mask.data[..., jnp.newaxis]
            v_mask_3d_tracer = state.v_mask.data[..., jnp.newaxis]
            active_3d = mask_3d

        # Full 3D velocity (barotropic + baroclinic) from state after
        # barotropic correction.  The barotropic solver preserves the
        # baroclinic perturbation u' = u - U_bar and replaces the
        # barotropic component with the time-averaged U_bar_avg.
        u_3d = state_new.u.data   # (n_lat, n_lon+1, nlev)
        v_3d = state_new.v.data   # (n_lat+1, n_lon, nlev)

        # Correct the barotropic component so that depth-integrated
        # transport matches Hu_avg exactly.  The correction is the
        # difference between <H*U> (time-averaged transport) and
        # <U>*H (time-averaged velocity times pre-barotropic H).
        # H + Hu reductions per face share the h_u_old/h_v_old weight
        # on the level axis — fuse into one stack each.
        _u_pair = jnp.sum(jnp.stack([h_u_old, u_3d * h_u_old], axis=-1), axis=-2)
        H_u_old, Hu_3d = _u_pair[..., 0], _u_pair[..., 1]  # (n_lat, n_lon+1)
        _v_pair = jnp.sum(jnp.stack([h_v_old, v_3d * h_v_old], axis=-1), axis=-2)
        H_v_old, Hv_3d = _v_pair[..., 0], _v_pair[..., 1]  # (n_lat+1, n_lon)
        delta_U = (Hu_avg - Hu_3d) / jnp.maximum(H_u_old, 1e-10)
        delta_V = (Hv_avg - Hv_3d) / jnp.maximum(H_v_old, 1e-10)
        u_corrected = u_3d + delta_U[..., jnp.newaxis]
        v_corrected = v_3d + delta_V[..., jnp.newaxis]

        # Per-layer mass fluxes with full 3D velocity structure.
        # Unlike the previous barotropic-only distribution (which gave
        # uniform velocity at all depths and identically zero w),
        # this preserves baroclinic shear and produces non-zero vertical
        # velocity from Ekman pumping/suction.
        mass_flux_u = h_u_old * u_corrected * u_mask_3d_tracer
        mass_flux_v = h_v_old * v_corrected * v_mask_3d_tracer

        # Flux-form tracer update (horizontal + vertical)
        #
        # Both horizontal and vertical transport use the barotropic-averaged
        # per-layer divergence for consistency:
        #   h_new * T_new = h_old * T_mid
        #     - dt * div_h(mf_k * T_face_h)        [horizontal flux]
        #     - dt * (w_{k-1/2}*T_{k-1/2} - w_{k+1/2}*T_{k+1/2})  [vertical flux]
        #
        # The horizontal flux integrates to zero by the 2D divergence theorem.
        # The vertical flux telescopes to surface/bottom (both zero).
        # Total conservation is exact.


        h_k_new = compute_layer_thickness(
            state_new.eta.data, state_new.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m,
        )

        # Diagnose w from barotropic-averaged per-layer divergence
        # (consistent with the horizontal transport used for tracers).
        # mass_flux_u/v are thickness-weighted (h*u), so flux_div_k
        # is div(h*u) [m/s] and already includes layer thickness.
        flux_div_k = divergence_cgrid(mass_flux_u, mass_flux_v, self.grid)
        w_baro = diagnose_w_from_flux_div(
            flux_div_k, self.z_coord, thickness_weighted=True,
        )

        T_mid = state_new.T.data  # tracer after diffusion+physics Euler step
        S_mid = state_new.S.data

        # GM/Redi isopycnal mixing (if configured)
        if self.config.gm_redi is not None:
            dT_gm, dS_gm = gm_redi_tracer_tendency_latlon(
                T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
                self.grid, self.z_coord, self.config.gm_redi,
                eos=self.config.eos, eos_linear=self.config.eos_linear,
                mask=state.land_mask.data,
                u_mask=state.u_mask.data,
                v_mask=state.v_mask.data,
            )
            T_mid = T_mid + dt * dT_gm * mask_3d
            S_mid = S_mid + dt * dS_gm * mask_3d

        if self.config.tracer_advection == "som":
            # SOM (Prather 1986): Second Order Moments advection (#210).
            # Advects polynomial sub-cell distributions (mean + 9 moments)
            # via directional sweeps.  Near-zero spurious diapycnal mixing,
            # fully differentiable (no limiter).
            # Initialise moments: use existing Fields, or create from zeros.
            # Always produce Field output (not None → Field transition) so
            # the pytree structure is stable for jax.lax.scan.
            dims_mom = ("lat", "lon", "level", "moment")
            if state.T_som is not None:
                T_mom = state.T_som.data
                S_mom = state.S_som.data
            else:
                T_mom = jnp.zeros((*T_mid.shape, 9), dtype=T_mid.dtype)
                S_mom = jnp.zeros((*S_mid.shape, 9), dtype=S_mid.dtype)
                # Pre-create Fields on state_new so the output pytree
                # always has the same structure as the input.
                state_new = state_new._replace(
                    T_som=Field(data=T_mom, name="T_som",
                                dims=dims_mom, units=""),
                    S_som=Field(data=S_mom, name="S_som",
                                dims=dims_mom, units=""),
                )

            T_corrected, T_mom_new = som_advect_tracers(
                T_mid, T_mom, mass_flux_u, mass_flux_v, w_baro,
                h_k_old, h_k_new, self.grid, dt, mask,
            )
            S_corrected, S_mom_new = som_advect_tracers(
                S_mid, S_mom, mass_flux_u, mass_flux_v, w_baro,
                h_k_old, h_k_new, self.grid, dt, mask,
            )

            state_new = state_new._replace(
                T_som=state_new.T_som.replace(data=T_mom_new),
                S_som=state_new.S_som.replace(data=S_mom_new),
            )
        else:
            _adv = self.config.tracer_advection
            _tti = self.config.tracer_time_integrator
            _T_flux_div_cur = None
            _S_flux_div_cur = None

            # AB2: ensure pytree structure is stable for jax.lax.scan.
            # When the input state has None carry fields, pre-create
            # zero-filled Fields on state_new so the output pytree
            # always matches the input (same pattern as SOM moments).
            if _tti == "ab2" and state.T_flux_div_prev is None:
                from legoesm.core.field import Field as _Field
                _dims_fd = ("lat", "lon", "level")
                _zero_fd = jnp.zeros_like(T_mid)
                state_new = state_new._replace(
                    T_flux_div_prev=_Field(
                        data=_zero_fd, name="T_flux_div_prev",
                        dims=_dims_fd, units="m/s"),
                    S_flux_div_prev=_Field(
                        data=_zero_fd, name="S_flux_div_prev",
                        dims=_dims_fd, units="m/s"),
                )

            for tr_name in ['T', 'S']:
                tr = T_mid if tr_name == 'T' else S_mid

                if _tti == "rk3":
                    # RK3: 3 sub-stages, 3x cost, 3rd-order temporal
                    tr_new = _ssp_rk3_tracer_step(
                        tr, _adv,
                        mass_flux_u, mass_flux_v, w_baro,
                        h_k_old, h_k_new, h_u_old, h_v_old,
                        self.grid, dt, active_3d,
                    )
                else:
                    # Compute flux divergence (single evaluation for Euler/AB2)
                    div_hut, vert_flux_div = _compute_advection_flux_div(
                        tr, _adv,
                        mass_flux_u, mass_flux_v, w_baro,
                        h_k_old, h_u_old, h_v_old, self.grid, dt,
                    )
                    total_flux_div = div_hut + vert_flux_div

                    if _tti == "ab2":
                        # Adams-Bashforth 2: extrapolate flux divergence
                        # F^{n+1/2} = (3/2+eps)*F^n - (1/2+eps)*F^{n-1}
                        eps = self.config.ab2_epsilon
                        prev_field = (state.T_flux_div_prev
                                      if tr_name == 'T'
                                      else state.S_flux_div_prev)
                        if prev_field is not None:
                            fd_prev = prev_field.data
                            effective_fd = ((1.5 + eps) * total_flux_div
                                            - (0.5 + eps) * fd_prev)
                        else:
                            # First step: fall back to Euler
                            effective_fd = total_flux_div
                        hT_new = h_k_old * tr - dt * effective_fd
                    else:
                        # Forward Euler (default)
                        hT_new = h_k_old * tr - dt * total_flux_div

                    tr_new = hT_new / jnp.maximum(h_k_new, 1e-10)
                    tr_new = jnp.where(active_3d > 0.5, tr_new, tr)

                    # Store current flux divergence for AB2 carry
                    if _tti == "ab2":
                        if tr_name == 'T':
                            _T_flux_div_cur = total_flux_div
                        else:
                            _S_flux_div_cur = total_flux_div

                if tr_name == 'T':
                    T_corrected = tr_new
                else:
                    S_corrected = tr_new

            # Persist AB2 previous flux divergence on state
            if _tti == "ab2":
                from legoesm.core.field import Field as _Field
                _dims_fd = ("lat", "lon", "level")
                state_new = state_new._replace(
                    T_flux_div_prev=_Field(
                        data=_T_flux_div_cur, name="T_flux_div_prev",
                        dims=_dims_fd, units="m/s"),
                    S_flux_div_prev=_Field(
                        data=_S_flux_div_cur, name="S_flux_div_prev",
                        dims=_dims_fd, units="m/s"),
                )

        # Include vertical velocity diagnostic in state
        # w_baro has shape (..., nlev+1) on half levels, interpolate to full levels (..., nlev)
        w_full = 0.5 * (w_baro[..., :-1] + w_baro[..., 1:])  # Average adjacent half levels
        w_field = state.w.replace(data=w_full, name="w")

        state_new = state_new._replace(
            T=state_new.T.replace(data=T_corrected),
            S=state_new.S.replace(data=S_corrected),
            w=w_field,
        )

        # 8. Freshwater forcing (virtual salt flux only)
        #
        # The freshwater eta tendency (F_fw_eta) is now applied inside
        # the barotropic continuity equation (via F_slow_eta), so no
        # post-hoc eta correction is needed.  Only the virtual salt
        # flux remains here, applied to the top layer of S.
        if freshwater is not None and self.config.freshwater_closure != "none":
            dz_0 = h_k_new[..., 0]
            dS_fw = virtual_salt_flux(
                freshwater, S_ref=self.config.S_ref, dz_0=dz_0, rho_0=self.config.rho_0,
            )
            # Cast the freshwater contribution to S's dtype so the
            # scatter add does not silently widen on x64 mode (the
            # freshwater struct is built at JAX-default precision in
            # init helpers, which can be f64 while S runs at the
            # storage policy's f32).
            _S_dtype = state_new.S.data.dtype
            S_fw = state_new.S.data.at[..., 0].add(
                (dt * dS_fw * mask).astype(_S_dtype),
            )
            state_new = state_new._replace(
                S=state_new.S.replace(data=S_fw),
            )

        # 8b. Implicit (backward-Euler) vertical mixing for u, v, T, S.
        #
        # When this branch is active, the PE tendency function has
        # skipped its explicit ``config.A_v`` block and every vertical-
        # mixing / convection scheme has been called with
        # ``apply_diffusion=False`` (so they only contributed K_v/A_v
        # *profiles* — and KPP non-local fluxes — to the explicit
        # update).  We now apply the full diffusion implicitly so the
        # stiff ``K_conv = 1 m²/s`` and surface-BL diffusivities are
        # not constrained by the explicit CFL limit
        # ``dt < dz² / (2K)`` — see issue #204.
        #
        # The solve uses zero-flux boundary conditions at the surface
        # and bottom and is split-stepped (Lie splitting, 1st-order)
        # after tracer advection, GM/Redi, and the freshwater virtual
        # salt flux — matching MOM6's diabatic-process ordering.
        if self.config.implicit_vertical_mixing:
            state_new = self._apply_implicit_vertical_mixing(
                state_new, dt, surface_forcing,
                K_v_phys=tend.K_v, A_v_phys=tend.A_v,
            )

        # 9. Conservation fixers
        if self.config.use_conservation_fixer:
            state_new = ocean_conservation_fixer(
                state_new, state, self.grid, self.z_coord, self.config,
            )

        # ── BGC source/sink step ─────────────────────────────────────────
        if state_new.biogeo is not None and self._bgc_cfg is not None:
            from legoesm.ocean.biogeochemistry import step_ocean_biogeochemistry
            biogeo_new, _ = step_ocean_biogeochemistry(
                state_new.biogeo,
                state_new.T.data, state_new.S.data,
                self.z_coord.dz_ref, self.z_coord.z_full_ref,
                state_new.land_mask.data,
                dt, self._bgc_cfg,
            )
            state_new = state_new._replace(biogeo=biogeo_new)

        return cast_pytree(state_new, None, "storage", allow_downcast=True)

    @staticmethod
    def _symmetrize_fold(state, fold):
        """Enforce fold symmetry on the fold row.

        Scalars (eta, T, S) at fold-partner cells must be equal.
        Velocity v at the fold face must be antisymmetric.
        """
        perm = fold.perm_T

        # Scalars: average fold partners
        eta = state.eta.data
        eta_sym = 0.5 * (eta[-1:] + eta[-1:, perm])
        eta = eta.at[-1].set(eta_sym[0])

        T = state.T.data
        T_sym = 0.5 * (T[-1:] + T[-1:, perm, :])
        T = T.at[-1].set(T_sym[0])

        S = state.S.data
        S_sym = 0.5 * (S[-1:] + S[-1:, perm, :])
        S = S.at[-1].set(S_sym[0])

        # v at fold face (last v-row): antisymmetric
        v = state.v.data
        v_fold = v[-1:]  # (1, n_lon, nlev)
        v_partner = v_fold[:, perm, :]
        v_sym = 0.5 * (v_fold - v_partner)
        v = v.at[-1].set(v_sym[0])

        return state._replace(
            eta=state.eta.replace(data=eta),
            T=state.T.replace(data=T),
            S=state.S.replace(data=S),
            v=state.v.replace(data=v),
        )

    def _apply_implicit_vertical_mixing(
        self,
        state: LatLonCGridOceanState,
        dt: float,
        surface_forcing,
        K_v_phys=None,
        A_v_phys=None,
    ) -> LatLonCGridOceanState:
        """Backward-Euler vertical diffusion for ``u, v, T, S``.

        Uses the K_v / A_v profiles already computed by the physics
        function (passed via ``K_v_phys`` / ``A_v_phys``), adds the
        ``LatLonCGridOceanConfig.A_v`` / ``K_v`` background floors,
        and applies one tridiagonal solve per column to each prognostic
        field.  The solver enforces zero-flux boundary conditions, so
        the column-mean (and hence the barotropic mode for u, v) is
        preserved exactly.

        When K_v_phys / A_v_phys are None (no physics function, or
        physics that doesn't produce K profiles), falls back to
        ``compute_vertical_K_profiles`` for a fresh computation.

        Called only when ``config.implicit_vertical_mixing == True``.
        """
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean, build_dz_half,
            compute_vertical_K_profiles,
        )
        from legoesm.ocean.vertical import compute_ocean_jacobian
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            interp_cell_to_uface,
        )

        if K_v_phys is not None and A_v_phys is not None:
            # Fast path: use K profiles already computed by the physics
            # function, just add the config background floors.
            nlev = state.T.data.shape[-1]
            dtype = state.T.data.dtype
            K_v_cell = K_v_phys + jnp.asarray(self.config.K_v, dtype=dtype)
            A_v_cell = A_v_phys + jnp.asarray(self.config.A_v, dtype=dtype)
        else:
            # Fallback: recompute K profiles (expensive for KPP).
            physics_config = self.config.physics
            if physics_config is None:
                from legoesm.ocean.physics.combined import OceanPhysicsConfig
                from legoesm.ocean.physics.vertical_mixing.config import (
                    VerticalMixingConfig,
                )
                from legoesm.ocean.physics.convection.config import (
                    OceanConvectionConfig,
                )
                physics_config = OceanPhysicsConfig(
                    vertical_mixing=VerticalMixingConfig(scheme="none"),
                    convection=OceanConvectionConfig(scheme="none"),
                )
            u_cell = 0.5 * (state.u.data[:, :-1, :] + state.u.data[:, 1:, :])
            v_cell = 0.5 * (state.v.data[:-1, :, :] + state.v.data[1:, :, :])
            cc_state = state._replace(
                u=state.u.replace(data=u_cell),
                v=state.v.replace(data=v_cell),
            )
            K_v_cell, A_v_cell = compute_vertical_K_profiles(
                cc_state, self.z_coord, surface_forcing, physics_config,
                A_v_background=float(self.config.A_v),
                K_v_background=float(self.config.K_v),
            )

        # dz at cell centers (jacobian-corrected so the eta-stretched
        # column heights match the partial-cell / z* layer thicknesses
        # used by every other operator in this step).
        J_cell = compute_ocean_jacobian(
            state.eta.data, state.H_bathy.data, self.z_coord,
        )
        dz_cell = self.z_coord.dz_ref * J_cell[..., jnp.newaxis]
        dz_half_cell = build_dz_half(dz_cell)

        mask_3d = state.land_mask.data[..., jnp.newaxis]

        # ---- Tracers (cell-centered: K aligns with T, S directly) ----
        K_v_cell = K_v_cell.astype(state.T.data.dtype)
        T_new = implicit_vertical_diffusion_ocean(
            state.T.data, K_v_cell, dz_cell, dz_half_cell, dt,
        )
        S_new = implicit_vertical_diffusion_ocean(
            state.S.data, K_v_cell, dz_cell, dz_half_cell, dt,
        )
        T_new = jnp.where(mask_3d > 0.5, T_new, state.T.data)
        S_new = jnp.where(mask_3d > 0.5, S_new, state.S.data)

        # ---- Momentum (u at u-faces, v at v-faces) ----
        # Interpolate A_v and dz from cell centers to face centers.  The
        # solver only needs dz to be positive (it clips internally) and
        # the resulting tridiagonal system is well-posed on any column
        # with at least two wet levels.
        A_v_cell = A_v_cell.astype(state.u.data.dtype)
        A_v_u = interp_cell_to_uface(A_v_cell)            # (n_lat, n_lon+1, nlev-1)
        A_v_v = _interp_to_v_points(A_v_cell)             # (n_lat+1, n_lon, nlev-1)
        dz_u = interp_cell_to_uface(dz_cell)
        dz_v = _interp_to_v_points(dz_cell)
        dz_half_u = build_dz_half(dz_u)
        dz_half_v = build_dz_half(dz_v)
        u_mask_3d = state.u_mask.data[..., jnp.newaxis]
        v_mask_3d = state.v_mask.data[..., jnp.newaxis]
        u_new = implicit_vertical_diffusion_ocean(
            state.u.data, A_v_u, dz_u, dz_half_u, dt,
        )
        v_new = implicit_vertical_diffusion_ocean(
            state.v.data, A_v_v, dz_v, dz_half_v, dt,
        )
        u_new = jnp.where(u_mask_3d > 0.5, u_new, state.u.data)
        v_new = jnp.where(v_mask_3d > 0.5, v_new, state.v.data)

        return state._replace(
            u=state.u.replace(data=u_new),
            v=state.v.replace(data=v_new),
            T=state.T.replace(data=T_new),
            S=state.S.replace(data=S_new),
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LatLonCGridOceanState, dt: float,
             freshwater=None, surface_forcing=None,
             sponge=None) -> LatLonCGridOceanState:
        """Advance one time step using split-explicit stepping.

        JIT-compiled wrapper around ``_step_impl``.  For use inside an
        outer JIT context (e.g. ``lax.scan``), call ``_step_impl``
        directly to avoid nested JIT boundaries.

        Parameters
        ----------
        state : LatLonCGridOceanState
        dt : float
            Time step [seconds].
        freshwater : FreshwaterForcing or None
            Freshwater forcing (P, E, runoff, ice).  If None, no
            freshwater mass/salt flux is applied.
        surface_forcing : OceanSurfaceForcing or None

        Returns
        -------
        LatLonCGridOceanState
        """
        return self._step_impl(state, dt, freshwater=freshwater,
                               surface_forcing=surface_forcing,
                               sponge=sponge)

    def step_checked(
        self,
        state: LatLonCGridOceanState,
        dt: float,
        freshwater=None,
        surface_forcing=None,
        sponge=None,
    ) -> LatLonCGridOceanState:
        """Advance one timestep with host-side runtime validation."""
        if not self._cfl_checked:
            self.check_barotropic_cfl(dt)
            self._cfl_checked = True
        state_new = self.step(state, dt, freshwater=freshwater,
                              surface_forcing=surface_forcing,
                              sponge=sponge)
        if self.config.enable_runtime_checks:
            self._assert_runtime_invariants(state_new)
        return state_new

    def _assert_runtime_invariants(self, state: LatLonCGridOceanState) -> None:
        """Host-side runtime checks for debugging/regression hardening."""
        mask = state.land_mask.data
        wet = mask > 0.5
        land = ~wet

        # Face mask consistency: u_mask/v_mask must match land_mask
        u_expected, v_expected = compute_face_masks(mask)
        if not (bool(jnp.all(state.u_mask.data == u_expected))
                and bool(jnp.all(state.v_mask.data == v_expected))):
            raise ValueError(
                "C-grid ocean: u_mask/v_mask inconsistent with land_mask. "
                "Use replace_land_mask() or land_mask_override instead of "
                "raw state._replace(land_mask=...).")

        # Fuse all reductions into a single ``jnp.stack`` + host pull
        # so the runtime check costs one device→host stall instead of
        # 7-9.  Same pattern as the loop-3 ``ocean_model.py`` rewrite.
        water_col = state.eta.data + state.H_bathy.data
        wet3 = wet[..., jnp.newaxis]
        T_ocean = jnp.where(wet3, state.T.data, jnp.nan)
        S_ocean = jnp.where(wet3, state.S.data, jnp.nan)

        finite_ok = (
            jnp.all(jnp.isfinite(state.u.data))
            & jnp.all(jnp.isfinite(state.v.data))
            & jnp.all(jnp.isfinite(state.T.data))
            & jnp.all(jnp.isfinite(state.S.data))
            & jnp.all(jnp.isfinite(state.eta.data))
        )
        any_wet = jnp.any(wet)
        _eta_dtype = state.eta.data.dtype
        _stats = jnp.stack([
            finite_ok.astype(_eta_dtype),
            any_wet.astype(_eta_dtype),
            jnp.min(jnp.where(wet, water_col, jnp.inf)).astype(_eta_dtype),
            jnp.max(jnp.abs(jnp.where(wet, state.eta.data, 0.0))).astype(_eta_dtype),
            jnp.nanmin(T_ocean).astype(_eta_dtype),
            jnp.nanmax(T_ocean).astype(_eta_dtype),
            jnp.nanmin(S_ocean).astype(_eta_dtype),
            jnp.nanmax(S_ocean).astype(_eta_dtype),
        ])
        host = np.asarray(_stats)
        finite_ok_h = bool(host[0] > 0.5)
        any_wet_h = bool(host[1] > 0.5)
        min_wc = float(host[2]) if any_wet_h else float("inf")
        eta_abs = float(host[3]) if any_wet_h else 0.0
        T_min = float(host[4]) if any_wet_h else float("nan")
        T_max = float(host[5]) if any_wet_h else float("nan")
        S_min = float(host[6]) if any_wet_h else float("nan")
        S_max = float(host[7]) if any_wet_h else float("nan")

        if not finite_ok_h:
            raise FloatingPointError(
                "C-grid ocean: non-finite state detected")

        if min_wc < self.config.min_water_column_m:
            raise ValueError(
                f"C-grid ocean: water column too small. "
                f"min(eta+H)={min_wc:.6g} m, "
                f"threshold={self.config.min_water_column_m:.6g} m",
            )

        if eta_abs > self.config.max_abs_eta_m:
            raise ValueError(
                f"C-grid ocean: |eta|={eta_abs:.3g} exceeds "
                f"threshold {self.config.max_abs_eta_m:.3g}",
            )

        if any_wet_h and (
            T_min < self.config.temperature_min_c
            or T_max > self.config.temperature_max_c
        ):
            raise ValueError(
                f"C-grid ocean: T range [{T_min:.3f}, {T_max:.3f}] "
                f"outside bounds [{self.config.temperature_min_c:.3f}, "
                f"{self.config.temperature_max_c:.3f}]",
            )

        if any_wet_h and (
            S_min < self.config.salinity_min_psu
            or S_max > self.config.salinity_max_psu
        ):
            raise ValueError(
                f"C-grid ocean: S range [{S_min:.3f}, {S_max:.3f}] "
                f"outside bounds [{self.config.salinity_min_psu:.3f}, "
                f"{self.config.salinity_max_psu:.3f}]",
            )

    def integrate(
        self,
        state: LatLonCGridOceanState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[LatLonCGridOceanState, list[LatLonCGridOceanState]]:
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : LatLonCGridOceanState
        duration : float
            Total integration time [seconds].
        dt : float
            Time step [seconds].
        save_every : int
            Save state every N steps.

        Returns
        -------
        final_state, trajectory
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt!r}")
        if duration < 0.0:
            raise ValueError(f"duration must be >= 0, got {duration!r}")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every!r}")

        n_steps = int(duration / dt)
        if duration > 0.0 and n_steps < 1:
            raise ValueError(
                f"zero steps; increase duration or reduce dt "
                f"(duration={duration!r}, dt={dt!r})",
            )

        trajectory = [state]
        step_fn = self.step_checked if self.config.enable_runtime_checks else self.step
        for i in range(n_steps):
            state = step_fn(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def integrate_scan(
        self,
        state: LatLonCGridOceanState,
        n_steps: int,
        dt: float,
    ) -> tuple[LatLonCGridOceanState, LatLonCGridOceanState]:
        """Integrate using jax.lax.scan (differentiable).

        Parameters
        ----------
        state : LatLonCGridOceanState
        n_steps : int
        dt : float

        Returns
        -------
        final_state, trajectory (stacked)

        Notes
        -----
        For AB2: the initial carry must have Field (not None) for
        ``T_flux_div_prev`` / ``S_flux_div_prev`` so the pytree
        structure is stable across scan iterations.  If they are None,
        this method pre-initializes them with zero-filled Fields.
        The first step then uses AB2 with zero previous tendency,
        giving an effective coefficient of (3/2+eps) ≈ 1.6 rather
        than the Euler fallback of 1.0.  At typical CFL values
        (≤ 0.3) this is stable.
        """
        # Pre-initialize AB2 carry fields so the pytree structure
        # is stable across scan iterations (None → Field transition
        # would crash jax.lax.scan).
        if (self.config.tracer_time_integrator == "ab2"
                and state.T_flux_div_prev is None):
            from legoesm.core.field import Field
            _dims_fd = ("lat", "lon", "level")
            _zero = jnp.zeros_like(state.T.data)
            state = state._replace(
                T_flux_div_prev=Field(
                    data=_zero, name="T_flux_div_prev",
                    dims=_dims_fd, units="m/s"),
                S_flux_div_prev=Field(
                    data=_zero, name="S_flux_div_prev",
                    dims=_dims_fd, units="m/s"),
            )

        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, xs=None, length=n_steps,
        )
        return final_state, trajectory
