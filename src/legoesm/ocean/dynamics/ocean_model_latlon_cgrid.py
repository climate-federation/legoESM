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

from legoesm.core.precision import cast_pytree
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
)
from legoesm.ocean.state import (
    LatLonCGridOceanState,
    LatLonCGridOceanConfig,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
)
from legoesm.ocean.freshwater import freshwater_eta_tendency, virtual_salt_flux


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

    # h at u-faces
    h_west = jnp.roll(h_k, 1, axis=1)
    h_u = 0.5 * (h_west + h_k)
    h_u = jnp.concatenate([h_u, h_u[:, 0:1, :]], axis=1)

    # h at v-faces
    h_v_interior = 0.5 * (h_k[:-1] + h_k[1:])
    zero_h = jnp.zeros((1, n_lon, nlev), dtype=h_k.dtype)
    h_v = jnp.concatenate([zero_h, h_v_interior, zero_h], axis=0)

    # --- Depth-averaged velocity (barotropic component) ---
    H_u = jnp.maximum(jnp.sum(h_u, axis=-1), min_water_col)
    U_bar = jnp.sum(u * h_u, axis=-1) / H_u * u_mask

    H_v = jnp.maximum(jnp.sum(h_v, axis=-1), min_water_col)
    V_bar = jnp.sum(v * h_v, axis=-1) / H_v * v_mask

    # --- Perturbation velocity ---
    u_prime = (u - U_bar[..., jnp.newaxis]) * u_mask_3d
    v_prime = (v - V_bar[..., jnp.newaxis]) * v_mask_3d

    # --- Coriolis parameter at face points ---
    f_cell = grid.f.astype(u.dtype)

    # f at u-points: average of flanking cells
    f_u = 0.5 * (jnp.roll(f_cell, 1, axis=1) + f_cell)
    f_u = jnp.concatenate([f_u, f_u[:, 0:1]], axis=1)  # (n_lat, n_lon+1)

    # f at v-points: average of flanking cells
    f_v_interior = 0.5 * (f_cell[:-1] + f_cell[1:])
    f_v = jnp.concatenate(
        [f_cell[0:1], f_v_interior, f_cell[-1:]], axis=0,
    )  # (n_lat+1, n_lon)

    # --- Forward step: update u' using old v' ---
    # Average v' to u-points (Sadourny 4-point average)
    v_west = jnp.roll(v_prime, 1, axis=1)
    v_at_u = 0.25 * (v_prime[:-1] + v_prime[1:]
                      + v_west[:-1] + v_west[1:])
    v_at_u = jnp.concatenate([v_at_u, v_at_u[:, 0:1, :]], axis=1)

    u_prime_new = (u_prime + dt * f_u[:, :, jnp.newaxis] * v_at_u) * u_mask_3d

    # --- Backward step: update v' using NEW u' ---
    # Average u'_new to v-points (Sadourny 4-point average)
    u_at_v_interior = 0.25 * (
        u_prime_new[:-1, :-1] + u_prime_new[:-1, 1:]
        + u_prime_new[1:, :-1] + u_prime_new[1:, 1:]
    )
    zero_row = jnp.zeros((1, n_lon, nlev), dtype=u.dtype)
    u_at_v = jnp.concatenate([zero_row, u_at_v_interior, zero_row], axis=0)

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
    ):
        self.grid = grid
        self.z_coord = z_coord
        self.config = config or LatLonCGridOceanConfig()
        self._validate_config(self.config)
        self._cfl_checked = False

        if self.config.physics is not None:
            from legoesm.ocean.physics.combined import make_ocean_physics
            self._physics_fn = make_ocean_physics(self.config.physics)
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
        # grid.dx and grid.dy are "distance over 2 cells", so cell width = dx/2
        dx_min = min(float(jnp.min(self.grid.dx)) / 2.0, self.grid.dy / 2.0)

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

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LatLonCGridOceanState, dt: float,
             freshwater=None, surface_forcing=None,
             sponge=None) -> LatLonCGridOceanState:
        """Advance one time step using split-explicit stepping.

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

        # Compute layer thickness at u/v faces for depth-averaging
        h_k_pre = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m,
        )
        # h at u-faces
        h_u_pre = 0.5 * (jnp.roll(h_k_pre, 1, axis=1) + h_k_pre)
        h_u_pre = jnp.concatenate([h_u_pre, h_u_pre[:, 0:1, :]], axis=1)
        H_u_pre = jnp.maximum(jnp.sum(h_u_pre, axis=-1), 1e-10)
        # h at v-faces
        h_v_pre_int = 0.5 * (h_k_pre[:-1] + h_k_pre[1:])
        _n_lon = h_k_pre.shape[1]
        _nlev = h_k_pre.shape[2]
        _z_row = jnp.zeros((1, _n_lon, _nlev), dtype=h_k_pre.dtype)
        h_v_pre = jnp.concatenate([_z_row, h_v_pre_int, _z_row], axis=0)
        H_v_pre = jnp.maximum(jnp.sum(h_v_pre, axis=-1), 1e-10)

        # Depth-averaged tendency → slow forcing for barotropic solver
        F_slow_u = jnp.sum(du_dt * h_u_pre, axis=-1) / H_u_pre * state.u_mask.data
        F_slow_v = jnp.sum(dv_dt * h_v_pre, axis=-1) / H_v_pre * state.v_mask.data

        # Perturbation tendency (depth-mean removed) → applied to 3D
        du_dt_pert = du_dt - F_slow_u[..., jnp.newaxis]
        dv_dt_pert = dv_dt - F_slow_v[..., jnp.newaxis]

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

        # 6. Barotropic substeps with slow-forcing coupling
        dt_s = dt / self.config.n_barotropic_substeps

        # Freshwater mass flux for barotropic continuity equation
        F_slow_eta = None
        if freshwater is not None and self.config.freshwater_closure != "none":
            F_slow_eta = freshwater_eta_tendency(
                freshwater, self.config.rho_0,
            ) * state.land_mask.data

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
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _interp_to_v_points,
            _upwind_to_u_points,
            _upwind_to_v_points,
            _tvd_to_u_points,
            _tvd_to_v_points,
        )
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            divergence_cgrid, interp_cell_to_uface,
        )

        mask = state.land_mask.data

        # Layer thickness at face points
        h_u_old = interp_cell_to_uface(h_k_old)  # (n_lat, n_lon+1, nlev)
        h_v_old = _interp_to_v_points(h_k_old)  # (n_lat+1, n_lon, nlev)
        H_u_old = jnp.sum(h_u_old, axis=-1)     # (n_lat, n_lon+1)
        H_v_old = jnp.sum(h_v_old, axis=-1)     # (n_lat+1, n_lon)

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
        Hu_3d = jnp.sum(u_3d * h_u_old, axis=-1)
        Hv_3d = jnp.sum(v_3d * h_v_old, axis=-1)
        delta_U = (Hu_avg - Hu_3d) / jnp.maximum(H_u_old, 1e-10)
        delta_V = (Hv_avg - Hv_3d) / jnp.maximum(H_v_old, 1e-10)
        u_corrected = u_3d + delta_U[..., jnp.newaxis]
        v_corrected = v_3d + delta_V[..., jnp.newaxis]

        # Per-layer mass fluxes with full 3D velocity structure.
        # Unlike the previous barotropic-only distribution (which gave
        # uniform velocity at all depths and identically zero w),
        # this preserves baroclinic shear and produces non-zero vertical
        # velocity from Ekman pumping/suction.
        mass_flux_u = h_u_old * u_corrected * state.u_mask.data[..., jnp.newaxis]
        mass_flux_v = h_v_old * v_corrected * state.v_mask.data[..., jnp.newaxis]

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
        from legoesm.ocean.vertical import (
            diagnose_w_from_flux_div,
            flux_form_vertical_tracer_advection,
            flux_form_vertical_tracer_advection_tvd,
        )

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
            from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon import (
                gm_redi_tracer_tendency_latlon,
            )
            dT_gm, dS_gm = gm_redi_tracer_tendency_latlon(
                T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
                self.grid, self.z_coord, self.config.gm_redi,
                eos=self.config.eos, eos_linear=self.config.eos_linear,
            )
            T_mid = T_mid + dt * dT_gm * mask_3d
            S_mid = S_mid + dt * dS_gm * mask_3d

        if self.config.tracer_advection == "som":
            # SOM (Prather 1986): Second Order Moments advection (#210).
            # Advects polynomial sub-cell distributions (mean + 9 moments)
            # via directional sweeps.  Near-zero spurious diapycnal mixing,
            # fully differentiable (no limiter).
            from legoesm.ocean.advection_som import som_advect_tracers
            from legoesm.core.field import Field

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
          for tr_name in ['T', 'S']:
            tr = T_mid if tr_name == 'T' else S_mid

            if self.config.tracer_advection == "ppm_fct":
                # PPM + FCT: 4th-order PPM accuracy with Zalesak limiter
                # for guaranteed monotonicity. Stable at any CFL.
                from legoesm.ocean.advection import fct_tracer_advection
                div_hut, vert_flux_div = fct_tracer_advection(
                    tr, mass_flux_u, mass_flux_v, w_baro,
                    h_k_old, self.grid, dt,
                )
            elif self.config.tracer_advection == "ppm":
                # Raw PPM (unstable at low CFL — for testing only)
                from legoesm.ocean.advection import (
                    ppm_to_u_points, ppm_to_v_points,
                    flux_form_vertical_tracer_advection_ppm,
                )
                tr_u = ppm_to_u_points(tr, mass_flux_u)
                tr_v = ppm_to_v_points(tr, mass_flux_v)
                tracer_flux_u = mass_flux_u * tr_u
                tracer_flux_v = mass_flux_v * tr_v
                div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, self.grid)
                vert_flux_div = flux_form_vertical_tracer_advection_ppm(
                    tr, w_baro, h_k_old, dt)
            elif self.config.tracer_advection == "dst3":
                # DST-3 with Sweby limiter, applied independently per
                # direction. Third-order in space and time, monotone (#210).
                from legoesm.ocean.advection import (
                    dst3_to_u_points, dst3_to_v_points,
                    flux_form_vertical_tracer_advection_dst3,
                )
                tr_u = dst3_to_u_points(tr, mass_flux_u, h_u_old, self.grid, dt)
                tr_v = dst3_to_v_points(tr, mass_flux_v, h_v_old, self.grid, dt)
                tracer_flux_u = mass_flux_u * tr_u
                tracer_flux_v = mass_flux_v * tr_v
                div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, self.grid)
                vert_flux_div = flux_form_vertical_tracer_advection_dst3(
                    tr, w_baro, h_k_old, dt)
            elif self.config.tracer_advection == "dst3_multidim":
                # DST-3 with multi-dimensional transverse correction.
                # More accurate at diagonal flows but less robust.
                from legoesm.ocean.advection import multidim_tracer_advection
                div_hut, vert_flux_div = multidim_tracer_advection(
                    tr, mass_flux_u, mass_flux_v, w_baro,
                    h_k_old, h_u_old, h_v_old, self.grid, dt,
                )
            else:
                # Horizontal flux: div(mf_k * T_face)
                if self.config.tracer_advection == "tvd":
                    tr_u = _tvd_to_u_points(tr, mass_flux_u)
                    tr_v = _tvd_to_v_points(tr, mass_flux_v)
                else:
                    tr_u = _upwind_to_u_points(tr, mass_flux_u)
                    tr_v = _upwind_to_v_points(tr, mass_flux_v)
                tracer_flux_u = mass_flux_u * tr_u
                tracer_flux_v = mass_flux_v * tr_v
                div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, self.grid)

                # Flux-form vertical advection:
                # TVD Van Leer reduces implicit numerical diffusion from
                # K_num~|w|*dz/2 (upwind) to ~0 in smooth regions (#209).
                if self.config.tracer_advection == "tvd":
                    vert_flux_div = flux_form_vertical_tracer_advection_tvd(
                        tr, w_baro, h_k_old, dt)
                else:
                    vert_flux_div = flux_form_vertical_tracer_advection(tr, w_baro)

            # Full flux-form tracer update:
            # h_new * T_new = h_old * T_old - dt * vert_flux_div - dt * div_h(mf*T_face)
            hT_new = h_k_old * tr - dt * vert_flux_div - dt * div_hut
            tr_new = hT_new / jnp.maximum(h_k_new, 1e-10)
            tr_new = jnp.where(mask_3d > 0.5, tr_new, tr)
            if tr_name == 'T':
                T_corrected = tr_new
            else:
                S_corrected = tr_new

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
            S_fw = state_new.S.data.at[..., 0].add(dt * dS_fw * mask)
            state_new = state_new._replace(
                S=state_new.S.replace(data=S_fw),
            )

        # 9. Conservation fixers
        if self.config.use_conservation_fixer:
            from legoesm.ocean.conservation import ocean_conservation_fixer
            state_new = ocean_conservation_fixer(
                state_new, state, self.grid, self.z_coord, self.config,
            )

        return cast_pytree(state_new, None, "storage")

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
        from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
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
        """
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, xs=None, length=n_steps,
        )
        return final_state, trajectory
