"""MPAS ocean model on Voronoi meshes.

Orchestrates split-explicit time stepping: baroclinic (slow) tendencies
computed by :func:`mpas_ocean_baroclinic_tendencies`, then barotropic
(fast) substeps update the free surface and reconcile velocities.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from legoesm.core.precision import cast_pytree
from legoesm.core.state import MPASOceanState, MPASOceanTendencies
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    diagnose_w_from_flux_div,
    flux_form_vertical_tracer_advection,
)
from legoesm.core.operators_voronoi import divergence_cell_3d
from legoesm.ocean.dynamics.ocean_pe_mpas import mpas_ocean_baroclinic_tendencies
from legoesm.ocean.dynamics.barotropic_mpas import (
    barotropic_substeps_mpas,
    reconcile_3d_velocity,
)
from legoesm.ocean.conservation_mpas import mpas_ocean_conservation_fixer
from legoesm.ocean.freshwater import FreshwaterForcing, freshwater_eta_tendency
from legoesm.core.operators_voronoi import tangential_velocity_3d


def _forward_backward_coriolis_mpas_3d(
    u_3d: jnp.ndarray,
    dt: float,
    mesh,
    z_coord: OceanZStarCoordinate,
    config: MPASOceanConfig,
    mask: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
) -> jnp.ndarray:
    """Apply semi-implicit Coriolis to baroclinic perturbation velocity.

    Uses the same trapezoidal predictor-corrector scheme as the barotropic
    solver (barotropic_mpas.py), but operates on the perturbation velocity
    u' = u - u_bar only.  This avoids double-counting with the barotropic
    solver's Coriolis treatment of the depth-mean flow.

    Matches the latlon C-grid pattern in _forward_backward_coriolis_3d
    (ocean_model_latlon_cgrid.py).

    Parameters
    ----------
    u_3d : (nEdges, nlev) full 3D velocity
    dt : time step [s]
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig
    mask : (nCells,) land mask
    eta, H_bathy : (nCells,) for layer thickness computation

    Returns
    -------
    u_3d_new : (nEdges, nlev) full velocity with Coriolis applied to perturbation
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = (mask[c1] * mask[c2])[:, jnp.newaxis]  # (nEdges, 1)

    # Layer thickness at edges for depth averaging
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    h_e = 0.5 * (h_k[c1] + h_k[c2])  # (nEdges, nlev)
    H_e = jnp.maximum(jnp.sum(h_e, axis=1, keepdims=True), config.min_water_column_m)

    # Depth-averaged velocity
    u_bar = jnp.sum(u_3d * h_e, axis=1, keepdims=True) / H_e  # (nEdges, 1)
    u_bar = u_bar * edge_mask

    # Perturbation velocity
    u_prime = (u_3d - u_bar) * edge_mask  # (nEdges, nlev)

    # Coriolis parameter at edges
    f_e = mesh.fEdge[:, jnp.newaxis]  # (nEdges, 1)

    # Semi-implicit (trapezoidal predictor-corrector):
    # 1. Predict with old tangential velocity
    v_t_old = tangential_velocity_3d(u_prime, mesh)
    u_prime_star = (u_prime + dt * f_e * v_t_old) * edge_mask

    # 2. Correct with averaged tangential velocity
    v_t_star = tangential_velocity_3d(u_prime_star, mesh)
    u_prime_new = (u_prime + dt * f_e * 0.5 * (v_t_old + v_t_star)) * edge_mask

    # Reconstruct full velocity
    return u_prime_new + u_bar


class MPASOceanModel:
    """MPAS ocean model with split-explicit time stepping.

    Parameters
    ----------
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig or None
    """

    def __init__(
        self,
        mesh: VoronoiMesh,
        z_coord: OceanZStarCoordinate,
        config: MPASOceanConfig | None = None,
    ):
        self.mesh = mesh
        self.z_coord = z_coord
        self.config = config or MPASOceanConfig()

        if self.config.physics is not None:
            from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
            self._physics_fn = make_mpas_ocean_physics(self.config.physics)
        else:
            self._physics_fn = None

    def tendencies(
        self,
        state: MPASOceanState,
        freshwater: FreshwaterForcing | None = None,
        surface_forcing=None,
    ) -> MPASOceanTendencies:
        """Compute baroclinic tendencies."""
        return mpas_ocean_baroclinic_tendencies(
            state, self.mesh, self.z_coord, self.config,
            freshwater=freshwater,
            physics_fn=self._physics_fn,
            surface_forcing=surface_forcing,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: MPASOceanState,
        dt: float,
        freshwater: FreshwaterForcing | None = None,
        surface_forcing=None,
    ) -> MPASOceanState:
        """Advance one full timestep (baroclinic + barotropic).

        Parameters
        ----------
        state : MPASOceanState
        dt : float
            Baroclinic timestep [s].
        freshwater : FreshwaterForcing or None
            Freshwater forcing (P, E, runoff, ice). If None, no freshwater.
        surface_forcing : optional
            External surface forcing passed to the physics pipeline.

        Returns
        -------
        MPASOceanState
        """
        state = cast_pytree(state, None, "compute")

        config = self.config
        mesh = self.mesh
        z_coord = self.z_coord
        mask = state.land_mask.data

        # 1. Compute baroclinic tendencies
        tend = self.tendencies(state, freshwater=freshwater,
                               surface_forcing=surface_forcing)

        # 2. Update tracers (forward Euler)
        T_new = state.T.data + dt * tend.dT_dt.data
        S_new = state.S.data + dt * tend.dS_dt.data

        # Fill land cells with ocean-neighbor average (Neumann BC) so that
        # subsequent operators see smooth values at coastlines instead of
        # the sharp ocean-to-zero discontinuity that `* mask` would create.
        c1_m = mesh.cellsOnEdge[0]
        c2_m = mesh.cellsOnEdge[1]
        m1 = mask[c1_m, jnp.newaxis]
        m2 = mask[c2_m, jnp.newaxis]
        nbr_sum = jnp.zeros_like(T_new).at[c1_m].add(T_new[c2_m] * m2)
        nbr_sum = nbr_sum.at[c2_m].add(T_new[c1_m] * m1)
        nbr_cnt = jnp.zeros_like(T_new).at[c1_m].add(m2)
        nbr_cnt = nbr_cnt.at[c2_m].add(m1)
        nbr_avg_T = nbr_sum / jnp.maximum(nbr_cnt, 1.0)
        mask_e = mask[:, jnp.newaxis]
        T_new = jnp.where(mask_e > 0.5, T_new, nbr_avg_T)

        nbr_sum_S = jnp.zeros_like(S_new).at[c1_m].add(S_new[c2_m] * m2)
        nbr_sum_S = nbr_sum_S.at[c2_m].add(S_new[c1_m] * m1)
        nbr_avg_S = nbr_sum_S / jnp.maximum(nbr_cnt, 1.0)
        S_new = jnp.where(mask_e > 0.5, S_new, nbr_avg_S)

        # 3. Update 3D velocity with baroclinic tendency (non-Coriolis)
        u_baro = state.u.data + dt * tend.du_dt.data

        # 3b. Forward-backward Coriolis on perturbation velocity
        # Coriolis is excluded from the baroclinic tendencies (issue #103)
        # and applied here to the perturbation velocity u' = u - u_bar only.
        # The barotropic solver handles depth-mean Coriolis separately.
        u_baro = _forward_backward_coriolis_mpas_3d(
            u_baro, dt, mesh, z_coord, config, mask,
            state.eta.data, state.H_bathy.data,
        )

        # 4. Barotropic substeps
        # The baroclinic tendency is already applied to u_baro, so the
        # barotropic solver computes u_bar from the updated velocity.
        # No F_slow_u is needed (same pattern as cubed-sphere barotropic.py).
        n_sub = config.n_barotropic_substeps
        dt_baro = dt / n_sub

        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]

        # Create intermediate state with updated velocity for barotropic
        state_for_baro = state._replace(
            u=state.u.replace(data=u_baro),
        )

        # Freshwater mass flux for barotropic continuity equation
        F_slow_eta = None
        if freshwater is not None and config.freshwater_closure != "none":
            F_slow_eta = freshwater_eta_tendency(freshwater, config.rho_0) * mask

        eta_new, u_bar_new, Hu_avg = barotropic_substeps_mpas(
            state_for_baro, mesh, z_coord, config, dt_baro, n_sub,
            F_slow_eta=F_slow_eta,
        )

        # 5. Layer thicknesses before and after barotropic
        h_k_old = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord,
            min_water_column_m=config.min_water_column_m,
        )  # (nCells, nlev)
        h_k_new = compute_layer_thickness(
            eta_new, state.H_bathy.data, z_coord,
            min_water_column_m=config.min_water_column_m,
        )  # (nCells, nlev)

        # 6. Reconcile 3D velocity
        # Compute u_bar_old from the UPDATED state (state_for_baro),
        # not the original. This ensures depth_avg(u_3d_new) = u_bar_new.
        h_e_k = 0.5 * (h_k_old[c1] + h_k_old[c2])  # (nEdges, nlev)
        H_total = jnp.maximum(state.eta.data + state.H_bathy.data,
                              config.min_water_column_m)
        H_e = 0.5 * (H_total[c1] + H_total[c2])
        u_bar_old = jnp.sum(u_baro * h_e_k, axis=1) / jnp.maximum(H_e, 1e-10)

        u_3d_new = reconcile_3d_velocity(
            u_baro, u_bar_old, u_bar_new, mesh, mask,
        )

        # 7. Barotropic correction for transport-consistent tracer advection
        #
        # Correct the reconciled 3D velocity so that depth-integrated
        # transport matches the time-averaged barotropic transport Hu_avg
        # exactly.  This ensures mass flux consistency between the
        # barotropic continuity equation (which produced eta_new) and
        # the tracer transport (Hallberg & Adcroft 2009, issue #145).
        #
        # The correction is a uniform (depth-independent) velocity shift:
        #   delta_u = (Hu_avg - sum_k(u_3d * h_e)) / H_e
        # This preserves baroclinic shear while matching Hu_avg.
        edge_mask = mask[c1] * mask[c2]
        H_e_old = jnp.sum(h_e_k, axis=1)  # (nEdges,)
        Hu_3d = jnp.sum(u_3d_new * h_e_k, axis=1)  # (nEdges,)
        delta_u = (Hu_avg - Hu_3d) / jnp.maximum(H_e_old, 1e-10)
        u_transport = u_3d_new + delta_u[:, jnp.newaxis]  # (nEdges, nlev)

        # Per-layer mass fluxes with full 3D velocity structure.
        # Preserves baroclinic shear and produces non-zero w from
        # Ekman pumping/suction (unlike uniform barotropic distribution
        # which gives w ≡ 0).
        mass_flux = h_e_k * u_transport * edge_mask[:, jnp.newaxis]

        # 8. Diagnose vertical velocity from per-layer flux divergence
        #
        # From continuity: dh_k/dt + div_h(h_k * u_k) + w_{k-1/2} - w_{k+1/2} = 0
        # We accumulate div_h(h_k * u_k) bottom-up to get w at interfaces.
        flux_div_3d = divergence_cell_3d(mass_flux, mesh)  # (nCells, nlev)

        w = diagnose_w_from_flux_div(
            flux_div_3d, z_coord, thickness_weighted=True,
        )  # (nCells, nlev+1)

        # 9. Flux-form tracer transport (horizontal + vertical)
        #
        # Both horizontal and vertical transport use the barotropic-averaged
        # per-layer mass fluxes for consistency:
        #   h_new * T_new = h_old * T_mid
        #     - dt * div_h(mass_flux * T_face_h)   [horizontal flux]
        #     - dt * (w * T_face_v)                 [vertical flux]
        #
        # T_mid contains diffusion+physics from the Euler step (step 2).
        # Advection (horizontal + vertical) is applied here.
        # This matches the latlon C-grid algorithm (ocean_model_latlon_cgrid.py).
        mask_3d = mask[:, jnp.newaxis]  # (nCells, 1)

        for tr_name in ['T', 'S']:
            tr = T_new if tr_name == 'T' else S_new

            # Horizontal flux: upwind interpolation to edges
            # MPAS convention: u > 0 means flow from c1 to c2 (edge normal).
            # Upwind: use tracer from the upstream cell.
            tr_c1 = tr[c1]  # (nEdges, nlev)
            tr_c2 = tr[c2]  # (nEdges, nlev)
            tr_upwind = jnp.where(mass_flux > 0, tr_c1, tr_c2)
            tracer_flux = mass_flux * tr_upwind  # (nEdges, nlev)
            div_hut = divergence_cell_3d(tracer_flux, mesh)  # (nCells, nlev)

            # Vertical flux divergence (upwind, zero at surface/bottom)
            vert_flux_div = flux_form_vertical_tracer_advection(tr, w)

            # Full flux-form tracer update:
            # h_new * T_new = h_old * T_mid - dt * vert - dt * horiz
            hT_new = h_k_old * tr - dt * vert_flux_div - dt * div_hut
            tr_new = hT_new / jnp.maximum(h_k_new, 1e-10)
            tr_new = jnp.where(mask_3d > 0.5, tr_new, 0.0)

            if tr_name == 'T':
                T_corrected = tr_new
            else:
                S_corrected = tr_new

        # Final state construction with explicit land masking
        T_final = T_corrected
        S_final = S_corrected
        
        state_new = MPASOceanState(
            u=state.u.replace(data=u_3d_new),
            T=state.T.replace(data=T_final),
            S=state.S.replace(data=S_final),
            eta=state.eta.replace(data=eta_new * mask),
            H_bathy=state.H_bathy,
            land_mask=state.land_mask,
        )

        # 7. Conservation fixers
        if config.use_conservation_fixer:
            state_new = mpas_ocean_conservation_fixer(
                state_new, state, mesh, z_coord, config,
            )

        return cast_pytree(state_new, None, "storage")

    def step_checked(
        self,
        state: MPASOceanState,
        dt: float,
        freshwater=None,
        surface_forcing=None,
    ) -> MPASOceanState:
        """Advance one timestep with host-side runtime validation.

        Unlike the previous implementation which silently clipped tracers,
        this raises on out-of-bounds values so the caller sees the failure.
        """
        state_new = self.step(state, dt, freshwater=freshwater,
                              surface_forcing=surface_forcing)
        if self.config.enable_runtime_checks:
            self._assert_runtime_invariants(state_new)
        return state_new

    def _assert_runtime_invariants(self, state: MPASOceanState) -> None:
        """Host-side runtime checks (matching cubed-sphere ocean model)."""
        mask = state.land_mask.data
        wet = mask > 0.5

        finite_ok = bool(
            jnp.all(jnp.isfinite(state.u.data))
            & jnp.all(jnp.isfinite(state.T.data))
            & jnp.all(jnp.isfinite(state.S.data))
            & jnp.all(jnp.isfinite(state.eta.data))
        )
        if not finite_ok:
            raise FloatingPointError(
                "MPAS ocean runtime check failed: non-finite state detected"
            )

        config = self.config
        T_wet = state.T.data[wet[:, jnp.newaxis].broadcast_to(state.T.data.shape)]
        S_wet = state.S.data[wet[:, jnp.newaxis].broadcast_to(state.S.data.shape)]

        if T_wet.size > 0:
            T_min_val = float(jnp.min(T_wet))
            T_max_val = float(jnp.max(T_wet))
            if T_min_val < config.temperature_min_c or T_max_val > config.temperature_max_c:
                raise ValueError(
                    f"MPAS ocean runtime check failed: T out of bounds "
                    f"[{T_min_val:.2f}, {T_max_val:.2f}] vs "
                    f"[{config.temperature_min_c}, {config.temperature_max_c}]"
                )
        if S_wet.size > 0:
            S_min_val = float(jnp.min(S_wet))
            S_max_val = float(jnp.max(S_wet))
            if S_min_val < config.salinity_min_psu or S_max_val > config.salinity_max_psu:
                raise ValueError(
                    f"MPAS ocean runtime check failed: S out of bounds "
                    f"[{S_min_val:.2f}, {S_max_val:.2f}] vs "
                    f"[{config.salinity_min_psu}, {config.salinity_max_psu}]"
                )

    def integrate(
        self,
        state: MPASOceanState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ):
        """Forward integration.

        Parameters
        ----------
        state : MPASOceanState
        duration : float
            Total integration time [s].
        dt : float
            Timestep [s].
        save_every : int
            Save trajectory every N steps.

        Returns
        -------
        (final_state, trajectory)
        """
        n_steps = int(duration / dt)
        trajectory = []

        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def integrate_scan(
        self,
        state: MPASOceanState,
        n_steps: int,
        dt: float,
    ):
        """Differentiable integration via jax.lax.scan.

        Parameters
        ----------
        state : MPASOceanState
        n_steps : int
        dt : float

        Returns
        -------
        (final_state, trajectory)
        """
        def scan_fn(carry, _):
            s = self.step(carry, dt)
            return s, s

        final, trajectory = jax.lax.scan(scan_fn, state, None, length=n_steps)
        return final, trajectory
