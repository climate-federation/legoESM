"""Spectral ocean primitive equations on the Gaussian grid.

Uses the pseudospectral (SH transform) method in the
vorticity-divergence formulation, following spectral_pe.py.

Prognostic variables (spectral space):
    vor_hat  : Vorticity SH coefficients, (n_sh, nlev)
    div_hat  : Divergence SH coefficients, (n_sh, nlev)
    T_hat    : Temperature SH coefficients, (n_sh, nlev)
    S_hat    : Salinity SH coefficients, (n_sh, nlev)
    eta_hat  : Sea surface height SH coefficients, (n_sh,)

Land masking: applied in grid space before every SH analysis.
Gibbs oscillations near coastlines controlled by spectral hyperdiffusion.

Equations (Boussinesq hydrostatic, vorticity-divergence form):
    d(vor)/dt  = -div((vor+f)*v) + curl(vert_adv + mixing)
    d(div)/dt  = curl((vor+f)*v) - lap(K + p'/rho_0) + div(vert_adv + mixing)
    d(T)/dt    = -div(T*v) + T*div(v) - w*dT/dz + K_h*lap(T) + vert_diff(T)
    d(S)/dt    = -div(S*v) + S*div(v) - w*dS/dz + K_h*lap(S) + vert_diff(S)
    d(eta)/dt  = -sum_k(div * h_k)
"""

from __future__ import annotations

from functools import partial
import warnings

import jax
import jax.numpy as jnp

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)

from legoesm.core.field import Field
from legoesm.core.operators import is_distributed
from legoesm.parallel.reductions import global_sum_mpi
from legoesm.grids.gaussian import (
    GaussianGrid,
    sh_analysis,
    sh_analysis_3d,
    sh_synthesis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
    uv_from_vordiv_3d,
    spectral_hyperdiffusion,
    spectral_hyperdiffusion_3d,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm.parallel.metal import place_spectral_grid
from legoesm.ocean.eos import compute_hydrostatic_pressure, make_eos_fn, scale_depth
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
    upwind_vertical_gradient,
)
from legoesm.ocean.state import SpectralOceanState, SpectralOceanConfig
from legoesm.ocean.physics.mixing import vertical_diffusion


def _spectral_cell_area(grid: GaussianGrid) -> jnp.ndarray:
    """Per-cell area on the Gaussian grid (m^2), shape (n_lat, n_lon)."""
    dlon = 2.0 * jnp.pi / grid.n_lon
    return (grid.radius ** 2) * grid.weights[:, jnp.newaxis] * dlon


def _spectral_global_sum(local_value: jnp.ndarray) -> jnp.ndarray:
    """MPI-aware global sum for spectral ocean reductions."""
    if is_distributed():
        return global_sum_mpi(local_value)
    return local_value


def _spectral_conservation_fixer(
    state_new: SpectralOceanState,
    state_old: SpectralOceanState,
    grid: GaussianGrid,
    z_coord: OceanZStarCoordinate,
    config: SpectralOceanConfig,
) -> SpectralOceanState:
    """Apply volume/heat/salt conservation corrections in grid space."""
    mask = state_old.land_mask_grid.data
    mask_3d = mask[..., jnp.newaxis]
    area = _spectral_cell_area(grid)
    weighted_area = mask * area
    # Batch the three 2D syntheses (eta_old, eta_new, H_bathy) along a
    # trailing axis — same passive-trailing-axis exploit as Loop 144/151
    # but extended with H_bathy_hat.  3 SH-syntheses → 1.
    _ehb_pair = jnp.stack(
        [
            state_old.eta_hat.data,
            state_new.eta_hat.data,
            state_old.H_bathy_hat.data,
        ],
        axis=-1,
    )  # (n_sh, 3)
    _ehb_grid = sh_synthesis_3d(grid, _ehb_pair).real  # (n_lat, n_lon, 3)
    eta_old = _ehb_grid[..., 0] * mask
    eta_new = _ehb_grid[..., 1] * mask
    H_bathy = _ehb_grid[..., 2]
    H_bathy = jnp.maximum(H_bathy, 1.0) * mask + 1.0 * (1.0 - mask)
    eta_floor = jnp.asarray(config.min_water_column_m, dtype=eta_new.dtype) - H_bathy
    eta_new = jnp.maximum(eta_new, eta_floor) * mask

    local_vol_terms = jnp.stack(
        [
            jnp.sum(eta_old * weighted_area),
            jnp.sum(eta_new * weighted_area),
            jnp.sum(weighted_area),
        ],
    )
    vol_old, vol_new, ocean_area = _spectral_global_sum(local_vol_terms)
    eta_corr = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
    eta_fixed = jnp.maximum(eta_new + eta_corr * mask, eta_floor) * mask

    h_k_old = compute_layer_thickness(
        eta_old, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    h_k_new = compute_layer_thickness(
        eta_fixed, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )

    # Batch the four (n_sh, nlev) tracer syntheses (T_old, T_new, S_old,
    # S_new) into a single sh_synthesis_3d on a (n_sh, nlev*4) tensor.
    # 4 SH-syntheses → 1.
    n_sh_c, nlev_c = state_old.T_hat.data.shape
    _ts_stack = jnp.stack(
        [state_old.T_hat.data, state_new.T_hat.data,
         state_old.S_hat.data, state_new.S_hat.data],
        axis=-1,
    )  # (n_sh, nlev, 4)
    _ts_grid = sh_synthesis_3d(
        grid, _ts_stack.reshape(n_sh_c, nlev_c * 4),
    ).real.reshape(grid.n_lat, grid.n_lon, nlev_c, 4)
    T_old = _ts_grid[..., 0]
    T_new = _ts_grid[..., 1]
    S_old = _ts_grid[..., 2]
    S_new = _ts_grid[..., 3]

    # Five column reductions split into two groups by their h-weight:
    #   * ``h_k_old`` weight: ``T_old``, ``S_old``
    #   * ``h_k_new`` weight: ``T_new``, ``S_new``, ``ones (volume)``
    # Each group is one stacked column reduction; the area-weighted
    # outer sum then collapses with ``axis=tuple(range(weighted_area.ndim))``
    # so the 5 nested ``jnp.sum`` pairs become 2 column reductions
    # + 1 area reduction.  Concatenate in the order the
    # ``_spectral_global_sum`` contract expects:
    # ``heat_old, heat_new, salt_old, salt_new, ocean_volume``.
    _old_inner = jnp.sum(
        jnp.stack([T_old * h_k_old, S_old * h_k_old], axis=-1), axis=-2,
    )  # columns: [T_old, S_old]
    _new_inner = jnp.sum(
        jnp.stack(
            [T_new * h_k_new, S_new * h_k_new, h_k_new], axis=-1,
        ),
        axis=-2,
    )  # columns: [T_new, S_new, vol_new]
    _inner = jnp.stack(
        [
            _old_inner[..., 0],  # heat_old
            _new_inner[..., 0],  # heat_new
            _old_inner[..., 1],  # salt_old
            _new_inner[..., 1],  # salt_new
            _new_inner[..., 2],  # ocean_volume
        ],
        axis=-1,
    )
    local_tracer_terms = jnp.sum(
        _inner * weighted_area[..., None],
        axis=tuple(range(weighted_area.ndim)),
    )
    heat_old, heat_new, salt_old, salt_new, ocean_volume = _spectral_global_sum(
        local_tracer_terms,
    )
    heat_corr = (heat_old - heat_new) / jnp.maximum(ocean_volume, 1.0)
    T_fixed = T_new + heat_corr * mask_3d

    salt_corr = (salt_old - salt_new) / jnp.maximum(ocean_volume, 1.0)
    S_fixed = S_new + salt_corr * mask_3d

    # Batch the three SH analyses (T_fixed, S_fixed, eta_fixed) into a
    # single ``sh_analysis_3d`` call.  T_fixed/S_fixed contribute
    # ``nlev`` slots each along the trailing axis; eta_fixed (2D) is
    # promoted via ``[..., None]`` to a single trailing slot and
    # concatenated.  Total trailing axis = ``2*nlev + 1``.  3 SH-analyses
    # → 1.  Split the result back into (T, S) and (eta) using slot
    # indexing.
    _ts_fixed = jnp.stack([T_fixed, S_fixed], axis=-1)  # (..., nlev, 2)
    _ts_fixed_flat = _ts_fixed.reshape(grid.n_lat, grid.n_lon, nlev_c * 2)
    _ts_eta_input = jnp.concatenate(
        [_ts_fixed_flat, eta_fixed[..., jnp.newaxis]], axis=-1,
    )  # (n_lat, n_lon, 2*nlev + 1)
    _ts_eta_hat = sh_analysis_3d(grid, _ts_eta_input)  # (n_sh, 2*nlev + 1)
    _ts_fixed_hat = _ts_eta_hat[:, :nlev_c * 2].reshape(-1, nlev_c, 2)
    _eta_fixed_hat = _ts_eta_hat[:, nlev_c * 2]
    return state_new._replace(
        eta_hat=state_new.eta_hat.replace(data=_eta_fixed_hat),
        T_hat=state_new.T_hat.replace(data=_ts_fixed_hat[..., 0]),
        S_hat=state_new.S_hat.replace(data=_ts_fixed_hat[..., 1]),
    )


def spectral_ocean_tendencies(
    state: SpectralOceanState,
    grid: GaussianGrid,
    z_coord: OceanZStarCoordinate,
    config: SpectralOceanConfig,
) -> SpectralOceanState:
    """Compute spectral tendencies for the ocean PE.

    Pseudospectral workflow:
    1. Transform prognostic fields to grid space
    2. Apply land mask
    3. Compute nonlinear products (EOS, advection, pressure)
    4. Mask products in grid space
    5. SH analysis -> spectral tendencies
    6. Spectral hyperdiffusion

    Returns tendencies as same pytree structure (for SSP-RK3).
    """
    a = grid.radius
    g = config.g
    rho_0 = config.rho_0
    mask = state.land_mask_grid.data  # (n_lat, n_lon)
    mask_3d = mask[..., jnp.newaxis]  # (n_lat, n_lon, 1)

    # --- 1. Transform to grid space ---
    # Merge the (vor, div, T, S) 3D batch with the (eta, H_bathy) 2D
    # pair via ``jnp.concatenate``: trailing axis = ``nlev*4 + 2``.
    # ``sh_synthesis_3d`` treats any trailing axis as a passive batch,
    # so different "level" sizes (nlev vs 1) combine cleanly into a
    # single ``segment_sum`` + IRFFT.  6 SH syntheses → 1.  Loop 181
    # extends Loop 180 (acoustic update path).
    n_sh_t, nlev_t = state.vor_hat.data.shape
    _vdts_stack = jnp.stack(
        [
            state.vor_hat.data,
            state.div_hat.data,
            state.T_hat.data,
            state.S_hat.data,
        ],
        axis=-1,
    )  # (n_sh, nlev, 4)
    _vdts_flat = _vdts_stack.reshape(n_sh_t, nlev_t * 4)
    _vdtseh_flat = jnp.concatenate(
        [_vdts_flat, state.eta_hat.data[:, jnp.newaxis],
         state.H_bathy_hat.data[:, jnp.newaxis]],
        axis=-1,
    )  # (n_sh, nlev*4 + 2)
    _vdtseh_grid_flat = sh_synthesis_3d(grid, _vdtseh_flat)
    _vdts_grid = _vdtseh_grid_flat[..., : nlev_t * 4].reshape(
        grid.n_lat, grid.n_lon, nlev_t, 4,
    )
    vor = _vdts_grid[..., 0] * mask_3d
    div = _vdts_grid[..., 1] * mask_3d
    # Keep tracer extensions smooth across coastlines; apply mask on tendencies.
    T = _vdts_grid[..., 2]
    S = _vdts_grid[..., 3]
    eta = _vdtseh_grid_flat[..., nlev_t * 4] * mask     # (n_lat, n_lon)
    H_bathy = _vdtseh_grid_flat[..., nlev_t * 4 + 1].real
    H_bathy = jnp.maximum(H_bathy, 1.0) * mask + 1.0 * (1.0 - mask)
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.real.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta.real, eta_floor) * mask

    # --- 2. Velocities ---
    u_cos, v_cos = uv_from_vordiv_3d(
        grid, state.vor_hat.data, state.div_hat.data,
    )
    cos_lat_3d = grid.cos_lat[:, jnp.newaxis, jnp.newaxis]
    # Guard against near-polar amplification when cos(lat) becomes tiny.
    cos_lat_safe = jnp.maximum(cos_lat_3d, 1.0e-6)
    u = u_cos / cos_lat_safe * mask_3d
    v = v_cos / cos_lat_safe * mask_3d

    # --- 3. Layer thickness and Jacobian ---
    J = compute_ocean_jacobian(
        eta_safe, H_bathy.real, z_coord, min_water_column_m=config.min_water_column_m,
    )
    h_k = compute_layer_thickness(
        eta_safe, H_bathy.real, z_coord, min_water_column_m=config.min_water_column_m,
    )

    # --- 4. EOS and hydrostatic pressure ---
    T_real = T.real
    S_real = S.real
    # Two-pass EOS-pressure coupling: improves consistency versus a
    # single rho(p=0) seed evaluation in long integrations.
    eos_fn = make_eos_fn(config.eos, getattr(config, 'eos_linear', None))
    rho = eos_fn(T_real, S_real, jnp.zeros_like(T_real))
    for _ in range(2):
        p_hydro = compute_hydrostatic_pressure(
            rho,
            eta_safe,
            z_coord.dz_ref,
            J.real,
            rho_0,
            g,
        )
        rho = eos_fn(T_real, S_real, p_hydro)
    p_hydro = compute_hydrostatic_pressure(
        rho,
        eta_safe,
        z_coord.dz_ref,
        J.real,
        rho_0,
        g,
    )
    rho_prime = rho - rho_0

    # Baroclinic pressure perturbation (top-down cumsum)
    dp_layer = rho_prime * g * h_k.real
    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer  # at cell center

    # --- 5. Kinetic energy (pole-safe via oc2 transform) ---
    # KE·cos²φ avoids the 1/cos² singularity at the poles; the factor
    # is absorbed by sh_analysis_oc2_3d in the energy variable below.
    KE_cos2 = 0.5 * (u_cos.real**2 + v_cos.real**2)  # KE·cos²φ

    # --- 6. Absolute vorticity ---
    abs_vor = vor + grid.f[..., jnp.newaxis]

    # --- 7. Diagnose z-star transport velocity ---
    # Use the FLUX-FORM divergence ``∇·(h_k · v_k)`` rather than the
    # advective form ``h_k · ∇·v_k``.  On z-star with spatially-varying
    # ``h_k(η, x, y)``, the two differ by ``v_k · ∇h_k`` — small for
    # |η|/H_max ~ 1e-4 but non-zero, and using the advective form
    # makes the diagnosed ``w_euler[0]`` inconsistent with the
    # prognostic ``∂η/∂t`` (which uses the flux form, line 552), so
    # the surface kinematic BC ``w(η) = ∂η/∂t`` is silently violated.
    # Iter-77 fix: compute hu_oc2/hv_dmu in a small pre-batch and use
    # the flux-form ``div_hv`` for the w cumulative integral.

    # --- 8. Spectral operators (moved up so we can compute div_hv) ---
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    # Small pre-batch: just (hu_cos, hv_cos) → hu_oc2, hv_dmu.
    # Caches the result so the main batch below skips these.
    hu_cos_pre = h_k.real * u_cos * mask_3d
    hv_cos_pre = h_k.real * v_cos * mask_3d
    n_lat_pre, n_lon_pre, nlev_pre = hu_cos_pre.shape
    hu_oc2_pre = sh_analysis_oc2_3d(grid, hu_cos_pre)  # (n_sh, nlev)
    hv_dmu_pre = sh_analysis_dmu_3d(grid, hv_cos_pre)
    div_hv_hat_pre = im_over_a[:, jnp.newaxis] * hu_oc2_pre - one_over_a * hv_dmu_pre
    div_hv_pre = sh_synthesis_3d(grid, div_hv_hat_pre).real * mask_3d

    div_h_rev = div_hv_pre[..., ::-1]
    cumsum_rev = jnp.cumsum(div_h_rev, axis=-1)
    w_inner = -cumsum_rev[..., ::-1]
    # Pad with zero on the bottom — single Pad HLO op vs alloc-zeros
    # + concatenate.
    _pad_axes_w = ((0, 0),) * (w_inner.ndim - 1)
    w_euler = jnp.pad(w_inner, (*_pad_axes_w, (0, 1)))
    # z-star correction: subtract grid velocity so ẇ[0]=0, ẇ[nlev]=0.
    # ``w_euler[0]`` now equals ``∂η/∂t`` exactly (consistent with the
    # prognostic form at line 552 below), so the kinematic BC closes.
    sigma = (z_coord.z_half_ref + z_coord.H_max) / z_coord.H_max
    deta_dt_local = w_euler[..., 0:1]
    w = w_euler - sigma * deta_dt_local

    # --- 9-12. Vorticity fluxes + energy + vertical advection (batched) ---
    # The vor/div tendencies need oc2 and dmu of {A_vor, B_vor,
    # vert_u_cos, vert_v_cos}, plus oc2 of KE_cos2*mask and a single
    # ``sh_analysis_3d`` of the energy-variable scalar potential.
    # Compute all the grid inputs first, then batch:
    #
    # * 5 oc2 forwards (A_vor, B_vor, vert_u_cos, vert_v_cos, KE)  → 1
    # * 4 dmu forwards (A_vor, B_vor, vert_u_cos, vert_v_cos)      → 1
    # * 1 sh_analysis_3d for the scalar energy potential           — stays
    #
    # Loop 188 — fold the free-surface mass-flux pair (hu_cos for oc2,
    # hv_cos for dmu) into the same earlier batch.  Both are
    # ``h_k * u_cos|v_cos * mask_3d`` and h_k is already computed at
    # this point, so the inputs are cheap to construct here.  When
    # ``A_v == 0`` this saves the two standalone ``sh_analysis_oc2_3d``
    # / ``sh_analysis_dmu_3d`` calls in section 16; when ``A_v > 0``
    # it shrinks the velocity-vdiff batch to (vdiff_u_cos, vdiff_v_cos)
    # only — same total SH count there but avoids a redundant per-batch
    # gather of the same fields.  10 SH-analyses → 3 (with hu/hv
    # always batched in).
    A_vor = abs_vor * u_cos * mask_3d
    B_vor = abs_vor * v_cos * mask_3d

    vert_adv_u = _vertical_advection_spectral(u.real, w, z_coord, J.real)
    vert_adv_v = _vertical_advection_spectral(v.real, w, z_coord, J.real)
    vert_u_cos = vert_adv_u * grid.cos_lat[:, jnp.newaxis, jnp.newaxis] * mask_3d
    vert_v_cos = vert_adv_v * grid.cos_lat[:, jnp.newaxis, jnp.newaxis] * mask_3d

    KE_cos2_masked = KE_cos2 * mask_3d
    # Free-surface mass-flux spectral coefficients ``hu_oc2`` and
    # ``hv_dmu`` were already computed in the iter-77 pre-batch above
    # (used to build the flux-form ``div_hv`` for the w cumulative
    # integral).  Reuse them here so the main batch shrinks back from
    # 6/5 → 5/4 channels and avoids a redundant SH analysis.
    n_lat_o, n_lon_o, nlev_o = A_vor.shape
    _ocean_oc2_stack = jnp.stack(
        [A_vor, B_vor, vert_u_cos, vert_v_cos, KE_cos2_masked], axis=-1,
    )  # (..., nlev, 5)
    _ocean_dmu_stack = jnp.stack(
        [A_vor, B_vor, vert_u_cos, vert_v_cos], axis=-1,
    )  # (..., nlev, 4)
    _ocean_oc2 = sh_analysis_oc2_3d(
        grid, _ocean_oc2_stack.reshape(n_lat_o, n_lon_o, nlev_o * 5),
    ).reshape(-1, nlev_o, 5)
    _ocean_dmu = sh_analysis_dmu_3d(
        grid, _ocean_dmu_stack.reshape(n_lat_o, n_lon_o, nlev_o * 4),
    ).reshape(-1, nlev_o, 4)
    A_vor_oc2 = _ocean_oc2[..., 0]
    B_vor_oc2 = _ocean_oc2[..., 1]
    vert_u_oc2 = _ocean_oc2[..., 2]
    vert_v_oc2 = _ocean_oc2[..., 3]
    KE_oc2 = _ocean_oc2[..., 4]
    A_vor_dmu = _ocean_dmu[..., 0]
    B_vor_dmu = _ocean_dmu[..., 1]
    vert_u_dmu = _ocean_dmu[..., 2]
    vert_v_dmu = _ocean_dmu[..., 3]

    flux_vor_div = im_over_a[:, jnp.newaxis] * A_vor_oc2 - one_over_a * B_vor_dmu
    flux_vor_curl = im_over_a[:, jnp.newaxis] * B_vor_oc2 + one_over_a * A_vor_dmu

    # --- 10. Energy variable: E = K + p'/rho_0 + g*eta ---
    # Subtract the area-weighted mean pressure at each level to remove
    # spurious horizontal gradients from bathymetry variations.  The
    # global-mean pressure gradient is identically zero on a sphere,
    # so this does not affect the physics — it only suppresses spectral
    # ringing from the land-ocean boundary discontinuity.
    #
    # The barotropic PGF g*eta is included here as a depth-uniform term.
    # At T21 with dt=300s and H<=5500m, the maximum barotropic gravity
    # wave eigenfrequency gives omega*dt ~ 0.2 << 1.73 (RK3 stability
    # limit), so explicit treatment is stable.  The eta_hyperdiff
    # provides additional damping of high-wavenumber barotropic modes
    # as a safety margin.
    weights = grid.weights[:, jnp.newaxis, jnp.newaxis]  # (n_lat, 1, 1)
    ocean_area = jnp.sum(mask[..., jnp.newaxis] * weights, axis=(0, 1), keepdims=True)
    ocean_area = jnp.maximum(ocean_area, _TINY)
    p_prime_mean = jnp.sum(
        p_prime * mask_3d * weights, axis=(0, 1), keepdims=True,
    ) / ocean_area
    p_prime_anom = (p_prime - p_prime_mean) * mask_3d
    # Barotropic PGF: g*eta broadcast to all levels (Boussinesq)
    g_eta_3d = (g * eta_safe)[..., jnp.newaxis]  # (n_lat, n_lon, 1)
    E_hat = KE_oc2 + sh_analysis_3d(
        grid, (p_prime_anom / rho_0 + g_eta_3d) * mask_3d,
    )

    # --- 11. Horizontal tendencies (combine flux + vert-advection) ---
    vert_vor = im_over_a[:, jnp.newaxis] * vert_v_oc2 + one_over_a * vert_u_dmu
    vert_div = im_over_a[:, jnp.newaxis] * vert_u_oc2 - one_over_a * vert_v_dmu
    dvor_hat = -flux_vor_div + vert_vor
    ddiv_hat = flux_vor_curl - grid.lap[:, jnp.newaxis] * E_hat + vert_div

    # --- 13. Tracer equations (vectorized over T, S) ---
    tracers = jnp.stack([T.real, S.real], axis=0)  # (2, n_lat, n_lon, nlev)
    tracers_hat = jnp.stack([state.T_hat.data, state.S_hat.data], axis=0)

    # Stack tracers along trailing axis and fold into the level dim so
    # the three SH analyses below (oc2, dmu, plain) each run ONCE on a
    # ``(n_lat, n_lon, nlev*n_tracers)`` tensor instead of being
    # ``vmap``'d over the leading tracer axis (which materialises a
    # separate FFT/sum kernel per tracer).  Same passive-trailing-axis
    # exploit as the velocity-vdiff + free-surface block in Section 16.
    n_tr = tracers.shape[0]
    n_lat_t, n_lon_t, nlev_t = tracers.shape[1:]
    # ``moveaxis`` brings the tracer axis to the end without a copy
    # (XLA fuses the layout change with downstream ops).
    tracers_t = jnp.moveaxis(tracers, 0, -1)  # (n_lat, n_lon, nlev, n_tr)

    _u_cos_flat = (
        tracers_t * (u_cos * mask_3d)[..., jnp.newaxis]
    ).reshape(n_lat_t, n_lon_t, nlev_t * n_tr)
    _v_cos_flat = (
        tracers_t * (v_cos * mask_3d)[..., jnp.newaxis]
    ).reshape(n_lat_t, n_lon_t, nlev_t * n_tr)
    _u_oc2_flat = sh_analysis_oc2_3d(grid, _u_cos_flat)   # (n_sh, nlev*n_tr)
    _v_dmu_flat = sh_analysis_dmu_3d(grid, _v_cos_flat)
    _flux_flat = (
        im_over_a[:, jnp.newaxis] * _u_oc2_flat
        - one_over_a * _v_dmu_flat
    )
    _flux = _flux_flat.reshape(_flux_flat.shape[0], nlev_t, n_tr)
    tracer_flux_div = jnp.moveaxis(_flux, -1, 0)  # (n_tr, n_sh, nlev)

    # Combine the grid-space tendency contributions before the SH
    # forward transform.  ``sh_analysis_3d`` is linear, so summing
    # tracer_div + tracer_vert_adv (+ tracer_vdiff*mask if K_v > 0) on
    # the grid first folds three separate vmap'd SH-analysis calls
    # into a single trailing-axis-batched call (Loop 94 exploit) and
    # the per-tracer dimension also folds into the same trailing axis
    # (this commit) — saves another ``n_tr - 1 = 1`` SH analysis per
    # RK substage on top of the prior fold.
    tracer_div_t = (
        tracers_t * (div.real * mask_3d)[..., jnp.newaxis]
    )  # (n_lat, n_lon, nlev, n_tr)
    tracer_vert_adv = jax.vmap(
        lambda q: _vertical_advection_spectral(q, w, z_coord, J.real),
        in_axes=-1,
        out_axes=-1,
    )(tracers_t) * mask_3d[..., jnp.newaxis]

    tracer_grid_sum_t = tracer_div_t + tracer_vert_adv
    if config.K_v > 0:
        tracer_vdiff = jax.vmap(
            lambda q: vertical_diffusion(q, z_coord, J.real, config.K_v),
            in_axes=-1,
            out_axes=-1,
        )(tracers_t)
        tracer_grid_sum_t = tracer_grid_sum_t + tracer_vdiff * mask_3d[..., jnp.newaxis]

    _dtr_grid_flat = tracer_grid_sum_t.reshape(
        n_lat_t, n_lon_t, nlev_t * n_tr,
    )
    # Defer the ``_dtr_sh`` analysis so it can be batched with
    # ``deta_dt_grid`` below (Loop 185) — initialize ``dtr_hat`` with
    # only the (already-spectral) flux-divergence contribution; the
    # grid-tendency SH analysis is added after the merged batch.
    dtr_hat = -tracer_flux_div

    # --- 15. Explicit viscosity/diffusion ---
    if config.A_h > 0 or config.K_h > 0:
        lap = grid.lap[:, jnp.newaxis]  # negative semi-definite eigenvalues
        if config.A_h > 0:
            dvor_hat = dvor_hat + config.A_h * lap * state.vor_hat.data
            ddiv_hat = ddiv_hat + config.A_h * lap * state.div_hat.data
        if config.K_h > 0:
            dtr_hat = dtr_hat + config.K_h * lap[jnp.newaxis, ...] * tracers_hat

    # --- 15-16. Velocity vertical diffusion + free-surface tendency ---
    # ``hu_oc2`` and ``hv_dmu`` are now produced by the earlier batched
    # oc2/dmu (Loop 188) — the standalone analyses that used to live
    # in this block are gone.  When ``A_v > 0`` the velocity-vdiff
    # vector still needs its own oc2/dmu pair (different inputs, no
    # cheap fold into the earlier batch since vdiff requires the
    # vertical_diffusion stencil to run first).
    if config.A_v > 0:
        vel_uv = jnp.stack([u.real, v.real], axis=0)
        vdiff_uv = jax.vmap(
            lambda q: vertical_diffusion(q, z_coord, J.real, config.A_v),
            in_axes=0, out_axes=0,
        )(vel_uv)
        vdiff_u = vdiff_uv[0] * mask_3d
        vdiff_v = vdiff_uv[1] * mask_3d
        vdiff_u_cos = vdiff_u * grid.cos_lat[:, jnp.newaxis, jnp.newaxis]
        vdiff_v_cos = vdiff_v * grid.cos_lat[:, jnp.newaxis, jnp.newaxis]

        n_lat_v, n_lon_v, nlev_v = vdiff_u_cos.shape
        _vdh_uv_stack = jnp.stack(
            [vdiff_u_cos, vdiff_v_cos], axis=-1,
        )  # (..., nlev, 2)
        _vdh_uv_flat = _vdh_uv_stack.reshape(n_lat_v, n_lon_v, nlev_v * 2)
        _vdh_oc2 = sh_analysis_oc2_3d(grid, _vdh_uv_flat).reshape(-1, nlev_v, 2)
        _vdh_dmu = sh_analysis_dmu_3d(grid, _vdh_uv_flat).reshape(-1, nlev_v, 2)
        vdiff_u_oc2, vdiff_v_oc2 = _vdh_oc2[..., 0], _vdh_oc2[..., 1]
        vdiff_u_dmu, vdiff_v_dmu = _vdh_dmu[..., 0], _vdh_dmu[..., 1]

        dvor_hat = dvor_hat + (
            im_over_a[:, jnp.newaxis] * vdiff_v_oc2
            + one_over_a * vdiff_u_dmu
        )
        ddiv_hat = ddiv_hat + (
            im_over_a[:, jnp.newaxis] * vdiff_u_oc2
            - one_over_a * vdiff_v_dmu
        )

    # --- 16. Free-surface tendency ---
    # Use flux-form continuity explicitly: dη/dt = -sum_k div(h_k * v_k).
    # This avoids the div(v)*h approximation error on deforming z-star layers.
    # Reuse ``div_hv_pre`` synthesized in section 7 above (same hu/hv
    # spectral coefficients, no need to synthesize twice — iter-78
    # Codex follow-up on the iter-77 flux-form fix).
    deta_dt_grid = -jnp.sum(div_hv_pre, axis=-1) * mask

    # Merge the deferred ``_dtr_grid_flat`` plain analysis with the
    # ``deta_dt_grid`` 2D analysis via ``jnp.concatenate`` along the
    # trailing axis: ``nlev*n_tr + 1`` slots.  ``sh_analysis_3d``
    # treats the trailing axis as a passive batch — same Loop 184
    # exploit as the spectral PE (dT, Phi, dlnps_dt) merge.  2 SH
    # analyses → 1.
    _dtr_eta_input = jnp.concatenate(
        [_dtr_grid_flat, deta_dt_grid[..., jnp.newaxis]], axis=-1,
    )  # (n_lat, n_lon, nlev*n_tr + 1)
    _dtr_eta_hat = sh_analysis_3d(grid, _dtr_eta_input)
    _dtr_sh_flat = _dtr_eta_hat[:, : nlev_t * n_tr]
    deta_hat = _dtr_eta_hat[:, nlev_t * n_tr]
    _dtr_sh = _dtr_sh_flat.reshape(_dtr_sh_flat.shape[0], nlev_t, n_tr)
    dtr_hat = dtr_hat + jnp.moveaxis(_dtr_sh, -1, 0)

    # --- 17. Spectral hyperdiffusion ---
    if config.hyperdiff_coeff > 0:
        # Batch all four pointwise hyperdiffusions (vor, div, T, S)
        # into one call by stacking along a trailing axis.
        # ``spectral_hyperdiffusion_3d`` is purely ``damping * coeffs``,
        # so the trailing axis is a passive batch.  Same exploit as
        # Loops 120/121 for spectral PE/NH; here we collapse 2 + 2
        # vmap'd tracer calls (4 total) into 1.
        n_sh_h, nlev_h = state.vor_hat.data.shape
        # tracers_hat has shape (n_tracers, n_sh, nlev) — move tracer axis
        # to trailing for stacking.
        _T_hat = tracers_hat[0]  # (n_sh, nlev)
        _S_hat = tracers_hat[1]
        _vdts_stack = jnp.stack(
            [state.vor_hat.data, state.div_hat.data, _T_hat, _S_hat], axis=-1,
        )  # (n_sh, nlev, 4)
        _hd_stack = spectral_hyperdiffusion_3d(
            grid, _vdts_stack.reshape(n_sh_h, nlev_h * 4),
            config.hyperdiff_coeff, config.hyperdiff_order,
        ).reshape(n_sh_h, nlev_h, 4)
        dvor_hat = dvor_hat + _hd_stack[..., 0]
        ddiv_hat = ddiv_hat + _hd_stack[..., 1]
        dtr_hat = dtr_hat + jnp.stack(
            [_hd_stack[..., 2], _hd_stack[..., 3]], axis=0,
        )

    # --- 17b. Barotropic (eta) hyperdiffusion ---
    # The spectral solver uses unsplit SSP-RK3 for the entire system,
    # including the fast barotropic gravity-wave mode.  For T21 and
    # typical ocean time steps (dt ~ 3600 s), the barotropic CFL
    # (omega*dt = c*n/a*dt) exceeds the RK3 imaginary-axis stability
    # limit (~1.73) at wavenumbers n > ~12.  Without explicit damping
    # on eta, those modes amplify each step, producing SSH amplitudes
    # ~12x larger than the split-explicit (forward-backward) solvers
    # used on cubed-sphere/lat-lon/MPAS grids.
    #
    # Adding biharmonic hyperdiffusion on eta_hat compensates for the
    # RK3 growth factor.  The damping is scale-selective: negligible at
    # planetary scales (n < 5) and strong at small scales (n > 15),
    # preserving the physical gravity-wave frequency and large-scale
    # amplitude while stabilising the high-wavenumber tail.
    if config.eta_hyperdiff_coeff > 0:
        deta_hat = deta_hat + spectral_hyperdiffusion(
            grid, state.eta_hat.data,
            config.eta_hyperdiff_coeff, config.hyperdiff_order,
        )

    dT_hat = dtr_hat[0]
    dS_hat = dtr_hat[1]

    # Return as same pytree structure
    return SpectralOceanState(
        vor_hat=state.vor_hat.replace(data=dvor_hat),
        div_hat=state.div_hat.replace(data=ddiv_hat),
        T_hat=state.T_hat.replace(data=dT_hat),
        S_hat=state.S_hat.replace(data=dS_hat),
        eta_hat=state.eta_hat.replace(data=deta_hat),
        H_bathy_hat=state.H_bathy_hat.replace(
            data=jnp.zeros_like(state.H_bathy_hat.data),
        ),
        land_mask_grid=state.land_mask_grid.replace(
            data=jnp.zeros_like(state.land_mask_grid.data),
        ),
    )


def _vertical_advection_spectral(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
) -> jnp.ndarray:
    """Vertical advection -w * d(field)/dz with upwind scheme (momentum)."""
    w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
    jac_safe = jnp.maximum(jacobian[..., jnp.newaxis], 1.0e-10)

    dz_half = z_coord.dz_half_ref * jac_safe
    grad = upwind_vertical_gradient(field, dz_half, w_full)
    return -w_full * grad


# ==============================================================================
# Model class
# ==============================================================================

class SpectralOceanModel(IntegrationMixin):
    """Spectral ocean model on the Gaussian grid.

    Uses SSP-RK3 for time integration of the full spectral tendencies.
    Unlike the finite-volume ocean core, this path is currently not
    split-explicit and does not apply barotropic subcycling.

    On Metal (Apple Silicon), SH transforms require complex128 which
    is unsupported on Metal GPU. The model auto-routes computation to
    CPU and transfers results back to Metal, matching the atmospheric
    spectral model pattern.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with SH transform matrices.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    config : SpectralOceanConfig, optional
        Model configuration.
    allow_unsupported_backend : bool
        If True, skip backend checks (for testing).
    """

    def __init__(
        self,
        grid: GaussianGrid,
        z_coord: OceanZStarCoordinate,
        config: SpectralOceanConfig | None = None,
        *,
        allow_unsupported_backend: bool = False,
    ):
        warnings.warn(
            "SpectralOceanModel is unsupported: land boundary handling "
            "in spectral space causes Gibbs ringing and unreliable masking. "
            "Use OceanModel (cubed-sphere), LatLonCGridOceanModel, or "
            "MPASOceanModel instead. See issue #99.",
            FutureWarning,
            stacklevel=2,
        )
        self.z_coord = z_coord
        self.config = config or SpectralOceanConfig()
        self._validate_config(self.config)
        self._use_cpu_for_spectral = False
        self._cpu_device = None
        self._default_device = None
        if self.config.n_barotropic_substeps != 1:
            warnings.warn(
                "SpectralOceanConfig.n_barotropic_substeps is currently ignored "
                "in SpectralOceanModel (unsplit spectral stepping).",
                RuntimeWarning,
                stacklevel=2,
            )

        if not jax.config.jax_enable_x64:
            msg = (
                "SpectralOceanModel requires float64/complex128 arithmetic. "
                "Set JAX_ENABLE_X64=True before importing JAX modules."
            )
            if allow_unsupported_backend:
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            else:
                raise ValueError(msg)

        placement = place_spectral_grid(
            grid, allow_unsupported=allow_unsupported_backend
        )
        self.grid = placement.grid
        self._use_cpu_for_spectral = placement.use_cpu_for_spectral
        self._cpu_device = placement.cpu_device
        self._default_device = placement.default_device

    @staticmethod
    def _validate_config(config: SpectralOceanConfig) -> None:
        """Validate spectral-ocean configuration ranges early."""
        nonnegative = {
            "A_h": config.A_h,
            "K_h": config.K_h,
            "A_v": config.A_v,
            "K_v": config.K_v,
            "hyperdiff_coeff": config.hyperdiff_coeff,
            "eta_hyperdiff_coeff": config.eta_hyperdiff_coeff,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")

        if config.hyperdiff_order < 1:
            raise ValueError(
                "hyperdiff_order must be >= 1, got "
                f"{config.hyperdiff_order!r}",
            )
        if config.n_barotropic_substeps < 1:
            raise ValueError(
                "n_barotropic_substeps must be >= 1, got "
                f"{config.n_barotropic_substeps!r}",
            )
        if config.min_water_column_m <= 0.0:
            raise ValueError(
                "min_water_column_m must be > 0, got "
                f"{config.min_water_column_m!r}",
            )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: SpectralOceanState, dt: float,
    ) -> SpectralOceanState:
        """Advance one time step using SSP-RK3."""
        def tendency_fn(s):
            return spectral_ocean_tendencies(
                s, self.grid, self.z_coord, self.config,
            )

        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result_cpu = dispatch_integrator(state_cpu, tendency_fn, dt, self.config.time_integrator)
            if self.config.use_conservation_fixer:
                result_cpu = _spectral_conservation_fixer(
                    result_cpu, state_cpu, self.grid, self.z_coord, self.config,
                )
            return jax.device_put(result_cpu, self._default_device)

        result = dispatch_integrator(state, tendency_fn, dt, self.config.time_integrator)
        if self.config.use_conservation_fixer:
            result = _spectral_conservation_fixer(
                result, state, self.grid, self.z_coord, self.config,
            )
        return result

    @partial(jax.jit, static_argnums=(0,))
    def _step_on_cpu(
        self, state: SpectralOceanState, dt: float,
    ) -> SpectralOceanState:
        """Step without device transfers (for batched CPU integration on Metal)."""
        def tendency_fn(s):
            return spectral_ocean_tendencies(
                s, self.grid, self.z_coord, self.config,
            )
        result = dispatch_integrator(state, tendency_fn, dt, self.config.time_integrator)
        if self.config.use_conservation_fixer:
            result = _spectral_conservation_fixer(
                result, state, self.grid, self.z_coord, self.config,
            )
        return result

    def integrate(
        self,
        state: SpectralOceanState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[SpectralOceanState, list[SpectralOceanState]]:
        """Integrate forward for a given duration.

        On Metal, batches CPU transfers: transfer state to CPU once,
        run all steps on CPU, then transfer results back to Metal.
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
                "integration has zero steps; increase duration or reduce dt "
                f"(duration={duration!r}, dt={dt!r})",
            )

        if self._use_cpu_for_spectral:
            return self._integrate_on_cpu(state, n_steps, dt, save_every)

        trajectory = [state]
        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory

    def _integrate_on_cpu(self, state, n_steps, dt, save_every):
        """Batch integration on CPU: transfer once, not per step."""
        state_cpu = jax.device_put(state, self._cpu_device)
        trajectory_cpu = [state_cpu]

        for i in range(n_steps):
            state_cpu = self._step_on_cpu(state_cpu, dt)
            if (i + 1) % save_every == 0:
                trajectory_cpu.append(state_cpu)

        state_out = jax.device_put(state_cpu, self._default_device)
        trajectory_out = [
            jax.device_put(s, self._default_device) for s in trajectory_cpu
        ]
        return state_out, trajectory_out


# ==============================================================================
# Initialization
# ==============================================================================

def rest_state_spectral_ocean(
    grid: GaussianGrid,
    z_coord: OceanZStarCoordinate,
    T_water_init_C: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
) -> SpectralOceanState:
    """Create a rest-state spectral ocean initial condition.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    T_water_init_C, T_deep : float
        Surface and deep temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].
    H_max : float
        Maximum ocean depth [m].
    land_lat_threshold : float
        Latitude threshold for land mask [degrees].
    """
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if land_lat_threshold < 0.0 or land_lat_threshold > 90.0:
        raise ValueError(
            "land_lat_threshold must be in [0, 90] degrees, "
            f"got {land_lat_threshold!r}",
        )
    for name, value in {
        "T_water_init_C": T_water_init_C,
        "T_deep": T_deep,
        "S_uniform": S_uniform,
    }.items():
        if not bool(jnp.isfinite(jnp.asarray(value))):
            raise ValueError(f"{name} must be finite, got {value!r}")

    nlev = z_coord.n_levels
    n_sh = grid.n_sh

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)

    # Land mask in grid space — use a smooth tanh transition to avoid
    # Gibbs ringing at the land-ocean boundary in spectral space.
    lat_deg = jnp.abs(grid.lat2d) * (180.0 / jnp.pi)
    if land_lat_threshold >= 90.0:
        # No land: global ocean with uniform depth.
        mask = jnp.ones((grid.n_lat, grid.n_lon))
        H_bathy_grid = jnp.full((grid.n_lat, grid.n_lon), H_max)
    else:
        taper_width = 5.0  # degrees
        mask = 0.5 * (1.0 - jnp.tanh((lat_deg - land_lat_threshold) / taper_width))
        H_bathy_grid = 1.0 + (H_max - 1.0) * mask

    # Temperature profile (exponential stratification).  ``scale_depth``
    # comes from ``legoesm.ocean.eos`` (the canonical 1000 m e-folding
    # depth for ocean stratification) rather than a local literal —
    # iter-66 audit fix.
    T_profile = T_deep + (T_water_init_C - T_deep) * jnp.exp(z_coord.z_full_ref / scale_depth)
    T_grid = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, :],
        (grid.n_lat, grid.n_lon, nlev),
    )

    # Salinity
    S_grid = jnp.full(
        (grid.n_lat, grid.n_lon, nlev), S_uniform,
    )

    # Transform to spectral
    vor_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    div_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    T_hat = sh_analysis_3d(grid, T_grid.astype(jnp.float64))
    S_hat = sh_analysis_3d(grid, S_grid.astype(jnp.float64))
    eta_hat = jnp.zeros(n_sh, dtype=jnp.complex128)
    H_bathy_hat = sh_analysis(grid, H_bathy_grid.astype(jnp.float64))

    return SpectralOceanState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="degC"),
        S_hat=Field(data=S_hat, name="S_hat", dims=dims_3d, units="PSU"),
        eta_hat=Field(data=eta_hat, name="eta_hat", dims=dims_2d, units="m"),
        H_bathy_hat=Field(data=H_bathy_hat, name="H_bathy_hat", dims=dims_2d, units="m"),
        land_mask_grid=Field(data=mask, name="land_mask", dims=("lat", "lon"), units=""),
    )
