"""Silvestri et al. 2024 §5 baroclinic-jet experiment (the QG2/SM2/UP3/W9V/W9D matrix).

A periodic channel on a spherical sector (60°S–40°S, 20° wide, 1 km deep, 50
levels) with a meridional buoyancy FRONT in thermal-wind balance, restored to
its initial ZONAL-MEAN state on a 50-day timescale so the jet equilibrates
without damping the mesoscale eddies (Soufflet et al. 2016). This is distinct
from the localized-jet ``eady_uniform`` experiment: a full-channel front (paper
Eqs 52-53), not a Gaussian-enveloped jet, and zonal-mean restoring, not a wall
sponge.

Initial buoyancy (Eqs 52-53):
    b(φ,z) = N²·z + Δb·B(γ(φ)),
    B(γ) = 0 (γ<0); (γ − sinγ·cosγ)/π (0≤γ≤π); 1 (γ>π),
    γ(φ) = π/2 − 2π·(φ−φ₀)/Δφ,   φ₀=−50°, Δφ=20°, Δb=5e-3 m/s² (≈2.5°C), N²=4e-6.
Represented in the model's linear-EOS temperature: b = g·α_T·(T−T_ref) ⇒
    T(φ,z) = T_ref + (N²·z + Δb·B(φ)) / (g·α_T).
Initial velocity: thermal wind  f·∂u/∂z = −∂b/∂y, u(z=−H)=0 ⇒ u depth-linear.

The dycore stack is the corrected eddy-resolving stack (implicit-CN barotropic,
WENO7 tracer, RK3 + AB2, smc03 PGF, linear EOS, vertical background ν/κ, no GM/
KPP); the momentum scheme is selected by ``apply_silvestri_scheme`` (UP3/W9V/
W9D/SM2/QG2). Resolutions 1/8°,1/16°,1/32° (Ny=20/res), L_d≈5.5→6.75 km.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field


# ---------------------------------------------------------------------------
# Config (paper §5 values)
# ---------------------------------------------------------------------------

class SilvestriJetConfig(NamedTuple):
    lat_south: float = -60.0
    lat_north: float = -40.0
    lat_center: float = -50.0        # φ₀
    lon_west: float = -10.0
    lon_east: float = 10.0           # 20° periodic
    H_max: float = 1000.0            # 1 km deep
    N2: float = 4.0e-6               # background stratification [1/s²]
    delta_b: float = 5.0e-3          # front buoyancy jump [m/s²] (≈2.5°C)
    # Linear EOS: b = g·α_T·(T−T_ref). α_T chosen so Δb ↔ ΔT≈2.5°C.
    alpha_T: float = 2.04e-4
    T_ref_C: float = 10.0
    S_uniform: float = 35.0
    rho_0: float = constants.rho_ocean
    # Vertical background mixing (paper: ν=1e-4, κ=1e-5).
    A_v: float = 1.0e-4
    K_v: float = 1.0e-5
    # Zonal-mean restoring (Soufflet 2016 / paper): τ = 50 days.
    restoring_timescale_days: float = 50.0
    # White-noise kick on T to seed the instability [K].
    noise_amplitude_K: float = 1.0e-3
    noise_seed: int = 0

    @property
    def Delta_phi_rad(self) -> float:
        return np.radians(self.lat_north - self.lat_south)

    @property
    def restoring_gamma(self) -> float:
        return 1.0 / (self.restoring_timescale_days * 86400.0)


def silvestri_front_B(lat_rad: np.ndarray, config: SilvestriJetConfig) -> np.ndarray:
    """The meridional front profile B(φ) ∈ [0,1] (Eqs 52-53). 1 at the south
    edge, 0 at the north edge, 1/2 at the centre."""
    phi0 = np.radians(config.lat_center)
    gamma = (np.pi / 2.0) - 2.0 * np.pi * (lat_rad - phi0) / config.Delta_phi_rad
    B_mid = (gamma - np.sin(gamma) * np.cos(gamma)) / np.pi
    return np.where(gamma < 0.0, 0.0, np.where(gamma > np.pi, 1.0, B_mid))


# ---------------------------------------------------------------------------
# Recipe + restoring
# ---------------------------------------------------------------------------

class SilvestriJetRecipe(NamedTuple):
    model_config: object
    physics_config: object
    grid: object
    z_coord: object
    wall_mask: object
    initial_state: object
    restoring: object        # SilvestriRestoring (targets + gamma)


class SilvestriRestoring(NamedTuple):
    gamma: float                 # relaxation rate [1/s]
    T_ref_zm: jnp.ndarray        # (n_lat, nlev) initial zonal-mean T target
    S_ref_zm: jnp.ndarray        # (n_lat, nlev)
    u_ref_zm: jnp.ndarray        # (n_lat, n_lon+1?, nlev) — stored at u-points
    v_ref_zm: jnp.ndarray        # (n_lat+1, n_lon, nlev)


def apply_zonal_mean_restoring(field: jnp.ndarray, ref_zm: jnp.ndarray,
                               gamma: float, dt: float,
                               wrap: bool = False) -> jnp.ndarray:
    """Relax the ZONAL-MEAN component of ``field`` toward ``ref_zm`` (the initial
    zonal-mean profile), leaving the eddy (zero-zonal-mean) part untouched.

    tendency = −γ·(⟨field⟩_x − ref_zm), applied uniformly in longitude. Because
    the correction is constant in x, the eddy part is unchanged — this restores
    the mean jet/transport without damping the mesoscale (Soufflet et al. 2016).
    Explicit (γ·dt ≪ 1 for τ=50 d, dt~minutes).

    ``wrap=True`` for the u-field, whose last column (n_lon) is the periodic
    duplicate of column 0: the zonal mean is taken over the DISTINCT columns
    ``[:-1]`` (else col 0 is double-weighted), but the correction still applies
    to all columns so the wrap stays consistent.
    """
    f_for_mean = field[:, :-1, ...] if wrap else field
    zm = jnp.mean(f_for_mean, axis=1, keepdims=True)         # (n_lat, 1[, nlev])
    # ref_zm is (n_lat[, nlev]); add the lon axis to broadcast.
    ref_b = ref_zm[:, None, ...]
    return field - gamma * dt * (zm - ref_b)


def restore_state(state, restoring: SilvestriRestoring, dt: float):
    """Apply the zonal-mean restoring to T, S, u, v of a state (driver helper)."""
    g = restoring.gamma
    return state._replace(
        T=state.T.replace(data=apply_zonal_mean_restoring(
            state.T.data, restoring.T_ref_zm, g, dt)),
        S=state.S.replace(data=apply_zonal_mean_restoring(
            state.S.data, restoring.S_ref_zm, g, dt)),
        u=state.u.replace(data=apply_zonal_mean_restoring(
            state.u.data, restoring.u_ref_zm, g, dt, wrap=True)),
        v=state.v.replace(data=apply_zonal_mean_restoring(
            state.v.data, restoring.v_ref_zm, g, dt)),
    )


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def _build_initial_state(grid, z_coord, config: SilvestriJetConfig):
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean

    n_lat, n_lon = grid.n_lat, grid.n_lon
    wall_mask = np.ones((n_lat, n_lon), dtype=np.float32)
    wall_mask[0, :] = 0.0
    wall_mask[-1, :] = 0.0
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=config.H_max,
        T_water_init_C=config.T_ref_C, T_deep=config.T_ref_C,
        S_uniform=config.S_uniform, land_mask_override=wall_mask)

    z_full = np.asarray(z_coord.z_full_ref)                  # (nlev,) ≤ 0
    nlev = len(z_full)
    lat_rad = np.asarray(grid.lat)                           # (n_lat,)
    R = constants.R_earth
    g = constants.g

    # --- Temperature from the buoyancy front: T = T_ref + b/(g α_T) ---
    B = silvestri_front_B(lat_rad, config)                  # (n_lat,)
    mask = np.asarray(state.land_mask.data)                 # (n_lat, n_lon)
    T = np.empty((n_lat, n_lon, nlev), dtype=np.float64)
    for k in range(nlev):
        b_col = config.N2 * z_full[k] + config.delta_b * B  # (n_lat,)
        T[:, :, k] = (config.T_ref_C + b_col[:, None] / (g * config.alpha_T)) * mask

    # White-noise kick (zero at walls).
    rng = np.random.default_rng(config.noise_seed)
    noise = config.noise_amplitude_K * rng.standard_normal((n_lat, n_lon, nlev))
    T = T + noise * mask[:, :, None]

    # --- Thermal-wind velocity: f ∂u/∂z = −∂b/∂y, u(−H)=0 ---
    y = R * lat_rad                                          # (n_lat,)
    dBdy = np.gradient(B, y)                                 # (n_lat,)
    dbdy = config.delta_b * dBdy                             # ∂b/∂y (z-independent)
    f_lat = 2.0 * constants.Omega * np.sin(lat_rad)         # (n_lat,)
    f_safe = np.where(np.abs(f_lat) < 1e-12, np.sign(f_lat + 1e-30) * 1e-12, f_lat)
    H = config.H_max
    # u depth-linear, zero at the bottom FACE z=−H (z_full is cell-centred, so
    # the deepest cell carries a small residual u — vanishing is at the face).
    u = np.zeros((n_lat, n_lon + 1, nlev), dtype=np.float64)
    for k in range(nlev):
        u_col = -(1.0 / f_safe) * dbdy * (z_full[k] + H)     # (n_lat,) [m/s]
        u[:, :, k] = u_col[:, None]
    # Zero u on the N/S wall rows (the lon-roll combines the two lon-adjacent
    # cell masks; for N/S walls the zeroing comes from the zero wall ROWS).
    u_face_mask = np.minimum(mask, np.roll(mask, 1, axis=1))
    u_face_mask = np.concatenate([u_face_mask, u_face_mask[:, :1]], axis=1)
    u = u * u_face_mask[:, :, None]

    state = state._replace(
        T=state.T.replace(data=jnp.asarray(T)),
        u=state.u.replace(data=jnp.asarray(u)),
    )
    return state, wall_mask


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_silvestri_baroclinic_jet_setup(
        *, n_lat: int, n_lon: int, scheme: str = "W9V", nlev: int = 50,
        config: SilvestriJetConfig = None,
        stabilize: bool = False) -> SilvestriJetRecipe:
    """Assemble the paper §5 baroclinic-jet model for a given momentum ``scheme``
    (UP3/W9V/W9D/SM2/QG2). ``n_lat`` ≈ 20/resolution (1/8°→160, 1/16°→320,
    1/32°→640 over the 20° band)."""
    from legoesm.grids.latlon import create_regional_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.eos import LinearEOSConfig
    from legoesm.ocean.experiments.silvestri_schemes import apply_silvestri_scheme

    if config is None:
        config = SilvestriJetConfig()

    grid, wall_mask = create_regional_latlon_grid(
        n_lat, n_lon, config.lat_south, config.lat_north,
        lon_west=config.lon_west, lon_east=config.lon_east, periodic_x=True)
    # Uniform vertical spacing dz = H_max/nlev (paper §5: fixed dz=20 m at 50
    # levels over 1 km). dz_surface == dz_deep makes create_ocean_z_star uniform.
    _dz = config.H_max / nlev
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=config.H_max,
                                  dz_surface=_dz, dz_deep=_dz)

    base_config = LatLonCGridOceanConfig(
        barotropic_solver="implicit_cn",
        tracer_advection="weno7",            # paper: 7th-order WENO tracer (all cases)
        tracer_time_integrator="rk3",
        outer_integrator="ab2",
        pgf_scheme="smc03",
        A_v=config.A_v, K_v=config.K_v,
        eos="linear",
        eos_linear=LinearEOSConfig(
            alpha_T=config.alpha_T, rho_ref=config.rho_0,
            T_ref=config.T_ref_C, S_ref=config.S_uniform),
        gm_redi=None,
    )
    model_config = apply_silvestri_scheme(base_config, scheme)

    # Eddy-resolving stabilization backstop (B5-stab finding): legoESM's WENO
    # vector-invariant under-dissipates the C-grid grid-scale mode vs Oceananigans
    # and blows up at eddy-resolving resolution WITHOUT an explicit closure. The
    # Eady min-dissipation combination A_h=1000 + C_smag=0.1 + smag_cfl_safety=0.5
    # stabilizes it while keeping the eddies. Applied ONLY to the no-closure WENO/
    # flux schemes (lateral_friction_scheme=="none"); SM2/QG2 keep their own
    # closure (adding A_h/C_smag would trip the double-friction guard). NOT
    # paper-faithful (the paper's WENO has no closure) — a documented legoESM cost.
    if stabilize and model_config.lateral_friction_scheme == "none":
        model_config = model_config._replace(
            A_h=1000.0, C_smag=0.1, smag_cfl_safety=0.5)

    initial_state, wall_mask = _build_initial_state(grid, z_coord, config)

    # Restoring targets = the (zonally-uniform) initial zonal-mean profiles.
    restoring = SilvestriRestoring(
        gamma=config.restoring_gamma,
        T_ref_zm=jnp.mean(initial_state.T.data, axis=1),
        S_ref_zm=jnp.mean(initial_state.S.data, axis=1),
        # u has a periodic wrap column (n_lon) duplicating col 0 — average over
        # the distinct columns [:-1] (consistent with apply_..._restoring wrap=True).
        u_ref_zm=jnp.mean(initial_state.u.data[:, :-1], axis=1),
        v_ref_zm=jnp.mean(initial_state.v.data, axis=1),
    )
    return SilvestriJetRecipe(
        model_config=model_config, physics_config=None, grid=grid,
        z_coord=z_coord, wall_mask=wall_mask, initial_state=initial_state,
        restoring=restoring)
