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

The dycore stack is the eddy-resolving stack (split-explicit barotropic free
surface, WENO7 tracer, RK3 + AB2, smc03 PGF, linear EOS, vertical background ν/κ,
no GM/KPP); the momentum scheme is selected by ``apply_silvestri_scheme`` (UP3/
W9V/W9D/SM2/QG2). Resolutions 1/8°,1/16°,1/32° (Ny=20/res), L_d≈5.5→6.75 km.

Turbulent-stage fix (FAITHFUL, NO dissipation backstop — barotropic_diffusion_alpha=0).
The eddy-resolving jet blew up at the turbulent stage (~day 72) from the C-grid
**barotropic Coriolis 2Δx rotational null mode** (docs/issues/barotropic_mode_noise.md
§A), diagnosed by a validated state-bridge against the Oceananigans oracle: the
in-substep 4-point ``f·V_at_u`` annihilates the 2Δx-in-lon barotropic mode, leaving
it unconstrained to grow under the eddy field.  Cure = the Oceananigans split-explicit
architecture (NO in-substep barotropic Coriolis):
  * ``coriolis_scheme="explicit_ab2"`` routes the planetary f×u to the barotropic mode
    through ``F_slow`` (depth-mean of du_dt) and gates the in-substep Coriolis OFF —
    no 2Δx null mode;
  * ``barotropic_slow_forcing_ab2=True`` AB2 time-centers ``F_slow`` to match
    Oceananigans' AB2-extrapolated ``Gᵁ`` — this keeps the barotropic eta–U geostrophic
    balance (without it the SSH drifts and blows ~day 38).

SCOPE / KNOWN LIMITATION (honest):  This Coriolis fix removes the EARLY barotropic
failures (the d38 SSH drift and the d54 2Δx null mode).  W9V then tracks the oracle to
~day 50 and survives the 80-day target at ``barotropic_diffusion_alpha=0`` (NO
SSH-diffusion / viscosity backstop) — validated to 80 d by
``scripts/tmp/_silvestri_eps_validate.py``.  BUT a SEPARATE, filter-independent
eddy-scale instability remains: once the eddy field reaches the oracle's saturation
amplitude (~0.10 m/s) around day 55, legoESM fails to equilibrate and runs away to
blow-up at ~day 90 (cosine) / ~day 104 (power_law) — both filters diverge at day 55, so
this is NOT a barotropic-Coriolis or time-filter issue.  The oracle saturates at
~0.10 m/s and runs indefinitely; legoESM reaches ~0.7 m/s by day 80 (7× over-energetic,
mid-runaway).  This residual is the documented WENO-momentum / eddy-equilibration
problem (interior 2Δx under-dissipation / inverse-cascade arrest), a separate effort.
The no-closure WENO/UP3 schemes are correctly more energetic than the closure SM2/QG2
through the tracked window.  For a stable long run, use a small ``barotropic_diffusion_
alpha`` (oracle amplitude at ~0.005) until the eddy-equilibration fix lands.
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
    lon_east: float = 10.0           # 20° periodic (FIDELITY NOTE below)
    # FIDELITY CAVEAT vs the Oceananigans oracle: the oracle is a 16°-zonal
    # CARTESIAN β-plane (Lx = deg2rad(16)·R·cos50 ≈ 1145 km, 128 pts ⇒ 1/8°);
    # this experiment is a 20°-zonal SPHERICAL lat-lon periodic channel (~0.156°
    # zonal at 128 pts, cos(lat)-varying width).  So the cured §5 run TRACKS A
    # BOUNDED TURBULENT TRAJECTORY in the same physical regime — it is NOT a
    # bit-for-bit oracle match (different grid geometry + 25% wider, coarser
    # zonal channel).  Matching the oracle's 16° β-plane is a separate fidelity
    # task; the faithful barotropic-Coriolis cure (survives 80 d at α=0) is
    # validated here on the spherical channel.
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
    # GH #480: the day-11 eddy-permitting blow-up is now cured at the ROOT by
    # the faithful tracer-wall Neumann fill (LatLonCGridOceanConfig.
    # tracer_wall_neumann_fill, default ON) — the masked-land cold cell (T=0)
    # was contaminating the wide WENO tracer stencil at the free-slip walls and
    # manufacturing a spurious near-wall front (matched to the Oceananigans
    # oracle: v→buoyancy wall gain 68×→1.2×).  W9V/W9D now survive §5 with NO
    # closure and NO wall filter.  The wall Shapiro filter below is retained as
    # an optional belt-and-braces knob but defaults OFF (0.0).
    wall_grid_filter_rate_s: float = 0.0

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
        # FAITHFUL barotropic stack = the Oceananigans split-explicit oracle's,
        # with NO dissipation backstop (barotropic_diffusion_alpha=0). The
        # eddy-resolving turbulent blow-up was the C-grid barotropic Coriolis
        # 2Δx rotational null mode (docs/issues/barotropic_mode_noise.md §A):
        # the in-substep 4-point f·V_at_u annihilates the 2Δx-in-lon mode →
        # unconstrained → grows under the eddies → blows ~day 72. Cure (matches
        # Oceananigans, which has NO in-substep barotropic Coriolis):
        #   * coriolis_scheme="explicit_ab2" — route the planetary f×u to the
        #     barotropic mode via F_slow (depth-mean of du_dt), gate the
        #     in-substep f·V_at_u OFF → no null mode;
        #   * barotropic_slow_forcing_ab2=True — AB2 time-center F_slow to match
        #     Oceananigans' AB2-extrapolated Gᵁ → keeps the barotropic eta-U
        #     geostrophic balance (without it the SSH drifts → blows ~day 38).
        # This removes the EARLY barotropic failures. The remaining ~day-89 blow-up
        # was NOT an eddy-equilibration residual (as long believed) — it was TWO
        # faithful gaps, both now closed (2026-06-23, see the scoreboard): (1) the
        # vertical momentum advection advected only u' (omitting −∂(w·U_bar)/∂z) —
        # fixed by weno_vertadv_full_velocity=True below; (2) the TIMESTEP — the oracle
        # runs an adaptive wizard (cfl=0.3, Δt 300–900 s) that shrinks at the transient
        # peak, while a FIXED dt=900 (the oracle's max) CFL-violated at the
        # over-energized peak → NaN day 89. With full-velocity vertadv + the oracle's
        # adaptive dt (or a fixed dt ≤ ~450 s within its range), §5 W9V survives the
        # FULL 200-d transient at alpha=0 with NO backstop, max|u| ~1 m/s (the oracle's
        # amplitude). The A_h+Smag `stabilize` backstop below is NO LONGER NEEDED.
        barotropic_solver="explicit_substep",
        coriolis_scheme="explicit_ab2",
        barotropic_slow_forcing_ab2=True,
        barotropic_diffusion_alpha=0.0,      # NO SSH-diffusion backstop (faithful)
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
        # GH #480: suppress the un-dissipatable 2dx-in-lon grid mode that grows at
        # the N/S free-slip walls (WENO vector-invariant momentum). Boundary-localised
        # to the wall rows only (no-op for the non-WENO schemes UP3/SM2/QG2). Lets the
        # eddy-permitting W9V/W9D jet run with NO closure (vs the A_h+Smag `stabilize`
        # backstop below). Validated: §5 W9V survives 16 d at the Oceananigans amplitude.
        wall_grid_filter_rate_s=config.wall_grid_filter_rate_s,
        # FAITHFUL §5 closure (2026-06-23): advect the FULL horizontal momentum
        # vertically (Oceananigans w·∂u/∂z over full u), not legoESM's default
        # baroclinic perturbation u' which omits −∂(w·U_bar)/∂z. The perturbation form
        # let the interior baroclinic eddies run away; the full-velocity form lets §5
        # W9V survive the full transient (200 d, no backstop) with the oracle's adaptive
        # timestep (cfl=0.3, max_Δt=900 s). See docs/ocean_fidelity/oceananigans_reproduction_scoreboard.md.
        weno_vertadv_full_velocity=True,
    )
    model_config = apply_silvestri_scheme(base_config, scheme)

    # Eddy-resolving stabilization backstop — NO LONGER NEEDED for a faithful §5 run
    # (2026-06-23): the blow-up it patched was the vertical-advection + fixed-dt CFL
    # gap, now closed (full-velocity vertadv + the oracle's adaptive dt). Kept ONLY as
    # an opt-in (`stabilize=True`, default False) for non-faithful robustness probes;
    # the faithful paper run (stabilize=False) survives the full transient with NO
    # closure. A_h=1000 + C_smag=0.1 + smag_cfl_safety=0.5 is the old Eady backstop;
    # it trips the double-friction guard against SM2/QG2 so is WENO/flux-only.
    if stabilize and model_config.lateral_friction_scheme == "none":
        model_config = model_config._replace(
            A_h=1000.0, C_smag=0.1, smag_cfl_safety=0.5)

    initial_state, wall_mask = _build_initial_state(grid, z_coord, config)

    # Seed the AB2 barotropic-slow-forcing prev fields (zeros) so the lax.scan
    # carry pytree is stable from step 1 (the model step stores a Field each
    # step when barotropic_slow_forcing_ab2 is on; with prev=zeros the first
    # step is (3/2+ε)·F_slow ≈ 1.6× — the SAME AB2 cold-start convention as the
    # outer baroclinic integrator, transient and negligible on the near-balanced
    # IC).  Required: any driver using barotropic_slow_forcing_ab2 under
    # lax.scan MUST seed these (else a step-1 None→Field transition breaks the
    # scan carry); build_silvestri does it here.  No-op for the default stack.
    if getattr(model_config, "barotropic_slow_forcing_ab2", False):
        from legoesm.core.field import Field as _Field_fs0
        _z_u = jnp.zeros_like(initial_state.u.data[:, :, 0])
        _z_v = jnp.zeros_like(initial_state.v.data[:, :, 0])
        initial_state = initial_state._replace(
            F_slow_u_prev=_Field_fs0(data=_z_u, name="F_slow_u_prev",
                                     dims=("lat", "lon_u"), units="m/s^2"),
            F_slow_v_prev=_Field_fs0(data=_z_v, name="F_slow_v_prev",
                                     dims=("lat_v", "lon"), units="m/s^2"))

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
