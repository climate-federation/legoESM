"""Model integration bridge for turbulence.

Provides `make_turbulence_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

Key difference from radiation/convection: turbulence produces momentum
tendencies (du_dt, dv_dt) that are nonzero. For the spectral PE dycore,
these must be projected to spectral vorticity/divergence tendencies.

Supported model types:
- "hydrostatic"  : PrimitiveEquationModel (sigma coordinates)
- "nonhydrostatic": CompressibleEulerModel (z* coordinates)
- "spectral_pe"  : SpectralPEModel (Gaussian grid + sigma coordinates)
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticState,
    HydrostaticTendencies,
    NonHydrostaticState,
    NonHydrostaticTendencies,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import (
    HeightCoordinate,
    SigmaCoordinate,
    TerrainMetric,
    pressure_from_sigma,
)
from legoesm import constants

from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.grids.gaussian import (
    sh_analysis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
)
from legoesm.atmosphere.physics.turbulence.smagorinsky import smagorinsky_turbulence
from legoesm.atmosphere.physics.turbulence.louis import louis_turbulence
from legoesm.atmosphere.physics.turbulence.tke import tke_turbulence
from legoesm.atmosphere.physics.turbulence.clubb_lite import clubb_lite_turbulence
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
    holtslag_boville_turbulence,
)
from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence
from legoesm.atmosphere.physics.turbulence.edmf import edmf_turbulence
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)


def _get_turbulence_fn(config: TurbulenceConfig):
    """Select the turbulence backend based on config.scheme."""
    if config.scheme == "smagorinsky":
        return "smagorinsky", smagorinsky_turbulence, config.smagorinsky
    elif config.scheme == "louis":
        return "louis", louis_turbulence, config.louis
    elif config.scheme == "tke":
        return "tke", tke_turbulence, config.tke
    elif config.scheme == "clubb_lite":
        return "clubb_lite", clubb_lite_turbulence, config.clubb_lite
    elif config.scheme == "holtslag_boville":
        return "holtslag_boville", holtslag_boville_turbulence, config.holtslag_boville
    elif config.scheme == "ysu":
        return "ysu", ysu_turbulence, config.ysu
    elif config.scheme == "edmf":
        return "edmf", edmf_turbulence, config.edmf
    elif config.scheme == "none":
        return "none", None, None
    else:
        raise ValueError(f"Unknown turbulence scheme: {config.scheme!r}")


from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma as _compute_heights_from_sigma,
    compute_rho as _compute_rho,
)
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.grids.gaussian import (
    sh_analysis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
)


def make_turbulence_physics(
    turbulence_config: TurbulenceConfig,
    model_type: str = "hydrostatic",
    dt: float = 300.0,
) -> Callable:
    """Create a physics function for turbulence matching a model's signature.

    Parameters
    ----------
    turbulence_config : TurbulenceConfig
        Turbulence configuration (selects scheme).
    model_type : str
        One of "hydrostatic", "nonhydrostatic", "spectral_pe".
    dt : float
        Model time step [s].

    Returns
    -------
    Callable
        Physics function with the correct signature for the model.
    """
    if model_type == "hydrostatic":
        return _make_hydrostatic_turbulence(turbulence_config, dt)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_turbulence(turbulence_config, dt)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_turbulence(turbulence_config, dt)
    elif model_type == "mpas":
        # MPAS dispatch is intentionally NOT supported: the turbulence
        # bridges read both ``state.u`` and ``state.v`` directly, but
        # MPAS stores only the normal velocity ``state.u`` on edges
        # and has ``state.v is None``.  Until an edge→cell wind
        # interpolator is added, fail fast with a clear message rather
        # than a confusing ``AttributeError`` deep inside the bridge.
        raise NotImplementedError(
            "Turbulence on MPAS Voronoi mesh is not yet supported.  "
            "The bridge expects cell-centered u and v winds but MPAS "
            "stores only the normal velocity on edges.  Edge→cell "
            "interpolation is a follow-up; until then, run MPAS with "
            "turbulence='none'."
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe', 'mpas'."
        )


# ===========================================================================
# Hydrostatic PE
# ===========================================================================

def _make_hydrostatic_turbulence(
    turbulence_config: TurbulenceConfig,
    dt: float,
) -> Callable:
    """Create turbulence physics_fn for PrimitiveEquationModel.

    Signature: (state, grid, sigma_coord, phys_state=None) -> HydrostaticTendencies

    When *phys_state* (a ``PhysicsState``) is passed, TKE is read from
    ``phys_state.tke`` and the updated TKE is returned as the second
    element of the result tuple.

    Turbulence produces nonzero du_dt, dv_dt (unlike convection/radiation).
    When tracers are available, q_v is read from ``state.tracers["q_v"]``
    and the moisture tendency ``dq_v_dt`` is returned via ``tracer_tendencies``.
    """
    scheme_name, turb_fn, scheme_config = _get_turbulence_fn(turbulence_config)
    needs_tke = scheme_name in ("tke", "clubb_lite", "edmf")

    def physics_fn(
        state: HydrostaticState,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
        phys_state=None,
    ):
        tke_out = None
        T = state.T.data          # (6, n, n, nlev)
        u = state.u.data
        v = state.v.data
        p_s = state.p_s.data      # (6, n, n)

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape

        # Pressure
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)

        # Reshape to columns
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        u_col = u.reshape(ncol, nlev)
        v_col = v.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        # Extract water vapor from tracers if available; else assume dry.
        # Pin all defaulted allocations to the state precision so we never
        # silently promote an x64 zero into a float32 column path (which
        # poisons downstream scan carries with mixed-precision dtypes).
        _state_dtype = T.dtype
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(ncol, nlev)
        else:
            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")

        if turb_fn is None:
            return HydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_turb", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_turb", dims=dims_3d, units="m/s^2"),
                dT_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dT_dt_turb", dims=dims_3d, units="K/s"),
                dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dp_s_dt_turb", dims=dims_2d, units="Pa/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dphis_dt_turb", dims=dims_2d, units="m^2/s^3"),
            ), tke_out

        # Heights and density
        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col)
        rho = _compute_rho(T_col, p_full_col)

        # Surface conditions
        T_sfc = T_col[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])

        if needs_tke:
            # Read TKE from explicit PhysicsState if provided.
            if phys_state is not None:
                tke_in = phys_state.tke
                # Reshape if needed (PhysicsState stores flat columns).
                if tke_in.shape != (ncol, nlev):
                    tke_in = jnp.full((ncol, nlev), scheme_config.tke_min, dtype=_state_dtype)
            else:
                tke_in = jnp.full((ncol, nlev), scheme_config.tke_min, dtype=_state_dtype)

            turb_out, tke_new = turb_fn(
                u_col, v_col, T_col, q_v_col, tke_in,
                p_full_col, p_half_col, z_full, z_half,
                T_sfc, q_sfc, rho, dt, scheme_config,
            )
            tke_out = tke_new
        else:
            turb_out = turb_fn(
                u_col, v_col, T_col, q_v_col,
                p_full_col, p_half_col, z_full, z_half,
                T_sfc, q_sfc, rho, dt, scheme_config,
            )

        du_dt = turb_out.du_dt.reshape(shape_3d)
        dv_dt = turb_out.dv_dt.reshape(shape_3d)
        dT_dt = turb_out.dT_dt.reshape(shape_3d)

        # Propagate moisture tendency from turbulence backend
        tracer_tends = {
            "q_v": Field(
                data=turb_out.dq_v_dt.reshape(shape_3d),
                name="dq_v_dt_turb",
                dims=dims_3d, units="kg/kg/s",
            ),
        }

        tendencies = HydrostaticTendencies(
            du_dt=Field(data=du_dt, name="du_dt_turb", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt, name="dv_dt_turb", dims=dims_3d, units="m/s^2"),
            dT_dt=Field(data=dT_dt, name="dT_dt_turb", dims=dims_3d, units="K/s"),
            dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dp_s_dt_turb", dims=dims_2d, units="Pa/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dphis_dt_turb", dims=dims_2d, units="m^2/s^3"),
            tracer_tendencies=tracer_tends,
        )
        return tendencies, tke_out

    def reset_state():
        return None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_turbulence(
    turbulence_config: TurbulenceConfig,
    dt: float,
) -> Callable:
    """Create turbulence physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric, phys_state=None)
               -> (NonHydrostaticTendencies, tke_out)

    When *phys_state* (a ``PhysicsState``) is passed, TKE is read from
    ``phys_state.tke`` and the updated TKE is returned as the second
    element of the result tuple.
    """
    scheme_name, turb_fn, scheme_config = _get_turbulence_fn(turbulence_config)
    needs_tke = scheme_name in ("tke", "clubb_lite", "edmf")

    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        phys_state=None,
    ):
        tke_out = None
        theta_p = state.theta_prime.data
        rho_p = state.rho_prime.data
        u_data = state.u.data
        v_data = state.v.data
        tracers = state.tracers.data

        theta_0 = height_coord.theta_ref
        rho_0 = height_coord.rho_ref

        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p,
            rho_0 + rho_p,
        )

        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape
        shape_w = state.w.data.shape
        shape_2d = state.phis.data.shape
        n_tracers = tracers.shape[-1] if tracers.ndim >= 5 else 0

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        # Mirror the hydrostatic bridge: pin defaulted allocations to the
        # state precision so x64-default zeros do not poison the column
        # path.
        _state_dtype = T.dtype
        _phis_dtype = state.phis.data.dtype

        if turb_fn is None:
            return NonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_turb", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_turb", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_turb", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dtheta_prime_dt_turb", dims=dims_3d, units="K/s"),
                drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_turb", dims=dims_3d, units="kg/m^3/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_turb", dims=dims_2d, units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_turb", dims=dims_tr, units="1/s"),
            ), tke_out

        # Use terrain-aware heights for NH columns.
        z_full = terrain_metric.z_full_3d.reshape(ncol, nlev)
        z_half = terrain_metric.z_half_3d.reshape(ncol, nlev + 1)

        # Interface pressure from evolving column state (not fixed reference).
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        ).reshape(ncol, nlev + 1)

        # Reshape to columns
        T_col = T.reshape(ncol, nlev)
        u_col = u_data.reshape(ncol, nlev)
        v_col = v_data.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        rho_col = rho_total.reshape(ncol, nlev)

        q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)
        if n_tracers > 0:
            q_v_col = tracers[..., 0].reshape(ncol, nlev)

        T_sfc = T_col[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])

        if needs_tke:
            if phys_state is not None:
                tke_in = phys_state.tke
                if tke_in.shape != (ncol, nlev):
                    tke_in = jnp.full((ncol, nlev), scheme_config.tke_min, dtype=_state_dtype)
            else:
                tke_in = jnp.full((ncol, nlev), scheme_config.tke_min, dtype=_state_dtype)
            turb_out, tke_new = turb_fn(
                u_col, v_col, T_col, q_v_col, tke_in,
                p_full_col, p_half, z_full, z_half,
                T_sfc, q_sfc, rho_col, dt, scheme_config,
            )
            tke_out = tke_new
        else:
            turb_out = turb_fn(
                u_col, v_col, T_col, q_v_col,
                p_full_col, p_half, z_full, z_half,
                T_sfc, q_sfc, rho_col, dt, scheme_config,
            )

        du_dt = turb_out.du_dt.reshape(shape_3d)
        dv_dt = turb_out.dv_dt.reshape(shape_3d)
        dT_dt = turb_out.dT_dt.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        dtracers = jnp.zeros_like(tracers)
        if n_tracers > 0:
            dq_v_dt = turb_out.dq_v_dt.reshape(shape_3d)
            dtracers = dtracers.at[..., 0].set(dq_v_dt)

        tendencies = NonHydrostaticTendencies(
            du_dt=Field(data=du_dt, name="du_dt_turb", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt, name="dv_dt_turb", dims=dims_3d, units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_turb", dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_turb", dims=dims_3d, units="K/s"),
            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_turb", dims=dims_3d, units="kg/m^3/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_turb", dims=dims_2d, units="m^2/s^3"),
            dtracers_dt=Field(data=dtracers, name="dtracers_dt_turb", dims=dims_tr, units="1/s"),
        )
        return tendencies, tke_out

    def reset_state():
        return None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_turbulence(
    turbulence_config: TurbulenceConfig,
    dt: float,
) -> Callable:
    """Create turbulence physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord, grid_fields=None, phys_state=None)
               -> (SpectralHydrostaticState, tke_out)

    When *phys_state* (a ``PhysicsState``) is passed, TKE is read from
    ``phys_state.tke`` and the updated TKE is returned as the second
    element of the result tuple.
    """
    scheme_name, turb_fn, scheme_config = _get_turbulence_fn(turbulence_config)
    needs_tke = scheme_name in ("tke", "clubb_lite", "edmf")

    def physics_fn(state, grid, sigma_coord, grid_fields=None, phys_state=None):
        tke_out = None

        # Transform spectral state to grid space (or reuse precomputed fields).
        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        u = fields['u']         # (n_lat, n_lon, nlev)
        v = fields['v']
        T = fields['T']
        p_s = fields['p_s']     # (n_lat, n_lon)

        nlev = sigma_coord.n_levels
        n_lat, n_lon = p_s.shape

        # Pressure at full and half levels
        sigma_full = sigma_coord.sigma_full
        sigma_half = sigma_coord.sigma_half
        p_full = p_s[..., None] * sigma_full
        p_half = p_s[..., None] * sigma_half

        # Reshape to columns
        ncol = n_lat * n_lon
        T_col = T.reshape(ncol, nlev)
        u_col = u.reshape(ncol, nlev)
        v_col = v.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        # Pin the column-physics dtype to the gridded state precision so
        # we never silently flow x64 zeros into the column path.
        _state_dtype = T.dtype
        q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        if turb_fn is None:
            return SpectralHydrostaticState(
                vor_hat=state.vor_hat.replace(data=zero_3d),
                div_hat=state.div_hat.replace(data=zero_3d),
                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
                lnps_hat=state.lnps_hat.replace(data=zero_2d),
                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            ), tke_out

        # Heights and density
        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col)
        rho = _compute_rho(T_col, p_full_col)

        T_sfc = T_col[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])

        if needs_tke:
            if phys_state is not None:
                tke_in = phys_state.tke
                if tke_in.shape != (ncol, nlev):
                    tke_in = jnp.full((ncol, nlev), scheme_config.tke_min, dtype=_state_dtype)
            else:
                tke_in = jnp.full((ncol, nlev), scheme_config.tke_min, dtype=_state_dtype)
            turb_out, tke_new = turb_fn(
                u_col, v_col, T_col, q_v_col, tke_in,
                p_full_col, p_half_col, z_full, z_half,
                T_sfc, q_sfc, rho, dt, scheme_config,
            )
            tke_out = tke_new
        else:
            turb_out = turb_fn(
                u_col, v_col, T_col, q_v_col,
                p_full_col, p_half_col, z_full, z_half,
                T_sfc, q_sfc, rho, dt, scheme_config,
            )

        # Reshape tendencies to grid space
        du_dt = turb_out.du_dt.reshape(n_lat, n_lon, nlev)
        dv_dt = turb_out.dv_dt.reshape(n_lat, n_lon, nlev)
        dT_dt = turb_out.dT_dt.reshape(n_lat, n_lon, nlev)

        # Project wind tendencies to spectral vorticity/divergence
        # (same pattern as held_suarez_forcing_spectral)
        a = grid.radius
        im_over_a = 1j * grid.ms.astype(jnp.float64) / a
        one_over_a = 1.0 / a

        cos_lat_3d = grid.cos_lat[:, None, None]  # (n_lat, 1, 1)
        du_cos = du_dt * cos_lat_3d
        dv_cos = dv_dt * cos_lat_3d

        # curl(du, dv) -> vorticity tendency
        dvor_hat = (
            im_over_a[:, None] * sh_analysis_oc2_3d(grid, dv_cos)
            + one_over_a * sh_analysis_dmu_3d(grid, du_cos)
        )

        # div(du, dv) -> divergence tendency
        ddiv_hat = (
            im_over_a[:, None] * sh_analysis_oc2_3d(grid, du_cos)
            - one_over_a * sh_analysis_dmu_3d(grid, dv_cos)
        )

        # Temperature tendency to spectral
        dT_hat = sh_analysis_3d(grid, dT_dt)

        tendencies = SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=dvor_hat),
            div_hat=state.div_hat.replace(data=ddiv_hat),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        )
        return tendencies, tke_out

    def reset_state():
        return None

    physics_fn.reset_state = reset_state
    return physics_fn
