"""Model integration bridge for microphysics.

Provides `make_microphysics_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

Supported model types:
- "hydrostatic"  : PrimitiveEquationModel (sigma coordinates)
- "nonhydrostatic": CompressibleEulerModel (z* coordinates)
- "spectral_pe"  : SpectralPEModel (Gaussian grid + sigma coordinates)
"""

from __future__ import annotations

from typing import Callable

import jax
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

from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.atmosphere.physics._shared import zero_like_tracers
from legoesm.grids.gaussian import sh_analysis_3d
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.ml_emulator import (
    ml_microphysics,
    MicrophysicsEmulator,
)
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)


def _get_microphysics_fn(config: MicrophysicsConfig):
    """Select the microphysics backend based on config.scheme.

    Returns
    -------
    scheme_name : str
    micro_fn : callable or None
    scheme_config : NamedTuple or None
    """
    if config.scheme == "kessler":
        return "kessler", kessler_microphysics, config.kessler
    elif config.scheme == "sundqvist":
        return "sundqvist", sundqvist_microphysics, config.sundqvist
    elif config.scheme == "seifert_beheng":
        return "seifert_beheng", seifert_beheng_microphysics, config.seifert_beheng
    elif config.scheme == "morrison":
        return "morrison", morrison_microphysics, config.morrison
    elif config.scheme == "thompson":
        return "thompson", thompson_microphysics, config.thompson
    elif config.scheme == "ml_emulator":
        return "ml_emulator", ml_microphysics, config.ml_emulator
    elif config.scheme == "none":
        return "none", None, None
    else:
        raise ValueError(f"Unknown microphysics scheme: {config.scheme!r}")


from legoesm.atmosphere.physics._shared import (
    compute_layer_dz as _compute_heights_from_sigma,
    compute_rho as _compute_rho,
)


def make_microphysics_physics(
    microphysics_config: MicrophysicsConfig,
    model_type: str = "hydrostatic",
    dt: float = 300.0,
) -> Callable:
    """Create a physics function for microphysics matching a model's signature.

    Parameters
    ----------
    microphysics_config : MicrophysicsConfig
        Microphysics configuration (selects scheme).
    model_type : str
        One of "hydrostatic", "nonhydrostatic", "spectral_pe".
    dt : float
        Model time step [s].

    Returns
    -------
    Callable
        Physics function with the correct signature for the model.
    """
    # ``model_type="mpas"`` reuses the hydrostatic factory: the
    # ``_make_hydrostatic_microphysics`` bridge reshapes
    # ``(*shape_2d, nlev)`` to ``(ncol, nlev)`` and never references
    # grid lat/lon — works identically for cubed-sphere ``(face, n, n)``,
    # lat-lon ``(n_lat, n_lon)``, and MPAS Voronoi ``(nCells,)``.
    if model_type in ("hydrostatic", "mpas"):
        return _make_hydrostatic_microphysics(microphysics_config, dt)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_microphysics(microphysics_config, dt)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_microphysics(microphysics_config, dt)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe', 'mpas'."
        )


# ===========================================================================
# Hydrostatic PE
# ===========================================================================

def _make_hydrostatic_microphysics(
    microphysics_config: MicrophysicsConfig,
    dt: float,
) -> Callable:
    """Create microphysics physics_fn for PrimitiveEquationModel.

    Signature: (state, grid, sigma_coord) -> HydrostaticTendencies
    """
    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]

    def physics_fn(
        state: HydrostaticState,
        grid,
        sigma_coord: SigmaCoordinate,
    ) -> HydrostaticTendencies:
        T = state.T.data
        p_s = state.p_s.data

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape

        # Derive Field metadata from the input state so the returned
        # tendencies match the underlying grid: cubed-sphere uses
        # ("face","x","y",...), lat-lon uses ("lat","lon",...), and
        # MPAS uses ("nCells",...).
        dims_3d = state.T.dims
        dims_2d = state.p_s.dims
        u_shape = state.u.data.shape
        u_dims = state.u.dims
        v_dims = state.v.dims if state.v is not None else None
        v_shape = state.v.data.shape if state.v is not None else None

        # Pin defaulted allocations to the state precision so we never
        # silently flow x64 zeros into the column physics path.
        _state_dtype = T.dtype

        def _zero_dv_dt():
            """``None`` for MPAS (no v), Field of zeros otherwise."""
            if state.v is None:
                return None
            return Field(
                data=jnp.zeros(v_shape, dtype=_state_dtype),
                name="dv_dt_micro", dims=v_dims, units="m/s^2",
            )

        if micro_fn is None:
            return HydrostaticTendencies(
                du_dt=Field(
                    data=jnp.zeros(u_shape, dtype=_state_dtype),
                    name="du_dt_micro", dims=u_dims, units="m/s^2",
                ),
                dv_dt=_zero_dv_dt(),
                dT_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dT_dt_micro", dims=dims_3d, units="K/s"),
                dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dp_s_dt_micro", dims=dims_2d, units="Pa/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
            )

        # Pressure at full and half levels
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)

        # Reshape to columns generically across cubed-sphere
        # ``shape_2d=(6,n,n)``, lat-lon ``(n_lat,n_lon)``, and MPAS
        # ``(nCells,)``.
        ncol = 1
        for s in shape_2d:
            ncol *= int(s)
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        # Helper: extract a tracer from the tracer dict, returning a
        # column-reshaped (ncol, nlev) array clipped to non-negative.
        def _get_tracer(name):
            if state.tracers is not None and name in state.tracers:
                raw = state.tracers[name]
                data = raw.data if hasattr(raw, "data") else raw
                return jnp.maximum(data.reshape(ncol, nlev), 0.0)
            return jnp.zeros((ncol, nlev), dtype=_state_dtype)

        # Extract water vapor from tracers if available; else assume dry.
        q_v_col = _get_tracer("q_v")

        rho = _compute_rho(T_col, p_full_col)
        dz = _compute_heights_from_sigma(T_col, p_half_col)

        # Extract actual hydrometeor state from tracers (fall back to zero
        # for any species not present in the tracer registry).
        hydrometeors = HydrometeorState(
            q_c=_get_tracer("q_c"),
            q_r=_get_tracer("q_r"),
            q_i=_get_tracer("q_i"),
            q_s=_get_tracer("q_s"),
            q_g=_get_tracer("q_g"),
            N_c=_get_tracer("N_c"),
            N_r=_get_tracer("N_r"),
            N_i=_get_tracer("N_i"),
        )

        if is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = MicrophysicsEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output, key=key,
                )
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half_col, rho, dz, dt,
                scheme_config, _ml_model_cache[0],
            )
        else:
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half_col, rho, dz, dt, scheme_config,
            )

        dT_dt = micro_out.dT_dt.reshape(shape_3d)

        # Propagate tracer tendencies from microphysics backend
        tracer_tends = {
            "q_v": Field(data=micro_out.dq_v_dt.reshape(shape_3d),
                         name="dq_v_dt_micro", dims=dims_3d, units="kg/kg/s"),
            "q_c": Field(data=micro_out.dq_c_dt.reshape(shape_3d),
                         name="dq_c_dt_micro", dims=dims_3d, units="kg/kg/s"),
            "q_r": Field(data=micro_out.dq_r_dt.reshape(shape_3d),
                         name="dq_r_dt_micro", dims=dims_3d, units="kg/kg/s"),
            "q_i": Field(data=micro_out.dq_i_dt.reshape(shape_3d),
                         name="dq_i_dt_micro", dims=dims_3d, units="kg/kg/s"),
            "q_s": Field(data=micro_out.dq_s_dt.reshape(shape_3d),
                         name="dq_s_dt_micro", dims=dims_3d, units="kg/kg/s"),
            "q_g": Field(data=micro_out.dq_g_dt.reshape(shape_3d),
                         name="dq_g_dt_micro", dims=dims_3d, units="kg/kg/s"),
        }

        return HydrostaticTendencies(
            du_dt=Field(
                data=jnp.zeros(u_shape, dtype=_state_dtype),
                name="du_dt_micro", dims=u_dims, units="m/s^2",
            ),
            dv_dt=_zero_dv_dt(),
            dT_dt=Field(data=dT_dt, name="dT_dt_micro", dims=dims_3d, units="K/s"),
            dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dp_s_dt_micro", dims=dims_2d, units="Pa/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
            tracer_tendencies=tracer_tends,
        )

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_microphysics(
    microphysics_config: MicrophysicsConfig,
    dt: float,
) -> Callable:
    """Create microphysics physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric) -> NonHydrostaticTendencies
    """
    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]

    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
    ) -> NonHydrostaticTendencies:
        theta_p = state.theta_prime.data
        rho_p = state.rho_prime.data
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

        # Pin defaulted allocations to the state precision so x64 zeros
        # do not silently flow into the column physics path.
        _state_dtype = T.dtype
        _phis_dtype = state.phis.data.dtype
        if micro_fn is None:
            return NonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_micro", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_micro", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_micro", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dtheta_prime_dt_micro", dims=dims_3d, units="K/s"),
                drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_micro", dims=dims_3d, units="kg/m^3/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_micro", dims=dims_tr, units="1/s"),
            )

        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]

        # Terrain-aware layer thickness and interface pressure.
        z_half_3d = terrain_metric.z_half_3d
        dz = jnp.abs(z_half_3d[..., :-1] - z_half_3d[..., 1:]).reshape(ncol, nlev)
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=z_half_3d,
        ).reshape(ncol, nlev + 1)

        # Reshape to columns
        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        rho_col = rho_total.reshape(ncol, nlev)

        # Map tracers -> HydrometeorState
        # [0]=q_v, [1]=q_c, [2]=q_r, [3]=q_i, [4]=q_s, [5]=q_g, [6]=N_c, [7]=N_r, [8]=N_i
        def _get_tracer(idx):
            if n_tracers > idx:
                return tracers[..., idx].reshape(ncol, nlev)
            return jnp.zeros((ncol, nlev), dtype=_state_dtype)

        q_v_col = _get_tracer(0)
        hydrometeors = HydrometeorState(
            q_c=_get_tracer(1),
            q_r=_get_tracer(2),
            q_i=_get_tracer(3),
            q_s=_get_tracer(4),
            q_g=_get_tracer(5),
            N_c=_get_tracer(6),
            N_r=_get_tracer(7),
            N_i=_get_tracer(8),
        )

        if is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = MicrophysicsEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output, key=key,
                )
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half, rho_col, dz, dt,
                scheme_config, _ml_model_cache[0],
            )
        else:
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half, rho_col, dz, dt, scheme_config,
            )

        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner).
        dT_dt = micro_out.dT_dt.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        # Map output fields -> dtracers_dt
        dtracers = jnp.zeros_like(tracers)
        # Tracer mapping: 0=q_v, 1=q_c, 2=q_r, 3=q_i, 4=q_s, 5=q_g, 6=N_c, 7=N_r, 8=N_i
        tend_fields = [
            micro_out.dq_v_dt, micro_out.dq_c_dt, micro_out.dq_r_dt,
            micro_out.dq_i_dt, micro_out.dq_s_dt, micro_out.dq_g_dt,
            micro_out.dN_c_dt, micro_out.dN_r_dt, micro_out.dN_i_dt,
        ]
        for idx, field in enumerate(tend_fields):
            if n_tracers > idx:
                dtracers = dtracers.at[..., idx].set(field.reshape(shape_3d))

        return NonHydrostaticTendencies(
            du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_micro", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_micro", dims=dims_3d, units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_micro", dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_micro", dims=dims_3d, units="K/s"),
            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_micro", dims=dims_3d, units="kg/m^3/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
            dtracers_dt=Field(data=dtracers, name="dtracers_dt_micro", dims=dims_tr, units="1/s"),
        )

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_microphysics(
    microphysics_config: MicrophysicsConfig,
    dt: float,
) -> Callable:
    """Create microphysics physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord, grid_fields=None) -> SpectralHydrostaticState

    The bridge pulls ``q_v`` and the full hydrometeor state out of
    ``state.tracers`` (when present), runs the column microphysics
    backend, and returns a ``SpectralHydrostaticState`` whose ``T_hat``
    carries the spectral latent-heating tendency *and* whose ``tracers``
    dict carries grid-space ``dq_v_dt`` / ``dq_c_dt`` / ``dq_r_dt`` /
    etc.  The dycore RHS (``spectral_pe_tendencies``) adds these tracer
    tendencies to its own advective tendencies during the SSP-RK stages.
    """
    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]

    # Tracer key → MicrophysicsOutput attribute name.  Mirrors the
    # ``HydrometeorState`` field layout in ``microphysics/output.py``
    # plus ``q_v``.  The dycore RHS only flows tendencies for keys that
    # exist on the input ``state.tracers``; missing keys are silently
    # dropped (no carry to write into).
    _TRACER_TEND_MAP = {
        "q_v": "dq_v_dt",
        "q_c": "dq_c_dt",
        "q_r": "dq_r_dt",
        "q_i": "dq_i_dt",
        "q_s": "dq_s_dt",
        "q_g": "dq_g_dt",
        "N_c": "dN_c_dt",
        "N_r": "dN_r_dt",
        "N_i": "dN_i_dt",
    }

    def physics_fn(state, grid, sigma_coord, grid_fields=None):
        # Transform spectral state to grid space
        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        T = fields['T']
        p_s = fields['p_s']

        nlev = sigma_coord.n_levels
        n_lat, n_lon = p_s.shape

        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)
        # Pin the column-physics dtype to the gridded state precision so
        # we do not silently flow x64 zeros into the column path.
        _state_dtype = T.dtype

        if micro_fn is None:
            # Mirror the input tracer pytree shape with zeros so the
            # orchestrator's accumulator and the dycore RHS see a
            # consistent tendency structure even when microphysics is
            # disabled.
            return SpectralHydrostaticState(
                vor_hat=state.vor_hat.replace(data=zero_3d),
                div_hat=state.div_hat.replace(data=zero_3d),
                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
                lnps_hat=state.lnps_hat.replace(data=zero_2d),
                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
                tracers=zero_like_tracers(state.tracers),
            )

        # Pressure at full and half levels
        sigma_full = sigma_coord.sigma_full
        sigma_half = sigma_coord.sigma_half
        p_full = p_s[..., None] * sigma_full
        p_half = p_s[..., None] * sigma_half

        # Reshape to columns
        ncol = n_lat * n_lon
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        # Pull tracer fields out of ``state.tracers`` and reshape to the
        # column-physics ``(ncol, nlev)`` layout.  Backend microphysics
        # schemes assume non-negative mixing ratios, so clip on the way
        # in (matches the hydrostatic bridge's ``_get_tracer``).
        def _get_tracer(name):
            if state.tracers is not None and name in state.tracers:
                raw = state.tracers[name]
                data = raw.data if hasattr(raw, "data") else raw
                return jnp.maximum(data.reshape(ncol, nlev), 0.0)
            return jnp.zeros((ncol, nlev), dtype=_state_dtype)

        q_v_col = _get_tracer("q_v")

        rho = _compute_rho(T_col, p_full_col)
        dz = _compute_heights_from_sigma(T_col, p_half_col)

        hydrometeors = HydrometeorState(
            q_c=_get_tracer("q_c"),
            q_r=_get_tracer("q_r"),
            q_i=_get_tracer("q_i"),
            q_s=_get_tracer("q_s"),
            q_g=_get_tracer("q_g"),
            N_c=_get_tracer("N_c"),
            N_r=_get_tracer("N_r"),
            N_i=_get_tracer("N_i"),
        )

        if is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = MicrophysicsEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output, key=key,
                )
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half_col, rho, dz, dt,
                scheme_config, _ml_model_cache[0],
            )
        else:
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half_col, rho, dz, dt, scheme_config,
            )

        dT_dt = micro_out.dT_dt.reshape(n_lat, n_lon, nlev)

        # Transform T tendency to spectral space
        dT_hat = sh_analysis_3d(grid, dT_dt)

        # Build the tracer tendency dict in grid-space ``(n_lat, n_lon,
        # nlev)`` layout, matching ``SpectralHydrostaticState.tracers``.
        # Wrap each tendency back into the same container type as the
        # input state's tracer (``Field`` vs raw ``jax.Array``) so the
        # SSP-RK ``tree.map`` pytree leaves line up.  Untouched tracer
        # keys are mirrored as zeros via ``zero_like_tracers``.
        tracers_tend = None
        if state.tracers is not None:
            tt = {}
            for name, attr in _TRACER_TEND_MAP.items():
                if name not in state.tracers:
                    continue
                template = state.tracers[name]
                tend_grid = getattr(micro_out, attr).reshape(
                    n_lat, n_lon, nlev,
                )
                if hasattr(template, "data") and hasattr(template, "replace"):
                    tt[name] = template.replace(
                        data=tend_grid.astype(template.data.dtype),
                    )
                else:
                    tt[name] = tend_grid.astype(template.dtype)
            # Mirror any untouched tracer keys (e.g. a passive scalar
            # the user attached) as zeros so the orchestrator's
            # accumulator and the dycore RHS see a complete pytree.
            zeros = zero_like_tracers(state.tracers)
            if zeros is not None:
                for k, zv in zeros.items():
                    tt.setdefault(k, zv)
            tracers_tend = tt

        return SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=zero_3d),
            div_hat=state.div_hat.replace(data=zero_3d),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            tracers=tracers_tend,
        )

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn
