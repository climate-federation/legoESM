"""Model integration bridge for convection.

Provides `make_convection_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

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

from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.dca import dca_convection
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.mass_flux import mass_flux_convection
from legoesm.atmosphere.physics.convection.edmf import edmf_convection
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)


def _get_convection_fn(config: ConvectionConfig):
    """Select the convection backend based on config.scheme.

    Returns
    -------
    scheme_name : str
        Name of the scheme.
    conv_fn : callable or None
        Backend convection function.
    scheme_config : NamedTuple or None
        Scheme-specific configuration.
    """
    if config.scheme == "sbm":
        return "sbm", sbm_convection, config.sbm
    elif config.scheme == "dca":
        return "dca", dca_convection, config.dca
    elif config.scheme == "kuo":
        return "kuo", kuo_convection, config.kuo
    elif config.scheme == "mass_flux":
        return "mass_flux", mass_flux_convection, config.mass_flux
    elif config.scheme == "edmf":
        return "edmf", edmf_convection, config.edmf
    elif config.scheme == "none":
        return "none", None, None
    else:
        raise ValueError(f"Unknown convection scheme: {config.scheme!r}")


def make_convection_physics(
    convection_config: ConvectionConfig,
    model_type: str = "hydrostatic",
    dt: float = 300.0,
) -> Callable:
    """Create a physics function for convection matching a model's signature.

    Parameters
    ----------
    convection_config : ConvectionConfig
        Convection configuration (selects SBM, DCA, Kuo, mass_flux,
        EDMF, or none).
    model_type : str
        One of "hydrostatic", "nonhydrostatic", "spectral_pe".
    dt : float
        Model time step [s]. Needed for relaxation timescale.

    Returns
    -------
    Callable
        Physics function with the correct signature for the model.
    """
    if model_type == "hydrostatic":
        return _make_hydrostatic_convection(convection_config, dt)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_convection(convection_config, dt)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_convection(convection_config, dt)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe'."
        )


# ===========================================================================
# Hydrostatic PE
# ===========================================================================

def _make_hydrostatic_convection(
    convection_config: ConvectionConfig,
    dt: float,
) -> Callable:
    """Create convection physics_fn for PrimitiveEquationModel.

    Signature: (state, grid, sigma_coord, phys_state=None)
               -> (HydrostaticTendencies, conv_prog_new | None)

    When *phys_state* is passed, the convective prognostic variable is
    read from ``phys_state.conv_prog`` and the updated value is returned
    as the second element of the result tuple.
    """
    scheme_name, conv_fn, scheme_config = _get_convection_fn(convection_config)
    is_prognostic = scheme_name in ("mass_flux", "edmf")
    prog_key = None
    prog_init = None
    if is_prognostic:
        if scheme_name == "mass_flux":
            prog_key, prog_init = "M_c", scheme_config.M_c_init
        else:  # edmf
            prog_key, prog_init = "a_u", scheme_config.a_u_init

    def physics_fn(
        state: HydrostaticState,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
        phys_state=None,
    ):
        T = state.T.data          # (6, n, n, nlev)
        p_s = state.p_s.data      # (6, n, n)

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape

        # Pressure at full and half levels
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)

        # Reshape to columns: (6,n,n,...) -> (ncol, ...)
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        # Pin defaulted allocations to the state precision so x64-default
        # zeros do not silently flow into the column physics path.
        _state_dtype = T.dtype
        _ps_dtype = state.p_s.data.dtype
        # Extract water vapor from tracers if available; else assume dry.
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(ncol, nlev)
        else:
            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        conv_prog_out = None
        if conv_fn is None:
            # "none" scheme: return zero tendencies
            dT_dt = jnp.zeros(shape_3d, dtype=_state_dtype)
        elif is_prognostic:
            if phys_state is not None:
                prog_in = phys_state.conv_prog
                if prog_in.shape != (ncol,):
                    prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)
            else:
                prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)

            conv_out, prog_new = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                **{prog_key: prog_in},
                dt=dt, config=scheme_config,
            )
            conv_prog_out = prog_new
            dT_dt = conv_out.dT_dt.reshape(shape_3d)
        else:
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
            )
            dT_dt = conv_out.dT_dt.reshape(shape_3d)

        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")

        # Propagate tracer tendencies from convection backend
        tracer_tends = None
        if conv_fn is not None:
            dq_v_dt = conv_out.dq_v_dt.reshape(shape_3d)
            tracer_tends = {
                "q_v": Field(
                    data=dq_v_dt, name="dq_v_dt_conv",
                    dims=dims_3d, units="kg/kg/s",
                ),
            }

        tendencies = HydrostaticTendencies(
            du_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_conv",
                dims=dims_3d, units="m/s^2",
            ),
            dv_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_conv",
                dims=dims_3d, units="m/s^2",
            ),
            dT_dt=Field(
                data=dT_dt, name="dT_dt_conv",
                dims=dims_3d, units="K/s",
            ),
            dp_s_dt=Field(
                data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dp_s_dt_conv",
                dims=dims_2d, units="Pa/s",
            ),
            dphis_dt=Field(
                data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dphis_dt_conv",
                dims=dims_2d, units="m^2/s^3",
            ),
            tracer_tendencies=tracer_tends,
        )
        return tendencies, conv_prog_out

    def reset_state():
        pass

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_convection(
    convection_config: ConvectionConfig,
    dt: float,
) -> Callable:
    """Create convection physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric, phys_state=None)
               -> (NonHydrostaticTendencies, conv_prog_new | None)

    When *phys_state* is passed, the convective prognostic variable is
    read from ``phys_state.conv_prog`` and the updated value is returned
    as the second element of the result tuple.
    """
    scheme_name, conv_fn, scheme_config = _get_convection_fn(convection_config)
    is_prognostic = scheme_name in ("mass_flux", "edmf")
    prog_key = None
    prog_init = None
    if is_prognostic:
        if scheme_name == "mass_flux":
            prog_key, prog_init = "M_c", scheme_config.M_c_init
        else:  # edmf
            prog_key, prog_init = "a_u", scheme_config.a_u_init

    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        phys_state=None,
    ) -> NonHydrostaticTendencies:
        theta_p = state.theta_prime.data   # (6, n, n, nlev)
        rho_p = state.rho_prime.data       # (6, n, n, nlev)
        tracers = state.tracers.data       # (6, n, n, nlev, n_tracers)

        # Reference profiles
        theta_0 = height_coord.theta_ref
        rho_0 = height_coord.rho_ref

        # Total fields
        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p,
            rho_0 + rho_p,
        )

        # Temperature and pressure
        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape
        shape_w = state.w.data.shape
        shape_2d = state.phis.data.shape
        n_tracers = tracers.shape[-1] if tracers.ndim >= 5 else 0

        # Interface pressure from evolving column state (not fixed reference).
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        )

        # Reshape to columns
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        # Pin defaulted allocations to the state precision so x64 zeros
        # do not silently flow into the column physics path.
        _state_dtype = T.dtype
        _phis_dtype = state.phis.data.dtype
        # Extract q_v from tracers if available
        if n_tracers > 0:
            q_v_col = tracers[..., 0].reshape(ncol, nlev)
        else:
            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        if conv_fn is None:
            return NonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_conv", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_conv", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_conv", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dtheta_prime_dt_conv", dims=dims_3d, units="K/s"),
                drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_conv", dims=dims_3d, units="kg/m^3/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_conv", dims=dims_2d, units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_conv", dims=dims_tr, units="1/s"),
            )

        conv_prog_out = None
        if is_prognostic:
            if phys_state is not None:
                prog_in = phys_state.conv_prog
                if prog_in.shape != (ncol,):
                    prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)
            else:
                prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)

            conv_out, prog_new = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                **{prog_key: prog_in},
                dt=dt, config=scheme_config,
            )
            conv_prog_out = prog_new
        else:
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
            )

        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner)
        dT_dt = conv_out.dT_dt.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        # Tracer tendencies
        dtracers = jnp.zeros_like(tracers)
        if n_tracers > 0:
            dq_v_dt = conv_out.dq_v_dt.reshape(shape_3d)
            dtracers = dtracers.at[..., 0].set(dq_v_dt)

        tendencies = NonHydrostaticTendencies(
            du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_conv", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_conv", dims=dims_3d, units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_conv", dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_conv", dims=dims_3d, units="K/s"),
            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_conv", dims=dims_3d, units="kg/m^3/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_conv", dims=dims_2d, units="m^2/s^3"),
            dtracers_dt=Field(data=dtracers, name="dtracers_dt_conv", dims=dims_tr, units="1/s"),
        )
        return tendencies, conv_prog_out

    def reset_state():
        pass

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_convection(
    convection_config: ConvectionConfig,
    dt: float,
) -> Callable:
    """Create convection physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord, grid_fields=None, phys_state=None)
               -> (SpectralHydrostaticState, conv_prog_new | None)

    When *phys_state* is passed, the convective prognostic variable is
    read from ``phys_state.conv_prog`` and the updated value is returned
    as the second element of the result tuple.
    """
    scheme_name, conv_fn, scheme_config = _get_convection_fn(convection_config)
    is_prognostic = scheme_name in ("mass_flux", "edmf")
    prog_key = None
    prog_init = None
    if is_prognostic:
        if scheme_name == "mass_flux":
            prog_key, prog_init = "M_c", scheme_config.M_c_init
        else:  # edmf
            prog_key, prog_init = "a_u", scheme_config.a_u_init

    def physics_fn(state, grid, sigma_coord, grid_fields=None, phys_state=None):
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralHydrostaticState,
            spectral_pe_to_grid,
        )
        from legoesm.grids.gaussian import sh_analysis_3d

        # 1. Transform spectral state to grid space
        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        T = fields['T']         # (n_lat, n_lon, nlev)
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
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        # Pin the column-physics dtype to the gridded state precision so
        # we never silently flow x64 zeros into the column path.
        _state_dtype = T.dtype
        # Extract water vapor if spectral state carries tracers.
        if hasattr(state, "tracers") and state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(ncol, nlev)
        else:
            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        conv_prog_out = None
        if conv_fn is None:
            dT_dt = jnp.zeros_like(T)
        elif is_prognostic:
            if phys_state is not None:
                prog_in = phys_state.conv_prog
                if prog_in.shape != (ncol,):
                    prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)
            else:
                prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)

            conv_out, prog_new = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                **{prog_key: prog_in},
                dt=dt, config=scheme_config,
            )
            conv_prog_out = prog_new
            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)
        else:
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
            )
            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)

        # Transform T tendency to spectral space
        dT_hat = sh_analysis_3d(grid, dT_dt)

        # No wind or surface pressure tendencies from convection
        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        tendencies = SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=zero_3d),
            div_hat=state.div_hat.replace(data=zero_3d),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        )
        return tendencies, conv_prog_out

    def reset_state():
        pass

    physics_fn.reset_state = reset_state
    return physics_fn
