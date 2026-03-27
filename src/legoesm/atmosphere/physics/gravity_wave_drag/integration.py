"""Model integration bridge for gravity wave drag.

Provides `make_gwd_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

GWD produces momentum tendencies (du_dt, dv_dt) and temperature
tendencies (dT_dt). For the spectral PE dycore, wind tendencies
are projected to spectral vorticity/divergence.

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

from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
    prognostic_spectral_gwd,
)
from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
    ml_gwd,
    GWDEmulator,
)
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)


def _get_gwd_fn(config: GravityWaveDragConfig):
    """Select the GWD backend based on config.scheme."""
    if config.scheme == "rayleigh":
        return "rayleigh", rayleigh_gwd, config.rayleigh
    elif config.scheme == "lindzen":
        return "lindzen", lindzen_gwd, config.lindzen
    elif config.scheme == "mcfarlane":
        return "mcfarlane", mcfarlane_gwd, config.mcfarlane
    elif config.scheme == "hines":
        return "hines", hines_gwd, config.hines
    elif config.scheme == "prognostic_spectral":
        return "prognostic_spectral", prognostic_spectral_gwd, config.prognostic_spectral
    elif config.scheme == "ml_emulator":
        return "ml_emulator", ml_gwd, config.ml_emulator
    elif config.scheme == "none":
        return "none", None, None
    else:
        raise ValueError(f"Unknown GWD scheme: {config.scheme!r}")


from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma as _compute_heights_from_sigma,
    compute_rho as _compute_rho,
)


def make_gwd_physics(
    gwd_config: GravityWaveDragConfig,
    model_type: str = "hydrostatic",
    dt: float = 300.0,
) -> Callable:
    """Create a physics function for GWD matching a model's signature.

    Parameters
    ----------
    gwd_config : GravityWaveDragConfig
        GWD configuration (selects scheme).
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
        return _make_hydrostatic_gwd(gwd_config, dt)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_gwd(gwd_config, dt)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_gwd(gwd_config, dt)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe'."
        )


# ===========================================================================
# Hydrostatic PE
# ===========================================================================

def _make_hydrostatic_gwd(
    gwd_config: GravityWaveDragConfig,
    dt: float,
) -> Callable:
    """Create GWD physics_fn for PrimitiveEquationModel.

    Signature: (state, grid, sigma_coord, phys_state=None)
               -> HydrostaticTendencies

    When *phys_state* is passed, the GWD wave action spectrum is read
    from ``phys_state.gwd_spectrum`` and the updated spectrum is
    returned as the second element of the result tuple.
    """
    scheme_name, gwd_fn, scheme_config = _get_gwd_fn(gwd_config)
    is_prognostic = scheme_name == "prognostic_spectral"
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]

    def physics_fn(
        state: HydrostaticState,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
        phys_state=None,
    ):
        gwd_spectrum_out = None
        T = state.T.data
        u = state.u.data
        v = state.v.data
        p_s = state.p_s.data

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape

        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)

        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        u_col = u.reshape(ncol, nlev)
        v_col = v.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")

        if gwd_fn is None:
            tendencies = HydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d), name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d), name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
                dT_dt=Field(data=jnp.zeros(shape_3d), name="dT_dt_gwd", dims=dims_3d, units="K/s"),
                dp_s_dt=Field(data=jnp.zeros(shape_2d), name="dp_s_dt_gwd", dims=dims_2d, units="Pa/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
            )
            return tendencies, gwd_spectrum_out

        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col)
        rho = _compute_rho(T_col, p_full_col)

        # Latitude: use grid.lat_face if available, else zeros
        lat = _get_lat_hydrostatic(grid, ncol)

        if is_prognostic:
            sc = scheme_config
            if phys_state is not None:
                spec_in = phys_state.gwd_spectrum
                if spec_in.shape[0] != ncol:
                    spec_in = jnp.full(
                        (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
                    )
            else:
                spec_in = jnp.full(
                    (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
                )
            gwd_out, spec_new = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, sc, spec_in,
            )
            gwd_spectrum_out = spec_new
        elif is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = GWDEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output,
                    key=key,
                )
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config,
                _ml_model_cache[0],
            )
        else:
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config,
            )

        du_dt = gwd_out.du_dt.reshape(shape_3d)
        dv_dt = gwd_out.dv_dt.reshape(shape_3d)
        dT_dt = gwd_out.dT_dt.reshape(shape_3d)

        tendencies = HydrostaticTendencies(
            du_dt=Field(data=du_dt, name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt, name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
            dT_dt=Field(data=dT_dt, name="dT_dt_gwd", dims=dims_3d, units="K/s"),
            dp_s_dt=Field(data=jnp.zeros(shape_2d), name="dp_s_dt_gwd", dims=dims_2d, units="Pa/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
        )
        return tendencies, gwd_spectrum_out

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn


def _get_lat_hydrostatic(grid, ncol):
    """Extract latitude array for hydrostatic columns."""
    return jnp.asarray(grid.grid_lat).reshape(-1)[:ncol]


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_gwd(
    gwd_config: GravityWaveDragConfig,
    dt: float,
) -> Callable:
    """Create GWD physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric, phys_state=None)
               -> NonHydrostaticTendencies
    """
    scheme_name, gwd_fn, scheme_config = _get_gwd_fn(gwd_config)
    is_prognostic = scheme_name == "prognostic_spectral"
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]

    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        phys_state=None,
    ):
        gwd_spectrum_out = None
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

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]

        if gwd_fn is None:
            tendencies = NonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d), name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d), name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w), name="dw_dt_gwd", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d), name="dtheta_prime_dt_gwd", dims=dims_3d, units="K/s"),
                drho_prime_dt=Field(data=jnp.zeros(shape_3d), name="drho_prime_dt_gwd", dims=dims_3d, units="kg/m^3/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_gwd", dims=dims_tr, units="1/s"),
            )
            return tendencies, gwd_spectrum_out

        # Terrain-aware heights and interface pressure.
        z_full = terrain_metric.z_full_3d.reshape(ncol, nlev)
        z_half = terrain_metric.z_half_3d.reshape(ncol, nlev + 1)
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        ).reshape(ncol, nlev + 1)

        T_col = T.reshape(ncol, nlev)
        u_col = u_data.reshape(ncol, nlev)
        v_col = v_data.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        rho_col = rho_total.reshape(ncol, nlev)

        lat = _get_lat_hydrostatic(grid, ncol)

        if is_prognostic:
            sc = scheme_config
            if phys_state is not None:
                spec_in = phys_state.gwd_spectrum
                if spec_in.shape[0] != ncol:
                    spec_in = jnp.full(
                        (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
                    )
            else:
                spec_in = jnp.full(
                    (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
                )
            gwd_out, spec_new = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half,
                z_full, z_half, rho_col, lat, dt, sc, spec_in,
            )
            gwd_spectrum_out = spec_new
        elif is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = GWDEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output,
                    key=key,
                )
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half,
                z_full, z_half, rho_col, lat, dt, scheme_config,
                _ml_model_cache[0],
            )
        else:
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half,
                z_full, z_half, rho_col, lat, dt, scheme_config,
            )

        du_dt = gwd_out.du_dt.reshape(shape_3d)
        dv_dt = gwd_out.dv_dt.reshape(shape_3d)
        dT_dt = gwd_out.dT_dt.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        tendencies = NonHydrostaticTendencies(
            du_dt=Field(data=du_dt, name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt, name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w), name="dw_dt_gwd", dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_gwd", dims=dims_3d, units="K/s"),
            drho_prime_dt=Field(data=jnp.zeros(shape_3d), name="drho_prime_dt_gwd", dims=dims_3d, units="kg/m^3/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
            dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_gwd", dims=dims_tr, units="1/s"),
        )
        return tendencies, gwd_spectrum_out

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_gwd(
    gwd_config: GravityWaveDragConfig,
    dt: float,
) -> Callable:
    """Create GWD physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord, grid_fields=None, phys_state=None)
               -> SpectralHydrostaticState
    """
    scheme_name, gwd_fn, scheme_config = _get_gwd_fn(gwd_config)
    is_prognostic = scheme_name == "prognostic_spectral"
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]

    def physics_fn(state, grid, sigma_coord, grid_fields=None, phys_state=None):
        gwd_spectrum_out = None
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralHydrostaticState,
            spectral_pe_to_grid,
        )
        from legoesm.grids.gaussian import (
            sh_analysis_3d,
            sh_analysis_oc2_3d,
            sh_analysis_dmu_3d,
        )

        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        u = fields['u']
        v = fields['v']
        T = fields['T']
        p_s = fields['p_s']

        nlev = sigma_coord.n_levels
        n_lat, n_lon = p_s.shape

        sigma_full = sigma_coord.sigma_full
        sigma_half = sigma_coord.sigma_half
        p_full = p_s[..., None] * sigma_full
        p_half = p_s[..., None] * sigma_half

        ncol = n_lat * n_lon
        T_col = T.reshape(ncol, nlev)
        u_col = u.reshape(ncol, nlev)
        v_col = v.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        if gwd_fn is None:
            tendencies = SpectralHydrostaticState(
                vor_hat=state.vor_hat.replace(data=zero_3d),
                div_hat=state.div_hat.replace(data=zero_3d),
                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
                lnps_hat=state.lnps_hat.replace(data=zero_2d),
                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            )
            return tendencies, gwd_spectrum_out

        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col)
        rho = _compute_rho(T_col, p_full_col)

        # Latitude from Gaussian grid
        lat = jnp.broadcast_to(grid.lat[:, None], (n_lat, n_lon)).reshape(ncol)

        if is_prognostic:
            sc = scheme_config
            if phys_state is not None:
                spec_in = phys_state.gwd_spectrum
                if spec_in.shape[0] != ncol:
                    spec_in = jnp.full(
                        (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
                    )
            else:
                spec_in = jnp.full(
                    (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
                )
            gwd_out, spec_new = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, sc, spec_in,
            )
            gwd_spectrum_out = spec_new
        elif is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = GWDEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output,
                    key=key,
                )
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config,
                _ml_model_cache[0],
            )
        else:
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config,
            )

        du_dt = gwd_out.du_dt.reshape(n_lat, n_lon, nlev)
        dv_dt = gwd_out.dv_dt.reshape(n_lat, n_lon, nlev)
        dT_dt = gwd_out.dT_dt.reshape(n_lat, n_lon, nlev)

        # Project wind tendencies to spectral vorticity/divergence
        a = grid.radius
        im_over_a = 1j * grid.ms.astype(jnp.float64) / a
        one_over_a = 1.0 / a

        cos_lat_3d = grid.cos_lat[:, None, None]
        du_cos = du_dt * cos_lat_3d
        dv_cos = dv_dt * cos_lat_3d

        dvor_hat = (
            im_over_a[:, None] * sh_analysis_oc2_3d(grid, dv_cos)
            + one_over_a * sh_analysis_dmu_3d(grid, du_cos)
        )
        ddiv_hat = (
            im_over_a[:, None] * sh_analysis_oc2_3d(grid, du_cos)
            - one_over_a * sh_analysis_dmu_3d(grid, dv_cos)
        )

        dT_hat = sh_analysis_3d(grid, dT_dt)

        tendencies = SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=dvor_hat),
            div_hat=state.div_hat.replace(data=ddiv_hat),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        )
        return tendencies, gwd_spectrum_out

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn
