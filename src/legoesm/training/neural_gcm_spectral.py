"""Learned physics + spectral PE dycore: SFNO or column MLP modes.

Couples a learned physics parameterization to the spectral primitive
equation dynamical core, with end-to-end gradient flow.  Two modes:

- **SFNO** (Kochkov et al. 2024): global SFNO predicts tendencies for
  all variables (u, v, T, q, lnps) on the Gaussian grid.
- **Column MLP** (Rasp et al. 2018): per-column MLP predicts T
  tendencies from local column state (T, u, v, q, p_s).  No spatial
  coupling — physically motivated for subgrid physics.

Both modes share the same pipeline::

    ERA5 (daily snapshots)
      -> carry_to_spectral_state   [SH analysis]
      -> spectral_rollout           [lax.scan: dycore + learned physics]
      -> spectral_state_vs_carry_loss  [SH synthesis + MSE vs ERA5 target]
      -> eqx.filter_value_and_grad -> optax update

Relationship to other modules
-----------------------------
- ``sfno_dycore_coupling.py`` / ``atmosphere.physics.neural_physics``: couple
  learned physics to *grid-space* dycores via ``build_segment_fn``.
- ``sfno_pe.py``: SFNO replaces the entire dycore.
- This module: learned physics *augments* the spectral PE dycore via
  ``spectral_pe_tendencies(..., physics_tendency=learned_output)``.
"""

from __future__ import annotations

import logging
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralHydrostaticState,
    SpectralPEConfig,
    spectral_pe_tendencies,
    spectral_pe_to_grid,
    _compute_spectral_filter,
    _compute_sponge_factor,
    _apply_sponge_filter,
    _apply_spectral_filter_to_state,
    _apply_filter_to_tracers,
)
from legoesm.core.field import Field
from legoesm.grids.gaussian import (
    GaussianGrid,
    create_gaussian_grid,
    sh_analysis,
    sh_analysis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
)
from legoesm.grids.vertical import SigmaCoordinate, create_sigma_coordinate
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.channel_packing import PE3DChannelSpec, pack_pe_state, unpack_pe_output
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.training.era5_to_state import (
    TrainingERA5Config,
    era5_to_spectral_carry,
)
from legoesm.training.losses import LossConfig, level_weights

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class NeuralGCMSpectralConfig(NamedTuple):
    """Configuration for NeuralGCM spectral training."""
    # Grid
    n_max: int = 42              # Spectral truncation (T42 ~ 2.8 deg)
    n_levels: int = 10           # Vertical levels
    sigma_top: float = 0.05      # Model top sigma (~ 50 hPa)

    # Dynamics
    dt: float = 600.0            # Dycore timestep [s] (CFL-safe for T42 explicit RK3)
    pe_config: SpectralPEConfig = SpectralPEConfig(
        hyperdiff_coeff=2.5e15,
        hyperdiff_order=2,
        time_integrator="ssp_rk3",
        spectral_filter_strength=0.01,
        spectral_filter_order=8,
    )

    # SFNO architecture
    sfno_embed_dim: int = 128
    sfno_n_blocks: int = 4
    sfno_mlp_expansion: int = 4

    # Training
    n_epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    optimizer: str = "adamw"     # adamw | adam | muon — passed to ml.training.create_optimizer
    warmup_steps: int = 100      # used by the warmup-cosine schedule

    # Data
    n_train_days: int = 365      # Number of daily IC/target pairs
    start_year: int = 2015       # ERA5 year to use

    # Loss
    loss_config: LossConfig = LossConfig()

    # Logging
    log_every: int = 5
    checkpoint_dir: str = "checkpoints/neural_gcm_spectral"


# =============================================================================
# State conversion: SegmentCarry -> SpectralHydrostaticState
# =============================================================================

def carry_to_spectral_state(
    carry,
    grid: GaussianGrid,
    include_tracers: bool = True,
) -> SpectralHydrostaticState:
    """Convert a SegmentCarry (grid-space) to SpectralHydrostaticState.

    Performs SH analysis to transform grid-space fields (u, v, T, p_s,
    phis) into spectral coefficients (vor_hat, div_hat, T_hat, lnps_hat,
    phis_hat).

    When ``include_tracers=True`` (the default), the SegmentCarry's
    moisture fields (``q_v``, ``q_c``, ``q_r``) are packaged as
    grid-space ``Field`` entries on ``state.tracers``.  This matches
    the spectral PE dycore's tracer-pytree convention: tracers stay on
    the grid even though the prognostic dynamics fields live in
    spectral space.  Set ``include_tracers=False`` to fall back to the
    pre-tracer dry pipeline (``state.tracers=None``).

    Parameters
    ----------
    carry : SegmentCarry
        Grid-space state from ERA5 ingestion.
    grid : GaussianGrid
        Gaussian grid with SH transform matrices.
    include_tracers : bool
        Whether to attach ``carry.q_v`` / ``q_c`` / ``q_r`` as grid-
        space ``Field`` entries on ``state.tracers``.

    Returns
    -------
    SpectralHydrostaticState
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "carry_to_spectral_state requires JAX_ENABLE_X64=True for "
            "float64 spectral transforms."
        )

    u = carry.u.astype(jnp.float64)
    v = carry.v.astype(jnp.float64)
    T = carry.T.astype(jnp.float64)
    p_s = carry.p_s.astype(jnp.float64)
    phis = carry.phis.astype(jnp.float64)

    # log(surface pressure) -> spectral
    lnps = jnp.log(jnp.maximum(p_s, 1.0))
    lnps_hat = sh_analysis(grid, lnps)

    # Temperature -> spectral
    T_hat = sh_analysis_3d(grid, T)

    # Surface geopotential -> spectral
    phis_hat = sh_analysis(grid, phis)

    # (u, v) -> (vor_hat, div_hat) via spectral curl/divergence
    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a
    cos_lat_3d = grid.cos_lat[:, None, None]

    u_cos = u * cos_lat_3d
    v_cos = v * cos_lat_3d

    vor_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, u_cos)
    )
    div_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, v_cos)
    )

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)
    grid_dims_3d = ("lat", "lon", "level")

    tracers = None
    if include_tracers:
        # Build a tracer dict from the SegmentCarry's q_v / q_c / q_r.
        # Cast to float64 to match the spectral PE precision contract.
        tracers = {
            "q_v": Field(
                carry.q_v.astype(jnp.float64),
                name="q_v", dims=grid_dims_3d, units="kg/kg",
            ),
            "q_c": Field(
                carry.q_c.astype(jnp.float64),
                name="q_c", dims=grid_dims_3d, units="kg/kg",
            ),
            "q_r": Field(
                carry.q_r.astype(jnp.float64),
                name="q_r", dims=grid_dims_3d, units="kg/kg",
            ),
        }

    return SpectralHydrostaticState(
        vor_hat=Field(vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(lnps_hat, name="lnps_hat", dims=dims_2d, units="-"),
        phis_hat=Field(phis_hat, name="phis_hat", dims=dims_2d, units="m2/s2"),
        tracers=tracers,
    )


# =============================================================================
# SFNO as spectral physics
# =============================================================================

def _channel_input_scale(nlev: int) -> jnp.ndarray:
    """Per-channel normalization factors for PE3DChannelSpec.

    Divides each channel by its typical magnitude so the SFNO sees O(1)
    inputs regardless of physical units.  Layout: [u, v, T, q, lnps, phis].
    """
    return jnp.concatenate([
        jnp.full(nlev, 30.0),       # u [m/s]
        jnp.full(nlev, 30.0),       # v [m/s]
        jnp.full(nlev, 300.0),      # T [K]
        jnp.full(nlev, 0.01),       # q [kg/kg]
        jnp.array([12.0]),          # lnps [-]
        jnp.array([50000.0]),       # phis [m2/s2]
    ])


def _channel_tendency_scale(nlev: int) -> jnp.ndarray:
    """Per-channel output scaling from O(1) SFNO output to physical tendencies.

    Typical atmospheric physics tendency magnitudes per second.
    """
    return jnp.concatenate([
        jnp.full(nlev, 1e-4),       # du/dt [m/s^2]
        jnp.full(nlev, 1e-4),       # dv/dt [m/s^2]
        jnp.full(nlev, 1e-4),       # dT/dt [K/s]
        jnp.full(nlev, 1e-7),       # dq/dt [kg/kg/s]
        jnp.array([1e-6]),          # dlnps/dt [1/s]
        jnp.array([0.0]),           # dphis/dt (static)
    ])


def make_sfno_spectral_physics(sfno: SFNO, grid: GaussianGrid):
    """Create a physics_fn compatible with SpectralPEModel.step().

    The returned function has signature::

        physics_fn(state, grid, sigma_coord)
            -> SpectralHydrostaticState  (tendencies)

    Includes input normalization and output scaling so that a
    randomly-initialized SFNO produces stable, small tendencies.

    Internally:
    1. ``pack_pe_state`` transforms spectral state to grid-space tensor
    2. Input is normalized to O(1) per channel
    3. SFNO forward pass (residual_prediction=False) produces O(1) output
    4. Output is scaled to physical tendency magnitudes
    5. ``unpack_pe_output`` converts to spectral tendencies

    Parameters
    ----------
    sfno : SFNO
        Spherical Fourier Neural Operator (eqx.Module).
    grid : GaussianGrid
        Grid for SH transforms inside pack/unpack.

    Returns
    -------
    callable
        Physics function for the spectral PE dycore.
    """
    nlev = (sfno.config.in_channels - 2) // 4  # n_channels = 4*nlev + 2
    in_scale = _channel_input_scale(nlev)
    out_scale = _channel_tendency_scale(nlev)

    def physics_fn(state, grid_, sigma_coord):
        packed = pack_pe_state(state, grid_)
        # Normalize inputs to O(1)
        packed_norm = packed / jnp.maximum(in_scale, 1e-10)
        # SFNO forward: O(1) in, O(1) out
        output_norm = sfno(packed_norm.astype(jnp.float32), grid_)
        # Scale to physical tendency magnitudes
        output = output_norm * out_scale
        return unpack_pe_output(
            output.astype(jnp.float64), state, grid_, mode="tendencies",
        )

    return physics_fn


# =============================================================================
# Column MLP as spectral physics (Rasp et al. 2018 style)
# =============================================================================

# The column MLP physics component lives in the physics directory:
#   legoesm.atmosphere.physics.learned_column
# Re-export the coupling function for training convenience.
from legoesm.atmosphere.physics.learned_column import (  # noqa: E402
    make_column_physics_fn as make_column_mlp_spectral_physics,
    build_column_physics,
)


# =============================================================================
# Physics-based parameterizations with trainable parameters
# =============================================================================

def make_physics_params_spectral_physics(params, grid, dt):
    """Create a spectral PE physics_fn from trainable physics parameters.

    Rebuilds the combined physics (radiation + convection + turbulence + ...)
    with the current parameter values as JAX arrays so that gradients
    flow through the physics computations back to the parameters.

    The trainable parameters are injected into the scheme configs:
    - ``tau_equator``, ``tau_pole`` → gray radiation optical depth
    - ``sbm_tau_c``, ``sbm_RH_ref`` → SBM convection timescale/humidity

    Parameters
    ----------
    params : TrainablePhysicsParams
        Current trainable parameter values (eqx.Module).
    grid : GaussianGrid
        Grid for spectral transforms.
    dt : float
        Dycore timestep [s].

    Returns
    -------
    callable
        ``physics_fn(state, grid, sigma_coord) -> SpectralHydrostaticState``
    """
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import (
        RadiationConfig, GrayRadiationConfig,
    )
    from legoesm.atmosphere.physics.convection.config import (
        ConvectionConfig, SBMConfig,
    )

    p = params.as_dict()

    # Build gray radiation config with trainable tau
    gray_cfg = GrayRadiationConfig(
        tau_equator=p.get('tau_equator', 7.2),
        tau_pole=p.get('tau_pole', 1.8),
    )
    rad_cfg = RadiationConfig(scheme="gray", gray=gray_cfg)

    # Build SBM convection config with trainable timescale + RH
    sbm_cfg = SBMConfig(
        tau_c=p.get('sbm_tau_c', 7200.0),
        RH_ref=p.get('sbm_RH_ref', 0.7),
    )
    conv_cfg = ConvectionConfig(scheme="sbm", sbm=sbm_cfg)

    physics_config = PhysicsConfig(radiation=rad_cfg, convection=conv_cfg)
    raw_fn = make_physics(physics_config, model_type="spectral_pe", dt=dt)

    def physics_fn(state, grid_, sigma_coord):
        result = raw_fn(state, grid_, sigma_coord)
        return result[0] if isinstance(result, tuple) else result

    return physics_fn


# =============================================================================
# Differentiable spectral rollout
# =============================================================================

def _compute_tracer_filter(
    grid: GaussianGrid,
    pe_config: SpectralPEConfig,
    spectral_filter: jnp.ndarray | None,
    dt: float,
) -> jnp.ndarray | None:
    """Build the per-SH-mode multiplicative filter applied to grid-space
    tracers in :func:`spectral_rollout`.

    Mirrors :meth:`SpectralPrimitiveEquationModel._ensure_tracer_filter`:
    combines the spectral exponential filter (de-aliasing) with the
    implicit hyperdiffusion factor ``exp(-nu · eig · dt_eff)``.  Returns
    ``None`` when neither knob is active (caller should then skip the
    SH round-trip on tracers entirely).
    """
    components = []
    if spectral_filter is not None:
        components.append(spectral_filter)
    if pe_config.hyperdiff_coeff > 0:
        nu = pe_config.hyperdiff_coeff
        order = pe_config.hyperdiff_order
        eig = (grid.ls * (grid.ls + 1) / grid.radius ** 2) ** order
        integrator = pe_config.time_integrator.lower()
        dt_eff = 2.0 * dt if 'leapfrog' in integrator else dt
        components.append(jnp.exp(-nu * eig * dt_eff))
    if not components:
        return None
    combined = components[0]
    for c in components[1:]:
        combined = combined * c
    return combined


def spectral_rollout(
    initial_state: SpectralHydrostaticState,
    physics_fn,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    pe_config: SpectralPEConfig,
    dt: float,
    n_steps: int,
    sponge_factor: jnp.ndarray | None = None,
    spectral_filter: jnp.ndarray | None = None,
) -> SpectralHydrostaticState:
    """Roll out spectral PE + SFNO physics for n_steps using lax.scan.

    Each step:
    1. Compute combined tendencies (dynamics + SFNO physics)
    2. Integrate with SSP-RK3 (or configured integrator)
    3. Apply sponge layer damping (if enabled)
    4. Apply spectral filter (if enabled)
    5. Apply tracer filter (combined spectral + hyperdiff via SH round-
       trip) when ``initial_state.tracers`` is non-empty AND either
       ``spectral_filter`` or ``pe_config.hyperdiff_coeff > 0`` is on.

    Gradient checkpointing is applied per step so memory scales as
    O(1) per step rather than O(n_steps).

    Parameters
    ----------
    initial_state : SpectralHydrostaticState
        Initial condition in spectral space.
    physics_fn : callable
        ``(state, grid, sigma_coord) -> SpectralHydrostaticState``
        SFNO physics function from ``make_sfno_spectral_physics``.
    grid : GaussianGrid
    sigma_coord : SigmaCoordinate
    pe_config : SpectralPEConfig
    dt : float
        Time step [s].
    n_steps : int
        Number of steps to integrate.
    sponge_factor : array or None
        Precomputed sponge damping factors per level.
    spectral_filter : array or None
        Precomputed exponential spectral filter.

    Returns
    -------
    SpectralHydrostaticState
        State after n_steps * dt seconds.
    """
    integrator_name = pe_config.time_integrator
    ms = grid.ms  # for sponge filter

    # Precompute the tracer filter (mirrors the precomputation done by
    # the model class for ordinary stepping).  ``None`` when neither
    # the spectral filter nor hyperdiffusion is enabled.
    tracer_filter = _compute_tracer_filter(
        grid, pe_config, spectral_filter, dt,
    )

    def tendency_fn(s):
        phys = physics_fn(s, grid, sigma_coord)
        return spectral_pe_tendencies(s, grid, sigma_coord, pe_config, phys)

    def step_fn(state, _):
        new_state = dispatch_integrator(state, tendency_fn, dt, integrator_name)

        # Implicit sponge damping at model top
        if sponge_factor is not None:
            new_state = _apply_sponge_filter(new_state, sponge_factor, ms)

        # Exponential spectral filter on highest wavenumbers
        if spectral_filter is not None:
            new_state = _apply_spectral_filter_to_state(
                new_state, spectral_filter,
            )

        # Combined spectral + implicit hyperdiff applied to grid-space
        # tracers via one SH round-trip per tracer per step (no-op when
        # tracer_filter is None or state.tracers is None).
        if tracer_filter is not None and new_state.tracers is not None:
            new_state = new_state._replace(
                tracers=_apply_filter_to_tracers(
                    new_state.tracers, tracer_filter, grid,
                )
            )

        return new_state, None

    step_fn_ckpt = jax.checkpoint(step_fn, prevent_cse=False)

    final_state, _ = jax.lax.scan(
        step_fn_ckpt, initial_state, None, length=n_steps,
    )
    return final_state


# =============================================================================
# Loss function
# =============================================================================

def spectral_state_vs_carry_loss(
    pred_state: SpectralHydrostaticState,
    target_carry,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    sigma_full: jnp.ndarray,
    config: LossConfig = LossConfig(),
) -> jnp.ndarray:
    """Compute weighted MSE between predicted spectral state and ERA5 target.

    Converts the predicted spectral state to grid space via SH synthesis,
    then compares against the target SegmentCarry fields.

    Parameters
    ----------
    pred_state : SpectralHydrostaticState
        Predicted state after rollout.
    target_carry : SegmentCarry
        Target ERA5 state on the Gaussian grid.
    grid : GaussianGrid
    sigma_coord : SigmaCoordinate
    sigma_full : (nlev,)
        Sigma levels for pressure weighting.
    config : LossConfig

    Returns
    -------
    scalar loss
    """
    lev_w = level_weights(sigma_full, config=config)

    # Spectral -> grid-space
    fields = spectral_pe_to_grid(pred_state, grid, sigma_coord)

    loss = jnp.float32(0.0)

    # Per-variable scale denominators — same convention as carry_mse
    # (iter-69 fix carried over to the spectral path so identical
    # LossConfigs behave consistently across spectral and grid losses).
    if config.normalize_by_scale:
        T_norm = config.T_scale ** 2
        wind_norm = config.wind_scale ** 2
        q_norm = config.q_scale ** 2
        ps_norm = config.ps_scale ** 2
    else:
        T_norm = wind_norm = q_norm = ps_norm = 1.0

    # Temperature: (n_lat, n_lon, nlev)
    dT = fields['T'].astype(jnp.float32) - target_carry.T
    loss = loss + config.w_T * jnp.mean(dT ** 2 * lev_w) / T_norm

    # Winds: (n_lat, n_lon, nlev)
    du = fields['u'].astype(jnp.float32) - target_carry.u
    dv = fields['v'].astype(jnp.float32) - target_carry.v
    loss = loss + config.w_u * jnp.mean(du ** 2 * lev_w) / wind_norm
    loss = loss + config.w_v * jnp.mean(dv ** 2 * lev_w) / wind_norm

    # Surface pressure: (n_lat, n_lon)
    dp = fields['p_s'].astype(jnp.float32) - target_carry.p_s
    loss = loss + config.w_ps * jnp.mean(dp ** 2) / ps_norm

    # Specific humidity (q_v): contribute to the loss only when the
    # predicted state actually carries a ``q_v`` tracer (i.e., the
    # rollout was set up via ``carry_to_spectral_state(..., include_
    # tracers=True)``).  Skipping silently when missing keeps the
    # function backward-compatible with dry training pipelines.
    if (
        pred_state.tracers is not None
        and "q_v" in pred_state.tracers
    ):
        _qv_raw = pred_state.tracers["q_v"]
        qv_grid = (
            _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
        ).astype(jnp.float32)
        dq = qv_grid - target_carry.q_v
        loss = loss + config.w_q * jnp.mean(dq ** 2 * lev_w) / q_norm

    return loss


# =============================================================================
# Data loading
# =============================================================================

def load_training_data(
    config: NeuralGCMSpectralConfig,
    grid: GaussianGrid,
    sigma: SigmaCoordinate,
    cache_dir: str = "data/era5_cache",
):
    """Load ERA5 daily IC/target pairs and convert to spectral states.

    Opens the ERA5 Zarr store once and batch-loads all needed time slices,
    avoiding per-slice GCS connection overhead.

    Parameters
    ----------
    config : NeuralGCMSpectralConfig
    grid : GaussianGrid
    sigma : SigmaCoordinate
    cache_dir : str
        Local cache directory for ERA5 Zarr data.

    Returns
    -------
    ic_states : list[SpectralHydrostaticState]
        Initial conditions in spectral space.
    target_carries : list[SegmentCarry]
        Targets in grid space (for loss computation).
    """
    import numpy as np
    from legoesm.training.era5_to_state import (
        _open_era5_zarr, _resolve_var, ERA5Slice,
        regrid_latlon_to_gaussian, regrid_2d_to_gaussian,
    )
    era5_config = TrainingERA5Config(dt_hours=6)
    n_days = config.n_train_days
    # Time indices: need days 0..n_days (n_days+1 snapshots for n_days pairs)
    time_indices = [d * 4 for d in range(n_days + 1)]

    logger.info(
        f"Loading {n_days} daily ERA5 pairs "
        f"(opening Zarr store once, reading {len(time_indices)} snapshots)..."
    )

    # Open store once
    store = era5_config.zarr_store
    ds = _open_era5_zarr(store)

    # Read lat/lon and pressure levels
    lat = np.deg2rad(ds.lat.values.astype(np.float64))
    lon = np.deg2rad(ds.lon.values.astype(np.float64))
    plev_hPa = np.array(era5_config.levels, dtype=np.float64)
    plev_Pa = np.sort(plev_hPa * 100.0)
    level_dim = "level" if "level" in ds.dims else "pressure_level"

    # Load surface geopotential (static, no time dim)
    phis_var = _resolve_var(ds, "geopotential_at_surface")
    if phis_var:
        phis_era5 = ds[phis_var].values.astype(np.float32)
    else:
        phis_era5 = np.zeros((len(lat), len(lon)), dtype=np.float32)
    phis_gauss = regrid_2d_to_gaussian(phis_era5, lat, lon, grid)

    sigma_full = np.asarray(sigma.sigma_full)

    def _load_one_snapshot(time_idx):
        """Load one ERA5 snapshot and regrid to model grid."""
        ds_t = ds.isel(time=time_idx)

        def _get_3d(name):
            r = _resolve_var(ds_t, name)
            if r is None:
                return np.zeros((len(lat), len(lon), len(plev_Pa)))
            data = ds_t[r].sel({level_dim: list(era5_config.levels)}).values
            if data.ndim == 3:
                dims = list(ds_t[r].dims)
                spatial = {"lat", "lon", "latitude", "longitude"}
                level_axis = next(
                    (i for i, d in enumerate(dims) if d not in spatial), 0,
                )
                if level_axis != 2:
                    data = np.moveaxis(data, level_axis, -1)
            if plev_hPa[0] > plev_hPa[-1]:
                data = data[..., ::-1]
            return data.astype(np.float32)

        def _get_2d(name):
            r = _resolve_var(ds_t, name)
            if r is None:
                r = _resolve_var(ds, name)
                if r is None:
                    return np.zeros((len(lat), len(lon)), dtype=np.float32)
                return ds[r].values.squeeze().astype(np.float32)
            data = ds_t[r].values
            data = data.squeeze()
            while data.ndim > 2:
                data = data[0]
            return data.astype(np.float32)

        era5 = ERA5Slice(
            T=_get_3d("temperature"),
            u=_get_3d("u_component_of_wind"),
            v=_get_3d("v_component_of_wind"),
            q=_get_3d("specific_humidity"),
            p_s=_get_2d("surface_pressure"),
            sst=_get_2d("skin_temperature"),
            phis=phis_era5,
            lat=lat, lon=lon, plev_Pa=plev_Pa,
        )
        return era5_to_spectral_carry(era5, grid, sigma)

    # Load all snapshots
    import time as _time
    t0 = _time.time()
    carries = []
    for i, tidx in enumerate(time_indices):
        carries.append(_load_one_snapshot(tidx))
        if (i + 1) % 50 == 0:
            elapsed = _time.time() - t0
            logger.info(f"  Loaded {i+1}/{len(time_indices)} snapshots ({elapsed:.0f}s)")

    logger.info(f"Loaded {len(time_indices)} snapshots ({_time.time()-t0:.0f}s)")

    # Build IC/target pairs
    ic_states = []
    target_carries = []
    for d in range(n_days):
        ic_states.append(carry_to_spectral_state(carries[d], grid))
        target_carries.append(carries[d + 1])

    logger.info(f"Built {n_days} IC/target pairs")
    return ic_states, target_carries


# =============================================================================
# Training entry point
# =============================================================================

def _train_spectral_loop(
    model: eqx.Module,
    make_physics_fn,
    grid: GaussianGrid,
    sigma: SigmaCoordinate,
    ic_states,
    target_carries,
    config: NeuralGCMSpectralConfig,
):
    """Shared training loop for any learned-physics model coupled to the
    spectral PE dycore.

    Parameters
    ----------
    model : eqx.Module
        Learnable model (SFNO or NeuralPhysics).
    make_physics_fn : callable
        ``(model, grid) -> physics_fn`` factory.
    grid, sigma : grid and vertical coordinate.
    ic_states : list of SpectralHydrostaticState.
    target_carries : list of SegmentCarry.
    config : NeuralGCMSpectralConfig.

    Returns
    -------
    model : updated eqx.Module
    loss_history : list[float]
    """
    sigma_full = jnp.asarray(sigma.sigma_full)
    pe_config = config.pe_config

    sponge_factor = None
    if pe_config.sponge_tau > 0:
        sponge_factor = _compute_sponge_factor(
            sigma.sigma_full, pe_config.sponge_sigma,
            pe_config.sponge_tau, config.dt,
        )
    spectral_filter = None
    if pe_config.spectral_filter_strength > 0:
        spectral_filter = _compute_spectral_filter(
            grid.ls, grid.n_max,
            order=pe_config.spectral_filter_order,
            cutoff_fraction=pe_config.spectral_filter_strength,
        )

    from legoesm.ml.training import TrainingConfig, create_optimizer

    n_steps_per_day = int(86400 / config.dt)
    optimizer = create_optimizer(TrainingConfig(
        lr=config.lr,
        warmup_steps=config.warmup_steps,
        total_steps=max(1, config.n_epochs * max(1, len(ic_states))),
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
        optimizer=config.optimizer,
    ))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss_history = []

    logger.info(
        f"Training: {config.n_epochs} epochs, "
        f"{len(ic_states)} samples/epoch, "
        f"{n_steps_per_day} dycore steps/day (dt={config.dt}s)"
    )

    def make_loss_fn(ic_spectral, target_carry):
        def loss_fn(m):
            physics_fn = make_physics_fn(m, grid)
            pred = spectral_rollout(
                ic_spectral, physics_fn, grid, sigma, pe_config,
                config.dt, n_steps_per_day,
                sponge_factor, spectral_filter,
            )
            return spectral_state_vs_carry_loss(
                pred, target_carry, grid, sigma,
                sigma_full, config.loss_config,
            )
        return loss_fn

    for epoch in range(config.n_epochs):
        epoch_loss = 0.0
        t0 = time.time()
        grad_norm_val = 0.0

        for sample_idx, (ic, target) in enumerate(zip(ic_states, target_carries)):
            loss_fn = make_loss_fn(ic, target)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

            # --- NaN / Inf detection (outside JIT, values are materialized) ---
            loss_val = float(loss)
            if jnp.isnan(loss) or jnp.isinf(loss):
                raise RuntimeError(
                    f"NaN/Inf loss detected at epoch {epoch}, sample {sample_idx} "
                    f"(loss={loss_val}). "
                    "Check CFL conditions, parameter bounds, and input data."
                )
            grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
            grad_norm_val = float(grad_norm)
            if jnp.isnan(grad_norm) or jnp.isinf(grad_norm):
                raise RuntimeError(
                    f"NaN/Inf gradient detected at epoch {epoch}, sample "
                    f"{sample_idx} (grad_norm={grad_norm_val}). "
                    "Consider reducing learning rate or adding gradient clipping."
                )

            epoch_loss += loss_val

            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array),
                opt_state,
                eqx.filter(model, eqx.is_array),
            )
            model = eqx.apply_updates(model, updates)

        avg_loss = epoch_loss / max(len(ic_states), 1)
        loss_history.append(avg_loss)

        if epoch % config.log_every == 0 or epoch == config.n_epochs - 1:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:4d}: loss={avg_loss:.6f}, "
                f"grad_norm={grad_norm_val:.6e}, time={elapsed:.1f}s"
            )

        if (epoch + 1) % 10 == 0 or epoch == config.n_epochs - 1:
            from pathlib import Path
            from legoesm.ml.training import save_checkpoint
            ckpt_dir = Path(config.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.eqx"
            save_checkpoint(model, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    return model, loss_history


def train_neural_gcm_spectral(
    config: NeuralGCMSpectralConfig = NeuralGCMSpectralConfig(),
    cache_dir: str = "data/era5_cache",
    seed: int = 0,
):
    """Train NeuralGCM: SFNO physics + spectral PE dycore.

    Returns (trained_sfno, loss_history).
    """
    grid = create_gaussian_grid(config.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(config.n_levels, sigma_top=config.sigma_top)

    spec = PE3DChannelSpec(nlev=config.n_levels)
    sfno = SFNO(
        SFNOConfig(
            in_channels=spec.n_channels,
            out_channels=spec.n_channels,
            embed_dim=config.sfno_embed_dim,
            n_blocks=config.sfno_n_blocks,
            mlp_expansion=config.sfno_mlp_expansion,
            residual_prediction=False,
        ),
        grid,
        key=jax.random.PRNGKey(seed),
    )
    n_p = sum(x.size for x in jax.tree.leaves(eqx.filter(sfno, eqx.is_array)))
    logger.info(f"SFNO: {spec.n_channels}ch, {config.sfno_embed_dim}d, "
                f"{config.sfno_n_blocks} blocks, {n_p:,} params")

    ic_states, target_carries = load_training_data(config, grid, sigma, cache_dir)

    return _train_spectral_loop(
        sfno, make_sfno_spectral_physics,
        grid, sigma, ic_states, target_carries, config,
    )


def train_column_mlp_spectral(
    config: NeuralGCMSpectralConfig = NeuralGCMSpectralConfig(),
    cache_dir: str = "data/era5_cache",
    seed: int = 0,
    hidden_dim: int = 256,
    n_layers: int = 4,
    residual_scale: float = 0.01,
):
    """Train column MLP physics (Rasp 2018) + spectral PE dycore.

    Returns (trained_neural_physics, loss_history).
    """
    grid = create_gaussian_grid(config.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(config.n_levels, sigma_top=config.sigma_top)

    nn_phys = build_column_physics(
        nlev=config.n_levels,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        residual_scale=residual_scale,
        key=jax.random.PRNGKey(seed),
    )
    n_p = sum(x.size for x in jax.tree.leaves(eqx.filter(nn_phys, eqx.is_array)))
    logger.info(f"Column MLP: {config.n_levels} levels, {hidden_dim}d, "
                f"{n_layers} layers, {n_p:,} params")

    ic_states, target_carries = load_training_data(config, grid, sigma, cache_dir)

    return _train_spectral_loop(
        nn_phys, make_column_mlp_spectral_physics,
        grid, sigma, ic_states, target_carries, config,
    )


def train_physics_params_spectral(
    config: NeuralGCMSpectralConfig = NeuralGCMSpectralConfig(),
    cache_dir: str = "data/era5_cache",
):
    """Train physics-based parameterization parameters + spectral PE dycore.

    Tunes the parameters of combined physics schemes (gray radiation,
    SBM convection) by backpropagating through both the physics
    computations and the spectral dynamical core.

    Trainable parameters (via ``TrainablePhysicsParams``):
    - ``tau_equator``, ``tau_pole``: gray radiation optical depths
    - ``sbm_tau_c``: SBM convection relaxation timescale
    - ``sbm_RH_ref``: SBM convection reference relative humidity

    Returns (trained_params, loss_history).
    """
    from legoesm.training.trainable_params import TrainablePhysicsParams

    grid = create_gaussian_grid(config.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(config.n_levels, sigma_top=config.sigma_top)

    params = TrainablePhysicsParams.from_defaults()
    n_p = len(params.raw_values)
    logger.info(f"Physics params: {n_p} trainable ({', '.join(params.raw_values)})")
    for k, v in params.as_dict().items():
        logger.info(f"  {k} = {float(v):.4f}")

    ic_states, target_carries = load_training_data(config, grid, sigma, cache_dir)

    dt = config.dt

    def _make_physics_fn(p, grid_):
        return make_physics_params_spectral_physics(p, grid_, dt)

    return _train_spectral_loop(
        params, _make_physics_fn,
        grid, sigma, ic_states, target_carries, config,
    )
