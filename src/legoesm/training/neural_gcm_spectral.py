"""NeuralGCM-style training: SFNO physics + spectral PE dynamical core.

Couples a Spherical Fourier Neural Operator (SFNO) as a learned physics
parameterization to the spectral primitive equation dynamical core, with
end-to-end gradient flow through both.  The SFNO replaces all subgrid
physics (radiation, convection, boundary layer, etc.) while the spectral
PE dycore handles resolved dynamics (advection, pressure gradient,
Coriolis, vertical transport).

Training pipeline::

    ERA5 (daily snapshots)
      -> carry_to_spectral_state  [SH analysis]
      -> spectral_rollout          [lax.scan: dycore + SFNO physics]
      -> spectral_state_vs_carry_loss  [SH synthesis + MSE vs ERA5 target]
      -> eqx.filter_value_and_grad -> optax update

Architecture follows Kochkov et al. (2024) "Neural General Circulation
Models for Weather and Climate", Nature 632, 1060-1066.

Relationship to other SFNO modules
-----------------------------------
- ``sfno_dycore_coupling.py`` couples SFNO as physics for *grid-space*
  dycores (cubed-sphere, FV) via ``build_segment_fn`` / ``SegmentCarry``.
  Does NOT work with the spectral PE dycore.
- ``sfno_pe.py`` uses SFNO as the *entire dycore* (replaces PE dynamics).
  NeuralGCM keeps the traditional PE dycore and adds SFNO *physics*.
- This module bridges the gap: SFNO physics tendencies are passed to
  ``spectral_pe_tendencies(..., physics_tendency=sfno_output)`` so that
  the SFNO augments (not replaces) the spectral PE dynamics.
- Building blocks reused: ``SFNO``, ``pack_pe_state``, ``unpack_pe_output``,
  ``PE3DChannelSpec``, ``dispatch_integrator``, ``spectral_pe_tendencies``,
  ERA5 pipeline, loss functions.
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
    ensure_local_cache,
    load_era5_slice,
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
) -> SpectralHydrostaticState:
    """Convert a SegmentCarry (grid-space) to SpectralHydrostaticState.

    Performs SH analysis to transform grid-space fields (u, v, T, p_s,
    phis) into spectral coefficients (vor_hat, div_hat, T_hat, lnps_hat,
    phis_hat).

    Parameters
    ----------
    carry : SegmentCarry
        Grid-space state from ERA5 ingestion.
    grid : GaussianGrid
        Gaussian grid with SH transform matrices.

    Returns
    -------
    SpectralHydrostaticState
    """
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

    return SpectralHydrostaticState(
        vor_hat=Field(vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(lnps_hat, name="lnps_hat", dims=dims_2d, units="-"),
        phis_hat=Field(phis_hat, name="phis_hat", dims=dims_2d, units="m2/s2"),
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
# Differentiable spectral rollout
# =============================================================================

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

    # Temperature: (n_lat, n_lon, nlev)
    dT = fields['T'].astype(jnp.float32) - target_carry.T
    loss = loss + config.w_T * jnp.mean(dT ** 2 * lev_w)

    # Winds: (n_lat, n_lon, nlev)
    du = fields['u'].astype(jnp.float32) - target_carry.u
    dv = fields['v'].astype(jnp.float32) - target_carry.v
    loss = loss + config.w_u * jnp.mean(du ** 2 * lev_w)
    loss = loss + config.w_v * jnp.mean(dv ** 2 * lev_w)

    # Surface pressure: (n_lat, n_lon)
    dp = fields['p_s'].astype(jnp.float32) - target_carry.p_s
    loss = loss + config.w_ps * jnp.mean(dp ** 2)

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
        _regrid_latlon_to_gaussian, _regrid_2d_to_gaussian,
    )
    from legoesm.training.vertical_interp import interp_pressure_to_sigma
    from legoesm.driver.compiled_segments import pack_carry
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

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
    phis_gauss = _regrid_2d_to_gaussian(phis_era5, lat, lon, grid)

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

def train_neural_gcm_spectral(
    config: NeuralGCMSpectralConfig = NeuralGCMSpectralConfig(),
    cache_dir: str = "data/era5_cache",
    seed: int = 0,
):
    """Train a NeuralGCM: SFNO physics coupled to spectral PE dycore.

    End-to-end differentiable training where gradients flow through
    both the SFNO weights and the spectral dynamical core.

    Parameters
    ----------
    config : NeuralGCMSpectralConfig
        Full training configuration.
    cache_dir : str
        Local cache directory for ERA5 data.
    seed : int
        Random seed for SFNO weight initialization.

    Returns
    -------
    sfno : SFNO
        Trained SFNO model.
    loss_history : list[float]
        Per-epoch average loss.
    """
    # --- 1. Create grid and vertical coordinate ---
    logger.info(
        f"Creating T{config.n_max} grid with {config.n_levels} levels"
    )
    grid = create_gaussian_grid(config.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(config.n_levels, sigma_top=config.sigma_top)
    sigma_full = jnp.asarray(sigma.sigma_full)

    # --- 2. Build SFNO ---
    spec = PE3DChannelSpec(nlev=config.n_levels)
    n_ch = spec.n_channels
    sfno_config = SFNOConfig(
        in_channels=n_ch,
        out_channels=n_ch,
        embed_dim=config.sfno_embed_dim,
        n_blocks=config.sfno_n_blocks,
        mlp_expansion=config.sfno_mlp_expansion,
        residual_prediction=False,  # predict tendencies, not states
    )
    key = jax.random.PRNGKey(seed)
    sfno = SFNO(sfno_config, grid, key=key)

    n_params = sum(
        x.size for x in jax.tree.leaves(eqx.filter(sfno, eqx.is_array))
    )
    logger.info(
        f"SFNO: {n_ch} channels, {config.sfno_embed_dim}d embed, "
        f"{config.sfno_n_blocks} blocks, {n_params:,} parameters"
    )

    # --- 3. Load ERA5 training data ---
    ic_states, target_carries = load_training_data(
        config, grid, sigma, cache_dir,
    )

    # --- 4. Precompute dycore filters ---
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

    # --- 5. Optimizer ---
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adamw(config.lr, weight_decay=config.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(sfno, eqx.is_array))

    # --- 6. Training loop ---
    n_steps_per_day = int(86400 / config.dt)
    loss_history = []

    logger.info(
        f"Training: {config.n_epochs} epochs, "
        f"{len(ic_states)} samples/epoch, "
        f"{n_steps_per_day} dycore steps/day (dt={config.dt}s)"
    )

    def make_loss_fn(ic_spectral, target_carry):
        """Build differentiable loss for one IC/target pair."""
        def loss_fn(model):
            physics_fn = make_sfno_spectral_physics(model, grid)
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

        for ic, target in zip(ic_states, target_carries):
            loss_fn = make_loss_fn(ic, target)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(sfno)
            epoch_loss += float(loss)

            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array),
                opt_state,
                eqx.filter(sfno, eqx.is_array),
            )
            sfno = eqx.apply_updates(sfno, updates)

        avg_loss = epoch_loss / max(len(ic_states), 1)
        loss_history.append(avg_loss)

        if epoch % config.log_every == 0 or epoch == config.n_epochs - 1:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:4d}: loss={avg_loss:.6f}, "
                f"time={elapsed:.1f}s"
            )

        # Checkpoint
        if (epoch + 1) % 10 == 0 or epoch == config.n_epochs - 1:
            from pathlib import Path
            from legoesm.ml.training import save_checkpoint
            ckpt_dir = Path(config.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.eqx"
            save_checkpoint(sfno, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    return sfno, loss_history
