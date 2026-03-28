"""Training driver for differentiable dycore + WeatherBench.

Three training modes:
1. Physics parameter tuning (Mode 1)
2. Neural GCM (Mode 2) — neural network replaces/augments physics
3. SFNO coupled to dycore (Mode 3)

All modes share the same core loop:
    IC from ERA5 → rollout through dycore → loss vs ERA5 target → grad → update
"""

from __future__ import annotations

import logging
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from legoesm.driver.compiled_segments import (
    build_segment_fn,
    pack_forcing,
    SegmentForcing,
)
from legoesm.training.losses import combined_loss, LossConfig
from legoesm.training.dycore_rollout import (
    RolloutConfig,
    single_day_rollout,
    differentiable_rollout,
)

logger = logging.getLogger(__name__)


class TrainingState(NamedTuple):
    """Mutable training state."""
    step: int
    best_loss: float


# ======================================================================
# Mode 1: Physics parameter tuning
# ======================================================================

def train_physics_params(
    model,
    grid,
    sigma,
    physics_pipeline,
    initial_carries,
    target_carries,
    forcings,
    *,
    n_epochs: int = 100,
    lr: float = 1e-3,
    rollout_days: int = 1,
    dt: float = 600.0,
    loss_config: LossConfig = LossConfig(),
    log_every: int = 10,
):
    """Train physics parameters via gradient descent through the dycore.

    Parameters
    ----------
    model : dynamics model with .step()
    grid : CubedSphereGrid or GaussianGrid
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate
    physics_pipeline : PhysicsPipeline
    initial_carries : list of SegmentCarry — ICs from ERA5
    target_carries : list of SegmentCarry — targets from ERA5
    forcings : list of SegmentForcing — SST/SIC forcing per sample
    n_epochs : int
    lr : float — learning rate
    rollout_days : int — days per rollout
    dt : float — timestep
    loss_config : LossConfig
    log_every : int

    Returns
    -------
    TrainablePhysicsParams — optimized parameters
    list[float] — loss history
    """
    from legoesm.training.trainable_params import TrainablePhysicsParams

    # Initialize learnable parameters
    params = TrainablePhysicsParams.from_defaults()

    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

    sigma_full = jnp.asarray(sigma.sigma_full)
    step_unified = physics_pipeline.build_step_unified()

    loss_history = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for i, (ic, target, forcing) in enumerate(
            zip(initial_carries, target_carries, forcings)
        ):
            def _loss_fn(params_):
                seg_kw = params_.to_segment_kwargs()
                run_seg = build_segment_fn(
                    model=model,
                    step_unified=step_unified,
                    grid=grid,
                    sigma_full=sigma_full,
                    dsigma=jnp.asarray(sigma.dsigma),
                    dt=dt,
                    rad_update_steps=1,
                    microphysics="none",
                    fix_moisture=False,
                    fix_mass=False,
                    fric_decay=jnp.ones(sigma_full.shape[0]),
                    qv_smooth_coeff=0.0,
                    lat=grid.lat,
                    lon=grid.lon,
                    start_day=0.0,
                    gradient_checkpoint=True,
                    **seg_kw,
                )
                pred = single_day_rollout(ic, forcing, run_seg.raw, dt=dt)
                return combined_loss(pred, target, sigma_full, config=loss_config)

            loss, grads = eqx.filter_value_and_grad(_loss_fn)(params)
            epoch_loss += float(loss)

            # Update parameters
            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array),
                opt_state,
                eqx.filter(params, eqx.is_array),
            )
            params = eqx.apply_updates(params, updates)

        avg_loss = epoch_loss / max(len(initial_carries), 1)
        loss_history.append(avg_loss)

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            elapsed = time.time() - t0
            phys = params.as_dict()
            param_str = ", ".join(f"{k}={float(v):.4f}" for k, v in phys.items())
            logger.info(
                f"Epoch {epoch:4d}: loss={avg_loss:.6f}, "
                f"time={elapsed:.1f}s, params=[{param_str}]"
            )

    return params, loss_history


# ======================================================================
# Mode 2: Neural GCM training
# ======================================================================

def train_neural_gcm(
    model,
    grid,
    sigma,
    neural_physics,
    initial_carries,
    target_carries,
    forcings,
    *,
    n_epochs: int = 100,
    lr: float = 1e-4,
    dt: float = 600.0,
    loss_config: LossConfig = LossConfig(),
    log_every: int = 10,
):
    """Train a neural physics network coupled to the dycore.

    The neural_physics module (eqx.Module) provides tendencies
    that replace or augment traditional physics inside the compiled
    segment loop.

    Parameters
    ----------
    neural_physics : NeuralPhysics (eqx.Module)
        Neural network producing PhysicsOutput.
    (other params same as train_physics_params)

    Returns
    -------
    neural_physics : updated NeuralPhysics
    list[float] — loss history
    """
    from legoesm.training.neural_physics import make_neural_step_unified

    optimizer = optax.adamw(lr, weight_decay=1e-5)
    opt_state = optimizer.init(eqx.filter(neural_physics, eqx.is_array))

    sigma_full = jnp.asarray(sigma.sigma_full)
    loss_history = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for ic, target, forcing in zip(initial_carries, target_carries, forcings):

            def _loss_fn(nn_phys):
                step_unified = make_neural_step_unified(nn_phys, grid)
                run_seg = build_segment_fn(
                    model=model,
                    step_unified=step_unified,
                    grid=grid,
                    sigma_full=sigma_full,
                    dsigma=jnp.asarray(sigma.dsigma),
                    dt=dt,
                    rad_update_steps=1,
                    microphysics="none",
                    fix_moisture=False,
                    fix_mass=False,
                    fric_decay=jnp.ones(sigma_full.shape[0]),
                    qv_smooth_coeff=0.0,
                    lat=grid.lat,
                    lon=grid.lon,
                    start_day=0.0,
                    gradient_checkpoint=True,
                )
                pred = single_day_rollout(ic, forcing, run_seg.raw, dt=dt)
                return combined_loss(pred, target, sigma_full, config=loss_config)

            loss, grads = eqx.filter_value_and_grad(_loss_fn)(neural_physics)
            epoch_loss += float(loss)

            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array),
                opt_state,
                eqx.filter(neural_physics, eqx.is_array),
            )
            neural_physics = eqx.apply_updates(neural_physics, updates)

        avg_loss = epoch_loss / max(len(initial_carries), 1)
        loss_history.append(avg_loss)

        if epoch % log_every == 0:
            logger.info(f"Epoch {epoch:4d}: loss={avg_loss:.6f}, time={time.time()-t0:.1f}s")

    return neural_physics, loss_history


# ======================================================================
# Mode 3: SFNO coupled to dycore
# ======================================================================

def train_sfno_coupled(
    model,
    grid,
    sigma,
    sfno_physics,
    initial_carries,
    target_carries,
    forcings,
    *,
    n_epochs: int = 100,
    lr: float = 5e-4,
    dt: float = 600.0,
    coupling_mode: str = "correction",
    loss_config: LossConfig = LossConfig(),
    log_every: int = 10,
):
    """Train SFNO coupled to the differentiable dycore.

    Parameters
    ----------
    sfno_physics : SFNOPhysics (eqx.Module)
        SFNO wrapped for physics interface compatibility.
    coupling_mode : str
        "correction" (SFNO adds to physics) or "replacement" (SFNO replaces physics).
    (other params same as train_neural_gcm)

    Returns
    -------
    sfno_physics : updated SFNOPhysics
    list[float] — loss history
    """
    from legoesm.training.sfno_dycore_coupling import make_sfno_step_unified

    optimizer = optax.adamw(lr, weight_decay=1e-5)
    opt_state = optimizer.init(eqx.filter(sfno_physics, eqx.is_array))

    sigma_full = jnp.asarray(sigma.sigma_full)
    loss_history = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for ic, target, forcing in zip(initial_carries, target_carries, forcings):

            def _loss_fn(sfno_ph):
                step_unified = make_sfno_step_unified(
                    sfno_ph, grid, mode=coupling_mode,
                )
                run_seg = build_segment_fn(
                    model=model,
                    step_unified=step_unified,
                    grid=grid,
                    sigma_full=sigma_full,
                    dsigma=jnp.asarray(sigma.dsigma),
                    dt=dt,
                    rad_update_steps=1,
                    microphysics="none",
                    fix_moisture=False,
                    fix_mass=False,
                    fric_decay=jnp.ones(sigma_full.shape[0]),
                    qv_smooth_coeff=0.0,
                    lat=grid.lat,
                    lon=grid.lon,
                    start_day=0.0,
                    gradient_checkpoint=True,
                )
                pred = single_day_rollout(ic, forcing, run_seg.raw, dt=dt)
                return combined_loss(pred, target, sigma_full, config=loss_config)

            loss, grads = eqx.filter_value_and_grad(_loss_fn)(sfno_physics)
            epoch_loss += float(loss)

            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array),
                opt_state,
                eqx.filter(sfno_physics, eqx.is_array),
            )
            sfno_physics = eqx.apply_updates(sfno_physics, updates)

        avg_loss = epoch_loss / max(len(initial_carries), 1)
        loss_history.append(avg_loss)

        if epoch % log_every == 0:
            logger.info(f"Epoch {epoch:4d}: loss={avg_loss:.6f}, time={time.time()-t0:.1f}s")

    return sfno_physics, loss_history
