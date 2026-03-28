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
from typing import Callable

import jax.numpy as jnp
import equinox as eqx
import optax

from legoesm.driver.compiled_segments import build_segment_fn
from legoesm.training.losses import combined_loss, LossConfig
from legoesm.training.dycore_rollout import single_day_rollout

logger = logging.getLogger(__name__)


# ======================================================================
# Shared helpers (avoid copy-paste across modes)
# ======================================================================

def _build_training_segment(model, step_unified, grid, sigma, dt, **extra_kwargs):
    """Build a segment function with standard training defaults.

    Encapsulates the boilerplate kwargs shared by all training modes.
    Returns the compiled segment function (use ``.raw`` for AD).
    """
    sigma_full = jnp.asarray(sigma.sigma_full)
    return build_segment_fn(
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
        **extra_kwargs,
    )


def _training_loop(
    make_loss_fn: Callable,
    params,
    optimizer,
    initial_carries,
    target_carries,
    forcings,
    sigma_full,
    *,
    n_epochs: int = 100,
    loss_config: LossConfig = LossConfig(),
    log_every: int = 10,
    log_params: bool = False,
):
    """Generic training loop shared by all modes.

    Parameters
    ----------
    make_loss_fn : callable
        ``(params, ic, target, forcing) -> scalar_loss``
        Factory that creates the differentiable loss for one sample.
    params : eqx.Module
        Initial learnable parameters.
    optimizer : optax.GradientTransformation
    initial_carries, target_carries, forcings : lists of training data
    sigma_full : jax.Array
    n_epochs, loss_config, log_every : training config
    log_params : bool
        If True, log parameter values each epoch (for physics param tuning).

    Returns
    -------
    params : updated parameters
    loss_history : list[float]
    """
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))
    loss_history = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for sample_idx, (ic, target, forcing) in enumerate(
            zip(initial_carries, target_carries, forcings)
        ):
            loss_fn = make_loss_fn(params, ic, target, forcing)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(params)

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
                eqx.filter(params, eqx.is_array),
            )
            params = eqx.apply_updates(params, updates)

        avg_loss = epoch_loss / max(len(initial_carries), 1)
        loss_history.append(avg_loss)

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            elapsed = time.time() - t0
            # Compute gradient norm for the last sample of the epoch
            msg = (
                f"Epoch {epoch:4d}: loss={avg_loss:.6f}, "
                f"grad_norm={grad_norm_val:.6e}, time={elapsed:.1f}s"
            )
            if log_params and hasattr(params, 'as_dict'):
                phys = params.as_dict()
                param_str = ", ".join(f"{k}={float(v):.4f}" for k, v in phys.items())
                msg += f", params=[{param_str}]"
            logger.info(msg)

    return params, loss_history


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
    n_epochs, lr, dt, loss_config, log_every : training config

    Returns
    -------
    TrainablePhysicsParams — optimized parameters
    list[float] — loss history
    """
    from legoesm.training.trainable_params import TrainablePhysicsParams

    params = TrainablePhysicsParams.from_defaults()
    step_unified = physics_pipeline.build_step_unified()
    sigma_full = jnp.asarray(sigma.sigma_full)

    def make_loss_fn(_params, ic, target, forcing):
        def loss_fn(params_):
            seg_kw = params_.to_segment_kwargs()
            run_seg = _build_training_segment(
                model, step_unified, grid, sigma, dt, **seg_kw,
            )
            pred = single_day_rollout(ic, forcing, run_seg.raw, dt=dt)
            return combined_loss(pred, target, sigma_full, config=loss_config)
        return loss_fn

    return _training_loop(
        make_loss_fn, params, optax.adam(lr),
        initial_carries, target_carries, forcings, sigma_full,
        n_epochs=n_epochs, loss_config=loss_config,
        log_every=log_every, log_params=True,
    )


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

    sigma_full = jnp.asarray(sigma.sigma_full)

    def make_loss_fn(_params, ic, target, forcing):
        def loss_fn(nn_phys):
            step_unified = make_neural_step_unified(nn_phys, grid)
            run_seg = _build_training_segment(
                model, step_unified, grid, sigma, dt,
            )
            pred = single_day_rollout(ic, forcing, run_seg.raw, dt=dt)
            return combined_loss(pred, target, sigma_full, config=loss_config)
        return loss_fn

    return _training_loop(
        make_loss_fn, neural_physics, optax.adamw(lr, weight_decay=1e-5),
        initial_carries, target_carries, forcings, sigma_full,
        n_epochs=n_epochs, loss_config=loss_config, log_every=log_every,
    )


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
    coupling_mode : "correction" or "replacement"
    (other params same as train_neural_gcm)

    Returns
    -------
    sfno_physics : updated SFNOPhysics
    list[float] — loss history
    """
    from legoesm.training.sfno_dycore_coupling import make_sfno_step_unified

    sigma_full = jnp.asarray(sigma.sigma_full)

    def make_loss_fn(_params, ic, target, forcing):
        def loss_fn(sfno_ph):
            step_unified = make_sfno_step_unified(
                sfno_ph, grid, mode=coupling_mode,
            )
            run_seg = _build_training_segment(
                model, step_unified, grid, sigma, dt,
            )
            pred = single_day_rollout(ic, forcing, run_seg.raw, dt=dt)
            return combined_loss(pred, target, sigma_full, config=loss_config)
        return loss_fn

    return _training_loop(
        make_loss_fn, sfno_physics, optax.adamw(lr, weight_decay=1e-5),
        initial_carries, target_carries, forcings, sigma_full,
        n_epochs=n_epochs, loss_config=loss_config, log_every=log_every,
    )
