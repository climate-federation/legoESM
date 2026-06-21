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

def _build_training_segment(model, step_unified, grid, sigma, dt, *,
                            microphysics="none", rad_update_steps=1,
                            **extra_kwargs):
    """Build a segment function with standard training defaults.

    Encapsulates the boilerplate kwargs shared by all training modes.
    Returns the compiled segment function (use ``.raw`` for AD).

    ``microphysics`` selects the prognostic-condensate scheme applied in the
    carry hot loop (``"none"`` keeps the saturation-adjustment closure).  The
    classical physics variant threads its real scheme (e.g. ``"kessler"``)
    here so q_c/q_r/precip evolve; NN-replacement variants keep ``"none"``
    (the network subsumes condensation).

    ``rad_update_steps`` sub-cycles radiation: rrtmgp runs every N dynamics
    steps (1 = every step).  N>1 cuts the dominant rrtmgp cost of the
    classical variant — radiation varies slowly, so hourly-ish updates are
    standard GCM practice and keep ``held_*`` fresh enough for the flux loss
    (the last update lands within N steps of the rollout end).  Moot for the
    NN-replacement variants (their step_unified ignores ``need_rad``).
    """
    sigma_full = jnp.asarray(sigma.sigma_full)
    return build_segment_fn(
        model=model,
        step_unified=step_unified,
        grid=grid,
        sigma_full=sigma_full,
        dsigma=jnp.asarray(sigma.dsigma),
        dt=dt,
        rad_update_steps=rad_update_steps,
        microphysics=microphysics,
        fix_moisture=False,
        fix_mass=False,
        fric_decay=jnp.ones(sigma_full.shape[0]),
        qv_smooth_coeff=0.0,
        # 2D horizontal coords (GridProtocol ``grid_lat``/``grid_lon``):
        # radiation's compute_radiation_core flattens lat to (ncol,), so it
        # needs the 2D (n_lat, n_lon) field, not the 1D ``grid.lat`` (which
        # on a lat-lon / Gaussian grid is just the n_lat latitudes).  Equal
        # to ``grid.lat`` on cubed-sphere (already 2D), so this is safe for
        # every grid the training driver targets.
        lat=grid.grid_lat,
        lon=grid.grid_lon,
        start_day=0.0,
        gradient_checkpoint=True,
        **extra_kwargs,
    )


# Public alias: the AIMIP lat-lon orchestrator reuses this exact segment
# build for held-out evaluation (same defaults as training) rather than
# importing the private symbol or re-deriving the kwargs.
build_training_segment = _build_training_segment


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
    # Optimize INEXACT (float/complex) arrays only — exactly what
    # eqx.filter_value_and_grad differentiates.  is_array would also pull in
    # any INT arrays (e.g. an SFNO's non-static Gaussian grid index arrays
    # ms/ls), which then get None grad → optax tree-structure mismatch
    # ("Expected None, got Array").
    opt_state = optimizer.init(eqx.filter(params, eqx.is_inexact_array))
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
                eqx.filter(grads, eqx.is_inexact_array),
                opt_state,
                eqx.filter(params, eqx.is_inexact_array),
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
    rollout_hours: float = 24.0,
    grad_clip: float = 1.0,
    microphysics: str = "none",
    rad_update_steps: int = 1,
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

    # Single source of truth: the segment's saturation-adjustment toggle
    # (do_sat_adjust = microphysics=='none') MUST match whether the pipeline
    # actually produces condensate tendencies.  micro_fn is None iff the
    # pipeline microphysics is 'none' (_resolve_microphysics).  A mismatch
    # (e.g. microphysics='kessler' but a 'none' pipeline) would disable
    # sat-adjust while step_unified returns zero dq_c/dq_r -> supersaturation
    # accumulates unchecked.  Fail loud rather than silently blow up.
    _pipeline_has_micro = getattr(physics_pipeline, "micro_fn", None) is not None
    if _pipeline_has_micro != (microphysics != "none"):
        raise ValueError(
            f"microphysics={microphysics!r} disagrees with the physics_pipeline "
            f"(micro_fn is {'set' if _pipeline_has_micro else 'None'}). Build the "
            "pipeline and pass the segment microphysics from the SAME config so "
            "the saturation-adjustment toggle matches the condensate scheme."
        )

    params = TrainablePhysicsParams.from_defaults()
    step_unified = physics_pipeline.build_step_unified()
    sigma_full = jnp.asarray(sigma.sigma_full)

    def make_loss_fn(_params, ic, target, forcing):
        def loss_fn(params_):
            seg_kw = params_.to_segment_kwargs()
            run_seg = _build_training_segment(
                model, step_unified, grid, sigma, dt,
                microphysics=microphysics, rad_update_steps=rad_update_steps,
                **seg_kw,
            )
            pred = single_day_rollout(
                ic, forcing, run_seg.raw, dt=dt, hours=rollout_hours)
            return combined_loss(pred, target, sigma_full, grid=grid, config=loss_config)
        return loss_fn

    # Gradient clipping guards against the occasional adjoint spike from
    # the long differentiable rollout (esp. with the shorter horizons).
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    return _training_loop(
        make_loss_fn, params, optimizer,
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
    rollout_hours: float = 24.0,
    grad_clip: float = 1.0,
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
    from legoesm.atmosphere.physics.neural_physics import make_neural_step_unified
    from legoesm.core.grid_adapters import make_adapter

    sigma_full = jnp.asarray(sigma.sigma_full)
    adapter = make_adapter(grid)

    def make_loss_fn(_params, ic, target, forcing):
        def loss_fn(nn_phys):
            step_unified = make_neural_step_unified(nn_phys, adapter)
            run_seg = _build_training_segment(
                model, step_unified, grid, sigma, dt,
            )
            pred = single_day_rollout(
                ic, forcing, run_seg.raw, dt=dt, hours=rollout_hours)
            return combined_loss(pred, target, sigma_full, grid=grid, config=loss_config)
        return loss_fn

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(lr, weight_decay=1e-5))
    return _training_loop(
        make_loss_fn, neural_physics, optimizer,
        initial_carries, target_carries, forcings, sigma_full,
        n_epochs=n_epochs, loss_config=loss_config, log_every=log_every,
    )


def train_sfno_latlon(
    model,
    grid,
    sigma,
    sfno,
    gauss_grid,
    nlev,
    regrid_ll2g,
    regrid_g2ll,
    gauss_n_lat,
    gauss_n_lon,
    initial_carries,
    target_carries,
    forcings,
    *,
    n_epochs: int = 100,
    lr: float = 5e-4,
    dt: float = 600.0,
    rollout_hours: float = 24.0,
    grad_clip: float = 1.0,
    tendency_scale: float = 1.0e-5,
    loss_config: LossConfig = LossConfig(),
    log_every: int = 10,
):
    """Train an SFNO on a LAT-LON carry via the Gaussian-regrid bridge.

    The SFNO lives on a Gaussian grid (its SHT requires it); the bridge
    (``make_sfno_step_unified_latlon``) regrids the lat-lon prognostics to
    Gaussian, runs the SFNO, and regrids the tendencies back — all
    differentiable.  Replacement mode (SFNO is the physics).
    """
    from legoesm.training.sfno_dycore_coupling import (
        make_sfno_step_unified_latlon, SFNOPhysics,
    )

    sigma_full = jnp.asarray(sigma.sigma_full)

    # Differentiate ONLY the SFNO (all float/complex params).  The Gaussian
    # grid (with INT spherical-harmonic index arrays ms/ls) is a closure
    # const — bundling it into the differentiated module makes eqx grad emit
    # an Array cotangent for the int leaves where it expects None
    # ("Expected None, got Array").
    def make_loss_fn(_params, ic, target, forcing):
        def loss_fn(sfno_):
            sphys = SFNOPhysics(sfno=sfno_, grid=gauss_grid, nlev=nlev)
            step_unified = make_sfno_step_unified_latlon(
                sphys, regrid_ll2g, regrid_g2ll, gauss_n_lat, gauss_n_lon,
                tendency_scale=tendency_scale,
            )
            run_seg = _build_training_segment(model, step_unified, grid, sigma, dt)
            pred = single_day_rollout(
                ic, forcing, run_seg.raw, dt=dt, hours=rollout_hours)
            return combined_loss(pred, target, sigma_full, grid=grid, config=loss_config)
        return loss_fn

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(lr, weight_decay=1e-5))
    return _training_loop(
        make_loss_fn, sfno, optimizer,
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
    rollout_hours: float = 24.0,
    grad_clip: float = 1.0,
    coupling_mode: str = "correction",
    physics_pipeline=None,
    loss_config: LossConfig = LossConfig(),
    log_every: int = 10,
):
    """Train SFNO coupled to the differentiable dycore.

    Parameters
    ----------
    sfno_physics : SFNOPhysics (eqx.Module)
    coupling_mode : "correction" or "replacement"
    physics_pipeline : PhysicsPipeline, optional
        Required when coupling_mode="correction" to provide the traditional
        physics step that SFNO corrects.
    (other params same as train_neural_gcm)

    Returns
    -------
    sfno_physics : updated SFNOPhysics
    list[float] — loss history
    """
    from legoesm.training.sfno_dycore_coupling import (
        make_sfno_step_unified, SFNOPhysics,
    )

    sigma_full = jnp.asarray(sigma.sigma_full)

    # Build traditional physics step for correction mode.
    traditional_step = None
    if coupling_mode == "correction":
        if physics_pipeline is None:
            raise ValueError(
                "physics_pipeline is required when coupling_mode='correction'. "
                "Pass the PhysicsPipeline so the SFNO can compute corrections "
                "on top of traditional physics."
            )
        traditional_step = physics_pipeline.build_step_unified()

    # Differentiate ONLY the SFNO (float/complex params).  SFNOPhysics.grid is
    # a non-static GaussianGrid whose FLOAT leaves (Pnm/Hnm/wPnm/lap — the SHT
    # transform matrices) would otherwise be swept up by is_inexact_array and
    # corrupted by AdamW (grad + weight decay) — and its INT index arrays
    # ms/ls would break optax's tree structure.  The grid + nlev are closure
    # consts; only sfno_physics.sfno is trained.
    sfno0 = sfno_physics.sfno
    gauss_grid = sfno_physics.grid
    nlev = sfno_physics.nlev

    def make_loss_fn(_params, ic, target, forcing):
        def loss_fn(sfno_):
            sfno_ph = SFNOPhysics(sfno=sfno_, grid=gauss_grid, nlev=nlev)
            step_unified = make_sfno_step_unified(
                sfno_ph,
                mode=coupling_mode,
                traditional_step_unified=traditional_step,
            )
            run_seg = _build_training_segment(
                model, step_unified, grid, sigma, dt,
            )
            pred = single_day_rollout(
                ic, forcing, run_seg.raw, dt=dt, hours=rollout_hours)
            return combined_loss(pred, target, sigma_full, grid=grid, config=loss_config)
        return loss_fn

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(lr, weight_decay=1e-5))
    trained_sfno, hist = _training_loop(
        make_loss_fn, sfno0, optimizer,
        initial_carries, target_carries, forcings, sigma_full,
        n_epochs=n_epochs, loss_config=loss_config, log_every=log_every,
    )
    # Re-wrap the trained SFNO so the caller gets a usable SFNOPhysics back
    # (preserves the previous return contract).
    return SFNOPhysics(sfno=trained_sfno, grid=gauss_grid, nlev=nlev), hist
