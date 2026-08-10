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

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from legoesm.driver.compiled_segments import build_segment_fn
from legoesm.training.losses import combined_loss, LossConfig
from legoesm.training.dycore_rollout import single_day_rollout

logger = logging.getLogger(__name__)


# Default gradient-clip norm and warmup for the dycore training drivers.  A raw
# optax.adam(lr)/adamw(lr) has NO clipping or warmup, so a single large adjoint
# (chaotic dynamics, a bad sample) can blow the parameters up before the loop's
# NaN guard even fires.  Routing through ml.training.create_optimizer adds
# warmup -> cosine decay + global-norm clipping (single source of truth; no
# re-implemented schedule here).
_DRIVER_GRAD_CLIP_NORM = 1.0
_DRIVER_WARMUP_STEPS = 100


def _make_driver_optimizer(
    lr: float,
    optimizer_kind: str,
    n_epochs: int,
    n_samples: int,
    *,
    weight_decay: float = 0.0,
    grad_clip_norm: float = _DRIVER_GRAD_CLIP_NORM,
    warmup_steps: int = _DRIVER_WARMUP_STEPS,
):
    """Build a warmup+cosine+clip optimizer via ``ml.training.create_optimizer``.

    The dycore training modes only expose a peak ``lr``; this derives the total
    step count (``n_epochs * n_samples``) for the cosine schedule and clamps the
    warmup to it, then defers to the shared optimizer factory so clipping and the
    schedule are NOT re-implemented per mode.
    """
    from legoesm.ml.training import TrainingConfig, create_optimizer

    total_steps = max(int(n_epochs) * max(int(n_samples), 1), 1)
    # A linear warmup makes the LR exactly 0 on every step < warmup (step 0
    # always included for warmup >= 1), so a SHORT run (a 1-step smoke or a
    # few-step fine-tune) whose whole length is <= the warmup would return
    # init_value (lr == 0) and silently skip the update the raw optax.adam(lr)
    # path DID perform.  Disable warmup entirely when the run is no longer than
    # the requested warmup (warmup == 0 -> full peak LR from step 0, pure cosine
    # decay); otherwise keep the full warmup for a normal-length run.
    warmup = int(warmup_steps) if total_steps > int(warmup_steps) else 0
    cfg = TrainingConfig(
        lr=lr,
        warmup_steps=warmup,
        total_steps=total_steps,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        optimizer=optimizer_kind,
    )
    return create_optimizer(cfg)


# ======================================================================
# Shared helpers (avoid copy-paste across modes)
# ======================================================================

def build_training_segment(model, step_unified, grid, sigma, dt,
                            fric_decay=None, **extra_kwargs):
    """Build a segment function with standard training defaults.

    Encapsulates the boilerplate kwargs shared by all training modes.
    Returns the compiled segment function (use ``.raw`` for AD).

    ``fric_decay``: per-level Rayleigh-friction decay factors (the driver's
    ``exp(-k_f dt)`` boundary-layer profile). Pass the DRIVER's profile for
    rollouts whose physics provides no dissipation of its own (pure-dycore
    epoch-0 neural_gcm / sfno): without it the forward 6 h rollout stays
    finite but the 720-step ADJOINT through the undamped dycore returns NaN
    gradients (#797 bug 11). ``None`` keeps the legacy no-friction ones.
    """
    sigma_full = jnp.asarray(sigma.sigma_full)
    if fric_decay is None:
        fric_decay = jnp.ones(sigma_full.shape[0])
    # DEFAULTS, not hardcodes. These four used to be passed positionally-by-name
    # alongside ``**extra_kwargs``, so ANY caller supplying one of them —
    # ``microphysics`` is the common case — died with
    #   TypeError: build_segment_fn() got multiple values for keyword argument
    # which is how the classical lat-lon AIMIP variant kept failing. Worse than
    # the crash is what the hardcodes meant when nobody passed them: every
    # carry-based training rollout ran with NO microphysics and NO mass fixer
    # regardless of the experiment's configuration, so the tuned model was not
    # the configured model. Caller values now win.
    seg_defaults = dict(
        rad_update_steps=1,
        microphysics="none",
        fix_moisture=False,
        fix_mass=False,
    )
    seg_defaults.update(extra_kwargs)
    return build_segment_fn(
        model=model,
        step_unified=step_unified,
        grid=grid,
        sigma_full=sigma_full,
        dsigma=jnp.asarray(sigma.dsigma),
        dt=dt,
        fric_decay=jnp.asarray(fric_decay),
        # Kept 0.0 on every grid: finalize_split_step's q_v smoothing calls the
        # cube ``hyperdiffusion_3d`` (reads cube-only ``grid.halo_interp_offsets``)
        # regardless of grid, so a nonzero coeff would crash on lat-lon.  The
        # coeff=0 static gate skips it; wiring a lat-lon ∇⁴ operator here is the
        # follow-up needed before training can smooth q_v on lat-lon.
        qv_smooth_coeff=0.0,
        # Radiation flattens lat/lon to columns (ncol = n_lat*n_lon), so it needs
        # the 2D grid field.  Lat-lon grids store a 1D lat/lon vector plus a 2D
        # ``lat2d``/``lon2d``; the cube's ``lat``/``lon`` are already 2D (6,n,n)
        # and have no ``lat2d``.  Mirror the production driver (grid.lat2d).
        lat=getattr(grid, "lat2d", grid.lat),
        lon=getattr(grid, "lon2d", grid.lon),
        start_day=0.0,
        gradient_checkpoint=True,
        **seg_defaults,
    )


def _build_train_step(make_run_seg, optimizer, sigma_full, grid, dt, loss_config,
                      rollout_hours: float = 24.0):
    """Build the ONCE-jitted differentiable train step shared by all modes.

    This is the core of the OOM/recompile fix.  The previous driver built
    ``run_seg = build_segment_fn(...)`` -- and therefore the whole closure
    tree -- inside ``loss_fn`` on EVERY ``eqx.filter_value_and_grad`` call,
    with no ``filter_jit`` on the step.  The full windowed rollout + reverse
    adjoint was thus re-traced eagerly every sample*epoch (the documented
    ~1.5 GiB/sample leak).  Wrapping the whole step in :func:`eqx.filter_jit`
    traces the build/rollout/adjoint exactly ONCE and caches the XLA graph;
    subsequent samples are pure forward+backward+update calls.

    Parameters
    ----------
    make_run_seg : callable
        ``(trainable) -> compiled segment fn`` (use ``.raw`` for AD).
        Called INSIDE the differentiated ``loss_fn`` with the TRACED
        trainable leaf so (a) gradients flow into the trainable and
        (b) the segment build is captured by the single surrounding
        ``filter_jit`` trace rather than rebuilt eagerly per sample.
        Pre-building it OUTSIDE the loss with concrete params would bake
        the params in as constants and zero their gradients.  The three
        modes differ ONLY in this factory: physics-param tuning feeds
        ``params.to_segment_kwargs()``; neural-GCM / SFNO build their
        ``step_unified`` from the (traced) network.
    optimizer : optax.GradientTransformation
        Captured as a closure constant (static under ``filter_jit``); its
        ``update`` runs inside the jitted step.
    sigma_full, grid, dt, loss_config
        Static rollout/loss configuration captured in the closure.

    Returns
    -------
    callable
        ``eqx.filter_jit``-wrapped
        ``(params, opt_state, ic, target, forcing) ->
        (params, opt_state, loss, grad_norm)``.  ``ic``/``target``/
        ``forcing`` are TRACED ARGS (SegmentForcing doctrine), never
        closure-captured, so changing per-sample values neither bake in
        nor force recompiles.
    """
    def _train_step(params, opt_state, ic, target, forcing):
        def loss_fn(trainable):
            run_seg = make_run_seg(trainable)
            # ``.raw`` = the non-JIT, non-donating segment variant.  Buffer
            # donation conflicts with reverse-mode AD, so it MUST stay
            # inside ``filter_value_and_grad`` (do not swap for a donating
            # variant).
            #
            # Route through the SHARED rollout+loss instead of an inline
            # ``single_day_rollout`` + ``combined_loss``. Two reasons:
            #
            # * the horizon MUST match the lead the targets were loaded at —
            #   ``single_day_rollout`` defaults to 24 h, and the lat-lon driver
            #   loads 6 h targets, a mismatch that trains without error and
            #   forecasts badly;
            # * ``multi_step_rollout_loss`` is documented as "shared by every
            #   AIMIP trainer so the rollout+loss is defined ONCE", but nothing
            #   in production called it — its only caller was its own unit test.
            #   So ``loss_config.multi_step_hours`` was SILENTLY IGNORED: the
            #   lat-lon driver built tuple-of-lead targets and passed them to a
            #   loss that only ever did one rollout.
            return multi_step_rollout_loss(
                ic, forcing, run_seg.raw, dt=dt,
                rollout_hours=rollout_hours, target=target,
                sigma_full=sigma_full, grid=grid, loss_config=loss_config,
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(params, eqx.is_array),
        )
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss, grad_norm

    return eqx.filter_jit(_train_step)


def _training_loop(
    train_step: Callable,
    params,
    optimizer,
    initial_carries,
    target_carries,
    forcings,
    *,
    n_epochs: int = 100,
    log_every: int = 10,
    log_params: bool = False,
):
    """Generic training loop shared by all modes.

    Parameters
    ----------
    train_step : callable
        ``(params, opt_state, ic, target, forcing) ->
        (params, opt_state, loss, grad_norm)``.  Built ONCE per training
        call by :func:`_build_train_step` and wrapped in ``eqx.filter_jit``
        so the segment fn / rollout / adjoint is traced a single time
        instead of eagerly per sample*epoch.
    params : eqx.Module
        Initial learnable parameters.
    optimizer : optax.GradientTransformation
        Used here only to initialise ``opt_state``; the per-step update
        lives inside ``train_step``.
    initial_carries, target_carries, forcings : lists of training data
    n_epochs, log_every : training config
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
        grad_norm_val = 0.0

        for sample_idx, (ic, target, forcing) in enumerate(
            zip(initial_carries, target_carries, forcings)
        ):
            params, opt_state, loss, grad_norm = train_step(
                params, opt_state, ic, target, forcing
            )

            # --- NaN / Inf detection (outside JIT, values are materialized) ---
            loss_val = float(loss)
            if jnp.isnan(loss) or jnp.isinf(loss):
                raise RuntimeError(
                    f"NaN/Inf loss detected at epoch {epoch}, sample {sample_idx} "
                    f"(loss={loss_val}). "
                    "Check CFL conditions, parameter bounds, and input data."
                )
            grad_norm_val = float(grad_norm)
            if jnp.isnan(grad_norm) or jnp.isinf(grad_norm):
                raise RuntimeError(
                    f"NaN/Inf gradient detected at epoch {epoch}, sample "
                    f"{sample_idx} (grad_norm={grad_norm_val}). "
                    "Consider reducing learning rate or adding gradient clipping."
                )

            epoch_loss += loss_val

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

def multi_step_rollout_loss(
    ic, forcing, run_seg_raw, *, dt, rollout_hours, target,
    sigma_full, grid, loss_config, truncated_bptt=True,
):
    """Single- or multi-step autoregressive supervision loss (shared by
    every AIMIP trainer so the rollout+loss is defined ONCE).

    ``loss_config.multi_step_hours`` empty  ->  ONE ``single_day_rollout``
    to ``rollout_hours`` vs ``target`` (a single SegmentCarry): the legacy
    single-horizon path, bit-identical to the old inline trainer body.

    Non-empty  ->  GenCast-style multi-step (NeuralGCM / AIMIP protocol):
    chain autoregressive segments between consecutive leads and sum
    ``combined_loss`` after each segment against ``target[k]`` (``target``
    is then a tuple of K carries), weighted by
    ``loss_config.multi_step_weights`` (default uniform, normalised by the
    weight sum).

    ``truncated_bptt`` (default True) ``stop_gradient``s the carry between
    segments, so each loss term backprops ONLY through its own segment.
    The lat-lon primitive-equation ADJOINT explodes to NaN through a
    >~6 h differentiable chain even when the forward is finite (job
    8533825) — so full backprop-through-time over a 24 h+ multi-step
    rollout is unusable on this stack.  Truncated BPTT keeps every
    gradient a stable short-horizon adjoint while the FORWARD still chains
    autoregressively: the model is supervised on its OWN drifted
    6/12/18 h states, not only the ERA5 IC, which is the point of
    multi-step training.  The spectral stack uses full BPTT (it is
    adjoint-stable); this flag exists for the lat-lon stack.

    NB the per-segment ``forcing`` is reused as-is; the diurnal solar
    phase is therefore held at the IC time-of-day across segments.
    # ponytail: approximate diurnal phase across segments; thread
    # forcing.seconds_of_day per lead if a diurnal-sensitive metric needs it.
    """
    ms_hours = tuple(int(h) for h in (loss_config.multi_step_hours or ()))
    if not ms_hours:
        pred = single_day_rollout(
            ic, forcing, run_seg_raw, dt=dt, hours=rollout_hours)
        return combined_loss(
            pred, target, sigma_full, grid=grid, config=loss_config)

    # Segment lengths = gaps between consecutive (sorted) leads.
    leads = sorted(ms_hours)
    seg_hours, prev = [], 0
    for h in leads:
        gap = h - prev
        if gap <= 0:
            raise ValueError(
                f"multi_step_hours={ms_hours} must be strictly increasing "
                "positive hour leads.")
        seg_hours.append(gap)
        prev = h
    if not isinstance(target, (tuple, list)) or len(target) != len(seg_hours):
        raise ValueError(
            f"multi-step loss needs {len(seg_hours)} target carries (one per "
            f"lead {leads}); got "
            f"{type(target).__name__} of length "
            f"{len(target) if isinstance(target, (tuple, list)) else 'n/a'}.")
    weights = tuple(float(w) for w in (loss_config.multi_step_weights or ()))
    if weights and len(weights) != len(seg_hours):
        raise ValueError(
            f"multi_step_weights length {len(weights)} != number of leads "
            f"{len(seg_hours)}.")
    if not weights:
        weights = (1.0,) * len(seg_hours)
    wsum = float(sum(weights))

    state = ic
    total = jnp.asarray(0.0)
    for k, sh in enumerate(seg_hours):
        state = single_day_rollout(
            state, forcing, run_seg_raw, dt=dt, hours=sh)
        total = total + weights[k] * combined_loss(
            state, target[k], sigma_full, grid=grid, config=loss_config)
        if truncated_bptt:
            state = jax.lax.stop_gradient(state)
    return total / wsum


def train_physics_params(
    model,
    grid,
    sigma,
    physics_pipeline,
    initial_carries,
    target_carries,
    forcings,
    *,
    rollout_hours: float = 24.0,
    n_epochs: int = 100,
    lr: float = 1e-3,
    dt: float = 600.0,
    grad_clip_norm: float = _DRIVER_GRAD_CLIP_NORM,
    warmup_steps: int = _DRIVER_WARMUP_STEPS,
    loss_config: LossConfig = LossConfig(),
    log_every: int = 10,
    rad_stop_gradient: bool = False,
    **segment_kwargs,
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
    rollout_hours : float
        Supervision horizon [h]. MUST equal the lead the ``target_carries``
        were loaded at — ``single_day_rollout`` defaults to 24 h, so leaving
        it implicit silently scores a 24 h forecast against a 6 h target.
    rad_stop_gradient : bool
        Treat radiation as a forcing (no gradient through it).
    **segment_kwargs
        Physics configuration of the rollout, forwarded verbatim to
        ``build_training_segment`` (``microphysics``, ``rad_update_steps``,
        ...). Not decoration: a rollout built without the caller's settings
        is a different model from the one being tuned.

    Returns
    -------
    TrainablePhysicsParams — optimized parameters
    list[float] — loss history
    """
    from legoesm.training.trainable_params import TrainablePhysicsParams

    params = TrainablePhysicsParams.from_defaults()
    step_unified = physics_pipeline.build_step_unified(
        rad_stop_gradient=rad_stop_gradient)
    sigma_full = jnp.asarray(sigma.sigma_full)

    def make_run_seg(trainable):
        # ``step_unified`` is param-independent (built once above); only the
        # segment kwargs carry the (traced) trainable values, so the gradient
        # path to ``trainable`` runs through ``build_segment_fn``.
        # MERGE, do not double-splat: ``to_segment_kwargs()`` and the caller's
        # ``segment_kwargs`` overlap (both carry ``microphysics``), and two
        # ``**`` of the same key is a TypeError at the call — which is how the
        # classical lat-lon variant died even after the signature was restored.
        # The TRAINED value wins: it is the quantity being optimised, and
        # letting the caller's static default override it would silently zero
        # that parameter's gradient contribution to the rollout.
        seg_kw = {**segment_kwargs, **trainable.to_segment_kwargs()}
        return build_training_segment(
            model, step_unified, grid, sigma, dt, **seg_kw,
        )

    optimizer = _make_driver_optimizer(
        lr, "adam", n_epochs, len(initial_carries),
        grad_clip_norm=grad_clip_norm, warmup_steps=warmup_steps,
    )
    train_step = _build_train_step(
        make_run_seg, optimizer, sigma_full, grid, dt, loss_config,
        rollout_hours=rollout_hours,
    )
    return _training_loop(
        train_step, params, optimizer,
        initial_carries, target_carries, forcings,
        n_epochs=n_epochs, log_every=log_every, log_params=True,
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
    rollout_hours: float = 24.0,
    n_epochs: int = 100,
    lr: float = 1e-4,
    dt: float = 600.0,
    weight_decay: float = 1e-5,
    grad_clip_norm: float = _DRIVER_GRAD_CLIP_NORM,
    warmup_steps: int = _DRIVER_WARMUP_STEPS,
    loss_config: LossConfig = LossConfig(),
    log_every: int = 10,
    **segment_kwargs,
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

    def make_run_seg(nn_phys):
        step_unified = make_neural_step_unified(nn_phys, adapter)
        return build_training_segment(
            model, step_unified, grid, sigma, dt, **segment_kwargs)

    optimizer = _make_driver_optimizer(
        lr, "adamw", n_epochs, len(initial_carries),
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm, warmup_steps=warmup_steps,
    )
    train_step = _build_train_step(
        make_run_seg, optimizer, sigma_full, grid, dt, loss_config,
        rollout_hours=rollout_hours,
    )
    return _training_loop(
        train_step, neural_physics, optimizer,
        initial_carries, target_carries, forcings,
        n_epochs=n_epochs, log_every=log_every,
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
    physics_pipeline=None,
    weight_decay: float = 1e-5,
    grad_clip_norm: float = _DRIVER_GRAD_CLIP_NORM,
    warmup_steps: int = _DRIVER_WARMUP_STEPS,
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
    from legoesm.training.sfno_dycore_coupling import make_sfno_step_unified

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

    def make_run_seg(sfno_ph):
        step_unified = make_sfno_step_unified(
            sfno_ph,
            mode=coupling_mode,
            traditional_step_unified=traditional_step,
        )
        return build_training_segment(model, step_unified, grid, sigma, dt)

    optimizer = _make_driver_optimizer(
        lr, "adamw", n_epochs, len(initial_carries),
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm, warmup_steps=warmup_steps,
    )
    train_step = _build_train_step(
        make_run_seg, optimizer, sigma_full, grid, dt, loss_config,
    )
    return _training_loop(
        train_step, sfno_physics, optimizer,
        initial_carries, target_carries, forcings,
        n_epochs=n_epochs, log_every=log_every,
    )
