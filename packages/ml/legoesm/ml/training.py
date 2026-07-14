"""Training loop for SFNO.

Provides a complete training pipeline with:
- AdamW / Adam / MUON optimizer dispatch with warmup + cosine decay schedule
- Gradient clipping
- JIT-compiled training step using equinox
- Checkpoint save/load via equinox serialization

References
----------
- Watt-Meyer et al. (2023). ACE. arXiv:2310.02074.
- Jordan et al. (2024). MUON: Momentum Orthogonalized via Newton-Schulz.
"""

from __future__ import annotations

import os
from typing import NamedTuple
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from legoesm.grids.gaussian import GaussianGrid
from legoesm.ml.loss import area_weighted_mse, weighted_mae


class TrainingConfig(NamedTuple):
    """Configuration for SFNO training.

    Attributes
    ----------
    lr : float
        Peak learning rate.
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total training steps (for cosine decay).
    weight_decay : float
        AdamW weight decay (ignored when optimizer='adam' or 'muon').
    batch_size : int
        Training batch size.
    n_autoregressive_steps : int
        Number of autoregressive rollout steps for training loss.
    grad_clip_norm : float
        Maximum gradient norm for clipping.
    checkpoint_dir : str
        Directory for saving checkpoints.
    checkpoint_every : int
        Save checkpoint every N steps.
    optimizer : str
        Optimizer kind: 'adamw' (default, preserves SFNO behavior),
        'adam' (no weight decay), or 'muon' (Momentum Orthogonalized
        via Newton-Schulz; uses ``optax.contrib.muon``).  Muon
        orthogonalizes matrix-shaped parameters and is a no-op on
        scalar leaves, so it composes cleanly with mixed
        scheme-scalar + neural-weight parameter pytrees.
    """
    lr: float = 5e-4
    warmup_steps: int = 1000
    total_steps: int = 100_000
    weight_decay: float = 1e-5
    batch_size: int = 4
    n_autoregressive_steps: int = 2
    grad_clip_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1000
    optimizer: str = "adamw"


def create_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """Create optimizer with warmup, cosine decay, and gradient clipping.

    Schedule: linear warmup -> cosine decay to 0.

    Optimizer selected by ``config.optimizer``:

    - ``adamw`` (default): legacy SFNO behavior, uses ``config.weight_decay``.
    - ``adam``: plain Adam (weight_decay ignored).
    - ``muon``: Momentum Orthogonalized via Newton-Schulz
      (``optax.contrib.muon``).  Applies the orthogonalized update to
      *every* parameter leaf, which is the historical default but
      destabilizes SFNO training during early steps because the
      Newton-Schulz iteration is poorly conditioned on the
      randomly-initialized decoder weights -- this is the
      ``epoch-1 silent exit`` regression the AIMIP suite hit on 34M-
      parameter SFNO under the 20-day windowed setup.
    - ``muon_partitioned``: MUON applied only to 2-D weight matrices
      whose smaller dimension is at least ``muon_min_dim`` (default
      32); AdamW handles 1-D biases, scalars, and small matrices.
      This is the recommended SFNO-safe deployment pattern from the
      original MUON paper (Jordan et al. 2024 sec. 5) and the one
      legoESM should use when ``aimip_optimizer: muon`` is requested.
      Use this in place of ``muon`` once you trust the SFNO branch
      again.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.

    Returns
    -------
    optax.GradientTransformation
        Composed optimizer.

    Raises
    ------
    ValueError
        If ``config.optimizer`` is not one of the supported names.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps,
        end_value=0.0,
    )

    if config.optimizer == "adamw":
        core = optax.adamw(
            learning_rate=schedule, weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        core = optax.adam(learning_rate=schedule)
    elif config.optimizer == "muon":
        try:
            from optax.contrib import muon
        except ImportError as exc:
            raise ImportError(
                "config.optimizer='muon' requires a version of optax that "
                "ships optax.contrib.muon (>= 0.2.4). Upgrade optax or "
                "select optimizer='adamw'/'adam'."
            ) from exc
        core = muon(learning_rate=schedule)
    elif config.optimizer == "muon_partitioned":
        core = _muon_partitioned_optimizer(
            schedule=schedule,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer {config.optimizer!r}; "
            f"expected one of 'adamw', 'adam', 'muon', 'muon_partitioned'."
        )

    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        core,
    )


# Minimum smaller-axis dimension for a parameter leaf to receive the
# MUON update under ``muon_partitioned``.  Smaller matrices, 1-D bias
# vectors, and scalar physics knobs route to AdamW instead, where the
# Newton-Schulz orthogonalization is either degenerate (1-D leaves) or
# numerically unstable on randomly-initialized very small matrices.
_MUON_MIN_DIM_DEFAULT = 32


def _muon_partitioned_optimizer(
    schedule, weight_decay: float, min_dim: int = _MUON_MIN_DIM_DEFAULT,
) -> optax.GradientTransformation:
    """Build a multi-transform optimizer routing MUON / AdamW per leaf.

    The MUON branch only fires on 2-D weight matrices with both
    dimensions at least ``min_dim``; everything else (biases, scalars,
    embedding tables narrower than ``min_dim``) flows through AdamW.

    This pairing is the SFNO-safe MUON deployment from the original
    Jordan et al. (2024) recipe -- applying Newton-Schulz to small or
    1-D leaves is what broke the AIMIP suite (silent epoch-1 NaN
    under MUON-on-all-leaves).
    """
    from optax.contrib import muon
    import jax

    muon_tx = muon(learning_rate=schedule)
    adam_tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)

    def _label(params):
        def _classify(leaf):
            if not hasattr(leaf, "ndim"):
                return "adam"
            if leaf.ndim == 2 and min(leaf.shape) >= min_dim:
                return "muon"
            return "adam"
        return jax.tree_util.tree_map(_classify, params)

    return optax.multi_transform(
        {"muon": muon_tx, "adam": adam_tx},
        _label,
    )


def make_train_step(optimizer: optax.GradientTransformation, grid: GaussianGrid):
    """Build the JIT-compiled SFNO training step ONCE.

    ``optimizer`` and ``grid`` are captured as closure constants instead of
    being passed as call arguments, because :func:`eqx.filter_jit` with
    ``donate="warn"`` marks EVERY array-leaf argument as donatable.  ``grid``
    and the ``optimizer`` state are REUSED every iteration and never returned,
    so donating them invalidated their buffers and triggered deleted-buffer
    reuse on a donating backend.  Only the per-step ``model`` / ``opt_state``
    (returned) and the consumed ``batch_*`` arrays remain donatable args.

    Returns
    -------
    callable
        ``(model, opt_state, batch_input, batch_target) ->
        (model, opt_state, loss)`` wrapped in ``eqx.filter_jit(donate="warn")``.
    """

    @eqx.filter_jit(donate="warn")
    def train_step(
        model: eqx.Module,
        opt_state: optax.OptState,
        batch_input: jnp.ndarray,
        batch_target: jnp.ndarray,
    ) -> tuple[eqx.Module, optax.OptState, jnp.ndarray]:
        """Single JIT-compiled training step.

        Donates the input ``model`` and ``opt_state`` buffers (``donate="warn"``)
        so XLA can reuse the underlying device memory for the updated values
        instead of holding both copies live until reassignment.  For SFNO with
        typical (embed_dim=256, n_blocks=8) this halves the peak weight +
        optimiser-state memory at every step on Levante.  ``grid`` and
        ``optimizer`` are closed over (NOT donatable args) so their reused
        buffers are never invalidated.
        """
        def loss_fn(model):
            # vmap over batch dimension
            pred = jax.vmap(lambda x: model(x, grid))(batch_input)
            return area_weighted_mse(
                pred, batch_target,
                grid.weights,
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    return train_step


def save_checkpoint(
    model,
    path: str | Path,
) -> None:
    """Save a model / pytree checkpoint **atomically**.

    Serialises ``model`` (any pytree whose array leaves ``eqx`` can
    write — an :class:`eqx.Module`, or e.g. a ``(model, opt_state, ...)``
    tuple) to a temp file in the destination directory and then
    :func:`os.replace`-renames it onto ``path``.  The rename is atomic
    on POSIX, so a walltime kill mid-write can never leave a truncated
    or half-serialised checkpoint at ``path``: readers always see either
    the previous complete checkpoint or the new complete one, never a
    corrupt in-between (the non-atomic-write corruption source called
    out in #942).

    Parameters
    ----------
    model : eqx.Module or pytree
        Model / state to save.
    path : str or Path
        Destination file path for the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        eqx.tree_serialise_leaves(str(tmp), model)
        os.replace(tmp, path)
    finally:
        # ``os.replace`` consumes ``tmp`` on success; this only fires if
        # serialisation raised, cleaning up the partial temp file.
        if tmp.exists():
            tmp.unlink()


def load_checkpoint(
    model_template: eqx.Module,
    path: str | Path,
) -> eqx.Module:
    """Load model checkpoint.

    Parameters
    ----------
    model_template : eqx.Module
        A model with the same structure (used for tree shape).
    path : str or Path
        File path of the checkpoint.

    Returns
    -------
    eqx.Module
        Model with loaded weights.
    """
    return eqx.tree_deserialise_leaves(str(path), model_template)


@eqx.filter_jit
def validate_step(
    model: eqx.Module,
    batch_input: jnp.ndarray,
    batch_target: jnp.ndarray,
    grid: GaussianGrid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled validation step (no gradient).

    Parameters
    ----------
    model : eqx.Module
        Current model.
    batch_input : array, shape (batch_size, n_lat, n_lon, n_channels)
        Input batch.
    batch_target : array, shape (batch_size, n_lat, n_lon, n_channels)
        Target batch.
    grid : GaussianGrid
        Grid for area-weighted loss.

    Returns
    -------
    (mse, mae)
        Area-weighted MSE and MAE on the validation batch.
    """
    pred = jax.vmap(lambda x: model(x, grid))(batch_input)
    w = grid.weights
    mse = area_weighted_mse(pred, batch_target, w)
    mae = weighted_mae(pred, batch_target, w)
    return mse, mae


def train_sfno(
    model: eqx.Module,
    grid: GaussianGrid,
    era5_config,
    training_config: TrainingConfig,
    val_era5_config=None,
    log_every: int = 100,
) -> eqx.Module:
    """Full training loop for SFNO with ERA5 GCS streaming.

    Parameters
    ----------
    model : eqx.Module
        Initial SFNO model.
    grid : GaussianGrid
        Grid for area-weighted loss and model evaluation.
    era5_config : ERA5Config
        Training data configuration.
    training_config : TrainingConfig
        Training hyperparameters.
    val_era5_config : ERA5Config, optional
        Validation data configuration. If None, no validation.
    log_every : int
        Print loss every N steps.

    Returns
    -------
    eqx.Module
        Trained model.
    """
    from legoesm.ml.data.era5_loader import create_training_iterator

    optimizer = create_optimizer(training_config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    # Build the jitted step ONCE; optimizer + grid are closed over (not
    # donatable args) so their reused buffers survive across steps.
    train_step = make_train_step(optimizer, grid)

    train_iter = create_training_iterator(
        era5_config,
        batch_size=training_config.batch_size,
        seed=0,
    )

    # Optional: validation iterator
    val_iter = None
    if val_era5_config is not None:
        val_iter = create_training_iterator(
            val_era5_config,
            batch_size=training_config.batch_size,
            seed=42,
            shuffle=False,
        )

    for step in range(training_config.total_steps):
        batch_input, batch_target = next(train_iter)
        model, opt_state, loss = train_step(
            model, opt_state, batch_input, batch_target,
        )

        if step % log_every == 0:
            print(f"Step {step:6d} | train_loss: {float(loss):.6f}")

        # Periodic validation
        if (
            val_iter is not None
            and step > 0
            and step % training_config.checkpoint_every == 0
        ):
            val_input, val_target = next(val_iter)
            val_mse, val_mae = validate_step(model, val_input, val_target, grid)
            print(
                f"Step {step:6d} | val_mse: {float(val_mse):.6f} "
                f"| val_mae: {float(val_mae):.6f}"
            )

        # Checkpoint
        if (
            step > 0
            and step % training_config.checkpoint_every == 0
        ):
            ckpt_path = Path(training_config.checkpoint_dir) / f"step_{step:06d}.eqx"
            save_checkpoint(model, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = Path(training_config.checkpoint_dir) / "final.eqx"
    save_checkpoint(model, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")

    return model
