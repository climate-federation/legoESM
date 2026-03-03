"""Training loop for SFNO.

Provides a complete training pipeline with:
- AdamW optimizer with warmup + cosine decay schedule
- Gradient clipping
- JIT-compiled training step using equinox
- Checkpoint save/load via equinox serialization

References
----------
- Watt-Meyer et al. (2023). ACE. arXiv:2310.02074.
"""

from __future__ import annotations

from typing import NamedTuple
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from legoesm.grids.gaussian import GaussianGrid
from legoesm.ml.loss import area_weighted_mse, autoregressive_loss


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
        AdamW weight decay.
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


def create_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """Create optimizer with warmup, cosine decay, and gradient clipping.

    Schedule: linear warmup → cosine decay to 0.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.

    Returns
    -------
    optax.GradientTransformation
        Composed optimizer.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps,
        end_value=0.0,
    )

    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )


@eqx.filter_jit
def train_step(
    model: eqx.Module,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch_input: jnp.ndarray,
    batch_target: jnp.ndarray,
    grid: GaussianGrid,
) -> tuple[eqx.Module, optax.OptState, jnp.ndarray]:
    """Single JIT-compiled training step.

    Parameters
    ----------
    model : eqx.Module (SFNO)
        Current model.
    opt_state : optax.OptState
        Current optimizer state.
    optimizer : optax.GradientTransformation
        Optimizer.
    batch_input : array, shape (batch_size, n_lat, n_lon, n_channels)
        Input batch.
    batch_target : array, shape (batch_size, n_lat, n_lon, n_channels)
        Target batch.
    grid : GaussianGrid
        Grid for area-weighted loss.

    Returns
    -------
    (updated_model, updated_opt_state, loss)
    """
    def loss_fn(model):
        # vmap over batch dimension
        pred = jax.vmap(lambda x: model(x, grid))(batch_input)
        return area_weighted_mse(
            pred, batch_target,
            grid.weights.astype(jnp.float32),
        )

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


def save_checkpoint(
    model: eqx.Module,
    path: str | Path,
) -> None:
    """Save model checkpoint.

    Parameters
    ----------
    model : eqx.Module
        Model to save.
    path : str or Path
        File path for the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(path), model)


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
