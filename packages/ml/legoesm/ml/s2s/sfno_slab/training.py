"""Autoregressive stochastic S2S training helpers for the local SFNO."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import time

import equinox as eqx
import jax
import jax.image as jimage
import jax.numpy as jnp

from legoesm.grids.gaussian import GaussianGrid
from legoesm.ml.loss import (
    area_weighted_afcrps,
    area_weighted_mse,
    weighted_mae,
)
from legoesm.ml.training import create_optimizer, save_checkpoint


class S2SStochasticConfig(NamedTuple):
    """Noise-conditioning and ensemble settings for stochastic SFNO training."""

    ensemble_members: int = 1
    noise_channels: int = 0
    noise_lat: int = 8
    noise_lon: int = 16
    use_time_signal: bool = False
    afcrps_alpha: float = 0.95


class S2STrainingConfig(NamedTuple):
    """Configuration for autoregressive S2S SFNO training."""

    lr: float = 5e-4
    warmup_steps: int = 1000
    total_steps: int = 100_000
    weight_decay: float = 1e-5
    batch_size: int = 2
    n_autoregressive_steps: int = 42
    grad_clip_norm: float = 1.0
    # Optimizer selector consumed by ``create_optimizer`` (adamw = the
    # documented legacy-SFNO default; matches TrainingConfig.optimizer).
    # Without it create_optimizer raised AttributeError on config.optimizer.
    optimizer: str = "adamw"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1000
    validation_every: int = 1000
    validation_batches: int = 4
    save_best_checkpoint: bool = True
    save_final_checkpoint: bool = False
    save_step_checkpoints: bool = False
    stochastic_seed: int = 0
    stochastic_config: S2SStochasticConfig = S2SStochasticConfig()
    tendency_prediction: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str = "disabled"
    wandb_run_name: str | None = None
    wandb_group: str | None = None


def cast_model_to_float32(model: eqx.Module) -> eqx.Module:
    """Cast model array leaves to float32 while preserving non-array leaves."""
    return jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.float32) if eqx.is_array(leaf) else leaf,
        model,
    )


def count_trainable_parameters(model: eqx.Module) -> int:
    """Return the number of trainable array parameters in an Equinox model."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return int(sum(leaf.size for leaf in leaves))


def estimate_sfno_parameter_count(
    *,
    n_sh: int,
    in_channels: int,
    out_channels: int,
    embed_dim: int,
    n_blocks: int,
    mlp_expansion: int,
) -> int:
    """Analytic parameter-count estimate for the local SFNO."""
    encoder = embed_dim * in_channels + embed_dim
    decoder = out_channels * embed_dim + out_channels

    spectral_conv = 2 * n_sh * embed_dim * embed_dim
    layer_norm = 2 * embed_dim
    mlp_hidden = embed_dim * mlp_expansion
    mlp1 = mlp_hidden * embed_dim + mlp_hidden
    mlp2 = embed_dim * mlp_hidden + embed_dim
    block = spectral_conv + layer_norm + mlp1 + mlp2

    return int(encoder + decoder + n_blocks * block)


def extra_input_channels(stochastic_config: S2SStochasticConfig) -> int:
    """Return the extra input channels introduced by stochastic conditioning."""
    return int(stochastic_config.noise_channels) + int(bool(stochastic_config.use_time_signal))


def _sample_noise_field(
    rng_key: jax.Array,
    n_lat: int,
    n_lon: int,
    stochastic_config: S2SStochasticConfig,
) -> jnp.ndarray:
    if stochastic_config.noise_channels <= 0:
        return jnp.zeros((n_lat, n_lon, 0), dtype=jnp.float32)
    coarse = jax.random.normal(
        rng_key,
        (
            stochastic_config.noise_channels,
            stochastic_config.noise_lat,
            stochastic_config.noise_lon,
        ),
        dtype=jnp.float32,
    )
    resized = _resize_noise_periodic_longitude(coarse, n_lat=n_lat, n_lon=n_lon)
    return jnp.moveaxis(resized, 0, -1).astype(jnp.float32)


def _resize_noise_periodic_longitude(
    coarse_noise: jnp.ndarray,
    *,
    n_lat: int,
    n_lon: int,
) -> jnp.ndarray:
    """Resize coarse noise while keeping the longitude wrap periodic."""
    coarse_periodic = jnp.concatenate([coarse_noise, coarse_noise[..., :1]], axis=-1)
    return jimage.resize(
        coarse_periodic,
        (coarse_noise.shape[0], n_lat, n_lon + 1),
        method="linear",
    )[..., :n_lon]


def _lead_signal_field(
    *,
    n_lat: int,
    n_lon: int,
    lead_index: int | jnp.ndarray,
    n_steps: int,
) -> jnp.ndarray:
    denom = jnp.float32(max(int(n_steps), 1))
    value = (jnp.asarray(lead_index, dtype=jnp.float32) + jnp.float32(1.0)) / denom
    return jnp.full((n_lat, n_lon, 1), value, dtype=jnp.float32)


def assemble_model_input(
    atmosphere: jnp.ndarray,
    forcing: jnp.ndarray,
    *,
    stochastic_config: S2SStochasticConfig = S2SStochasticConfig(),
    rng_key: jax.Array | None = None,
    lead_index: int | jnp.ndarray = 0,
    n_steps: int = 1,
) -> jnp.ndarray:
    """Assemble the model input from atmosphere, forcing, noise, and lead signal."""
    atmosphere = atmosphere.astype(jnp.float32)
    forcing = forcing.astype(jnp.float32)
    parts = [atmosphere, forcing]

    if stochastic_config.noise_channels > 0:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        noise = _sample_noise_field(rng_key, atmosphere.shape[0], atmosphere.shape[1], stochastic_config)
        parts.append(noise)

    if stochastic_config.use_time_signal:
        parts.append(
            _lead_signal_field(
                n_lat=atmosphere.shape[0],
                n_lon=atmosphere.shape[1],
                lead_index=lead_index,
                n_steps=n_steps,
            )
        )

    return jnp.concatenate(parts, axis=-1)


def predict_next_atmosphere(
    model: eqx.Module,
    atmosphere: jnp.ndarray,
    forcing: jnp.ndarray,
    grid: GaussianGrid,
    *,
    stochastic_config: S2SStochasticConfig = S2SStochasticConfig(),
    rng_key: jax.Array | None = None,
    lead_index: int | jnp.ndarray = 0,
    n_steps: int = 1,
    tendency_prediction: bool = False,
) -> jnp.ndarray:
    """Predict one daily atmospheric step from the current state and forcing."""
    model_input = assemble_model_input(
        atmosphere,
        forcing,
        stochastic_config=stochastic_config,
        rng_key=rng_key,
        lead_index=lead_index,
        n_steps=n_steps,
    )
    prediction = model(model_input, grid).astype(jnp.float32)
    if tendency_prediction:
        prediction = atmosphere + prediction
    return prediction


def rollout_with_forcing(
    model: eqx.Module,
    initial_input: jnp.ndarray,
    forcing_sequence: jnp.ndarray,
    grid: GaussianGrid,
    *,
    rng_key: jax.Array | None = None,
    stochastic_config: S2SStochasticConfig = S2SStochasticConfig(),
    tendency_prediction: bool = False,
) -> jnp.ndarray:
    """Roll out the atmosphere model with forcing, noise, and lead-time conditioning.

    ``forcing_sequence[t]`` is the exogenous forcing presented for the next
    predicted day ``t + 1``. The scan therefore carries the predicted
    atmosphere state forward while swapping in the teacher-forced boundary
    condition for the next step.
    """
    state_dtype = jnp.float32
    initial_input = initial_input.astype(state_dtype)
    forcing_sequence = forcing_sequence.astype(state_dtype)

    n_steps = forcing_sequence.shape[0]
    n_forcing_channels = forcing_sequence.shape[-1]
    n_atmos_channels = initial_input.shape[-1] - n_forcing_channels
    initial_atmosphere = initial_input[..., :n_atmos_channels]
    initial_forcing = initial_input[..., n_atmos_channels:]

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    step_keys = jax.random.split(rng_key, n_steps)
    lead_indices = jnp.arange(n_steps, dtype=jnp.int32)

    def step_fn(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        inputs: tuple[jnp.ndarray, jax.Array, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        current_atmosphere, current_forcing = carry
        next_forcing, step_key, lead_index = inputs
        prediction = predict_next_atmosphere(
            model,
            current_atmosphere,
            current_forcing,
            grid,
            stochastic_config=stochastic_config,
            rng_key=step_key,
            lead_index=lead_index,
            n_steps=n_steps,
            tendency_prediction=tendency_prediction,
        )
        return (prediction, next_forcing.astype(state_dtype)), prediction

    _, predictions = jax.lax.scan(
        step_fn,
        (initial_atmosphere, initial_forcing),
        (forcing_sequence, step_keys, lead_indices),
    )
    return predictions


def _sample_ensemble_rollouts(
    model: eqx.Module,
    batch_input: jnp.ndarray,
    batch_forcing: jnp.ndarray,
    grid: GaussianGrid,
    rng_key: jax.Array,
    stochastic_config: S2SStochasticConfig,
    tendency_prediction: bool,
) -> jnp.ndarray:
    batch_size = batch_input.shape[0]
    ensemble_keys = jax.random.split(
        rng_key,
        batch_size * stochastic_config.ensemble_members,
    ).reshape(batch_size, stochastic_config.ensemble_members, 2)

    def sample_batch_member(
        initial_input: jnp.ndarray,
        forcing_sequence: jnp.ndarray,
        member_keys: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(
            lambda member_key: rollout_with_forcing(
                model,
                initial_input,
                forcing_sequence,
                grid,
                rng_key=member_key,
                stochastic_config=stochastic_config,
                tendency_prediction=tendency_prediction,
            )
        )(member_keys)

    return jax.vmap(sample_batch_member)(batch_input, batch_forcing, ensemble_keys)


def _target_tendencies(
    batch_input: jnp.ndarray,
    batch_target: jnp.ndarray,
) -> jnp.ndarray:
    """Convert absolute target states into one-step tendencies."""
    initial_atmosphere = batch_input[..., : batch_target.shape[-1]].astype(jnp.float32)
    previous_truth = jnp.concatenate(
        [initial_atmosphere[:, None, ...], batch_target[:, :-1, ...]],
        axis=1,
    )
    return (batch_target - previous_truth).astype(jnp.float32)


def _ensemble_prediction_tendencies(
    batch_input: jnp.ndarray,
    predictions: jnp.ndarray,
) -> jnp.ndarray:
    """Convert absolute ensemble predictions into one-step tendencies."""
    initial_atmosphere = batch_input[..., : predictions.shape[-1]].astype(jnp.float32)
    initial_broadcast = jnp.broadcast_to(
        initial_atmosphere[:, None, None, ...],
        predictions[:, :, :1, ...].shape,
    )
    previous_prediction = jnp.concatenate(
        [initial_broadcast, predictions[:, :, :-1, ...]],
        axis=2,
    )
    return (predictions - previous_prediction).astype(jnp.float32)


@eqx.filter_jit
def train_s2s_step(
    model: eqx.Module,
    opt_state,
    optimizer,
    batch_input: jnp.ndarray,
    batch_target: jnp.ndarray,
    batch_forcing: jnp.ndarray,
    grid: GaussianGrid,
    rng_key: jax.Array,
    stochastic_config: S2SStochasticConfig,
    tendency_prediction: bool,
) -> tuple[eqx.Module, object, jnp.ndarray]:
    """Single JIT-compiled stochastic S2S training step."""
    state_dtype = jnp.float32
    batch_input = batch_input.astype(state_dtype)
    batch_target = batch_target.astype(state_dtype)
    batch_forcing = batch_forcing.astype(state_dtype)
    weights = grid.weights.astype(state_dtype)

    def loss_fn(model):
        predictions = _sample_ensemble_rollouts(
            model,
            batch_input,
            batch_forcing,
            grid,
            rng_key,
            stochastic_config,
            tendency_prediction,
        )
        if tendency_prediction:
            target_for_loss = _target_tendencies(batch_input, batch_target)
            predictions_for_loss = _ensemble_prediction_tendencies(batch_input, predictions)
        else:
            target_for_loss = batch_target
            predictions_for_loss = predictions
        predictions_for_loss = jnp.swapaxes(predictions_for_loss, 0, 1)
        return area_weighted_afcrps(
            predictions_for_loss,
            target_for_loss,
            weights,
            alpha=stochastic_config.afcrps_alpha,
        )

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def validate_s2s_step(
    model: eqx.Module,
    batch_input: jnp.ndarray,
    batch_target: jnp.ndarray,
    batch_forcing: jnp.ndarray,
    grid: GaussianGrid,
    rng_key: jax.Array,
    stochastic_config: S2SStochasticConfig,
    tendency_prediction: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Validation metrics for stochastic autoregressive S2S rollouts."""
    state_dtype = jnp.float32
    batch_input = batch_input.astype(state_dtype)
    batch_target = batch_target.astype(state_dtype)
    batch_forcing = batch_forcing.astype(state_dtype)
    predictions = _sample_ensemble_rollouts(
        model,
        batch_input,
        batch_forcing,
        grid,
        rng_key,
        stochastic_config,
        tendency_prediction,
    )
    weights = grid.weights.astype(state_dtype)
    if tendency_prediction:
        target_for_loss = _target_tendencies(batch_input, batch_target)
        predictions_for_loss = _ensemble_prediction_tendencies(batch_input, predictions)
    else:
        target_for_loss = batch_target
        predictions_for_loss = predictions
    afcrps = area_weighted_afcrps(
        jnp.swapaxes(predictions_for_loss, 0, 1),
        target_for_loss,
        weights,
        alpha=stochastic_config.afcrps_alpha,
    )
    ensemble_mean = jnp.mean(predictions, axis=1)
    mse = area_weighted_mse(ensemble_mean, batch_target, weights)
    mae = weighted_mae(ensemble_mean, batch_target, weights)
    return afcrps, mse, mae


def train_sfno_s2s(
    model: eqx.Module,
    grid: GaussianGrid,
    train_iterator,
    training_config: S2STrainingConfig,
    *,
    val_iterator=None,
    log_every: int = 100,
) -> eqx.Module:
    """Train a local stochastic SFNO on S2S batches from the iterator."""
    optimizer = create_optimizer(training_config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    best_val_loss = None
    best_path = Path(training_config.checkpoint_dir) / "best.eqx"
    start_time = time.time()
    wandb_run = None
    last_validation_step = None
    train_rng = jax.random.PRNGKey(training_config.stochastic_seed)
    val_rng = jax.random.PRNGKey(training_config.stochastic_seed + 1_000_000)

    if training_config.wandb_project and training_config.wandb_mode != "disabled":
        import wandb

        wandb_config = training_config._asdict()
        wandb_config["stochastic_config"] = training_config.stochastic_config._asdict()

        wandb_run = wandb.init(
            project=training_config.wandb_project,
            entity=training_config.wandb_entity,
            mode=training_config.wandb_mode,
            name=training_config.wandb_run_name,
            group=training_config.wandb_group,
            dir=str(Path(training_config.checkpoint_dir)),
            config=wandb_config,
        )
        wandb_run.define_metric("trainer_step")
        wandb_run.define_metric("train/*", step_metric="trainer_step")
        wandb_run.define_metric("val/*", step_metric="trainer_step")

    def run_validation(step: int, elapsed: float) -> None:
        nonlocal best_val_loss, last_validation_step, val_rng
        if val_iterator is None:
            return

        val_losses: list[float] = []
        val_mses: list[float] = []
        val_maes: list[float] = []
        n_batches = max(1, int(training_config.validation_batches))
        for _ in range(n_batches):
            val_input, val_target, val_forcing = next(val_iterator)
            n_val_steps = min(val_target.shape[1], training_config.n_autoregressive_steps)
            val_rng, batch_key = jax.random.split(val_rng)
            val_loss, val_mse, val_mae = validate_s2s_step(
                model,
                val_input,
                val_target[:, :n_val_steps],
                val_forcing[:, :n_val_steps],
                grid,
                batch_key,
                training_config.stochastic_config,
                training_config.tendency_prediction,
            )
            val_losses.append(float(val_loss))
            val_mses.append(float(val_mse))
            val_maes.append(float(val_mae))

        val_loss_value = float(sum(val_losses) / len(val_losses))
        val_mse_value = float(sum(val_mses) / len(val_mses))
        val_mae_value = float(sum(val_maes) / len(val_maes))
        if wandb_run is not None:
            wandb_run.log(
                {
                    "trainer_step": step,
                    "val/loss": val_loss_value,
                    "val/ensemble_mean_mse": val_mse_value,
                    "val/ensemble_mean_mae": val_mae_value,
                    "train/elapsed_sec": elapsed,
                },
                step=step,
            )
        print(
            f"Step {step:6d} | val_loss: {val_loss_value:.6f} "
            f"| val_ens_mse: {val_mse_value:.6f} "
            f"| val_ens_mae: {val_mae_value:.6f}",
            flush=True,
        )
        if training_config.save_best_checkpoint and (
            best_val_loss is None or val_loss_value < best_val_loss
        ):
            best_val_loss = val_loss_value
            save_checkpoint(model, best_path)
            if wandb_run is not None:
                wandb_run.log({"val/best_loss": best_val_loss}, step=step)
            print(
                f"Saved best checkpoint: {best_path} (val_loss={best_val_loss:.6f})",
                flush=True,
            )
        last_validation_step = step

    for step in range(training_config.total_steps):
        batch_input, batch_target, batch_forcing = next(train_iterator)
        n_steps = min(batch_target.shape[1], training_config.n_autoregressive_steps)
        batch_target = batch_target[:, :n_steps]
        batch_forcing = batch_forcing[:, :n_steps]
        train_rng, step_key = jax.random.split(train_rng)
        model, opt_state, loss = train_s2s_step(
            model,
            opt_state,
            optimizer,
            batch_input,
            batch_target,
            batch_forcing,
            grid,
            step_key,
            training_config.stochastic_config,
            training_config.tendency_prediction,
        )
        elapsed = time.time() - start_time
        loss_value = float(loss)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "trainer_step": step,
                    "train/loss": loss_value,
                    "train/elapsed_sec": elapsed,
                },
                step=step,
            )

        if step % log_every == 0:
            print(f"Step {step:6d} | train_loss: {loss_value:.6f}", flush=True)

        if (
            val_iterator is not None
            and step > 0
            and step % training_config.validation_every == 0
        ):
            run_validation(step, elapsed)

        if (
            training_config.save_step_checkpoints
            and step > 0
            and step % training_config.checkpoint_every == 0
        ):
            ckpt_path = Path(training_config.checkpoint_dir) / f"step_{step:06d}.eqx"
            save_checkpoint(model, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}", flush=True)

    final_step = training_config.total_steps - 1
    final_elapsed = time.time() - start_time
    if val_iterator is not None and last_validation_step != final_step:
        run_validation(final_step, final_elapsed)

    if training_config.save_best_checkpoint and best_val_loss is None:
        save_checkpoint(model, best_path)
        print(f"Saved best checkpoint without validation trigger: {best_path}", flush=True)

    if training_config.save_final_checkpoint:
        final_path = Path(training_config.checkpoint_dir) / "final.eqx"
        save_checkpoint(model, final_path)
        print(f"Training complete. Final checkpoint: {final_path}", flush=True)
    else:
        print(f"Training complete. Best checkpoint: {best_path}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    return model
