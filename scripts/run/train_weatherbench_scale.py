#!/usr/bin/env python
"""WeatherBench scale training (Derecho / Levante), data-parallel across ranks.

Multi-node data-parallel: one MPI rank per GPU, each rank trains the SAME
replicated model on a distinct ERA5 shard; gradients are averaged across ranks
each step via ``legoesm.training.data_parallel`` (mpi4jax ``global_sum_mpi``).
Supports all three modes (``--mode physics|neural_gcm|sfno``) at 0.7 deg lat-lon.

Launch: ``scripts/cluster/{derecho/train_wb.pbs, levante/train_wb.slurm}``.
Single-process (no MPI) also works (num_processes == 1 -> the serial path), for a
single-GPU smoke.

The arg-parse / config layer (``build_scale_config_from_args``) is import-light
and JAX-free (unit-tested on the login node); all heavy imports live inside
``main`` so ``--help`` and the CLI test do not pull in JAX.
"""
from __future__ import annotations

import argparse
from typing import NamedTuple

VALID_MODES = ("physics", "neural_gcm", "sfno")


class ScaleConfig(NamedTuple):
    mode: str
    config_path: str
    resolution_deg: float
    n_epochs: int
    multi_step_hours: tuple
    lr: float
    optimizer: str
    grad_accum: int
    out_dir: str
    resume: bool
    eval_wb2: bool
    smoke: bool


def _parse_hours(s):
    if not s:
        return ()
    return tuple(int(x) for x in str(s).split(",") if x != "")


def build_scale_config_from_args(argv=None) -> ScaleConfig:
    """Parse CLI into a ScaleConfig. Import-light + JAX-free (login-node testable)."""
    p = argparse.ArgumentParser(description="WeatherBench data-parallel scale training")
    p.add_argument("--mode", choices=VALID_MODES, default="neural_gcm")
    p.add_argument("--config", default="config/wb/scale/train_07deg.yaml")
    p.add_argument("--resolution", type=float, default=0.7, dest="resolution_deg")
    p.add_argument("--epochs", type=int, default=40, dest="n_epochs")
    p.add_argument("--multi-step-hours", default="6,12", dest="multi_step_hours")
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--optimizer", default="adamw")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--out", default="results/wb_scale", dest="out_dir")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-wb2", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    if a.mode not in VALID_MODES:
        raise SystemExit(f"unknown --mode {a.mode!r}; choose from {VALID_MODES}")
    return ScaleConfig(
        mode=a.mode, config_path=a.config, resolution_deg=a.resolution_deg,
        n_epochs=a.n_epochs, multi_step_hours=_parse_hours(a.multi_step_hours),
        lr=a.lr, optimizer=a.optimizer, grad_accum=a.grad_accum,
        out_dir=a.out_dir, resume=a.resume, eval_wb2=a.eval_wb2, smoke=a.smoke,
    )


def _mpi_rank_size():
    """(rank, num_processes) from MPI. RAISES if a multi-rank launcher is present
    but mpi4py init fails -- otherwise every rank would silently train
    independently (no cross-rank gradient average). Returns (0, 1) only when no
    multi-rank launcher is detected."""
    import os
    launcher = 1
    for v in ("SLURM_NTASKS", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MPI_LOCALNRANKS"):
        val = os.environ.get(v, "")
        if val.isdigit():
            launcher = max(launcher, int(val))
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except Exception as exc:
        if launcher > 1:
            raise RuntimeError(
                f"multi-rank launcher detected (size={launcher}) but mpi4py init failed "
                f"({exc}); gradients would NOT be averaged across ranks -- aborting") from exc
        return 0, 1


def main(argv=None):
    cfg = build_scale_config_from_args(argv)

    # Heavy imports here so --help / the CLI test stay JAX-free.
    import logging

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import yaml

    from legoesm.training.data_parallel import (
        shard_samples, mpi_data_parallel_training_loop,
    )
    from legoesm.ml.training import create_optimizer, TrainingConfig

    rank, nproc = _mpi_rank_size()
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    log = logging.getLogger("wb_scale")
    log.info("WB scale train: mode=%s res=%.2fdeg ranks=%d", cfg.mode, cfg.resolution_deg, nproc)

    yml = yaml.safe_load(open(cfg.config_path))
    if cfg.smoke:  # tiny override for a single-GPU pipeline check
        yml["n_lat"], yml["n_lon"], yml["nlev"] = 32, 64, 8
        cfg = cfg._replace(n_epochs=1)

    # --- build grid / sigma / physics pipeline / model / mode loss_fn ---
    #  (reuses the same package builders run_aimip_latlon uses; the mode differs
    #   only in the make_run_seg factory, per training_driver._build_train_step)
    from legoesm.training.scale_build import build_mode_components  # thin adapter (below)
    model, grid, sigma, params, make_run_seg, loss_config, dt = build_mode_components(cfg, yml)

    # --- ERA5 IC/target/forcing samples, sharded across ranks ---
    from legoesm.training.scale_build import load_era5_samples
    samples = load_era5_samples(cfg, yml, grid, sigma)          # list of (ic, target, forcing)
    local = shard_samples(samples, rank, nproc)
    if nproc > 1 and len(local) == 0:
        raise RuntimeError(
            f"rank {rank}: empty local shard (global samples={len(samples)} < ranks={nproc}); "
            "reduce ranks or add training data")
    log.info("ERA5 samples: %d global, %d local/rank", len(samples), len(local))

    # --- data-parallel loss over Equinox array-leaves ---
    from legoesm.training.dycore_rollout import single_day_rollout
    from legoesm.training.losses import combined_loss
    sigma_full = jnp.asarray(sigma.sigma_full)
    arr, static = eqx.partition(params, eqx.is_inexact_array)

    def loss_fn(arr_leaves, sample):
        trainable = eqx.combine(arr_leaves, static)
        ic, target, forcing = sample
        pred = single_day_rollout(ic, forcing, make_run_seg(trainable).raw, dt=dt)
        return combined_loss(pred, target, sigma_full, grid=grid, config=loss_config)

    total_steps = cfg.n_epochs * max(len(local), 1)
    # Honor the YAML warmup but clamp it below total_steps: the cosine schedule
    # needs decay_steps = total_steps - warmup_steps > 0, which the tiny --smoke
    # run (total_steps ~ a handful) otherwise violates.
    warmup = min(int(yml.get("warmup_steps", TrainingConfig().warmup_steps)),
                 max(1, total_steps // 10))
    optimizer = create_optimizer(TrainingConfig(
        lr=cfg.lr, optimizer=cfg.optimizer,
        warmup_steps=warmup, total_steps=total_steps,
    ))
    opt_state = optimizer.init(arr)

    import os
    os.makedirs(cfg.out_dir, exist_ok=True)

    def on_epoch(epoch, mean_loss, cur_arr, cur_opt_state):
        if rank == 0:
            log.info("Epoch %4d: loss=%.6f", epoch, mean_loss)
            path = os.path.join(cfg.out_dir, f"epoch_{epoch:04d}.eqx")
            eqx.tree_serialise_leaves(path, eqx.combine(cur_arr, static))  # CURRENT params

    arr, opt_state, history = mpi_data_parallel_training_loop(
        loss_fn, arr, opt_state, optimizer, local, cfg.n_epochs, nproc, on_epoch=on_epoch)
    params = eqx.combine(arr, static)
    log.info("training done: final loss=%.6f", history[-1] if history else float("nan"))

    if cfg.eval_wb2 and rank == 0:
        log.info("running WB2 scorecard on the held-out window ...")
        from legoesm.training.scale_build import evaluate_wb2
        evaluate_wb2(cfg, yml, model, params, make_run_seg, grid, sigma)


if __name__ == "__main__":
    main()
