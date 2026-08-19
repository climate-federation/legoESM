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
VALID_TRAINING_CORES = ("latlon", "spectral")


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
    # #817 blocker 1: 'latlon' = the explicit C-grid production core;
    # 'spectral' = the Gaussian semi-implicit training core whose implicit
    # gravity-wave treatment keeps the training adjoint bounded (and whose
    # grid has no pole-cell CFL clamp).  Appended AFTER the original fields
    # so positional construction in existing tests stays valid.
    training_core: str = "latlon"
    # #1047 ask a: training-window length [days] per train year. None = fall
    # back to YAML ``n_training_days`` then 3 (the historical hardcoded value
    # that data-starved the column MLP). --smoke still forces 1. Appended last
    # so positional construction in existing tests stays valid.
    n_days: int | None = None


def _parse_hours(s):
    if not s:
        return ()
    return tuple(int(x) for x in str(s).split(",") if x != "")


def _clamped_warmup(total_steps: int, desired_warmup: int) -> int:
    """Warmup step count safe for ``optax.warmup_cosine_decay_schedule``.

    The schedule builds its cosine phase with ``decay_steps - warmup_steps``
    (``decay_steps`` == ``total_steps`` here), which optax requires to be
    strictly positive.  A tiny run (``--smoke`` gives ``total_steps`` of a few)
    otherwise trips ``cosine_decay_schedule requires positive decay_steps``.

    Honor the configured warmup verbatim whenever it is valid, only capping at
    ``total_steps - 1`` so ``decay_steps >= 1`` (this is the *sole* constraint
    optax imposes — no silent hyperparameter coercion beyond it).  The cap binds
    only when ``desired >= total_steps``, e.g. the degenerate ``total_steps == 1``
    (one epoch over a single local sample) -> warmup 0.  A YAML
    ``warmup_steps: <negative>`` is floored to 0 (no warmup), never passed
    through as a negative to optax.
    """
    return min(max(0, int(desired_warmup)), max(0, total_steps - 1))


def _apply_smoke_overrides(cfg: ScaleConfig, yml: dict) -> ScaleConfig:
    """Shrink ``yml`` (in place) + ``cfg`` to a fast single-GPU wiring check.

    A ~minutes pipeline check, per the runbook: a tiny 32x64x8 grid, one epoch,
    and **gray radiation** in place of rrtmgp.  The full rrtmgp k-distribution
    graph's JIT compile alone exceeds a 40-min single-GPU walltime inside the
    differentiable checkpointed rollout (issue #797 item 6); gray flows through
    the identical wiring under test (``compute_radiation_core``'s lat/lon
    flatten, the q_v smoothing gate) but compiles in seconds, so the smoke
    actually completes.  Opt back into the heavy path (to smoke the rrtmgp
    compile itself, with a bumped walltime) via ``smoke_radiation: rrtmgp``.
    """
    yml["n_lat"], yml["n_lon"], yml["nlev"] = 32, 64, 8
    yml["radiation"] = str(yml.get("smoke_radiation", "gray"))
    return cfg._replace(n_epochs=1)


def build_scale_config_from_args(argv=None) -> ScaleConfig:
    """Parse CLI into a ScaleConfig. Import-light + JAX-free (login-node testable)."""
    p = argparse.ArgumentParser(description="WeatherBench data-parallel scale training")
    p.add_argument("--mode", choices=VALID_MODES, default="neural_gcm")
    p.add_argument("--config", default="config/wb/scale/train_07deg.yaml")
    # The model grid comes from the YAML (n_lat/n_lon/nlev or spectral.n_max),
    # NOT from this flag — a --resolution that silently disagreed with the YAML
    # was the #817 papercut ("--resolution is a no-op on the model grid").
    # Default None = derive from the YAML for logging; an explicit value that
    # mismatches the YAML grid is now a hard error in main() instead of a
    # silent no-op.
    p.add_argument("--resolution", type=float, default=None, dest="resolution_deg",
                   help="Informational check only: must match the YAML grid "
                        "(180/n_lat). To change resolution, edit the YAML.")
    p.add_argument("--training-core", choices=VALID_TRAINING_CORES,
                   default="latlon", dest="training_core",
                   help="latlon = explicit C-grid production core; spectral = "
                        "Gaussian semi-implicit training core (#817 blocker 1: "
                        "bounded adjoint, no pole-cell dt clamp).")
    p.add_argument("--epochs", type=int, default=40, dest="n_epochs")
    p.add_argument("--n-days", type=int, default=None, dest="n_days",
                   help="Training-window length [days] per train year (#1047). "
                        "Default None = YAML n_training_days, else 3. More days = "
                        "more samples (the WB arm was data-starved at 3). --smoke "
                        "forces 1.")
    p.add_argument("--multi-step-hours", default="6,12", dest="multi_step_hours")
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--optimizer", default="adamw")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--out", default="results/wb_scale", dest="out_dir")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-wb2", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    # --mode is already restricted by argparse ``choices=VALID_MODES`` (exits 2
    # on an unknown value), so no manual membership guard is needed here.
    if a.n_epochs < 1:
        # total_steps = n_epochs * n_local_samples feeds optax's cosine
        # decay_steps, which must be positive; a zero/negative epoch count
        # would make it non-positive.
        raise SystemExit(f"--epochs must be >= 1, got {a.n_epochs}")
    if a.n_days is not None and a.n_days < 1:
        raise SystemExit(f"--n-days must be >= 1, got {a.n_days}")
    return ScaleConfig(
        mode=a.mode, config_path=a.config, resolution_deg=a.resolution_deg,
        n_epochs=a.n_epochs, multi_step_hours=_parse_hours(a.multi_step_hours),
        lr=a.lr, optimizer=a.optimizer, grad_accum=a.grad_accum,
        out_dir=a.out_dir, resume=a.resume, eval_wb2=a.eval_wb2, smoke=a.smoke,
        training_core=a.training_core, n_days=a.n_days,
    )


def _check_resolution_matches_yaml(cfg: ScaleConfig, yml: dict) -> float:
    """Resolve the run's resolution [deg] from the YAML grid; reject a
    mismatched explicit ``--resolution`` (the #817 no-op papercut).

    The model grid is built from ``yml["n_lat"]/["n_lon"]`` (lat-lon core) or
    ``yml["spectral"]["n_max"]`` (spectral core); ``--resolution`` never fed it.
    ``None`` (the default) derives 180/n_lat for logging.  An explicit value
    that disagrees by more than 5% is a HARD error — the old behaviour trained
    at the YAML grid while logging the flag's value.
    """
    derived = 180.0 / float(yml["n_lat"])
    if cfg.resolution_deg is None:
        return derived
    if abs(cfg.resolution_deg - derived) > 0.05 * derived:
        raise SystemExit(
            f"--resolution {cfg.resolution_deg:g} deg does not match the YAML "
            f"grid ({yml['n_lat']}x{yml['n_lon']} = {derived:.3g} deg). The "
            f"grid comes from the YAML; edit n_lat/n_lon (and nlev) there "
            f"instead of passing --resolution.")
    return cfg.resolution_deg


def _mpi_rank_size():
    """(rank, num_processes) — shared impl in legoesm.training.data_parallel
    (one copy for both this driver and the AIMIP chunked DP loop)."""
    from legoesm.training.data_parallel import mpi_rank_size
    return mpi_rank_size()



def check_surface_drag_confound(yml, mode, training_core):
    """Refuse a run whose learned arm silently loses its surface stress.

    A campaign may declare ``neural_gcm.surface_drag_confounded:
    core_does_not_read_the_key`` — meaning its learned column cannot be given
    the classical arm's surface stress, because the core it runs never reads
    the key. That is true of the lat-lon core and FALSE of the spectral one,
    which does read it. Selecting the spectral core with such a config would
    score a learned arm carrying no surface stress against a classical arm
    that has one: the #1464 confound, back with a label on it.
    """
    neural = yml.get("neural_gcm") or {}
    if not isinstance(neural, dict):
        return
    declared = neural.get("surface_drag_confounded")
    if (mode == "neural_gcm" and training_core == "spectral"
            and declared == "core_does_not_read_the_key"
            and neural.get("surface_drag") is not True):
        raise SystemExit(
            "this config declares surface_drag_confounded: "
            "'core_does_not_read_the_key', which is only true on the lat-lon "
            "core -- the spectral core DOES read neural_gcm.surface_drag, so "
            "this run would give the learned arm no surface stress while the "
            "classical arm it is compared against has one (#1464). Either run "
            "--training-core latlon, or set neural_gcm.surface_drag: true with "
            "surface_drag_scheme equal to classical.turbulence.")


def main(argv=None):
    cfg = build_scale_config_from_args(argv)

    # Heavy imports here so --help / the CLI test stay JAX-free.
    import logging

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import yaml

    from legoesm.training.data_parallel import (
        mpi_data_parallel_training_loop,
    )
    from legoesm.ml.training import create_optimizer, TrainingConfig

    rank, nproc = _mpi_rank_size()
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    log = logging.getLogger("wb_scale")

    yml = yaml.safe_load(open(cfg.config_path))
    check_surface_drag_confound(yml, cfg.mode, cfg.training_core)
    if cfg.smoke:  # tiny, gray-radiation single-GPU wiring check (see helper)
        cfg = _apply_smoke_overrides(cfg, yml)
    # Resolution comes FROM the YAML grid; an explicit mismatched --resolution
    # is a hard error, not a silent no-op (#817 papercut). Under --smoke the
    # grid is forced to 32x64, so the derived value is used verbatim.
    res_deg = (180.0 / float(yml["n_lat"]) if cfg.smoke
               else _check_resolution_matches_yaml(cfg, yml))
    log.info("WB scale train: mode=%s core=%s res=%.2fdeg ranks=%d",
             cfg.mode, cfg.training_core, res_deg, nproc)

    # --- build grid / sigma / physics pipeline / model / mode loss_fn ---
    #  (reuses the same package builders run_aimip_latlon uses; the mode differs
    #   only in the make_run_seg factory, per training_driver._build_train_step)
    from legoesm.training.scale_build import build_mode_components  # thin adapter (below)
    model, grid, sigma, params, make_run_seg, loss_config, dt = build_mode_components(cfg, yml)

    # Warm the RRTMGP optics-table cache with a CONCRETE build, before anything
    # traced runs.  RRTMGP reads its NetCDF gas-optics tables on first use and
    # stashes them in a module-level cache keyed on static file paths and gas
    # concentrations.  The classical arm constructs that stack inside
    # ``make_run_seg``, which ``loss_fn`` below reaches under
    # ``eqx.filter_jit(jax.value_and_grad(...))`` (data_parallel.py) — and
    # inside a trace ``jnp.array(<numpy table>)`` is a tracer, so the loader's
    # own ``np.asarray`` on it dies with TracerArrayConversionError (job
    # 26905933, after the 6-minute ERA5 load; reproduced standalone on JAX
    # 0.10.0).  One concrete call here fills the cache — its key holds only
    # static paths/flags, never a trained leaf, so the in-trace build hits it
    # and skips the load.  It is also where a broken physics config now fails,
    # seconds in rather than minutes.  ``neural_gcm``/``sfno`` build a cheap
    # closure here and are unaffected.  Mirrors the warm-up the AIMIP trainer
    # runs before its train step (``neural_gcm_spectral.main``).
    # Single-rank by construction in this campaign (ranks=1); at >1 rank every
    # rank reads its own copy of the tables, because the cache is per-process.
    # ``RRTMGP.preload_mpi`` (rank 0 reads + broadcasts) would avoid that, but
    # it needs the arm's RRTMGPConfig, which is built inside the physics factory
    # and not exposed here — wire it through if WB ever trains multi-rank.
    make_run_seg(params)

    # --- ERA5 IC/target/forcing samples, sharded across ranks ---
    # #1286: the loader builds ONLY this rank's contiguous shard (fix B — never
    # the full global list) and keeps it HOST-resident (fix A — the training
    # loop device_puts one sample at a time).  The rank's slice is byte-for-byte
    # the old ``shard_samples(build_all(), rank, nproc)`` partition, so gradient
    # semantics are unchanged; we no longer materialize the global GPU-resident
    # set that OOM'd at T106.
    from legoesm.training.scale_build import load_era5_samples
    local = load_era5_samples(cfg, yml, grid, sigma,
                              rank=rank, nproc=nproc, host_resident=True)
    if nproc > 1 and len(local) == 0:
        raise RuntimeError(
            f"rank {rank}: empty local shard (ranks={nproc} > global samples); "
            "reduce ranks or add training data")
    log.info("ERA5 samples: %d local/rank (host-resident, sharded-at-build)",
             len(local))

    # --- data-parallel loss over Equinox array-leaves ---
    from legoesm.training.losses import combined_loss
    from legoesm.training.scale_build import rollout_hours
    sigma_full = jnp.asarray(sigma.sigma_full)
    arr, static = eqx.partition(params, eqx.is_inexact_array)

    # The rollout horizon MUST match the target's lead time: load_era5_samples
    # pairs each IC with the state rollout_hours later (the first
    # multi_step_hours lead, default 6 h).  A fixed 24 h single_day_rollout here
    # would score a 24 h forecast against a 6 h target.
    roll_steps = int(rollout_hours(cfg, yml) * 3600.0 / dt)

    def loss_fn(arr_leaves, sample):
        trainable = eqx.combine(arr_leaves, static)
        ic, target, forcing = sample
        pred = make_run_seg(trainable).raw(ic, roll_steps, forcing)
        return combined_loss(pred, target, sigma_full, grid=grid, config=loss_config)

    total_steps = cfg.n_epochs * max(len(local), 1)
    # Honor the YAML warmup but clamp it safely below total_steps (see
    # ``_clamped_warmup``): the cosine schedule needs decay_steps > 0, which the
    # tiny --smoke run otherwise violates.
    warmup = _clamped_warmup(
        total_steps, yml.get("warmup_steps", TrainingConfig().warmup_steps))
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
