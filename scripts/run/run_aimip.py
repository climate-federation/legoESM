#!/usr/bin/env python
"""AIMIP intercomparison driver for legoESM.

Trains and evaluates a fixed set of atmospheric-physics variants on
the same ERA5 IC/target window, writing a unified scorecard JSON.

All variants share the spectral primitive-equation dynamical core
(``SpectralPrimitiveEquationModel``) on a Gaussian grid so the
comparison reflects the choice of physics representation rather than
the dycore.  The variants are:

* ``classical`` — the FULL physics suite: Tiedtke convection, Louis
  turbulence, surface bulk fluxes, McFarlane gravity-wave drag,
  Xu-Randall cloud fraction, Sundqvist microphysics and RRTMGP
  correlated-k radiation (``aimip_radiation``; gray is the cheap
  opt-in / smoke backend).  Tunables exposed via
  :class:`legoesm.training.aimip_params.AIMIPClassicalParams`
  and trained end-to-end through the differentiable spectral PE
  rollout.
* ``column_nn`` — ALL physics, radiation included, learned by a column
  MLP (Rasp et al., 2018 style) via
  :func:`legoesm.training.neural_gcm_spectral.train_column_mlp_spectral`;
  no classical scheme runs alongside it.
* ``sfno_physics`` — SFNO replaces the entire gridded physics step
  (radiation included), dycore retained, via
  :func:`legoesm.training.neural_gcm_spectral.train_neural_gcm_spectral`.
* ``sfno_full`` — SFNO as the full atmospheric emulator (no dycore,
  no physics tendency) via
  :func:`legoesm.training.neural_gcm_spectral.train_sfno_full_spectral`.
  The trained model is :class:`SFNOPrimitiveEquationModel` operating
  in ``state_update`` mode at a macro time step ``dt_sfno`` (default
  6 h), with post-hoc dry-air-mass and moisture-budget corrections.

Usage
-----
::

    JAX_ENABLE_X64=1 python scripts/run_aimip.py \\
        --suite config/aimip/aimip_suite.yaml [--smoke]

The ``--smoke`` flag overrides the suite to T21 / 8 levels / 1 epoch /
2 train days so the full intercomparison fits in a few minutes for
end-to-end verification.

Optimization
------------
All variants flow gradients through ``eqx.filter_value_and_grad`` and
default to MUON via :func:`legoesm.ml.training.create_optimizer`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import yaml


logger = logging.getLogger("aimip")


from legoesm.driver.config import AIMIP_VARIANTS as _AIMIP_VARIANTS_FULL
from legoesm.ml.loss import latitude_weighted_bias, latitude_weighted_rmse

# Run-time variants (drop the empty string which means "not AIMIP").
_VALID_VARIANTS = tuple(v for v in _AIMIP_VARIANTS_FULL if v)


# ----------------------------------------------------------------------
# YAML loading + overlay merge
# ----------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Shallow dict merge (overlay keys override base)."""
    out = dict(base)
    out.update(overlay)
    return out


def _apply_smoke_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg.update(dict(
        n_max=21,
        nlev=8,
        n_train_days=2,
        n_eval_days=1,
        aimip_n_epochs=1,
        aimip_warmup=1,
        dt=1800.0,
        # Force the cheap radiation backend + single-day rollout for
        # smoke runs.  The full AIMIP base config defaults to RRTMGP +
        # spatial surface fields for production training, but those
        # add ~10x compile cost and aren't useful for smoke-level
        # end-to-end verification.
        aimip_radiation="gray",
        aimip_spatial_surface=False,
        aimip_rollout_days=1,
        # Drop any window list from the base (the all-years scale base
        # carries 1152 pairs; even aimip_era5.yaml carries several): the
        # smoke contract is a minutes-scale end-to-end check, so fall back
        # to the start_year + n_train_days/n_eval_days loader path.
        train_windows=None,
        eval_windows=None,
        aimip_rollout_curriculum=None,
        aimip_chunk_windows=0,
    ))
    return cfg


# ----------------------------------------------------------------------
# Variant dispatch
# ----------------------------------------------------------------------

def _surface_forcing_cfg(cfg: dict[str, Any]) -> tuple[str | None, str | None]:
    """Resolve the prescribed-surface-forcing config for NN variants.

    Returns ``(forcing_path, cache_path)``.  ``aimip_surface_forcing``
    (default True) gates it; the cache is keyed by spectral truncation so
    a T63 cache can never be silently reused at T106 (the loader also
    hard-validates ncol).  Forcing is what gives column_nn / sfno_physics
    a prescribed-SST (AMIP / interannual-variability) pathway.
    """
    if not bool(cfg.get("aimip_surface_forcing", True)):
        return None, None
    from legoesm.training.aimip_amip_forcing import DEFAULT_AIMIP_FORCING
    path = str(cfg.get("aimip_forcing_path") or DEFAULT_AIMIP_FORCING)
    n_max = int(cfg["n_max"])
    cache = str(
        cfg.get("aimip_forcing_cache")
        or f"results/aimip_forcing/gaussian_forcing_T{n_max}.npz"
    )
    return path, cache


def _build_spectral_config(cfg: dict[str, Any]):
    """Translate AIMIP YAML dict into NeuralGCMSpectralConfig."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig
    from legoesm.training.losses import LossConfig

    loss_kwargs = cfg.get("loss", {}) or {}
    # Named loss preset (D5): numbers live once in
    # config/wb/loss_presets/<name>.yaml; the suite's own loss: block wins
    # key-by-key.
    _preset = cfg.get("loss_preset")
    if _preset:
        from legoesm.training.loss_presets import (
            load_loss_preset,
            merge_loss_preset,
        )
        loss_kwargs = merge_loss_preset(load_loss_preset(str(_preset)), loss_kwargs)
    _unknown = set(loss_kwargs) - set(LossConfig._fields)
    if _unknown:
        raise ValueError(
            f"Unknown loss config keys {sorted(_unknown)} (typo?); valid "
            f"keys are the LossConfig fields in training/losses.py."
        )
    # YAML lists -> tuples for fields LossConfig declares as tuples.
    # Keeps the NamedTuple hashable for ``filter_jit`` static-arg
    # comparisons and matches the tuple-typed default.
    _tuple_fields = {"multi_step_hours", "multi_step_weights"}
    loss_config = LossConfig(**{
        k: (tuple(v) if k in _tuple_fields and v is not None else v)
        for k, v in loss_kwargs.items()
    })

    # Rollout curriculum (aimip_rollout_curriculum: [[lead_hours, epochs],
    # ...]): the loader must build a target at EVERY curriculum lead, so
    # loss.multi_step_hours is forced to the sorted unique leads (equal
    # weights; the curriculum path scores one lead per phase anyway).
    curriculum = tuple(
        (int(h), int(ep))
        for h, ep in (cfg.get("aimip_rollout_curriculum") or ())
    ) or None
    if curriculum:
        _leads = tuple(sorted({h for h, _ in curriculum}))
        loss_config = loss_config._replace(
            multi_step_hours=_leads,
            multi_step_weights=(1.0,) * len(_leads),
        )

    sfno_embed = int(cfg.get("sfno_embed_dim", 128))
    sfno_n_blocks = int(cfg.get("sfno_n_blocks", 4))
    sfno_mlp_expansion = int(cfg.get("sfno_mlp_expansion", 4))

    return NeuralGCMSpectralConfig(
        n_max=int(cfg["n_max"]),
        # Config key drift (nlev vs n_levels, CLAUDE.md naming debt): a merge
        # left this read as "nlev" while the AIMIP configs declare "n_levels",
        # so the spectral suite KeyError'd at step 0 (the v10 T63 ~2K path was
        # fully blocked). Accept either key.
        n_levels=int(cfg["nlev"] if "nlev" in cfg else cfg["n_levels"]),
        dt=float(cfg["dt"]),
        pe_config=SpectralPEConfig(
            hyperdiff_coeff=2.5e15,
            hyperdiff_order=2,
            time_integrator="ssp_rk3",
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
        ),
        sfno_embed_dim=sfno_embed,
        sfno_n_blocks=sfno_n_blocks,
        sfno_mlp_expansion=sfno_mlp_expansion,
        n_epochs=int(cfg["aimip_n_epochs"]),
        lr=float(cfg["aimip_lr"]),
        weight_decay=float(cfg["aimip_weight_decay"]),
        grad_clip_norm=float(cfg["aimip_grad_clip"]),
        optimizer=str(cfg.get("aimip_optimizer", "muon")),
        warmup_steps=int(cfg.get("aimip_warmup", 100)),
        n_train_days=int(cfg["n_train_days"]),
        start_year=int(cfg.get("train_year", cfg.get("years", [2015])[0])),
        early_stop_patience=int(cfg.get("aimip_patience", 0)),
        early_stop_min_delta=float(cfg.get("aimip_min_delta", 1.0e-3)),
        windows=tuple(
            tuple(int(x) for x in w[:3])
            for w in (cfg.get("train_windows") or ())
        ) or None,
        rollout_days=int(cfg.get("aimip_rollout_days", 1)),
        rollout_hours=int(cfg.get("aimip_rollout_hours", 0)),
        rollout_curriculum=curriculum,
        chunk_windows=int(cfg.get("aimip_chunk_windows", 0)),
        chunk_prefetch=bool(cfg.get("aimip_chunk_prefetch", False)),
        data_parallel=bool(cfg.get("aimip_data_parallel", False)),
        ema_decay=float(cfg.get("aimip_ema_decay", 0.0)),
        spatial_lr_scale=float(cfg.get("aimip_spatial_lr_scale", 1.0)),
        rad_update_interval=int(cfg.get("aimip_rad_update_interval", 1)),
        loss_config=loss_config,
        log_every=int(cfg.get("log_every", 1)),
        checkpoint_dir=str(Path(cfg["output_dir"]) / cfg["aimip_variant"]),
    )


def _train_variant(
    variant: str,
    cfg: dict[str, Any],
    cache_dir: str,
    *,
    resume: bool = False,
):
    """Dispatch on AIMIP variant.  Returns (trained_model, loss_history).

    When ``resume`` is True the checkpoint dir
    ``{output_dir}/{aimip_variant}`` is scanned for the highest-numbered
    ``epoch_NNNN.eqx`` and training resumes from epoch+1.  Used by the
    chained-resubmission SLURM driver.
    """
    spec_cfg = _build_spectral_config(cfg)

    resume_from_dir = (
        Path(cfg["output_dir"]) / cfg["aimip_variant"] if resume else None
    )

    if variant == "classical":
        return _train_aimip_classical(
            spec_cfg, cache_dir, cfg=cfg, resume_from_dir=resume_from_dir,
        )

    if variant == "column_nn":
        from legoesm.training.neural_gcm_spectral import (
            train_column_mlp_spectral,
        )
        forcing_path, forcing_cache = _surface_forcing_cfg(cfg)
        return train_column_mlp_spectral(
            config=spec_cfg,
            cache_dir=cache_dir,
            seed=int(cfg.get("nn_seed", 0)),
            hidden_dim=int(cfg.get("nn_hidden_dim", 256)),
            n_layers=int(cfg.get("nn_n_layers", 4)),
            resume_from_dir=resume_from_dir,
            surface_forcing_path=forcing_path,
            forcing_cache_path=forcing_cache,
        )

    if variant == "sfno_physics":
        from legoesm.training.neural_gcm_spectral import (
            train_neural_gcm_spectral,
        )
        forcing_path, forcing_cache = _surface_forcing_cfg(cfg)
        return train_neural_gcm_spectral(
            config=spec_cfg,
            cache_dir=cache_dir,
            seed=int(cfg.get("sfno_seed", 0)),
            resume_from_dir=resume_from_dir,
            surface_forcing_path=forcing_path,
            forcing_cache_path=forcing_cache,
        )

    if variant == "sfno_full":
        from legoesm.training.neural_gcm_spectral import (
            train_sfno_full_spectral,
        )
        return train_sfno_full_spectral(
            config=spec_cfg,
            cache_dir=cache_dir,
            seed=int(cfg.get("sfno_seed", 0)),
            dt_sfno=float(cfg.get("dt_sfno", 21600.0)),
            resume_from_dir=resume_from_dir,
        )

    raise ValueError(f"Unknown AIMIP variant: {variant!r}")


def _train_aimip_classical(
    spec_cfg,
    cache_dir: str,
    *,
    cfg: dict | None = None,
    resume_from_dir=None,
):
    cfg = cfg or {}
    """Train the AIMIP classical variant (Tiedtke/Louis/Surface/McFarlane/XR).

    Builds a Gaussian grid + spectral PE dycore, loads ERA5 daily
    IC/target pairs, and runs the shared ``_train_spectral_loop`` with
    :class:`AIMIPClassicalParams` and
    :func:`make_aimip_classical_spectral_physics` as the
    ``make_physics_fn`` factory.
    """
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.neural_gcm_spectral import (
        _train_spectral_loop,
        load_training_data,
        maybe_resume_model,
    )

    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(
        spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top,
    )

    spatial_surface = bool(cfg.get("aimip_spatial_surface", False))
    params = AIMIPClassicalParams.from_defaults(
        spatial_surface=spatial_surface,
        spatial_init_std=float(cfg.get("aimip_spatial_init_std", 0.0)),
        spatial_seed=int(cfg.get("aimip_spatial_seed", 0)),
    )
    n_scalar = len(params.raw_values)
    n_spatial = (
        params.spatial_surface.n_trainable() if params.spatial_surface else 0
    )
    logger.info(
        f"AIMIPClassicalParams: {n_scalar} trainable scheme knobs"
        + (
            f" + {n_spatial} spatial coefs across "
            f"{len(params.spatial_surface.fields)} surface fields"
            if spatial_surface else ""
        )
    )

    params, start_epoch = maybe_resume_model(params, resume_from_dir)

    # Host-resident dataset (#1155): classical loads ALL pairs up front (no
    # chunking) — at T106 all-years the eager device build is ~130 GB, an
    # unconditional GPU OOM (job 6758505 died at snapshot ~1200/4032 during
    # LOADING). Build on host; the shared loop stages each sample per step.
    ic_states, target_carries, _ic_times = load_training_data(
        spec_cfg, grid, sigma, cache_dir,
        windows=spec_cfg.windows,
        host_resident=True,
    )

    dt = spec_cfg.dt
    # Classical = full physics suite + RRTMGP radiation by default; the
    # cheap gray backend is opt-in (aimip_radiation: gray, and what the
    # --smoke overrides select).
    radiation = str(cfg.get("aimip_radiation", "rrtmgp"))
    rad_update_interval = int(cfg.get("aimip_rad_update_interval", 6))
    # RRTMGP g-point checkpoint: True (default) = byte-for-byte legacy but the
    # prevent_cse=True per-g-point body inflates the GPU compile ~Ng-fold
    # (multi-hour). False = one reused g-point body (answer-identical, minutes
    # to compile) — safe for the rollout-checkpointed training path. Default
    # True; set aimip_rrtmgp_gpoint_checkpoint=false for classical training.
    rrtmgp_gpoint_checkpoint = bool(
        cfg.get("aimip_rrtmgp_gpoint_checkpoint", True))
    # G-point vmap block size: >0 -> fast compile (one block body) with bounded
    # backward memory (holds block_size g-points, not all ~256). 0 = scan path.
    rrtmgp_gpoint_batch_size = int(cfg.get("aimip_rrtmgp_gpoint_batch_size", 16))

    # Derive the land mask from surface geopotential (phis > 0 over
    # land).  Static across samples so we extract it once.  Using a
    # soft sigmoid keeps the lat-lon surface-parameter gradients
    # smooth across coastlines (vs. a hard step that would clip them).
    land_mask = None
    if spatial_surface and target_carries:
        # ``target_carries[0]`` is a single SegmentCarry on the legacy
        # path and a tuple of K SegmentCarry on the multi-step
        # autoregressive path.  Surface geopotential is static across
        # snapshots so any of them works; unwrap when needed.
        ref_carry = target_carries[0]
        # A multi-step target is a PLAIN tuple of carries; a single-step
        # target is ONE SegmentCarry — itself a NamedTuple (tuple subclass),
        # so isinstance(.., tuple) is True for BOTH and would unwrap a single
        # carry to its first FIELD (an array) -> `.phis` AttributeError. This
        # broke the single-step v10 T63 path when multi-step was added.
        # ``type(..) is tuple`` matches the plain tuple only.
        if type(ref_carry) is tuple:
            ref_carry = ref_carry[0]
        from legoesm.training.aimip_spatial import land_mask_from_phis
        from legoesm.training.neural_gcm_spectral import stage_sample
        # ref_carry lives on host under host_resident loading (#1155,
        # UNCOMMITTED — see stage_sample's semantics note). The land mask is
        # a closure constant of the jitted physics; stage it explicitly so
        # its placement is deliberate rather than an implicit per-trace
        # transfer, and so this site stays correct if the loader ever
        # commits its outputs.
        land_mask = stage_sample(land_mask_from_phis(
            jnp.asarray(ref_carry.phis), smooth=True,
        ))

    # When rad gating is on (``aimip_rad_update_interval > 1``),
    # ``make_aimip_classical_spectral_physics`` returns a
    # ``(non_rad_fn, rad_fn)`` tuple and the rollout uses lax.cond
    # to compute the RRTMGP step periodically.  Otherwise the legacy
    # single-callable path runs.
    split_rad = rad_update_interval > 1

    # Physics scheme dispatch from YAML.  Defaults reproduce the legacy
    # AIMIP classical recipe (tiedtke + louis + mcfarlane + none).
    # Used by the combinatorial physics sweep
    # (scripts/run/run_aimip_classical_sweep_stage1.py).
    conv_scheme = str(cfg.get("aimip_convection", "tiedtke"))
    turb_scheme = str(cfg.get("aimip_turbulence", "louis"))
    gwd_scheme = str(cfg.get("aimip_gwd", "mcfarlane"))
    micro_scheme = str(cfg.get("aimip_microphysics", "none"))
    cloud_scheme = str(cfg.get("aimip_cloud", "xu_randall"))
    logger.info(
        f"AIMIP classical physics: conv={conv_scheme} turb={turb_scheme} "
        f"gwd={gwd_scheme} micro={micro_scheme} cloud={cloud_scheme} "
        f"rad={radiation}"
    )

    # Training-period-mean GHG for the classical RRTMGP (CO2 matters for the
    # CLASSICAL variant: a physical radiation scheme trained at present-day
    # defaults sees a systematically wrong forcing for 1979-2012 samples).
    # A per-sample transient value is unwarranted at 6-12 h forecast leads
    # (the radiative signal of a few ppm is far below the loss floor) and
    # the transient path already runs in the AMIP fine-tune + inference;
    # here the STATIC mid-training-period concentration removes the mean
    # bias at zero plumbing cost (closure constants — no retrace).
    _ghg_mid = None
    if radiation == "rrtmgp" and spec_cfg.windows:
        from legoesm.training.aimip_amip_forcing import ghg_vmr_at_year
        _years = [int(w[0]) for w in spec_cfg.windows]
        _mid_year = int(round(sum(_years) / len(_years)))
        _ghg_mid = {k: float(v) for k, v in ghg_vmr_at_year(_mid_year).items()}
        logger.info(f"Classical RRTMGP GHG pinned to training-period mid-year "
                    f"{_mid_year}: co2={_ghg_mid['co2']:.2e}")

    def _make_physics_fn(p, grid_):
        built = make_aimip_classical_spectral_physics(
            p, grid_, dt,
            radiation=radiation,
            rad_update_interval_steps=rad_update_interval,
            convection_scheme=conv_scheme,
            turbulence_scheme=turb_scheme,
            gwd_scheme=gwd_scheme,
            microphysics_scheme=micro_scheme,
            cloud_scheme=cloud_scheme,
            land_mask=land_mask,
            split_rad=split_rad,
            rrtmgp_gpoint_checkpoint=rrtmgp_gpoint_checkpoint,
            rrtmgp_gpoint_batch_size=rrtmgp_gpoint_batch_size,
        )
        if _ghg_mid is None:
            return built
        # Wrap the rad fn so every call carries the mid-period GHG unless
        # the caller supplied its own (transient) value in ``forcing``.
        def _with_ghg(rad_fn):
            def _wrapped(*args, forcing=None, **kwargs):
                fc = dict(forcing or {})
                fc.setdefault("ghg_vmr", _ghg_mid)
                return rad_fn(*args, forcing=fc, **kwargs)
            return _wrapped
        if isinstance(built, tuple):
            non_rad_fn, rad_fn = built
            return non_rad_fn, _with_ghg(rad_fn)
        # Non-split combined fn: its forcing-kwarg contract is not
        # guaranteed — leave unwrapped (rrtmgp effectively always runs
        # split_rad; the combined path is the gray/no-gating config where
        # GHG does not apply). Say so loudly rather than silently
        # dropping the pinned concentration.
        logger.warning(
            "GHG pinning computed but NOT applied: physics fn is not "
            "split-rad (rad_update_interval<=1?) — training runs with "
            "the radiation scheme's default GHG."
        )
        return built

    return _train_spectral_loop(
        params, _make_physics_fn,
        grid, sigma, ic_states, target_carries, spec_cfg,
        start_epoch=start_epoch,
        host_staged=True,   # dataset loaded host-resident above (#1155)
    )


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def _evaluate_variant(
    variant: str,
    trained_model,
    cfg: dict[str, Any],
    cache_dir: str,
    *,
    period: str = "test",
) -> dict[str, float]:
    """Evaluate trained model on a named ERA5 window set.

    Parameters
    ----------
    period : str
        ``"test"``  -> held-out ``eval_windows`` (default; matches the
                       legacy behavior).
        ``"train"`` -> ``train_windows`` (in-sample forecast skill for
                       the AIMIP-fleet annual-mean comparison plot).

    Returns a dict of mean ``LossConfig``-weighted error + per-variable
    RMSE/bias on the chosen window.  Both share the same loss as the
    training objective so the scorecard is directly comparable across
    variants.
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        compute_spectral_filter,
        compute_sponge_factor,
    )
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.aimip_params import (
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.neural_gcm_spectral import (
        load_training_data,
        make_column_mlp_spectral_physics,
        make_sfno_spectral_physics,
        spectral_rollout,
        spectral_state_vs_carry_loss,
    )

    spec_cfg = _build_spectral_config(cfg)
    if period == "train":
        windows_raw = cfg.get("train_windows") or ()
        n_days = int(cfg.get("n_train_days", 2))
        year = int(cfg.get("train_year", spec_cfg.start_year))
    elif period == "test":
        windows_raw = cfg.get("eval_windows") or ()
        n_days = int(cfg.get("n_eval_days", 2))
        year = int(cfg.get("eval_year", spec_cfg.start_year))
    else:
        raise ValueError(f"Unknown eval period {period!r}; expected 'train' or 'test'.")
    eval_windows = tuple(
        tuple(int(x) for x in w[:3]) for w in windows_raw
    ) or None
    eval_cfg = spec_cfg._replace(
        n_train_days=n_days,
        start_year=year,
        windows=eval_windows,
    )

    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(
        spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top,
    )
    ic_states, target_carries, eval_ic_times = load_training_data(
        eval_cfg, grid, sigma, cache_dir, windows=eval_cfg.windows,
    )
    # NN variants trained WITH prescribed surface forcing must be
    # evaluated with the same inputs (a forced network scored unforced
    # would see out-of-distribution proxies and mis-rank the variants).
    eval_forcings = None
    if variant in ("column_nn", "sfno_physics"):
        forcing_path, forcing_cache = _surface_forcing_cfg(cfg)
        if forcing_path is not None:
            from legoesm.training.aimip_amip_forcing import (
                build_amip_sample_forcings,
            )
            eval_forcings = build_amip_sample_forcings(
                eval_ic_times, grid, forcing_path=forcing_path,
                cache_path=forcing_cache,
            )

    pe_config = spec_cfg.pe_config
    sponge_factor = None
    if pe_config.sponge_tau > 0:
        sponge_factor = compute_sponge_factor(
            sigma.sigma_full, pe_config.sponge_sigma,
            pe_config.sponge_tau, spec_cfg.dt,
        )
    spectral_filter = None
    if pe_config.spectral_filter_strength > 0:
        spectral_filter = compute_spectral_filter(
            grid.ls, grid.n_max,
            order=pe_config.spectral_filter_order,
            cutoff_fraction=pe_config.spectral_filter_strength,
        )

    sigma_full = jnp.asarray(sigma.sigma_full)
    n_steps_per_day = int(86400 / spec_cfg.dt)
    eval_rollout_days = int(getattr(spec_cfg, "rollout_days", 1) or 1)
    eval_rollout_hours_cfg = int(getattr(spec_cfg, "rollout_hours", 0) or 0)
    # When multi-step autoregressive supervision is on, evaluate the
    # forecast at the longest training lead (it dominates the scorecard
    # and matches the held-out horizon the model was supervised at).
    multi_step_hours_eval = tuple(
        int(h) for h in (
            (spec_cfg.loss_config.multi_step_hours or ())
            if spec_cfg.loss_config is not None else ()
        )
    )
    if multi_step_hours_eval:
        eval_rollout_hours = max(multi_step_hours_eval)
    elif eval_rollout_hours_cfg > 0:
        eval_rollout_hours = eval_rollout_hours_cfg
    else:
        eval_rollout_hours = eval_rollout_days * 24
    n_steps_eval = int(round(eval_rollout_hours * 3600.0 / spec_cfg.dt))
    eval_rad_interval = int(cfg.get("aimip_rad_update_interval", 1))

    # Build the per-variant rollout closure (model is frozen for eval).
    # ``sfno_full`` doesn't use a physics_fn / dycore at all -- it
    # iterates the trained SFNO step directly at ``dt_sfno``.  All other
    # variants share the spectral_rollout(physics_fn) path.
    eval_physics_pair = None  # (non_rad_fn, rad_fn) when split active
    eval_full_emulator_rollout = None  # set only when variant == "sfno_full"
    physics_fn = None
    if variant == "classical":
        eval_land_mask = None
        if bool(cfg.get("aimip_spatial_surface", False)) and target_carries:
            ref_carry = target_carries[0]
            if type(ref_carry) is tuple:   # plain tuple=multi-step; carry NamedTuple is not
                ref_carry = ref_carry[0]
            from legoesm.training.aimip_spatial import land_mask_from_phis
            eval_land_mask = land_mask_from_phis(
                jnp.asarray(ref_carry.phis), smooth=True,
            )
        eval_split_rad = eval_rad_interval > 1
        built = make_aimip_classical_spectral_physics(
            trained_model, grid, spec_cfg.dt,
            radiation=str(cfg.get("aimip_radiation", "rrtmgp")),
            rad_update_interval_steps=eval_rad_interval,
            convection_scheme=str(cfg.get("aimip_convection", "tiedtke")),
            turbulence_scheme=str(cfg.get("aimip_turbulence", "louis")),
            gwd_scheme=str(cfg.get("aimip_gwd", "mcfarlane")),
            microphysics_scheme=str(cfg.get("aimip_microphysics", "none")),
            cloud_scheme=str(cfg.get("aimip_cloud", "xu_randall")),
            land_mask=eval_land_mask,
            split_rad=eval_split_rad,
        )
        if isinstance(built, tuple):
            eval_physics_pair = built
            physics_fn = built[0]  # non-rad; rad threaded separately
        else:
            physics_fn = built
    elif variant == "column_nn":
        physics_fn = make_column_mlp_spectral_physics(trained_model, grid)
    elif variant == "sfno_physics":
        physics_fn = make_sfno_spectral_physics(trained_model, grid)
    elif variant == "sfno_full":
        from legoesm.atmosphere.dynamics.neural.sfno_pe import (
            SFNOPrimitiveEquationConfig,
            SFNOPrimitiveEquationModel,
        )
        from legoesm.ml.channel_packing import PE3DChannelSpec
        from legoesm.ml.sfno import SFNOConfig
        from legoesm.ml.normalization import load_normalization_stats
        _channels = PE3DChannelSpec(nlev=spec_cfg.n_levels).n_channels
        eval_dt_sfno = float(cfg.get("dt_sfno", 21600.0))
        # sfno_full trains WITH per-channel Z-score normalization; reload the
        # SAME norm_stats.npz the training wrote into checkpoint_dir
        # ({output_dir}/{aimip_variant}) so this in-run eval applies the
        # identical transform (state_update denormalises the output as a full
        # state).  The eqx checkpoint carries only SFNO leaves, so the sidecar
        # (not the checkpoint) is the source of truth for the stats.
        _stats_path = Path(cfg["output_dir"]) / cfg["aimip_variant"] / "norm_stats.npz"
        if not _stats_path.exists():
            raise SystemExit(
                f"sfno_full eval: normalization sidecar not found at "
                f"{_stats_path}. train_sfno_full_spectral writes it per run; "
                f"the model was trained WITH normalization, so eval cannot "
                f"proceed without the matching stats."
            )
        eval_norm_stats = load_normalization_stats(_stats_path)
        eval_pe_cfg = SFNOPrimitiveEquationConfig(
            sfno_config=SFNOConfig(
                in_channels=_channels,
                out_channels=_channels,
                embed_dim=spec_cfg.sfno_embed_dim,
                n_blocks=spec_cfg.sfno_n_blocks,
                mlp_expansion=spec_cfg.sfno_mlp_expansion,
                residual_prediction=False,
            ),
            mode="state_update",
            dt_sfno=eval_dt_sfno,
            correct_mass=True,
            # Not yet wired in the SFNO PE bridge (spectral moisture tracer
            # needs synthesis/clip/re-analysis); previously silently ignored.
            correct_moisture_budget=False,
            clip_q=False,
            use_normalization=True,
        )
        eval_full_wrapper = SFNOPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma,
            config=eval_pe_cfg, sfno_model=trained_model,
            norm_stats=eval_norm_stats,
        )
        n_steps_eval_sfno = max(
            1, int(round(eval_rollout_hours * 3600.0 / eval_dt_sfno))
        )
        logger.info(
            f"sfno_full eval: {n_steps_eval_sfno} SFNO steps "
            f"@ dt_sfno={eval_dt_sfno:.0f}s (= {eval_rollout_hours} h)"
        )

        def _scan_body(s, _):
            return eval_full_wrapper.step(s, eval_dt_sfno), None

        @jax.jit
        def eval_full_emulator_rollout(state):
            final, _ = jax.lax.scan(
                _scan_body, state, jnp.arange(n_steps_eval_sfno),
            )
            return final
    else:
        raise ValueError(f"Unknown variant in eval: {variant!r}")

    from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_to_grid

    losses: list[float] = []
    # ``T`` reports the mid-level (~500 hPa) cross-section for direct
    # comparability to WeatherBench T@500.  ``T_sfc`` reports the
    # lowest-sigma-level (near-surface) cross-section so we can place
    # the AIMIP-fleet annual-mean ``tas`` comparison on the same y-axis
    # in the multi-step intercomparison plot.
    per_var_rmse: dict[str, list[float]] = {k: [] for k in ("T", "T_sfc", "u", "v", "p_s")}
    per_var_bias: dict[str, list[float]] = {k: [] for k in ("T", "T_sfc", "u", "v", "p_s")}

    weights = jnp.asarray(grid.weights)

    for sample_idx, (ic, target) in enumerate(zip(ic_states, target_carries)):
        # Multi-step training => loader returns a tuple of K target
        # carries.  Eval only scores against the longest lead.  A single
        # SegmentCarry is a NamedTuple (tuple subclass), so use type(..) is
        # tuple — isinstance would unwrap a single carry to its last FIELD.
        if type(target) is tuple:
            target = target[-1]
        if eval_full_emulator_rollout is not None:
            pred = eval_full_emulator_rollout(ic)
        elif eval_physics_pair is not None:
            non_rad_fn, rad_fn = eval_physics_pair
            pred = spectral_rollout(
                ic, non_rad_fn, grid, sigma, pe_config,
                spec_cfg.dt, n_steps_eval,
                sponge_factor, spectral_filter,
                rad_physics_fn=rad_fn,
                rad_update_interval=eval_rad_interval,
            )
        else:
            pred = spectral_rollout(
                ic, physics_fn, grid, sigma, pe_config,
                spec_cfg.dt, n_steps_eval,
                sponge_factor, spectral_filter,
                forcing_base=(
                    eval_forcings[sample_idx]
                    if eval_forcings is not None else None
                ),
            )
        losses.append(float(
            spectral_state_vs_carry_loss(
                pred, target, grid, sigma, sigma_full, spec_cfg.loss_config,
            )
        ))

        # Per-variable area-weighted RMSE/bias on the Gaussian grid.
        pred_grid = spectral_pe_to_grid(pred, grid, sigma)
        # Target carry stores 3D arrays with axes (n_lat, n_lon, nlev).
        targets_3d = {
            "T": jnp.asarray(target.T),
            "u": jnp.asarray(target.u),
            "v": jnp.asarray(target.v),
        }
        for name, t_arr in targets_3d.items():
            p_arr = pred_grid[name]
            # Mid-level (nlev//2) cross-section for the scorecard so
            # RMSE numbers are comparable to WeatherBench T@500 hPa.
            mid = p_arr.shape[-1] // 2
            per_var_rmse[name].append(
                float(latitude_weighted_rmse(p_arr[..., mid], t_arr[..., mid], weights))
            )
            per_var_bias[name].append(
                float(latitude_weighted_bias(p_arr[..., mid], t_arr[..., mid], weights))
            )
            # Near-surface level (sigma closest to surface) — used
            # in the multi-step intercomparison plot to compare on
            # the same y-axis as the AIMIP fleet's annual-mean ``tas``.
            if name == "T":
                surf = p_arr.shape[-1] - 1  # last sigma level
                per_var_rmse["T_sfc"].append(
                    float(latitude_weighted_rmse(
                        p_arr[..., surf], t_arr[..., surf], weights,
                    ))
                )
                per_var_bias["T_sfc"].append(
                    float(latitude_weighted_bias(
                        p_arr[..., surf], t_arr[..., surf], weights,
                    ))
                )
        # Surface pressure (2D).
        p_s_target = jnp.asarray(target.p_s)
        per_var_rmse["p_s"].append(
            float(latitude_weighted_rmse(pred_grid["p_s"], p_s_target, weights))
        )
        per_var_bias["p_s"].append(
            float(latitude_weighted_bias(pred_grid["p_s"], p_s_target, weights))
        )

    def _agg(lst: list[float]) -> dict[str, float]:
        if not lst:
            return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(sum(lst) / len(lst)),
            "min": float(min(lst)),
            "max": float(max(lst)),
        }

    return {
        "n_eval_days": len(losses),
        "loss": _agg(losses),
        "rmse": {k: _agg(v) for k, v in per_var_rmse.items()},
        "bias": {k: _agg(v) for k, v in per_var_bias.items()},
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, required=True,
        help="Path to the AIMIP suite manifest YAML.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Override config with a minimal smoke-test profile.",
    )
    parser.add_argument(
        "--variants", type=str, default="",
        help=(
            "Comma-separated subset of suite variants to run "
            "(overrides the suite manifest list when set)."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume each variant from the highest-numbered "
            "epoch_NNNN.eqx in its output checkpoint dir, if one "
            "exists.  Used by the chained-resubmit SLURM driver so a "
            "walltime-killed job can continue from where it left off."
        ),
    )
    args = parser.parse_args()

    # Resolve the MPI rank BEFORE configuring logging so that under a
    # data-parallel launch only rank 0 logs at INFO; the other ranks log at
    # WARNING, otherwise every INFO line is duplicated x nranks (#985 papercut).
    from legoesm.training.data_parallel import mpi_rank_size
    _rank, _nproc = mpi_rank_size()

    logging.basicConfig(
        level=logging.INFO if _rank == 0 else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not jax.config.x64_enabled:
        print(
            "ERROR: JAX x64 mode required for spectral transforms.\n"
            "Re-run with: JAX_ENABLE_X64=1 python scripts/run_aimip.py ...",
            file=sys.stderr,
        )
        sys.exit(1)

    suite = _load_yaml(args.suite)
    base = _load_yaml(Path(suite["base"]))
    # Per-suite cfg overrides (used by mini-sweeps that vary nn_seed /
    # sfno_seed / aimip_lr without duplicating the full base YAML).
    # Additive: absent field -> no-op, so existing suites are unaffected.
    if suite.get("cfg_overrides"):
        base = _merge(base, suite["cfg_overrides"])
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = list(suite.get("variants") or ())
    if not variants:
        raise ValueError(f"Suite {args.suite} lists no variants")
    for v in variants:
        if v not in _VALID_VARIANTS:
            raise ValueError(
                f"Unknown AIMIP variant {v!r}; "
                f"expected one of {_VALID_VARIANTS}"
            )

    output_dir = Path(suite.get("output_dir", base.get("output_dir", "results/aimip")))
    if args.smoke:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = base.get("cache_dir", ".cache/aimip_era5")

    results: dict[str, Any] = {
        "suite": str(args.suite),
        "smoke": args.smoke,
        "variants": {},
    }

    # Under an MPI launch (aimip_data_parallel), _train_variant returns on EVERY
    # rank holding the identical replicated model. Only rank 0 evaluates +
    # persists (params.eqx / the scorecard) so the ranks don't clobber those
    # files or multiply the eval work; a barrier after each variant resyncs the
    # ranks before the next variant's collective (DP) training phase.
    # (_rank/_nproc resolved above, before logging setup.)

    for variant in variants:
        overlay = _load_yaml(
            args.suite.parent / f"variant_{variant}.yaml"
        )
        cfg = _merge(base, overlay)
        cfg["aimip_variant"] = variant
        cfg["output_dir"] = str(output_dir)
        if args.smoke:
            cfg = _apply_smoke_overrides(cfg)

        logger.info("=" * 60)
        logger.info(f"AIMIP variant: {variant}")
        logger.info("=" * 60)

        t0 = time.time()
        model, loss_history = _train_variant(
            variant, cfg, cache_dir, resume=args.resume,
        )
        train_elapsed = time.time() - t0

        # Evaluate + persist on rank 0 ONLY (all ranks share the identical
        # replicated model; concurrent writers would clobber params.eqx / the
        # scorecard and N x the eval cost).
        if _rank == 0:
            # Evaluate on BOTH the training windows (in-sample skill, for
            # the AIMIP-fleet annual-mean overlay during 2015-2016) and
            # the held-out test windows (default 2017).  Both reports use
            # an identical loss / rollout horizon so they are directly
            # comparable in absolute K.
            eval_metrics_test = _evaluate_variant(
                variant, model, cfg, cache_dir, period="test",
            )
            eval_metrics_train = _evaluate_variant(
                variant, model, cfg, cache_dir, period="train",
            )
            ckpt_path = output_dir / variant / "params.eqx"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            from legoesm.ml.training import save_checkpoint
            save_checkpoint(model, ckpt_path)

            results["variants"][variant] = {
                "train_loss_history": [float(x) for x in loss_history],
                "train_seconds": train_elapsed,
                "eval_metrics": eval_metrics_test,
                "eval_metrics_train_period": eval_metrics_train,
                "checkpoint": str(ckpt_path),
            }
            # ``loss_history`` is empty on an eval-only resume (all epochs already
            # done, start_epoch == n_epochs -> zero training iterations); guard the
            # [-1] so the scorecard write below still runs (e.g. scorecard regen).
            last_train_loss = loss_history[-1] if loss_history else float("nan")
            logger.info(
                f"{variant}: train_loss[-1]={last_train_loss:.6f}, "
                f"test_loss={eval_metrics_test['loss']['mean']:.6f}, "
                f"test RMSE T={eval_metrics_test['rmse']['T']['mean']:.3f}K "
                f"T_sfc={eval_metrics_test['rmse']['T_sfc']['mean']:.3f}K | "
                f"train RMSE T={eval_metrics_train['rmse']['T']['mean']:.3f}K "
                f"T_sfc={eval_metrics_train['rmse']['T_sfc']['mean']:.3f}K, "
                f"train_time={train_elapsed:.1f}s"
            )

        # Resync ranks before the next variant's collective (DP) training so a
        # fast rank does not enter the next allreduce while rank 0 is still
        # evaluating. Safe here: training + its prefetch threads are done, so no
        # other collective is in flight.
        if _nproc > 1:
            from mpi4py import MPI
            MPI.COMM_WORLD.Barrier()

    # Merge with any existing scorecard so multiple --variants invocations
    # share one results/.../aimip_scorecard.json.  Rank 0 only (the other ranks
    # never populated ``results`` and must not race on the file).
    if _rank == 0:
        scorecard_path = output_dir / "aimip_scorecard.json"
        if scorecard_path.exists():
            with scorecard_path.open() as fh:
                existing = json.load(fh)
            existing_variants = existing.get("variants", {}) if isinstance(existing, dict) else {}
            merged = dict(existing_variants)
            merged.update(results["variants"])
            results["variants"] = merged
        with scorecard_path.open("w") as fh:
            json.dump(results, fh, indent=2)
        logger.info(f"Wrote AIMIP scorecard: {scorecard_path}")

    if _rank != 0:
        return   # non-zero ranks are done; only rank 0 prints the summary

    # Final scorecard summary table -- printed to stdout (and the SLURM
    # log) so the per-variant in-sample (training) and held-out (test)
    # forecast skill is visible at a glance.  Mirrors the structure of
    # ``aimip_scorecard.json`` for the variables that drive the AIMIP
    # comparison plot.
    print("\n" + "=" * 78)
    print("AIMIP scorecard summary")
    print("=" * 78)
    header = (
        f"{'variant':20s}  "
        f"{'period':8s}  "
        f"{'loss':>8s}  "
        f"{'T@500':>8s}  "
        f"{'T_sfc':>8s}  "
        f"{'u':>8s}  "
        f"{'v':>8s}  "
        f"{'p_s':>10s}"
    )
    print(header)
    print("-" * len(header))

    def _row(name: str, period: str, em: dict) -> str:
        def g(k1, k2):
            return (
                em.get(k1, {}).get(k2, {}).get("mean", float("nan"))
                if em else float("nan")
            )
        loss = (em or {}).get("loss", {}).get("mean", float("nan"))
        return (
            f"{name:20s}  {period:8s}  "
            f"{loss:8.4f}  "
            f"{g('rmse','T'):8.3f}  "
            f"{g('rmse','T_sfc'):8.3f}  "
            f"{g('rmse','u'):8.3f}  "
            f"{g('rmse','v'):8.3f}  "
            f"{g('rmse','p_s'):10.1f}"
        )

    for vname, vdata in results["variants"].items():
        em_test = vdata.get("eval_metrics", {})
        em_train = vdata.get("eval_metrics_train_period", {})
        print(_row(vname, "train", em_train))
        print(_row(vname, "test", em_test))
    print("=" * 78)
    print(
        "Units: RMSE in K (T, T_sfc, u, v), Pa (p_s).  ``loss`` is the "
        "combined MSE+CRPS+spectral training objective."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
