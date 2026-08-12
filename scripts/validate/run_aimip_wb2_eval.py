#!/usr/bin/env python
"""WeatherBench-2 forecast-scorecard eval for a run_aimip variant checkpoint.

Scores ONE trained AIMIP variant (``epoch_NNNN.eqx``) against ERA5 on the WB2
headline variables, producing a ``scorecard.json`` with ``model`` +
``persistence`` + ``climatology`` sections in the SAME schema as
``scripts/validate/run_weatherbench_eval.py`` (so ``scripts/plot/plot_wb_scorecard.py``
overlays all three variants + published SOTA with no changes).

Three variants (dispatch-hardened; an unknown ``--variant`` is a hard
``SystemExit``, never a silent default):

* ``classical``  — the sweep-winner classical physics
  (edmf/louis/mcfarlane/sundqvist/xu_randall) + RRTMGP on the spectral PE
  dycore. Rolled via ``neural_gcm_spectral.spectral_rollout`` (with the split
  rad path when ``aimip_rad_update_interval > 1``).
* ``column_nn``  — per-column MLP as the WHOLE physics (aimip_radiation
  "none"). Rolled via ``spectral_rollout`` with the column-MLP physics_fn.
* ``sfno_full``  — the FULL SFNO emulator (replaces dycore+physics, NOT
  sfno_physics). Rolled by scanning ``SFNOPrimitiveEquationModel.step`` at the
  macro step ``dt_sfno``. Leads convert to SFNO steps as
  ``round(lead_hours * 3600 / dt_sfno)``.

Reuse, not reimplementation
---------------------------
The scoring/rollout/climatology numerics are the EXISTING, proven ones. This
driver only:
  1. rebuilds the per-variant untrained pytree skeleton EXACTLY as
     ``run_aimip`` builds it in training (so ``eqx.tree_deserialise_leaves``
     matches), then loads the checkpoint into it;
  2. constructs the grid / sigma / pe_config / sponge / spectral-filter with
     ``run_aimip._build_spectral_config`` + the same helpers ``run_aimip``'s
     ``_evaluate_variant`` uses;
  3. wraps the per-variant rollout as an orchestrator ``rollout_fn`` closing
     over the REAL grid/sigma/pe_config/dt;
  4. calls the SHARED evaluations helpers (``build_forecast_cases``,
     ``climatology_from_cases``, ``run_wb_forecast_eval``,
     ``climatology_scorecard``) and reuses ``run_weatherbench_eval``'s
     ``_nest_scorecard`` / ``_assert_leads_on_dt_grid`` / ``_print_table``.

The arg-parse layer (``build_eval_config_from_args``) is import-light + JAX-free
(login-node testable); all heavy imports live inside ``main``.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import NamedTuple

# Repo root + scripts/run on sys.path so ``import evaluations`` and the
# run_aimip / run_weatherbench_eval helpers resolve. This file lives at
# scripts/validate/; evaluations/ is at the repo root, run_aimip.py at
# scripts/run/. Mirrors the PYTHONPATH the SLURM env exports.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS_RUN = _REPO / "scripts" / "run"
if str(_SCRIPTS_RUN) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_RUN))

VALID_VARIANTS = ("classical", "column_nn", "sfno_full")
_SECONDS_PER_HOUR = 3600.0


class EvalConfig(NamedTuple):
    variant: str
    config_path: str | None       # a pre-merged variant YAML ...
    suite_path: str | None        # ... OR a suite (base+overlay merged like run_aimip)
    checkpoint: str
    leads_hours: tuple
    eval_year: int | None
    n_inits: int
    init_stride_hours: int
    resolution_deg: float
    out: str
    climatology: str
    climatology_samples: int
    sf_override: float | None
    members: tuple
    ensemble_ic_noise: float
    ensemble_noise_scale: float
    mc_dropout_members: int
    probabilistic: bool


VALID_CLIMATOLOGY = ("annual", "window")


def _parse_leads(s):
    if not s:
        raise SystemExit("--leads must list at least one lead time in hours")
    leads = tuple(int(x) for x in str(s).split(",") if x != "")
    if not leads or any(lead <= 0 for lead in leads):
        raise SystemExit(f"--leads must be positive integers (hours), got {s!r}")
    if len(set(leads)) != len(leads):
        # A duplicate lead scores the same valid time twice AND breaks the
        # ensemble's per-case identity: rollout_fn detects "next case" by
        # seeing a lead it has already served, so a repeat inside one case
        # advances the case counter and re-draws the IC/dropout keys mid-case
        # (codex review 2026-08-01).
        raise SystemExit(f"--leads must not repeat a lead time, got {s!r}")
    return leads


def _default_out(variant: str) -> str:
    return f"results/aimip_wbcompare/{variant}/scorecard.json"


def build_eval_config_from_args(argv=None) -> EvalConfig:
    """Parse CLI into an EvalConfig. Import-light + JAX-free (login-node testable).

    An unknown ``--variant`` raises ``SystemExit`` (argparse ``choices``), never
    a silent default — dispatch hardening (CLAUDE.md).
    """
    p = argparse.ArgumentParser(
        description="WB2 forecast-scorecard eval for a run_aimip variant checkpoint.")
    p.add_argument("--variant", required=True, choices=VALID_VARIANTS,
                   help="AIMIP variant the checkpoint was trained as. Unknown "
                        "-> SystemExit (no silent fallback).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config",
                     help="A PRE-MERGED variant YAML the checkpoint was trained "
                          "with (base+overlay already combined).")
    src.add_argument("--suite",
                     help="A suite YAML (config/aimip/wbcompare/suite.yaml). The "
                          "merged config is reconstructed for --variant EXACTLY as "
                          "run_aimip does (base <- cfg_overrides <- variant_<v>.yaml), "
                          "so the eval config matches training without a hand-merge.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to an epoch_NNNN.eqx checkpoint.")
    p.add_argument("--leads", default="24,72,120,240", dest="leads",
                   help="Comma-separated forecast leads [h]; each must be an "
                        "exact multiple of the core dt and the ERA5 cadence.")
    p.add_argument("--eval-year", type=int, default=None, dest="eval_year",
                   help="Eval calendar year (default: YAML eval_years[0] or "
                        "eval_year).")
    p.add_argument("--n-inits", type=int, default=8, dest="n_inits")
    p.add_argument("--init-stride-hours", type=int, default=24,
                   dest="init_stride_hours")
    p.add_argument("--resolution-deg", type=float, default=1.5,
                   dest="resolution_deg",
                   help="WB2 target-grid resolution [deg] (default 1.5).")
    p.add_argument("--climatology", choices=VALID_CLIMATOLOGY, default="annual",
                   help="Climatology reference. 'annual' (default) averages "
                        "--climatology-samples snapshots spread across the eval "
                        "year. 'window' is the legacy eval-window mean, which is "
                        "a LOCAL mean unless the inits span a year and then "
                        "scores below the persistence floor - kept only for "
                        "reproducing old scorecards.")
    p.add_argument("--climatology-samples", type=int, default=73,
                   dest="climatology_samples",
                   help="Snapshots for --climatology annual (73 ~ every 5 days).")
    p.add_argument("--spectral-filter-strength", type=float, default=None,
                   dest="sf_override",
                   help="OVERRIDE the post-step spectral filter at eval "
                        "(default: the trained pe_config value). Larger = "
                        "WEAKER damping (it is the filter value at n_max). "
                        "Use to test whether the model is over-smoothed; note "
                        "this is a deliberate train/eval mismatch and must be "
                        "labelled as such in any comparison.")
    p.add_argument("--member", action="append", default=[], dest="members",
                   help="EXTRA member checkpoint(s) for a multi-seed ensemble "
                        "(repeatable). Member forecasts are averaged at each "
                        "lead — ArchesWeather-Mx4. Weights are NOT averaged: "
                        "that is meaningless across independent inits.")
    p.add_argument("--ensemble-ic-noise", type=float, default=0.0,
                   dest="ensemble_ic_noise",
                   help="Per-member IC perturbation, as a multiple of the 6 h "
                        "residual scales. 0 = seed spread only.")
    p.add_argument("--ensemble-noise-scale", type=float, default=1.0,
                   dest="ensemble_noise_scale",
                   help="Multiplies --ensemble-ic-noise (ArchesWeatherGen's "
                        "rho; they use 1.05 ~ the overfitting fraction) to "
                        "cure under-dispersion.")
    p.add_argument("--mc-dropout-members", type=int, default=0,
                   dest="mc_dropout_members",
                   help="MC-Dropout members drawn PER checkpoint (U-Cast, "
                        "arXiv:2604.09041). Total ensemble = "
                        "n_checkpoints x this. Requires a network trained "
                        "with sfno_dropout > 0; 0 = off (deterministic "
                        "forward pass, byte-identical to before).")
    p.add_argument("--probabilistic", action="store_true",
                   help="Score CRPS / spread / spread-skill from the member "
                        "ensemble instead of only the ensemble-mean RMSE. "
                        "NOTE: this also switches the ensemble mean from "
                        "mean-then-diagnose to diagnose-then-mean, which is "
                        "the more correct construct but differs for the "
                        "NONLINEAR diagnostics (z500, mslp) — a deterministic "
                        "number from this mode is NOT comparable to one "
                        "without it.")
    p.add_argument("--out", default=None,
                   help="Scorecard JSON path (default "
                        "results/aimip_wbcompare/<variant>/scorecard.json).")
    a = p.parse_args(argv)

    if a.ensemble_ic_noise < 0.0:
        raise SystemExit(
            f"--ensemble-ic-noise must be >= 0, got {a.ensemble_ic_noise}")
    if not a.ensemble_noise_scale > 0.0:
        raise SystemExit(
            f"--ensemble-noise-scale must be > 0, got {a.ensemble_noise_scale}")
    if a.members and a.variant != "sfno_full":
        raise SystemExit(
            "--member (multi-seed ensemble) is implemented for sfno_full only; "
            f"got --variant {a.variant}.")
    if a.mc_dropout_members < 0:
        raise SystemExit(
            f"--mc-dropout-members must be >= 0, got {a.mc_dropout_members}")
    if a.mc_dropout_members and a.variant != "sfno_full":
        raise SystemExit(
            "--mc-dropout-members is implemented for sfno_full only; "
            f"got --variant {a.variant}.")
    if a.probabilistic:
        # CRPS needs >= 2 members. Seeds contribute len(members)+1 (the primary
        # checkpoint plus each --member); dropout multiplies that.
        n_ens = (1 + len(a.members)) * max(1, a.mc_dropout_members)
        if n_ens < 2:
            raise SystemExit(
                "--probabilistic needs an ensemble of at least 2 members; got "
                f"{n_ens}. Add --member checkpoints or --mc-dropout-members "
                ">= 2. (With one member the CRPS degenerates to the MAE.)")

    if a.climatology_samples < 1:
        raise SystemExit(
            f"--climatology-samples must be >= 1, got {a.climatology_samples}")

    if a.n_inits < 1:
        raise SystemExit(f"--n-inits must be >= 1, got {a.n_inits}")
    if a.init_stride_hours < 1:
        raise SystemExit(
            f"--init-stride-hours must be >= 1, got {a.init_stride_hours}")
    if a.resolution_deg <= 0:
        raise SystemExit(
            f"--resolution-deg must be > 0, got {a.resolution_deg}")

    out = a.out if a.out is not None else _default_out(a.variant)
    return EvalConfig(
        variant=a.variant, config_path=a.config, suite_path=a.suite,
        checkpoint=a.checkpoint,
        leads_hours=_parse_leads(a.leads), eval_year=a.eval_year,
        n_inits=a.n_inits, init_stride_hours=a.init_stride_hours,
        resolution_deg=a.resolution_deg, out=out,
        climatology=a.climatology,
        climatology_samples=a.climatology_samples,
        sf_override=a.sf_override,
        members=tuple(a.members),
        ensemble_ic_noise=a.ensemble_ic_noise,
        ensemble_noise_scale=a.ensemble_noise_scale,
        mc_dropout_members=a.mc_dropout_members,
        probabilistic=bool(a.probabilistic))


def merged_cfg_from_suite(run_aimip, suite_path, variant):
    """Reconstruct the merged config for ``variant`` from a suite, EXACTLY as
    run_aimip's suite loader does: base <- cfg_overrides <- variant_<v>.yaml,
    then stamp ``aimip_variant``. Reuses run_aimip's own ``_load_yaml``/``_merge``
    (no re-implemented merge) so the eval config is byte-identical to training's.
    """
    from pathlib import Path
    suite = run_aimip._load_yaml(Path(suite_path))
    cfg = run_aimip._load_yaml(Path(suite["base"]))
    if suite.get("cfg_overrides"):
        cfg = run_aimip._merge(cfg, suite["cfg_overrides"])
    overlay = run_aimip._load_yaml(Path(suite_path).parent / f"variant_{variant}.yaml")
    cfg = run_aimip._merge(cfg, overlay)
    cfg["aimip_variant"] = variant
    return cfg


def leads_to_sfno_steps(lead_hours: int, dt_sfno: float) -> int:
    """Convert a forecast lead [h] to SFNO macro steps at ``dt_sfno`` [s].

    Mirrors ``run_aimip._evaluate_variant``'s sfno_full path:
    ``max(1, round(lead_hours * 3600 / dt_sfno))``. Pure + JAX-free so the CLI
    test can exercise it on the login node.
    """
    if lead_hours <= 0:
        raise ValueError(f"lead_hours must be positive, got {lead_hours}")
    if dt_sfno <= 0:
        raise ValueError(f"dt_sfno must be positive, got {dt_sfno}")
    return max(1, int(round(lead_hours * _SECONDS_PER_HOUR / dt_sfno)))


def _load_run_aimip():
    """Import scripts/run/run_aimip.py by file path (its module name is not a
    package; mirrors scripts/cluster/wb_forecast/check_config.sbatch)."""
    entry = _SCRIPTS_RUN / "run_aimip.py"
    spec = importlib.util.spec_from_file_location("run_aimip", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sfno_full_norm_stats_path(checkpoint_path):
    """Resolve the sfno_full normalization sidecar next to a checkpoint.

    train_sfno_full_spectral writes ``norm_stats.npz`` into
    ``config.checkpoint_dir`` (== the directory holding ``epoch_NNNN.eqx``), so
    the sidecar is the checkpoint's parent dir + ``norm_stats.npz``.  Pure path
    logic (no jax / no I/O) so the login-safe CLI test can exercise it.
    """
    if checkpoint_path is None:
        raise ValueError(
            "sfno_full eval needs the checkpoint path to locate norm_stats.npz "
            "(the per-channel Z-score sidecar written at train time); got None."
        )
    return str(Path(checkpoint_path).resolve().parent / "norm_stats.npz")


def _load_sfno_full_norm_stats(checkpoint_path):
    """Load the sfno_full per-channel Z-score stats sidecar for eval.

    Fails LOUDLY if the sidecar is missing: sfno_full is trained WITH
    normalization, so evaluating without the matching stats would feed the
    network raw ~1e5-magnitude channels (garbage / NaN) — a silent
    use_normalization=False fallback would be worse than a clear error.
    """
    import os

    from legoesm.ml.normalization import load_normalization_stats

    path = sfno_full_norm_stats_path(checkpoint_path)
    if not os.path.exists(path):
        raise SystemExit(
            f"sfno_full normalization sidecar not found: {path}\n"
            f"train_sfno_full_spectral writes norm_stats.npz next to the "
            f"checkpoints; re-run training (it is created per run) or point "
            f"--checkpoint at a run directory that contains it."
        )
    return load_normalization_stats(path)


def assert_arch_matches_checkpoint(arch, checkpoint_path, member_paths=()):
    """Refuse an architecture that disagrees with the checkpoint's sidecar.

    ``train_sfno_full_spectral`` writes ``sfno_arch.json`` next to the
    checkpoints.  Most architecture fields would fail loudly on a leaf-shape
    mismatch anyway; ``dropout`` would NOT — it is a static field, so a p=0.1
    checkpoint deserialises cleanly into a p=0.0 skeleton and then produces a
    silently mis-calibrated MC-Dropout ensemble (or a silently deterministic
    one).  Absent sidecar = a checkpoint from before this existed: skipped
    with a warning, never a hard failure.
    """
    import json
    import logging
    from pathlib import Path

    log = logging.getLogger("aimip_wb2_eval")
    for path in (str(checkpoint_path), *[str(p) for p in member_paths]):
        sidecar = Path(path).resolve().parent / "sfno_arch.json"
        if not sidecar.exists():
            log.warning(
                "no sfno_arch.json next to %s — cannot verify the checkpoint "
                "was trained with dropout=%.3g; scores are UNVERIFIED against "
                "the training architecture", path, float(arch.dropout))
            continue
        rec = json.loads(sidecar.read_text())
        fields = ("embed_dim", "n_blocks", "mlp_expansion", "dropout",
                  "in_channels", "out_channels", "residual_prediction")
        missing = [f for f in fields if f not in rec]
        if missing:
            # An incomplete sidecar verifies less than it appears to. Say so:
            # a stale record that happens to omit ``dropout`` would otherwise
            # read as a clean check (codex review 2026-08-01).
            log.warning(
                "%s does not record %s — those fields are UNVERIFIED for %s",
                sidecar, missing, path)
        for field in fields:
            want = getattr(arch, field)
            got = rec.get(field)
            if got is None:
                continue
            if float(got) != float(want):
                raise SystemExit(
                    f"architecture mismatch for {path}: trained with "
                    f"{field}={got}, evaluating as {field}={want} "
                    f"(from {sidecar}). Fix the --suite/--config so the "
                    f"eval reconstructs the architecture that was trained."
                )


def _default_schemes() -> dict[str, str]:
    """The same default-scheme table the trainer uses (one source).

    Eval and training each carried their own copy and had already drifted:
    microphysics defaulted to "none" here while training used a real scheme,
    so an omitted key scored a different model than it trained.
    """
    from legoesm.training.aimip_params import CLASSICAL_DEFAULT_SCHEMES
    return CLASSICAL_DEFAULT_SCHEMES


def _build_skeleton(variant, cfg, spec_cfg, grid):
    """Reconstruct the UNTRAINED pytree skeleton exactly as run_aimip builds it
    in training, so ``eqx.tree_deserialise_leaves(checkpoint, skeleton)`` matches.

    * classical  -> ``AIMIPClassicalParams.from_defaults(...)``
      (run_aimip.py::_train_aimip_classical, ~L318).
    * column_nn  -> ``build_column_physics(nlev, hidden_dim, n_layers, key)``
      (neural_gcm_spectral.train_column_mlp_spectral, ~L3453).
    * sfno_full  -> ``SFNO(SFNOConfig(...), grid, key)``
      (neural_gcm_spectral.train_sfno_full_spectral, ~L3153).
    """
    if variant == "classical":
        from legoesm.training.aimip_params import AIMIPClassicalParams
        # Matches _train_aimip_classical: spatial_surface + init_std + seed all
        # read from the SAME cfg keys with the SAME defaults.
        skeleton = AIMIPClassicalParams.from_defaults(
            spatial_surface=bool(cfg.get("aimip_spatial_surface", False)),
            spatial_init_std=float(cfg.get("aimip_spatial_init_std", 0.0)),
            spatial_seed=int(cfg.get("aimip_spatial_seed", 0)),
        )
        # aimip_trainable_schemes wraps the checkpoint in an
        # AIMIPTrainableBundle (legacy leaves + spec-driven scheme params);
        # deserialising such a checkpoint into a bare AIMIPClassicalParams
        # fails on leaf mismatch. Mirror run_aimip's construction exactly
        # (same cfg keys, same field-level ownership exclusion) so the eval
        # skeleton matches what was trained.
        if cfg.get("aimip_trainable_schemes"):
            from legoesm.training.aimip_params import (
                AIMIPTrainableBundle,
                aimip_legacy_owned_fields,
                aimip_scheme_keys_for,
            )
            from legoesm.training.param_collector import (
                build_trainable_params,
            )
            _tier = cfg.get("aimip_trainable_schemes")
            _keys = aimip_scheme_keys_for(
                convection=str(cfg.get("aimip_convection", "tiedtke")),
                turbulence=str(cfg.get("aimip_turbulence", "louis")),
                gwd=str(cfg.get("aimip_gwd", "mcfarlane")),
                microphysics=str(cfg.get(
                    "aimip_microphysics", _default_schemes()["microphysics"])),
                radiation=str(cfg.get("aimip_radiation", "rrtmgp")),
                cloud=str(cfg.get("aimip_cloud", "xu_randall")),
            )
            schemes = build_trainable_params(
                active_scheme_keys=_keys,
                tier=(_tier if isinstance(_tier, str) else "extended"),
                exclude=tuple(sorted(aimip_legacy_owned_fields(
                    cloud_scheme=str(cfg.get("aimip_cloud", "xu_randall"))))),
            )
            skeleton = AIMIPTrainableBundle(
                classical=skeleton, schemes=schemes)
        return skeleton

    if variant == "column_nn":
        # Same builder the trainer uses (training.model_registry), so the
        # skeleton cannot drift from what wrote the checkpoint. residual_scale
        # is left at the factory default, as run_aimip leaves it.
        from legoesm.training.model_registry import build_variant
        return build_variant(
            "column_nn", nlev=spec_cfg.n_levels,
            seed=int(cfg.get("nn_seed", 0)),
            overrides={"nn_hidden_dim": int(cfg.get("nn_hidden_dim", 256)),
                       "n_layers": int(cfg.get("nn_n_layers", 4))},
        )

    if variant == "sfno_full":
        # Same builder the trainer uses, so the skeleton cannot drift from what
        # wrote the checkpoint (this block used to restate the channel layout
        # and every architecture field by hand).
        from legoesm.training.model_registry import build_variant
        return build_variant(
            "sfno_full", nlev=spec_cfg.n_levels, grid=grid,
            seed=int(cfg.get("sfno_seed", 0)),
            overrides={
                "sfno_embed_dim": spec_cfg.sfno_embed_dim,
                "sfno_n_blocks": spec_cfg.sfno_n_blocks,
                "sfno_mlp_expansion": spec_cfg.sfno_mlp_expansion,
                "sfno_dropout": spec_cfg.sfno_dropout,
                "sfno_history_steps": int(
                    getattr(spec_cfg, "sfno_history_steps", 0) or 0),
            },
        )

    # Unreachable: build_eval_config_from_args already gates --variant. Kept as a
    # hard error (dispatch hardening) so a future caller of _build_skeleton with a
    # bad variant fails loudly instead of returning None.
    raise ValueError(f"Unknown variant in _build_skeleton: {variant!r}")


def _build_rollout_fn(variant, trained, cfg, spec_cfg, grid, sigma, pe_config,
                      sponge_factor, spectral_filter, checkpoint_path=None,
                      ensemble_ic_noise=0.0, ensemble_noise_scale=1.0,
                      sf_override=None, mc_dropout_members=0,
                      probabilistic=False, member_paths=()):
    """Return ``(rollout_fn, dt_for_orchestrator)`` for the variant.

    The orchestrator calls ``rollout_fn(state, physics_fn, grid_, sigma_,
    pe_config_, dt_, n_steps, forcing_base=...)`` and computes ``n_steps`` from
    ``lead_hours * 3600 / dt_for_orchestrator``. We close over the REAL
    grid/sigma/pe_config/dt and ignore the orchestrator's placeholder args
    (which are the None model/grid/sigma/pe_config we hand it) except ``n_steps``.

    * classical / column_nn: dt_for_orchestrator = spec_cfg.dt (the dycore
      micro-step); n_steps = lead*3600/dt is the dycore step count.
    * sfno_full: dt_for_orchestrator = dt_sfno (the SFNO MACRO step); the
      orchestrator then hands us n_steps already in SFNO-step units, so the
      lead->step conversion (leads_to_sfno_steps) is done consistently by the
      orchestrator's own ``round(lead*3600/dt)`` with dt = dt_sfno.
    """
    import jax
    import jax.numpy as jnp

    from legoesm.training.aimip_params import (
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.campaign_driver import parse_bool_flag
    from legoesm.training.neural_gcm_spectral import (
        make_column_mlp_spectral_physics,
        spectral_rollout,
    )

    if variant == "classical":
        rad_update_interval = int(cfg.get("aimip_rad_update_interval", 1))
        split_rad = rad_update_interval > 1
        built = make_aimip_classical_spectral_physics(
            trained, grid, spec_cfg.dt,
            radiation=str(cfg.get("aimip_radiation", "gray")),
            rad_update_interval_steps=rad_update_interval,
            convection_scheme=str(cfg.get("aimip_convection", "tiedtke")),
            turbulence_scheme=str(cfg.get("aimip_turbulence", "louis")),
            surface_bulk_scheme=str(
                cfg.get("aimip_surface_bulk_scheme", "constant")),
            gwd_scheme=str(cfg.get("aimip_gwd", "mcfarlane")),
            microphysics_scheme=str(cfg.get(
                "aimip_microphysics", _default_schemes()["microphysics"])),
            allow_unfilled_families=parse_bool_flag(
    cfg.get("aimip_allow_unfilled_families", False)),
            cloud_scheme=str(cfg.get("aimip_cloud", "xu_randall")),
            land_mask=None,
            split_rad=split_rad,
        )
        if isinstance(built, tuple):
            non_rad_fn, rad_fn = built

            def rollout_fn(state, physics_fn, grid_, sigma_, pe_config_, dt_,
                           n_steps, *, forcing_base=None):
                return spectral_rollout(
                    state, non_rad_fn, grid, sigma, pe_config,
                    spec_cfg.dt, int(n_steps),
                    sponge_factor, spectral_filter,
                    rad_physics_fn=rad_fn,
                    rad_update_interval=rad_update_interval,
                )
        else:
            physics_fn = built

            def rollout_fn(state, _physics_fn, grid_, sigma_, pe_config_, dt_,
                           n_steps, *, forcing_base=None):
                return spectral_rollout(
                    state, physics_fn, grid, sigma, pe_config,
                    spec_cfg.dt, int(n_steps),
                    sponge_factor, spectral_filter,
                    forcing_base=forcing_base,
                )
        return rollout_fn, spec_cfg.dt

    if variant == "column_nn":
        physics_fn = make_column_mlp_spectral_physics(trained, grid)

        def rollout_fn(state, _physics_fn, grid_, sigma_, pe_config_, dt_,
                       n_steps, *, forcing_base=None):
            return spectral_rollout(
                state, physics_fn, grid, sigma, pe_config,
                spec_cfg.dt, int(n_steps),
                sponge_factor, spectral_filter,
                forcing_base=forcing_base,
            )
        return rollout_fn, spec_cfg.dt

    if variant == "sfno_full":
        from legoesm.atmosphere.dynamics.neural.sfno_pe import (
            SFNOPrimitiveEquationConfig,
            SFNOPrimitiveEquationModel,
        )
        from legoesm.ml.channel_packing import PE3DChannelSpec
        from legoesm.ml.sfno import SFNOConfig
        channels = PE3DChannelSpec(nlev=spec_cfg.n_levels).n_channels
        dt_sfno = float(cfg.get("dt_sfno", 21600.0))
        # train_sfno_full_spectral trains with per-channel Z-score
        # normalization ON and writes the stats to norm_stats.npz next to the
        # checkpoints.  Reload that SAME sidecar so eval applies the IDENTICAL
        # transform (state_update denormalises the network output as a full
        # state — correct).  The eqx checkpoint carries only the SFNO leaves,
        # not the wrapper's norm_stats, so the sidecar is the source of truth.
        norm_stats = _load_sfno_full_norm_stats(checkpoint_path)
        pe_cfg = SFNOPrimitiveEquationConfig(
            sfno_config=SFNOConfig(
                in_channels=channels,
                out_channels=channels,
                embed_dim=spec_cfg.sfno_embed_dim,
                n_blocks=spec_cfg.sfno_n_blocks,
                mlp_expansion=spec_cfg.sfno_mlp_expansion,
                residual_prediction=False,
                dropout=spec_cfg.sfno_dropout,
            ),
            mode="state_update",
            dt_sfno=dt_sfno,
            correct_mass=True,
            correct_moisture_budget=False,
            # Must MATCH _train_sfno_full_loop's pe_emulator_cfg exactly.
            # Both read them off ``pe_config``, which
            # run_aimip._build_spectral_config derives from the merged suite,
            # so the eval rolls what training rolled (clip_q = ACE2 corrector
            # step 1; the spectral filter damps the vor/div cascade that
            # otherwise NaNs the rollout by 72 h).
            clip_q=True,
            spectral_filter_strength=(
                pe_config.spectral_filter_strength if sf_override is None
                else float(sf_override)),
            spectral_filter_order=pe_config.spectral_filter_order,
            use_normalization=True,
        )
        # ``trained`` is a LIST: one entry per ensemble member. A single
        # checkpoint gives the deterministic path, byte-identical to before.
        _members = trained if isinstance(trained, (list, tuple)) else [trained]
        wrappers = [
            SFNOPrimitiveEquationModel(
                grid=grid, sigma_coord=sigma, config=pe_cfg, sfno_model=m,
                norm_stats=norm_stats,
            )
            for m in _members
        ]

        # Refuse an architecture the checkpoint was NOT trained with. Matters
        # most for ``dropout``, the one field a mismatch cannot fail loudly on.
        if checkpoint_path is not None:
            assert_arch_matches_checkpoint(
                pe_cfg.sfno_config, checkpoint_path,
                member_paths=member_paths)

        _n_mc = int(mc_dropout_members)
        if _n_mc > 0 and float(pe_cfg.sfno_config.dropout) <= 0.0:
            raise SystemExit(
                "--mc-dropout-members requires a checkpoint trained with "
                "sfno_dropout > 0; this suite resolves sfno_dropout=0.0, so "
                "every dropout member would be the SAME forecast. (Applying "
                "dropout at inference to a net trained without it is not the "
                "U-Cast recipe — there dropout is active during training too.)"
            )

        def _roll_one(wrapper, s0, n, key=None):
            """One member trajectory. ``key`` -> a FRESH dropout mask at every
            macro step (``fold_in`` on the step index): reusing one mask for
            the whole trajectory freezes a single perturbed network and
            under-disperses the ensemble."""
            def _body(s, i):
                step_key = None if key is None else jax.random.fold_in(key, i)
                return wrapper.step(s, dt_sfno, key=step_key), None
            final, _ = jax.lax.scan(_body, s0, jnp.arange(n))
            return final

        # Per-member IC perturbation (ArchesWeatherGen's dispersion fix, adapted
        # to our IC-perturbation ensemble instead of a generative head). The
        # amplitude is ensemble_ic_noise x noise_scale: the second factor is
        # their rho, "slightly higher than 1 ... roughly corresponds to the
        # percentage of overfitting", which inflates member spread to cure the
        # under-dispersion an overfitted deterministic model produces.
        _ic_noise = float(ensemble_ic_noise) * float(ensemble_noise_scale)
        # Per-CASE perturbation identity. The orchestrator calls rollout_fn once
        # per (case, lead); leads for one case must share a draw, so the counter
        # advances only when a NEW case's shortest lead comes round. Tracked in
        # a mutable cell because rollout_fn is called positionally by the
        # orchestrator and cannot take a case argument.
        _case_id = [0]
        _seen_steps: set = set()

        def _perturb(state, j, d):
            """IC draw for member ``(checkpoint j, dropout draw d)`` of the
            current case.  Off (identity) when ``--ensemble-ic-noise`` is 0.

            The key must vary per (member, CASE) — keying on member alone gave
            every initialization the IDENTICAL noise field, i.e. a fixed
            member-specific spatial artifact rather than an ensemble (codex
            review 2026-07-31) — and must stay CONSTANT across leads within
            one case, so a trajectory is a coherent forecast rather than
            re-noised at each lead; hence the case counter, not a global one.
            ``d`` enters the key so the K x N members carry K x N independent
            IC draws rather than K (codex review 2026-08-01).

            With MC dropout OFF there is only one draw per checkpoint, and the
            key is the LEGACY form (no ``d`` fold) so every previously scored
            IC-perturbation ensemble reproduces bit-for-bit.
            """
            if _ic_noise <= 0.0:
                return state
            from legoesm.training.losses import LossConfig
            from legoesm.training.neural_gcm_spectral import (
                perturb_spectral_ic,
            )
            base = jax.random.PRNGKey(1000 + j)
            key = jax.random.fold_in(
                (base if _n_mc <= 0 else jax.random.fold_in(base, d)),
                int(_case_id[0]))
            return perturb_spectral_ic(
                state, key, LossConfig(ensemble_ic_noise=_ic_noise))

        def rollout_fn(state, _physics_fn, grid_, sigma_, pe_config_, dt_,
                       n_steps, *, forcing_base=None):
            # The orchestrator passes dt=dt_sfno, so n_steps is already in SFNO
            # macro-step units (round(lead*3600/dt_sfno)). Clamp to >=1 to match
            # _evaluate_variant (a sub-macro-step lead still runs one SFNO step).
            n = max(1, int(n_steps))
            if _ic_noise > 0.0 or _n_mc > 0:
                # A repeat of an already-seen lead means the orchestrator has
                # moved on to the next case.
                if int(n) in _seen_steps:
                    _case_id[0] += 1
                    _seen_steps.clear()
                _seen_steps.add(int(n))
            finals = []
            for j, w in enumerate(wrappers):
                # The IC draw must vary per MEMBER, not per checkpoint: with
                # --mc-dropout-members it used to be drawn once outside the
                # dropout loop, so the K x N ensemble carried only K
                # independent IC perturbations and was correlated within each
                # checkpoint (codex review).
                for d in range(max(1, _n_mc)):
                    s0 = _perturb(state, j, d)
                    if _n_mc > 0:
                        # U-Cast's K x N ensemble: K checkpoints (deep
                        # ensemble) x N MC-Dropout draws each. The base key
                        # varies per (checkpoint j, draw d, CASE) and is held
                        # FIXED across leads within a case, so one member is a
                        # coherent trajectory rather than a re-randomised
                        # field per lead.
                        mkey = jax.random.fold_in(
                            jax.random.fold_in(
                                jax.random.PRNGKey(2000 + j), d),
                            int(_case_id[0]))
                        finals.append(_roll_one(w, s0, n, key=mkey))
                    else:
                        finals.append(_roll_one(w, s0, n))
            if len(finals) == 1:
                return finals[0]
            if probabilistic:
                # Hand the orchestrator every MEMBER: it diagnoses each, means
                # the FIELDS (correct for the nonlinear z500/mslp diagnostics)
                # and adds CRPS / spread / spread-skill. Deliberately a
                # different reduction from the legacy branch below — hence the
                # opt-in flag and its "not comparable" warning.
                return finals
            # ENSEMBLE MEAN in the model's own output space (the spectral
            # state), matching "averaging their outputs at inference time".
            # CAVEAT: z500 and mslp are diagnosed NONLINEARLY from the state, so
            # mean-then-diagnose != diagnose-then-mean for those two. Averaging
            # the state is the construct the reference uses and keeps the
            # orchestrator interface (which consumes a state) intact.
            return jax.tree.map(lambda *xs: sum(xs) / float(len(xs)), *finals)

        # Orchestrator drives leads in dt_sfno units for this variant.
        return rollout_fn, dt_sfno

    raise ValueError(f"Unknown variant in _build_rollout_fn: {variant!r}")


def main(argv=None, ds=None):
    """Run the WB2 scorecard for one variant. ``ds`` injects a pre-opened ERA5
    dataset (network-free tests)."""
    cfg_args = build_eval_config_from_args(argv)

    import json
    import logging
    import os

    import equinox as eqx
    import yaml

    # Spectral cores require float64 (SH transforms). Enable x64 at the entry
    # point, before any grid/array construction, so a bare invocation runs
    # correctly without the caller exporting JAX_ENABLE_X64=1.
    import jax
    jax.config.update("jax_enable_x64", True)

    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        compute_spectral_filter,
        compute_sponge_factor,
    )
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.era5_to_state import TrainingERA5Config

    from evaluations.wb_era5_cases import (
        build_forecast_cases,
        climatology_from_cases,
        climatology_from_era5,
        climatology_scorecard,
    )
    from evaluations.wb_orchestrator import run_wb_forecast_eval

    # Reuse run_weatherbench_eval's helpers (nest / lead-grid assert / table)
    # rather than copying them — loaded by file path (module is not a package).
    _wbe_entry = Path(__file__).resolve().parent / "run_weatherbench_eval.py"
    _wbe_spec = importlib.util.spec_from_file_location("run_wb_eval", _wbe_entry)
    _wbe = importlib.util.module_from_spec(_wbe_spec)
    _wbe_spec.loader.exec_module(_wbe)
    _nest_scorecard = _wbe._nest_scorecard
    _assert_leads_on_dt_grid = _wbe._assert_leads_on_dt_grid
    _print_table = _wbe._print_table

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("aimip_wb2_eval")

    run_aimip = _load_run_aimip()

    # Config source: a pre-merged --config, or reconstruct the training merge
    # from --suite + --variant (base <- cfg_overrides <- variant overlay).
    if cfg_args.suite_path is not None:
        yml = merged_cfg_from_suite(run_aimip, cfg_args.suite_path, cfg_args.variant)
        log.info("merged config for variant=%s from suite=%s",
                 cfg_args.variant, cfg_args.suite_path)
    else:
        yml = yaml.safe_load(open(cfg_args.config_path))
    # Eval year: explicit flag > YAML eval_years[0] > YAML eval_year.
    if cfg_args.eval_year is not None:
        eval_year = cfg_args.eval_year
    elif yml.get("eval_years"):
        eval_year = int(yml["eval_years"][0])
    elif yml.get("eval_year") is not None:
        eval_year = int(yml["eval_year"])
    else:
        raise SystemExit(
            "eval year not resolvable: pass --eval-year or set eval_years / "
            "eval_year in the config YAML.")
    cadence = int(yml.get("era5_cadence_hours", 6))

    spec_cfg = run_aimip._build_spectral_config(yml)
    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(
        spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top,
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

    # --- rebuild the untrained skeleton + load the checkpoint into it ---
    if not os.path.exists(cfg_args.checkpoint):
        raise SystemExit(f"--checkpoint not found: {cfg_args.checkpoint}")
    skeleton = _build_skeleton(cfg_args.variant, yml, spec_cfg, grid)
    trained = eqx.tree_deserialise_leaves(cfg_args.checkpoint, skeleton)

    # Multi-seed ensemble: each --member is an INDEPENDENTLY seeded training
    # run of the same architecture, loaded into its own copy of the skeleton.
    # A missing member is a hard error — silently scoring a 3-member ensemble
    # that was reported as 4 would be an unattributable skill difference.
    if cfg_args.members:
        for extra in cfg_args.members:
            if not os.path.exists(extra):
                raise SystemExit(f"--member not found: {extra}")
        trained = [trained] + [
            eqx.tree_deserialise_leaves(
                extra, _build_skeleton(cfg_args.variant, yml, spec_cfg, grid))
            for extra in cfg_args.members
        ]
        log.info(
            "multi-seed ensemble: %d member(s); forecasts averaged per lead "
            "(ic_noise=%.4g x noise_scale=%.4g)", len(trained),
            cfg_args.ensemble_ic_noise, cfg_args.ensemble_noise_scale)

    rollout_fn, dt_orch = _build_rollout_fn(
        cfg_args.variant, trained, yml, spec_cfg, grid, sigma, pe_config,
        sponge_factor, spectral_filter, checkpoint_path=cfg_args.checkpoint,
        ensemble_ic_noise=cfg_args.ensemble_ic_noise,
        ensemble_noise_scale=cfg_args.ensemble_noise_scale,
        sf_override=cfg_args.sf_override,
        mc_dropout_members=cfg_args.mc_dropout_members,
        probabilistic=cfg_args.probabilistic,
        member_paths=cfg_args.members,
    )
    # The orchestrator computes n_steps = lead*3600/dt_orch; leads must land on
    # that grid (classical/column_nn: dt=dycore dt; sfno_full: dt=dt_sfno).
    _assert_leads_on_dt_grid(cfg_args.leads_hours, dt_orch)

    # --- ERA5 cases (ICs + forcing + WB2-grid verification) ---
    era5_cfg = TrainingERA5Config(dt_hours=cadence)._replace(
        zarr_store=yml["era5_zarr"])
    cases = build_forecast_cases(
        era5_cfg, grid, sigma,
        leads_hours=cfg_args.leads_hours, eval_year=eval_year,
        n_inits=cfg_args.n_inits, init_stride_hours=cfg_args.init_stride_hours,
        resolution_deg=cfg_args.resolution_deg, ds=ds)
    log.info(
        "WB2 AIMIP eval: variant=%s year=%d inits=%d leads=%s res=%.3gdeg "
        "dt_orch=%.1fs", cfg_args.variant, eval_year, len(cases),
        list(cfg_args.leads_hours), cfg_args.resolution_deg, dt_orch)

    # Annual-sampled climatology by default. The eval-window mean is a LOCAL
    # mean whenever the inits do not span a year: with the default 8 inits at
    # 24 h stride it scored 58.7 m for z500 at 24 h, BELOW the 63.9 m
    # persistence floor — impossible for a real climatology at day 1 — versus
    # 107.8 m sampled across the year (measured 2026-07-28).
    if cfg_args.climatology == "window":
        clim = climatology_from_cases(cases)
    else:
        clim = climatology_from_era5(
            era5_cfg, eval_year=eval_year,
            n_samples=cfg_args.climatology_samples,
            resolution_deg=cfg_args.resolution_deg, ds=ds)

    # --- model + persistence via the SHARED orchestrator (identical masking) ---
    # physics_fn/grid/sigma/pe_config passed as None placeholders: our rollout_fn
    # closes over the real ones. dt = dt_orch so n_steps is computed in the right
    # unit for each variant.
    model_sc = run_wb_forecast_eval(
        None, grid, sigma, None, dt_orch, cases, cfg_args.leads_hours, clim,
        resolution_deg=cfg_args.resolution_deg, rollout_fn=rollout_fn)
    persist_sc = run_wb_forecast_eval(
        None, grid, sigma, None, dt_orch, cases, cfg_args.leads_hours, clim,
        resolution_deg=cfg_args.resolution_deg,
        rollout_fn=lambda s, *a, **k: s)
    clim_sc = climatology_scorecard(
        cases, cfg_args.leads_hours, clim, resolution_deg=cfg_args.resolution_deg)

    meta = {
        "variant": cfg_args.variant,
        "config": cfg_args.config_path,
        "suite": cfg_args.suite_path,
        "checkpoint": cfg_args.checkpoint,
        "eval_year": eval_year,
        "leads_hours": list(cfg_args.leads_hours),
        "n_inits": len(cases),
        "init_stride_hours": cfg_args.init_stride_hours,
        "era5_cadence_hours": cadence,
        "resolution_deg": cfg_args.resolution_deg,
        "dt_seconds": dt_orch,
        # Ensemble construction AND rollout dynamics — every number in this
        # file depends on all of it. Two materially different ensembles must
        # never produce indistinguishable scorecards (codex review
        # 2026-08-01): record the member checkpoints, the IC-perturbation
        # amplitude and the spectral-filter override, not just the counts.
        "n_checkpoint_members": 1 + len(cfg_args.members),
        "member_checkpoints": list(cfg_args.members),
        "ensemble_ic_noise": cfg_args.ensemble_ic_noise,
        "ensemble_noise_scale": cfg_args.ensemble_noise_scale,
        "spectral_filter_override": cfg_args.sf_override,
        "mc_dropout_members": cfg_args.mc_dropout_members,
        "ensemble_size": (
            (1 + len(cfg_args.members))
            * max(1, cfg_args.mc_dropout_members)),
        "probabilistic": cfg_args.probabilistic,
        "ensemble_mean_convention": (
            "diagnose-then-mean" if cfg_args.probabilistic
            else "mean-then-diagnose"),
        "climatology_mode": cfg_args.climatology,
        "climatology_samples": (
            cfg_args.climatology_samples
            if cfg_args.climatology == "annual" else len(cases)),
        "climatology_note": (
            "climatology = annual-mean over evenly-spaced eval-year snapshots "
            "(--climatology annual, default) or the eval-window sample mean "
            "(--climatology window). The window mode is a LOCAL mean unless the "
            "inits span a year, and then scores BELOW persistence at 24 h "
            "(58.7 m vs 63.9 m for z500, measured 2026-07-28) - do not report "
            "it. Neither equals the published WB2 reference, a 1990-2019 "
            "day-of-year climatology that removes the seasonal cycle and scores "
            "lower still (83.6 m for z500 after /g); quote the published number "
            "when comparing to the leaderboard."),
    }
    out = {
        "meta": meta,
        "model": _nest_scorecard(model_sc),
        "persistence": _nest_scorecard(persist_sc),
        "climatology": _nest_scorecard(clim_sc),
    }

    os.makedirs(os.path.dirname(os.path.abspath(cfg_args.out)), exist_ok=True)
    with open(cfg_args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    log.info("wrote scorecard -> %s", cfg_args.out)
    _print_table(model_sc, persist_sc, cfg_args.leads_hours, log)
    return out


if __name__ == "__main__":
    main()
