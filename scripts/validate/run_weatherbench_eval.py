#!/usr/bin/env python
"""WeatherBench-2 checkpoint eval driver (issue #919).

Scores a trained WB scale-training checkpoint (``epoch_NNNN.eqx``) against ERA5
on the WB2 headline variables, producing a ``scorecard.json`` with ``model`` +
``persistence`` + ``climatology`` sections and a per-key x per-lead table.

Rollout = the TRAINER's own segment machinery. We reuse
``scale_build.build_mode_components(cfg, yml)`` to get the params skeleton +
``make_run_seg``, load the checkpoint into that skeleton with
``eqx.tree_deserialise_leaves``, and inject a ``rollout_fn`` that wraps
``make_run_seg(trained).raw(carry, n_steps, forcing)`` with
``spectral_state_to_carry`` / ``carry_to_spectral_state`` round-trips. This
guarantees the eval rolls EXACTLY what training rolled (same semi-implicit
config, sponge, spectral filter, forcing handling) — hand-reconstructing
``physics_fn`` + ``pe_config`` would silently drop the spectral filter.

The persistence floor comes free by running the orchestrator with an identity
rollout (t0 held constant); the climatology floor scores the eval-window
sample-mean field set. Both are must-beat baselines.

The arg-parse layer (``build_eval_config_from_args``) is import-light and
JAX-free (login-node testable, per the ``train_weatherbench_scale.py``
convention); all heavy imports live inside ``main``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

# Repo root on sys.path so ``import evaluations`` resolves (this file lives at
# scripts/validate/; evaluations/ is at the repo root). Mirrors the PYTHONPATH
# that scripts/cluster/wb_forecast/env.sh exports for the SLURM jobs.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

VALID_MODES = ("physics", "neural_gcm", "sfno")
VALID_TRAINING_CORES = ("spectral", "latlon")
_SECONDS_PER_HOUR = 3600.0


class EvalConfig(NamedTuple):
    mode: str
    config_path: str
    training_core: str
    checkpoint: str
    leads_hours: tuple
    eval_year: int | None
    n_inits: int
    init_stride_hours: int
    resolution_deg: float
    out: str


class _ModeCfg(NamedTuple):
    """Minimal cfg for ``scale_build.build_mode_components`` (reads .mode +
    .training_core only on the spectral path)."""
    mode: str
    training_core: str


def _parse_leads(s):
    if not s:
        raise SystemExit("--leads must list at least one lead time in hours")
    leads = tuple(int(x) for x in str(s).split(",") if x != "")
    if not leads or any(lead <= 0 for lead in leads):
        raise SystemExit(f"--leads must be positive integers (hours), got {s!r}")
    return leads


def build_eval_config_from_args(argv=None) -> EvalConfig:
    """Parse CLI into an EvalConfig. Import-light + JAX-free (login-node testable)."""
    p = argparse.ArgumentParser(
        description="WB2 checkpoint eval driver (issue #919)")
    p.add_argument("--config", required=True,
                   help="Training YAML (same one the checkpoint was trained with).")
    p.add_argument("--mode", choices=VALID_MODES, default="neural_gcm")
    p.add_argument("--training-core", choices=VALID_TRAINING_CORES,
                   default="spectral", dest="training_core",
                   help="v1 supports 'spectral' only (the orchestrator is "
                        "spectral-state-based); 'latlon' hard-errors.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to an epoch_NNNN.eqx checkpoint.")
    p.add_argument("--leads", default="24,72", dest="leads",
                   help="Comma-separated forecast leads [h]; each must be an "
                        "exact multiple of the core dt and the ERA5 cadence.")
    p.add_argument("--eval-year", type=int, default=None, dest="eval_year",
                   help="Eval calendar year (default: YAML eval_years[0]).")
    p.add_argument("--n-inits", type=int, default=4, dest="n_inits")
    p.add_argument("--init-stride-hours", type=int, default=24,
                   dest="init_stride_hours")
    p.add_argument("--resolution-deg", type=float, default=1.5,
                   dest="resolution_deg",
                   help="WB2 target-grid resolution [deg] (default 1.5).")
    p.add_argument("--out", default="results/wb_eval/scorecard.json")
    a = p.parse_args(argv)

    # v1: spectral core only. Hard SystemExit (not a silent fallback) on latlon,
    # with a clear message — argparse ``choices`` accepts it so we can explain.
    if a.training_core == "latlon":
        raise SystemExit(
            "run_weatherbench_eval: --training-core latlon is not supported in "
            "v1; the WB2 orchestrator is spectral-state-based (it diagnoses a "
            "SpectralHydrostaticState). Train/eval with --training-core spectral.")
    if a.n_inits < 1:
        raise SystemExit(f"--n-inits must be >= 1, got {a.n_inits}")
    if a.init_stride_hours < 1:
        raise SystemExit(
            f"--init-stride-hours must be >= 1, got {a.init_stride_hours}")

    return EvalConfig(
        mode=a.mode, config_path=a.config, training_core=a.training_core,
        checkpoint=a.checkpoint, leads_hours=_parse_leads(a.leads),
        eval_year=a.eval_year, n_inits=a.n_inits,
        init_stride_hours=a.init_stride_hours, resolution_deg=a.resolution_deg,
        out=a.out)


def _assert_leads_on_dt_grid(leads_hours, dt):
    """SystemExit unless every lead is an exact multiple of the core dt.

    The orchestrator also guards this (ValueError), but a login-node CLI wants a
    clean early error naming the offending lead + the core dt.
    """
    for lead in leads_hours:
        steps = lead * _SECONDS_PER_HOUR / dt
        if abs(steps - round(steps)) > 1e-6:
            raise SystemExit(
                f"--leads: lead {lead} h ({lead * _SECONDS_PER_HOUR:g}s) is not "
                f"an exact multiple of the core dt={dt:g}s (ratio {steps}). Pick "
                "leads on the dt grid (spectral dt divides any multiple of 6 h "
                "for the tuned dt=450 s).")


def _nest_scorecard(flat):
    """``{(key, lead): metrics}`` -> ``{key: {str(lead): metrics}}`` (JSON-safe)."""
    nested: dict = {}
    for (key, lead), metrics in flat.items():
        nested.setdefault(key, {})[str(lead)] = metrics
    return nested


def _print_table(model, persistence, leads_hours, log):
    """Per-key x per-lead RMSE table (model vs the persistence floor)."""
    from evaluations.wb_forecast import HEADLINE_FIELD_KEYS

    leads = sorted(int(x) for x in leads_hours)
    header = "  ".join(f"{lead:>10d}h" for lead in leads)
    log.info("WB2 RMSE (model | persistence floor) by field x lead:")
    log.info("%-16s  %s", "field", header)
    for key in HEADLINE_FIELD_KEYS:
        cells = []
        for lead in leads:
            m = model.get((key, lead), {}).get("rmse", float("nan"))
            p = persistence.get((key, lead), {}).get("rmse", float("nan"))
            cells.append(f"{m:10.4g}|{p:<.4g}")
        log.info("%-16s  %s", key, "  ".join(cells))


def main(argv=None, ds=None):
    """Run the WB2 scorecard. ``ds`` injects a pre-opened ERA5 dataset (tests)."""
    cfg = build_eval_config_from_args(argv)

    import json
    import logging
    import os

    import equinox as eqx
    import yaml
    from legoesm.training.era5_to_state import TrainingERA5Config
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state,
        spectral_state_to_carry,
    )
    from legoesm.training.scale_build import (
        build_mode_components,
        check_surface_drag_confound,
    )

    from evaluations.wb_era5_cases import (
        build_forecast_cases,
        climatology_from_cases,
        climatology_scorecard,
    )
    from evaluations.wb_orchestrator import run_wb_forecast_eval

    # Spectral cores require float64 (SH transforms). Enable x64 here — inside
    # the CLI entry point, before any grid/array construction — so a bare
    # ``python run_weatherbench_eval.py ...`` runs correctly without the caller
    # having exported JAX_ENABLE_X64=1. (Entry-point scope, not module-level:
    # avoids the fp32-session-leak wart that a module-top update would cause.)
    import jax
    jax.config.update("jax_enable_x64", True)

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("wb_eval")

    yml = yaml.safe_load(open(cfg.config_path))
    eval_year = cfg.eval_year if cfg.eval_year is not None else int(yml["eval_years"][0])
    cadence = int(yml.get("era5_cadence_hours", 6))

    # --- trained model = the trainer's own components + the checkpoint ---
    mode_cfg = _ModeCfg(mode=cfg.mode, training_core=cfg.training_core)
    _model, grid, sigma, params, make_run_seg, _loss_config, dt = \
        build_mode_components(mode_cfg, yml)
    _assert_leads_on_dt_grid(cfg.leads_hours, dt)

    if not os.path.exists(cfg.checkpoint):
        raise SystemExit(f"--checkpoint not found: {cfg.checkpoint}")
    from legoesm.ml.checkpoint_io import load_checkpoint_or_fail
    trained = load_checkpoint_or_fail(
        cfg.checkpoint, params, what=f"WB mode {cfg.mode}")
    run_seg = make_run_seg(trained)   # built ONCE (no closure churn in the loop)

    def rollout_fn(state, physics_fn, grid_, sigma_, pe_config_, dt_, n_steps,
                   *, forcing_base=None):
        # Roll EXACTLY what training rolled: run_seg closes over the real
        # pe_config/sponge/filter/dt from build_mode_components; the orchestrator's
        # physics_fn/grid/sigma/pe_config/dt args are the (None) placeholders and
        # are intentionally ignored. state <-> carry are documented inverses (#858).
        carry = spectral_state_to_carry(state, grid, sigma)
        final_carry = run_seg.raw(carry, int(n_steps), forcing_base)
        return carry_to_spectral_state(final_carry, grid)

    # --- ERA5 cases (ICs + forcing + WB2-grid verification) ---
    era5_cfg = TrainingERA5Config(dt_hours=cadence)._replace(zarr_store=yml["era5_zarr"])
    cases = build_forecast_cases(
        era5_cfg, grid, sigma,
        leads_hours=cfg.leads_hours, eval_year=eval_year, n_inits=cfg.n_inits,
        init_stride_hours=cfg.init_stride_hours,
        resolution_deg=cfg.resolution_deg, ds=ds)
    log.info("WB2 eval: mode=%s year=%d inits=%d leads=%s res=%.3gdeg dt=%.1fs",
             cfg.mode, eval_year, len(cases), list(cfg.leads_hours),
             cfg.resolution_deg, dt)

    clim = climatology_from_cases(cases)

    # --- model + persistence via the shared orchestrator (identical masking) ---
    model_sc = run_wb_forecast_eval(
        None, grid, sigma, None, dt, cases, cfg.leads_hours, clim,
        resolution_deg=cfg.resolution_deg, rollout_fn=rollout_fn)
    # Persistence = the t0 state held constant (identity rollout); the
    # orchestrator diagnoses t0 and scores it against each lead's verification.
    persist_sc = run_wb_forecast_eval(
        None, grid, sigma, None, dt, cases, cfg.leads_hours, clim,
        resolution_deg=cfg.resolution_deg, rollout_fn=lambda s, *a, **k: s)
    clim_sc = climatology_scorecard(
        cases, cfg.leads_hours, clim, resolution_deg=cfg.resolution_deg)

    meta = {
        "issue": 919,
        "mode": cfg.mode,
        "training_core": cfg.training_core,
        "config": cfg.config_path,
        "checkpoint": cfg.checkpoint,
        "eval_year": eval_year,
        "leads_hours": list(cfg.leads_hours),
        "n_inits": len(cases),
        "init_stride_hours": cfg.init_stride_hours,
        "era5_cadence_hours": cadence,
        "resolution_deg": cfg.resolution_deg,
        "dt_seconds": dt,
        # A confounded comparison must say so IN THE FILE, not only in a log
        # line nobody reads back. `None` = the learned arm carries the same
        # surface stress as the classical one it is scored against (#1464).
        "surface_drag_confound": check_surface_drag_confound(
            yml, cfg.mode, cfg.training_core),
        "climatology_note": (
            "v1 climatology = eval-window sample mean over verification "
            "snapshots. Self-consistent for ranking checkpoints under identical "
            "sampling; NOT comparable to published WB2 ACC (1990-2019 hourly "
            "climatology). RMSE/bias compare directly; the climatology section "
            "also does not apply the model's below-ground pressure-level mask."),
    }
    out = {
        "meta": meta,
        "model": _nest_scorecard(model_sc),
        "persistence": _nest_scorecard(persist_sc),
        "climatology": _nest_scorecard(clim_sc),
    }

    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    with open(cfg.out, "w") as fh:
        json.dump(out, fh, indent=2)
    log.info("wrote scorecard -> %s", cfg.out)
    _print_table(model_sc, persist_sc, cfg.leads_hours, log)
    return out


if __name__ == "__main__":
    main()
