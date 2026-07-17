#!/usr/bin/env python
"""WeatherBench 3-family forecast campaign — one driver, runs on Ginsburg or Derecho.

Trains and scores the three legoESM atmosphere families through the SHARED AIMIP
spectral stack, then plots a lead-time scorecard against the published
WeatherBench-2 leaderboard:

    * ``physics``    -> differentiable trainable-physics params (NeuralGCM-hybrid)
    * ``neural_gcm`` -> per-column MLP physics (NeuralGCM-style)
    * ``sfno``       -> SFNO replaces gridded physics, dycore retained

It is a THIN orchestrator: it does NO training/eval/plot numerics of its own.
Each stage delegates to the canonical entry points, so the campaign rolls
EXACTLY what those tools roll (no re-implemented rollout/loss/metric):

    train -> ``scripts/run/train_weatherbench_scale.py::main``   (MPI data-parallel)
    eval  -> ``scripts/validate/run_weatherbench_eval.py::main`` (WB2 scorecard + floors)
    plot  -> ``scripts/plot/plot_wb_scorecard.py::main``         (families vs SOTA)

MPI contract: the TRAIN stage runs on every rank (the data-parallel loop inside
train_weatherbench_scale averages gradients across ranks). The EVAL and PLOT
stages are single-process and run on RANK 0 ONLY, after a barrier that guarantees
every rank has finished writing its training checkpoints. Single-process
(nproc == 1) runs the whole thing serially.

CONTROLLED COMPARISON (CLAUDE.md): every family trains + evals on the SAME
``--config`` (identical grid, ERA5 sampling, eval year, leads, metric); only the
per-mode model differs. The scorecard numbers are therefore directly comparable
across families and against the SOTA CSV (which carries published WB2 RMSE).

The arg-parse / stage-planning layer is import-light and JAX-free (login-node +
CI testable, per the train/eval driver convention); all heavy imports and the
delegated ``main`` calls live inside ``run_campaign``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple

# Repo root + the sibling script dirs on sys.path so we can import the canonical
# train/eval/plot entry points by module name (mirrors run_weatherbench_eval's
# ``import evaluations`` bootstrap). This file lives at scripts/run/.
_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "scripts" / "run", _REPO / "scripts" / "validate",
           _REPO / "scripts" / "plot"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

VALID_MODES = ("physics", "neural_gcm", "sfno")
VALID_STAGES = ("train", "eval", "plot")
_SECONDS_PER_HOUR = 3600.0

# The neural families (column MLP / SFNO) replace the physics pipeline; the
# WB2 eval driver names them by the training MODE, which is what the checkpoint
# was built with — so the campaign mode == the eval --mode verbatim.


class CampaignConfig(NamedTuple):
    config_path: str
    modes: tuple
    stages: tuple
    training_core: str
    out_root: str
    n_epochs: int | None
    allow_missing_artifacts: bool
    leads_hours: tuple
    eval_year: int | None
    n_inits: int
    init_stride_hours: int
    resolution_deg: float
    sota_csv: str | None
    metric: str
    smoke: bool


def _parse_csv_ints(s, name):
    vals = tuple(int(x) for x in str(s).split(",") if x != "")
    if not vals or any(v <= 0 for v in vals):
        raise SystemExit(f"--{name} must be positive integers (comma-separated), got {s!r}")
    return vals


def _parse_list(s, valid, name):
    """Comma-separated membership list; unknown token is a HARD error (dispatch
    hardening — a typo'd mode/stage must never silently select nothing)."""
    items = tuple(x.strip() for x in str(s).split(",") if x.strip())
    if not items:
        raise SystemExit(f"--{name} must list at least one of {list(valid)}")
    bad = [x for x in items if x not in valid]
    if bad:
        raise SystemExit(f"--{name}: unknown {bad}; valid = {list(valid)}")
    # de-dup preserving order
    seen: set = set()
    ordered = tuple(x for x in items if not (x in seen or seen.add(x)))
    return ordered


def _load_campaign_yaml(config_path) -> dict:
    """Load the campaign YAML (import-light, JAX-free). Missing file is a hard
    error — the config is required for every stage and drives the declared
    experiment; a silent default would defeat the controlled comparison."""
    path = config_path
    if not os.path.exists(path) and not os.path.isabs(path):
        # Fall back to a repo-root-relative lookup so the default resolves
        # regardless of the CWD pytest / a login shell runs from.
        alt = _REPO / path
        if alt.exists():
            path = str(alt)
    if not os.path.exists(path):
        raise SystemExit(f"--config {config_path!r} not found")
    import yaml
    return yaml.safe_load(open(path)) or {}


def build_campaign_config_from_args(argv=None) -> CampaignConfig:
    """Parse CLI into a CampaignConfig. Import-light + JAX-free (login-node testable)."""
    p = argparse.ArgumentParser(
        description="WeatherBench 3-family forecast campaign (train + eval + scorecard).")
    p.add_argument("--config", default="config/wb/campaign/spectral_t63.yaml",
                   help="Campaign YAML (same one every family trains + evals with).")
    p.add_argument("--modes", default=",".join(VALID_MODES),
                   help=f"Comma-separated model families; subset of {list(VALID_MODES)}.")
    p.add_argument("--stages", default=",".join(VALID_STAGES),
                   help=f"Comma-separated pipeline stages; subset of {list(VALID_STAGES)}.")
    p.add_argument("--training-core", choices=("spectral",), default="spectral",
                   dest="training_core",
                   help="Campaign is spectral-only (the WB2 eval driver is "
                        "spectral-state based; latlon eval is not supported).")
    p.add_argument("--out-root", default="results/wb_campaign", dest="out_root")
    p.add_argument("--epochs", type=int, default=None, dest="n_epochs",
                   help="Override YAML n_epochs (default: use the YAML value).")
    p.add_argument("--leads", default="24,72,120,240", dest="leads",
                   help="Comma-separated forecast leads [h]; each an exact "
                        "multiple of the core dt and the ERA5 cadence.")
    p.add_argument("--eval-year", type=int, default=None, dest="eval_year",
                   help="Eval calendar year (default: YAML eval_years[0]).")
    p.add_argument("--n-inits", type=int, default=8, dest="n_inits")
    p.add_argument("--init-stride-hours", type=int, default=24, dest="init_stride_hours")
    p.add_argument("--resolution-deg", type=float, default=1.5, dest="resolution_deg",
                   help="WB2 target-grid resolution [deg] (default 1.5).")
    p.add_argument("--sota-csv", default="config/wb/sota/wb2_headline_rmse.csv",
                   dest="sota_csv",
                   help="Published WB2 headline RMSE CSV for the comparison plot; "
                        "pass '' to plot the legoESM families + floors only.")
    p.add_argument("--metric", default="rmse", choices=("rmse", "acc"),
                   help="Scorecard metric to plot.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny wiring check: forwards --smoke to the trainer.")
    p.add_argument("--allow-missing-artifacts", action="store_true",
                   dest="allow_missing_artifacts",
                   help="Do not fail the campaign when a requested eval finds no "
                        "checkpoint / plot finds no scorecards (default: hard error "
                        "so a failed stage is never reported as success).")
    a = p.parse_args(argv)

    if a.n_epochs is not None and a.n_epochs < 1:
        raise SystemExit(f"--epochs must be >= 1, got {a.n_epochs}")
    if a.n_inits < 1:
        raise SystemExit(f"--n-inits must be >= 1, got {a.n_inits}")
    if a.init_stride_hours < 1:
        raise SystemExit(f"--init-stride-hours must be >= 1, got {a.init_stride_hours}")

    # Resolve n_epochs / eval_year from the campaign YAML when the CLI omits
    # them, so the DECLARED experiment (the whole point of the controlled
    # comparison) is what actually runs — otherwise the trainer/eval defaults
    # silently override the YAML and every family runs a different-length job
    # than the config says. CLI flag still wins for a one-off override.
    yml = _load_campaign_yaml(a.config)
    n_epochs = a.n_epochs
    if n_epochs is None and yml.get("n_epochs") is not None:
        n_epochs = int(yml["n_epochs"])
        if n_epochs < 1:
            raise SystemExit(f"{a.config}: n_epochs must be >= 1, got {n_epochs}")
    eval_year = a.eval_year
    if eval_year is None:
        _years = yml.get("eval_years")
        if _years:
            eval_year = int(_years[0])

    return CampaignConfig(
        config_path=a.config,
        modes=_parse_list(a.modes, VALID_MODES, "modes"),
        stages=_parse_list(a.stages, VALID_STAGES, "stages"),
        training_core=a.training_core,
        out_root=a.out_root,
        n_epochs=n_epochs,
        allow_missing_artifacts=a.allow_missing_artifacts,
        leads_hours=_parse_csv_ints(a.leads, "leads"),
        eval_year=eval_year,
        n_inits=a.n_inits,
        init_stride_hours=a.init_stride_hours,
        resolution_deg=a.resolution_deg,
        sota_csv=(a.sota_csv or None),
        metric=a.metric,
        smoke=a.smoke,
    )


def mode_out_dir(out_root, mode):
    """Per-family checkpoint + scorecard directory."""
    return os.path.join(out_root, mode)


def scorecard_path(out_root, mode):
    return os.path.join(mode_out_dir(out_root, mode), "scorecard.json")


def build_train_argv(cfg: CampaignConfig, mode: str) -> list:
    """argv for train_weatherbench_scale.main for one family (pure — testable)."""
    argv = [
        "--config", cfg.config_path,
        "--mode", mode,
        "--training-core", cfg.training_core,
        "--out", mode_out_dir(cfg.out_root, mode),
        # NOTE: no --resume. train_weatherbench_scale parses the flag but never
        # acts on it (no checkpoint restore), so passing it advertised a
        # restart-chaining contract the trainer does not honor. Omit until the
        # trainer implements resume; a restart currently retrains from scratch.
    ]
    if cfg.n_epochs is not None:
        argv += ["--epochs", str(cfg.n_epochs)]
    if cfg.smoke:
        argv += ["--smoke"]
    return argv


def build_eval_argv(cfg: CampaignConfig, mode: str, checkpoint: str) -> list:
    """argv for run_weatherbench_eval.main for one family (pure — testable)."""
    argv = [
        "--config", cfg.config_path,
        "--mode", mode,
        "--training-core", cfg.training_core,
        "--checkpoint", checkpoint,
        "--leads", ",".join(str(h) for h in cfg.leads_hours),
        "--n-inits", str(cfg.n_inits),
        "--init-stride-hours", str(cfg.init_stride_hours),
        "--resolution-deg", str(cfg.resolution_deg),
        "--out", scorecard_path(cfg.out_root, mode),
    ]
    if cfg.eval_year is not None:
        argv += ["--eval-year", str(cfg.eval_year)]
    return argv


def build_plot_argv(cfg: CampaignConfig, family_scorecards: dict, out_png: str) -> list:
    """argv for plot_wb_scorecard.main (pure — testable).

    ``family_scorecards`` maps mode -> scorecard.json path (only the families
    that actually produced a scorecard are plotted).
    """
    argv = []
    for mode, path in family_scorecards.items():
        argv += ["--scorecard", f"{mode}={path}"]
    if cfg.sota_csv:
        argv += ["--sota-csv", cfg.sota_csv]
    argv += ["--metric", cfg.metric, "--out", out_png]
    return argv


def latest_checkpoint(out_dir):
    """Newest ``epoch_NNNN.eqx`` in ``out_dir`` (highest epoch), or None."""
    d = Path(out_dir)
    if not d.is_dir():
        return None
    ckpts = sorted(d.glob("epoch_*.eqx"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return str(ckpts[-1]) if ckpts else None


def _barrier(nproc):
    """Collective barrier when running under MPI; no-op single-process.

    Ensures every rank has finished the TRAIN stage (and flushed checkpoints)
    before rank 0 begins the single-process EVAL/PLOT stages.
    """
    if nproc <= 1:
        return
    # Under multi-rank, the barrier is load-bearing: rank 0 must NOT start eval
    # until every rank has flushed its checkpoints. Let an import/barrier failure
    # RAISE — swallowing it would let rank 0 evaluate incomplete checkpoints and
    # still exit successfully.
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()


def run_campaign(cfg: CampaignConfig):
    """Execute the requested stages. Returns the assembled scorecard-path map."""
    import logging

    from legoesm.training.data_parallel import mpi_rank_size

    rank, nproc = mpi_rank_size()
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    log = logging.getLogger("wb_campaign")

    import train_weatherbench_scale as trainer
    import run_weatherbench_eval as evaler

    if rank == 0:
        os.makedirs(cfg.out_root, exist_ok=True)
        log.info("WB campaign: modes=%s stages=%s core=%s ranks=%d out=%s",
                 list(cfg.modes), list(cfg.stages), cfg.training_core, nproc,
                 cfg.out_root)

    # --- TRAIN (all ranks; the data-parallel loop is collective) ---
    if "train" in cfg.stages:
        for mode in cfg.modes:
            if rank == 0:
                log.info("=== TRAIN %s ===", mode)
            trainer.main(build_train_argv(cfg, mode))

    _barrier(nproc)

    # --- EVAL + PLOT (rank 0 only; single-process) ---
    family_scorecards: dict = {}
    missing: list = []
    if rank == 0:
        if "eval" in cfg.stages:
            for mode in cfg.modes:
                ckpt = latest_checkpoint(mode_out_dir(cfg.out_root, mode))
                if ckpt is None:
                    log.warning("=== EVAL %s SKIPPED: no checkpoint in %s "
                                "(train it first) ===",
                                mode, mode_out_dir(cfg.out_root, mode))
                    missing.append(f"eval:{mode} (no checkpoint)")
                    continue
                log.info("=== EVAL %s (ckpt=%s) ===", mode, ckpt)
                evaler.main(build_eval_argv(cfg, mode, ckpt))
                family_scorecards[mode] = scorecard_path(cfg.out_root, mode)
        else:
            # plot-only: pick up any scorecards already on disk
            for mode in cfg.modes:
                sc = scorecard_path(cfg.out_root, mode)
                if os.path.exists(sc):
                    family_scorecards[mode] = sc

        if "plot" in cfg.stages:
            if not family_scorecards:
                log.warning("=== PLOT SKIPPED: no family scorecards found under %s ===",
                            cfg.out_root)
                missing.append("plot (no family scorecards)")
            else:
                # Drop a missing SOTA CSV to a families-only plot rather than
                # letting load_sota_headline raise FileNotFoundError — the CSV is
                # produced by a separate fetch job (scripts/data/fetch_wb2_sota.py)
                # that may not have run yet.
                plot_cfg = cfg
                if cfg.sota_csv and not os.path.exists(cfg.sota_csv):
                    log.warning("SOTA CSV %s not found — plotting families + floors "
                                "only (run scripts/data/fetch_wb2_sota.py to add "
                                "the published-model overlay).", cfg.sota_csv)
                    plot_cfg = cfg._replace(sota_csv=None)
                import plot_wb_scorecard as plotter
                out_png = os.path.join(cfg.out_root, "wb_scorecard.png")
                log.info("=== PLOT %d families -> %s ===",
                         len(family_scorecards), out_png)
                plotter.main(build_plot_argv(plot_cfg, family_scorecards, out_png))

        # A requested stage that produced nothing is a FAILURE, not a success:
        # otherwise a diverged/OOM training run (no checkpoint) or a missing
        # scorecard would exit 0 and read as "campaign done".
        if missing and not cfg.allow_missing_artifacts:
            raise SystemExit(
                "WB campaign: requested artifacts missing: "
                + "; ".join(missing)
                + " (pass --allow-missing-artifacts to downgrade to a warning)."
            )

    return family_scorecards


def main(argv=None):
    cfg = build_campaign_config_from_args(argv)
    return run_campaign(cfg)


if __name__ == "__main__":
    main()
