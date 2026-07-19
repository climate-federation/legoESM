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


def _parse_leads(s):
    if not s:
        raise SystemExit("--leads must list at least one lead time in hours")
    leads = tuple(int(x) for x in str(s).split(",") if x != "")
    if not leads or any(lead <= 0 for lead in leads):
        raise SystemExit(f"--leads must be positive integers (hours), got {s!r}")
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
    p.add_argument("--out", default=None,
                   help="Scorecard JSON path (default "
                        "results/aimip_wbcompare/<variant>/scorecard.json).")
    a = p.parse_args(argv)

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
        resolution_deg=a.resolution_deg, out=out)


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
    import jax

    if variant == "classical":
        from legoesm.training.aimip_params import AIMIPClassicalParams
        # Matches _train_aimip_classical: spatial_surface + init_std + seed all
        # read from the SAME cfg keys with the SAME defaults.
        return AIMIPClassicalParams.from_defaults(
            spatial_surface=bool(cfg.get("aimip_spatial_surface", False)),
            spatial_init_std=float(cfg.get("aimip_spatial_init_std", 0.0)),
            spatial_seed=int(cfg.get("aimip_spatial_seed", 0)),
        )

    if variant == "column_nn":
        from legoesm.atmosphere.physics.neural_physics import (
            build_column_physics,
        )
        # Matches train_column_mlp_spectral(seed=nn_seed, hidden_dim=nn_hidden_dim,
        # n_layers=nn_n_layers, residual_scale defaulted). residual_scale is left
        # at build_column_physics' default (run_aimip does not override it), so a
        # checkpoint trained through run_aimip matches without passing it here.
        return build_column_physics(
            nlev=spec_cfg.n_levels,
            hidden_dim=int(cfg.get("nn_hidden_dim", 256)),
            n_layers=int(cfg.get("nn_n_layers", 4)),
            key=jax.random.PRNGKey(int(cfg.get("nn_seed", 0))),
        )

    if variant == "sfno_full":
        from legoesm.ml.channel_packing import PE3DChannelSpec
        from legoesm.ml.sfno import SFNO, SFNOConfig
        # Matches train_sfno_full_spectral: in==out channels from PE3DChannelSpec,
        # embed/blocks/mlp_expansion from spec_cfg, residual_prediction=False,
        # keyed on sfno_seed.
        channels = PE3DChannelSpec(nlev=spec_cfg.n_levels).n_channels
        arch = SFNOConfig(
            in_channels=channels,
            out_channels=channels,
            embed_dim=spec_cfg.sfno_embed_dim,
            n_blocks=spec_cfg.sfno_n_blocks,
            mlp_expansion=spec_cfg.sfno_mlp_expansion,
            residual_prediction=False,
        )
        return SFNO(arch, grid, key=jax.random.PRNGKey(int(cfg.get("sfno_seed", 0))))

    # Unreachable: build_eval_config_from_args already gates --variant. Kept as a
    # hard error (dispatch hardening) so a future caller of _build_skeleton with a
    # bad variant fails loudly instead of returning None.
    raise ValueError(f"Unknown variant in _build_skeleton: {variant!r}")


def _build_rollout_fn(variant, trained, cfg, spec_cfg, grid, sigma, pe_config,
                      sponge_factor, spectral_filter, checkpoint_path=None):
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
            gwd_scheme=str(cfg.get("aimip_gwd", "mcfarlane")),
            microphysics_scheme=str(cfg.get("aimip_microphysics", "none")),
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
            ),
            mode="state_update",
            dt_sfno=dt_sfno,
            correct_mass=True,
            correct_moisture_budget=False,
            clip_q=False,
            use_normalization=True,
        )
        wrapper = SFNOPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma, config=pe_cfg, sfno_model=trained,
            norm_stats=norm_stats,
        )

        def _scan_body(s, _):
            return wrapper.step(s, dt_sfno), None

        def rollout_fn(state, _physics_fn, grid_, sigma_, pe_config_, dt_,
                       n_steps, *, forcing_base=None):
            # The orchestrator passes dt=dt_sfno, so n_steps is already in SFNO
            # macro-step units (round(lead*3600/dt_sfno)). Clamp to >=1 to match
            # _evaluate_variant (a sub-macro-step lead still runs one SFNO step).
            n = max(1, int(n_steps))
            final, _ = jax.lax.scan(_scan_body, state, jnp.arange(n))
            return final
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

    # _build_spectral_config derives checkpoint_dir from output_dir/variant, but
    # this eval loads the model from --checkpoint (not checkpoint_dir) and reads
    # the sfno_full norm-stats sidecar next to it — so output_dir is UNUSED here.
    # Minimal suites (e.g. config/aimip/scale/*) keep output_dir at the suite top
    # level, not in the merged base+variant config, so inject a harmless default.
    yml.setdefault("output_dir", "results/aimip_wb2_eval")
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

    rollout_fn, dt_orch = _build_rollout_fn(
        cfg_args.variant, trained, yml, spec_cfg, grid, sigma, pe_config,
        sponge_factor, spectral_filter, checkpoint_path=cfg_args.checkpoint,
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

    clim = climatology_from_cases(cases)

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
        "climatology_note": (
            "climatology = eval-window sample mean over verification snapshots. "
            "Self-consistent for ranking variants under identical sampling; NOT "
            "comparable to published WB2 ACC (1990-2019 hourly climatology). "
            "RMSE/bias compare directly."),
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
