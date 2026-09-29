"""Arg-parse + pure-helper tests for the AIMIP WB2 eval driver (JAX-free, login-safe).

Mirrors ``test_run_weatherbench_eval_cli.py``: the driver's top level must NOT
import jax (heavy imports live inside ``main``), so it loads on a login node and
this test exercises only the import-light layer:
  * ``build_eval_config_from_args`` (defaults, round-trip, required + unknown
    ``--variant`` -> SystemExit, per-variant default out-path, bad leads/inits);
  * ``leads_to_sfno_steps`` (the sfno_full lead->macro-step conversion).

The driver + evaluations submodules are loaded by FILE PATH so neither the
driver's heavy imports NOR evaluations/__init__.py (which imports jax) is pulled
in — same pattern as tests/unit/test_fetch_wb2_sota.py.
"""
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "scripts" / "validate" / "run_aimip_wb2_eval.py"
_spec = importlib.util.spec_from_file_location("run_aimip_wb2_eval", _ENTRY)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)          # top-level must NOT import jax


# ---------------------------------------------------------------- arg-parse ---
def test_argparse_roundtrip():
    cfg = mod.build_eval_config_from_args(
        ["--variant", "classical", "--config", "cfg.yaml", "--checkpoint",
         "ck/epoch_0003.eqx", "--leads", "24,72", "--eval-year", "2020",
         "--n-inits", "3", "--init-stride-hours", "48", "--resolution-deg",
         "1.5", "--out", "out/sc.json"])
    assert cfg.variant == "classical"
    assert cfg.config_path == "cfg.yaml"
    assert cfg.checkpoint == "ck/epoch_0003.eqx"
    assert cfg.leads_hours == (24, 72)
    assert cfg.eval_year == 2020
    assert cfg.n_inits == 3
    assert cfg.init_stride_hours == 48
    assert cfg.resolution_deg == 1.5
    assert cfg.out == "out/sc.json"


def test_argparse_defaults():
    cfg = mod.build_eval_config_from_args(
        ["--variant", "column_nn", "--config", "c.yaml", "--checkpoint",
         "e.eqx"])
    assert cfg.variant == "column_nn"
    assert cfg.leads_hours == (24, 72, 120, 240)
    assert cfg.eval_year is None
    assert cfg.n_inits == 8
    assert cfg.init_stride_hours == 24
    assert cfg.resolution_deg == 1.5


def test_variant_required():
    """--variant is required; omitting it is a hard SystemExit (argparse)."""
    with pytest.raises(SystemExit):
        mod.build_eval_config_from_args(
            ["--config", "c.yaml", "--checkpoint", "e.eqx"])


def test_unknown_variant_hard_errors():
    """An unknown --variant is a hard SystemExit, NOT a silent default
    (dispatch hardening)."""
    with pytest.raises(SystemExit):
        mod.build_eval_config_from_args(
            ["--variant", "sfno_bogus", "--config", "c.yaml",
             "--checkpoint", "e.eqx"])


@pytest.mark.parametrize("variant", ["classical", "column_nn", "sfno_full"])
def test_all_valid_variants_accepted(variant):
    cfg = mod.build_eval_config_from_args(
        ["--variant", variant, "--config", "c.yaml", "--checkpoint", "e.eqx"])
    assert cfg.variant == variant
    assert variant in mod.VALID_VARIANTS


@pytest.mark.parametrize("variant", ["classical", "column_nn", "sfno_full"])
def test_per_variant_default_out_path(variant):
    cfg = mod.build_eval_config_from_args(
        ["--variant", variant, "--config", "c.yaml", "--checkpoint", "e.eqx"])
    assert cfg.out == f"results/aimip_wbcompare/{variant}/scorecard.json"
    assert cfg.out == mod._default_out(variant)


def test_explicit_out_overrides_default():
    cfg = mod.build_eval_config_from_args(
        ["--variant", "sfno_full", "--config", "c.yaml", "--checkpoint",
         "e.eqx", "--out", "custom/path.json"])
    assert cfg.out == "custom/path.json"


def test_bad_leads_rejected():
    for bad in ["0,24", "-6,24", ""]:
        with pytest.raises(SystemExit):
            mod.build_eval_config_from_args(
                ["--variant", "classical", "--config", "c.yaml",
                 "--checkpoint", "e.eqx", "--leads", bad])


def test_bad_n_inits_rejected():
    with pytest.raises(SystemExit):
        mod.build_eval_config_from_args(
            ["--variant", "classical", "--config", "c.yaml",
             "--checkpoint", "e.eqx", "--n-inits", "0"])


def test_bad_init_stride_rejected():
    with pytest.raises(SystemExit):
        mod.build_eval_config_from_args(
            ["--variant", "classical", "--config", "c.yaml",
             "--checkpoint", "e.eqx", "--init-stride-hours", "0"])


def test_bad_resolution_rejected():
    with pytest.raises(SystemExit):
        mod.build_eval_config_from_args(
            ["--variant", "classical", "--config", "c.yaml",
             "--checkpoint", "e.eqx", "--resolution-deg", "0"])


# ------------------------------------------------------- leads -> sfno steps ---
def test_leads_to_sfno_steps_basic():
    # 6 h macro step: 24 h = 4 steps, 240 h = 40 steps.
    assert mod.leads_to_sfno_steps(24, 21600.0) == 4
    assert mod.leads_to_sfno_steps(72, 21600.0) == 12
    assert mod.leads_to_sfno_steps(240, 21600.0) == 40


def test_leads_to_sfno_steps_clamps_to_one():
    # A sub-macro-step lead still runs at least one SFNO step (matches
    # run_aimip._evaluate_variant's max(1, ...)).
    assert mod.leads_to_sfno_steps(3, 21600.0) == 1
    assert mod.leads_to_sfno_steps(1, 21600.0) == 1


def test_leads_to_sfno_steps_rounds():
    # 30 h / 6 h = 5.0 exactly; 27 h / 6 h = 4.5 -> round-half-to-even = 4.
    assert mod.leads_to_sfno_steps(30, 21600.0) == 5
    assert mod.leads_to_sfno_steps(27, 21600.0) == 4


def test_leads_to_sfno_steps_rejects_nonpositive():
    with pytest.raises(ValueError):
        mod.leads_to_sfno_steps(0, 21600.0)
    with pytest.raises(ValueError):
        mod.leads_to_sfno_steps(24, 0.0)


# ------------------------------------------------- config-source (--config/--suite) --
def test_suite_source_accepted():
    cfg = mod.build_eval_config_from_args(
        ["--variant", "classical", "--suite", "config/aimip/wbcompare/suite.yaml",
         "--checkpoint", "e.eqx"])
    assert cfg.suite_path.endswith("suite.yaml")
    assert cfg.config_path is None


def test_config_and_suite_mutually_exclusive():
    with pytest.raises(SystemExit):   # both -> argparse error
        mod.build_eval_config_from_args(
            ["--variant", "classical", "--config", "c.yaml",
             "--suite", "s.yaml", "--checkpoint", "e.eqx"])


def test_config_or_suite_required():
    with pytest.raises(SystemExit):   # neither -> argparse error
        mod.build_eval_config_from_args(
            ["--variant", "classical", "--checkpoint", "e.eqx"])


def test_merged_cfg_from_suite_matches_run_aimip_merge(tmp_path):
    """merged_cfg_from_suite reproduces base <- cfg_overrides <- variant overlay
    using a stub run_aimip exposing _load_yaml/_merge (no jax)."""
    import yaml

    (tmp_path / "base.yaml").write_text("a: 1\nnlev: 8\naimip_convection: sbm\n")
    (tmp_path / "suite.yaml").write_text(
        "base: %s\ncfg_overrides: {a: 2, eval_years: [2020]}\n"
        "variants: [classical]\n" % (tmp_path / "base.yaml"))
    (tmp_path / "variant_classical.yaml").write_text("aimip_convection: edmf\n")

    class _Stub:
        @staticmethod
        def _load_yaml(p):
            return yaml.safe_load(open(p))

        @staticmethod
        def _merge(b, o):
            out = dict(b)
            out.update(o)   # shallow is enough for this fixture
            return out

    cfg = mod.merged_cfg_from_suite(_Stub, str(tmp_path / "suite.yaml"), "classical")
    assert cfg["a"] == 2                       # cfg_overrides won over base
    assert cfg["aimip_convection"] == "edmf"   # variant overlay won last
    assert cfg["eval_years"] == [2020]
    assert cfg["aimip_variant"] == "classical"


# ---------------------------------------------- sfno_full norm-stats sidecar ---
def test_sfno_full_norm_stats_path_resolves_next_to_checkpoint():
    """The sidecar resolves to norm_stats.npz in the checkpoint's parent dir
    (== train's config.checkpoint_dir); pure path logic, no jax/IO."""
    p = mod.sfno_full_norm_stats_path(
        "results/aimip_wbcompare/sfno_full/epoch_0007.eqx")
    assert p.endswith("results/aimip_wbcompare/sfno_full/norm_stats.npz")
    # A different epoch in the same dir maps to the SAME sidecar.
    p2 = mod.sfno_full_norm_stats_path(
        "results/aimip_wbcompare/sfno_full/epoch_0000.eqx")
    assert p == p2


def test_sfno_full_norm_stats_path_rejects_none():
    with pytest.raises(ValueError):
        mod.sfno_full_norm_stats_path(None)


# ------------------------------------------------------- schema compatibility --
def test_valid_variants_tuple_matches_config_dir():
    """The driver's VALID_VARIANTS matches the three wbcompare variant configs
    on disk, so every scorecard the driver can produce has a training config."""
    cfg_dir = _ROOT / "config" / "aimip" / "wbcompare"
    for v in mod.VALID_VARIANTS:
        assert (cfg_dir / f"variant_{v}.yaml").exists(), (
            f"missing config/aimip/wbcompare/variant_{v}.yaml for variant {v!r}")


# ---------------------------------------------------- ZM land mask (#1812) ---
def test_classical_zm_rollout_gets_the_era5_land_mask(monkeypatch):
    """The standalone evaluator's classical ZM lane runs on the ERA5 land-sea
    mask (the one run_aimip trains on); without it ZM refuses the grid."""
    import jax
    import jax.numpy as jnp

    import legoesm.training.aimip_spatial as sp
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import isothermal_rest_state_spectral
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.aimip_params import AIMIPClassicalParams

    ra_spec = importlib.util.spec_from_file_location(
        "run_aimip_for_wb2_eval_test", _ROOT / "scripts" / "run" / "run_aimip.py")
    run_aimip = importlib.util.module_from_spec(ra_spec)
    ra_spec.loader.exec_module(run_aimip)
    cfg = run_aimip._merge(
        run_aimip._load_yaml(_ROOT / "config" / "aimip" / "aimip_era5.yaml"),
        {"n_max": 8, "aimip_variant": "classical",
         "aimip_spatial_surface": False, "aimip_radiation": "gray",
         "aimip_rad_update_interval": 1, "aimip_convection": "zhang_mcfarlane",
         "aimip_turbulence": "louis", "aimip_microphysics": "sundqvist",
         "aimip_cloud": "xu_randall"})
    spec_cfg = run_aimip._build_spectral_config(cfg)
    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sig = create_sigma_coordinate(spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top)
    calls = []
    era5_cfg, ds = object(), object()   # the evaluator's store and dataset

    def fake_era5(g, config=None, *, ds=None):
        calls.append((g.n_lat, config, ds))
        return jnp.where(jnp.asarray(g.lat2d) > 0.0, 1.0, 0.0)

    monkeypatch.setattr(sp, "era5_land_fraction", fake_era5)
    rollout_fn, _dt = mod._build_rollout_fn(
        "classical", AIMIPClassicalParams.from_defaults(), cfg, spec_cfg,
        grid, sig, spec_cfg.pe_config, None, None, era5_cfg=era5_cfg, ds=ds)
    assert calls == [(grid.n_lat, era5_cfg, ds)]
    shp = (grid.n_lat, grid.n_lon, spec_cfg.n_levels)
    state = isothermal_rest_state_spectral(
        grid, sig, T_init=290.0, p_s_init=1.0e5,
        tracers={"q_v": jnp.full(shp, 1.2e-2), "q_c": jnp.full(shp, 1e-5),
                 "q_i": jnp.full(shp, 1e-6)})
    out = rollout_fn(state, None, None, None, None, None, 1)
    assert all(bool(jnp.all(jnp.isfinite(x)))
               for x in jax.tree_util.tree_leaves(out)
               if jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact))
