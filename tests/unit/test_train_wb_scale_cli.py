"""Arg-parse + config tests for the WB scale-training entry (JAX-free, login-safe)."""
import importlib.util
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "scripts" / "run" / "train_weatherbench_scale.py"
_spec = importlib.util.spec_from_file_location("train_wb_scale", _ENTRY)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)          # top-level must NOT import jax


def test_argparse_roundtrip():
    cfg = mod.build_scale_config_from_args(
        ["--mode", "neural_gcm", "--resolution", "0.7", "--epochs", "20",
         "--multi-step-hours", "6,12", "--eval-wb2"])
    assert cfg.mode == "neural_gcm"
    assert cfg.resolution_deg == 0.7
    assert cfg.n_epochs == 20
    assert cfg.multi_step_hours == (6, 12)
    assert cfg.eval_wb2 is True


def test_argparse_defaults_and_all_modes():
    cfg = mod.build_scale_config_from_args([])
    # --resolution defaults to None (#817 papercut fix): the grid comes from
    # the YAML; the value is derived for logging and an explicit mismatch is a
    # hard error (see test_resolution_yaml_check below).
    assert cfg.mode == "neural_gcm" and cfg.resolution_deg is None and cfg.grad_accum == 1
    assert cfg.training_core == "latlon"   # default core: byte-unchanged path
    for m in ("physics", "neural_gcm", "sfno"):
        assert mod.build_scale_config_from_args(["--mode", m]).mode == m


def test_argparse_rejects_bad_mode():
    with pytest.raises(SystemExit):
        mod.build_scale_config_from_args(["--mode", "bogus"])


def test_argparse_training_core_roundtrip_and_rejects_bad():
    """#817: --training-core selects the spectral (semi-implicit) training
    core; unknown values are rejected by argparse choices."""
    for core in ("latlon", "spectral"):
        cfg = mod.build_scale_config_from_args(["--training-core", core])
        assert cfg.training_core == core
    with pytest.raises(SystemExit):
        mod.build_scale_config_from_args(["--training-core", "bogus"])


def test_resolution_yaml_check():
    """#817 papercut: --resolution must MATCH the YAML grid or hard-error —
    the old flag silently logged one resolution while training at another."""
    yml = {"n_lat": 256, "n_lon": 512}
    # None (default) -> derived from the YAML.
    cfg = mod.build_scale_config_from_args([])
    assert abs(mod._check_resolution_matches_yaml(cfg, yml) - 180.0 / 256) < 1e-9
    # Matching explicit value passes.
    cfg = mod.build_scale_config_from_args(["--resolution", "0.703125"])
    assert mod._check_resolution_matches_yaml(cfg, yml) == 0.703125
    # Mismatch (2.8 deg vs a 0.7 deg YAML) -> SystemExit, not a silent no-op.
    cfg = mod.build_scale_config_from_args(["--resolution", "2.8"])
    with pytest.raises(SystemExit, match="does not match the YAML"):
        mod._check_resolution_matches_yaml(cfg, yml)


def test_config_yaml_loads():
    y = yaml.safe_load(open(_ROOT / "config" / "wb" / "scale" / "train_07deg.yaml"))
    assert y["grid"] == "latlon" and y["dt"] > 0
    assert {"w_T", "multi_step_hours"} <= set(y["loss"])
    assert y["loss"]["w_spec_crps_T"] == 0.0            # spectral-CRPS off at scale
    assert y["train_years"] and y["eval_years"] == [2020]


def test_n_days_cli_roundtrip_and_reject():
    """#1047 ask a: --n-days sets the training-window length; default None."""
    assert mod.build_scale_config_from_args([]).n_days is None
    assert mod.build_scale_config_from_args(["--n-days", "60"]).n_days == 60
    with pytest.raises(SystemExit, match="--n-days must be >= 1"):
        mod.build_scale_config_from_args(["--n-days", "0"])


def test_resolve_n_days_precedence():
    """CLI cfg.n_days > YAML n_training_days > 3; --smoke forces 1 (#1047)."""
    from legoesm.training.scale_build import _resolve_n_days

    base = mod.build_scale_config_from_args([])          # n_days=None, smoke=False
    assert _resolve_n_days(base, {}) == 3                # historical default
    assert _resolve_n_days(base, {"n_training_days": 60}) == 60   # YAML wins over 3
    cli = mod.build_scale_config_from_args(["--n-days", "10"])
    assert _resolve_n_days(cli, {"n_training_days": 60}) == 10    # CLI wins over YAML
    assert _resolve_n_days(base._replace(smoke=True), {"n_training_days": 60}) == 1
    with pytest.raises(ValueError, match="must be >= 1"):
        _resolve_n_days(base, {"n_training_days": 0})


def test_rollout_hours_matches_first_lead():
    """The training rollout horizon = the FIRST multi_step_hours lead — the same
    lead load_era5_samples uses to pick the target, so pred and target stay at
    the same forecast time (the old hardwired single_day_rollout scored a 24 h
    forecast against a 6 h target)."""
    from legoesm.training.scale_build import rollout_hours

    cfg = mod.build_scale_config_from_args(["--multi-step-hours", "12,24"])
    assert rollout_hours(cfg, {}) == 12.0
    # CLI default (6,12) -> 6 h
    cfg = mod.build_scale_config_from_args([])
    assert rollout_hours(cfg, {}) == 6.0
    # no CLI leads -> YAML loss.multi_step_hours wins; nothing at all -> 6 h
    cfg = cfg._replace(multi_step_hours=())
    assert rollout_hours(cfg, {"loss": {"multi_step_hours": [12, 24]}}) == 12.0
    assert rollout_hours(cfg, {}) == 6.0


def test_rrtmgp_cache_is_warmed_before_the_traced_loss():
    """The classical arm builds RRTMGP inside ``make_run_seg``, and ``loss_fn``
    calls that under ``eqx.filter_value_and_grad``.  With a cold optics cache
    the NetCDF gas-optics load then runs against Equinox tracers and the job
    dies with TracerArrayConversionError (job 26905933, six minutes of ERA5
    loading wasted first).  ``main`` must therefore build the segment once with
    CONCRETE params, before the training loop.
    """
    import ast

    tree = ast.parse(_ENTRY.read_text())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")

    # It must be a bare statement in main's OWN body: a call nested in an inner
    # def is the traced one this guards against, and one wrapped in ``if
    # cfg.mode == ...`` or a swallowing ``try`` leaves the classical arm exactly
    # as broken as before (codex).
    warm = [stmt.value for stmt in main.body
            if isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "make_run_seg"]
    assert warm, ("main() must call make_run_seg(...) as an unconditional "
                  "top-level statement — it warms the RRTMGP optics-table "
                  "cache outside the trace")
    assert any(isinstance(a, ast.Name) and a.id == "params"
               for call in warm for a in call.args), (
        "the warm-up must pass the concrete params pytree, not a placeholder")

    # ... and before the ERA5 load, so a broken physics config fails in seconds
    # rather than after minutes of data loading.
    era5 = [n.lineno for n in ast.walk(main)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "load_era5_samples"]
    assert era5, "ERA5 loader call not found — did main() get restructured?"
    assert min(c.lineno for c in warm) < min(era5), (
        "the warm-up must run BEFORE the ERA5 load, not after it")


def _confounded_yaml(**extra):
    neural = {"surface_drag_confounded": "core_does_not_read_the_key"}
    neural.update(extra)
    return {"neural_gcm": neural, "classical": {"turbulence": "louis"}}


def test_the_spectral_core_refuses_a_config_that_declares_the_key_unread():
    """#1464: that declaration is a property of the CORE, not of the file.

    The lat-lon core never reads ``neural_gcm.surface_drag``; the spectral one
    does. Running a config that declares the key unread on the spectral core
    would hand the learned arm no surface stress while the classical arm it is
    scored against carries Louis -- the confound, wearing the label that says
    it is not there."""
    with pytest.raises(SystemExit) as e:
        mod.check_surface_drag_confound(_confounded_yaml(), "neural_gcm",
                                        "spectral")
    assert "surface_drag" in str(e.value)


def test_the_latlon_core_accepts_the_same_config():
    """On the core the declaration is about, it is simply true."""
    mod.check_surface_drag_confound(_confounded_yaml(), "neural_gcm", "latlon")


def test_asking_for_the_drag_clears_the_refusal():
    """A config that enables the drag is equalised, whatever it declares."""
    mod.check_surface_drag_confound(
        _confounded_yaml(surface_drag=True, surface_drag_scheme="louis"),
        "neural_gcm", "spectral")
