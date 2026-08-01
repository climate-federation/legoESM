"""Direct tests for the LES-vs-SCM turbulence tuning driver.

Covers the things that make the comparison controlled (arms identical outside
turbulence, every scheme actually tunable, confounded cases refused) without
running a rollout, which needs a real LES reference and a GPU.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")

_ROOT = Path(__file__).resolve().parents[3]


def _load_driver():
    path = _ROOT / "scripts" / "run" / "run_scm_les_turbulence_tuning.py"
    spec = importlib.util.spec_from_file_location("scm_les_tuning", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scm_les_tuning"] = mod
    spec.loader.exec_module(mod)
    return mod


drv = _load_driver()


# --- scheme coverage --------------------------------------------------------

def test_scheme_list_matches_the_dispatch():
    """Every selectable closure except 'none' must be a candidate."""
    import inspect

    from legoesm.atmosphere.physics.turbulence import integration
    src = inspect.getsource(integration.get_turbulence_fn)
    for scheme in drv.TURBULENCE_SCHEMES:
        assert f'"{scheme}"' in src, f"{scheme} not in get_turbulence_fn"
    assert "none" not in drv.TURBULENCE_SCHEMES
    assert len(drv.TURBULENCE_SCHEMES) == 9


@pytest.mark.parametrize("scheme", drv.TURBULENCE_SCHEMES)
def test_every_scheme_builds_a_config(scheme):
    cfg = drv.build_physics_config(scheme, prescribed_fluxes=False)
    assert cfg.turbulence.scheme == scheme
    for other in ("radiation", "convection", "microphysics",
                  "gravity_wave_drag"):
        assert getattr(cfg, other).scheme == "none"


@pytest.mark.parametrize("scheme", drv.TURBULENCE_SCHEMES)
def test_every_scheme_exposes_tunable_parameters(scheme):
    """A scheme with no registered __param_spec__ cannot be tuned at all."""
    keys = drv._scheme_keys_for(scheme)
    assert keys, scheme
    params = drv._initial_params(scheme, "extended")
    assert params.constraints, f"{scheme} has no tier<=2 tunable params"


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unknown turbulence scheme"):
        drv.build_physics_config("louie", prescribed_fluxes=False)


# --- the control the whole comparison rests on ------------------------------

def test_arms_differ_only_in_turbulence():
    configs = {s: drv.build_physics_config(s, prescribed_fluxes=False)
               for s in drv.TURBULENCE_SCHEMES}
    drv._assert_arms_differ_only_in_turbulence(configs)   # must not raise


def test_arm_check_catches_a_non_turbulence_difference():
    configs = {s: drv.build_physics_config(s, prescribed_fluxes=False)
               for s in ("louis", "ysu")}
    bad = configs["ysu"]
    configs["ysu"] = bad._replace(
        radiation=bad.radiation._replace(scheme="gray")
    )
    with pytest.raises(RuntimeError, match="would not be controlled"):
        drv._assert_arms_differ_only_in_turbulence(configs)


@pytest.mark.parametrize("scheme", drv.TURBULENCE_SCHEMES)
def test_prescribed_fluxes_zero_the_bulk_heat_coefficient(scheme):
    """Otherwise the SCM double-counts the deck's prescribed SHF."""
    cfg = drv.build_physics_config(scheme, prescribed_fluxes=True)
    sub = getattr(cfg.turbulence, scheme)
    assert sub.surface.Ch_neutral == 0.0
    plain = drv.build_physics_config(scheme, prescribed_fluxes=False)
    assert getattr(plain.turbulence, scheme).surface.Ch_neutral != 0.0


# --- confounded cases are refused, not silently scored ----------------------

def test_dycoms_is_refused_for_a_radiation_mismatch():
    assert "dycoms" in drv._RADIATION_MISMATCH
    with pytest.raises(SystemExit, match="Stevens"):
        drv.main(["--case", "dycoms", "--les-dir", "/nonexistent"])


def test_dycoms_override_gets_past_the_refusal(tmp_path):
    """--allow-radiation-mismatch must change the failure mode, not be a no-op."""
    with pytest.raises(Exception) as excinfo:
        drv.main(["--case", "dycoms", "--les-dir", str(tmp_path),
                  "--allow-radiation-mismatch"])
    # it should now fail on the missing LES reference, not the refusal
    assert "Stevens" not in str(excinfo.value)


def test_unknown_scheme_on_the_cli_is_a_hard_error():
    with pytest.raises(SystemExit, match="unknown scheme"):
        drv.main(["--case", "bomex", "--les-dir", "/nonexistent",
                  "--schemes", "nope"])


# --- parameter plumbing -----------------------------------------------------

def test_clubb_targets_the_nested_params_tuple():
    """Full CLUBB's tunable leaves live in .params, not on CLUBBConfig."""
    keys = drv._scheme_keys_for("clubb")
    assert any("CLUBBParams" in k or "clubb" in k.lower() for k in keys)
    cfg = drv.build_physics_config("clubb", prescribed_fluxes=False)
    assert cfg.turbulence.clubb is not None


@pytest.mark.parametrize("scheme", ["louis", "ysu", "clubb"])
def test_apply_trainable_params_changes_the_config(scheme):
    cfg = drv.build_physics_config(scheme, prescribed_fluxes=False)
    params = drv._initial_params(scheme, "extended")
    out = drv._apply_trainable_params(cfg, params)
    assert out.turbulence.scheme == scheme
    # non-turbulence components untouched
    assert out.radiation == cfg.radiation


@pytest.mark.parametrize("scheme", ["louis", "tke"])
def test_bounds_hold_at_initialization(scheme):
    drv._assert_strict_bounds(drv._initial_params(scheme, "extended"))


def test_strict_bounds_reject_an_escaped_value():
    import jax.numpy as jnp
    from legoesm.training.trainable_params import TrainablePhysicsParams
    params = drv._initial_params("louis", "core")
    c = params.constraints[0]
    # a huge raw value saturates sigmoid to the bound => must be caught
    bad = TrainablePhysicsParams(
        raw_values={**params.raw_values,
                    c.name: jnp.asarray(1.0e9, dtype=jnp.float64)},
        constraints=params.constraints,
    )
    with pytest.raises(RuntimeError, match="escaped strict bounds"):
        drv._assert_strict_bounds(bad)


# --- CLI --------------------------------------------------------------------

def test_parse_args_defaults():
    args = drv.parse_args(["--case", "bomex", "--les-dir", "x"])
    assert args.schemes == "all"
    assert args.tier == "extended"
    assert args.optimizer == "muon"
    assert args.analysis_hours == drv.DEFAULT_ANALYSIS_HOURS


def test_scored_variables_exclude_fluxes():
    from legoesm.training.les_reference import SCORED_VARIABLES
    assert "wth" not in SCORED_VARIABLES and "wqv" not in SCORED_VARIABLES
    assert set(SCORED_VARIABLES) == {"theta", "qv", "u", "v"}


# --- adversarial-review fixes ----------------------------------------------

def test_scored_variables_default_excludes_the_wind():
    """The SCM uses a constant-Cd drag against the LES's z0 log-law wall model,
    so tuning against u/v would absorb a surface-drag mismatch into the
    turbulence parameters. Heat and moisture fluxes ARE matched."""
    args = drv.parse_args(["--case", "bomex", "--les-dir", "x"])
    assert args.score_variables == "theta,qv"


def test_unknown_scored_variable_is_a_hard_error():
    with pytest.raises(SystemExit, match="unknown scored variable"):
        drv.main(["--case", "bomex", "--les-dir", "/nonexistent",
                  "--score-variables", "theta,notavar"])


def test_empty_scored_variable_selection_is_a_hard_error():
    with pytest.raises(SystemExit, match="selected nothing"):
        drv.main(["--case", "bomex", "--les-dir", "/nonexistent",
                  "--score-variables", " , "])


def test_mismatched_hours_is_refused_as_a_window_confound(tmp_path, monkeypatch):
    """A user --hours that does not end where the LES reference ends would
    compare different time windows (e.g. SCM 4-6 h vs LES 21-23 h)."""
    import types

    fake_ref = types.SimpleNamespace(
        source_dir=str(tmp_path), window_hours=(21.0, 23.0), n_frames=5,
        mask=__import__("numpy").ones(4, dtype=bool),
        weights=__import__("numpy").full(4, 0.25),
        scored_variables=lambda: ("theta", "qv"),
    )
    monkeypatch.setattr(drv, "load_les_reference", lambda *a, **k: fake_ref)
    monkeypatch.setattr(drv, "load_sam_scm_case",
                        lambda *a, **k: types.SimpleNamespace(
                            nlev=4, p_s=1.0e5, sigma_top=0.7,
                            z_full=__import__("numpy").linspace(3000, 20, 4),
                            les_domain_top_m=3000.0,
                            forcing=types.SimpleNamespace(prescribe="fluxes")))
    with pytest.raises(SystemExit, match="different times"):
        drv.main(["--case", "bomex", "--les-dir", str(tmp_path),
                  "--hours", "6"])


# --- the rollout actually runs (would have caught the lax.cond shape bug) ---

def _bomex_available() -> bool:
    try:
        from legoesm.atmosphere.forcing.scm.sam_case_scm import (
            SAM_SCM_CASES, resolve_sam_case_dir,
        )
        resolve_sam_case_dir(SAM_SCM_CASES["bomex"].gsam_dir)
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _bomex_available(), reason="BOMEX gSAM deck not cached")
def test_rollout_runs_and_padding_shapes_match():
    """Integrate a few steps on CPU with a chunk count that FORCES padding.

    The first pipeline run failed with

        cond branches must have equal output types ... float64[1] vs float64[]

    because p_s.data is (1, 1, 1) so [0, 0] left a shape-(1,) entry while
    masked_step's padding was scalar. nsteps=5 with chunk_steps=4 pads the
    last chunk, so this exercises the lax.cond branch that mismatched.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.atmosphere.forcing.scm.sam_case_scm import load_sam_scm_case

    case = load_sam_scm_case("bomex", nlev=16, dt=60.0)
    cfg = drv.build_physics_config("louis", prescribed_fluxes=True)
    hours = 5 * 60.0 / 3600.0                      # exactly 5 steps
    means, ps_hist = drv._rollout_means(
        None, base_cfg=cfg, case=case, dt=60.0, hours=hours,
        analysis_hours=hours, chunk_steps=4,        # 2 chunks => 3 padded steps
    )
    import numpy as _np
    assert ps_hist.shape == (5,), f"padding not sliced off: {ps_hist.shape}"
    for name in ("T", "qv", "u", "v"):
        arr = _np.asarray(means[name])
        assert arr.shape == (16,), (name, arr.shape)
        assert _np.all(_np.isfinite(arr)), name
    # padding is zeros; if it leaked into the mean p_s would be far from p_s
    drv._assert_surface_pressure_static(ps_hist, case.p_s)


@pytest.mark.skipif(not _bomex_available(), reason="BOMEX gSAM deck not cached")
def test_padding_would_be_caught_by_the_pressure_assert():
    """The p_s drift assert must actually fire on a bad history."""
    import numpy as _np
    with pytest.raises(RuntimeError, match="surface pressure drifted"):
        drv._assert_surface_pressure_static(_np.zeros(5), 1.015e5)
