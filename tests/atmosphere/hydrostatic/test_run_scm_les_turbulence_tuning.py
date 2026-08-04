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
def test_apply_trainable_params_actually_moves_a_parameter(scheme):
    """Non-vacuous: a no-op _apply_trainable_params must FAIL this.

    The previous version only reasserted the scheme name and the radiation
    config, both of which survive a no-op, so it proved nothing.
    """
    import jax.numpy as jnp
    from legoesm.training.trainable_params import TrainablePhysicsParams

    cfg = drv.build_physics_config(scheme, prescribed_fluxes=False)
    params = drv._initial_params(scheme, "extended")
    c = params.constraints[0]
    # perturb one raw leaf well away from its default
    bumped = TrainablePhysicsParams(
        raw_values={**params.raw_values,
                    c.name: params.raw_values[c.name] + jnp.asarray(1.5)},
        constraints=params.constraints,
    )
    out = drv._apply_trainable_params(cfg, bumped)

    def _target(config):
        sub = getattr(config.turbulence, scheme)
        return sub.params if scheme == "clubb" else sub

    before = float(getattr(_target(cfg), c.field))
    after = float(getattr(_target(out), c.field))
    assert after != before, (
        f"{scheme}.{c.field} unchanged ({before}); overrides were not applied"
    )
    assert out.turbulence.scheme == scheme
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


# --- registry namespace: class names collide across components -------------

@pytest.mark.parametrize("scheme", drv.TURBULENCE_SCHEMES)
def test_collected_params_are_fields_of_the_target_config(scheme):
    """Non-vacuous version of the earlier 'has tunable params' check.

    The registry's class names are NOT unique: the ocean's vertical mixing also
    registers a TKEConfig, so a name-only lookup collected ocean parameters and
    apply_param_overrides then raised mid-rollout with "TKEConfig has no
    field(s) ['Prandtl_tke0', 'lc_coeff', ...]". Assert every collected field
    actually exists on the config the overrides are spliced into.
    """
    cfg = drv.build_physics_config(scheme, prescribed_fluxes=False)
    sub = getattr(cfg.turbulence, scheme)
    target = sub.params if scheme == "clubb" else sub
    fields = set(type(target)._fields)
    params = drv._initial_params(scheme, "extended")
    assert params.constraints, scheme
    for c in params.constraints:
        assert c.field in fields, (
            f"{scheme}: collected {c.field!r} which is not a field of "
            f"{type(target).__name__}"
        )


def test_scheme_keys_are_all_atmospheric():
    for scheme in drv.TURBULENCE_SCHEMES:
        for key in drv._scheme_keys_for(scheme):
            assert key.startswith("atm.turb."), (scheme, key)


def test_ocean_tke_config_really_does_collide():
    """Proves the guard is not vacuous: the collision it defends against
    exists in the registry right now."""
    from legoesm.training.param_collector import build_registry
    keys = {m.scheme_key for m in build_registry()
            if m.config_class == "TKEConfig"}
    assert len(keys) > 1, "expected an atm/ocean TKEConfig name collision"
    assert any(not k.startswith("atm.turb.") for k in keys)


# --- CLUBB: prognostic path makes most parameters live ----------------------

def test_clubb_defaults_to_the_prognostic_path():
    """The diagnostic default reads only 6 CLUBBParams fields and just 4 have a
    live gradient, so tuning it exercises almost nothing."""
    cfg = drv.build_physics_config("clubb", prescribed_fluxes=False)
    assert cfg.turbulence.clubb is not None
    assert cfg.turbulence.clubb.prognostic is True


def test_clubb_prognostic_can_be_disabled():
    cfg = drv.build_physics_config("clubb", prescribed_fluxes=False,
                                   clubb_prognostic=False)
    assert cfg.turbulence.clubb.prognostic is False


def test_unimplemented_clubb_params_are_named_and_real():
    """Each listed parameter must exist on CLUBBParams (so the list cannot rot
    into naming nonsense) AND appear nowhere in clubb.py as an attribute read.
    """
    import re
    from pathlib import Path
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBParams

    fields = set(CLUBBParams._fields)
    src = Path(
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py"
    ).read_text()
    assert drv.CLUBB_UNIMPLEMENTED_PARAMS, "list should not be empty"
    for name in drv.CLUBB_UNIMPLEMENTED_PARAMS:
        assert name in fields, f"{name} is not a CLUBBParams field"
        # an attribute read would look like `.name` / `params.name`
        assert not re.search(rf"\.{re.escape(name)}\b", src), (
            f"{name} IS read in clubb.py; it should be removed from "
            "CLUBB_UNIMPLEMENTED_PARAMS"
        )


# --- surface layer: one derived config, identical on every arm --------------

def _bomex_case(nlev=32):
    from legoesm.atmosphere.forcing.scm.sam_case_scm import load_sam_scm_case
    return load_sam_scm_case("bomex", nlev=nlev, dt=60.0)


@pytest.mark.skipif(not _bomex_available(), reason="BOMEX gSAM deck not cached")
def test_surface_cd_is_the_neutral_log_law_at_the_les_roughness():
    """The SCM default Cd=1.5e-3 gives u*=0.339 m/s at BOMEX's 8.75 m/s trade
    wind; the LES wall model at z0=1e-4 gives 0.287. Derive it instead."""
    import math
    from legoesm import constants
    case = _bomex_case()
    surf = drv.build_surface_config(case)
    expect = (constants.kappa_vk / math.log(surf.z_ref / surf.z0)) ** 2
    assert surf.Cd_neutral == pytest.approx(expect, rel=1e-12)
    assert surf.z0 == pytest.approx(case.spec.les_z0_m)
    # z_ref must be the level whose wind the flux routine is handed
    assert surf.z_ref == pytest.approx(float(case.z_full[-1]))
    assert surf.Cd_neutral < 1.5e-3, "should be below the generic SCM default"
    u_ref = float(abs(case.u_profile[-1]))
    ustar = surf.Cd_neutral ** 0.5 * u_ref
    assert 0.24 < ustar < 0.33, f"u*={ustar} far from the LES 0.287 m/s"


@pytest.mark.skipif(not _bomex_available(), reason="BOMEX gSAM deck not cached")
def test_prescribed_flux_case_zeroes_the_heat_coefficient():
    case = _bomex_case()
    assert case.forcing.prescribe == "fluxes"
    assert drv.build_surface_config(case).Ch_neutral == 0.0


@pytest.mark.skipif(not _bomex_available(), reason="BOMEX gSAM deck not cached")
def test_most_is_refused_when_the_heat_flux_is_prescribed():
    """The MOST path derives heat from its own scaling and ignores Ch_neutral,
    so combining it with a prescribed flux double-counts."""
    case = _bomex_case()
    with pytest.raises(ValueError, match="counted twice"):
        drv.build_surface_config(case, bulk_scheme="most")


@pytest.mark.skipif(not _bomex_available(), reason="BOMEX gSAM deck not cached")
def test_every_arm_gets_the_identical_surface_config():
    """The surface boundary must not be a per-scheme degree of freedom."""
    case = _bomex_case()
    surf = drv.build_surface_config(case)
    seen = {}
    for scheme in drv.TURBULENCE_SCHEMES:
        cfg = drv.build_physics_config(
            scheme, prescribed_fluxes=True, surface=surf,
        )
        sub = getattr(cfg.turbulence, scheme)
        seen[scheme] = sub.surface
    first = seen[drv.TURBULENCE_SCHEMES[0]]
    for scheme, s in seen.items():
        assert s == first, f"{scheme} has a different surface config"
        assert s == surf


@pytest.mark.skipif(not _bomex_available(), reason="BOMEX gSAM deck not cached")
def test_surface_config_rejects_an_unknown_bulk_scheme():
    with pytest.raises(ValueError, match="not supported here"):
        drv.build_surface_config(_bomex_case(), bulk_scheme="nonsense")


def test_scm_refuses_most_with_a_prescribed_heat_flux():
    """Regression for a gap in the SCM's own guard.

    surface_layer.compute_surface_fluxes routes ("most", "coare3",
    "large_yeager") to the SAME compute_most_fluxes path, which ignores
    Ch_neutral, but _validate_prescribed_fluxes_no_double_count only listed
    coare3 and large_yeager -- so prescribe="fluxes" + bulk_scheme="most"
    silently double-counted the prescribed surface heat flux. The Ch_neutral
    check below it cannot catch that, because under MOST Ch_neutral is inert
    and is legitimately 0.
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
    from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig

    forcing = SCMForcing(prescribe="fluxes",
                         w_th_s=lambda _t: jnp.asarray(0.01))
    for bad in ("most", "coare3", "large_yeager"):
        cfg = drv.build_physics_config(
            "louis", prescribed_fluxes=True,
            surface=SurfaceLayerConfig(bulk_scheme=bad, Ch_neutral=0.0),
        )
        with pytest.raises(ValueError, match="double-count"):
            SingleColumnModel._validate_prescribed_fluxes_no_double_count(
                cfg, forcing,
            )
    # the constant path with Ch_neutral=0 remains allowed
    ok = drv.build_physics_config(
        "louis", prescribed_fluxes=True,
        surface=SurfaceLayerConfig(bulk_scheme="constant", Ch_neutral=0.0),
    )
    SingleColumnModel._validate_prescribed_fluxes_no_double_count(ok, forcing)


def test_main_exits_nonzero_when_an_arm_fails(tmp_path, monkeypatch):
    """A campaign whose arms all failed must not look like a clean sweep.

    Per-arm exceptions are caught deliberately, so without this the exit code
    was 0 no matter how many arms died -- which is exactly how a V100S run with
    a faulting arm was first mistaken for a success.
    """
    import types
    import numpy as _np

    fake_ref = types.SimpleNamespace(
        source_dir=str(tmp_path), window_hours=(4.0, 6.0), n_frames=13,
        mask=_np.ones(4, dtype=bool), weights=_np.full(4, 0.25),
        z_les=_np.linspace(20.0, 3000.0, 8), profiles={}, profiles_les={},
        scored_variables=lambda: ("theta", "qv"),
        window_label="4.00-6.00 h (13 frames)",
    )
    fake_case = types.SimpleNamespace(
        nlev=4, p_s=1.0e5, sigma_top=0.7, spec=types.SimpleNamespace(
            les_z0_m=1e-4, bulk_ch=None, bulk_ce=None),
        z_full=_np.linspace(3000.0, 20.0, 4),
        p_full=_np.linspace(7e4, 1e5, 4),
        u_profile=_np.full(4, -8.0), v_profile=_np.zeros(4),
        les_domain_top_m=3000.0,
        forcing=types.SimpleNamespace(prescribe="fluxes"),
    )
    monkeypatch.setattr(drv, "load_les_reference", lambda *a, **k: fake_ref)
    monkeypatch.setattr(drv, "load_sam_scm_case", lambda *a, **k: fake_case)
    monkeypatch.setattr(drv, "_half_pressures",
                        lambda case: _np.linspace(7e4, 1e5, 5))

    def _boom(*_a, **_k):
        raise RuntimeError("simulated CUDA fault")

    monkeypatch.setattr(drv, "evaluate_scheme", _boom)

    rc = drv.main(["--case", "bomex", "--les-dir", str(tmp_path),
                   "--outdir", str(tmp_path / "out"),
                   "--schemes", "louis", "--skip-tuning"])
    assert rc == 1, "a failed arm must produce a nonzero exit code"


# --- shared helpers, not re-derived formulas --------------------------------

def test_neutral_drag_matches_the_in_loop_log_law():
    """The helper must equal the solver's own neutral limit.

    compute_most_fluxes initialises u* = kappa*U/ln(z_ref/z0); the neutral drag
    is sqrt(Cd)*U. Two spellings of one formula can drift, so pin them.
    """
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.core.bulk_flux import neutral_drag_coefficient

    for z_ref, z0, u in ((20.0, 1e-4, 8.75), (10.0, 0.1, 5.0), (30.0, 1e-3, 12.0)):
        cd = float(neutral_drag_coefficient(z_ref, z0))
        in_loop_ustar = constants.kappa_vk * u / max(
            float(jnp.log(z_ref / z0)), 0.5)
        assert cd ** 0.5 * u == pytest.approx(in_loop_ustar, rel=1e-12), (
            z_ref, z0)


def test_no_inline_exner_or_virtual_temperature_in_this_stack():
    """These formulas have canonical homes; a local copy is how the SCM, the
    global model and the CRM drift apart."""
    import re
    from pathlib import Path
    targets = [
        "scripts/run/run_scm_les_turbulence_tuning.py",
        "packages/atmosphere/legoesm/atmosphere/forcing/scm/sam_case_scm.py",
        "packages/ml/legoesm/training/les_reference.py",
        "tests/atmosphere/hydrostatic/unit/test_sam_case_scm.py",
    ]
    exner = re.compile(r"p_ref\s*\)\s*\*\*\s*constants\.kappa")
    virt = re.compile(r"1\.0\s*/\s*constants\.epsilon\s*-\s*1\.0")
    for rel in targets:
        src = Path(rel).read_text()
        assert not exner.search(src), f"{rel}: inline Exner; use exner_function"
        assert not virt.search(src), (
            f"{rel}: inline virtual-T coefficient; use virtual_temperature")


# --- round-3 review fixes ---------------------------------------------------

def test_gradient_dead_arms_are_not_ranked():
    """A closure whose parameters are disconnected from the loss must not be
    ranked -- possibly first -- on its untouched default score."""
    import inspect
    src = inspect.getsource(drv._write_outputs)
    assert '_RANKABLE = {"ok", "tuned"}' in src, (
        "no_active_gradient / no_tunable_params must NOT be rankable")
    assert "no_active_gradient" not in src.split("_RANKABLE")[1][:120]


@pytest.mark.skipif(not _bomex_available(), reason="BOMEX gSAM deck not cached")
def test_surface_config_actually_reaches_the_kernel():
    """Non-vacuous: assert the surface config CHANGES the integration.

    The earlier tests checked registry entries and NamedTuple replacement, so a
    surface parameter that never reached the flux consumer (as bulk_ce did not)
    passed them all. Perturbing Cd must move the answer.
    """
    import jax
    import numpy as _np
    jax.config.update("jax_enable_x64", True)
    from legoesm.atmosphere.forcing.scm.sam_case_scm import load_sam_scm_case

    case = load_sam_scm_case("bomex", nlev=16, dt=60.0)
    surf = drv.build_surface_config(case)
    hours = 5 * 60.0 / 3600.0

    def _run(surface):
        cfg = drv.build_physics_config("louis", prescribed_fluxes=True,
                                       surface=surface)
        means, _ps = drv._rollout_means(
            None, base_cfg=cfg, case=case, dt=60.0, hours=hours,
            analysis_hours=hours, chunk_steps=5)
        return _np.asarray(means["u"])

    base = _run(surf)
    bumped = _run(surf._replace(Cd_neutral=surf.Cd_neutral * 3.0))
    assert not _np.allclose(base, bumped), (
        "tripling Cd_neutral did not change the wind profile; the surface "
        "config is not reaching the flux consumer")


@pytest.mark.skipif(not _bomex_available(), reason="BOMEX gSAM deck not cached")
def test_dt_must_divide_the_analysis_window(tmp_path, monkeypatch):
    """Otherwise the two sides average different spans while the report claims
    one window."""
    import types
    import numpy as _np
    fake_ref = types.SimpleNamespace(
        source_dir=str(tmp_path), window_hours=(4.0, 6.0), n_frames=13,
        mask=_np.ones(4, dtype=bool), weights=_np.full(4, 0.25),
        z_les=_np.linspace(20.0, 3000.0, 8), profiles={}, profiles_les={},
        scored_variables=lambda: ("theta", "qv"),
        window_label="4.00-6.00 h",
    )
    monkeypatch.setattr(drv, "load_les_reference", lambda *a, **k: fake_ref)
    monkeypatch.setattr(drv, "_half_pressures",
                        lambda case: _np.linspace(7e4, 1e5, 17))
    # 2 h window, dt = 7000 s does not divide 7200 s
    with pytest.raises(SystemExit, match="does not divide"):
        drv.main(["--case", "bomex", "--les-dir", str(tmp_path),
                  "--outdir", str(tmp_path / "o"), "--dt", "7000",
                  "--nlev", "16", "--schemes", "louis", "--skip-tuning"])
