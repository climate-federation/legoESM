"""Unit tests for the legoESM configuration wizard's pure core.

Two jobs:

* **Versioning guards.** The wizard must not drift from the model as it evolves.
  These tests assert every curated menu is derived from / a subset of the live
  model registries, so a model-side rename or removal fails *here* (forcing the
  wizard to be updated in the same change) instead of silently emitting a dead
  flag.
* **Routing correctness.** Each branch's ``build_plan`` produces the real
  existing launcher command, and ``emit_bundle`` writes a version-stamped bundle.

The wizard lives under ``scripts/experiment/`` (tooling, not installed source),
so it is imported by path — the same mechanism ``tests/unit/test_init_experiment.py``
uses.
"""
from __future__ import annotations

import importlib
import os
import shlex
import sys
from pathlib import Path

import pytest

# Apple-Silicon Metal JAX is broken for this repo; the registry calls import the
# atmosphere package (and thus JAX) lazily, so pin CPU before anything triggers it.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXP_DIR = _REPO_ROOT / "scripts" / "experiment"


@pytest.fixture(scope="module")
def wc():
    if str(_EXP_DIR) not in sys.path:
        sys.path.insert(0, str(_EXP_DIR))
    return importlib.import_module("wizard_core")


# =====================================================================
# Versioning: menus are derived from the live model registries
# =====================================================================

def test_schema_version_is_semver(wc):
    parts = wc.WIZARD_SCHEMA_VERSION.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_provenance_keys(wc):
    p = wc.provenance()
    assert p["wizard_schema_version"] == wc.WIZARD_SCHEMA_VERSION
    assert set(p) == {"wizard_schema_version", "legoesm_version", "git_commit", "created_utc"}
    assert p["legoesm_version"]  # non-empty (real version or "unknown")


def test_integrators_subset_of_live_dispatch(wc):
    from legoesm.timestepping.dispatch import available_integrators
    avail = set(available_integrators())
    opts = wc.integrator_options()
    assert opts, "curated integrator menu is empty"
    # Every offered integrator must be a real dispatch key (no dead flags).
    assert set(opts) <= avail


def test_dynamics_options_match_model(wc):
    from legoesm.atmosphere.dynamics import DYNAMICS_OPTIONS
    assert wc.dynamics_options() == list(DYNAMICS_OPTIONS)


def test_discretizations_exclude_aliases_and_data_driven(wc):
    from legoesm.atmosphere.dynamics import DISCRETIZATION_OPTIONS
    opts = wc.discretization_options()
    assert set(opts) <= set(DISCRETIZATION_OPTIONS)
    for hidden in ("finite_volume", "centered", "sfno", "u_cast"):
        assert hidden not in opts
    # ...but data-driven can be opted back in.
    assert "sfno" in wc.discretization_options(include_data_driven=True)


def test_grid_types_from_driver_matrix_without_plane(wc):
    grids = set(wc.grid_types())
    combo_grids = {c["grid_type"] for c in wc.driver_combos()}
    assert grids == combo_grids - {"plane"}
    assert "cubed_sphere" in grids


def test_aimip_variants_nonempty_subset(wc):
    from legoesm.driver.config import AIMIP_VARIANTS
    assert wc.aimip_variants() == [v for v in AIMIP_VARIANTS if v]


def test_amip_dataset_and_radiation_menus(wc):
    assert "analytical" in wc.AMIP_DATASETS  # the no-data default
    assert set(wc.RADIATION_SCHEMES) == {"gray", "rrtmg", "rrtmgp"}


def test_aimip_launchable_variants_subset(wc):
    from legoesm.driver.config import AIMIP_VARIANTS
    launch = wc.aimip_launchable_variants()
    assert launch, "no launchable AIMIP variants discovered"
    # launchable ⊆ non-empty AIMIP_VARIANTS, each with a shipped overlay file.
    assert set(launch) <= {v for v in AIMIP_VARIANTS if v}
    assert "classical" in launch and "column_nn" in launch
    # sfno_full's overlay now ships beside the base suite (full-atmosphere
    # emulator rung of the ladder), so it is launchable from the wizard.
    assert "sfno_full" in launch


def test_amip_menus_subset_of_run_amip_parser(wc):
    """AMIP menus must stay a subset of run_amip.py's argparse choices.

    Drift guard (same spirit as the integrator/grid registry tests): a renamed
    --dataset/--radiation/--grid-type choice in run_amip.py fails here instead of
    emitting a flag the launcher rejects.  run_amip.py reconfigures sys.stdout at
    import (clashes with pytest capture), so introspect it in a subprocess.
    """
    import json
    import subprocess
    code = (
        "import json,importlib.util,pathlib;"
        f"p=pathlib.Path(r'{_REPO_ROOT}')/'scripts'/'run'/'run_amip.py';"
        "s=importlib.util.spec_from_file_location('_ra',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "pr=m.build_arg_parser();"
        "c={o:list(a.choices) for a in pr._actions if a.choices for o in a.option_strings};"
        "print(json.dumps(c))"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert res.returncode == 0, res.stderr
    choices = json.loads(res.stdout.strip().splitlines()[-1])
    assert set(wc.RADIATION_SCHEMES) <= set(choices["--radiation"])
    assert set(wc.AMIP_DATASETS) <= set(choices["--dataset"])
    assert set(wc.grid_types()) <= set(choices["--grid-type"])
    # The AMIP integrator menu must stay within run_amip.py's restricted set
    # (it omits ssp_rk54_scan, which the general dispatch table has).
    assert set(wc.AMIP_INTEGRATORS) <= set(choices["--time-integrator"])


def test_coupled_presets_match_registry(wc):
    from legoesm.driver.coupled_config import PRESETS
    keys = [k for k, _ in wc.coupled_presets()]
    assert keys == list(PRESETS)
    # Descriptions come from the preset docstrings (so the land ladder tracks).
    descs = dict(wc.coupled_presets())
    assert descs["slab_richards"]  # has a description


def test_complexity_vocab_matches_model(wc):
    from legoesm.components import LandComplexity, OceanComplexity
    assert wc.land_complexity_levels() == [c.value for c in LandComplexity]
    assert wc.ocean_complexity_levels() == [c.value for c in OceanComplexity]


def test_templates_and_machines_discovered(wc):
    assert wc.templates(), "no experiment templates discovered"
    assert "default" in wc.machine_names()


# =====================================================================
# Gating predicates
# =====================================================================

def test_legal_discretizations(wc):
    sw = wc.legal_discretizations("shallow_water")
    assert "cdgrid" in sw and "sfno" not in sw
    hy = wc.legal_discretizations("hydrostatic")
    assert "spectral" in hy


def test_legal_grids_spectral_is_gaussian_only(wc):
    assert wc.legal_grids("hydrostatic", "spectral") == ["gaussian"]
    assert "cubed_sphere" in wc.legal_grids("shallow_water", "cdgrid")


def test_is_supported(wc):
    assert wc.is_supported("hydrostatic", "cdgrid", "cubed_sphere")
    assert not wc.is_supported("shallow_water", "spectral", "cubed_sphere")


def test_legal_integrators_spectral_pe_locked(wc):
    assert wc.legal_integrators("hydrostatic", "spectral") == ["ssp_rk54"]
    assert wc.legal_integrators("hydrostatic", "cdgrid") == wc.integrator_options()


def test_backend_and_mpi_helpers(wc):
    assert wc.backend_to_platforms("gpu") == "cuda"
    assert wc.backend_to_platforms("mpi") == "cpu"
    assert wc.wrap_mpi("legoesm run config.yaml", 4) == "mpirun -np 4 legoesm run config.yaml"


def test_cmd_shell_quotes_paths(wc):
    # A path with a space must be quoted so the emitted run.sh stays valid + safe.
    assert "'/tmp/a b/run.py'" in wc._cmd(["python", "/tmp/a b/run.py", "--x", "1"])
    # Clean tokens are left bare (byte-identical to a plain join).
    assert wc._cmd(["python", "x.py"]) == "python x.py"


def test_render_run_sh_quotes_workdir(wc):
    sh = wc.render_run_sh({"jax_platforms": "cpu"}, run_cmd="python x.py",
                          workdir="/tmp/a b/repo")
    assert "cd '/tmp/a b/repo'" in sh
    # Default (no workdir) keeps the bundle-dir form.
    sh2 = wc.render_run_sh({"jax_platforms": "cpu"}, run_cmd="python x.py")
    assert 'cd "$(dirname "$0")"' in sh2


# =====================================================================
# Per-branch build_plan routing
# =====================================================================

def _atm_global_answers(wc, *, backend="cpu", bits=32):
    """A valid global-atmosphere answer set anchored to a real template."""
    from legoesm.config import Config
    tmpls = wc.templates()
    template = "3d_idealized/held_suarez"
    if template not in tmpls:
        template = next((t for t in tmpls if "held_suarez" in t), tmpls[0])
    mt = Config.from_yaml(str(wc._resolve_template(template))).get(
        "atmosphere.dynamics", "hydrostatic"
    )
    disc = wc.legal_discretizations(mt)[0]
    grid = wc.legal_grids(mt, disc)[0]
    integ = wc.legal_integrators(mt, disc)[0]
    ans = {
        "objective": "simulate", "component": "atmosphere", "atm_kind": "global",
        "template": template, "model_type": mt, "discretization": disc,
        "grid_type": grid, "integrator": integ, "resolution": 16,
        "duration_hours": 48.0, "precision_bits": bits, "backend": backend,
    }
    if mt != "shallow_water":
        ans["nlev"] = 20
    if backend == "mpi":
        ans["mpi_ranks"] = 4
    return ans


def test_build_atm_global(wc):
    plan = wc.build_plan(_atm_global_answers(wc))
    assert plan.kind == "atm_global"
    assert plan.config is not None
    assert plan.run_cmd == "legoesm run config.yaml"
    assert plan.enable_x64 is False
    # The config is strict-valid (build_plan calls validate_strict).
    plan.config.to_experiment_config().validate_strict()


def test_build_atm_global_x64_and_mpi(wc):
    plan = wc.build_plan(_atm_global_answers(wc, backend="mpi", bits=64))
    assert plan.enable_x64 is True
    assert plan.run_cmd.startswith("mpirun -np 4 legoesm run config.yaml")
    assert plan.jax_platforms == "cpu"


def test_build_atm_global_rejects_unsupported_combo(wc):
    ans = _atm_global_answers(wc)
    ans["discretization"] = "spectral"
    ans["grid_type"] = "cubed_sphere"  # spectral needs gaussian
    with pytest.raises(ValueError):
        wc.build_plan(ans)


def test_build_coupled_land_complexity(wc):
    ans = {
        "objective": "simulate", "component": "coupled", "preset": "slab_richards",
        "resolution": 16, "nlev": 20, "days": 30, "radiation": "gray",
        "precision_bits": 64, "backend": "cpu",
    }
    plan = wc.build_plan(ans)
    assert plan.kind == "coupled" and plan.config is None
    assert "run_coupled.py" in plan.run_cmd
    assert "--preset slab_richards" in plan.run_cmd
    assert plan.enable_x64 is True


def test_build_coupled_rejects_unknown_preset(wc):
    ans = {"objective": "simulate", "component": "coupled", "preset": "nope",
           "precision_bits": 32, "backend": "cpu"}
    with pytest.raises(ValueError):
        wc.build_plan(ans)


def test_ocean_case_grids_derived_per_case(wc):
    cg = wc.ocean_case_grids()
    assert cg, "no ocean cases discovered"
    # A global case runs on the global grids...
    assert "cubed_sphere" in cg.get("barotropic_wave", [])
    # ...but a regional gyre case does NOT (this is the bug the gating fixes).
    dg = cg.get("barotropic_double_gyre", [])
    assert dg and "cubed_sphere" not in dg
    assert wc.legal_ocean_grids("barotropic_wave") == cg["barotropic_wave"]


def test_build_ocean(wc):
    ans = {"objective": "simulate", "component": "ocean",
           "ocean_case": "barotropic_wave", "ocean_grid": "latlon",
           "quick": True, "precision_bits": 64, "backend": "cpu"}
    plan = wc.build_plan(ans)
    assert plan.kind == "ocean"
    assert "run_ocean_test_matrix.py" in plan.run_cmd
    assert "--only barotropic_wave" in plan.run_cmd
    assert "--grid latlon" in plan.run_cmd
    assert "--quick" in plan.run_cmd


def test_build_ocean_rejects_invalid_case_grid(wc):
    # barotropic_double_gyre is regional-only; cubed_sphere would run zero tests.
    ans = {"objective": "simulate", "component": "ocean",
           "ocean_case": "barotropic_double_gyre", "ocean_grid": "cubed_sphere",
           "quick": True, "precision_bits": 64, "backend": "cpu"}
    with pytest.raises(ValueError):
        wc.build_plan(ans)


def test_build_les_crm_abl_f32(wc):
    ans = {"objective": "simulate", "component": "atmosphere", "atm_kind": "les_crm",
           "les_case": "abl", "abl_case": "gabls1", "precision_bits": 32, "backend": "cpu"}
    plan = wc.build_plan(ans)
    assert plan.kind == "les_crm"
    assert "run_les_plane.py" in plan.run_cmd
    assert "--case gabls1" in plan.run_cmd
    assert "--f32" in plan.run_cmd


def test_build_scm(wc):
    ans = {"objective": "simulate", "component": "atmosphere", "atm_kind": "scm",
           "scm_case": "rce", "days": 20.0, "nlev": 30,
           "precision_bits": 64, "backend": "cpu"}
    plan = wc.build_plan(ans)
    assert plan.kind == "scm"
    assert plan.run_cmd.split()[1].endswith("run_scm_test_matrix.py")
    assert "rce" in plan.run_cmd
    assert "--days 20.0" in plan.run_cmd


def test_build_train_spectral_forces_x64(wc):
    ans = {"objective": "train", "train_mode": "neural_gcm_spectral",
           "n_max": 42, "n_levels": 10, "epochs": 5, "lr": 3e-4,
           "train_days": 30, "year": 2015, "cache_dir": "data/era5_cache",
           "backend": "gpu", "precision_bits": 64}
    plan = wc.build_plan(ans)
    assert plan.kind == "train"
    assert "train_neural_gcm_spectral.py" in plan.run_cmd
    assert plan.enable_x64 is True
    assert plan.jax_platforms == "cuda"


def test_build_train_rejects_cli_less_mode(wc):
    ans = {"objective": "train", "train_mode": "physics_params", "backend": "cpu",
           "precision_bits": 64}
    with pytest.raises(ValueError):
        wc.build_plan(ans)


def test_build_plan_unknown_combo_raises(wc):
    with pytest.raises(ValueError):
        wc.build_plan({"objective": "simulate", "component": "atmosphere",
                       "atm_kind": "nope"})


def _amip_answers(wc, *, discretization="cdgrid", grid="cubed_sphere",
                  bits=32, backend="cpu"):
    """A valid AMIP answer set; (disc, grid) is a driver-supported triple."""
    integ = wc.legal_amip_integrators(discretization)[0]
    ans = {
        "objective": "simulate", "component": "atmosphere", "atm_kind": "amip",
        "dataset": "analytical", "discretization": discretization,
        "grid_type": grid, "integrator": integ,
        "resolution": 16, "nlev": 30, "days": 10, "dt": 600.0,
        "radiation": "gray", "precision_bits": bits, "backend": backend,
    }
    if backend == "mpi":
        ans["mpi_ranks"] = 4
    return ans


def test_build_amip(wc):
    plan = wc.build_plan(_amip_answers(wc))
    assert plan.kind == "amip" and plan.config is None
    assert "run_amip.py" in plan.run_cmd
    assert "--dataset analytical" in plan.run_cmd
    assert "--grid-type cubed_sphere" in plan.run_cmd
    assert "--discretization cdgrid" in plan.run_cmd  # explicit, not inferred
    assert "--radiation gray" in plan.run_cmd
    assert plan.enable_x64 is False  # 32-bit, non-Gaussian


def test_build_amip_gaussian_forces_x64(wc):
    plan = wc.build_plan(_amip_answers(wc, discretization="spectral", grid="gaussian", bits=32))
    assert plan.enable_x64 is True  # spectral transform requires x64
    assert "--discretization spectral" in plan.run_cmd
    assert "--time-integrator ssp_rk54" in plan.run_cmd


def test_build_amip_rejects_unsupported_combo(wc):
    # spectral is gaussian-only; pairing it with cubed_sphere must fail fast.
    ans = _amip_answers(wc, discretization="spectral", grid="cubed_sphere")
    with pytest.raises(ValueError):
        wc.build_plan(ans)


def test_legal_amip_integrators_excludes_non_run_amip(wc):
    # run_amip.py has no ssp_rk54_scan, so the AMIP menu must not offer it even
    # though the general dispatch table (integrator_options) does.
    cdg = wc.legal_amip_integrators("cdgrid")
    assert cdg, "no AMIP integrators for cdgrid"
    assert set(cdg) <= set(wc.AMIP_INTEGRATORS)
    if "ssp_rk54_scan" in wc.integrator_options():
        assert "ssp_rk54_scan" not in cdg
    # Spectral PE stays locked to SSP-RK54 (and it is in the AMIP set).
    assert wc.legal_amip_integrators("spectral") == ["ssp_rk54"]


def test_build_amip_rejects_non_run_amip_integrator(wc):
    ans = _amip_answers(wc, discretization="cdgrid")
    ans["integrator"] = "ssp_rk54_scan"  # valid dispatch key, not a run_amip choice
    with pytest.raises(ValueError):
        wc.build_plan(ans)


def test_build_amip_mpi_wrap(wc):
    plan = wc.build_plan(_amip_answers(wc, backend="mpi"))
    assert plan.run_cmd.startswith("mpirun -np 4 ")
    assert plan.jax_platforms == "cpu"


def test_build_amip_observed_requires_forcing_path(wc):
    # cobe/hadisst without an SST file would make run_amip.py abort → fail fast.
    ans = _amip_answers(wc)
    ans["dataset"] = "cobe"
    with pytest.raises(ValueError):
        wc.build_plan(ans)


def test_build_amip_observed_with_forcing_path(wc):
    ans = _amip_answers(wc)
    ans["dataset"] = "hadisst"
    ans["forcing_path"] = "/data/sst.nc"
    plan = wc.build_plan(ans)
    assert "--dataset hadisst" in plan.run_cmd
    assert "--forcing-path /data/sst.nc" in plan.run_cmd
    assert any("/data/sst.nc" in n for n in plan.notes)


def test_build_aimip_variant(wc):
    variant = wc.aimip_launchable_variants()[0]
    ans = {"objective": "train", "train_mode": "aimip", "aimip_variant": variant,
           "smoke": True, "backend": "gpu", "precision_bits": 64}
    plan = wc.build_plan(ans)  # routes train → _build_aimip
    assert plan.kind == "aimip"
    assert "run_aimip.py" in plan.run_cmd
    # Suite passed absolute so it resolves from the repo-root workdir.
    assert f"--suite {wc._REPO_ROOT / 'config/aimip/aimip_suite.yaml'}" in plan.run_cmd
    assert f"--variants {variant}" in plan.run_cmd
    assert "--smoke" in plan.run_cmd
    assert plan.enable_x64 is True       # spectral forces x64
    assert plan.workdir == str(wc._REPO_ROOT)
    assert plan.jax_platforms == "cuda"  # gpu backend


def test_build_aimip_no_smoke(wc):
    variant = wc.aimip_launchable_variants()[0]
    ans = {"objective": "train", "train_mode": "aimip", "aimip_variant": variant,
           "smoke": False, "backend": "cpu", "precision_bits": 64}
    assert "--smoke" not in wc.build_plan(ans).run_cmd


def test_build_aimip_rejects_unlaunchable_variant(wc, monkeypatch):
    # All shipped variants now have base-suite overlays, so simulate a variant
    # whose overlay is missing (renamed/removed file) via the launchable list.
    monkeypatch.setattr(
        wc, "aimip_launchable_variants", lambda *a, **k: ["classical"],
    )
    ans = {"objective": "train", "train_mode": "aimip", "aimip_variant": "sfno_full",
           "smoke": False, "backend": "cpu", "precision_bits": 64}
    with pytest.raises(ValueError):
        wc.build_plan(ans)


# =====================================================================
# emit_bundle: version-stamped, runnable bundle
# =====================================================================

def test_emit_coupled_bundle(wc, tmp_path):
    import yaml
    ans = {"objective": "simulate", "component": "coupled", "preset": "slab_simple",
           "resolution": 16, "nlev": 20, "days": 30, "radiation": "gray",
           "precision_bits": 64, "backend": "cpu"}
    plan = wc.build_plan(ans)
    out = tmp_path / "run1"
    machine = wc.load_machine_profile("default")
    written = wc.emit_bundle(plan, out, name="t1", machine=machine)

    assert "config" not in written  # coupled path has no config.yaml
    run_sh = Path(written["run_sh"]).read_text()
    assert "export JAX_PLATFORMS=cpu" in run_sh
    assert "export JAX_ENABLE_X64=1" in run_sh  # 64-bit
    assert "run_coupled.py" in run_sh
    assert os.access(written["run_sh"], os.X_OK)

    rec = yaml.safe_load(Path(written["wizard_yaml"]).read_text())
    assert rec["wizard_schema_version"] == wc.WIZARD_SCHEMA_VERSION
    assert rec["experiment"]["kind"] == "coupled"
    assert rec["selections"]["preset"] == "slab_simple"
    assert rec["launch"]["command"] == plan.run_cmd
    assert rec["notes"]


def test_emit_atm_bundle_writes_config(wc, tmp_path):
    plan = wc.build_plan(_atm_global_answers(wc))
    out = tmp_path / "atm"
    machine = wc.load_machine_profile("default")
    written = wc.emit_bundle(plan, out, name="atm", machine=machine)
    assert "config" in written
    assert Path(written["config"]).is_file()
    run_sh = Path(written["run_sh"]).read_text()
    assert "legoesm run config.yaml" in run_sh
    assert "export JAX_ENABLE_X64=1" not in run_sh  # 32-bit default


def test_emit_amip_bundle_no_config(wc, tmp_path):
    plan = wc.build_plan(_amip_answers(wc, bits=64))
    out = tmp_path / "amip_run"
    written = wc.emit_bundle(plan, out, name="amip", machine=wc.load_machine_profile("default"))
    assert "config" not in written  # AMIP routes to run_amip.py, no config.yaml
    run_sh = Path(written["run_sh"]).read_text()
    assert "run_amip.py" in run_sh
    assert "export JAX_ENABLE_X64=1" in run_sh  # 64-bit


def test_emit_aimip_bundle(wc, tmp_path):
    import yaml
    variant = wc.aimip_launchable_variants()[0]
    plan = wc.build_plan({"objective": "train", "train_mode": "aimip",
                          "aimip_variant": variant, "smoke": True,
                          "backend": "cpu", "precision_bits": 64})
    out = tmp_path / "aimip_run"
    written = wc.emit_bundle(plan, out, name="aimip", machine=wc.load_machine_profile("default"))
    assert "config" not in written
    run_sh = Path(written["run_sh"]).read_text()
    assert "run_aimip.py" in run_sh
    assert "export JAX_ENABLE_X64=1" in run_sh  # spectral forces x64
    # Launches from the repo root (not the bundle dir) so the suite's relative
    # base:/overlay/cache paths resolve; the suite itself is passed absolute.
    # (shlex.quote leaves a metachar-free path bare.)
    assert f"cd {shlex.quote(str(wc._REPO_ROOT))}" in run_sh
    assert 'cd "$(dirname "$0")"' not in run_sh
    assert str(wc._REPO_ROOT / "config/aimip/aimip_suite.yaml") in run_sh
    rec = yaml.safe_load(Path(written["wizard_yaml"]).read_text())
    assert rec["experiment"]["kind"] == "aimip"
    assert rec["selections"]["aimip_variant"] == variant


def test_emit_refuses_nonempty_dir(wc, tmp_path):
    ans = {"objective": "simulate", "component": "ocean",
           "ocean_case": "overflow", "ocean_grid": "cubed_sphere",
           "quick": True, "precision_bits": 64, "backend": "cpu"}
    plan = wc.build_plan(ans)
    out = tmp_path / "busy"
    out.mkdir()
    (out / "stuff.txt").write_text("x")
    with pytest.raises(ValueError):
        wc.emit_bundle(plan, out, name="x", machine=wc.load_machine_profile("default"))
