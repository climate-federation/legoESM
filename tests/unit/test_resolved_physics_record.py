"""Resolved physics in the run manifest, strict replay and the closure check
(docs/architecture/reproducibility.md, P0 step 1).

Each lane records the scheme configs it hands its kernels into
``config.resolved_physics``; ``legoesm reproduce`` refuses a manifest whose
configuration today's code does not rebuild exactly, or whose rerun resolves
different physics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.restart import (
    RESOLVED_PHYSICS_KEYS,
    RUN_MANIFEST_FILENAME,
    physics_closure_rows,
    physics_config_from_record,
    read_run_manifest,
    record_resolved_physics,
    resolved_physics_record,
    write_run_manifest,
)

if TYPE_CHECKING:
    from legoesm.driver.model_driver import ModelDriver

DAYS_2_STEPS = 1201.0 / 86400.0   # dt=600 s -> 2 steps


def _turb(**louis):
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    tc = TurbulenceConfig(scheme="louis")
    return tc._replace(louis=tc.louis._replace(**louis)) if louis else tc


# --- record + closure helpers -------------------------------------------------

def test_record_tags_types_and_closure_detects_a_changed_leaf() -> None:
    a = resolved_physics_record({"turbulence": _turb()})
    assert a["turbulence"]["__type__"].endswith(":TurbulenceConfig")
    assert physics_closure_rows(a, copy.deepcopy(a)) == []
    field = next(k for k, v in _turb().louis._asdict().items()
                 if isinstance(v, float))
    b = resolved_physics_record(
        {"turbulence": _turb(**{field: getattr(_turb().louis, field) * 2})})
    rows = physics_closure_rows(a, b)
    assert [r[0] for r in rows] == [f"turbulence.louis.{field}"]


def test_closure_treats_nan_as_equal_and_flags_missing_fields() -> None:
    rec = {"x": {"v": float("nan"), "w": 1.0}}
    assert physics_closure_rows(rec, json.loads(json.dumps(rec))) == []
    rows = physics_closure_rows(rec, {"x": {"v": float("nan")}})
    assert rows == [("x.w", 1.0, "<absent>")]


def test_record_resolved_physics_writes_once_and_refuses_a_different_record(
        tmp_path: Path) -> None:
    write_run_manifest(tmp_path, ExperimentConfig())
    rec = resolved_physics_record({"turbulence": _turb()})
    record_resolved_physics(tmp_path, rec)
    assert read_run_manifest(tmp_path)["config"]["resolved_physics"] == rec
    record_resolved_physics(tmp_path, rec)                 # same: a resume
    other = resolved_physics_record({"turbulence": _turb(l_mix_max=50.0)})
    with pytest.raises(RuntimeError, match="different resolved physics"):
        record_resolved_physics(tmp_path, other)
    assert read_run_manifest(tmp_path)["config"]["resolved_physics"] == rec


def _full_record(**configs):
    return resolved_physics_record({**dict.fromkeys(RESOLVED_PHYSICS_KEYS),
                                    **configs})


def _driver_at(tmp_path: Path, **kw):
    from legoesm.driver.model_driver import ModelDriver
    d = ModelDriver(ExperimentConfig(), **kw)
    d._output_dir = tmp_path
    return d


def test_closure_against_the_reference_record(tmp_path: Path) -> None:
    write_run_manifest(tmp_path, ExperimentConfig())
    rec = _full_record(turbulence=_turb())
    _driver_at(tmp_path, expected_resolved_physics=rec)._record_resolved_physics(
        turbulence=_turb())                                # equal: written
    assert read_run_manifest(tmp_path)["config"]["resolved_physics"] == rec
    out = tmp_path / "other"
    write_run_manifest(out, ExperimentConfig())
    with pytest.raises(RuntimeError, match="reference run's record"):
        _driver_at(out, expected_resolved_physics=rec)._record_resolved_physics(
            turbulence=_turb(l_mix_max=50.0))
    assert "resolved_physics" not in read_run_manifest(out)["config"]


def test_resume_with_different_physics_is_refused(tmp_path: Path) -> None:
    write_run_manifest(tmp_path, ExperimentConfig())
    _driver_at(tmp_path)._record_resolved_physics(turbulence=_turb())
    _driver_at(tmp_path)._record_resolved_physics(turbulence=_turb())  # resume
    with pytest.raises(RuntimeError, match="different resolved physics"):
        _driver_at(tmp_path)._record_resolved_physics(
            turbulence=_turb(l_mix_max=50.0))


class _FakeComm:
    def __init__(self, rank, root_failed):
        self.rank, self.root_failed = rank, root_failed

    def Get_size(self):
        return 2

    def Get_rank(self):
        return self.rank

    def bcast(self, value, root=0):
        return value if self.rank == root else self.root_failed


@pytest.mark.parametrize("root_failed", [False, True])
def test_other_processes_follow_process_0(tmp_path: Path, monkeypatch,
                                          root_failed) -> None:
    """A process other than 0 neither checks nor writes; it raises exactly
    when process 0 did, so no process is left waiting in a collective."""
    import sys
    import types
    mpi = types.SimpleNamespace(Is_initialized=lambda: True,
                                COMM_WORLD=_FakeComm(1, root_failed))
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)
    write_run_manifest(tmp_path, ExperimentConfig())
    d = _driver_at(tmp_path, expected_resolved_physics=_full_record(
        turbulence=_turb(l_mix_max=50.0)))                 # would refuse on 0
    if root_failed:
        with pytest.raises(RuntimeError, match="refused on process 0"):
            d._record_resolved_physics(turbulence=_turb())
    else:
        d._record_resolved_physics(turbulence=_turb())
    assert "resolved_physics" not in read_run_manifest(tmp_path)["config"]


def test_config_from_record_is_strict() -> None:
    rec = resolved_physics_record({"t": _turb(l_mix_max=137.0)})["t"]
    assert physics_config_from_record(rec) == _turb(l_mix_max=137.0)
    del rec["louis"]["l_mix_max"]
    with pytest.raises(ValueError, match="missing \\['l_mix_max'\\]"):
        physics_config_from_record(rec)
    with pytest.raises(ValueError, match="non-legoesm"):
        physics_config_from_record({"__type__": "os:PathLike"})
    with pytest.raises(ValueError, match="not a legoesm config type"):
        physics_config_from_record({"__type__": "legoesm.driver.restart:os.PathLike"})
    with pytest.raises(ValueError, match="summary"):
        physics_config_from_record({"__array_summary__": {}})


# --- lanes --------------------------------------------------------------------

def _cube(tmp: Path, **kw) -> "ModelDriver":
    from legoesm.driver.model_driver import ModelDriver
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(energy_fixer=False, dt=1800.0),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=3601.0 / 86400.0, dataset="analytical", radiation="gray",
        microphysics="none", convection="none", turbulence="louis",
        precision="fp64", **kw)
    d = ModelDriver(cfg, output_dir=str(tmp))
    d.setup()
    assert d.run() == "COMPLETED"
    return d


def test_pipeline_lane_records_at_run_start_including_the_override(
        tmp_path: Path) -> None:
    d = _cube(tmp_path, turbulence_override=_turb(l_mix_max=137.0))
    m = read_run_manifest(tmp_path)
    rec = m["config"]["resolved_physics"]
    assert set(rec) == set(RESOLVED_PHYSICS_KEYS)
    assert rec["turbulence"] == resolved_physics_record(
        {"t": d.physics.turbulence_config})["t"]
    # The override is dropped from resolved_config but kept here.
    assert m["config"]["resolved_config"]["turbulence_override"] is None
    assert json.dumps(rec["turbulence"]).count("137.0") >= 1
    assert rec["radiation"]["scheme"]["__type__"].endswith(
        ":GrayRadiationConfig")
    assert rec["radiation"] == resolved_physics_record(
        {"r": d.physics.radiation_config})["r"]
    assert rec["cloud"] is None and rec["land"] is None   # none configured
    for k, t in (("solar", "SolarConfig"), ("ozone", "OzoneConfig"),
                 ("aerosol", "AerosolConfig"), ("ghg", "GHGConfig")):
        assert rec[k]["__type__"].endswith(":" + t)
    assert physics_config_from_record(rec["turbulence_override"]) == _turb(
        l_mix_max=137.0)


def _spectral(tmp: Path, cfg_extra=None, **kw):
    from legoesm.driver.model_driver import ModelDriver
    fields = dict(
        grid=GridConfig(grid_type="gaussian", resolution=8, nlev=8),
        dycore=DycoreConfig(energy_fixer=False, model_type="hydrostatic",
                            discretization="spectral", dt=600.0),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=DAYS_2_STEPS, dataset="analytical", radiation="gray",
        convection="none", turbulence="louis", precision="fp64",
        distributed=False)
    fields.update(cfg_extra or {})
    return ModelDriver(ExperimentConfig(**fields), output_dir=str(tmp), **kw)


def _reproduce(manifest_dir: Path, out: Path) -> None:
    from legoesm.cli import cmd_reproduce
    cmd_reproduce(argparse.Namespace(
        manifest=str(manifest_dir / RUN_MANIFEST_FILENAME), check=True,
        output=str(out), allow_version_drift=False))


def test_spectral_lane_records_and_reproduce_closes(tmp_path: Path,
                                                    monkeypatch,
                                                    caplog) -> None:
    import legoesm.driver.model_driver as md
    ref = tmp_path / "ref"
    d = _spectral(ref)
    d.setup()
    # The setup-time pipeline is NOT this lane's physics: nothing recorded yet.
    assert "resolved_physics" not in read_run_manifest(ref)["config"]
    assert d.run() == "COMPLETED"
    rec = read_run_manifest(ref)["config"]["resolved_physics"]
    assert rec["turbulence"]["__type__"].endswith(":TurbulenceConfig")
    assert rec["radiation"]["__type__"].endswith(":RadiationConfig")
    assert rec["solar"]["__type__"].endswith(":SolarConfig")

    caplog.set_level(logging.INFO)
    _reproduce(ref, tmp_path / "rerun")
    assert "MATCH" in caplog.text
    assert (read_run_manifest(tmp_path / "rerun")["config"]["resolved_physics"]
            == rec)

    # Resolver drift: the rerun's lane resolves different turbulence.
    real = md.turbulence_config_for
    monkeypatch.setattr(md, "turbulence_config_for",
                        lambda cfg: real(cfg)._replace(
                            louis=real(cfg).louis._replace(l_mix_max=50.0)))
    caplog.clear()
    with pytest.raises(SystemExit) as exc:
        _reproduce(ref, tmp_path / "drifted")
    assert exc.value.code == 1
    assert "Resolved physics differs from the reference run" in caplog.text
    assert "turbulence.louis.l_mix_max" in caplog.text


def test_reproduce_replays_full_overrides(tmp_path: Path, caplog) -> None:
    """Both overrides are dropped from (turbulence) or mangled in (GWD, a
    positional list) resolved_config; replay takes them from the record."""
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    ref = tmp_path / "ref"
    d = _spectral(ref, cfg_extra=dict(
        turbulence_override=_turb(l_mix_max=137.0),
        gravity_wave_drag="rayleigh",
        gravity_wave_drag_override=GravityWaveDragConfig(scheme="rayleigh")))
    d.setup()
    assert d.run() == "COMPLETED"
    caplog.set_level(logging.INFO)
    _reproduce(ref, tmp_path / "rerun")
    assert "MATCH" in caplog.text


def test_mpas_lane_records_and_closure_refuses(tmp_path: Path) -> None:
    from legoesm.driver.model_driver import ModelDriver

    def cfg():
        return ExperimentConfig(
            grid=GridConfig(grid_type="mpas", resolution=3, nlev=8,
                            vertical_coord="hybrid"),
            dycore=DycoreConfig(discretization="mpas", dt=300.0),
            output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
            days=601.0 / 86400.0, dataset="analytical", radiation="gray",
            convection="none", turbulence="louis", precision="fp64",
            distributed=False)

    d = ModelDriver(cfg(), output_dir=str(tmp_path / "a"))
    d.setup()
    assert d.run() == "COMPLETED"
    rec = read_run_manifest(tmp_path / "a")["config"]["resolved_physics"]
    assert rec["radiation"]["__type__"].endswith(":RadiationConfig")
    assert rec["ghg"]["__type__"].endswith(":GHGConfig")
    bad = copy.deepcopy(rec)
    bad["turbulence"]["louis"]["l_mix_max"] = -1.0
    d2 = ModelDriver(cfg(), output_dir=str(tmp_path / "b"),
                     expected_resolved_physics=bad)
    d2.setup()
    with pytest.raises(RuntimeError, match="turbulence.louis.l_mix_max"):
        d2.run()


# --- strict replay in reproduce ----------------------------------------------

@pytest.mark.parametrize("physics", [None, {}])
def test_reproduce_refuses_a_manifest_without_a_physics_record(
        tmp_path: Path, caplog, physics) -> None:
    path = write_run_manifest(tmp_path, ExperimentConfig())
    if physics is not None:
        m = json.loads(path.read_text())
        m["config"]["resolved_physics"] = physics
        path.write_text(json.dumps(m))
    caplog.set_level(logging.INFO)
    with pytest.raises(SystemExit) as exc:
        _reproduce(tmp_path, tmp_path / "out")
    assert exc.value.code == 2
    assert "no resolved-physics record" in caplog.text


def test_reproduce_refuses_a_field_filled_from_todays_default(
        tmp_path: Path, caplog) -> None:
    """A field the record lacks passes the schema-growth hash check but is
    refused by strict replay instead of being filled from today's default."""
    path = write_run_manifest(tmp_path, ExperimentConfig())
    m = json.loads(path.read_text())
    del m["config"]["resolved_config"]["seed"]
    m["config"]["config_hash"] = hashlib.sha256(json.dumps(
        m["config"]["resolved_config"], sort_keys=True).encode()).hexdigest()
    m["config"]["resolved_physics"] = {k: None for k in RESOLVED_PHYSICS_KEYS}
    path.write_text(json.dumps(m))
    caplog.set_level(logging.INFO)
    with pytest.raises(SystemExit) as exc:
        _reproduce(tmp_path, tmp_path / "out")
    assert exc.value.code == 2
    assert "does not rebuild the recorded configuration exactly" in caplog.text
    assert "seed: recorded='<absent>'" in caplog.text


@pytest.mark.parametrize("cfg,own", [
    (dict(dycore=DycoreConfig(discretization="fv3_duo",
                              column_lane=True)), True),
    (dict(dycore=DycoreConfig(discretization="fv3_duo")), True),
    (dict(grid=GridConfig(grid_type="latlon"), enable_latlon_spmd=True,
          turbulence="louis"), True),
    (dict(grid=GridConfig(grid_type="latlon"), enable_latlon_spmd=True,
          held_suarez_forcing=True), True),
    (dict(grid=GridConfig(grid_type="latlon")), False),
])
def test_lanes_that_build_their_own_physics(cfg, own) -> None:
    """The setup pipeline is recorded only where it is what the kernels use;
    these lanes (run dispatch) record their own build instead."""
    from legoesm.driver.model_driver import ModelDriver
    assert ModelDriver(ExperimentConfig(**cfg))._lane_builds_own_physics() is own


def _spy_pipelines(monkeypatch, module):
    built = []
    real = module.build_physics_pipeline

    def spy(*a, **k):
        built.append(real(*a, **k))
        return built[-1]
    monkeypatch.setattr(module, "build_physics_pipeline", spy)
    return built


def _pipeline_record(p):
    return resolved_physics_record(dict(convection=p.convection_config,
                                        microphysics=p.micro_config,
                                        turbulence=p.turbulence_config,
                                        gravity_wave_drag=p.gwd_config,
                                        radiation=p.radiation_config))


def _sharded_cfg(grid, n_devices, **dyc):
    return ExperimentConfig(
        grid=grid,
        dycore=DycoreConfig(model_type="hydrostatic", dt=100.0, fix_mass=True,
                            **dyc),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=201.0 / 86400.0, dataset="analytical", radiation="gray",
        turbulence="tke", convection="none", microphysics="none",
        gravity_wave_drag="none", cloud_scheme="none", fix_moisture=True,
        distributed=False, n_devices=n_devices,
        enable_latlon_spmd=grid.grid_type == "latlon")


def _assert_records_lane_pipeline(tmp_path, cfg, built) -> None:
    from legoesm.driver.model_driver import ModelDriver
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    assert d._lane_builds_own_physics()
    assert d.run(compiled=True) == "COMPLETED"
    lane = built[-1]
    assert lane is not d.physics          # the lane's own build, not setup's
    rec = read_run_manifest(tmp_path)["config"]["resolved_physics"]
    want = _pipeline_record(lane)
    assert {k: rec[k] for k in want} == want
    assert rec["solar"]["__type__"].endswith(":SolarConfig")


def test_lat_band_lane_records_its_band_pipeline(tmp_path: Path,
                                                 monkeypatch) -> None:
    import jax
    import legoesm.driver.model_driver as md
    if len(jax.devices()) < 2:
        pytest.skip("needs --xla_force_host_platform_device_count=2")
    built = _spy_pipelines(monkeypatch, md)
    _assert_records_lane_pipeline(tmp_path, _sharded_cfg(
        GridConfig(grid_type="latlon", resolution=8, nlev=4), 2,
        discretization="latlon_cgrid"), built)


def test_tiled_cube_lane_records_its_tile_pipeline(tmp_path: Path,
                                                   monkeypatch) -> None:
    import jax
    import legoesm.driver.physics_pipeline as pp   # tile builder imports it locally
    if len(jax.devices()) < 24:
        pytest.skip("needs --xla_force_host_platform_device_count=24")
    built = _spy_pipelines(monkeypatch, pp)
    _assert_records_lane_pipeline(tmp_path, _sharded_cfg(
        GridConfig(grid_type="cubed_sphere", resolution=8, nlev=4), 24,
        discretization="cdgrid", energy_fixer=False, hyperdiff_scale=0.0,
        div_damp_scale=0.0, a_h_scale=0.0), built)


def test_idealized_lanes_record_the_kessler_config_they_run(
        tmp_path: Path) -> None:
    """The FV3-duo and simple tiled-cube lanes run Kessler with the run's
    config as ``_spectral_micro_config`` resolves it, and nothing else."""
    from legoesm.driver.model_driver import _spectral_micro_config
    write_run_manifest(tmp_path, ExperimentConfig())
    cfg = ExperimentConfig(microphysics="kessler",
                           hard_saturation_adjustment=True)
    d = _driver_at(tmp_path)
    d.config = cfg
    d._record_idealized_kessler_physics("FV3-duo Kessler")
    rec = read_run_manifest(tmp_path)["config"]["resolved_physics"]
    want = _spectral_micro_config(cfg, lane="FV3-duo Kessler").kessler
    assert rec["microphysics"] == resolved_physics_record({"m": want})["m"]
    assert rec == _full_record(microphysics=want)

def test_a_lane_that_records_nothing_fails_the_run(tmp_path: Path,
                                                   monkeypatch) -> None:
    """Predicate and dispatch disagreeing (the lane is told to record its own
    build but does not) is caught when the run completes."""
    from legoesm.driver.model_driver import ModelDriver
    monkeypatch.setattr(ModelDriver, "_lane_builds_own_physics",
                        lambda self: True)
    with pytest.raises(RuntimeError, match="without recording its resolved"):
        _cube(tmp_path)


def test_reproduce_refuses_an_untagged_override(tmp_path: Path,
                                                caplog) -> None:
    path = write_run_manifest(tmp_path, ExperimentConfig())
    m = json.loads(path.read_text())
    m["config"]["resolved_physics"] = {k: None for k in RESOLVED_PHYSICS_KEYS}
    m["config"]["resolved_physics"]["turbulence_override"] = {"scheme": "louis"}
    path.write_text(json.dumps(m))
    caplog.set_level(logging.INFO)
    with pytest.raises(SystemExit) as exc:
        _reproduce(tmp_path, tmp_path / "out")
    assert exc.value.code == 2
    assert "not a tagged config record" in caplog.text


def test_held_suarez_with_a_scheme_records_the_band_pipeline(
        tmp_path: Path, monkeypatch) -> None:
    """Held-Suarez plus a configured scheme runs the operator-split lat-band
    lane, so its band pipeline (here Louis turbulence) is recorded."""
    import jax
    import legoesm.driver.model_driver as md
    if len(jax.devices()) < 2:
        pytest.skip("needs --xla_force_host_platform_device_count=2")
    built = _spy_pipelines(monkeypatch, md)
    cfg = _sharded_cfg(GridConfig(grid_type="latlon", resolution=8, nlev=4), 2,
                       discretization="latlon_cgrid")._replace(
        held_suarez_forcing=True, radiation="none", turbulence="louis")
    _assert_records_lane_pipeline(tmp_path, cfg, built)
    rec = read_run_manifest(tmp_path)["config"]["resolved_physics"]
    assert rec["turbulence"] is not None


def test_pure_held_suarez_lat_band_lane_records_no_schemes(
        tmp_path: Path) -> None:
    """Pure Held-Suarez (every scheme 'none') runs the stateless lat-band
    lane, which runs no scheme kernel, so none is recorded."""
    import jax
    from legoesm.driver.model_driver import ModelDriver
    if len(jax.devices()) < 2:
        pytest.skip("needs --xla_force_host_platform_device_count=2")
    cfg = _sharded_cfg(GridConfig(grid_type="latlon", resolution=8, nlev=4), 2,
                       discretization="latlon_cgrid")._replace(
        held_suarez_forcing=True, radiation="none", turbulence="none")
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    assert not d._operator_split_spmd_active()
    assert d.run(compiled=True) == "COMPLETED"
    rec = read_run_manifest(tmp_path)["config"]["resolved_physics"]
    assert rec == _full_record()


def test_dry_spectral_path_records_its_default_gray_radiation(
        tmp_path: Path) -> None:
    """With no scheme configured the spectral lane runs a default gray
    radiation + Rayleigh friction, not the setup pipeline."""
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    d = _spectral(tmp_path, cfg_extra=dict(turbulence="none"))
    d.setup()
    assert d.run() == "COMPLETED"
    rec = read_run_manifest(tmp_path)["config"]["resolved_physics"]
    assert rec["radiation"] == resolved_physics_record(
        {"r": {"scheme": GrayRadiationConfig(), "orbit": None}})["r"]
    assert rec["turbulence"] is None and rec["convection"] is None


def test_pipeline_records_the_cloud_config_its_radiation_builds(
        tmp_path: Path) -> None:
    """With a cloud scheme on, the record holds the CloudConfig that the
    radiation path builds (knobs + library defaults), and the orbit."""
    from legoesm.driver.model_driver import ModelDriver
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(energy_fixer=False, dt=1800.0),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=1, dataset="analytical", radiation="rrtmgp",
        cloud_scheme="sundqvist", cloud_rh_crit=0.85, orbital_insolation=True,
        microphysics="none", convection="none", turbulence="louis",
        precision="fp64")
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()                          # pipeline built; no time stepping
    d._physics_recorded = False
    d._record_pipeline_physics(d.physics)
    rec = read_run_manifest(tmp_path)["config"]["resolved_physics"]
    want = d.physics.resolved_cloud_config("sundqvist", False)
    assert rec["cloud"] == resolved_physics_record({"c": want})["c"]
    assert want.rh_crit == 0.85
    assert rec["radiation"]["scheme"]["__type__"].endswith(":RRTMGPConfig")
    assert rec["radiation"]["orbit"] is not None
    # Convective cloud reaches the optics only where the lane passes the
    # lagged convective precip: compiled / operator-split yes, per-step no.
    d.physics._cloud_convective = True
    for threaded in (True, False):
        out = tmp_path / f"conv{threaded}"
        write_run_manifest(out, cfg)
        d._output_dir = out
        d._record_pipeline_physics(d.physics, conv_precip_threaded=threaded)
        got = read_run_manifest(out)["config"]["resolved_physics"]["cloud"]
        assert got == resolved_physics_record({"c": d.physics.resolved_cloud_config(
            "sundqvist", threaded)})["c"]
    assert d.physics.resolved_cloud_config("sundqvist", True) != \
        d.physics.resolved_cloud_config("sundqvist", False)


@pytest.mark.parametrize("compiled", [True, False])
def test_run_tells_the_record_whether_conv_precip_reaches_radiation(
        tmp_path: Path, monkeypatch, compiled) -> None:
    from legoesm.driver.model_driver import ModelDriver
    seen = []
    real = ModelDriver._record_pipeline_physics

    def spy(self, pipeline, conv_precip_threaded=True, **kw):
        seen.append(conv_precip_threaded)
        return real(self, pipeline, conv_precip_threaded, **kw)
    monkeypatch.setattr(ModelDriver, "_record_pipeline_physics", spy)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(energy_fixer=False, dt=1800.0),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=3601.0 / 86400.0, dataset="analytical", radiation="gray",
        microphysics="none", convection="none", turbulence="louis",
        precision="fp64")
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    assert d.run(compiled=compiled) == "COMPLETED"
    assert seen == [compiled]


@pytest.mark.parametrize("compiled", [True, False])
def test_reproduce_reruns_the_rollout_and_reapplies_post_setup_schemes(
        tmp_path: Path, monkeypatch, caplog, compiled) -> None:
    """A driver that replaces a pipeline scheme config after setup (as
    run_amip does for --params / trained configs) is reproduced: the rerun
    uses the recorded rollout and the recorded scheme configs, and the
    closure check passes."""
    from legoesm.driver.model_driver import ModelDriver
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(energy_fixer=False, dt=1800.0),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=3601.0 / 86400.0, dataset="analytical", radiation="gray",
        microphysics="none", convection="none", turbulence="louis",
        precision="fp64")
    ref = tmp_path / "ref"
    d = ModelDriver(cfg, output_dir=str(ref))
    d.setup()
    tuned = d.physics.turbulence_config._replace(l_mix_max=137.0)
    d.physics.turbulence_config = tuned                 # post-setup change
    assert d.run(compiled=compiled) == "COMPLETED"
    rec = read_run_manifest(ref)["config"]["resolved_physics"]
    assert rec["rollout"] == ("compiled" if compiled else "per_step")
    assert rec["turbulence"] == resolved_physics_record({"t": tuned})["t"]

    seen = []
    real_run = ModelDriver.run

    def spy(self, *a, **k):
        seen.append(k.get("compiled", True))
        return real_run(self, *a, **k)
    monkeypatch.setattr(ModelDriver, "run", spy)
    caplog.set_level(logging.INFO)
    _reproduce(ref, tmp_path / "rerun")
    assert "MATCH" in caplog.text
    assert seen == [compiled]


def test_reproduce_refuses_an_unreplayable_pipeline_scheme(
        tmp_path: Path, monkeypatch, caplog) -> None:
    """A recorded pipeline scheme that cannot be rebuilt (e.g. an array leaf
    stored only as a digest) is a clean refusal, not a traceback."""
    from legoesm.driver.model_driver import ModelDriver
    monkeypatch.setattr(ModelDriver, "setup", lambda self: None)
    path = write_run_manifest(tmp_path, ExperimentConfig())
    m = json.loads(path.read_text())
    m["config"]["resolved_physics"] = {k: None for k in RESOLVED_PHYSICS_KEYS}
    m["config"]["resolved_physics"].update(
        rollout="compiled", turbulence={"__array_summary__": {}})
    path.write_text(json.dumps(m))
    caplog.set_level(logging.INFO)
    with pytest.raises(SystemExit) as exc:
        _reproduce(tmp_path, tmp_path / "out")
    assert exc.value.code == 2
    assert "cannot reapply the recorded pipeline schemes" in caplog.text
