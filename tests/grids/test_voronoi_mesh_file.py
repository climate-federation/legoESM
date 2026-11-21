"""Saved MPAS meshes: round trip, hash refusals and the driver's file path.

A regenerated SCVT mesh is not bitwise portable (level 4: 1,973 of 2,562 cells
at other indices on Snellius than on a Mac), so runs can pin a saved mesh.
These tests hold the file format to "nothing downstream can tell the
difference" and the driver to "no unpinned or regenerated mesh sneaks in".
"""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy, get_policy, set_policy)
from legoesm.grids.voronoi import VoronoiMesh, create_voronoi_mesh  # noqa: E402
from legoesm.grids.voronoi_mesh_file import (  # noqa: E402
    MeshFileError,
    load_voronoi_mesh,
    load_voronoi_mesh_for_run,
    read_voronoi_mesh_metadata,
    save_voronoi_mesh,
    voronoi_mesh_sha256,
)

LEVEL = 2
LLOYD = 50


@pytest.fixture(autouse=True)
def _fresh_builds_and_policy(monkeypatch):
    """No disk cache (a fresh build every time) and the policy restored."""
    monkeypatch.setenv("LEGOESM_MESH_CACHE_DISABLE", "1")
    before = get_policy()
    yield
    set_policy(before)


def _assert_same_mesh(a: VoronoiMesh, b: VoronoiMesh) -> None:
    """Field by field: same Python/JAX type, same dtype, same bits."""
    assert type(a) is type(b)
    assert a._fields == b._fields
    for name in a._fields:
        x, y = getattr(a, name), getattr(b, name)
        assert type(x) is type(y), name
        if x is None:
            continue
        if isinstance(x, (int, float)):
            assert x == y, name
            continue
        assert x.dtype == y.dtype, name
        assert x.shape == y.shape, name
        xa, ya = np.asarray(x), np.asarray(y)
        assert xa.tobytes() == ya.tobytes(), name


def _bare(mesh: VoronoiMesh) -> VoronoiMesh:
    """The mesh without the surface fields _create_topography attaches."""
    return mesh._replace(subgrid_topo_stddev=None, land_frac=None)


def _rewrite(src, dst, mutate):
    """Copy an npz entry by entry, letting ``mutate`` change the dict."""
    with np.load(src, allow_pickle=False) as d:
        entries = {k: np.array(d[k]) for k in d.files}
    mutate(entries)
    with open(dst, "wb") as f:
        np.savez(f, **entries)


@pytest.mark.parametrize("policy", ["fp32", "fp64"])
def test_round_trip_is_bitwise_identical(tmp_path, policy):
    set_policy(getattr(PrecisionPolicy, policy)())
    mesh = create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD)
    path = tmp_path / "m.npz"
    digest = save_voronoi_mesh(mesh, path, resolution=LEVEL,
                               lloyd_iterations=LLOYD)
    assert digest == voronoi_mesh_sha256(mesh)
    back = load_voronoi_mesh(path, expected_sha256=digest)
    _assert_same_mesh(mesh, back)
    assert voronoi_mesh_sha256(back) == digest


def test_float64_file_under_float32_policy_matches_a_fresh_float32_build(
        tmp_path):
    set_policy(PrecisionPolicy.fp64())
    path = tmp_path / "m64.npz"
    digest = save_voronoi_mesh(
        create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD), path,
        resolution=LEVEL, lloyd_iterations=LLOYD)
    set_policy(PrecisionPolicy.fp32())
    fresh32 = create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD)
    _assert_same_mesh(fresh32, load_voronoi_mesh(path, expected_sha256=digest))


def test_float32_file_refused_under_float64_policy(tmp_path):
    set_policy(PrecisionPolicy.fp32())
    path = tmp_path / "m32.npz"
    save_voronoi_mesh(create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD),
                      path, resolution=LEVEL, lloyd_iterations=LLOYD)
    set_policy(PrecisionPolicy.fp64())
    with pytest.raises(MeshFileError, match="precision cannot"):
        load_voronoi_mesh(path)


def test_optional_fields_round_trip(tmp_path):
    mesh = create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD)
    mesh = mesh._replace(land_frac=mesh.areaCell * 0 + 0.25)
    path = tmp_path / "m.npz"
    save_voronoi_mesh(mesh, path, resolution=LEVEL, lloyd_iterations=LLOYD)
    _assert_same_mesh(mesh, load_voronoi_mesh(path))


def test_metadata_recorded(tmp_path):
    mesh = create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD)
    path = tmp_path / "m.npz"
    digest = save_voronoi_mesh(mesh, path, resolution=LEVEL,
                               lloyd_iterations=LLOYD)
    meta = read_voronoi_mesh_metadata(path)
    assert meta["sha256"] == digest
    assert meta["resolution"] == LEVEL
    assert meta["lloyd_iterations"] == LLOYD
    for key in ("generator_commit", "platform", "versions", "scalar_types"):
        assert key in meta
    assert {"numpy", "scipy"} <= set(meta["versions"])


def test_tampered_array_is_refused(tmp_path):
    path = tmp_path / "m.npz"
    save_voronoi_mesh(create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD),
                      path, resolution=LEVEL, lloyd_iterations=LLOYD)
    bad = tmp_path / "bad.npz"

    def swap_two_cells(e):
        lat = e["latCell"].copy()
        lat[[0, 1]] = lat[[1, 0]]
        e["latCell"] = lat

    _rewrite(path, bad, swap_two_cells)
    with pytest.raises(MeshFileError, match="does not match the stored hash"):
        load_voronoi_mesh(bad)


def test_tampered_scalar_or_dtype_is_refused(tmp_path):
    path = tmp_path / "m.npz"
    save_voronoi_mesh(create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD),
                      path, resolution=LEVEL, lloyd_iterations=LLOYD)
    for i, mutate in enumerate((
            lambda e: e.__setitem__("radius", e["radius"] * 1.001),
            lambda e: e.__setitem__(
                "cellsOnEdge", e["cellsOnEdge"].astype(np.int64)))):
        bad = tmp_path / f"bad{i}.npz"
        _rewrite(path, bad, mutate)
        with pytest.raises(MeshFileError):
            load_voronoi_mesh(bad)


def test_missing_or_extra_field_is_refused(tmp_path):
    path = tmp_path / "m.npz"
    save_voronoi_mesh(create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD),
                      path, resolution=LEVEL, lloyd_iterations=LLOYD)
    for i, mutate in enumerate((lambda e: e.pop("dcEdge"),
                                lambda e: e.__setitem__("bogus", np.zeros(3)))):
        bad = tmp_path / f"bad{i}.npz"
        _rewrite(path, bad, mutate)
        with pytest.raises(MeshFileError, match="do not match VoronoiMesh"):
            load_voronoi_mesh(bad)


def test_wrong_expected_hash_is_refused(tmp_path):
    path = tmp_path / "m.npz"
    digest = save_voronoi_mesh(
        create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD), path,
        resolution=LEVEL, lloyd_iterations=LLOYD)
    wrong = ("0" if digest[0] != "0" else "1") + digest[1:]
    with pytest.raises(MeshFileError, match="differs from the expected"):
        load_voronoi_mesh(path, expected_sha256=wrong)
    load_voronoi_mesh(path, expected_sha256=digest.upper())   # case-insensitive


def test_hash_ignores_zip_layout(tmp_path):
    """Compression and entry order do not change the content hash."""
    path = tmp_path / "m.npz"
    digest = save_voronoi_mesh(
        create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD), path,
        resolution=LEVEL, lloyd_iterations=LLOYD)
    with np.load(path, allow_pickle=False) as d:
        entries = {k: np.array(d[k]) for k in reversed(d.files)}
    other = tmp_path / "compressed.npz"
    with open(other, "wb") as f:
        np.savez_compressed(f, **entries)
    assert path.read_bytes() != other.read_bytes()
    load_voronoi_mesh(other, expected_sha256=digest)


def test_run_loader_refusals(tmp_path):
    path = tmp_path / "m.npz"
    digest = save_voronoi_mesh(
        create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD), path,
        resolution=LEVEL, lloyd_iterations=LLOYD)
    with pytest.raises(MeshFileError, match="without its SHA-256"):
        load_voronoi_mesh_for_run(path, resolution=LEVEL, expected_sha256="")
    with pytest.raises(MeshFileError, match="resolution"):
        load_voronoi_mesh_for_run(path, resolution=LEVEL + 1,
                                  expected_sha256=digest)
    with pytest.raises(MeshFileError, match="lloyd_iterations"):
        load_voronoi_mesh_for_run(path, resolution=LEVEL,
                                  expected_sha256=digest, lloyd_iterations=5)
    with pytest.raises(MeshFileError, match="does not exist"):
        load_voronoi_mesh_for_run(tmp_path / "nope.npz", resolution=LEVEL,
                                  expected_sha256=digest)


def test_cli_writes_a_loadable_float64_file(tmp_path, capsys):
    import importlib.util
    from pathlib import Path
    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "grids" / "save_mpas_mesh.py")
    spec = importlib.util.spec_from_file_location("save_mpas_mesh", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "cli.npz"
    assert mod.main([str(LEVEL), str(out), "--lloyd-iterations",
                     str(LLOYD)]) == 0
    printed = capsys.readouterr().out
    digest = [ln.split()[-1] for ln in printed.splitlines()
              if ln.startswith("mpas_mesh_sha256:")][0]
    meta = read_voronoi_mesh_metadata(out)
    assert meta["sha256"] == digest
    assert meta["dtypes"]["latCell"] == "<f8"
    with pytest.raises(SystemExit):              # refuses to overwrite
        mod.main([str(LEVEL), str(out)])


# ----------------------------------------------------------------------------
# Driver and run script
# ----------------------------------------------------------------------------

def _driver_cfg(grid_type="mpas", **grid_kw):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type=grid_type, resolution=LEVEL, nlev=3,
                        vertical_coord="sigma", **grid_kw),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="mpas"),
        topography="flat",
        radiation="gray",
        days=1,
    )


@pytest.fixture
def mesh_file(tmp_path):
    set_policy(PrecisionPolicy.fp64())
    path = tmp_path / "mesh.npz"
    digest = save_voronoi_mesh(
        create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD), path,
        resolution=LEVEL, lloyd_iterations=LLOYD)
    return path, digest


@pytest.mark.parametrize("grid_type", ["mpas", "voronoi"])
def test_driver_loads_the_file_and_never_calls_create_grid(
        tmp_path, mesh_file, monkeypatch, grid_type):
    """Also under the legacy alias run_coupled.py passes (--grid voronoi)."""
    from legoesm.driver.model_driver import ModelDriver
    import legoesm.grids.factory as factory
    import legoesm.grids.voronoi as voronoi
    path, digest = mesh_file
    cfg = _driver_cfg(grid_type, mpas_mesh_file=str(path),
                      mpas_mesh_sha256=digest)
    driver = ModelDriver(cfg, output_dir=str(tmp_path / "run"))

    def boom(*a, **k):
        raise AssertionError("mesh regenerated despite mpas_mesh_file")

    monkeypatch.setattr(factory, "create_grid", boom)
    monkeypatch.setattr(voronoi, "create_voronoi_mesh", boom)
    driver.static_topography_phis()           # validate, bootstrap, grid, topo
    _assert_same_mesh(_bare(driver.grid), load_voronoi_mesh(path))


def test_driver_flag_off_is_unchanged(tmp_path):
    from legoesm.driver.model_driver import ModelDriver
    cfg = _driver_cfg()
    driver = ModelDriver(cfg, output_dir=str(tmp_path / "run"))
    driver.static_topography_phis()
    _assert_same_mesh(_bare(driver.grid),
                      create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD))


def test_driver_refuses_mismatches(tmp_path, mesh_file):
    from legoesm.driver.model_driver import ModelDriver
    path, digest = mesh_file
    wrong = ("0" if digest[0] != "0" else "1") + digest[1:]
    # no hash: validate_strict and the driver both refuse
    with pytest.raises(ValueError, match="mpas_mesh_sha256"):
        _driver_cfg(mpas_mesh_file=str(path)).validate_strict()
    with pytest.raises(MeshFileError, match="without its SHA-256"):
        ModelDriver._load_mpas_mesh_file(
            _driver_cfg(mpas_mesh_file=str(path)).grid, lloyd_iterations=LLOYD)
    with pytest.raises(MeshFileError, match="differs from the expected"):
        ModelDriver(_driver_cfg(mpas_mesh_file=str(path),
                                mpas_mesh_sha256=wrong),
                    output_dir=str(tmp_path / "a")).static_topography_phis()
    from legoesm.driver.config import GridConfig
    with pytest.raises(MeshFileError, match="resolution"):
        ModelDriver._load_mpas_mesh_file(
            GridConfig(grid_type="mpas", resolution=LEVEL + 1,
                       mpas_mesh_file=str(path), mpas_mesh_sha256=digest),
            lloyd_iterations=LLOYD)
    with pytest.raises(ValueError, match="only valid with grid_type"):
        from legoesm.driver.config import ExperimentConfig
        ExperimentConfig(grid=GridConfig(
            grid_type="latlon", mpas_mesh_file=str(path),
            mpas_mesh_sha256=digest)).validate_strict()


def test_run_amip_flags_and_deck_keys_reach_grid_config(tmp_path):
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run.run_amip import build_arg_parser, build_config_from_args
    parser = build_arg_parser()
    cfg = build_config_from_args(parser.parse_args([]))
    assert cfg.grid.mpas_mesh_file == "" and cfg.grid.mpas_mesh_sha256 == ""

    cfg = build_config_from_args(build_arg_parser().parse_args(
        ["--grid-type", "mpas", "--mpas-mesh-file", "m.npz",
         "--mpas-mesh-sha256", "ab" * 32]))
    assert cfg.grid.mpas_mesh_file == "m.npz"
    assert cfg.grid.mpas_mesh_sha256 == "ab" * 32

    deck = tmp_path / "deck.yaml"
    deck.write_text("grid_type: mpas\nmpas_mesh_file: deck.npz\n"
                    f"mpas_mesh_sha256: {'cd' * 32}\n")
    parser = build_arg_parser()
    keys = load_yaml_config(str(deck), parser)
    parser.set_defaults(**keys)
    cfg = build_config_from_args(parser.parse_args([]))
    assert cfg.grid.mpas_mesh_file == "deck.npz"
    assert cfg.grid.mpas_mesh_sha256 == "cd" * 32


def test_config_json_round_trip_keeps_the_pin():
    from legoesm.driver.config import (
        experiment_config_from_dict, experiment_config_to_dict)
    cfg = _driver_cfg(mpas_mesh_file="m.npz", mpas_mesh_sha256="ef" * 32)
    back = experiment_config_from_dict(
        json.loads(json.dumps(experiment_config_to_dict(cfg))))
    assert back.grid.mpas_mesh_file == "m.npz"
    assert back.grid.mpas_mesh_sha256 == "ef" * 32


def test_unhashed_metadata_cannot_change_scalars_and_int32_overflow_is_refused(tmp_path):
    from legoesm.grids.voronoi_mesh_file import mesh_arrays_sha256
    mesh = create_voronoi_mesh(LEVEL, lloyd_iterations=LLOYD)
    path = tmp_path / "m.npz"
    digest = save_voronoi_mesh(mesh, path, resolution=LEVEL, lloyd_iterations=LLOYD)

    def retype(e):
        meta = json.loads(str(e["_metadata"][()]))
        meta["scalar_types"] = {k: "float" for k in meta.get("scalar_types", {})}
        e["_metadata"] = np.asarray(json.dumps(meta))
    meta_bad = tmp_path / "meta.npz"
    _rewrite(path, meta_bad, retype)
    back = load_voronoi_mesh(meta_bad, expected_sha256=digest)
    assert isinstance(back.nCells, int) and back.nCells == mesh.nCells

    def overflow(e):
        c = e["cellsOnEdge"].astype(np.int64)
        c[0, 0] = 2 ** 32 + int(c[0, 0])          # wraps to the same int32
        e["cellsOnEdge"] = c
        e["_sha256"] = np.asarray(mesh_arrays_sha256(
            {k: v for k, v in e.items() if not k.startswith("_")}))
    big = tmp_path / "big.npz"
    _rewrite(path, big, overflow)
    with pytest.raises(MeshFileError, match="outside int32"):
        load_voronoi_mesh(big)
