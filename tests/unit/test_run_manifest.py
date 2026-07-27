"""Run-manifest provenance (Stage A1 reproducibility spine).

The manifest is the legoESM half of the unified provenance document: written
once at run start, it must capture enough to reconstruct the run (resolved
config + environment) and embed the legoESM-ocean-runners tag block so an
``experiment.tag`` is a valid subset.  These tests pin the schema and prove the
resolved config round-trips back into an ExperimentConfig.
"""

from __future__ import annotations

import json
from pathlib import Path

from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    experiment_config_from_dict,
)
from legoesm.driver.restart import (
    RUN_MANIFEST_FILENAME,
    RUN_MANIFEST_SCHEMA_VERSION,
    build_run_manifest,
    read_run_manifest,
    write_run_manifest,
)


def _sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=48, nlev=40),
        dycore=DycoreConfig(model_type="shallow_water", discretization="cdgrid", dt=600.0),
        days=5,
    )


def test_manifest_has_all_provenance_sections() -> None:
    m = build_run_manifest(_sample_config(), command_line="legoesm run x.yaml")
    assert m["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    # The four provenance layers (master plan §8.2) + the run section.
    for section in ("legoESM", "reproducibility", "config", "result", "run"):
        assert section in m, f"missing manifest section {section!r}"
    # Le Sommer tag-block field names (so experiment.tag is a valid subset).
    assert set(m["legoESM"]) >= {"ref", "commit"}
    assert set(m["reproducibility"]) >= {"runner_tag", "python_version", "jax_version"}
    assert m["run"]["command_line"] == "legoesm run x.yaml"


def test_resolved_config_roundtrips_to_experiment_config() -> None:
    cfg = _sample_config()
    m = build_run_manifest(cfg)
    restored = experiment_config_from_dict(m["config"]["resolved_config"])
    assert restored == cfg


def test_state_digest_is_none_before_a_run() -> None:
    m = build_run_manifest(_sample_config())
    assert m["result"]["state_digest"] is None
    # ... and is recorded when supplied (the reproduce/checkpoint path).
    m2 = build_run_manifest(_sample_config(), state_digest="abc123")
    assert m2["result"]["state_digest"] == "abc123"


def test_optional_provenance_fields_are_threaded() -> None:
    m = build_run_manifest(
        _sample_config(),
        runner_tag="ocean-runners@v1",
        patches=["jerlov_sw.patch"],
        rng_seeds={"master": 42},
        dataset_provenance=[{"name": "JRA55do", "version": "sample"}],
        model_weights_provenance={"checkpoint": "physics.eqx"},
    )
    assert m["reproducibility"]["runner_tag"] == "ocean-runners@v1"
    assert m["reproducibility"]["patches"] == ["jerlov_sw.patch"]
    assert m["result"]["rng_seeds"] == {"master": 42}
    assert m["result"]["dataset_provenance"][0]["name"] == "JRA55do"
    assert m["result"]["model_weights_provenance"] == {"checkpoint": "physics.eqx"}


def test_dataset_provenance_entry_shape_and_checksum(tmp_path: Path) -> None:
    """Entry records identity + cheap integrity facts; sha256 is opt-in."""
    import hashlib

    from legoesm.driver.restart import dataset_provenance_entry

    f = tmp_path / "forcing.nc"
    f.write_bytes(b"not-really-netcdf")
    entry = dataset_provenance_entry(f, dataset_id="amip_sst_sic")
    assert entry["id"] == "amip_sst_sic"
    assert entry["path"] == str(f.resolve())
    assert entry["exists"] is True
    assert entry["size_bytes"] == len(b"not-really-netcdf")
    assert isinstance(entry["mtime_utc"], str) and "T" in entry["mtime_utc"]
    assert entry["sha256"] is None  # hashing is opt-in (multi-GB forcing)

    hashed = dataset_provenance_entry(f, compute_sha256=True)
    assert hashed["id"] == "forcing.nc"  # default id = file name
    assert hashed["sha256"] == hashlib.sha256(b"not-really-netcdf").hexdigest()
    # A catalog-supplied checksum passes through verbatim (no re-hash).
    pinned = dataset_provenance_entry(f, sha256="cafe" * 16)
    assert pinned["sha256"] == "cafe" * 16


def test_dataset_provenance_entry_missing_and_dir_store(tmp_path: Path) -> None:
    """Missing path is recorded (exists=False), never raises; Zarr-style
    directory stores get identity + mtime but no size/sha256."""
    from legoesm.driver.restart import dataset_provenance_entry

    gone = dataset_provenance_entry(tmp_path / "nope.zarr")
    assert gone["exists"] is False
    assert gone["size_bytes"] is None
    assert gone["mtime_utc"] is None
    assert gone["sha256"] is None

    store = tmp_path / "cache.zarr"
    store.mkdir()
    d = dataset_provenance_entry(store, compute_sha256=True)
    assert d["exists"] is True
    assert d["size_bytes"] is None  # directory: size/hash not computed
    assert d["sha256"] is None
    assert isinstance(d["mtime_utc"], str)


def test_dataset_provenance_entries_roundtrip_in_manifest(tmp_path: Path) -> None:
    """Entries are _json_safe-clean and survive write->read; a v1 manifest
    stays valid with and without them (populating the optional field is NOT
    a schema bump)."""
    from legoesm.driver.restart import (
        dataset_provenance_entry,
        validate_run_manifest,
    )

    f = tmp_path / "etopo.nc"
    f.write_bytes(b"\x00" * 8)
    entries = [dataset_provenance_entry(f, dataset_id="etopo_1deg")]
    out = tmp_path / "with_datasets"
    write_run_manifest(out, _sample_config(), dataset_provenance=entries)
    m = read_run_manifest(out)
    validate_run_manifest(m)
    assert m["result"]["dataset_provenance"] == entries

    out2 = tmp_path / "without_datasets"
    write_run_manifest(out2, _sample_config())
    m2 = read_run_manifest(out2)
    validate_run_manifest(m2)
    assert m2["result"]["dataset_provenance"] == []


def test_write_then_read_roundtrips(tmp_path: Path) -> None:
    cfg = _sample_config()
    path = write_run_manifest(tmp_path, cfg, command_line="legoesm run x.yaml")
    assert path.name == RUN_MANIFEST_FILENAME
    assert path.is_file()
    # JSON is sorted + indented (stable diffs).
    text = path.read_text()
    assert json.loads(text)  # parses
    # read_run_manifest accepts the directory or the file path.
    from_dir = read_run_manifest(tmp_path)
    from_file = read_run_manifest(path)
    assert from_dir == from_file
    assert (
        experiment_config_from_dict(from_dir["config"]["resolved_config"]) == cfg
    )


def test_write_creates_missing_directory(tmp_path: Path) -> None:
    nested = tmp_path / "runs" / "exp01"
    path = write_run_manifest(nested, _sample_config())
    assert path.is_file()


def test_numpy_and_path_provenance_are_normalized() -> None:
    """NumPy scalars / Path become JSON-native, not silently stringified later."""
    import numpy as np

    m = build_run_manifest(
        _sample_config(),
        rng_seeds={"master": np.int64(7)},
        dataset_provenance=[Path("/data/jra55.zarr")],
    )
    seed = m["result"]["rng_seeds"]["master"]
    assert seed == 7 and isinstance(seed, int) and not isinstance(seed, np.integer)
    assert m["result"]["dataset_provenance"] == ["/data/jra55.zarr"]


def test_in_memory_manifest_matches_written_file(tmp_path: Path) -> None:
    """build_run_manifest() must equal what write_run_manifest() persists.

    (Regression for the old ``default=str`` path, where read-back could differ
    from the in-memory dict — e.g. an int seed becoming a string.)
    """
    import numpy as np

    kwargs = dict(rng_seeds={"master": np.int64(7)}, runner_tag="t1")
    in_memory = build_run_manifest(_sample_config(), **kwargs)
    path = write_run_manifest(tmp_path, _sample_config(), **kwargs)
    written = read_run_manifest(path)
    # ``creation_time`` is stamped per-build, so drop it before comparing the
    # value-normalisation (the actual regression target: no lossy stringify).
    in_memory["run"].pop("creation_time")
    written["run"].pop("creation_time")
    assert written == in_memory


def test_unsupported_provenance_value_raises() -> None:
    """A non-JSON value fails loudly so the CLI warns instead of writing lossy."""
    import pytest

    with pytest.raises(TypeError):
        build_run_manifest(_sample_config(), rng_seeds={"bad": object()})


def test_unsupported_provenance_key_raises() -> None:
    """A non-string dict key fails loudly (no silent str() coercion/collision)."""
    import pytest

    with pytest.raises(TypeError, match="key must be str"):
        build_run_manifest(_sample_config(), rng_seeds={1: "a", "1": "b"})


def test_pytree_state_digest_is_backend_agnostic() -> None:
    """The final-state digest must handle any backend's pytree layout.

    Covers spectral-like (complex coefficients) and MPAS-like (``None`` leaf)
    states, not just the grid-point checkpoint layout.
    """
    import jax.numpy as jnp

    from legoesm.driver.restart import pytree_state_digest

    s1 = {"T": jnp.ones((2, 3)), "T_hat": jnp.ones((2,), dtype=jnp.complex64), "v": None}
    s2 = {"T": jnp.ones((2, 3)), "T_hat": jnp.ones((2,), dtype=jnp.complex64), "v": None}
    assert pytree_state_digest(s1) == pytree_state_digest(s2)  # deterministic
    s3 = {"T": jnp.zeros((2, 3)), "T_hat": jnp.ones((2,), dtype=jnp.complex64), "v": None}
    assert pytree_state_digest(s1) != pytree_state_digest(s3)  # value-sensitive
    # Extra trees (tracers / carry) participate in the digest.
    assert pytree_state_digest(s1, {"q": jnp.ones(2)}) != pytree_state_digest(s1)


def test_pytree_state_digest_captures_structure() -> None:
    """Structure (keys / None slots), not just array bytes, is in the digest."""
    import jax.numpy as jnp

    from legoesm.driver.restart import pytree_state_digest

    arr = jnp.ones(2)
    # A None slot appearing must change the digest (it is structural in JAX).
    assert pytree_state_digest({"v": None, "a": arr}) != pytree_state_digest({"a": arr})
    # The same array under a different key must change the digest.
    assert pytree_state_digest({"a": arr}) != pytree_state_digest({"b": arr})


def test_reproduce_rejects_output_equal_to_reference_dir(tmp_path: Path) -> None:
    """`reproduce --output <reference dir>` is rejected before any rerun."""
    import argparse

    import pytest

    from legoesm.cli import cmd_reproduce
    from legoesm.driver.restart import record_state_digest

    write_run_manifest(tmp_path, _sample_config())
    record_state_digest(tmp_path, "deadbeef")  # complete the reference
    args = argparse.Namespace(
        manifest=str(tmp_path / RUN_MANIFEST_FILENAME),
        check=True,
        output=str(tmp_path),
    )
    with pytest.raises(SystemExit) as exc:
        cmd_reproduce(args)
    assert exc.value.code == 2  # guard exit, before the model ever runs


def test_no_digest_recorded_for_failed_run(tmp_path: Path, monkeypatch) -> None:
    """A BLOWUP/failed run must not record a digest (no false reproduce reference)."""
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver

    driver = ModelDriver(ExperimentConfig())
    driver._output_dir = tmp_path
    driver._mpi_rank = None
    write_run_manifest(tmp_path, driver._input_config)  # manifest with no digest yet
    monkeypatch.setattr(driver, "_run_compiled", lambda *a, **k: "BLOWUP at day 1.0")

    status = driver.run(compiled=True)
    assert status.startswith("BLOWUP")
    assert read_run_manifest(tmp_path)["result"]["state_digest"] is None


def test_record_and_read_state_digest(tmp_path: Path) -> None:
    """The post-run digest update fills result.state_digest and nothing else."""
    import pytest

    from legoesm.driver.restart import (
        record_state_digest,
        recorded_state_digest,
        validate_run_manifest,
    )

    write_run_manifest(tmp_path, _sample_config())
    m0 = read_run_manifest(tmp_path)
    # No digest yet -> reproduce --check has nothing to compare against.
    with pytest.raises(ValueError, match="no recorded"):
        recorded_state_digest(m0)

    record_state_digest(tmp_path, "abc123")
    m1 = read_run_manifest(tmp_path)
    assert m1["result"]["state_digest"] == "abc123"
    assert recorded_state_digest(m1) == "abc123"
    # Provenance (config/env) is untouched, and the manifest stays valid.
    assert m1["config"]["config_hash"] == m0["config"]["config_hash"]
    assert m1["legoESM"] == m0["legoESM"]
    validate_run_manifest(m1)


def test_record_state_digest_rejects_invalid_manifest(tmp_path: Path) -> None:
    import pytest

    from legoesm.driver.restart import RUN_MANIFEST_FILENAME, record_state_digest

    (tmp_path / RUN_MANIFEST_FILENAME).write_text("not json")
    with pytest.raises(Exception):
        record_state_digest(tmp_path, "x")


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    write_run_manifest(tmp_path, _sample_config())
    assert (tmp_path / RUN_MANIFEST_FILENAME).is_file()
    assert not list(tmp_path.glob("*.tmp")), "atomic temp file not cleaned up"


def test_manifest_for_a_shipped_config(tmp_path: Path) -> None:
    """End-to-end: a real config -> manifest whose resolved_config rebuilds it.

    Mirrors what ``legoesm run`` writes at startup for config/williamson_test2.
    """
    from legoesm.config import Config

    ec = Config.from_yaml("config/williamson_test2.yaml").to_experiment_config()
    path = write_run_manifest(tmp_path, ec, command_line="legoesm run williamson_test2.yaml")
    m = read_run_manifest(path)
    assert experiment_config_from_dict(m["config"]["resolved_config"]) == ec
    assert m["config"]["config_hash"]  # non-empty sha256


def test_driver_manifest_uses_resolved_normalized_config(tmp_path: Path) -> None:
    """The driver records its RESOLVED config (normalized grid alias), rank-0 only.

    Regression for: (1) manifest describing the pre-normalized input rather than
    the config the run uses; (2) every MPI rank writing the same file.
    """
    from legoesm.driver.config import ExperimentConfig, GridConfig
    from legoesm.driver.model_driver import ModelDriver

    # Legacy alias "voronoi" normalizes to "mpas" in ModelDriver.__init__.
    driver = ModelDriver(ExperimentConfig(grid=GridConfig(grid_type="voronoi")))
    assert driver.config.grid.grid_type == "mpas"

    driver._output_dir = tmp_path
    driver._mpi_rank = None  # single-rank -> writes
    driver._write_run_manifest()
    m = read_run_manifest(tmp_path)
    assert m["config"]["resolved_config"]["grid"]["grid_type"] == "mpas"
    # The master RNG seed is recorded for reproducibility.
    assert m["result"]["rng_seeds"] == {"master": driver.config.seed}

    # A non-zero MPI rank must NOT write (no race on the shared file).
    rank1_dir = tmp_path / "rank1"
    rank1_dir.mkdir()
    driver._output_dir = rank1_dir
    driver._mpi_rank = 1
    driver._write_run_manifest()
    assert not (rank1_dir / RUN_MANIFEST_FILENAME).exists()


def test_driver_manifest_records_dataset_provenance(tmp_path: Path) -> None:
    """Path-typed ExperimentConfig fields land in [result].dataset_provenance;
    empty path fields are skipped (default config -> empty list)."""
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver

    forcing = tmp_path / "amip_sst.nc"
    forcing.write_bytes(b"\x01\x02")
    driver = ModelDriver(ExperimentConfig(forcing_path=str(forcing)))
    driver._output_dir = tmp_path / "run"
    driver._mpi_rank = None
    driver._write_run_manifest()
    m = read_run_manifest(tmp_path / "run")
    entries = m["result"]["dataset_provenance"]
    assert [e["id"] for e in entries] == ["forcing_path"]
    assert entries[0]["path"] == str(forcing.resolve())
    assert entries[0]["exists"] is True

    # No path fields set -> provenance list stays empty (no noise entries).
    bare = ModelDriver(ExperimentConfig())
    bare._output_dir = tmp_path / "bare"
    bare._mpi_rank = None
    bare._write_run_manifest()
    assert read_run_manifest(tmp_path / "bare")["result"]["dataset_provenance"] == []


def test_driver_manifest_records_input_config_not_setup_mutated(tmp_path: Path) -> None:
    """The manifest must record the INPUT config so `reproduce` replays the run.

    Regression: setup() mutates self.config in place (e.g. the CFL-driven dt
    reduction). Recording the post-mutation config (dt already reduced) takes a
    different setup path and fails to reproduce; the manifest must capture the
    input (pre-mutation) config instead.
    """
    from legoesm.driver.config import DycoreConfig, ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver

    driver = ModelDriver(ExperimentConfig(dycore=DycoreConfig(dt=1800.0)))
    # Simulate a setup-time mutation (e.g. CFL dt reduction 1800 -> 600).
    driver.config = driver.config._replace(
        dycore=driver.config.dycore._replace(dt=600.0)
    )
    driver._output_dir = tmp_path
    driver._mpi_rank = None
    driver._write_run_manifest()
    m = read_run_manifest(tmp_path)
    assert m["config"]["resolved_config"]["dycore"]["dt"] == 1800.0  # input, not mutated


def test_driver_manifest_is_write_once(tmp_path: Path) -> None:
    """A re-setup must preserve the original run-start manifest, not clobber it."""
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver

    driver = ModelDriver(ExperimentConfig())
    driver._output_dir = tmp_path
    driver._mpi_rank = None
    driver._write_run_manifest()
    original = (tmp_path / RUN_MANIFEST_FILENAME).read_text()

    # Second call (retry / resume) must leave the original bytes untouched.
    driver._write_run_manifest()
    assert (tmp_path / RUN_MANIFEST_FILENAME).read_text() == original


def test_driver_manifest_rejects_reuse_with_different_config(tmp_path: Path) -> None:
    """Reusing an output dir with a different config is fatal (no provenance mix)."""
    import pytest

    from legoesm.driver.config import ExperimentConfig, GridConfig
    from legoesm.driver.model_driver import ModelDriver

    d1 = ModelDriver(ExperimentConfig(grid=GridConfig(resolution=48)))
    d1._output_dir = tmp_path
    d1._mpi_rank = None
    d1._write_run_manifest()  # writes manifest for config A

    d2 = ModelDriver(ExperimentConfig(grid=GridConfig(resolution=96)))
    d2._output_dir = tmp_path
    d2._mpi_rank = None
    with pytest.raises(RuntimeError, match="DIFFERENT config"):
        d2._write_run_manifest()


def test_driver_manifest_failure_is_fatal(tmp_path: Path, monkeypatch) -> None:
    """A manifest write failure aborts setup rather than silently continuing."""
    import pytest

    import legoesm.driver.restart as restart_mod
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(restart_mod, "write_run_manifest", _boom)
    driver = ModelDriver(ExperimentConfig())
    driver._output_dir = tmp_path  # empty dir -> takes the write path
    driver._mpi_rank = None
    with pytest.raises(OSError, match="disk full"):
        driver._write_run_manifest()


def test_driver_manifest_fails_closed_on_corrupt_existing(tmp_path: Path) -> None:
    """A present-but-corrupt manifest must abort, not be silently preserved."""
    import pytest

    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver

    (tmp_path / RUN_MANIFEST_FILENAME).write_text("}{ not valid json")
    driver = ModelDriver(ExperimentConfig())
    driver._output_dir = tmp_path
    driver._mpi_rank = None
    with pytest.raises(RuntimeError, match="invalid"):
        driver._write_run_manifest()


def test_driver_manifest_fails_closed_on_schema_invalid_existing(tmp_path: Path) -> None:
    """Valid JSON but missing the reconstructable config is also fail-closed."""
    import json

    import pytest

    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver

    # Looks plausible (has schema_version + config_hash) but no resolved_config.
    (tmp_path / RUN_MANIFEST_FILENAME).write_text(
        json.dumps({"schema_version": 999, "config": {"config_hash": "x"}})
    )
    driver = ModelDriver(ExperimentConfig())
    driver._output_dir = tmp_path
    driver._mpi_rank = None
    with pytest.raises(RuntimeError, match="invalid"):
        driver._write_run_manifest()


def test_validate_run_manifest_accepts_a_real_manifest() -> None:
    from legoesm.driver.restart import validate_run_manifest

    validate_run_manifest(build_run_manifest(_sample_config()))  # no raise


def test_validate_run_manifest_rejects_bad_schema_version() -> None:
    import pytest

    from legoesm.driver.restart import validate_run_manifest

    m = build_run_manifest(_sample_config())
    m["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        validate_run_manifest(m)


def test_validate_run_manifest_rejects_missing_resolved_config() -> None:
    import pytest

    from legoesm.driver.restart import validate_run_manifest

    m = build_run_manifest(_sample_config())
    m["config"]["resolved_config"] = {}
    with pytest.raises(ValueError, match="resolved_config"):
        validate_run_manifest(m)


def test_validate_run_manifest_rejects_null_section() -> None:
    import pytest

    from legoesm.driver.restart import validate_run_manifest

    m = build_run_manifest(_sample_config())
    m["run"] = None  # present name, but not an object
    with pytest.raises(ValueError, match=r"\[run\]"):
        validate_run_manifest(m)


def test_validate_run_manifest_rejects_section_missing_keys() -> None:
    import pytest

    from legoesm.driver.restart import validate_run_manifest

    m = build_run_manifest(_sample_config())
    m["legoESM"] = {}  # an object, but lacks ref/commit
    with pytest.raises(ValueError, match="legoESM"):
        validate_run_manifest(m)


def test_write_run_manifest_exclusive_raises_on_existing(tmp_path: Path) -> None:
    import pytest

    write_run_manifest(tmp_path, _sample_config(), exclusive=True)  # creates it
    with pytest.raises(FileExistsError):
        write_run_manifest(tmp_path, _sample_config(), exclusive=True)  # loses the race


def test_validate_run_manifest_rejects_hash_config_mismatch() -> None:
    """A config_hash that does not match its resolved_config is rejected."""
    import pytest

    from legoesm.driver.restart import validate_run_manifest

    m = build_run_manifest(_sample_config())
    m["config"]["config_hash"] = "deadbeef"  # no longer matches resolved_config
    with pytest.raises(ValueError, match="does not match"):
        validate_run_manifest(m)



def test_validate_run_manifest_accepts_schema_growth() -> None:
    """A manifest written BEFORE new ExperimentConfig fields existed must still
    validate: rebuild fills the new fields with defaults, and the restricted
    hash over the manifest's own keys matches.  Reproduces the 2026-07-27
    chain-killer (every added morrison_* field bricked older run dirs)."""
    import hashlib
    import json

    from legoesm.driver.restart import validate_run_manifest

    m = build_run_manifest(_sample_config())
    resolved = m["config"]["resolved_config"]
    # Simulate the OLD writer: drop recently-added fields from the stored
    # dict and store the hash a pre-change writer would have recorded.
    for k in ("morrison_flavor", "morrison_fall_a_i",
              "morrison_ice_snow_d_auto", "morrison_hom_ice_nuc_N"):
        assert k in resolved, f"test premise broken: {k} not serialized"
        del resolved[k]
    m["config"]["config_hash"] = hashlib.sha256(json.dumps(
        resolved, sort_keys=True).encode("utf-8")).hexdigest()
    validate_run_manifest(m)  # must not raise


def test_schema_growth_refuses_non_default_new_field() -> None:
    """The growth tolerance must NOT excuse a NEW field at a NON-default
    value: that is a real config difference, not schema growth."""
    import hashlib
    import json

    from legoesm.driver.restart import config_hash_matches

    # Morrison baseline so the ONLY delta below is the NEW field itself
    # (codex round 2: a microphysics+flavor double change confounded the
    # negative case — the stored microphysics hash alone would refuse it).
    old = _sample_config()._replace(microphysics="morrison")
    m = build_run_manifest(old)
    resolved = dict(m["config"]["resolved_config"])
    del resolved["morrison_flavor"]
    stored_hash = hashlib.sha256(json.dumps(
        resolved, sort_keys=True).encode("utf-8")).hexdigest()
    same_but_grown = old  # defaults -> accepted
    assert config_hash_matches(stored_hash, resolved, same_but_grown)
    different = old._replace(morrison_flavor="sam")
    assert not config_hash_matches(stored_hash, resolved, different)


def test_schema_growth_still_detects_tamper() -> None:
    """Restricted-hash acceptance must not weaken tamper detection on keys
    the manifest DOES store."""
    import hashlib
    import json

    from legoesm.driver.restart import config_hash_matches

    old = _sample_config()
    m = build_run_manifest(old)
    resolved = dict(m["config"]["resolved_config"])
    del resolved["morrison_flavor"]
    stored_hash = hashlib.sha256(json.dumps(
        resolved, sort_keys=True).encode("utf-8")).hexdigest()
    tampered = old._replace(days=999)
    assert not config_hash_matches(stored_hash, resolved, tampered)


def test_schema_growth_fails_closed_when_defaults_unavailable() -> None:
    """A config type that cannot default-construct (ocean records: required
    constructor args) gets NO growth tolerance — strict False, never an
    exception (codex 2026-07-27 round 2: ocean fallback semantics)."""
    import hashlib
    import json
    import typing

    from legoesm.driver.restart import config_hash_matches

    class _NoDefaults(typing.NamedTuple):
        a: float
        b: float = 1.0

    cfg = _NoDefaults(a=2.0)
    stored = {"a": 2.0}  # pre-growth manifest lacking 'b'
    stored_hash = hashlib.sha256(json.dumps(
        stored, sort_keys=True).encode("utf-8")).hexdigest()
    assert config_hash_matches(stored_hash, stored, cfg,
                               kind="atmosphere") is False
