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
from legoesm.io.restart import (
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

    import legoesm.io.restart as restart_mod
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
    from legoesm.io.restart import validate_run_manifest

    validate_run_manifest(build_run_manifest(_sample_config()))  # no raise


def test_validate_run_manifest_rejects_bad_schema_version() -> None:
    import pytest

    from legoesm.io.restart import validate_run_manifest

    m = build_run_manifest(_sample_config())
    m["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        validate_run_manifest(m)


def test_validate_run_manifest_rejects_missing_resolved_config() -> None:
    import pytest

    from legoesm.io.restart import validate_run_manifest

    m = build_run_manifest(_sample_config())
    m["config"]["resolved_config"] = {}
    with pytest.raises(ValueError, match="resolved_config"):
        validate_run_manifest(m)


def test_validate_run_manifest_rejects_null_section() -> None:
    import pytest

    from legoesm.io.restart import validate_run_manifest

    m = build_run_manifest(_sample_config())
    m["run"] = None  # present name, but not an object
    with pytest.raises(ValueError, match=r"\[run\]"):
        validate_run_manifest(m)


def test_validate_run_manifest_rejects_section_missing_keys() -> None:
    import pytest

    from legoesm.io.restart import validate_run_manifest

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

    from legoesm.io.restart import validate_run_manifest

    m = build_run_manifest(_sample_config())
    m["config"]["config_hash"] = "deadbeef"  # no longer matches resolved_config
    with pytest.raises(ValueError, match="does not match"):
        validate_run_manifest(m)
