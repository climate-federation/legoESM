"""scripts/data/migrate_cloudbench_nudging.py: only the relaxation changes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from legoesm.atmosphere.les_suite import cloudbench as cb

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "data" / "migrate_cloudbench_nudging.py"


def _load():
    spec = importlib.util.spec_from_file_location("migrate_cloudbench_nudging", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _old_artifact(path, mod):
    z = (np.arange(1, 480) + 0.5) * 12.5
    setup = {"source": "CloudBench", "height_offset_m": 6.25,
             "forcing_recipe": {**mod.OLD_RECIPE, "source": "swirl-lm defaults",
                                "tau_wind_s": 21600.0, "coriolis": "none"}}
    meta = {"case_name": "cloudbench_s001_m01_amip", "les_setup": setup}
    arrays = {"heights_m": z, "theta": np.random.default_rng(0).normal(size=(3, z.size)),
              "nudge_rate": mod._old_rate(z - 6.25),
              "__meta__": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)}
    np.savez_compressed(path, **arrays)
    return arrays


def test_only_the_relaxation_changes(tmp_path):
    mod = _load()
    src, dst = tmp_path / "old", tmp_path / "new"
    src.mkdir()
    old = _old_artifact(src / "cloudbench_s001_m01_amip.npz", mod)
    (src / "held_out_sites.json").write_text("[1]")
    (src / ".partial").mkdir()                 # the builder's scratch directory
    mod.main([str(src), str(dst)])
    assert json.loads((dst / "nudging_migration.json").read_text())["skipped_non_files"] == [".partial"]
    with np.load(dst / "cloudbench_s001_m01_amip.npz") as new:
        assert new["theta"].tobytes() == old["theta"].tobytes()
        assert new["heights_m"].tobytes() == old["heights_m"].tobytes()
        assert np.allclose(new["nudge_rate"], cb.nudge_rate(old["heights_m"] - 6.25), rtol=0, atol=0)
        assert not np.allclose(new["nudge_rate"], old["nudge_rate"])
        meta = json.loads(bytes(new["__meta__"]).decode())
    rec = meta["les_setup"]["forcing_recipe"]
    assert {k: rec[k] for k in cb.NUDGE_RECIPE} == cb.NUDGE_RECIPE
    assert rec["tau_wind_s"] == 21600.0 and rec["coriolis"] == "none"
    assert (dst / "held_out_sites.json").read_text() == "[1]"
    with np.load(src / "cloudbench_s001_m01_amip.npz") as untouched:
        assert untouched["nudge_rate"].tobytes() == old["nudge_rate"].tobytes()


def test_an_artifact_not_on_the_old_recipe_is_refused(tmp_path):
    mod = _load()
    src = tmp_path / "old"
    src.mkdir()
    _old_artifact(src / "cloudbench_s001_m01_amip.npz", mod)
    mod.main([str(src), str(tmp_path / "new")])
    with pytest.raises(ValueError, match="expected the old"):     # already migrated
        mod.migrate_npz(tmp_path / "new" / "cloudbench_s001_m01_amip.npz", tmp_path / "x.npz")
    with pytest.raises(SystemExit, match="differ from SRC"):
        mod.main([str(src), str(src)])


def test_a_non_empty_destination_is_refused(tmp_path):
    mod = _load()
    src, dst = tmp_path / "old", tmp_path / "new"
    src.mkdir()
    dst.mkdir()
    (dst / "stale.npz").write_bytes(b"x")
    with pytest.raises(SystemExit, match="not empty"):
        mod.main([str(src), str(dst)])
