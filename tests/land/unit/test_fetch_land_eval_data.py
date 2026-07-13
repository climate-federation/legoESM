"""Unit tests for ``scripts/data/fetch_land_eval_data.py`` (no network)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "data" / "fetch_land_eval_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fetch_led", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass type-resolution (which looks up
    # sys.modules[cls.__module__] under `from __future__ import annotations`
    # on Python 3.12+) succeeds.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_registry_is_well_formed():
    mod = _load_module()
    assert mod.DATASETS
    for key, ds in mod.DATASETS.items():
        assert ds.key == key
        assert ds.access in {"in_tree", "fetch", "bundle", "derived"}
        assert ds.marker  # every dataset has a presence marker


def test_check_manifest_detects_present_and_missing(tmp_path):
    mod = _load_module()
    # Materialise just the fortran_v2 marker under a fake repo root.
    ds = mod.DATASETS["fortran_v2"]
    d = tmp_path / ds.repo_relpath
    d.mkdir(parents=True)
    (d / "CHATS7_2007-05_flux.out").write_text("1 2 3\n")

    status = mod.check_manifest(tmp_path, bundle_root=None)
    assert status["fortran_v2"]["present"] is True
    assert status["fortran_v2"]["path"] == str(d)
    # A dataset we did not create is reported missing, not crashed.
    assert status["zenodo_obs"]["present"] is False
    assert status["zenodo_obs"]["path"] is None


def test_check_manifest_uses_bundle_root(tmp_path):
    mod = _load_module()
    bundle = tmp_path / "bundle"
    ds = mod.DATASETS["zenodo_obs"]
    d = bundle / ds.bundle_relpath
    d.mkdir(parents=True)
    (d / "chats_30min_data_2007_05_ver.csv").write_text("x\n")

    status = mod.check_manifest(tmp_path, bundle_root=bundle)
    assert status["zenodo_obs"]["present"] is True
    assert status["zenodo_obs"]["path"] == str(d)


def test_main_check_runs(tmp_path, capsys, monkeypatch):
    mod = _load_module()
    monkeypatch.delenv(mod.DATA_ROOT_ENV, raising=False)
    rc = mod.main(["--check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Land-evaluation data check" in out


def test_fetch_bad_target_rejected():
    mod = _load_module()
    with pytest.raises(SystemExit):
        mod.main(["--fetch", "not_a_source"])
