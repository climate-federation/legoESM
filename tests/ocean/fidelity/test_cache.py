"""Unit tests for legoesm.ocean.fidelity.cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from legoesm.ocean.fidelity import cache


def test_get_cache_root_uses_primary_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGOESM_OCEAN_FIDELITY_CACHE", str(tmp_path / "primary"))
    monkeypatch.delenv("LEGOESM_CACHE_DIR", raising=False)
    root = cache.get_cache_root()
    assert root == tmp_path / "primary"
    assert root.is_dir()


def test_get_cache_root_falls_back_to_shared_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LEGOESM_OCEAN_FIDELITY_CACHE", raising=False)
    monkeypatch.setenv("LEGOESM_CACHE_DIR", str(tmp_path / "shared"))
    root = cache.get_cache_root()
    assert root == tmp_path / "shared" / "ocean_fidelity"
    assert root.is_dir()


def test_get_cache_root_default_home(tmp_path, monkeypatch):
    monkeypatch.delenv("LEGOESM_OCEAN_FIDELITY_CACHE", raising=False)
    monkeypatch.delenv("LEGOESM_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = cache.get_cache_root()
    assert root == tmp_path / ".cache" / "legoesm" / "ocean_fidelity"
    assert root.is_dir()


def test_sub_creates_known_subdir(isolated_cache):
    path = cache.sub("veros")
    assert path.is_dir()
    assert path.name == "veros"
    assert path.parent == isolated_cache


def test_sub_each_allowed_subdir(isolated_cache):
    for name in cache.ALLOWED_SUBDIRS:
        path = cache.sub(name)
        assert path.is_dir()
        assert path.name == name


def test_sub_rejects_unknown(isolated_cache):
    with pytest.raises(ValueError, match="Unknown cache subdir"):
        cache.sub("not_a_real_subdir")


def test_hash_key_stable_across_dict_order():
    a = cache.hash_key({"a": 1, "b": 2, "c": [1, 2, 3]})
    b = cache.hash_key({"c": [1, 2, 3], "b": 2, "a": 1})
    assert a == b


def test_hash_key_returns_16_hex():
    key = cache.hash_key({"x": 1})
    assert isinstance(key, str)
    assert len(key) == 16
    int(key, 16)  # decodes as hex


def test_hash_key_changes_on_value_change():
    a = cache.hash_key({"a": 1})
    b = cache.hash_key({"a": 2})
    assert a != b


def test_hash_key_coerces_nonjson_via_str():
    key = cache.hash_key({"path": Path("/tmp/foo")})
    assert len(key) == 16


def test_write_json_round_trip(tmp_path):
    path = tmp_path / "out.json"
    payload = {"x": 1, "y": [1, 2, 3], "nested": {"a": True}}
    cache.write_json(path, payload)
    assert cache.read_json(path) == payload


def test_write_json_overwrites_atomically(tmp_path):
    path = tmp_path / "out.json"
    cache.write_json(path, {"version": 1})
    cache.write_json(path, {"version": 2})
    assert cache.read_json(path) == {"version": 2}


def test_write_json_creates_parent_dirs(tmp_path):
    path = tmp_path / "a" / "b" / "c.json"
    cache.write_json(path, {"ok": True})
    assert path.exists()


def test_write_json_cleans_tmp_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "out.json"

    def boom(*args, **kwargs):
        raise RuntimeError("simulated json failure")

    monkeypatch.setattr(cache.json, "dump", boom)
    with pytest.raises(RuntimeError, match="simulated json failure"):
        cache.write_json(path, {"x": 1})
    assert not path.exists()
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp_")]
    assert leftovers == []
