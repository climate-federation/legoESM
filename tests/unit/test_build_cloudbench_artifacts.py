"""scripts/data/build_cloudbench_artifacts.py: case listing, sharding, skip, failures."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "data" / "build_cloudbench_artifacts.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_cloudbench_artifacts", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _forcing(tmp_path):
    root = tmp_path / "forcing"
    for site in (0, 1, 2, 10):
        for month in (1, 7):
            (root / str(site) / str(month) / "amip").mkdir(parents=True)
    (root / "3" / "1" / "amip-p4k").mkdir(parents=True)        # other experiment only
    (root / "MD5SUMS").write_text("")
    return root


def test_list_cases_and_shard(tmp_path):
    mod = _load()
    cases = mod.list_cases(_forcing(tmp_path), "amip")
    assert cases == [(s, m) for s in (0, 1, 2, 10) for m in (1, 7)]
    parts = [mod.shard(cases, i, 3) for i in range(3)]
    assert sorted(sum(parts, [])) == cases and all(parts)
    with pytest.raises(SystemExit):
        mod.shard(cases, 3, 3)


def _tiny_artifact(code_sha256):
    import numpy as np
    from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact
    z, t = np.array([10.0, 20.0]), np.array([0.0, 60.0])
    prof = np.ones((2, 2))
    return LESReferenceArtifact(
        case_name="c", sgs="s", heights_m=z, times_s=t, theta=300 * prof, u=prof, v=prof,
        wtheta_resolved=0 * prof, wtheta_sgs=0 * prof, prescribe="none",
        les_setup={"converter": {"code_sha256": code_sha256}})


def test_convert_skips_only_current_loadable_artifacts(tmp_path, monkeypatch):
    mod = _load()
    from legoesm.atmosphere.les_suite import cloudbench as cb
    from legoesm.atmosphere.les_suite.bridge import save_artifact
    out = tmp_path / "out"
    out.mkdir()
    prov = {"code_sha256": "now"}
    good = out / f"{cb.case_name(0, 1, 'amip')}.npz"
    save_artifact(_tiny_artifact("now"), good)
    name, status, _, tb = mod._convert((0, 1, "amip", str(tmp_path), str(out), prov))
    assert status == "skipped" and tb is None

    def boom(*a, **k):
        raise RuntimeError("stream broke")
    monkeypatch.setattr(cb, "build_artifact", boom)
    save_artifact(_tiny_artifact("older code"), good)
    assert mod._convert((0, 1, "amip", str(tmp_path), str(out), prov))[1] == "failed"
    good.write_bytes(b"corrupt")
    assert mod._convert((0, 1, "amip", str(tmp_path), str(out), prov))[1] == "failed"
    art = _tiny_artifact("now")
    object.__setattr__(art, "les_setup", None)
    save_artifact(art, good)
    assert mod._convert((0, 1, "amip", str(tmp_path), str(out), prov))[1] == "failed"
    name, status, _, tb = mod._convert((1, 7, "amip", str(tmp_path), str(out), prov))
    assert status == "failed" and "stream broke" in tb
    assert not list(out.glob("*tmp*"))


def test_code_hash_follows_the_source_of_imported_modules(tmp_path, monkeypatch):
    mod = _load()
    import sys
    import types

    import legoesm.atmosphere.les_suite.cloudbench  # noqa: F401
    assert "legoesm.atmosphere.forcing.idealized.large_scale_forcing" in sys.modules
    src = tmp_path / "probe.py"
    src.write_text("x = 1\n")
    probe = types.ModuleType("legoesm._hash_probe")
    probe.__file__ = str(src)
    monkeypatch.setitem(sys.modules, "legoesm._hash_probe", probe)
    a = mod._converter_code_sha256()
    assert a == mod._converter_code_sha256() and len(a) == 64
    src.write_text("x = 2\n")                       # same module set, new source
    assert mod._converter_code_sha256() != a


def _die(job):
    import os
    from pathlib import Path
    (Path(job[4]) / f"died-{os.getpid()}").write_text("")
    os._exit(9)


def test_a_dead_worker_stops_the_batch_with_failure(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "_convert", _die)
    out = tmp_path / "out"
    rc = mod.main(["--forcing-dir", str(_forcing(tmp_path)), "--out-dir", str(out),
                   "--workers", "2"])
    assert rc == 1
    rec = json.loads((out / "failures.jsonl").read_text().splitlines()[-1])
    assert rec["error"].startswith("a worker process died") and len(rec["unfinished"]) == 8
    assert list(out.glob("died-*"))                     # the worker really ran and died


def test_convert_removes_a_partial_file_when_the_save_fails(tmp_path, monkeypatch):
    mod = _load()
    from legoesm.atmosphere.les_suite import bridge
    from legoesm.atmosphere.les_suite import cloudbench as cb
    out = tmp_path / "out"
    out.mkdir()

    class Art:
        les_setup = {}
    monkeypatch.setattr(cb, "build_artifact", lambda *a, **k: Art())

    def half_save(art, path):
        path.write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(bridge, "save_artifact", half_save)
    prov = {"code_sha256": "now"}
    _, status, _, tb = mod._convert((1, 7, "amip", str(tmp_path), str(out), prov))
    assert status == "failed" and "disk full" in tb
    assert Art.les_setup["converter"] == prov
    assert list(out.glob("*.npz")) == [] and list((out / ".partial").iterdir()) == []


def test_held_out_file_is_written_once_and_checked(tmp_path):
    mod = _load()
    path = tmp_path / "held_out_sites.json"
    mod._write_held_out(path, {"a": 1})
    mod._write_held_out(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["held_out_sites.json"]
    with pytest.raises(SystemExit, match="refusing to mix splits"):
        mod._write_held_out(path, {"a": 2})


def test_main_refuses_a_different_held_out_draw(tmp_path):
    mod = _load()
    out = tmp_path / "out"
    out.mkdir()
    (out / "held_out_sites.json").write_text(json.dumps({"held_out_sites": [0]}))
    with pytest.raises(SystemExit, match="refusing to mix splits"):
        mod.main(["--forcing-dir", str(_forcing(tmp_path)), "--out-dir", str(out)])
