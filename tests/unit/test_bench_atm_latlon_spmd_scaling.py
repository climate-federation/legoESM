"""Smoke test for the lat-band SPMD scaling bench
(scripts/bench/bench_atm_latlon_spmd_scaling.py): the model/IC builder produces a
consistent C-grid state, and the device-oversubscription guard fires. Timing /
multi-device runs are exercised on-cluster, not here."""
from __future__ import annotations

import importlib.util
import os

import pytest

_BENCH = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "bench",
    "bench_atm_latlon_spmd_scaling.py")


def _load():
    spec = importlib.util.spec_from_file_location("_bench_spmd", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_returns_consistent_cgrid_state():
    mod = _load()
    n_lat, n_lon, nlev = 8, 8, 4
    model, c0 = mod._build(n_lat, n_lon, nlev)
    # C-grid staggering: u (n_lat, n_lon+1, nlev), v (n_lat+1, n_lon, nlev),
    # T/p_s at centers.
    assert c0.u.shape == (n_lat, n_lon + 1, nlev)
    assert c0.v.shape == (n_lat + 1, n_lon, nlev)
    assert c0.T.shape == (n_lat, n_lon, nlev)
    assert c0.p_s.shape == (n_lat, n_lon)
    assert model.grid.n_lat == n_lat


def test_main_rejects_device_oversubscription(monkeypatch):
    """--n-devices greater than the visible device count must SystemExit, not
    silently run on fewer bands."""
    import jax
    mod = _load()
    too_many = len(jax.devices()) + 8
    monkeypatch.setattr(
        "sys.argv",
        ["bench", "--n-devices", str(too_many), "--n-lat", "8", "--n-lon", "8",
         "--nlev", "4", "--steps", "2"])
    with pytest.raises(SystemExit):
        mod.main()


def test_main_rejects_negative_segment_steps(monkeypatch):
    """--segment-steps < 0 must SystemExit, not silently fall back."""
    mod = _load()
    monkeypatch.setattr(
        "sys.argv",
        ["bench", "--n-devices", "1", "--n-lat", "8", "--n-lon", "8",
         "--nlev", "4", "--steps", "2", "--segment-steps", "-1"])
    with pytest.raises(SystemExit):
        mod.main()


def test_main_segment_mode_single_device(tmp_path, monkeypatch):
    """M2b --segment-steps lane end-to-end (nd=1, tiny grid): the record
    carries the segment receipt fields (segment_mode, per_block_ms, per-step
    derivation, metadata.extra facts) and a HEALTHY run is stamped
    finite_ok=true / valid=true with every block completed."""
    import json
    mod = _load()
    out = tmp_path / "seg.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        ["bench", "--n-devices", "1", "--n-lat", "8", "--n-lon", "8",
         "--nlev", "4", "--steps", "3", "--warmup", "1",
         "--segment-steps", "2", "--dt", "60", "--out", str(out)])
    assert mod.main() == 0
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert rec["segment_mode"] is True
    assert rec["segment_steps"] == 2
    assert len(rec["per_block_ms"]) == 3
    assert len(rec["per_step_ms"]) == 3
    # per-step numbers derive from whole blocks: block_ms / segment_steps.
    assert rec["per_step_ms"][1] == pytest.approx(
        rec["per_block_ms"][1] / 2.0, rel=0.05)
    # Validity receipt (codex batch4): healthy run -> finite + valid, all
    # blocks completed, throughput present.
    assert rec["finite_ok"] is True
    assert rec["valid"] is True
    assert rec["completed_blocks"] == 3
    assert rec["sypd"] is not None and rec["sypd"] > 0.0
    assert rec["metadata"]["extra"]["finite_ok"] is True
    assert rec["metadata"]["extra"]["valid"] is True
    assert rec["metadata"]["extra"]["completed_blocks"] == 3
    assert rec["metadata"]["extra"]["segment_mode"] is True
    assert rec["metadata"]["extra"]["segment_steps"] == 2
    # nd=1: no mesh, no geometry stacks -> residency field is None.
    assert rec["metadata"]["extra"]["geometry_bytes_per_device"] is None


def test_main_segment_mode_divergence_marks_record_invalid(
        tmp_path, monkeypatch, capsys):
    """A diverging segment run (huge dt -> non-finite state) must be stamped
    finite_ok=false + valid=false with nulled throughput, a completed_blocks
    receipt, and a diverged entry under metadata._incomplete — never
    serialized as a normal measurement (codex batch4 M2b defect)."""
    import json
    mod = _load()
    out = tmp_path / "div.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        ["bench", "--n-devices", "1", "--n-lat", "8", "--n-lon", "8",
         "--nlev", "4", "--steps", "3", "--warmup", "1",
         "--segment-steps", "2", "--dt", "1e18", "--out", str(out)])
    assert mod.main() == 0
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert rec["finite_ok"] is False
    assert rec["valid"] is False
    # The loop stops at the first non-finite block; every timed block is
    # receipted.
    assert 1 <= rec["completed_blocks"] <= 3
    assert len(rec["per_block_ms"]) == rec["completed_blocks"]
    # Throughput from a diverging trajectory is explicitly nulled.
    assert rec["sypd"] is None
    assert rec["mcells_per_s"] is None
    assert rec["metadata"]["extra"]["finite_ok"] is False
    assert rec["metadata"]["extra"]["valid"] is False
    assert any(p.startswith("diverged")
               for p in rec["metadata"]["_incomplete"])
    outtxt = capsys.readouterr().out
    assert "marked INVALID" in outtxt
    assert "INVALID[diverged" in outtxt


def test_main_per_step_lane_finite_unchecked_is_null(tmp_path, monkeypatch):
    """The legacy per-step lane has NO in-graph finite check: its rows must
    say so (finite_ok=null) while remaining valid — never a fabricated
    finite_ok=true."""
    import json
    mod = _load()
    out = tmp_path / "perstep.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        ["bench", "--n-devices", "1", "--n-lat", "8", "--n-lon", "8",
         "--nlev", "4", "--steps", "2", "--warmup", "1",
         "--dt", "60", "--out", str(out)])
    assert mod.main() == 0
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert rec["segment_mode"] is False
    assert rec["finite_ok"] is None
    assert rec["valid"] is True
    assert rec["completed_blocks"] is None
