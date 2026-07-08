"""Direct tests for the spectral-transform microbenchmark (audit item 9)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BENCH = (Path(__file__).resolve().parents[2]
          / "scripts" / "bench" / "bench_spectral_transform_micro.py")
_spec = importlib.util.spec_from_file_location("bench_sh_micro", _BENCH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_cost_model_scaling():
    a = mod.transform_flops(64, 128, 946, 30)
    b = mod.transform_flops(128, 256, 3741, 30)
    # EXACT constants locked (codex: monotonicity alone lets a
    # denominator drift pass): 8 real FLOPs per complex MAC, analysis +
    # synthesis = 2 * n_lat * n_sh * nlev MACs.
    assert a["gemm_flops"] == 4 * 2 * 64 * 946 * 30  # real x complex MAC
    assert a["fft_flops"] == 2 * 30 * 64 * 5 * 128 * 7  # log2(128)=7
    # GEMM grows ~ n_lat*n_sh; fraction of FLOPs grows with truncation.
    assert b["gemm_flops"] > 4 * a["gemm_flops"]
    assert b["gemm_fraction_of_flops"] > a["gemm_fraction_of_flops"]
    assert 0.0 < a["gemm_fraction_of_flops"] < 1.0
    assert a["arithmetic_intensity_flop_per_byte"] > 1.0


def test_main_rejects_bad_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "--truncations", "5"])
    with pytest.raises(SystemExit):
        mod.main()
    monkeypatch.setattr(sys, "argv", ["bench", "--repeats", "0"])
    with pytest.raises(SystemExit):
        mod.main()


def test_main_t21_smoke_round_trip_exact(tmp_path, monkeypatch):
    out = tmp_path / "m.json"
    monkeypatch.setattr(sys, "argv", [
        "bench", "--truncations", "21", "--nlev", "4",
        "--repeats", "2", "--warmup", "1", "--out", str(out)])
    assert mod.main() == 0
    payload = json.loads(out.read_text())
    rows = payload["rows"]
    # --sh-gemm both (default): one row per Legendre path, mode recorded
    # (a legacy row can never masquerade as the GEMM measurement; codex).
    assert {r["legendre_path"] for r in rows} == {
        "legacy_segment_sum", "gemm"}
    for row in rows:
        assert row["truncation"] == 21
        # Band-limited input: the recorded error is the TRANSFORM's own
        # (projection applied before timing), so it must sit at the f64
        # round-off scale, not the truncation scale.
        assert row["band_limited_round_trip_error_max"] < 1e-9
        assert row["round_trip_median_ms"] > 0
    md = payload["metadata"]
    assert md["grid"] == "spectral"
    assert md["scaling_kind"] == "throughput"
    assert md["extra"]["sh_gemm_modes"] == "both"
    assert "_incomplete" not in md
