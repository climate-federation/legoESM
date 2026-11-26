"""The per-component spectral-LES step profiler (scripts/bench/profile_spectral_les_step.py)."""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "scripts" / "bench"))
sys.path.insert(0, str(_REPO / "scripts" / "run"))

import profile_spectral_les_step as prof  # noqa: E402


def test_fft_flops_is_half_the_complex_count():
    assert prof.fft_flops(4, 8, 3) == 2.5 * 32 * math.log2(32) * 3


def test_trace_categories_groups_by_hlo_name_and_drops_host_frames(tmp_path):
    op = {"hlo_op": "x", "hlo_module": "m"}
    ev = [{"ph": "X", "name": "fft.1", "dur": 3000, "args": op},
          {"ph": "X", "name": "fft.22", "dur": 1000, "args": op},
          {"ph": "X", "name": "add_fusion", "dur": 500, "args": op},
          {"ph": "X", "name": "end: add_fusion", "dur": 500},
          {"ph": "X", "name": "$api.py:1 block_until_ready", "dur": 9000},
          {"ph": "X", "name": "ThunkExecutor::Execute", "dur": 9000},
          {"ph": "X", "name": "ThreadpoolListener::StartRegion", "dur": 9000},
          {"ph": "i", "name": "fft.3", "dur": 7000, "args": op}]
    d = tmp_path / "plugins" / "profile" / "run"
    d.mkdir(parents=True)
    with gzip.open(d / "perfetto_trace.json.gz", "wt") as f:
        json.dump({"traceEvents": ev}, f)
    got = prof.trace_categories(str(tmp_path), n_calls=2)
    assert got == [("fft", 2.0, 1.0), ("add_fusion", 0.25, 0.5)]


def test_profile_rejects_non_dry_convective_case():
    import pytest
    with pytest.raises(SystemExit, match="dry_convective"):
        prof.main(["--case", "sbl_gabls1", "--nx", "8", "--ny", "8", "--nz", "6"])


def test_profile_smoke_tiny_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["profile_spectral_les_step.py"])
    out = tmp_path / "p.jsonl"
    rc = prof.main(["--precision", "float64", "--nx", "8", "--ny", "8",
                    "--nz", "6", "--reps", "1", "--output", str(out)])
    assert rc == 0
    rec = json.loads(out.read_text().splitlines()[-1])
    names = [r["component"] for r in rec["components"]]
    assert names[0] == "step" and "eddy_viscosity" in names
    assert all(r["ms"] > 0 and r["bytes"] > 0 for r in rec["components"])
