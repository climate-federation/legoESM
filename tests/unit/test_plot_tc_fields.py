"""Smoke test for ``scripts/plot/plot_tc_fields.py``.

Synthesize a few tiny ``sfc_*.npz`` snapshots with the keys the montage reader
expects, run ``main()`` end-to-end, and assert the PNG is written. Also covers
the empty-directory guard (``SystemExit``). Exercises the snapshot glob, the
linspace/unique frame pick, the level ``argmin``, the per-row field stacking,
and the imshow/quiver/colorbar/savefig path.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]  # tests/unit/<file> -> repo root


def _load_main():
    path = REPO / "scripts" / "plot" / "plot_tc_fields.py"
    spec = importlib.util.spec_from_file_location("plot_tc_fields", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plot_tc_fields"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_snapshot(path: Path, day: float, nlev: int = 6, ny: int = 8, nx: int = 8) -> None:
    rng = np.random.default_rng(int(day))
    np.savez(
        path,
        heights=np.linspace(0.0, 15_000.0, nlev),
        Lx=512_000.0,
        dx=64_000.0,
        day=float(day),
        u_sfc=rng.standard_normal((ny, nx)),
        v_sfc=rng.standard_normal((ny, nx)),
        T_levels=rng.standard_normal((nlev, ny, nx)) + 290.0,
        cwv=rng.uniform(20.0, 60.0, (ny, nx)),
        precip=rng.uniform(0.0, 50.0, (ny, nx)),
    )


def test_plot_tc_fields_writes_png(tmp_path, monkeypatch):
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i, day in enumerate((60.0, 61.0, 62.0)):
        _write_snapshot(snaps / f"sfc_{i:03d}.npz", day)
    mod = _load_main()
    monkeypatch.setattr(
        sys, "argv",
        ["plot_tc_fields.py", "--run", str(tmp_path), "--times", "2", "--t0-day", "60.0"],
    )
    mod.main()
    assert (tmp_path / "tc_fields.png").stat().st_size > 0


def test_plot_tc_fields_no_snapshots_raises(tmp_path, monkeypatch):
    (tmp_path / "snapshots").mkdir()
    mod = _load_main()
    monkeypatch.setattr(sys, "argv", ["plot_tc_fields.py", "--run", str(tmp_path)])
    with pytest.raises(SystemExit):
        mod.main()
