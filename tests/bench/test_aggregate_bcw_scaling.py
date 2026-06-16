"""Direct tests for the baroclinic-wave scaling collector (pure stdlib path)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

# Import the collector module by file path (scripts/ is not an importable pkg).
import importlib.util

_AGG = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "aggregate_bcw_scaling.py"
_spec = importlib.util.spec_from_file_location("aggregate_bcw_scaling", _AGG)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def _write(d: Path, name: str, payload: dict) -> None:
    (d / name).parent.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(payload))


def _case(grid, phys, mode, res, n, prec, sypd):
    return {
        "n_ranks": n, "resolution": res, "n_levels": 26, "precision": prec,
        "mode": mode, "grid_type": grid, "physics_level": phys,
        "dt_seconds": 390.0, "time_per_step_ms": 1000.0 / sypd,
        "sypd": sypd, "total_cells": 12345, "mcells_per_s": 1.0,
        "scaling_efficiency": 1.0, "compile_time_s": 2.0,
    }


def test_collect_backend_case_and_km(tmp_path):
    cpu = tmp_path / "dry_icosahedral_cpu_np16_1"
    gpu = tmp_path / "dry_icosahedral_gpu_g4_2"
    moi = tmp_path / "dry_icosahedral_cpu_np16_3"
    _write(cpu, "a.json", _case("icosahedral", "none", "strong", 5, 16, "float64", 29.4))
    _write(gpu, "b.json", _case("icosahedral", "none", "strong", 5, 4, "float32", 88.0))
    _write(moi, "c.json", _case("icosahedral", "moist", "strong", 5, 16, "float64", 20.0))
    rows, dropped = agg.collect(tmp_path)
    assert dropped == 0
    by = {(r["backend"], r["case"], r["precision"]): r for r in rows}
    assert by[("CPU", "dry", "float64")]["n_devices"] == 16
    assert by[("GPU", "dry", "float32")]["n_devices"] == 4
    assert ("CPU", "moist", "float64") in by
    # icosahedral L5: nCells=10242 -> R*sqrt(4pi/N) ~ 223 km nominal spacing
    # (L6 ~ 111 km, L7 ~ 56 km).
    km = by[("CPU", "dry", "float64")]["resolution_km"]
    assert 180.0 < km < 280.0


def test_dedup_keeps_max_sypd(tmp_path):
    d1 = tmp_path / "dry_icosahedral_cpu_np16_1"
    d2 = tmp_path / "dry_icosahedral_cpu_np16_2"
    _write(d1, "a.json", _case("icosahedral", "none", "strong", 5, 16, "float64", 25.0))
    _write(d2, "a.json", _case("icosahedral", "none", "strong", 5, 16, "float64", 31.0))
    rows, dropped = agg.collect(tmp_path)
    assert dropped == 1
    assert len(rows) == 1
    assert rows[0]["sypd"] == 31.0


def test_resolution_km_families():
    # Finer resolution => smaller km, monotone within each family.
    assert agg.resolution_km("icosahedral", 6) < agg.resolution_km("icosahedral", 5)
    assert agg.resolution_km("latlon", 128) < agg.resolution_km("latlon", 64)
    assert agg.resolution_km("cubed-sphere", 96) < agg.resolution_km("cubed-sphere", 48)
    assert math.isnan(agg.resolution_km("mystery", 1))


def test_backend_from_json_overrides_path(tmp_path):
    # Dir name has no _cpu_/_gpu_ token; the JSON's recorded backend wins.
    d = tmp_path / "val_moist_999"
    payload = _case("icosahedral", "moist", "strong", 5, 4, "float64", 50.0)
    payload["backend"] = "gpu"
    _write(d, "x.json", payload)
    rows, _ = agg.collect(tmp_path)
    assert len(rows) == 1 and rows[0]["backend"] == "GPU"


def test_skips_non_case_json(tmp_path):
    d = tmp_path / "dry_icosahedral_cpu_np16_1"
    _write(d, "notacase.json", {"hello": "world"})
    _write(d, "case.json", _case("icosahedral", "none", "strong", 5, 16, "float64", 29.4))
    rows, _ = agg.collect(tmp_path)
    assert len(rows) == 1
