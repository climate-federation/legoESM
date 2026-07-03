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
    # Dir name has no _cpu_/_gpu_ token (and is not a val_ dir); the JSON's
    # recorded backend wins.
    d = tmp_path / "moist_ico_999"
    payload = _case("icosahedral", "moist", "strong", 5, 4, "float64", 50.0)
    payload["backend"] = "gpu"
    _write(d, "x.json", payload)
    rows, _ = agg.collect(tmp_path)
    assert len(rows) == 1 and rows[0]["backend"] == "GPU"


def test_ingests_nested_ocean_schema(tmp_path):
    d = tmp_path / "scaling_cpu_ocean"
    d.mkdir()
    payload = {
        "backend": "CPU", "mode": "ocean_strong",
        "results": [{
            "n_ranks": 16, "resolution": 192, "precision": "float64",
            "mode": "ocean_strong", "sypd": 24.3, "mcells_per_s": 21.8,
            "total_cells": 1474560, "time_per_step_ms": 67.4,
        }],
    }
    (d / "ocean_strong_r192_np16_float64.json").write_text(json.dumps(payload))
    rows, _ = agg.collect(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["component"] == "ocean" and r["case"] == "ocean"
    assert r["backend"] == "CPU" and r["n_devices"] == 16 and r["mode"] == "strong"


def test_ingests_nested_atm_report(tmp_path):
    # run_levante_gpu_scaling.py writes a nested report with mode 'strong'
    # (NOT 'ocean_*'); each result carries grid_type + n_gpus. It must be
    # flattened as component='atm' with the right grid, not dropped.
    d = tmp_path / "scaling" / "20260624T2252Z"
    payload = {
        "backend": "GPU", "mode": "strong", "precisions": ["float32"],
        "results": [{
            "n_gpus": 2, "resolution": 96, "n_levels": 26, "precision": "float32",
            "mode": "strong", "physics_level": "none", "grid_type": "cubed-sphere",
            "sypd": 12.5, "mcells_per_s": 240.0, "total_cells": 1492992,
            "time_per_step_ms": 6.8,
        }],
    }
    _write(d, "strong_scaling.json", payload)
    rows, _ = agg.collect(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["component"] == "atm" and r["grid"] == "cubed-sphere"
    assert r["backend"] == "GPU" and r["n_devices"] == 2 and r["n_resource"] == 2
    assert r["case"] == "dry" and r["mode"] == "strong" and r["sypd"] == 12.5


def test_multi_root_collect(tmp_path):
    a = tmp_path / "bcw_scaling" / "dry_icosahedral_cpu_np16_1"
    o = tmp_path / "scaling_cpu_ocean"
    a.mkdir(parents=True)
    o.mkdir(parents=True)
    _write(a, "x.json", _case("icosahedral", "none", "strong", 5, 16, "float64", 29.4))
    (o / "y.json").write_text(json.dumps({
        "backend": "CPU", "results": [{
            "n_ranks": 8, "resolution": 192, "precision": "float64",
            "mode": "ocean_strong", "sypd": 30.0}]}))
    rows, _ = agg.collect([a.parent, o])
    comps = {r["component"] for r in rows}
    assert comps == {"atm", "ocean"}


def test_skips_validation_dirs(tmp_path):
    prod = tmp_path / "dry_icosahedral_cpu_np16_1"
    val = tmp_path / "val_moist_999"
    _write(prod, "a.json", _case("icosahedral", "none", "strong", 5, 16, "float64", 29.4))
    _write(val, "b.json", _case("icosahedral", "moist", "strong", 4, 1, "float64", 80.0))
    rows, _ = agg.collect(tmp_path)
    assert len(rows) == 1
    assert rows[0]["case"] == "dry"  # the val_ moist probe is excluded


def test_skips_non_case_json(tmp_path):
    d = tmp_path / "dry_icosahedral_cpu_np16_1"
    _write(d, "notacase.json", {"hello": "world"})
    _write(d, "case.json", _case("icosahedral", "none", "strong", 5, 16, "float64", 29.4))
    rows, _ = agg.collect(tmp_path)
    assert len(rows) == 1


def test_unknown_nested_schema_is_skipped_not_atm(tmp_path):
    # A nested {results:[...]} report that is NEITHER ocean (mode 'ocean_*') NOR
    # the atm harness (no grid_type / physics_level markers) must be SKIPPED, not
    # silently mislabeled component='atm' (codex review of the else->atm default).
    d = tmp_path / "scaling" / "mystery"
    payload = {
        "backend": "GPU", "mode": "strong",
        "results": [{"n_gpus": 2, "resolution": 96, "precision": "float32",
                     "mode": "strong", "sypd": 9.0, "mcells_per_s": 100.0,
                     "total_cells": 1000}],   # NO grid_type, NO physics_level
    }
    _write(d, "strong_scaling.json", payload)
    rows, _ = agg.collect(tmp_path)
    assert rows == []                          # skipped, not flattened as atm


def _write_jsonl(d: Path, name: str, recs: list[dict]) -> None:
    (d / name).parent.mkdir(parents=True, exist_ok=True)
    with (d / name).open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_collect_ingests_routeb_spmd_jsonl(tmp_path):
    # route-B SPMD bench output is JSONL and THROUGHPUT-only (steady_median_ms +
    # cells, no sypd); collect() must ingest it and compute mcells_per_s.
    d = tmp_path / "routeb_sweep_x" / "atm_latlon"
    cells = 512 * 1024 * 26
    recs = [
        {"mode": "strong", "n_devices": 8, "n_lat": 512, "n_lon": 1024, "nlev": 26,
         "physics": "moist", "platform": "gpu", "multicontroller": True,
         "compile_ms": 10000.0, "steady_median_ms": 5.0, "cells": cells},
        {"mode": "strong", "n_devices": 4, "n_lat": 512, "n_lon": 1024, "nlev": 26,
         "physics": "moist", "platform": "gpu", "multicontroller": True,
         "compile_ms": 9000.0, "steady_median_ms": 9.0, "cells": cells},
    ]
    _write_jsonl(d, "spmd_N8.jsonl", recs)
    rows, _ = agg.collect(tmp_path)
    latlon = {r["n_devices"]: r for r in rows if r["grid"] == "latlon"}
    assert set(latlon) == {4, 8}                       # both swept points ingested
    r8 = latlon[8]
    assert r8["backend"] == "GPU" and r8["resolution"] == 512
    assert r8["sypd"] is None                          # throughput-only
    assert r8["case"] == "moist"
    assert r8["mcells_per_s"] == pytest.approx(cells / 5.0e-3 / 1e6, rel=1e-3)


def test_collect_routeb_ocean_jsonl_labeled_ocean(tmp_path):
    d = tmp_path / "routeb_sweep_x" / "ocean_latlon"
    rec = {"mode": "strong", "n_devices": 8, "n_lat": 288, "n_lon": 576, "nlev": 20,
           "platform": "gpu", "steady_median_ms": 33.0, "cells": 288 * 576 * 20}
    _write_jsonl(d, "spmd_N8.jsonl", [rec])
    rows, _ = agg.collect(tmp_path)
    r = next(r for r in rows if r["n_devices"] == 8)
    assert r["component"] == "ocean" and r["case"] == "ocean"
    assert r["backend"] == "GPU" and r["sypd"] is None
