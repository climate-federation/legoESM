"""Direct tests for the baroclinic-wave scaling collector (pure stdlib path)."""

from __future__ import annotations

# Import the collector module by file path (scripts/ is not an importable pkg).
import importlib.util
import json
import math
from pathlib import Path

import pytest

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


def test_collect_ingests_routeb_icosahedral_spmd_jsonl(tmp_path):
    # bench_mpas_spmd_scaling (route-B ico) emits a DIFFERENT schema than latlon:
    # grid under metadata.grid + `subdivision` (no n_lat).  collect() must label
    # it "icosahedral" with a real resolution_km -- NOT silently relabel latlon.
    d = tmp_path / "routeb_sweep_x" / "mpas_multicontroller"
    cells = 163842 * 8                                   # L7 nCells * nlev
    recs = [
        {"n_devices": n, "subdivision": 7, "nlev": 8, "physics": "none",
         "platform": "gpu", "multicontroller": True, "steady_median_ms": ms,
         "cells": cells, "metadata": {"grid": "icosahedral", "precision": "float32"}}
        for n, ms in ((6, 4.0), (3, 7.0), (1, 20.0))
    ]
    _write_jsonl(d, "mpas_spmd_scaling.jsonl", recs)
    rows, _ = agg.collect(tmp_path)
    ico = {r["n_devices"]: r for r in rows if r["grid"] == "icosahedral"}
    assert set(ico) == {1, 3, 6}                         # face-divisor ladder ingested
    assert not [r for r in rows if r["grid"] == "latlon"]  # NOT mislabelled latlon
    r6 = ico[6]
    assert r6["backend"] == "GPU" and r6["component"] == "atm"
    assert r6["resolution"] == 7 and r6["precision"] == "float32"
    assert r6["sypd"] is None                            # throughput-only
    # icosahedral L7: nCells=163842 -> R*sqrt(4pi/N) ~ 55 km nominal spacing
    assert 40.0 < r6["resolution_km"] < 75.0
    assert r6["mcells_per_s"] == pytest.approx(cells / 4.0e-3 / 1e6, rel=1e-3)


def test_spmd_latlon_without_metadata_still_latlon(tmp_path):
    # Legacy latlon SPMD records (no metadata block) must still default to
    # grid="latlon" via n_lat -- the generalization is backward-compatible.
    d = tmp_path / "routeb_sweep_x" / "atm_latlon"
    rec = {"mode": "strong", "n_devices": 8, "n_lat": 256, "n_lon": 512, "nlev": 26,
           "physics": "none", "platform": "gpu", "steady_median_ms": 5.0,
           "cells": 256 * 512 * 26}
    _write_jsonl(d, "spmd_N8.jsonl", [rec])
    rows, _ = agg.collect(tmp_path)
    r = next(r for r in rows if r["n_devices"] == 8)
    assert r["grid"] == "latlon" and r["resolution"] == 256


def test_legacy_spmd_record_with_dt_backfills_sypd(tmp_path):
    # A legacy (pre-#894) SPMD record that recorded a model `dt` (mpas/ico do)
    # but no top-level sypd must have sypd BACKFILLED via the fallback parser
    # (canonical 365.25-day formula), so a re-aggregate fills the SYPD panel for
    # existing ico runs without re-running.  latlon (no dt) stays None.
    d = tmp_path / "routeb_sweep_x" / "mpas_atm"
    cells = 163842 * 8
    rec = {"n_devices": 6, "subdivision": 7, "nlev": 8, "physics": "none",
           "platform": "gpu", "steady_median_ms": 4.0, "cells": cells, "dt": 2400.0,
           "metadata": {"grid": "icosahedral"}}
    _write_jsonl(d, "mpas_spmd_scaling.jsonl", [rec])
    rows, _ = agg.collect(tmp_path)
    r = next(r for r in rows if r["grid"] == "icosahedral")
    # sypd = (dt / t_step) / (365.25*86400) * 86400, t_step = 4.0ms = 4e-3 s
    assert r["sypd"] == pytest.approx((2400.0 / 4e-3) / (365.25 * 86400.0) * 86400.0, rel=1e-6)
    assert r["dt_seconds"] == 2400.0


def test_collect_routeb_cpu_jsonl_n_resource_in_cores(tmp_path):
    # route-B CPU sweep is 1 process per full node (platform=cpu), so n_resource
    # is expressed in CORES (n_devices * 128) -> the "CPU nodes" plot axis
    # (cores/128) then reads node counts.
    d = tmp_path / "routeb_cpu_x" / "atm_latlon_cpu"
    cells = 256 * 512 * 26
    rec = {"mode": "strong", "n_devices": 4, "n_lat": 256, "n_lon": 512, "nlev": 26,
           "physics": "none", "platform": "cpu", "steady_median_ms": 50.0, "cells": cells}
    _write_jsonl(d, "spmd_R256_N4.jsonl", [rec])
    rows, _ = agg.collect(tmp_path)
    r = next(r for r in rows if r["n_devices"] == 4)
    assert r["backend"] == "CPU"
    assert r["n_resource"] == 4 * 128 and r["n_cores"] == 4 * 128
    assert r["sypd"] is None
    assert r["mcells_per_s"] == pytest.approx(cells / 50.0e-3 / 1e6, rel=1e-3)


def test_collect_routeb_ocean_jsonl_labeled_ocean(tmp_path):
    d = tmp_path / "routeb_sweep_x" / "ocean_latlon"
    rec = {"mode": "strong", "n_devices": 8, "n_lat": 288, "n_lon": 576, "nlev": 20,
           "platform": "gpu", "steady_median_ms": 33.0, "cells": 288 * 576 * 20}
    _write_jsonl(d, "spmd_N8.jsonl", [rec])
    rows, _ = agg.collect(tmp_path)
    r = next(r for r in rows if r["n_devices"] == 8)
    assert r["component"] == "ocean" and r["case"] == "ocean"
    assert r["backend"] == "GPU" and r["sypd"] is None


def _cs_spmd_case(res, n_ranks, sypd, backend="cpu"):
    """A FLAT cube cs-spmd payload as emitted by
    run_cpu_mpi_scaling.py --cs-spmd (grid_type='cubed-sphere', device
    ladder = face divisors, resolution = face-edge N).  Measured shape:
    job on Ginsburg C24/L10/np1 -> sypd 383.7.

    Models the REAL route-B node-fill: each rung fills a 128-core node with
    THREADS=128/N per rank, so n_cores ~ 128 for EVERY rung (N in {1,2,3,6} ->
    {128,128,126,126}) while n_devices varies.  Emitting cpus_per_task/n_cores
    here (as the real run does) is load-bearing: without n_devices in the dedup
    key the near-constant n_cores collapses the 4-rung curve to ~2 points."""
    _node_cores = 128
    cpt = max(1, _node_cores // n_ranks)
    return {
        "n_ranks": n_ranks, "resolution": res, "n_levels": 10,
        "precision": "float64", "mode": "single", "backend": backend,
        "grid_type": "cubed-sphere", "physics_level": "none",
        "dt_seconds": 450.0, "time_per_step_ms": 1000.0 / sypd,
        "sypd": sypd, "total_cells": 6 * res * res * 10,
        "mcells_per_s": 10.8, "compile_time_s": 5.0,
        "cpus_per_task": cpt, "n_cores": n_ranks * cpt,
        "decomposition": f"cs-spmd np{n_ranks}",
    }


def test_ingests_cube_cs_spmd_face_ladder(tmp_path):
    """#764 item 1: the cube route-B throughput lane.  The flat cube
    cs-spmd JSON (face-divisor device ladder) must normalize into cube
    throughput rows with the right grid, distinct n_devices per rung, and
    a single resolution_km (the ladder shares one face-edge resolution,
    UNLIKE the latlon device axis) — so a cube curve can be plotted
    without being overlaid on the latlon device axis."""
    # Face-divisor ladder at a fixed face-edge resolution C48.
    for i, n in enumerate((1, 2, 3, 6)):
        d = tmp_path / f"cubed-sphere_none_single_r48_n{n}"
        _write(d, "r.json", _cs_spmd_case(48, n, 40.0 - i))
    rows, dropped = agg.collect(tmp_path)
    assert dropped == 0
    cube = [r for r in rows if r["grid"] == "cubed-sphere"]
    assert {r["n_devices"] for r in cube} == {1, 2, 3, 6}
    assert all(r["component"] == "atm" and r["case"] == "dry" for r in cube)
    # Node-fill: every rung ~fills the 128-core node, so n_cores is nearly
    # constant ({128,128,126,126}) — the 4 rungs survive ONLY because n_devices
    # is in the dedup key.  Asserting the collapse condition makes this a real
    # regression guard: revert n_devices from _key and dropped becomes 2.
    # Tolerance = the exact node-fill remainder for this ladder (128 - the
    # smallest N*(128//N)), not a magic 2, so a ladder/node-size change stays
    # honest.
    _node_cores = 128
    _fill_spread = _node_cores - min(n * (_node_cores // n) for n in (1, 2, 3, 6))
    assert max(r["n_cores"] for r in cube) - min(r["n_cores"] for r in cube) \
        <= _fill_spread
    assert len({r["n_resource"] for r in cube}) < len(cube)   # cores alone collapse
    # One shared face-edge resolution across the whole ladder.
    assert {r["resolution"] for r in cube} == {48}
    assert len({r["resolution_km"] for r in cube}) == 1
    assert all(r["resolution_km"] > 0 for r in cube)


def test_cube_and_latlon_lanes_are_distinct_curves(tmp_path):
    """The cube face-divisor ladder and the latlon lat-band ladder must
    stay SEPARABLE rows (different grid) at the same device count — they
    are different curves, not points on one device axis (#764)."""
    # SAME resolution AND device count for both grids, so ONLY `grid`
    # distinguishes them — a collector key that dropped `grid` but kept
    # (resolution, n_devices) would collapse these to one row (codex
    # round-19 Low: the prior 48-vs-96 version couldn't catch that).
    dc = tmp_path / "cubed-sphere_none_single_r48_n6"
    dl = tmp_path / "latlon_none_single_r48_n6"
    _write(dc, "c.json", _cs_spmd_case(48, 6, 35.0))
    _write(dl, "l.json", _case("latlon", "none", "single", 48, 6, "float64",
                               60.0))
    rows, dropped = agg.collect(tmp_path)
    same_dev = [r for r in rows if r["n_devices"] == 6]
    assert dropped == 0 and len(same_dev) == 2      # NOT merged into one
    assert {r["grid"] for r in same_dev} == {"cubed-sphere", "latlon"}


# ---------------------------------------------------------------------------
# SPMD bench-lane JSONL ingestion (bench_*_spmd_scaling append-per-line recs)
# ---------------------------------------------------------------------------

def _spmd_rec(**over):
    """A bench_atm_latlon_spmd_scaling-shaped flat record (one JSONL line)."""
    rec = {
        "mode": "strong", "n_devices": 4, "n_lat": 128, "n_lon": 256,
        "nlev": 30, "physics": "none", "steps": 12, "platform": "gpu",
        "steady_median_ms": 25.0,
        "grid_type": "latlon", "resolution": 128, "n_levels": 30,
        "precision": "float32", "physics_level": "none", "backend": "gpu",
        "dt_seconds": 60.0, "time_per_step_ms": 25.0,
        "total_cells": 128 * 256 * 30, "sypd": 5.67,
        "mcells_per_s": 39.3,
        "metadata": {"virtual_cpu_devices": False},
    }
    rec.update(over)
    return rec


def _write_jsonl(d: Path, name: str, recs: list) -> None:
    (d / name).parent.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_ingests_spmd_jsonl_lane(tmp_path):
    _write_jsonl(tmp_path / "latlon_gpu_1", "spmd_scaling.jsonl", [
        _spmd_rec(n_devices=1, sypd=2.0, mcells_per_s=10.0),
        _spmd_rec(n_devices=4, sypd=5.67, mcells_per_s=39.3),
    ])
    rows, dropped = agg.collect(tmp_path)
    assert dropped == 0
    assert len(rows) == 2
    by_n = {r["n_devices"]: r for r in rows}
    # n_devices (NOT n_ranks/process count) is the ladder axis: a single-
    # process 4-device SPMD run must land at n=4.
    assert set(by_n) == {1, 4}
    assert by_n[4]["backend"] == "GPU"       # from the 'backend'/'platform' key
    assert by_n[4]["sypd"] == 5.67           # SYPD now present for latlon GPU
    assert by_n[4]["mcells_per_s"] == 39.3
    assert by_n[4]["component"] == "atm"
    assert by_n[4]["grid"] == "latlon"


def test_spmd_jsonl_ocean_component_is_kept(tmp_path):
    _write_jsonl(tmp_path / "ocean_gpu_1", "ocean_spmd.jsonl", [
        _spmd_rec(component="ocean", grid_type="latlon", sypd=1.5),
    ])
    rows, _ = agg.collect(tmp_path)
    assert len(rows) == 1
    assert rows[0]["component"] == "ocean"
    assert rows[0]["case"] == "ocean"


def test_spmd_jsonl_virtual_cpu_proxy_rows_are_skipped(tmp_path):
    # Forced host-platform CPU devices = communication-overhead proxy, not
    # hardware scaling; the aggregator must never chart it as a CPU curve.
    _write_jsonl(tmp_path / "latlon_cpu_1", "spmd_scaling.jsonl", [
        _spmd_rec(platform="cpu", backend="cpu",
                  metadata={"virtual_cpu_devices": True}),
    ])
    rows, _ = agg.collect(tmp_path)
    assert rows == []


def test_spmd_jsonl_skips_ab_and_val_dirs(tmp_path):
    _write_jsonl(tmp_path / "_ab_fused" / "x", "spmd.jsonl", [_spmd_rec()])
    _write_jsonl(tmp_path / "val_smoke", "spmd.jsonl", [_spmd_rec()])
    rows, _ = agg.collect(tmp_path)
    assert rows == []


def test_spmd_jsonl_blank_and_corrupt_lines_are_skipped(tmp_path):
    p = tmp_path / "latlon_gpu_1"
    p.mkdir(parents=True)
    (p / "spmd.jsonl").write_text(
        json.dumps(_spmd_rec()) + "\n\nnot-json{{{\n")
    rows, _ = agg.collect(tmp_path)
    assert len(rows) == 1


# ---- schema-v2 provenance survives into the tidy CSV -------------------------
# Regression: `transport` / `decomposition` are REQUIRED keys in every source
# record (metadata schema v2) so a route-A mpi4jax row can be told from a
# route-B NCCL row "from the record alone" -- but the aggregator dropped them,
# so the tidy CSV everyone plots from could not answer "which route was this?".
# A 2-D-pencil route-A latlon ladder consequently sat in a directory named
# `routeb_cpu_latlon` and the route had to be inferred from filenames.

def test_provenance_fields_are_in_the_tidy_schema():
    for k in ("transport", "decomposition", "schema_version",
              "gpu_direct_active", "host_staged_halo", "launcher"):
        assert k in agg.FIELDS, f"{k} missing from the tidy CSV schema"


def test_provenance_read_from_nested_metadata(tmp_path):
    """The flat run_cpu_mpi_scaling JSONs nest the audit fields under
    ``metadata`` -- the aggregator must look there, not only at top level."""
    d = tmp_path / "dry_latlon_cpu_np128_1"
    payload = _case("latlon", "none", "strong", 192, 128, "float32", 0.56)
    payload["metadata"] = {
        "schema_version": 2, "transport": "mpi4jax",
        "decomposition": "latlon_2d_pencil", "gpu_direct_active": False,
        "host_staged_halo": True, "launcher": "mpiexec",
    }
    _write(d, "a.json", payload)
    rows, _ = agg.collect(tmp_path)
    assert len(rows) == 1
    assert rows[0]["transport"] == "mpi4jax"          # route A, not route B
    assert rows[0]["decomposition"] == "latlon_2d_pencil"
    assert rows[0]["launcher"] == "mpiexec"


def test_provenance_read_from_top_level(tmp_path):
    """Records that carry the audit fields at top level work too."""
    d = tmp_path / "dry_latlon_gpu_g16_1"
    payload = _case("latlon", "none", "strong", 1024, 16, "float32", 29.8)
    payload.update(transport="nccl", decomposition="latlon_1d_band",
                   schema_version=2)
    _write(d, "b.json", payload)
    rows, _ = agg.collect(tmp_path)
    assert rows[0]["transport"] == "nccl"             # route B
    assert rows[0]["decomposition"] == "latlon_1d_band"


def test_legacy_record_gets_empty_not_guessed_transport(tmp_path):
    """A pre-v2 record must yield EMPTY provenance, never a directory-name
    guess -- inferring 'routeb_cpu_latlon' -> route B is the exact error this
    change retires."""
    d = tmp_path / "routeb_cpu_latlon"
    _write(d, "c.json", _case("latlon", "none", "strong", 192, 128, "float32", 0.56))
    rows, _ = agg.collect(tmp_path)
    assert rows[0]["transport"] == ""
    assert rows[0]["decomposition"] == ""
