"""Direct tests for the BCW scaling-vs-iteration ledger."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

_LED = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "bcw_scaling_ledger.py"
_spec = importlib.util.spec_from_file_location("bcw_scaling_ledger", _LED)
led = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(led)

_FIELDS = ["backend", "grid", "case", "precision", "mode", "n_devices",
           "resolution", "resolution_km", "n_levels", "sypd", "time_per_step_ms",
           "total_cells", "mcells_per_s", "scaling_efficiency", "dt_seconds",
           "physics_level", "compile_time_s", "source"]


def _row(backend, n, sypd, mc, res=6, prec="float64", case="dry"):
    d = {f: "" for f in _FIELDS}
    d.update(backend=backend, grid="icosahedral", case=case, precision=prec,
             mode="strong", n_devices=n, resolution=res, sypd=sypd,
             mcells_per_s=mc)
    return d


def _write_tidy(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_series_metrics_strong_efficiency():
    # Perfect doubling => efficiency 1.0; here SYPD goes 10->18 over 1->2 dev.
    rows = [_row("GPU", 1, 10.0, 50.0), _row("GPU", 2, 18.0, 90.0)]
    m = led.series_metrics(rows)
    assert m["max_ndev"] == 2
    assert abs(m["strong_eff"] - 0.9) < 1e-9   # (18/10)/(2/1)=0.9
    assert m["peak_sypd"] == 18.0
    assert m["peak_mcells_per_s"] == 90.0


def test_snapshot_increments_iteration(tmp_path):
    tidy = tmp_path / "tidy.csv"
    ledger = tmp_path / "ledger.csv"
    _write_tidy(tidy, [_row("GPU", 1, 10.0, 50.0), _row("GPU", 2, 18.0, 90.0)])
    it1 = led.snapshot(tidy, ledger, note="iter1", timestamp="2026-06-16T00:00:00Z")
    it2 = led.snapshot(tidy, ledger, note="iter2", timestamp="2026-06-16T01:00:00Z")
    assert it1 == 1 and it2 == 2
    with ledger.open() as f:
        recs = list(csv.DictReader(f))
    assert {int(r["iteration"]) for r in recs} == {1, 2}
    assert all(r["commit"] is not None for r in recs)


def test_plot_creates_png(tmp_path):
    ledger = tmp_path / "ledger.csv"
    tidy = tmp_path / "tidy.csv"
    _write_tidy(tidy, [_row("GPU", 1, 10.0, 50.0), _row("GPU", 2, 18.0, 90.0)])
    led.snapshot(tidy, ledger, note="a", timestamp="t1")
    led.snapshot(tidy, ledger, note="b", timestamp="t2")
    out = tmp_path / "iter.png"
    res = led.plot(ledger, out, grid="icosahedral")
    assert res is not None and out.exists() and out.stat().st_size > 1000
