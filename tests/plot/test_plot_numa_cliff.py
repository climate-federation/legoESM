"""Test the NUMA-cliff comparison plotter (packed vs hybrid split)."""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
_P = Path(__file__).resolve().parents[2] / "scripts" / "plot" / "plot_numa_cliff.py"
spec = importlib.util.spec_from_file_location("plot_numa_cliff", _P)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def _w(d, name, **kw):
    (d / name).parent.mkdir(parents=True, exist_ok=True)
    base = {"grid_type": "icosahedral", "resolution": 5, "precision": "float64",
            "mode": "strong", "sypd": 10.0}
    base.update(kw); (d / name).write_text(json.dumps(base))

def test_collect_splits_packed_vs_hybrid(tmp_path):
    _w(tmp_path / "a", "p32.json", n_ranks=32, cpus_per_task=1, n_cores=32, sypd=14.0)
    _w(tmp_path / "b", "h32.json", n_ranks=8, cpus_per_task=4, n_cores=32, sypd=34.0)
    _w(tmp_path / "c", "h64.json", n_ranks=16, cpus_per_task=4, n_cores=64, sypd=55.0)
    by = mod.collect(tmp_path, "icosahedral", 5, "float64", "strong")
    assert set(by) == {1, 4}
    assert by[1][32] == 14.0 and by[4][32] == 34.0 and by[4][64] == 55.0

def test_excludes_ab_and_val(tmp_path):
    _w(tmp_path / "_ab_x", "p.json", n_ranks=32, cpus_per_task=1, n_cores=32)
    _w(tmp_path / "val_y", "p.json", n_ranks=32, cpus_per_task=1, n_cores=32)
    by = mod.collect(tmp_path, "icosahedral", 5, "float64", "strong")
    assert by == {} or all(not v for v in by.values())

def test_main_writes_png(tmp_path):
    _w(tmp_path / "a", "p.json", n_ranks=32, cpus_per_task=1, n_cores=32, sypd=14.0)
    _w(tmp_path / "b", "h.json", n_ranks=8, cpus_per_task=4, n_cores=32, sypd=34.0)
    import sys
    out = tmp_path / "cliff.png"
    sys.argv = ["x", "--root", str(tmp_path), "--out", str(out)]
    assert mod.main() == 0 and out.exists() and out.stat().st_size > 1000
