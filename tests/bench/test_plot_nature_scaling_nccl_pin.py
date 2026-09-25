"""The scaling figure keeps only MPAS GPU rows at their lane's NCCL channel
pin: 64 for the atmosphere (since 2026-09-25), 32 for the ocean rows."""
import importlib.util
import json
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "scripts" / "plot" / "plot_nature_scaling.py"
_spec = importlib.util.spec_from_file_location("plot_nature_scaling", _P)
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)


def _row(d: Path, name, comp, grid, ch, ms=5.0):
    rec = {"component": comp, "grid_type": grid, "platform": "gpu", "precision": "float32",
           "subdivision": 9, "nlev": 40, "n_devices": 128, "steady_median_ms": ms,
           "valid": True,
           "metadata": {"extra": {"nccl_env": {"NCCL_MIN_NCHANNELS": ch, "NCCL_MAX_NCHANNELS": ch,
                                               "NCCL_P2P_NET_CHUNKSIZE": "131072"},
                                  "pcg_fixed_iters": 20, "pcg_precond": "poly",
                                  "pcg_poly_sweeps": 4}}}
    (d / f"{name}.jsonl").write_text(json.dumps(rec) + "\n")


def test_atmosphere_pin_is_64_ocean_stays_32(tmp_path):
    plot.NLEV = plot._nlev_map(40)
    _row(tmp_path, "atm64", "mpas_atm", "icosahedral", "64", 3.0)
    _row(tmp_path, "atm32", "mpas_atm", "icosahedral", "32", 4.0)
    _row(tmp_path, "oce32", "mpas_ocean", "mpas", "32", 30.0)
    _row(tmp_path, "oce64", "mpas_ocean", "mpas", "64", 20.0)
    best = plot.load([str(tmp_path)])
    by_grid = {k[1]: v[0] for k, v in best.items()}
    assert by_grid == {"icosahedral": 3.0, "mpas": 30.0}
