"""The scaling figure keeps only MPAS GPU rows at their lane's NCCL channel
pin: 64 for the atmosphere and 8 for the ocean (both since 2026-09-25)."""
import importlib.util
import json
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "scripts" / "plot" / "plot_nature_scaling.py"
_spec = importlib.util.spec_from_file_location("plot_nature_scaling", _P)
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)


def _row(d: Path, name, comp, grid, ch, ms=5.0, variant="standard", solver=None):
    rec = {"component": comp, "grid_type": grid, "platform": "gpu", "precision": "float32",
           "subdivision": 9, "nlev": 40, "n_devices": 128, "steady_median_ms": ms,
           "valid": True,
           "metadata": {"extra": {"nccl_env": {"NCCL_MIN_NCHANNELS": ch, "NCCL_MAX_NCHANNELS": ch,
                                               "NCCL_P2P_NET_CHUNKSIZE": "131072"},
                                  "pcg_fixed_iters": 20, "pcg_precond": "poly",
                                  "pcg_poly_sweeps": 4}}}
    x = rec["metadata"]["extra"]
    if variant is not None:
        x["pcg_variant"] = variant
    if solver is not None:
        x["pcg_solver_path"] = solver
    (d / f"{name}.jsonl").write_text(json.dumps(rec) + "\n")


def test_atmosphere_pin_is_64_ocean_is_8(tmp_path):
    plot.NLEV = plot._nlev_map(40)
    _row(tmp_path, "atm64", "mpas_atm", "icosahedral", "64", 3.0)
    _row(tmp_path, "atm32", "mpas_atm", "icosahedral", "32", 4.0)
    _row(tmp_path, "oce8", "mpas_ocean", "mpas", "8", 30.0)
    _row(tmp_path, "oce32", "mpas_ocean", "mpas", "32", 20.0)
    best = plot.load([str(tmp_path)])
    by_grid = {k[1]: v[0] for k, v in best.items()}
    assert by_grid == {"icosahedral": 3.0, "mpas": 30.0}


def test_single_reduce_ocean_rows_are_refused(tmp_path):
    plot.NLEV = plot._nlev_map(40)
    _row(tmp_path, "std", "mpas_ocean", "mpas", "8", 37.0)
    _row(tmp_path, "sr", "mpas_ocean", "mpas", "8", 35.0, variant="single_reduce")
    _row(tmp_path, "sr_stock", "mpas_ocean", "mpas", "8", 34.0, variant="single_reduce",
         solver="stock_cg_to_tol")
    best = plot.load([str(tmp_path)])
    assert [v[0] for v in best.values()] == [37.0]


def test_ocean_row_without_variant_field_is_kept(tmp_path):
    """Receipts older than the pcg_variant stamp ran the standard recurrence."""
    plot.NLEV = plot._nlev_map(40)
    _row(tmp_path, "legacy", "mpas_ocean", "mpas", "8", 37.0, variant=None)
    assert [v[0] for v in plot.load([str(tmp_path)]).values()] == [37.0]
