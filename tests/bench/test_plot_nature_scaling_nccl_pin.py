"""The scaling figure keeps only MPAS GPU rows at their lane's NCCL channel
pin: 64 for the atmosphere and 8 for the ocean (both since 2026-09-25)."""
import importlib.util
import json
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "scripts" / "plot" / "plot_nature_scaling.py"
_spec = importlib.util.spec_from_file_location("plot_nature_scaling", _P)
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)


def _row(d: Path, name, comp, grid, ch, ms=5.0, variant="standard", solver=None,
         precond="gpoly"):
    rec = {"component": comp, "grid_type": grid, "platform": "gpu", "precision": "float32",
           "subdivision": 9, "nlev": 40, "n_devices": 128, "steady_median_ms": ms,
           "valid": True, "finite_ok": True,
           "metadata": {"extra": {"nccl_env": {"NCCL_MIN_NCHANNELS": ch, "NCCL_MAX_NCHANNELS": ch,
                                               "NCCL_P2P_NET_CHUNKSIZE": "131072"},
                                  "pcg_fixed_iters": 20, "pcg_precond": precond,
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


def test_block_local_poly_ocean_rows_are_refused(tmp_path):
    """The figure plots the production global preconditioner (gpoly4)."""
    plot.NLEV = plot._nlev_map(40)
    _row(tmp_path, "gpoly", "mpas_ocean", "mpas", "8", 30.0)
    _row(tmp_path, "poly", "mpas_ocean", "mpas", "8", 20.0, precond="poly")
    assert [v[0] for v in plot.load([str(tmp_path)]).values()] == [30.0]


def _cube(d: Path, name, ms, finite_ok=None):
    rec = {"component": "atmosphere", "grid_type": "cubed-sphere", "mode": "strong", "kt": 4,
           "n_devices": 96, "resolution": 768, "nlev": 40, "platform": "gpu",
           "precision": "float32", "steady_median_ms": ms}
    if finite_ok is not None:
        rec["finite_ok"] = finite_ok
    (d / f"{name}.jsonl").write_text(json.dumps(rec) + "\n")


def test_unchecked_cube_rows_plot_but_unchecked_mpas_atm_rows_do_not(tmp_path):
    plot.NLEV = plot._nlev_map(40)
    _cube(tmp_path, "cube", 29.0)
    _row(tmp_path, "atm", "mpas_atm", "icosahedral", "64", 3.0)
    rec = json.loads((tmp_path / "atm.jsonl").read_text())
    del rec["finite_ok"]
    (tmp_path / "atm.jsonl").write_text(json.dumps(rec) + "\n")
    assert {k[1]: v[0] for k, v in plot.load([str(tmp_path)]).items()} == {"cubed-sphere": 29.0}


def test_cube_row_that_failed_finite_check_is_refused(tmp_path):
    plot.NLEV = plot._nlev_map(40)
    _cube(tmp_path, "cube", 29.0, finite_ok=False)
    assert plot.load([str(tmp_path)]) == {}


def test_unchecked_flag_follows_the_selected_row(tmp_path):
    """A faster checked cube row wins its key and clears the unchecked flag."""
    plot.NLEV = plot._nlev_map(40)
    _cube(tmp_path, "slow_unchecked", 30.0)
    _cube(tmp_path, "fast_checked", 29.0, finite_ok=True)
    best = plot.load([str(tmp_path)])
    assert [v[0] for v in best.values()] == [29.0] and not plot.UNCHECKED_KEYS
    (tmp_path / "fast_checked.jsonl").unlink()
    assert len(plot.load([str(tmp_path)])) == 1 and len(plot.UNCHECKED_KEYS) == 1
