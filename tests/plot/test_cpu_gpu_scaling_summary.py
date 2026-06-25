"""Direct test for scripts/plot/plot_cpu_gpu_scaling_summary.py — the
consolidated CPU+GPU weak+strong atm+ocean efficiency plotter.

Exercises the GPU-ledger parser (``gpu_points``: full-step vs barotropic-PCG
kernel classification, grid->component map, weak/strong, non-eff rows ignored,
precision-from-metric) and that ``make_summary_figure`` renders with both the
CPU tidy curves and the GPU 2-device points.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = Path(__file__).resolve().parents[2] / "scripts" / "plot" / \
    "plot_cpu_gpu_scaling_summary.py"
_m = importlib.util.spec_from_file_location("plot_cpu_gpu_scaling_summary", _SPEC)
plot_mod = importlib.util.module_from_spec(_m)
_m.loader.exec_module(plot_mod)


def _mk(grid, mode, value, metric, job="9"):
    return {"grid": grid, "mode": mode, "value": value, "metric": metric, "job": job}


def _indicator_rows():
    """A slice mirroring the real scaling_indicators.csv GPU 2-device rows."""
    mk = _mk
    return [
        mk("atm_latlon", "strong", "0.821", "eff_2gpu_atm_latlon", "8486204"),
        mk("atm_latlon", "weak", "0.552", "eff_2gpu_atm_latlon_weak", "8487754"),
        mk("atm_icosahedral", "strong", "0.829", "eff_2gpu_atm_icosahedral"),
        mk("atm_cube", "strong", "0.73", "eff_2gpu_f64"),
        mk("atm_cube", "strong", "0.46", "eff_2gpu_f32"),
        mk("ocean_latlon", "strong", "0.920", "eff_2gpu_ocean_fullstep_360n60"),
        mk("ocean_latlon", "weak", "0.570", "eff_2gpu_ocean_weak"),
        mk("ocean_latlon", "strong", "0.4280",
           "ocean_spmd_pcg_single_reduce_eff_ndev2", "8486096"),
        # rows that MUST be ignored (not 2-device efficiency):
        mk("atm_cube", "strong", "12", "ops3d_np24"),
        mk("ocean_latlon", "strong", "1.0000",
           "ocean_spmd_pcg_standard_eff_ndev1"),   # ndev1 baseline, not ndev2
    ]


def test_gpu_points_classifies_fullstep_vs_kernel():
    g = plot_mod.gpu_points(_indicator_rows())
    atm_strong = g[("atm", "strong")]
    # latlon/ico/cube full-step points present, all kind=fullstep
    grids = {p["grid"] for p in atm_strong}
    assert {"latlon", "icosahedral", "cubed-sphere"} <= grids
    assert all(p["kind"] == "fullstep" for p in atm_strong)
    ocean_strong = g[("ocean", "strong")]
    kinds = {p["kind"] for p in ocean_strong}
    assert "fullstep" in kinds and "kernel" in kinds       # both ocean rows kept
    kern = [p for p in ocean_strong if p["kind"] == "kernel"][0]
    assert kern["eff"] == 0.428
    assert "kernel" in kern["label"]


def test_gpu_points_precision_and_mode():
    g = plot_mod.gpu_points(_indicator_rows())
    cube = [p for p in g[("atm", "strong")] if p["grid"] == "cubed-sphere"]
    precs = {p["precision"] for p in cube}
    assert precs == {"float64", "float32"}                 # both cube prec rows
    assert ("atm", "weak") in g and ("ocean", "weak") in g  # weak mode mapped


def test_gpu_points_ignores_non_2dev_rows():
    g = plot_mod.gpu_points(_indicator_rows())
    # ops3d_np24 (capability) + ndev1 baseline must NOT appear as points
    for pts in g.values():
        assert all(p["eff"] != 12 for p in pts)
        assert all(p["eff"] != 1.0 for p in pts)


def _tidy_rows():
    out = []
    data = {
        ("atm", "latlon", "float64", "strong"): {1: 10, 2: 14, 4: 21, 8: 26},
        ("atm", "latlon", "float64", "weak"): {1: 10, 2: 12, 4: 15, 8: 17},
        ("ocean", "latlon", "float64", "strong"): {8: 11, 16: 19, 32: 26},
        ("ocean", "latlon", "float64", "weak_band"): {8: 9, 16: 15},
    }
    for (comp, grid, prec, mode), d in data.items():
        for n, mc in d.items():
            out.append({"component": comp, "grid": grid, "precision": prec,
                        "mode": mode, "backend": "CPU",
                        "n_devices": str(n), "mcells_per_s": str(mc)})
    return out


def test_make_summary_figure_writes_png(tmp_path):
    out = plot_mod.make_summary_figure(_tidy_rows(), _indicator_rows(), tmp_path)
    assert out.exists() and out.stat().st_size > 0
    assert out.name == "cpu_gpu_scaling_summary.png"


def test_make_summary_figure_empty(tmp_path):
    out = plot_mod.make_summary_figure([], [], tmp_path)
    assert out.exists()                                    # "no data" panels still render


def test_make_summary_figure_gpu_only(tmp_path):
    """CPU tidy missing but GPU ledger present -> still renders the GPU points."""
    out = plot_mod.make_summary_figure([], _indicator_rows(), tmp_path)
    assert out.exists() and out.stat().st_size > 0


# --- negative / hardening cases (codex review of the silent-default class) ---

def test_gpu_points_skips_unknown_or_blank_mode():
    """An unrecognized/blank mode must be SKIPPED, never assumed 'strong'."""
    rows = [
        _mk("atm_latlon", "", "0.7", "eff_2gpu_atm_latlon"),       # blank
        _mk("atm_latlon", "stron", "0.7", "eff_2gpu_atm_latlon"),  # typo
        _mk("atm_latlon", "diagnostic", "0.7", "eff_2gpu_atm_latlon"),
    ]
    g = plot_mod.gpu_points(rows)
    assert all(("atm", m) not in g for m in ("strong", "weak"))    # nothing mis-plotted


def test_gpu_points_pcg_kernel_is_ocean_only_and_exact():
    """`pcg`+`eff_ndev2` on a NON-ocean grid (or not the exact ocean prefix) is
    NOT a kernel point — the broad-substring match was the codex finding."""
    rows = [
        _mk("atm_latlon", "strong", "0.5", "atm_pcg_thing_eff_ndev2"),   # not ocean
        _mk("ocean_latlon", "strong", "0.5", "some_pcg_eff_ndev2"),      # wrong prefix
    ]
    g = plot_mod.gpu_points(rows)
    assert g == {}                                                 # neither is a valid point


def test_gpu_points_dedup_last_wins():
    """Append-only reruns with the same (grid, metric) -> last value wins, single point."""
    rows = [
        _mk("atm_latlon", "strong", "0.50", "eff_2gpu_atm_latlon", "old"),
        _mk("atm_latlon", "strong", "0.82", "eff_2gpu_atm_latlon", "new"),
    ]
    pts = plot_mod.gpu_points(rows)[("atm", "strong")]
    assert len(pts) == 1 and pts[0]["eff"] == 0.82 and pts[0]["job"] == "new"
