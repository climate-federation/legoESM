"""Direct tests for ``scripts/plot/plot_ad_overhead.py``.

The figure's job is to show that checkpointing is cheaper in BOTH time and
memory, so the load path must never silently drop or fake a case: a failed
forward baseline means no ratio exists, and a grad OOM is a RESULT that has
to stay visible (it is exactly what happens at the finest resolutions).
"""
from __future__ import annotations

import importlib.util
import json
import os

import matplotlib
import pytest

matplotlib.use("Agg")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_MODULE_PATH = os.path.join(_REPO_ROOT, "scripts", "plot",
                            "plot_ad_overhead.py")


def _load():
    spec = importlib.util.spec_from_file_location("plot_ad_overhead",
                                                  _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plot = _load()


def _case(grid="latlon", res=192, grad_ok=True, fwd_ok=True, ckpt_ok=True):
    def _mode(ok, ratio, tape, flops=1.0):
        if not ok:
            return {"ok": False, "error": "RESOURCE_EXHAUSTED: OOM"}
        return {"ok": True, "ratio_vs_forward": ratio, "temp_bytes": tape,
                "flop_ratio_vs_forward": flops}

    return {
        "grid": grid, "resolution": res, "n_levels": 26, "steps": 8,
        "precision": "float32", "physics_level": "none",
        "modes": {
            "forward": _mode(fwd_ok, 1.0, 141_005_360),
            "grad": _mode(grad_ok, 5.45, 7_962_915_032),
            "grad_ckpt": _mode(ckpt_ok, 4.91, 726_254_464),
        },
    }


def _write(tmp_path, cases):
    for i, c in enumerate(cases):
        (tmp_path / f"case_{i}.json").write_text(json.dumps(c))
    return str(tmp_path)


def test_load_cases_reads_ratios_and_tape(tmp_path):
    cases, skipped = plot.load_cases(_write(tmp_path, [_case()]))
    assert skipped == []
    assert len(cases) == 1
    assert cases[0]["modes"]["grad"]["ratio"] == pytest.approx(5.45)
    assert cases[0]["modes"]["grad_ckpt"]["tape"] == 726_254_464
    assert cases[0]["forward_tape"] == 141_005_360


def test_case_with_failed_forward_is_skipped_not_plotted(tmp_path):
    """No baseline means no ratio exists — the case must not be drawn."""
    cases, skipped = plot.load_cases(_write(tmp_path, [_case(fwd_ok=False)]))
    assert cases == []
    assert len(skipped) == 1
    assert "forward failed" in skipped[0][1]


def test_grad_oom_is_retained_as_a_result(tmp_path):
    """A plain-BPTT OOM while checkpointing survives IS the finding.

    It must reach the figure (marked), not be silently dropped — dropping it
    would make the finest resolutions look untested rather than infeasible.
    """
    cases, skipped = plot.load_cases(_write(tmp_path, [_case(grad_ok=False)]))
    assert skipped == []
    assert len(cases) == 1
    assert cases[0]["modes"]["grad"]["ratio"] is None
    assert cases[0]["modes"]["grad_ckpt"]["ratio"] == pytest.approx(4.91)


def test_case_with_no_gradient_mode_at_all_is_skipped(tmp_path):
    cases, skipped = plot.load_cases(
        _write(tmp_path, [_case(grad_ok=False, ckpt_ok=False)]))
    assert cases == []
    assert "no gradient mode ran" in skipped[0][1]


def test_empty_dir_is_a_hard_error(tmp_path):
    with pytest.raises(SystemExit, match="no JSON files"):
        plot.load_cases(str(tmp_path))


def test_cases_sorted_by_fixed_grid_order(tmp_path):
    """Grid order is fixed, never filesystem/glob order."""
    cases, _ = plot.load_cases(_write(tmp_path, [
        _case(grid="icosahedral", res=6),
        _case(grid="latlon", res=192),
        _case(grid="cubed-sphere", res=48),
    ]))
    assert [c["grid"] for c in cases] == [
        "latlon", "cubed-sphere", "icosahedral"]


def test_make_figure_writes_png_and_pdf(tmp_path):
    src = tmp_path / "j"
    src.mkdir()
    cases, _ = plot.load_cases(_write(src, [_case()]))
    written = plot.make_figure(cases, str(tmp_path / "out"), "fig_test")
    assert len(written) == 2
    for path in written:
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_make_figure_survives_an_oom_case(tmp_path):
    """Rendering must not crash when a bar is missing."""
    src = tmp_path / "j"
    src.mkdir()
    cases, _ = plot.load_cases(_write(src, [_case(grad_ok=False)]))
    written = plot.make_figure(cases, str(tmp_path / "out"), "fig_oom")
    assert all(os.path.exists(p) for p in written)


def test_palette_is_not_the_resolution_ramp():
    """Colour means resolution in the scaling figures; it must not also mean
    AD mode here, or one hue carries two meanings across the figure set."""
    blues_ramp = {"#9ecae1", "#4292c6", "#084594"}
    assert plot.COL_GRAD not in blues_ramp
    assert plot.COL_CKPT not in blues_ramp
