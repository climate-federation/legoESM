"""Direct test for the Levante GPU scaling comparison plotter."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import sys

_PLT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "plot" / "plot_levante_gpu_scaling_comparison.py"
)
_spec = importlib.util.spec_from_file_location("plot_levante_gpu_scaling_comparison", _PLT)
plot = importlib.util.module_from_spec(_spec)
# Register before exec_module so @dataclass can resolve cls.__module__ via sys.modules.
sys.modules["plot_levante_gpu_scaling_comparison"] = plot
_spec.loader.exec_module(plot)


def _make_result(n_gpus, resolution, precision, mode, ms, sypd, mcells):
    return {
        "n_gpus": n_gpus,
        "resolution": resolution,
        "precision": precision,
        "mode": mode,
        "time_per_step_ms": ms,
        "sypd": sypd,
        "mcells_per_s": mcells,
        "scaling_efficiency": 1.0 / (ms / (sypd / 100.0 + 0.001) + 0.001),
    }


def _write_campaign(base: Path) -> Path:
    """Write a synthetic 2-GPU-count campaign under base/."""
    for n_gpus in (1, 4):
        for mode in ("weak", "strong"):
            gpu_dir = base / f"gpu_{n_gpus}" / "20260101T000000Z"
            gpu_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for prec in ("float32", "float64"):
                factor = 2.0 if prec == "float64" else 1.0
                if mode == "weak":
                    results.append(_make_result(n_gpus, 4, prec, mode,
                                                0.3 * factor * (1 + 0.05 * n_gpus),
                                                5000 / factor, 200 / factor))
                else:
                    for res in (4, 5):
                        results.append(_make_result(n_gpus, res, prec, mode,
                                                    0.3 * factor * (5 / n_gpus + 0.2),
                                                    5000 / factor * n_gpus * 0.9, 200 / factor))
            payload = {"mode": mode, "precisions": ["float32", "float64"],
                       "timestamp_utc": "2026-01-01T00:00:00+00:00",
                       "backend": "GPU", "hostname": "test", "results": results}
            (gpu_dir / f"{mode}_scaling.json").write_text(json.dumps(payload))
    return base


def test_load_campaign_reads_json(tmp_path):
    cam = _write_campaign(tmp_path / "campaign")
    points = plot._load_campaign(cam, label="test")
    assert len(points) > 0
    assert all(p.n_gpus in (1, 4) for p in points)
    assert all(p.precision in ("float32", "float64") for p in points)
    assert all(p.mode in ("weak", "strong") for p in points)
    assert all(p.campaign == "test" for p in points)


def test_make_figure_creates_png(tmp_path):
    cam = _write_campaign(tmp_path / "campaign")
    points = plot._load_campaign(cam)
    out = plot.make_figure(points, tmp_path / "out")
    assert out is not None and out.exists()
    assert out.stat().st_size > 5_000


def test_make_figure_empty_returns_none(tmp_path):
    assert plot.make_figure([], tmp_path / "out") is None


def test_write_summary_creates_txt(tmp_path):
    cam = _write_campaign(tmp_path / "campaign")
    points = plot._load_campaign(cam)
    out = plot.write_summary(points, tmp_path / "out")
    assert out.exists()
    text = out.read_text()
    assert "Weak Scaling" in text
    assert "Strong Scaling" in text
    assert "float32" in text


def test_cli_missing_campaign_dir_returns_1(monkeypatch):
    monkeypatch.setattr("sys.argv", ["plot_levante_gpu_scaling_comparison.py"])
    assert plot.main() == 1


def test_cli_end_to_end(tmp_path, monkeypatch):
    cam = _write_campaign(tmp_path / "campaign")
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "plot_levante_gpu_scaling_comparison.py",
        "--campaign-dir", str(cam),
        "--output-dir", str(out),
    ])
    assert plot.main() == 0
    assert (out / "scaling_comparison.png").exists()
    assert (out / "scaling_summary.txt").exists()
