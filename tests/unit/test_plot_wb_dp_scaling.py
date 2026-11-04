"""The WB data-parallel ladder plotter parses timing logs and draws the PNG."""
import importlib.util
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "scripts/plot/plot_wb_dp_scaling.py"
spec = importlib.util.spec_from_file_location("plot_wb_dp_scaling", _P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _log(d, name, steps, ranks=None):
    lines = ([f"[wb_dp] abc ranks={ranks} deck=x mode=y start=1"] if ranks else [])
    lines += [f"INFO:legoesm.training.data_parallel:epoch 0 step {i + 1}/9 "
              f"loss=1.0 {t} s" for i, t in enumerate(steps)]
    (d / name).write_text("\n".join(lines))


def test_points_median_after_warmup_and_gpu_count(tmp_path):
    _log(tmp_path, "timing_t9_ngcm_6h_1.out", [99, 50, 40, 8.0, 8.2, 8.4])
    _log(tmp_path, "timing_t9dp_ngcm_6h_2.out", [99, 9, 9, 9.0, 9.5, 10.0], ranks=32)
    _log(tmp_path, "timing_t9_sfno_72h_3.out", [1, 2])            # no steady step
    _log(tmp_path, "timing_t9_sfnogc_24h_4.out", [1, 1, 1, 5, 5])
    pts = mod.collect(str(tmp_path / "timing_*.out"), exclude=("sfnogc",))
    assert pts == {("ngcm", 6, 1): 8.2, ("ngcm", 6, 32): 9.5}
    out = mod.make_figure(pts, tmp_path / "s" / "p.png")
    assert Path(out).stat().st_size > 1000


def test_ranks_logging_the_same_step_count_once_at_the_slowest(tmp_path):
    # 4 ranks each log steps 1-5; step k takes 10+k s on the slowest rank.
    steps = [(k, 10 + k - 0.5 * r) for k in range(1, 6) for r in range(4)]
    lines = ["[wb_dp] abc ranks=4 deck=x mode=y start=1"]
    lines += [f"INFO:x:epoch 0 step {k}/9 loss=1.0 {t} s" for k, t in steps]
    (tmp_path / "timing_t9L_ngcm_6h_g4_1.out").write_text("\n".join(lines))
    # steps 4 and 5 remain after the three warm-up steps: median(14, 15)
    assert mod.collect(str(tmp_path / "timing_*.out")) == {("ngcm", 6, 4): 14.5}


def test_no_points_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="no timing points"):
        mod.main(["--logs", str(tmp_path / "*.out"), "--out", str(tmp_path / "p.png")])
