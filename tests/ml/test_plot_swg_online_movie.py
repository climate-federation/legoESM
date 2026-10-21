"""Pure helpers of the SWG online comparison movie script: frame days, score line, shared colour limit, run table."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/plot/plot_swg_online_movie.py"
spec = importlib.util.spec_from_file_location("plot_swg_online_movie", SCRIPT)
movie = importlib.util.module_from_spec(spec)
spec.loader.exec_module(movie)


def test_frame_days_inclusive_and_strided():
    assert movie.frame_days(3651, 3655, 2) == [3651, 3653, 3655]
    assert movie.frame_days(3651, 3651, 1) == [3651]
    for bad in ((0, 5, 1), (5, 4, 1), (1, 5, 0)):
        with pytest.raises(ValueError):
            movie.frame_days(*bad)


def test_score_line_rounds_the_three_reported_scores():
    med = {"mean_face_transport_rmse_m3_s": 723726.26, "wet_eke_map_nrmse": 0.24501672, "basin_eke_ratio": 1.0748385}
    assert movie.score_line(med) == "transport 0.72e6 | EKE map 0.25 | basin EKE 1.07"


def test_symmetric_limit_pools_arrays_and_ignores_nan():
    a = np.array([[np.nan, -4.0], [1.0, 2.0]])
    b = np.array([0.5, 3.0])
    assert movie.symmetric_limit([a, b], 1.0) == 4.0
    assert movie.symmetric_limit([np.full((3, 3), -2.0)], 0.5) == 2.0


def test_run_table_keys_are_unique_and_scores_belong_to_their_run():
    keys = [run[0] for run in movie.RUNS]
    assert len(set(keys)) == len(keys) == 5
    for _, _, root, label in movie.RUNS:
        score_file = movie.SCORES / f"{label}.json"
        if not score_file.is_file():
            pytest.skip("score files are not on this machine")
        assert Path(json.loads(score_file.read_text())["root"]) == root
