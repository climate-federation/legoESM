"""Direct, oracle-independent tests for the DINO twin/NEMO map comparator."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.dino_1226 import twin_nemo_ts_maps as M


def test_one_ring_alignment_is_explicit_and_fail_closed():
    full = np.zeros((199, 52, 3), dtype=np.float64)
    cropped = M.crop_one_ring("synthetic", full)
    assert cropped.shape == (197, 50, 3)
    with pytest.raises(RuntimeError, match="expected full horizontal"):
        M.crop_one_ring("wrong", np.zeros((197, 50, 3)))


def test_planted_wet_cell_moves_statistics_and_dry_cell_does_not():
    receipt = M.planted_violation_self_test()
    assert receipt["passed"]
    assert receipt["before"]["max_abs_difference"] == 0.0
    assert receipt["after"]["max_abs_difference"] == 0.25
    assert receipt["after"]["rms_difference"] > 0.0
    assert receipt["dry_cell_excluded"]


def test_argmax_locator_reports_indices_coordinates_depth_and_sign():
    reference = np.zeros((2, 3, 4), dtype=np.float64)
    candidate = reference.copy()
    candidate[1, 2, 3] = -0.75
    candidate[0, 0, 0] = 9.0  # dry and therefore excluded
    wet = np.ones_like(candidate, dtype=bool)
    wet[0, 0, 0] = False
    lat = np.asarray([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    lon = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    depth = np.broadcast_to(np.asarray([5.0, 15.0, 30.0, 50.0]), candidate.shape)
    peak = M.locate_abs_argmax(candidate, reference, wet, lat, lon, depth)
    assert peak["index_jik_map_zero_based"] == [1, 2, 3]
    assert peak["index_jik_full_zero_based"] == [2, 3, 3]
    assert peak["latitude_deg_north"] == 20.0
    assert peak["longitude_deg_east"] == 3.0
    assert peak["depth_m"] == 50.0
    assert peak["difference"] == -0.75
    assert peak["abs_difference"] == 0.75


def test_cli_self_test_requires_no_oracle_inputs(capsys):
    assert M.main(["--self-test"]) == 0
    assert '"passed": true' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --run-dino-dir : the second producer (added with the DINO true-frame fix).
# Synthetic run directories, so these run anywhere -- no NEMO file, no GPU.
# ---------------------------------------------------------------------------

def _write_run(tmp_path, *, days=(0.0, 360.0), shape=(199, 52), nlev=36,
               t_fill=10.0):
    import numpy as np
    snaps = tmp_path / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(days):
        np.savez(
            snaps / f"snapshot_{i:05d}.npz",
            time_days=np.float64(d), time_seconds=np.float64(d * 86400.0),
            T=np.full((*shape, nlev), t_fill, dtype=np.float32),
            S=np.full((*shape, nlev), 35.0, dtype=np.float32),
            eta=np.zeros(shape, dtype=np.float32),
            land_mask=np.ones(shape, dtype=np.float32),
        )
    return tmp_path


def test_run_dino_loader_returns_the_full_nemo_frame_in_fp64(tmp_path):
    import numpy as np
    run = _write_run(tmp_path / "run")
    T, S, ssh, land, stamps = M.load_run_dino_snapshot(run, 360)
    assert T.shape == (199, 52, 36) and S.shape == (199, 52, 36)
    assert ssh.shape == (199, 52) and land.shape == (199, 52)
    assert T.dtype == S.dtype == np.float64
    # The SOURCE dtype is reported truthfully rather than asserted to be fp64.
    assert stamps["field_dtypes"]["T"] == "float32"
    assert stamps["upcast_to_float64_for_scoring"] is True
    assert stamps["producer"] == "run_dino_snapshot"
    assert stamps["snapshot_time_days"] == 360.0


def test_run_dino_loader_pads_a_short_ladder_to_nemos_36_levels(tmp_path):
    import numpy as np
    run = _write_run(tmp_path / "run35", nlev=35)
    T, _S, _ssh, _land, stamps = M.load_run_dino_snapshot(run, 360)
    assert T.shape[2] == 36
    assert stamps["vertical_zero_pad_to_36_levels"] == 1
    # The pad is zeros, i.e. only ever compared against NEMO's dry 36th level.
    assert np.array_equal(T[..., 35], np.zeros((199, 52)))
    assert np.allclose(T[..., 34], 10.0)


def test_run_dino_loader_refuses_a_non_nemo_frame(tmp_path):
    # The whole point of the frame fix: a 48x195 run must not be silently
    # pasted into NEMO's domain.
    run = _write_run(tmp_path / "old", shape=(195, 48), nlev=35)
    with pytest.raises(RuntimeError, match="nemo-faithful-grid"):
        M.load_run_dino_snapshot(run, 360)


def test_run_dino_loader_refuses_a_missing_day(tmp_path):
    run = _write_run(tmp_path / "run")
    with pytest.raises(RuntimeError, match="no snapshot at day"):
        M.load_run_dino_snapshot(run, 180)


def test_run_dino_loader_refuses_an_unstable_run(tmp_path):
    import numpy as np
    run = _write_run(tmp_path / "nan")
    bad = run / "snapshots" / "snapshot_00001.npz"
    with np.load(bad) as z:
        fields = {k: z[k] for k in z.files}
    fields["T"] = fields["T"].copy()
    fields["T"][10, 10, 0] = np.nan
    np.savez(bad, **fields)
    with pytest.raises(RuntimeError, match="not finite"):
        M.load_run_dino_snapshot(run, 360)
    # ...and a finite but unphysical state is refused too (no `stable` stamp
    # exists on this producer, so the check has to be made here).
    run2 = _write_run(tmp_path / "hot", t_fill=1e3)
    with pytest.raises(RuntimeError, match="unstable"):
        M.load_run_dino_snapshot(run2, 360)


def test_arm_and_run_dino_readers_meet_at_one_scoring_path():
    # `load_candidate` is the only fork; if a second scoring path appeared,
    # one of these would stop being a pure reader.
    import inspect
    src = inspect.getsource(M.load_candidate)
    assert "load_run_dino_snapshot" in src and "load_arm_npz" in src
    assert "error_stats" not in src and "save_figures" not in src


def test_nemo_time_level_defaults_to_now_and_offers_the_before_level():
    """Rule 1d: the MLF level a comparison scores against is a CHOICE.

    A day-0 comparison against a from-rest kt=1 record must read the Kbb
    level (tb/sb/sshb); the now level has already taken the Euler step, and
    its sshn is 1.17e-1 m where the initial ssh is exactly 0.
    """
    parser = M.build_parser()
    assert parser.parse_args([]).nemo_time_level == "now"
    assert parser.parse_args(
        ["--nemo-time-level", "before"]).nemo_time_level == "before"
    with pytest.raises(SystemExit):
        parser.parse_args(["--nemo-time-level", "kaa"])


def test_both_time_levels_name_real_restart_variables():
    # Non-vacuity for the mapping itself: a typo in either tuple would send
    # the rebuilder looking for a variable that does not exist, and the
    # rebuilder returns a SHORT dict rather than raising.
    src = inspect.getsource(M.run)
    assert '"now": ("tn", "sn", "sshn")' in src
    assert '"before": ("tb", "sb", "sshb")' in src
    assert 'require(set(raw) == set(level)' in src
