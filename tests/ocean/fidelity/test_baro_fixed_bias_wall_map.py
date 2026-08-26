"""Tests for the state-constant end-wall bias probe.

Every test here exists to make one specific WRONG answer impossible.  The
probe's own conclusions rest on four pieces of logic -- the constancy
statistic, the coherence of a row, the per-cell regression that vetoes an
amplitude coincidence, and the frame-alignment guard on the NEMO dump -- and
each is exercised against a case whose answer is known independently of the
code under test.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_PROBE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "scripts", "validate",
    "ocean_fidelity", "dino_1226", "baro_fixed_bias_wall_map.py")


def _load():
    spec = importlib.util.spec_from_file_location("baro_fixed_bias_wall_map",
                                                  os.path.abspath(_PROBE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


# --------------------------------------------------------------------------
# the constancy statistic
# --------------------------------------------------------------------------
def test_selftest_passes_its_own_known_answers():
    st = M.constancy_selftest()
    assert st["constant_departure_share"] < 1e-12
    assert st["constant_corr_min"] > 0.99
    # An ALTERNATING field must NOT read as constant -- this is the two-step
    # mode itself, and a sign-blind statistic would call it state-constant.
    assert st["alternating_corr_min"] < -0.99
    assert st["alternating_departure_share"] > 0.5


def test_constancy_reports_a_real_departure():
    """A field with a known 10% wobble must report ~10%, not ~0."""
    rng = np.random.default_rng(1)
    wet = np.ones((6, 6), dtype=bool)
    base = rng.standard_normal((6, 6))
    base /= np.sqrt((base ** 2).mean())
    series = np.stack([base + 0.1 * rng.standard_normal((6, 6))
                       for _ in range(5)], axis=0)
    got = M.constancy(series, wet)
    assert 0.05 < got["max_departure_share"] < 0.25


def test_constancy_is_sign_aware():
    """An anti-correlated pair must surface as a NEGATIVE minimum correlation.

    If this ever passes with a magnitude-only correlation the probe would
    report the alternating mode as a fixed field.
    """
    wet = np.ones((4, 4), dtype=bool)
    base = np.random.default_rng(2).standard_normal((4, 4))
    series = np.stack([base, -base, base, -base, base], axis=0)
    assert M.constancy(series, wet)["cross_state_corr_min"] < -0.99


# --------------------------------------------------------------------------
# row coherence
# --------------------------------------------------------------------------
def test_row_profile_coherence_separates_offset_from_wiggle():
    """A uniform row scores 1; a zero-mean row of the SAME rms scores ~0."""
    wet = np.ones((2, 8), dtype=bool)
    field = np.zeros((2, 8))
    field[0, :] = 3.0                       # uniform offset
    field[1, :] = 3.0 * ((-1.0) ** np.arange(8))   # same rms, zero mean
    prof = M.row_profile(field, wet)
    assert prof[0]["coherence"] == pytest.approx(1.0)
    assert prof[1]["coherence"] < 1e-12
    assert prof[0]["rms"] == pytest.approx(prof[1]["rms"])


def test_row_profile_marks_dry_rows_rather_than_averaging_them():
    wet = np.ones((3, 4), dtype=bool)
    wet[1] = False
    prof = M.row_profile(np.ones((3, 4)), wet)
    assert prof[1]["n_wet"] == 0 and prof[1]["rms"] is None


# --------------------------------------------------------------------------
# wall rows come from the MASK
# --------------------------------------------------------------------------
def test_wall_rows_are_read_from_the_mask_not_hardcoded():
    wet = np.zeros((10, 4), dtype=bool)
    wet[2:7] = True
    assert M.wall_rows(wet) == (2, 6)
    # the staggering case the real data has: the two components legitimately
    # name different northern rows, and the function must follow each mask.
    wet_v = np.zeros((10, 4), dtype=bool)
    wet_v[2:6] = True
    assert M.wall_rows(wet_v) == (2, 5)


def test_wall_rows_rejects_an_empty_mask():
    with pytest.raises(SystemExit):
        M.wall_rows(np.zeros((4, 4), dtype=bool))


# --------------------------------------------------------------------------
# the per-cell regression, and the coincidence veto
# --------------------------------------------------------------------------
def test_cell_regression_recovers_a_known_slope():
    rng = np.random.default_rng(3)
    vel = rng.standard_normal((1, 40))
    bias = -2.5e-5 * vel
    got = M.cell_regression(bias, vel, np.ones((1, 40), dtype=bool))
    assert got["slope"] == pytest.approx(-2.5e-5, rel=1e-9)
    assert got["corr"] == pytest.approx(-1.0, abs=1e-9)


def test_cell_regression_reports_low_correlation_for_unrelated_fields():
    rng = np.random.default_rng(4)
    got = M.cell_regression(rng.standard_normal((1, 200)),
                            rng.standard_normal((1, 200)),
                            np.ones((1, 200), dtype=bool))
    assert abs(got["corr"]) < 0.3


def _pair_inputs(corr_north: float, n: int = 60):
    """Two wall rows whose AMPLITUDES track the velocity by construction.

    The southern row's bias is exactly proportional to its velocity; the
    northern row's is a blend controlled by ``corr_north`` but rescaled to the
    amplitude proportionality would predict.  So the amplitude test always
    passes and only the per-cell test can tell the two cases apart.
    """
    rng = np.random.default_rng(5)
    nrow = 8
    wet = np.zeros((nrow, n), dtype=bool)
    wet[1] = True
    wet[nrow - 2] = True
    vel = np.zeros((nrow, n))
    vel[1] = rng.standard_normal(n)
    vel[nrow - 2] = 10.0 * rng.standard_normal(n)
    k = 1e-5
    bias = np.zeros((nrow, n))
    bias[1] = k * vel[1]
    mix = corr_north * vel[nrow - 2] / vel[nrow - 2].std() + \
        np.sqrt(max(1 - corr_north ** 2, 0)) * rng.standard_normal(n)
    mix *= (k * vel[nrow - 2]).std() / mix.std()
    bias[nrow - 2] = mix
    return bias, vel, wet


def test_amplitude_coincidence_flag_fires_when_cells_do_not_track():
    bias, vel, wet = _pair_inputs(corr_north=0.02)
    p = M.mirrored_pairs(bias, vel, wet, depth=1)[0]
    # the amplitudes agree, by construction
    assert abs(p["bias_amp_ratio_n_over_s"] / p["vel_amp_ratio_n_over_s"]
               - 1.0) < 0.25
    assert p["AMPLITUDE_COINCIDENCE_FLAG"] is True


def test_amplitude_coincidence_flag_stays_silent_when_cells_do_track():
    bias, vel, wet = _pair_inputs(corr_north=0.99)
    p = M.mirrored_pairs(bias, vel, wet, depth=1)[0]
    assert abs(p["bias_amp_ratio_n_over_s"] / p["vel_amp_ratio_n_over_s"]
               - 1.0) < 0.25
    assert p["AMPLITUDE_COINCIDENCE_FLAG"] is False


def test_flag_needs_only_ONE_end_to_fail():
    """The veto is `any`, not `all`.

    Proportionality is a claim about both walls, so one wall whose cells do not
    track already voids it.  With `all` the flag stayed silent on exactly the
    row pair that motivated it in the real data (south -0.50, north +0.05).
    """
    bias, vel, wet = _pair_inputs(corr_north=0.02)
    j_s, j_n = M.wall_rows(wet)
    south = M.cell_regression(bias[j_s], vel[j_s], wet[j_s])
    north = M.cell_regression(bias[j_n], vel[j_n], wet[j_n])
    assert abs(south["corr"]) > 0.9        # this end DOES track
    assert abs(north["corr"]) < 0.3        # this end does not
    assert M.mirrored_pairs(bias, vel, wet, depth=1)[0][
        "AMPLITUDE_COINCIDENCE_FLAG"] is True


# --------------------------------------------------------------------------
# loaders refuse to guess
# --------------------------------------------------------------------------
def test_load_maps_refuses_a_hole_in_the_series(tmp_path):
    with pytest.raises(SystemExit, match="no deposit map"):
        M.load_maps(str(tmp_path), kts=(5760,))


def test_load_nemo_2d_refuses_a_wrong_element_count(tmp_path):
    p = tmp_path / "wnd_dump_zu_frc_inc.bin"
    np.zeros(17, dtype="<f8").tofile(p)
    with pytest.raises(SystemExit, match="elements"):
        M.load_nemo_2d(str(tmp_path), "wnd_dump_zu_frc_inc.bin")


def test_load_nemo_2d_refuses_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="no NEMO dump"):
        M.load_nemo_2d(str(tmp_path), "nope.bin")


def test_wind_candidate_refuses_a_frame_mismatch(tmp_path):
    """The alignment control: if the wind dump's wet set is not the map's, the
    two arrays are on different frames and every comparison downstream is void.
    """
    wu = np.zeros((M.JPJ, M.JPI))
    wu[10:20, :] = 1.0
    wu.tofile(tmp_path / "wnd_dump_zu_frc_inc.bin")
    np.zeros((M.JPJ, M.JPI)).tofile(tmp_path / "wnd_dump_zv_frc_inc.bin")
    wetu = np.zeros((M.JPJ, M.JPI), dtype=bool)
    wetu[30:40, :] = True          # deliberately NOT the dump's nonzero set
    with pytest.raises(SystemExit, match="different frames"):
        M.wind_candidate(str(tmp_path), np.zeros((M.JPJ, M.JPI)),
                         np.zeros((M.JPJ, M.JPI)), wetu, wetu,
                         np.ones(68) / 68.0, 0.15, 117.0)


def test_wind_candidate_flags_a_nonzero_meridional_increment(monkeypatch,
                                                            tmp_path):
    """The decisive geometry fact must be MEASURED, not assumed.

    If NEMO ever dumps a nonzero meridional wind increment, the probe must
    report it rather than keep printing that the term is zero.
    """
    wu = np.zeros((M.JPJ, M.JPI))
    wu[10:20, :] = 1.0
    wu.tofile(tmp_path / "wnd_dump_zu_frc_inc.bin")
    wv = np.zeros((M.JPJ, M.JPI))
    wv[5, 5] = 3.0
    wv.tofile(tmp_path / "wnd_dump_zv_frc_inc.bin")
    wetu = (wu != 0.0)
    monkeypatch.setattr(M, "coriolis_at_row", lambda seqdump, j: 1.0e-4)
    got = M.wind_candidate(str(tmp_path), np.zeros((M.JPJ, M.JPI)),
                           np.zeros((M.JPJ, M.JPI)), wetu, wetu,
                           np.ones(68) / 68.0, 0.15, 117.0)
    assert got["v_increment_exactly_zero"] is False
    assert got["v_increment_max_abs"] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# the two-gridpoint zigzag share
# --------------------------------------------------------------------------
def test_two_dx_share_is_one_for_a_pure_zigzag():
    wet = np.ones((1, 20), dtype=bool)
    field = (3.0 * (-1.0) ** np.arange(20)).reshape(1, 20)
    assert M.two_dx_share(field, wet, 0) == pytest.approx(1.0, abs=1e-12)


def test_two_dx_share_is_zero_for_a_smooth_zero_mean_row():
    """The case the statistic exists to separate: zero MEAN but no zigzag.

    Coherence alone cannot tell this row from a checkerboard; this must.
    """
    wet = np.ones((1, 20), dtype=bool)
    x = np.sin(2 * np.pi * np.arange(20) / 20.0).reshape(1, 20)
    assert abs(x[0].mean()) < 1e-12          # genuinely zero-mean
    assert M.two_dx_share(x, wet, 0) < 1e-10


def test_two_dx_share_ignores_a_constant_offset():
    """The basis vector is de-meaned, so a pure offset must score 0, not 1."""
    wet = np.ones((1, 20), dtype=bool)
    assert M.two_dx_share(np.full((1, 20), 5.0), wet, 0) < 1e-12


def test_two_dx_share_declines_to_report_a_too_short_row():
    wet = np.zeros((1, 20), dtype=bool)
    wet[0, :4] = True
    assert M.two_dx_share(np.ones((1, 20)), wet, 0) is None


# --------------------------------------------------------------------------
# the headline reduction (review 2026-08-26: 12 of 14 mutations were green)
# --------------------------------------------------------------------------
def test_wall_statistic_is_an_rms_not_a_cancelling_signed_mean():
    """A zero-MEAN row of real amplitude must report its amplitude, not ~0.

    This is the meridional wall row's actual character, and a signed-mean
    reduction there would report 4e-07 m/s of bias as nothing.
    """
    wet = np.zeros((12, 20), dtype=bool)
    wet[1] = True
    wet[10] = True
    bias = np.zeros((12, 20))
    bias[1] = 7.0 * (-1.0) ** np.arange(20)      # zero mean, amplitude 7
    bias[10] = 3.0
    got = M.wall_and_interior(bias, wet, exclude=2)
    assert got["south"] == pytest.approx(7.0)
    assert abs(bias[1][wet[1]].mean()) < 1e-12   # the trap it avoids


def test_interior_excludes_the_wall_band_on_both_sides():
    wet = np.ones((21, 6), dtype=bool)
    bias = np.zeros((21, 6))
    bias[:4] = 100.0          # wall band, must NOT enter the interior median
    bias[17:] = 100.0
    bias[4:17] = 1.0
    got = M.wall_and_interior(bias, wet, exclude=4)
    assert got["interior_median"] == pytest.approx(1.0)
    assert got["n_interior_rows"] == 13


def test_interior_median_moves_if_the_exclusion_shrinks():
    """Guards the exclusion width itself: it must actually do something.

    The wall band has to OUTNUMBER the interior for a median to move, which an
    earlier version of this test got wrong -- with 8 wall rows against 13
    interior rows the median never budged and the test passed for the wrong
    reason.  Here 18 rows are wall band and 3 are interior.
    """
    wet = np.ones((21, 6), dtype=bool)
    bias = np.full((21, 6), 100.0)
    bias[9:12] = 1.0
    wide = M.wall_and_interior(bias, wet, exclude=9)
    narrow = M.wall_and_interior(bias, wet, exclude=0)
    assert wide["interior_median"] == pytest.approx(1.0)
    assert wide["n_interior_rows"] == 3
    assert narrow["interior_median"] == pytest.approx(100.0)
    assert narrow["n_interior_rows"] == 21


def test_constancy_departure_is_against_the_MEAN_not_state_zero():
    """A series that drifts symmetrically about its mean.

    Measured against state 0 the departure would be twice as large; against the
    mean it is the half-range.  The claim quotes the against-the-mean number.
    """
    wet = np.ones((1, 6), dtype=bool)
    base = np.full((1, 6), 10.0)
    series = np.stack([base - 1.0, base, base + 1.0], axis=0)
    got = M.constancy(series, wet)
    assert got["state_mean_rms"] == pytest.approx(10.0)
    assert max(got["departure_rms"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# section 4 numbers: suppression, jbar, shape correlation, rotated bound
# --------------------------------------------------------------------------
def _fake_seqdump(tmp_path, wu, wv=None, f_const=1.0e-4):
    wu.tofile(tmp_path / "wnd_dump_zu_frc_inc.bin")
    (np.zeros_like(wu) if wv is None else wv).tofile(
        tmp_path / "wnd_dump_zv_frc_inc.bin")
    return str(tmp_path)


def _patch_coriolis(monkeypatch, value):
    monkeypatch.setattr(M, "coriolis_at_row", lambda seqdump, j: value)


def test_wall_suppression_ratio_is_peak_over_wall_not_inverted(monkeypatch,
                                                               tmp_path):
    """A wind term 100x weaker at the wall must read 100x, not 0.01x."""
    wu = np.zeros((M.JPJ, M.JPI))
    wu[5:60, :] = 1.0
    wu[5, :] = 0.01                       # the southern-most live row
    wetu = (wu != 0.0)
    _patch_coriolis(monkeypatch, 1.0e-4)
    got = M.wind_candidate(_fake_seqdump(tmp_path, wu),
                           np.zeros((M.JPJ, M.JPI)), np.zeros((M.JPJ, M.JPI)),
                           wetu, wetu, np.ones(68) / 68.0, 1.0, 100.0)
    assert got["u_increment_peak"] == pytest.approx(1.0)
    assert got["wall_vs_peak_suppression_south"] == pytest.approx(100.0)


def test_jbar_is_the_weighted_mean_substep_index_one_based(monkeypatch,
                                                           tmp_path):
    """Uniform weights over n substeps give jbar = (n+1)/2, not (n-1)/2."""
    wu = np.zeros((M.JPJ, M.JPI))
    wu[5:60, :] = 1.0
    wetu = (wu != 0.0)
    _patch_coriolis(monkeypatch, 1.0e-4)
    got = M.wind_candidate(_fake_seqdump(tmp_path, wu),
                           np.zeros((M.JPJ, M.JPI)), np.zeros((M.JPJ, M.JPI)),
                           wetu, wetu, np.ones(10) / 10.0, 1.0, 1.0)
    assert got["jbar_substeps"] == pytest.approx(5.5)


def test_shape_correlation_uses_both_profiles(monkeypatch, tmp_path):
    """A bias shaped OPPOSITE to the wind must score negative, not +1."""
    wu = np.zeros((M.JPJ, M.JPI))
    rows = np.arange(5, 60)
    wu[5:60, :] = np.linspace(1.0, 0.01, rows.size)[:, None]
    wetu = (wu != 0.0)
    bias = np.zeros((M.JPJ, M.JPI))
    bias[5:60, :] = np.linspace(0.01, 1.0, rows.size)[:, None]   # reversed
    _patch_coriolis(monkeypatch, 1.0e-4)
    got = M.wind_candidate(_fake_seqdump(tmp_path, wu), bias,
                           np.zeros((M.JPJ, M.JPI)), wetu, wetu,
                           np.ones(68) / 68.0, 1.0, 1.0)
    assert got["row_profile_shape_corr_with_bias"] < -0.9


def test_rotated_bound_scales_with_coriolis_and_vanishes_without_it(
        monkeypatch, tmp_path):
    """The whole point of the rotated bound: no rotation, no meridional deposit.

    Review 2026-08-26 refuted the earlier claim that a zero meridional FORCING
    means a zero meridional RESPONSE.  This pins the correction: the deposit is
    linear in f and is exactly zero at f = 0.
    """
    wu = np.zeros((M.JPJ, M.JPI))
    wu[5:60, :] = 1.0
    wetu = (wu != 0.0)
    args = (np.zeros((M.JPJ, M.JPI)), np.zeros((M.JPJ, M.JPI)), wetu, wetu,
            np.ones(68) / 68.0, 1.0, 10.0)
    sd = _fake_seqdump(tmp_path, wu)

    _patch_coriolis(monkeypatch, 0.0)
    zero = M.wind_candidate(sd, *args)["bounds"]["response_1.0"][
        "rotated_meridional"]["south"]["rotated_v_deposit_from_FULL_wind_term"]
    assert zero == 0.0

    _patch_coriolis(monkeypatch, 1.0e-4)
    one = M.wind_candidate(sd, *args)["bounds"]["response_1.0"][
        "rotated_meridional"]["south"]["rotated_v_deposit_from_FULL_wind_term"]
    _patch_coriolis(monkeypatch, 2.0e-4)
    two = M.wind_candidate(sd, *args)["bounds"]["response_1.0"][
        "rotated_meridional"]["south"]["rotated_v_deposit_from_FULL_wind_term"]
    assert one > 0.0
    assert two == pytest.approx(2.0 * one)


def test_exclusion_verdict_is_computed_not_asserted(monkeypatch, tmp_path):
    """A candidate LARGER than the bias must report DOES NOT EXCLUDE.

    The southern wall is exactly this case in the real data, and an earlier
    draft's prose called the wind 'excluded' everywhere.
    """
    wu = np.full((M.JPJ, M.JPI), 1.0)
    wu[0] = 0.0
    wu[-1] = 0.0
    wetu = (wu != 0.0)
    tiny = np.zeros((M.JPJ, M.JPI))       # bias far smaller than the candidate
    _patch_coriolis(monkeypatch, 1.0e-4)
    got = M.wind_candidate(_fake_seqdump(tmp_path, wu), tiny, tiny,
                           wetu, wetu, np.ones(68) / 68.0, 1.0, 100.0)
    b = got["bounds"]["response_1.0"]
    assert b["u_EXCLUDES_south"] is False
    assert b["rotated_meridional"]["south"]["EXCLUDES"] is False


# --------------------------------------------------------------------------
# loader guards that back the state series
# --------------------------------------------------------------------------
def _write_map(path, kt, wetu, wetv, n=(M.JPJ, M.JPI)):
    z = {"dU_avg": np.zeros(n), "dU_sub": np.zeros(n),
         "wetu": wetu.astype(float), "nemo_Ubar_avg": np.ones(n),
         "dV_avg": np.zeros(n), "dV_sub": np.zeros(n),
         "wetv": wetv.astype(float), "nemo_Vbar_avg": np.ones(n),
         "wgt_primary": np.ones(68) / 68.0, "ic_step": kt}
    np.savez(path, **z)


def test_load_maps_rejects_a_map_whose_ic_step_disagrees(tmp_path):
    wet = np.ones((M.JPJ, M.JPI), dtype=bool)
    _write_map(tmp_path / "deposit_map_kt5760.npz", 9999, wet, wet)
    with pytest.raises(SystemExit, match="carries ic_step"):
        M.load_maps(str(tmp_path), kts=(5760,))


def test_load_maps_rejects_a_mask_that_changes_between_states(tmp_path):
    """The state MEAN over a changing mask averages different cells."""
    a = np.ones((M.JPJ, M.JPI), dtype=bool)
    b = a.copy()
    b[10] = False
    _write_map(tmp_path / "deposit_map_kt5760.npz", 5760, a, a)
    _write_map(tmp_path / "deposit_map_kt5761.npz", 5761, b, b)
    with pytest.raises(SystemExit, match="differs across states"):
        M.load_maps(str(tmp_path), kts=(5760, 5761))


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_geometry_reports_unavailable_rather_than_guessing(tmp_path):
    got = M.wall_geometry(str(tmp_path))
    assert got["available"] is False and "domain_cfg_out.nc" in got["reason"]


def _geo_row(lat, depth, e2t):
    return {"lat_deg_mean_wet": lat, "mean_depth_m": depth,
            "e2t_m_mean_wet": e2t, "j": 0, "n_wet": 50}


def test_symmetry_summary_computes_the_latitude_difference():
    """Mutating this to a hard 0.0 left the whole suite green before."""
    got = M.symmetry_summary(_geo_row(-60.0, 1000.0, 100.0),
                             _geo_row(+70.0, 1000.0, 100.0))
    assert got["abs_lat_diff_deg"] == pytest.approx(10.0)
    assert got["GEOMETRY_ASYMMETRY_FLAG"] is True


def test_symmetry_summary_flags_a_depth_asymmetry():
    got = M.symmetry_summary(_geo_row(-69.5, 2000.0, 100.0),
                             _geo_row(+69.5, 2300.0, 100.0))
    assert got["depth_ratio_n_over_s"] == pytest.approx(1.15)
    assert got["worst_fractional_asymmetry"] == pytest.approx(0.15)
    assert got["GEOMETRY_ASYMMETRY_FLAG"] is True


def test_symmetry_summary_stays_silent_on_a_genuinely_symmetric_pair():
    got = M.symmetry_summary(_geo_row(-69.504, 2200.0, 38934.6),
                             _geo_row(+69.504, 2210.0, 38934.6))
    assert got["abs_lat_diff_deg"] == pytest.approx(0.0, abs=1e-9)
    assert got["GEOMETRY_ASYMMETRY_FLAG"] is False


# --------------------------------------------------------------------------
# the v-face zonal metric gap (raised by mechanism review 2026-08-26)
# --------------------------------------------------------------------------
def test_vface_gap_is_exactly_zero_when_the_conventions_agree():
    """The control: if NEMO's v latitude IS the tracer midpoint, no gap.

    Without this the formula could manufacture a gap out of nothing and the
    whole candidate would be an artifact of the probe.
    """
    gphit = np.linspace(-69.0, 69.0, 40)[:, None] * np.ones((1, 4))
    gphiv = np.zeros_like(gphit)
    gphiv[:-1] = 0.5 * (gphit[:-1] + gphit[1:])
    _, gap = M.vface_gap_from_latitudes(gphit, gphiv)
    assert np.abs(gap).max() < 1e-15


def test_vface_gap_sign_is_positive_when_lego_face_is_wider():
    """Positive must mean legoESM WIDER, i.e. cos(midpoint) > cos(gphiv).

    A sign flip here would point the candidate at the wrong side of the
    balance, so it is pinned rather than trusted to the comment.
    """
    gphit = np.array([[10.0], [20.0]])
    # NEMO's v latitude further from the equator than the midpoint (15) ->
    # smaller cosine -> legoESM's face is the wider one -> positive gap.
    gphiv = np.array([[16.0], [0.0]])
    _, gap = M.vface_gap_from_latitudes(gphit, gphiv)
    assert gap[0, 0] > 0.0
    gphiv2 = np.array([[14.0], [0.0]])
    _, gap2 = M.vface_gap_from_latitudes(gphit, gphiv2)
    assert gap2[0, 0] < 0.0


def test_vface_gap_vanishes_at_the_equator_and_grows_poleward():
    """The latitude SHAPE is the reason this candidate is credible."""
    lat = np.linspace(-69.0, 69.0, 199)[:, None]
    gphit = lat * np.ones((1, 3))
    off = 0.001
    gphiv = np.zeros_like(gphit)
    gphiv[:-1] = 0.5 * (gphit[:-1] + gphit[1:]) + off * np.sign(
        0.5 * (gphit[:-1] + gphit[1:]))
    _, gap = M.vface_gap_from_latitudes(gphit, gphiv)
    eq = int(np.argmin(np.abs(gphit[:-1, 0])))
    assert abs(gap[eq, 0]) < abs(gap[0, 0]) / 100.0
    assert abs(gap[0, 0]) > 0.0
    assert abs(gap[-1, 0]) > 0.0


def test_vface_metric_reports_unavailable_rather_than_guessing(tmp_path):
    got = M.vface_zonal_metric_gap(str(tmp_path))
    assert got["available"] is False


def test_vface_shape_test_scores_a_matching_shape_high():
    """Control: a deficit built to BE the gap must correlate ~+1."""
    nlat, nlon = 40, 5
    gphit = np.linspace(-69.0, 69.0, nlat + 1)[:, None] * np.ones((1, nlon))
    gphiv = np.zeros_like(gphit)
    gphiv[:-1] = 0.5 * (gphit[:-1] + gphit[1:]) + 0.001 * np.sign(
        0.5 * (gphit[:-1] + gphit[1:]))
    _, gap = M.vface_gap_from_latitudes(gphit, gphiv)
    wetv = np.ones((nlat, nlon), dtype=bool)
    vel = np.ones((nlat, nlon))
    bias = np.abs(gap[:nlat]) * vel
    got = M.vface_shape_test({"_gphit": gphit, "_gphiv": gphiv},
                             bias, vel, wetv)
    assert got["corr"] > 0.99


def test_vface_shape_test_scores_an_unrelated_shape_near_zero():
    """The case the real data is in: right order, wrong latitude structure.

    Without this the shape test could report agreement for any candidate of
    roughly the right size, which is the two-row coincidence this probe exists
    to refuse.
    """
    nlat, nlon = 40, 5
    gphit = np.linspace(-69.0, 69.0, nlat + 1)[:, None] * np.ones((1, nlon))
    gphiv = np.zeros_like(gphit)
    gphiv[:-1] = 0.5 * (gphit[:-1] + gphit[1:]) + 0.001 * np.sign(
        0.5 * (gphit[:-1] + gphit[1:]))
    wetv = np.ones((nlat, nlon), dtype=bool)
    vel = np.ones((nlat, nlon))
    rng = np.random.default_rng(7)
    bias = np.abs(rng.standard_normal((nlat, nlon))) * 3e-5
    got = M.vface_shape_test({"_gphit": gphit, "_gphiv": gphiv},
                             bias, vel, wetv)
    assert abs(got["corr"]) < 0.4
