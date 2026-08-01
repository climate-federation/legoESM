"""Direct tests for the LES time-mean reference loader.

The reference decides what every turbulence closure is scored against, so the
guarantees that make the comparison controlled -- an explicit averaging window,
NaN being fatal, one-directional interpolation, mass weighting -- are pinned
here on synthetic frames with known answers.
"""
from __future__ import annotations

import numpy as np
import pytest
from legoesm.training.les_reference import (
    DIAGNOSTIC_VARIABLES,
    SCORED_VARIABLES,
    load_les_reference,
    mass_weights_from_pressure,
)

NZ_LES = 40
DOMAIN_TOP = 2000.0


def _write_frames(tmp_path, n_frames=6, *, nz=NZ_LES, corrupt=None,
                  drop_var=None, t_step=0.5, values=None):
    """Write ``n_frames`` synthetic LES profile frames; return the dir."""
    prof = tmp_path / "profiles"
    prof.mkdir(parents=True, exist_ok=True)
    z = np.linspace(25.0, DOMAIN_TOP, nz)
    for i in range(n_frames):
        payload = {
            "t_hours": np.asarray(t_step * (i + 1)),
            "z": z,
            "theta": 300.0 + 0.003 * z + (0.0 if values is None else values[i]),
            "qv": 1.0e-2 - 2.0e-6 * z,
            "u": -8.0 + 0.001 * z,
            "v": np.zeros(nz),
            "wth": 0.05 * (1.0 - z / DOMAIN_TOP),
            "wqv": 1.0e-5 * (1.0 - z / DOMAIN_TOP),
            "tke": 0.5 * np.ones(nz),
        }
        if drop_var is not None:
            payload.pop(drop_var)
        if corrupt is not None and i == n_frames - 1:
            bad = payload[corrupt].copy()
            bad[3] = np.nan
            payload[corrupt] = bad
        np.savez(prof / f"prof_{i:03d}.npz", **payload)
    return tmp_path


def _scm_grid(nlev=20, top=3000.0):
    """Top-to-bottom SCM heights + half pressures."""
    z_scm = np.linspace(top, 20.0, nlev)
    p_half = np.linspace(7.0e4, 1.0e5, nlev + 1)
    return z_scm, p_half


def _load(tmp_path, **kw):
    z_scm, p_half = _scm_grid()
    kwargs = dict(case="synthetic", z_scm=z_scm, p_half=p_half,
                  domain_top_m=DOMAIN_TOP, analysis_hours=1.0)
    kwargs.update(kw)
    return load_les_reference(tmp_path, **kwargs)


# --- mass weights -----------------------------------------------------------

def test_mass_weights_sum_to_one_and_zero_outside_mask():
    p_half = np.linspace(5.0e4, 1.0e5, 6)
    mask = np.array([False, True, True, True, False])
    w = mass_weights_from_pressure(p_half, mask)
    assert w.shape == (5,)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w[~mask] == 0.0)
    # uniform dp => equal weights on the masked levels
    np.testing.assert_allclose(w[mask], 1.0 / 3.0)


def test_mass_weights_follow_layer_mass_not_level_count():
    """A thick layer must outweigh a thin one."""
    p_half = np.array([0.0, 1.0, 10.0])       # dp = [1, 9]
    mask = np.array([True, True])
    w = mass_weights_from_pressure(p_half, mask)
    np.testing.assert_allclose(w, [0.1, 0.9])


def test_empty_mask_raises():
    with pytest.raises(ValueError, match="selects no SCM level"):
        mass_weights_from_pressure(np.linspace(0, 1, 4),
                                   np.array([False, False, False]))


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="one more entry"):
        mass_weights_from_pressure(np.linspace(0, 1, 4), np.array([True, True]))


# --- the averaging window ---------------------------------------------------

def test_window_is_trailing_and_recorded(tmp_path):
    _write_frames(tmp_path, n_frames=6, t_step=0.5)      # t = 0.5 .. 3.0 h
    ref = _load(tmp_path, analysis_hours=1.0)
    # trailing 1 h of a run ending at 3.0 h => frames at 2.0, 2.5, 3.0
    assert ref.n_frames == 3
    assert ref.window_hours == (2.0, 3.0)
    assert "2.00-3.00 h" in ref.window_label and "3 frames" in ref.window_label


def test_time_mean_is_an_actual_mean(tmp_path):
    offsets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    _write_frames(tmp_path, n_frames=6, t_step=0.5, values=offsets)
    ref = _load(tmp_path, analysis_hours=1.0)
    # frames 4,5,6 (offsets 4,5,6) => mean offset 5.0 above the base profile
    z = ref.z_les
    expected = 300.0 + 0.003 * z + 5.0
    np.testing.assert_allclose(ref.profiles_les["theta"], expected)


def test_single_frame_window_is_refused(tmp_path):
    _write_frames(tmp_path, n_frames=4, t_step=1.0)      # t = 1..4 h
    with pytest.raises(ValueError, match="need >= 2"):
        _load(tmp_path, analysis_hours=0.1)


def test_missing_profiles_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no LES profile frames"):
        _load(tmp_path)


# --- NaN is fatal -----------------------------------------------------------

@pytest.mark.parametrize("var", ["theta", "wth", "qv"])
def test_non_finite_frame_is_fatal_not_averaged_away(tmp_path, var):
    _write_frames(tmp_path, n_frames=6, corrupt=var)
    with pytest.raises(ValueError, match="non-finite"):
        _load(tmp_path)


def test_missing_scored_variable_raises(tmp_path):
    _write_frames(tmp_path, n_frames=6, drop_var="qv")
    with pytest.raises(ValueError, match="scored variable"):
        _load(tmp_path)


def test_missing_flux_raises(tmp_path):
    _write_frames(tmp_path, n_frames=6, drop_var="wth")
    with pytest.raises(ValueError, match="no 'wth'"):
        _load(tmp_path)


# --- interpolation / masking ------------------------------------------------

def test_mask_excludes_levels_outside_the_les_domain(tmp_path):
    _write_frames(tmp_path, n_frames=6)
    ref = _load(tmp_path)
    assert ref.mask.any()
    assert np.all(ref.z_scm[ref.mask] <= DOMAIN_TOP)
    # everything outside the LES domain is NaN, so ignoring the mask fails loud
    for name in ref.scored_variables():
        assert np.all(np.isnan(ref.profiles[name][~ref.mask]))
        assert np.all(np.isfinite(ref.profiles[name][ref.mask]))


def test_interpolation_reproduces_a_linear_profile_exactly(tmp_path):
    """theta is linear in z, so interpolation onto SCM levels must be exact."""
    _write_frames(tmp_path, n_frames=6)
    ref = _load(tmp_path)
    z = ref.z_scm[ref.mask]
    np.testing.assert_allclose(
        ref.profiles["theta"][ref.mask], 300.0 + 0.003 * z, rtol=1e-12,
    )
    np.testing.assert_allclose(
        ref.profiles["wth"][ref.mask], 0.05 * (1.0 - z / DOMAIN_TOP), rtol=1e-12,
    )


def test_scored_and_diagnostic_variables_are_separated(tmp_path):
    _write_frames(tmp_path, n_frames=6)
    ref = _load(tmp_path)
    assert set(ref.scored_variables()) == set(SCORED_VARIABLES)
    # fluxes are carried but NOT scored
    assert "wth" in ref.profiles and "wth" in DIAGNOSTIC_VARIABLES
    assert "wth" not in ref.scored_variables()


def test_ascending_scm_grid_is_rejected(tmp_path):
    _write_frames(tmp_path, n_frames=6)
    _z, p_half = _scm_grid()
    with pytest.raises(ValueError, match="top-to-bottom"):
        _load(tmp_path, z_scm=np.linspace(20.0, 3000.0, 20), p_half=p_half)


def test_scm_entirely_above_the_les_domain_raises(tmp_path):
    _write_frames(tmp_path, n_frames=6)
    _z, p_half = _scm_grid()
    with pytest.raises(ValueError, match="no SCM level"):
        _load(tmp_path, z_scm=np.linspace(9000.0, 5000.0, 20), p_half=p_half)


def test_inconsistent_frame_variable_sets_are_refused(tmp_path):
    _write_frames(tmp_path, n_frames=6)
    # add a 7th frame missing a variable, inside the window
    prof = tmp_path / "profiles"
    z = np.linspace(25.0, DOMAIN_TOP, NZ_LES)
    np.savez(prof / "prof_006.npz", t_hours=np.asarray(3.5), z=z,
             theta=300.0 + 0.003 * z, qv=np.ones(NZ_LES) * 1e-2,
             u=np.zeros(NZ_LES), v=np.zeros(NZ_LES), wth=np.zeros(NZ_LES))
    with pytest.raises(ValueError, match="different variable set"):
        _load(tmp_path, analysis_hours=1.0)
