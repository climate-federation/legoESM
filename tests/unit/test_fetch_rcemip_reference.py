"""``scripts/data/fetch_rcemip_reference.py`` — offline (no-network) tests.

The load-bearing property is the ROUND TRIP: the campaign reads our reference
back by inverting moist static energy for q_v
(``run_scm_rce_campaign._reference_profiles``), so if our MSE reconstruction
and its inversion disagree the reference is silently wrong in q_v while every
file is present and every shape is right.  That is exactly the class of defect
a file-count gate cannot see, so it is tested explicitly here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "data"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fetch_rcemip_reference as frr  # noqa: E402

from legoesm import constants  # noqa: E402

MSE_KJ_TO_J = 1.0e3       # the campaign's own conversion


def _synthetic(nt=40, nz=12):
    """A crude but finite RCE-like column, hourly for ~1.7 days."""
    z = np.linspace(50.0, 20_000.0, nz)
    time_s = np.arange(nt, dtype=float) * 3600.0
    T = 300.0 - 0.0067 * z
    T = np.tile(T, (nt, 1)) + 0.01 * np.arange(nt)[:, None]
    qv = np.tile(0.01865 * np.exp(-z / 4000.0), (nt, 1))
    qcloud = np.tile(1e-4 * np.exp(-((z - 8000.0) / 3000.0) ** 2), (nt, 1))
    cond = qcloud * 1.5
    return dict(T=T, qv=qv, qcloud=qcloud, cond=cond), z, time_s


def test_object_url_targets_the_object_store_not_the_browser():
    """swiftbrowser.dkrz.de returns HTTP 200 + an HTML page for a .nc path, so
    pointing at it yields a 9 KB web page named *.nc and a zero exit code.
    The object endpoint is swift.dkrz.de/v1."""
    u = frr.object_url("SAM_CRM", "RCE_small300", "ta_avg")
    assert u.startswith("https://swift.dkrz.de/v1/")
    assert "swiftbrowser" not in u
    assert u.endswith("/SAM_CRM/RCE_small300/1D/"
                      "SAM_CRM_RCE_small300_1D_ta_avg.nc")


def test_check_netcdf_rejects_an_html_page_and_deletes_it(tmp_path):
    """Non-vacuity for the magic-byte guard: it must FAIL on the exact failure
    that got through before (an HTML page saved as .nc), and must remove the
    file so a rerun cannot pick it up from cache."""
    bad = tmp_path / "ta_avg.nc"
    bad.write_bytes(b"\n<!DOCTYPE html>\n<html><head><title>Swiftbrowser")
    with pytest.raises(SystemExit):
        frr._check_netcdf(bad, "http://example/ta_avg.nc")
    assert not bad.exists(), "a bad download must not survive in the cache"

    empty = tmp_path / "empty.nc"
    empty.write_bytes(b"")
    with pytest.raises(SystemExit):
        frr._check_netcdf(empty, "http://example/empty.nc")

    # ... and PASSES on a real netCDF-3 signature.
    good = tmp_path / "good.nc"
    good.write_bytes(b"CDF\x01" + b"\x00" * 64)
    frr._check_netcdf(good, "http://example/good.nc")
    assert good.exists()


def test_build_reference_writes_the_campaign_layout(tmp_path):
    profs, z, t = _synthetic()
    frr.build_reference(profs, z, t, tmp_path, last_days=1.0, n_snapshots=5,
                        to_mixing_ratio=False, source="unit-test")
    vols = sorted((tmp_path / "snapshots3d").glob("vol_*.npz"))
    sfcs = sorted((tmp_path / "snapshots").glob("sfc_*.npz"))
    assert len(vols) == 5, "campaign requires >= 5 vol_*.npz"
    assert len(sfcs) == 5
    assert (tmp_path / "SOURCE.txt").is_file()
    with np.load(vols[0]) as ds:
        # 3D fields must be (1, 1, nz): the campaign means over axes (0, 1).
        for k in ("T", "mse", "cond", "qcloud", "w"):
            assert ds[k].shape == (1, 1, z.size), (k, ds[k].shape)
        assert ds["z"].shape == (z.size,)
    with np.load(sfcs[-1]) as ds:
        assert {"heights", "T_levels", "qv_levels", "cond_levels"} <= set(ds.files)
        assert ds["heights"].shape == (4,)


def test_mse_roundtrip_recovers_qv_exactly(tmp_path):
    """THE control: invert MSE the way the campaign does and get q_v back.

    ``run_scm_rce_campaign._reference_profiles`` computes
    ``qv = (mse*1e3 - c_pd*T - g*z)/L_v``.  Reproduce that here.
    """
    profs, z, t = _synthetic()
    frr.build_reference(profs, z, t, tmp_path, last_days=1.0, n_snapshots=6,
                        to_mixing_ratio=False, source="unit-test")
    for path in sorted((tmp_path / "snapshots3d").glob("vol_*.npz")):
        with np.load(path) as ds:
            T = np.asarray(ds["T"], dtype=float)
            mse_J = np.asarray(ds["mse"], dtype=float) * MSE_KJ_TO_J
            z_b = np.asarray(ds["z"], dtype=float).reshape(1, 1, -1)
            qv_back = (mse_J - constants.c_pd * T
                       - constants.g * z_b) / constants.L_v
            day = float(ds["day"])
        it = int(round(day * 86400.0 / 3600.0))
        np.testing.assert_allclose(qv_back[0, 0], profs["qv"][it], rtol=1e-12,
                                   atol=1e-15)


def test_mixing_ratio_conversion_changes_qv_by_the_right_amount(tmp_path):
    profs, z, t = _synthetic()
    frr.build_reference(profs, z, t, tmp_path / "spec", last_days=1.0,
                        n_snapshots=5, to_mixing_ratio=False, source="t")
    frr.build_reference(profs, z, t, tmp_path / "mix", last_days=1.0,
                        n_snapshots=5, to_mixing_ratio=True, source="t")

    def _qv(d):
        path = sorted((d / "snapshots3d").glob("vol_*.npz"))[-1]
        with np.load(path) as ds:
            T = np.asarray(ds["T"], float)
            z_b = np.asarray(ds["z"], float).reshape(1, 1, -1)
            return ((np.asarray(ds["mse"], float) * MSE_KJ_TO_J
                     - constants.c_pd * T - constants.g * z_b)
                    / constants.L_v)[0, 0]

    q = _qv(tmp_path / "spec")
    r = _qv(tmp_path / "mix")
    np.testing.assert_allclose(r, q / (1.0 - q), rtol=1e-12)
    # ~1.9% at the surface value 0.01865 -- big enough to matter, small enough
    # that a silent swap would not be caught by eye.
    assert 0.015 < float(r[0] / q[0] - 1.0) < 0.025


def test_build_reference_refuses_a_window_with_too_few_times(tmp_path):
    profs, z, t = _synthetic(nt=6)
    with pytest.raises(SystemExit):
        frr.build_reference(profs, z, t, tmp_path, last_days=0.05,
                            n_snapshots=10, to_mixing_ratio=False, source="t")


def test_build_reference_rejects_non_finite_input(tmp_path):
    profs, z, t = _synthetic()
    profs["T"][-1, 3] = np.nan          # NaN must be FATAL, never nan-reduced
    with pytest.raises(SystemExit):
        frr.build_reference(profs, z, t, tmp_path, last_days=1.0,
                            n_snapshots=5, to_mixing_ratio=False, source="t")


def test_build_reference_rejects_z_profile_mismatch(tmp_path):
    profs, z, t = _synthetic()
    with pytest.raises(SystemExit):
        frr.build_reference(profs, z[:-1], t, tmp_path, last_days=1.0,
                            n_snapshots=5, to_mixing_ratio=False, source="t")


def test_cli_rejects_fewer_than_five_snapshots():
    with pytest.raises(SystemExit):
        frr.main(["--n-snapshots", "3", "--check-size"])


class _FakeTime:
    def __init__(self, values, units):
        self.values = np.asarray(values)
        self.attrs = {"units": units} if units is not None else {}


class _FakeDS:
    def __init__(self, values, units):
        self._t = _FakeTime(values, units)

    def __getitem__(self, k):
        assert k == "time"
        return self._t


def test_time_to_seconds_honours_the_files_units_attribute():
    """RCEMIP 1D files store time in DAYS. Reading them as seconds turns a
    100-day run into 100 s and makes every equilibrium window empty."""
    days = np.array([0.0208333, 50.0, 99.9791667])
    got = frr.time_to_seconds(_FakeDS(days, "day"))
    np.testing.assert_allclose(got, days * 86400.0)
    assert got[-1] / 86400.0 == pytest.approx(99.979, abs=1e-3)
    # hours and seconds also supported
    np.testing.assert_allclose(
        frr.time_to_seconds(_FakeDS(np.array([2.0]), "hours")), [7200.0])
    np.testing.assert_allclose(
        frr.time_to_seconds(_FakeDS(np.array([5.0]), "seconds")), [5.0])


def test_time_to_seconds_refuses_to_guess_unknown_units():
    with pytest.raises(SystemExit):
        frr.time_to_seconds(_FakeDS(np.array([1.0]), "fortnights"))
    with pytest.raises(SystemExit):
        frr.time_to_seconds(_FakeDS(np.array([1.0]), None))


def test_default_humidity_is_the_native_mixing_ratio():
    """QV_avg is a MIXING RATIO (matches our q_v tracer) and is complete;
    hus_avg is specific humidity and is 83% fill for SAM_CRM/RCE_small300."""
    assert frr.HUMIDITY_VARS["qv"] == "QV_avg"
    assert frr.HUMIDITY_VARS["hus"] == "hus_avg"
    assert "hus_avg" not in frr.REQUIRED_VARS


def test_reference_is_written_top_down_like_the_plane_crm(tmp_path):
    """RCEMIP stores z ASCENDING; legoESM plane snapshots store it DESCENDING
    and run_scm_rce_campaign's sigma builder requires that. Writing the source
    order raised 'Derived CRM-matching sigma half-levels are not monotone.'"""
    profs, z_asc, t = _synthetic()
    assert z_asc[0] < z_asc[-1], "fixture must start ASCENDING"
    frr.build_reference(profs, z_asc, t, tmp_path, last_days=1.0,
                        n_snapshots=5, to_mixing_ratio=False, source="t")
    with np.load(sorted((tmp_path / "snapshots3d").glob("vol_*.npz"))[0]) as ds:
        z = np.asarray(ds["z"], float)
        T = np.asarray(ds["T"], float)[0, 0]
    assert np.all(np.diff(z) < 0), "z must be strictly decreasing (top-down)"
    assert z[0] == pytest.approx(z_asc[-1])      # model top first
    assert z[-1] == pytest.approx(z_asc[0])      # surface last
    # The PROFILE must be flipped with it -- a flipped z with an unflipped T is
    # the silent version of this bug.
    assert T[-1] > T[0], "surface must be warmer than the top after flipping"
    np.testing.assert_allclose(T, profs["T"][0][::-1])


def test_reference_already_top_down_is_left_alone(tmp_path):
    profs, z_asc, t = _synthetic()
    z_desc = z_asc[::-1]
    profs_desc = {k: v[:, ::-1] for k, v in profs.items()}
    frr.build_reference(profs_desc, z_desc, t, tmp_path, last_days=1.0,
                        n_snapshots=5, to_mixing_ratio=False, source="t")
    with np.load(sorted((tmp_path / "snapshots3d").glob("vol_*.npz"))[0]) as ds:
        np.testing.assert_allclose(np.asarray(ds["z"], float), z_desc)
        np.testing.assert_allclose(np.asarray(ds["T"], float)[0, 0],
                                   profs_desc["T"][0])
