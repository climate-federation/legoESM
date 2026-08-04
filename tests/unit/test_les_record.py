"""Direct tests for the shared LES frame recorder (``scripts/run/les_record.py``).

The planar-mean profiles this module writes are the LES *reference* that the
SCM turbulence closures are tuned against, so the flux maths is validated here
on synthetic fields with a KNOWN analytic answer before any run quotes it
(CLAUDE.md: validate the instrument before quoting its number).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_les_record():
    path = _ROOT / "scripts" / "run" / "les_record.py"
    spec = importlib.util.spec_from_file_location("les_record", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["les_record"] = mod
    spec.loader.exec_module(mod)
    return mod


les_record = _load_les_record()


def _checkerboard(ny, nx, nz, amp):
    """(ny, nx, nz) field of +amp / -amp on a checkerboard, exact zero mean."""
    j = np.arange(ny)[:, None, None]
    i = np.arange(nx)[None, :, None]
    sign = np.where((i + j) % 2 == 0, 1.0, -1.0)
    return amp * np.ones((ny, nx, nz)) * sign


# --- flux maths on a known analytic case -----------------------------------

def test_wth_recovers_known_kinematic_heat_flux():
    """w' = ±A, theta' = ±B perfectly correlated  =>  <w'theta'> = A*B exactly."""
    ny, nx, nz = 8, 8, 5
    z = np.linspace(10.0, 100.0, nz)
    A, B = 0.5, 0.2                       # m/s, K
    wc3 = _checkerboard(ny, nx, nz, A)
    theta3 = 300.0 + _checkerboard(ny, nx, nz, B)   # mean 300 K exactly
    u3 = np.zeros((ny, nx, nz))
    v3 = np.zeros((ny, nx, nz))

    prof = les_record._profiles(z, u3, v3, wc3, theta3, 0.1, "unit")

    assert "wth" in prof
    np.testing.assert_allclose(prof["theta"], 300.0, atol=1e-12)
    np.testing.assert_allclose(prof["wth"], A * B, rtol=1e-12)


def test_wth_sign_is_positive_up():
    """Warm air moving UP (w'>0 where theta'>0) must give POSITIVE w'theta'.

    z is positive up, so an upward heat flux is positive. Anti-correlating the
    two fields must flip the sign and nothing else.
    """
    ny, nx, nz = 6, 6, 4
    z = np.linspace(10.0, 40.0, nz)
    wc3 = _checkerboard(ny, nx, nz, 0.4)
    theta3 = 290.0 + _checkerboard(ny, nx, nz, 0.25)

    up = les_record._profiles(z, np.zeros_like(wc3), np.zeros_like(wc3),
                              wc3, theta3, 0.1, "unit")
    down = les_record._profiles(z, np.zeros_like(wc3), np.zeros_like(wc3),
                                -wc3, theta3, 0.1, "unit")

    assert np.all(up["wth"] > 0.0)
    np.testing.assert_allclose(down["wth"], -up["wth"], rtol=1e-12)


def test_scalar_flux_matches_known_answer_and_mean():
    """Optional scalars emit both the planar mean and <w'x'>, same convention."""
    ny, nx, nz = 8, 8, 3
    z = np.linspace(10.0, 30.0, nz)
    A, C = 0.3, 1.0e-3
    wc3 = _checkerboard(ny, nx, nz, A)
    qv3 = 1.0e-2 + _checkerboard(ny, nx, nz, C)

    prof = les_record._profiles(z, np.zeros_like(wc3), np.zeros_like(wc3),
                                wc3, np.full((ny, nx, nz), 300.0), 0.1, "unit",
                                scalars={"qv": qv3})

    np.testing.assert_allclose(prof["qv"], 1.0e-2, rtol=1e-12)
    np.testing.assert_allclose(prof["wqv"], A * C, rtol=1e-12)


def test_zero_perturbation_gives_zero_flux():
    """A horizontally uniform field transports nothing."""
    ny, nx, nz = 4, 4, 3
    z = np.linspace(10.0, 30.0, nz)
    wc3 = np.full((ny, nx, nz), 0.7)          # uniform => w' == 0
    theta3 = 300.0 + np.linspace(0.0, 2.0, nz)[None, None, :] * np.ones((ny, nx, 1))

    prof = les_record._profiles(z, np.zeros_like(wc3), np.zeros_like(wc3),
                                wc3, theta3, 0.1, "unit",
                                scalars={"qv": np.full((ny, nx, nz), 5e-3)})

    np.testing.assert_allclose(prof["wth"], 0.0, atol=1e-15)
    np.testing.assert_allclose(prof["wqv"], 0.0, atol=1e-15)


# --- record_frame end-to-end ------------------------------------------------

def _fields(ny=6, nx=6, nz=4):
    rng = np.random.default_rng(0)
    return (rng.normal(size=(ny, nx, nz)),          # u
            rng.normal(size=(ny, nx, nz)),          # v
            rng.normal(size=(ny, nx, nz)),          # w
            300.0 + rng.normal(size=(ny, nx, nz)))  # theta


def test_record_frame_dry_writes_wth_and_no_moist_keys(tmp_path):
    nz = 4
    z = np.linspace(10.0, 40.0, nz)
    u3, v3, wc3, th3 = _fields(nz=nz)
    h_idx, h_z = les_record.select_heights(z, 40.0)

    les_record.record_frame(tmp_path, 0, 0.5, "dry", z, u3, v3, wc3, th3,
                            100.0, 100.0, h_idx, h_z, 0.1)

    prof = np.load(tmp_path / "profiles" / "prof_000.npz", allow_pickle=True)
    assert "wth" in prof.files
    assert "qv" not in prof.files and "wqv" not in prof.files
    # legacy layout preserved
    for key in ("z", "theta", "u", "v", "spd", "wvar", "uu", "vv", "ww",
                "tke", "uw", "vw", "u_star", "z0", "case"):
        assert key in prof.files, key


def test_record_frame_moist_writes_qv_and_fluxes_without_key_collision(tmp_path):
    """qc/qr moved into the shared scalar path; must not collide with the
    cloud_frac extras nor drop the legacy qc/qr profile keys."""
    ny, nx, nz = 6, 6, 4
    z = np.linspace(10.0, 40.0, nz)
    u3, v3, wc3, th3 = _fields(ny, nx, nz)
    rng = np.random.default_rng(1)
    qv3 = 1.0e-2 + 1e-4 * rng.normal(size=(ny, nx, nz))
    qc3 = np.abs(1e-5 * rng.normal(size=(ny, nx, nz)))
    qr3 = np.abs(1e-6 * rng.normal(size=(ny, nx, nz)))
    h_idx, h_z = les_record.select_heights(z, 40.0)

    les_record.record_frame(tmp_path, 2, 1.5, "moist", z, u3, v3, wc3, th3,
                            100.0, 100.0, h_idx, h_z, 0.1,
                            qv3=qv3, qc3=qc3, rho_z=np.ones(nz), qr3=qr3,
                            surface_precip=np.zeros((ny, nx)))

    prof = np.load(tmp_path / "profiles" / "prof_002.npz", allow_pickle=True)
    for key in ("qv", "wqv", "qc", "wqc", "qr", "wqr", "wth", "cloud_frac",
                "surface_precip_mean"):
        assert key in prof.files, key
    np.testing.assert_allclose(prof["qv"], qv3.mean((0, 1)), rtol=1e-12)
    np.testing.assert_allclose(prof["qc"], qc3.mean((0, 1)), rtol=1e-12)
    # cloud fraction is a threshold count, NOT the scalar-mean path
    np.testing.assert_allclose(prof["cloud_frac"], (qc3 > 1.0e-5).mean((0, 1)))

    snap = np.load(tmp_path / "snapshots" / "snap_002.npz", allow_pickle=True)
    for key in ("w", "theta", "u", "v", "qc", "qr", "lwp", "surface_precip"):
        assert key in snap.files, key


@pytest.mark.parametrize("bad", [None])
def test_scalars_none_is_equivalent_to_omitted(bad):
    ny, nx, nz = 4, 4, 3
    z = np.linspace(10.0, 30.0, nz)
    u3, v3, wc3, th3 = _fields(ny, nx, nz)
    a = les_record._profiles(z, u3, v3, wc3, th3, 0.1, "u", scalars=bad)
    b = les_record._profiles(z, u3, v3, wc3, th3, 0.1, "u")
    assert set(a) == set(b)
