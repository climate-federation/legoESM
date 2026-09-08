"""#1455 basin-budget lane: the southern-basin decomposition probe's own arithmetic.

``basin_seasonal_decomp.py`` imports its transports, geometry, masks and state
loaders from the recorded harness.  The arithmetic it OWNS is three reductions
of a velocity field over an arbitrary row slice — the bottom-referenced
barotropic/baroclinic split, the per-T-row profile, and the transport moments
with their per-level profile — plus the pre-run gates that decide whether any
of its numbers may be read.  Those are what is tested here.

The probe's own ``--self-check`` asserts the same invariants at run time; it is
invoked directly so a regression fails in CI rather than in the middle of a
campaign.  The oracle's ``mesh_mask.nc`` and the verdict members are required
(the probe's geometry is the oracle's), so the module is skipped where they are
absent rather than mocked — a mocked mesh would test different arithmetic from
the one the campaign runs.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "scripts" / "validate" / "ocean_fidelity" / "dino_1226"


@pytest.fixture(scope="module")
def P():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(PROBE_DIR))
    sys.path.insert(0, str(PROBE_DIR.parent))
    try:
        try:
            import basin_seasonal_decomp as B
        except (OSError, SystemExit, FileNotFoundError) as exc:
            pytest.skip(f"oracle geometry not available on this host: {exc}")
        return B
    finally:
        for p in (str(PROBE_DIR), str(PROBE_DIR.parent)):
            try:
                sys.path.remove(p)
            except ValueError:
                pass


@pytest.fixture(scope="module")
def u_random(P):
    import acc_thermal_wind as A
    return np.random.default_rng(4242).normal(size=A.umask.shape) * 0.05


def test_bottom_referenced_split_reproduces_the_recorded_band_split(P, u_random):
    """The generalisation must not have drifted from the split the campaign's
    recorded channel-band numbers were produced with."""
    import acc_thermal_wind as A
    band = slice(A.J0, A.J1 + 1)
    bc_r, bt_r = P.bc_bt_rows(u_random, A.umask, band)
    bc_b, bt_b = A.bc_bt_band(u_random, A.umask)
    assert np.allclose(bc_r, bc_b, rtol=0, atol=1e-12)
    assert np.allclose(bt_r, bt_b, rtol=0, atol=1e-12)


def test_bottom_referenced_split_is_exact_on_the_southern_rows(P, u_random):
    import acc_thermal_wind as A
    south = slice(0, A.J0)
    bc, bt = P.bc_bt_rows(u_random, A.umask, south)
    direct = np.einsum(
        "jik,j->i",
        np.where(A.umask[south], u_random[south] * A.e3t0[south], 0.0),
        np.asarray(A.e2u_col, dtype=np.float64)[south]) / 1e6
    assert np.allclose(bc + bt, direct, rtol=0, atol=1e-10)


def test_per_row_profile_sums_to_the_group_transport(P, u_random):
    import acc_driver_decomp as D
    import acc_thermal_wind as A
    south = slice(0, A.J0)
    rows = P.row_transport(u_random, A.umask, south)
    assert rows.shape == (A.J0,)
    assert float(rows.sum()) == pytest.approx(
        float(D._avg(D.group_transport(u_random, A.umask, south))), abs=1e-10)


def test_per_level_profile_sums_to_the_transport_it_decomposes(P, u_random):
    """Guards the reduction that turns a per-level array into the same longitude
    mean the transport uses: a divisor of NX instead of NX-4 would scale every
    per-level number by 52/48 and no other assertion here would notice."""
    import acc_driver_decomp as D
    import acc_thermal_wind as A
    south = slice(0, A.J0)
    m0, m1, lev = P.transport_moments(u_random, A.umask, south)
    assert lev.shape == (A.NZ,)
    assert float(lev.sum()) == pytest.approx(float(D._avg(m0)), abs=1e-12)
    # the first moment is a depth-weighted version of the SAME integrand, so a
    # uniform velocity puts the centroid at the transport-weighted mean depth
    ones = np.ones_like(u_random)
    m0u, m1u, _ = P.transport_moments(ones, A.umask, south)
    z = float(D._avg(m1u) / D._avg(m0u))
    assert 0.0 < z < float(A.gdept1d[-1])


def test_first_moment_is_not_the_complement_of_the_transport(P, u_random):
    """The bottom-referenced pair is algebraically one number; (M0, M1) is not.
    If a refactor ever made M1 a function of M0 alone this fails."""
    import acc_thermal_wind as A
    south = slice(0, A.J0)
    m0a, m1a, _ = P.transport_moments(u_random, A.umask, south)
    scaled = u_random.copy()
    scaled[..., A.NZ // 2:] *= 2.0          # deepen the transport, same sign
    m0b, m1b, _ = P.transport_moments(scaled, A.umask, south)
    ra, rb = float(np.sum(m1a) / np.sum(m0a)), float(np.sum(m1b) / np.sum(m0b))
    assert abs(rb - ra) > 1.0               # the centroid moved by metres


def test_probe_self_checks_pass(P):
    """The gates the probe refuses to print a number without."""
    assert P.self_checks(verbose=False) is True
