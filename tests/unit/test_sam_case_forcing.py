"""SAM CASE forcing-file reader tests (iter-32, C3 data layer).

Parses the SAM ``snd`` / ``lsf`` / ``sfc`` ASCII files used to drive GATE / LBA
/ DYNAMO deep-convection cases and interpolates them onto the legoESM plane-CRM
levels.  The synthetic-file tests are the portable oracle; an extra test
validates against the REAL ``CASES/GATE_IDEAL`` files when the gSAM tree is
present locally.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from legoesm.atmosphere.forcing.sam_case_forcing import (
    extend_sounding_to_top,
    interp_forcing_to_levels,
    interp_sounding_to_levels,
    read_sam_lsf,
    read_sam_sfc,
    read_sam_snd,
    surface_at_day,
    us_standard_atmosphere_temperature,
)
from legoesm import constants


# a sounding topping in the lower stratosphere (16 km) for SND-TOP tests
_SND_STRAT = """ z[m] p[mb] tp[K] q[g/kg] u[m/s] v[m/s]
 0., 3, 1000.  day,levels,pres0
     0.0   1000.0   300.0   16.0   -1.0   0.0
  8000.0    356.0   330.0    1.0   -5.0   0.0
 16000.0    103.0   380.0    0.01  -8.0   0.0
"""


_SND = """ z[m] p[mb] tp[K] q[g/kg] u[m/s] v[m/s]
 0., 3, 1000.  day,levels,pres0
    0.0   1000.0   300.0   16.0   -1.0   0.0
  100.0    988.0   301.0   15.0   -2.0   0.5
  200.0    976.0   302.0   14.0   -3.0   1.0
"""

_LSF = """ z[m] p[mb] tls qls uls vls wls
 0., 2, 1000.  day,levels,pres0
    0.0   1000.0   0.1000E-04   0.2000E-08   -1.0   0.0   -0.010
  100.0    988.0   0.2000E-04   0.3000E-08   -2.0   0.0   -0.020
 1., 2, 1000.  day,levels,pres0
    0.0   1000.0   0.3000E-04   0.4000E-08   -3.0   0.0   -0.030
  100.0    988.0   0.4000E-04   0.5000E-08   -4.0   0.0   -0.040
"""

_SFC = """       day    sst(K)    H(W/m2)   LE(W/m2) TAU(m2/s2)
     0.000    300.00       10.0       50.0       0.10
     1.000    302.00       20.0       60.0       0.20
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_read_snd(tmp_path):
    snd = read_sam_snd(_write(tmp_path, "snd", _SND))
    assert snd.z.shape == (3,)
    assert np.allclose(snd.z, [0.0, 100.0, 200.0])
    assert np.allclose(snd.theta, [300.0, 301.0, 302.0])
    # g/kg → kg/kg conversion
    assert np.allclose(snd.q_v, [0.016, 0.015, 0.014])
    assert np.allclose(snd.u, [-1.0, -2.0, -3.0])
    assert snd.pres0 == pytest.approx(1000.0)


def test_read_lsf_time_blocks(tmp_path):
    lsf = read_sam_lsf(_write(tmp_path, "lsf", _LSF))
    assert lsf.days.shape == (2,)
    assert np.allclose(lsf.days, [0.0, 1.0])
    assert lsf.w_ls.shape == (2, 2)            # (n_time, n_lev)
    assert np.allclose(lsf.w_ls[0], [-0.010, -0.020])
    assert np.allclose(lsf.T_ls[1], [3.0e-5, 4.0e-5])
    assert np.allclose(lsf.qv_ls[0], [2.0e-9, 3.0e-9])


def test_read_sfc(tmp_path):
    sfc = read_sam_sfc(_write(tmp_path, "sfc", _SFC))
    assert np.allclose(sfc.days, [0.0, 1.0])
    assert np.allclose(sfc.sst, [300.0, 302.0])
    assert np.allclose(sfc.lhf, [50.0, 60.0])


def test_truncated_block_raises(tmp_path):
    bad = " hdr\n 0., 5, 1000. day,levels,pres0\n 0.0 1.0 2.0 3.0 4.0 5.0\n"
    with pytest.raises(ValueError, match="truncated"):
        read_sam_snd(_write(tmp_path, "snd_bad", bad))


def test_pressure_coord_sounding_rejected(tmp_path):
    """Decreasing-height (pressure-coordinate) soundings are rejected (D)."""
    bad = (" z p tp q u v\n 0., 2, 1000. day,levels,pres0\n"
           "  200.0 976.0 302.0 14.0 -3.0 1.0\n"
           "  100.0 988.0 301.0 15.0 -2.0 0.5\n")
    with pytest.raises(ValueError, match="STRICTLY INCREASING"):
        read_sam_snd(_write(tmp_path, "snd_p", bad))


def test_rh_sounding_rejected(tmp_path):
    """SND-RH (iter-56): a NEGATIVE q entry is SAM's RELATIVE-HUMIDITY [%]
    convention (forcing.f90:111 ``-q/100·qsat``). Reading it as g/kg would yield
    a NEGATIVE mixing ratio — the silent-units trap class. read_sam_snd must
    REJECT it (GATE/LBA/RCE soundings are all q≥0 g/kg)."""
    bad = (" z p tp q u v\n 0., 2, 1000. day,levels,pres0\n"
           "    0.0 1000.0 300.0 -80.0 -1.0 0.0\n"     # q=-80 ⇒ RH 80%
           "  100.0  988.0 301.0 -75.0 -2.0 0.5\n")
    with pytest.raises(NotImplementedError, match="relative-humidity|RH"):
        read_sam_snd(_write(tmp_path, "snd_rh", bad))


def test_us_standard_atmosphere_anchors():
    """SND-TOP (iter-57): the US Standard Atmosphere 1976 base-layer anchors —
    288.15 K at SL, isothermal 216.65 K through 11-20 km, +1 K/km 20-32 km."""
    z = np.array([0.0, 11000.0, 15000.0, 20000.0, 30000.0])
    T = us_standard_atmosphere_temperature(z)
    np.testing.assert_allclose(T, [288.15, 216.65, 216.65, 216.65, 226.65],
                               rtol=1e-6)


def test_extend_sounding_std_atm_ratio(tmp_path):
    """SND-TOP (iter-57): above the snd top the absolute T follows SAM's US-std-
    atm RATIO ``T(z)=T_top·T_std(z)/T_std(z_top)`` — ISOTHERMAL through 11-20 km
    then WARMING above 20 km (a flat-isothermal fill missed that); θ rises
    (stable), q decays exp(-Δz/3000), in-sounding levels unchanged."""
    snd = read_sam_snd(_write(tmp_path, "snd", _SND_STRAT))   # tops at 16 km
    ext = extend_sounding_to_top(snd, 25000.0, margin=2000.0, dz=250.0)
    np.testing.assert_allclose(ext.z[:snd.z.size], snd.z)     # bottom preserved
    np.testing.assert_allclose(ext.theta[:snd.theta.size], snd.theta)
    assert ext.z[-1] >= 27000.0                               # covers top+margin
    above = ext.z > snd.z[-1]
    z_a = ext.z[above]
    T = ext.theta[above] * (ext.p[above] * 100.0 / constants.p_ref) ** constants.kappa
    T_top = snd.theta[-1] * (snd.p[-1] * 100.0 / constants.p_ref) ** constants.kappa
    # SAM std-atm-ratio property
    T_exp = T_top * (us_standard_atmosphere_temperature(z_a)
                     / us_standard_atmosphere_temperature(np.array([16000.0]))[0])
    np.testing.assert_allclose(T, T_exp, rtol=1e-3)          # hydrostatic-discrete
    # isothermal 16-20 km, then warming above 20 km
    np.testing.assert_allclose(T[z_a < 20000.0], T_top, rtol=2e-3)
    assert T[np.argmin(np.abs(z_a - 26000.0))] > T_top + 2.0  # +1 K/km above 20 km
    assert np.all(np.diff(ext.theta[above]) > 0.0)           # θ rises (stable)
    np.testing.assert_allclose(
        ext.q_v[above], snd.q_v[-1] * np.exp(-(z_a - snd.z[-1]) / 3000.0),
        rtol=1e-10)
    assert np.all(ext.q_v >= 0.0)


def test_extend_sounding_noop_when_covered(tmp_path):
    """A sounding already covering the model top (+margin) is unchanged."""
    snd = read_sam_snd(_write(tmp_path, "snd", _SND_STRAT))   # tops at 16 km
    ext = extend_sounding_to_top(snd, 10000.0, margin=0.0)    # need ≤10 km < 16
    assert ext.z.size == snd.z.size


def test_nonfinite_sounding_rejected(tmp_path):
    """codex iter-56 LOW: a NaN moisture (or any non-finite) value slips through
    the ``q < 0`` / ``θ[0] < 0`` sign checks (both False for NaN) and would
    silently corrupt the IC — read_sam_snd must reject non-finite soundings."""
    bad = (" z p tp q u v\n 0., 2, 1000. day,levels,pres0\n"
           "    0.0 1000.0 300.0 16.0 -1.0 0.0\n"
           "  100.0  988.0 301.0  nan -2.0 0.5\n")
    with pytest.raises(ValueError, match="non-finite|NaN"):
        read_sam_snd(_write(tmp_path, "snd_nan", bad))


def test_omega_lsf_rejected(tmp_path):
    """wls_kind='omega' is explicitly rejected, not silently mis-read (E)."""
    with pytest.raises(NotImplementedError, match="omega|ω"):
        read_sam_lsf(_write(tmp_path, "lsf", _LSF), wls_kind="omega")


def test_lsf_nonmonotonic_days_rejected(tmp_path):
    """Decreasing lsf block days are rejected (F)."""
    bad = (" z p tls qls uls vls wls\n"
           " 1., 1, 1000. day,levels,pres0\n"
           "   0.0 1000.0 1e-5 2e-8 -1.0 0.0 -0.01\n"
           " 0., 1, 1000. day,levels,pres0\n"
           "   0.0 1000.0 2e-5 3e-8 -2.0 0.0 -0.02\n")
    with pytest.raises(ValueError, match="STRICTLY INCREASING"):
        read_sam_lsf(_write(tmp_path, "lsf_bad", bad))


# --------------------------------------------------------------------------
# Vertical + time interpolation
# --------------------------------------------------------------------------

def test_interp_sounding_linear_in_height(tmp_path):
    snd = read_sam_snd(_write(tmp_path, "snd", _SND))
    # model levels top-to-bottom; 150 m and 50 m are midpoints
    z_model = np.array([150.0, 50.0])
    out = interp_sounding_to_levels(snd, z_model)
    assert np.allclose(out["theta"], [301.5, 300.5])
    assert np.allclose(out["q_v"], [0.0145, 0.0155])


def test_interp_sounding_sam_edges(tmp_path):
    """SAM-faithful edges (codex iter-32 A): hold above the top, LINEARLY
    EXTRAPOLATE below the lowest level."""
    snd = read_sam_snd(_write(tmp_path, "snd", _SND))
    out = interp_sounding_to_levels(snd, np.array([500.0, -50.0]))
    assert out["theta"][0] == pytest.approx(302.0)   # above top → hold top
    # below base: slope (301-300)/100 = 0.01 ⇒ 300 + 0.01*(-50) = 299.5
    assert out["theta"][1] == pytest.approx(299.5)


def test_interp_forcing_height_and_time(tmp_path):
    lsf = read_sam_lsf(_write(tmp_path, "lsf", _LSF))
    z_model = np.array([0.0])
    # day=0.5: w_ls block0=-0.010, block1=-0.030 ⇒ time-mid = -0.020
    out = interp_forcing_to_levels(lsf, z_model, day=0.5)
    assert out["w_ls"][0] == pytest.approx(-0.020)
    assert out["T_adv"][0] == pytest.approx(2.0e-5)       # 1e-5 ↔ 3e-5
    assert out["u_ls"][0] == pytest.approx(-2.0)          # -1 ↔ -3


def test_surface_time_interp(tmp_path):
    sfc = read_sam_sfc(_write(tmp_path, "sfc", _SFC))
    s = surface_at_day(sfc, day=0.5)
    assert s["sst"] == pytest.approx(301.0)
    assert s["shf"] == pytest.approx(15.0)
    assert s["lhf"] == pytest.approx(55.0)


# --------------------------------------------------------------------------
# Real gSAM GATE_IDEAL oracle (skipped when the tree is absent)
# --------------------------------------------------------------------------

_GSAM = "/home/gentine/Documents/Code/gSAM/gsam1.8.7/gSAM1.8.7/CASES/GATE_IDEAL"


@pytest.mark.skipif(
    not os.path.isdir(_GSAM), reason="gSAM CASES/GATE_IDEAL not present",
)
def test_real_gate_ideal_files():
    snd = read_sam_snd(os.path.join(_GSAM, "snd"))
    lsf = read_sam_lsf(os.path.join(_GSAM, "lsf"))
    sfc = read_sam_sfc(os.path.join(_GSAM, "sfc"))
    # 33-level GATE_IDEAL sounding
    assert snd.z.shape == (33,)
    assert snd.z[0] == pytest.approx(46.6)
    assert snd.theta[0] == pytest.approx(297.876)
    assert snd.q_v[0] == pytest.approx(0.0165)            # 16.5 g/kg
    assert snd.pres0 == pytest.approx(1012.0)
    # lsf: 2 identical (steady) time blocks
    assert lsf.days.shape == (2,)
    assert np.allclose(lsf.w_ls[0], lsf.w_ls[1])          # steady
    assert lsf.T_ls[0, 0] == pytest.approx(0.1620e-5)
    # constant SST 299.88 K
    assert np.allclose(sfc.sst, 299.88)
    # interpolate onto a coarse model grid (top-to-bottom)
    z_model = np.linspace(15000.0, 50.0, 30)
    fp = interp_forcing_to_levels(lsf, z_model, day=0.0)
    assert fp["w_ls"].shape == (30,)
    assert np.all(np.isfinite(fp["w_ls"]))


# -------- SAM grd vertical-grid reader (VGRID-B) --------

_GRD = """ 25.0 1 50.0
 75.0 2 50.0
 125.0 3 50.0
 175.0 4 50.0
"""   # uniform 50 m, 4 scalar levels (centres)


def test_read_sam_grd_uniform(tmp_path):
    from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_grd
    grd = read_sam_grd(_write(tmp_path, "grd", _GRD))
    assert grd.z_full_bottom_up.shape == (4,)
    np.testing.assert_allclose(grd.z_full_bottom_up, [25.0, 75.0, 125.0, 175.0])
    # interfaces = midpoints of scalar levels, surface=0, top extrapolated;
    # stored TOP-TO-BOTTOM ⇒ [200, 150, 100, 50, 0]
    np.testing.assert_allclose(grd.z_half, [200.0, 150.0, 100.0, 50.0, 0.0],
                               atol=1e-9)
    # cell centre == midpoint of its interfaces
    zi = grd.z_half[::-1]
    np.testing.assert_allclose(0.5 * (zi[:-1] + zi[1:]), grd.z_full_bottom_up)


def test_read_sam_grd_rejects_nonmonotone(tmp_path):
    from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_grd
    bad = " 75.0 1 50.0\n 25.0 2 50.0\n"   # decreasing scalar levels
    with pytest.raises(ValueError, match="strictly increasing"):
        read_sam_grd(_write(tmp_path, "grd_bad", bad))


def test_create_height_coord_from_z_half_rejects_increasing():
    import jax.numpy as jnp
    from legoesm.grids.vertical import create_height_coordinate_from_z_half
    with pytest.raises(ValueError, match="STRICTLY DECREASING"):
        # legoESM needs top-to-bottom (decreasing); this is increasing
        create_height_coordinate_from_z_half(jnp.asarray([0.0, 100.0, 200.0]))


def test_create_height_coordinate_factoring_identity():
    """codex iter-52 B: the uniform `create_height_coordinate` is numerically
    UNCHANGED by the factoring — it equals `create_height_coordinate_from_z_half`
    with a linspace z_half."""
    import jax.numpy as jnp
    from legoesm.grids.vertical import (
        create_height_coordinate, create_height_coordinate_from_z_half)
    n, H = 20, 10000.0
    a = create_height_coordinate(n, H)
    b = create_height_coordinate_from_z_half(jnp.linspace(H, 0.0, n + 1))
    for fld in ("z_full", "z_half", "dz", "dz_half", "rho_ref", "theta_ref",
                "exner_ref", "rho_ref_half", "exner_ref_half"):
        np.testing.assert_array_equal(
            np.asarray(getattr(a, fld)), np.asarray(getattr(b, fld)))


@pytest.mark.skipif(
    not os.path.isdir(_GSAM), reason="gSAM CASES/GATE_IDEAL not present",
)
def test_real_gate_grd_reproduces_sam_grid():
    """The exact SAM GATE grd: 266 levels, dz=50 m BL, dz=100 m UNIFORM through
    the deep-convection layer (5-17 km) — the profile the geometric stretch
    cannot match (codex VGRID-B)."""
    import jax.numpy as jnp
    from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_grd
    from legoesm.grids.vertical import create_height_coordinate_from_z_half
    grd = read_sam_grd(os.path.join(_GSAM, "grd"))
    assert grd.z_full_bottom_up.shape[0] == 266
    assert grd.z_full_bottom_up[0] == pytest.approx(25.0)            # lowest centre
    thick = np.diff(grd.z_half[::-1])                       # cell thicknesses
    np.testing.assert_allclose(thick[:25], 50.0, atol=1.0)  # 50 m BL
    mid = (grd.z_full_bottom_up > 5000.0) & (grd.z_full_bottom_up < 17000.0)
    np.testing.assert_allclose(thick[mid], 100.0, atol=1.0)  # 100 m deep conv
    # the custom-level builder reproduces it (266-level HeightCoordinate)
    hc = create_height_coordinate_from_z_half(jnp.asarray(grd.z_half))
    assert hc.n_levels == 266
    assert float(hc.z_full[-1]) == pytest.approx(25.0, abs=1.0)
    assert bool(np.all(np.isfinite(np.asarray(hc.rho_ref))))


# -------- prescribed radiative forcing (rad file, LBA) --------

_RAD = """ p[mb] or z(m)   (dt/dt)rad [K/s]
 0., 3
    0.0   -0.1600E-04
 5000.0   -0.2000E-04
20000.0    0.0000E+00
 1., 3
    0.0   -0.1000E-04
 5000.0   -0.1500E-04
20000.0    0.0000E+00
"""


def test_read_sam_rad(tmp_path):
    from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_rad
    rad = read_sam_rad(_write(tmp_path, "rad", _RAD))
    assert rad.days.shape == (2,)
    assert rad.dTdt_rad.shape == (2, 3)
    assert rad.dTdt_rad[0, 0] == pytest.approx(-1.6e-5)
    assert rad.dTdt_rad[0, 2] == pytest.approx(0.0)         # zero in stratosphere


def test_interp_rad_to_levels(tmp_path):
    from legoesm.atmosphere.forcing.sam_case_forcing import (
        read_sam_rad, interp_rad_to_levels)
    rad = read_sam_rad(_write(tmp_path, "rad", _RAD))
    # z=2500 (midpoint 0..5000) at day=0: between -1.6e-5 and -2.0e-5 ⇒ -1.8e-5
    out = interp_rad_to_levels(rad, np.array([2500.0]), day=0.0)
    assert out[0] == pytest.approx(-1.8e-5, rel=1e-6)
    # day=0.5: block0 -1.6e-5, block1 -1.0e-5 at z=0 ⇒ time-mid -1.3e-5
    out2 = interp_rad_to_levels(rad, np.array([0.0]), day=0.5)
    assert out2[0] == pytest.approx(-1.3e-5, rel=1e-6)
