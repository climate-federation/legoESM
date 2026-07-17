"""Oracle-faithfulness pins for the Fox-Kemper MLE closed forms.

Oracle: NEMO 5.0.1 ``src/OCE/TRA/tramle.F90`` (ORCA1 ``ln_mle=.true., nn_mle=1``,
``rn_ce=0.06``, ``rn_lat=20``) and Fox-Kemper, Ferrari & Hallberg (2008) JPO 38,
1145-1165.  The scheme's CLOSED FORMS in
``legoesm.ocean.physics.lateral_mixing.mle`` are pinned to round-off (rel 1e-12)
against an INDEPENDENT numpy reimplementation whose fixed coefficients are typed
from the NEMO source (NOT read back from the module — non-circular) and
separately canaried against the shipped module constants:

  MLE coefficient        rc_f = rn_ce / (5 km · f0),  f0 = 2Ω sin(rn_lat)     (nn_mle=1)
  face streamfunction    ψ_m  = rc_f · H² · face_width · dbm_face · cap        [m³/s]
  vertical structure     μ(z) = max(0, (1−ζ²)(1 + 5/21 ζ²)),  ζ = 1 − 2·gdepw/H
  face MLD               min / avg / max of the two neighbour MLDs (raises on unknown)

The 5/21 vertical-structure factor and the 5 km ``rc_f`` normalising length are
the NEMO fixed constants (``rn_ce`` = MLEConfig.ce is the only tunable knob); both
are canaried so a config/constant drift fails.  Complements ``test_mle.py``
(behavioral: μ endpoints/peak, positivity, one rc_f point, grad).

SCOPE — the ``mle_mld_and_buoyancy`` Δρ mixed-layer detection is DEFERRED to a
separate audit (see the mle number-budget follow-up memory).  A codex review
surfaced that its NEMO-faithfulness is NOT yet clean: the reference-level pick
``searchsorted(z_centers, ref_depth)`` selects the first centre BELOW ref_depth,
contradicting both the module's own "at/above the ref depth" comment and NEMO's
``nla10`` (the level ABOVE the interface just below ~10 m) — a SUSPECTED
off-by-one bug that needs the tramle.F90 source to confirm + ocean validation to
fix; and ``bm`` uses legoESM ``g``/``rho_ocean``, not NEMO's ``g_nemo``/``rau0``
(a ~5e-5 relative constant departure).  Pinning it here against a numpy
restatement of the MODULE would mask that suspected defect, so it is left out
until resolved against the NEMO source.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ocean.physics.lateral_mixing import mle
from legoesm.ocean.physics.lateral_mixing.mle import (
    face_mld,
    mle_coefficient,
    mle_mld_and_buoyancy,
    mle_streamfunction_magnitude,
    mle_vertical_structure,
)

from legoesm import constants

jax.config.update("jax_enable_x64", True)

# --- NEMO tramle.F90 fixed constants (typed from source; canaried below) ---
_O_R5_21 = 5.0 / 21.0        # vertical-structure factor in mu(z)
_O_RC_F_LENGTH_M = 5.0e3     # NEMO "5.e3" normalising length in rc_f


# ------------------------- independent numpy oracle -------------------------

def _rc_f_oracle(ce, lat_deg):
    f0 = 2.0 * float(constants.Omega) * math.sin(math.radians(lat_deg))
    return ce / (_O_RC_F_LENGTH_M * f0)


def _mu_oracle(gdepw_over_H):
    zeta = 1.0 - 2.0 * np.asarray(gdepw_over_H, dtype=float)
    z2 = zeta * zeta
    return np.maximum(0.0, (1.0 - z2) * (1.0 + _O_R5_21 * z2))


def _psim_oracle(rc_f, H, width, dbm, cap):
    return rc_f * H * H * width * dbm * cap


# ===================== constant canaries (non-circular) =====================

def test_module_constants_match_nemo():
    assert mle._R5_21 == pytest.approx(_O_R5_21, rel=1e-15)
    assert mle._RC_F_LENGTH_SCALE_M == _O_RC_F_LENGTH_M
    assert mle.MLEConfig().ce == 0.06            # ORCA1 rn_ce
    assert mle.MLEConfig().lat_ref_deg == 20.0   # ORCA1 rn_lat
    assert float(constants.Omega) == pytest.approx(7.292e-5, rel=1e-12)  # const-ok: canaried Omega


# ===================== MLE coefficient rc_f =====================

@pytest.mark.parametrize("ce", [0.06, 0.033, 0.12, 0.3])
@pytest.mark.parametrize("lat", [10.0, 20.0, 45.0, 89.0, -30.0])
def test_mle_coefficient_matches_nemo(ce, lat):
    got = float(mle_coefficient(ce, lat))
    assert got == pytest.approx(_rc_f_oracle(ce, lat), rel=1e-12)


def test_mle_coefficient_length_scale_canary():
    # rc_f uses the NEMO 5 km normalising length, not 10 km: a wrong scale halves
    # rc_f (proves the pin canaries _RC_F_LENGTH_SCALE_M, not just proportionality).
    got = float(mle_coefficient(0.06, 20.0))
    assert got == pytest.approx(_rc_f_oracle(0.06, 20.0), rel=1e-12)
    wrong = 0.06 / (10.0e3 * 2.0 * float(constants.Omega) * math.sin(math.radians(20.0)))
    assert abs(got - wrong) / got > 0.4


def test_mle_coefficient_zero_coriolis_guard_raises():
    # |sin(lat)| ~ 0 at 0, +/-180 deg ⇒ f0→0 ⇒ raise (not a huge finite artefact).
    for bad in (0.0, 0.5, -0.9, 180.0, -180.0):
        with pytest.raises(ValueError):
            mle_coefficient(0.06, bad)


# ===================== vertical structure mu(z) =====================

def test_vertical_structure_full_curve_matches_nemo():
    frac = jnp.linspace(0.0, 1.0, 41)
    got = np.asarray(mle_vertical_structure(frac))
    assert np.allclose(got, _mu_oracle(frac), rtol=1e-12, atol=1e-12)


def test_vertical_structure_5_21_coefficient_canary():
    # The 5/21 factor is the NEMO value; 1/3 (a plausible transcription) shifts
    # mu measurably at mid-column (proves the pin canaries _R5_21).
    frac = np.array([0.25, 0.5, 0.75])
    got = np.asarray(mle_vertical_structure(jnp.asarray(frac)))
    zeta = 1.0 - 2.0 * frac
    z2 = zeta * zeta
    wrong = np.maximum(0.0, (1.0 - z2) * (1.0 + (1.0 / 3.0) * z2))
    assert np.allclose(got, _mu_oracle(frac), rtol=1e-12)
    assert np.max(np.abs(got - wrong)) > 1e-3


def test_vertical_structure_clamped_outside_mixed_layer():
    # gdepw/H outside [0,1] ⇒ ζ outside [−1,1] ⇒ (1−ζ²)<0 ⇒ max(0,·) clamps to 0.
    frac = jnp.asarray([-0.2, -0.01, 1.01, 1.5])
    got = np.asarray(mle_vertical_structure(frac))
    assert np.allclose(got, 0.0, atol=1e-12)
    # and it is exactly 0 (not just small) at ζ=±1 endpoints
    assert float(mle_vertical_structure(jnp.asarray(0.0))) == pytest.approx(0.0, abs=1e-12)
    assert float(mle_vertical_structure(jnp.asarray(1.0))) == pytest.approx(0.0, abs=1e-12)


# ===================== face streamfunction magnitude =====================

@pytest.mark.parametrize("rc_f,H,width,dbm,cap", [
    (1.2e-3, 120.0, 5.0e4, 3.0e-8, 1.0e5),
    (5.0e-4, 40.0, 2.5e4, -1.0e-8, 4.0e4),
    (2.0e-3, 300.0, 1.0e5, 7.0e-9, 1.11e5),
])
def test_streamfunction_magnitude_matches_product(rc_f, H, width, dbm, cap):
    got = float(mle_streamfunction_magnitude(
        rc_f, jnp.asarray(H), jnp.asarray(width), jnp.asarray(dbm), jnp.asarray(cap)))
    assert got == pytest.approx(_psim_oracle(rc_f, H, width, dbm, cap), rel=1e-12)


def test_streamfunction_scales_as_H_squared():
    # The FK H² scaling: doubling the ML depth quadruples ψ_m.
    base = float(mle_streamfunction_magnitude(
        1.2e-3, jnp.asarray(100.0), jnp.asarray(5.0e4), jnp.asarray(3.0e-8), jnp.asarray(1.0e5)))
    dbl = float(mle_streamfunction_magnitude(
        1.2e-3, jnp.asarray(200.0), jnp.asarray(5.0e4), jnp.asarray(3.0e-8), jnp.asarray(1.0e5)))
    assert dbl == pytest.approx(4.0 * base, rel=1e-12)


# ===================== face MLD dispatch =====================

def test_face_mld_modes_and_unknown_raises():
    a = jnp.asarray([10.0, 20.0, 50.0])
    b = jnp.asarray([30.0, 5.0, 50.0])
    assert np.allclose(np.asarray(face_mld(a, b, "min")), np.minimum(np.asarray(a), np.asarray(b)))
    assert np.allclose(np.asarray(face_mld(a, b, "avg")), 0.5 * (np.asarray(a) + np.asarray(b)))
    assert np.allclose(np.asarray(face_mld(a, b, "max")), np.maximum(np.asarray(a), np.asarray(b)))
    with pytest.raises(ValueError):
        face_mld(a, b, "median")


# ===================== MLD + buoyancy (NEMO tramle.F90 / domzgr.F90) ========
#
# Previously DEFERRED (see the mle-mld follow-up memory): pinning against the
# module would have masked a suspected reference-level off-by-one.  Resolved
# against NEMO 5.0.1 source, quoted here so the pins are NON-CIRCULAR:
#
#   domzgr.F90:  zrefdep = 10._wp - 0.1_wp * MINVAL( e3w_1d )
#                nlb10 = MINLOC( gdepw_1d, mask = gdepw_1d > zrefdep, dim = 1 )
#                nla10 = nlb10 - 1
#   tramle.F90:  inml_mle(ji,jj) = mbkt(ji,jj) + 1
#                DO_3DS( ..., jpkm1, nlb10, -1 )
#                   IF( rhop(ji,jj,jk) > rhop(ji,jj,nla10) + rn_rho_c_mle )
#                      inml_mle(ji,jj) = jk
#                zc  = e3t * MIN( MAX( 0, inml_mle-jk ), 1 )
#                zmld += zc ;  zbm += zc*(rho0 - rhop)*r1_rho0
#                zbm = grav * zbm / MAX( e3t(1), zmld )
#
# So nla10 is the T-level CONTAINING ~10 m (selected from W-interfaces with
# the 0.1*min(e3w) tolerance), and the density scan runs over jk >= nlb10
# ONLY.  The old module code picked the first level CENTRE at-or-below 10 m
# and gated the scan on centre depth — both pinned wrong below.


def _mld(rho_row, dz_row, wet_row, z_faces, **kw):
    rho = jnp.asarray(rho_row).reshape(1, 1, -1)
    dz = jnp.asarray(dz_row).reshape(1, 1, -1)
    wet = jnp.asarray(wet_row).reshape(1, 1, -1)
    zmld, bm, in_ml = mle_mld_and_buoyancy(
        rho, dz, wet, z_faces=jnp.asarray(z_faces),
        rho_c_mle=0.01, ref_depth_m=10.0,
        rho0=constants.rho_ocean, grav=constants.g, **kw)
    return float(zmld[0, 0]), float(bm[0, 0]), np.asarray(in_ml[0, 0])


def test_nla10_is_the_level_containing_ten_metres_not_the_next_centre():
    """Codex counterexample resolved against domzgr.F90: interfaces
    [0,15,50,100] (centres 5,25,75) -> nla10 = the 5 m level (its cell spans
    0-15 m and CONTAINS 10 m).  The old centre-searchsorted picked the 25 m
    level, whose rho_ref here (1024.5) shifts the threshold enough to keep
    level 1 inside the mixed layer -> zmld 50 instead of NEMO's 15."""
    zmld, _, in_ml = _mld(
        [1024.0, 1024.5, 1026.0], [15.0, 35.0, 50.0], [1, 1, 1],
        [0.0, 15.0, 50.0, 100.0])
    # rho_ref = level 0 (1024.0): level 1 already exceeds +0.01 -> ML = {0}.
    assert zmld == pytest.approx(15.0, rel=1e-12)
    assert in_ml.tolist() == [1.0, 0.0, 0.0]


def test_nla10_uses_interfaces_not_last_centre_at_or_above():
    """Interfaces [0,8,30,60] (centres 4,19,45): 10 m lies INSIDE level 1
    (8-30 m), so NEMO references level 1 even though its CENTRE (19 m) is
    below 10 m.  A 'last centre above 10 m' rule would pick level 0 — whose
    rho here (1026) makes nothing exceed and the whole column mixed."""
    zmld, _, in_ml = _mld(
        [1026.0, 1024.0, 1024.5], [8.0, 22.0, 30.0], [1, 1, 1],
        [0.0, 8.0, 30.0, 60.0])
    # rho_ref = level 1 (1024.0): level 2 exceeds -> ML = {0, 1} -> 8+22 m.
    assert zmld == pytest.approx(30.0, rel=1e-12)
    assert in_ml.tolist() == [1.0, 1.0, 0.0]


def test_zrefdep_tolerance_keeps_an_exact_ten_metre_interface_shallow():
    """domzgr's zrefdep = 10 - 0.1*min(e3w): an interface at EXACTLY 10 m
    counts as 'deeper than zrefdep', so the level ABOVE it is the reference.
    Without the tolerance the strict > 10 mask would skip that interface and
    reference one level deeper (threshold from 1025 -> zmld 20, not 10)."""
    zmld, _, _ = _mld(
        [1024.0, 1025.0, 1026.0], [10.0, 10.0, 20.0], [1, 1, 1],
        [0.0, 10.0, 20.0, 40.0])
    assert zmld == pytest.approx(10.0, rel=1e-12)


def test_density_scan_starts_below_nla10():
    """NEMO scans jk = jpkm1..nlb10 ONLY: a dense spike AT/ABOVE the
    reference level must not truncate the mixed layer.  Interfaces
    [0,8,30,60] -> nla10 = level 1; level 0 is far denser than rho_ref +
    rho_c but sits above the scan window, and nothing below exceeds -> the
    whole wet column is mixed (inml_mle stays at mbkt+1)."""
    zmld, _, in_ml = _mld(
        [1030.0, 1024.0, 1024.005], [8.0, 22.0, 30.0], [1, 1, 1],
        [0.0, 8.0, 30.0, 60.0])
    assert zmld == pytest.approx(60.0, rel=1e-12)
    assert in_ml.tolist() == [1.0, 1.0, 1.0]


def test_no_exceed_partial_wet_column_mixes_to_the_bottom():
    """tramle's inml_mle = mbkt+1 initialization: nothing exceeding ->
    every WET level is mixed (top-contiguous wet columns)."""
    zmld, _, in_ml = _mld(
        [1024.0, 1024.0, 1024.0], [10.0, 10.0, 20.0], [1, 1, 0],
        [0.0, 10.0, 20.0, 40.0])
    assert zmld == pytest.approx(20.0, rel=1e-12)
    assert in_ml.tolist() == [1.0, 1.0, 0.0]


def test_bm_matches_the_nemo_formula_under_nemo_constants():
    """zbm = grav * sum_ML[e3t*(rho0-rhop)/rho0] / max(e3t(1), zmld), computed
    here with grav = constants.g_nemo for exact NEMO parity (the production
    default constants.g is the SANCTIONED legoESM departure, ~5e-5)."""
    rho = jnp.asarray([1020.0, 1022.0, 1026.0]).reshape(1, 1, 3)
    dz = jnp.asarray([10.0, 15.0, 30.0]).reshape(1, 1, 3)
    wet = jnp.ones((1, 1, 3))
    faces = jnp.asarray([0.0, 10.0, 25.0, 55.0])
    rho0 = float(constants.rho_ocean)
    zmld, bm, _ = mle_mld_and_buoyancy(
        rho, dz, wet, z_faces=faces, rho_c_mle=0.01, ref_depth_m=10.0,
        rho0=rho0, grav=constants.g_nemo)
    # nla10 = level 0 (interface at exactly 10 m stays shallow); level 1
    # exceeds 1020.01 -> ML = {0}; zmld = 10; zbm over level 0 only.
    assert float(zmld[0, 0]) == pytest.approx(10.0, rel=1e-12)
    expected = (constants.g_nemo
                * (10.0 * (rho0 - 1020.0) / rho0)
                / max(10.0, 10.0))
    assert float(bm[0, 0]) == pytest.approx(expected, rel=1e-12)


def test_e3w_surface_element_is_the_doubled_half_cell():
    """NEMO e3w_1d(1) = 2*gdept_1d(1) (surface half-cell doubled — the same
    reconstruction the PGF uses), NOT gdept_1d(1) itself (codex round 2).
    Faces [0, 100, 200, 300, 400]: midpoint centres give e3w(1) = 100, so
    zrefdep = 0 and the interface at exactly... rather: with the UNdoubled
    surface element zrefdep would be 10 - 5 = 5 here on faces [0, 10, ...]
    ladders.  Pinned via the discriminating ladder below instead; this case
    just locks the doubled element on a coarse uniform grid where zrefdep
    collapses to 0 and nla10 = level 0."""
    rho = jnp.asarray([1024.0, 1025.0, 1026.0, 1026.5]).reshape(1, 1, 4)
    dz = jnp.full((1, 1, 4), 100.0)
    wet = jnp.ones((1, 1, 4))
    faces = jnp.asarray([0.0, 100.0, 200.0, 300.0, 400.0])
    zmld, _, _ = mle_mld_and_buoyancy(
        rho, dz, wet, z_faces=faces, rho_c_mle=0.01, ref_depth_m=10.0,
        rho0=constants.rho_ocean, grav=constants.g)
    # nla10 = level 0; level 1 exceeds -> ML = {0}.
    assert float(zmld[0, 0]) == pytest.approx(100.0, rel=1e-12)


def test_exact_gdept_changes_the_tolerance_hence_nla10():
    """Partial-cell ladders (NEMO gdept_1d != face midpoints): e3w_1d — and
    through the 0.1*min(e3w) tolerance possibly nla10 itself — must come from
    the EXACT T-depths when the coordinate carries them (codex P1), with the
    NEMO surface element e3w(1) = 2*gdept(1).

    Faces [0, 5, 9.7, 20, 40], exact gdept [4, 5.5, 15, 30]:
    e3w = [8, 1.5, 9.5, 15] -> zrefdep = 9.85 -> the 9.7 m face stays
    SHALLOW -> nla10 = level 2.  Midpoint centres [2.5, 7.35, 14.85, 30]:
    e3w = [5, 4.85, 7.5, 15.15] -> zrefdep = 9.515 -> the 9.7 m face is
    DEEP -> nla10 = level 1.  Only the exact-gdept answer is NEMO's on this
    ladder."""
    rho = jnp.asarray([1030.0, 1024.0, 1024.3, 1024.35]).reshape(1, 1, 4)
    dz = jnp.asarray([5.0, 4.7, 10.3, 20.0]).reshape(1, 1, 4)
    wet = jnp.ones((1, 1, 4))
    faces = jnp.asarray([0.0, 5.0, 9.7, 20.0, 40.0])
    gdept = jnp.asarray([4.0, 5.5, 15.0, 30.0])
    # Exact gdept: nla10 = level 2 (rho 1024.3); level 3 (1024.35) exceeds
    # +0.01 -> ML = {0, 1, 2} -> zmld = 20.
    zmld_exact, _, _ = mle_mld_and_buoyancy(
        rho, dz, wet, z_faces=faces, z_centers_ref=gdept,
        rho_c_mle=0.01, ref_depth_m=10.0,
        rho0=constants.rho_ocean, grav=constants.g)
    assert float(zmld_exact[0, 0]) == pytest.approx(20.0, rel=1e-12)
    # Midpoint fallback: nla10 = level 1 (rho 1024.0); level 2 exceeds ->
    # ML = {0, 1} -> zmld = 9.7.  (NEMO's own answer only when gdept
    # genuinely sits at the midpoints.)
    zmld_mid, _, _ = mle_mld_and_buoyancy(
        rho, dz, wet, z_faces=faces,
        rho_c_mle=0.01, ref_depth_m=10.0,
        rho0=constants.rho_ocean, grav=constants.g)
    assert float(zmld_mid[0, 0]) == pytest.approx(9.7, rel=1e-12)
