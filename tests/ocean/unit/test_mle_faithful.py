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
