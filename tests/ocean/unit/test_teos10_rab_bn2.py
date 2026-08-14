"""NEMO TEOS-10 rab / bn2 — the coefficient set ORCA1 actually runs.

ORCA1's namelist_cfg sets ``ln_teos10 = .true.`` (line 308, with ``ln_eos80``
commented out) and NEMO then sets ``l_useCT = .TRUE.``. Our card had been
computing stratification from a clipped in-situ density gradient, and job
9407791 showed that is the whole remaining mixing-length deficit: swapping
only the N2 that seeds the buoyancy length moved the zero-step ratio against
NEMO's own length from 0.897 to 0.997 (Southern Ocean), 0.914 to 0.998
(tropics) and 0.792 to 1.004 (Arctic).

These tests guard the transcription, which is where a silent wrong number
would enter: 122 coefficients across three families.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _eos():
    from legoesm.ocean import eos
    return eos


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def test_coefficient_families_are_complete():
    """52 density + 35 thermal + 35 haline, matching NEMO's block exactly.

    A missing coefficient would silently evaluate as a KeyError only on the
    branch that reads it, so count them up front.
    """
    c = _eos()._ROQUET_TEOS10
    fams = {}
    for k in c:
        if k[:3] in ("EOS", "ALP", "BET"):
            fams[k[:3]] = fams.get(k[:3], 0) + 1
    assert fams == {"EOS": 52, "ALP": 35, "BET": 35}, fams


def test_normalization_differs_from_eos80_where_nemo_says_it_does():
    """The two easy-to-miss differences, pinned.

    NEMO's TEOS-10 branch sets rdeltaS = 32 (EOS-80 uses 20) and
    r1_S0 = 0.875/35.16504, the Absolute-Salinity scaling (EOS-80 uses 1/40).
    Copying the EOS-80 normalization onto TEOS-10 coefficients would be a
    plausible-looking, entirely wrong EOS.
    """
    t = _eos()._ROQUET_TEOS10
    e = _eos()._ROQUET_EOS80
    assert t["rdeltaS"] == 32.0 and e["rdeltaS"] == 20.0
    assert t["r1_S0"] == pytest.approx(0.875 / 35.16504, rel=1e-15)
    assert e["r1_S0"] == pytest.approx(1.0 / 40.0, rel=1e-15)
    # shared
    assert t["r1_T0"] == pytest.approx(1.0 / 40.0, rel=1e-15)
    assert t["r1_Z0"] == 1.0e-4


def test_coefficients_are_not_the_eos80_set():
    """Non-vacuity: the two tables must actually differ.

    If a copy-paste had left the EOS-80 numbers under the TEOS-10 name every
    other test here would still pass.
    """
    t, e = _eos()._ROQUET_TEOS10, _eos()._ROQUET_EOS80
    shared = [k for k in t if k.startswith("EOS") and k in e]
    assert len(shared) == 52
    differing = [k for k in shared if t[k] != e[k]]
    assert len(differing) == 52, (
        f"only {len(differing)}/52 density coefficients differ between the "
        "TEOS-10 and EOS-80 sets; they should all differ")


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def test_alpha_beta_physical_magnitudes():
    """Sanity-check against known seawater values before anything is built on it.

    At 10 degC, 35 psu, near-surface: thermal expansion is ~1.7e-4 /K and
    haline contraction ~7.6e-4 /psu for seawater. Getting the /zs on beta
    wrong (it is inside NEMO's expression, not a normalization) lands beta
    around 5x off, which this catches.
    """
    import jax.numpy as jnp
    alpha, beta = _eos().nemo_roquet_alpha_beta(
        jnp.array([10.0]), jnp.array([35.0]), jnp.array([0.0]))
    a, b = float(alpha[0]), float(beta[0])
    assert 1.0e-4 < a < 2.5e-4, f"alpha {a:.3e} outside the seawater range"
    assert 7.0e-4 < b < 8.2e-4, f"beta {b:.3e} outside the seawater range"


def test_alpha_grows_with_temperature():
    """Thermal expansion increases with temperature in seawater."""
    import jax.numpy as jnp
    T = jnp.array([0.0, 10.0, 20.0, 30.0])
    S = jnp.full_like(T, 35.0)
    z = jnp.zeros_like(T)
    alpha, _ = _eos().nemo_roquet_alpha_beta(T, S, z)
    a = np.asarray(alpha)
    assert np.all(np.diff(a) > 0), a


def test_bn2_teos10_differs_from_seos_and_is_signed():
    """The wiring is live, and it keeps the signed convention.

    Signed matters: the whole finding is that clipping at zero suppresses the
    long mixing lengths NEMO produces in convectively neutral water.
    """
    import jax.numpy as jnp
    e = _eos()
    nlev = 12
    gdept = jnp.asarray(np.linspace(5.0, 500.0, nlev))
    gdepw = 0.5 * (gdept[:-1] + gdept[1:])
    # stably stratified column with ONE inverted pair
    _t = np.linspace(18.0, 4.0, nlev)
    _t[5] = _t[6] - 0.5                          # cell 5 colder than 6 -> unstable
    T = jnp.asarray(_t)
    S = jnp.full((nlev,), 35.0)
    n2_seos = e.compute_buoyancy_frequency_nemo_bn2(T, S, gdept, gdepw)
    n2_teos = e.compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw, eos_form="teos10")
    a, b = np.asarray(n2_seos), np.asarray(n2_teos)
    assert a.shape == b.shape == (nlev - 1,)
    assert not np.allclose(a, b), "teos10 produced the S-EOS answer"
    # same sign structure: both must see the inverted pair as unstable
    # Interior interface i sits between cell i (upper) and cell i+1 (lower),
    # so inverting cells 5 and 6 makes interface 5 unstable -- not 4. The
    # first revision asserted 4 and failed; the INDEX was wrong, not the code.
    assert b[5] < 0.0, f"teos10 lost the unstable interface: {b[5]:.3e}"
    assert a[5] < 0.0
    # and agree to within a modest factor elsewhere (same physics, better fit)
    stable = np.arange(nlev - 1) != 5
    r = b[stable] / a[stable]
    assert np.all((r > 0.5) & (r < 2.0)), r


def test_unknown_eos_form_raises():
    """Dispatch hardening: a typo must not silently pick the S-EOS."""
    import jax.numpy as jnp
    e = _eos()
    T = jnp.full((4,), 10.0)
    S = jnp.full((4,), 35.0)
    gd = jnp.asarray([5.0, 15.0, 30.0, 50.0])
    gw = jnp.asarray([10.0, 22.0, 40.0])
    with pytest.raises(ValueError, match="eos_form"):
        e.compute_buoyancy_frequency_nemo_bn2(T, S, gd, gw, eos_form="teos-10")


def test_default_stays_seos_bit_identical():
    """The default path must not move: every prior arm scored on it."""
    import jax.numpy as jnp
    e = _eos()
    T = jnp.asarray(np.linspace(15.0, 3.0, 8))
    S = jnp.asarray(np.linspace(34.5, 34.9, 8))
    gd = jnp.asarray(np.linspace(5.0, 300.0, 8))
    gw = 0.5 * (gd[:-1] + gd[1:])
    a = np.asarray(e.compute_buoyancy_frequency_nemo_bn2(T, S, gd, gw))
    b = np.asarray(e.compute_buoyancy_frequency_nemo_bn2(
        T, S, gd, gw, eos_form="seos"))
    assert np.array_equal(a, b)


def test_alpha_beta_is_differentiable():
    """Both are used inside the TKE closure, which must stay grad-safe."""
    import jax
    import jax.numpy as jnp
    e = _eos()

    def f(T):
        alpha, beta = e.nemo_roquet_alpha_beta(
            T, jnp.full_like(T, 35.0), jnp.full_like(T, 100.0))
        return jnp.sum(alpha) + jnp.sum(beta)

    g = jax.grad(f)(jnp.array([5.0, 15.0, 25.0]))
    assert np.all(np.isfinite(np.asarray(g))), g
    assert np.any(np.asarray(g) != 0.0)


def test_alpha_beta_match_the_derivative_of_the_density_polynomial():
    """The decisive check on NEMO's ``zn / zs`` on beta.

    NEMO writes ``pab(jp_sal) = zn / zs * r1_rho0`` with zs = sqrt(scaled S),
    and there is no explicit r1_S0/2 chain-rule factor in that line -- it is
    absorbed into the BET coefficients. If it were NOT absorbed, our beta
    would be off by r1_S0/(2*zs), i.e. a factor of several.

    Rather than reason about it, differentiate the density polynomial these
    coefficients belong to. NEMO's convention is

        alpha = -(1/rho0) d(rho)/dT ,   beta = +(1/rho0) d(rho)/dS

    so a central difference on ``nemo_roquet_eos`` evaluated with the SAME
    TEOS-10 coefficient set must reproduce ``nemo_roquet_alpha_beta``. This
    also catches a wrong normalization constant, since both sides would have
    to be wrong identically to agree.
    """
    import jax.numpy as jnp
    from legoesm import constants
    e = _eos()
    rho0 = e.rho_0
    T0, S0, depth = 10.0, 35.0, 500.0
    p = constants.rho_ocean * constants.g * depth      # nemo_roquet_eos takes Pa

    def rho(T, S):
        return float(e.nemo_roquet_eos(
            jnp.array([T]), jnp.array([S]), jnp.array([p]),
            coeffs=e._ROQUET_TEOS10, rho0=rho0)[0])

    dT, dS = 1.0e-3, 1.0e-3
    alpha_fd = -(rho(T0 + dT, S0) - rho(T0 - dT, S0)) / (2 * dT) / rho0
    beta_fd = (rho(T0, S0 + dS) - rho(T0, S0 - dS)) / (2 * dS) / rho0

    alpha, beta = e.nemo_roquet_alpha_beta(
        jnp.array([T0]), jnp.array([S0]), jnp.array([depth]))
    a, b = float(alpha[0]), float(beta[0])

    assert a == pytest.approx(alpha_fd, rel=2e-4), (
        f"alpha {a:.6e} != d(rho)/dT {alpha_fd:.6e}")
    assert b == pytest.approx(beta_fd, rel=2e-4), (
        f"beta {b:.6e} != d(rho)/dS {beta_fd:.6e} -- the /zs factor on "
        "NEMO's beta line is wrong (dropped, doubled, or double-counted)")
