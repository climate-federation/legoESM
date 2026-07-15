"""Oracle-faithfulness tests for the Louis (1979)/LTG82 stability functions.

Oracle: the PUBLISHED closed forms of Louis (1979), Boundary-Layer Meteorol. 17,
187-202 (Eqs. 19-20), with the momentum/heat coefficient split of Louis, Tiedtke
& Geleyn (1982). These pin ``physics._shared.louis_stability_functions`` against
INDEPENDENTLY WRITTEN expressions of the paper equations — distinct from
``test_turbulence_shared_helpers.py``, whose ``_louis_reference`` is a copy of the
OLD louis.py code block (a dedup-regression lock, not an oracle check).

FAITHFUL forms (stable/unstable branch + LTG82 b-split) are pinned where the
sigmoid blend collapses to a single branch; DEPARTURES (c=16.6 Holtslag&De Bruin
1988 vs Louis 5.0; the smooth blend replacing the hard Ri switch) are pinned as
canaries so a silent change trips them.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from legoesm.atmosphere.physics._shared import louis_stability_functions
from legoesm.atmosphere.physics.turbulence.config import LouisConfig

# Louis (1979) originals / LTG82 split (the oracle constants).
LOUIS1979_B = 5.0
LOUIS1979_C = 5.0
LOUIS1979_D = 5.0
LTG82_B_HEAT_RATIO = 1.5
# Holtslag & De Bruin (1988) unstable-branch c used as our default.
HDB1988_C = 16.6
SHARP = 100.0  # blend sharpness; sigmoid(100*|Ri|>=0.3) is 1 (or 0) to ~1e-13


@pytest.fixture(autouse=True)
def _enable_x64():
    """Numeric faithfulness checks in float64 (restored after)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _call(Ri, l_mix, dz, b, c, d, sharp=SHARP, b_heat=None):
    fm, fh = louis_stability_functions(
        np.asarray(Ri, dtype=np.float64),
        np.asarray(l_mix, dtype=np.float64),
        np.asarray(dz, dtype=np.float64),
        b, c, d, sharp, b_heat=b_heat,
    )
    return np.asarray(fm), np.asarray(fh)


# ===========================================================================
# FAITHFUL: the published closed forms
# ===========================================================================
def test_faithful_stable_branch_is_louis1979_eq20():
    """f = 1/(1 + 2b*Ri/sqrt(1+d*Ri)) at stable Ri (blend -> stable branch)."""
    Ri = np.array([0.3, 0.7, 1.5, 4.0])  # sigmoid(100*Ri) == 1 to ~1e-13
    l_mix = np.full_like(Ri, 30.0)
    dz = np.full_like(Ri, 50.0)
    fm, fh = _call(Ri, l_mix, dz, LOUIS1979_B, HDB1988_C, LOUIS1979_D,
                   b_heat=LTG82_B_HEAT_RATIO * LOUIS1979_B)
    expected_m = 1.0 / (1.0 + 2.0 * LOUIS1979_B * Ri / np.sqrt(1.0 + LOUIS1979_D * Ri))
    b_h = LTG82_B_HEAT_RATIO * LOUIS1979_B
    expected_h = 1.0 / (1.0 + 2.0 * b_h * Ri / np.sqrt(1.0 + LOUIS1979_D * Ri))
    assert np.allclose(fm, expected_m, rtol=1e-9, atol=1e-12)
    assert np.allclose(fh, expected_h, rtol=1e-9, atol=1e-12)


def test_faithful_unstable_branch_is_louis1979_eq19_interior():
    """f = 1 - 2b*Ri/(1 + 3bc*l^2*sqrt|Ri|/dz^2) at unstable Ri (blend->unstable).

    The interior geometry factor C = 3bc*l^2/dz^2 replaces the surface-layer
    C = 3bc*a^2*sqrt(z/z0); the FORM (Louis 1979 Eq. 19) is what we pin.
    """
    Ri = np.array([-0.3, -0.8, -2.0, -6.0])  # sigmoid(100*Ri) == 0 to ~1e-13
    l_mix = np.array([10.0, 25.0, 40.0, 60.0])
    dz = np.array([50.0, 50.0, 80.0, 100.0])
    fm, fh = _call(Ri, l_mix, dz, LOUIS1979_B, HDB1988_C, LOUIS1979_D,
                   b_heat=LTG82_B_HEAT_RATIO * LOUIS1979_B)
    denom = 1.0 + 3.0 * LOUIS1979_B * HDB1988_C * l_mix ** 2 * np.sqrt(np.abs(Ri)) / dz ** 2
    expected_m = 1.0 - 2.0 * LOUIS1979_B * Ri / denom
    b_h = LTG82_B_HEAT_RATIO * LOUIS1979_B
    expected_h = 1.0 - 2.0 * b_h * Ri / denom
    assert np.allclose(fm, expected_m, rtol=1e-9, atol=1e-12)
    assert np.allclose(fh, expected_h, rtol=1e-9, atol=1e-12)


def test_faithful_neutral_limit_unity():
    """Ri = 0 -> f_m = f_h = 1 exactly (both branches equal 1 at neutral)."""
    Ri = np.zeros(3)
    fm, fh = _call(Ri, np.full(3, 30.0), np.full(3, 50.0),
                   LOUIS1979_B, HDB1988_C, LOUIS1979_D,
                   b_heat=LTG82_B_HEAT_RATIO * LOUIS1979_B)
    assert np.allclose(fm, 1.0, rtol=0, atol=1e-12)
    assert np.allclose(fh, 1.0, rtol=0, atol=1e-12)


def test_faithful_ltg82_split_prandtl_direction_and_km_invariance():
    """LTG82: Pr_t = K_m/K_h < 1 unstable, > 1 stable; K_m independent of b_heat.

    b_heat > b enhances heat mixing more than momentum when unstable (f_h > f_m)
    and suppresses it more when stable (f_h < f_m); b_heat = b -> f_h == f_m.
    Because only the numerator carries b_heat, f_m must NOT move with b_heat.
    """
    l_mix, dz = np.full(1, 30.0), np.full(1, 50.0)
    b, c, d = LOUIS1979_B, HDB1988_C, LOUIS1979_D
    b_h = LTG82_B_HEAT_RATIO * b

    fm_u, fh_u = _call([-0.5], l_mix, dz, b, c, d, b_heat=b_h)
    assert fh_u[0] > fm_u[0]  # heat more enhanced (unstable)
    fm_s, fh_s = _call([0.5], l_mix, dz, b, c, d, b_heat=b_h)
    assert fh_s[0] < fm_s[0]  # heat more suppressed (stable)

    # b_heat = b recovers the Louis-1979 single function.
    fm_eq, fh_eq = _call([-0.5], l_mix, dz, b, c, d, b_heat=b)
    assert np.allclose(fh_eq, fm_eq, rtol=1e-12, atol=1e-14)

    # K_m (hence f_m) is invariant to b_heat.
    fm_a, _ = _call([-0.5], l_mix, dz, b, c, d, b_heat=b)
    fm_b, _ = _call([-0.5], l_mix, dz, b, c, d, b_heat=b_h)
    assert np.allclose(fm_a, fm_b, rtol=1e-12, atol=1e-14)


def test_faithful_monotone_decreasing_and_positive():
    """f_m strictly decreases with Ri (more stable -> less mixing) and stays > 0."""
    Ri = np.linspace(-5.0, 5.0, 101)
    l_mix = np.full_like(Ri, 30.0)
    dz = np.full_like(Ri, 50.0)
    fm, fh = _call(Ri, l_mix, dz, LOUIS1979_B, HDB1988_C, LOUIS1979_D,
                   b_heat=LTG82_B_HEAT_RATIO * LOUIS1979_B)
    assert np.all(np.diff(fm) < 0.0)
    assert np.all(fm > 0.0) and np.all(fh > 0.0)


# ===========================================================================
# DEPARTURES: re-tuned constant + smoothing that differ from Louis (1979)
# ===========================================================================
def test_departure_default_constants_are_louis1979_except_c():
    """Config defaults: b=5,d=5 (Louis 1979), c=16.6 (Holtslag&De Bruin 1988).

    Canary: b/d ARE the Louis originals; c is the documented departure; the
    heat ratio is LTG82's 1.5. A silent change to any trips this.
    """
    d = LouisConfig()
    assert d.b_louis == LOUIS1979_B      # Louis (1979)
    assert d.d_louis == LOUIS1979_D      # Louis (1979)
    assert d.c_louis == HDB1988_C        # Holtslag & De Bruin (1988), != Louis 5.0
    assert d.c_louis != LOUIS1979_C
    assert d.b_heat_ratio == LTG82_B_HEAT_RATIO  # LTG82 3b/2b


def test_departure_sigmoid_blend_smooths_the_hard_switch():
    """The sigmoid blend replaces Louis's hard Ri<0/Ri>=0 switch (differentiable).

    Right at neutral the true hard switch is ambiguous but both branches equal 1;
    just off neutral a FINITE-sharpness blend departs from the hard switch by a
    measurable amount, while sharpness->inf recovers it. Pin both: (a) finite
    sharpness gives a blended (not single-branch) value near neutral; (b) large
    sharpness collapses to the pure branch.
    """
    l_mix, dz = np.full(1, 30.0), np.full(1, 50.0)
    b, c, d = LOUIS1979_B, HDB1988_C, LOUIS1979_D
    Ri = np.array([0.02])  # just stable

    # Pure stable-branch (hard-switch) value.
    hard_stable = 1.0 / (1.0 + 2.0 * b * Ri / np.sqrt(1.0 + d * Ri))

    # Finite sharpness => blend of unstable(=1) and stable => strictly ABOVE the
    # pure stable value (differentiable smoothing, a real departure near neutral).
    fm_soft, _ = _call(Ri, l_mix, dz, b, c, d, sharp=5.0)
    assert fm_soft[0] > hard_stable[0]
    assert not np.allclose(fm_soft, hard_stable, rtol=1e-3)

    # Large sharpness => recovers the hard switch to round-off.
    fm_hard, _ = _call(Ri, l_mix, dz, b, c, d, sharp=1.0e4)
    assert np.allclose(fm_hard, hard_stable, rtol=1e-9, atol=1e-12)
