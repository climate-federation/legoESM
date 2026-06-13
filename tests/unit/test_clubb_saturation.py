"""Unit tests for the CLUBB saturation adapter + Flatau thermo curves.

Covers (a) the canonical Flatau SVP curves added to ``legoesm.thermo`` and
(b) the ``clubb_saturation`` mixing-ratio adapter (CAM default: Flatau,
mixing ratio with CLUBB's AD-safe denominator guard).

Part of the fuller CLUBB port — see ``docs/md_files/clubb.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    sat_mixrat_ice,
    sat_mixrat_liq,
)
from legoesm.thermo import (  # noqa: E402
    saturation_vapor_pressure,
    saturation_vapor_pressure_flatau,
    saturation_vapor_pressure_ice_flatau,
)

from legoesm import constants  # noqa: E402


def _K(Tc):
    return constants.T_freeze + np.asarray(Tc, dtype=np.float64)


# ---------------------------------------------------------------------------
# Flatau SVP curves (canonical, in legoesm.thermo)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "Tc, esat_liq_golden",
    [
        (-40.0, 19.05702),
        (-20.0, 125.52871),
        (0.0, 611.58370),
        (20.0, 2339.72679),
        (40.0, 7375.28692),
    ],
)
def test_flatau_liquid_golden(Tc, esat_liq_golden):
    got = float(saturation_vapor_pressure_flatau(jnp.asarray(_K(Tc))))
    assert got == pytest.approx(esat_liq_golden, rel=1e-5)


@pytest.mark.parametrize(
    "Tc, esat_ice_golden",
    [
        (-40.0, 12.84093),
        (-20.0, 103.33688),
        (0.0, 609.86899),
    ],
)
def test_flatau_ice_golden(Tc, esat_ice_golden):
    got = float(saturation_vapor_pressure_ice_flatau(jnp.asarray(_K(Tc))))
    assert got == pytest.approx(esat_ice_golden, rel=1e-5)


def test_flatau_positive_and_monotonic():
    T = jnp.asarray(_K(np.linspace(-80.0, 45.0, 200)))
    e_liq = saturation_vapor_pressure_flatau(T)
    assert jnp.all(e_liq > 0)
    assert jnp.all(jnp.diff(e_liq) > 0)   # strictly increasing in T


def test_flatau_ice_below_liquid_subfreezing():
    """Over ice SVP < over liquid below 0 deg C (thermodynamic requirement)."""
    T = jnp.asarray(_K(np.linspace(-60.0, -1.0, 100)))
    e_liq = saturation_vapor_pressure_flatau(T)
    e_ice = saturation_vapor_pressure_ice_flatau(T)
    assert jnp.all(e_ice < e_liq)


def test_flatau_close_to_tetens_in_normal_range():
    """Flatau and the existing Tetens curve agree within a few % over 240-310 K."""
    T = jnp.asarray(np.linspace(240.0, 310.0, 100))
    e_flatau = saturation_vapor_pressure_flatau(T)
    e_tetens = saturation_vapor_pressure(T)
    rel = jnp.abs(e_flatau - e_tetens) / e_tetens
    assert float(jnp.max(rel)) < 0.03


# ---------------------------------------------------------------------------
# CLUBB mixing-ratio adapter
# ---------------------------------------------------------------------------

def test_sat_mixrat_liq_matches_definition_normal_regime():
    p = jnp.asarray([1.0e5, 8.5e4, 5.0e4])
    T = jnp.asarray(_K([20.0, 5.0, -10.0]))
    esat = saturation_vapor_pressure_flatau(T)
    expected = constants.epsilon * esat / (p - esat)
    np.testing.assert_allclose(np.asarray(sat_mixrat_liq(p, T)), np.asarray(expected), rtol=1e-12)


def test_sat_mixrat_ice_matches_definition_normal_regime():
    p = jnp.asarray([8.0e4, 5.0e4])
    T = jnp.asarray(_K([-20.0, -40.0]))
    esat = saturation_vapor_pressure_ice_flatau(T)
    expected = constants.epsilon * esat / (p - esat)
    np.testing.assert_allclose(np.asarray(sat_mixrat_ice(p, T)), np.asarray(expected), rtol=1e-12)


def test_sat_mixrat_fallback_to_epsilon_when_esat_exceeds_p():
    # Hot + very low pressure -> esat >= p - 1 -> fallback rsat = epsilon.
    p = jnp.asarray([1.0e3])
    T = jnp.asarray(_K([60.0]))
    out = float(sat_mixrat_liq(p, T)[0])
    assert out == pytest.approx(constants.epsilon, rel=1e-12)


def test_sat_mixrat_liq_positive_and_increasing_with_T():
    p = jnp.full((50,), 9.0e4)
    T = jnp.asarray(_K(np.linspace(-30.0, 35.0, 50)))
    r = sat_mixrat_liq(p, T)
    assert jnp.all(r > 0)
    assert jnp.all(jnp.diff(r) > 0)


def test_sat_mixrat_liq_grad_finite_including_fallback():
    """Reverse-mode gradient is finite even where the denominator guard fires."""
    # Mix of normal columns and a fallback column (hot+low-p).
    p = jnp.asarray([1.0e5, 5.0e4, 1.0e3])
    T = jnp.asarray(_K([20.0, -10.0, 60.0]))

    def loss(T_):
        return jnp.sum(sat_mixrat_liq(p, T_))

    g = jax.grad(loss)(T)
    assert jnp.all(jnp.isfinite(g))


def test_sat_mixrat_liq_jit_clean():
    p = jnp.asarray([1.0e5, 7.0e4])
    T = jnp.asarray(_K([15.0, -5.0]))
    out = jax.jit(sat_mixrat_liq)(p, T)
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
