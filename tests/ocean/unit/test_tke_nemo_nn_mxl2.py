"""NEMO ``nn_mxl=2`` mixing length (``tke_mxl_choice=4``).

The ORCA1 namelist runs ``nn_mxl=2``; legoESM implemented only ``nn_mxl=3``.
NEMO's zdftke.F90 CASE(2) applies BOTH slope sweeps sequentially in place and
sets ``zmxld = zmxlm`` -- a SINGLE length for the eddy coefficient and the
dissipation -- whereas CASE(3) keeps the two envelopes and uses
``zmxlm = min(lup,ldown)``, ``zmxld = sqrt(lup*ldown)``.

Because ``min(lup,ldown) <= sqrt(lup*ldown)``, choice 4 has the SMALLER
dissipation length, hence LARGER ``eps = c_eps e^{3/2} / l_eps`` and less
retained TKE.  The two coincide where the envelopes agree (strong
stratification) and diverge where they do not (weakly stratified deep columns),
which is why this is a high-latitude-selective lever.  These tests pin exactly
that, including the negative control that the schemes must AGREE when the
envelopes agree -- otherwise a "difference" could be an unrelated bug.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import compute_mixing_lengths


def _cfg(choice):
    return TKEConfig(tke_mxl_choice=choice)


def _column(n=40, dz=10.0, n2=1.0e-9, e=1.0e-3, seed=0):
    """One column: TKE at interfaces, N^2 at interfaces, cell thicknesses.

    DEFAULTS ARE THE WEAKLY STRATIFIED (polar-like) REGIME on purpose.  With
    the earlier defaults (N2=1e-5, e=1e-4) the buoyancy length sqrt(2e)/N is
    ~4.5 m, far below the 10 m slope bound, so the buoyancy term alone sets the
    length and lup == ldown EVERYWHERE -- in which case sqrt(lup*ldn) ==
    min(lup,ldn) and the two NEMO schemes are identical BY CONSTRUCTION.  That
    is the tropics-like case (covered by test_schemes_agree_when_envelopes_
    agree), and it makes any "the schemes differ" assertion vacuous.  Here the
    buoyancy length is ~1400 m, so the SLOPE bound governs and the envelopes
    genuinely diverge.
    """
    e_arr = jnp.full((1, n - 1), e)
    n2_arr = jnp.full((1, n - 1), n2)
    dz_half = jnp.full((1, n - 1), dz)
    dz_cell = jnp.full((1, n), dz)
    return e_arr, n2_arr, dz_half, dz_cell


def _lengths(choice, **kw):
    e, n2, dz_half, dz_cell = _column(**kw)
    return compute_mixing_lengths(e, n2, dz_half, _cfg(choice),
                                  dz_cell=dz_cell,
                                  l_surface_anchor=jnp.array([1.0]))


class TestNemoNnMxl2:
    def test_choice4_uses_a_single_length(self):
        """nn_mxl=2: l_eps IS l_k (zmxld = zmxlm), exactly."""
        l_k, l_eps = _lengths(4)
        assert jnp.array_equal(l_k, l_eps)

    def test_choice3_uses_two_lengths(self):
        """Control: nn_mxl=3 must NOT collapse, else the test above is vacuous."""
        l_k, l_eps = _lengths(3)
        assert not jnp.array_equal(l_k, l_eps)

    def test_choice4_dissipation_length_is_never_larger(self):
        """min(lup,ldn) <= sqrt(lup*ldn) pointwise -> more dissipation."""
        _, eps4 = _lengths(4)
        _, eps3 = _lengths(3)
        assert bool(jnp.all(eps4 <= eps3 + 1e-12))
        assert float(jnp.max(eps3 - eps4)) > 0.0      # strictly somewhere

    def test_eddy_length_is_identical_between_the_two(self):
        """Only l_eps differs; l_k = min(lup,ldn) in BOTH (zdftke CASE 2/3)."""
        k4, _ = _lengths(4)
        k3, _ = _lengths(3)
        assert jnp.allclose(k4, k3, rtol=0.0, atol=0.0)

    def test_schemes_agree_when_envelopes_agree(self):
        """NEGATIVE CONTROL: strong stratification shrinks the buoyancy length
        below the slope bound, so lup ~ ldown and the two schemes must nearly
        coincide.  This is what makes the lever latitude-selective; if they
        differed here too, the difference would not be the claimed mechanism.
        """
        _, eps4 = _lengths(4, n2=1.0e-2, e=1.0e-8, dz=50.0)
        _, eps3 = _lengths(3, n2=1.0e-2, e=1.0e-8, dz=50.0)
        rel = float(jnp.max(jnp.abs(eps3 - eps4) / jnp.maximum(eps3, 1e-30)))
        assert rel < 1.0e-6, rel

    def test_weak_stratification_makes_them_diverge(self):
        """The complementary case: a weakly stratified deep column, where the
        slope bound (not buoyancy) sets the length, gives lup >> ldown near the
        surface and a large l_eps ratio.
        """
        _, eps4 = _lengths(4)          # defaults ARE the weak-stratification case
        _, eps3 = _lengths(3)
        ratio = float(jnp.max(eps3 / jnp.maximum(eps4, 1e-30)))
        assert ratio > 1.5, ratio

    def test_choice4_requires_cell_thicknesses(self):
        """dz_cell is needed for the sweeps; its absence must RAISE."""
        e, n2, dz_half, _ = _column()
        with pytest.raises(ValueError, match="dz_cell"):
            compute_mixing_lengths(e, n2, dz_half, _cfg(4), dz_cell=None,
                                   l_surface_anchor=jnp.array([1.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
