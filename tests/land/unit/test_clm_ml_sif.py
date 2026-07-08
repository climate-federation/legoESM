"""SIF extraction from the CLM-ML multilayer canopy ``mlcanopy_inst``.

Exercises ``_extract_clm_ml_sif`` with a MOCK ``mlcanopy`` exposing the CLM-ml
per-(layer, leaf) arrays, so the extraction + leaf-area weighting + spval
masking + shared-core reuse are validated WITHOUT the optional ``clm-ml-jax``
dependency.  SIF is a pure consumer of the model's native ``je_leaf`` +
``apar_leaf``; the key gate is the big-leaf equivalence: feeding
``je = actual_electron_transport(An, Ci, Gamma*)`` reproduces the big-leaf
``leaf_sif`` exactly.

``tests/land/integration/test_clm_ml_sif_real.py`` drives the REAL installed
model end-to-end (CHATS7 tower site) when ``clm-ml-jax`` is importable; here we
validate the legoESM-side glue + numerics against a controlled mock.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.land.canopy.sif import (
    SIFConfig, actual_electron_transport, leaf_sif, leaf_sif_from_je,
)
from legoesm.land.canopy.clm_ml_interface import _extract_clm_ml_sif

SPVAL = 1.0e36
CFG = SIFConfig()


def _je(An, Ci, gstar):
    return float(actual_electron_transport(jnp.asarray(An), jnp.asarray(Ci), jnp.asarray(gstar)))


def _mock_mlcanopy(nlev, nleaf, *, je, apar, dpai, fracsun):
    """Build a 1-column mock matching the legoESM-interface array layout.

    The interface builds mlcanopy via ``create_mlcanopy(1, ncol)`` (begp=1), so
    the PATCH axis is 1-based: column i (0..ncol-1) lives at index i+1, index 0
    is the unused pad.  The layer and leaf axes are likewise 1-based with a spval
    pad at index 0.  ``je/apar`` are (nlev, nleaf); ``dpai/fracsun`` are (nlev,).
    Shapes (2, nlev+1, nleaf+1) / (2, nlev+1) so ``[1, 1:, 1:]`` selects the
    filled block for column i=0.  Only the fields the extractor reads are given.
    """
    def pad3(a):
        out = jnp.full((2, nlev + 1, nleaf + 1), SPVAL)
        return out.at[1, 1:, 1:].set(jnp.asarray(a))

    def pad2(a):
        out = jnp.full((2, nlev + 1), SPVAL)
        return out.at[1, 1:].set(jnp.asarray(a))

    return types.SimpleNamespace(
        je_leaf=pad3(je), apar_leaf=pad3(apar),
        dpai_profile=pad2(dpai), fracsun_profile=pad2(fracsun),
    )


def test_extract_matches_big_leaf_for_single_sunlit_element():
    # One layer, sunlit only (fracsun=1 -> shaded leaf_area=0), dpai=1 so the
    # sunlit leaf-area weight is 1.  Feeding je = actual_electron_transport(...)
    # => CLM-ML SIF must equal the big-leaf leaf_sif exactly.
    An_v, Ci_v, apar_v, gs_v = 15.0, 280.0, 800.0, 43.0
    ml = _mock_mlcanopy(
        nlev=1, nleaf=2,
        je=[[_je(An_v, Ci_v, gs_v), SPVAL]], apar=[[apar_v, SPVAL]],
        dpai=[1.0], fracsun=[1.0])
    out = _extract_clm_ml_sif(ml, ncol=1, sif_cfg=CFG)
    assert out is not None and out.shape == (1,)
    ref = float(leaf_sif(jnp.asarray(An_v), jnp.asarray(Ci_v),
                         jnp.asarray(gs_v), jnp.asarray(apar_v), CFG))
    assert float(out[0]) == pytest.approx(ref, rel=1e-6)


def test_spval_layer_is_masked_out():
    # A 2nd layer with dpai=spval must contribute nothing (== the 1-layer case).
    je_v = _je(15.0, 280.0, 43.0)
    one = _mock_mlcanopy(
        nlev=1, nleaf=2, je=[[je_v, SPVAL]], apar=[[800.0, SPVAL]],
        dpai=[1.0], fracsun=[1.0])
    two = _mock_mlcanopy(
        nlev=2, nleaf=2,
        je=[[je_v, SPVAL], [SPVAL, SPVAL]], apar=[[800.0, SPVAL], [SPVAL, SPVAL]],
        dpai=[1.0, SPVAL], fracsun=[1.0, SPVAL])
    s1 = float(_extract_clm_ml_sif(one, 1, CFG)[0])
    s2 = float(_extract_clm_ml_sif(two, 1, CFG)[0])
    assert jnp.isfinite(s2)
    assert s2 == pytest.approx(s1, rel=1e-6)


def test_sunlit_plus_shaded_sums():
    # Both leaves filled: total = sunlit-area SIF + shaded-area SIF.
    je_sun, je_sh = _je(18.0, 280.0, 45.0), _je(6.0, 300.0, 43.0)
    ml = _mock_mlcanopy(
        nlev=1, nleaf=2,
        je=[[je_sun, je_sh]], apar=[[900.0, 200.0]], dpai=[1.0], fracsun=[0.6])
    out = float(_extract_clm_ml_sif(ml, 1, CFG)[0])
    # Hand sum: leaf_sif_from_je(je, apar) * leaf_area, la_sun=0.6, la_sha=0.4.
    exp = (float(leaf_sif_from_je(jnp.asarray(je_sun), jnp.asarray(900.0), CFG)) * 0.6
           + float(leaf_sif_from_je(jnp.asarray(je_sh), jnp.asarray(200.0), CFG)) * 0.4)
    assert out == pytest.approx(exp, rel=1e-6)


def test_returns_none_when_field_absent():
    # hasattr fallback: a port missing je_leaf -> None (never crash flux path).
    ml = _mock_mlcanopy(
        nlev=1, nleaf=2, je=[[30.0, SPVAL]], apar=[[800.0, SPVAL]],
        dpai=[1.0], fracsun=[1.0])
    del ml.je_leaf
    assert _extract_clm_ml_sif(ml, 1, CFG) is None


def test_escape_probability_scales():
    mk = lambda fesc: _extract_clm_ml_sif(_mock_mlcanopy(
        nlev=1, nleaf=2, je=[[30.0, SPVAL]], apar=[[800.0, SPVAL]],
        dpai=[1.0], fracsun=[1.0]),
        1, SIFConfig(escape_probability=fesc))
    assert float(mk(0.5)[0]) == pytest.approx(0.5 * float(mk(1.0)[0]), rel=1e-6)


def test_two_patch_columns_are_independent():
    # 2-column container built the interface way (begp=1): columns 0,1 live at
    # patch indices 1,2, index 0 is pad.  Each column must map to its OWN patch.
    # Catches a patch-axis off-by-one (e.g. reading [i] instead of [i+1], which
    # would pick up the index-0 pad / shift columns; a bug only ncol>1 exposes).
    def _2patch(nlev, nleaf, je, apar, dpai, fracsun):  # inputs (2, ...) = 2 cols
        def p3(a):
            out = jnp.full((3, nlev + 1, nleaf + 1), SPVAL)
            return out.at[1:, 1:, 1:].set(jnp.asarray(a))   # cols at index 1,2
        def p2(a):
            out = jnp.full((3, nlev + 1), SPVAL)
            return out.at[1:, 1:].set(jnp.asarray(a))
        return types.SimpleNamespace(je_leaf=p3(je), apar_leaf=p3(apar),
                                     dpai_profile=p2(dpai), fracsun_profile=p2(fracsun))

    # column 0: lit; column 1: dark (apar=0) -> SIF 0.
    ml = _2patch(1, 2,
                 je=[[[40.0, 20.0]], [[40.0, 20.0]]],
                 apar=[[[900.0, 300.0]], [[0.0, 0.0]]],
                 dpai=[[1.0], [1.0]], fracsun=[[0.7], [0.7]])
    out = _extract_clm_ml_sif(ml, ncol=2, sif_cfg=CFG)
    assert out.shape == (2,)
    # column 0 matches the 1-column extraction of its data; column 1 == 0.
    ref0 = float(_extract_clm_ml_sif(_mock_mlcanopy(
        nlev=1, nleaf=2, je=[[40.0, 20.0]], apar=[[900.0, 300.0]],
        dpai=[1.0], fracsun=[0.7]), 1, CFG)[0])
    assert float(out[0]) == pytest.approx(ref0, rel=1e-6) and ref0 > 0.0
    assert float(out[1]) == 0.0
