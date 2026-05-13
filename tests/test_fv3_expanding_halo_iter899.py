"""FV3_3D iter 899: unit tests for fv3_corner_laplacian_nord_expanding_halo.

Verifies the FV3-faithful single-pad-multi-step wrapper:
  - nord=0: passthrough.
  - nord=1: bit-for-bit matches re-pad path (only 1 iteration).
  - nord=2: produces finite output, differs from re-pad path
            (intermediate cube-vertex behaviour differs).
  - nord >= 3: raises NotImplementedError.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core._fv3_divergence_corner import (
    fv3_corner_laplacian_nord,
    fv3_corner_laplacian_nord_expanding_halo,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid as _create_cdgrid


def _cdgrid(n):
    return _create_cdgrid(create_cubed_sphere(n))


def test_nord_0_passthrough():
    """nord=0: returns input unchanged."""
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=8990)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out = fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(divg))


def test_nord_1_matches_repad_path():
    """nord=1: bit-for-bit matches iter-892 re-pad fv3_corner_laplacian_nord
    (only one iteration; no cross-iteration corner-fill difference)."""
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=8991)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out_expand = fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=1)
    out_repad = fv3_corner_laplacian_nord(divg, cdgrid, nord=1)
    np.testing.assert_array_equal(np.asarray(out_expand),
                                   np.asarray(out_repad))


def test_nord_2_finite():
    """nord=2 expanding-halo produces finite output."""
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=8992)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out = fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=2)
    assert out.shape == (6, n + 1, n + 1)
    assert jnp.all(jnp.isfinite(out))


def test_nord_2_approximates_repad():
    """nord=2 expanding-halo matches re-pad path within ULP tolerance.

    EXPECTED finding: pad_halo's default avg-mode corner fill makes
    the halo=2 single-pad cross-panel data identical to repeated
    halo=1 re-pads at the inner ring.  Thus the existing legoESM
    re-pad nord=2 path is already FV3-faithful within machine
    precision — no behavioural change is required.

    This test pins that equivalence so a future refactor that
    breaks it surfaces immediately.  The iter-898 wider-halo
    helpers + iter-899 wrapper remain as documentation of the
    FV3 sw_core.F90 single-pad convention even though the legacy
    path is numerically equivalent.
    """
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=8993)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out_expand = fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=2)
    out_repad = fv3_corner_laplacian_nord(divg, cdgrid, nord=2)
    np.testing.assert_allclose(
        np.asarray(out_expand), np.asarray(out_repad),
        rtol=1e-12, atol=1e-12,
    )


def test_nord_2_quantitative_equivalence_iter901():
    """FV3_3D iter 901: pin the exact max-abs-diff between expanding-
    halo and re-pad nord=2 paths.

    iter-899 measured max abs diff = 3e-24 (machine-epsilon noise
    around double-precision arithmetic with rearranged op order).
    Pin upper bound at 1e-20 — well above machine epsilon (~2.2e-16
    for f64) but tight enough to catch any non-trivial behavioural
    divergence in either path.

    If this test ever fails, one of:
      - pad_halo corner-fill mode silently changed
      - h1/h2 step arithmetic was modified
      - re-pad legacy iter-892 wrapper was altered
    Investigate which path drifted before relaxing the bound.
    """
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=9011)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))
    out_expand = fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=2)
    out_repad = fv3_corner_laplacian_nord(divg, cdgrid, nord=2)
    max_diff = float(jnp.max(jnp.abs(out_expand - out_repad)))
    assert max_diff < 1e-20, (
        f"expanding-halo vs re-pad nord=2 max_diff={max_diff:.2e} "
        f"(expected < 1e-20 per iter-899 measurement of 3e-24).  "
        f"Behavioural drift in one of pad_halo / h1-step / h2-step / "
        f"legacy re-pad wrapper.  Investigate before relaxing bound."
    )


def test_nord_3_raises():
    """nord >= 3 not yet implemented — raises NotImplementedError."""
    n = 8
    cdgrid = _cdgrid(n)
    divg = jnp.zeros((6, n + 1, n + 1))
    with pytest.raises(NotImplementedError, match="nord=3"):
        fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=3)


def test_nord_negative_raises():
    """Negative nord raises ValueError."""
    n = 8
    cdgrid = _cdgrid(n)
    divg = jnp.zeros((6, n + 1, n + 1))
    with pytest.raises(ValueError, match=">= 0"):
        fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=-1)


def test_uniform_zero_lap_at_nord_2():
    """Uniform divg_d → near-zero Laplacian even after nord=2."""
    n = 8
    cdgrid = _cdgrid(n)
    divg = jnp.full((6, n + 1, n + 1), 1.0)
    out = fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=2)
    # Interior should be near-zero (Laplacian of constant is 0)
    interior_max = float(jnp.max(jnp.abs(out[:, 2:n - 1, 2:n - 1])))
    assert interior_max < 1e-6
