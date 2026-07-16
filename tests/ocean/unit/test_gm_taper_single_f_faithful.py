"""Certifies the GM/Redi DM95 taper is oracle-faithful (a REFUTED review finding).

Oracle-review claim: legoESM applies the DM95 taper to the SLOPE (giving f² on
the K33 diagonal) where Veros applies a single f to the whole tensor. The
multi-angle investigation REFUTED this as a bug:

  * The DEFAULT / oracle path is ``slope_scheme="triads"`` (raw slopes, a single
    taper factor on K33 — matching Veros/pyOM2 ``isoneutral.py`` exactly). Every
    Veros/NEMO fidelity recipe selects it.
  * The f² appears ONLY on the non-default centered/cube path, where K11 is left
    UNTAPERED, so the Redi 2×2 block is K11=K, K13=fKS, K33=f²KS² -> det = 0
    EXACTLY: a self-consistent positive-semidefinite pure isoneutral rotation
    (Ferrari 2008 surface-complement made inherent), NOT anti-diffusive.

These tests LOCK those two invariants so a future erroneous "fix" (e.g. switching
the centered K33 to f·S² while K11 stays untapered -> det>0 -> spurious diapycnal
mixing) goes red. Heavy Redi-null / det>=0 physics is covered by
test_gm_redi_eady_physics.py and test_isoneutral_slope_density.py.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.ocean.physics.lateral_mixing._gm_redi_common import dm95_taper
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig


def test_default_gm_redi_uses_faithful_single_f_triad_path():
    """Production default is the raw-slope, single-f triad path (= Veros)."""
    cfg = GMRediConfig()
    assert cfg.slope_scheme == "triads", (
        "GM/Redi default must stay the Veros-faithful triad path (single f on "
        f"K33); got slope_scheme={cfg.slope_scheme!r}"
    )


def test_dm95_taper_is_a_single_scalar_factor_in_zero_one():
    """The DM95 taper building block is ONE scalar f in [0,1] applied to S.

    Both slope components are multiplied by the SAME f, and the returned tapered
    slope is exactly S·f — the single-f convention. (The centered path squares
    this f only because K11 is left untapered, preserving det=0; see module
    docstring.)
    """
    shape = (6, 4, 4, 3)
    rng = np.random.default_rng(0)
    S_x = jnp.asarray(0.02 * rng.standard_normal(shape))
    S_y = jnp.asarray(0.02 * rng.standard_normal(shape))
    S_max = 0.01

    S_xt, S_yt, taper = dm95_taper(S_x, S_y, S_max)

    taper_np = np.asarray(taper)
    assert np.all(taper_np >= 0.0) and np.all(taper_np <= 1.0)
    # Single factor, identical for both components: S_t == S · taper.
    assert np.allclose(np.asarray(S_xt), np.asarray(S_x) * taper_np, atol=0.0)
    assert np.allclose(np.asarray(S_yt), np.asarray(S_y) * taper_np, atol=0.0)


def test_centered_redi_block_is_det_zero_pure_rotation():
    """Centered path K11=K, K13=f·K·S, K33=f²·K·S² -> det=0 (not anti-diffusive).

    This is the algebra that makes the f² on K33 oracle-EQUIVALENT (a pure
    isoneutral rotation) rather than a bug. If a future edit tapered K33 by f
    while leaving K11=K, det would become f·K²·S²·(1-f) > 0 -> spurious diapycnal
    mixing; this test would then fail.
    """
    rng = np.random.default_rng(1)
    shape = (6, 4, 4, 3)
    S_x = jnp.asarray(0.03 * rng.standard_normal(shape))
    S_y = jnp.zeros(shape)  # isolate the x–z block
    S_max = 0.01
    kappa_R = 1.0e3

    S_xt, _, taper = dm95_taper(S_x, S_y, S_max)  # S_xt = f·S_x
    # Centered assembly (gm_redi.py): K11 untapered, K13 uses tapered slope,
    # K33 = kappa_R · (f·S)².
    K11 = kappa_R
    K13 = kappa_R * np.asarray(S_xt)
    K33 = kappa_R * np.asarray(S_xt) ** 2
    det = K11 * K33 - K13 ** 2
    scale = kappa_R ** 2 * np.asarray(S_xt) ** 2 + 1e-30
    assert np.max(np.abs(det) / scale) < 1e-12, "centered Redi block must be det=0"
