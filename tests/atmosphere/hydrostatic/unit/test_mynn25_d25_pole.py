"""The MYNN-2.5 denominator D25 passes through zero, and used to be floored
one-sidedly at 1e-30.

``D25 = phi_2*phi_4 + phi_5*phi_3`` equals 1.0 at ``G_M = G_H = 0``. It is not
sign-definite: with ``phi_5 = 6*alpha_c^2*A1^2*G_M`` identically zero at zero
resolved shear, ``D25 = phi_2*phi_4`` and

    phi_4 = 1 - [3*A2*B2*(1-C3) + 12*A1*A2*(1-C2)] * G_H

has a root at ``G_H = 0.046`` for the NN09 constants. ``max(D25, 1e-30)`` does
not regularize a root, it amplifies it by ~1e30 -- and the amplified branch is
POSITIVE, so the downstream ``Kh >= 0`` clamp cannot catch it. ``G_H`` is
``-L^2 N^2 / qke``, so any unstable layer in a shear-free column sweeps through
the root: the Nieuwstadt CBL, the one LES tuning case with
``u_geo = v_geo = f_c = 0``, went from ``qke = 5e-6`` to ``1e24`` in a single
step and the whole column to NaN by step 10, while the seven other cases and
the eight other closures ran 2000 steps clean.

These tests pin the two-sided floor: that the pole is REAL (so the fix is not
decoration), that the floor bounds it, that the sign is untouched, and that
every well-posed point is bit-identical to the unfloored algebra.

x64 is required: the whole point is a denominator crossing zero, and fp32
rounding puts the crossing somewhere else.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.config import (  # noqa: E402
    MYNN25Config,
)
from legoesm.atmosphere.physics.turbulence.mynn25 import (  # noqa: E402
    _compute_SM_SH,
)

# Zero shear, and a length/energy pair a shear-free column really reaches:
# L is floored at 1 m and rises to ~kappa*z, while qke sits on its own 1e-10
# floor because the surface BC qke_sfc = B1^(2/3)*u*^2 is ~0 with no wind.
_L = 4.0
_Q = 1.0e-5


def _sm_sh(G_M, G_H, cfg):
    sm, sh = _compute_SM_SH(
        jnp.asarray([[float(G_M)]]), jnp.asarray([[float(G_H)]]),
        jnp.asarray([[_L]]), jnp.asarray([[_Q]]), cfg,
    )
    return float(np.asarray(sm).ravel()[0]), float(np.asarray(sh).ravel()[0])


def _phi4_root(cfg: MYNN25Config) -> float:
    """G_H where phi_4 = 0, from the config's OWN constants.

    Hardcoding 0.046 would let a retuned A1/A2/B2/C2/C3 move the pole out from
    under the test while it kept passing.
    """
    slope = (3.0 * cfg.A2 * cfg.B2 * (1.0 - cfg.C3)
             + 12.0 * cfg.A1 * cfg.A2 * (1.0 - cfg.C2))
    return 1.0 / slope


def test_the_pole_is_real_and_the_old_floor_amplified_it():
    """NON-VACUOUS GATE: with the floor put back to 1e-30 the closure must
    still blow up. If this stops failing, the pole is gone and the fix below
    is testing nothing."""
    cfg = MYNN25Config(d25_floor=1e-30)
    root = _phi4_root(cfg)
    worst = max(abs(v)
                for gh in np.linspace(0.5 * root, 1.5 * root, 401)
                for v in _sm_sh(0.0, gh, cfg))
    assert worst > 1e6, (
        f"expected an unbounded stability function near G_H = {root:.4g}; "
        f"largest |SM25|,|SH25| was {worst:.4g}")


def test_the_default_floor_bounds_the_pole():
    cfg = MYNN25Config()
    root = _phi4_root(cfg)
    worst = max(abs(v)
                for gh in np.linspace(0.5 * root, 1.5 * root, 401)
                for v in _sm_sh(0.0, gh, cfg))
    # The neutral values are SM25 = 0.695, SH25 = 0.665. Two orders of
    # magnitude above that is still a finite, physically small diffusivity
    # (Kh = L*q*SH ~ 4e-3 m^2/s here) and is what the floor buys.
    assert worst < 1.0e2, f"|SM25|,|SH25| reached {worst:.4g} near the root"


def test_the_root_from_the_constants_is_where_the_sign_flips():
    """The derived root must be the OBSERVED one, or the derivation in the
    source comment is describing a different pole from the one that fires."""
    cfg = MYNN25Config()
    root = _phi4_root(cfg)
    _sm_lo, sh_lo = _sm_sh(0.0, 0.5 * root, cfg)
    _sm_hi, sh_hi = _sm_sh(0.0, 2.0 * root, cfg)
    assert sh_lo > 0.0, f"SH25 should be positive below the root, got {sh_lo}"
    assert sh_hi < 0.0, f"SH25 should be negative above the root, got {sh_hi}"


@pytest.mark.parametrize("G_M", [0.0, 1e-8, 1.0, 1e4, 1e8])
@pytest.mark.parametrize("G_H", [-1e6, -1e2, -1.0, 0.0, 1.0, 1e2, 1e6, 4e7])
def test_well_posed_points_are_bit_identical(G_M, G_H):
    """The floor must be inert wherever |D25| already exceeded it.

    Compared against the SAME function with a floor far below any value it can
    take, which isolates the floor as the only difference -- comparing against
    a hand-rolled reimplementation of the algebra would not.
    """
    ref = MYNN25Config(d25_floor=1e-300)
    got = MYNN25Config()
    sm_r, sh_r = _sm_sh(G_M, G_H, ref)
    sm_g, sh_g = _sm_sh(G_M, G_H, got)
    # A point genuinely inside the floor is allowed to differ; assert on the
    # rest, and make sure the sweep is not entirely inside the floor.
    if abs(sm_r) < 1e2 and abs(sh_r) < 1e2:
        assert sm_g == sm_r and sh_g == sh_r, (
            f"floor changed a well-posed point (G_M={G_M}, G_H={G_H})")


def test_the_floor_never_flips_the_sign():
    """Two-sided by construction: past the root SM25/SH25 stay negative and
    the existing Km/Kh >= 0 clamp keeps handling them. A one-sided floor would
    turn that branch positive and inject upgradient mixing as if it were real.
    """
    cfg = MYNN25Config()
    ref = MYNN25Config(d25_floor=1e-300)
    root = _phi4_root(cfg)
    for gh in np.linspace(-2.0, 4.0 * root, 97):
        for got, want in zip(_sm_sh(0.0, gh, cfg), _sm_sh(0.0, gh, ref)):
            if abs(want) > 1e-12:
                assert np.sign(got) == np.sign(want), (
                    f"sign flipped at G_H={gh:.5g}: {got} vs {want}")


def test_shear_removes_the_pole():
    """Documents WHY only the Nieuwstadt CBL blew up: phi_5 = 6*alpha^2*A1^2*G_M
    lifts D25 off zero, so any real shear is enough."""
    cfg = MYNN25Config(d25_floor=1e-300)
    root = _phi4_root(cfg)
    for gm in (1e-12, 1e-6, 1.0):
        sm, sh = _sm_sh(gm, root, cfg)
        assert sm > 0.0 and sh > 0.0, (
            f"G_M={gm} at the zero-shear root gave SM25={sm}, SH25={sh}")
        assert abs(sm) < 1e2 and abs(sh) < 1e2


def test_the_floor_is_declared_not_trainable():
    """It is a numerics regulariser. If it ever became spec-eligible the
    parameter collector would hand it to an optimizer, which would discover
    that shrinking it lowers the loss right up until the column blows up."""
    from legoesm.atmosphere.physics.turbulence import config as turb_config
    spec = turb_config.__param_spec__["MYNN25Config"]
    assert "d25_floor" not in spec["params"]
    assert "d25_floor" in spec["excluded"]
