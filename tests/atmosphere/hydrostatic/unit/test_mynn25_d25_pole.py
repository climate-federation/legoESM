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

They also pin the three sibling guards an adversarial review turned up on the
same path, each MEASURED before it was believed:

* reverse-mode AD returned NaN through an UNSELECTED ``jnp.where`` branch --
  ``L_S_stable_mid`` divides by ``1 + 2.7*zeta``, which vanishes at
  ``zeta = -0.370``, an ordinary unstable surface-layer value. This module is
  gradient-tuned, so that was a live defect, not a hypothetical;
* ``|Ri|`` reached ~1e30 before being squared, which overflows float32;
* ``F1``, a divisor built entirely from TRAINABLE coefficients, reaches zero
  inside the declared sigmoid bounds.

x64 is required for the D25 tests: the whole point is a denominator crossing
zero, and fp32 rounding puts the crossing somewhere else.
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
    _compute_master_length,
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


# --- sibling guards on the same path ----------------------------------------

def _master_length_sum(w_thv_sfc, L_obukhov, n=6):
    """Sum of the master length over a column, as a function of two scalars.

    Both arguments are differentiated, which is what exposes an unselected
    branch: `zeta = z / L_obukhov` sweeps the whole column, so some level lands
    on `1 + 2.7*zeta = 0` for the right Obukhov length.
    """
    z = jnp.linspace(400.0, 10.0, n)[None, :]
    return jnp.sum(_compute_master_length(
        q_half=jnp.full((1, n), 1e-2),
        z_half_geom=z,
        dz_half=jnp.full((1, n), 20.0),
        L_obukhov=jnp.asarray([L_obukhov]),
        dthv_dz_half=jnp.full((1, n), -1e-3),
        w_thv_sfc=jnp.asarray([w_thv_sfc]),
        th_ref=300.0,
        config=MYNN25Config(),
    ))


# -27.0 puts the column's LOWEST level (10 m) on zeta = -1/2.7; -1080.0 puts
# its HIGHEST (400 m) there. Both measured NaN before the double-where.
@pytest.mark.parametrize("L_obukhov", [-27.0, -1080.0, -100.0, -5.0, 30.0])
def test_master_length_gradient_is_finite_through_the_unselected_branch(
        L_obukhov):
    """`jnp.where` evaluates BOTH branches, so a division by zero in the one
    that is discarded still poisons the cotangent. mynn25 is gradient-tuned
    against LES, so a NaN here is a dead optimizer, not a curiosity."""
    val = float(_master_length_sum(0.05, L_obukhov))
    g_flux = float(jax.grad(_master_length_sum, argnums=0)(0.05, L_obukhov))
    g_obu = float(jax.grad(_master_length_sum, argnums=1)(0.05, L_obukhov))
    assert np.isfinite(val), f"forward went non-finite at L_obu={L_obukhov}"
    assert np.isfinite(g_flux), f"d/dw_thv is {g_flux} at L_obu={L_obukhov}"
    assert np.isfinite(g_obu), f"d/dL_obukhov is {g_obu} at L_obu={L_obukhov}"


def test_the_unselected_branch_substitution_leaves_the_forward_answer_alone():
    """The guard may only touch zeta < 0, where the branch is discarded."""
    from legoesm.atmosphere.physics.turbulence.mynn25 import _MYNN_LS_STABLE_MID
    # Stable side (zeta > 0) is where L_S_stable_mid is actually SELECTED.
    for L_obukhov in (10.0, 50.0, 400.0):
        val = float(_master_length_sum(-0.01, L_obukhov))
        assert np.isfinite(val) and val > 0.0
    # And the denominator it guards really does vanish where claimed.
    assert abs(1.0 + _MYNN_LS_STABLE_MID * (-1.0 / _MYNN_LS_STABLE_MID)) < 1e-15


def test_ri_is_bounded_below_the_float32_square_overflow():
    """Ri is squared in the level-2 discriminant. Unbounded it reached ~1e30
    (G_M is exactly zero with no shear and the divisor floor is 1e-30), whose
    square overflows float32 to inf and then to NaN."""
    from legoesm.atmosphere.physics.turbulence.mynn25 import _RI_MAX
    assert _RI_MAX ** 2 < np.finfo(np.float32).max, (
        f"_RI_MAX={_RI_MAX:g} squared still overflows float32")
    # And it must be set as HIGH as that allows, not at a comfortable round
    # number: every entry the clip touches is one the closure would otherwise
    # have evaluated, so a lower bound perturbs more columns for no extra
    # protection. Measured: 1e15 moved ekman, wangara and astex at round-off.
    assert _RI_MAX ** 2 > 0.001 * np.finfo(np.float32).max, (
        f"_RI_MAX={_RI_MAX:g} is far below the float32 headroom and is "
        "clipping columns it does not need to")


def test_f1_reaches_zero_inside_the_declared_tunable_bounds():
    """NON-VACUOUS: the coefficients the floor protects really can hit it.

    F1 divides Rf1/Ri1 and is built entirely from spec'd trainable fields, so
    an optimizer exploring the sigmoid bounds walks onto the singularity.
    """
    from legoesm.atmosphere.physics.turbulence import config as turb_config
    spec = turb_config.__param_spec__["MYNN25Config"]["params"]
    singular = dict(A1=0.7, A2=0.4, B1=15.0, B2=8.0,
                    C1=0.15266666666666667, C2=1.4, C3=0.95, C5=0.5,
                    gamma1=0.15)
    for name, value in singular.items():
        lo, hi = spec[name]["bounds"]
        assert lo <= value <= hi, (
            f"{name}={value} is outside its declared bounds ({lo}, {hi}); "
            "the example no longer demonstrates a reachable singularity")
    c = MYNN25Config(**singular)
    f1 = (c.B1 * (c.gamma1 - c.C1) + 2.0 * c.A1 * (3.0 - 2.0 * c.C2)
          + 3.0 * c.A2 * (1.0 - c.C2) * (1.0 - c.C5))
    assert abs(f1) < 1e-12, f"F1 = {f1!r} at the singular parameter set"

    sm, sh = _sm_sh(1.0, 1.0, c)
    assert np.isfinite(sm) and np.isfinite(sh), (
        f"SM25={sm}, SH25={sh} at F1 = {f1!r}")


def test_the_f_floor_is_inert_at_the_nn09_defaults():
    """F1 ~ 6.3 and F2 ~ 5.0 there, so the floor must change nothing."""
    from legoesm.atmosphere.physics.turbulence.mynn25 import _F_FLOOR
    c = MYNN25Config()
    f1 = (c.B1 * (c.gamma1 - c.C1) + 2.0 * c.A1 * (3.0 - 2.0 * c.C2)
          + 3.0 * c.A2 * (1.0 - c.C2) * (1.0 - c.C5))
    assert abs(f1) > 1e3 * _F_FLOOR, (
        f"|F1| = {abs(f1):g} is within a factor 1000 of the floor "
        f"{_F_FLOOR:g}; the guard is no longer inert at the defaults")


@pytest.mark.parametrize("bad", [0.0, -1e-2])
def test_a_nonpositive_d25_floor_is_refused_at_function_entry(bad):
    """Zero reinstates the pole; negative silently disables the guard, because
    _away_from_zero then returns the raw denominator on both branches."""
    from legoesm.atmosphere.physics.turbulence.mynn25 import mynn25_turbulence
    n = 4
    ones = jnp.ones((1, n))
    with pytest.raises(ValueError, match="d25_floor"):
        mynn25_turbulence(
            u=ones, v=ones, T=300.0 * ones, q_v=0.0 * ones,
            qke=1e-4 * ones,
            p_full=jnp.linspace(7e4, 1e5, n)[None, :],
            p_half=jnp.linspace(6.9e4, 1.01e5, n + 1)[None, :],
            z_full=jnp.linspace(400.0, 10.0, n)[None, :],
            z_half=jnp.linspace(420.0, 0.0, n + 1)[None, :],
            T_sfc=jnp.asarray([301.0]), q_sfc=jnp.asarray([0.0]),
            rho=1.2 * ones, dt=60.0,
            config=MYNN25Config(d25_floor=bad),
        )


def test_away_from_zero_is_inert_outside_the_floor_and_keeps_the_sign():
    from legoesm.atmosphere.physics.turbulence.mynn25 import _away_from_zero
    x = jnp.asarray([-5.0, -1e-9, 0.0, 1e-9, 5.0])
    got = np.asarray(_away_from_zero(x, 1e-3))
    assert got[0] == -5.0 and got[4] == 5.0, "must not touch well-posed values"
    assert got[1] == -1e-3 and got[3] == 1e-3, "must push out to +/- the floor"
    assert got[2] == 1e-3, "the exact zero takes the positive branch"


def test_the_guards_do_not_promote_a_float32_column_to_float64():
    """The F1/F2 floor sits on quantities built from Python floats.

    Wrapping them in `jnp.asarray` would give them a STRONG float64 dtype
    whenever jax_enable_x64 is on, and that promotes the whole level-2 block
    away from a float32 column -- a silent precision and memory change in
    every global run, with no error anywhere. x64 is on in this module, so the
    check is live.
    """
    assert jax.config.read("jax_enable_x64"), (
        "this test only means something with x64 enabled")
    f32 = jnp.float32
    sm, sh = _compute_SM_SH(
        jnp.full((1, 2), 1.0, dtype=f32), jnp.full((1, 2), 0.5, dtype=f32),
        jnp.full((1, 2), 4.0, dtype=f32), jnp.full((1, 2), 1e-2, dtype=f32),
        MYNN25Config(),
    )
    assert sm.dtype == f32, f"SM25 came back {sm.dtype}, not float32"
    assert sh.dtype == f32, f"SH25 came back {sh.dtype}, not float32"
