"""EKE closure against Veros's own ``set_eke_diffusivities_kernel`` (d657322b).

Fixture ``oracle_fixtures/veros_d657322b_eke.npz``: a synthetic 10x9x4 state run
through the pinned Veros clone (probe and votes: ``oracle_audit/
shelf_npzd_bs_oracle``), with E < 0, E ~ 0, Rhines-limited, deformation-limited,
equatorial-branch, l_min-floored and kappa-capped cells.  Dissipation is Veros's
``eke_c_eps*sqrteke/eke_len*maskW`` times ``max(eke, 0)``, compared on wet
(``maskW = 1``) cells only: the port leaves the masking to its caller.

RULE 5, per switch (positive part of E, min(L_mid, L_eq), the two eke_len
switches, the kappa_GM cap, the dissipation length floor):
* widths 0 reproduce Veros to 1e-12;
* the smooth defaults' gap from Veros is pinned per cell (1e-6 rel, 1e-12 abs);
* gradients through each transition are finite and non-zero at the defaults.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.ocean.physics.lateral_mixing import eke as E

jax.config.update("jax_enable_x64", True)

FIX = np.load(Path(__file__).parent / "oracle_fixtures" / "veros_d657322b_eke.npz")
C_K, C_EPS, L_MIN, CROSS, CRHIN, K_MAX = FIX["settings"]
BASE = dict(c_k=C_K, c_eps=C_EPS, l_min=L_MIN, eke_cross=CROSS, eke_crhin=CRHIN, kappa_gm_max=K_MAX)
HARD = E.EKEConfig(**BASE, e_pos_width=0.0, length_switch_width=0.0, kappa_cap_width=0.0)
SMOOTH = E.EKEConfig(**BASE)
QUANTITIES = ("L_rossby", "L_rhines", "eke_len", "K_gm", "dissipation")
WET = FIX["maskW"] > 0


def _cells(q, a):
    return a[WET] if q == "dissipation" else a


def _closure(cfg):
    intN = np.sum(np.sqrt(np.maximum(0.0, FIX["Nsqr"])) * FIX["dzw"] * FIX["maskW"], axis=2)
    eke = jnp.asarray(FIX["eke"])
    Lr = E.eke_deformation_radius(jnp.asarray(intN), jnp.asarray(FIX["coriolis_t"]), jnp.asarray(FIX["beta"]), cfg)
    Lrh = E.eke_rhines_length(eke, jnp.asarray(FIX["beta"])[..., None], cfg)
    Ll = E.eke_len_composite(Lr[..., None], Lrh, cfg)
    K = E.eke_kappa_gm(eke, Ll, cfg)
    eps = -E.eke_local_tendency(eke, jnp.zeros_like(eke), Ll, cfg)
    return {k: np.asarray(x) for k, x in zip(QUANTITIES, (Lr, Lrh, Ll, K, eps))}


@pytest.mark.parametrize("q", QUANTITIES)
def test_hard_limit_matches_veros(q):
    got, ref = _cells(q, _closure(HARD)[q]), _cells(q, FIX[f"veros_{q}"])
    assert (np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)).max() <= 1e-12


@pytest.mark.parametrize("q", QUANTITIES)
def test_smooth_default_gap_pinned(q):
    gap = _cells(q, _closure(SMOOTH)[q] - FIX[f"veros_{q}"])
    np.testing.assert_allclose(gap, _cells(q, FIX[f"pin_{q}"]), rtol=1e-6, atol=1e-12)


_BETA = 2.0e-11
_SWITCHES = {
    # name: (scalar function of x at the defaults, x at the transition, finite-difference step)
    "positive_part_E": (lambda x: E.eke_kappa_gm(x, jnp.asarray(1.0e3), SMOOTH), 0.0, 1.0e-11),
    "min_Lmid_Leq": (lambda x: E.eke_deformation_radius(
        x, jnp.asarray(1.0e-5), jnp.asarray(_BETA), SMOOTH), jnp.pi * 1.0e-5 ** 2 / (2.0 * _BETA), 1.0e-7),
    "min_cross_rhines": (lambda x: E.eke_len_composite(x, jnp.asarray(5.0e3), SMOOTH), 5.0e3, 1.0e-3),
    "max_lmin": (lambda x: E.eke_len_composite(x, jnp.asarray(1.0e6), SMOOTH), L_MIN, 1.0e-3),
    "kappa_cap": (lambda x: E.eke_kappa_gm(jnp.asarray(1.0), x, SMOOTH), K_MAX / C_K, 1.0e-2),
    "dissipation_lmin": (lambda x: E.eke_local_tendency(jnp.asarray(1.0e-2), jnp.asarray(0.0), x, SMOOTH), L_MIN, 1.0e-3),
}


@pytest.mark.parametrize("name", list(_SWITCHES))
def test_gradient_through_switch(name):
    f, x0, eps = _SWITCHES[name]
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    check_grads(f, (x0,), order=1, modes=["fwd", "rev"], eps=eps)
    g = jax.grad(f)(x0)
    assert np.isfinite(g) and g != 0.0


_W_E, _W_L, _W_K = SMOOTH.e_pos_width, SMOOTH.length_switch_width, SMOOTH.kappa_cap_width
_C1, _LEQ = 2.5, np.sqrt(2.5 / (2.0 * _BETA))
_BAND = {
    # name: (value at the defaults, value at width 0, +1 if the switch is a max, -1 if a min, width)
    "positive_part_E": (lambda c: E.eke_positive_part(jnp.asarray(_W_E / 4), c), +1, _W_E),
    "min_Lmid_Leq": (lambda c: E.eke_deformation_radius(
        jnp.asarray(np.pi * _C1), jnp.asarray(_C1 / (_LEQ + _W_L / 4)), jnp.asarray(_BETA), c), -1, _W_L),
    "min_cross_rhines": (lambda c: E.eke_len_composite(
        jnp.asarray(5.0e3 + _W_L / 4), jnp.asarray(5.0e3), c), -1, _W_L),
    "max_lmin": (lambda c: E.eke_len_composite(jnp.asarray(L_MIN - _W_L / 4), jnp.asarray(1.0e6), c), +1, _W_L),
    "rossby_lmin": (lambda c: E.eke_mixing_length(jnp.asarray(L_MIN - _W_L / 4), c), +1, _W_L),
    "kappa_cap": (lambda c: E.eke_kappa_gm(
        jnp.asarray(1.0), jnp.asarray((K_MAX - _W_K / 4) / C_K), c), -1, _W_K),
}


@pytest.mark.parametrize("name", list(_BAND))
def test_smooth_switch_active_inside_its_band(name):
    """A quarter-width inside the band the smooth switch departs from the hard one, by at most width/8."""
    f, sign, w = _BAND[name]
    d = sign * (float(f(SMOOTH)) - float(f(HARD)))
    assert 0.0 < d <= w / 8.0


def test_dissipation_floor_active_inside_its_band():
    L = jnp.asarray(L_MIN + _W_L / 4)
    hard = -E.eke_local_tendency(jnp.asarray(1.0e-2), jnp.asarray(0.0), L, HARD)
    smooth = -E.eke_local_tendency(jnp.asarray(1.0e-2), jnp.asarray(0.0), L, SMOOTH)
    assert 0.0 < float(hard - smooth) <= float(hard) * (_W_L / 8.0) / L_MIN


@pytest.mark.parametrize("e", [-1.0e-4, -_W_E / 2, 0.0])
def test_gradients_finite_for_nonpositive_energy(e):
    """Rhines length, kappa and tendency gradients stay finite where E <= 0 (both square roots guarded)."""
    def f(E_, beta):
        Lrh = E.eke_rhines_length(E_, beta, SMOOTH)
        return Lrh + E.eke_kappa_gm(E_, jnp.asarray(1.0e3), SMOOTH) + E.eke_local_tendency(
            E_, jnp.asarray(1.0e-5), jnp.asarray(1.0e3), SMOOTH)
    args = (jnp.asarray(e), jnp.asarray(_BETA))
    g = jax.grad(f, argnums=(0, 1))(*args)
    t = jax.jvp(f, args, (jnp.asarray(1.0), jnp.asarray(1.0e-12)))[1]
    assert all(np.isfinite(np.asarray(x)) for x in (*g, t))
