"""RULE 5 frazil switches: width 0 is main's hard clip, default gaps pinned, gradients.
Run with JAX_ENABLE_X64=1 on CPU. Fixture banked from origin/main (batch_frazil_edmf_kf/frz)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.ocean.eos import freezing_point
from legoesm import constants
from legoesm.ocean.physics.frazil import FrazilConfig, apply_frazil

_FIX = Path(__file__).parent / "oracle_fixtures" / "frazil_rule5_main_d99609735b.npz"
_P = jnp.array([0., 1e4, 2e4])


def _tf():
    return freezing_point(jnp.full(3, 35.), _P, scheme="linear_S") - constants.T_freeze


def cases():
    """(theta, S, dz, p, mode): points inside, at the edges of and outside each band."""
    tf, s, dz = _tf(), jnp.full(3, 35.), jnp.array([5., 10., 20.])
    cfg = FrazilConfig(enabled=True, freezing_scheme="linear_S", rising_frazil="direct",
                       supercooling_switch_width_K=0.0)
    deep = tf + jnp.array([0.0, -0.1, -0.2])
    rising = apply_frazil(deep, s, dz, True, _P, cfg).ice_mass_per_area_kg_m2
    cap = rising / (cfg.rho_sw_kg_m3 * dz[0])  # surface melt fraction that melts all rising ice
    s_melt = (s[0] + cap * cfg.ice_salinity_psu) / (1 + cap)
    tf_melt = freezing_point(s_melt, _P[0], scheme="linear_S") - constants.T_freeze
    t_melt_all = tf_melt + cap * (cfg.L_f_j_kg / cfg.c_p_sw_j_kg_k + tf_melt)  # fixed point at y = cap
    return {
        "form_warm_edge": (tf + jnp.array([0.003, 0.5, 0.5]), s, dz, _P, "direct"),
        "melt_edge": (deep.at[0].set(t_melt_all + 0.002), s, dz, _P, "remelt"),
        "form_band": (tf + jnp.array([-0.004, 0.0, 0.003]), s, dz, _P, "direct"),
        "form_deep": (tf + jnp.array([-1.0, -0.5, -0.2]), s, dz, _P, "direct"),
        "melt_band": (tf + jnp.array([0.002, -0.3, -0.3]), s, dz, _P, "remelt"),
        "melt_cap": (tf + jnp.array([0.31, -0.1, -0.2]), s, dz, _P, "remelt"),
        "melt_tiny_cap": (tf + jnp.array([0.3, -0.0005, 0.0]), s, dz, _P, "remelt"),
        "melt_full": (tf + jnp.array([3.0, -0.1, -0.2]), s, dz, _P, "remelt"),
        "warm_nocarry": (tf + jnp.array([0.001, 2.0, 5.0]), s, dz, _P, "remelt"),
    }


def _run(case, width=None):
    t, s, dz, p, mode = cases()[case]
    cfg = FrazilConfig(enabled=True, freezing_scheme="linear_S", rising_frazil=mode)
    if width is not None:
        cfg = cfg._replace(supercooling_switch_width_K=width)
    return apply_frazil(t, s, dz, True, p, cfg)


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


@pytest.mark.parametrize("case", list(cases()))
def test_width_zero_is_the_main_hard_clip_bitwise(fix, case):
    out = _run(case, 0.0)
    for k in out._fields:
        np.testing.assert_array_equal(np.asarray(getattr(out, k)), fix[f"main_{case}_{k}"])


@pytest.mark.parametrize("case", list(cases()))
def test_default_gap_pinned_per_case(fix, case):
    out = _run(case)
    for k in out._fields:
        got = np.asarray(getattr(out, k)) - fix[f"main_{case}_{k}"]
        np.testing.assert_allclose(got, fix[f"gap_{case}_{k}"], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("case", list(cases()))
def test_default_conserves_mass_and_salt_and_never_melts_missing_ice(case):
    t, s, dz, p, mode = cases()[case]
    out = _run(case)
    rho, si = FrazilConfig().rho_sw_kg_m3, FrazilConfig().ice_salinity_psu
    ice = float(out.ice_mass_per_area_kg_m2)
    assert ice >= 0.0
    np.testing.assert_allclose(rho * float(jnp.sum(out.dz_m)) + ice, rho * float(jnp.sum(dz)), rtol=1e-14)
    np.testing.assert_allclose(rho * float(jnp.sum(out.dz_m * out.S_psu)) + ice * si,
                               rho * float(jnp.sum(dz * s)), rtol=1e-14)


def test_no_rising_ice_means_no_melt_and_no_formation_while_rising():
    """cap = 0 below the surface: deep warm cells are untouched bitwise at the default."""
    t, s, dz, p, _ = cases()["warm_nocarry"]
    out = _run("warm_nocarry")
    np.testing.assert_array_equal(np.asarray(out.dz_m[1:]), np.asarray(dz[1:]))
    np.testing.assert_array_equal(np.asarray(out.S_psu[1:]), np.asarray(s[1:]))


@pytest.mark.parametrize("case", ["form_warm_edge", "melt_edge"])
def test_gradient_through_the_switches(case):
    """Points where the hard clip has zero derivative (no ice formed / all
    rising ice melted) and the smooth switch supplies one."""
    t, s, dz, p, mode = cases()[case]

    def f(t0, width=None):
        cfg = FrazilConfig(enabled=True, freezing_scheme="linear_S", rising_frazil=mode)
        if width is not None:
            cfg = cfg._replace(supercooling_switch_width_K=width)
        return apply_frazil(t.at[0].set(t0), s, dz, True, p, cfg).ice_mass_per_area_kg_m2

    assert float(jax.grad(f)(t[0], 0.0)) == 0.0
    g = float(jax.grad(f)(t[0]))
    assert np.isfinite(g) and g != 0.0
    check_grads(f, (t[0],), order=1, modes=("rev",), eps=1e-6)


@pytest.mark.parametrize("w", [-0.01, float("nan"), True])
def test_width_is_validated(w):
    with pytest.raises(ValueError):
        apply_frazil(jnp.zeros(3), jnp.full(3, 35.), jnp.ones(3), True, _P,
                     FrazilConfig(enabled=True, supercooling_switch_width_K=w))
