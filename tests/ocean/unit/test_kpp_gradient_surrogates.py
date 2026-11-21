"""KPPConfig.gradient_surrogates: exact forward values, a logistic derivative in the backward pass (docs/88 6.12 a2).

1. The gate's forward value is jnp.where(x > 0, a, b) bit for bit. Its backward pass is the derivative of the
   logistic blend s a + (1 - s) b with s = sigmoid(x / width), the form fixed in docs/88 6.12 (a2):
   g s for a, g (1 - s) for b, g (a - b) s (1 - s) / width for x (checked against those formulas).
   Broadcast arguments get cotangents of their own shapes; a width that is not positive raises.
2. kpp_vertical_mixing gives bit-identical K_v, A_v, dT_dt and dS_dt with the surrogates on and off, on columns
   whose boundary-layer base and convective layers cross interfaces, in float64 and in float32 with x64 off.
3. With the boundary-layer base surrogate, d(sum K_v)/d Ri_crit is finite and differs from the exact gradient;
   with surrogates off it is the exact gradient.
4. An unknown switch name raises; the default is off.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing import kpp as kpp_mod
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _columns(n=16, nlev=40, seed=0):
    from legoesm.ocean.vertical import create_z_star_from_thicknesses
    from legoesm.ocean.eos import wright_eos
    rng = np.random.default_rng(seed)
    dz = np.concatenate([np.full(20, 5.0), np.full(nlev - 20, 20.0)])
    zc = create_z_star_from_thicknesses(jnp.asarray(dz))
    z = np.cumsum(dz) - 0.5 * dz
    h = rng.uniform(20, 90, n)[:, None]
    T = 27.0 - 12.0 * (1 + np.tanh((z[None] - h) / 15.0)) / 2 + rng.normal(0, 0.02, (n, nlev))
    T[:4, :3] -= 0.2 * np.arange(3)[None, ::-1]          # a few statically unstable tops (convective switch)
    S = np.full_like(T, 35.0)
    u = 0.8 * np.exp(-((z[None] - 100.0) / 40.0) ** 2) - 0.4 * np.exp(-z[None] / 20.0) * rng.uniform(0.5, 1.5, (n, 1))
    v = 0.05 * rng.normal(size=(n, nlev))
    T, S, u, v = (jnp.asarray(a) for a in (T, S, u, v))
    rho = wright_eos(T, S, jnp.zeros_like(T))
    Bf = jnp.asarray(rng.uniform(-3e-7, 3e-7, n))
    return zc, T, S, u, v, rho, Bf


def _run(cfg, zc, T, S, u, v, rho, Bf):
    n = T.shape[0]
    return kpp_mod.kpp_vertical_mixing(u, v, T, S, rho, jnp.zeros(n), zc, jnp.ones(n), cfg,
                                       tau_x=jnp.full(n, -0.06), tau_y=jnp.full(n, 0.01), B_f=Bf,
                                       Q_sfc_T=jnp.full(n, -2e-5), Q_sfc_S=jnp.zeros(n), apply_diffusion=False)


def test_gate_forward_exact_and_backward_formula():
    gate = kpp_mod._make_surrogate_gate(0.1)
    x = jnp.asarray([-0.3, -1e-12, 0.0, 1e-12, 0.05, 0.4])
    a = jnp.asarray([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    b = jnp.asarray([1.0, 1.5, 0.5, 2.5, -1.0, 0.0])
    np.testing.assert_array_equal(np.asarray(gate(x, a, b)), np.asarray(jnp.where(x > 0, a, b)))
    gx, ga, gb = jax.grad(lambda x, a, b: jnp.sum(gate(x, a, b)), argnums=(0, 1, 2))(x, a, b)
    s = 1.0 / (1.0 + np.exp(-np.asarray(x) / 0.1))
    np.testing.assert_allclose(np.asarray(gx), (np.asarray(a) - np.asarray(b)) * s * (1 - s) / 0.1, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(ga), s, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(gb), 1 - s, rtol=1e-12)


def test_gate_broadcasting_and_width_check():
    gate = kpp_mod._make_surrogate_gate(0.1)
    x = jnp.asarray([[-0.2], [0.3]])                      # (2, 1)
    a = jnp.asarray(2.0)                                  # scalar
    b = jnp.ones((2, 3))
    gx, ga, gb = jax.grad(lambda x, a, b: jnp.sum(gate(x, a, b)), argnums=(0, 1, 2))(x, a, b)
    assert gx.shape == x.shape and ga.shape == a.shape and gb.shape == b.shape
    s = 1.0 / (1.0 + np.exp(-np.array([[-0.2], [0.3]]) / 0.1))
    np.testing.assert_allclose(float(ga), float(3 * s.sum()), rtol=1e-12)     # a broadcast to 6 cells
    np.testing.assert_allclose(np.asarray(gb), np.broadcast_to(1 - s, (2, 3)), rtol=1e-12)
    for w in (0.0, -0.1):
        with pytest.raises(ValueError):
            kpp_mod._make_surrogate_gate(w)


@pytest.mark.parametrize("switches", ["bl_base", "convective", "bl_base,convective"])
def test_forward_bitwise_unchanged(switches):
    cols = _columns()
    base = KPPConfig(vt2_velocity_scale="candidate", actual_buoyancy_flux=True)
    off = _run(base, *cols)
    on = _run(base._replace(gradient_surrogates=True, surrogate_switches=switches), *cols)
    for name in ("K_v", "A_v", "dT_dt", "dS_dt"):
        np.testing.assert_array_equal(np.asarray(getattr(on, name)), np.asarray(getattr(off, name)), err_msg=name)


@pytest.mark.parametrize("switches", ["bl_base", "bl_base,convective"])
def test_forward_bitwise_unchanged_float32(switches):
    jax.config.update("jax_enable_x64", False)            # the autouse fixture restores it
    zc, T, S, u, v, rho, Bf = _columns()
    cols = (zc,) + tuple(jnp.asarray(a, dtype=jnp.float32) for a in (T, S, u, v, rho, Bf))
    base = KPPConfig(vt2_velocity_scale="candidate", actual_buoyancy_flux=True)
    off = jax.jit(lambda: _run(base, *cols))()
    on = jax.jit(lambda: _run(base._replace(gradient_surrogates=True, surrogate_switches=switches), *cols))()
    for name in ("K_v", "A_v", "dT_dt", "dS_dt"):
        assert getattr(on, name).dtype == getattr(off, name).dtype
        np.testing.assert_array_equal(np.asarray(getattr(on, name)), np.asarray(getattr(off, name)), err_msg=name)


def test_switches_actually_cross():
    """The test columns exercise both switches (otherwise the bitwise test above is vacuous)."""
    zc, T, S, u, v, rho, Bf = _columns()
    out = _run(KPPConfig(vt2_velocity_scale="candidate", actual_buoyancy_flux=True), zc, T, S, u, v, rho, Bf)
    K = np.asarray(out.K_v)
    assert (K >= 0.9).any()                               # K_conv = 1 m2/s present
    assert (np.diff(K, axis=-1) != 0).any()


def test_surrogate_gradient_finite_and_default_exact():
    cols = _columns()
    base = KPPConfig(vt2_velocity_scale="candidate", actual_buoyancy_flux=True)

    def f(rc, cfg):
        return jnp.sum(_run(cfg._replace(Ri_crit=rc), *cols).K_v)
    g_exact = float(jax.grad(f)(jnp.asarray(0.3), base))
    g_sur = float(jax.grad(f)(jnp.asarray(0.3), base._replace(gradient_surrogates=True, surrogate_switches="bl_base")))
    assert np.isfinite(g_exact) and np.isfinite(g_sur)
    assert g_sur != g_exact                               # the base switch now contributes
    assert KPPConfig().gradient_surrogates is False


def test_unknown_switch_raises():
    cols = _columns(n=2)
    cfg = KPPConfig(gradient_surrogates=True, surrogate_switches="bl_base,nonsense")
    with pytest.raises(ValueError):
        _run(cfg, *cols)
