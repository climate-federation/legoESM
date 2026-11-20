"""core-0030: KPP's N^2 is the local buoyancy gradient, not the difference of in-situ densities (ENSO-project notes (iadicarlo/legoESM-enso) 88 6.13).

The in-situ difference carries the compressibility of seawater, about +4e-5 s^-2, in V_t^2, the interior Richardson
number and the convective trigger. LMD94 (Eqs. 23, 27), MOM6, MITgcm and POP all difference two parcels at one
pressure. Pass criteria C1 to C6 of ENSO-project notes (iadicarlo/legoESM-enso) 88 6.13:
1. C1, uniform column: local N^2 exactly zero, in-situ 4.0e-5 to 4.2e-5 s^-2 (float32 and float64 for the first);
2. C2, local form equals g (alpha dT/dz - beta dS/dz) from autodiff of the EOS, and prints the size of the choice of
   reference pressure and of the p = 0 bulk Richardson numerator against level pressure;
3. C3, "insitu_signed" is bit for bit the earlier expression;
4. C4, finite gradients on stratified, uniform, unstable and dry columns, float32 and float64, eager and jit;
5. C5, a weakly unstable interface below the boundary layer gets K_conv only with "adiabatic"; a stable interface
   gets a smaller Ri;
6. C6, a linear EOS gives identical results in both modes; an unknown mode raises.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import compute_buoyancy_frequency, rho_0, wright_eos
from legoesm.ocean.physics.vertical_mixing import kpp as kpp_mod
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.vertical import create_z_star_from_thicknesses

NLEV, DZ = 40, 5.0
G = constants.g


@pytest.fixture(params=[True, False], ids=["float64", "float32"])
def x64(request):
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", request.param)
    yield request.param
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture()
def float64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _depths():
    return np.arange(NLEV) * DZ + 0.5 * DZ


def _zc():
    return create_z_star_from_thicknesses(jnp.full(NLEV, DZ))


def _insitu_rho(T, S, eos=wright_eos):
    p = rho_0 * G * jnp.asarray(_depths())
    return eos(T, S, p)


def _n2(rho, T, S, mode, eos_fn=None):
    z = _zc()
    return kpp_mod._kpp_interface_n2(rho, T, S, jnp.zeros((1,)), z, jnp.ones((1,)), KPPConfig(n2_mode=mode), G,
                                     eos_fn=eos_fn)


def _column(profile="stratified"):
    z = _depths()
    if profile == "uniform":
        T, S = np.full(NLEV, 20.0), np.full(NLEV, 35.0)
    elif profile == "unstable":
        T = 20.0 + 0.02 * (z - z[0]) / DZ               # warming with depth: statically unstable, N^2 about -6e-5
        S = np.full(NLEV, 35.0)
    else:
        T = 28.0 - 14.0 * (1.0 + np.tanh((z - 60.0) / 15.0)) / 2.0
        S = 35.0 + 0.5 * (1.0 + np.tanh((z - 80.0) / 30.0)) / 2.0
    return jnp.asarray(T[None]), jnp.asarray(S[None])


def _kpp(T, S, mode, eos_fn=None, u=None, v=None, j=1.0, tau=0.03, zc=None):
    eos = eos_fn if eos_fn is not None else wright_eos
    rho = _insitu_rho(T, S, eos=eos)
    uu = jnp.zeros_like(T) if u is None else u
    vv = jnp.zeros_like(T) if v is None else v
    n = T.shape[0]
    zcol = _zc() if zc is None else zc          # built outside jit (z-star coordinate setup is not traceable)
    return kpp_mod.kpp_vertical_mixing(uu, vv, T, S, rho, jnp.zeros((n,)), zcol, jnp.full((n,), j),
                                       KPPConfig(n2_mode=mode), tau_x=jnp.full((n,), tau), tau_y=jnp.zeros((n,)),
                                       B_f=jnp.full((n,), -5e-8), Q_sfc_T=jnp.full((n,), -2e-5),
                                       Q_sfc_S=jnp.zeros((n,)), apply_diffusion=False, eos_fn=eos_fn)


def test_default_is_local():
    assert KPPConfig().n2_mode == "adiabatic"


def test_bad_mode_raises(float64):
    T, S = _column()
    with pytest.raises(ValueError, match="n2_mode"):
        _n2(_insitu_rho(T, S), T, S, "insitu")


def test_c1_uniform_column_zero_local_and_insitu_bias(x64):
    T, S = _column("uniform")
    rho = _insitu_rho(T, S)
    local = np.asarray(_n2(rho, T, S, "adiabatic"))
    assert np.array_equal(local, np.zeros_like(local))
    insitu = np.asarray(_n2(rho, T, S, "insitu_signed"))
    if x64:
        assert insitu.min() > 4.0e-5 and insitu.max() < 4.2e-5


def test_c2_local_form_equals_autodiff_and_pressure_choice(float64):
    T, S = _column("stratified")
    rho = _insitu_rho(T, S)
    local = np.asarray(_n2(rho, T, S, "adiabatic"))[0]
    z = _depths()
    p_up, p_lo = rho_0 * G * z[:-1], rho_0 * G * z[1:]
    p_if = 0.5 * (p_up + p_lo)
    Ti, Si = np.asarray(T)[0], np.asarray(S)[0]
    tm, sm = 0.5 * (Ti[:-1] + Ti[1:]), 0.5 * (Si[:-1] + Si[1:])
    drho_dt = np.asarray(jax.vmap(jax.grad(wright_eos, 0))(jnp.asarray(tm), jnp.asarray(sm), jnp.asarray(p_up)))
    drho_ds = np.asarray(jax.vmap(jax.grad(wright_eos, 1))(jnp.asarray(tm), jnp.asarray(sm), jnp.asarray(p_up)))
    ana = -(G / rho_0) * (drho_dt * (Ti[:-1] - Ti[1:]) + drho_ds * (Si[:-1] - Si[1:])) / DZ
    assert np.max(np.abs(local - ana)) <= 1.0e-6

    # sizes reported in ENSO-project notes (iadicarlo/legoESM-enso) 88 6.13 (read with pytest -s)
    def n2_at(p):
        ru = np.asarray(wright_eos(jnp.asarray(Ti[:-1]), jnp.asarray(Si[:-1]), jnp.asarray(p)))
        rl = np.asarray(wright_eos(jnp.asarray(Ti[1:]), jnp.asarray(Si[1:]), jnp.asarray(p)))
        return -(G / rho_0) * (ru - rl) / DZ
    upper, iface, lower = n2_at(p_up), n2_at(p_if), n2_at(p_lo)
    p = rho_0 * G * z
    bulk0 = np.asarray(wright_eos(jnp.asarray(Ti), jnp.asarray(Si), jnp.zeros(NLEV)))
    bulk0 = bulk0 - bulk0[0]
    tk = np.full(NLEV, Ti[0])
    sk = np.full(NLEV, Si[0])
    bulkp = np.asarray(wright_eos(jnp.asarray(Ti), jnp.asarray(Si), jnp.asarray(p))
                       - wright_eos(jnp.asarray(tk), jnp.asarray(sk), jnp.asarray(p)))
    rel = np.abs(bulkp[1:] - bulk0[1:]) / np.maximum(np.abs(bulk0[1:]), 1e-6)
    print(f"max |N2(upper)-N2(iface)| {np.max(np.abs(upper - iface)):.2e}, "
          f"|N2(lower)-N2(iface)| {np.max(np.abs(lower - iface)):.2e} s-2")
    print(f"bulk numerator p=0 against level pressure, largest relative difference {np.nanmax(rel):.3f}"
          f" at {z[1:][np.nanargmax(rel)]:.0f} m")
    assert np.max(np.abs(upper - iface)) <= 1.0e-6


def test_c3_insitu_signed_is_earlier_expression(float64):
    T, S = _column("stratified")
    rho = _insitu_rho(T, S)
    z = _zc()
    earlier = compute_buoyancy_frequency(rho, z.dz_ref, jnp.ones((1,)))
    np.testing.assert_array_equal(np.asarray(_n2(rho, T, S, "insitu_signed")), np.asarray(earlier))
    # the whole KPP call: only the N^2 helper differs between the two
    orig = kpp_mod._kpp_interface_n2
    try:
        kpp_mod._kpp_interface_n2 = lambda rho_, *a, **k: compute_buoyancy_frequency(rho_, z.dz_ref, jnp.ones((1,)))
        forced = _kpp(T, S, "adiabatic")
    finally:
        kpp_mod._kpp_interface_n2 = orig
    out = _kpp(T, S, "insitu_signed")
    for a, b in ((out.K_v, forced.K_v), (out.A_v, forced.A_v), (out.h_bl, forced.h_bl), (out.dT_dt, forced.dT_dt)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("profile,j", [("stratified", 1.0), ("uniform", 1.0), ("unstable", 1.0), ("dry", 0.0)])
def test_c4_gradients_finite_eager_and_jit(x64, profile, j):
    T, S = _column("uniform" if profile == "dry" else profile)
    if profile == "dry":
        T, S = jnp.zeros_like(T), jnp.zeros_like(S)
    u = 0.05 * jnp.sin(jnp.asarray(_depths())[None] / 20.0)
    zc = _zc()

    def loss(T_, S_):
        out = _kpp(T_, S_, "adiabatic", u=u, j=j, zc=zc)
        return jnp.sum(out.K_v) + jnp.sum(out.A_v) + jnp.sum(out.h_bl)

    g_eager = jax.grad(loss, argnums=(0, 1))(T, S)
    g_jit = jax.jit(jax.grad(loss, argnums=(0, 1)))(T, S)
    for ge, gj in zip(g_eager, g_jit):
        assert np.all(np.isfinite(np.asarray(ge))), f"{profile}: eager gradient not finite"
        assert np.all(np.isfinite(np.asarray(gj))), f"{profile}: jit gradient not finite"
    out = _kpp(T, S, "adiabatic", u=u, j=j, zc=zc)
    assert np.all(np.isfinite(np.asarray(out.K_v))) and np.all(np.isfinite(np.asarray(out.h_bl)))
    # Eager and jit agree to float32 precision: the Wright EOS runs in float32 by the default precision policy
    # even with x64 state, and the XLA operation order then differs by rounding (observed 3e-4 relative on this
    # sum of K_v, A_v and h_bl, which is sensitive through the sigmoid crossing of the boundary layer depth).
    np.testing.assert_allclose(np.asarray(loss(T, S)), np.asarray(jax.jit(loss)(T, S)), rtol=1e-3, atol=1e-3)


def test_c5_weak_instability_below_the_bl_gets_kconv_only_when_local(float64):
    z = _depths()
    # stable column, then a warming of 0.017 K across one interface at 150 m (true N^2 about -1e-5, in-situ positive)
    T = 28.0 - 0.05 * z
    k = int(np.argmin(np.abs(z - 150.0)))
    T[k + 1:] += 0.017 + 0.05 * DZ
    T, S = jnp.asarray(T[None]), jnp.full((1, NLEV), 35.0)
    rho = _insitu_rho(T, S)
    local_n2 = np.asarray(_n2(rho, T, S, "adiabatic"))[0, k]
    insitu_n2 = np.asarray(_n2(rho, T, S, "insitu_signed"))[0, k]
    assert -3e-5 < local_n2 < -2e-6 and insitu_n2 > 0.0, (local_n2, insitu_n2)
    out_l = _kpp(T, S, "adiabatic", tau=0.005)
    out_i = _kpp(T, S, "insitu_signed", tau=0.005)
    assert float(out_l.h_bl[0]) < 100.0 and float(out_i.h_bl[0]) < 100.0
    assert float(out_l.K_v[0, k]) > 0.5
    assert float(out_i.K_v[0, k]) < 0.01


def test_c5_stable_interface_smaller_ri_and_more_shear_mixing(float64):
    z = _depths()
    # N^2 about 8e-5 s^-2 (in situ 1.2e-4), and a shear layer of S^2 = 3.2e-4 s^-2 at 140 to 160 m, in the deep-cycle
    # range, so that Ri sits below Ri_0 with both forms and well below the boundary layer base
    T = 25.0 - 3.3e-2 * z
    T, S = jnp.asarray(T[None]), jnp.full((1, NLEV), 35.0)
    u = jnp.asarray(np.clip(1.8e-2 * (z - 140.0), 0.0, 0.36)[None])
    out_l = _kpp(T, S, "adiabatic", u=u, tau=0.005)
    out_i = _kpp(T, S, "insitu_signed", u=u, tau=0.005)
    k = int(np.argmin(np.abs(z - 148.0)))
    assert float(out_l.h_bl[0]) < 100.0 and float(out_i.h_bl[0]) < 100.0
    assert float(out_l.K_v[0, k]) > float(out_i.K_v[0, k])


def test_c6_linear_eos_modes_identical(float64):
    def linear(T_, S_, p_):
        return 1027.0 - 0.2 * (T_ - 10.0) + 0.8 * (S_ - 35.0) + 0.0 * p_

    T, S = _column("stratified")
    rho = _insitu_rho(T, S, eos=linear)
    a = np.asarray(_n2(rho, T, S, "adiabatic", eos_fn=linear))
    b = np.asarray(_n2(rho, T, S, "insitu_signed", eos_fn=linear))
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_partial_cell_n2_uses_live_centre_spacing(float64):
    """On a partial-cell coordinate the adiabatic N^2 divides by the live centre spacing of h_k."""
    from types import SimpleNamespace
    from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic, compute_hydrostatic_pressure
    T, S = _column()
    rho = _insitu_rho(T, S)
    z = _zc()
    h_k = jnp.asarray(z.dz_ref)[None, :].at[0, -1].set(0.3 * float(z.dz_ref[-1]))  # a shortened bottom cell
    zpc = SimpleNamespace(dz_ref=z.dz_ref, is_active=jnp.ones_like(h_k))
    J = jnp.ones((1,))
    got = kpp_mod._kpp_interface_n2(rho, T, S, jnp.zeros((1,)), zpc, J, KPPConfig(), G, h_k=h_k)
    p = compute_hydrostatic_pressure(rho, jnp.zeros((1,)), z.dz_ref, J, rho_0, G, h_actual=h_k)
    want = compute_buoyancy_frequency_adiabatic(T, S, p, z.dz_ref, J, rho_ref=rho_0, g=G,
                                                dz_half=0.5 * (h_k[..., :-1] + h_k[..., 1:]))
    ref = compute_buoyancy_frequency_adiabatic(T, S, p, z.dz_ref, J, rho_ref=rho_0, g=G)
    assert np.allclose(np.asarray(got), np.asarray(want), rtol=1e-12, atol=0.0)
    assert abs(float(got[0, -1]) - float(ref[0, -1])) > 1e-3 * abs(float(ref[0, -1]))  # the bottom interface differs
