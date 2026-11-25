"""core-0029: CATKE's N^2 is the local buoyancy gradient, as Oceananigans' CATKE (SeawaterBuoyancy ∂z_b =
g (alpha ∂zT - beta ∂zS)), not the difference of in-situ densities, which carries the compressibility of seawater
(about +4e-5 s^-2 near the surface) and so hid static instability from the closure."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic, compute_ocean_rho_and_pressure  # noqa: E402
from legoesm.ocean.physics.combined import OceanPhysicsConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing._shared import compute_N2  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.config import CATKEConfig, VerticalMixingConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.k_profiles import compute_vertical_K_profiles  # noqa: E402
from legoesm.ocean.scm import make_ocean_column_state  # noqa: E402
from legoesm.ocean.state import OceanSurfaceForcing  # noqa: E402
from legoesm.ocean.vertical import compute_ocean_jacobian, create_z_star_from_thicknesses  # noqa: E402

NLEV, DZ = 30, 5.0
SHP = (1, 1, 1)


def _column(T):
    zcol = create_z_star_from_thicknesses(jnp.full(NLEV, DZ))
    st = make_ocean_column_state(NLEV, T_profile=jnp.asarray(T), S_profile=jnp.full(NLEV, 34.9), z_coord=zcol)
    return st, zcol


def _zc():
    return np.arange(NLEV) * DZ + 0.5 * DZ


def _run(T, mode, q_net, nstep=1, e0=1.0e-5, eos_fn=None, tau=0.01):
    st, zcol = _column(T)
    cfg = OceanPhysicsConfig(vertical_mixing=VerticalMixingConfig(scheme="catke", catke=CATKEConfig(n2_mode=mode)))
    sfc = OceanSurfaceForcing(q_net=jnp.full(SHP, q_net), tau_x=jnp.full(SHP, tau), tau_y=jnp.zeros(SHP),
                              freshwater=jnp.zeros(SHP))
    e = jnp.full((1, 1, 1, NLEV - 1), e0)
    for _ in range(nstep):
        K, A, e = compute_vertical_K_profiles(st, zcol, sfc, cfg, tke_old=e, dt_tke=1800.0, return_tke=True,
                                              eos_fn=eos_fn)
    return np.asarray(K[0, 0, 0]), np.asarray(e[0, 0, 0])


def test_default_is_local():
    assert CATKEConfig().n2_mode == "adiabatic"


def test_in_situ_difference_is_biased_and_local_is_not():
    st, zcol = _column(np.full(NLEV, 26.0))
    J = compute_ocean_jacobian(st.eta.data, st.H_bathy.data, zcol)
    rho, p = compute_ocean_rho_and_pressure(st, zcol, J)
    n2_insitu = np.asarray(compute_N2(rho, jnp.full((1, 1, 1, NLEV - 1), DZ), 1025.0, n2_mode="insitu_signed"))
    n2_local = np.asarray(compute_buoyancy_frequency_adiabatic(st.T.data, st.S.data, p, zcol.dz_ref, J))
    assert np.median(n2_insitu) == pytest.approx(4.05e-5, rel=0.05)    # the compressibility of seawater
    assert np.max(np.abs(n2_local)) < 1e-9


def test_inversion_under_warming_surface_is_mixed():
    # T rises by 0.01 K/m from 15 to 50 m (local N2 about -3e-5 s-2) under a warming surface (q_net +150 W m-2):
    # Oceananigans' CATKE mixes it through the negative-Ri shear length and the buoyancy production of TKE.
    zc = _zc()
    T = 26.0 + 0.01 * np.clip(zc - 15.0, 0.0, 35.0)
    zi = zc[:-1] + 0.5 * DZ
    lay = (zi >= 25.0) & (zi <= 45.0)
    K_new, e_new = _run(T, "adiabatic", 150.0, nstep=4)
    K_old, e_old = _run(T, "insitu_signed", 150.0, nstep=4)
    assert K_new[lay].min() > 0.1                                       # of order 0.5 m2/s, the reference estimate
    assert K_old[lay].max() < 0.01                                      # the earlier behaviour left it nearly unmixed


def test_linear_eos_modes_agree():
    # with no pressure dependence in the EOS the two forms are the same N2, so the same mixing
    def lin(T, S, p=None):
        return 1025.0 * (1.0 - 2.0e-4 * (T - 10.0) + 7.6e-4 * (S - 35.0))
    zc = _zc()
    T = 26.0 - 0.05 * np.clip(zc - 30.0, 0.0, None)
    K_a, e_a = _run(T, "adiabatic", -100.0, nstep=2, eos_fn=lin)
    K_i, e_i = _run(T, "insitu_signed", -100.0, nstep=2, eos_fn=lin)
    assert np.allclose(K_a, K_i, rtol=1e-10, atol=1e-14)
    assert np.allclose(e_a, e_i, rtol=1e-10, atol=1e-16)


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        _run(np.full(NLEV, 26.0), "clipped", 0.0)
