"""Direct unit tests for the prognostic EKE closure (build-spec gate E1).

Tests the pure closure properties: kappa_GM monotone + nonnegative in E, the
production form (kappa_GM * sigma^2), dissipation sign + E^{3/2} scaling, the
mixing-length floor, and finiteness — independent of the state/step coupling.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.physics.lateral_mixing.eke import (
    EKEConfig,
    eke_kappa_gm,
    eke_local_tendency,
    eke_mixing_length,
)


def test_eke_config_defaults_match_veros_acc():
    cfg = EKEConfig()
    assert cfg.c_k == 0.4
    assert cfg.c_eps == 0.5
    assert cfg.l_min == 100.0


def test_mixing_length_floor():
    cfg = EKEConfig(l_min=100.0)
    L_rossby = jnp.array([10.0, 100.0, 5.0e4])
    L = eke_mixing_length(L_rossby, cfg)
    assert float(L[0]) == 100.0      # floored
    assert float(L[1]) == 100.0      # at the floor
    assert float(L[2]) == 5.0e4      # above the floor, unchanged


def test_kappa_gm_nonnegative_and_monotone_in_E():
    cfg = EKEConfig()
    L = jnp.full((20,), 3.0e4)
    E = jnp.linspace(0.0, 1.0, 20)
    kappa = eke_kappa_gm(E, L, cfg)
    assert jnp.all(kappa >= 0.0), "kappa_GM must be non-negative"
    # monotone non-decreasing in E (sqrt).
    assert jnp.all(jnp.diff(kappa) >= -1e-12), "kappa_GM must be monotone in E"
    # exact form away from the regulariser.
    E1 = jnp.array([0.25]); L1 = jnp.array([3.0e4])
    np.testing.assert_allclose(
        np.asarray(eke_kappa_gm(E1, L1, cfg)),
        np.asarray(cfg.c_k * L1 * jnp.sqrt(E1)), rtol=1e-6,
    )


def test_kappa_gm_capped():
    cfg = EKEConfig(kappa_gm_max=2.0e3)
    kappa = eke_kappa_gm(jnp.array([1.0e6]), jnp.array([1.0e5]), cfg)
    assert float(kappa[0]) == 2.0e3


def test_tendency_zero_at_zero_E():
    """At E=0: production (kappa~0) and dissipation (E^{3/2}=0) both vanish."""
    cfg = EKEConfig()
    E = jnp.zeros((5,)); sigma = jnp.full((5,), 1.0e-5); L = jnp.full((5,), 3.0e4)
    t = eke_local_tendency(E, sigma, L, cfg)
    assert float(jnp.max(jnp.abs(t))) < 1e-15


def test_production_form_and_sign():
    """Production = kappa_GM * sigma^2 >= 0; equals the closed form."""
    cfg = EKEConfig()
    E = jnp.array([0.04]); sigma = jnp.array([2.0e-5]); L = jnp.array([3.0e4])
    # With dissipation subtracted; isolate by checking production-only via a
    # tiny E where dissipation (E^{3/2}) is sub-dominant, plus the closed form.
    kappa = eke_kappa_gm(E, L, cfg)
    prod = kappa * sigma ** 2
    diss = cfg.c_eps * E ** 1.5 / L
    np.testing.assert_allclose(
        np.asarray(eke_local_tendency(E, sigma, L, cfg)),
        np.asarray(prod - diss), rtol=1e-10,
    )
    assert float(prod[0]) >= 0.0


def test_dissipation_dominates_at_large_E():
    """With no production (sigma=0) the tendency is pure dissipation: <= 0 and
    scales as E^{3/2}."""
    cfg = EKEConfig()
    L = jnp.array([3.0e4])
    sigma0 = jnp.array([0.0])
    for E in (jnp.array([0.01]), jnp.array([0.1]), jnp.array([1.0])):
        t = eke_local_tendency(E, sigma0, L, cfg)
        assert float(t[0]) <= 0.0, "dissipation-only tendency must be <= 0"
    # E^{3/2} scaling: doubling... 8x E -> 8^{1.5}=~22.6x dissipation magnitude.
    t1 = -float(eke_local_tendency(jnp.array([0.1]), sigma0, L, cfg)[0])
    t8 = -float(eke_local_tendency(jnp.array([0.8]), sigma0, L, cfg)[0])
    np.testing.assert_allclose(t8 / t1, 8.0 ** 1.5, rtol=1e-6)


def test_tendency_finite_on_field():
    cfg = EKEConfig()
    rng = np.random.default_rng(0)
    E = jnp.asarray(np.abs(rng.standard_normal((8, 16))) * 0.05)
    sigma = jnp.asarray(np.abs(rng.standard_normal((8, 16))) * 1e-5)
    L = jnp.asarray(rng.uniform(1e4, 5e4, (8, 16)))
    t = eke_local_tendency(E, sigma, L, cfg)
    assert t.shape == (8, 16)
    assert jnp.all(jnp.isfinite(t))


# ---------------------------------------------------------------------------
# E2 — GM/Redi coupling (prognostic kappa_GM) + config + validation
# ---------------------------------------------------------------------------


def test_gmredi_config_accepts_eke():
    """GMRediConfig has an optional eke field (presence-based selection)."""
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    assert GMRediConfig().eke is None
    cfg = GMRediConfig(eke=EKEConfig())
    assert isinstance(cfg.eke, EKEConfig)


def test_validate_eke_config_raises_on_bad_params():
    from legoesm.ocean.physics.lateral_mixing.eke import validate_eke_config
    validate_eke_config(EKEConfig())  # default must pass
    for bad in (EKEConfig(c_k=0.0), EKEConfig(c_eps=-1.0), EKEConfig(l_min=0.0),
                EKEConfig(kappa_gm_max=0.0)):
        try:
            validate_eke_config(bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


def _eke_coupling_inputs(E_val):
    import numpy as np
    from legoesm.ocean.vertical import create_ocean_z_star
    nlat, nlon, nlev = 4, 6, 5
    z = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    # Stable stratification: rho increases with depth (k index).
    rho = jnp.asarray(
        1025.0 + np.linspace(0.0, 2.0, nlev)[None, None, :]
        * np.ones((nlat, nlon, 1))
    )
    S_x = jnp.full((nlat, nlon, nlev - 1), 1.0e-3)
    S_y = jnp.full((nlat, nlon, nlev - 1), 5.0e-4)
    jac = jnp.ones((nlat, nlon))
    f = jnp.full((nlat, nlon), 1.0e-4)
    E = jnp.full((nlat, nlon), float(E_val))
    return E, rho, S_x, S_y, z, jac, f


def test_compute_eke_kappa_gm_prognostic_and_monotone():
    """The prognostic kappa_GM is >= 0, finite, increases with E, and returns the
    Eady rate + mixing length for the EKE source/sink. Reuses the shared
    _eady_growth_and_length (no duplicate numerics)."""
    from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_eke_kappa_gm,
    )
    vcfg, ecfg = VisbeckConfig(), EKEConfig()
    E_lo, *rest = _eke_coupling_inputs(0.01)
    k_lo, sig, L = compute_eke_kappa_gm(E_lo, *rest, vcfg, ecfg)
    E_hi, *rest_hi = _eke_coupling_inputs(0.25)
    k_hi, _, _ = compute_eke_kappa_gm(E_hi, *rest_hi, vcfg, ecfg)
    assert k_lo.shape == (4, 6) and sig.shape == (4, 6) and L.shape == (4, 6)
    assert jnp.all(jnp.isfinite(k_lo)) and jnp.all(jnp.isfinite(sig))
    assert jnp.all(k_lo >= 0.0)
    assert jnp.all(L >= ecfg.l_min)              # mixing length floored
    assert float(jnp.mean(k_hi)) > float(jnp.mean(k_lo))  # kappa grows with E
    # E=0 -> kappa_GM = 0 (no prognostic mixing without eddy energy).
    E0, *rest0 = _eke_coupling_inputs(0.0)
    k0, _, _ = compute_eke_kappa_gm(E0, *rest0, vcfg, ecfg)
    assert float(jnp.max(k0)) < 1e-6


# ---------------------------------------------------------------------------
# E3 — positivity (semi-implicit dissipation, no clipping)
# ---------------------------------------------------------------------------


def test_E3_positivity_preserved_over_steps():
    """The local EKE update keeps E >= 0 over many steps for a wide range of
    E0/sigma/dt — by construction (semi-implicit), no floor/clip needed."""
    from legoesm.ocean.physics.lateral_mixing.eke import eke_apply_local_source
    cfg = EKEConfig()
    rng = np.random.default_rng(7)
    L = jnp.asarray(rng.uniform(1e4, 5e4, (6, 8)))
    for dt in (300.0, 3600.0, 86400.0, 10.0 * 86400.0):  # incl. huge dt
        E = jnp.asarray(np.abs(rng.standard_normal((6, 8))) * 0.05)
        sigma = jnp.asarray(np.abs(rng.standard_normal((6, 8))) * 1e-5)
        for _ in range(50):
            E = eke_apply_local_source(E, sigma, L, cfg, dt)
            assert jnp.all(E >= 0.0), f"E went negative at dt={dt}"
            assert jnp.all(jnp.isfinite(E))


def test_E3_grows_from_small_E_when_forced():
    """With production (sigma>0), E grows away from ~0 toward a bounded steady
    state (production dominates near 0; dissipation ~ E^{3/2} caps it)."""
    from legoesm.ocean.physics.lateral_mixing.eke import eke_apply_local_source
    cfg = EKEConfig()
    L = jnp.full((1,), 3.0e4)
    sigma = jnp.full((1,), 3.0e-5)
    E = jnp.full((1,), 1.0e-6)
    traj = [float(E[0])]
    for _ in range(400):
        E = eke_apply_local_source(E, sigma, L, cfg, 3600.0)
        traj.append(float(E[0]))
    assert traj[-1] > traj[0], "E should grow under forcing"
    assert jnp.isfinite(E[0]) and float(E[0]) < 1e3, "E should stay bounded"
    # near steady state: last step changes little.
    assert abs(traj[-1] - traj[-2]) < 0.05 * traj[-1] + 1e-9


# ---------------------------------------------------------------------------
# E5 — differentiability
# ---------------------------------------------------------------------------


def test_E5_differentiable_through_closure_and_coupling():
    """jax.grad through the EKE local update + the prognostic kappa_GM coupling is
    finite and nonzero."""
    from legoesm.ocean.physics.lateral_mixing.eke import eke_apply_local_source
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_eke_kappa_gm,
    )
    from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
    cfg, vcfg = EKEConfig(), VisbeckConfig()
    E0, rho, S_x, S_y, z, jac, f = _eke_coupling_inputs(0.04)

    def loss(E):
        kappa, sigma, L = compute_eke_kappa_gm(
            E, rho, S_x, S_y, z, jac, f, vcfg, cfg)
        E1 = eke_apply_local_source(E, sigma, L, cfg, 3600.0)
        return jnp.sum(kappa ** 2) + jnp.sum(E1 ** 2)

    g = jax.grad(loss)(E0)
    assert jnp.all(jnp.isfinite(g)), "non-finite grad through EKE closure/coupling"
    assert float(jnp.max(jnp.abs(g))) > 0.0, "zero grad — path not differentiated"
