"""Unit tests for the closed-form live-pool (biomass + LAI) forward
(``legoesm.land.carbon.live_pool_forward``).

Two tiers: (1) PURE closed-form value / sign / AD checks (no spin-up, sub-second); and
(2) the CRUX closed-form-vs-spin-up FIDELITY gate -- ``precompute_fast_analytic_inputs``
(the default-parameter spin-up) records the annual-mean equilibrium live pools, and the
closed form at the defaults must reproduce them cover-weighted to ~10-15%.

Compute-node scale for the fidelity test (JIT-compiles the coupled land+carbon spin-up);
run via the sbatch/srun wrapper, NOT the login node.  ``JAX_ENABLE_X64=1``; forced CPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import numpy.testing as npt


def _npp_flags(n=3):
    npp = np.linspace(300.0, 900.0, n)          # gC/m2/yr
    woody = np.array([1.0, 0.0, 1.0])[:n]       # woody, herbaceous, woody
    evergreen = np.array([1.0, 0.0, 0.0])[:n]   # evergreen, deciduous, deciduous
    return npp, woody, evergreen


def test_biomass_lai_nonnegative_finite_shape():
    """biomass [kgC/m2] and LAI [m2/m2] are non-negative, finite, one value per archetype."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.live_pool_forward import compute_biomass_lai

    npp, woody, ever = _npp_flags()
    biomass, lai = compute_biomass_lai(npp, woody, ever, CarbonConfig(scheme="differland"))
    biomass, lai = np.asarray(biomass), np.asarray(lai)
    assert biomass.shape == (3,) and lai.shape == (3,)
    assert np.all(np.isfinite(biomass)) and np.all(np.isfinite(lai))
    assert np.all(biomass >= 0.0) and np.all(lai >= 0.0)


def test_lai_scales_inverse_lcma():
    """LAI = C_fol / LCMA -> doubling LCMA halves LAI exactly (C_fol is LCMA-independent)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.live_pool_forward import compute_biomass_lai

    npp, woody, ever = _npp_flags()
    _, lai1 = compute_biomass_lai(npp, woody, ever, CarbonConfig(scheme="differland", LCMA=50.0))
    _, lai2 = compute_biomass_lai(npp, woody, ever, CarbonConfig(scheme="differland", LCMA=100.0))
    npt.assert_allclose(np.asarray(lai2), 0.5 * np.asarray(lai1), rtol=1e-10)


def test_wood_scales_inverse_tor_wood():
    """C_wood = A_wood / (tor_wood * days_per_year) -> doubling tor_wood halves the wood
    pool (residence-time sign check); a woody archetype's biomass drops accordingly."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.live_pool_forward import compute_live_pools

    npp = np.array([800.0]); woody = np.array([1.0]); ever = np.array([1.0])
    _, _, cw1 = compute_live_pools(npp, woody, ever, CarbonConfig(scheme="differland", tor_wood=1e-4))
    _, _, cw2 = compute_live_pools(npp, woody, ever, CarbonConfig(scheme="differland", tor_wood=2e-4))
    npt.assert_allclose(np.asarray(cw2), 0.5 * np.asarray(cw1), rtol=1e-10)
    assert float(np.asarray(cw1)[0]) > 0.0


def test_evergreen_leaf_pool_scales_with_leaf_lifespan():
    """For an EVERGREEN archetype k_leaf = 1/leaf_lifespan, so C_fol = a_fol * leaf_lifespan *
    NPP is LINEAR in leaf_lifespan (longer leaf residence -> more foliage / LAI)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.live_pool_forward import compute_live_pools

    npp = np.array([600.0]); woody = np.array([1.0]); ever = np.array([1.0])  # evergreen
    cfol1, _, _ = compute_live_pools(npp, woody, ever, CarbonConfig(scheme="differland", leaf_lifespan=1.5))
    cfol2, _, _ = compute_live_pools(npp, woody, ever, CarbonConfig(scheme="differland", leaf_lifespan=3.0))
    npt.assert_allclose(np.asarray(cfol2), 2.0 * np.asarray(cfol1), rtol=1e-6)


def test_deciduous_sheds_faster_than_evergreen():
    """The DALEC990 deciduous leaf-fall pulse gives a LARGER annual leaf turnover than the
    continuous evergreen 1/leaf_lifespan, so a deciduous archetype holds LESS annual-mean
    foliage than the SAME archetype run evergreen (the fix for the LAI over-prediction)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.live_pool_forward import annual_leaf_turnover, compute_live_pools

    cfg = CarbonConfig(scheme="differland")
    k_ever, k_decid = (float(x) for x in annual_leaf_turnover(cfg))
    assert k_decid > k_ever > 0.0, (k_ever, k_decid)
    npt.assert_allclose(k_ever, 1.0 / cfg.leaf_lifespan, rtol=5e-3)   # ~1/leaf_lifespan
    npp = np.array([600.0]); woody = np.array([1.0])
    cfol_e, _, _ = compute_live_pools(npp, woody, np.array([1.0]), cfg)
    cfol_d, _, _ = compute_live_pools(npp, woody, np.array([0.0]), cfg)
    assert float(np.asarray(cfol_d)[0]) < float(np.asarray(cfol_e)[0])


def test_biomass_increases_with_npp():
    """More NPP -> more standing biomass (every pool input scales with NPP)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.live_pool_forward import compute_biomass_lai

    cfg = CarbonConfig(scheme="differland")
    woody = np.array([1.0]); ever = np.array([1.0])
    b_lo, _ = compute_biomass_lai(np.array([200.0]), woody, ever, cfg)
    b_hi, _ = compute_biomass_lai(np.array([900.0]), woody, ever, cfg)
    assert float(np.asarray(b_hi)[0]) > float(np.asarray(b_lo)[0]) > 0.0


def test_herbaceous_grows_no_wood_and_routes_to_roots():
    """A herbaceous archetype (is_woody=0) grows NO wood; the structural remainder is
    invested belowground (roots), matching step_carbon_differland's routing."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.live_pool_forward import compute_live_pools

    npp = np.array([700.0]); ever = np.array([0.0])
    cfg = CarbonConfig(scheme="differland")
    # Herbaceous: no wood pool.
    cfol_h, croot_h, cwood_h = compute_live_pools(npp, np.array([0.0]), ever, cfg)
    npt.assert_allclose(np.asarray(cwood_h), 0.0, atol=1e-12)
    assert float(np.asarray(croot_h)[0]) > 0.0
    # With EQUAL root/wood residence the herbaceous root pool (base + remainder) exceeds the
    # woody root pool (base only) by exactly the remainder / k -- the remainder is rerouted,
    # not lost.
    cfg_eq = CarbonConfig(scheme="differland", tor_root=1e-4, tor_wood=1e-4)
    _, croot_w, cwood_w = compute_live_pools(npp, np.array([1.0]), ever, cfg_eq)
    _, croot_h2, cwood_h2 = compute_live_pools(npp, np.array([0.0]), ever, cfg_eq)
    npt.assert_allclose(np.asarray(croot_h2), np.asarray(croot_w) + np.asarray(cwood_w),
                        rtol=1e-10)
    npt.assert_allclose(np.asarray(cwood_h2), 0.0, atol=1e-12)


def test_build_live_pool_forward_matches_compute():
    """The trainer's precompute-bound closure equals the all-in-one compute for any config."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.fast_analytic import FastAnalyticInputs
    from legoesm.land.carbon.live_pool_forward import (
        build_live_pool_forward, compute_biomass_lai,
    )

    npp, woody, ever = _npp_flags()
    pre = FastAnalyticInputs(
        lit_to_som_annual=np.zeros(3), a_wood_annual=np.zeros(3),
        soil_T_traj=np.full((3, 2), 285.0), soil_frozen_fraction=np.zeros(3),
        precip=np.full(3, 3e-5), dt_days=0.0417,
        npp_pos_annual=npp, is_woody=woody, is_evergreen=ever,
        live_ref_C_fol=np.zeros(3), live_ref_C_root=np.zeros(3), live_ref_C_wood=np.zeros(3))
    fn = build_live_pool_forward(pre)
    for cfg in (CarbonConfig(scheme="differland"),
                CarbonConfig(scheme="differland", f_fol=0.3, LCMA=80.0, tor_wood=2e-4)):
        b1, l1 = fn(cfg)
        b2, l2 = compute_biomass_lai(npp, woody, ever, cfg)
        npt.assert_allclose(np.asarray(b1), np.asarray(b2), rtol=1e-12)
        npt.assert_allclose(np.asarray(l1), np.asarray(l2), rtol=1e-12)


def test_live_pool_forward_differentiable_in_carbon_params():
    """AD de-risk: grad of a biomass+LAI MSE loss w.r.t. the CarbonConfig allocation /
    residence / phenology / LCMA leaves is FINITE and NON-ZERO (single-step graph, no
    spin-up).  Mixes an evergreen + deciduous archetype so both leaf-turnover branches (and
    hence leaf_lifespan's gradient path) are exercised."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.live_pool_forward import compute_biomass_lai
    from legoesm.training.param_collector import (
        apply_param_overrides, build_trainable_params,
    )

    npp = jnp.asarray([400.0, 700.0, 900.0])
    woody = jnp.asarray([1.0, 0.0, 1.0])
    ever = jnp.asarray([1.0, 0.0, 0.0])          # evergreen + two deciduous
    # Full tier-2 land.carbon set (includes the seven live-pool leaves); the biomass+lai
    # loss depends only on the live-pool subset, so the others' grad is a finite 0.
    params = build_trainable_params(
        active_scheme_keys={"land.carbon"}, tier="extended", dtype=jnp.float64)
    tgt_b = jnp.asarray([3.0, 1.0, 8.0])
    tgt_l = jnp.asarray([2.0, 0.8, 4.0])

    def loss(p):
        ov = p.to_overrides().get("land.carbon", {})
        cfg = apply_param_overrides(CarbonConfig(scheme="differland"), ov)
        biomass, lai = compute_biomass_lai(npp, woody, ever, cfg)
        return jnp.mean((biomass - tgt_b) ** 2) + jnp.mean((lai - tgt_l) ** 2)

    val, grads = eqx.filter_value_and_grad(loss)(params)
    assert np.isfinite(float(val))
    gmax = {c.field: float(jnp.max(jnp.abs(grads.raw_values[c.name])))
            for c in params.constraints}
    assert all(np.isfinite(v) for v in gmax.values()), gmax
    # The four leaves the biomass+lai loss directly depends on must all move it.
    for f in ("f_fol", "leaf_lifespan", "tor_wood", "LCMA"):
        assert gmax[f] > 1e-12, (f, gmax)


# ---------------------------------------------------------------------------
# CRUX: the closed form reproduces the spin-up equilibrium live pools at the defaults.
# ---------------------------------------------------------------------------
def test_live_pool_matches_spinup_equilibrium_at_defaults():
    """precompute_fast_analytic_inputs (the default-parameter spin-up) records the
    annual-mean equilibrium live pools; the closed-form live-pool forward at the defaults
    must reproduce the biomass / LAI cover-weighted to ~10-15% (the sign/units + NPP/
    allocation-wiring fidelity gate).  A tiny 3-cell world (tropical + temperate)."""
    import jax.numpy as jnp
    from legoesm.land.carbon.climate_features import reduce_climatology_to_features
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.global_init import (
        build_archetypes, precompute_fast_analytic_inputs,
    )
    from legoesm.land.carbon.live_pool_forward import build_live_pool_forward

    ncell, npft = 3, 17
    w = np.zeros((ncell, npft)); w[0, 4] = 1.0; w[1, 7] = 1.0; w[2, 13] = 1.0  # 13 = a grass
    months = np.arange(12)
    season = np.sin(2 * np.pi * months / 12.0)
    t = np.stack([298.0 + 2.0 * season, 281.0 + 14.0 * season, 288.0 + 8.0 * season])
    pr = np.stack([np.full(12, 6e-5), np.full(12, 2.5e-5), np.full(12, 3.0e-5)])
    sw = np.full((ncell, 12), 210.0); nr = np.full((ncell, 12), 90.0)
    feats = reduce_climatology_to_features(t, pr, sw, nr)
    soil = np.array(["loam", "loam", "loam"])
    tab, cid, cw = build_archetypes(
        w, feats, soil, np.ones(ncell, bool), k_per_pft=1, seed=0)

    spin = dict(n_spinup=40, n_verify=6, dt=7200.0, n_layers=6, soil_depth=2.0)
    pre, _real_som = precompute_fast_analytic_inputs(tab, **spin)

    fn = build_live_pool_forward(pre)
    biomass_cf, lai_cf = (np.asarray(x) for x in fn(CarbonConfig(scheme="differland")))

    ref_cfol = np.asarray(pre.live_ref_C_fol)
    ref_biomass = (ref_cfol + np.asarray(pre.live_ref_C_root)
                   + np.asarray(pre.live_ref_C_wood)) / 1000.0
    ref_lai = ref_cfol / CarbonConfig().LCMA

    def cover_weighted_rel(pred, ref):
        cover = np.asarray(soc_cover(cid, cw, tab))
        wsum = float(np.sum(cover)) or 1.0
        rmse = float(np.sqrt(np.sum(cover * (pred - ref) ** 2) / wsum))
        mean_ref = float(np.sum(cover * ref) / wsum)
        return rmse / max(abs(mean_ref), 1e-6)

    bio_rel = cover_weighted_rel(biomass_cf, ref_biomass)
    lai_rel = cover_weighted_rel(lai_cf, ref_lai)
    print(f"\n[live-pool match] biomass cf={np.round(biomass_cf, 2)} ref="
          f"{np.round(ref_biomass, 2)} kgC/m2 cover_rel={bio_rel:.4f}; "
          f"LAI cf={np.round(lai_cf, 2)} ref={np.round(ref_lai, 2)} cover_rel={lai_rel:.4f}")
    assert np.all(np.isfinite(biomass_cf)) and np.all(np.isfinite(lai_cf))
    # ~10-15% target; allow up to 20% (the within-year foliage covariance residual, the
    # SAME class as the fast SOC forward's ~6-8%), and fail loud past that.
    assert bio_rel < 0.20, (biomass_cf, ref_biomass, bio_rel)
    assert lai_rel < 0.20, (lai_cf, ref_lai, lai_rel)


def soc_cover(cid, cw, tab):
    """Per-archetype cover weight (reuse the shared SOC cover-weight helper)."""
    from legoesm.land.carbon.soc_observations import per_archetype_cover_weight
    n_arch = int(np.asarray(tab.pft_id).shape[0])
    return per_archetype_cover_weight(cid, cw, n_arch=n_arch)
