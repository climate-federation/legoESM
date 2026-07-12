"""Unit tests for the fast closed-form SOM equilibrium
(:mod:`legoesm.land.carbon.fast_analytic`) and its precompute
(:func:`legoesm.land.carbon.global_init.precompute_fast_analytic_inputs`).

Two layers:

* PURE closed-form checks (synthetic ``FastAnalyticInputs``; no spin-up) -- the
  forward-substitution cascade matches a hand computation, the per-pool turnover
  reuses the model's exact kinetics, ``jax.grad`` w.r.t. every SOM leaf is finite
  and non-zero, and raising ``som_freeze_floor`` lowers a cold archetype's SOC.
* The CRUX match: on a tiny synthetic world, the closed form at the production
  defaults reproduces ``som_total(equilibrate_archetypes)`` per archetype.

Compute-node scale (JIT-compiles the coupled land+carbon step for the match
test) -- run via the sbatch/srun wrapper, NOT the login node.
``JAX_ENABLE_X64=1``; forced CPU here.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import numpy.testing as npt
import pytest

# SOM fields in a fixed order for building traced configs.
SOM_FIELDS = (
    "tor_som_active", "tor_som_slow", "tor_som_passive",
    "f_active_to_slow", "f_slow_to_passive", "som_freeze_floor",
    "Q10_het_exp", "cwd_humification_eff",
)


def _synthetic_inputs(*, n_arch, temp, precip, lit, a_wood, n_samples, dt_days,
                      frozen_fraction=0.0):
    """Build a FastAnalyticInputs with a CONSTANT (isothermal) T trajectory so the
    cascade can be hand-verified.  ``frozen_fraction`` (default 0 -> f_perma~1, no
    perennial-frost protection) sets the annual frozen fraction driving f_perma."""
    import jax.numpy as jnp
    from legoesm.land.carbon.fast_analytic import FastAnalyticInputs

    T = jnp.full((n_arch, n_samples), float(temp))
    # The live-pool fields are irrelevant to analytic_som_soc; fill placeholders so the
    # (now 11-field) NamedTuple constructs.
    z = jnp.zeros((n_arch,))
    return FastAnalyticInputs(
        lit_to_som_annual=jnp.full((n_arch,), float(lit)),
        a_wood_annual=jnp.full((n_arch,), float(a_wood)),
        soil_T_traj=T,
        soil_frozen_fraction=jnp.full((n_arch,), float(frozen_fraction)),
        precip=jnp.full((n_arch,), float(precip)),
        dt_days=float(dt_days),
        npp_pos_annual=z, is_woody=jnp.ones((n_arch,)), is_evergreen=z,
        live_ref_C_fol=z, live_ref_C_root=z, live_ref_C_wood=z,
    )


def test_forward_substitute_cascade_matches_hand():
    """C_active=I/k_a ; C_slow=f_a2s*I/k_s ; C_passive=f_s2p*f_a2s*I/k_p."""
    import jax.numpy as jnp
    from legoesm.land.carbon.fast_analytic import _forward_substitute_cascade

    i_active = jnp.asarray([200.0, 400.0])
    k_a = jnp.asarray([0.3, 0.5]); k_s = jnp.asarray([0.04, 0.05])
    k_p = jnp.asarray([1.6e-3, 2.0e-3])
    f_a2s, f_s2p = 0.30, 0.30
    ca, cs, cp = _forward_substitute_cascade(i_active, k_a, k_s, k_p, f_a2s, f_s2p)
    npt.assert_allclose(np.asarray(ca), np.asarray(i_active) / np.asarray(k_a), rtol=1e-12)
    npt.assert_allclose(np.asarray(cs),
                        f_a2s * np.asarray(i_active) / np.asarray(k_s), rtol=1e-12)
    npt.assert_allclose(np.asarray(cp),
                        f_s2p * f_a2s * np.asarray(i_active) / np.asarray(k_p), rtol=1e-12)


def test_som_decomposition_rate_reuses_model_kinetics():
    """The public wrapper == _effective_rate(_som_decomp_modifier * tor, dt)."""
    import jax.numpy as jnp
    from legoesm.land.carbon.carbon_cycle import (
        _effective_rate, _som_decomp_modifier, som_decomposition_rate,
    )
    from legoesm.land.carbon.config import CarbonConfig

    cfg = CarbonConfig(scheme="differland")
    T = jnp.asarray([[280.0, 290.0], [270.0, 300.0]])
    precip = jnp.asarray([[3e-5], [3e-5]])
    dt_days = 1.0 / 24.0
    got = som_decomposition_rate(T, precip, cfg.tor_som_active, cfg, dt_days)
    expect = _effective_rate(
        _som_decomp_modifier(T, precip, cfg) * cfg.tor_som_active, dt_days)
    npt.assert_allclose(np.asarray(got), np.asarray(expect), rtol=1e-12)


def test_analytic_som_soc_isothermal_matches_hand():
    """At a CONSTANT T with dt_days=1 (so _effective_rate(r,1)==r), the closed-form
    SOC equals the hand cascade using the model's own modifier m."""
    import jax.numpy as jnp
    from legoesm.land.carbon.carbon_cycle import (
        _som_decomp_modifier, perennial_frost_protection,
    )
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.fast_analytic import analytic_som_soc

    cfg = CarbonConfig(scheme="differland")
    temp, precip = 293.15, float(cfg.precip_ref)       # warm, reference moisture
    n_samples, dt_days = 365, 1.0                      # eff_rate(r, 1) == r; 365-day year
    lit, a_wood = 250.0, 300.0
    frozen_fraction = 0.4                               # some frost -> f_perma < 1 exercised
    pre = _synthetic_inputs(n_arch=1, temp=temp, precip=precip, lit=lit,
                            a_wood=a_wood, n_samples=n_samples, dt_days=dt_days,
                            frozen_fraction=frozen_fraction)

    # The surrogate scales every SOM turnover by the SAME per-column f_perma
    # (perennial-frost protection), so the hand cascade must too.
    fperma = float(perennial_frost_protection(jnp.asarray(frozen_fraction), cfg))
    m = float(_som_decomp_modifier(
        jnp.asarray([[temp]]), jnp.asarray([[precip]]), cfg)[0, 0]) * fperma
    days = n_samples * dt_days                          # 365
    k_a = days * m * cfg.tor_som_active
    k_s = days * m * cfg.tor_som_slow
    k_p = days * m * cfg.tor_som_passive
    i_active = lit + cfg.cwd_humification_eff * a_wood
    soc_hand = (
        i_active / k_a
        + cfg.f_active_to_slow * i_active / k_s
        + cfg.f_slow_to_passive * cfg.f_active_to_slow * i_active / k_p
    )
    soc = float(np.asarray(analytic_som_soc(pre, cfg))[0])
    npt.assert_allclose(soc, soc_hand, rtol=1e-9)
    assert soc > 0.0


def test_analytic_som_soc_grad_finite_nonzero():
    """jax.grad of sum(SOC) w.r.t. every SOM config leaf is finite and non-zero."""
    import jax
    import jax.numpy as jnp
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.fast_analytic import FastAnalyticInputs, analytic_som_soc

    cfg0 = CarbonConfig(scheme="differland")
    # Cold-ish seasonal trajectory so som_freeze_floor + Q10 both have signal
    # (the modifier's freeze sigmoid is unsaturated only where T crosses freezing).
    n_arch, n_samples = 2, 240
    phase = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
    temp = 276.0 + 14.0 * np.cos(phase)                # crosses freezing
    _z = jnp.zeros((n_arch,))
    pre = FastAnalyticInputs(
        lit_to_som_annual=jnp.full((n_arch,), 250.0),
        a_wood_annual=jnp.full((n_arch,), 300.0),
        soil_T_traj=jnp.asarray(np.tile(temp, (n_arch, 1))),
        soil_frozen_fraction=jnp.full((n_arch,), 0.5),   # perennially-frozen-ish
        precip=jnp.full((n_arch,), 3e-5),
        dt_days=1.0 / 24.0,
        npp_pos_annual=_z, is_woody=jnp.ones((n_arch,)), is_evergreen=_z,
        live_ref_C_fol=_z, live_ref_C_root=_z, live_ref_C_wood=_z)

    defaults = jnp.asarray([float(getattr(cfg0, f)) for f in SOM_FIELDS])

    def loss(vec):
        cfg = cfg0._replace(**{f: vec[i] for i, f in enumerate(SOM_FIELDS)})
        return jnp.sum(analytic_som_soc(pre, cfg))

    val, grad = jax.value_and_grad(loss)(defaults)
    assert np.isfinite(float(val))
    g = np.asarray(grad)
    assert np.all(np.isfinite(g)), g
    for i, f in enumerate(SOM_FIELDS):
        assert abs(g[i]) > 0.0, f"zero gradient for {f}"


def test_som_freeze_floor_lowers_cold_soc():
    """A higher freeze floor speeds cold-soil decomposition -> LESS SOC (the
    calibration lever): monotone decreasing in som_freeze_floor for a cold
    trajectory."""
    import jax.numpy as jnp
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.fast_analytic import analytic_som_soc

    n_samples = 240
    phase = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
    temp = 272.0 + 12.0 * np.cos(phase)                # cold, frozen much of year
    from legoesm.land.carbon.fast_analytic import FastAnalyticInputs
    _z = jnp.zeros((1,))
    pre = FastAnalyticInputs(
        lit_to_som_annual=jnp.asarray([250.0]),
        a_wood_annual=jnp.asarray([300.0]),
        soil_T_traj=jnp.asarray(temp[None, :]),
        soil_frozen_fraction=jnp.asarray([0.6]),         # cold, frozen much of year
        precip=jnp.asarray([3e-5]),
        dt_days=1.0 / 24.0,
        npp_pos_annual=_z, is_woody=jnp.ones((1,)), is_evergreen=_z,
        live_ref_C_fol=_z, live_ref_C_root=_z, live_ref_C_wood=_z)
    base = CarbonConfig(scheme="differland")
    soc_lo = float(np.asarray(analytic_som_soc(pre, base._replace(som_freeze_floor=0.02)))[0])
    soc_hi = float(np.asarray(analytic_som_soc(pre, base._replace(som_freeze_floor=0.25)))[0])
    assert soc_hi < soc_lo, (soc_hi, soc_lo)


def test_perennial_frost_protection_raises_cold_soc_warm_unchanged():
    """IDEALIZED equilibrium: a perennially-frozen cold-wet column accumulates
    MORE SOC WITH the permafrost/anaerobic protection than without, while a warm
    column (frozen_fraction=0) is UNCHANGED -- so the mechanism raises
    high-latitude SOC without perturbing the already-correct temperate/tropical
    stocks (corr structure preserved)."""
    import numpy as np
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.fast_analytic import analytic_som_soc

    base = CarbonConfig(scheme="differland")
    # Protection DISABLED: permafrost_protection_min=1 -> f_perma == 1 for any
    # frozen_fraction (the clean before/after control).
    off = base._replace(permafrost_protection_min=1.0)

    # Cold, perennially-frozen column (high annual frozen fraction).
    cold = _synthetic_inputs(n_arch=1, temp=266.0, precip=2.0e-5, lit=200.0,
                             a_wood=250.0, n_samples=240, dt_days=1.0 / 24.0,
                             frozen_fraction=0.8)
    soc_cold_on = float(np.asarray(analytic_som_soc(cold, base))[0])
    soc_cold_off = float(np.asarray(analytic_som_soc(cold, off))[0])
    assert soc_cold_on > soc_cold_off, (soc_cold_on, soc_cold_off)
    # Substantial accumulation for true permafrost (f_perma ~0.34 at phi=0.8 ->
    # ~3x turnover reduction), not a marginal nudge.
    assert soc_cold_on > 1.5 * soc_cold_off, (soc_cold_on, soc_cold_off)

    # Warm column (never frozen): the protection is inert.
    warm = _synthetic_inputs(n_arch=1, temp=298.0, precip=4.0e-5, lit=200.0,
                             a_wood=250.0, n_samples=240, dt_days=1.0 / 24.0,
                             frozen_fraction=0.0)
    soc_warm_on = float(np.asarray(analytic_som_soc(warm, base))[0])
    soc_warm_off = float(np.asarray(analytic_som_soc(warm, off))[0])
    npt.assert_allclose(soc_warm_on, soc_warm_off, rtol=2e-3)


# ---------------------------------------------------------------------------
# CRUX: the closed form reproduces the spin-up equilibrium at the defaults.
# ---------------------------------------------------------------------------
def test_analytic_matches_equilibrate_at_defaults():
    """precompute_fast_analytic_inputs + analytic_som_soc(defaults) reproduces
    som_total(equilibrate_archetypes) per archetype (few %) on a tiny world."""
    import jax.numpy as jnp
    from legoesm.land.carbon.climate_features import reduce_climatology_to_features
    from legoesm.land.carbon.config import CarbonConfig, som_total
    from legoesm.land.carbon.fast_analytic import analytic_som_soc
    from legoesm.land.carbon.global_init import (
        build_archetypes, equilibrate_archetypes, precompute_fast_analytic_inputs,
    )

    # 3 cells, tropical (pft 4) + temperate (pft 7); temperate cell has a real
    # seasonal cycle so the freeze/Q10 seasonal integral is exercised.
    ncell, npft = 3, 17
    w = np.zeros((ncell, npft)); w[0, 4] = 1.0; w[1, 7] = 1.0; w[2, 7] = 1.0
    months = np.arange(12)
    season = np.sin(2 * np.pi * months / 12.0)
    t = np.stack([298.0 + 2.0 * season, 281.0 + 14.0 * season, 275.0 + 16.0 * season])
    pr = np.stack([np.full(12, 6e-5), np.full(12, 2.5e-5), np.full(12, 2.0e-5)])
    sw = np.full((ncell, 12), 210.0); nr = np.full((ncell, 12), 90.0)
    feats = reduce_climatology_to_features(t, pr, sw, nr)
    soil = np.array(["loam", "loam", "loam"])
    tab, _cid, _cw = build_archetypes(
        w, feats, soil, np.ones(ncell, bool), k_per_pft=1, seed=0)

    spin = dict(n_spinup=40, n_verify=6, dt=7200.0, n_layers=6, soil_depth=2.0)
    eq, _qc = equilibrate_archetypes(tab, **spin)
    real = np.asarray(som_total(eq))                         # (n_arch,) gC/m2

    pre, real_from_pre = precompute_fast_analytic_inputs(tab, **spin)
    analytic = np.asarray(analytic_som_soc(pre, CarbonConfig(scheme="differland")))

    # The precompute's own equilibrium must equal equilibrate_archetypes' (same
    # spin-up), and the closed form must reproduce it to a few %.
    npt.assert_allclose(np.asarray(real_from_pre), real, rtol=1e-6, atol=1e-6)
    rel = np.abs(analytic - real) / np.maximum(np.abs(real), 1.0)
    print(f"\n[match] analytic={np.round(analytic, 2)} gC/m2  real={np.round(real, 2)}"
          f"  rel={np.round(rel, 4)}  mean={np.mean(rel):.4f} max={np.max(rel):.4f}")
    assert np.all(np.isfinite(analytic))
    # Recording the SOM active-pool inputs (lit_to_som / a_wood) from the reset's
    # own last-transient-year fluxes makes the closed form reproduce
    # analytic_slow_pool_equilibrium's INPUT chain exactly (residual only the small
    # k_X soil-T term) -- was ~7.5% when a_wood was sampled from a shifted post-verify
    # phase of the ~27-yr wood pool.  Tightened to LOCK the gain: measured mean 0.5% /
    # max 1.0% on this tiny world (small headroom for cross-hardware float variance in
    # the coupled spin-up).
    assert float(np.mean(rel)) < 0.03, (analytic, real, rel)
    assert float(np.max(rel)) < 0.05, (analytic, real, rel)
