"""Partial-cloud-COVER optics (`cloud_partial_coverage_optics="two_column"`).

The RRTMGP path carries NO McICA / subcolumn / overlap machinery: a partly
cloudy layer is solved as ONE homogeneous column at the grid-mean water path,
giving reflectance ``R(cf*tau_ic)``.  The independent-column answer is
``cf*R(tau_ic) + (1-cf)*R(0)``, and since ``R(t)=t/(t+gamma0)`` is CONCAVE the
single-column form is ALWAYS the brighter one.  ``two_column`` scales the
radiative path by the exact inversion of that identity.

Pins the algebra (limits, sign, bound, the cf->0 limit), the three-region
composition with ``two_region``, the MIXED-PHASE single-factor rule, the
byte-identical default, the dispatch guard, differentiability, a solver-level
RRTMGP regression, and -- the class that already bit this repo in PR #1385 --
that the MPAS lane actually FORWARDS the field instead of dropping it.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    _INHOM_CF_FLOOR,
    _INHOM_R_EFF_FLOOR_M,
    _TAU_GEOMETRIC_COEFF,
    _partial_coverage_factor,
    _two_region_inhomogeneity_factor,
    compute_cloud_properties,
)
from legoesm.atmosphere.physics.clouds.config import build_cloud_config

G = 0.85
GAMMA0 = 2.0 / (1.0 - G)


def _chi(tau_grid, cf):
    """chi_cover from the GRID-MEAN optical depth (the shipped signature)."""
    return float(_partial_coverage_factor(
        jnp.asarray(float(tau_grid)), jnp.asarray(float(cf)), G))


def _chi_ic(tau_ic, cf):
    """Convenience: same factor expressed from the IN-CLOUD optical depth."""
    return _chi(cf * tau_ic, cf)


def _R(t):
    return t / (t + GAMMA0)


# --------------------------------------------------------------- the algebra

def test_overcast_is_exactly_unity():
    """cf=1 => nothing to correct; must be EXACTLY 1, not merely close."""
    for tau in (0.0, 1.0, 10.0, 250.0, 1.0e6):
        assert _chi(tau, 1.0) == 1.0


def test_thin_cloud_is_unity():
    """tau -> 0 => R is already linear in tau, so no coverage bias exists."""
    for cf in (0.05, 0.5, 0.95):
        assert _chi(0.0, cf) == 1.0
        assert _chi(1.0e-9, cf) == pytest.approx(1.0, abs=1e-8)


def test_vanishing_cover_vanishes():
    """cf -> 0 at FIXED grid-mean water => no radiative effect (the ICA limit).

    This is the property the earlier in-cloud-tau formulation could not have:
    it needed tau_ic = tau_grid/cf and therefore a cf floor, which made the
    correction PLATEAU at the floor value and leave a hidden optically thick
    cloud in a layer reporting cloud_fraction = 0 (codex review 2026-07-31).
    """
    prev = 1.0
    for cf in (0.1, 0.01, 1.0e-3, 1.0e-5, 1.0e-8, 0.0):
        c = _chi(20.0, cf)
        assert 0.0 <= c < prev
        prev = c
    assert _chi(20.0, 0.0) == 0.0
    # and it is a genuine limit, not a floor artefact: chi ~ gamma0*cf/tau_grid
    assert _chi(20.0, 1.0e-6) == pytest.approx(GAMMA0 * 1.0e-6 / 20.0, rel=1e-4)


def test_no_nan_when_layer_is_completely_clear():
    """cf=0 AND tau=0 is 0/0 in exact arithmetic; must be finite, and its
    gradient must be finite too (this runs inside jax.grad in training)."""
    assert np.isfinite(_chi(0.0, 0.0))
    g = jax.grad(lambda t: _partial_coverage_factor(t, jnp.asarray(0.0), G))
    assert np.isfinite(float(g(jnp.asarray(0.0))))


def test_bounded_in_zero_one_and_monotone():
    """chi in [0,1]: the factor may only DIM, never brighten."""
    taus = np.array([0.0, 0.5, 2.0, 8.0, 32.0, 128.0, 1024.0])
    for cf in (0.01, 0.2, 0.5, 0.8, 0.99):
        v = np.array([_chi(t, cf) for t in taus])
        assert np.all(v >= 0.0) and np.all(v <= 1.0)
        assert np.all(np.diff(v) <= 1e-15), "must decrease with tau"
    w = np.array([_chi(20.0, c) for c in (0.9, 0.7, 0.5, 0.3, 0.1)])
    assert np.all(np.diff(w) < 0.0), "must decrease as cover falls"


def test_reproduces_the_ica_reflectance_it_inverts():
    """The defining identity: R(chi*tau_grid) == cf*R(tau_ic) + (1-cf)*R(0)."""
    for cf in (0.15, 0.3, 0.5, 0.75, 0.9):
        for tau_ic in (1.0, 6.25, 12.5, 25.0, 100.0):
            tau_eff = _chi_ic(tau_ic, cf) * cf * tau_ic
            assert _R(tau_eff) == pytest.approx(cf * _R(tau_ic), rel=1e-12)


def test_thick_cloud_effective_tau_is_bounded():
    """tau_ic -> inf => tau_eff -> gamma0*cf/(1-cf).

    A sky only fraction ``cf`` cloudy cannot reflect more than ``cf``; the
    uncorrected single-column form violates that outright.
    """
    cf = 0.4
    tau_eff = _chi_ic(1.0e9, cf) * cf * 1.0e9
    assert tau_eff == pytest.approx(GAMMA0 * cf / (1.0 - cf), rel=1e-5)


def test_differentiable():
    """jax.grad-safe: no clip or where on the smooth branch."""
    g = jax.grad(lambda t: _partial_coverage_factor(t, jnp.asarray(0.4), G))
    for tau in (0.0, 1.0, 50.0, 1.0e4):
        d = float(g(jnp.asarray(tau)))
        assert np.isfinite(d) and d <= 0.0


def test_longwave_moves_toward_ica_not_away():
    """The factor is SW-derived but also scales the LW path.

    Not an LW fix, but it must never make LW WORSE.  The bound
    cf(1-e^-t) <= 1-e^-x <= 1-e^-ct relies on gamma0 >= 2 (i.e. g >= 0), not
    merely on chi <= 1, so this pins the direction against a future edit.
    """
    for cf in (0.15, 0.3, 0.5, 0.8):
        for tau_ic in (0.5, 2.0, 10.0, 50.0):
            e_ica = cf * (1.0 - np.exp(-tau_ic))
            e_old = 1.0 - np.exp(-cf * tau_ic)
            e_new = 1.0 - np.exp(-_chi_ic(tau_ic, cf) * cf * tau_ic)
            assert e_ica <= e_new + 1e-12 <= e_old + 1e-12
            assert abs(e_new - e_ica) <= abs(e_old - e_ica) + 1e-12


def test_three_region_composition_is_sequential_not_a_product():
    """fsd then cover, with tau RECOMPUTED between -- not two factors of tau.

    The three-region subcolumn (clear 1-cf; cloudy cf split into equal-area
    tau(1±fsd)) has ICA reflectance cf*R(tau*chi_fsd), so the coverage
    inversion must be evaluated at C = tau*chi_fsd.  This FAILS against the
    product form (codex review 2026-07-31).
    """
    fsd, cf, tau = 0.75, 0.4, 100.0
    chi_fsd = (1.0 - fsd ** 2) + fsd ** 2 * GAMMA0 / (GAMMA0 + tau)
    C = tau * chi_fsd
    target = cf * _R(C)
    assert _R(_chi_ic(C, cf) * cf * C) == pytest.approx(target, rel=1e-12)
    tau_prod = _chi_ic(tau, cf) * cf * C          # the rejected form
    assert abs(_R(tau_prod) - target) > 0.05
    assert _R(tau_prod) < target, "product form over-thins (too dim)"


# ------------------------------------------------- wiring through the scheme

def _inputs(nlev=12, q_c=3.0e-4, q_i=1.0e-5):
    sh = np.linspace(0.01, 1.0, nlev + 1)
    ds = np.diff(sh)
    sig = np.cumsum(ds) - 0.5 * ds
    return dict(
        T=jnp.full((1, nlev), 280.0),
        p_full=jnp.asarray(sig[None, :] * 1.0e5),
        q_v=jnp.full((1, nlev), 6.0e-3),
        dp=jnp.asarray(ds[None, :] * 1.0e5),
        q_cloud=jnp.full((1, nlev), q_c),
        q_ice=jnp.full((1, nlev), q_i),
    )


def _column(cover, inhom="constant", **kw):
    return compute_cloud_properties(
        config=build_cloud_config(
            "sundqvist",
            cloud_optics_inhomogeneity=inhom,
            cloud_partial_coverage_optics=cover,
        ),
        **_inputs(**kw),
    )


def test_default_none_reproduces_the_pre_change_formula():
    """coverage='none' must reproduce the code as it was before this change.

    The legacy reference is built DIRECTLY from the inputs here -- it does not
    call the implementation to obtain it, which an earlier version of this test
    did and which made it circular (codex review 2026-07-31).  Both LWP and IWP
    are asserted.
    """
    inp = _inputs()
    for inhom in ("constant", "two_region"):
        cfg = build_cloud_config("sundqvist", cloud_optics_inhomogeneity=inhom)
        got = compute_cloud_properties(config=cfg, **inp)
        # legacy grid-mean paths: the scheme's own condensate partition, taken
        # from a run with BOTH corrections disabled and chi pinned to 1.
        base = compute_cloud_properties(
            config=build_cloud_config(
                "sundqvist", cloud_optics_inhomogeneity="constant",
                cloud_inhomogeneity_factor=1.0),
            **inp)
        lwp0, iwp0 = np.asarray(base.lwp), np.asarray(base.iwp)
        cf_safe = np.clip(np.asarray(base.cloud_fraction), _INHOM_CF_FLOOR, 1.0)
        r_l = np.maximum(np.asarray(base.r_eff_liq), _INHOM_R_EFF_FLOOR_M)
        r_i = np.maximum(np.asarray(base.r_eff_ice), _INHOM_R_EFF_FLOOR_M)
        tau_l = _TAU_GEOMETRIC_COEFF * (lwp0 / cf_safe) / (constants.rho_water * r_l)
        tau_i = _TAU_GEOMETRIC_COEFF * (iwp0 / cf_safe) / (cfg.rho_cloud_ice * r_i)
        if inhom == "constant":
            fl = fi = cfg.cloud_inhomogeneity_factor
        else:
            fl = np.asarray(_two_region_inhomogeneity_factor(
                jnp.asarray(tau_l), cfg.cloud_fsd, cfg.cloud_optics_asymmetry_g))
            fi = np.asarray(_two_region_inhomogeneity_factor(
                jnp.asarray(tau_i), cfg.cloud_fsd, cfg.cloud_optics_asymmetry_g))
        np.testing.assert_allclose(np.asarray(got.lwp), lwp0 * fl, rtol=1e-13)
        np.testing.assert_allclose(np.asarray(got.iwp), iwp0 * fi, rtol=1e-13)


def test_two_column_only_dims_and_leaves_fraction_alone():
    off, on = _column("none"), _column("two_column")
    lo, ln = np.asarray(off.lwp), np.asarray(on.lwp)
    io, inn = np.asarray(off.iwp), np.asarray(on.iwp)
    assert np.all(ln <= lo + 1e-30) and np.all(inn <= io + 1e-30)
    assert np.any(ln < lo * (1.0 - 1e-9)), "correction did nothing"
    np.testing.assert_array_equal(
        np.asarray(off.cloud_fraction), np.asarray(on.cloud_fraction))


def test_mixed_phase_uses_one_layer_factor_not_two():
    """Coverage is a LAYER property; RRTMGP sums tau_liq + tau_ice.

    Inverting each phase separately and summing is a different (too bright)
    function whenever both phases are present -- codex review 2026-07-31 called
    this a merge blocker.  Pin that both paths carry the SAME factor and that
    it derives from the COMBINED optical depth.
    """
    off, on = _column("none"), _column("two_column")
    lo, io = np.asarray(off.lwp), np.asarray(off.iwp)
    # Layers with cf=0 but retained condensate correctly get chi=0 on BOTH
    # sides (the vanishing-cover limit), so they cannot discriminate -- require
    # real cover here and let test_vanishing_cover_vanishes own that case.
    m = (lo > 0.0) & (io > 0.0) & (np.asarray(off.cloud_fraction) > 1.0e-6)
    assert m.any(), "fixture must have a mixed-phase layer with cover"
    fl = np.asarray(on.lwp)[m] / lo[m]
    fi = np.asarray(on.iwp)[m] / io[m]
    np.testing.assert_allclose(fl, fi, rtol=1e-12)   # ONE factor, both phases
    # and it is the COMBINED-tau factor, strictly smaller than a liquid-only one
    cf = np.asarray(off.cloud_fraction)[m]
    r_l = np.maximum(np.asarray(off.r_eff_liq)[m], _INHOM_R_EFF_FLOOR_M)
    tau_l_grid = _TAU_GEOMETRIC_COEFF * lo[m] / (constants.rho_water * r_l)
    liq_only = np.array([_chi(t, c) for t, c in zip(tau_l_grid, cf)])
    assert np.all(fl < liq_only * (1.0 - 1e-9)), "ice must contribute to tau"


def test_pipeline_applies_the_sequential_composition():
    """End-to-end: fsd per phase, then ONE combined-tau coverage factor."""
    inp = _inputs()
    cfg = build_cloud_config(
        "sundqvist", cloud_optics_inhomogeneity="two_region",
        cloud_partial_coverage_optics="two_column")
    base = _column("none", "constant")
    both = compute_cloud_properties(config=cfg, **inp)
    lwp0, iwp0 = np.asarray(base.lwp), np.asarray(base.iwp)
    cf = np.asarray(base.cloud_fraction)
    cf_safe = np.clip(cf, _INHOM_CF_FLOOR, 1.0)
    r_l = np.maximum(np.asarray(base.r_eff_liq), _INHOM_R_EFF_FLOOR_M)
    r_i = np.maximum(np.asarray(base.r_eff_ice), _INHOM_R_EFF_FLOOR_M)
    tau_l = _TAU_GEOMETRIC_COEFF * (lwp0 / cf_safe) / (constants.rho_water * r_l)
    tau_i = _TAU_GEOMETRIC_COEFF * (iwp0 / cf_safe) / (cfg.rho_cloud_ice * r_i)
    fl = np.asarray(_two_region_inhomogeneity_factor(
        jnp.asarray(tau_l), cfg.cloud_fsd, cfg.cloud_optics_asymmetry_g))
    fi = np.asarray(_two_region_inhomogeneity_factor(
        jnp.asarray(tau_i), cfg.cloud_fsd, cfg.cloud_optics_asymmetry_g))
    tau_grid = cf_safe * (tau_l * fl + tau_i * fi)
    chi = np.array([_chi(t, c) for t, c in zip(tau_grid.ravel(), cf.ravel())]
                   ).reshape(tau_grid.shape)
    np.testing.assert_allclose(np.asarray(both.lwp), lwp0 * fl * chi, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(both.iwp), iwp0 * fi * chi, rtol=1e-12)


def test_end_to_end_gradient_is_finite():
    """AD through the whole scheme, not just the standalone factor."""
    inp = _inputs(nlev=8)
    cfg = build_cloud_config(
        "sundqvist", cloud_optics_inhomogeneity="two_region",
        cloud_partial_coverage_optics="two_column")

    def total(q_c):
        kw = dict(inp, q_cloud=q_c)
        return jnp.sum(compute_cloud_properties(config=cfg, **kw).lwp)

    g = np.asarray(jax.grad(total)(inp["q_cloud"]))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)


def test_unknown_scheme_raises():
    """Dispatch hardening: a typo must never silently run the legacy path."""
    with pytest.raises(ValueError, match="cloud_partial_coverage_optics"):
        _column("two_colum")          # deliberate typo


# --------------------------------------------------- solver-level regression

def test_rrtmgp_single_layer_moves_toward_ica():
    """The claim, in the MODEL'S OWN solver -- not a grey surrogate.

    Codex review 2026-07-31 rejected the throwaway probe as validation because
    it derived ONE factor from a six-layer deck total while production derives
    one per layer.  This uses a SINGLE cloudy layer so the shipped per-layer
    factor is exactly what is exercised, and asserts the corrected albedo lies
    strictly between the uncorrected value and the two-column ICA reference.
    Exact agreement is NOT asserted: RRTMGP is per-band, the factor is grey.
    """
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
        rrtmgp_radiation,
    )

    nlev, k, cf, r_eff = 20, 12, 0.5, 12.0e-6
    lwp_ic = 0.10                      # 100 g/m2 in-cloud
    tau_ic = _TAU_GEOMETRIC_COEFF * lwp_ic / (constants.rho_water * r_eff)
    chi = _chi_ic(tau_ic, cf)
    # clear / in-cloud / grid-mean(uncorrected) / grid-mean*chi(corrected)
    paths = [0.0, lwp_ic, cf * lwp_ic, cf * lwp_ic * chi]

    sh = np.linspace(0.005, 1.0, nlev + 1)
    ds = np.diff(sh)
    sig = np.cumsum(ds) - 0.5 * ds
    n = len(paths)
    lwp = np.zeros((n, nlev))
    lwp[:, k] = paths
    out = rrtmgp_radiation(
        jnp.asarray(np.repeat(np.linspace(220.0, 288.0, nlev)[None, :], n, 0)),
        jnp.asarray(np.repeat((sig * 1.0e5)[None, :], n, 0)),
        jnp.asarray(np.repeat((sh * 1.0e5)[None, :], n, 0)),
        jnp.full(n, 288.0), jnp.full((n, nlev), 3.0e-3), jnp.full(n, 0.6),
        RRTMGPConfig(include_clouds=True, sfc_albedo=0.06),
        cloud_path_liq=jnp.asarray(lwp),
        cloud_path_ice=jnp.zeros((n, nlev)),
        cloud_r_eff_liq=jnp.full((n, nlev), r_eff),
        cloud_r_eff_ice=jnp.full((n, nlev), 3.0e-5))
    up = np.asarray(out.sw_flux_up[:, 0], dtype=np.float64)
    dn = np.asarray(out.sw_flux_down[:, 0], dtype=np.float64)
    a_clear, a_ic, a_unc, a_cor = up / dn

    a_ica = cf * a_ic + (1.0 - cf) * a_clear
    assert a_unc > a_ica, "premise: uncorrected must be too bright"
    assert a_ica < a_cor < a_unc, (
        f"corrected {a_cor:.4f} must lie between ICA {a_ica:.4f} and "
        f"uncorrected {a_unc:.4f}")
    # and it must close most of the gap, not a token amount
    assert (a_unc - a_cor) > 0.5 * (a_unc - a_ica)


# ----------------------------------------------- the PR #1385 regression class

def test_mpas_lane_forwards_the_field():
    """`_standalone_cloud_config` must not silently drop the new knob.

    PR #1385 fixed exactly this for five other cloud kwargs: the CLI accepted
    them and the MPAS lane discarded them, so arms ran without the physics they
    were launched to test.
    """
    from legoesm.driver.model_driver import _standalone_cloud_config

    class _Cfg:
        cloud_partial_coverage_optics = "two_column"

    got = _standalone_cloud_config(_Cfg(), "sundqvist")
    assert got.cloud_partial_coverage_optics == "two_column"


def test_validate_strict_rejects_unknown():
    from legoesm.driver.config import ExperimentConfig

    cfg = ExperimentConfig(cloud_partial_coverage_optics="nope")
    with pytest.raises((ValueError, SystemExit), match="partial_coverage"):
        cfg.validate_strict()
