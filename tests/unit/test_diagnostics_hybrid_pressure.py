"""CMOR plev output must use the ACTUAL vertical coordinate's pressures.

``HybridSigmaPressureCoordinate.sigma_full`` is a COMPATIBILITY VIEW returning
``A_full + B_full``; its own docstring warns it is for utilities that do not
assume pure-sigma pressure dependence.  ``DiagnosticCollector`` assumed exactly
that (``p = sigma_full * p_s``), which is wrong by ``A*(p_s - p_ref)`` -- a few
hPa near sea level but tens of hPa over high terrain, landing CMOR
``ta``/``ua``/``hus`` on the wrong pressure surfaces.

Hybrid is the DRIVER DEFAULT (``GridConfig.vertical_coord = "hybrid"``,
``run_amip.py --vertical-coord`` default), so this affected every run that did
not explicitly ask for sigma.
"""
import numpy as np
import pytest

from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels

NLEV = 20


def _collector(coord, with_vcoord=True):
    return DiagnosticCollector(
        nlev=NLEV,
        sigma_full=coord.sigma_full,
        dsigma=coord.dsigma,
        vcoord=coord if with_vcoord else None,
    )


def test_hybrid_full_level_pressure_matches_the_coordinate():
    """p = A p_ref + B p_s, not (A+B) p_s."""
    coord = make_hybrid_levels(NLEV)
    dc = _collector(coord)
    for p_s in (1.0e5, 1.013e5, 6.0e4):          # sea level and high terrain
        got = np.asarray(dc._p_full(np.array([p_s])))
        want = np.asarray(coord.pressure_at_full(np.array([p_s])))
        np.testing.assert_allclose(got, want, rtol=1e-12)


def test_the_old_pure_sigma_formula_was_materially_wrong_over_terrain():
    """Quantify the defect this test exists to prevent.

    Not a tautology: it pins that the error is LARGE where p_s departs from
    p_ref, so a future 'simplification' back to sigma_full*p_s fails loudly
    rather than drifting silently.
    """
    coord = make_hybrid_levels(NLEV)
    p_s = np.array([6.0e4])                       # ~Tibetan plateau
    right = np.asarray(coord.pressure_at_full(p_s))
    wrong = p_s[..., None] * np.asarray(coord.sigma_full)
    err = np.abs(right - wrong).max()
    assert err > 1.0e4, (
        f"expected a >100 hPa error over terrain, got {err/100:.1f} hPa")
    # and the error is exactly A*(p_s - p_ref); rel=1e-6 because the
    # coordinate coefficient arrays are fp32.
    expect = np.abs(np.asarray(coord.A_full) * (p_s - coord.p_ref)).max()
    assert err == pytest.approx(expect, rel=1e-6)


def test_sea_level_error_is_small_which_is_why_it_hid():
    coord = make_hybrid_levels(NLEV)
    p_s = np.array([1.013e5])
    right = np.asarray(coord.pressure_at_full(p_s))
    wrong = p_s[..., None] * np.asarray(coord.sigma_full)
    # ~5 hPa: small enough to look like rounding in a sea-level column,
    # which is exactly why this survived until a terrain column was checked.
    assert np.abs(right - wrong).max() < 6.0e2


def test_pure_sigma_is_unchanged_by_the_fix():
    """The sigma lane (what the MPAS AMIP arms run) must be byte-identical."""
    coord = create_sigma_coordinate(NLEV)
    p_s = np.array([1.0e5, 9.0e4])
    with_v = np.asarray(_collector(coord)._p_full(p_s))
    legacy = p_s[..., None] * np.asarray(coord.sigma_full)
    np.testing.assert_allclose(with_v, legacy, rtol=1e-13)


def test_layer_thickness_also_uses_the_coordinate():
    """dp = dA p_ref + dB p_s; dsigma*p_s is wrong by dA*(p_s - p_ref)."""
    coord = make_hybrid_levels(NLEV)
    dc = _collector(coord)
    p_s = np.array([6.0e4])
    np.testing.assert_allclose(
        np.asarray(dc._dp(p_s)),
        np.asarray(coord.layer_thickness_dp(p_s)), rtol=1e-12)
    # dp must still sum to the column mass the coordinate defines
    got = np.asarray(dc._dp(p_s)).sum()
    want = np.asarray(coord.layer_thickness_dp(p_s)).sum()
    assert got == pytest.approx(want, rel=1e-12)


def test_plev_interpolation_lands_on_the_right_level():
    """End-to-end: interpolating log(p) must return log of the target levels.

    The routine interpolates LINEARLY IN log-p, so feeding log(p_model) as the
    field makes the exact expected answer log(p_target) -- an identity that
    isolates the pressure MAPPING.  (Feeding p itself is not an identity: a
    field linear in p is not linear in log p, and the residual near the coarse
    model top would swamp the mapping error under test.)

    Output is ordered by ASCENDING pressure (``np.sort(CMIP6_PLEV19)``), not
    the descending order of the CMIP6_PLEV19 constant.
    """
    coord = make_hybrid_levels(NLEV)
    dc = _collector(coord)
    p_s = np.array([6.0e4])                       # high terrain: worst case
    p_model = np.asarray(coord.pressure_at_full(p_s))
    out = dc._interp_to_plev19(np.log(p_model), p_s)
    assert out is not None
    from legoesm.driver.diagnostics import CMIP6_PLEV19

    target = np.sort(np.asarray(CMIP6_PLEV19, dtype=float))
    got = np.asarray(out)[0]
    inside = (target >= p_model.min()) & (target <= p_model.max())
    assert inside.sum() >= 5, "fixture must bracket several CMIP levels"
    np.testing.assert_allclose(got[inside], np.log(target[inside]), rtol=1e-6)

    # and the OLD pure-sigma bracketing would have failed this: it maps the
    # target to plev/p_s and searches the shared 1-D A+B column.
    sig = np.asarray(coord.sigma_full)
    idx_old = np.clip(np.searchsorted(sig, target / p_s[0]), 1, len(sig) - 1)
    idx_new = np.clip(np.sum(p_model[..., None] < target, axis=-2)[0],
                      1, p_model.shape[-1] - 1)
    assert np.any(idx_old != idx_new), (
        "fixture must expose the bracketing difference this fix is about")


def test_collector_without_vcoord_still_works():
    """Back-compat: tests and callers that pass no vcoord keep pure-sigma."""
    coord = create_sigma_coordinate(NLEV)
    dc = _collector(coord, with_vcoord=False)
    p_s = np.array([1.0e5])
    np.testing.assert_allclose(
        np.asarray(dc._p_full(p_s)),
        p_s[..., None] * np.asarray(coord.sigma_full), rtol=1e-13)


def test_production_driver_passes_the_vcoord():
    """model_driver must wire vcoord, else the fix is dead code on every lane.

    Same failure class as PR #1385 (CLI accepted, lane silently dropped it).
    """
    import inspect

    from legoesm.driver import model_driver

    src = inspect.getsource(model_driver)
    i = src.index("self.diagnostics = DiagnosticCollector(")
    assert "vcoord=self.sigma," in src[i:i + 2000], (
        "DiagnosticCollector construction must pass vcoord=self.sigma")


def test_column_integrals_accept_a_hybrid_dp():
    """prw/clwvi/clivi must integrate the REAL layer mass, not p_s*dsigma.

    Codex review 2026-07-31 found the pressure fix was incomplete: six
    `column_water_vapor` sites still used the pure-sigma layer mass.

    The error is a vertical REDISTRIBUTION, not a mass error -- ``sum(dp)`` is
    ``p_s - p_top`` either way, so it cancels exactly for a uniform tracer and
    is ZERO at ``p_s = p_ref``.  That is why it hid.  It only bites for a
    BOTTOM-HEAVY tracer over terrain, which is exactly what q_v is: with a 2 km
    scale height this measures +26% prw at 800 hPa and +44% at 700 hPa.
    """
    import jax.numpy as jnp

    from legoesm import constants
    from legoesm.diagnostics.column_integrals import column_water_vapor

    coord = make_hybrid_levels(30)
    p_s = jnp.array([7.0e4])                  # elevated terrain, valid range
    dp = coord.layer_thickness_dp(p_s)
    p_full = np.asarray(coord.pressure_at_full(p_s))[0]
    # realistic bottom-heavy moisture; a UNIFORM profile cannot see this bug
    q = jnp.asarray((0.02 * np.exp(-(7.0e4 - p_full) / 2.0e4))[None, :])

    good = float(np.asarray(column_water_vapor(q, p_s, coord.dsigma, dp=dp))[0])
    legacy = float(np.asarray(column_water_vapor(q, p_s, coord.dsigma))[0])
    exact = float(jnp.sum(q[0] * dp[0])) / constants.g
    assert good == pytest.approx(exact, rel=1e-10)
    assert legacy > good * 1.3, (
        f"pure-sigma prw should be >30% high at 700 hPa; got "
        f"{100 * (legacy - good) / good:.1f}%")

    # and it must be EXACTLY right at sea level, where the bug is invisible
    p0 = jnp.array([1.0e5])
    dp0 = coord.layer_thickness_dp(p0)
    q0 = jnp.full((1, 30), 5.0e-3)
    assert float(np.asarray(column_water_vapor(q0, p0, coord.dsigma, dp=dp0))[0]) == (
        pytest.approx(float(np.asarray(column_water_vapor(q0, p0, coord.dsigma))[0]),
                      rel=2e-3))


def test_collector_cwv_uses_the_hybrid_layer_mass():
    """The collector's own helper must feed the corrected layer mass."""
    import jax.numpy as jnp

    coord = make_hybrid_levels(NLEV)
    dc = _collector(coord)
    p_s = jnp.array([6.0e4])
    np.testing.assert_allclose(
        np.asarray(dc._dp(p_s)),
        np.asarray(coord.layer_thickness_dp(p_s)), rtol=1e-12)


def test_hybrid_layer_pressures_are_monotonic_only_above_a_p_s_threshold():
    """Count-based bracketing needs ascending per-column pressure -- and the
    DEFAULT hybrid coordinate does not always provide it.

    ``make_hybrid_levels`` builds ``B = eta**3``,
    ``A = eta - B + (p_top/p_ref)(1-eta)``, so
    ``dp = deta*(p_ref - p_top) - dB*(p_ref - p_s)``.  Near the surface
    ``dB ~ 3 deta``, giving ``dp > 0`` only for

        p_s > (2 p_ref + p_top) / 3  ~=  667 hPa.

    Below that the lowest layers get NEGATIVE mass -- i.e. Tibet and the
    Antarctic plateau (550-650 hPa).  That is a defect in the COORDINATE, not
    in the diagnostics, and is tracked separately; this test pins the real
    validity range so the bracketing assumption above is explicit and a change
    to the coordinate cannot silently widen or narrow it.
    """
    for nlev in (20, 30, 40):
        for stretching in (0.0, 2.0):
            coord = make_hybrid_levels(nlev, stretching=stretching)
            thresh = (2.0 * coord.p_ref + 200.0) / 3.0
            for p_s in (7.0e4, 8.0e4, 1.0e5, 1.05e5):     # above threshold
                dp = np.asarray(coord.layer_thickness_dp(np.array([p_s])))
                pf = np.asarray(coord.pressure_at_full(np.array([p_s])))
                assert np.all(dp > 0.0), f"nlev={nlev} st={stretching} p_s={p_s}"
                assert np.all(np.diff(pf, axis=-1) > 0.0)
            # and the documented failure below it is REAL, not hypothetical
            dp_lo = np.asarray(coord.layer_thickness_dp(np.array([6.0e4])))
            assert dp_lo.min() < 0.0, (
                "expected negative layer mass below the threshold; if this "
                "now passes the coordinate was fixed -- update the note above")
            assert 6.0e4 < thresh < 7.0e4


# ---------------------------------------------------------------------------
# CMIP ``tas``: the 2 m similarity profile follows the experiment's
# surface_stability_scheme (codex 2026-08-02: _tas_2m hardcoded the default,
# so a grachev/gryanik coare3 run published a default-native tas while its
# fluxes used the selected stable functions).
# ---------------------------------------------------------------------------
def _tas_stub_state(nlev):
    """Minimal state stub for _tas_2m: strongly stable surface layer
    (T_low = 300 K over a 220 K surface, 10 m/s wind)."""
    from types import SimpleNamespace

    import jax.numpy as jnp

    ncol = 2
    t3 = jnp.broadcast_to(jnp.linspace(250.0, 300.0, nlev), (ncol, nlev))
    u3 = jnp.full((ncol, nlev), 10.0)
    v3 = jnp.zeros((ncol, nlev))
    p_s = jnp.full((ncol,), 1.0e5)
    return SimpleNamespace(
        T=SimpleNamespace(data=t3),
        u=SimpleNamespace(data=u3),
        v=SimpleNamespace(data=v3),
        p_s=SimpleNamespace(data=p_s),
    )


def _tas_for(stability_scheme=None, bulk_scheme=None):
    import jax.numpy as jnp

    coord = create_sigma_coordinate(NLEV)
    kw = {}
    if stability_scheme is not None:
        kw["surface_stability_scheme"] = stability_scheme
    if bulk_scheme is not None:
        kw["surface_bulk_scheme"] = bulk_scheme
    dc = DiagnosticCollector(
        nlev=NLEV, sigma_full=coord.sigma_full, dsigma=coord.dsigma,
        vcoord=coord, **kw)
    state = _tas_stub_state(NLEV)
    q_v = jnp.full((2, NLEV), 2e-3)
    sst = jnp.full((2,), 220.0)
    sic = jnp.zeros((2,))
    return np.asarray(dc._tas_2m(state, q_v, sst, sic, T_ice=271.35))


def test_tas_2m_follows_surface_stability_scheme():
    tas_default = _tas_for()                       # no kwargs: legacy path
    tas_dyer = _tas_for("dyer1974")                # explicit default
    tas_grachev = _tas_for("grachev2007_sheba")
    # Default is byte-identical with and without the new kwarg.
    np.testing.assert_array_equal(tas_default, tas_dyer)
    # The selected SHEBA tail moves the stable 2 m temperature materially.
    assert np.all(np.abs(tas_grachev - tas_default) > 0.5), (
        tas_default, tas_grachev)
    # Physical bracket: between the surface and the lowest-level temperature.
    for tas in (tas_default, tas_grachev):
        assert np.all(tas >= 220.0) and np.all(tas <= 300.0)


def test_tas_2m_follows_surface_bulk_scheme():
    """The tas profile uses the experiment's MOST bulk scheme; the non-MOST
    default "constant" keeps the historical coare3 stand-in byte-identically
    (so default runs are unchanged), while large_yeager (linear Dyer stable
    branch) departs from the coare3 native (BH91-form) profile."""
    tas_default = _tas_for()
    tas_constant = _tas_for(bulk_scheme="constant")
    tas_coare3 = _tas_for(bulk_scheme="coare3")
    tas_ly = _tas_for(bulk_scheme="large_yeager")
    np.testing.assert_array_equal(tas_default, tas_constant)
    np.testing.assert_array_equal(tas_default, tas_coare3)
    assert np.all(np.abs(tas_ly - tas_default) > 0.5), (tas_default, tas_ly)
    assert np.all(tas_ly >= 220.0) and np.all(tas_ly <= 300.0)


# ---------------------------------------------------------------------------
# _tas_2m over LAND.  The diagnostic built its surface from SST / sea ice only,
# with a saturated humidity everywhere, so over land it published the
# neighbouring OCEAN's temperature — a February Arctic land bias of -10 K could
# not be read from this field because the field contained no land.
# ---------------------------------------------------------------------------
def _tas_land_for(**kw):
    import jax.numpy as jnp

    coord = create_sigma_coordinate(NLEV)
    dc = DiagnosticCollector(nlev=NLEV, sigma_full=coord.sigma_full,
                            dsigma=coord.dsigma, vcoord=coord)
    state = _tas_stub_state(NLEV)
    q_v = jnp.full((2, NLEV), 2e-3)
    sst = jnp.full((2,), 220.0)
    sic = jnp.zeros((2,))
    return np.asarray(dc._tas_2m(state, q_v, sst, sic, T_ice=271.35, **kw))


def test_tas_2m_over_land_follows_the_land_skin_not_the_ocean():
    import jax.numpy as jnp
    land = jnp.ones((2,))
    warm = _tas_land_for(T_land=jnp.full((2,), 260.0), land_fraction=land)
    cold = _tas_land_for(T_land=jnp.full((2,), 240.0), land_fraction=land)
    assert np.all(warm > cold + 1.0), (warm, cold)


def test_tas_2m_over_land_ignores_the_sea_surface():
    """With land fraction 1 the SST must not reach the answer at all."""
    import jax.numpy as jnp
    coord = create_sigma_coordinate(NLEV)
    dc = DiagnosticCollector(nlev=NLEV, sigma_full=coord.sigma_full,
                            dsigma=coord.dsigma, vcoord=coord)
    state = _tas_stub_state(NLEV)
    q_v = jnp.full((2, NLEV), 2e-3)
    kw = dict(T_land=jnp.full((2,), 250.0), land_fraction=jnp.ones((2,)))
    a = np.asarray(dc._tas_2m(state, q_v, jnp.full((2,), 220.0),
                              jnp.zeros((2,)), T_ice=271.35, **kw))
    b = np.asarray(dc._tas_2m(state, q_v, jnp.full((2,), 300.0),
                              jnp.zeros((2,)), T_ice=271.35, **kw))
    np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)


def test_tas_2m_zero_land_fraction_collapses_to_the_ocean_branch():
    """A land fraction of zero must reproduce the no-land-arguments answer.

    This pins the BLEND, not history: both sides run the current code, so it
    cannot certify equality with the pre-land implementation.  That parity was
    checked separately by executing the parent commit's ``_tas_2m`` beside this
    one over stable, neutral and unstable fixtures (exact equality); the two
    land tests above are what make this one able to fail.
    """
    import jax.numpy as jnp
    legacy = _tas_land_for()
    zero_land = _tas_land_for(T_land=jnp.full((2,), 240.0),
                              land_fraction=jnp.zeros((2,)))
    np.testing.assert_allclose(zero_land, legacy, rtol=0.0, atol=0.0)


def test_shared_plev_weights_match_per_field_interpolation():
    """The CMOR feed builds plev19 weights once and applies them to every
    field; that must equal the one-call interpolation bit for bit, and the
    weights must depend on p_s (so reuse across different p_s would show)."""
    dc = _collector(make_hybrid_levels(NLEV))
    rng = np.random.default_rng(7)
    p_s = rng.uniform(5.0e4, 1.04e5, 200)
    w = dc._plev19_weights(p_s)
    for _ in range(3):
        f = rng.uniform(-1.0, 1.0, (200, NLEV))
        np.testing.assert_array_equal(
            dc._apply_plev19(f, w).view(np.uint64),
            np.asarray(dc._interp_to_plev19(f, p_s)).view(np.uint64))
    other = dc._plev19_weights(p_s * 0.8)
    assert not np.array_equal(w[2], other[2])
    assert not np.array_equal(w[0], other[0])
