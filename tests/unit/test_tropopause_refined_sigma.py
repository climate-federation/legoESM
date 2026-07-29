"""Tropopause-refined sigma: redistribute layers at FIXED level count.

The uniform-in-sigma default spaces every layer ~33 hPa at 30 levels, so
the tropical tropopause layer gets ~3 levels and the model forms no cold
point (measured 2026-07-25: coldest tropical level at the ~26 hPa lid).
Adding levels does not fix the distribution: the L40 run (uniform sigma,
24.4 hPa in EVERY layer) blew up at day 46 at dt=60, across levels 0-11
(10-302 hPa).  That failure has NOT been attributed to a mechanism, so no
test here claims the redistribution is "safer" than L40 — the refined grid
is in fact 42% THINNER than L40 at 99-175 hPa, inside the same band.

These pin what IS established: identity at refine=1, more TTL resolution at
refine>1, a bounded boundary-layer cost, a bounded grid-stretching ratio,
and — the load-bearing conservation property — that the dycore's vertical
biharmonic filter still conserves column enthalpy and water on the
non-uniform grid (it did not; see test_del4_filter_conserves_column).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    vertical_del4_T_tendency,
)
from legoesm.grids.vertical import (
    create_sigma_coordinate,
    make_hybrid_levels,
    tropopause_refined_sigma_half,
)

P_S = 985.0  # hPa, the campaign's typical surface pressure


def _dp(half):
    return np.diff(np.asarray(half)) * P_S


class TestGridProperties:
    def test_identity_at_refine_one(self):
        """refine=1 must reproduce the uniform grid EXACTLY — the default
        path stays byte-identical."""
        got = tropopause_refined_sigma_half(30, refine=1.0)
        np.testing.assert_allclose(got, np.linspace(0.01, 1.0, 31), atol=1e-12)

    def test_monotone_and_endpoints_exact(self):
        for n in (20, 30, 40):
            h = tropopause_refined_sigma_half(n, refine=3.0)
            assert h.shape == (n + 1,)
            assert np.all(np.diff(h) > 0)
            assert h[0] == 0.01 and h[-1] == 1.0

    def test_refinement_adds_ttl_levels(self):
        """The whole point: more levels between 70 and 200 hPa."""
        uni = tropopause_refined_sigma_half(30, refine=1.0) * P_S
        ref = tropopause_refined_sigma_half(30, refine=3.0) * P_S
        n_uni = int(((uni >= 70) & (uni <= 200)).sum())
        n_ref = int(((ref >= 70) & (ref <= 200)).sum())
        assert n_ref > n_uni, (n_uni, n_ref)

    def test_ttl_spacing_improves(self):
        """Layer thickness AT the tropopause shrinks."""
        for h, label in ((tropopause_refined_sigma_half(30, refine=1.0), "uni"),
                         (tropopause_refined_sigma_half(30, refine=3.0), "ref")):
            p = np.asarray(h) * P_S
            mid = 0.5 * (p[1:] + p[:-1])
            near = np.abs(mid - 120.0) < 60.0
            thick = _dp(h)[near].mean()
            if label == "uni":
                uni_thick = thick
            else:
                assert thick < 0.75 * uni_thick, (uni_thick, thick)

    def test_top_layer_not_thinned(self):
        """The TOP layer (k=0, the one whose 1/p adiabatic term and Exner
        amplification are largest) must not thin.  Deliberately NOT labelled a
        safety property for the L40 failure: that spanned levels 0-11 and the
        refined grid is thinner than L40 at 99-175 hPa.  Strict > so a mutant
        that silently disables the refinement (flat density) fails here too."""
        uni = _dp(tropopause_refined_sigma_half(30, refine=1.0))
        ref = _dp(tropopause_refined_sigma_half(30, refine=3.0))
        assert ref[0] > uni[0], (uni[0], ref[0])

    def test_grid_stretching_ratio_bounded(self):
        """GRID SMOOTHNESS is the criterion that actually bounds `refine`
        (the vertical advective CFL is only 0.26 at refine=3/dt=75, measured).
        Every grid this model has run successfully has a max adjacent-layer
        thickness ratio <= 1.07; refine=3 costs 1.62 and the config validator
        stops at 3.  Pinned so a density change cannot quietly worsen it."""
        for refine, cap in ((2.0, 1.35), (3.0, 1.65)):
            d = np.diff(tropopause_refined_sigma_half(30, refine=refine))
            ratio = float(np.max(np.maximum(d[1:] / d[:-1], d[:-1] / d[1:])))
            assert 1.0 < ratio < cap, (refine, ratio)

    def test_boundary_layer_cost_is_bounded(self):
        """The boundary layer PAYS for the tropopause — an accepted, measured
        cost, not a free lunch: the lowest layer coarsens ~30% (33 -> 42 hPa
        at 30 levels, refine=3).  Pinned so a future density change cannot
        quietly make it worse (and nowhere near the 166 hPa a naive
        log-spaced grid would give)."""
        uni = _dp(tropopause_refined_sigma_half(30, refine=1.0))
        ref = _dp(tropopause_refined_sigma_half(30, refine=3.0))
        assert 1.0 < ref[-1] / uni[-1] < 1.35, (uni[-1], ref[-1])

    def test_layers_come_from_the_mid_troposphere(self):
        """Conservation of levels: the TTL gain is paid for by the
        (over-resolved) mid-troposphere, not the ends."""
        uni = tropopause_refined_sigma_half(30, refine=1.0) * P_S
        ref = tropopause_refined_sigma_half(30, refine=3.0) * P_S
        band = lambda p, lo, hi: int(((p >= lo) & (p <= hi)).sum())  # noqa: E731
        assert band(ref, 400, 800) < band(uni, 400, 800)

    def test_monotone_in_refine_strength(self):
        counts = []
        for r in (1.0, 2.0, 3.0, 5.0):
            p = tropopause_refined_sigma_half(30, refine=r) * P_S
            counts.append(int(((p >= 70) & (p <= 200)).sum()))
        assert counts == sorted(counts), counts

    @pytest.mark.parametrize("bad", [
        dict(refine=0.5), dict(width=0.0), dict(sigma_refine=1.5),
        dict(sigma_top=0.2, sigma_refine=0.1),
    ])
    def test_invalid_parameters_raise(self, bad):
        with pytest.raises(ValueError):
            tropopause_refined_sigma_half(30, **bad)


class TestCoordinateIntegration:
    def test_default_is_bit_identical(self):
        a = create_sigma_coordinate(30)
        b = create_sigma_coordinate(30, tropopause_refine=1.0)
        np.testing.assert_array_equal(np.asarray(a.sigma_half),
                                      np.asarray(b.sigma_half))
        np.testing.assert_array_equal(np.asarray(a.dsigma),
                                      np.asarray(b.dsigma))

    def test_refined_coordinate_is_self_consistent(self):
        """The derived Simmons-Burridge quantities must stay finite and
        physical on the refined grid (they divide by dsigma and take logs
        of sigma ratios)."""
        c = create_sigma_coordinate(30, tropopause_refine=3.0)
        for name in ("sigma_full", "sigma_half", "dsigma", "ln_ratio",
                     "alpha", "fractional_sigma", "dsigma_full"):
            arr = np.asarray(getattr(c, name))
            assert np.all(np.isfinite(arr)), name
        assert np.all(np.asarray(c.dsigma) > 0)
        # alpha is the Simmons-Burridge weight: strictly within (0, 1).
        alpha = np.asarray(c.alpha)
        assert np.all((alpha > 0) & (alpha < 1)), alpha

    def test_refined_cold_point_band_resolved(self):
        """End-to-end statement of the defect being fixed: at 30 levels the
        refined grid puts >= 4 full levels in 70-200 hPa (uniform gives 2)."""
        uni = np.asarray(create_sigma_coordinate(30).sigma_full) * P_S
        ref = np.asarray(
            create_sigma_coordinate(30, tropopause_refine=3.0).sigma_full) * P_S
        n_uni = int(((uni >= 70) & (uni <= 200)).sum())
        n_ref = int(((ref >= 70) & (ref <= 200)).sum())
        # Measured baseline: uniform puts 4 full levels in the band; the
        # refinement doubles it.  (Only 3 sit above 100 hPa uniformly.)
        assert n_uni == 4 and n_ref >= 7, (n_uni, n_ref)


class TestConfigPlumbing:
    def test_survives_the_amip_serialization_round_trip(self):
        """``ExperimentConfig <-> AMIPExperimentConfig`` is a live
        deserialization boundary (``driver.checkpoint._config_from_dict_auto``
        routes AMIP-format JSON through it).  Dropping the field there would
        silently report a refined run as uniform."""
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, config_to_dict,
            experiment_config_from_dict,
        )
        cfg = ExperimentConfig(grid=GridConfig(vertical_coord="sigma",
                                               nlev=30, tropopause_refine=3.0))
        assert experiment_config_from_dict(
            config_to_dict(cfg)).grid.tropopause_refine == 3.0
        assert ExperimentConfig.from_amip_config(
            cfg.to_amip_config()).grid.tropopause_refine == 3.0

    @pytest.mark.parametrize("vc,refine,ok", [
        ("sigma", 1.0, True), ("sigma", 3.0, True),
        ("sigma", 0.5, False), ("sigma", 3.5, False),
        ("sigma", float("nan"), False), ("hybrid", 3.0, False),
    ])
    def test_validate_strict_bounds(self, vc, refine, ok):
        """The bound is GRID SMOOTHNESS, not vertical CFL: refine=3 gives a
        1.62 adjacent-layer ratio against <= 1.07 on every grid that has run.
        3.0 is the largest value with a stability arm behind it."""
        from legoesm.driver.config import ExperimentConfig, GridConfig
        cfg = ExperimentConfig(grid=GridConfig(vertical_coord=vc,
                                               tropopause_refine=refine))
        if ok:
            cfg.validate_strict()
        else:
            with pytest.raises(ValueError):
                cfg.validate_strict()


class TestNonUniformGridConservation:
    """The dycore's #930 vertical biharmonic filter is a fourth difference on
    the level INDEX, so ``sum_k tend_k == 0`` — which is column conservation
    ONLY when dsigma is constant.  Making the sigma grid non-uniform turned
    that into a real column source/sink: measured -7.86 W/m2 of enthalpy and
    -0.135 mm/day of water at refine=3 for a +-5 K / +-1 g/kg 2-delta mode
    (and -0.99 W/m2 on the already-shipped stretched HYBRID L40 grid).  These
    pin the mass-weighted closure on every grid the model can build."""

    NU = 2.0e-6      # nu_vert4_T, the AMIP/coupled value
    PS_PA = 98500.0

    @staticmethod
    def _checkerboard(n, mean, amp):
        return mean + amp * (-1.0) ** np.arange(n)

    def _leak(self, dsigma, profile):
        """Tendency + its mass-weighted column sum.

        The weight is NORMALISED to unit column so the cases can be compared
        in one unit: the sigma entries carry Delta-sigma (dimensionless) while
        the hybrid entry carries dp [Pa].  Normalising makes every weight a
        fraction of column mass, after which ``c_pd * p_s / g * sum`` is the
        column enthalpy source in W/m2.  The projection normalises internally
        too, so this cannot mask a real closure failure.
        """
        ds = np.asarray(dsigma, dtype=np.float64)
        ds = ds / ds.sum()
        tend = np.asarray(
            vertical_del4_T_tendency(jnp.asarray(profile, dtype=jnp.float64),
                                     self.NU, jnp.asarray(ds)),
            dtype=np.float64)
        return tend, float((tend * ds).sum())

    @pytest.mark.parametrize("label,dsigma", [
        ("sigma-uniform-L30",
         np.diff(np.linspace(0.01, 1.0, 31))),
        ("sigma-refine3-L30",
         np.diff(tropopause_refined_sigma_half(30, refine=3.0))),
        ("sigma-refine2-L40",
         np.diff(tropopause_refined_sigma_half(40, refine=2.0))),
        # HYBRID: the weight must be the TRUE dp = dA*p_ref + dB*p_s, NOT the
        # coordinate's own `dsigma` (= dA + dB), which is the thickness only
        # at p_s = p_ref.  p_s = 620 hPa is a plateau column, where the two
        # differ most (codex round 2).
        ("hybrid-L40-stretch2-ps620",
         np.asarray(
             make_hybrid_levels(40, 200.0, stretching=2.0).dA
             * float(constants.p_ref)
             + make_hybrid_levels(40, 200.0, stretching=2.0).dB * 62000.0,
             dtype=np.float64)),
    ])
    def test_del4_filter_conserves_column(self, label, dsigma):
        """MASS-weighted column integral of the filter tendency is zero on
        EVERY grid, not just the uniform one."""
        n = len(dsigma)
        T = self._checkerboard(n, 250.0, 5.0)
        q = self._checkerboard(n, 5e-3, 1e-3)
        for name, prof, to_flux in (
            ("enthalpy [W/m2]", T,
             constants.c_pd * self.PS_PA / constants.g),
            ("water [mm/day]", q, self.PS_PA / constants.g * 86400.0),
        ):
            tend, massw = self._leak(dsigma, prof)
            assert abs(massw * to_flux) < 1e-9, (label, name, massw * to_flux)
            # Non-vacuous: the filter must still be DOING something.
            assert np.abs(tend).max() > 0.0, (label, name)

    def test_legacy_path_still_index_conserving_and_bit_identical(self):
        """``dsigma=None`` keeps the old behaviour exactly, and on a UNIFORM
        grid passing dsigma changes nothing beyond float round-off — so the
        century run's uniform-grid history is not perturbed."""
        T = jnp.asarray(self._checkerboard(30, 250.0, 5.0), dtype=jnp.float64)
        legacy = np.asarray(vertical_del4_T_tendency(T, self.NU), np.float64)
        ds_uni = np.diff(np.linspace(0.01, 1.0, 31))
        fixed = np.asarray(
            vertical_del4_T_tendency(T, self.NU, jnp.asarray(ds_uni)),
            np.float64)
        assert abs(legacy.sum()) < 1e-18            # legacy index-space closure
        assert np.abs(fixed - legacy).max() < 1e-18  # << float32 ULP of a K/s

    def test_the_bug_is_real_without_the_fix(self):
        """Guards the guard: with ``dsigma=None`` the refined grid DOES leak,
        so the parametrized test above is not passing vacuously."""
        ds = np.diff(tropopause_refined_sigma_half(30, refine=3.0))
        T = jnp.asarray(self._checkerboard(30, 250.0, 5.0), dtype=jnp.float64)
        unfixed = np.asarray(vertical_del4_T_tendency(T, self.NU), np.float64)
        leak_W = (constants.c_pd * self.PS_PA / constants.g
                  * float((unfixed * ds).sum()))
        assert abs(leak_W) > 1.0, leak_W   # measured -7.86 W/m2

    def test_hybrid_needs_the_true_dp_not_dA_plus_dB(self):
        """The hybrid coordinate's own ``dsigma`` (= dA + dB) is the layer
        thickness ONLY at p_s = p_ref, so using it as the mass weight leaves
        the filter conservative only to the p_s/p_ref departure.  Pins that
        the call site passes the real ``dp`` (codex round 2 CONFIRMED-BUG)."""
        hy = make_hybrid_levels(40, 200.0, stretching=2.0)
        dA = np.asarray(hy.dA, np.float64)
        dB = np.asarray(hy.dB, np.float64)
        p_s = 62000.0                                   # plateau column
        dp_true = dA * float(constants.p_ref) + dB * p_s
        dsig_proxy = np.asarray(hy.dsigma_eff, np.float64)
        T = jnp.asarray(self._checkerboard(40, 250.0, 5.0), dtype=jnp.float64)
        to_W = constants.c_pd / constants.g

        def residual(weight_used):
            t = np.asarray(
                vertical_del4_T_tendency(T, self.NU,
                                         jnp.asarray(weight_used)), np.float64)
            return abs(to_W * float((t * dp_true).sum()))

        assert residual(dp_true) < 1e-9                  # correct weight
        assert residual(dsig_proxy) > 1e-2, residual(dsig_proxy)  # the proxy leaks
