

class TestNEMONormalization:
    """`normalization="live_s"` = NEMO ``sbcssr`` nn_sssr=2, which ORCA1 runs.

    NEMO (sbcssr.F90:132-134):
        zerp = zsrp * coefice * (sss_m - sss_target) / MAX(sss_m, 1e-20)
    i.e. the salinity->water-flux conversion divides by the LIVE surface
    salinity.  legoESM historically divided by S_target, which is the
    virtual-salt convention and is why the real_freshwater closure refused to
    run alongside restoring.
    """

    def _cfg(self, normalization, **kw):
        from legoesm.ocean.forcing.sss_restoring import SSSRestoringConfig

        return SSSRestoringConfig(
            enabled=True, tau_restore_days_default=45.5, z1_m=10.0,
            ice_gate=False, normalization=normalization, **kw)

    def _call(self, cfg, S_model, S_target):
        import jax.numpy as jnp

        from legoesm.ocean.forcing.sss_restoring import (
            compute_sss_restoring_flux,
        )

        shp = jnp.shape(S_model)
        return compute_sss_restoring_flux(
            S_model_top=jnp.asarray(S_model),
            S_target=jnp.asarray(S_target),
            lat_deg=jnp.zeros(shp), lon_deg=jnp.zeros(shp),
            ice_concentration=jnp.zeros(shp), config=cfg)

    def test_default_is_unchanged(self):
        """The default must stay `s_target` so existing runs are untouched."""
        from legoesm.ocean.forcing.sss_restoring import SSSRestoringConfig

        assert SSSRestoringConfig().normalization == "s_target"

    def test_identical_when_model_equals_target(self):
        """The two forms can only differ where the model has drifted."""
        import numpy as np

        S = np.array([34.5, 30.0, 36.0])
        a = self._call(self._cfg("s_target"), S, S)["freshwater_flux"]
        b = self._call(self._cfg("live_s"), S, S)["freshwater_flux"]
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)

    def test_live_s_matches_the_nemo_formula(self):
        """Reproduce zerp from the Fortran, up to the sign convention.

        NEMO's `erp` enters `emp` (positive UPWARD, i.e. out of the ocean);
        ours is positive INTO the ocean, so the two differ by a minus sign and
        nothing else.  The piston magnitude is z1/tau, which for the OMIP deck
        (10 m, 45.5 d) is NEMO's rn_deds = -220 mm/day to 3 significant digits.
        """
        import numpy as np

        from legoesm import constants

        S_model = np.array([30.0, 34.5, 38.0])      # fresh / near / salty
        S_target = np.array([34.5, 34.5, 34.5])
        cfg = self._cfg("live_s")
        got = np.asarray(
            self._call(cfg, S_model, S_target)["freshwater_flux"])

        rho0 = constants.rho_ocean
        inv_tau = 1.0 / (45.5 * 86400.0)
        # zsrp [kg/m2/s] equivalent of z1/tau, then NEMO's live-S denominator
        zsrp = rho0 * cfg.z1_m * inv_tau
        want = zsrp * (S_model - S_target) / np.maximum(S_model, cfg.S_floor)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    def test_piston_velocity_is_nemo_rn_deds(self):
        """z1/tau for the OMIP deck must be NEMO's 220 mm/day."""
        mm_per_day = 10.0 / 45.5 * 1000.0
        assert abs(mm_per_day - 220.0) < 0.5

    def test_forms_diverge_where_the_model_drifted(self):
        """A salty Arctic cell is exactly where the denominator matters."""
        import numpy as np

        S_model, S_target = np.array([31.0]), np.array([29.6])   # +1.4 psu
        a = float(np.asarray(
            self._call(self._cfg("s_target"), S_model, S_target)
            ["freshwater_flux"])[0])
        b = float(np.asarray(
            self._call(self._cfg("live_s"), S_model, S_target)
            ["freshwater_flux"])[0])
        assert a != b
        # live_s divides by the LARGER number here, so it restores more weakly
        assert abs(b) < abs(a)
        np.testing.assert_allclose(b / a, 29.6 / 31.0, rtol=1e-10)

    def test_unknown_normalization_raises(self):
        import numpy as np
        import pytest as _pytest

        with _pytest.raises(ValueError, match="unknown SSSRestoringConfig"):
            self._call(self._cfg("nemo"), np.array([34.0]), np.array([34.0]))


class TestRealFreshwaterRestoringGuard:
    """`real_freshwater` x `--sss-restore`: refused only for the NON-NEMO form.

    The rule lives in `main` AFTER the model is built, so a CLI-level test
    cannot reach it without a real eORCA1 mesh (an earlier version of this gate
    tried, and the run died building the grid before the guard ever ran).  The
    predicate is therefore split out and tested directly.
    """

    def _f(self, *a):
        from scripts.run.run_omip_core2 import (
            real_freshwater_restoring_conflict,
        )

        return real_freshwater_restoring_conflict(*a)

    def test_s_target_with_real_freshwater_is_refused(self):
        msg = self._f("real_freshwater", True, None)
        assert msg is not None
        # the message must name the way OUT, not just the problem
        assert "--sss-restore-normalization live_s" in msg

    def test_live_s_with_real_freshwater_is_allowed(self):
        assert self._f("real_freshwater", True, "live_s") is None

    def test_explicit_s_target_is_still_refused(self):
        assert self._f("real_freshwater", True, "s_target") is not None

    def test_no_restoring_is_allowed(self):
        assert self._f("real_freshwater", False, None) is None

    def test_virtual_salt_flux_closure_is_untouched(self):
        for norm in (None, "s_target", "live_s"):
            assert self._f("virtual_salt_flux", True, norm) is None
            assert self._f(None, True, norm) is None
