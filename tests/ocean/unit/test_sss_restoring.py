

import pytest


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

    def test_normalization_is_inert_away_from_the_cap(self):
        """The APPLIED tendency is normalization-independent unless capped.

        `dS_dt_top` is re-derived as `-freshwater_flux * S_safe/(rho_0*z1)`,
        and `freshwater_flux` was built by DIVIDING by that same `S_safe`, so
        the two cancel.  This is the property that makes the `live_s` option
        insufficient to lift the real_freshwater guard -- assert it directly so
        nobody re-derives the wrong conclusion from the flux diagnostic.
        """
        import numpy as np

        S_model = np.array([31.0, 34.5, 36.0])
        S_target = np.array([29.6, 34.5, 35.0])
        a = np.asarray(self._call(self._cfg("s_target"), S_model, S_target)
                       ["dS_dt_top"])
        b = np.asarray(self._call(self._cfg("live_s"), S_model, S_target)
                       ["dS_dt_top"])
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0.0)

    def test_normalization_bites_only_where_the_cap_binds(self):
        """With a cap tight enough to bind, the applied tendency DOES move."""
        import numpy as np

        S_model, S_target = np.array([31.0]), np.array([29.6])
        tight = dict(max_flux_kg_m2_s=1.0e-6)     # far below the uncapped flux
        a = float(np.asarray(
            self._call(self._cfg("s_target", **tight), S_model, S_target)
            ["dS_dt_top"])[0])
        b = float(np.asarray(
            self._call(self._cfg("live_s", **tight), S_model, S_target)
            ["dS_dt_top"])[0])
        assert a != b
        # at the cap the applied tendency scales with S_safe
        np.testing.assert_allclose(b / a, 31.0 / 29.6, rtol=1e-10)

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
        # the message must explain WHY, in terms of the path that runs
        assert "dS_dt_top" in msg

    def test_live_s_does_NOT_lift_the_guard(self):
        """RETRACTED 2026-08-12 (codex 9383572): an earlier revision allowed
        this, claiming live_s makes restoring a genuine water flux.  It does
        not -- `apply_sss_restoring_step*` consume `dS_dt_top` and edit the
        tracer directly, and S_safe cancels in that re-derivation except at the
        cap.  Restoring stays virtual-salt-like whatever the denominator."""
        msg = self._f("real_freshwater", True, "live_s")
        assert msg is not None
        assert "does NOT lift this" in msg

    def test_explicit_s_target_is_still_refused(self):
        assert self._f("real_freshwater", True, "s_target") is not None

    def test_no_restoring_is_allowed(self):
        assert self._f("real_freshwater", False, None) is None

    def test_virtual_salt_flux_closure_is_untouched(self):
        for norm in (None, "s_target", "live_s"):
            assert self._f("virtual_salt_flux", True, norm) is None
            assert self._f(None, True, norm) is None


class TestRestoringHeatFlux:
    """NEMO carries the HEAT CONTENT of the water restoring moves.

    ``qns = qns - erp * rcp * sst_m`` (sbcssr.F90:138).  Our applier had NO
    heat term at all, which is one of the three structural gaps (the others
    being that restoring never enters the water budget and so never moves
    SSH/volume).
    """

    def _cfg(self, **kw):
        from legoesm.ocean.forcing.sss_restoring import SSSRestoringConfig

        return SSSRestoringConfig(
            enabled=True, tau_restore_days_default=45.5, z1_m=10.0,
            ice_gate=False, **kw)

    def _call(self, S_model, S_target, sst_C=None, **kw):
        import jax.numpy as jnp

        from legoesm.ocean.forcing.sss_restoring import (
            compute_sss_restoring_flux,
        )

        shp = jnp.shape(S_model)
        return compute_sss_restoring_flux(
            S_model_top=jnp.asarray(S_model),
            S_target=jnp.asarray(S_target),
            lat_deg=jnp.zeros(shp), lon_deg=jnp.zeros(shp),
            ice_concentration=jnp.zeros(shp), config=self._cfg(**kw),
            sst_C=(None if sst_C is None else jnp.asarray(sst_C)))

    def test_absent_unless_sst_is_supplied(self):
        """Back-compatible: every existing caller passes no SST."""
        import numpy as np

        out = self._call(np.array([34.0]), np.array([34.0]))
        assert out["heat_flux"] is None

    def test_matches_nemo_qns_term(self):
        """qns contribution = +F*rcp*sst, F positive INTO the ocean.

        NEMO's erp is positive UPWARD (added to emp), ours is positive into the
        ocean, so erp = -F and `qns -= erp*rcp*sst` becomes `qns += F*rcp*sst`.
        """
        import numpy as np

        from legoesm import constants

        S_model = np.array([31.0, 34.5, 36.0])
        S_target = np.array([29.6, 34.5, 35.0])
        sst = np.array([-1.8, 12.0, 25.0])
        out = self._call(S_model, S_target, sst_C=sst)
        F = np.asarray(out["freshwater_flux"])
        np.testing.assert_allclose(
            np.asarray(out["heat_flux"]),
            F * constants.c_p_seawater * sst, rtol=1e-12)

    def test_sign_a_too_salty_cell_gains_water_and_heat(self):
        """Physical direction, stated so a flipped sign cannot pass silently.

        A cell saltier than target must be FRESHENED: water in (F > 0). That
        water carries the ocean's own SST, so at a positive SST the heat term
        is positive (into the ocean) and the column grows without cooling.
        """
        import numpy as np

        out = self._call(np.array([31.0]), np.array([29.6]), sst_C=np.array([4.0]))
        assert float(np.asarray(out["freshwater_flux"])[0]) > 0.0
        assert float(np.asarray(out["heat_flux"])[0]) > 0.0

    def test_freezing_point_sst_gives_a_negative_term(self):
        """At sub-zero SST the same inflow REMOVES heat -- rcp*sst is signed."""
        import numpy as np

        out = self._call(np.array([31.0]), np.array([29.6]),
                         sst_C=np.array([-1.8]))
        assert float(np.asarray(out["freshwater_flux"])[0]) > 0.0
        assert float(np.asarray(out["heat_flux"])[0]) < 0.0

    def test_zero_where_there_is_no_restoring(self):
        import numpy as np

        out = self._call(np.array([34.5]), np.array([34.5]),
                         sst_C=np.array([10.0]))
        np.testing.assert_allclose(np.asarray(out["heat_flux"]), 0.0, atol=1e-18)


class TestRestoringChannelFlag:
    """`--sss-restore-channel water_flux` = NEMO nn_sssr=2 routing.

    The two channels are EXCLUSIVE. Under `water_flux` the flux enters
    `fw.restoring` / `q_net` before the step and the post-step tracer edit is
    skipped; running both would apply restoring twice and still look plausible,
    which is the sharpest failure mode of this change.
    """

    def _parse(self, *extra):
        from scripts.run.run_omip_core2 import _build_arg_parser

        return _build_arg_parser().parse_args(
            ["--grid", "tripole", "--mesh", "m.nc", *extra])

    def _main_with(self, argv, monkeypatch):
        import sys

        from scripts.run import run_omip_core2

        monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", *argv])
        return run_omip_core2.main

    def test_default_is_the_tracer_channel(self):
        assert self._parse().sss_restore_channel is None

    def test_round_trip(self):
        a = self._parse("--sss-restore-channel", "water_flux")
        assert a.sss_restore_channel == "water_flux"

    def test_unknown_channel_rejected(self):
        import pytest as _pytest

        with _pytest.raises(SystemExit):
            self._parse("--sss-restore-channel", "emp")

    def test_water_flux_requires_live_s(self, monkeypatch):
        """s_target is the virtual-salt conversion; routing it as water would
        move mass at a rate derived from the wrong denominator."""
        import pytest as _pytest

        main = self._main_with(
            ["--grid", "tripole", "--mesh", "m.nc", "--sss-restore",
             "--sss-restore-channel", "water_flux"], monkeypatch)
        with _pytest.raises(SystemExit, match="requires --sss-restore-normalization"):
            main()

    def test_water_flux_without_restoring_rejected(self, monkeypatch):
        import pytest as _pytest

        main = self._main_with(
            ["--grid", "tripole", "--mesh", "m.nc",
             "--sss-restore-channel", "water_flux",
             "--sss-restore-normalization", "live_s"], monkeypatch)
        with _pytest.raises(SystemExit, match="restoring that is switched off"):
            main()

    def test_water_flux_ALONE_reports_the_most_specific_problem(self, monkeypatch):
        """`--sss-restore-channel water_flux` and nothing else.

        The first ordering fix only moved the CLOSURE check, so this input
        still reported "requires live_s" -- and the test above could not see it
        because it supplies live_s (codex 9387497). With no restoring at all,
        the useful diagnosis is that the channel has nothing to route.
        """
        import pytest as _pytest

        main = self._main_with(
            ["--grid", "tripole", "--mesh", "m.nc",
             "--sss-restore-channel", "water_flux"], monkeypatch)
        with _pytest.raises(SystemExit, match="restoring that is switched off"):
            main()

    def test_kpp_buoyancy_gets_restoring_even_with_no_other_freshwater(self):
        """`sf.freshwater` may be None; the restoring must still reach KPP.

        The first version guarded with `if sf.freshwater is not None`, so on a
        run with no other freshwater the restoring drove volume and heat but
        NOT haline buoyancy -- a partial application, which is worse than
        either extreme because it is self-consistent-looking (codex 9387497).
        """
        import inspect

        from scripts.run import run_omip_core2

        src = inspect.getsource(run_omip_core2.main)
        block = src[src.index("if _sss_water_flux:"):]
        block = block[:block.index("state = _ensure_sharded_state")]
        assert "if sf.freshwater is not None:" not in block, (
            "restoring is skipped from the KPP buoyancy sum when there is no "
            "other freshwater; it must be SET, not skipped.")
        assert "_fw_restore if sf.freshwater is None" in block

    @pytest.mark.parametrize("loop_name", ["main", "run_fesom_forced_loop"])
    def test_the_two_channels_are_exclusive_in_source(self, loop_name):
        """The post-step applier must be guarded by `not _sss_water_flux`.

        Asserted on the source of the function that RUNS, because reaching this
        branch in a unit test needs a real eORCA1 mesh. A double application
        would not crash -- it would quietly double the restoring -- so the
        exclusivity is worth pinning even by this weaker means.

        EVERY loop that carries both channels is checked, not just ``main``.
        The FESOM lane runs its own loop and returns before ``main``'s, so a
        check on ``main`` alone leaves the identical failure mode ungated
        there -- deleting the fesom guard would keep every suite green.
        """
        import inspect

        from scripts.run import run_omip_core2

        src = inspect.getsource(getattr(run_omip_core2, loop_name))
        assert "and not _sss_water_flux:" in src, (
            f"{loop_name}: the post-step SSS-restoring applier is no longer "
            "guarded against the water-flux channel; restoring would be "
            "applied twice.")
        # and the pre-step branch must exist, in either spelling
        assert ("if _sss_water_flux:" in src
                or "if sss_restore_cfg is not None and _sss_water_flux:" in src), (
            f"{loop_name}: no pre-step water-flux branch; the water_flux "
            "channel would silently do nothing.")

    def test_water_flux_requires_real_freshwater(self, monkeypatch):
        """The DOUBLE-APPLICATION guard (codex 9387241 RED).

        `virtual_salt_flux(freshwater, ...)` builds its net internally and that
        net includes `fw.restoring`, so under the default closure a routed
        restoring would reach the ocean twice: as volume through eta/z-star,
        and again as the closure's virtual-salt tendency.
        """
        import pytest as _pytest

        main = self._main_with(
            ["--grid", "tripole", "--mesh", "m.nc", "--sss-restore",
             "--sss-restore-channel", "water_flux",
             "--sss-restore-normalization", "live_s"], monkeypatch)
        with _pytest.raises(SystemExit, match="requires --freshwater-closure"):
            main()

    def test_the_full_nemo_combination_is_allowed(self):
        """water_flux + live_s + real_freshwater must NOT be refused.

        This is the combination the real_freshwater guard's own message has
        always pointed at: restoring routed as a genuine water flux with the
        tracer-side edit skipped.
        """
        from scripts.run.run_omip_core2 import (
            real_freshwater_restoring_conflict,
        )

        assert real_freshwater_restoring_conflict(
            "real_freshwater", True, "live_s", "water_flux") is None

    def test_the_exemption_is_narrow(self):
        """Neither piece alone lifts the guard."""
        from scripts.run.run_omip_core2 import (
            real_freshwater_restoring_conflict,
        )

        # right channel, wrong conversion
        assert real_freshwater_restoring_conflict(
            "real_freshwater", True, "s_target", "water_flux") is not None
        # right conversion, tracer channel (the retracted claim)
        assert real_freshwater_restoring_conflict(
            "real_freshwater", True, "live_s", None) is not None
        assert real_freshwater_restoring_conflict(
            "real_freshwater", True, "live_s", "tracer") is not None

    def test_water_flux_block_handles_absent_freshwater_forcing(self):
        """`fw` is None when a run has no P-E / runoff / ice.

        The first version of the routing called `fw._replace` unconditionally,
        which would crash at step 1 on such a run (codex 9387241). Under this
        channel the restoring IS physical freshwater, so it needs a carrier.
        """
        import inspect

        from scripts.run import run_omip_core2

        src = inspect.getsource(run_omip_core2.main)
        block = src[src.index("if _sss_water_flux:"):]
        block = block[:block.index("state = _ensure_sharded_state")]
        assert "if fw is None:" in block, (
            "the water-flux routing assumes `fw` exists; it is None on runs "
            "with no P-E, runoff or ice.")
        assert "FreshwaterForcing(" in block


class TestRestoringExcludedFromNormalization:
    """The global freshwater normalization must NOT touch the restoring flux.

    `--no-normalize-freshwater` is off in the OMIP runs (the manifests resolve
    `normalize_freshwater: true`), so with restoring in the water budget the
    normalizer was removing the restoring's own global mean from every cell:
    a spurious uniform water flux AND a globally weakened restoring, neither
    of which raises anything. NEMO adds `erp` straight to `emp` and never
    normalizes it (sbcssr.F90:137).
    """

    def test_normalizing_the_physical_net_leaves_restoring_intact(self):
        """Reference semantics, independent of the model classes.

        normalize(phys) + restoring must have the SAME area-mean restoring as
        the raw restoring field -- i.e. the restoring's global mean survives,
        while the physical net's is removed.
        """
        import numpy as np

        area = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.ones(4)
        phys = np.array([1.0, -2.0, 3.0, 0.5])        # unbalanced, as CORE-II is
        rest = np.array([0.2, 0.2, 0.2, 0.2])         # nonzero global mean

        def amean(x):
            return float(np.sum(x * area * mask) / np.sum(area * mask))

        combined = phys + rest
        # WRONG (what the code did): normalize the whole net
        wrong = combined - amean(combined) * mask
        # RIGHT: normalize only the physical part, add restoring back
        right = (phys - amean(phys) * mask) + rest

        assert abs(amean(right - phys + phys)) >= 0.0        # sanity, no-op
        # the physical imbalance is removed in BOTH
        assert abs(amean(right) - amean(rest)) < 1e-12
        # but the wrong form has ZERO net, having eaten the restoring mean
        assert abs(amean(wrong)) < 1e-12
        # and the two differ by exactly the restoring's global mean
        np.testing.assert_allclose(right - wrong, amean(rest) * mask,
                                   rtol=1e-12, atol=1e-15)

    def test_both_model_paths_exclude_restoring(self):
        """Pin the fix in the two normalizers that run."""
        import inspect

        from legoesm.ocean.dynamics import (
            ocean_model_latlon_cgrid, ocean_model_mpas,
        )

        for mod in (ocean_model_latlon_cgrid, ocean_model_mpas):
            src = inspect.getsource(mod)
            assert 'getattr(freshwater, "restoring", None)' in src, (
                f"{mod.__name__} no longer excludes the restoring flux from "
                "the freshwater normalization")
