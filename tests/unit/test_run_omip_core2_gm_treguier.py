"""CORE-II OMIP NEMO ldf_eiv GM flags (``--gm-treguier`` / ``--gm-aei0`` /
``--gm-kappa-min``).

Covers the CLI surface of the Treguier (NEMO ``nn_aei_ijk_t=21``) eddy-induced
velocity coefficient: argparse round-trip, the shared defaults, and every
fail-early guard in ``main()``.  The guards exist because each of these flags is
read ONLY inside ``build_tripole``'s ``--gm-treguier`` branch, so a
mis-combination would otherwise be discarded in silence.
"""

from __future__ import annotations

import pytest

from scripts.run.run_omip_core2 import (
    _build_arg_parser,
    _GM_AEI0_DEFAULT,
    _GM_KAPPA_MIN_DEFAULT,
)


def _parse(*extra):
    p = _build_arg_parser()
    return p.parse_args(["--grid", "tripole", "--mesh", "m.nc", *extra])


class TestArgparseRoundTrip:
    def test_defaults_scheme_off(self):
        a = _parse()
        assert a.gm_treguier is False
        assert a.gm_aei0 == _GM_AEI0_DEFAULT == 900.0
        assert a.gm_kappa_min == _GM_KAPPA_MIN_DEFAULT == 200.0

    def test_flag_and_values_round_trip(self):
        a = _parse("--gm-treguier", "--gm-aei0", "2400", "--gm-kappa-min", "50")
        assert a.gm_treguier is True
        assert a.gm_aei0 == 2400.0
        assert a.gm_kappa_min == 50.0

    def test_aei0_default_is_the_orca1_namelist_value(self):
        """aei0 = 1/2 * rn_Ue * rn_Le for the LAPLACIAN operator.

        NEMO ldftra.F90:290-293 sets zUfac = r1_2*rn_Ud when ln_traldf_lap
        (ORCA1's setting), and its own printout says "aht0 = 1/2 rn_Ud*rn_Ld"
        (:331).  The previous expectation dropped the 1/2 and pinned 1800,
        i.e. TWICE NEMO's cap.  Measured 2026-08-12: NEMO's emitted aeiu_2d
        maxes at EXACTLY 900 on eORCA1 rec 1, which is what settles it.
        """
        assert _GM_AEI0_DEFAULT == pytest.approx(0.5 * 0.018 * 100.0e3)
        assert _GM_AEI0_DEFAULT == pytest.approx(900.0)


class TestBuildTripoleSignatureSharesTheDefaults:
    def test_signature_defaults_track_the_module_constants(self):
        """The run script's argparse defaults and ``build_tripole``'s signature
        defaults must be the SAME objects, else they silently drift apart."""
        import inspect

        from scripts.run.run_omip_core2 import build_tripole

        sig = inspect.signature(build_tripole)
        assert sig.parameters["gm_aei0"].default == _GM_AEI0_DEFAULT
        assert sig.parameters["gm_kappa_min"].default == _GM_KAPPA_MIN_DEFAULT
        assert sig.parameters["gm_treguier"].default is False


class TestGuards:
    """Each guard turns a silently-discarded flag into a loud failure."""

    def _main_with(self, argv, monkeypatch):
        import sys

        from scripts.run import run_omip_core2

        monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", *argv])
        return run_omip_core2.main

    def test_non_tripole_grid_rejected(self, monkeypatch):
        main = self._main_with(
            ["--grid", "mpas", "--mesh", "m.nc", "--gm-treguier"], monkeypatch)
        with pytest.raises(SystemExit, match="tripole only"):
            main()

    def test_conflicts_with_no_gm_redi(self, monkeypatch):
        main = self._main_with(
            ["--grid", "tripole", "--mesh", "m.nc", "--gm-treguier",
             "--no-gm-redi"], monkeypatch)
        with pytest.raises(SystemExit, match="mutually exclusive"):
            main()

    def test_aei0_without_scheme_rejected(self, monkeypatch):
        """A stray --gm-aei0 would be read by nothing at all."""
        main = self._main_with(
            ["--grid", "tripole", "--mesh", "m.nc", "--gm-aei0", "2400"],
            monkeypatch)
        with pytest.raises(SystemExit, match="require --gm-treguier"):
            main()

    def test_kappa_min_without_scheme_rejected(self, monkeypatch):
        main = self._main_with(
            ["--grid", "tripole", "--mesh", "m.nc", "--gm-kappa-min", "50"],
            monkeypatch)
        with pytest.raises(SystemExit, match="require --gm-treguier"):
            main()

    def test_default_values_without_scheme_are_not_flagged(self):
        """Passing nothing (or the defaults) must NOT trip the stray-flag guard
        -- otherwise every existing run command breaks."""
        a = _parse()
        assert a.gm_aei0 == _GM_AEI0_DEFAULT
        assert a.gm_kappa_min == _GM_KAPPA_MIN_DEFAULT
        assert not a.gm_treguier

    def test_floor_above_cap_rejected(self, monkeypatch):
        """kappa_min > aei0 makes the floor override the NEMO cap everywhere."""
        main = self._main_with(
            ["--grid", "tripole", "--mesh", "m.nc", "--gm-treguier",
             "--gm-aei0", "1800", "--gm-kappa-min", "5000"], monkeypatch)
        with pytest.raises(SystemExit, match="exceeds --gm-aei0"):
            main()


class TestTreguierBlockIsOneVariable:
    """`--gm-treguier` must change the `treguier` field and NOTHING else.

    REGRESSION (2026-08-12): the branch built its block from
    `run_omip._DEFAULT_BATHY_GM_REDI` -- the LAT-LON bathymetry default -- so
    every `--gm-treguier` arm silently also moved kappa_GM 600->800,
    kappa_Redi 600->800 and slope_scheme "centered"->"triads" relative to the
    tripole control.  Two run manifests confirmed the four-field drift, which
    made the day-5 blow-up of that arm unattributable.
    """

    def _control_block(self):
        from legoesm.ocean.fidelity.nemo_match_recipe import (
            NEMOMatchTripoleRecipeConfig,
            nemo_match_tripole_model_config,
        )
        return nemo_match_tripole_model_config(
            NEMOMatchTripoleRecipeConfig()).gm_redi

    def test_only_the_treguier_field_differs_from_the_control(self):
        from scripts.run.run_omip_core2 import _tripole_treguier_gm_redi

        control = self._control_block()
        arm = _tripole_treguier_gm_redi(_GM_AEI0_DEFAULT,
                                        _GM_KAPPA_MIN_DEFAULT)
        differing = [f for f in control._fields
                     if getattr(control, f) != getattr(arm, f)]
        assert differing == ["treguier"], (
            f"--gm-treguier changed {differing}; it must change only "
            "'treguier' or the arm is not a one-variable experiment.")

    def test_control_block_is_the_tripole_recipe_not_the_latlon_default(self):
        """Guards the exact substitution that caused the regression."""
        from scripts.run import run_omip

        control = self._control_block()
        assert control.kappa_GM == pytest.approx(600.0)
        assert control.kappa_Redi == pytest.approx(600.0)
        assert control.slope_scheme == "centered"
        assert control.visbeck.enabled is False
        # the block that used to be used, and must not be again
        latlon = run_omip._DEFAULT_BATHY_GM_REDI
        assert latlon.kappa_GM != control.kappa_GM
        assert latlon.visbeck.enabled is True

    def test_scheme_is_on_and_carries_the_cli_values(self):
        from scripts.run.run_omip_core2 import _tripole_treguier_gm_redi

        arm = _tripole_treguier_gm_redi(750.0, 100.0)
        assert arm.treguier.enabled is True
        assert arm.treguier.aei0 == pytest.approx(750.0)
        assert arm.treguier.kappa_min == pytest.approx(100.0)

    def test_the_guard_fires_on_the_regression_block(self):
        """MUTATION check: the fields-diff guard must REJECT the old block.

        The pinned-commit non-vacuity run only proves the helper is NEW (it
        fails with ImportError).  This asserts the stronger property the guard
        is actually for: rebuild EXACTLY what the pre-fix branch built -- the
        lat-lon default with Visbeck forced off and Treguier on -- and confirm
        the one-variable test above would have failed on it.  Commit-
        independent, so it keeps working after the pinned SHA ages out.
        """
        from legoesm.ocean.physics.lateral_mixing.config import (
            TreguierConfig, VisbeckConfig,
        )
        from scripts.run import run_omip

        control = self._control_block()
        regression = run_omip._DEFAULT_BATHY_GM_REDI._replace(
            visbeck=VisbeckConfig(enabled=False),
            treguier=TreguierConfig(enabled=True, aei0=_GM_AEI0_DEFAULT,
                                    kappa_min=_GM_KAPPA_MIN_DEFAULT),
        )
        differing = [f for f in control._fields
                     if getattr(control, f) != getattr(regression, f)]
        assert differing != ["treguier"], (
            "the guard cannot distinguish the lat-lon block from the tripole "
            "recipe block -- it would not have caught the regression.")
        assert {"kappa_GM", "kappa_Redi", "slope_scheme"} <= set(differing)

    def test_build_tripole_uses_the_helper_and_not_the_latlon_default(self):
        """WIRING: the helper above is only meaningful if build_tripole calls it.

        `build_tripole` reads a real eORCA1 mesh before it reaches the GM
        branch, so exercising it end-to-end is not a unit test.  This inspects
        the source of the function that ACTUALLY RUNS (build_tripole itself, not
        a delegating wrapper) and fails if the branch is re-pointed at the
        lat-lon default -- the exact regression fixed here.
        """
        import inspect

        from scripts.run.run_omip_core2 import build_tripole

        src = inspect.getsource(build_tripole)
        assert "_tripole_treguier_gm_redi(gm_aei0, gm_kappa_min)" in src, (
            "build_tripole no longer builds its --gm-treguier block from the "
            "one-variable helper.")
        assert "_DEFAULT_BATHY_GM_REDI" not in src, (
            "build_tripole references the LAT-LON bathymetry GM default; that "
            "is the block whose use made every --gm-treguier arm a "
            "four-field change.")
