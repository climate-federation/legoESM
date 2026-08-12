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
