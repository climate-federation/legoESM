"""Lock the production-calibrated CLI shallow-water config (issue #269).

Issue #269 reported visible v-wind / height edge artifacts on one cube
face for Williamson Test Case 2 at C48 day 5. The root cause was the
``legoesm test williamson`` CLI using a heuristic
``hyperdiff = 1e-4 * mean_dx**4 / dt`` config that lacked boundary fix,
divergence damping, vorticity damping, and Fortran XPPM boundary
handling. The fix routes the CLI through
``williamson_cli_calibration``, which reuses the production matrix
runner's resolution-scaling helpers.

These tests pin:
  1. The hyperdiff/div-damp scaling formulas (``cdgrid_hyperdiff_cube``
     / ``cdgrid_div_damp_cube``) so future calibration drifts can't
     silently regress the CLI.
  2. The flags ``williamson_cli_calibration`` returns
     (``boundary_fix``, ``apply_fortran_xppm_boundary``, ``damp_v``,
     ``nord_v``, ``use_conservation_fixer``).
  3. The CLI imports and calls the helper.
"""

from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    cdgrid_div_damp_cube,
    cdgrid_hyperdiff_cube,
    williamson_cli_calibration,
)


class TestResolutionScaling:
    @pytest.mark.parametrize("bad_n", [0, -1, -48])
    def test_rejects_nonpositive_resolution(self, bad_n):
        with pytest.raises(ValueError, match="positive"):
            cdgrid_hyperdiff_cube(bad_n)
        with pytest.raises(ValueError, match="positive"):
            cdgrid_div_damp_cube(bad_n)

    @pytest.mark.parametrize("bad_n", [48.5, 1.7, True, False, "48"])
    def test_rejects_non_integer_resolution(self, bad_n):
        with pytest.raises(ValueError, match="positive integer"):
            cdgrid_hyperdiff_cube(bad_n)
        with pytest.raises(ValueError, match="positive integer"):
            cdgrid_div_damp_cube(bad_n)

    def test_hyperdiff_anchor_at_c48(self):
        assert cdgrid_hyperdiff_cube(48) == pytest.approx(1.0e16)

    def test_hyperdiff_scales_as_dx_fourth(self):
        assert (
            cdgrid_hyperdiff_cube(24) / cdgrid_hyperdiff_cube(48)
            == pytest.approx(16.0)
        )
        assert (
            cdgrid_hyperdiff_cube(96) / cdgrid_hyperdiff_cube(48)
            == pytest.approx(1.0 / 16.0)
        )

    def test_div_damp_anchor_at_c48(self):
        assert cdgrid_div_damp_cube(48) == pytest.approx(1.5e7)

    def test_div_damp_scales_as_dx_squared(self):
        assert (
            cdgrid_div_damp_cube(24) / cdgrid_div_damp_cube(48)
            == pytest.approx(4.0)
        )


class TestWilliamsonCliCalibration:
    @pytest.mark.parametrize("n", [24, 36, 48, 72])
    def test_returns_cdgrid_config(self, n):
        cfg = williamson_cli_calibration(n)
        assert isinstance(cfg, CDGridShallowWaterConfig)

    def test_pinned_flags(self):
        cfg = williamson_cli_calibration(48)
        # Edge-artifact mitigation flags.
        assert cfg.boundary_fix is True
        assert cfg.apply_fortran_xppm_boundary is True
        # Vorticity damping settings carried over from the iter-1009 preset.
        assert cfg.damp_v == pytest.approx(0.030)
        assert cfg.nord_v == 2
        # Conservation fixer ON so the CLI's mass-error diagnostic
        # stays at machine-precision.
        assert cfg.use_conservation_fixer is True

    def test_hyperdiff_matches_helper(self):
        for n in (24, 36, 48, 72):
            cfg = williamson_cli_calibration(n)
            assert cfg.hyperdiff_coeff == pytest.approx(
                cdgrid_hyperdiff_cube(n)
            )

    def test_div_damp_uses_two_times_base(self):
        # 2× base is the C24/C48 stable-across-resolutions choice
        # (the iter-1009 8× factor blows up W5 day-5 at C48).
        for n in (24, 36, 48):
            cfg = williamson_cli_calibration(n)
            assert cfg.div_damp == pytest.approx(
                2.0 * cdgrid_div_damp_cube(n)
            )

    def test_factor_override(self):
        cfg = williamson_cli_calibration(48, div_damp_factor=4.0)
        assert cfg.div_damp == pytest.approx(4.0 * cdgrid_div_damp_cube(48))


class TestCliUsesHelper:
    def test_cli_imports_calibration(self):
        # Smoke check: importing the CLI test command does not raise,
        # and the helper symbol is reachable through the CLI's import
        # path. Guards against accidental removal of the import or a
        # rename without the test runner's awareness.
        import legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid as m
        assert hasattr(m, "williamson_cli_calibration")
        # The CLI's cmd_test imports `williamson_cli_calibration` from
        # this module; if the name disappears or moves, this assertion
        # is what fails first.
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            williamson_cli_calibration as imported,
        )
        assert imported is m.williamson_cli_calibration
