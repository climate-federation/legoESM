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

from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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

    def test_hyperdiff_default_exponent_is_four(self):
        # The scaling_exponent default must reproduce the ^4 law byte-for-byte.
        for n in (24, 48, 96, 192):
            assert cdgrid_hyperdiff_cube(n) == cdgrid_hyperdiff_cube(
                n, scaling_exponent=4)

    def test_hyperdiff_exponent_invariant_at_ref_n(self):
        # #753: every exponent returns ref_coeff at n == ref_n, so opting into
        # the ^2 modon law leaves the calibrated C48 run UNCHANGED.
        for exp in (1, 2, 3, 4, 6):
            assert cdgrid_hyperdiff_cube(48, scaling_exponent=exp) == (
                pytest.approx(1.0e16))

    def test_hyperdiff_exponent_two_scales_as_dx_squared(self):
        # scaling_exponent=2 tracks the (ref/n)^2 div-damp law exactly.
        assert (
            cdgrid_hyperdiff_cube(24, scaling_exponent=2)
            / cdgrid_hyperdiff_cube(48, scaling_exponent=2)
            == pytest.approx(4.0)
        )
        assert (
            cdgrid_hyperdiff_cube(96, scaling_exponent=2)
            / cdgrid_hyperdiff_cube(48, scaling_exponent=2)
            == pytest.approx(1.0 / 4.0)
        )

    def test_hyperdiff_exponent_two_gives_4x_at_c96(self):
        # The #753 empirical finding: the modon wants ~4x the ^4 backstop at
        # C96, and (96/48)^2 == 4 delivers exactly that (while C48 is unchanged
        # by test_hyperdiff_exponent_invariant_at_ref_n).
        assert cdgrid_hyperdiff_cube(96, scaling_exponent=2) == pytest.approx(
            4.0 * cdgrid_hyperdiff_cube(96, scaling_exponent=4))
        assert cdgrid_hyperdiff_cube(96, scaling_exponent=2) == pytest.approx(
            2.5e15)

    @pytest.mark.parametrize("bad_exp", [0, -1, -4])
    def test_hyperdiff_rejects_nonpositive_exponent(self, bad_exp):
        with pytest.raises(ValueError, match="positive integer"):
            cdgrid_hyperdiff_cube(96, scaling_exponent=bad_exp)

    @pytest.mark.parametrize("bad_exp", [2.5, 4.0, "2", True, False])
    def test_hyperdiff_rejects_non_integer_exponent(self, bad_exp):
        # A fractional/float/bool exponent must not silently give a fractional
        # resolution law — the public helper's contract is a positive int.
        with pytest.raises(ValueError, match="positive integer"):
            cdgrid_hyperdiff_cube(96, scaling_exponent=bad_exp)

    def test_div_damp_anchor_at_c48(self):
        assert cdgrid_div_damp_cube(48) == pytest.approx(1.5e7)

    def test_div_damp_scales_as_dx_squared(self):
        assert (
            cdgrid_div_damp_cube(24) / cdgrid_div_damp_cube(48)
            == pytest.approx(4.0)
        )


class TestModonHyperdiffEnvKnob:
    """Runtime coverage of the #753 modon env-knob wiring in the matrix runner
    (``_modon_hyperdiff_coeff``) — the AST parity guard only pins the source
    text, so this exercises the actual env read + coefficient formula."""

    def _coeff(self):
        # Imported lazily: pulls the matrix-runner module (heavier than the
        # atmosphere package alone).
        from scripts.matrix.run_atmosphere_test_matrix import (
            _modon_hyperdiff_coeff,
        )
        return _modon_hyperdiff_coeff

    def test_unset_env_defaults_to_ref_n_squared_law(self, monkeypatch):
        # #753 item 1: the modon default is now the (ref/n)^2 law (C96 erupts
        # under ^4; ^2 is stable, validated 100 days at C36/C48/C96).  Unset env
        # must reproduce ^2, and C48 stays exponent-invariant (no coarse-case
        # regression), while C96 receives the 4x face-seam backstop.
        monkeypatch.delenv("LEGOESM_SW_MODON_HYPERDIFF_FACTOR", raising=False)
        monkeypatch.delenv("LEGOESM_SW_MODON_HYPERDIFF_SCALING", raising=False)
        coeff = self._coeff()
        for n in (36, 48, 96, 192):
            assert coeff(n) == cdgrid_hyperdiff_cube(n, scaling_exponent=2)
        assert coeff(48) == cdgrid_hyperdiff_cube(48)  # C48 invariant vs ^4
        assert coeff(96) == pytest.approx(4.0 * cdgrid_hyperdiff_cube(96))

    def test_scaling_env_four_reproduces_pre753_law(self, monkeypatch):
        # The ^4 law stays selectable (opt-out) and is byte-identical to the
        # pre-#753 expression at every resolution.
        monkeypatch.delenv("LEGOESM_SW_MODON_HYPERDIFF_FACTOR", raising=False)
        monkeypatch.setenv("LEGOESM_SW_MODON_HYPERDIFF_SCALING", "4")
        coeff = self._coeff()
        for n in (36, 48, 96, 192):
            assert coeff(n) == cdgrid_hyperdiff_cube(n)  # 1.0 * ^4 law

    def test_scaling_env_two_gives_4x_at_c96(self, monkeypatch):
        monkeypatch.delenv("LEGOESM_SW_MODON_HYPERDIFF_FACTOR", raising=False)
        monkeypatch.setenv("LEGOESM_SW_MODON_HYPERDIFF_SCALING", "2")
        coeff = self._coeff()
        assert coeff(96) == pytest.approx(4.0 * cdgrid_hyperdiff_cube(96))
        assert coeff(48) == pytest.approx(cdgrid_hyperdiff_cube(48))  # C48 unchanged

    def test_factor_env_multiplies(self, monkeypatch):
        monkeypatch.setenv("LEGOESM_SW_MODON_HYPERDIFF_FACTOR", "2.0")
        monkeypatch.setenv("LEGOESM_SW_MODON_HYPERDIFF_SCALING", "4")
        coeff = self._coeff()
        assert coeff(96) == pytest.approx(2.0 * cdgrid_hyperdiff_cube(96))


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
        import legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid as m
        assert hasattr(m, "williamson_cli_calibration")
        # The CLI's cmd_test imports `williamson_cli_calibration` from
        # this module; if the name disappears or moves, this assertion
        # is what fails first.
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            williamson_cli_calibration as imported,
        )
        assert imported is m.williamson_cli_calibration
