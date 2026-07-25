"""Homogeneous (Koop/Ren-MacKenzie) cirrus ice nucleation: reachability.

The physics was already implemented in ``MorrisonConfig`` and AD-safe, but
defaulted OFF with NO route from ``ExperimentConfig`` or the CLI — so no
production run could enable it.  With it off, M2005 deposition (rate
proportional to N_i^(2/3)) has no ice to grow on in ice-free cirrus air and
nothing caps RH over ice: the century reached RH_ice = 288% at 228 K /
222 hPa before detonating (2026-07-25 autopsy).  These pin the wiring and
the threshold behaviour that makes that impossible.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig,
    MorrisonConfig,
    apply_microphysics_experiment_flags,
)


class TestApplier:
    def test_flag_reaches_morrison(self):
        got = apply_microphysics_experiment_flags(
            MorrisonConfig(), "morrison", homogeneous_ice_nucleation=True)
        assert got.homogeneous_ice_nucleation is True

    def test_default_off_is_byte_identical(self):
        base = MorrisonConfig()
        got = apply_microphysics_experiment_flags(base, "morrison")
        assert got == base

    def test_unsupported_scheme_raises_loudly(self):
        """Dispatch hardening: a scheme without the field must FAIL, never
        silently ignore the request."""
        with pytest.raises(ValueError, match="homogeneous_ice_nucleation"):
            apply_microphysics_experiment_flags(
                KesslerConfig(), "kessler", homogeneous_ice_nucleation=True)

    def test_composes_with_the_other_flags(self):
        got = apply_microphysics_experiment_flags(
            MorrisonConfig(), "morrison",
            homogeneous_ice_nucleation=True,
            hard_saturation_adjustment=True,
            hard_sat_adjust_threshold=1.05,
        )
        assert got.homogeneous_ice_nucleation is True
        assert got.hard_saturation_adjustment is True
        assert got.hard_sat_adjust_threshold == 1.05


class TestThresholdPhysics:
    """S_hom(T) is the whole point: it must sit in the Koop range at cirrus
    temperatures, well BELOW the 2.88 the century actually reached."""

    def _s_hom(self, cfg, T):
        return float(np.clip(cfg.koop_s_hom_a - cfg.koop_s_hom_b * T,
                             cfg.koop_s_hom_min, cfg.koop_s_hom_max))

    @pytest.mark.parametrize("T,lo,hi", [
        (185.0, 1.55, 1.70),   # Koop ~1.64
        (200.0, 1.50, 1.65),   # ~1.58
        (228.0, 1.40, 1.55),   # the detonation cell: ~1.47
        (235.0, 1.40, 1.50),   # ~1.44
    ])
    def test_threshold_in_koop_range(self, T, lo, hi):
        assert lo <= self._s_hom(MorrisonConfig(), T) <= hi

    def test_threshold_far_below_the_observed_runaway(self):
        """The century hit RH_ice 2.88 at 228 K; the threshold must be well
        under that, or enabling this changes nothing."""
        assert self._s_hom(MorrisonConfig(), 228.0) < 1.6

    def test_threshold_decreases_with_temperature(self):
        cfg = MorrisonConfig()
        vals = [self._s_hom(cfg, T) for T in (190.0, 205.0, 220.0, 233.0)]
        assert vals == sorted(vals, reverse=True), vals

    def test_cold_gate_excludes_the_mixed_phase(self):
        """Homogeneous freezing of aqueous haze needs T <~ -38 C; the gate
        must not fire in the warm mixed phase."""
        assert MorrisonConfig().hom_freeze_T_max <= 240.0


class TestPipelineReachability:
    """The applier being correct is NOT enough -- the LANES must call it.
    ``_resolve_microphysics`` nested the call inside a hard-saturation-override
    guard whose two scalars both default to None, so the flag was SILENTLY
    INERT on the FV lane: the helper tests above all passed while no FV run
    could ever enable the physics."""

    def _fv(self, **kw):
        from legoesm.driver.config import ExperimentConfig
        from legoesm.driver.physics_pipeline import _resolve_microphysics
        cfg = ExperimentConfig(microphysics="morrison", **kw)
        _fn, micro = _resolve_microphysics(cfg)
        return micro

    def test_fv_lane_threads_the_flag_at_default_settings(self):
        got = self._fv(homogeneous_ice_nucleation=True)
        assert got.homogeneous_ice_nucleation is True, (
            "flag lost between ExperimentConfig and MorrisonConfig on the FV "
            "lane (the hard-sat-override guard swallowed it)")

    def test_fv_lane_default_off(self):
        assert self._fv().homogeneous_ice_nucleation is False

    def test_fv_lane_still_threads_the_hard_sat_overrides(self):
        got = self._fv(homogeneous_ice_nucleation=True,
                       hard_saturation_adjustment=True,
                       hard_sat_adjust_threshold=1.05)
        assert got.homogeneous_ice_nucleation is True
        assert got.hard_sat_adjust_threshold == 1.05


class TestValidateStrict:
    """Only MorrisonConfig carries the field, and microphysics='none'
    early-returns before either lane's applier -- so a bad combination must be
    refused at CONFIG time, not silently ignored (or raised deep in a run)."""

    def _cfg(self, micro):
        from legoesm.driver.config import ExperimentConfig
        return ExperimentConfig(microphysics=micro,
                                homogeneous_ice_nucleation=True)

    def test_morrison_is_accepted(self):
        self._cfg("morrison").validate_strict()

    @pytest.mark.parametrize("micro", ["kessler", "thompson", "none"])
    def test_non_morrison_is_refused(self, micro):
        with pytest.raises(Exception, match="homogeneous_ice_nucleation"):
            self._cfg(micro).validate_strict()

    def test_default_off_accepts_any_scheme(self):
        from legoesm.driver.config import ExperimentConfig
        ExperimentConfig(microphysics="kessler").validate_strict()


def test_cli_and_config_round_trip():
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args,
    )
    parser = build_arg_parser()
    off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert off.homogeneous_ice_nucleation is False

    on = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--homogeneous-ice-nucleation",
    ]), parser))
    assert on.homogeneous_ice_nucleation is True
    on.validate_strict()


def test_scheme_actually_consumes_supersaturation():
    """End-to-end on the leaf: at cirrus conditions ABOVE the threshold the
    flag must produce a vapour sink the off-state lacks.  Without this the
    wiring could be inert and every other test would still pass.

    NOTE this previously called ``morrison_microphysics`` with a keyword form
    that does not exist and SKIPPED on the resulting TypeError, i.e. the
    commit's physics was untested.  The real signature is positional:
    ``(T, q_v, HydrometeorState, p_full, p_half, rho, dz, dt, config)``.
    The full leaf battery (thresholds, number budget, boundedness, multi-step
    relaxation, conservation, AD) lives in
    ``test_homogeneous_ice_nucleation_leaf.py``.
    """
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics,
    )
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from legoesm.thermo import saturation_mixing_ratio_ice

    T = jnp.full((1, 1), 220.0)                # cirrus, below the cold gate
    p = jnp.full((1, 1), 20000.0)              # 200 hPa
    q_v = 2.5 * saturation_mixing_ratio_ice(T, p)   # RH_ice 250% >> S_hom
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
                          N_c=z, N_r=z, N_i=z)
    out = {}
    for label, flag in (("off", False), ("on", True)):
        cfg = MorrisonConfig()._replace(homogeneous_ice_nucleation=flag)
        out[label] = morrison_microphysics(
            T, q_v, hm, p, jnp.full((1, 2), 20000.0),
            p / (constants.R_d * T), jnp.full((1, 1), 400.0), 240.0, cfg)
    dqv_off = float(jnp.sum(out["off"].dq_v_dt))
    dqv_on = float(jnp.sum(out["on"].dq_v_dt))
    assert dqv_off < 0.0                       # Cooper seed mass alone
    assert dqv_on < 2.0 * dqv_off, (dqv_off, dqv_on)
