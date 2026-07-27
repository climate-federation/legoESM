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


class TestMorrisonScalarOverlay:
    """The five morrison_* ExperimentConfig scalars, unwired since their
    introduction (flag-reachability audit cause 1/2, 2026-07-25): the overlay
    in ``_resolve_microphysics`` must map each onto its MorrisonConfig leaf,
    stay byte-identical at defaults, and fail loudly on non-Morrison schemes.
    ``morrison_dep_coeff``'s declaration was corrected 1e-8 -> 1e-3 (it was
    1e5 OFF the leaf default while unwired) BEFORE wiring, so rest state is a
    no-op."""

    def _leaf(self, **kw):
        from legoesm.driver.config import ExperimentConfig
        from legoesm.driver.physics_pipeline import _resolve_microphysics
        return _resolve_microphysics(
            ExperimentConfig(microphysics="morrison", **kw))[1]

    def test_defaults_are_byte_identical(self):
        from legoesm.atmosphere.physics.microphysics.config import (
            MorrisonConfig,
        )
        assert self._leaf() == MorrisonConfig()

    def test_declared_default_matches_the_leaf(self):
        """The cause-2 defect: 1e-8 declared vs 1e-3 real.  Locked equal so
        the overlay can never silently re-tune a run at rest."""
        from legoesm.atmosphere.physics.microphysics.config import (
            MorrisonConfig,
        )
        from legoesm.driver.config import ExperimentConfig
        e = ExperimentConfig()
        m = MorrisonConfig()
        for exp, leaf in (("morrison_bergeron_rate", "bergeron_rate"),
                          ("morrison_rime_coeff", "rime_coeff"),
                          ("morrison_dep_coeff", "dep_coeff"),
                          ("morrison_agg_coeff", "agg_coeff"),
                          ("morrison_k_au", "k_au"),
                          ("morrison_fall_a_i", "fall_a_i"),
                          ("morrison_ice_snow_d_auto", "ice_snow_d_auto"),
                          ("morrison_hom_ice_nuc_N", "hom_ice_nuc_N")):
            assert getattr(e, exp) == getattr(m, leaf), (exp, leaf)

    @pytest.mark.parametrize("exp,leaf,val", [
        ("morrison_bergeron_rate", "bergeron_rate", 2.5e-3),
        ("morrison_rime_coeff", "rime_coeff", 0.5),
        ("morrison_dep_coeff", "dep_coeff", 3e-4),
        ("morrison_agg_coeff", "agg_coeff", 5e-3),
        ("morrison_k_au", "k_au", 1.2e3),
        ("morrison_fall_a_i", "fall_a_i", 2100.0),
        ("morrison_ice_snow_d_auto", "ice_snow_d_auto", 100.0e-6),
        ("morrison_hom_ice_nuc_N", "hom_ice_nuc_N", 1.0e5),
    ])
    def test_each_scalar_reaches_its_leaf(self, exp, leaf, val):
        if exp == "morrison_hom_ice_nuc_N":
            # the leaf is only live inside the hom branch; pair the override
            # with the flag exactly as validate_strict requires.
            got = self._leaf(homogeneous_ice_nucleation=True, **{exp: val})
            assert getattr(got, leaf) == val
            return
        got = self._leaf(**{exp: val})
        assert getattr(got, leaf) == val, (
            f"{exp} did not reach MorrisonConfig.{leaf} — the audit's "
            "cause-1 wiring gap has reopened")

    @pytest.mark.parametrize("scheme", ["kessler", "thompson", "p3"])
    def test_non_morrison_scheme_raises_on_override(self, scheme):
        """HARD scheme gate, not field presence: Thompson carries all five
        same-named leaves and P3 four, with DIFFERENT defaults (P3
        rime_coeff=0.5 vs 1.0) — a presence-keyed overlay silently retuned
        them (codex 2026-07-26 Critical)."""
        from legoesm.driver.config import ExperimentConfig
        from legoesm.driver.physics_pipeline import _resolve_microphysics
        cfg = ExperimentConfig(microphysics=scheme,
                               morrison_dep_coeff=3e-4)
        with pytest.raises(ValueError, match="morrison"):
            _resolve_microphysics(cfg)

    @pytest.mark.parametrize("scheme", ["kessler", "thompson", "p3"])
    def test_non_morrison_scheme_default_is_untouched(self, scheme):
        """Fresh defaults on ANY other scheme must resolve byte-identical to
        the bare scheme config — the codex-reproduced defect was a default
        P3 run whose rime_coeff changed 0.5 -> 1.0."""
        from legoesm.atmosphere.physics.microphysics.config import (
            MicrophysicsConfig,
        )
        from legoesm.driver.config import ExperimentConfig
        from legoesm.driver.physics_pipeline import _resolve_microphysics
        _fn, leaf = _resolve_microphysics(
            ExperimentConfig(microphysics=scheme))
        bare = getattr(MicrophysicsConfig(scheme=scheme), scheme)
        assert leaf == bare, (
            f"default {scheme} run altered by the morrison overlay")

    def test_legacy_serialized_1e8_is_migrated_on_load(self):
        """An old JSON carries the then-inert morrison_dep_coeff=1e-8; loading
        it verbatim would now retune deposition 1e5 down (or raise on
        non-Morrison).  The loader migrates the OLD DEFAULT to the new one —
        preserving what the old run actually did (nothing)."""
        from legoesm.driver.config import (
            ExperimentConfig, experiment_config_from_dict,
        )
        d = ExperimentConfig(microphysics="morrison")._asdict()
        d["morrison_dep_coeff"] = 1e-8
        d = {k: v for k, v in d.items() if not hasattr(v, "_asdict")}
        with pytest.warns(UserWarning, match="Migrating to 1e-3"):
            got = experiment_config_from_dict(d)
        assert got.morrison_dep_coeff == 1e-3

    def test_explicit_non_default_value_is_kept_on_load(self):
        from legoesm.driver.config import experiment_config_from_dict
        got = experiment_config_from_dict(
            {"microphysics": "morrison", "morrison_dep_coeff": 5e-4})
        assert got.morrison_dep_coeff == 5e-4

    def test_validate_strict_rejects_nan_and_out_of_bounds(self):
        from legoesm.driver.config import ExperimentConfig
        for bad in (float("nan"), 1e-8, 1.0):
            with pytest.raises(Exception, match="morrison_dep_coeff"):
                ExperimentConfig(microphysics="morrison",
                                 morrison_dep_coeff=bad).validate_strict()

    def test_flavor_sam_reaches_the_leaf(self):
        got = self._leaf(morrison_flavor="sam")
        assert got.morrison_flavor == "sam"

    def test_flavor_default_mg_is_byte_identical(self):
        from legoesm.atmosphere.physics.microphysics.config import (
            MorrisonConfig,
        )
        assert self._leaf(morrison_flavor="mg") == MorrisonConfig()

    def test_flavor_on_non_morrison_raises(self):
        from legoesm.driver.config import ExperimentConfig
        from legoesm.driver.physics_pipeline import _resolve_microphysics
        with pytest.raises(ValueError, match="morrison_flavor"):
            ExperimentConfig(microphysics="thompson",
                             morrison_flavor="sam").validate_strict()
        with pytest.raises(ValueError, match="morrison_flavor"):
            _resolve_microphysics(ExperimentConfig(
                microphysics="thompson", morrison_flavor="sam"))

    def test_flavor_membership_validated(self):
        from legoesm.driver.config import ExperimentConfig
        with pytest.raises(ValueError, match="morrison_flavor"):
            ExperimentConfig(microphysics="morrison",
                             morrison_flavor="gcm").validate_strict()

    def test_hom_nuc_N_override_requires_the_flag(self):
        """morrison_hom_ice_nuc_N is read only inside the hom-nucleation
        branch — with the flag off the override would be silently inert
        (the exact class the scheme gate closes), so validate_strict must
        refuse the combination and accept it once the flag is on."""
        from legoesm.driver.config import ExperimentConfig
        with pytest.raises(ValueError, match="morrison_hom_ice_nuc_N"):
            ExperimentConfig(microphysics="morrison",
                             morrison_hom_ice_nuc_N=1e5).validate_strict()
        ExperimentConfig(microphysics="morrison",
                         homogeneous_ice_nucleation=True,
                         morrison_hom_ice_nuc_N=1e5).validate_strict()

    def test_validate_strict_bounds_cover_the_ice_knobs(self):
        from legoesm.driver.config import ExperimentConfig
        for field, bad in (("morrison_fall_a_i", 10.0),
                           ("morrison_ice_snow_d_auto", 5e-3),
                           ("morrison_hom_ice_nuc_N", 1e9)):
            with pytest.raises(Exception, match=field):
                kw = {field: bad}
                if field == "morrison_hom_ice_nuc_N":
                    kw["homogeneous_ice_nucleation"] = True
                ExperimentConfig(microphysics="morrison",
                                 **kw).validate_strict()

    def test_params_map_carries_all_five(self):
        """--params routing (codex finding: the map was never extended, so
        calibration files could not use the wiring).  End-to-end threading of
        every map entry is machine-verified by the canonical
        ``test_atm_scalar_map_is_pipeline_threaded``; this pins membership."""
        from legoesm.driver.run_config_yaml import _ATM_SCALAR_PARAM_MAP as M
        for leaf, flat in (("bergeron_rate", "morrison_bergeron_rate"),
                           ("rime_coeff", "morrison_rime_coeff"),
                           ("dep_coeff", "morrison_dep_coeff"),
                           ("agg_coeff", "morrison_agg_coeff"),
                           ("k_au", "morrison_k_au"),
                           ("fall_a_i", "morrison_fall_a_i"),
                           ("ice_snow_d_auto", "morrison_ice_snow_d_auto"),
                           ("hom_ice_nuc_N", "morrison_hom_ice_nuc_N")):
            assert M[f"atm.micro.MorrisonConfig.{leaf}"] == flat

    def test_tuning_catalog_agrees_with_the_leaf_default(self):
        """tuning.py advertised default=1e-8 range 3e-9..3e-8 — five orders
        below the real leaf.  Locked to bracket the true default."""
        from legoesm.atmosphere.physics.microphysics.config import (
            MorrisonConfig,
        )
        from legoesm.tuning import TUNING_PARAMETERS
        t = TUNING_PARAMETERS["morrison_dep_coeff"]
        d = MorrisonConfig().dep_coeff
        assert t.default == d
        assert t.min_val < d < t.max_val


class TestRound2Findings:
    """Codex morrison-wiring round 2 (review-2.md): the four STILL-OPEN."""

    def test_none_scheme_refuses_touched_scalar_at_config_time(self):
        """Item 3: microphysics='none' early-returns before either lane's
        overlay, so the resolver gate never fires — validate_strict must
        refuse instead (the silent-drop class this wiring exists to close)."""
        from legoesm.driver.config import ExperimentConfig
        cfg = ExperimentConfig(microphysics="none", morrison_dep_coeff=3e-4)
        with pytest.raises(ValueError, match="microphysics='morrison'"):
            cfg.validate_strict()

    def test_none_scheme_defaults_accepted(self):
        from legoesm.driver.config import ExperimentConfig
        ExperimentConfig(microphysics="none").validate_strict()

    def test_fresh_explicit_1e8_is_refused_by_bounds(self):
        """Item 2: 1e-8 is OUTSIDE [1e-4, 1e-2], so no modern config can mean
        it — which is what makes the load-time migration coherent (any config
        carrying it must be legacy)."""
        from legoesm.driver.config import ExperimentConfig
        with pytest.raises(ValueError, match="morrison_dep_coeff"):
            ExperimentConfig(microphysics="morrison",
                             morrison_dep_coeff=1e-8).validate_strict()

    def test_sub_tolerance_perturbation_is_uniformly_the_default(self):
        """Item 4: touched-ness and application share ONE rel_tol=1e-6, so a
        below-noise perturbation resolves byte-identical to the default —
        never half-recognised."""
        from legoesm.atmosphere.physics.microphysics.config import (
            MorrisonConfig,
        )
        from legoesm.driver.config import ExperimentConfig
        from legoesm.driver.physics_pipeline import _resolve_microphysics
        got = _resolve_microphysics(ExperimentConfig(
            microphysics="morrison",
            morrison_dep_coeff=1e-3 * (1 + 4.7e-8)))[1]   # f32 storage noise
        assert got == MorrisonConfig()

    def test_above_tolerance_retune_is_applied(self):
        from legoesm.driver.config import ExperimentConfig
        from legoesm.driver.physics_pipeline import _resolve_microphysics
        got = _resolve_microphysics(ExperimentConfig(
            microphysics="morrison",
            morrison_dep_coeff=1e-3 * 1.01))[1]
        assert got.dep_coeff == pytest.approx(1.01e-3)

    def test_tuning_catalog_within_strict_bounds(self):
        """Item 6: the catalog advertised rime_coeff to 5.0 and agg_coeff to
        1e-5 while validate_strict enforces [0,2] / [1e-4,1e-2] — advertised
        proposals would be rejected.  Catalog must sit inside the gates."""
        from legoesm.tuning import TUNING_PARAMETERS
        gates = {"morrison_bergeron_rate": (1e-4, 1e-2),
                 "morrison_rime_coeff": (0.0, 2.0),
                 "morrison_dep_coeff": (1e-4, 1e-2),
                 "morrison_agg_coeff": (1e-4, 1e-2),
                 "morrison_k_au": (50.0, 5000.0)}
        for name, (lo, hi) in gates.items():
            t = TUNING_PARAMETERS[name]
            assert lo <= t.min_val <= t.max_val <= hi, (
                f"{name}: catalog [{t.min_val}, {t.max_val}] outside "
                f"validate_strict [{lo}, {hi}]")


class TestRound3Findings:
    """Codex morrison-wiring round 3: strict/resolver tolerance unity and the
    schema-version migration gate."""

    def test_strict_accepts_f32_noise_on_any_scheme(self):
        """Item 4: a float32-noise perturbation must be 'default' in BOTH the
        resolver AND validate_strict — previously strict rejected it on
        kessler/none while the resolver would have applied nothing."""
        from legoesm.driver.config import ExperimentConfig
        noisy = 1e-3 * (1 + 4.7e-8)
        for scheme in ("kessler", "none", "morrison"):
            ExperimentConfig(microphysics=scheme,
                             morrison_dep_coeff=noisy).validate_strict()

    def test_marked_modern_dict_is_not_migrated(self):
        """Item 2: a dict carrying config_schema_version >= 2 with 1e-8 is a
        DELIBERATE modern value — loaded verbatim, then refused by strict
        bounds, never silently rewritten."""
        import warnings as _w

        from legoesm.driver.config import experiment_config_from_dict
        d = {"microphysics": "morrison", "morrison_dep_coeff": 1e-8,
             "config_schema_version": 2}
        with _w.catch_warnings():
            _w.simplefilter("error")          # any warning -> failure
            got = experiment_config_from_dict(d)
        assert got.morrison_dep_coeff == 1e-8
        with pytest.raises(ValueError, match="morrison_dep_coeff"):
            got.validate_strict()

    def test_unmarked_legacy_dict_still_migrates(self):
        from legoesm.driver.config import experiment_config_from_dict
        with pytest.warns(UserWarning, match="Migrating to 1e-3"):
            got = experiment_config_from_dict(
                {"microphysics": "morrison", "morrison_dep_coeff": 1e-8})
        assert got.morrison_dep_coeff == 1e-3

    def test_codec_round_trip_carries_the_marker(self):
        from legoesm.driver.config import (
            ExperimentConfig, config_to_dict, experiment_config_from_dict,
        )
        d = config_to_dict(ExperimentConfig(microphysics="morrison"))
        assert d["config_schema_version"] == 2
        got = experiment_config_from_dict(d)
        assert got.morrison_dep_coeff == 1e-3
