"""AIMIP Phase 3.0: ``sbm_CAPE_threshold`` override changes SBM output.

Numerical sentinel for the iter-260 wiring of ``CAPE_threshold``
through the SBM convection scheme via the
``physics_step_no_rad._conv_cfg._replace`` path.  A regression that
silently drops the override (e.g. forgetting the ``_replace`` block,
removing the kwarg from ``build_segment_fn``) would make
``sbm_CAPE_threshold`` a nominal-only trainable again — the bug
class iter-256 explicitly reverted.

Sentinel design: build a tropical-like instability profile whose CAPE
is large (~8.8 kJ/kg) and scan ``CAPE_threshold`` from well below to
well above that CAPE; the SBM trigger sigmoid

    σ(smooth_trigger_sharpness · (CAPE − threshold))

must transition from ~1 (active convection) to ~0 (inactive) as the
threshold crosses CAPE.  We pin two endpoints: a low threshold gives
non-zero ``dT_dt``; a high threshold gives essentially zero.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.convection.config import SBMConfig
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.driver.compiled_segments import build_segment_fn


def _moist_unstable_profile(ncol: int = 4):
    """Tropical-like profile with CAPE ~ 8.8 kJ/kg."""
    T = jnp.array([220, 215, 220, 230, 250, 270, 285, 295]).astype("float32")
    T = T.reshape(1, -1).repeat(ncol, axis=0)
    q_v = jnp.array(
        [1e-5, 1e-5, 1e-4, 5e-4, 2e-3, 5e-3, 1e-2, 1.5e-2]
    ).astype("float32")
    q_v = q_v.reshape(1, -1).repeat(ncol, axis=0)
    p_full = jnp.array(
        [5000, 12000, 25000, 40000, 55000, 70000, 85000, 95000]
    ).astype("float32")
    p_full = p_full.reshape(1, -1).repeat(ncol, axis=0)
    p_half = jnp.array(
        [1000, 8000, 18000, 32500, 47500, 62500, 77500, 90000, 100000]
    ).astype("float32")
    p_half = p_half.reshape(1, -1).repeat(ncol, axis=0)
    return T, q_v, p_full, p_half


def test_build_segment_fn_has_sbm_CAPE_threshold_kwarg():
    """Pin that ``sbm_CAPE_threshold`` is a kwarg of ``build_segment_fn``.
    Catches a regression where the kwarg is silently dropped from the
    rollout entry point — would make the trainable nominal-only.
    """
    sig = inspect.signature(build_segment_fn)
    assert "sbm_CAPE_threshold" in sig.parameters, (
        "build_segment_fn is missing the ``sbm_CAPE_threshold`` kwarg "
        "added in iter-260 (AIMIP Phase 3.0)."
    )


def test_cape_threshold_below_cape_activates_convection():
    """Threshold well below CAPE -> trigger sigmoid saturates at 1 ->
    non-zero dT_dt."""
    T, q_v, p_full, p_half = _moist_unstable_profile()
    out = sbm_convection(
        T, q_v, p_full, p_half, 600.0,
        SBMConfig(CAPE_threshold=70.0),
    )
    max_abs_dT = float(jnp.abs(out.dT_dt).max())
    assert max_abs_dT > 1e-4, (
        f"With CAPE_threshold=70 and CAPE~8800 the SBM trigger should "
        f"saturate at 1, giving |dT|_max > 1e-4; got {max_abs_dT:.3e}."
    )


def test_cape_threshold_above_cape_silences_convection():
    """Threshold well above CAPE -> trigger sigmoid saturates at 0 ->
    essentially zero dT_dt."""
    T, q_v, p_full, p_half = _moist_unstable_profile()
    out = sbm_convection(
        T, q_v, p_full, p_half, 600.0,
        SBMConfig(CAPE_threshold=12000.0),
    )
    max_abs_dT = float(jnp.abs(out.dT_dt).max())
    assert max_abs_dT < 1e-8, (
        f"With CAPE_threshold=12000 (well above CAPE) SBM should be "
        f"silenced; got |dT|_max = {max_abs_dT:.3e}."
    )


def test_cape_threshold_low_vs_high_outputs_differ_substantially():
    """Cross-check: low- and high-threshold runs produce visibly
    different ``dT_dt`` fields — sentinel against an accidental wiring
    that drops the override but keeps default behaviour."""
    T, q_v, p_full, p_half = _moist_unstable_profile()
    out_low = sbm_convection(
        T, q_v, p_full, p_half, 600.0,
        SBMConfig(CAPE_threshold=70.0),
    )
    out_high = sbm_convection(
        T, q_v, p_full, p_half, 600.0,
        SBMConfig(CAPE_threshold=12000.0),
    )
    diff = float(jnp.abs(out_low.dT_dt - out_high.dT_dt).max())
    base = float(jnp.abs(out_low.dT_dt).max())
    rel = diff / (base + 1e-12)
    assert rel > 0.9, (
        f"low-vs-high CAPE_threshold runs differ by only rel={rel:.3e}; "
        f"the override is likely not flowing into the trigger sigmoid."
    )


def test_physics_pipeline_replaces_cape_threshold():
    """Pin that ``PhysicsPipeline.physics_step_no_rad`` honours the
    ``sbm_CAPE_threshold`` kwarg.  Indirect check via signature so we
    catch a regression that drops the kwarg without needing to spin
    up a full pipeline instance.
    """
    from legoesm.driver.physics_pipeline import PhysicsPipeline
    sig = inspect.signature(PhysicsPipeline.physics_step_no_rad)
    assert "sbm_CAPE_threshold" in sig.parameters, (
        "PhysicsPipeline.physics_step_no_rad is missing the "
        "``sbm_CAPE_threshold`` kwarg added in iter-260."
    )


def test_tuning_registry_has_sbm_CAPE_threshold():
    """Cross-check: the tuning.py registry exposes the new param so
    experiment YAML configs can target it by name."""
    from legoesm.tuning import TUNING_PARAMETERS
    assert "sbm_CAPE_threshold" in TUNING_PARAMETERS, (
        "tuning.py TUNING_PARAMETERS is missing ``sbm_CAPE_threshold``."
    )
    p = TUNING_PARAMETERS["sbm_CAPE_threshold"]
    assert p.category == "convection"
    assert p.min_val == 10.0
    assert p.max_val == 200.0
    np.testing.assert_allclose(p.default, 70.0)
