"""Deploy a saved correction-campaign output as a production turbulence override.

Covers ``legoesm.training.deploy_correction`` — the loader that closes the
"campaign output JSON → production AMIP/CMIP run" loop (iter 57): it must turn
both campaign output shapes (single ``--diagnosis-method`` and multi
``--coefficients``) into a per-column ``TurbulenceConfig`` that ``ExperimentConfig.
validate_strict`` accepts, and raise LOUDLY on a malformed / non-CLUBB output.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.turbulence.config import (
    CLUBBLiteConfig,
    TurbulenceConfig,
)
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
)
from legoesm.training.deploy_correction import (
    corrected_clubb_config,
    corrected_turbulence_override,
)


def _single_output(field="C_K", vals=(0.5, 0.6, 0.7)):
    # Mirrors the single-coefficient campaign JSON (top-level field array + extras).
    return {field: list(vals), "biases": [1.0, 0.5], "accepted": [True],
            "step_fractions": [1.0]}


def _multi_output(fields=None):
    fields = fields or {"clubb_lite_C_K": [0.5, 0.6],
                        "clubb_lite_Pr_t": [0.4, 0.5],
                        "clubb_lite_C_eps": [0.1, 0.2]}
    return {"coefficients": ["C_K", "Pr_t", "C_eps"], "fields": fields,
            "biases": [1.0, 0.5]}


# --------------------------------------------------------------------------- #
# corrected_clubb_config — both shapes
# --------------------------------------------------------------------------- #
def test_single_output_builds_per_column_field():
    cfg = corrected_clubb_config(_single_output("Pr_t", (0.3, 0.4, 0.5)))
    assert isinstance(cfg, CLUBBLiteConfig)
    np.testing.assert_allclose(np.asarray(cfg.Pr_t), [0.3, 0.4, 0.5])
    # The untouched fields keep their scalar defaults (byte-identical to prod).
    assert float(cfg.C_K) == float(CLUBBLiteConfig().C_K)
    assert float(cfg.C_eps) == float(CLUBBLiteConfig().C_eps)


def test_multi_output_builds_all_three_fields():
    cfg = corrected_clubb_config(_multi_output())
    np.testing.assert_allclose(np.asarray(cfg.C_K), [0.5, 0.6])
    np.testing.assert_allclose(np.asarray(cfg.Pr_t), [0.4, 0.5])
    np.testing.assert_allclose(np.asarray(cfg.C_eps), [0.1, 0.2])


def test_multi_output_partial_fields():
    cfg = corrected_clubb_config(_multi_output({"clubb_lite_C_K": [0.5, 0.6]}))
    np.testing.assert_allclose(np.asarray(cfg.C_K), [0.5, 0.6])
    assert float(cfg.Pr_t) == float(CLUBBLiteConfig().Pr_t)


# --------------------------------------------------------------------------- #
# corrected_turbulence_override — dict / path / file
# --------------------------------------------------------------------------- #
def test_override_from_dict():
    over = corrected_turbulence_override(_multi_output())
    assert isinstance(over, TurbulenceConfig)
    assert over.scheme == "clubb_lite"
    np.testing.assert_allclose(np.asarray(over.clubb_lite.C_K), [0.5, 0.6])


def test_override_from_path_and_file(tmp_path):
    p = tmp_path / "campaign_out.json"
    p.write_text(json.dumps(_single_output("C_eps", (0.1, 0.2))))
    over_path = corrected_turbulence_override(str(p))
    np.testing.assert_allclose(np.asarray(over_path.clubb_lite.C_eps), [0.1, 0.2])
    with open(p) as f:
        over_file = corrected_turbulence_override(f)
    np.testing.assert_allclose(np.asarray(over_file.clubb_lite.C_eps), [0.1, 0.2])


# --------------------------------------------------------------------------- #
# The whole point: the loaded override is accepted by validate_strict
# --------------------------------------------------------------------------- #
def test_loaded_override_passes_validate_strict():
    over = corrected_turbulence_override(_multi_output())
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite", turbulence_override=over,
    )
    cfg.validate_strict()  # must not raise — override is a clubb_lite TurbulenceConfig


def test_scheme_mismatch_rejected_by_validate_strict():
    # Deploying a clubb_lite override onto a non-clubb base must fail loudly.
    over = corrected_turbulence_override(_multi_output())
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="none", turbulence_override=over,
    )
    with pytest.raises((ValueError, AssertionError)):
        cfg.validate_strict()


# --------------------------------------------------------------------------- #
# Dispatch hardening — malformed / non-CLUBB output raises
# --------------------------------------------------------------------------- #
def test_unknown_promotion_key_raises():
    with pytest.raises(ValueError, match="unsupported promotion_key"):
        corrected_clubb_config({"fields": {"gray_tau_equator": [1.0, 2.0]}})


def test_empty_output_raises():
    # No "fields" + no C_K/Pr_t/C_eps → single branch, zero recognized fields.
    with pytest.raises(ValueError, match="exactly one"):
        corrected_clubb_config({"biases": [1.0]})


def test_single_output_with_two_fields_raises():
    with pytest.raises(ValueError, match="exactly one"):
        corrected_clubb_config({"C_K": [0.5], "Pr_t": [0.4]})


def test_empty_fields_dict_raises():
    with pytest.raises(ValueError, match="no corrected CLUBB"):
        corrected_clubb_config({"fields": {}})


def test_non_dict_toplevel_raises():
    with pytest.raises(ValueError, match="must be a JSON object"):
        corrected_clubb_config([0.5, 0.6])


def test_non_dict_fields_raises():
    with pytest.raises(ValueError, match="'fields' must be a dict"):
        corrected_clubb_config({"fields": None})


def test_empty_array_raises():
    with pytest.raises(ValueError, match="is empty"):
        corrected_clubb_config({"C_K": []})


def test_non_finite_array_raises():
    with pytest.raises(ValueError, match="non-finite"):
        corrected_clubb_config({"C_K": [0.5, float("nan")]})


def test_rank2_array_raises():
    with pytest.raises(ValueError, match="1-D per-column"):
        corrected_clubb_config({"C_K": [[0.5, 0.6]]})


def test_arrays_are_jax_for_differentiable_deploy():
    over = corrected_turbulence_override(_multi_output())
    assert isinstance(over.clubb_lite.C_K, jnp.ndarray)


def test_integer_json_coerced_to_float_for_autodiff():
    # A hand-edited integer-valued coefficient must NOT enter as a non-diff int.
    cfg = corrected_clubb_config({"C_K": [1, 2]})
    assert jnp.issubdtype(cfg.C_K.dtype, jnp.floating)
    np.testing.assert_allclose(np.asarray(cfg.C_K), [1.0, 2.0])
