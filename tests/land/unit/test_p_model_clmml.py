"""CLM-ML P-model switches: config validation and backend injection plumbing.

Config-level: the switch/typo/provider-conflict matrix on
``CLMMLCanopyConfig.validate``.  Backend-level: the per-column canopy-top
Vcmax25 / Jmax-ratio / Medlyn-g1 injection points exist with the documented
signatures and default to the per-PFT tables when the arrays are ``None``
(inspected structurally; the full-column behavioural identity runs in the
CLM-ML integration suite, which needs the warm backend state).
"""

from __future__ import annotations

import inspect

import pytest

from legoesm.land.canopy.config import CLMMLCanopyConfig


def test_validate_switch_matrix():
    with pytest.raises(ValueError, match="capacity_scheme"):
        CLMMLCanopyConfig(capacity_scheme="pmodel").validate()
    with pytest.raises(ValueError, match="g1_source"):
        CLMMLCanopyConfig(g1_source="xi").validate()
    with pytest.raises(ValueError, match="medlyn"):
        CLMMLCanopyConfig(stomatal_model="wue", g1_source="p_model").validate()
    with pytest.raises(ValueError, match="two .*providers|providers"):
        CLMMLCanopyConfig(capacity_scheme="p_model",
                          vcmax25_override=55.0).validate()
    with pytest.raises(ValueError, match="pmodel_rjv25"):
        CLMMLCanopyConfig(pmodel_rjv25=True).validate()
    CLMMLCanopyConfig(stomatal_model="medlyn", capacity_scheme="p_model",
                      g1_source="p_model", pmodel_rjv25=True).validate()


def test_backend_injection_points_exist():
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import (
        MLCanopyNitrogenProfileMod as npm)
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import (
        MLLeafPhotosynthesisMod as lpm)
    sig_n = inspect.signature(npm.CanopyNitrogenProfile)
    assert "vcmax25top_col" in sig_n.parameters
    assert "jv_ratio_col" in sig_n.parameters
    assert sig_n.parameters["vcmax25top_col"].default is None
    sig_l = inspect.signature(lpm.LeafPhotosynthesis)
    assert "g1_med_col" in sig_l.parameters
    assert sig_l.parameters["g1_med_col"].default is None


def test_interface_guards():
    import jax.numpy as jnp
    from legoesm.land.canopy import clm_ml_interface as iface
    sig = inspect.signature(iface.compute_clm_ml_canopy_fluxes)
    for nm in ("vcmax25_col_jax", "g1_med_col_jax", "jv_ratio_col_jax",
               "tacclim_col_jax"):
        assert nm in sig.parameters and sig.parameters[nm].default is None
    # The guard source enforces: provider conflict, gs_type mismatch, shape.
    src = inspect.getsource(iface.compute_clm_ml_canopy_fluxes)
    assert "two providers" in src
    assert "silently inert" in src
    assert "shape (ncol,)" in src


def test_multilayer_arm_selects_pathway_per_column():
    """C3/C4 pathway selection is a static host mask (c3psn) feeding a
    per-column where() between the coordination and rpmodel-c4 optima."""
    src = open(
        "packages/land/legoesm/land/multilayer_land.py").read()
    assert "acclimated_capacities_c4" in src
    assert "c3psn" in src
    assert "_c3_mask" in src
