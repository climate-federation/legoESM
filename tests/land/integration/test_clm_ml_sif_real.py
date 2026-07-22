"""SIF extraction against the REAL clm-ml-jax ``mlcanopy_type`` (importorskip).

Skips when the optional ``clm-ml-jax`` dependency is absent.  Constructs a real
``create_mlcanopy`` container, fills ONE sunlit element with a known
electron-transport rate, and asserts ``_extract_clm_ml_sif`` reproduces the
big-leaf ``leaf_sif`` — validating the extractor against the ACTUAL installed
NamedTuple (real field names, real spval padding, real shapes), not a mock.

This is the cheap glue check (no model timestep).  The full end-to-end diurnal
run lives in the offline driver; see ``scripts`` / the SIF campaign notes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

mlt = pytest.importorskip("legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType")

from legoesm.land.canopy.sif import (  # noqa: E402
    SIFConfig, actual_electron_transport, leaf_sif,
)
from legoesm.land.canopy.clm_ml_interface import _extract_clm_ml_sif  # noqa: E402

CFG = SIFConfig()


def test_extractor_on_real_mlcanopy_type_matches_big_leaf():
    # Real container built the interface way: create_mlcanopy(1, ncol) (begp=1)
    # -> patch-dim size ncol+1, column 0 at index 1, index 0 is the unused pad;
    # the layer/leaf axes are 1-based with a spval pad at index 0.
    ml = mlt.create_mlcanopy(begp=1, endp=1)
    assert {"je_leaf", "apar_leaf", "dpai_profile", "fracsun_profile"} <= set(ml._fields)
    assert ml.je_leaf.shape[0] == 2  # 1-based patch axis, size endp+1

    An, Ci, gs, ap = 15.0, 280.0, 43.0, 800.0
    je = float(actual_electron_transport(jnp.asarray(An), jnp.asarray(Ci), jnp.asarray(gs)))

    # Fill column 0 (patch index 1), layer 1, both leaves; shaded (il=2) -> zero
    # leaf area via fracsun=1 so only the sunlit element contributes.  Every
    # other element keeps apar=spval (masked) or zero leaf area, so it drops out.
    ml = ml._replace(
        je_leaf=ml.je_leaf.at[1, 1, 1].set(je).at[1, 1, 2].set(je),
        apar_leaf=ml.apar_leaf.at[1, 1, 1].set(ap).at[1, 1, 2].set(ap),
        dpai_profile=ml.dpai_profile.at[1, 1].set(1.0),
        fracsun_profile=ml.fracsun_profile.at[1, 1].set(1.0),
    )

    out = _extract_clm_ml_sif(ml, ncol=1, sif_cfg=CFG)
    assert out is not None and out.shape == (1,)
    ref = float(leaf_sif(jnp.asarray(An), jnp.asarray(Ci),
                         jnp.asarray(gs), jnp.asarray(ap), CFG))
    assert float(out[0]) == pytest.approx(ref, rel=1e-6)
