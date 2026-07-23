"""convection_config_for: tuned ExperimentConfig fields reach combined lanes.

The MPAS/spectral lanes previously built ``ConvectionConfig(scheme=...)``
with bare defaults, so ``bechtold_*`` / ``sbm_*`` /
``convective_precip_efficiency`` tuning silently never reached them (the
2026-07-23 stack-switch discovery). These tests pin the shared resolver
wrapper.
"""

from __future__ import annotations

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
)
from legoesm.driver.physics_pipeline import convection_config_for


def _cfg(**kw):
    return ExperimentConfig(
        grid=GridConfig(resolution=8, nlev=8),
        dycore=DycoreConfig(dt=600.0),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        **kw,
    )


def test_bechtold_tuning_reaches_leaf():
    cc = convection_config_for(_cfg(
        convection="bechtold",
        bechtold_cape_threshold=65.0,
        convective_precip_efficiency=0.8,
    ))
    assert cc.scheme == "bechtold"
    assert cc.bechtold.cape_threshold == 65.0
    assert cc.bechtold.precip_efficiency == 0.8


def test_sbm_tuning_reaches_leaf():
    cc = convection_config_for(_cfg(convection="sbm", sbm_tau_c=3600.0,
                                    sbm_RH_ref=0.75))
    assert cc.scheme == "sbm"
    assert cc.sbm.tau_c == 3600.0
    assert cc.sbm.rh_ref == 0.75


def test_none_passthrough():
    cc = convection_config_for(_cfg(convection="none"))
    assert cc.scheme == "none"
