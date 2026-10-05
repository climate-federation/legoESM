"""Water-budget closure guard in ``build_physics_pipeline``.

Root cause of the 2026-06-15 coarse-CMIP6 bug: ``microphysics='none'`` with
active convection (and, worse, an active cloud scheme).  Convection detrains
condensate into ``q_c`` and surface precipitation is owned by microphysics, so
``microphysics='none'`` gives identically-zero ``pr`` AND an unbounded ``q_c``
water trap that corrupts the cloud-radiation optics (TOA fluxes diverged).

The pipeline must WARN loudly (idealized dry-convection runs are legitimate, so
not a hard error).  These tests pin that the warning fires exactly when the
water budget is open, and stays silent when it is closed.
"""
from __future__ import annotations

import logging

import jax.numpy as jnp

from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.driver.config import ExperimentConfig
from legoesm.grids.cubed_sphere import create_cubed_sphere

_NLEV = 6


class _Sigma:
    sigma_full = jnp.linspace(0.1, 0.95, _NLEV)
    sigma_half = jnp.linspace(0.05, 1.0, _NLEV + 1)
    dsigma = jnp.diff(jnp.linspace(0.05, 1.0, _NLEV + 1))

    def pressure_at_full(self, p_s):
        return p_s[..., None] * self.sigma_full

    def pressure_at_half(self, p_s):
        return p_s[..., None] * self.sigma_half

    def layer_thickness_dp(self, p_s):
        return p_s[..., None] * self.dsigma


def _build_and_capture(caplog, *, convection, microphysics, cloud_scheme="none"):
    grid = create_cubed_sphere(4)
    cfg = ExperimentConfig(
        radiation="gray", convection=convection,
        microphysics=microphysics, cloud_scheme=cloud_scheme,
    )
    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="legoesm.driver.physics_pipeline"
    ):
        build_physics_pipeline(grid, _Sigma(), cfg)
    return caplog.text


def test_warns_for_detraining_scheme_without_microphysics(caplog):
    # A TRUE-detrainment scheme (mass_flux, detrains_to_cloud=True) feeds q_c,
    # whose only sink is microphysics -> with micro=none it traps water.
    txt = _build_and_capture(caplog, convection="mass_flux", microphysics="none")
    assert "water trap" in txt
    assert "microphysics" in txt


def test_silent_for_adjustment_scheme_without_microphysics(caplog):
    # TOA-drift fix: sbm (Betts-Miller ADJUSTMENT, detrains_to_cloud=False)
    # precipitates its convective drying DIRECTLY, so it closes the water budget
    # without microphysics and must NOT trip the q_c-trap guard.
    txt = _build_and_capture(caplog, convection="sbm", microphysics="none")
    assert "water trap" not in txt


def test_silent_when_microphysics_enabled(caplog):
    txt = _build_and_capture(caplog, convection="mass_flux", microphysics="kessler")
    assert "water trap" not in txt


def test_silent_when_convection_disabled(caplog):
    # No convection -> no detrained condensate -> no trap even with micro=none.
    txt = _build_and_capture(caplog, convection="none", microphysics="none")
    assert "water trap" not in txt


def test_warning_flags_cloud_radiation_corruption(caplog):
    # The dangerous variant: an active cloud scheme reads the unbounded q_c
    # from a TRUE-detrainment scheme with no microphysics sink.
    txt = _build_and_capture(
        caplog, convection="mass_flux", microphysics="none",
        cloud_scheme="sundqvist",
    )
    assert "water trap" in txt
    assert "cloud-radiation optics" in txt
