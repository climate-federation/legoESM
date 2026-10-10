"""Spectral lane surface albedo (review 2026-10-10 F25).

The spectral full-physics loop handed radiation no ``sfc_albedo``, so every
column -- sea ice and land included -- reflected with the scalar config albedo.
It now sends the same ocean/ice/land blend as the MPAS loop.  Checked at the
radiation backend call itself: the blend is built with one value per column
and arrives there as the surface-albedo override (it was None).  (A state
EFFECT is not a usable check here: under gray radiation with a prescribed
surface temperature the atmosphere's heating does not depend on the surface
albedo -- measured bit-identical with a 0.9 albedo.)
Gray radiation + Louis so the loop takes the full-physics branch.
"""
from __future__ import annotations

import numpy as np
import pytest
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver

RES, NLEV, DT = 8, 8, 600.0
FOUR_STEPS_DAYS = 2401.0 / 86400.0


def test_blended_albedo_reaches_spectral_radiation(monkeypatch, tmp_path):
    import legoesm.atmosphere.physics.radiation.integration as rad
    import legoesm.forcing.surface_utils as su
    real_blend, real_backend = su.blended_surface_albedo, rad._call_radiation_backend
    blends, overrides = [], []

    def blend_spy(sic, *a, **k):
        out = real_blend(sic, *a, **k)
        blends.append(np.asarray(out).shape)
        return out

    def backend_spy(*a, **k):
        alb = k.get("sfc_albedo_override")
        overrides.append(None if alb is None else tuple(np.shape(alb)))
        return real_backend(*a, **k)

    monkeypatch.setattr(su, "blended_surface_albedo", blend_spy)
    monkeypatch.setattr(rad, "_call_radiation_backend", backend_spy)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="gaussian", resolution=RES, nlev=NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="spectral",
                            dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=FOUR_STEPS_DAYS, dataset="analytical", radiation="gray",
        convection="none", turbulence="louis", precision="fp64",
        distributed=False)
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    assert d.run() == "COMPLETED"
    ncol = int(d.grid.n_lat) * int(d.grid.n_lon)
    assert blends and set(blends) == {(ncol,)}, blends
    assert overrides and set(overrides) == {(ncol,)}, overrides


def test_dynamic_albedo_refused_on_spectral(tmp_path):
    """The zenith ocean albedo is not wired here: refused, not silently fixed."""
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="gaussian", resolution=RES, nlev=NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="spectral",
                            dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=FOUR_STEPS_DAYS, dataset="analytical", radiation="gray",
        convection="none", turbulence="louis", precision="fp64",
        distributed=False, dynamic_albedo=True)
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    with pytest.raises(ValueError, match="dynamic_albedo"):
        d.run()
