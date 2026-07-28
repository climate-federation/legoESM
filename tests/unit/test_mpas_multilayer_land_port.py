"""Multilayer (Richards) land tile on the MPAS lane (tasks/mpas_land_port.md).

Covers the port seams end-to-end, offline + portable (loaders monkeypatched,
no NetCDF/network):

1. ``_setup_multilayer_land`` builds per-cell land columns from
   ``VoronoiMesh.latCell/lonCell`` (the old crash: no ``lat/lat2d``).
2. The MPAS driver loop steps the tile (soil state EVOLVES over a short run)
   with the skin-T feedback through ``forcing['T_sfc']``, and the atmosphere
   stays finite.
3. The surface-flux export chain carries the new DOWNWELLING fields
   (``HydrostaticTendencies.sw_down_sfc/lw_down_sfc`` — the land forcing).
4. Checkpoint round-trip: ``land_ml_*`` ride the MPAS npz and restore
   bit-exact (fail-loud contract of #730/#770).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver

MPAS_RES, MPAS_NLEV, DT = 3, 8, 300.0   # icosahedral level 3 = 642 cells
FOUR_STEPS_DAYS = 1201.0 / 86400.0      # int(1201/300) = 4 steps
TWO_STEPS_DAYS = 601.0 / 86400.0


def _fake_surface_map(path, lat_deg, lon_deg):
    """Uniform-loam, all-bare-soil CLM map sized to the requested columns
    (same synthetic map as test_multilayer_land_driver)."""
    n = int(np.asarray(lat_deg).size)
    pft = np.zeros((n, 17)); pft[:, 0] = 1.0
    o = np.ones(n)
    return dict(
        pft_fractions=jnp.asarray(pft),
        theta_wp=jnp.asarray(0.12 * o), theta_fc=jnp.asarray(0.30 * o),
        glacier_frac=jnp.asarray(np.zeros(n)),
        pct_sand=jnp.asarray(40.0 * o), pct_clay=jnp.asarray(20.0 * o),
        theta_r=jnp.asarray(0.05 * o), theta_sat=jnp.asarray(0.45 * o),
        alpha_vg=jnp.asarray(2.0 * o), n_vg=jnp.asarray(1.4 * o),
        K_sat=jnp.asarray(1.0e-5 * o),
    )


def _patch_land_loaders(monkeypatch):
    import legoesm.land.clm_surface_map as clm
    import legoesm.grids.topography as topo
    monkeypatch.setattr(clm, "download_clm_surfdata", lambda *a, **k: "synthetic")
    monkeypatch.setattr(clm, "load_clm_surface", _fake_surface_map)
    # Half-land everywhere; grid_lat (GridProtocol) so the patch is
    # layout-agnostic (VoronoiMesh has no ``.lat``).
    monkeypatch.setattr(
        topo, "load_land_fraction",
        lambda grid, path, *a, **k: jnp.full(
            jnp.asarray(grid.grid_lat).shape, 0.5))


def _build_driver(tmpdir: str, days: float, *, turbulence: str = "none",
                  beta_soil: bool = False,
                  use_multilayer: bool = True) -> ModelDriver:
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=MPAS_RES,
                        nlev=MPAS_NLEV, vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=1),
        days=days, dataset="analytical", radiation="gray",
        convection="none", turbulence=turbulence, precision="fp64",
        distributed=False,
        land_mask_path="synthetic.nc",      # truthy -> land block (loader patched)
        use_multilayer_land=use_multilayer,
        multilayer_n_layers=4, multilayer_soil_depth=2.0,
        mpas_land_beta_soil=beta_soil,
    )
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


def test_setup_builds_land_columns_on_voronoi(monkeypatch, tmp_path):
    """latCell/lonCell drive the column build: (nCells, n_layers) soil state,
    land_ml_lat == mesh.latCell (radians)."""
    from legoesm.land.state import MultiLayerLandState
    _patch_land_loaders(monkeypatch)
    d = _build_driver(str(tmp_path), FOUR_STEPS_DAYS)

    ncell = int(np.asarray(d.grid.latCell).size)
    st = d._land_ml_state
    assert isinstance(st, MultiLayerLandState)
    assert st.T_soil.shape == (ncell, 4)
    np.testing.assert_allclose(
        np.asarray(d.physics.land_ml_lat), np.asarray(d.grid.latCell))
    assert np.all((np.asarray(st.T_soil) > 180.0)
                  & (np.asarray(st.T_soil) < 340.0))


def test_run_steps_land_and_stays_finite(monkeypatch, tmp_path):
    """A short MPAS run advances the soil columns (state EVOLVES — the tile is
    stepped, not carried) and the atmosphere stays finite with the skin-T
    feedback active."""
    _patch_land_loaders(monkeypatch)
    d = _build_driver(str(tmp_path), FOUR_STEPS_DAYS)
    T_soil_0 = np.asarray(d._land_ml_state.T_soil).copy()

    assert d.run() == "COMPLETED"

    T_soil_1 = np.asarray(d._land_ml_state.T_soil)
    assert np.all(np.isfinite(np.asarray(d.state.T.data)))
    assert np.all(np.isfinite(T_soil_1))
    # The tile stepped: surface soil layer moved somewhere on the mesh.
    assert np.max(np.abs(T_soil_1[:, 0] - T_soil_0[:, 0])) > 0.0


def test_tendencies_carry_downwelling_fields():
    """HydrostaticTendencies grew sw_down_sfc/lw_down_sfc (None defaults,
    appended last): the land-forcing export contract.  A field rename or
    reorder breaks the MPAS marshal silently — this pins it."""
    from legoesm.core.state import HydrostaticTendencies
    fields = HydrostaticTendencies._fields
    assert fields.index("sw_down_sfc") == len(fields) - 2
    assert fields.index("lw_down_sfc") == len(fields) - 1
    # Defaults are None so every positional constructor stays valid.
    assert HydrostaticTendencies._field_defaults["sw_down_sfc"] is None
    assert HydrostaticTendencies._field_defaults["lw_down_sfc"] is None


def test_traced_beta_soil_reaches_turbulence(monkeypatch, tmp_path):
    """#1312 phase 2b: mpas_land_beta_soil threads a per-cell root-zone
    beta_soil into the turbulence surface humidity (forcing['beta_land']).

    The synthetic map cold-starts the soil BETWEEN wilting and field capacity
    (theta ~ 0.5*theta_sat = 0.225, wp 0.12, fc 0.30) so beta_soil is strictly
    inside (beta_min, 1) — the throttle must CHANGE the integrated state
    relative to the same run without the flag (which runs the land fraction
    saturated, beta = 1)."""
    _patch_land_loaders(monkeypatch)
    d_off = _build_driver(str(tmp_path / "off"), FOUR_STEPS_DAYS,
                          turbulence="louis")
    assert d_off.run() == "COMPLETED"

    d_on = _build_driver(str(tmp_path / "on"), FOUR_STEPS_DAYS,
                         turbulence="louis", beta_soil=True)
    assert d_on.run() == "COMPLETED"

    # The seeded traced beta exists, has cell shape, and is a REAL throttle
    # (strictly below 1 somewhere: the soil is between wp and fc).
    from legoesm.land.multilayer_land import land_tile_beta_soil
    beta = np.asarray(land_tile_beta_soil(
        d_on._land_ml_state.theta_soil, d_on.physics.land_ml_cfg,
        d_on.physics.land_ml_params))
    ncell = int(np.asarray(d_on.grid.latCell).size)
    assert beta.shape == (ncell,)
    assert np.all((beta >= 0.0) & (beta <= 1.0))
    assert float(beta.min()) < 0.999, "soil between wp and fc must throttle"

    dq = np.max(np.abs(np.asarray(d_on.state.tracers["q_v"].data)
                       - np.asarray(d_off.state.tracers["q_v"].data)))
    assert np.isfinite(np.asarray(d_on.state.T.data)).all()
    assert dq > 0.0, (
        "mpas_land_beta_soil=True left q_v bit-identical to the saturated "
        "run — forcing['beta_land'] is not reaching the turbulence "
        "surface flux")


def test_beta_soil_without_multilayer_land_is_refused(monkeypatch, tmp_path):
    """Inert-corner rejection: the flag without the multilayer land has no
    soil moisture to derive beta from — refused FAIL-EARLY at config
    validation (driver construction), not silently ignored."""
    _patch_land_loaders(monkeypatch)
    with pytest.raises(ValueError, match="mpas_land_beta_soil"):
        _build_driver(str(tmp_path), FOUR_STEPS_DAYS, turbulence="louis",
                      beta_soil=True, use_multilayer=False)


def test_checkpoint_roundtrips_land_state(monkeypatch, tmp_path):
    """land_ml_* ride the MPAS checkpoint npz and restore bit-exact into a
    fresh driver (the #730 fail-loud restore contract, now on the MPAS lane)."""
    _patch_land_loaders(monkeypatch)
    outA = str(tmp_path / "a")
    dA = _build_driver(outA, TWO_STEPS_DAYS)
    assert dA.run() == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(outA, "checkpoint_day_*.npz")))[-1]

    dB = _build_driver(str(tmp_path / "b"), TWO_STEPS_DAYS)
    dB.load_checkpoint(ckpt)
    for f in dA._land_ml_state._fields:
        a = getattr(dA._land_ml_state, f)
        b = getattr(dB._land_ml_state, f)
        if a is None:
            assert b is None, f"land_ml.{f}: None saved but restored non-None"
            continue
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"land_ml.{f}: checkpoint round-trip not bit-exact")
