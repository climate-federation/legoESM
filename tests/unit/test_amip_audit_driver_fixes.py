"""Driver-level regressions from the 2026-07-17 AMIP audit.

Covers:
- preset-dataset forcing configs forwarding ``sic_path`` and the variable-name
  overrides (they used to be silently dropped, reading SIC from the SST file);
- MPAS ``ic='default'`` passing phis into ``held_suarez_init_mpas`` so surface
  pressure is hydrostatically reduced over topography (it used to patch phis in
  afterwards, leaving p_s flat over terrain — a startup pressure shock).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402


def test_preset_dataset_forwards_sic_path_and_var_overrides(monkeypatch):
    """``--dataset hadisst --sic-path ice.nc --sst-var tosbcs`` must reach the
    loader config; unset overrides ("" per run_amip) keep the preset's values."""
    import legoesm.forcing.amip as amip_mod

    captured = {}

    def _fake_load(config, grid, start_year=None):
        captured["config"] = config
        captured["start_year"] = start_year
        return SimpleNamespace(times=None, sst=None, sic=None, config=config)

    monkeypatch.setattr(amip_mod, "load_amip_forcing", _fake_load)
    preset = amip_mod.get_amip_preset("hadisst")

    cfg = SimpleNamespace(
        dataset="hadisst",
        grid=SimpleNamespace(grid_type="cubed_sphere"),
        forcing_path="/tmp/hadisst_sst.nc",
        T_ice=constants.T_freeze_ocean,
        sst_offset=preset.sst_offset,
        sic_scale=preset.sic_scale,
        sic_path="/tmp/hadisst_ice.nc",
        sst_var="tosbcs",
        sic_var="",
        time_var="",
        lat_var="",
        lon_var="",
        start_year=1979,
    )
    fake_self = SimpleNamespace(
        config=cfg,
        grid=SimpleNamespace(grid_lat=np.zeros((2, 2))),
    )

    ModelDriver._create_forcing(fake_self)

    fc = captured["config"]
    assert fc.sic_path == "/tmp/hadisst_ice.nc"      # was silently dropped
    assert fc.sst_var == "tosbcs"                    # explicit override wins
    assert fc.sic_var == preset.sic_var              # unset keeps preset value
    assert fc.time_var == preset.time_var
    assert fc.lat_var == preset.lat_var
    assert fc.lon_var == preset.lon_var
    assert fc.path == "/tmp/hadisst_sst.nc"
    assert captured["start_year"] == 1979


def test_mpas_default_ic_reduces_p_s_over_topography(tmp_path, monkeypatch):
    """MPAS ``ic='default'`` with real topography must construct the state with
    phis (p_s = p_ref·exp(-phis/(R_d·T_init))), not patch phis in after the
    fact with a flat p_s.  The analytic ``gaussian`` topography builder has no
    Voronoi support, so inject a synthetic phis via ``_create_topography`` —
    the injection point real file topography also feeds (``_phis_data``)."""

    def _synthetic_topo(self) -> None:
        n_cells = self.grid.nCells
        z_s = np.zeros(n_cells)
        z_s[: n_cells // 2] = np.linspace(0.0, 3000.0, n_cells // 2)
        self._phis_data = jnp.asarray(constants.g * z_s)
        self._f_land = jnp.asarray((z_s > 0).astype(np.float64))

    monkeypatch.setattr(ModelDriver, "_create_topography", _synthetic_topo)

    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=5),
        dycore=DycoreConfig(dt=1.0, discretization="mpas"),
        dataset="analytical",
        radiation="none",
        days=1.0 / 86400.0,
    )
    driver = ModelDriver(cfg, output_dir=str(tmp_path / "run"))
    driver.setup()

    phis = np.asarray(driver.state.phis.data, dtype=np.float64)
    p_s = np.asarray(driver.state.p_s.data, dtype=np.float64)
    assert np.any(phis > 0), "gaussian topography should be non-flat on MPAS"

    expected = constants.p_ref * np.exp(
        -phis / (constants.R_d * cfg.T_init)
    )
    np.testing.assert_allclose(p_s, expected, rtol=1e-6)
    # The old bug: p_s identically p_ref while phis carries mountains.
    assert p_s[np.argmax(phis)] < constants.p_ref * 0.999
