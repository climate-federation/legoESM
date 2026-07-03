"""End-to-end: an offline ERA5-built AMIP forcing drives a real ModelDriver run (iter 420).

iter-419 added ``build_era5_amip_forcing`` (the FORCING-side twin of
``open_local_era5_dataset``) and validated it through ``load_amip_forcing`` in ISOLATION.
But no test ran an actual ``ModelDriver.setup().run()`` with a custom forcing — so the
composition (build forcing -> wire into ExperimentConfig -> run a model integration ->
extract the column state) was unproven.  This locks that path with a DIFFERENTIAL that
proves the run CONSUMES the prescribed SST (not merely loads the forcing object): two AMIP
runs that differ ONLY by a uniform +15 K SST bump must yield a warmer boundary layer
(``get_sst_sic`` is a closure over the forcing, so a finite/gradiented value there does not
prove the segment step read it — the standard-atmosphere IC already carries an equator-pole
gradient — codex-review iter 420).  CI-portable (synthetic archive, coarse grid, gray
radiation, 1 day).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._offline_era5_rda import write_forcing_archive

#: This file's forcing-archive source grid (16x32) — passed to the shared writer.
_FORCING_GRID = {"nlat": 16, "nlon": 32}


def _run_amip(fcfg, *, radiation="gray", nlev=5, resolution=8, rad_update_steps=1):
    """Wire an AMIPForcingConfig onto an ExperimentConfig via the SHARED field map
    (``apply_amip_forcing_to_config`` — the same the campaign applies) and run a tiny
    hydrostatic AMIP ModelDriver for one day with the given radiation scheme."""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.driver.model_driver import ModelDriver

    from scripts.data.load_local_era5 import apply_amip_forcing_to_config

    base = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=1), radiation=radiation, days=1,
        rad_update_steps=rad_update_steps)
    cfg = apply_amip_forcing_to_config(base, fcfg)
    driver = ModelDriver(cfg)
    driver.setup()
    driver.run()
    return driver


@pytest.mark.slow
def test_offline_era5_forcing_drives_amip_modeldriver_run(tmp_path):
    from legoesm.training.run_to_column_mean import amip_column_state

    from legoesm import constants
    from scripts.data.load_local_era5 import build_era5_amip_forcing

    # --- Base run (the gradient SST). ---
    write_forcing_archive(tmp_path, **_FORCING_GRID)
    fcfg = build_era5_amip_forcing(
        str(tmp_path), "20200101", str(tmp_path / "amip_forcing.nc"), hour_stride=24)
    assert fcfg.dataset == "custom" and fcfg.sst_var == "SSTK" and fcfg.sic_var == "CI"
    driver_a = _run_amip(fcfg)

    # The prescribed ERA5 SST loaded + regridded: finite, the equator->pole gradient survived
    # the regrid to the (coarser) model grid, and the seawater freeze floor was applied.
    sst_applied = np.asarray(driver_a.get_sst_sic(0.0)[0])
    assert bool(np.all(np.isfinite(sst_applied)))
    assert float(sst_applied.max()) > float(sst_applied.min())        # gradient, not constant
    assert float(sst_applied.min()) >= constants.T_freeze_ocean       # freeze floor applied
    assert float(sst_applied.max()) <= 301.0
    # The AMIP column state (atm + prescribed SST) extracts finite — the compare-side input.
    col_a = amip_column_state(driver_a, day=0.0)
    assert bool(np.all(np.isfinite(np.asarray(col_a.T))))
    assert bool(np.all(np.isfinite(np.asarray(col_a.q_v))))
    t_bot_a = float(np.asarray(col_a.T)[..., -1].mean())

    # --- DIFFERENTIAL run: the SAME setup with a uniform +15 K SST bump. ---
    # If the run merely LOADED the forcing (without the segment step reading it), the two
    # boundary layers would be identical; a warmer ocean MUST warm the BL — proving the SST
    # is consumed.  The smoke measured ~+6.8 K mean bottom-T for +15 K SST; assert >> noise.
    write_forcing_archive(tmp_path, sst_add=15.0, **_FORCING_GRID)
    fcfg_warm = build_era5_amip_forcing(
        str(tmp_path), "20200101", str(tmp_path / "amip_forcing_warm.nc"), hour_stride=24)
    driver_b = _run_amip(fcfg_warm)
    t_bot_b = float(np.asarray(amip_column_state(driver_b, day=0.0).T)[..., -1].mean())

    assert t_bot_b > t_bot_a + 2.0, (
        f"a +15 K SST forcing should warm the boundary layer, but mean bottom-level T went "
        f"{t_bot_a:.3f} -> {t_bot_b:.3f} K (delta {t_bot_b - t_bot_a:.3f}); the run is not "
        "consuming the prescribed SST.")


@pytest.mark.slow
def test_offline_era5_forcing_drives_rrtmgp_amip_run(tmp_path):
    """The offline ERA5 forcing drives a REALISTIC-radiation (rrtmgp) AMIP run (iter 423).

    The 419-422 arc validated the offline forcing with GRAY radiation; the empirical demo
    the operator runs at HPC scale uses rrtmgp (the realism lever, iter 418), and that
    composition (rrtmgp + a custom ERA5 SST forcing + ModelDriver AMIP) was never exercised
    — a launch-time rrtmgp failure would waste the HPC run.  This de-risks it: rrtmgp needs
    ~20 levels (Held-Suarez rrtmgp uses nlev=20) and a coarse radiation cadence
    (``rad_update_steps``) keeps the cost down.  Asserts the run COMPLETES with a finite
    state, that rrtmgp (NOT a silent gray fallback) ran and applied a NON-zero radiation
    tendency, and the prescribed-SST gradient reached the model — NOT a physical profile (a
    1-day coarse run is far from equilibrium)."""
    from legoesm.training.run_to_column_mean import amip_column_state

    from scripts.data.load_local_era5 import build_era5_amip_forcing

    write_forcing_archive(tmp_path, **_FORCING_GRID)
    fcfg = build_era5_amip_forcing(
        str(tmp_path), "20200101", str(tmp_path / "amip_forcing.nc"), hour_stride=24)
    driver = _run_amip(
        fcfg, radiation="rrtmgp", nlev=20, resolution=4, rad_update_steps=12)

    col = amip_column_state(driver, day=0.0)
    t_col = np.asarray(col.T)
    assert bool(np.all(np.isfinite(t_col)))                  # rrtmgp produced a finite state
    assert bool(np.all(np.isfinite(np.asarray(col.q_v))))
    assert float(t_col.max() - t_col.min()) > 2.0            # not a degenerate constant column

    # rrtmgp ACTUALLY ran (the dispatch has no silent gray fallback — radiation="rrtmgp"
    # selects rrtmgp) AND applied a non-zero radiation tendency (codex-review iter 423): a
    # regression that accidentally selected gray, or a no-op radiation, fails here.
    assert driver.config.radiation == "rrtmgp"
    rad_tend = (driver._carry_aux or {}).get("held_dT_rad")
    assert rad_tend is not None, "rrtmgp wrote no radiation tendency to carry_aux"
    assert float(np.max(np.abs(np.asarray(rad_tend)))) > 0.0, \
        "rrtmgp radiation tendency is identically zero — a no-op or wrong scheme"

    sst_applied = np.asarray(driver.get_sst_sic(0.0)[0])
    assert bool(np.all(np.isfinite(sst_applied)))
    assert float(sst_applied.max() - sst_applied.min()) > 10.0   # the forcing reached it
