"""Phase-1 coupled 3D-ocean wiring (docs/ocean/coupled_3d_ocean_plan.md): the coupled
driver can step the prognostic LatLonCGridOceanModel (ocean_mode='dynamic') on a
SHARED lat-lon grid with the atmosphere, instead of only a thermodynamic slab.

These tests pin the config dispatch + the build path (a real coupled segment is
exercised by the integration smoke `scripts/tmp/_ocean3d_smoke.sbatch`, which is
too heavy/JIT-bound for unit CI).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import pytest

from legoesm.driver.coupled_config import CoupledConfig
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.simple_ocean import SimpleOceanConfig


def test_config_holds_dynamic_ocean():
    """CoupledConfig accepts ocean_mode='dynamic' + a LatLonCGridOceanConfig +
    the 3D-ocean knobs (the union widening); slab default is unchanged."""
    slab = CoupledConfig()
    assert slab.ocean_mode == "slab"
    assert isinstance(slab.ocean_config, SimpleOceanConfig)
    # 3D-ocean knobs default but present
    assert slab.ocean_nlev == 20 and slab.ocean_dt_s == 300.0
    dyn = CoupledConfig(
        ocean_mode="dynamic", ocean_config=LatLonCGridOceanConfig.from_flat(),
        ocean_nlev=8, ocean_dt_s=300.0,
    )
    assert dyn.ocean_mode == "dynamic"
    assert isinstance(dyn.ocean_config, LatLonCGridOceanConfig)


def test_config_holds_woa_restoring_taus():
    """CoupledConfig exposes the WOA-restoring timescales; default 0 = off."""
    c = CoupledConfig()
    assert c.ocean_restore_sst_tau_days == 0.0
    assert c.ocean_restore_sss_tau_days == 0.0
    r = CoupledConfig(ocean_restore_sst_tau_days=30.0,
                      ocean_restore_sss_tau_days=60.0)
    assert r.ocean_restore_sst_tau_days == 30.0
    assert r.ocean_restore_sss_tau_days == 60.0


def test_apply_ocean_restoring_noop_when_off_or_no_target():
    """_apply_ocean_restoring is a no-op (returns same state) when both taus are
    0 OR no WOA target was loaded; relaxes the surface when configured."""
    from types import SimpleNamespace
    import jax.numpy as jnp
    import numpy as np
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=2.0, S_uniform=34.0)
    T_tgt = jnp.full((grid.n_lat, grid.n_lon), 25.0)
    S_tgt = jnp.full((grid.n_lat, grid.n_lon), 36.0)

    # tau=0 => no-op
    fake = SimpleNamespace(
        coupled_cfg=CoupledConfig(ocean_mode="dynamic"),
        _ocean_T_target=T_tgt, _ocean_S_target=S_tgt, _ocean_state=state)
    CoupledESMDriver._apply_ocean_restoring(fake, 3600.0)
    assert fake._ocean_state is state

    # tau>0 but no target => no-op
    fake2 = SimpleNamespace(
        coupled_cfg=CoupledConfig(ocean_mode="dynamic",
                                  ocean_restore_sst_tau_days=30.0),
        _ocean_T_target=None, _ocean_S_target=None, _ocean_state=state)
    CoupledESMDriver._apply_ocean_restoring(fake2, 3600.0)
    assert fake2._ocean_state is state

    # tau>0 + target => surface warms toward 25 C
    T0 = np.asarray(state.T.data).copy()
    fake3 = SimpleNamespace(
        coupled_cfg=CoupledConfig(ocean_mode="dynamic",
                                  ocean_restore_sst_tau_days=10.0),
        _ocean_T_target=T_tgt, _ocean_S_target=S_tgt, _ocean_state=state)
    CoupledESMDriver._apply_ocean_restoring(fake3, 3600.0)
    Tn = np.asarray(fake3._ocean_state.T.data)
    assert np.any(Tn[..., 0] > T0[..., 0])           # surface relaxed up
    np.testing.assert_array_equal(Tn[..., 1:], T0[..., 1:])  # subsurface intact


def test_init_ocean_rejects_unknown_mode():
    """_init_ocean dispatch raises on an unknown ocean_mode (no silent
    else->slab; CLAUDE.md dispatch-hardening)."""
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    # Build a driver far enough to call _init_ocean with a bogus mode.  Use a
    # tiny lat-lon atm; _init_ocean runs after the atm grid exists.
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(dt=60.0, model_type="hydrostatic",
                            discretization="latlon_cgrid"),
        radiation="gray", convection="none", days=1,
    )
    drv = CoupledESMDriver(
        atm, CoupledConfig(ocean_mode="garbage_mode"),
    )
    drv._atm.setup()
    with pytest.raises(ValueError, match="unknown ocean_mode"):
        drv._init_ocean()


def test_dynamic_ocean_requires_latlon_grid():
    """ocean_mode='dynamic' on a non-lat-lon ocean grid raises (same-grid Phase
    1 only; cube-atm + tripole-ocean needs the deferred cross-grid remap)."""
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=4),
        dycore=DycoreConfig(dt=60.0, model_type="hydrostatic"),
        radiation="gray", convection="none", days=1,
    )
    drv = CoupledESMDriver(
        atm, CoupledConfig(ocean_mode="dynamic",
                           ocean_config=LatLonCGridOceanConfig.from_flat(), ocean_nlev=4),
    )
    drv._atm.setup()
    with pytest.raises(ValueError, match="requires a regular lat-lon ocean grid"):
        drv._init_ocean()


# ---------------------------------------------------------------------------
# WOA realistic initial condition (ocean_ic="woa")
# ---------------------------------------------------------------------------

import os

_WOA_T = "data/woa18/woa18_decav_t00_01.nc"
_WOA_S = "data/woa18/woa18_decav_s00_01.nc"


def test_config_holds_woa_ic_fields():
    """CoupledConfig exposes the WOA-IC knobs; defaults keep the rest state."""
    c = CoupledConfig()
    assert c.ocean_ic == "rest"
    assert c.woa_t_path is None and c.woa_s_path is None
    w = CoupledConfig(ocean_ic="woa", woa_t_path="t.nc", woa_s_path="s.nc")
    assert w.ocean_ic == "woa" and w.woa_t_path == "t.nc"


def test_woa_ic_requires_paths():
    """ocean_ic='woa' without WOA paths raises (no silent fallback to rest)."""
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(dt=60.0, model_type="hydrostatic",
                            discretization="latlon_cgrid"),
        radiation="gray", convection="none", days=1,
    )
    drv = CoupledESMDriver(
        atm, CoupledConfig(ocean_mode="dynamic",
                           ocean_config=LatLonCGridOceanConfig.from_flat(),
                           ocean_nlev=4, ocean_ic="woa"),
    )
    drv._atm.setup()
    with pytest.raises(ValueError, match="requires woa_t_path"):
        drv._init_ocean()


def test_init_dynamic_ocean_rejects_unknown_ic():
    """_init_dynamic_ocean raises on an unknown ocean_ic (dispatch-hardening)."""
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(dt=60.0, model_type="hydrostatic",
                            discretization="latlon_cgrid"),
        radiation="gray", convection="none", days=1,
    )
    drv = CoupledESMDriver(
        atm, CoupledConfig(ocean_mode="dynamic",
                           ocean_config=LatLonCGridOceanConfig.from_flat(),
                           ocean_nlev=4, ocean_ic="garbage_ic"),
    )
    drv._atm.setup()
    with pytest.raises(ValueError, match="ocean_ic must be"):
        drv._init_ocean()


@pytest.mark.skipif(
    not (os.path.exists(_WOA_T) and os.path.exists(_WOA_S)),
    reason="WOA18 files not present (data/woa18/)",
)
def test_woa_ocean_mask_realistic():
    """woa_ocean_mask on a 2deg lat-lon grid yields a physical land/ocean split
    (both land and ocean present; global ocean fraction ~0.6-0.75)."""
    import numpy as np
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_woa import woa_ocean_mask
    grid = create_latlon_grid(n_lat=90, n_lon=180)  # 2 deg
    mask = woa_ocean_mask(grid, _WOA_T)
    assert mask.shape == (90, 180)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    ocean_frac = float(mask.mean())
    assert 0.55 < ocean_frac < 0.80, f"ocean_frac={ocean_frac} unphysical"
    # Land must exist (continents) and ocean must exist.
    assert mask.min() == 0.0 and mask.max() == 1.0


# ---------------------------------------------------------------------------
# Dynamic 3D-ocean checkpoint / restart (ckpt v2): the prognostic
# LatLonCGridOceanState is flattened into the coupled npz so an ocean_mode=
# 'dynamic' run can be checkpointed and resumed (job-chained multi-month
# equilibration).  Previously a dynamic restart raised "ckpt v2 deferred".
# ---------------------------------------------------------------------------


def _make_dynamic_driver(output_dir, nlev=4):
    """Minimal dynamic-3D-ocean coupled driver on a tiny shared lat-lon grid
    with the rest IC (no WOA files needed).  Builds only the atm grid + the
    ocean (the path the dynamic-ocean unit tests above use)."""
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(dt=60.0, model_type="hydrostatic",
                            discretization="latlon_cgrid"),
        radiation="gray", convection="none", days=1,
    )
    drv = CoupledESMDriver(
        atm,
        CoupledConfig(ocean_mode="dynamic",
                      ocean_config=LatLonCGridOceanConfig(),
                      ocean_nlev=nlev, ocean_ic="rest"),
        output_dir=str(output_dir),
    )
    drv._atm.setup()
    drv._init_ocean()
    return drv


def test_flatten_restore_ocean_state_roundtrip(tmp_path):
    """The pytree flatten/restore helpers round-trip a LatLonCGridOceanState
    through npz bitwise, and restore ALL leaves atomically (the land/u/v mask
    triple stays mutually consistent — a partial restore would leak mass)."""
    import numpy as np
    from legoesm.driver.coupled_esm_driver import (
        _flatten_pytree_to_npz, _restore_pytree_from_npz)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=2.0, S_uniform=34.0)
    # Perturb T/eta to a non-IC pattern so a coincidental match is impossible.
    state = state._replace(
        T=state.T.replace(data=state.T.data + 1.234),
        eta=state.eta.replace(data=state.eta.data + 0.567))

    arrays = _flatten_pytree_to_npz(state, "ocean3d_")
    assert any(k.startswith("ocean3d_") for k in arrays)
    np.savez(tmp_path / "o.npz", **arrays)
    data = np.load(tmp_path / "o.npz")

    # Restore into a FRESH (unperturbed) rest state — must overwrite to match.
    fresh = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=2.0, S_uniform=34.0)
    restored, ok = _restore_pytree_from_npz(fresh, data, "ocean3d_", "o.npz")
    assert ok
    for name in ("T", "S", "u", "v", "eta", "H_bathy",
                 "land_mask", "u_mask", "v_mask"):
        np.testing.assert_array_equal(
            np.asarray(getattr(state, name).data),
            np.asarray(getattr(restored, name).data),
            err_msg=f"leaf {name} did not round-trip")
    # A prefix with no matching keys is a no-op (restored=False).
    _, ok2 = _restore_pytree_from_npz(fresh, data, "nope_", "o.npz")
    assert ok2 is False


def test_dynamic_ocean_checkpoint_restores_and_no_longer_raises(tmp_path):
    """load_coupled_checkpoint restores the full 3D-ocean state from a v2
    checkpoint (no longer raises 'ckpt v2 deferred')."""
    import numpy as np
    import jax.numpy as jnp
    from legoesm.driver.coupled_esm_driver import _flatten_pytree_to_npz

    drv = _make_dynamic_driver(tmp_path)
    assert getattr(drv, "_is_dynamic_ocean", False)
    # An "evolved day-5" ocean = the fresh IC perturbed (T+2.5, eta+0.3).
    target = drv._ocean_state._replace(
        T=drv._ocean_state.T.replace(data=drv._ocean_state.T.data + 2.5),
        eta=drv._ocean_state.eta.replace(data=drv._ocean_state.eta.data + 0.3))
    arrays = {
        "_ckpt_version": np.asarray(2, np.int64),
        "_ckpt_ocean3d_shape": np.asarray(target.T.data.shape, np.int64),
    }
    arrays.update(_flatten_pytree_to_npz(target, "ocean3d_"))
    np.savez(tmp_path / "coupled_day_0005.npz", **arrays)

    drv2 = _make_dynamic_driver(tmp_path)
    pre_mean = float(jnp.mean(drv2._ocean_state.T.data))
    drv2.load_coupled_checkpoint(5.0, checkpoint_dir=str(tmp_path))  # no raise
    np.testing.assert_array_equal(
        np.asarray(drv2._ocean_state.T.data), np.asarray(target.T.data))
    np.testing.assert_array_equal(
        np.asarray(drv2._ocean_state.eta.data), np.asarray(target.eta.data))
    # The restore genuinely overwrote the fresh IC.
    assert float(jnp.mean(drv2._ocean_state.T.data)) != pytest.approx(pre_mean)


def test_dynamic_ocean_pre_v2_checkpoint_raises(tmp_path):
    """A dynamic run restarting from a pre-ckpt-v2 file (slab-only keys, no
    ocean3d_*) refuses loudly rather than silently resume the ocean at its IC."""
    import numpy as np
    drv = _make_dynamic_driver(tmp_path)
    np.savez(
        tmp_path / "coupled_day_0003.npz",
        _ckpt_version=np.asarray(1, np.int64),
        ocean_T_sfc=np.zeros((8, 16)), ocean_T_deep=np.zeros((8, 16)))
    with pytest.raises(ValueError, match="no 3D-ocean state"):
        drv.load_coupled_checkpoint(3.0, checkpoint_dir=str(tmp_path))


def test_dynamic_ocean_shape_mismatch_raises(tmp_path):
    """A v2 checkpoint whose saved 3D-ocean shape differs from the current ocean
    grid/levels is refused (no silent mis-mapping)."""
    import numpy as np
    from legoesm.driver.coupled_esm_driver import _flatten_pytree_to_npz
    drv = _make_dynamic_driver(tmp_path, nlev=4)
    arrays = {
        "_ckpt_version": np.asarray(2, np.int64),
        "_ckpt_ocean3d_shape": np.asarray((8, 16, 99), np.int64),  # wrong nlev
    }
    arrays.update(_flatten_pytree_to_npz(drv._ocean_state, "ocean3d_"))
    np.savez(tmp_path / "coupled_day_0002.npz", **arrays)
    with pytest.raises(ValueError, match="3D-ocean shape"):
        drv.load_coupled_checkpoint(2.0, checkpoint_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Dynamic-ocean cold-start stack wiring: _init_dynamic_ocean forces the
# OMIP-validated CFL-capped Laplacian-Smagorinsky eddy viscosity + implicit_cn
# barotropic solver via LatLonCGridOceanConfig.replace_flat(...) (~line 337).
# replace_flat routes the flat names into their nested sub-configs
# (C_smag_lap / smag_cfl_safety -> LateralViscosityConfig; barotropic_solver ->
# BarotropicConfig); this pins that the overrides actually land there on the
# built model (the smag cap is what keeps the WOA/geostrophic cold start from
# blowing up, so a broken route would silently un-stabilise the run).
# ---------------------------------------------------------------------------


def test_dynamic_ocean_config_wires_smag_cap_and_implicit_barotropic(tmp_path):
    """The dynamic 3D-ocean build wires the cold-start viscosity cap +
    implicit_cn barotropic into the model config's nested sub-configs."""
    drv = _make_dynamic_driver(tmp_path)
    ocfg = drv._ocean_model.config
    # CFL-capped Laplacian-Smagorinsky (LateralViscosityConfig).
    assert ocfg.lateral_viscosity.C_smag_lap == 3.0
    assert ocfg.lateral_viscosity.smag_cfl_safety == 0.125
    # Implicit Crank-Nicolson barotropic solver (BarotropicConfig).
    assert ocfg.barotropic.barotropic_solver == "implicit_cn"
