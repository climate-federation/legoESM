"""``--frazil`` wires the frazil closure into the JRA55 block scan.

Non-vacuous by construction: a column seeded 0.5 K below the freezing point on
its top two levels stays supercooled through one forced step with the flag
off (the slab tile disables the freeze cap and nothing else warms it), and is
lifted to the freezing point with ice deposited in the slab tile with the flag
on, the latent heat matching the ice mass.  Shares the tiny lat-lon JRA55
fixture of the sea-ice scan tests (the block builder is lane-agnostic; the
ico9 launcher selects the flag for MPAS).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_sea_ice_scan_frazil", _HERE / "test_run_omip_sea_ice_scan.py")
_scan = importlib.util.module_from_spec(_spec)
sys.modules["_sea_ice_scan_frazil"] = _scan
_spec.loader.exec_module(_scan)
run_omip = _scan.run_omip

jax.config.update("jax_enable_x64", True)


def _setup(tmp_path, *, frazil: bool, sea_ice: bool = True):
    cache = _scan._make_synthetic_cache(tmp_path, n_lat=8, n_lon=16, n_records=64)
    grid, z_coord, _, model, _ = _scan._make_tiny_latlon_setup(n_lat=8, n_lon=16)
    T_woa, S_woa = _scan._make_woa_like_targets(grid, nlev=4)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _scan._argparse_namespace(jra55_cache=str(cache))
    args.jra55_sea_ice = sea_ice
    args.frazil = frazil
    js = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa)
    return model, state, js


def _supercool(state, j=3, i=5, depth_K=0.5):
    from legoesm import constants
    tf_C = constants.T_freeze_ocean - constants.T_freeze
    T = np.asarray(state.T.data).copy()
    T[j, i, :2] = tf_C - depth_K
    return state._replace(T=state.T.replace(data=jnp.asarray(T))), tf_C


@pytest.mark.parametrize("frazil", [False, True])
def test_supercooled_column_makes_ice_only_with_frazil(tmp_path, frazil):
    from legoesm import constants
    from legoesm.ocean.vertical import compute_layer_thickness
    model, state, js = _setup(tmp_path, frazil=frazil)
    assert js["enable_frazil"] is frazil
    state, tf_C = _supercool(state)
    dt = 300.0
    block_fn = run_omip._build_jra55_block_fn(model, js, dt)
    atm_stack, runoff_stack = run_omip._preload_jra55_forcing_block(0, 1, dt, js)
    ice0 = js["ice_state_init"]
    s1, ice1 = block_fn(state, atm_stack, runoff_stack, jnp.int32(0), ice0)
    T_col = np.asarray(s1.T.data)[3, 5, :2]
    V = lambda ice: np.asarray(ice.h_ice.data) * np.asarray(ice.concentration.data)
    dV = float((V(ice1) - V(ice0))[3, 5])
    assert np.all(np.isfinite(np.asarray(s1.T.data)))
    if not frazil:
        assert np.all(T_col < tf_C), T_col
        assert dV == 0.0
        return
    # The tracer is potential temperature; the closure equilibrates the IN-SITU
    # temperature at the (pressure-insensitive) freezing point, so the tracer
    # sits below it by the adiabatic offset.  Invert exactly as the driver does.
    from legoesm.ocean.eos import in_situ_temperature
    dz_ref = np.asarray(js["frazil_z_coord"].dz_ref, dtype=np.float64)
    p_dbar = constants.rho_ocean * constants.g * (np.cumsum(dz_ref) - 0.5 * dz_ref) * 1e-4
    T_insitu = np.asarray(in_situ_temperature(
        jnp.asarray(np.asarray(s1.S.data)[3, 5, :2]), jnp.asarray(T_col), jnp.asarray(p_dbar[:2])))
    assert np.all(T_insitu >= tf_C - 1e-4), (T_col, T_insitu)
    assert dV > 0.0, dV
    # Enthalpy closure at the column, against the frazil-off trajectory so the
    # forced step's own heating cancels: rho*cp*sum(dT*h) == Lf * ice mass.
    model0, state0, js0 = _setup(tmp_path / "off", frazil=False)
    state0, _ = _supercool(state0)
    s0, _ = run_omip._build_jra55_block_fn(model0, js0, dt)(
        state0, atm_stack, runoff_stack, jnp.int32(0), js0["ice_state_init"])
    h = np.asarray(compute_layer_thickness(s1.eta.data, s1.H_bathy.data, js["frazil_z_coord"]))[3, 5]
    dT = (np.asarray(s1.T.data) - np.asarray(s0.T.data))[3, 5]
    q_ocean = constants.rho_ocean * constants.c_sw * float(np.sum(dT * h))
    ice_mass = dV * js["ice_config"].rho_ice
    # The closure conserves cp*M*T with the frozen mass leaving the liquid:
    # cp*(M'T' - M T) = m*Lf, i.e. cp*M*(T'-T) = m*(Lf + cp*T') with T' at the
    # freezing point (the lane keeps the liquid thickness fixed under the
    # virtual-salt-flux closure, so the fixed-mass sum above is what the
    # ocean sees).
    q_ice = ice_mass * (constants.L_f + constants.c_sw * tf_C)
    assert q_ocean == pytest.approx(q_ice, rel=5e-3), (q_ocean, q_ice)
    # Water: the frozen mass left the column through sea level.
    d_eta = float((np.asarray(s1.eta.data) - np.asarray(s0.eta.data))[3, 5])
    assert d_eta == pytest.approx(-ice_mass / model.config.rho_0, rel=1e-6)
    # Nothing happens away from the seeded column.
    assert float(np.abs(V(ice1) - V(ice0)).sum()) == pytest.approx(abs(dV), rel=1e-12)


def test_frazil_requires_the_slab_tile(tmp_path):
    with pytest.raises(SystemExit, match="requires --jra55-sea-ice"):
        _setup(tmp_path, frazil=True, sea_ice=False)
