"""Ohlmann (2003) chlorophyll double exponential (J. Climate 16, 1337-1351). The reference values below were copied
from the paper's Table 1a (PDF, page 1339; transmission relative to the net irradiance just beneath the surface)."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.ocean.physics import shortwave_penetration as sp  # noqa: E402

# Table 1a of the paper: chl, A1, B1, A2, B2 (column order of the paper)
TABLE_1A = """
0.01 0.4488 0.0319 0.2940 0.3306
0.02 0.4563 0.0355 0.2894 0.3433
0.03 0.4622 0.0384 0.2858 0.3537
0.05 0.4715 0.0434 0.2800 0.3705
0.10 0.4877 0.0532 0.2703 0.4031
0.15 0.4993 0.0612 0.2628 0.4262
0.20 0.5084 0.0681 0.2571 0.4456
0.25 0.5159 0.0743 0.2523 0.4621
0.30 0.5223 0.0800 0.2481 0.4763
0.35 0.5278 0.0853 0.2444 0.4889
0.40 0.5326 0.0902 0.2411 0.4999
0.45 0.5369 0.0949 0.2382 0.5100
0.50 0.5408 0.0993 0.2356 0.5191
0.60 0.5474 0.1077 0.2309 0.5347
0.70 0.5529 0.1154 0.2269 0.5477
0.80 0.5576 0.1227 0.2235 0.5588
0.90 0.5615 0.1294 0.2206 0.5682
1.00 0.5649 0.1359 0.2181 0.5764
1.50 0.5757 0.1640 0.2106 0.6042
2.00 0.5802 0.1876 0.2089 0.6206
2.50 0.5808 0.2082 0.2113 0.6324
3.00 0.5788 0.2264 0.2167 0.6425
"""
T1A = np.array([[float(x) for x in line.split()] for line in TABLE_1A.strip().splitlines()])


def test_table_equals_paper_table_1a():
    assert np.allclose(sp.OHLMANN_CHL, T1A[:, 0], atol=0, rtol=0)
    assert np.allclose(sp.OHLMANN_A1, T1A[:, 1], atol=0, rtol=0)
    assert np.allclose(sp.OHLMANN_B1, T1A[:, 2], atol=0, rtol=0)
    assert np.allclose(sp.OHLMANN_A2, T1A[:, 3], atol=0, rtol=0)
    assert np.allclose(sp.OHLMANN_B2, T1A[:, 4], atol=0, rtol=0)


def test_parameters_at_rows_and_between_rows():
    A1, A2, B1, B2 = sp.ohlmann_parameters(jnp.asarray(T1A[:, 0]))
    assert np.allclose(A1, T1A[:, 1]) and np.allclose(B1, T1A[:, 2])
    assert np.allclose(A2, T1A[:, 3]) and np.allclose(B2, T1A[:, 4])
    # linear in chl between rows (paper section 4)
    a1, _, b1, _ = sp.ohlmann_parameters(jnp.asarray(0.125))
    assert float(a1) == pytest.approx(0.5 * (0.4877 + 0.4993), abs=1e-12)
    assert float(b1) == pytest.approx(0.5 * (0.0532 + 0.0612), abs=1e-12)
    # clamped to the paper's range 0.01 to 3.0
    lo = sp.ohlmann_parameters(jnp.asarray(0.001))
    hi = sp.ohlmann_parameters(jnp.asarray(8.0))
    assert np.allclose([float(x) for x in lo], [0.4488, 0.2940, 0.0319, 0.3306])
    assert np.allclose([float(x) for x in hi], [0.5788, 0.2167, 0.2264, 0.6425])


def test_text_value_coare():
    # paper section 6: TOGA COARE, chl 0.13, attenuation 0.058 m-1 for the first term
    _, _, b1, _ = sp.ohlmann_parameters(jnp.asarray(0.13))
    assert float(b1) == pytest.approx(0.058, abs=5e-4)


def _kernel(chl, dz, sw=200.0):
    dz = jnp.asarray(dz)[None, :]
    wet = jnp.ones_like(dz)
    dT = sp.shortwave_penetration_ohlmann_tendency(jnp.asarray([sw]), jnp.asarray([chl]), dz, wet,
                                                   rho_0=1025.0, c_sw=3985.0)
    return np.asarray(dT[0]) * 1025.0 * 3985.0 * np.asarray(dz[0])          # W m-2 per cell


def test_profile_conservation_and_top_cell():
    dz = np.r_[np.full(10, 5.0), np.full(10, 20.0)]
    chl = 0.2
    q = _kernel(chl, dz)
    assert q.sum() == pytest.approx(200.0, rel=1e-12)                    # the column takes the whole net shortwave
    A1, A2, B1, B2 = (float(x) for x in sp.ohlmann_parameters(jnp.asarray(chl)))
    Tr = lambda z: A1 * np.exp(-B1 * z) + A2 * np.exp(-B2 * z)          # noqa: E731
    assert q[0] == pytest.approx(200.0 * (1.0 - Tr(5.0)), rel=1e-12)    # includes the 1 - A1 - A2 part
    zf = np.cumsum(dz)
    for k in range(1, len(dz) - 1):
        assert q[k] == pytest.approx(200.0 * (Tr(zf[k - 1]) - Tr(zf[k])), rel=1e-10)
    assert q[-1] == pytest.approx(200.0 * Tr(zf[-2]), rel=1e-10)        # what reaches the bottom stays there


def _box_heat_gain(sw, chl, scheme="ohlmann_2003", profile=False):
    """Heat gained by a small lat-lon box in one step with q_net = 200 W m-2 (as test_shortwave_single_owner)."""
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.grids.factory import create_regional_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.init_latlon_cgrid import make_partial_cell_latlon, rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.state import LatLonCGridOceanConfig, OceanSurfaceForcing
    from legoesm.ocean.vertical import compute_layer_thickness, create_z_star_from_thicknesses
    set_policy(PrecisionPolicy.fp64())
    grid, wall = create_regional_grid("latlon", n_lat=6, n_lon=8, lat_south=-3.0, lat_north=3.0,
                                      lon_west=200.0, lon_east=208.0)
    ocean = np.asarray(wall) > 0.5
    dz = np.geomspace(10.0, 400.0, 6)
    dz *= 2000.0 / dz.sum()
    zc = create_z_star_from_thicknesses(dz)
    zc_pc, H, lm = make_partial_cell_latlon(zc, np.where(ocean, 2000.0, 0.0), ocean.astype(float))
    st = rest_state_latlon_cgrid_ocean(grid, zc_pc, land_mask_override=jnp.asarray(lm),
                                       H_bathy_override=jnp.asarray(H), H_max=float(zc.H_max))
    lm = np.asarray(lm)
    zmid = np.cumsum(dz) - dz / 2
    T0 = (4.0 + 20.0 * np.exp(-zmid / 200.0))[None, None, :] * lm[..., None]
    st = st._replace(T=st.T.replace(data=jnp.asarray(np.broadcast_to(T0, st.T.data.shape).copy())))
    cfg = LatLonCGridOceanConfig.from_flat(physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"), lateral_mixing=LateralMixingConfig(scheme="none")))
    ph = cfg.physics._replace(surface_forcing=cfg.physics.surface_forcing._replace(shortwave_scheme=scheme),
                              shortwave_penetration=None)
    model = LatLonCGridOceanModel(grid, zc_pc, cfg._replace(physics=ph))
    sf = OceanSurfaceForcing(tau_x=jnp.zeros_like(st.eta.data), tau_y=jnp.zeros_like(st.eta.data),
                             q_net=jnp.asarray(200.0 * lm), sw_down=None if sw is None else jnp.asarray(sw * lm),
                             chl=None if chl is None else jnp.asarray(chl * np.ones_like(lm)))
    new = model.step(st, 1800.0, surface_forcing=sf)
    area = np.asarray(grid.area)[..., None]

    def heat(s, by_level=False):
        h = np.asarray(compute_layer_thickness(s.eta.data, s.H_bathy.data, zc_pc))
        x = h * np.asarray(s.T.data) * area
        return x.sum((0, 1)) if by_level else float(np.sum(x))
    if profile:
        return heat(new, True) - heat(st, True)
    return heat(new) - heat(st)


def test_lat_lon_step_gains_q_net_and_needs_chl():
    ref = _box_heat_gain(None, None)                 # all of q_net non solar
    got = _box_heat_gain(200.0, 0.2)                 # q_net with sw_down = q_net through the Ohlmann profile
    assert got == pytest.approx(ref, rel=1e-9)
    with pytest.raises(ValueError, match="requires"):
        _box_heat_gain(200.0, None)


def test_lat_lon_step_puts_shortwave_below_the_top_cell():
    # with all of q_net as shortwave, the levels below the top gain heat that the non-solar run puts in the top
    ref = _box_heat_gain(None, None, profile=True)
    got = _box_heat_gain(200.0, 0.2, profile=True)
    assert got[0] < ref[0]
    assert np.all(got[1:3] > ref[1:3])


def test_more_chlorophyll_traps_more_near_the_surface():
    dz = np.full(30, 5.0)
    below50 = lambda c: _kernel(c, dz)[10:].sum()                       # noqa: E731
    assert below50(0.05) > below50(0.2) > below50(1.0)
