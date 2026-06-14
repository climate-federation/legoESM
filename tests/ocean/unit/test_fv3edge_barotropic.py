"""Regression guard: the gated FV3Edge ocean barotropic (algorithmically-faithful
upwind-flux SW core) — cc<->edge lift round-trip + stability + zonal symmetry.

fv3_faithful: `barotropic_staggering="fv3edge"` routes the ocean barotropic through
the TRUE FV3 edge-staggered SW core (`FV3EdgeShallowWaterModel`: upwind
absolute-vorticity flux + `d2a2c_vect`), the algorithmically-faithful counterpart
to the default corner-staggered `fv3sw` (`CDGridShallowWaterModel`, centered).
The only new machinery is the cc<->edge-midpoint vector lift
(`_cc_to_edge_vector`/`_edge_to_cc_vector`); these tests fail if that lift loses
its round-trip accuracy or if the gated path stops resolving the A-grid geostrophic
artifact / goes unstable.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.grids.cubed_sphere import create_cubed_sphere  # noqa: E402
from legoesm.grids.cubed_sphere_cdgrid import (  # noqa: E402
    create_cubed_sphere_cdgrid, cell_centre_angles_from_4edge,
)
from legoesm.ocean.dynamics.barotropic_cgrid import (  # noqa: E402
    _cc_to_edge_vector, _edge_to_cc_vector, _derive_fv3edge_masks,
)


def test_cc_edge_lift_round_trips():
    """cc(face-local) -> FV3 edge-midpoint -> cc recovers a smooth field, and the
    edge winds match the analytic edge init (interp vs exact)."""
    n = 24
    grid = create_cubed_sphere(n)
    cd = create_cubed_sphere_cdgrid(grid)
    ca, sa = cell_centre_angles_from_4edge(cd)
    ca = np.asarray(ca); sa = np.asarray(sa)
    lat = np.asarray(grid.lat)
    u_geo = 30.0 * np.cos(lat); v_geo = np.zeros_like(u_geo)
    u_fl = jnp.asarray(ca * u_geo + sa * v_geo)
    v_fl = jnp.asarray(-sa * u_geo + ca * v_geo)

    u_d, v_d = _cc_to_edge_vector(u_fl, v_fl, cd)
    assert u_d.shape == (6, n, n + 1) and v_d.shape == (6, n + 1, n)
    u_cc2, v_cc2 = _edge_to_cc_vector(u_d, v_d)

    def rel(a, b):
        return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2))
                     / (np.sqrt(np.nanmean(np.asarray(b) ** 2)) + 1e-30))
    assert rel(u_cc2, u_fl) < 0.01, "cc->edge->cc u round-trip lost accuracy"
    assert rel(v_cc2, v_fl) < 0.01, "cc->edge->cc v round-trip lost accuracy"
    # edge winds vs analytic geographic->edge init
    u_d_analytic = np.asarray(cd.cos_angle_edge_x) * (30.0 * np.cos(np.asarray(cd.lat_edge_x)))
    assert rel(u_d, u_d_analytic) < 0.01


def test_fv3edge_masks_block_coasts():
    n = 8
    grid = create_cubed_sphere(n)
    mask = jnp.ones((6, n, n))
    mask = mask.at[0, 0, 0].set(0.0)  # one land cell
    ux, vy = _derive_fv3edge_masks(mask)
    assert ux.shape == (6, n, n + 1) and vy.shape == (6, n + 1, n)
    # edges touching the land cell are zeroed (impermeable)
    assert float(ux[0, 0, 0]) == 0.0 and float(ux[0, 0, 1]) == 0.0
    assert float(vy[0, 0, 0]) == 0.0 and float(vy[0, 1, 0]) == 0.0


def test_fv3edge_barotropic_stable_and_resolves_artifact():
    """End-to-end: an OceanModel with barotropic_staggering='fv3edge' steps the
    zonal-thermal-wind geostrophic case stably and keeps eta near-zonal (the
    A-grid artifact was ~40% non-zonal)."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "rom_t", "scripts/matrix/run_ocean_test_matrix.py")
    rom = importlib.util.module_from_spec(spec)
    sys.modules["rom_t"] = rom
    spec.loader.exec_module(rom)
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.grids.regridding import (
        get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)
    tc = rom.TestCase(case="geostrophic_adjustment", grid_type="cubed_sphere",
                      resolution="C24", duration_days=10.0, quick_days=1.0)
    grid, z, config, _, _, _, _ = rom._create_ocean_setup(tc)
    m = OceanModel(grid, z, config._replace(barotropic_staggering="fv3edge"))
    st = rom._create_rest_state(tc, grid, z)
    st = rom._add_baroclinic_perturbation(st, "cubed_sphere", grid, z)
    for _ in range(int(0.5 * 86400 / 300.0)):
        st = m.step(st, 300.0)
    assert bool(jnp.all(jnp.isfinite(st.eta.data))), "fv3edge barotropic unstable"
    w = get_cubedsphere_to_latlon_weights(24, n_lon=360, n_lat=181)
    ll = apply_cubedsphere_to_latlon(np.asarray(st.eta.data), w)
    zm = np.nanmean(ll, axis=1, keepdims=True)
    nz = float(np.nansum((ll - zm) ** 2)
               / max(np.nansum((ll - np.nanmean(ll)) ** 2), 1e-30))
    assert nz < 0.10, (
        f"fv3edge geostrophic eta non-zonal fraction {nz:.3f} (>10%): the "
        "A-grid computational-mode artifact (~40%) should be resolved."
    )
