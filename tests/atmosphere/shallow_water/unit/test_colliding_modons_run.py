"""Colliding-modons SW test case (issue #521) — FV3 C-D-grid IC + short
prognostic run.

Complements ``test_colliding_modons_ic.py`` (centred + lat-lon IC fields) by
exercising the ``colliding_modons_cdgrid`` assembler on the FV3 edge-midpoint
C-D grid and a short non-rotating ``FV3EdgeShallowWaterModel`` integration
(mass conservation, boundedness, modon evolution, no cube-corner blow-up).

Sci test: the mass-conservation and wind-antisymmetry tolerances require
float64, so the module is skipped under the default float32 backend (run via
``JAX_ENABLE_X64=1``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

if not jax.config.read("jax_enable_x64"):
    pytest.skip(
        "colliding-modons prognostic test needs float64 (JAX_ENABLE_X64=1)",
        allow_module_level=True,
    )

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3EdgeShallowWaterModel, iter1009_dual_target_config)
from tests.test_cases.colliding_modons import (
    colliding_modons_cdgrid, _modon_winds_geo, _MODON_UMAX, _MODON_H0)


def _u_east(lon, lat, radius):
    """Scalar eastward wind from the shared kernel (v_north is identically 0)."""
    u_east, _ = _modon_winds_geo(lon, lat, radius)
    return u_east


@pytest.fixture
def fp64_policy():
    """Pin the legoESM precision policy to fp64 for mass-conservation pins.

    ``JAX_ENABLE_X64`` alone does not change the model compute dtype — the
    step casts through the ACTIVE PrecisionPolicy, and the anchored mass
    fixer's per-step correction rounds to the state dtype, capping fp32
    relative conservation at ~1e-7 (the 1e-10 pins here need fp64).
    Restored after the test so no policy leaks across the session.
    """
    from legoesm.core.precision import get_policy, set_policy, PrecisionPolicy

    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def test_modon_ic_fields():
    """Uniform free surface, peak zonal wind = Umax, antisymmetric twin jets,
    Coriolis zeroed (non-rotating)."""
    grid = create_cubed_sphere(24)
    cd = create_cubed_sphere_cdgrid(grid)
    state, cd_nr = colliding_modons_cdgrid(grid, cd)

    # uniform h = 5000 m
    assert np.allclose(np.asarray(state.h), _MODON_H0)
    # non-rotating: f zeroed
    assert float(jnp.max(jnp.abs(cd_nr.f_corner))) == 0.0
    assert float(jnp.max(jnp.abs(cd.f_corner))) > 0.0   # original was rotating

    # peak geographic wind at the two centres has the right sign + magnitude
    R = grid.radius
    u_w = float(_u_east(jnp.array(jnp.pi * 0.5), jnp.array(0.0), R))
    u_e = float(_u_east(jnp.array(jnp.pi * 1.5), jnp.array(0.0), R))
    assert u_w == pytest.approx(_MODON_UMAX, abs=0.1)     # westerly +50
    assert u_e == pytest.approx(-_MODON_UMAX, abs=0.1)    # easterly -50
    # antisymmetry of the twin (centres are mirror images): u(lon+pi) = -u(lon)
    lon = jnp.linspace(0.0, 2 * jnp.pi, 37)
    u_a = _u_east(lon, jnp.zeros_like(lon), R)
    u_b = _u_east(lon + jnp.pi, jnp.zeros_like(lon), R)
    assert float(jnp.max(jnp.abs(u_a + u_b))) < 1e-9
    # projected D-grid winds never exceed the geographic peak
    assert float(jnp.max(jnp.abs(state.u_d))) <= _MODON_UMAX + 1e-6


def test_modon_prognostic_run_stable_and_conserves_mass():
    """Short non-rotating SW integration: finite, mass-conserving, modons evolve,
    no cube-corner blow-up."""
    # N=36: the iter1009 dual-target preset is CALIBRATED/validated at C36
    # (audit item 13).  The prior C24 ran the preset at an uncalibrated
    # resolution where its div-damp/filter coefficients are off-design.
    n = 36
    grid = create_cubed_sphere(n)
    model = FV3EdgeShallowWaterModel(grid, iter1009_dual_target_config(n))
    state, cd_nr = colliding_modons_cdgrid(grid, model.cdgrid)
    model.cdgrid = cd_nr                       # non-rotating planet (FV3 f0=fC=0)
    model.set_initial_mass(state)

    area = model.cdgrid.base.area
    mass0 = float(jnp.sum(state.h * area))
    u0 = state.u_d

    step = jax.jit(lambda s: model.step(s, 300.0))
    s = state
    for _ in range(96):                        # 8 h at dt=300 s
        s = step(s)
    jax.block_until_ready(s.h)

    assert np.all(np.isfinite(np.asarray(s.h)))
    assert np.all(np.isfinite(np.asarray(s.u_d)))
    mass1 = float(jnp.sum(s.h * area))
    # FV3 flux-form cube SW conserves mass to the divergence-damping/filter
    # roundoff, not machine precision (cf. the established cube SW convention
    # test_fv_cubesphere.py: mass_drift < 1e-4).  h is uniform so mass0 is
    # IC-independent; the drift here is the scheme's at the CALIBRATED C36
    # (the iter1009 dual-target preset is validated at N=36).
    # 1e-5 keeps a real regression guard (100x tighter than the cube convention)
    # with comfortable margin over the measured drift.
    assert abs(mass1 / mass0 - 1.0) < 1e-5           # mass conserved (cube SW)
    assert float(jnp.max(jnp.abs(s.u_d))) < 200.0    # bounded, no blow-up
    assert float(jnp.max(jnp.abs(s.u_d - u0))) > 1e-3  # modons actually evolved
    # free surface stays physical (no negative depth)
    assert float(jnp.min(s.h)) > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Lat-lon wiring (#521 follow-up): matrix catalog + non-rotating C-grid run
# ---------------------------------------------------------------------------

def _load_matrix_module():
    """Load the matrix-runner script WITHOUT permanent sys.path pollution
    (codex round-12 Low): spec-load under a private module name; nothing
    is inserted into sys.path and the bare name never enters sys.modules."""
    import importlib.util
    import sys
    from pathlib import Path
    script = (Path(__file__).resolve().parents[4]
              / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    name = "_modons_matrix_catalog"
    spec = importlib.util.spec_from_file_location(name, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_matrix_registers_all_four_grids():
    """Catalog-backed (name, grid) gate: ``--test colliding_modons`` must
    select a REAL case on EVERY grid type (never a silent no-op) — the
    case is a standard cross-grid SW test."""
    M = _load_matrix_module()
    mat = M._build_test_matrix()
    cm = [t for t in mat if t.case == "colliding_modons"]
    grids = sorted(t.grid_type for t in cm)
    assert grids == ["cubed_sphere", "icosahedral", "latlon",
                     "spectral"], grids
    for t in cm:
        assert t.run_kwargs.get("test_num") == 8, t.run_kwargs
    assert M.RUNNERS.get("colliding_modons") is M.run_shallow_water


def test_latlon_nonrotating_run_stable_and_conserves_mass(fp64_policy):
    """Short prognostic latlon C-grid modons run: non-rotating grid (f=0),
    finite fields, area-weighted mass conserved, and the flow stays
    equator-centred (winds do not blow up)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterModel, CGridLatLonShallowWaterConfig,
        CGridLatLonShallowWaterState,
    )
    from tests.test_cases.colliding_modons import (
        colliding_modons_latlon, _modon_winds_geo, _MODON_UMAX,
    )

    grid = create_latlon_grid(24, 48, omega=0.0)
    cm = colliding_modons_latlon(grid)
    R = grid.radius
    lon_f = jnp.concatenate([grid.lon - 0.5 * grid.dlon,
                             (grid.lon - 0.5 * grid.dlon)[0:1] + 2 * jnp.pi])
    u_face, _ = _modon_winds_geo(lon_f[None, :], grid.lat[:, None], R)
    lat_f = jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, grid.n_lat + 1)
    _, v_face = _modon_winds_geo(grid.lon[None, :], lat_f[:, None], R)
    state = CGridLatLonShallowWaterState(
        h=cm.h.data, u=u_face, v=v_face, h_s=jnp.zeros_like(cm.h.data))

    model = CGridLatLonShallowWaterModel(
        grid, CGridLatLonShallowWaterConfig(A_h=1e5,
                                            anchor_mass_to_initial=True),
        dt=120.0)
    # f64 accumulation for the mass pins (same rationale as the MPAS test:
    # policy/cache-dependent f32 geometry makes an f32 sum read as drift).
    def _mass(st):
        return float(jnp.sum(st.h.astype(jnp.float64)
                             * grid.area.astype(jnp.float64)))

    mass0 = _mass(state)
    s = state
    for _ in range(30):
        s = model.step(s, 120.0)
    assert bool(jnp.all(jnp.isfinite(s.h)))
    assert bool(jnp.all(jnp.isfinite(s.u)))
    mass1 = _mass(s)
    assert abs(mass1 - mass0) / mass0 < 1e-10
    # Winds bounded: no instability spike beyond ~2x the initial jet.
    assert float(jnp.max(jnp.abs(s.u))) < 2.0 * _MODON_UMAX


# ---------------------------------------------------------------------------
# MPAS + spectral wiring (#521 all-grid standard case): non-rotating runs
# ---------------------------------------------------------------------------

def test_mpas_nonrotating_run_stable_and_conserves_mass(fp64_policy):
    """Short prognostic MPAS modons run on an omega=0 Voronoi mesh:
    Coriolis identically zero, finite fields, mass conserved, winds
    bounded."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.shallow_water_mpas import (
        MPASShallowWaterModel, MPASShallowWaterConfig,
    )
    from tests.test_cases.colliding_modons import (
        colliding_modons_mpas, _MODON_UMAX,
    )

    mesh = create_voronoi_mesh(3, omega=0.0)   # 642 cells: fast CI size
    assert float(jnp.max(jnp.abs(mesh.fVertex))) == 0.0
    assert float(jnp.max(jnp.abs(mesh.fEdge))) == 0.0

    state = colliding_modons_mpas(mesh)
    assert float(jnp.max(jnp.abs(state.u.data))) <= _MODON_UMAX + 1e-6

    model = MPASShallowWaterModel(
        mesh, MPASShallowWaterConfig(anchor_mass_to_initial=True))

    # Mass diagnostics MUST accumulate in f64: the mesh cache can serve
    # f32 geometry (built under the default policy), and an f32
    # sum(h*area) carries ~3e-8 relative rounding — which then reads as
    # fake "drift" against the f64 post-step sum.  The model itself
    # conserves to ~1e-16 (fixer accumulates in the f64 budget dtype).
    def _mass(st):
        return float(jnp.sum(st.h.data.astype(jnp.float64)
                             * mesh.areaCell.astype(jnp.float64)))

    mass0 = _mass(state)
    s = state
    step = jax.jit(lambda st: model.step(st, 300.0))
    for _ in range(30):
        s = step(s)
    jax.block_until_ready(s.h.data)

    assert bool(jnp.all(jnp.isfinite(s.h.data)))
    assert bool(jnp.all(jnp.isfinite(s.u.data)))
    mass1 = _mass(s)
    assert abs(mass1 - mass0) / mass0 < 1e-10
    assert float(jnp.max(jnp.abs(s.u.data))) < 2.0 * _MODON_UMAX
    assert float(jnp.min(s.h.data)) > 0.0


def test_spectral_nonrotating_run_stable_and_conserves_mass(fp64_policy):
    """Short prognostic spectral modons run on an omega=0 Gaussian grid:
    f = 0, finite spectral state, mean geopotential (mass) conserved,
    physical winds bounded."""
    from legoesm import constants
    from legoesm.grids.gaussian import (
        create_gaussian_grid, sh_synthesis, uv_from_vordiv,
    )
    from legoesm.atmosphere.dynamics.spectral_sw import (
        SpectralShallowWaterModel, SpectralSWConfig,
    )
    from tests.test_cases.colliding_modons import (
        colliding_modons_spectral, _MODON_UMAX, _MODON_H0,
    )

    grid = create_gaussian_grid(21, omega=0.0)
    assert float(jnp.max(jnp.abs(grid.f))) == 0.0

    model = SpectralShallowWaterModel(
        grid, SpectralSWConfig(spectral_filter_order=8))
    state = model.filter_initial_state(colliding_modons_spectral(grid))

    def _mean_h(st):
        phi = sh_synthesis(grid, st.phi_hat.data)
        w = grid.grid_area / jnp.sum(grid.grid_area)
        return float(jnp.sum(phi * w) / constants.g)

    h_bar0 = _mean_h(state)
    assert h_bar0 == pytest.approx(_MODON_H0, rel=1e-6)

    s = state
    step = jax.jit(lambda st: model.step(st, 300.0))
    for _ in range(30):
        s = step(s)
    jax.block_until_ready(s.phi_hat.data)

    assert bool(jnp.all(jnp.isfinite(s.phi_hat.data)))
    assert bool(jnp.all(jnp.isfinite(s.vor_hat.data)))
    assert abs(_mean_h(s) / h_bar0 - 1.0) < 1e-10
    u_cos, v_cos = uv_from_vordiv(grid, s.vor_hat.data, s.div_hat.data)
    cos2d = grid.cos_lat[:, None]
    ws = jnp.sqrt((u_cos / cos2d) ** 2 + (v_cos / cos2d) ** 2)
    assert float(jnp.max(ws)) < 2.0 * _MODON_UMAX
