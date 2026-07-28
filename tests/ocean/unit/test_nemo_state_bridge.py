"""Unit tests for the NEMO -> legoESM state bridge (``fidelity.nemo_state_bridge``).

Self-contained: builds a small synthetic beta-plane ``NemoGrid``/``NemoState``
(no external NEMO run) and checks the geometry (Coriolis reconstruction), the
velocity staggering (NEMO east/north face -> legoESM u/v faces), and the state
placement.
"""
import numpy as np
import pytest

from legoesm.ocean.fidelity.nemo_io import NemoBeforeState, NemoGrid, NemoState
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_before_state_topo,
    bridge_nemo_to_legoesm,
    bridge_nemo_to_legoesm_topo,
)

NY, NX, NZ = 4, 5, 3
DXY = 1.0e5
F0_TRUE, BETA_TRUE = 1.0e-4, 1.0e-5   # per-cell f increment over dy


def _synthetic():
    ones2 = np.ones((NY, NX))
    # ff_t linear in y: f(y_c[j]) with y_c[j]=(j+0.5)*dy  -> f0 + beta*(j+0.5)*dy
    yc = (np.arange(NY) + 0.5) * DXY
    ff_t = (F0_TRUE + BETA_TRUE * yc)[:, None] * np.ones((1, NX))
    # ff_f lives at the F-point (NE corner) = y_g[j] = j*dy, a half-cell south of
    # y_c[j]; same beta slope (what the bridge checks), physically offset value.
    yg = np.arange(NY) * DXY
    ff_f = (F0_TRUE + BETA_TRUE * yg)[:, None] * np.ones((1, NX))
    grid = NemoGrid(
        glamt=ones2 * np.arange(NX)[None, :], gphit=ones2 * np.arange(NY)[:, None],
        e1t=ones2 * DXY, e2t=ones2 * DXY, e1u=ones2 * DXY, e2v=ones2 * DXY,
        ff_t=ff_t, ff_f=ff_f,
        e3t_1d=np.array([10.0, 20.0, 30.0]),
        gdept_1d=np.array([5.0, 20.0, 45.0]),
        gdepw_1d=np.array([0.0, 10.0, 30.0]),
        tmask=np.ones((NY, NX, NZ)), umask=np.ones((NY, NX, NZ)),
        vmask=np.ones((NY, NX, NZ)),
    )
    rng = np.random.default_rng(0)
    state = NemoState(
        T=15.0 + rng.random((NY, NX, NZ)), S=35.0 + rng.random((NY, NX, NZ)),
        u=rng.random((NY, NX, NZ)), v=rng.random((NY, NX, NZ)),
        ssh=0.01 * rng.random((NY, NX)), rhd=None,
    )
    return grid, state


def test_bridge_geometry_and_staggering():
    grid, state = _synthetic()
    out = bridge_nemo_to_legoesm(grid, state)

    # Coriolis reconstructed to ~roundoff (ff_t is exactly the beta-plane f).
    assert out.f_match_max_abs < 1e-12, out.f_match_max_abs

    st = out.state
    assert st.T.data.shape == (NY, NX, NZ)
    assert st.u.data.shape == (NY, NX + 1, NZ)      # west-face array, n_lon+1
    assert st.v.data.shape == (NY + 1, NX, NZ)      # south-face array, n_lat+1
    assert st.eta.data.shape == (NY, NX)

    # Staggering. legoESM u[j,i] is the WEST face of T-cell (j,i)
    # (latlon_cgrid_operators.py:3623); NEMO u(i) is the EAST face = west face of
    # cell i+1. So NEMO u[k-1] must land at u_face[:,k] (west face of cell k),
    # i.e. u_face[:,1:]==nemo_u with a west-wall column at index 0. A prepend/
    # append flip of this convention is caught here; the end-to-end validation is
    # the momentum-tendency match against NEMO.
    u = np.asarray(st.u.data)
    assert np.allclose(u[:, 0, :], 0.0)             # west wall
    assert np.allclose(u[:, 1:, :], state.u)        # interior + east wall = NEMO u
    # NEMO north-face v -> v_face[1:], south wall zero.
    v = np.asarray(st.v.data)
    assert np.allclose(v[0, :, :], 0.0)             # south wall
    assert np.allclose(v[1:, :, :], state.v)
    # Divergence-identity guard: under legoESM's west-face stencil the bridged
    # zonal convergence at an interior cell equals NEMO's east-face difference
    # (fails if the prepend flips to an append). Use a NONLINEAR u so a shift
    # changes the interior value.
    nemo_u_nl = (np.arange(NX) ** 2).astype(float)[None, :, None] * np.ones_like(state.u)
    out_nl = bridge_nemo_to_legoesm(grid, state._replace(u=nemo_u_nl))
    uf = np.asarray(out_nl.state.u.data)
    for k in range(1, NX):                          # interior west faces
        lego_diff = uf[:, k + 1, :] - uf[:, k, :]   # west-face stencil
        nemo_diff = nemo_u_nl[:, k, :] - nemo_u_nl[:, k - 1, :]
        assert np.allclose(lego_diff, nemo_diff), k

    # T/S placed directly at T-points.
    assert np.allclose(np.asarray(st.T.data), state.T)
    assert np.allclose(np.asarray(st.S.data), state.S)


def test_bridge_rejects_bad_coriolis():
    import pytest
    grid, state = _synthetic()
    # Corrupt ff_t so it is no longer the linear beta-plane the fit assumes.
    bad = grid.ff_t.copy()
    bad[2, 1] += 5e-5
    grid = grid._replace(ff_t=bad)
    with pytest.raises(ValueError, match="Coriolis mismatch"):
        bridge_nemo_to_legoesm(grid, state)


# ---------------------------------------------------------------------------
# Mercator + topography bridge (bridge_nemo_to_legoesm_topo) — the DINO path.
# Builds a synthetic uniform lat-lon (Mercator-consistent metrics) grid with
# column-varying bottom depth and exercises geometry, topography, periodicity.
# ---------------------------------------------------------------------------
TNY, TNX, TNZ = 6, 8, 4
LAT0, DLAT_D = -30.0, 5.0     # degrees, uniform
LON0, DLON_D = 0.0, 5.0


def _synthetic_topo():
    from legoesm import constants
    lat_deg = LAT0 + DLAT_D * np.arange(TNY)         # (TNY,)
    lon_deg = LON0 + DLON_D * np.arange(TNX)
    gphit = lat_deg[:, None] * np.ones((1, TNX))
    glamt = lon_deg[None, :] * np.ones((TNY, 1))
    gphiv = (lat_deg + 0.5 * DLAT_D)[:, None] * np.ones((1, TNX))  # north faces
    R, Om = constants.R_earth, constants.Omega
    lat_r = np.deg2rad(lat_deg)
    # Metrics from the sphere formulas create_latlon_geometry uses -> exact match.
    e1t = (R * np.deg2rad(DLON_D) * np.cos(lat_r))[:, None] * np.ones((1, TNX))
    e2t = np.full((TNY, TNX), R * np.deg2rad(DLAT_D))
    ff_t = (2.0 * Om * np.sin(lat_r))[:, None] * np.ones((1, TNX))
    e3t_1d = np.array([10.0, 20.0, 30.0, 40.0])
    depth_cum = np.cumsum(e3t_1d)
    gdept_1d = depth_cum - e3t_1d / 2.0
    gdepw_1d = np.concatenate([[0.0], depth_cum[:-1]])
    # Full-step topography: the southern row is shallow (2 wet levels), the rest
    # deep (all 4); every column keeps >=2 wet levels (no dry columns).
    k_bot = np.full((TNY, TNX), TNZ, dtype=int)
    k_bot[0, :] = 2
    tmask = (np.arange(TNZ)[None, None, :] < k_bot[:, :, None]).astype(float)
    grid = NemoGrid(
        glamt=glamt, gphit=gphit, e1t=e1t, e2t=e2t, e1u=e1t.copy(), e2v=e2t.copy(),
        ff_t=ff_t, ff_f=ff_t.copy(),
        e3t_1d=e3t_1d, gdept_1d=gdept_1d, gdepw_1d=gdepw_1d,
        tmask=tmask, umask=tmask.copy(), vmask=tmask.copy(), gphiv=gphiv,
    )
    rng = np.random.default_rng(1)
    state = NemoState(
        T=15.0 + rng.random((TNY, TNX, TNZ)), S=35.0 + rng.random((TNY, TNX, TNZ)),
        u=rng.random((TNY, TNX, TNZ)), v=rng.random((TNY, TNX, TNZ)),
        ssh=0.01 * rng.random((TNY, TNX)), rhd=None,
    )
    return grid, state, k_bot


def test_topo_bridge_geometry_matches_nemo_metrics():
    """Built geometry reproduces NEMO's e1t/e2t/ff_t (Mercator, from mesh arrays)."""
    grid, state, _ = _synthetic_topo()
    out = bridge_nemo_to_legoesm_topo(grid, state, periodic_i=True)
    geom = out.geometry
    # dx_T = e1t, dy_T = e2t, f_T = ff_t to near-roundoff (x64).
    assert np.max(np.abs(np.asarray(geom.dx_T) - grid.e1t)) < 1e-4 * grid.e1t.max()
    assert np.max(np.abs(np.asarray(geom.dy_T) - grid.e2t)) < 1e-4 * grid.e2t.max()
    assert out.f_match_max_abs < 1e-3 * np.abs(grid.ff_t).max()


def test_topo_bridge_metric_convention_default_is_bit_identical():
    """#1226: metric_convention default omission == explicit "exact"."""
    grid, state, _ = _synthetic_topo()
    out_default = bridge_nemo_to_legoesm_topo(grid, state, periodic_i=True)
    out_exact = bridge_nemo_to_legoesm_topo(
        grid, state, periodic_i=True, metric_convention="exact")
    for f in ("dx_T", "dy_T", "area_T", "dx_v", "dy_v", "area_q"):
        np.testing.assert_array_equal(
            getattr(out_default.geometry, f), getattr(out_exact.geometry, f))


def test_topo_bridge_metric_convention_isotropic_forwards_and_raises():
    """metric_convention="nemo_isotropic" is forwarded to create_latlon_geometry:
    dy_T becomes dx_T (NEMO's pe1t=pe2t isotropic identity). The synthetic
    fixture's e2t is built to match the "exact" formula (a true R*dlat), so an
    isotropic bridge is compared against an isotropic-consistent e2t (:=e1t)
    here -- otherwise the bridge's own e2t-vs-dy_T sanity guard (a REAL safety
    check on live NEMO mesh_mask data, not weakened here) would legitimately
    reject this fixture as a mismatched build."""
    grid, state, _ = _synthetic_topo()
    grid_iso = grid._replace(e2t=grid.e1t.copy())
    out_iso = bridge_nemo_to_legoesm_topo(
        grid_iso, state, periodic_i=True, metric_convention="nemo_isotropic")
    np.testing.assert_allclose(
        np.asarray(out_iso.geometry.dy_T), np.asarray(out_iso.geometry.dx_T),
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="metric_convention"):
        bridge_nemo_to_legoesm_topo(
            grid, state, periodic_i=True, metric_convention="bogus")


def test_topo_bridge_topography_and_staggering():
    grid, state, k_bot = _synthetic_topo()
    out = bridge_nemo_to_legoesm_topo(grid, state, periodic_i=True)
    st = out.state
    # Per-column bottom depth = sum of the top k_bot cell thicknesses.
    depth_cum = np.cumsum(grid.e3t_1d)
    H_expect = depth_cum[k_bot - 1]
    assert np.allclose(np.asarray(st.H_bathy.data), H_expect)
    # Surface land mask = tmask[:, :, 0] (all wet here).
    assert np.array_equal(out.land_mask, grid.tmask[:, :, 0] > 0.5)
    # Periodic u-face: west face of col 0 = NEMO east face of the last col (wrap).
    u = np.asarray(st.u.data)
    assert u.shape == (TNY, TNX + 1, TNZ)
    assert np.allclose(u[:, 0, :], state.u[:, -1, :])   # periodic image, NOT a wall
    assert np.allclose(u[:, 1:, :], state.u)
    # v is meridional (real N/S walls): south wall zero.
    v = np.asarray(st.v.data)
    assert np.allclose(v[0, :, :], 0.0)
    assert np.allclose(v[1:, :, :], state.v)


def test_topo_bridge_closed_vs_periodic_u():
    grid, state, _ = _synthetic_topo()
    closed = bridge_nemo_to_legoesm_topo(grid, state, periodic_i=False)
    u = np.asarray(closed.state.u.data)
    assert np.allclose(u[:, 0, :], 0.0)                 # closed -> west wall
    assert np.allclose(u[:, 1:, :], state.u)


def test_topo_bridge_requires_gphiv():
    import pytest
    grid, state, _ = _synthetic_topo()
    grid = grid._replace(gphiv=None)
    with pytest.raises(ValueError, match="gphiv"):
        bridge_nemo_to_legoesm_topo(grid, state)


def test_topo_bridge_rejects_interior_holes():
    # NB the guard detects mask TOPOLOGY (interior holes), NOT ln_zps partial
    # cells (whose mask is identical to full-step) — see the bridge docstring.
    import pytest
    grid, state, _ = _synthetic_topo()
    # Punch an interior hole (masked cell above a wet cell) -> not full-step.
    bad = grid.tmask.copy()
    bad[3, 3, 1] = 0.0   # level 1 dry but level 2/3 wet below -> interior hole
    grid = grid._replace(tmask=bad)
    with pytest.raises(ValueError, match="full-step"):
        bridge_nemo_to_legoesm_topo(grid, state)


def _synthetic_topo_mercator():
    """Stretched-latitude (Mercator) grid so create_latlon_geometry takes the
    VARIABLE-dlat branch and the reconstructed ``lat_face`` (from gphiv) actually
    drives ``dy_T`` — the whole reason gphiv was added.  Uniform-dlat grids ignore
    lat_face (latlon.py:1319), so this is the case that catches a face sign-flip /
    off-by-one / N-S swap in the ``2*lat_1d[0]-gphiv[0]`` reflection."""
    from legoesm import constants
    R, Om = constants.R_earth, constants.Omega
    # Mercator: uniform in the Mercator y-coordinate -> stretched latitude faces.
    y = -0.6 + 0.18 * np.arange(TNY + 1)              # (TNY+1,) uniform Mercator y
    lat_face = 2.0 * np.arctan(np.exp(y)) - np.pi / 2.0   # (TNY+1,) rad, stretched
    lat_1d = 0.5 * (lat_face[:-1] + lat_face[1:])     # centres = face midpoints
    lon_deg = LON0 + DLON_D * np.arange(TNX)
    gphit = np.rad2deg(lat_1d)[:, None] * np.ones((1, TNX))
    glamt = lon_deg[None, :] * np.ones((TNY, 1))
    gphiv = np.rad2deg(lat_face[1:])[:, None] * np.ones((1, TNX))   # north faces
    e1t = (R * np.deg2rad(DLON_D) * np.cos(lat_1d))[:, None] * np.ones((1, TNX))
    e2t = (R * (lat_face[1:] - lat_face[:-1]))[:, None] * np.ones((1, TNX))  # exact faces
    ff_t = (2.0 * Om * np.sin(lat_1d))[:, None] * np.ones((1, TNX))
    e3t_1d = np.array([10.0, 20.0, 30.0, 40.0])
    depth_cum = np.cumsum(e3t_1d)
    grid = NemoGrid(
        glamt=glamt, gphit=gphit, e1t=e1t, e2t=e2t, e1u=e1t.copy(), e2v=e2t.copy(),
        ff_t=ff_t, ff_f=ff_t.copy(),
        e3t_1d=e3t_1d, gdept_1d=depth_cum - e3t_1d / 2.0,
        gdepw_1d=np.concatenate([[0.0], depth_cum[:-1]]),
        tmask=np.ones((TNY, TNX, TNZ)), umask=np.ones((TNY, TNX, TNZ)),
        vmask=np.ones((TNY, TNX, TNZ)), gphiv=gphiv,
    )
    rng = np.random.default_rng(2)
    state = NemoState(
        T=15.0 + rng.random((TNY, TNX, TNZ)), S=35.0 + rng.random((TNY, TNX, TNZ)),
        u=rng.random((TNY, TNX, TNZ)), v=rng.random((TNY, TNX, TNZ)),
        ssh=0.01 * rng.random((TNY, TNX)), rhd=None,
    )
    return grid, state, lat_face


def test_topo_bridge_variable_dlat_uses_faces():
    """On a stretched (Mercator) grid, dy_T is driven by the reconstructed
    lat_face; it must match NEMO e2t = R·Δ(lat_face) to roundoff.  A flipped /
    off-by-one south-face reflection makes dy_T[0] wrong and fails here."""
    from legoesm import constants
    grid, state, lat_face = _synthetic_topo_mercator()
    out = bridge_nemo_to_legoesm_topo(grid, state, periodic_i=True)
    dy_T = np.asarray(out.geometry.dy_T)
    # Guard against the uniform-dlat branch silently taking over (finding #2):
    # the true e2t must actually vary across latitude on this grid.
    assert grid.e2t.max() - grid.e2t.min() > 1e-3 * grid.e2t.max()
    assert np.max(np.abs(dy_T - grid.e2t)) < 1e-6 * grid.e2t.max()
    # And the southernmost row (driven by the reflected south face) is correct.
    e2t_south = constants.R_earth * (lat_face[1] - lat_face[0])
    assert abs(float(dy_T[0, 0]) - e2t_south) < 1e-6 * e2t_south


# ---------------------------------------------------------------------------
# bridge_before_state_topo (#1317 leap-frog before-level bridge)
# ---------------------------------------------------------------------------
def _synthetic_before(grid, rng_seed=3):
    rng = np.random.default_rng(rng_seed)
    ny, nx, nz = grid.tmask.shape
    return NemoBeforeState(
        T=15.0 + rng.random((ny, nx, nz)), S=35.0 + rng.random((ny, nx, nz)),
        u=rng.random((ny, nx, nz)), v=rng.random((ny, nx, nz)),
        ssh=0.01 * rng.random((ny, nx)),
        tau_x=0.1 * rng.random((ny, nx)), tau_y=0.1 * rng.random((ny, nx)),
    )


def test_bridge_before_state_topo_populates_before_fields():
    grid, state, _ = _synthetic_topo()
    br = bridge_nemo_to_legoesm_topo(grid, state, periodic_i=True)
    before = _synthetic_before(grid)

    st = bridge_before_state_topo(br, grid, before, periodic_i=True)
    assert st.T_before is not None and st.T_before.data.shape == (TNY, TNX, TNZ)
    assert st.S_before is not None
    assert st.u_before.data.shape == br.state.u.data.shape
    assert st.v_before.data.shape == br.state.v.data.shape
    assert st.eta_before.data.shape == br.state.eta.data.shape
    assert st.tau_x_prev is not None
    assert st.tau_y_prev is not None

    # SAME staggering convention as the now-level bridge: NEMO u -> west-wall
    # prepend (or periodic wrap), NEMO v -> south-wall prepend.
    wet3 = grid.tmask > 0.5
    u_before = np.asarray(st.u_before.data)
    assert np.allclose(u_before[:, 1:, :][wet3], before.u[wet3])
    v_before = np.asarray(st.v_before.data)
    assert np.allclose(v_before[1:, :, :][wet3], before.v[wet3])
    assert np.allclose(np.asarray(st.eta_before.data), before.ssh)
    assert np.allclose(np.asarray(st.tau_x_prev), before.tau_x)
    assert np.allclose(np.asarray(st.tau_y_prev), before.tau_y)

    # T/S at wet cells match the raw restart exactly (only dry cells are
    # Neumann-filled, same as the now-level T/S bridge).
    assert np.allclose(np.asarray(st.T_before.data)[wet3], before.T[wet3])
    assert np.allclose(np.asarray(st.S_before.data)[wet3], before.S[wet3])

    # now-level fields are untouched (this function is purely additive).
    assert np.allclose(np.asarray(st.T.data), np.asarray(br.state.T.data))


def test_bridge_before_state_topo_missing_tau_leaves_prev_none():
    """A restart without utau_b/vtau_b (NemoBeforeState.tau_x/tau_y=None)
    leaves state.tau_x_prev/tau_y_prev at their None default -- the
    _leapfrog_step Euler-start branch (gated on state.u_before, not
    tau_x_prev) still seeds "before := now" for the centred-forcing carry
    on step 1, matching the NEMO nit000 convention."""
    grid, state, _ = _synthetic_topo()
    br = bridge_nemo_to_legoesm_topo(grid, state, periodic_i=True)
    before = _synthetic_before(grid)._replace(tau_x=None, tau_y=None)

    st = bridge_before_state_topo(br, grid, before, periodic_i=True)
    assert st.u_before is not None   # velocity/tracer before-state still set
    assert st.tau_x_prev is None
    assert st.tau_y_prev is None


def test_bridge_before_state_topo_closed_basin_u_wall():
    grid, state, _ = _synthetic_topo()
    br = bridge_nemo_to_legoesm_topo(grid, state, periodic_i=False)
    before = _synthetic_before(grid)
    st = bridge_before_state_topo(br, grid, before, periodic_i=False)
    u_before = np.asarray(st.u_before.data)
    assert np.allclose(u_before[:, 0, :], 0.0)   # closed -> west wall, not periodic
