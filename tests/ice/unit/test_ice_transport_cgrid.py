"""Sea-ice tracer transport on the lat-lon C-grid geometry (tripole family).

The tripole eORCA grid (``LatLonCGridGeometry``) previously degraded to
``transport='none'`` (no ice advection at all — no Fram/Bering export).  The
new C-grid branch (``transport.fv_flux_divergence_latlon_cgrid``) advects the
conserved ice scalars with the fold-aware donor-cell upwind + the ocean's own
conservative ``divergence_cgrid``, driven by the cell-centred geographic
free-drift velocities rotated onto the faces.

These tests pin:
  * exact global conservation of the advected inventory (sum q*area_T) on the
    regular C-grid geometry AND with an ACTIVE synthetic north fold (the seam
    flux pairs must cancel — the fold-parity hazard);
  * monotonicity/positivity of the donor-cell scheme;
  * zero-velocity no-op;
  * the transport/dynamics capability split (transport-capable but
    dynamics-INcapable) + step_sea_ice wiring end-to-end on the geometry.

Run under JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import FoldDescriptor, create_latlon_geometry
from legoesm.ice.sea_ice import (
    grid_supports_ice_dynamics,
    grid_supports_ice_transport,
)
from legoesm.ice.transport import (
    advect_ice_tracers,
    fv_flux_divergence_latlon_cgrid,
)

N_LAT, N_LON = 16, 12


def _geometry(fold: bool = False):
    g = create_latlon_geometry(N_LAT, N_LON)
    if fold:
        rev = jnp.asarray(np.arange(N_LON)[::-1].copy(), dtype=jnp.int32)
        g = g._replace(fold=FoldDescriptor(
            is_active=True, fold_j=N_LAT - 1, cap_j=N_LAT - 1,
            perm_T=rev, perm_v=rev,
            vector_sign_u=-1.0, vector_sign_v=-1.0))
        # NON-VACUOUS seam (codex): the regular geometry's north v-face row
        # sits at the pole (dx_v = 0), which silences the seam flux and made
        # the original fold test pass trivially.  Give the seam a nonzero,
        # perm-symmetric metric (constant in lon, like a real fold row of
        # duplicated physical faces) so pair cancellation is actually
        # exercised.
        dx_v = jnp.asarray(g.dx_v)
        seam_dx = jnp.full((g.dx_v.shape[1],), float(jnp.mean(dx_v[1:-1])))
        g = g._replace(dx_v=dx_v.at[-1].set(seam_dx))
    return g


def _rng_fields(seed=0):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.uniform(0.0, 1.0, (N_LAT, N_LON)))
    u = jnp.asarray(rng.uniform(-0.3, 0.3, (N_LAT, N_LON)))
    v = jnp.asarray(rng.uniform(-0.3, 0.3, (N_LAT, N_LON)))
    return q, u, v


@pytest.mark.parametrize("fold", [False, True])
def test_cgrid_tendency_conserves_global_integral(fold):
    """Flux-form telescoping: sum(dq/dt * area_T) == 0 to round-off — on the
    regular geometry (wall poles + periodic lon) AND with the ACTIVE synthetic
    north fold, where the seam flux pairs must cancel (the fold-parity
    hazard this scheme was built around)."""
    g = _geometry(fold=fold)
    q, u, v = _rng_fields(3)
    tend = fv_flux_divergence_latlon_cgrid(q, u, v, g)
    area = np.asarray(g.area_T)
    total = float((np.asarray(tend) * area).sum())
    scale = float((np.abs(np.asarray(tend)) * area).sum()) + 1e-30
    assert abs(total) / scale < 1e-12, (total, scale)
    assert np.all(np.isfinite(np.asarray(tend)))


def test_cgrid_advection_conserves_and_stays_monotone():
    """advect_ice_tracers on the C-grid geometry conserves total ice volume
    (sum h*a*area_T) to round-off and keeps 0 <= conc <= 1 (donor-cell
    monotonicity) over many substeps."""
    g = _geometry(fold=False)
    rng = np.random.default_rng(7)
    conc = jnp.asarray(rng.uniform(0.0, 1.0, (N_LAT, N_LON)))
    h = jnp.asarray(rng.uniform(0.0, 2.0, (N_LAT, N_LON)))
    T = jnp.full((N_LAT, N_LON), 260.0)
    u = jnp.asarray(rng.uniform(-0.2, 0.2, (N_LAT, N_LON)))
    v = jnp.asarray(rng.uniform(-0.2, 0.2, (N_LAT, N_LON)))
    area = np.asarray(g.area_T)
    vol0 = float((np.asarray(h * conc) * area).sum())
    h2, c2, T2 = advect_ice_tracers(h, conc, T, u, v, g, dt=1800.0,
                                    n_subcycles=4)
    vol1 = float((np.asarray(h2 * c2) * area).sum())
    np.testing.assert_allclose(vol1, vol0, rtol=1e-12)
    assert float(jnp.min(c2)) >= 0.0
    assert float(jnp.max(c2)) <= 1.0 + 1e-12
    assert np.all(np.isfinite(np.asarray(h2)))
    assert np.all(np.isfinite(np.asarray(T2)))


def test_cgrid_advection_zero_velocity_is_identity():
    g = _geometry(fold=False)
    q, _, _ = _rng_fields(11)
    z = jnp.zeros_like(q)
    tend = fv_flux_divergence_latlon_cgrid(q, z, z, g)
    np.testing.assert_allclose(np.asarray(tend), 0.0, atol=1e-15)


def test_capability_split_transport_yes_dynamics_no():
    """LatLonCGridGeometry: transport-capable, dynamics-INcapable — mEVP must
    still be rejected (no curvilinear strain-rate ops) while 'advect' passes
    the transport gate."""
    g = _geometry(fold=False)
    assert grid_supports_ice_transport(g) is True
    assert grid_supports_ice_dynamics(g) is False
    # And every dynamics-capable grid remains transport-capable (superset).
    from legoesm.grids.latlon import create_latlon_grid
    a_grid = create_latlon_grid(8, 16)
    assert grid_supports_ice_dynamics(a_grid) is True
    assert grid_supports_ice_transport(a_grid) is True


def test_step_sea_ice_advects_on_cgrid_geometry():
    """End-to-end: step_sea_ice(dynamics='free_drift', transport='advect') on
    the C-grid geometry runs FINITE and actually MOVES ice (wind-driven free
    drift + live advection), and mEVP still raises (no strain-rate ops).
    Exact transport conservation is pinned separately above — a full step
    also runs thermodynamic sources, so no budget assert here."""
    from legoesm import constants
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ice import SeaIceConfig, init_dynamic_ice_state, step_sea_ice
    from legoesm.ice.config import BrineConfig

    g = _geometry(fold=False)
    shape = (N_LAT, N_LON)
    cfg = SeaIceConfig(dynamics="free_drift", transport="advect",
                       brine=BrineConfig(enabled=True))
    st = init_dynamic_ice_state(shape, S_ice_init=0.0)
    rng = np.random.default_rng(5)
    st = st._replace(
        h_ice=st.h_ice.replace(data=jnp.asarray(
            rng.uniform(0.2, 1.5, shape))),
        concentration=st.concentration.replace(data=jnp.asarray(
            rng.uniform(0.1, 0.9, shape))),
        T_ice=st.T_ice.replace(data=jnp.full(shape, 262.0)),
    )
    o = jnp.ones(shape)
    atm = AtmToSurface(
        sw_down=jnp.zeros(shape), lw_down=302.0 * o,   # ~LW balance at 262 K
        precip_total=jnp.zeros(shape), precip_snow=jnp.zeros(shape),
        T_lowest=262.0 * o, q_lowest=1.5e-3 * o,
        u_lowest=3.0 * o, v_lowest=1.0 * o,
        p_lowest=1.0e5 * o, p_surface=1.0e5 * o, rho_lowest=1.3 * o,
        cos_zenith=jnp.zeros(shape), co2_ppmv=jnp.asarray(400.0),
        has_radiation=jnp.asarray(1.0), has_precipitation=jnp.asarray(1.0),
    )
    sst = jnp.full(shape, float(constants.T_freeze_ocean))
    new_st, resp = step_sea_ice(st, atm, sst, jnp.zeros(shape),
                                jnp.zeros(shape), cfg, U_min=0.0, dt=1800.0,
                                grid=g)
    for arr in (new_st.h_ice.data, new_st.concentration.data,
                resp.ocean_heat_extraction, resp.salt_flux,
                resp.freshwater_flux):
        assert np.all(np.isfinite(np.asarray(arr)))
    # Ice moved (advection is live: nonzero wind-driven free drift).
    assert not np.allclose(np.asarray(new_st.concentration.data),
                           np.asarray(st.concentration.data))
    # mEVP still rejected on this geometry (strain-rate ops absent).
    cfg_mevp = SeaIceConfig(dynamics="mevp", transport="advect",
                            brine=BrineConfig(enabled=True))
    with pytest.raises(ValueError, match="strain-rate"):
        step_sea_ice(st, atm, sst, jnp.zeros(shape), jnp.zeros(shape),
                     cfg_mevp, U_min=0.0, dt=1800.0, grid=g)


def test_seam_flux_shared_donor_pair_cancellation_and_positivity():
    """codex r2 #1: the seam flux must be ONE shared donor-cell upwind flux
    per fold pair — no extraction from an EMPTY partner cell (the bare flux
    antisymmetrization violated this: a one-sided seam pair leaked inventory).

    The velocity is COLUMN-LOCALIZED (only column i moves) so the two fold
    partners disagree — v_pair = (v(i) - v(p))/2 is genuinely nonzero — a
    uniform flow on the synthetic identity-angle fold would antisymmetrize to
    zero and make the test vacuous."""
    g = _geometry(fold=True)
    perm = np.asarray(g.fold.perm_T)
    i = 2
    p = int(perm[i])
    assert p != i
    q = jnp.zeros((N_LAT, N_LON)).at[-1, i].set(1.0)   # only cell i holds ice
    u = jnp.zeros((N_LAT, N_LON))
    v_out = jnp.zeros((N_LAT, N_LON)).at[:, i].set(0.4)   # col-i northward
    area = np.asarray(g.area_T)

    # Flow OUT of donor cell i across the seam: the empty partner may only
    # RECEIVE ice.
    tend = np.asarray(fv_flux_divergence_latlon_cgrid(q, u, v_out, g))
    assert tend[-1, p] > 1e-15, "seam exchange did not fire (vacuous setup)"
    # Flow reversed (INTO cell i from the empty partner's side): the shared
    # donor is the EMPTY partner -> zero seam flux -> partner untouched (this
    # is exactly the inventory leak of the naive antisymmetrization).
    tend_rev = np.asarray(fv_flux_divergence_latlon_cgrid(q, u, -v_out, g))
    assert abs(tend_rev[-1, p]) < 1e-15, tend_rev[-1, p]
    # Global conservation exact in both flow directions.
    for t in (tend, tend_rev):
        total = float((t * area).sum())
        scale = float((np.abs(t) * area).sum()) + 1e-30
        assert abs(total) / scale < 1e-12 or scale < 1e-20, (total, scale)


_EORCA_MESH = "data/grids/eORCA1.2_mesh_mask.nc"


@pytest.mark.skipif(
    not __import__("pathlib").Path(_EORCA_MESH).is_file(),
    reason="eORCA1.2 mesh not present on host")
def test_cgrid_tendency_conserves_on_real_eorca_mesh():
    """REAL-mesh closure (codex repro case): constant q = 1 under a uniform
    geographic eastward flow on the actual eORCA1.2 tripole geometry must
    integrate to ZERO global tendency — this is where the synthetic fold was
    blind (its seam metric was zero) and where the un-symmetrized seam flux
    leaked ~3e-4 relative."""
    from legoesm.grids.tripole import create_tripole_grid
    g = create_tripole_grid(_EORCA_MESH)
    shape = (int(g.n_lat), int(g.n_lon))
    q = jnp.ones(shape)
    u = jnp.ones(shape)          # uniform geographic eastward flow
    v = jnp.zeros(shape)
    tend = fv_flux_divergence_latlon_cgrid(q, u, v, g)
    area = np.asarray(g.area_T)
    total = float((np.asarray(tend) * area).sum())
    scale = float((np.abs(np.asarray(tend)) * area).sum()) + 1e-30
    assert abs(total) / scale < 1e-10, (total, scale)
    # And a mixed random flow, random tracer — the general closure.
    rng = np.random.default_rng(1)
    q2 = jnp.asarray(rng.uniform(0.0, 1.0, shape))
    u2 = jnp.asarray(rng.uniform(-0.3, 0.3, shape))
    v2 = jnp.asarray(rng.uniform(-0.3, 0.3, shape))
    tend2 = fv_flux_divergence_latlon_cgrid(q2, u2, v2, g)
    total2 = float((np.asarray(tend2) * area).sum())
    scale2 = float((np.abs(np.asarray(tend2)) * area).sum()) + 1e-30
    assert abs(total2) / scale2 < 1e-10, (total2, scale2)


def test_runner_enables_tripole_ice_transport():
    """Drift guard: the OMIP runner selects transport='advect' from the
    TRANSPORT gate (not the dynamics gate) so tripole ice advects."""
    import inspect
    from scripts.run import run_omip_core2 as R
    src = inspect.getsource(R.main)
    assert "grid_supports_ice_transport(grid)" in src
    assert '_transport = "advect" if _supports_transport else "none"' in src