"""NEMO ``key_linssh`` (linear free surface) coordinate mode.

Unit gates for the three choke points (jacobian, layer thickness, diagnosed w)
plus a GYRE one-step integration smoke. The flag-off path must remain the
unchanged z* behaviour (golden-gate property — asserted directly here against
the analytic z* formulas).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np

from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
    diagnose_w_from_flux_div,
)


def _coord(linssh: bool, n=4, H=100.0):
    dz = jnp.full((n,), H / n)
    z_half = -jnp.concatenate([jnp.zeros(1), jnp.cumsum(dz)])
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    return OceanZStarCoordinate(
        n_levels=n, H_max=H, z_full_ref=z_full, z_half_ref=z_half,
        dz_ref=dz, dz_half_ref=z_full[:-1] - z_full[1:],
        linear_free_surface=linssh,
    )


def test_jacobian_frozen_at_reference():
    eta = jnp.asarray([[0.5, -0.3], [1.0, 0.0]])
    H_bathy = 100.0
    zs = compute_ocean_jacobian(eta, H_bathy, _coord(False))
    ls = compute_ocean_jacobian(eta, H_bathy, _coord(True))
    # z*: J = (eta+H)/H varies with eta; linssh: J == H_bathy/H_max == 1 always
    np.testing.assert_allclose(np.asarray(zs), (np.asarray(eta) + 100.0) / 100.0)
    np.testing.assert_allclose(np.asarray(ls), 1.0, rtol=0, atol=0)


def test_layer_thickness_frozen():
    eta = jnp.asarray([[2.0]])
    h_ls = compute_layer_thickness(eta, 100.0, _coord(True))
    h_ref = compute_layer_thickness(jnp.zeros_like(eta), 100.0, _coord(False))
    np.testing.assert_allclose(np.asarray(h_ls), np.asarray(h_ref), rtol=0)


def test_w_skips_sigma_redistribution():
    """linssh w == the fixed-thickness continuity w (w_euler): surface w =
    deta/dt, bottom w = 0, and it differs from the z* w by exactly
    sigma*deta_dt at interior interfaces."""
    rng = np.random.default_rng(0)
    flux_div = jnp.asarray(rng.standard_normal((3, 3, 4)) * 1e-5)
    w_ls = diagnose_w_from_flux_div(flux_div, _coord(True),
                                    thickness_weighted=True)
    w_zs = diagnose_w_from_flux_div(flux_div, _coord(False),
                                    thickness_weighted=True)
    # bottom interface exactly zero in linssh (fixed column)
    np.testing.assert_allclose(np.asarray(w_ls)[..., -1], 0.0, atol=0)
    # surface w = deta/dt = -sum_k div(h u)
    np.testing.assert_allclose(
        np.asarray(w_ls)[..., 0],
        -np.asarray(flux_div).sum(axis=-1), rtol=1e-12)
    # z* differs by sigma * deta_dt (the redistribution the gate removes)
    coord = _coord(False)
    sigma = np.asarray((coord.z_half_ref + coord.H_max) / coord.H_max)
    deta = np.asarray(w_ls)[..., 0:1]
    np.testing.assert_allclose(
        np.asarray(w_zs), np.asarray(w_ls) - sigma * deta, rtol=1e-12)


def test_gyre_linssh_one_step_finite():
    """The GYRE recipe with the linssh coordinate constructs and steps finite
    (the harness GYRE_LINSSH toggle path)."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import (
        _NEMO_GYRE_DT_S,
        build_nemo_gyre_recipe,
        nemo_gyre_wind_forcing,
    )

    r = build_nemo_gyre_recipe()
    r = r._replace(z_coord=r.z_coord._replace(linear_free_surface=True))
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)
    st = r.initial_state
    n_lat, n_lon = st.T.data.shape[0], st.T.data.shape[1]
    new = model.step(
        st, dt=_NEMO_GYRE_DT_S,
        surface_forcing=nemo_gyre_wind_forcing(n_lat, n_lon, 0.0))
    for f in (new.T.data, new.S.data, new.u.data, new.v.data, new.eta.data,
              new.w.data):
        assert bool(jnp.all(jnp.isfinite(f)))
    # NB the state w is CELL-CENTRE (midpoint of interfaces): the deepest cell
    # carries w_iface[-2]/2, legitimately nonzero under flow. The linssh
    # interface invariant (w_iface[-1] == 0 exactly) is unit-tested directly in
    # test_w_skips_sigma_redistribution above.
    # eta must still evolve (linear free surface, not rigid lid).
    assert float(jnp.max(jnp.abs(new.eta.data))) > 0.0


def test_linssh_surface_dilution_sign():
    """SIGN GATE (CLAUDE.md sign mandate): the linssh top-cell flux must
    DILUTE — a rising surface (w0 = deta/dt > 0, water entering the column)
    removes content from the frozen top cell: F[0] = w0*T0 > 0 (positive-up
    flux convention, vert_flux_div[0] = F[0]-F[1]), so
    d(h0*T0)/dt = -w0*T0 < 0.  Flag-off: F[0] == 0 exactly."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _compute_advection_flux_div,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    r = build_nemo_gyre_recipe()
    grid = r.grid
    st = r.initial_state
    n_lat, n_lon = st.T.data.shape[0], st.T.data.shape[1]
    nlev = st.T.data.shape[2]
    T0 = 10.0
    tr = jnp.full((n_lat, n_lon, nlev), T0)
    # zero horizontal transport; prescribed rising surface w0 > 0
    mfu = jnp.zeros((n_lat, n_lon + 1, nlev))
    mfv = jnp.zeros((n_lat + 1, n_lon, nlev))
    w0 = 1.0e-4
    w_baro = jnp.zeros((n_lat, n_lon, nlev + 1)).at[..., 0].set(w0)
    h = jnp.broadcast_to(r.z_coord.dz_ref, (n_lat, n_lon, nlev))
    hu = jnp.broadcast_to(r.z_coord.dz_ref, (n_lat, n_lon + 1, nlev))
    hv = jnp.broadcast_to(r.z_coord.dz_ref, (n_lat + 1, n_lon, nlev))

    for flag, expect in ((True, w0 * T0), (False, 0.0)):
        _, vfd = _compute_advection_flux_div(
            tr, "centered", mfu, mfv, w_baro, h, hu, hv, grid,
            dt=100.0, linssh_top_flux=flag)
        np.testing.assert_allclose(
            np.asarray(vfd)[..., 0], expect, rtol=0, atol=1e-15,
            err_msg=f"linssh_top_flux={flag}: F[0] must be {expect} "
                    "(dilution sign; a flip would ANTI-dilute)")


def test_barotropic_coriolis_split_validation_and_nonvacuity():
    """barotropic_coriolis_split: typo raises; "live" requires explicit_ab2 +
    explicit_substep; and the live split actually changes the step (non-vacuous)
    while staying finite.

    SCOPE, stated precisely (codex, #1388): the non-vacuity half is proven for
    the constructed ``ene`` + ``nemo_boxcar_centred`` pairing and through ``u``
    only. It does NOT cover the GYRE/AB3-AM4 recipe (that combination is
    rejected outright, which this test now also asserts) nor the more exact
    DINO ``nemo_boxcar_ab3`` path, which carries different outer-integrator
    requirements."""
    import pytest

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import (
        _NEMO_GYRE_DT_S,
        build_nemo_gyre_recipe,
        nemo_gyre_wind_forcing,
    )

    r = build_nemo_gyre_recipe()
    # the card moved to ene_total (planetary INSIDE the vorticity flux), which
    # requires the "frozen" split (the live subtraction is the face-f form).
    assert r.model_config.barotropic_coriolis_split == "frozen"
    assert r.model_config.vorticity_scheme == "ene_total"

    with pytest.raises(ValueError, match="barotropic_coriolis_split"):
        LatLonCGridOceanModel(
            r.grid, r.z_coord,
            r.model_config._replace(barotropic_coriolis_split="typo"))
    with pytest.raises(ValueError, match="explicit_ab2"):
        LatLonCGridOceanModel(
            r.grid, r.z_coord,
            r.model_config._replace(coriolis_scheme="matsuno_split"))
    # ene_total + live is stencil-inconsistent -> rejected
    with pytest.raises(ValueError, match="ene_total"):
        LatLonCGridOceanModel(
            r.grid, r.z_coord,
            r.model_config._replace(barotropic_coriolis_split="live"))
    # live remains valid on the face-f split form ("ene" relative-only card).
    # The recipe now selects the AB3-AM4 filter, which the live split rejects
    # (the AB3 substep applies live Coriolis to the EXTRAPOLATED U_mid while the
    # pre-step subtraction uses the plain pre-step U_bar, so substep 0 would not
    # cancel bit-exactly). Take the guard's own instruction and pair live with
    # the boxcar filter, NEMO's DINO selection (#1388).
    _boxcar = r.model_config.barotropic._replace(
        barotropic_time_filter="nemo_boxcar_centred")
    _live_ok = r.model_config._replace(
        vorticity_scheme="ene", barotropic_coriolis_split="live",
        barotropic=_boxcar)
    LatLonCGridOceanModel(r.grid, r.z_coord, _live_ok)
    # ... and the pairing guard itself is live: AB3-AM4 + live must still raise.
    with pytest.raises(ValueError, match="nemo_ab3am4"):
        LatLonCGridOceanModel(
            r.grid, r.z_coord,
            r.model_config._replace(vorticity_scheme="ene",
                                    barotropic_coriolis_split="live"))

    st = r.initial_state
    n_lat, n_lon = st.T.data.shape[0], st.T.data.shape[1]
    sf = nemo_gyre_wind_forcing(n_lat, n_lon, 0.0)
    cfg_ene = r.model_config._replace(vorticity_scheme="ene",
                                      barotropic=_boxcar)
    m_live = LatLonCGridOceanModel(
        r.grid, r.z_coord, cfg_ene._replace(barotropic_coriolis_split="live"))
    m_frozen = LatLonCGridOceanModel(r.grid, r.z_coord, cfg_ene)
    a = m_live.step(st, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    b = m_frozen.step(st, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    for f in (a.u.data, a.v.data, a.eta.data):
        assert bool(jnp.all(jnp.isfinite(f)))
    # non-vacuous: live vs frozen differ once the flow is nonzero (after the
    # first step u=0 -> the pre-step subtraction is 0 and live f x U acts on
    # the in-window transport; one more step from the evolved state).
    a2 = m_live.step(a, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    b2 = m_frozen.step(b, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    assert float(jnp.max(jnp.abs(a2.u.data - b2.u.data))) > 0.0
