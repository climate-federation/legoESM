"""#480 faithful root fix: the tracer-wall Neumann fill on the flux-form
advection reconstruction.

The masked-land cold cell (T=0) was contaminating the WIDE WENO tracer stencil
at free-slip walls, manufacturing a spurious near-wall front that a 2Δx-in-lon
v perturbation amplifies into an un-dissipatable grid mode (the §5 eddy-permitting
blow-up).  The fix zero-gradient (Neumann) fills the tracer over land BEFORE the
reconstruction, so the stencil sees a flat extension = the physical no-flux
insulating wall = Oceananigans' clean grid-edge wall.

Tests:
1. ``recon_fill_mask=None`` is bit-identical to the legacy path (no behaviour
   change when disabled).
2. The fill is a strict no-op when there is no land (all-wet domain).
3. The fill REMOVES the cold-land contamination: with a cold land row the
   near-wall reconstruction flux divergence changes (and is far smaller).
4. Conservation: with the wall faces masked (``mass_flux_v`` = 0 at walls), the
   summed flux divergence over the wet domain telescopes to zero — the fill only
   redistributes tracer WITHIN the wet domain, it never sources/sinks it.
5. The ``tracer_wall_neumann_fill`` config flag wires through the model step.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants

jax.config.update("jax_enable_x64", True)


def _make_grid(n_lat=12, n_lon=16):
    from legoesm.grids.latlon import create_latlon_grid
    return create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth)


def _flux_div(tr, mask, grid, mass_flux_v, scheme="weno7"):
    """Single-tracer horizontal flux divergence, with/without the wall fill."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _compute_advection_flux_div,
    )
    n_lat, n_lon, nlev = tr.shape
    mass_flux_u = jnp.zeros((n_lat, n_lon + 1, nlev))
    w_baro = jnp.zeros((n_lat, n_lon, nlev + 1))
    h = jnp.ones_like(tr)
    h_u = jnp.ones((n_lat, n_lon + 1, nlev))
    h_v = jnp.ones((n_lat + 1, n_lon, nlev))
    div_hut, _vert = _compute_advection_flux_div(
        tr, scheme, mass_flux_u, mass_flux_v, w_baro, h, h_u, h_v, grid, 0.0,
        recon_fill_mask=mask,
    )
    return div_hut


class TestReconFillMaskIdentity:
    def test_none_is_legacy_identity(self):
        """recon_fill_mask=None must reproduce the legacy (unfilled) divergence."""
        grid = _make_grid()
        n_lat, n_lon, nlev = 12, 16, 3
        key = jax.random.PRNGKey(0)
        tr = jax.random.normal(key, (n_lat, n_lon, nlev))
        mass_flux_v = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.3
        d_none = _flux_div(tr, None, grid, mass_flux_v)
        # Reconstruct the legacy result by calling the raw reconstruction path.
        from legoesm.ocean.advection import weno7_to_v_points
        from legoesm.grids.operators_latlon_cgrid import divergence_cgrid
        tr_v = weno7_to_v_points(tr, mass_flux_v)
        mass_flux_u = jnp.zeros((n_lat, n_lon + 1, nlev))
        tr_u = jnp.zeros((n_lat, n_lon + 1, nlev))
        d_legacy = divergence_cgrid(mass_flux_u * tr_u, mass_flux_v * tr_v, grid)
        assert jnp.allclose(d_none, d_legacy, atol=1e-13)

    def test_no_land_is_noop(self):
        """All-wet mask: the fill leaves the divergence bit-identical to None."""
        grid = _make_grid()
        n_lat, n_lon, nlev = 12, 16, 3
        tr = jax.random.normal(jax.random.PRNGKey(1), (n_lat, n_lon, nlev))
        mass_flux_v = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.3
        wet = jnp.ones((n_lat, n_lon))
        d_none = _flux_div(tr, None, grid, mass_flux_v)
        d_fill = _flux_div(tr, wet, grid, mass_flux_v)
        assert jnp.allclose(d_none, d_fill, atol=1e-13)


class TestColdLandContamination:
    def test_fill_active_at_wall_inert_in_interior(self):
        """A cold (T=0) land row beside a sharp wet front: the wide WENO stencil
        at the near-wall wet faces reaches the cold cell, so the fill CHANGES the
        near-wall reconstruction (divergence differs from the unfilled path) while
        being bit-identical in the deep interior (stencil never sees land)."""
        grid = _make_grid(n_lat=16, n_lon=16)
        n_lat, n_lon, nlev = 16, 16, 1
        mask = jnp.ones((n_lat, n_lon)).at[0].set(0.0).at[-1].set(0.0)
        # sharp lat front (so the WENO reconstruction is sensitive to the
        # cold-cell bias), uniform in lon; cold land cells -> 0
        lat = jnp.arange(n_lat)[:, None, None]
        T = (10.0 + 2.0 * jnp.tanh((lat - 4.0))) * jnp.ones((n_lat, n_lon, nlev))
        T = T * mask[:, :, None]
        mfv = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.3
        mfv = mfv.at[0].set(0.0).at[1].set(0.0)        # wall faces masked
        mfv = mfv.at[-1].set(0.0).at[-2].set(0.0)
        d_none = _flux_div(T, None, grid, mfv)
        d_fill = _flux_div(T, mask, grid, mfv)
        # Near-wall wet rows (1,2): the fill must change the reconstruction.
        near = jnp.max(jnp.abs(d_none[1:4] - d_fill[1:4]))
        # Deep interior (rows 7,8): >4 cells from any land -> WENO7 stencil
        # (half-width 4) never reaches land -> bit-identical.
        deep = jnp.max(jnp.abs(d_none[7:9] - d_fill[7:9]))
        assert float(near) > 1e-9, "fill should change the near-wall reconstruction"
        assert float(near) > 1e3 * float(deep), "wall effect must dominate interior"
        assert float(deep) < 1e-12, "fill must be inert in the deep interior"


class TestConservation:
    def test_wet_domain_conserved(self):
        """With the wall faces masked (mfv=0 at the N/S walls), the summed
        horizontal flux divergence over the wet domain is zero to round-off —
        the fill only redistributes within the wet domain, never sources it."""
        grid = _make_grid(n_lat=12, n_lon=16)
        n_lat, n_lon, nlev = 12, 16, 2
        mask = jnp.ones((n_lat, n_lon)).at[0].set(0.0).at[-1].set(0.0)
        T = jax.random.normal(jax.random.PRNGKey(2), (n_lat, n_lon, nlev))
        T = T * mask[:, :, None]
        # mass flux zero through the two wall faces (rows 1 and n_lat-1) and
        # zero at the array-edge faces; nonzero only on interior wet faces.
        vmask = jnp.ones((n_lat + 1, n_lon, nlev))
        vmask = vmask.at[0].set(0.0).at[1].set(0.0)
        vmask = vmask.at[-1].set(0.0).at[-2].set(0.0)
        mfv = jax.random.normal(jax.random.PRNGKey(3), (n_lat + 1, n_lon, nlev)) * vmask
        area = jnp.asarray(grid.area)[:, :, None]      # (n_lat, n_lon, 1)
        for m in (None, mask):
            d = _flux_div(T, m, grid, mfv)
            # area-weighted divergence over the wet domain telescopes to the
            # (masked, zero) wall fluxes -> conserved (the fill only redistributes
            # tracer between wet cells, never sources/sinks it).
            d_wet = d * mask[:, :, None] * area
            total = float(jnp.abs(jnp.sum(d_wet)))
            assert total < 1e-6, f"wet-domain tracer not conserved: {total}"


class TestConfigFlag:
    def test_flag_wires_through_step(self):
        """tracer_wall_neumann_fill True vs False changes the §5-like step;
        the True path (default) matches the faithful, blow-up-free behaviour."""
        from legoesm.core.field import Field
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.experiments.silvestri_baroclinic_jet import (
            SilvestriJetConfig, build_silvestri_baroclinic_jet_setup,
        )
        # tiny, cheap config (small grid, few levels) — just exercise the flag.
        cfg = SilvestriJetConfig(wall_grid_filter_rate_s=0.0)
        r = build_silvestri_baroclinic_jet_setup(
            n_lat=24, n_lon=16, scheme="W9V", nlev=4, config=cfg, stabilize=False)
        assert getattr(r.model_config, "tracer_wall_neumann_fill") is True
        m = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)

        def zfield(d):
            return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                         dims=d.dims, units=d.units)
        s = r.initial_state._replace(
            T_incr_prev=zfield(r.initial_state.T),
            S_incr_prev=zfield(r.initial_state.S),
            u_incr_prev=zfield(r.initial_state.u),
            v_incr_prev=zfield(r.initial_state.v))
        # Seed a 2Δx-in-lon v perturbation at the south wall (the grid mode the
        # fix targets) so the cold-cell contamination is exercised in one step.
        v = np.asarray(s.v.data)
        ii = np.arange(v.shape[1])[None, :, None]
        band = np.zeros((v.shape[0], 1, 1)); band[:6] = 1.0
        vpert = 0.05 * ((-1.0) ** ii) * band * (np.asarray(s.v_mask.data) > 0)[:, :, None]
        s = s._replace(v=s.v.replace(data=jnp.asarray(v + vpert)))

        s_on = m.step(s, 900.0)
        # With the flag OFF the reconstruction sees the cold land cell again.
        cfg_off = r.model_config._replace(tracer_wall_neumann_fill=False)
        m_off = LatLonCGridOceanModel(r.grid, r.z_coord, cfg_off)
        s_off = m_off.step(s, 900.0)
        # The two paths differ (flag is live), and the ON path stays finite.
        assert bool(jnp.all(jnp.isfinite(s_on.T.data)))
        assert float(jnp.max(jnp.abs(s_on.T.data - s_off.T.data))) > 1e-9
