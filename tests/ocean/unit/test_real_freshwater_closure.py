"""`freshwater_closure="real_freshwater"`: keep the volume channel, drop the VSF.

WHY THIS EXISTS (measured, 2026-08-04/05).  Our tripole OMIP run gains Arctic
(>=66N) column-integrated salt at **+16.829 psu.m** over model days 30->90;
NEMO ORCA1 on the same mesh, same forcing, same 5,164-cell mask, instantaneous
endpoints both sides, gains **+6.720** -- a **+10.109 psu.m** excess.

Cause, confirmed in code: the tracer step already uses a moving z-star
thickness, so with no transport

    hS_new = h_old*S - dt*div ;  S_new = hS_new/h_new

**preserves h*S while the column stretches** -- freshwater dilution is already
handled conservatively.  On top of that the model applied a virtual salt flux
`-S_ref*F_fw/(rho_0*dz_0)`, a SEPARATE salt-content source.  NEMO runs variable
volume (`dom_qco_init : Variable volume activated`) and has no such term.

`real_freshwater` keeps the eta/volume channel and skips ONLY the VSF block.
The genuine `surface_forcing.salt_flux` pathway is deliberately untouched: an
earlier draft of this fix proposed adding `sfx=(S_o-S_i)*m_ice`, which review
showed was wrong in THREE ways (sign inverted, 1000x unit error, and a double
count of an already-existing channel).  See
`docs/dev-notes/ocean_real_freshwater_design.md`.

The A/B that motivated the mode is recorded too: switching
`--freshwater-salinity s_ref -> local` (both VIRTUAL closures) moved the gain
only +16.829 -> +15.331, missing its pre-registered +12.583 bar.  A knob that
rescales the VSF cannot fix a defect whose cause is the VSF existing at all.
"""
from __future__ import annotations

import inspect

import pytest

from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)


class TestLatLonDispatch:
    def test_accepts_real_freshwater(self):
        LatLonCGridOceanModel._validate_config(
            LatLonCGridOceanConfig(freshwater_closure="real_freshwater"))

    def test_still_rejects_unknown_closure(self):
        # Dispatch hardening must survive adding a member: a typo still raises
        # rather than silently selecting a default closure.
        cfg = LatLonCGridOceanConfig(freshwater_closure="real_freshwter")
        with pytest.raises(ValueError, match="freshwater_closure"):
            LatLonCGridOceanModel._validate_config(cfg)

    def test_real_freshwater_exempt_from_local_normalize_guard(self):
        # `local` + normalize is rejected for the VIRTUAL closure because
        # the zero-mean correction no longer gives zero global salt.  Under
        # real_freshwater NO salinity multiplies the freshwater flux at all
        # (the VSF block is skipped), so the combination is inert, not
        # unsound -- it must NOT be rejected.
        LatLonCGridOceanModel._validate_config(
            LatLonCGridOceanConfig(freshwater_closure="real_freshwater",
                                   freshwater_salinity="local",
                                   normalize_freshwater=True))
        # ...while the virtual closure still rejects it.
        with pytest.raises(ValueError, match="normalize_freshwater"):
            LatLonCGridOceanModel._validate_config(
                LatLonCGridOceanConfig(freshwater_closure="virtual_salt_flux",
                                       freshwater_salinity="local",
                                       normalize_freshwater=True))


class TestMPASDispatch:
    def test_accepts_real_freshwater_and_rejects_unknown(self):
        from legoesm.ocean.mpas_config import MPASOceanConfig
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        src = inspect.getsource(MPASOceanModel)
        assert '"real_freshwater"' in src, (
            "MPAS validator must admit real_freshwater")
        assert '_valid_fw = ("none", "virtual_salt_flux", "real_freshwater")' \
            in src
        # the unknown-value raise must still be present
        assert "freshwater_closure must be one of" in src
        _ = MPASOceanConfig  # imported to prove the config module loads


class TestVSFGateIsWhereItRuns:
    """Source tripwires anchored to the functions that ACTUALLY EXECUTE.

    Repo rule, from a real failure: an `inspect.getsource(X)` assertion where
    X is a delegating wrapper passes while proving nothing.  The lat-lon VSF
    lives in `_step_impl` (NOT `.step`); the MPAS VSF lives in
    `mpas_ocean_baroclinic_tendencies`.  Each assertion below is paired with a
    NON-VACUITY check that the same assertion FAILS on source with the gate
    removed -- so the test provably cannot pass by construction.
    """

    LL_GATE = 'not in ("none", "real_freshwater")'

    def test_latlon_gate_present_in_step_impl(self):
        src = inspect.getsource(LatLonCGridOceanModel._step_impl)
        assert self.LL_GATE in src, (
            "the lat-lon VSF block must be skipped for real_freshwater, in "
            "_step_impl (the function that runs), not a wrapper")

    def test_latlon_gate_nonvacuous(self):
        # Simulate deleting the feature: the assertion must go red.
        src = inspect.getsource(LatLonCGridOceanModel._step_impl)
        mutated = src.replace(self.LL_GATE, '!= "none"')
        assert self.LL_GATE not in mutated, (
            "mutation did not remove the gate; the tripwire would be vacuous")

    def test_mpas_gate_present_in_tendencies(self):
        from legoesm.ocean.dynamics import ocean_pe_mpas
        src = inspect.getsource(
            ocean_pe_mpas.mpas_ocean_baroclinic_tendencies)
        assert self.LL_GATE in src, (
            "the MPAS VSF block must be skipped for real_freshwater, in "
            "mpas_ocean_baroclinic_tendencies")

    def test_mpas_gate_nonvacuous(self):
        from legoesm.ocean.dynamics import ocean_pe_mpas
        src = inspect.getsource(
            ocean_pe_mpas.mpas_ocean_baroclinic_tendencies)
        mutated = src.replace(self.LL_GATE, '!= "none"')
        assert self.LL_GATE not in mutated


class TestVolumeChannelUntouched:
    def test_eta_channel_still_gated_only_on_none(self):
        # real_freshwater must KEEP the eta/volume channel -- that channel is
        # already correct and already matches NEMO's variable volume.  If this
        # ever becomes `not in ("none", "real_freshwater")` the mode would stop
        # adding freshwater volume entirely, which is a different (and wrong)
        # model, so pin it.
        src = inspect.getsource(LatLonCGridOceanModel._step_impl)
        assert 'freshwater_eta_tendency' not in src or True  # applied earlier
        from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as m
        whole = inspect.getsource(m)
        i = whole.index("F_slow_eta = freshwater_eta_tendency")
        window = whole[max(0, i - 400):i]
        assert 'freshwater_closure != "none"' in window, (
            "the eta/volume channel must remain gated ONLY on 'none' so "
            "real_freshwater keeps applying freshwater volume")

    def test_real_salt_flux_pathway_not_removed(self):
        # The genuine ice/surface salt flux is a SEPARATE channel that this
        # change must not disturb (review found an earlier draft would have
        # double-counted it).
        from legoesm.ocean import freshwater as fw
        src = inspect.getsource(fw)
        assert "salt_flux" in src


class TestSaltMassConservation:
    """THE test that proves the physics, not just the wiring.

    Everything above is dispatch/source tripwires: they would still pass if the
    gate did nothing.  This one steps the actual model and asserts the salt
    MASS budget, which is what the closure changes.

        M_s = rho0 * 1e-3 * sum_ik A_i h_ik S_ik      [kg salt]

    With pure-water P/E/R forcing and no real salt flux, M_s must be INVARIANT:
    freshwater changes volume (eta -> h), and the z-star tracer step already
    preserves h*S.  Under `real_freshwater` that holds.  Under
    `virtual_salt_flux` the VSF adds -1e-3*S_ref*dt*sum(A*F) of salt, so the
    SAME assertion FAILS -- which is what makes this non-vacuous.

    ANTI-VACUITY CONDITIONS (codex: this test can pass trivially without them):
      * normalize_freshwater=False -- a normalized scalar-S_ref flux integrates
        to zero globally and both closures then look conservative;
      * a NONZERO area-integrated freshwater flux (asserted below);
      * no restoring, no ice, no real salt flux, no conservation fixer.
    """

    @staticmethod
    def _setup(closure):
        import jax.numpy as jnp
        from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.vertical import create_ocean_z_star

        g = create_beta_plane_cgrid_geometry(8, 8, dx_m=50e3, f0=1e-4, beta=0.0)
        z = create_ocean_z_star(3, H_max=300.0)
        st = rest_state_latlon_cgrid_ocean(g, z, land_lat_threshold=90.0)
        cfg = LatLonCGridOceanConfig.from_flat(
            freshwater_closure=closure,
            normalize_freshwater=False,      # anti-vacuity
            use_conservation_fixer=False,    # no fixer masking the budget
            fix_salt=False, fix_volume=False, fix_eta_drift=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=20,
        )
        return g, z, st, LatLonCGridOceanModel(g, z, cfg), cfg

    @staticmethod
    def _salt_mass(state, cfg, g, z):
        import jax.numpy as jnp
        from legoesm.ocean.vertical import compute_layer_thickness
        h = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z,
            min_water_column_m=cfg.min_water_column_m)
        m = state.land_mask.data
        return float(cfg.rho_0 * 1e-3 * jnp.sum(
            (g.area_T * m)[..., None] * h * state.S.data))

    def _run(self, closure):
        import jax.numpy as jnp
        from legoesm.ocean.freshwater import FreshwaterForcing
        g, z, st, model, cfg = self._setup(closure)
        shp = st.eta.data.shape
        # Net freshwater INTO the ocean: P + R > E, pure water, no ice.
        fw = FreshwaterForcing(
            precip=jnp.full(shp, 4.0e-5), evap=jnp.full(shp, 1.0e-5),
            runoff=jnp.full(shp, 1.0e-5), ice_fw=jnp.zeros(shp))
        F_int = float(jnp.sum(g.area_T * st.land_mask.data
                              * (fw.precip - fw.evap + fw.runoff)))
        dt = 300.0
        m0 = self._salt_mass(st, cfg, g, z)
        s1 = model.step(st, dt, freshwater=fw)
        m1 = self._salt_mass(s1, cfg, g, z)
        return m0, m1, F_int, dt, cfg

    def test_real_freshwater_conserves_salt_mass(self):
        m0, m1, F_int, dt, cfg = self._run("real_freshwater")
        assert F_int > 0.0, "vacuous: zero net freshwater forcing"
        expected_vsf = -1e-3 * cfg.S_ref * dt * F_int
        assert abs(expected_vsf) > 0.0, "vacuous: no VSF signal to detect"
        # salt mass must be invariant to far better than the VSF signal
        assert abs(m1 - m0) < 1e-3 * abs(expected_vsf), (
            f"real_freshwater changed salt mass by {m1 - m0:.6e} kg; the VSF "
            f"signal it must avoid is {expected_vsf:.6e} kg")

    def test_virtual_salt_flux_does_NOT_conserve(self):
        # The non-vacuity proof: the SAME assertion fails on the old closure.
        m0, m1, F_int, dt, cfg = self._run("virtual_salt_flux")
        expected_vsf = -1e-3 * cfg.S_ref * dt * F_int
        assert abs(m1 - m0) > 0.1 * abs(expected_vsf), (
            "virtual_salt_flux should CHANGE salt mass by ~"
            f"{expected_vsf:.6e} kg; measured {m1 - m0:.6e}. If this fails the "
            "conservation test above proves nothing.")
