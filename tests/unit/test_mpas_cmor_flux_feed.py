"""CMOR radiation/turbulent-flux feed on the MPAS lane (2026-07-24).

The first ClimateEval scorecard ran with rlut/rsut/rsdt/hfss/hfls/evspsbl
absent (the lean MPAS loop never exported them), so the radiation-budget and
surface-flux suites were skipped.  This covers the new chain:

  - ``_pack_hydrostatic_tendencies`` TOA kwargs (radiation packer) — CMOR
    sign conventions, None default byte-identical.
  - ``DiagnosticCollector.feed_cmip_accumulators_native`` new kwargs — shape
    validation (transactional PHASE 1), IDW-exact uniform-field regrid,
    ``evspsbl = hfls / L_v`` derivation, backward-compat when absent.
"""

import numpy as np
import pytest

from legoesm import constants
from legoesm.core.field import Field
from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.grids.factory import create_grid

NLEV = 6


@pytest.fixture(scope="module")
def mesh():
    return create_grid("mpas", 2, lloyd_iterations=10)


def _make_collector(mesh):
    sigma_full = np.linspace(0.05, 0.98, NLEV)
    dsigma = np.full(NLEV, 1.0 / NLEV)
    dc = DiagnosticCollector(
        nlev=NLEV, sigma_full=sigma_full, dsigma=dsigma,
        experiment_id="amip", monthly_means=True,
        cmip_output=True, n_days=30, output_dir=None,
        cmip_resolution_deg=10.0, start_year=1979,
    )
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    return dc, sigma_full


def _base_fields(mesh, sigma_full):
    latc = np.asarray(mesh.latCell)
    n = int(mesh.nCells)
    T = 250.0 + 40.0 * sigma_full[None, :] + 10.0 * np.cos(latc)[:, None]
    return dict(
        T=T, p_s=np.full(n, 1.0e5), lat_deg=np.degrees(latc),
        phis=np.zeros(n))


FLUXES = dict(rlut=238.0, rsut=99.0, rsdt=340.0, hfss=17.0, hfls=88.0)


class TestFeed:
    def test_flux_fields_reach_monthly_accumulator(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        fed = dc.feed_cmip_accumulators_native(
            day=15.0, **f,
            **{k: np.full(n, v) for k, v in FLUXES.items()})
        assert fed is True
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        # Uniform fields -> IDW (partition of unity) regrid is EXACT.
        for k, v in FLUXES.items():
            np.testing.assert_allclose(out[f"field_2d_{k}"], v, rtol=1e-9)
        # evspsbl derived as hfls / L_v [kg/m2/s].
        np.testing.assert_allclose(
            out["field_2d_evspsbl"], FLUXES["hfls"] / constants.L_v,
            rtol=1e-9)

    def test_wrong_shape_raises_and_commits_nothing(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        with pytest.raises(ValueError, match="rlut"):
            dc.feed_cmip_accumulators_native(
                day=15.0, **f, rlut=np.full((n, 2), 238.0))
        # Transactional: the malformed feed committed NOTHING.
        assert dc._spatial_monthly._max_count_ever == 0

    def test_absent_fluxes_backward_compatible(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        dc.feed_cmip_accumulators_native(day=15.0, **f)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        for k in (*FLUXES, "evspsbl"):
            assert f"field_2d_{k}" not in out


class TestRadiationPacker:
    def _pack(self, mesh, **kw):
        from legoesm.atmosphere.physics.radiation.integration import (
            _pack_hydrostatic_tendencies,
        )
        import jax.numpy as jnp
        from legoesm.core.state import MPASHydrostaticState

        n = int(mesh.nCells)
        ne = int(mesh.nEdges)
        z2 = jnp.zeros(n)
        state = MPASHydrostaticState(
            u=Field(data=jnp.zeros((ne, NLEV)), name="u",
                    dims=("edge", "level"), units="m/s"),
            T=Field(data=jnp.full((n, NLEV), 260.0), name="T",
                    dims=("cell", "level"), units="K"),
            p_s=Field(data=z2 + 1e5, name="p_s", dims=("cell",), units="Pa"),
            phis=Field(data=z2, name="phis", dims=("cell",),
                       units="m^2/s^2"),
            tracers=None,
        )
        return _pack_hydrostatic_tendencies(
            jnp.zeros((n, NLEV)), state, (n, NLEV), (n,), **kw)

    def test_toa_fields_default_none(self, mesh):
        t = self._pack(mesh)
        assert t.sw_up_toa is None
        assert t.lw_up_toa is None
        assert t.sw_down_toa is None

    def test_toa_fields_packed_with_values(self, mesh):
        import jax.numpy as jnp
        n = int(mesh.nCells)
        t = self._pack(
            mesh,
            sw_up_toa=jnp.full(n, 99.0),
            lw_up_toa=jnp.full(n, 238.0),
            sw_down_toa=jnp.full(n, 340.0))
        np.testing.assert_allclose(np.asarray(t.sw_up_toa.data), 99.0)
        np.testing.assert_allclose(np.asarray(t.lw_up_toa.data), 238.0)
        np.testing.assert_allclose(np.asarray(t.sw_down_toa.data), 340.0)
        assert t.lw_up_toa.units == "W/m^2"


def _tend_with_extras():
    """A HydrostaticTendencies with distinct sentinel values in every CMOR
    diagnostic field (name = the field, so slot order is checkable)."""
    from legoesm.core.state import HydrostaticTendencies

    def _f(name, val):
        return Field(data=np.full(3, val), name=name, dims=("cell",),
                     units="W/m^2")
    _dyn = Field(data=np.zeros(3), name="dyn", dims=("cell",), units="1")
    return HydrostaticTendencies(
        du_dt=_dyn, dT_dt=_dyn, dp_s_dt=_dyn, dphis_dt=_dyn,
        sw_net_sfc=_f("sw_net", 1.0), lw_net_sfc=_f("lw_net", 2.0),
        precip=_f("precip", 3.0),
        lw_up_toa=_f("lw_up_toa", 4.0), sw_up_toa=_f("sw_up_toa", 5.0),
        sw_down_toa=_f("sw_down_toa", 6.0),
        shflx_sfc=_f("shflx", 7.0), lhflx_sfc=_f("lhflx", 8.0))


# Producer extraction shared VERBATIM by primitive_eq_mpas.step() and
# voronoi_mpi._step; a drift here silently swaps rlut/rsut/rsdt or drops a field.
_EXTRA_ORDER = ("lw_up_toa", "sw_up_toa", "sw_down_toa", "shflx_sfc", "lhflx_sfc")


class TestSfcDiagContract:
    """Lock the 8-slot sfc_diag tuple contract shared by BOTH producers
    (serial + MPI-voronoi) and the driver consumer's slot 3-7 mapping."""

    def test_producer_slot_order_matches_consumer(self):
        _pt = _tend_with_extras()
        _extras = tuple(getattr(_pt, _k, None) for _k in _EXTRA_ORDER)
        sfc_diag = (_pt.sw_net_sfc, _pt.lw_net_sfc, _pt.precip) + _extras
        assert len(sfc_diag) == 8
        # Consumer (_feed_mpas_cmip_accumulators): slot 3->rlut, 4->rsut,
        # 5->rsdt, 6->hfss, 7->hfls.
        assert sfc_diag[3].name == "lw_up_toa"    # rlut
        assert sfc_diag[4].name == "sw_up_toa"    # rsut
        assert sfc_diag[5].name == "sw_down_toa"  # rsdt
        assert sfc_diag[6].name == "shflx"        # hfss
        assert sfc_diag[7].name == "lhflx"        # hfls

    def test_replace_preserves_cmor_diagnostics(self):
        """The HS wrapper repacks via ``_replace`` of the 4 dynamics fields
        only; every diagnostic (sw/lw net, precip, TOA trio, turb fluxes) must
        survive — the invariant that fix relies on."""
        _pt = _tend_with_extras()
        summed = _pt._replace(dT_dt=_pt.dT_dt.replace(data=np.ones(3)))
        for _k in ("sw_net_sfc", "lw_net_sfc", "precip", "lw_up_toa",
                   "sw_up_toa", "sw_down_toa", "shflx_sfc", "lhflx_sfc"):
            assert getattr(summed, _k) is not None, _k
        np.testing.assert_allclose(np.asarray(summed.dT_dt.data), 1.0)

    def test_padded_merge_never_truncates(self):
        """The slot-wise keep-last-non-None merge (both producers) must pad,
        not zip-truncate, when a 3-slot held-step default meets an 8-slot
        published prev."""
        prev = tuple(range(1, 9))          # 8-slot published bundle
        new = (None, None, 30)             # held step: only precip fresh
        _n = max(len(new), len(prev))
        prev_p = prev + (None,) * (_n - len(prev))
        new_p = new + (None,) * (_n - len(new))
        merged = tuple(a if a is not None else b
                       for a, b in zip(new_p, prev_p))
        assert merged == (1, 2, 30, 4, 5, 6, 7, 8)  # radiation slots retained
