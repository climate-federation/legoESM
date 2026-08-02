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

import types

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


# CMIP6 clear-sky QUARTET (sfc_diag slots 12-15) with values chosen to be
# physically ordered against their all-sky partners in FLUXES:
#   rsutcs < rsut (99)   rlutcs > rlut (238)   rsdscs/rldscs are surface.
CLEARSKY = dict(rsutcs=41.0, rlutcs=266.0, rsdscs=250.0, rldscs=290.0)


class TestClearSkyFeed:
    """CMOR ``rsutcs``/``rlutcs``/``rsdscs``/``rldscs`` publication on the
    MPAS lane (PR #1437 dead-plumbing closure): the clear-sky quartet must
    reach the SAME spatial monthly accumulator as its all-sky partners."""

    def test_clear_sky_quartet_reaches_monthly_accumulator(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        fed = dc.feed_cmip_accumulators_native(
            day=15.0, **f,
            **{k: np.full(n, v) for k, v in FLUXES.items()},
            **{k: np.full(n, v) for k, v in CLEARSKY.items()})
        assert fed is True
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        for k, v in CLEARSKY.items():
            assert f"field_2d_{k}" in out, (
                f"{k} never reached the CMOR accumulator — the #1437 "
                f"dead-plumbing symptom")
            # Uniform field -> IDW (partition of unity) regrid is EXACT.
            np.testing.assert_allclose(out[f"field_2d_{k}"], v, rtol=1e-9)

    def test_published_values_are_not_flipped_or_swapped(self, mesh):
        """Each clear-sky field must carry ITS OWN value with ITS OWN sign —
        a swapped pair or a stray negation would still 'publish four
        fields'."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            day=15.0, **f,
            **{k: np.full(n, v) for k, v in FLUXES.items()},
            **{k: np.full(n, v) for k, v in CLEARSKY.items()})
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        for k, v in CLEARSKY.items():
            got = float(np.mean(out[f"field_2d_{k}"]))
            assert got == pytest.approx(v, rel=1e-9), (
                f"{k} published {got}, expected {v} (swap or sign flip)")
        # Cloud radiative effect signs survive the whole feed.
        assert float(np.mean(out["field_2d_rsut"])) > \
            float(np.mean(out["field_2d_rsutcs"]))
        assert float(np.mean(out["field_2d_rlutcs"])) > \
            float(np.mean(out["field_2d_rlut"]))

    def test_clear_sky_variables_are_in_the_amon_table(self):
        """A field the CMOR writer has no entry for is published nowhere."""
        from legoesm.io.cmor_output import lookup_cmor_entry
        for k in CLEARSKY:
            table, entry = lookup_cmor_entry(k)
            assert table == "Amon", k
            assert entry["units"] == "W m-2", k
            assert "assuming_clear_sky" in entry["standard_name"], k

    def test_clear_sky_fields_are_treated_as_interval_means(self):
        """They come from the per-step accumulator like rsut/rlut, so they
        must sit in the _FLUX_2D midpoint-binned set — NOT the endpoint
        snapshot set, which would shift them one interval late."""
        import inspect
        src = inspect.getsource(
            DiagnosticCollector.feed_cmip_accumulators_native)
        flux_block = src.split("_FLUX_2D = ")[1].split("_FLUX_DAILY")[0]
        for k in CLEARSKY:
            assert f'"{k}"' in flux_block, (
                f"{k} is not a _FLUX_2D member — it would be calendar-binned "
                f"as an endpoint snapshot, not an interval mean")

    @pytest.mark.parametrize("bad", sorted(CLEARSKY))
    def test_malformed_clear_sky_input_raises_and_commits_nothing(
            self, mesh, bad):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        with pytest.raises(ValueError, match=bad):
            dc.feed_cmip_accumulators_native(
                day=15.0, **f, **{bad: np.zeros(n - 1)})
        assert dc._spatial_monthly._max_count_ever == 0

    def test_absent_clear_sky_is_skipped_never_zeroed(self, mesh):
        """Backward compatibility: with the diagnostic off the fields must be
        ABSENT.  A zero rsutcs would read as a black planet downstream."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            day=15.0, **f, **{k: np.full(n, v) for k, v in FLUXES.items()})
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        for k in CLEARSKY:
            assert f"field_2d_{k}" not in out

    def test_accumulator_covers_the_clear_sky_slots(self):
        """``_MPASSfcFluxAccum`` must average slots 12-15 — a slot missing
        from SLOTS is silently dropped from every interval mean."""
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        for slot in (12, 13, 14, 15):
            assert slot in _MPASSfcFluxAccum.SLOTS, slot
        acc = _MPASSfcFluxAccum()
        for v in (10.0, 20.0, 60.0):
            row = [None] * 16
            row[12] = types.SimpleNamespace(data=np.full(4, v))
            row[15] = types.SimpleNamespace(data=np.full(4, 2.0 * v))
            acc.add(tuple(row))
        np.testing.assert_allclose(acc.mean(12), 30.0)   # (10+20+60)/3
        np.testing.assert_allclose(acc.mean(15), 60.0)
        assert acc.mean(13) is None                      # never fed

    def test_driver_reads_the_clear_sky_slots_by_index(self):
        """The consumer that RUNS (``_feed_mpas_cmip_accumulators``) must map
        slots 12-15 onto the four CMOR names and forward them to the feed."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(
            ModelDriver._feed_mpas_cmip_accumulators)
        for name, slot in (("rsutcs", 12), ("rlutcs", 13),
                           ("rsdscs", 14), ("rldscs", 15)):
            assert f"{name} = _sfc_slot({slot})" in src, (
                f"{name} is not read from slot {slot}")
            assert f"{name}={name}," in src, (
                f"{name} is read but never forwarded to the CMOR feed — "
                f"exactly the #1437 dead-plumbing failure mode")


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
        shflx_sfc=_f("shflx", 7.0), lhflx_sfc=_f("lhflx", 8.0),
        sw_down_sfc=_f("sw_down_sfc", 9.0), lw_down_sfc=_f("lw_down_sfc", 10.0),
        tau_x_sfc=_f("tau_x", 11.0), tau_y_sfc=_f("tau_y", 12.0),
        sw_up_toa_clearsky=_f("sw_up_toa_clr", 13.0),
        lw_up_toa_clearsky=_f("lw_up_toa_clr", 14.0),
        sw_down_sfc_clearsky=_f("sw_down_sfc_clr", 15.0),
        lw_down_sfc_clearsky=_f("lw_down_sfc_clr", 16.0))


# Producer extraction shared VERBATIM by primitive_eq_mpas.step() and
# voronoi_mpi._step; a drift here silently swaps rlut/rsut/rsdt or drops a field.
_EXTRA_ORDER = ("lw_up_toa", "sw_up_toa", "sw_down_toa",
                "shflx_sfc", "lhflx_sfc",
                "sw_down_sfc", "lw_down_sfc",
                "tau_x_sfc", "tau_y_sfc",
                "sw_up_toa_clearsky", "lw_up_toa_clearsky",
                "sw_down_sfc_clearsky", "lw_down_sfc_clearsky")


class TestSfcDiagContract:
    """Lock the 16-slot sfc_diag tuple contract shared by BOTH producers
    (serial + MPI-voronoi) and the driver consumer's slot mapping."""

    def test_producer_slot_order_matches_consumer(self):
        _pt = _tend_with_extras()
        _extras = tuple(getattr(_pt, _k, None) for _k in _EXTRA_ORDER)
        sfc_diag = (_pt.sw_net_sfc, _pt.lw_net_sfc, _pt.precip) + _extras
        assert len(sfc_diag) == 16
        # Consumer (_feed_mpas_cmip_accumulators): slot 0->sw_net (rsus
        # derivation), 1->lw_net (rlus), 3->rlut, 4->rsut, 5->rsdt,
        # 6->hfss, 7->hfls, 8->rsds, 9->rlds (+ _marshal_land_forcing),
        # 10->tauu (sign-flipped), 11->tauv (sign-flipped).
        assert sfc_diag[0].name == "sw_net"
        assert sfc_diag[1].name == "lw_net"
        assert sfc_diag[3].name == "lw_up_toa"      # rlut
        assert sfc_diag[4].name == "sw_up_toa"      # rsut
        assert sfc_diag[5].name == "sw_down_toa"    # rsdt
        assert sfc_diag[6].name == "shflx"          # hfss
        assert sfc_diag[7].name == "lhflx"          # hfls
        assert sfc_diag[8].name == "sw_down_sfc"    # rsds
        assert sfc_diag[9].name == "lw_down_sfc"    # rlds
        assert sfc_diag[10].name == "tau_x"         # tauu = -slot10
        assert sfc_diag[11].name == "tau_y"         # tauv = -slot11
        # Clear-sky quartet: 12/13 TOA outgoing (+up, pair with 3/4),
        # 14/15 surface downwelling (+down, pair with 8/9).  NO sign flip
        # on any of the four.
        assert sfc_diag[12].name == "sw_up_toa_clr"    # rsutcs
        assert sfc_diag[13].name == "lw_up_toa_clr"    # rlutcs
        assert sfc_diag[14].name == "sw_down_sfc_clr"  # rsdscs
        assert sfc_diag[15].name == "lw_down_sfc_clr"  # rldscs

    def test_both_producers_extract_the_same_extra_order(self):
        """The serial and MPI producers must list the SAME extras keys in
        the SAME order — a drift silently remaps CMOR fields on one lane."""
        import inspect
        from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas
        from legoesm.parallel import voronoi_mpi

        def _keys_in(src):
            found = []
            for k in _EXTRA_ORDER:
                pos = src.find(f'"{k}"')
                assert pos >= 0, f"{k} missing from a producer's extras"
                found.append((pos, k))
            return [k for _, k in sorted(found)]

        # _step_jit is the symbol that BUILDS the tuple on the serial lane
        # (step() only merges it) — asserting against step() would pass
        # while proving nothing (attribution-gate rule).
        src_serial = inspect.getsource(
            primitive_eq_mpas.MPASPrimitiveEquationModel._step_jit)
        src_mpi = inspect.getsource(voronoi_mpi.make_voronoi_mpi_step)
        assert _keys_in(src_serial) == list(_EXTRA_ORDER)
        assert _keys_in(src_mpi) == list(_EXTRA_ORDER)

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


# ---------------------------------------------------------------------------
# #1353: per-step interval accumulation + snapshot cell_methods honesty
# ---------------------------------------------------------------------------


class TestSfcFluxAccum:
    """``_MPASSfcFluxAccum``: the per-step flux averager the eager MPAS
    loop feeds so CMOR monthlies of diurnal fields are true interval means
    (issue #1353 — pre-fix they were 00 UTC snapshots labeled time: mean)."""

    def _mk(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        return _MPASSfcFluxAccum()

    def _field(self, v, n=4):
        import types
        return types.SimpleNamespace(data=np.full(n, float(v)))

    def test_mean_is_arithmetic_mean_per_slot(self):
        acc = self._mk()
        for v in (100.0, 200.0, 600.0):
            acc.add((None, None, None, None, self._field(v)))  # slot 4 rsut
        m = acc.mean(4)
        np.testing.assert_allclose(m, 300.0)
        assert m.dtype == np.float64

    def test_none_slots_and_short_tuples_are_safe(self):
        acc = self._mk()
        acc.add(None)                       # no diag yet
        acc.add(())                         # empty tuple
        acc.add((None, None, self._field(2.0)))       # 3-slot legacy: precip
        acc.add((None, None, self._field(4.0), self._field(240.0)))
        np.testing.assert_allclose(acc.mean(2), 3.0)   # precip: 2 samples
        np.testing.assert_allclose(acc.mean(3), 240.0)  # rlut: 1 sample
        assert acc.mean(7) is None                     # hfls never fed

    def test_reset_clears_all(self):
        acc = self._mk()
        acc.add((None, None, self._field(1.0)))
        acc.reset()
        assert acc.mean(2) is None

    def test_counts_independent_per_slot(self):
        """Radiation slots refresh less often than precip upstream; each
        slot must average over its OWN sample count."""
        acc = self._mk()
        acc.add((None, None, self._field(1.0), self._field(200.0)))
        acc.add((None, None, self._field(3.0)))        # held-radiation step
        np.testing.assert_allclose(acc.mean(2), 2.0)
        np.testing.assert_allclose(acc.mean(3), 200.0)


class TestFeedUsesIntervalMeans:
    def test_feed_prefers_accum_mean_over_snapshot(self, mesh):
        """With the per-step accumulator populated, the CMOR feed must hand
        the interval MEAN to the collector, not the last instantaneous
        ``_sfc_diag`` value — and must reset the accumulator afterwards."""
        import types
        from legoesm.driver.model_driver import (
            ModelDriver, _MPASSfcFluxAccum)

        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]

        def _field(a):
            return types.SimpleNamespace(data=np.asarray(a))

        acc = _MPASSfcFluxAccum()
        # Two per-step samples: precip 2e-5 then 4e-5 -> mean 3e-5.
        acc.add((None, None, _field(np.full(n, 2.0e-5))))
        acc.add((None, None, _field(np.full(n, 4.0e-5))))

        # Edge-normal wind state: zeros are fine for this assertion.
        u_edge = np.zeros((int(mesh.nEdges), NLEV))
        fake = types.SimpleNamespace(
            diagnostics=dc,
            grid=mesh,
            _mpas_sfc_accum=acc,
            config=types.SimpleNamespace(
                output=types.SimpleNamespace(diag_days=1.0)),
            state=types.SimpleNamespace(
                u=_field(u_edge), T=_field(f["T"]), p_s=_field(f["p_s"]),
                phis=_field(f["phis"]), tracers=None,
            ),
            # Instantaneous end-of-interval snapshot deliberately DIFFERENT
            # from the accumulated mean.
            model=types.SimpleNamespace(
                _sfc_diag=(None, None, _field(np.full(n, 9.0e-5)))),
        )
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)

        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        # rtol accommodates the documented fp32 fallback of the device-side
        # sums when JAX x64 is off (CI default) — the assertion under test
        # is mean-vs-snapshot (3e-5 vs 9e-5), not bit precision.
        np.testing.assert_allclose(out["field_2d_pr"], 3.0e-5, rtol=1e-6)
        # Interval accumulator restarted for the next interval.
        assert acc.mean(2) is None

    def test_feed_falls_back_to_snapshot_without_accum(self, mesh):
        """No accumulator (pre-#1353 paths, non-MPAS drivers): the feed
        reads the instantaneous slots exactly as before."""
        import types
        from legoesm.driver.model_driver import ModelDriver

        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]

        def _field(a):
            return types.SimpleNamespace(data=np.asarray(a))

        u_edge = np.zeros((int(mesh.nEdges), NLEV))
        fake = types.SimpleNamespace(
            diagnostics=dc,
            grid=mesh,
            state=types.SimpleNamespace(
                u=_field(u_edge), T=_field(f["T"]), p_s=_field(f["p_s"]),
                phis=_field(f["phis"]), tracers=None,
            ),
            model=types.SimpleNamespace(
                _sfc_diag=(None, None, _field(np.full(n, 9.0e-5)))),
        )
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        np.testing.assert_allclose(out["field_2d_pr"], 9.0e-5, rtol=1e-9)


class TestSnapshotCellMethods:
    def test_snapshot_vars_written_time_point(self, mesh, tmp_path):
        """``cmip_snapshot_vars`` stamps ``cell_methods='time: point'`` on
        snapshot-fed vars while accumulated vars keep the table's
        ``time: mean`` — checked on the actual NetCDF files."""
        xr = pytest.importorskip("xarray")
        from legoesm.io.cmor_output import CFWriter

        dc, _ = _make_collector(mesh)
        dc.cf_writer = CFWriter(
            output_dir=tmp_path, experiment_id="amip",
            model_id="legoESM", ref_date="1979-01-01")
        dc.cmip_snapshot_vars = {"tas"}

        nlat, nlon = dc._cmip_nlat, dc._cmip_nlon
        data = {
            "months": [(0, 1)],
            "field_2d_tas": np.full((1, nlat, nlon), 288.0),
            "field_2d_pr": np.full((1, nlat, nlon), 3.0e-5),
        }
        dc._write_cmip_data(data)

        tas_files = sorted(tmp_path.rglob("tas_*.nc"))
        pr_files = sorted(tmp_path.rglob("pr_*.nc"))
        assert tas_files and pr_files
        with xr.open_dataset(tas_files[0]) as ds:
            assert (ds["tas"].attrs["cell_methods"]
                    == "time: point within days time: mean over days")
            assert "aliased" in ds["tas"].attrs["comment"]
        with xr.open_dataset(pr_files[0]) as ds:
            assert ds["pr"].attrs["cell_methods"] == "time: mean"
            assert "comment" not in ds["pr"].attrs

    def test_default_none_keeps_table_cell_methods(self, mesh, tmp_path):
        """Cube/lat-lon segment-accumulated path: ``cmip_snapshot_vars``
        unset -> table defaults untouched (regression guard)."""
        xr = pytest.importorskip("xarray")
        from legoesm.io.cmor_output import CFWriter

        dc, _ = _make_collector(mesh)
        dc.cf_writer = CFWriter(
            output_dir=tmp_path, experiment_id="amip",
            model_id="legoESM", ref_date="1979-01-01")
        assert dc.cmip_snapshot_vars is None

        nlat, nlon = dc._cmip_nlat, dc._cmip_nlon
        data = {
            "months": [(0, 1)],
            "field_2d_tas": np.full((1, nlat, nlon), 288.0),
        }
        dc._write_cmip_data(data)
        tas_files = sorted(tmp_path.rglob("tas_*.nc"))
        assert tas_files
        with xr.open_dataset(tas_files[0]) as ds:
            assert ds["tas"].attrs["cell_methods"] == "time: mean"


class TestHeldRadiationContract:
    """Codex-1 finding 7: model the PRODUCTION held-slot contract — the
    element-wise merge keeps the last radiation value present on every
    step's tuple, so per-step sampling of the held value is the correct
    zero-order-hold discrete time mean."""

    def _field(self, v, n=4):
        import types
        return types.SimpleNamespace(data=np.full(n, float(v)))

    def test_held_value_every_step_gives_zoh_time_mean(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum()
        # Radiation cadence 2: values refresh at steps 0,2 and are HELD
        # (present, unchanged) at steps 1,3 — the merged tuple always
        # carries the slot (production merge keeps last non-None).
        for v in (200.0, 200.0, 260.0, 260.0):
            acc.add((None, None, None, self._field(v)))
        np.testing.assert_allclose(acc.mean(3), 230.0)  # ZOH time mean

    def test_fp32_source_accumulates_in_float64(self):
        import types
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum()
        x32 = types.SimpleNamespace(
            data=np.full(4, 340.25, dtype=np.float32))
        for _ in range(3):
            acc.add((None, None, x32))
        m = acc.mean(2)
        assert m.dtype == np.float64
        np.testing.assert_allclose(m, np.float64(np.float32(340.25)))


class TestFluxMidpointBinning:
    """Codex-1 finding 1: interval-mean flux fields are calendar-binned at
    the interval MIDPOINT, so January's last daily mean ([30,31]) stays in
    January while the co-fed state snapshot (at day 31.0 = Feb 1 00 UTC)
    bins to February."""

    def test_month_boundary_split(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        fed = dc.feed_cmip_accumulators_native(
            31.0,                       # Feb 1 00 UTC endpoint
            **f,
            precip=np.full(n, 3.0e-5),  # interval MEAN over [30, 31]
            flux_interval_days=1.0,
        )
        assert fed
        keys_2d = dc._spatial_monthly._data_2d
        jan, feb = (0, 1), (0, 2)
        assert "pr" in keys_2d.get(jan, {}), "flux mean must bin to January"
        assert "pr" not in keys_2d.get(feb, {})
        assert "tas" in keys_2d.get(feb, {}), "state snapshot bins to February"
        assert "tas" not in keys_2d.get(jan, {})

    def test_interior_day_same_bucket_single_commit(self, mesh):
        """Away from month boundaries the split is a no-op: one bucket,
        and the per-bucket call count stays at the legacy one-add_2d-per-
        feed (the _same_bin collapse)."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            15.0, **f, precip=np.full(n, 3.0e-5), flux_interval_days=1.0)
        keys_2d = dc._spatial_monthly._data_2d
        jan = (0, 1)
        assert "pr" in keys_2d.get(jan, {}) and "tas" in keys_2d.get(jan, {})
        assert dc._spatial_monthly._call_counts[jan] == 2  # one 2d + one 3d

    def test_no_interval_means_endpoint_binning_unchanged(self, mesh):
        """flux_interval_days=None (instantaneous fallback / legacy
        callers): everything bins at the endpoint exactly as before."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            31.0, **f, precip=np.full(n, 3.0e-5))
        keys_2d = dc._spatial_monthly._data_2d
        feb = (0, 2)
        assert "pr" in keys_2d.get(feb, {}) and "tas" in keys_2d.get(feb, {})
        assert (0, 1) not in keys_2d


class TestRound3:
    """Codex-2 round: year-boundary binning, checkpoint round-trip of the
    partial interval, wallclock day-table attr parity, daily-bucket
    midpoint binning, multi-day-cadence fallback."""

    def test_year_boundary_flux_binning(self, mesh):
        """Endpoint day 365.0 = Jan 1 of year 1; the [364,365] flux mean
        must stay in YEAR 0 December."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            365.0, **f, precip=np.full(n, 3.0e-5), flux_interval_days=1.0)
        keys_2d = dc._spatial_monthly._data_2d
        dec_y0, jan_y1 = (0, 12), (1, 1)
        assert "pr" in keys_2d.get(dec_y0, {})
        assert "pr" not in keys_2d.get(jan_y1, {})
        assert "tas" in keys_2d.get(jan_y1, {})

    def test_daily_pr_bins_to_covered_day(self, mesh):
        """Interior day: the [14,15] pr mean lands in the day-15 daily
        bucket (doy 15), NOT the endpoint day-16 bucket (codex-2 finding
        7 — a month-level collapse had re-introduced the one-day lag)."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            15.0, **f, precip=np.full(n, 3.0e-5), flux_interval_days=1.0)
        daily = dc._spatial_daily._data
        assert "pr" in daily.get((0, 15), {}), "pr must bin to covered day"
        assert "pr" not in daily.get((0, 16), {})
        assert "tas" in daily.get((0, 16), {})  # snapshot at endpoint

    def test_multiday_cadence_no_midpoint_requested(self):
        """Driver-side gate: cadences > 1 day never request midpoint
        binning (they can straddle months) — enforced by the driver
        passing flux_interval_days=None; here we assert the feed treats
        None as endpoint semantics (legacy)."""
        # Covered structurally by
        # TestFluxMidpointBinning.test_no_interval_means_endpoint_binning
        # _unchanged; this test pins the DRIVER gate expression.
        import inspect
        from legoesm.driver import model_driver
        src = inspect.getsource(
            model_driver.ModelDriver._feed_mpas_cmip_accumulators)
        assert "_win_days <= 1.0" in src and "round(_per_day)" in src, (
            "driver must gate flux_interval_days to windows that divide "
            "the day evenly (derived from the integer step count)")

    def test_accum_checkpoint_roundtrip(self):
        """dump() -> (npz-like dict) -> restore() resumes the partial
        interval: counts and means continue exactly."""
        import types
        from legoesm.driver.model_driver import _MPASSfcFluxAccum

        def _field(v, n=4):
            return types.SimpleNamespace(data=np.full(n, float(v)))

        acc = _MPASSfcFluxAccum()
        acc.add((None, None, _field(2.0e-5), _field(240.0)))
        acc.add((None, None, _field(4.0e-5)))
        payload = acc.dump()
        assert set(payload) == {"cmor_fluxsum_2", "cmor_fluxcnt_2",
                                "cmor_fluxsum_3", "cmor_fluxcnt_3",
                                "cmor_fluxsteps", "cmor_fluxexpected",
                                "cmor_fluxday0", "cmor_fluxdt_s"}

        acc2 = _MPASSfcFluxAccum()
        staged = dict(payload)
        n_res = acc2.restore(staged)
        assert n_res == 2 and not staged      # keys popped
        acc2.add((None, None, _field(6.0e-5)))  # post-restart sample
        np.testing.assert_allclose(acc2.mean(2), 4.0e-5)  # (2+4+6)/3
        np.testing.assert_allclose(acc2.mean(3), 240.0)

    def test_empty_accum_dumps_nothing(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        assert _MPASSfcFluxAccum().dump() == {}

    def test_wallclock_daily_write_carries_snapshot_attrs(self, mesh):
        """finalize_cmip_daily (the wallclock-exit path) must pass the same
        per-var honesty overrides as the end-of-run writer (codex-2
        finding 4)."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.cmip_snapshot_vars = {"tas", "ps", "psl", "prw",
                                 "ta", "hus", "ua", "va"}
        dc.feed_cmip_accumulators_native(
            15.0, **f, precip=np.full(n, 3.0e-5), flux_interval_days=1.0)

        calls = {}

        class _W:
            def write_daily(self, data, lat, lon, extra_attrs_by_var=None):
                calls["extra"] = extra_attrs_by_var
                return []
        dc.cf_writer = _W()
        dc.finalize_cmip_daily(current_day=16.0)
        assert calls["extra"] is not None
        assert calls["extra"]["tas"]["cell_methods"] == "time: point"
        assert "tasmax" in calls["extra"]
        assert "pr" not in calls["extra"]     # true interval mean

    def test_snapshot_attrs_none_without_snapshot_mode(self, mesh):
        dc, _ = _make_collector(mesh)
        assert dc._daily_snapshot_attrs() is None

    def test_accum_roundtrip_through_npz(self, tmp_path):
        """The payload must survive an actual np.savez/np.load cycle (the
        real checkpoint medium): counts come back as 0-d int arrays."""
        import types
        from legoesm.driver.model_driver import _MPASSfcFluxAccum

        def _field(v, n=4):
            return types.SimpleNamespace(data=np.full(n, float(v)))

        acc = _MPASSfcFluxAccum()
        acc.add((None, None, _field(2.0), _field(240.0)))
        acc.add((None, None, _field(4.0)))
        fn = tmp_path / "ckpt.npz"
        np.savez(fn, **acc.dump())
        with np.load(fn) as z:
            staged = {k: np.asarray(z[k]) for k in z.files}
        acc2 = _MPASSfcFluxAccum()
        assert acc2.restore(staged) == 2 and not staged
        acc2.add((None, None, _field(6.0)))
        np.testing.assert_allclose(acc2.mean(2), 4.0)
        np.testing.assert_allclose(acc2.mean(3), 240.0)


class TestRound4:
    """Codex-3: absolute-step diag phase, fail-loud restore, negative epoch."""

    def test_diag_trigger_uses_absolute_step(self):
        """A restart chain must keep ONE global diagnostic phase, else a
        restored partial flux interval is completed at a fresh job-local
        multiple (codex-3 HIGH).  Pin the absolute-step expression in the
        MPAS loop."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._run_mpas)
        assert "(DIAG_PHASE + step + 1) % DIAG_INTERVAL == 0" in src, (
            "MPAS diagnostics must trigger on the phased step index")
        assert ("DIAG_PHASE = start_step if cfg.output.diag_days > 0 else 0"
                in src), (
            "periodic cadences phase on start_step; the once-at-the-end "
            "sentinel stays job-local (cfg.days is per-link on this lane)")

    def test_absolute_phase_completes_restored_interval(self):
        """Behavioural form of the same rule: with DIAG_INTERVAL=4 and a
        restart at absolute step 2, the first feed must come after 2 more
        steps (completing the interval), not after 4."""
        DIAG = 4
        start_step = 2
        fired = [s for s in range(4) if (start_step + s + 1) % DIAG == 0]
        assert fired[0] == 1, "first feed completes the partial interval"
        legacy = [s for s in range(4) if (s + 1) % DIAG == 0]
        assert legacy[0] == 3, "legacy job-local phase would restart the clock"

    def test_restore_rejects_half_pair(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum()
        with pytest.raises(ValueError, match="incomplete"):
            acc.restore({"cmor_fluxsum_2": np.zeros(4)})
        with pytest.raises(ValueError, match="incomplete"):
            acc.restore({"cmor_fluxcnt_3": np.int64(5)})

    def test_restore_rejects_nonpositive_count(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        with pytest.raises(ValueError, match="non-positive"):
            _MPASSfcFluxAccum().restore(
                {"cmor_fluxsum_2": np.zeros(4),
                 "cmor_fluxcnt_2": np.int64(0)})

    def test_checkpoint_key_whitelist_matches_dump(self):
        import types
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum()
        for i in _MPASSfcFluxAccum.SLOTS:
            tup = tuple(
                types.SimpleNamespace(data=np.ones(3)) if j == i else None
                for j in range(i + 1))
            acc.add(tup)
        assert set(acc.dump()) == set(_MPASSfcFluxAccum.checkpoint_keys())

    def test_negative_epoch_flux_binning(self, mesh):
        """Negative simulation days (pre-epoch runs) must bin consistently
        instead of clamping to day 0 (codex-3 LOW)."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            -364.0, **f, precip=np.full(n, 3.0e-5), flux_interval_days=1.0)
        keys = dc._spatial_monthly._data_2d
        # Endpoint -364.0 -> (year -1, doy 2); midpoint -364.5 -> (year -1,
        # doy 1.5): same January bucket of year -1.  The pre-fix clamp sent
        # the flux mean to day 0 == (year 0, January) instead.
        assert (-1, 1) in keys, keys
        assert "pr" in keys[(-1, 1)]
        assert (0, 1) not in keys, "clamped to the epoch bucket"

    def test_load_clears_live_accumulator_before_staging(self):
        """Codex-4: a reused driver's LIVE accumulator must not survive a
        checkpoint load — save_checkpoint prefers the live one, so a stale
        partial interval would overwrite the payload just loaded."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver.load_checkpoint)
        assert "self._mpas_sfc_accum = None" in src, (
            "MPAS checkpoint load must drop the live flux accumulator")

    def test_save_prefers_live_then_staged(self):
        """Save path contract: live accumulator WITH samples wins; else the
        staged payload is forwarded (load -> save without a run)."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver.save_checkpoint)
        assert "_facc.has_samples()" in src
        assert 'k.startswith("cmor_flux")' in src

    def test_feed_off_run_drops_staged_partial_interval(self):
        """Codex-5: a feed-OFF link (multi-rank / CMOR off) that loads a
        staged partial interval must DROP it — forwarding those sums would
        let a later feed-ON link resume samples that skip this link's
        steps, producing a wrong interval mean."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._run_mpas)
        assert "DROPPED a partial diag" in src, (
            "feed-off runs must drop (and log) staged cmor_flux* payloads")
        # The drop must be in the else-branch of the accumulator build,
        # i.e. reachable only when the feed is off.
        idx_build = src.index("if self._mpas_cmip_feed_on else None")
        idx_drop = src.index("DROPPED a partial diag")
        assert idx_drop > idx_build


class TestPartialWindowGate:
    """Codex-6: the first SHORT window after an off-cadence restart or a
    feed-off gap must not publish a partial mean labeled as a full
    interval — the flux fields are withheld for that window."""

    def _field(self, v, n=4):
        import types
        return types.SimpleNamespace(data=np.full(n, float(v)))

    def test_is_complete_tracks_step_count(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum(expected_steps=3)
        acc.add((None, None, self._field(1.0)))
        assert not acc.is_complete()
        acc.add((None, None, self._field(2.0)))
        acc.add((None, None, self._field(3.0)))
        assert acc.is_complete()
        acc.reset()
        assert acc._steps == 0

    def test_steps_count_even_without_exports(self):
        """A step whose physics exported nothing still consumed model time."""
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum(expected_steps=2)
        acc.add(None)
        assert not acc.is_complete()
        acc.add(None)
        assert acc.is_complete()

    def test_expected_zero_disables_gate(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum()
        acc.add((None, None, self._field(1.0)))
        assert acc.is_complete()

    def test_steps_survive_checkpoint_roundtrip(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum(expected_steps=4)
        acc.add((None, None, self._field(2.0)))
        acc.add((None, None, self._field(4.0)))
        payload = acc.dump()
        assert "cmor_fluxsteps" in payload
        acc2 = _MPASSfcFluxAccum(expected_steps=4)
        acc2.restore(dict(payload))
        assert not acc2.is_complete()          # 2 of 4 so far
        acc2.add((None, None, self._field(6.0)))
        acc2.add((None, None, self._field(8.0)))
        assert acc2.is_complete()              # whole window, across links
        np.testing.assert_allclose(acc2.mean(2), 5.0)

    def test_restore_without_steps_key_raises(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        with pytest.raises(ValueError, match="cmor_fluxsteps"):
            _MPASSfcFluxAccum(expected_steps=4).restore(
                {"cmor_fluxsum_2": np.zeros(4),
                 "cmor_fluxcnt_2": np.int64(2)})

    def test_feed_withholds_flux_on_partial_window(self, mesh):
        """Partial window: pr must be ABSENT (not the accumulated partial
        mean, and not the instantaneous fallback either)."""
        import types
        from legoesm.driver.model_driver import (
            ModelDriver, _MPASSfcFluxAccum)

        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]

        def _field(a):
            return types.SimpleNamespace(data=np.asarray(a))

        acc = _MPASSfcFluxAccum(expected_steps=4)     # window needs 4 steps
        acc.add((None, None, _field(np.full(n, 2.0e-5))))
        acc.add((None, None, _field(np.full(n, 4.0e-5))))   # only 2 seen

        u_edge = np.zeros((int(mesh.nEdges), NLEV))
        fake = types.SimpleNamespace(
            diagnostics=dc, grid=mesh, _mpas_sfc_accum=acc,
            config=types.SimpleNamespace(
                output=types.SimpleNamespace(diag_days=1.0),
                dycore=types.SimpleNamespace(dt=21600.0)),   # 4 steps = 1 d
            state=types.SimpleNamespace(
                u=_field(u_edge), T=_field(f["T"]), p_s=_field(f["p_s"]),
                phis=_field(f["phis"]), tracers=None),
            model=types.SimpleNamespace(
                _sfc_diag=(None, None, _field(np.full(n, 9.0e-5)))),
        )
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        assert "field_2d_pr" not in out, "partial-window flux must be withheld"
        assert "field_2d_tas" in out, "state snapshots still feed"
        assert acc.mean(2) is None                    # reset for next window

    def test_feed_publishes_complete_window(self, mesh):
        import types
        from legoesm.driver.model_driver import (
            ModelDriver, _MPASSfcFluxAccum)

        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]

        def _field(a):
            return types.SimpleNamespace(data=np.asarray(a))

        acc = _MPASSfcFluxAccum(expected_steps=2)
        acc.add((None, None, _field(np.full(n, 2.0e-5))))
        acc.add((None, None, _field(np.full(n, 4.0e-5))))

        u_edge = np.zeros((int(mesh.nEdges), NLEV))
        fake = types.SimpleNamespace(
            diagnostics=dc, grid=mesh, _mpas_sfc_accum=acc,
            config=types.SimpleNamespace(
                output=types.SimpleNamespace(diag_days=1.0),
                dycore=types.SimpleNamespace(dt=43200.0)),   # 2 steps = 1 d
            state=types.SimpleNamespace(
                u=_field(u_edge), T=_field(f["T"]), p_s=_field(f["p_s"]),
                phis=_field(f["phis"]), tracers=None),
            model=types.SimpleNamespace(
                _sfc_diag=(None, None, _field(np.full(n, 9.0e-5)))),
        )
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        np.testing.assert_allclose(out["field_2d_pr"], 3.0e-5, rtol=1e-6)


class TestWindowIdentity:
    """Codex-7: a resumed window must be DISCARDED when its identity does
    not match this run (cadence changed, or the calendar was rebased), and
    completion is EXACT (an overlong window is not 'complete')."""

    def _field(self, v, n=4):
        import types
        return types.SimpleNamespace(data=np.full(n, float(v)))

    def _payload(self, expected, steps, day0):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum(expected_steps=expected,
                                window_start_day=day0, dt_s=10800.0)
        for _ in range(steps):
            acc.add((None, None, self._field(2.0)))
        return acc.dump()

    def test_overlong_window_is_not_complete(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum(expected_steps=6)
        for _ in range(10):
            acc.add((None, None, self._field(1.0)))
        assert not acc.is_complete(), "overlong window must not publish"

    def test_cadence_change_discards_payload(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        staged = self._payload(expected=6, steps=4, day0=10.0)
        acc = _MPASSfcFluxAccum(expected_steps=8, window_start_day=10.5,
                                dt_s=10800.0)
        assert acc.restore(dict(staged), resume_day=10.5, dt_s=10800.0) == 0
        assert not acc.has_samples()

    def test_calendar_rebase_discards_payload(self):
        """--restart-start-day moves the epoch: the carried window would
        continue at day 10.5 but the run resumes at day 100."""
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        staged = self._payload(expected=6, steps=4, day0=10.0)
        acc = _MPASSfcFluxAccum(expected_steps=6, window_start_day=100.0,
                                dt_s=10800.0)
        assert acc.restore(dict(staged), resume_day=100.0,
                           dt_s=10800.0) == 0
        assert not acc.has_samples()

    def test_contiguous_resume_is_accepted(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        dt = 10800.0                       # 3 h steps, 8 per day
        staged = self._payload(expected=8, steps=3, day0=10.0)
        resume = 10.0 + 3 * dt / 86400.0
        acc = _MPASSfcFluxAccum(expected_steps=8, window_start_day=resume,
                                dt_s=dt)
        assert acc.restore(dict(staged), resume_day=resume, dt_s=dt) == 1
        assert acc._steps == 3 and acc.window_start_day == 10.0
        for _ in range(5):
            acc.add((None, None, self._field(2.0)))
        assert acc.is_complete()           # 8 steps == one full day

    def test_staged_keys_always_popped_on_discard(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        staged = self._payload(expected=6, steps=4, day0=10.0)
        acc = _MPASSfcFluxAccum(expected_steps=8, window_start_day=10.5,
                                dt_s=10800.0)
        acc.restore(staged, resume_day=10.5, dt_s=10800.0)
        assert not [k for k in staged if k.startswith("cmor_flux")], (
            "a discarded payload must not be left staged for re-save")

    def test_reset_rolls_window_origin(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum(expected_steps=4, window_start_day=1.0)
        acc.add((None, None, self._field(1.0)))
        acc.reset(window_start_day=2.0)
        assert acc.window_start_day == 2.0 and acc._steps == 0

    def test_dt_change_discards_payload(self):
        """Codex-8: same step count, different dt = different duration."""
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        staged = self._payload(expected=8, steps=3, day0=10.0)  # dt 10800
        resume = 10.0 + 3 * 10800.0 / 86400.0
        acc = _MPASSfcFluxAccum(expected_steps=8, window_start_day=resume,
                                dt_s=10700.0)
        assert acc.restore(dict(staged), resume_day=resume,
                           dt_s=10700.0) == 0

    def test_missing_identity_key_discards_payload(self):
        """A payload without full identity cannot be validated → discard."""
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        staged = self._payload(expected=8, steps=3, day0=10.0)
        del staged["cmor_fluxdt_s"]
        acc = _MPASSfcFluxAccum(expected_steps=8, window_start_day=10.375,
                                dt_s=10800.0)
        assert acc.restore(staged, resume_day=10.375, dt_s=10800.0) == 0
        assert not [k for k in staged if k.startswith("cmor_flux")]

    def test_small_rebase_no_longer_slips_through(self):
        """A sub-step calendar rebase must be caught (was accepted under the
        old half-step tolerance)."""
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        dt = 3600.0
        staged = self._payload(expected=24, steps=1, day0=10.0)
        # helper builds at dt=10800; rebuild at dt=3600 for this check
        from legoesm.driver.model_driver import _MPASSfcFluxAccum as A
        src = A(expected_steps=24, window_start_day=10.0, dt_s=dt)
        src.add((None, None, self._field(1.0)))
        staged = src.dump()
        contiguous = 10.0 + dt / 86400.0
        acc = A(expected_steps=24, window_start_day=contiguous, dt_s=dt)
        # 100 s rebase — under the old 0.5*dt slack this was accepted
        assert acc.restore(dict(staged),
                           resume_day=contiguous + 100.0 / 86400.0,
                           dt_s=dt) == 0
        assert acc.restore(dict(staged), resume_day=contiguous, dt_s=dt) == 1


class TestBinningPhaseGate:
    def test_driver_requires_on_grid_window_phase(self):
        """Codex-8: midpoint binning also needs the window to sit ON the
        1/N-day grid, else it can straddle midnight/month boundaries."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(
            ModelDriver._feed_mpas_cmip_accumulators)
        assert "_phase" in src and "round(_phase)" in src

    def test_driver_labels_true_sampling_cadence(self):
        """The snapshot cadence label uses DIAG_INTERVAL*DT, not the
        requested diag_days that step arithmetic truncated."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._run_mpas)
        assert "DIAG_INTERVAL * DT / 86400.0" in src


class TestDriftingCadenceHonesty:
    """Codex-9: a cadence that is not a whole number of days drifts through
    the day, so the metadata must not claim a fixed 00 UTC phase or a
    single sample per day."""

    def test_daily_attrs_declare_drift(self, mesh):
        dc, _ = _make_collector(mesh)
        dc.cmip_snapshot_vars = {"tas"}
        dc.cmip_snapshot_cadence_days = 8 * 10000.0 / 86400.0   # 0.9259 d
        attrs = dc._daily_snapshot_attrs()
        assert "DRIFTING" in attrs["tas"]["comment"]
        assert "00 UTC" not in attrs["tas"]["comment"]
        assert "0-2" in attrs["tas"]["comment"]
        assert "drifting" in attrs["tasmax"]["comment"]
        # A day cell that is a MEAN of 0-2 points is not a "point": keep the
        # table's own cell_methods and disclose in the comment.
        assert "cell_methods" not in attrs["tas"]
        assert "cell_methods" not in attrs["tasmax"]

    def test_daily_attrs_fixed_phase_when_whole_days(self, mesh):
        dc, _ = _make_collector(mesh)
        dc.cmip_snapshot_vars = {"tas"}
        dc.cmip_snapshot_cadence_days = 2.0
        dc.cmip_snapshot_phase_frac = 0.0        # integral start day
        attrs = dc._daily_snapshot_attrs()
        assert "00 UTC" in attrs["tas"]["comment"]
        assert "DRIFTING" not in attrs["tas"]["comment"]
        # Exactly one sample per day cell -> "time: point" is literally true.
        assert attrs["tas"]["cell_methods"] == "time: point"

    def test_monthly_comment_drops_fixed_phase_claim(self, mesh, tmp_path):
        xr = pytest.importorskip("xarray")
        from legoesm.io.cmor_output import CFWriter
        dc, _ = _make_collector(mesh)
        dc.cf_writer = CFWriter(output_dir=tmp_path, experiment_id="amip",
                                model_id="legoESM", ref_date="1979-01-01")
        dc.cmip_snapshot_vars = {"tas"}
        dc.cmip_snapshot_cadence_days = 0.9259259259259259
        nlat, nlon = dc._cmip_nlat, dc._cmip_nlon
        dc._write_cmip_data({"months": [(0, 1)],
                             "field_2d_tas": np.full((1, nlat, nlon), 288.0)})
        with xr.open_dataset(sorted(tmp_path.rglob("tas_*.nc"))[0]) as ds:
            assert "DRIFTING" in ds["tas"].attrs["comment"]
            assert "fixed phase" not in ds["tas"].attrs["comment"]

    def test_phase_tolerance_is_absolute_in_days(self):
        """A relative phase tolerance would admit an off-grid window by ~an
        hour after a century; the gate must use an absolute day tolerance."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._feed_mpas_cmip_accumulators)
        assert "_phase_err_days" in src and "1e-9" in src
        # And the arithmetic itself: an off-grid 1-day window at day 36500
        # must be rejected.
        day, win = 36500.0 + 3154.0 / 86400.0, 1.0
        phase = day / win
        assert abs(phase - round(phase)) * win >= 1e-9

    def test_phase_text_derived_not_asserted(self, mesh):
        """Codex-10: a fractional start_day shifts the sampling time of day;
        the metadata must report the derived phase, not claim 00 UTC."""
        dc, _ = _make_collector(mesh)
        dc.cmip_snapshot_vars = {"tas"}
        dc.cmip_snapshot_cadence_days = 1.0
        dc.cmip_snapshot_phase_frac = 0.5          # noon start
        attrs = dc._daily_snapshot_attrs()
        assert "12:00 UTC" in attrs["tas"]["comment"]
        assert "00 UTC," not in attrs["tas"]["comment"]
        dc.cmip_snapshot_phase_frac = 0.0
        assert "00 UTC" in dc._daily_snapshot_attrs()["tas"]["comment"]
        dc.cmip_snapshot_phase_frac = None
        assert "fixed time of day" in (
            dc._daily_snapshot_attrs()["tas"]["comment"])

    def test_driver_derives_phase_from_start_day(self):
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._run_mpas)
        assert "cmip_snapshot_phase_frac" in src and "% 1.0" in src

    def test_sentinel_cadence_still_declares_snapshots(self):
        """Codex-11: diag_days<=0 is a once-at-end FEED on this lane, not
        'no diagnostics' — its single end-of-run state sample must still be
        disclosed, not written with the table's time: mean."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._run_mpas)
        assert "float(cfg.output.diag_days) <= 0.0" in src
        assert "_true_cad_days >= 1.0" in src

    def test_phase_boundary_restart_uses_next_sample(self):
        """On a restart exactly on a cadence boundary the next sample is a
        FULL interval ahead, not the current position."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._run_mpas)
        assert "_rem = DIAG_INTERVAL - (DIAG_PHASE % DIAG_INTERVAL)" in src
        # arithmetic: on-boundary start_step -> a full interval, never 0
        for start_step, interval in ((0, 8), (8, 8), (16, 8), (3, 8)):
            rem = interval - (start_step % interval)
            assert 1 <= rem <= interval

    def test_phase_text_rounds_across_midnight_to_00utc(self, mesh):
        dc, _ = _make_collector(mesh)
        dc.cmip_snapshot_phase_frac = 1.0 - 1e-7      # 8.6 ms before midnight
        assert dc._snapshot_phase_text() == "00 UTC"

    @staticmethod
    def _phase_frac(diag_days, start_day, start_step, interval, dt):
        """Mirror of the driver's phase derivation (both branches)."""
        diag_phase = start_step if diag_days > 0 else 0
        rem = interval - (diag_phase % interval)
        return (start_day + rem * dt / 86400.0) % 1.0

    def test_sentinel_phase_matches_actual_sample_day(self):
        """Codex-12: the sentinel's trigger is job-LOCAL, so its single
        sample lands a full local interval after the restart — the phase
        must say so, not derive from start_step.

        2-day link, dt=6 h (interval=8), start_step=10, START_DAY=2.5:
        the sample is at day 4.5 = 12:00 UTC.
        """
        dt, interval, start_day, start_step = 21600.0, 8, 2.5, 10
        frac = self._phase_frac(0.0, start_day, start_step, interval, dt)
        assert abs(frac - 0.5) < 1e-12, "sentinel sample is at 12:00 UTC"
        # The pre-fix derivation (start_step-based) claimed 00 UTC.
        stale = (start_day
                 + (interval - (start_step % interval)) * dt / 86400.0) % 1.0
        assert abs(stale - 0.0) < 1e-12

    def test_periodic_phase_still_absolute(self):
        """The periodic path keeps its absolute phase (DIAG_PHASE =
        start_step): same 2-day/6-h layout, first sample at day 4.0."""
        dt, interval, start_day, start_step = 21600.0, 8, 2.5, 10
        frac = self._phase_frac(2.0, start_day, start_step, interval, dt)
        assert abs(frac - 0.0) < 1e-12

    def test_phase_text_of_sentinel_case(self, mesh):
        dc, _ = _make_collector(mesh)
        dc.cmip_snapshot_vars = {"tas"}
        dc.cmip_snapshot_cadence_days = 2.0
        dc.cmip_snapshot_phase_frac = self._phase_frac(
            0.0, 2.5, 10, 8, 21600.0)
        assert "12:00 UTC" in dc._daily_snapshot_attrs()["tas"]["comment"]
