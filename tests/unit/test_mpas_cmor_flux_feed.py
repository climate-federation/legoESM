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


def _as_driver(ns):
    """Bind the ``ModelDriver`` methods the CMOR feed calls on ``self``.

    ``types.SimpleNamespace`` stand-ins cannot inherit them, and the feed
    delegates its native-field construction to ``_mpas_cmip_native_kwargs``
    (shared with the multi-rank gather path).
    """
    import functools
    from legoesm.driver.model_driver import ModelDriver
    ns._mpas_cmip_native_kwargs = functools.partial(
        ModelDriver._mpas_cmip_native_kwargs, ns)
    return ns


@pytest.fixture(scope="module")
def mesh():
    return create_grid("mpas", 2, lloyd_iterations=10)


def _make_collector(mesh, cloud_config=None):
    sigma_full = np.linspace(0.05, 0.98, NLEV)
    dsigma = np.full(NLEV, 1.0 / NLEV)
    dc = DiagnosticCollector(
        nlev=NLEV, sigma_full=sigma_full, dsigma=dsigma,
        experiment_id="amip", monthly_means=True,
        cmip_output=True, n_days=30, output_dir=None,
        cmip_resolution_deg=10.0, start_year=1979,
        cloud_config=cloud_config,
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


FLUXES = dict(rlut=238.0, rsut=99.0, rsdt=340.0, hfss=17.0, hfls=88.0,
              # Clear-sky pair (#843): physically rsutcs <= rsut (clear sky
              # reflects LESS) and rlutcs >= rlut (clear sky emits MORE).
              rsutcs=77.0, rlutcs=262.0)


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
        # Clear-sky pair (#843): default None => leaf absent, byte-identical.
        assert t.sw_up_toa_clr is None
        assert t.lw_up_toa_clr is None

    def test_toa_fields_packed_with_values(self, mesh):
        import jax.numpy as jnp
        n = int(mesh.nCells)
        t = self._pack(
            mesh,
            sw_up_toa=jnp.full(n, 99.0),
            lw_up_toa=jnp.full(n, 238.0),
            sw_down_toa=jnp.full(n, 340.0),
            sw_up_toa_clr=jnp.full(n, 77.0),
            lw_up_toa_clr=jnp.full(n, 262.0))
        np.testing.assert_allclose(np.asarray(t.sw_up_toa.data), 99.0)
        np.testing.assert_allclose(np.asarray(t.lw_up_toa.data), 238.0)
        np.testing.assert_allclose(np.asarray(t.sw_down_toa.data), 340.0)
        np.testing.assert_allclose(np.asarray(t.sw_up_toa_clr.data), 77.0)
        np.testing.assert_allclose(np.asarray(t.lw_up_toa_clr.data), 262.0)
        assert t.lw_up_toa.units == "W/m^2"
        assert t.lw_up_toa_clr.units == "W/m^2"


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
        sw_up_toa_clr=_f("sw_up_toa_clr", 11.0),
        lw_up_toa_clr=_f("lw_up_toa_clr", 12.0),
        sed_substeps_required=Field(
            data=np.full(3, 13, dtype=np.int32), name="sed_substeps_required",
            dims=("cell",), units="1"))


# The ONE slot contract both producers build from (core.state).  Spelled out
# here so a silent edit to the shared constant still has to face the explicit
# slot map below; a drift swaps rlut/rsut/rsdt, misindexes the land-forcing
# slots 8/9, or drops the clear-sky pair.
_EXTRA_ORDER = ("lw_up_toa", "sw_up_toa", "sw_down_toa",
                "shflx_sfc", "lhflx_sfc",
                "sw_down_sfc", "lw_down_sfc",
                "sw_up_toa_clr", "lw_up_toa_clr",
                # slot 12 (2026-09-22): the microphysics' required CFL
                # sedimentation sub-step count, so an overflow is visible in
                # a real run on BOTH producers.
                "sed_substeps_required")


class TestSfcDiagContract:
    """Lock the 12-slot sfc_diag tuple contract of BOTH producers (serial
    primitive_eq_mpas._step_jit and MPI parallel.voronoi_mpi._step) and the
    driver consumer's slot mapping (3-7 all-sky fluxes, 8/9 land-forcing
    downwelling, 10/11 clear-sky)."""

    def test_shared_contract_is_the_one_both_producers_import(self):
        """Both producers must build ``_extras`` from the SAME constant — the
        drift that hand-maintained copies actually suffered: the MPI producer
        stopped at slot 7 while the consumer read slots 10/11, so a ONE-rank
        Voronoi MPI run (which _mpas_cmip_feed_enabled turns the feed ON for)
        accepted --clear-sky-diag and published no rsutcs/rlutcs."""
        from legoesm.core.state import (
            MPAS_SFC_DIAG_EXTRA_KEYS,
            MPAS_SFC_DIAG_MPI_UNPUBLISHED,
        )
        assert MPAS_SFC_DIAG_EXTRA_KEYS == _EXTRA_ORDER
        # Both modules must reference the shared name, not a literal copy.
        for mod, sym in (
            ("legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas",
             "MPAS_SFC_DIAG_EXTRA_KEYS"),
            ("legoesm.parallel.voronoi_mpi", "MPAS_SFC_DIAG_EXTRA_KEYS"),
            ("legoesm.parallel.voronoi_mpi", "MPAS_SFC_DIAG_MPI_UNPUBLISHED"),
        ):
            m = __import__(mod, fromlist=[sym])
            assert getattr(m, sym) is (
                MPAS_SFC_DIAG_EXTRA_KEYS
                if sym == "MPAS_SFC_DIAG_EXTRA_KEYS"
                else MPAS_SFC_DIAG_MPI_UNPUBLISHED), f"{mod}.{sym}"
        # The MPI lane may leave slots EMPTY but never SHORTEN the tuple: a
        # key it does not publish must still be a member of the contract.
        for _k in MPAS_SFC_DIAG_MPI_UNPUBLISHED:
            assert _k in MPAS_SFC_DIAG_EXTRA_KEYS
        # ...and it must NOT drop the clear-sky pair (the blocker above).
        assert "sw_up_toa_clr" not in MPAS_SFC_DIAG_MPI_UNPUBLISHED
        assert "lw_up_toa_clr" not in MPAS_SFC_DIAG_MPI_UNPUBLISHED
        # #1321: the surface downwelling pair is PUBLISHED on the MPI lane —
        # withholding it made _marshal_land_forcing return None every step, so
        # the interactive multilayer land silently never advanced.
        assert "sw_down_sfc" not in MPAS_SFC_DIAG_MPI_UNPUBLISHED
        assert "lw_down_sfc" not in MPAS_SFC_DIAG_MPI_UNPUBLISHED

    def test_mpi_producer_publishes_clear_sky_at_contract_slots(self):
        """Rebuild the MPI producer's extras expression on a tendency that
        carries every field: it must be 12 slots long with the clear-sky pair
        at 10/11 (an 8-slot tuple is the defect this locks out)."""
        from legoesm.core.state import (
            MPAS_SFC_DIAG_EXTRA_KEYS,
            MPAS_SFC_DIAG_MPI_UNPUBLISHED,
        )
        _pt = _tend_with_extras()
        _extras = tuple(
            None if _k in MPAS_SFC_DIAG_MPI_UNPUBLISHED
            else getattr(_pt, _k, None)
            for _k in MPAS_SFC_DIAG_EXTRA_KEYS)
        sfc = (_pt.sw_net_sfc, _pt.lw_net_sfc, _pt.precip) + _extras
        assert len(sfc) == 13
        assert sfc[10].name == "sw_up_toa_clr"   # rsutcs
        assert sfc[11].name == "lw_up_toa_clr"   # rlutcs
        # the MPI producer publishes the sub-step count at the same slot the
        # serial one does, so the driver's overflow report works on both
        from legoesm.driver.model_driver import _sed_substeps_slot
        assert _sed_substeps_slot() == 12
        assert sfc[12].name == "sed_substeps_required"
        # #1321: the land downwelling pair is now PUBLISHED at slots 8/9.
        # While it was withheld, ``_marshal_land_forcing``'s ``_sd[8] is None``
        # guard declined every step and the Richards soil never advanced.
        assert sfc[8].name == "sw_down_sfc"
        assert sfc[9].name == "lw_down_sfc"

    def test_producer_slot_order_matches_consumer(self):
        _pt = _tend_with_extras()
        _extras = tuple(getattr(_pt, _k, None) for _k in _EXTRA_ORDER)
        sfc_diag = (_pt.sw_net_sfc, _pt.lw_net_sfc, _pt.precip) + _extras
        assert len(sfc_diag) == 13
        # Consumer (_feed_mpas_cmip_accumulators): slot 3->rlut, 4->rsut,
        # 5->rsdt, 6->hfss, 7->hfls, 10->rsutcs, 11->rlutcs; slots 8/9 are
        # the _marshal_land_forcing downwelling pair.
        assert sfc_diag[3].name == "lw_up_toa"    # rlut
        assert sfc_diag[4].name == "sw_up_toa"    # rsut
        assert sfc_diag[5].name == "sw_down_toa"  # rsdt
        assert sfc_diag[6].name == "shflx"        # hfss
        assert sfc_diag[7].name == "lhflx"        # hfls
        assert sfc_diag[8].name == "sw_down_sfc"  # land forcing
        assert sfc_diag[9].name == "lw_down_sfc"  # land forcing
        assert sfc_diag[10].name == "sw_up_toa_clr"  # rsutcs
        assert sfc_diag[11].name == "lw_up_toa_clr"  # rlutcs

    def test_both_producers_build_extras_from_the_shared_contract(self):
        """Assert against the symbols that RUN — the serial
        ``MPASPrimitiveEquationModel._step_jit`` and the MPI
        ``make_voronoi_mpi_step`` — that neither re-introduces a hand-written
        key list.  Checked on the AST of the ``_extras`` assignment, NOT on
        the source text: a text match is satisfied by a passing mention in a
        COMMENT (verified — a text-matching version of this test passed
        against a deliberately regressed 8-slot MPI producer)."""
        import ast
        import inspect
        import textwrap

        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
        )
        from legoesm.parallel.voronoi_mpi import make_voronoi_mpi_step

        for fn, tag in ((MPASPrimitiveEquationModel._step_jit,
                         "primitive_eq_mpas._step_jit"),
                        (make_voronoi_mpi_step,
                         "voronoi_mpi.make_voronoi_mpi_step")):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            rhs = [n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_extras"
                           for t in n.targets)]
            assert len(rhs) == 1, f"{tag}: expected one _extras assignment"
            names = {n.id for n in ast.walk(rhs[0])
                     if isinstance(n, ast.Name)}
            assert "MPAS_SFC_DIAG_EXTRA_KEYS" in names, (
                f"{tag} no longer builds sfc_diag extras from the shared "
                "contract in core.state — a hand-copied key list is exactly "
                "the drift that dropped rsutcs/rlutcs on the one-rank MPI "
                "lane.")
            strs = {n.value for n in ast.walk(rhs[0])
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            assert not strs, (
                f"{tag} hard-codes contract keys {sorted(strs)} again; every "
                "key must come from MPAS_SFC_DIAG_EXTRA_KEYS.")
            # ...and consumes the WHOLE contract: `KEYS[:5]` would satisfy the
            # two checks above while re-creating the short-tuple defect.
            assert not any(isinstance(n, ast.Subscript)
                           for n in ast.walk(rhs[0])), (
                f"{tag} slices MPAS_SFC_DIAG_EXTRA_KEYS; the producer must "
                "publish EVERY contract slot (leave one empty via "
                "MPAS_SFC_DIAG_MPI_UNPUBLISHED, never truncate).")
            # ...and the published tuple is actually built from `_extras`,
            # not from some other expression while `_extras` is computed and
            # dropped.
            concat = [n for n in ast.walk(tree)
                      if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add)
                      and isinstance(n.right, ast.Name)
                      and n.right.id == "_extras"]
            assert concat, (
                f"{tag} computes _extras but never concatenates it onto the "
                "published sfc_diag tuple.")
            # ...and the concatenation is not truncated afterwards:
            # ``((base) + _extras)[:8]`` would satisfy every check above.
            for sub in (n for n in ast.walk(tree)
                        if isinstance(n, ast.Subscript)):
                assert not any(isinstance(x, ast.Name) and x.id == "_extras"
                               for x in ast.walk(sub)), (
                    f"{tag} slices a tuple built from _extras; the published "
                    "sfc_diag must carry EVERY contract slot.")

    def test_replace_preserves_cmor_diagnostics(self):
        """The HS wrapper repacks via ``_replace`` of the 4 dynamics fields
        only; every diagnostic (sw/lw net, precip, TOA trio, turb fluxes,
        downwelling pair, clear-sky pair) must survive — the invariant that
        fix relies on."""
        _pt = _tend_with_extras()
        summed = _pt._replace(dT_dt=_pt.dT_dt.replace(data=np.ones(3)))
        for _k in ("sw_net_sfc", "lw_net_sfc", "precip", "lw_up_toa",
                   "sw_up_toa", "sw_down_toa", "shflx_sfc", "lhflx_sfc",
                   "sw_down_sfc", "lw_down_sfc",
                   "sw_up_toa_clr", "lw_up_toa_clr"):
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
        _as_driver(fake)
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
        _as_driver(fake)
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        np.testing.assert_allclose(out["field_2d_pr"], 9.0e-5, rtol=1e-9)


class TestMeanCellMethods:
    def test_state_vars_written_time_mean(self, mesh, tmp_path):
        """State and flux means retain time: mean in actual NetCDF files."""
        xr = pytest.importorskip("xarray")
        from legoesm.io.cmor_output import CFWriter

        dc, _ = _make_collector(mesh)
        dc.cf_writer = CFWriter(
            output_dir=tmp_path, experiment_id="amip",
            model_id="legoESM", ref_date="1979-01-01")

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
        # The producer supplies the TIME clause; the writer keeps the
        # table's AREA clause (Amon tas/pr are "area: time: mean"), so the
        # honesty override must not silently drop "area:".
        with xr.open_dataset(tas_files[0]) as ds:
            assert (ds["tas"].attrs["cell_methods"]
                    == "area: time: mean")
            assert "aliased" not in ds["tas"].attrs.get("comment", "")
        with xr.open_dataset(pr_files[0]) as ds:
            assert ds["pr"].attrs["cell_methods"] == "area: time: mean"
            assert "comment" not in ds["pr"].attrs

    def test_default_none_keeps_table_cell_methods(self, mesh, tmp_path):
        """The shared writer keeps mean semantics for segment means too."""
        xr = pytest.importorskip("xarray")
        from legoesm.io.cmor_output import CFWriter

        dc, _ = _make_collector(mesh)
        dc.cf_writer = CFWriter(
            output_dir=tmp_path, experiment_id="amip",
            model_id="legoESM", ref_date="1979-01-01")
        assert not hasattr(dc, "cmip_snapshot_vars")

        nlat, nlon = dc._cmip_nlat, dc._cmip_nlon
        data = {
            "months": [(0, 1)],
            "field_2d_tas": np.full((1, nlat, nlon), 288.0),
        }
        dc._write_cmip_data(data)
        tas_files = sorted(tmp_path.rglob("tas_*.nc"))
        assert tas_files
        with xr.open_dataset(tas_files[0]) as ds:
            # The official Amon entry, untouched: "area: time: mean".
            assert ds["tas"].attrs["cell_methods"] == "area: time: mean"


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
        # The gate lives in _mpas_cmip_native_kwargs, the field-building
        # helper _feed_mpas_cmip_accumulators delegates to (and which the
        # multi-rank gather path shares) — inspect the symbol that RUNS.
        src = inspect.getsource(
            model_driver.ModelDriver._mpas_cmip_native_kwargs)
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

    def test_wallclock_daily_write_carries_hourly_extreme_attrs(self, mesh):
        """finalize_cmip_daily (the wallclock-exit path) must pass the same
        per-var honesty overrides as the end-of-run writer (codex-2
        finding 4)."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            15.0, **f, precip=np.full(n, 3.0e-5), flux_interval_days=1.0)

        dc.feed_daily_extremes_native(15.0, tas=f["T"][:, -1])
        calls = {}

        class _W:
            def write_daily(self, data, lat, lon, extra_attrs_by_var=None):
                calls["extra"] = extra_attrs_by_var
                return []
        dc.cf_writer = _W()
        dc.finalize_cmip_daily(current_day=16.0)
        assert calls["extra"] is not None
        assert "tas" not in calls["extra"]
        assert "tasmax" in calls["extra"]
        assert "pr" not in calls["extra"]     # true interval mean

    def test_extreme_attrs_none_without_hourly_samples(self, mesh):
        dc, _ = _make_collector(mesh)
        assert dc._daily_extreme_attrs() is None

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
        _as_driver(fake)
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
        _as_driver(fake)
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

    def test_full_window_with_every_energy_slot_fed_is_ready(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        acc = _MPASSfcFluxAccum(expected_steps=8, window_start_day=0.0,
                                dt_s=10800.0)
        for _ in range(8):
            acc.add(tuple(self._field(1.0) for _ in range(8)))
        assert acc.window_ready(acc.ENERGY_SLOTS)
        acc.add(tuple(self._field(1.0) for _ in range(8)))   # overlong
        assert not acc.window_ready(acc.ENERGY_SLOTS)

    def test_legacy_checkpoint_leaves_new_slots_short_and_not_ready(self):
        """A checkpoint written before slots 0/1 existed restores slot 2 with
        the pre-restart count; slots 0/1 then see only the remainder.  The
        window completes (step count matches) but the energy slots are NOT
        interval means over it -- ``window_ready`` must refuse (#1354)."""
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        dt = 10800.0
        staged = self._payload(expected=8, steps=3, day0=10.0)   # slot 2 only
        resume = 10.0 + 3 * dt / 86400.0
        acc = _MPASSfcFluxAccum(expected_steps=8, window_start_day=resume,
                                dt_s=dt)
        assert acc.restore(dict(staged), resume_day=resume, dt_s=dt) == 1
        for _ in range(5):
            acc.add((self._field(1.0), self._field(1.0), self._field(2.0)))
        assert acc.is_complete()
        assert acc.mean(0) is not None and acc.mean(2) is not None
        assert acc.window_ready((2,))
        assert not acc.window_ready((0, 1, 2)), (
            "slots 0/1 cover 5 of 8 steps; must not be stamped interval mean")

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
        src = inspect.getsource(ModelDriver._mpas_cmip_native_kwargs)
        assert "_phase" in src and "round(_phase)" in src


class TestFluxPhaseTolerance:
    """Flux interval binning retains an absolute day tolerance."""


    def test_phase_tolerance_is_absolute_in_days(self):
        """A relative phase tolerance would admit an off-grid window by ~an
        hour after a century; the gate must use an absolute day tolerance."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._mpas_cmip_native_kwargs)
        assert "_phase_err_days" in src and "1e-9" in src
        # And the arithmetic itself: an off-grid 1-day window at day 36500
        # must be rejected.
        day, win = 36500.0 + 3154.0 / 86400.0, 1.0
        phase = day / win
        assert abs(phase - round(phase)) * win >= 1e-9


# ---------------------------------------------------------------------------
# #843 lean-lane port: cloud CMOR trio (clt / clwvi / clivi) + clear-sky pair
# ---------------------------------------------------------------------------


def _cloud_collector(mesh):
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    return _make_collector(
        mesh, cloud_config=build_cloud_config("sundqvist",
                                              convective_cloud=False))


class TestCloudCmorFeed:
    """clwvi/clivi/clt derived by ``feed_cmip_accumulators_native`` from the
    cloud tracers: the RADIATIVE water path (grid-mean lwp/iwp as fed to the
    cloud optics, incl. the cf*q_c_diagnostic floor) and the max-random
    stratiform total cover."""

    def test_explicit_condensate_paths_exact(self, mesh):
        """Synthetic column with a bone-dry RH (cf=0 -> no diagnostic floor)
        and uniform explicit q_c/q_i: the radiative path reduces EXACTLY to
        the prognostic column integral sum(q * dp / g) = q * p_s / g."""
        from legoesm import constants

        dc, sigma_full = _cloud_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        q_c = np.full((n, NLEV), 2.0e-4)
        q_i = np.full((n, NLEV), 1.0e-4)
        # RH ~ 0 everywhere -> sundqvist cf = 0 -> floor contributes nothing.
        q_v = np.full((n, NLEV), 1.0e-10)
        fed = dc.feed_cmip_accumulators_native(
            day=15.0, **f, q_v=q_v, q_c=q_c, q_i=q_i)
        assert fed is True
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        p_s = 1.0e5
        # dsigma sums to 1 -> sum(dp) = p_s exactly (pure-sigma collector).
        expect_clivi = 1.0e-4 * p_s / constants.g
        expect_clwvi = 3.0e-4 * p_s / constants.g
        # rtol is float32-safe: the path integral runs through JAX float32
        # reductions under the default precision policy (observed rel err
        # ~1.5e-9 at x64, ~1e-7 at float32); this is a SEMANTIC mass-path
        # test, not a bit-precision contract.
        np.testing.assert_allclose(out["field_2d_clivi"], expect_clivi,
                                   rtol=1e-6)
        np.testing.assert_allclose(out["field_2d_clwvi"], expect_clwvi,
                                   rtol=1e-6)
        # cf=0 everywhere -> clt = 0 (max-random of a clear column).
        np.testing.assert_allclose(out["field_2d_clt"], 0.0, atol=1e-9)

    def test_paths_are_mass_not_optics_thinned(self, mesh):
        """clwvi/clivi are a MASS content (CMIP6
        ``atmosphere_mass_content_of_cloud_condensed_water``), so the
        cloud-OPTICS sub-grid inhomogeneity and partial-coverage factors —
        which ``compute_cloud_properties`` applies to lwp/iwp before
        returning them — must NOT thin the published path.  With chi = 0.4
        and two_column coverage on the collector's cloud config the answer
        must stay the same exact column integral as
        ``test_explicit_condensate_paths_exact``."""
        from legoesm.atmosphere.physics.clouds.config import build_cloud_config

        thinned = build_cloud_config(
            "sundqvist", convective_cloud=False)._replace(
                cloud_optics_inhomogeneity="constant",
                cloud_inhomogeneity_factor=0.4,
                cloud_partial_coverage_optics="two_column")
        dc, sigma_full = _make_collector(mesh, cloud_config=thinned)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            day=15.0, **f,
            q_v=np.full((n, NLEV), 1.0e-10),
            q_c=np.full((n, NLEV), 2.0e-4),
            q_i=np.full((n, NLEV), 1.0e-4))
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        # float32-safe rtol; see test_explicit_condensate_paths_exact.
        np.testing.assert_allclose(
            out["field_2d_clivi"], 1.0e-4 * 1.0e5 / constants.g, rtol=1e-6)
        np.testing.assert_allclose(
            out["field_2d_clwvi"], 3.0e-4 * 1.0e5 / constants.g, rtol=1e-6)

    def test_water_path_note_written_without_snapshot_vars(self, mesh):
        """The "WHICH water path" note is a SEMANTICS statement, not a
        sampling one: a sub-daily MPAS cadence sets no ``cmip_snapshot_vars``
        and must still carry it.  The cube lane (prognostic clwvi, flag never
        set) must not."""
        dc, sigma_full = _cloud_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            day=15.0, **f, q_v=np.full((n, NLEV), 1.0e-10),
            q_c=np.full((n, NLEV), 2.0e-4))
        assert dc._cloud_paths_radiative is True

        captured = {}

        class _W:
            def write_field(self, *, var_name, extra_attrs=None, **kw):
                captured[var_name] = extra_attrs

            def __getattr__(self, _name):     # any other writer call: no-op
                return lambda *a, **k: None

        dc.cf_writer = _W()
        dc._write_cmip_data(
            {"months": [(0, 1)],
             "field_2d_clwvi": [np.zeros((dc._cmip_nlat, dc._cmip_nlon))],
             "field_2d_rsut": [np.zeros((dc._cmip_nlat, dc._cmip_nlon))]})
        _c = captured["clwvi"]["comment"]
        assert "DIAGNOSTIC condensed-water path" in _c
        # It must disclose all three ways a reader would misread it: it is
        # not the prognostic mass (the floor is included), not an optical
        # path (the optics thinning is excluded), and not a record of what
        # radiation actually solved with (stratiform-only, no CLUBB cf).
        assert "NOT the prognostic condensed water mass" in _c
        assert "not an optical path" in _c
        assert "NOT necessarily what the radiation solved with" in _c
        assert captured["rsut"] is None      # untouched fields keep table attrs

    def test_radiative_floor_visible_with_zero_prognostic_condensate(
            self, mesh):
        """THE radiative-path property: a saturated column with ZERO
        prognostic condensate still has a positive water path (the
        cf*q_c_diagnostic in-cloud floor the radiation actually saw) and a
        positive cloud cover.  The prognostic-only definition would report
        exactly 0 here."""
        from legoesm.thermo import saturation_mixing_ratio

        dc, sigma_full = _cloud_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        # RH = 0.97 > rh_crit on every layer, via the shared saturation
        # helper on the collector's own p_full (no re-derived Tetens).
        p_full = np.asarray(dc._p_full(f["p_s"]))
        q_sat = np.asarray(saturation_mixing_ratio(
            np.asarray(f["T"], dtype=np.float64), p_full))
        q_v = 0.97 * q_sat
        q_c = np.zeros((n, NLEV))
        dc.feed_cmip_accumulators_native(
            day=15.0, **f, q_v=q_v, q_c=q_c, q_i=None)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        assert np.all(out["field_2d_clwvi"] > 0.0), (
            "radiative clwvi must include the diagnostic condensate floor")
        assert np.all(out["field_2d_clt"] > 0.0)
        assert np.all(out["field_2d_clt"] <= 100.0)

    def test_clt_bounds_and_units_percent(self, mesh):
        """clt is in CMIP % units: a fully saturated column -> ~100, and
        always within [0, 100]."""
        from legoesm.thermo import saturation_mixing_ratio

        dc, sigma_full = _cloud_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        p_full = np.asarray(dc._p_full(f["p_s"]))
        q_sat = np.asarray(saturation_mixing_ratio(
            np.asarray(f["T"], dtype=np.float64), p_full))
        dc.feed_cmip_accumulators_native(
            day=15.0, **f, q_v=1.0 * q_sat, q_c=np.zeros((n, NLEV)))
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        np.testing.assert_allclose(out["field_2d_clt"], 100.0, atol=1e-6)

    def test_no_cloud_fields_without_q_c_or_cloud_config(self, mesh):
        """Default feed (no q_c) and a cloud-config-less collector both emit
        NO cloud fields — the byte-identical-off contract."""
        from legoesm.thermo import saturation_mixing_ratio

        # (a) cloud config present, q_c absent.
        dc, sigma_full = _cloud_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        dc.feed_cmip_accumulators_native(
            day=15.0, **f, q_v=np.full((f["p_s"].shape[0], NLEV), 1e-3))
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        for k in ("clt", "clwvi", "clivi"):
            assert f"field_2d_{k}" not in out
        # (b) q_c present, collector has no cloud config (cloud_scheme none).
        dc2, _ = _make_collector(mesh)
        n = f["p_s"].shape[0]
        p_full = np.asarray(dc2._p_full(f["p_s"]))
        q_sat = np.asarray(saturation_mixing_ratio(
            np.asarray(f["T"], dtype=np.float64), p_full))
        dc2.feed_cmip_accumulators_native(
            day=15.0, **f, q_v=0.9 * q_sat,
            q_c=np.full((n, NLEV), 1e-4))
        out2 = dc2._spatial_monthly.finalize(min_sample_fraction=0)
        for k in ("clt", "clwvi", "clivi"):
            assert f"field_2d_{k}" not in out2

    def test_bad_q_c_shape_raises_and_commits_nothing(self, mesh):
        dc, sigma_full = _cloud_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        with pytest.raises(ValueError, match="q_c"):
            dc.feed_cmip_accumulators_native(
                day=15.0, **f, q_v=np.full((n, NLEV), 1e-3),
                q_c=np.full(n, 1e-4))     # (n,) not (n, nlev)
        assert dc._spatial_monthly._max_count_ever == 0


class TestClearSkyDriverFeed:
    """The driver glue (`_feed_mpas_cmip_accumulators`): slots 10/11 ->
    rsutcs/rlutcs, cloud tracers -> clt/clwvi/clivi — all gated by
    ``config.output.clear_sky_diag`` (default OFF = byte-identical feed)."""

    def _fake(self, mesh, dc, clear_sky_on, with_cs_slots):
        import types

        _, sigma_full = None, None
        sigma_full = np.linspace(0.05, 0.98, NLEV)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]

        def _field(a):
            return types.SimpleNamespace(data=np.asarray(a))

        sfc = [None] * 12
        sfc[2] = _field(np.full(n, 2.0e-5))       # precip
        sfc[4] = _field(np.full(n, 99.0))         # rsut
        sfc[3] = _field(np.full(n, 238.0))        # rlut
        if with_cs_slots:
            sfc[10] = _field(np.full(n, 77.0))    # rsutcs
            sfc[11] = _field(np.full(n, 262.0))   # rlutcs
        u_edge = np.zeros((int(mesh.nEdges), NLEV))
        q_v = np.full((n, NLEV), 5.0e-3)
        q_c = np.full((n, NLEV), 2.0e-4)
        q_i = np.full((n, NLEV), 1.0e-4)
        return _as_driver(types.SimpleNamespace(
            diagnostics=dc,
            grid=mesh,
            config=types.SimpleNamespace(
                output=types.SimpleNamespace(clear_sky_diag=clear_sky_on)),
            state=types.SimpleNamespace(
                u=_field(u_edge), T=_field(f["T"]), p_s=_field(f["p_s"]),
                phis=_field(f["phis"]),
                tracers={"q_v": _field(q_v), "q_c": _field(q_c),
                         "q_i": _field(q_i)},
            ),
            model=types.SimpleNamespace(_sfc_diag=tuple(sfc)),
        ))

    def test_flag_on_feeds_all_five_new_fields(self, mesh):
        from legoesm.driver.model_driver import ModelDriver

        dc, _ = _cloud_collector(mesh)
        fake = self._fake(mesh, dc, clear_sky_on=True, with_cs_slots=True)
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        np.testing.assert_allclose(out["field_2d_rsutcs"], 77.0, rtol=1e-9)
        np.testing.assert_allclose(out["field_2d_rlutcs"], 262.0, rtol=1e-9)
        for k in ("clt", "clwvi", "clivi"):
            assert f"field_2d_{k}" in out, k
        # Clear-sky inequalities on the uniform synthetic fluxes.
        assert np.all(out["field_2d_rsutcs"] <= out["field_2d_rsut"] + 1e-9)
        assert np.all(out["field_2d_rlutcs"] >= out["field_2d_rlut"] - 1e-9)

    def test_flag_off_feeds_none_of_the_new_fields(self, mesh):
        """Default OFF: slots 10/11 are never produced upstream AND the
        cloud tracers are not forwarded — none of the five new fields may
        appear even though q_c/q_i sit in the state."""
        from legoesm.driver.model_driver import ModelDriver

        dc, _ = _cloud_collector(mesh)
        fake = self._fake(mesh, dc, clear_sky_on=False, with_cs_slots=False)
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        for k in ("rsutcs", "rlutcs", "clt", "clwvi", "clivi"):
            assert f"field_2d_{k}" not in out, k
        # ...while the pre-existing fields still flow (the feed itself ran).
        assert "field_2d_rsut" in out


class TestClearSkyPassEffective:
    """``--clear-sky-diag`` must WORK or REFUSE AUDIBLY, never silently cost
    2x radiation and publish nothing (#843 / the #1385 silent-drop class)."""

    def _f(self, **kw):
        from legoesm.driver.model_driver import clear_sky_pass_effective
        base = dict(clear_sky_diag=True, radiation="rrtmgp",
                    spatial_feed_on=True)
        return clear_sky_pass_effective(**{**base, **kw})

    def test_flag_off_is_silent_and_skips_the_pass(self):
        assert self._f(clear_sky_diag=False) == (False, None)

    def test_publishable_config_runs_the_pass_without_warning(self):
        assert self._f() == (True, None)

    def test_no_radiation_skips_with_a_reason(self):
        run, why = self._f(radiation="none")
        assert run is False
        assert why is not None and "radiation='none'" in why
        # ...and it must NOT claim the cloud trio is lost: that path needs no
        # radiation at all and is still published.
        assert "clt/clwvi/clivi is unaffected" in why

    def test_unreached_diag_boundary_skips_with_a_reason(self):
        """--days 1 --diag-days 5: the feed loop never fires, so the second
        pass would be paid on every radiation step for a file nobody writes."""
        run, why = self._f(feed_steps_reached=False)
        assert run is False
        assert why is not None and "diagnostic boundary" in why

    def test_spatial_feed_off_skips_with_a_reason_naming_the_missing_flag(self):
        """monthly_means alone feeds only the ZONAL accumulator, which holds
        none of the five new fields — so the pass must be skipped there too,
        not merely when all CMOR output is off."""
        run, why = self._f(spatial_feed_on=False)
        assert run is False
        assert why is not None and "--cmip-output" in why
        assert "zonal" in why

    def test_run_mpas_uses_the_helper_for_the_radiation_config(self):
        """The gate must reach the RadiationConfig the MPAS lane BUILDS —
        passing the raw flag there is the defect this helper exists to stop.
        Asserted on the AST of the symbol that runs (_run_mpas)."""
        import ast
        import inspect
        import textwrap

        from legoesm.driver.model_driver import ModelDriver

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(ModelDriver._run_mpas)))
        kw = [k for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name)
              and n.func.id == "RadiationConfig"
              for k in (n.keywords or []) if k.arg == "clear_sky_diag"]
        assert len(kw) == 1, (
            "expected exactly one RadiationConfig(clear_sky_diag=...) in "
            "_run_mpas")
        names = {n.attr for n in ast.walk(kw[0].value)
                 if isinstance(n, ast.Attribute)}
        assert "_mpas_clear_sky_effective" in names, (
            "_run_mpas passes the RAW flag into RadiationConfig; it must pass "
            "the publishability-checked value from clear_sky_pass_effective.")
        assert "clear_sky_diag" not in names


class TestClearSkyRadiationFactory:
    """RadiationConfig.clear_sky_diag on the lean hydrostatic/MPAS factory:
    the clouds-off second pass attaches sw_up_toa_clr/lw_up_toa_clr; off
    (default) leaves them None; unsupported model types refuse loudly."""

    @pytest.fixture(scope="class")
    def mpas_setup(self):
        from legoesm.atmosphere.forcing.idealized.held_suarez import (
            held_suarez_init_mpas,
        )
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.grids.voronoi import create_voronoi_mesh

        mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
        sig = create_sigma_coordinate(NLEV, sigma_top=0.05)
        state = held_suarez_init_mpas(mesh, sig, T_init=285.0,
                                      perturbation_amplitude=0.0)
        return mesh, sig, state

    def _fn(self, clear_sky_diag):
        from legoesm.atmosphere.physics.radiation.config import (
            RadiationConfig,
        )
        from legoesm.atmosphere.physics.radiation.integration import (
            make_radiation_physics,
        )
        cfg = RadiationConfig(scheme="gray", clear_sky_diag=clear_sky_diag)
        return make_radiation_physics(cfg, model_type="mpas")

    def test_flag_off_leaves_clear_sky_fields_none(self, mpas_setup):
        mesh, sig, state = mpas_setup
        t = self._fn(False)(state, mesh, sig)
        assert t.sw_up_toa_clr is None
        assert t.lw_up_toa_clr is None
        assert t.sw_up_toa is not None    # all-sky diagnostics unaffected

    def test_flag_on_attaches_clear_sky_toa(self, mpas_setup):
        """Gray radiation has no cloud interaction, so the clouds-off second
        pass must reproduce the all-sky TOA fluxes exactly — a plumbing
        check that is also the correct clear-sky value for gray."""
        mesh, sig, state = mpas_setup
        t = self._fn(True)(state, mesh, sig)
        assert t.sw_up_toa_clr is not None
        assert t.lw_up_toa_clr is not None
        sw = np.asarray(t.sw_up_toa.data)
        sw_clr = np.asarray(t.sw_up_toa_clr.data)
        lw = np.asarray(t.lw_up_toa.data)
        lw_clr = np.asarray(t.lw_up_toa_clr.data)
        assert np.all(np.isfinite(sw_clr)) and np.all(np.isfinite(lw_clr))
        np.testing.assert_allclose(sw_clr, sw, rtol=1e-12)
        np.testing.assert_allclose(lw_clr, lw, rtol=1e-12)
        # The heating that drives the model comes from the ALL-SKY pass
        # only: with the flag toggled the tendency must be identical.
        t_off = self._fn(False)(state, mesh, sig)
        np.testing.assert_allclose(np.asarray(t.dT_dt.data),
                                   np.asarray(t_off.dT_dt.data), rtol=0,
                                   atol=0)

    def test_unwired_model_type_refuses_loudly(self):
        from legoesm.atmosphere.physics.radiation.config import (
            RadiationConfig,
        )
        from legoesm.atmosphere.physics.radiation.integration import (
            make_radiation_physics,
        )
        cfg = RadiationConfig(scheme="gray", clear_sky_diag=True)
        with pytest.raises(NotImplementedError, match="clear_sky_diag"):
            make_radiation_physics(cfg, model_type="spectral_pe")

    def test_rrtmgp_clear_sky_inequalities(self, mpas_setup):
        """The PHYSICS of rsutcs/rlutcs, on the real RRTMGP cloud optics
        (the gray test above can only check plumbing — gray radiation has no
        cloud interaction, so its clear sky is trivially the all-sky value).
        Removing the clouds must reflect LESS shortwave (rsutcs <= rsut) and
        emit MORE longwave (rlutcs >= rlut), STRICTLY and pointwise, in the
        same +up CMOR sign convention as rsut/rlut.  The all-sky heating —
        the only thing that reaches the state — must be bit-identical with
        the diagnostic on or off.

        NOTE on the state: the shared ``mpas_setup`` fixture is ISOTHERMAL,
        and a positive LW cloud radiative effect needs cloud tops COLDER
        than the surface.  On the isothermal column the two passes differ by
        only ~0.2 W/m2 of spectral/surface-emissivity residual and the sign
        of ``rlutcs - rlut`` is meaningless (measured: -0.23 W/m2).  So this
        test imposes a standard-troposphere lapse rate first — the LW
        inequality is a property of a stratified atmosphere, not an
        identity."""
        from legoesm.atmosphere.physics.clouds.config import build_cloud_config
        from legoesm.atmosphere.physics.radiation.config import (
            RRTMGPConfig,
            RadiationConfig,
        )
        from legoesm.atmosphere.physics.radiation.integration import (
            make_radiation_physics,
        )
        from legoesm.thermo import saturation_mixing_ratio

        mesh, sig, state = mpas_setup
        p_full = sig.pressure_at_full(state.p_s.data)
        # T = T_s (p/p_s)^(R_d*Gamma/g), Gamma = 6.5 K/km (ICAO troposphere),
        # floored at a 200 K stratosphere -> cold cloud tops over a warm
        # surface (T_sfc = T[..., -1] in the radiation physics_fn).
        _expo = constants.R_d * 6.5e-3 / constants.g
        t_lapse_k = np.maximum(
            np.asarray(state.T.data)
            * np.asarray(p_full / state.p_s.data[..., None]) ** _expo,
            200.0)
        state = state._replace(T=state.T.replace(data=t_lapse_k))
        # Near-saturated column so the sundqvist fraction + its in-cloud
        # condensate floor give radiatively ACTIVE cloud (a dry column would
        # make both passes identical and the test vacuous).
        q_sat = saturation_mixing_ratio(t_lapse_k, p_full)
        _z = q_sat * 0.0

        def _f(name, d):
            return Field(data=d, name=name, dims=("nCells", "nlev"),
                         units="kg/kg")

        moist = state._replace(tracers={
            "q_v": _f("q_v", 0.95 * q_sat),
            "q_c": _f("q_c", _z), "q_i": _f("q_i", _z)})

        def _run(clear_sky):
            cfg = RadiationConfig(
                scheme="rrtmgp", cloud_scheme="sundqvist",
                rrtmgp=RRTMGPConfig(include_clouds=True),
                cloud_config=build_cloud_config("sundqvist",
                                                convective_cloud=False),
                clear_sky_diag=clear_sky)
            return make_radiation_physics(cfg, model_type="mpas")(
                moist, mesh, sig)

        t = _run(True)
        sw, sw_c = (np.asarray(t.sw_up_toa.data),
                    np.asarray(t.sw_up_toa_clr.data))
        lw, lw_c = (np.asarray(t.lw_up_toa.data),
                    np.asarray(t.lw_up_toa_clr.data))
        assert np.all(np.isfinite(sw_c)) and np.all(np.isfinite(lw_c))
        # STRICT: a >= that also passes when the second pass silently
        # returned the all-sky fluxes would prove nothing.
        assert np.all(sw_c < sw), f"min SW CRE {np.min(sw - sw_c)}"
        assert np.all(lw_c > lw), f"min LW CRE {np.min(lw_c - lw)}"
        # ...and on the AREA mean (the form the CMOR budget is judged on).
        w = np.asarray(mesh.areaCell)
        w = w / w.sum()
        assert float(sw_c @ w) < float(sw @ w)
        assert float(lw_c @ w) > float(lw @ w)
        # Trajectory neutrality under the real optics.
        np.testing.assert_array_equal(np.asarray(t.dT_dt.data),
                                      np.asarray(_run(False).dT_dt.data))
