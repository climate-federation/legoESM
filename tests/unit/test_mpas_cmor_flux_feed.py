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


# CMIP6 clear-sky QUARTET (sfc_diag slots 10-13) with values chosen to be
# physically ordered against their all-sky partners in FLUXES:
#   rsutcs < rsut (99)   rlutcs > rlut (238)   rsdscs/rldscs are surface.
CLEARSKY = dict(rsutcs=41.0, rlutcs=266.0, rsdscs=250.0, rldscs=290.0)

# FLUXES minus everything CLEARSKY also carries — the "clear-sky diagnostic
# OFF" feed.  FLUXES itself gained rsutcs/rlutcs upstream (#1525), so passing
# it whole is no longer a clear-sky-free feed.
ALLSKY_ONLY = {k: v for k, v in FLUXES.items() if k not in CLEARSKY}


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
            **{k: np.full(n, v)
               for k, v in {**FLUXES, **CLEARSKY}.items()})
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
            **{k: np.full(n, v)
               for k, v in {**FLUXES, **CLEARSKY}.items()})
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
            day=15.0, **f,
            **{k: np.full(n, v) for k, v in ALLSKY_ONLY.items()})
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        for k in CLEARSKY:
            assert f"field_2d_{k}" not in out

    def test_accumulator_covers_the_clear_sky_slots(self):
        """``_MPASSfcFluxAccum`` must average slots 10-13 — a slot missing
        from SLOTS is silently dropped from every interval mean."""
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        for slot in (10, 11, 12, 13):
            assert slot in _MPASSfcFluxAccum.SLOTS, slot
        acc = _MPASSfcFluxAccum()
        for v in (10.0, 20.0, 60.0):
            row = [None] * 14
            row[10] = types.SimpleNamespace(data=np.full(4, v))
            row[13] = types.SimpleNamespace(data=np.full(4, 2.0 * v))
            acc.add(tuple(row))
        np.testing.assert_allclose(acc.mean(10), 30.0)   # (10+20+60)/3
        np.testing.assert_allclose(acc.mean(13), 60.0)
        assert acc.mean(11) is None                      # never fed

    def test_driver_reads_the_clear_sky_slots_by_index(self):
        """The consumer that RUNS (``_feed_mpas_cmip_accumulators``) must map
        slots 10-13 onto the four CMOR names and forward them to the feed."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(
            ModelDriver._feed_mpas_cmip_accumulators)
        for name, slot in (("rsutcs", 10), ("rlutcs", 11),
                           ("rsdscs", 12), ("rldscs", 13)):
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
        sw_down_sfc_clr=_f("sw_down_sfc_clr", 13.0),
        lw_down_sfc_clr=_f("lw_down_sfc_clr", 14.0),
        sw_up_sfc_clr=_f("sw_up_sfc_clr", 15.0),
        precip_solid=_f("precip_solid", 16.0),
        tau_x_sfc=_f("tau_x", 17.0), tau_y_sfc=_f("tau_y", 18.0))


# The ONE slot contract both producers build from (core.state).  Spelled out
# here so a silent edit to the shared constant still has to face the explicit
# slot map below; a drift swaps rlut/rsut/rsdt, misindexes the land-forcing
# slots 8/9, or drops a clear-sky field.
_EXTRA_ORDER = ("lw_up_toa", "sw_up_toa", "sw_down_toa",
                "shflx_sfc", "lhflx_sfc",
                "sw_down_sfc", "lw_down_sfc",
                "sw_up_toa_clr", "lw_up_toa_clr",
                "sw_down_sfc_clr", "lw_down_sfc_clr", "sw_up_sfc_clr",
                "precip_solid",
                "tau_x_sfc", "tau_y_sfc")


class TestSfcDiagContract:
    """Lock the 18-slot sfc_diag tuple contract of BOTH producers (serial
    ``primitive_eq_mpas._step_jit`` and MPI ``parallel.voronoi_mpi._step``)
    and the driver consumer's slot mapping (0/1 sfc net, 2 precip, 3-7
    all-sky fluxes, 8/9 land-forcing downwelling, 10-14 clear-sky, 15 prsn,
    16/17 surface stress)."""

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
        # ...and it must NOT drop any clear-sky field (the blocker above).
        for _k in ("sw_up_toa_clr", "lw_up_toa_clr", "sw_down_sfc_clr",
                   "lw_down_sfc_clr", "sw_up_sfc_clr"):
            assert _k not in MPAS_SFC_DIAG_MPI_UNPUBLISHED
        # #1321: the surface downwelling pair is PUBLISHED on the MPI lane —
        # withholding it made _marshal_land_forcing return None every step, so
        # the interactive multilayer land silently never advanced.
        assert "sw_down_sfc" not in MPAS_SFC_DIAG_MPI_UNPUBLISHED
        assert "lw_down_sfc" not in MPAS_SFC_DIAG_MPI_UNPUBLISHED

    def test_mpi_producer_publishes_clear_sky_at_contract_slots(self):
        """Rebuild the MPI producer's extras expression on a tendency that
        carries every field: it must be 18 slots long with the clear-sky TOA
        pair at 10/11 (an 8-slot tuple is the defect this locks out)."""
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
        assert len(sfc) == 18
        assert sfc[10].name == "sw_up_toa_clr"   # rsutcs
        assert sfc[11].name == "lw_up_toa_clr"   # rlutcs
        # #1321: the land downwelling pair is now PUBLISHED at slots 8/9.
        # While it was withheld, ``_marshal_land_forcing``'s ``_sd[8] is None``
        # guard declined every step and the Richards soil never advanced.
        assert sfc[8].name == "sw_down_sfc"
        assert sfc[9].name == "lw_down_sfc"

    def test_producer_slot_order_matches_consumer(self):
        _pt = _tend_with_extras()
        _extras = tuple(getattr(_pt, _k, None) for _k in _EXTRA_ORDER)
        sfc_diag = (_pt.sw_net_sfc, _pt.lw_net_sfc, _pt.precip) + _extras
        assert len(sfc_diag) == 18
        # Consumer (_feed_mpas_cmip_accumulators): slot 0->sw_net (rsus
        # derivation), 1->lw_net (rlus), 3->rlut, 4->rsut, 5->rsdt,
        # 6->hfss, 7->hfls, 8->rsds, 9->rlds (+ _marshal_land_forcing).
        assert sfc_diag[0].name == "sw_net"
        assert sfc_diag[1].name == "lw_net"
        assert sfc_diag[3].name == "lw_up_toa"      # rlut
        assert sfc_diag[4].name == "sw_up_toa"      # rsut
        assert sfc_diag[5].name == "sw_down_toa"    # rsdt
        assert sfc_diag[6].name == "shflx"          # hfss
        assert sfc_diag[7].name == "lhflx"          # hfls
        assert sfc_diag[8].name == "sw_down_sfc"    # rsds
        assert sfc_diag[9].name == "lw_down_sfc"    # rlds
        # Clear-sky quartet: 10/11 TOA outgoing (+up, pair with 3/4),
        # 12/13 surface downwelling (+down, pair with 8/9).  NO sign flip
        # on any of the four.
        assert sfc_diag[10].name == "sw_up_toa_clr"    # rsutcs
        assert sfc_diag[11].name == "lw_up_toa_clr"    # rlutcs
        assert sfc_diag[12].name == "sw_down_sfc_clr"  # rsdscs
        assert sfc_diag[13].name == "lw_down_sfc_clr"  # rldscs
        # Slot 14 is surface UPWELLING (+up) — the opposite orientation to
        # its slot-12 partner rsdscs (+down).  Still no sign flip.
        assert sfc_diag[14].name == "sw_up_sfc_clr"     # rsuscs
        # Slot 15 is the frozen SUBSET of slot 2's total precip, same sense.
        assert sfc_diag[15].name == "precip_solid"      # prsn
        # Slots 16/17 carry the MODEL-convention surface stress; the consumer
        # flips the sign to get CMOR tauu/tauv.
        assert sfc_diag[16].name == "tau_x"         # tauu = -slot16
        assert sfc_diag[17].name == "tau_y"         # tauv = -slot17

    def test_both_producers_build_extras_from_the_shared_contract(self):
        """Assert against the symbols that RUN — the serial
        ``MPASPrimitiveEquationModel._step_jit`` and the MPI
        ``make_voronoi_mpi_step`` — that neither re-introduces a hand-written
        key list.  Checked on the AST of the ``_extras`` assignment, NOT on
        the source text: a text match is satisfied by a passing mention in a
        COMMENT (verified — a text-matching version of this test passed
        against a deliberately regressed 8-slot MPI producer).

        SUPERSEDES the pre-merge ``test_both_producers_extract_the_same_
        extra_order``, which searched the producers for QUOTED key literals:
        that check is now unsatisfiable BY DESIGN (both producers iterate the
        shared constant and contain no key literals at all), so keeping it
        would have failed for the very reason the contract is now safe."""
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
        # The producer supplies the TIME clause; the writer keeps the
        # table's AREA clause (Amon tas/pr are "area: time: mean"), so the
        # honesty override must not silently drop "area:".
        with xr.open_dataset(tas_files[0]) as ds:
            assert (ds["tas"].attrs["cell_methods"]
                    == "area: mean time: point within days "
                       "time: mean over days")
            assert "aliased" in ds["tas"].attrs["comment"]
        with xr.open_dataset(pr_files[0]) as ds:
            assert ds["pr"].attrs["cell_methods"] == "area: time: mean"
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


# ===========================================================================
# CMIP6 Amon gap closure: rtmt + the near-surface set (huss/uas/vas/sfcWind)
# ===========================================================================

class TestRtmt:
    """``rtmt`` = net DOWNWARD radiative flux at the top of the model.

    The CMIP6 table declares ``positive="down"``, so with rsdt positive
    down and rsut/rlut positive up the only correct combination is
    ``rsdt - rsut - rlut``.  A sign slip here is invisible in a
    presence-only check: with the FLUXES values it would still produce a
    plausible-looking O(1-100) W/m^2 number.
    """

    def test_rtmt_is_rsdt_minus_rsut_minus_rlut(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            day=15.0, **f,
            **{k: np.full(n, v) for k, v in FLUXES.items()})
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        assert "field_2d_rtmt" in out
        expect = FLUXES["rsdt"] - FLUXES["rsut"] - FLUXES["rlut"]
        assert expect == pytest.approx(3.0)          # 340 - 99 - 238
        np.testing.assert_allclose(
            out["field_2d_rtmt"], expect, rtol=1e-9)

    def test_positive_attribute_is_down(self):
        """The table's ``positive`` is the contract the sign walk must
        match — assert it rather than trusting the comment."""
        from legoesm.io.cmor_output import lookup_cmor_entry
        table, entry = lookup_cmor_entry("rtmt")
        assert table == "Amon"
        assert entry["positive"] == "down"
        assert entry["units"] == "W m-2"

    def test_sign_responds_to_each_term(self, mesh):
        """Perturb ONE term at a time and check rtmt moves by exactly the
        signed amount — catches a swapped rsut/rlut or a dropped minus."""
        base = FLUXES["rsdt"] - FLUXES["rsut"] - FLUXES["rlut"]
        for term, sign in (("rsdt", +1.0), ("rsut", -1.0), ("rlut", -1.0)):
            dc, sigma_full = _make_collector(mesh)
            f = _base_fields(mesh, sigma_full)
            n = f["p_s"].shape[0]
            fluxes = dict(FLUXES)
            fluxes[term] = fluxes[term] + 10.0
            dc.feed_cmip_accumulators_native(
                day=15.0, **f,
                **{k: np.full(n, v) for k, v in fluxes.items()})
            out = dc._spatial_monthly.finalize(min_sample_fraction=0)
            got = float(np.mean(out["field_2d_rtmt"]))
            assert got == pytest.approx(base + sign * 10.0, rel=1e-9), term

    @pytest.mark.parametrize("missing", ["rsdt", "rsut", "rlut"])
    def test_absent_term_skips_rtmt_never_zeroes_it(self, mesh, missing):
        """A zero rtmt would read as exact radiative equilibrium."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        fluxes = {k: np.full(n, v) for k, v in FLUXES.items()
                  if k != missing}
        dc.feed_cmip_accumulators_native(day=15.0, **f, **fluxes)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        assert "field_2d_rtmt" not in out

    def test_rtmt_is_an_interval_mean(self):
        """It is a linear combination of the interval-mean rsdt/rsut/rlut,
        so it must ride the _FLUX_2D midpoint bin with them."""
        import inspect
        src = inspect.getsource(
            DiagnosticCollector.feed_cmip_accumulators_native)
        flux_block = src.split("_FLUX_2D = ")[1].split("_FLUX_DAILY")[0]
        assert '"rtmt"' in flux_block


# Near-surface probe values: a wind that REVERSES between the two samples,
# so the monthly mean of uas is 0 while the mean SPEED is 12 m/s.  That is
# the discriminating case for sfcWind.
NEAR_SFC_Q = 0.012          # kg/kg at the lowest model level


class TestNearSurfaceSet:
    def _feed(self, dc, mesh, sigma_full, day, u, v, q=NEAR_SFC_Q):
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        nlev = f["T"].shape[1]
        # Lowest model level is index -1; put a DIFFERENT value aloft so a
        # wrong level index is caught rather than silently passing.
        def _col(surface_value, aloft):
            col = np.full((n, nlev), aloft, dtype=np.float64)
            col[:, -1] = surface_value
            return col
        return dc.feed_cmip_accumulators_native(
            day=day, **f,
            q_v=_col(q, 0.5 * q),
            u_east=_col(u, 3.0 * u if u else 7.0),
            v_north=_col(v, 3.0 * v if v else 7.0))

    def test_fields_reach_monthly_accumulator_from_lowest_level(
            self, mesh):
        dc, sigma_full = _make_collector(mesh)
        self._feed(dc, mesh, sigma_full, 15.0, u=12.0, v=0.0)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        for name in ("huss", "uas", "vas", "sfcWind"):
            assert f"field_2d_{name}" in out, name
        # Uniform field -> IDW (partition of unity) regrid is EXACT.
        np.testing.assert_allclose(
            out["field_2d_huss"], NEAR_SFC_Q, rtol=1e-9)
        np.testing.assert_allclose(out["field_2d_uas"], 12.0, rtol=1e-9)
        np.testing.assert_allclose(out["field_2d_vas"], 0.0, atol=1e-12)
        np.testing.assert_allclose(out["field_2d_sfcWind"], 12.0, rtol=1e-9)

    def test_sfcwind_is_the_mean_speed_not_the_speed_of_the_mean(
            self, mesh):
        """THE classic error on this variable.  Two samples with opposite
        zonal wind: mean(uas) = 0 but mean(|V|) = 12 m/s.  A writer that
        formed sqrt(mean(uas)^2 + mean(vas)^2) would publish 0."""
        dc, sigma_full = _make_collector(mesh)
        self._feed(dc, mesh, sigma_full, 10.0, u=+12.0, v=0.0)
        self._feed(dc, mesh, sigma_full, 20.0, u=-12.0, v=0.0)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        uas = float(np.mean(out["field_2d_uas"]))
        wind = float(np.mean(out["field_2d_sfcWind"]))
        assert uas == pytest.approx(0.0, abs=1e-9)
        assert wind == pytest.approx(12.0, rel=1e-9), (
            "sfcWind collapsed toward the speed of the MEAN wind")

    def test_speed_uses_both_components(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        self._feed(dc, mesh, sigma_full, 15.0, u=3.0, v=4.0)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        np.testing.assert_allclose(
            out["field_2d_sfcWind"], 5.0, rtol=1e-9)

    def test_absent_inputs_are_skipped_never_zeroed(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        dc.feed_cmip_accumulators_native(day=15.0, **f)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        for name in ("huss", "uas", "vas", "sfcWind"):
            assert f"field_2d_{name}" not in out, name

    def test_variables_are_in_the_amon_table_with_reference_heights(self):
        from legoesm.io.cmor_output import (
            _VAR_REFERENCE_HEIGHT, lookup_cmor_entry,
        )
        expect = {"huss": ("1", 2.0), "uas": ("m s-1", 10.0),
                  "vas": ("m s-1", 10.0), "sfcWind": ("m s-1", 10.0)}
        for name, (units, height) in expect.items():
            table, entry = lookup_cmor_entry(name)
            assert table == "Amon", name
            assert entry["units"] == units, name
            assert _VAR_REFERENCE_HEIGHT[name] == height, name

    def test_driver_marks_them_as_snapshots(self):
        """They are instantaneous end-of-interval samples like hus/ua/va,
        so the lean MPAS driver must disclose that on the written file."""
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._run_mpas)
        block = src.split("cmip_snapshot_vars = {")[1].split("}")[0]
        for name in ("huss", "uas", "vas", "sfcWind"):
            assert f'"{name}"' in block, name


class TestNewAmonVarsReachTheWriter:
    """End-to-end through the REAL writer: feed -> accumulator -> NetCDF.
    Presence alone is not enough — units and magnitudes are asserted."""

    def test_files_carry_table_units_and_plausible_values(
            self, mesh, tmp_path):
        xr = pytest.importorskip("xarray")
        from legoesm.io.cmor_output import CFWriter

        dc, sigma_full = _make_collector(mesh)
        dc.cf_writer = CFWriter(
            output_dir=tmp_path, experiment_id="amip",
            model_id="legoESM", ref_date="1979-01-01")
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        nlev = f["T"].shape[1]
        q = np.full((n, nlev), 0.004)
        q[:, -1] = 0.012
        u = np.full((n, nlev), 20.0)
        u[:, -1] = 6.0
        v = np.full((n, nlev), 0.0)
        v[:, -1] = 8.0
        dc.feed_cmip_accumulators_native(
            day=15.0, **f, q_v=q, u_east=u, v_north=v,
            **{k: np.full(n, val) for k, val in FLUXES.items()})
        dc._write_cmip_data(
            dc._spatial_monthly.finalize(min_sample_fraction=0))

        expect = {
            "rtmt": ("W m-2", 3.0),                    # 340 - 99 - 238
            "huss": ("1", 0.012),
            "uas": ("m s-1", 6.0),
            "vas": ("m s-1", 8.0),
            "sfcWind": ("m s-1", 10.0),                # hypot(6, 8)
        }
        for name, (units, value) in expect.items():
            paths = sorted(tmp_path.rglob(f"{name}_Amon_*.nc"))
            assert paths, f"{name} was never written by the CMOR writer"
            with xr.open_dataset(paths[0]) as ds:
                assert ds[name].attrs["units"] == units, name
                got = float(np.nanmean(ds[name].values))
                assert got == pytest.approx(value, rel=1e-5), name
        # ``positive`` is a real physical claim on rtmt — check the file.
        with xr.open_dataset(
                sorted(tmp_path.rglob("rtmt_Amon_*.nc"))[0]) as ds:
            assert ds["rtmt"].attrs["positive"] == "down"


# ===========================================================================
# Amon tasmin / tasmax — monthly MEAN of the WITHIN-DAY extrema
# ===========================================================================

class TestAmonDailyExtremes:
    """CMIP6 ``Amon`` declares

        tasmax  "area: mean time: maximum within days time: mean over days"

    i.e. the mean over days of each day's maximum.  A monthly MAXIMUM (the
    easy mistake) is a different, much larger statistic.
    """

    def _acc(self):
        from legoesm.diagnostics.monthly_means import (
            SpatialMonthlyAccumulator,
        )
        return SpatialMonthlyAccumulator(
            nlat=2, nlon=3,
            daily_extreme_fields={"tas": ("tasmin", "tasmax")})

    @staticmethod
    def _f(value, nlat=2, nlon=3):
        return {"tas": np.full((nlat, nlon), float(value))}

    def test_table_cell_methods_is_mean_over_days_of_daily_extrema(self):
        from legoesm.io.cmor_output import lookup_cmor_entry
        for name, word in (("tasmax", "maximum"), ("tasmin", "minimum")):
            table, entry = lookup_cmor_entry(name)
            assert table == "Amon", name
            assert entry["cell_methods"] == (
                f"area: mean time: {word} within days time: mean over days")

    def test_monthly_mean_of_daily_extrema_not_monthly_extremum(self):
        """Day 1 spans 280..300 (max 300), day 2 spans 270..290 (max 290).
        Mean of daily maxima = 295.  The monthly MAXIMUM would be 300 and
        the monthly MEAN of all samples 285 — both wrong."""
        acc = self._acc()
        for v in (280.0, 300.0, 290.0):
            acc.add_2d(1.0, 0, self._f(v))
        for v in (270.0, 290.0, 275.0):
            acc.add_2d(2.0, 0, self._f(v))
        out = acc.finalize(min_sample_fraction=0)
        np.testing.assert_allclose(out["field_2d_tasmax"][0], 295.0)
        np.testing.assert_allclose(out["field_2d_tasmin"][0], 275.0)
        # Sanity: the plain monthly mean of tas is a different number.
        np.testing.assert_allclose(
            out["field_2d_tas"][0], np.mean(
                [280.0, 300.0, 290.0, 270.0, 290.0, 275.0]))

    def test_days_are_weighted_equally_regardless_of_sample_count(self):
        """"mean OVER DAYS" — a day sampled 4x must not outweigh a day
        sampled once."""
        acc = self._acc()
        for _ in range(4):
            acc.add_2d(1.0, 0, self._f(300.0))
        acc.add_2d(2.0, 0, self._f(280.0))
        out = acc.finalize(min_sample_fraction=0)
        np.testing.assert_allclose(out["field_2d_tasmax"][0], 290.0)

    def test_last_partial_day_is_committed_at_finalize(self):
        acc = self._acc()
        acc.add_2d(1.0, 0, self._f(300.0))
        assert "tas" in acc._day_ext            # still pending
        out = acc.finalize(min_sample_fraction=0)
        np.testing.assert_allclose(out["field_2d_tasmax"][0], 300.0)
        assert not acc._day_ext                 # flushed

    def test_extrema_land_in_their_own_month_across_a_boundary(self):
        acc = self._acc()
        acc.add_2d(31.0, 0, self._f(300.0))     # 31 Jan
        acc.add_2d(32.0, 0, self._f(250.0))     # 1 Feb
        out = acc.finalize(min_sample_fraction=0)
        assert out["months"] == [(0, 1), (0, 2)]
        np.testing.assert_allclose(out["field_2d_tasmax"][0], 300.0)
        np.testing.assert_allclose(out["field_2d_tasmax"][1], 250.0)

    def test_pop_completed_months_does_not_lose_the_boundary_day(self):
        """The pending 31-Jan day belongs to a month about to be popped —
        it must be committed, not freed with the bucket."""
        acc = self._acc()
        acc.add_2d(31.0, 0, self._f(300.0))
        popped = acc.pop_completed_months(0, 2)
        assert popped["months"] == [(0, 1)]
        np.testing.assert_allclose(popped["field_2d_tasmax"][0], 300.0)

    def test_pop_leaves_the_in_progress_day_open(self):
        acc = self._acc()
        acc.add_2d(31.0, 0, self._f(300.0))     # January, completed below
        acc.add_2d(32.0, 0, self._f(250.0))     # February, in progress
        acc.pop_completed_months(0, 2)
        assert "tas" in acc._day_ext            # Feb 1 still open
        acc.add_2d(32.0, 0, self._f(260.0))     # same day, warmer
        out = acc.finalize(min_sample_fraction=0)
        np.testing.assert_allclose(out["field_2d_tasmax"][0], 260.0)

    def test_call_counts_untouched_by_the_extreme_commits(self):
        """The daily commits must not inflate the per-bucket call count —
        that count drives the partial-month guard for EVERY variable."""
        from legoesm.diagnostics.monthly_means import (
            SpatialMonthlyAccumulator,
        )
        acc = self._acc()
        ref = SpatialMonthlyAccumulator(nlat=2, nlon=3)
        for day, v in ((1.0, 300.0), (2.0, 280.0), (3.0, 290.0)):
            acc.add_2d(day, 0, self._f(v))
            ref.add_2d(day, 0, self._f(v))
        assert acc._call_counts == ref._call_counts
        assert acc._max_count_ever == ref._max_count_ever

    def test_disabled_by_default_is_byte_identical(self):
        from legoesm.diagnostics.monthly_means import (
            SpatialMonthlyAccumulator,
        )
        acc = SpatialMonthlyAccumulator(nlat=2, nlon=3)
        acc.add_2d(1.0, 0, self._f(300.0))
        out = acc.finalize(min_sample_fraction=0)
        assert "field_2d_tasmax" not in out
        assert "field_2d_tasmin" not in out

    def test_state_roundtrip_carries_the_pending_day(self):
        acc = self._acc()
        acc.add_2d(1.0, 0, self._f(300.0))
        acc.add_2d(1.0, 0, self._f(270.0))      # same day, still pending
        b = self._acc()
        b.set_state(acc.get_state())
        # The restored accumulator continues the SAME day.
        b.add_2d(1.0, 0, self._f(310.0))
        out = b.finalize(min_sample_fraction=0)
        np.testing.assert_allclose(out["field_2d_tasmax"][0], 310.0)
        np.testing.assert_allclose(out["field_2d_tasmin"][0], 270.0)

    def test_pre_feature_checkpoint_restores_without_a_pending_day(self):
        """``day_ext`` is optional on read, so schema version 1 checkpoints
        written before this feature still load."""
        import json
        acc = self._acc()
        acc.add_2d(1.0, 0, self._f(300.0))
        state = dict(acc.get_state())
        manifest = json.loads(str(state["__manifest__"].item()))
        del manifest["day_ext"]
        state["__manifest__"] = np.asarray(json.dumps(manifest))
        b = self._acc()
        b.set_state(state)                       # must not raise
        assert b._day_ext == {}

    def test_collector_configures_tas_extremes(self, mesh):
        dc, _ = _make_collector(mesh)
        assert dc._spatial_monthly.daily_extreme_fields == {
            "tas": ("tasmin", "tasmax")}

    def test_reaches_the_real_writer_with_amon_units(self, mesh, tmp_path):
        xr = pytest.importorskip("xarray")
        from legoesm.io.cmor_output import CFWriter

        dc, sigma_full = _make_collector(mesh)
        dc.cf_writer = CFWriter(
            output_dir=tmp_path, experiment_id="amip",
            model_id="legoESM", ref_date="1979-01-01")
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        # Two samples on day 10 (288 / 300 K) and one on day 11 (280 K):
        # daily maxima 300 and 280 -> Amon tasmax = 290 K.
        for day, tas in ((10.0, 288.0), (10.0, 300.0), (11.0, 280.0)):
            dc.feed_cmip_accumulators_native(
                day=day, **f, tas=np.full(n, tas))
        dc._write_cmip_data(
            dc._spatial_monthly.finalize(min_sample_fraction=0))

        for name, value in (("tasmax", 290.0), ("tasmin", 284.0)):
            paths = sorted(tmp_path.rglob(f"{name}_Amon_*.nc"))
            assert paths, f"{name} was never written"
            with xr.open_dataset(paths[0]) as ds:
                assert ds[name].attrs["units"] == "K"
                got = float(np.nanmean(ds[name].values))
                assert got == pytest.approx(value, rel=1e-6), name


# ===========================================================================
# Amon rsuscs — clear-sky SURFACE UPWELLING shortwave (sfc_diag slot 14)
# ===========================================================================

class TestRsuscs:
    """The clear-sky quartet's missing fifth member.

    Sign convention, walked once here and asserted below:
      rsdscs [W/m^2, +DOWN]  surface downwelling, clear sky   (slot 14)
      rsuscs [W/m^2, +UP]    surface upwelling,   clear sky   (slot 14)
    They are a DOWN/UP pair, not two same-signed fluxes, and the table
    declares ``positive="up"`` on rsuscs alone.
    """

    VALUE = 37.5        # ~0.15 albedo against the CLEARSKY rsdscs of 250

    def test_reaches_the_monthly_accumulator_unflipped(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            day=15.0, **f,
            **{k: np.full(n, v)
               for k, v in {**FLUXES, **CLEARSKY}.items()},
            rsuscs=np.full(n, self.VALUE))
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        assert "field_2d_rsuscs" in out
        np.testing.assert_allclose(
            out["field_2d_rsuscs"], self.VALUE, rtol=1e-9)
        # Physical ordering the pair must satisfy: 0 <= rsuscs <= rsdscs
        # (surface albedo in [0, 1]).  A sign flip breaks the left bound.
        got = float(np.mean(out["field_2d_rsuscs"]))
        assert 0.0 <= got <= float(np.mean(out["field_2d_rsdscs"]))

    def test_table_declares_positive_up(self):
        from legoesm.io.cmor_output import lookup_cmor_entry
        table, entry = lookup_cmor_entry("rsuscs")
        assert table == "Amon"
        assert entry["positive"] == "up"
        assert entry["units"] == "W m-2"
        assert entry["standard_name"] == (
            "surface_upwelling_shortwave_flux_in_air_assuming_clear_sky")

    def test_is_an_interval_mean(self):
        import inspect
        src = inspect.getsource(
            DiagnosticCollector.feed_cmip_accumulators_native)
        flux_block = src.split("_FLUX_2D = ")[1].split("_FLUX_DAILY")[0]
        assert '"rsuscs"' in flux_block

    def test_malformed_input_raises_and_commits_nothing(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        with pytest.raises(ValueError, match="rsuscs"):
            dc.feed_cmip_accumulators_native(
                day=15.0, **f, rsuscs=np.zeros(n - 1))
        assert dc._spatial_monthly._max_count_ever == 0

    def test_absent_is_skipped_never_zeroed(self, mesh):
        """A zero rsuscs would read as a perfectly black surface."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            day=15.0, **f,
            **{k: np.full(n, v) for k, v in FLUXES.items()})
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        assert "field_2d_rsuscs" not in out

    def test_accumulator_covers_slot_14(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        assert 14 in _MPASSfcFluxAccum.SLOTS
        acc = _MPASSfcFluxAccum()
        for v in (10.0, 20.0, 60.0):
            row = [None] * 15
            row[14] = types.SimpleNamespace(data=np.full(4, v))
            acc.add(tuple(row))
        np.testing.assert_allclose(acc.mean(14), 30.0)

    def test_driver_reads_slot_14_and_forwards_it(self):
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._feed_mpas_cmip_accumulators)
        assert "rsuscs = _sfc_slot(14)" in src
        assert "rsuscs=rsuscs," in src

    def test_radiation_packer_carries_it_with_none_default(self, mesh):
        """The packer must default it to None (byte-identical when the
        clear-sky diagnostic is off) and pass a supplied array through
        unchanged."""
        from legoesm.atmosphere.physics.radiation.integration import (
            _pack_hydrostatic_tendencies,
        )
        from legoesm.core.field import Field as _F
        from legoesm.core.state import MPASHydrostaticState

        n, nlev = int(mesh.nCells), NLEV
        state = MPASHydrostaticState(
            u=_F(data=np.zeros((int(mesh.nEdges), nlev)), name="u",
                 dims=("edge", "lev"), units="m/s"),
            T=_F(data=np.full((n, nlev), 250.0), name="T",
                 dims=("cell", "lev"), units="K"),
            p_s=_F(data=np.full(n, 1.0e5), name="p_s",
                   dims=("cell",), units="Pa"),
            phis=_F(data=np.zeros(n), name="phis",
                    dims=("cell",), units="m2/s2"))
        dT = np.zeros((n, nlev))
        bare = _pack_hydrostatic_tendencies(dT, state, (n, nlev), (n,))
        assert bare.sw_up_sfc_clr is None
        packed = _pack_hydrostatic_tendencies(
            dT, state, (n, nlev), (n,),
            sw_up_sfc_clr=np.full(n, 42.0))
        np.testing.assert_allclose(
            np.asarray(packed.sw_up_sfc_clr.data), 42.0)

    def test_producer_reads_the_surface_half_level_not_toa(self):
        """rsuscs is ``sw_flux_up[:, -1]`` (SURFACE) — reading ``[:, 0]``
        would silently publish rsutcs a second time under a surface name.

        ``_make_hydrostatic_radiation`` is the factory that BUILDS the
        physics_fn running on the MPAS lane, so its source is where the
        clear-sky block actually lives; asserting against
        ``make_radiation_physics`` (a dispatcher that only selects it)
        would pass while proving nothing."""
        import inspect
        from legoesm.atmosphere.physics.radiation import integration
        src = inspect.getsource(integration._make_hydrostatic_radiation)
        assert "_sw_up_sfc_clr = _rad_out_clr.sw_flux_up[:, -1]" in src
        assert "sw_up_sfc_clr=_sw_up_sfc_clr" in src

    def test_end_to_end_through_the_real_writer(self, mesh, tmp_path):
        xr = pytest.importorskip("xarray")
        from legoesm.io.cmor_output import CFWriter

        dc, sigma_full = _make_collector(mesh)
        dc.cf_writer = CFWriter(
            output_dir=tmp_path, experiment_id="amip",
            model_id="legoESM", ref_date="1979-01-01")
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            day=15.0, **f,
            **{k: np.full(n, v)
               for k, v in {**FLUXES, **CLEARSKY}.items()},
            rsuscs=np.full(n, self.VALUE))
        dc._write_cmip_data(
            dc._spatial_monthly.finalize(min_sample_fraction=0))

        paths = sorted(tmp_path.rglob("rsuscs_Amon_*.nc"))
        assert paths, "rsuscs was never written by the CMOR writer"
        with xr.open_dataset(paths[0]) as ds:
            assert ds["rsuscs"].attrs["units"] == "W m-2"
            assert ds["rsuscs"].attrs["positive"] == "up"
            got = float(np.nanmean(ds["rsuscs"].values))
            assert got == pytest.approx(self.VALUE, rel=1e-5)


# ===========================================================================
# Amon prsn — snowfall flux (sfc_diag slot 15)
# ===========================================================================

class TestPrsn:
    """``prsn`` is the SOLID-phase part of the surface sedimentation flux —
    a SUBSET of ``pr``, in the SAME positive-into-the-surface sense, and in
    the same kg/m2/s units.  Both are sums of the SAME per-species
    dt-limited surface fluxes, so 0 <= prsn <= pr holds by construction.
    """

    PR = 9.0e-5         # kg/m2/s  ~ 7.8 mm/day
    PRSN = 2.0e-5       # kg/m2/s  ~ 1.7 mm/day of snow water equivalent

    def test_reaches_the_monthly_accumulator_as_a_subset_of_pr(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            day=15.0, **f,
            precip=np.full(n, self.PR), prsn=np.full(n, self.PRSN))
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        assert "field_2d_prsn" in out
        np.testing.assert_allclose(
            out["field_2d_prsn"], self.PRSN, rtol=1e-9)
        got_pr = float(np.mean(out["field_2d_pr"]))
        got_sn = float(np.mean(out["field_2d_prsn"]))
        assert 0.0 <= got_sn <= got_pr

    def test_table_entry(self):
        from legoesm.io.cmor_output import lookup_cmor_entry
        table, entry = lookup_cmor_entry("prsn")
        assert table == "Amon"
        assert entry["units"] == "kg m-2 s-1"
        assert entry["standard_name"] == "snowfall_flux"
        # A flux with no declared sign convention — like pr.
        assert entry.get("positive", "") == ""

    def test_is_an_interval_mean_like_pr(self):
        import inspect
        src = inspect.getsource(
            DiagnosticCollector.feed_cmip_accumulators_native)
        flux_block = src.split("_FLUX_2D = ")[1].split("_FLUX_DAILY")[0]
        assert '"prsn"' in flux_block

    def test_malformed_input_raises_and_commits_nothing(self, mesh):
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        with pytest.raises(ValueError, match="prsn"):
            dc.feed_cmip_accumulators_native(
                day=15.0, **f, prsn=np.zeros(n - 1))
        assert dc._spatial_monthly._max_count_ever == 0

    def test_absent_is_skipped_never_zeroed(self, mesh):
        """A zero prsn is the CLAIM "it never snows", not an absence."""
        dc, sigma_full = _make_collector(mesh)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            day=15.0, **f, precip=np.full(n, self.PR))
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        assert "field_2d_pr" in out
        assert "field_2d_prsn" not in out

    def test_accumulator_covers_slot_15(self):
        from legoesm.driver.model_driver import _MPASSfcFluxAccum
        assert 15 in _MPASSfcFluxAccum.SLOTS
        acc = _MPASSfcFluxAccum()
        for v in (1.0e-5, 2.0e-5, 6.0e-5):
            row = [None] * 16
            row[15] = types.SimpleNamespace(data=np.full(4, v))
            acc.add(tuple(row))
        np.testing.assert_allclose(acc.mean(15), 3.0e-5)

    def test_driver_reads_slot_15_and_forwards_it(self):
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._feed_mpas_cmip_accumulators)
        assert "prsn = _sfc_slot(15)" in src
        assert "prsn=prsn," in src

    def test_morrison_reports_the_frozen_subset(self):
        """The production scheme.  ``precipitation_solid`` must be exactly
        ice+snow+graupel — using the total (or including rain) would
        publish pr twice under two names."""
        import inspect
        from legoesm.atmosphere.physics.microphysics import morrison
        src = inspect.getsource(morrison)
        assert ("precipitation_solid = precip_i + precip_s + precip_g"
                in src)
        assert "precipitation = precip_r + precip_i + precip_s + precip_g" \
            in src

    def test_microphysics_output_default_is_none(self):
        """Schemes that do not resolve the split must leave it unset."""
        from legoesm.atmosphere.physics.microphysics.output import (
            MicrophysicsOutput, make_zero_output,
        )
        assert "precipitation_solid" in MicrophysicsOutput._fields
        assert (MicrophysicsOutput._field_defaults["precipitation_solid"]
                is None)
        assert make_zero_output(2, 3).precipitation_solid is None

    def test_tendency_carrier_default_is_none(self):
        from legoesm.core.state import HydrostaticTendencies
        assert "precip_solid" in HydrostaticTendencies._fields
        assert (HydrostaticTendencies._field_defaults["precip_solid"]
                is None)

    def test_combined_physics_sums_it_alongside_precip(self):
        """The pair must come from the SAME module set — a module counted
        for pr but not prsn would break the subset relation."""
        import inspect
        from legoesm.atmosphere.physics import combined
        src = inspect.getsource(combined._make_hydrostatic_combined)
        assert 'if getattr(t, "precip_solid", None) is not None:' in src
        assert "precip_solid_accum + t.precip_solid.data" in src

    def test_end_to_end_through_the_real_writer(self, mesh, tmp_path):
        xr = pytest.importorskip("xarray")
        from legoesm.io.cmor_output import CFWriter

        dc, sigma_full = _make_collector(mesh)
        dc.cf_writer = CFWriter(
            output_dir=tmp_path, experiment_id="amip",
            model_id="legoESM", ref_date="1979-01-01")
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(
            day=15.0, **f,
            precip=np.full(n, self.PR), prsn=np.full(n, self.PRSN))
        dc._write_cmip_data(
            dc._spatial_monthly.finalize(min_sample_fraction=0))

        paths = sorted(tmp_path.rglob("prsn_Amon_*.nc"))
        assert paths, "prsn was never written by the CMOR writer"
        with xr.open_dataset(paths[0]) as ds:
            assert ds["prsn"].attrs["units"] == "kg m-2 s-1"
            got = float(np.nanmean(ds["prsn"].values))
            assert got == pytest.approx(self.PRSN, rel=1e-5)
            # Plausibility: a snowfall rate in mm/day, not a mislabelled
            # accumulated depth or a per-hour rate.
            assert 0.0 < got * 86400.0 < 100.0


# ---------------------------------------------------------------------------
# Ported from origin/main's #1525 (#843) suite at the 2026-08-10 merge.
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
        return types.SimpleNamespace(
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
        )

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
