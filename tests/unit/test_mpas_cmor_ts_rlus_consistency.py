"""CMOR ``ts`` must be the SAME surface anchor the radiation emitted from.

Defect (found 2026-08-02 on three completed MPAS AMIP runs): the implied
surface emissivity ``rlus / (sigma * ts**4)`` spanned 0.271 .. 1.081 —
values above 1 are thermodynamically impossible and one cell radiated like a
175 K surface.  The extremes were LAND-only (ocean cells closed to 0.1 W/m^2
RMS) because ``_feed_mpas_cmip_accumulators`` RECONSTRUCTED ``ts`` as the bare
``blend_surface_temperature(sst, sic, T_ice)`` ocean/ice blend, while
``_run_mpas`` hands the physics a ``forcing["T_sfc"]`` anchor that ALSO
carries

  1. the land-lapse correction (``mpas_land_lapse_K_per_km``, active on every
     production run), and
  2. the interactive multilayer-land skin blend
     (``use_multilayer_land``, one-step lag over ``f_land``).

``rlus`` is derived from that anchor (``rlus = rlds - lw_net_sfc``, and the
radiation emits ``eps*sigma*T_sfc**4 + (1-eps)*rlds``), so publishing a
DIFFERENT temperature as ``ts`` breaks the pair.  The fix stashes the resolved
anchor in ``_run_mpas`` and publishes THAT.

The reference reconstruction below keeps the reference's own exclusions: the
surface is NOT a blackbody (``RRTMGPConfig.sfc_emissivity``), so the physical
bound on ``rlus / (sigma * ts**4)`` is ``[eps, 1]``, not ``{1}``.
"""
from __future__ import annotations

import inspect
import types

import numpy as np
import pytest

from legoesm import constants
from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.driver.model_driver import ModelDriver
from legoesm.grids.factory import create_grid

NLEV = 6

# Surface longwave emissivity of the radiation boundary (RRTMGPConfig default).
# Measured against slabff_on's CMOR output, eps=0.98 closed
# rlus = eps*sigma*T_sfc**4 + (1-eps)*rlds over 1225 ocean cells to
# 0.109 W/m^2 RMS (eps=1.0 gave 1.194).
SFC_EMISSIVITY = 0.98


@pytest.fixture(scope="module")
def mesh():
    return create_grid("mpas", 2, lloyd_iterations=10)


def _make_collector(mesh):
    sigma_full = np.linspace(0.05, 0.98, NLEV)
    dc = DiagnosticCollector(
        nlev=NLEV, sigma_full=sigma_full, dsigma=np.full(NLEV, 1.0 / NLEV),
        experiment_id="amip", monthly_means=True, cmip_output=True,
        n_days=30, output_dir=None, cmip_resolution_deg=10.0,
        start_year=1979,
    )
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    return dc, sigma_full


def _field(a):
    return types.SimpleNamespace(data=np.asarray(a))


def _surface_sweep(mesh):
    """Representative surface sweep on the native mesh.

    ``sst``/``sic`` are the prescribed ocean/ice boundary (what the OLD ``ts``
    published).  ``anchor`` is what ``_run_mpas`` actually gives the
    radiation: the same blend over ocean, but shifted over land by the lapse
    correction and the interactive skin — smoothly in latitude so the IDW
    regrid stays accurate, and spanning BOTH signs:

      * warm low-latitude land skin (hot desert, skin >> nearest-ocean fill)
      * cold high-latitude land skin (polar night / elevated, skin << fill)
    """
    latc = np.asarray(mesh.latCell)
    n = int(mesh.nCells)
    sst = 290.0 + 12.0 * np.cos(latc)          # 278 K poles .. 302 K equator
    sic = np.zeros(n)
    # Land fraction: a smooth mid-latitude/polar-weighted "continent" so the
    # ancher offset is a smooth function of position (IDW-friendly).
    f_land = 0.5 * (1.0 + np.sin(latc)) ** 2 / 4.0 + 0.25
    # Skin offset: +22 K at the equator (desert), -45 K at the north pole.
    skin_offset = 22.0 * np.cos(latc) - 45.0 * np.sin(latc) ** 2
    anchor = sst + f_land * skin_offset
    return dict(sst=sst, sic=sic, anchor=anchor, f_land=f_land, n=n)


def _fake_driver(mesh, dc, sigma_full, sweep, *, anchor=None):
    """Lightweight ``_feed_mpas_cmip_accumulators`` stand-in.

    ``rlds`` and ``lw_net_sfc`` are built from the ANCHOR through the actual
    radiative surface law, exactly as the model does: the feed derives
    ``rlus = rlds - lw_net_sfc``.
    """
    latc = np.asarray(mesh.latCell)
    n = sweep["n"]
    T = 250.0 + 40.0 * sigma_full[None, :] + 10.0 * np.cos(latc)[:, None]
    u_edge = np.zeros((int(mesh.nEdges), NLEV))
    rlds = 0.75 * constants.sigma_sb * sweep["anchor"] ** 4
    rlus = (SFC_EMISSIVITY * constants.sigma_sb * sweep["anchor"] ** 4
            + (1.0 - SFC_EMISSIVITY) * rlds)
    # lw_net_sfc is positive INTO the surface (radiation packer convention).
    lw_net = rlds - rlus
    fake = types.SimpleNamespace(
        diagnostics=dc,
        grid=mesh,
        config=types.SimpleNamespace(T_ice=constants.T_freeze_ocean,
                                     mpas_ice_skin_prognostic=False,
                                     dycore=types.SimpleNamespace(dt=75.0)),
        get_sst_sic=lambda day: (sweep["sst"], sweep["sic"]),
        state=types.SimpleNamespace(
            u=_field(u_edge), T=_field(T), p_s=_field(np.full(n, 1.0e5)),
            phis=_field(np.zeros(n)), tracers=None,
        ),
        model=types.SimpleNamespace(
            # slots 0/1 = sw/lw net (+into sfc), 8/9 = sw/lw down.
            _sfc_diag=(None, _field(lw_net), None, None, None, None, None,
                       None, None, _field(rlds)),
        ),
        _mpas_sfc_accum=None,
    )
    if anchor is not None:
        fake._mpas_T_sfc_anchor = anchor
    return fake, rlds, rlus


def _published(dc):
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    return {k: out[f"field_2d_{k}"][0] for k in ("ts", "rlds", "rlus")}


class TestTsPublishesTheResolvedAnchor:
    def test_ts_is_the_anchor_not_the_bare_ocean_blend(self, mesh):
        """``ts`` must equal the anchor the physics consumed.

        Pre-fix this published ``blend_surface_temperature(sst, sic, T_ice)``
        — the ocean/ice blend with NO land-lapse and NO land skin — so it
        matched the blend and NOT the anchor.
        """
        dc, sigma_full = _make_collector(mesh)
        sweep = _surface_sweep(mesh)
        fake, _, _ = _fake_driver(mesh, dc, sigma_full, sweep,
                                  anchor=sweep["anchor"])
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
        pub = _published(dc)

        anchor_r = dc._regrid_to_latlon_2d(sweep["anchor"])
        blend_r = dc._regrid_to_latlon_2d(sweep["sst"])   # sic == 0
        # The sweep is a real discriminator: the two candidates differ.
        assert np.nanmax(np.abs(anchor_r - blend_r)) > 10.0
        np.testing.assert_allclose(pub["ts"], anchor_r, rtol=1e-9)

    def test_absent_anchor_falls_back_to_the_ocean_blend(self, mesh):
        """No stashed anchor (non-SST lane, or before the first step) keeps
        the previous behaviour rather than dropping ``ts``."""
        dc, sigma_full = _make_collector(mesh)
        sweep = _surface_sweep(mesh)
        fake, _, _ = _fake_driver(mesh, dc, sigma_full, sweep, anchor=None)
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
        pub = _published(dc)
        blend_r = dc._regrid_to_latlon_2d(sweep["sst"])
        np.testing.assert_allclose(pub["ts"], blend_r, rtol=1e-9)


class TestImpliedEmissivityBounds:
    def test_implied_emissivity_stays_physical_across_the_sweep(self, mesh):
        """``rlus / (sigma * ts**4)`` must sit in ``[eps, 1]``.

        This is the published-file check that failed on all three runs
        (0.271 .. 1.081).  A value above 1 is a surface radiating more than a
        blackbody at its own reported temperature; below ``eps`` means ``ts``
        is warmer than the surface that actually emitted.
        """
        dc, sigma_full = _make_collector(mesh)
        sweep = _surface_sweep(mesh)
        fake, _, _ = _fake_driver(mesh, dc, sigma_full, sweep,
                                  anchor=sweep["anchor"])
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
        pub = _published(dc)

        em = pub["rlus"] / (constants.sigma_sb * pub["ts"] ** 4)
        em = em[np.isfinite(em)]
        assert em.size > 0
        # Tolerance covers the IDW regrid's Jensen residual only: ts and rlus
        # are interpolated independently, so regrid(T^4) >= regrid(T)^4 by a
        # few 1e-3 on a smooth field.  It is ~2 orders below the 0.27/1.08
        # excursions this guards.
        assert em.max() <= 1.0 + 5e-3, f"emissivity > 1: max {em.max():.4f}"
        assert em.min() >= SFC_EMISSIVITY - 5e-3, \
            f"emissivity below eps: min {em.min():.4f}"

    def test_the_bound_is_violated_by_the_bare_ocean_blend(self, mesh):
        """Non-vacuous control: publishing the OLD ``ts`` (the ocean/ice
        blend) against the SAME ``rlus`` breaks the bound in BOTH directions.

        Guarantees the assertion above is a real tripwire, not a test that
        would pass on any input.
        """
        dc, sigma_full = _make_collector(mesh)
        sweep = _surface_sweep(mesh)
        fake, _, rlus = _fake_driver(mesh, dc, sigma_full, sweep, anchor=None)
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
        pub = _published(dc)
        em = pub["rlus"] / (constants.sigma_sb * pub["ts"] ** 4)
        em = em[np.isfinite(em)]
        assert em.max() > 1.0 + 5e-3      # hot land skin vs cooler SST fill
        assert em.min() < SFC_EMISSIVITY - 5e-3   # cold polar skin vs fill


def test_run_mpas_stashes_the_anchor_after_the_land_skin_blend():
    """``_run_mpas`` — the function that RUNS in this lane — must stash the
    resolved anchor, and must do it AFTER the interactive-land blend.

    Naming the delegating helper would prove nothing; the blend and the stash
    both live in ``_run_mpas`` itself.  Ordering is load-bearing: stashing
    before the land blend would re-publish the un-corrected ocean/ice field
    and silently restore the defect.
    """
    src = inspect.getsource(ModelDriver._run_mpas)
    assert "_mpas_T_sfc_anchor" in src, \
        "_run_mpas no longer stashes the CMOR ts anchor"
    stash = src.index("self._mpas_T_sfc_anchor")
    land_blend = src.index("_f_land_cells * _land_T_skin")
    assert stash > land_blend, \
        "the anchor is stashed BEFORE the interactive-land skin blend"


def test_feed_reads_the_stashed_anchor():
    """The consumer side names the same attribute (a stash nothing reads is
    dead code)."""
    src = inspect.getsource(ModelDriver._feed_mpas_cmip_accumulators)
    assert "_mpas_T_sfc_anchor" in src
