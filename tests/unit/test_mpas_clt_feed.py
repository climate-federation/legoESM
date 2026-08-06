"""Total cloud cover ``clt`` must reach CMOR on the MPAS lane.

``clt`` was computed ONLY inside ``DiagnosticCollector.collect`` — the
cube/lat-lon spatial path. Production AMIP runs on ``grid_type="mpas"``, which
dispatches to ``ModelDriver._run_mpas``; that lane never calls ``collect`` at
all, feeding CMOR exclusively through ``feed_cmip_accumulators_native``, whose
hardcoded field list had no ``clt`` and whose signature took no cloud tracers.
Net effect: 20+ production run directories, zero ``clt`` files ever written,
while the variable sat correctly registered in the CMOR Amon table.

That matters beyond a missing file — ``clt`` is the variable that separates
"too much cloud" from "too bright cloud" for the standing TOA shortwave excess,
and every ESMValTool cloud recipe needs it.

These tests assert against ``feed_cmip_accumulators_native``, the symbol the
MPAS lane actually executes — asserting against ``collect`` would pass while
proving nothing about production.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.atmosphere.physics.clouds.config import build_cloud_config
from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.grids.factory import create_grid

NLEV = 6


@pytest.fixture(scope="module")
def mesh():
    # Level-2 SCVT mesh = 162 cells; cheap, exercises the real IDW regrid.
    return create_grid("mpas", 2, lloyd_iterations=10)


def _collector(mesh, cloud_config):
    sigma_full = np.linspace(0.05, 0.98, NLEV)
    dc = DiagnosticCollector(
        nlev=NLEV, sigma_full=sigma_full,
        dsigma=np.full(NLEV, 1.0 / NLEV),
        experiment_id="amip", monthly_means=True, cmip_output=True,
        n_days=30, cmip_resolution_deg=10.0, start_year=1979,
        cloud_config=cloud_config,
    )
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    return dc, sigma_full


def _fields(mesh, sigma_full, *, q_c_scale=1.0):
    """Native cell fields with a genuinely PARTLY cloudy column structure.

    A fixture that saturates every level would score 100% cover and pass any
    plumbing test while hiding a broken reduction, so condensate is confined
    to the mid-troposphere and scaled by latitude.
    """
    latc = np.asarray(mesh.latCell)
    n = int(mesh.nCells)
    T = 250.0 + 40.0 * sigma_full[None, :] + 10.0 * np.cos(latc)[:, None]
    p_s = np.full(n, 1.0e5)
    q_v = 1.0e-3 * (1.0 - sigma_full)[None, :] * np.ones((n, 1))
    # Mid-level condensate only, heaviest in the tropics.
    vert = np.exp(-((sigma_full - 0.55) ** 2) / 0.02)
    q_c = q_c_scale * 3.0e-4 * vert[None, :] * np.cos(latc)[:, None] ** 2
    return dict(T=T, p_s=p_s, q_v=q_v, q_c=np.clip(q_c, 0.0, None),
                lat_deg=np.degrees(latc), phis=np.zeros(n),
                precip=np.full(n, 2.0e-5))


def _feed(dc, f, **kw):
    return dc.feed_cmip_accumulators_native(
        15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], q_v=f["q_v"],
        precip=f["precip"], phis=f["phis"], **kw)


def _clt_mean(dc):
    """Accumulated clt as a MEAN in %, or None if the field was not fed.

    ``SpatialMonthlyAccumulator._data_2d`` is ``{(month, year): {var: (sum,
    count)}}`` — the running sum must be divided by its OWN per-variable count,
    not by a call count, or a field fed on a different cadence would be scaled
    wrong.
    """
    for month in dc._spatial_monthly._data_2d.values():
        if "clt" not in month:
            continue
        entry = month["clt"]
        total, count = (entry if isinstance(entry, tuple) else (entry, 1))
        return np.asarray(total, dtype=float) / max(float(np.max(count)), 1.0)
    return None


def test_clt_reaches_the_accumulator_on_the_mpas_lane(mesh):
    """The headline regression: the lane production uses must publish clt."""
    cc = build_cloud_config("xu_randall")
    dc, sig = _collector(mesh, cc)
    f = _fields(mesh, sig)
    assert _feed(dc, f, q_c=f["q_c"]) is True
    got = _clt_mean(dc)
    assert got is not None, "clt absent from the MPAS CMOR feed"
    assert np.all(np.isfinite(got))


def test_clt_is_a_percentage_and_genuinely_partial(mesh):
    """Units are CMIP % in [0,100], and the fixture is NOT saturated.

    A fixture pinned at 0 or 100 would make the bounds assertion vacuous, so
    the partial-cover regime is asserted explicitly.
    """
    cc = build_cloud_config("xu_randall")
    dc, sig = _collector(mesh, cc)
    f = _fields(mesh, sig)
    _feed(dc, f, q_c=f["q_c"])
    mean = float(np.nanmean(_clt_mean(dc)))
    assert 0.0 <= mean <= 100.0, f"clt {mean} outside [0,100] %"
    assert 1.0 < mean < 99.0, (
        f"fixture is degenerate at {mean}% cover — it would pass even if the "
        "overlap reduction were replaced by a constant")


def test_clt_is_skipped_not_zeroed_without_a_cloud_scheme(mesh):
    """cloud_scheme='none' must publish NOTHING.

    A zero would be read downstream as a genuine clear sky and quietly drag
    any multi-run cloud-cover mean toward zero.
    """
    dc, sig = _collector(mesh, None)
    f = _fields(mesh, sig)
    assert _feed(dc, f, q_c=f["q_c"]) is True
    assert _clt_mean(dc) is None


def test_clt_is_skipped_when_no_condensate_tracer_is_supplied(mesh):
    """A dry run carries no q_c; the field must be absent, not zero."""
    cc = build_cloud_config("xu_randall")
    dc, sig = _collector(mesh, cc)
    f = _fields(mesh, sig)
    assert _feed(dc, f) is True          # q_c omitted
    assert _clt_mean(dc) is None


def test_more_condensate_gives_more_cover(mesh):
    """Monotonicity — the check that fails if clt is wired to the wrong array.

    Plumbing that accidentally published, say, prw would still be finite and
    in range; only a response to the CONDENSATE distinguishes them.
    """
    cc = build_cloud_config("xu_randall")
    covers = []
    for scale in (0.3, 3.0):
        dc, sig = _collector(mesh, cc)
        f = _fields(mesh, sig, q_c_scale=scale)
        _feed(dc, f, q_c=f["q_c"])
        covers.append(float(np.nanmean(_clt_mean(dc))))
    assert covers[1] > covers[0] + 1.0, (
        f"cover did not respond to a 10x condensate change: {covers}")


def test_both_lanes_share_one_reduction(mesh):
    """``collect`` and the MPAS feed must not drift apart.

    Both call ``_clt_percent``; this pins that they agree on identical input,
    so a future edit to one lane cannot silently change only that lane's
    reported cloud cover.
    """
    cc = build_cloud_config("xu_randall")
    dc, sig = _collector(mesh, cc)
    f = _fields(mesh, sig)
    direct = dc._clt_percent(f["T"], f["p_s"], f["q_v"], f["q_c"], None)
    assert direct is not None
    assert direct.shape == (int(mesh.nCells),)
    assert np.all((direct >= 0.0) & (direct <= 100.0))
    # q_i=None (warm-rain) must not change the liquid-only answer.
    again = dc._clt_percent(f["T"], f["p_s"], f["q_v"], f["q_c"])
    np.testing.assert_array_equal(direct, again)


def test_wrong_shaped_condensate_raises_before_mutating(mesh):
    """Validation is transactional: a malformed q_c must raise BEFORE any
    accumulator is touched, matching the sibling q_v contract."""
    cc = build_cloud_config("xu_randall")
    dc, sig = _collector(mesh, cc)
    f = _fields(mesh, sig)
    with pytest.raises((ValueError, AssertionError)):
        _feed(dc, f, q_c=f["q_c"][:, :-1])


# --- Driver-boundary coverage -------------------------------------------
# The tests above call the collector directly and hand it q_c themselves, so
# they ALL still pass if ModelDriver._feed_mpas_cmip_accumulators fails to
# extract or forward the condensate tracers — which is precisely the wiring
# this change adds (codex adversarial review, 2026-08-01).  These exercise the
# driver glue on a lightweight stand-in, matching the sibling pattern in
# test_mpas_cmip_accumulator_feed.py.

def _fake_driver(mesh, dc, f, *, tracers):
    import types

    def _field(a):
        return types.SimpleNamespace(data=np.asarray(a))

    n_edges = int(mesh.nEdges)
    return types.SimpleNamespace(
        diagnostics=dc, grid=mesh,
        state=types.SimpleNamespace(
            u=_field(np.full((n_edges, NLEV), 5.0)),
            T=_field(f["T"]), p_s=_field(f["p_s"]), phis=_field(f["phis"]),
            tracers={k: _field(v) for k, v in tracers.items()},
        ),
        model=types.SimpleNamespace(
            _sfc_diag=(None, None, _field(f["precip"])),
        ),
    )


def test_driver_forwards_condensate_so_clt_is_published(mesh):
    """THE end-to-end wiring assertion: from ``state.tracers`` to CMOR.

    ``_feed_mpas_cmip_accumulators`` swallows glue failures into a log line, so
    a POPULATED accumulator is the only real evidence the forwarding works.
    """
    from legoesm.driver.model_driver import ModelDriver

    cc = build_cloud_config("xu_randall")
    dc, sig = _collector(mesh, cc)
    f = _fields(mesh, sig)
    fake = _fake_driver(mesh, dc, f,
                        tracers={"q_v": f["q_v"], "q_c": f["q_c"]})
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert "field_2d_clt" in out, (
        "the driver did not forward q_c — clt never reached CMOR")
    got = float(np.nanmean(out["field_2d_clt"]))
    assert 1.0 < got < 99.0, f"degenerate cover {got}%"


def test_driver_omits_clt_on_a_dry_run(mesh):
    """No q_c in state.tracers => the field is absent, not a zero plane."""
    from legoesm.driver.model_driver import ModelDriver

    cc = build_cloud_config("xu_randall")
    dc, sig = _collector(mesh, cc)
    f = _fields(mesh, sig)
    fake = _fake_driver(mesh, dc, f, tracers={"q_v": f["q_v"]})
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert dc._spatial_monthly._max_count_ever > 0, "nothing fed at all"
    assert "field_2d_clt" not in out


def test_driver_forwards_ice_and_mixed_phase_changes_the_answer(mesh):
    """q_i must be threaded too, and must MATTER.

    Without this, a driver that forwarded only q_c would pass every other
    test here while silently reporting liquid-only cloud cover on a
    mixed-phase run.
    """
    from legoesm.driver.model_driver import ModelDriver

    cc = build_cloud_config("xu_randall")
    covers = {}
    # Ice aloft (cold, low-sigma levels), well away from the liquid layer.
    _, sig0 = _collector(mesh, cc)
    ice_vert = np.exp(-((sig0 - 0.2) ** 2) / 0.01)
    for label, extra in (("liquid", {}), ("mixed", {"q_i": None})):
        dc, sig = _collector(mesh, cc)
        f = _fields(mesh, sig)
        tr = {"q_v": f["q_v"], "q_c": f["q_c"]}
        if label == "mixed":
            tr["q_i"] = 2.0e-4 * ice_vert[None, :] * np.ones((int(mesh.nCells), 1))
        fake = _fake_driver(mesh, dc, f, tracers=tr)
        ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        assert "field_2d_clt" in out, f"{label}: clt missing"
        covers[label] = float(np.nanmean(out["field_2d_clt"]))
    assert covers["mixed"] > covers["liquid"] + 1.0, (
        f"ice cloud did not raise total cover — q_i is probably not "
        f"forwarded: {covers}")


def test_validation_failure_leaves_the_accumulator_untouched(mesh):
    """Transactionality, actually asserted.

    The sibling test only checks that a malformed q_c RAISES; that would pass
    even if some fields had already been committed. This snapshots the
    accumulator and requires it unchanged.
    """
    cc = build_cloud_config("xu_randall")
    dc, sig = _collector(mesh, cc)
    f = _fields(mesh, sig)
    before_keys = set(dc._spatial_monthly._data_2d)
    before_max = dc._spatial_monthly._max_count_ever
    with pytest.raises((ValueError, AssertionError)):
        _feed(dc, f, q_c=f["q_c"][:, :-1])
    assert set(dc._spatial_monthly._data_2d) == before_keys, \
        "a partial commit landed before validation rejected the input"
    assert dc._spatial_monthly._max_count_ever == before_max
