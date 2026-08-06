# Adversarial review ROUND 4: MPAS monthly-CMOR accumulator feed

You are an independent adversarial reviewer. Round 4. Rounds 1–3 validated the
FEED, the 2 m tas, the MPI gate, the CMIP-only clean-completion write, and the
PHASE1/PHASE2 structure. Round 3 left exactly two items; this round fixes both.
Re-review ONLY these two fixes + the one residual (inherited) note. Cite lines.

## Round-3 findings and disposition

**#1 (HIGH) — exact-checkpoint-cadence completion left a stale terminal
sidecar.** FIXED. When the last step is a checkpoint multiple, the in-loop
periodic checkpoint writes `cmor_accum_day_<final>.npz` BEFORE the finalizer
runs, and the after-loop "final checkpoint" is skipped — so `_suppress_cmor_
sidecar` (which only blocks a FUTURE write) could not remove it. The finalizer
now takes `final_day` and, on a successful write, EXPLICITLY UNLINKS the
terminal-day sidecar (`cmor_accum_day_<round(final_day)>.npz`) in addition to
setting the suppress flag. So both cases are covered: non-exact cadence →
suppress blocks the still-to-come final checkpoint's sidecar; exact cadence →
the already-written sidecar is retired. A restart from the final checkpoint then
finds no sidecar → resumes with empty accumulators (a COMPLETED run has no
in-progress month to carry) → no re-append. Test:
`test_driver_finalize_retires_exact_cadence_sidecar` (pre-creates the same-day
sidecar, finalizes, asserts it is gone + suppress set + NetCDF written).

**#2 (MEDIUM) — shape-correct but non-numeric `lat_deg` still committed
spatial before failing in the zonal `np.digitize`.** FIXED. PHASE 1 now coerces
`lat_deg` with `np.asarray(lat_deg, dtype=np.float64)` (a non-numeric object/
string vector raises HERE, before any commit) and additionally rejects
non-finite values. The spatial fields are shape-guaranteed by
`_regrid_to_latlon_*` (a non-numeric `T` would already raise inside the regrid,
still PHASE 1), and every zonal field is derived from the same `(nCells, …)`
arrays as the validated `lat` — so `np.add.at` lengths always match. Tests:
`test_feed_atomic_on_bad_lat_deg` (length mismatch),
`test_feed_atomic_on_nonnumeric_or_nonfinite_lat` (object vector + NaN).

**Residual (inherited) — the CMOR writer is not itself transactional.** A
mid-write I/O failure can leave a partially-written NetCDF (the shared `save()`
path has the identical property — `write_field` appends field-by-field). I
softened the finalizer's failure log (no "rerun the writer" claim) and
documented that a durable per-write progress record is the follow-up for full
crash-safety. On such a failure the sidecar is intentionally NOT retired. This
is pre-existing writer behaviour, not introduced by this change.

## Facts

- Sidecar filename: `cmor_accum_day_{int(round(day)):04d}.npz` in
  `self._output_dir` (matches `_save_cmor_accumulator_sidecar`).
- The finalizer runs on `run_status == "COMPLETED"` and `_mpas_cmip_feed_on`
  (serial / 1-rank), rank-0 only, BEFORE the final-checkpoint block.
- `final_day = START_DAY + n_steps_total * DT / 86400.0` (the same day the
  final/periodic checkpoint uses).

## THE ROUND-4 CHANGES

### ROUND-4 incremental fixes (addressing round-3 HIGH + MEDIUM)

#### diagnostics.py :: PHASE-1 lat_deg numeric+finite validation (1458-1483)
```python
        z3d: dict[str, np.ndarray] = {}
        lat_np = None
        if have_zonal:
            # Validate the zonal lat vector HERE, in PHASE 1, before any commit:
            # the zonal add_* bin via ``np.digitize(lat, ...)`` / ``np.add.at``
            # and would raise mid-commit — AFTER the spatial add_* already
            # mutated their buckets.  Catching it pre-commit keeps the whole
            # feed transactional (the spatial fields are shape-guaranteed by
            # ``_regrid_to_latlon_*`` — a non-numeric one would already raise in
            # the regrid, still PHASE 1 — so lat_deg is the only remaining
            # commit-time failure mode).  Force a numeric float array so a
            # shape-correct-but-non-numeric (object/string) lat_deg raises here,
            # not inside ``np.digitize`` mid-commit; also require finite values.
            lat_np = np.asarray(lat_deg, dtype=np.float64)
            _ncol = int(T_np.shape[0])
            if lat_np.shape != (_ncol,):
                raise ValueError(
                    "feed_cmip_accumulators_native: lat_deg shape "
                    f"{lat_np.shape} != expected ({_ncol},)")
            if not np.all(np.isfinite(lat_np)):
                raise ValueError(
                    "feed_cmip_accumulators_native: lat_deg has non-finite "
                    "values")
            z2d['T_low'] = np.asarray(T_low)   # lowest model level, NOT 2 m tas
            if precip_np is not None:
                z2d['precip'] = precip_np * 86400.0     # [mm/day], like collect()
```
#### model_driver.py :: _finalize_mpas_cmip (now retires the terminal sidecar; 5390-5451)
```python
    def _finalize_mpas_cmip(self, final_day: float | None = None) -> None:
        """Write the CMOR NetCDF (``Amon`` / ``day`` / ``fx``) from the fed
        accumulators at a CLEAN MPAS completion.

        ``_run_mpas`` bypasses the shared :meth:`_finalize_run` (whose
        ``diagnostics.save()`` ALSO writes the cube/lat-lon ``timeseries.npz`` +
        snapshots this lean path tracks SEPARATELY in ``_ts`` — calling it here
        would clobber the lightweight timeseries).  So mirror ONLY the
        CMIP-file finalize of ``save()``.  Wallclock-graceful exits already
        flush COMPLETED months incrementally and POP them
        (:meth:`_maybe_wallclock_exit`); this covers the final / standalone
        clean completion, writing whatever remains (in-progress + not-yet-
        flushed months, subject to the finalize partial-month guard) — which
        otherwise left the now-fed accumulators unwritten (empty ``cmor/``).

        The writers do NOT pop, so this is TERMINAL.  On success it (a) sets
        ``_suppress_cmor_sidecar`` so a still-to-be-written final checkpoint
        skips its CMOR sidecar, AND (b) RETIRES (unlinks) the terminal-day
        sidecar if one was ALREADY written this step — the exact-checkpoint-
        cadence case, where the in-loop periodic checkpoint wrote the sidecar
        BEFORE this finalizer runs and the after-loop "final checkpoint" is
        skipped, so suppression alone (which only blocks a FUTURE write) would
        leave a stale same-day sidecar that a run-EXTENDING restart would
        restore and re-append (duplicate ``time`` coords).  Mirrors the intent
        of :meth:`_finalize_run`.  Rank-0 only; fully guarded (a writer failure
        must not turn a COMPLETED run into a crash after the science is done —
        the accumulators/sidecar are left intact for inspection).

        NOTE: the underlying CMOR writer appends field-by-field and is not
        itself transactional, so a mid-write I/O failure can leave a partially
        written NetCDF (inherited from the shared ``save()`` path); on such a
        failure the sidecar is intentionally NOT retired, but a blind retry
        could still duplicate the already-appended fields — a durable
        per-write progress record is the follow-up for full crash-safety."""
        diag = getattr(self, "diagnostics", None)
        if diag is None or getattr(diag, "cf_writer", None) is None:
            return
        _is_root = (getattr(self, "_mpi_rank", None) is None
                    or self._mpi_rank == 0)
        if not _is_root:
            return
        try:
            diag._write_cmip_monthly_files()
            diag._write_cmip_daily_files()
            diag._write_cmip_fixed_files()
            diag.cf_writer.close()
            # Terminal: block any future sidecar write AND retire a same-day
            # sidecar already written by the in-loop periodic checkpoint.
            self._suppress_cmor_sidecar = True
            if final_day is not None:
                _out = getattr(self, "_output_dir", None)
                if _out is not None:
                    _sc = Path(_out) / (
                        f"cmor_accum_day_{int(round(final_day)):04d}.npz")
                    if _sc.exists():
                        _sc.unlink()
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.error(
                "  MPAS CMOR NetCDF finalize FAILED (accumulators/sidecar left "
                "intact for inspection; the NetCDF may be partially written — "
                "the shared CMOR writer is not transactional): %s", exc)

```
#### model_driver.py :: caller passes final_day (6450-6462)
```python
        # Write the CMOR NetCDF from the (now-fed) accumulators on a CLEAN
        # completion.  MUST run BEFORE the final checkpoint below so its
        # ``_suppress_cmor_sidecar`` (set on a successful write) reaches a
        # still-to-be-written final checkpoint; ``final_day`` also lets it
        # RETIRE a same-day sidecar already written by the last in-loop
        # periodic checkpoint (exact-checkpoint-cadence completion).  Gated on
        # the feed being active (serial / 1-rank with CMIP output); a no-op
        # otherwise.
        if run_status == "COMPLETED" and self._mpas_cmip_feed_on:
            self._finalize_mpas_cmip(
                START_DAY + n_steps_total * DT / 86400.0)

        # Final checkpoint so the next chain link resumes from the exact end
```


## Test status

25 tests in `tests/unit/test_mpas_cmip_accumulator_feed.py` pass. Broader sweep
(collector, voronoi regrid, cmor accumulator restart, no-hardcoded-constants,
physics-contracts) = 3539 passed, 2 skipped. One PRE-EXISTING unrelated failure
(`test_cmor_experiments_restart.py::…test_wallclock_exit_writes_cmor_fx_before_exit`;
its fake lacks `_save_cmor_accumulator_sidecar`, which the byte-identical
committed `_maybe_wallclock_exit` calls — not touched by this change).

## Your task

Confirm: (a) does retiring the terminal-day sidecar (unlink) fully close the
exact-cadence re-append hazard, and is the filename/day computation exactly the
one the periodic checkpoint used? (b) is the numeric+finite `lat_deg` coercion
sufficient to make the commit transactional across all supported configs
(spatial-only, zonal-only, both)? (c) any NEW bug in the unlink path (e.g.,
wrong dir, deleting a needed non-terminal sidecar) or the tests? If both fixes
are correct and only the documented inherited writer non-atomicity remains, say
the review is CLEAN and there are no substantive blocking bugs.


## APPENDIX A: full updated test file

```python
"""MPAS (Voronoi) CMOR monthly/daily/zonal accumulator feed.

Regression tests for the historical gap where the lean MPAS AMIP loop
constructed the ``DiagnosticCollector`` CMOR accumulators
(``SpatialMonthlyAccumulator`` / ``SpatialDailyAccumulator`` /
``MonthlyAccumulator``) and SAVED the restart sidecar, but NEVER fed them —
so every manifest showed ``call_counts: []`` / ``max_count_ever: 0``.

Two layers are covered:

* ``DiagnosticCollector.feed_cmip_accumulators_native`` — regrids native
  Voronoi cell fields to the CMOR lat-lon grid and bins the zonal means,
  populating all three accumulators.
* ``ModelDriver._feed_mpas_cmip_accumulators`` — the driver glue that
  reconstructs geographic cell winds from the edge-normal ``state.u``,
  pulls the surface-precip export, and calls the collector method.

Plus regrid unit checks (uniform -> uniform; hemisphere-split -> correct
sign per hemisphere) on the collector's Voronoi-capable ``_regrid_to_latlon``
helpers.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.factory import create_grid
from legoesm.grids.voronoi import reconstruct_cell_velocity
from legoesm.driver.diagnostics import DiagnosticCollector

NLEV = 6


@pytest.fixture(scope="module")
def mesh():
    # Level-2 SCVT mesh = 162 cells / 480 edges; cheap, enough to exercise
    # both the IDW regrid and the Perot edge->cell wind reconstruction.
    return create_grid("mpas", 2, lloyd_iterations=10)


def _make_collector(mesh, *, monthly_means=True, cmip_output=True,
                    cmip_resolution_deg=10.0, output_dir=None):
    sigma_full = np.linspace(0.05, 0.98, NLEV)
    dsigma = np.full(NLEV, 1.0 / NLEV)
    dc = DiagnosticCollector(
        nlev=NLEV, sigma_full=sigma_full, dsigma=dsigma,
        experiment_id="amip", monthly_means=monthly_means,
        cmip_output=cmip_output, n_days=30, output_dir=output_dir,
        cmip_resolution_deg=cmip_resolution_deg, start_year=1979,
    )
    if cmip_output:
        dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    return dc, sigma_full, dsigma


def _synthetic_cell_fields(mesh, sigma_full):
    """Physically-plausible native cell fields on the mesh."""
    latc = np.asarray(mesh.latCell)
    n = int(mesh.nCells)
    # T: warmer near the surface (large sigma) and near the equator.
    T = 250.0 + 40.0 * sigma_full[None, :] + 10.0 * np.cos(latc)[:, None]
    p_s = np.full(n, 1.0e5)
    q_v = 1e-3 * (1.0 - sigma_full)[None, :] * np.ones((n, 1))
    precip = np.full(n, 2.0e-5)      # kg/m2/s
    phis = np.zeros(n)
    lat_deg = np.degrees(latc)
    # Uniform east wind U=7 as an edge-normal field: u_edge = U*cos(angleEdge).
    U = 7.0
    ang = np.asarray(mesh.angleEdge)
    u_edge = np.broadcast_to(
        (U * np.cos(ang))[:, None], (int(mesh.nEdges), NLEV)
    ).copy()
    return dict(T=T, p_s=p_s, q_v=q_v, precip=precip, phis=phis,
                lat_deg=lat_deg, u_edge=u_edge, U=U)


# ---------------------------------------------------------------------------
# Collector-level feed
# ---------------------------------------------------------------------------

def test_feed_populates_all_three_accumulators(mesh):
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)

    # Pre-condition: this is the exact broken symptom — empty accumulators.
    assert dc._spatial_monthly._max_count_ever == 0
    assert dc._spatial_daily._max_count_ever == 0
    assert dc.monthly_accum._max_count_ever == 0

    fed = dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
        q_v=f["q_v"], u_east=np.asarray(u_east), v_north=np.asarray(v_north),
        precip=f["precip"], phis=f["phis"],
    )
    assert fed is True

    # All three accumulators now have non-empty per-bucket call counts.
    for acc in (dc._spatial_monthly, dc._spatial_daily, dc.monthly_accum):
        assert acc._max_count_ever > 0
        assert len(acc._call_counts) > 0
        assert all(c > 0 for c in acc._call_counts.values())

    # day 15 -> month 1 (Jan), year 0.  Daily bucket keyed on (year, doy=16).
    assert (0, 1) in dc._spatial_monthly._call_counts
    assert (0, 16) in dc._spatial_daily._call_counts


def test_feed_binned_values_are_plausible(mesh):
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
        q_v=f["q_v"], u_east=np.asarray(u_east), v_north=np.asarray(v_north),
        precip=f["precip"], phis=f["phis"],
    )
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert out["months"] == [(0, 1)]

    # ps / pr are spatially uniform -> IDW (partition of unity) is EXACT.
    np.testing.assert_allclose(out["field_2d_ps"], 1.0e5, rtol=1e-9)
    np.testing.assert_allclose(out["field_2d_pr"], 2.0e-5, rtol=1e-9)
    # psl == ps here (phis == 0).
    np.testing.assert_allclose(out["field_2d_psl"], 1.0e5, rtol=1e-9)

    # tas: bounded near-surface temperature; prw positive column vapour.
    tas = out["field_2d_tas"]
    assert np.all(np.isfinite(tas)) and tas.min() > 250.0 and tas.max() < 320.0
    assert np.all(out["field_2d_prw"] > 0.0)

    # 3-D ta on plev19: finite, physical, and the eastward wind ua recovers
    # the ~U=7 m/s uniform flow while va stays ~0.
    ta = out["field_3d_ta"]
    assert ta.shape[-1] == 19
    assert np.all(np.isfinite(ta)) and ta.min() > 180.0 and ta.max() < 330.0
    # Vertical structure (non-vacuous): PLEV19 is ascending, so index -1 is
    # the 100000 Pa (surface) level and index 0 the 100 Pa (top).  The input
    # T warms toward the surface, so ta[...,-1] > ta[...,0] — a vertical flip
    # or a collapsed profile would fail this.
    assert float(np.nanmean(ta[..., -1])) > float(np.nanmean(ta[..., 0])) + 10.0
    assert abs(float(np.nanmean(out["field_3d_ua"])) - f["U"]) < 1.5
    assert abs(float(np.nanmean(out["field_3d_va"]))) < 0.5


def test_feed_zonal_and_daily_finalize(mesh):
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)
    # Feed two intervals in the same month so the mean is well defined.
    for day in (10.0, 20.0):
        dc.feed_cmip_accumulators_native(
            day=day, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
            q_v=f["q_v"], u_east=np.asarray(u_east),
            v_north=np.asarray(v_north), precip=f["precip"], phis=f["phis"],
        )
    # Zonal monthly finalize: T profile present, plausible.
    zout = dc.monthly_accum.finalize(min_sample_fraction=0)
    assert (0, 1) in zout["months"]
    assert "profile_T" in zout
    prof_T = zout["profile_T"]
    finite = np.isfinite(prof_T)
    assert finite.any()
    assert prof_T[finite].min() > 180.0 and prof_T[finite].max() < 340.0

    # Daily finalize: two distinct days, tas extremes tracked.
    dout = dc._spatial_daily.finalize(min_sample_fraction=0)
    assert len(dout["days"]) == 2
    assert "field_2d_tas" in dout
    assert "field_2d_tas_min" in dout and "field_2d_tas_max" in dout


def test_feed_dry_minimal_inputs(mesh):
    """A dry run (no q_v / precip / winds) still feeds tas/ps + zonal T."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    fed = dc.feed_cmip_accumulators_native(
        day=5.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
    )
    assert fed is True
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert "field_2d_tas" in out and "field_2d_ps" in out
    # No moisture / precip / wind fields fabricated.
    assert "field_2d_pr" not in out
    assert "field_3d_hus" not in out
    assert "field_3d_ua" not in out
    assert dc.monthly_accum._max_count_ever > 0


def test_feed_noop_when_accumulators_disabled(mesh):
    """No CMIP output and no monthly means -> feed is a no-op (returns False)."""
    dc, sigma_full, _ = _make_collector(
        mesh, monthly_means=False, cmip_output=False)
    assert dc._spatial_monthly is None
    f = _synthetic_cell_fields(mesh, sigma_full)
    fed = dc.feed_cmip_accumulators_native(
        day=5.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"])
    assert fed is False


def test_feed_manifest_roundtrip_nonempty(mesh, tmp_path):
    """The restart sidecar manifest is non-empty after a feed (the exact
    field that was ``call_counts: []`` in the broken pilot runs)."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
        q_v=f["q_v"], u_east=np.asarray(u_east), v_north=np.asarray(v_north),
        precip=f["precip"], phis=f["phis"],
    )
    sidecar = tmp_path / "cmor_accum_day_0015.npz"
    dc.save_cmor_accumulators(sidecar)

    z = np.load(sidecar, allow_pickle=True)
    manifests = {k: json.loads(str(z[k].item())) if isinstance(z[k].item(), str)
                 else z[k].item()
                 for k in z.keys() if "manifest" in k}
    assert manifests, "sidecar wrote no manifests"
    for name, m in manifests.items():
        assert m["max_count_ever"] > 0, f"{name} still empty after feed"
        assert m["call_counts"], f"{name} call_counts empty after feed"


# ---------------------------------------------------------------------------
# Regrid helper: uniform + hemisphere-sign
# ---------------------------------------------------------------------------

def test_regrid_uniform_field_is_uniform(mesh):
    dc, _, _ = _make_collector(mesh)
    n = int(mesh.nCells)
    out2d = dc._regrid_to_latlon_2d(np.full(n, 5.0))
    assert out2d is not None
    assert out2d.shape == (dc._cmip_nlat, dc._cmip_nlon)
    np.testing.assert_allclose(out2d, 5.0, atol=1e-9)

    out3d = dc._regrid_to_latlon_3d(
        np.full((n, NLEV), 3.0) * np.arange(1, NLEV + 1))
    assert out3d.shape == (dc._cmip_nlat, dc._cmip_nlon, NLEV)
    for lvl in range(NLEV):
        np.testing.assert_allclose(out3d[..., lvl], 3.0 * (lvl + 1), atol=1e-9)


def test_regrid_hemisphere_split_keeps_sign(mesh):
    """A +1 (NH) / -1 (SH) cell field regrids to a lat-lon field that is
    positive in the northern rows and negative in the southern rows."""
    dc, _, _ = _make_collector(mesh)
    latc = np.asarray(mesh.latCell)
    field = np.where(latc >= 0.0, 1.0, -1.0)
    out = dc._regrid_to_latlon_2d(field)
    assert out is not None

    lat_cent = dc._voronoi_regrid_weights.lat_cent  # linspace(-90, 90, nlat)
    nh = lat_cent > 20.0
    sh = lat_cent < -20.0
    assert nh.any() and sh.any()
    # Well inside each hemisphere the IDW blend cannot flip the sign.
    assert np.all(out[nh, :] > 0.5), "northern rows lost their + sign"
    assert np.all(out[sh, :] < -0.5), "southern rows lost their - sign"
    # Values stay within the source range (IDW is a convex combination).
    assert out.min() >= -1.0 - 1e-9 and out.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Driver glue: reconstruct winds + pull precip + feed
# ---------------------------------------------------------------------------

def test_driver_helper_feeds_via_reconstruct(mesh):
    """``ModelDriver._feed_mpas_cmip_accumulators`` reconstructs cell winds
    from the edge-normal ``state.u``, extracts the surface-precip export, and
    feeds all three accumulators.  Exercised on a lightweight stand-in so the
    exact glue (shapes, handles) is covered without a full driver build."""
    from legoesm.driver.model_driver import ModelDriver

    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)

    def _field(a):
        return types.SimpleNamespace(data=np.asarray(a))

    fake = types.SimpleNamespace(
        diagnostics=dc,
        grid=mesh,
        state=types.SimpleNamespace(
            u=_field(f["u_edge"]),          # (nEdges, nlev) edge-normal
            T=_field(f["T"]),
            p_s=_field(f["p_s"]),
            phis=_field(f["phis"]),
            tracers={"q_v": _field(f["q_v"])},
        ),
        model=types.SimpleNamespace(
            _sfc_diag=(None, None, _field(f["precip"])),
        ),
    )
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)

    # The try/except in the helper swallows glue bugs into a log line, so a
    # populated accumulator is the real assertion that the wiring works.
    assert dc._spatial_monthly._max_count_ever > 0
    assert dc.monthly_accum._max_count_ever > 0
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    # Winds threaded through -> ua present and ~uniform east flow recovered.
    assert "field_3d_ua" in out
    assert abs(float(np.nanmean(out["field_3d_ua"])) - f["U"]) < 1.5
    # Precip export threaded through -> pr present and exact-uniform.
    np.testing.assert_allclose(out["field_2d_pr"], 2.0e-5, rtol=1e-9)


def test_driver_helper_dry_run_no_precip(mesh):
    """No ``_sfc_diag`` precip slot -> feed still populates tas/ps (no pr)."""
    from legoesm.driver.model_driver import ModelDriver

    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)

    def _field(a):
        return types.SimpleNamespace(data=np.asarray(a))

    fake = types.SimpleNamespace(
        diagnostics=dc,
        grid=mesh,
        get_sst_sic=None,
        state=types.SimpleNamespace(
            u=_field(f["u_edge"]), T=_field(f["T"]), p_s=_field(f["p_s"]),
            phis=_field(f["phis"]), tracers=None,
        ),
        model=types.SimpleNamespace(_sfc_diag=None),
    )
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert "field_2d_tas" in out and "field_2d_ps" in out
    assert "field_2d_pr" not in out
    assert "field_3d_hus" not in out


# ---------------------------------------------------------------------------
# 2 m tas / psl / 850 hPa — non-vacuous physical structure
# ---------------------------------------------------------------------------

def test_feed_uses_provided_2m_tas(mesh):
    """A distinct 2 m `tas` field is regridded as `tas` — NOT the lowest
    model level (guards against the field being ignored or overwritten)."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    tas2m = f["T"][:, -1] + 5.0                    # a clearly-distinct field
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], tas=tas2m)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    tas_r = out["field_2d_tas"]
    low_r = dc._regrid_to_latlon_2d(f["T"][:, -1])
    # Regrid is linear -> the +5 offset survives; tas must equal low+5, not low.
    np.testing.assert_allclose(tas_r[0], low_r + 5.0, atol=1e-6)
    assert float(np.nanmean(tas_r)) > float(np.nanmean(low_r)) + 4.9


def test_feed_psl_exceeds_ps_over_mountains(mesh):
    """psl reduces surface pressure to sea level: psl > ps where phis > 0,
    psl == ps where phis == 0."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    phis = np.full(int(mesh.nCells), 2000.0 * constants.g)   # ~2000 m orog
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], phis=phis)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    psl = out["field_2d_psl"][0]
    ps = out["field_2d_ps"][0]
    assert np.all(psl > ps + 1.0), "psl must exceed ps over positive orography"
    # ~2 km, ~290 K -> hypsometric factor exp(gz/RT) ~ 1.27 -> ~127 hPa higher.
    assert float(np.mean(psl - ps)) > 5000.0

    # Sanity: phis == 0 -> psl == ps.
    dc0, sig0, _ = _make_collector(mesh)
    dc0.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
        phis=np.zeros(int(mesh.nCells)))
    out0 = dc0._spatial_monthly.finalize(min_sample_fraction=0)
    np.testing.assert_allclose(
        out0["field_2d_psl"][0], out0["field_2d_ps"][0], rtol=1e-9)


def test_feed_level_varying_winds_850(mesh):
    """A vertically-sheared zonal jet: ua850 recovers the 850 hPa level value
    and differs from the surface — a non-vacuous 850-slice / plev check."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    # U increases with height: U(sigma) = 5 + 20*(1 - sigma).
    ang = np.asarray(mesh.angleEdge)
    U_lev = 5.0 + 20.0 * (1.0 - sigma_full)          # surface ~5.4, top ~24
    u_edge = (U_lev[None, :] * np.cos(ang)[:, None])
    u_east, v_north = reconstruct_cell_velocity(u_edge, mesh)
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
        u_east=np.asarray(u_east), v_north=np.asarray(v_north))
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    ua = out["field_3d_ua"][0]                       # (nlat, nlon, 19)
    from legoesm.io.cmor_output import CMIP6_PLEV19
    plev = np.sort(np.asarray(CMIP6_PLEV19))
    i850 = int(np.argmin(np.abs(plev - 85000.0)))
    i_top = 0
    # p_s = 1e5 -> sigma_850 = 0.85 -> U ~ 5 + 20*0.15 = 8.0 m/s.
    ua850 = float(np.nanmean(ua[..., i850]))
    ua_top = float(np.nanmean(ua[..., i_top]))
    assert abs(ua850 - 8.0) < 2.0, f"ua850={ua850} not near the 850 hPa value"
    assert ua_top > ua850 + 3.0, "jet must strengthen aloft (non-vacuous shear)"


# ---------------------------------------------------------------------------
# Calendar buckets + restart resume
# ---------------------------------------------------------------------------

def test_feed_month_year_buckets(mesh):
    """Distinct simulated days land in the correct (year, month) buckets,
    including a Dec bucket and a year-1 bucket (restart-chain calendar)."""
    from legoesm.forcing.time_utils import day_to_calendar
    from legoesm.diagnostics.monthly_means import MonthlyAccumulator as _MA

    def _key(day):
        doy, _ = day_to_calendar(day)
        return (int(day // 365.0), _MA.day_to_month(doy))

    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    days = [360.0, 400.0]                            # Dec y0, then y1
    for d in days:
        dc.feed_cmip_accumulators_native(
            day=d, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"])
    keys = set(dc._spatial_monthly._call_counts)
    exp = {_key(d) for d in days}
    assert keys == exp
    years = {k[0] for k in keys}
    assert years == {0, 1}                           # crossed a calendar year
    assert (0, 12) in keys                           # December of year 0


def test_load_cmor_accumulators_resume_roundtrip(mesh, tmp_path):
    """A fed sidecar restored via load_cmor_accumulators into a FRESH collector
    reproduces the same finalized monthly means (the real restart path)."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], q_v=f["q_v"],
        u_east=np.asarray(u_east), v_north=np.asarray(v_north),
        precip=f["precip"], phis=f["phis"])
    sidecar = tmp_path / "cmor_accum_day_0015.npz"
    dc.save_cmor_accumulators(sidecar)
    ref = dc._spatial_monthly.finalize(min_sample_fraction=0)

    dc2, _, _ = _make_collector(mesh)
    assert dc2._spatial_monthly._max_count_ever == 0
    restored = dc2.load_cmor_accumulators(sidecar)
    assert restored is True
    got = dc2._spatial_monthly.finalize(min_sample_fraction=0)
    assert got["months"] == ref["months"]
    for k in ("field_2d_tas", "field_2d_ps", "field_2d_pr", "field_3d_ta"):
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-9, equal_nan=True)


# ---------------------------------------------------------------------------
# Serial / MPI gate + driver-side 2 m tas
# ---------------------------------------------------------------------------

def test_feed_gate_serial_mpi_singlerank(mesh):
    """`_mpas_cmip_feed_enabled`: serial and 1-rank feed; multi-rank does not,
    but reports wants_cmip=True so the caller can warn."""
    from legoesm.driver.model_driver import ModelDriver
    dc, _, _ = _make_collector(mesh)

    # Serial (no voronoi layout).
    serial = types.SimpleNamespace(_voronoi_layout=None, _mpi_world_size=1)
    assert ModelDriver._mpas_cmip_feed_enabled(serial, dc) == (True, True)

    # Multi-rank cell partition -> disabled, but wants_cmip True.
    multi = types.SimpleNamespace(
        _voronoi_layout=object(), _mpi_world_size=4)
    assert ModelDriver._mpas_cmip_feed_enabled(multi, dc) == (False, True)

    # 1-rank "distributed" layout owns the whole mesh -> safe to feed.
    single = types.SimpleNamespace(
        _voronoi_layout=object(), _mpi_world_size=1)
    assert ModelDriver._mpas_cmip_feed_enabled(single, dc) == (True, True)

    # No CMIP output at all -> neither.
    dc_off, _, _ = _make_collector(mesh, monthly_means=False, cmip_output=False)
    assert ModelDriver._mpas_cmip_feed_enabled(serial, dc_off) == (False, False)


def test_driver_helper_computes_2m_tas(mesh):
    """With prescribed sst warmer than the lowest air level, the driver helper
    publishes a 2 m MOST `tas` distinct from (warmer than) the lowest level."""
    from legoesm.driver.model_driver import ModelDriver
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    T_low = f["T"][:, -1]

    def _field(a):
        return types.SimpleNamespace(data=np.asarray(a))

    sst = T_low + 10.0                               # warm surface -> unstable
    sic = np.zeros(int(mesh.nCells))
    fake = types.SimpleNamespace(
        diagnostics=dc,
        grid=mesh,
        config=types.SimpleNamespace(T_ice=271.4),
        get_sst_sic=lambda day: (sst, sic),
        state=types.SimpleNamespace(
            u=_field(f["u_edge"]), T=_field(f["T"]), p_s=_field(f["p_s"]),
            phis=_field(f["phis"]), tracers={"q_v": _field(f["q_v"])},
        ),
        model=types.SimpleNamespace(_sfc_diag=(None, None, _field(f["precip"]))),
    )
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    tas_r = out["field_2d_tas"][0]
    low_r = dc._regrid_to_latlon_2d(T_low)
    # 2 m sits between the warm surface and the cooler lowest level -> warmer
    # than the lowest level; a bug that published T[...,-1] would fail this.
    assert float(np.nanmean(tas_r)) > float(np.nanmean(low_r)) + 0.5
    assert not np.allclose(tas_r, low_r)


# ---------------------------------------------------------------------------
# Atomic commit (no partial state on a bad input)
# ---------------------------------------------------------------------------

def test_feed_atomic_on_bad_lat_deg(mesh):
    """A malformed `lat_deg` is rejected in PHASE 1 (before any add_*), so all
    three accumulators are left untouched — the feed is transactional."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    bad_lat = np.zeros(int(mesh.nCells) + 3)          # wrong length
    with pytest.raises(ValueError, match="lat_deg shape"):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=bad_lat, precip=f["precip"])
    # Nothing committed anywhere (would be > 0 if the spatial add ran first).
    assert dc._spatial_monthly._max_count_ever == 0
    assert dc._spatial_daily._max_count_ever == 0
    assert dc.monthly_accum._max_count_ever == 0
    assert not dc._spatial_monthly._data_2d
    assert not dc.monthly_accum._data


def test_feed_atomic_on_nonnumeric_or_nonfinite_lat(mesh):
    """A shape-correct but non-numeric (or non-finite) `lat_deg` is rejected in
    PHASE 1 too — before np.digitize could raise mid-commit."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    n = int(mesh.nCells)
    # Non-numeric object vector of the right length.
    bad_obj = np.array(["x"] * n, dtype=object)
    with pytest.raises((ValueError, TypeError)):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=bad_obj, precip=f["precip"])
    assert dc._spatial_monthly._max_count_ever == 0
    assert dc.monthly_accum._max_count_ever == 0

    # Non-finite (NaN) latitude.
    bad_nan = f["lat_deg"].copy()
    bad_nan[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=bad_nan, precip=f["precip"])
    assert dc._spatial_monthly._max_count_ever == 0
    assert dc.monthly_accum._max_count_ever == 0


# ---------------------------------------------------------------------------
# Clean-completion CMOR NetCDF write
# ---------------------------------------------------------------------------

def test_finalize_writes_cmor_netcdf(mesh, tmp_path):
    """The CMIP-file writers emit non-empty NetCDF from the fed accumulators
    (the clean-completion output the lean MPAS loop previously skipped)."""
    pytest.importorskip("netCDF4")
    dc, sigma_full, _ = _make_collector(mesh, output_dir=str(tmp_path))
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)
    # A handful of daily samples in one month so finalize keeps it.
    for d in (10.0, 11.0, 12.0):
        dc.feed_cmip_accumulators_native(
            day=d, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], q_v=f["q_v"],
            u_east=np.asarray(u_east), v_north=np.asarray(v_north),
            precip=f["precip"], phis=f["phis"])
    dc._write_cmip_monthly_files()
    dc._write_cmip_daily_files()
    dc._write_cmip_fixed_files()
    dc.cf_writer.close()
    ncs = list((tmp_path / "cmor").rglob("*.nc"))
    assert ncs, "no CMOR NetCDF written"
    assert all(p.stat().st_size > 0 for p in ncs)
    # tas (a core Amon var) must be among the emitted files.
    assert any("tas" in p.name for p in ncs)


def test_driver_finalize_mpas_cmip(mesh, tmp_path):
    """`ModelDriver._finalize_mpas_cmip` writes the CMOR NetCDF and marks the
    terminal sidecar suppressed (mirrors the compiled finalizer)."""
    pytest.importorskip("netCDF4")
    from legoesm.driver.model_driver import ModelDriver
    dc, sigma_full, _ = _make_collector(mesh, output_dir=str(tmp_path))
    f = _synthetic_cell_fields(mesh, sigma_full)
    for d in (10.0, 11.0):
        dc.feed_cmip_accumulators_native(
            day=d, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], precip=f["precip"])
    fake = types.SimpleNamespace(diagnostics=dc, _mpi_rank=None)
    ModelDriver._finalize_mpas_cmip(fake)
    assert fake._suppress_cmor_sidecar is True
    ncs = list((tmp_path / "cmor").rglob("*.nc"))
    assert ncs and all(p.stat().st_size > 0 for p in ncs)


def test_driver_finalize_retires_exact_cadence_sidecar(mesh, tmp_path):
    """Exact-checkpoint-cadence completion: a same-day sidecar already written
    by the in-loop periodic checkpoint is RETIRED after a successful finalize
    (else a run-extending restart would restore + re-append it)."""
    pytest.importorskip("netCDF4")
    from legoesm.driver.model_driver import ModelDriver
    dc, sigma_full, _ = _make_collector(mesh, output_dir=str(tmp_path))
    f = _synthetic_cell_fields(mesh, sigma_full)
    dc.feed_cmip_accumulators_native(
        day=25.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], precip=f["precip"])
    # Simulate the periodic checkpoint's same-day sidecar (day 25 -> 0025).
    stale = tmp_path / "cmor_accum_day_0025.npz"
    dc.save_cmor_accumulators(stale)
    assert stale.exists()

    fake = types.SimpleNamespace(
        diagnostics=dc, _mpi_rank=None, _output_dir=tmp_path)
    ModelDriver._finalize_mpas_cmip(fake, final_day=25.0)
    assert fake._suppress_cmor_sidecar is True
    assert not stale.exists(), "terminal sidecar was not retired"
    ncs = list((tmp_path / "cmor").rglob("*.nc"))
    assert ncs and all(p.stat().st_size > 0 for p in ncs)


def test_driver_finalize_mpas_cmip_noop_without_writer(mesh):
    """No CMIP writer (monthly-means-only) -> finalize is a guarded no-op."""
    from legoesm.driver.model_driver import ModelDriver
    dc, _, _ = _make_collector(mesh, monthly_means=True, cmip_output=False)
    assert dc.cf_writer is None
    fake = types.SimpleNamespace(diagnostics=dc, _mpi_rank=None)
    ModelDriver._finalize_mpas_cmip(fake)          # must not raise
    assert getattr(fake, "_suppress_cmor_sidecar", False) is False

```

## APPENDIX B: round-3 review (verbatim)

Verdict: not clean. The CMIP-only writer choice is correct, but two correctness holes remain.

- **HIGH — exact-cadence completion leaves the terminal CMOR sidecar in place.** The periodic checkpoint runs first when the final step is a checkpoint multiple ([model_driver.py:6405](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6405)) and MPAS checkpointing writes the sidecar immediately ([model_driver.py:4105](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:4105)). Only afterward does the new finalizer set suppression ([model_driver.py:6435](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6435), [model_driver.py:5425](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5425)). The purported final checkpoint is explicitly skipped in that exact-multiple case ([model_driver.py:6443](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6443)). A restart restores that same-day sidecar ([run_amip.py:2528](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/run/run_amip.py:2528)), and the writers did not pop it, so an extension can append duplicate times. Suppression only prevents a *future* sidecar write; it does not remove the one already written.

  Defer the final periodic checkpoint until after finalization, or explicitly retire/replace the same-day sidecar after successful finalization. Add an exact-checkpoint-cadence integration/state-machine test; the present test calls only the helper ([test_mpas_cmip_accumulator_feed.py:575](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_cmip_accumulator_feed.py:575)).

- **MEDIUM — the feed is still not transactional for a shape-correct, non-numeric `lat_deg`.** Phase 1 checks only shape ([diagnostics.py:1461](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1461), [diagnostics.py:1470](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1470)). A `(nCells,)` object/string vector passes it. In `MonthlyAccumulator.add_2d`, the bucket and call count mutate before `np.digitize` processes latitude ([monthly_means.py:120](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/diagnostics/monthly_means.py:120), [monthly_means.py:128](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/diagnostics/monthly_means.py:128)). That operation can raise for non-numeric objects, after the spatial commits at [diagnostics.py:1491](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1491).

  Convert/validate `lat_deg` to a numeric floating array in Phase 1 (and ideally validate finite values), then use that converted array for commit. Also validate every zonal field’s horizontal shape: zonal `add_2d` has no shape guard before `np.add.at` ([monthly_means.py:132](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/diagnostics/monthly_means.py:132)); this matters in the supported zonal-only configuration. The added test covers only a length mismatch ([test_mpas_cmip_accumulator_feed.py:530](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_cmip_accumulator_feed.py:530)).

Also, the “rerun writer or resume” claim is unsafe after a partial I/O failure: the new finalizer catches an exception after sequential writers ([model_driver.py:5421](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5421)), while individual fields may already have been appended ([diagnostics.py:1919](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1919), [cmor_output.py:1667](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/io/cmor_output.py:1667)). Retaining the complete sidecar then makes a retry append duplicates. That needs transactional staging or a durable per-write progress record.

On the positive side, calling only the three CMIP writers plus `close()` is the right subset of `save()` ([diagnostics.py:1785](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1785)); it avoids clobbering MPAS’s separate lightweight timeseries. On an ordinary successful completion that is not an exact checkpoint multiple, the ordering works, and the wallclock pop logic avoids a new normal-path double write.
