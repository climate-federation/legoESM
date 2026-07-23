# Adversarial review ROUND 3: MPAS monthly-CMOR accumulator feed

You are an independent adversarial reviewer. This is round 3. Rounds 1–2
already validated the FEED itself (tas 2 m, gate/warning, 1-rank permit, unit
conventions, shapes). Round 2 left exactly two items; this round addresses both.
Re-review ONLY whether these two fixes are correct and safe, and whether they
introduce any NEW bug. Cite line numbers.

## Round-2 findings and my disposition

**#1 (HIGH) — a normally completed MPAS run did not write the populated CMOR
accumulators to NetCDF.** FIXED. `_run_mpas` bypasses the shared
`_finalize_run` (whose `diagnostics.save()` ALSO writes the cube-path
`timeseries.npz` + snapshots this lean loop tracks separately in `_ts`, so
calling the full `save()` would clobber them). So I added `_finalize_mpas_cmip`,
which calls ONLY the CMIP-file writers (`_write_cmip_monthly_files` /
`_write_cmip_daily_files` / `_write_cmip_fixed_files` + `cf_writer.close()`) —
the exact CMIP subset of `save()`. It runs on a CLEAN completion (`run_status ==
"COMPLETED"`), rank-0 only, BEFORE the final checkpoint, and on success sets
`self._suppress_cmor_sidecar = True` so the terminal checkpoint writes no CMOR
sidecar (the writers do not pop; a run-extending restart would else re-append —
duplicate `time` coords). This mirrors `_finalize_run` exactly (which sets
`_suppress_cmor_sidecar` around its final checkpoint). Wallclock-graceful exits
already flush+POP completed months incrementally (`_maybe_wallclock_exit`), so
the final link only writes the un-popped remainder — no double-write.
Restart-chain reasoning: an intermediate link wallclock-exits (never reaches
this clean-completion path); only the FINAL/standalone link completes cleanly →
writes the remainder. Tests: `test_finalize_writes_cmor_netcdf` (emits non-empty
`cmor/*.nc` incl. `tas`), `test_driver_finalize_mpas_cmip` (writes + sets the
suppress flag), `test_driver_finalize_mpas_cmip_noop_without_writer`.

**#2 (MEDIUM) — the feed was not a true transaction.** FIXED. PHASE 1 was
already non-mutating for regrid/interp, but the zonal `add_*` (via
`np.digitize`/`np.add.at`) could raise on a malformed `lat_deg` AFTER the
spatial `add_*` had committed. I now validate `lat_deg.shape == (nCells,)` in
PHASE 1, before any commit. The spatial fields are shape-guaranteed by
`_regrid_to_latlon_*` (always `(nlat,nlon)`/`(nlat,nlon,nlev)` or `None`), so
`add_2d`/`add_3d` cannot raise on shape; with `lat_deg` pre-validated, PHASE 2
has no remaining commit-time failure mode. The driver-level `try/except` stays
loud-but-nonfatal (matches `_save_cmor_accumulator_sidecar`; a 100-yr run must
not abort on a diagnostic glitch). Test: `test_feed_atomic_on_bad_lat_deg`
(bad `lat_deg` -> `ValueError` in PHASE 1, all three accumulators untouched).

## Key facts for your reasoning

- `_write_cmip_monthly_files()` calls `self._spatial_monthly.finalize()`
  (default `min_sample_fraction=0.5`, drops partial months < 50% of max
  samples) then `_write_cmip_data`; it does NOT pop.
- `_maybe_wallclock_exit` (unchanged): `flush_cmip_monthly(day, write=True)`
  POPS completed months, `finalize_cmip_daily`, `finalize_cmip_fixed`,
  re-persists the drained sidecar, `sys.exit(0)`.
- `_finalize_run` (compiled path, unchanged): `diagnostics.save()` then sets
  `_suppress_cmor_sidecar=True` around the final checkpoint.
- `_save_cmor_accumulator_sidecar` early-returns when `_suppress_cmor_sidecar`.
- MPAS restart contract: each link runs `--days = remaining`; it either
  wallclock-exits (long chain) or completes cleanly (final/standalone link).
- The MPAS final checkpoint block runs after the loop:
  `if CHECKPOINT_INTERVAL>0 and COMPLETED and n_steps_total%INTERVAL!=0:
   save_checkpoint(...)`.

## THE INCREMENTAL CHANGES (round 3)

### ROUND-3 incremental changes (on top of round-2, which you already blessed)

#### diagnostics.py :: PHASE-1 lat_deg validation (atomicity, 1455-1476)
```python
                        daily_2d[_dst] = fields_3d[_src_name][:, :, _idx850]

        z2d: dict[str, np.ndarray] = {}
        z3d: dict[str, np.ndarray] = {}
        lat_np = None
        if have_zonal:
            lat_np = np.asarray(lat_deg)
            # Validate the zonal lat vector HERE, in PHASE 1, before any commit:
            # the zonal add_* bin via ``np.digitize(lat, ...)`` and ``np.add.at``
            # and would raise mid-commit on a length mismatch — AFTER the spatial
            # add_* already mutated their buckets.  Catching it pre-commit keeps
            # the whole feed transactional (the spatial fields are shape-
            # guaranteed by ``_regrid_to_latlon_*``, so this is the only
            # remaining commit-time failure mode).
            _ncol = int(T_np.shape[0])
            if lat_np.shape != (_ncol,):
                raise ValueError(
                    "feed_cmip_accumulators_native: lat_deg shape "
                    f"{lat_np.shape} != expected ({_ncol},)")
            z2d['T_low'] = np.asarray(T_low)   # lowest model level, NOT 2 m tas
            if precip_np is not None:
                z2d['precip'] = precip_np * 86400.0     # [mm/day], like collect()
```
#### model_driver.py :: _finalize_mpas_cmip (5390-5431)
```python
    def _finalize_mpas_cmip(self) -> None:
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

        The writers do NOT pop, so this is TERMINAL: on success it sets
        ``_suppress_cmor_sidecar`` so the final checkpoint carries no CMOR
        sidecar (a completed run has no in-progress month to resume, and a
        run-EXTENDING restart would otherwise re-append the already-written
        months — duplicate ``time`` coords).  Mirrors :meth:`_finalize_run`.
        Rank-0 only; fully guarded (a writer failure must not turn a COMPLETED
        run into a crash after the science is done — the sidecar/accumulators
        remain intact for a manual rewrite)."""
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
            self._suppress_cmor_sidecar = True
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.error(
                "  MPAS CMOR NetCDF finalize FAILED (the accumulators were fed "
                "and the sidecar is intact — rerun the writer or resume): %s",
                exc)

```
#### model_driver.py :: clean-completion finalize call in _run_mpas (6429-6446)
```python

        # Write the CMOR NetCDF from the (now-fed) accumulators on a CLEAN
        # completion.  MUST run BEFORE the final checkpoint below so its
        # ``_suppress_cmor_sidecar`` (set on a successful write) reaches the
        # terminal sidecar.  Gated on the feed being active (serial / 1-rank
        # with CMIP output); a no-op otherwise.
        if run_status == "COMPLETED" and self._mpas_cmip_feed_on:
            self._finalize_mpas_cmip()

        # Final checkpoint so the next chain link resumes from the exact end
        # state.  Skipped (a) on blow-up — state is non-finite — and (b) when
        # the last loop step already hit the periodic cadence, which would
        # re-write the identical file (wasted device→host transfer + I/O
        # every whole-multiple job boundary).
        if (CHECKPOINT_INTERVAL > 0 and run_status == "COMPLETED"
                and n_steps_total % CHECKPOINT_INTERVAL != 0):
            _final_day = START_DAY + n_steps_total * DT / 86400.0
            self.save_checkpoint(start_step + n_steps_total, _final_day)
```


## The updated tests (full file)

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


def test_driver_finalize_mpas_cmip_noop_without_writer(mesh):
    """No CMIP writer (monthly-means-only) -> finalize is a guarded no-op."""
    from legoesm.driver.model_driver import ModelDriver
    dc, _, _ = _make_collector(mesh, monthly_means=True, cmip_output=False)
    assert dc.cf_writer is None
    fake = types.SimpleNamespace(diagnostics=dc, _mpi_rank=None)
    ModelDriver._finalize_mpas_cmip(fake)          # must not raise
    assert getattr(fake, "_suppress_cmor_sidecar", False) is False

```

## Note on the test suite

One PRE-EXISTING unrelated test fails on this tree and on committed HEAD:
`test_cmor_experiments_restart.py::test_wallclock_exit_writes_cmor_fx_before_exit`
— its `SimpleNamespace` fake lacks `_save_cmor_accumulator_sidecar`, which the
COMMITTED `_maybe_wallclock_exit` calls (HEAD and working tree are byte-
identical for that method; my diff does not touch it). Not introduced here. All
21 tests in the new file pass; other CMOR/collector/regrid/constant suites pass.

## Your task

Re-review ONLY the two round-3 fixes. Specifically: (a) Is calling the CMIP-only
writers (not full `save()`) correct, and does `_suppress_cmor_sidecar` ordering
w.r.t. the final checkpoint actually prevent the terminal sidecar? (b) Any
double-write / partial-month / restart-chain hazard I mis-reasoned? (c) Is the
`lat_deg` PHASE-1 validation sufficient to make the commit transactional, or is
there another commit-time failure mode? (d) Any NEW bug in the added code or
tests? If both fixes are correct and no substantive bugs remain, say so
explicitly and state the review is clean.


## APPENDIX: round-2 review (verbatim)

Verdict: not ready for end-to-end CMOR output. The `tas` and MPI changes are sound, but two substantive issues remain.

1. **High — a normally completed MPAS run still does not write the populated CMOR accumulators.** `_run_mpas()` ends by writing only the lightweight timeseries and returning ([model_driver.py:6398](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6398)). CMOR NetCDF writing occurs in `DiagnosticCollector.save()` ([diagnostics.py:1771](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1771)), which is called by the shared compiled-path finalizer ([model_driver.py:8678](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:8678)), but MPAS bypasses that finalizer ([model_driver.py:5151](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5151)). The only MPAS CMOR flush is the special wall-clock-exit path ([model_driver.py:5028](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5028)). Thus a clean normal completion can retain correct in-memory/sidecar accumulators yet produce no CMOR files. None of the new tests exercises normal MPAS finalization or asserts emitted CMOR files.

2. **Medium — the feed is not atomic as documented.** Phase 1 is correctly non-mutating, but Phase 2 commits each accumulator sequentially ([diagnostics.py:1475](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1475)). The accumulator methods mutate bucket/count state before validating all inputs ([monthly_means.py:481](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/diagnostics/monthly_means.py:481), [monthly_means.py:809](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/diagnostics/monthly_means.py:809)); zonal inputs receive no shape validation before commit ([diagnostics.py:1458](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1458)). For example, a malformed `lat_deg` can commit spatial monthly/daily data, then fail in zonal `np.add.at`; the outer handler logs and continues. This is much better against regrid/interpolation failures, but not a transaction. Add an error-path test asserting all three accumulator states are unchanged after a Phase-1 failure and after an invalid zonal shape.

The claimed `tas` fix itself checks out. The cube call remains unchanged ([diagnostics.py:912](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:912)); with default arguments, the refactor binds the exact former `state.u/v[..., -1]` values before identical arithmetic ([diagnostics.py:628](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:628)). The MPAS path reconstructs cell winds and supplies them as overrides ([model_driver.py:5332](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5332), [model_driver.py:5360](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5360)); SST/SIC flattening and `T_ice` forwarding are correct, and the inner fallback is scoped correctly to MOST failure.

The one-rank MPI permit is safe: a one-rank partition owns every cell, and halo construction adds only non-owned neighbours ([voronoi_partition.py:383](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_partition.py:383)). The gate and rank-0 multi-rank warning are correct ([model_driver.py:5285](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5285), [model_driver.py:5460](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5460)). Minor doc nit: `_feed_mpas_cmip_accumulators` still says “SERIAL only” despite now permitting one-rank distributed layouts.

The inherited items are accurately characterized: plev extrapolation uses the same helper as `collect()`, and zonal bands are indeed arithmetic cell means. The snapshot/`pr` limitation remains scientifically material but is clearly documented, as allowed. The new physical tests are non-vacuous for `tas`, `psl`, 850 hPa winds, calendar buckets, and restoration; they still lack atomicity and real MPAS-to-CMOR-output coverage.

I could not independently rerun pytest in this read-only review environment, so I do not independently confirm the stated pass count.
