"""Auto-dt heuristic for the cross-grid RCE driver.

Single source of truth for the per-grid-type, per-resolution dt
ladder used by ``scripts/run_rce.py``. Lives in
``legoesm.driver`` (not in the script) so:

  1. The cross-grid test suite can import + call the EXACT same
     function the production driver uses, instead of maintaining
     a hand-copied mirror that can silently diverge (Codex iter-19
     LOW finding).
  2. Other drivers (long-run wrappers, batch sweeps) can reuse
     the heuristic without copy-paste.

Empirical lineage (full table in CRM_implementation.md iter-12..26):

  voronoi/MPAS V4:   dt=300 PASS (30-day, max|v|=2.3 m/s, iter-12)
                     dt>=450 BLOWUP at day 1 (iter-8)
  cubed_sphere C24:  dt=600 PASS (30-day, max|v|=7.2 m/s, iter-12)
  cubed_sphere C48:  dt=300 BLOWUP at day 25 (max|v|=236 m/s, iter-12)
                     dt=150 PASS 30-day (verified iter-13/15)
  cubed_sphere C72:  dt=75  PASS 30-day (max|v|=17.9 m/s, iter-26)
  cubed_sphere C96:  dt=75  BLOWUP at day 20 (max|v|=527 m/s, iter-20)
                     dt=37  PASS at 10-day (iter-22); 30-day in flight
  gaussian T21:      dt=600 PASS (30-day, iter-12)

iter-21: N>96 RAISES (silent extrapolation hid the iter-20 mistake).
"""
from __future__ import annotations


def auto_dt_rce(grid_type: str, resolution: int) -> float:
    """Return the empirically-measured stable outer dt [s] for the
    cross-grid RCE driver.

    Parameters
    ----------
    grid_type : {"cubed_sphere", "gaussian", "latlon", "voronoi"}
    resolution : int
        N for cubed_sphere (CXX), N_max for spectral (TXX), n_lat for
        lat-lon (LLXX), level for voronoi/MPAS (VXX).

    Returns
    -------
    dt : float
        Recommended outer timestep in seconds. Pass via ``--dt`` to
        ``scripts/run_rce.py`` (or read from
        ``scripts.run_rce.main``'s auto-dt branch, which calls this).

    Raises
    ------
    ValueError
        For ``cubed_sphere`` / ``gaussian`` / ``latlon`` at
        ``resolution > 96``. The iter-13 / iter-20 measurements
        showed that extrapolating the ladder past the last measured
        boundary is unsafe (C96 BLOWUP at the iter-13-extrapolated
        dt=75). High-resolution runs must pass an explicit ``--dt``
        and update both this function + the regression test once a
        30-day measurement lands.
    """
    if grid_type == "voronoi":
        return 300.0
    if resolution <= 24:
        return 600.0
    if resolution <= 48:
        return 150.0
    if resolution <= 72:
        return 75.0
    if resolution <= 96:
        return 37.0
    raise ValueError(
        f"auto-dt has no validated value for N={resolution} (>96). "
        "The iter-13/iter-20 ladder past N=48 was already shown to "
        f"over-extrapolate (C96 BLOWUP at iter-13 dt=75). To run at "
        f"N={resolution}, pass an explicit --dt (start with dt=10 "
        "and watch the BLOWUP gate at 200 m/s), then update the "
        "ladder + tests after a 30-day stability measurement."
    )
