"""Auto-dt heuristic for the cross-grid RCE driver.

Single source of truth for the per-grid-type, per-resolution dt
ladder used by ``scripts/run/run_rce.py``. Lives in
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

Empirical scaling (iter-28 fit on the cubed_sphere ladder)
----------------------------------------------------------
Fitting ``ln(dt) = α · ln(dx) + c`` over C24, C48, C72, C96 yields
α ≈ 2.0 — i.e. ``dt ∝ dx²`` rather than the linear ``dt ∝ dx``
expected from advective CFL. The destabilising mode in our RCE
setup is consistent with a *diffusive* CFL scaling, which matches
the inverse-square scaling observed in the empirical ladder. This
is the structural reason the iter-13 inverse-linear extrapolation
(``dt=75`` at C96) was too loose: linear-CFL undershoots, diffusive-
CFL gets the right scaling.

The empirical ladder is preserved because (a) the iter-12..26
measurements explicitly bracket each branch with PASS/BLOWUP
evidence at the boundary, and (b) wrapping in a formula would
hide the per-measurement provenance. The reference function
:func:`empirical_dt_dx2` is provided for diagnostic use only.
"""
from __future__ import annotations


# Empirical anchor (iter-12 C24 measurement): dt=600 s at dx≈240753 m.
# Other points (C48 dt=150, C72 dt=75, C96 dt=37) all sit on the same
# dt ∝ dx² curve within ±15 %.
_DT_DX2_K = 600.0 / (240753.0 ** 2)  # s / m² ≈ 1.04e-8


def empirical_dt_dx2(dx_min: float) -> float:
    """Reference dt from the iter-28 empirical ``dt ∝ dx²`` fit.

    Diagnostic only. The actual production picks come from
    :func:`auto_dt_rce` so each branch is anchored to a specific
    measurement (iter-12 .. iter-26) rather than an extrapolation.
    Use this function to compare an alternative resolution against
    the ladder before adding it.

    Parameters
    ----------
    dx_min : float
        Minimum grid spacing [m]. For cubed_sphere use
        :func:`legoesm.core.cfl.estimate_min_dx_cubed_sphere(N)`.

    Returns
    -------
    dt : float
        Recommended dt [s] from the empirical fit.
    """
    return _DT_DX2_K * dx_min ** 2


# Voronoi-family unstructured grids all use the voronoi-safe dt: MPAS uses an
# SCVT Voronoi mesh, and the icosahedral grid shares the conservative 300 s
# default pending its own 30-day measurement. The structured resolution ladder
# in ``auto_dt_rce`` is only validated for the quasi-uniform ladder grids below.
_VORONOI_LIKE_GRIDS = ("voronoi", "mpas", "mpas_voronoi", "icosahedral")
_LADDER_GRIDS = ("cubed_sphere", "gaussian", "latlon")


def auto_dt_rce(grid_type: str, resolution: int) -> float:
    """Return the empirically-measured stable outer dt [s] for the
    cross-grid RCE driver.

    Parameters
    ----------
    grid_type : str
        One of the ladder grids ``{"cubed_sphere", "gaussian", "latlon"}`` or
        a voronoi-family grid ``{"voronoi", "mpas", "mpas_voronoi",
        "icosahedral"}`` (all map to the voronoi-safe dt). Any other value
        raises ``ValueError`` rather than silently using the ladder.
    resolution : int
        N for cubed_sphere (CXX), N_max for spectral (TXX), n_lat for
        lat-lon (LLXX), level for voronoi/MPAS (VXX).

    Returns
    -------
    dt : float
        Recommended outer timestep in seconds. Pass via ``--dt`` to
        ``scripts/run/run_rce.py`` (or read from
        ``scripts.run.run_rce.main``'s auto-dt branch, which calls this).

    Raises
    ------
    ValueError
        For an unknown ``grid_type`` (not a ladder or voronoi-family grid), or
        for a ladder grid (``cubed_sphere`` / ``gaussian`` / ``latlon``) at
        ``resolution > 96``. The iter-13 / iter-20 measurements showed that
        extrapolating the ladder past the last measured boundary is unsafe
        (C96 BLOWUP at the iter-13-extrapolated dt=75). High-resolution runs
        must pass an explicit ``--dt`` and update both this function + the
        regression test once a 30-day measurement lands.
    """
    if grid_type in _VORONOI_LIKE_GRIDS:
        return 300.0
    if grid_type not in _LADDER_GRIDS:
        raise ValueError(
            f"auto_dt_rce: unknown grid_type {grid_type!r}; expected one of "
            f"{_LADDER_GRIDS + _VORONOI_LIKE_GRIDS}. The structured resolution "
            "ladder is only validated for the quasi-uniform ladder grids; "
            "unstructured grids must map to the voronoi-safe branch."
        )
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
