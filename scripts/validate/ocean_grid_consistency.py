"""Cross-grid consistency + theory check for the ocean benchmark suite.

PASS/FAIL says a run stayed inside its own gates. It does NOT say the
dycores agree with EACH OTHER, nor that any of them reproduces the
analytic answer. This script asks those two questions of the artifacts
the matrix already writes:

  CONSISTENCY  arms are NOT co-located as saved (the matrix leaves latlon
               native and regrids unstructured arms), so each is
               NEAREST-NEIGHBOUR sampled onto one common 2-degree mesh and
               compared cell-by-cell: pairwise area-weighted RMS
               difference in the FIELD'S OWN UNITS (no normalisation --
               a normalised column was tried and removed, see
               cross_grid_rms).
               The tripole arm is EXCLUDED by default -- it keeps NEMO's
               continents (wet fraction 0.53 native vs 0.89), so a
               cell-by-cell comparison against the land-free arms measures
               the land mask, not the dycore.

  THEORY       where the case has a closed-form answer, the measured
               number is printed next to it:
                 * barotropic_wave       phase speed  c = sqrt(g H)
                 * lock_exchange         front speed vs Benjamin (1968)
                                         c = 0.5 sqrt(g' H) -- reported,
                                         with a refinement study
                                         (--convergence) showing the
                                         deficit shrinking but NOT a
                                         demonstrated limit; the shipped
                                         global config is far too coarse
                 * rest_state_*          exact: |u| = 0, d(eta) = 0
               Cases WITHOUT a defensible closed form on this geometry
               (geostrophic_adjustment, phillips_two_layer,
               inertia_gravity_wave) are reported as consistency-only and
               say so, rather than being compared to a formula that does
               not apply -- the mistake the IGW gate made.

  SELF-ERROR   ``--self-error``: the tolerance the CONSISTENCY block is read
               against, MEASURED rather than chosen. One arm is run against
               ITSELF at twice the resolution through the SAME matrix
               entry point (only ``--resolution`` differs), and the
               difference is reduced with the SAME common mesh, the SAME
               land erosion and the SAME area weights as an arm-to-arm
               pair. Two independent discretisations cannot agree more
               closely than one agrees with itself, so that number is the
               floor and therefore the standard.

Run AFTER the suite. Default mode reads only saved artifacts; ``--self-error``
and ``--convergence`` are the two modes that run the model.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from legoesm import constants as C

_REPO = Path(__file__).resolve().parents[2]

#: Arms compared cell-by-cell: the AQUAPLANET ones only.
#:
#: tripole and fesom are both excluded, for the same reason: they solve a
#: DIFFERENT BASIN. tripole keeps NEMO's continents. fesom's packaged "pi"
#: mesh is a real-geometry global ocean mesh too -- MEASURED 2026-08-11,
#: the distance from a lat-lon target to the nearest mesh node is 2695 km
#: over central Asia, 1808 km over the Sahara, 1410 km over the Amazon,
#: 891 km over Australia and 941 km over Antarctica, against 144 km in the
#: north Atlantic, and the mesh's own ``depth`` field runs -30 m to
#: -6000 m with every node wet, i.e. it has real coastlines and real
#: bathymetry. 22% of the target mesh is more than 570 km from any node.
#:
#: The lat-lon and MPAS arms are aquaplanets whose only land is poleward of
#: 80 deg. Comparing them against fesom cell-by-cell measured the coastline,
#: not the dycore -- and until the regrid gained a target-side cutoff
#: (2026-08-11) it did not even do that, because the continents were filled
#: by extrapolation and the land mask called them ocean (wet fraction 0.94
#: against 0.89; it is 0.75 once the cutoff is applied).
#:
#: RETRACTED with this change: every previously reported fesom|latlon and
#: fesom|mpas cross-arm number, on every case. They were geometry
#: differences reported as dycore differences.
# cubed_sphere REMOVED 2026-08-11: not an ocean grid (user directive).
# Its self-error on geostrophic adjustment was 0.738 degC -- 213x the
# lat-lon reference -- so no cross-arm statement involving it was
# interpretable anyway; every such pair was already being reported as
# budget-widened or unassessable.
AQUAPLANET_GRIDS = ["latlon", "mpas"]
REAL_GEOMETRY_GRIDS = ["fesom", "tripole"]
CONSISTENCY_GRIDS = AQUAPLANET_GRIDS
ALL_GRIDS = CONSISTENCY_GRIDS + REAL_GEOMETRY_GRIDS

#: LAST-RESORT arm-to-arm threshold, in the FIELD'S OWN UNITS, used ONLY
#: for a case whose self-error has not been measured. It is the rounded-up
#: LAT-LON self-error of the first two cases measured (geostrophic
#: adjustment, barotropic wave), so applying it to a THIRD case assumes
#: that case has the same discretisation error -- an assumption with no
#: support, and the reason ``--self-error`` now measures every case
#: instead. An earlier revision used a hand-picked 1e-3 in both fields,
#: which no pair of independent discretisations can meet and which
#: therefore reported "DISAGREE" for everything.
AGREE_TOL = {"SST": 1.0e-1, "eta": 2.5e-2}

#: PER-ARM, PER-CASE self-error: that arm run against ITSELF at 2x
#: resolution, reduced through the SAME common mesh / erosion / weights as
#: a cross-arm pair, so the two numbers are directly comparable. Regenerate
#: with ``--self-error`` (which prints this literal block); the values are
#: committed so the default report needs no model runs.
#:
#: Two uses:
#:   1. the REFERENCE arm's entry sets the case tolerance (``_tolerance``),
#:      so each case is judged against its own discretisation error rather
#:      than another case's;
#:   2. an arm whose OWN self-error exceeds that tolerance is not
#:      interpretable against it -- its "disagreement" and "held to another
#:      arm's convergence rate" are indistinguishable (GLM-5.2).
#:
#: IT IS A SCALE, NOT A PROVEN FLOOR. Two schemes CAN agree more closely
#: than one agrees with its own refinement -- when they share a dominant
#: truncation error, or when both hold an exact invariant (a rest state),
#: or when both are wrong the same way (GLM-5.2 2026-08-10). So a pair
#: INSIDE the tolerance is "not distinguishable from discretisation", never
#: "verified equivalent"; only a pair OUTSIDE it is a claim.
#:
#: dt IS HELD FIXED while dx halves. Deliberate: the cross-arm pairs this
#: tolerance serves also run at one shared DEFAULT_DT with different dx, so
#: a fixed-dt refinement is the MATCHED protocol, and differencing two runs
#: at the same dt cancels the temporal truncation error and leaves the
#: spatial one -- which is what separates the arms. It does mean the two
#: runs sit at different Courant numbers; a constant-CFL refinement would
#: measure a different (combined) quantity.
#:
#: CHAOTIC CASES ARE DIFFERENT. For phillips_two_layer (baroclinic
#: instability) a 2x-resolution difference saturates at the field amplitude
#: once the eddies decorrelate, so its entry is a DECORRELATION scale, not
#: a convergence error; a pair inside it says only "these two decorrelated
#: no faster than one arm decorrelates from its own refinement".
#:
#: Mesh-file-backed arms cannot be measured this way: fesom ships only the
#: "pi" mesh and tripole only "eorca1", so neither has a 2x sibling.
#:
#: THE VALUE STORED IS THE *EVOLUTION* SELF-ERROR: the arm's
#: ``(final - initial)`` at one resolution against the same thing at 2x,
#: i.e. exactly the quantity the cross-arm verdict is taken on. Storing
#: the raw final-field difference instead made the two sides different
#: quantities, so a pair could be flagged IC-dominated and outside budget
#: in the same breath. The raw number is printed beside it by
#: ``--self-error``; for the two cases whose initial condition is a step
#: or a sharp bump the two differ by more than an order of magnitude
#: (lat-lon lock exchange: raw 2.613 degC, evolution 4.4e-2 degC).
#:
#: MEASURED 2026-08-10 through ``--self-error`` (the block it prints).
#: Every entry comes from two suite invocations differing only in
#: ``--resolution``; there are no hand-entered numbers left here.
#:
#: These REPLACE four values obtained interactively earlier the same day
#: (latlon geostrophic 8.20e-2 / bwave 2.34e-2, cube 9.45e-1 / 2.23e-2).
#: On the RAW quantity this path reproduces the barotropic-wave pair to
#: ~4% and comes out 20-25% lower on geostrophic adjustment; the earlier
#: protocol was not recorded, so that gap cannot be attributed -- which is
#: the whole reason the measurement is now an argument to this script
#: instead of a session note.
#:
#: Several arms FAIL their own case gate (cube geostrophic and Phillips,
#: every barotropic-wave arm). That is a pre-existing statement about the
#: physics, not about the snapshot, so the self-error is still measured
#: from them and ``--self-error`` prints a NOTE naming each one.
SELF_ERROR = {
    # cubed_sphere entries kept as the RECORD of why it was dropped from
    # the ocean matrix (its geostrophic self-error is 213x the reference);
    # the arm is no longer run, so nothing reads them.
    #
    # RE-MEASURED 2026-08-13 (job 26917825) after the split-explicit
    # barotropic averaging window was re-centred on t+dt. That fix changes
    # how every arm carries a gravity wave, so the pre-fix numbers were
    # measurements of a different model.
    ("cubed_sphere", "barotropic_wave"): 2.370e-2,
    ("latlon", "barotropic_wave"): 2.300e-2,
    ("mpas", "barotropic_wave"): 1.777e-2,
    ("cubed_sphere", "geostrophic_adjustment"): 7.381e-1,
    ("latlon", "geostrophic_adjustment"): 3.208e-3,
    ("mpas", "geostrophic_adjustment"): 3.299e-3,
    ("cubed_sphere", "inertia_gravity_wave"): 2.336e-1,
    ("latlon", "inertia_gravity_wave"): 2.380e-1,
    ("mpas", "inertia_gravity_wave"): 2.095e-1,
    # Chaotic case -- a decorrelation scale, not a convergence error. The
    # MPAS entry GREW 2.0x with the re-centred window (5.817e-2 ->
    # 1.172e-1), i.e. this arm's answer is now MORE sensitive to
    # resolution than before, and the tolerance it sets is looser. That is
    # the direction --refinement exists to catch, and it does: with a
    # refined-level budget the cross-arm difference goes from 0.20 of the
    # budget at 36x72/ico4 to 1.08 at 72x144/ico5.
    ("cubed_sphere", "phillips_two_layer"): 4.222e-2,
    ("latlon", "phillips_two_layer"): 8.591e-2,
    ("mpas", "phillips_two_layer"): 1.172e-1,
    # Discontinuity -- kept for completeness; the lock exchange's verdict
    # comes from FRONT displacement (RMS_NOT_GATED / FRONT_SELF_ERROR_KM).
    ("latlon", "lock_exchange"): 4.379e-2,
    ("mpas", "lock_exchange"): 4.954e-2,
}

#: Cases whose FIELD RMS is reported but NOT turned into a verdict, with
#: the reason and the metric that IS gated. The lock exchange is a
#: discontinuity: its RMS is dominated by where each mesh puts the step
#: (MEASURED -- the lat-lon self-error is 2.61 degC raw and 0.039 degC
#: after removing a 2-degree rigid shift), so gating on it would be gating
#: on mesh phase. Reporting an "agree/disagree" from a metric that was
#: just shown not to adjudicate the case is the defect codex flagged
#: 2026-08-10.
RMS_NOT_GATED = {
    "lock_exchange": ("a temperature step: the RMS measures where each mesh "
                      "puts the front, not the dycores -- gated on front "
                      "DISPLACEMENT below instead"),
}

#: Per-arm front-DISPLACEMENT self-error [km], same 2x-resolution protocol,
#: max over the case's two fronts. This is the tolerance for the
#: front-position verdict, and ``--self-error`` prints a paste block for
#: it just as it does for SELF_ERROR. MEASURED 2026-08-10.
#:
#: Note the scale separation that makes the metric worth having: the same
#: pair of runs differs by 115 km (lat-lon) / 223 km (MPAS) in where the
#: front SITS -- pure mesh phase -- and by ~1-3 km in how far it MOVED.
FRONT_SELF_ERROR_KM = {
    # RE-MEASURED 2026-08-13 (job 26917825), same re-centred-window rerun.
    ("latlon", "lock_exchange"): 0.962,
    ("mpas", "lock_exchange"): 1.88,
}

#: Arm whose self-error defines each case's tolerance. lat-lon is the
#: reference because it is the only arm registered for every case in
#: CASE_FIELD and because its C-grid is the suite's baseline
#: discretisation.
TOLERANCE_ARM = "latlon"


def _tolerance(case: str) -> tuple[float, str]:
    """Agreement tolerance for a case, and where it came from.

    MEASURED (the reference arm's own self-error at 2x resolution) when
    that measurement exists; otherwise the field-level fallback, which is
    another case's number and is labelled as such at every print site.
    """
    key = (TOLERANCE_ARM, case)
    if key in SELF_ERROR:
        return SELF_ERROR[key], f"measured {TOLERANCE_ARM} self-error"
    return AGREE_TOL[CASE_FIELD[case]], (
        f"FALLBACK {CASE_FIELD[case]}-field default -- this case's "
        f"self-error is UNMEASURED, run --self-error")

#: field used for the cross-grid comparison, per case
CASE_FIELD = {
    "rest_state_stratified_with_land": "SST",
    "rest_state_uniform_with_land": "SST",
    "rest_state_stratified_no_land": "SST",
    "rest_state_uniform_no_land": "SST",
    "barotropic_wave": "eta",
    "geostrophic_adjustment": "SST",
    "phillips_two_layer": "eta",
    "inertia_gravity_wave": "eta",
    "lock_exchange": "SST",
}


def _registered_resolution(case: str, grid: str) -> str | None:
    """Resolution the CURRENT matrix registers for THIS (case, grid).

    Read from ``_build_test_matrix()`` itself, not from GRID_RESOLUTIONS:
    several cases override the per-grid default (barotropic_wave runs
    latlon at 48x72), so the global table is not the authority.
    """
    import importlib.util
    import sys
    if "_rm_reg" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "_rm_reg", _REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_rm_reg"] = mod
        spec.loader.exec_module(mod)
    rm = sys.modules["_rm_reg"]
    if not hasattr(rm, "_REGISTERED_RES"):
        rm._REGISTERED_RES = {(tc.case, tc.grid_type): tc.resolution
                              for tc in rm._build_test_matrix()}
    return rm._REGISTERED_RES.get((case, grid))


def _find(root: Path, case: str, grid: str) -> Path | None:
    """Locate an arm's snapshot, REFUSING artifacts from a stale run.

    results/ is written in place: a suite rerun overwrites only the arms it
    ran, so directories from earlier configurations survive (mpas/ico3 after
    the move to ico4; fesom no-land runs after that pair was unregistered).
    Reading those silently mixes runs -- it made an earlier revision of this
    script report six pairs for cases that now have three, and quote an
    ico3 number as if it were the current arm (codex 2026-08-10). Only the
    resolution the matrix registers TODAY is accepted; anything else is
    reported so it can be deleted rather than used.
    """
    want = _registered_resolution(case, grid)
    hits = [h for h in root.glob(f"{case}/**/{grid}/*/snapshots_latlon.npz")
            if h.parent.parent.name == grid]
    if not hits:
        return None
    fresh = [h for h in hits if want is not None and h.parent.name == want]
    if not fresh:
        stale = sorted({h.parent.name for h in hits})
        print(f"    [stale artifact ignored] {case}/{grid}: found {stale}, "
              f"current registry says {want!r} -- rerun or delete")
        return None
    return fresh[0]


def _load(npz: Path, field: str):
    z = np.load(npz)
    a = np.asarray(z[field], dtype=np.float64)
    m = np.asarray(z["land_mask"], dtype=np.float64)
    m = m[-1] if m.ndim == 3 else m
    return (np.asarray(z["lat"]), np.asarray(z["lon"]),
            a, m, np.asarray(z["times_days"], dtype=np.float64))


def _to_common_mesh(lat, lon, a, m, nlat=91, nlon=180):
    """NEAREST-NEIGHBOUR sample (a, mask) onto a fixed lat-lon mesh.

    Nearest-neighbour, not bilinear: it cannot manufacture values outside
    the field's own range, which matters because a land cell carries a
    ~20 degC arm-dependent fill. The cost is that a land value can be
    pulled up to half a source cell into the ocean, which is what _erode
    then removes.

    The matrix leaves an arm on its NATIVE mesh when that mesh is already
    lat-lon (latlon stays 36x72) and regrids the unstructured arms to
    181x360, so the saved artifacts are NOT co-located. Interpolating both
    onto one coarse mesh (2 deg by default) is what makes a cell-by-cell
    comparison meaningful; it is coarser than every arm, so it smooths
    rather than invents.
    """
    lat_t = np.linspace(-89.0, 89.0, nlat)
    lon_t = np.linspace(0.0, 358.0, nlon)
    # Nearest-neighbour in each axis is enough at this coarsening and
    # cannot manufacture values outside the field's own range.
    lat_src = np.asarray(lat, dtype=np.float64)
    lon_src = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
    ji = np.abs(lat_t[:, None] - lat_src[None, :]).argmin(axis=1)
    ii = np.abs(((lon_t[:, None] - lon_src[None, :] + 180.0) % 360.0)
                - 180.0).argmin(axis=1)
    return (lat_t, lon_t, a[np.ix_(ji, ii)], m[np.ix_(ji, ii)])


def _erode(mask):
    """Shrink a boolean mask by one cell in all EIGHT directions.

    Arms disagree about what a LAND cell holds -- the lat-lon C-grid pins
    land tracers at 0 degC while MPAS Neumann-fills them with an
    ocean-neighbour average -- so one land cell leaking through the
    regrid contributes a ~20 degC difference and swamps the interior
    signal. MEASURED 2026-08-10: unmasked, the geostrophic-adjustment
    latlon|mpas difference reads 2.98 degC and is ENTIRELY the two polar
    (>80 deg, land) bins, while every interior bin agrees to
    0.003-0.05 degC.

    Longitude wraps, latitude does NOT: rolling across the north pole
    into the south pole would erode with the wrong neighbour (codex
    2026-08-10). Diagonals are included because nearest-neighbour
    regridding can pull a land value across a corner.
    """
    m = np.asarray(mask)
    pad = np.zeros((m.shape[0] + 2, m.shape[1]), dtype=bool)
    pad[1:-1] = m                      # dry beyond the poles, never wrapped
    pad = np.concatenate([pad[:, -1:], pad, pad[:, :1]], axis=1)  # lon wraps
    out = np.ones_like(m, dtype=bool)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            out &= pad[di:di + m.shape[0], dj:dj + m.shape[1]]
    return out


#: cos(lat) weights on the common mesh (see _to_common_mesh).
_W_LAT = np.cos(np.radians(np.linspace(-89.0, 89.0, 91)))[:, None]


def _wrms(d, w):
    return float(np.sqrt(np.sum(w * d ** 2) / max(w.sum(), 1e-30)))


def _pair_diff(A, mA, B, mB, shift: bool = True):
    """Area-weighted difference of two fields ALREADY on the common mesh.

    ONE implementation, used by the arm-to-arm comparison AND by the
    per-arm self-error. They have to share it: the self-error is only a
    valid tolerance for the cross-arm number if both are reduced with the
    same masking, the same erosion and the same weights.

    Returns:
      rms_abs        area-weighted RMS difference on the COMMON wet cells
      rms_shifted    the same after removing the best rigid zonal shift, and
      shift_deg      that shift. A dispersive wave that two arms propagate
                     at slightly different speeds shows a large rms_abs and
                     a much smaller rms_shifted: that is a PHASE difference,
                     not different physics. Equal values mean the fields
                     genuinely differ (GLM-5.2).
      overlap_frac   fraction of the mesh both arms call ocean -- pairs with
                     different overlap are NOT directly comparable, so the
                     number is printed rather than hidden.
    """
    both = _erode(mA & mB & np.isfinite(A) & np.isfinite(B))
    # (see _erode)
    if not both.any():
        return dict(rms_abs=float("nan"), rms_shifted=float("nan"),
                    shift_deg=float("nan"), overlap_frac=0.0)
    w = np.where(both, np.broadcast_to(_W_LAT, A.shape), 0.0)
    rms = _wrms(np.where(both, A - B, 0.0), w)
    # Cheapest phase/physics discriminator: minimise over a rigid zonal
    # shift (one roll per candidate; the mesh is 2 deg).
    best, best_shift = rms, 0.0
    n_lon = A.shape[1] if shift else 1
    for k in range(1, n_lon):
        Bk = np.roll(B, k, axis=1)
        mk = _erode(mA & np.roll(mB, k, axis=1)
                    & np.isfinite(A) & np.isfinite(Bk))
        if not mk.any():
            continue
        wk = np.where(mk, np.broadcast_to(_W_LAT, A.shape), 0.0)
        r = _wrms(np.where(mk, A - Bk, 0.0), wk)
        if r < best:
            best, best_shift = r, (k if k <= n_lon // 2 else k - n_lon) * 2.0
    return dict(rms_abs=rms, rms_shifted=best, shift_deg=best_shift,
                overlap_frac=float(both.sum() / both.size))


def _on_common_mesh(npz: Path, field: str, t_index: int = -1):
    """(field, wet mask) at one output time, sampled onto the common mesh."""
    lat, lon, a, m, _t = _load(npz, field)
    _lt, _ln, A, M = _to_common_mesh(lat, lon, a[t_index], m)
    return A, M > 0.5


def _describes_this_case(npz: Path, case: str, grid: str | None = None
                         ) -> bool:
    """Does the ``results.txt`` beside this snapshot record THIS case, on
    THIS grid, finished? The same evidence :func:`_run_matrix_arm` accepts.

    The grid is checked as well as the case: a perfectly valid snapshot of
    the same case on ANOTHER grid would otherwise be accepted under the
    wrong key and compared against itself's sibling (codex round 2). The
    RESOLUTION is deliberately not checked -- the refinement caller exists
    to pass one the registry does not name.
    """
    rt = Path(npz).parent / "results.txt"
    if not rt.is_file():
        return False
    rec = {}
    for line in rt.read_text().splitlines():
        if ": " in line:
            k, _, val = line.partition(": ")
            rec.setdefault(k.strip(), val.strip())
    if rec.get("test") != case or rec.get("status") not in ("PASS", "FAIL"):
        return False
    return grid is None or rec.get("grid") == grid


def cross_grid_rms(root: Path, case: str, grids=CONSISTENCY_GRIDS,
                   npz_by_grid=None):
    """Arm-to-arm difference of the final field, in the field's own units.

    Per-pair entries are documented on :func:`_pair_diff`. A case-level
    reference amplitude (independent of any pair) is returned separately so
    the differences can be read against one fixed scale.

    ``npz_by_grid`` supplies the snapshots directly instead of locating the
    registered ones under ``root`` -- the same escape hatch
    :func:`front_position` already takes, and what lets
    :func:`refinement_agreement` reduce a REFINED pair through this exact
    code path rather than a second copy of it.

    A supplied path skips :func:`_find`, and therefore skips its refusal of
    artifacts left behind by an earlier configuration. Each one is
    re-checked here against its own ``results.txt``: the case must match
    and the record must carry a verdict. Without that, a copied or stale
    NPZ handed in by a caller would be reduced as though it were current
    (codex 2026-08-13). The RESOLUTION is deliberately not checked, because
    the whole point of the refinement caller is to pass a resolution the
    registry does not name.
    """
    field = CASE_FIELD[case]
    got, got0 = {}, {}
    src = (npz_by_grid if npz_by_grid is not None
           else {g: _find(root, case, g) for g in grids})
    for g, p in src.items():
        if p is None:
            continue
        if npz_by_grid is not None and not _describes_this_case(p, case,
                                                                 g):
            print(f"    [supplied artifact refused] {case}/{g}: "
                  f"{p} does not record a completed run of this case")
            continue
        got[g] = _on_common_mesh(p, field)
        got0[g] = _on_common_mesh(p, field, t_index=0)
    if not got:
        return {}, float("nan")
    # ONE case-level reference: the area-weighted anomaly RMS of the arm
    # with the largest wet area, chosen without reference to any pair.
    ref_name = max(got, key=lambda g: got[g][1].sum())
    Aref, Mref = got[ref_name]
    wr = np.where(Mref & np.isfinite(Aref), _W_LAT, 0.0)
    mean_ref = float(np.sum(np.where(wr > 0, Aref, 0.0) * wr) / max(wr.sum(), 1e-30))
    case_ref = float(np.sqrt(np.sum(wr * (np.where(wr > 0, Aref, mean_ref)
                                          - mean_ref) ** 2)
                             / max(wr.sum(), 1e-30)))

    out = {}
    for a_name, b_name in itertools.combinations(sorted(got), 2):
        A, mA = got[a_name]
        B, mB = got[b_name]
        rec = _pair_diff(A, mA, B, mB)
        # CONTROL: the same difference at t=0, before either dycore has
        # taken a step. Every arm gets the SAME analytic initial condition,
        # so whatever this reads is pure discretisation of that IC -- mesh
        # phase, cell size, the regrid.
        A0, mA0 = got0[a_name]
        B0, mB0 = got0[b_name]
        rec["rms_initial"] = _pair_diff(A0, mA0, B0, mB0, shift=False)["rms_abs"]
        # DECISIVE: || (A_final - A_0) - (B_final - B_0) ||. This cancels the
        # IC-discretisation pattern (A_0 - B_0) ALGEBRAICALLY and leaves only
        # what the two dycores did differently. Comparing rms_abs against
        # rms_initial cannot do that job: an RMS is one scalar, so an arm
        # that damps the IC pattern and one that amplifies it can land on the
        # same RMS by coincidence, and growth ORTHOGONAL to the IC difference
        # inflates the RMS only by sqrt(2) even when it equals it (GLM-5.2
        # 2026-08-10, which is also why the earlier "final <= 2x t=0" rule
        # was dropped -- that factor had no derivation).
        rec["rms_evolution"] = _pair_diff(A - A0, mA & mA0, B - B0, mB & mB0,
                                          shift=False)["rms_abs"]
        # Is the FINAL difference the same PATTERN as the t=0 one, merely
        # rescaled? corr ~ 1 says yes (IC discretisation carried along);
        # corr ~ 0 says the discrepancy is new.
        both = _erode(mA & mB & mA0 & mB0 & np.isfinite(A) & np.isfinite(B)
                      & np.isfinite(A0) & np.isfinite(B0))
        if both.sum() > 2:
            w = np.where(both, np.broadcast_to(_W_LAT, A.shape), 0.0)
            df, d0 = np.where(both, A - B, 0.0), np.where(both, A0 - B0, 0.0)
            mf = np.sum(w * df) / w.sum()
            m0 = np.sum(w * d0) / w.sum()
            cov = np.sum(w * (df - mf) * (d0 - m0))
            den = np.sqrt(np.sum(w * (df - mf) ** 2)
                          * np.sum(w * (d0 - m0) ** 2))
            rec["pattern_corr_with_t0"] = float(cov / den) if den > 0 else \
                float("nan")
        else:
            rec["pattern_corr_with_t0"] = float("nan")
        out[f"{a_name}|{b_name}"] = rec
    return out, case_ref


# ---------------------------------------------------------------------------
# Front-position metric (lock exchange)
#
# A smooth-field RMS cannot adjudicate a DISCONTINUITY. Two arms that place
# the same sharp front in the same place, but smear it over 1 cell versus 3,
# differ by ~the full temperature contrast over those cells, so the SST RMS
# reads "huge" while the physics they disagree about -- where the gravity
# current has got to -- is identical. Conversely two arms that both keep the
# front razor-sharp but 10 degrees apart can score a MODEST RMS on a coarse
# mesh. The front POSITION is the quantity the case is about, so it is
# measured directly (GLM-5.2 + codex 2026-08-10).
# ---------------------------------------------------------------------------

def _crossings(lon, row, level):
    """Sub-cell longitudes where ``row`` crosses ``level``, PERIODIC in lon.

    LINEAR SUB-CELL interpolation, not cell-index precision: on this
    geometry the front moves a fraction of a cell over the run, so a
    nearest-cell front reports exactly zero motion on every arm (measured
    -- that was this probe's first, wrong, answer).

    The wrap-around pair (last column, first column) IS included. The lock
    exchange's cold/warm boundary sits at ``front_longitude`` = 0, which on
    a 0..358 mesh falls BETWEEN the last column and the first; scanning
    only interior pairs misses one of the case's two fronts entirely.

    Returns ``(longitudes, n_crossings)``. The COUNT is reported alongside
    the positions because an arm that oscillates around ``level`` produces
    extra crossings -- that is a monotonicity failure, and it would
    otherwise hide inside a position that still looks reasonable.

    Cells sitting EXACTLY at ``level`` are skipped and the crossing is
    located between the nearest NON-ZERO cells on either side, so:
      * a plateau of exact zeros is ONE crossing at the plateau centre,
        not one crossing per zero cell, and
      * a tangential touch that returns the way it came (``-, 0, -``) is
        NOT a crossing at all.
    An earlier revision appended a crossing for every zero cell and read
    ``[-1, 0, 0, +1]`` as three fronts, and its successor still read
    ``[-1, 0, -1, +1]`` as three (codex 2026-08-10, both measured). With
    no exact zeros this reduces to the ordinary adjacent-pair rule.

    A NaN (land) between two samples BREAKS the pair: no crossing is
    reported across a land gap rather than one being invented there.
    """
    lon = np.asarray(lon, dtype=np.float64)
    d = np.asarray(row, dtype=np.float64) - level
    n = d.size
    finite = np.isfinite(d)
    idx = [i for i in range(n) if finite[i] and d[i] != 0.0]
    if len(idx) < 2:
        return [], 0
    xs = []
    for k in range(len(idx)):
        i, j = idx[k], idx[(k + 1) % len(idx)]
        if (d[i] > 0.0) == (d[j] > 0.0):
            continue                       # same side: no crossing
        # Everything strictly between them is an exact zero or a NaN; a
        # NaN means the two samples are on opposite sides of land.
        span = [(i + 1 + s) % n for s in range((j - i - 1) % n)]
        if any(not finite[s] for s in span):
            continue
        w = d[i] / (d[i] - d[j])                       # 0..1 between cells
        # FORWARD (eastward) arc from i to j, which is the arc the span
        # above actually walked. The signed SHORTEST difference is wrong
        # as soon as a zero plateau makes that arc longer than 180 deg:
        # for [-1, 0, 0, +1] at [0, 90, 180, 270] it put the plateau
        # crossing at 315 instead of 135 (codex 2026-08-10, round 2).
        # Requires the longitude axis to increase eastward, which every
        # saved artifact does (latlon 0..355, regridded 0..360).
        step = (lon[j] - lon[i]) % 360.0
        xs.append(float((lon[i] + w * step) % 360.0))
    return xs, len(xs)


def _front_offsets(npz: Path, level: float, refs, t_index: int = -1,
                   field: str = "SST", k_level: int | None = None):
    """Signed front offset from each reference longitude, per common latitude.

    Crossings are found on the arm's OWN saved mesh (sub-cell precision in
    longitude is the whole point), and only the resulting smooth
    offset-vs-latitude CURVE is interpolated onto the common latitudes, so
    arms can be differenced row by row without coarsening the front itself.

    Offsets, not absolute longitudes: ``((x - ref + 180) % 360) - 180`` is
    continuous across the 0/360 seam, where the front actually sits.
    """
    lat, lon, a, mask, _t = _load(npz, field)   # _load already collapses mask
    fld = a[t_index] if k_level is None else a[t_index][..., k_level]
    nlat = lat.size
    off = {r: np.full(nlat, np.nan) for r in refs}
    counts = np.full(nlat, np.nan)
    for j in range(nlat):
        row = np.where(mask[j, :] > 0.5, fld[j, :], np.nan)
        if np.isfinite(row).sum() < 4:
            continue
        xs, n = _crossings(lon, row, level)
        counts[j] = n
        if not xs:
            continue
        for r in refs:
            dl = np.array([((x - r + 180.0) % 360.0) - 180.0 for x in xs])
            best = dl[int(np.argmin(np.abs(dl)))]
            # The two reference fronts are 180 deg apart, so a crossing
            # more than 90 deg from a reference belongs to the OTHER one.
            # Without this, a row where land hides the 0E front assigns
            # its surviving 180E crossing to the 0E reference at ~180 deg
            # and reports it as a colossal displacement (codex 2026-08-10).
            # A front that genuinely migrated past 90 deg would be
            # RELABELLED by the same rule; ``max_offset`` below is how that
            # would show up, since this identification stops being
            # trustworthy long before the front gets there (round 2).
            if abs(best) < 90.0:
                off[r][j] = best
    lat_t = np.linspace(-89.0, 89.0, 91)
    lat_a = np.asarray(lat, dtype=np.float64)
    order = np.argsort(lat_a)               # np.interp needs ascending x
    out = {}
    for r in refs:
        good = np.isfinite(off[r][order])
        if good.sum() < 2:
            out[r] = np.full(lat_t.size, np.nan)
            continue
        src = lat_a[order][good]
        vals = np.interp(lat_t, src, off[r][order][good],
                         left=np.nan, right=np.nan)
        # Do NOT interpolate across a gap (a land band, or latitudes where
        # no crossing exists): np.interp draws a straight line through it
        # and invents a front where the arm has none. A nearest-source
        # distance test is NOT enough -- one missing latitude leaves a 2h
        # gap whose midpoint is only h from a source point, so it survives
        # the test (codex 2026-08-10). Split the source into contiguous
        # SEGMENTS instead and blank every target that falls in a break.
        native = float(np.median(np.diff(lat_a[order]))) if lat_a.size > 1 \
            else np.inf
        breaks = np.nonzero(np.diff(src) > 1.5 * native)[0]
        for b in breaks:
            vals[(lat_t > src[b]) & (lat_t < src[b + 1])] = np.nan
        out[r] = vals
    ok = np.isfinite(counts)
    return out, (float(np.median(counts[ok])) if ok.any() else float("nan"))


def _front_sep_km(off_a, off_b):
    """Area-weighted RMS zonal separation of two front curves, in km.

    cos(lat) appears TWICE and for two different reasons -- stated because
    the two are easy to conflate (GLM-5.2 2026-08-10):
      * inside ``d_km`` it is the metric term converting a longitude
        difference into a distance along the latitude circle;
      * in ``w`` it is the area weight, the same cos(lat) every other
        reduction in this script uses, so a row near the pole (little
        area) does not count as much as a row at the equator.
    """
    lat_t = np.linspace(-89.0, 89.0, 91)
    good = np.isfinite(off_a) & np.isfinite(off_b)
    if not good.any():
        return float("nan"), 0
    w = np.cos(np.radians(lat_t))[good]
    # WRAPPED difference: both offsets live in (-180, 180], so +179 minus
    # -179 is -2 degrees, not +358 (codex 2026-08-10).
    d_deg = ((off_a[good] - off_b[good] + 180.0) % 360.0) - 180.0
    d_km = (np.radians(d_deg) * C.R_earth
            * np.cos(np.radians(lat_t[good])) / 1e3)
    return float(np.sqrt(np.sum(w * d_km ** 2) / w.sum())), int(good.sum())


def front_position(root: Path, grids=ALL_GRIDS, npz_by_grid=None,
                   field: str = "SST", k_level: int | None = None):
    """Arm-to-arm agreement on WHERE the lock-exchange front is.

    Per arm: median crossing count (2 = the case's two fronts and nothing
    else), and each front's displacement from its initial position.

    Per pair, the headline number is the separation of the DISPLACEMENTS,
    not of the final positions. The arms do not start with the front in the
    same place: the initial condition is a step at ``front_longitude``, and
    each mesh resolves that step at its own cell edge, so at t=0 the
    equatorial front already sits at 357.5 deg on lat-lon 36x72, 357.8 on
    MPAS and 0.48 on FESOM (MEASURED 2026-08-10). That mesh phase is up to
    ~6 deg = several hundred km and it is not a dycore difference; taking
    it out is the same move as comparing anomalies rather than absolutes.
    The initial separation is reported next to it as the irreducible floor.

    CAVEAT, not fixed here: differencing displacements removes the OFFSET
    but not everything the offset causes -- a front released half a cell
    further west spins its head up against a slightly different cell
    geometry, so its propagation is not perfectly phase-independent
    (GLM-5.2). The check that would settle it is a regression of
    displacement difference on initial offset across the arms; with three
    comparable arms and displacements of order 1 km against a 200-900 km
    offset it would have no power, so it is not run and the limitation is
    stated instead.
    """
    from legoesm.ocean.experiments.lock_exchange import LockExchangeConfig
    cfg = LockExchangeConfig()
    level = 0.5 * (cfg.T_cold_C + cfg.T_warm_C)
    x0 = float(cfg.front_longitude) % 360.0
    refs = (x0, (x0 + 180.0) % 360.0)
    paths = (npz_by_grid if npz_by_grid is not None
             else {g: _find(root, "lock_exchange", g) for g in grids})
    arms, final, initial = {}, {}, {}
    for g, p in paths.items():
        if p is None:
            continue
        o1, n1 = _front_offsets(p, level, refs, t_index=-1,
                                field=field, k_level=k_level)
        o0, _n0 = _front_offsets(p, level, refs, t_index=0,
                                 field=field, k_level=k_level)
        final[g], initial[g] = o1, o0
        moved = {}
        for r in refs:
            d, npts = _front_sep_km(o1[r], o0[r])
            moved[f"front_{r:.0f}E_moved_km"] = d
            moved[f"front_{r:.0f}E_rows"] = npts
        # Largest distance any row's front sits from its reference. The
        # 90-degree assignment window in _front_offsets silently relabels
        # a front that migrates past it, so this is the number that says
        # whether the identification is still safe.
        max_off = max((float(np.nanmax(np.abs(o1[r])))
                       for r in refs if np.isfinite(o1[r]).any()),
                      default=float("nan"))
        arms[g] = dict(median_crossings=n1, max_offset_deg=max_off, **moved)
    pairs = {}
    for a_name, b_name in itertools.combinations(sorted(final), 2):
        row = {}
        for r in refs:
            # Wrapped, for the same reason as in _front_sep_km.
            disp_a = ((final[a_name][r] - initial[a_name][r] + 180.0)
                      % 360.0) - 180.0
            disp_b = ((final[b_name][r] - initial[b_name][r] + 180.0)
                      % 360.0) - 180.0
            d, npts = _front_sep_km(disp_a, disp_b)
            row[f"front_{r:.0f}E_sep_km"] = d
            row[f"front_{r:.0f}E_init_sep_km"] = _front_sep_km(
                initial[a_name][r], initial[b_name][r])[0]
            row[f"front_{r:.0f}E_rows"] = npts
        pairs[f"{a_name}|{b_name}"] = row
    return dict(level_degC=level, refs_deg=list(refs), arms=arms, pairs=pairs)


def theory_lock_exchange(root: Path, grids=ALL_GRIDS):
    """Gravity-current front speed vs Benjamin (1968) c = 0.5 sqrt(g' H).

    The front is tracked as the longitude where the equatorial SST crosses
    the mid-temperature, measured from its initial position.
    """
    from legoesm.ocean.experiments.lock_exchange import LockExchangeConfig
    cfg = LockExchangeConfig()
    g_prime = C.g * 2.0e-4 * (cfg.T_warm_C - cfg.T_cold_C)   # linear EOS
    c_theory = 0.5 * np.sqrt(g_prime * cfg.H_max)
    T_mid = 0.5 * (cfg.T_cold_C + cfg.T_warm_C)
    rows = {}
    for grid in grids:
        p = _find(root, "lock_exchange", grid)
        if p is None:
            continue
        lat, lon, a, m, t = _load(p, "SST")
        j = int(np.argmin(np.abs(lat)))            # equatorial row

        def _front(k):
            """Longitude of the first DESCENDING (warm -> cold) equatorial
            crossing of T_mid, with LINEAR SUB-CELL interpolation. Cell-index
            precision is not enough: theory predicts ~1.9 deg of travel in 5
            days against a 1-2 deg output mesh, so a nearest-cell front
            reports exactly zero motion on every arm (measured -- that was
            this probe's first, wrong, answer).

            Interior pairs only, unlike :func:`_crossings`: this measures ONE
            front's displacement between two times, and taking the same
            array-order crossing at both times is what makes the difference a
            displacement. The wrap-aware, both-fronts version is the
            consistency metric (:func:`front_position`)."""
            row = np.where(m[j, :] > 0.5, a[k, j, :], np.nan)
            if np.isfinite(row).sum() < 4:
                return np.nan
            d = row - T_mid
            idx = [i for i in range(len(d) - 1)
                   if np.isfinite(d[i]) and np.isfinite(d[i + 1])
                   and d[i] > 0 >= d[i + 1]]
            if not idx:
                return np.nan
            i = idx[0]
            w = d[i] / (d[i] - d[i + 1])          # 0..1 between the cells
            return float(lon[i] + w * (lon[i + 1] - lon[i]))

        x0, x1 = _front(0), _front(len(t) - 1)
        dt_s = (t[-1] - t[0]) * 86400.0
        dx_deg = abs(((x1 - x0 + 180.0) % 360.0) - 180.0)
        speed = dx_deg * (np.pi / 180.0) * C.R_earth / dt_s if dt_s else np.nan
        # CAN THIS ARM RESOLVE THE ANSWER AT ALL? The comparison is only a
        # measurement if the front is expected to cross a few cells during
        # the run. On the global aquaplanet arms it is not: at 36x72 the
        # equatorial cell is 556 km wide and Benjamin predicts 42.8 km of
        # travel in the registered 1 day, i.e. 0.08 of ONE CELL. What the
        # sub-cell interpolation then reports is how the two arms smeared a
        # step, not how fast a gravity current ran -- the same defect as
        # the old IGW L2 gate, which rewarded a frozen dycore. The
        # resolved measurement lives on the latlon_regional Petersen
        # channel (1 km cells, 64 km domain), not here.
        dlon = float(np.median(np.abs(np.diff(np.asarray(lon,
                                                          dtype=np.float64)))))
        cell_m = dlon * (np.pi / 180.0) * C.R_earth
        cells = c_theory * dt_s / cell_m if cell_m > 0 else np.nan
        resolvable = bool(cells >= 1.0)
        rows[grid] = dict(front_deg_moved=dx_deg, speed_m_s=speed,
                          # REFUSED, not merely annotated: an unresolvable
                          # arm carries no ratio at all, so a reader who
                          # takes the number out of the JSON cannot quote
                          # "0.01 x theory" as a model result (codex
                          # 2026-08-13). The raw speed stays, because it is
                          # what was measured; the COMPARISON is what the
                          # resolution cannot support.
                          ratio_to_theory=(speed / c_theory if resolvable
                                           else None),
                          output_cell_width_km=cell_m / 1e3,
                          theory_cells_on_output_mesh=float(cells),
                          resolvable=resolvable)
    return dict(theory_c_m_s=c_theory, g_prime=g_prime, arms=rows)


def theory_rest_state(root: Path, case: str, grids=ALL_GRIDS):
    """Exact expectation: a rest state stays at rest."""
    rows = {}
    for grid in grids:
        p = _find(root, case, grid)
        if p is None:
            continue
        z = np.load(p)
        m = np.asarray(z["land_mask"], dtype=np.float64)
        m = m[-1] if m.ndim == 3 else m
        wet = m > 0.5
        # u_sfc can be FACE-staggered (n_lon+1 on a C-grid), so it does
        # not accept the cell mask; compare shapes before indexing rather
        # than assuming (it raised IndexError on latlon's 73 columns).
        if "u_sfc" not in z:
            # Not every arm's extractor saves a surface velocity; say so
            # instead of dying or silently reporting 0.
            rows[grid] = dict(max_abs_u_final=float("nan"),
                              u_is_cell_shaped=False,
                              max_abs_eta_final=float(np.nanmax(np.abs(
                                  np.asarray(z["eta"],
                                             dtype=np.float64)[-1][wet]))))
            continue
        u = np.asarray(z["u_sfc"], dtype=np.float64)[-1]
        u_wet = u[wet] if u.shape == wet.shape else u[np.isfinite(u)]
        if u_wet.size == 0:      # all-NaN face field: say so, not "nan"
            u_wet = np.array([np.nan])
        eta = np.asarray(z["eta"], dtype=np.float64)
        rows[grid] = dict(
            max_abs_u_final=float(np.nanmax(np.abs(u_wet))),
            u_is_cell_shaped=bool(u.shape == wet.shape),
            max_abs_eta_final=float(np.nanmax(np.abs(eta[-1][wet]))))
    return dict(theory="|u| = 0, |eta| = 0 exactly", arms=rows)


def theory_barotropic_wave(root: Path, grids=ALL_GRIDS):
    """Shallow-water phase speed c = sqrt(g H) [reported, not gated].

    The suite's barotropic wave is a Gaussian bump on a GLOBAL basin, so
    the crest disperses and wraps; a single propagation distance is not a
    clean speed measurement. The theoretical speed is printed so the
    scale is on the page, and the arm-to-arm spread of peak |eta| is the
    part that is actually comparable.
    """
    H = 5500.0
    rows = {}
    for grid in grids:
        p = _find(root, "barotropic_wave", grid)
        if p is None:
            continue
        lat, lon, a, m, t = _load(p, "eta")
        wet = m > 0.5
        rows[grid] = dict(peak_eta_final=float(np.nanmax(np.abs(a[-1][wet]))),
                          peak_eta_initial=float(np.nanmax(np.abs(a[0][wet]))))
    return dict(theory_c_m_s=float(np.sqrt(C.g * H)),
                note="global basin: dispersive + wrapping, speed not gated",
                arms=rows)


def theory_lock_exchange_convergence(resolutions=("36x72", "72x144",
                                                   "144x288"), days=5.0):
    """Does the front speed CONVERGE to Benjamin under refinement?

    This is the check that turns the lock exchange from "the front barely
    moves, so something is broken" into a result. It RUNS the model (the
    only part of this script that does), because convergence cannot be
    read off a single saved run.

    MEASURED 2026-08-10 on lat-lon (5 days):
        dx 556 km -> 0.041 m/s (0.082 x Benjamin)
        dx 278 km -> 0.110 m/s (0.222 x)
        dx 139 km -> 0.247 m/s (0.499 x)
    The deficit shrinks with dx and the trend is toward 0.5*sqrt(g'H),
    but this is NOT a demonstrated convergence: three points, the finest
    still 2x below theory, and none of them asymptotic (the finest cell,
    139 km, is still ~14x the deformation radius). The honest statement
    is "the front-speed deficit decreases under refinement, limit not
    demonstrated"; settling it needs a run at dx of order the
    deformation radius (~10 km), which the global configuration cannot
    reach (codex + GLM-5.2 both refused the stronger claim).
    The global suite configuration is ~4 orders of
    magnitude too coarse to resolve a 20 m-deep gravity-current head
    (dx/H ~ 2.8e4), so the shipped case does NOT reproduce Benjamin and
    is not expected to -- its value is the RPE mixing metric, not front
    dynamics.

    REFUTED on the way: the arrest is not ROTATIONAL. Rerunning with
    f = 0 gave a front speed identical to 4 significant figures, so the
    ~10 km deformation radius is not what limits the current.
    """
    import importlib.util
    import sys
    from types import SimpleNamespace
    from legoesm.ocean.experiments.lock_exchange import LockExchangeConfig

    sys.path.insert(0, str(_REPO / "scripts" / "matrix"))
    spec = importlib.util.spec_from_file_location(
        "_rm_conv", _REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py")
    rm = importlib.util.module_from_spec(spec)
    sys.modules["_rm_conv"] = rm
    spec.loader.exec_module(rm)

    cfg = LockExchangeConfig()
    g_prime = C.g * 2.0e-4 * (cfg.T_warm_C - cfg.T_cold_C)
    c_theory = 0.5 * np.sqrt(g_prime * cfg.H_max)
    T_mid = 0.5 * (cfg.T_cold_C + cfg.T_warm_C)
    rows = {}
    for res in resolutions:
        tc = SimpleNamespace(grid_type="latlon", resolution=res,
                             case="lock_exchange", run_kwargs={})
        grid, z, config, model, _ck, _lo, _la = rm._create_ocean_setup(
            tc, nlev=cfg.nlev, H_max=cfg.H_max)
        st = rm._create_rest_state(tc, grid, z, H_max=cfg.H_max)
        st = rm._init_lock_exchange(st, "latlon", grid, z)
        lat_d = np.degrees(np.asarray(grid.lat))
        lon_d = np.degrees(np.asarray(grid.lon))
        j = int(np.argmin(np.abs(lat_d)))

        def _front(state):
            row = np.asarray(state.T.data)[j, :, 0] - T_mid
            idx = [i for i in range(len(row) - 1) if row[i] > 0 >= row[i + 1]]
            if not idx:
                return np.nan
            i = idx[0]
            w = row[i] / (row[i] - row[i + 1])
            return lon_d[i] + w * (lon_d[i + 1] - lon_d[i])

        x0 = _front(st)
        for _ in range(int(days * 86400 / rm.DEFAULT_DT)):
            st = model.step(st, rm.DEFAULT_DT)
        dx_deg = abs(((_front(st) - x0 + 180.0) % 360.0) - 180.0)
        speed = dx_deg * (np.pi / 180.0) * C.R_earth / (days * 86400.0)
        rows[res] = dict(
            dx_km=float(2 * np.pi * C.R_earth / np.asarray(grid.lon).size / 1e3),
            speed_m_s=float(speed), ratio_to_theory=float(speed / c_theory))
    return dict(theory_c_m_s=float(c_theory), arms=rows)


# ---------------------------------------------------------------------------
# Per-arm, per-case SELF-ERROR: the measured tolerance
# ---------------------------------------------------------------------------

#: Cases the self-error is measured for. The rest-state cases are excluded
#: because their expectation is EXACT (|u| = 0) and already gated as such --
#: a convergence-rate tolerance is meaningless there.
SELF_ERROR_CASES = ["geostrophic_adjustment", "barotropic_wave",
                    "phillips_two_layer", "inertia_gravity_wave",
                    "lock_exchange"]

#: Arms with a 2x sibling mesh. fesom ("pi") and tripole ("eorca1") are
#: mesh-file-backed and ship exactly one mesh each, so they cannot be
#: refined and cannot be self-measured -- stated, not silently omitted.
SELF_ERROR_GRIDS = ["latlon", "mpas"]


def _refine(grid: str, res: str) -> str | None:
    """The registered resolution's 2x sibling, or None if the arm has none.

    One refinement step, matched across grid families to a LINEAR ratio of
    2: C24 -> C48 and 36x72 -> 72x144 halve the cell width, and one more
    icosahedral SUBDIVISION quadruples the cell count (ico4 = 2562 cells,
    ico5 = 10242), i.e. halves it too -- which is why the step is on the
    subdivision index and not on a cell count. Comparing self-errors taken
    at DIFFERENT refinement ratios would systematically excuse whichever
    arm was refined least (GLM-5.2 2026-08-10).
    """
    if grid == "cubed_sphere" and res.startswith("C") and res[1:].isdigit():
        return f"C{int(res[1:]) * 2}"
    if grid == "mpas" and res.startswith("ico") and res[3:].isdigit():
        return f"ico{int(res[3:]) + 1}"
    if "x" in res and all(p.isdigit() for p in res.split("x")):
        nlat, nlon = res.split("x")
        return f"{int(nlat) * 2}x{int(nlon) * 2}"
    return None


def _run_matrix_arm(case: str, grid: str, res: str, root: Path,
                    prefix: list[str], force: bool = False) -> Path | None:
    """Run ONE matrix arm and return its snapshot, or None if it did not run.

    Deliberately the matrix's OWN entry point in a subprocess rather than a
    re-implementation of the case setup here: the self-error is only a
    tolerance for the suite's numbers if it is produced by the suite's
    initial condition, timestep, gates and extraction. The two invocations
    of a pair differ in ``--resolution`` and NOTHING else, which is the
    whole controlled comparison.
    """
    out = root / case / grid / res

    def _accept(where: str) -> Path | None:
        """A snapshot counts only if it is a COMPLETE record of THIS run.

        Exit status alone is not evidence and the presence of a file is
        even less: a failed child can leave an earlier snapshot in place,
        and a copied directory looks identical (codex 2026-08-10). The
        sibling ``results.txt`` carries the case, grid, resolution and
        verdict the runner itself recorded; the first three must match.

        A recorded ``FAIL`` is ACCEPTED and flagged. The gate verdict is a
        statement about physics (cube geostrophic adjustment has failed
        its gate since long before this script existed), not about whether
        the snapshot faithfully records what that arm did -- and requiring
        PASS made the self-error of a known-failing arm unmeasurable while
        re-running it on every invocation. What is refused is a run that
        left no verdict at all.
        """
        for h in sorted(out.glob(f"**/{grid}/{res}/snapshots_latlon.npz")):
            if h.parent.parent.name != grid or h.parent.name != res:
                continue
            rt = h.parent / "results.txt"
            if not rt.is_file():
                print(f"    [{where}] {case}/{grid}/{res}: no results.txt "
                      f"beside the snapshot -- refusing it")
                continue
            # EXACT key/value comparison, not substring: "resolution:
            # 36x72_backup" contains "resolution: 36x72" (codex round 2).
            rec = {}
            for line in rt.read_text().splitlines():
                if ": " in line:
                    k, _, val = line.partition(": ")
                    rec.setdefault(k.strip(), val.strip())
            if (rec.get("test"), rec.get("grid"), rec.get("resolution")) \
                    != (case, grid, res):
                print(f"    [{where}] {case}/{grid}/{res}: results.txt "
                      f"describes a different run -- refusing it")
                continue
            if rec.get("status") not in ("PASS", "FAIL"):
                print(f"    [{where}] {case}/{grid}/{res}: the run left no "
                      f"PASS/FAIL verdict -- refusing it")
                continue
            if rec["status"] == "FAIL":
                print(f"    [{where}] {case}/{grid}/{res}: NOTE this arm "
                      f"FAILS its own gate; its self-error is still a "
                      f"discretisation scale, but say so when quoting it")
            return h
        return None

    if not force:
        hit = _accept("cached")
        if hit is not None:
            return hit
    cmd = prefix + [
        sys.executable,
        str(_REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py"),
        "--only", f"={case}", "--grid", grid, "--resolution", res,
        "--output", str(out)]
    print(f"    running {case}/{grid}/{res} ...", flush=True)
    env = dict(os.environ, JAX_ENABLE_X64="1")
    proc = subprocess.run(cmd, cwd=_REPO, env=env,
                          capture_output=True, text=True)
    # Exit status is NOT the acceptance test: the matrix exits 1 whenever
    # a case fails its gate, and several arms here do so for pre-existing
    # physics reasons, so gating on it would reject exactly the runs whose
    # self-error is most worth having (codex round 2). The written record
    # is the evidence; the exit code is reported only when there is none.
    hit = _accept("after run")
    if hit is None:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
        print(f"    [no usable snapshot] {case}/{grid}/{res} "
              f"(exit {proc.returncode})\n      {tail}")
    return hit


def self_error(root: Path, cases=None, grids=None, prefix=(), force=False):
    """Each arm against ITSELF at 2x resolution, reduced like a cross-arm pair.

    RUNS the model (twice per arm). The result is directly comparable to
    the CONSISTENCY block's ``rms_abs`` because it goes through the same
    :func:`_on_common_mesh` sampling and the same :func:`_pair_diff`
    reduction -- if it did not, it would not be a tolerance for it.

    For the lock exchange the front-position separation is reported too:
    the case is a discontinuity, and the field RMS cannot tell "the front
    is somewhere else" from "the front is in the same place but one cell
    wider" (see the front-position section).
    """
    cases = list(cases or SELF_ERROR_CASES)
    grids = list(grids or SELF_ERROR_GRIDS)
    out = {}
    for case in cases:
        for grid in grids:
            base = _registered_resolution(case, grid)
            if base is None:
                out[f"{grid}|{case}"] = dict(
                    skipped="arm not registered for this case")
                continue
            out[f"{grid}|{case}"] = _self_error_pair(
                case, grid, base, root, prefix, force)
    return out


def _self_error_pair(case: str, grid: str, base: str, root: Path,
                     prefix=(), force: bool = False):
    """One arm's own discretisation error at resolution ``base``.

    Split out of :func:`self_error` so :func:`refinement_agreement` can ask
    for it at the REFINED resolution too, through the same runs and the
    same reduction. A budget measured only at the coarse level cannot say
    whether a cross-arm difference that grows under refinement is the two
    dycores parting company or simply each arm's own solution still moving.
    """
    field = CASE_FIELD[case]
    fine = _refine(grid, base)
    if fine is None:
        return dict(skipped=f"no 2x sibling for {grid} {base!r}")
    pb = _run_matrix_arm(case, grid, base, root, list(prefix), force)
    pf = _run_matrix_arm(case, grid, fine, root, list(prefix), force)
    if pb is None or pf is None:
        return dict(skipped="one of the two runs produced no snapshot "
                            "(see log above)", base=base, refined=fine)
    # The evolution below is (final - initial) on each side, so the two
    # runs must share BOTH endpoints. Cadence in between may differ;
    # endpoints may not (codex round 2).
    tb = np.load(pb)["times_days"]
    tf = np.load(pf)["times_days"]
    if not (np.isclose(tb[0], 0.0) and np.isclose(tf[0], 0.0)
            and np.isclose(tb[-1], tf[-1])):
        return dict(skipped=f"time bases differ: base spans "
                            f"{tb[0]}..{tb[-1]} d, refined "
                            f"{tf[0]}..{tf[-1]} d",
                    base=base, refined=fine)
    A, mA = _on_common_mesh(pb, field)
    B, mB = _on_common_mesh(pf, field)
    A0, mA0 = _on_common_mesh(pb, field, t_index=0)
    B0, mB0 = _on_common_mesh(pf, field, t_index=0)
    rec = dict(_pair_diff(A, mA, B, mB),
               base=base, refined=fine, field=field)
    # The tolerance has to be the SAME QUANTITY the cross-arm verdict is
    # taken on, or a row can be flagged "IC-dominated" and "outside budget"
    # in the same breath. Both sides are therefore the
    # difference-of-differences.
    rec["rms_evolution"] = _pair_diff(A - A0, mA & mA0, B - B0, mB & mB0,
                                      shift=False)["rms_abs"]
    if case in RMS_NOT_GATED:
        fp = front_position(root, npz_by_grid={"base": pb, "refined": pf})
        pair = fp["pairs"].get("base|refined", {})
        # DISPLACEMENT separations only. "front_0E_init_sep_km" also ends
        # in "_sep_km" and is the mesh-phase floor, two orders of magnitude
        # larger; including it emitted a 115 km front tolerance instead of
        # 0.96 km.
        rec["front_sep_km"] = {
            k: v for k, v in pair.items()
            if k.endswith("_sep_km") and not k.endswith("_init_sep_km")}
        rec["median_crossings"] = {
            k: v["median_crossings"] for k, v in fp["arms"].items()}
    return rec


def _evolution_amplitude(paths: dict, field: str) -> float:
    """Area-weighted RMS of one arm's OWN evolution, (final - initial).

    The normaliser has to be the same KIND of quantity as the thing it
    normalises. The refinement numerator is an evolution difference, but
    the case reference amplitude ``cross_grid_rms`` returns is the FINAL
    field's anomaly RMS, which still contains the static initial structure
    -- a case whose answer barely evolves has a large final anomaly and a
    tiny evolution, and dividing one by the other manufactures agreement
    (codex 2026-08-13).

    The reference arm is the one with the largest AREA-weighted wet
    region, using the same cos(lat) weights as every other reduction here;
    an unweighted cell count would prefer whichever arm happens to keep
    more polar cells.
    """
    best, best_area = None, -1.0
    for g, npz in sorted(paths.items()):
        A, mA = _on_common_mesh(npz, field)
        w = np.where(mA & np.isfinite(A), _W_LAT, 0.0)
        area = float(w.sum())
        if area > best_area:
            best, best_area = g, area
    if best is None:
        return float("nan")
    A, mA = _on_common_mesh(paths[best], field)
    A0, mA0 = _on_common_mesh(paths[best], field, t_index=0)
    both = _erode(mA & mA0 & np.isfinite(A) & np.isfinite(A0))
    if not both.any():
        return float("nan")
    w = np.where(both, np.broadcast_to(_W_LAT, A.shape), 0.0)
    return _wrms(np.where(both, A - A0, 0.0), w)


def _native_evolution_amplitude(paths: dict, fields=("SST", "eta")):
    """How far each arm's own solution moved, ON ITS OWN MESH.

    WHY THIS IS NOT A DUPLICATE of the common-mesh amplitude. Every other
    number in this file is reduced on the shared 2-degree mesh, which the
    MPAS arm reaches through an unstructured regrid and the lat-lon arm
    reaches through something close to the identity. If the two arms are
    found to evolve by different amounts, the FIRST question is whether the
    regrid did it -- a resolution-dependent smoothing applied to one arm
    and not the other would produce exactly that, and would mean the
    comparison pipeline, not the models, is the finding.

    So the same quantity is computed a second time from
    ``snapshots_native.npz``, on each arm's own cells with no regrid in the
    path at all. If the ratio survives, the regrid is not the explanation.

    MEASURED 2026-08-13 on phillips_two_layer, native vs common mesh:

        field         common   native
        SST coarse     2.9x     2.79x
        SST refined    2.2x     2.13x
        eta coarse     1.4x     1.34x
        eta refined    1.4x     1.33x

    i.e. it survives, and the regrid-artefact reading (GLM-5.2's leading
    hypothesis) is refuted. The MPAS arm genuinely moves its surface
    tracer about 2-3x further than the lat-lon arm on this case.
    """
    out = {}
    for field in fields:
        per_arm = {}
        for g, npz in sorted(paths.items()):
            native = Path(npz).parent / "snapshots_native.npz"
            if not native.is_file():
                per_arm[g] = None
                continue
            z = np.load(native)
            if field not in z.files or "land_mask" not in z.files:
                per_arm[g] = None
                continue
            a = np.asarray(z[field], dtype=np.float64)
            m = np.asarray(z["land_mask"], dtype=np.float64)
            m = m[-1] if m.ndim == a.ndim else m
            wet = m > 0.5
            d = (a[-1] - a[0])[wet]
            d = d[np.isfinite(d)]
            # UNWEIGHTED on purpose, and it must stay a like-for-like
            # comparison of the SAME arm against itself across levels
            # rather than an area-weighted global mean: the two native
            # meshes have different cell areas, so this is a scale, not a
            # conserved quantity. It is only ever read as a RATIO.
            per_arm[g] = float(np.sqrt(np.mean(d ** 2))) if d.size else None
        names = [g for g in sorted(per_arm) if per_arm[g]]
        rec = dict(per_arm=per_arm)
        if len(names) == 2:
            a, b = names
            rec["ratio"] = float(per_arm[b] / per_arm[a])
            rec["ratio_of"] = f"{b}/{a}"
        out[field] = rec
    return out


def _difference_growth(paths: dict, field: str):
    """Arm-to-arm evolution difference at EVERY shared output time.

    Returns the series, plus the e-folding time of a log-linear fit over
    the growing part. The fit is reported with its own R^2 so a series that
    is not exponential at all cannot be quoted as a growth rate.

    THE R^2 CUTS BOTH WAYS, and the earlier revision of this docstring got
    that wrong (GLM-5.2, 2026-08-13): a POOR fit is not evidence that the
    growth is non-exponential, it is evidence that the fit decides nothing.
    Concluding "so the instability reading is refuted" from R^2 = 0.24 is
    the same error as concluding the opposite from it. Only a fit that is
    GOOD and has a slope near zero refutes exponential growth; anything
    else leaves the question open, and the honest control is a single-arm
    perturbed-initial-condition pair, which this function does not run.
    """
    names = sorted(paths)
    if len(names) < 2:
        return {}
    if len(names) > 2:
        # One fit, one pair. Returning it unlabelled let the printer show
        # the same curve under EVERY pair when more than two arms were
        # requested (codex round 2). The pair is recorded and the printer
        # only shows it against that pair.
        return dict(skipped=f"{len(names)} arms supplied; this fit is "
                            f"defined for a single pair",
                    pair=f"{names[0]}|{names[1]}")
    a, b = names[0], names[1]
    ta = np.load(paths[a])["times_days"]
    tb = np.load(paths[b])["times_days"]
    n = min(len(ta), len(tb))
    if n < 3 or not np.allclose(ta[:n], tb[:n]):
        return dict(skipped="the two arms do not share an output cadence")
    A0, mA0 = _on_common_mesh(paths[a], field, t_index=0)
    B0, mB0 = _on_common_mesh(paths[b], field, t_index=0)
    times, series = [], []
    for i in range(n):
        A, mA = _on_common_mesh(paths[a], field, t_index=i)
        B, mB = _on_common_mesh(paths[b], field, t_index=i)
        series.append(_pair_diff(A - A0, mA & mA0, B - B0, mB & mB0,
                                 shift=False)["rms_abs"])
        times.append(float(ta[i]))
    s = np.asarray(series, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    good = np.isfinite(s) & (s > 0) & (t > t[0])
    out = dict(pair=f"{a}|{b}", times_days=times, rms_evolution=series)
    if good.sum() >= 3:
        y = np.log(s[good])
        x = t[good]
        slope, icept = np.polyfit(x, y, 1)
        resid = y - (slope * x + icept)
        var = float(np.sum((y - y.mean()) ** 2))
        # A FLAT series has zero log variance. Flooring the denominator
        # made it report R^2 = 1.0 -- a perfect fit to nothing, read as
        # "cleanly exponential" (codex 2026-08-13). Undefined is the
        # correct answer there, and the caller must handle it.
        out["log_fit_r2"] = (1.0 - float(np.sum(resid ** 2)) / var
                             if var > 1e-12 else float("nan"))
        # The SIGNED slope, always. Reporting only "e_folding = inf" for a
        # non-positive slope hid the difference between a flat series and
        # one that is DECAYING cleanly, and an exact exponential decay
        # printed as R^2 = 1.0 with e_folding = inf, which reads as growth
        # that could not be timed.
        out["log_slope_per_day"] = float(slope)
        out["trend"] = ("growing" if slope > 0 else
                        "decaying" if slope < 0 else "flat")
        out["e_folding_days"] = float(1.0 / slope) if slope > 0 else \
            float("inf")
        # Dynamic range: a fit over a series that barely moves cannot
        # distinguish exponential from anything else, whatever its R^2.
        out["log_range"] = float(y.max() - y.min())
    return out


#: Reporting bands for the refinement ratio. NOT derived from anything:
#: one refinement step gives one number with no uncertainty attached, so
#: these are labels on a continuum, not calibrated thresholds, and every
#: place that prints a verdict says so. Deriving them would need the
#: spread of the ratio over repeated refinements, which the suite does not
#: produce (codex 2026-08-13).
_REFINE_CONVERGING_BELOW = 0.75
_REFINE_DIVERGING_ABOVE = 1.25


def _refinement_verdict(rec: dict, pair: str) -> dict:
    """Verdict for one pair, with every reason it might not be one.

    Returns ``{"label", "ratio", "quantity", "caveats"}``. The label is
    UNASSESSABLE whenever the number that would carry it is not finite --
    an earlier revision let a NaN fall through both comparisons and print
    DIVERGING, i.e. missing data was reported as the worst possible
    result (codex 2026-08-13).
    """
    caveats = []
    if rec.get("front_ratio") is not None and pair in rec.get("front_ratio",
                                                              {}):
        ratio = rec["front_ratio"][pair]
        quantity = "front displacement"
    else:
        ratio = rec.get("ratio_normalised", {}).get(pair, float("nan"))
        quantity = "field RMS, normalised by the case amplitude"
    dob = rec.get("difference_over_budget", {}).get("refined", {})
    budgeted = np.isfinite(dob.get(pair, float("nan")))
    if not budgeted:
        # Attached BEFORE the non-finite early return, so a row that is
        # both unbudgeted and unmeasurable still says both (codex round 2).
        caveats.append(
            "the refined level has no measured self-error, so this trend "
            "has no tolerance of its own (pass --refined-budget)")
    if not np.isfinite(ratio):
        caveats.append("the ratio is not finite: one level produced no "
                       "comparable measurement")
        return dict(label="UNASSESSABLE", ratio=ratio, quantity=quantity,
                    budgeted=bool(budgeted), caveats=caveats)
    # Is the case's own answer settled? On the RMS lane that is a real
    # question with a measured answer. On the FRONT lane it is NOT: the
    # amplitude that would be compared is built from the field RMS this
    # file has already declared unable to adjudicate the case, so using it
    # to add or drop a caveat would smuggle the invalid quantity back into
    # the verdict (codex round 3). Say it is undefined instead.
    on_front = quantity.startswith("front")
    settled = True
    if on_front:
        caveats.append(
            "no case-amplitude check on the front lane: the amplitude this "
            "file measures is the field RMS, which is exactly the quantity "
            "the case is NOT judged by")
        settled = False
    elif not rec.get("case_converged", True):
        caveats.append(
            f"the case's own amplitude moved "
            f"{100 * rec['case_amplitude_moved']:.0f}% between levels")
        settled = False
    # NOT "DIVERGING". Two resolutions are a DELTA, not a trend, and in
    # this suite the difference's slope IN TIME is negative -- it appears
    # early and decays -- so "diverging" would be read as the simulation
    # coming apart, which is the opposite of what is measured (GLM-5.2,
    # 2026-08-13). The words describe the delta and nothing else.
    label = ("DIFFERENCE SHRANK" if ratio < _REFINE_CONVERGING_BELOW else
             "DIFFERENCE FLAT" if ratio < _REFINE_DIVERGING_ABOVE else
             "DIFFERENCE GREW")
    # And the thing a reader actually needs: is it now outside the arms'
    # own tolerance?
    over = rec.get("difference_over_budget", {}).get("refined", {}).get(pair)
    if over is not None and np.isfinite(over) and over > 1.0:
        label = f"BUDGET EXCEEDED ({over:.2f}x) — {label.lower()}"
        caveats.append(
            "the two dycores now differ by MORE than either arm's own "
            "resolution sensitivity")
    # The WORD, not a footnote. A caveat under a line that says
    # "CONVERGING" is still read as a convergence claim (codex rounds 2
    # and 3). Two separate things can remove the claim and leave only a
    # direction: no tolerance at the refined level, and a case whose own
    # answer is still moving between the two levels. Either one demotes
    # the headline; the direction stays visible in the parentheses.
    if not budgeted or not settled:
        why = ("UNBUDGETED" if not budgeted and settled else
               "UNCONVERGED" if budgeted and not settled else
               "UNBUDGETED, UNCONVERGED")
        label = f"{why} TREND ({label.lower()})"
    return dict(label=label, ratio=float(ratio), quantity=quantity,
                budgeted=bool(budgeted), case_settled=bool(settled),
                caveats=caveats)


#: Cell-shaped fields worth decomposing a disagreement into, and the term
#: each one points at. u_sfc/v_sfc are FACE-staggered on the C-grid arm
#: (n_lon+1 / n_lat+1 columns) and are not comparable cell-by-cell, so the
#: cell-shaped ``speed_sfc`` stands in for the velocity.
_DECOMPOSE_FIELDS = {
    "eta": "free surface — pressure gradient and the divergent mode",
    "SST": "surface tracer — advection and mixing",
    "speed_sfc": "surface velocity — Coriolis and momentum advection",
}

#: Fields whose two arms are NOT reduced by the same operator, with the
#: reason. A row listed here is reported and must NOT be read as a
#: cross-grid result: the difference it shows contains the difference
#: between the two reductions.
_NOT_LIKE_FOR_LIKE = {
    "speed_sfc":
        "the arms build this with DIFFERENT operators. The lat-lon C-grid "
        "averages each staggered component to cell centres with a 2-point "
        "mean; MPAS reconstructs a cell-centred vector from the edge "
        "normals by Perot, which averages over ~6 edges and therefore "
        "smooths more. A smaller MPAS speed follows from the reduction "
        "alone, so this row cannot separate a dynamical difference from "
        "the reconstruction. MEASURED 2026-08-13 and it does read the "
        "other way from every other field -- lat-lon moves 2.1x further "
        "here while MPAS moves 2.9x further in SST -- which is exactly "
        "what a smoothing reduction would produce and is NOT evidence "
        "that MPAS has a weaker surface flow.",
}


def _field_decomposition(paths: dict, t_index: int = -1):
    """Which FIELD carries the arm-to-arm disagreement?

    One norm over one field says how MUCH two dycores disagree; it cannot
    say about WHAT. Splitting the same evolution difference across the
    saved fields localises the term: a disagreement that lives in the free
    surface points at the pressure gradient and the divergent mode, one in
    the surface tracer at advection and mixing, one in the velocity at the
    Coriolis discretisation (GLM-5.2, 2026-08-13).

    Each field's difference is divided by THAT FIELD'S own evolution
    amplitude, because eta in metres and SST in degrees cannot be compared
    raw. The returned fraction is therefore "how far apart the arms are,
    in units of how much this field actually moved".

    TWO WARNINGS ON READING THE FRACTION (GLM-5.2, 2026-08-13):

    * It is unreliable for a field the case barely moves. A relaxed tracer
      on a timescale comparable to the run length has a structurally small
      denominator, so a bounded difference divides into a fraction above 1
      that says nothing physical. Read the ABSOLUTE difference and its
      growth for such a field, not the fraction.
    * The row for the case's OWN headline field is the headline number
      restated, not new information. What the decomposition adds is the
      OTHER rows -- specifically whether they share the headline's trend.
    """
    names = sorted(paths)
    if len(names) < 2:
        return {}
    a, b = names[0], names[1]
    out = {}
    for field, meaning in _DECOMPOSE_FIELDS.items():
        try:
            A, mA = _on_common_mesh(paths[a], field, t_index=t_index)
            B, mB = _on_common_mesh(paths[b], field, t_index=t_index)
            A0, mA0 = _on_common_mesh(paths[a], field, t_index=0)
            B0, mB0 = _on_common_mesh(paths[b], field, t_index=0)
        except KeyError:
            # NAMED, not skipped. speed_sfc is saved by the lat-lon
            # extractor and NOT by the MPAS one, so the velocity row simply
            # vanished from the table -- which reads as "velocity agrees"
            # rather than "velocity was never compared" (2026-08-13, the
            # same class as the map plotter returning 9 of 10 figures in
            # silence).
            out[field] = dict(skipped="not saved by every arm",
                              points_at=meaning)
            continue
        d = _pair_diff(A - A0, mA & mA0, B - B0, mB & mB0,
                       shift=False)["rms_abs"]
        # SYMMETRIC, and on the NUMERATOR'S OWN SUPPORT. An earlier
        # revision divided by the first sorted arm's motion, computed on
        # that arm's mask -- so "lat-lon" became the yardstick purely by
        # alphabetical order, on a different set of cells from the
        # difference it was scaling, and a nearly stationary lat-lon arm
        # would have sent the ratio to infinity however normally MPAS
        # evolved (codex 2026-08-13). Both arms are reported, and the
        # headline scale is the larger of the two, on the same cells the
        # difference used.
        both = _erode(mA & mA0 & mB & mB0 & np.isfinite(A) & np.isfinite(A0)
                      & np.isfinite(B) & np.isfinite(B0))
        w = np.where(both, np.broadcast_to(_W_LAT, A.shape), 0.0)
        moved_a = _wrms(np.where(both, A - A0, 0.0), w)
        moved_b = _wrms(np.where(both, B - B0, 0.0), w)
        scale = max(moved_a, moved_b)
        out[field] = dict(difference=float(d),
                          moved={a: float(moved_a), b: float(moved_b)},
                          field_moved=float(scale),
                          fraction=float(d / scale) if scale > 0
                          else float("nan"),
                          points_at=meaning)
    return out


def refinement_agreement(root: Path, cases=None, grids=None, prefix=(),
                         force=False, budget_levels=("base",)):
    """Does the arm-to-arm difference SHRINK when both arms are refined?

    WHY THIS EXISTS. The consistency block asks "is the arm-to-arm
    difference inside one arm's own discretisation error?" and every case
    currently answers yes -- with the budget 1.5x to 245x larger than the
    difference it is judging. A test that cannot fail has not established
    agreement; it has established that the instrument is blunt. Measured
    2026-08-11, EVOLUTION difference against its budget:
    geostrophic_adjustment 0.69, barotropic_wave 0.48, phillips 0.36,
    inertia_gravity_wave 0.38, lock-exchange front displacement 0.004.

    The question a coarse pair cannot answer is whether the two dycores are
    converging to the SAME solution or merely to solutions that are both
    blurry enough to overlap. Refining BOTH arms one step discriminates:

      * difference falls with the refinement  -> the arms are converging
        together; the coarse agreement was real.
      * difference stays flat while each arm's own solution moves -> a
        genuine cross-grid disagreement that the coarse budget was hiding.

    Both resolutions are reduced through :func:`cross_grid_rms`, i.e. the
    same common mesh, erosion and area weights as the headline number, and
    the refined runs are the ones :func:`self_error` already produces, so
    the usual invocation runs no extra model.
    """
    cases = list(cases or SELF_ERROR_CASES)
    grids = list(grids or SELF_ERROR_GRIDS)
    out = {}
    for case in cases:
        rec = {"levels": {}}
        for level in ("base", "refined"):
            paths, res_used = {}, {}
            for grid in grids:
                res = _registered_resolution(case, grid)
                if res is None:
                    continue
                if level == "refined":
                    res = _refine(grid, res)
                    if res is None:
                        continue
                p = _run_matrix_arm(case, grid, res, root, list(prefix),
                                    force)
                if p is not None:
                    paths[grid], res_used[grid] = p, res
            if len(paths) < 2:
                rec["levels"][level] = dict(
                    skipped="fewer than two arms produced a snapshot",
                    resolutions=res_used)
                continue
            # Endpoints must match on BOTH sides or the difference mixes a
            # time offset into a discretisation one -- the same guard
            # self_error applies to its own pair.
            ends = {g: np.load(p)["times_days"] for g, p in paths.items()}
            t_end = {g: float(t[-1]) for g, t in ends.items()}
            if (max(t_end.values()) - min(t_end.values())
                    > 1e-9 * max(1.0, max(abs(v) for v in t_end.values()))):
                rec["levels"][level] = dict(
                    skipped=f"arms end at different times: {t_end}",
                    resolutions=res_used)
                continue
            pairs, case_ref = cross_grid_rms(root, case, npz_by_grid=paths)
            rec["levels"][level] = dict(
                pairs=pairs, case_ref=case_ref, resolutions=res_used,
                case_ref_evolution=_evolution_amplitude(paths,
                                                        CASE_FIELD[case]))
            # A case this file has already declared un-adjudicable by field
            # RMS must not be adjudicated by field RMS here either. For the
            # lock exchange the quantity is how far the front MOVED, so the
            # refinement question is asked of that instead.
            if case in RMS_NOT_GATED:
                fp = front_position(root, npz_by_grid=paths)
                rec["levels"][level]["front"] = {
                    pair: {k: v for k, v in rows.items()
                           if k.endswith("_sep_km")
                           and not k.endswith("_init_sep_km")}
                    for pair, rows in fp["pairs"].items()}
            # IS THE DISAGREEMENT GROWING EXPONENTIALLY? An unstable case
            # amplifies ANY difference, including the two meshes'
            # discretisation of the same initial condition, at the flow's
            # own growth rate. If the arm-to-arm difference grows like
            # exp(t/tau) with tau comparable to the case's instability
            # timescale, then a larger difference at higher resolution is
            # the instability doing its job on a sharper initial state --
            # NOT evidence that the two dycores disagree about the physics.
            # Distinguishing those two readings is the whole question for
            # phillips_two_layer, and a single end-time RMS cannot.
            rec["levels"][level]["difference_growth"] = _difference_growth(
                paths, CASE_FIELD[case])
            rec["levels"][level]["by_field"] = _field_decomposition(paths)
            rec["levels"][level]["native_amplitude"] = \
                _native_evolution_amplitude(paths)
            # THE BUDGET AT THIS LEVEL. Without it a growing cross-arm
            # difference is unreadable: each arm's own solution is still
            # moving under refinement, and the question is whether the two
            # arms are parting company FASTER than that.
            # Measuring it at the REFINED level costs a 4x run per arm
            # (lat-lon 144x288, MPAS ico6), so it is opt-in. When it is not
            # asked for the level records that it is UNMEASURED rather than
            # quietly reusing the coarse budget, which would compare a
            # refined difference against a coarse tolerance and flatter the
            # refined arm.
            rec["levels"][level]["self_error"] = (
                {g: _self_error_pair(case, g, r, root, prefix, force)
                 for g, r in sorted(res_used.items())}
                if level in budget_levels else
                {g: dict(skipped="budget not measured at this level "
                                 "(pass --refined-budget)")
                 for g in sorted(res_used)})
        # For a case this file has already declared un-adjudicable by field
        # RMS, the refinement question is asked of the FRONT DISPLACEMENT --
        # not merely reported next to an RMS verdict that was taken anyway
        # (codex 2026-08-13: printing the front under an RMS verdict is not
        # the same as gating on it).
        if case in RMS_NOT_GATED:
            fb = rec["levels"].get("base", {}).get("front", {})
            ff = rec["levels"].get("refined", {}).get("front", {})
            rec["front_ratio"] = {}
            for k in sorted(set(fb) & set(ff)):
                # EVERY front must be resolved on both levels. Python's
                # max() over [finite, nan] returns the finite value, so a
                # pair with one unresolved front was scoring a verdict off
                # the other one (codex round 2).
                lo_v, hi_v = list(fb[k].values()), list(ff[k].values())
                if (not lo_v or not hi_v
                        or not all(np.isfinite(x) for x in lo_v + hi_v)):
                    rec["front_ratio"][k] = float("nan")
                    continue
                lo, hi = max(lo_v), max(hi_v)
                rec["front_ratio"][k] = (float(hi / lo) if lo > 0
                                         else float("nan"))
        b = rec["levels"].get("base", {}).get("pairs", {})
        f = rec["levels"].get("refined", {}).get("pairs", {})
        rec["ratio_refined_over_base"] = {
            k: (float(f[k]["rms_evolution"] / b[k]["rms_evolution"])
                if b.get(k, {}).get("rms_evolution", 0.0) > 0 else float("nan"))
            for k in sorted(set(b) & set(f))}
        # CONTROL, without which the raw ratio is unreadable. Refining an
        # unstable or front-resolving case makes the SOLUTION bigger too --
        # more resolved eddy energy, a sharper gravity current -- so a
        # difference that grows in absolute terms may be a constant
        # FRACTION of a growing signal, which is convergence, not
        # disagreement. Both levels are therefore also divided by that
        # level's own case reference amplitude (the area-weighted anomaly
        # RMS of the widest arm), and it is the normalised ratio that
        # carries the verdict.
        cb = rec["levels"].get("base", {}).get("case_ref_evolution",
                                                float("nan"))
        cf = rec["levels"].get("refined", {}).get("case_ref_evolution",
                                                  float("nan"))
        rec["case_ref_base"], rec["case_ref_refined"] = cb, cf
        # IS THE CASE ITSELF CONVERGED? If the solution's own amplitude
        # moves a lot between the two levels, neither level is near the
        # continuous answer and NO ratio here -- raw or normalised -- is a
        # convergence result. It is then only a statement about relative
        # rates of approach to an unknown limit (GLM-5.2, 2026-08-13). The
        # 25% band is a reporting threshold, not a physical one, and is
        # labelled as such wherever it is printed.
        rec["case_amplitude_moved"] = (
            float(abs(cf - cb) / cb) if (cb > 0 and np.isfinite(cf))
            else float("nan"))
        rec["case_converged"] = bool(rec["case_amplitude_moved"] <= 0.25) \
            if np.isfinite(rec["case_amplitude_moved"]) else False
        # D/E at each level: the cross-arm difference measured in units of
        # the arms' OWN discretisation error there. This, not the bare
        # difference, is what has to fall for "the arms agree" to mean
        # anything -- and it is the number the coarse-only consistency
        # block reports without ever asking whether it improves.
        for level in ("base", "refined"):
            se_l = rec["levels"].get(level, {}).get("self_error", {})
            finite = {g: v["rms_evolution"] for g, v in se_l.items()
                      if np.isfinite(v.get("rms_evolution", np.nan))}
            rec.setdefault("budget_arms", {})[level] = sorted(finite)
            # PER PAIR, and only when BOTH of that pair's arms were
            # measured. A max over every arm in the case let one unrelated
            # arm's large self-error relax every pair, and let a pair with
            # only ONE measured arm receive a finite ratio it had not
            # earned (codex 2026-08-13). MAX of the two, not the sum: if
            # both arms converge to the same limit L then
            # |A - B| <= |A - L| + |B - L| ~ e_A + e_B, so the sum is the
            # loose bound and the max is the stricter of the two, which is
            # the direction this whole change pushes.
            # For a case whose verdict is taken on the FRONT, the budget
            # must be the front self-error too. Building it from the RMS
            # self-error let --refined-budget clear the "unbudgeted"
            # caveat using the very quantity the case declares invalid
            # (codex round 2). _self_error_pair already records
            # front_sep_km for exactly these cases.
            on_front = case in RMS_NOT_GATED
            if on_front:
                finite = {}
                for g, v in se_l.items():
                    fs = list(v.get("front_sep_km", {}).values())
                    if fs and all(np.isfinite(x) for x in fs):
                        finite[g] = max(fs)
                rec["budget_arms"][level] = sorted(finite)
            per_pair, pair_budget = {}, {}
            for k in sorted(rec["levels"].get(level, {}).get("pairs", {})):
                ga, _, gb = k.partition("|")
                if ga not in finite or gb not in finite:
                    per_pair[k] = float("nan")
                    pair_budget[k] = float("nan")
                    continue
                bud = max(finite[ga], finite[gb])
                pair_budget[k] = float(bud)
                # On the front lane the numerator is the level's own front
                # separation in km, taken from the same table the verdict
                # uses; on the RMS lane it is the evolution difference.
                num = (max(rec["levels"][level].get("front", {})
                           .get(k, {}).values(), default=float("nan"))
                       if on_front
                       else rec["levels"][level]["pairs"][k]["rms_evolution"])
                per_pair[k] = (float(num / bud)
                               if bud > 0 and np.isfinite(num)
                               else float("nan"))
            rec.setdefault("budget", {})[level] = pair_budget
            rec.setdefault("difference_over_budget", {})[level] = per_pair
            # PEAK OVER TIME, not only the final sample. The final time is
            # ONE sample of a difference that need not be monotone, and on
            # phillips it is not: D(t) spikes in the first output interval
            # and then OSCILLATES with a ~2-day period, so day 10 lands in
            # a trough at one resolution and mid-swing at the other. A
            # final-time comparison then reads that sampling phase as a
            # resolution effect -- the same defect as gating the old
            # barotropic wave on the global peak at one instant. GLM-5.2
            # asked for this a round before it was implemented.
            gr = rec["levels"].get(level, {}).get("difference_growth", {})
            series = np.asarray(gr.get("rms_evolution", []), dtype=np.float64)
            peak = (float(np.nanmax(series)) if series.size
                    else float("nan"))
            rec.setdefault("difference_peak", {})[level] = peak
            # UNITS. The peak series is the FIELD difference, but on the
            # front lane the budget is a displacement in km. Dividing one
            # by the other printed a meaningless 0.01 for the lock exchange
            # (caught immediately on the first run of this metric). There
            # is no front-displacement TIME SERIES to take a peak of, so
            # the ratio is undefined on that lane and says so.
            rec.setdefault("peak_over_budget", {})[level] = {
                k: (float(peak / pair_budget[k])
                    if not on_front and np.isfinite(peak)
                    and np.isfinite(pair_budget.get(k, np.nan))
                    and pair_budget[k] > 0 else float("nan"))
                for k in pair_budget}
            # PER ARM as well as the max. A max-based ratio can pass while
            # one arm carries almost all of the self-error and the other is
            # tight -- and which arm is the loose one changes what to do
            # next (GLM-5.2, 2026-08-13).
            rec.setdefault("budget_per_arm", {})[level] = dict(finite)
        rec["ratio_normalised"] = {
            k: (float((f[k]["rms_evolution"] / cf)
                      / (b[k]["rms_evolution"] / cb))
                if (cb > 0 and cf > 0
                    and b.get(k, {}).get("rms_evolution", 0.0) > 0)
                else float("nan"))
            for k in sorted(set(b) & set(f))}
        # THE SIGNATURE THAT SEPARATES A DEFECT FROM A COARSE MESH.
        # Refining changes two things at once: how far apart the arms are
        # (D) and how well each arm knows its own answer (E). Only one
        # combination is diagnostic on its own:
        #
        #   E falls  and D rises  -> each arm is converging, and they are
        #                            converging to DIFFERENT limits. That
        #                            is an algorithmic inconsistency, not
        #                            a resolution artefact.
        #   E rises  and D rises  -> the case is de-settling; nothing can
        #                            be adjudicated at these resolutions.
        #   E rises  and D falls  -> D/E improves for the wrong reason: the
        #                            tolerance loosened. Not evidence.
        #   E falls  and D falls  -> the arms are converging together.
        #
        # A D/E ratio alone cannot tell the first case from the third
        # (GLM-5.2, 2026-08-13), which is exactly the pair a reader will
        # confuse.
        bud = rec.get("budget", {})
        # THE t=0 DIFFERENCE, WHICH CONSTRAINS THE INITIAL-CONDITION STORY
        # WITHOUT SETTLING IT. If it SHRINKS under refinement while the
        # evolution difference grows, the arms start closer together and
        # finish further apart.
        #
        # THAT IS A CONSTRAINT, NOT AN EXONERATION, and an earlier revision
        # of this comment claimed the latter (GLM-5.2, 2026-08-13). What is
        # measured here is the difference AFTER both arms are sampled onto
        # the COMMON mesh; what drives each simulation is the IC on its own
        # NATIVE mesh, and regridding is not an orthogonal projection, so
        # the two do not decompose additively. A t=0 difference that is
        # smaller in NORM can still carry more power in the directions that
        # grow -- for a Phillips jet, the unstable manifold. Settling it
        # needs both arms started from ONE high-resolution analytic field
        # regridded to each native mesh, which this function does not do.
        rec["initial_difference"] = {
            k: dict(base=float(b[k]["rms_initial"]),
                    refined=float(f[k]["rms_initial"]),
                    ratio=(float(f[k]["rms_initial"] / b[k]["rms_initial"])
                           if b[k]["rms_initial"] > 0 else float("nan")))
            for k in sorted(set(b) & set(f))}
        rec["signature"] = {}
        for k in sorted(set(b) & set(f)):
            e_lo = bud.get("base", {}).get(k, float("nan"))
            e_hi = bud.get("refined", {}).get(k, float("nan"))
            d_lo = b[k]["rms_evolution"]
            d_hi = f[k]["rms_evolution"]
            if not (np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo > 0
                    and d_lo > 0):
                rec["signature"][k] = "unmeasured"
                continue
            # The D ratio quoted here is FINAL-TIME. Where the difference
            # is not monotone in time that number is sampling-dependent --
            # on phillips the final-time ratio is 3.30x and the PEAK ratio
            # is 1.17x -- so the peak one is quoted beside it and neither
            # is allowed to stand alone.
            pk = rec.get("difference_peak", {})
            p_lo, p_hi = pk.get("base", np.nan), pk.get("refined", np.nan)
            peak_txt = (f"; on PEAK sampling {p_hi / p_lo:.2f}x"
                        if np.isfinite(p_lo) and np.isfinite(p_hi)
                        and p_lo > 0 else "")
            e_up, d_up = e_hi > e_lo, d_hi > d_lo
            rec["signature"][k] = (
                "converging to DIFFERENT limits: each arm settles "
                "(tolerance {:.2f}x) while they move apart ({:.2f}x{})"
                .format(e_hi / e_lo, d_hi / d_lo, peak_txt)
                if (not e_up and d_up) else
                "case de-settling: BOTH the difference ({:.2f}x) and the "
                "tolerance ({:.2f}x) grow -- not adjudicable here"
                .format(d_hi / d_lo, e_hi / e_lo) if (e_up and d_up) else
                "tolerance loosened ({:.2f}x) faster than the difference "
                "shrank ({:.2f}x) -- the ratio improved for the wrong "
                "reason".format(e_hi / e_lo, d_hi / d_lo)
                if (e_up and not d_up) else
                "converging together: difference {:.2f}x, tolerance "
                "{:.2f}x".format(d_hi / d_lo, e_hi / e_lo))
        rec["verdict"] = {
            k: _refinement_verdict(rec, k) for k in
            sorted(set(b) & set(f))}
        out[case] = rec
    return out


def _lock_exchange_front_report(a, grids, report, case: str) -> None:
    """The lock exchange's ACTUAL verdict: do the arms move the front the
    same distance?

    Measured at the SURFACE and at the BOTTOM. The dense current runs along
    the bottom and that is the front the case is about (Ilicak et al. 2012;
    Petersen et al. 2015); the surface expression can differ between arms
    through vertical mixing alone (GLM-5.2 2026-08-10).

    Gated against FRONT_SELF_ERROR_KM -- one arm's own front displacement
    difference across a 2x resolution change -- for the same reason the
    field tolerance is a measured self-error and not a chosen number.
    """
    report["consistency"][case]["front_position"] = {}
    for where, fld, klev in (("surface", "SST", None),
                             ("bottom", "T_3d", -1)):
        try:
            fp = front_position(a.runs_root, grids, field=fld, k_level=klev)
        except KeyError:
            print(f"    FRONT POSITION [{where}]: field {fld!r} not saved by "
                  f"every arm -- skipped")
            continue
        report["consistency"][case]["front_position"][where] = fp
        r0, r1 = fp["refs_deg"]
        print(f"    FRONT POSITION [{where}] "
              f"(T = {fp['level_degC']:.1f} degC crossings; "
              f"fronts at {r0:.0f}E and {r1:.0f}E)")
        for g, v in sorted(fp["arms"].items()):
            print(f"      {g:13s} crossings/row {v['median_crossings']:.0f}"
                  f"   moved {v[f'front_{r0:.0f}E_moved_km']:7.1f} / "
                  f"{v[f'front_{r1:.0f}E_moved_km']:7.1f} km")
        bad = [g for g, v in fp["arms"].items()
               if np.isfinite(v["median_crossings"])
               and v["median_crossings"] > len(fp["refs_deg"])]
        if bad:
            print(f"      NOTE: {', '.join(sorted(bad))} show more than the "
                  f"{len(fp['refs_deg'])} fronts the case has -- the isotherm "
                  f"is multivalued there and its POSITION is not a "
                  f"trustworthy single number")
        far = sorted(g for g, v in fp["arms"].items()
                     if np.isfinite(v["max_offset_deg"])
                     and v["max_offset_deg"] > 45.0)
        if far:
            print(f"      NOTE: {', '.join(far)} has a front more than 45 deg "
                  f"from its reference; past 90 deg the two fronts swap "
                  f"identity and the displacement becomes meaningless")
        agree, differ, unassessable, budgets = [], [], [], {}
        for k, v in sorted(fp["pairs"].items()):
            ga, gb = k.split("|")
            have = [FRONT_SELF_ERROR_KM.get((g, case)) for g in (ga, gb)]
            seps = [v[f"front_{r:.0f}E_sep_km"] for r in fp["refs_deg"]]
            # Both fronts must be resolved and at least one arm must have a
            # measured front self-error. max() over a list containing NaN
            # returns the finite value depending on order, so a pair with
            # ONE unresolved front used to pass on the surviving one
            # (codex round 2).
            # BOTH arms must have a measured front self-error, the same
            # rule as the field block: judging fesom against lat-lon's
            # convergence rate is exactly what that rule forbids, and
            # fesom/tripole ship one mesh each so they can never have one.
            if not all(np.isfinite(s) for s in seps) or any(e is None
                                                            for e in have):
                unassessable.append(k)
            else:
                budgets[k] = max(have)
                (agree if max(seps) <= budgets[k] else differ).append(k)
            print(f"      {k:30s} displacement differs by "
                  f"{v[f'front_{r0:.0f}E_sep_km']:6.1f} / "
                  f"{v[f'front_{r1:.0f}E_sep_km']:6.1f} km "
                  f"[mesh-phase floor "
                  f"{v[f'front_{r0:.0f}E_init_sep_km']:6.1f} / "
                  f"{v[f'front_{r1:.0f}E_init_sep_km']:6.1f} km, "
                  f"{v[f'front_{r0:.0f}E_rows']} rows]")
        print(f"      -> {len(agree)}/{len(budgets)} assessable pairs move "
              f"the front together within the measured front self-error"
              + (f"; OUTSIDE: {', '.join(sorted(differ))}" if differ else ""))
        if unassessable:
            print(f"      UNASSESSABLE: {', '.join(sorted(unassessable))} -- "
                  f"a front is unresolved, or neither arm has a measured "
                  f"front self-error")
        fp["pairs_agreeing"] = sorted(agree)
        fp["pairs_disagreeing"] = sorted(differ)
        fp["pairs_unassessable"] = sorted(unassessable)
        fp["pair_budgets_km"] = budgets


CONSISTENCY_ONLY = {
    "geostrophic_adjustment":
        "no closed form on a global sphere with varying f: the adjusted "
        "state depends on the full basin geometry",
    "phillips_two_layer":
        "growth rate depends on the resolved instability spectrum; the "
        "quasi-geostrophic Phillips rate does not apply to this geometry",
    "inertia_gravity_wave":
        "the case's own analytic solution is an f-plane plane wave and is "
        "invalid on the sphere (see run_inertia_gravity_wave)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=Path,
                    default=_REPO / "results" / "ocean_grid_benchmark")
    ap.add_argument("--out", type=Path,
                    default=_REPO / "results" / "ocean_grid_benchmark"
                    / "cross_grid_consistency.json")
    ap.add_argument("--convergence", action="store_true",
                    help="also RUN the lock exchange at 3 resolutions to "
                         "test convergence to the Benjamin front speed "
                         "(the only part of this script that runs a model)")
    ap.add_argument("--include-real-geometry", "--include-tripole",
                    dest="include_real_geometry", action="store_true",
                    help="also compare the REAL-GEOMETRY arms "
                         f"({', '.join(REAL_GEOMETRY_GRIDS)}) cell-by-cell "
                         "against the aquaplanets. Off by default: those "
                         "arms have continents, so the number measures the "
                         "coastline as much as the dycore.")
    ap.add_argument("--self-error", action="store_true",
                    help="MEASURE the tolerance instead of assuming it: run "
                         "each arm against itself at 2x resolution through "
                         "the same matrix entry point and reduce the "
                         "difference exactly like a cross-arm pair. RUNS the "
                         "model (two runs per arm per case).")
    ap.add_argument("--self-error-cases", type=str, default="",
                    help="comma-separated subset of "
                         f"{','.join(SELF_ERROR_CASES)}")
    ap.add_argument("--self-error-grids", type=str, default="",
                    help="comma-separated subset of "
                         f"{','.join(SELF_ERROR_GRIDS)}")
    ap.add_argument("--self-error-root", type=Path,
                    default=_REPO / "results" / "ocean_self_error",
                    help="where the self-error runs are written; an existing "
                         "snapshot is reused rather than rerun")
    ap.add_argument("--run-prefix", type=str, default="",
                    help="command prefix for the model runs, e.g. "
                         "'srun --jobid=<id> --overlap -n1 -c8'. Compute must "
                         "not run on a login node.")
    ap.add_argument("--force-rerun", action="store_true",
                    help="rerun self-error arms whose snapshot already exists")
    ap.add_argument("--refined-budget", action="store_true",
                    help="also measure each arm's OWN discretisation error "
                         "at the REFINED resolution, so the refined "
                         "cross-arm difference has a tolerance of its own. "
                         "Costs a 4x run per arm per case (lat-lon 144x288, "
                         "MPAS ico6) -- off by default.")
    ap.add_argument("--refinement", action="store_true",
                    help="ask whether the arm-to-arm difference SHRINKS when "
                         "both arms are refined one step. Reads the same "
                         "runs --self-error produces, so after a --self-error "
                         "pass it runs no model; on its own it runs the same "
                         "two runs per arm.")
    a = ap.parse_args()
    if not a.runs_root.is_dir():
        raise SystemExit(f"no suite output under {a.runs_root}")
    grids = (ALL_GRIDS if a.include_real_geometry
             else CONSISTENCY_GRIDS)

    report = {"consistency": {}, "theory": {}, "consistency_only": {}}

    # Measured FIRST: its numbers are the tolerances the consistency block
    # below is read against.
    if a.self_error:
        se = self_error(
            a.self_error_root,
            cases=[c for c in a.self_error_cases.split(",") if c] or None,
            grids=[g for g in a.self_error_grids.split(",") if g] or None,
            prefix=shlex.split(a.run_prefix), force=a.force_rerun)
        report["self_error"] = se
        print("\nSELF-ERROR  (each arm vs ITSELF at 2x resolution, same "
              "matrix entry point, same reduction as a cross-arm pair)")
        for key in sorted(se):
            grid, case = key.split("|")
            v = se[key]
            if "skipped" in v:
                print(f"    {case:24s} {grid:13s} SKIPPED — {v['skipped']}")
                continue
            if not np.isfinite(v["rms_evolution"]):
                # A non-finite measurement must never become a tolerance:
                # every comparison against NaN is False, so the pairs would
                # come out neither agreeing nor disagreeing (codex
                # 2026-08-10).
                print(f"    {case:24s} {grid:13s} REJECTED — non-finite "
                      f"self-error; the committed value stands")
                continue
            was = SELF_ERROR.get((grid, case))
            new = v["rms_evolution"]
            if was is not None and new > 1.5 * was:
                # A measured value only ever RELAXES the standard, so a
                # degraded arm silently buys itself a pass. Say so loudly.
                print(f"    {case:24s} {grid:13s} WARNING: self-error grew "
                      f"{was:.3e} -> {new:.3e} ({new / was:.1f}x); "
                      f"the tolerance it sets is now looser")
            SELF_ERROR[(grid, case)] = new
            print(f"    {case:24s} {grid:13s} {v['base']:>7s} vs "
                  f"{v['refined']:>8s}  {v['field']:3s} "
                  f"EVOLUTION {new:9.3e}  (raw rms {v['rms_abs']:9.3e})"
                  f"   overlap {v['overlap_frac']:.2f}")
            if "front_sep_km" in v:
                fs = "  ".join(f"{k[:-7]} {x:7.1f} km"
                               for k, x in sorted(v["front_sep_km"].items()))
                print(f"    {'':24s} {'':13s} front separation: {fs}"
                      f"   crossings/row "
                      f"{v['median_crossings']}")
        print("\n    paste into SELF_ERROR (the committed table):")
        for (g, c), e in sorted(SELF_ERROR.items()):
            print(f'        ("{g}", "{c}"): {e:.3g},')
        # The front verdict has its OWN committed table; measuring it and
        # then not emitting it left lock exchange gated on stale numbers
        # (codex round 2).
        front = {}
        for key, v in se.items():
            if "front_sep_km" not in v:
                continue
            grid, case = key.split("|")
            worst = max(v["front_sep_km"].values())
            if np.isfinite(worst):
                front[(grid, case)] = worst
        if front:
            print("\n    paste into FRONT_SELF_ERROR_KM:")
            for (g, c), e in sorted(front.items()):
                print(f'        ("{g}", "{c}"): {e:.3g},')

    if a.refinement:
        ra = refinement_agreement(
            a.self_error_root,
            cases=[c for c in a.self_error_cases.split(",") if c] or None,
            grids=[g for g in a.self_error_grids.split(",") if g] or None,
            prefix=shlex.split(a.run_prefix), force=a.force_rerun,
            budget_levels=(("base", "refined") if a.refined_budget
                           else ("base",)))
        report["refinement_agreement"] = ra
        print("\nREFINEMENT OF AGREEMENT  (does the arm-to-arm EVOLUTION "
              "difference shrink when BOTH arms are refined one step?)")
        print("    A ratio well below 1 means the difference is shrinking "
              "faster than the signal; near or above 1 it is not.\n"
              "    That is a DIRECTION, and it only becomes a convergence "
              "statement once the refined level has a\n"
              "    measured tolerance of its own -- until then every row "
              "here is labelled UNBUDGETED TREND.")
        for case in sorted(ra):
            lv = ra[case]["levels"]
            for pair, ratio in sorted(ra[case]["ratio_refined_over_base"]
                                      .items()):
                b = lv["base"]["pairs"][pair]["rms_evolution"]
                f = lv["refined"]["pairs"][pair]["rms_evolution"]
                rb = "/".join(f"{g}:{r}" for g, r
                              in sorted(lv["base"]["resolutions"].items()))
                rf = "/".join(f"{g}:{r}" for g, r
                              in sorted(lv["refined"]["resolutions"].items()))
                # ONE verdict, computed in the analysis and merely printed
                # here, taken on the quantity the case is actually judged
                # by (front displacement where the field RMS was declared
                # invalid), UNASSESSABLE when the number is not finite,
                # and carrying every reason it might not mean what it says.
                v = ra[case]["verdict"][pair]
                rn = ra[case]["ratio_normalised"].get(pair, float("nan"))
                print(f"    {case:24s} {pair:16s} {b:9.3e} -> {f:9.3e}  "
                      f"RMS ratio {ratio:5.2f}  normalised {rn:5.2f}")
                print(f"    {'':24s} {'':16s} VERDICT {v['label']} on "
                      f"{v['quantity']} (ratio {v['ratio']:.2f}; the "
                      f"{_REFINE_CONVERGING_BELOW}/{_REFINE_DIVERGING_ABOVE} "
                      f"bands are REPORTING LABELS, not calibrated "
                      f"thresholds -- one refinement step carries no "
                      f"uncertainty estimate)")
                for c in v["caveats"]:
                    print(f"    {'':24s} {'':16s}   caveat: {c}")
                print(f"    {'':24s} {'':16s} base {rb}   refined {rf}"
                      f"   case EVOLUTION amplitude "
                      f"{ra[case]['case_ref_base']:.3e} -> "
                      f"{ra[case]['case_ref_refined']:.3e}")
                for level in ("base", "refined"):
                    gr = lv.get(level, {}).get("difference_growth", {})
                    if gr.get("pair") != pair:
                        continue
                    if "log_slope_per_day" not in gr:
                        if "skipped" in gr:
                            print(f"    {'':24s} {'':16s} {level:8s} growth "
                                  f"fit SKIPPED — {gr['skipped']}")
                        continue
                    r2 = gr.get("log_fit_r2", float("nan"))
                    rate = (f"e-folds every {gr['e_folding_days']:.2f} d"
                            if gr["trend"] == "growing"
                            else f"{gr['trend']} (slope "
                                 f"{gr['log_slope_per_day']:+.3f} /d)")
                    note = ("" if (np.isfinite(r2) and r2 > 0.9
                                   and gr.get("log_range", 0.0) > 1.0)
                            else "  -- NOT decisive: a poor or "
                                 "low-dynamic-range log fit refutes "
                                 "NOTHING, in either direction")
                    print(f"    {'':24s} {'':16s} {level:8s} disagreement "
                          f"{rate} (log fit R2 {r2:.2f}, log range "
                          f"{gr.get('log_range', float('nan')):.2f}){note}")
                pk = ra[case].get("peak_over_budget", {})
                pv = ra[case].get("difference_peak", {})
                if pv.get("base") is not None and case not in RMS_NOT_GATED:
                    print(f"    {'':24s} {'':16s} PEAK over time / budget: "
                          f"base "
                          f"{pk.get('base', {}).get(pair, float('nan')):5.2f}"
                          f"   refined "
                          f"{pk.get('refined', {}).get(pair, float('nan')):5.2f}"
                          f"   (peak D {pv.get('base', float('nan')):.3e} -> "
                          f"{pv.get('refined', float('nan')):.3e}) -- the "
                          f"final-time row below is ONE sample of a "
                          f"difference that need not be monotone")
                dob = ra[case].get("difference_over_budget", {})
                bud = ra[case].get("budget", {})
                print(f"    {'':24s} {'':16s} difference / budget: "
                      f"base {dob.get('base', {}).get(pair, float('nan')):5.2f}"
                      f"   refined "
                      f"{dob.get('refined', {}).get(pair, float('nan')):5.2f}"
                      f"   (pair budget "
                      f"{bud.get('base', {}).get(pair, float('nan')):.3e}"
                      f" -> "
                      f"{bud.get('refined', {}).get(pair, float('nan')):.3e}"
                      f")")
                for level in ("base", "refined"):
                    per = ra[case].get("budget_per_arm", {}).get(level, {})
                    if per:
                        print(f"    {'':24s} {'':16s}   {level:8s} per-arm "
                              f"tolerance " + "  ".join(
                                  f"{g} {v:.3e}" for g, v in sorted(per.items())))
                for level in ("base", "refined"):
                    na = lv.get(level, {}).get("native_amplitude", {})
                    rows = [(f, v) for f, v in sorted(na.items())
                            if "ratio" in v]
                    if not rows:
                        continue
                    print(f"    {'':24s} {'':16s} {level:8s} how far each arm "
                          f"moved ON ITS OWN MESH (no regrid in the path) --"
                          f" if these ratios match the common-mesh ones, the "
                          f"regrid is NOT the explanation:")
                    for f, v in rows:
                        per = "  ".join(f"{g} {x:.3e}" for g, x
                                        in sorted(v["per_arm"].items())
                                        if x)
                        print(f"    {'':24s} {'':16s}   {f:10s} "
                              f"{v['ratio']:5.2f}x ({v['ratio_of']})   {per}")
                sig = ra[case].get("signature", {}).get(pair)
                if sig:
                    print(f"    {'':24s} {'':16s} SIGNATURE: {sig}")
                ic = ra[case].get("initial_difference", {}).get(pair)
                if ic and np.isfinite(ic["ratio"]):
                    print(f"    {'':24s} {'':16s} t=0 difference in "
                          f"{CASE_FIELD[case]} only: {ic['base']:.3e} -> "
                          f"{ic['refined']:.3e} ({ic['ratio']:.2f}x) -- "
                          f"CONSTRAINS the initial-condition story, does "
                          f"NOT settle it: it is the headline field alone, "
                          f"on the COMMON mesh, and each arm is driven by "
                          f"its own NATIVE-mesh IC")
                for level in ("base", "refined"):
                    bf = lv.get(level, {}).get("by_field", {})
                    if not bf:
                        continue
                    print(f"    {'':24s} {'':16s} {level:8s} by field, "
                          f"difference / how far that field moved:")
                    for fld, v in sorted(
                            bf.items(),
                            key=lambda kv: -kv[1].get("fraction", 0.0)
                            if np.isfinite(kv[1].get("fraction", np.nan))
                            else 0.0):
                        if "skipped" in v:
                            print(f"    {'':24s} {'':16s}   {fld:10s} "
                                  f"NOT COMPARED — {v['skipped']} "
                                  f"({v['points_at']} is therefore "
                                  f"UNMEASURED, not agreeing)")
                            continue
                        if fld in _NOT_LIKE_FOR_LIKE:
                            print(f"    {'':24s} {'':16s}   {fld:10s} "
                                  f"REPORTED, NOT COMPARABLE — "
                                  f"{_NOT_LIKE_FOR_LIKE[fld]}")
                        mv = "/".join(f"{g} {x:.3e}" for g, x
                                      in sorted(v.get("moved", {}).items()))
                        print(f"    {'':24s} {'':16s}   {fld:10s} "
                              f"{v['fraction']:6.2f}  ({v['difference']:.3e} "
                              f"vs the larger arm motion "
                              f"{v['field_moved']:.3e}; each arm moved "
                              f"{mv})  {v['points_at']}")
                if case in RMS_NOT_GATED:
                    print(f"    {'':24s} {'':16s} NOTE this row's ratio and "
                          f"budget are a SCALAR displacement in km, not a "
                          f"field norm -- not comparable with the other "
                          f"cases' numbers (GLM-5.2 2026-08-13)")
                    print(f"    {'':24s} {'':16s} RMS NOT THE VERDICT here "
                          f"({RMS_NOT_GATED[case]}); front displacement:")
                    for level in ("base", "refined"):
                        fr = lv[level].get("front", {}).get(pair, {})
                        cell = "  ".join(f"{k[:-7]} {v:7.1f} km"
                                         for k, v in sorted(fr.items()))
                        print(f"    {'':24s} {'':16s}   {level:8s} {cell}")
            for level, v in sorted(lv.items()):
                if "skipped" in v:
                    print(f"    {case:24s} {level:16s} SKIPPED — "
                          f"{v['skipped']}")

    print("\nCROSS-GRID CONSISTENCY  (arm-to-arm RMS difference of the final "
          "field, in the field's own units; 0 = identical)")
    print(f"  arms compared: {grids}"
          + ("" if a.include_real_geometry else
             f"   [{', '.join(REAL_GEOMETRY_GRIDS)} excluded: real "
             f"continents, not the aquaplanet basin these arms share]"))
    for case in CASE_FIELD:
        if _find(a.runs_root, case, "latlon") is None:
            continue
        try:
            pairs, ref = cross_grid_rms(a.runs_root, case, grids)
        except ValueError as exc:
            print(f"\n{case}: SKIPPED — {exc}")
            continue
        report["consistency"][case] = dict(pairs=pairs, case_reference=ref)
        _ = ref
        if not pairs:
            continue
        worst = max(pairs, key=lambda k: pairs[k]["rms_abs"])
        print(f"\n{case}  (field {CASE_FIELD[case]}; case reference "
              f"amplitude {ref:.3e})")
        for k, v in sorted(pairs.items()):
            flag = "  <-- worst" if k == worst else ""
            print(f"    {k:30s} rms {v['rms_abs']:9.3e}  "
                  f"shifted {v['rms_shifted']:9.3e} "
                  f"@ {v['shift_deg']:+5.0f}d  "
                  f"t=0 {v['rms_initial']:9.3e}  "
                  f"EVOLUTION {v['rms_evolution']:9.3e}  "
                  f"corr(t0) {v['pattern_corr_with_t0']:+.2f}"
                  f"  ov {v['overlap_frac']:.2f}{flag}")
        # The verdict is on rms_evolution, not on rms_abs vs rms_initial.
        ic_bound = [k for k, v in pairs.items()
                    if np.isfinite(v["rms_evolution"]) and v["rms_abs"] > 0
                    and v["rms_evolution"] < 0.25 * v["rms_abs"]]
        if ic_bound:
            print(f"       NOTE: for {', '.join(sorted(ic_bound))} the "
                  f"DYCORE-INDUCED part (EVOLUTION column) is under a quarter "
                  f"of the raw difference -- that row is mostly "
                  f"discretisation of the shared IC, not the dycores")
        if case in RMS_NOT_GATED:
            print(f"    -> RMS NOT GATED for this case: {RMS_NOT_GATED[case]}")
            report["consistency"][case]["rms_not_gated"] = RMS_NOT_GATED[case]
            _lock_exchange_front_report(a, grids, report, case)
            continue

        tol, provenance = _tolerance(case)
        # PER-PAIR uncertainty budget, not one number for the whole table:
        # a pair containing an arm whose OWN self-error is 9.4x the
        # reference cannot be judged against the reference (codex
        # 2026-08-10 -- the previous revision printed DISAGREE and then a
        # note saying the verdict did not apply).
        # THREE classes, not two. A pair is only ASSESSABLE when BOTH arms
        # have a measured self-error: without one there is no scale to
        # judge that arm against, and holding it to another arm's
        # convergence rate is the mistake this whole block exists to stop.
        # fesom ("pi") and tripole ("eorca1") ship one mesh each, so their
        # rows are permanently unassessable -- said out loud, and kept OUT
        # of the agreement count rather than passed on a borrowed budget
        # (codex round 2 + GLM-5.2).
        budgets, sources, unassessable = {}, {}, []
        for k in pairs:
            ga, gb = k.split("|")
            have = {g: SELF_ERROR.get((g, case)) for g in (ga, gb)}
            if any(e is None for e in have.values()):
                unassessable.append(k)
                continue
            worst_arm = max(have, key=lambda g: have[g])
            budgets[k] = max(tol, have[worst_arm])
            sources[k] = (worst_arm if have[worst_arm] > tol
                          else f"reference ({provenance})")
        # Judged on the EVOLUTION column -- the dycore-induced part -- and
        # the budget is the same quantity measured on one arm against its
        # own refinement. Judging the raw RMS against an evolution budget
        # (or the reverse) is what made a row read "IC-dominated" and
        # "outside budget" at once.
        agree = [k for k in budgets if pairs[k]["rms_evolution"] <= budgets[k]]
        differ = [k for k in budgets if pairs[k]["rms_evolution"] > budgets[k]]
        print(f"    -> {len(agree)}/{len(budgets)} assessable pairs inside "
              f"their own uncertainty budget on EVOLUTION "
              f"(reference {tol:g}, {provenance})"
              + (f"; OUTSIDE: {', '.join(sorted(differ))}" if differ else ""))
        if unassessable:
            print(f"       UNASSESSABLE ({len(unassessable)}): "
                  f"{', '.join(sorted(unassessable))} -- an arm in each has "
                  f"no measurable self-error, so there is no standard to "
                  f"judge it against; NOT counted as agreeing")
        for k in sorted(k for k in budgets if budgets[k] > tol):
            print(f"       budget for {k} widened to {budgets[k]:.3e} by "
                  f"{sources[k]}'s own self-error ({budgets[k] / tol:.1f}x the "
                  f"reference) -- inside it means only 'no worse than that "
                  f"arm's own resolution sensitivity'")
        report["consistency"][case]["agree_tol"] = tol
        report["consistency"][case]["agree_tol_provenance"] = provenance
        report["consistency"][case]["pair_budgets"] = budgets
        report["consistency"][case]["pairs_agreeing"] = sorted(agree)
        report["consistency"][case]["pairs_disagreeing"] = sorted(differ)
        report["consistency"][case]["pairs_unassessable"] = sorted(unassessable)


    print("\n\nTHEORY")
    le = theory_lock_exchange(a.runs_root)
    report["theory"]["lock_exchange"] = le
    print(f"\nlock_exchange — Benjamin (1968) front speed "
          f"0.5*sqrt(g'H) = {le['theory_c_m_s']:.3f} m/s "
          f"(g' = {le['g_prime']:.4f} m/s^2)")
    for g, r in le["arms"].items():
        # The width quoted is the SAVED OUTPUT mesh, which is what the
        # front-crossing is measured on. Every arm's own model mesh here is
        # coarser than or equal to it, so this is the OPTIMISTIC bound.
        flag = ("" if r["resolvable"] else
                "   NOT A MEASUREMENT: theory moves the front only "
                f"{r['theory_cells_on_output_mesh']:.2f} of one "
                f"{r['output_cell_width_km']:.0f} km OUTPUT cell over this "
                f"run (model mesh {_registered_resolution('lock_exchange', g)})")
        ratio = (f"({r['ratio_to_theory']:.2f} x theory)"
                 if r["ratio_to_theory"] is not None else "(ratio REFUSED)")
        print(f"    {g:13s} moved {r['front_deg_moved']:5.2f} deg -> "
              f"{r['speed_m_s']:.3f} m/s  {ratio}{flag}")
    if not any(r["resolvable"] for r in le["arms"].values()):
        print("    -> NO arm here can resolve the Benjamin speed. The ratios "
              "above are the sub-cell smearing of a temperature step, not a "
              "front speed, and must not be quoted as a dycore result. The "
              "resolved test is the latlon_regional Petersen channel "
              "(1 km cells); --convergence shows the deficit shrinking under "
              "refinement.")
    bw = theory_barotropic_wave(a.runs_root)
    report["theory"]["barotropic_wave"] = bw
    print(f"\nbarotropic_wave — sqrt(gH) = {bw['theory_c_m_s']:.1f} m/s "
          f"[{bw['note']}]")
    for g, r in bw["arms"].items():
        print(f"    {g:13s} peak|eta| {r['peak_eta_initial']:.3f} -> "
              f"{r['peak_eta_final']:.3f} m")
    for case in ("rest_state_stratified_with_land",
                 "rest_state_uniform_with_land"):
        rs = theory_rest_state(a.runs_root, case)
        report["theory"][case] = rs
        print(f"\n{case} — {rs['theory']}")
        for g, r in rs["arms"].items():
            star = ("" if r["u_is_cell_shaped"]
                    else " (u on faces; NaN = no cell-shaped u saved)")
            print(f"    {g:13s} max|u| {r['max_abs_u_final']:.3e} m/s   "
                  f"max|eta| {r['max_abs_eta_final']:.3e} m{star}")

    if a.convergence:
        conv = theory_lock_exchange_convergence()
        report["theory"]["lock_exchange_convergence"] = conv
        print(f"\nlock_exchange CONVERGENCE — does the front approach "
              f"Benjamin {conv['theory_c_m_s']:.3f} m/s as dx shrinks?")
        for res, r in conv["arms"].items():
            print(f"    {res:9s} dx={r['dx_km']:6.1f} km  "
                  f"{r['speed_m_s']:.4f} m/s  "
                  f"({r['ratio_to_theory']:.3f} x theory)")

    print("\n\nCONSISTENCY-ONLY (no defensible closed form on this geometry)")
    for case, why in CONSISTENCY_ONLY.items():
        report["consistency_only"][case] = why
        print(f"    {case}: {why}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nCOMPLETED: {a.out}")


if __name__ == "__main__":
    main()
