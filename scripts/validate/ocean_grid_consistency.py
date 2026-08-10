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

Run AFTER the suite; reads only saved artifacts (no model runs).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from legoesm import constants as C

_REPO = Path(__file__).resolve().parents[2]

#: Arms compared cell-by-cell. tripole excluded -- different basin.
CONSISTENCY_GRIDS = ["cubed_sphere", "latlon", "mpas", "fesom"]
ALL_GRIDS = CONSISTENCY_GRIDS + ["tripole"]

#: Arm-to-arm agreement threshold, in the FIELD'S OWN UNITS, set from the
#: INTRINSIC DISCRETISATION ERROR rather than by hand.
#:
#: Two different dycores cannot agree more closely than one dycore agrees
#: with ITSELF across a resolution change of the size that separates the
#: arms. That self-difference is the achievable floor, so it is the
#: standard. MEASURED 2026-08-10 (lat-lon C-grid, 36x72 vs 72x144, ocean
#: cells only, same IC and dt):
#:     geostrophic_adjustment  SST  8.20e-2 degC
#:     barotropic_wave         eta  2.34e-2 m
#: The thresholds below are those numbers rounded up. An earlier revision
#: used a hand-picked 1e-3 in both fields, which no pair of independent
#: discretisations can meet and which therefore reported "DISAGREE" for
#: everything -- a threshold set before the measurement, the exact habit
#: the reviewers flagged twice in this campaign.
AGREE_TOL = {"SST": 1.0e-1, "eta": 2.5e-2}

#: PER-DYCORE self-error, same protocol (that arm against ITSELF at 2x
#: resolution). The tolerance above is derived from LAT-LON and is only
#: fair to an arm whose own self-error is comparable. MEASURED
#: 2026-08-10:
#:     latlon 36x72 vs 72x144   geostrophic SST 8.20e-2 | bwave eta 2.34e-2
#:     cube   C24   vs C48      geostrophic SST 9.45e-1 | bwave eta 2.23e-2
#: Cube's own self-error on geostrophic adjustment is 9.4x the lat-lon
#: tolerance, so "cube disagrees with everyone" on that case is NOT
#: supportable -- the standard is unfair to it. On the barotropic wave
#: cube's self-error (2.23e-2) is inside the tolerance, so ITS cross-arm
#: difference there is a genuine dycore difference. FESOM cannot be
#: measured this way: only the "pi" mesh ships, so it has no 2x sibling.
#: (GLM-5.2 review: without this, "failing" and "held to another arm's
#: convergence rate" are indistinguishable.)
SELF_ERROR = {
    ("latlon", "geostrophic_adjustment"): 8.20e-2,
    ("latlon", "barotropic_wave"): 2.34e-2,
    ("cubed_sphere", "geostrophic_adjustment"): 9.45e-1,
    ("cubed_sphere", "barotropic_wave"): 2.23e-2,
}

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


def cross_grid_rms(root: Path, case: str, grids=CONSISTENCY_GRIDS):
    """Arm-to-arm difference of the final field, in the field's own units.

    Reported per pair:
      rms_abs        area-weighted RMS difference on the COMMON wet cells
      rms_shifted    the same after removing the best zonal shift, and
      shift_deg      that shift. A dispersive wave that two arms propagate
                     at slightly different speeds shows a large rms_abs and
                     a much smaller rms_shifted: that is a PHASE difference,
                     not different physics. Equal values mean the fields
                     genuinely differ (GLM-5.2).
      overlap_frac   fraction of the mesh both arms call ocean -- pairs with
                     different overlap are NOT directly comparable, so the
                     number is printed rather than hidden.

    A case-level reference amplitude (independent of any pair) is returned
    separately so the differences can be read against one fixed scale.
    """
    field = CASE_FIELD[case]
    got = {}
    for g in grids:
        p = _find(root, case, g)
        if p is None:
            continue
        lat, lon, a, m, t = _load(p, field)
        lat_t, lon_t, A, M = _to_common_mesh(lat, lon, a[-1], m)
        got[g] = (A, M > 0.5)
    if not got:
        return {}, float("nan")
    # ONE case-level reference: the area-weighted anomaly RMS of the arm
    # with the largest wet area, chosen without reference to any pair.
    w_lat = np.cos(np.radians(np.linspace(-89.0, 89.0, 91)))[:, None]
    ref_name = max(got, key=lambda g: got[g][1].sum())
    Aref, Mref = got[ref_name]
    wr = np.where(Mref & np.isfinite(Aref), w_lat, 0.0)
    mean_ref = float(np.sum(np.where(wr > 0, Aref, 0.0) * wr) / max(wr.sum(), 1e-30))
    case_ref = float(np.sqrt(np.sum(wr * (np.where(wr > 0, Aref, mean_ref)
                                          - mean_ref) ** 2)
                             / max(wr.sum(), 1e-30)))

    def _wrms(d, w):
        return float(np.sqrt(np.sum(w * d ** 2) / max(w.sum(), 1e-30)))

    out = {}
    for a_name, b_name in itertools.combinations(sorted(got), 2):
        A, mA = got[a_name]
        B, mB = got[b_name]
        both = _erode(mA & mB & np.isfinite(A) & np.isfinite(B))
        # (see _erode)
        if not both.any():
            out[f"{a_name}|{b_name}"] = dict(rms_abs=float("nan"),
                                             rms_shifted=float("nan"),
                                             shift_deg=float("nan"),
                                             overlap_frac=0.0)
            continue
        w = np.where(both, np.broadcast_to(w_lat, A.shape), 0.0)
        rms = _wrms(np.where(both, A - B, 0.0), w)
        # Cheapest phase/physics discriminator: minimise over a rigid
        # zonal shift (one roll per candidate; the mesh is 2 deg).
        best, best_shift = rms, 0.0
        n_lon = A.shape[1]
        for k in range(1, n_lon):
            Bk = np.roll(B, k, axis=1)
            mk = _erode(mA & np.roll(mB, k, axis=1)
                        & np.isfinite(A) & np.isfinite(Bk))
            if not mk.any():
                continue
            wk = np.where(mk, np.broadcast_to(w_lat, A.shape), 0.0)
            r = _wrms(np.where(mk, A - Bk, 0.0), wk)
            if r < best:
                best, best_shift = r, (k if k <= n_lon // 2 else k - n_lon) * 2.0
        out[f"{a_name}|{b_name}"] = dict(
            rms_abs=rms, rms_shifted=best, shift_deg=best_shift,
            overlap_frac=float(both.sum() / both.size))
    return out, case_ref


def theory_lock_exchange(root: Path, grids=ALL_GRIDS):
    """Gravity-current front speed vs Benjamin (1968) c = 0.5 sqrt(g' H).

    The front is tracked as the longitude where the equatorial SST crosses
    the mid-temperature, measured from its initial position.
    """
    from legoesm.ocean.experiments.lock_exchange import LockExchangeConfig
    cfg = LockExchangeConfig()
    g_prime = C.g * 2.0e-4 * (cfg.T_warm_C - cfg.T_cold_C)   # linear EOS
    c_theory = 0.5 * np.sqrt(g_prime * cfg.H_max)
    rows = {}
    for grid in grids:
        p = _find(root, "lock_exchange", grid)
        if p is None:
            continue
        lat, lon, a, m, t = _load(p, "SST")
        j = int(np.argmin(np.abs(lat)))            # equatorial row
        T_mid = 0.5 * (cfg.T_cold_C + cfg.T_warm_C)
        def _front(k):
            """Longitude where the equatorial SST crosses T_mid, with
            LINEAR SUB-CELL interpolation. Cell-index precision is not
            enough: theory predicts ~1.9 deg of travel in 5 days against a
            1-2 deg output mesh, so a nearest-cell front reports exactly
            zero motion on every arm (measured -- that was this probe's
            first, wrong, answer)."""
            row = np.where(m[j, :] > 0.5, a[k, j, :], np.nan)
            if np.isfinite(row).sum() < 4:
                return np.nan
            d = row - T_mid
            # Descending (warm -> cold) crossings, the front of interest.
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
        rows[grid] = dict(front_deg_moved=dx_deg, speed_m_s=speed,
                          ratio_to_theory=speed / c_theory)
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
    ap.add_argument("--include-tripole", action="store_true",
                    help="also compare tripole cell-by-cell (it has a "
                         "different basin; the number measures the land "
                         "mask as much as the dycore)")
    a = ap.parse_args()
    if not a.runs_root.is_dir():
        raise SystemExit(f"no suite output under {a.runs_root}")
    grids = (ALL_GRIDS if a.include_tripole else CONSISTENCY_GRIDS)

    report = {"consistency": {}, "theory": {}, "consistency_only": {}}
    print("CROSS-GRID CONSISTENCY  (arm-to-arm RMS difference of the final "
          "field, in the field's own units; 0 = identical)")
    print(f"  arms compared: {grids}"
          + ("" if a.include_tripole else "   [tripole excluded: different basin]"))
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
                  f"shift-corrected {v['rms_shifted']:9.3e} "
                  f"@ {v['shift_deg']:+5.0f} deg  "
                  f"overlap {v['overlap_frac']:.2f}{flag}")
        tol = AGREE_TOL[CASE_FIELD[case]]
        agree = [k for k, v in pairs.items() if v["rms_abs"] <= tol]
        differ = [k for k, v in pairs.items() if v["rms_abs"] > tol]
        print(f"    -> {len(agree)}/{len(pairs)} pairs agree to "
              f"{tol:g} (field units)"
              + (f"; DISAGREE: {', '.join(sorted(differ))}" if differ else ""))
        # Name any arm whose OWN self-error exceeds this tolerance: its
        # "disagreement" is not interpretable against this standard.
        unfair = sorted({g for (g, c), e in SELF_ERROR.items()
                         if c == case and e > tol
                         and any(g in d for d in differ)})
        if unfair:
            print(f"       NOTE: {', '.join(unfair)} has a self-error at 2x "
                  f"resolution ABOVE this tolerance -- its rows are not "
                  f"interpretable here (the standard is another arm's)")
        report["consistency"][case]["agree_tol"] = tol
        report["consistency"][case]["pairs_agreeing"] = sorted(agree)
        report["consistency"][case]["pairs_disagreeing"] = sorted(differ)

    print("\n\nTHEORY")
    le = theory_lock_exchange(a.runs_root)
    report["theory"]["lock_exchange"] = le
    print(f"\nlock_exchange — Benjamin (1968) front speed "
          f"0.5*sqrt(g'H) = {le['theory_c_m_s']:.3f} m/s "
          f"(g' = {le['g_prime']:.4f} m/s^2)")
    for g, r in le["arms"].items():
        print(f"    {g:13s} moved {r['front_deg_moved']:5.2f} deg -> "
              f"{r['speed_m_s']:.3f} m/s  "
              f"({r['ratio_to_theory']:.2f} x theory)")
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
