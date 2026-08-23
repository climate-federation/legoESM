"""Global tracer-content probe for tripole (eORCA1) OMIP snapshots.

Why this exists
---------------
The MLE d90 arm (job 9337800) shows surface-mean SSS falling 34.25 -> 29.71
over 90 days.  ``mean_sss`` in ``diag_timeseries.csv`` is a SURFACE mean, so
that curve cannot distinguish the two candidate mechanisms:

* a conservation defect  -- the volume-integrated salt content falls; or
* vertical redistribution -- total salt is conserved and the surface freshens
  because salt is displaced downward (over-restratification traps fresh water
  at the surface).

This probe computes the discriminating number: the volume integral of S (and
T) on the NATIVE eORCA1 mesh, per snapshot, plus a per-depth-bin breakdown so
a redistribution shows up as compensating bin changes at constant total.

Per docs/ocean/fidelity/fidelity_to_fesom2jax_level_plan.md (#1492) Phase 0.3
this is a COMMITTED probe with locked conventions and provenance stamped into
its JSON output; inline heredoc probes are not citable.

Conventions (locked)
--------------------
* fp64 throughout (numpy f8; no JAX import needed).
* Snapshot arrays are (nlev, 332, 362) = eORCA1 (331, 360) plus one north-fold
  ghost row (row 331 is the reversed ghost of row 330) and a cyclic east-west
  overlap column on each side.  The native frame is ``[:, 0:331, 1:361]``
  (offset j0=0, i0=1, verified to max|dlat| = 1.4e-14 deg against NEMO
  nav_lat -- see memory omip-arctic-two-defects-native-mesh).
* WET = ``tmask`` from the NEMO mesh_mask (per level), NOT the snapshot's
  ``land_mask`` (which stores 1.0 on OCEAN but is 2-D only).
* Cell volume = ``e1t * e2t * e3t_0`` (z-coordinate reference thickness),
  scaled per column by the z-star dilation ``(H + eta) / H`` so a volume
  change via the free surface is not misread as a tracer-content change.
* Global integrals are plain sums over wet native cells -- the mesh metrics
  ARE the area weights on this distorted grid (never cos(lat) here).

Usage:
    python scripts/validate/ocean_fidelity/global_tracer_content.py \
        --snapshot results/omip_nemo/<run>/snapshot_day0090.npz [more ...] \
        --mesh-mask data/grids/eORCA1.2_mesh_mask.nc \
        --json-out results/omip_nemo/<run>/tracer_content.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

# Depth-bin edges [m] for the redistribution breakdown: surface mixed layer,
# thermocline, intermediate, deep.
_DEPTH_BIN_EDGES_M = (0.0, 50.0, 200.0, 1000.0, np.inf)

# eORCA1 native frame inside the snapshot arrays: rows 0..330 (row 331 is the
# reversed north-fold ghost), columns 1..360 (cyclic overlap on both sides).
_NATIVE_J = slice(0, 331)
_NATIVE_I = slice(1, 361)


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        return "unknown"


def load_mesh_metrics(mesh_mask_path):
    """(e1t, e2t, e3t_0, tmask) on the native (nlev, 331, 360) frame, f8."""
    try:
        import netCDF4 as nc
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"netCDF4 required to read the mesh mask: {exc}")
    ds = nc.Dataset(mesh_mask_path)
    try:
        def v(name):
            return np.asarray(ds.variables[name][:], dtype=np.float64).squeeze()
        e1t, e2t = v("e1t"), v("e2t")
        # e3t_0 is the 3-D reference thickness with partial cells; some mesh
        # files only carry the 1-D e3t_1d -- refuse those rather than
        # silently ignoring partial cells.
        if "e3t_0" not in ds.variables:
            raise SystemExit("mesh mask has no 3-D e3t_0; refusing to fake "
                             "partial cells from e3t_1d")
        e3t = v("e3t_0")
        tmask = v("tmask")
    finally:
        ds.close()
    if e3t.ndim != 3 or tmask.ndim != 3:
        raise SystemExit(f"expected 3-D e3t_0/tmask, got {e3t.shape}/{tmask.shape}")
    # The mesh file carries the FULL (jpj=332, jpi=362) domain including the
    # cyclic overlap columns and the north-fold ghost row; summing them would
    # double-count.  Slice to the native frame here so every consumer sees
    # only unique physical cells.
    if e3t.shape[1:] != (332, 362):
        raise SystemExit(f"expected eORCA1 (nlev, 332, 362) mesh, got {e3t.shape}")
    return (e1t[_NATIVE_J, _NATIVE_I], e2t[_NATIVE_J, _NATIVE_I],
            e3t[:, _NATIVE_J, _NATIVE_I], tmask[:, _NATIVE_J, _NATIVE_I])


def load_mesh_latitude(mesh_mask_path):
    """``gphit`` on the SAME native (331, 360) frame as load_mesh_metrics.

    Separate loader so a caller that needs a geographic region (e.g. an Arctic
    integral) gets the latitude on the identical frame, instead of slicing the
    full (332, 362) field itself and risking a one-row offset against the
    metrics.
    """
    try:
        import netCDF4 as nc
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"netCDF4 required to read the mesh mask: {exc}")
    ds = nc.Dataset(mesh_mask_path)
    try:
        gphit = np.asarray(ds.variables["gphit"][:], dtype=np.float64).squeeze()
    finally:
        ds.close()
    if gphit.shape != (332, 362):
        raise SystemExit(f"expected eORCA1 (332, 362) gphit, got {gphit.shape}")
    return gphit[_NATIVE_J, _NATIVE_I]


def load_mesh_longitude(mesh_mask_path):
    """``glamt`` on the SAME native (331, 360) frame as load_mesh_latitude.

    Wrapped to NEMO's -180..180 convention, which is the convention the
    lon-boxed equatorial regions (nino3/nino4) are written in.  ``glamt`` on
    eORCA1 is already in that range, but the wrap is applied unconditionally
    so a mesh written on 0..360 selects the same water rather than an empty
    box -- a silently-empty region is the failure this loader exists to avoid.
    """
    try:
        import netCDF4 as nc
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"netCDF4 required to read the mesh mask: {exc}")
    ds = nc.Dataset(mesh_mask_path)
    try:
        glamt = np.asarray(ds.variables["glamt"][:], dtype=np.float64).squeeze()
    finally:
        ds.close()
    if glamt.shape != (332, 362):
        raise SystemExit(f"expected eORCA1 (332, 362) glamt, got {glamt.shape}")
    return (glamt[_NATIVE_J, _NATIVE_I] + 180.0) % 360.0 - 180.0


def load_mesh_depth_1d(mesh_mask_path):
    """``gdept_1d`` as a plain (nlev,) array of level-centre depths [m]."""
    try:
        import netCDF4 as nc
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"netCDF4 required to read the mesh mask: {exc}")
    ds = nc.Dataset(mesh_mask_path)
    try:
        g = np.asarray(ds.variables["gdept_1d"][:], dtype=np.float64).squeeze()
    finally:
        ds.close()
    if g.ndim != 1:
        raise SystemExit(f"expected 1-D gdept_1d, got {g.shape}")
    return g


def tracer_content(T3d, S3d, eta, e1t, e2t, e3t, tmask, gdept=None,
                   region_mask=None):
    """Volume integrals of T and S with the z-star column dilation.

    All inputs on the SAME native frame.  Returns a dict of plain floats.
    """
    T3d = np.asarray(T3d, dtype=np.float64)
    S3d = np.asarray(S3d, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    wet = tmask > 0.5
    if region_mask is not None:
        # Restrict the integral to a horizontal region (e.g. Arctic, or the
        # open-water / ice-covered split of it).  Applied to the WET mask so
        # every downstream quantity -- volume, salt, the depth bins -- is
        # restricted consistently and no term is left global by accident.
        wet = wet & (np.asarray(region_mask, dtype=bool)[None] if
                     np.ndim(region_mask) == 2 else
                     np.asarray(region_mask, dtype=bool))
    dV_ref = e1t[None] * e2t[None] * e3t * wet          # (nlev, nj, ni)
    H = (e3t * wet).sum(axis=0)                          # column depth [m]
    col_wet = H > 0.0
    dilation = np.where(col_wet, (H + eta * col_wet) / np.where(col_wet, H, 1.0), 0.0)
    dV = dV_ref * dilation[None]

    if not wet.any():
        # An empty region is a CALLER error (bad lat bounds, mask on the wrong
        # frame).  Returning zeros would be a plausible-looking number that
        # passes every finite check downstream -- the exact failure this
        # campaign has already paid for once.
        raise SystemExit("FATAL: region selects zero wet cells")
    if not np.isfinite(T3d[wet]).all() or not np.isfinite(S3d[wet]).all():
        # NaN in a wet cell must be FATAL, not silently nansum'd away.
        n_bad = int((~np.isfinite(T3d[wet])).sum() + (~np.isfinite(S3d[wet])).sum())
        raise SystemExit(f"FATAL: {n_bad} non-finite wet tracer values")

    out = {
        "volume_m3": float(dV.sum()),
        "volume_ref_m3": float(dV_ref.sum()),
        "salt_content_psu_m3": float((S3d * dV).sum()),
        "heat_content_C_m3": float((T3d * dV).sum()),
        "mean_S_psu": float((S3d * dV).sum() / dV.sum()),
        "mean_T_C": float((T3d * dV).sum() / dV.sum()),
    }
    # Per-depth-bin salt content: a pure vertical redistribution keeps the
    # TOTAL fixed while the bins trade against each other.
    if gdept is not None:
        bins = {}
        for lo, hi in zip(_DEPTH_BIN_EDGES_M[:-1], _DEPTH_BIN_EDGES_M[1:]):
            sel = (gdept >= lo) & (gdept < hi)
            bins[f"{lo:g}-{hi:g}m"] = float((S3d * dV * sel[:, None, None]).sum())
        out["salt_by_depth_bin_psu_m3"] = bins
    return out


def _native(a):
    """Strip the snapshot's fold ghost row and cyclic overlap columns, and put
    the level axis FIRST.

    Snapshot 3-D arrays are (nj=332, ni=362, nlev) -- level LAST (measured,
    job 9361671: T.shape == (332, 362, 75)); the mesh metrics are level-first.
    The shapes are asserted rather than inferred so a layout change fails
    loudly instead of silently mis-slicing (the array's layout is an API)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2:
        if a.shape != (332, 362):
            raise SystemExit(f"expected 2-D snapshot (332, 362), got {a.shape}")
        return a[_NATIVE_J, _NATIVE_I]
    if a.ndim == 3:
        if a.shape[:2] != (332, 362):
            raise SystemExit(f"expected 3-D snapshot (332, 362, nlev), got {a.shape}")
        return np.transpose(a[_NATIVE_J, _NATIVE_I, :], (2, 0, 1))
    raise SystemExit(f"expected 2-D or 3-D snapshot array, got shape {a.shape}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", nargs="+", required=True,
                   help="One or more tripole snapshot .npz files (scored in order).")
    p.add_argument("--mesh-mask", required=True,
                   help="eORCA1.2_mesh_mask.nc with e1t/e2t/e3t_0/tmask/gdept_1d.")
    p.add_argument("--json-out", default=None)
    a = p.parse_args()

    e1t, e2t, e3t, tmask = load_mesh_metrics(a.mesh_mask)
    # Bin by the 1-D reference depth of each level (fine for binning; partial
    # cells only matter in the bottom bin boundary, and every bin is reported).
    try:
        import netCDF4 as nc
        ds = nc.Dataset(a.mesh_mask)
        gdept = np.asarray(ds.variables["gdept_1d"][:], dtype=np.float64).squeeze()
        ds.close()
    except Exception:
        gdept = None

    report = {"generated_by": str(Path(__file__).resolve()),
              "git_sha": _git_sha(Path(__file__).resolve().parents[3]),
              "mesh_mask": str(a.mesh_mask),
              "conventions": "native frame [0:331,1:361]; e1t*e2t*e3t_0*tmask; "
                             "z-star dilation (H+eta)/H; fp64",
              "snapshots": {}}
    prev = None
    for snap in a.snapshot:
        z = np.load(snap)
        for k in ("T", "S", "eta"):
            if k not in z:
                raise SystemExit(f"FATAL: {snap} lacks '{k}'")
        T3, S3, et = _native(z["T"]), _native(z["S"]), _native(z["eta"])
        if T3.shape != e3t.shape:
            raise SystemExit(f"FATAL: snapshot native frame {T3.shape} != "
                             f"mesh {e3t.shape}; wrong mesh or wrong slicing")
        r = tracer_content(T3, S3, et, e1t, e2t, e3t, tmask, gdept)
        report["snapshots"][str(snap)] = r
        line = (f"{Path(snap).parent.name}/{Path(snap).name}: "
                f"salt {r['salt_content_psu_m3']:.6e} psu m3, "
                f"mean S {r['mean_S_psu']:.4f}, vol {r['volume_m3']:.6e} m3")
        if prev is not None:
            ds_rel = (r["salt_content_psu_m3"] / prev - 1.0)
            line += f"  (salt vs previous: {ds_rel:+.3e} relative)"
        prev = r["salt_content_psu_m3"]
        print(line)

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(report, indent=2))
        print(f"[report] {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
