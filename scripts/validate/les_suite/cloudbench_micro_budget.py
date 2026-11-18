"""Does the LES microphysics source close the CloudBench day-5 budget residual?

The jump audit (``cloudbench_jump_audit.py``) left a day-5 LES budget residual
that has no microphysics term, because the artifacts do not carry one. The public
stores do: ``theta_li_microphysics_source`` and ``q_t_microphysics_source``. Per
case this streams their horizontal means on every ``FRAME_STRIDE``-th day-5 frame (default every frame), time-weighted by the
trapezoid rule as the residual is,
and reports, per height band, the artifact budget residual (``les_budget``) and
the microphysics source's band mean under two readings: per unit mass, and per
unit volume (divided cell by cell by rho before the horizontal mean). It also reports the
residual and the recorded nudging term side by side. Numbers only, no verdict.

usage: cloudbench_micro_budget.py OUT_DIR CASE [CASE ...]   (CASE = sSSS_mMM)
Run with JAX_ENABLE_X64=1 on a compute node.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cosci_eval as ce  # noqa: E402
from cloudbench_jump_audit import ART, BANDS, FORCING, les_budget  # noqa: E402
from legoesm.atmosphere.les_suite.cloudbench import (  # noqa: E402
    STORE_URL,
    parse_case_name,
    read_sounding,
)

import os

FRAME_STRIDE = int(os.environ.get("MICRO_FRAME_STRIDE", "1"))
VARS = ("theta_li_microphysics_source", "q_t_microphysics_source", "rho")


def store_means(url, frames):
    """Horizontal means per frame on the store's physical heights (node coordinate
    + half a cell, as the artifact builder places them), the frame times [s], and
    the per-volume reading mean(source / rho) taken cell by cell."""
    ds = xr.open_zarr(url, consolidated=True)
    zc = np.asarray(ds["z"].values, dtype=np.float64)
    z = zc + 0.5 * (zc[1] - zc[0])
    t = np.asarray(ds["t"].values).astype(np.float64) / 1e9

    def one(i):
        f = ds.isel(t=i)
        out = {v: np.asarray(f[v].mean(("x", "y")).values, dtype=np.float64) for v in VARS}
        for v in VARS[:2]:
            out[v + "/rho"] = np.asarray((f[v] / f["rho"]).mean(("x", "y")).values,
                                         dtype=np.float64)
        return out
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, frames))
    return z, t[frames], {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def case_report(case):
    path = ART / f"cloudbench_{case}_amip.npz"
    art = ce.load_artifact(path)
    site = parse_case_name(art.case_name)[0]
    par = art.les_setup["parameters"]
    sounding = read_sounding(FORCING / str(site) / str(int(par["month"]))
                             / par["experiment"] / "sounding.csv")
    lb = les_budget(art, sounding)
    nt = len(art.times_s) - 1
    frames = list(range(0, nt, FRAME_STRIDE))
    url = STORE_URL.format(site=site, month=int(par["month"]), exp=par["experiment"])
    z_store, t_store, m = store_means(url, frames)
    z_art = np.asarray(art.heights_m, dtype=np.float64)

    def tmean(a):
        if len(t_store) == 1:
            return a[0]
        return np.trapezoid(a, t_store, axis=0) / (t_store[-1] - t_store[0])
    rho_art = np.interp(z_art, sounding["z"], sounding["rho"])
    out = {"url": url, "frames_used": frames, "regime": ce._regime(art.case_name),
           "artifact_heights_off_store_nodes_m": float(np.max(np.abs(
               z_art - z_store[np.argmin(np.abs(z_store[:, None] - z_art[None]), axis=0)]))),
           "bands": {}}
    for key, var, sc in (("theta_l", "theta_li_microphysics_source", 86400.0),
                         ("q_t", "q_t_microphysics_source", 86400e3)):
        per_mass = np.interp(z_art, z_store, tmean(m[var]))
        per_vol = np.interp(z_art, z_store, tmean(m[var + "/rho"]))
        for lo, hi in BANDS:
            msk = (z_art >= lo) & (z_art < hi)
            wt = rho_art[msk]
            b = lb[key]["bands"][f"{lo}-{hi}"]
            out["bands"].setdefault(f"{lo}-{hi}", {})[key] = {
                "residual_no_micro": b["residual"], "nudging": b["nudging"],
                "micro_per_mass": float((per_mass[msk] * wt).sum() / wt.sum() * sc),
                "micro_per_volume": float((per_vol[msk] * wt).sum() / wt.sum() * sc)}
    return out


def main() -> int:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    arts = [(str(ART / f"cloudbench_{c}_amip.npz"),
             ce.load_artifact(ART / f"cloudbench_{c}_amip.npz")) for c in sys.argv[2:]]
    ce.configure_cloudbench(arts, FORCING, Path(
        "/glade/derecho/scratch/pg2328/les_suite/external/cloudbench/cloudbench_statistics.nc"))
    res = {"git_sha": ce._code_id(), "frame_stride": FRAME_STRIDE, "cases": {}}
    for c in sys.argv[2:]:
        try:
            res["cases"][c] = case_report(c)
        except Exception as err:   # a failed fetch is recorded, never silently skipped
            res["cases"][c] = {"error": repr(err)}
        print(c, json.dumps(res["cases"][c].get("bands", res["cases"][c]).get("3000-4000",
              res["cases"][c].get("error"))), flush=True)
        (out_dir / "micro_budget.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
