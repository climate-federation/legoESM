#!/usr/bin/env python3
"""Zonal-mean pressure cross-sections for a pair of arms, against ERA5.

The maps show where a change lands horizontally; they cannot show which LAYER
moved, and a moisture correction that fixes the column total by drying the
wrong level is not a fix.  One row per variable: the control's bias against
ERA5, the arm's bias, and the arm minus the control, all zonal means on the
published pressure levels.

Usage: pair_cross_section.py --ctl <run> --arm <run> [--vars hus ta] --out DIR
"""
from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402

SPEC = {  # var: (scale, units, symmetric bias limit, symmetric delta limit, name)
    "hus": (1000.0, "g/kg", 3.0, 1.0, "specific humidity"),
    "ta": (1.0, "K", 6.0, 1.5, "temperature"),
    "ua": (1.0, "m/s", 10.0, 2.0, "zonal wind"),
}


def _era5_zonal(var, months, mlat, mplev):
    """ERA5 zonal-mean climatology on the model's latitudes and levels.

    The shared ``_ref_clim`` bins a 2-D field onto the model grid and asserts
    a (lat, lon) shape, so it cannot carry a pressure-level variable.  A zonal
    mean does not need that machinery: average the reference over longitude
    first, then interpolate in latitude and in log-pressure.  Interpolating in
    LOG pressure matters -- humidity falls roughly exponentially with height,
    and linear-in-pressure interpolation across the published levels would
    bias the reference moist aloft.
    """
    import glob
    import xarray as xr
    fs = sorted(glob.glob(f"{rb.ERA5}/{var}/*.nc"))
    if not fs:
        return None
    d = xr.open_mfdataset(fs, combine="by_coords") if len(fs) > 1 \
        else xr.open_dataset(fs[0])
    if var not in d:
        return None
    v = d[var].sel(time=slice(f"{rb.REF_MIN_YEAR}-01-01", None))
    if v.time.size == 0:
        return None
    # Zonal mean FIRST: this is a cross-section, so longitude is collapsed
    # either way, and averaging it before the monthly climatology turns a
    # decades-long 4-D read into a 3-D one.  Leaving it last made the figure
    # take longer to draw than the model took to simulate the month.
    clim = (v.mean("lon").groupby("time.month").mean("time")
            .sel(month=months).mean("month").load())
    rlat = np.asarray(clim["lat"], dtype=np.float64)
    rp = np.asarray(clim["plev"], dtype=np.float64)
    arr = np.asarray(clim, dtype=np.float64)
    if arr.shape != (rp.size, rlat.size):
        arr = arr.T
    if rlat[0] > rlat[-1]:                       # ascending for np.interp
        rlat, arr = rlat[::-1], arr[:, ::-1]
    order = np.argsort(rp)
    rp, arr = rp[order], arr[order]
    out = np.empty((mplev.size, mlat.size))
    lat_interp = np.stack([np.interp(mlat, rlat, row) for row in arr])
    for j in range(mlat.size):
        out[:, j] = np.interp(np.log(mplev), np.log(rp), lat_interp[:, j])
    return out


def zonal(run, var, with_ref=False):
    """Zonal-mean field, and the ERA5 zonal mean only when asked for.

    The reference is OPT-IN because the pressure-level ERA5 store is a single
    82 GB file per variable and building a climatology from it takes longer
    than everything else here combined.  The panel this figure exists for --
    which LAYER the change moved -- needs no reference, so the default
    compares the two runs and says so, rather than quietly substituting a
    shorter averaging window, which would be an unrecorded choice of
    reference period.
    """
    d = rb._load_model(run, var)
    if d is None:
        return None
    lat = np.asarray(d.lat, dtype=np.float64)
    plev = np.asarray(d["plev"], dtype=np.float64)
    o = _era5_zonal(var, rb._month_labels(d), lat, plev) if with_ref else None
    m = np.asarray(d[var]).mean(0).mean(-1)      # time mean, then zonal mean
    return m, o, lat, plev / 100.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctl", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--vars", nargs="+", default=["hus", "ta"])
    ap.add_argument("--out", default=".")
    ap.add_argument("--ctl-label", default=None)
    ap.add_argument("--arm-label", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--with-reference", action="store_true",
                    help="add the ERA5 bias panels (slow: the pressure-level "
                         "reference is an 82 GB file per variable)")
    a = ap.parse_args(argv)

    rows = [v for v in a.vars if v in SPEC and zonal(a.ctl, v) is not None]
    if not rows:
        raise SystemExit("FATAL: neither run publishes any requested variable "
                         "on pressure levels -- refusing to write an empty figure")
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.1 * len(rows)),
                             squeeze=False)
    cl = a.ctl_label or a.ctl
    al = a.arm_label or a.arm
    for r, var in enumerate(rows):
        scale, unit, blim, dlim, name = SPEC[var]
        mc, o, lat, p = zonal(a.ctl, var, a.with_reference)[:4]
        ma = zonal(a.arm, var, False)[0]
        if o is not None:
            panels = [((mc - o) * scale, blim, f"{cl}\nminus ERA5"),
                      ((ma - o) * scale, blim, f"{al}\nminus ERA5"),
                      ((ma - mc) * scale, dlim, "corrected minus control")]
        else:
            # No reference: the two states and their difference.  The first
            # two get a full-field scale and say "absolute" so no reader takes
            # them for biases.
            fl = float(np.nanmax(np.abs(mc * scale)))
            panels = [(mc * scale, fl, f"{cl}\n(absolute field)"),
                      (ma * scale, fl, f"{al}\n(absolute field)"),
                      ((ma - mc) * scale, dlim, "corrected minus control")]
        for c, (field, lim, sub) in enumerate(panels):
            ax = axes[r][c]
            h = ax.contourf(lat, p, field, levels=np.linspace(-lim, lim, 21),
                            cmap="RdBu_r", extend="both")
            ax.invert_yaxis()
            ax.set_yscale("log")
            ax.set_yticks([1000, 700, 500, 300, 200, 100])
            ax.set_yticklabels(["1000", "700", "500", "300", "200", "100"])
            ax.set_xlabel("latitude")
            if c == 0:
                ax.set_ylabel(f"{name}\npressure [hPa]")
            ax.set_title(sub, fontsize=10)
            fig.colorbar(h, ax=ax, label=unit)
    if a.title:
        fig.suptitle(a.title, fontsize=13)
    fig.tight_layout()
    out = f"{a.out}/xsec_{a.arm}_vs_{a.ctl}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
