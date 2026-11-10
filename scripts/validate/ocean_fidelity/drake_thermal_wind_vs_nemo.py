"""Does the cross-frontal density contrast explain our strong ACC?

TWO MEASURED FACTS, NOT YET CONNECTED. Our Drake transport is 162.34 Sv
against NEMO's 152.27, a ratio of 1.066, with the excess concentrated in a
jet that also sits about half a degree north of NEMO's. Separately, our
near-surface stratification in the Weddell is far too weak -- N2 in the top
five metres runs 0.014 to 0.27 of NEMO's. A Southern Ocean with too little
near-surface stratification and a too-strong ACC is the kind of pair that can
share a surface-buoyancy cause. Nothing measured connects them yet, and this
probe exists to decide that rather than to assert it.

THE TEST. An ACC transport is set by the density contrast across the current
through thermal wind: the zonal shear follows the meridional density
gradient, so the baroclinic transport relative to the bottom goes as the
DOUBLE depth integral of the cross-frontal density difference. If our jet is
strong because the water on either side of it is wrong, that double integral
must be larger than NEMO's by roughly the transport ratio. If the contrast
matches, the jet difference comes from somewhere else entirely -- topography,
the momentum closure, or the barotropic solver -- and the two Southern Ocean
findings are INDEPENDENT.

PRE-REGISTERED BEFORE LOOKING, as a number with a sign:
  ratio >= 1.05   the density contrast accounts for most of the transport
                  excess; the shared-cause hypothesis SURVIVES and the
                  near-surface buoyancy is worth chasing as a common driver.
  ratio <= 1.02   the contrast matches NEMO's; the transport excess is NOT
                  explained by the density field, the two findings are
                  INDEPENDENT, and the jet must be chased elsewhere.
  1.02 - 1.05     ambiguous; say so rather than picking a side.
Additionally, the shared-cause reading requires the excess to be
CONCENTRATED IN THE UPPER FEW HUNDRED METRES, where the stratification
difference lives. A contrast excess that is uniform with depth, or that sits
in the abyss, points at the deep density field instead and does not support
it however the ratio lands.

IDENTICAL INSTRUMENT ON BOTH SIDES. Both models' T and S go through the same
nemo_seos_eos, evaluated at the same geometric depths, so the comparison is
of the two STATES and not of two equations. Both sides carry Conservative
Temperature and Absolute Salinity (NEMO's grid_T labels them `to`/`so` and
its CF metadata is wrong about what they are; that was established earlier in
this campaign by reading the Fortran).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from global_tracer_content import (  # noqa: E402
    _native, load_mesh_depth_1d, load_mesh_latitude, load_mesh_longitude,
    load_mesh_metrics,
)


def _rho(T, S, gdept):
    """In-situ density via the model's own NEMO simplified EOS.

    p is handed in as rho0*g*depth so the routine's own p/(rho0*g) recovers
    exactly the geometric depth -- the same ladder on both sides.
    """
    import jax.numpy as jnp
    from legoesm.ocean.eos import NemoSEOSConfig, nemo_seos_eos
    from legoesm import constants
    cfg = NemoSEOSConfig()
    p = cfg.rho0 * constants.g * gdept
    return np.asarray(nemo_seos_eos(jnp.asarray(T), jnp.asarray(S),
                                    jnp.asarray(p), cfg))


def _double_integral(drho, dz, wet):
    """Depth-integrate twice: proportional to baroclinic transport.

    inner(z) = integral from z up to the surface of drho
    D        = integral over depth of inner
    Only wet levels contribute; a dry level contributes nothing rather than
    carrying a zero-filled density into the sum, which is the mistake that
    produced a 414x diffusion ratio and a fake unstable layer earlier on this
    branch.
    """
    d = np.where(wet, drho, 0.0)
    h = np.where(wet, dz, 0.0)
    inner = np.cumsum(d * h)          # surface-down running integral
    return float(np.sum(inner * h))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--nemo-gridt", required=True)
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--rec", type=int, required=True)
    p.add_argument("--lon", type=float, default=-65.0)
    p.add_argument("--lat-range", default="-68,-55")
    p.add_argument("--n-edge-rows", type=int, default=3,
                   help="rows averaged at each end of the section to define "
                        "the north and south sides of the front")
    p.add_argument("--south-lat", default=None, metavar="A,B",
                   help="explicit southern box instead of the section's "
                        "southern edge. The edge default puts the southern "
                        "box on the ANTARCTIC SHELF, which truncates the "
                        "cross-front integral at the shelf depth -- use this "
                        "to place both boxes in deep water. Equals form.")
    p.add_argument("--north-lat", default=None, metavar="A,B")
    p.add_argument("--out-json", default=None)
    a = p.parse_args()

    import netCDF4 as nc

    _e1t, _e2t, e3t, tmask = load_mesh_metrics(a.mesh_mask)
    lat = load_mesh_latitude(a.mesh_mask)
    lon = load_mesh_longitude(a.mesh_mask) % 360.0
    gdept = load_mesh_depth_1d(a.mesh_mask)

    la0, la1 = (float(v) for v in a.lat_range.split(","))
    lon0 = a.lon % 360.0
    rows = (lat >= la0) & (lat <= la1)
    dlon = np.abs((lon - lon0 + 180.0) % 360.0 - 180.0)
    with np.errstate(invalid="ignore"):
        i = int(np.nanargmin(np.nanmean(np.where(rows, dlon, np.nan), axis=0)))
    sel = rows[:, i] & (tmask[:, :, i].max(axis=0) > 0.5)
    jj = np.where(sel)[0]
    if jj.size < 2 * a.n_edge_rows:
        raise SystemExit(f"FATAL: only {jj.size} wet rows at i={i}")
    def _box(spec, fallback):
        if spec is None:
            return fallback
        b0, b1 = (float(v) for v in spec.split(","))
        k = jj[(lat[jj, i] >= b0) & (lat[jj, i] <= b1)]
        if k.size == 0:
            raise SystemExit(f"FATAL: box {spec!r} contains no wet row")
        return k

    south = _box(a.south_lat, jj[:a.n_edge_rows])
    north = _box(a.north_lat, jj[-a.n_edge_rows:])
    print(f"[section] i={i}, lon {lon[jj, i].mean():.2f}E, "
          f"{jj.size} wet rows from {lat[jj[0], i]:.2f} to "
          f"{lat[jj[-1], i]:.2f}")
    print(f"[section] south side {lat[south, i].min():.2f} to "
          f"{lat[south, i].max():.2f}; north side {lat[north, i].min():.2f} "
          f"to {lat[north, i].max():.2f}")

    z = dict(np.load(a.snapshot))
    T_our = np.transpose(_native(z["T"]), (1, 2, 0))
    S_our = np.transpose(_native(z["S"]), (1, 2, 0))

    ds = nc.Dataset(a.nemo_gridt)
    try:
        def v(n):
            x = ds.variables[n][a.rec]
            x = x.filled(np.nan) if np.ma.isMaskedArray(x) else np.asarray(x)
            return np.transpose(np.asarray(x, dtype=np.float64), (1, 2, 0))
        T_nem, S_nem = v("to"), v("so")
    finally:
        ds.close()
    if T_nem.shape != T_our.shape:
        raise SystemExit(f"FATAL: NEMO {T_nem.shape} vs ours {T_our.shape}")

    wet = np.transpose(tmask, (1, 2, 0)) > 0.5
    dz = np.transpose(e3t, (1, 2, 0))

    out = {"i": i, "rec": a.rec, "sides": {}, "models": {}}
    for tag, (T, S) in (("ours", (T_our, S_our)), ("nemo", (T_nem, S_nem))):
        rho = _rho(T, S, gdept[None, None, :])
        # Side profiles: average over the edge rows using only wet cells, so
        # a shallow column never dilutes a deep level with a dry value.
        prof = {}
        for side, idx in (("south", south), ("north", north)):
            w = wet[idx, i, :]
            r = np.where(w, rho[idx, i, :], 0.0).sum(axis=0)
            n = w.sum(axis=0)
            prof[side] = np.where(n > 0, r / np.maximum(n, 1), np.nan)
        drho = prof["south"] - prof["north"]
        colwet = np.isfinite(drho)
        # HOW DEEP DOES THIS TEST ACTUALLY REACH? The cross-front difference
        # is undefined below the SHALLOWER side's floor, so the double
        # integral silently stops there. With the southern box on the
        # Antarctic shelf that truncates the ACC's deep limb out of the test
        # entirely, and a ratio computed on the remaining top few hundred
        # metres would be quoted as though it covered the current. Printed,
        # not assumed.
        kmax = int(np.max(np.where(colwet)[0])) if colwet.any() else -1
        print(f"[{tag}] cross-front defined over {int(colwet.sum())} of "
              f"{colwet.size} levels, deepest {gdept[kmax]:.1f} m")
        dzc = dz[jj, i, :].mean(axis=0)
        D = _double_integral(np.nan_to_num(drho), dzc, colwet)
        out["models"][tag] = {
            "double_integral": D,
            "drho_by_level": np.where(colwet, drho, np.nan).tolist(),
            "south_rho": prof["south"].tolist(),
            "north_rho": prof["north"].tolist(),
        }
        print(f"[{tag}] cross-front drho at 5 m {drho[2]:+.4f}, 100 m "
              f"{drho[np.argmin(np.abs(gdept - 100))]:+.4f}, 1000 m "
              f"{drho[np.argmin(np.abs(gdept - 1000))]:+.4f} kg/m3; "
              f"double integral {D:.6e}")

    Do, Dn = (out["models"][k]["double_integral"] for k in ("ours", "nemo"))
    ratio = Do / Dn if Dn else float("nan")
    out["ratio_ours_over_nemo"] = ratio
    print(f"\n[verdict] density-contrast ratio ours/NEMO = {ratio:.4f} "
          "(transport ratio was 1.066)")
    if ratio >= 1.05:
        print("[verdict] SHARED-CAUSE HYPOTHESIS SURVIVES: the cross-frontal "
              "density contrast accounts for most of the transport excess. "
              "Check below that the excess sits in the upper few hundred "
              "metres before chasing surface buoyancy as the common driver.")
    elif ratio <= 1.02:
        print("[verdict] INDEPENDENT: the density contrast matches NEMO's, so "
              "the transport excess is NOT explained by the density field. "
              "The jet must be chased in the momentum closure, the "
              "topography, or the barotropic solver -- not in surface "
              "buoyancy.")
    else:
        print("[verdict] AMBIGUOUS between the pre-registered bands. No "
              "direction claimed.")

    # WHERE the contrast difference sits decides the reading regardless of the
    # ratio, so it is reported rather than left implied.
    do = np.asarray(out["models"]["ours"]["drho_by_level"], dtype=float)
    dn = np.asarray(out["models"]["nemo"]["drho_by_level"], dtype=float)
    ok = np.isfinite(do) & np.isfinite(dn)
    diff = np.abs(np.where(ok, do - dn, 0.0))
    tot = diff.sum()
    if tot > 0:
        upper = diff[gdept <= 300.0].sum() / tot
        print(f"[verdict] {100 * upper:.1f}% of the contrast difference sits "
              f"above 300 m ({int((gdept <= 300).sum())} of {gdept.size} "
              "levels)")
        out["fraction_above_300m"] = float(upper)

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(out, indent=1))
        print(f"[report] {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
