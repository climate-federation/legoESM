#!/usr/bin/env python
"""Why the equatorial closure is starved: too little energy, or too much
stratification, or too short a length?

WHY.  ``equatorial_diffusivity_oracle.py`` measured the symptom: below roughly
12 m our vertical heat diffusivity sits on its background floor while NEMO's
stays three orders of magnitude above it, and the viscosity is starved by the
same factor.  Both being starved rules out a Prandtl PARTITION error and puts
the defect in the mixing coefficient itself.

A TKE closure builds that coefficient as ``K ~ C_k * l * sqrt(e)``, so exactly
three things can produce it, and they call for different fixes:

  * the turbulent energy ``e`` is too small   -> a PRODUCTION problem
    (surface input, shear production) ;
  * the stratification ``N2`` is too strong   -> a FEEDBACK
    (our own warm surface restratifying the column and shutting the closure
    off, which would make the bias self-sustaining) ;
  * the mixing length ``l`` is too short      -> a LENGTH-SCALE problem
    (the buoyancy length sqrt(2e)/N, its limiters, or the surface value).

This measures ``e`` and ``N2`` directly on both sides and reports the length
each side's own K implies, so the three are separated rather than ranked.

MATCHED, deliberately, and this is the whole point of the probe:

  * BOTH sides' N2 goes through the model's own
    ``compute_buoyancy_frequency_nemo_bn2`` (NEMO eosbn2.F90 bn2_t), so the
    stratification is one instrument, not two.  Reading NEMO's archived
    ``bn2`` instead would compare its convention against ours.
  * both sides use the SAME mesh, the same native frame, the same interface
    ladder and the same box.
  * the 5-day record is DERIVED from the day (D/5 - 1) and never defaulted.

LIMITS, stated rather than discovered later:
  * our side is an instantaneous snapshot and NEMO's a five-day mean, so the
    top ~10 m carries a diurnal-phase difference the band medians cannot
    remove.  Read the 15-65 m band, which has little diurnal signal.
  * an interface is used only where BOTH touching cells are wet: NEMO's fill
    values are read as T=0, S=0, which at a wet/dry interface differences real
    water against fresh and reads as violently unstable.
  * the implied length is a DIAGNOSIS of each side's own K and e, not a field
    either model stores; it is printed as a ratio, which is the only part of
    it that means anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
for _p in ("ocean", "core"):
    sys.path.insert(0, str(_HERE.parents[3] / "packages" / _p))

from global_tracer_content import (  # noqa: E402
    _native, load_mesh_depth_1d, load_mesh_latitude, load_mesh_longitude,
    load_mesh_metrics,
)
# The MATCHED pair, already written and already paired: our staggered u/v carry
# an extra face column/row, NEMO's carry none, and each needs its own transform
# onto T points. Comparing a face velocity against a cell-centre average is the
# exact staggering error that put a retracted "1.2 % agreement" into a merged
# PR, so both sides are moved to T points here rather than one of them.
from frozen_column_tke_twin import (  # noqa: E402
    centre_uv_collocated, centre_uv_extra_column,
)

#: Bands matched to the diffusivity probe so the two can be read side by side.
_BANDS = (("surface 15-65 m", 15.0, 65.0),
          ("entrainment 65-105 m", 65.0, 105.0),
          ("background >105 m", 105.0, 1.0e9))

#: NEMO zdftke rn_ediff; K_m = rn_ediff * l * sqrt(2e) in zdftke.F90.  Only the
#: RATIO of the two implied lengths is reported, so this cancels -- it is named
#: so the formula in the code is the closure's, not an invented one.
_C_K = 0.1


def mesh_w_1d(mesh_mask_path):
    """Interface depths and RAW w-cell thicknesses (never reconstructed)."""
    import netCDF4 as nc
    ds = nc.Dataset(mesh_mask_path)
    try:
        g = np.asarray(ds.variables["gdepw_1d"][:], dtype=np.float64).squeeze()
        if "e3w_1d" not in ds.variables:
            raise SystemExit(f"FATAL: {mesh_mask_path} carries no e3w_1d; the "
                             "N2 closure requires raw-mesh w thicknesses")
        e = np.asarray(ds.variables["e3w_1d"][:], dtype=np.float64).squeeze()
    finally:
        ds.close()
    return g, e


def bn2(T, S, gdept, gdepw_int, e3w_int):
    """The MODEL'S OWN NEMO bn2. T/S level-LAST, per the function's contract."""
    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
    return np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S),
        jnp.asarray(gdept), jnp.asarray(gdepw_int),
        e3w_int=jnp.asarray(e3w_int)))


def shear2(u, v, e3w_int):
    """|du/dz|^2 + |dv/dz|^2 at interior interfaces, level-LAST arrays.

    Differenced across the SAME interfaces N2 uses and divided by the SAME raw
    e3w, so the two quantities land on one ladder and their ratio is a real
    Richardson number rather than two fields on neighbouring half-levels.

    Both sides go through this one function from their own u/v, so the C-grid
    staggering -- identical on our tripole and on NEMO's eORCA1 -- cancels.
    No rotation is applied: the box straddles no fold and the squared sum of
    the two components is invariant under the grid rotation anyway.

    Both sides arrive here on T POINTS, each through its own centring
    transform, so neither is a face velocity paired against a cell average.
    """
    dz = e3w_int[None, None, :]
    du = (u[..., :-1] - u[..., 1:]) / dz
    dv = (v[..., :-1] - v[..., 1:]) / dz
    return du * du + dv * dv


def band_median(field, wet, box, z_int, lo, hi):
    """Median over the box and depth band, wet interfaces only."""
    sel = (z_int >= lo) & (z_int < hi)
    if not sel.any():
        return float("nan"), 0
    vals = field[box][:, sel]
    ok = wet[box][:, sel] > 0.5
    if not ok.any():
        return float("nan"), 0
    return float(np.median(vals[ok])), int(ok.sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--nemo-gridt", required=True)
    p.add_argument("--nemo-day", type=float, required=True,
                   help="Day the 5-day record ENDS on; the record index is "
                        "D/5 - 1 and is never defaulted.")
    p.add_argument("--nemo-gridw", default=None,
                   help="Optional grid_W, for NEMO's archived avm/avt so the "
                        "implied mixing lengths can be compared.")
    p.add_argument("--nemo-gridu", default=None,
                   help="NEMO grid_U. With --nemo-gridv this adds the VERTICAL "
                        "SHEAR on both sides, computed the same way from each "
                        "side's own u/v, so the staggering cancels in the "
                        "ratio. Shear production is the dominant TKE source at "
                        "the equator, so this is what separates a production "
                        "failure from a defect inside the TKE equation.")
    p.add_argument("--nemo-gridv", default=None)
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--lat-halfwidth", type=float, default=2.0)
    p.add_argument("--lon-west", type=float, default=200.0)
    p.add_argument("--lon-east", type=float, default=260.0)
    p.add_argument("--out-json", default=None)
    a = p.parse_args()

    if a.nemo_day <= 0 or a.nemo_day % 5 != 0:
        raise SystemExit(f"--nemo-day wants a positive multiple of 5 (the "
                         f"output is 5-day means), got {a.nemo_day}")
    rec = int(a.nemo_day // 5) - 1

    _, _, e3t, tmask = load_mesh_metrics(a.mesh_mask)
    gdept = load_mesh_depth_1d(a.mesh_mask)
    gdepw, e3w = mesh_w_1d(a.mesh_mask)
    gdepw_int, e3w_int = gdepw[1:], e3w[1:]          # interior interfaces
    lat = load_mesh_latitude(a.mesh_mask)
    lon = load_mesh_longitude(a.mesh_mask) % 360.0
    box = (np.abs(lat) <= a.lat_halfwidth) & (lon >= a.lon_west) & (lon <= a.lon_east)
    if not box.any():
        raise SystemExit("no cells in the requested box")
    print(f"box |lat| <= {a.lat_halfwidth}, {a.lon_west:g}-{a.lon_east:g}E "
          f"-> {int(box.sum())} columns; NEMO record {rec} (day {a.nemo_day:g})")

    z = np.load(a.snapshot)
    for k in ("T", "S", "tke"):
        if k not in z:
            raise SystemExit(f"FATAL: {a.snapshot} lacks '{k}'. TKE is written "
                             "by the tripole lane only; this probe refuses to "
                             "substitute a reconstruction for the stored field.")
    T_our = np.transpose(_native(z["T"]), (1, 2, 0))       # (nj, ni, nlev)
    S_our = np.transpose(_native(z["S"]), (1, 2, 0))
    tke_our = np.transpose(_native(z["tke"]), (1, 2, 0))   # (nj, ni, nlev-1)

    import netCDF4 as nc
    ds = nc.Dataset(a.nemo_gridt)
    try:
        def pick(names, what):
            for n in names:
                if n in ds.variables:
                    return n
            raise SystemExit(f"FATAL: no {what} in {a.nemo_gridt}")
        tn = pick(("to", "bigthetao", "thetao", "votemper"), "3-D temperature")
        sn = pick(("so", "so_abs", "vosaline"), "3-D salinity")
        print(f"[nemo] T={tn} S={sn}")
        def v(n):
            return np.ma.filled(np.ma.masked_invalid(ds.variables[n][rec]),
                                0.0).astype(np.float64)
        T_nemo = np.transpose(v(tn), (1, 2, 0))
        S_nemo = np.transpose(v(sn), (1, 2, 0))
    finally:
        ds.close()
    if T_nemo.shape != T_our.shape:
        raise SystemExit(f"NEMO {T_nemo.shape} vs ours {T_our.shape}")

    n2_our = bn2(T_our, S_our, gdept, gdepw_int, e3w_int)
    n2_nemo = bn2(T_nemo, S_nemo, gdept, gdepw_int, e3w_int)

    s2_our = s2_nemo = None
    if a.nemo_gridu and a.nemo_gridv:
        for k in ("u", "v"):
            if k not in z:
                raise SystemExit(f"FATAL: {a.snapshot} lacks '{k}'; the shear "
                                 "comparison needs both velocity components")
        u_c, v_c = centre_uv_extra_column(z["u"], z["v"])
        s2_our = shear2(np.transpose(_native(u_c), (1, 2, 0)),
                        np.transpose(_native(v_c), (1, 2, 0)), e3w_int)
        def _vel(path, names, what):
            d = nc.Dataset(path)
            try:
                nm = next((n for n in names if n in d.variables), None)
                if nm is None:
                    raise SystemExit(f"FATAL: no {what} in {path}; looked for "
                                     + ", ".join(names))
                print(f"[nemo] {what}={nm}")
                arr = np.ma.filled(
                    np.ma.masked_invalid(d.variables[nm][rec]), 0.0)
            finally:
                d.close()
            return np.transpose(np.asarray(arr, dtype=np.float64), (1, 2, 0))
        un, vn = centre_uv_collocated(
            _vel(a.nemo_gridu, ("uo", "vozocrtx", "uoce"), "u"),
            _vel(a.nemo_gridv, ("vo", "vomecrty", "voce"), "v"))
        s2_nemo = shear2(un, vn, e3w_int)
        if s2_nemo.shape != s2_our.shape:
            raise SystemExit(f"NEMO shear {s2_nemo.shape} vs ours {s2_our.shape}")

    # Both cells touching an interface must be wet, or a wet/dry interface
    # differences real water against NEMO's T=0/S=0 fill and reads unstable.
    wet_c = (tmask > 0.5)                                   # (nlev, nj, ni)
    wet_i = np.transpose(wet_c[:-1] & wet_c[1:], (1, 2, 0))  # (nj, ni, nlev-1)

    nemo_K = {}
    if a.nemo_gridw:
        dsw = nc.Dataset(a.nemo_gridw)
        try:
            for nm in ("avm", "avt"):
                if nm in dsw.variables:
                    arr = np.ma.filled(
                        np.ma.masked_invalid(dsw.variables[nm][rec]), 0.0)
                    nemo_K[nm] = np.transpose(np.asarray(arr, dtype=np.float64),
                                              (1, 2, 0))[..., 1:]
        finally:
            dsw.close()
        print(f"[nemo] grid_W carries: {sorted(nemo_K) or 'neither avm nor avt'}")

    report = {"snapshot": str(a.snapshot), "nemo_gridt": str(a.nemo_gridt),
              "nemo_record": rec, "nemo_day": a.nemo_day, "bands": {}}
    print(f"\n{'band':22s} {'N2 ours':>10s} {'N2 NEMO':>10s} {'xN2':>6s}"
          f" {'e ours':>10s} {'sqrt(e) x':>10s} {'n':>8s}")
    for name, lo, hi in _BANDS:
        n2o, n = band_median(n2_our, wet_i, box, gdepw_int, lo, hi)
        n2n, _ = band_median(n2_nemo, wet_i, box, gdepw_int, lo, hi)
        eo, _ = band_median(tke_our, wet_i, box, gdepw_int, lo, hi)
        rn2 = n2o / n2n if n2n else float("nan")
        row = {"n2_ours": n2o, "n2_nemo": n2n, "n2_ratio": rn2,
               "tke_ours": eo, "n_interfaces": n}
        print(f"{name:22s} {n2o:10.3e} {n2n:10.3e} {rn2:6.2f}"
              f" {eo:10.3e} {'':>10s} {n:8d}")
        if s2_our is not None:
            so, _ = band_median(s2_our, wet_i, box, gdepw_int, lo, hi)
            sn, _ = band_median(s2_nemo, wet_i, box, gdepw_int, lo, hi)
            rs2 = so / sn if sn else float("nan")
            row.update({"shear2_ours": so, "shear2_nemo": sn,
                        "shear2_ratio": rs2,
                        "ri_ours": n2o / so if so else float("nan"),
                        "ri_nemo": n2n / sn if sn else float("nan")})
            print(f"{'':22s} shear^2 {so:10.3e} {sn:10.3e} (x{rs2:.2f})"
                  f"   Ri {row['ri_ours']:8.3f} {row['ri_nemo']:8.3f}")
        report["bands"][name] = row

    if s2_our is not None:
        print("\nSHEAR: a ratio near 0.25 (half the velocity difference) makes "
              "the collapse SELF-CONSISTENT -- weak current, weak production, "
              "weak coefficient, weaker current. A ratio near 1 puts the "
              "defect INSIDE the TKE equation, because production is then "
              "available and the energy still is not there.")
        print("Ri here is N2/shear^2 from the two MEASURED fields on one "
              "ladder; it is not the Prandtl inversion and the two are "
              "independent estimates of the same number.")
    print("\nREAD IT AS: N2 ratio near 1 exonerates stratification, so a "
          "starved K is then an ENERGY or LENGTH problem. A ratio well above 1 "
          "means our own warm surface is restratifying the column and shutting "
          "the closure off -- a feedback, and the bias would be self-sustaining.")
    print("Our snapshot is INSTANTANEOUS and NEMO's record a five-day MEAN; "
          "read the 15-65 m band, not the surface one.")
    if not nemo_K:
        print("NEMO's TKE is not archived in the 5-day output, so `e` cannot be "
              "compared directly; ours is printed for the length arithmetic.")

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(report, indent=2))
        print(f"[report] {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
