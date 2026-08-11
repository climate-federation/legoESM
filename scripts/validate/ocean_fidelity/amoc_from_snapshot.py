"""AMOC@26.5N from a tripole snapshot via NEMO's OWN diagnostic core.

The circulation card shows our day-90 in-run AMOC at -0.13 Sv while NEMO's
matched-window (RUN_GATEWAY cold start, days 86-90) value is 15.9 Sv.  Two
mechanisms could explain it and this probe discriminates:

* the in-run diagnostic (``compute_amoc_from_state``) is broken/mismatched
  -> this probe, which pushes OUR snapshot v through the SAME ``amoc_core``
  that produced NEMO's 15.9, will report ~15 Sv;
* the model genuinely spins up no overturning -> this probe reports ~0 too,
  and the defect is physics (or the v field itself).

Conventions: snapshot arrays are (332, 362, nlev) with the native frame
[0:331, 1:361] (see global_tracer_content.py); domain_cfg carries the full
(332, 362) frame, sliced identically.  voe3 = v * e3v_0 * (H+eta)/H is the
same z-star construction the tracer-content probe uses; v is taken AT the
snapshot's v-points, which coincide with NEMO's on the shared eORCA1 mesh
(max|dlat| 1.4e-14).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
from nemo_transports import amoc_core  # noqa: E402

_NATIVE_J = slice(0, 331)
_NATIVE_I = slice(1, 361)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", nargs="+", required=True)
    p.add_argument("--domain-cfg", required=True)
    p.add_argument("--target-lat", type=float, default=26.5)
    a = p.parse_args()

    import netCDF4 as nc
    ds = nc.Dataset(a.domain_cfg)

    def v2(name):
        return np.asarray(ds.variables[name][:], dtype=np.float64).squeeze()[
            _NATIVE_J, _NATIVE_I]

    e1v = v2("e1v")
    gphiv = v2("gphiv")
    glamv = v2("glamv")
    e3v = np.asarray(ds.variables["e3v_0"][:], dtype=np.float64).squeeze()[
        :, _NATIVE_J, _NATIVE_I]
    # v-point wetness from e3v_0 > 0 is not reliable (reference thicknesses
    # fill land); use bottom_level when present.
    if "bottom_level" in ds.variables:
        nbot = v2("bottom_level").astype(int)
    else:
        nbot = None
    ds.close()
    nlev = e3v.shape[0]
    kidx = np.arange(nlev)[:, None, None]
    vmask = np.ones_like(e3v) if nbot is None else (kidx < nbot[None]).astype(float)
    depthv = np.cumsum(e3v * vmask, axis=0)

    for snap in a.snapshot:
        z = np.load(snap)
        v = np.asarray(z["v"], dtype=np.float64)
        if v.shape[:2] != (332, 362):
            raise SystemExit(f"unexpected v shape {v.shape}")
        v = np.transpose(v[_NATIVE_J, _NATIVE_I, :], (2, 0, 1))[:nlev]
        eta = np.asarray(z["eta"], dtype=np.float64)[_NATIVE_J, _NATIVE_I]
        H = (e3v * vmask).sum(axis=0)
        dil = np.where(H > 0, (H + eta) / np.where(H > 0, H, 1.0), 0.0)
        voe3 = v * e3v * vmask * dil[None]
        r = amoc_core(voe3, e1v, gphiv, glamv, depthv,
                      target_lat=a.target_lat)
        print(f"{Path(snap).parent.name}/{Path(snap).name}: "
              f"AMOC@{a.target_lat}N = {r['amoc_Sv']:.2f} Sv "
              f"(row lat {r['row_lat_deg']:.2f}, depth of max "
              f"{r['depth_of_max_m']:.0f} m)")
        # sanity: v magnitude in the Atlantic band at that row
        print(f"  v stats: max|v| {np.abs(v).max():.3f} m/s, "
              f"rms wet v {np.sqrt(np.mean(v[vmask > 0]**2)):.4f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
