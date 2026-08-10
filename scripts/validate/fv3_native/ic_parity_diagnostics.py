"""FIRST PARITY MEASUREMENT: the port's test_case=-13 IC vs the oracle's.

The oracle prints its t=0 diagnostics before the first step
(run_hydro_1step_gfs/run_out.txt:188-192, the GFS-constants build):

    U  max =    20.140716845665871       min =   -19.876553508063072
    V  max =    20.347051347649494       min =   -19.876553508063072
    Total surface pressure (mb) =    1000.0000000000001
    ZS = 0

Those are GLOBAL over all six faces. This builds the same IC through the port
and compares.

WHAT WOULD MAKE THIS MEASUREMENT MEANINGLESS, and how each is guarded:
  * wrong constant set -> GFS is passed explicitly and printed
  * wrong grid conventions -> oracle_conventions=True is printed
  * wrong window (halo included) -> the compute slice is printed with shapes
  * a max taken over uninitialised halo -> only the written window is scored,
    and the count of scored points is printed so a silent empty slice shows
  * degrees vs radians -> the corner lat range is printed; it must span
    about +/-pi/4 per face, never +/-45

The oracle's u/v are the D-grid winds straight out of DCMIP16_BC; no rotation
or remap has happened at t=0.
"""
import sys

import numpy as np

from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
from legoesm.core.fv3_native_eta import set_eta_analytic
from legoesm.core.fv3_native_dcmip16_bc import GFS_CONSTANTS
from legoesm.core.fv3_native_dcmip16_ic import dcmip16_bc_face
from legoesm.grids.fv3_native_metrics import great_circle_dist as _gcd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
NG = 3
KM = 5

ORACLE = {"u_max": 20.140716845665871, "u_min": -19.876553508063072,
          "v_max": 20.347051347649494, "v_min": -19.876553508063072}


def gcd_radius(p1, p2, radius):
    """great_circle_dist with an explicit radius (the oracle's 3-arg form)."""
    return _gcd(np.asarray(p1, dtype=np.float64),
                np.asarray(p2, dtype=np.float64)) * radius


print(f"=== C{N} npz={KM}, GFS constants, oracle_conventions=True ===")
print(f"constants: radius={GFS_CONSTANTS.radius} grav={GFS_CONSTANTS.grav} "
      f"rdgas={GFS_CONSTANTS.rdgas} omega={GFS_CONSTANTS.omega}")
ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                 oracle_conventions=True)
ak, bk, ptop, ks = set_eta_analytic(KM)
print(f"ak={ak}\nbk={bk}\nptop={ptop} ks={ks}")

gs0 = ctx["gs6"][0]
print(f"grid_lon shape={np.asarray(gs0['grid_lon']).shape} "
      f"agrid_lon shape={np.asarray(gs0['agrid_lon']).shape}")

u_all, v_all, pt_all, delp_all = [], [], [], []
for t in range(6):
    gs = ctx["gs6"][t]
    glon = np.asarray(gs["grid_lon"], dtype=np.float64)
    glat = np.asarray(gs["grid_lat"], dtype=np.float64)
    alon = np.asarray(gs["agrid_lon"], dtype=np.float64)
    alat = np.asarray(gs["agrid_lat"], dtype=np.float64)

    # compute window: centres is..ie -> [NG, NG+N), corners one larger
    cs = slice(NG, NG + N)
    cc = slice(NG, NG + N + 1)
    corners = np.stack([glon[cc, cc], glat[cc, cc]], axis=-1)
    centres = np.stack([alon[cs, cs], alat[cs, cs]], axis=-1)

    if t == 0:
        print(f"corners {corners.shape} centres {centres.shape}")
        print(f"corner lat range [{corners[...,1].min():.6f},"
              f"{corners[...,1].max():.6f}] rad "
              f"(must be radians, |lat|<=pi/2={np.pi/2:.6f})")

    out = dcmip16_bc_face(corners, centres, ak, bk, KM,
                          constants=GFS_CONSTANTS, do_pert=True,
                          great_circle_dist=gcd_radius)
    u_all.append(out["u"])
    v_all.append(out["v"])
    pt_all.append(out["pt"])
    delp_all.append(out["delp"])
    print(f"  face {t+1}: u{out['u'].shape} v{out['v'].shape} "
          f"umax={out['u'].max():.9f} vmax={out['v'].max():.9f}")

u = np.concatenate([a.ravel() for a in u_all])
v = np.concatenate([a.ravel() for a in v_all])
pt = np.concatenate([a.ravel() for a in pt_all])
delp = np.concatenate([a.ravel() for a in delp_all])
print(f"\nscored points: u={u.size} v={v.size} (nonzero guard: "
      f"{'OK' if u.size and v.size else 'EMPTY SLICE -- measurement void'})")
assert np.isfinite(u).all() and np.isfinite(v).all(), "non-finite winds"

print("\n=== PARITY vs the oracle's t=0 diagnostics ===")
got = {"u_max": u.max(), "u_min": u.min(), "v_max": v.max(), "v_min": v.min()}
worst = 0.0
for k in ("u_max", "u_min", "v_max", "v_min"):
    o, g = ORACLE[k], got[k]
    abs_d = abs(g - o)
    rel_d = abs_d / abs(o)
    worst = max(worst, rel_d)
    print(f"  {k:6s} oracle={o!r:24s} port={g!r:24s} "
          f"abs={abs_d:.6e} rel={rel_d:.6e}")
print(f"\nworst relative difference: {worst:.6e}")
print(f"quad-geometry parity floor is ~1e-14 relative; "
      f"{'AT/BELOW FLOOR' if worst < 1e-13 else 'ABOVE FLOOR -- real defect'}")

print(f"\npt   range [{pt.min():.6f},{pt.max():.6f}] K  (virtual T; "
      f"adiabatic=.true. so the restart's pt is virtual)")
print(f"delp range [{delp.min():.6f},{delp.max():.6f}] Pa (expect 10000)")
print("PROBE_DONE")
