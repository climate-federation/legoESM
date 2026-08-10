"""The port<->oracle face map, WITH transposes, required to be a bijection.

The earlier dihedral search excluded transposes (u is (48,49), v is (49,48),
so a transpose does not preserve shape within one component) and returned a
NON-INJECTIVE map: port face 2 "best-matched" oracle tiles 1 AND 2 while port
face 1 matched nothing. A non-injective best-match on a problem that must be a
bijection is the tell that the search space is too small -- so this widens it.

SHAPES. port u is (i,j) = (48,49); port v is (i,j) = (49,48).
The oracle stores u as (j,i) = (49,48) and v as (j,i) = (48,49). Hence:
  * port u vs oracle u  -> compare port_u against oracle_u.T   (no transpose)
  * port u vs oracle v  -> compare port_u against oracle_v     (as stored:
                           the face is transposed, which sends u <-> v)

MATCHING KEY. The base state is zonally symmetric and therefore DEGENERATE --
several faces carry identical fields, which is what made the first search
non-injective. The -13 field (base + localized perturbation) breaks that
degeneracy on the perturbed tiles; the rest are pinned by elimination once a
bijection is enforced. The full 6x6 cost matrix is printed so the degeneracy
is visible rather than hidden behind a single "best" label.
"""
import numpy as np
import netCDF4 as nc

from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
from legoesm.core.fv3_native_eta import set_eta_analytic
from legoesm.core.fv3_native_dcmip16_bc import GFS_CONSTANTS
from legoesm.core.fv3_native_dcmip16_ic import dcmip16_bc_face
from legoesm.grids.fv3_native_metrics import great_circle_dist as _gcd

N, NG, KM = 48, 3, 5
P13 = "/burg-archive/glab/users/pg2328/fv3_oracle_pinned/run_hydro_zerostep"
LEV = 0        # top layer, both sides index 0


def gcdr(p1, p2, r):
    return _gcd(np.asarray(p1, float), np.asarray(p2, float)) * r


ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                 oracle_conventions=True)
ak, bk, _, _ = set_eta_analytic(KM)

port_u, port_v = [], []
for t in range(6):
    gs = ctx["gs6"][t]
    cs, cc = slice(NG, NG + N), slice(NG, NG + N + 1)
    co = np.stack([np.asarray(gs["grid_lon"])[cc, cc],
                   np.asarray(gs["grid_lat"])[cc, cc]], -1)
    ce = np.stack([np.asarray(gs["agrid_lon"])[cs, cs],
                   np.asarray(gs["agrid_lat"])[cs, cs]], -1)
    o = dcmip16_bc_face(co, ce, ak, bk, KM, do_pert=True,
                        constants=GFS_CONSTANTS, great_circle_dist=gcdr)
    port_u.append(o["u"][:, :, LEV])       # (48, 49)
    port_v.append(o["v"][:, :, LEV])       # (49, 48)

orc_u, orc_v = [], []
for t in range(1, 7):
    d = nc.Dataset(f"{P13}/RESTART/fv_core.res.tile{t}.nc")
    orc_u.append(np.array(d["u"][:])[0, LEV])   # stored (j,i) = (49,48)
    orc_v.append(np.array(d["v"][:])[0, LEV])   # stored (j,i) = (48,49)
print(f"port u {port_u[0].shape} v {port_v[0].shape} | "
      f"oracle u {orc_u[0].shape} v {orc_v[0].shape}")


def dihedral(a):
    yield "id", a
    yield "fi", a[::-1, :]
    yield "fj", a[:, ::-1]
    yield "r180", a[::-1, ::-1]


def score(pu, pv, ou, ov):
    """Best (rel, label) over: no-transpose (u~u.T, v~v.T) and
    transpose (u~v, v~u.T.T=u... ) with dihedral maps and sign."""
    best = (np.inf, "")
    cands = [
        ("u~u", pu, ou.T),      # face not transposed
        ("v~v", pv, ov.T),
        ("u~v", pu, ov),        # face transposed: u <-> v
        ("v~u", pv, ou),
    ]
    for tag, p, o in cands:
        if p.shape != o.shape:
            continue
        for nm, c in dihedral(p):
            for s in (1.0, -1.0):
                sc = max(np.abs(o).max(), np.abs(c).max())
                if sc < 1e-12:
                    continue
                r = np.abs(s * c - o).max() / sc
                if r < best[0]:
                    best = (r, f"{tag} {nm} s{s:+.0f}")
    return best


print("\ncost matrix rel(port face -> oracle tile), best transform:")
M = np.full((6, 6), np.inf)
L = [["" for _ in range(6)] for _ in range(6)]
for pf in range(6):
    for ot in range(6):
        r, lbl = score(port_u[pf], port_v[pf], orc_u[ot], orc_v[ot])
        M[pf, ot], L[pf][ot] = r, lbl
    row = "  ".join(f"{M[pf, o]:9.2e}" for o in range(6))
    print(f"  face {pf+1}: {row}")

print("\nper-face best:")
used = {}
for pf in range(6):
    ot = int(np.argmin(M[pf]))
    print(f"  port face {pf+1} -> oracle tile {ot+1}  rel={M[pf,ot]:.6e}  "
          f"[{L[pf][ot]}]")
    used.setdefault(ot, []).append(pf + 1)

dup = {k + 1: v for k, v in used.items() if len(v) > 1}
missing = [t + 1 for t in range(6) if t not in used]
print(f"\nbijection: {'YES' if not dup and not missing else 'NO'}")
if dup:
    print(f"  tiles claimed by >1 face: {dup}")
if missing:
    print(f"  tiles claimed by none:    {missing}")
n_floor = sum(1 for pf in range(6) if M[pf].min() < 1e-12)
print(f"faces matching at <1e-12: {n_floor}/6")
print("PROBE_DONE")
