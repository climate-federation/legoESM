"""Settle the eORCA1 C-grid index convention FROM THE MESH FILE, not from memory.

Two things block the Arctic-gateway oracle measurement, and both are answerable
by reading coordinates rather than by reasoning about NEMO documentation:

1. IS NEMO's ``u(j,i)`` EAST or WEST of ``T(j,i)``?  Decisive test: compare the
   u-point longitude ``glamu`` with the T-point longitude ``glamt`` in the same
   index.  East => glamu[i] > glamt[i].  Same for ``gphiv`` vs ``gphit`` in the
   meridional direction.  The grid loader (grids/tripole.py:536-543) states the
   WEST convention in its comment while standard NEMO documents EAST; on a
   quasi-uniform mesh a one-column metric shift is nearly a no-op numerically,
   so this cannot be settled by "the runs look fine".

2. WHAT IS THE HALO STRUCTURE?  mesh_mask is (332, 362) while the grid_U/grid_V
   output is (331, 360).  Which rows/columns are the halo, and is the zonal
   wrap a duplicate of interior columns?

Prints only.  No verdict is baked in -- the interpretation belongs in the
analysis, not in the tool.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

MESH = ("/burg-archive/glab/users/pg2328/legoESM/data/grids/"
        "eORCA1.2_mesh_mask.nc")
GRIDV = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/"
         "EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_V.nc")


def _sq(a):
    a = np.asarray(a)
    while a.ndim > 2:
        a = a[0]
    return a


def main() -> int:
    dm = xr.open_dataset(MESH, decode_times=False)
    print("[mesh] vars:", sorted(v for v in dm.variables)[:28])
    for n in ("glamt", "glamu", "gphit", "gphiv", "e1t", "e2u", "e1v", "tmask"):
        if n in dm:
            print(f"[mesh] {n:<7s} shape {tuple(dm[n].shape)}")

    glamt, glamu = _sq(dm["glamt"].values), _sq(dm["glamu"].values)
    gphit, gphiv = _sq(dm["gphit"].values), _sq(dm["gphiv"].values)
    ny, nx = glamt.shape

    # Mid-latitude, mid-basin row: away from the tripole fold and the wrap.
    j = int(np.argmin(np.abs(gphit[:, nx // 2] - 0.0)))
    i0 = nx // 2
    print(f"\n[zonal] equator-ish row j={j}, columns {i0}..{i0+3}")
    print("    i   glamt      glamu    glamu-glamt")
    for i in range(i0, i0 + 4):
        print(f" {i:4d} {glamt[j, i]:9.4f} {glamu[j, i]:9.4f} "
              f"{glamu[j, i] - glamt[j, i]:+9.4f}")

    j2 = int(np.argmin(np.abs(gphit[:, i0] - 40.0)))
    print(f"\n[merid] row j={j2} (lat~40N), column i={i0}")
    print("    j   gphit      gphiv    gphiv-gphit")
    for jj in range(j2, j2 + 4):
        print(f" {jj:4d} {gphit[jj, i0]:9.4f} {gphiv[jj, i0]:9.4f} "
              f"{gphiv[jj, i0] - gphit[jj, i0]:+9.4f}")

    print("\n[halo] zonal wrap: compare first/last columns of glamt")
    for i in (0, 1, 2, nx - 3, nx - 2, nx - 1):
        print(f"   i={i:4d}  glamt={glamt[j, i]:9.4f}")
    print("[halo] duplicate test: glamt[:,0] vs glamt[:,nx-2], "
          f"max|diff| = {np.nanmax(np.abs(glamt[:, 0] - glamt[:, nx - 2])):.3e}")
    print("[halo] duplicate test: glamt[:,1] vs glamt[:,nx-1], "
          f"max|diff| = {np.nanmax(np.abs(glamt[:, 1] - glamt[:, nx - 1])):.3e}")
    print(f"[halo] row 0 lat range: {np.nanmin(gphit[0]):.3f} .. "
          f"{np.nanmax(gphit[0]):.3f}")
    print(f"[halo] row 1 lat range: {np.nanmin(gphit[1]):.3f} .. "
          f"{np.nanmax(gphit[1]):.3f}")

    dV = xr.open_dataset(GRIDV, decode_times=False)
    vname = "vo" if "vo" in dV else "voce"
    print(f"\n[out ] {vname} shape {tuple(dV[vname].shape)}; "
          f"e3v present: {'e3v' in dV}")
    print("[out ] grid_V vars:", sorted(v for v in dV.variables)[:20])
    for cand in ("nav_lat", "nav_lon"):
        if cand in dV:
            print(f"[out ] {cand} shape {tuple(dV[cand].shape)}")
    if "nav_lon" in dV:
        nl = _sq(dV["nav_lon"].values)
        print(f"[out ] nav_lon[0,:4] = {nl[0, :4]}")
        print(f"[mesh] glamt[0,:6]   = {glamt[0, :6]}")
        print(f"[mesh] glamt[1,1:5]  = {glamt[1, 1:5]}")
    return 0




def scope() -> int:
    """SCOPE the tripole.py misalignment: where does e2u actually vary with i?

    grids/tripole.py places NEMO's e2u[:, i] at our u-face i, but the mesh says
    NEMO's u(i) is the face EAST of T(i) = our face i+1.  That misassignment is
    EXACTLY a no-op wherever e2u does not vary along i, which is most of a
    lat-lon mesh.  Print the along-i variation by latitude band so the blast
    radius is measured rather than asserted.
    """
    dm = xr.open_dataset(MESH, decode_times=False)
    # INTERIOR ONLY.  The mesh carries two zonal halo columns (glamt[:,0] ==
    # glamt[:,360] and glamt[:,1] == glamt[:,361], verified exactly), so the
    # physical domain is [1:, 1:361] = (331, 360).  Rolling across the halo
    # instead compares a column against a duplicate and manufactures huge
    # spurious jumps -- the first version of this probe did exactly that and
    # its numbers are void.
    sl = (slice(1, None), slice(1, 361))
    e2u = _sq(dm["e2u"].values)[sl]
    e1u = _sq(dm["e1u"].values)[sl]
    gphit = _sq(dm["gphit"].values)[sl]
    ok = (e2u > 0.0) & (e1u > 0.0)
    # the interior IS periodic in i, so this roll is the real eastern neighbour
    d2 = np.where(ok, np.abs(np.roll(e2u, -1, axis=1) - e2u) / np.where(e2u > 0, e2u, 1.0), np.nan)
    d1 = np.where(ok, np.abs(np.roll(e1u, -1, axis=1) - e1u) / np.where(e1u > 0, e1u, 1.0), np.nan)
    print("\n[scope] INTERIOR ONLY. relative |e(i+1)-e(i)|/e by latitude band")
    print("   band            e2u max     e2u p99     e1u max     e1u p99    n")
    for lo, hi in [(-90, -60), (-60, -30), (-30, 30), (30, 60),
                   (60, 66), (66, 75), (75, 90)]:
        m = (gphit >= lo) & (gphit < hi) & ok
        if not m.any():
            continue
        print(f" {lo:+4d}..{hi:+4d}  {np.nanmax(d2[m]):10.3e} "
              f"{np.nanpercentile(d2[m], 99):10.3e} {np.nanmax(d1[m]):10.3e} "
              f"{np.nanpercentile(d1[m], 99):10.3e} {int(m.sum()):7d}")
    # where the worst e2u cells actually are
    flat = np.where(np.isfinite(d2), d2, -1.0)
    j, i = np.unravel_index(np.argmax(flat), flat.shape)
    print(f"[scope] worst e2u cell: j={j} i={i} lat={gphit[j, i]:.2f} "
          f"ratio={flat[j, i]:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or scope())
