#!/usr/bin/env python
"""GATE: legoESM's analytic DINO mesh must equal NEMO's mesh_mask BIT-FOR-BIT.

`run_dino.py --nemo-faithful-grid` builds NEMO's DINO domain analytically, with
no NEMO file read at run time.  This gate checks that construction against the
oracle's own `mesh_mask.nc` and exits non-zero on ANY inequality.

THE BAR IS EXACT: zero cells unequal, per field.  The single exception is
declared, bounded and measured below (`FF_ULP_WAIVER`), not left to judgment.

Three sections, each fatal:

  1. COVERAGE (oracle-fidelity Rule 1) -- every variable in mesh_mask.nc is
     either VERIFIED here or WAIVED with a written reason.  A variable that is
     neither is a hard failure, so adding a field to the oracle's output can
     never silently go unchecked.
  2. BIT-EXACTNESS -- per-field cells-unequal / max|delta| / max-ulp table.
  3. RULE 10 (score through the model's own path) -- rebuild the domain by
     calling the SAME functions `run_dino.py` calls, and check (a) it is
     leaf-for-leaf identical to the domain the certified twin gets by READING
     mesh_mask.nc through the same bridge, and (b) the census the twin harness
     runs (`kamm_run5y_v3.py`): wet cells and u/v face masks vs NEMO.

Usage
-----
    JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/dino_1226/\\
        nemo_dino_mesh_gate.py [--mesh-mask PATH] [--plant]

`--plant` flips one wet cell of the transcribed mask and must make the gate
exit non-zero -- the non-vacuity check for the gate itself.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys

import numpy as np

DEFAULT_MESH = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
                "RUN_TRAJ/mesh_mask.nc")

# The ONLY tolerated inequality, and it is the ORACLE's own rounding, not ours.
#
# MEASURED (2026-09-08): our ff_t/ff_f differ from NEMO's by at most 3 ulp /
# 2.7e-20 absolute.  The cause is READ OFF THE BINARY, not inferred:
#
#   objdump --disassemble='__usrdef_hgr_MOD_usr_def_hgr' BLD/bin/nemo.exe
#     -> 14x _ZGVbN2v_sin@plt   (glibc libmvec, 2-wide VECTOR sine, <=4 ulp)
#        2x sin@plt, 2x cos@plt, 2x asin@plt, 2x tanh@plt  (scalar)
#
# The whole-array assignments `pff_f(:,:) = 2.*omega*SIN(rad*pphif(:,:))`
# (usrdef_hgr.F90:152-153) are vectorised by `-O3` into the VECTOR sine, while
# the elementwise cos/asin/tanh inside the DO_2D loop stay scalar -- which is
# exactly why every other field here is bit-exact and only these two are not.
# glibc's scalar sin is correctly rounded on all 199 inputs; the sine implied
# by NEMO's output straddles it symmetrically (ulp offsets -3:2 -2:15 -1:56
# 0:71 +1:53 +2:2), the signature of an approximate vector routine rather than
# of a wrong constant (a wrong `omega` would be one-signed).
#
# NOT ADOPTED DELIBERATELY: calling libmvec would reproduce the file exactly,
# and was demonstrated to (0/10348 unequal through a C shim), but it would make
# legoESM's mesh depend on the HOST's glibc version.  A scalar-math NEMO
# rebuild is the fix that belongs on the oracle's side; see
# `docs/ocean/fidelity/dino_setup_audit.md`.
#
# NOTE this residual never enters legoESM: the bridge BUILDS its Coriolis from
# `gphit` (which is bit-exact) and only reads `ff_t` to check itself.
FF_ULP_WAIVER = 4

# Rule 1 dispositions for every variable mesh_mask.nc carries.  VERIFIED rows
# are checked below; WAIVED rows carry the reason here.  Anything the file has
# that is in neither list fails the coverage section.
WAIVED = {
    "time_counter": "file bookkeeping, not a domain field",
}


def _ulp(a: np.ndarray, b: np.ndarray) -> int:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    return int(np.max(np.abs(a.view(np.int64) - b.view(np.int64))))


class Table:
    def __init__(self):
        self.rows, self.failed = [], False

    def check(self, name, built, oracle, *, ulp_waiver=0, note=""):
        built = np.asarray(built)
        oracle = np.asarray(oracle)
        if built.shape != oracle.shape:
            self.rows.append((name, built.size, float("nan"), -1,
                              f"SHAPE {built.shape} vs {oracle.shape}"))
            self.failed = True
            return
        ne = int(np.sum(built != oracle))
        mx = float(np.max(np.abs(built.astype(np.float64)
                                 - oracle.astype(np.float64)))) if ne else 0.0
        # ulp is only meaningful for a ROUNDING-scale gap; a whole-value
        # difference would print a meaningless 4.6e18 int-view distance.
        ulp = (_ulp(built, oracle)
               if ne and np.issubdtype(built.dtype, np.floating)
               and mx < 1e-9 else 0)
        verdict = "EXACT"
        if ne:
            if ulp_waiver and ulp <= ulp_waiver:
                verdict = f"WAIVED<={ulp_waiver}ulp"
            else:
                verdict = "FAIL"
                self.failed = True
        self.rows.append((name, ne, mx, ulp, verdict + (" " + note if note else "")))

    def report(self):
        print(f"{'field':12s} {'cells!=':>9s} {'max|d|':>12s} {'ulp':>4s}  verdict")
        for n, ne, mx, ulp, v in self.rows:
            print(f"{n:12s} {ne:9d} {mx:12.3e} {ulp:4d}  {v}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-mask", default=DEFAULT_MESH)
    ap.add_argument("--plant", action="store_true",
                    help="flip one wet cell of the transcribed mask; the gate "
                         "MUST then exit non-zero (non-vacuity self-test)")
    args = ap.parse_args()

    # Rule 1c: an oracle comparison runs fp64, and it says so.
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask
    from legoesm.ocean.experiments import dino as dino_mod

    if args.plant:
        _true = ndm.nemo_dino_mesh

        def _planted(nml=ndm.NEMO_DINO_R1):
            g = _true(nml)
            tm = g.tmask.copy()
            # Dry the DEEPEST wet cell of one interior column: a one-cell lie
            # about the bathymetry that keeps the column top-contiguous, so it
            # slips past the bridge's own full-step topology guard and has to
            # be caught by THIS gate rather than by an exception.
            j, i = 100, 25
            k = int(tm[j, i].sum()) - 1
            tm[j, i, k] = 0.0
            um = tm * np.roll(tm, -1, axis=1)
            vm = np.zeros_like(tm)
            vm[:-1] = tm[:-1] * tm[1:]
            return g._replace(tmask=tm, umask=um, vmask=vm)

        ndm.nemo_dino_mesh = _planted
        dino_mod.nemo_faithful_dino_domain.cache_clear()
        print("PLANT ACTIVE: one interior surface cell flipped\n")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    with open(args.mesh_mask, "rb") as fh:
        mesh_sha = hashlib.sha256(fh.read()).hexdigest()[:16]
    print(f"legoESM {sha}   mesh_mask {args.mesh_mask}")
    print(f"mesh_mask sha256[:16] = {mesh_sha}   precision = fp64\n")

    import netCDF4 as nc
    d = nc.Dataset(args.mesh_mask)
    O = lambda k: np.asarray(d[k][0])                     # strip time axis
    O3 = lambda k: np.moveaxis(np.asarray(d[k][0]), 0, -1)   # (z,y,x) -> (y,x,z)

    nml = ndm.NEMO_DINO_R1
    n_lon, n_lat, jpk = ndm.nemo_dino_domain_size(nml)
    print(f"SECTION 0  domain size from usr_def_nam: "
          f"{n_lon} x {n_lat} x {jpk}")
    dims_ok = (len(d.dimensions["x"]), len(d.dimensions["y"]),
               len(d.dimensions["nav_lev"])) == (n_lon, n_lat, jpk)
    print(f"           mesh_mask dimensions match: {dims_ok}\n")

    hgr = ndm.nemo_dino_hgr(nml)
    lad = ndm.vertical_ladders(nml, jpk)
    bathy, k_top, k_bot = ndm.nemo_dino_bathymetry(nml, hgr, lad["gdept_1d"])
    g = ndm.nemo_dino_mesh(nml)

    t = Table()
    print("SECTION 1+2  bit-exactness vs NEMO mesh_mask.nc")
    for f in ("glamt", "glamu", "glamv", "glamf",
              "gphit", "gphiu", "gphiv", "gphif",
              "e1t", "e1u", "e1v", "e1f", "e2t", "e2u", "e2v", "e2f"):
        t.check(f, hgr[f], O(f))
    for f in ("ff_t", "ff_f"):
        t.check(f, hgr[f], O(f), ulp_waiver=FF_ULP_WAIVER,
                note="(NEMO binary's own SIN; never consumed by legoESM)")
    for f in ("gdept_1d", "gdepw_1d", "e3t_1d", "e3w_1d"):
        t.check(f, lad[f], O(f).ravel())
    for f, src in (("gdept_0", "gdept_0"), ("gdepw_0", "gdepw_0"),
                   ("e3t_0", "e3t_0"), ("e3w_0", "e3w_0")):
        t.check(f, np.broadcast_to(lad[src][None, None, :],
                                   (n_lat, n_lon, jpk)), O3(f))
    # e3tw_to_other_e3 (zgr_lib.F90:206-264): column averages of a
    # horizontally-uniform ladder, so u/v/f/uw/vw equal e3t_0 / e3w_0.
    for f, src in (("e3u_0", "e3t_0"), ("e3v_0", "e3t_0"), ("e3f_0", "e3t_0"),
                   ("e3uw_0", "e3w_0"), ("e3vw_0", "e3w_0")):
        t.check(f, np.broadcast_to(lad[src][None, None, :],
                                   (n_lat, n_lon, jpk)), O3(f))
    t.check("tmask", g.tmask, O3("tmask"))
    t.check("umask", g.umask, O3("umask"))
    t.check("vmask", g.vmask, O3("vmask"))
    # fmask = tmask(i,j)*tmask(i+1,j)*tmask(i,j+1)*tmask(i+1,j+1) (dommsk:152)
    _tm = g.tmask
    _te = np.roll(_tm, -1, axis=1)
    fmask = np.zeros_like(_tm)
    fmask[:-1] = _tm[:-1] * _te[:-1] * _tm[1:] * _te[1:]
    t.check("fmask", fmask, O3("fmask"))
    t.check("tmaskutil", g.tmask.max(axis=-1), O("tmaskutil"))
    t.check("umaskutil", g.umask.max(axis=-1), O("umaskutil"))
    t.check("vmaskutil", g.vmask.max(axis=-1), O("vmaskutil"))
    # mbathy = MAX(k_bot,1); the N/S closure zeroes k_top only (domzgr:315),
    # so mbathy keeps the bathymetric level on the first/last row.
    t.check("mbathy", np.maximum(k_bot, 1), O("mbathy"))
    # float32 copies NEMO writes for CF conventions
    t.check("nav_lon", hgr["glamt"].astype(np.float32),
            np.asarray(d["nav_lon"][:]))
    t.check("nav_lat", hgr["gphit"].astype(np.float32),
            np.asarray(d["nav_lat"][:]))
    t.check("nav_lev", lad["gdept_1d"].astype(np.float32),
            np.asarray(d["nav_lev"][:]))
    # These two have no legoESM counterpart, so they are checked against the
    # constant the config forces rather than against a built field: `misf` is
    # the ice-shelf top level (1 everywhere -- ld_isfcav=F, no cavities) and
    # `stiffness` is domzgr's Haney-number diagnostic (0 on a full-step grid).
    # A non-constant here would mean the run was NOT the config we transcribe.
    t.check("misf", np.ones((n_lat, n_lon), dtype=np.int32), O("misf"))
    t.check("stiffness", np.zeros((n_lat, n_lon)), O("stiffness"))
    t.report()

    checked = {r[0] for r in t.rows}
    unaccounted = sorted(set(d.variables) - checked - set(WAIVED))
    print(f"\nSECTION 1  coverage: {len(checked)} verified, "
          f"{len(WAIVED)} waived, {len(unaccounted)} unaccounted")
    for k, why in WAIVED.items():
        print(f"  WAIVED  {k}: {why}")
    if unaccounted:
        print(f"  UNACCOUNTED (hard failure): {unaccounted}")
        t.failed = True

    # ---- SECTION 3: Rule 10 -- through run_dino.py's OWN path -------------
    print("\nSECTION 3  the driver's own path "
          "(dino_lat_lon_grid / _vertical / _state)")
    cfg = dino_mod.nemo_faithful_dino_config(
        base=dino_mod.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dino_mod.dino_lat_lon_grid(cfg)
    z = dino_mod.dino_lat_lon_vertical(grid, cfg)
    st = dino_mod.dino_lat_lon_state(grid, z, cfg)
    print(f"  built {type(grid).__name__} {grid.n_lat}x{grid.n_lon}, "
          f"{z.n_levels} levels, dtype {np.asarray(grid.lat).dtype}")

    # (a) identical to the domain the certified twin gets from the FILE, leaf
    #     for leaf -- the strongest statement of "no second convention".
    from legoesm.ocean.fidelity.nemo_io import NemoState
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        bridge_nemo_to_legoesm_topo)
    fg = read_nemo_mesh_mask(args.mesh_mask, nn_hls=0)
    _z = np.zeros(fg.tmask.shape)
    file_br = bridge_nemo_to_legoesm_topo(
        fg, NemoState(T=_z, S=_z, u=_z, v=_z,
                      ssh=np.zeros(fg.tmask.shape[:2]), rhd=None),
        periodic_i=True, full_step=True)
    ours = dino_mod.nemo_faithful_dino_domain()
    import jax.tree_util as _jtu

    def _same(a, b):
        """Leaf-for-leaf equality, descending into nested pytrees."""
        la = _jtu.tree_leaves(a)
        lb = _jtu.tree_leaves(b)
        if len(la) != len(lb):
            return False
        for x, y in zip(la, lb):
            x, y = np.asarray(x), np.asarray(y)
            if x.shape != y.shape or not bool(np.array_equal(x, y)):
                return False
        return _jtu.tree_structure(a) == _jtu.tree_structure(b)

    n_diff = 0
    for label, a_obj, b_obj in (("geometry", ours.geometry, file_br.geometry),
                                ("z_coord", ours.z_coord, file_br.z_coord)):
        for fld in a_obj._fields:
            if not _same(getattr(a_obj, fld), getattr(b_obj, fld)):
                print(f"  DIFF  {label}.{fld}")
                n_diff += 1
    same_mask = bool(np.array_equal(np.asarray(ours.land_mask),
                                    np.asarray(file_br.land_mask)))
    print(f"  analytic domain == file-read domain: "
          f"{n_diff} differing leaves, land_mask equal: {same_mask}")
    if n_diff or not same_mask:
        t.failed = True

    # (b) the certified twin's own census (kamm_run5y_v3.py:29-42)
    tmN = O3("tmask") > 0.5
    wet = (np.asarray(z.is_active)
           & (np.asarray(st.land_mask.data) > 0.5)[:, :, None])
    umN = O3("umask")[:, :, 0] > 0.5
    vmN = O3("vmask")[:, :, 0] > 0.5
    umL = np.asarray(st.u_mask.data)[:, 1:n_lon + 1] > 0.5
    vmL = np.asarray(st.v_mask.data)[1:n_lat + 1, :] > 0.5
    census = [("3-D wet cells", int(np.sum(wet != tmN))),
              ("surface u-face mask", int(np.sum(umL != umN))),
              ("surface v-face mask", int(np.sum(vmL != vmN)))]
    for name, ne in census:
        print(f"  {name:22s} cells differing from NEMO: {ne}")
        if ne:
            t.failed = True
    print(f"  wet cells: {int(tmN.sum())} 3-D / "
          f"{int(tmN[:, :, 0].sum())} surface")

    print("\nGATE " + ("FAIL" if t.failed else "PASS"))
    return 1 if t.failed else 0


if __name__ == "__main__":
    sys.exit(main())
