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
# rebuild is the fix that belongs on the oracle's side --
# `scripts/experiment/dino/nemo_scalar_math_rebuild.sh`, which pre-registers
# what confirms and what refutes this diagnosis.
#
# SCOPE OF THE WAIVER -- CORRECTED 2026-09-10, IT IS NOW LOAD-BEARING.
# `ff_t` still never reaches a tendency: the DINO bridge
# `bridge_nemo_to_legoesm_topo` reads it at ONE site (nemo_state_bridge.py:616)
# as a tolerance check of the Coriolis it BUILT from `gphit`, which is
# bit-exact.  `ff_f` is different.  Filling NemoGrid's optional half (without
# which the standalone card aborts at step 1 on `nemo_e3w_source=
# 'mesh_reference'`) makes `_nemo_een_barotropic_operands` available on the
# analytic path as it already was on the file-read path, and that operand
# bundle carries `ff_f` RAW into the EEN barotropic Coriolis.  So this residual
# DOES now enter a tendency -- measured 5824 of 10348 vertices differing, max 2
# ulp / 2.711e-20 s^-1 -- and the sentence that used to stand here ("no ff_t
# value reaches a tendency", read as covering both) no longer covers it.
# It is a BOUND, not a licence: section 3 checks the same <=FF_ULP_WAIVER
# bound on that leaf and fails outside it.  The fix belongs on the oracle's
# side -- `scripts/experiment/dino/nemo_scalar_math_rebuild.sh`.
#
# The sibling beta-plane bridge `bridge_nemo_to_legoesm` (nemo_state_bridge.py:
# 111) also consumes both arrays, but serves GYRE, does not use this
# transcription, and is out of this gate's scope.
FF_ULP_WAIVER = 4

# Rule 1 dispositions for every variable mesh_mask.nc carries.  VERIFIED rows
# are checked below; WAIVED rows carry the reason here.  Anything the file has
# that is in neither list fails the coverage section.
WAIVED = {
    "time_counter": "file bookkeeping, not a domain field",
}


def _monotone(x: np.ndarray) -> np.ndarray:
    """IEEE-754 bit pattern remapped so integer order == float order.

    The naive `a.view(int64) - b.view(int64)` is NOT a ulp distance across
    zero: +/-5e-11 are two ulps apart as floats but 2^63 apart as raw
    patterns, so a SIGN FLIP produced a huge NEGATIVE difference that an
    `abs(...) <= waiver` test could wave through.  ff_f never crosses zero
    here (min |ff_f| = 1.27e-6 against a 2.7e-20 residual), so the waiver was
    right by luck rather than by construction; this makes it right by
    construction.
    """
    i = np.ascontiguousarray(x, dtype=np.float64).view(np.int64)
    return np.where(i < 0, np.int64(-(2 ** 63)) - i, i)


def _ulp(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.max(np.abs(_monotone(np.asarray(a, dtype=np.float64))
                             - _monotone(np.asarray(b, dtype=np.float64)))))


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
        # difference would print a meaningless 4.6e18 int-view distance, so it
        # is reported as 0 there.  ``rounding_scale`` therefore has to gate the
        # WAIVER as well: without it a zeroed, doubled or hemisphere-flipped
        # ff_f (max|d| ~ 1.4e-04, i.e. 100% wrong) got ulp=0 and passed
        # ``0 <= FF_ULP_WAIVER``.  The waiver is a BOUND, not an escape hatch.
        rounding_scale = (ne and np.issubdtype(built.dtype, np.floating)
                          and mx < 1e-9)
        ulp = _ulp(built, oracle) if rounding_scale else 0
        verdict = "EXACT"
        if ne:
            if ulp_waiver and rounding_scale and ulp <= ulp_waiver:
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
    ap.add_argument("--plant-dry-level", action="store_true",
                    help="set the CONSTRUCTED ladder's dry level to an "
                         "arbitrary value; the waiver MUST refuse it")
    ap.add_argument("--plant-ladder", action="store_true",
                    help="nudge ONE level of the CONSTRUCTED vertical ladder "
                         "by 1 ulp; section 4 MUST then exit non-zero "
                         "(non-vacuity self-test for the geometry rows)")
    args = ap.parse_args()

    # Rule 1c: an oracle comparison runs fp64, and it says so.
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    from legoesm.ocean.experiments import dino as dino_mod
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask  # noqa: F401

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
        dino_mod._nemo_faithful_dino_domain.cache_clear()
        print("PLANT ACTIVE: one interior surface cell flipped\n")

    if args.plant_dry_level:
        # The ONE thing the scored rows deliberately do not cover.  A dry
        # level is now SCORED like every other level (the waiver is gone), so
        # this plant must make the two ladder rows go red.  It is kept because
        # it is the only plant that targets the dry level specifically.
        from legoesm.ocean.fidelity import nemo_state_bridge as _nsb2
        _true_dry = _nsb2.effective_vertical_scale_factors

        def _planted_dry(grid, tmask, mode=None):
            e3t, td, src = _true_dry(grid, tmask, mode=mode)
            lev = np.asarray(tmask).any(axis=(0, 1))
            e3t = e3t.copy()
            td = td.copy()
            e3t[~lev] = 300.0
            td[~lev] = td[~lev] * 1.01
            return e3t, td, src

        _nsb2.effective_vertical_scale_factors = _planted_dry
        dino_mod._nemo_faithful_dino_domain.cache_clear()
        print("DRY-LEVEL PLANT ACTIVE: the dry level holds neither NEMO "
              "ladder's value\n")

    if args.plant_ladder:
        # Plant INSIDE the construction, not on the arrays after it: the point
        # is to prove section 4 reads the ladder the card is built from.  The
        # raw z_coord.nemo_* mesh fields are NOT derived from this return
        # value, so their rows stay green -- that asymmetry is the evidence
        # that the derived rows are the ones being tested.
        from legoesm.ocean.fidelity import nemo_state_bridge as _nsb
        _true_evsf = _nsb.effective_vertical_scale_factors

        def _planted_evsf(grid, tmask, mode=None):
            # EVERY level by 1 ulp, not one level.  Measured: a 1-ulp plant on
            # the surface level alone is 2.2e-16 m, which vanishes inside a
            # 4000 m column sum (ulp 4.5e-13), so it never reaches ht_0/hu_0
            # and the plant would certify less than it appears to.  Planting
            # every level moves the column depth on all 9920 wet columns.
            e3t, td, src = _true_evsf(grid, tmask, mode=mode)
            return (np.nextafter(e3t, np.inf), np.nextafter(td, np.inf), src)

        _nsb.effective_vertical_scale_factors = _planted_evsf
        dino_mod._nemo_faithful_dino_domain.cache_clear()
        print("LADDER PLANT ACTIVE: EVERY level of the constructed ladder "
              "moved by 1 ulp\n")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    with open(args.mesh_mask, "rb") as fh:
        mesh_sha = hashlib.sha256(fh.read()).hexdigest()[:16]
    print(f"legoESM {sha}   mesh_mask {args.mesh_mask}")
    print(f"mesh_mask sha256[:16] = {mesh_sha}   precision = fp64\n")

    import netCDF4 as nc
    d = nc.Dataset(args.mesh_mask)
    def O(k):        # strip the time axis
        return np.asarray(d[k][0])

    def O3(k):       # (z,y,x) -> (y,x,z), matching nemo_io
        return np.moveaxis(np.asarray(d[k][0]), 0, -1)

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
                note="(NEMO binary's own vector SIN; the DINO bridge reads "
                     "ff_t only to check itself)")
    for f in ("gdept_1d", "gdepw_1d", "e3t_1d", "e3w_1d"):
        t.check(f, lad[f], O(f).ravel())
    for f, src in (("gdept_0", "gdept_0"), ("gdepw_0", "gdepw_0"),
                   ("e3t_0", "e3t_0"), ("e3w_0", "e3w_0")):
        t.check(f, np.broadcast_to(lad[src][None, None, :],
                                   (n_lat, n_lon, jpk)), O3(f))
    # e3tw_to_other_e3 (zgr_lib.F90:206-264): column averages of a
    # horizontally-uniform ladder, so u/v/f/uw/vw equal e3t_0 / e3w_0.
    for f, src in (("e3f_0", "e3t_0"), ("e3uw_0", "e3w_0"), ("e3vw_0", "e3w_0")):
        t.check(f, np.broadcast_to(lad[src][None, None, :],
                                   (n_lat, n_lon, jpk)), O3(f))
    # e3u_0/e3v_0 are checked as the ARRAYS THE NemoGrid CARRIES, not as the
    # ladder they were built from: the two are only the same if the assembly
    # into NemoGrid is also right, and this is the gate's only sight of it.
    t.check("e3u_0", g.e3u_0, O3("e3u_0"))
    t.check("e3v_0", g.e3v_0, O3("e3v_0"))
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
    # hu_0/hv_0 are DERIVED, not dumped (nemo_io.py:174-188 sums e3u_0*umask),
    # so the oracle side is the file-read NemoGrid rather than a mesh_mask
    # variable -- but they still have to be checked, because two probes read
    # them (momentum_jacobian_probe.py, acc_momentum_budget.py) and nothing
    # else in this gate would notice them zeroed or swapped.
    _fg = read_nemo_mesh_mask(args.mesh_mask, nn_hls=0)
    for f in ("hu_0", "hv_0"):
        t.check(f, getattr(g, f), getattr(_fg, f))
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
    from legoesm.core.precision import get_policy
    print(f"  built {type(grid).__name__} {grid.n_lat}x{grid.n_lon}, "
          f"{z.n_levels} levels, dtype {np.asarray(grid.lat).dtype}")
    print(f"  precision policy storage={get_policy().storage.__name__} -- this "
          "section certifies the CONSTRUCTION at this policy; run_dino.py "
          "forces fp64 for every NEMO-fidelity run, and the analytic domain "
          "now REFUSES to build below fp64 (its gdept_0/e3w_0 identity misses "
          "by 1.2e-4 m at float32).")

    # (a) identical to the domain the certified twin gets from the FILE, leaf
    #     for leaf -- the strongest statement of "no second convention".
    from legoesm.ocean.fidelity.nemo_io import NemoState
    from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
    fg = read_nemo_mesh_mask(args.mesh_mask, nn_hls=0)
    _z = np.zeros(fg.tmask.shape)
    file_br = bridge_nemo_to_legoesm_topo(
        fg, NemoState(T=_z, S=_z, u=_z, v=_z,
                      ssh=np.zeros(fg.tmask.shape[:2]), rhd=None),
        periodic_i=True, full_step=True, carry_native_lat_deg=True,
        # Same ladder as the card pins (#1728); without this the two sides of
        # the comparison would stand on DIFFERENT vertical grids and the check
        # would fail for a reason that has nothing to do with the mesh
        # transcription it exists to test.
        e3t_mode="both")
    ours = dino_mod.nemo_faithful_dino_domain()
    import jax.tree_util as _jtu

    def _same(a, b, ulp_waiver=0):
        """Leaf-for-leaf equality, descending into nested pytrees.

        ``ulp_waiver`` admits ONLY the documented ff_f residual (see
        FF_ULP_WAIVER above), and only as a BOUND: a leaf that differs by
        more than that, or by more than a rounding-scale amount, still fails.
        """
        la = _jtu.tree_leaves(a)
        lb = _jtu.tree_leaves(b)
        if len(la) != len(lb):
            return False
        for x, y in zip(la, lb):
            x, y = np.asarray(x), np.asarray(y)
            if x.shape != y.shape:
                return False
            if bool(np.array_equal(x, y)):
                continue
            if not ulp_waiver or not np.issubdtype(x.dtype, np.floating):
                return False
            mx = float(np.max(np.abs(x.astype(np.float64)
                                     - y.astype(np.float64))))
            if mx >= 1e-9 or _ulp(x, y) > ulp_waiver:
                return False
            print(f"  WAIVED<={ulp_waiver}ulp  leaf differs by "
                  f"{_ulp(x, y)} ulp / {mx:.3e} (the NEMO binary's ff_f)")
        return _jtu.tree_structure(a) == _jtu.tree_structure(b)

    n_diff = 0
    for label, a_obj, b_obj in (("geometry", ours.geometry, file_br.geometry),
                                ("z_coord", ours.z_coord, file_br.z_coord)):
        for fld in a_obj._fields:
            # nemo_een_barotropic carries ff_f RAW (nemo_state_bridge.py:69-83);
            # it is the ONE leaf the documented ff_f waiver reaches.
            waiver = (FF_ULP_WAIVER if fld == "nemo_een_barotropic" else 0)
            if not _same(getattr(a_obj, fld), getattr(b_obj, fld),
                         ulp_waiver=waiver):
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

    # ---- SECTION 4: the geometry the MODEL CONSUMES ----------------------
    #
    # Section 1+2 proves the TRANSCRIPTION equals NEMO's mesh.  It is blind to
    # whether the model RUNS on it (oracle-fidelity Rule 2: a bit-exact state
    # gate cannot see the boxes holding the state).  For months it did not:
    # the card cut its staircase from the 1-D reference ladder e3t_1d while
    # NEMO's DINO integrates the 3-D e3t_0 -- `key_qco key_vco_3d`
    # (cfgs/DINO/cpp_DINO.fcm:1) makes `E3t_0(i,j,k)` expand to `e3t_3d`
    # (domzgr_substitute.h90:102) and leaves e3t_1d reachable only under
    # key_vco_1d (:72).
    #
    # So this section scores the geometry taken from the domain the driver's
    # own path CONSTRUCTS (grid/z/st above) against mesh_mask.nc, field by
    # field, at the same bar as everything else: ZERO CELLS UNEQUAL.
    print("\nSECTION 4  the geometry the MODEL CONSUMES "
          "(NEMO key_qco key_vco_3d)")
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_vface
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        compute_layer_thickness, min_cell_to_uface)

    g4 = Table()
    tmask3 = O3("tmask") > 0.5
    umask3 = O3("umask") > 0.5
    vmask3 = O3("vmask") > 0.5
    e3t0, gdept0, gdepw0 = O3("e3t_0"), O3("gdept_0"), O3("gdepw_0")
    e3w0, e3u0, e3v0 = O3("e3w_0"), O3("e3u_0"), O3("e3v_0")
    e3uw0, e3vw0 = O3("e3uw_0"), O3("e3vw_0")
    lev_any = tmask3.any(axis=(0, 1))

    # C1 -- the live layer thickness.  NEMO: e3t(i,j,k,t) = E3t_0*(1+r3t*tmask)
    # (domzgr_substitute.h90:45 Tmsk, :102 E3t_0=e3t_3d; r3t = ssh*r1_ht_0,
    # domqco.F90:160).  At rest r3t = 0, so the reference half is all of it.
    # Rule 10: the model's OWN statement, with the card's own floor -- not a
    # re-derivation with a convenient constant.
    mc4, _ = dino_mod.dino_lat_lon_model_config(grid, cfg, physics=True)
    h_live = np.asarray(compute_layer_thickness(
        st.eta.data, st.H_bathy.data, z,
        min_water_column_m=mc4.min_water_column_m))
    g4.check("e3t(Kmm) live thickness", h_live * tmask3, e3t0 * tmask3)
    # The 1-D ladder the staircase is cut from, per level, against the value
    # NEMO's own 3-D array holds at that level.
    #
    # THE ORACLE SIDE IS NEVER BUILT FROM THE MODEL SIDE.  An earlier version
    # of this row fell back to `dz[k]` on a level with no wet cell, i.e. it
    # compared the model's ladder against ITSELF and printed EXACT; a reviewer
    # planted an arbitrary 300 m at every dry level and the whole section
    # still passed.  e3t_0/gdept_0 are horizontally uniform over the WHOLE
    # array, dry cells included (measured spread exactly 0.0 at every level),
    # so there is always a real oracle value to take.
    dz = np.asarray(z.dz_ref).ravel()
    nlev4 = e3t0.shape[-1]
    for _nm, _a in (("e3t_0", e3t0), ("gdept_0", gdept0), ("e3w_0", e3w0)):
        _sp = float(np.max(_a.max(axis=(0, 1)) - _a.min(axis=(0, 1))))
        if _sp != 0.0:
            print(f"  {_nm} is NOT horizontally uniform (max spread {_sp:.3e} "
                  "m): the per-level rows below are not well defined")
            g4.failed = True
    e3t_lev = e3t0[0, 0, :]
    gdept_lev = gdept0[0, 0, :]
    td4_all = np.asarray(z.t_depth_ref).ravel()
    # NO WAIVER.  The permanently-dry level takes NEMO's own value like every
    # other level (user decision 2026-09-10: "do exactly what NEMO does"), so
    # the two ladder rows below are scored over ALL 36 levels, not over the
    # wet ones only.  What used to be waived here was a real 111.088 m /
    # 55.544 m gap on level 36; it is now zero, and the rows themselves are
    # what proves it.  --plant-dry-level remains the non-vacuity check: it
    # sets the dry level to an arbitrary value and these rows must go red.
    dry = ~lev_any
    if dry.any():
        act = np.asarray(z.is_active)
        hp = np.asarray(z.h_partial)
        print(f"  {int(dry.sum())} permanently-dry level(s): "
              f"{list(1 + np.where(dry)[0])} -- scored, not waived. They DO "
              "set H_max and the three deepest reference interfaces, which "
              "is why they are scored.")
        # THESE TWO HARD FAILS ARE INDEPENDENT OF THE WAIVER AND SURVIVE IT.
        # A dry level in the MIDDLE of a column is a different object (it
        # would sit between two integrated cells); and a dry level that
        # carries an active cell or a thickness is being integrated after
        # all.  Deleting the waiver deleted neither check -- a review of the
        # first version of this diff found they had become print-only, which
        # would have made the gate weaker than the version it replaced.
        if not np.all(dry[len(dry) - int(dry.sum()):]):
            print(f"  dry levels {1 + np.where(dry)[0]} are not the TRAILING "
                  "levels of the column: this is not NEMO's dummy bottom "
                  "level")
            g4.failed = True
        n_act = int(act[..., dry].sum())
        max_hp = float(np.abs(hp[..., dry]).max())
        print(f"  dry levels carry {n_act} active cells and max|h_partial| "
              f"= {max_hp:.3e} m")
        if n_act or max_hp != 0.0:
            print("  a permanently-dry level carries an active cell or a "
                  "thickness on this run: it is being integrated")
            g4.failed = True
        # H_max MOVED when the waiver went (4506.374704 -> 4617.462287 m on
        # this card).  It is only harmless because this card's coordinate is
        # the PARTIAL-CELL one, whose Jacobian is built from h_partial.  On
        # the z-star branch the Jacobian is (eta + H_bathy)/H_max and every
        # cell would move ~2.7%.  So the branch is PRINTED and asserted here
        # rather than left to the reader (review finding).
        from legoesm.ocean.vertical import OceanPartialCellCoordinate
        is_pc = isinstance(z, OceanPartialCellCoordinate)
        print(f"  H_max = {float(z.H_max):.6f} m, "
              f"z_half_ref[-1] = {float(np.asarray(z.z_half_ref)[-1]):.6f}, "
              f"z_full_ref[-1] = {float(np.asarray(z.z_full_ref)[-1]):.6f}, "
              f"dz_half_ref[-1] = "
              f"{float(np.asarray(z.dz_half_ref)[-1]):.6f}; coordinate is "
              f"{type(z).__name__} (partial-cell: {is_pc})")
        if not is_pc:
            print("  this card is NOT on the partial-cell coordinate, so "
                  "H_max enters the vertical Jacobian directly and the dry "
                  "level is no longer inert: re-measure before trusting any "
                  "row here")
            g4.failed = True
    g4.check("e3t_0 reference ladder", dz, e3t_lev)
    # C5/C6 -- the S-EOS depth (eosbn2.F90:297 zh = gdept(Knn)) and the sco
    # pressure-gradient depth (dynhpg.F90:353/378 gdept_z0), both read from
    # z_coord.t_depth_ref (eos.py:823, ocean_pe_latlon_cgrid.py:1900-1906).
    g4.check("gdept_0 T-depth ladder", td4_all, gdept_lev)
    # raw mesh fields the fidelity arms read straight off the coordinate
    g4.check("gdepw_0 (z_coord raw)", np.asarray(z.nemo_gdepw_0), gdepw0)
    g4.check("e3w_0 (z_coord raw)", np.asarray(z.nemo_e3w_0), e3w0)
    g4.check("e3t_0 (z_coord raw)", np.asarray(z.nemo_e3t_0), e3t0)
    # e3uw_0/e3vw_0: zgr_lib.F90:206-264 builds them as column averages of a
    # horizontally-uniform e3w_0, so the raw e3w_0 the momentum vertical solve
    # interpolates to the face IS them.  Checked, not assumed.
    #
    # WHAT THESE TWO ROWS DO AND DO NOT TEST.  The left side is the array the
    # coordinate CARRIES, so they catch it being absent, zeroed, truncated or
    # swapped -- they do NOT test the face interpolation, which on a uniform
    # ladder is the identity either way.
    g4.check("e3uw_0 == e3w_0", np.asarray(z.nemo_e3w_0), e3uw0)
    g4.check("e3vw_0 == e3w_0", np.asarray(z.nemo_e3w_0), e3vw0)
    # C9 -- the EEN relative-vorticity metric.  ln_dynvor_een = .true.
    # (namelist_cfg:331), so e3f_0vor divides relative vorticity at every
    # f-point and is built from the 3-D e3t_0.  The card carries NEMO's own
    # e3f_0 raw in the EEN barotropic operand bundle.
    _een = getattr(z, "nemo_een_barotropic", None)
    if _een is None:
        g4.rows.append(("e3f_0 (EEN operand)", -1, float("nan"), 0,
                        "FAIL no nemo_een_barotropic on this coordinate"))
        g4.failed = True
    else:
        g4.check("e3f_0 (EEN operand)", np.asarray(_een.e3f_0), O3("e3f_0"))
    # C3 -- e3u_0/e3v_0 as the model builds them: the shallower neighbour's
    # reference thickness (vertical.py nemo_qco_card_mesh_operands).
    # BLIND SPOT, declared: NEMO builds e3u_0 = 0.5*(e3t(i)+e3t(i+1))
    # (zgr_lib.F90:230), not the minimum taken here.  On a horizontally
    # uniform ladder the two coincide exactly, so this row goes green WITHOUT
    # testing the operator; on a non-uniform card it would not.
    h_ref = jnp.asarray(np.asarray(z.h_partial))
    e3u_live = np.asarray(min_cell_to_uface(h_ref))[:, 1:, :]
    e3v_live = np.asarray(min_cell_to_vface(h_ref, grid))[1:, :, :]
    g4.check("e3u_0 live face thickness", e3u_live * umask3, e3u0 * umask3)
    g4.check("e3v_0 live face thickness", e3v_live * vmask3, e3v0 * vmask3)
    # C2/C3 -- the column depths.  domain.F90:144-145.
    ht0 = (e3t0 * tmask3).sum(axis=-1)
    hu0 = (e3u0 * umask3).sum(axis=-1)
    hv0 = (e3v0 * vmask3).sum(axis=-1)
    g4.check("ht_0 column depth", np.asarray(st.H_bathy.data), ht0)
    g4.check("hu_0 (z_coord raw)", np.asarray(z.nemo_hu_0), hu0)
    g4.check("hv_0 (z_coord raw)", np.asarray(z.nemo_hv_0), hv0)
    g4.check("hu_0 live face sum", (e3u_live * umask3).sum(axis=-1), hu0)
    g4.check("hv_0 live face sum", (e3v_live * vmask3).sum(axis=-1), hv0)
    # C4 -- the reciprocals the qco barotropic path divides by.
    # NEMO: r1_hu_0 = ssumask/(hu_0 + 1 - ssumask)   (domain.F90:159).
    # legoESM: wet_u/(hu_0 + 1 - wet_u) with wet_u = (hu_0 > 0)
    # (vertical.py:180-182).
    #
    # WHAT THIS ROW TESTS: the PREDICATE and the OPERAND, not the formula.
    # legoESM's two sites (vertical.py:181, ocean_model_latlon_cgrid.py:7516)
    # build the reciprocal inside traced functions, so the formula is written
    # out again here rather than called -- which means an arithmetic change at
    # those sites would NOT show up.  What it does catch, and what is the
    # actual question on this card, is whether legoESM's wet-U predicate
    # (hu_0 > 0) is NEMO's own ssumask and whether hu_0 itself is NEMO's.
    ssu, ssv = O("umaskutil"), O("vmaskutil")
    mh_u = np.asarray(z.nemo_hu_0)
    mh_v = np.asarray(z.nemo_hv_0)
    wet_u = (mh_u > 0.0).astype(np.float64)
    wet_v = (mh_v > 0.0).astype(np.float64)
    g4.check("r1_hu_0", wet_u / (mh_u + 1.0 - wet_u),
             ssu / (hu0 + 1.0 - ssu))
    g4.check("r1_hv_0", wet_v / (mh_v + 1.0 - wet_v),
             ssv / (hv0 + 1.0 - ssv))
    g4.report()
    if g4.failed:
        t.failed = True
    if args.plant_ladder:
        # Non-vacuity is checked ROW BY ROW, not by "the section failed": a
        # plant that trips one row while leaving the rest green would still
        # read as a pass here, and those rows would be certifying nothing.
        # The raw z_coord.nemo_* fields are deliberately NOT in this set --
        # they are carried straight off the mesh and are not derived from the
        # planted ladder, so they MUST stay green, and that asymmetry is what
        # shows the derived rows are the ones under test.
        must_fail = {
            "e3t(Kmm) live thickness", "e3t_0 reference ladder",
            "gdept_0 T-depth ladder", "e3u_0 live face thickness",
            "e3v_0 live face thickness", "ht_0 column depth",
            "hu_0 live face sum", "hv_0 live face sum",
        }
        verdicts = {r[0]: r[4] for r in g4.rows}
        survived = sorted(n for n in must_fail
                          if not verdicts.get(n, "").startswith("FAIL"))
        unexpected = sorted(n for n, v in verdicts.items()
                            if n not in must_fail and v.startswith("FAIL"))
        print(f"  LADDER PLANT: {len(must_fail) - len(survived)} of "
              f"{len(must_fail)} ladder-derived rows flipped to FAIL")
        if survived:
            print(f"  PLANT DID NOT REACH (rows certify nothing): {survived}")
            t.failed = True
        if unexpected:
            print(f"  PLANT REACHED A RAW MESH ROW (it should not): "
                  f"{unexpected}")
            t.failed = True
        if not survived and not unexpected:
            print("  plant is NON-VACUOUS and correctly scoped")
            # A planted run must still exit non-zero overall.
            t.failed = True

    print("\nGATE " + ("FAIL" if t.failed else "PASS"))
    return 1 if t.failed else 0


if __name__ == "__main__":
    sys.exit(main())
