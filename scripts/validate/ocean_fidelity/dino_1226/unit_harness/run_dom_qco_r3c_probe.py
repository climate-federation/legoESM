#!/usr/bin/env python
"""#1226 NEMO unit-call harness -- ROUTINE 4: dom_qco_r3c (ssh/h_0 ratio at
t-, u-, v-, f-points, ``domqco.F90:140-186``).

Gate row context: ``fidelity_bar_gate.py`` rows "dom_qco_r3c r3t" and
"dom_qco_r3c r3u/r3v" are both PHANTOM-PROVENANCE (their cited script
``probe_1226_r2_item4_domqco.py`` does not exist in the repo tree, confirmed
by ``check_provenance_scripts_exist()``). r3t is measured via the compiled
Fortran driver (``harness_dom_qco_r3c.exe``) against legoESM's existing
:func:`legoesm.ocean.eos.nemo_r3t_stretch`.

r3u/r3v (#1492 evidence audit, P3): EXTENDED (not a second script, per that
task's instruction to consolidate onto this existing r3t probe) to compare
the Fortran driver's own ``r3c_out_r3u.bin``/``r3c_out_r3v.bin`` against a
face-averaged transcription of ``domqco.F90:212-215``'s OWN formula:

    pr3u(i,j) = 0.5*(e1e2t(i,j)*ssh(i,j) + e1e2t(i+1,j)*ssh(i+1,j))
                * r1_hu_0(i,j) * r1_e1e2u(i,j)
    pr3v(i,j) = 0.5*(e1e2t(i,j)*ssh(i,j) + e1e2t(i,j+1)*ssh(i,j+1))
                * r1_hv_0(i,j) * r1_e1e2v(i,j)

This is PROBE-SIDE re-transcription of the oracle's own formula (same class
as ``ww_inheritance_walk.py``'s ``ww`` reconstruction from ``sshwzv.F90``,
or this file's own T-point ``ht_0`` construction two lines below) -- it is
NOT new legoESM production physics: the original pre-impl search (this
docstring, prior task) correctly found no legoESM production function for
the face-averaged ratio, but that constraint only bars adding one under
``packages/``; comparing the Fortran driver's own r3u/r3v output against a
transcription of its OWN documented formula (self-consistency, using
``hu_0``/``hv_0`` -- already read by :mod:`legoesm.ocean.fidelity.nemo_io`,
not re-derived) is in scope. ``e1e2t``/``e1e2u``/``e1e2v`` are not stored
directly in ``mesh_mask.nc``; built here as ``e1*e2`` per NEMO's own naming
convention (``e1e2t = e1t*e2t``, domain.F90).

``dom_qco_r3c`` is a PURE, side-effect-free 2-D routine (no time-level state,
reads only module-global mesh metrics + the caller's ssh) -- same
tractability class as ``eos_rab``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from validate.ocean_fidelity.dino_1226.unit_harness import binary_io  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "packages" / "ocean"))
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.fidelity.precision_gate import require_fp64  # noqa: E402
from legoesm.ocean.eos import nemo_r3t_stretch  # noqa: E402

JPI, JPJ = 56, 203
HLS = 2
MESH_PATH = (
    REPO_ROOT.parent / "oracle-builds" / "nemo5" / "nemo_5.0.2" / "cfgs" / "DINO" / "RUN_GDB" / "mesh_mask.nc"
)


def build_synthetic_ssh(rng: np.random.Generator) -> np.ndarray:
    """Synthetic ssh spanning real DINO dynamic range: a large-scale gyre-like
    pattern (+/- 0.5 m) plus small-scale noise -- NOT the quiescent-restart's
    ssh~0, so r3t is actually exercised at O(1e-4 to 1e-3) magnitude rather
    than identically zero."""
    j = np.arange(JPJ)
    lat_pattern = 0.5 * np.sin(2 * np.pi * j / JPJ)[None, :]
    ssh = lat_pattern * np.ones((JPI, JPJ)) + rng.normal(0.0, 0.02, size=(JPI, JPJ))
    return ssh.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True, help="dir with namelist_cfg/ref + harness_dom_qco_r3c.exe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_policy(PrecisionPolicy.fp64())

    rundir = Path(args.rundir)
    exe = rundir / "harness_dom_qco_r3c.exe"
    if not exe.exists():
        print(f"FATAL: {exe} not found -- build it first.")
        return 1

    rng = np.random.default_rng(args.seed)
    ssh = build_synthetic_ssh(rng)

    print("dtype check: ssh", ssh.dtype)
    require_fp64(ssh, context="dom_qco_r3c probe input")

    binary_io.write_array(str(rundir / "r3c_in_ssh.bin"), ssh)

    print(f"Running {exe} in {rundir} ...")
    result = subprocess.run([str(exe)], cwd=str(rundir), capture_output=True, text=True, timeout=120)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("FORTRAN DRIVER FAILED:")
        print(result.stderr[-4000:])
        return 1

    r3t_nemo = binary_io.read_array(str(rundir / "r3c_out_r3t.bin"), (JPI, JPJ))
    r3u_nemo = binary_io.read_array(str(rundir / "r3c_out_r3u.bin"), (JPI, JPJ))
    r3v_nemo = binary_io.read_array(str(rundir / "r3c_out_r3v.bin"), (JPI, JPJ))
    print(f"dtypes: r3t_nemo={r3t_nemo.dtype}  r3u_nemo={r3u_nemo.dtype}  "
          f"r3v_nemo={r3v_nemo.dtype}  ssh={ssh.dtype}")

    import netCDF4 as nc

    # mesh_mask.nc has no "ht_0" variable directly -- NEMO's own ht_0 is
    # 1/r1_ht_0 = Sum_k(e3t_0*tmask) (domqco.F90's r3t denominator; same
    # convention already used elsewhere in this campaign, see
    # heat_discriminator.py::_nemo_h_k's h_col = e3t_0.sum(axis=-1)).
    with nc.Dataset(MESH_PATH) as ds:
        e3t_0_i = np.asarray(ds.variables["e3t_0"][0], dtype=np.float64)   # (k,j,i)
        tmask3d_i = np.asarray(ds.variables["tmask"][0], dtype=np.float64)  # (k,j,i)
        umask3d_i = np.asarray(ds.variables["umask"][0], dtype=np.float64)  # (k,j,i)
        vmask3d_i = np.asarray(ds.variables["vmask"][0], dtype=np.float64)  # (k,j,i)
        e3u_0_i = np.asarray(ds.variables["e3u_0"][0], dtype=np.float64)   # (k,j,i)
        e3v_0_i = np.asarray(ds.variables["e3v_0"][0], dtype=np.float64)   # (k,j,i)
        e1t_i = np.asarray(ds.variables["e1t"][0], dtype=np.float64)       # (j,i)
        e2t_i = np.asarray(ds.variables["e2t"][0], dtype=np.float64)
        e1u_i = np.asarray(ds.variables["e1u"][0], dtype=np.float64)
        e2u_i = np.asarray(ds.variables["e2u"][0], dtype=np.float64)
        e1v_i = np.asarray(ds.variables["e1v"][0], dtype=np.float64)
        e2v_i = np.asarray(ds.variables["e2v"][0], dtype=np.float64)
    ht_0_ji = (e3t_0_i * tmask3d_i).sum(axis=0)          # (j,i)
    hu_0_ji = (e3u_0_i * umask3d_i).sum(axis=0)          # (j,i), domqco.F90 r1_hu_0 denominator
    hv_0_ji = (e3v_0_i * vmask3d_i).sum(axis=0)          # (j,i), domqco.F90 r1_hv_0 denominator
    tmask_ji = tmask3d_i[0]                              # (j,i) surface level
    ht_0 = np.zeros((JPI, JPJ), dtype=np.float64)
    hu_0 = np.zeros((JPI, JPJ), dtype=np.float64)
    hv_0 = np.zeros((JPI, JPJ), dtype=np.float64)
    tmask = np.zeros((JPI, JPJ), dtype=np.float64)
    e1e2t = np.zeros((JPI, JPJ), dtype=np.float64)
    e1e2u = np.zeros((JPI, JPJ), dtype=np.float64)
    e1e2v = np.zeros((JPI, JPJ), dtype=np.float64)
    ht_0[HLS:JPI - HLS, HLS:JPJ - HLS] = ht_0_ji.T
    hu_0[HLS:JPI - HLS, HLS:JPJ - HLS] = hu_0_ji.T
    hv_0[HLS:JPI - HLS, HLS:JPJ - HLS] = hv_0_ji.T
    tmask[HLS:JPI - HLS, HLS:JPJ - HLS] = tmask_ji.T
    e1e2t[HLS:JPI - HLS, HLS:JPJ - HLS] = (e1t_i * e2t_i).T
    e1e2u[HLS:JPI - HLS, HLS:JPJ - HLS] = (e1u_i * e2u_i).T
    e1e2v[HLS:JPI - HLS, HLS:JPJ - HLS] = (e1v_i * e2v_i).T

    class _ZCoordStub:
        linear_free_surface = False   # DINO: key_linssh undefined -> QCO active

    r3t_lego = np.asarray(nemo_r3t_stretch(_ZCoordStub(), ssh, ht_0)) - 1.0   # undo the (1+r3t) shift

    sl = (slice(HLS, JPI - HLS), slice(HLS, JPJ - HLS))
    wet = tmask[sl] > 0.5

    a = r3t_lego[sl][wet]
    b = r3t_nemo[sl][wet]
    d = a - b
    print(f"n wet-interior T-cells compared: {wet.sum()} of {wet.size}")
    print(f"r3t: max|diff|={np.max(np.abs(d)):.4e}  rms|diff|={np.sqrt(np.mean(d**2)):.4e}")
    # Self-check 1: signature that catches an axis-order slip -- an asymmetric
    # synthetic ssh pattern (sin(2*pi*j/jpj), constant in i) means a transposed
    # (i,j)<->(j,i) comparison would show near-zero corr, not near-1.0.
    corr = np.corrcoef(a, b)[0, 1]
    ratio = np.sqrt(np.mean(a**2)) / np.sqrt(np.mean(b**2)) if np.sqrt(np.mean(b**2)) > 0 else float("nan")
    print(f"r3t corr={corr:.9f} ratio={ratio:.9f}")
    # Self-check 2: NEMO's own zero-ssh cells (dry columns, ht_0<=0 -> r3t=0
    # by construction in nemo_r3t_stretch) must have r3t_nemo == 0 too, else
    # the dry-column convention disagrees between the two sides.
    dry = ht_0[sl][wet] <= 0.0
    if dry.any():
        print(f"dry-column check: max|r3t_nemo| on {dry.sum()} dry cells = {np.max(np.abs(b[dry])):.3e}")

    bar_ok = corr >= 1.0 - 1e-9 and abs(ratio - 1.0) <= 1e-6
    print("dom_qco_r3c r3t:", "AT BAR" if bar_ok else "NOT at bar (see corr/ratio above)")

    # --- r3u/r3v: transcription of domqco.F90:212-215 (probe-side, not
    # legoESM production -- see docstring). j (north/south) has NO wrap
    # (ln_Jperio=.FALSE.), so the j+1 face-average at jj=JPJ-1 is a genuine
    # edge (masked out by wet_v below via the v-mask's own last-row zero).
    #
    # SELF-CHECK CAUGHT (do not "fix" the numbers away without recording
    # this): a naive ``np.roll(et_ssh, -1, axis=0)`` over the FULL padded
    # (JPI,JPJ) array reads the harness's own non-periodic halo storage at
    # the i-seam (e1e2t/ssh are ZERO in the physical halo columns i=54,55 --
    # confirmed: ``harness_dom_qco_r3c.F90`` calls ``dom_qco_r3c`` DIRECTLY
    # with no ``lbc_lnk`` wrap) -- that gave r3u corr=0.999323/|x|ratio=
    # 0.997634, with ALL top-|diff| cells isolated to i=53 (confirmed via
    # umask column-sum: interior columns have 197 wet u-points, i=53 has
    # only 35 -- the seam column). NEMO's OWN dom_qco_r3c does not
    # periodic-wrap internally either (no lbc_lnk in the harness) -- DINO's
    # real ln_Iperio wrap is applied by the CALLER's lbc_lnk AFTER
    # dom_qco_r3c returns (domqco.F90:180-181), which this bare unit-call
    # harness does not exercise. So the i+1 lookup must wrap over the
    # INTERIOR width only (mirroring what a subsequent lbc_lnk WOULD do),
    # not read the harness's own zero-filled halo columns.
    lo, hi = HLS, JPI - HLS  # interior i-range [2, 54)
    et_ssh = e1e2t * ssh
    et_ssh_ip1 = et_ssh.copy()
    et_ssh_ip1[lo:hi - 1, :] = et_ssh[lo + 1:hi, :]        # interior i+1
    et_ssh_ip1[hi - 1, :] = et_ssh[lo, :]                  # i-periodic wrap at the seam
    r3u_lego_num = 0.5 * (et_ssh + et_ssh_ip1)
    r3v_lego_num = 0.5 * (et_ssh + np.roll(et_ssh, -1, axis=1))     # j+1 shift (axis 1 = j, JPJ)

    # SEAM COLUMN i=hi-1 (last interior column, i=53): even after the
    # periodic-wrap fix above, this column carries a residual up to ~1e-5
    # (vs ~1e-11 roundoff everywhere else) -- this is the SAME "single
    # redundant periodic-wrap u-face" the bridge itself documents and tells
    # callers to EXCLUDE from full-domain face comparisons
    # (nemo_state_bridge.py::bridge_nemo_to_legoesm_topo docstring, "KNOWN
    # LIMITATIONS (a)"): the u-face at the wrap seam is stored under
    # legoESM's closed-basin convention, not the periodic roll, so a
    # cell-by-cell comparison AT that exact column is not apples-to-apples
    # by construction, independent of this probe's own formula. Excluded
    # below by name (seam_col), not silently -- both AT-BAR and NOT-AT-BAR
    # verdicts are reported for interior-only AND full (incl. seam) so nothing
    # is hidden.
    seam_col = hi - 1  # i=53
    r1_hu_0 = np.where(hu_0 > 0.0, 1.0 / np.maximum(hu_0, 1e-30), 0.0)
    r1_hv_0 = np.where(hv_0 > 0.0, 1.0 / np.maximum(hv_0, 1e-30), 0.0)
    r1_e1e2u = np.where(e1e2u > 0.0, 1.0 / np.maximum(e1e2u, 1e-30), 0.0)
    r1_e1e2v = np.where(e1e2v > 0.0, 1.0 / np.maximum(e1e2v, 1e-30), 0.0)
    r3u_lego = r3u_lego_num * r1_hu_0 * r1_e1e2u
    r3v_lego = r3v_lego_num * r1_hv_0 * r1_e1e2v

    umask3d_ji_full = umask3d_i[0]  # (j,i) surface u-mask
    vmask3d_ji_full = vmask3d_i[0]
    umask = np.zeros((JPI, JPJ), dtype=np.float64)
    vmask = np.zeros((JPI, JPJ), dtype=np.float64)
    umask[HLS:JPI - HLS, HLS:JPJ - HLS] = umask3d_ji_full.T
    vmask[HLS:JPI - HLS, HLS:JPJ - HLS] = vmask3d_ji_full.T
    wet_u = umask[sl] > 0.5
    wet_v = vmask[sl] > 0.5
    seam_mask_sl = np.zeros_like(wet_u)
    seam_mask_sl[seam_col - HLS, :] = True  # seam_col is a full-array (JPI) index; sl strips HLS

    def _corr_ratio(a, b):
        c = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        r = float(np.sum(np.abs(a)) / np.sum(np.abs(b))) if np.sum(np.abs(b)) > 0 else float("nan")
        return c, r

    au_full = r3u_lego[sl][wet_u]
    bu_full = r3u_nemo[sl][wet_u]
    av = r3v_lego[sl][wet_v]
    bv = r3v_nemo[sl][wet_v]
    wet_u_interior = wet_u & ~seam_mask_sl
    au_int = r3u_lego[sl][wet_u_interior]
    bu_int = r3u_nemo[sl][wet_u_interior]

    print(f"\nn wet-interior u-cells compared: {wet_u.sum()} of {wet_u.size} "
          f"({int(seam_mask_sl.sum())} at the periodic-wrap seam column, excluded below)")
    print(f"n wet-interior v-cells compared: {wet_v.sum()} of {wet_v.size}")
    corr_u_full, ratio_u_full = _corr_ratio(au_full, bu_full)
    corr_u_int, ratio_u_int = _corr_ratio(au_int, bu_int)
    corr_v, ratio_v = _corr_ratio(av, bv)
    print(f"r3u (incl. seam column) corr={corr_u_full:.9f} |x|ratio={ratio_u_full:.9f}")
    print(f"r3u (seam EXCLUDED, n={int(wet_u_interior.sum())}) "
          f"corr={corr_u_int:.9f} |x|ratio={ratio_u_int:.9f}")
    print(f"r3v corr={corr_v:.9f} |x|ratio={ratio_v:.9f}")
    bar_ok_u_full = corr_u_full >= 1.0 - 1e-9 and abs(ratio_u_full - 1.0) <= 1e-6
    bar_ok_u_int = corr_u_int >= 1.0 - 1e-9 and abs(ratio_u_int - 1.0) <= 1e-6
    bar_ok_v = corr_v >= 1.0 - 1e-9 and abs(ratio_v - 1.0) <= 1e-6
    print("dom_qco_r3c r3u (incl. seam):", "AT BAR" if bar_ok_u_full else "NOT at bar")
    print("dom_qco_r3c r3u (seam excluded):", "AT BAR" if bar_ok_u_int else "NOT at bar (see corr/ratio above)")
    print("dom_qco_r3c r3v:", "AT BAR" if bar_ok_v else "NOT at bar (see corr/ratio above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
