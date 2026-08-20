#!/usr/bin/env python
"""#1226 NEMO unit-call harness -- ROUTINE 2: dyn_zad (vertical momentum
advection trend, vector form, ``dynzad.F90:40-133``).

DINO's active scheme (quoted): ``cfgs/DINO/EXP00/namelist_cfg:321
ln_dynadv_vec = .true.`` -> ``dynadv.F90``'s np_VEC_c2 branch calls
``dyn_keg`` then ``dyn_zad`` -- ``dyn_zad`` IS the live vertical-advection
routine for this config.

This calls the compiled Fortran driver (``harness_dyn_zad.exe``) with a
synthetic u/v/w velocity field spanning real dynamic range, and compares
against legoESM's ``nemo_advective_vertical_momentum_advection``
(``ocean/vertical.py:1149``).

Per the task brief this is compared against the existing
``fidelity_bar_gate.py`` "dyn_adv ZAD" row tuple (0.999200, 0.995116) as a
SANITY CHECK, not reproduced -- the two probes are INDEPENDENT (this one
uses synthetic inputs + a direct routine call; the gate row's own probe,
``probe_1226_keg_zad_split.py``, is cited in the gate's provenance table but
does NOT exist in the repo tree -- confirmed by `find`, see report). A
disagreement between the two is NOT resolved here (oracle-fidelity skill
Rule 1e: reconcile before recording, never pick a winner) -- this script
only reports its OWN independently-measured number.
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
from legoesm.ocean.vertical import nemo_advective_vertical_momentum_advection  # noqa: E402

JPI, JPJ, JPK = 56, 203, 36
HLS = 2
MESH_PATH = (
    REPO_ROOT.parent / "oracle-builds" / "nemo5" / "nemo_5.0.2" / "cfgs" / "DINO" / "RUN_GDB" / "mesh_mask.nc"
)


def load_mesh():
    import netCDF4 as nc

    with nc.Dataset(MESH_PATH) as ds:
        e1e2t = np.asarray(ds.variables["e1t"][0] * ds.variables["e2t"][0], dtype=np.float64)  # (j,i)
        e1e2u = np.asarray(ds.variables["e1u"][0] * ds.variables["e2u"][0], dtype=np.float64)
        e3u_0 = np.asarray(ds.variables["e3u_0"][0], dtype=np.float64)   # (k,j,i)
        e3v_0 = np.asarray(ds.variables["e3v_0"][0], dtype=np.float64)
        umask = np.asarray(ds.variables["umask"][0], dtype=np.float64)  # (k,j,i)
        vmask = np.asarray(ds.variables["vmask"][0], dtype=np.float64)
        tmask = np.asarray(ds.variables["tmask"][0], dtype=np.float64)
    return e1e2t, e1e2u, e3u_0, e3v_0, umask, vmask, tmask


def embed_interior(interior_2d_or_3d, full_shape):
    """Embed a mesh_mask.nc interior-only field into the harness's full
    (halo-included) (jpi,jpj[,jpk]) array, transposing (k,j,i)->(i,j,k) or
    (j,i)->(i,j) as needed. Halo cells left at 0 -- callers must restrict
    comparisons to the interior slice, exactly as the eos_rab probe does.
    """
    out = np.zeros(full_shape, dtype=np.float64)
    if interior_2d_or_3d.ndim == 2:
        arr_ij = interior_2d_or_3d.transpose(1, 0)   # (j,i)->(i,j)
        out[HLS:JPI - HLS, HLS:JPJ - HLS] = arr_ij
    else:
        arr_ijk = interior_2d_or_3d.transpose(2, 1, 0)  # (k,j,i)->(i,j,k)
        out[HLS:JPI - HLS, HLS:JPJ - HLS, :] = arr_ijk
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_policy(PrecisionPolicy.fp64())
    rundir = Path(args.rundir)
    exe = rundir / "harness_dyn_zad.exe"
    if not exe.exists():
        print(f"FATAL: {exe} not found -- build it first.")
        return 1

    rng = np.random.default_rng(args.seed)
    # synthetic velocity/vertical-velocity with real dynamic range: an ACC-like
    # jet profile in u decaying with depth, small-scale noise on top, and a
    # w field with a sign-changing vertical structure (not a quiescent zero
    # field) so dyn_zad's w*du/dz term is actually exercised.
    k = np.arange(JPK)
    depth_decay = np.exp(-k / 10.0)
    u_jet = 0.3 * depth_decay[None, None, :] * np.sin(np.linspace(0, np.pi, JPJ))[None, :, None]
    uu = u_jet + rng.normal(0.0, 0.02, size=(JPI, JPJ, JPK))
    vv = 0.05 * depth_decay[None, None, :] + rng.normal(0.0, 0.02, size=(JPI, JPJ, JPK))
    ww = 1e-4 * np.sin(2 * np.pi * k / JPK)[None, None, :] + rng.normal(0.0, 2e-5, size=(JPI, JPJ, JPK))

    uu3 = np.stack([uu, uu, uu], axis=-1)  # Nbb,Nnn,Naa identical -- dyn_zad reads only Kmm=Nnn
    vv3 = np.stack([vv, vv, vv], axis=-1)

    require_fp64(uu3, context="dyn_zad probe input")

    binary_io.write_array(str(rundir / "zad_in_uu.bin"), uu3)
    binary_io.write_array(str(rundir / "zad_in_vv.bin"), vv3)
    binary_io.write_array(str(rundir / "zad_in_ww.bin"), ww)

    print(f"Running {exe} in {rundir} ...")
    result = subprocess.run([str(exe)], cwd=str(rundir), capture_output=True, text=True, timeout=120)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("FORTRAN DRIVER FAILED:")
        print(result.stderr[-4000:])
        return 1

    du_nemo = binary_io.read_array(str(rundir / "zad_out_du.bin"), (JPI, JPJ, JPK))
    dv_nemo = binary_io.read_array(str(rundir / "zad_out_dv.bin"), (JPI, JPJ, JPK))

    e1e2t_i, e1e2u_i, e3u_0_i, e3v_0_i, umask_i, vmask_i, tmask_i = load_mesh()
    e1e2t = embed_interior(e1e2t_i, (JPI, JPJ))
    e1e2u = embed_interior(e1e2u_i, (JPI, JPJ))
    e3u_0 = embed_interior(e3u_0_i, (JPI, JPJ, JPK))
    umask = embed_interior(umask_i, (JPI, JPJ, JPK))

    # w_area_half at u-face, interface jk+1 (0-indexed k=jk): NEMO's
    # zzWfu = e1e2t(i,j)*ww(i,j,jk+1) + e1e2t(i+1,j)*ww(i+1,j,jk+1)
    # (dynzad.F90:93-97) -- i.e. i+1 in Fortran 1-index is the SAME-j,
    # next-i neighbour; in this script's 0-indexed (i,j,k) convention with
    # i increasing eastward, that's arr[i+1,j,k]. Interfaces run 1..jpk-1
    # (Fortran jk+1 for jk=1..jpk-2) i.e. 0-indexed interior interfaces
    # k=1..jpk-2 (interface index 0 = surface = 0, matching the padded
    # convention in nemo_advective_vertical_momentum_advection).
    ew = e1e2t[..., None] * ww   # (jpi,jpj,jpk), value at LEVEL k (T-point)
    # interface k (1..jpk-1) sits between level k-1 and level k; NEMO's ww is
    # already a W-POINT (interface) array in its own (i,j,jk) indexing, so
    # ew[...,k] IS the interface-k quantity already -- confirmed by dynzad.F90
    # reading ww(i,j,jk+1) directly (no averaging in k).
    #
    # IMPORTANT (bug found + fixed during this probe's development, see
    # report): nemo_advective_vertical_momentum_advection's OWN formula is
    # G[k] = 2*w_area_half[k]*(u[k-1]-u[k]) -- the function reconstructs
    # NEMO's zzWfu=zWfi+zWf SUM internally via that explicit factor of 2.
    # So w_area_half must be the grid.py convention's interp_cell_to_uface
    # AVERAGE 0.5*(ew[i]+ew[i+1]), NOT the raw sum -- feeding the sum here
    # double-counts and was caught by an exact 2x ratio on first run.
    w_area_half = np.zeros((JPI, JPJ, JPK + 1), dtype=np.float64)
    ew_uface_avg = np.zeros_like(ew)
    ew_uface_avg[:-1, :, :] = 0.5 * (ew[:-1, :, :] + ew[1:, :, :])
    w_area_half[:, :, 1:JPK] = ew_uface_avg[:, :, 1:JPK]
    # surface (k=0) and bottom (k=JPK) interfaces are architecturally zero,
    # matching NEMO's zWdzU init / pww(jpk)=0 -- already zero from np.zeros.

    h_u = np.maximum(e3u_0, 1e-10)
    face_area = np.maximum(e1e2u, 1e-10)

    du_lego = np.asarray(
        nemo_advective_vertical_momentum_advection(uu, w_area_half, h_u, face_area[..., None], face_active=umask)
    )

    hls, jpkm1 = HLS, JPK - 1
    sl = (slice(hls, JPI - hls), slice(hls, JPJ - hls), slice(0, jpkm1))
    wet_u = umask[sl] > 0.5

    d = du_lego[sl][wet_u] - du_nemo[sl][wet_u]
    print(f"n wet-interior u-cells compared: {wet_u.sum()} of {wet_u.size}")
    print(f"du: max|diff|={np.max(np.abs(d)):.4e}  rms|diff|={np.sqrt(np.mean(d**2)):.4e}")
    a, b = du_lego[sl][wet_u], du_nemo[sl][wet_u]
    corr = np.corrcoef(a, b)[0, 1]
    ratio = np.sqrt(np.mean(a**2)) / np.sqrt(np.mean(b**2))
    print(f"du corr={corr:.6f} ratio={ratio:.6f}")
    print("For reference (NOT reproduced, independent probe, see docstring): "
          "fidelity_bar_gate.py 'dyn_adv ZAD' row = (0.999200, 0.995116)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
