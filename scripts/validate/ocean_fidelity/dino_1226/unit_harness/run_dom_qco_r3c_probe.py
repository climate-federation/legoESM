#!/usr/bin/env python
"""#1226 NEMO unit-call harness -- ROUTINE 4: dom_qco_r3c (ssh/h_0 ratio at
t-, u-, v-, f-points, ``domqco.F90:140-186``).

Gate row context: ``fidelity_bar_gate.py`` rows "dom_qco_r3c r3t" and
"dom_qco_r3c r3u/r3v" are both PHANTOM-PROVENANCE (their cited script
``probe_1226_r2_item4_domqco.py`` does not exist in the repo tree, confirmed
by ``check_provenance_scripts_exist()``). This probe measures ONLY the r3t
half via the compiled Fortran driver (``harness_dom_qco_r3c.exe``) against
legoESM's existing :func:`legoesm.ocean.eos.nemo_r3t_stretch` -- the u/v/f
outputs are still recorded by this probe (the Fortran driver produces all
four) but are NOT compared here: pre-impl search
(``grep -rn "r3u\\|r3v" packages/ocean/legoesm/ocean/``) found NO legoESM
production function computing the face-averaged ratio at u-/v-/f-points
(only the T-point stretch factor exists). Writing a new one would be new
PHYSICS CODE under ``packages/`` at probe-authoring time, which is
explicitly out of scope for this task (READ-ONLY on ``packages/``/``src/``)
-- so r3u/r3v stay UNMEASURED here rather than compared against ad hoc probe-
side algebra standing in for a production implementation.

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
    r3u_nemo = binary_io.read_array(str(rundir / "r3c_out_r3u.bin"), (JPI, JPJ))  # produced, not compared (see docstring)
    r3v_nemo = binary_io.read_array(str(rundir / "r3c_out_r3v.bin"), (JPI, JPJ))  # produced, not compared (see docstring)
    del r3u_nemo, r3v_nemo  # explicit: these outputs exist but this probe does not judge them

    import netCDF4 as nc

    # mesh_mask.nc has no "ht_0" variable directly -- NEMO's own ht_0 is
    # 1/r1_ht_0 = Sum_k(e3t_0*tmask) (domqco.F90's r3t denominator; same
    # convention already used elsewhere in this campaign, see
    # heat_discriminator.py::_nemo_h_k's h_col = e3t_0.sum(axis=-1)).
    with nc.Dataset(MESH_PATH) as ds:
        e3t_0_i = np.asarray(ds.variables["e3t_0"][0], dtype=np.float64)   # (k,j,i)
        tmask3d_i = np.asarray(ds.variables["tmask"][0], dtype=np.float64)  # (k,j,i)
    ht_0_ji = (e3t_0_i * tmask3d_i).sum(axis=0)          # (j,i)
    tmask_ji = tmask3d_i[0]                              # (j,i) surface level
    ht_0 = np.zeros((JPI, JPJ), dtype=np.float64)
    tmask = np.zeros((JPI, JPJ), dtype=np.float64)
    ht_0[HLS:JPI - HLS, HLS:JPJ - HLS] = ht_0_ji.T
    tmask[HLS:JPI - HLS, HLS:JPJ - HLS] = tmask_ji.T

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
    print("dom_qco_r3c r3u/r3v: UNMEASURED by this probe -- no legoESM production "
          "function computes the u-/v-face ratio (pre-impl search found none); "
          "the Fortran driver's r3u/r3v outputs are written to disk but not judged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
