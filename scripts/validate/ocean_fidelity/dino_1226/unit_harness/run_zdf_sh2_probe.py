#!/usr/bin/env python
"""#1226 NEMO unit-call harness -- ROUTINE 3: zdf_sh2 (shear production
term of TKE, ``zdfsh2.F90:37-102``).

Context (ESCALATION 1, ``.claude/ralph_dino_fidelity_to_bar_task.md``): the
existing DUMP-BASED probe (``sh2_walk.py``) measures this row at NEMO's own
QUIESCENT restart state (kt=57601), where 95.2% of wet cells have
``|sh2| < 1e-9`` (below noise floor) -- so only a "restricted-to-signal"
subset is meaningful, giving ratio 0.9036 (NOT ~1.0) even though the
transcription (``vertical_shear_face_native``) is an exact port.

This probe drives ``zdf_sh2`` DIRECTLY through the compiled Fortran driver
with a SYNTHETIC velocity/avm field spanning REAL dynamic range (a sheared
jet decaying with depth, now != before by a real O(shear) amount, avm at a
realistic O(1e-3) KPP scale) -- testing whether the "no signal" problem was
a property of the ROUTINE (transcription bug hidden by cancellation) or of
the STATE the dump-based method happened to probe.

Fair-comparison note: ``vertical_shear_face_native`` deliberately does NOT
face-average ``avm`` (its own docstring scope limit -- it returns a bare
shear-production-EQUIVALENT that the caller multiplies by a single
per-interface K_M). NEMO's raw ``p_sh2`` output DOES have avm baked in via
face-averaging (``avm(i+1,j,jk)+avm(i,j,jk)``). To compare like-for-like
without re-deriving avm face-averaging as a NEW candidate, this probe feeds
a SPATIALLY-UNIFORM avm (so face-averaging is a no-op: avm(i+1)+avm(i) =
2*avm_uniform), then multiplies legoESM's ``shear_sq_equivalent`` output by
that same avm_uniform before comparing -- exactly reconstructing the
documented ``P_s = K_M * shear_sq_equivalent`` contract instead of guessing.
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
from legoesm.ocean.physics.vertical_mixing._shared import vertical_shear_face_native  # noqa: E402

JPI, JPJ, JPK = 56, 203, 36
HLS = 2
AVM_UNIFORM = 1.0e-3   # m^2/s, realistic KPP-scale eddy viscosity
MESH_PATH = (
    REPO_ROOT.parent / "oracle-builds" / "nemo5" / "nemo_5.0.2" / "cfgs" / "DINO" / "RUN_GDB" / "mesh_mask.nc"
)


def load_mesh():
    import netCDF4 as nc

    with nc.Dataset(MESH_PATH) as ds:
        umask = np.asarray(ds.variables["umask"][0], dtype=np.float64)  # (k,j,i)
        vmask = np.asarray(ds.variables["vmask"][0], dtype=np.float64)
        e3t_1d = np.asarray(ds.variables["e3t_1d"][0, :], dtype=np.float64)  # (k,) static ref spacing
    return umask, vmask, e3t_1d


def embed_interior_3d(interior_kji, full_shape):
    out = np.zeros(full_shape, dtype=np.float64)
    arr_ijk = interior_kji.transpose(2, 1, 0)
    out[HLS:JPI - HLS, HLS:JPJ - HLS, :] = arr_ijk
    return out


def build_synthetic_shear(rng: np.random.Generator):
    """A sheared jet decaying with depth: |u| ~ O(0.1-0.3 m/s) at surface,
    now != before by an O(shear) amount, spanning real dynamic range --
    the opposite of the quiescent restart's near-zero shear."""
    k = np.arange(JPK)
    u_now = 0.3 * np.exp(-k / 8.0)[None, None, :] * np.ones((JPI, JPJ, JPK)) + rng.normal(0, 0.01, (JPI, JPJ, JPK))
    u_before = u_now + rng.normal(0.0, 0.02, (JPI, JPJ, JPK))
    v_now = 0.1 * np.exp(-k / 8.0)[None, None, :] * np.ones((JPI, JPJ, JPK)) + rng.normal(0, 0.01, (JPI, JPJ, JPK))
    v_before = v_now + rng.normal(0.0, 0.02, (JPI, JPJ, JPK))
    return u_now, u_before, v_now, v_before


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    set_policy(PrecisionPolicy.fp64())
    rundir = Path(args.rundir)
    exe = rundir / "harness_zdf_sh2.exe"
    if not exe.exists():
        print(f"FATAL: {exe} not found -- build it first.")
        return 1

    rng = np.random.default_rng(args.seed)
    u_now, u_before, v_now, v_before = build_synthetic_shear(rng)
    avm = np.full((JPI, JPJ, JPK), AVM_UNIFORM, dtype=np.float64)

    uu3 = np.stack([u_before, u_now, u_now], axis=-1)  # Nbb=1,Nnn=2,Naa=3(unused)
    vv3 = np.stack([v_before, v_now, v_now], axis=-1)

    require_fp64(uu3, avm, context="zdf_sh2 dynamic-range probe input")

    binary_io.write_array(str(rundir / "sh2_in_uu.bin"), uu3)
    binary_io.write_array(str(rundir / "sh2_in_vv.bin"), vv3)
    binary_io.write_array(str(rundir / "sh2_in_avm.bin"), avm)

    print(f"Running {exe} in {rundir} ...")
    result = subprocess.run([str(exe)], cwd=str(rundir), capture_output=True, text=True, timeout=120)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("FORTRAN DRIVER FAILED:")
        print(result.stderr[-4000:])
        return 1

    p_sh2_nemo = binary_io.read_array(str(rundir / "sh2_out_p_sh2.bin"), (JPI, JPJ, JPK))

    print(f"NEMO p_sh2 dynamic range: min={p_sh2_nemo.min():.4e} max={p_sh2_nemo.max():.4e}")
    frac_below_floor = np.mean(np.abs(p_sh2_nemo) < 1e-9)
    print(f"fraction |p_sh2| < 1e-9 (the quiescent-state noise floor): {frac_below_floor:.4f}"
          f"  (ESCALATION 1's quiescent-restart probe measured 0.952 here)")

    umask_i, vmask_i, e3t_1d = load_mesh()
    umask = embed_interior_3d(umask_i, (JPI, JPJ, JPK))
    vmask = embed_interior_3d(vmask_i, (JPI, JPJ, JPK))

    # dz_half: T-point distance between cell centres, static reference
    # (documented scope limit of vertical_shear_face_native -- Candidate B
    # measured this substitution as negligible for DINO, corr 1.000/ratio
    # 0.9999). gdept_1d differences give the T-to-T spacing directly.
    with __import__("netCDF4").Dataset(MESH_PATH) as ds:
        gdept_1d = np.asarray(ds.variables["gdept_1d"][0, :], dtype=np.float64)
    dz_half_1d = gdept_1d[1:] - gdept_1d[:-1]   # (jpk-1,)
    dz_half = np.broadcast_to(dz_half_1d[None, None, :], (JPI, JPJ, JPK - 1))

    # legoESM's grid convention is (n_lat, n_lon, nlev) = NEMO's (j, i, k) --
    # the OPPOSITE axis order from this script's (i, j, k) throughout
    # (confirmed: operators_latlon_cgrid.py docstrings "f : (n_lat, n_lon,
    # ...)"). Transpose (i,j,k)->(j,i,k) before calling. u_face/v_face also
    # need one EXTRA column/row (n_lon+1 / n_lat+1) for the periodic-wrap
    # closure face -- DINO is i-periodic (ln_Iperio=.true., usrdef_nam.F90:
    # 157) and j-closed (ldJperio=.false.); append the wrap column in i,
    # and repeat the last row in j (a closed boundary has no real "extra"
    # v-face beyond jpj -- legoESM's own docstring notes the padded value
    # there is masked out by wvmask and never selected).
    def to_lego_axes(a_ijk):
        return np.transpose(a_ijk, (1, 0, 2))   # (i,j,k) -> (j,i,k)

    u_now_l = to_lego_axes(u_now)
    v_now_l = to_lego_axes(v_now)
    u_before_l = to_lego_axes(u_before)
    v_before_l = to_lego_axes(v_before)
    umask_l = to_lego_axes(umask)
    vmask_l = to_lego_axes(vmask)
    dz_half_l = to_lego_axes(dz_half)

    def add_i_wrap(a):   # (j,i,k) -> (j,i+1,k), periodic closure in i
        return np.concatenate([a, a[:, :1, :]], axis=1)

    def add_j_wrap(a):   # (j,i,k) -> (j+1,i,k), closed boundary repeat-last-row
        return np.concatenate([a, a[-1:, :, :]], axis=0)

    u_face_now = add_i_wrap(u_now_l)
    u_face_before = add_i_wrap(u_before_l)
    v_face_now = add_j_wrap(v_now_l)
    v_face_before = add_j_wrap(v_before_l)
    u_mask_face = add_i_wrap(umask_l)
    v_mask_face = add_j_wrap(vmask_l)

    shear_eq_l = np.asarray(
        vertical_shear_face_native(
            u_face_now, v_face_now, u_face_before, v_face_before, dz_half_l, u_mask_face, v_mask_face
        )
    )
    shear_eq = np.transpose(shear_eq_l, (1, 0, 2))   # back to (i,j,k)
    # reconstruct P_s = K_M * shear_sq_equivalent with the SAME uniform avm
    # NEMO's face-averaging collapsed to. NEMO's own zsh2u/zsh2v carry
    # (avm(i+1,j,jk)+avm(i,j,jk)) = 2*AVM_UNIFORM here (uniform avm), so the
    # reconstruction needs 2*AVM_UNIFORM, not AVM_UNIFORM alone -- an
    # earlier version of this script used AVM_UNIFORM and got an exact 2x
    # ratio (0.499945), the SAME class of bug as dyn_zad's earlier 2x
    # w_area_half mistake -- caught the same way, by the exact-power-of-2
    # signature.
    p_sh2_lego = (2.0 * AVM_UNIFORM) * shear_eq

    # p_sh2_lego has shape (...,nlev-1) at interfaces k=0..nlev-2 (between
    # T-levels k,k+1); NEMO's p_sh2 is (...,jpk) w-point, written for
    # jk=2..jpkm1 (1-indexed, i.e. 0-indexed interior w-levels 1..jpk-2),
    # with jk=1 and jk=jpk explicitly zeroed (zdfsh2.F90:96-99). Interface
    # k (0-indexed, between T k,k+1) corresponds to NEMO's w-level jk=k+2
    # (1-indexed) -- i.e. 0-indexed w-level k+1. Align on that mapping.
    p_sh2_lego_wlevel = np.zeros((JPI, JPJ, JPK), dtype=np.float64)
    p_sh2_lego_wlevel[:, :, 1:JPK - 1] = p_sh2_lego[:, :, :JPK - 2]

    hls = HLS
    sl = (slice(hls, JPI - hls), slice(hls, JPJ - hls), slice(1, JPK - 1))
    wet = umask[sl] > 0.5   # w-level wet test via the T-mask proxy (umask here is a placeholder;
                              # true w-wetness would need wmask -- restrict to cells where BOTH
                              # bracketing T-cells are wet via u/v masks already folded into shear_eq)
    a = p_sh2_lego_wlevel[sl]
    b = p_sh2_nemo[sl]
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]

    d = a - b
    print(f"n cells compared (unrestricted, interior domain): {a.size}")
    print(f"max|diff|={np.max(np.abs(d)):.4e}  rms|diff|={np.sqrt(np.mean(d**2)):.4e}")
    corr = np.corrcoef(a, b)[0, 1]
    ratio = np.sqrt(np.mean(a**2)) / np.sqrt(np.mean(b**2))
    print(f"UNRESTRICTED: corr={corr:.6f} ratio={ratio:.6f}")

    signal_mask = np.abs(b) >= 1e-9
    if signal_mask.sum() > 1:
        ar, br = a[signal_mask], b[signal_mask]
        corr_r = np.corrcoef(ar, br)[0, 1]
        ratio_r = np.sqrt(np.mean(ar**2)) / np.sqrt(np.mean(br**2))
        print(f"RESTRICTED-TO-SIGNAL (|nemo|>=1e-9, n={signal_mask.sum()} of {signal_mask.size}): "
              f"corr={corr_r:.6f} ratio={ratio_r:.6f}")
        print("For reference (quiescent-restart method, ESCALATION 1): restricted-to-signal ratio 0.9036")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
