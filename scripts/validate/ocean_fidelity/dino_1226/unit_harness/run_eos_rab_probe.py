#!/usr/bin/env python
"""#1226 NEMO unit-call harness -- ROUTINE 1 (POSITIVE CONTROL): eos_rab.

Drives NEMO's ``eos_rab`` (``eosbn2.F90`` INTERFACE -> ``rab_3d``, the
np_seos branch DINO uses -- confirmed by ``namelist_cfg:233 ln_seos=.true.``
and by grepping every ``CALL eos_rab`` site in ``cfgs/DINO/MY_SRC`` +
``src/OCE``, all of which pass a 4-D ``ts(:,:,:,:,K)`` array, resolving to
``rab_3d`` not ``rab_2d``/``rab_0d``) via a COMPILED FORTRAN DRIVER
(``cfgs/DINO/UNIT_HARNESS/harness_eos_rab.exe``) that calls the real routine
DIRECTLY with a synthetic T/S field written by this script, rather than
parsing a dumped full-timestep state.

This is the harness's REQUIRED positive control (task brief): if this does
not reproduce ~machine roundoff, the harness has a bug and no claim about
``dyn_zad``/``zdf_sh2`` from the same harness can be trusted.

Usage:
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/validate/ocean_fidelity/dino_1226/unit_harness/run_eos_rab_probe.py

Requires the Fortran driver already built (see
``cfgs/DINO/UNIT_HARNESS/BUILD.md``-equivalent instructions in the task
report -- not duplicated here to avoid a second source of truth for the
build recipe) and a run directory containing DINO's own
``namelist_cfg``/``namelist_ref`` (unmodified) plus the compiled
``harness_eos_rab.exe``.
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
from legoesm.ocean.eos import NemoSEOSConfig, nemo_seos_alpha_beta  # noqa: E402

JPI, JPJ, JPK = 56, 203, 36   # DINO domain, confirmed at runtime by the driver itself


def build_synthetic_ts(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic T/S spanning realistic DINO dynamic range (not a quiescent
    restart): T in roughly [-2, 28] C decaying with depth-index, S in
    [33, 37] PSU, plus noise -- so alpha/beta are exercised over the S-EOS
    polynomial's full operating range, not just one water mass.
    """
    k = np.arange(JPK, dtype=np.float64)
    t_profile = 26.0 * np.exp(-k / 8.0) - 1.0        # warm surface, cold abyss
    s_profile = 35.0 + 0.5 * np.sin(k / 5.0)
    t = t_profile[None, None, :] + rng.normal(0.0, 0.5, size=(JPI, JPJ, JPK))
    s = s_profile[None, None, :] + rng.normal(0.0, 0.2, size=(JPI, JPJ, JPK))
    return t, s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True, help="dir with namelist_cfg/ref + harness_eos_rab.exe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_policy(PrecisionPolicy.fp64())

    rundir = Path(args.rundir)
    exe = rundir / "harness_eos_rab.exe"
    if not exe.exists():
        print(f"FATAL: {exe} not found -- build it first (see report).")
        return 1

    rng = np.random.default_rng(args.seed)
    t, s = build_synthetic_ts(rng)
    ts = np.stack([t, s], axis=-1)   # (jpi,jpj,jpk,2) jp_tem=0, jp_sal=1 (0-indexed here; Fortran jp_tem=1,jp_sal=2)

    print("dtype check: ts", ts.dtype)
    require_fp64(ts, context="eos_rab positive control input")

    ts_path = rundir / "eos_rab_in_ts.bin"
    kmm_path = rundir / "eos_rab_in_kmm.bin"
    ab_path = rundir / "eos_rab_out_ab.bin"
    neos_path = rundir / "eos_rab_out_neos.bin"

    binary_io.write_array(str(ts_path), ts)
    kmm = 2   # NEMO's Nnn index after nemo_init (Nbb=1,Nnn=2,Naa=3); eos_rab only uses
              # Kmm to select which QCO r3t band-name macro path some branches take --
              # the np_seos branch ignores Kmm's TIME LEVEL entirely (see rab_3d_t
              # np_seos branch, eosbn2.F90: it reads gdept(...,Kmm) but the harness
              # sits at rest, r3t==0 for every Kmm, so this choice is inert here).
    binary_io.write_scalar_int(str(kmm_path), kmm)

    print(f"Running {exe} in {rundir} ...")
    result = subprocess.run([str(exe)], cwd=str(rundir), capture_output=True, text=True, timeout=120)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print("FORTRAN DRIVER FAILED:")
        print(result.stderr[-4000:])
        return 1

    ab = binary_io.read_array(str(ab_path), (JPI, JPJ, JPK, 2))
    neos = binary_io.read_scalar_int(str(neos_path))
    print(f"NEMO neos = {neos} (expect 1 = np_seos; np_teos10=-1, np_eos80=0, np_seos=1)")
    if neos != 1:
        print("FATAL: DINO did not resolve to np_seos as expected -- STOP, do not compare.")
        return 1

    alpha_nemo = ab[..., 0]
    beta_nemo = ab[..., 1]

    # gdept the harness's tmask/tmask=1 rest state actually used: r3t=0 at
    # rest (no SSH perturbation from usrdef_istate before any step), so
    # NEMO's live gdept(:,:,:,Kmm) == the static gdept_0 ladder. We don't
    # assume this -- the driver's own eos_rab call already consumed
    # whatever gdept NEMO's module state holds; here we only need gdept_0
    # to feed the SAME depth into legoESM's independent formula. Read it
    # from NEMO's own mesh_mask.nc (already-dumped geometry, same DINO
    # domain, no assumption about SSH needed since it's the T-point
    # REFERENCE depth ladder, not the live one).
    import netCDF4 as nc

    mesh_path = REPO_ROOT.parent / "oracle-builds" / "nemo5" / "nemo_5.0.2" / "cfgs" / "DINO" / "RUN_GDB" / "mesh_mask.nc"
    with nc.Dataset(mesh_path) as ds:
        # gdept_0 is genuinely 3-D (bathymetry/partial-cell dependent) --
        # CONFIRMED by direct comparison: broadcasting the 1-D gdept_1d
        # ladder instead differs from the real gdept_0 by up to 105 m at
        # some (j,i,k) (deep levels near sloped topography). An EARLIER
        # version of this script used the 1-D broadcast and got alpha
        # corr=0.999917/ratio=1.002008 (NOT roundoff) -- RETRACTED here:
        # that was a bug in THIS COMPARISON SCRIPT (wrong depth field fed
        # to the independent legoESM formula), not a harness or eos_rab
        # defect. Recorded per the project's retraction discipline.
        gdept_0_khji = np.asarray(ds.variables["gdept_0"][0], dtype=np.float64)  # (k,j,i) interior-only
        # mesh_mask.nc stores the INTERIOR-only domain (jpk, jpj-2*hls, jpi-2*hls);
        # transpose to (i,j,k) to match this script's (jpi,jpj,jpk) convention.
        gdept_0_interior = gdept_0_khji.transpose(2, 1, 0)
        tmask_interior = np.asarray(ds.variables["tmask"][0], dtype=np.float64).transpose(2, 1, 0)
    hls_geom = 2
    gdept_0 = np.zeros((JPI, JPJ, JPK), dtype=np.float64)
    gdept_0[hls_geom:JPI - hls_geom, hls_geom:JPJ - hls_geom, :] = gdept_0_interior

    cfg = NemoSEOSConfig()
    alpha_lego, beta_lego = nemo_seos_alpha_beta(t, s, gdept_0, cfg)
    alpha_lego = np.asarray(alpha_lego)
    beta_lego = np.asarray(beta_lego)

    # Restrict the comparison to (a) the INTERIOR domain (mesh_mask.nc has no
    # halo, avoiding any assumption about how the i-periodic/j-closed halo is
    # filled) and (b) levels 1..jpk-2 (0-indexed: 0..jpk-2 exclusive of the
    # last), because rab_3d_t's own DO_3D loop is `1,jpkm1` (eosbn2.F90) --
    # NEMO NEVER writes the deepest level, so comparing it would blame the
    # harness for a level NEMO's own routine does not touch, not a real
    # mismatch. hls=2 confirmed from _load_full_3d's convention in the
    # existing dyn_zad_ldf_walk.py probe (same DINO domain).
    hls = 2
    jpkm1 = JPK - 1
    sl = (slice(hls, JPI - hls), slice(hls, JPJ - hls), slice(0, jpkm1))
    wet = tmask_interior[..., :jpkm1] > 0.5

    a_lego_w = alpha_lego[sl][wet]
    a_nemo_w = alpha_nemo[sl][wet]
    b_lego_w = beta_lego[sl][wet]
    b_nemo_w = beta_nemo[sl][wet]

    d_alpha = a_lego_w - a_nemo_w
    d_beta = b_lego_w - b_nemo_w
    print(f"n wet-interior cells compared (of {wet.size}): {wet.sum()}")
    print(f"alpha: max|diff|={np.max(np.abs(d_alpha)):.3e}  rms|diff|={np.sqrt(np.mean(d_alpha**2)):.3e}")
    print(f"beta : max|diff|={np.max(np.abs(d_beta)):.3e}  rms|diff|={np.sqrt(np.mean(d_beta**2)):.3e}")
    corr_a = np.corrcoef(a_lego_w, a_nemo_w)[0, 1]
    corr_b = np.corrcoef(b_lego_w, b_nemo_w)[0, 1]
    ratio_a = np.sqrt(np.mean(a_lego_w**2)) / np.sqrt(np.mean(a_nemo_w**2))
    ratio_b = np.sqrt(np.mean(b_lego_w**2)) / np.sqrt(np.mean(b_nemo_w**2))
    print(f"alpha corr={corr_a:.9f} ratio={ratio_a:.9f}")
    print(f"beta  corr={corr_b:.9f} ratio={ratio_b:.9f}")

    bar_ok = np.max(np.abs(d_alpha)) < 1e-9 and np.max(np.abs(d_beta)) < 1e-9
    print("POSITIVE CONTROL:", "PASS (machine roundoff)" if bar_ok else "FAIL")
    return 0 if bar_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
