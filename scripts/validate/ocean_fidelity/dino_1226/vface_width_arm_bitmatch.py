#!/usr/bin/env python
"""#1455 -- does the FAITHFUL CONSTRUCTION reproduce the override ARM's array?

WHY THIS EXISTS.  The v-face zonal width override arm
(``substep_traj_compare.py`` with ``DINO_1455_SUB_VFACE=nemo``) substituted
NEMO's own ``e1v`` ARRAY into the loop and produced the five-state table that
``dino_wall_fixed_bias.md`` section 6a scores.  The fix that followed is a
different object: it changes how legoESM CONSTRUCTS that width, evaluating it
at the true v-face (Mercator) latitude -- NEMO's ``gphiv``,
``usrdef_hgr.F90:113`` -- instead of at the mean of the two adjacent tracer
latitudes.

Those two are only the same experiment WHERE they produce the same array.  This
probe decides that, offline, with no model run, by rebuilding BOTH arrays and
comparing them element by element.

SCOPE, and it is not optional (adversarial review finding 2).  An identical
ARRAY is not an identical EXPERIMENT.  The arm substituted into the barotropic
substep loop's kwargs and its EEN pre-block and ran THAT LOOP; the builder fix
changes the width for the whole model.  So a bit-match licenses exactly one
claim -- that the arm's states already measure this width inside the barotropic
substep loop -- and licenses nothing about the baroclinic-side consumers the arm
never perturbed (the F-point lateral-viscosity coefficient, GM/Redi's isoneutral
tensor, the bolus velocity, MLE, the baroclinic EEN weighting).  The verdict
this probe prints says so.

It deliberately reconstructs the arm's array from the arm's OWN source lines
(``substep_traj_compare.py:1174-1177``) rather than importing that module,
which would drag in the whole stepping harness; the reconstruction is pinned by
asserting the arm's own row-alignment discriminator here too, so a drifted
alignment fails loudly instead of comparing the wrong rows.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      .venv/bin/python -m \
      scripts.validate.ocean_fidelity.dino_1226.vface_width_arm_bitmatch
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[3]
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SEQDUMP = os.environ.get(
    "DINO_NEMO_RUN_SEQDUMP",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_Y20_1R")


def stamp() -> None:
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  dirty_tracked={len(dirt.splitlines())}")
    print(f"PROVENANCE  seqdump={SEQDUMP}")
    print(f"PROVENANCE  JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}")
    try:
        import jax
        print(f"PROVENANCE  backend={jax.default_backend()} "
              f"devices={[d.platform for d in jax.devices()]}")
    except Exception as _e:                      # pragma: no cover
        print(f"PROVENANCE  backend=UNKNOWN ({_e})")


def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    import netCDF4 as nc

    import wall_term_discriminators as W

    stamp()
    _g, br, _cfg, _mc = W.build_lego()
    geom = br.geometry
    dxv_fix = np.asarray(geom.dx_v, dtype=np.float64)      # (jpj+1, jpi)
    jpj = dxv_fix.shape[0] - 1
    jpi = dxv_fix.shape[1]
    print(f"\nlegoESM v-face frame: ({jpj + 1}, {jpi})")

    # TWO MESH FILES, and that is a confound until it is checked (adversarial
    # review finding 4).  The geometry above is built from
    # ``wall_term_discriminators.MESH``; the reference array below is read from
    # the ARM's own oracle directory, because the object being reproduced is
    # the arm's substituted array and not some other run's.  They are the same
    # mesh today -- but "the same today" is luck, not a control, so the two
    # files' e1v are compared before either is used.
    arm_path = os.path.join(SEQDUMP, "mesh_mask.nc")
    d = nc.Dataset(arm_path)
    try:
        e1v = np.asarray(d.variables["e1v"][0], dtype=np.float64)
        # NEMO's own v-face wet mask, for the wall-row exclusion below.
        vmask = np.asarray(d.variables["vmask"][0], dtype=np.float64)
    finally:
        d.close()
    d2 = nc.Dataset(str(W.MESH))
    try:
        e1v_geom_mesh = np.asarray(d2.variables["e1v"][0], dtype=np.float64)
    finally:
        d2.close()
    print(f"  arm's mesh      : {arm_path}")
    print(f"  geometry's mesh : {W.MESH}")
    if e1v_geom_mesh.shape != e1v.shape or not np.array_equal(
            e1v_geom_mesh, e1v):
        raise SystemExit(
            "FATAL: the mesh the geometry was built from and the mesh the "
            "arm's reference array is read from carry DIFFERENT e1v. The "
            "comparison below would be between two different meshes.")
    print("  the two meshes carry a bit-identical e1v -- not assumed, checked")
    if e1v.shape != (jpj, jpi):
        raise SystemExit(
            f"NEMO e1v is {e1v.shape}, want ({jpj}, {jpi}) -- this probe is "
            "reading a different mesh than the geometry was built on")

    # THE ARM'S ROW ALIGNMENT, re-established here rather than assumed.  Same
    # two-hypothesis discriminator the arm uses: the correct map must score
    # small AND at least 100x better than the best wrong one.
    def resid(shift: int) -> float:
        j = np.arange(1, jpj + 1) + shift
        ok = (j >= 0) & (j < jpj) & (dxv_fix[1:jpj + 1, 0] > 0.0)
        cand = e1v[np.clip(j, 0, jpj - 1), 0]
        return float(np.median(
            np.abs(dxv_fix[1:jpj + 1, 0][ok] - cand[ok]) / cand[ok]))

    scores = {sh: resid(sh) for sh in (-2, -1, 0, 1, 2)}
    best_wrong = min(v for sh, v in scores.items() if sh != -1)
    for sh, sc in sorted(scores.items()):
        print(f"  row map lego j <- NEMO j{sh:+d}: median rel={sc:.3e}"
              + ("   <- the arm's map" if sh == -1 else ""))
    if not (scores[-1] < 1e-3 and scores[-1] * 100.0 < best_wrong):
        raise SystemExit(
            f"FATAL: row alignment not established (chosen {scores[-1]:.3e}, "
            f"best wrong {best_wrong:.3e}) -- the comparison below would be "
            "between different physical rows")

    # THE ARM'S ARRAY, rebuilt exactly as substep_traj_compare.py:1174-1177
    # builds it: rows 1..jpj-1 take NEMO's e1v[0..jpj-2]; row 0 and row jpj
    # keep legoESM's pole-zeroed wall.
    # The two END rows are copied from dxv_fix and are therefore EQUAL BY
    # CONSTRUCTION -- they cannot contribute to the count below, so the count
    # alone would quietly report "0 differ" over 104 tautological cells
    # (adversarial review finding 5).  The arm asserted the pole-zero wall at
    # its own point of use; assert it here too, so those rows are CHECKED
    # rather than merely uncompared.
    # legoESM v-face j corresponds to NEMO's V row j-1 (established by the
    # alignment discriminator above), so the two legoESM wall rows are NEMO's
    # first and last V rows.  vmask is (nlev, jpj, jpi).
    for jw, nemo_j in ((0, 0), (jpj, jpj - 1)):
        if float(vmask[:, nemo_j, :].max()) > 0.5:
            raise SystemExit(
                f"FATAL: legoESM v row {jw} (NEMO V row {nemo_j}) carries wet "
                "faces. The arm leaves that row at legoESM's own metric, so "
                "the arm and the construction fix would differ on a row that "
                "TRANSPORTS -- and this comparison excludes it. (The arm "
                "refuses to run for the same reason.)")
        if float(np.abs(dxv_fix[jw]).max()) != 0.0:
            raise SystemExit(
                f"FATAL: v row {jw} is not the pole-zeroed wall both the arm "
                f"and the faithful construction assume (max "
                f"{np.abs(dxv_fix[jw]).max():.3e}); the arm leaves that row at "
                "legoESM's own metric, so the two would differ there and this "
                "comparison would not see it")
    dxv_arm = np.array(dxv_fix)
    dxv_arm[1:jpj + 1] = e1v[np.clip(np.arange(1, jpj + 1) - 1, 0, jpj - 1)]
    dxv_arm[jpj] = dxv_fix[jpj]

    diff = dxv_fix - dxv_arm
    n_diff = int((diff != 0.0).sum())
    n_taut = 2 * jpi          # the two end rows, equal by construction
    print(f"\n=== THE BIT-MATCH ===")
    print(f"  cells compared        : {dxv_fix.size} "
          f"({dxv_fix.size - n_taut} substituted + {n_taut} end-wall cells "
          f"that are equal by construction and asserted zero above)")
    print(f"  cells differing       : {n_diff}")
    print(f"  max |difference| [m]  : {np.abs(diff).max():.6e}")
    if n_diff:
        jj, ii = np.nonzero(diff)
        rows = sorted(set(jj.tolist()))
        print(f"  differing v-rows      : {rows[:12]}"
              f"{' ...' if len(rows) > 12 else ''}")
        rel = np.abs(diff[jj, ii]) / np.maximum(np.abs(dxv_arm[jj, ii]), 1e-30)
        print(f"  max |relative|        : {rel.max():.6e}")

    # THE BACKEND MATTERS AND IT WAS NEARLY MISSED (2026-08-27).  The first
    # run of this probe was on CPU and reported 0 differing cells; the same
    # probe on GPU reports 1352, at a max RELATIVE difference of 2.5e-16 --
    # one ulp of float64, from the device transcendental library rounding
    # cos() differently.  That is 1.3e11 times smaller than the 3.3485e-05
    # defect and physically irrelevant, but "BIT-IDENTICAL" is a claim about
    # bits and it is FALSE on GPU.  So the verdict is three-valued and the
    # backend is stamped above, rather than a CPU run being quoted as though
    # it were universal.
    ulp = 2.220446049250313e-16                  # float64 eps
    rel_max = 0.0
    if n_diff:
        jj, ii = np.nonzero(diff)
        rel_max = float((np.abs(diff[jj, ii])
                         / np.maximum(np.abs(dxv_arm[jj, ii]), 1e-30)).max())
    # The registered call, COMPUTED from the numbers above.
    print()
    if 0 < n_diff and rel_max <= 4.0 * ulp:
        print(f"  VERDICT: EQUIVALENT TO {rel_max / ulp:.1f} ULP, not "
              f"bit-identical on this backend.")
        print(f"  {n_diff} cells differ, max relative {rel_max:.3e} -- "
              f"float64 rounding in the device")
        print("  cos(), NOT a construction difference. It is "
              f"{3.3485e-05 / rel_max:.1e}x smaller than the")
        print("  defect being fixed. The construction is the same one; the "
              "bits are not.")
        rc = 0
    elif n_diff == 0:
        print("  VERDICT: BIT-IDENTICAL. The faithful construction produces "
              "exactly the array the")
        print("  override arm substituted.")
        print()
        print("  SCOPE THIS EXACTLY (adversarial review finding 2). The arm "
              "substituted into the")
        print("  BAROTROPIC SUBSTEP LOOP's kwargs and its EEN pre-block, and "
              "ran that loop. So what")
        print("  this equality buys is: the arm's five states already measure "
              "this width's effect")
        print("  INSIDE the barotropic substep loop, and no re-run is owed to "
              "re-measure THAT.")
        print("  The builder fix additionally moves baroclinic-side consumers "
              "the arm never touched")
        print("  (the F-point lateral-viscosity coefficient, GM/Redi, the "
              "bolus velocity, MLE, the")
        print("  baroclinic EEN weighting). Those all move TOWARD NEMO, but "
              "their magnitude is NOT")
        print("  measured by this equality and must not be claimed from it.")
        rc = 0
    else:
        print("  VERDICT: NOT bit-identical. The fix and the arm are "
              "DIFFERENT experiments; the")
        print("  arm's numbers may NOT be quoted for the fix without a "
              "re-run.")
        rc = 1

    # A control the comparison would be worthless without: the PRE-fix
    # construction must NOT bit-match, or this probe cannot distinguish the
    # two constructions at all.
    lat_s = np.asarray(geom.lat_T, dtype=np.float64)[:, 0]
    cos_mid = np.cos(0.5 * (lat_s[:-1] + lat_s[1:]))
    dxv_pre = np.zeros_like(dxv_fix)
    dxv_pre[1:-1] = (float(geom.radius) * float(br.geometry.dlon)
                     * cos_mid)[:, None]
    pre_diff = int((dxv_pre[1:jpj] != dxv_arm[1:jpj]).sum())
    pre_rel = float((np.abs(dxv_pre[1:jpj] - dxv_arm[1:jpj])
                     / dxv_arm[1:jpj]).max())
    print(f"\n  CONTROL (the pre-fix T-midpoint construction vs the same arm "
          f"array):")
    print(f"    cells differing {pre_diff}, max relative {pre_rel:.4e}")
    if pre_diff == 0:
        raise SystemExit(
            "FATAL: the PRE-fix construction also bit-matches the arm, so "
            "this probe cannot tell the two constructions apart and its "
            "verdict above is vacuous")
    print("    the control fires: the two constructions ARE distinguishable "
          "by this comparison.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
