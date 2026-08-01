#!/usr/bin/env python
"""#1226 dyn_ldf CONTRADICTION RECONCILIATION -- Measurement A vs Measurement B.

Two #1226 measurements of the SAME quantity (``dynldf_lev_lap``'s Krhs
increment) disagree ~35x (Rule 1e): ``ww_inheritance_walk.py::
measure_dyn_ldf_corrected`` (commit ``f51cb3318``) reports corr 0.999999999 /
err_norm ~4.5e-05 (roundoff) feeding a REAL restart state through legoESM's
production ``tendencies_with_diagnostics`` API; ``run_dynldf_lap_probe.py``
(commit ``e08132968``, THIS unit-call harness) reports corr~0.965 / ratio
~1.033-1.035 feeding a synthetic velocity field DIRECTLY into
``nemo_ldf_lap_viscosity_cgrid``.

This script does NOT re-derive either recorded number from scratch. It finds
the one config difference between the two call sites that both measure the
SAME operator and checks whether it explains the gap (Rule 1e step 2-3: find
an upstream quantity/config field both should agree on).

THE CANDIDATE, found by reading both scripts + the operator body + NEMO's
source (not inferred):

  - Measurement A goes through ``ocean_pe_latlon_cgrid.py``'s production
    ``lateral_viscosity_operator == "nemo_div_curl"`` branch
    (``ocean_pe_latlon_cgrid.py:2701-2722``), which builds a REAL 3-D
    staircase vertex mask (``_visc_vmask``, from ``is_active`` via
    ``compute_vertex_mask``) and passes it as ``nemo_ldf_lap_viscosity_cgrid``'s
    ``vertex_mask=`` kwarg (also passes ``mask=mask``). The comment at
    :2701-2705 states this exists BECAUSE a bare 2-D vertex mask broadcast
    over levels leaves zeta LIVE at submerged staircase side walls -- "an
    accidental NO-SLIP on every slope/sill face ... that NEMO's 3-D fmask
    zeroes (free-slip)".

  - Measurement B (``run_dynldf_lap_probe.py:194-197``) calls
    ``nemo_ldf_lap_viscosity_cgrid(u_face, v_face, geom, ahmt, ahmf,
    u_mask=u_mask, v_mask=v_mask_arr)`` -- NO ``mask=`` kwarg AND no
    ``vertex_mask=`` kwarg.

  - ``nemo_ldf_lap_viscosity_cgrid``'s own body
    (``latlon_cgrid_operators.py:1373-1384``): the vertex mask on ``zeta``
    is applied ONLY inside ``if mask is not None:``. Since B never passes
    ``mask``, that whole block is skipped -- B's ``zeta`` (curl/vorticity
    term) is COMPLETELY UNMASKED, at every vertex, including staircase
    side-wall vertices.

  - NEMO's own source confirms this masking is NOT optional:
    ``dynldf_lev_rot_scheme.h90:23`` computes
    ``zcur = ahmf(...) * e3f(...) * r1_e1e2f(...) * (...)`` with the inline
    comment "``ahmf already * by fmask``" -- i.e. NEMO folds the fmask
    multiplication into ``ahmf`` at INIT time, not as a separate step in
    this routine. ``ldfdyn.F90:330`` confirms: ``ahmf(:,:,1:jpkm1) =
    ahmf(:,:,1:jpkm1) * fmask(:,:,1:jpkm1)`` -- a full 3-D per-level
    multiplication. legoESM's ``nemo_lateral_viscosity_coefficients``
    (``latlon_cgrid_operators.py:1290-1296``) returns a PURE 1-D
    latitude-only ``ahmf`` with NO fmask folded in (it cannot carry a 3-D
    mask, being ``(n_lat+1,)``), so the caller MUST supply the masking via
    ``vertex_mask=``/``mask=`` -- exactly what production does and B does
    not.

SELF-CHECKS (mandatory):
  1. Reproduce Measurement B's OWN recorded number (corr~0.965-0.966,
     ratio~1.033-1.035 for u, seed 0) by re-running the IDENTICAL synthetic
     setup unmodified -- BEFORE touching anything.
  2. Re-run the SAME synthetic setup with ONLY the vertex mask wired in
     (``mask=`` from the T-mask at the test level, letting the operator
     build ``vertex_mask`` internally via ``compute_vertex_mask`` -- the
     SAME code path production uses, not a hand-rolled mask).
  3. Report both corr/ratio side by side. If (2) reaches AT BAR
     (corr>=0.999999, ratio~1.0), the gap is EXPLAINED by the missing mask
     kwarg -- a harness bug in B, not a real operator defect -- and
     Measurement A's "harness artifact" retraction stands (Measurement B's
     result was itself the harness artifact, from an incomplete kwarg
     list, not a physics defect in ``nemo_ldf_lap_viscosity_cgrid``).

RESULT OF STEPS 1-3 (recorded, not re-argued): step 2's fix barely moves the
needle (u: corr 0.964925->0.967346, ratio 1.034960->1.029331; v: corr
0.983836->0.989739, ratio 1.017671->1.004946) -- NOT an explanation, the
vertex-mask hypothesis is REFUTED as the dominant cause. So this script adds
a SECOND, orthogonal split (Rule 1e step 2: find an upstream quantity both
sides should agree on) that isolates DATA (synthetic jet+noise vs the REAL
restart velocity) from CALL PATH (direct ``nemo_ldf_lap_viscosity_cgrid`` call
vs the full ``tendencies_with_diagnostics`` production API):

  4. Feed the REAL restart Kbb-level velocity (``state.u_before``/``v_before``,
     the SAME corrected time level Task B (of the earlier task) established)
     through B's OWN direct-operator call convention (bypassing
     ``tendencies_with_diagnostics`` entirely), sliced to the SAME single
     level k=2 B tests at, and compare against NEMO's REAL dumped
     ``ldf_dump_du.bin``/``ldf_dump_dv.bin`` at k=2 -- the SAME oracle dump
     Measurement A uses (not a re-derivation).
  5. If this ALSO reaches AT BAR (matching Measurement A's full-3-D result,
     restricted to one level), the disagreement is caused by Measurement B's
     SYNTHETIC INPUT DATA (or the periodic-seam handling it specifically
     stresses with an i-uniform-like pattern), not by the direct-call path
     itself, not by ``tendencies_with_diagnostics`` machinery A relies on
     that B's direct call skips. If it does NOT reach bar, the direct-call
     path itself (not the vertex mask, not the data) is implicated, and the
     next localisation step is the periodic-seam construction in
     ``_u_east_to_face_periodic``/``curl_vertex_cgrid``.

Never lets this probe print its own verdict on WHICH measurement is
"correct" -- only the numbers, side by side. That call is made by the task
report reading this output, not by the probe.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.unit_harness.run_dynldf_lap_vertexmask_reconcile
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
from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode  # noqa: E402
from legoesm.ocean.fidelity.nemo_io import (  # noqa: E402
    read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (  # noqa: E402
    bridge_nemo_to_legoesm_topo, bridge_before_state_topo,
)
from legoesm.ocean.experiments.dino import dino_config_for_recipe  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    nemo_ldf_lap_viscosity_cgrid, nemo_lateral_viscosity_coefficients,
)

JPI, JPJ, JPK = 56, 203, 36
HLS = 2
RUN_DIR = REPO_ROOT.parent / "oracle-builds" / "nemo5" / "nemo_5.0.2" / "cfgs" / "DINO" / "RUN_GDB"
MESH_PATH = RUN_DIR / "mesh_mask.nc"
RESTART = "DINO_00057600_restart.nc"
RN_UV = 0.27   # DINO namelist_cfg:369 rn_Uv [m/s] -- &namdyn_ldf, nn_ahm_ijk_t=20
K_TEST_LEVEL = 2   # same test level as run_dynldf_lap_probe.py


def build_synthetic_velocity(rng: np.random.Generator):
    """IDENTICAL to run_dynldf_lap_probe.py::build_synthetic_velocity -- same
    seed reproduces the SAME field, required for self-check 1."""
    j = np.arange(JPJ)
    u_pattern = 0.3 * np.sin(2 * np.pi * j / JPJ)[None, :] * np.ones((JPI, JPJ))
    v_pattern = 0.1 * np.cos(2 * np.pi * j / JPJ)[None, :] * np.ones((JPI, JPJ))
    u = u_pattern + rng.normal(0.0, 0.03, size=(JPI, JPJ))
    v = v_pattern + rng.normal(0.0, 0.03, size=(JPI, JPJ))
    return u.astype(np.float64), v.astype(np.float64)


def _corr_ratio(a, b):
    corr = float(np.corrcoef(a, b)[0, 1])
    ratio = float(np.sqrt(np.mean(a ** 2)) / np.sqrt(np.mean(b ** 2)))
    return corr, ratio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default=str(RUN_DIR))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_policy(PrecisionPolicy.fp64())
    rundir = Path(args.rundir)
    exe = rundir / "harness_dynldf_lap.exe"
    if not exe.exists():
        print(f"FATAL: {exe} not found -- build it first.")
        return 1

    rng = np.random.default_rng(args.seed)
    u_2d, v_2d = build_synthetic_velocity(rng)

    uu_3d = np.broadcast_to(u_2d[:, :, None], (JPI, JPJ, JPK)).copy()
    vv_3d = np.broadcast_to(v_2d[:, :, None], (JPI, JPJ, JPK)).copy()
    uu4 = np.stack([uu_3d, uu_3d, uu_3d], axis=-1)
    vv4 = np.stack([vv_3d, vv_3d, vv_3d], axis=-1)

    require_fp64(uu4, context="dynldf_lap vertex-mask reconcile input")

    binary_io.write_array(str(rundir / "ldf_in_uu.bin"), uu4)
    binary_io.write_array(str(rundir / "ldf_in_vv.bin"), vv4)

    print(f"Running {exe} in {rundir} ...")
    result = subprocess.run([str(exe)], cwd=str(rundir), capture_output=True, text=True, timeout=120)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print("FORTRAN DRIVER FAILED:")
        print(result.stderr[-4000:])
        return 1

    du_nemo = binary_io.read_array(str(rundir / "ldf_out_du.bin"), (JPI, JPJ, JPK))
    dv_nemo = binary_io.read_array(str(rundir / "ldf_out_dv.bin"), (JPI, JPJ, JPK))

    grid = read_nemo_mesh_mask(str(MESH_PATH), nn_hls=0)
    now = read_nemo_restart(str(rundir / RESTART), nn_hls=0)
    cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")

    u_2d_interior_ji = u_2d[HLS:JPI - HLS, HLS:JPJ - HLS].T
    v_2d_interior_ji = v_2d[HLS:JPI - HLS, HLS:JPJ - HLS].T
    assert u_2d_interior_ji.shape == now.u.shape[:2]
    u_nemo_face = np.broadcast_to(u_2d_interior_ji[:, :, None], now.u.shape).astype(np.float64)
    v_nemo_face = np.broadcast_to(v_2d_interior_ji[:, :, None], now.v.shape).astype(np.float64)
    synth_state = now._replace(u=u_nemo_face, v=v_nemo_face)

    br = bridge_nemo_to_legoesm_topo(grid, synth_state, periodic_i=True, full_step=True, omega=cfg.omega)
    geom = br.geometry
    u_face = np.asarray(br.state.u.data)
    v_face = np.asarray(br.state.v.data)
    u_mask = np.asarray(br.state.u_mask.data)
    v_mask_arr = np.asarray(br.state.v_mask.data)
    print(f"bridge f_T match max|diff| = {br.f_match_max_abs:.3e}  (build self-check)")

    ahmt, ahmf = nemo_lateral_viscosity_coefficients(geom, half_UM=0.5 * RN_UV)

    # --- Reproduce B's own recorded call, UNMODIFIED (self-check 1) ---
    du_lego_face_B, dv_lego_face_B = nemo_ldf_lap_viscosity_cgrid(
        u_face, v_face, geom, ahmt, ahmf,
        u_mask=u_mask, v_mask=v_mask_arr,
    )

    # --- Same call, WITH the T-mask at the test level wired in (mask=),
    # letting the operator build vertex_mask internally via
    # compute_vertex_mask -- the SAME code path production uses (Rule 1e:
    # do not hand-roll the mask, reuse the operator's own construction).
    tmask3_ji = np.asarray(grid.tmask)  # (n_lat, n_lon, nlev), already halo-stripped
    t_mask_level = (tmask3_ji[..., K_TEST_LEVEL] > 0.5).astype(np.float64)
    du_lego_face_fix, dv_lego_face_fix = nemo_ldf_lap_viscosity_cgrid(
        u_face, v_face, geom, ahmt, ahmf,
        mask=t_mask_level, u_mask=u_mask, v_mask=v_mask_arr,
    )

    def _to_nemo_ij(face2level):
        du2d = face2level[0][:, 1:, K_TEST_LEVEL]
        dv2d = face2level[1][1:, :, K_TEST_LEVEL]
        return du2d.T, dv2d.T

    du_B_ij, dv_B_ij = _to_nemo_ij((du_lego_face_B, dv_lego_face_B))
    du_fix_ij, dv_fix_ij = _to_nemo_ij((du_lego_face_fix, dv_lego_face_fix))

    import netCDF4 as nc
    with nc.Dataset(MESH_PATH) as ds:
        umask_i = np.asarray(ds.variables["umask"][0, K_TEST_LEVEL], dtype=np.float64)
        vmask_i = np.asarray(ds.variables["vmask"][0, K_TEST_LEVEL], dtype=np.float64)
    umask_ij = umask_i.T
    vmask_ij = vmask_i.T

    sl = (slice(HLS, JPI - HLS), slice(HLS, JPJ - HLS))
    wet_u = umask_ij > 0.5
    wet_v = vmask_ij > 0.5

    b_u = du_nemo[sl + (K_TEST_LEVEL,)][wet_u]
    b_v = dv_nemo[sl + (K_TEST_LEVEL,)][wet_v]

    a_u_B = du_B_ij[wet_u]
    a_v_B = dv_B_ij[wet_v]
    a_u_fix = du_fix_ij[wet_u]
    a_v_fix = dv_fix_ij[wet_v]

    corr_u_B, ratio_u_B = _corr_ratio(a_u_B, b_u)
    corr_v_B, ratio_v_B = _corr_ratio(a_v_B, b_v)
    corr_u_fix, ratio_u_fix = _corr_ratio(a_u_fix, b_u)
    corr_v_fix, ratio_v_fix = _corr_ratio(a_v_fix, b_v)

    print("\n" + "=" * 78)
    print("SELF-CHECK 1: reproduce Measurement B's own recorded number "
          "(no mask kwarg, unmodified)")
    print("=" * 78)
    print(f"  u: corr={corr_u_B:.6f} ratio={ratio_u_B:.6f}  "
          f"(B recorded ~0.965-0.966 / ~1.033-1.035)")
    print(f"  v: corr={corr_v_B:.6f} ratio={ratio_v_B:.6f}")
    b_repro_u = abs(corr_u_B - 0.965) < 0.02
    print(f"  reproduces B's regime (corr within 0.02 of ~0.965): {b_repro_u}")
    if not b_repro_u:
        print("  STOP: this probe does not reproduce Measurement B's own "
              "number -- do not trust the fix-comparison below.")
        return 1

    print("\n" + "=" * 78)
    print("SELF-CHECK 2: SAME call + mask= (T-mask @ k=2) wired in "
          "(operator builds vertex_mask internally, production's own path)")
    print("=" * 78)
    print(f"  u: corr={corr_u_fix:.9f} ratio={ratio_u_fix:.9f}")
    print(f"  v: corr={corr_v_fix:.9f} ratio={ratio_v_fix:.9f}")

    print("\n" + "=" * 78)
    print("SIDE BY SIDE (no verdict printed here -- read by the task report)")
    print("=" * 78)
    print(f"  u  no-mask: corr={corr_u_B:.6f} ratio={ratio_u_B:.6f}   "
          f"with-mask: corr={corr_u_fix:.9f} ratio={ratio_u_fix:.9f}")
    print(f"  v  no-mask: corr={corr_v_B:.6f} ratio={ratio_v_B:.6f}   "
          f"with-mask: corr={corr_v_fix:.9f} ratio={ratio_v_fix:.9f}")
    print(f"  n wet u-cells (k={K_TEST_LEVEL}): {int(wet_u.sum())} of {wet_u.size}")
    print(f"  n wet v-cells (k={K_TEST_LEVEL}): {int(wet_v.sum())} of {wet_v.size}")

    # =========================================================================
    # STEP 4/5: isolate DATA (synthetic vs real) from CALL PATH (direct
    # operator call vs tendencies_with_diagnostics). Feed the REAL restart
    # Kbb-level velocity through B's OWN direct-call convention, sliced to
    # the SAME single level k=2, compared against NEMO's REAL dumped
    # ldf_dump_du.bin/dv.bin -- the SAME oracle dump Measurement A uses.
    # =========================================================================
    print("\n" + "=" * 78)
    print("STEP 4/5: REAL restart data (Kbb) through B's DIRECT-CALL path, "
          "single level k=2, vs NEMO's REAL ldf_dump_du/dv.bin")
    print("=" * 78)
    e3t_mode = require_explicit_e3t_mode(context="run_dynldf_lap_vertexmask_reconcile step4")
    print(f"  LEGOESM_NEMO_E3T={e3t_mode!r}")

    g_full = read_nemo_mesh_mask(str(MESH_PATH), nn_hls=0)
    s_full = read_nemo_restart(str(RUN_DIR / "DINO_00057600_restart.nc"), nn_hls=0)
    br_full = bridge_nemo_to_legoesm_topo(g_full, s_full, periodic_i=True, full_step=True)
    before_full = read_nemo_restart_before(str(RUN_DIR / "DINO_00057600_restart.nc"), nn_hls=0)
    br_before_full = bridge_before_state_topo(br_full._replace(state=br_full.state), g_full, before_full,
                                               periodic_i=True)
    require_fp64(br_full.geometry, br_full.z_coord, br_full.state, context="step4")

    u_before_face = np.asarray(br_before_full.u_before.data)
    v_before_face = np.asarray(br_before_full.v_before.data)
    u_mask_full = np.asarray(br_full.state.u_mask.data)
    v_mask_full = np.asarray(br_full.state.v_mask.data)

    # T-mask at the test level -- the SAME mask= wiring self-check 2 above
    # established is available (production passes ``mask=`` too, at
    # ocean_pe_latlon_cgrid.py:2721); this isolates the direct-call path
    # from the "no mask=" harness omission simultaneously.
    tmask_full_lvl = (np.asarray(g_full.tmask)[..., K_TEST_LEVEL] > 0.5).astype(np.float64)

    ahmt_full, ahmf_full = nemo_lateral_viscosity_coefficients(br_full.geometry, half_UM=0.5 * RN_UV)
    # 2x2 control: real data WITH and WITHOUT mask=, isolating the mask's
    # true effect on REAL data (vs its ~0 effect measured on synthetic data
    # in self-checks 1/2 above).
    du_real_face_nomask, dv_real_face_nomask = nemo_ldf_lap_viscosity_cgrid(
        u_before_face, v_before_face, br_full.geometry, ahmt_full, ahmf_full,
        u_mask=u_mask_full, v_mask=v_mask_full,
    )
    du_real_face, dv_real_face = nemo_ldf_lap_viscosity_cgrid(
        u_before_face, v_before_face, br_full.geometry, ahmt_full, ahmf_full,
        mask=tmask_full_lvl, u_mask=u_mask_full, v_mask=v_mask_full,
    )
    du_real_ij = du_real_face[:, 1:, K_TEST_LEVEL].T   # (n_lon, n_lat), drop west-wall col
    dv_real_ij = dv_real_face[1:, :, K_TEST_LEVEL].T   # drop south-wall row
    du_real_nomask_ij = du_real_face_nomask[:, 1:, K_TEST_LEVEL].T
    dv_real_nomask_ij = dv_real_face_nomask[1:, :, K_TEST_LEVEL].T

    jpi_l, jpj_l, jpkm1_l = 56, 203, 35
    du_nemo_real = np.fromfile(str(RUN_DIR / "ldf_dump_du.bin"), dtype="<f8").reshape(
        jpkm1_l, jpj_l, jpi_l)[K_TEST_LEVEL, HLS:-HLS, HLS:-HLS]
    dv_nemo_real = np.fromfile(str(RUN_DIR / "ldf_dump_dv.bin"), dtype="<f8").reshape(
        jpkm1_l, jpj_l, jpi_l)[K_TEST_LEVEL, HLS:-HLS, HLS:-HLS]
    # nemo dump is (j,i) already interior -- match du_real_ij's (i,j) by transposing.
    du_nemo_real_ij = du_nemo_real.T
    dv_nemo_real_ij = dv_nemo_real.T

    n_i = min(du_real_ij.shape[0], du_nemo_real_ij.shape[0])
    n_j = min(du_real_ij.shape[1], du_nemo_real_ij.shape[1])
    umask_full_lvl = (np.asarray(g_full.umask)[:, :, K_TEST_LEVEL] > 0.5).T  # (i,j)
    vmask_full_lvl = (np.asarray(g_full.vmask)[:, :, K_TEST_LEVEL] > 0.5).T
    n_i_u = min(n_i, umask_full_lvl.shape[0])
    n_j_u = min(n_j, umask_full_lvl.shape[1])
    n_i_v = min(n_i, vmask_full_lvl.shape[0])
    n_j_v = min(n_j, vmask_full_lvl.shape[1])
    wu = umask_full_lvl[:n_i_u, :n_j_u]
    wv = vmask_full_lvl[:n_i_v, :n_j_v]

    a_u_real = du_real_ij[:n_i_u, :n_j_u][wu]
    b_u_real = du_nemo_real_ij[:n_i_u, :n_j_u][wu]
    a_v_real = dv_real_ij[:n_i_v, :n_j_v][wv]
    b_v_real = dv_nemo_real_ij[:n_i_v, :n_j_v][wv]
    a_u_real_nomask = du_real_nomask_ij[:n_i_u, :n_j_u][wu]
    a_v_real_nomask = dv_real_nomask_ij[:n_i_v, :n_j_v][wv]

    corr_u_real, ratio_u_real = _corr_ratio(a_u_real, b_u_real)
    corr_v_real, ratio_v_real = _corr_ratio(a_v_real, b_v_real)
    corr_u_real_nm, ratio_u_real_nm = _corr_ratio(a_u_real_nomask, b_u_real)
    corr_v_real_nm, ratio_v_real_nm = _corr_ratio(a_v_real_nomask, b_v_real)
    print(f"  u (real data, direct call, NO mask=, k={K_TEST_LEVEL}): "
          f"corr={corr_u_real_nm:.9f} ratio={ratio_u_real_nm:.9f}  n={int(wu.sum())}")
    print(f"  v (real data, direct call, NO mask=, k={K_TEST_LEVEL}): "
          f"corr={corr_v_real_nm:.9f} ratio={ratio_v_real_nm:.9f}  n={int(wv.sum())}")
    print(f"  u (real data, direct call, WITH mask=, k={K_TEST_LEVEL}): "
          f"corr={corr_u_real:.9f} ratio={ratio_u_real:.9f}  n={int(wu.sum())}")
    print(f"  v (real data, direct call, WITH mask=, k={K_TEST_LEVEL}): "
          f"corr={corr_v_real:.9f} ratio={ratio_v_real:.9f}  n={int(wv.sum())}")
    print("  (compare: Measurement A's full-3-D/full-API result was corr "
          "0.999999999 for both u,v -- this isolates whether restricting to "
          "ONE level + the direct-call path alone already degrades the match, "
          "vs whether B's SYNTHETIC data is additionally implicated)")

    print("\n" + "=" * 78)
    print("FULL 2x2 (data x mask=), no verdict printed here")
    print("=" * 78)
    print(f"  u  synthetic/no-mask={corr_u_B:.6f}  synthetic/with-mask={corr_u_fix:.9f}  "
          f"real/no-mask={corr_u_real_nm:.9f}  real/with-mask={corr_u_real:.9f}")
    print(f"  v  synthetic/no-mask={corr_v_B:.6f}  synthetic/with-mask={corr_v_fix:.9f}  "
          f"real/no-mask={corr_v_real_nm:.9f}  real/with-mask={corr_v_real:.9f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
