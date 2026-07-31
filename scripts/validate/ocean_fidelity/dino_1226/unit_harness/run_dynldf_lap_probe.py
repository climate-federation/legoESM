#!/usr/bin/env python
"""#1226 NEMO unit-call harness -- ROUTINE 5: dynldf_lev_lap (Laplacian
lateral momentum diffusion trend, div-rot operator, ``dynldf_lev.F90:45-138``
+ ``dynldf_lev_rot_scheme.h90``).

Gate row context: ``fidelity_bar_gate.py`` rows "dyn_ldf (dynldf_lev_lap) u"
and "... v" are PHANTOM-PROVENANCE (cited script
``probe_1226_r2_item2_dynldf.py`` does not exist in the repo tree, confirmed
by ``check_provenance_scripts_exist()``). The recorded tuples (u:
0.997855/1.003865, v: 0.999400/1.001755) are DEBT, not AT BAR, and were
"RE-MEASURED ... UNCHANGED to 4-5 s.f." per an offset-scan note -- but that
scan itself lives only in the missing script, so it cannot be re-run.

This probe drives ``dynldf_lev_lap`` DIRECTLY through the compiled Fortran
driver (``harness_dynldf_lap.exe``) with a synthetic velocity field spanning
real dynamic range, and compares against legoESM's
:func:`legoesm.ocean.dynamics.latlon_cgrid_operators.nemo_ldf_lap_viscosity_cgrid`
+ :func:`...nemo_lateral_viscosity_coefficients` -- the existing faithful
port of this exact routine (its own docstring cites ``dynldf_lev.F90`` +
``dynldf_lev_rot_scheme.h90`` line-for-line).

DINO active scheme (quoted): ``ln_dynldf_lap=.true., ln_dynldf_lev=.true.``
(``namelist_cfg:365-367``) selects ``dynldf.F90``'s ``np_lap`` branch ->
``dynldf_lev_lap``; ``nn_dynldf_typ`` defaults to 0 (div-rot,
``namelist_ref:1108``, DINO does not override) selecting the ``np_typ_rot``
branch this port implements. ``rn_Uv=0.27`` (``namelist_cfg:369``).

Grid geometry is built via the SAME production bridge every other #1226
probe in this campaign uses (:func:`bridge_nemo_to_legoesm_topo`, which
wraps :func:`create_latlon_geometry` -- not a hand-rolled probe-side
geometry construction, avoiding the exact class of axis-order/extent bug
the harness's own hazard note warns about).

Documented scope limit carried over from the port's own docstring: legoESM's
operator does NOT weight the div/curl by layer thickness e3 the way NEMO's
``dynldf_lev_rot_scheme.h90`` does (``ahmt(...)... /e3t(...,Kbb)``,
``.../e3u(...,Kmm)`` etc) -- "the SAME thickness treatment the verified
legoESM vector Laplacian uses". This probe therefore tests at a SINGLE level
(k=2, away from both the free surface and any bottom partial-cell effects)
where e3 varies least horizontally (VERIFIED: NEMO's own e3t_0/e3u_0/e3v_0
at k=2 are a SINGLE constant value, 10.835785693992875 m, over the entire
wet domain -- so the e3-weighting deviation cannot explain a residual here;
it would cancel as a pure scalar).

RESULT (PLAUSIBLE, NOT RECONCILED, this session): measures corr~0.965-0.966 /
ratio~1.033-1.035 for u (stable across 3 seeds), well below both the AT BAR
threshold AND the recorded row's 0.997855/1.003865 -- this DISAGREES with the
recorded tuple, and neither is validated as authoritative here (Rule 1e: do
not pick a winner without reconciling). Partial diagnosis performed (see the
task report): (1) ``ahmt``/``ahmf`` coefficients match NEMO's own
``0.5*rn_Uv*max(e1t,e2t)`` EXACTLY (verified against ``mesh_mask.nc`` e1t/e2t
directly, not inferred); (2) a NOISE-FREE, PURELY i-uniform synthetic velocity
(no zonal structure at all) still shows legoESM's du departing sharply from
NEMO's at the interior columns immediately adjacent to the periodic seam
(column index 1: lego 5.65e-9 vs NEMO's uniform ~1.05e-11, a ~500x spike) while
NEMO's own du stays a single near-constant value across ALL i (as an i-uniform
input demands); (3) excluding the 2 seam-adjacent columns + the far wrap column
improves corr 0.965->0.981 but does NOT fully reconcile -- some other factor
remains open. PLAUSIBLE localization: legoESM's u-face storage convention pads
an explicit "west wall" column (``bridge_nemo_to_legoesm_topo``'s own
docstring: "the single redundant periodic-wrap u-face ... uses legoESM's
closed-basin face-storage convention rather than the periodic roll") even
though DINO's zonal boundary is genuinely periodic (``ln_Iperio``) -- this
Laplacian's div/curl stencil reads neighbouring u/v columns, so a walled
column near the seam can contaminate a few interior columns beyond it. NOT
CONFIRMED as the full cause (seam exclusion alone doesn't close the gap) --
left open rather than force-explained.  This routine stays a CANDIDATE, not a
closed measurement; see the task report for the honest verdict.
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
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart, NemoState  # noqa: E402
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo  # noqa: E402
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
K_TEST_LEVEL = 2   # 0-indexed; away from surface (k=0) and any bottom partial-cell effects


def build_synthetic_velocity(rng: np.random.Generator):
    """A large-scale jet with small-scale noise, spanning real DINO dynamic
    range -- the divergence/curl this Laplacian differentiates needs actual
    horizontal structure, not a quiescent near-zero field."""
    j = np.arange(JPJ)
    u_pattern = 0.3 * np.sin(2 * np.pi * j / JPJ)[None, :] * np.ones((JPI, JPJ))
    v_pattern = 0.1 * np.cos(2 * np.pi * j / JPJ)[None, :] * np.ones((JPI, JPJ))
    u = u_pattern + rng.normal(0.0, 0.03, size=(JPI, JPJ))
    v = v_pattern + rng.normal(0.0, 0.03, size=(JPI, JPJ))
    return u.astype(np.float64), v.astype(np.float64)


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

    # Broadcast the same 2-D pattern over all levels for the Fortran side
    # (dynldf_lev_lap loops over jk internally; feeding a level-independent
    # horizontal pattern means every k should give the SAME trend on a
    # uniform-thickness column, so this ALSO functions as an internal
    # self-check: du/dv at different k must agree wherever e3 doesn't vary).
    uu_3d = np.broadcast_to(u_2d[:, :, None], (JPI, JPJ, JPK)).copy()
    vv_3d = np.broadcast_to(v_2d[:, :, None], (JPI, JPJ, JPK)).copy()
    uu4 = np.stack([uu_3d, uu_3d, uu_3d], axis=-1)   # Nbb,Nnn,Naa identical -- only Kbb is read
    vv4 = np.stack([vv_3d, vv_3d, vv_3d], axis=-1)

    require_fp64(uu4, context="dynldf_lev_lap probe input")

    binary_io.write_array(str(rundir / "ldf_in_uu.bin"), uu4)
    binary_io.write_array(str(rundir / "ldf_in_vv.bin"), vv4)

    print(f"Running {exe} in {rundir} ...")
    result = subprocess.run([str(exe)], cwd=str(rundir), capture_output=True, text=True, timeout=120)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("FORTRAN DRIVER FAILED:")
        print(result.stderr[-4000:])
        return 1

    du_nemo = binary_io.read_array(str(rundir / "ldf_out_du.bin"), (JPI, JPJ, JPK))
    dv_nemo = binary_io.read_array(str(rundir / "ldf_out_dv.bin"), (JPI, JPJ, JPK))

    # Build legoESM's geometry via the SAME production bridge every other
    # #1226 probe uses (not a hand-rolled construction).
    grid = read_nemo_mesh_mask(str(MESH_PATH), nn_hls=0)
    now = read_nemo_restart(str(rundir / RESTART), nn_hls=0)
    cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")

    # Override the restart's velocities with our synthetic field (same
    # face convention the bridge expects: NEMO u at its own EAST face,
    # NEMO v at its own NORTH face, shape (n_lat, n_lon, nlev)).
    # now.u/v are HALO-STRIPPED interior arrays (n_lat, n_lon, nlev) =
    # (199, 52, 36) here; this probe's own u_2d/v_2d are the harness's full
    # (jpi,jpj)=(56,203) i,j-indexed arrays WITH the 2-cell halo -- slice to
    # the interior and transpose (i,j)->(j,i) before feeding the restart's
    # velocity field (same halo/axis convention as harness_dynldf_lap.F90's
    # own docstring: HLS=2 both sides of both axes).
    u_2d_interior_ji = u_2d[HLS:JPI - HLS, HLS:JPJ - HLS].T   # (j,i)
    v_2d_interior_ji = v_2d[HLS:JPI - HLS, HLS:JPJ - HLS].T
    assert u_2d_interior_ji.shape == now.u.shape[:2], (
        f"interior shape mismatch: {u_2d_interior_ji.shape} vs {now.u.shape[:2]}")
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

    du_lego_face, dv_lego_face = nemo_ldf_lap_viscosity_cgrid(
        u_face, v_face, geom, ahmt, ahmf,
        u_mask=u_mask, v_mask=v_mask_arr,
    )

    # legoESM's u-face convention: u_face[:, 0] is a west wall (0), NEMO's u
    # at column i maps to u_face[:, i+1] (see nemo_state_bridge._u_east_to_face,
    # already exercised by the bridge call above). Strip that wall column
    # before comparing against NEMO's own (jpi,jpj,jpk) i-indexed du_nemo.
    # du_lego_face/dv_lego_face are 3-D (n_lat, n_lon[+1], nlev) since
    # br.state.u/v carry all levels -- select K_TEST_LEVEL FIRST (before any
    # transpose) to avoid a 3-D .T silently reversing all three axes instead
    # of swapping just the first two.
    du_lego_2d = du_lego_face[:, 1:, K_TEST_LEVEL]     # (n_lat, n_lon) drop west-wall column
    dv_lego_2d = dv_lego_face[1:, :, K_TEST_LEVEL]     # (n_lat, n_lon) drop south-wall row

    # legoESM axes are (n_lat, n_lon) = NEMO's (j, i), and (unlike the
    # harness's own (jpi,jpj) arrays) ALREADY HALO-FREE interior-only
    # (read_nemo_mesh_mask/read_nemo_restart strip NEMO's nn_hls=2 halo --
    # same convention this campaign's other bridge-based probes rely on).
    # Transpose legoESM (j,i) -> (i,j) to match the harness's i-major
    # convention; NO further slicing needed on this side.
    du_lego_ij = du_lego_2d.T   # (n_lon, n_lat) = interior (i,j)
    dv_lego_ij = dv_lego_2d.T

    import netCDF4 as nc
    with nc.Dataset(MESH_PATH) as ds:
        umask_i = np.asarray(ds.variables["umask"][0, K_TEST_LEVEL], dtype=np.float64)  # (j,i) interior
        vmask_i = np.asarray(ds.variables["vmask"][0, K_TEST_LEVEL], dtype=np.float64)
    umask_ij = umask_i.T   # (i,j) interior, matches du_lego_ij's shape
    vmask_ij = vmask_i.T

    assert du_lego_ij.shape == umask_ij.shape, (
        f"shape mismatch: du_lego_ij {du_lego_ij.shape} vs umask_ij {umask_ij.shape}")

    # du_nemo is the harness's FULL (jpi,jpj,jpk) array WITH halo -- slice to
    # the SAME interior window before comparing against legoESM's already-
    # interior array.
    sl = (slice(HLS, JPI - HLS), slice(HLS, JPJ - HLS))
    wet_u = umask_ij > 0.5
    wet_v = vmask_ij > 0.5

    a_u = du_lego_ij[wet_u]
    b_u = du_nemo[sl + (K_TEST_LEVEL,)][wet_u]
    a_v = dv_lego_ij[wet_v]
    b_v = dv_nemo[sl + (K_TEST_LEVEL,)][wet_v]

    d_u = a_u - b_u
    d_v = a_v - b_v
    print(f"n wet-interior u-cells compared (k={K_TEST_LEVEL}): {wet_u.sum()} of {wet_u.size}")
    print(f"du: max|diff|={np.max(np.abs(d_u)):.4e}  rms|diff|={np.sqrt(np.mean(d_u**2)):.4e}")
    corr_u = np.corrcoef(a_u, b_u)[0, 1]
    ratio_u = np.sqrt(np.mean(a_u**2)) / np.sqrt(np.mean(b_u**2))
    print(f"du corr={corr_u:.6f} ratio={ratio_u:.6f}")

    print(f"n wet-interior v-cells compared (k={K_TEST_LEVEL}): {wet_v.sum()} of {wet_v.size}")
    print(f"dv: max|diff|={np.max(np.abs(d_v)):.4e}  rms|diff|={np.sqrt(np.mean(d_v**2)):.4e}")
    corr_v = np.corrcoef(a_v, b_v)[0, 1]
    ratio_v = np.sqrt(np.mean(a_v**2)) / np.sqrt(np.mean(b_v**2))
    print(f"dv corr={corr_v:.6f} ratio={ratio_v:.6f}")

    # Self-check 1: an independent-level consistency check on the NEMO side
    # itself -- with a horizontally level-independent input velocity and
    # (away from the surface/bottom) near-uniform e3, du_nemo at k=2 and k=3
    # should closely agree; a large disagreement would mean the "single
    # level, e3 varies least" premise is wrong for this pair of levels.
    b_u_k3 = du_nemo[sl + (K_TEST_LEVEL + 1,)][wet_u]
    if b_u_k3.shape == b_u.shape:
        level_consistency = np.max(np.abs(b_u_k3 - b_u))
        print(f"self-check: NEMO du(k={K_TEST_LEVEL}) vs du(k={K_TEST_LEVEL+1}) max|diff|="
              f"{level_consistency:.4e} (should be small if e3 is near-uniform there)")

    print("For reference (NOT reproduced, independent probe, see docstring): "
          "fidelity_bar_gate.py 'dyn_ldf (dynldf_lev_lap) u' row = (0.997855, 1.003865), "
          "'... v' row = (0.999400, 1.001755)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
