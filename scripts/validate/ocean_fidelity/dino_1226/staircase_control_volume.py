#!/usr/bin/env python
"""#1455 -- the implicit vertical solve's u-face column depth at staircase faces.

Residual candidate #3 of ``docs/ocean/fidelity/dino_wall_ldf_alignment.md``,
raised by physics review and MEASURED here rather than left named.

THE MECHANISM, read from the current code (not from prose):

  * ``ocean/vertical.py:621``  the partial-cell thickness is ZEROED below the
    sea floor: ``h_partial = jnp.where(is_active, h_full, 0.0)``.
  * ``ocean_model_latlon_cgrid.py:6589``  the implicit vertical momentum solve
    builds its u-face thickness as an ARITHMETIC mean of that already-masked
    field: ``dz_u = interp_cell_to_uface(dz_cell)``.  At a face whose two
    columns have different bottom levels, one side contributes a full
    thickness and the other a zero, so the face gets HALF a level that exists
    on neither column.
  * ``ocean_model_latlon_cgrid.py:6628``  the mask applied to that solve is the
    2-D surface face mask broadcast down, so the spurious half-level is not
    removed afterwards.

NEMO does it in the opposite order: ``e3u_0`` is built on the unmasked
reference ladder and ``umask`` removes the level, so a face whose two columns
disagree simply has no water at that level.

THE MEASUREMENT.  Compare legoESM's column depth ``sum_k dz_u`` against NEMO's
own ``hu_0 = sum_k e3u_0 * umask`` from the mesh file, per row, over wet
u-columns only.  A nonzero difference at a staircase face IS the defect, and
its size is directly the divisor bias the implicit solve carries.  Zero retires
the candidate permanently.

WHY THE WALL ROWS.  The southern wall rows are a bathymetric staircase (30-31
wet levels at row 1 rising to 34 by row 4, against 35 in the interior) AND they
are shallower (about 2260 m against 4000 m), so a fixed absolute bias is a
larger relative one there.  The measured defect must be reported BOTH ways --
absolute and relative -- because only the relative one can be compared against
the 34%-on-four-rows fingerprint.

FIXED by PR #1642 (merged to main 2026-08-22, ported onto this branch): the
average is now multiplied by the both-cells-wet face mask, so the live model
reproduces ``hu_0`` exactly on every wet u-column.  This probe reports BOTH
rules every run and reads which one the model actually uses from the model's
own source, so it keeps measuring the defect after the repair rather than
becoming a tautology.

This probe prints numbers and never prints a verdict.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      .venv/bin/python -m \
      scripts.validate.ocean_fidelity.dino_1226.staircase_control_volume
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import netCDF4 as nc

_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    interp_cell_to_uface)
from legoesm.ocean.experiments.dino import dino_config_for_recipe  # noqa: E402
from legoesm.ocean.fidelity.nemo_io import (  # noqa: E402
    read_nemo_mesh_mask, read_nemo_restart)
from legoesm.ocean.fidelity.nemo_state_bridge import (  # noqa: E402
    bridge_nemo_to_legoesm_topo)

DINO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")
MESH = DINO / "RUN_TRAJ" / "mesh_mask.nc"
RESTART = DINO / "RUN_STEPDUMP" / "DINO_00005760_restart.nc"
WALL_ROWS = [1, 2, 3, 4]


def main() -> int:
    set_policy(PrecisionPolicy.fp64())
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  dirty_tracked={len(dirt.splitlines())}")
    print(f"PROVENANCE  mesh={MESH}  restart={RESTART}")
    print(f"PROVENANCE  JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}")

    g = read_nemo_mesh_mask(str(MESH), nn_hls=0)
    s = read_nemo_restart(str(RESTART), nn_hls=0)
    cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    # e3t_mode="both" is the ladder the twin actually runs
    # (kamm_twin_90d.py resolve_ladder_mode default, stamped in every arm's log
    # as LEGOESM_NEMO_E3T=both). Omitting it silently builds the 1-D reference
    # ladder instead, which differs from NEMO's own thicknesses and would show
    # up here as a bias on FLAT columns -- the control below catches exactly
    # that, and did.
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True,
                                     omega=cfg.omega, e3t_mode="both")
    zc = br.z_coord
    act = np.asarray(zc.is_active)
    # The SAME expression ocean/vertical.py:621 builds, on the same coordinate.
    dz_cell = np.where(act, np.broadcast_to(np.asarray(zc.dz_ref), act.shape),
                       0.0).astype(np.float64)
    # BOTH rules, so this probe reports the defect AND its repair in one run.
    #   unmasked mean : the arithmetic cell->face mean the implicit solve used
    #                    before PR #1642 -- the defect.
    #   masked mean   : the same average multiplied by the both-cells-wet face
    #                   mask, which is what the solve uses now (PR #1642,
    #                   merged to main).  Which one the shipped model runs is
    #                   read from the model source below, so this probe cannot
    #                   drift away from it.
    dz_u_mean = np.asarray(interp_cell_to_uface(dz_cell), dtype=np.float64)
    act_u = np.minimum(np.roll(act, 1, axis=1), act).astype(np.float64)
    act_u = np.concatenate([act_u, act_u[:, 0:1]], axis=1)
    dz_u_masked = dz_u_mean * act_u
    import inspect

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    _src = inspect.getsource(
        LatLonCGridOceanModel._apply_implicit_vertical_mixing)
    _live_is_masked = "dz_u = dz_u * _act_u3.astype(dz_u.dtype)" in _src
    print(f"LIVE MODEL RULE  face control volume is "
          f"{'the MASKED average (PR #1642)' if _live_is_masked else 'the UNMASKED average (pre-#1642 defect)'}"
          f"  (read from _apply_implicit_vertical_mixing's source)")
    dz_u = dz_u_masked if _live_is_masked else dz_u_mean
    # legoESM's u-face array carries a west-wall column at index 0; NEMO's u
    # column i is index i+1 (nemo_state_bridge._u_east_to_face).
    H_lego = dz_u.sum(axis=-1)[:, 1:]
    H_mean = dz_u_mean.sum(axis=-1)[:, 1:]
    H_min = dz_u_masked.sum(axis=-1)[:, 1:]

    with nc.Dataset(MESH) as ds:
        e3u0 = np.asarray(ds["e3u_0"][0], dtype=np.float64)   # (k,j,i)
        umask = np.asarray(ds["umask"][0], dtype=np.float64)
        gphit = np.asarray(ds["gphit"][0], dtype=np.float64)
    H_nemo = (e3u0 * umask).sum(axis=0)
    wet = umask.max(axis=0) > 0
    if H_lego.shape != H_nemo.shape:
        raise SystemExit(f"shape mismatch {H_lego.shape} vs {H_nemo.shape}")

    d = np.where(wet, H_lego - H_nemo, 0.0)
    d_mean = np.where(wet, H_mean - H_nemo, 0.0)
    d_min = np.where(wet, H_min - H_nemo, 0.0)
    print(f"\nBOTH RULES vs NEMO hu_0 = sum_k e3u_0*umask, over "
          f"{int(wet.sum())} wet u-columns:")
    print(f"  arithmetic mean : max |diff| = {np.abs(d_mean).max():.4f} m, "
          f"columns off by >1e-9 m = {int((np.abs(d_mean) > 1e-9).sum())}")
    print(f"  masked average  : max |diff| = {np.abs(d_min).max():.4e} m, "
          f"columns off by >1e-9 m = {int((np.abs(d_min) > 1e-9).sum())}")
    print(f"\nu-columns compared: {int(wet.sum())}")
    print(f"max |sum_k dz_u - hu_0| anywhere: {np.abs(d).max():.4f} m")
    print(f"columns differing by more than 1e-9 m: "
          f"{int((np.abs(d) > 1e-9).sum())} of {int(wet.sum())}")

    print(f"\n{'row':>4}{'lat':>9}{'H_lego':>10}{'H_nemo':>10}"
          f"{'mean bias':>11}{'rel':>9}{'max bias':>10}")
    rel_wall, rel_int = [], []
    for j in list(WALL_ROWS) + [6, 8, 12, 20, 40, 99, 150]:
        w = wet[j]
        if not w.any():
            continue
        hl, hn = H_lego[j][w].mean(), H_nemo[j][w].mean()
        rel = (hl - hn) / hn
        tag = "  WALL" if j in WALL_ROWS else ""
        print(f"{j:>4}{gphit[j, 25]:>9.2f}{hl:>10.1f}{hn:>10.1f}"
              f"{hl - hn:>11.2f}{rel:>9.4f}{np.abs(d[j][w]).max():>10.2f}{tag}")
        (rel_wall if j in WALL_ROWS else rel_int).append(rel)
    print(f"\nmean relative bias, four wall rows: {np.mean(rel_wall):.4f}")
    print(f"mean relative bias, sampled interior rows: {np.mean(rel_int):.4f}")
    _rw, _ri = float(np.mean(rel_wall)), float(np.mean(rel_int))
    if abs(_ri) < 1e-12:
        print("wall / interior enrichment: UNDEFINED (interior bias is zero "
              "-- the rule under test reproduces hu_0 exactly)")
    else:
        print(f"wall / interior enrichment: {_rw / _ri:.2f}x")
    if not np.isfinite([_rw, _ri]).all():
        raise SystemExit("non-finite relative bias -- fatal")

    # CONTROL. The comparison must be able to return zero. A column with a FLAT
    # bottom has no staircase face, so the two rules must agree there exactly;
    # if they disagree even on flat columns, the difference above is not the
    # staircase mechanism but a convention offset.
    nlev = act.sum(axis=-1)
    flat = wet.copy()
    flat[:, :] = False
    # NEMO's u column i is the EAST face of T-cell i, so its two columns are
    # cells i and i+1 -- NOT i-1 and i. Getting that backwards makes the
    # control test the wrong faces, which is how it first went red.
    for j in range(1, act.shape[0] - 1):
        row = nlev[j]
        for i in range(act.shape[1] - 1):
            if wet[j, i] and row[i] == row[i + 1]:
                flat[j, i] = True
    fd = np.abs(d[flat])
    print(f"\nCONTROL, u-faces whose two columns have the SAME bottom level "
          f"({int(flat.sum())} of them): max |difference| = {fd.max():.3e} m")
    assert fd.max() < 1e-9, (
        f"flat-bottom faces already differ by {fd.max():.3e} m -- the bias "
        "measured above is not the staircase mechanism but a convention "
        "offset, and the reading must not be attributed to staircases")
    print("  (the control is zero on flat faces under EITHER rule; the "
          "staircase faces are where the two rules can differ, and the "
          "'BOTH RULES' block above says by how much)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
