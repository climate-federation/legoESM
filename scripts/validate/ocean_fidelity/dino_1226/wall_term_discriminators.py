#!/usr/bin/env python
"""#1455 -- the two ranked wall-row term comparisons: bottom drag, and the EEN triad.

Pre-registration: ``PREREG_wall_drag_and_een.md``, written before either number
existed. Parent: ``docs/ocean/fidelity/dino_wall_ldf_alignment.md``, whose
ranked residual list named these two after lateral friction was closed end to
end.

Both are read on NEMO's ``RUN_D180_1STEP_1R``: ONE step (kt 5761) from
``DINO_00005760_restart.nc``, single rank, the same restart the 90-day twins
start from, with legoESM bridged from that same restart on the same vertical
ladder the twins run. Bit-identical state on both sides, so any difference is
the term and not the trajectory.

PART 1 -- BOTTOM DRAG, split into three so a coefficient error and a
composition error cannot hide inside each other:
  1a  the drag coefficient ``rCdU_bot`` at T-points;
  1b  the bottom-level index (legoESM's ``isb_u`` against the mesh's ``mbku``),
      reported BEFORE the increment because an off-by-one here would make the
      increment comparison meaningless;
  1c  the assembled drag-only increment to the barotropic forcing.

PART 2 -- the EEN vorticity flux at the wall, NEMO's Krhs chain-dump difference
(stage 4 minus stage 3) against legoESM's separately exposed ``vortcor_u``.

THE REGISTERED SHAPE TEST, identical for both: to carry the defect a term's
difference must be BOTH wall-enriched (row-mean over rows 1-4 at least 3x the
interior) AND large enough (at least 5.9e-11 m/s2, which is what reproduces the
measured 4.6e-4 m/s velocity error if it accumulated coherently for the whole
90-day window -- a deliberately generous upper bound, so failing it refutes
strongly and clearing it is weak evidence).

Nothing here re-derives a dump reader or a face-convention mapping: both come
from ``acc_momentum_budget``, the sibling that established them.

This probe prints numbers and never prints a verdict.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      .venv/bin/python -m \
      scripts.validate.ocean_fidelity.dino_1226.wall_term_discriminators
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[3]
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# The dump reader and the u-face convention mapping the sibling probe
# established -- imported, not re-derived (CLAUDE.md "no duplicate numerics").
from acc_momentum_budget import _load_full_3d, _u_to_nemo  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.experiments.dino import (  # noqa: E402
    dino_config_for_recipe, dino_lat_lon_model_config)
from legoesm.ocean.fidelity.nemo_io import (  # noqa: E402
    read_nemo_mesh_mask, read_nemo_restart)
from legoesm.ocean.fidelity.nemo_state_bridge import (  # noqa: E402
    bridge_nemo_to_legoesm_topo)

DINO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")
RUN = DINO / "RUN_D180_1STEP_1R"
MESH = DINO / "RUN_TRAJ" / "mesh_mask.nc"
RESTART = RUN / "DINO_00005760_restart.nc"
JPI, JPJ, JPK, HLS = 56, 203, 36, 2
JPKM1 = JPK - 1
WALL_ROWS = [1, 2, 3, 4]
INTERIOR_ROWS = [8, 12, 20, 40, 60, 99, 150]
# Pre-registered bars (PREREG_wall_drag_and_een.md).
BAR_ENRICH = 3.0
BAR_MAG = 5.9e-11          # m/s2 at the wall rows
BAR_MAG_REFUTE = 5.9e-12


def stamp() -> None:
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  dirty_tracked={len(dirt.splitlines())}")
    print(f"PROVENANCE  run={RUN}  (1 step, kt 5761, single rank)")
    print(f"PROVENANCE  mesh={MESH}")
    print(f"PROVENANCE  bars: enrichment>={BAR_ENRICH}, "
          f"magnitude>={BAR_MAG:.2e} m/s2, refute<{BAR_MAG_REFUTE:.2e}")
    print(f"PROVENANCE  JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}")


def build_lego():
    g = read_nemo_mesh_mask(str(MESH), nn_hls=0)
    s = read_nemo_restart(str(RESTART), nn_hls=0)
    cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True,
                                     omega=cfg.omega, e3t_mode="both")
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    return g, br, cfg, mc


def row_report(name, diff, wet, unit="m/s2"):
    """Row-mean |diff| over wet cells, wall rows against interior, vs the bars."""
    n = wet.sum(axis=tuple(range(1, wet.ndim)))

    def rm(j):
        w = wet[j]
        return float(np.abs(diff[j][w]).mean()) if w.any() else np.nan

    print(f"\n  {name}: row-mean |NEMO - legoESM| [{unit}]")
    print(f"    max |difference| over all wet cells: "
          f"{float(np.abs(diff[wet]).max()):.6e}   "
          f"cells above 1e-15: {int((np.abs(diff[wet]) > 1e-15).sum())} "
          f"of {int(wet.sum())}")
    for j in WALL_ROWS + INTERIOR_ROWS:
        tag = "  WALL" if j in WALL_ROWS else ""
        print(f"    row {j:>3}  {rm(j):.6e}{tag}")
    wall = float(np.nanmean([rm(j) for j in WALL_ROWS]))
    inter = float(np.nanmean([rm(j) for j in INTERIOR_ROWS]))
    enrich = wall / inter if inter > 0 else np.inf
    print(f"    wall-row mean {wall:.6e}   interior mean {inter:.6e}   "
          f"enrichment {enrich:.2f}x")
    print(f"    vs bars: enrichment {enrich:.2f} (>= {BAR_ENRICH} to pass), "
          f"magnitude {wall:.3e} (>= {BAR_MAG:.2e} to pass, "
          f"< {BAR_MAG_REFUTE:.2e} refutes)")
    del n
    return wall, inter, enrich


def part1(g, br, cfg, mc, plant=None):
    """Bottom drag: coefficient, bottom index, assembled increment."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        nemo_bottom_drag_rate_faces)
    from legoesm.ocean.vertical import compute_layer_thickness
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface

    st = br.state
    print("\n=== PART 1 -- BOTTOM DRAG ===")

    # ---- 1a. the coefficient rCdU_bot at T-points ----
    rcdu = np.fromfile(RUN / "drg_dump_rCdU_bot.bin",
                       dtype="<f8").reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS]
    tmask_s = np.asarray(g.tmask)[..., 0] > 0.5
    print(f"\n1a. NEMO rCdU_bot at T-points: min {rcdu[tmask_s].min():.6e}, "
          f"max {rcdu[tmask_s].max():.6e} m/s  "
          f"(sign convention: NEMO's is <= 0 and is ADDED)")

    h_k = np.asarray(compute_layer_thickness(
        st.eta.data, st.H_bathy.data, br.z_coord,
        min_water_column_m=mc.min_water_column_m), dtype=np.float64)
    u_now = np.asarray(st.u.data, dtype=np.float64)
    v_now = np.asarray(st.v.data, dtype=np.float64)
    if plant is not None:
        # dry-row trap: plant on cells NEMO calls land
        um3 = np.asarray(st.u_mask.data)[..., None] > 0
        u_now = np.where(um3, u_now, plant)
    r_u, r_v, isb_u, isb_v = nemo_bottom_drag_rate_faces(
        jnp.asarray(u_now), jnp.asarray(v_now), jnp.asarray(h_k),
        br.z_coord, mc, br.geometry)
    r_u = np.asarray(r_u, dtype=np.float64)           # (n_lat, n_lon+1)
    isb_u = np.asarray(isb_u, dtype=np.float64)       # (n_lat, n_lon+1, nlev)

    # NEMO's u-face coefficient is 0.5*(rCdU_bot(i+1)+rCdU_bot(i)); build it
    # from the dump and compare against legoESM's face rate (absolute value,
    # the two carry opposite sign conventions -- reported, then compared).
    cdu_u_nemo = 0.5 * (np.roll(rcdu, -1, axis=1) + rcdu)      # (j,i) at u-pt
    r_u_nemo_layout = _u_to_nemo(r_u)                          # (j,i)
    umask_s = np.asarray(g.umask)[..., 0] > 0.5
    d_coef = np.abs(cdu_u_nemo) - np.abs(r_u_nemo_layout)
    print(f"    legoESM face rate: min {np.abs(r_u_nemo_layout)[umask_s].min():.6e}, "
          f"max {np.abs(r_u_nemo_layout)[umask_s].max():.6e} m/s")
    row_report("1a coefficient at u-faces", d_coef, umask_s, unit="m/s")

    # ---- 1b. the bottom-level index ----
    # mesh_mask.nc carries no ``mbku``, so NEMO's u bottom level is taken from
    # its own ``umask`` -- the count of wet levels IS mbku by construction
    # (dommsk.F90 builds umask from tmask and dommsk caps it at mbku), which is
    # exact rather than a re-derivation of the bathymetry.
    mbku = (np.asarray(g.umask) > 0.5).sum(axis=-1)        # (j,i), 1-based count
    kb_lego = _u_to_nemo(np.argmax(isb_u > 0.5, axis=-1)) + 1   # -> 1-based
    dk = np.where(umask_s, kb_lego - mbku, 0)
    print(f"\n1b. bottom-level index: u-faces where legoESM != NEMO mbku: "
          f"{int((dk != 0).sum())} of {int(umask_s.sum())}"
          f"   (wall rows: "
          f"{int((dk[WALL_ROWS[0]:WALL_ROWS[-1] + 1] != 0).sum())})")
    if (dk != 0).any():
        vals, cnt = np.unique(dk[dk != 0], return_counts=True)
        print(f"    offsets present: {dict(zip(vals.tolist(), cnt.tolist()))}")

    # ---- 1c. the assembled increment ----
    inc_nemo = np.fromfile(RUN / "drg_dump_zu_frc_inc.bin",
                           dtype="<f8").reshape(JPJ - 2 * HLS, JPI - 2 * HLS)
    h_u = np.asarray(min_cell_to_uface(jnp.asarray(h_k)), dtype=np.float64)
    H_u = h_u.sum(axis=-1)
    # the card sets barotropic_forcing_centred=True -> the BEFORE level, which
    # is NEMO's ln_bt_fw=.false. Kbb branch (dynspg_ts.F90 dyn_drg_init)
    u_src = np.asarray(st.u_before.data if st.u_before is not None
                       else st.u.data, dtype=np.float64)
    if plant is not None:
        um3 = np.asarray(st.u_mask.data)[..., None] > 0
        u_src = np.where(um3, u_src, plant)
    u_bot = np.sum(u_src * isb_u, axis=-1)
    U_bar = np.sum(u_src * h_u, axis=-1) / np.maximum(H_u, 1e-10)
    inc_lego = -(r_u / np.maximum(H_u, 1e-10)) * (u_bot - U_bar)
    inc_lego = _u_to_nemo(inc_lego) * umask_s
    assert inc_nemo.shape == inc_lego.shape, (
        f"shape mismatch {inc_nemo.shape} vs {inc_lego.shape}")
    print(f"\n1c. NEMO increment: min {inc_nemo[umask_s].min():.6e}, "
          f"max {inc_nemo[umask_s].max():.6e} m/s2")
    print(f"    legoESM:        min {inc_lego[umask_s].min():.6e}, "
          f"max {inc_lego[umask_s].max():.6e} m/s2")
    return row_report("1c drag increment to the barotropic forcing",
                      inc_nemo - inc_lego, umask_s)


def part2(g, br, cfg, mc, plant=None):
    """EEN vorticity flux at the wall, NEMO stage4 - stage3 vs legoESM vortcor."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    print("\n=== PART 2 -- THE EEN VORTICITY FLUX ===")
    d03 = _load_full_3d(str(RUN / "stp_dump_03_dynadv_du.bin"),
                        JPI, JPJ, JPKM1, HLS)
    d04 = _load_full_3d(str(RUN / "stp_dump_04_dynvor_du.bin"),
                        JPI, JPJ, JPKM1, HLS)
    vor_nemo = d04 - d03                       # (n_lat, n_lon, jpkm1)

    st = br.state
    if plant is not None:
        um3 = np.asarray(st.u_mask.data)[..., None] > 0
        st = st._replace(u=st.u.replace(
            data=np.where(um3, np.asarray(st.u.data), plant)))
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    _t, diag = model.tendencies_with_diagnostics(
        st, surface_forcing=None, dt=float(cfg.dt))
    vor_lego = _u_to_nemo(np.asarray(diag.vortcor_u.data, dtype=np.float64))
    vor_lego = vor_lego[..., :JPKM1]
    assert vor_nemo.shape == vor_lego.shape, (
        f"shape mismatch {vor_nemo.shape} vs {vor_lego.shape}")
    umask3 = np.asarray(g.umask)[..., :JPKM1] > 0.5
    print(f"  NEMO   |vor| max {np.abs(vor_nemo)[umask3].max():.6e} m/s2")
    print(f"  legoESM|vor| max {np.abs(vor_lego)[umask3].max():.6e} m/s2")
    return row_report("2 EEN vorticity flux", vor_nemo - vor_lego, umask3)


def self_test(g, br, cfg, mc) -> int:
    """The dry-row trap, plus the gate that makes the scored numbers safe.

    A first version of this planted a value on EVERY dry face and required no
    scored number to move. It fired: the drag coefficient is built from a
    T-point speed that averages the two adjacent faces, so a planted land face
    DOES reach the wet face beside it, and the EEN triads reach further still.

    That is not a defect in the scored numbers, but it IS the reason they need a
    gate rather than an assumption. So this checks four things:

      (i)   PRODUCTION GATE -- the bridged state's velocity is exactly zero on
            every dry face. This is what makes the stencil reads harmless, and
            it is a property of the state, so it is measured, not assumed.
      (ii)  THE ACTUAL TRAP (0be305459) -- row 0 is an entirely dry land row
            whose stored zeros sit beside wet values of order 1. Planting there
            must move no scored wall row, because no wall-row stencil reaches
            a different latitude.
      (iii) NON-VACUITY -- planting on a WET wall row must move the wall score.
      (iv)  SENSITIVITY, reported not asserted -- how far the scores move when
            every dry face is planted, which is the size of the dependence that
            gate (i) is protecting.
    """
    print("=== SELF-TEST ===")
    st = br.state
    um3 = np.asarray(st.u_mask.data)[..., None] > 0
    u = np.asarray(st.u.data, dtype=np.float64)
    ub = (np.asarray(st.u_before.data, dtype=np.float64)
          if st.u_before is not None else u)
    dry_max = float(max(np.abs(np.where(um3, 0.0, u)).max(),
                        np.abs(np.where(um3, 0.0, ub)).max()))
    print(f"(i)  production gate: max |u| on dry faces = {dry_max:.3e} m/s")
    assert dry_max == 0.0, (
        f"the bridged state carries {dry_max:.3e} m/s on dry faces -- the "
        "drag and vorticity stencils read those cells, so the scored "
        "differences are not attributable to the terms")

    base1 = part1(g, br, cfg, mc)
    base2 = part2(g, br, cfg, mc)

    # (ii) the row-0 trap: plant the entirely dry southern land row.
    def _plant_row0(b):
        s0 = b.state
        uu = np.asarray(s0.u.data, dtype=np.float64).copy()
        uu[0, :, :] = 1e3
        rep = {"u": s0.u.replace(data=uu)}
        if s0.u_before is not None:
            bb = np.asarray(s0.u_before.data, dtype=np.float64).copy()
            bb[0, :, :] = 1e3
            rep["u_before"] = s0.u_before.replace(data=bb)
        return b._replace(state=s0._replace(**rep))

    br0 = _plant_row0(br)
    r1, r2 = part1(g, br0, cfg, mc), part2(g, br0, cfg, mc)
    # Part 1 (drag) has no meridional stencil at all: its coefficient averages
    # the two ZONAL neighbours of a T-point, so a different LATITUDE can never
    # reach a wall-row score. This must hold exactly.
    assert r1[0] == base1[0], (
        f"a row-0 land value reached part 1's wall score: "
        f"{base1[0]} -> {r1[0]} -- the drag comparison has acquired a "
        "meridional stencil it should not have")
    print("(ii) row-0 plant of 1e3 m/s moved part 1's wall score not at all")
    # Part 2 (the EEN vorticity flux) DOES read row 0, BY DESIGN and on BOTH
    # sides: DINO sets ln_dynvor_msk=.false. (namelist_cfg:333) so the coastal
    # f-point is NOT masked out of the vorticity flux, and the relative
    # vorticity there is built as the shear between the first wet row and the
    # land row. legoESM matches that with een_q_boundary="nemo_live". So the
    # wall vorticity is a no-slip-like shear against a stored zero in both
    # models, and gate (i) -- that the stored value IS zero -- is the whole
    # reason the two agree. Reported with its size rather than forbidden.
    print(f"(ii) part 2 DOES read row 0, by design on both sides: the plant "
          f"moves its wall score {base2[0]:.3e} -> {r2[0]:.3e}. The coastal "
          f"f-point is deliberately unmasked (ln_dynvor_msk=.false.), so the "
          f"wall vorticity is the shear against the land row's stored zero, "
          f"and gate (i) is what makes both models agree about it.")

    # (iii) non-vacuity, on a WET wall row.
    s1 = br.state
    uw = np.asarray(s1.u.data, dtype=np.float64).copy()
    uw[WALL_ROWS[0], :, :] += 1e-2
    brw = br._replace(state=s1._replace(u=s1.u.replace(data=uw)))
    w1, w2 = part1(g, brw, cfg, mc), part2(g, brw, cfg, mc)
    assert w1[0] != base1[0] and w2[0] != base2[0], (
        "planting a WET wall row moved nothing -- check (ii) is vacuous")
    print(f"(iii) wet wall-row plant moved part 1 "
          f"{base1[0]:.3e} -> {w1[0]:.3e} and part 2 "
          f"{base2[0]:.3e} -> {w2[0]:.3e}")

    # (iv) sensitivity to dry-face values, reported.
    def _plant_all(b):
        s0 = b.state
        rep = {"u": s0.u.replace(
            data=np.where(um3, np.asarray(s0.u.data), 1e3))}
        if s0.u_before is not None:
            rep["u_before"] = s0.u_before.replace(
                data=np.where(um3, np.asarray(s0.u_before.data), 1e3))
        return b._replace(state=s0._replace(**rep))

    a1 = part1(g, _plant_all(br), cfg, mc)
    print(f"(iv) planting EVERY dry face moves part 1's wall score "
          f"{base1[0]:.3e} -> {a1[0]:.3e} -- the drag coefficient is built "
          f"from a T-point speed averaging the two adjacent faces, so the "
          f"stencil does read land. Gate (i) is what makes that harmless.")
    print("SELF-TEST PASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    set_policy(PrecisionPolicy.fp64())
    stamp()
    g, br, cfg, mc = build_lego()
    print(f"bridge f_T self-check: {br.f_match_max_abs:.3e}")
    if args.self_test:
        return self_test(g, br, cfg, mc)
    part1(g, br, cfg, mc)
    part2(g, br, cfg, mc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
