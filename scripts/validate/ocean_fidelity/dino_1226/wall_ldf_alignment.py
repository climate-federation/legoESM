#!/usr/bin/env python
"""#1455 -- the free-slip southern wall in the lateral-momentum-viscosity operator.

Numeric backing for the line-by-line alignment table in
``docs/ocean/fidelity/dino_wall_ldf_alignment.md``.  Answers, from data rather
than from reading, three questions about how NEMO and legoESM meet DINO's
southern free-slip wall in ``dyn_ldf``:

A. **the coefficient and the masks.**  NEMO folds ``fmask`` into ``ahmf`` once
   at init (``ldfdyn.F90:330``) and ``tmask`` into ``ahmt``; legoESM keeps a
   pure geometric ``ahmt``/``ahmf`` and supplies the same masking through the
   operator's ``mask=``/``vertex_mask=`` kwargs
   (``ocean_pe_latlon_cgrid.py:2752-2769``).  This part compares the two
   EFFECTIVE coefficient fields cell by cell, on every level, with the wall
   rows called out separately.

B. **the viscous flux through the wall face.**  Free slip means the wall
   f-point carries no stress.  This part evaluates the operator's own wall
   contribution on both sides and requires it to be exactly zero.

C. **the mask dimensionality.**  NEMO masks this operator in 3-D on every leg;
   legoESM's production call hands it the 2-D cell and face masks
   (``ocean_pe_latlon_cgrid.py:4246-4247``).  This part applies both maskings to
   the same state, WITH a control that plants a velocity on the faces below the
   sea floor -- without that control the answer is zero for the trivial reason
   that the state is already zero there, which is a fact about the state and
   not about the operator.

D. **the e3 (layer-thickness) weighting**, the other known transcription
   difference: NEMO weights the divergence and the vorticity by the layer
   thickness (``dynldf_lev_rot_scheme.h90:23,27-29,41,51``) and the shipped card
   does not (``lateral_viscosity_e3_weighting="off"``).  This part applies BOTH
   treatments to the SAME bridged state and reports the per-row difference in
   the resulting viscous tendency, so the difference can be compared against the
   measured wall fingerprint (a 34% amplitude error on four rows) instead of
   argued about.

Both models are fed the SAME state -- NEMO's own day-180 restart, bridged with
the production bridge every sibling probe uses -- so nothing here depends on the
two trajectories having stayed together.

THE DRY-ROW TRAP (0be305459).  DINO's row 0 is entirely dry and both models
store exact zeros there next to wet values of order 1.  A stencil that reaches
into it silently returns a land value.  ``--self-test`` plants a large value on
the dry row and on the dry columns and requires every scored quantity to be
unchanged; it also plants a value that SHOULD move a scored quantity, so the
test is not vacuous in either direction.

This probe prints numbers and never prints a verdict.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
      JAX_ENABLE_X64=1 .venv/bin/python \
      scripts/validate/ocean_fidelity/dino_1226/wall_ldf_alignment.py
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import netCDF4 as nc  # noqa: E402

from legoesm.ocean.experiments.dino import dino_config_for_recipe  # noqa: E402
from legoesm.ocean.fidelity.nemo_io import (  # noqa: E402
    read_nemo_mesh_mask,
    read_nemo_restart,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (  # noqa: E402
    bridge_nemo_to_legoesm_topo,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    compute_face_masks_3d,
    nemo_lateral_viscosity_coefficients,
    nemo_ldf_lap_viscosity_cgrid,
    nemo_ldf_lap_viscosity_e3_cgrid,
)
from legoesm.ocean.vertical import compute_layer_thickness  # noqa: E402
from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

DINO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")
MESH = DINO / "RUN_TRAJ" / "mesh_mask.nc"
RESTART = DINO / "RUN_STEPDUMP" / "DINO_00005760_restart.nc"

# DINO namelist_cfg (&namdyn_ldf): ln_dynldf_lap, ln_dynldf_lev,
# nn_ahm_ijk_t=20, rn_Uv=0.27; &namlbc: rn_shlat=0 (free slip).
RN_UV = 0.27
# Cell rows 1..4 are the four wall rows 69.50S..68.43S the parent document's
# row decomposition names; row 0 is entirely dry (verified below, not assumed).
WALL_ROWS = (1, 2, 3, 4)


def stamp() -> None:
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  dirty_tracked={len(dirt.splitlines())}")
    print(f"PROVENANCE  mesh={MESH}")
    print(f"PROVENANCE  restart={RESTART}")
    print(f"PROVENANCE  rn_Uv={RN_UV}  rn_shlat=0  nn_ahm_ijk_t=20  "
          f"ln_dynldf_lap=T ln_dynldf_lev=T nn_dynldf_typ=0(div-rot)")
    print(f"PROVENANCE  JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}  "
          f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}")


def load_nemo():
    """NEMO's own masks and horizontal metrics, halo-stripped, as (j,i)."""
    with nc.Dataset(MESH) as ds:
        out = {
            "tmask": np.asarray(ds["tmask"][0], dtype=np.float64),   # (k,j,i)
            "fmask": np.asarray(ds["fmask"][0], dtype=np.float64),
            "umask": np.asarray(ds["umask"][0], dtype=np.float64),
            "vmask": np.asarray(ds["vmask"][0], dtype=np.float64),
            "e1t": np.asarray(ds["e1t"][0], dtype=np.float64),       # (j,i)
            "e2t": np.asarray(ds["e2t"][0], dtype=np.float64),
            "e1f": np.asarray(ds["e1f"][0], dtype=np.float64),
            "e2f": np.asarray(ds["e2f"][0], dtype=np.float64),
            "gphit": np.asarray(ds["gphit"][0], dtype=np.float64),
        }
    return out


def build_lego(plant_dry: float = 0.0):
    """Bridge NEMO's restart into legoESM and return everything the operator reads.

    ``plant_dry``: value written into the DRY cells of the velocity field
    before the operator runs (the dry-row trap self-test).  0.0 = untouched.
    """
    g = read_nemo_mesh_mask(str(MESH), nn_hls=0)
    s = read_nemo_restart(str(RESTART), nn_hls=0)
    cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True,
                                     omega=cfg.omega)
    geom = br.geometry
    u = np.asarray(br.state.u.data, dtype=np.float64)
    v = np.asarray(br.state.v.data, dtype=np.float64)
    u_mask = np.asarray(br.state.u_mask.data, dtype=np.float64)
    v_mask = np.asarray(br.state.v_mask.data, dtype=np.float64)
    cell_mask = np.asarray(br.state.land_mask.data, dtype=np.float64)
    h_k = np.asarray(br.state.h.data, dtype=np.float64) if hasattr(
        br.state, "h") else None
    if plant_dry:
        # Plant on the DRY faces only. Production always multiplies u/v by the
        # face mask on entry, so a correct operator must not see this at all.
        um3 = u_mask[..., None] if u_mask.ndim == 2 else u_mask
        vm3 = v_mask[..., None] if v_mask.ndim == 2 else v_mask
        u = np.where(um3 > 0.0, u, plant_dry)
        v = np.where(vm3 > 0.0, v, plant_dry)
    return br, geom, cfg, u, v, u_mask, v_mask, cell_mask, h_k


def effective_coefficients(br, geom, cell_mask):
    """legoESM's EFFECTIVE ahmt/ahmf (coefficient x mask), the NEMO analogue.

    Reproduces the production wiring at ``ocean_pe_latlon_cgrid.py:2752-2769``
    (the 3-D staircase vertex mask built from ``z_coord.is_active``), not a
    re-derivation of it.
    """
    A_h_base = 0.5 * RN_UV * geom.radius * geom.dlon
    half_UM = A_h_base / (geom.radius * geom.dlon)
    ahmt, ahmf = nemo_lateral_viscosity_coefficients(geom, half_UM)
    ahmt = np.asarray(ahmt, dtype=np.float64)     # (n_lat,)
    ahmf = np.asarray(ahmf, dtype=np.float64)     # (n_lat+1,)

    vertex_mask = np.asarray(compute_vertex_mask(cell_mask, grid=geom),
                             dtype=np.float64)     # (n_lat+1, n_lon+1)
    act = getattr(br.z_coord, "is_active", None)
    if act is not None:
        cell3 = np.asarray(act, dtype=np.float64) * cell_mask[..., None]
        import jax
        vm3 = np.asarray(jax.vmap(
            lambda m2: compute_vertex_mask(m2, grid=geom),
            in_axes=-1, out_axes=-1)(cell3), dtype=np.float64)
        vm3 = vm3 * vertex_mask[..., None]
    else:
        vm3 = None
    return half_UM, ahmt, ahmf, vertex_mask, vm3


def part_a(nemo, geom, half_UM, ahmt, ahmf, vertex_mask, vm3, cell_mask):
    """Effective coefficient and mask comparison, NEMO vs legoESM."""
    tmask, fmask = nemo["tmask"], nemo["fmask"]
    nk, nj, ni = tmask.shape
    print("\n--- A. effective coefficients and masks -----------------------")
    print(f"grid (j,i,k) = ({nj},{ni},{nk});  half_UM = {half_UM!r} "
          f"(NEMO 0.5*rn_Uv = {0.5 * RN_UV!r})")

    # geometry sanity: which rows are dry, and at what latitude
    dry_rows = [j for j in range(nj) if tmask[0, j].sum() == 0]
    print(f"entirely dry cell rows (NEMO tmask k=0): {dry_rows}")
    print("wall rows: " + ", ".join(
        f"j={j} phi={nemo['gphit'][j, ni // 2]:+.3f}deg" for j in WALL_ROWS))

    # NEMO's effective coefficients (ldfdyn.F90:328-330)
    ahmt_nemo = 0.5 * RN_UV * np.maximum(nemo["e1t"], nemo["e2t"])   # (j,i)
    ahmf_nemo = 0.5 * RN_UV * np.maximum(nemo["e1f"], nemo["e2f"])
    ahmt_eff_nemo = ahmt_nemo[None, :, :] * tmask                    # (k,j,i)
    ahmf_eff_nemo = ahmf_nemo[None, :, :] * fmask

    # legoESM's effective coefficients, on the same (k,j,i) layout.
    # legoESM T-row j  == NEMO T-row j.
    # legoESM vertex row j+1 is the corner between cell rows j and j+1
    #   == NEMO f-point row j (fmask(ji,jj) uses tmask rows jj, jj+1).
    # legoESM vertex column c is the corner west of cell column c
    #   == NEMO f-point column c-1.
    # legoESM's T-point coefficient carries the 2-D cell mask ONLY: the
    # production call at ocean_pe_latlon_cgrid.py:4246 hands the operator
    # ``mask`` (the 2-D land mask), not the 3-D ``mask_3d`` every other
    # momentum term in the same function receives.
    ahmt_eff_lego = (ahmt[None, :, None] * cell_mask[None, :, :]
                     * np.ones((nk, 1, 1)))
    vm = vm3 if vm3 is not None else np.broadcast_to(
        vertex_mask[..., None], vertex_mask.shape + (nk,))
    # (n_lat+1, n_lon+1, nk) -> NEMO's (k, j, i) f-layout
    vm_nemo = np.moveaxis(vm, -1, 0)[:, 1:, 1:]        # rows 1.., cols 1..
    ahmf_eff_lego = ahmf[None, 1:, None] * vm_nemo

    # The T-coefficient comparison is deliberately split: the SURFACE level
    # (where the 2-D and 3-D masks agree by construction) isolates the
    # coefficient arithmetic, and the full 3-D field then exposes the
    # masking difference on its own.
    for name, a, b in (
            ("ahmt(T) surf", ahmt_eff_nemo[:1], ahmt_eff_lego[:1]),
            ("ahmt(T) 3-D", ahmt_eff_nemo, ahmt_eff_lego),
            ("ahmf(F)*fmask", ahmf_eff_nemo, ahmf_eff_lego)):
        d = np.abs(a - b)
        scale = max(np.abs(a).max(), 1e-300)
        print(f"{name:>16s}: max|NEMO-lego| = {d.max():.6e}  "
              f"rel = {d.max() / scale:.3e}  (over all {a.size} points)")
        wall = d[:, WALL_ROWS[0]:WALL_ROWS[-1] + 1, :]
        print(f"{'':>16s}  wall rows only: max = {wall.max():.6e}")

    # the masks themselves, as booleans
    mask_lego_T = np.broadcast_to(cell_mask[None, :, :], tmask.shape)
    act_disagree = np.abs(mask_lego_T * np.ones_like(tmask) - tmask)
    n_sub = int(((ahmt_eff_lego > 0) & (ahmt_eff_nemo == 0)).sum())
    n_sub_wall = int(((ahmt_eff_lego[:, WALL_ROWS[0]:WALL_ROWS[-1] + 1] > 0)
                      & (ahmt_eff_nemo[:, WALL_ROWS[0]:WALL_ROWS[-1] + 1]
                         == 0)).sum())
    print(f"{'submerged T-pts':>16s}: cells NEMO masks off and legoESM does "
          f"not: {n_sub} total, {n_sub_wall} in the four wall rows")
    print(f"{'2-D cell mask':>16s}: cells where legoESM wet != NEMO tmask "
          f"(surface only): {int(np.abs(cell_mask - tmask[0]).sum())}")
    vtx_disagree = np.abs(np.moveaxis(vm, -1, 0)[:, 1:, 1:] - fmask)
    print(f"{'3-D vertex mask':>16s}: f-points where legoESM != NEMO fmask: "
          f"{int((vtx_disagree > 1e-12).sum())} of {fmask.size}")
    per_level = [(k, int((vtx_disagree[k] > 1e-12).sum())) for k in range(nk)]
    bad = [p for p in per_level if p[1]]
    print(f"{'':>16s}  levels with any disagreement: "
          f"{bad if bad else 'none'}")
    del act_disagree
    return ahmf_eff_nemo, ahmf_eff_lego


def part_b(ahmf_eff_nemo, ahmf_eff_lego, nemo, zeta_wall_lego):
    """The wall f-row must carry no viscous stress on either side.

    Two things are needed and both are checked: the COEFFICIENT at the wall
    corner must be zero (that is what free slip means here, and it is what
    makes the corner unable to carry a stress whatever the flow does), and the
    STRESS the operator actually forms there -- ``ahmf * zeta`` at the wall
    corner -- must be zero on the real state.  Printing only the coefficient
    would be printing the input to the claim rather than the claim.
    """
    print("\n--- B. the viscous flux through the wall face -----------------")
    # NEMO's wall f-row for a southern wall at cell row 0 is f-row 0
    # (fmask(ji,0) = tmask(ji,0)*tmask(ji+1,0)*tmask(ji,1)*tmask(ji+1,1)).
    wall_f = 0
    print(f"NEMO   fmask at the wall f-row {wall_f}: "
          f"max = {nemo['fmask'][:, wall_f, :].max():.3e}")
    print(f"NEMO   ahmf*fmask at the wall f-row: "
          f"max = {np.abs(ahmf_eff_nemo[:, wall_f, :]).max():.3e}")
    print(f"legoESM ahmf*vertex_mask at the same f-row: "
          f"max = {np.abs(ahmf_eff_lego[:, wall_f, :]).max():.3e}")
    print(f"legoESM ahmf*zeta at the same f-row, on the REAL state: "
          f"max = {np.abs(zeta_wall_lego).max():.3e} m2/s2")
    for name, arr in (("NEMO ahmf*fmask", ahmf_eff_nemo[:, wall_f, :]),
                      ("legoESM ahmf*vertex_mask", ahmf_eff_lego[:, wall_f, :]),
                      ("legoESM ahmf*zeta on the real state", zeta_wall_lego)):
        m = float(np.abs(arr).max())
        assert m == 0.0, (
            f"{name} is {m:.3e} at the wall f-row, not zero -- the wall is "
            "not free-slip on that side")
    print("(all exactly 0 -- free slip, rn_shlat=0: the wall f-point carries "
          "no shear stress, so no lateral momentum crosses the wall)")


def part_c(br, geom, u, v, u_mask, v_mask, cell_mask, vm3, ahmt, ahmf, nemo):
    """The 2-D vs 3-D masking of the SAME operator, on the SAME state.

    Production hands ``nemo_ldf_lap_viscosity_cgrid`` the 2-D ``mask`` /
    ``u_mask`` / ``v_mask`` (``ocean_pe_latlon_cgrid.py:4246-4247``) while every
    other momentum term in the same function receives the partial-cell 3-D
    masks.  NEMO masks the same operator in 3-D on both points: ``ahmt`` is
    multiplied by the 3-D ``tmask`` and ``ahmf`` by the 3-D ``fmask``
    (``ldfdyn.F90:328-330``) and the velocities it reads are 3-D masked.

    Applying both maskings to the bridged state as it stands returns exactly
    zero, and that number is NOT a statement about the operator: the state's
    velocity is already zero below the sea floor, so the cells the two maskings
    disagree about contribute nothing whichever mask is used.  A probe that
    returns the same answer for a correct and an incorrect operator is not
    measuring the operator.  So this also runs a CONTROL that plants a velocity
    on exactly those below-seafloor faces and re-runs both arms: that says how
    much the masking would be worth if the state were ever nonzero there, and
    it is what makes the zero above meaningful.

    It also reports the tendency the production (2-D-masked) arm produces on
    those below-seafloor faces, which is not zero, since that is real wiring
    debt even though nothing downstream reads it.
    """
    print("\n--- C. 2-D vs 3-D masking of the same operator ----------------")
    act = getattr(br.z_coord, "is_active", None)
    if act is None:
        print("no is_active on the bridged coordinate; part C skipped")
        return None
    import jax.numpy as jnp
    act3 = jnp.asarray(np.asarray(act, dtype=np.float64))
    um3, vm3f = compute_face_masks_3d(act3, grid=geom)
    um3 = np.asarray(um3, dtype=np.float64) * (
        u_mask[..., None] if u_mask.ndim == 2 else u_mask)
    vm3f = np.asarray(vm3f, dtype=np.float64) * (
        v_mask[..., None] if v_mask.ndim == 2 else v_mask)
    cm3 = np.asarray(act3, dtype=np.float64) * cell_mask[..., None]

    prod_u, prod_v = nemo_ldf_lap_viscosity_cgrid(
        u, v, geom, ahmt, ahmf, mask=cell_mask, u_mask=u_mask,
        v_mask=v_mask, vertex_mask=vm3)
    full_u, full_v = nemo_ldf_lap_viscosity_cgrid(
        u, v, geom, ahmt, ahmf, mask=cm3, u_mask=um3,
        v_mask=vm3f, vertex_mask=vm3)
    prod_u = np.asarray(prod_u)
    full_u = np.asarray(full_u)
    prod_v = np.asarray(prod_v)
    full_v = np.asarray(full_v)

    # Score on WET u-faces only, per row, depth- and longitude-averaged so a
    # deeper row does not automatically score higher.  Weight by the 3-D face
    # mask: a difference confined to cells NEMO calls land is not a difference
    # in the model's answer, and must be reported separately from one that
    # lands on live water.
    wet = um3
    n_wet = wet.sum(axis=(1, 2))

    dry = 1.0 - wet
    n_dry = dry.sum(axis=(1, 2))

    def rma(x, w, n):
        """Mean |x| over the cells w selects -- divided by THAT cell count.

        Normalising a dry-cell sum by the wet-cell count understates it by the
        dry/wet ratio, which at these rows is a factor of four to six.
        """
        return np.where(n > 0, (np.abs(x) * w).sum(axis=(1, 2)),
                        np.nan) / np.maximum(n, 1)

    base = rma(prod_u, wet, n_wet)
    dwet = rma(full_u - prod_u, wet, n_wet)
    ddry = rma(full_u - prod_u, dry, n_dry)
    prod_dry = rma(prod_u, dry, n_dry)
    print(" row   phi[deg]   mean|du_prod|   d(3D-2D) on WET   rel"
          "     d on DRY cells")
    interior = []
    for j in list(WALL_ROWS) + [8, 12, 20, 40, 99, 150]:
        if j >= len(base):
            continue
        rel = dwet[j] / base[j] if base[j] > 0 else np.nan
        tag = "WALL" if j in WALL_ROWS else ""
        print(f"{j:4d} {nemo['gphit'][j, 26]:+9.3f}  {base[j]:14.6e}  "
              f"{dwet[j]:16.6e}  {rel:9.2e}  {ddry[j]:14.6e}  {tag}")
        if j not in WALL_ROWS:
            interior.append(rel)
    w = [dwet[j] / base[j] for j in WALL_ROWS if base[j] > 0]
    print(f"wall-row mean relative change on wet faces: {np.mean(w):.3e}")
    print(f"interior-row mean relative change:          "
          f"{np.nanmean(interior):.3e}")
    print(f"max|3D-2D| anywhere on a wet face: "
          f"{np.abs((full_u - prod_u) * wet).max():.6e} m/s2  "
          f"(v: {np.abs((full_v - prod_v) * vm3f).max():.6e})")

    # How much of the field the two maskings even disagree about, and whether
    # the state is nonzero there -- the fact that decides what the zero means.
    dis_u = ((u_mask[..., None] if u_mask.ndim == 2 else u_mask) > 0) & (um3 <= 0)
    print(f"u-faces the two maskings disagree about (2-D wet, 3-D dry): "
          f"{int(dis_u.sum())}")
    print(f"  max|u| on those faces in this state: "
          f"{np.abs(np.where(dis_u, u, 0.0)).max():.6e} m/s")
    print(f"  tendency the PRODUCTION 2-D-masked arm writes there: "
          f"max {np.abs(np.where(dis_u, prod_u, 0.0)).max():.6e} m/s2, "
          f"row-mean {np.nanmax(prod_dry):.6e} "
          f"(wiring debt: below the sea floor, nothing downstream reads it)")

    # THE CONTROL. Plant on exactly the disputed faces and re-run both arms.
    u_p = np.where(dis_u, 0.5, u)
    dis_v = ((v_mask[..., None] if v_mask.ndim == 2 else v_mask) > 0) & (
        np.asarray(compute_face_masks_3d(act3, grid=geom)[1]) <= 0)
    v_p = np.where(dis_v, 0.5, v)
    p2 = np.asarray(nemo_ldf_lap_viscosity_cgrid(
        u_p, v_p, geom, ahmt, ahmf, mask=cell_mask, u_mask=u_mask,
        v_mask=v_mask, vertex_mask=vm3)[0])
    p3 = np.asarray(nemo_ldf_lap_viscosity_cgrid(
        u_p, v_p, geom, ahmt, ahmf, mask=cm3, u_mask=um3, v_mask=vm3f,
        vertex_mask=vm3)[0])
    ctl = np.abs((p3 - p2) * wet).max()
    print(f"CONTROL, 0.5 m/s planted on the {int(dis_u.sum())} disputed "
          f"u-faces: max|3D-2D| on a WET face becomes {ctl:.6e} m/s2, "
          f"{ctl / max(np.abs(prod_u * wet).max(), 1e-300):.2f}x the wet-face "
          f"tendency -- so the zero above is a property of THIS STATE (u=0 "
          f"below the sea floor), not of the two maskings being equivalent")
    assert ctl > 0.0, (
        "the control planted a velocity on the disputed faces and the two "
        "maskings STILL agree exactly -- this A/B cannot distinguish them and "
        "its zero proves nothing")
    return dict(wall_rel=float(np.mean(w)),
                interior_rel=float(np.nanmean(interior)),
                control=float(ctl))


def part_d(br, geom, u, v, u_mask, v_mask, cell_mask, vm3, ahmt, ahmf, nemo):
    """The e3 (layer-thickness) weighting: both treatments on the SAME state.

    NEMO weights the divergence and the vorticity by the layer thickness
    (``dynldf_lev_rot_scheme.h90:23`` e3f on zcur, ``:27-29`` e3t/e3u/e3v inside
    zdiv, ``:41,:51`` the /e3u,/e3v outside); the shipped legoESM card runs
    ``lateral_viscosity_e3_weighting="off"``.  This is the surviving
    transcription difference, so its SHAPE has to be measured against the
    measured wall fingerprint -- a 34% amplitude error confined to four rows
    with the rest of the basin right to 3% -- before it can be called the owner.

    Both arms are the model's own operators, not a re-transcription: the shipped
    ``nemo_ldf_lap_viscosity_cgrid`` and the faithful
    ``nemo_ldf_lap_viscosity_e3_cgrid``, on the identical bridged state.
    """
    print("\n--- D. e3 weighting: both treatments on the same state --------")
    eta = np.asarray(br.state.eta.data, dtype=np.float64)
    H = np.asarray(br.state.H_bathy.data, dtype=np.float64)
    h_k = np.asarray(compute_layer_thickness(eta, H, br.z_coord),
                     dtype=np.float64)
    kw = dict(mask=cell_mask, u_mask=u_mask, v_mask=v_mask, vertex_mask=vm3)
    off_u, off_v = nemo_ldf_lap_viscosity_cgrid(u, v, geom, ahmt, ahmf, **kw)
    e3_u, e3_v = nemo_ldf_lap_viscosity_e3_cgrid(u, v, geom, ahmt, ahmf, h_k,
                                                 **kw)
    off_u = np.asarray(off_u)
    e3_u = np.asarray(e3_u)

    um3, _ = compute_face_masks_3d(
        __import__("jax").numpy.asarray(
            np.asarray(br.z_coord.is_active, dtype=np.float64)), grid=geom)
    wet = np.asarray(um3, dtype=np.float64) * (
        u_mask[..., None] if u_mask.ndim == 2 else u_mask)
    n_wet = wet.sum(axis=(1, 2))

    def rma(x):
        return np.where(n_wet > 0, (np.abs(x) * wet).sum(axis=(1, 2)),
                        np.nan) / np.maximum(n_wet, 1)

    base = rma(off_u)
    d = rma(e3_u - off_u)
    print(" row   phi[deg]   mean|du_off|    mean|d(e3-off)|      rel")
    interior = []
    for j in list(WALL_ROWS) + [8, 12, 20, 40, 99, 150]:
        if j >= len(base):
            continue
        rel = d[j] / base[j] if base[j] > 0 else np.nan
        tag = "  WALL" if j in WALL_ROWS else ""
        print(f"{j:4d} {nemo['gphit'][j, 26]:+9.3f}  {base[j]:13.6e}  "
              f"{d[j]:16.6e}  {rel:9.4f}{tag}")
        if j not in WALL_ROWS:
            interior.append(rel)
    w = [d[j] / base[j] for j in WALL_ROWS if base[j] > 0]
    print(f"wall-row mean relative change from the e3 weighting: "
          f"{np.mean(w):.4f}")
    print(f"interior-row mean relative change:                   "
          f"{np.nanmean(interior):.4f}")
    print(f"enrichment (wall / interior): "
          f"{np.mean(w) / np.nanmean(interior):.2f}x")
    # the same scores over EVERY row, so the four wall rows are not being
    # compared against a hand-picked interior sample
    allrel = np.where(base > 0, d / np.maximum(base, 1e-300), np.nan)
    print(f"all rows: median rel {np.nanmedian(allrel):.4f}, "
          f"90th pct {np.nanpercentile(allrel, 90):.4f}, "
          f"max {np.nanmax(allrel):.4f} at row "
          f"{int(np.nanargmax(allrel))}")
    return dict(wall_rel=float(np.mean(w)),
                interior_rel=float(np.nanmean(interior)))


def self_test() -> int:
    """The dry-row trap, both directions.

    (i) planting a large value in the DRY cells must move no scored quantity;
    (ii) planting the same value in a WET cell of a wall row must move one.
    """
    print("=== SELF-TEST: the dry-row trap ===")
    br, geom, cfg, u, v, um, vm, cm, h_k = build_lego()
    _, ahmt, ahmf, vtx, vm3 = effective_coefficients(br, geom, cm)
    kw = dict(mask=cm, u_mask=um, v_mask=vm, vertex_mask=vm3)
    base = np.asarray(nemo_ldf_lap_viscosity_cgrid(u, v, geom, ahmt, ahmf,
                                                   **kw)[0])

    _, _, _, u_p, v_p, _, _, _, _ = build_lego(plant_dry=1e3)
    planted = np.asarray(nemo_ldf_lap_viscosity_cgrid(u_p, v_p, geom, ahmt,
                                                      ahmf, **kw)[0])
    moved = np.abs(planted - base).max()
    print(f"(i)  dry-cell plant of 1e3 m/s: max|d(du)| = {moved:.6e}")
    assert moved == 0.0, (
        "a dry-cell value reached the scored tendency -- the wall stencil "
        f"is contaminated (max move {moved:.3e})")

    u_w = u.copy()
    u_w[WALL_ROWS[0], 10, 0] += 1e-3
    wetted = np.asarray(nemo_ldf_lap_viscosity_cgrid(u_w, v, geom, ahmt, ahmf,
                                                     **kw)[0])
    moved_w = np.abs(wetted - base).max()
    print(f"(ii) wet-cell plant of 1e-3 m/s on wall row "
          f"{WALL_ROWS[0]}: max|d(du)| = {moved_w:.6e}")
    assert moved_w > 0.0, (
        "planting a WET wall-row value moved nothing -- the check in (i) is "
        "vacuous")
    print("SELF-TEST PASS: dry cells are excluded and the check can fail")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    set_policy(PrecisionPolicy.fp64())
    stamp()
    if args.self_test:
        return self_test()

    nemo = load_nemo()
    br, geom, cfg, u, v, um, vm, cm, h_k = build_lego()
    print(f"bridge f_T self-check: max|geom.f_T - NEMO ff_t| = "
          f"{br.f_match_max_abs:.3e}")
    half_UM, ahmt, ahmf, vtx, vm3 = effective_coefficients(br, geom, cm)
    a_n, a_l = part_a(nemo, geom, half_UM, ahmt, ahmf, vtx, vm3, cm)
    # the stress the operator actually forms at the wall corner on this state:
    # ahmf * zeta at legoESM vertex row 1 (== NEMO f-row 0, the wall corner).
    from legoesm.grids.operators_latlon_cgrid import curl_vertex_cgrid
    _um = um[..., None] if um.ndim == 2 else um
    _vm = vm[..., None] if vm.ndim == 2 else vm
    _zeta = np.asarray(curl_vertex_cgrid(u * _um, v * _vm, geom))
    _zeta_wall = _zeta[1] * (vm3[1] if vm3 is not None else vtx[1][..., None]) \
        * ahmf[1]
    part_b(a_n, a_l, nemo, _zeta_wall)
    part_c(br, geom, u, v, um, vm, cm, vm3, ahmt, ahmf, nemo)
    part_d(br, geom, u, v, um, vm, cm, vm3, ahmt, ahmf, nemo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
