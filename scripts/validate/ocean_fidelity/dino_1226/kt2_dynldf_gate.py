#!/usr/bin/env python
"""The MOMENTUM row at kt=2, operator to operator: ``dynldf_lev_lap``.

WHY.  At kt=2 the zonal velocity carries 6.02% of NEMO's own step -- 6x the
next row (eta at 0.98%) and 120x the temperature row -- and a tracer statement
cannot move it at that size in one step.  The named suspect is the lateral
viscosity, because NEMO's iso-level Laplacian is THREE-WAY in its heights
where legoESM passes ONE live thickness.  This gate MEASURES the operator
instead of arguing about it (Rule 4: an alignment table that comes out clean
is a real result -- it eliminates a suspect).

WHAT DINO RUNS, read from the compiled source and the run's own ocean.output:

* ``namdyn_ldf``: ``ln_dynldf_lap=.true.``, ``ln_dynldf_lev=.true.``,
  ``nn_ahm_ijk_t=20``, ``rn_Uv=0.27`` (``RUN_TRAJ/namelist_cfg:171-176``)
  -> ``nldf_dyn = np_lap`` and ``ocean.output`` prints "iso-level laplacian
  operator".
* ``dynldf.f90:96-98``: ``CASE( np_lap )  CALL dynldf_lev_lap( kt, Kbb, Kmm,
  puu, pvv, Krhs )`` -- the ``dyn_ldf_lap`` of ``dynldf_lap_blp`` is COMMENTED
  OUT (``!!st``), so the routine that runs is ``dynldf_lev.f90``.
* ``nn_dynldf_typ = 0`` (``ocean.output:897``) -> ``np_typ_rot``, the
  vorticity-divergence operator, ``dynldf_lev_rot_scheme.h90``.

THE ALIGNMENT TABLE (``dynldf_lev.f90`` line numbers are the compiled
``BLD/ppsrc/nemo/dynldf_lev.f90``):

| # | statement | NEMO | legoESM ``nemo_ldf_lap_viscosity_e3_cgrid`` |
|---|---|---|---|
| N1 | velocity operand | ``pu(...,Kbb)``, ``pv(...,Kbb)`` at :124-129 | ``ldf_state[2:4]`` = the Kbb velocity (ocean_pe:4714-4716) -- MATCH |
| N2 | curl height | ``e3f_3d*(1+r3f*fe3mask)`` :124 -- ``r3f`` carries NO time index; ``domqco.f90:178`` and ``stpmlf.f90:350`` set it from the **NOW** ssh | ``h_vtx = min_cell_to_vertex(h_k)`` with ``h_k`` the step-entry (NOW) thickness -- MATCH in time level, DIFF in construction (min-rule of live thicknesses vs ``e3f_0`` x an area-weighted f-point stretch, :223) |
| N3 | divergence heights | ``e3u_3d*(1+r3u(**Kbb**))``, ``e3v_3d*(1+r3v(**Kbb**))``, divisor ``e3t_3d*(1+r3t(**Kbb**))`` :127-129 | all three from the NOW ``h_k`` -- **DIFF: NEMO is BEFORE here** |
| N4 | final divisor | ``e3u_3d*(1+r3u(**Kmm**))``, ``e3v_3d*(1+r3v(**Kmm**))`` :133,:137 | NOW ``h_u``/``h_v`` -- MATCH |
| N5 | coefficient | ``ahmf``/``ahmt``, already multiplied by fmask/tmask (:124,:127) | ``nemo_lateral_viscosity_coefficients`` + explicit masks -- MATCH in form |
| N6 | sign/assembly | ``- curl(curl) + grad(div)`` on u, ``+ curl(curl) + grad(div)`` on v, each times umask/vmask :131-138 | same signs, same masks -- MATCH |

So exactly ONE row is a time-level DIFF (N3) and one is a construction DIFF
(N2).  This gate sizes what those are worth by scoring the operator against
NEMO's own ``utrd_ldf``/``vtrd_ldf`` at kt=2, with NEMO's own operands on both
sides: the kt=1 restart's ``ub``/``vb`` ARE the Kbb velocity at kt=2, and its
``sshb``/``sshn`` ARE ``ssh(Kbb)``/``ssh(Kmm)`` there (the index rotation at
``stpmlf.f90:577-580`` precedes ``rst_write`` at ``:590``).

WHAT THIS GATE CANNOT SEE (Rule 2).  It scores ONE operator's tendency, so it
can only EXONERATE or INDICT ``dyn_ldf``; it says nothing about which other
momentum term owns the u row if this one comes out clean.  And
``utrd_ldf`` is ``puu(Krhs)`` after minus before (``dynldf.f90:80-92``,
``:118``), a difference of RHS accumulators, so it carries NEMO's own
cancellation noise -- the floor row prints how big that is.

Usage
-----
    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
      python scripts/validate/ocean_fidelity/dino_1226/kt2_dynldf_gate.py \\
        --run-dir /data/abyssal/.../nemo_kt2_trends
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from rebuild_nemo_restart import rebuild                        # noqa: E402

CERTIFIED_KT1 = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
                 "RUN_FROMREST_KT1")


def _row(name, lego, nemo, wet):
    """One scored row.

    A ROW WHOSE REFERENCE IS IDENTICALLY ZERO IS **UNMEASURED**, NEVER AT BAR.
    A diff reviewer showed the first version of this gate printed four AT BAR
    rows that were zero-vs-zero, called itself GATE PASS, and was byte-
    identical under its own --plant -- the repo's own "a control that perturbs
    a zero is not a control", shipped as a gate.  A zero reference now
    FAILS the gate with the reason printed, which is the honest verdict: this
    gate cannot say anything about dyn_ldf at kt=2.
    """
    d = np.abs(np.asarray(lego) - np.asarray(nemo))[wet]
    n = int((d != 0.0).sum())
    x, y = np.asarray(lego)[wet], np.asarray(nemo)[wet]
    den = float(y @ y)
    ratio = float(x @ y) / den if den else float("nan")
    nemo_rms = float(np.sqrt(np.mean(y ** 2)))
    if nemo_rms == 0.0:
        verdict = "UNMEASURED (reference is identically zero)"
        bad = 1
    elif n == 0:
        verdict, bad = "AT BAR", 0
    else:
        verdict, bad = "DEBT", 1
    print(f"  {name:34s}{n:>9d}{d.max():13.4e}"
          f"{float(np.sqrt(np.mean(d ** 2))):13.4e}"
          f"{nemo_rms:13.4e}{ratio:14.9f}  {verdict}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--kt1-dir", default=CERTIFIED_KT1)
    ap.add_argument("--plant", action="store_true",
                   help="scale legoESM's u tendency by (1 + 1e-12); every row "
                        "MUST then move.  Without it a 'clean' row proves "
                        "nothing about the gate's resolution.")
    a = ap.parse_args()

    k1 = os.path.join(a.kt1_dir, "DINO_00000001_restart_*.nc")
    k2 = os.path.join(a.run_dir, "DINO_00000002_restart_*.nc")
    for pat in (k1, k2):
        if not glob.glob(pat):
            raise SystemExit(f"no tiles match {pat}")
    R1 = rebuild(k1, ["ub", "vb", "sshb", "sshn"])
    R2 = rebuild(k2, ["utrd_ldf", "vtrd_ldf"])
    for k in ("utrd_ldf", "vtrd_ldf"):
        if k not in R2:
            raise SystemExit(f"the kt=2 restart carries no {k}: ln_dyn_trd off?")

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                          # Rule 1c
    import jax.numpy as jnp
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_vertex_mask, nemo_lateral_viscosity_coefficients,
        nemo_ldf_lap_viscosity_e3_cgrid)
    from legoesm.ocean.vertical import compute_layer_thickness

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    state0 = dm.dino_lat_lon_state(grid, z, cfg)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    g = ndm.nemo_dino_mesh()
    H = state0.H_bathy.data
    uwet = np.asarray(g.umask > 0.5)
    vwet = np.asarray(g.vmask > 0.5)
    cell = np.asarray(g.tmask > 0.5)

    # Rule 10: PRINT the fields that exist.  The first version of this line
    # read two attribute names the config does not have and printed None for
    # both, which proved nothing about which operator the card selects.
    _lv = mc.lateral_viscosity
    print("  lateral_viscosity fields: "
          + "  ".join(f"{f}={getattr(_lv, f)!r}" for f in _lv._fields
                      if "oper" in f or "e3" in f or f == "A_h"))
    half_UM = mc.lateral_viscosity.A_h / (grid.radius * grid.dlon)
    ahmt, ahmf = nemo_lateral_viscosity_coefficients(grid, half_UM)
    print(f"  ahmt[0]={float(ahmt[0]):.6f}  ahmt[-1]={float(ahmt[-1]):.6f}  "
          f"ahmf[0]={float(ahmf[0]):.6f}  dtype {np.asarray(ahmt).dtype}")

    def _O3(k, R):
        return np.nan_to_num(np.moveaxis(R[k], 0, -1))

    # legoESM's redundant west/south face, dropped exactly as
    # kt2_leapfrog_gate.py:310-322 drops it -- here INVERTED, because the
    # operator wants the full arrays back.
    u_kbb = np.zeros((199, 53, 36))
    u_kbb[:, 1:, :] = _O3("ub", R1)
    u_kbb[:, 0, :] = u_kbb[:, -1, :]            # periodic image west of cell 0
    v_kbb = np.zeros((200, 52, 36))
    v_kbb[1:, :, :] = _O3("vb", R1)             # closed southern wall stays 0

    # legoESM's OWN face masks, on its OWN staggering (199,53) / (200,52) --
    # NEMO's umask/vmask are the east/north faces only and cannot be handed to
    # the operator directly.  Broadcast to 3-D with the per-level wet mask so
    # the submerged staircase faces are free-slip (the fmask analogue).
    _act = np.asarray(getattr(z, "is_active"), dtype=float)
    _um2 = np.asarray(state0.u_mask.data, dtype=float)
    _vm2 = np.asarray(state0.v_mask.data, dtype=float)
    # u-face i sits between cells i-1 and i (face 0 = the periodic image of
    # face 52), so its wetness is the MIN of the two cells it separates --
    # the same min-rule the operator uses for its face thicknesses.
    _a_u = np.minimum(np.concatenate([_act[:, -1:, :], _act], axis=1),
                      np.concatenate([_act, _act[:, :1, :]], axis=1))
    _a_v = np.minimum(np.concatenate([np.zeros_like(_act[:1]), _act], axis=0),
                      np.concatenate([_act, np.zeros_like(_act[:1])], axis=0))
    u_mask3 = (_um2[..., None] if _um2.ndim == 2 else _um2) * _a_u
    v_mask3 = (_vm2[..., None] if _vm2.ndim == 2 else _vm2) * _a_v
    vmask3 = np.asarray(np.stack(
        [np.asarray(compute_vertex_mask(
            jnp.asarray(_act[..., k] * cell[..., k].astype(float)), grid=grid))
         for k in range(_act.shape[-1])], axis=-1))

    print("\nlegoESM's dynldf_lev_lap tendency vs NEMO's utrd/vtrd_ldf at kt=2,"
          " NEMO's OWN operands on both sides")
    print(f"  {'arm':34s}{'cells!=':>9s}{'max|d|':>13s}{'rms':>13s}"
          f"{'NEMO rms':>13s}{'ratio':>14s}")
    bad = 0
    out = {}
    print(f"  Kbb VELOCITY OPERAND (the kt=1 restart's ub/vb): "
          f"max|ub| {np.abs(np.nan_to_num(R1['ub'])).max():.4e}  "
          f"max|vb| {np.abs(np.nan_to_num(R1['vb'])).max():.4e} m/s")
    for tag, sshkey in (("N3 as legoESM has it (Kmm)", "sshn"),
                        ("N3 as NEMO has it  (Kbb)", "sshb")):
        eta = jnp.asarray(np.nan_to_num(np.asarray(R1[sshkey],
                                                   dtype=np.float64)))
        h_k = compute_layer_thickness(eta, H, z,
                                      min_water_column_m=mc.min_water_column_m)
        du, dv = nemo_ldf_lap_viscosity_e3_cgrid(
            jnp.asarray(u_kbb), jnp.asarray(v_kbb), grid, ahmt, ahmf, h_k,
            mask=state0.land_mask.data,
            u_mask=jnp.asarray(u_mask3), v_mask=jnp.asarray(v_mask3),
            vertex_mask=jnp.asarray(vmask3))
        du = np.asarray(du, dtype=np.float64)
        dv = np.asarray(dv, dtype=np.float64)
        if a.plant:
            du = du * (1.0 + 1e-12)
        out[tag] = (du, dv)
        bad += _row(f"u  {tag}", du[:, 1:, :] if du.shape[1] == 53 else du,
                    _O3("utrd_ldf", R2), uwet)
        bad += _row(f"v  {tag}", dv[1:, :, :] if dv.shape[0] == 200 else dv,
                    _O3("vtrd_ldf", R2), vwet)

    # THE DISCRIMINATOR the kt=2 preregistration asked for: how big is the
    # N3 statement ITSELF, against the u row it is supposed to own?
    a1 = out["N3 as legoESM has it (Kmm)"][0]
    a2 = out["N3 as NEMO has it  (Kbb)"][0]
    d = np.abs(a1 - a2)
    du_n3 = d[:, 1:, :] if d.shape[1] == 53 else d
    nemo_u = _O3("utrd_ldf", R2)
    _n3 = float(np.sqrt(np.mean(du_n3[uwet] ** 2)))
    _nr = float(np.sqrt(np.mean(nemo_u[uwet] ** 2)))
    print("\n  THE N3 STATEMENT ALONE (Kmm height minus Kbb height), on u:")
    print(f"    rms {_n3:.4e} m/s^2   max {du_n3[uwet].max():.4e}")
    if _nr == 0.0:
        print("    NEMO's own ldf tendency is identically zero at this step, "
              "so there is nothing to take a fraction OF -- the statement's "
              "relative size is UNDEFINED here, not small.")
    else:
        print(f"    = {_n3 / _nr:.3e} of NEMO's own ldf tendency")
    print(f"    x rDt = {_n3 * 5400.0:.4e} m/s against the kt=2 u STATE "
          f"residual 1.4101e-04 m/s "
          f"(fraction {_n3 * 5400.0 / 1.4101e-04:.3e})")
    if a.plant:
        print("  PLANT ACTIVE (u scaled by 1+1e-12): every u row above must "
              "differ from the unplanted run")
    # ---- Rule 3 / Rule 1: a bucket that is identically zero is NOT evidence
    # of agreement, and the only way to know which momentum term DOES own the
    # u row is to look at what NEMO's own buckets contain.  Ranked inventory,
    # every utrd_* the restart carries, so the next owner is named from the
    # oracle's numbers and not from a suspicion.
    import netCDF4 as _nc
    _d0 = _nc.Dataset(sorted(glob.glob(k2))[0])
    _keys = sorted(v for v in _d0.variables
                   if v.startswith("utrd_") or v.startswith("vtrd_"))
    _d0.close()
    RA = rebuild(k2, _keys)
    print("\n  NEMO's OWN momentum trend buckets at kt=2 (wet rms, m/s^2), "
          "ranked -- a bucket at 0.0 cannot own any row")
    rank = []
    for kk in _keys:
        if kk not in RA:
            continue
        w = uwet if kk.startswith("u") else vwet
        v = np.nan_to_num(np.moveaxis(RA[kk], 0, -1))
        rank.append((float(np.sqrt(np.mean(v[w] ** 2))), kk,
                     int((v[w] != 0).sum())))
    for r, kk, nz in sorted(rank, reverse=True):
        print(f"    {kk:14s} rms {r:11.4e}   nonzero {nz:>8d}")

    print(f"\n{'GATE PASS' if bad == 0 else f'GATE FAIL ({bad} rows)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
