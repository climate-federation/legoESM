#!/usr/bin/env python
"""Scaling study for the DINO twin's implicit-vertical-mixing gradient DIVISOR.

QUESTION (a hypothesis to be MEASURED): the certified DINO twin
(``nemo_dino_kamm_mlf``) runs a non-NEMO divisor in the backward-Euler
vertical-mixing solve.  Is that difference big enough to be part of what makes
legoESM and NEMO DISTINGUISHABLE at 20 years
(``dino_multi_year_climate_equivalence_result.md``)?

THE ORACLE (Rule 0 -- read it, quote it)
----------------------------------------
NEMO's implicit tracer solve divides the vertical gradient by ``e3w(...,Kmm)``::

    TRA/trazdf.F90:219-221
        zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / e3w(ji,jj,jk  ,Kmm)
        zws(ji,jk) = - p2dt * zwt(ji,jk+1) / e3w(ji,jj,jk+1,Kmm)
        zwd(ji,jk) = e3t(ji,jj,jk,Kaa) - ( zwi(ji,jk) + zws(ji,jk) )

and the momentum solve does the same with ``e3uw``/``e3vw``::

    DYN/dynzdf.F90:182-185
        zzwi = - zDt_2 * ( avm(ji+1,jj,jk) + avm(ji,jj,jk) )
             / ( e3u(ji,jj,jk,Kaa) * e3uw(ji,jj,jk  ,Kmm) ) * wumask(...)

The time levels are pinned by the MLF call sites (Rule 1d)::

    stpmlf.F90:370   CALL tra_zdf( kstp, Nbb, Nnn, Nrhs, ts, Naa )
    stpmlf.F90:267   CALL dyn_zdf( kstp, Nbb, Nnn, Nrhs, uu, vv, Naa )

so ``Kmm -> Nnn`` (NOW) for the divisor and ``Kaa -> Naa`` (AFTER) for the
diagonal thickness.  DINO builds with ``key_qco key_vco_3d``
(``cfgs/DINO/cpp_DINO.fcm``), so ``DOM/domzgr_substitute.h90:131`` +
``:49`` expand the divisor to::

    e3w(i,j,k,t) = E3w_0(i,j,k) * (1 + r3t(i,j,t))     [key_qco, no mask]
    E3w_0(i,j,k) = e3w_3d(i,j,k)                       [key_vco_3d, :108]

and ``r3t = ssh/ht_0``, i.e. NEMO's divisor is

    D_nemo = e3w_0(i,j,k) * (1 + eta_NOW / H)

where ``e3w_0`` is the T-POINT DEPTH DIFFERENCE ``gdept_0(k) - gdept_0(k-1)``
(verified from the oracle's own mesh_mask, rel err 0.0), NOT the interface
midpoint ``0.5*(e3t_k + e3t_{k-1})``.

LEGOESM'S EXECUTING ARM
-----------------------
``LatLonCGridOceanModel._apply_implicit_vertical_mixing``
(ocean_model_latlon_cgrid.py, the ``else`` branch of the divisor block) runs

    D_lego = build_dz_half(h_partial * J(eta_AFTER))
           = 0.5*(dz_k + dz_{k+1}) * (1 + eta_AFTER / H)

with both ``implicit_vmix_dzw_slot`` and the (since-removed)
``implicit_vmix_e3t_now_divisor``
False on the certified card (measured here, Rule 10).  So it differs from NEMO
in TWO independent ways:

  (a) SLOT      midpoint 0.5*(dz_k+dz_{k+1}) instead of e3w_0 = d(gdept)
  (b) TIME LVL  eta_AFTER instead of eta_NOW

``implicit_vmix_dzw_slot=True`` fixes (a) only (it uses ``z_coord.dz_half_ref``,
which on this card IS NEMO's ``e3w_0`` -- measured, rel 0.0);
``implicit_vmix_e3t_now_divisor=True`` fixed (b) only.  Neither flag alone was
NEMO.  On a single call with one state, eta_NOW == eta_AFTER, so the dzw arm IS
exactly NEMO's divisor and isolates (a) as a clean one-variable arm (Rule 7).

WHAT THIS SCRIPT MEASURES
-------------------------
1. GEOMETRY (Rule 10 + Rule 1c): resolved config flags, dtypes, and the divisor
   fields themselves -- ``D_nemo`` vs ``D_lego`` relative difference, per level
   and over wet interfaces, on the twin's REAL grid.
2. CALIBRATION (Rule 3) before any residual is quoted:
     C1  identical divisors -> tendency difference EXACTLY 0
     C2  eta-independence of the ratio (J cancels analytically; check it)
     C3  a planted uniform +1% divisor -> vmix tendency moves as predicted
3. ONE-STEP TENDENCY: the REAL ``_apply_implicit_vertical_mixing``, called twice
   on the SAME state and the SAME K_v/A_v profiles, differing only in the
   divisor.  Reported against (i) the total vertical-mixing tendency and (ii)
   the total one-step tendency of the full model step.

Run (CPU, fp64)::

    PYTHONPATH=packages/core:packages/ocean:... JAX_PLATFORMS=cpu \\
    JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/dino_1226/\\
    dino_zdf_divisor_scaling.py --states restart

``--states restart,m0_day360,m0_day7200`` also substitutes the 20-year
ensemble member m0's end-states from the climate-equivalence archive.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# fp64 BEFORE any legoESM construction (Rule 1c: JAX_ENABLE_X64 is not enough).
from legoesm.core.precision import PrecisionPolicy, set_policy

set_policy(PrecisionPolicy.fp64())

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kamm_twin_90d import (  # noqa: E402
    DT,
    RESTART_FILE,
    RUN_STEPDUMP,
    RUN_TRAJ,
    _build_twin_state,
    seasonal_t0_seconds,
)

from legoesm.ocean.physics.vertical_mixing.implicit_solver import (  # noqa: E402
    build_dz_half,
)
from legoesm.ocean.vertical import (  # noqa: E402
    OceanPartialCellCoordinate,
    compute_ocean_jacobian,
)

ARCHIVE = os.environ.get(
    "DINO_CLIMEQ_ARCHIVE",
    "/data/abyssal/dbalwada/dino-climate-equivalence-20y-01a04e34",
)


def _np(x):
    return np.asarray(x, dtype=np.float64)


_E3W_REF: np.ndarray | None = None


def divisor_fields(state, z_coord, config, eta_now=None):
    """Return ``(D_lego, D_nemo)`` -- the two candidate gradient divisors [m].

    ``D_lego`` is the EXECUTING arm's midpoint-of-AFTER divisor; ``D_nemo`` is
    NEMO's ``e3w(Kmm) = e3w_0 * (1 + eta_now/H)``.  Both shape
    ``(..., nlev-1)``.  This mirrors the model's own construction exactly
    (same helpers, same Jacobian); it does not re-derive it.
    """
    eta_after = _np(state.eta.data)
    eta_n = eta_after if eta_now is None else _np(eta_now)
    H = _np(state.H_bathy.data)
    J_after = _np(compute_ocean_jacobian(
        jnp.asarray(eta_after), jnp.asarray(H), z_coord,
        min_water_column_m=config.min_water_column_m))
    J_now = _np(compute_ocean_jacobian(
        jnp.asarray(eta_n), jnp.asarray(H), z_coord,
        min_water_column_m=config.min_water_column_m))
    if isinstance(z_coord, OceanPartialCellCoordinate):
        dz_cell = _np(z_coord.h_partial) * J_after[..., None]
    else:
        dz_cell = _np(z_coord.dz_ref) * J_after[..., None]
    d_lego = _np(build_dz_half(jnp.asarray(dz_cell)))
    d_nemo = _E3W_REF * J_now[..., None]
    return d_lego, d_nemo


_E3W_CACHE: dict[str, np.ndarray] = {}


def nemo_e3w_ref(mesh_path: str, nlev: int) -> np.ndarray:
    """NEMO's OWN ``e3w_0`` array, read from the oracle's ``mesh_mask.nc``.

    Rule 0: this is the array the ``E3w_0(i,j,k)`` macro expands to under
    ``key_vco_3d`` (domzgr_substitute.h90:108), i.e. literally the divisor
    geometry ``trazdf.F90:219-220`` uses -- not a reconstruction.  DINO builds
    full-step ``ln_zco`` (RUN_20Y/namelist_cfg:70), so ``e3w_0`` is constant in
    (i,j); that is CHECKED here rather than assumed.

    Returned aligned to legoESM's interface index: entry ``k`` is the divisor
    between cells ``k`` and ``k+1``, i.e. NEMO's ``e3w_0(k+1)`` (1-indexed
    ``e3w(k)`` spans ``gdept(k-1) -> gdept(k)``).  Length ``nlev-1``.
    """
    if mesh_path not in _E3W_CACHE:
        import netCDF4 as nc
        d = nc.Dataset(mesh_path)
        a = _np(d["e3w_0"][:]).squeeze()
        spread = float(np.abs(a - a[:, :1, :1]).max())
        if spread > 1e-9:
            raise SystemExit(
                f"e3w_0 is not horizontally uniform (spread {spread:.3e} m); "
                "this probe's 1-D reduction is only valid on a full-step "
                "ln_zco mesh")
        _E3W_CACHE[mesh_path] = a[:, 0, 0]
    prof = _E3W_CACHE[mesh_path]
    if prof.shape[0] < nlev:
        raise SystemExit(
            f"mesh e3w_0 has {prof.shape[0]} levels, model has {nlev}")
    return prof[1:nlev]


def wet_interface_mask(state, z_coord):
    """Interfaces where BOTH neighbouring T cells are wet (the only ones the
    solve couples -- everything else is zeroed by the face-activity guards)."""
    surf = _np(state.land_mask.data) > 0.5
    wet = np.broadcast_to(surf[..., None], _np(state.T.data).shape).copy()
    if isinstance(z_coord, OceanPartialCellCoordinate):
        wet &= np.asarray(z_coord.is_active)
    return wet[..., :-1] & wet[..., 1:]


def _stats(rel, mask, label):
    v = rel[mask]
    return {
        "label": label,
        "n": int(v.size),
        "max_abs": float(np.abs(v).max()) if v.size else 0.0,
        "median_abs": float(np.median(np.abs(v))) if v.size else 0.0,
        "mean": float(v.mean()) if v.size else 0.0,
    }


def report_divisor(state, z_coord, config, tag, eta_now=None):
    d_lego, d_nemo = divisor_fields(state, z_coord, config, eta_now=eta_now)
    mask = wet_interface_mask(state, z_coord)
    rel = np.zeros_like(d_nemo)
    good = d_nemo != 0.0
    rel[good] = d_lego[good] / d_nemo[good] - 1.0
    s = _stats(rel, mask, tag)
    print(f"\n[{tag}] D_lego / D_nemo - 1   over {s['n']} wet interfaces")
    print(f"   max|rel| = {s['max_abs']:.4e}   median|rel| = "
          f"{s['median_abs']:.4e}   mean = {s['mean']:+.4e}")
    nlev = rel.shape[-1]
    print("   per-interface (k: n_wet, median rel, max|rel|):")
    for k in range(nlev):
        mk = mask[..., k]
        if not mk.any():
            continue
        vk = rel[..., k][mk]
        print(f"     k={k:2d}  n={int(mk.sum()):6d}  med={np.median(vk):+9.3e}"
              f"  max|{np.abs(vk).max():.3e}|")
    return rel, mask, s


def step_once(model, state, sf, forcing, z_coord, cfg, t_seconds):
    """One PRODUCTION step, the twin's own sequence (kamm_twin_90d:2124-2133).

    Surface forcing first (``leapfrog_rhs`` threads the rate into the step's
    Nnn RHS, ``applied_now`` mutates T/S), then ``model.step``.  Every arm runs
    this identical sequence; only the model's divisor differs.
    """
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
    )
    if getattr(cfg, "surface_tendency_placement", "applied_now") == "leapfrog_rhs":
        st_f, rate = apply_dino_lat_lon_surface_forcing(
            state, forcing, z_coord, cfg, DT, t_seconds=t_seconds,
            return_rate=True)
        return model.step(st_f, DT, surface_forcing=sf,
                          external_tracer_rate=rate)
    st_f = apply_dino_lat_lon_surface_forcing(
        state, forcing, z_coord, cfg, DT, t_seconds=t_seconds)
    return model.step(st_f, DT, surface_forcing=sf)


def field_diffs(a, b, names=("T", "S", "u", "v")):
    out = {}
    for n in names:
        x = _np(getattr(a, n).data)
        y = _np(getattr(b, n).data)
        d = x - y
        out[n] = {"max_abs": float(np.abs(d).max()),
                  "rms": float(np.sqrt((d ** 2).mean()))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    ap.add_argument("--run-traj", default=RUN_TRAJ)
    ap.add_argument("--run-stepdump", default=RUN_STEPDUMP)
    ap.add_argument("--states", default="restart",
                    help="comma list of restart,m0_day360,m0_day7200")
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--days", type=int, default=0,
                    help="also integrate the executing and NEMO-divisor arms "
                         "this many days from the restart and diff the end "
                         "states (0 = one-step only)")
    args = ap.parse_args()

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    print(f"git SHA {sha}")
    print(f"precision policy control dtype = "
          f"{PrecisionPolicy.fp64().control}")

    os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        args.recipe, args.run_traj, args.run_stepdump, e3t_mode="both")
    t0_sec = seasonal_t0_seconds(f"{args.run_stepdump}/{RESTART_FILE}")

    global _E3W_REF
    _E3W_REF = nemo_e3w_ref(f"{args.run_traj}/mesh_mask.nc",
                            int(np.asarray(br.z_coord.dz_ref).shape[-1]))

    print("\n--- RESOLVED CARD (Rule 10: instantiate and print) ---")
    for f in ("implicit_vmix_dzw_slot", "zdf_implicit_solver_evaluation",
              "implicit_vertical_mixing", "outer_integrator"):
        print(f"  {f:34s} = {getattr(mc, f, '<absent>')!r}")
    zc = br.z_coord
    print(f"  z_coord type                       = {type(zc).__name__}")
    for f in ("dz_ref", "dz_half_ref", "z_full_ref", "h_partial"):
        a = getattr(zc, f, None)
        if a is None:
            print(f"  {f:34s} = <absent>")
        else:
            a = np.asarray(a)
            print(f"  {f:34s} dtype={a.dtype} shape={a.shape}")
    for f in ("T", "S", "u", "v", "eta"):
        print(f"  state.{f:29s} dtype={_np(getattr(st, f).data).dtype} "
              f"shape={getattr(st, f).data.shape}")

    # --- is dz_half_ref the same object as the midpoint? (the whole premise) --
    dzr = _np(zc.dz_ref)
    dzh = _np(zc.dz_half_ref)
    mid = _np(build_dz_half(jnp.asarray(dzr)))
    mid_b = np.broadcast_to(mid, dzh.shape) if dzh.shape != mid.shape else mid
    static_rel = mid_b / dzh - 1.0
    print(f"\n  STATIC slot offset  midpoint/dz_half_ref - 1 : "
          f"max|{np.abs(static_rel).max():.4e}|  "
          f"median|{np.median(np.abs(static_rel)):.4e}|")

    # Is the coordinate SELF-CONSISTENT?  dz_half_ref should be the
    # centre-to-centre spacing of z_full_ref (that is what a gradient divisor
    # is).  If it is instead the midpoint 0.5*(dz_k+dz_{k+1}), then NO slot on
    # this ladder carries NEMO's e3w and the dzw flag is a NO-OP.
    zf = _np(zc.z_full_ref)
    ctr = np.broadcast_to(_E3W_REF, dzh.shape).copy()
    ctr_b = np.broadcast_to(ctr, dzh.shape) if ctr.shape != dzh.shape else ctr
    self_rel = dzh / ctr_b - 1.0
    print(f"  SLOT vs NEMO     dz_half_ref/e3w_0 - 1 : "
          f"max|{np.abs(self_rel).max():.4e}|  "
          f"median|{np.median(np.abs(self_rel)):.4e}|")
    mid_vs_ctr = mid_b / ctr_b - 1.0 if mid_b.shape == ctr_b.shape else (
        np.broadcast_to(mid, ctr_b.shape) / ctr_b - 1.0)
    print(f"  TRUE NEMO GAP    midpoint/e3w_0 - 1 : "
          f"max|{np.abs(mid_vs_ctr).max():.4e}|  "
          f"median|{np.median(np.abs(mid_vs_ctr)):.4e}|")

    # --- SHIPPED DIVISOR vs D_NEMO (the receipt row for the arm collapse) ---
    # The card's executing divisor is now produced by the single canonical
    # ``nemo_e3w_kmm`` (trazdf.F90:219-221 / dynzdf.F90:200-203).  Compare it
    # against NEMO's own mesh ``e3w_0`` * (1+r3t) built INDEPENDENTLY here from
    # ``mesh_mask.nc`` and the state's eta -- Rule 10: print the number the
    # claim rests on rather than trusting the constructor's own identity check.
    from legoesm.ocean.physics.vertical_mixing import nemo_e3w_kmm
    from legoesm.ocean.eos import nemo_r3t_stretch
    from legoesm.ocean.vertical import compute_layer_thickness
    _eta = _np(st.eta.data)
    _H = _np(st.H_bathy.data)
    _stretch = nemo_r3t_stretch(zc, jnp.asarray(_eta), jnp.asarray(_H))
    _e3t_now = compute_layer_thickness(
        jnp.asarray(_eta), jnp.asarray(_H), zc,
        min_water_column_m=mc.min_water_column_m)
    _shipped = _np(nemo_e3w_kmm(zc, _e3t_now, _stretch))
    _d_nemo = np.broadcast_to(_E3W_REF, _shipped.shape) * (
        1.0 + np.where(_H > 0.0, _eta / np.where(_H > 0.0, _H, 1.0), 0.0)
    )[..., None]
    _wet = np.broadcast_to(_np(st.land_mask.data)[..., None] > 0.5,
                           _shipped.shape)
    _rel = np.abs(_shipped[_wet] / _d_nemo[_wet] - 1.0)
    print(f"  SHIPPED DIVISOR  nemo_e3w_kmm / (e3w_0*(1+r3t)) - 1 : "
          f"max|{_rel.max():.4e}|  median|{np.median(_rel):.4e}|  "
          f"n={_rel.size}  dtype={_shipped.dtype}")
    print("  per-k  dz_ref   z_full_ref   NEMO_e3w_0   dz_half_ref   midpoint")
    for k in range(min(dzr.shape[-1], 36)):
        c = ctr[..., k].mean() if k < ctr.shape[-1] else float("nan")
        h = dzh[..., k].mean() if k < dzh.shape[-1] else float("nan")
        m = mid[..., k].mean() if k < mid.shape[-1] else float("nan")
        print(f"    k={k:2d} {dzr[..., k].mean():9.3f} {zf[..., k].mean():10.3f}"
              f" {c:10.3f} {h:12.3f} {m:10.3f}")
    results_self = {
        "self_consistency_max_abs": float(np.abs(self_rel).max()),
        "midpoint_vs_center_max_abs": float(np.abs(mid_vs_ctr).max()),
        "midpoint_vs_center_median_abs": float(np.median(np.abs(mid_vs_ctr))),
    }

    results = {"git_sha": sha, "recipe": args.recipe,
               "flags": {f: bool(getattr(mc, f, False)) for f in (
                   "implicit_vmix_dzw_slot",
                   "zdf_implicit_solver_evaluation")},
               "static_slot_offset": {
                   "max_abs": float(np.abs(static_rel).max()),
                   "median_abs": float(np.median(np.abs(static_rel)))},
               "ladder": results_self,
               "states": {}}

    # ---------------- states ----------------
    states = {}
    for name in [s.strip() for s in args.states.split(",") if s.strip()]:
        if name == "restart":
            states[name] = st
            continue
        member, _, day = name.partition("_day")
        npz = np.load(f"{args.archive}/arms/{member}.npz", allow_pickle=False)
        sub = st
        for fld, key in (("T", "T3d"), ("S", "S3d"), ("u", "u3d"),
                         ("v", "v3d")):
            sub = sub._replace(**{fld: getattr(sub, fld).replace(
                data=jnp.asarray(_np(npz[f"{key}_day{day}"]),
                                 dtype=_np(getattr(st, fld).data).dtype))})
        sub = sub._replace(eta=sub.eta.replace(
            data=jnp.asarray(_np(npz[f"eta3d_day{day}"]))))
        states[name] = sub
        print(f"\nloaded {name}: T {npz[f'T3d_day{day}'].shape} "
              f"{npz[f'T3d_day{day}'].dtype}")

    # ---------------- arm construction ----------------
    # ONE VARIABLE (Rule 7).  ``implicit_vmix_dzw_slot`` is the only lever that
    # reads ``dz_half_ref`` inside the solve (omlc:8105 tracer, :8282-8283
    # u/v), so NEMO's divisor = (dz_half_ref := NEMO e3w_0) AND that flag.
    # ``dz_half_ref`` is ALSO read outside the divisor block, so every arm
    # ships a CONTAMINATION CONTROL: the same modified ladder with the flag
    # OFF.  Divisor effect = arm - its control; whatever the control moves is
    # not the divisor.
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    mc_slot = mc._replace(implicit_vmix_dzw_slot=True)
    zc_nemo = zc._replace(dz_half_ref=jnp.asarray(_E3W_REF))
    zc_1pct = zc._replace(dz_half_ref=jnp.asarray(_np(zc.dz_half_ref) * 1.01))
    # Total ablation of the implicit vertical mixing (Rule 3's floor): the
    # implicit flux is K/D, so D -> 1e6 D switches the term off without
    # touching K, the state, or any other operator.
    zc_off = zc._replace(dz_half_ref=jnp.asarray(_np(zc.dz_half_ref) * 1e6))

    def mk(coord, conf):
        return LatLonCGridOceanModel(br.geometry, coord, conf)

    arms = {
        "A_exec": mk(zc, mc),            # executing midpoint divisor
        "B_nemo": mk(zc_nemo, mc_slot),  # NEMO e3w_0 divisor
        "B_ctl": mk(zc_nemo, mc),        # contamination control for B
        "S_noop": mk(zc, mc_slot),       # flag on, ladder unchanged (no-op)
        "P_1pct": mk(zc_1pct, mc_slot),  # planted +1% divisor
        "P_ctl": mk(zc_1pct, mc),        # its contamination control
        "Z_abl": mk(zc_off, mc_slot),    # vertical mixing ablated
        "Z_ctl": mk(zc_off, mc),         # its contamination control
    }

    t_sec = t0_sec + DT

    def run(state, keys):
        return {k: step_once(arms[k], state, sf, forcing, zc, cfg, t_sec)
                for k in keys}

    # ---------------- calibration (Rule 3) ----------------
    print("\n=== CALIBRATION (before any residual is quoted) ===")
    st0 = states[list(states)[0]]
    r0 = run(st0, ("A_exec", "S_noop", "P_1pct", "P_ctl", "Z_abl", "Z_ctl"))

    c1 = field_diffs(r0["A_exec"],
                     step_once(arms["A_exec"], st0, sf, forcing, zc, cfg, t_sec))
    print(f"  C1  repeat of the same arm        max|dT| = "
          f"{c1['T']['max_abs']:.3e}   (must be EXACTLY 0)")
    assert all(v["max_abs"] == 0.0 for v in c1.values()), "C1 FAILED"

    c1b = field_diffs(r0["A_exec"], r0["S_noop"])
    print(f"  C1b slot flag, ladder unchanged   max|dT| = "
          f"{c1b['T']['max_abs']:.3e}   (no-op: dz_half_ref == midpoint)")

    d_l0, d_n0 = divisor_fields(st0, zc, mc)
    bumped = st0._replace(eta=st0.eta.replace(
        data=st0.eta.data + 1.0 * (_np(st0.land_mask.data) > 0.5)))
    d_l1, d_n1 = divisor_fields(bumped, zc, mc)
    msk = wet_interface_mask(st0, zc)
    rr0 = np.where(d_n0 != 0, d_l0 / np.where(d_n0 != 0, d_n0, 1.0), 0.0)
    rr1 = np.where(d_n1 != 0, d_l1 / np.where(d_n1 != 0, d_n1, 1.0), 0.0)
    c2 = float(np.abs(rr1 - rr0)[msk].max())
    print(f"  C2  D ratio under a +1 m eta bump = {c2:.3e}"
          f"   (J cancels analytically -> ~0)")

    # C4: the ablation arm must remove a FIRST-ORDER amount, else it is not a
    # usable denominator.
    abl = field_diffs(r0["A_exec"], r0["Z_abl"])
    abl_c = field_diffs(r0["A_exec"], r0["Z_ctl"])
    print(f"  C4  vmix ablation (D x 1e6)       max|dT| = "
          f"{abl['T']['max_abs']:.3e} K   contamination "
          f"{abl_c['T']['max_abs']:.3e} K")

    # C3: a planted UNIFORM +1% divisor.  Backward-Euler flux is K/D, so the
    # vmix increment should shrink by ~1% of itself.
    p = field_diffs(r0["S_noop"], r0["P_1pct"])
    p_c = field_diffs(r0["A_exec"], r0["P_ctl"])
    c3 = p["T"]["max_abs"] / max(abl["T"]["max_abs"], 1e-300)
    print(f"  C3  planted +1% divisor           max|dT| = "
          f"{p['T']['max_abs']:.3e} K  /  ablation "
          f"{abl['T']['max_abs']:.3e} K  = {c3:.4f}   (predicted ~0.01)")
    print(f"      C3 contamination control        max|dT| = "
          f"{p_c['T']['max_abs']:.3e} K")
    results["calibration"] = {
        "C1_repeat_max_dT": c1["T"]["max_abs"],
        "C1b_slot_noop_max_dT": c1b["T"]["max_abs"],
        "C2_ratio_shift_1m_eta": c2,
        "C3_planted_1pct_over_ablation": c3,
        "C3_contamination_max_dT": p_c["T"]["max_abs"],
        "C4_ablation": abl,
        "C4_ablation_contamination": abl_c,
    }

    # ---------------- per-state measurement ----------------
    for name, state in states.items():
        print(f"\n================ STATE {name} ================")
        rel, mask, sstat = report_divisor(state, zc, mc, name)

        r = (r0 if state is st0
             else run(state, ("A_exec", "B_nemo", "B_ctl", "Z_abl")))
        if "B_nemo" not in r:
            r = dict(r, **run(state, ("B_nemo", "B_ctl")))
        a, b, bc = r["A_exec"], r["B_nemo"], r["B_ctl"]
        z = r["Z_abl"]
        d_ab = field_diffs(a, b)
        d_contam = field_diffs(a, bc)
        d_abl = field_diffs(a, z)
        print("  contamination control (ladder changed, flag OFF): "
              + "  ".join(f"{k}={v['max_abs']:.2e}"
                          for k, v in d_contam.items()))

        entry = {"divisor_rel": sstat, "arm_diff": d_ab,
                 "contamination": d_contam, "vmix_ablation": d_abl,
                 "ratios": {}}
        for fld in ("T", "S", "u", "v"):
            x0 = _np(getattr(state, fld).data)
            tot = float(np.abs(_np(getattr(a, fld).data) - x0).max())
            vm = d_abl[fld]["max_abs"]
            arm = d_ab[fld]["max_abs"]
            entry["ratios"][fld] = {
                "arm_diff_max": arm,
                "arm_diff_rms": d_ab[fld]["rms"],
                "vmix_ablation_max": vm,
                "total_step_incr_max": tot,
                "arm_over_vmix": arm / vm if vm else float("nan"),
                "arm_over_total": arm / tot if tot else float("nan"),
            }
            print(f"  {fld}: |arm|={arm:.4e}  |vmix|={vm:.4e}"
                  f"  |total step|={tot:.4e}"
                  f"  (i)={arm / vm if vm else float('nan'):.4e}"
                  f"  (ii)={arm / tot if tot else float('nan'):.4e}")
        results["states"][name] = entry

    # ---------------- integrated arm (preregistered) ----------------
    if args.days > 0:
        import time as _time
        import jax
        steps = 32 * args.days
        print(f"\n=== INTEGRATED ARM: {args.days} day(s) = {steps} steps ===")

        def jstep(m):
            if getattr(cfg, "surface_tendency_placement",
                       "applied_now") == "leapfrog_rhs":
                return jax.jit(lambda s_, e_: m.step(
                    s_, DT, surface_forcing=sf, external_tracer_rate=e_))
            return jax.jit(lambda s_: m.step(s_, DT, surface_forcing=sf))

        from legoesm.ocean.experiments.dino import (
            apply_dino_lat_lon_surface_forcing,
        )
        _rhs = getattr(cfg, "surface_tendency_placement",
                       "applied_now") == "leapfrog_rhs"
        traj = {}
        for key in ("A_exec", "B_nemo", "B_ctl"):
            dyn = jstep(arms[key])
            s_ = states["restart"]
            t_ = _time.time()
            for k in range(steps):
                ts = t0_sec + (k + 1) * DT
                if _rhs:
                    s_, rate = apply_dino_lat_lon_surface_forcing(
                        s_, forcing, zc, cfg, DT, t_seconds=ts,
                        return_rate=True)
                    s_ = dyn(s_, rate)
                else:
                    s_ = apply_dino_lat_lon_surface_forcing(
                        s_, forcing, zc, cfg, DT, t_seconds=ts)
                    s_ = dyn(s_)
            traj[key] = s_
            print(f"  {key}: {steps} steps in {_time.time() - t_:.0f}s  "
                  f"finite={bool(np.isfinite(_np(s_.T.data)).all())}")
        d_int = field_diffs(traj["A_exec"], traj["B_nemo"])
        d_int_c = field_diffs(traj["A_exec"], traj["B_ctl"])
        print(f"  day-{args.days} arm difference (executing vs NEMO divisor):")
        for k, v in d_int.items():
            print(f"    {k}: max={v['max_abs']:.4e}  rms={v['rms']:.4e}")
        print("  contamination control: "
              + "  ".join(f"{k}={v['max_abs']:.2e}"
                          for k, v in d_int_c.items()))
        one = results["states"]["restart"]["arm_diff"]
        print(f"  growth vs one step, T: "
              f"{d_int['T']['max_abs'] / one['T']['max_abs']:.1f}x")
        results["integrated"] = {
            "days": args.days, "steps": steps,
            "arm_diff": d_int, "contamination": d_int_c,
            "growth_T_vs_one_step":
                d_int["T"]["max_abs"] / one["T"]["max_abs"],
        }

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
