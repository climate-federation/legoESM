#!/usr/bin/env python
"""S17 (``dyn_zdf``) BRACKET -- feed legoESM's implicit vertical-momentum solve
NEMO's OWN pre-stage input and compare against NEMO's post-stage dump.

WHY THIS EXISTS / WHAT IT RETRACTS
----------------------------------
The committed ``dyn_zdf_probe.py`` concludes (docstring + printed VERDICT):

    "NO NEMO DUMP EXISTS that captures the true pre-dyn_zdf Krhs (stage 6 is
     BEFORE dyn_spg's own ... corrections; no dump brackets Krhs AFTER those
     additions)"

That is **WRONG, and this script retracts it.**  ``Nrhs = Naa``
(nemogcm.F90:396 -- quoted by that same docstring), so
``stp_dump_state_and_bt(kstp,7,'dynspg',uu(:,:,:,Naa),...)`` at stpmlf.F90:293
dumps ``uu(:,:,:,Nrhs)`` **after ``dyn_spg`` returns and before ``dyn_zdf``
runs**: it IS the true pre-dyn_zdf Krhs.  The probe got the first half right
(stage 7 is a tendency, not a velocity) and then drew the opposite conclusion
from it.

Proof that stage 7 is exactly ``Krhs`` after dyn_spg's own additions, from
NEMO's numbers alone (``s14_s17_nemo_arith.py``, run at RUN_GDB kt=57601):

  * BAROCLINIC part of ``uu(Nrhs)``: stage6 -> stage7 relative change
    **8.07e-16** (bit-identical) -- exactly what dynspg_ts.F90:351
    (``puu(Krhs) -= zu_frc``, a depth-INDEPENDENT per-column shift) predicts.
  * DEPTH MEAN of ``uu(Nrhs)``: RMS 3.77e-06 -> 6.43e-10, ratio 1.70e-04 --
    the mean is removed at :351 and only ``(uu_b(Kaa)-uu_b(Kbb))*r1_Dt`` is
    added back at :1124 (``ln_dynadv_vec=.true.``, namelist_cfg:321).

A "mislabelled non-state" cannot stand in that exact algebraic relation to
stage 6.

THE BRACKET
-----------
NEMO's own pre-solve velocity, reconstructed from NEMO quantities only
(dynzdf.F90:138-140 then :167-170, ``ln_dynadv_vec=.true.``,
``ln_drgimp=.true.`` namelist_ref:817, ``ln_dynspg_ts=.true.``
namelist_cfg:352)::

    naa_A = ( uu(Kbb) + rDt * uu(Krhs) ) * umask          ! dynzdf.F90:139
    naa_B = ( naa_A - uu_b(Kaa) )        * umask          ! dynzdf.F90:168

``uu(Kbb)`` = restart ``ub``/``vb`` (restart.F90:357-358 reads ``ub`` into
Kbb under MLF), ``uu(Krhs)`` = stage-7 dump, ``uu_b(Kaa)`` = the
``stp_dump_07_dynspg_ub`` companion dump.

CONTROL C1 (validates the reconstruction AND pins rDt): ``dyn_zdf`` only
changes ``naa_B`` through the vertical-mixing tridiagonal + the bottom-drag
diagonal, so ``naa_B`` must already be close to stage-8 in the weakly-mixed
interior.  The script scans ``rDt in {rn_Dt, 2*rn_Dt}`` and reports both; the
wrong one is off by an O(1) factor and the check fails loudly.

ARMS (one variable each)
  self   legoESM's own pre-solve u,v -> its own solve  (== dyn_zdf_probe's
         end-to-end number; reproduced here as a cross-check)
  null   the substitution hook installed but substituting legoESM's OWN
         arrays.  CONTROL: must be BIT-IDENTICAL to ``self``.
  fed    NEMO's ``naa_B`` substituted -> legoESM's solve.  **This is S17
         ISOLATED**: its residual vs stage-8 is created by the solve alone.

Every comparison is additionally split into DEPTH-MEAN and BAROCLINIC parts
(weights ``e3u_0*umask`` from ``mesh_mask.nc``), because NEMO's stage-8 is a
nearly depth-mean-FREE field by construction (:168 subtracts ``uu_b(Kaa)``
and ``mlf_baro_corr``, stpmlf.F90:566-645, re-adds it only LATER) while
legoESM's post-solve velocity need not be.  A total err_norm that does not
say which of the two it is cannot separate S14 (the barotropic solve) from
S17 (the vertical solve).

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.s17_dynzdf_bracket
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import dataclasses

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.precision_gate import (
    require_fp64, require_explicit_e3t_mode,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing,
)

from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    RUN_DIR, DT, _load_full_3d, _load_full,
)
from scripts.validate.ocean_fidelity.dino_1226.bn2_alpha_compare import _read_dims

set_policy(PrecisionPolicy.fp64())

RESTART = "DINO_00057600_restart.nc"


# --------------------------------------------------------------------------
# substitution hook: replace the implicit solve's INPUT u,v
# --------------------------------------------------------------------------
_SUB: dict[str, object] = {"u": None, "v": None, "n_calls": 0}


def _run_with_hook(model, st, sf):
    """Return (pre_u, pre_v, post_u, post_v) around the do_momentum=True call.

    When ``_SUB['u']`` is set, the captured PRE is replaced by that array
    before the real routine runs, so the solve sees NEMO's own input.
    """
    captured: dict[str, np.ndarray] = {}
    model_cls = type(model)
    _real = model_cls._apply_implicit_vertical_mixing

    def _spy(self, state, dt, surface_forcing, *args, **kwargs):
        if kwargs.get("do_momentum", True) and "pre_u" not in captured:
            _SUB["n_calls"] = int(_SUB["n_calls"]) + 1
            if _SUB["u"] is not None:
                import jax.numpy as jnp
                state = state._replace(
                    u=state.u.replace(data=jnp.asarray(_SUB["u"])),
                    v=state.v.replace(data=jnp.asarray(_SUB["v"])),
                )
            captured["pre_u"] = np.asarray(state.u.data)
            captured["pre_v"] = np.asarray(state.v.data)
        result = _real(self, state, dt, surface_forcing, *args, **kwargs)
        if kwargs.get("do_momentum", True) and "post_u" not in captured:
            out = result[0] if isinstance(result, tuple) else result
            captured["post_u"] = np.asarray(out.u.data)
            captured["post_v"] = np.asarray(out.v.data)
        return result

    model_cls._apply_implicit_vertical_mixing = _spy
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        model_cls._apply_implicit_vertical_mixing = _real
    assert {"pre_u", "pre_v", "post_u", "post_v"} <= captured.keys()
    return captured


# --------------------------------------------------------------------------
def _split(x, w, mask):
    """(depth-mean broadcast, baroclinic) with NEMO's e3*_0*mask weights."""
    hh = (w * mask).sum(-1)
    mean = np.where(hh > 0, (w * x * mask).sum(-1) / np.where(hh > 0, hh, 1.0), 0.0)
    return np.broadcast_to(mean[..., None], x.shape), x - mean[..., None]


def _en(lego, nemo, mask):
    m = mask & np.isfinite(lego) & np.isfinite(nemo)
    lo, ne = lego[m], nemo[m]
    rms = float(np.sqrt(np.mean(ne ** 2))) if ne.size else float("nan")
    corr = (float(np.corrcoef(lo, ne)[0, 1])
            if lo.size >= 2 and lo.std() > 0 and ne.std() > 0 else float("nan"))
    return (float(np.sqrt(np.mean((lo - ne) ** 2))) / rms if rms > 0 else float("nan"),
            rms, float(np.sqrt(np.mean(lo ** 2))),
            float(np.max(np.abs(lo - ne))) if lo.size else float("nan"), corr)


def _report(tag, lego, nemo, mask, w):
    lm, lb = _split(lego, w, mask)
    nm, nb = _split(nemo, w, mask)
    for name, (a, b) in (("total   ", (lego, nemo)),
                         ("depthmean", (lm, nm)),
                         ("barocl. ", (lb, nb))):
        e, rn, rl, mx, cc = _en(a, b, mask)
        print(f"    {tag:<26s} {name:<10s} err_norm={e:.4e}  corr={cc:9.6f}  "
              f"RMS(nemo)={rn:.4e}  RMS(lego)={rl:.4e}  max|d|={mx:.4e}")


def main() -> int:
    require_explicit_e3t_mode(context="s17_dynzdf_bracket")
    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"recipe nemo_dino_kamm_mlf: outer_integrator={dcfg.outer_integrator!r} "
          f"zdf_drag_in_matrix={dcfg.zdf_drag_in_matrix} "
          f"surface_stress_implicit="
          f"{getattr(dcfg, 'surface_stress_implicit', None)!r} "
          f"barotropic_solver={getattr(dcfg, 'barotropic_solver', None)!r}")

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, now, periodic_i=True, full_step=True,
                                     omega=dcfg.omega)
    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0,
                              sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.z_coord, br.state.T.data, br.state.S.data,
                 br.geometry.dx_u, context="s17_dynzdf_bracket")
    print(f"  dtypes: state.u={br.state.u.data.dtype} "
          f"dx_u={br.geometry.dx_u.dtype} nemo ub={bef.u.dtype} "
          f"e3u_0={np.asarray(g.e3u_0).dtype if hasattr(g,'e3u_0') else 'n/a'}")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    L3 = lambda n: _load_full_3d(os.path.join(RUN_DIR, n), jpi, jpj, jpk - 1, hls)
    L2 = lambda n: _load_full(os.path.join(RUN_DIR, n), jpi, jpj, hls)
    krhs_u, krhs_v = L3("stp_dump_07_dynspg_u.bin"), L3("stp_dump_07_dynspg_v.bin")
    st8_u, st8_v = L3("stp_dump_08_dynzdf_u.bin"), L3("stp_dump_08_dynzdf_v.bin")
    uub, vvb = L2("stp_dump_07_dynspg_ub.bin"), L2("stp_dump_07_dynspg_vb.bin")

    umask3 = np.asarray(g.umask) > 0.5
    vmask3 = np.asarray(g.vmask) > 0.5
    # NEMO's own e3u_0/e3v_0 for the depth-mean weights (mesh_mask, halo-free)
    from netCDF4 import Dataset
    with Dataset(os.path.join(RUN_DIR, "mesh_mask.nc")) as nc:
        e3u_0 = np.moveaxis(np.asarray(nc.variables["e3u_0"][:]).squeeze(), 0, -1)
        e3v_0 = np.moveaxis(np.asarray(nc.variables["e3v_0"][:]).squeeze(), 0, -1)
    print(f"  shapes: krhs_u{krhs_u.shape} st8_u{st8_u.shape} "
          f"umask3{umask3.shape} e3u_0{e3u_0.shape} ub{bef.u.shape}")

    nk = krhs_u.shape[-1]                       # 35 = jpkm1
    cu = (slice(None), slice(0, umask3.shape[1]), slice(0, nk))
    um = umask3[..., :nk]
    vm = vmask3[..., :nk]
    wu, wv = e3u_0[..., :nk], e3v_0[..., :nk]

    # ---------------- CONTROL C1: BIT-EXACT reconstruction -----------------
    # RETRACTION (Rule 11, recorded in place): two earlier versions of this
    # control compared ``naa_B`` against stage-8 (the POST-solve dump) and
    # "pinned" rDt=2700 s, contradicting NEMO's own ocean.output:249
    # ("Modified Leap-Frog (MLF): rDt = 5400.0000000000000") and :259
    # ("ln_1st_euler = F").  Both were measuring the wrong thing: stage-8 is
    # separated from naa_B by the tridiagonal solve, whose effect is the same
    # size as the rDt term.  The correct control is NEMO's OWN dump of the
    # quantity being reconstructed: dynzdf.F90:347-350 stores
    # ``z1226_u1_pre(ji,jj) = puu(ji,jj,1,Kaa)`` -- the level-1 value AFTER
    # :139 (+rDt*Krhs), AFTER :168 (-uu_b(Kaa)) and BEFORE the surface-stress
    # add -- into ``zdf_dump_u1_prestress.bin``.
    print("\n=== CONTROL C1: naa_B vs NEMO's own zdf_dump_u1_prestress ===")
    pre1_u = np.fromfile(os.path.join(RUN_DIR, "zdf_dump_u1_prestress.bin"),
                         dtype="<f8").reshape(um.shape[0], um.shape[1])
    pre1_v = np.fromfile(os.path.join(RUN_DIR, "zdf_dump_v1_prestress.bin"),
                         dtype="<f8").reshape(vm.shape[0], vm.shape[1])
    m1u, m1v = um[..., 0], vm[..., 0]
    best = None
    for rdt in (0.0, DT, 2.0 * DT):
        rec = (bef.u[..., 0] + rdt * krhs_u[..., 0] - uub) * m1u
        d = rec[m1u] - pre1_u[m1u]
        rmsd = float(np.sqrt(np.mean(d ** 2)))
        print(f"    rDt={rdt:8.1f}  RMS(rec-nemo)={rmsd:.4e}  "
              f"max|d|={np.abs(d).max():.4e}  "
              f"corr={np.corrcoef(rec[m1u], pre1_u[m1u])[0,1]:.10f}")
        if best is None or rmsd < best[0]:
            best = (rmsd, rdt)
    rDt = 2.0 * DT
    if abs(best[1] - rDt) > 1e-9 or best[0] > 1e-14:
        raise SystemExit(
            f"C1 FAILED: level-1 reconstruction picks rDt={best[1]} with "
            f"residual {best[0]:.3e}; NEMO's ocean.output declares 5400.0 and "
            f"the residual must be 0 (Rule 1e -- reconcile, do not proceed).")
    print(f"  -> C1 PASS: naa_B reproduces NEMO's level-1 pre-stress state "
          f"BIT-EXACTLY at rDt={rDt:.1f} s.  stage-7 IS the pre-dyn_zdf Krhs; "
          "dyn_zdf_probe.py's 'no dump exists' verdict is RETRACTED.")
    dv1 = ((bef.v[..., 0] + rDt * krhs_v[..., 0] - vvb) * m1v)[m1v] - pre1_v[m1v]
    print(f"     v-component level-1 residual max|d|={np.abs(dv1).max():.3e}")

    naaA_u = (bef.u[..., :nk] + rDt * krhs_u) * um
    naaA_v = (bef.v[..., :nk] + rDt * krhs_v) * vm
    naaB_u = (naaA_u - uub[..., None]) * um
    naaB_v = (naaA_v - vvb[..., None]) * vm

    # ---------------- arm 'self' (must run first: the scan needs it) ------
    results = {}
    _SUB["u"] = _SUB["v"] = None
    results["self"] = _run_with_hook(model, br.state, sf)

    # ---- ALIGNMENT SELF-CHECK (mandatory before any pre/post number) -----
    # dyn_zdf_probe.py crops legoESM's (199,53,36) u / (200,52,36) v from the
    # FRONT (``a[:n0,:n1,:n2]``).  carry_injection_discriminator.py:433-434
    # instead drops the FIRST lon column of u and the FIRST lat row of v
    # (``st.u.data[:,1:,:]`` / ``st.v.data[1:,:,:]``).  Those two cannot both
    # be right.  Scan and take the sharp minimum; a flat scan means the
    # comparison is not resolving alignment at all and no number is readable.
    print("\n=== ALIGNMENT SCAN: lego pre_u vs naa_A over lon-offset ===")
    best_o = None
    for o in (0, 1):
        cu = results["self"]["pre_u"][:, o:o + um.shape[1], :nk]
        e, rn, rl, mx, cc = _en(cu, naaA_u, um)
        print(f"    u lon-offset {o}: err_norm={e:.4e} corr={cc:.6f} max|d|={mx:.3e}")
        if best_o is None or e < best_o[0]:
            best_o = (e, o)
    OU = best_o[1]
    best_p = None
    for o in (0, 1):
        cv = results["self"]["pre_v"][o:o + vm.shape[0], :vm.shape[1], :nk]
        e, rn, rl, mx, cc = _en(cv, naaA_v, vm)
        print(f"    v lat-offset {o}: err_norm={e:.4e} corr={cc:.6f} max|d|={mx:.3e}")
        if best_p is None or e < best_p[0]:
            best_p = (e, o)
    OV = best_p[1]
    print(f"  -> using u lon-offset {OU}, v lat-offset {OV}")

    def crop_u(a):
        return a[:, OU:OU + um.shape[1], :nk]

    def crop_v(a):
        return a[OV:OV + vm.shape[0], :vm.shape[1], :nk]

    # ---------------- remaining arms --------------------------------------
    # NEMO's OWN surface-stress deposit, taken from its own dumps:
    # dynzdf.F90:355-359 adds zDt_2*(utau_b+utauU)/(e3u(1,Kaa)*rho0) to
    # puu(:,:,1,Kaa) between the prestress and poststress captures.  It is an
    # addition to the tridiagonal RHS, so adding it to the solve's INPUT is
    # arithmetically the same thing.
    ws_u = (np.fromfile(os.path.join(RUN_DIR, "zdf_dump_u1_poststress.bin"),
                        dtype="<f8").reshape(um.shape[0], um.shape[1])
            - np.fromfile(os.path.join(RUN_DIR, "zdf_dump_u1_prestress.bin"),
                          dtype="<f8").reshape(um.shape[0], um.shape[1]))
    ws_v = (np.fromfile(os.path.join(RUN_DIR, "zdf_dump_v1_poststress.bin"),
                        dtype="<f8").reshape(vm.shape[0], vm.shape[1])
            - np.fromfile(os.path.join(RUN_DIR, "zdf_dump_v1_prestress.bin"),
                          dtype="<f8").reshape(vm.shape[0], vm.shape[1]))
    print(f"\n  NEMO wind-stress deposit at level 1: RMS(u)="
          f"{float(np.sqrt(np.mean(ws_u[um[...,0]]**2))):.4e}  "
          f"RMS(v)={float(np.sqrt(np.mean(ws_v[vm[...,0]]**2))):.4e}")

    for arm in ("null", "fed", "fed_ws"):
        pu = results["self"]["pre_u"].copy()
        pv = results["self"]["pre_v"].copy()
        if arm != "null":
            pu[:, OU:OU + um.shape[1], :nk] = naaB_u
            pv[OV:OV + vm.shape[0], :vm.shape[1], :nk] = naaB_v
            if arm == "fed_ws":
                pu[:, OU:OU + um.shape[1], 0] += ws_u
                pv[OV:OV + vm.shape[0], :vm.shape[1], 0] += ws_v
        _SUB["u"], _SUB["v"] = pu, pv
        results[arm] = _run_with_hook(model, br.state, sf)
        _SUB["u"] = _SUB["v"] = None

    print(f"\n[hook] substitution calls = {_SUB['n_calls']} (expect 4)")
    if _SUB["n_calls"] != 4:
        raise SystemExit("HOOK MISFIRED -- arms unreadable")
    d = np.abs(results["null"]["post_u"] - results["self"]["post_u"]).max()
    print(f"[CONTROL null] max|post_u(null)-post_u(self)| = {d:.3e}  "
          f"{'OK (bit-identical)' if d == 0.0 else '*** HOOK PERTURBS ***'}")
    if d != 0.0:
        raise SystemExit("null control failed -- fed arms unreadable")

    print("\n=== PRE-SOLVE input: legoESM vs NEMO's own (S14 + upstream) ===")
    _report("lego pre_u vs naa_A", crop_u(results["self"]["pre_u"]), naaA_u, um, wu)
    # WIND-STRESS PLACEMENT CONTROL.  The card runs
    # surface_stress_implicit=False (dino.py), so legoESM deposits the wind
    # stress in the EXPLICIT RHS while NEMO deposits it INSIDE dyn_zdf
    # (dynzdf.F90:355-359) -- i.e. naa_A does NOT contain it and legoESM's
    # pre-solve state DOES.  Adding NEMO's own deposit to naa_A makes the
    # comparison group-fair.  The v-component is the built-in falsifier:
    # DINO's wind is purely zonal, so RMS(ws_v)=0 and v must show NO such gap.
    naaA_ws = naaA_u.copy()
    naaA_ws[..., 0] += ws_u
    _report("lego pre_u vs naa_A+ws", crop_u(results["self"]["pre_u"]),
            naaA_ws, um, wu)
    lpu = crop_u(results["self"]["pre_u"])
    print("    per-level err_norm(lego pre_u vs naa_A) k=0..6, then k=10,20,30:")
    for k in (0, 1, 2, 3, 4, 5, 6, 10, 20, 30):
        e, rn, rl, mx, cc = _en(lpu[..., k:k+1], naaA_u[..., k:k+1],
                                um[..., k:k+1])
        e2, _, _, _, _ = _en(lpu[..., k:k+1], naaA_ws[..., k:k+1],
                             um[..., k:k+1])
        print(f"      k={k:2d}  vs naa_A={e:.4e}   vs naa_A+ws={e2:.4e}   "
              f"RMS(nemo)={rn:.4e}")
    _report("lego pre_u vs naa_B", crop_u(results["self"]["pre_u"]), naaB_u, um, wu)
    _report("lego pre_v vs naa_B", crop_v(results["self"]["pre_v"]), naaB_v, vm, wv)

    print("\n=== POST-SOLVE vs NEMO stage-8 ===")
    print("  arm 'self' = end-to-end (inherits everything upstream)")
    _report("self post_u", crop_u(results["self"]["post_u"]), st8_u, um, wu)
    _report("self post_v", crop_v(results["self"]["post_v"]), st8_v, vm, wv)
    print("  arm 'fed'  = NEMO's naa_B in, WITHOUT NEMO's in-dyn_zdf wind "
          "stress (CONFOUNDED: the card runs surface_stress_implicit=False, "
          "so legoESM never adds it back -- read fed_ws instead)")
    _report("fed  post_u", crop_u(results["fed"]["post_u"]), st8_u, um, wu)
    _report("fed  post_v", crop_v(results["fed"]["post_v"]), st8_v, vm, wv)
    print("  arm 'fed_ws' = S17 ISOLATED, GROUP-fair (naa_B + NEMO's own "
          "level-1 stress deposit in)")
    _report("fed_ws post_u", crop_u(results["fed_ws"]["post_u"]), st8_u, um, wu)
    _report("fed_ws post_v", crop_v(results["fed_ws"]["post_v"]), st8_v, vm, wv)
    print("  reference: naa_B itself vs stage-8 (= 'no solve at all')")
    _report("naa_B (no solve) u", naaB_u, st8_u, um, wu)
    _report("naa_B (no solve) v", naaB_v, st8_v, vm, wv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
