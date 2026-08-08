#!/usr/bin/env python
"""S17 PHASE-2 CONFIRM: which dyn_zdf COEFFICIENT owns the levels 1-6 residual?

Extends ``s17_dynzdf_bracket.py`` (which feeds NEMO's exact pre-solve u,v and
still leaves err_norm 3.3e-02 (u) / 1.7e-02 (v)).  That bracket substitutes the
solve's *input state* only; the solve's *coefficients* stay legoESM's own.  The
Phase-1 alignment table (dynzdf.F90 read end to end under DINO's resolved
namelist) named exactly two live DIFF rows in the tridiagonal build:

  D1  e3uw divisor.  NEMO dynzdf.F90:183/185 divides by ``e3uw(ji,jj,jk,Kmm)``
      which under key_qco/key_vco_3d expands (domzgr_substitute.h90:130) to
      ``e3uw_3d(i,j,k)*(1+r3u(i,j,Kmm))``.  ``e3uw_3d`` == ``e3w_1d`` on DINO's
      full-step zco grid (verified from mesh_mask.nc: max|e3uw_0-e3w_1d| = 0 on
      wet points).  legoESM builds the same divisor as
      ``build_dz_half`` = 0.5*(dz_k+dz_{k+1}) (implicit_solver.py:480).
      MEASURED from mesh_mask.nc: e3w_1d is NOT that midpoint -- it deviates by
      +0.07% at k=0 rising to 0.9% at depth (analytic z-stretch evaluated
      pointwise vs. averaged).

  D2  avm.  The bracket feeds NEMO's u,v but the viscosity is legoESM's own TKE
      closure output, whose recorded gate row ("zdftke composite avt/avm") is
      corr 0.99791 / ratio 0.9965 -- DEBT, and it is an INPUT to this solve.
      NEMO's own ``avm`` as dyn_zdf sees it is dumped: ldftra.F90:956 writes
      ``avm(ji,jj,jk)`` at stpmlf.F90:203, which is AFTER zdf_phy (:190,
      closure + zdf_evd) and BEFORE dyn_zdf (:305); nothing between those two
      lines touches avm.  So ``dump_avm.bin`` IS dyn_zdf's avm.

ARMS (one variable each, all stacked on the bracket's ``fed_ws`` state so the
known level-0 wind-stress PLACEMENT difference is neutralised):

  fed_ws     NEMO u,v + NEMO's own level-1 stress deposit; legoESM coefficients
  +avm       ... and NEMO's avm substituted at the K-PROFILE level (so
             legoESM's own seafloor masking + cell->face interpolation still
             run: a true one-variable change)
  +e3uw      ... and the midpoint dz_half replaced by NEMO's e3w profile,
             holding the eta stretch fixed (dz_half *= e3uw_0[k+1]/midpoint)
  +both      both substitutions

CONTROLS RUN (all must pass or no arm is readable):
  K1  wrapper installed, substituting NOTHING       -> bit-identical to fed_ws
  K2  wrapper substituting its OWN captured arrays  -> bit-identical to fed_ws
  K3  T-grid alignment: legoESM A_v_cell vs NEMO avm must correlate at the
      recorded gate value (~0.998); a scan over the level offset must peak
      SHARPLY at the transcribed offset (+1), not be flat
  K4  hook fire counts printed and asserted (a hook that fires 0 times has
      burned this campaign before)
  K5  dtypes of every comparand printed (fp64 gate)

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.s17_coeff_arms
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import dataclasses

import numpy as np
import jax.numpy as jnp
import netCDF4

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.precision_gate import (
    require_fp64, require_explicit_e3t_mode,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_nemo_to_legoesm_topo, bridge_before_state_topo,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing,
)

from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    RUN_DIR, DT, _load_full_3d, _load_full,
)
from scripts.validate.ocean_fidelity.dino_1226.bn2_alpha_compare import _read_dims
from scripts.validate.ocean_fidelity.dino_1226.s17_dynzdf_bracket import (
    _SUB, _run_with_hook, _en, _report, RESTART,
)

set_policy(PrecisionPolicy.fp64())

# ---------------------------------------------------------------------------
# coefficient substitution hooks
# ---------------------------------------------------------------------------
_COEF: dict[str, object] = {
    "A_v_cell": None,     # (n_lat, n_lon, nlev-1) NEMO avm at legoESM interfaces
    "dzh_scale": None,    # (nlev-1,) multiplicative e3w/midpoint correction
    "n_kprof": 0,
    "n_solve": 0,
    "echo": False,        # K2: re-inject the wrapper's own captured arrays
    "seen": {},
}


def _install_hooks():
    """Patch the two module attributes the model re-imports on every call."""
    import legoesm.ocean.physics.vertical_mixing as VM

    real_kprof = VM.compute_vertical_K_profiles
    real_solve = VM.implicit_vertical_diffusion_ocean

    def kprof(*a, **kw):
        out = real_kprof(*a, **kw)
        _COEF["n_kprof"] = int(_COEF["n_kprof"]) + 1
        A_sub = _COEF["A_v_cell"]
        if A_sub is None:
            return out
        # returns (K_v, A_v) or (K_v, A_v, tke)
        lst = list(out)
        _COEF["seen"]["A_v_cell_lego"] = np.asarray(lst[1])
        lst[1] = jnp.asarray(A_sub, dtype=lst[1].dtype)
        return tuple(lst)

    def solve(field, K, dz, dz_half, dt, *, extra_diag=0.0):
        _COEF["n_solve"] = int(_COEF["n_solve"]) + 1
        tag = "u" if field.shape[1] > field.shape[0] else "v"
        _COEF["seen"][f"dzh_{tag}"] = np.asarray(dz_half)
        _COEF["seen"][f"Av_{tag}"] = np.asarray(K)
        if _COEF["echo"]:
            # K2: feed back byte-identical copies through the same path
            K = jnp.asarray(np.asarray(K))
            dz_half = jnp.asarray(np.asarray(dz_half))
        sc = _COEF["dzh_scale"]
        if sc is not None:
            dz_half = dz_half * jnp.asarray(sc, dtype=dz_half.dtype)
        return real_solve(field, K, dz, dz_half, dt, extra_diag=extra_diag)

    VM.compute_vertical_K_profiles = kprof
    VM.implicit_vertical_diffusion_ocean = solve
    return VM, real_kprof, real_solve


def main() -> int:
    require_explicit_e3t_mode(context="s17_coeff_arms")
    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    # --- #1317 S17 A/B: EVD trigger time levels, ONE variable.
    # LEGOESM_EVD_TL={card|solver_state|nemo_now_before}; "card" (default)
    # leaves DINO_RECIPES["nemo_dino_kamm_mlf"] untouched.
    _evd_tl = os.environ.get("LEGOESM_EVD_TL", "card")
    if _evd_tl != "card":
        dcfg = dataclasses.replace(dcfg,
                                   convection_evd_n2_time_level=_evd_tl)
    print(f"[A/B] LEGOESM_EVD_TL={_evd_tl} -> "
          f"convection_evd_n2_time_level="
          f"{dcfg.convection_evd_n2_time_level!r}")
    print(f"card: surface_stress_implicit={dcfg.surface_stress_implicit} "
          f"zdf_baroclinic_only={dcfg.zdf_baroclinic_only} "
          f"zdf_drag_in_matrix={dcfg.zdf_drag_in_matrix} "
          f"outer_integrator={dcfg.outer_integrator!r}")

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, now, periodic_i=True, full_step=True,
                                     omega=dcfg.omega)
    br = br._replace(state=bridge_before_state_topo(br, g, bef,
                                                    periodic_i=True))
    if br.state.u_before is None:
        raise SystemExit("before-level seed FAILED")
    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0,
                              sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.z_coord, br.state.T.data, br.state.S.data,
                 br.geometry.dx_u, context="s17_coeff_arms")
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    L3 = lambda n: _load_full_3d(os.path.join(RUN_DIR, n), jpi, jpj, jpk - 1, hls)
    L2 = lambda n: _load_full(os.path.join(RUN_DIR, n), jpi, jpj, hls)
    krhs_u, krhs_v = L3("stp_dump_07_dynspg_u.bin"), L3("stp_dump_07_dynspg_v.bin")
    st8_u, st8_v = L3("stp_dump_08_dynzdf_u.bin"), L3("stp_dump_08_dynzdf_v.bin")
    uub, vvb = L2("stp_dump_07_dynspg_ub.bin"), L2("stp_dump_07_dynspg_vb.bin")
    avm_nemo = L3("dump_avm.bin")                       # (199, 52, 35) 0-based lv

    umask3 = np.asarray(g.umask) > 0.5
    vmask3 = np.asarray(g.vmask) > 0.5
    with netCDF4.Dataset(os.path.join(RUN_DIR, "mesh_mask.nc")) as nc:
        e3u_0 = np.moveaxis(np.asarray(nc.variables["e3u_0"][:]).squeeze(), 0, -1)
        e3v_0 = np.moveaxis(np.asarray(nc.variables["e3v_0"][:]).squeeze(), 0, -1)
        e3t_1d = np.asarray(nc.variables["e3t_1d"][:]).squeeze()
        e3w_1d = np.asarray(nc.variables["e3w_1d"][:]).squeeze()
    nk = krhs_u.shape[-1]
    um, vm = umask3[..., :nk], vmask3[..., :nk]
    wu, wv = e3u_0[..., :nk], e3v_0[..., :nk]
    rDt = 2.0 * DT

    print(f"[K5 dtypes] avm_nemo={avm_nemo.dtype} e3w_1d={e3w_1d.dtype} "
          f"state.u={br.state.u.data.dtype} u_shape={br.state.u.data.shape} "
          f"nemo nk={nk} jpk={jpk}")

    # ---- reconstruct NEMO's pre-solve input (s17's naa_B + its stress) ----
    naaB_u = ((bef.u[..., :nk] + rDt * krhs_u) * um - uub[..., None]) * um
    naaB_v = ((bef.v[..., :nk] + rDt * krhs_v) * vm - vvb[..., None]) * vm
    pre1u = np.fromfile(os.path.join(RUN_DIR, "zdf_dump_u1_prestress.bin"),
                        dtype="<f8").reshape(um.shape[0], um.shape[1])
    post1u = np.fromfile(os.path.join(RUN_DIR, "zdf_dump_u1_poststress.bin"),
                         dtype="<f8").reshape(um.shape[0], um.shape[1])
    pre1v = np.fromfile(os.path.join(RUN_DIR, "zdf_dump_v1_prestress.bin"),
                        dtype="<f8").reshape(vm.shape[0], vm.shape[1])
    post1v = np.fromfile(os.path.join(RUN_DIR, "zdf_dump_v1_poststress.bin"),
                         dtype="<f8").reshape(vm.shape[0], vm.shape[1])
    ws_u, ws_v = post1u - pre1u, post1v - pre1v
    d1 = np.abs((naaB_u[..., 0] - pre1u)[um[..., 0]]).max()
    print(f"[C1] naa_B level-1 vs NEMO prestress max|d| = {d1:.3e} "
          f"{'PASS' if d1 < 1e-14 else '*** FAIL ***'}")
    if d1 >= 1e-14:
        raise SystemExit("C1 failed -- arms unreadable")
    print(f"     RMS(ws_u)={np.sqrt(np.mean(ws_u[um[...,0]]**2)):.4e}  "
          f"RMS(ws_v)={np.sqrt(np.mean(ws_v[vm[...,0]]**2)):.4e} "
          "(v==0 is the falsifier: v's residual cannot be the wind stress)")

    # ---- baseline self arm + alignment scan (same protocol as s17) --------
    _SUB["u"] = _SUB["v"] = None
    self_res = _run_with_hook(model, br.state, sf)
    naaA_u = (bef.u[..., :nk] + rDt * krhs_u) * um
    naaA_v = (bef.v[..., :nk] + rDt * krhs_v) * vm
    OU = min((0, 1), key=lambda o: _en(
        self_res["pre_u"][:, o:o + um.shape[1], :nk], naaA_u, um)[0])
    OV = min((0, 1), key=lambda o: _en(
        self_res["pre_v"][o:o + vm.shape[0], :vm.shape[1], :nk], naaA_v, vm)[0])
    print(f"[align] u lon-offset={OU}  v lat-offset={OV}")
    crop_u = lambda a: a[:, OU:OU + um.shape[1], :nk]
    crop_v = lambda a: a[OV:OV + vm.shape[0], :vm.shape[1], :nk]

    # ---- build the two NEMO-derived coefficient arrays --------------------
    nlev = br.state.T.data.shape[-1]
    # D2: legoESM interface j  <->  NEMO 0-based level j+1 (dynzdf.F90:182-185
    # alpha at jk uses avm(jk); legoESM alpha_k uses interface k-1, jk = k+1).
    A_v_nemo = np.zeros(avm_nemo.shape[:2] + (nlev - 1,), dtype=np.float64)
    ncopy = min(nlev - 1, avm_nemo.shape[-1] - 1)
    A_v_nemo[..., :ncopy] = avm_nemo[..., 1:1 + ncopy]
    # interface nlev-2 <-> NEMO jk=jpk, wumask(jpk)=0 -> term vanishes -> 0.

    # D1: e3uw_0[k+1] / midpoint(e3u_0)[k], eta stretch held fixed.
    mid_1d = 0.5 * (e3t_1d[:-1] + e3t_1d[1:])
    dzh_scale = np.ones(nlev - 1, dtype=np.float64)
    ns = min(nlev - 1, mid_1d.size)
    dzh_scale[:ns] = e3w_1d[1:1 + ns] / mid_1d[:ns]
    print(f"[D1] dz_half scale (e3w_1d/midpoint) k=0..6: "
          + " ".join(f"{v:.6f}" for v in dzh_scale[:7])
          + f" | max dev {np.abs(dzh_scale - 1).max():.3e}")

    VM, real_kprof, real_solve = _install_hooks()
    try:
        def run(arm, feed=True, A_sub=None, sc=None, echo=False):
            pu, pv = self_res["pre_u"].copy(), self_res["pre_v"].copy()
            if feed:
                pu[:, OU:OU + um.shape[1], :nk] = naaB_u
                pv[OV:OV + vm.shape[0], :vm.shape[1], :nk] = naaB_v
                pu[:, OU:OU + um.shape[1], 0] += ws_u
                pv[OV:OV + vm.shape[0], :vm.shape[1], 0] += ws_v
            _SUB["u"], _SUB["v"] = pu, pv
            _COEF["A_v_cell"], _COEF["dzh_scale"], _COEF["echo"] = A_sub, sc, echo
            k0, s0 = _COEF["n_kprof"], _COEF["n_solve"]
            r = _run_with_hook(model, br.state, sf)
            print(f"  [{arm}] kprof fired {_COEF['n_kprof']-k0}, "
                  f"solve fired {_COEF['n_solve']-s0}")
            _SUB["u"] = _SUB["v"] = None
            _COEF["A_v_cell"] = _COEF["dzh_scale"] = None
            _COEF["echo"] = False
            return r

        print("\n=== CONTROLS ===")
        base = run("fed_ws")
        k1 = run("K1 wrapper-null")
        d = np.abs(k1["post_u"] - base["post_u"]).max()
        print(f"  [K1] max|d| vs fed_ws = {d:.3e} "
              f"{'PASS (bit-identical)' if d == 0.0 else '*** WRAPPER PERTURBS ***'}")
        if d != 0.0:
            raise SystemExit("K1 failed")
        k2 = run("K2 echo", echo=True)
        d = np.abs(k2["post_u"] - base["post_u"]).max()
        print(f"  [K2] max|d| vs fed_ws = {d:.3e} "
              f"{'PASS' if d == 0.0 else '*** ECHO PERTURBS ***'}")
        if d != 0.0:
            raise SystemExit("K2 failed")

        # K3: T-grid + level alignment of the NEMO avm against legoESM's own
        A_lego = _COEF["seen"]["A_v_cell_lego"] if "A_v_cell_lego" in _COEF["seen"] \
            else None
        if A_lego is None:
            probe = run("K3 capture", A_sub=A_v_nemo)
            A_lego = _COEF["seen"]["A_v_cell_lego"]
        tmask = np.asarray(g.tmask)[..., :nlev - 1] > 0.5
        print(f"  [K3] A_v_cell shapes lego={A_lego.shape} nemo={A_v_nemo.shape}"
              f"  dtypes {A_lego.dtype}/{A_v_nemo.dtype}")
        for off in (-1, 0, 1, 2):
            src = avm_nemo[..., max(0, 1 + off):]
            n = min(nlev - 1, src.shape[-1])
            cand = np.zeros_like(A_v_nemo)
            cand[..., :n] = src[..., :n]
            e, rn, rl, mx, cc = _en(A_lego, cand, tmask)
            print(f"       level-offset {off:+d}: err_norm={e:.4e} corr={cc:.6f}"
                  f"  ratio(RMS lego/nemo)={rl/rn if rn else float('nan'):.6f}")

        print("\n=== ARMS (all vs NEMO stage-8, depth-mean/baroclinic split) ===")
        arms = {
            "fed_ws (lego coeffs)": base,
            "+avm (NEMO avm)": run("+avm", A_sub=A_v_nemo),
            "+e3uw (NEMO e3w)": run("+e3uw", sc=dzh_scale),
            "+both": run("+both", A_sub=A_v_nemo, sc=dzh_scale),
        }
        for name, r in arms.items():
            print(f"\n  --- {name} ---")
            _report("u", crop_u(r["post_u"]), st8_u, um, wu)
            _report("v", crop_v(r["post_v"]), st8_v, vm, wv)

        print("\n=== PER-LEVEL baroclinic err_norm (u), k=0..8 ===")
        hdr = "   k  " + "".join(f"{n:>22s}" for n in arms)
        print(hdr)
        from scripts.validate.ocean_fidelity.dino_1226.s17_dynzdf_bracket import _split
        nb = _split(st8_u, wu, um)[1]
        for k in range(0, 9):
            row = f"  {k:2d}  "
            for name, r in arms.items():
                lb = _split(crop_u(r["post_u"]), wu, um)[1]
                e = _en(lb[..., k:k+1], nb[..., k:k+1], um[..., k:k+1])[0]
                row += f"{e:22.4e}"
            print(row)
        print("\n=== PER-LEVEL baroclinic err_norm (v), k=0..8 ===")
        print(hdr)
        nbv = _split(st8_v, wv, vm)[1]
        for k in range(0, 9):
            row = f"  {k:2d}  "
            for name, r in arms.items():
                lb = _split(crop_v(r["post_v"]), wv, vm)[1]
                e = _en(lb[..., k:k+1], nbv[..., k:k+1], vm[..., k:k+1])[0]
                row += f"{e:22.4e}"
            print(row)
    finally:
        VM.compute_vertical_K_profiles = real_kprof
        VM.implicit_vertical_diffusion_ocean = real_solve
    print(f"\n[K4] total hook fires: kprof={_COEF['n_kprof']} "
          f"solve={_COEF['n_solve']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
