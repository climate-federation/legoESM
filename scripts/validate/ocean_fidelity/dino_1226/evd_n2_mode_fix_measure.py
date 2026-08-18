#!/usr/bin/env python
"""#1226 EVD N2 mode fix -- THE MEASUREMENT.

One production one-step of the ``nemo_dino_kamm_mlf`` card against NEMO's own
next state, run TWICE with exactly ONE variable changed: the EVD trigger's N^2
formula (``DINOConfig.convection_n2_mode``).

  arm "adiabatic"  the PRE-FIX card: parcel displacement of both interface
                   cells to the upper cell's pressure, full nonlinear in-situ
                   density differenced.
  arm "nemo_bn2"   the POST-FIX card: legoESM's exact transcription of NEMO
                   ``bn2`` (``eosbn2.F90:1459-1466``), which is the array
                   ``zdfevd.F90:93/:119`` actually tests.

Everything else -- IC, forcing, geometry, time levels, dt, TKE, GM/Redi -- is
byte-identical between the arms (skill Rule 7), and both arms are built from
the SAME ``build_replay_ic`` call so the state cannot drift.

REPORTS
  (i)   the one-step tracer error at the known spike (158,42,10), plus the
        whole-field |dT| distribution p50/p99/p99.9/max over wet cells.  A fix
        that collapses the spike while degrading the bulk is skill Rule 8 and
        must be reported, not hidden -- hence the percentiles, not just the max.
  (ii)  the domain-wide EVD trigger population vs NEMO's own rn2/rn2b dumps:
        n fired / missed / spurious over all wet interior interfaces.
  (iii) the assembled diffusivity/viscosity vs NEMO's POST-EVD dumps
        ``dump_avt.bin`` / ``dump_avm.bin``.  Those two are written from
        ``MY_SRC/ldftra.F90:934-937``, i.e. inside the ldf block at
        ``stpmlf.F90:210-231``, which runs AFTER ``zdf_phy`` (``:204``) and
        therefore after ``zdf_evd`` -- so they are the right reference for a
        trigger change.  (``tke_dump_av*_final.bin`` are the zdftke-internal
        pre-EVD arrays and are NOT used here.)

CONTROLS: fp64 policy printed; ``LEGOESM_NEMO_E3T`` printed; day-0 gate inside
``build_replay_ic``; realized card printed per arm (Rule 10); wind forcing
printed; ``jax.clear_caches()`` between arms; the K-profile hook must fire in
both arms or the run aborts; NaN anywhere in a NEMO dump is FATAL.

Run:
CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  python scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/evd_n2_mode_fix_measure.py
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(_SCRIPTS_OCEAN_FIDELITY)))
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DT = 2700.0
J, I, KSPIKE = 158, 42, 10        # the known spike cell
IFACE = 9                          # legoESM interface 9 == NEMO 0-based w-level 10
EVD_THR = -1.0e-12                 # zdfevd.F90:93 literal
ARMS = ("adiabatic", "nemo_bn2")


def main() -> int:
    import jax
    import multistep_replay as mr
    from evd_before_arm_n2 import (
        SEQDUMP, _load_interior, _load_full, dims,
    )
    from s17_dynzdf_bracket import _en
    from legoesm.core.precision import get_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
        dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays,
        dino_step_surface_forcing,
    )
    import legoesm.ocean.physics.vertical_mixing as VM

    if not mr.have_step1_artifacts():
        print("SKIP: oracle artifacts not present")
        return 0
    print(f"PRECISION control dtype = {get_policy().control}")
    print(f"LEGOESM_NEMO_E3T = {os.environ.get('LEGOESM_NEMO_E3T')!r}")
    print(f"SEQDUMP = {SEQDUMP}")

    # ---- NEMO references (one load, shared by both arms) -------------------
    d = dims()
    rn2 = _load_interior(f"{SEQDUMP}/tke_dump_rn2.bin")
    rn2b = _load_interior(f"{SEQDUMP}/tke_dump_rn2b.bin")
    avt_n = _load_full(f"{SEQDUMP}/dump_avt.bin")
    avm_n = _load_full(f"{SEQDUMP}/dump_avm.bin")
    for nm, a in (("rn2", rn2), ("rn2b", rn2b), ("avt", avt_n), ("avm", avm_n)):
        if not np.isfinite(a).all():
            raise SystemExit(f"*** {nm} non-finite -- FATAL")
        print(f"  [nemo] {nm:5s} shape={a.shape} dtype={a.dtype}")
    print("  [nemo] dump_avt/dump_avm are POST-EVD (MY_SRC/ldftra.F90:934-937,"
          " written from the ldf block at stpmlf.F90:210-231, after zdf_phy:204)")

    g, br, cfg0, st0 = mr.build_replay_ic()
    ns = mr.nemo_now_state_at(mr.IC_STEP + 1)
    tm3 = np.asarray(g.tmask) > 0.5
    wif = tm3[..., :-1] & tm3[..., 1:]
    print(f"  [dtype] T={np.asarray(st0.T.data).dtype} "
          f"dz_ref={np.asarray(br.z_coord.dz_ref).dtype} "
          f"eta={np.asarray(st0.eta.data).dtype}")

    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg0)
    wind = bool(getattr(cfg0, "wind_through_step", False))
    sf = dino_step_surface_forcing(forcing) if wind else None
    print(f"  FORCING wind_through_step={wind}"
          + (f"  tau_x range=[{float(np.asarray(sf.tau_x).min()):.4f},"
             f"{float(np.asarray(sf.tau_x).max()):.4f}]" if wind else ""))
    if not wind:
        raise SystemExit("*** wind is OFF -- the wind-on harness is mandatory "
                         "for every stepping probe (commit 9d0de849a)")

    results: dict[str, dict] = {}
    for mode in ARMS:
        jax.clear_caches()
        cfg = dataclasses.replace(cfg0, convection_n2_mode=mode)
        mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
        ed = mc.physics.convection.enhanced_diffusion
        print(f"\n=== ARM {mode!r} ==========================================")
        print(f"  [realized] n2_mode={ed.n2_mode!r} thr={ed.n2_threshold!r} "
              f"two_level={ed.two_level_trigger} "
              f"evd_n2_time_level={ed.evd_n2_time_level!r} "
              f"smooth={ed.smooth_transition} K_conv={ed.K_conv}")
        print(f"  [realized] tke.n2_mode={mc.physics.vertical_mixing.tke.n2_mode!r} "
              f"vmix={mc.physics.vertical_mixing.scheme!r} "
              f"placement={getattr(cfg,'surface_tendency_placement',None)!r}")
        if ed.n2_mode != mode:
            raise SystemExit(f"*** arm did not take: {ed.n2_mode!r} != {mode!r}")

        cap: dict = {"K": None, "A": None, "n": 0}
        real = VM.compute_vertical_K_profiles

        def spy(*a, _cap=cap, _real=real, **kw):
            out = _real(*a, **kw)
            _cap["n"] += 1
            if _cap["K"] is None:
                _cap["K"] = np.asarray(out[0])
                _cap["A"] = np.asarray(out[1])
            return out

        model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
        VM.compute_vertical_K_profiles = spy
        try:
            if getattr(cfg, "surface_tendency_placement", None) == "leapfrog_rhs":
                st, rate = apply_dino_lat_lon_surface_forcing(
                    st0, forcing, br.z_coord, cfg, DT, t_seconds=DT,
                    return_rate=True)
            else:
                st = apply_dino_lat_lon_surface_forcing(
                    st0, forcing, br.z_coord, cfg, DT, t_seconds=DT)
                rate = None
            # Eager so the K/A hook sees concrete arrays.  The spike value
            # this reproduces (5.710106e-02) was measured under jit by
            # evd_before_arm_n2.py, so a match is also a jit-parity control.
            with jax.disable_jit():
                st1 = model.step(st, DT, surface_forcing=sf,
                                 external_tracer_rate=rate)
            st1.T.data.block_until_ready()
        finally:
            VM.compute_vertical_K_profiles = real
        if cap["n"] < 1 or cap["K"] is None:
            raise SystemExit("*** K-profile hook never fired -- the measurement "
                             "would be of nothing")
        print(f"  [control] K-profile hook fired {cap['n']}x  PASS")

        dT = np.asarray(st1.T.data) - ns.T
        results[mode] = {"dT": dT, "K": cap["K"], "A": cap["A"]}

    # ================= (i) the spike + the whole-field distribution =========
    print("\n--- (i) one-step tracer error vs NEMO ------------------------")
    print("  mode        spike(158,42,10)     p50        p99        p99.9      "
          "max          argmax")
    for mode in ARMS:
        dT = results[mode]["dT"]
        a = np.abs(dT)[tm3]
        full = np.where(tm3, np.abs(dT), -np.inf)
        idx = np.unravel_index(int(np.argmax(full)), full.shape)
        print(f"  {mode:11s} {dT[J,I,KSPIKE]: .6e}   "
              f"{np.percentile(a,50):.3e}  {np.percentile(a,99):.3e}  "
              f"{np.percentile(a,99.9):.3e}  {a.max():.6e}  "
              f"{tuple(int(v) for v in idx)}")
    b = float(results["adiabatic"]["dT"][J, I, KSPIKE])
    n = float(results["nemo_bn2"]["dT"][J, I, KSPIKE])
    print(f"  SPIKE collapse: {b:.6e} -> {n:.6e}   ratio={abs(n)/abs(b):.3e}")
    for pct in (50, 90, 99, 99.9):
        pa = float(np.percentile(np.abs(results["adiabatic"]["dT"])[tm3], pct))
        pn = float(np.percentile(np.abs(results["nemo_bn2"]["dT"])[tm3], pct))
        print(f"  BULK p{pct:<5}: {pa:.6e} -> {pn:.6e}  "
              f"({'better' if pn < pa else 'WORSE' if pn > pa else 'same'}"
              f", x{pn/pa:.4f})")
    dd = results["nemo_bn2"]["dT"] - results["adiabatic"]["dT"]
    ch = np.abs(dd)[tm3] > 0
    print(f"  cells whose dT CHANGED at all: {int(ch.sum())} of {int(tm3.sum())} "
          f"({100*ch.mean():.4f}%)  -- surgical vs global")

    # ================= (ii) the trigger population vs NEMO ==================
    print("\n--- (ii) EVD trigger population vs NEMO ----------------------")
    from legoesm.ocean.eos import (
        compute_buoyancy_frequency_adiabatic, compute_buoyancy_frequency_nemo_bn2,
        compute_hydrostatic_pressure, make_eos_fn, maybe_partial_h_actual,
        nemo_bn2_live_ladders,
    )
    from legoesm.ocean.physics.vertical_mixing.k_profiles import _compute_rho
    from legoesm.ocean.vertical import compute_ocean_jacobian
    mc0, _ = dino_lat_lon_model_config(br.geometry, cfg0)
    cc = mc0.physics.constants
    eos_fn = make_eos_fn(eos=mc0.eos, eos_linear=mc0.eos_linear)
    T_nn = np.asarray(st0.T.data); S_nn = np.asarray(st0.S.data)
    T_bb = np.asarray(st0.T_before.data); S_bb = np.asarray(st0.S_before.data)
    eta_nn = st0.eta.data
    Jz = compute_ocean_jacobian(eta_nn, st0.H_bathy.data, br.z_coord)
    tdep, wdep = nemo_bn2_live_ladders(br.z_coord, eta_nn, st0.H_bathy.data)

    def _bn2(T, S):
        return np.asarray(compute_buoyancy_frequency_nemo_bn2(
            T, S, tdep, wdep, g=cc.g))

    def _adia(T, S):
        stx = st0._replace(T=st0.T.replace(data=T), S=st0.S.replace(data=S))
        rho = _compute_rho(stx, br.z_coord, Jz, eos_fn=eos_fn)
        p = compute_hydrostatic_pressure(
            rho, eta_nn, br.z_coord.dz_ref, Jz, cc.rho_0,
            h_actual=maybe_partial_h_actual(stx, br.z_coord))
        return np.asarray(compute_buoyancy_frequency_adiabatic(
            T, S, p, br.z_coord.dz_ref,
            np.where(np.asarray(Jz) <= 0.0, 1.0, np.asarray(Jz)),
            eos_fn=eos_fn, rho_ref=cc.rho_0, g=cc.g))

    # Interfaces 1..34 = NEMO w-levels 2..35 (the recorded 60845 window). The
    # FULL window slice(0, 35) gives 70389/70389/0/0 — verified by review;
    # kept at the recorded window so the headline reproduces.
    lv = slice(1, 35)
    Wm = wif[..., lv]
    nemo_fire = (np.minimum(rn2[..., 2:36], rn2b[..., 2:36]) <= EVD_THR) & Wm
    print(f"  wet interior interfaces = {int(Wm.sum())}   "
          f"NEMO fires = {int(nemo_fire.sum())}")
    for mode, fn in (("adiabatic", _adia), ("nemo_bn2", _bn2)):
        f = (np.minimum(fn(T_nn, S_nn)[..., lv], fn(T_bb, S_bb)[..., lv])
             <= EVD_THR) & Wm
        miss = int((nemo_fire & ~f).sum()); extra = int((f & ~nemo_fire).sum())
        print(f"  {mode:11s} fires={int(f.sum()):6d}  missed={miss:5d}  "
              f"spurious={extra:5d}")

    # ================= (iii) avt / avm vs the POST-EVD dumps ================
    print("\n--- (iii) diffusivity/viscosity vs NEMO POST-EVD dumps -------")
    for lab, key, ref in (("avt", "K", avt_n), ("avm", "A", avm_n)):
        print(f"  {lab}:")
        for mode in ARMS:
            L = results[mode][key]
            nif = L.shape[-1]
            nlv = min(nif, ref.shape[-1] - 1)
            N = np.zeros_like(L)
            N[..., :nlv] = ref[..., 1:1 + nlv]   # lego iface j <-> NEMO w j+1
            M = np.zeros(L.shape, dtype=bool)
            M[..., :nlv] = wif[..., 1:1 + nlv]
            e, rn_, rl_, mx, corr = _en(L, N, M)
            evd_l = float((L[M] > 95).mean()); evd_n = float((N[M] > 95).mean())
            print(f"    {mode:11s} err_norm={e:.4e} corr={corr:.6f} "
                  f"ratio={rl_/rn_:.6f} max|d|={mx:.4e}  "
                  f"frac at EVD level(>95): lego={100*evd_l:.4f}% "
                  f"nemo={100*evd_n:.4f}%  n={int(M.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
