#!/usr/bin/env python
"""#1317 S17: WHY do legoESM and NEMO disagree on the zdf_evd trigger?

TRANSCRIPTION FIRST (the alignment table lives in the session report); this
script only CONFIRMS the DIFF rows the table named.  NEMO source, verbatim:

    src/OCE/ZDF/zdfevd.F90:118-121   (nn_evdm=1, the avm branch)
       DO_3D( 0, 0, 0, 0, 1, jpkm1 )
          IF(  MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 )   &
             &  p_avm(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)

with (MY_SRC/stpmlf.F90:186-187)
    rn2b = bn2( ts(...,Nbb), rab_b, Nnn )   <- BEFORE T/S, NOW geometry
    rn2  = bn2( ts(...,Nnn), rab_n, Nnn )   <- NOW    T/S, NOW geometry
and zdfphy.F90:312-323 (avm := avm_k at jk=2..jpkm1, THEN zdf_evd overwrites).

SECTION 1 is a pure-numpy INSTRUMENT VALIDATION on NEMO's own arrays only:
reconstruct the rule above from ``tke_dump_rn2.bin`` (registry "now") and
``tke_dump_rn2b.bin`` (registry "before") and require it to reproduce
``dump_avm.bin``'s EVD population.  If it does not, this session does not
understand the trigger and must escalate rather than measure legoESM.

SECTION 2 then splits legoESM's disagreement into
  (a) the N^2 FORMULA  (adiabatic parcel displacement vs NEMO bn2), and
  (b) the N^2 TIME LEVEL (production arm 1 is the post-explicit Kaa state;
      NEMO's two arms are Nnn and Nbb),
by evaluating legoESM's OWN trigger at NEMO's levels and diffing the masks.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.evd_trigger_align
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import dataclasses
import numpy as np

RN_EVD = 100.0          # ocean.output:738 rn_evd
EVD_THR = -1.0e-12      # zdfevd.F90:119 literal


# ===========================================================================
# SECTION 1 -- pure NEMO arrays.  No legoESM state, no model run.
# ===========================================================================
def section1(run_dir):
    from scripts.validate.ocean_fidelity.dino_1226.bn2_alpha_compare import (
        _read_dims, _load_interior,
    )
    from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
        _load_full_3d,
    )
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    jpi, jpj, jpk, hls = _read_dims(run_dir)
    ni, nj = jpi - 2 * hls, jpj - 2 * hls
    print(f"[dims] jpi={jpi} jpj={jpj} jpk={jpk} nn_hls={hls} -> "
          f"interior ({nj},{ni})")
    for d in ("tke_dump_rn2.bin", "tke_dump_rn2b.bin"):
        print(f"[registry] {d:22s} time level = {time_level_for_dump(d)!r}")

    rn2 = _load_interior(os.path.join(run_dir, "tke_dump_rn2.bin"), ni, nj)
    rn2b = _load_interior(os.path.join(run_dir, "tke_dump_rn2b.bin"), ni, nj)
    avm = _load_full_3d(os.path.join(run_dir, "dump_avm.bin"),
                        jpi, jpj, jpk - 1, hls)
    print(f"[shapes] rn2={rn2.shape}{rn2.dtype} rn2b={rn2b.shape}{rn2b.dtype} "
          f"avm={avm.shape}{avm.dtype}")
    for nm, a in (("rn2", rn2), ("rn2b", rn2b), ("avm", avm)):
        if not np.isfinite(a).all():
            raise SystemExit(f"*** {nm} has non-finite entries -- FATAL")

    from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask
    g = read_nemo_mesh_mask(os.path.join(run_dir, "mesh_mask.nc"), nn_hls=0)
    # mesh_mask.nc carries no wmask; NEMO builds it in dommsk.F90:176-180 as
    #   wmask(:,:,1)=tmask(:,:,1);  wmask(:,:,jk)=tmask(jk)*tmask(jk-1)
    tmask = np.asarray(g.tmask)
    wmask = np.empty_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] * tmask[..., :-1]
    print(f"[mask] wmask (dommsk.F90:176-180) {wmask.shape}{wmask.dtype}  "
          f"wet={int((wmask > 0.5).sum())} vs tmask wet="
          f"{int((tmask > 0.5).sum())}")

    # levels: legoESM interface j <-> NEMO 0-based level j+1 (s17_avm_perlevel).
    # dump_avm carries 0-based levels 0..jpk-2; rn2/rn2b carry 0..jpk-1.
    nlv = avm.shape[-1]
    lv = slice(1, nlv)                    # 0-based 1..34 == jk 2..35
    W = wmask[..., lv] > 0.5
    R2, R2B, AVM = rn2[..., lv], rn2b[..., lv], avm[..., lv]

    m_or = (np.minimum(R2, R2B) <= EVD_THR) & W       # NEMO's live rule
    m_now = (R2 <= EVD_THR) & W
    m_bef = (R2B <= EVD_THR) & W
    m_and = (np.maximum(R2, R2B) <= EVD_THR) & W      # the commented-out form
    m_dump = (AVM > 0.95 * RN_EVD) & W

    nwet = int(W.sum())
    print(f"\n(1) reconstruction vs NEMO's own dump   n_wet={nwet}")
    exact = float(np.abs(AVM[m_dump] - RN_EVD).max()) if m_dump.any() else 0.0
    print(f"    max|avm - rn_evd| on the fired population = {exact:.3e} "
          f"(expect 0 -- zdf_evd is a hard SET)")
    for lab, m in (("MIN(rn2,rn2b)<=-1e-12  [LIVE rule]", m_or),
                   ("rn2 only               [control]", m_now),
                   ("rn2b only              [control]", m_bef),
                   ("MAX(rn2,rn2b)<=-1e-12  [control]", m_and)):
        xor = int((m ^ m_dump).sum())
        print(f"    {lab:36s} n={int(m.sum()):7d} ({100*m.sum()/nwet:6.3f}%)"
              f"  XOR vs dump = {xor:7d} ({100*xor/nwet:7.4f}%)")
    # ---- row: NEMO SETs avm := rn_evd; legoESM takes MAX(closure, K_conv)
    #      (k_profiles.py:338 under vmix_background_mode="nemo_max_floor").
    #      The two differ only where the CLOSURE already exceeds rn_evd.
    #      tke_dump_avm_final.bin = zdftke's exit avm = NEMO's avm_k (PRE-EVD).
    clo = _load_interior(os.path.join(run_dir, "tke_dump_avm_final.bin"),
                         ni, nj)[..., lv]
    over = (clo > RN_EVD) & m_or
    print(f"\n(1b) SET vs MAX: closure avm > rn_evd on EVD-fired cells: "
          f"n={int(over.sum())} of {int(m_or.sum())} fired  "
          f"(max closure avm on fired = {float(clo[m_or].max()):.4e})")

    # ---- reconciliation of the TWO references quoted in this campaign.
    def _cr(a, b, m):
        x, y = a[m], b[m]
        return (float(np.corrcoef(x, y)[0, 1]),
                float(np.sqrt((x ** 2).mean()) / np.sqrt((y ** 2).mean())))
    c1, r1 = _cr(clo, AVM, W)
    print(f"(1c) reference reconciliation (both NEMO's own arrays):")
    print(f"     tke_dump_avm_final (closure exit, PRE-EVD)  vs  "
          f"dump_avm (POST-EVD)   corr={c1:.6f}  rms-ratio={r1:.6f}")
    print(f"     -> the two references are DIFFERENT ARRAYS; EVD alone moves "
          f"NEMO's own avm by this much.")

    return dict(m_or=m_or, m_now=m_now, m_bef=m_bef, m_dump=m_dump, W=W,
                lv=lv, nwet=nwet, rn2=rn2, rn2b=rn2b, run_dir=run_dir)


# ===========================================================================
# SECTION 2 -- legoESM's trigger, same state, arm by arm.
# ===========================================================================
def section2(s1):
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    set_policy(PrecisionPolicy.fp64())
    print(f"\n[precision] control dtype = {get_policy().control}")

    from legoesm.ocean.fidelity.precision_gate import (
        require_fp64, require_explicit_e3t_mode,
    )
    from legoesm.ocean.fidelity.nemo_io import (
        read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
    )
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        bridge_nemo_to_legoesm_topo, bridge_before_state_topo,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe, dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing,
    )
    from legoesm.ocean.eos import make_eos_fn
    import legoesm.ocean.physics.convection.enhanced_diffusion as ED
    import legoesm.ocean.physics.vertical_mixing as VM
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        _enhanced_diffusion_K,          # probe-only private import
    )
    from scripts.validate.ocean_fidelity.dino_1226.s17_dynzdf_bracket import (
        _SUB, _run_with_hook, RESTART,
    )

    run_dir = s1["run_dir"]
    require_explicit_e3t_mode(context="evd_trigger_align")
    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    g = read_nemo_mesh_mask(os.path.join(run_dir, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(run_dir, RESTART), nn_hls=0)
    bef = read_nemo_restart_before(os.path.join(run_dir, RESTART), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, now, periodic_i=True, full_step=True,
                                     omega=dcfg.omega)
    br = br._replace(state=bridge_before_state_topo(br, g, bef,
                                                    periodic_i=True))
    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0,
                              sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.z_coord, br.state.T.data, br.state.S.data,
                 br.geometry.dx_u, context="evd_trigger_align")
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    sf = dino_step_surface_forcing(
        dino_lat_lon_surface_forcing_arrays(br.geometry, cfg))

    # ---- Rule 10: print the REALIZED card values, do not trust the source.
    pc = mc.physics
    ed = pc.convection.enhanced_diffusion
    print(f"[card] convection.scheme={pc.convection.scheme!r}")
    print(f"[card] EnhancedDiffusionConfig: K_conv={ed.K_conv} K_bg={ed.K_bg} "
          f"nu_conv={ed.nu_conv} nu_bg={ed.nu_bg}")
    print(f"[card]   smooth_transition={ed.smooth_transition} "
          f"sigmoid_sharpness={ed.sigmoid_sharpness} n2_mode={ed.n2_mode!r} "
          f"n2_threshold={ed.n2_threshold!r} "
          f"two_level_trigger={ed.two_level_trigger}")
    print(f"[card] vmix.scheme={pc.vertical_mixing.scheme!r} "
          f"vmix_background_mode="
          f"{getattr(pc.vertical_mixing, 'vmix_background_mode', None)!r} "
          f"tke.n2_before_advection="
          f"{getattr(pc.vertical_mixing.tke, 'n2_before_advection', None)} "
          f"tke.tke_n2_time_level="
          f"{getattr(pc.vertical_mixing.tke, 'tke_n2_time_level', None)!r}")
    print(f"[card] surface_tendency_placement="
          f"{getattr(cfg, 'surface_tendency_placement', None)!r}")
    fire = 0.5 * float(ed.K_conv)

    # ---- production run, capturing (i) A_v_total and (ii) every EVD arm.
    arms: list[dict] = []
    real_flag = ED.convective_K_A_flag

    def flag_spy(rho, dz_ref, jacobian, c, **kw):
        out = real_flag(rho, dz_ref, jacobian, c, **kw)
        arms.append({"T": np.asarray(kw["T"]), "A": np.asarray(out[1])})
        return out

    cap: dict = {"A": None, "n": 0}
    real_kp = VM.compute_vertical_K_profiles

    def kp_spy(*a, **kw):
        out = real_kp(*a, **kw)
        cap["n"] += 1
        if cap["A"] is None:
            cap["A"] = np.asarray(out[1])
        return out

    ED.convective_K_A_flag = flag_spy
    VM.compute_vertical_K_profiles = kp_spy
    try:
        _SUB["u"] = _SUB["v"] = None
        _run_with_hook(model, br.state, sf)      # eager (jax.disable_jit)
    finally:
        ED.convective_K_A_flag = real_flag
        VM.compute_vertical_K_profiles = real_kp

    print(f"\n[control] K-profile spy fired {cap['n']}x, EVD-arm spy "
          f"{len(arms)}x  {'PASS' if cap['n'] >= 1 and len(arms) >= 2 else '*** FAIL ***'}")
    if cap["A"] is None or len(arms) < 2:
        raise SystemExit("spies did not capture the production arms")
    T_nn = np.asarray(br.state.T.data)
    T_bb = np.asarray(br.state.T_before.data)
    # Compare on WET cells only: land carries fill values that differ between
    # the bridged state and the model's internal masking, and a global max
    # over land cannot discriminate the arms (first version of this control
    # returned the same 26.39 for both arms -- it was measuring land).
    tm = np.asarray(g.tmask)[..., :T_nn.shape[-1]] > 0.5
    if tm.shape != T_nn.shape:
        raise SystemExit(f"tmask {tm.shape} vs T {T_nn.shape}")
    def dw(a, b):
        return float(np.abs(a - b)[tm].max())
    print(f"[control] arm-identification on WET cells (n={int(tm.sum())}):")
    print(f"            max|T_arm0 - T_Nnn| = {dw(arms[0]['T'], T_nn):.6e}")
    print(f"            max|T_arm1 - T_Nnn| = {dw(arms[1]['T'], T_nn):.6e}"
          "   <- expect 0.0 (arm1 IS the Nnn step-entry state)")
    print(f"            max|T_Nnn  - T_Nbb| = {dw(T_nn, T_bb):.6e}")
    print(f"            max|T_arm0 - T_Nbb| = {dw(arms[0]['T'], T_bb):.6e}"
          "   (arm0 is neither: it is the post-explicit Kaa state)")

    # ---- counterfactual arms, evaluated on NEMO's OWN two levels.
    conv = pc.convection
    eos_fn = make_eos_fn(eos=mc.eos, eos_linear=mc.eos_linear)
    cc = pc.constants
    with jax.disable_jit():
        _, A_nn = _enhanced_diffusion_K(br.state, br.z_coord, conv,
                                        eos_fn=eos_fn, before_tracers=None,
                                        cc=cc)
        _, A_faith = _enhanced_diffusion_K(
            br.state, br.z_coord, conv, eos_fn=eos_fn,
            before_tracers=(br.state.T_before.data, br.state.S_before.data),
            cc=cc)
    A_nn, A_faith = np.asarray(A_nn), np.asarray(A_faith)

    # ---- align to NEMO levels and compare masks.
    W, m_or, m_now, m_bef = s1["W"], s1["m_or"], s1["m_now"], s1["m_bef"]
    n = W.shape[-1]                        # NEMO 0-based levels 1..n
    def to_nemo(a):
        if a.shape[:-1] != W.shape[:-1]:
            raise SystemExit(f"horizontal shape mismatch {a.shape} vs {W.shape}")
        return a[..., :n]                  # lego iface j <-> NEMO level j+1

    masks = {
        "PRODUCTION  A_v_total        (arm0=Kaa OR arm1=Nnn)":
            to_nemo(np.asarray(cap["A"])) > fire,
        "arm0 alone  N2(Kaa post-expl)": to_nemo(arms[0]["A"]) > fire,
        "arm1 alone  N2(Nnn)":           to_nemo(arms[1]["A"]) > fire,
        "lego N2(Nnn) only            (single level)": to_nemo(A_nn) > fire,
        "lego OR(N2(Nnn),N2(Nbb))     [NEMO's levels]": to_nemo(A_faith) > fire,
    }
    nwet = int(W.sum())
    print(f"\n(2) trigger populations on NEMO's wet W-levels 1..{n}  "
          f"n_wet={nwet}")
    print(f"    {'NEMO MIN(rn2,rn2b)<=-1e-12':46s} n={int(m_or.sum()):7d} "
          f"({100*m_or.sum()/nwet:6.3f}%)")
    for lab, m in masks.items():
        m = m & W
        extra = int((m & ~m_or).sum())
        miss = int((~m & m_or).sum())
        print(f"    {lab:46s} n={int(m.sum()):7d} ({100*m.sum()/nwet:6.3f}%)"
              f"  spurious={extra:6d}  missed={miss:6d}  XOR={extra+miss:6d}"
              f" ({100*(extra+miss)/nwet:6.3f}%)")
    print(f"\n    reference decomposition targets: the recorded avm residual "
          f"had 1751 cells (0.527%) in the rn_evd flip band.")
    return 0


def main() -> int:
    run_dir = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
    s1 = section1(run_dir)
    return section2(s1)


if __name__ == "__main__":
    raise SystemExit(main())
