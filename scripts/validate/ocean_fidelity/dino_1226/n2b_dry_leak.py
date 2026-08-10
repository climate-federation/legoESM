#!/usr/bin/env python
"""#1317: is the un-extrapolated ``n2_tracers_before`` (rn2b) a LIVE leak?

TRANSLATION FIRST.  NEMO source, verbatim:

  src/OCE/TRA/eosbn2.F90:1465-1467   (bn2_t, the routine behind every rn2/rn2b)
      pn2(ji,jj,jk) = grav * (  zaw * ( pts(jk-1,jp_tem) - pts(jk,jp_tem) )    &
         &                    - zbw * ( pts(jk-1,jp_sal) - pts(jk,jp_sal) )  ) &
         &            / e3w(ji,jj,jk,Kmm) * wmask(ji,jj,jk)
      -> rn2b IS EXACTLY ZERO at every dry w-point.  NEMO never carries a
         sub-seafloor rn2b value at all.

  MY_SRC/stpmlf.F90:186              CALL bn2( ts(:,:,:,:,Nbb), rab_b, rn2b, Nnn )

  The TWO consumers of rn2b inside zdftke (MY_SRC/zdftke.F90):
   C1 Prandtl  :477-497   IF (rn2b<=0) zri=0 ELSE zri=rn2b*p_avm/(p_sh2+rn_bshear)
                          p_pdlr = MAX(0.1, ri_cri/MAX(ri_cri, zri))
                          consumed at :843  p_avt = MAX(pdlr*p_avt, ...) * wmask
   C2 Langmuir :435-451   zpelc(jk) = zpelc(jk-1) + MAX(rn2b,0)*gdepw*e3w
                          imlc = SHALLOWEST jk with zpelc > zWlc2 (init mbkt+1)
                          zhlc = gdepw(imlc)   -> feeds every WET level above

  Resolved namelist, RUN_GDB/ocean.output:
      :766  nn_pdl = 1        (C1 live in NEMO)
      :780  ln_lc  = T        (C2 live in NEMO)
      :788  ri_cri = 0.2222222222222222
      :767  rn_bshear = 1e-20

legoESM equivalents:
  k_profiles.py: ``n2_tracers`` IS sub-seafloor-extrapolated (#1226 rock-fill
  guard); ``n2_tracers_before`` is NOT, and flows raw into tke.py's ``N2b``
  (tke.py:2170) which feeds (C1) ``compute_K_from_tke(N2_prandtl=N2b)`` and
  (C2) ``nemo_langmuir_tke_source(taum, N2b, ...)``.

This probe answers, in order:
  S1  realized card values (Rule 10 -- never trust a default).
  S2  STRUCTURAL: can a WET interface's N2b stencil read a dry cell at all?
      Interface k spans T-cells k,k+1 (eos.compute_buoyancy_frequency_nemo_bn2
      :604-613) and is wet iff cell k+1 is active (_wet_interface_mask:87-105),
      so the answer should be NO -- measured, not assumed.
  S3  MAGNITUDE of the raw-vs-extrapolated N2b difference, split wet/dry.
  S4  A/B on the PRODUCTION step, ONE variable: extrapolate n2_tracers_before
      or not.  Diff K_v/A_v and the full post-step state.
  S5  INSTRUMENT VALIDATION (falsifiability): the SAME hook, but substituting
      a +1e-3 degC perturbation on ACTIVE cells.  If S4 is null and S5 is
      non-null, the S4 null is a real "inert", not a dead probe.
      (An earlier draft used TKEConfig.lc=True as the control -- RETRACTED:
      the shipped kamm card already has lc=True and etau_mode='below_ml',
      so that "control" was byte-identical to S4 and proved nothing.  Rule 10.)
  S6  the one COLUMN-GLOBAL route (C2 Langmuir h_lc), measured with the real
      forcing, plus the robustness margin and the SEPARATE h_lc-fallback DIFF
      it surfaces.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.n2b_dry_leak
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import dataclasses

import numpy as np

RUN_DIR = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB")
RESTART = "DINO_00057600_restart.nc"
DT = 2700.0                    # rn_Dt (ocean.output:249 rDt = 2*rn_Dt = 5400)


# ---------------------------------------------------------------------------
def build():
    """Bridged production DINO model + state + forcing, fp64."""
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    set_policy(PrecisionPolicy.fp64())
    print(f"[precision] control dtype = {get_policy().control}")

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

    require_explicit_e3t_mode(context="n2b_dry_leak")
    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, now, periodic_i=True, full_step=True,
                                     omega=dcfg.omega)
    br = br._replace(state=bridge_before_state_topo(br, g, bef,
                                                    periodic_i=True))
    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0,
                              sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.z_coord, br.state.T.data, br.state.S.data,
                 br.geometry.dx_u, context="n2b_dry_leak")
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    sf = dino_step_surface_forcing(
        dino_lat_lon_surface_forcing_arrays(br.geometry, cfg))
    return g, br, cfg, mc, model, sf


# ---------------------------------------------------------------------------
def s1_card(mc, br):
    tke = mc.physics.vertical_mixing.tke
    print("\n=== S1  realized card (Rule 10: instantiated, not read off a "
          "default) ===")
    for k in ("n2_mode", "tke_n2_time_level", "n2_before_advection",
              "prognostic", "tke_dry_wmask", "prandtl_ri", "lc", "lc_coeff",
              "etau_mode", "buoyancy_timing"):
        print(f"    TKEConfig.{k:22s} = {getattr(tke, k, '<absent>')!r}")
    print(f"    vmix.scheme            = "
          f"{mc.physics.vertical_mixing.scheme!r}")
    print(f"    vmix_background_mode   = "
          f"{getattr(mc.physics.vertical_mixing, 'vmix_background_mode', None)!r}")
    z = br.z_coord
    for f in ("t_depth_ref", "dz_ref", "z_full_ref", "z_half_ref",
              "h_partial"):
        v = getattr(z, f, None)
        if v is not None:
            print(f"    z_coord.{f:14s} dtype = {np.asarray(v).dtype}")
    ia = np.asarray(z.is_active)
    print(f"    z_coord.is_active      shape={ia.shape} dtype={ia.dtype}  "
          f"active={int((ia > 0.5).sum())} dry={int((ia <= 0.5).sum())}")
    print(f"    state.T/S dtype        = {np.asarray(br.state.T.data).dtype} /"
          f" {np.asarray(br.state.S.data).dtype}")
    return tke


# ---------------------------------------------------------------------------
def s2_structural(br, mc):
    """Can a WET interface's N2b stencil touch a dry T-cell?  Measure it."""
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        _wet_interface_mask,                       # probe-only private import
    )
    print("\n=== S2  STRUCTURAL: wet-interface N2b stencil vs the dry mask ===")
    act = np.asarray(br.z_coord.is_active) > 0.5          # (..., nlev)
    wet_if = np.asarray(_wet_interface_mask(br.z_coord)) > 0.5   # (..., nlev-1)
    # interface k reads T-cells k and k+1 (eos.py:604-613)
    upper_dry = ~act[..., :-1]
    lower_dry = ~act[..., 1:]
    stencil_dry = upper_dry | lower_dry
    bad = wet_if & stencil_dry
    print(f"    interior interfaces      = {wet_if.size}")
    print(f"    wet interfaces           = {int(wet_if.sum())}")
    print(f"    dry interfaces           = {int((~wet_if).sum())}")
    print(f"    WET interfaces whose (k, k+1) stencil includes a DRY cell "
          f"= {int(bad.sum())}")
    # monotonicity control: a wet interface must have BOTH cells active
    print(f"      of which upper cell dry = {int((wet_if & upper_dry).sum())}"
          f"   lower cell dry = {int((wet_if & lower_dry).sum())}")
    # falsifiability control: the SAME test on the dry interfaces must be
    # non-zero, otherwise the metric is vacuous.
    print(f"    [control] DRY interfaces whose stencil includes a dry cell "
          f"= {int(((~wet_if) & stencil_dry).sum())}  (must be > 0)")
    return act, wet_if


# ---------------------------------------------------------------------------
def s3_n2b_magnitude(br, mc, act, wet_if):
    """How much does the extrapolation change N2b, split wet / dry?"""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.vertical import extrapolate_below_seafloor
    from legoesm.ocean.eos import (
        nemo_bn2_live_ladders, compute_buoyancy_frequency_nemo_bn2,
    )
    print("\n=== S3  raw vs extrapolated N2b (n2_mode='nemo_bn2') ===")
    z = br.z_coord
    st = br.state
    T_bb = np.asarray(st.T_before.data)
    S_bb = np.asarray(st.S_before.data)
    td, wd = nemo_bn2_live_ladders(z, st.eta.data, st.H_bathy.data)
    # Same call the production path makes (_shared.compute_N2:561-563: cfg
    # defaulted, g from the constants config) -- do NOT invent extra kwargs.
    gg = mc.physics.constants.g

    def n2b(T, S):
        with jax.disable_jit():
            return np.asarray(compute_buoyancy_frequency_nemo_bn2(
                jnp.asarray(T), jnp.asarray(S), td, wd, g=gg))

    raw = n2b(T_bb, S_bb)
    with jax.disable_jit():
        T_f = np.asarray(extrapolate_below_seafloor(jnp.asarray(T_bb), z))
        S_f = np.asarray(extrapolate_below_seafloor(jnp.asarray(S_bb), z))
    fil = n2b(T_f, S_f)
    d = np.abs(raw - fil)
    print(f"    N2b dtype                = {raw.dtype}")
    print(f"    max|dN2b| on WET ifaces  = {float(d[wet_if].max()):.6e}   "
          f"n_changed = {int((d[wet_if] > 0).sum())}")
    print(f"    max|dN2b| on DRY ifaces  = {float(d[~wet_if].max()):.6e}   "
          f"n_changed = {int((d[~wet_if] > 0).sum())}")
    print(f"    max|raw N2b| on DRY      = {float(np.abs(raw[~wet_if]).max()):.6e}"
          f"   (NEMO's rn2b there is EXACTLY 0 -- eosbn2.F90:1467 *wmask)")
    print(f"    max|fill N2b| on DRY     = "
          f"{float(np.abs(fil[~wet_if]).max()):.6e}")
    # tracer-level control (the number quoted for the EVD arm)
    dT = np.abs(T_bb - T_f)
    print(f"    [control] max|dT_before| = {float(dT.max()):.6f} degC over "
          f"{int((dT > 0).sum())} cells "
          f"({int((T_bb[~act] == 0.0).sum())} dry cells held exactly 0)")
    return raw, fil


# ---------------------------------------------------------------------------
def _ab_step(model, st, sf, extrapolate):
    """One production step, with/without the n2_tracers_before fill."""
    import jax
    import jax.numpy as jnp
    import legoesm.ocean.physics.vertical_mixing as VM
    from legoesm.ocean.vertical import extrapolate_below_seafloor

    real = VM.compute_vertical_K_profiles
    cap = {"K": [], "A": [], "n": 0, "patched": 0}

    def spy(state, z_coord, *a, **kw):
        if extrapolate and kw.get("n2_tracers_before") is not None:
            tb, sb = kw["n2_tracers_before"]
            kw["n2_tracers_before"] = (
                extrapolate_below_seafloor(jnp.asarray(tb), z_coord),
                extrapolate_below_seafloor(jnp.asarray(sb), z_coord),
            )
            cap["patched"] += 1
        out = real(state, z_coord, *a, **kw)
        cap["n"] += 1
        cap["K"].append(np.asarray(out[0]))
        cap["A"].append(np.asarray(out[1]))
        return out

    VM.compute_vertical_K_profiles = spy
    try:
        with jax.disable_jit():
            new = model.step(st, DT, surface_forcing=sf)
    finally:
        VM.compute_vertical_K_profiles = real
    return new, cap


def _report_ab(tag, a, b, ca, cb, wet_if):
    print(f"\n--- {tag} ---")
    print(f"    K-profile calls: baseline {ca['n']}, patched {cb['n']} "
          f"(n2_tracers_before substituted {cb['patched']}x)  "
          f"{'PASS' if cb['patched'] > 0 and ca['n'] == cb['n'] else '*** FAIL: hook never fired ***'}")
    worst = 0.0
    for i, (x, y) in enumerate(zip(ca["K"], cb["K"])):
        dk = np.abs(x - y)
        da = np.abs(ca["A"][i] - cb["A"][i])
        m = wet_if if dk.shape[-1] == wet_if.shape[-1] else Ellipsis
        print(f"    call {i}: max|dK_v| all={float(dk.max()):.6e} "
              f"wet={float(dk[m].max()):.6e} | "
              f"max|dA_v| all={float(da.max()):.6e} "
              f"wet={float(da[m].max()):.6e}")
        worst = max(worst, float(dk.max()), float(da.max()))
    for f in ("T", "S", "u", "v", "eta", "tke"):
        xa = getattr(a, f, None)
        xb = getattr(b, f, None)
        if xa is None or xb is None:
            continue
        pa, pb = np.asarray(xa.data), np.asarray(xb.data)
        eq = np.array_equal(pa, pb)
        print(f"    state.{f:4s} max|d| = {float(np.abs(pa - pb).max()):.6e}"
              f"   array_equal = {eq}")
        worst = max(worst, float(np.abs(pa - pb).max()))
    print(f"    ==> worst |delta| anywhere = {worst:.6e}")
    return worst


# ---------------------------------------------------------------------------
def main():
    g, br, cfg, mc, model, sf = build()
    s1_card(mc, br)
    act, wet_if = s2_structural(br, mc)
    s3_n2b_magnitude(br, mc, act, wet_if)

    print("\n=== S4  A/B on the PRODUCTION card (one variable: the fill) ===")
    a, ca = _ab_step(model, br.state, sf, extrapolate=False)
    b, cb = _ab_step(model, br.state, sf, extrapolate=True)
    w4 = _report_ab("S4 production card (TKEConfig.lc as shipped)",
                    a, b, ca, cb, wet_if)

    print("\n=== S5  FALSIFIABILITY CONTROL: the SAME A/B, perturbed on a WET "
          "cell ===")
    print("    A 0.0 in S4 is only meaningful if this hook CAN move the step.")
    print("    Perturb n2_tracers_before by +1e-3 degC on ACTIVE cells only;")
    print("    every number below MUST be non-zero.")
    w5 = _s5_wet_perturbation_control(model, br, sf, wet_if)

    print("\n=== S6  the only COLUMN-GLOBAL route: Langmuir h_lc "
          "(zdftke.F90:435-451) ===")
    s6_langmuir(br, mc, wet_if, sf)

    print("\n=== VERDICT INPUTS (interpretation belongs in the report) ===")
    print(f"    S4 worst |delta| (shipped card, dry-cell fill)   = {w4:.6e}")
    print(f"    S5 worst |delta| (wet-cell perturbation control) = {w5:.6e}"
          f"   {'PASS (instrument is live)' if w5 > 0 else '*** FAIL: probe is DEAD, S4 proves nothing ***'}")


def _s5_wet_perturbation_control(model, br, sf, wet_if):
    """Same hook, but the substitution is a WET-cell perturbation."""
    import jax
    import jax.numpy as jnp
    import numpy as _np
    import legoesm.ocean.physics.vertical_mixing as VM

    act = jnp.asarray(_np.asarray(br.z_coord.is_active) > 0.5)
    real = VM.compute_vertical_K_profiles

    def run(perturb):
        cap = {"K": [], "A": [], "n": 0, "patched": 0}

        def spy(state, z_coord, *a, **kw):
            if perturb and kw.get("n2_tracers_before") is not None:
                tb, sb = kw["n2_tracers_before"]
                kw["n2_tracers_before"] = (
                    jnp.where(act, jnp.asarray(tb) + 1.0e-3, jnp.asarray(tb)),
                    sb,
                )
                cap["patched"] += 1
            out = real(state, z_coord, *a, **kw)
            cap["n"] += 1
            cap["K"].append(_np.asarray(out[0]))
            cap["A"].append(_np.asarray(out[1]))
            return out

        VM.compute_vertical_K_profiles = spy
        try:
            with jax.disable_jit():
                new = model.step(br.state, DT, surface_forcing=sf)
        finally:
            VM.compute_vertical_K_profiles = real
        return new, cap

    a, ca = run(False)
    b, cb = run(True)
    return _report_ab("S5 wet-cell +1e-3 degC on n2_tracers_before "
                      "(MUST be non-zero)", a, b, ca, cb, wet_if)


def s6_langmuir(br, mc, wet_if, sf):
    """h_lc from raw vs filled N2b -- the one route a dry value could take."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.vertical import extrapolate_below_seafloor
    from legoesm.ocean.eos import (
        nemo_bn2_live_ladders, compute_buoyancy_frequency_nemo_bn2,
    )
    from legoesm.ocean.physics.vertical_mixing.tke import (
        _NEMO_TKE_LC_CSD,                          # probe-only private import
    )
    z, st = br.z_coord, br.state
    td, wd = nemo_bn2_live_ladders(z, st.eta.data, st.H_bathy.data)
    gg = mc.physics.constants.g
    with jax.disable_jit():
        raw = np.asarray(compute_buoyancy_frequency_nemo_bn2(
            st.T_before.data, st.S_before.data, td, wd, g=gg))
        fil = np.asarray(compute_buoyancy_frequency_nemo_bn2(
            extrapolate_below_seafloor(st.T_before.data, z),
            extrapolate_below_seafloor(st.S_before.data, z), td, wd, g=gg))
    pos_dry = (raw > 0) & ~wet_if
    print(f"    dry interfaces with raw N2b > 0 (the ones that can enter "
          f"MAX(rn2b,0)) = {int(pos_dry.sum())} of {int((~wet_if).sum())}"
          f"   max = {float(raw[pos_dry].max()) if pos_dry.any() else 0.0:.6e}")
    # depth_w / dz_half exactly as the k_profiles caller builds them
    from legoesm.ocean.vertical import compute_ocean_jacobian
    J = compute_ocean_jacobian(st.eta.data, st.H_bathy.data, z)
    depth_w = -np.asarray(z.z_half_ref[1:-1])
    dz_half = np.asarray(jnp.broadcast_to(
        z.dz_half_ref * J[..., None],
        st.T.data.shape[:-1] + (z.n_levels - 1,)))

    def h_lc(n2):
        pe = np.cumsum(np.maximum(n2, 0.0) * depth_w * dz_half, axis=-1)
        return pe

    pe_r, pe_f = h_lc(raw), h_lc(fil)
    # h_lc depends on taum via the threshold; scan a wide band of thresholds
    # so the answer is not one arbitrary forcing value.
    print(f"    _NEMO_TKE_LC_CSD = {_NEMO_TKE_LC_CSD}")
    print("    columns whose SHALLOWEST pe-crossing index moves, per "
          "threshold half_Wlc2:")
    for thr in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3, 1e6):
        er, ef = pe_r > thr, pe_f > thr
        ir = np.where(er.any(-1), er.argmax(-1), -1)
        i_f = np.where(ef.any(-1), ef.argmax(-1), -1)
        print(f"      thr={thr:<8g}  moved = {int((ir != i_f).sum()):6d} "
              f"of {ir.size} columns")
    print("    (the crossing index is the SHALLOWEST True; a sub-seafloor "
          "contribution is strictly DEEPER than every wet one, so it can only "
          "move the index in a column where NO wet interface crosses.)")

    # ---- RECONCILIATION (skill Rule 1e): the threshold scan above is a
    # PROXY.  Evaluate the REAL consumer with the REAL forcing instead.
    from legoesm.ocean.physics.vertical_mixing.tke import (
        nemo_langmuir_tke_source,
    )
    cfg_tke = mc.physics.vertical_mixing.tke
    taum = getattr(sf, "taum", None)
    if taum is None:
        tx = np.asarray(sf.tau_x)
        ty = np.asarray(sf.tau_y)
        taum = np.hypot(tx, ty)
    taum = np.asarray(taum)
    print(f"\n    [reconcile] production taum: shape={taum.shape} "
          f"dtype={taum.dtype} min={taum.min():.4e} max={taum.max():.4e} "
          f"n_zero={int((taum == 0).sum())}")
    with jax.disable_jit():
        src_r = np.asarray(nemo_langmuir_tke_source(
            jnp.asarray(taum), jnp.asarray(raw), jnp.asarray(depth_w),
            jnp.asarray(dz_half), cfg_tke))
        src_f = np.asarray(nemo_langmuir_tke_source(
            jnp.asarray(taum), jnp.asarray(fil), jnp.asarray(depth_w),
            jnp.asarray(dz_half), cfg_tke))
    d = np.abs(src_r - src_f)
    act = np.asarray(br.z_coord.is_active) > 0.5
    wetcol = act.any(-1)
    print(f"    [reconcile] max|d langmuir_src| ALL ifaces = {d.max():.6e}  "
          f"(n>0: {int((d > 0).sum())})")
    print(f"    [reconcile] max|d langmuir_src| WET ifaces = "
          f"{d[wet_if].max():.6e}  (n>0: {int((d[wet_if] > 0).sum())})")
    # which columns moved in the proxy scan, and are they even ocean?
    thr = _NEMO_TKE_LC_CSD * taum
    er, ef = pe_r > thr[..., None], pe_f > thr[..., None]
    ir = np.where(er.any(-1), er.argmax(-1), -1)
    i_f = np.where(ef.any(-1), ef.argmax(-1), -1)
    moved = ir != i_f
    print(f"    [reconcile] at the REAL threshold 0.0699*taum: crossing index "
          f"moves in {int(moved.sum())} columns "
          f"({int((moved & wetcol).sum())} of them ocean, "
          f"{int((moved & ~wetcol).sum())} land)")
    if moved.any():
        print(f"                those columns' taum: "
              f"min={taum[moved].min():.4e} max={taum[moved].max():.4e}; "
              f"n with taum==0: {int((taum[moved] == 0).sum())}")

    # ---- ROBUSTNESS, not just today's value.  h_lc is immune to the dry fill
    # only while the SHALLOWEST crossing is already at a WET interface.  The
    # margin is pe(deepest wet interface) / threshold: >1 => immune.
    pe_wet = np.where(wet_if, pe_r, -np.inf).max(-1)     # pe at deepest wet if
    ocean = wetcol & np.isfinite(pe_wet)
    ratio = pe_wet[ocean] / np.maximum(thr[ocean], 1e-300)
    print(f"    [robustness] ocean columns = {int(ocean.sum())}; "
          f"pe(deepest WET iface)/threshold: min={ratio.min():.4e} "
          f"median={np.median(ratio):.4e}")
    at_risk = ocean.copy()
    at_risk[ocean] = ratio <= 1
    print(f"                 ocean columns with ratio <= 1 (h_lc NOT set by a "
          f"wet interface -> the dry fill COULD move it) = "
          f"{int(at_risk.sum())}")
    if at_risk.any():
        # In those columns h_lc falls back to the deepest interface UNLESS the
        # sub-seafloor pe pushes the total over the threshold.  This is the
        # precise falsification test.
        tot = pe_r[..., -1]
        over = at_risk & (tot > thr)
        print(f"                 ...of which the RAW (unfilled) total-column "
              f"pe exceeds the threshold = {int(over.sum())}  "
              f"(max pe_total/thr among the at-risk = "
              f"{float((tot[at_risk] / np.maximum(thr[at_risk], 1e-300)).max()):.4e})")
        print(f"                 => in those columns h_lc falls back to the "
              f"deepest interface in BOTH arms, hence the exact 0 above.")
        # ---- SEPARATE, PRE-EXISTING DIFF surfaced by the above (NOT the dry
        # fill).  NEMO initialises imlc = mbkt+1 (zdftke.F90:444) so the
        # fallback zhlc = gdepw at the SEAFLOOR w-point of THAT column;
        # tke.py:1720 falls back to depth_b[..., -1] = the deepest interior
        # interface of the WHOLE grid, bathymetry-independent.
        nact = act.sum(-1)                                  # cells per column
        idx = np.clip(nact - 1, 0, len(depth_w) - 1)        # NEMO imlc=mbkt+1
        h_nemo = depth_w[idx]
        h_lego = np.full_like(h_nemo, depth_w[-1])
        r = h_lego[at_risk] / np.maximum(h_nemo[at_risk], 1e-30)
        print(f"    [separate DIFF, NOT the dry fill] fallback h_lc: NEMO "
              f"gdepw(mbkt+1) vs legoESM depth_w[-1] "
              f"(tke.py:1720 vs zdftke.F90:444)")
        print(f"                 over the {int(at_risk.sum())} columns that "
              f"actually take the fallback: h_lego/h_nemo min={r.min():.3f} "
              f"median={np.median(r):.3f} max={r.max():.3f}; "
              f"max|dh_lc| = {float(np.abs(h_lego - h_nemo)[at_risk].max()):.1f} m")


if __name__ == "__main__":
    main()
