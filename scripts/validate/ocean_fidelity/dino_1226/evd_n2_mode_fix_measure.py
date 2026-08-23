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

CONTROLS: fp64 policy printed; ``LEGOESM_NEMO_E3T`` printed; git SHA +
effective flags stamped; day-0 gate inside ``build_replay_ic``; realized card
printed per arm (Rule 10); wind forcing printed and the run ABORTS if the wind
is off; ``jax.clear_caches()`` between arms; the K-profile hook must fire in
both arms or the run aborts; NaN anywhere in a NEMO dump is FATAL; the
``eos_depth`` sensitivity arm must measurably change the N^2 field or the run
aborts (a sensitivity arm that perturbs nothing is not a control).

EOS DEPTH (2026-08-19 audit).  The kamm card sets ``eos_depth="geometric"``
(dino.py:1045) but that field reaches the PGF and GM/Redi paths only.  The EVD
trigger's density comes from ``k_profiles._enhanced_diffusion_K``, which calls
``_compute_rho(state, z_coord, J, eos_fn=eos_fn)`` with NO ``eos_depth`` -- the
``"insitu"`` DEFAULT -- and the model builds that ``eos_fn`` with no ``rho0``
either.  So this probe's offline reproduction passing nothing was FAITHFUL to
the model as run, and passing ``"geometric"`` would have made it DIVERGE from
the path it measures.  The effective value is now asserted from the live source
(the run aborts if ``_enhanced_diffusion_K`` ever starts passing the kwarg) and
``"geometric"`` is carried as a labelled SENSITIVITY arm, not as the model.
MEASURED: the choice moves the census by ZERO at every threshold and in both
windows, while genuinely changing the N^2 field -- see the non-vacuity line.
Under the shipped ``n2_mode="nemo_bn2"`` it cannot matter at all: ``bn2`` reads
T/S and the geometric gdept/gdepw ladders and consults no density helper.

REUSING A REFERENCE PATH MEANS DECLARING WHAT WAS DROPPED.  This offline block
reproduces ``_enhanced_diffusion_K``'s density/N^2 chain, and two steps of that
path are deliberately NOT carried here:
  * ``extrapolate_below_seafloor`` on both trigger arms (k_profiles.py:377-396).
    KEPT OUT because the model's own comment records it as moving zero wet
    interfaces, and the census masks to wet interior interfaces anyway.
  * the threaded ``n2_tracers`` / ``n2_tracers_before`` -- this block uses the
    raw ``st0.T/S`` and ``st0.T_before/S_before``.  Under this card
    (``evd_n2_time_level="nemo_now_before"``) those ARE the step-entry Nnn/Nbb
    tracers, which is why the census reproduces the live-hook numbers; a card
    selecting a different time level would break that equivalence.
Neither is a free pass: both are stated so the next reader can check them
rather than rediscover them.

CENSUS WINDOW (2026-08-19 audit).  This probe previously ran ONE window
(interfaces 1..34) at ONE threshold and carried the FULL-window / five-decade
figures in a comment marked "verified by review".  It now RUNS both windows and
all five thresholds, so the recorded headline is backed by committed code.

RESULT (2026-08-19, fp64, LEGOESM_NEMO_E3T=both, wind ON, day-0 gate
0.000e+00, K-profile hook fired in both arms).  The census the card comment and
commit 5a9be32ba record is REPRODUCED BY THIS COMMITTED PROBE, both windows,
all five thresholds.  n2_mode="nemo_bn2" is EXACT (0 missed, 0 spurious)
everywhere:

  window 0..34 FULL -- 332214 wet interior interfaces
    threshold     NEMO   nemo_bn2            adiabatic
      -1e-14     70475   70475  0/0 EXACT    70454   21 missed,  0 spurious
      -1e-13     70473   70473  0/0 EXACT    70449   25 missed,  1 spurious
      -1e-12     70389   70389  0/0 EXACT    70341   48 missed,  0 spurious
      -1e-11     67454   67454  0/0 EXACT    67418   36 missed,  0 spurious
      -1e-10     41874   41874  0/0 EXACT    41742  133 missed,  1 spurious

  window 1..34 (this probe's former single window) -- 322294 interfaces
      -1e-12     60845   60845  0/0 EXACT    60797   48 missed,  0 spurious
      (full table printed by the run)

So the recorded headline "70389/70389, 0 missed / 0 spurious, at every
threshold" STANDS, and the five NEMO counts in the dino.py card comment
(41874 / 67454 / 70389 / 70473 / 70475) match exactly.  eos_depth moves none
of it -- see the EOS DEPTH note above.

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


def _enhanced_diffusion_K_source():
    """The function whose density call this probe reproduces offline.

    Named explicitly (not a wrapper) so ``inspect.getsource`` asserts against
    the symbol that actually computes the EVD trigger's density.
    """
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        _enhanced_diffusion_K,
    )
    return _enhanced_diffusion_K


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
    mr.provenance("evd_n2_mode_fix_measure")
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

    # ---- EOS DEPTH, realized and PROVED, not assumed (Rule 10) -------------
    # The CARD sets eos_depth (dino.py:1045) and the model config carries it,
    # but that field reaches the PGF and GM/Redi paths ONLY.  The EVD trigger's
    # density comes from _enhanced_diffusion_K (k_profiles.py), which calls
    # _compute_rho(state, z_coord, J, eos_fn=eos_fn) with NO eos_depth kwarg,
    # i.e. the "insitu" DEFAULT -- and the model builds that eos_fn with no
    # rho0 either (ocean_model_latlon_cgrid.py, _vmix_eos_fn = _make_eos_fn(
    # eos=..., eos_linear=...)).  Both are asserted below from the live source,
    # so this probe reproduces the path the model ACTUALLY RUNS instead of the
    # path the card's top-level field would suggest.  Feeding "geometric" here
    # would make the probe DIVERGE from the model it is measuring.
    # TWO guards, because the first guard is never the only guard: the CALLER
    # could start passing the kwarg, OR the CALLEE's default could change under
    # it.  Both would silently move the model's convention out from under this
    # offline reproduction, so the effective value is READ from the callee's
    # signature rather than hard-coded, and the caller is checked separately.
    import inspect
    from legoesm.ocean.eos import compute_ocean_rho as _cor
    _kp_src = inspect.getsource(_enhanced_diffusion_K_source())
    if "eos_depth" in _kp_src:
        raise SystemExit(
            "*** _enhanced_diffusion_K now mentions eos_depth -- this probe's "
            "offline reproduction is stale; re-derive the effective depth "
            "convention from the source before trusting any number below")
    _EVD_EOS_DEPTH = inspect.signature(_cor).parameters["eos_depth"].default
    if _EVD_EOS_DEPTH not in ("insitu", "geometric"):
        raise SystemExit(f"*** unexpected compute_ocean_rho eos_depth default "
                         f"{_EVD_EOS_DEPTH!r}")
    print(f"  [eos depth] card cfg.eos_depth = "
          f"{getattr(cfg0, 'eos_depth', '<absent>')!r}   "
          f"model mc.eos_depth = {getattr(mc0, 'eos_depth', '<absent>')!r}")
    print(f"  [eos depth] the EVD trigger's density path "
          f"(k_profiles._enhanced_diffusion_K) passes NO eos_depth -> "
          f"effective {_EVD_EOS_DEPTH!r} (asserted from source above).")
    print(f"  [eos depth] and n2_mode={mc0.physics.convection.enhanced_diffusion.n2_mode!r} "
          f"on this card, whose bn2 form uses T/S + the geometric gdept/gdepw "
          f"ladders directly and consults NO density helper at all -- so "
          f"eos_depth cannot reach the SHIPPED trigger. It reaches only the "
          f"'adiabatic' comparison arm, whose sensitivity is measured below.")

    eos_fn = make_eos_fn(eos=mc0.eos, eos_linear=mc0.eos_linear)
    # WHY THIS ARM NEEDS rho0, AND WHY THE OBVIOUS REASON IS WRONG.
    # make_eos_fn's rho0 is consumed ONLY by the "nemo_eos80" branch (its own
    # docstring: zh = (p/(rho0*g))*r1_Z0 in the nemo_eos80 polynomial).  This
    # card runs "nemo_seos", which recovers depth as zh = p/(cfg.rho0*g) with
    # cfg.rho0 FIXED inside NemoSEOSConfig.  So the p = rho0*g*gdept factor
    # cancels here NOT because the kwarg was threaded, but because
    # constants.rho_0 and NemoSEOSConfig.rho0 both happen to be 1026.0.
    # Asserted, because if they ever diverge the arm silently measures the
    # rho0 mismatch (a 0.1% depth stretch, ~3x the in-situ-vs-geometric
    # difference it exists to measure) instead of the depth convention.
    eos_fn_geo = make_eos_fn(eos=mc0.eos, eos_linear=mc0.eos_linear,
                             rho0=cc.rho_0)
    from legoesm.ocean.eos import NemoSEOSConfig as _NSC
    if mc0.eos == "nemo_seos":
        _seos_rho0 = float(getattr(mc0.eos_nemo_seos, "rho0", None)
                           if getattr(mc0, "eos_nemo_seos", None) is not None
                           else _NSC().rho0)
        if abs(_seos_rho0 - float(cc.rho_0)) > 1e-9:
            raise SystemExit(
                f"*** eos_depth='geometric' arm is INVALID: NemoSEOSConfig.rho0"
                f"={_seos_rho0} != constants rho_0={cc.rho_0}, so the "
                f"p=rho0*g*gdept factor does NOT cancel and this arm would "
                f"measure a depth stretch, not the depth convention")

    T_nn = np.asarray(st0.T.data); S_nn = np.asarray(st0.S.data)
    T_bb = np.asarray(st0.T_before.data); S_bb = np.asarray(st0.S_before.data)
    eta_nn = st0.eta.data
    Jz = compute_ocean_jacobian(eta_nn, st0.H_bathy.data, br.z_coord)
    tdep, wdep = nemo_bn2_live_ladders(br.z_coord, eta_nn, st0.H_bathy.data)

    def _bn2(T, S):
        return np.asarray(compute_buoyancy_frequency_nemo_bn2(
            T, S, tdep, wdep, g=cc.g))

    def _adia(T, S, eos_depth=None):
        """Parcel-displacement N^2.  ``eos_depth=None`` -> the depth convention
        the MODEL's EVD path actually uses (``_EVD_EOS_DEPTH``); any other
        value is a deliberate SENSITIVITY arm and is NOT what the model runs.
        """
        depth = _EVD_EOS_DEPTH if eos_depth is None else eos_depth
        efn = eos_fn_geo if depth == "geometric" else eos_fn
        stx = st0._replace(T=st0.T.replace(data=T), S=st0.S.replace(data=S))
        rho = _compute_rho(stx, br.z_coord, Jz, eos_fn=efn, eos_depth=depth,
                           rho0=(cc.rho_0 if depth == "geometric" else None))
        p = compute_hydrostatic_pressure(
            rho, eta_nn, br.z_coord.dz_ref, Jz, cc.rho_0,
            h_actual=maybe_partial_h_actual(stx, br.z_coord))
        return np.asarray(compute_buoyancy_frequency_adiabatic(
            T, S, p, br.z_coord.dz_ref,
            np.where(np.asarray(Jz) <= 0.0, 1.0, np.asarray(Jz)),
            eos_fn=efn, rho_ref=cc.rho_0, g=cc.g))

    # ---- the census: BOTH windows x FIVE thresholds ------------------------
    # The recorded headline "70389/70389, 0 missed / 0 spurious, at every
    # threshold" (5a9be32ba, and the dino.py card comment) was NOT produced by
    # this committed probe: it ran ONE window (interfaces 1..34 -> 60845) at
    # ONE threshold, and the 70389 / five-decade figures existed only in a
    # comment marked "verified by review".  Both windows and the full sweep are
    # RUN here so the claim is backed by committed code or corrected.
    #   window "1..34"  lego interfaces 1..34 == NEMO 0-based w-levels 2..35
    #   window "0..34"  lego interfaces 0..34 == NEMO 0-based w-levels 1..35
    #                   (the FULL window; adds the topmost interface)
    THRESHOLDS = (-1e-14, -1e-13, -1e-12, -1e-11, -1e-10)
    WINDOWS = (("0..34 FULL", slice(0, 35), slice(1, 36)),
               ("1..34 (recorded)", slice(1, 35), slice(2, 36)))
    n2 = {
        "nemo_bn2": (_bn2(T_nn, S_nn), _bn2(T_bb, S_bb)),
        "adiabatic": (_adia(T_nn, S_nn), _adia(T_bb, S_bb)),
        "adiabatic[eos_depth=geometric]": (_adia(T_nn, S_nn, "geometric"),
                                           _adia(T_bb, S_bb, "geometric")),
    }
    # NON-VACUITY of the eos_depth sensitivity arm.  The two adiabatic rows
    # below come out with IDENTICAL counts; that is only meaningful if the two
    # N^2 FIELDS actually differ.  A sensitivity arm that perturbs nothing is
    # not a control (a control that perturbs a zero proves nothing), so the
    # field difference is measured and a null one is FATAL.
    _ai = n2["adiabatic"][0]
    _ag = n2["adiabatic[eos_depth=geometric]"][0]
    _fin = np.isfinite(_ai) & np.isfinite(_ag)
    _dmax = float(np.abs(_ai - _ag)[_fin].max())
    _scale = float(np.abs(_ai)[_fin].max())
    print(f"\n  [non-vacuity] adiabatic N^2 insitu-vs-geometric field DIFF: "
          f"max={_dmax:.4e} s^-2  (|N^2| max={_scale:.4e}, "
          f"rel={_dmax/max(_scale,1e-300):.3e})")
    # A 1-ULP difference would pass a bare "> 0" while proving nothing, so the
    # bar is RELATIVE: the arm must move the field by more than fp64 roundoff
    # on the field's own scale.
    if not _dmax > 1e-12 * _scale:
        raise SystemExit(
            f"*** the eos_depth=geometric arm moved the N^2 field by only "
            f"{_dmax:.3e} (scale {_scale:.3e}) -- at or below roundoff, so it "
            f"did not meaningfully take and its identical census below is "
            f"vacuous, not a null result")

    for wlab, lv, nlv in WINDOWS:
        Wm = wif[..., lv]
        nemo_min = np.minimum(rn2[..., nlv], rn2b[..., nlv])
        print(f"\n  window {wlab}: wet interior interfaces = {int(Wm.sum())}")
        print(f"    {'threshold':>10s} {'NEMO':>7s} | "
              + " | ".join(f"{k:>34s}" for k in n2))
        for thr in THRESHOLDS:
            nemo_fire = (nemo_min <= thr) & Wm
            cells = []
            for k, (a, b) in n2.items():
                f = (np.minimum(a[..., lv], b[..., lv]) <= thr) & Wm
                miss = int((nemo_fire & ~f).sum())
                extra = int((f & ~nemo_fire).sum())
                cells.append(f"{int(f.sum()):7d} m={miss:<5d} s={extra:<5d}"
                             f"{'  EXACT' if (miss == 0 and extra == 0) else '       '}")
            print(f"    {thr:10.0e} {int(nemo_fire.sum()):7d} | "
                  + " | ".join(cells))

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
