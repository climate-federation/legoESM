#!/usr/bin/env python
"""#1226 join-seam follow-up: (a) de-artifact the RESET-arm metrics and
(b) DISCRIMINATE whether the missing restart CARRIES own the ~0.16 K/step
per-step injected divergence.

This EXTENDS ``join_seam_growth.py`` (same RESET protocol, same bridge, same
window).  It does NOT re-derive its result: the RESET arm's per-step
divergence and its flatness across states are taken as CONFIRMED input.

(a) METRIC ARTIFACT
-------------------
``join_seam_growth`` reduces every field with ``max|.|`` and the prior lane
found ``du/deta/den/de3t`` identical to 5 significant figures across 32
independent ocean states -- the signature of ONE fixed cell.  Here every
field additionally reports its ARGMAX INDEX, the 99.9/99/50th percentile over
the same wet set, and the count of cells above 10%/1% of the max, so
"flat" can be separated from "one cell".

(b) THE CARRY DISCRIMINATOR
---------------------------
NEMO's restart carries 18 live step-entry values; the bridge reads 11.
The restart file itself (``DINO_<kt>_restart_*.nc``) contains every missing
one -- ``avm_k``, ``avt_k``, ``dissl``, ``sbc_hc_b``, ``sbc_sc_b``,
``qsr_hc_b``, ``fraqsr_1lev``, ``qns_b``, ``sfx_b`` -- so the missing carry
can be INJECTED as a measurement, with NO production change.

Oracle lines that define each injection (Rule 0):

  * ``trasbc.F90:152-154``
      ``pts(ji,jj,1,jn,Krhs) += zfact*( sbc_tsc_b + sbc_tsc ) / e3t(1,Kmm)``
      with ``zfact=0.5`` at every step but ``nit000``, and
      ``sbc_tsc_b(:,:,:) = sbc_tsc(:,:,:)`` (``:130``) at step entry.
      ``trasbc.F90:159-160`` writes the restart under the names
      ``sbc_hc_b``/``sbc_sc_b`` from the CURRENT step's ``sbc_tsc`` -- so
      ``restart[kt].sbc_hc_b`` is exactly the ``sbc_tsc_b`` NEMO uses during
      step ``kt+1``.
  * ``traqsr.F90:206-208``
      ``pts(ji,jj,jk,jp_tem,Krhs) += 0.5*( qsr_hc_b(jk) + qsr_hc(jk) )
      / e3t(jk,Kmm)`` for every level, with ``qsr_hc_b = qsr_hc`` (``:166``)
      at step entry and ``restart[kt].qsr_hc_b = qsr_hc`` of step ``kt``
      (``:241``).
  * ``zdfphy.F90:284-286`` (``avt(jk)=avt_k(jk)``, ``avm(jk)=avm_k(jk)``)
      -- the closure Kz that ``tra_zdf``/``dyn_zdf`` of THAT step use.
      ``zdf_tke`` (``zdfphy.F90:281``) OVERWRITES ``avm_k``/``avt_k`` before
      that copy, so ``restart[kt].avm_k`` is the Kz NEMO used during step
      ``kt`` and the Kz that step ``kt+1``'s ``zdf_sh2`` (``zdfphy.F90:268``)
      reads at ENTRY.

Arms (each is ONE variable off its stated reference, Rule 7):

  base        legoESM as the recipe runs it (``surface_tendency_placement=
              'applied_now'``).  THE REFERENCE.
  null_K      identity monkeypatch through the SAME injection hook as the B
              arms.  CONTROL: must be BIT-IDENTICAL to ``base``; if it is
              not, the hook itself perturbs and no B number is readable.
  B_prev      Kz := ``restart[kt-1].avm_k/avt_k``  (the PREVIOUS step's Kz --
              what legoESM's carried-TKE-derived Kz approximates).
  B_now       Kz := ``restart[kt].avm_k/avt_k``    (NEMO's OWN Kz for the
              step being taken).  This is the closure-ownership test.
  A_lf        ``surface_tendency_placement='leapfrog_rhs'`` with legoESM's
              own rate.  Isolates PLACEMENT alone.
  A_nemo_avg  leapfrog_rhs with NEMO's exact ``0.5*(b+n)`` surface tracer
              content trend (trasbc + traqsr).  Fully faithful surface
              forcing.
  A_nemo_now  leapfrog_rhs with NEMO's ``n``-only trend (no b-average).
              ``A_nemo_avg`` vs ``A_nemo_now`` isolates the MISSING CARRY
              ``sbc_hc_b``/``qsr_hc_b`` specifically.
  null_rate   leapfrog_rhs with legoESM's own rate pushed through the SAME
              rate-substitution hook the A_nemo arms use.  CONTROL: must be
              BIT-IDENTICAL to ``A_lf``.
  AB          A_nemo_avg + B_now together.
  tsec        ``base`` with the model-time-correct ``t_seconds`` for the
              RESET arm.  CONTROL on an instrument bug in
              ``join_seam_growth``'s reset loop: it calls ``run_replay(1)``
              per state, so ``t_seconds`` is ALWAYS ``1*dt`` even at state
              j=31, i.e. the seasonal forcing phase can be up to one day
              stale.  Quantifies that, it does not fix it.

WHAT IS FAITHFUL AND WHAT IS AN APPROXIMATION (stated up front)
---------------------------------------------------------------
  FAITHFUL   the A_nemo_* rates: they are NEMO's own ``sbc_tsc``/``qsr_hc``
             arrays, divided by legoESM's live top-cell ``e3t(Kmm)``
             (``compute_layer_thickness`` on the SAME entry ``eta`` NEMO's
             Kmm level holds), i.e. NEMO's arithmetic on NEMO's numbers.
  FAITHFUL   the B arms' Kz values themselves (NEMO's ``avm_k``/``avt_k``,
             surface w-level dropped exactly as ``bridge_tke_from_restart``
             drops it).
  APPROX     the B arms override the Kz RETURNED by
             ``tke_set_diffusivities`` -- the Kz that drives this step's
             ``tra_zdf``/``dyn_zdf``.  The ``TKEPostMixingContext`` (which
             drives the TKE integration and hence the NEXT step's Kz) is
             left at legoESM's own values.  Over a ONE-STEP RESET arm that
             next-step path never runs, so the approximation does not enter
             the reported number -- but it means these arms do NOT test
             ``dissl`` (which only enters the TKE tridiagonal,
             ``zdftke.F90:414/419``).  ``dissl`` is therefore UNTESTED here.
  APPROX     ``A_lf`` and every A_nemo arm change the placement as well, so
             they are read against ``A_lf``, never against ``base``.

Usage
-----
    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \\
      scripts/validate/ocean_fidelity/dino_1226/carry_injection_discriminator.py \\
      [n_states] [arm,arm,...]
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DT = 2700.0

ALL_ARMS = ("base", "null_K", "B_prev", "B_now", "B_all",
            "A_lf", "null_rate", "A_nemo_avg", "A_nemo_now", "AB", "tsec")

# --- injection hooks (module-level so the monkeypatch closure is trivial) ---
_INJECT: dict[str, object] = {"K": None, "n_K_calls": 0,
                              "n_evd_keep": None, "evd_keep": True}
# Cells whose K exceeds this are EVD-active (K_conv = nu_conv = 100 m^2/s
# = NEMO rn_evd) and keep legoESM's value; the TKE closure Kz never
# approaches it (measured p99.9 of NEMO avt_k = 0.75 m^2/s).
_EVD_KEEP = 50.0


def _install_k_hook():
    """Wrap ``compute_vertical_K_profiles`` so an arm can substitute NEMO's Kz.

    THE HOOK POINT WAS VERIFIED, NOT ASSUMED.  The first version of this probe
    wrapped ``tke_set_diffusivities`` (the ``buoyancy_timing=
    "post_mixing_veros"`` branch, k_profiles.py:806) and the call counter came
    back **0** -- this recipe realises ``buoyancy_timing="pre_mixing"``, so
    that branch never runs and every B arm would have silently measured
    NOTHING.  The live path is
    ``ocean_model_latlon_cgrid.py:5696`` -> ``compute_vertical_K_profiles``
    (returning ``K_v_cell, A_v_cell, tke_new``), imported from the PACKAGE at
    ``:5572``.  The counter is printed every run and a zero raises.

    EVD IS PRESERVED.  ``k_profiles.py:317-340`` composes the closure Kz with
    the enhanced-diffusion (EVD) field as ``maximum(K_closure, K_conv)``;
    NEMO does the same (``ln_zdfevd=.true.``, ``rn_evd=100``, ``nn_evdm=1`` in
    RUN_TWIN_STEP1/namelist_cfg:388-390) but writes the restart's
    ``avt_k``/``avm_k`` BEFORE ``zdf_evd`` (``zdfphy.F90:284-286`` copies
    ``avt_k`` into ``avt``, and ``zdf_evd`` then overwrites ``avt``).  So
    substituting ``avt_k`` wholesale would ALSO delete legoESM's EVD and
    confound the arm.  Instead the injection replaces only cells where EVD did
    NOT fire (``K < _EVD_KEEP``); EVD cells (K pinned at ``K_conv = 100``) keep
    legoESM's value.  The preserved-cell count is printed.
    """
    import legoesm.ocean.physics.vertical_mixing as vmix_pkg
    import legoesm.ocean.physics.vertical_mixing.k_profiles as kp_mod

    if getattr(vmix_pkg.compute_vertical_K_profiles, "_carry_probe", False):
        return
    orig = kp_mod.compute_vertical_K_profiles

    def patched(*a, **kw):
        out = orig(*a, **kw)
        _INJECT["n_K_calls"] = int(_INJECT["n_K_calls"]) + 1
        inj = _INJECT["K"]
        if inj is None:
            return out
        import jax.numpy as jnp
        K_v, A_v = out[0], out[1]
        avm, avt = inj
        if avt.shape != K_v.shape or avm.shape != A_v.shape:
            raise SystemExit(
                f"Kz injection shape mismatch: NEMO avt {avt.shape} avm "
                f"{avm.shape} vs legoESM K_v {K_v.shape} A_v {A_v.shape}")
        # ``B_all`` drops the EVD preservation entirely -- the BRACKET arm:
        # B_now keeps legoESM's convective cells (so it is blind to an
        # EVD-trigger disagreement), B_all replaces them too (so it is
        # unfaithful where NEMO's own post-``zdf_evd`` avt is 100).  If BOTH
        # sit at the base value the whole Kz field is exonerated either way.
        _keep = bool(_INJECT["evd_keep"])
        keep_k = (K_v > _EVD_KEEP) if _keep else jnp.zeros(K_v.shape, bool)
        keep_a = (A_v > _EVD_KEEP) if _keep else jnp.zeros(A_v.shape, bool)
        # (no host-side count here: K_v is a TRACER inside the jitted step)
        K_new = jnp.where(keep_k, K_v, jnp.asarray(avt, dtype=K_v.dtype))
        A_new = jnp.where(keep_a, A_v, jnp.asarray(avm, dtype=A_v.dtype))
        return (K_new, A_new) + tuple(out[2:])

    patched._carry_probe = True
    # The model imports the symbol from the PACKAGE
    # (ocean_model_latlon_cgrid.py:5572 ``from
    # legoesm.ocean.physics.vertical_mixing import ...
    # compute_vertical_K_profiles``), not from ``k_profiles`` -- patching only
    # the defining module leaves the package alias untouched and the hook
    # never fires (measured: 0 calls).  Patch BOTH.
    kp_mod.compute_vertical_K_profiles = patched
    vmix_pkg.compute_vertical_K_profiles = patched


def _nemo_restart_fields(kt: int, names: list[str]) -> dict[str, np.ndarray]:
    """Stitched restart fields at step ``kt``, (y, x[, z]) float64.

    Same stitcher and same ``(z,y,x)->(y,x,z)`` halo-free reshape
    ``multistep_replay._rebuild_step`` uses (that symbol is private, so the
    two-line public equivalent is written out rather than imported).
    """
    from multistep_replay import RUN_TWIN_STEP1
    from rebuild_nemo_restart import rebuild

    raw = rebuild(f"{RUN_TWIN_STEP1}/DINO_{kt:08d}_restart_*.nc", names)
    missing = [n for n in names if n not in raw]
    if missing:
        raise SystemExit(f"restart {kt} missing {missing}")
    out = {}
    for n, a in raw.items():
        a = np.asarray(a, dtype=np.float64)
        out[n] = np.moveaxis(a, 0, -1) if a.ndim == 3 else a
    return out


def _field_stats(diff: np.ndarray, mask: np.ndarray) -> dict:
    """max / argmax / percentiles of |diff| over the masked set.

    ``diff`` is the FULL-SHAPE difference and ``mask`` a boolean broadcastable
    to it, so the argmax can be reported as a real grid index (Rule: keep
    argmax metadata -- a max over tiles can peak at different physical
    locations in different runs, and a claim of "localised" must be checkable).
    """
    m = np.broadcast_to(mask, diff.shape)
    a = np.abs(diff)
    idx = np.unravel_index(int(np.argmax(np.where(m, a, -np.inf))), a.shape)
    sel = a[m]
    mx = float(a[idx])
    return {
        "max": mx,
        "argmax": tuple(int(v) for v in idx),
        "p99.9": float(np.percentile(sel, 99.9)),
        "p99": float(np.percentile(sel, 99.0)),
        "p50": float(np.percentile(sel, 50.0)),
        "n_gt_10pct": int((sel > 0.1 * mx).sum()) if mx > 0 else 0,
        "n_gt_1pct": int((sel > 0.01 * mx).sum()) if mx > 0 else 0,
        "n": int(sel.size),
    }


def _one_step(arm, st, br, cfg, model, forcing, kt, t_seconds,
              nemo_rate, nemo_K_prev, nemo_K_now):
    """Advance ONE step from the bridged state under arm ``arm``.

    ``kt`` is the step being TAKEN (state goes from restart[kt-1] to
    restart[kt]).
    """
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
        dino_step_surface_forcing,
    )

    # #1455 retraction fix: the card routes the WIND MOMENTUM through
    # model.step(surface_forcing=sf) (run_dino.py:665-670,763-767); the
    # applicator SKIPS its eq-7 wind when wind_through_step=True
    # (dino.py:3756), so surface_forcing=None dropped the wind entirely.
    # EVERY recorded arm/ratio of this probe (avt_k/avm_k exonerations,
    # A/B placement ratios, the 0.16 K/step base) was measured WIND-OFF.
    _wind = bool(getattr(cfg, "wind_through_step", False))
    sf_step = dino_step_surface_forcing(forcing) if _wind else None

    _INJECT["K"] = None
    _INJECT["evd_keep"] = arm != "B_all"
    if arm == "B_prev":
        _INJECT["K"] = nemo_K_prev
    elif arm in ("B_now", "B_all", "AB"):
        _INJECT["K"] = nemo_K_now

    placement = ("leapfrog_rhs"
                 if arm in ("A_lf", "null_rate", "A_nemo_avg", "A_nemo_now",
                            "AB")
                 else "applied_now")
    cfg_arm = dataclasses.replace(cfg, surface_tendency_placement=placement)

    if placement == "leapfrog_rhs":
        st, rate = apply_dino_lat_lon_surface_forcing(
            st, forcing, br.z_coord, cfg_arm, DT, t_seconds=t_seconds,
            return_rate=True)
        if arm == "A_nemo_avg" or arm == "AB":
            rate = nemo_rate["avg"]
        elif arm == "A_nemo_now":
            rate = nemo_rate["now"]
        # null_rate / A_lf: legoESM's own rate, pushed through the same slot.
        st = model.step(st, DT, surface_forcing=sf_step,
                        external_tracer_rate=rate)
    else:
        st = apply_dino_lat_lon_surface_forcing(
            st, forcing, br.z_coord, cfg_arm, DT, t_seconds=t_seconds)
        st = model.step(st, DT, surface_forcing=sf_step,
                        external_tracer_rate=None)
    _INJECT["K"] = None
    return st


def main(argv: list[str]) -> int:
    n_states = int(argv[1]) if len(argv) > 1 else 4
    arms = tuple(argv[2].split(",")) if len(argv) > 2 else ALL_ARMS
    bad = [a for a in arms if a not in ALL_ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; known: {ALL_ARMS}")

    import jax.numpy as jnp  # noqa: F401  (imported for the hook)
    import multistep_replay as mr
    from legoesm.core.precision import get_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.experiments.dino import (
        dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    if not mr.have_step1_artifacts():
        print("SKIP: RUN_TRAJ/RUN_TWIN_STEP1 oracle artifacts not present")
        return 0

    print(f"PRECISION control dtype = {get_policy().control}")
    print(f"LEGOESM_NEMO_E3T = {os.environ.get('LEGOESM_NEMO_E3T')!r}")
    print(f"n_states = {n_states}   arms = {arms}")

    _install_k_hook()

    base_ic = mr.IC_STEP
    model = None
    results: dict[str, list] = {a: [] for a in arms}
    base_stats: list[dict] = []

    try:
        for j in range(n_states):
            mr.IC_STEP = base_ic + j
            kt_entry = base_ic + j
            kt = kt_entry + 1
            g, br, cfg, st0 = mr.build_replay_ic()

            if model is None:
                mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
                model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
                forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
                # FORCING state, adjacent to the day-0 gate (printed by
                # build_replay_ic just above): _one_step passes sf iff
                # wind_through_step (#1455 retraction fix).
                _wind = bool(getattr(cfg, "wind_through_step", False))
                if _wind:
                    from legoesm.ocean.experiments.dino import (
                        dino_step_surface_forcing)
                    _sf = dino_step_surface_forcing(forcing)
                    _tlo = float(np.min(np.asarray(_sf.tau_x)))
                    _thi = float(np.max(np.asarray(_sf.tau_x)))
                else:
                    _tlo = _thi = 0.0
                print(f"  FORCING: wind_through_step={_wind} "
                      f"surface_stress_implicit="
                      f"{getattr(cfg, 'surface_stress_implicit', None)} "
                      f"tau_x[Pa] range=[{_tlo:.4f},{_thi:.4f}]", flush=True)
                zc = br.z_coord
                for f in ("t_depth_ref", "dz_ref", "z_full_ref", "z_half_ref"):
                    if hasattr(zc, f):
                        print(f"  GEOMETRY dtype {f} = "
                              f"{np.asarray(getattr(zc, f)).dtype}")
                print(f"  STATE dtypes T={st0.T.data.dtype} "
                      f"S={st0.S.data.dtype} u={st0.u.data.dtype} "
                      f"eta={st0.eta.data.dtype} tke={st0.tke.data.dtype} "
                      f"H_bathy={st0.H_bathy.data.dtype}")
                tmask3 = np.asarray(g.tmask) > 0.5
                umask3 = np.asarray(g.umask) > 0.5
                vmask3 = np.asarray(g.vmask) > 0.5
                wet2 = np.asarray(br.land_mask.data) > 0.5
                h_bathy = st0.H_bathy.data

            # --- NEMO's own carries for the step kt_entry -> kt -----------
            need = ["sbc_hc_b", "sbc_sc_b", "qsr_hc_b", "avm_k", "avt_k"]
            f_prev = _nemo_restart_fields(kt_entry, need)
            f_now = _nemo_restart_fields(kt, need)

            e3t_entry = np.asarray(compute_layer_thickness(
                st0.eta.data, h_bathy, br.z_coord,
                min_water_column_m=mc.min_water_column_m))
            m3 = np.asarray(br.land_mask.data)[..., None]

            def _rate(w_prev: float):
                """trasbc.F90:152 + traqsr.F90:206 with weight ``w_prev`` on
                the before-value (0.5 = NEMO; 0.0 = now-only ablation)."""
                w_now = 1.0 - w_prev
                hc = w_prev * f_prev["qsr_hc_b"] + w_now * f_now["qsr_hc_b"]
                dT = hc / e3t_entry
                sbc_t = (w_prev * f_prev["sbc_hc_b"]
                         + w_now * f_now["sbc_hc_b"])
                sbc_s = (w_prev * f_prev["sbc_sc_b"]
                         + w_now * f_now["sbc_sc_b"])
                dT[..., 0] += sbc_t / e3t_entry[..., 0]
                dS = np.zeros_like(dT)
                dS[..., 0] = sbc_s / e3t_entry[..., 0]
                # tmask3 (3-D wet) not the 2-D surface mask: e3t is 0 below
                # the seafloor inside a wet COLUMN, so a 2-D check would fire
                # on legitimately dry sub-seafloor levels.
                if not (np.isfinite(dT[tmask3]).all()
                        and np.isfinite(dS[tmask3]).all()):
                    raise SystemExit(
                        "non-finite NEMO surface tracer trend over wet cells")
                dT = np.where(tmask3, dT, 0.0)
                dS = np.where(tmask3, dS, 0.0)
                return (jnp.asarray(dT), jnp.asarray(dS))

            nemo_rate = {"avg": _rate(0.5), "now": _rate(0.0)}

            def _K(f):
                avm = f["avm_k"][..., 1:]
                avt = f["avt_k"][..., 1:]
                if not np.isfinite(avm[wet2]).all():
                    raise SystemExit("non-finite avm_k over wet columns")
                return (avm, avt)

            nemo_K_prev, nemo_K_now = _K(f_prev), _K(f_now)

            ns = mr.nemo_now_state_at(kt)
            nb = mr.nemo_before_state_at(kt)
            en_nemo = mr.nemo_en_at(kt)[..., 1:]

            for arm in arms:
                # A monkeypatched closure constant is baked in at TRACE time:
                # ``model.step`` is jitted, so the SECOND arm reuses the first
                # arm's compiled step and the injection silently does nothing
                # (measured: hook fired 1x for 4 arms, all four numbers
                # identical).  Clearing the cache forces a retrace per arm.
                import jax
                import time as _time
                jax.clear_caches()
                _t0 = _time.time()
                t_sec = (j + 1) * DT if arm == "tsec" else 1 * DT
                st = _one_step(arm, st0, br, cfg, model, forcing,
                               kt, t_sec, nemo_rate, nemo_K_prev, nemo_K_now)

                print(f"    [{arm}] step wall {_time.time() - _t0:6.1f}s "
                      f"K-calls={_INJECT['n_K_calls']}", flush=True)
                dT = np.asarray(st.T.data)[tmask3] - ns.T[tmask3]
                dS = np.asarray(st.S.data)[tmask3] - ns.S[tmask3]
                results[arm].append((float(np.abs(dT).max()),
                                     float(np.abs(dS).max()),
                                     float(np.percentile(np.abs(dT), 99.9)),
                                     float(np.percentile(np.abs(dS), 99.9))))

                if arm == "base":
                    e3t_l = np.asarray(compute_layer_thickness(
                        st.eta.data, h_bathy, br.z_coord,
                        min_water_column_m=mc.min_water_column_m))
                    e3t_n = np.asarray(compute_layer_thickness(
                        ns.ssh, h_bathy, br.z_coord,
                        min_water_column_m=mc.min_water_column_m))
                    u_f = np.asarray(st.u.data)[:, 1:, :]
                    v_f = np.asarray(st.v.data)[1:, :, :]
                    ub_f = np.asarray(st.u_before.data)[:, 1:, :]
                    rows = {
                        "max_dT_now": (np.asarray(st.T.data) - ns.T, tmask3),
                        "max_dS_now": (np.asarray(st.S.data) - ns.S, tmask3),
                        "max_du_now": (u_f - ns.u, umask3),
                        "max_dv_now": (v_f - ns.v, vmask3),
                        "max_deta_now": (
                            np.asarray(st.eta.data) - ns.ssh, wet2),
                        "max_den": (np.asarray(st.tke.data) - en_nemo,
                                    wet2[..., None]),
                        "max_de3t": (e3t_l - e3t_n, tmask3),
                        "max_dT_before": (
                            np.asarray(st.T_before.data) - nb.T, tmask3),
                        "max_du_before": (ub_f - nb.u, umask3),
                    }
                    if os.environ.get("DUMP_FIELDS"):
                        # Rule "diff ARRAYS, never printed summaries": the
                        # per-state percentile TABLE matching to 5 digits is
                        # suggestive, not proof, that the divergence FIELD is
                        # state-independent.  Dump the fields so they can be
                        # differenced directly.
                        np.savez(f"{os.environ['DUMP_FIELDS']}/fields_j{j}.npz",
                                 **{k: np.asarray(v[0])
                                    for k, v in rows.items()})
                    base_stats.append(
                        {k: _field_stats(d, m)
                         for k, (d, m) in rows.items()})
            print(f"  state j={j} (kt {kt_entry}->{kt}) done", flush=True)
    finally:
        mr.IC_STEP = base_ic

    print(f"\n[hook] compute_vertical_K_profiles calls = "
          f"{_INJECT['n_K_calls']} "
          f"({_INJECT['n_K_calls'] / max(1, n_states * len(arms)):.2f}/step)"
          )
    if _INJECT["n_K_calls"] == 0:
        raise SystemExit("HOOK NEVER FIRED -- every B arm measured nothing")

    # ---------------- (a) metric artifact ------------------------------
    if base_stats:
        print("\n=== (a) BASE arm per-field structure (RESET, per state) ===")
        print(f"{'field':<15s} {'state':>5s} {'max':>11s} {'p99.9':>11s} "
              f"{'p99':>11s} {'p50':>11s} {'>10%':>6s} {'>1%':>7s} "
              f"{'n':>7s}  argmax")
        for k in base_stats[0]:
            for j, s in enumerate(base_stats):
                d = s[k]
                print(f"{k:<15s} {j:5d} {d['max']:11.4e} {d['p99.9']:11.4e} "
                      f"{d['p99']:11.4e} {d['p50']:11.4e} "
                      f"{d['n_gt_10pct']:6d} {d['n_gt_1pct']:7d} "
                      f"{d['n']:7d}  {d['argmax']}")

    # ---------------- (b) discriminator --------------------------------
    print("\n=== (b) CARRY-INJECTION DISCRIMINATOR "
          "(RESET arm, per-state max|lego-NEMO|) ===")
    print(f"{'arm':<12s} " + " ".join(f"{'j=%d dT' % j:>11s}"
                                       for j in range(n_states))
          + f" {'mean dT':>11s} {'mean dS':>11s} {'mean p99.9 dT':>14s}")
    for arm in arms:
        r = np.array(results[arm])
        print(f"{arm:<12s} " + " ".join(f"{v:11.4e}" for v in r[:, 0])
              + f" {r[:, 0].mean():11.4e} {r[:, 1].mean():11.4e}"
                f" {r[:, 2].mean():14.4e}")

    def _rel(a, b):
        ra, rb = np.array(results[a]), np.array(results[b])
        return ra[:, 0].mean() / rb[:, 0].mean(), ra[:, 1].mean() / rb[:, 1].mean()

    print("\n--- CONTROLS (must be exactly 1.000000 / 1.000000) ---")
    for a, b in (("null_K", "base"), ("null_rate", "A_lf")):
        if a in results and b in results:
            t, s = _rel(a, b)
            ok = "OK" if (abs(t - 1) < 1e-12 and abs(s - 1) < 1e-12) else \
                "*** HOOK PERTURBS -- B/A numbers UNREADABLE ***"
            print(f"  {a}/{b}: dT {t:.12f}  dS {s:.12f}   {ok}")

    print("\n--- ratios (arm / reference), <1 means the injection HELPED ---")
    for a, b in (("B_prev", "base"), ("B_now", "base"),
                 ("B_all", "base"),
                 ("A_lf", "base"),
                 ("A_nemo_avg", "A_lf"), ("A_nemo_now", "A_lf"),
                 ("A_nemo_avg", "A_nemo_now"), ("AB", "A_lf"),
                 ("tsec", "base")):
        if a in results and b in results:
            t, s = _rel(a, b)
            print(f"  {a:<12s}/{b:<12s}  dT {t:8.4f}   dS {s:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
