#!/usr/bin/env python
"""#1226 y20 per-step injection: CONFIRMING ARM for the ONE statement-level
DIFF the traadv_fct read named -- the PRE-ADVECTION PHYSICS FOLD.

THE DIFF (oracle lines both sides, Rule 0)
------------------------------------------
NEMO ``tra_adv_fct`` advects the RAW leap-frog levels:

  * upstream flux + upstream guess (``fct_up1_1stp``) read
    ``pt(:,:,:,jn,Kbb)``           -- traadv_fct.F90:157 (call), :404/:411
  * the ``nonosc`` monotonicity bounds read the SAME raw ``pt(Kbb)``
    together with ``zta_up1``      -- traadv_fct.F90:311
  * the high-order (nn_fct_h=nn_fct_v=2 centred) faces read ``pt(Kmm)``
                                    -- traadv_fct.F90:186-187, :255-257

and NOTHING has been added to ``pt(Kbb)``/``pt(Kmm)`` at that point: every
earlier tracer call of the step (``tra_sbc`` stpmlf.F90:481, ``tra_qsr``
:488) writes into ``pt(Krhs)``, and ``tra_ldf`` (:531, the isoneutral /
GM-Redi lateral operator) does not run until AFTER ``tra_adv`` (:511).

legoESM advects a tracer that ALREADY carries the whole pre-advection
explicit-physics increment of the step:

  * high-order tracer   = ``T_mid``
      ocean_model_latlon_cgrid.py:4372  ``T_mid = state_new.T.data``
      ("tracer after diffusion+physics Euler step"), plus the GM/Redi
      increment at :4665 ``T_mid = T_mid + dt*dT_gm*active_3d``
  * monotonicity base   = ``T_before + (T_mid - T_now)``
      ocean_model_latlon_cgrid.py:4771-4772

so with ``Delta := T_mid - T_now`` legoESM feeds ``(Kbb + Delta, Kmm +
Delta)`` where NEMO feeds ``(Kbb, Kmm)``.  ``Delta`` is a full 2*dt
explicit-physics increment; at a mixed-layer base it is dominated by the
isoneutral/GM tendency, i.e. exactly the place the surviving y20 per-step
injection lives.

WHY THIS DIFF AND NOT THE OTHER TWO the read found
--------------------------------------------------
The read also named, and this probe does NOT test:

  D-thick   NEMO's upstream guess is ``(e3t(Kbb)*pt_b + p2dt*ztra)/e3t(Kaa)``
            (traadv_fct.F90:441) and its beta normaliser is
            ``e1e2t*e3t(Kaa)/p2dt`` (:876); legoESM passes the NOW-eta
            thickness as ``h_k`` and re-derives ``h_new`` locally
            (advection.py:1016-1039) -- the code's own docstring flags this
            as approximate under leap-frog.  BOUNDED, NOT MEASURED: the
            three thickness levels differ only by ``e3t_0*dssh/H``, i.e.
            ~1e-6 relative for DINO's ssh, so the induced ``q_td`` / box
            error is ~1e-6 K -- four orders below the 5.7e-2 K spike.
            Labelled PLAUSIBLE-negligible, not cleared.

  D-e3w     NEMO's implicit vertical solve divides the gradient by
            ``e3w(k,Kmm)`` (trazdf.F90:219-220) and NEMO's own
            ``e3w_1d(k) == gdept_1d(k)-gdept_1d(k-1)`` EXACTLY (verified
            from RUN_SEQDUMP_Y20_1R/mesh_mask.nc: relative error 0.0e+00 at
            all 36 levels).  legoESM divides by the interface MIDPOINT
            ``0.5*(dz_k+dz_k+1)`` (ocean_model_latlon_cgrid.py:6386 ->
            implicit_solver.py:472-481), and its ``dz_half_ref`` is that
            same midpoint (vertical.py:236) because
            ``create_z_star_from_thicknesses`` builds ``z_full_ref`` as the
            interface midpoint and parks NEMO's true ``gdept`` ladder in
            ``t_depth_ref`` for the PGF only.  The midpoint is 0.073% too
            large at k=1 rising to 0.324%, and 0.167% / 0.226% at the two
            spike levels k=7 / k=10.  This is a REAL metric DIFF and a
            separate work item; by magnitude it cannot own a 5.7e-2 K spike
            (it perturbs the vertical-mixing increment by ~0.2%).

ARMS (one variable, Rule 7)
---------------------------
  base       legoESM exactly as the DINO card runs it.  THE REFERENCE.
             Must reproduce the recorded spike (continuity control).
  null_fold  the substitution hook installed but handed back the arrays it
             was given.  CONTROL: must be BIT-IDENTICAL to ``base``.  If it
             is not, the hook perturbs and no arm number is readable.
  no_fold    the ONLY variable: the four tracer arguments of the advection
             call are replaced by the RAW step-entry levels -- ``T(Nnn)``,
             ``S(Nnn)`` for the high-order face and ``T(Nbb)``, ``S(Nbb)``
             for the monotonicity base -- i.e. NEMO's ``pt(Kmm)`` /
             ``pt(Kbb)``.  Nothing else changes; the physics increment is
             still applied to the state through the normal combine, it is
             simply no longer inside the advected field.

PREDICTION, written before the run (Rule: name confirm/refute first)
--------------------------------------------------------------------
  CONFIRMS  the spike (max|dT| over wet cells) collapses by >= 10x.
  REFUTES   it moves by < 2x.  Then the FCT chain is exonerated for the
            spike and the escalation shape is: advection faithful,
            vertical diffusion faithful, the injection lives in the JOIN.
  Anything in between is a partial share and is reported as such, with no
  tuning.

Run:
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  python scripts/validate/ocean_fidelity/dino_1226/fct_physics_fold_arm.py [n_states] [arms]
"""
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DT = 2700.0
ALL_ARMS = ("base", "null_fold", "no_fold")

# Substitution channel, module level so the patch closure stays trivial.
_SUB: dict[str, object] = {"mode": "off", "raw": None, "n_calls": 0}


def _install_adv_hook():
    """Wrap the advection-pair entry point so an arm can swap the tracer args.

    HOOK POINT VERIFIED, NOT ASSUMED: ``_step_impl`` calls the module-global
    ``compute_advection_flux_div_pair`` (ocean_model_latlon_cgrid.py:4799 and
    :4816) -- the public name defined in that same module at :687 -- so
    rebinding the attribute on the MODULE is what the call resolves.  The
    call counter is printed every run and a zero raises.
    """
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as mod

    if getattr(mod.compute_advection_flux_div_pair, "_fold_probe", False):
        return
    orig = mod.compute_advection_flux_div_pair

    def patched(tr_a, tr_b, tracer_advection, *a, **kw):
        _SUB["n_calls"] = int(_SUB["n_calls"]) + 1
        mode = _SUB["mode"]
        # ``_leapfrog_step`` runs ``_step_impl`` TWICE (:7895 the Nnn advective
        # pass, :7943 the Nbb dissipative pass whose advective result is
        # DISCARDED).  Only the first passes ``_fct_tracer_before``, so
        # ``tr_a_before is None`` identifies the discarded pass -- leave it
        # untouched so the arm stays single-variable.
        if mode == "off" or kw.get("tr_a_before") is None:
            _SUB["n_passthru"] = int(_SUB.get("n_passthru", 0)) + 1
            return orig(tr_a, tr_b, tracer_advection, *a, **kw)
        raw = _SUB["raw"]
        if raw is None:
            raise SystemExit("substitution arm selected but no raw levels set")
        T_now, S_now, T_bef, S_bef = raw
        if mode == "null_fold":
            # Identity: hand back exactly what we were given.  The control.
            new_a, new_b = tr_a, tr_b
            new_ab, new_bb = kw.get("tr_a_before"), kw.get("tr_b_before")
        elif mode == "no_fold":
            for nm, got, want in (("tr_a", tr_a, T_now), ("tr_b", tr_b, S_now)):
                if got.shape != want.shape:
                    raise SystemExit(
                        f"{nm} shape {got.shape} != raw {want.shape}")
            new_a, new_b = T_now, S_now
            new_ab, new_bb = T_bef, S_bef
            _SUB["n_sub"] = int(_SUB.get("n_sub", 0)) + 1
        else:
            raise SystemExit(f"unknown substitution mode {mode!r}")
        kw = dict(kw)
        kw["tr_a_before"] = new_ab
        kw["tr_b_before"] = new_bb
        return orig(new_a, new_b, tracer_advection, *a, **kw)

    patched._fold_probe = True
    mod.compute_advection_flux_div_pair = patched


def _stats(diff: np.ndarray, mask: np.ndarray) -> dict:
    m = np.broadcast_to(mask, diff.shape)
    a = np.abs(diff)
    idx = np.unravel_index(int(np.argmax(np.where(m, a, -np.inf))), a.shape)
    sel = a[m]
    mx = float(a[idx])
    return {"max": mx, "argmax": tuple(int(v) for v in idx),
            "p99.9": float(np.percentile(sel, 99.9)),
            "p50": float(np.percentile(sel, 50.0)), "n": int(sel.size)}


def main(argv: list[str]) -> int:
    n_states = int(argv[1]) if len(argv) > 1 else 1
    arms = tuple(argv[2].split(",")) if len(argv) > 2 else ALL_ARMS
    bad = [a for a in arms if a not in ALL_ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; known: {ALL_ARMS}")

    import jax
    import jax.numpy as jnp
    import multistep_replay as mr
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

    if not mr.have_step1_artifacts():
        print("SKIP: RUN_TRAJ/RUN_TWIN_STEP1 oracle artifacts not present")
        return 0

    print(f"PRECISION control dtype = {get_policy().control}")
    print(f"LEGOESM_NEMO_E3T = {os.environ.get('LEGOESM_NEMO_E3T')!r}")
    print(f"n_states = {n_states}   arms = {arms}")

    _install_adv_hook()

    base_ic = mr.IC_STEP
    model = None
    out: dict[str, list] = {a: [] for a in arms}

    try:
        for j in range(n_states):
            mr.IC_STEP = base_ic + j
            kt = base_ic + j + 1
            g, br, cfg, st0 = mr.build_replay_ic()

            if model is None:
                mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
                model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
                forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
                # RULE 10: instantiate and PRINT every resolved switch this
                # probe's claims rest on -- never quote a default.
                for f in ("tracer_advection", "outer_integrator",
                          "surface_tendency_placement", "wind_through_step",
                          "gm_bolus_advection", "tracer_wall_neumann_fill",
                          "implicit_vmix_dzw_slot",
                          "zdf_implicit_solver_evaluation",
                          "implicit_K33", "msc_stabilize"):
                    print(f"  CFG {f} = "
                          f"{getattr(cfg, f, getattr(mc, f, '<absent>'))!r}")
                print(f"  CFG dt = {getattr(cfg, 'dt', None)!r}   "
                      f"probe DT = {DT}")
                _wind = bool(getattr(cfg, "wind_through_step", False))
                _sf = dino_step_surface_forcing(forcing) if _wind else None
                _tlo = float(np.min(np.asarray(_sf.tau_x))) if _wind else 0.0
                _thi = float(np.max(np.asarray(_sf.tau_x))) if _wind else 0.0
                print(f"  FORCING: wind_through_step={_wind} "
                      f"tau_x[Pa] range=[{_tlo:.4f},{_thi:.4f}]", flush=True)
                zc = br.z_coord
                for f in ("t_depth_ref", "dz_ref", "z_full_ref",
                          "z_half_ref", "dz_half_ref"):
                    v = getattr(zc, f, None)
                    if v is not None:
                        print(f"  GEOMETRY dtype {f} = "
                              f"{np.asarray(v).dtype}")
                tmask3 = np.asarray(g.tmask) > 0.5
                h_bathy = st0.H_bathy.data

                # D-e3w magnitude, printed for the record (NOT an arm):
                # legoESM's implicit-solve w-thickness (interface midpoint)
                # vs NEMO's e3w = gdept(k)-gdept(k-1).
                td = np.asarray(getattr(zc, "t_depth_ref", None))
                dzr = np.asarray(zc.dz_ref)
                if td.ndim == 1:
                    e3w_nemo = td[1:] - td[:-1]
                    e3w_lego = 0.5 * (dzr[:-1] + dzr[1:])
                    rel = (e3w_lego - e3w_nemo) / e3w_nemo
                    print("  D-e3w rel(midpoint vs gdept-diff): "
                          f"k7={rel[6]:.3e} k10={rel[9]:.3e} "
                          f"max={np.abs(rel).max():.3e}")

            _wind = bool(getattr(cfg, "wind_through_step", False))
            _placement = getattr(cfg, "surface_tendency_placement", None)
            sf_step = dino_step_surface_forcing(forcing) if _wind else None
            ns = mr.nemo_now_state_at(kt)

            raw = (jnp.asarray(st0.T.data), jnp.asarray(st0.S.data),
                   jnp.asarray(st0.T_before.data),
                   jnp.asarray(st0.S_before.data))

            states = {}
            for arm in arms:
                # A monkeypatched constant is baked in at TRACE time; the
                # jitted step must be re-traced per arm or arm 2 silently
                # reuses arm 1's graph (carry_injection_discriminator's
                # lesson).
                jax.clear_caches()
                _SUB["mode"] = "off" if arm == "base" else arm
                _SUB["raw"] = raw
                # PRODUCTION placement (printed above): 'leapfrog_rhs' =>
                # the surface tracer trend is threaded into the Nnn leap-frog
                # RHS, not applied post-step (dino.py:3584 refuses the mix).
                if _placement == "leapfrog_rhs":
                    st, rate = apply_dino_lat_lon_surface_forcing(
                        st0, forcing, br.z_coord, cfg, DT, t_seconds=DT,
                        return_rate=True)
                else:
                    st = apply_dino_lat_lon_surface_forcing(
                        st0, forcing, br.z_coord, cfg, DT, t_seconds=DT)
                    rate = None
                st = model.step(st, DT, surface_forcing=sf_step,
                                external_tracer_rate=rate)
                _SUB["mode"] = "off"
                states[arm] = st
                dT = _stats(np.asarray(st.T.data) - ns.T, tmask3)
                dS = _stats(np.asarray(st.S.data) - ns.S, tmask3)
                out[arm].append((dT, dS))
                print(f"    [{arm}] adv-calls={_SUB['n_calls']} "
                      f"sub={_SUB.get('n_sub', 0)} "
                      f"passthru={_SUB.get('n_passthru', 0)}  "
                      f"dT max={dT['max']:.6e} @{dT['argmax']} "
                      f"p99.9={dT['p99.9']:.3e} p50={dT['p50']:.3e} | "
                      f"dS max={dS['max']:.6e} @{dS['argmax']} "
                      f"p99.9={dS['p99.9']:.3e}", flush=True)

            if "base" in states and "null_fold" in states:
                d = float(np.abs(np.asarray(states["null_fold"].T.data)
                                 - np.asarray(states["base"].T.data)).max())
                verdict = ("BIT-IDENTICAL" if d == 0.0 else
                           "NOT identical -- hook perturbs, arms UNREADABLE")
                print(f"    CONTROL null_fold vs base  max|dT| = {d:.3e} "
                      f"({verdict})")
    finally:
        mr.IC_STEP = base_ic

    if _SUB["n_calls"] == 0:
        raise SystemExit("advection hook never fired -- wrong hook point")

    print("\n=== SUMMARY (max|lego - NEMO| over wet cells, per step) ===")
    for arm in arms:
        for j, (dT, dS) in enumerate(out[arm]):
            print(f"  {arm:10s} state{j}  dT {dT['max']:.6e} @{dT['argmax']}"
                  f"  dS {dS['max']:.6e} @{dS['argmax']}")
    if "base" in arms and "no_fold" in arms:
        for j in range(len(out["base"])):
            b = out["base"][j][0]["max"]
            n = out["no_fold"][j][0]["max"]
            r = b / n if n > 0 else float("inf")
            verdict = ("CONFIRMS (>=10x collapse)" if r >= 10.0 else
                       "REFUTES (<2x)" if r < 2.0 else "PARTIAL")
            print(f"  state{j} spike ratio base/no_fold = {r:.3f}  -> "
                  f"{verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
