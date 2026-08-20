"""#1226 canonical state-initialized 90-day twin runner.

A "twin" run initializes legoESM directly from a *developed* NEMO restart
(bridged onto legoESM's staggered C-grid via
:func:`bridge_nemo_to_legoesm_topo`) rather than from the analytic paper rest
state, then integrates forward so the two models can be compared step-for-step
/ day-for-day starting from an identical, dynamically-active IC. This isolates
tendency/scheme mismatches from spin-up-trajectory divergence.

DAY-0 GATE (mandatory, not optional): a twin is only a twin if the legoESM
state at step 0 is bit-identical (to 1e-8) to the NEMO restart it was bridged
from, on wet cells, AND the velocity field is non-trivially nonzero. On
2026-07-24 a defect let a twin instrument silently fall back to the analytic
rest-state IC (``dino_lat_lon_state(...)``) instead of the bridged restart
(``br.state``) -- producing a "twin" that was actually a from-rest spin-up,
which invalidated every day-0..90 comparison built on top of it. This module
hardens that check into an assert-and-raise gate (`verify_day0_matches_restart`)
that runs before any integration and is unit-tested directly (see
``tests/ocean/unit/test_dino_1226_instruments.py``): a rest-state start must be
IMPOSSIBLE to smuggle through un-flagged.

INTEGRATOR-MEMORY HANDSHAKE CAVEAT: by default the bridge carries only
now-level prognostic fields (T/S/eta/u/v), NOT NEMO's internal integrator
memory (before-level leapfrog fields, TKE closure state). Days 1-4 of a
default twin therefore run on a legoESM-native "cold start" for that memory
while NEMO continues from its own warmed-up state -- expect the two
trajectories to diverge fastest during this handshake window before settling
into a slower, scheme-driven drift. Do not read days 1-4 as a
scheme-fidelity signal for a default (non-bridged) run.

``--bridge-before`` (#1317) REMOVES this caveat for the leap-frog before-level
state: it seeds ``state.{T,S,u,v,eta}_before`` from the NEMO restart's own
``tb/sb/ub/vb``/``sshb`` (the Modified-Leap-Frog integrator's third time
level), so the twin's step-0 entry state is EXACTLY NEMO's -- a real leap-frog
continuation, not a forward-Euler-from-now start. Required (not merely
optional) for ``nemo_dino_kamm_mlf``'s ``tke_n2_time_level="nemo_before"`` /
``tke_shear_production="nemo_burchard"`` axes: without it, ``model.step``
raises ``ValueError`` at step 0 (``state.T_before``/``S_before`` are ``None``
until the model's own Euler-start populates them AFTER step 1 -- too late for
a card that reads them every step from step 0). ``--bridge-tke`` (TKE closure
memory) is a SEPARATE, independent caveat/flag -- still cold-start by
default.

Usage
-----
    python kamm_twin_90d.py <recipe> <out.npz> [--days 90] [--save-3d]

NEMO artifact paths default to the machine-local oracle-build tree and can be
overridden via env vars (``DINO_NEMO_RUN_TRAJ``, ``DINO_NEMO_RUN_STEPDUMP``) or
CLI flags, for portability off this box.
"""
import argparse
import dataclasses
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    apply_dino_lat_lon_surface_forcing,
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)
from legoesm.core.field import Field
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
    read_nemo_restart_en,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_before_state_topo,
    bridge_nemo_to_legoesm_topo,
)

# NEMO oracle-build artifact roots (mesh/restart donors). Override via env var
# or --run-traj/--run-stepdump for a different machine/build layout.
RUN_TRAJ = os.environ.get(
    "DINO_NEMO_RUN_TRAJ",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ",
)
RUN_STEPDUMP = os.environ.get(
    "DINO_NEMO_RUN_STEPDUMP",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP",
)
RESTART_FILE = "DINO_00005760_restart.nc"  # developed day-180 state

DT = float(__import__("os").environ.get("DINO_DT", "2700.0"))
STEPS_PER_DAY = 32  # 32 * 2700s = 86400s = 1 day
SNAP_DAYS = (0, 30, 60, 90)  # full 3-D T/S snapshot days when --save-3d


def verify_day0_matches_restart(st, restart_state, land_mask, *, tol: float = 1e-8) -> None:
    """Day-0 gate: raise SystemExit unless ``st`` == the NEMO restart on wet cells.

    Guards against the 2026-07-24 defect where a "twin" silently started from
    the analytic rest-state IC instead of the bridged restart. Checks:
      (1) max|dT| = max|d_eta| = max|du| = max|dv| = 0 (to `tol`) vs the raw
          NEMO restart, restricted to wet cells;
      (2) max|u0| > 0.1 -- a rest-state start has u == 0 identically, so any
          twin claiming a developed IC must show real velocity.

    Parameters
    ----------
    st : legoESM ocean state (post-bridge, pre-integration)
    restart_state : NemoState (the raw restart read by ``read_nemo_restart``)
    land_mask : (n_lat, n_lon) wet mask (``br.land_mask``)
    """
    wet = np.asarray(land_mask) > 0.5
    t0_lego = np.asarray(st.T.data[:, :, 0])
    t0_nemo = np.asarray(restart_state.T[:, :, 0])
    eta0_lego = np.asarray(st.eta.data)
    eta0_nemo = np.asarray(restart_state.ssh)
    u0_lego = np.asarray(st.u.data[:, 1:, 0])   # east face of cell i -> NEMO un[i]
    u0_nemo = np.asarray(restart_state.u[:, :, 0])
    v0_lego = np.asarray(st.v.data[1:, :, 0])   # north face of cell j -> NEMO vn[j]
    v0_nemo = np.asarray(restart_state.v[:, :, 0])

    d_t = float(np.max(np.abs(t0_lego[wet] - t0_nemo[wet])))
    d_eta = float(np.max(np.abs(eta0_lego[wet] - eta0_nemo[wet])))
    d_u = float(np.max(np.abs(u0_lego[wet] - u0_nemo[wet])))
    d_v = float(np.max(np.abs(v0_lego[wet] - v0_nemo[wet])))
    max_u0 = float(np.max(np.abs(u0_lego[wet])))
    max_v0 = float(np.max(np.abs(v0_lego[wet])))

    print(f"DAY-0 VERIFY vs NEMO restart (wet cells): "
          f"max|dT|={d_t:.3e}  max|d_eta|={d_eta:.3e}  max|du|={d_u:.3e}  max|dv|={d_v:.3e}  "
          f"max|u0|={max_u0:.4f}  max|v0|={max_v0:.4f}", flush=True)

    failures = []
    if not d_t < tol:
        failures.append(f"T mismatch vs NEMO restart: {d_t:.3e} >= {tol:.1e}")
    if not d_eta < tol:
        failures.append(f"eta mismatch vs NEMO restart: {d_eta:.3e} >= {tol:.1e}")
    if not d_u < tol:
        failures.append(f"u mismatch vs NEMO restart: {d_u:.3e} >= {tol:.1e}")
    if not d_v < tol:
        failures.append(f"v mismatch vs NEMO restart: {d_v:.3e} >= {tol:.1e}")
    if not max_u0 > 0.1:
        failures.append(
            f"max|u0|={max_u0:.4f} <= 0.1 -- looks like a REST-STATE start, not a "
            "developed-restart twin (2026-07-24 defect: verify the caller passed "
            "br.state, not dino_lat_lon_state(...))"
        )
    if not max_v0 > 0.1:
        failures.append(
            f"max|v0|={max_v0:.4f} <= 0.1 -- looks like a REST-STATE start, not a "
            "developed-restart twin (2026-07-24 defect: verify the caller passed "
            "br.state, not dino_lat_lon_state(...))"
        )
    if failures:
        raise SystemExit(
            "DAY-0 GATE FAILED -- refusing to run a twin that is not verifiably "
            "initialized from the NEMO restart:\n  " + "\n  ".join(failures)
        )


def bridge_tke_from_restart(st, restart_en, land_mask):
    """Seed ``st.tke`` from a NEMO restart's ``en`` (TKE closure integrator memory).

    ``restart_en`` is the raw ``(n_lat, n_lon, jpk)`` array from
    :func:`read_nemo_restart_en` (index 0 = surface w-level, matching
    ``gdepw_1d``). legoESM's ``state.tke`` is ``(n_lat, n_lon, nlev-1)`` at
    the interior interfaces (dims ``("lat","lon","level")``); since
    ``jpk == nlev`` (NEMO w/T-levels share one ``nav_lev`` axis), dropping the
    surface w-level index (``restart_en[..., 1:]``) leaves exactly ``nlev-1``
    levels aligned index-for-index with lego's interior interfaces. Masked to
    wet columns (matches the cold-start seed's ``land_mask``-gated fill).
    """
    en_interior = np.asarray(restart_en, dtype=np.float64)[..., 1:]  # drop w-level 0 (surface)
    wet = (np.asarray(land_mask) > 0.5)[:, :, None]
    tke_data = jnp.asarray(np.where(wet, en_interior, 0.0), dtype=st.T.data.dtype)
    return st._replace(tke=Field(data=tke_data, name="tke",
                                  dims=("lat", "lon", "level"), units="m^2/s^2"))


def _print_before_bridge_verify(st, before, grid) -> None:
    """Print max|d_tb|/max|d_sb|/max|d_ub|/max|d_vb| vs the raw restart
    before-level (wet cells) -- the --bridge-before day-0 gate companion to
    ``verify_day0_matches_restart``'s now-level check.

    T/S use the FULL 3-D ``tmask`` (not the 2-D surface ``land_mask``): under
    full-step topography a wet surface column still has dry cells below
    ``k_bot``, where NEMO stores a raw 0.0 but the bridge's Neumann-fill
    extrapolates a nonzero value (matches the now-level bridge's own T/S
    fill) -- indexing those cells with the 2-D mask would spuriously flag
    the intentional fill as a mismatch.
    """
    tmask3 = np.asarray(grid.tmask) > 0.5
    d_tb = float(np.max(np.abs(np.asarray(st.T_before.data)[tmask3] - before.T[tmask3])))
    d_sb = float(np.max(np.abs(np.asarray(st.S_before.data)[tmask3] - before.S[tmask3])))
    umask3 = np.asarray(grid.umask) > 0.5
    vmask3 = np.asarray(grid.vmask) > 0.5
    d_ub = float(np.max(np.abs(
        np.asarray(st.u_before.data)[:, 1:, :][umask3] - before.u[umask3])))
    d_vb = float(np.max(np.abs(
        np.asarray(st.v_before.data)[1:, :, :][vmask3] - before.v[vmask3])))
    print(f"BEFORE-LEVEL BRIDGE VERIFY vs NEMO restart tb/sb/ub/vb (wet cells): "
          f"max|d_tb|={d_tb:.3e}  max|d_sb|={d_sb:.3e}  max|d_ub|={d_ub:.3e}  "
          f"max|d_vb|={d_vb:.3e}", flush=True)


def _build_twin_state(recipe: str, run_traj: str, run_stepdump: str, *,
                       bridge_tke: bool = False, bridge_before: bool = False,
                       vmix_scheme: str | None = None,
                       use_gm_redi: bool | None = None,
                       surface_tendency_placement: str | None = None,
                       restart_file: str = RESTART_FILE):
    """Bridge the NEMO restart into a legoESM state and run the day-0 gate.

    ``restart_file``: basename of the (rebuilt, single-file) NEMO restart
    inside ``run_stepdump``. Default is the day-180 spin-up state; pass e.g.
    "DINO_00057600_restart.nc" to twin from NEMO's *year-5* state, which turns
    the twin into a matched-state GROWTH-RATE test (same developed state, same
    window, one variable = the model) rather than a from-rest comparison.

    ``vmix_scheme``: optional override of ``DINOConfig.vmix_scheme`` (e.g.
    "constant" for the #1317 TKE-vs-constant-mixing discriminator). ``None``
    (default) leaves the recipe's own vmix_scheme untouched.

    ``use_gm_redi``: optional override of ``DINOConfig.use_gm_redi`` (GM
    ablation discriminator: False zeroes the eddy-induced/bolus coefficient
    -- kappa_GM and the Treguier/Visbeck adaptive-kappa diagnostics -- while
    leaving kappa_Redi / the isoneutral slope machinery untouched and active,
    since dino.py's isoneutral GMRediConfig branch builds Redi unconditionally
    and only gates kappa_GM/treguier.enabled/visbeck.enabled on this flag;
    see dino.py:2437-2503). ``None`` (default) leaves the recipe's own value.

    ``surface_tendency_placement``: optional override of
    ``DINOConfig.surface_tendency_placement`` (#1492 A/B: "applied_now" is
    the legacy defect path -- surface_forcing mutates T/S directly before
    model.step(); "leapfrog_rhs" is the NEMO-faithful fix -- the surface
    tendency rate is folded into the Nnn RHS instead, see dino.py:262-281).
    ``None`` (default) leaves the recipe's own value.

    Returns (br, cfg, mc, model, forcing, sf, st) ready to integrate.
    """
    g = read_nemo_mesh_mask(f"{run_traj}/mesh_mask.nc", nn_hls=0)
    s = read_nemo_restart(f"{run_stepdump}/{restart_file}", nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    cfg = dataclasses.replace(dino_config_for_recipe(recipe),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    if vmix_scheme is not None:
        cfg = dataclasses.replace(cfg, vmix_scheme=vmix_scheme)
    if use_gm_redi is not None:
        cfg = dataclasses.replace(cfg, use_gm_redi=use_gm_redi)
    if surface_tendency_placement is not None:
        cfg = dataclasses.replace(cfg, surface_tendency_placement=surface_tendency_placement)
    _ba = os.environ.get("DINO_BOLUS_ADV")
    if _ba:
        # #1226: "through_fct" folds the GM bolus into the ADVECTING MASS FLUX;
        # "centred" applies it as a tendency instead. On NEMO's true vertical
        # grid the bolus transport is 37% larger, so this isolates whether the
        # instability arrives through the advecting velocity.
        cfg = dataclasses.replace(cfg, gm_bolus_advection=_ba)
        print(f"ABLATION: gm_bolus_advection={_ba}")
    _rt = os.environ.get("DINO_RECONCILE_TARGET")
    if _rt:
        # NEMO dyn_spg_ts N6 (dynspg_ts.F90:1170): which time-averaged barotropic
        # mean the 3-D momentum depth-mean is reconciled onto ("velocity_avg" =
        # primary boxcar; "transport_avg" = un_adv/hu secondary/transport mean).
        # The confirmation A/B for the barotropic-reconcile term: one variable.
        cfg = dataclasses.replace(cfg, barotropic_reconcile_target=_rt)
        print(f"ABLATION: barotropic_reconcile_target={_rt}")

    # CRITICAL: st MUST be the NEMO-restart-carrying bridged state (br.state)
    # -- NOT dino_lat_lon_state(...) (the analytic paper-IC rest state), which
    # would silently spin up from rest instead of twinning the restart. See
    # the 2026-07-24 defect note in the module docstring.
    st = br.state
    print("twin from developed NEMO state (br.state, NOT dino_lat_lon_state)")
    verify_day0_matches_restart(st, s, br.land_mask)

    # OPTIONAL: bridge NEMO's leap-frog BEFORE-level state (tb/sb/ub/vb, the
    # MLF integrator's THIRD time level) onto state.{T,S,u,v,eta}_before, so
    # the twin's leapfrog entry state is EXACTLY NEMO's -- not a
    # forward-Euler cold start (see kamm_twin_90d.py module docstring: this
    # removes the integrator-memory caveat for tracers/velocities).
    # nemo_dino_kamm_mlf sets tke_n2_time_level="nemo_before" +
    # tke_shear_production="nemo_burchard", both of which READ these fields
    # every step -- bridging is what makes those axes correct from step 0
    # instead of only after the model's own Euler-start populates them.
    if bridge_before:
        before = read_nemo_restart_before(f"{run_stepdump}/{restart_file}", nn_hls=0)
        st = bridge_before_state_topo(br._replace(state=st), g, before, periodic_i=True)
        _print_before_bridge_verify(st, before, g)

    # OPTIONAL: bridge NEMO's developed TKE closure memory (`en`) onto lego's
    # cold-start `state.tke` -- isolates whether the TKE cold-start (vs the
    # bridged prognostic T/S/eta/u/v) drives the day 0-4 surface-layer
    # handshake divergence (#1317). Off by default (matches the module
    # docstring's documented cold-start caveat) so `--bridge-tke` is additive,
    # not a silent behavior change.
    if bridge_tke:
        en_restart = read_nemo_restart_en(f"{run_stepdump}/{restart_file}", nn_hls=0)
        st = bridge_tke_from_restart(st, en_restart, br.land_mask)
        wet = np.asarray(br.land_mask) > 0.5
        d_en = float(np.max(np.abs(
            np.asarray(st.tke.data)[wet] - en_restart[..., 1:][wet])))
        print(f"TKE BRIDGE: seeded state.tke from NEMO restart en "
              f"(w-level 1..{en_restart.shape[-1]-1} -> interior interface "
              f"0..{en_restart.shape[-1]-2})  max|d_en|={d_en:.3e}", flush=True)

    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    if os.environ.get("DINO_NEMO_KMM_DIVISOR") is not None:
        # #1226 W1: NEMO-faithful implicit-solve gradient divisor (trazdf.F90:
        # 219-220 e3w(...,Kmm), NOW/pre-solve thickness) vs legoESM's default
        # AFTER-solve midpoint divisor. LatLonCGridOceanConfig field, set on
        # mc post-construction (same pattern as dino_year_screen_fullframe.py).
        _v = os.environ["DINO_NEMO_KMM_DIVISOR"]
        if _v not in ("0", "1"):
            raise SystemExit(
                f"Unknown DINO_NEMO_KMM_DIVISOR={_v!r}: expected '0' or '1'")
        # mc is a NamedTuple (LatLonCGridOceanConfig), not a dataclass -> _replace.
        mc = mc._replace(implicit_vmix_e3t_now_divisor=(_v == "1"))
        print(f"ABLATION: implicit_vmix_e3t_now_divisor={mc.implicit_vmix_e3t_now_divisor}")
    # #1492 P2: NEMO-faithful step-composition A/B (docs/ocean/fidelity/
    # nemo_mlf_step_transcription_spec.md resolved decision 2). Default "" =
    # legacy (outer_integrator="leapfrog", unchanged); "nemo_mlf" routes to
    # the single-pass stpmlf.F90 transcription via the REAL dispatch. Opt-in
    # measurement knob only -- NOT a recipe/kamm default (same pattern as
    # dino_year_screen_fullframe.py). Unknown value raises.
    _OI = os.environ.get("DINO_OUTER_INTEGRATOR", "")
    if _OI:
        if _OI not in ("leapfrog", "nemo_mlf"):
            raise SystemExit(
                f"Unknown DINO_OUTER_INTEGRATOR={_OI!r}: expected "
                "'leapfrog' or 'nemo_mlf'")
        # nemo_mlf HARD-REQUIRES the NEMO e3w(Kmm) divisor at construction
        # (spec resolved decision 4) -- auto-force it so the env knob alone
        # is sufficient without also setting DINO_NEMO_KMM_DIVISOR.
        mc = mc._replace(
            outer_integrator=_OI,
            implicit_vmix_e3t_now_divisor=(
                True if _OI == "nemo_mlf"
                else mc.implicit_vmix_e3t_now_divisor))
        print(f"ABLATION: outer_integrator={mc.outer_integrator} "
              f"implicit_vmix_e3t_now_divisor={mc.implicit_vmix_e3t_now_divisor}")
    print(f"barotropic_diffusion_alpha={mc.barotropic.barotropic_diffusion_alpha} "
          f"barotropic_face_depth={mc.barotropic.barotropic_face_depth} "
          f"zdf_drag_in_matrix={mc.zdf_drag_in_matrix} "
          f"zdf_baroclinic_only={mc.zdf_baroclinic_only} "
          f"barotropic_drag_substep={mc.barotropic_drag_substep}", flush=True)

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)
    print(f"slope_scheme={mc.gm_redi.slope_scheme} "
          f"kappa_GM_max={float(jnp.max(jnp.abs(mc.gm_redi.kappa_GM))):.1f}")
    print(f"tau_x[Pa] min/max = {float(jnp.min(sf.tau_x)):.3f}/{float(jnp.max(sf.tau_x)):.3f}")
    return br, cfg, mc, model, forcing, sf, st


def run_twin(recipe: str, out_path: str, *, n_days: int = 90, save_3d: bool = False,
             run_traj: str = RUN_TRAJ, run_stepdump: str = RUN_STEPDUMP,
             bridge_tke: bool = False, bridge_before: bool = False,
             vmix_scheme: str | None = None,
             use_gm_redi: bool | None = None,
             surface_tendency_placement: str | None = None,
             perturb_seed: int | None = None) -> bool:
    """Run the state-initialized twin for ``n_days`` and save an npz. Returns stable.

    ``perturb_seed``: optional #1492 item-2.2 noise-control lane -- if set,
    applies a 1e-14-relative multiplicative perturbation to the bridged
    now-level T (``T *= 1 + 1e-14 * N(0,1)`` per grid point,
    ``numpy.random.default_rng(perturb_seed)``), mirroring
    ``scripts/tmp/_perturb_restart_ensemble.py``'s documented ensemble
    pattern (same eps, same draw shape) but applied post-bridge to the
    legoESM state's now-level T only -- this twin runner's default
    (non-``--bridge-before``) state carries no before-level T, so there is
    no tb to perturb in lockstep.
    """
    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        recipe, run_traj, run_stepdump, bridge_tke=bridge_tke,
        bridge_before=bridge_before, vmix_scheme=vmix_scheme,
        use_gm_redi=use_gm_redi,
        surface_tendency_placement=surface_tendency_placement)

    if perturb_seed is not None:
        rng = np.random.default_rng(perturb_seed)
        t0 = np.asarray(st.T.data, dtype=np.float64)
        factor = 1.0 + 1e-14 * rng.standard_normal(t0.shape)
        t_pert = jnp.asarray(t0 * factor, dtype=st.T.data.dtype)
        d_t = float(np.max(np.abs(np.asarray(t_pert) - t0)))
        rel = float(np.max(np.abs(np.asarray(t_pert) - t0) / np.maximum(np.abs(t0), 1e-30)))
        print(f"PERTURB seed={perturb_seed}: max|dT|={d_t:.3e}  max_rel|dT/T|={rel:.3e}",
              flush=True)
        st = st._replace(T=st.T.replace(data=t_pert))
    nsteps = STEPS_PER_DAY * n_days

    # #1492: "leapfrog_rhs" placement REQUIRES return_rate=True + threading
    # the rate into model.step(external_tracer_rate=...) (run_dino.py's
    # driver wiring, mirrored; _check_surface_tendency_placement raises on a
    # mismatch, so the legacy applied_now loop cannot silently run a
    # leapfrog_rhs config).
    _sf_placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    if _sf_placement == "leapfrog_rhs":
        dyn = jax.jit(lambda st, ext: model.step(st, DT, surface_forcing=sf,
                                                  external_tracer_rate=ext))
    else:
        dyn = jax.jit(lambda st: model.step(st, DT, surface_forcing=sf))

    # #1455 SEASONAL-CLOCK OFFSET (opt-in measurement knob, default UNCHANGED).
    # DINO's analytic surface forcing is a function of the DAY OF YEAR
    # (usrdef_sbc.F90:536-547: ztime = REAL(kt)*rn_Dt, the ABSOLUTE step
    # index).  This loop passes t_seconds = (k+1)*DT, i.e. it restarts the
    # seasonal year at zero even though the bridged state is NEMO's step
    # KT0 = 5760 (RUN_90D_TWIN/ocean.output:711 -- kt 5761 is 0001/07/01,
    # nday_year 181), so T* and Qsr run half a year out of phase with the
    # NEMO run this twin is compared against.
    #   unset / "0"  -> t = (k+1)*DT               (BIT-IDENTICAL to every
    #                                               previously recorded run)
    #   "restart"    -> t = (KT0 + k+1)*DT, KT0 parsed from the restart name
    #   <integer>    -> t = (<integer> + k+1)*DT
    _kt0_env = os.environ.get("DINO_TWIN_SEASONAL_KT0", "0")
    if _kt0_env == "restart":
        _digits = "".join(c for c in RESTART_FILE if c.isdigit())
        if not _digits:
            raise SystemExit(
                f"DINO_TWIN_SEASONAL_KT0=restart but RESTART_FILE={RESTART_FILE!r} "
                "carries no step number to parse")
        kt0 = int(_digits)
    else:
        try:
            kt0 = int(_kt0_env)
        except ValueError:
            raise SystemExit(
                f"Unknown DINO_TWIN_SEASONAL_KT0={_kt0_env!r}: expected "
                "'restart' or an integer step index") from None
        if kt0 < 0:
            raise SystemExit(
                f"DINO_TWIN_SEASONAL_KT0={kt0} is negative; expected >= 0")
    print(f"seasonal clock: t_seconds = ({kt0} + k+1)*{DT:.0f}s  "
          f"(day-of-year at step 1 = {((kt0 + 1) * DT / 86400.0) % 360.0 + 1:.2f})",
          flush=True)

    land_mask = np.asarray(st.land_mask.data)
    n_lat, n_lon = br.geometry.n_lat, br.geometry.n_lon

    eta_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    sst_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    u_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    v_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)

    snap_days = tuple(d for d in SNAP_DAYS if d <= n_days) if save_3d else ()
    t3d, s3d, eta3d, u3d, v3d = {}, {}, {}, {}, {}
    if save_3d:
        t3d[0] = np.asarray(st.T.data, dtype=np.float32)
        s3d[0] = np.asarray(st.S.data, dtype=np.float32)
        eta3d[0] = np.asarray(st.eta.data, dtype=np.float32)
        # full-depth u/v faces (NOT just the surface level captured by
        # u_daily/v_daily below) -- required for ACC (acc_thermal_wind.py's
        # acc_full integrates over all NZ levels), so a snapshot day's u/v
        # must carry the whole water column, matching T3d/S3d's full depth.
        u3d[0] = np.asarray(st.u.data, dtype=np.float32)
        v3d[0] = np.asarray(st.v.data, dtype=np.float32)
        print("captured day-0 3-D T/S/u/v snapshot", flush=True)

    blew_up_at = None
    t0 = time.time()
    for k in range(nsteps):
        if _sf_placement == "leapfrog_rhs":
            st, _ext_rate = apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, DT, t_seconds=(kt0 + k + 1) * DT,
                return_rate=True)
            st = dyn(st, _ext_rate)
        else:
            st = apply_dino_lat_lon_surface_forcing(st, forcing, br.z_coord, cfg, DT,
                                                     t_seconds=(kt0 + k + 1) * DT)
            st = dyn(st)

        if (k + 1) % STEPS_PER_DAY == 0:
            day_idx = (k + 1) // STEPS_PER_DAY - 1
            day_num = day_idx + 1
            t_now3d = np.asarray(st.T.data)
            m = land_mask > 0.5
            finite = bool(np.isfinite(t_now3d[m]).all())
            tmax = float(t_now3d[m].max()) if finite else float("nan")
            tmin = float(t_now3d[m].min()) if finite else float("nan")
            unstable = not (finite and tmax < 60.0)

            eta_now = np.asarray(st.eta.data, dtype=np.float32)
            t_now = np.asarray(st.T.data[:, :, 0], dtype=np.float32)
            u_full = np.asarray(st.u.data[:, :, 0], dtype=np.float32)
            v_full = np.asarray(st.v.data[:, :, 0], dtype=np.float32)
            u_now = u_full[:, 1:]
            v_now = v_full[1:, :]

            eta_daily[day_idx] = eta_now
            sst_daily[day_idx] = t_now
            u_daily[day_idx] = u_now
            v_daily[day_idx] = v_now

            if save_3d and day_num in snap_days:
                t3d[day_num] = np.asarray(st.T.data, dtype=np.float32)
                s3d[day_num] = np.asarray(st.S.data, dtype=np.float32)
                eta3d[day_num] = np.asarray(st.eta.data, dtype=np.float32)
                u3d[day_num] = np.asarray(st.u.data, dtype=np.float32)
                v3d[day_num] = np.asarray(st.v.data, dtype=np.float32)
                print(f"  captured day {day_num} full 3-D T/S/u/v snapshot", flush=True)

            print(f"  day {(k+1)*DT/86400:6.1f}  T[{tmin:.1f},{tmax:.1f}] finite={finite} "
                  f"max|eta|={np.nanmax(np.abs(eta_now)):.4f} "
                  f"max|u|={np.nanmax(np.abs(u_now)):.4f} "
                  f"max|v|={np.nanmax(np.abs(v_now)):.4f} wall={time.time()-t0:.0f}s", flush=True)

            if unstable:
                blew_up_at = k + 1
                print(f"BLOWUP detected at step {k+1} (day {(k+1)*DT/86400:.2f}) -- "
                      f"T range [{tmin:.2f},{tmax:.2f}] finite={finite}", flush=True)
                break

    stable = blew_up_at is None
    print(f"DONE {'blew up at step ' + str(blew_up_at) if blew_up_at else 'nsteps=' + str(nsteps)} "
          f"STABLE={stable}  wall={time.time()-t0:.0f}s", flush=True)

    save_kwargs = dict(
        eta=eta_daily, sst=sst_daily, u=u_daily, v=v_daily,
        land_mask=land_mask.astype(np.float32),
        day=np.arange(1, n_days + 1, dtype=np.int32),
        blew_up_at_step=(blew_up_at if blew_up_at is not None else -1),
        stable=stable,
    )
    for d in snap_days:
        if d in t3d:
            save_kwargs[f"T3d_day{d}"] = t3d[d]
            save_kwargs[f"S3d_day{d}"] = s3d[d]
            save_kwargs[f"eta3d_day{d}"] = eta3d[d]
            save_kwargs[f"u3d_day{d}"] = u3d[d]
            save_kwargs[f"v3d_day{d}"] = v3d[d]

    np.savez(out_path, **save_kwargs)
    print(f"SAVED {out_path}  stable={stable}  "
          f"3-D snapshots at days={sorted(snap_days)}", flush=True)
    return stable


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("recipe", help="dino recipe name, e.g. nemo_dino_kamm_mlf")
    p.add_argument("out", help="output .npz path")
    p.add_argument("--days", type=int, default=90, help="twin length in days (default 90)")
    p.add_argument("--save-3d", action="store_true",
                    help="also save full 3-D T/S/eta/u/v at days 0/30/60/90")
    p.add_argument("--run-traj", default=RUN_TRAJ, help="NEMO RUN_TRAJ dir (mesh_mask donor)")
    p.add_argument("--run-stepdump", default=RUN_STEPDUMP,
                    help="NEMO RUN_STEPDUMP dir (restart donor)")
    p.add_argument("--bridge-tke", action="store_true",
                    help="seed state.tke from the NEMO restart's en (#1317 TKE "
                         "cold-start isolation experiment); default off (cold start)")
    p.add_argument("--bridge-before", action="store_true",
                    help="seed state.{T,S,u,v,eta}_before from the NEMO restart's "
                         "tb/sb/ub/vb/sshb (#1317 leap-frog before-level bridge); "
                         "required for nemo_dino_kamm_mlf's "
                         "tke_n2_time_level=nemo_before / "
                         "tke_shear_production=nemo_burchard to read a real "
                         "before-state from step 0 (else ValueError). Default "
                         "off (matches the module docstring's cold-start caveat)")
    p.add_argument("--vmix-scheme", default=None,
                    help="override DINOConfig.vmix_scheme (e.g. 'constant' for "
                         "the #1317 TKE-vs-constant-mixing discriminator); "
                         "default None leaves the recipe's own scheme untouched")
    p.add_argument("--use-gm-redi", dest="use_gm_redi", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="override DINOConfig.use_gm_redi (GM ablation "
                         "discriminator: --no-use-gm-redi zeroes kappa_GM / "
                         "disables the Treguier/Visbeck adaptive-kappa "
                         "diagnostics while leaving kappa_Redi / isoneutral "
                         "slopes untouched -- see dino.py:2437-2503); "
                         "default None leaves the recipe's own value")
    p.add_argument("--surface-tendency-placement", default=None,
                    choices=("applied_now", "leapfrog_rhs"),
                    help="override DINOConfig.surface_tendency_placement "
                         "(#1492 A/B: 'applied_now' legacy defect vs "
                         "'leapfrog_rhs' NEMO-faithful fix); default None "
                         "leaves the recipe's own value")
    p.add_argument("--perturb-seed", type=int, default=None,
                    help="#1492 item-2.2 noise control: apply a 1e-14-relative "
                         "multiplicative perturbation to the bridged now-level "
                         "T using numpy.random.default_rng(seed); default None "
                         "= no perturbation")
    return p.parse_args(argv)


def _smoke_check_vmix_scheme_override():
    """Runnable check: --vmix-scheme actually overrides cfg.vmix_scheme.

    No NEMO bridge/restart required -- exercises the same
    dataclasses.replace path _build_twin_state uses.
    """
    base = dino_config_for_recipe("nemo_dino_kamm_mlf")
    assert base.vmix_scheme == "tke", (
        f"expected nemo_dino_kamm_mlf default vmix_scheme='tke', got {base.vmix_scheme!r}")
    overridden = dataclasses.replace(base, vmix_scheme="constant")
    assert overridden.vmix_scheme == "constant"
    assert base.vmix_scheme == "tke", "override must not mutate the original cfg"
    print("OK: --vmix-scheme override changes cfg.vmix_scheme (tke -> constant)")

    assert base.use_gm_redi is True, (
        f"expected nemo_dino_kamm_mlf default use_gm_redi=True, got {base.use_gm_redi!r}")
    gm_off = dataclasses.replace(base, use_gm_redi=False)
    assert gm_off.use_gm_redi is False
    assert base.use_gm_redi is True, "override must not mutate the original cfg"
    print("OK: --use-gm-redi override changes cfg.use_gm_redi (True -> False)")

    # Base-AGNOSTIC by design: this self-check proves the OVERRIDE MECHANISM
    # works; it must not pin the card's default.  It previously asserted
    # 'applied_now' and therefore went RED the moment the card was CORRECTED
    # to 'leapfrog_rhs' (#1492 -- 'applied_now' under the leap-frog discards
    # ~56% of every applied surface flux, retention (1-2g)/(2(1-g)) = 4/9 at
    # g=0.1; the recorded 10-yr A/B closes 98.7% of the ACC gap and takes
    # sigma>1.6 dense water from 0.000x to 1.031x NEMO).  A harness self-check
    # that fails BECAUSE the model was fixed is a broken tripwire.
    _placements = ("applied_now", "leapfrog_rhs")
    assert base.surface_tendency_placement in _placements, (
        f"unknown surface_tendency_placement "
        f"{base.surface_tendency_placement!r} on nemo_dino_kamm_mlf")
    _other = _placements[1 - _placements.index(base.surface_tendency_placement)]
    flipped = dataclasses.replace(base, surface_tendency_placement=_other)
    assert flipped.surface_tendency_placement == _other
    assert base.surface_tendency_placement != _other, (
        "override must not mutate the original cfg")
    print("OK: --surface-tendency-placement override changes "
          f"cfg.surface_tendency_placement "
          f"({base.surface_tendency_placement} -> {_other})")


def _provenance_gate() -> None:
    """Stamp source provenance and REFUSE to run from a dirty tracked tree.

    Added after the 2026-08-19/20 reconciliation (#1455, a009c6812): two runs
    of this harness at byte-identical committed source differed by 2.5 Sv in
    day-90 ACC, and the second review's log forensics left "an uncommitted
    working-tree edit" as the sole surviving candidate -- the difference is
    unrecoverable because nothing stamped the tree state. Every future run
    prints the HEAD sha and the tracked-file dirt count; a dirty tree aborts
    unless LEGOESM_ALLOW_DIRTY=1 is set explicitly (and then the dirt list is
    printed so the log carries what the tree carried).
    """
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE: HEAD={sha} dirty_tracked_files={len(dirt.splitlines())}")
    if dirt:
        print("PROVENANCE: dirty tracked files:")
        for line in dirt.splitlines():
            print(f"  {line}")
        if os.environ.get("LEGOESM_ALLOW_DIRTY") != "1":
            raise SystemExit(
                "REFUSING to run from a dirty tracked tree (see #1455 "
                "a009c6812: an uncommitted edit produced an unattributable "
                "2.5 Sv shift). Commit or stash, or set LEGOESM_ALLOW_DIRTY=1 "
                "to run anyway with the dirt list stamped in the log.")


def main(argv=None):
    args = _parse_args(argv)
    _provenance_gate()
    if args.recipe == "smoke-check":
        _smoke_check_vmix_scheme_override()
        return
    run_twin(args.recipe, args.out, n_days=args.days, save_3d=args.save_3d,
              run_traj=args.run_traj, run_stepdump=args.run_stepdump,
              bridge_tke=args.bridge_tke, bridge_before=args.bridge_before,
              vmix_scheme=args.vmix_scheme, use_gm_redi=args.use_gm_redi,
              surface_tendency_placement=args.surface_tendency_placement,
              perturb_seed=args.perturb_seed)


if __name__ == "__main__":
    main()
