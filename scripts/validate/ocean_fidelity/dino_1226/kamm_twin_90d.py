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

DT = 2700.0
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
                       bridge_tke: bool = False, bridge_before: bool = False):
    """Bridge the NEMO restart into a legoESM state and run the day-0 gate.

    Returns (br, cfg, mc, model, forcing, sf, st) ready to integrate.
    """
    g = read_nemo_mesh_mask(f"{run_traj}/mesh_mask.nc", nn_hls=0)
    s = read_nemo_restart(f"{run_stepdump}/{RESTART_FILE}", nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    cfg = dataclasses.replace(dino_config_for_recipe(recipe),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)

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
        before = read_nemo_restart_before(f"{run_stepdump}/{RESTART_FILE}", nn_hls=0)
        st = bridge_before_state_topo(br._replace(state=st), g, before, periodic_i=True)
        _print_before_bridge_verify(st, before, g)

    # OPTIONAL: bridge NEMO's developed TKE closure memory (`en`) onto lego's
    # cold-start `state.tke` -- isolates whether the TKE cold-start (vs the
    # bridged prognostic T/S/eta/u/v) drives the day 0-4 surface-layer
    # handshake divergence (#1317). Off by default (matches the module
    # docstring's documented cold-start caveat) so `--bridge-tke` is additive,
    # not a silent behavior change.
    if bridge_tke:
        en_restart = read_nemo_restart_en(f"{run_stepdump}/{RESTART_FILE}", nn_hls=0)
        st = bridge_tke_from_restart(st, en_restart, br.land_mask)
        wet = np.asarray(br.land_mask) > 0.5
        d_en = float(np.max(np.abs(
            np.asarray(st.tke.data)[wet] - en_restart[..., 1:][wet])))
        print(f"TKE BRIDGE: seeded state.tke from NEMO restart en "
              f"(w-level 1..{en_restart.shape[-1]-1} -> interior interface "
              f"0..{en_restart.shape[-1]-2})  max|d_en|={d_en:.3e}", flush=True)

    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
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
             bridge_tke: bool = False, bridge_before: bool = False) -> bool:
    """Run the state-initialized twin for ``n_days`` and save an npz. Returns stable."""
    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        recipe, run_traj, run_stepdump, bridge_tke=bridge_tke,
        bridge_before=bridge_before)
    nsteps = STEPS_PER_DAY * n_days

    dyn = jax.jit(lambda st: model.step(st, DT, surface_forcing=sf))

    land_mask = np.asarray(st.land_mask.data)
    n_lat, n_lon = br.geometry.n_lat, br.geometry.n_lon

    eta_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    sst_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    u_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    v_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)

    snap_days = tuple(d for d in SNAP_DAYS if d <= n_days) if save_3d else ()
    t3d, s3d, eta3d = {}, {}, {}
    if save_3d:
        t3d[0] = np.asarray(st.T.data, dtype=np.float32)
        s3d[0] = np.asarray(st.S.data, dtype=np.float32)
        eta3d[0] = np.asarray(st.eta.data, dtype=np.float32)
        print("captured day-0 3-D T/S snapshot", flush=True)

    blew_up_at = None
    t0 = time.time()
    for k in range(nsteps):
        st = apply_dino_lat_lon_surface_forcing(st, forcing, br.z_coord, cfg, DT,
                                                 t_seconds=(k + 1) * DT)
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
                print(f"  captured day {day_num} full 3-D T/S snapshot", flush=True)

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
                    help="also save full 3-D T/S at days 0/30/60/90")
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
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_twin(args.recipe, args.out, n_days=args.days, save_3d=args.save_3d,
              run_traj=args.run_traj, run_stepdump=args.run_stepdump,
              bridge_tke=args.bridge_tke, bridge_before=args.bridge_before)


if __name__ == "__main__":
    main()
