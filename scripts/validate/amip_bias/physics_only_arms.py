#!/usr/bin/env python
"""Physics-only stepping of a production MPAS AMIP state, one process added per arm.

The run is relaunched from a checkpoint with its own arguments
(``mpas_onestep_param_grad.launch_argv``) and the production physics build
(``combined.make_physics`` config, args and kwargs, full- and held-radiation
variants) is captured together with the model state, forcing and physics
carry at the first ``model.step`` -- the same capture as ``heating_budget``.
Each arm rebuilds the physics with some modules set to ``scheme="none"`` and
steps it WITHOUT dynamics on the whole mesh for ``--hours``, applying the
physics exactly as ``MPASPrimitiveEquationModel.step`` does
(``apply_physics_water_mass`` then ``state += dt * tendency``, ``dt`` the
dycore step; the physics-cadence write step already returns the n-step
increment), alternating the full- and held-radiation builds on the run's
``rad_update_steps / physics_update_steps`` cadence. The ``+fixers`` arm adds
the step's positivity stage (``apply_water_positivity``, run's settings).
The forcing's solar clock (``seconds_of_day``, ``day_of_year``) advances with
the calls; every other forcing field is held at the start.
Not covered offline: horizontal and vertical transport, the dry-mass fixer
(global), the vertical T filter, the land model's evolution beyond what the
physics carry holds (land and lagged surface-flux forcing fields are held).

Reported per arm and box (ocean, area-weighted): 950-850 hPa cloud fraction
(CLUBB carry), grid-mean q_c, the inversion pressure, and the box-mean q_c
change per physics call (net of everything in that arm).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpas_onestep_param_grad as H  # noqa: E402
from clubb_column_ab import BOXES, inversion_pressure  # noqa: E402

ARMS = {                      # modules switched OFF in each arm (one change per step)
    "clubb": ("microphysics", "convection", "radiation", "gravity_wave_drag"),
    "+micro": ("convection", "radiation", "gravity_wave_drag"),
    "+conv": ("radiation", "gravity_wave_drag"),
    "+rad": ("gravity_wave_drag",),
    "+fixers": ("gravity_wave_drag",),
}


def arm_off(arm, gwd_on):
    """Modules switched off in *arm*; gravity-wave drag follows ``gwd_on``."""
    off = tuple(m for m in ARMS[arm] if m != "gravity_wave_drag")
    return off if gwd_on else off + ("gravity_wave_drag",)


def arm_config(cfg, off):
    return cfg._replace(**{m: getattr(cfg, m)._replace(scheme="none") for m in off})


def capture(run, day, scratch, run_amip_dir):
    import legoesm.atmosphere.physics.combined as combined
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationModel
    builds, cap = [], {}
    real = combined.make_physics

    def spy_make(config, *a, **kw):
        builds.append((config, a, dict(kw)))
        return real(config, *a, **kw)

    def spy_step(self, state, dt, physics_fn=None, forcing=None, phys_state=None):
        cap.update(model=self, state=state, dt=dt, forcing=forcing, phys_state=phys_state)
        raise H._Captured()
    combined.make_physics = spy_make
    MPASPrimitiveEquationModel.step = spy_step
    sys.path.insert(0, str(run_amip_dir))
    import run_amip
    if Path(run_amip.__file__).resolve().parent != Path(run_amip_dir).resolve():
        raise SystemExit(f"imported {run_amip.__file__}, expected one in {run_amip_dir}")
    try:
        run_amip.main(H.launch_argv(run, day, scratch))
    except H._Captured:
        pass
    else:
        raise SystemExit("driver returned without calling model.step")
    combined.make_physics = real
    full = [b for b in builds if b[2].get("need_rad", True) is not False]
    held = [b for b in builds if b[2].get("need_rad", True) is False]
    if not full or not held:
        raise SystemExit(f"captured {len(full)} full / {len(held)} held physics builds")
    return cap, full[-1], held[-1], real


def box_stats(T, qc, cf, p_full, p_half, area, sel):
    win = (p_full >= 85000.0) & (p_full <= 95000.0)
    w = np.where(win, np.diff(p_half, axis=1), 0.0)
    lay = lambda x: (x * w).sum(1) / w.sum(1)      # noqa: E731
    a = area[sel]
    m = lambda x: float((x[sel] * a).sum() / a.sum())   # noqa: E731
    return {"cf_950_850": m(lay(cf)), "qc_950_850_gkg": m(lay(qc)) * 1e3,
            "inversion_hPa": m(inversion_pressure(T, p_full)) / 100}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory relative to LEGOESM_AMIP_RUNS")
    ap.add_argument("day", type=int)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--mesh-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inspect", action="store_true", help="print the capture and stop")
    ap.add_argument("--gwd", choices=("off", "on"), required=True,
                    help="gravity-wave drag in every arm (production settings when on)")
    ap.add_argument("--run-amip-dir", required=True,
                    help="scripts/run of the checkout whose code wrote the checkpoint")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap, (cfg, pa, pkw), (_, ha, hkw), make = capture(a.run, a.day, out / "_launch", a.run_amip_dir)
    import jax
    import jax.numpy as jnp
    from legoesm.core.conservation import apply_physics_water_mass, apply_water_positivity
    from legoesm.atmosphere.physics.combined import physics_cache_seed
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASHydrostaticState
    model, state0, DT = cap["model"], cap["state"], float(cap["dt"])
    mesh, sig, forcing = model.mesh, model.sigma_coord, cap["forcing"]
    rc = json.load(open(H.ROOT / a.run / "run_manifest.json"))["config"]["resolved_config"]
    nphys = int(rc["physics_update_steps"])
    rad_every = int(rc["rad_update_steps"]) // nphys
    ncall = int(round(a.hours * 3600.0 / (DT * nphys)))
    print(f"captured ({time.time() - t0:.0f}s): dt={DT} physics every {nphys} steps, "
          f"radiation every {rad_every} physics calls, {ncall} calls; full kw={pkw}; "
          f"held kw={hkw}; forcing keys={sorted((forcing or {}).keys())}", flush=True)
    if a.inspect:
        return 0
    m = np.load(a.mesh_npz)
    lat, lon = np.asarray(m["lat"]), ((np.asarray(m["lon"]) + 180) % 360) - 180
    area, land = np.asarray(m["area"], float), np.asarray(m["land"]) != 0
    sels = {b: (lat >= x[0]) & (lat <= x[1]) & (lon >= x[2]) & (lon <= x[3]) & ~land
            for b, x in BOXES.items()}
    c = model.config
    res = {}
    s0, d0 = float(forcing["seconds_of_day"]), float(forcing["day_of_year"])

    def forcing_at(n):
        """The run's forcing with the clock advanced n physics calls (the driver
        advances the solar time; everything else is held at the start)."""
        t = s0 + n * DT * nphys
        f = dict(forcing)
        f["seconds_of_day"] = jnp.asarray(t % 86400.0, dtype=jnp.asarray(forcing["seconds_of_day"]).dtype)
        f["day_of_year"] = jnp.asarray(d0 + t // 86400.0, dtype=jnp.asarray(forcing["day_of_year"]).dtype)
        return f

    for arm in a.arms:
        acfg = arm_config(cfg, arm_off(arm, a.gwd == "on"))
        f_full = make(acfg, *pa, **pkw)
        f_held = make(acfg, *ha, **hkw)
        fixers = arm == "+fixers"

        def apply(fn, st, ps, fc):
            r = fn(st, mesh, sig, phys_state=ps, forcing=fc)
            t, ps_new = (r[0], r[1]) if type(r) is tuple else (r, ps)
            p_s = st.p_s.data + DT * t.dp_s_dt.data
            tr = st.tracers
            if tr is not None and t.tracer_tendencies is not None:
                tr, p_s = apply_physics_water_mass(tr, t.tracer_tendencies, p_s, sig, DT)
            st = MPASHydrostaticState(u=st.u.replace(data=st.u.data + DT * t.du_dt.data),
                                      T=st.T.replace(data=st.T.data + DT * t.dT_dt.data),
                                      p_s=st.p_s.replace(data=p_s), phis=st.phis, tracers=tr)
            if fixers:
                ph = sig.pressure_at_half(st.p_s.data)
                dp = jnp.maximum(ph[..., 1:] - ph[..., :-1], 0.0)
                tr, T = apply_water_positivity(
                    st.tracers, st.T.data, dp, conservative=c.conservative_tracer_clamp,
                    energy_consistent=c.energy_consistent_moisture_clip, area=mesh.areaCell)
                st = st._replace(tracers=tr, T=st.T.replace(data=T))
            return st, ps_new
        j_full, j_held = jax.jit(lambda s, p, f: apply(f_full, s, p, f)), \
            jax.jit(lambda s, p, f: apply(f_held, s, p, f))
        # the cadence cache is shaped by the physics that fills it; reseed it from
        # this arm's full variant (combined.physics_cache_seed, as the driver does)
        ps0 = cap["phys_state"]
        st, ps = state0, ps0._replace(held_physics=physics_cache_seed(
            f_full, state0, mesh, sig, ps0, forcing_at(0)))
        rows, hist = {b: [] for b in BOXES}, []
        t1 = time.time()
        for n in range(ncall):
            qc_prev = np.asarray(st.tracers["q_c"].data, float)
            st, ps = (j_full if n % rad_every == 0 else j_held)(st, ps, forcing_at(n))
            T = np.asarray(st.T.data, float)
            qc = np.asarray(st.tracers["q_c"].data, float)
            if not (np.all(np.isfinite(T)) and np.all(np.isfinite(qc))):
                res[arm] = {"first_nonfinite_call": n}
                break
            ph = np.asarray(sig.pressure_at_half(st.p_s.data), float)
            pf = 0.5 * (ph[:, 1:] + ph[:, :-1])
            cf = np.asarray(ps.cloud_fraction, float)
            for b, sel in sels.items():
                s = box_stats(T, qc, cf, pf, ph, area, sel)
                s["dqc_per_call_gkg"] = box_stats(T, qc - qc_prev, cf, pf, ph, area,
                                                  sel)["qc_950_850_gkg"]
                rows[b].append(s)
            if n == 0:
                print(f"{arm}: first call done ({time.time() - t1:.0f}s incl. compile)", flush=True)
        else:
            res[arm] = {}
            for b in BOXES:
                half = rows[b][ncall // 2:]
                res[arm][b] = {k: float(np.mean([r[k] for r in half])) for k in half[0]}
                res[arm][b]["end"] = rows[b][-1]
        print(arm, json.dumps(res[arm]), f"({time.time() - t1:.0f}s)", flush=True)
    ph0 = np.asarray(sig.pressure_at_half(state0.p_s.data), float)
    init = {b: box_stats(np.asarray(state0.T.data, float),
                         np.asarray(state0.tracers["q_c"].data, float),
                         np.asarray(cap["phys_state"].cloud_fraction, float),
                         0.5 * (ph0[:, 1:] + ph0[:, :-1]), ph0, area, sel)
            for b, sel in sels.items()}
    sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse",
                          "HEAD"], capture_output=True, text=True).stdout.strip()
    import legoesm
    (out / f"physics_only_arms_gwd{a.gwd}.json").write_text(json.dumps(
        {"meta": {"probe_git": sha, "model_package": legoesm.__file__, "gwd": a.gwd, "run": a.run, "day": a.day, "hours": a.hours, "dt": DT,
                  "physics_update_steps": nphys, "rad_every_calls": rad_every},
         "init": init, "arms": res}, indent=1))
    print("init", json.dumps(init))
    return 0


if __name__ == "__main__":
    sys.exit(main())
