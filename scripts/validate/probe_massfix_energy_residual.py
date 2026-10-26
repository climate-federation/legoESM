"""Energy the dry-mass fixer changes OUTSIDE the dycore energy-fixer window
(review 2026-10-14 F15 follow-up V8), measured on a production checkpoint
with the full production step (dynamics + energy fixer + physics + rails).

Restores a checkpoint through the production driver and runs N steps of the
real loop.  Operators are instrumented (jax.debug.callback, numbers only),
one row per call, tagged by call site:
  efix    : energy the dycore energy fixer returns   = c_pd * dT * M   [J]
  efix_mf : same, for the restore after the dry-mass fixer (F15 V8; absent
            before that fix or with energy_fixer off)
  massfix : energy change across the dry-mass fixer [J]
with E = sum_area((c_pd T + KE) dp/g + phis p_s/g), the fixer's own energy
(``dycore_column_energy``; TRiSK KE on MPAS, cell-centre KE on lat-lon).
MPAS: massfix = E(mass fixer out) - E(in), i.e. BEFORE the restore (raw).
Lat-lon: the mass fixer (and the restore) sit inside the safety rails, so
massfix = E(rails, fix_mass on) - E(rails, fix_mass off) on the same input,
i.e. AFTER the restore (net).  ``summarize`` reports both raw and net on both
lanes (net = raw + efix_mf).  The lat-lon isolation needs the T-neutral
positivity borrow (conservative_tracer_clamp=True); the probe refuses
otherwise.  Rows are aggregated per tag (arrival order is not step order:
callbacks are unordered), and every tag must have one row per step.
Single process only (plain area sums, no owned masks / MPI reductions).
Rates are W/m^2 = dE / (A * dt), + = energy added.

  --lane mpas   : ``--argv-file`` = run_amip CLI (deck + data paths) of the run
  --lane latlon : ``--config-json`` = the run's experiment_config.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

ROOT = Path(os.environ.get("WT", ".")).resolve()
ROWS: list[tuple[str, float, float]] = []
_CTX: dict = {"mf_out": None, "in_rails": False}


def _rec(tag):
    def f(de, dps):
        ROWS.append((tag, float(de), float(dps)))
    return f


def _fp64(T):
    """efix J = c_pd*dT*M is the heating APPLIED only if T += dT is not
    rounded away: refuse fp32 T storage (trace-time dtype check)."""
    if T.dtype != np.float64:
        raise SystemExit(f"probe needs fp64 T storage, got {T.dtype}")


def _driver_mpas(a, days):
    spec = importlib.util.spec_from_file_location(
        "run_amip_probe", ROOT / "scripts" / "run" / "run_amip.py")
    ra = importlib.util.module_from_spec(spec)
    sys.modules["run_amip_probe"] = ra
    spec.loader.exec_module(ra)
    argv = open(a.argv_file).read().split() + ["--output", a.out]
    parser = ra.build_arg_parser()
    from legoesm.driver.run_config_yaml import load_yaml_config
    deck = argv[argv.index("--config") + 1]
    deck = str(ROOT / deck) if not os.path.isabs(deck) else deck
    argv[argv.index("--config") + 1] = deck
    keys = load_yaml_config(deck, parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    args = parser.parse_args(argv)
    # the checkpoint carries the land state (the runs had land_ic_path="")
    args.land_ic = ""
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    cfg = ra.build_config_from_args(args)
    return cfg._replace(days=days, output=cfg.output._replace(output_dir=a.out))


def _driver_json(a, days):
    from legoesm.driver.config import load_experiment_config
    cfg = load_experiment_config(a.config_json)
    return cfg._replace(days=days, output=cfg.output._replace(output_dir=a.out))


def _instrument_mpas():
    import jax
    import jax.numpy as jnp

    import legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas as pm
    from legoesm import constants
    from legoesm.core.conservation import dycore_column_energy
    from legoesm.core.operators_voronoi import kinetic_energy_cell_3d

    def energy(s, mesh, sig):
        ph = sig.pressure_at_half(s.p_s.data)
        e = dycore_column_energy(s.T.data, kinetic_energy_cell_3d(s.u.data, mesh),
                                 ph[..., 1:] - ph[..., :-1], s.phis.data, s.p_s.data)
        return jnp.sum(e * mesh.areaCell)

    orig_mf, orig_ef = pm._fix_mass_mpas_hydro, pm.voronoi_energy_fixer_dT

    def mf(state_new, state_old, mesh, total_area=None, target_mass=None,
           sigma_coord=None):
        out = orig_mf(state_new, state_old, mesh, total_area=total_area,
                      target_mass=target_mass, sigma_coord=sigma_coord)
        a = mesh.areaCell
        dps = jnp.sum((out.p_s.data - state_new.p_s.data) * a) / jnp.sum(a)
        _CTX["mf_out"] = out   # the restore's ``post`` is this object
        jax.debug.callback(_rec("massfix"),
                           energy(out, mesh, sigma_coord)
                           - energy(state_new, mesh, sigma_coord), dps)
        return out

    def ef(pre, post, mesh, sigma_coord, area, sum_fn=None):
        dT = orig_ef(pre, post, mesh, sigma_coord, area, sum_fn)
        ph = sigma_coord.pressure_at_half(post.p_s.data)
        m = jnp.sum(jnp.sum(ph[..., 1:] - ph[..., :-1], axis=-1) * area) / constants.g
        _fp64(post.T.data)
        tag = "efix_mf" if post is _CTX["mf_out"] else "efix"
        if tag == "efix_mf":
            _CTX["mf_out"] = None
        jax.debug.callback(_rec(tag), constants.c_pd * dT * m, dT)
        return dT

    pm._fix_mass_mpas_hydro, pm.voronoi_energy_fixer_dT = mf, ef


def _instrument_latlon():
    import jax
    import jax.numpy as jnp

    import legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid as pl
    from legoesm import constants
    from legoesm.core.conservation import dycore_column_energy

    def energy(s, grid, sig):
        ph = sig.pressure_at_half(s.p_s)
        uc, vc = pl._face_to_cell_u(s.u), pl._face_to_cell_v(s.v)
        e = dycore_column_energy(s.T, 0.5 * (uc * uc + vc * vc),
                                 ph[..., 1:] - ph[..., :-1], s.phis, s.p_s)
        return jnp.sum(e * grid.area)

    M = pl.CGridLatLonPrimitiveEquationModel
    orig_rails, orig_ef = M._apply_safety_rails, pl.gridded_energy_fixer_dT

    def rails(self, state, target_mass=None, pre_state=None, *, grid=None,
              sigma_coord=None):
        g = self.grid if grid is None else grid
        sc = self.sigma_coord if sigma_coord is None else sigma_coord
        if not self.config.conservative_tracer_clamp:
            raise SystemExit("on-minus-off isolation needs the T-neutral "
                             "positivity borrow (conservative_tracer_clamp)")
        _CTX["in_rails"] = True   # trace time: the restore is called in here
        try:
            on = orig_rails(self, state, target_mass, pre_state, grid=grid,
                            sigma_coord=sigma_coord)
        finally:
            _CTX["in_rails"] = False
        cfg = self.config
        self.config = cfg._replace(fix_mass=False)
        try:
            off = orig_rails(self, state, target_mass, pre_state, grid=grid,
                             sigma_coord=sigma_coord)
        finally:
            self.config = cfg
        dps = jnp.sum((on.p_s - off.p_s) * g.area) / jnp.sum(g.area)
        jax.debug.callback(_rec("massfix"), energy(on, g, sc) - energy(off, g, sc),
                           dps)
        return on

    def ef(pre, post, grid, sigma_coord, **kw):   # owned_mask on newer trees
        dT = orig_ef(pre, post, grid, sigma_coord, **kw)
        ph = sigma_coord.pressure_at_half(post[3])
        m = jnp.sum(jnp.sum(ph[..., 1:] - ph[..., :-1], axis=-1) * grid.area) / constants.g
        _fp64(post[0])
        tag = "efix_mf" if _CTX["in_rails"] else "efix"
        jax.debug.callback(_rec(tag), constants.c_pd * dT * m, dT)
        return dT

    M._apply_safety_rails, pl.gridded_energy_fixer_dT = rails, ef


def _convert_p3_latlon(src, out_dir):
    """Re-key a fu/composable-p3 lat-lon column checkpoint (``llc_u_face`` /
    ``llc_v_face`` + flattened column fields) into main's ``latloncol_ckpt_v1``
    schema.  Same data: C-grid face winds as stored, cell fields reshaped
    row-major (n_lat, n_lon) -- the ColumnMesh order."""
    d = dict(np.load(src, allow_pickle=False))
    nlat, nlon1, nlev = d["llc_u_face"].shape
    nlon = nlon1 - 1
    to2 = lambda x: x.reshape((nlat, nlon) + x.shape[1:])  # noqa: E731
    d["latloncol_schema"] = np.asarray("latloncol_ckpt_v1")
    d["latloncol_u"], d["latloncol_v"] = d.pop("llc_u_face"), d.pop("llc_v_face")
    for k in ("T", "p_s", "phis"):
        d[f"latloncol_{k}"] = to2(d[k])
    names = sorted(str(n) for n in d["tracer_names"])
    d["latloncol_tracer_names"] = np.asarray(names)
    for i, nm in enumerate(names):
        d[f"latloncol_q_{i}"] = to2(d[f"trc_{nm}"])
    out = os.path.join(out_dir, "converted_" + os.path.basename(src))
    np.savez(out, **d)
    print(f"converted {src} -> {out} (nlat={nlat} nlon={nlon} nlev={nlev})",
          flush=True)
    return out


def summarize(rows, A, dt, lane, steps):
    """Per-tag rates [W/m^2] (+ = energy added) and the mass fixer's RAW
    (before the restore) and NET (after it) energy change, net = raw +
    efix_mf.  Tags are never paired by arrival index.  Every recorded tag
    must have exactly ``steps`` rows (efix and massfix are required)."""
    by: dict = {}
    for tag, de, x in rows:
        by.setdefault(tag, []).append((de, x))
    for tag in ("efix", "massfix", *by):
        n = len(by.get(tag, ()))
        if n != steps:
            raise RuntimeError(f"{tag}: {n} rows for {steps} steps")
    out: dict = {}
    for tag, v in sorted(by.items()):
        r = np.array([de for de, _ in v]) / (A * dt)
        out[f"{tag}_mean_Wm2"] = float(r.mean())
        out[f"{tag}_min_Wm2"], out[f"{tag}_max_Wm2"] = float(r.min()), float(r.max())
    restore = out.get("efix_mf_mean_Wm2", 0.0)
    if lane == "mpas":
        raw = out["massfix_mean_Wm2"]
        net = raw + restore
    else:
        net = out["massfix_mean_Wm2"]
        raw = net - restore
    out.update(massfix_raw_Wm2=raw, massfix_net_Wm2=net,
               dps_absmean_Pa=float(np.mean([abs(x) for _, x in by["massfix"]])),
               ratio_raw_over_efix=raw / out["efix_mean_Wm2"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=("mpas", "latlon"), required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--argv-file", default="")
    ap.add_argument("--config-json", default="")
    ap.add_argument("--convert-p3", action="store_true")
    a = ap.parse_args()
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = len(subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True).stdout.splitlines())
    print(f"WT={ROOT} sha={sha} dirty={dirty} lane={a.lane} restart={a.restart} "
          f"steps={a.steps}", flush=True)
    {"mpas": _instrument_mpas, "latlon": _instrument_latlon}[a.lane]()
    if a.lane == "mpas":
        cfg = _driver_mpas(a, 1.0)
    else:
        cfg = _driver_json(a, 1.0)
    dt = float(cfg.dycore.dt)
    cfg = cfg._replace(days=(a.steps + 0.5) * dt / 86400.0)
    print(f"dt={dt} fix_mass={cfg.dycore.fix_mass} "
          f"energy_fixer={cfg.dycore.energy_fixer} grid={cfg.grid.grid_type} "
          f"res={cfg.grid.resolution} column_lane={cfg.dycore.column_lane}",
          flush=True)
    os.makedirs(a.out, exist_ok=True)
    if a.convert_p3:
        a.restart = _convert_p3_latlon(a.restart, a.out)
    from legoesm.driver.model_driver import ModelDriver
    drv = ModelDriver(cfg)
    drv.setup()
    if a.convert_p3:
        # The P3 run's terrain product predates main's; the restore refuses a
        # different terrain.  Adopt the CHECKPOINT's phis (the dynamics state
        # is balanced with it) in the fresh state the restore compares to.
        import jax.numpy as jnp
        ck = np.load(a.restart)["latloncol_phis"]
        nat = drv.state.native
        dphis = float(np.max(np.abs(np.asarray(nat.phis) - ck)))
        print(f"adopting checkpoint terrain: max|dphis|={dphis:.3e} m2/s2",
              flush=True)
        # External check of the re-keying order: a wrong (n_lat, n_lon)
        # reshape scrambles phis by O(1e4) m2/s2 against main's own terrain.
        if dphis > 1e-6:
            raise SystemExit("converted terrain differs from the grid's: "
                             "check the re-keying order")
        drv.state = drv.model.from_bundle(
            nat._replace(phis=jnp.asarray(ck, dtype=nat.phis.dtype)))
        if getattr(drv, "_phis_data", None) is not None:
            drv._phis_data = jnp.reshape(jnp.asarray(ck), drv._phis_data.shape)
    step0, day0 = drv.load_checkpoint(a.restart)
    print(f"restored step={step0} day={day0} model={type(drv.model).__name__}",
          flush=True)
    area = np.asarray(drv.grid.areaCell if hasattr(drv.grid, "areaCell")
                      else drv.grid.area)
    A = float(area.sum())
    drv.run(start_step=step0, start_day=day0)
    import jax
    jax.effects_barrier()   # every debug callback has landed in ROWS
    print(f"A={A:.6e} rows=" + str({t: sum(r[0] == t for r in ROWS)
                                    for t in sorted({r[0] for r in ROWS})}),
          flush=True)
    for tag, de, x in ROWS:   # arrival order, NOT step order
        unit = "Pa" if tag == "massfix" else "K"
        print(f"{tag:8s} {de / (A * dt):+.6e} W/m2 ({x:+.3e} {unit})", flush=True)
    print(json.dumps({"lane": a.lane, "restart": a.restart, "sha": sha,
                      "dirty": dirty, "dt": dt, "steps": a.steps,
                      **summarize(ROWS, A, dt, a.lane, a.steps)}), flush=True)


if __name__ == "__main__":
    main()
