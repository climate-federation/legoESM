"""Ocean-WB arm (c): train the lat-lon ocean's physics knobs against daily ORAS5.

The model is run_omip's OWN 1-degree lat-lon production setup (captured at the entry of
``_run_omip_loop``, as the phase-1 probe does) on the eORCA1 L75 vertical grid, so ORAS5 level k
is model level k.  Trained: ``LATLON_DYNAMICS_TRAINABLE`` (8 knobs, user-approved limits),
started from the run's resolved values.  Loss: :func:`legoesm.training.oras5_ocean.l1_loss`
against every daily target of an N-day rollout, averaged over the days.

Pieces reused from the WB trainer: the no-inert gate (``assert_no_inert_over`` over every sample
before the first update), ``create_optimizer``, ``ema_update``,
and ``training_history.jsonl`` for ``scripts/plot/plot_wb_training_history.py``.

Forcing: only ``--forcing none`` exists so far.  An unforced run is PLUMBING ONLY: its loss is
never comparable with a forced run, and every output is labelled so.

Detail choices (votes author/codex/GLM, 2026-11-11): level k -> k with a 1 m guard (3-0); L75
thicknesses generated from the eORCA1 mesh mask, not committed (2-1); loss over every daily target
(3-0); Adam lr 1e-2, warmup 10, clip 1.0 (3-0); EMA 0.99 (3-0); smoke 2 days x 4 dates x 2
epochs (3-0); gate over all samples (3-0).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MESH_MASK = Path("/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/data/grids/eORCA1.2_mesh_mask.nc")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", required=True, type=Path, help="ORAS5 L75 day store")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--forcing", required=True, choices=("none", "bulk", "prescribed"),
                   help="'none' = unforced, PLUMBING ONLY; 'bulk' = ERA5 state + coupler bulk "
                        "formulas; 'prescribed' = ERA5's own fluxes")
    p.add_argument("--era5-store", type=Path,
                   default=Path("/work/bb1596/b309178/diffESM/ocean_wb/era5_daily"))
    p.add_argument("--clim-dir", type=Path,
                   default=Path("/work/bb1596/b309178/diffESM/ocean_wb/climatology"))
    p.add_argument("--mesh-mask", type=Path, default=MESH_MASK)
    p.add_argument("--dz-file", type=Path,
                   default=Path("/work/bb1596/b309178/diffESM/ocean_wb/eorca1_L75_dz.txt"))
    p.add_argument("--days", type=int, default=2, help="rollout length [days]")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--first-year", type=int, default=1993)
    p.add_argument("--last-year", type=int, default=2017)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--dt", type=float, default=1800.0)
    return p.parse_args(argv)


def write_dz_file(mesh_mask, path):
    """eORCA1 L75 ``e3t_1d`` -> text thicknesses for run_omip's ``--dz-ref-file``."""
    if not path.exists():
        import xarray as xr
        e = np.asarray(xr.open_dataset(mesh_mask)["e3t_1d"].values, np.float64).ravel()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(f"# e3t_1d from {mesh_mask}\n" + "\n".join(f"{x:.9f}" for x in e) + "\n")
        tmp.replace(path)
    return np.loadtxt(path)


def capture_model(argv):
    """run_omip's production model + initial state at ``_run_omip_loop`` entry."""
    sys.path.insert(0, str(REPO / "scripts/run"))
    import run_omip as RO

    class _Captured(Exception):
        pass

    def _capture(model, state, grid_type, grid, z_coord, dt, *a, **k):
        raise _Captured((model, state, grid, z_coord, dt))

    args = RO.parse_args(argv)
    saved, RO._run_omip_loop = RO._run_omip_loop, _capture
    try:
        RO.run_omip_single("latlon", args)
    except _Captured as c:
        return c.args[0]
    finally:
        RO._run_omip_loop = saved
    raise SystemExit("run_omip_single returned without reaching _run_omip_loop")


def make_step(model, dt):
    """``model.step``'s restoring-mode path with a (possibly traced) config override."""
    pc = model.config

    def step(s, cfg, fw=None, sf=None):
        n = model._step_impl(s, dt, freshwater=fw, surface_forcing=sf, config=cfg)
        if pc.polar_filter.use_polar_filter:
            n = model._apply_polar_filter(n, dt)
        if pc.freeze_floor:
            n = model._apply_freeze_floor(n)
        if getattr(pc, "ew_cyclic_overlap", False):
            n = model._apply_ew_cyclic_overlap(n)
        return n
    return step


def gate_failure(loss, grads):
    """Non-finite loss or gradient on ONE sample -> message, else None (the gate's max over
    samples uses fmax, which would hide a NaN sample)."""
    bad = [n for n, g in grads.items() if not np.isfinite(float(g))]
    if np.isfinite(float(loss)) and not bad:
        return None
    return f"loss {float(loss)}, non-finite gradient for {bad}"


def d20_mae(model, oras5, keep, cosw):
    """Area-weighted mean |model - ORAS5| D20 per lead over the scored cells; arrays
    (sample, lead, lat, lon), ``cosw`` (lat, lon).  Unscored (incl. dry, NaN) cells drop out."""
    w = cosw[None, None] * keep
    err = np.where(keep, np.abs(model - oras5), 0.0)
    return (err * w).sum(axis=(0, 2, 3)) / w.sum(axis=(0, 2, 3))


def sample_dates(first_year, last_year, days, n, seed):
    lo = np.datetime64(f"{first_year}-01-01")
    hi = np.datetime64(f"{last_year}-12-31") - np.timedelta64(days, "D")
    span = int((hi - lo) / np.timedelta64(1, "D")) + 1
    pick = np.random.default_rng(seed).choice(span, size=n, replace=False)
    return [lo + np.timedelta64(int(i), "D") for i in sorted(pick)]


def noleap_day(date):
    """Noleap day-of-year index (Feb 29 pooled with Feb 28), as the climatologies."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts.data.build_ocean_wb_climatologies import noleap_doy
    return noleap_doy(date)


def forcing_arrays(a, date, days):
    """Per rollout day k (ERA5 of date+k): ERA5 channels, ice climatology, runoff climatology."""
    from legoesm.training.era5_ocean_forcing import load_era5_day
    ice = np.load(a.clim_dir / "siconc_oras5_1993_2017_daily.npy", mmap_mode="r")
    ro = np.load(a.clim_dir / "runoff_jra55do_daily.npy", mmap_mode="r")
    ds = [date + np.timedelta64(k, "D") for k in range(days)]
    doy = [noleap_day(d) for d in ds]
    return {"era5": jnp.asarray(np.stack([load_era5_day(a.era5_store, d) for d in ds]), jnp.float64),
            "ice": jnp.asarray(np.stack([ice[k] for k in doy]), jnp.float64),
            "runoff": jnp.asarray(np.stack([ro[k] for k in doy]), jnp.float64),
            "day": jnp.asarray([k + 0.5 for k in doy], jnp.float64)}


def build_sample(O, store, date, days, wet3, um, vm, base):
    ic, ic_val = O.day_to_fields(O.load_day(store, date), wet3, um, vm)
    tgts, ws = [], []
    for k in range(1, days + 1):
        t, tv = O.day_to_fields(O.load_day(store, date + np.timedelta64(k, "D")), wet3, um, vm)
        tgts.append({f: jnp.asarray(t[f]) for f in O.FIELDS})
        ws.append(O.weights(base, {f: tv[f] & ic_val[f] for f in O.FIELDS}))
    stack = lambda xs: {f: jnp.stack([x[f] for x in xs]) for f in O.FIELDS}
    return ({"ic": {f: jnp.asarray(ic[f]) for f in O.FIELDS}, "tgt": stack(tgts),
             "w": stack(ws)}, ic, O.weights(base, ic_val))


def main(argv=None):
    a = parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    from legoesm.ml.training import TrainingConfig, create_optimizer
    from legoesm.training import oras5_ocean as O
    from legoesm.training.ema import ema_update, init_ema
    from legoesm.training.inert_params import assert_no_inert_over
    from legoesm.training.trainable_ocean_params import (
        LATLON_DYNAMICS_TRAINABLE, TrainableOceanParams, names)
    import optax

    label = "PLUMBING ONLY (unforced)" if a.forcing == "none" else a.forcing
    a.out.mkdir(parents=True, exist_ok=True)
    dz = write_dz_file(a.mesh_mask, a.dz_file)
    argv_omip = ["--grid", "latlon", "--resolution", "180x360", "--nlev", str(dz.size),
                 "--dz-ref-file", str(a.dz_file), "--H-max", f"{dz.sum():.9f}",
                 "--dt", str(a.dt), "--days", "1",
                 "--bathymetry", str(REPO / "data/bathymetry/etopo_1deg.nc"),
                 "--implicit-vertical-mixing", "--north-cap-lat", "80", "--precision", "fp64",
                 "--output", str(a.out / "omip_setup")]
    model, state0, grid, z_coord, dt_run = capture_model(argv_omip)
    assert float(dt_run) == a.dt, dt_run
    O.check_levels(a.store, z_coord.z_full_ref)
    lat_c = np.rad2deg(np.asarray(grid.lat)).ravel()
    # the model's own wet geometry: partial bottom cells, nothing below its seabed
    zc = model.z_coord
    wet3 = (np.asarray(state0.land_mask.data)[..., None] > 0) & np.asarray(zc.is_active)
    um, vm = np.asarray(state0.u_mask.data), np.asarray(state0.v_mask.data)
    base = O.base_weights(lat_c, np.asarray(zc.h_partial) * wet3)
    dates = sample_dates(a.first_year, a.last_year, a.days, a.n_samples, a.seed)
    built = [build_sample(O, a.store, d, a.days, wet3, um, vm, base) for d in dates]
    std = O.field_std([b[1] for b in built], [b[2] for b in built])    # fixed, pooled IC days
    samples = [dict(b[0], std={f: jnp.asarray(std[f]) for f in O.FIELDS}) for b in built]
    print(f"[{label}] loss normaliser std {std}", flush=True)
    if a.forcing != "none":
        for x, d in zip(samples, dates):
            x["frc"] = forcing_arrays(a, d, a.days)
    model.prime_step_caches(O.set_state_fields(state0, samples[0]["ic"]))
    base_cfg = model.config
    step = make_step(model, a.dt)
    from legoesm.coupler.config import CouplerConfig
    from legoesm.training.era5_ocean_forcing import surface_forcing
    # OMIP's LY09 bulk config, with ERA5's reference heights: wind 10 m, T/q 2 m
    coupler_cfg = CouplerConfig(bulk_scheme="large_yeager", z_ref=10.0, z_t_atm=2.0,
                                z_q_atm=2.0, ocean_albedo=0.06)
    lat_rad = jnp.asarray(np.deg2rad(lat_c))[:, None]
    lon_rad = jnp.asarray(np.deg2rad(np.rad2deg(np.asarray(grid.lon)).ravel()))[None, :]
    ocean2 = jnp.asarray(wet3[..., 0])

    def day_forcing(c, f):
        """(freshwater, surface_forcing) for one model day, or (None, None) unforced."""
        if a.forcing == "none":
            return None, None
        return surface_forcing(a.forcing, f["era5"], c.T.data[..., 0], f["runoff"], f["ice"],
                               ocean2, lat_rad=lat_rad, lon_rad=lon_rad, day=f["day"],
                               ref_year=1993, coupler_cfg=coupler_cfg)   # zenith uses doy only
    spd = int(round(86400.0 / a.dt))

    def run_day(c, cfg, f):
        """One model day; forcing (if any) recomputed every step from the current SST."""
        @jax.checkpoint
        def stp(cc, _):
            fw, sf = day_forcing(cc, f)
            return step(cc, cfg, fw, sf), None
        return jax.lax.scan(stp, c, None, length=spd)[0]

    def day_inputs(x):
        return x["frc"] if a.forcing != "none" else jnp.zeros(a.days)

    def loss_fn(params, x):
        cfg = params.to_latlon_config(base_cfg)

        @jax.checkpoint
        def day(c, tw):
            c = run_day(c, cfg, tw[2])
            return c, O.l1_loss(O.state_fields(c), tw[0], tw[1], x["std"])

        s = O.set_state_fields(state0, x["ic"])
        return jnp.mean(jax.lax.scan(day, s, (x["tgt"], x["w"], day_inputs(x)))[1])

    params = TrainableOceanParams.from_values(
        {n: float(base_cfg.flat_get(n)) for n in names(LATLON_DYNAMICS_TRAINABLE)},
        LATLON_DYNAMICS_TRAINABLE)
    kn = names(LATLON_DYNAMICS_TRAINABLE)

    # gate: every trained knob carries a finite nonzero gradient on the samples (any sample)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    t0 = time.time()
    per = []
    for j, x in enumerate(samples):
        l0, g = vg(params, x)
        per.append({n: g.raw[i] for i, n in enumerate(kn)})
        msg = gate_failure(l0, per[-1])
        if msg:
            raise SystemExit(f"[{label}] GATE FAILED: sample {j} ({dates[j]}) {msg}")
    gate = {n: float(max(abs(float(p[n])) for p in per)) for n in kn}
    print(f"[{label}] gate |grad| max over {len(samples)} samples: {gate}", flush=True)
    assert_no_inert_over(per)
    print(f"[{label}] GATE PASSED ({time.time() - t0:.0f} s)", flush=True)

    steps = a.epochs * len(samples)
    opt = create_optimizer(TrainingConfig(lr=a.lr, warmup_steps=min(a.warmup, steps - 1),
                                          total_steps=steps, optimizer="adam",
                                          grad_clip_norm=1.0))
    opt_state = opt.init(params)
    ema = init_ema(params)
    hist_path = a.out / "training_history.jsonl"
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (a.out / "run_meta.json").write_text(json.dumps({
        "label": label, "forcing": a.forcing, "commit": commit, "dates": [str(d) for d in dates],
        "args": {k: str(v) for k, v in vars(a).items()}, "omip_argv": argv_omip,
        "gate_abs_grad_max": gate, "start_values": {n: float(v) for n, v in params.as_dict().items()}},
        indent=1))
    hist_path.write_text("")
    for ep in range(a.epochs):
        losses = []
        for i in np.random.default_rng(a.seed + 1 + ep).permutation(len(samples)):
            loss, g = vg(params, samples[i])         # one sample per update, one device
            upd, opt_state = opt.update(g, opt_state, params)
            params = optax.apply_updates(params, upd)
            ema = ema_update(ema, params, a.ema_decay)
            losses.append(float(loss))
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), "label": label,
               "params": {n: float(v) for n, v in params.as_dict().items()},
               "ema_params": {n: float(v) for n, v in ema.as_dict().items()}}
        with open(hist_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{label}] epoch {ep} mean loss {row['train_loss']:.6f}", flush=True)
    # D20 DIAGNOSTIC (not in the loss): ORAS5 vs model, same isotherm_depth on both, scored where
    # ORAS5 crosses 20 C; EMA parameters, every sample and lead day.
    from legoesm.ocean.diagnostics import isotherm_depth
    z_c = jnp.abs(jnp.asarray(z_coord.z_full_ref))
    wet_j = jnp.asarray(wet3)
    cosw = np.cos(np.deg2rad(lat_c))[:, None] * np.ones((1, wet3.shape[1]))

    @jax.jit
    def rollout_T(p, x):
        cfg = p.to_latlon_config(base_cfg)
        def day(c, f):
            c = run_day(c, cfg, f)
            return c, c.T.data
        return jax.lax.scan(day, O.set_state_fields(state0, x["ic"]), day_inputs(x))[1]

    d20_o, d20_m, keep = [], [], []
    for x in samples:
        Tm = rollout_T(ema, x)
        o = np.asarray(isotherm_depth(x["tgt"]["T"], z_c, wet_mask=wet_j[None]))
        m = np.asarray(isotherm_depth(Tm, z_c, wet_mask=wet_j[None]))
        Tt = np.asarray(x["tgt"]["T"])
        obs = np.all((np.asarray(x["w"]["T"]) > 0) | ~wet3[None], axis=-1)   # no filled level
        k = ((Tt[..., 0] > 20.0) & np.any((Tt <= 20.0) & wet3[None], axis=-1)
             & wet3[None, ..., 0] & obs)                    # ORAS5 crosses 20 C, observed column
        d20_o.append(o); d20_m.append(m); keep.append(k)
    d20_o, d20_m, keep = map(np.stack, (d20_o, d20_m, keep))       # (sample, lead, lat, lon)
    mae = d20_mae(d20_m, d20_o, keep, cosw)
    np.savez(a.out / "d20_diagnostic.npz", oras5=d20_o, model=d20_m, scored=keep,
             mae_by_lead_m=mae, lead_days=np.arange(1, a.days + 1), lat=lat_c,
             dates=np.array([str(d) for d in dates]), label=label, width_K=0.5)
    print(f"[{label}] D20 area-weighted MAE by lead day [m]: {mae.round(2).tolist()}", flush=True)
    print(f"[{label}] TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
