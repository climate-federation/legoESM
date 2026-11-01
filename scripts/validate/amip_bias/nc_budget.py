"""Cloud-droplet NUMBER budget by process on a production MPAS AMIP state (#2216).

Question: the CAM6 MG2 prognostic droplet number is activated at ~50 cm^-3/day
but holds only ~12 cm^-3 grid-mean.  Which process removes it?

Driven by ``heating_budget.py --nc-budget`` (same relaunch-from-checkpoint
capture).  One un-jitted model step of the run's OWN code on its OWN state,
plus the same step with no physics (what the held steps of the physics cadence
run).  Rows are rates [per second] averaged over one physics window of
``n_steps`` dycore steps (the physics step applies n_steps x the physics rate,
the held steps zero), taking THIS step's dynamics and fixers as the window's
(an assumption: one state):

* dynamics row      = (N_c after the dycore's RK dynamics - N_c at step start)/dt,
                      read from the state the step hands to its physics;
* physics rows      = each physics module's N_c rate in that step (the macmic
                      sub-cycled modules averaged over their calls, as the
                      combined physics weights them);
* microphysics rows = the MG2 droplet-number terms the scheme itself publishes
                      (``MicrophysicsOutput.nc_budget``, applied, post-limiter),
                      converted to per-kg with each call's own density;
* fixers after dynamics = (N_c at the end of the no-physics step - N_c after
                      dynamics)/dt: positivity borrow and dry-mass fix;
* fixers after physics, extra = the physics step's fixer change beyond that,
                      / n_steps (water-mass application, borrow of the physics
                      increment's undershoots).

Gates (abort on failure): microphysics terms sum to the microphysics module's
tendency; module tendencies sum to the combined physics tendency.  The rows sum
to the step's total change by construction, so that closure proves nothing and
is not claimed.  Instantaneous, one step (00 UTC of the checkpoint day): it
ranks processes, it does NOT close a multi-day inventory.

Aggregation is mass-weighted (cell area x layer dp / g) and expressed per unit
VOLUME of air in the region (cm^-3, cm^-3/day), so rows and inventory share one
normalisation and the sink time inventory / sink is independent of it.
Regions: global and Southern Ocean 40-65S (all cells); layers: whole column and
950-700 hPa by layer-midpoint pressure.

In-cloud number: from MG2's first sub-step inputs, cells with q_c >= 1e-6 kg/kg:
grid-mean = sum(N V) / sum(V), in-cloud = sum(N V) / sum(cf V) (MG2's own cloud
fraction argument).  NUMBERS ONLY, no verdict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REGIONS = {"global": (-90.0, 90.0), "SO 40-65S": (-65.0, -40.0)}
LAYERS = {"column": None, "950-700 hPa": (700.0, 950.0)}
QC_CLOUDY = 1.0e-6          # kg/kg, cloudy-cell threshold of the in-cloud row
TERRAIN_DRIFT_MAX = 1.0e-5  # relative; reduction-order drift accepted on relaunch
MICRO_TERMS = ("activation", "autoconversion", "accretion", "riming_snow",
               "riming_graupel", "sedimentation", "ice_melt_homog_freeze",
               "size_readjust", "no_cloud_clear")
OKABE_ITO = ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
             "#CC79A7", "#000000", "#999999")
PER_M3_S_TO_CM3_DAY = 1.0e-6 * 86400.0


def gate(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        raise SystemExit(f"GATE FAILED: {msg}")


def closes(parts, total, rtol=1e-9) -> tuple[bool, float]:
    """``sum(parts) == total`` cell by cell, relative to the cell's gross."""
    parts = [np.asarray(p, dtype=np.float64) for p in parts]
    gross = sum(np.abs(p) for p in parts) + np.abs(total)
    resid = np.abs(sum(parts) - total)
    worst = float((resid / np.maximum(gross, 1e-300)).max())   # per cell, no global floor
    return worst <= rtol, worst


def mean_calls(arrays):
    """Mean over a module's calls: the macmic loop weights each of its N
    sub-step tendencies by 1/N (a once-called module has N = 1)."""
    return sum(np.asarray(a, dtype=np.float64) for a in arrays) / len(arrays)


def masks(lat_deg, p_mid_hpa):
    """{(region, layer): bool (ncol, nlev)} cell masks."""
    out = {}
    for rn, (lo, hi) in REGIONS.items():
        col = (lat_deg >= lo) & (lat_deg <= hi)
        for ln, band in LAYERS.items():
            lev = np.ones_like(p_mid_hpa, bool) if band is None else (
                (p_mid_hpa >= band[0]) & (p_mid_hpa <= band[1]))
            out[(rn, ln)] = col[:, None] & lev
    return out


def aggregate(rows: dict, n0, mass, rho, mk):
    """Per-volume region means.  ``rows`` [1/kg/s], ``n0`` [1/kg], ``mass``
    [kg] and ``rho`` [kg/m3] per cell.  Returns inventory [cm^-3], rows
    [cm^-3/day] and the sink time [h] of every negative row and of all of them."""
    vol = (mass / rho)[mk].sum()
    inv = (n0 * mass)[mk].sum() / vol * 1e-6
    r = {k: (v * mass)[mk].sum() / vol * PER_M3_S_TO_CM3_DAY for k, v in rows.items()}
    sink = -sum(v for v in r.values() if v < 0)
    tau = {k: (inv / -v * 24.0 if v < 0 else np.inf) for k, v in r.items()}
    return inv, r, (inv / sink * 24.0 if sink > 0 else np.inf), tau


def incloud(n_vol, cf, q_c, vol, mk):
    """Grid-mean and in-cloud droplet number [cm^-3] over cloudy cells."""
    c = mk & (q_c >= QC_CLOUDY)
    for name, a in (("N_c", n_vol), ("cloud fraction", cf), ("q_c", q_c)):
        if not np.isfinite(a[mk]).all():
            raise SystemExit(f"FATAL: {np.count_nonzero(~np.isfinite(a[mk]))} non-finite {name}")
    v = vol[c]
    return {"n_cells": int(c.sum()),
            "grid_mean": float((n_vol[c] * v).sum() / v.sum() * 1e-6),
            "in_cloud": float((n_vol[c] * v).sum() / (cf[c] * v).sum() * 1e-6),
            "cloud_fraction": float((cf[c] * v).sum() / v.sum()),
            "n_cf_zero": int((c & (cf <= 0)).sum()),
            "median_in_cloud": float(np.median(n_vol[c & (cf > 0)] / cf[c & (cf > 0)]) * 1e-6)}


def figure(results: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(next(iter(results.values()))[1])
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(13, 0.42 * len(keys) + 2), sharey=True)
    y = np.arange(len(keys))
    for ax, rn in zip(np.atleast_1d(axes), REGIONS):
        for j, ln in enumerate(LAYERS):
            inv, r, tau_all, _ = results[(rn, ln)]
            ax.barh(y + (j - 0.5) * 0.4, [r[k] for k in keys], height=0.4,
                    color=OKABE_ITO[4 if j == 0 else 0],
                    label=f"{ln}: N = {inv:.1f} cm-3, sink time {tau_all:.1f} h")
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set_title(rn)
        ax.set_xlabel("droplet number tendency [cm$^{-3}$ day$^{-1}$]")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_yticks(y, keys)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def install_micro_spy(micro_calls: list) -> None:
    """Make every microphysics build publish its budget and record each EAGER
    call; the scheme outputs other than the budgets are unchanged (pinned by
    test_morrison_nc_budget), and the budgets are stripped before returning."""
    import jax
    import legoesm.atmosphere.physics.microphysics.integration as mi
    real = mi._get_microphysics_fn

    def get(config):
        name, fn, cfg = real(config)
        if fn is None or not hasattr(cfg, "publish_qc_budget"):
            return name, fn, cfg

        def spy(T, q_v, hyd, p_full, p_half, rho, dz, dt, scheme_cfg, **kw):
            out = fn(T, q_v, hyd, p_full, p_half, rho, dz, dt,
                     scheme_cfg._replace(publish_qc_budget=True), **kw)
            if out.nc_budget is not None and not isinstance(out.dN_c_dt, jax.core.Tracer):
                f = lambda a: np.asarray(a, dtype=np.float64)   # noqa: E731
                micro_calls.append({
                    "terms": {k: f(v) for k, v in out.nc_budget.items()},
                    "rho": f(rho), "N_c": f(hyd.N_c), "q_c": f(hyd.q_c),
                    "cf": f(kw["cloud_fraction"]), "dN_c_dt": f(out.dN_c_dt)})
            return out._replace(qc_budget=None, nc_budget=None)
        return name, spy, cfg
    mi._get_microphysics_fn = get


def report_terrain_drift() -> None:
    """The single-process relaunch rebuilds the terrain product, which can differ
    from the 4-GPU run's by reduction-order round-off (measured 7.1e-2 m2/s2 =
    1.4e-6 of max phis on jan1979_main1027), just above the driver's 1e-6
    restart refusal.  The step uses the checkpoint's own phis either way; the
    refusal is turned into a printed line so the number travels with the
    output."""
    from legoesm.driver.model_driver import ModelDriver
    real = ModelDriver._check_restart_terrain

    def check(self):
        try:
            real(self)
        except RuntimeError as e:
            a = np.asarray(self.state.phis.data, dtype=np.float64)
            b = np.asarray(self._phis_data, dtype=np.float64)
            rel = float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))
            if "terrain product (max |d|" not in str(e) or rel > TERRAIN_DRIFT_MAX:
                raise
            print(f"TERRAIN DRIFT {rel:.2e} of max phis (reported, not refused): {e}", flush=True)
    ModelDriver._check_restart_terrain = check


def run(model, state, DT, pf_full, forcing, phys_state, calls, micro_calls,
        out_dir: Path, tag: str) -> int:
    """The mode body; ``calls`` and ``micro_calls`` are filled by the spies
    during the step's own physics call."""
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.atmosphere.physics._shared import compute_rho

    seen: dict = {}

    def pf_rec(st, *a, **kw):
        calls.clear()
        micro_calls.clear()
        seen["state"] = st
        res = pf_full(st, *a, **kw)
        seen["res"] = res
        return res

    target_mass = model._target_mass
    if model.config.fix_mass and model.config.anchor_mass_to_initial and target_mass is None:
        target_mass = model.compute_mass(state)
    step_raw = type(model)._step_jit.__wrapped__
    new = step_raw(model, state, DT, pf_rec, target_mass, forcing, phys_state)[0]
    gate("state" in seen, "the step called its physics once, eagerly")
    # the same step with NO physics: the held steps of the physics cadence
    # (zero physics increment) run the same dynamics and fixers
    held = step_raw(model, state, DT, None, target_mass, forcing, phys_state)[0]

    f = lambda a: np.asarray(a, dtype=np.float64)   # noqa: E731
    n0, npd, n1, n1h = (f(s.tracers["N_c"].data) for s in (state, seen["state"], new, held))
    tend = seen["res"][0] if type(seen["res"]) is tuple else seen["res"]   # as the step
    phys = f(tend.tracer_tendencies["N_c"].data)

    rows: dict[str, np.ndarray] = {}
    mod_rows = {}
    for name, ts in calls.items():
        arr = [t.tracer_tendencies["N_c"].data for t in ts
               if t.tracer_tendencies is not None and "N_c" in t.tracer_tendencies]
        mod_rows[name] = mean_calls(arr) if arr else np.zeros_like(n0)
        print(f"module {name}: {len(ts)} call(s)", flush=True)
    # Physics cadence: the physics step applies n_steps x the physics rate
    # (the whole window's increment), the held steps apply zero.  n_steps is
    # read off the applied tendency and must be an integer that closes.
    msum = sum(mod_rows.values())
    big = np.abs(msum) > 1e-3 * np.abs(msum).max()
    n_steps = int(round(float(np.median(phys[big] / msum[big]))))
    ok, worst = closes([n_steps * v for v in mod_rows.values()], phys)
    gate(n_steps >= 1 and ok, f"module N_c tendencies x {n_steps} (physics cadence) sum to "
                              f"the applied physics (worst rel {worst:.2e})")

    gate(len(micro_calls) == len(calls.get("microphysics", ())) > 0,
         f"{len(micro_calls)} microphysics calls published an nc_budget")
    micro = {k: mean_calls([c["terms"][k] / c["rho"] for c in micro_calls])
             for k in micro_calls[0]["terms"]}
    ok, worst = closes(list(micro.values()), mod_rows["microphysics"])
    gate(ok, f"MG2 nc_budget terms sum to the microphysics tendency (worst rel {worst:.2e})")

    # Rates [1/kg/s] averaged over one physics window of n_steps dycore steps,
    # taking this step's dynamics and held-step fixers as representative of
    # the window's (one state, so that is an assumption, printed):
    fix_held = (n1h - npd) / DT                        # fixers on a held step
    fix_phys = (n1 - npd) / DT - phys                  # fixers on the physics step
    rows["dynamics (transport)"] = (npd - n0) / DT
    rows.update({f"micro: {k}": v for k, v in micro.items()})
    rows.update({f"{k}": v for k, v in mod_rows.items() if k != "microphysics"})
    rows["fixers after dynamics (positivity, mass fix)"] = fix_held
    rows["fixers after physics, extra"] = (fix_phys - fix_held) / n_steps

    sig = model.sigma_coord
    p_s = f(state.p_s.data)
    ph = f(sig.pressure_at_half(jnp.asarray(p_s)))
    pm = f(sig.pressure_at_full(jnp.asarray(p_s)))
    area = f(model.mesh.areaCell)
    mass = area[:, None] * (ph[:, 1:] - ph[:, :-1]) / constants.g
    rho = f(compute_rho(jnp.asarray(f(state.T.data)), jnp.asarray(pm),
                        jnp.asarray(f(state.tracers["q_v"].data))))
    lat = np.rad2deg(f(model.mesh.latCell))
    qg0 = f(state.tracers["q_g"].data) if "q_g" in state.tracers else np.zeros_like(n0)
    mks = masks(lat, pm / 100.0)
    neg_dyn = np.minimum(npd, 0.0) - np.minimum(n0, 0.0)

    c0 = micro_calls[0]
    vol = mass / rho
    results, save = {}, {}
    print(f"\n{tag}: dycore dt={DT} s, physics every {n_steps} steps; rows = rates averaged "
          "over one physics window (this step's dynamics and fixers taken as the window's), "
          "cm-3/day per volume of air", flush=True)
    for key, mk in mks.items():
        inv, r, tau_all, tau = aggregate(rows, n0, mass, rho, mk)
        _, rneg, _, _ = aggregate({"neg": neg_dyn / DT}, n0, mass, rho, mk)
        ic = incloud(c0["N_c"], c0["cf"], c0["q_c"], vol, mk)
        qg = (qg0 * mass)[mk].sum() / mass[mk].sum()
        results[key] = (inv, r, tau_all, tau)
        print(f"\n[{key[0]} | {key[1]}] inventory {inv:.3f} cm-3; all-sink time {tau_all:.2f} h; "
              f"negative number left by dynamics {rneg['neg']:+.3f} cm-3/day", flush=True)
        print(f"  cloudy cells (q_c>={QC_CLOUDY:g}) n={ic['n_cells']}: grid-mean "
              f"{ic['grid_mean']:.2f} cm-3, in-cloud {ic['in_cloud']:.2f} cm-3 "
              f"(median over cf>0 {ic['median_in_cloud']:.2f}; {ic['n_cf_zero']} with cf=0), "
              f"mean cloud fraction {ic['cloud_fraction']:.3f}; mean graupel q_g {qg:.3e} kg/kg")
        for k, v in r.items():
            print(f"  {k:40s} {v:+10.4f}" + (f"   sink time {tau[k]:8.2f} h" if v < 0 else ""))
        save["|".join(key)] = np.array([inv, tau_all] + [r[k] for k in r])
        save["|".join(key) + "|incloud"] = np.array(
            [ic["n_cells"], ic["grid_mean"], ic["in_cloud"], ic["median_in_cloud"],
             ic["cloud_fraction"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"nc_budget_{tag}.npz", row_names=np.array(list(rows)),
             dt=DT, **save)
    png = out_dir / f"nc_budget_{tag}.png"
    figure(results, png, f"Droplet-number budget by process, {tag}, one step "
                         f"(dt = {DT:g} s), per volume of air")
    print(f"\nwrote {png}", flush=True)
    return 0
