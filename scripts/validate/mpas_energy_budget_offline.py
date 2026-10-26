#!/usr/bin/env python3
"""OFFLINE column energy-budget closure for an MPAS AMIP run (#1354).

The in-run tracker (``mpas_energy_budget_leak.py``) needs the end-of-run
``timeseries.npz``; a chain link killed by wallclock never writes one.  This
recomputes the same budget from what every link DOES leave behind:

  storage  E at each checkpoint, by the tracker's own function
           (``column_moist_static_energy``: c_p T + L_v q_v - L_f q_frozen
           + Phi + KE, Perot cell winds, q_frozen = q_i + q_s + q_g) and, as a
           second definition, ``process_ledger.column_total_energy`` (c_p T +
           KE + L_v q_v - L_f q_frozen + Phi_s p_s / g).  Area-weighted
           (``mesh.areaCell``) global means; dE/dt = Delta E / Delta t.
  fluxes   window means from the CMOR monthly-accumulator sidecars
           (``cmor_accum_day_<d>.npz``): per variable, (sum_1 - sum_0) over
           every month bucket / (count_1 - count_0), cos-lat mean on the 5-deg
           CMOR grid.  The count increment must equal the window length in days
           (one interval-mean sample per day) or the window is refused.

SIGNS (everything positive INTO the atmospheric column; same as the tracker):

  F    = (rsdt - rsut - rlut) - (rsds - rsus + rlds - rlus) + hfss + hfls
  LEAK = dE/dt - F                       (> 0: energy appears with no flux)

Equivalent to the tracker's LEAK = sfc_rad - hfss - hfls - (toa_net - dE/dt).
NOT in F: the enthalpy carried by precipitation (frozen precip leaving raises
E by L_f * snowfall -- printed as the upper bound L_f * pr) and the latent
correction of sublimation.

``--control TIMESERIES`` validates the instrument on a run that ALSO has the
in-run interval-mean series: the mean of the tracker's daily samples over the
window's days is printed next to the offline number for every term.

Prints numbers only; no verdict.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

from legoesm import constants

_VAL = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_VAL / "amip_bias"))
from cloud_layers import cell_order  # noqa: E402

FLUX_VARS = ("rsdt", "rsut", "rlut", "rsds", "rsus", "rlds", "rlus",
             "hfss", "hfls", "pr")


def build_mesh(exp):
    """The run's mesh, built the way the driver builds it
    (``ModelDriver._create_grid``: mpas + 50 Lloyd iterations)."""
    g = exp["grid"]
    if g["grid_type"] != "mpas":
        raise SystemExit("FATAL: MPAS checkpoints only")
    from legoesm.grids.factory import create_grid
    return create_grid("mpas", int(g["resolution"]), lloyd_iterations=50)


def load_energy(path, mesh):
    """Global area-weighted means of the two energy definitions plus state
    checks, for one checkpoint (cells put into mesh order)."""
    import jax.numpy as jnp
    from legoesm.diagnostics.energy_budget import column_moist_static_energy
    from legoesm.diagnostics.process_ledger import column_total_energy
    from legoesm.grids.voronoi import reconstruct_cell_velocity

    z = np.load(path, allow_pickle=True)
    ncell = int(np.asarray(mesh.areaCell).size)
    order = cell_order(z, ncell)
    if not np.array_equal(order, np.arange(ncell)):
        raise SystemExit(f"FATAL: {path}: cell order is not the identity; edge "
                         "order cannot be proven, refusing the KE term")
    T = np.asarray(z["T"], dtype=np.float64)
    p_s = np.asarray(z["p_s"], dtype=np.float64)
    phis = np.asarray(z["phis"], dtype=np.float64)
    q_v = np.asarray(z["trc_q_v"], dtype=np.float64)
    q_fr = sum(np.asarray(z[k], dtype=np.float64)
               for k in ("trc_q_i", "trc_q_s", "trc_q_g") if k in z.files)
    u_e = np.asarray(z["u"], dtype=np.float64)
    if u_e.shape[0] != int(np.asarray(mesh.dvEdge).size):
        raise SystemExit(f"FATAL: {path}: {u_e.shape[0]} edges vs mesh "
                         f"{np.asarray(mesh.dvEdge).size}")
    for name, a in (("T", T), ("p_s", p_s), ("q_v", q_v), ("u", u_e)):
        if not np.isfinite(a).all():
            raise SystemExit(f"FATAL: {path}: non-finite {name}")
    uc, vc_ = (np.asarray(x) for x in reconstruct_cell_velocity(jnp.asarray(u_e), mesh))
    # Pressure from the run's OWN recorded (A, B): p = A p_ref + B p_s
    # (the checkpoint's meta_vgrid, as cap_water.load_state reads it).
    vg = np.asarray(z["meta_vgrid"], dtype=np.float64)
    p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * p_s[:, None]
    dp = np.diff(p_half, axis=1)
    if not (dp > 0).all():
        raise SystemExit(f"FATAL: {path}: pressure not increasing top-down")
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    sig_h = vg[0] + vg[1]
    area = np.asarray(mesh.areaCell, dtype=np.float64)
    w = area / area.sum()
    E_mse = np.asarray(column_moist_static_energy(
        jnp.asarray(T), jnp.asarray(q_v), jnp.asarray(uc), jnp.asarray(vc_),
        jnp.asarray(phis), jnp.asarray(p_s), jnp.asarray(np.diff(sig_h)),
        jnp.asarray(0.5 * (sig_h[1:] + sig_h[:-1])), dp=jnp.asarray(dp),
        p_full=jnp.asarray(p_full), q_frozen=jnp.asarray(q_fr)))
    E_te = np.asarray(column_total_energy(
        jnp.asarray(p_s), None, jnp.asarray(T), u=jnp.asarray(uc),
        v=jnp.asarray(vc_), phis=jnp.asarray(phis), q_v=jnp.asarray(q_v),
        q_ice=jnp.asarray(q_fr), dp=jnp.asarray(dp)))
    # Instrument check on the winds: Perot KE vs the edge-based MPAS KE
    # (KE_i = sum_e dc*dv/4 u_e^2 / A_i; summed over cells each edge counts
    # twice, so the global mean is sum_e (dc*dv/2) u_e^2 / sum_e (dc*dv/2) --
    # no factor 1/2: the mean of u_n^2 over orientations is already |V|^2/2).  A wrong edge order
    # makes the reconstructed vectors cancel and this ratio collapse.
    dpm = (dp * w[:, None]).sum(0) / constants.g          # mean layer mass
    ke_perot = float((0.5 * (uc ** 2 + vc_ ** 2) * dp / constants.g * w[:, None]).sum())
    ae = 0.5 * np.asarray(mesh.dcEdge) * np.asarray(mesh.dvEdge)   # edge area
    ke_edge = float((u_e ** 2 * ae[:, None]).sum(0) @ dpm / ae.sum())
    return {
        "day": float(z["day"]), "E_mse": float((E_mse * w).sum()),
        "E_te": float((E_te * w).sum()),
        "T_min": float(T.min()), "T_max": float(T.max()),
        "T_mass": float(((T * dp).sum(1) / dp.sum(1) * w).sum()),
        "cwv": float(((q_v * dp).sum(1) / constants.g * w).sum()),
        "ke_perot": ke_perot, "ke_edge": ke_edge,
        "argmax_T": int(np.unravel_index(T.argmax(), T.shape)[0]),
    }


def accum_sums(path):
    """{var: (sum over ALL month buckets (nlat,nlon), total count)} from a
    sidecar; a missing path (day 0) is the empty accumulator."""
    if path is None:
        return {}
    z = np.load(path, allow_pickle=True)
    man = json.loads(str(z["monthly.__manifest__"]))
    if man["type"] != "SpatialMonthlyAccumulator":
        raise SystemExit(f"{path}: unexpected accumulator {man['type']!r}")
    out = {}
    for y, m, var, count, key in man["data_2d"]:
        if var not in FLUX_VARS:
            continue
        s = np.asarray(z[f"monthly.{key}"], dtype=np.float64)
        if var in out:
            out[var] = (out[var][0] + s, out[var][1] + int(count))
        else:
            out[var] = (s, int(count))
    return out


def window_flux_means(run, d0, d1):
    p0 = None if d0 == 0 else run / f"cmor_accum_day_{d0:04d}.npz"
    a, b = accum_sums(p0), accum_sums(run / f"cmor_accum_day_{d1:04d}.npz")
    out = {}
    for var in FLUX_VARS:
        if var not in b:
            raise SystemExit(f"{run}: sidecar day {d1} lacks {var}")
        s0, c0 = a.get(var, (0.0, 0))
        dn = b[var][1] - c0
        if dn != d1 - d0:
            raise SystemExit(f"{run}/{var}: {dn} samples in days {d0}..{d1}, "
                             f"expected {d1 - d0} (one per day)")
        f = (b[var][0] - s0) / dn
        lat = -87.5 + 5.0 * np.arange(f.shape[0])     # 5-deg CMOR grid, symmetric
        if f.shape != (36, 72):
            raise SystemExit(f"{run}/{var}: unexpected CMOR grid {f.shape}")
        wl = np.cos(np.deg2rad(lat))
        out[var] = float((f.mean(1) * wl).sum() / wl.sum())
    return out


def control_means(ts_path, d0, d1, offset=0):
    """Mean of the in-run tracker's daily samples over days d0+1 .. d1
    (series days are relative to the link start: absolute = days + offset)."""
    t = np.load(ts_path, allow_pickle=True)
    days = np.asarray(t["days"], dtype=float) + offset
    k = (days > d0 + 1e-9) & (days <= d1 + 1e-9)
    if int(k.sum()) != d1 - d0:
        raise SystemExit(f"control: {int(k.sum())} samples in ({d0},{d1}]")
    if float(np.min(np.asarray(t["energy_flux_interval_mean"])[k])) < 1.0:
        raise SystemExit("control: series is not interval means")
    g = {n: float(np.asarray(t[n], dtype=float)[k].mean()) for n in
         ("energy_toa_net", "energy_dE_dt", "sw_net_sfc", "lw_net_sfc",
          "hfss", "hfls")}
    g["F"] = (g["energy_toa_net"] - g["sw_net_sfc"] - g["lw_net_sfc"]
              + g["hfss"] + g["hfls"])
    g["LEAK"] = g["energy_dE_dt"] - g["F"]
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("--days", type=int, nargs="+", required=True,
                    help="checkpoint days; consecutive pairs are the windows "
                         "(0 = start, uses no checkpoint: windows from 0 are "
                         "flux-only and skipped for the budget)")
    ap.add_argument("--control", type=pathlib.Path, default=None)
    ap.add_argument("--control-day-offset", type=int, default=0,
                    help="absolute day of the control series' day 0 (the "
                         "run's start_day; its series counts from there)")
    args = ap.parse_args(argv)
    try:
        sha = subprocess.run(["git", "-C", str(_VAL), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = "?"
    import legoesm.diagnostics.energy_budget as _eb
    exp = json.load(open(args.run / "experiment_config.json"))
    print(f"run {args.run}  probe git {sha}  legoesm from {pathlib.Path(_eb.__file__).parents[3]}")
    mesh = build_mesh(exp)
    st = {}
    for d in args.days:
        st[d] = load_energy(args.run / f"checkpoint_day_{d:04d}.npz", mesh)
        if abs(st[d]["day"] - d) > 1e-6:
            raise SystemExit(f"FATAL: checkpoint day stamp {st[d]['day']} != {d}")
        s = st[d]
        print(f"day {d:4d}  E_mse {s['E_mse']:.6e}  E_te {s['E_te']:.6e} J/m2  "
              f"T_mass {s['T_mass']:.2f}  T[{s['T_min']:.1f},{s['T_max']:.1f}] "
              f"(argmax cell {s['argmax_T']})  CWV {s['cwv']:.2f}  "
              f"KE perot/edge {s['ke_perot']:.4e}/{s['ke_edge']:.4e} = "
              f"{s['ke_perot'] / s['ke_edge']:.4f}")
    hdr = (f"{'window':>9s} {'dEmse/dt':>9s} {'dEte/dt':>8s} {'toa_net':>8s} "
           f"{'sfc_rad':>8s} {'hfss':>7s} {'hfls':>7s} {'F':>8s} "
           f"{'LEAKmse':>8s} {'LEAKte':>8s} {'Lf*pr':>6s}")
    print("\n[W/m2]  F = toa_net - sfc_rad + hfss + hfls;  LEAK = dE/dt - F")
    print(hdr)
    rows = []
    for d0, d1 in zip(args.days[:-1], args.days[1:]):
        dt = (d1 - d0) * 86400.0
        f = window_flux_means(args.run, d0, d1)
        toa = f["rsdt"] - f["rsut"] - f["rlut"]
        sfc = f["rsds"] - f["rsus"] + f["rlds"] - f["rlus"]
        F = toa - sfc + f["hfss"] + f["hfls"]
        dEm = (st[d1]["E_mse"] - st[d0]["E_mse"]) / dt
        dEt = (st[d1]["E_te"] - st[d0]["E_te"]) / dt
        lfp = constants.L_f * f["pr"]  # latent-ok: frozen-MSE reference L_f, the tracker's own convention
        rows.append((dEm - F, dEt - F))
        print(f"{d0:4d}-{d1:<4d} {dEm:9.3f} {dEt:8.3f} {toa:8.3f} {sfc:8.3f} "
              f"{f['hfss']:7.3f} {f['hfls']:7.3f} {F:8.3f} {dEm - F:8.3f} "
              f"{dEt - F:8.3f} {lfp:6.2f}")
        if args.control is not None:
            g = control_means(args.control, d0, d1, args.control_day_offset)
            print(f"{'tracker':>9s} {g['energy_dE_dt']:9.3f} {'':>8s} "
                  f"{g['energy_toa_net']:8.3f} "
                  f"{g['sw_net_sfc'] + g['lw_net_sfc']:8.3f} {g['hfss']:7.3f} "
                  f"{g['hfls']:7.3f} {g['F']:8.3f} {g['LEAK']:8.3f}")
    L = np.array(rows)
    print(f"\nLEAK over {len(rows)} windows: mse mean {L[:, 0].mean():.3f} "
          f"[{L[:, 0].min():.3f}, {L[:, 0].max():.3f}]  te mean "
          f"{L[:, 1].mean():.3f} [{L[:, 1].min():.3f}, {L[:, 1].max():.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
