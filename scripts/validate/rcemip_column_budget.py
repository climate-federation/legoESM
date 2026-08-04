"""Column energy + water budget for an existing RCEMIP plane-CRM run.

Reads the snapshots a `run_rcemip_plane.py` run already wrote
(``snapshots/sfc_*.npz`` + ``snapshots3d/vol_*.npz``) and answers ONE
question: does the RCE column budget CLOSE, or is the run accumulating
energy/water without limit?

Why this exists
---------------
Run 9285110 (128x128, dx=2 km, 80 sim-days, rrtmgp + kessler) produced a
realistic ~3 mm/day precip proxy at day 4.5 and then decayed to ~0.1 mm/day
while CWV grew MONOTONICALLY 40 -> 128 mm and T(11.5 km) warmed 217 -> 261 K.
That is a budget non-closure, not a CFL blow-up.  This script measures which
term is responsible.

INSTRUMENT CONTROLS (per CLAUDE.md "validate the instrument before quoting
its number").  Every one of these runs on every invocation and a failure is
FATAL, not a warning:

  C1  Reproduce the run's OWN ``cwv`` field from the independent
      mse->q_v inversion and hydrostatic mass element.  If our CWV does not
      match the driver's stored CWV to `--cwv-tol` relative, our column
      integral is wrong and every number below it is void.
  C2  NaN / missing frames are FATAL (no nanmean hiding a failure).
  C3  The hydrostatic pressure integration is checked against the Wing 2018
      analytic RCEMIP pressure profile at day 0 tolerance -- the reference
      the IC was built from.
  C4  Every printed rate states the WINDOW it was computed over, and both
      sides of any comparison use the SAME window.

Saturation is taken from ``legoesm.thermo`` and constants from
``legoesm.constants`` -- never re-derived here (CLAUDE.md shared-utilities).

The script prints measurements only.  It deliberately prints NO verdict:
the interpretation belongs in the analysis, not baked into the tool.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (  # noqa: E402
    WING_P_SFC,
    WING_Q_SFC_DEFAULT,
    wing2018_pressure_profile,
)

MSE_KJ_TO_J = 1.0e3  # rce_snapshot stores MSE in kJ/kg
SEC_PER_DAY = 86400.0
_RCEMIP1_C_H = 1.5e-3
_RCEMIP1_GUST_MS = 5.0


def _fatal(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _require_finite(name: str, a: np.ndarray) -> np.ndarray:
    """C2: NaN/inf anywhere is FATAL, never silently reduced away."""
    if not np.all(np.isfinite(a)):
        n_bad = int(np.sum(~np.isfinite(a)))
        _fatal(f"{name} has {n_bad}/{a.size} non-finite entries.")
    return a


def hydrostatic_pressure(z_m: np.ndarray, T: np.ndarray, qv: np.ndarray,
                         p_sfc: float) -> np.ndarray:
    """Hydrostatic p on the model half/full levels from the snapshot's OWN
    virtual temperature.

    Integrates ``dp/dz = -rho g`` with ``rho = p/(R_d T_v)`` upward from
    ``p_sfc``, i.e. ``p_{k+1} = p_k * exp(-g dz / (R_d T_v))``.  Uses the
    layer-mean T_v so the integral is second-order in dz.

    ``z_m`` is ordered TOP-DOWN in the plane CRM snapshots (z[0] is the model
    top), matching ``theta_prime[..., -1]`` being the surface.  We integrate
    from the surface (last index) upward.

    ``z_m`` holds FULL-level heights, so the first step carries the pressure
    from the ground (z = 0, p = ``p_sfc``) up to the lowest full level at its
    ACTUAL height ``z[0]`` — not a half-layer guess.

    Returns p with the same leading shape as ``T``.
    """
    # Work bottom-up, then flip back.
    z = z_m[::-1]                       # ascending height
    Tv = (T * (1.0 + (1.0 / constants.epsilon - 1.0) * qv))[..., ::-1]
    nz = z.size
    p = np.empty_like(Tv)
    p[..., 0] = p_sfc * np.exp(
        -constants.g * z[0] / (constants.R_d * Tv[..., 0]))
    for k in range(1, nz):
        dz = z[k] - z[k - 1]
        Tv_mid = 0.5 * (Tv[..., k] + Tv[..., k - 1])
        p[..., k] = p[..., k - 1] * np.exp(
            -constants.g * dz / (constants.R_d * Tv_mid))
    return p[..., ::-1]


def layer_thickness(z_m: np.ndarray) -> np.ndarray:
    """Layer thickness dz [m] per full level, top-down ordering.

    Half levels are the midpoints between full levels, with the surface pinned
    at z=0 and the model top mirrored from the topmost layer.  This reproduces
    the geometry the driver's own CWV diagnostic integrates over
    (``rce_diagnostics.column_water_vapor_plane`` uses the model ``dz``), which
    is why control C1 can be expected to close.
    """
    z_asc = z_m[::-1]
    nz = z_asc.size
    zh = np.empty(nz + 1)
    zh[0] = 0.0                                     # surface
    zh[1:nz] = 0.5 * (z_asc[:-1] + z_asc[1:])
    zh[nz] = z_asc[-1] + (z_asc[-1] - zh[nz - 1])   # mirror the top layer
    return np.diff(zh)[::-1]


def layer_mass(p: np.ndarray, z_m: np.ndarray, T: np.ndarray,
               qv: np.ndarray) -> np.ndarray:
    """Mass element dm = rho*dz [kg/m^2] per full level, top-down ordering.

    ``rho = p / (R_d T_v)`` from the ideal gas law on the snapshot's own state.

    An earlier version used ``dm = -dp/g`` with geometric-mean half-level
    pressures.  That form disagreed with the driver's stored CWV by 6.2% (its
    top boundary p=0 and its extrapolated bottom half-level are both crude),
    which control C1 caught and rejected.  The ``rho*dz`` form matches the
    height-based integral the driver actually performs.
    """
    Tv = T * (1.0 + (1.0 / constants.epsilon - 1.0) * qv)
    rho = p / (constants.R_d * Tv)
    return rho * layer_thickness(z_m)


def qv_from_mse(mse_kJ: np.ndarray, T: np.ndarray,
                z_m: np.ndarray) -> np.ndarray:
    """Invert the stored moist static energy for q_v (same algebra the SCM-RCE
    campaign uses in ``run_scm_rce_campaign._reference_profiles``)."""
    mse_J = mse_kJ.astype(np.float64) * MSE_KJ_TO_J
    z_b = z_m.reshape((1,) * (mse_J.ndim - 1) + (-1,))
    return (mse_J - constants.c_pd * T - constants.g * z_b) / constants.L_v


def _load_vol(path: Path) -> dict:
    with np.load(path) as ds:
        out = {k: np.asarray(ds[k], dtype=np.float64) for k in
               ("day", "z", "T", "mse", "cond", "qcloud", "w")}
    return out


def analyse_volumes(vol_files: list[Path], p_sfc: float,
                    cwv_tol: float, ref_cwv: dict[float, float],
                    abort_tol: float = 0.20) -> None:
    print("\n=== COLUMN BUDGET FROM 3D VOLUMES ===")
    print("all integrals: mass element dm = -dp/g, p from hydrostatic "
          "integration of the snapshot's OWN virtual temperature")
    rows = []
    for path in vol_files:
        d = _load_vol(path)
        day = float(d["day"])
        z = _require_finite("z", d["z"])
        T = _require_finite("T", d["T"])
        qv = qv_from_mse(d["mse"], T, z)
        _require_finite("qv(mse-inverted)", qv)
        qc = _require_finite("qcloud", d["qcloud"])
        cond = _require_finite("cond", d["cond"])
        p = hydrostatic_pressure(z, T, qv, p_sfc)
        dm = layer_mass(p, z, T, qv)

        cwv = np.sum(qv * dm, axis=-1)                     # kg/m^2 == mm
        cwp = np.sum(cond * dm, axis=-1)
        # Column MOIST static energy content [J/m^2].
        col_mse = np.sum(d["mse"] * MSE_KJ_TO_J * dm, axis=-1)
        # Dry static energy content, to separate sensible from latent.
        col_dse = np.sum(
            (constants.c_pd * T + constants.g * z.reshape(1, 1, -1)) * dm,
            axis=-1)

        qsat = np.asarray(saturation_mixing_ratio(jnp.asarray(T),
                                                  jnp.asarray(p)))
        rh = qv / np.maximum(qsat, 1e-30)

        rows.append(dict(
            day=day, path=path.name,
            cwv=float(cwv.mean()), cwv_max=float(cwv.max()),
            cwp=float(cwp.mean()),
            col_mse=float(col_mse.mean()), col_dse=float(col_dse.mean()),
            rh_prof=rh.mean(axis=(0, 1)), z=z,
            qc_max=float(qc.max()), qc_mean=float(qc.mean()),
            cond_max=float(cond.max()),
            w_max=float(np.abs(d["w"]).max()),
            T_prof=T.mean(axis=(0, 1)),
            p_prof=p.mean(axis=(0, 1)),
        ))

    # ---- C1: our CWV must reproduce the driver's own stored CWV ----
    n_checked = 0
    worst_rel = 0.0
    for r in rows:
        ref = ref_cwv.get(round(r["day"], 3))
        if ref is None:
            continue
        n_checked += 1
        rel = abs(r["cwv"] - ref) / max(ref, 1e-12)
        worst_rel = max(worst_rel, rel)
        status = "OK" if rel <= cwv_tol else "MISMATCH"
        print(f"  C1 day {r['day']:6.2f}: our CWV={r['cwv']:7.3f} mm  "
              f"driver CWV={ref:7.3f} mm  rel={rel:.4f}  {status}")
        if rel > abort_tol:
            _fatal(
                f"C1 FAILED at day {r['day']}: our column integral disagrees "
                f"with the driver's stored CWV by {rel:.3%} > {abort_tol:.3%}. "
                "The mass element or the mse->qv inversion is wrong; every "
                "number derived from it is void.")
    if n_checked == 0:
        _fatal("C1 could not run: no sfc snapshot shares a day with any "
               "volume snapshot. Refusing to report unvalidated integrals.")
    if worst_rel > cwv_tol:
        print(f"\n  !! C1 RESIDUAL {worst_rel:.2%} (> {cwv_tol:.0%} target). "
              "EVERY volume-derived number below carries AT LEAST this\n"
              "     relative uncertainty and must be quoted with it. Our mass "
              "element is rho*dz with rho from\n"
              "     hydrostatic p and the snapshot's T_v; the driver "
              "integrates its OWN (rho_ref + rho'). The vol_*.npz\n"
              "     files do not store rho, so the driver's exact integral "
              "cannot be reproduced from them.\n"
              "     Surface-table numbers are UNAFFECTED -- they use the "
              "driver's own stored cwv/precip directly.")
    else:
        print(f"  C1 PASSED on {n_checked} matched day(s) "
              f"(worst residual {worst_rel:.2%}).")

    print("\n  day |  CWV[mm] CWVmax |  CWP[mm] | colMSE[GJ/m2] colDSE[GJ/m2]"
          " | max|w| | qc_max   cond_max")
    for r in rows:
        print(f"{r['day']:6.2f} | {r['cwv']:8.2f} {r['cwv_max']:7.2f} | "
              f"{r['cwp']:8.4f} | {r['col_mse']/1e9:12.5f} "
              f"{r['col_dse']/1e9:12.5f} | {r['w_max']:6.2f} | "
              f"{r['qc_max']:.3e} {r['cond_max']:.3e}")

    # ---- C4: rate over a stated window, same window on both sides ----
    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        dt_s = (b["day"] - a["day"]) * SEC_PER_DAY
        if dt_s <= 0:
            _fatal("volume snapshots are not time-ordered.")
        dmse = (b["col_mse"] - a["col_mse"]) / dt_s
        ddse = (b["col_dse"] - a["col_dse"]) / dt_s
        dcwv = (b["cwv"] - a["cwv"]) / (b["day"] - a["day"])
        print(f"\n  WINDOW days {a['day']:.2f} -> {b['day']:.2f} "
              f"({b['day']-a['day']:.2f} d), same window for every rate below:")
        print(f"    d(column MSE)/dt      = {dmse:+8.2f} W/m^2")
        print(f"    d(column DSE)/dt      = {ddse:+8.2f} W/m^2  (sensible)")
        print(f"    d(column latent)/dt   = {dmse-ddse:+8.2f} W/m^2")
        print(f"    d(CWV)/dt             = {dcwv:+8.4f} mm/day "
              f"(= net E-P-condensate storage)")

    print("\n  RH [%] and T [K] horizontal-mean profiles "
          "(first vs last volume, SAME levels):")
    a, b = rows[0], rows[-1]
    print(f"    {'z[km]':>7} {'p[hPa]':>8} | "
          f"{'T@' + format(a['day'], '.0f') + 'd':>9} "
          f"{'T@' + format(b['day'], '.0f') + 'd':>9} | "
          f"{'RH@' + format(a['day'], '.0f') + 'd':>9} "
          f"{'RH@' + format(b['day'], '.0f') + 'd':>9}")
    for k in range(a["z"].size):
        print(f"    {a['z'][k]/1e3:7.2f} {a['p_prof'][k]/100:8.1f} | "
              f"{a['T_prof'][k]:9.2f} {b['T_prof'][k]:9.2f} | "
              f"{100*a['rh_prof'][k]:9.1f} {100*b['rh_prof'][k]:9.1f}")


def analyse_surface(sfc_files: list[Path], t_sfc: float,
                    stride: int) -> dict[float, float]:
    """Precip / CWV trajectory + an RCEMIP bulk surface-flux estimate.

    Returns {day: driver_cwv_mean} so the volume analysis can validate its own
    column integral against the driver's stored CWV (control C1).
    """
    print("=== SURFACE TRAJECTORY (driver's own stored fields) ===")
    print("  precip is the driver's PROXY (q_r[k_sfc]*rho*5 m/s), NOT a "
          "microphysics flux -- see rce_diagnostics.precipitation_rate_proxy_plane")
    print(f"  E is a BULK ESTIMATE recomputed here: C_h={_RCEMIP1_C_H}, "
          f"gust floor={_RCEMIP1_GUST_MS} m/s, T_sfc={t_sfc} K")
    print("  P_budget is the WATER-BUDGET estimate P = E - d(CWV)/dt, "
          "centred-differenced between consecutive snapshots.")
    print("  CWV_max/CWV_mean is the convective-cell discriminator: real cells "
          "give max >> mean; the column-symmetric trap never crosses 60 mm.")
    ref = {}
    qsat_sfc = float(saturation_mixing_ratio(
        jnp.asarray(t_sfc), jnp.asarray(WING_P_SFC)))
    days, cwv_m, cwv_x, p_proxy, p_max, e_est, spd_m = ([] for _ in range(7))
    for path in sfc_files:
        with np.load(path) as ds:
            day = float(ds["day"])
            precip = _require_finite("precip", np.asarray(ds["precip"]))
            cwv = _require_finite("cwv", np.asarray(ds["cwv"]))
            qv_s = _require_finite("qv_sfc", np.asarray(ds["qv_sfc"]))
            spd = np.sqrt(np.asarray(ds["u_sfc"]) ** 2
                          + np.asarray(ds["v_sfc"]) ** 2)
        ref[round(day, 3)] = float(cwv.mean())
        # RCEMIP1 bulk evaporation with the code's OWN coefficients
        # (rce_surface_flux._RCEMIP1_C_H / _RCEMIP1_GUSTINESS_FLOOR_MS).
        # Surface air density from the ideal gas law on the run's own p_sfc and
        # near-surface humidity -- no hardcoded density literal.
        rho_s = WING_P_SFC / (constants.R_d * t_sfc
                              * (1.0 + (1.0 / constants.epsilon - 1.0) * qv_s))
        spd_eff = np.sqrt(spd ** 2 + _RCEMIP1_GUST_MS ** 2)
        evap = rho_s * _RCEMIP1_C_H * spd_eff * (qsat_sfc - qv_s)
        days.append(day)
        cwv_m.append(float(cwv.mean()))
        cwv_x.append(float(cwv.max()))
        p_proxy.append(float(precip.mean()))
        p_max.append(float(precip.max()))
        e_est.append(float(evap.mean()) * SEC_PER_DAY)
        spd_m.append(float(spd.mean()))
    days = np.asarray(days)
    cwv_m = np.asarray(cwv_m)
    e_est = np.asarray(e_est)
    # d(CWV)/dt by centred differences on the SAME grid as E.
    dcwv = np.gradient(cwv_m, days)
    p_budget = e_est - dcwv

    print("\n   day | P_proxy  P_max | P_budget | CWV_mean CWV_max  max/mean | "
          "E_est | dCWV/dt | |U|sfc")
    print("       |   [mm/d]       |   [mm/d] |    [mm]     [mm]           | "
          "[mm/d] |  [mm/d] |  [m/s]")
    for i in range(len(days)):
        if i % stride and i != len(days) - 1:
            continue
        print(f"{days[i]:6.2f} | {p_proxy[i]:7.4f} {p_max[i]:6.1f} | "
              f"{p_budget[i]:8.3f} | {cwv_m[i]:8.2f} {cwv_x[i]:8.2f} "
              f"{cwv_x[i]/max(cwv_m[i], 1e-9):9.3f} | {e_est[i]:6.3f} | "
              f"{dcwv[i]:7.3f} | {spd_m[i]:6.3f}")

    # C4: rates over an explicitly stated window, printed next to the number.
    for lo, hi in ((0.0, 10.0), (10.0, 30.0), (days[-1] - 20.0, days[-1])):
        m = (days >= lo) & (days <= hi)
        if m.sum() < 2:
            continue
        print(f"\n  WINDOW days {lo:.1f}-{hi:.1f} ({int(m.sum())} snapshots), "
              "every number below from THIS window only:")
        print(f"    P_budget = E - dCWV/dt : {p_budget[m].mean():7.3f} mm/day "
              f"(Wing 2018 RCE plateau ~3 mm/day)")
        print(f"    P_proxy  (q_r * 5 m/s) : "
              f"{np.asarray(p_proxy)[m].mean():7.3f} mm/day")
        print(f"    E_est                  : {e_est[m].mean():7.3f} mm/day")
        print(f"    CWV_mean               : {cwv_m[m].mean():7.2f} mm "
              f"(drift {cwv_m[m][-1] - cwv_m[m][0]:+.2f} mm over the window)")
        print(f"    CWV_max/CWV_mean       : "
              f"{np.mean(np.asarray(cwv_x)[m] / cwv_m[m]):7.3f}")
    return ref


def check_analytic_pressure(p_sfc: float) -> None:
    """C3: the hydrostatic integrator must reproduce the Wing 2018 analytic
    RCEMIP pressure profile on the analytic IC it was built from."""
    from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
        wing2018_qv_profile, wing2018_temperature_profile)
    z = np.linspace(50.0, 32_000.0, 200)
    zj = jnp.asarray(z)
    qv = np.asarray(wing2018_qv_profile(zj, q_sfc=WING_Q_SFC_DEFAULT))
    T = np.asarray(wing2018_temperature_profile(zj, q_sfc=WING_Q_SFC_DEFAULT))
    p_ana = np.asarray(wing2018_pressure_profile(zj, q_sfc=WING_Q_SFC_DEFAULT))
    # our integrator wants TOP-DOWN ordering
    p_num = hydrostatic_pressure(z[::-1], T[::-1], qv[::-1], p_sfc)[::-1]
    rel = np.abs(p_num - p_ana) / p_ana
    worst = float(rel.max())
    print(f"  C3 hydrostatic integrator vs Wing2018 analytic p: "
          f"max rel err = {worst:.4%} "
          f"(at z={float(z[int(np.argmax(rel))])/1e3:.1f} km)")
    if worst > 0.02:
        _fatal(f"C3 FAILED: hydrostatic pressure integrator is off by "
               f"{worst:.2%} vs the analytic profile the IC was built from.")
    print("  C3 PASSED.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--p-sfc", type=float, default=WING_P_SFC)
    p.add_argument("--T-sfc", type=float, default=300.0)
    p.add_argument("--sfc-stride", type=int, default=8)
    p.add_argument("--cwv-tol", type=float, default=0.05,
                   help="max relative disagreement between our column CWV "
                        "and the driver's stored CWV before C1 fails")
    p.add_argument("--cwv-abort-tol", type=float, default=0.20,
                   help="C1 residual above which the run is ABORTED rather "
                        "than reported with an uncertainty tag.")
    p.add_argument("--include-quarantine", action="store_true")
    args = p.parse_args(argv)

    run_dir = args.run_dir
    if not run_dir.is_dir():
        _fatal(f"--run-dir {run_dir} does not exist.")
    print(f"run-dir : {run_dir}")
    print(f"argv    : {' '.join(sys.argv[1:])}")

    print("\n=== INSTRUMENT CONTROLS ===")
    check_analytic_pressure(args.p_sfc)

    sfc = sorted(Path(x) for x in glob.glob(str(run_dir / "snapshots" /
                                                "sfc_*.npz")))
    vol = sorted(Path(x) for x in glob.glob(str(run_dir / "snapshots3d" /
                                                "vol_*.npz")))
    if args.include_quarantine:
        vol += sorted(Path(x) for x in
                      glob.glob(str(run_dir / "_quarantine_post_tip" /
                                    "vol_*.npz")))
        vol = sorted(vol, key=lambda q: q.name)
    if not sfc:
        _fatal(f"no sfc_*.npz under {run_dir/'snapshots'}")
    if not vol:
        _fatal(f"no vol_*.npz under {run_dir/'snapshots3d'}")
    print(f"  found {len(sfc)} sfc snapshots, {len(vol)} volume snapshots")

    ref_cwv = analyse_surface(sfc, args.T_sfc, max(1, args.sfc_stride))
    analyse_volumes(vol, args.p_sfc, args.cwv_tol, ref_cwv,
                    abort_tol=args.cwv_abort_tol)
    print("\nDIAG_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
