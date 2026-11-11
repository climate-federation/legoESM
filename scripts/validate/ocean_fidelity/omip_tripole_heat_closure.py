#!/usr/bin/env python
"""Global heat-closure gate for a tripole OMIP run (conservation gate, P6).

For each pair of consecutive snapshots of a run made with --flux-accumulate,
compares the change of global ocean heat content with the heat the run
applied over that window:

    residual = dH/dt - (Q_qnet + Q_geothermal + Q_isf_latent)        [W]

* dH: rho_0 * c_sw * sum(T * h * area * wet) at both window ends, through
  ``legoesm.ocean.budgets.compute_tracer_budget`` with the run's own
  thickness: NEMO e3t_1d -> z* (``create_z_star_from_thicknesses``) ->
  partial cells from the snapshot's H_bathy (``create_partial_cell_coordinate``)
  -> eta stretch (``compute_layer_thickness``), as run_omip_core2 builds it.
* Q_qnet: window-mean APPLIED q_net (snapshot ``q_net_mean``) x area; q_net
  already carries the SSS-restoring heat and the ice blending.
* Q_geothermal: the constant seafloor flux over wet columns (post-step BC).
* Q_isf_latent: -fwf * L_fus_isf over the ice-shelf melt cells.

NOT closed: freshwater heat outside q_net.  In the model (freshwater.py,
the entry-heat rule) P-E and restoring water enter at 0 degC, so their heat
is inside q_net; runoff enters at max(SST, 0) and ice meltwater at SST, and
those two carry heat this probe cannot see (the run saves no runoff/ice
freshwater fields); so does the global freshwater-normalisation correction,
which joins the top-cell remainder entering at SST.  Runoff uses a smooth
max(SST, 0).  Their size (~0.1 W/m2 global for ~1 Sv runoff + ~1.5 Sv ice
melt; the normalisation term is the removed global mean x SST) is stated as
the unmatched term, not computed.

--ew-cyclic-overlap runs store NEMO's two halo columns as copies; every sum
here counts each ocean column once (``drop_cyclic_overlap``).

Window pairing: the accumulator adds the COMPLETED state of every step and
the forcing applied over it, so snapshot k's flux means cover exactly the
steps between snapshot k-1 and snapshot k.

Each term is converted to heat-content units with the constants its step
uses: geothermal heats with constants.rho_ocean/c_sw, the ISF step with the
run's rho_0 and constants.c_p_seawater, while content uses the manifest
rho_0/c_sw.  The ISF month is taken per step exactly as the driver does
(``_runoff_month_idx``: day = step*dt/86400, NOLEAP month ends, side=right).

Controls built in (a failing control is fatal, never a warning):
* the snapshots must carry ``z_interface_ref`` (--kprofile-snapshots) and the
  reconstructed reference interfaces must equal it;
* every window must carry ``q_net_mean``, a step count equal to the step
  difference of its two snapshots, and a finite positive duration equal to
  its day difference;
* configurations that change heat outside these terms (prescribed-ice
  freezing relaxation, WOA nudging, geothermal map) are refused.

Exit 1 when any window's |residual| exceeds --max-residual-wm2.

fp64 throughout; NaN anywhere is fatal.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from legoesm import constants  # noqa: E402

_SEC_PER_DAY = 86400.0
_SNAP_RE = re.compile(r"snapshot_(day\d{4}|final)\.npz$")
_MONTH_END = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def _fatal_nonfinite(name, arr):
    if not np.all(np.isfinite(arr)):
        raise SystemExit(f"FATAL: non-finite values in {name}")


def drop_cyclic_overlap(snap):
    """--ew-cyclic-overlap stores NEMO's two halo columns (0 == nx-2,
    nx-1 == 1); count each ocean column once by masking the copies."""
    lm = snap["land_mask"].copy()
    lm[:, 0] = 0.0
    lm[:, -1] = 0.0
    return {**snap, "land_mask": lm}


def load_snapshot(path):
    keys = ("T", "eta", "H_bathy", "land_mask", "cell_area", "step",
            "time_days", "z_interface_ref", "q_net_mean", "flux_mean_n_steps",
            "flux_mean_window_s")
    with np.load(path) as z:
        out = {k: np.asarray(z[k], dtype=np.float64) for k in keys if k in z.files}
    for k in ("T", "eta", "H_bathy", "land_mask", "cell_area"):
        if k not in out:
            raise SystemExit(f"FATAL: {path} lacks {k}")
        _fatal_nonfinite(f"{path}:{k}", out[k])
    return out


def build_z_coord(e3t_1d, gdept_1d, H_bathy):
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())   # the runner's default; ladders round in fp64
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate,
        create_z_star_from_thicknesses,
    )
    zs = create_z_star_from_thicknesses(
        np.asarray(e3t_1d, dtype=np.float64), gdept_1d,
        nemo_e3w_source="depth_difference")
    return zs, create_partial_cell_coordinate(zs, np.asarray(H_bathy))


def heat_content(snap, z_coord, rho_0, c_p):
    from legoesm.ocean.budgets import compute_tracer_budget

    def _f(a):
        return SimpleNamespace(data=a)
    state = SimpleNamespace(T=_f(snap["T"]), S=_f(np.zeros_like(snap["T"])),
                            eta=_f(snap["eta"]), H_bathy=_f(snap["H_bathy"]),
                            land_mask=_f(snap["land_mask"]))
    tb = compute_tracer_budget(state, z_coord, grid_type="tripole",
                               grid=SimpleNamespace(area=snap["cell_area"]),
                               rho_0=rho_0, c_p=c_p)
    return tb.heat_content, tb.volume


def wet_area(snap):
    return float((snap["cell_area"] * (snap["land_mask"] > 0.5)).sum())


def geothermal_power(snap, flux_wm2):
    wet_col = (snap["land_mask"] > 0.5) & (snap["H_bathy"] > 0.0)
    return float(flux_wm2 * (snap["cell_area"] * wet_col).sum())


def isf_latent_power(fwf, area, L_fus):
    return float(-(fwf * area).sum() * L_fus)


def step_month_weights(step_a, step_b, dt):
    """Fraction of the window's steps in each calendar month (driver rule)."""
    steps = np.arange(int(step_a) + 1, int(step_b) + 1)
    months = np.searchsorted(_MONTH_END, (steps * dt / _SEC_PER_DAY) % 365.0,
                             side="right")
    return np.bincount(months, minlength=12)[:12] / steps.size


def close_windows(snaps, z_coord_for, rho_0, c_p, flux_geo_wm2, isf_fwf_for,
                  L_fus, geo_scale=1.0, isf_scale=1.0):
    """Per-window closure records for consecutive snapshots (list of dicts)."""
    rows = []
    for a, b in zip(snaps[:-1], snaps[1:]):
        if "q_net_mean" not in b:
            raise SystemExit(f"FATAL: window ending day {b['time_days']} has no "
                             "q_net_mean (run without --flux-accumulate)")
        n_steps = int(b["flux_mean_n_steps"])
        dstep = int(b["step"]) - int(a["step"])
        if n_steps != dstep:
            raise SystemExit(f"FATAL: window ending day {float(b['time_days'])}: "
                             f"accumulator holds {n_steps} steps, snapshots "
                             f"differ by {dstep}")
        dt_win = float(b["flux_mean_window_s"])
        span = (float(b["time_days"]) - float(a["time_days"])) * _SEC_PER_DAY
        if not (np.isfinite(dt_win) and dt_win > 0.0
                and abs(dt_win - span) <= 1e-6 * span):
            raise SystemExit(f"FATAL: window ending day {float(b['time_days'])}: "
                             f"duration {dt_win} s != snapshot span {span} s")
        H_a, _ = heat_content(a, z_coord_for(a), rho_0, c_p)
        H_b, _ = heat_content(b, z_coord_for(b), rho_0, c_p)
        dHdt = (H_b - H_a) / dt_win
        q = b["q_net_mean"]
        _fatal_nonfinite("q_net_mean", q)
        wet = b["land_mask"] > 0.5
        Q_qnet = float((q * b["cell_area"] * wet).sum())
        Q_geo = geo_scale * geothermal_power(b, flux_geo_wm2)
        Q_isf = (0.0 if isf_fwf_for is None else isf_scale * isf_latent_power(
            isf_fwf_for(a, b, dt_win / n_steps), b["cell_area"], L_fus))
        resid = dHdt - (Q_qnet + Q_geo + Q_isf)
        if not np.isfinite(resid):
            raise SystemExit(f"FATAL: non-finite residual, window ending day "
                             f"{float(b['time_days'])}")
        A = wet_area(b)
        row = {"day_start": float(a["time_days"]), "day_end": float(b["time_days"]),
               "n_steps": n_steps, "window_s": dt_win,
               "dHdt_W": dHdt, "Q_qnet_W": Q_qnet, "Q_geothermal_W": Q_geo,
               "Q_isf_latent_W": Q_isf, "residual_W": resid,
               "residual_Wm2": resid / A,
               "residual_pct_of_qnet": (100.0 * resid / abs(Q_qnet)
                                        if Q_qnet != 0.0 else None)}
        rows.append(row)
    return rows


def _manifest_args(run_dir):
    m = json.loads((Path(run_dir) / "run_manifest.json").read_text())
    from scripts.run.run_omip_core2 import _build_arg_parser
    argv = shlex.split(m["run"]["command_line"])[1:]
    args = _build_arg_parser().parse_args(argv)
    consts = m["config"]["resolved_config"]["runtime_config"]["constants"]
    return m, args, float(consts["rho_0"]), float(consts["c_sw"])


def _git_sha():
    try:
        return subprocess.run(["git", "-C", _PROJECT_ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--domain-cfg", required=True,
                   help="NEMO vertical file the run read (e3t_1d, gdept_1d)")
    p.add_argument("--max-residual-wm2", type=float, required=True,
                   help="gate: fail when any window's |residual| exceeds this")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    import xarray as xr
    run_dir = Path(a.run_dir)
    manifest, rargs, rho_0, c_p = _manifest_args(run_dir)
    if getattr(rargs, "geothermal_map", None):
        raise SystemExit("FATAL: --geothermal-map runs are not supported (constant flux only)")
    if (rargs.ice_thermo or rargs.freeze_floor
            or float(rargs.nudge_woa_tau_days) > 0.0):
        raise SystemExit("FATAL: --ice-thermo freezing relaxation, --freeze-floor or "
                         "WOA nudging change heat outside the closed terms; not supported")
    if not (rargs.partial_cell and rargs.nemo_vertical) or rargs.fp32:
        raise SystemExit("FATAL: only --partial-cell --nemo-vertical fp64 tripole runs "
                         "are supported (the geometry reconstructed here)")
    if any(run_dir.glob("snapshot_year*.npz")) or (
            float(rargs.snapshot_every_days) != int(rargs.snapshot_every_days)):
        raise SystemExit("FATAL: annual or fractional-day snapshots also drain the flux "
                         "means; only whole-day snapshots are supported")
    from legoesm.ocean.physics.geothermal import GeothermalConfig
    flux_geo = 0.0
    if rargs.geothermal:
        flux_geo = (float(rargs.geothermal_flux_wm2)
                    if rargs.geothermal_flux_wm2 is not None
                    else float(GeothermalConfig().flux_wm2))
    with xr.open_dataset(a.domain_cfg, decode_times=False) as ds:
        e3t_1d = np.asarray(ds["e3t_1d"]).squeeze().astype(np.float64)
        gdept_1d = (np.asarray(ds["gdept_1d"]).squeeze().astype(np.float64)
                    if "gdept_1d" in ds else None)

    by_step = {}
    for p in sorted(run_dir.glob("snapshot_*.npz")):
        if _SNAP_RE.search(p.name):
            s = load_snapshot(p)
            by_step.setdefault(int(s["step"]), (p, s))   # final == last day: keep one
    paths = [by_step[k][0] for k in sorted(by_step)]
    snaps = [by_step[k][1] for k in sorted(by_step)]
    if rargs.ew_cyclic_overlap:
        snaps = [drop_cyclic_overlap(s) for s in snaps]
    if len(snaps) < 2:
        raise SystemExit("FATAL: need at least two snapshots at distinct steps")

    def z_coord_for(s):
        zs, pc = build_z_coord(e3t_1d, gdept_1d, s["H_bathy"])
        zi = s.get("z_interface_ref")
        if zi is None:
            raise SystemExit("FATAL: snapshot lacks z_interface_ref (run without "
                             "--kprofile-snapshots); geometry cannot be checked")
        rec = np.abs(np.asarray(zs.z_half_ref, dtype=np.float64)[1:-1])
        if rec.shape != zi.shape or not np.allclose(rec, zi, rtol=0, atol=1e-9):
            raise SystemExit("FATAL: reconstructed interfaces != snapshot "
                             "z_interface_ref (wrong --domain-cfg?)")
        return pc

    isf_fwf_for = None
    if getattr(rargs, "isf", False) and rargs.isf_forcing_file:
        from legoesm.ocean.forcing.isf_spe import load_isf_spe_forcing
        s0 = snaps[0]
        isf = load_isf_spe_forcing(rargs.isf_forcing_file, _snap_lat(paths[0]),
                                   _snap_lon(paths[0]), land_mask=s0["land_mask"])
        fwf12 = np.asarray(isf.fwf, dtype=np.float64)

        def isf_fwf_for(sa, sb, dt):
            w = step_month_weights(sa["step"], sb["step"], dt)
            return np.tensordot(w, fwf12, axes=1)

    # Content units: rho_0*c_sw (manifest).  Geothermal heats with
    # rho_ocean*c_sw (library), the ISF step with rho_0*c_p_seawater.
    geo_scale = rho_0 * c_p / (float(constants.rho_ocean) * float(constants.c_sw))
    isf_scale = c_p / float(constants.c_p_seawater)
    rows = close_windows(snaps, z_coord_for, rho_0, c_p, flux_geo, isf_fwf_for,
                         float(constants.L_fus_isf_nemo),
                         geo_scale=geo_scale, isf_scale=isf_scale)
    out = {"run_dir": str(run_dir), "git_sha": _git_sha(),
           "run_git_sha": manifest.get("legoESM", {}).get("commit"),
           "domain_cfg": a.domain_cfg, "rho_0": rho_0, "c_sw": c_p,
           "geothermal_flux_wm2": flux_geo, "snapshots": [p.name for p in paths],
           "windows": rows}
    for r in rows:
        print(f"[closure] day {r['day_start']:6.2f}->{r['day_end']:6.2f}: "
              f"dH/dt {r['dHdt_W']:.4e} W  q_net {r['Q_qnet_W']:.4e}  "
              f"geo {r['Q_geothermal_W']:.3e}  isf {r['Q_isf_latent_W']:.3e}  "
              f"residual {r['residual_Wm2']:+.4f} W/m2 "
              f"({r['residual_pct_of_qnet'] if r['residual_pct_of_qnet'] is None else round(r['residual_pct_of_qnet'], 3)} % of q_net)")
    worst = max(abs(r["residual_Wm2"]) for r in rows)
    out["max_residual_wm2_gate"] = a.max_residual_wm2
    out["worst_abs_residual_wm2"] = worst
    out["verdict"] = "PASS" if worst <= a.max_residual_wm2 else "FAIL"
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, indent=1))
    print(f"[closure] worst |residual| {worst:.4f} W/m2 vs gate "
          f"{a.max_residual_wm2} W/m2: {out['verdict']}")
    return 0 if out["verdict"] == "PASS" else 1


def _snap_lat(path):
    with np.load(path) as z:
        return np.asarray(z["lat_T"], dtype=np.float64)


def _snap_lon(path):
    with np.load(path) as z:
        return np.asarray(z["lon_T"], dtype=np.float64)


if __name__ == "__main__":
    sys.exit(main())
