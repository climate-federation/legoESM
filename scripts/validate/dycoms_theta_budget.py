"""Close the potential-temperature budget of a stratocumulus LES run.

Answers "what is warming the boundary layer?" from the SAVED profile frames --
no rerun, no hot-loop instrumentation. The frames already carry the resolved
turbulent heat flux ``w'theta'`` (``les_record._profiles``), which is the
dominant term, so the only unknowns are the ones worth isolating.

Budget, all as boundary-layer-mean d(theta)/dt [K/h], z positive UP:

    d<theta>/dt  =  -d(w'theta')/dz        resolved turbulent flux divergence
                    - w_ls d<theta>/dz     large-scale subsidence
                    + Q_rad                parameterized longwave
                    + surface flux
                    + RESIDUAL             SGS flux divergence + latent
                                           heating + NUMERICAL mixing

RESIDUAL is the point. Every term to its left is measured from the frames or
known exactly from the case specification, so whatever is left is the part that
is NOT resolved physics. A large positive residual in a stratocumulus run that
is losing its cloud says the warming comes from sub-grid or numerical
transport, not from the resolved circulation -- a different bug, and a
different fix, from "the radiation is too weak".

Written because an inferred budget did NOT close: a DYCOMS RF01 run warmed at
+0.153 K/h while radiation cooled at -0.230 K/h, and attributing the rest to
surface flux and entrainment left ~0.2 K/h unexplained. Guessing which term
that was would have cost GPU-hours per guess.

Usage::

    python scripts/validate/dycoms_theta_budget.py results/les_ref/dycoms
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics._shared import exner_function

# Stevens et al. (2005) RF01 parameterized longwave + subsidence, matching
# run_dycoms_les.py. Named here with their source so this diagnostic cannot
# silently drift from the driver it audits.
_KAPPA_RAD = 85.0        # LW absorption [m^2/kg]
_F0 = 70.0               # cloud-top jump [W/m^2]
_F1 = 22.0               # cloud-base jump [W/m^2]
_DIV = 3.75e-6           # large-scale divergence D [1/s]; w_ls = -D z
_A_RAD = 1.0
_QT_INV = 8.0e-3         # q_t isoline defining z_i [kg/kg]
_P_SFC_PA = 1.0178e5     # DYCOMS_RF01 deck pres0
_SCALE_HEIGHT_M = 8.5e3  # only sets the reference density profile


def _reference_column(z):
    """Reference pressure, density and Exner on the LES height grid."""
    p = _P_SFC_PA * np.exp(-np.asarray(z) / _SCALE_HEIGHT_M)
    exner = np.asarray(exner_function(p), dtype=np.float64)
    T = 289.0 * exner                      # RF01 mixed-layer theta ~ 289 K
    rho = p / (constants.R_d * T)
    return p, rho, exner


def _stevens_lw(z, qc, qv, rho, exner):
    """theta tendency [K/s] from the RF01 parameterized longwave, and z_i."""
    cp = constants.c_pd
    dz = float(z[1] - z[0])
    dq = _KAPPA_RAD * rho * qc * dz
    Q_bot = np.concatenate([[0.0], np.cumsum(dq)])
    Q_top = Q_bot[-1] - Q_bot
    q_t = qv + qc
    idx = np.where(q_t >= _QT_INV)[0]
    z_i = (z[idx.max()] + 0.5 * dz) if idx.size else 0.0
    z_f = np.concatenate([z - 0.5 * dz, [z[-1] + 0.5 * dz]])
    dz_i = np.clip(z_f - z_i, 0.0, None)
    term3 = (_A_RAD * np.interp(z_i, z, rho) * cp * _DIV
             * (0.25 * dz_i ** (4.0 / 3.0) + z_i * dz_i ** (1.0 / 3.0)))
    F = _F0 * np.exp(-Q_top) + _F1 * np.exp(-Q_bot) + term3
    return -(np.diff(F) / dz) / (rho * cp * exner), z_i


def _ddz(field, z):
    """Centred d/dz on the LES grid, one-sided at the ends."""
    return np.gradient(np.asarray(field, dtype=np.float64),
                       np.asarray(z, dtype=np.float64))


def budget(les_dir: Path, *, bl_top_frac: float = 0.9, shf_w_m2: float = 15.0):
    files = sorted(glob.glob(str(Path(les_dir) / "profiles" / "prof_*.npz")))
    if len(files) < 2:
        raise SystemExit(f"need >= 2 frames in {les_dir}/profiles")
    frames = []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            missing = {"z", "theta", "qc", "qv", "wth"} - set(d.files)
            if missing:
                raise SystemExit(
                    f"{Path(f).name} lacks {sorted(missing)} -- the frames "
                    "predate the turbulent-flux recording in les_record.py")
            frames.append({k: np.asarray(d[k]) for k in d.files})

    z = frames[0]["z"]
    for fr in frames[1:]:
        if fr["z"].shape != z.shape or not np.allclose(fr["z"], z, atol=1e-6):
            raise SystemExit("frames are on different vertical grids")
    _p, rho, exner = _reference_column(z)
    cp = constants.c_pd

    rows = []
    for a, b in zip(frames[:-1], frames[1:]):
        dt_h = float(b["t_hours"]) - float(a["t_hours"])
        if dt_h <= 0:
            continue
        # Terms are evaluated at the interval MIDPOINT, which is what a centred
        # difference of the state is consistent with; either endpoint biases it.
        mid = {k: 0.5 * (a[k] + b[k]) for k in ("theta", "qc", "qv", "wth")}
        rad, z_i = _stevens_lw(z, mid["qc"], mid["qv"], rho, exner)
        bl = (z <= bl_top_frac * z_i) if z_i > 0 else (z <= z.max())
        if not bl.any():
            continue

        total = (b["theta"] - a["theta"]) / (dt_h * 3600.0)
        turb = -_ddz(mid["wth"], z)                 # -d(w'theta')/dz
        subs = -(-_DIV * z) * _ddz(mid["theta"], z)  # -w_ls dtheta/dz
        h = max(float(z[bl].max()), 1.0)
        sfc = shf_w_m2 / (float(rho[0]) * cp * h)

        def m(x):
            return float(np.mean(np.asarray(x)[bl])) * 3600.0

        t_tot, t_turb, t_subs, t_rad = m(total), m(turb), m(subs), m(rad)
        t_sfc = sfc * 3600.0
        rows.append(dict(t=float(b["t_hours"]), z_i=z_i, total=t_tot,
                         turb=t_turb, subs=t_subs, rad=t_rad, sfc=t_sfc,
                         resid=t_tot - (t_turb + t_subs + t_rad + t_sfc)))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("les_dir", type=Path)
    p.add_argument("--shf", type=float, default=15.0,
                   help="prescribed surface sensible heat flux [W/m2]")
    p.add_argument("--bl-top-frac", type=float, default=0.9)
    args = p.parse_args(argv)

    rows = budget(args.les_dir, bl_top_frac=args.bl_top_frac, shf_w_m2=args.shf)
    print(f"theta budget, BL mean (z <= {args.bl_top_frac:g} z_i), K/h "
          f"-- {args.les_dir}")
    print(f"{'t[h]':>5} {'z_i':>5} {'total':>8} {'turb':>8} {'subs':>8} "
          f"{'rad':>8} {'sfc':>7} {'RESID':>8}")
    for r in rows:
        print(f"{r['t']:5.2f} {r['z_i']:5.0f} {r['total']:+8.3f} "
              f"{r['turb']:+8.3f} {r['subs']:+8.3f} {r['rad']:+8.3f} "
              f"{r['sfc']:+7.3f} {r['resid']:+8.3f}")

    tot = float(np.mean([r["total"] for r in rows]))
    res = float(np.mean([r["resid"] for r in rows]))
    print(f"\nrun mean: total {tot:+.3f} K/h, residual {res:+.3f} K/h "
          f"({100.0 * abs(res) / max(abs(tot), 1e-9):.0f}% of the total)")
    print("=> " + ("the budget does NOT close on resolved physics: most of "
                   "the warming is SGS, latent, or NUMERICAL transport."
                   if abs(res) > 0.5 * abs(tot) else
                   "the budget closes on resolved physics + radiation."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
