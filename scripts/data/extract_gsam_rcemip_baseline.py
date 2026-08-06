"""Extract the TINY tracked baseline from gSAM's RCEMIP1 RCE300 sounding.

Why this script exists
----------------------
The gSAM 1.8.8 tree is an EXTERNAL reference source and must not be committed
(``CLAUDE.md``: "Never commit ``docs/references/``" — reference sources are
cited, not vendored).  But an oracle test needs something to compare against on
a machine that has no gSAM checkout.  CLAUDE.md's one carve-out is the
visual-regression precedent: *tiny numeric baselines ARE tracked*.

So this script distils the 74-level sounding down to ~15 levels and writes
``tests/oracle_baselines/gsam_rcemip300_snd.json`` with FULL provenance (upstream
URL, file name, SHA256 of the whole source file, the level indices taken, and
the command that produced it).  The heavy source stays external; the test that
consumes the baseline can additionally open the real file when
``LEGOESM_GSAM_ROOT`` is set and check all 74 levels.

Source
------
gSAM 1.8.8, ``http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz``,
``CASES/RCEMIP1/snd_rcemip_300s6.11.2`` (``CASES/RCEMIP1/snd`` is a
byte-identical copy — gSAM's RCEMIP1 deck runs the 300 K case by default).
The deck also ships ``snd_rcemip_295s6.11.2`` and ``snd_rcemip_305s6.11.2``, so
gSAM uses a DIFFERENT sounding per SST rather than one profile across cases.

Format: a text column header, then per time block ``day nlev p_sfc[mb]``
followed by ``nlev`` rows of ``z[m] p[mb] theta[K] q[g/kg] u[m/s] v[m/s]``.
The file has two identical blocks (day 0 and day 1000); ``read_sam_snd`` uses
the first, and so does this script (via the same reader — no second parser).

Usage
-----
.. code-block:: bash

   python scripts/data/extract_gsam_rcemip_baseline.py \\
       --snd $LEGOESM_GSAM_ROOT/CASES/RCEMIP1/snd_rcemip_300s6.11.2 \\
       --out tests/oracle_baselines/gsam_rcemip300_snd.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

UPSTREAM_URL = "http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz"
UPSTREAM_MEMBER = "gSAM1.8.8/CASES/RCEMIP1/snd_rcemip_300s6.11.2"


def select_levels(z: np.ndarray, rh: np.ndarray, T: np.ndarray,
                  n_target: int = 15) -> list[int]:
    """Level indices for the baseline: a height spread PLUS the two levels the
    acceptance criteria name explicitly (max RH, cold point) and the endpoints.

    Picking purely by index spread can miss the max-RH level, which is one of
    the numbers the oracle test asserts — so those extrema are added by name
    rather than hoped for.
    """
    n = z.size
    idx = set(np.linspace(0, n - 1, n_target - 2).astype(int).tolist())
    idx.add(0)                      # surface level (z = 37 m acceptance point)
    idx.add(n - 1)                  # model-top level
    idx.add(int(np.argmax(rh)))     # max-RH level
    idx.add(int(np.argmin(T)))      # tropopause cold point
    return sorted(idx)


def calibrate_wing_constants(z, T_v, q_v, z_t, z_q1, z_q2):
    """Fit the Wing analytic constants ``(T_v0, Gamma, q_sfc)`` to a sounding.

    Returns a dict with TWO estimators for the temperature pair, because they
    disagree in a way that matters and the choice must be visible:

    ``lstsq``
        Least squares on ``T_v(z)`` over ``z <= z_t``.  Minimises the mean
        tropospheric residual — and therefore misses BOTH endpoints, because
        gSAM's lapse is not constant (it is steeper in the well-sampled lower
        troposphere).  Chasing this estimator moves the analytic tropopause
        5 K below the oracle's cold point.
    ``endpoints``
        The straight line through the back-extrapolated surface ``T_v(0)`` and
        the oracle's COLD POINT.  Both are quantities the oracle gate asserts
        (surface T/RH, tropopause temperature) and both are physically
        load-bearing — the surface sets the initial saturation, the cold point
        sets OLR, cirrus and CAPE.  This is the estimator that ships.

    ``q_sfc`` is the surface level extrapolated to z=0 through the Wing shape
    function, so the sounding's lowest level is reproduced exactly.  A
    least-squares ``q_0`` over the troposphere is reported alongside; they
    differ by ~4 % because gSAM's moisture is not exactly the Wing shape.
    """
    z = np.asarray(z, dtype=np.float64)
    T_v = np.asarray(T_v, dtype=np.float64)
    q_v = np.asarray(q_v, dtype=np.float64)
    trop = z <= z_t
    if trop.sum() < 5:
        raise ValueError("calibrate_wing_constants: too few tropospheric levels")

    shape = np.exp(-z / z_q1) * np.exp(-((z / z_q2) ** 2))
    q0_point = float(q_v[0] / shape[0])
    q0_lstsq = float(np.dot(q_v[trop], shape[trop])
                     / np.dot(shape[trop], shape[trop]))

    A = np.stack([np.ones(int(trop.sum())), z[trop]], axis=1)
    coef, *_ = np.linalg.lstsq(A, T_v[trop], rcond=None)
    Tv0_ls, slope_ls = float(coef[0]), float(coef[1])
    fit = A @ coef
    r2 = 1.0 - float(np.sum((T_v[trop] - fit) ** 2)
                     / np.sum((T_v[trop] - T_v[trop].mean()) ** 2))

    # Endpoint-constrained: back-extrapolate the surface with the local lapse
    # measured over the lowest ~2 km, then pin the tropopause value at z_t to
    # the oracle's cold point so T_v0 - Gamma*z_t reproduces it exactly.
    near = z <= 2000.0
    A2 = np.stack([np.ones(int(near.sum())), z[near]], axis=1)
    c2, *_ = np.linalg.lstsq(A2, T_v[near], rcond=None)
    Tv0_end = float(c2[0])
    k_cold = int(np.argmin(T_v))
    T_cold = float(T_v[k_cold])
    gamma_end = (Tv0_end - T_cold) / float(z_t)

    def _mae(T_v0, gamma):
        return float(np.abs((T_v0 - gamma * z[trop]) - T_v[trop]).mean())

    return {
        "q_sfc": q0_point,
        "q_sfc_lstsq": q0_lstsq,
        "lstsq": {"T_v0": Tv0_ls, "Gamma": -slope_ls, "r2": r2,
                  "trop_Tv_mae": _mae(Tv0_ls, -slope_ls),
                  "cold_point_err": Tv0_ls + slope_ls * z_t - T_cold},
        "endpoints": {"T_v0": Tv0_end, "Gamma": gamma_end,
                      "trop_Tv_mae": _mae(Tv0_end, gamma_end),
                      "cold_point_err": Tv0_end - gamma_end * z_t - T_cold},
        "oracle_cold_point_T_v": T_cold,
        "oracle_cold_point_z": float(z[k_cold]),
        "n_trop_levels": int(trop.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snd", required=True,
                   help="path to gSAM CASES/RCEMIP1/snd_rcemip_300s6.11.2")
    p.add_argument("--out", required=True, help="output JSON baseline path")
    p.add_argument("--n-levels", type=int, default=15,
                   help="approximate number of levels to vendor (default 15)")
    args = p.parse_args(argv)

    from legoesm import constants
    from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_snd
    from legoesm.thermo import saturation_mixing_ratio

    src = Path(args.snd)
    raw = src.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()

    snd = read_sam_snd(src)
    z = np.asarray(snd.z, dtype=np.float64)
    p_mb = np.asarray(snd.p, dtype=np.float64)
    theta = np.asarray(snd.theta, dtype=np.float64)
    q_v = np.asarray(snd.q_v, dtype=np.float64)      # kg/kg (reader: g/kg/1000)
    if not np.all(np.isfinite(np.stack([z, p_mb, theta, q_v]))):
        raise SystemExit(f"{src}: non-finite sounding values — refusing to "
                         "vendor a baseline from it.")

    # Derived diagnostics, computed with the MODEL's own thermodynamics
    # (legoesm.constants + legoesm.thermo — never a re-derived Tetens fit).
    p_pa = p_mb * 100.0
    T = theta * (p_pa / constants.p_ref) ** constants.kappa
    rh = q_v / np.asarray(saturation_mixing_ratio(T, p_pa), dtype=np.float64)

    # Calibration lives HERE, in a tracked and unit-tested script, rather than
    # in a throwaway probe: two shipped physical constants depend on it, so the
    # fit has to be reviewable and reproducible from the repo.
    from legoesm.atmosphere.idealized import rcemip_initial_conditions as _ic
    T_v = T * (1.0 + (1.0 / constants.epsilon - 1.0) * q_v)
    calib = calibrate_wing_constants(z, T_v, q_v, _ic.WING_Z_T,
                                     _ic.WING_Z_Q1, _ic.WING_Z_Q2)

    idx = select_levels(z, rh, T, n_target=args.n_levels)
    payload = {
        "_comment": (
            "TINY tracked oracle baseline — a ~15-level distillation of gSAM "
            "1.8.8's RCEMIP1 RCE300 sounding. The gSAM tree itself is external "
            "and NOT committed; this is the visual-regression 'tiny numeric "
            "baseline' carve-out. Regenerate with the command in "
            "'provenance.command'."),
        "provenance": {
            "upstream_url": UPSTREAM_URL,
            "upstream_member": UPSTREAM_MEMBER,
            "source_sha256": sha,
            "source_bytes": len(raw),
            "source_n_levels": int(z.size),
            "source_block": "first time block (day 0); the file holds two "
                            "identical blocks (day 0 and day 1000)",
            "reader": "legoesm.atmosphere.forcing.sam_case_forcing.read_sam_snd",
            "extraction_script": "scripts/data/extract_gsam_rcemip_baseline.py",
            # A reproduction command must be RUNNABLE by the next person, so it
            # names the tracked destination rather than whatever scratch path
            # this invocation happened to use.
            "command": (f"python scripts/data/extract_gsam_rcemip_baseline.py "
                        f"--snd $LEGOESM_GSAM_ROOT/CASES/RCEMIP1/{src.name} "
                        f"--out tests/oracle_baselines/gsam_rcemip300_snd.json "
                        f"--n-levels {args.n_levels}"),
            "written_to": str(args.out),
            "level_indices": idx,
        },
        "units": {
            "z": "m", "p": "mb", "theta": "K", "q_v": "kg/kg",
            "T": "K (derived: theta*(p/p_ref)^kappa)",
            "rh": ("dimensionless (derived: q_v / "
                   "legoesm.thermo.saturation_mixing_ratio(T, p); a MIXING-"
                   "RATIO ratio, matching SAM's q column convention)"),
        },
        "wing_calibration": calib,
        # The deck's own header surface pressure [mb] — what the driver's
        # --sounding p_sfc guard compares against. Vendored rather than
        # assumed equal to WING_P_SFC.
        "pres0_mb": float(snd.pres0),
        "oracle_is": ["z", "p", "theta", "q_v", "pres0_mb"],
        "derived_by_legoesm_thermo": ["T", "rh"],
        "z": [float(z[k]) for k in idx],
        "p": [float(p_mb[k]) for k in idx],
        "theta": [float(theta[k]) for k in idx],
        "q_v": [float(q_v[k]) for k in idx],
        "T": [float(T[k]) for k in idx],
        "rh": [float(rh[k]) for k in idx],
        "full_column_summary": {
            "z_min": float(z.min()), "z_max": float(z.max()),
            "T_surface": float(T[0]), "q_v_surface": float(q_v[0]),
            "rh_surface": float(rh[0]),
            "rh_max": float(rh.max()),
            "rh_max_z": float(z[int(np.argmax(rh))]),
            "T_cold_point": float(T.min()),
            "T_cold_point_z": float(z[int(np.argmin(T))]),
            "T_top": float(T[-1]),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out} — {len(idx)} of {z.size} levels")
    print(f"  source sha256 {sha}")
    print(f"  surface: z={z[0]:.1f} m T={T[0]:.3f} K q={q_v[0]*1e3:.4f} g/kg "
          f"RH={rh[0]:.4f}")
    print(f"  max RH {rh.max():.4f} at z={z[int(np.argmax(rh))]:.0f} m; "
          f"cold point {T.min():.2f} K at z={z[int(np.argmin(T))]:.0f} m")
    print("\n--- Wing analytic calibration ---")
    print(f"  q_sfc  point-extrap {calib['q_sfc']*1e3:.4f} g/kg   "
          f"lstsq {calib['q_sfc_lstsq']*1e3:.4f} g/kg")
    for name in ("lstsq", "endpoints"):
        c = calib[name]
        print(f"  {name:10s} T_v0={c['T_v0']:.3f} K  "
              f"Gamma={c['Gamma']*1e3:.4f} K/km  "
              f"trop |dT_v| mean={c['trop_Tv_mae']:.3f} K  "
              f"cold-point err={c['cold_point_err']:+.3f} K")
    print(f"  oracle cold point T_v={calib['oracle_cold_point_T_v']:.3f} K "
          f"at z={calib['oracle_cold_point_z']:.0f} m "
          f"({calib['n_trop_levels']} tropospheric levels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
