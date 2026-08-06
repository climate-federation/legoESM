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
            "command": (f"python scripts/data/extract_gsam_rcemip_baseline.py "
                        f"--snd $LEGOESM_GSAM_ROOT/CASES/RCEMIP1/"
                        f"{src.name} --out {args.out} "
                        f"--n-levels {args.n_levels}"),
            "level_indices": idx,
        },
        "units": {
            "z": "m", "p": "mb", "theta": "K", "q_v": "kg/kg",
            "T": "K (derived: theta*(p/p_ref)^kappa)",
            "rh": ("dimensionless (derived: q_v / "
                   "legoesm.thermo.saturation_mixing_ratio(T, p); a MIXING-"
                   "RATIO ratio, matching SAM's q column convention)"),
        },
        "oracle_is": ["z", "p", "theta", "q_v"],
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
