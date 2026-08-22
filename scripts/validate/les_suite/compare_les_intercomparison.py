#!/usr/bin/env python
"""Gate-0/gate-1 validator: LES CBL output vs the published convective envelope.

Reads a finished spectral-CBL run's ``cbl_profiles.npz`` (written by
``scripts/run/run_spectral_cbl.py``) and scores its horizontal-mean profiles
against the universal dry-CBL convective-scaling envelope (Nieuwstadt et al. 1993 +
entrainment-ratio literature) via
``legoesm.atmosphere.les_suite.intercomparison``. Exits non-zero if any band fails
— this is the LES_SUITE.md D7 credibility gate, blocking downstream tuning trust.

Usage::

    python scripts/validate/les_suite/compare_les_intercomparison.py \\
        results/les_suite/gate0_nieuwstadt/cbl_profiles.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from legoesm.atmosphere.les_suite.intercomparison import gate_from_cbl_profiles


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("profiles", type=Path, help="cbl_profiles.npz from run_spectral_cbl")
    p.add_argument("--theta0", type=float, default=300.0,
                   help="reference potential temperature [K] for w_* (default 300)")
    args = p.parse_args(argv)

    if not args.profiles.exists():
        print(f"error: {args.profiles} does not exist", file=sys.stderr)
        return 2
    data = np.load(args.profiles)
    for key in ("z", "theta", "ww", "wth", "Q0"):
        if key not in data.files:
            print(f"error: {args.profiles} missing key {key!r} "
                  f"(has {data.files})", file=sys.stderr)
            return 2

    z_i = float(data["zi"]) if "zi" in data.files else None
    gate = gate_from_cbl_profiles(
        data["z"], data["theta"], data["ww"], data["wth"],
        Q0=float(data["Q0"]), theta0=args.theta0, z_i_m=z_i,
    )
    print(gate.report())
    return 0 if gate.passed else 1


if __name__ == "__main__":
    sys.exit(main())
