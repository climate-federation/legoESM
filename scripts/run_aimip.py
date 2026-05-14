#!/usr/bin/env python
"""AIMIP (AI Model Intercomparison Project) simulation.

AIMIP differs from AMIP in two main ways:
1. **Initial conditions**: ERA5 reanalysis at a specific date (default
   2020-01-01) instead of an idealised analytic state.
2. **Atmospheric physics parameters**: trained on 5-yr ERA5 (2015-2019)
   via :mod:`scripts.train_aimip_physics_params` and loaded at runtime
   from ``--params`` (or left at defaults for an untuned baseline).

Both ICs and SST/SIC come from ERA5 directly — no external PCMDI SST
file required.  Radiation is RRTMGP by default with era-appropriate
GHG concentrations (CMIP6 historical trajectory; auto-set per the
``aimip`` template in :mod:`legoesm.forcing.experiments`).

Usage
-----

Forward integration (Phase 1; untuned baseline):

.. code-block:: bash

    JAX_ENABLE_X64=1 python scripts/run_aimip.py \\
        --resolution 72x144 --grid-type latlon --discretization latlon_cgrid \\
        --days 1 --ic-year 2020-01-01 --ic-path /path/to/era5.zarr

After training (Phase 3 produces ``params_best.eqx``):

.. code-block:: bash

    JAX_ENABLE_X64=1 python scripts/run_aimip.py \\
        --resolution 72x144 --grid-type latlon --discretization latlon_cgrid \\
        --days 365 --ic-year 2020-01-01 --ic-path /path/to/era5.zarr \\
        --params outputs/aimip/params_best.eqx

This script is a thin wrapper that re-uses the ``scripts.run_amip``
CLI + ``legoesm.driver.ModelDriver`` machinery.  AIMIP-specific
defaults (radiation=rrtmgp, ic=era5, experiment=aimip, GHG=2015-2020
mean) are applied before delegating to ``run_amip.main``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``scripts.run_amip`` importable when this file is invoked
# directly via ``python scripts/run_aimip.py``.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from legoesm.forcing.experiments import EXPERIMENT_TEMPLATES, ghg_at_year


def _aimip_default_ghg(year: float) -> tuple[float, float, float]:
    """Return CMIP6 historical GHG concentrations interpolated to ``year``.

    Falls back to the 2015-2020 mean (~410 ppm CO2, ~1880 ppb CH4,
    ~333 ppb N2O) when ``year`` is None or invalid.
    """
    if year is None:
        # 2015-2020 midpoint = 2017.5
        return ghg_at_year("aimip", 2017.5)
    try:
        return ghg_at_year("aimip", float(year))
    except (ValueError, TypeError):
        return ghg_at_year("aimip", 2017.5)


def main(argv: list[str] | None = None) -> None:
    # First parse just the AIMIP-specific args so we can compute
    # defaults for the AMIP-CLI args.
    aimip_parser = argparse.ArgumentParser(
        add_help=False,
        description="AIMIP-specific arguments (delegated to run_amip otherwise)",
    )
    aimip_parser.add_argument(
        "--ic-year", type=str, default="2020-01-01",
        help="ERA5 IC date as YYYY-MM-DD (default 2020-01-01 — start of test year).",
    )
    aimip_parser.add_argument(
        "--params", type=str, default="",
        help="Path to trained TrainablePhysicsParams .eqx file.  "
             "Empty = untuned baseline (Phase 1).",
    )
    aimip_args, remaining = aimip_parser.parse_known_args(argv)

    # Compute era-appropriate GHGs from the IC year so the user
    # doesn't have to override --co2/--ch4/--n2o manually.
    try:
        year_float = float(aimip_args.ic_year.split("-")[0])
    except (ValueError, AttributeError, IndexError):
        year_float = 2020.0
    co2, ch4, n2o = _aimip_default_ghg(year_float)

    # Build the args list we'll pass to run_amip.  AIMIP defaults
    # are inserted IF the user hasn't already provided them in
    # ``remaining``.  This lets the user override on the CLI when
    # needed (e.g. --co2-ppmv 350 for a sensitivity probe).
    def _has(flag: str) -> bool:
        return any(arg == flag or arg.startswith(flag + "=") for arg in remaining)

    aimip_defaults: list[str] = []
    if not _has("--dataset"):
        aimip_defaults.extend(["--dataset", "analytical"])
    if not _has("--ic"):
        aimip_defaults.extend(["--ic", "era5"])
    if not _has("--radiation"):
        aimip_defaults.extend(["--radiation", "rrtmgp"])
    if not _has("--experiment"):
        aimip_defaults.extend(["--experiment", "aimip"])
    if not _has("--co2-ppmv"):
        aimip_defaults.extend(["--co2-ppmv", f"{co2:.3f}"])
    if not _has("--ch4-ppbv"):
        aimip_defaults.extend(["--ch4-ppbv", f"{ch4:.1f}"])
    if not _has("--n2o-ppbv"):
        aimip_defaults.extend(["--n2o-ppbv", f"{n2o:.2f}"])

    # Default grid: latlon C-grid per AIMIP plan decision (Phase 1).
    if not _has("--grid-type"):
        aimip_defaults.extend(["--grid-type", "latlon"])
    if not _has("--discretization"):
        aimip_defaults.extend(["--discretization", "latlon_cgrid"])

    full_argv = aimip_defaults + remaining
    print(f"[aimip] effective AIMIP defaults: {aimip_defaults}")
    print(f"[aimip] IC year={aimip_args.ic_year}; "
          f"era-GHG: CO2={co2:.2f} CH4={ch4:.0f} N2O={n2o:.2f}")
    if aimip_args.params:
        print(f"[aimip] loading trained physics params from {aimip_args.params}")
        print("[aimip] NOTE: physics-param loading is a Phase-3 hookup "
              "(not yet wired; this flag is a placeholder).")

    # ``run_amip`` lives in the same scripts/ directory which we added
    # to sys.path at module load time.
    from run_amip import main as run_amip_main  # type: ignore[import-not-found]
    run_amip_main(full_argv)


if __name__ == "__main__":
    main(sys.argv[1:])
