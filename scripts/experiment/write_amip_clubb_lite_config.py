"""Write a STARTER AMIP base config for the LES-informed correction campaign.

The campaign launcher (``scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch``)
and CLI (``scripts/run/run_correction_campaign.py --config <path>``) need a base
:class:`~legoesm.driver.config.ExperimentConfig` whose ``turbulence`` is
``"clubb_lite"`` (the campaign corrects ``clubb_lite``'s C_K / Pr_t / C_eps — it
does NOT switch schemes).  This emits a minimal, runnable such config as JSON.

The config is built PROGRAMMATICALLY (never a hand-written JSON that could drift
from the schema): :func:`build_amip_clubb_lite_config` always produces a
schema-valid ``ExperimentConfig`` with ``turbulence="clubb_lite"``, and the CLI
serializes it via ``experiment_config_to_dict``.  The defaults are a COARSE
starter (latlon 8, 10 levels, gray radiation) — scale ``--resolution`` / ``--nlev``
/ ``--dt`` up for a production demonstration.

Usage::

    python scripts/experiment/write_amip_clubb_lite_config.py configs/amip_clubb_lite.json
    python scripts/experiment/write_amip_clubb_lite_config.py out.json --resolution 16 --nlev 30
"""

from __future__ import annotations

import argparse
import json

from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    experiment_config_to_dict,
)


def build_amip_clubb_lite_config(
    *, resolution: int = 8, nlev: int = 10, dt: float = 600.0, radiation: str = "gray"
) -> ExperimentConfig:
    """A runnable AMIP :class:`ExperimentConfig` with ``turbulence="clubb_lite"``.

    ``turbulence`` is fixed to ``"clubb_lite"`` (the campaign's requirement); the
    grid (lat-lon), vertical resolution, timestep, and radiation are exposed so the
    starter can be scaled toward a production run.  Hydrostatic finite-volume dycore
    — the AMIP default the comparison + LES spin-off were built against.
    """
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=int(resolution), nlev=int(nlev)),
        dycore=DycoreConfig(
            dt=float(dt), model_type="hydrostatic", discretization="finite_volume"),
        radiation=radiation,
        turbulence="clubb_lite",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", help="output path for the JSON config")
    p.add_argument("--resolution", type=int, default=8, help="lat-lon resolution (default 8)")
    p.add_argument("--nlev", type=int, default=10, help="vertical levels (default 10)")
    p.add_argument("--dt", type=float, default=600.0, help="dycore timestep [s] (default 600)")
    p.add_argument("--radiation", default="gray", help="radiation scheme (default gray)")
    args = p.parse_args(argv)
    cfg = build_amip_clubb_lite_config(
        resolution=args.resolution, nlev=args.nlev, dt=args.dt, radiation=args.radiation)
    with open(args.out, "w") as f:
        json.dump(experiment_config_to_dict(cfg), f, indent=2)
    print(f"[config] wrote AMIP clubb_lite base config (turbulence=clubb_lite, "
          f"latlon {args.resolution} L{args.nlev}) to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
