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

# A latlon resolution at/below this is a COARSE STARTER (toy ≈ 8×16 grid) — fine for a
# smoke / dry-run but far too coarse for a meaningful ERA5 comparison. ``main`` reminds the
# operator to scale up, so a production campaign is not run at toy resolution by mistake
# (the runbook + sbatch quick-start both show ``--resolution 8``).
_STARTER_RESOLUTION_MAX = 16


def build_amip_clubb_lite_config(
    *, resolution: int = 8, nlev: int = 10, dt: float = 600.0, radiation: str = "gray",
    days: int = 200,
) -> ExperimentConfig:
    """A runnable AMIP :class:`ExperimentConfig` with ``turbulence="clubb_lite"``.

    ``turbulence`` is fixed to ``"clubb_lite"`` (the campaign's requirement); the
    grid (lat-lon), vertical resolution, timestep, radiation, and run length are
    exposed so the starter can be scaled toward a production run.  ``days`` is the
    run length, i.e. the CLIMATOLOGY WINDOW the model time-mean is computed over and
    compared to the (matched) ERA5 mean — keep it long enough for a stable mean (the
    200-day default ≈ 40 samples at the 5-day diagnostic cadence) and aligned to the
    ERA5 window (runbook §6). Hydrostatic finite-volume dycore — the AMIP default the
    comparison + LES spin-off were built against.
    """
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=int(resolution), nlev=int(nlev)),
        dycore=DycoreConfig(
            dt=float(dt), model_type="hydrostatic", discretization="finite_volume"),
        radiation=radiation,
        turbulence="clubb_lite",
        days=int(days),
    )
    # Fail-fast on a bad scheme literal (e.g. a typo'd --radiation like "rrtmpg") via the
    # CANONICAL ExperimentConfig validator BEFORE the starter config is written to disk —
    # else the bad value only surfaces later when the campaign LOADS the file (a worse
    # operator UX: a written-but-broken config). validate_strict checks radiation against
    # the same ("none","gray","rrtmgp","rrtmg") set the model build uses, so no duplication.
    cfg.validate_strict()
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", help="output path for the JSON config")
    p.add_argument("--resolution", type=int, default=8, help="lat-lon resolution (default 8)")
    p.add_argument("--nlev", type=int, default=10, help="vertical levels (default 10)")
    p.add_argument("--dt", type=float, default=600.0, help="dycore timestep [s] (default 600)")
    p.add_argument("--radiation", default="gray", help="radiation scheme (default gray)")
    p.add_argument("--days", type=int, default=200,
                   help="run length = the CLIMATOLOGY WINDOW the time-mean is computed over "
                        "(default 200; keep it long + aligned to the ERA5 window)")
    args = p.parse_args(argv)
    cfg = build_amip_clubb_lite_config(
        resolution=args.resolution, nlev=args.nlev, dt=args.dt, radiation=args.radiation,
        days=args.days)
    with open(args.out, "w") as f:
        json.dump(experiment_config_to_dict(cfg), f, indent=2)
    print(f"[config] wrote AMIP clubb_lite base config (turbulence=clubb_lite, "
          f"latlon {args.resolution} L{args.nlev}, {args.days}-day climatology) to {args.out}")
    if args.resolution <= _STARTER_RESOLUTION_MAX:
        print(
            f"[config] NOTE: latlon resolution {args.resolution} is a COARSE STARTER "
            "(fine for a smoke / dry-run). For a PRODUCTION ERA5 comparison scale "
            "--resolution / --nlev UP — and lower --dt to keep the CFL stable (a too-large "
            "dt at higher resolution diverges, and the campaign's baseline-divergence guard "
            "would then fail the run loud).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
