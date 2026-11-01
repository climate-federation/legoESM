#!/usr/bin/env python
"""Build the WeatherBench model-grid state cache for one campaign deck.

Every ERA5 analysis the deck's training samples and its validation inits need
is converted to the model grid ONCE (``scale_build.wb_state_record``) and
written as float32 to ``state_cache_dir`` (``legoesm.training.wb_state_cache``).

Run under SLURM, never on a login node:
  --init            create the empty store (one job, before the array);
  --task K --n-tasks N   write every N-th time starting at K (array tasks);
  --check           print how many needed times are written (exit 1 if any
                    is missing).
Already-written times are skipped, so a killed task is simply re-run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[2]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--mode", required=True,
                   choices=("physics", "neural_gcm", "sfno"))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--init", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--task", type=int)
    p.add_argument("--n-tasks", type=int, default=1, dest="n_tasks")
    return p


def needed_indices(cfg, yml, times):
    """Sorted ERA5 indices of every training sample's and validation init's
    initial condition and supervised leads."""
    import numpy as np

    from legoesm.training import scale_build as sb

    cadence = int(yml.get("era5_cadence_hours", 6))
    _, stride, step = sb.wb_lead_strides(cfg, yml, cadence)
    decks = [yml]
    if yml.get("validation"):
        decks.append(sb.wb_validation_yaml(yml, int(yml["validation"]["n_inits"])))
    out = set()
    for d in decks:
        for _, a, b in sb._training_sample_indices(cfg, d, times, 24 // cadence,
                                                   stride):
            out.update([a, *range(a + step, b + 1, step)])
    return np.asarray(sorted(out), dtype=np.int64)


def main(argv=None):
    a = build_parser().parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    import yaml

    from legoesm.training import scale_build as sb
    from legoesm.training import wb_state_cache as wsc
    from legoesm.training.era5_to_state import open_era5_zarr

    yml = yaml.safe_load(open(a.config))
    sb.validate_wb_campaign_yaml(yml)
    if sb.wb_state_cache_root(yml) is None:
        raise SystemExit(f"{a.config} names no state_cache_dir")
    cfg = SimpleNamespace(mode=a.mode, training_core="spectral", smoke=False,
                          windows=None, n_days=None, multi_step_hours=None)
    _, grid, sigma, *_ = sb.build_mode_components(cfg, yml)
    era5_cfg, sst_cfg, micro = sb.wb_state_cache_configs(cfg, yml)
    ds = open_era5_zarr(era5_cfg.zarr_store)
    cloud_ds = (open_era5_zarr(era5_cfg.cloud_zarr)
                if era5_cfg.load_cloud_condensate else None)
    flux_ds = ((open_era5_zarr(sst_cfg.flux_zarr) if sst_cfg.flux_zarr else ds)
               if sst_cfg.load_surface_fluxes else None)
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    idx = needed_indices(cfg, yml, times)
    path = sb.wb_state_cache_path(cfg, yml, grid, sigma)
    print(f"state cache {path}: {len(idx)} ERA5 times "
          f"({times[idx[0]]} .. {times[idx[-1]]})", flush=True)

    def record(i):
        return sb.wb_state_record(int(i), era5_cfg, sst_cfg, ds, cloud_ds,
                                  flux_ds, grid, sigma, micro, times)

    if a.init:
        arrays, structure = wsc.record_arrays(*record(idx[0]))
        # Arms whose records are identical share one store (same tag).
        wsc.create_store(path, idx, arrays, structure, {})
        print("created", flush=True)
        return 0
    if a.check:
        missing = wsc.StateCache(path, record(idx[0])).missing(idx)
        print(f"written {len(idx) - len(missing)}/{len(idx)}", flush=True)
        return 1 if missing else 0
    if not 0 <= a.task < a.n_tasks:
        raise SystemExit(f"--task {a.task} outside 0..{a.n_tasks - 1}")
    import zarr
    done = np.asarray(zarr.open_group(str(path), mode="r")["done"])
    for k in range(a.task, len(idx), a.n_tasks):
        if done[k]:
            continue
        arrays, _ = wsc.record_arrays(*record(idx[k]))
        wsc.write_time(path, idx[k], arrays)
        print(f"wrote {k + 1}/{len(idx)} {times[idx[k]]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(_REPO))
    raise SystemExit(main())
