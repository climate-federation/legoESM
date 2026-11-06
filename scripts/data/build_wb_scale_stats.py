#!/usr/bin/env python
"""Loss and near-surface-head scales from a WeatherBench model-grid state cache.

Owner decisions 2026-11-01/02: every scale is an area-weighted statistic over
the cached ERA5 analyses of 2015-2019 only (the training years), on the model
grid, written next to the cache as ``<store>_scales.json``; the decks copy the
numbers so every run names its scales explicitly.

* loss: std of the 6 h change, per model level for T, u, v, q and once for
  p_s and the ERA5 t2m, u10, v10, mslp targets (pairs of cached analyses
  exactly 6 h apart, both inside the years);
* head inputs: mean and std of every input (lowest three levels of T, u, v, q,
  p_s, land fraction, surface geopotential, sin/cos latitude and sin/cos
  local time; no soil moisture, owner 2026-11-02);
* head outputs: std of the gap ERA5 minus the eval's current estimate
  (``diagnose_headline_fields``) for t2m, u10, v10.

Run under SLURM (one task per store), never on a login node.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
YEARS = (2015, 2019)
N_HEAD_LEVELS = 3


class Moments:
    """Area-weighted running mean/std, float64, per trailing ``shape``."""

    def __init__(self):
        self.w = self.s1 = self.s2 = 0.0

    def add(self, x, w):
        """``x`` (n_lat, n_lon[, k]); ``w`` weights broadcastable to it."""
        x = np.asarray(x, np.float64)
        w = np.broadcast_to(np.asarray(w, np.float64), x.shape)
        axes = (0, 1)
        self.w = self.w + w.sum(axis=axes)
        self.s1 = self.s1 + (w * x).sum(axis=axes)
        self.s2 = self.s2 + (w * x * x).sum(axis=axes)

    def result(self):
        mean = self.s1 / self.w
        var = np.maximum(self.s2 / self.w - mean ** 2, 0.0)
        return mean, np.sqrt(var)


def _area(grid):
    """(n_lat, n_lon) Gaussian quadrature weights per cell."""
    w = np.asarray(grid.weights, np.float64)
    return np.broadcast_to(w[:, None], (w.size, int(grid.n_lon)))


def accumulate(records, pairs, grid, diagnose):
    """Every statistic from ``records(i) -> (carry, sfc, forcing)`` over the
    analyses ``i`` in ``pairs`` (``(i, j)`` = 6 h apart; ``j`` None when ``i``
    is in the years but its successor is not)."""
    a = _area(grid)
    n_lat, n_lon = a.shape
    lat = np.asarray(grid.lat, np.float64)[:, None]
    lon = np.asarray(grid.lon, np.float64)[None, :]
    lev = {k: Moments() for k in ("T", "u", "v", "q_v")}
    one = {k: Moments() for k in ("p_s", "t2m", "u10", "v10", "mslp")}
    head = {k: Moments() for k in (
        "T", "u", "v", "q_v", "p_s", "land_frac", "phis",
        "sin_lat", "cos_lat", "sin_local_time", "cos_local_time")}
    gap = {k: Moments() for k in ("t2m", "u10", "v10")}
    a3 = a[:, :, None]
    prev = None
    for i, j in pairs:
        cur = prev if prev is not None and prev[0] == i else (i, records(i))
        c, s, f = cur[1]
        for k in ("T", "u", "v", "q_v"):
            head[k].add(np.asarray(getattr(c, k))[..., -N_HEAD_LEVELS:], a3)
        head["p_s"].add(c.p_s, a)
        head["phis"].add(c.phis, a)
        land = np.asarray(f["land_frac"]).reshape(n_lat, n_lon)
        head["land_frac"].add(land, a)
        head["sin_lat"].add(np.broadcast_to(np.sin(lat), a.shape), a)
        head["cos_lat"].add(np.broadcast_to(np.cos(lat), a.shape), a)
        utc = 2.0 * np.pi * float(f["seconds_of_day"]) / 86400.0
        head["sin_local_time"].add(np.broadcast_to(np.sin(utc + lon), a.shape), a)
        head["cos_local_time"].add(np.broadcast_to(np.cos(utc + lon), a.shape), a)
        est = diagnose(c)
        for k in gap:
            gap[k].add(np.asarray(s[k]) - np.asarray(est[k]), a)
        if j is not None:
            nxt = (j, records(j))
            c2, s2, _ = nxt[1]
            for k in lev:
                lev[k].add(np.asarray(getattr(c2, k)) - np.asarray(getattr(c, k)), a3)
            one["p_s"].add(np.asarray(c2.p_s) - np.asarray(c.p_s), a)
            for k in ("t2m", "u10", "v10", "mslp"):
                one[k].add(np.asarray(s2[k]) - np.asarray(s[k]), a)
            prev = nxt
        else:
            prev = None
    out = {"loss": {}, "head_inputs": {}, "head_outputs": {}}
    for k, m in lev.items():
        out["loss"][f"{'q' if k == 'q_v' else k}_resid_scale_lev"] = m.result()[1].tolist()
    for k, m in one.items():
        out["loss"][f"{'ps' if k == 'p_s' else k}_resid_scale"] = float(m.result()[1])
    for k, m in head.items():
        mean, std = m.result()
        out["head_inputs"][k] = {"mean": np.asarray(mean).tolist(),
                                 "std": np.asarray(std).tolist()}
    for k, m in gap.items():
        out["head_outputs"][f"{k}_gap_std"] = float(m.result()[1])
    return out


def pairs_in_years(store_index, times, years=YEARS):
    """``(i, j)`` over the store's analyses inside ``years``: ``j`` = the next
    analysis exactly 6 h later when it is cached and inside the years too."""
    have = set(int(i) for i in store_index)
    yr = lambda i: int(str(np.datetime64(times[i], "Y")))  # noqa: E731
    keep = sorted(i for i in have if years[0] <= yr(i) <= years[1])
    out = []
    for i in keep:
        j = i + 1
        ok = (j in have and j < len(times) and years[0] <= yr(j) <= years[1]
              and times[j] - times[i] == np.timedelta64(6, "h"))
        out.append((i, j if ok else None))
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--mode", required=True, choices=("physics", "neural_gcm", "sfno"))
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    import yaml
    import zarr

    from evaluations.wb_forecast import diagnose_headline_fields
    from legoesm.training import scale_build as sb
    from legoesm.training import wb_state_cache as wsc
    from legoesm.training.era5_to_state import open_era5_zarr
    from legoesm.training.neural_gcm_spectral import carry_to_spectral_state

    yml = yaml.safe_load(open(a.config))
    sb.validate_wb_campaign_yaml(yml)
    cfg = SimpleNamespace(mode=a.mode, training_core="spectral", smoke=False,
                          windows=None, n_days=None, multi_step_hours=None)
    _, grid, sigma, *_ = sb.build_mode_components(cfg, yml)
    era5_cfg, sst_cfg, micro = sb.wb_state_cache_configs(cfg, yml)
    ds = open_era5_zarr(era5_cfg.zarr_store)
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    cloud_ds = (open_era5_zarr(era5_cfg.cloud_zarr)
                if era5_cfg.load_cloud_condensate else None)
    flux_ds = ((open_era5_zarr(sst_cfg.flux_zarr) if sst_cfg.flux_zarr else ds)
               if sst_cfg.load_surface_fluxes else None)
    # Names the store only (its layout is part of the name): the builder's
    # own template time, so it exists in the ERA5 stores the cache was built
    # from.
    from scripts.data.build_wb_state_cache import needed_indices
    template = sb.wb_state_record(int(needed_indices(cfg, yml, times)[0]),
                                  era5_cfg, sst_cfg, ds, cloud_ds, flux_ds,
                                  grid, sigma, micro, times)
    path = sb.wb_state_cache_path(cfg, yml, grid, sigma, template)
    store_index = np.asarray(zarr.open_group(str(path), mode="r")["era5_index"])
    pairs = pairs_in_years(store_index, times)
    if not any(j is not None for _, j in pairs):
        raise SystemExit(f"{path} holds no 6 h pair of analyses in "
                         f"{YEARS[0]}-{YEARS[1]}")
    fresh = sb.wb_state_record(pairs[0][0], era5_cfg, sst_cfg, ds, cloud_ds,
                               flux_ds, grid, sigma, micro, times)
    cache = wsc.StateCache(path, fresh)
    wsc.assert_cache_matches(fresh, cache.read(pairs[0][0]))
    need = [i for p in pairs for i in p if i is not None]
    missing = cache.missing(need)
    if missing:
        raise SystemExit(f"{path}: {len(missing)} needed times unwritten")

    @jax.jit
    def _diag(c):
        d = diagnose_headline_fields(carry_to_spectral_state(c, grid), grid, sigma)
        return {k: d.fields[k] for k in ("t2m", "u10", "v10")}

    stats = accumulate(cache.read, pairs, grid, lambda c: jax.device_get(_diag(c)))
    sha = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    stats["meta"] = {
        "store": str(path), "years": list(YEARS), "git_sha": sha,
        "n_analyses": len(pairs), "n_pairs_6h": sum(j is not None for _, j in pairs),
        "first": str(times[pairs[0][0]]), "last": str(times[pairs[-1][0]]),
        "head_levels": f"lowest {N_HEAD_LEVELS} model levels, top to bottom",
        "weights": "Gaussian quadrature weight per cell",
        "std": "about the area-weighted mean",
    }
    out = path.parent / f"{path.stem}_scales.json"
    out.write_text(json.dumps(stats, indent=1))
    print(f"wrote {out}: {stats['meta']['n_analyses']} analyses, "
          f"{stats['meta']['n_pairs_6h']} 6 h pairs", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(_REPO))
    raise SystemExit(main())
