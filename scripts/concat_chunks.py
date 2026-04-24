"""Concatenate snapshot NPZ files from chunked continuation runs.

Writes a combined ``snapshots_native.npz`` + ``land_mask`` etc. and
``mean_timeseries.csv`` + ``conservation_timeseries.csv`` so the
existing diagnostic scripts see a single long run.

Usage:

    python scripts/concat_chunks.py \
        --chunks results/ocean/.../som_U02_Bh2.3e11_Cs0.2_600d \
                 results/ocean/.../som_U02_0600to1200d \
                 results/ocean/.../som_U02_1200to1800d \
                 results/ocean/.../som_U02_1800to2400d \
                 results/ocean/.../som_U02_2400to3000d \
                 results/ocean/.../som_U02_3000to3600d \
        --out results/ocean/.../som_U02_fullchain_3600d
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _concat_npz(chunk_dirs, fname):
    """Concatenate time-indexed arrays across chunks, adding a cumulative
    time offset so each chunk's local-time [0, 600] d maps to absolute
    model-days [0, 600, 1200, …]."""
    arrays = {}
    cum_days = 0.0
    cum_steps = 0
    for i, d in enumerate(chunk_dirs):
        data = np.load(d / fname, allow_pickle=True)
        nt = data["times_days"].shape[0]
        offsets = {"times_days": cum_days, "steps": cum_steps}
        for k in data.files:
            a = np.asarray(data[k])
            # For continuations (i > 0) drop the first entry to avoid
            # duplicating the chunk-boundary snapshot.
            if a.ndim >= 1 and a.shape[0] == nt:
                a = a[1:] if i > 0 else a
                if k in offsets:
                    a = a + offsets[k]
            else:
                # scalar or static — take from first chunk only
                if i > 0:
                    continue
            arrays.setdefault(k, []).append(a)
        # Advance the cumulative offset by this chunk's final local time
        cum_days += float(data["times_days"][-1])
        cum_steps += int(data["steps"][-1])
    return {k: np.concatenate(chunks, axis=0) for k, chunks in arrays.items()}


def _concat_csv(chunk_dirs, fname):
    """Concatenate per-chunk CSV timeseries, adding cumulative offsets to
    the time-like columns so the result is one continuous absolute-time
    series."""
    dfs = []
    cum_days = 0.0
    cum_steps = 0
    for i, d in enumerate(chunk_dirs):
        f = d / fname
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if i > 0 and len(df) > 0:
            df = df.iloc[1:].copy()
        if "time_days" in df.columns:
            df["time_days"] = df["time_days"] + cum_days
        if "step" in df.columns:
            df["step"] = df["step"] + cum_steps
        dfs.append(df)
        # Advance offsets using the raw (un-shifted) last row of the
        # original on-disk CSV — since we just added cum to df, read fresh
        raw = pd.read_csv(f)
        if "time_days" in raw.columns and len(raw) > 0:
            cum_days += float(raw["time_days"].iloc[-1])
        if "step" in raw.columns and len(raw) > 0:
            cum_steps += int(raw["step"].iloc[-1])
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Concatenating snapshots_native.npz …")
    snaps = _concat_npz(args.chunks, "snapshots_native.npz")
    np.savez_compressed(args.out / "snapshots_native.npz", **snaps)
    print(f"  → {args.out / 'snapshots_native.npz'}  "
          f"times: {snaps['times_days'][0]:.0f} → {snaps['times_days'][-1]:.0f} d  "
          f"({len(snaps['times_days'])} snapshots)")

    for csv_name in ("mean_timeseries.csv", "conservation_timeseries.csv"):
        df = _concat_csv(args.chunks, csv_name)
        if df is not None:
            df.to_csv(args.out / csv_name, index=False)
            print(f"  → {args.out / csv_name}  ({len(df)} rows)")

    # Copy the final results.txt content with concatenated summary
    last_results = args.chunks[-1] / "results.txt"
    if last_results.exists():
        (args.out / "results.txt").write_text(
            last_results.read_text()
            + f"\nchunks: {len(args.chunks)}\n"
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
