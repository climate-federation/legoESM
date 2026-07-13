"""Filter FLUXNET FULLSET half-hourly / hourly CSVs to the target windows for
E3, E4, E6a of the MLC experiment plan.

Target windows (from docs/MLC_experiment_plan/experiment_plan.md §5.2, 4.3, 6.3):
  * US-MMS (DBF): 2011-10-01 → 2013-12-31 (spinup + 2012 heatwave + 2013 training)
  * FI-Hyy (ENF): 2012-07-01 → 2013-12-31
  * US-Ton (SAV): 2011-07-01 → 2012-12-31

Reads the ~300-830 MB source CSVs one chunk at a time; writes ~10-30 MB slim
per-site subsets under ``data/fluxnet/<site>/``.

Usage
-----
    python scripts/data/filter_fluxnet_to_target_years.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "docs" / "MLC_experiment_plan" / "Data"
OUT_ROOT = REPO_ROOT / "data" / "fluxnet"

SITES = {
    "US-MMS": {
        "src": (
            DATA_ROOT
            / "AMF_US-MMS_FLUXNET_1999-2023_v1.3_r1"
            / "AMF_US-MMS_FLUXNET_FLUXMET_HR_1999-2023_v1.3_r1.csv"
        ),
        "start": 201110010000,
        "end": 201312312300,
        "freq_min": 60,
    },
    "FI-Hyy": {
        "src": (
            DATA_ROOT
            / "ICOS_FI-Hyy_FLUXNET_1997-2024_v1.3_r1"
            / "ICOS_FI-Hyy_FLUXNET_FLUXMET_HH_1997-2024_v1.3_r1.csv"
        ),
        "start": 201207010000,
        "end": 201312312330,
        "freq_min": 30,
    },
    "US-Ton": {
        "src": (
            DATA_ROOT
            / "AMF_US-Ton_FLUXNET_2001-2025_v1.3_r1"
            / "AMF_US-Ton_FLUXNET_FLUXMET_HH_2001-2025_v1.3_r1.csv"
        ),
        "start": 201107010000,
        "end": 201212312330,
        "freq_min": 30,
    },
}


def filter_site(site: str, cfg: dict) -> None:
    src = cfg["src"]
    if not src.is_file():
        raise FileNotFoundError(src)
    out_dir = OUT_ROOT / site
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (src.stem + "_slim.csv")

    print(f"[{site}] reading {src.name} ({src.stat().st_size / 1e6:.0f} MB) …",
          flush=True)

    chunks = []
    kept = 0
    total = 0
    for chunk in pd.read_csv(src, chunksize=500_000, dtype={"TIMESTAMP_START": "int64",
                                                             "TIMESTAMP_END": "int64"}):
        total += len(chunk)
        m = (chunk["TIMESTAMP_START"] >= cfg["start"]) & (
            chunk["TIMESTAMP_START"] <= cfg["end"])
        sub = chunk[m]
        if not sub.empty:
            chunks.append(sub)
            kept += len(sub)
    df = pd.concat(chunks, ignore_index=True)

    # Sanity: expected row count in the window
    span_days = (pd.to_datetime(str(cfg["end"]), format="%Y%m%d%H%M")
                 - pd.to_datetime(str(cfg["start"]), format="%Y%m%d%H%M")).total_seconds() / 86400
    expected = int(round(span_days * 1440 / cfg["freq_min"]))
    df.to_csv(out, index=False)
    print(
        f"[{site}] wrote {out.name}: {kept}/{total} rows kept "
        f"({expected} expected, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB)",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sites", nargs="+", choices=sorted(SITES), default=sorted(SITES))
    args = p.parse_args(argv)
    for s in args.sites:
        filter_site(s, SITES[s])
    return 0


if __name__ == "__main__":
    sys.exit(main())
