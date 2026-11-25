"""Command line: discover, download, process, report."""
import argparse
import json
from pathlib import Path

import xarray as xr

from . import config, dataset_builder, discovery, download, report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="arm_sgp_downloader")
    parser.add_argument("command", choices=("discover", "download", "process", "report"))
    parser.add_argument("--root", type=Path, default=config.DEFAULT_ROOT)
    parser.add_argument("--period", choices=("all",), default="all")
    args = parser.parse_args(argv)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "discover":
        result = discovery.run(root)
        print(len(result["files"]), "files catalogued")
    elif args.command == "download":
        listing = root / "discovery.json"
        entries = json.loads(listing.read_text())["files"] if listing.exists() else discovery.run(root)["files"]
        records = download.run(root, entries)
        print({s: sum(r["status"] == s for r in records) for s in {r["status"] for r in records}})
    elif args.command == "process":
        dataset_builder.build(root)
        print("unified dataset written")
    else:
        print(json.dumps(report.run(root, xr.open_dataset(root / "unified" / "sgp_cf_30min.nc")), indent=1)[:2000])


if __name__ == "__main__":
    main()
