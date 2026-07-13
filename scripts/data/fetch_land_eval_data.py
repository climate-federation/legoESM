"""Fetch / verify the land-evaluation datasets.

Implements the per-dataset access policy in
``docs/land/evaluation_data_manifest.md``: report what is present, download
the public Zenodo CHATS observations, and print actionable instructions for
the account-gated datasets (FLUXNET / MODIS).  Nothing large enters git.

Usage
-----
    # report presence of every dataset (no network):
    python scripts/data/fetch_land_eval_data.py --check

    # download the public Zenodo CHATS 30-min obs (record 17426258):
    python scripts/data/fetch_land_eval_data.py --fetch zenodo

The dataset registry (:data:`DATASETS`) is the machine-readable twin of the
manifest table; ``--check`` resolves each against the repo tree and, if set,
``$LEGO_LAND_EVAL_DATA``.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT_ENV = "LEGO_LAND_EVAL_DATA"
ZENODO_RECORD = "17426258"


@dataclass(frozen=True)
class Dataset:
    """One manifest row.

    ``access`` is ``in_tree`` | ``fetch`` | ``bundle`` | ``derived``.
    ``repo_relpath`` is checked first; ``bundle_relpath`` (if any) is checked
    under ``$LEGO_LAND_EVAL_DATA``.  ``marker`` is a glob that must match at
    least one file for the dataset to count as present.
    """

    key: str
    label: str
    access: str
    repo_relpath: str | None
    bundle_relpath: str | None
    marker: str
    doi: str = ""


DATASETS: dict[str, Dataset] = {
    "chats7_forcing": Dataset(
        "chats7_forcing", "CHATS7 tower forcing NetCDF", "in_tree",
        "clm-ml-jax/src/input_files/tower-forcing/CHATS7", None, "2007-05.nc",
    ),
    "fortran_v2": Dataset(
        "fortran_v2", "Fortran CLM-ML v2 reference .out", "in_tree",
        "docs/output_files_clm_ml-v2", "clm_ml_v2_fortran",
        "CHATS7_2007-05_flux.out",
    ),
    "jax_standalone": Dataset(
        "jax_standalone", "JAX-standalone CLM-ML .out", "in_tree",
        "clm-ml-jax/src/output_files/JAX_outputs_05_2007_31days",
        "clm_ml_jax_standalone", "CHATS7_2007-05_flux.out",
    ),
    "zenodo_obs": Dataset(
        "zenodo_obs", "CHATS 30-min multi-level obs (Zenodo 17426258)",
        "fetch", "docs/MLC_experiment_plan/Data/Zenodo_17426258_CHATS_30-min",
        "Zenodo_17426258_CHATS_30-min", "chats_30min_data_2007_05*.csv",
        doi="10.5281/zenodo.17426258",
    ),
    "fluxnet_us_mms": Dataset(
        "fluxnet_us_mms", "US-MMS FLUXNET slim CSV", "derived",
        "data/fluxnet/US-MMS", "fluxnet/US-MMS", "*_slim.csv",
    ),
    "fluxnet_fi_hyy": Dataset(
        "fluxnet_fi_hyy", "FI-Hyy FLUXNET slim CSV", "derived",
        "data/fluxnet/FI-Hyy", "fluxnet/FI-Hyy", "*_slim.csv",
    ),
    "fluxnet_us_ton": Dataset(
        "fluxnet_us_ton", "US-Ton FLUXNET slim CSV", "derived",
        "data/fluxnet/US-Ton", "fluxnet/US-Ton", "*_slim.csv",
    ),
}

# account-gated portals to point users at when a fetch target is not public.
_PORTAL_HINT = {
    "fluxnet_us_mms": "AmeriFlux BASE US-MMS (https://ameriflux.lbl.gov/)",
    "fluxnet_us_ton": "AmeriFlux BASE US-Ton (https://ameriflux.lbl.gov/)",
    "fluxnet_fi_hyy": "ICOS / FLUXNET FI-Hyy (https://www.icos-cp.eu/)",
}


def _resolve(ds: Dataset, repo_root: pathlib.Path,
             bundle_root: pathlib.Path | None) -> pathlib.Path | None:
    """Return the directory where ``ds`` is present, else None."""
    candidates: list[pathlib.Path] = []
    if ds.repo_relpath:
        candidates.append(repo_root / ds.repo_relpath)
    if bundle_root and ds.bundle_relpath:
        candidates.append(bundle_root / ds.bundle_relpath)
    for c in candidates:
        if c.is_dir() and any(c.glob(ds.marker)):
            return c
    return None


def check_manifest(
    repo_root: pathlib.Path,
    bundle_root: pathlib.Path | None = None,
) -> dict[str, dict]:
    """Resolve every dataset; return ``{key: {present, path, ...}}``."""
    status: dict[str, dict] = {}
    for key, ds in DATASETS.items():
        found = _resolve(ds, repo_root, bundle_root)
        status[key] = {
            "label": ds.label,
            "access": ds.access,
            "present": found is not None,
            "path": str(found) if found else None,
        }
    return status


def fetch_zenodo(record_id: str, dest: pathlib.Path) -> list[pathlib.Path]:
    """Download every file of a public Zenodo record into ``dest``.

    Uses only the standard library (urllib + json).  Network access is
    confined to this function — importing/``--check`` never touches the
    network.
    """
    import json
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    api = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api) as resp:  # noqa: S310 (trusted host)
        meta = json.loads(resp.read().decode())
    written: list[pathlib.Path] = []
    for f in meta.get("files", []):
        url = f["links"]["self"]
        name = f["key"]
        out = dest / name
        print(f"  downloading {name} ...", flush=True)
        urllib.request.urlretrieve(url, out)  # noqa: S310
        written.append(out)
    return written


def _print_check(status: dict[str, dict]) -> None:
    print(f"\nLand-evaluation data check  (repo={REPO_ROOT})")
    root = os.environ.get(DATA_ROOT_ENV)
    print(f"  ${DATA_ROOT_ENV} = {root or '<unset>'}\n")
    for key, s in status.items():
        mark = "OK " if s["present"] else "-- "
        loc = s["path"] or "MISSING"
        print(f"  [{mark}] {key:<16} ({s['access']:<8}) {loc}")
        if not s["present"] and key in _PORTAL_HINT:
            print(f"          fetch from: {_PORTAL_HINT[key]}")
    missing = [k for k, s in status.items() if not s["present"]]
    if missing:
        print(
            f"\n  {len(missing)} dataset(s) missing. See "
            "docs/land/evaluation_data_manifest.md; public Zenodo obs via "
            "--fetch zenodo."
        )
    else:
        print("\n  all datasets present.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch/verify land-evaluation datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--check", action="store_true",
        help="Report presence of every dataset (default if no --fetch).",
    )
    p.add_argument(
        "--fetch", choices=["zenodo"], default=None,
        help="Download a public dataset (zenodo = CHATS obs record "
             f"{ZENODO_RECORD}).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bundle = os.environ.get(DATA_ROOT_ENV)
    bundle_root = pathlib.Path(os.path.expanduser(bundle)) if bundle else None

    if args.fetch == "zenodo":
        ds = DATASETS["zenodo_obs"]
        dest = REPO_ROOT / (ds.repo_relpath or "")
        print(f"Fetching Zenodo record {ZENODO_RECORD} -> {dest}")
        files = fetch_zenodo(ZENODO_RECORD, dest)
        print(f"  wrote {len(files)} file(s).")
        return 0

    # default action: check.
    _print_check(check_manifest(REPO_ROOT, bundle_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
