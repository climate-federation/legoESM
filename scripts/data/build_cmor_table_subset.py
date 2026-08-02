#!/usr/bin/env python
"""Build the vendored CMIP6 CMOR-table subset used by ``legoesm.io.cmor_output``.

Why this script exists
----------------------
The CMOR variable metadata (``units``, ``standard_name``, ``long_name``,
``cell_methods``, ``cell_measures``, ``positive``, ``type``) used to be
hand-typed into ~9 Python dicts in ``cmor_output.py``.  Hand-maintained
metadata drifts: an audit against the official PCMDI tables found wrong
``cell_methods`` on 29 of 35 ``Amon`` variables, wrong ``units`` on
``cli``/``clw``, and a completely missing ``positive`` attribute on all
15 flux variables.  The writer is now driven by the official tables
instead, and this script produces the vendored copy.

What it does
------------
Downloads the official tables from PCMDI/cmip6-cmor-tables (pinned by git
ref) and writes a TRIMMED subset -- only the variables legoESM actually
writes, only the entry fields the writer consumes -- into

    packages/core/legoesm/io/cmor_tables/

Trimming keeps the vendored payload around 60 kB instead of ~1.8 MB, and
keeps the repo offline-safe: nothing at runtime touches the network.

Re-run this script (with network access) to refresh the vendored copy
against a newer table release; ``git diff`` then shows exactly which
metadata changed.

Usage
-----
    python scripts/data/build_cmor_table_subset.py [--ref main] [--check]

``--check`` re-downloads and compares against the vendored copy without
writing, exiting non-zero on drift (for a manual/nightly audit -- it is
NOT wired into CI, which must stay offline).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict

RAW_BASE = "https://raw.githubusercontent.com/PCMDI/cmip6-cmor-tables/{ref}/Tables"
REPO_URL = "https://github.com/PCMDI/cmip6-cmor-tables"
# Pinned so a re-run is reproducible.  Bump deliberately, then inspect the
# resulting ``git diff`` of the vendored JSON before committing.
DEFAULT_REF = "087fe45d21c082e28723e0f930e4266abe91b853"  # 2025-09-04

# Variables legoESM writes, per official CMIP6 table.  Adding a new output
# variable means adding it here and re-running this script -- NEVER
# hand-typing an entry into the vendored JSON.
WANTED: Dict[str, tuple] = {
    "Amon": (
        "cli", "clivi", "clt", "clw", "clwvi", "evspsbl", "hfls", "hfss",
        "hur", "hurs", "hus", "pr", "prw", "ps", "psl", "rlds", "rldscs",
        "rlus", "rlut", "rlutcs", "rsds", "rsdscs", "rsdt", "rsus", "rsut",
        "rsutcs", "ta", "tas", "tauu", "tauv", "ts", "ua", "va", "wap", "zg",
    ),
    # ``rsut`` is NOT a CMIP6 ``day`` variable -- it lives in ``CFday``.
    # ``ua850``/``va850`` do not exist in ANY CMIP6 table; the 850 hPa winds
    # are ``ua``/``va`` on the ``plev8`` coordinate in ``day``.
    "day": ("pr", "psl", "rlut", "tas", "tasmax", "tasmin", "ua", "va"),
    "CFday": ("rsut",),
    "fx": ("areacella", "orog", "sftlf"),
    "Lmon": ("gpp", "lai", "mrso", "mrsos", "tsl"),
    "Omon": (
        "hfds", "masso", "mlotst", "msftmz", "msftyz", "so", "soga", "sos",
        "sosga", "tauuo", "tauvo", "thetao", "thetaoga", "tos", "tosga",
        "uo", "vo", "volo", "wfo", "wo", "zos",
    ),
    "Ofx": ("areacello", "deptho", "masscello", "sftof", "thkcello", "volcello"),
    "SImon": ("siconc", "sitemptop", "sithick", "siu", "siv"),
}

# Coordinate axes we need the ``requested`` level lists / scalar values for.
WANTED_AXES = ("plev19", "plev8", "plev7h", "height2m", "height10m", "sdepth")

# Entry fields the writer consumes.  Everything else (valid_min/max,
# ok_min_mean_abs, ...) is dropped.
ENTRY_FIELDS = (
    "frequency", "modeling_realm", "standard_name", "units", "cell_methods",
    "cell_measures", "long_name", "comment", "dimensions", "out_name",
    "type", "positive",
)

HEADER_FIELDS = (
    "data_specs_version", "cmor_version", "table_id", "realm", "table_date",
    "missing_value", "int_missing_value", "product", "approx_interval",
    "generic_levels", "mip_era", "Conventions",
)

AXIS_FIELDS = (
    "standard_name", "units", "axis", "long_name", "out_name", "positive",
    "requested", "stored_direction", "type", "value", "valid_min", "valid_max",
    "must_have_bounds",
)

# Global attributes the writer must populate, plus the ``Conventions``
# regex -- extracted from CMIP6_CV.json so the writer never hard-codes a
# value the CV rejects (the old writer emitted the invalid "CF-1.8").
CV_KEYS = ("required_global_attributes", "Conventions", "experiment_id")


def _fetch(ref: str, name: str) -> Dict[str, Any]:
    url = f"{RAW_BASE.format(ref=ref)}/{name}"
    with urllib.request.urlopen(url, timeout=60) as fh:  # noqa: S310
        return json.loads(fh.read().decode("utf-8"))


def _trim_table(raw: Dict[str, Any], wanted: tuple, ref: str) -> Dict[str, Any]:
    ve = raw["variable_entry"]
    missing = [v for v in wanted if v not in ve]
    if missing:
        raise KeyError(
            f"{raw['Header']['table_id']}: requested variables absent from the "
            f"official table: {missing}"
        )
    out = {
        "_provenance": {
            "source": REPO_URL,
            "ref": ref,
            "note": (
                "Trimmed subset generated by "
                "scripts/data/build_cmor_table_subset.py -- do not hand-edit."
            ),
        },
        "Header": {k: raw["Header"][k] for k in HEADER_FIELDS if k in raw["Header"]},
        "variable_entry": {
            v: {k: ve[v][k] for k in ENTRY_FIELDS if k in ve[v]} for v in sorted(wanted)
        },
    }
    return out


def _trim_coordinate(raw: Dict[str, Any], ref: str) -> Dict[str, Any]:
    ax = raw["axis_entry"]
    return {
        "_provenance": {"source": REPO_URL, "ref": ref},
        "axis_entry": {
            a: {k: ax[a][k] for k in AXIS_FIELDS if k in ax[a]}
            for a in WANTED_AXES
            if a in ax
        },
    }


def _trim_cv(raw: Dict[str, Any], ref: str) -> Dict[str, Any]:
    cv = raw["CV"]
    experiments = cv.get("experiment_id", {})
    return {
        "_provenance": {"source": REPO_URL, "ref": ref},
        "required_global_attributes": cv["required_global_attributes"],
        # Regex the ``Conventions`` global attribute must match.
        "Conventions": cv.get("Conventions"),
        # experiment_id -> the CV's long ``experiment`` string (a REQUIRED
        # global attribute the old writer omitted entirely).
        "experiment": {
            k: v.get("experiment", "") for k, v in sorted(experiments.items())
        },
        "nominal_resolution": cv.get("nominal_resolution", []),
    }


def build(ref: str, out_dir: Path, check: bool) -> int:
    payload: Dict[str, Dict[str, Any]] = {}
    for table, wanted in WANTED.items():
        payload[f"CMIP6_{table}.json"] = _trim_table(
            _fetch(ref, f"CMIP6_{table}.json"), wanted, ref
        )
    payload["CMIP6_coordinate.json"] = _trim_coordinate(
        _fetch(ref, "CMIP6_coordinate.json"), ref
    )
    payload["CMIP6_CV.json"] = _trim_cv(_fetch(ref, "CMIP6_CV.json"), ref)

    rc = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        text = json.dumps(data, indent=1, sort_keys=False) + "\n"
        target = out_dir / name
        if check:
            current = target.read_text() if target.exists() else ""
            if current != text:
                print(f"DRIFT: {target}")
                rc = 1
        else:
            target.write_text(text)
            print(f"wrote {target} ({len(text)} bytes)")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default=DEFAULT_REF, help="git ref of cmip6-cmor-tables")
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[2]
            / "packages/core/legoesm/io/cmor_tables"
        ),
    )
    ap.add_argument("--check", action="store_true", help="report drift, write nothing")
    args = ap.parse_args(argv)
    return build(args.ref, Path(args.out), args.check)


if __name__ == "__main__":
    sys.exit(main())
