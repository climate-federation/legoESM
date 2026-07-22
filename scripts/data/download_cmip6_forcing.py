#!/usr/bin/env python
"""Download real CMIP6 / input4MIPs AMIP forcing off Levante — no account needed.

Before this, the only local AMIP forcing was the SYNTHETIC deck
(``generate_amip_forcing.py``); the REAL CMIP6 forcing was reachable only on
Levante (``/pool/data/ICON/...``, see ``inspect_cmip6_forcing.py``) and the
staging helper (``stage_amip_realdata.py``) could only VALIDATE a file you had
already downloaded by hand.  This script actually FETCHES the canonical
input4MIPs forcing datasets from ESGF.

input4MIPs is OPEN data: the files are served over the public LLNL Globus HTTPS
endpoint with NO ESGF account (the same access ``download_jra55_iaf.py`` already
uses for the JRA55-do OMIP forcing — JRA55-do is itself input4MIPs).  We reuse
that proven pattern: query the ESGF Solr catalog for the file docs, pick the
public HTTPS replica, and pull it with a resumable ``curl``.

The files land RAW (their native input4MIPs variable names + units, e.g.
``tosbcs`` in K, ``siconcbcs`` in percent).  The AMIP loaders resolve variables
by NAME + CF units (not by path), so most channels are consumed directly:
``sst_sic``, ``ozone`` and ``volcanic`` need only ``run_amip_cmip6_deck.py``
flags (e.g. ``--sst-file`` + ``--sst-var tosbcs --sst-offset 0 --sic-var
siconcbcs --sic-scale 0.01``).  ``ghg``, ``solar`` and ``aerosol`` need a thin
one-off adapter first (a variable rename / spectral rebin) — flagged per channel
by ``--list`` and printed after a run.  This script's job is the DOWNLOAD; the
per-channel adapters are a small follow-up (see each channel's ``adapter_note``).

Usage::

    # See what would be fetched (no bytes moved):
    python scripts/data/download_cmip6_forcing.py --channels sst_sic ozone \\
        --out-dir data/cmip6_forcing --dry-run

    # Fetch the AMIP-II boundary condition + ozone:
    python scripts/data/download_cmip6_forcing.py --channels sst_sic ozone \\
        --out-dir data/cmip6_forcing

    # List the channel registry (source_ids + the deck flags they feed):
    python scripts/data/download_cmip6_forcing.py --list

Not every channel's numeric CONTENT is validated here (that is what a live ESGF
fetch + ``stage_amip_realdata.py`` do); this script's job is to resolve the
right input4MIPs files and move the bytes reproducibly.  The Solr query returns
the LATEST published version, so a bumped dataset version is handled without a
code change.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Channel registry: each AMIP/CMIP6 forcing channel -> its input4MIPs dataset.
#
# ``source_id`` is the canonical input4MIPs producer-dataset id; the Solr query
# resolves the concrete file URLs + versions at fetch time, so only the stable
# facets (project, source_id, variable_id) are pinned here.  ``deck_flags`` is
# the exact ``run_amip_cmip6_deck.py`` invocation that consumes the RAW files
# (printed after a run) — the raw input4MIPs variable name + unit convention.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForcingSpec:
    channel: str            # our channel key
    source_id: str          # input4MIPs source_id (producer dataset)
    variable_ids: tuple[str, ...]   # input4MIPs variable_id(s) to fetch
    target_mip: str         # input4MIPs target_mip facet ("CMIP" for the DECK set)
    description: str
    deck_flags: str         # how run_amip_cmip6_deck.py consumes the raw file(s)
    # True: the AMIP loader reads the RAW input4MIPs file as-is (its variable
    # name + CF units are what the loader detects, verified against the schema
    # in packages/tools/legoesm/forcing/{amip,external}.py).  False: the raw
    # file needs a thin adapter (variable rename / spectral rebin) first.
    directly_consumable: bool = True
    adapter_note: str = ""  # the adaptation a non-consumable channel needs
    verified: bool = False  # True: source_id cross-checked against repo code
    extra_facets: dict = field(default_factory=dict)


# Only stable facets are pinned; the Solr search resolves version + file URLs.
# ``verified=True`` means the source_id/variable is cross-checked against code
# already in this repo (stage_amip_realdata.py / inspect_cmip6_forcing.py); the
# others are the well-known input4MIPs producers for that channel and should be
# confirmed against the ESGF catalog for the exact CMIP6 vintage you want.
FORCINGS: dict[str, ForcingSpec] = {
    "sst_sic": ForcingSpec(
        channel="sst_sic",
        source_id="PCMDI-AMIP-1-1-9",
        variable_ids=("tosbcs", "siconcbcs"),
        target_mip="CMIP",
        description="AMIP II prescribed SST + sea-ice boundary conditions "
                    "(tosbcs [K], siconcbcs [percent]).",
        # The deck takes ONE --sst-file and reads SIC from it via --sic-var
        # (run_amip_cmip6_deck.py has no --sic-path).  PCMDI usually ships one
        # file carrying BOTH tosbcs and siconcbcs; if the download yields them as
        # two files, merge them into one (or drive run_amip.py directly, which
        # does take --sic-path).  The loader units-guard
        # (amip.py:_validate_sst_sic_units) fails loud on a wrong var/offset/
        # scale, so a mismatch never runs silently.
        deck_flags="--sst-file <tosbcs+siconcbcs..nc> --sst-var tosbcs "
                   "--sst-offset 0 --sic-var siconcbcs --sic-scale 0.01",
        verified=True,   # stage_amip_realdata.py documents exactly this dataset
    ),
    "ozone": ForcingSpec(
        channel="ozone",
        source_id="UReading-CCMI-1-0",
        variable_ids=("vmro3",),
        target_mip="CMIP",
        description="CMIP6 historical ozone volume mixing ratio (vmro3).",
        # external.py:_detect_ozone_varname auto-detects 'vmro3'; units mol/mol.
        deck_flags="--ozone-source standard --ozone-forcing external "
                   "--ozone-file <vmro3..nc>",
        verified=True,   # matches inspect_cmip6_forcing.py's ozone_cmip6 pattern
    ),
    "volcanic": ForcingSpec(
        channel="volcanic",
        source_id="UOEXETER-CMIP-2-1-0",
        variable_ids=("ext_sun", "ext_earth"),
        target_mip="CMIP",
        description="CMIP6 stratospheric (volcanic) aerosol optical "
                    "properties, band-resolved (ext_sun SW / ext_earth LW).",
        # external.py:_load_volcanic_extinction_anchored reads ext_sun/ext_earth
        # directly (integrates over altitude, band-collapses to a broadband AOD).
        deck_flags="--aerosol-forcing external --volcanic-aerosol-file "
                   "<bc_aeropt..nc> --volcanic-aerosol-scale 1 "
                   "--volcanic-aerosol-lw",
    ),
    "ghg": ForcingSpec(
        channel="ghg",
        source_id="UoM-CMIP-1-2-0",
        # All five the loader requires (external.py:_load_ghg_annual_file wants
        # CO2/CH4/N2O/CFC_11/CFC_12) — fetch every raw input to build them.
        variable_ids=(
            "mole_fraction_of_carbon_dioxide_in_air",
            "mole_fraction_of_methane_in_air",
            "mole_fraction_of_nitrous_oxide_in_air",
            "mole_fraction_of_cfc11_in_air",
            "mole_fraction_of_cfc12_in_air",
        ),
        target_mip="CMIP",
        description="CMIP6 well-mixed greenhouse-gas mole fractions "
                    "(CO2/CH4/N2O/CFC-11/CFC-12, global-mean annual).",
        deck_flags="--ghg-forcing external --ghg-file <ghg.nc> (after adapt)",
        directly_consumable=False,
        # Loader wants ONE file with vars named CO2/CH4/N2O/CFC_11/CFC_12 on a
        # fractional-year axis; the raw input4MIPs 'gm' files are one-gas-per-
        # file with CF-long variable names.
        adapter_note="run scripts/data/adapt_cmip6_ghg.py --in-dir <ghg dir> "
                     "--out ghg.nc (merges the five per-gas files, renames "
                     "mole_fraction_of_<gas>_in_air -> CO2/CH4/N2O/CFC_11/"
                     "CFC_12 on a fractional-year axis).",
    ),
    "solar": ForcingSpec(
        channel="solar",
        source_id="SOLARIS-HEPPA-CMIP-6-1",
        variable_ids=("tsi", "ssi"),
        target_mip="CMIP",
        description="CMIP6 solar forcing: total (tsi) + spectral (ssi) "
                    "irradiance.",
        deck_flags="--solar-source spectral_file --solar-file <solar.nc> "
                   "--solar-tsi-var TSI --solar-spectral-var SSI_frac "
                   "--solar-spectral-band-order rrtmg_sw (after adapt)",
        directly_consumable=False,
        # Loader wants TSI (W/m2) + SSI_frac (14 RRTMG-SW band FRACTIONS summing
        # to 1); the raw SOLARIS-HEPPA ssi is high-spectral-resolution W/m2/nm.
        adapter_note="bin the native-resolution ssi into the 14 RRTMG-SW bands "
                     "and normalise to fractions (SSI_frac); keep tsi as TSI.",
    ),
    "aerosol": ForcingSpec(
        channel="aerosol",
        source_id="MPI-M-MACv2SP-1-0",
        variable_ids=("aod", "ssa", "asy"),
        target_mip="CMIP",
        description="MACv2-SP simple-plume anthropogenic tropospheric aerosol "
                    "optical properties (AOD/SSA/asymmetry).",
        deck_flags="--aerosol-forcing external --aerosol-file <aod.nc> "
                   "(after adapt; SSA/asy -> RRTMGPConfig.aerosol_ssa_bands/"
                   "aerosol_g_bands, the per-band optics from PR #1276)",
        directly_consumable=False,
        # MACv2-SP is a PLUME-PARAMETER file, not a gridded 'aod' field; the
        # loader (external.py) reads a gridded 'aod'. The ICON Kinne
        # aeropt_kinne_sw_b14 files ARE gridded+directly read but are DKRZ-
        # preprocessed, not published on ESGF as such.
        adapter_note="evaluate the MACv2-SP simple plume to a gridded, "
                     "band-resolved AOD/SSA/asymmetry (Stevens 2017), or fetch "
                     "the ICON Kinne aeropt files from a DKRZ mirror.",
    ),
}


# ---------------------------------------------------------------------------
# ESGF Solr lookup (public; no auth) — mirrors download_jra55_iaf.py.
# ---------------------------------------------------------------------------

# ESGF index nodes.  ESGF is federated but discovery is currently PARTITIONED
# (DOE/US vs Europe/AU), so a single node may not see every dataset — we query
# several and take the first NON-EMPTY answer (an empty answer might just be a
# partition gap, so we try the next node before concluding "not found").  LLNL
# is the primary (it backs download_jra55_iaf.py's input4MIPs fetch); DKRZ +
# CEDA cover the European side.  ``--search-url`` overrides with a single node.
ESGF_SEARCH_NODES = (
    "https://esgf-node.llnl.gov/esg-search/search/",
    "https://esgf-data.dkrz.de/esg-search/search/",
    "https://esgf.ceda.ac.uk/esg-search/search/",
)
ESGF_SEARCH = ESGF_SEARCH_NODES[0]


def _solr_query(spec: ForcingSpec, variable_id: str,
                search_url: str | None = None) -> list[dict]:
    """Return the input4MIPs File docs for one (source_id, variable_id).

    Isolated + injectable (``search_url``) so tests can point at a stub.  With
    no explicit node, tries the ESGF index mirrors in turn and returns the first
    NON-EMPTY result (discovery is partitioned, so an empty answer from one node
    is not authoritative — try the rest first).  Returns ``[]`` ONLY when EVERY
    node answered cleanly-empty (an authoritative "not found"); raises if any
    node failed and none held the dataset (inconclusive — unreachable != absent).
    ``target_mip`` is included so a source_id reused across target_mips can't
    return the wrong dataset.
    """
    facets = {
        "project": "input4MIPs",
        "source_id": spec.source_id,
        "variable_id": variable_id,
        "target_mip": spec.target_mip,
        "type": "File",
        "format": "application/solr+json",
        "limit": 5000,
    }
    facets.update(spec.extra_facets)
    qs = urllib.parse.urlencode(facets)
    nodes = (search_url,) if search_url else ESGF_SEARCH_NODES
    clean_empty = 0            # nodes that answered cleanly with zero docs
    last_err: Exception | None = None
    for node in nodes:
        url = f"{node}?{qs}"
        answered = False
        for attempt in range(3):
            out = subprocess.run(
                ["curl", "-sL", "--max-time", "60", url],
                capture_output=True, text=True,
            )
            if out.returncode == 0 and out.stdout:
                try:
                    docs = json.loads(out.stdout)["response"]["docs"]
                except (json.JSONDecodeError, KeyError) as e:
                    last_err = e
                else:
                    if docs:
                        return docs      # first non-empty answer wins
                    answered = True      # clean empty -> try the next node
                    break
            time.sleep(2 ** attempt)
        if answered:
            clean_empty += 1
    # A definitive "not found" needs EVERY node to have answered cleanly-empty;
    # if some node FAILED (network/parse) we can't rule out that it held the
    # dataset, so that is an error, not an empty result (codex review).
    if clean_empty == len(nodes):
        return []
    raise RuntimeError(
        f"ESGF Solr query for {spec.source_id}/{variable_id} was inconclusive: "
        f"{clean_empty}/{len(nodes)} nodes answered empty, the rest failed "
        f"(last error: {last_err}). Retry or pass --search-url for a live node.")


def _pick_https(doc: dict) -> str | None:
    """Extract the public HTTPS file-server URL from a Solr doc.

    ESGF ``url`` entries are ``<url>|<mime>|<service>``; we want the plain HTTPS
    ``HTTPServer`` replica (no OpenID / Globus-auth needed for input4MIPs).  The
    service field is matched EXACTLY — an HTTPS URL that is actually an OPeNDAP
    (``dodsC``) or Globus-GUI endpoint must not be treated as a downloadable
    file (codex review).
    """
    for u in doc.get("url", []):
        parts = u.split("|")
        plain = parts[0]
        service = parts[2] if len(parts) >= 3 else ""
        if plain.startswith("https://") and service == "HTTPServer":
            return plain
    return None


def _latest_version(docs: list[dict]) -> list[dict]:
    """Keep only the File docs of the newest dataset version.

    A Solr File search can return several published versions; the ``version``
    field (an int like ``20200101``) orders them.  Without this a fetch could
    mix files from different vintages.  Missing versions sort as 0 (kept only
    if nothing is versioned).
    """
    def ver(d: dict) -> int:
        v = d.get("version", 0)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
    if not docs:
        return docs
    newest = max(ver(d) for d in docs)
    return [d for d in docs if ver(d) == newest]


def resolve_files(spec: ForcingSpec, variable_id: str,
                  search_url: str | None = None) -> list[tuple[str, str, int]]:
    """Resolve ``(filename, https_url, size_bytes)`` for one variable.

    Returns every file of the latest version (input4MIPs forcing is often split
    into per-decade files), so the caller can pull a whole time range.
    """
    docs = _latest_version(_solr_query(spec, variable_id, search_url))
    out: list[tuple[str, str, int]] = []
    for doc in docs:
        url = _pick_https(doc)
        if url is None:
            continue
        title = doc.get("title") or url.rsplit("/", 1)[-1]
        out.append((title, url, int(doc.get("size", 0))))
    return out


# ---------------------------------------------------------------------------
# Download (resumable curl) — mirrors download_jra55_iaf.py._download_one.
# ---------------------------------------------------------------------------

def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def download_one(url: str, dest: Path, expected_size: int,
                 force: bool = False) -> bool:
    """Fetch one file with a resumable, retrying curl; verify size if known."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        actual = dest.stat().st_size
        if expected_size and actual == expected_size:
            print(f"  [skip] {dest.name} ({_human(actual)})")
            return True
        print(f"  [resume] {dest.name} "
              f"({_human(actual)} of {_human(expected_size)})")
    else:
        print(f"  [get] {dest.name} ({_human(expected_size)})")

    cmd = [
        "curl", "--fail", "--location", "--retry", "5", "--retry-delay", "10",
        "--continue-at", "-", "--show-error", "--silent",
        "--output", str(dest), url,
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  [error] curl exit {proc.returncode} for {dest.name}",
              file=sys.stderr)
        return False
    actual = dest.stat().st_size
    if expected_size and actual != expected_size:
        print(f"  [warn] {dest.name}: got {_human(actual)}, "
              f"expected {_human(expected_size)}", file=sys.stderr)
        return False
    print(f"  [done] {dest.name}")
    return True


# ---------------------------------------------------------------------------
# Plan + CLI
# ---------------------------------------------------------------------------

def build_plan(specs: list[ForcingSpec], out_dir: Path,
               search_url: str | None = None
               ) -> tuple[list[tuple[ForcingSpec, str, str, int, Path]],
                          list[tuple[str, str]]]:
    """Resolve every (channel, variable) to concrete files under ``out_dir``.

    Layout: ``<out-dir>/<channel>/<filename>`` — one directory per channel so a
    channel's files are grouped for the deck flags that point at them.

    Returns ``(plan, missing)`` where ``missing`` is the list of
    ``(channel, variable_id)`` that ESGF returned NO file for.  A missing
    variable is NOT silently dropped into a "success": a channel needs ALL its
    variables (e.g. sst_sic needs both tosbcs AND siconcbcs), so the caller
    treats a non-empty ``missing`` as a failure (codex review).
    """
    plan: list[tuple[ForcingSpec, str, str, int, Path]] = []
    missing: list[tuple[str, str]] = []
    # dest -> (owner label, url) of the file already planned for that path.
    seen_dests: dict[Path, tuple[str, str]] = {}
    for spec in specs:
        for variable_id in spec.variable_ids:
            files = resolve_files(spec, variable_id, search_url)
            if not files:
                print(f"  [missing] {spec.channel}/{variable_id} "
                      f"(no input4MIPs file for source_id {spec.source_id})",
                      file=sys.stderr)
                missing.append((spec.channel, variable_id))
                continue
            planned_here = 0
            satisfied_by_shared = False
            had_conflict = False
            for fname, url, size in files:
                dest = out_dir / spec.channel / fname
                if dest in seen_dests:
                    owner, owner_url = seen_dests[dest]
                    if owner_url == url:
                        # The SAME physical file already planned also carries
                        # this variable (e.g. one combined PCMDI file serves both
                        # tosbcs and siconcbcs) -> this variable is satisfied; do
                        # not download it twice.
                        satisfied_by_shared = True
                    else:
                        # Different source, identical filename -> a real conflict
                        # that would overwrite; keep the first, flag loudly.  Even
                        # if this variable plans OTHER files, a dropped time slice
                        # makes it incomplete (had_conflict below).
                        had_conflict = True
                        print(f"  [warn] filename conflict at {dest}: "
                              f"{owner} vs {spec.channel}/{variable_id} "
                              f"(different source); keeping the first.",
                              file=sys.stderr)
                    continue
                seen_dests[dest] = (f"{spec.channel}/{variable_id}", url)
                plan.append((spec, variable_id, url, size, dest))
                planned_here += 1
            # Incomplete (never a silent partial success, codex review) if the
            # variable added NO new file and no shared file covered it, OR a
            # conflict dropped one of its files (a missing time slice).
            if had_conflict or (planned_here == 0 and not satisfied_by_shared):
                missing.append((spec.channel, variable_id))
    return plan, missing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--channels", nargs="+", default=None,
                   choices=sorted(FORCINGS.keys()),
                   help="Forcing channels to fetch. Default: all.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (created; one subdir per channel).")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve + print the plan (URLs, sizes) without "
                        "downloading.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if a local file matches the size.")
    p.add_argument("--list", action="store_true",
                   help="Print the channel registry (source_ids + deck flags) "
                        "and exit.")
    p.add_argument("--search-url", default=None,
                   help="ESGF Solr search endpoint (single node). Default: try "
                        f"the federation mirrors {ESGF_SEARCH_NODES}.")
    return p.parse_args(argv)


def _print_registry() -> None:
    print("input4MIPs forcing channels:\n")
    for spec in FORCINGS.values():
        mark = "verified" if spec.verified else "confirm-vs-ESGF"
        consume = ("consumed raw" if spec.directly_consumable
                   else "needs adapter")
        print(f"  {spec.channel:9s} [{mark}; {consume}]  "
              f"source_id={spec.source_id}")
        print(f"      {spec.description}")
        print(f"      vars: {', '.join(spec.variable_ids)}")
        print(f"      deck: {spec.deck_flags}")
        if not spec.directly_consumable:
            print(f"      adapt: {spec.adapter_note}")
        print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        _print_registry()
        return 0
    if args.out_dir is None:
        print("--out-dir is required (unless --list).", file=sys.stderr)
        return 2

    keys = args.channels or list(FORCINGS.keys())
    specs = [FORCINGS[k] for k in keys]

    print("=" * 70)
    print("CMIP6 / input4MIPs AMIP forcing download")
    print("=" * 70)
    print(f"  channels : {keys}")
    print(f"  out_dir  : {args.out_dir}")
    print(f"  search   : {args.search_url}\n")

    print("Resolving files from the ESGF Solr catalog ...")
    plan, missing = build_plan(specs, args.out_dir, args.search_url)
    for spec, variable_id, url, size, dest in plan:
        print(f"  {spec.channel:9s} {variable_id:40s} {_human(size):>10s} "
              f"-> {dest.relative_to(args.out_dir)}")
    total = sum(s for _, _, _, s, _ in plan)
    print(f"\nPlan: {len(plan)} files, {_human(total)} total")
    if missing:
        # A channel is INCOMPLETE if any of its variables did not resolve — a
        # partial channel (e.g. SST without SIC) must not read as success.
        print("\nWARNING: unresolved variables (channel is INCOMPLETE):",
              file=sys.stderr)
        for channel, variable_id in missing:
            print(f"  - {channel}/{variable_id}", file=sys.stderr)
        print("  Check the source_id/version against the ESGF catalog "
              "(--list) or pass --search-url for a live index node.",
              file=sys.stderr)
    if not plan:
        print("Nothing to fetch (no files resolved).", file=sys.stderr)
        return 1
    if args.dry_run:
        # Even a dry run reports non-zero if the plan is incomplete, so a
        # scripted caller cannot mistake a partial plan for a full one.
        return 1 if missing else 0

    print("\nDownloading ...")
    failures = 0
    for spec, variable_id, url, size, dest in plan:
        if not download_one(url, dest, size, force=args.force):
            failures += 1
    print(f"\nDone. {len(plan) - failures}/{len(plan)} files complete.")

    print("\nConsume the raw files with run_amip_cmip6_deck.py, e.g.:")
    for spec in specs:
        tag = "" if spec.directly_consumable else "  [needs adapter first]"
        print(f"  # {spec.channel}{tag}: {spec.deck_flags}")
        if not spec.directly_consumable:
            print(f"  #   adapter: {spec.adapter_note}")
    # Non-zero if any download failed OR any channel was incomplete.
    return 0 if (failures == 0 and not missing) else 1


if __name__ == "__main__":
    raise SystemExit(main())
