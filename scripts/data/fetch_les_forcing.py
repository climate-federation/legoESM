#!/usr/bin/env python
"""Copy the SAM/gSAM forcing decks for legoESM's LES/CRM cases into a local,
repo-relative cache (``data/les_cases/<CASE>/``) so the cases run without an
external gSAM checkout or a per-machine ``LEGOESM_GSAM_ROOT``.

The decks are third-party SAM input (Khairoutdinov's gSAM ``CASES/``); they are
NOT committed (``data/`` is gitignored). This script is the reproducible way to
populate the cache from any gSAM checkout — point it at the checkout root (the
directory containing ``CASES/``) via ``--gsam-root`` or ``$LEGOESM_GSAM_ROOT``.

``legoesm.atmosphere.forcing.sam_case_forcing.resolve_sam_case_dir`` falls back to this
cache, so once populated the gSAM drivers (BOMEX/RICO/DYCOMS/GATE/LBA) find
their decks with no flag.

The 3 dry-PBL cases (GABLS1, Wangara, Ekman) are ANALYTIC (in-code profiles via
``run_les_plane.py --case``); they need no deck and are intentionally absent.

Usage
-----
    python scripts/data/fetch_les_forcing.py            # auto-detect gSAM root
    python scripts/data/fetch_les_forcing.py --gsam-root /path/to/gSAM1.8.7
    LEGOESM_GSAM_ROOT=/path/to/gSAM1.8.7 python scripts/data/fetch_les_forcing.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# legoESM LES/CRM case -> gSAM CASES/<dir>. Driver-backed cases first, then the
# decks that have no legoESM driver yet (kept local for future integration).
CASE_TO_GSAM_DIR: dict[str, tuple[str, ...]] = {
    "BOMEX": ("BOMEX",),                          # run_bomex_les.py
    "RICO": ("RICO",),                            # run_rico_les.py
    "DYCOMSII": ("DYCOMS_RF01", "DYCOMS_RF02"),   # run_dycoms_les.py (RF01) + RF02
    "GATE": ("GATE_IDEAL",),                      # run_gate_plane.py
    "LBA": ("LBA",),                              # run_lba_plane.py
    "ARM97": ("ARM9707",),                        # no driver yet
    "ASTEX": ("ASTEX209",),                       # no driver yet
    "TOGA": ("TOGA",),                            # no driver yet (TOGA_LONG excluded: 14 MB)
}

# Analytic dry-PBL cases — no deck, documented so the script is self-describing.
ANALYTIC_CASES = ("GABLS1", "Wangara", "Ekman")

# Repo root = two levels up from scripts/data/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEST_ROOT = _REPO_ROOT / "data" / "les_cases"

# Candidate gSAM roots (the dir that CONTAINS ``CASES/``) when neither
# --gsam-root nor $LEGOESM_GSAM_ROOT is given.
_AUTODETECT = (
    _REPO_ROOT.parent / "Code" / "gSAM" / "gSAM1.8.7",
    Path.home() / "Documents" / "Code" / "gSAM" / "gSAM1.8.7",
)

# Skip Fortran source and oversized binary dumps (e.g. TOGA_LONG's 14 MB files).
_MAX_FILE_BYTES = 2 * 1024 * 1024


def _resolve_gsam_root(arg: str | None) -> Path:
    # An EXPLICIT --gsam-root must win or fail — never silently fall back to a
    # different source.
    if arg:
        root = Path(arg).expanduser()
        if (root / "CASES").is_dir():
            return root
        raise SystemExit(
            f"--gsam-root {root} has no CASES/ subdirectory.")
    candidates: list[Path] = []
    env = os.environ.get("LEGOESM_GSAM_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(_AUTODETECT)
    for c in candidates:
        if (c / "CASES").is_dir():
            return c
    raise SystemExit(
        "Could not find a gSAM checkout (a directory containing CASES/). "
        "Pass --gsam-root <path> or set LEGOESM_GSAM_ROOT. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def copy_deck(src_dir: Path, dest_dir: Path) -> tuple[int, int]:
    """Copy a case deck, skipping Fortran source (``*.f``) and files > 2 MB.

    Returns ``(files_copied, bytes_copied)``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    n, nbytes = 0, 0
    for f in sorted(src_dir.iterdir()):
        if (not f.is_file() or f.suffix == ".f"
                or f.stat().st_size > _MAX_FILE_BYTES):
            continue
        shutil.copy2(f, dest_dir / f.name)
        n += 1
        nbytes += f.stat().st_size
    return n, nbytes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gsam-root", default=None,
                   help="gSAM checkout root (dir containing CASES/). "
                        "Default: $LEGOESM_GSAM_ROOT then auto-detect.")
    p.add_argument("--dest", default=str(DEST_ROOT),
                   help=f"Destination cache (default: {DEST_ROOT}).")
    p.add_argument("--only", default=None,
                   help="Comma-separated case names to fetch (default: all).")
    args = p.parse_args(argv)

    gsam_root = _resolve_gsam_root(args.gsam_root)
    if args.only:
        wanted = args.only.split(",")
        unknown = [c for c in wanted if c not in CASE_TO_GSAM_DIR]
        if unknown:
            raise SystemExit(
                f"Unknown case(s): {unknown}. Known: "
                f"{sorted(CASE_TO_GSAM_DIR)} (analytic, no deck: {ANALYTIC_CASES})")
        cases = {k: CASE_TO_GSAM_DIR[k] for k in wanted}
    else:
        cases = CASE_TO_GSAM_DIR
    dest_root = Path(args.dest).expanduser()

    print(f"gSAM root: {gsam_root}")
    print(f"dest:      {dest_root}")
    total_files, total_bytes, missing = 0, 0, []
    for case, gsam_dirs in cases.items():
        for gdir in gsam_dirs:
            src = gsam_root / "CASES" / gdir
            if not src.is_dir():
                missing.append(f"{case} -> CASES/{gdir}")
                continue
            n, b = copy_deck(src, dest_root / gdir)
            total_files += n
            total_bytes += b
            print(f"  {case:9s} CASES/{gdir:14s} -> {dest_root / gdir}  "
                  f"({n} files, {b / 1024:.0f} KB)")

    print(f"copied {total_files} files, {total_bytes / 1024:.0f} KB total")
    print(f"analytic (no deck, in-code): {', '.join(ANALYTIC_CASES)}")
    if missing:
        print("MISSING in this gSAM checkout: " + "; ".join(missing),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
