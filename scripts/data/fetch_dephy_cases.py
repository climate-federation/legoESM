"""Download the published DEPHY-format SCM case files this repo can run.

The field-campaign cases (Sandu & Stevens' stratocumulus-to-cumulus transition,
M-PACE, COMBLE, the ARM SGP cumulus case) are distributed by the groups that
built them, and this script fetches them rather than vendoring copies:
provenance stays with the publisher and the 36 MB transition files stay out of
the history.

Sources are PINNED, to an upstream commit and to the SHA-256 of the bytes this
repo was tested against, so picking up an upstream update is a deliberate act:
the fetch fails on a mismatch, and the new file has to be checked against the
case's reference before the digest in the registry is changed. A forcing file
that changes under a fixed case name is an uncontrolled comparison.

Files land in the SAME cache the gSAM decks use, under a ``dephy/``
subdirectory::

    data/les_cases/dephy/<file>.nc

so ``$LEGOESM_LES_FORCING`` moves every forcing file this repo reads at once.
The case registry itself -- URL, reference, which programme row each case is --
lives in :mod:`legoesm.atmosphere.forcing.scm.dephy_scm`; this script only
moves bytes.

Usage
-----
.. code-block:: bash

    python scripts/data/fetch_dephy_cases.py --list
    python scripts/data/fetch_dephy_cases.py --only mpace --only comble
    python scripts/data/fetch_dephy_cases.py            # everything

Needs outbound HTTPS. On a machine without it, fetch elsewhere and copy the
files into the cache directory printed by ``--list``.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from legoesm.atmosphere.forcing.scm.dephy_scm import (  # noqa: E402
    DEPHY_SCM_CASES,
    resolve_dephy_case_path,
)

_TIMEOUT_S = 120
# A truncated NetCDF still opens far enough to give a confusing error deep in
# the loader, so a download is written to a temporary name and moved into place
# only once it is complete.
_PARTIAL_SUFFIX = ".partial"


def _cache_dir() -> Path:
    """Directory the case files are written to (from the registry, not guessed)."""
    any_case = next(iter(DEPHY_SCM_CASES))
    return Path(resolve_dephy_case_path(any_case)).parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected_sha: str = "") -> int:
    tmp = dest.with_name(dest.name + _PARTIAL_SUFFIX)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
            # Not every opener reports a status (file:// does not), so only
            # a status that is present and not 200 is a failure.
            status = getattr(response, "status", None)
            if status is not None and status != 200:
                raise urllib.error.HTTPError(
                    url, status, "unexpected status", response.headers, None)
            with tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    size = tmp.stat().st_size
    if size == 0:
        tmp.unlink(missing_ok=True)
        raise OSError(f"{url} returned an empty file")
    # A 200 response is not proof the bytes are the case: a proxy error page,
    # an LFS pointer or an upstream edit all arrive as a perfectly valid
    # download. Compare against the digest the repo was tested with, and never
    # move a mismatched file into the cache.
    if expected_sha:
        got = _sha256(tmp)
        if got != expected_sha:
            tmp.unlink(missing_ok=True)
            raise OSError(
                f"{url}\n  checksum mismatch: expected {expected_sha}, got "
                f"{got} ({size} B). The upstream file has changed. Verify the "
                "new file against the case's reference before updating the "
                "sha256 in DEPHY_SCM_CASES -- a silently different forcing is "
                "a different experiment under the same case name."
            )
    tmp.replace(dest)
    return size


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--only", action="append", default=None, metavar="CASE",
                   help="fetch just this case; repeatable. Default: all.")
    p.add_argument("--list", action="store_true",
                   help="print the registry and where each file would go, "
                        "and download nothing.")
    p.add_argument("--force", action="store_true",
                   help="re-download a case that is already cached.")
    args = p.parse_args(argv)

    names = sorted(DEPHY_SCM_CASES)
    if args.only:
        unknown = sorted(set(args.only) - set(names))
        if unknown:
            # A selector that matches nothing is a hard error, never a silent
            # no-op: a typo would otherwise "succeed" having fetched nothing.
            raise SystemExit(
                f"unknown case(s) {unknown}; choose from {names}")
        names = [n for n in names if n in set(args.only)]

    if args.list:
        print(f"cache: {_cache_dir()}")
        for name in names:
            spec = DEPHY_SCM_CASES[name]
            here = Path(resolve_dephy_case_path(name))
            state = "present" if here.is_file() else "MISSING"
            print(f"\n{name}  [{state}]")
            print(f"  regime    : {spec.regime}")
            print(f"  campaign  : {spec.campaign}")
            print(f"  reference : {spec.reference}")
            print(f"  url       : {spec.url}")
            print(f"  sha256    : {spec.sha256 or '(none recorded)'}")
            print(f"  path      : {here}")
        return 0

    failures: list[str] = []
    for name in names:
        spec = DEPHY_SCM_CASES[name]
        dest = Path(resolve_dephy_case_path(name))
        if dest.is_file() and not args.force:
            if spec.sha256 and _sha256(dest) != spec.sha256:
                print(f"[BAD ] {name}: cached file does not match the "
                      f"registry checksum; re-run with --force")
                failures.append(name)
                continue
            print(f"[skip] {name}: already cached ({dest.stat().st_size} B)")
            continue
        print(f"[get ] {name}: {spec.url}")
        try:
            size = _download(spec.url, dest, spec.sha256)
        except Exception as exc:                       # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            failures.append(name)
            continue
        print(f"[ok  ] {name}: {size} B -> {dest}")

    if failures:
        print(f"\n{len(failures)} case(s) failed: {sorted(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
