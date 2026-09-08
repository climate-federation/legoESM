"""Correct the CASE LABEL on an archived LES reference, against evidence.

Two production LES drivers had a ``--case-label`` that defaulted to the
literal name of their FIRST deck instead of following ``--case``, so a run of
the second deck was stamped with the first deck's name. ``legoesm.training``'s
reference loader then refused the directory -- correctly, because the label is
the only thing it has to confirm a reference belongs to the case it is being
built for, and the pairs are deliberately not cross-aliased.

The frames themselves are the right physics. Regenerating them costs 5.2 GPU
hours (ASTEX, measured) to change one string, and the question the re-run
would answer -- "is this really ASTEX?" -- is already answerable offline from
the frames. So this relabels, and it REFUSES to do so unless the content
matches the case being claimed.

What it checks before touching anything, per case: the run length, the domain
top, and the frame count. Those separate the confusable pairs decisively:
ASTEX is 6.00 h to 1995 m, DYCOMS is 4.00 h to 1496 m; Wangara is 8 h from
theta0 = 277 K, the Nieuwstadt CBL is 4 h from 300 K.

What it does:
  * refuses if the frames already carry the target label (nothing to do),
  * refuses if the evidence does not match the target case,
  * copies the whole directory to ``<dir>.pre-relabel`` first,
  * rewrites every frame with the corrected ``case`` and NOTHING else,
  * verifies every other array is bit-identical to the backup,
  * writes ``RELABELLED.json`` recording what changed, on what evidence, and
    the git SHA that did it -- a relabelled directory must never be
    indistinguishable from one that was always right.

Usage::

    python scripts/data/relabel_les_reference.py results/les_ref/astex astex
    python scripts/data/relabel_les_reference.py results/les_ref/astex astex \\
        --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

# (hours, domain top [m], n_frames) with the tolerance each is checked at.
# Deliberately NOT a "does it look plausible" test: these are the numbers that
# tell the confusable pair apart, and a case with no entry cannot be relabelled
# by this tool at all.
_EXPECTED: dict[str, dict] = {
    "astex": {"hours": 6.0, "z_top_m": 2000.0, "n_frames": 37,
              "not": "dycoms (4 h to 1500 m)"},
    "dycoms": {"hours": 4.0, "z_top_m": 1500.0, "n_frames": 25,
               "not": "astex (6 h to 2000 m)"},
    "wangara": {"hours": 8.0, "z_top_m": 2000.0, "n_frames": 49,
                "not": "cbl (4 h to 1600 m)"},
    "cbl": {"hours": 4.0, "z_top_m": 1600.0, "n_frames": 25,
            "not": "wangara (8 h to 2000 m)"},
}
_HOURS_TOL = 0.05          # h; frame times are exact to a timestep
_Z_TOL_FRAC = 0.02         # the top cell centre sits half a cell below the lid


def _frames(d: Path) -> list[Path]:
    fs = sorted(d.glob("profiles/prof_*.npz"))
    if not fs:
        raise SystemExit(f"no profile frames under {d}/profiles")
    return fs


def _evidence(frames: list[Path]) -> dict:
    first = np.load(frames[0], allow_pickle=True)
    last = np.load(frames[-1], allow_pickle=True)
    return {
        "n_frames": len(frames),
        "hours": float(np.asarray(last["t_hours"]).ravel()[-1]),
        "z_top_m": float(np.asarray(first["z"]).ravel().max()),
        "stamped": str(first["case"]),
    }


def _check(evidence: dict, case: str) -> list[str]:
    """Reasons the evidence does NOT support ``case``; empty means it does."""
    if case not in _EXPECTED:
        return [f"no signature registered for {case!r}; this tool relabels "
                f"only {sorted(_EXPECTED)}"]
    want = _EXPECTED[case]
    bad = []
    if abs(evidence["hours"] - want["hours"]) > _HOURS_TOL:
        bad.append(f"run length {evidence['hours']:.3f} h, expected "
                   f"{want['hours']} h for {case} (this looks like "
                   f"{want['not']})")
    if abs(evidence["z_top_m"] - want["z_top_m"]) > _Z_TOL_FRAC * want["z_top_m"]:
        bad.append(f"domain top {evidence['z_top_m']:.0f} m, expected "
                   f"~{want['z_top_m']:.0f} m for {case}")
    if evidence["n_frames"] != want["n_frames"]:
        bad.append(f"{evidence['n_frames']} frames, expected "
                   f"{want['n_frames']} for {case}")
    return bad


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return "unknown"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path, help="results/les_ref/<case>")
    p.add_argument("case", help="the case the frames REALLY are")
    p.add_argument("--apply", action="store_true",
                   help="without this the tool only reports; it never writes "
                        "by default, because it edits archived data.")
    args = p.parse_args(argv)

    frames = _frames(args.directory)
    ev = _evidence(frames)
    print(f"{args.directory}: {ev['n_frames']} frames, stamped "
          f"{ev['stamped']!r}, {ev['hours']:.3f} h, top {ev['z_top_m']:.0f} m")

    if ev["stamped"] == args.case:
        print(f"already stamped {args.case!r}; nothing to do")
        return 0
    bad = _check(ev, args.case)
    if bad:
        print(f"REFUSING to relabel as {args.case!r}:")
        for b in bad:
            print(f"  - {b}")
        return 2
    print(f"evidence supports {args.case!r} "
          f"(expected {_EXPECTED[args.case]['hours']} h, "
          f"~{_EXPECTED[args.case]['z_top_m']:.0f} m, "
          f"{_EXPECTED[args.case]['n_frames']} frames)")
    if not args.apply:
        print("dry run; pass --apply to rewrite the frames")
        return 0

    backup = args.directory.with_name(args.directory.name + ".pre-relabel")
    if backup.exists():
        raise SystemExit(
            f"{backup} already exists; refusing to overwrite a backup. Move "
            "or delete it once you are satisfied with the relabelled copy.")
    shutil.copytree(args.directory, backup)
    print(f"backed up to {backup}")

    for path in frames:
        payload = {k: v for k, v in np.load(path, allow_pickle=True).items()}
        payload["case"] = np.asarray(args.case)
        np.savez(path, **payload)

    # VERIFY, do not assume: every array except `case` must be bit-identical.
    for path in frames:
        new = np.load(path, allow_pickle=True)
        old = np.load(backup / "profiles" / path.name, allow_pickle=True)
        assert set(new.files) == set(old.files), f"{path.name} lost a key"
        for k in old.files:
            if k == "case":
                continue
            a, b = np.asarray(old[k]), np.asarray(new[k])
            if a.dtype.kind in "fc":
                same = np.array_equal(a, b, equal_nan=True)
            else:
                same = np.array_equal(a, b)
            if not same:
                raise SystemExit(
                    f"{path.name}: {k!r} CHANGED during relabelling; the "
                    f"backup at {backup} is the good copy.")
        if str(new["case"]) != args.case:
            raise SystemExit(f"{path.name}: label did not take")
    print(f"rewrote {len(frames)} frames; every other array bit-identical")

    (args.directory / "RELABELLED.json").write_text(json.dumps({
        "was": ev["stamped"],
        "now": args.case,
        "why": (
            "The production LES driver's --case-label defaulted to the name of "
            "its FIRST deck instead of following --case, so a run of the "
            "second deck carried the first deck's name. The frames are the "
            "right physics; only the label was wrong."
        ),
        "evidence": ev,
        "expected_for_case": _EXPECTED[args.case],
        "backup": str(backup),
        "git_sha": _git_sha(),
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.directory / 'RELABELLED.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
