#!/usr/bin/env python
"""Build + run MITgcm ``tutorial_barotropic_gyre`` and archive its output as the
legoESM fidelity **oracle reference** for the ``barotropic_gyre`` case.

This is the Slice-2 generator behind
:mod:`legoesm.ocean.fidelity.mitgcm_runner`. MITgcm is a Fortran model run
out-of-process, so the oracle reference is produced *here* (on a machine with a
Fortran toolchain) and then loaded offline by the harness. The product is a
directory in the layout the runner expects::

    <ref-root>/barotropic_gyre/
        U.<iter>.{meta,data}  V.<iter>.{meta,data}  Eta.<iter>.{meta,data}
        XC.{meta,data}  YC.{meta,data}  RC.{meta,data}  DRF.{meta,data}
        Depth.{meta,data}  hFacC/W/S.{meta,data}
        output.txt                          # monitor log (for mitgcm_monitor)

Output-format note: MITgcm dumps state at the first and last step by default
(``dumpInitAndLast`` is on), but writes PER-TILE files (``U.<iter>.001.001.data``)
that the dependency-free reader does not glue. So this script sets
``globalFiles=.TRUE.`` (PARM01) to get single global ``U.<iter>.data`` files the
loader reads directly, and re-asserts ``dumpInitAndLast`` for safety — without
otherwise altering the physics. (The stock ``verification/.../results/`` ships
only ``output.txt``; the field files are simply not archived there, which is why
this regeneration step exists.)

Prerequisites (this script checks and fails LOUDLY if absent):
  * a Fortran compiler + MPI wrapper (``mpif90``/``gfortran``);
  * MITgcm source (``--mitgcm-root`` or ``$MITGCM_ROOTDIR``; the
    ``verification/tutorial_barotropic_gyre`` deck must be present).

Run::

    python scripts/data/generate_mitgcm_barotropic_gyre_reference.py \
        --mitgcm-root /path/to/MITgcm --ref-root $LEGOESM_OCEAN_FIDELITY_MITGCM_REF

Use ``--dry-run`` to print the plan (toolchain, paths, archive map) without
building. The pure helpers (``resolve_mitgcm_root``, ``patch_data_for_field_dumps``,
``archive_plan``) are unit-tested in
``tests/ocean/fidelity/test_generate_mitgcm_gyre_reference.py``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

CASE = "barotropic_gyre"
VERIFICATION_SUBPATH = Path("verification") / "tutorial_barotropic_gyre"

# Field dumps (per iteration) + time-independent grid descriptors to archive.
PROGNOSTIC_FIELDS = ("U", "V", "Eta")
GRID_FIELDS = (
    "XC", "YC", "RC", "DRF", "DXG", "DYG", "RAC", "Depth",
    "hFacC", "hFacW", "hFacS",
)

# Candidate roots probed when neither --mitgcm-root nor $MITGCM_ROOTDIR is set.
_FALLBACK_ROOTS = (
    "/swot/SUM01/spencer/MITgcm",
    "/swot/SUM05/takaya/MITgcm",
)


class ToolchainError(RuntimeError):
    """Raised when the Fortran build prerequisites are missing."""


def resolve_mitgcm_root(explicit: str | None) -> Path:
    """Resolve the MITgcm source root, validating the gyre deck is present.

    Order: ``explicit`` (``--mitgcm-root``) -> ``$MITGCM_ROOTDIR`` -> known
    fallback trees. Raises ``FileNotFoundError`` if none contains the
    ``verification/tutorial_barotropic_gyre`` deck.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("MITGCM_ROOTDIR")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(Path(p) for p in _FALLBACK_ROOTS)

    for root in candidates:
        if (root / VERIFICATION_SUBPATH / "input" / "data").is_file():
            return root
    raise FileNotFoundError(
        "Could not locate a MITgcm source tree with "
        f"{VERIFICATION_SUBPATH}/input/data. Pass --mitgcm-root or set "
        f"$MITGCM_ROOTDIR. Tried: {[str(c) for c in candidates]}"
    )


def check_fortran_toolchain() -> str:
    """Return the first available Fortran/MPI compiler, else raise ToolchainError."""
    for tool in ("mpif90", "mpifort", "gfortran", "ifort"):
        path = shutil.which(tool)
        if path:
            return path
    raise ToolchainError(
        "No Fortran compiler found (looked for mpif90/mpifort/gfortran/ifort). "
        "MITgcm reference generation requires building the Fortran model — run "
        "this on a machine with a Fortran toolchain (e.g. a cluster node), then "
        "point the harness at the produced reference via "
        "$LEGOESM_OCEAN_FIDELITY_MITGCM_REF."
    )


def _insert_into_namelist(text: str, namelist: str, line: str, *, key: str) -> str:
    """Insert ``line`` just after the ``&<namelist>`` opener, if ``key`` absent.

    Idempotent (guarded by ``key not in text``). Raises if the namelist is
    missing so a malformed deck fails loudly instead of silently no-op'ing.
    """
    if key in text:
        return text
    idx = text.find(f"&{namelist}")
    if idx == -1:
        raise ValueError(f"input/data has no &{namelist} namelist to patch")
    insert_at = text.find("\n", idx) + 1
    return text[:insert_at] + line + text[insert_at:]


def patch_data_for_field_dumps(data_text: str, *, n_timesteps: int | None = None) -> str:
    """Patch an ``input/data`` namelist so the dumps land where the loader reads.

    The crucial change is ``globalFiles=.TRUE.`` in ``PARM01``: by default MITgcm
    writes per-tile files (``U.<iter>.001.001.data``), which the dependency-free
    :mod:`legoesm.ocean.fidelity.mitgcm_io` reader does not glue — globalFiles
    makes it emit single global ``U.<iter>.data`` files instead. ``dumpInitAndLast``
    (already a MITgcm default) is set explicitly as belt-and-suspenders so the
    state IS dumped at the last step. The physics namelists (PARM01 dynamics,
    PARM04/05) are otherwise untouched — this changes ONLY I/O. Optionally
    overrides ``nTimeSteps`` (anchored to the line start so a commented
    ``# nTimeSteps=...`` is never edited).
    """
    data_text = _insert_into_namelist(
        data_text, "PARM01", " globalFiles=.TRUE.,\n", key="globalFiles"
    )
    data_text = _insert_into_namelist(
        data_text, "PARM03", " dumpInitAndLast=.TRUE.,\n", key="dumpInitAndLast"
    )
    if n_timesteps is not None:
        import re

        data_text = re.sub(
            r"(?m)^(\s*)nTimeSteps\s*=\s*\d+",
            rf"\g<1>nTimeSteps={int(n_timesteps)}",
            data_text,
            count=1,
        )
    return data_text


def archive_plan(run_dir: Path, ref_dir: Path, iteration: int) -> list[tuple[Path, Path]]:
    """Pure mapping of MITgcm run-dir outputs -> reference-archive destinations.

    Returns ``(src, dst)`` pairs for the field dumps at ``iteration``, the grid
    descriptors, and the monitor log. Callers copy only the pairs whose ``src``
    exists (a case may not emit every grid descriptor).
    """
    pairs: list[tuple[Path, Path]] = []
    suffix = f"{iteration:010d}"
    for field in PROGNOSTIC_FIELDS:
        for ext in (".meta", ".data"):
            name = f"{field}.{suffix}{ext}"
            pairs.append((run_dir / name, ref_dir / name))
    for field in GRID_FIELDS:
        for ext in (".meta", ".data"):
            name = f"{field}{ext}"
            pairs.append((run_dir / name, ref_dir / name))
    pairs.append((run_dir / "output.txt", ref_dir / "output.txt"))
    return pairs


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"  $ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def build_and_run(
    root: Path, work: Path, *, n_timesteps: int | None,
    optfile: str | None = None, jobs: int = 4,
) -> Path:
    """genmake2 + make + run the gyre in ``work``; return the run directory.

    ``optfile`` is passed to ``genmake2 -of`` (REQUIRED for a modern GNU
    toolchain: gcc>=14 makes implicit C declarations an error and gfortran>=10
    rejects argument-mismatch, so legacy MITgcm needs an optfile adding
    ``-std=gnu89 -Wno-implicit-function-declaration`` to CFLAGS and
    ``-fallow-argument-mismatch -fallow-invalid-boz`` to FFLAGS).
    """
    deck = root / VERIFICATION_SUBPATH
    build = work / "build"
    run = work / "run"
    build.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)

    genmake2 = root / "tools" / "genmake2"
    genmake_cmd = [str(genmake2), "-mods", str(deck / "code"), "-rootdir", str(root)]
    if optfile:
        genmake_cmd += ["-of", str(optfile)]
    _run(genmake_cmd, cwd=build)
    _run(["make", "depend"], cwd=build)
    _run(["make", "-j", str(int(jobs))], cwd=build)

    # Stage inputs, patch the deck for field dumps, link the executable.
    for item in (deck / "input").iterdir():
        shutil.copy(item, run / item.name)
    data_path = run / "data"
    data_path.write_text(
        patch_data_for_field_dumps(data_path.read_text(), n_timesteps=n_timesteps)
    )
    shutil.copy(build / "mitgcmuv", run / "mitgcmuv")
    with open(run / "output.txt", "w") as out:
        subprocess.run([str(run / "mitgcmuv")], cwd=str(run), check=True, stdout=out)
    return run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mitgcm-root", default=None, help="MITgcm source root")
    ap.add_argument(
        "--ref-root", default=os.environ.get("LEGOESM_OCEAN_FIDELITY_MITGCM_REF"),
        help="reference archive root (default $LEGOESM_OCEAN_FIDELITY_MITGCM_REF)",
    )
    ap.add_argument("--work", default=None, help="scratch build/run dir")
    ap.add_argument("--n-timesteps", type=int, default=None, help="override nTimeSteps")
    ap.add_argument(
        "--optfile", default=None,
        help="genmake2 -of build-options file (needed for modern gcc/gfortran)",
    )
    ap.add_argument("--jobs", type=int, default=4, help="make -j parallelism")
    ap.add_argument("--dry-run", action="store_true", help="print plan, do not build")
    args = ap.parse_args(argv)

    root = resolve_mitgcm_root(args.mitgcm_root)
    print(f"MITgcm root: {root}")
    if not args.ref_root:
        print("ERROR: set --ref-root or $LEGOESM_OCEAN_FIDELITY_MITGCM_REF", file=sys.stderr)
        return 2
    ref_dir = Path(args.ref_root).expanduser() / CASE
    work = Path(args.work).expanduser() if args.work else ref_dir / "_work"

    if args.dry_run:
        try:
            tool = check_fortran_toolchain()
            print(f"Fortran toolchain: {tool}")
        except ToolchainError as exc:
            print(f"Fortran toolchain: MISSING ({exc})")
        print(f"Reference dir: {ref_dir}")
        print(f"Scratch dir:   {work}")
        print("Would archive (last iteration, paths relative to run dir):")
        for src, dst in archive_plan(Path("<run>"), ref_dir, iteration=10):
            print(f"  {src.name} -> {dst}")
        return 0

    check_fortran_toolchain()
    run = build_and_run(
        root, work, n_timesteps=args.n_timesteps,
        optfile=args.optfile, jobs=args.jobs,
    )

    # Discover the last iteration MITgcm dumped (from U.<iter>.meta).
    iters = sorted(
        int(p.name[len("U.") : -len(".meta")])
        for p in run.glob("U.*.meta")
        if p.name[len("U.") : -len(".meta")].isdigit()
    )
    if not iters:
        print("ERROR: MITgcm produced no U.<iter> field dumps", file=sys.stderr)
        return 1
    last = iters[-1]
    ref_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src, dst in archive_plan(run, ref_dir, iteration=last):
        if src.exists():
            shutil.copy(src, dst)
            copied += 1
    print(f"Archived {copied} files for iteration {last} -> {ref_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
