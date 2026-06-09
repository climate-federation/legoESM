#!/usr/bin/env python
"""Install one (or a few) legoESM federation members under **plain pip**.

Why this exists
---------------
Post-carve (see ``FEDERATION.md``) legoESM ships as a uv workspace of members
(``legoesm-core``, ``legoesm-atmosphere``, ``legoesm-ocean``, ...).  Each member
declares its inter-member deps as ordinary distributions::

    dependencies = ["legoesm-core~=0.1.0", ...]

and those resolve to the in-tree source **only via** ``[tool.uv.sources]``
(``workspace = true``).  That table is understood by ``uv`` alone — plain ``pip``
ignores it and goes looking on PyPI, where the members are not published yet::

    ERROR: Could not find a version that satisfies the requirement
    legoesm-core~=0.1.0 (from legoesm-atmosphere) ... (from versions: none)

So ``pip install ./packages/atmosphere`` (and even the documented root
``pip install -e ".[dev]"``) fail for pip users without uv.  This script gives pip
users a working install path by resolving the member dependency DAG locally —
either as editable installs or from a freshly built local wheelhouse — instead of
the absent PyPI index.

Usage
-----
Install one Earth-system component standalone (editable; pulls only its core)::

    python scripts/experiment/install_federation.py atmosphere
    python scripts/experiment/install_federation.py ocean land ice

Install with an extra (e.g. the SFNO neural cores, which reach into ml)::

    python scripts/experiment/install_federation.py atmosphere --extras ml

Install the whole stack (every member + the root meta package), the pip
equivalent of ``uv sync`` / the documented ``pip install -e ".[dev]"``::

    python scripts/experiment/install_federation.py --all --extras dev

Build a non-editable local wheelhouse and install from it (no ``-e``)::

    python scripts/experiment/install_federation.py atmosphere --wheels

Just print the pip command(s) without running them::

    python scripts/experiment/install_federation.py atmosphere --dry-run

Note
----
If you have ``uv`` installed, ``uv sync`` (whole workspace) or
``uv pip install --package legoesm-atmosphere`` is simpler and needs no
wheelhouse — this script is the pip-only fallback.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKGS = REPO / "packages"


def _canon(name: str) -> str:
    """PEP 503/685 canonical form: collapse ``[-_.]`` runs to ``-`` and lowercase."""
    return re.sub(r"[-_.]+", "-", name).lower()

#: Folder names of the workspace members (``packages/<name>``); the distribution
#: is ``legoesm-<name>``.  ``core`` first so the printed command is substrate-led;
#: the rest in directory order.  Discovered from disk so adding a member needs no
#: edit here — only its ``pyproject.toml`` (parsed below) defines its deps.
ALL_MEMBERS = ("core",) + tuple(
    sorted(p.name for p in PKGS.iterdir() if (p / "pyproject.toml").is_file() and p.name != "core")
)
#: A few aliases so users can say what they mean.
ALIASES = {
    "legoesm": "core", "cryosphere": "ice", "sea-ice": "ice", "seaice": "ice",
    "atm": "atmosphere", "atmos": "atmosphere", "training": "ml", "neural": "ml",
}
for _m in ALL_MEMBERS:
    ALIASES[f"legoesm-{_m}"] = _m  # accept distribution names too


def _member_pyproject(member: str) -> dict:
    with (PKGS / member / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _dist_to_member(spec: str) -> str | None:
    """``"legoesm-core~=0.1.0"`` -> ``"core"`` (a member), else ``None``.

    Distribution names are matched PEP 503/685 canonically, so an underscore or
    dotted spelling (``legoesm_core``) still resolves to the federation edge
    instead of being missed and sent to PyPI.
    """
    name = spec.split(";", 1)[0].strip()
    for sep in ("~=", "==", ">=", "<=", "!=", ">", "<", "=", "[", " "):
        name = name.split(sep)[0]
    canon = _canon(name.strip())
    if canon.startswith("legoesm-"):
        m = canon.removeprefix("legoesm-")
        if m in ALL_MEMBERS:
            return m
    return None


def member_deps(member: str, extras: tuple[str, ...] = ()) -> set[str]:
    """The OTHER members ``member`` pulls — hard deps plus the named extras' deps.

    Read straight from the member's ``pyproject.toml`` so the federation DAG can
    never drift from what is actually shipped (FEDERATION.md).  ``extras`` are the
    optional-dependency groups requested for THIS member (e.g. atmosphere[ml] pulls
    legoesm-ml); transitive members are pulled with hard deps only, matching pip's
    "extras apply to named specs only" rule.
    """
    proj = _member_pyproject(member).get("project", {})
    specs = list(proj.get("dependencies", []))
    # Canonicalize the extra-group keys (PEP 685) so a canonical ``ex`` resolves
    # regardless of how the group was spelled in the manifest.
    opt = {_canon(k): v for k, v in proj.get("optional-dependencies", {}).items()}
    for ex in extras:
        specs += list(opt.get(_canon(ex), []))
    out = {m for s in specs if (m := _dist_to_member(s)) and m != member}
    return out


def _project(member: str) -> dict:
    """``[project]`` table of a member (short name) or the root meta (``"meta"``)."""
    if member == "meta":
        with (REPO / "pyproject.toml").open("rb") as fh:
            return tomllib.load(fh).get("project", {})
    return _member_pyproject(member).get("project", {})


def declared_extras(member: str) -> set[str]:
    """The optional-dependency group names a member (or ``"meta"``) defines.

    Returned in PEP 685 canonical form so matching against user-supplied extras is
    spelling-insensitive (``ML`` / ``m_l`` both match a declared ``ml``).
    """
    return {_canon(e) for e in _project(member).get("optional-dependencies", {})}


def _normalize(name: str) -> str:
    key = _canon(name.strip())
    key = ALIASES.get(key, key)
    if key not in ALL_MEMBERS:
        raise SystemExit(
            f"unknown member {name!r}; choose from {', '.join(ALL_MEMBERS)} "
            f"(aliases: {', '.join(sorted(ALIASES))})"
        )
    return key


def transitive_closure(
    members: list[str], extras_for: dict[str, tuple[str, ...]] | None = None
) -> list[str]:
    """All members that must be installed together to satisfy ``members``' deps.

    A requested member contributes both its hard deps AND the deps of the extras
    requested for it (``extras_for``); everything pulled transitively contributes
    only its hard deps.  Returned core-first / canonical order so the command is
    deterministic (pip itself solves the whole set together, order-independent).
    """
    extras_for = extras_for or {}
    seen: set[str] = set()
    stack = list(members)
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        # Extras only expand the originally-requested members, never transitively.
        ex = extras_for.get(m, ()) if m in members else ()
        stack.extend(member_deps(m, ex))
    return [m for m in ALL_MEMBERS if m in seen]


def _spec(member: str, extras: tuple[str, ...], path: Path) -> str:
    """A pip target ``path[extras]`` — extras only attach to the requested member.

    (Each member declares its own extras; ``--extras ml`` on ``atmosphere`` must
    not be forwarded to ``core``, which has no ``ml`` extra.)
    """
    return f"{path}[{','.join(extras)}]" if extras else str(path)


def build_editable_cmd(
    members: list[str], extras_for: dict[str, tuple[str, ...]], include_meta: bool
) -> list[str]:
    cmd = [sys.executable, "-m", "pip", "install"]
    for m in members:
        cmd += ["-e", _spec(m, extras_for.get(m, ()), REPO / "packages" / m)]
    if include_meta:
        cmd += ["-e", _spec("meta", extras_for.get("meta", ()), REPO)]
    return cmd


def build_wheels(members: list[str], outdir: Path, include_meta: bool = False) -> None:
    try:
        import build  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(
            "--wheels needs the build frontend: pip install build hatchling"
        )
    # The root meta wheel must be built too when requested (--all), or a wheelhouse
    # install would silently omit ``legoesm`` (and its dev/all extras + console
    # entry point) even though pip can resolve every member from the wheelhouse.
    targets = [(m, REPO / "packages" / m) for m in members]
    if include_meta:
        targets.append(("meta (legoesm)", REPO))
    for label, src in targets:
        r = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--no-isolation",
             "--outdir", str(outdir), str(src)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise SystemExit(f"building legoesm-{label} failed:\n{r.stdout}\n{r.stderr}")
        print(f"  built legoesm-{label}" if label != "meta (legoesm)" else "  built legoesm (meta)")


def build_wheelhouse_cmd(
    members: list[str], extras_for: dict[str, tuple[str, ...]], wheelhouse: Path,
    include_meta: bool = False,
) -> list[str]:
    # ``--find-links`` makes the local wheelhouse an index, so pip resolves the
    # ``legoesm-*`` inter-member deps from it (and everything else from PyPI).
    cmd = [sys.executable, "-m", "pip", "install", "--find-links", str(wheelhouse)]
    for m in members:
        ex = extras_for.get(m, ())
        cmd.append(f"legoesm-{m}[{','.join(ex)}]" if ex else f"legoesm-{m}")
    if include_meta:
        ex = extras_for.get("meta", ())
        cmd.append(f"legoesm[{','.join(ex)}]" if ex else "legoesm")
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Install legoESM federation members under plain pip "
                    "(resolves the inter-member dependency DAG locally).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "members", nargs="*",
        help="members to install (e.g. atmosphere ocean land ice coupler ml tools); "
             "their dependency members are pulled automatically",
    )
    ap.add_argument("--all", action="store_true",
                    help="install every member plus the root meta package")
    ap.add_argument("--extras", default="",
                    help="comma-separated extras attached to the requested member(s) "
                         "(e.g. ml, dev, viz)")
    ap.add_argument("--wheels", action="store_true",
                    help="build a local wheelhouse and install from it (non-editable) "
                         "instead of editable installs")
    ap.add_argument("--wheelhouse", type=Path, default=None,
                    help="where to put built wheels (default: a temp dir; "
                         "implies --wheels)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the pip command(s) without running them")
    args = ap.parse_args(argv)

    if args.wheelhouse is not None:
        args.wheels = True

    # Canonicalize user extras (PEP 685) so matching + the emitted spec are
    # spelling-insensitive and always valid for pip.
    extras = tuple(_canon(e.strip()) for e in args.extras.split(",") if e.strip())

    # Extras attach to the requested members.  For --all the natural target is the
    # root meta package (which carries the aggregate extras: dev, all, viz, ...) —
    # the component members only define narrow extras (e.g. atmosphere[ml]).
    if args.all:
        requested = list(ALL_MEMBERS)
        # Route --all extras to the meta package only (it carries the aggregate
        # dev/all/viz extras); the closure is every member regardless.
        raw_extras_for = {"meta": extras} if extras else {}
        include_meta = True
    else:
        if not args.members:
            ap.error("name at least one member, or pass --all")
        requested = [_normalize(m) for m in args.members]
        raw_extras_for = {m: extras for m in requested} if extras else {}
        include_meta = False

    # Attach an extra to a target ONLY if that target actually declares it —
    # otherwise pip silently warns "does not provide the extra" and the user's
    # intent (e.g. atmosphere[ml]) is quietly dropped.  Warn loudly instead, and
    # error if a requested extra is declared by NONE of the targets (a typo).
    extras_for: dict[str, tuple[str, ...]] = {}
    for tgt, exs in raw_extras_for.items():
        keep = tuple(e for e in exs if e in declared_extras(tgt))
        dropped = [e for e in exs if e not in keep]
        if dropped:
            print(f"warning: legoesm-{tgt} does not declare extra(s) "
                  f"{', '.join(dropped)} — skipping for it", file=sys.stderr)
        if keep:
            extras_for[tgt] = keep
    if extras and not extras_for:
        ap.error(f"none of the requested target(s) declare extra(s) "
                 f"{', '.join(extras)}; nothing to attach")

    # Closure must account for member deps pulled IN by the requested extras (e.g.
    # atmosphere[ml] needs legoesm-ml installed too), else the editable build emits
    # ``atmosphere[ml]`` whose ``legoesm-ml~=0.1.0`` has no source.  --all is the
    # whole set already.
    members = list(ALL_MEMBERS) if args.all else transitive_closure(requested, extras_for)

    pulled = [m for m in members if m not in requested]
    print(f"requested: {', '.join('legoesm-' + m for m in requested)}"
          + (" + root meta (legoesm)" if include_meta else ""))
    if pulled:
        print(f"pulled as dependencies: {', '.join('legoesm-' + m for m in pulled)}")

    if args.wheels:
        wheel_targets = [f"legoesm-{m}" for m in members] + (
            ["legoesm (meta)"] if include_meta else [])
        if args.dry_run:
            # Never build (a side effect) on a dry run — just show what WOULD run.
            placeholder = args.wheelhouse or Path("<temp-wheelhouse>")
            cmd = build_wheelhouse_cmd(requested, extras_for, placeholder,
                                       include_meta=include_meta)
            print(f"(dry-run) would build wheels: {', '.join(wheel_targets)}")
            print("+ " + " ".join(cmd))
            return 0
        wh_ctx: tempfile.TemporaryDirectory | None = None
        if args.wheelhouse is not None:
            wheelhouse = args.wheelhouse
            wheelhouse.mkdir(parents=True, exist_ok=True)
        else:
            wh_ctx = tempfile.TemporaryDirectory()
            wheelhouse = Path(wh_ctx.name)
        try:
            print(f"building wheels into {wheelhouse} ...")
            build_wheels(members, wheelhouse, include_meta=include_meta)
            cmd = build_wheelhouse_cmd(requested, extras_for, wheelhouse,
                                       include_meta=include_meta)
            print("+ " + " ".join(cmd))
            return subprocess.run(cmd).returncode
        finally:
            if wh_ctx is not None:
                wh_ctx.cleanup()
    else:
        cmd = build_editable_cmd(members, extras_for, include_meta)
        print("+ " + " ".join(cmd))
        if args.dry_run:
            return 0
        return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
