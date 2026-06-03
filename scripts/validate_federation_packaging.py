#!/usr/bin/env python3
"""Full-CI federation packaging validation (wheel builds + root-absent install).

Proves the uv-workspace carve (FEDERATION.md) is actually shippable — not just
import-linter-clean in the dev tree — by BUILDING each member wheel and INSTALLING
a subset WITHOUT the root meta package, then asserting:

  1. every member wheel builds (hatchling namespace package);
  2. each wheel ships ONLY its own ``legoesm/<subpkg>`` (+ core's loose substrate
     modules) and NO ``legoesm/__init__.py`` (PEP-420 — sibling members must be
     able to contribute to the same namespace);
  3. the dependency DAG: every component wheel ``Requires-Dist: legoesm-core``,
     and ``legoesm-core`` requires NO other member (it is the root);
  4. ROOT-ABSENT install: ``legoesm-core`` + ``legoesm-ocean`` alone import and
     run standalone (``import legoesm.ocean`` + ``from legoesm import constants``),
     with the namespace merged across the two separately-built wheels;
  5. cross-member dycore sharing survives the carve: with ``legoesm-atmosphere``
     also installed, the ocean resolves the atmosphere FV3 SW dycore through the
     ``legoesm.sw_barotropics`` entry point — without importing the atmosphere.

Run from the repo root:  ``python scripts/validate_federation_packaging.py``
Exits non-zero (and prints FAIL) on the first broken invariant.  Intended for the
CI matrix; needs ``build`` + ``hatchling`` (``pip install build hatchling``).
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MEMBERS = (
    "legoesm-core",
    "legoesm-atmosphere",
    "legoesm-ocean",
    "legoesm-land",
    "legoesm-ice",
)
COMPONENTS = ("legoesm-atmosphere", "legoesm-ocean", "legoesm-land", "legoesm-ice")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"FAIL: {msg}")
    sys.exit(1)


def build_wheels(outdir: Path) -> dict[str, Path]:
    wheels: dict[str, Path] = {}
    for m in MEMBERS:
        r = _run([
            sys.executable, "-m", "build", "--wheel", "--no-isolation",
            "--outdir", str(outdir), str(REPO / "packages" / m),
        ])
        if r.returncode != 0:
            _fail(f"building {m}:\n{r.stdout}\n{r.stderr}")
        hits = sorted(outdir.glob(f"{m.replace('-', '_')}-*.whl"))
        if not hits:
            _fail(f"{m} produced no wheel")
        wheels[m] = hits[-1]
        print(f"  built {hits[-1].name}")
    return wheels


def check_wheel_contents(wheels: dict[str, Path]) -> None:
    for m, whl in wheels.items():
        names = zipfile.ZipFile(whl).namelist()
        if any(n == "legoesm/__init__.py" for n in names):
            _fail(f"{m} ships legoesm/__init__.py — breaks the PEP-420 namespace")
        subpkgs = {
            n.split("/")[1] for n in names
            if n.startswith("legoesm/") and "/" in n[len("legoesm/"):]
        }
        if m == "legoesm-core":
            if "ocean" in subpkgs or "atmosphere" in subpkgs:
                _fail("legoesm-core wheel leaks a component subpackage")
            if "constants.py" not in {Path(n).name for n in names}:
                _fail("legoesm-core wheel is missing the substrate constants.py")
        else:
            sub = m.split("-", 1)[1]
            stray = subpkgs - {sub}
            if stray:
                _fail(f"{m} wheel ships foreign subpackages {stray}")
            # A component wheel must ship NO loose top-level legoesm/*.py module —
            # the substrate's constants.py / thermo.py / registry.py / _version.py
            # belong to legoesm-core ONLY; a duplicate in a component wheel would
            # shadow/conflict in the merged namespace.
            loose = {
                n[len("legoesm/"):]
                for n in names
                if n.startswith("legoesm/")
                and "/" not in n[len("legoesm/"):]
                and n.endswith(".py")
            }
            if loose:
                _fail(f"{m} wheel ships loose substrate modules {loose} "
                      f"(those belong to legoesm-core only)")
    print("  wheel contents OK (namespace, isolation, no loose-module leakage)")


def check_dependency_dag(wheels: dict[str, Path]) -> None:
    def _requires(whl: Path) -> list[str]:
        for n in zipfile.ZipFile(whl).namelist():
            if n.endswith("METADATA"):
                txt = zipfile.ZipFile(whl).read(n).decode()
                return [
                    ln.split(":", 1)[1].strip()
                    for ln in txt.splitlines()
                    if ln.lower().startswith("requires-dist: legoesm-")
                ]
        return []

    for c in COMPONENTS:
        reqs = " ".join(_requires(wheels[c]))
        if "legoesm-core" not in reqs:
            _fail(f"{c} does not Requires-Dist legoesm-core")
    if any("legoesm-" in r for r in _requires(wheels["legoesm-core"])):
        _fail("legoesm-core depends on another member — it must be the DAG root")
    print("  dependency DAG OK (components -> core; core -> nothing)")


def _site_packages() -> str:
    return sysconfig.get_paths()["purelib"]


def check_root_absent_install(wheels: dict[str, Path], tmp: Path) -> None:
    target = tmp / "root_absent"
    r = _run([
        sys.executable, "-m", "pip", "install", "--quiet", "--target", str(target),
        "--no-deps", str(wheels["legoesm-core"]), str(wheels["legoesm-ocean"]),
    ])
    if r.returncode != 0:
        _fail(f"installing core+ocean wheels:\n{r.stderr}")
    code = (
        "import legoesm; assert legoesm.__file__ is None, 'not a namespace pkg'\n"
        "import legoesm.core, legoesm.ocean, legoesm.grids\n"
        "from legoesm import constants, thermo\n"
        "from legoesm.ocean.dynamics.barotropic_mpas import barotropic_substeps_mpas\n"
        "assert abs(constants.g - 9.80616) < 1e-6\n"
        "print('OK')\n"
    )
    env = {"PYTHONPATH": f"{target}:{_site_packages()}", "JAX_PLATFORMS": "cpu",
           "PATH": "/usr/bin:/bin"}
    r = _run([sys.executable, "-S", "-c", code], env=env)
    if r.returncode != 0 or "OK" not in r.stdout:
        _fail(f"root-absent core+ocean import:\n{r.stdout}\n{r.stderr}")
    print("  root-absent install OK (core+ocean import + run, no root, no atmosphere)")


def check_entry_point_sharing(wheels: dict[str, Path], tmp: Path) -> None:
    target = tmp / "ep_share"
    r = _run([
        sys.executable, "-m", "pip", "install", "--quiet", "--target", str(target),
        "--no-deps", str(wheels["legoesm-core"]), str(wheels["legoesm-atmosphere"]),
        str(wheels["legoesm-ocean"]),
    ])
    if r.returncode != 0:
        _fail(f"installing core+atmosphere+ocean:\n{r.stderr}")
    code = (
        "from legoesm.registry import SW_BAROTROPIC_REGISTRY\n"
        "fn = SW_BAROTROPIC_REGISTRY.get('fv3sw')\n"
        "assert 'atmosphere' in fn.__module__, fn.__module__\n"
        "print('OK')\n"
    )
    env = {"PYTHONPATH": f"{target}:{_site_packages()}", "JAX_PLATFORMS": "cpu",
           "PATH": "/usr/bin:/bin"}
    r = _run([sys.executable, "-S", "-c", code], env=env)
    if r.returncode != 0 or "OK" not in r.stdout:
        _fail(f"entry-point dycore sharing (root-absent):\n{r.stdout}\n{r.stderr}")
    print("  entry-point sharing OK (ocean resolves atmosphere fv3sw, root-absent)")


def main() -> int:
    try:
        import build  # noqa: F401
    except ModuleNotFoundError:
        _fail("the 'build' frontend is not installed (pip install build hatchling)")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("Building member wheels...")
        wheels = build_wheels(tmp / "wheels")
        print("Checking wheel contents...")
        check_wheel_contents(wheels)
        print("Checking dependency DAG...")
        check_dependency_dag(wheels)
        print("Root-absent install...")
        check_root_absent_install(wheels, tmp)
        print("Entry-point dycore sharing...")
        check_entry_point_sharing(wheels, tmp)
    print("PASS: federation packaging validated (wheels + root-absent install).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
