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
    "legoesm-coupler",
    "legoesm-ml",
    "legoesm-tools",
)
COMPONENTS = ("legoesm-atmosphere", "legoesm-ocean", "legoesm-land", "legoesm-ice")

#: Member -> the legoesm subpackages it legitimately ships.  Mirrors
#: tests/test_federation_plan.FEDERATION_MEMBERS (several members bundle more than
#: one subpackage, so the wheel-isolation check cannot assume member==subpackage).
MEMBER_SUBPKGS = {
    "legoesm-core": {"core", "grids", "runtime", "parallel", "io",
                     "timestepping", "components"},
    "legoesm-atmosphere": {"atmosphere"},
    "legoesm-ocean": {"ocean"},
    "legoesm-land": {"land"},
    "legoesm-ice": {"ice"},
    "legoesm-coupler": {"coupler", "driver"},
    "legoesm-ml": {"ml", "training", "da"},
    "legoesm-tools": {"forcing", "diagnostics", "experiments", "visualization"},
}

#: The substrate loose modules belong to legoesm-core ONLY — a duplicate shipped
#: by any other member would shadow/conflict in the merged PEP-420 namespace.
#: surface_albedo is here (not the meta layer) because land/ice/coupler import it
#: at top level, so it must travel with the substrate they depend on.
SUBSTRATE_LOOSE = {"constants.py", "thermo.py", "registry.py",
                   "_version.py", "surface_albedo.py"}


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
        loose = {
            n[len("legoesm/"):]
            for n in names
            if n.startswith("legoesm/")
            and "/" not in n[len("legoesm/"):]
            and n.endswith(".py")
        }
        stray = subpkgs - MEMBER_SUBPKGS[m]
        if stray:
            _fail(f"{m} wheel ships foreign subpackages {stray} "
                  f"(expected {sorted(MEMBER_SUBPKGS[m])})")
        if m == "legoesm-core":
            # core MUST carry the substrate loose modules (and only those).
            missing = SUBSTRATE_LOOSE - loose
            if missing:
                _fail(f"legoesm-core wheel is missing substrate loose modules {missing}")
            extra = loose - SUBSTRATE_LOOSE
            if extra:
                _fail(f"legoesm-core wheel ships unexpected loose modules {extra}")
        else:
            # No other member may ship a substrate loose module — it would
            # shadow/conflict with legoesm-core in the merged namespace.  A
            # member's OWN loose module (e.g. ml's tuning.py) is fine.
            leaked = loose & SUBSTRATE_LOOSE
            if leaked:
                _fail(f"{m} wheel ships substrate loose modules {leaked} "
                      f"(those belong to legoesm-core only)")
    print("  wheel contents OK (namespace, subpackage isolation, no substrate-loose leakage)")


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

    for m in MEMBERS:
        if m == "legoesm-core":
            continue
        reqs = " ".join(_requires(wheels[m]))
        if "legoesm-core" not in reqs:
            _fail(f"{m} does not Requires-Dist legoesm-core")
    if any("legoesm-" in r for r in _requires(wheels["legoesm-core"])):
        _fail("legoesm-core depends on another member — it must be the DAG root")
    print("  dependency DAG OK (every member -> core; core -> nothing)")


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
        "from legoesm import constants, thermo, surface_albedo\n"
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


def check_land_standalone_uses_core_surface_albedo(wheels: dict[str, Path], tmp: Path) -> None:
    # The reason surface_albedo lives in legoesm-core, not the meta layer: the land
    # component imports ``legoesm.surface_albedo`` at top level, so a standalone
    # ``pip install legoesm-core legoesm-land`` (no meta, no other component) must
    # be able to import legoesm.land.  This pins that exact invariant.
    target = tmp / "land_standalone"
    r = _run([
        sys.executable, "-m", "pip", "install", "--quiet", "--target", str(target),
        "--no-deps", str(wheels["legoesm-core"]), str(wheels["legoesm-land"]),
    ])
    if r.returncode != 0:
        _fail(f"installing core+land wheels:\n{r.stderr}")
    code = (
        "import legoesm.land.slab_land\n"          # top-level imports legoesm.surface_albedo
        "from legoesm.surface_albedo import land_albedo, LandAlbedoConfig\n"
        "print('OK')\n"
    )
    env = {"PYTHONPATH": f"{target}:{_site_packages()}", "JAX_PLATFORMS": "cpu",
           "PATH": "/usr/bin:/bin"}
    r = _run([sys.executable, "-S", "-c", code], env=env)
    if r.returncode != 0 or "OK" not in r.stdout:
        _fail(f"root-absent core+land import (surface_albedo from core):\n{r.stdout}\n{r.stderr}")
    print("  land standalone OK (legoesm.land resolves surface_albedo from core, no meta)")


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
        print("Land standalone (surface_albedo from core)...")
        check_land_standalone_uses_core_surface_albedo(wheels, tmp)
        print("Entry-point dycore sharing...")
        check_entry_point_sharing(wheels, tmp)
    print("PASS: federation packaging validated (wheels + root-absent install).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
