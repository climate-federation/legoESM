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

import ast
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
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

#: Which member owns a given top-level ``legoesm.<X>`` import target.  Used to
#: check that every cross-member import in a wheel has a declared dependency
#: (hard OR optional extra) — a standalone install must not ship a module that
#: crashes on a missing sibling member.
SUBPKG_MEMBER = {sub: m for m, subs in MEMBER_SUBPKGS.items() for sub in subs}
LOOSE_MEMBER = {
    "constants": "legoesm-core", "thermo": "legoesm-core", "registry": "legoesm-core",
    "surface_albedo": "legoesm-core", "_version": "legoesm-core",
    "tuning": "legoesm-ml",
    "cli": "legoesm", "config": "legoesm", "dycore_factory": "legoesm",
    "supported_matrix": "legoesm", "taxonomy": "legoesm",
}


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"FAIL: {msg}")
    sys.exit(1)


def build_wheels(outdir: Path) -> dict[str, Path]:
    wheels: dict[str, Path] = {}
    for m in MEMBERS:
        # Folder names drop the ``legoesm-`` prefix (we are already in legoesm):
        # the distribution is ``legoesm-core`` but its source lives in packages/core.
        member_dir = REPO / "packages" / m.removeprefix("legoesm-")
        r = _run([
            sys.executable, "-m", "build", "--wheel", "--no-isolation",
            "--outdir", str(outdir), str(member_dir),
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


def _declared_member_reqs(whl: Path) -> set[str]:
    """All ``legoesm-*`` names in the wheel's Requires-Dist (hard OR extra-gated)."""
    names: set[str] = set()
    for n in zipfile.ZipFile(whl).namelist():
        if n.endswith("METADATA"):
            for ln in zipfile.ZipFile(whl).read(n).decode().splitlines():
                if ln.lower().startswith("requires-dist:"):
                    spec = ln.split(":", 1)[1].strip().split(";", 1)[0].strip()
                    name = spec.split()[0] if spec else ""
                    for sep in ("~=", "==", ">=", "<=", "!=", ">", "<", "="):
                        name = name.split(sep)[0]
                    name = name.strip()
                    if name.startswith("legoesm-"):
                        names.add(name)
            break
    return names


def _legoesm_import_targets(src: str) -> set[str]:
    """Members imported by a source file via ``legoesm.<X>...`` (any X owner)."""
    out: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        for mod in mods:
            p = mod.split(".")
            if p[0] != "legoesm" or len(p) < 2:
                continue
            owner = SUBPKG_MEMBER.get(p[1]) or LOOSE_MEMBER.get(p[1])
            if owner:
                out.add(owner)
    return out


def check_import_edges_covered_by_metadata(wheels: dict[str, Path]) -> None:
    # Every cross-member ``legoesm.*`` import in a wheel's shipped source must have
    # a declared dependency (hard or optional extra) on the target member.  Without
    # this, a standalone install can ship a public module that crashes on import of
    # a missing sibling (e.g. atmosphere's SFNO dycore -> legoesm.ml).
    for m, whl in wheels.items():
        declared = _declared_member_reqs(whl) | {m, "legoesm-core"}
        targets: set[str] = set()
        z = zipfile.ZipFile(whl)
        for n in z.namelist():
            if n.startswith("legoesm/") and n.endswith(".py"):
                targets |= _legoesm_import_targets(z.read(n).decode("utf-8", "replace"))
        missing = {t for t in targets if t != m} - declared
        if missing:
            _fail(f"{m} wheel imports {sorted(missing)} but its METADATA declares no "
                  f"dependency (hard or extra) on them — a standalone install would "
                  f"ship modules that crash on import")
    print("  import-edge coverage OK (every cross-member import has a declared dep/extra)")


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


def check_sfno_gated_by_ml_extra(wheels: dict[str, Path], tmp: Path) -> None:
    # The SFNO neural dycore is an OPTIONAL feature: it lives in legoesm-atmosphere
    # but imports legoesm.ml at top level, so it is reachable only via the
    # legoesm-atmosphere[ml] extra.  Prove the extra is load-bearing: WITHOUT ml,
    # importing the SFNO module must fail cleanly on the missing legoesm.ml — never
    # silently "work" (which would mean the extra is spurious or ml leaked in).
    target = tmp / "atm_no_ml"
    r = _run([
        sys.executable, "-m", "pip", "install", "--quiet", "--target", str(target),
        "--no-deps", str(wheels["legoesm-core"]), str(wheels["legoesm-atmosphere"]),
    ])
    if r.returncode != 0:
        _fail(f"installing core+atmosphere wheels:\n{r.stderr}")
    code = (
        "import legoesm.atmosphere\n"                      # base must import fine
        "try:\n"
        "    import legoesm.atmosphere.dynamics.neural.sfno_sw\n"
        "    print('UNEXPECTED-OK')\n"
        "except ModuleNotFoundError as e:\n"
        "    assert 'legoesm.ml' in str(e), str(e)\n"
        "    print('OK')\n"
    )
    env = {"PYTHONPATH": f"{target}:{_site_packages()}", "JAX_PLATFORMS": "cpu",
           "PATH": "/usr/bin:/bin"}
    r = _run([sys.executable, "-S", "-c", code], env=env)
    if r.returncode != 0 or "OK" not in r.stdout or "UNEXPECTED-OK" in r.stdout:
        _fail(f"SFNO must be gated by the [ml] extra:\n{r.stdout}\n{r.stderr}")
    print("  SFNO gating OK (base atmosphere imports; sfno_sw needs the [ml] extra)")


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
        print("Checking import-edge metadata coverage...")
        check_import_edges_covered_by_metadata(wheels)
        print("Root-absent install...")
        check_root_absent_install(wheels, tmp)
        print("Land standalone (surface_albedo from core)...")
        check_land_standalone_uses_core_surface_albedo(wheels, tmp)
        print("SFNO gated by [ml] extra...")
        check_sfno_gated_by_ml_extra(wheels, tmp)
        print("Entry-point dycore sharing...")
        check_entry_point_sharing(wheels, tmp)
    print("PASS: federation packaging validated (wheels + root-absent install).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
