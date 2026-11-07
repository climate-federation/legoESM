#!/usr/bin/env python
"""Pure, UI-agnostic logic for the legoESM configuration wizard.

This module knows nothing about questionary or any terminal I/O — that lives in
``wizard.py``.  Here we (1) expose the *option lists* the wizard offers, (2) the
*gating predicates* that prune illegal combinations, (3) turn a complete answer
set into a :class:`LaunchPlan`, and (4) *emit* a runnable bundle
(``config.yaml`` / ``run.sh`` / ``wizard.yaml``).

Versioned with the model
------------------------
The interface must not drift from the model as it evolves.  Two mechanisms:

* **Live derivation.** Every option list is read from the model's own registries
  at call time — the supported (model_type, discretization, grid) matrix
  (:func:`driver.component_factory.supported_matrix`), ``DYNAMICS_OPTIONS`` /
  ``DISCRETIZATION_OPTIONS``, :func:`timestepping.dispatch.available_integrators`,
  the coupled ``PRESETS`` registry, the :mod:`legoesm.components` complexity
  taxonomy, ``config/templates/``, and ``config/machines/``.  Add a grid or a
  dynamical core to the model and the wizard offers it with no edit here.
* **Stamped provenance.** Every emitted bundle records ``wizard_schema_version``
  (this module's hand-authored structure), ``legoesm_version`` and the git
  commit, so a config built today is always traceable to the exact model it was
  built against — and ``tests/unit/test_wizard_core.py`` asserts the curated
  menus are subsets of the live registries, so a model rename fails loudly here.

Bump :data:`WIZARD_SCHEMA_VERSION` (semver) whenever the question structure,
answer keys, or branch→launcher routing change in a breaking way.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# scripts/experiment is on sys.path[0] when run directly; make the sibling
# helpers importable when this module is imported by name (wizard / tests).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from init_experiment import (  # noqa: E402
    _TEMPLATES_DIR,
    render_run_sh,
)
from lego_detect_machine import load_machine, resolve_machine  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Bump on breaking changes to the question structure / answer keys / routing.
WIZARD_SCHEMA_VERSION = "1.0.0"


# =====================================================================
# Provenance / versioning
# =====================================================================

def model_version() -> str:
    """Installed legoESM version (``importlib.metadata`` via ``legoesm._version``)."""
    try:
        from legoesm._version import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001 — version is best-effort metadata
        return "unknown"


def git_commit(repo_root: Path | str = _REPO_ROOT) -> str:
    """Short git commit of the working tree, or ``""`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return ""


def provenance(repo_root: Path | str = _REPO_ROOT) -> dict[str, str]:
    """The version stamp written into every emitted bundle."""
    return {
        "wizard_schema_version": WIZARD_SCHEMA_VERSION,
        "legoesm_version": model_version(),
        "git_commit": git_commit(repo_root),
        "created_utc": datetime.now(UTC).isoformat(),
    }


# =====================================================================
# Live option registries (derived from the model, never hardcoded)
# =====================================================================

# Curated, user-facing integrator names.  Aliases (ssp3, rk3, ...) are hidden
# but a test asserts CURATED ⊆ available_integrators() so a model-side rename
# breaks loudly here rather than silently emitting a dead --integrator.
_CURATED_INTEGRATORS = (
    "ssp_rk3", "ssp_rk3_scan", "ssp_rk34", "ssp_rk54", "ssp_rk54_scan", "rk4",
)

# Data-driven discretizations need a trained checkpoint and cannot be launched
# from the process-based YAML path (the YAML→ExperimentConfig boundary does not
# carry the checkpoint), so they are excluded from the *simulate* menu and live
# under the *train* objective instead.
_DATA_DRIVEN_DISCRETIZATIONS = frozenset({"sfno", "u_cast"})
# Resolved grid-aware by the driver from grid_type; not standalone menu items.
_AMBIGUOUS_DISCRETIZATIONS = frozenset({"finite_volume", "centered"})


def driver_combos() -> list[dict[str, str]]:
    """The driver-supported (model_type, discretization, grid_type) matrix.

    Single source of truth for combination legality — the same table the driver
    itself dispatches on, so the wizard can never offer a combo the driver
    rejects.
    """
    from legoesm.driver.component_factory import supported_matrix
    return supported_matrix()


def grid_types() -> list[str]:
    """Global grids reachable through the process-based simulate path."""
    grids = {c["grid_type"] for c in driver_combos()}
    grids.discard("plane")  # plane is the LES/CRM branch, asked separately
    return sorted(grids)


def dynamics_options() -> list[str]:
    """Atmosphere dynamical-complexity rungs (``model_type``)."""
    from legoesm.atmosphere.dynamics import DYNAMICS_OPTIONS
    return list(DYNAMICS_OPTIONS)


def discretization_options(*, include_data_driven: bool = False) -> list[str]:
    """Canonical spatial discretizations (legacy aliases dropped)."""
    from legoesm.atmosphere.dynamics import DISCRETIZATION_OPTIONS
    out = []
    for d in DISCRETIZATION_OPTIONS:
        if d in _AMBIGUOUS_DISCRETIZATIONS:
            continue
        if d in _DATA_DRIVEN_DISCRETIZATIONS and not include_data_driven:
            continue
        out.append(d)
    return out


def integrator_options() -> list[str]:
    """User-facing time integrators (subset of the live dispatch table)."""
    from legoesm.timestepping.dispatch import available_integrators
    avail = set(available_integrators())
    return [name for name in _CURATED_INTEGRATORS if name in avail]


def aimip_variants() -> list[str]:
    """Non-empty AIMIP / ML-emulation variants."""
    from legoesm.driver.config import AIMIP_VARIANTS
    return [v for v in AIMIP_VARIANTS if v]


def coupled_presets() -> list[tuple[str, str]]:
    """``(preset_key, one-line description)`` from the live coupled registry.

    Descriptions are the preset factories' own docstrings, so the *land of
    various complexity* ladder (none → slab → PFT → multilayer/Richards →
    carbon) tracks whatever the presets actually do.
    """
    from legoesm.driver.coupled_config import PRESETS
    out: list[tuple[str, str]] = []
    for key, fn in PRESETS.items():
        doc = (fn.__doc__ or "").strip().splitlines()
        out.append((key, doc[0] if doc else key))
    return out


def land_complexity_levels() -> list[str]:
    """The model's land-complexity vocabulary (taxonomy enum, versioned)."""
    from legoesm.components import LandComplexity
    return [c.value for c in LandComplexity]


def ocean_complexity_levels() -> list[str]:
    """The model's ocean-complexity vocabulary (taxonomy enum, versioned)."""
    from legoesm.components import OceanComplexity
    return [c.value for c in OceanComplexity]


def templates() -> list[str]:
    """Experiment-template ids under ``config/templates/`` (``cat/name``)."""
    if not _TEMPLATES_DIR.is_dir():
        return []
    return sorted(
        str(p.relative_to(_TEMPLATES_DIR))[:-5]
        for p in _TEMPLATES_DIR.rglob("*.yaml")
        if not p.name.startswith("_")
    )


def machine_names() -> list[str]:
    """Machine-profile names under ``config/machines/``."""
    mdir = _REPO_ROOT / "config" / "machines"
    if not mdir.is_dir():
        return []
    return sorted(p.stem for p in mdir.glob("*.yaml"))


# ---- curated case catalogues for the non-template launchers ----------
# These route to existing matrix/plane/SCM scripts.
LES_ABL_CASES = ("neutral", "ekman", "gabls1", "wangara")
SCM_CASES = ("rce", "gabls1", "ekman", "wangara")


def ocean_case_grids() -> dict[str, list[str]]:
    """Map each ocean case → grids that have a real ocean-matrix entry.

    Derived from ``run_ocean_test_matrix._build_test_matrix()`` (the same per-case
    grid logic the runner dispatches on), so the wizard never offers a
    ``--only CASE --grid GRID`` pair the matrix would filter to zero tests.  Many
    cases are regional/channel-only (gyres, ACC, Eady), which is exactly why a
    flat case×grid product is wrong.  Imported by path like the other tooling
    helpers (the matrix runner is a sibling script, not installed source).
    """
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "matrix"))
    from run_ocean_test_matrix import _build_test_matrix  # noqa: PLC0415
    grids: dict[str, set[str]] = {}
    for tc in _build_test_matrix():
        grids.setdefault(tc.case, set()).add(tc.grid_type)
    return {case: sorted(gs) for case, gs in sorted(grids.items())}


def ocean_cases() -> list[str]:
    """Ocean cases that have at least one runnable grid (from the live matrix)."""
    return list(ocean_case_grids())


def legal_ocean_grids(case: str) -> list[str]:
    """Grids the ocean matrix actually runs for ``case`` (empty if unknown)."""
    return ocean_case_grids().get(case, [])

# Radiation schemes offered by the prescribed-SST (AMIP) + coupled flows.
RADIATION_SCHEMES = ("gray", "rrtmg", "rrtmgp")
# AMIP prescribed-SST sources (run_amip.py --dataset).  ``custom`` is omitted —
# it needs --forcing-path/--sst-var plumbing the wizard does not collect.
AMIP_DATASETS = ("analytical", "cobe", "hadisst")
# run_amip.py's --time-integrator choices are a *restricted* subset of the
# general dispatch table (notably no ssp_rk54_scan), so the AMIP menu intersects
# against these — a drift test asserts AMIP_INTEGRATORS ⊆ run_amip.py's choices.
AMIP_INTEGRATORS = ("ssp_rk3", "ssp_rk3_scan", "ssp_rk34", "ssp_rk54", "rk4")

# AIMIP (ML-emulated physics) intercomparison.  run_aimip.py TRAINS each variant
# from a suite manifest + per-variant overlay; the base suite and the
# classical/column_nn/sfno_physics/sfno_full overlays all ship under
# config/aimip/, so every non-empty AIMIP_VARIANTS entry is launchable from
# the base suite — see aimip_launchable_variants().
_AIMIP_BASE_SUITE = "config/aimip/aimip_suite.yaml"


# =====================================================================
# Gating predicates (pure functions over the live registries)
# =====================================================================

def legal_discretizations(model_type: str) -> list[str]:
    """Discretizations that have a supported grid for ``model_type``."""
    avail = set(discretization_options())
    out = {
        c["discretization"] for c in driver_combos()
        if c["model_type"] == model_type and c["discretization"] in avail
    }
    return sorted(out)


def legal_grids(model_type: str, discretization: str) -> list[str]:
    """Grids supported for a (model_type, discretization) pair."""
    out = {
        c["grid_type"] for c in driver_combos()
        if c["model_type"] == model_type
        and c["discretization"] == discretization
        and c["grid_type"] != "plane"
    }
    return sorted(out)


def is_supported(model_type: str, discretization: str, grid_type: str) -> bool:
    return any(
        c["model_type"] == model_type
        and c["discretization"] == discretization
        and c["grid_type"] == grid_type
        for c in driver_combos()
    )


def legal_integrators(model_type: str, discretization: str) -> list[str]:
    """Integrators valid for a discretization.

    Spectral hydrostatic PE is stability-locked to SSP-RK54 by the driver
    (``component_factory``), so only that is offered there — matching the
    driver's own hard constraint rather than letting the user pick a combo it
    will reject at construction.
    """
    if model_type == "hydrostatic" and discretization == "spectral":
        return ["ssp_rk54"]
    return integrator_options()


def legal_amip_integrators(discretization: str) -> list[str]:
    """Integrators valid for an AMIP run on ``discretization``.

    AMIP routes to ``run_amip.py``, whose ``--time-integrator`` choices are a
    subset of the general dispatch table (it omits ``ssp_rk54_scan``), so the
    driver-legal set is intersected with what ``run_amip.py`` accepts.  Spectral
    hydrostatic PE stays locked to SSP-RK54, exactly as in :func:`legal_integrators`.
    """
    allowed = set(AMIP_INTEGRATORS)
    return [i for i in legal_integrators("hydrostatic", discretization) if i in allowed]


def aimip_launchable_variants(repo_root: Path | str = _REPO_ROOT) -> list[str]:
    """AIMIP variants launchable from the base suite (overlay must exist).

    The ML-emulated-physics ladder: ``classical`` (full process-physics control
    incl. RRTMGP radiation) → ``column_nn`` (ALL physics incl. radiation learned
    by a per-column NN) → ``sfno_physics`` (SFNO physics with spectral-dycore
    coupling) → ``sfno_full`` (SFNO emulates the entire atmosphere, no dycore).
    A variant is offered only when it is a real
    ``AIMIP_VARIANTS`` key *and* has a ``variant_<name>.yaml`` overlay beside the
    base suite, so a renamed/removed overlay drops from the menu instead of
    emitting a ``--variants`` flag that ``run_aimip.py`` rejects.
    """
    overlay_dir = (Path(repo_root) / _AIMIP_BASE_SUITE).parent
    return [
        v for v in aimip_variants()
        if (overlay_dir / f"variant_{v}.yaml").is_file()
    ]


# =====================================================================
# Backend / precision mapping
# =====================================================================

def backend_to_platforms(backend: str) -> str:
    """``cpu``/``gpu``/``mpi`` → JAX_PLATFORMS value."""
    return {"cpu": "cpu", "gpu": "cuda", "mpi": "cpu"}[backend]


def effective_machine(base: dict[str, Any], jax_platforms: str) -> dict[str, Any]:
    """Machine profile with the user's platform choice applied."""
    m = dict(base)
    if jax_platforms:
        m["jax_platforms"] = jax_platforms
    return m


def wrap_mpi(run_cmd: str, ranks: int) -> str:
    return f"mpirun -np {int(ranks)} {run_cmd}"


def _cmd(parts: list[Any]) -> str:
    """Join an argv list into a shell-safe command line.

    Every token is ``shlex.quote``-d so an absolute path containing spaces or
    shell metacharacters (repo root, cache dir, …) cannot break or inject into
    the emitted ``run.sh``.  Clean tokens are returned unchanged, so the rendered
    commands are byte-identical to a plain join on ordinary paths.
    """
    return " ".join(shlex.quote(str(p)) for p in parts)


# =====================================================================
# LaunchPlan + emit
# =====================================================================

@dataclass
class LaunchPlan:
    """A fully-resolved, runnable plan produced from a complete answer set."""

    kind: str                      # atm_global | coupled | ocean | les_crm | scm | train
    run_cmd: str                   # the command run.sh will execute
    jax_platforms: str             # cpu | cuda
    enable_x64: bool               # 64-bit → export JAX_ENABLE_X64=1
    summary: dict[str, Any]        # axis → chosen value (for display + record)
    config: Any = None             # legoesm.config.Config for the template path; else None
    notes: list[str] = field(default_factory=list)  # honest caveats / prerequisites
    workdir: str | None = None     # run.sh cwd; None → bundle dir, else this path
                                   # (AIMIP needs repo root for its relative configs)


def _resolve_template(template: str) -> Path:
    rel = template[:-5] if template.endswith(".yaml") else template
    path = _TEMPLATES_DIR / f"{rel}.yaml"
    if not path.is_file():
        from legoesm.experiment_registry import retired_template_message
        retired = retired_template_message(template)
        if retired is not None:
            raise ValueError(retired)
        raise ValueError(
            f"template {template!r} not found at {path}; "
            f"available: {', '.join(templates()) or '(none)'}"
        )
    return path


def _precision_overrides(bits: int) -> list[tuple[str, Any]]:
    mode, dyn = ("fp64", "float64") if int(bits) == 64 else ("fp32", "float32")
    return [
        ("hardware.precision.mode", mode),
        ("hardware.precision.dynamics", dyn),
    ]


# =====================================================================
# Per-branch plan builders
# =====================================================================

def _build_atm_global(answers: dict[str, Any], machine: dict[str, Any]) -> LaunchPlan:
    """Process-based global atmosphere → ``legoesm run config.yaml``.

    Reuses the template harness: a template seeds the case (dynamics + physics +
    forcing), and the wizard's numeric/grid/backend choices are applied as
    overrides, then strict-validated before the plan is returned.
    """
    from legoesm.config import Config

    template = answers["template"]
    cfg = Config.from_yaml(str(_resolve_template(template)))

    model_type = answers["model_type"]
    discretization = answers["discretization"]
    grid_type = answers["grid_type"]
    if not is_supported(model_type, discretization, grid_type):
        raise ValueError(
            f"unsupported combination: model_type={model_type!r}, "
            f"discretization={discretization!r}, grid_type={grid_type!r}"
        )

    bits = int(answers["precision_bits"])
    backend = answers["backend"]
    overrides: list[tuple[str, Any]] = [
        ("grid.type", grid_type),
        ("grid.resolution", int(answers["resolution"])),
        ("atmosphere.dynamics", model_type),
        ("atmosphere.discretization", discretization),
        ("atmosphere.time_integrator", answers["integrator"]),
        ("time.duration_hours", float(answers["duration_hours"])),
        *_precision_overrides(bits),
    ]
    if answers.get("nlev"):
        overrides.append(("grid.n_levels", int(answers["nlev"])))
    if backend == "mpi":
        overrides.append(("hardware.parallelism.distributed", True))
        if answers.get("mpi_ranks"):
            overrides.append(("hardware.parallelism.n_devices", int(answers["mpi_ranks"])))

    for key, value in overrides:
        cfg.set(key, value)

    # Fail fast on an illegal resolved config before anything is written.
    cfg.to_experiment_config().validate_strict()

    run_cmd = "legoesm run config.yaml"
    if backend == "mpi":
        run_cmd = wrap_mpi(run_cmd, int(answers.get("mpi_ranks", 2)))

    return LaunchPlan(
        kind="atm_global",
        run_cmd=run_cmd,
        jax_platforms=backend_to_platforms(backend),
        enable_x64=(bits == 64),
        config=cfg,
        summary={
            "objective": "simulate", "component": "atmosphere",
            "atm_kind": "global", "case_template": template,
            "model_type": model_type, "discretization": discretization,
            "grid_type": grid_type, "resolution": int(answers["resolution"]),
            "integrator": answers["integrator"],
            "duration_hours": float(answers["duration_hours"]),
            "precision_bits": bits, "backend": backend,
        },
    )


def _build_coupled(answers: dict[str, Any], machine: dict[str, Any]) -> LaunchPlan:
    """Coupled atm+surface (land of various complexity) → ``run_coupled.py --preset``.

    The ``--preset`` axis IS the land-complexity ladder: ``aquaplanet`` (no land)
    → ``slab_simple`` (bucket) → ``slab_pft`` → ``slab_richards`` (multilayer) →
    ``slab_carbon`` → ``full_coupled`` (land+ocean carbon).
    """
    preset = answers["preset"]
    valid = {k for k, _ in coupled_presets()}
    if preset not in valid:
        raise ValueError(f"unknown coupled preset {preset!r}; available: {sorted(valid)}")

    bits = int(answers["precision_bits"])
    backend = answers["backend"]
    script = _REPO_ROOT / "scripts" / "run" / "run_coupled.py"
    parts = [
        "python", str(script),
        "--preset", preset,
        "--resolution", str(int(answers.get("resolution", 16))),
        "--nlev", str(int(answers.get("nlev", 20))),
        "--days", str(int(answers.get("days", 30))),
        "--radiation", str(answers.get("radiation", "gray")),
        "--output", "output",
    ]
    run_cmd = _cmd(parts)
    if backend == "mpi":
        run_cmd = wrap_mpi(run_cmd, int(answers.get("mpi_ranks", 2)))

    return LaunchPlan(
        kind="coupled",
        run_cmd=run_cmd,
        jax_platforms=backend_to_platforms(backend),
        enable_x64=(bits == 64),
        summary={
            "objective": "simulate", "component": "coupled", "preset": preset,
            "resolution": int(answers.get("resolution", 16)),
            "days": int(answers.get("days", 30)),
            "radiation": answers.get("radiation", "gray"),
            "precision_bits": bits, "backend": backend,
        },
        notes=[
            "run_coupled.py selects ocean/land/carbon via --preset only; finer land "
            "knobs (CoupledConfig fields) require editing the preset in Python.",
        ],
    )


def _build_ocean(answers: dict[str, Any], machine: dict[str, Any]) -> LaunchPlan:
    """Ocean-only → ``run_ocean_test_matrix.py --only CASE --grid GRID``."""
    case = answers["ocean_case"]
    grid = answers["ocean_grid"]
    legal = legal_ocean_grids(case)
    if grid not in legal:
        raise ValueError(
            f"ocean case {case!r} has no matrix entry on grid {grid!r}; "
            f"runnable grids: {legal or '(unknown case)'}"
        )
    bits = int(answers["precision_bits"])
    script = _REPO_ROOT / "scripts" / "matrix" / "run_ocean_test_matrix.py"
    parts = ["python", str(script), "--only", case, "--grid", grid, "--output", "output"]
    if answers.get("quick"):
        parts.append("--quick")
    return LaunchPlan(
        kind="ocean",
        run_cmd=_cmd(parts),
        jax_platforms=backend_to_platforms(answers["backend"]),
        enable_x64=(bits == 64),
        summary={
            "objective": "simulate", "component": "ocean", "ocean_case": case,
            "ocean_grid": grid, "quick": bool(answers.get("quick", False)),
            "precision_bits": bits, "backend": answers["backend"],
        },
        notes=[
            "Ocean has no standalone driver; runs via the ocean test-matrix runner. "
            "Full case list: `python scripts/matrix/run_ocean_test_matrix.py --list`.",
        ],
    )


def _build_les_crm(answers: dict[str, Any], machine: dict[str, Any]) -> LaunchPlan:
    """LES/CRM doubly-periodic plane → the bespoke plane scripts."""
    case = answers["les_case"]
    bits = int(answers["precision_bits"])
    run_dir = _REPO_ROOT / "scripts" / "run"
    if case == "rising_thermal":
        script = run_dir / "run_plane_rising_thermal.py"
        parts = ["python", str(script), "--output", "output"]
    elif case == "rcemip":
        script = run_dir / "run_rcemip_plane.py"
        parts = [
            "python", str(script),
            "--precision", "float64" if bits == 64 else "float32",
            "--output", "output",
        ]
    elif case == "abl":
        abl = answers["abl_case"]
        script = run_dir / "run_les_plane.py"
        parts = ["python", str(script), "--case", abl, "--output", "output"]
        if bits == 32:
            parts.append("--f32")
    else:
        raise ValueError(f"unknown LES case {case!r}")
    return LaunchPlan(
        kind="les_crm",
        run_cmd=_cmd(parts),
        jax_platforms=backend_to_platforms(answers["backend"]),
        enable_x64=(bits == 64),
        summary={
            "objective": "simulate", "component": "atmosphere", "atm_kind": "les_crm",
            "les_case": case, "abl_case": answers.get("abl_case"),
            "precision_bits": bits, "backend": answers["backend"],
        },
        notes=[
            "Plane LES/CRM uses bespoke scripts (not `legoesm run`); grid/sizes "
            "default to the case's calibrated values — pass extra flags to tune.",
        ],
    )


def _build_scm(answers: dict[str, Any], machine: dict[str, Any]) -> LaunchPlan:
    """Single-column → ``run_scm_test_matrix.py CASE``."""
    case = answers["scm_case"]
    bits = int(answers["precision_bits"])
    script = _REPO_ROOT / "scripts" / "matrix" / "run_scm_test_matrix.py"
    parts = ["python", str(script), case]
    if case == "rce" and answers.get("days"):
        parts += ["--days", str(float(answers["days"]))]
    elif case != "rce" and answers.get("hours"):
        parts += ["--hours", str(float(answers["hours"]))]
    if answers.get("nlev"):
        parts += ["--nlev", str(int(answers["nlev"]))]
    return LaunchPlan(
        kind="scm",
        run_cmd=_cmd(parts),
        jax_platforms=backend_to_platforms(answers["backend"]),
        enable_x64=(bits == 64),
        summary={
            "objective": "simulate", "component": "atmosphere", "atm_kind": "scm",
            "scm_case": case, "precision_bits": bits, "backend": answers["backend"],
        },
        notes=["SCM is a separate single-column harness, independent of grid/dycore."],
    )


def _build_amip(answers: dict[str, Any], machine: dict[str, Any]) -> LaunchPlan:
    """Prescribed-SST atmosphere (AMIP) → ``run_amip.py`` (the ModelDriver path).

    A process-based global atmosphere forced by observed (COBE / HadISST) or
    analytical SST.  Discretization / grid / integrator / resolution / duration /
    radiation are surfaced; convection + turbulence stay at ``run_amip.py``'s
    defaults.  ``run_amip.py`` does not infer the discretization from the grid
    (it only auto-promotes the spectral case), so the wizard emits an explicit
    ``--discretization`` for the chosen (model_type, discretization, grid) triple
    — the same gating as the global flow.  The Gaussian grid uses the spectral
    transform, so x64 is forced there regardless of the user's bit choice.
    """
    dataset = answers["dataset"]
    grid = answers["grid_type"]
    discretization = answers["discretization"]
    if not is_supported("hydrostatic", discretization, grid):
        raise ValueError(
            f"unsupported AMIP combination: discretization={discretization!r}, "
            f"grid_type={grid!r} (model_type=hydrostatic)"
        )
    integrator = answers["integrator"]
    if integrator not in legal_amip_integrators(discretization):
        raise ValueError(
            f"integrator {integrator!r} is not valid for an AMIP run on "
            f"discretization={discretization!r}; choose from "
            f"{legal_amip_integrators(discretization)}"
        )
    # run_amip.py rejects every non-analytical dataset that lacks --forcing-path,
    # so require the SST file here rather than emit a run.sh that aborts on first
    # launch.
    forcing_path = (answers.get("forcing_path") or "").strip()
    if dataset != "analytical" and not forcing_path:
        raise ValueError(
            f"--dataset {dataset!r} needs a prescribed-SST file: set "
            f"answers['forcing_path'] (only 'analytical' runs without one)."
        )
    bits = int(answers["precision_bits"])
    backend = answers["backend"]
    # Gaussian grid → spectral transform → x64 mandatory (matches run_amip docs).
    enable_x64 = (bits == 64) or (grid == "gaussian")
    script = _REPO_ROOT / "scripts" / "run" / "run_amip.py"
    parts = [
        "python", str(script),
        "--dataset", dataset,
        "--grid-type", grid,
        "--discretization", discretization,
        "--resolution", str(int(answers.get("resolution", 16))),
        "--nlev", str(int(answers.get("nlev", 40))),
        "--days", str(int(answers.get("days", 30))),
        "--dt", str(float(answers.get("dt", 600.0))),
        "--time-integrator", integrator,
        "--radiation", answers.get("radiation", "gray"),
        "--output", "output",
    ]
    if forcing_path:
        parts += ["--forcing-path", forcing_path]
    run_cmd = _cmd(parts)
    if backend == "mpi":
        run_cmd = wrap_mpi(run_cmd, int(answers.get("mpi_ranks", 2)))

    notes = [
        "AMIP runs via run_amip.py (prescribed SST). Convection/turbulence stay "
        "at run_amip.py defaults (sbm / none) — pass extra flags to change them.",
    ]
    if dataset != "analytical":
        notes.append(
            f"SST source: {forcing_path} — variable names use run_amip.py defaults; "
            "pass --sst-var / --sic-path / etc. to override."
        )
    return LaunchPlan(
        kind="amip",
        run_cmd=run_cmd,
        jax_platforms=backend_to_platforms(backend),
        enable_x64=enable_x64,
        summary={
            "objective": "simulate", "component": "atmosphere", "atm_kind": "amip",
            "dataset": dataset, "forcing_path": forcing_path or "(analytical)",
            "grid_type": grid, "discretization": discretization,
            "resolution": int(answers.get("resolution", 16)),
            "nlev": int(answers.get("nlev", 40)),
            "days": int(answers.get("days", 30)),
            "dt": float(answers.get("dt", 600.0)),
            "integrator": integrator,
            "radiation": answers.get("radiation", "gray"),
            "precision_bits": bits, "backend": backend,
        },
        notes=notes,
    )


def _build_aimip(answers: dict[str, Any], machine: dict[str, Any]) -> LaunchPlan:
    """ML-emulated physics (AIMIP) → ``run_aimip.py`` trains a suite variant.

    The *ML-emulation of physics* axis: ``classical`` (full process physics
    incl. RRTMGP radiation) → ``column_nn`` (all physics incl. radiation
    learned per-column) → ``sfno_physics`` (SFNO physics, dycore retained) →
    ``sfno_full`` (SFNO full-atmosphere emulator).
    Routes to the shipped base suite + per-variant
    overlay under ``config/aimip/``.  This **trains** the variant (it is not a
    forward run) and needs an ERA5 cache; x64 is forced (spectral).
    """
    variant = answers["aimip_variant"]
    launchable = set(aimip_launchable_variants())
    if variant not in launchable:
        raise ValueError(
            f"AIMIP variant {variant!r} is not launchable from the base suite; "
            f"available: {sorted(launchable)}"
        )
    script = _REPO_ROOT / "scripts" / "run" / "run_aimip.py"
    # The suite manifest + its relative ``base:``/overlay/cache/output paths are
    # repo-root-relative; the bundle's run.sh therefore launches from the repo
    # root (workdir below), and the suite is passed absolute so it resolves
    # regardless of where the bundle is written.
    suite_abs = _REPO_ROOT / _AIMIP_BASE_SUITE
    parts = [
        "python", str(script),
        "--suite", str(suite_abs),
        "--variants", variant,
    ]
    if answers.get("smoke"):
        parts.append("--smoke")
    return LaunchPlan(
        kind="aimip",
        run_cmd=_cmd(parts),
        jax_platforms=backend_to_platforms(answers["backend"]),
        enable_x64=True,  # spectral transforms require x64
        workdir=str(_REPO_ROOT),  # suite's relative base:/cache/output resolve here
        summary={
            "objective": "train", "train_mode": "aimip",
            "aimip_variant": variant, "smoke": bool(answers.get("smoke", False)),
            "backend": answers["backend"],
        },
        notes=[
            f"AIMIP TRAINS the '{variant}' variant via run_aimip.py over the "
            f"{_AIMIP_BASE_SUITE} manifest (base config/aimip/aimip_era5.yaml); it "
            "needs an ERA5 cache. run.sh launches from the repo root so the "
            "suite's relative paths resolve; outputs land under the suite's "
            "output_dir (results/aimip_*), not the bundle dir. Add the smoke "
            "option for a tiny test config. x64 is forced (spectral).",
        ],
    )


def _build_train(answers: dict[str, Any], machine: dict[str, Any]) -> LaunchPlan:
    """Train on observed data (ERA5).

    Dispatches on ``train_mode``: the AIMIP intercomparison (ML-emulated physics,
    via :func:`_build_aimip`) or the spectral neural-GCM emulator.  Only these
    have a one-command CLI today; the grid-space modes (physics-param tuning,
    neural physics, SFNO correction) are Python-only and are surfaced as a note
    rather than a fabricated command.
    """
    mode = answers["train_mode"]
    if mode == "aimip":
        return _build_aimip(answers, machine)
    if mode != "neural_gcm_spectral":
        raise ValueError(
            f"train mode {mode!r} has no CLI launcher; only "
            f"'neural_gcm_spectral' is launchable. The grid-space modes "
            f"(train_physics_params / train_neural_gcm / train_sfno_coupled) are "
            f"Python-only — see packages/ml/legoesm/training/training_driver.py."
        )
    script = _REPO_ROOT / "scripts" / "run" / "train_neural_gcm_spectral.py"
    parts = [
        "python", str(script),
        "--n-max", str(int(answers.get("n_max", 42))),
        "--n-levels", str(int(answers.get("n_levels", 10))),
        "--epochs", str(int(answers.get("epochs", 50))),
        "--lr", str(float(answers.get("lr", 3e-4))),
        "--train-days", str(int(answers.get("train_days", 365))),
        "--year", str(int(answers.get("year", 2015))),
        "--cache-dir", str(answers.get("cache_dir", "data/era5_cache")),
    ]
    # Spectral transforms require x64 unconditionally.
    return LaunchPlan(
        kind="train",
        run_cmd=_cmd(parts),
        jax_platforms=backend_to_platforms(answers["backend"]),
        enable_x64=True,
        summary={
            "objective": "train", "train_mode": mode,
            "n_max": int(answers.get("n_max", 42)),
            "n_levels": int(answers.get("n_levels", 10)),
            "epochs": int(answers.get("epochs", 50)),
            "train_days": int(answers.get("train_days", 365)),
            "year": int(answers.get("year", 2015)),
            "backend": answers["backend"],
        },
        notes=[
            "ERA5 must be staged first: a Zarr cache at the --cache-dir (or network "
            "access to the WeatherBench2 GCS bucket). x64 is forced (spectral).",
        ],
    )


_BUILDERS = {
    ("simulate", "atmosphere", "global"): _build_atm_global,
    ("simulate", "atmosphere", "amip"): _build_amip,
    ("simulate", "atmosphere", "les_crm"): _build_les_crm,
    ("simulate", "atmosphere", "scm"): _build_scm,
    ("simulate", "coupled", None): _build_coupled,
    ("simulate", "ocean", None): _build_ocean,
    ("train", None, None): _build_train,
}


def build_plan(answers: dict[str, Any], machine: dict[str, Any] | None = None) -> LaunchPlan:
    """Resolve a complete answer set into a :class:`LaunchPlan`.

    Raises ``ValueError`` / ``KeyError`` on an incomplete or inconsistent answer
    set; the TUI shell is responsible for collecting valid answers via the gating
    predicates above.
    """
    if machine is None:
        machine = resolve_machine()
    objective = answers["objective"]
    component = answers.get("component") if objective == "simulate" else None
    atm_kind = answers.get("atm_kind") if component == "atmosphere" else None
    builder = _BUILDERS.get((objective, component, atm_kind))
    if builder is None:
        raise ValueError(
            f"no launcher for objective={objective!r}, component={component!r}, "
            f"atm_kind={atm_kind!r}"
        )
    return builder(answers, machine)


def emit_bundle(
    plan: LaunchPlan,
    out_dir: Path | str,
    *,
    name: str,
    machine: dict[str, Any],
    repo_root: Path | str = _REPO_ROOT,
) -> dict[str, str]:
    """Write the runnable bundle (``config.yaml`` / ``run.sh`` / ``wizard.yaml``).

    Fails fast (before writing anything) if ``out_dir`` exists and is non-empty,
    matching ``init_experiment``'s no-half-built-dir contract.
    """
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"--output-dir {out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)

    eff_machine = effective_machine(machine, plan.jax_platforms)
    written: dict[str, str] = {}

    if plan.config is not None:
        cfg_path = out / "config.yaml"
        plan.config.to_yaml(str(cfg_path))
        written["config"] = str(cfg_path)

    run_sh = render_run_sh(
        eff_machine, run_cmd=plan.run_cmd, enable_x64=plan.enable_x64,
        workdir=plan.workdir,
    )
    run_path = out / "run.sh"
    run_path.write_text(run_sh)
    run_path.chmod(0o755)
    written["run_sh"] = str(run_path)

    record = {
        "experiment": {"name": name, "kind": plan.kind},
        **provenance(repo_root),
        "machine": eff_machine.get("name", "default"),
        "selections": plan.summary,
        "launch": {
            "command": plan.run_cmd,
            "jax_platforms": plan.jax_platforms,
            "enable_x64": plan.enable_x64,
        },
        "notes": plan.notes,
    }
    rec_path = out / "wizard.yaml"
    rec_path.write_text(yaml.safe_dump(record, sort_keys=False))
    written["wizard_yaml"] = str(rec_path)
    return written


def load_machine_profile(name: str | None) -> dict[str, Any]:
    """Explicit-by-name or auto-detected machine profile."""
    return load_machine(name) if name else resolve_machine()
