#!/usr/bin/env python
"""Interactive wizard to configure and launch a legoESM run.

A guided questionary front-end over the existing config + template + launcher
machinery (all the routing/validation lives in :mod:`wizard_core`, which is the
unit-tested, UI-agnostic half).  The wizard asks the axes of a simulation —
component(s), region/model type, grid, integrator, duration, precision, device,
physics-vs-ML — prunes illegal combinations live from the model's own
registries, writes a runnable bundle (``config.yaml`` / ``run.sh`` /
``wizard.yaml`` with a version stamp), and optionally runs it.

Run it::

    python scripts/experiment/wizard.py        # or:  legoesm wizard

Needs the optional ``questionary`` dependency::

    pip install 'legoesm[wizard]'
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wizard_core as wc  # noqa: E402


def _require_questionary():
    try:
        import questionary  # noqa: PLC0415
        return questionary
    except ImportError:
        print(
            "ERROR: the wizard needs the 'questionary' package.\n"
            "  pip install 'legoesm[wizard]'   (or:  pip install questionary)",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _abort_if_none(value):
    """questionary returns None on Ctrl-C / ESC — treat as a clean abort."""
    if value is None:
        print("\nAborted.")
        raise SystemExit(130)
    return value


def _select(q, message: str, choices: list):
    return _abort_if_none(q.select(message, choices=choices).ask())


def _text(q, message: str, default: str = ""):
    return _abort_if_none(q.text(message, default=default).ask())


def _confirm(q, message: str, default: bool = False) -> bool:
    return _abort_if_none(q.confirm(message, default=default).ask())


def _int(q, message: str, default: int) -> int:
    while True:
        raw = _text(q, message, default=str(default))
        try:
            return int(raw)
        except ValueError:
            print(f"  '{raw}' is not an integer.")


def _float(q, message: str, default: float) -> float:
    while True:
        raw = _text(q, message, default=str(default))
        try:
            return float(raw)
        except ValueError:
            print(f"  '{raw}' is not a number.")


# ---------------------------------------------------------------------
# Per-branch question flows -> answer dict
# ---------------------------------------------------------------------

def _ask_backend_precision(q, answers: dict, *, allow_mpi: bool = True,
                           default_bits: int = 32) -> None:
    backends = ["cpu", "gpu"] + (["mpi"] if allow_mpi else [])
    answers["backend"] = _select(q, "Compute backend:", backends)
    if answers["backend"] == "mpi":
        answers["mpi_ranks"] = _int(q, "Number of MPI ranks:", 2)
    answers["precision_bits"] = int(
        _select(q, "Floating-point precision:",
                [f"{default_bits}", "64" if default_bits == 32 else "32"])
    )


def _flow_atm_global(q, answers: dict) -> None:
    from legoesm.config import Config

    tmpls = wc.templates()
    if not tmpls:
        raise SystemExit("No experiment templates found under config/templates/.")
    template = _select(q, "Starting case (template):", tmpls)
    answers["template"] = template
    model_type = Config.from_yaml(str(wc._resolve_template(template))).get(
        "atmosphere.dynamics", "hydrostatic"
    )
    print(f"  → template dynamics: {model_type}")
    answers["model_type"] = model_type

    discs = wc.legal_discretizations(model_type)
    answers["discretization"] = _select(q, "Spatial discretization:", discs)
    grids = wc.legal_grids(model_type, answers["discretization"])
    answers["grid_type"] = _select(q, "Grid type:", grids)
    integs = wc.legal_integrators(model_type, answers["discretization"])
    answers["integrator"] = _select(q, "Time integrator:", integs)
    answers["resolution"] = _int(q, "Horizontal resolution (cells/face or n_lat):", 32)
    if model_type != "shallow_water":
        answers["nlev"] = _int(q, "Vertical levels:", 30)
    answers["duration_hours"] = _float(q, "Run duration (hours):", 120.0)
    _ask_backend_precision(q, answers)


def _flow_amip(q, answers: dict) -> None:
    print(
        "\n  AMIP: process-based global atmosphere forced by prescribed SST.\n"
        "  ('analytical' SST needs no data; 'cobe'/'hadisst' need an SST file —\n"
        "  re-run the emitted command adding --forcing-path /path/to/SST.nc.)"
    )
    answers["dataset"] = _select(q, "SST dataset:", list(wc.AMIP_DATASETS))
    if answers["dataset"] != "analytical":
        # run_amip.py requires --forcing-path for observed SST; demand it now so
        # the emitted bundle actually runs.
        while True:
            path = _text(q, f"Path to the {answers['dataset']} SST file:", default="")
            if path.strip():
                answers["forcing_path"] = path.strip()
                break
            print("  An SST file path is required for non-analytical datasets.")
    # AMIP is hydrostatic PE; reuse the global gating so the emitted
    # (discretization, grid, integrator) is always a driver-supported triple.
    discs = wc.legal_discretizations("hydrostatic")
    answers["discretization"] = _select(q, "Spatial discretization:", discs)
    grids = wc.legal_grids("hydrostatic", answers["discretization"])
    answers["grid_type"] = _select(q, "Grid type:", grids)
    integs = wc.legal_amip_integrators(answers["discretization"])
    answers["integrator"] = _select(q, "Time integrator:", integs)
    answers["resolution"] = _int(q, "Horizontal resolution (cells/face or n_lat):", 16)
    answers["nlev"] = _int(q, "Vertical levels:", 40)
    answers["days"] = _int(q, "Run duration (days):", 30)
    answers["dt"] = _float(q, "Timestep dt (seconds):", 600.0)
    answers["radiation"] = _select(q, "Radiation scheme:", list(wc.RADIATION_SCHEMES))
    _ask_backend_precision(q, answers, default_bits=64)


def _flow_les_crm(q, answers: dict) -> None:
    case = _select(
        q, "LES / CRM case:",
        ["rising_thermal", "rcemip", "abl (GABLS1/Wangara/Ekman/neutral)"],
    )
    answers["les_case"] = {"rising_thermal": "rising_thermal",
                           "rcemip": "rcemip"}.get(case.split()[0], "abl")
    if answers["les_case"] == "abl":
        answers["abl_case"] = _select(q, "ABL case:", list(wc.LES_ABL_CASES))
    _ask_backend_precision(q, answers, allow_mpi=False, default_bits=64)


def _flow_scm(q, answers: dict) -> None:
    answers["scm_case"] = _select(q, "Single-column case:", list(wc.SCM_CASES))
    if answers["scm_case"] == "rce":
        answers["days"] = _float(q, "Duration (days):", 20.0)
    else:
        answers["hours"] = _float(q, "Duration (hours):", 9.0)
    answers["nlev"] = _int(q, "Vertical levels:", 30)
    _ask_backend_precision(q, answers, allow_mpi=False, default_bits=64)


def _flow_coupled(q, answers: dict) -> None:
    presets = wc.coupled_presets()
    labels = [f"{k}  —  {desc}" for k, desc in presets]
    print("\n  Land/ocean complexity ladder (the --preset axis):")
    chosen = _select(q, "Coupled preset (land of various complexity):", labels)
    answers["preset"] = chosen.split("  —  ")[0].strip()
    answers["resolution"] = _int(q, "Horizontal resolution (cells/face):", 16)
    answers["nlev"] = _int(q, "Atmosphere vertical levels:", 20)
    answers["days"] = _int(q, "Run duration (days):", 30)
    answers["radiation"] = _select(q, "Radiation scheme:", list(wc.RADIATION_SCHEMES))
    _ask_backend_precision(q, answers)


def _flow_ocean(q, answers: dict) -> None:
    answers["ocean_case"] = _select(q, "Ocean case:", wc.ocean_cases())
    grids = wc.legal_ocean_grids(answers["ocean_case"])
    answers["ocean_grid"] = _select(q, "Ocean grid:", grids)
    answers["quick"] = _confirm(q, "Quick (short) run?", default=True)
    _ask_backend_precision(q, answers, allow_mpi=False, default_bits=64)


def _flow_aimip(q, answers: dict) -> None:
    variants = wc.aimip_launchable_variants()
    if not variants:
        raise SystemExit(
            "No launchable AIMIP variants found "
            "(missing config/aimip/variant_*.yaml overlays)."
        )
    print(
        "\n  AIMIP: train an ML-emulated-physics variant on ERA5.\n"
        "    classical    — process-physics control (no ML physics)\n"
        "    column_nn    — per-column neural physics\n"
        "    sfno_physics — SFNO physics, spectral-dycore coupling\n"
        "  Needs an ERA5 cache (base: config/aimip/aimip_era5.yaml)."
    )
    answers["aimip_variant"] = _select(q, "AIMIP variant:", variants)
    answers["smoke"] = _confirm(q, "Smoke test (tiny config)?", default=True)
    answers["backend"] = _select(q, "Compute backend:", ["cpu", "gpu"])
    answers["precision_bits"] = 64  # spectral forces x64


def _flow_train(q, answers: dict) -> None:
    mode = _select(
        q, "Training mode:",
        ["aimip (ML-emulated physics: classical / column_nn / sfno_physics)",
         "neural_gcm_spectral (SFNO emulating dycore + physics — ML everything)"],
    )
    answers["train_mode"] = mode.split()[0]
    if answers["train_mode"] == "aimip":
        _flow_aimip(q, answers)
        return

    print(
        "\n  Training on observed data (ERA5).\n"
        "  neural_gcm_spectral: SFNO emulating dycore+physics.\n"
        "  Grid-space modes (physics-param tuning / neural physics / SFNO\n"
        "  correction) are Python-only — see training_driver.py."
    )
    answers["n_max"] = _int(q, "Spectral truncation n_max:", 42)
    answers["n_levels"] = _int(q, "Vertical levels:", 10)
    answers["epochs"] = _int(q, "Training epochs:", 50)
    answers["lr"] = _float(q, "Learning rate:", 3e-4)
    answers["train_days"] = _int(q, "Training window (days):", 365)
    answers["year"] = _int(q, "Start year:", 2015)
    answers["cache_dir"] = _text(q, "ERA5 cache dir:", default="data/era5_cache")
    answers["backend"] = _select(q, "Compute backend:", ["cpu", "gpu"])
    answers["precision_bits"] = 64  # spectral forces x64


def _collect_answers(q) -> dict:
    answers: dict = {}
    answers["objective"] = _select(
        q, "What do you want to do?",
        ["simulate (process-based physics)",
         "train an ML emulator (ML-emulated physics or ML-everything, on ERA5)"],
    ).split()[0]

    if answers["objective"] == "train":
        _flow_train(q, answers)
        return answers

    answers["component"] = _select(
        q, "Component(s) to run:",
        ["atmosphere", "coupled (atmosphere + land/ocean/ice)", "ocean"],
    ).split()[0]

    if answers["component"] == "atmosphere":
        kind = _select(
            q, "Atmosphere model type:",
            ["global (idealized / template-driven)",
             "amip (prescribed-SST, observed forcing)",
             "les_crm (LES/CRM doubly-periodic plane)",
             "scm (single column)"],
        )
        answers["atm_kind"] = kind.split()[0]
        {"global": _flow_atm_global, "amip": _flow_amip, "les_crm": _flow_les_crm,
         "scm": _flow_scm}[answers["atm_kind"]](q, answers)
    elif answers["component"] == "coupled":
        answers["component"] = "coupled"
        _flow_coupled(q, answers)
    else:
        answers["component"] = "ocean"
        _flow_ocean(q, answers)
    return answers


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    q = _require_questionary()

    print("=" * 64)
    print("  legoESM configuration wizard")
    print(f"  legoESM {wc.model_version()}  |  interface schema v{wc.WIZARD_SCHEMA_VERSION}")
    print("=" * 64)

    machines = wc.machine_names()
    machine_choice = "auto-detect"
    if machines:
        machine_choice = _select(q, "Machine profile:", ["auto-detect", *machines])
    machine = wc.load_machine_profile(None if machine_choice == "auto-detect" else machine_choice)

    answers = _collect_answers(q)

    name = _text(q, "Experiment name:", default="legoesm_run")
    out_dir = _text(q, "Output directory:", default=f"runs/{name}")

    try:
        plan = wc.build_plan(answers, machine)
    except Exception as exc:  # noqa: BLE001 — surface the resolution error cleanly
        print(f"\nERROR: could not build a launch plan: {exc}", file=sys.stderr)
        return 1

    print("\n" + "-" * 64)
    print(f"  Plan: {plan.kind}")
    for k, v in plan.summary.items():
        print(f"    {k:16s} {v}")
    print(f"    {'jax_platforms':16s} {plan.jax_platforms}")
    print(f"    {'enable_x64':16s} {plan.enable_x64}")
    print(f"  Command: {plan.run_cmd}")
    for note in plan.notes:
        print(f"  note: {note}")
    print("-" * 64)

    if not _confirm(q, f"Write bundle to {out_dir}?", default=True):
        print("Aborted (nothing written).")
        return 130

    try:
        written = wc.emit_bundle(plan, out_dir, name=name, machine=machine)
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: could not write bundle: {exc}", file=sys.stderr)
        return 1

    print("\n  Wrote:")
    for kind, path in written.items():
        print(f"    {kind:12s} {path}")
    print(f"\n  Run it:  cd {out_dir} && bash run.sh")

    if _confirm(q, "Run it now (bash run.sh)?", default=False):
        run_sh = Path(written["run_sh"])
        print(f"\n  Launching: bash {run_sh}\n")
        return subprocess.call(["bash", str(run_sh)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
