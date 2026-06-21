"""validate_strict membership hardening (Stage A1).

A typo in a dynamical-core axis must fail at config validation (a clear
ValueError listing the valid options), not silently slip through to a confusing
failure deep in the solver factory at JIT time.

Scope is the two axes with a *single* authoritative source of truth, so the
check cannot disagree with the runtime:

* model_type  -> ``atmosphere.dynamics.DYNAMICS_OPTIONS``
* discretization -> ``atmosphere.dynamics.DISCRETIZATION_OPTIONS``

(the very lists ``resolve_solver_name`` enforces).  These tests derive their
cases from those constants, so they prove *parity* with the runtime gate rather
than re-listing values that could drift.

``radiation`` / ``grid_type`` / ``time_integrator`` are intentionally NOT
hardened yet — their accepted sets are inconsistent across multiple dispatch
paths (e.g. ``physics_pipeline`` silently maps any unknown radiation scheme to
rrtmgp) and must be reconciled first; see the note in ``driver/config.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from legoesm.atmosphere.dynamics import DISCRETIZATION_OPTIONS, DYNAMICS_OPTIONS
from legoesm.config import Config
from legoesm.driver.config import DycoreConfig, ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_config_passes() -> None:
    ExperimentConfig().validate_strict()


# ---- parity: accept exactly the canonical dynamical-core axis members ----
@pytest.mark.parametrize("model_type", DYNAMICS_OPTIONS)
def test_every_canonical_model_type_accepted(model_type: str) -> None:
    ExperimentConfig(
        dycore=DycoreConfig(model_type=model_type, discretization="cdgrid")
    ).validate_strict()


@pytest.mark.parametrize("discretization", DISCRETIZATION_OPTIONS)
def test_every_canonical_discretization_accepted(discretization: str) -> None:
    ExperimentConfig(
        dycore=DycoreConfig(model_type="hydrostatic", discretization=discretization)
    ).validate_strict()


# ---- corpus guard: no shipped config that translates is newly rejected ----
def _all_config_yamls() -> list[Path]:
    files = sorted((REPO_ROOT / "config").glob("*.yaml"))
    files += sorted((REPO_ROOT / "config" / "aimip").glob("*.yaml"))
    return files


@pytest.mark.parametrize("config_path", _all_config_yamls(), ids=lambda p: p.name)
def test_translatable_configs_pass_validate_strict(config_path: Path) -> None:
    """Every config the basic loader can translate must still validate_strict.

    Configs that do not flow through this loader (config/aimip/ variants whose
    ``radiation:`` is a bare string the basic translator does not accept) raise
    during translation and are skipped — they use run_aimip.py's own loader and
    fold into this surface at Stage C1.
    """
    try:
        ec = Config.from_yaml(str(config_path)).to_experiment_config()
    except (AttributeError, TypeError, ValueError) as exc:
        pytest.skip(f"{config_path.name} not consumed by the basic loader: {exc}")
    ec.validate_strict()


# ---- rejection of typos ----
def test_bad_model_type_raises() -> None:
    with pytest.raises(ValueError, match="model_type"):
        ExperimentConfig(dycore=DycoreConfig(model_type="flat_earth")).validate_strict()


def test_bad_discretization_raises() -> None:
    with pytest.raises(ValueError, match="discretization"):
        ExperimentConfig(
            dycore=DycoreConfig(discretization="bogus")
        ).validate_strict()


# ---- surface bulk-flux scheme membership ----
@pytest.mark.parametrize("scheme", ["constant", "coare3", "large_yeager"])
def test_every_surface_bulk_scheme_accepted(scheme: str) -> None:
    # A non-"constant" scheme requires a turbulence scheme (the atmosphere
    # surface layer it upgrades); pair it with one so the membership check is
    # what's exercised, not the turbulence guard.
    turb = "none" if scheme == "constant" else "holtslag_boville"
    ExperimentConfig(
        surface_bulk_scheme=scheme, turbulence=turb
    ).validate_strict()


def test_bad_surface_bulk_scheme_raises() -> None:
    with pytest.raises(ValueError, match="surface_bulk_scheme"):
        ExperimentConfig(surface_bulk_scheme="bulk_richardson").validate_strict()


def test_non_constant_surface_bulk_requires_turbulence() -> None:
    # A MOST surface scheme upgrades the atmosphere surface layer via the
    # turbulence config; turbulence='none' would leave the atmosphere on its
    # fallback flux while the ocean tile switches => reject loudly.
    with pytest.raises(ValueError, match="surface_bulk_scheme"):
        ExperimentConfig(
            surface_bulk_scheme="coare3", turbulence="none"
        ).validate_strict()
    # With a turbulence scheme present it is accepted.
    ExperimentConfig(
        surface_bulk_scheme="coare3", turbulence="holtslag_boville"
    ).validate_strict()
