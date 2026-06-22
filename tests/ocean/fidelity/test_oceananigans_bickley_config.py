"""Pin the faithful bickley_jet eddy-regime config.

CASE 2 (bickley) passes its statistical enstrophy-decay bar ONLY with the
faithful scheme types that match Oceananigans: a MATCHED vertex-f Coriolis
(``coriolis_scheme="matsuno_split"`` -- closest to HydrostaticSphericalCoriolis
EnstrophyConserving, f & zeta co-located at the FF vertex) and the
ImplicitFreeSurface analog (``barotropic_solver="implicit_cn"`` with backward-
Euler theta=1.0). The old default ``explicit_ab2`` 4-point FACE-f Coriolis
supports a rotational 2dx null mode that drives a spurious late-time enstrophy
growth -> the eddy-regime under-dissipation FAILS the bar (ens_ratio 1.77 @ t12
vs the faithful config's 1.25). This test fails loudly if the bickley driver
silently reverts to the un-faithful config (commit 0d73fd48f).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture()
def bickley_module(monkeypatch):
    # Default (no env overrides) must yield the faithful config.
    for k in ("CORIOLIS_SCHEME", "BARO_SOLVER", "MOM_ADV"):
        monkeypatch.delenv(k, raising=False)
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.compare_oceananigans_bickley_jet as mod  # type: ignore
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_faithful_eddy_config_is_default(bickley_module):
    """build_bickley must default to the faithful matched-Coriolis + implicit
    free-surface config that passes CASE 2's statistical bar."""
    _grid, _wall, _z, _state, model = bickley_module.build_bickley()
    cfg = model.config
    assert cfg.coriolis_scheme == "matsuno_split", (
        f"coriolis_scheme={cfg.coriolis_scheme!r}; the eddy-regime bar needs the "
        "MATCHED vertex-f Coriolis (matsuno_split). A revert to explicit_ab2 "
        "re-introduces the 2dx null mode -> ens_ratio 1.77 (FAIL)."
    )
    assert cfg.barotropic_solver == "implicit_cn", (
        f"barotropic_solver={cfg.barotropic_solver!r}; the faithful ImplicitFreeSurface "
        "analog is implicit_cn (explicit_substep under-dissipates the roll-up)."
    )
    # Backward-Euler (theta=1.0), the ImplicitFreeSurface time discretization.
    assert float(cfg.barotropic_implicit_theta_eta) == 1.0
    assert float(cfg.barotropic_implicit_theta_pgf) == 1.0
    # weno9 vector-invariant momentum (the eddy-regime scheme under test).
    assert cfg.momentum_advection == "weno9"


def test_env_overrides_still_work(bickley_module, monkeypatch):
    """The CORIOLIS_SCHEME / BARO_SOLVER escape hatches remain (for A/B testing
    the un-faithful config), so the faithful default is a choice, not a lock-in."""
    monkeypatch.setenv("CORIOLIS_SCHEME", "explicit_ab2")
    monkeypatch.setenv("BARO_SOLVER", "explicit_substep")
    importlib.reload(bickley_module)
    _g, _w, _z, _s, model = bickley_module.build_bickley()
    assert model.config.coriolis_scheme == "explicit_ab2"
    assert model.config.barotropic_solver == "explicit_substep"
