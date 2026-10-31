"""#2241: the Large-Yeager (AeroBulk TURB_NCAR) smooth-switch widths are CONFIG
FIELDS on every component config that selects a bulk scheme, and every
production call of ``compute_most_fluxes`` passes them from that config.

The widths themselves (forms, defaults, oracle gaps, gradients) are tested in
``tests/core/test_aerobulk_ncar_oracle.py``; this file only tests the wiring.
"""
from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
from legoesm.atmosphere.physics.turbulence.surface_layer import compute_surface_fluxes
from legoesm.core.bulk_flux import (LARGE_YEAGER_WIDTH_DEFAULTS, NCAR_CD_CUTOFF_WIDTH,
                                    NCAR_DTHV_WIDTH, NCAR_NEUTRAL_WIDTH,
                                    compute_most_fluxes, large_yeager_widths)
from legoesm.coupler.config import CouplerConfig
from legoesm.coupler.lake.config import LakeConfig
from legoesm.ice.config import SeaIceConfig
from legoesm.land.config import LandConfig, MultiLayerLandConfig
from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
from legoesm.ocean.simple_ocean import SimpleOceanConfig

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = (SurfaceLayerConfig, CouplerConfig, LakeConfig, SeaIceConfig, LandConfig,
           MultiLayerLandConfig, BulkFormulaConfig, SimpleOceanConfig)
_HARD = dict.fromkeys(LARGE_YEAGER_WIDTH_DEFAULTS, 0.0)
# The 2 m temperature diagnostic takes its scheme from a diagnostics argument,
# not a component config; threading it is an open question on #2241.
_UNTHREADED = {"packages/coupler/legoesm/driver/diagnostics.py"}


def test_defaults_are_the_shared_constants():
    assert LARGE_YEAGER_WIDTH_DEFAULTS == {
        "large_yeager_neutral_width": NCAR_NEUTRAL_WIDTH,
        "large_yeager_dthv_width": NCAR_DTHV_WIDTH,
        "large_yeager_cd_cutoff_width": NCAR_CD_CUTOFF_WIDTH,
    }


@pytest.mark.parametrize("cls", CONFIGS, ids=lambda c: c.__name__)
def test_every_bulk_config_carries_the_widths(cls):
    assert large_yeager_widths(cls()) == LARGE_YEAGER_WIDTH_DEFAULTS
    cfg = cls()._replace(**_HARD)
    assert large_yeager_widths(cfg) == _HARD


def test_experiment_config_mirrors_the_widths():
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig()
    for k, d in LARGE_YEAGER_WIDTH_DEFAULTS.items():
        assert getattr(cfg, "surface_" + k) == d


def _calls_compute_most_fluxes(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
                getattr(node.func, "id", None) == "compute_most_fluxes"
                or getattr(node.func, "attr", None) == "compute_most_fluxes"):
            yield node


def test_every_package_call_site_passes_the_config_widths():
    """Each ``compute_most_fluxes(...)`` call under ``packages/`` carries
    ``**large_yeager_widths(<config>)`` (except the recorded diagnostic)."""
    files = [p for p in (ROOT / "packages").rglob("*.py")
             if "compute_most_fluxes(" in p.read_text()]
    seen, missing = 0, []
    for p in files:
        rel = str(p.relative_to(ROOT))
        for call in _calls_compute_most_fluxes(ast.parse(p.read_text())):
            seen += 1
            ok = any(kw.arg is None and isinstance(kw.value, ast.Call)
                     and getattr(kw.value.func, "id", None) == "large_yeager_widths"
                     for kw in call.keywords)
            if not ok and rel not in _UNTHREADED:
                missing.append(f"{rel}:{call.lineno}")
    assert seen >= 9, seen   # vacuity guard: the 9 known call sites are found
    assert not missing, missing


def _state():
    # near-neutral (dthv ~ +-0.02 K) and near-cutoff (32.5-33 m/s) columns, so
    # all three switches act.
    n = 12
    wind = jnp.concatenate([jnp.full(6, 8.0), jnp.linspace(32.55, 32.95, 6)])
    dT = jnp.linspace(-0.02, 0.02, n)
    T = jnp.full(n, 290.0)
    T_sfc = T + dT
    q = jnp.full(n, 0.010)
    q_sfc = q + 1e-6 * jnp.arange(n)
    rho = jnp.full(n, 1.2)
    return wind, jnp.zeros(n), T, q, T_sfc, q_sfc, rho


def test_surface_layer_config_widths_reach_the_solver():
    """The atmosphere surface layer runs on its config's widths: hard widths on
    the config reproduce a direct hard-width call bitwise and differ from the
    default."""
    x = _state()
    cfg = SurfaceLayerConfig(bulk_scheme="large_yeager", z_ref_model_level=False)
    kw = dict(z_ref=cfg.z_ref, z0_init=cfg.z0, scheme="large_yeager",
              n_iter=cfg.bulk_n_iter, unstable_gamma=cfg.most_unstable_gamma,
              stable_beta=cfg.most_stable_beta, z0h_z0_ratio=cfg.z0h_z0_ratio)
    hard = compute_surface_fluxes(*x, cfg._replace(**_HARD))
    ref_hard = compute_most_fluxes(*x, **kw, **_HARD)
    default = compute_surface_fluxes(*x, cfg)
    for a, b in zip(hard, ref_hard):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert any(not np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(hard, default))


def test_experiment_widths_reach_the_surface_layer_config():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.physics_pipeline import turbulence_config_for
    base = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(energy_fixer=False, dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="louis", surface_bulk_scheme="large_yeager")
    surf = turbulence_config_for(base).louis.surface
    assert large_yeager_widths(surf) == LARGE_YEAGER_WIDTH_DEFAULTS
    cfg = base._replace(**{"surface_" + k: v for k, v in _HARD.items()})
    assert large_yeager_widths(turbulence_config_for(cfg).louis.surface) == _HARD
    # one width alone: only it moves, the other two keep the default
    one = base._replace(surface_large_yeager_dthv_width=0.0)
    got = large_yeager_widths(turbulence_config_for(one).louis.surface)
    assert got == {**LARGE_YEAGER_WIDTH_DEFAULTS, "large_yeager_dthv_width": 0.0}
