"""Direct test for the #1320 blended-vs-tiled quantification probe.

Every new .py gets a test that imports and exercises the leaf module. What is
worth gating here is the probe's OWN control, because a probe whose control
does not bind is a probe that measures itself.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "scripts" / "validate" / "quantify_blended_vs_tiled_surface_1320.py"


@pytest.fixture(scope="module")
def mod():
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    spec = importlib.util.spec_from_file_location("q1320", _SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def cfgs():
    from legoesm.driver.config import (DycoreConfig, ExperimentConfig,
                                       GridConfig, OutputConfig)
    from legoesm.driver.physics_pipeline import resolve_tiled_surface_configs
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=10,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(discretization="mpas", dt=75.0),
        output=OutputConfig(output_dir="", diag_days=0),
        days=1, dataset="analytical", radiation="gray",
        convection="none", turbulence="louis",
        surface_bulk_scheme="coare3", precision="fp64")
    return resolve_tiled_surface_configs(cfg)


def test_the_control_binds(mod, cfgs):
    """Identical tiles must give identical fluxes from both constructions.

    This is the probe's only defence against measuring its own harness, so it
    is the thing to gate.
    """
    o, i, l = cfgs
    assert o is not None, "the probe's configuration resolves no surface law"
    assert float(mod.control(o, i, l, n=40)) < 1e-6


def test_the_control_is_not_vacuous(mod, cfgs):
    """...and it must FAIL when the tiles genuinely differ.

    A control that passes for any input proves nothing. Feeding the real
    coastal population through the same comparison has to produce a large
    difference, or the control above is just arithmetic.
    """
    o, i, l = cfgs
    cases = mod.build_cases(40)
    tiled, blended = mod.run(cases, o, i, l)[:2]
    assert float(np.max(np.abs(blended["shflx"] - tiled["shflx"]))) > 1.0


def test_the_production_lane_ocean_fraction_error_is_real(mod, cfgs):
    """The arm the first version of this probe was missing.

    With the interactive land on, the lane keeps
    ``heat = (1-f)*BULK(T_blend, q_blend) + f*F_land`` — so the OCEAN
    fraction is evaluated on the BLENDED surface. The land flux cancels in
    the difference against the correct form, and what is left must not be
    negligible, or the claim that the averaging-order defect survives the
    hand-over is wrong.
    """
    o, i, l = cfgs
    cases = mod.build_cases(200)
    _, _, _, prod, prod_ok = mod.run(cases, o, i, l)
    for k in ("shflx", "lhflx"):
        err = float(np.mean(np.abs(prod[k] - prod_ok[k])))
        scale = float(np.mean(np.abs(prod[k])))
        assert err > 0.1 * scale, (
            f"{k}: the production-lane ocean-fraction order error {err:.2f} "
            f"is under a tenth of the flux itself {scale:.2f} — the issue "
            f"comment's claim that this survives the land hand-over fails")


def test_the_decomposition_adds_up(mod, cfgs):
    """total == averaging_order + tile_law, exactly, by construction."""
    o, i, l = cfgs
    cases = mod.build_cases(40)
    tiled, blended, same_law = mod.run(cases, o, i, l)[:3]
    for k in ("shflx", "lhflx"):
        total = blended[k] - tiled[k]
        parts = (blended[k] - same_law[k]) + (same_law[k] - tiled[k])
        np.testing.assert_allclose(total, parts, rtol=1e-12, atol=1e-9)


def test_momentum_is_where_the_production_defect_lives(mod, cfgs):
    """The averaging ORDER barely moves the stress; the tile LAW dominates it.

    This pins the reading the issue comment reports: on the production lane
    the land fraction's heat fluxes come from the land model, so what the
    tiled port would still change is the surface STRESS, and the stress error
    is a tile-law error (ocean roughness over land), not an averaging-order
    one.
    """
    o, i, l = cfgs
    cases = mod.build_cases(200)
    tiled, blended, same_law = mod.run(cases, o, i, l)[:3]
    order = float(np.mean(np.abs(blended["tau"] - same_law["tau"])))
    law = float(np.mean(np.abs(same_law["tau"] - tiled["tau"])))
    assert law > 20.0 * order, (
        f"tile-law stress error {law:.4f} Pa is not dominant over the "
        f"averaging-order one {order:.4f} Pa — the issue comment's reading "
        f"does not hold on this sample")
