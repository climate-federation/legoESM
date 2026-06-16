"""Recipe-identity snapshots — the drift tripwire for #488.

Each registered ocean recipe factory is built with its DEFAULT config and its
structural scheme-selection identity is compared against the named recipe
catalog. The catalog is the explicit recipe of record AND this test is the
tripwire: if a model-config DEFAULT changes (or a factory is edited) such that a
recipe's dynamical-core identity drifts, the matching assertion fails LOUDLY with
the exact field — forcing a conscious "yes, the recipe really should change"
decision rather than a silent alteration of the ~17 global-overturning drivers +
matrix that consume these factories.

Why a snapshot and not just "== model default": several of these fields are
inherited defaults today, so a default flip would silently change the recipe.
The factories now PIN their structural identity explicitly; this test locks the
resolved values so the pin and the default can never drift apart unnoticed.

To intentionally change a recipe: edit the catalog entry and the factory in the
SAME commit (the diff documents the decision).
"""

from __future__ import annotations

import dataclasses

import pytest
from legoesm.ocean.experiments.eady_uniform import (
    EadyUniformConfig,
    eady_uniform_model_config,
)
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig,
    global_overturning_model_config,
    global_overturning_mpas_model_config,
)

from legoesm.ocean.recipes import get_recipe

# --- Frozen recipe identities (structural scheme selections only) -----------
# The registry is the named recipe of record; this file checks factories against
# it so catalog and experiment defaults cannot silently drift apart.

GO_LATLON_IDENTITY = get_recipe("legoesm_linear_v1")
GO_MPAS_IDENTITY = get_recipe("legoesm_linear_mpas_v1", "mpas")
EADY_IDENTITY = get_recipe("eady_weno5_v1")

# DINO (Kamm et al. 2025 NEMO double-gyre approximation). Snapshot-protected but
# NOT pinned in the factory — dino_lat_lon_model_config is Pierre's NEMO-fidelity
# code, so we record its identity here without editing it. NOTE the inherited
# defaults this surfaces: coriolis_scheme/outer_integrator/tracer_time_integrator
# are legoESM defaults (forward_euler / euler), NOT NEMO's leapfrog — a known
# NEMO-unfaithfulness (see #487, the NEMO recipe card). ke_gradient="hollingsworth"
# and barotropic="implicit_cn" ARE deliberate NEMO matches.
DINO_LATLON_IDENTITY = get_recipe("nemo_dino_v1")


def _assert_identity(mc, expected, label):
    actual = {f: getattr(mc, f) for f in expected}
    drift = {f: (expected[f], actual[f]) for f in expected
             if actual[f] != expected[f]}
    assert not drift, (
        f"{label} recipe identity DRIFTED (expected -> actual): {drift}. "
        "A model default likely changed. If this recipe SHOULD change, update "
        "the expected dict in this file in the same commit.")


def test_global_overturning_latlon_snapshot():
    mc = global_overturning_model_config(GlobalOverturningConfig())
    _assert_identity(mc, GO_LATLON_IDENTITY, "global_overturning (latlon)")
    assert mc.gm_redi is None          # GM/Redi off unless use_gm_redi


def test_global_overturning_mpas_snapshot():
    mc = global_overturning_mpas_model_config(GlobalOverturningConfig())
    _assert_identity(mc, GO_MPAS_IDENTITY, "global_overturning (mpas)")
    assert mc.gm_redi is None


def test_eady_uniform_snapshot():
    mc = eady_uniform_model_config(EadyUniformConfig())
    _assert_identity(mc, EADY_IDENTITY, "eady_uniform")
    assert mc.gm_redi is None          # eddies resolved


def test_dino_latlon_snapshot():
    """DINO's recipe identity is snapshot-protected even though its factory lives
    in Pierre's NEMO-fidelity code (we don't edit it here). Builds a tiny Mercator
    grid; the scheme identity is grid-resolution-independent."""
    from legoesm.grids.latlon import create_mercator_grid
    from legoesm.ocean.experiments.dino import DINOConfig, dino_lat_lon_model_config
    grid = create_mercator_grid(n_lon=16, lat_max_deg=70.0,
                                lon_west_deg=0.0, lon_east_deg=50.0)
    mc, _ = dino_lat_lon_model_config(grid, DINOConfig())
    _assert_identity(mc, DINO_LATLON_IDENTITY, "dino (latlon)")


def test_go_gm_redi_recipe_still_pins_identity():
    """The dynamical-core identity is independent of the GM/Redi toggle — a
    use_gm_redi run must keep the same pinned schemes (only gm_redi differs)."""
    cfg = dataclasses.replace(GlobalOverturningConfig(), use_gm_redi=True)
    mc = global_overturning_model_config(cfg)
    _assert_identity(mc, GO_LATLON_IDENTITY, "global_overturning (latlon, GM/Redi)")
    assert mc.gm_redi is not None


@pytest.mark.parametrize("identity", [
    GO_LATLON_IDENTITY, GO_MPAS_IDENTITY, EADY_IDENTITY, DINO_LATLON_IDENTITY])
def test_snapshots_are_nonempty(identity):
    """Guard against a vacuous snapshot (empty dict would pass _assert_identity
    trivially)."""
    assert len(identity) >= 6 and "eos" in identity
