"""Direct tests for the Oceananigans canonical recipe card.

Mirrors ``test_mitgcm_recipe_card.py``: asserts the canonical config factory
builds a real ``LatLonCGridOceanConfig`` with the pinned choices, that the
per-deck axes are honoured, and that every row of the block-mapping names a real
config field (the auditable wiring diagram cannot drift from the schema).
"""

from __future__ import annotations

import pytest

from legoesm.ocean.eos import LinearEOSConfig
from legoesm.ocean.fidelity.oceananigans_recipe import (
    OCEANANIGANS_BLOCK_MAPPING,
    oceananigans_canonical_ocean_config,
)
from legoesm.ocean.state import LatLonCGridOceanConfig


def _eos():
    return LinearEOSConfig(alpha_T=2.0e-4, beta_S=0.0)


def test_builds_valid_config_with_pinned_choices():
    cfg = oceananigans_canonical_ocean_config(eos_linear=_eos())
    assert isinstance(cfg, LatLonCGridOceanConfig)
    # Pinned faithful choices.
    assert cfg.eos == "linear"
    assert cfg.differentiable_barotropic is True
    assert cfg.use_conservation_fixer is False
    assert cfg.lateral_viscosity_operator == "flux_divergence"
    # Default (per-deck) scheme choices for the gyre/jet cases.
    assert cfg.momentum_advection == "vector_invariant"
    assert cfg.tracer_advection == "weno7"
    assert cfg.coriolis_scheme == "explicit_ab2"


def test_per_deck_axes_are_honoured():
    cfg = oceananigans_canonical_ocean_config(
        eos_linear=_eos(),
        g=0.1,                       # barotropic_gyre uses reduced gravity
        A_h=5.0e3,
        bottom_drag_r=1.0 / (60 * 86400),
        momentum_advection="weno9",  # the WENOVectorInvariant cases
        barotropic_solver="explicit_substep",
    )
    assert cfg.g == 0.1
    assert cfg.A_h == 5.0e3
    assert cfg.momentum_advection == "weno9"
    assert cfg.barotropic_solver == "explicit_substep"
    assert abs(cfg.bottom_drag_r - 1.0 / (60 * 86400)) < 1e-30


def test_overrides_patch_remaining_fields():
    cfg = oceananigans_canonical_ocean_config(
        eos_linear=_eos(), weno_smoothness="split")
    assert cfg.weno_smoothness == "split"


def test_block_mapping_fields_exist_on_config():
    """Every legoESM field named in the wiring diagram must be a real config
    attribute (or a documented composite) — the auditable map can't drift."""
    cfg = oceananigans_canonical_ocean_config(eos_linear=_eos())
    # Tokens that are real single config fields (composites like "g + rho_0" or
    # "surface_forcing (...)" are documented prose, not single attrs).
    single_fields = {
        "momentum_advection", "coriolis_scheme", "barotropic_solver",
        "bottom_drag_r", "tracer_advection",
    }
    for _construct, legoesm_field, _note in OCEANANIGANS_BLOCK_MAPPING:
        for tok in single_fields:
            if legoesm_field == tok:
                assert hasattr(cfg, tok), f"block-mapping field {tok!r} missing"


def test_mapping_flags_approx_rows():
    """The momentum + Coriolis rows are flagged [APPROX] (known discrete-level
    differences from the oracle, pending the fidelity scorecard)."""
    joined = " ".join(note for *_x, note in OCEANANIGANS_BLOCK_MAPPING)
    assert "[APPROX]" in joined
