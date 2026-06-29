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
    assert cfg.barotropic.differentiable_barotropic is True
    assert cfg.use_conservation_fixer is False
    assert cfg.lateral_viscosity_operator == "flux_divergence"
    # Default (per-deck) scheme choices for the gyre/jet cases.
    assert cfg.momentum_advection == "vector_invariant"
    assert cfg.tracer_advection == "weno7"
    assert cfg.coriolis_scheme == "explicit_ab2"
    # Time filter defaults PER SOLVER: inert cosine under the implicit_cn default
    # (no substep to filter — avoids the no-op warning), faithful SM2005 power_law
    # when a deck overrides to explicit_substep where it is consumed (#501: nested
    # under BarotropicConfig). docs/ocean/fidelity/oceananigans_recipe_wiring_plan §8.
    assert cfg.barotropic.barotropic_time_filter == "cosine"
    assert oceananigans_canonical_ocean_config(
        eos_linear=_eos(), barotropic_solver="explicit_substep",
    ).barotropic.barotropic_time_filter == "power_law"


def test_barotropic_time_filter_overridable():
    cfg = oceananigans_canonical_ocean_config(
        eos_linear=_eos(), barotropic_time_filter="cosine")
    assert cfg.barotropic.barotropic_time_filter == "cosine"


def test_power_law_filter_under_implicit_solver_warns_noop():
    """barotropic_time_filter is consumed ONLY by the explicit_substep substep; a
    non-default filter under an implicit solver is a silent no-op, so
    _validate_config warns loudly (not raises — harmless, just inert)."""
    import warnings

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    noop = oceananigans_canonical_ocean_config(
        eos_linear=_eos(), barotropic_solver="implicit_cn",
        coriolis_scheme="matsuno_split", barotropic_time_filter="power_law")
    with pytest.warns(UserWarning, match="has NO effect"):
        LatLonCGridOceanModel._validate_config(noop)
    # default cosine under the same solver, and explicit_substep (which consumes
    # the filter), must NOT warn.
    for ok in (
        oceananigans_canonical_ocean_config(
            eos_linear=_eos(), barotropic_solver="implicit_cn",
            coriolis_scheme="matsuno_split", barotropic_time_filter="cosine"),
        oceananigans_canonical_ocean_config(
            eos_linear=_eos(), barotropic_solver="explicit_substep",
            barotropic_time_filter="power_law"),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            LatLonCGridOceanModel._validate_config(ok)
        assert not any("has NO effect" in str(w.message) for w in caught)


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
    assert cfg.lateral_viscosity.A_h == 5.0e3
    assert cfg.momentum_advection == "weno9"
    assert cfg.barotropic.barotropic_solver == "explicit_substep"
    assert abs(cfg.bottom_drag.bottom_drag_r - 1.0 / (60 * 86400)) < 1e-30


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
    # #501: a token may be a direct attribute OR a flat member of a nested
    # sub-config (e.g. bottom_drag_r -> config.bottom_drag.bottom_drag_r); both
    # are real, addressable fields of the wiring diagram (flat_fields is the 1:1
    # flat name set kept stable across grouping).
    flat = cfg.flat_fields() if hasattr(cfg, "flat_fields") else set()
    for _construct, legoesm_field, _note in OCEANANIGANS_BLOCK_MAPPING:
        for tok in single_fields:
            if legoesm_field == tok:
                assert hasattr(cfg, tok) or tok in flat, (
                    f"block-mapping field {tok!r} missing")


def test_mapping_flags_approx_rows():
    """The momentum + Coriolis rows are flagged [APPROX] (known discrete-level
    differences from the oracle, pending the fidelity scorecard)."""
    joined = " ".join(note for *_x, note in OCEANANIGANS_BLOCK_MAPPING)
    assert "[APPROX]" in joined
