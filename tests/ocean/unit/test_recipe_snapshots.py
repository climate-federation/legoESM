"""Recipe-identity snapshots — the drift tripwire for #488.

Each registered ocean recipe factory is built with its DEFAULT config and its
structural scheme-selection identity is frozen here. This is the explicit recipe
of record AND a tripwire: if a model-config DEFAULT changes (or a factory is
edited) such that a recipe's dynamical-core identity drifts, the matching
assertion fails LOUDLY with the exact field — forcing a conscious "yes, the
recipe really should change" decision rather than a silent alteration of the
~17 global-overturning drivers + matrix that consume these factories.

Why a snapshot and not just "== model default": several of these fields are
inherited defaults today, so a default flip would silently change the recipe.
The factories now PIN their structural identity explicitly; this test locks the
resolved values so the pin and the default can never drift apart unnoticed.

To intentionally change a recipe: edit the factory, then update the expected
dict below in the SAME commit (the diff documents the decision).
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

# --- Frozen recipe identities (structural scheme selections only) -----------
# field -> expected value, per registered recipe built from its DEFAULT config.

GO_LATLON_IDENTITY = {
    "eos": "linear",
    "momentum_advection": "vector_invariant",
    "tracer_advection": "tvd",
    "pgf_scheme": "adcroft",
    "ke_gradient_scheme": "centered",
    "barotropic_solver": "explicit_substep",
    "coriolis_scheme": "matsuno_split",
    "outer_integrator": "forward_euler",
    "tracer_time_integrator": "euler",
    "implicit_vertical_mixing": True,
    "n_barotropic_substeps": 30,
}

GO_MPAS_IDENTITY = {
    "eos": "linear",
    "tracer_advection": "upwind",
    "pgf_scheme": "centered",
    "barotropic_solver": "explicit_substep",
    "pv_scheme": "enstrophy",
    "implicit_vertical_mixing": True,
}

EADY_IDENTITY = {
    "eos": "linear",
    "momentum_advection": "weno5",
    "tracer_advection": "weno5",
    "pgf_scheme": "smc03",
    "ke_gradient_scheme": "centered",
    "barotropic_solver": "implicit_cn",
    "outer_integrator": "ab2",
    "tracer_time_integrator": "rk3",
}


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


def test_go_gm_redi_recipe_still_pins_identity():
    """The dynamical-core identity is independent of the GM/Redi toggle — a
    use_gm_redi run must keep the same pinned schemes (only gm_redi differs)."""
    cfg = dataclasses.replace(GlobalOverturningConfig(), use_gm_redi=True)
    mc = global_overturning_model_config(cfg)
    _assert_identity(mc, GO_LATLON_IDENTITY, "global_overturning (latlon, GM/Redi)")
    assert mc.gm_redi is not None


@pytest.mark.parametrize("identity", [
    GO_LATLON_IDENTITY, GO_MPAS_IDENTITY, EADY_IDENTITY])
def test_snapshots_are_nonempty(identity):
    """Guard against a vacuous snapshot (empty dict would pass _assert_identity
    trivially)."""
    assert len(identity) >= 6 and "eos" in identity
