"""Q7 — config-footgun guards (Phase G).

Two footguns flagged in the strategy doc, both resolved by *fail-fast
validation* rather than reordering the 45-field NamedTuple (which would break
positional construction for legacy callers):

1. **EOS dispatch (``eos`` / ``eos_linear``).** ``make_eos_fn`` raises on an
   unknown ``eos`` literal (dispatch discipline). ``LatLonCGridOceanModel._
   validate_config`` now also rejects an unknown ``eos`` at CONSTRUCTION (not
   lazily at the first step), using the single ``VALID_EOS_SCHEMES`` source so
   the valid set is never duplicated.

2. **A_h single source of truth.** On the lat-lon C-grid, horizontal viscosity
   comes ONLY from the dynamics ``config.A_h``. The physics-pathway
   ``LateralMixingConfig`` carries its OWN ``HarmonicConfig.A_h`` — a competing
   source — but is cubed-sphere-only, so ``_validate_config`` rejects any
   non-"none" ``physics.lateral_mixing.scheme`` (including ``"harmonic"``) at
   construction. That guard is what makes ``config.A_h`` the sole A_h source on
   this grid.

   (A purely-static "two field named A_h" AST detector would false-positive:
   A_h legitimately appears in several *different* grid configs — OceanConfig,
   LatLonOceanConfig, HarmonicConfig — which are not a single config's two
   sources. The meaningful single-source check is therefore the runtime guard
   below, per the strategy doc's "curation avoids false positives" note.)
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import VALID_EOS_SCHEMES, LinearEOSConfig, make_eos_fn
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.lateral_mixing.config import (
    HarmonicConfig,
    LateralMixingConfig,
)
from legoesm.ocean.state import LatLonCGridOceanConfig


# ---------------------------------------------------------------------------
# Footgun 1 — EOS dispatch
# ---------------------------------------------------------------------------


def test_make_eos_fn_accepts_every_valid_scheme():
    """Every literal in the single VALID_EOS_SCHEMES source must dispatch to a
    real callable (catches a set entry with no matching branch)."""
    for scheme in VALID_EOS_SCHEMES:
        fn = make_eos_fn(scheme)
        assert callable(fn), scheme


def test_make_eos_fn_rejects_unknown_scheme():
    """Dispatch discipline: an unknown eos literal must raise ValueError (no
    silent fallback to a default scheme)."""
    with pytest.raises(ValueError, match="Unknown EOS scheme"):
        make_eos_fn("nonlin_typo")


def test_make_eos_fn_linear_defaults_eos_linear():
    """eos='linear' with eos_linear=None uses default LinearEOSConfig (the
    documented coupling); an explicit eos_linear is honoured."""
    fn_default = make_eos_fn("linear", eos_linear=None)
    assert callable(fn_default)
    fn_custom = make_eos_fn("linear", eos_linear=LinearEOSConfig(rho_ref=1025.0))
    assert callable(fn_custom)


def test_validate_config_rejects_unknown_eos_at_construction():
    """Fail-fast: an unknown eos must raise at config validation, not lazily at
    the first step. Uses the same VALID_EOS_SCHEMES source as make_eos_fn."""
    bad = LatLonCGridOceanConfig(eos="wrihgt")  # typo
    with pytest.raises(ValueError, match="eos must be one of"):
        LatLonCGridOceanModel._validate_config(bad)


def test_validate_config_accepts_every_valid_eos():
    """Each valid eos literal must pass construction validation — locking the
    validator's set to make_eos_fn's set (no fail-fast/lazy disagreement)."""
    for scheme in VALID_EOS_SCHEMES:
        cfg = LatLonCGridOceanConfig(eos=scheme)
        LatLonCGridOceanModel._validate_config(cfg)  # must not raise


# ---------------------------------------------------------------------------
# Footgun 2 — A_h single source of truth (lat-lon C-grid)
# ---------------------------------------------------------------------------


def test_A_h_single_source_competing_physics_harmonic_rejected():
    """The lat-lon C-grid takes A_h ONLY from config.A_h. A physics-pathway
    LateralMixingConfig with scheme='harmonic' carries a competing
    HarmonicConfig.A_h; that path is cubed-sphere-only, so it must be rejected
    at construction — leaving config.A_h as the sole A_h source."""
    competing = LatLonCGridOceanConfig(
        A_h=2.0e4,
        physics=OceanPhysicsConfig(
            lateral_mixing=LateralMixingConfig(
                scheme="harmonic", harmonic=HarmonicConfig(A_h=9.9e9),
            ),
        ),
    )
    with pytest.raises(ValueError, match="lat-lon C-grid"):
        LatLonCGridOceanModel._validate_config(competing)


def test_A_h_single_source_clean_config_passes():
    """With physics.lateral_mixing='none' there is no competing A_h source and
    construction validation passes; config.A_h is the single source."""
    clean = LatLonCGridOceanConfig(
        A_h=2.0e4,
        physics=OceanPhysicsConfig(
            lateral_mixing=LateralMixingConfig(scheme="none"),
        ),
    )
    LatLonCGridOceanModel._validate_config(clean)  # must not raise
    assert clean.A_h == 2.0e4


# ---------------------------------------------------------------------------
# Footgun 3 — momentum_advection dispatch (was a silent fallthrough)
# ---------------------------------------------------------------------------


def test_validate_config_rejects_unknown_momentum_advection():
    """An unknown momentum_advection literal must raise at construction, not
    silently fall through to vector-invariant (the prior behaviour)."""
    bad = LatLonCGridOceanConfig(momentum_advection="flux-form")  # typo/not-yet
    with pytest.raises(ValueError, match="momentum_advection must be one of"):
        LatLonCGridOceanModel._validate_config(bad)


def test_validate_config_accepts_valid_momentum_advection():
    """Each currently-valid momentum_advection literal passes validation."""
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        VALID_MOMENTUM_ADVECTION,
    )
    for scheme in VALID_MOMENTUM_ADVECTION:
        cfg = LatLonCGridOceanConfig(momentum_advection=scheme)
        LatLonCGridOceanModel._validate_config(cfg)  # must not raise


def test_validate_config_rejects_unknown_momentum_flux_scheme():
    """An unknown momentum_flux_scheme (used by momentum_advection='flux_form')
    must raise at construction."""
    bad = LatLonCGridOceanConfig(
        momentum_advection="flux_form", momentum_flux_scheme="quadratic",
    )
    with pytest.raises(ValueError, match="momentum_flux_scheme must be one of"):
        LatLonCGridOceanModel._validate_config(bad)


def test_validate_config_accepts_flux_form_with_valid_scheme():
    """flux_form with a valid momentum_flux_scheme passes construction."""
    for scheme in ("upwind", "centered"):
        cfg = LatLonCGridOceanConfig(
            momentum_advection="flux_form", momentum_flux_scheme=scheme,
        )
        LatLonCGridOceanModel._validate_config(cfg)  # must not raise
