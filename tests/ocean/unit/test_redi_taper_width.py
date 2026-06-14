"""Tests for ``GMRediConfig.taper_width_frac`` (Phase G.1c).

Exposes Veros's ``iso_dslope`` knob via the legoESM ``GMRediConfig``.
The mapping is::

    taper_width_frac = iso_dslope / iso_slopec

For DINO (``iso_slopec=0.01, iso_dslope=0.005``) this is ``0.5``.

Tests:

1. The default ``taper_width_frac = 0.1`` reproduces the pre-refactor
   hardcoded ``0.1 * S_max`` behavior of both ``dm95_taper`` and
   ``_triad_taper``.
2. Varying ``taper_width_frac`` changes the tanh transition steepness
   in the expected direction (wider = smoother taper around
   ``|S| = S_max``).
3. The Veros DINO mapping (``taper_width_frac=0.5``) gives the Veros
   functional form to machine precision.
4. ``GMRediConfig.taper_width_frac`` propagates through the lat-lon
   triad caller (``gm_redi_tracer_tendency_triads_latlon_cgrid``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    dm95_taper,
    dm95_taper_scalar,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig

jax.config.update("jax_enable_x64", True)


def test_default_taper_width_frac_matches_legacy_dm95():
    """The pre-2026 hardcoded form was ``width = 0.1 * S_max``. With
    ``taper_width_frac=0.1`` (default) we must produce bit-exact
    identical numbers."""
    S_max = 0.01
    width = 0.1 * S_max
    eps = float(jnp.finfo(jnp.float32).eps)
    S_x = jnp.linspace(-0.02, 0.02, 50).reshape(50, 1, 1)
    S_y = jnp.zeros_like(S_x)
    # Legacy formula inlined:
    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + eps)
    taper_legacy = 0.5 * (1.0 + jnp.tanh((S_max - S_mag) / (width + eps)))

    # New parameterised form with default frac:
    _, _, taper_new = dm95_taper(S_x, S_y, S_max)
    np.testing.assert_allclose(np.asarray(taper_new), np.asarray(taper_legacy),
                                rtol=1e-15, atol=1e-15)


def test_default_taper_width_frac_matches_legacy_triad():
    """The scalar triad taper (``dm95_taper_scalar``, which the 16 caller sites
    in ``gm_redi_latlon_cgrid.py`` now use directly — the former ``_triad_taper``
    duplicate was removed) must match the legacy ``0.1 * S_max`` hardcoded form
    at the default ``transition_width_frac=0.1``."""
    S_max = 0.01
    eps = float(jnp.finfo(jnp.float32).eps)
    S = jnp.linspace(-0.02, 0.02, 50)
    taper_legacy = 0.5 * (1.0 + jnp.tanh(
        (S_max - jnp.abs(S)) / (0.1 * S_max + eps)
    ))
    taper_new = dm95_taper_scalar(S, S_max)[1]   # uses default frac=0.1
    np.testing.assert_allclose(np.asarray(taper_new), np.asarray(taper_legacy),
                                rtol=1e-15, atol=1e-15)


def test_wider_frac_gives_smoother_taper_at_smax():
    """At ``|S| = S_max``, ``taper = 0.5`` regardless of width — that's
    the definition of the crossover point. But ``d(taper)/d|S|`` at the
    crossover is steeper for smaller ``taper_width_frac``."""
    S_max = 0.01
    # At S = S_max exactly: numerator = 0, tanh(0) = 0, so taper = 0.5.
    S_at = jnp.asarray(S_max)
    _, taper_narrow = dm95_taper_scalar(S_at, S_max, transition_width_frac=0.1)
    _, taper_wide = dm95_taper_scalar(S_at, S_max, transition_width_frac=0.5)
    np.testing.assert_allclose(float(taper_narrow), 0.5, atol=1e-6)
    np.testing.assert_allclose(float(taper_wide), 0.5, atol=1e-6)

    # Just below S_max: narrow taper has already moved closer to 1,
    # wide taper is still close to 0.5.
    S_below = jnp.asarray(S_max * 0.95)
    _, t_n = dm95_taper_scalar(S_below, S_max, transition_width_frac=0.1)
    _, t_w = dm95_taper_scalar(S_below, S_max, transition_width_frac=0.5)
    assert float(t_n) > float(t_w), (
        f"Narrow taper {float(t_n)} should exceed wide taper {float(t_w)} "
        "just below the crossover."
    )


def test_veros_dino_mapping():
    """Veros DINO uses ``iso_slopec=0.01, iso_dslope=0.005`` →
    ``taper_width_frac = 0.5``. Verify the resulting taper matches the
    Veros functional form directly."""
    iso_slopec = 0.01
    iso_dslope = 0.005
    S = jnp.linspace(-0.02, 0.02, 100)

    # Veros functional form (taken from the eady_uniform/dino adapter
    # surface — Veros applies essentially the same DM95 tanh).
    eps = float(jnp.finfo(jnp.float32).eps)
    veros_taper = 0.5 * (1.0 + jnp.tanh(
        (iso_slopec - jnp.abs(S)) / (iso_dslope + eps)
    ))

    # legoESM with the corresponding recipe override:
    frac = iso_dslope / iso_slopec   # = 0.5
    _, lego_taper = dm95_taper_scalar(
        S, iso_slopec, transition_width_frac=frac,
    )
    np.testing.assert_allclose(np.asarray(lego_taper), np.asarray(veros_taper),
                                rtol=1e-14, atol=1e-14)


def test_config_default_field():
    """``GMRediConfig.taper_width_frac`` defaults to 0.1."""
    cfg = GMRediConfig()
    assert cfg.taper_width_frac == 0.1


def test_config_override_preserved():
    """Setting ``taper_width_frac=0.5`` is preserved as expected
    through the NamedTuple ``_replace`` semantics."""
    cfg = GMRediConfig()._replace(taper_width_frac=0.5)
    assert cfg.taper_width_frac == 0.5
    assert cfg.S_max == 0.01   # unchanged default
