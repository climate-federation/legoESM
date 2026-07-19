"""Dispatch hardening: an unknown ``turbulence_closure`` must raise.

The plane CRM selects its SGS closure by a plain equality chain
(``_use_smag = _closure == "smagorinsky"`` and friends).  A typo'd closure
name matches NO branch, so every ``_use_*`` flag is False and the whole SGS
block is skipped — SILENTLY running with zero sub-grid turbulence, which is
INDISTINGUISHABLE from the intentional inviscid ``"none"`` mode.  That is a
different-physics-on-a-typo bug (CLAUDE.md: every scheme dispatch must raise
on unknown).

``validate_plane_config`` (called at model construction) is the fail-early
home for this static check; these tests pin that it rejects a typo and
accepts every VALID closure name.
"""

from __future__ import annotations

import jax
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    validate_plane_config,
)

jax.config.update("jax_enable_x64", True)

# Inviscid baseline: all diffusion off so the only field under test is the
# closure name (each valid closure supplies its own required coefficient).
_BASE = dict(sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
             hyperdiff_w_coeff=0.0)


def test_validate_rejects_unknown_turbulence_closure():
    cfg = CompressibleEulerConfig(turbulence_closure="smag", **_BASE)  # typo
    with pytest.raises(ValueError, match="turbulence_closure"):
        validate_plane_config(cfg)


@pytest.mark.parametrize("closure,extra", [
    ("smagorinsky", dict(smagorinsky_cs=0.2)),
    ("none", {}),
    ("vreman", dict(vreman_c=0.07)),
    ("amd", dict(amd_c=0.3)),
    ("molecular", dict(molecular_viscosity=1.5e-5, molecular_prandtl=0.7)),
])
def test_validate_accepts_known_turbulence_closures(closure, extra):
    """Every VALID closure name passes validation (guard is not over-eager)."""
    cfg = CompressibleEulerConfig(turbulence_closure=closure, **_BASE, **extra)
    validate_plane_config(cfg)          # must not raise
