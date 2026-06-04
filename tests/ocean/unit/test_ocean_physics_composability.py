"""Composability of ocean physics tendencies.

Any subset of the five ocean physics processes
{vertical_mixing, lateral_mixing, surface_forcing, convection,
shortwave_penetration} can be enabled, and the combined tendency is the EXACT
sum of the individually-enabled processes: ``make_ocean_physics`` evaluates each
enabled process on the same input state and sums their tendencies. A process is
disabled by setting its sub-config ``scheme='none'`` (vmix / lateral / surface /
convection — see ocean/physics/tendencies.make_none_physics_fn) or, for
shortwave penetration, by setting ``shortwave_penetration=None``. Mirrors the
atmosphere additivity guarantee in
``tests/unit/test_physics_combined.py::test_tendency_additivity``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init import rest_state_ocean
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.ocean.physics.combined import OceanPhysicsConfig, make_ocean_physics
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.shortwave_penetration import ShortwavePenetrationConfig
from legoesm.ocean.physics.vertical_mixing.tidal import TidalMixingConfig

_N = 4
_NLEV = 12
_TEND_FIELDS = ("du_dt", "dv_dt", "dT_dt", "dS_dt")

# Each composable process -> the _cfg kwargs that enable it (others stay off).
# shortwave_penetration is a config (not a scheme field) + needs forcing.sw_down.
_PROCS = {
    "vmix": dict(vmix="constant"),
    "lateral": dict(lateral="harmonic"),
    "surface": dict(surface="restoring"),
    "convection": dict(convection="enhanced_diffusion"),
    "shortwave": dict(shortwave=True),
}


def _state_grid_z(n=_N, nlev=_NLEV):
    grid = create_cubed_sphere(n)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    state = rest_state_ocean(
        grid, z_coord, T_water_init_C=18.0, T_deep=2.0, S_uniform=35.0,
    )
    # Perturb T (horizontal + vertical) so lateral mixing has gradients and
    # convection sees occasional instability — makes additivity exercise
    # non-trivial per-process tendencies.
    rng = np.random.default_rng(0)
    noise = jnp.asarray(rng.normal(size=state.T.data.shape)) * 0.5
    return state._replace(T=state.T.replace(data=state.T.data + noise)), grid, z_coord


def _forcing(n=_N):
    # Downward SW only — exercises the shortwave-penetration heating path
    # (surface_forcing='restoring' ignores this arg; shortwave reads sw_down).
    return OceanSurfaceForcing(sw_down=jnp.full((6, n, n), 200.0))


def _cfg(vmix="none", lateral="none", surface="none", convection="none",
         shortwave=False):
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme=vmix),
        lateral_mixing=LateralMixingConfig(scheme=lateral),
        surface_forcing=SurfaceForcingConfig(scheme=surface),
        bottom_drag=OceanPhysicsConfig().bottom_drag,  # deprecated -> 'none'
        convection=OceanConvectionConfig(scheme=convection),
        shortwave_penetration=ShortwavePenetrationConfig() if shortwave else None,
    )


def _cfg_for(names):
    kw = {}
    for name in names:
        kw.update(_PROCS[name])
    return _cfg(**kw)


def _run(cfg, state, grid, z_coord, forcing):
    return make_ocean_physics(cfg)(state, grid, z_coord, forcing)


def test_all_off_zero_tendencies():
    """All five processes disabled -> identically zero tendencies, even with a
    forcing object present (the SW path is gated on shortwave_penetration)."""
    state, grid, z = _state_grid_z()
    t = _run(_cfg(), state, grid, z, _forcing())
    for fld in _TEND_FIELDS:
        assert jnp.all(getattr(t, fld).data == 0.0), f"{fld} nonzero with all physics off"


def test_each_process_runs_finite():
    """Each single enabled process builds + runs with finite tendencies."""
    state, grid, z = _state_grid_z()
    f = _forcing()
    for name in _PROCS:
        t = _run(_cfg_for([name]), state, grid, z, f)
        for fld in ("dT_dt", "du_dt"):
            assert jnp.all(jnp.isfinite(getattr(t, fld).data)), f"{name}: {fld} non-finite"


def test_subset_additivity():
    """combined(any subset) == EXACT sum of the individually-enabled processes."""
    state, grid, z = _state_grid_z()
    f = _forcing()
    singles = {name: _run(_cfg_for([name]), state, grid, z, f) for name in _PROCS}

    subsets = [
        list(_PROCS),                              # all five
        ["vmix", "lateral"],                       # vertical + horizontal mixing
        ["surface", "convection"],                 # forcing + convection
        ["vmix", "lateral", "shortwave"],          # mixing + SW heating
        ["convection", "shortwave"],               # convection + SW heating
    ]
    for subset in subsets:
        t = _run(_cfg_for(subset), state, grid, z, f)
        for fld in _TEND_FIELDS:
            summed = sum(getattr(singles[k], fld).data for k in subset)
            err = float(jnp.max(jnp.abs(getattr(t, fld).data - summed)))
            assert err < 1e-10, f"subset {subset} {fld} additivity error {err:.2e}"


def test_shortwave_only_heats_temperature_only():
    """shortwave_penetration on, all schemes off -> nonzero dT_dt, zero u/v/S."""
    state, grid, z = _state_grid_z()
    t = _run(_cfg(shortwave=True), state, grid, z, _forcing())
    assert float(jnp.max(jnp.abs(t.dT_dt.data))) > 0.0, "SW penetration did not heat"
    for fld in ("du_dt", "dv_dt", "dS_dt"):
        assert jnp.all(getattr(t, fld).data == 0.0), f"SW penetration touched {fld}"


@pytest.mark.parametrize("vmix_scheme", ["constant", "none"])
def test_tidal_enabled_rejected_not_silently_ignored(vmix_scheme):
    """Tidal mixing rides on VerticalMixingConfig but is a SEPARATE caller-applied
    additive step (apply_tidal_mixing_step), not part of the make_ocean_physics
    composition. Enabling it on the composition config must RAISE, not silently
    no-op — including when vertical_mixing.scheme == 'none' (which bypasses
    make_vertical_mixing_physics, so make_ocean_physics guards it directly)."""
    cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme=vmix_scheme, tidal=TidalMixingConfig(enabled=True),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=OceanPhysicsConfig().bottom_drag,
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )
    with pytest.raises(NotImplementedError, match="tidal"):
        make_ocean_physics(cfg)


def test_disabling_removes_exactly_that_contribution():
    """combined(full) - combined(full minus P) == P-only, for P in {convection, shortwave}."""
    state, grid, z = _state_grid_z()
    f = _forcing()
    t_full = _run(_cfg_for(list(_PROCS)), state, grid, z, f)
    for proc in ("convection", "shortwave"):
        rest = [p for p in _PROCS if p != proc]
        t_rest = _run(_cfg_for(rest), state, grid, z, f)
        t_only = _run(_cfg_for([proc]), state, grid, z, f)
        for fld in _TEND_FIELDS:
            removed = getattr(t_full, fld).data - getattr(t_rest, fld).data
            err = float(jnp.max(jnp.abs(removed - getattr(t_only, fld).data)))
            assert err < 1e-10, f"removing {proc} != {proc}-only for {fld}: {err:.2e}"
