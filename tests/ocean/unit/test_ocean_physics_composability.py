"""Composability of ocean physics tendencies.

Any subset of {vertical_mixing, lateral_mixing, surface_forcing, convection}
can be enabled, and the combined tendency is the EXACT sum of the
individually-enabled processes: ``make_ocean_physics`` evaluates each enabled
process on the same input state and sums their tendencies, and ``scheme='none'``
disables a process (see ocean/physics/tendencies.make_none_physics_fn). Mirrors
the atmosphere additivity guarantee in
``tests/unit/test_physics_combined.py::test_tendency_additivity``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init import rest_state_ocean
from legoesm.ocean.physics.combined import OceanPhysicsConfig, make_ocean_physics
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig

# Per-process scheme exercised when "on" (others disabled via 'none').
_ON = {
    "vmix": "constant",
    "lateral": "harmonic",
    "surface": "restoring",
    "convection": "enhanced_diffusion",
}
_TEND_FIELDS = ("du_dt", "dv_dt", "dT_dt", "dS_dt")


def _state_grid_z(n=4, nlev=12):
    grid = create_cubed_sphere(n)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    state = rest_state_ocean(
        grid, z_coord, T_water_init_C=18.0, T_deep=2.0, S_uniform=35.0,
    )
    # Perturb T (horizontal + vertical structure) so lateral mixing has
    # gradients and convection sees occasional instability — makes the
    # additivity test exercise non-trivial per-process tendencies.
    rng = np.random.default_rng(0)
    noise = jnp.asarray(rng.normal(size=state.T.data.shape)) * 0.5
    return state._replace(T=state.T.replace(data=state.T.data + noise)), grid, z_coord


def _cfg(vmix="none", lateral="none", surface="none", convection="none"):
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme=vmix),
        lateral_mixing=LateralMixingConfig(scheme=lateral),
        surface_forcing=SurfaceForcingConfig(scheme=surface),
        bottom_drag=OceanPhysicsConfig().bottom_drag,  # deprecated -> 'none'
        convection=OceanConvectionConfig(scheme=convection),
        shortwave_penetration=None,
    )


def _run(cfg, state, grid, z_coord):
    return make_ocean_physics(cfg)(state, grid, z_coord, None)


def test_all_off_zero_tendencies():
    """Every scheme='none' -> identically zero tendencies (no-op physics)."""
    state, grid, z = _state_grid_z()
    t = _run(_cfg(), state, grid, z)
    for fld in _TEND_FIELDS:
        assert jnp.all(getattr(t, fld).data == 0.0), f"{fld} nonzero with all physics off"


def test_each_process_runs_finite():
    """Each single enabled process builds + runs, finite tendencies."""
    state, grid, z = _state_grid_z()
    for key, scheme in _ON.items():
        t = _run(_cfg(**{key: scheme}), state, grid, z)
        for fld in ("dT_dt", "du_dt"):
            assert jnp.all(jnp.isfinite(getattr(t, fld).data)), f"{key}: {fld} non-finite"


def test_subset_additivity():
    """combined(any subset) == EXACT sum of the individually-enabled processes."""
    state, grid, z = _state_grid_z()
    singles = {k: _run(_cfg(**{k: _ON[k]}), state, grid, z) for k in _ON}

    # Full set and a couple of proper subsets.
    subsets = [
        set(_ON),                       # all four
        {"vmix", "lateral"},            # vertical + horizontal mixing
        {"surface", "convection"},      # forcing + convection
        {"vmix", "lateral", "surface"}, # drop convection
    ]
    for subset in subsets:
        cfg = _cfg(**{k: _ON[k] for k in subset})
        t = _run(cfg, state, grid, z)
        for fld in _TEND_FIELDS:
            summed = sum(getattr(singles[k], fld).data for k in subset)
            err = float(jnp.max(jnp.abs(getattr(t, fld).data - summed)))
            assert err < 1e-10, (
                f"subset {sorted(subset)} {fld} additivity error {err:.2e}"
            )


def test_disabling_removes_exactly_that_contribution():
    """combined(full) - combined(full minus P) == P-only, for P=convection."""
    state, grid, z = _state_grid_z()
    t_full = _run(_cfg(**_ON), state, grid, z)
    minus = {**_ON, "convection": "none"}
    t_minus = _run(_cfg(**minus), state, grid, z)
    t_conv = _run(_cfg(convection=_ON["convection"]), state, grid, z)
    for fld in ("dT_dt", "dS_dt"):
        removed = getattr(t_full, fld).data - getattr(t_minus, fld).data
        err = float(jnp.max(jnp.abs(removed - getattr(t_conv, fld).data)))
        assert err < 1e-10, f"removing convection != conv-only for {fld}: {err:.2e}"


def test_at_least_one_process_contributes():
    """Sanity: the perturbed state actually excites physics (vmix diffuses the
    stratified column), so additivity isn't trivially 0==0 everywhere."""
    state, grid, z = _state_grid_z()
    t = _run(_cfg(vmix=_ON["vmix"]), state, grid, z)
    assert float(jnp.max(jnp.abs(t.dT_dt.data))) > 0.0
