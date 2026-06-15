"""Eady-uniform corrected recipe (the audited eddy-resolving rebuild).

Covers ``build_eady_uniform_setup`` → ``EadyUniformRecipe``: the canonical recipe
shape (matches ACCRecipe), the corrected Phase-G dycore stack wired by default,
both dycore tracks (WENO5 flux-form + NEMO-like vector-invariant+Hollingsworth),
the dissipation knobs, and that a forward step runs (scan-carry stable).

Context: the legacy matrix path wired an April-2026 baseline (explicit-substep
barotropic, forward-Euler integrators, centered-KE vector-invariant momentum, TVD
tracers, force-enabled KPP). This builder selects the current canonical blocks.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
from legoesm.ocean.experiments.eady_uniform import (
    build_eady_uniform_setup, EadyUniformRecipe, EadyUniformConfig,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel


@pytest.fixture(autouse=True)
def _fp64():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def test_recipe_shape_and_corrected_stack():
    """Returns an EadyUniformRecipe (ACCRecipe-shaped) with the corrected stack."""
    r = build_eady_uniform_setup(n_lat=24, n_lon=24)
    assert isinstance(r, EadyUniformRecipe)
    assert r._fields == ("model_config", "physics_config", "grid", "z_coord",
                         "wall_mask", "initial_state")
    c = r.model_config
    # Corrected eddy-resolving dycore parts (vs the stale April-2026 baseline).
    assert c.barotropic_solver == "implicit_cn"
    assert c.tracer_advection == "weno5"
    assert c.momentum_advection == "weno5"
    assert c.tracer_time_integrator == "rk3"
    assert c.outer_integrator == "ab2"
    assert c.pgf_scheme == "smc03"
    # Mesoscale parameterization OFF (eddies resolved); adiabatic BCI.
    assert c.gm_redi is None
    assert c.eos == "linear"
    assert r.physics_config.vertical_mixing.scheme == "none"   # KPP OFF
    LatLonCGridOceanModel(r.grid, r.z_coord, c)                 # constructs + validates


def test_nemo_like_track_builds():
    """The NEMO-like track: vector-invariant momentum + Hollingsworth KE gradient."""
    r = build_eady_uniform_setup(
        n_lat=24, n_lon=24, momentum_advection="vector_invariant",
        ke_gradient_scheme="hollingsworth")
    assert r.model_config.momentum_advection == "vector_invariant"
    assert r.model_config.ke_gradient_scheme == "hollingsworth"
    LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)


def test_dissipation_knobs_wired():
    """A_h (harmonic), B_h (biharmonic), C_smag, C_leith are selectable."""
    r = build_eady_uniform_setup(n_lat=24, n_lon=24, a_h=1500.0, b_h=2.5e11,
                                 c_smag=0.0, c_leith=2.0, smag_cfl_safety=0.5)
    c = r.model_config
    assert c.A_h == 1500.0 and c.B_h == 2.5e11
    assert c.C_smag == 0.0 and c.C_leith == 2.0


def test_forward_step_runs_and_is_finite():
    """One forward step from the thermal-wind IC runs, finite, scan-carry stable."""
    from legoesm.core.field import Field
    from jax.tree_util import tree_structure

    r = build_eady_uniform_setup(n_lat=24, n_lon=24, a_h=1500.0, c_smag=0.0)
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)
    s = r.initial_state
    # AB2 outer needs prior-increment carries seeded (constant scan pytree).
    def _z(d):
        return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                     dims=d.dims, units=d.units)
    s = s._replace(T_incr_prev=_z(s.T), S_incr_prev=_z(s.S),
                   u_incr_prev=_z(s.u), v_incr_prev=_z(s.v))
    nxt = model.step(s, 300.0)
    assert bool(jnp.all(jnp.isfinite(nxt.u.data)))
    assert bool(jnp.all(jnp.isfinite(nxt.T.data)))
    assert tree_structure(nxt) == tree_structure(s)


def test_eddy_resolving_min_dissipation_recipe():
    """The validated eddy-resolving min-dissipation combination builds + steps.

    A_h≈1000 (Laplacian) + C_smag≈0.1 (biharmonic Smag) + smag_cfl_safety — the
    ralph-loop result: stable, spectrally clean, strong eddies at 120×120 (pure
    Laplacian over-damps, pure biharmonic blows up; the combination is the min).
    """
    from legoesm.core.field import Field

    r = build_eady_uniform_setup(n_lat=24, n_lon=24, a_h=1000.0, c_smag=0.1,
                                 smag_cfl_safety=0.5)
    c = r.model_config
    assert c.A_h == 1000.0 and c.C_smag == 0.1 and c.smag_cfl_safety == 0.5
    assert c.B_h == 0.0 and c.C_leith == 0.0          # combination is A_h + C_smag only
    model = LatLonCGridOceanModel(r.grid, r.z_coord, c)
    s = r.initial_state
    def _z(d):
        return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                     dims=d.dims, units=d.units)
    s = s._replace(T_incr_prev=_z(s.T), S_incr_prev=_z(s.S),
                   u_incr_prev=_z(s.u), v_incr_prev=_z(s.v))
    nxt = model.step(s, 600.0)
    assert bool(jnp.all(jnp.isfinite(nxt.u.data)))


def test_eady_growth_rate_formula():
    """The config's analytical Eady growth rate is the textbook σ=0.31 f Λ/N."""
    import numpy as np
    cfg = EadyUniformConfig(U_surface=0.2)
    sigma = 0.31 * cfg.f0 * cfg.Lambda / cfg.N
    assert cfg.efolding_days == pytest.approx(1.0 / sigma / 86400.0)
    assert 15.0 < cfg.efolding_days < 25.0          # weak regime τ≈20 d
