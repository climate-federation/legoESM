"""Direct test of scripts/cluster/dycore_1029/temperature_budget.py (#1029).

The budget must (a) leave the model step unchanged, (b) close: its terms sum
to the step's T change, and the RK-weighted recorded stage tendencies equal
the integrator's increment, and (c) not be vacuous: every captured term is
non-zero on a moving state, del-4 vanishes with nu_del4=0, and the floor term
catches a level pushed below T_min.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (  # noqa: E402
    CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel,
    hydrostatic_to_cgrid)
import legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid as _pe  # noqa: E402
from legoesm.atmosphere.forcing.idealized.held_suarez import (  # noqa: E402
    held_suarez_forcing_latlon)
from legoesm.atmosphere.idealized.held_suarez_topo import (  # noqa: E402
    held_suarez_topo_init_latlon)
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.grids.vertical import standard_hybrid_levels  # noqa: E402

_P = (Path(__file__).resolve().parents[3]
      / "scripts/cluster/dycore_1029/temperature_budget.py")
_spec = importlib.util.spec_from_file_location("temperature_budget_1029", _P)
tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tb)


def _model(nu4=1e17):
    grid = create_latlon_grid(16, 32)
    sigma = standard_hybrid_levels(20)
    cfg = CGridLatLonPrimitiveEquationConfig(
        A_h=1e5, fix_mass=True, anchor_mass_to_initial=True,
        use_polar_filter=True, sponge_coeff=1e-4, nu_del4=nu4)
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=300.0)
    st = hydrostatic_to_cgrid(
        held_suarez_topo_init_latlon(grid, sigma, h_0=2000.0, seed=1), grid)
    target = model.compute_mass(st)
    for _ in range(5):
        st = model.step(st, 300.0, target_mass=target,
                        physics_fn=held_suarez_forcing_latlon)
    return model, st, target


@pytest.fixture(scope="module")
def moving():
    model, st, target = _model()
    out, terms, info = tb.step_budget(model, st, target)
    ref, _ = model._step_cgrid_impl(st, model.dt, target,
                                    held_suarez_forcing_latlon)
    return model, st, out, terms, info, ref


def test_step_unchanged_and_patches_restored(moving):
    model, st, out, terms, info, ref = moving
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(getattr(out, f), getattr(ref, f),
                                   rtol=1e-13, atol=1e-12)
    assert _pe.dispatch_integrator.__module__ == "legoesm.timestepping.dispatch"
    assert "_apply_safety_rails" not in vars(model)
    assert "_apply_physics_increment" not in vars(model)


def test_budget_closes(moving):
    _, st, out, terms, info, _ = moving
    scale = float(jnp.max(jnp.abs(out.T - st.T)))
    assert scale > 1e-3
    assert float(jnp.max(jnp.abs(info["sum_vs_step"]))) < 1e-9 * scale
    assert float(jnp.max(jnp.abs(info["rk_vs_integrator"]))) < 1e-9 * scale


def test_terms_not_vacuous(moving):
    _, _, _, terms, _, _ = moving
    for n in ("H", "V", "OM", "G", "LAP", "D4", "HS"):
        assert float(jnp.max(jnp.abs(terms[n]))) > 1e-6, n


def test_del4_zero_when_off_and_floor_caught():
    model, st, target = _model(nu4=0.0)
    st = st._replace(T=st.T.at[8, 5, 12].set(20.0))
    out, terms, info = tb.step_budget(model, st, target)
    assert float(jnp.max(jnp.abs(terms["D4"]))) == 0.0
    assert float(terms["FL"][8, 5, 12]) > 0.0
    assert info["Tmin_prefloor"] < model.config.T_min
    # Courant identity holds on every RK stage, including stages that carry
    # sub-floor T (the tendency advects the floor-clipped field).
    for x, sc in zip(info["courant_ident"], info["courant_scale"]):
        assert float(jnp.max(jnp.abs(x))) < 1e-9 * sc
    assert float(out.T[8, 5, 12]) >= model.config.T_min


def test_courant_ties_to_captured_vertical_advection(moving):
    model, st, _, _, info, _ = moving
    C0 = np.asarray(info["courant_stages"][0])
    assert C0.shape == st.T.shape and np.max(C0) > 1e-4
    rel = np.max(np.abs(np.asarray(info["courant_ident"][0]))) / info["courant_scale"][0]
    assert rel < 1e-9
    # Known answer: uniform flux F through uniform spacing -> C = F dt / dp.
    from legoesm.grids.vertical import pressure_from_hybrid
    ps = st.p_s[:1, :1]
    mf = jnp.full(ps.shape + (model.sigma_coord.n_levels + 1,), 0.5)
    C, _ = tb.vertical_courant(st.T[:1, :1], mf, ps, model.sigma_coord, 100.0)
    dp = np.abs(np.diff(np.asarray(pressure_from_hybrid(
        model.sigma_coord, ps, full=True)), axis=-1))[0, 0]
    np.testing.assert_allclose(np.asarray(C)[0, 0, 1:], 50.0 / dp, rtol=1e-12)
    assert float(C[0, 0, 0]) == 0.0


def test_floor_free_run_sets_floor_heat_aside():
    model, st, target = _model(nu4=0.0)
    st = st._replace(T=st.T.at[8, 5, 12].set(20.0))
    _, terms, info = tb.step_budget(model, st, target)
    rows = tb.floor_free_run(model, st, target, 1, (8, 5))
    assert terms["FL"][8, 5, 12] > 0.0
    assert rows[1]["Tmin"] < model.config.T_min        # floor heat removed
    assert rows[0]["A"] == tb.swing_amplitude(np.asarray(st.T)[8, 5])
    assert tb.swing_amplitude([0.0, 10.0, 0.0, 10.0, 0.0]) == 10.0
