"""Per-operator global heat ledger (``step_with_heat_ledger``).

The ledger is a diagnostic: it must not move the trajectory, its stages must
add up to the step's total content change, and its end content must be the
integral the heat-closure gate measures.  Tested on a small periodic lat-lon
C-grid with the eORCA cyclic overlap, real-freshwater closure, a surface heat
flux and P + runoff, so the freshwater-entry and surface stages are live.

Tolerances (detail choices, voted -- see the commit): the state comparison is
EXACT (``np.array_equal``); the telescoping sum and the cross-instrument check
against ``compute_tracer_budget`` hold to ``1e-12`` relative to the content.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

N_LAT, N_LON, N_STEPS, DT = 8, 16, 3, 600.0
REL_TOL = 1.0e-12


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    """The OMIP card runs PrecisionPolicy.fp64 (fp32 storage would quantise
    every content at ~1e-7 relative, far above the ledger's resolution)."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _model(perturb=True, **cfg_kw):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.core.field import Field
    grid = create_latlon_grid(N_LAT, N_LON)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    kw = dict(A_h=2.0e4, bottom_drag_r=1.0e-3, n_barotropic_substeps=8,
              enable_runtime_checks=False, ew_cyclic_overlap=True,
              freshwater_closure="real_freshwater")
    kw.update(cfg_kw)
    cfg = LatLonCGridOceanConfig.from_flat(**kw)
    rng = np.random.default_rng(1)
    T = np.asarray(state.T.data) + (rng.standard_normal(state.T.data.shape)
                                    if perturb else 0.0)
    T = T * np.asarray(state.land_mask.data)[..., None]
    state = state._replace(T=Field(jnp.asarray(T), name=state.T.name,
                                   dims=state.T.dims, units=state.T.units))
    return grid, z_coord, state, LatLonCGridOceanModel(grid, z_coord, cfg)


def _forcing(state):
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing
    shape = state.land_mask.data.shape
    lat = np.linspace(-1.0, 1.0, shape[0])[:, None] * np.ones(shape)
    q = jnp.asarray(80.0 * lat + 20.0)
    z = jnp.zeros(shape)
    sf = OceanSurfaceForcing(q_net=q, tau_x=jnp.full(shape, 0.05),
                             tau_y=z)
    fw = FreshwaterForcing(precip=jnp.full(shape, 3.0e-5), evap=z,
                           runoff=jnp.asarray(1.0e-4 * (lat > 0.5)),
                           ice_fw=z, restoring=None)
    return sf, fw


def _run(model, state, ledger, n=N_STEPS):
    sf, fw = _forcing(state)
    out = []
    for _ in range(n):
        if ledger:
            state, hl = model.step_with_heat_ledger(
                state, DT, surface_forcing=sf, freshwater=fw)
            out.append((state, {k: float(v) for k, v in hl.items()}))
        else:
            state = model.step(state, DT, surface_forcing=sf, freshwater=fw)
            out.append((state, None))
    return out


@pytest.fixture(scope="module")
def runs(_fp64):
    grid, z_coord, state, model = _model()
    return (grid, z_coord, state, model,
            _run(model, state, ledger=False), _run(model, state, ledger=True))


def test_state_bitwise_identical_with_ledger(runs):
    """Every leaf of the carried state (T, S, eta, u, v, tke, carries...)."""
    *_, off, on = runs
    for i, ((a, _), (b, _)) in enumerate(zip(off, on)):
        la, ta = jax.tree_util.tree_flatten(a)
        lb, tb = jax.tree_util.tree_flatten(b)
        assert ta == tb, f"step {i + 1}: pytree structure differs"
        assert len(la) >= 5
        for j, (x, y) in enumerate(zip(la, lb)):
            assert np.array_equal(np.asarray(x), np.asarray(y)), (
                f"step {i + 1}: leaf {j} moved")


def test_stages_telescope_to_the_step_change(runs):
    *_, on = runs
    prev_end = None
    for i, (_, hl) in enumerate(on):
        staged = sum(v for k, v in hl.items() if k.startswith("step_"))
        scale = abs(hl["entry"])
        assert abs(staged - (hl["end"] - hl["entry"])) <= REL_TOL * scale, i
        if prev_end is not None:      # the state handed on is the state read
            assert abs(hl["entry"] - prev_end) <= REL_TOL * scale, i
        prev_end = hl["end"]
        # every operator the card runs is present
        for k in ("step_explicit_rhs", "step_gm_redi", "step_advection",
                  "step_freshwater", "step_vertical", "step_ew_overlap"):
            assert k in hl, k


def test_explicit_rhs_split_adds_up(runs):
    *_, on = runs
    for _, hl in on:
        split = sum(v for k, v in hl.items() if k.startswith("split_rhs_"))
        assert abs(split - hl["step_explicit_rhs"]) <= REL_TOL * abs(hl["entry"])


def test_solver_thickness_matches_carried_thickness(runs):
    """The implicit solve conserves sum(dz_cell*T); on this coordinate its
    dz_cell is the carried h_new, so the mismatch split is round-off, and
    the advection flux-divergence split is reported."""
    *_, on = runs
    for _, hl in on:
        assert abs(hl["split_zdf_thickness_mismatch"]) <= REL_TOL * abs(hl["entry"])
        assert np.isfinite(hl["split_adv_flux_divergence"])
        assert np.isfinite(hl["split_adv_inactive_keep"])


def test_freshwater_stage_is_live(runs):
    """Runoff enters at max(SST, 0) under real_freshwater: non-zero heat."""
    *_, on = runs
    assert all(abs(hl["step_freshwater"]) > 0.0 for _, hl in on)


def test_end_content_matches_compute_tracer_budget(runs):
    """Cross-instrument: the ledger's end content is the closure gate's
    integral (compute_tracer_budget with the halo copies dropped)."""
    from types import SimpleNamespace
    from legoesm.ocean.budgets import compute_tracer_budget
    grid, z_coord, _, model, _, on = runs
    st, hl = on[-1]
    lm = np.asarray(st.land_mask.data).copy()
    lm[:, 0] = 0.0
    lm[:, -1] = 0.0
    f = lambda a: SimpleNamespace(data=a)  # noqa: E731
    probe = SimpleNamespace(T=st.T, S=st.S, eta=st.eta, H_bathy=st.H_bathy,
                            land_mask=f(lm))
    tb = compute_tracer_budget(probe, z_coord, grid_type="tripole",
                               grid=SimpleNamespace(area=np.asarray(grid.area)),
                               rho_0=1.0, c_p=1.0)
    assert abs(tb.heat_content - hl["end"]) <= REL_TOL * abs(hl["end"])


@pytest.mark.parametrize("implicit", [False, True])
def test_known_answer_surface_heat(implicit):
    """Known answer: a uniform-T rest ocean under a uniform q_net only (no
    shortwave, no water, no circulation) gains exactly dt * q_net * area /
    (rho_0 c_sw) over the counted columns -- in the step's TOTAL content
    change (delivered heat), and in the recorded source split, on both the
    explicit and the implicit surface route."""
    grid, _, state, model = _model(perturb=False, freshwater_closure="none",
                                   surface_forcing_implicit=implicit)
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.ocean.budgets import ledger_cell_weights
    shape = state.land_mask.data.shape
    q = 50.0
    sf = OceanSurfaceForcing(q_net=jnp.full(shape, q))
    _, hl = model.step_with_heat_ledger(state, DT, surface_forcing=sf)
    got = float(hl.get("split_rhs_surface", 0.0)) + float(
        hl.get("split_zdf_surface", 0.0))
    cc = model.config.constants
    w = np.asarray(ledger_cell_weights(grid.area, state.land_mask.data, True))
    want = DT * q * w.sum() / (cc.rho_0 * cc.c_sw)
    assert got == pytest.approx(want, rel=1.0e-10)
    # Delivered heat is a difference of two contents ~1e6 x larger, so it
    # resolves only to REL_TOL of the content (fp64 cancellation).
    tol = REL_TOL * abs(float(hl["entry"]))
    assert abs(float(hl["end"] - hl["entry"]) - want) <= tol
    staged = "step_vertical" if implicit else "step_explicit_rhs"
    assert abs(float(hl[staged]) - want) <= tol


def test_refuses_other_outer_integrators():
    _, _, state, model = _model(outer_integrator="ab2")
    with pytest.raises(ValueError, match="forward_euler"):
        model.step_with_heat_ledger(state, DT)


def test_ledger_weights_drop_halo_copies_once():
    from legoesm.ocean.budgets import ledger_cell_weights, tracer_content
    area = np.full((3, 6), 2.0)
    lm = np.ones((3, 6))
    w = np.asarray(ledger_cell_weights(area, lm, True))
    assert np.array_equal(w[:, [0, -1]], np.zeros((3, 2)))
    assert np.array_equal(w[:, 1:-1], np.full((3, 4), 2.0))
    assert np.array_equal(np.asarray(ledger_cell_weights(area, lm, False)), area)
    T = np.ones((3, 6, 2))
    h = np.full((3, 6, 2), 5.0)
    assert float(tracer_content(T, h, w)) == pytest.approx(3 * 4 * 2.0 * 2 * 5.0)


def test_refuses_private_hooks():
    """A hook that overrides the pre-solve tracers would be booked to an
    operator; the ledger refuses every non-default hook."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    grid, z_coord, state, model = _model()
    hooked = LatLonCGridOceanModel(
        grid, z_coord, model.config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            pre_implicit_tracer_override=(state.T.data, state.S.data)))
    with pytest.raises(ValueError, match="pre_implicit_tracer_override"):
        hooked.step_with_heat_ledger(state, DT)
