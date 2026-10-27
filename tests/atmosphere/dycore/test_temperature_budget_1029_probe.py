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


_PS = _P.parent / "courant_series.py"
_spec_s = importlib.util.spec_from_file_location("courant_series_1029", _PS)
cs = importlib.util.module_from_spec(_spec_s)
_spec_s.loader.exec_module(cs)


def test_sustained_onset_is_last_upward_crossing():
    st = [10, 11, 12, 13, 14, 15]
    assert cs.sustained_onset(st, [1, 1, 0, 1, 1, 1]) == 13   # not first-ever
    assert cs.first_true(st, [0, 1, 0, 1, 1, 1]) == 11
    assert cs.sustained_onset(st, [1, 1, 1, 1, 1, 0]) is None
    assert cs.sustained_onset(st, [1] * 6) == 10                # window start


def test_row_envelope_skips_hot_and_neighbours_periodically():
    row = np.zeros((10, 5))
    zig = np.array([0.0, 8.0, 0.0, 8.0, 0.0])                  # A = 8
    row[0] = zig                                               # hot column
    row[9] = row[1] = row[8] = row[2] = 2.0 * zig              # +-2, wrapped
    row[5] = 0.5 * zig                                         # quiet: A = 4
    assert cs.row_envelope(row, 0) == 4.0
    assert cs.row_envelope(row, 0, exclude=1) == 16.0


def test_series_row_ties_to_step_budget(moving):
    model, st, _, _, _, _ = moving
    ij, tgt = (8, 5), model.compute_mass(st)
    _, _, info = tb.step_budget(model, st, tgt)
    r = cs.series_row(model, st, tgt, ij)
    C = [np.asarray(c)[ij] for c in info["courant_stages"]]
    assert r["C"] == float(np.max(C[0])) > 0.0
    assert r["C3"] == float(max(np.max(c) for c in C))
    assert r["C3"] > r["C"]                                    # stages differ
    assert r["ident"] < 1e-9 and not r["floor"]
    assert r["A"] == tb.swing_amplitude(np.asarray(st.T)[ij])
    assert C[0][r["k_C"]] == r["C"]
    Tc = np.asarray(st.T)[ij]
    assert abs(Tc[r["k_A"]] - 0.5 * (Tc[r["k_A"] - 1] + Tc[r["k_A"] + 1])) == r["A"]
    o = cs.onsets([dict(step=1, **r), dict(step=2, **dict(r, C=2.0, C3=2.0))])
    assert o["t_C"] == o["t_C3"] == 2 and o["window"] == [1, 2]


def test_controls_fail_on_planted_violations_and_window_checked():
    ok = dict(A=1.0, E=2.0, C=0.5, C3=0.5, floor=False, ident=1e-12, Cq=0.01)
    o = cs.onsets([dict(step=1, **ok), dict(step=2, **ok)])
    assert o["K3_pass"] and o["K4_pass"]
    o = cs.onsets([dict(step=1, **ok), dict(step=2, **dict(ok, ident=2e-9))])
    assert not o["K3_pass"] and o["K4_pass"]
    o = cs.onsets([dict(step=1, **dict(ok, Cq=0.2)), dict(step=2, **ok)])
    assert o["K3_pass"] and not o["K4_pass"]
    assert cs.readable(dict(K3_pass=True, K4_pass=True))["C_readable"]
    assert not cs.readable(dict(K3_pass=True, K4_pass=False))["C_readable"]
    assert not cs.readable(dict(K3_pass=False, K4_pass=True))["C_readable"]
    cs.check_window([7, 8, 9], 10, 3, 10)
    for steps, trip, n, exp in (([7, 9], 10, 2, 10), ([7, 8], 10, 2, 10),
                                ([8, 9], None, 2, 10), ([8, 9], 10, 3, 10),
                                ([7, 8, 9], 10, 3, 11)):
        with pytest.raises(SystemExit):
            cs.check_window(steps, trip, n, exp)


def test_compare_states_reports_and_refuses_missing(tmp_path):
    ref, new = tmp_path / "ref", tmp_path / "new"
    ref.mkdir(), new.mkdir()
    with pytest.raises(SystemExit):
        cs.compare_states(ref, new)                         # empty reference
    z = {f: np.zeros(3) for f in ("u", "v", "T", "p_s", "phis")}
    np.savez(ref / "clean_9.npz", **z)
    np.savez(new / "clean_9.npz", **dict(z, T=np.array([0.0, -2.0, 1.0])))
    np.savez(new / "clean_8.npz", **z)                      # extra new: fine
    assert cs.compare_states(ref, new) == {"clean_9": dict(u=0.0, v=0.0, T=2.0, p_s=0.0)}
    np.savez(ref / "clean_7.npz", **z)                      # not in new
    with pytest.raises(FileNotFoundError):
        cs.compare_states(ref, new)
