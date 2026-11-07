"""Physics water tendencies carry their MASS on the p_s-coordinate lanes
(``apply_physics_water_mass``, the FV3 ``fv_update_phys`` nwat block on a
sigma/hybrid column).

Budget identities, exact to rounding, on sigma AND hybrid coordinates:
  * column water change == dt * sum_k dq_k * dp_k / g  (what physics meant)
  * column DRY mass  sum_k dp_k (1 - Q_k) / g  invariant
  * an inert passenger's column mass invariant, a number concentration too
  * p_s falls by exactly g * (precipitated water) when rain leaves
  * the old behaviour (q += dt*dq at fixed p_s) FAILS the dry-mass identity
    by exactly the precipitated water -- the defect this replaces.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.core.conservation import (  # noqa: E402
    WATER_MASS_SPECIES,
    apply_physics_water_mass,
)
from legoesm.core.field import Field  # noqa: E402
from legoesm.grids.vertical import (  # noqa: E402
    create_hybrid_coordinate,
    create_sigma_coordinate,
    make_cam6_l32_levels,
)

DT = 600.0
NCOL, NLEV = 5, 12


def _coords():
    return {
        "sigma": create_sigma_coordinate(NLEV, dtype=jnp.float64),
        "hybrid": make_cam6_l32_levels(),
    }


def _state(coord):
    nlev = int(np.asarray(coord.dsigma).shape[0])
    rng = np.random.default_rng(1)
    p_s = jnp.asarray(1.0e5 * (1.0 + 0.05 * rng.standard_normal(NCOL)))
    q_v = jnp.asarray(0.02 * rng.uniform(0.2, 1.0, (NCOL, nlev)))
    q_c = jnp.asarray(1e-3 * rng.uniform(0.0, 1.0, (NCOL, nlev)))
    q_r = jnp.asarray(2e-3 * rng.uniform(0.0, 1.0, (NCOL, nlev)))
    trc = jnp.asarray(rng.uniform(0.1, 1.0, (NCOL, nlev)))          # passenger
    N_c = jnp.asarray(1e8 * rng.uniform(0.1, 1.0, (NCOL, nlev)))    # per-mass number
    f = lambda a, nm: Field(data=a, name=nm, dims=("nCells", "level"), units="1")  # noqa: E731
    tr = {"q_v": f(q_v, "q_v"), "q_c": f(q_c, "q_c"), "q_r": f(q_r, "q_r"),
          "trc_o3": f(trc, "trc_o3"), "N_c": f(N_c, "N_c")}
    # tendencies: condensation (v -> c, mass-neutral), rain fall-out from
    # the lowest 3 layers (mass LEAVES), evaporation into the top layer
    dq_v = -2e-7 * rng.uniform(0.0, 1.0, (NCOL, nlev))
    dq_c = -dq_v
    dq_r = jnp.zeros((NCOL, nlev)).at[:, -3:].set(-1e-6 * rng.uniform(0.5, 1.0, (NCOL, 3)))
    dq_v = jnp.asarray(dq_v).at[:, 0].add(3e-7)
    tend = {"q_v": f(dq_v, "dq_v"), "q_c": f(dq_c, "dq_c"), "q_r": f(dq_r, "dq_r")}
    return tr, tend, p_s


def _column(q, dp):
    return np.asarray(jnp.sum(q * dp, axis=-1) / constants.g)


@pytest.mark.parametrize("name", ["sigma", "hybrid"])
def test_budget_identities(name):
    coord = _coords()[name]
    tr, tend, p_s = _state(coord)
    dp0 = coord.layer_thickness_dp(p_s)
    tr1, p_s1 = apply_physics_water_mass(tr, tend, p_s, coord, DT)
    dp1 = coord.layer_thickness_dp(p_s1)
    assert p_s1.shape == p_s.shape
    assert set(tr1) == set(tr)

    Q0 = sum(tr[k].data for k in WATER_MASS_SPECIES if k in tr)
    Q1 = sum(tr1[k].data for k in WATER_MASS_SPECIES if k in tr1)
    water0, water1 = _column(Q0, dp0), _column(Q1, dp1)
    meant = _column(sum(DT * tend[k].data for k in tend), dp0)
    scale = np.abs(water0).max()
    # (i) water change is what physics meant
    np.testing.assert_allclose(water1 - water0, meant, rtol=0, atol=1e-12 * scale)
    assert np.abs(meant).max() > 1e-6 * scale, "vacuous: no water moved"
    # (ii) dry mass in the LAYERS invariant up to the coordinate's own cap
    #      response: on sigma p_top = sigma_top*p_s moves with p_s (the cap
    #      is dry air, so sigma converts sigma_top of the water flux into
    #      cap air -- the fixer restores it); on hybrid p_top = ak[0] is
    #      fixed and the layer dry mass is exact.
    dry0, dry1 = _column(1.0 - Q0, dp0), _column(1.0 - Q1, dp1)
    ptop0 = np.asarray(coord.pressure_at_half(p_s)[..., 0]) / constants.g
    ptop1 = np.asarray(coord.pressure_at_half(p_s1)[..., 0]) / constants.g
    np.testing.assert_allclose(dry1 + ptop1, dry0 + ptop0, rtol=1e-11, atol=0)
    if name == "hybrid":
        np.testing.assert_allclose(dry1, dry0, rtol=1e-11, atol=0)
    # (iii) passengers: column mass invariant
    for k in ("trc_o3", "N_c"):
        np.testing.assert_allclose(_column(tr1[k].data, dp1), _column(tr[k].data, dp0),
                                   rtol=1e-11, atol=0)
    # (iv) p_s moves by g * water change, exactly
    np.testing.assert_allclose(np.asarray(p_s1 - p_s), constants.g * meant,
                               rtol=1e-11, atol=0)
    # (v) the OLD behaviour (q += dt*dq at fixed p_s) converts rain into
    #     dry air by exactly the precipitated water: the defect replaced
    Q_old = sum(tr[k].data + DT * tend[k].data for k in tend)
    dry_old = _column(1.0 - Q_old, dp0)
    np.testing.assert_allclose(dry_old - dry0, -meant, rtol=0, atol=1e-11 * scale)
    assert np.abs(dry_old - dry0).max() > 1e-6 * scale


def test_no_water_tendency_keeps_ps_and_applies_passenger_tendencies():
    coord = _coords()["sigma"]
    tr, tend, p_s = _state(coord)
    tr1, p_s1 = apply_physics_water_mass(tr, {}, p_s, coord, DT)
    assert p_s1 is p_s
    for k in tr:
        assert tr1[k] is tr[k]
    # a passenger-only tendency (e.g. ozone chemistry) is applied, p_s
    # untouched, water untouched (codex 2026-09-28: the first draft
    # returned early and dropped it)
    dtrc = tr["trc_o3"].replace(data=jnp.full_like(tr["trc_o3"].data, 1e-3))
    tr2, p_s2 = apply_physics_water_mass(tr, {"trc_o3": dtrc}, p_s, coord, DT)
    assert p_s2 is p_s
    # a water tendency for a species the state does NOT carry (q_i on a
    # warm-rain deck) adds no mass: p_s untouched (codex r2)
    dqi = tr["q_v"].replace(data=jnp.full_like(tr["q_v"].data, -1e-6))
    tr3, p_s3 = apply_physics_water_mass(tr, {"q_i": dqi}, p_s, coord, DT)
    assert p_s3 is p_s and "q_i" not in tr3
    np.testing.assert_allclose(np.asarray(tr2["trc_o3"].data),
                               np.asarray(tr["trc_o3"].data) + DT * 1e-3, rtol=1e-15)
    assert tr2["q_v"] is tr["q_v"]


def test_shift_ps_keep_tracer_mass_is_dry_air_only():
    from legoesm.core.conservation import dry_surface_pressure, shift_ps_keep_tracer_mass
    for name, coord in _coords().items():
        tr, _, p_s = _state(coord)
        corr = jnp.asarray(37.0)
        p_s1, tr1 = shift_ps_keep_tracer_mass(p_s, tr, coord, corr)
        dp0, dp1 = coord.layer_thickness_dp(p_s), coord.layer_thickness_dp(p_s1)
        for k in tr:
            np.testing.assert_allclose(_column(tr1[k].data, dp1), _column(tr[k].data, dp0),
                                       rtol=1e-12, atol=0, err_msg=f"{name}:{k}")
        d0 = np.asarray(dry_surface_pressure(p_s, tr, coord))
        d1 = np.asarray(dry_surface_pressure(p_s1, tr1, coord))
        np.testing.assert_allclose(d1 - d0, 37.0, rtol=1e-11, atol=0, err_msg=name)
        # the UNWEIGHTED shift is not exact: dry moves by less than 37 Pa
        d_unw = np.asarray(dry_surface_pressure(p_s1, tr, coord))
        assert np.all(d_unw - d0 < 37.0 * (1 - 1e-4)), name


def test_mass_neutral_phase_change_leaves_ps_and_layer_masses():
    coord = _coords()["hybrid"]
    tr, tend, p_s = _state(coord)
    tend = {"q_v": tend["q_v"].replace(data=-tend["q_c"].data), "q_c": tend["q_c"]}
    tr1, p_s1 = apply_physics_water_mass(tr, tend, p_s, coord, DT)
    np.testing.assert_array_equal(np.asarray(p_s1), np.asarray(p_s))
    np.testing.assert_allclose(np.asarray(tr1["q_v"].data),
                               np.asarray(tr["q_v"].data + DT * tend["q_v"].data),
                               rtol=1e-15, atol=0)
    np.testing.assert_array_equal(np.asarray(tr1["trc_o3"].data),
                                  np.asarray(tr["trc_o3"].data))


def test_jit_and_grad_flow():
    coord = _coords()["sigma"]
    tr, tend, p_s = _state(coord)
    raw = {k: v.data for k, v in tr.items()}
    traw = {k: v.data for k, v in tend.items()}

    def f(p_s_, q_v):
        out, ps1 = apply_physics_water_mass({**raw, "q_v": q_v}, traw, p_s_, coord, DT)
        return jnp.sum(ps1) + jnp.sum(out["q_v"] * coord.layer_thickness_dp(ps1))

    g_ps, g_q = jax.grad(f, argnums=(0, 1))(p_s, raw["q_v"])
    assert np.isfinite(np.asarray(g_ps)).all() and np.abs(np.asarray(g_q)).max() > 0
    a = jax.jit(f)(p_s, raw["q_v"]); b = f(p_s, raw["q_v"])
    np.testing.assert_allclose(float(a), float(b), rtol=1e-14)


# ---------------------------------------------------------------------------
# Through the MPAS step: physics application + DRY-mass fixer together
# ---------------------------------------------------------------------------

def _rain_only_physics(rate):
    """Rain leaves the lowest three layers at ``rate`` [1/s]; nothing else."""
    from legoesm.core.state import MPASHydrostaticTendencies

    def physics(state, mesh, sigma, forcing=None, phys_state=None):
        nc, nl = state.T.data.shape
        z = jnp.zeros((nc, nl))
        dq_r = z.at[:, -3:].set(-rate)
        f = lambda a, nm, dims, u: Field(data=a, name=nm, dims=dims, units=u)  # noqa: E731
        return MPASHydrostaticTendencies(
            du_dt=f(jnp.zeros_like(state.u.data), "du_dt", ("nEdges", "level"), "m/s^2"),
            dT_dt=f(z, "dT_dt", ("nCells", "level"), "K/s"),
            dp_s_dt=f(jnp.zeros((nc,)), "dp_s_dt", ("nCells",), "Pa/s"),
            dphis_dt=f(jnp.zeros((nc,)), "dphis_dt", ("nCells",), "m^2/s^3"),
            tracer_tendencies={"q_r": f(dq_r, "dq_r_dt", ("nCells", "level"), "kg/kg/s")},
        )
    return physics


@pytest.mark.parametrize("anchor, fix", [(False, True), (True, True), (False, False)])
def test_mpas_step_rain_leaves_through_ps_and_dry_mass_is_fixed(anchor, fix):
    pytest.importorskip("scipy")
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel)
    from legoesm.core.conservation import dry_surface_pressure
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    mesh = create_voronoi_mesh(subdivision_level=1)
    sigma = create_sigma_coordinate(10, dtype=jnp.float64)
    st = baroclinic_wave_init_mpas(mesh, sigma, perturbed=False, moist=True)
    # rain to remove: seed q_r so the tendency has something to take
    st = st._replace(tracers={**st.tracers, "q_r": st.tracers["q_r"].replace(
        data=jnp.full_like(st.tracers["q_r"].data, 2e-3))})
    cfg = MPASPrimitiveEquationConfig(fix_mass=fix, anchor_mass_to_initial=anchor)
    model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
    dt = 60.0
    rate = 1e-6
    area = np.asarray(mesh.areaCell, dtype=np.float64)

    def budgets(s):
        dp = sigma.layer_thickness_dp(s.p_s.data)
        Q = sum(s.tracers[k].data for k in ("q_v", "q_c", "q_r"))
        water = np.asarray(jnp.sum(dp * Q, axis=-1)) / constants.g
        dry = np.asarray(dry_surface_pressure(s.p_s.data, s.tracers, sigma)) / constants.g
        # p_top is the coordinate's dry cap (sigma_top * p_s here)
        return (water * area).sum(), (dry * area).sum(), np.asarray(s.p_s.data)

    w0, d0, ps0 = budgets(st)
    s1 = model.step(st, dt, physics_fn=_rain_only_physics(rate))
    if not hasattr(s1, "p_s"):
        s1 = s1[0]
    w1, d1, ps1 = budgets(s1)
    # expected rain: rate*dt over the lowest three layers of each column
    dp0 = np.asarray(sigma.layer_thickness_dp(st.p_s.data))
    rain = (rate * dt * dp0[:, -3:].sum(-1) / constants.g * area).sum()
    assert rain > 1e-9 * w0, "vacuous: no rain"
    # global dry mass fixed (the fixer), global water down by the rain
    # (dynamics is a closed exchange within the step; the fixer's
    # uniform p_s shift is dry air, so water is untouched by it)
    print(f"fix={fix} anchor={anchor}: dry rel {abs(d1 - d0) / d0:.3e}, "
          f"water/rain {(w1 - w0) / -rain:.8f}")
    if fix:
        # 2e-16 MEASURED with the tracer re-weighted fixer; the unweighted
        # shift leaves 1.3e-11 and must FAIL here (codex r2)
        np.testing.assert_allclose(d1, d0, rtol=1e-13, atol=0)
    # the rain is computed on the PRE-step layer masses while the physics
    # saw the post-dynamics p_s; and the fixer's uniform p_s shift moves
    # water by Q_bar*delta (the legacy fixer did the same).  Both are
    # O(1e-4) here (MEASURED 1.6e-4 with the fixer); the identity that is
    # exact is at the helper level (test_budget_identities).
    np.testing.assert_allclose(w1 - w0, -rain, rtol=1e-3, atol=0)
    # the OLD fixer (total p_s) would have refilled the rain as dry air:
    total0 = (ps0 * area).sum(); total1 = (ps1 * area).sum()
    np.testing.assert_allclose(total1 - total0, -constants.g * rain, rtol=1e-3, atol=0)
