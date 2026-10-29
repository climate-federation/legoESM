"""CAM6 Zhang-McFarlane (host PBL height, mixed-phase rain-evaporation qsat,
convective transport of the carried MG2 tracers) vs the CAM Fortran oracle.

CAM passes the PBL height ``pblh`` (pbuf, written by the previous step's
CLUBB call, clubb_intr.F90:2616) to ``zm_convr`` (convect_deep.F90:266), whose
launch-level search stops at the level holding the PBL top
(zm_conv.F90:738-741).  The hydrostatic/MPAS bridge -- the lane that runs the
MPAS and FV3-duo column production decks -- now hands ZM the lagged
``PhysicsState.pblh`` carry that the turbulence scheme publishes.

Fixture ``oracle_fixtures/zm_pblh_cam.npz``: inputs and outputs of the real
CAM cam_cesm2_1_rel_60 (SHA a03b84b7c4e34f965b115686f22a043b85739e56)
zm_conv.F90, run by ``sbatch --wait scripts/validate/oracle/zm_pblh/build.sh``
(legoESM constants, CAM's Goff-Gratch qsat_water for the plume, ientropy tightened to 1e-11 K; see
make_fixture.py).  Column ``nocturnal_elevated`` (pblh 150 m) does not convect
in CAM; with the old 700 hPa launch bound it did.

``zm_conv_evap`` compares against CAM's ``qsat`` (zm_conv.F90:1484), run
unmodified in the oracle: the 1-K table (wv_saturation.F90:216-226, estblf
:306-321) of the Goff-Gratch ``svp_trans`` vapour-pressure blend to ice between
0 and -20 degC (wv_sat_methods.F90:326-361).  ``convtran`` transports CLDLIQ, CLDICE, NUMLIQ, NUMICE,
RAINQM, SNOWQM, NUMRAI (zm_conv_intr.F90:829-846, micro_mg_cam.F90:416-435) =
legoESM ``q_c, q_i, N_c, N_i, q_r, q_s, N_r``.

Tolerance: max relative error <= 1e-10 in fp64 (user decision 2026-10-19).
ZM has no smoothing fields of its own.  Its qsat table's [tmin, tmax] clamp
and q = 1 cap are smooth by default (legoesm.thermo, RULE 5); for 150-340 K
and e_sat/p <= 0.77 (the fixture's maximum) they depart from CAM by < 3e-11
relative, inside this tolerance (the cap departs by 1.1e-3 at e_sat = p).
"""
from __future__ import annotations

import os
import types

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.combined import PhysicsConfig  # noqa: E402
from legoesm.atmosphere.physics.convection.config import ConvectionConfig, ZhangMcFarlaneConfig  # noqa: E402
from legoesm.atmosphere.physics.convection.integration import make_convection_physics  # noqa: E402
from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (  # noqa: E402
    CONVTRAN1_TRACERS,
    zhang_mcfarlane_convection,
)
from legoesm.atmosphere.physics.physics_state import init_physics_state, update_physics_state  # noqa: E402
from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import HydrostaticState  # noqa: E402
from legoesm.grids.vertical import make_cam6_l32_levels  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "oracle_fixtures", "zm_pblh_cam.npz")
TOL = 1e-10
# Fixture-input rebuild: last-digit libm differences across machines (measured
# worst 2.6e-14 relative, q_v); a real generator change is far larger.
REBUILD_RTOL = 1e-13
# fixture key -> legoESM tracer name -> oracle convtran output (make_fixture.TRACERS)
TRACERS = (("qc", "q_c", "dqc_convtran"), ("qi", "q_i", "dqi_convtran"),
           ("nc", "N_c", "dnc_convtran"), ("ni", "N_i", "dni_convtran"),
           ("qr", "q_r", "dqr_convtran"), ("qs", "q_s", "dqs_convtran"),
           ("nr", "N_r", "dnr_convtran"))


@pytest.fixture(scope="module", autouse=True)
def _fp64_policy():
    """The fixture was generated under the fp64 policy (the cam_l32
    coordinate coefficients follow the policy dtype); restore afterwards."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIXTURE))


def _rel(ours, oracle):
    ours, oracle = np.asarray(ours, float), np.asarray(oracle, float)
    assert ours.shape == oracle.shape, (ours.shape, oracle.shape)
    scale = np.max(np.abs(oracle))
    assert scale > 0.0, "oracle field is identically zero: the comparison would be vacuous"
    return float(np.max(np.abs(ours - oracle)) / scale)


def _oracle_tendencies(fx):
    cp, g = constants.c_pd, constants.g
    return {
        "dT_dt": (fx["oracle_heat"] + fx["oracle_evap_tend_s"] + fx["oracle_seten"]) / cp,
        "dq_v_dt": fx["oracle_qtnd"] + fx["oracle_evap_tend_q"],
        "dq_c": fx["oracle_dlf"],
        "dq_r": fx["oracle_ntprprd"],
        "du": fx["oracle_dudt"],
        "dv": fx["oracle_dvdt"],
        "mass_flux_up": fx["oracle_mcon"] * 100.0 / g,     # hPa/s -> kg/m^2/s
        "icwmr": np.where(fx["oracle_ideep"][:, None], fx["oracle_ql"], 0.0),
        "snow": fx["oracle_snow"] * 1000.0,                # m/s -> kg/m^2/s
        "cape": fx["oracle_cape"],
        "mb": fx["oracle_mb"] * 100.0 / g,
    }


def test_fixture_exercises_the_pbl_launch_bound(fx):
    names = list(fx["names"])
    noc = names.index("nocturnal_elevated")
    assert not fx["oracle_ideep"][noc] and fx["oracle_cape"][noc] < 70.0
    assert fx["oracle_ideep"].sum() >= 5
    assert "a03b84b7c4e34f965b115686f22a043b85739e56" in str(fx["meta"])


def _wrapper(fx, tracers=True, config=None):
    return zhang_mcfarlane_convection(
        jnp.asarray(fx["T"]), jnp.asarray(fx["q"]), jnp.asarray(fx["p_full"]),
        jnp.asarray(fx["p_half"]), jnp.asarray(fx["u"]), jnp.asarray(fx["v"]),
        jnp.zeros_like(jnp.asarray(fx["T"])), float(fx["dt"]),
        ZhangMcFarlaneConfig() if config is None else config,
        land_frac=jnp.asarray(fx["land"]), cld_frac=jnp.asarray(fx["cld"]),
        pblh=jnp.asarray(fx["pblh"]), pref_edge=jnp.asarray(fx["pref_edge"]),
        tracers=({nm: jnp.asarray(fx[key]) for key, nm, _ in TRACERS} if tracers else None))


def test_wrapper_with_pblh_matches_cam_oracle(fx):
    out, prog = _wrapper(fx)
    o = _oracle_tendencies(fx)
    ours = {"dT_dt": out.dT_dt, "dq_v_dt": out.dq_v_dt, "dq_c": out.dq_c_conv_dt,
            "dq_r": out.dq_r_conv_dt, "du": out.du_dt_conv, "dv": out.dv_dt_conv,
            "mass_flux_up": out.mass_flux_up, "icwmr": out.icwmr,
            "snow": out.precip_frozen_sfc, "cape": out.cape, "mb": prog[:, -1]}
    ours.update({nm: out.tracer_transport[nm] for _, nm, _ in TRACERS})
    o.update({nm: fx[f"oracle_{key}"] for _, nm, key in TRACERS})
    np.testing.assert_array_equal(np.asarray(out.convective_mask).astype(bool), fx["oracle_ideep"])
    errs = {k: _rel(ours[k], o[k]) for k in o}
    assert max(errs.values()) <= TOL, errs


def test_convtran_covers_the_cam6_convtran1_set_and_conserves_columns(fx):
    """The bridge transports exactly the MG2 convtran1 constituents legoESM
    carries; each transport is zero outside deep columns and conserves its
    column (flux form, convtran's netflux round-off guard aside; checked at
    the oracle-match tolerance)."""
    assert CONVTRAN1_TRACERS == tuple(nm for _, nm, _ in TRACERS)
    out, _ = _wrapper(fx)
    dp = np.diff(fx["p_half"], axis=1)
    deep = fx["oracle_ideep"]
    for key, nm, _ in TRACERS:
        d = np.asarray(out.tracer_transport[nm])
        assert np.all(d[~deep] == 0.0), nm
        col = np.abs(np.sum(d * dp, axis=1)) / np.max(np.abs(d) * dp)
        assert np.max(col) <= TOL, (nm, np.max(col))
        assert np.max(np.abs(d[deep])) > 0.0, nm
    no_tr, _ = _wrapper(fx, tracers=False)
    assert no_tr.tracer_transport is None


def test_evaporation_saturation_is_the_cam_qsat_table():
    """thermo.saturation_vapor_pressure_trans = CAM svp_trans on Goff-Gratch:
    water curve at and above T_freeze, ice curve at or below T_freeze - 20 K,
    the vapour-pressure blend (T_freeze - T)/20 in between.  The CAM qsat table
    (saturation_specific_humidity_cam_table) reproduces svp_trans at the 1-K
    nodes tmin + k and interpolates linearly between them (checked here away
    from the table bounds, where the default smooth clamp is CAM's to
    rounding; the bounds themselves are pinned in test_saturation_curves)."""
    from legoesm.thermo import (saturation_specific_humidity_cam_table,
                                saturation_vapor_pressure_goff_gratch,
                                saturation_vapor_pressure_ice_goff_gratch,
                                saturation_vapor_pressure_trans)
    t0 = constants.T_freeze
    warm = jnp.asarray([t0, t0 + 5.0, 300.0])
    cold = jnp.asarray([t0 - 20.0, t0 - 35.0, 200.0])
    np.testing.assert_array_equal(saturation_vapor_pressure_trans(warm),
                                  saturation_vapor_pressure_goff_gratch(warm))
    np.testing.assert_array_equal(saturation_vapor_pressure_trans(cold),
                                  saturation_vapor_pressure_ice_goff_gratch(cold))
    t = jnp.asarray(t0 - 7.0)
    w = 7.0 / 20.0
    np.testing.assert_allclose(saturation_vapor_pressure_trans(t),
                               w * saturation_vapor_pressure_ice_goff_gratch(t)
                               + (1.0 - w) * saturation_vapor_pressure_goff_gratch(t), rtol=TOL)
    p = 6.0e4
    eps = constants.epsilon

    def q(e):
        return eps * e / (p - (1.0 - eps) * e)
    tmin = 127.16
    for k in (100, 140, 150):       # 227.16, 267.16, 277.16 K
        e0 = float(saturation_vapor_pressure_trans(jnp.asarray(tmin + k)))
        e1 = float(saturation_vapor_pressure_trans(jnp.asarray(tmin + k + 1)))
        assert float(saturation_specific_humidity_cam_table(
            jnp.asarray(tmin + k), p)) == pytest.approx(q(e0), rel=TOL, abs=0.0)
        assert float(saturation_specific_humidity_cam_table(
            jnp.asarray(tmin + k + 0.25), p)) == pytest.approx(q(0.75 * e0 + 0.25 * e1), rel=TOL, abs=0.0)
    e_lo = float(saturation_vapor_pressure_trans(jnp.asarray(tmin)))
    assert float(saturation_specific_humidity_cam_table(jnp.asarray(100.0), p)) == pytest.approx(
        q(e_lo), rel=TOL, abs=0.0)


def _bridge(fx, pblh, zm=None):
    """The hydrostatic/MPAS convection bridge on the fixture columns."""
    ncol, nlev = fx["T"].shape
    coord = make_cam6_l32_levels()
    np.testing.assert_allclose(np.asarray(coord.pressure_at_full(jnp.asarray(fx["p_s"]))),
                               fx["p_full"], rtol=0, atol=0)
    tracers = {"q_v": Field(jnp.asarray(fx["q"]))}
    tracers.update({nm: Field(jnp.asarray(fx[key])) for key, nm, _ in TRACERS})
    state = HydrostaticState(
        u=Field(jnp.asarray(fx["u"])), v=Field(jnp.asarray(fx["v"])),
        T=Field(jnp.asarray(fx["T"])), p_s=Field(jnp.asarray(fx["p_s"])),
        phis=Field(jnp.zeros((ncol,))), tracers=tracers)
    zm = ZhangMcFarlaneConfig() if zm is None else zm
    ps = update_physics_state(init_physics_state(ncol, nlev, PhysicsConfig()),
                              {"cloud_fraction": jnp.asarray(fx["cld"]), "pblh": pblh})
    fn = make_convection_physics(ConvectionConfig(scheme="zhang_mcfarlane", zhang_mcfarlane=zm),
                                 model_type="mpas", dt=float(fx["dt"]))
    grid = types.SimpleNamespace(land_frac=jnp.asarray(fx["land"]))
    return fn(state, grid, coord, phys_state=ps)


def test_mpas_bridge_passes_the_carried_pblh_and_matches_cam(fx):
    """Fails if the bridge drops PhysicsState.pblh: the nocturnal column then
    launches above its 150 m PBL and convects where CAM does not."""
    tend, carry = _bridge(fx, jnp.asarray(fx["pblh"]))
    o = _oracle_tendencies(fx)
    dp = np.diff(fx["p_half"], axis=1)
    ours = {"dT_dt": tend.dT_dt.data, "dq_v_dt": tend.tracer_tendencies["q_v"].data,
            "dq_c": tend.tracer_tendencies["q_c"].data, "du": tend.du_dt.data,
            "dv": tend.dv_dt.data, "mass_flux_up": carry["conv_mass_flux_up"],
            "precip": tend.precip.data}
    o_cmp = {k: o[k] for k in ours if k != "precip"}
    o_cmp["precip"] = np.maximum(np.sum(o["dq_r"] * dp, axis=1), 0.0) / constants.g
    # every carried convtran1 tracer: its transport, plus the detrained
    # condensate dlf for q_c (CAM books dlf as cloud water in clubb_intr)
    o_cmp["dq_c"] = o["dq_c"] + fx["oracle_dqc_convtran"]
    for _, nm, key in TRACERS[1:]:
        ours[nm] = tend.tracer_tendencies[nm].data
        o_cmp[nm] = fx[f"oracle_{key}"]
    errs = {k: _rel(ours[k], o_cmp[k]) for k in ours}
    assert max(errs.values()) <= TOL, errs


def test_transport_switch_off_drops_exactly_the_tracer_transport(fx):
    """ZhangMcFarlaneConfig.transport_cloud_tracers=False: no convtran (the
    scheme returns tracer_transport=None even when handed tracers), q_c gets
    the detrained condensate alone, the other convtran1 tracers get no
    convective tendency, and T/q_v/winds/mass flux are bit-identical to on."""
    off_cfg = ZhangMcFarlaneConfig(transport_cloud_tracers=False)
    assert ZhangMcFarlaneConfig().transport_cloud_tracers is True
    out_off, _ = _wrapper(fx, config=off_cfg)
    assert out_off.tracer_transport is None
    pblh = jnp.asarray(fx["pblh"])
    on, c_on = _bridge(fx, pblh)
    off, c_off = _bridge(fx, pblh, zm=off_cfg)
    for a, b in ((on.dT_dt, off.dT_dt), (on.du_dt, off.du_dt), (on.dv_dt, off.dv_dt),
                 (on.tracer_tendencies["q_v"], off.tracer_tendencies["q_v"])):
        np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))
    np.testing.assert_array_equal(np.asarray(c_on["conv_mass_flux_up"]),
                                  np.asarray(c_off["conv_mass_flux_up"]))
    qc_off = np.asarray(off.tracer_tendencies["q_c"].data)
    np.testing.assert_allclose(qc_off, np.asarray(out_off.dq_c_conv_dt).reshape(qc_off.shape),
                               rtol=TOL, atol=0)
    assert np.any(np.asarray(on.tracer_tendencies["q_c"].data) != qc_off)
    for _, nm, _ in TRACERS[1:]:
        t = off.tracer_tendencies.get(nm)
        assert t is None or not np.any(np.asarray(t.data)), nm


def test_zm_gradient_is_finite_with_carried_pblh(fx):
    def loss(T):
        tend, _ = _bridge(dict(fx, T=T), jnp.asarray(fx["pblh"]))
        return jnp.sum(tend.dT_dt.data ** 2)
    grad = jax.grad(loss)(jnp.asarray(fx["T"]))
    assert np.all(np.isfinite(np.asarray(grad))) and float(jnp.max(jnp.abs(grad))) > 0.0

    def loss_tr(T, qi):
        tend, _ = _bridge(dict(fx, T=T, qi=qi), jnp.asarray(fx["pblh"]))
        return jnp.sum(tend.tracer_tendencies["q_i"].data ** 2) + jnp.sum(
            tend.tracer_tendencies["q_v"].data ** 2)
    g_t, g_qi = jax.grad(loss_tr, argnums=(0, 1))(jnp.asarray(fx["T"]), jnp.asarray(fx["qi"]))
    for g in (g_t, g_qi):
        assert np.all(np.isfinite(np.asarray(g))) and float(jnp.max(jnp.abs(g))) > 0.0
    # an ice-free column (q_i exactly 0, as above the freezing level in a run)
    g0 = jax.grad(loss_tr, argnums=1)(jnp.asarray(fx["T"]), jnp.zeros_like(jnp.asarray(fx["qi"])))
    assert np.all(np.isfinite(np.asarray(g0)))


def test_physics_state_seeds_pblh_at_zero_and_carries_updates():
    """CAM cold start: PBLH initialized to 0 when the initial file has none
    (physpkg.F90:419-424); the turbulence update then persists."""
    ps = init_physics_state(5, 4, PhysicsConfig())
    np.testing.assert_array_equal(np.asarray(ps.pblh), np.zeros(5))
    new = jnp.arange(5.0)
    ps2 = update_physics_state(ps, {"pblh": new})
    np.testing.assert_array_equal(np.asarray(ps2.pblh), np.asarray(new))
    np.testing.assert_array_equal(np.asarray(update_physics_state(ps2, {}).pblh), np.asarray(new))


def test_turbulence_publishes_pblh_into_the_physics_state():
    """The carry ZM reads: every turbulence call publishes its diagnosed PBL
    height under ``pblh`` (CAM clubb_intr.F90:2616 writes pbuf pblh)."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.integration import make_turbulence_physics
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate

    n, nlev = 4, 10
    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    dims = ("face", "x", "y", "level")
    state = state._replace(
        u=Field(jnp.full((6, n, n, nlev), 10.0), dims=dims),
        v=Field(jnp.full((6, n, n, nlev), 3.0), dims=dims),
        tracers={"q_v": Field(jnp.full((6, n, n, nlev), 1e-3), dims=dims)})
    fn = make_turbulence_physics(TurbulenceConfig(scheme="louis"), model_type="hydrostatic", dt=300.0)
    _, carry = fn(state, grid, sigma)
    assert isinstance(carry, dict) and "pblh" in carry
    pblh = np.asarray(carry["pblh"])
    assert pblh.shape == (6 * n * n,) and np.all(np.isfinite(pblh)) and np.all(pblh > 0.0)


def test_fixture_inputs_are_what_the_generator_builds(fx, tmp_path):
    """Direct test of scripts/validate/oracle/zm_pblh/make_fixture.py: its
    Python side rebuilds the committed inputs to REBUILD_RTOL, and its oracle-output
    parser reads the driver's record format."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "validate",
                        "oracle", "zm_pblh", "make_fixture.py")
    spec = importlib.util.spec_from_file_location("zm_pblh_make_fixture", path)
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    c = mf.build_columns()
    for k in ("T", "q", "p_s", "p_full", "p_half", "u", "v", "cld", "land", "pblh", "pref_edge",
              "qc", "qi", "nc", "ni", "qr", "qs", "nr"):
        np.testing.assert_allclose(c[k], fx[k], rtol=REBUILD_RTOL, atol=0.0, err_msg=k)
    p = tmp_path / "o.txt"
    p.write_text("DIMS 2 3 1\nLIMCNV 6\nF1 cape 2\n1.5\n0.0\nI1 ideep 2\n2\n0\n"
                 "F2 heat 2 3\n1\n2\n3\n4\n5\n6\n")
    o = mf.read_oracle(str(p))
    assert (o["ncol"], o["nlev"], o["lengath"], o["limcnv"]) == (2, 3, 1, 6)
    np.testing.assert_array_equal(o["heat"], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(o["ideep"], [2, 0])


# ---------------------------------------------------------------------------
# Oracle rows 49-50: every ZM bridge hands ZM the carried pblh; none without
# ---------------------------------------------------------------------------

class _Seen(Exception):
    """Raised by the ZM spy once it has recorded the bridge's arguments."""


def _spy_zm(monkeypatch):
    from legoesm.atmosphere.physics.convection import integration as CI
    seen = {}

    def spy(*a, **k):
        seen.update(k)
        raise _Seen
    monkeypatch.setattr(CI, "zhang_mcfarlane_convection", spy)
    return seen


_ZM_AQUA = ConvectionConfig(scheme="zhang_mcfarlane",
                            zhang_mcfarlane=ZhangMcFarlaneConfig(land_fraction="none"))


def test_nonhydrostatic_bridge_passes_the_carried_pblh(monkeypatch):
    from legoesm.core.state import NonHydrostaticState
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import compute_terrain_metric, create_height_coordinate

    seen = _spy_zm(monkeypatch)
    n, nlev = 2, 6
    grid = create_cubed_sphere(n)
    hc = create_height_coordinate(nlev, 30000.0)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    d3, z3 = ("face", "x", "y", "level"), jnp.zeros((6, n, n, nlev))
    state = NonHydrostaticState(
        u=Field(z3, dims=d3), v=Field(z3, dims=d3),
        w=Field(jnp.zeros((6, n, n, nlev + 1)), dims=("face", "x", "y", "level_half")),
        theta_prime=Field(z3, dims=d3), rho_prime=Field(z3, dims=d3),
        phis=Field(jnp.zeros((6, n, n)), dims=("face", "x", "y")),
        tracers=Field(jnp.full((6, n, n, nlev, 1), 1e-3),
                      dims=("face", "x", "y", "level", "tracer")))
    ncol = 6 * n * n
    ps = update_physics_state(init_physics_state(ncol, nlev, PhysicsConfig()),
                              {"pblh": jnp.arange(1.0, ncol + 1.0)})
    fn = make_convection_physics(_ZM_AQUA, model_type="nonhydrostatic", dt=600.0)
    with pytest.raises(_Seen):
        fn(state, grid, hc, tm, phys_state=ps)
    assert seen.get("pblh") is not None, "the nonhydrostatic bridge dropped pblh"
    np.testing.assert_array_equal(np.asarray(seen["pblh"]), np.asarray(ps.pblh))


def test_spectral_bridge_passes_the_carried_pblh(monkeypatch):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import isothermal_rest_state_spectral
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    seen = _spy_zm(monkeypatch)
    grid = create_gaussian_grid(n_max=21)
    sigma = create_sigma_coordinate(6)
    state = isothermal_rest_state_spectral(grid, sigma, perturbation_amplitude=0.0)
    z = jnp.zeros((grid.n_lat, grid.n_lon, sigma.n_levels))
    state = state._replace(tracers={
        "q_v": Field(z + 1e-3, dims=("lat", "lon", "level")),
        "q_c": Field(z, dims=("lat", "lon", "level"))})
    ncol = grid.n_lat * grid.n_lon
    ps = update_physics_state(init_physics_state(ncol, sigma.n_levels, PhysicsConfig()),
                              {"pblh": jnp.arange(1.0, ncol + 1.0)})
    fn = make_convection_physics(_ZM_AQUA, model_type="spectral_pe", dt=600.0)
    with pytest.raises(_Seen):
        fn(state, grid, sigma, phys_state=ps)
    assert seen.get("pblh") is not None, "the spectral bridge dropped pblh"
    np.testing.assert_array_equal(np.asarray(seen["pblh"]), np.asarray(ps.pblh))


def test_zm_without_pblh_is_refused(fx):
    """Row 50: no carry-less launch bound; CAM always hands ZM a pblh."""
    with pytest.raises(ValueError, match="no PBL height"):
        zhang_mcfarlane_convection(
            jnp.asarray(fx["T"]), jnp.asarray(fx["q"]), jnp.asarray(fx["p_full"]),
            jnp.asarray(fx["p_half"]), jnp.asarray(fx["u"]), jnp.asarray(fx["v"]),
            jnp.zeros_like(jnp.asarray(fx["T"])), float(fx["dt"]), ZhangMcFarlaneConfig(),
            land_frac=jnp.asarray(fx["land"]), pref_edge=jnp.asarray(fx["pref_edge"]))
    with pytest.raises(ValueError, match="no PBL height"):
        _bridge(fx, None)
