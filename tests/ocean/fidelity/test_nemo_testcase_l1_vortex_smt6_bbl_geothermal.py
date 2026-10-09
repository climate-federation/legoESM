"""SMT-6 / SMT-6b (Decision 110): NEMO tra_bbc and tra_bbl on the seamount.

trabbc.f90:158-159, :226   Krhs(mbkt) += r1_rho0_rcp*rn_geoflx_cst
                           / (e3t_0(mbkt)*(1+r3t(Kmm)))
eosbn2.f90:1344, :1347     S-EOS alpha/beta (rab_2d, np_seos)
trabbl.f90:427-431         diffusive gate SIGN(0.5, -zgdrho*mgrhu)
trabbl.f90:258-263         bottom-cell diffusive flux
stprk3_stg.f90:526-529     stage 3: tra_ldf, tra_bbc, tra_bbl, tra_dmp
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from legoesm.ocean.eos import NemoSEOSConfig
from legoesm.ocean.physics.bbl_adv import (
    apply_bbl_diffusive_tendency,
    nemo_bbl_diffusive_coefficients,
    nemo_bbl_diffusive_geometry,
)
from legoesm.ocean.physics.geothermal import nemo_tra_bbc_rate

RHO0, RCP = 1026.0, 3991.86795711963     # SMT-5 ocean.output phycst block
_GRID = SimpleNamespace(fold=None)


def _slope(T_shallow, T_deep, S_shallow=35.0, S_deep=35.0):
    """Two wet columns (1 and 2 levels) and a land column, one U face open."""
    h = np.array([[[500.0, 0.0], [500.0, 300.0], [0.0, 0.0]]])
    wet = (h > 0).any(-1).astype(float)
    gdept = np.broadcast_to(np.array([250.0, 750.0]), h.shape)
    e3 = np.full(h.shape, 500.0)
    one = np.ones(wet.shape)
    umask = np.array([[1.0, 0.0, 0.0]])
    geom = nemo_bbl_diffusive_geometry(
        h, wet, gdept, e3, e3, one * 3.0e4, one * 3.0e4, one * 3.0e4,
        one * 3.0e4, umask, np.zeros_like(umask), aht_m2_s=1000.0,
        grid=_GRID)
    T = np.zeros(h.shape)
    S = np.zeros(h.shape)
    T[0, 0, 0], T[0, 1, 1] = T_shallow, T_deep
    S[0, 0, 0], S[0, 1, 1] = S_shallow, S_deep
    return h, geom, T, S


def _nemo_gate_open(cfg, Ts, Td, Ss, Sd, depth_s, depth_d):
    """trabbl.f90:427-431 with eosbn2.f90 np_seos alpha/beta, in numpy."""
    def ab(T, S, z):
        zt, zs = T - cfg.T0, S - cfg.S0
        r1 = 1.0 / cfg.rho0
        a = (cfg.a0 * (1.0 + cfg.lambda1 * zt + cfg.mu1 * z) + cfg.nu * zs) * r1
        b = (cfg.b0 * (1.0 - cfg.lambda2 * zs - cfg.mu2 * z) - cfg.nu * zt) * r1
        return a, b
    a0, b0 = ab(Ts, Ss, depth_s)
    a1, b1 = ab(Td, Sd, depth_d)
    zgdrho = ((a1 + a0) * (Td - Ts) - (b1 + b0) * (Sd - Ss)) * 1.0
    arg = -zgdrho * 1.0                       # mgrhu = +1: east is deeper
    return not (arg >= 0.0)                   # SIGN(0.5, arg) = -0.5 opens


_DECK = NemoSEOSConfig(rho0=RHO0, a0=0.28, b0=0.0, lambda1=0.0, lambda2=0.0,
                       mu1=0.0, mu2=0.0, nu=0.0)
_FULL = NemoSEOSConfig(rho0=RHO0)            # nonzero b0, lambda, mu, nu


@pytest.mark.parametrize("cfg", [_DECK, _FULL], ids=["deck", "full_seos"])
@pytest.mark.parametrize("dT,dS", [(-1.0, 0.0), (1.0, 0.0), (0.0, 0.0),
                                   (0.0, 0.5), (0.0, -0.5), (0.3, 0.2)])
def test_seos_gate_opens_exactly_when_nemos_inequality_holds(cfg, dT, dS):
    Ts, Ss = 5.0, 35.0
    h, geom, T, S = _slope(Ts, Ts + dT, Ss, Ss + dS)
    depth = np.asarray(geom.dep_bot_ref)
    ahu, ahv = nemo_bbl_diffusive_coefficients(
        T, S, geom, bottom_depth_m=depth, rho_0=RHO0, grid=_GRID,
        eos_form="nemo_seos", seos_cfg=cfg)
    want = _nemo_gate_open(cfg, Ts, Ts + dT, Ss, Ss + dS, depth[0, 0],
                           depth[0, 1])
    assert bool(np.asarray(ahu)[0, 0] > 0.0) == want
    assert np.all(np.asarray(ahu)[0, 1:] == 0.0)
    assert np.all(np.asarray(ahv) == 0.0)


def test_deck_gate_is_a_bottom_temperature_sign_test():
    # warmer deep side (lighter) -> open; colder deep side or equal -> closed
    for dT, want in ((0.25, True), (-0.25, False), (0.0, False)):
        _, geom, T, S = _slope(5.0, 5.0 + dT)
        ahu, _ = nemo_bbl_diffusive_coefficients(
            T, S, geom, bottom_depth_m=np.asarray(geom.dep_bot_ref),
            rho_0=RHO0, grid=_GRID, eos_form="nemo_seos", seos_cfg=_DECK)
        assert bool(np.asarray(ahu)[0, 0] > 0.0) == want


def test_seos_gate_refuses_a_missing_deck_eos():
    _, geom, T, S = _slope(5.0, 6.0)
    with pytest.raises(ValueError, match="NemoSEOSConfig"):
        nemo_bbl_diffusive_coefficients(
            T, S, geom, bottom_depth_m=np.asarray(geom.dep_bot_ref),
            rho_0=RHO0, grid=_GRID, eos_form="nemo_seos")


def test_diffusive_flux_is_trabbl_258_263_bit_for_bit():
    h, geom, T, S = _slope(4.0, 4.75, 35.0, 35.2)
    ahu, ahv = nemo_bbl_diffusive_coefficients(
        T, S, geom, bottom_depth_m=np.asarray(geom.dep_bot_ref), rho_0=RHO0,
        grid=_GRID, eos_form="nemo_seos", seos_cfg=_DECK)
    ahu = np.asarray(ahu)
    assert ahu[0, 0] > 0.0
    stretch = np.array([[1.0003, 0.9998, 1.0]])
    h_kmm = h * stretch[..., None]
    area = np.full((1, 3), 9.0e8)
    zero = np.zeros_like(T)
    dT, dS = apply_bbl_diffusive_tendency(
        zero, zero, T, S, h_kmm, area, geom, ahu, np.asarray(ahv), grid=_GRID)
    for got, tr in ((np.asarray(dT), T), (np.asarray(dS), S)):
        zptb = np.array([tr[0, 0, 0], tr[0, 1, 1]])
        r1 = 1.0 / area[0, 0]
        # ji = 1: east neighbour ji+1 = 2 (the deep column), west is a wall
        want0 = ((ahu[0, 0] * (zptb[1] - zptb[0]) - 0.0 * (zptb[0] - 0.0))
                 + (0.0 * (0.0 - zptb[0]) - 0.0 * (zptb[0] - 0.0))) \
            * r1 / (h[0, 0, 0] * stretch[0, 0])
        want1 = ((0.0 * (0.0 - zptb[1]) - ahu[0, 0] * (zptb[1] - zptb[0]))
                 + (0.0 * (0.0 - zptb[1]) - 0.0 * (zptb[1] - 0.0))) \
            * r1 / (h[0, 1, 1] * stretch[0, 1])
        assert got[0, 0, 0] == want0 and got[0, 1, 1] == want1
        assert want0 != 0.0
        assert np.count_nonzero(got) == 2


def test_geothermal_rate_is_trabbc_158_159_bit_for_bit():
    e3t = np.array([[[500.0, 250.0, 0.0], [500.0, 500.0, 120.5]]])
    wet = (e3t > 0).astype(float)
    stretch = np.array([[1.0 + 3.7e-5, 1.0 - 1.1e-5]])
    got = np.asarray(nemo_tra_bbc_rate(86.4e-3, e3t, stretch, wet,
                                       rho0=RHO0, rcp=RCP))
    rho0_rcp = RHO0 * RCP                       # eosbn2.f90:2379
    qgh_trd0 = (1.0 / rho0_rcp) * 86.4e-3       # eosbn2.f90:2382, trabbc:226
    want = np.zeros_like(e3t)
    want[0, 1, 2] = qgh_trd0 / (e3t[0, 1, 2] * stretch[0, 1])
    want[0, 0, 1] = qgh_trd0 / (e3t[0, 0, 1] * stretch[0, 0])
    np.testing.assert_array_equal(got, want)


# --------------------------------------------------------------------------
# Cards and production placement.  NEMO-like inputs: the target T is the
# card's own initial T (NEMO dumps it after the anomaly), S 35, resto 1/86400.

def _write_inputs(T, active, root):
    import netCDF4  # noqa: N813

    nj, ni, nk = active.shape
    for name, var, field in (
            ("data_1m_potential_temperature_nomask.nc", "votemper", T),
            ("data_1m_salinity_nomask.nc", "vosaline", 35.0 * active)):
        zyx = np.zeros((nk + 1, nj, ni))
        zyx[:nk] = np.moveaxis(field * active, -1, 0)
        with netCDF4.Dataset(root / name, "w") as ds:
            for dim, size in (("x", ni), ("y", nj), ("z", nk + 1)):
                ds.createDimension(dim, size)
            ds.createDimension("time_counter", None)
            v = ds.createVariable(var, "f8", ("time_counter", "z", "y", "x"))
            v[:] = np.repeat(zyx[None], 12, axis=0)
    resto = np.zeros((nk + 1, nj, ni))
    resto[:nk] = np.moveaxis(active, -1, 0) * (1.0 / 86400.0)
    with netCDF4.Dataset(root / "resto.nc", "w") as ds:
        for dim, size in (("x", ni), ("y", nj), ("z", nk + 1)):
            ds.createDimension(dim, size)
        ds.createVariable("resto", "f8", ("z", "y", "x"))[:] = resto


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
        vortex_smt6b_cold_flank,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card4 = build_nemo_testcase_card("VORTEX_SMT4_VEC-zps")
    active = np.asarray(card4.recipe.z_coord.is_active, dtype=np.float64)
    T0 = np.asarray(card4.recipe.initial_state.T.data)
    out = {"smt4": card4}
    for case, T in (("VORTEX_SMT5_VEC-zps", T0), ("VORTEX_SMT6_VEC-zps", T0),
                    ("VORTEX_SMT6B_VEC-zps",
                     vortex_smt6b_cold_flank(T0, active))):
        root = tmp_path_factory.mktemp(case.replace("-", "_"))
        _write_inputs(T, active, root)
        out[case] = build_nemo_testcase_card(case, deck_root=root)
    return out


def test_smt6_is_smt5_plus_bbl_and_geothermal_and_6b_only_the_anomaly(cards):
    c5, c6 = cards["VORTEX_SMT5_VEC-zps"], cards["VORTEX_SMT6_VEC-zps"]
    c6b = cards["VORTEX_SMT6B_VEC-zps"]
    cfg5, cfg6 = c5.recipe.model_config, c6.recipe.model_config
    assert cfg5.bbl_diffusive_option == 0 and cfg5.nemo_geothermal_qgh_wm2 is None
    assert (cfg6.bbl_diffusive_option, cfg6.bbl_aht_m2_s, cfg6.bbl_adv_option,
            cfg6.nemo_geothermal_qgh_wm2) == (1, 1000.0, 0, 86.4e-3)
    assert (cfg6.rho_0, cfg6.physics.constants.c_sw) == (RHO0, RCP)
    assert cfg6._replace(bbl_diffusive_option=0, bbl_aht_m2_s=0.0,
                         nemo_geothermal_qgh_wm2=None,
                         nemo_tracer_damping=None) == cfg5._replace(
        nemo_tracer_damping=None)
    assert (c6.bbl_diffusive_option, c6.bbl_aht_m2_s) == (1, 1000.0)
    T6 = np.asarray(c6.recipe.initial_state.T.data)
    T6b = np.asarray(c6b.recipe.initial_state.T.data)
    np.testing.assert_array_equal(T6, np.asarray(c5.recipe.initial_state.T.data))
    moved = T6b != T6
    zc = c6.recipe.z_coord
    bottom = np.asarray(zc.bottom_level)
    k = np.arange(T6.shape[-1])
    wetcol = np.asarray(zc.is_active).any(-1)
    is_bottom = (k[None, None, :] == bottom[..., None]) & wetcol[..., None]
    assert not np.any(moved & ~is_bottom)
    assert sorted(np.unique(np.round(T6 - T6b, 12)[moved])) == [1.7, 3.4]


def _open_faces(card, T):
    from legoesm.ocean.physics.bbl_adv import nemo_bbl_diffusive_geometry

    zc = card.recipe.z_coord
    cfg = card.recipe.model_config
    raw = zc.nemo_een_barotropic
    geom = nemo_bbl_diffusive_geometry(
        zc.h_partial, card.recipe.initial_state.land_mask.data,
        zc.nemo_gdept_0, zc.nemo_bbl_e3u_0, zc.nemo_bbl_e3v_0,
        raw.e1u, raw.e2u, raw.e1v, raw.e2v, raw.umask, raw.vmask,
        aht_m2_s=cfg.bbl_aht_m2_s, grid=card.recipe.grid)
    ahu, ahv = nemo_bbl_diffusive_coefficients(
        T, card.recipe.initial_state.S.data, geom,
        bottom_depth_m=geom.dep_bot_ref, rho_0=cfg.rho_0,
        grid=card.recipe.grid, eos_form=cfg.eos, seos_cfg=cfg.eos_nemo_seos)
    return (int(np.count_nonzero(np.asarray(ahu))),
            int(np.count_nonzero(np.asarray(ahv))),
            int(np.count_nonzero(np.asarray(geom.mgrhu)
                                 * np.asarray(geom.u_active))),
            int(np.count_nonzero(np.asarray(geom.mgrhv)
                                 * np.asarray(geom.v_active))))


def test_the_gate_is_closed_on_smt6_and_1p7k_is_the_smallest_tenth_opening(
        cards):
    from legoesm.ocean.fidelity import nemo_testcase_recipe as rec

    c6 = cards["VORTEX_SMT6_VEC-zps"]
    T0 = np.asarray(c6.recipe.initial_state.T.data)
    active = np.asarray(c6.recipe.z_coord.is_active, dtype=np.float64)
    assert _open_faces(c6, T0) == (0, 0, 24, 24)
    assert _open_faces(c6, np.asarray(
        cards["VORTEX_SMT6B_VEC-zps"].recipe.initial_state.T.data)) == (
            24, 24, 24, 24)
    saved = rec._SMT6B_COLD_FLANK_K
    try:
        rec._SMT6B_COLD_FLANK_K = 1.6
        assert _open_faces(c6, rec.vortex_smt6b_cold_flank(T0, active))[:2] \
            == (0, 0)
    finally:
        rec._SMT6B_COLD_FLANK_K = saved


def test_validator_keeps_geothermal_on_the_smt6_cards_alone(cards):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        validate_nemo_testcase_card,
    )

    c5, c6 = cards["VORTEX_SMT5_VEC-zps"], cards["VORTEX_SMT6_VEC-zps"]
    leaked = c5._replace(recipe=c5.recipe._replace(
        model_config=c5.recipe.model_config._replace(
            nemo_geothermal_qgh_wm2=86.4e-3)))
    with pytest.raises(ValueError, match="geothermal"):
        validate_nemo_testcase_card(leaked)
    for q in (None, 0.0864001):
        bad = c6._replace(recipe=c6.recipe._replace(
            model_config=c6.recipe.model_config._replace(
                nemo_geothermal_qgh_wm2=q)))
        with pytest.raises(ValueError, match="geothermal"):
            validate_nemo_testcase_card(bad)


def test_card_refuses_a_target_that_is_not_its_initial_state(
        cards, tmp_path):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    c4 = cards["smt4"]
    active = np.asarray(c4.recipe.z_coord.is_active, dtype=np.float64)
    # SMT-6b against the un-perturbed dump: NEMO's istate and the card differ
    _write_inputs(np.asarray(c4.recipe.initial_state.T.data), active, tmp_path)
    with pytest.raises(ValueError, match="usr_def_istate identity"):
        build_nemo_testcase_card("VORTEX_SMT6B_VEC-zps", deck_root=tmp_path)


def _step(card, **hooks):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hooks))
    return model.step(card.recipe.initial_state, dt=card.dt_s, t_seconds=0.0)


def test_stage3_rhs_carries_tra_bbc_and_a_stage1_plant_fires(cards):
    c6 = cards["VORTEX_SMT6_VEC-zps"]
    zc = c6.recipe.z_coord
    bottom = np.asarray(zc.bottom_level)
    wetcol = np.asarray(zc.is_active).any(-1)
    k = np.arange(np.asarray(zc.h_partial).shape[-1])
    is_bottom = (k[None, None, :] == bottom[..., None]) & wetcol[..., None]

    def check(trace):
        geo, before, consumed = (np.asarray(x) for x in trace[2])
        np.testing.assert_array_equal(consumed, before + geo)
        assert np.all(geo[is_bottom] > 0.0) and np.all(geo[~is_bottom] == 0.0)
        # |r3t| < 1e-3 here (1.8e-4 measured): the divisor is e3t_0*(1+r3t)
        e3t = np.asarray(zc.nemo_e3t_0)[is_bottom]
        np.testing.assert_allclose(
            geo[is_bottom], 86.4e-3 / (RHO0 * RCP) / e3t, rtol=1e-3)

    check(_step(c6, expose_stage3_tracer_damping=True))
    with pytest.raises(AssertionError):
        check(_step(c6, expose_stage3_tracer_damping=True,
                    geothermal_stage1_plant=True))
    # stage 1 never sees the production heating; the plant moves it
    s1 = lambda card, plant: np.asarray(_step(  # noqa: E731
        card, expose_tracer_stage=1, geothermal_stage1_plant=plant).T.data)
    c5 = cards["VORTEX_SMT5_VEC-zps"]
    c6_nobbl = c6._replace(recipe=c6.recipe._replace(
        model_config=c6.recipe.model_config._replace(
            bbl_diffusive_option=0, bbl_aht_m2_s=0.0)))
    np.testing.assert_array_equal(s1(c6_nobbl, False), s1(c5, False))
    assert not np.array_equal(s1(c6_nobbl, True), s1(c5, False))


def test_smt6b_production_bbl_moves_only_bottom_cells_of_open_faces(cards):
    """BBL on vs off on the SMT-6b card: the step result differs, and on
    SMT-6 (gate closed) BBL on vs off is bit-identical."""
    def T_after(card, bbl):
        cfg = card.recipe.model_config
        if not bbl:
            cfg = cfg._replace(bbl_diffusive_option=0, bbl_aht_m2_s=0.0)
        c = card._replace(recipe=card.recipe._replace(model_config=cfg))
        return np.asarray(_step(c).T.data)

    c6, c6b = cards["VORTEX_SMT6_VEC-zps"], cards["VORTEX_SMT6B_VEC-zps"]
    np.testing.assert_array_equal(T_after(c6, True), T_after(c6, False))
    assert not np.array_equal(T_after(c6b, True), T_after(c6b, False))
