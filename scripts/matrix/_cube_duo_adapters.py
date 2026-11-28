"""FV3 duo adapters for the cubed-sphere rows of the atmosphere matrix.

Thin glue between ``run_atmosphere_test_matrix.py`` and the certified FV3
duo lanes.  NO numerics live here: every arithmetic step runs inside an
existing duo kernel; this module only builds contexts/initial states, fixes
the step cadence, and samples six-face A-grid diagnostics.

* Shallow water (W2/W5/W6/modons): the JAX km=1 duo stepper
  (``legoesm.core.fv3_duo_stepper``) with the Zenodo deck cadences the
  ``scripts/validate/fv3_native/run_duo_stepper_*`` runners use.
* Cosine bell (FV3 ``test_case = 1`` analogue): frozen winds, the delp
  transport half of the duo acoustic step (``d_sw1`` -> barrier 1 ->
  ``d_sw2``) WITHOUT ``p_grad_c`` and without the D-wind update -- the
  ``#ifdef SW_DYNAMICS ... test_case > 1`` gate of ``dyn_core.F90:649``.
  The C-grid winds are the duo ``c_sw`` D->C interpolation of the
  analytic solid-body D winds at ``dt2 = 0`` (no half-step advance),
  computed once.
* Hydrostatic 3-D (baroclinic / Held-Suarez): ``FV3DuoDynamicsModel`` on
  ``create_fv3_duo_grid`` with the DCMIP16 baroclinic-wave IC and, for
  Held-Suarez, the driver lane's certified face-stacked HS twin.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# shared geometry / sampling helpers
# ---------------------------------------------------------------------------


def _compute_window(ctx_np) -> slice:
    ng, n = ctx_np["ng"], ctx_np["n"]
    return slice(ng, ng + n)


def duo_cell_geometry(ctx_np):
    """``(lon_deg, lat_deg, area)`` of the six-face compute cells, (6, n, n)."""
    sl = _compute_window(ctx_np)
    gs6 = ctx_np["gs6"]
    lon = np.stack([np.asarray(g["agrid_lon"])[sl, sl] for g in gs6])
    lat = np.stack([np.asarray(g["agrid_lat"])[sl, sl] for g in gs6])
    area = np.stack([np.asarray(g["area"])[sl, sl] for g in gs6])
    return np.degrees(lon), np.degrees(lat), area


class _GeoWinds:
    """D-grid covariant winds -> geographic (east, north) at A-grid centres
    via the certified upstream ``c2l_ord2`` lens (the same lens every
    ``run_duo_stepper_*`` runner samples with)."""

    def __init__(self, ctx_np):
        from legoesm.grids.fv3_native_ext_vector import center_a_matrix
        self.n, self.ng = ctx_np["n"], ctx_np["ng"]
        self.sl = _compute_window(ctx_np)
        self.gs6 = ctx_np["gs6"]
        self.amat6 = [center_a_matrix(g) for g in self.gs6]

    def __call__(self, u6, v6):
        """``u6`` (6, m_a, m_b[, km]), ``v6`` (6, m_b, m_a[, km]) ->
        ``(u_east, v_north)`` (6, n, n[, km])."""
        from legoesm.grids.fv3_native_ext_vector import c2l_ord2_face
        u6 = np.asarray(u6, dtype=np.float64)
        v6 = np.asarray(v6, dtype=np.float64)
        sl = self.sl
        three_d = u6.ndim == 4
        km = u6.shape[-1] if three_d else 1
        ue = np.empty((6, self.n, self.n, km))
        vn = np.empty_like(ue)
        for t in range(6):
            g = self.gs6[t]
            for k in range(km):
                uk = u6[t][..., k] if three_d else u6[t]
                vk = v6[t][..., k] if three_d else v6[t]
                ua, va = c2l_ord2_face(uk, vk, g["dx"], g["dy"],
                                       self.amat6[t], self.n, self.ng)
                ue[t, :, :, k] = ua[sl, sl]
                vn[t, :, :, k] = va[sl, sl]
        if not three_d:
            return ue[..., 0], vn[..., 0]
        return ue, vn


# ---------------------------------------------------------------------------
# shallow water
# ---------------------------------------------------------------------------

#: Zenodo duo deck cadences (resolved namelist echoes, C48 decks), as
#: carried by ``run_duo_stepper_case6.CASE_DECKS`` / ``run_duo_stepper_w5``.
#: case 5 has no Zenodo run (``run_duo_stepper_w5`` docstring); it runs the
#: case-2/6 hord deck family at the case-6 cadence, like that runner.
_SW_DECKS = {
    2: {"dt_atmos": 3600.0, "n_split": 7, "d4_bg": 0.12},
    5: {"dt_atmos": 1200.0, "n_split": 7, "d4_bg": 0.12},
    6: {"dt_atmos": 1200.0, "n_split": 7, "d4_bg": 0.0},
    8: {"dt_atmos": 1200.0, "n_split": 7, "d4_bg": 0.12},
}


def build_duo_sw(test_num: int, n: int):
    """Six-face duo SW setup for Williamson 2/5/6 or colliding modons (8).

    Returns a namespace with ``state`` (face-stacked JAX dict), ``dt``
    (= the deck's ``dt_atmos`` outer block), ``step(state, dt)``,
    ``h(state)`` (fluid depth, (6, n, n)), ``winds(state)``, geometry,
    ``radius`` and a provenance ``note``.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.core.fv3_duo_stepper import (
        CUBE_PGRAD_BETA, SWConfig, advance_duo_outer_step,
        build_jax_duo_stepper_context, make_full_acoustic_step_sixface_jit,
        states_to_jax)
    from legoesm.core.fv3_native_duo_stepper import (
        SW_CFG_CASE8, build_six_face_duo_context, case6_six_face_state,
        w2_six_face_state)
    from legoesm.core.williamson_sw_analytic import (
        W5_H0_M, W5_UBAR_MS, solid_body_geopotential,
        williamson_5_mountain_height)
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_GRAV, FV3_OMEGA, FV3_RADIUS_M, analytic_swcore_state)

    if test_num not in _SW_DECKS:
        raise NotImplementedError(
            f"fv3_duo SW matrix adapter: no deck for test_num={test_num}")
    deck = _SW_DECKS[test_num]
    ctx_kw = dict(use_ext_bundle=True, oracle_conventions=True)
    hs_fn = None
    if test_num == 5:
        # Williamson (1992) mountain centre (270E, 30N) -- the SAME
        # williamson_5_mountain_height default every other matrix grid uses
        # (the FV3 oracle deck puts it at 90E; see williamson_sw_analytic).
        def hs_fn(lon, lat):
            return FV3_GRAV * williamson_5_mountain_height(lon, lat)
        ctx_kw["topo_fn"] = hs_fn
    if test_num == 8:
        ctx_kw["omega"] = 0.0           # non-rotating planet (FV3 case 8)
    ctx = build_six_face_duo_context(n, 3, **ctx_kw)

    if test_num == 2:
        states = w2_six_face_state(ctx)
    elif test_num == 6:
        states = case6_six_face_state(ctx)
    elif test_num == 5:
        def scalars_fn(ll):
            lon, lat = ll[..., 0], ll[..., 1]
            gh = solid_body_geopotential(
                lon, lat, radius=FV3_RADIUS_M, omega=FV3_OMEGA,
                u0=W5_UBAR_MS, gh0=W5_H0_M * FV3_GRAV) - hs_fn(lon, lat)
            return gh, np.ones_like(gh)
        states = []
        for gs in ctx["gs6"]:
            st = dict(analytic_swcore_state(gs, u0=W5_UBAR_MS, alpha=0.0,
                                            scalars_fn=scalars_fn))
            st["w"] = np.zeros_like(st["delp"])
            states.append(st)
    else:  # 8: colliding modons, tests.test_cases.colliding_modons formulas
        from tests.test_cases.colliding_modons import (
            _MODON_H0, _modon_winds_geo)

        def wind_fn(ll):
            u_e, v_n = _modon_winds_geo(ll[..., 0], ll[..., 1], FV3_RADIUS_M)
            return np.asarray(u_e), np.asarray(v_n)

        def scalars_fn(ll):
            delp = np.full(ll.shape[:-1], FV3_GRAV * _MODON_H0)
            return delp, np.ones_like(delp)
        states = []
        for gs in ctx["gs6"]:
            st = dict(analytic_swcore_state(gs, wind_fn=wind_fn,
                                            scalars_fn=scalars_fn))
            st["w"] = np.zeros_like(st["delp"])
            states.append(st)

    sw_cfg = SWConfig.from_mapping({**SW_CFG_CASE8, "d4_bg": deck["d4_bg"]})
    jctx = build_jax_duo_stepper_context(ctx)
    jstep = make_full_acoustic_step_sixface_jit()
    n_split = deck["n_split"]

    def step(state, dt):
        # production PG off-centering (FV3 beta, dyn_core.F90:1528; USER
        # APPROVED 2026-11-28) -- the oracle deck itself has beta = 0
        return advance_duo_outer_step(jctx, state, dt, n_split, d_ext=0.0,
                                      sw_cfg=sw_cfg, step_fn=jstep,
                                      beta=CUBE_PGRAD_BETA)

    lon_deg, lat_deg, area = duo_cell_geometry(ctx)
    sl = _compute_window(ctx)
    geo = _GeoWinds(ctx)

    def h(state):
        return np.asarray(state["delp"], dtype=np.float64)[:, sl, sl] / FV3_GRAV

    def winds(state):
        return geo(state["u"], state["v"])

    note = (f"fv3_duo JAX km=1 stepper, SW_CFG_CASE8 hord8 nord2 "
            f"d4_bg={deck['d4_bg']} d_ext=0 beta={CUBE_PGRAD_BETA:g} "
            f"(FV3 grad1_p_update), dt_atmos={deck['dt_atmos']:g}s "
            f"n_split={n_split}; FV3 GFS consts (a={FV3_RADIUS_M:g} m, "
            f"g={FV3_GRAV})")
    if test_num == 5:
        note += "; W5 mountain at Williamson (270E,30N) like other grids"
    return SimpleNamespace(
        ctx=ctx, state=states_to_jax(states), dt=deck["dt_atmos"],
        n_split=n_split, step=step, h=h, winds=winds, lon_deg=lon_deg,
        lat_deg=lat_deg, area=area, radius=FV3_RADIUS_M, grav=FV3_GRAV,
        note=note)


# ---------------------------------------------------------------------------
# cosine bell (FV3 test_case 1 analogue)
# ---------------------------------------------------------------------------

def _tc1_transport_step(jctx, delp6, uc6, vc6, dt, cfg):
    """delp transport by frozen C-grid winds: ``ext_scalar(delp)`` ->
    ``d_sw1`` per face -> BARRIER 1 -> ``d_sw2`` per face.  Exactly the
    transport half of ``fv3_duo_stepper.dsw12_step_sixface`` with the
    ``geopk``/``p_grad_c`` stage skipped (SW test_case 1)."""
    import jax.numpy as jnp
    from legoesm.core.fv3_duo_stepper import (
        average_allflux_shared_edges, d_sw1_duo, d_sw2_duo,
        ext_scalar_sixface)
    bd, npx, n, m_a, tab = jctx.bd, jctx.npx, jctx.n, jctx.m_a, jctx.tab
    delp6 = ext_scalar_sixface(delp6, tab, "A")
    pt6 = jnp.ones_like(delp6)
    w6 = jnp.zeros_like(delp6)
    zx = jnp.zeros((npx, n), dtype=jnp.float64)
    zy = jnp.zeros((n, npx), dtype=jnp.float64)
    zcx = jnp.zeros((npx, m_a), dtype=jnp.float64)
    zcy = jnp.zeros((m_a, npx), dtype=jnp.float64)
    s1 = [d_sw1_duo(delp6[t], pt6[t], w6[t], uc6[t], vc6[t], zx, zy, zcx,
                    zcy, jctx.gs6[t], jctx.flags6[t], bd, npx, npx, dt=dt,
                    hord_tr=cfg.hord_tr, hord_vt=cfg.hord_vt,
                    hord_tm=cfg.hord_tm, hord_dp=cfg.hord_dp,
                    nord_v=cfg.nord_v, nord_t=0, damp_v=cfg.damp_v,
                    damp_t=0.0, nq=int(tab.nq), workspace_sentinel=0.0)
          for t in range(6)]
    afx = jnp.stack([s["allflux_x"] for s in s1])
    afy = jnp.stack([s["allflux_y"] for s in s1])
    afx, afy = average_allflux_shared_edges(afx, afy, tab)
    out = [d_sw2_duo(s1[t]["delp"], s1[t]["pt"], afx[t], afy[t],
                     jctx.gs6[t], jctx.flags6[t], bd)["delp"]
           for t in range(6)]
    return jnp.stack(out)


def build_duo_cosine_bell(n: int, alpha: float, dt: float = 1800.0):
    """Six-face duo cosine-bell transport (Williamson case 1 / PL07 4.1)."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from legoesm.core.fv3_duo_stepper import (
        SWConfig, build_jax_duo_stepper_context, csw_step_sixface,
        exchange_post_pgrad_sixface, ext_vector_dgrid_sixface,
        states_to_jax)
    from legoesm.core.fv3_native_duo_stepper import (
        SW_CFG_CASE8, build_six_face_duo_context)
    from legoesm.core.williamson_sw_analytic import solid_body_rotation_speed
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_RADIUS_M, analytic_swcore_state)
    from tests.test_cases.cosine_bell import cosine_bell_exact

    ctx = build_six_face_duo_context(n, 3, use_ext_bundle=True,
                                     oracle_conventions=True)
    u0 = solid_body_rotation_speed(FV3_RADIUS_M)

    def scalars_fn(ll):
        h = np.asarray(cosine_bell_exact(ll[..., 0], ll[..., 1],
                                         FV3_RADIUS_M, 0.0, alpha))
        return h, np.ones_like(h)
    states = [dict(analytic_swcore_state(gs, u0=u0, alpha=alpha,
                                         scalars_fn=scalars_fn))
              for gs in ctx["gs6"]]
    jst = states_to_jax(states, keep_w=False)
    jctx = build_jax_duo_stepper_context(ctx)
    cfg = SWConfig.from_mapping({**SW_CFG_CASE8, "d4_bg": 0.12})
    # frozen C winds: duo D-vector exchange, then c_sw at dt2 = 0 (pure
    # D->C interpolation, no half-step advance) + the post-PG C-vector
    # exchange that fills the uc/vc halos d_sw1 reads.
    u6, v6 = ext_vector_dgrid_sixface(jst["u"], jst["v"], jctx.tab)
    csw = csw_step_sixface(jctx, {**jst, "u": u6, "v": v6}, 0.0)
    _, uc6, vc6 = exchange_post_pgrad_sixface(
        jctx, csw["divg_d"], csw["uc"], csw["vc"], nord=0)
    stepj = jax.jit(lambda d, dt_: _tc1_transport_step(
        jctx, d, uc6, vc6, dt_, cfg))

    def step(delp6, dt_):
        return stepj(delp6, dt_)

    lon_deg, lat_deg, area = duo_cell_geometry(ctx)
    sl = _compute_window(ctx)
    lon_r, lat_r = np.radians(lon_deg), np.radians(lat_deg)

    def h(delp6):
        return np.asarray(delp6, dtype=np.float64)[:, sl, sl]

    def exact(t):
        return np.asarray(cosine_bell_exact(lon_r, lat_r, FV3_RADIUS_M, t,
                                            alpha))

    note = (f"fv3_duo TC1 frozen-wind delp transport (d_sw1/barrier1/d_sw2,"
            f" hord_dp=8), c_sw(dt2=0) C winds, dt={dt:g}s")
    return SimpleNamespace(
        ctx=ctx, state=jnp.asarray(jst["delp"]), dt=dt, step=step, h=h,
        exact=exact, lon_deg=lon_deg, lat_deg=lat_deg, area=area,
        radius=FV3_RADIUS_M, note=note)


# ---------------------------------------------------------------------------
# hydrostatic 3-D (DCMIP16 baroclinic wave / Held-Suarez)
# ---------------------------------------------------------------------------

#: the vertical table the matrix rows run: CAM6 L32 (the duo's only table
#: between the analytic L5/L10 and FV3 L40).
DUO_HYDRO_KM = 32
DUO_HYDRO_ETA = "cam6_l32"
#: The duo deck's outer step: dt_atmos = 1920 s with its n_split = 8
#: acoustic sub-steps (~240 s each).  NOT clamped by the driver's generic
#: advective+acoustic CFL heuristic: FV3's dimensionless per-substep
#: damping (vtdm4 del-4 on delp/pt/vorticity, divergence damping) is tuned
#: per acoustic substep, so the clamp (C36: 1920 -> 200 s at fixed n_split)
#: applied it ~9.6x more often per day and drove the rest_state_topo
#: spurious wind (C36 --quick matrix row, 1 d: 13.47 m/s clamped vs 7.25
#: at the deck step -- both measured on the PRE-2026-11-27 deck, vtdm4=0.12
#: del-6; the FV3-rules deck, vtdm4=d4_bg/3 del-4, gives ~2.1 at the deck
#: step).
DUO_HYDRO_DT_ATMOS = 1920.0
#: DCMIP 2012 test 3-1 (X = 125 small planet): the deck step divided by X
#: (1920/125 = 15.36 s, the document's "divide the dynamics time step by
#: X") rounded DOWN to 15 s so 240 steps land exactly on its 3600 s
#: analysis time (vote 2026-11-26: author B, codex B).  Same n_split = 8;
#: the per-substep Courant number is at most the validated deck's.
DUO_GW31_DT_ATMOS = 15.0
DUO_GW31_KM = 10
DUO_GW31_ETA = "dcmip2012_gw31_l10"
#: The duo table of the matrix's sigma-labelled rows (Held-Suarez,
#: baroclinic, baroclinic_steady): the other grids' sigma coordinate
#: (``create_sigma_coordinate(nlev)``) as a fixed-lid FV3 table
#: (``fv3_sigma_eta_table``); km is the row's nlev.
DUO_SIGMA_ETA = "sigma"


def _cam6_l32_coord(model, what: str):
    """The CAM6 L32 ``HybridSigmaPressureCoordinate`` the shared analytic
    builders run on, after checking it IS the model's eta table."""
    from legoesm.grids.vertical import CAM6_L32_P0, make_cam6_l32_levels
    coord = make_cam6_l32_levels()
    if coord.n_levels != model.config.km or not (
            # the two tables round differently at ~1e-8 relative
            # (measured max 6.7e-4 Pa in ak); a different table is O(1)
            np.allclose(np.asarray(coord.A_half) * CAM6_L32_P0, model.ak,
                        rtol=1e-6, atol=1e-2)
            and np.allclose(np.asarray(coord.B_half), model.bk,
                            rtol=1e-6, atol=1e-12)):
        raise ValueError(f"{what}: the model's eta table is not the CAM6 "
                         f"L32 coordinate the IC is built on")
    return coord


def _gw31_coord(model, what: str):
    """The DCMIP 2012 test 3-1 App. F.3 L10 ``HybridSigmaPressureCoordinate``
    after checking it IS the model's eta table (same float64 source)."""
    from legoesm.grids.vertical import (
        DCMIP2012_GW31_PEQ, make_dcmip2012_gw31_levels)
    coord = make_dcmip2012_gw31_levels()
    if coord.n_levels != model.config.km or not (
            np.allclose(np.asarray(coord.A_half) * DCMIP2012_GW31_PEQ,
                        model.ak, rtol=1e-13, atol=1e-9)
            and np.allclose(np.asarray(coord.B_half), model.bk,
                            rtol=1e-13, atol=1e-15)):
        raise ValueError(f"{what}: the model's eta table is not the DCMIP "
                         f"2012 3-1 L10 coordinate the IC is built on")
    return coord


def rotated_jw_bundle(model, *, perturbed: bool, alpha: float) -> dict:
    """Rotated DCMIP-2016 (Ullrich 2014) baroclinic-wave bundle on the duo
    grid (DCMIP 2008 1-3-0 / 2-3-0 rotation recipe).

    The analytic state is the SAME builder every other matrix grid uses
    (``jablonowski_rotated._build_rotated_state_dense_grid``) on the run's
    own CAM6 L32 coordinate (``make_cam6_l32_levels``; its
    ``sigma_full * P0`` is the duo's full-level pressure at ``ps = P0``).
    T fills ``pt`` at A-grid centres; geographic winds at D-edge midpoints
    are projected onto the edge tangents by the certified
    ``analytic_swcore_state`` covariant projection.  Starting from the
    model's own DCMIP16 bundle keeps delp/ps/press (``ps = P0`` in both
    ICs, so the layer masses are identical) and the passive sphum
    passenger; only the compute windows of ``pt``/``u``/``v`` are
    overwritten (halos stay as the DCMIP16 builder leaves them; the step's
    entry exchanges fill them).
    """
    import jax.numpy as jnp
    from legoesm.grids.fv3_native_gridstruct import analytic_swcore_state
    from tests.test_cases.dcmip2008.jablonowski_rotated import (
        _build_rotated_state_dense_grid)

    coord = _cam6_l32_coord(model, "rotated baroclinic duo IC")
    km = model.config.km
    ctx = model.grid.ctx_np
    n, ng = ctx["n"], ctx["ng"]
    sl = _compute_window(ctx)
    bundle = model.dcmip16_initial_state(do_pert=False)
    st = dict(bundle["state"])
    pt = np.array(st["pt"])
    u = np.array(st["u"])
    v = np.array(st["v"])
    for t, gs in enumerate(ctx["gs6"]):
        lon = np.asarray(gs["agrid_lon"])[sl, sl]
        lat = np.asarray(gs["agrid_lat"])[sl, sl]
        _, _, T, _ = _build_rotated_state_dense_grid(
            lon, lat, coord, perturbed, float(alpha))
        pt[t, sl, sl, :] = np.asarray(T)
        cache = {}

        def winds3d(ll):
            key = ll.tobytes()
            if key not in cache:
                ue, vn, _, _ = _build_rotated_state_dense_grid(
                    ll[..., 0], ll[..., 1], coord, perturbed, float(alpha))
                cache[key] = (np.asarray(ue), np.asarray(vn))
            return cache[key]
        for k in range(km):
            d = analytic_swcore_state(
                gs, wind_fn=lambda ll, k=k: tuple(a[..., k]
                                                  for a in winds3d(ll)))
            u[t, sl, ng:ng + n + 1, k] = d["u"][sl, ng:ng + n + 1]
            v[t, ng:ng + n + 1, sl, k] = d["v"][ng:ng + n + 1, sl]
    st.update(pt=jnp.asarray(pt), u=jnp.asarray(u), v=jnp.asarray(v))
    return {**bundle, "state": st}


def held_suarez_rest_bundle(model, *, T_init: float = 300.0,
                            perturbation_amplitude: float = 1.0,
                            seed: int = 42) -> dict:
    """Held-Suarez initial state on the duo grid, identical in definition to
    every other matrix grid's (``held_suarez_init_latlon`` / ``_mpas`` /
    ``isothermal_rest_state_spectral``): isothermal ``T_init`` at rest,
    ``p_s = p_ref``, plus a ``N(0, 1) * perturbation_amplitude`` temperature
    perturbation (``jax.random.PRNGKey(seed)``) on the LOWEST level only.

    Built on the model's own DCMIP16 bundle, whose ``ps`` is the uniform
    1000 hPa = ``p_ref`` (so delp/press are the rest state's), with ``pt``
    (temperature in the bundle; the first step converts it) and the D-grid
    winds overwritten.  FV3's level index runs top -> bottom, so the lowest
    level is ``km - 1``.
    """
    import jax
    import jax.numpy as jnp
    from legoesm import constants

    bundle = model.dcmip16_initial_state(do_pert=False)
    ps = np.asarray(bundle["press"]["ps"])
    sl = _compute_window(model.grid.ctx_np)
    if not np.allclose(ps[:, sl, sl], constants.p_ref, rtol=0, atol=1e-6):
        raise ValueError("Held-Suarez duo IC needs the DCMIP16 bundle's "
                         "uniform ps == p_ref")
    st = dict(bundle["state"])
    pt = np.array(st["pt"])
    km = pt.shape[-1]
    pt[:, sl, sl, :] = T_init
    n = model.grid.n
    pert = np.asarray(jax.random.normal(jax.random.PRNGKey(seed), (6, n, n),
                                        dtype=jnp.float64)) * perturbation_amplitude
    pt[:, sl, sl, km - 1] += pert
    st.update(pt=jnp.asarray(pt), u=jnp.zeros_like(st["u"]),
              v=jnp.zeros_like(st["v"]))
    return {**bundle, "state": st}


# ---------------------------------------------------------------------------
# rows whose IC is a shared cube builder (DCMIP 2012 2-0-0, HS-topo,
# DCMIP 2008 3-1 / 3-2 / 5-0 / 6-0)
# ---------------------------------------------------------------------------

def _mountain_2_0_0_phis(h_0: float):
    """``phis(lon, lat)`` of the DCMIP 2012 2-0-0 ridged mountain, the
    composition both shared builders (``rest_state_topography_init`` and
    ``held_suarez_topo._phis_cube``) use: ``constants.g * z_s``."""
    from legoesm import constants
    from legoesm.atmosphere.idealized.topography import dcmip_2_0_0_mountain
    from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M

    def fn(lon, lat):
        return constants.g * np.asarray(dcmip_2_0_0_mountain(
            lon, lat, FV3_RADIUS_M, h_0=h_0))
    return fn


def _solid_body_winds(u_0: float):
    """``wind_fn(ll)`` of the DCMIP 2008 isothermal base flow
    ``(u_0 cos(lat), 0)`` (``dcmip2008._shared.isothermal_state_*``)."""
    def fn(ll):
        u = u_0 * np.cos(ll[..., 1])
        return u, np.zeros_like(u)
    return fn


def duo_shared_ic_spec(case: str, ic_kwargs: dict | None = None):
    """Per-row recipe for a cube row whose IC is a SHARED cube builder.

    Returns a namespace with ``builder(grid, coord) -> HydrostaticState``
    (the very ``*_init`` the legacy cube / every grid's dispatch names),
    ``phis_fn(lon, lat)`` for the duo grid's terrain (``None`` = flat),
    ``wind_fn(ll)`` for the geographic winds at D-edge midpoints (``None``
    = rest), ``omega`` for the grid (``None`` = FV3_OMEGA) and a ``label``.
    The terrain and winds the builder itself produces at the cell centres
    are checked against ``phis_fn`` / ``wind_fn`` in
    :func:`shared_ic_bundle`, so a recipe that drifted from its builder
    fails loudly.
    """
    kw = dict(ic_kwargs or {})
    if case == "rest_state_topo":
        from tests.test_cases.dcmip2012.rest_state_topography import (
            lapse_rate_state_topography_init as init)
        h_0 = float(kw.get("h_0", 2000.0))
        return SimpleNamespace(
            builder=lambda g, c: init(g, c, **kw),
            # DCMIP 2012 2-0-0: Omega = 0 (Table XII)
            phis_fn=_mountain_2_0_0_phis(h_0), wind_fn=None, omega=0.0,
            radius=None, eta=None,
            label=f"DCMIP 2012 2-0-0 rest state: constant lapse rate 6.5 K/km on native L32 levels (reference 12 km top not reproduced), non-rotating, h_0={h_0:g} m")
    if case == "held_suarez_topo":
        from legoesm.atmosphere.idealized.held_suarez_topo import (
            DEFAULT_H_0, held_suarez_topo_init as init)
        h_0 = float(kw.get("h_0", DEFAULT_H_0))
        return SimpleNamespace(
            builder=lambda g, c: init(g, c, **kw),
            phis_fn=_mountain_2_0_0_phis(h_0), wind_fn=None, omega=None,
            radius=None, eta=None,
            label=(f"held_suarez_topo_init (h_0={h_0:g} m, "
                   f"seed={kw.get('seed', 'default')})"))
    if case == "gravity_wave_3_1":
        # DCMIP 2012 test 3-1: non-rotating X = 125 small planet, App. F.3
        # L10 to a 10 km lid, u = u0 cos(lat)
        from tests.test_cases.dcmip2012 import gravity_wave_3_1 as gw31
        if kw:
            raise ValueError(f"{case}: no IC kwargs, got {sorted(kw)}")
        return SimpleNamespace(
            builder=gw31.gravity_wave_init, phis_fn=None,
            wind_fn=_solid_body_winds(gw31.U0), omega=gw31.OMEGA,
            radius=gw31.small_planet_radius(), eta=DUO_GW31_ETA,
            label=(f"DCMIP 2012 test 3-1 ({gw31.__name__}."
                   f"gravity_wave_init), a = R_earth/{gw31.X_SCALE:g}"))
    if case in ("inertio_gravity_3_2", "mountain_rossby_5_0"):
        from tests.test_cases.dcmip2008._shared import U0_DEFAULT
        if kw:
            raise ValueError(f"{case}: no IC kwargs, got {sorted(kw)}")
        wind_fn = _solid_body_winds(U0_DEFAULT)
        if case == "inertio_gravity_3_2":
            # DCMIP 2008 [6-3-0]: at rest (u0 = 0) on the rotating Earth
            from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
                inertio_gravity_init as init)
            phis_fn, omega, wind_fn = None, None, None
        else:
            from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
            from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
                _phis, mountain_rossby_init as init)

            def phis_fn(lon, lat):
                return np.array(_phis(lon, lat, FV3_RADIUS_M), dtype=np.float64)
            omega = None
        return SimpleNamespace(
            builder=init, phis_fn=phis_fn, wind_fn=wind_fn, omega=omega,
            radius=None, eta=None,
            label=f"DCMIP 2008 {case} ({init.__name__})")
    if case == "rossby_haurwitz_6_0":
        from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
        from tests.test_cases.dcmip2008.rossby_haurwitz_6_0 import (
            _w6_winds_geo, rossby_haurwitz_init as init)
        if kw:
            raise ValueError(f"{case}: no IC kwargs, got {sorted(kw)}")

        def wind_fn(ll):
            u, v = _w6_winds_geo(ll[..., 0], ll[..., 1], FV3_RADIUS_M)
            return np.asarray(u), np.asarray(v)
        return SimpleNamespace(
            builder=init, phis_fn=None, wind_fn=wind_fn, omega=None,
            radius=None, eta=None,
            label=f"DCMIP 2008 {case} ({init.__name__})")
    raise NotImplementedError(
        f"{case} on cube: no fv3_duo shared-IC recipe for this case")


def shared_ic_bundle(model, spec) -> dict:
    """The row's shared cube builder evaluated on the duo grid.

    The builder runs on a duck-typed cube grid whose cells are the duo
    A-grid compute cells (``n``, ``lon``/``lat`` (6, n, n) [rad], the duo
    ``radius``, ``angle = 0`` so its winds come back geographic) and on
    the run's own CAM6 L32 coordinate.  Its ``T`` fills ``pt``; its
    ``p_s`` (hydrostatic over ``phis`` with ``constants.R_d`` ==
    ``FV3_RDGAS``) sets ``delp = diff(ak + bk p_s)`` and the pressure
    bundle (``p_var_hydrostatic``, as ``era5_to_fv3_duo_bundle``); winds
    are ``spec.wind_fn`` projected onto the D-edge tangents by the
    certified ``analytic_swcore_state``.  Everything else (``w``, the
    passive humidity passenger, ``omga``, halos -- the step's entry
    exchanges fill them) is the model's own DCMIP16 bundle.

    Refuses (no silent substitution) when the builder's ``phis`` is not
    the grid's terrain or its centre winds are not ``spec.wind_fn``.
    """
    import jax.numpy as jnp
    from legoesm.core.fv3_dynamics import p_var_hydrostatic
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_KAPPA, FV3_OMEGA, analytic_swcore_state)

    coord = (_gw31_coord(model, "duo shared IC")
             if spec.eta == DUO_GW31_ETA else
             _cam6_l32_coord(model, "duo shared IC"))
    ctx = model.grid.ctx_np
    n, ng, km = ctx["n"], ctx["ng"], model.config.km
    sl = _compute_window(ctx)
    gs6 = ctx["gs6"]
    lon = np.stack([np.asarray(g["agrid_lon"])[sl, sl] for g in gs6])
    lat = np.stack([np.asarray(g["agrid_lat"])[sl, sl] for g in gs6])
    # omega: the duo's own rotation rate (the recipe's, else FV3_OMEGA) for
    # builders whose balance depends on it (mountain_rossby_5_0, Eq. 83)
    if spec.omega == 0.0 and any(
            np.any(np.asarray(g[k]) != 0.0) for g in gs6 for k in ("f0", "fC")):
        # the IC's non-rotating balance needs a non-rotating grid
        raise ValueError("duo shared IC: the recipe is non-rotating but the "
                         "grid's Coriolis parameter is not zero")
    # the grid's own radius (FV3_RADIUS_M unless a small planet was built)
    cells = SimpleNamespace(n=n, lon=jnp.asarray(lon), lat=jnp.asarray(lat),
                            radius=float(ctx["radius"]),
                            angle=jnp.zeros_like(lat),
                            omega=FV3_OMEGA if spec.omega is None
                            else float(spec.omega))
    hst = spec.builder(cells, coord)

    hs6 = ctx.get("hs6")
    grid_phis = (np.zeros((6, n, n)) if hs6 is None
                 else np.stack([np.asarray(h)[sl, sl] for h in hs6]))
    # the builder stores phis at the active precision policy's dtype
    # (held_suarez_init: storage dtype, fp32 by default), so compare it
    # with the grid's terrain rounded to THAT dtype
    ic_dtype = np.asarray(hst.phis.data).dtype
    ic_phis = np.asarray(hst.phis.data, dtype=np.float64)
    grid_phis = grid_phis.astype(ic_dtype).astype(np.float64)
    if (spec.phis_fn is None) != (hs6 is None) or not np.allclose(
            ic_phis, grid_phis, rtol=1e-12, atol=1e-9):
        raise ValueError("duo shared IC: the builder's phis is not the "
                         "grid's terrain (max diff "
                         f"{np.max(np.abs(ic_phis - grid_phis)):.3e})")
    u_c = np.asarray(hst.u.data, dtype=np.float64)
    v_c = np.asarray(hst.v.data, dtype=np.float64)
    if spec.wind_fn is None:
        ue, vn = np.zeros_like(lon), np.zeros_like(lon)
    else:
        ue, vn = spec.wind_fn(np.stack([lon, lat], axis=-1))
    scale = max(1.0, float(np.max(np.abs(u_c))))
    if not (np.allclose(u_c, np.asarray(ue)[..., None], rtol=0,
                        atol=1e-12 * scale)
            and np.allclose(v_c, np.asarray(vn)[..., None], rtol=0,
                            atol=1e-12 * scale)):
        raise ValueError("duo shared IC: the builder's winds are not the "
                         "recipe's wind_fn (or not depth-independent)")

    bundle = model.dcmip16_initial_state(do_pert=False)
    st = dict(bundle["state"])
    pt = np.array(st["pt"])
    delp = np.array(st["delp"])
    u = np.zeros_like(np.asarray(st["u"]))
    v = np.zeros_like(np.asarray(st["v"]))
    ps = np.asarray(hst.p_s.data, dtype=np.float64)
    ak, bk = np.asarray(model.ak), np.asarray(model.bk)
    pt[:, sl, sl, :] = np.asarray(hst.T.data, dtype=np.float64)
    delp[:, sl, sl, :] = (np.diff(ak)[None, None, None, :]
                          + np.diff(bk)[None, None, None, :] * ps[..., None])
    if spec.wind_fn is not None:
        # depth-independent: one projection per face, broadcast over km
        for t, gs in enumerate(gs6):
            d = analytic_swcore_state(gs, wind_fn=spec.wind_fn)
            u[t, sl, ng:ng + n + 1, :] = d["u"][sl, ng:ng + n + 1, None]
            v[t, ng:ng + n + 1, sl, :] = d["v"][ng:ng + n + 1, sl, None]
    st.update(pt=jnp.asarray(pt), delp=jnp.asarray(delp),
              u=jnp.asarray(u), v=jnp.asarray(v))
    press = p_var_hydrostatic(st["delp"], ptop=model.ptop, akap=FV3_KAPPA,
                              n=n, ng=ng, km=km)
    if not np.allclose(np.asarray(press["ps"])[:, sl, sl], ps, rtol=1e-12,
                       atol=0.0):
        raise ValueError("duo shared IC: p_var ps != the builder's p_s")
    return {**bundle, "state": st, "press": press}


def build_duo_hydro(n: int, *, perturbed: bool = True,
                    held_suarez: bool = False, km: int = DUO_HYDRO_KM,
                    eta: str = DUO_HYDRO_ETA,
                    rotated_alpha: float | None = None,
                    shared_case: str | None = None,
                    ic_kwargs: dict | None = None,
                    hydrostatic_logp: bool = False):
    """FV3DuoDynamicsModel (production deck defaults) on C``n``.

    ``dt`` is the deck's ``DUO_HYDRO_DT_ATMOS`` (1920 s, n_split 8), not the
    driver's generic CFL clamp (see the constant).
    ``rotated_alpha`` (not None): rotated DCMIP-2016 wave -- the planet's
    rotation axis tilted by alpha (``create_fv3_duo_grid(rotation_alpha)``)
    and :func:`rotated_jw_bundle` as the IC.
    ``eta=DUO_SIGMA_ETA`` (with ``km`` = the row's nlev): the sigma rows'
    table (pure sigma below a fixed 0.01 p_ref lid); only with the
    Held-Suarez rest IC or the un-rotated DCMIP16 wave, whose builders
    take any ak/bk.
    ``shared_case`` (not None): the row's IC is its shared cube builder
    (:func:`duo_shared_ic_spec`, ``ic_kwargs`` passed through), with the
    recipe's terrain (``phis_fn``, unfiltered like every other grid's) and
    rotation rate on the grid; combines with ``held_suarez`` for HS-topo.
    ``hydrostatic_logp``: ``FV3DuoConfig.hydrostatic_logp`` (Lin 1997
    sec. 3 log-p hydrostatic PGF OPTION; default False = FV3 p**kappa).
    Only the held_suarez_topo matrix row passes True (user decision
    2026-11-27); the row note records it.
    """
    if n != 36:
        # The deck step (1920 s, n_split 8) is validated on the full cube
        # matrix at C36 only; refuse other resolutions rather than run an
        # unvalidated Courant number (vote: codex A, author A).
        raise ValueError(f"build_duo_hydro: deck dt {DUO_HYDRO_DT_ATMOS:.0f} s "
                         f"is validated at C36 only, got C{n}")
    gw31 = shared_case == "gravity_wave_3_1"
    if not gw31 and eta == DUO_GW31_ETA:
        raise ValueError(f"eta={DUO_GW31_ETA!r} is the gravity_wave_3_1 "
                         f"small-planet table; it goes with that case only")
    if gw31 and (km, eta) != (DUO_HYDRO_KM, DUO_HYDRO_ETA):
        raise ValueError("gravity_wave_3_1 runs its own DCMIP 2012 L10 "
                         "table; do not pass km/eta")
    if eta == DUO_SIGMA_ETA and (shared_case is not None
                                 or rotated_alpha is not None):
        raise ValueError(f"eta={DUO_SIGMA_ETA!r}: the shared-IC and rotated "
                         f"builders are built on CAM6 L32 only")
    if gw31:
        # the one admitted small-planet configuration: C36, the 3-1 L10
        # table, a = R_earth/125 and DUO_GW31_DT_ATMOS -- checked below
        # against the validated deck Courant number
        km, eta = DUO_GW31_KM, DUO_GW31_ETA
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel, resolve_pgrad_beta)
    from legoesm.grids.factory import create_fv3_duo_grid

    spec = None
    grid_kw = {}
    if shared_case is not None:
        if rotated_alpha is not None:
            raise ValueError("shared_case and rotated_alpha are exclusive")
        if held_suarez != (shared_case == "held_suarez_topo"):
            raise ValueError("held_suarez forcing goes with shared_case="
                             "'held_suarez_topo' only")
        spec = duo_shared_ic_spec(shared_case, ic_kwargs)
        if spec.phis_fn is not None:
            grid_kw["phis_fn"] = spec.phis_fn
        if spec.omega is not None:
            grid_kw["omega"] = spec.omega
        if spec.radius is not None:
            grid_kw["radius"] = spec.radius
    elif ic_kwargs:
        raise ValueError("ic_kwargs needs shared_case")
    grid = create_fv3_duo_grid(
        n, rotation_alpha=0.0 if rotated_alpha is None else rotated_alpha,
        **grid_kw)
    if not isinstance(hydrostatic_logp, bool):
        raise TypeError("hydrostatic_logp must be a bool")
    cfg = FV3DuoConfig(km=km, eta=eta, hydrostatic=True,
                       hydrostatic_logp=hydrostatic_logp)
    model = FV3DuoDynamicsModel(grid, cfg)
    if spec is not None:
        bundle = shared_ic_bundle(model, spec)
    elif held_suarez:
        bundle = held_suarez_rest_bundle(model)
    elif rotated_alpha is None:
        # DCMIP-2016 document Eq. (14): the perturbation is truncated at
        # R_p, as on the other three grids (the FV3 oracle is not)
        bundle = model.dcmip16_initial_state(do_pert=perturbed,
                                             use_dcmip2016_perturbation=True)
    else:
        bundle = rotated_jw_bundle(model, perturbed=perturbed,
                                   alpha=rotated_alpha)
    dt = DUO_HYDRO_DT_ATMOS
    if gw31:
        from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
        dt = DUO_GW31_DT_ATMOS
        from tests.test_cases.dcmip2012.gravity_wave_3_1 import check_step
        # geometric time-step ceiling: the metrics are the deck's scaled by
        # radius/FV3_RADIUS_M, so dt <= 1920 s * radius / FV3_RADIUS_M keeps
        # the deck's dt/dx (the 3-1 wind and sound speeds, <= 20 m/s and
        # ~340 m/s, are no larger than the deck rows'; the 240-step C36
        # matrix run stays finite)
        if not (np.isfinite(dt) and 0.0 < dt <= DUO_HYDRO_DT_ATMOS
                * float(grid.ctx_np["radius"]) / FV3_RADIUS_M):
            raise ValueError("build_duo_hydro: the 3-1 step exceeds the "
                             "validated deck's geometric step ceiling")
        # and the run must land exactly on the document's 3000 s / 3600 s
        # analysis times (shared rule; the matrix cadence uses it too)
        check_step(dt)

    hs_fn = None
    if held_suarez:
        # the driver lane's face-stacked JAX HS twin
        # (ModelDriver._fv3_duo_apply_held_suarez_jax, single-process faces)
        import jax.numpy as jnp  # noqa: F401
        from legoesm.core.fv3_native_physics_coupling import (
            apply_held_suarez_step_sixface_jax, stack_held_suarez_metrics)
        tab = model.sixface_halo_tables
        amat6, lat6, wv6 = stack_held_suarez_metrics(grid.ctx_np)
        ng = grid.ng

        # strat=False: plain Held & Suarez (1994), as on every other grid.
        # FV3 solo's default (do_strat_HS_forcing, hswf.F90) adds a modified
        # stratosphere above 100 hPa (T_eq rising 2.25 K/km x cos(lat), 10-40 d
        # relaxation) that drives a ~25 m/s westerly jet at the top level and
        # made the matrix row non-comparable (30 d max|V| 25.7 vs 15.3 m/s).
        def _hs(state, press, dt_):
            return apply_held_suarez_step_sixface_jax(
                state, press, tab, amat6, lat6, wv6, dt=dt_, n=n, ng=ng,
                km=km, strat=False)
        hs_jit = jax.jit(_hs, static_argnums=(2,))

        def hs_fn(b, dt_):
            press = {k: b["press"][k] for k in ("pe", "peln", "pkz")}
            return {**b, "state": hs_jit(b["state"], press, float(dt_))}

    def step(b, dt_):
        b = model.step(b, dt_)
        if hs_fn is not None:
            b = hs_fn(b, dt_)
        return b

    ctx = grid.ctx_np
    lon_deg, lat_deg, area = duo_cell_geometry(ctx)
    sl = _compute_window(ctx)
    geo = _GeoWinds(ctx)
    ak, bk = np.asarray(model.ak), np.asarray(model.bk)
    p0 = 1.0e5
    level_values = 0.5 * ((ak[:-1] + ak[1:]) / p0 + (bk[:-1] + bk[1:]))

    def ps(b):
        return np.asarray(b["press"]["ps"], dtype=np.float64)[:, sl, sl]

    def temperature(b):
        return np.asarray(b["state"]["pt"], dtype=np.float64)[:, sl, sl, :]

    def winds(b, level=None):
        u = np.asarray(b["state"]["u"])
        v = np.asarray(b["state"]["v"])
        if level is not None:
            u, v = u[..., level], v[..., level]
        return geo(u, v)

    note = (f"fv3_duo FV3DuoDynamicsModel km={km} eta={eta} hydrostatic, "
            f"nord={cfg.nord} d4_bg={cfg.d4_bg} vtdm4="
            f"{'d4_bg/3' if cfg.vtdm4 is None else cfg.vtdm4} (FV3 rules) "
            f"sponge={cfg.sponge_del2_top_layers}"
            f"L, PG beta={resolve_pgrad_beta(cfg):g}, "
            f"n_split={cfg.n_split}, dt={dt:g}s ("
            + ("small-planet step: deck dt_atmos 1920 s / X = 125 rounded "
               "down to divide 3600 s" if gw31 else "deck dt_atmos")
            + ", no driver CFL clamp); " + (
                f"shared cube IC {spec.label} on the duo cells + "
                + ("DCMIP 2012 3-1 L10 (App. F.3), radius="
                   f"{float(grid.ctx_np['radius']):.6g} m"
                   if gw31 else "CAM6 L32")
                + ("" if spec.phis_fn is None else
                   ", grid terrain = the IC's phis (unfiltered)")
                + ("" if spec.omega is None else
                   f", omega={spec.omega:g} (non-rotating)")
                if spec is not None else
                "Held-Suarez IC: isothermal 300 K at rest, ps = p_ref, "
                "1 K lowest-level perturbation (seed 42), as every grid"
                if held_suarez else
                (f"DCMIP16 baroclinic-wave IC (do_pert={perturbed}"
                 + (", DCMIP-2016 Eq. 14-15 perturbation)" if perturbed
                    else ")"))
                if rotated_alpha is None else
                f"rotated DCMIP-2016 IC (jablonowski_rotated builder on CAM6 L32, "
                f"alpha={rotated_alpha:.4f}, perturbed={perturbed}), "
                f"rotated-planet Coriolis"))
    if held_suarez:
        note += " + Held-Suarez 1994 forcing (FV3 hswf, strat=F) after each step"
    if hydrostatic_logp:
        note += ("; hydrostatic PGF: Lin 1997 sec. 3 log-p contour OPTION "
                 "(departs from FV3 p**kappa one_grad_p)")
    return SimpleNamespace(
        model=model, grid=grid, state=bundle, dt=dt, step=step, ps=ps,
        temperature=temperature, winds=winds, lon_deg=lon_deg,
        lat_deg=lat_deg, area=area, km=km, eta=eta,
        level_values=level_values, note=note)


# ---------------------------------------------------------------------------
# DCMIP 2012 prescribed-wind tracer transport (tests 1-1 / 1-2)
# ---------------------------------------------------------------------------

#: The explicitly selected duo vertical table of the DCMIP 2012 transport
#: rows (Appendix F.3 L30, fixed lid at ~25.5 kPa).  CAM6 L32 stays the
#: default of every other duo row.
DUO_DCMIP_TRANSPORT_ETA = "dcmip2012_l30"
DUO_DCMIP_TRANSPORT_KM = 30


class _DgridProjection:
    """Geographic winds -> duo D-grid covariant edge components, with the
    certified ``analytic_swcore_state`` recipe (unit tangent at the edge's
    great-circle midpoint dotted with the local east/north vectors,
    ``get_unit_vect2`` / ``mid_pt_sphere`` / ``get_latlon_vector``), the
    geometry precomputed ONCE so a time-dependent wind is one JAX
    multiply-add per step.  Only the compute-window edges are written
    (``u[sl, ng:ng+n+1]``, ``v[ng:ng+n+1, sl]``, as every duo IC builder);
    the halos are filled by the step's own duo exchange."""

    def __init__(self, ctx_np):
        import jax.numpy as jnp
        from legoesm.grids.fv3_native_gridstruct import (
            _get_unit_vect2, _latlon_vectors)
        from legoesm.grids.fv3_native_metrics import mid_pt_sphere
        n, ng = ctx_np["n"], ctx_np["ng"]
        self.n, self.ng = n, ng
        sl = slice(ng, ng + n)
        se = slice(ng, ng + n + 1)
        acc = {k: [] for k in ("ulon", "ulat", "uex", "uey",
                               "vlon", "vlat", "vex", "vey")}
        for gs in ctx_np["gs6"]:
            ll = np.stack([gs["grid_lon"], gs["grid_lat"]], axis=-1)
            for name, a, b, win in (("u", ll[:-1, :], ll[1:, :], (sl, se)),
                                    ("v", ll[:, :-1], ll[:, 1:], (se, sl))):
                e = _get_unit_vect2(a, b)
                mid = mid_pt_sphere(a, b)
                ex, ey = _latlon_vectors(mid)
                acc[name + "lon"].append(mid[..., 0][win])
                acc[name + "lat"].append(mid[..., 1][win])
                acc[name + "ex"].append(np.einsum("...i,...i", e, ex)[win])
                acc[name + "ey"].append(np.einsum("...i,...i", e, ey)[win])
        for k, v in acc.items():
            setattr(self, k, jnp.asarray(np.stack(v), dtype=jnp.float64))

    def __call__(self, geo_fn, u_tmpl, v_tmpl):
        """``geo_fn(lon, lat) -> (u_east, v_north)`` each ``(..., km)``;
        returns D-grid ``(u6, v6)`` shaped like the templates, compute
        edges filled, halos zero."""
        import jax.numpy as jnp
        n, ng = self.n, self.ng
        ue, vn = geo_fn(self.ulon, self.ulat)
        uc = (ue * self.uex[..., None] + vn * self.uey[..., None])
        ue, vn = geo_fn(self.vlon, self.vlat)
        vc = (ue * self.vex[..., None] + vn * self.vey[..., None])
        u6 = jnp.zeros_like(u_tmpl).at[:, ng:ng + n, ng:ng + n + 1, :].set(
            uc.astype(u_tmpl.dtype))
        v6 = jnp.zeros_like(v_tmpl).at[:, ng:ng + n + 1, ng:ng + n, :].set(
            vc.astype(v_tmpl.dtype))
        return u6, v6


def tangent_gradient_sixface(f, X, radius):
    """Tangent-plane gradient [per m] of cell-centre fields on the six duo
    faces: ``f`` (6, n, n[, K]), ``X`` (6, n, n, 3) unit position vectors.
    Per face: index derivatives (``np.gradient``: centred inside the face,
    one-sided at its edges -- a diagnostic, no cross-seam stencil) of f and
    of X, then per cell the 3x3 solve ``g . dX/di = df/di``,
    ``g . dX/dj = df/dj``, ``g . X = 0``; divided by ``radius``.  Returns
    (6, n, n[, K], 3)."""
    f = np.asarray(f, dtype=np.float64)
    extra = f.shape[3:]
    fk = f.reshape(6, f.shape[1], f.shape[2], -1)
    g = np.empty(fk.shape + (3,))
    for t in range(6):
        dfi, dfj = np.gradient(fk[t], axis=(0, 1))          # (n, n, K)
        dXi = np.gradient(X[t], axis=0)
        dXj = np.gradient(X[t], axis=1)
        A = np.stack([dXi, dXj, X[t]], axis=-2)              # (n, n, 3, 3)
        b = np.stack([dfi, dfj, np.zeros_like(dfi)], -1)     # (n, n, K, 3)
        Ainv = np.linalg.inv(A)                              # (n, n, 3, 3)
        g[t] = np.einsum("ijab,ijkb->ijka", Ainv, b)
    return g.reshape(f.shape + (3,)) / radius


def interface_pressure_advection(u, v, pe, X, e_east, e_north, radius):
    """``V . grad(pe)`` [Pa/s] at the interior interfaces: ``u``, ``v``
    (6, n, n, km) cell-centre geographic winds averaged to the interior
    interfaces, ``pe`` (6, n, n, km-1) the interface pressures,
    ``tangent_gradient_sixface`` for the gradient, projected on the local
    east/north unit vectors (6, n, n, 3)."""
    g = tangent_gradient_sixface(pe, X, radius)
    ui = 0.5 * (u[..., 1:] + u[..., :-1])
    vi = 0.5 * (v[..., 1:] + v[..., :-1])
    return (ui * np.einsum("fijkx,fijx->fijk", g, e_east)
            + vi * np.einsum("fijkx,fijx->fijk", g, e_north))


def build_duo_dcmip_transport(n: int, *, wind_geo, init_latlon, sigma_coord,
                              n_tracers: int, dt: float,
                              n_sub: int | None = None,
                              period_s: float | None = None,
                              courant_target: float = 0.8,
                              ps_init_fn=None):
    """DCMIP 2012 1-1 / 1-2 / 1-3 prescribed-wind transport through FV3 duo's OWN
    tracer machinery, composed from the public duo phases (adapter only,
    the shallow-water case-1 precedent; user decision Q5-A).  Per outer
    step of ``dt`` (winds frozen at the step MID-time ``t + dt/2``):

    1. DCMIP ``(u, v)`` at the D-edge midpoints of every level (the
       ``sigma_coord`` reference levels, as every other grid's row),
       projected onto the edge tangents (:class:`_DgridProjection`);
    2. duo entry exchange (``exchange_state_halos_3d``: delp/pt, D winds);
    3. ``csw_phase_3d`` at ``dt2 = 0`` -- the pure D->C interpolation,
       no half-step advance, no ``p_grad_c`` (frozen winds);
    4. ``n_sub`` x ``dsw_transport_phase_3d(dt/n_sub, flux_cap)`` (its
       own post-``p_grad_c`` exchanges, ``d_sw1`` accumulating the mass /
       Courant capacitors, barrier 1, ``d_sw2`` delp update), with the
       post-``d_sw`` delp exchange between sub-steps, as the acoustic loop;
    5. ``tracer_2d_1l_sixface`` on the accumulated capacitors with
       ``dp1`` = delp before step 4 (fv_dynamics.F90:472-478 / :534);
    6. Lagrangian -> Eulerian: the Lagrangian interfaces ``pe1 = ptop +
       cumsum(delp)``, ``ps = pe1[km]`` (p_s EVOLVES with the discrete
       column mass divergence, user decision Q7-b, never reset), target
       ``pe2 = ak + bk ps``; tracers remapped with the certified FV3
       kernel (:func:`legoesm.grids.vertical.remap_tracers_fv3_pressure`,
       map1_q2 kord_tr 9), delp := diff(pe2).

    VERTICAL MOTION (user decision Q6-a): implicit -- the horizontal mass
    flux divergence moves the Lagrangian layers and the remap returns
    them to the eta levels, FV3's own mechanism.  The DCMIP omega /
    sigma_dot is NOT used; ``effective_omega()`` compares, for the first
    and the last ACCEPTED step, two estimates of the implicit vertical
    motion with the analytic ``p0 * sigma_dot`` (DCMIP's omega at the
    eta reference interfaces, cell centres, step mid-time): (i) the
    fixed-cell Lagrangian interface tendency ``(pe_lag - pe_eul) / dt``
    (``*_fixed_cell`` keys) and (ii) the MATERIAL omega, (i) plus the
    advection of interface pressure along the eta surface,
    ``V . grad(pe_eul)`` (= bk V . grad(p_s); unsuffixed keys;
    ``rms_interface_advection`` its size).

    ``n_sub`` (None = automatic): d_sw sub-steps per outer step.  The
    automatic choice is ``ceil(Cmax / courant_target)`` with ``Cmax`` the
    largest ``|V| dt / min(dx, dy)`` of the prescribed winds over the cell
    centres at every 4th step mid-time of ``[0, period_s]``; the realised
    per-sub-step C-grid Courant number is checked every step and >= 1 is
    refused; in automatic mode an over-Courant step is recomputed with
    more sub-steps before its remap (see ``step``).

    pt rides as a uniform 300 K passenger (the DCMIP isothermal
    atmosphere); w = 0; hydrostatic deck; ``nq`` passive tracers.

    ``ps_init_fn`` (1-3): ``(lon, lat) [rad] -> p_s [Pa]`` at the cell
    centres, the initial surface pressure over the terrain (DCMIP 1-3:
    ``p0 exp(-z_s/H)``); the initial ``delp = diff(ak + bk p_s)`` and the
    analytic reference of ``effective_omega`` is the coordinate-crossing
    air-mass flux ``p_s * sigma_dot`` (the DCMIP 1-3 ``sigma_dot`` is
    returned divided by ``p_s``).  ``None`` (1-1 / 1-2): ``p_s = p0``
    everywhere, reference ``p0 * sigma_dot``.  A hybrid ``sigma_coord``
    (the Kent et al. 2014, App. B coordinate, ``create_dcmip13_hybrid``)
    REQUIRES ``ps_init_fn`` (and a sigma one refuses it); its ``A p_ref``
    / ``B`` must equal the model's ``ak`` / ``bk`` (the same fixed-lid
    table).
    """
    import jax
    import jax.numpy as jnp
    if not jax.config.jax_enable_x64:
        # the duo lane is fp64-only; refuse rather than flip the process'
        # precision (the matrix sets it from JAX_ENABLE_X64 at import)
        raise RuntimeError("duo DCMIP transport needs JAX_ENABLE_X64=1")
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel, duo_sw_deck)
    from legoesm.core.fv3_acoustic_3d import exchange_state_halos_3d
    from legoesm.core.fv3_cgrid_phase_3d import csw_phase_3d
    from legoesm.core.fv3_dsw_phase_3d import dsw_transport_phase_3d
    from legoesm.core.fv3_tracer2d import (
        alloc_flux_capacitors, check_nsplt_schedule, tracer_2d_1l_sixface)
    from legoesm.grids.factory import create_fv3_duo_grid
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_RADIUS_M, analytic_swcore_state)
    from legoesm.grids.vertical import (
        dcmip2012_transport_eta_table, remap_tracers_fv3_pressure)

    km = DUO_DCMIP_TRANSPORT_KM
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be finite and positive, got {dt!r}")
    if int(sigma_coord.n_levels) != km:
        raise ValueError(f"DCMIP duo transport runs L{km}; the row's sigma "
                         f"coordinate has {sigma_coord.n_levels} levels")
    grid = create_fv3_duo_grid(n)
    cfg = FV3DuoConfig(km=km, eta=DUO_DCMIP_TRANSPORT_ETA, hydrostatic=True)
    model = FV3DuoDynamicsModel(grid, cfg)
    ak, bk, ptop = (np.asarray(model.ak), np.asarray(model.bk),
                    float(model.ptop))
    # NO L32 leak: the model's table must be the DCMIP L30 one, and its
    # p0 interfaces the row's sigma levels (to the latter's float32)
    ak_ref, bk_ref, ptop_ref = dcmip2012_transport_eta_table(km)
    if not (ak.shape == (km + 1,) and np.array_equal(ak, ak_ref)
            and np.array_equal(bk, bk_ref) and ptop == ptop_ref):
        raise ValueError("duo DCMIP transport: the model's eta table is not "
                         "the DCMIP 2012 L30 table")
    p0 = 1.0e5
    if not np.allclose((ak + bk * p0) / p0,
                       np.asarray(sigma_coord.sigma_half, np.float64),
                       rtol=0, atol=1e-6):
        raise ValueError("duo DCMIP transport: eta interfaces at p0 are not "
                         "the row's DCMIP sigma levels")
    if hasattr(sigma_coord, "B_half") and not (
            np.allclose(ak, np.asarray(sigma_coord.A_half, np.float64)
                        * float(sigma_coord.p_ref), rtol=0, atol=1e-6 * p0)
            and np.allclose(bk, np.asarray(sigma_coord.B_half, np.float64),
                            rtol=0, atol=1e-6)):
        # a hybrid row: its A p_ref / B (its own reference pressure) must
        # be the model's ak / bk
        raise ValueError("duo DCMIP transport: the row's hybrid A p_ref / B are "
                         "not the model's ak / bk")
    deck = duo_sw_deck(nord=cfg.nord, d4_bg=cfg.d4_bg, vtdm4=cfg.vtdm4,
                       sponge_del2_top_layers=cfg.sponge_del2_top_layers,
                       sponge_del2_top_factor=cfg.sponge_del2_top_factor,
                       sponge_d2_top=cfg.sponge_d2_top)
    ctx = grid.ctx_jax
    ctx_np = grid.ctx_np
    ng = grid.ng
    sl = _compute_window(ctx_np)

    # template bundle (shapes/halo conventions of the lane), then the
    # DCMIP state: ps = p0 everywhere (or ps_init_fn on the compute cells;
    # the halos are filled by the step's entry exchange), isothermal pt,
    # w = 0, no wind yet
    tmpl = model.dcmip16_initial_state(do_pert=False)
    st0 = dict(tmpl["state"])
    m_a = st0["delp"].shape[1]
    delp0 = np.broadcast_to(np.diff(ak + bk * p0), (6, m_a, m_a, km)).copy()
    st0 = {"delp": jnp.asarray(delp0),
           "pt": jnp.full_like(st0["pt"], 300.0),
           "w": jnp.zeros_like(st0["w"]),
           "u": jnp.zeros_like(st0["u"]),
           "v": jnp.zeros_like(st0["v"])}

    lon_deg, lat_deg, area = duo_cell_geometry(ctx_np)
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    ps_ref = None
    if ps_init_fn is None and hasattr(sigma_coord, "B_half"):
        # the hybrid (1-3) row's tracers sit on terrain-dependent levels:
        # a flat p_s = p0 start would put the air elsewhere
        raise ValueError("duo DCMIP transport: the hybrid coordinate needs "
                         "ps_init_fn (the terrain surface pressure)")
    if ps_init_fn is not None and not hasattr(sigma_coord, "B_half"):
        # terrain p_s on a sigma row: the duo's fixed-lid pressures would
        # differ from the row's p = sigma p_s levels away from p0
        raise ValueError("duo DCMIP transport: ps_init_fn needs the hybrid "
                         "coordinate whose A p0 / B are the model's ak / bk")
    if ps_init_fn is not None:
        ps_ref = np.asarray(ps_init_fn(jnp.asarray(lon), jnp.asarray(lat)),
                            np.float64)
        if ps_ref.shape != lon.shape or not np.all(ps_ref > ptop):
            raise ValueError("duo DCMIP transport: ps_init_fn must give a "
                             "(6, n, n) surface pressure above the lid")
        delp0[:, sl, sl, :] = np.diff(ak + bk * ps_ref[..., None], axis=-1)
        st0["delp"] = jnp.asarray(delp0)
        if not np.all(delp0[:, sl, sl, :] > 0):
            raise ValueError("duo DCMIP transport: non-positive initial delp")
    cells = SimpleNamespace(lon2d=jnp.asarray(lon.reshape(6 * n, n)),
                            lat2d=jnp.asarray(lat.reshape(6 * n, n)),
                            n_lat=6 * n, n_lon=n)
    q_ic = np.asarray(init_latlon(cells, sigma_coord).tracers.data,
                      dtype=np.float64).reshape(6, n, n, km, n_tracers)
    q6 = np.zeros((6, n_tracers, m_a, m_a, km))
    q6[:, :, sl, sl, :] = np.moveaxis(q_ic, -1, 1)

    proj = _DgridProjection(ctx_np)

    auto_sub = n_sub is None
    if n_sub is None:
        if period_s is None:
            raise ValueError("n_sub=None (automatic) needs period_s")
        dmin = min(float(np.min(np.asarray(g[k])[sl, sl]))
                   for g in ctx_np["gs6"] for k in ("dx", "dy")
                   if np.asarray(g[k]).ndim == 2)
        jl, jt = jnp.asarray(lon), jnp.asarray(lat)
        vmax = 0.0
        for k in range(0, max(1, int(round(period_s / dt))), 4):
            u_, v_, _ = wind_geo((k + 0.5) * dt, jl, jt, sigma_coord)
            vmax = max(vmax, float(jnp.max(jnp.hypot(u_, v_))))
        n_sub = max(1, int(np.ceil(vmax * dt / dmin / courant_target)))
    n_sub = int(n_sub)
    if n_sub < 1:
        raise ValueError(f"n_sub must be >= 1, got {n_sub}")

    def geo_at(t):
        def fn(lo, la):
            u, v, _ = wind_geo(t, lo, la, sigma_coord)
            return (jnp.asarray(u, jnp.float64), jnp.asarray(v, jnp.float64))
        return fn

    # consistency gate: the precomputed projection == the certified
    # analytic_swcore_state recipe (level 0, t = dt/2, compute edges)
    u_chk, v_chk = proj(geo_at(0.5 * dt), st0["u"], st0["v"])
    for t, gs in enumerate(ctx_np["gs6"][:2]):
        def wf(ll):
            u, v, _ = wind_geo(0.5 * dt, jnp.asarray(ll[..., 0]),
                               jnp.asarray(ll[..., 1]), sigma_coord)
            return np.asarray(u)[..., 0], np.asarray(v)[..., 0]
        d = analytic_swcore_state(gs, wind_fn=wf)
        if not (np.allclose(np.asarray(u_chk)[t, sl, ng:ng + n + 1, 0],
                            d["u"][sl, ng:ng + n + 1], rtol=1e-10,
                            atol=1e-10)
                and np.allclose(np.asarray(v_chk)[t, ng:ng + n + 1, sl, 0],
                                d["v"][ng:ng + n + 1, sl], rtol=1e-10,
                                atol=1e-10)):
            raise ValueError("duo DCMIP transport: D-grid projection differs "
                             "from analytic_swcore_state")

    def _dynamics(delp, t, dt_, n_sub):
        """Steps 1-4: the d_sw mass transport; returns the Lagrangian delp,
        the capacitors, dp1 and the realised per-sub-step Courant number.
        The tracer kernel runs only for an ACCEPTED attempt (``step``)."""
        u6, v6 = proj(geo_at(t + 0.5 * dt_), st0["u"], st0["v"])
        st = {**st0, "delp": delp, "u": u6, "v": v6}
        st = exchange_state_halos_3d(ctx, st, km, scalars=True, winds=True)
        csw = csw_phase_3d(ctx, st, dt2=0.0, km=km, nord=2, batched=True,
                           hydrostatic=True, remap_follows=True)
        dp1 = st["delp"]
        fc = alloc_flux_capacitors(n, ng, km)
        # csw (uc/vc/divg_d, the only csw fields d_sw reads) depends on the
        # frozen D winds only, so one c_sw serves every sub-step
        cmax = jnp.asarray(0.0)
        bad = jnp.asarray(False)
        for _ in range(n_sub):
            dsw = dsw_transport_phase_3d(ctx, st, csw, dt_ / n_sub, km,
                                         cfg=deck, hydrostatic=True,
                                         remap_follows=True, flux_cap=fc,
                                         batched=True)
            # this sub-step's own C-grid Courant numbers (increments)
            cmax = jnp.maximum(cmax, jnp.maximum(
                jnp.max(jnp.abs(dsw["cx"] - fc["cx"])),
                jnp.max(jnp.abs(dsw["cy"] - fc["cy"]))))
            fc = {k: dsw[k] for k in fc}
            # every intermediate d_sw2 layer mass must be finite and > 0
            dpc = dsw["delp"][:, sl, sl, :]
            bad = bad | jnp.any(~jnp.isfinite(dpc)) | jnp.any(~(dpc > 0))
            # post-d_sw delp/pt exchange before the next sub-step, as the
            # acoustic loop (pt: the uniform passenger d_sw2 carries)
            st = exchange_state_halos_3d(
                ctx, {**st, "delp": dsw["delp"], "pt": dsw["pt"]},
                km, scalars=True, winds=False)
        return st["delp"], fc, dp1, cmax, bad

    def _tracer(q, dp1, fc, dt_):
        """Step 5: tracer_2d_1L on the accepted capacitors."""
        return tracer_2d_1l_sixface(ctx, q, dp1, fc, km=km, nq=n_tracers,
                                    hord_tr=deck.hord_tr, dt=dt_,
                                    batched=True)

    def _remap(delp, q, delp_lag, q_lag):
        """Lagrangian -> Eulerian (step 6): p_s from the Lagrangian column,
        tracers by the FV3 kernel, delp onto the eta levels."""
        dpl = delp_lag[:, sl, sl, :]
        pe1 = jnp.concatenate(
            [jnp.full(dpl.shape[:-1] + (1,), ptop),
             ptop + jnp.cumsum(dpl, axis=-1)], axis=-1)
        ps = pe1[..., -1]
        pe2 = ak + bk * ps[..., None]
        pe2 = pe2.at[..., 0].set(ptop).at[..., -1].set(ps)
        qc = jnp.moveaxis(q_lag[:, :, sl, sl, :], 1, -1)
        q2 = remap_tracers_fv3_pressure(qc, pe1, pe2, kord=cfg.kord_tr)
        q_new = q.at[:, :, sl, sl, :].set(jnp.moveaxis(q2, -1, 1))
        delp_new = delp.at[:, sl, sl, :].set(jnp.diff(pe2, axis=-1))
        return delp_new, q_new

    def _omega_pair(delp, delp_lag, t, dt_):
        """Implicit vertical motion of an accepted step: the Lagrangian
        interface-pressure change at FIXED cells, (pe_lag - pe_eul)/dt, at
        the interior interfaces [Pa/s], and the analytic p0 * sigma_dot
        (which the transport never uses) at the cell centres, step
        mid-time.  Also returns the cell-centre winds and the Eulerian
        (start-of-step) interfaces pe_eul, from whose tangent-plane gradient
        ``_omega_stats`` adds the advection of interface pressure along the
        eta surface, V . grad(pe_eul) = bk V . grad(p_s), to form the
        material omega.  (The step-midpoint Lagrangian surface was tried
        and REJECTED by measurement: its slope is (dt/2) grad(omega), an
        O(dt) artefact of the finite step -- C36 1-2: material rel err
        1.08e-2 with it vs 2.64e-3 with pe_eul, fixed-cell 2.65e-3.)"""
        pe_e = ptop + jnp.cumsum(delp[:, sl, sl, :], axis=-1)[..., :-1]
        pe_l = ptop + jnp.cumsum(delp_lag[:, sl, sl, :], axis=-1)[..., :-1]
        u, v, sd = wind_geo(t + 0.5 * dt_, jnp.asarray(lon),
                            jnp.asarray(lat), sigma_coord)
        return ((pe_l - pe_e) / dt_,
                (p0 * jnp.asarray(sd, jnp.float64)[..., 1:-1]
                 if ps_ref is None else
                 jnp.asarray(ps_ref)[..., None]
                 * jnp.asarray(sd, jnp.float64)[..., 1:-1]),
                jnp.asarray(u, jnp.float64), jnp.asarray(v, jnp.float64),
                pe_e)

    dyn_jit = jax.jit(_dynamics, static_argnums=(2, 3))
    tracer_jit = jax.jit(_tracer)
    remap_jit = jax.jit(_remap)
    omega_jit = jax.jit(_omega_pair, static_argnums=(3,))
    cmax_seen = [0.0]
    ns_cur = [n_sub]
    omega_rec = {}
    last_phase = {}

    # unit position vectors of the cell centres and east/north unit vectors
    _X = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                   np.sin(lat)], -1)
    _ee = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], -1)
    _en = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon),
                    np.cos(lat)], -1)

    def _omega_stats(eff, ana, u, v, pe_ref, t_sample):
        eff, ana = np.asarray(eff), np.asarray(ana)
        u, v = np.asarray(u), np.asarray(v)
        adv = interface_pressure_advection(u, v, np.asarray(pe_ref), _X,
                                           _ee, _en, FV3_RADIUS_M)
        mat = eff + adv
        w = area[..., None] * np.ones_like(eff)
        rms = lambda x: float(np.sqrt(np.sum(w * x * x) / np.sum(w)))  # noqa: E731
        ra = rms(ana)
        out = {"t_sample": t_sample, "rms_analytic": ra,
               "rms_effective": rms(mat), "rms_effective_fixed_cell": rms(eff),
               "rms_interface_advection": rms(adv)}
        for key, x in (("", mat), ("_fixed_cell", eff)):
            out["abs_rms_err" + key] = rms(x - ana)
            out["rel_rms_err" + key] = (rms(x - ana) / ra if ra > 0
                                        else float("nan"))
            out["corr" + key] = (
                float(np.corrcoef(x.ravel(), ana.ravel())[0, 1])
                if ra > 0 and np.std(x) > 0 else float("nan"))
        return out

    def step(carry, dt_):
        """One outer step.  The horizontal phase (steps 1-5) runs first and
        its realised per-sub-step C-grid Courant number is checked on the
        host BEFORE the remap: < 1 is required (d_sw's delp transport is a
        Courant < 1 scheme).  With an explicit ``n_sub`` a violation is
        refused; in automatic mode the horizontal phase is RECOMPUTED from
        the same input with ``max(n_sub + 1, ceil(n_sub * C / 0.8))``
        sub-steps (at most 3 times, then refused), never clipped, and that
        count is kept for later steps."""
        (delp0_, q0_), t = carry
        dt_ = float(dt_)
        if not (np.isfinite(dt_) and dt_ > 0):
            raise ValueError(f"duo DCMIP transport: dt must be finite and "
                             f"positive, got {dt_!r}")
        ns_try = ns_cur[0]
        for _attempt in range(4):
            delp_lag, fc, dp1, cmax, bad = dyn_jit(delp0_, t, dt_, ns_try)
            cmax = float(cmax)
            ok = (not bool(bad)) and np.isfinite(cmax) and cmax < 1.0
            if ok:
                break
            # rejected attempt (over-Courant, or a non-finite / non-positive
            # intermediate d_sw2 layer mass -- its later sub-steps are then
            # garbage and the whole attempt is discarded)
            if not auto_sub or _attempt == 3:
                raise RuntimeError(
                    f"duo DCMIP transport: rejected step at t={float(t):g} s "
                    f"(dt={dt_:g} s, n_sub={ns_try}): sub-step Courant "
                    f"{cmax:.3f}, invalid intermediate delp={bool(bad)}; "
                    f"refusing")
            c_eff = cmax if (np.isfinite(cmax) and cmax >= 1.0) else 2.0
            ns_try = max(ns_try + 1, int(np.ceil(ns_try * c_eff / 0.8)))
        tr = tracer_jit(q0_, dp1, fc, dt_)
        check_nsplt_schedule({"nsplt": tr["nsplt"],
                              "nsplt_exceeded": tr["nsplt_exceeded"]})
        delp, q = remap_jit(delp0_, q0_, delp_lag, tr["q"])
        jax.block_until_ready((delp, q))         # remap refusals surface here
        if not (bool(jnp.all(jnp.isfinite(q[:, :, sl, sl, :])))
                and bool(jnp.all(jnp.isfinite(delp[:, sl, sl, :])))):
            raise RuntimeError(f"duo DCMIP transport: non-finite tracer or "
                               f"delp after the step at t={float(t):g} s; "
                               f"refusing")
        # diagnostics staged FIRST; step metadata published only for an
        # ACCEPTED step whose diagnostics also succeeded
        pair = jax.block_until_ready(omega_jit(delp0_, delp_lag, t, dt_))
        if not all(bool(np.all(np.isfinite(np.asarray(x)))) for x in pair):
            raise RuntimeError(f"duo DCMIP transport: non-finite omega "
                               f"diagnostic at t={float(t):g} s; refusing")
        ts = float(t) + 0.5 * dt_
        first = (_omega_stats(*pair, ts) if "first" not in omega_rec
                 else None)
        ns_cur[0] = ns_try
        last_phase.clear()
        last_phase.update(delp_in=delp0_, dp1=dp1, delp_lag=delp_lag, tr=tr)
        cmax_seen[0] = max(cmax_seen[0], cmax)
        if first is not None:
            omega_rec["first"] = first
        omega_rec["last_pair"] = (pair, ts)
        return ((delp, q), t + dt_)

    def effective_omega():
        """Implicit vs analytic omega stats of the first and the last
        ACCEPTED steps (``{}`` before any step)."""
        if "first" not in omega_rec:
            return {}
        pair, ts = omega_rec["last_pair"]
        return {"first": omega_rec["first"], "last": _omega_stats(*pair, ts)}

    def tracers(carry):
        (_, q), _ = carry
        return np.moveaxis(np.asarray(q)[:, :, sl, sl, :], 1, -1)

    def layer_mass(carry):
        """delp [Pa] on the compute cells, (6, n, n, km)."""
        (delp, _), _ = carry
        return np.asarray(delp)[:, sl, sl, :]

    def ps(carry):
        (delp, _), _ = carry
        return ptop + np.sum(np.asarray(delp)[:, sl, sl, :], axis=-1)

    def air_mass(carry):
        """sum(delp * area) over the modelled column (below the lid)."""
        return float(np.sum(layer_mass(carry).sum(-1) * area))

    note = (f"fv3_duo transport adapter (csw dt2=0 -> n_sub (initial "
            f"{n_sub}, accepted count reported per row) x dsw_transport"
            f"(flux_cap) -> tracer_2d_1l -> map1_q2 kord_tr={cfg.kord_tr}), "
            f"eta={DUO_DCMIP_TRANSPORT_ETA} km={km} ptop={ptop:.1f} Pa, "
            f"hord_tr={deck.hord_tr} hord_dp={deck.hord_dp}; vertical motion "
            f"implicit (Lagrangian delp + remap, DCMIP omega unused); p_s "
            f"evolves (no reset)")
    carry0 = ((jnp.asarray(delp0), jnp.asarray(q6)), 0.0)
    return SimpleNamespace(
        grid=grid, model=model, carry=carry0, step=step, tracers=tracers,
        layer_mass=layer_mass, ps=ps, effective_omega=effective_omega,
        air_mass=air_mass, n_sub_fn=lambda: ns_cur[0],
        last_phase=last_phase,
        lon_deg=lon_deg, lat_deg=lat_deg, area=area, radius=FV3_RADIUS_M,
        courant_max=lambda: cmax_seen[0],
        ak=ak, bk=bk, ptop=ptop, q_ic=q_ic, note=note, ps_ref=ps_ref)
