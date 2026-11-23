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
        SWConfig, advance_duo_outer_step, build_jax_duo_stepper_context,
        make_full_acoustic_step_sixface_jit, states_to_jax)
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
        return advance_duo_outer_step(jctx, state, dt, n_split, d_ext=0.0,
                                      sw_cfg=sw_cfg, step_fn=jstep)

    lon_deg, lat_deg, area = duo_cell_geometry(ctx)
    sl = _compute_window(ctx)
    geo = _GeoWinds(ctx)

    def h(state):
        return np.asarray(state["delp"], dtype=np.float64)[:, sl, sl] / FV3_GRAV

    def winds(state):
        return geo(state["u"], state["v"])

    note = (f"fv3_duo JAX km=1 stepper, SW_CFG_CASE8 hord8 nord2 "
            f"d4_bg={deck['d4_bg']} d_ext=0, dt_atmos={deck['dt_atmos']:g}s "
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


def rotated_jw_bundle(model, *, perturbed: bool, alpha: float) -> dict:
    """DCMIP 2008 4-1/4-2 rotated baroclinic-wave bundle on the duo grid.

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

    coord = _cam6_l32_coord(model, "rotated JW duo IC")
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
            phis_fn=_mountain_2_0_0_phis(h_0), wind_fn=None, omega=None,
            label=f"DCMIP 2012 2-0-0 rest state: constant lapse rate 6.5 K/km on native L32 levels (reference 12 km top not reproduced), h_0={h_0:g} m")
    if case == "held_suarez_topo":
        from legoesm.atmosphere.idealized.held_suarez_topo import (
            DEFAULT_H_0, held_suarez_topo_init as init)
        h_0 = float(kw.get("h_0", DEFAULT_H_0))
        return SimpleNamespace(
            builder=lambda g, c: init(g, c, **kw),
            phis_fn=_mountain_2_0_0_phis(h_0), wind_fn=None, omega=None,
            label=(f"held_suarez_topo_init (h_0={h_0:g} m, "
                   f"seed={kw.get('seed', 'default')})"))
    if case in ("gravity_wave_3_1", "inertio_gravity_3_2",
                "mountain_rossby_5_0"):
        from tests.test_cases.dcmip2008._shared import U0_DEFAULT
        if kw:
            raise ValueError(f"{case}: no IC kwargs, got {sorted(kw)}")
        if case == "gravity_wave_3_1":
            from tests.test_cases.dcmip2008.gravity_wave_3_1 import (
                gravity_wave_init as init)
            phis_fn, omega = None, 0.0   # DCMIP 2008 3-1: non-rotating
        elif case == "inertio_gravity_3_2":
            from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
                inertio_gravity_init as init)
            phis_fn, omega = None, None
        else:
            from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
            from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
                _phis, mountain_rossby_init as init)

            def phis_fn(lon, lat):
                return np.array(_phis(lon, lat, FV3_RADIUS_M), dtype=np.float64)
            omega = None
        return SimpleNamespace(
            builder=init, phis_fn=phis_fn,
            wind_fn=_solid_body_winds(U0_DEFAULT), omega=omega,
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
        FV3_KAPPA, FV3_RADIUS_M, analytic_swcore_state)

    coord = _cam6_l32_coord(model, "duo shared IC")
    ctx = model.grid.ctx_np
    n, ng, km = ctx["n"], ctx["ng"], model.config.km
    sl = _compute_window(ctx)
    gs6 = ctx["gs6"]
    lon = np.stack([np.asarray(g["agrid_lon"])[sl, sl] for g in gs6])
    lat = np.stack([np.asarray(g["agrid_lat"])[sl, sl] for g in gs6])
    cells = SimpleNamespace(n=n, lon=jnp.asarray(lon), lat=jnp.asarray(lat),
                            radius=FV3_RADIUS_M, angle=jnp.zeros_like(lat))
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
                    ic_kwargs: dict | None = None):
    """FV3DuoDynamicsModel (production deck defaults) on C``n``.

    ``dt`` follows the driver lane: the deck dt 1920 s clamped by the
    driver's generic ``cfl_check_and_adjust`` (``ModelDriver``).
    ``rotated_alpha`` (not None): DCMIP 2008 rotated JW -- the planet's
    rotation axis tilted by alpha (``create_fv3_duo_grid(rotation_alpha)``)
    and :func:`rotated_jw_bundle` as the IC.
    ``shared_case`` (not None): the row's IC is its shared cube builder
    (:func:`duo_shared_ic_spec`, ``ic_kwargs`` passed through), with the
    recipe's terrain (``phis_fn``, unfiltered like every other grid's) and
    rotation rate on the grid; combines with ``held_suarez`` for HS-topo.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.core.cfl import cfl_check_and_adjust
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
    elif ic_kwargs:
        raise ValueError("ic_kwargs needs shared_case")
    grid = create_fv3_duo_grid(
        n, rotation_alpha=0.0 if rotated_alpha is None else rotated_alpha,
        **grid_kw)
    cfg = FV3DuoConfig(km=km, eta=eta, hydrostatic=True)
    model = FV3DuoDynamicsModel(grid, cfg)
    if spec is not None:
        bundle = shared_ic_bundle(model, spec)
    elif held_suarez:
        bundle = held_suarez_rest_bundle(model)
    elif rotated_alpha is None:
        bundle = model.dcmip16_initial_state(do_pert=perturbed)
    else:
        bundle = rotated_jw_bundle(model, perturbed=perturbed,
                                   alpha=rotated_alpha)
    dt = float(cfl_check_and_adjust(1920.0, n, model_type="primitive_eq",
                                    grid_type="cubed_sphere"))

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

        def _hs(state, press, dt_):
            return apply_held_suarez_step_sixface_jax(
                state, press, tab, amat6, lat6, wv6, dt=dt_, n=n, ng=ng,
                km=km, strat=True)
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
            f"nord={cfg.nord} d4_bg={cfg.d4_bg} sponge={cfg.sponge_del2_top_layers}"
            f"L, n_split={cfg.n_split}, dt={dt:g}s (deck 1920 s + driver "
            f"CFL clamp); " + (
                f"shared cube IC {spec.label} on the duo cells + CAM6 L32"
                + ("" if spec.phis_fn is None else
                   ", grid terrain = the IC's phis (unfiltered)")
                + ("" if spec.omega is None else
                   f", omega={spec.omega:g} (non-rotating)")
                if spec is not None else
                "Held-Suarez IC: isothermal 300 K at rest, ps = p_ref, "
                "1 K lowest-level perturbation (seed 42), as every grid"
                if held_suarez else
                f"DCMIP16 baroclinic-wave IC (do_pert={perturbed})"
                if rotated_alpha is None else
                f"rotated JW IC (jablonowski_rotated builder on CAM6 L32, "
                f"alpha={rotated_alpha:.4f}, perturbed={perturbed}), "
                f"rotated-planet Coriolis"))
    if held_suarez:
        note += " + certified FV3 HS (strat=T) after each step"
    return SimpleNamespace(
        model=model, grid=grid, state=bundle, dt=dt, step=step, ps=ps,
        temperature=temperature, winds=winds, lon_deg=lon_deg,
        lat_deg=lat_deg, area=area, km=km, level_values=level_values,
        note=note)
