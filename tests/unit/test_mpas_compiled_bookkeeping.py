"""The MPAS eager loop's compiled per-step bookkeeping equals the eager code it
replaced (PR #2592), to round-off in float64.

Each compiled group is called exactly as the loop calls it (``jax.jit`` of the
module-level core with the run-static values bound by ``functools.partial``)
and compared with the PREVIOUS eager expressions, copied here from the driver
at 847d291863 (plus main's later handback LW correction and ice/open-water
split, which the merge carried into the cores).  The two flux-carrying groups
(surface-flux accumulator, land-forcing window) also close their budget:
interval mean x samples = sum of the per-step inputs.
"""
from __future__ import annotations

import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.driver import model_driver as md  # noqa: E402

_RTOL = 1e-13
_N = 23


def _rng(seed=0):
    return np.random.default_rng(seed)


def _close(a, b, rtol=_RTOL):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=0.0)


# --------------------------------------------------------------- anchors ----

def _anchors_eager_847d(sst, sic, skin, f_land, lapse_z, lapse_K_m):
    from legoesm.forcing.surface_utils import (
        blend_surface_temperature, land_lapse_adjusted_surface_temperature)
    sst = jnp.asarray(sst).reshape(-1)
    sic = jnp.asarray(sic).reshape(-1)
    ts_o = blend_surface_temperature(sst, sic, skin).reshape(-1)
    ts = ts_o
    if lapse_z is not None:
        ts = land_lapse_adjusted_surface_temperature(
            ts, f_land.astype(ts.dtype), lapse_z.astype(ts.dtype), lapse_K_m)
    return ts, ts_o


@pytest.mark.parametrize("lapse", [False, True])
@pytest.mark.parametrize("array_skin", [False, True])
def test_surface_anchors_compiled_equal_eager(lapse, array_skin):
    r = _rng(1)
    sst = 271.0 + 30.0 * r.random(_N)
    sic = r.random(_N)
    skin = jnp.asarray(250.0 + 20.0 * r.random(_N)) if array_skin else 263.15
    f_land = jnp.asarray(r.random(_N), dtype=jnp.float32)
    lapse_z = jnp.asarray(3000.0 * r.random(_N), dtype=jnp.float32) if lapse else None
    core = jax.jit(functools.partial(md.mpas_surface_anchors, f_land=f_land,
                                     lapse_z=lapse_z, lapse_K_m=-0.0065))
    got = core(sst, sic, skin)
    ref = _anchors_eager_847d(sst, sic, skin, f_land, lapse_z, -0.0065)
    for g, e in zip(got, ref):
        _close(g, e)
    # the skin is an argument: a changed skin moves the anchors (no stale capture)
    if array_skin:
        moved = core(sst, sic, skin + 5.0)[1]
        assert np.any(np.abs(np.asarray(moved) - np.asarray(got[1])) > 1e-6)


# ---------------------------------------------------------- sea-ice albedo --

@pytest.mark.parametrize("scheme", ["constant", "maykut_untersteiner"])
def test_sea_ice_albedo_refresh_compiled_equal_eager(scheme):
    from legoesm.forcing.surface_utils import (
        blend_surface_property, blended_surface_albedo)
    from legoesm.ice.shortwave import sea_ice_albedo
    r = _rng(2)
    skin = jnp.asarray(255.0 + 20.0 * r.random(_N))
    sic, alb_o = jnp.asarray(r.random(_N)), jnp.asarray(0.06 + 0.02 * r.random(_N))
    f_land, alb_l = jnp.asarray(r.random(_N)), jnp.asarray(0.1 + 0.3 * r.random(_N))
    kw = dict(scheme=scheme, albedo_ice=0.6, h_ice_m=2.0, melt_width_K=1.0)
    core = jax.jit(functools.partial(md.mpas_sea_ice_albedo_refresh, **kw))
    got = core(skin, sic, alb_o, f_land, alb_l)
    # 847d291863 eager: _ice_albedo_now() then the two blends
    a_ice = sea_ice_albedo(scheme, 0.6, T_skin=skin, h_ice_m=2.0, melt_width_K=1.0)
    ref = (a_ice, blend_surface_property(sic, a_ice, alb_o),
           blended_surface_albedo(sic, f_land, a_ice, alb_o, alb_l))
    for g, e in zip(got, ref):
        _close(g, e)


# ------------------------------------------------------ land-forcing marshal -

def _sd(sw, lw, precip, frozen):
    from legoesm.core.state import MPAS_SFC_DIAG_BASE_KEYS, MPAS_SFC_DIAG_EXTRA_KEYS
    slot = len(MPAS_SFC_DIAG_BASE_KEYS) + MPAS_SFC_DIAG_EXTRA_KEYS.index("precip_frozen")
    sd = [None] * (slot + 1)
    ns = types.SimpleNamespace
    sd[2] = None if precip is None else ns(data=precip)
    sd[8], sd[9] = ns(data=sw), ns(data=lw)
    sd[slot] = None if frozen is None else ns(data=frozen)
    return sd


def _marshal_eager_847d(sd, t_air, q_air, p_s, u_c, v_c, doy, sod, co2, lw_corr,
                        sigma, cos_zen_fn):
    from legoesm.core.coupling_fields import AtmToSurface, lowest_level_height
    from legoesm.thermo import moist_air_density
    sw_down = jnp.asarray(sd[8].data).reshape(-1)
    lw_down = jnp.asarray(sd[9].data).reshape(-1)
    if lw_corr is not None:
        lw_down = lw_down + lw_corr
    precip = (jnp.asarray(sd[2].data).reshape(-1)
              if sd[2] is not None else jnp.zeros_like(sw_down))
    precip_snow = md.land_snowfall_from_sfc_diag(sd, precip)
    T_air = t_air
    if q_air is None:
        q_air = jnp.zeros_like(T_air)
    p_s = jnp.asarray(p_s).reshape(-1)
    p_full = sigma.pressure_at_full(p_s)
    p_lowest = p_full[..., -1]
    z_lowest = lowest_level_height(T_air, sigma.pressure_at_half(p_s), p_full)
    return AtmToSurface(
        z_lowest=z_lowest, sw_down=sw_down, lw_down=lw_down, precip_total=precip,
        precip_snow=precip_snow, T_lowest=T_air, q_lowest=q_air,
        u_lowest=u_c, v_lowest=v_c, p_lowest=p_lowest, p_surface=p_s,
        rho_lowest=moist_air_density(T_air, p_lowest, q_air),
        cos_zenith=cos_zen_fn(jnp.asarray(doy, dtype=jnp.float64),
                              jnp.asarray(sod, dtype=jnp.float64) / 3600.0),
        co2_ppmv=jnp.full_like(T_air, co2),
        has_radiation=jnp.ones_like(T_air), has_precipitation=jnp.ones_like(T_air))


@pytest.mark.parametrize("edge_winds", [False, True])
@pytest.mark.parametrize("with_corr", [False, True])
@pytest.mark.parametrize("precip_kind", ["frozen", "none"])
def test_land_forcing_marshal_compiled_equal_eager(edge_winds, with_corr, precip_kind):
    from legoesm.grids.vertical import make_cam6_l32_levels
    r = _rng(3)
    sigma = make_cam6_l32_levels()
    cos_zen_fn = jax.jit(lambda doy, h: jnp.maximum(
        jnp.cos(2 * jnp.pi * (h / 24.0 + doy / 365.0) + jnp.arange(_N) * 0.3), 0.0))
    recon_fn = jax.jit(lambda u: (0.7 * u, -0.2 * u))
    sw, lw = 400.0 * r.random(_N), 250.0 + 150.0 * r.random(_N)
    precip = 1e-4 * r.random(_N) if precip_kind == "frozen" else None
    frozen = 0.5 * precip if precip_kind == "frozen" else None
    t_air, q_air = 260.0 + 40.0 * r.random(_N), 0.02 * r.random(_N)
    p_s = 9e4 + 1.2e4 * r.random(_N)
    u_edge = 10.0 * r.standard_normal(_N)
    u_c, v_c = (None, None) if edge_winds else (u_edge, 0.3 * u_edge)
    lw_corr = jnp.asarray(5.0 * r.standard_normal(_N)) if with_corr else None
    core = jax.jit(functools.partial(md.mpas_land_forcing, sigma=sigma,
                                     cos_zen_fn=cos_zen_fn, recon_fn=recon_fn),
                   static_argnames=("precip_published",))
    got = core(sw, lw, precip, frozen, t_air, q_air, p_s,
               u_edge if edge_winds else None, u_c, v_c,
               jnp.asarray(200.0), jnp.asarray(43200.0), 410.0, lw_corr,
               precip_published=precip is not None)
    ref_uc, ref_vc = recon_fn(u_edge) if edge_winds else (u_c, v_c)
    ref = _marshal_eager_847d(_sd(sw, lw, precip, frozen), t_air, q_air, p_s,
                              ref_uc, ref_vc, 200.0, 43200.0, 410.0, lw_corr,
                              sigma, cos_zen_fn)
    for name in ref._fields:
        g, e = getattr(got, name), getattr(ref, name)
        if e is None:
            assert g is None, name
        else:
            _close(g, e)


# ------------------------------------------------------------- ice skin ----

@pytest.mark.parametrize("fluxes", [False, True])
def test_ice_skin_advance_compiled_equal_eager(fluxes):
    from legoesm.forcing.surface_utils import advance_ice_skin
    r = _rng(4)
    skin = jnp.asarray(250.0 + 20.0 * r.random(_N))
    sw, lw = jnp.asarray(100.0 * r.random(_N)), jnp.asarray(-60.0 * r.random(_N))
    sh = jnp.asarray(20.0 * r.standard_normal(_N)) if fluxes else None
    lh = jnp.asarray(10.0 * r.random(_N)) if fluxes else None
    sic = jnp.asarray(r.random(_N))
    jit = jax.jit(advance_ice_skin, static_argnames=("dt_s", "h_ice_m", "melt_cap_width_K"))
    kw = dict(dt_s=900.0, h_ice_m=2.0, melt_cap_width_K=0.5)
    _close(jit(skin, sw, lw, sh, lh, sic, **kw), advance_ice_skin(skin, sw, lw, sh, lh, sic, **kw))


# ------------------------------------------------------- post-step filter --

class _F(tuple):
    """Minimal field: ``.data`` and ``.replace(data=)`` like the model's."""

    def __new__(cls, data):
        return tuple.__new__(cls, (data,))

    @property
    def data(self):
        return self[0]

    def replace(self, data):
        return _F(data)


jax.tree_util.register_pytree_node(_F, lambda f: ((f.data,), None), lambda _, c: _F(c[0]))


class _State(tuple):
    _fields = ("u", "p_s", "tracers")

    def __new__(cls, u, p_s, tracers):
        return tuple.__new__(cls, (u, p_s, tracers))

    u = property(lambda s: s[0])
    p_s = property(lambda s: s[1])
    tracers = property(lambda s: s[2])

    def _replace(self, **kw):
        d = dict(zip(self._fields, self))
        d.update(kw)
        return _State(**d)


jax.tree_util.register_pytree_node(_State, lambda s: (tuple(s), None), lambda _, c: _State(*c))


def _poststep_eager_847d(state, sponge_decay, smooth):
    if sponge_decay is not None:
        state = state._replace(u=state.u.replace(data=state.u.data * sponge_decay))
    if smooth is not None:
        trc = state.tracers
        new = dict(trc)
        new["q_v"] = trc["q_v"].replace(data=smooth(trc["q_v"].data, state.p_s.data))
        state = state._replace(tracers=new)
    return state


@pytest.mark.parametrize("sponge,smooth_on", [(False, False), (True, False),
                                              (False, True), (True, True)])
def test_poststep_compiled_equal_eager(sponge, smooth_on):
    r = _rng(5)
    state = _State(_F(jnp.asarray(r.standard_normal((_N, 4)))),
                   _F(jnp.asarray(9e4 + 1e4 * r.random(_N))),
                   {"q_v": _F(jnp.asarray(0.01 * r.random((_N, 4))))})
    decay = jnp.asarray(np.exp(-r.random(4))) if sponge else None
    smooth = (jax.jit(lambda q, ps: q - 0.1 * (q - q.mean(axis=0)) * (ps[:, None] / 1e5))
              if smooth_on else None)
    core = jax.jit(functools.partial(md.mpas_poststep, sponge_decay=decay,
                                     qv_smooth_fn=smooth))
    got, ref = core(state), _poststep_eager_847d(state, decay, smooth)
    for g, e in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(ref)):
        _close(g, e)


# ------------------------------------------- surface-flux accumulator -------

def _accum_eager_847d(steps, slots):
    """The 847d291863 per-slot eager loop of _MPASSfcFluxAccum.add."""
    sums, n = {}, {}
    for sd in steps:
        for i in slots:
            if len(sd) > i and sd[i] is not None:
                x = jnp.asarray(sd[i].data, dtype=jnp.float64)
                if i in sums:
                    sums[i] = sums[i] + x
                    n[i] += 1
                else:
                    sums[i] = x
                    n[i] = 1
    return sums, n


def test_flux_accumulator_bitwise_and_closes_its_budget():
    r = _rng(6)
    slots = md._MPASSfcFluxAccum.SLOTS
    ns = types.SimpleNamespace
    steps = []
    for k in range(7):
        sd = [None] * (max(slots) + 1)
        for i in slots:
            # slot 5 appears only from the fourth step: the seed path mid-window
            if i == 5 and k < 3:
                continue
            sd[i] = ns(data=jnp.asarray(100.0 * r.standard_normal(_N), dtype=jnp.float32))
        steps.append(sd)
    acc = md._MPASSfcFluxAccum()
    for sd in steps:
        acc.add(sd)
    ref, n_ref = _accum_eager_847d(steps, slots)
    assert acc._n == n_ref
    for i in slots:
        np.testing.assert_array_equal(np.asarray(acc._sum[i]), np.asarray(ref[i]))
        # budget: interval mean x samples = sum of the per-step inputs
        inputs = [np.asarray(sd[i].data, dtype=np.float64) for sd in steps
                  if sd[i] is not None]
        np.testing.assert_allclose(acc.mean(i) * n_ref[i], np.sum(inputs, axis=0),
                                   rtol=1e-12, atol=1e-9)


# ------------------------------------------------- land-forcing window -----

def _a2s(r):
    from legoesm.core.coupling_fields import AtmToSurface
    f = lambda: jnp.asarray(r.random(_N))  # noqa: E731
    return AtmToSurface(z_lowest=f(), sw_down=400.0 * f(), lw_down=300.0 * f(),
                        precip_total=1e-4 * f(), precip_snow=1e-5 * f(), T_lowest=f(),
                        q_lowest=f(), u_lowest=f(), v_lowest=f(), p_lowest=f(),
                        p_surface=f(), rho_lowest=f(), cos_zenith=f(), co2_ppmv=f(),
                        has_radiation=f(), has_precipitation=f())


def test_land_window_compiled_equal_eager_and_closes_its_budget():
    r = _rng(7)
    samples = [_a2s(r) for _ in range(6)]
    # compiled path, as the loop drives it
    s, cz, w = md._land_window_init(samples[0])
    for a in samples[1:]:
        s, cz, w = md._land_window_add(s, cz, w, a)
    # 847d291863 eager path
    s_e = None
    for k, a in enumerate(samples):
        s_e = a if s_e is None else jax.tree_util.tree_map(jnp.add, s_e, a)
        cw = a.sw_down + 1.0
        if k == 0:
            cz_e, w_e = a.cos_zenith * cw, cw
        else:
            cz_e, w_e = cz_e + a.cos_zenith * cw, w_e + cw
    for g, e in zip(jax.tree_util.tree_leaves(s), jax.tree_util.tree_leaves(s_e)):
        _close(g, e)
    _close(cz, cz_e)
    _close(w, w_e)
    # budget: the window sum of every forcing (precip mass, radiant energy) is
    # the sum of its samples
    for name in ("precip_total", "precip_snow", "sw_down", "lw_down"):
        _close(getattr(s, name), np.sum([np.asarray(getattr(a, name)) for a in samples], axis=0),
               rtol=1e-12)
