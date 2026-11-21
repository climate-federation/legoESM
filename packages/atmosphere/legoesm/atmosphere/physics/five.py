"""FIVE: fine vertical layers under a host column (Yamaguchi et al. 2017,
doi:10.1002/2016MS000815; Lee et al. 2021, doi:10.1029/2020MS002240).

The host physics runs unchanged on the host levels.  A fine column, whose
layers split every host layer below a pressure into equal-dp sublayers, is
carried in every column.  Where FIVE is active, radiation, turbulence and
microphysics run on the fine column, deep convection stays on the host and
is replayed to the fine grid, and the host takes the dp-weighted layer mean
of every fine tendency.  Grids exchange tendencies, never states:

* fine to host: the dp-weighted layer mean ``A``;
* host to fine, a water species: a negative layer increment is taken from
  each sublayer in proportion to its content, a positive one is spread with
  the Sheng and Zwiers (1998) shape (rescaled to the exact layer mean,
  constant over a layer where any sublayer would change sign); T, u, v use
  the shape rule on the net.

At each call the host minus the layer mean of the fine state is exchanged to
the fine grid first, so the fine state is synchronous with the host before
the physics runs.  Selection modes: ``always`` (every column) and ``forced``
(on and off by a fixed period).

KNOWN DIFFERENCE (RULE 5): the three switches on traced state are smooth,
each with a FiveConfig width (0 = the reference's hard switch): the
shape-versus-constant choice of :func:`spread` (C1 ramps of the sign test and
of the shape mean), the sign split of a water increment (C1 positive part),
and the sink fraction's cap at -1 and empty-layer test (C1 max, logistic).
Per-case gaps at the defaults are stored in tests/atmosphere/test_five_core.py.

Not ported from the reference fork: isotope passengers and the ``route_b``
selection (running-mean EIS/omega regime machine with hysteresis, hold and a
capacity whose overflow stops the run) -- its switches on traced state have no
smooth form that keeps their meaning.

Column use only: reached through ``make_physics(five=FiveConfig(...))``; no
driver switch or deck selects it.  The build refuses every lane it cannot run
on, including the production MPAS deck (its physics cadence cache).

Not built here (docs/85 13.24): the fine-grid vertical advection, a gather of
active columns (host and fine physics both run in every column and the result
is selected), ZM's gross sink and source (the net increment is split by
sign).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.smooth import max_c1, positive_part_c1, sigmoid_switch, smoothstep_cubic
from legoesm.timestepping.tridiagonal import thomas_solve

__physics_contract__ = {
    "summary": "FIVE fine-layer exchange and column selection around the host physics",
    "inputs": {"host state": "T K, u v m/s, tracers kg/kg", "p_s": "Pa"},
    "outputs": {"host tendencies": "layer means of the fine tendencies where active"},
    "sign_convention": "pressure increases downward; index 0 is the model top",
    # water, dp-weighted enthalpy and momentum, to round-off
    "conserves": ["moisture", "tracer", "energy", "momentum"],
    "differentiable": True,
    "reference": "Yamaguchi et al. 2017 doi:10.1002/2016MS000815; Lee et al. 2021 "
                 "doi:10.1029/2020MS002240; Sheng and Zwiers 1998 doi:10.1007/s003820050244",
    "idealized_test": "tests/atmosphere/test_five_core.py",
}

__param_spec__ = {
    "FiveConfig": {
        "scheme_key": "atm.five.FiveConfig",
        "excluded": {
            "dz_target_m": "numerics: target sublayer depth of the static fine grid",
            "p_fine_top_pa": "domain definition [Pa]: top of the refined part (CAM L32 742 hPa interface)",
            "forced_period_s": "experiment protocol: on/off period of the forced selection",
            "spread_width": "numerics: RULE 5 smooth-switch width, value voted 2026-11-21",
            "split_width_kg_kg": "numerics: RULE 5 smooth-switch width, value voted 2026-11-21",
            "cap_width": "numerics: RULE 5 smooth-switch width, value voted 2026-11-21",
            "empty_width_kg_kg": "numerics: RULE 5 smooth-switch width, value voted 2026-11-21",
        },
        "params": {},
    },
}

# --- standard atmosphere for the static sublayer count (ICAO 1993) ---
_STD_T0_K = 288.15
_STD_LAPSE_K_PER_M = 6.5e-3
_HOURS = 3600.0

# CLUBB moment fields on thermodynamic levels (clubb.CLUBBMomentState); the
# rest are on momentum levels.  Packed ascending (index 0 at the surface).
_ZT_MOMENTS = frozenset({"rtm", "thlm", "um", "vm", "wp3"})
# Variance and covariance moments that gain the between-sublayer spread on
# exit: moment -> (scalar a, scalar b) of the fine state.
_SPREAD_MOMENTS = {"thlp2": ("thl", "thl"), "rtp2": ("rt", "rt"),
                   "rtpthlp": ("rt", "thl"), "up2": ("u", "u"), "vp2": ("v", "v")}

FIVE_MODES = ("always", "forced")


class FiveConfig(NamedTuple):
    """FIVE switch, selection and RULE 5 widths (docs/85 13.18)."""
    enabled: bool = False
    mode: str = "always"
    # fine grid
    dz_target_m: float = 20.0
    p_fine_top_pa: float = 74000.0      # CAM L32 interface at 742.4 hPa is the top of the fine part
    # forced mode: on for one period, off for the next
    forced_period_s: float = 6.0 * _HOURS
    # RULE 5 widths (0 = the reference's hard switch; values voted 2026-11-21):
    # spread's sign and shape-mean ramps (dimensionless, relative to the layer
    # value), the water sign split [kg/kg per exchange], the sink-fraction cap
    # (dimensionless) and the empty-layer test [kg/kg]
    spread_width: float = 0.01
    split_width_kg_kg: float = 1.0e-9
    cap_width: float = 0.01
    empty_width_kg_kg: float = 1.0e-9


def validate_five_config(cfg: FiveConfig) -> None:
    if cfg.mode not in FIVE_MODES:
        raise ValueError(f"FiveConfig.mode={cfg.mode!r}; choose from {FIVE_MODES}")
    if not (cfg.dz_target_m > 0.0 and cfg.forced_period_s > 0.0):
        raise ValueError("FiveConfig: dz_target_m, forced_period_s must be > 0")
    for name in ("spread_width", "split_width_kg_kg", "cap_width", "empty_width_kg_kg"):
        w = getattr(cfg, name)
        if not (isinstance(w, (int, float)) and not isinstance(w, bool)
                and math.isfinite(w) and w >= 0.0):
            raise ValueError(f"FiveConfig.{name}={w!r} must be a finite number >= 0")


# ======================================================================
# grid
# ======================================================================
class FiveGrid(NamedTuple):
    coord: object          # fine HybridSigmaPressureCoordinate
    owner: np.ndarray      # (nf,) host layer of each fine layer
    n_sub: np.ndarray      # (nH,) sublayers per host layer
    onehot: jax.Array      # (nf, nH) float64, 1 where owner matches
    host_iface: np.ndarray  # (nH+1,) fine interface index of each host interface
    sub_index: np.ndarray  # (nH, max n_sub) fine index of each sublayer, nf = padding


def _std_height(p, p0):
    """Height of pressure p in the ICAO standard atmosphere [m]."""
    e = constants.R_d * _STD_LAPSE_K_PER_M / constants.g
    return _STD_T0_K / _STD_LAPSE_K_PER_M * (1.0 - (p / p0) ** e)


def build_five_grid(host_coord, cfg: FiveConfig) -> FiveGrid:
    """Nested fine coordinate: host layer K whose top interface lies at or below
    ``p_fine_top_pa`` (at the reference surface pressure) is split into
    ceil(dz_K / dz_target) equal-dp sublayers (A and B interpolated linearly,
    so the split holds at every surface pressure); other layers are copied."""
    from legoesm.grids.vertical import create_hybrid_coordinate
    A = np.asarray(host_coord.A_half, dtype=np.float64)
    B = np.asarray(host_coord.B_half, dtype=np.float64)
    p_ref = float(host_coord.p_ref)
    ph = A * p_ref + B * p_ref
    nH = A.size - 1
    n_sub = np.ones(nH, dtype=int)
    for K in range(nH):
        if ph[K] >= cfg.p_fine_top_pa:
            dz = _std_height(ph[K], p_ref) - _std_height(ph[K + 1], p_ref)
            n_sub[K] = max(1, int(math.ceil(dz / cfg.dz_target_m)))
    Af, Bf, owner, host_iface = [], [], [], [0]
    for K in range(nH):
        for j in range(n_sub[K]):
            s = j / n_sub[K]
            Af.append(A[K] + s * (A[K + 1] - A[K]))
            Bf.append(B[K] + s * (B[K + 1] - B[K]))
            owner.append(K)
        host_iface.append(len(owner))
    Af.append(A[-1])
    Bf.append(B[-1])
    dtype = host_coord.A_half.dtype
    # host interfaces are copied bitwise into the fine coefficients
    Af = np.asarray(Af)
    Bf = np.asarray(Bf)
    Af[host_iface] = np.asarray(host_coord.A_half)
    Bf[host_iface] = np.asarray(host_coord.B_half)
    nf = len(owner)
    coord = create_hybrid_coordinate(nf, Af, Bf, p_ref=host_coord.p_ref, dtype=dtype)
    owner = np.asarray(owner)
    onehot = np.zeros((nf, nH))
    onehot[np.arange(nf), owner] = 1.0
    sub_index = np.full((nH, int(np.max(n_sub))), nf)
    for K in range(nH):
        idx = np.flatnonzero(owner == K)
        sub_index[K, :idx.size] = idx
    return FiveGrid(coord=coord, owner=owner, n_sub=n_sub,
                    onehot=jnp.asarray(onehot, dtype=jnp.float64),
                    host_iface=np.asarray(host_iface), sub_index=sub_index)


# ======================================================================
# exchange operators, batched over columns: host (ncol, nH), fine (ncol, nf)
# ======================================================================
def to_host(x_f, dp_f, G: FiveGrid):
    """dp-weighted layer mean of a fine field."""
    return ((x_f * dp_f) @ G.onehot) / (dp_f @ G.onehot)


def expand(x_H, G: FiveGrid):
    """Host layer value in each of its sublayers."""
    return x_H[..., G.owner]


def _sz98_column(tau, ph, pf):
    """Sheng and Zwiers (1998) piecewise-linear fine values of one column.

    ``tau`` (nH,) layer values, ``ph`` (nH+1,) host interfaces, ``pf`` (nf+1,)
    fine interfaces.  The node values t solve M t = tau so that the integral
    of the linear interpolant through (Pm_K, t_K) over each host layer is
    tau_K (Y17 appendix B); the fine values are its fine-layer means."""
    dp = ph[1:] - ph[:-1]
    Pm = 0.5 * (ph[1:] + ph[:-1])
    dPm = Pm[1:] - Pm[:-1]
    a = jnp.concatenate([jnp.zeros(1, tau.dtype), 0.25 * dp[1:] / dPm])
    b = jnp.concatenate([0.25 * dp[:-1] / dPm, jnp.zeros(1, tau.dtype)])
    t = thomas_solve(0.5 * a, 1.0 - 0.5 * a - 0.5 * b, 0.5 * b, tau)
    xn = jnp.concatenate([ph[:1], Pm, ph[-1:]])
    yn = jnp.concatenate([t[:1], t, t[-1:]])
    seg = xn[1:] - xn[:-1]
    Fn = jnp.concatenate([jnp.zeros(1, tau.dtype), jnp.cumsum(seg * 0.5 * (yn[1:] + yn[:-1]))])
    j = jnp.clip(jnp.searchsorted(xn, pf, side="right") - 1, 0, seg.size - 1)
    s = pf - xn[j]
    F = Fn[j] + s * (yn[j] + 0.5 * s * (yn[j + 1] - yn[j]) / seg[j])
    return (F[1:] - F[:-1]) / (pf[1:] - pf[:-1])


def _ramp(x, width, strict):
    """C1 ramp in x: 1 above 0 (``strict``: above ``width``), 0 below
    ``-width`` (``strict``: below 0); width 0 is ``x >= 0`` (``x > 0``)."""
    if width == 0:
        return ((x > 0.0) if strict else (x >= 0.0)).astype(x.dtype)
    return smoothstep_cubic(x / width + (0.0 if strict else 1.0))


def _layer_prod(r_f, G: FiveGrid):
    """Product over each host layer's sublayers, (ncol, nf) -> (ncol, nH)."""
    pad = jnp.concatenate([r_f, jnp.ones_like(r_f[:, :1])], axis=1)
    return jnp.prod(pad[:, G.sub_index], axis=-1)


def spread(tau_H, ph_H, pf_f, dp_f, G: FiveGrid, width=0.0):
    """Host layer values -> fine values with layer means exactly tau_H.

    The Sheng and Zwiers shape, rescaled per layer, blended per layer with the
    constant tau_K by w = prod_k R(s_k / tau_K) R(m / tau_K) (both preserve
    the layer mean): R the C1 ramps of :func:`_ramp` of the given width
    (dimensionless), s_k the sublayer shape values, m the shape's layer mean.
    Width 0 is the reference: constant over a layer where any sublayer has the
    opposite sign to tau_K or the shape's mean has not tau_K's sign."""
    shape = jax.vmap(_sz98_column)(tau_H, ph_H, pf_f)
    m = to_host(shape, dp_f, G)
    nz = tau_H != 0.0
    inv = jnp.where(nz, 1.0 / jnp.where(nz, tau_H, 1.0), 0.0)
    w = (_layer_prod(_ramp(shape * expand(inv, G), width, False), G)
         * _ramp(m * inv, width, True))
    w = jnp.where(nz, w, 0.0)
    m_safe = jnp.where(w > 0.0, m, 1.0)
    return (expand(w * tau_H / m_safe, G) * shape + expand((1.0 - w) * tau_H, G))


class BulkSplit(NamedTuple):
    sink: jax.Array      # (ncol, nH) <= 0
    source: jax.Array    # (ncol, nH) >= 0
    content: jax.Array   # (ncol, nH) layer mean content before the exchange
    src_f: jax.Array     # (ncol, nf) fine source, layer means = source


def _sharp(width):
    return math.inf if width == 0 else 1.0 / width


def exchange_bulk(inc_H, x_f, ph_H, pf_f, dp_f, G: FiveGrid, cfg=None, gross=None):
    """Host increment of a water species -> fine increment (13.18).

    A sink removes the fraction f = S/X of every sublayer's content (X the
    layer mean); the light water decides every switch (docs/90 7.11): an
    empty layer (X <= 0) gives nothing, and f is capped at -1.  What a switch
    withholds stays in host minus A(fine) and the next sync exchanges it.
    ``gross``: optional ``(sink, source)`` of the host process; without it the
    net is split by sign.  ``cfg`` (FiveConfig) gives the RULE 5 widths of the
    sign split, the empty-layer test and the cap; None = the hard switches."""
    cfg = FiveConfig(spread_width=0.0, split_width_kg_kg=0.0, cap_width=0.0,
                     empty_width_kg_kg=0.0) if cfg is None else cfg
    if gross is None:
        S = -positive_part_c1(-inc_H, _sharp(cfg.split_width_kg_kg))
        P = inc_H - S
    else:
        S, P = gross
    X = to_host(x_f, dp_f, G)
    pos = X > 0.0
    has = 1.0 - sigmoid_switch(-X, _sharp(cfg.empty_width_kg_kg))
    f = has * jnp.where(pos, S / jnp.where(pos, X, 1.0), 0.0)
    f_eff = max_c1(f, -1.0, _sharp(cfg.cap_width))
    src_f = spread(P, ph_H, pf_f, dp_f, G, cfg.spread_width)
    return expand(f_eff, G) * x_f + src_f, BulkSplit(S, P, X, src_f)


def _moment_names():
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBMomentState
    return CLUBBMomentState._fields


def _interp_cols(xq, xp, fp):
    return jax.vmap(jnp.interp)(xq, xp, fp)


def moments_to_fine(m_H, ph_H, pf_f, G: FiveGrid):
    """Entry: fine moments from host moments (packed ascending, (ncol, 15, nzm))."""
    nH = G.n_sub.size
    nf = G.owner.size
    out = []
    for i, name in enumerate(_moment_names()):
        col = m_H[:, i, :]
        if name in _ZT_MOMENTS:
            td = col[:, :nH][:, ::-1]                       # top-down layers
            f = expand(td, G)[:, ::-1]
            f = jnp.concatenate([f, jnp.zeros_like(f[:, :1])], axis=1)
        else:
            td = col[:, ::-1]                                # top-down interfaces
            f = _interp_cols(pf_f, ph_H, td)[:, ::-1]
        out.append(f[:, None, :])
    res = jnp.concatenate(out, axis=1)
    assert res.shape[2] == nf + 1
    return res


def moments_to_host(m_f, fine_scalars, dp_f, G: FiveGrid):
    """Exit: host moments from fine moments; variances gain the between-sublayer
    variance (covariance) of the fine scalars over each host layer.

    ``fine_scalars``: top-down (ncol, nf) arrays ``thl``, ``rt``, ``u``, ``v``.
    Momentum-level host values: mean of the two adjacent host-layer means of
    the fine values (fine-layer centres), the top and surface copied."""
    nH = G.n_sub.size
    nf = G.owner.size
    out = []
    for i, name in enumerate(_moment_names()):
        col = m_f[:, i, :]
        if name in _ZT_MOMENTS:
            td = col[:, :nf][:, ::-1]
            h = to_host(td, dp_f, G)[:, ::-1]
            h = jnp.concatenate([h, jnp.zeros_like(h[:, :1])], axis=1)
        else:
            td = col[:, ::-1]                                # (ncol, nf+1)
            lay = to_host(0.5 * (td[:, 1:] + td[:, :-1]), dp_f, G)
            if name in _SPREAD_MOMENTS:
                a, b = _SPREAD_MOMENTS[name]
                xa, xb = fine_scalars[a], fine_scalars[b]
                da = xa - expand(to_host(xa, dp_f, G), G)
                db = xb - expand(to_host(xb, dp_f, G), G)
                lay = lay + to_host(da * db, dp_f, G)
            inner = 0.5 * (lay[:, 1:] + lay[:, :-1])
            h_td = jnp.concatenate([td[:, :1], inner, td[:, -1:]], axis=1)
            h = h_td[:, ::-1]
        out.append(h[:, None, :])
    res = jnp.concatenate(out, axis=1)
    assert res.shape[2] == nH + 1
    return res


def clubb_scalars(T, q_v, q_c, p_full):
    """thl and rt of a top-down column as CLUBB defines them (liquid only)."""
    exner = (p_full / constants.p_ref) ** (constants.R_d / constants.c_pd)
    thl = T / exner - constants.L_v * q_c / (constants.c_pd * exner)
    return thl, q_v + q_c


# ======================================================================
# route B diagnostics and selection
# ======================================================================
def init_selection(ncol, cfg: FiveConfig):
    return {"active": jnp.full((ncol,), True), "clock_s": jnp.zeros((), jnp.float64)}


def update_selection(sel, dt, cfg: FiveConfig):
    """One physics step of the (state-independent) selection.  Returns (new
    selection, entered, left)."""
    act = sel["active"]
    clock = sel["clock_s"] + dt
    if cfg.mode == "always":
        new = jnp.ones_like(act)
    else:
        new = jnp.full_like(act, jnp.floor(clock / cfg.forced_period_s) % 2 == 0)
    return {"active": new, "clock_s": clock}, new & ~act, act & ~new


# ======================================================================
# the wrapped physics
# ======================================================================
def sync_increment(fine, host_x, ph_H, pf_f, dp_f, G, cfg=None):
    """Exchange host - A(fine) to the fine grid (dicts of top-down arrays).

    ``fine``: {name: (ncol, nf)}, ``host_x``: {name: (ncol, nH)} (T in K, u v
    in m/s, water in kg/kg), ``ph_H`` (ncol, nH+1) and ``pf_f`` (ncol, nf+1)
    interface pressures [Pa], ``dp_f`` (ncol, nf) [Pa]; ``cfg`` the widths."""
    out = dict(fine)
    w = 0.0 if cfg is None else cfg.spread_width
    for k in fine:
        inc = host_x[k] - to_host(fine[k], dp_f, G)
        if k in ("T", "u", "v"):
            out[k] = fine[k] + spread(inc, ph_H, pf_f, dp_f, G, w)
        else:
            out[k] = fine[k] + exchange_bulk(inc, fine[k], ph_H, pf_f, dp_f, G, cfg)[0]
    return out




def _state_dict(state):
    d = {"T": state.T.data, "u": state.u.data}
    if state.v is not None:
        d["v"] = state.v.data
    for k, f in (state.tracers or {}).items():
        d[k] = f.data
    return d


def _with_fields(state, d):
    kw = dict(T=state.T.replace(data=d["T"]), u=state.u.replace(data=d["u"]))
    if state.v is not None:
        kw["v"] = state.v.replace(data=d["v"])
    if state.tracers:
        kw["tracers"] = {k: f.replace(data=d[k]) for k, f in state.tracers.items()}
    return state._replace(**kw)


def _map_carry(x, nH, nf, G, ph_H, pf_f):
    """A host physics-carry leaf onto the fine grid (constant per layer,
    interfaces interpolated in pressure); other shapes pass through."""
    if not hasattr(x, "shape") or x.ndim != 2:
        return x
    if x.shape[1] == nH:
        return expand(x, G)
    if x.shape[1] == nH + 1:
        return _interp_cols(pf_f, ph_H, x)
    return x


_NO_MAP = ("five_carry", "clubb_moments", "held_physics", "prng_key", "col_index")


def _host_carries_to_fine(ps_f, ps_h, mask, G, ph_H, pf_f):
    """Fine carry fields from the host where ``mask`` (ncol,), field by field:
    (ncol, nH) and (ncol, nH+1) host arrays mapped onto a fine array of the
    same rank; everything else kept.  ``mask`` None replaces everywhere."""
    nH, nf = G.n_sub.size, G.owner.size
    kw = {}
    for name in ps_f._fields:
        if name in _NO_MAP:
            continue
        f, h = getattr(ps_f, name), getattr(ps_h, name)
        if not (hasattr(f, "ndim") and hasattr(h, "ndim") and f.ndim == 2 and h.ndim == 2):
            continue
        m = _map_carry(h, nH, nf, G, ph_H, pf_f)
        if m.shape != f.shape or m is h:
            continue
        kw[name] = m if mask is None else jnp.where(mask[:, None], m, f)
    return ps_f._replace(**kw)


def _select_carry(new, old, active):
    """Field by field: ``new`` where active, ``old`` elsewhere (same shapes)."""
    kw = {}
    for name in new._fields:
        if name == "five_carry":
            continue
        a, b = getattr(new, name), getattr(old, name)
        if (hasattr(a, "shape") and hasattr(b, "shape") and a.shape == b.shape
                and a.ndim >= 1 and a.shape[0] == active.shape[0]):
            kw[name] = jnp.where(active.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
        else:
            kw[name] = a
    return new._replace(**kw)


def make_five_physics(host_fn, fine_fn, zm_fn, G: FiveGrid, cfg: FiveConfig, config,
                      dt: float):
    """Wrap ``host_fn`` with FIVE (see the module docstring).

    ``fine_fn`` is the same physics built with ``convection_override`` set to
    the replay returned by :func:`make_conv_replay`; ``zm_fn`` the host
    convection module (None without convection)."""
    from legoesm.atmosphere.physics.physics_state import update_physics_state
    nH = G.n_sub.size
    nf = G.owner.size
    has_moments = config.turbulence.scheme == "clubb"

    def physics_fn(state, grid, sigma_coord, phys_state=None, forcing=None):
        if phys_state is None:
            raise ValueError("FIVE needs a PhysicsState")
        if phys_state.five_carry is None:
            # first call: seed here, where the column state is at hand (one
            # retrace of the caller at the None -> dict transition)
            phys_state = _seed(state, sigma_coord, phys_state, G, cfg, config)
        forcing = dict(forcing or {})
        carry = phys_state.five_carry
        ps_H_in = phys_state._replace(five_carry=None)
        p_s = state.p_s.data
        ph_H = sigma_coord.pressure_at_half(p_s)
        pf_f = G.coord.pressure_at_half(p_s)
        pff = G.coord.pressure_at_full(p_s)
        dp_f = G.coord.layer_thickness_dp(p_s)
        host_x = _state_dict(state)
        fine = sync_increment(carry["fine"], host_x, ph_H, pf_f, dp_f, G, cfg)
        sel, entered, left = update_selection(carry["sel"], dt, cfg)
        active = sel["active"]

        # fine physics carry: entering columns take the host carries
        ps_f = _host_carries_to_fine(carry["ps"], ps_H_in, entered, G, ph_H, pf_f)
        if has_moments and ps_H_in.clubb_moments.shape[1:] == (15, nH + 1):
            mH = ps_H_in.clubb_moments
            ps_f = ps_f._replace(clubb_moments=jnp.where(
                entered[:, None, None], moments_to_fine(mH, ph_H, pf_f, G),
                ps_f.clubb_moments))

        # host physics, every column
        t_H, ps_H_out = host_fn(state, grid, sigma_coord, ps_H_in, forcing)

        # fine physics, every column (selected below)
        state_f = _with_fields(state, fine)
        frc_f = dict(forcing)
        if zm_fn is not None:
            frc_f[CONV_REPLAY_KEY] = (zm_fn(state, grid, sigma_coord, phys_state=ps_H_in,
                                            forcing=forcing), fine, ph_H, pf_f, dp_f)
        t_f, ps_f_out = fine_fn(state_f, grid, G.coord, ps_f, frc_f)

        # merge: active columns take the layer means of the fine tendencies
        def merge(h, f):
            if not (hasattr(h, "shape") and hasattr(f, "shape")):
                return h
            if h.ndim == 2 and f.ndim == 2 and h.shape[1] == nH and f.shape[1] == nf:
                return jnp.where(active[:, None], to_host(f, dp_f, G), h)
            if h.shape == f.shape and h.ndim >= 1 and h.shape[0] == active.shape[0]:
                return jnp.where(active.reshape((-1,) + (1,) * (h.ndim - 1)), f, h)
            return h
        t_out = jax.tree_util.tree_map(merge, t_H, t_f)

        # fine state advance where active; host carries from the fine column
        fd = _tend_dict(t_f)
        fine_new = {k: jnp.where(active[:, None], v + dt * fd[k], v) if k in fd else v
                    for k, v in fine.items()}
        ps_f_new = _select_carry(ps_f_out, ps_f, active)
        upd = {}
        if ps_H_out.pblh is not None and ps_f_new.pblh is not None:
            upd["pblh"] = jnp.where(active, ps_f_new.pblh, ps_H_out.pblh)
        if ps_H_out.cloud_fraction is not None and ps_f_new.cloud_fraction is not None \
                and ps_f_new.cloud_fraction.shape[-1] == nf:
            upd["cloud_fraction"] = jnp.where(
                active[:, None], to_host(ps_f_new.cloud_fraction, dp_f, G),
                ps_H_out.cloud_fraction)
        if has_moments and ps_H_out.clubb_moments.shape[1:] == (15, nH + 1):
            thl, rt = clubb_scalars(fine["T"], fine["q_v"], fine.get("q_c", 0.0 * fine["T"]), pff)
            sc = {"thl": thl, "rt": rt, "u": fine["u"], "v": fine.get("v", 0.0 * fine["u"])}
            upd["clubb_moments"] = jnp.where(
                left[:, None, None], moments_to_host(ps_f.clubb_moments, sc, dp_f, G),
                ps_H_out.clubb_moments)
        upd["five_carry"] = {"fine": fine_new, "ps": ps_f_new, "sel": sel}
        return t_out, update_physics_state(ps_H_out, upd)

    physics_fn._requires_phys_state = True
    physics_fn._five = (G, cfg)
    for attr in ("_wants_forcing",):
        if hasattr(host_fn, attr):
            setattr(physics_fn, attr, getattr(host_fn, attr))
    physics_fn._five_physics_config = config
    return physics_fn


def _tend_dict(t):
    d = {"T": t.dT_dt.data, "u": t.du_dt.data}
    if t.dv_dt is not None:
        d["v"] = t.dv_dt.data
    for k, f in (t.tracer_tendencies or {}).items():
        d[k] = f.data
    return d


def five_carry_seed(physics_fn, state, sigma_coord, phys_state):
    """Initial FIVE carry: the fine state constant per layer from the host,
    the fine physics carry from a fresh one with the host carries mapped."""
    G, cfg = physics_fn._five
    return _seed(state, sigma_coord, phys_state, G, cfg, physics_fn._five_physics_config)


def _seed(state, sigma_coord, phys_state, G, cfg, config):
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    ncol = state.T.data.shape[0]
    nH = G.n_sub.size
    nf = G.owner.size
    p_s = state.p_s.data
    ph_H = sigma_coord.pressure_at_half(p_s)
    pf_f = G.coord.pressure_at_half(p_s)
    fine = {k: expand(v, G) for k, v in _state_dict(state).items()}
    ps_f = init_physics_state(ncol=ncol, nlev=nf, physics_config=config,
                              dtype=state.T.data.dtype)
    ps_f = _host_carries_to_fine(ps_f, phys_state, None, G, ph_H, pf_f)
    m0 = ps_f.clubb_moments
    if config.turbulence.scheme == "clubb" and phys_state.clubb_moments.shape[1:] == (15, nH + 1):
        m0 = moments_to_fine(phys_state.clubb_moments, ph_H, pf_f, G)
    ps_f = ps_f._replace(clubb_moments=m0)
    carry = {"fine": fine, "ps": ps_f, "sel": init_selection(ncol, cfg)}
    return phys_state._replace(five_carry=carry)


# ======================================================================
# deep convection replayed on the fine grid
# ======================================================================
CONV_REPLAY_KEY = "five_conv_replay"


def replay_tendencies(td, fine, ph_H, pf_f, dp_f, G: FiveGrid, dt, cfg=None):
    """Host tendencies {name: (ncol, nH) per s} -> fine tendencies (ncol, nf).

    The water exchange has a cap (a sink fraction of the content), which is not
    linear: it takes the increment over ``dt`` and the result is divided by
    ``dt``.  T, u, v and the detrained condensate go through the linear spread
    as rates."""
    w = 0.0 if cfg is None else cfg.spread_width
    out = {}
    for k, v in td.items():
        if k in ("T", "u", "v") or k not in fine:
            out[k] = spread(v, ph_H, pf_f, dp_f, G, w)
        else:
            out[k] = exchange_bulk(v * dt, fine[k], ph_H, pf_f, dp_f, G, cfg)[0] / dt
    return out


def make_conv_replay(G: FiveGrid, config, dt: float, cfg=None):
    """Convection module for the fine physics: the host ZM output from the
    forcing, every tendency exchanged to the fine grid (13.18) as an
    increment over the physics step ``dt``, the detrained condensate spread as
    a source, per-level carries constant per layer."""
    w = 0.0 if cfg is None else cfg.spread_width
    nH = G.n_sub.size
    nf = G.owner.size

    def replay(state, grid, sigma_coord, phys_state=None, forcing=None):
        (t_zm, upd_zm), fine, ph_H, pf_f, dp_f = forcing[CONV_REPLAY_KEY]
        td = _tend_dict(t_zm)
        out = replay_tendencies(td, fine, ph_H, pf_f, dp_f, G, dt, cfg)
        tr = None
        if t_zm.tracer_tendencies is not None:
            tr = {k: f.replace(data=out[k]) for k, f in t_zm.tracer_tendencies.items()}
        kw = dict(dT_dt=t_zm.dT_dt.replace(data=out["T"]),
                  du_dt=t_zm.du_dt.replace(data=out["u"]), tracer_tendencies=tr)
        if t_zm.dv_dt is not None:
            kw["dv_dt"] = t_zm.dv_dt.replace(data=out["v"])
        t_f = t_zm._replace(**kw)

        def per_level(x):
            if hasattr(x, "data") and hasattr(x, "replace"):
                y = per_level(x.data)
                return x.replace(data=y)
            if hasattr(x, "ndim") and x.ndim == 2 and x.shape[1] in (nH, nH + 1):
                return _map_carry(x, nH, nf, G, ph_H, pf_f)
            return x
        for name in t_f._fields:
            if name in ("dT_dt", "du_dt", "dv_dt", "tracer_tendencies"):
                continue
            v = getattr(t_f, name)
            if v is not None:
                t_f = t_f._replace(**{name: per_level(v)})
        upd = {}
        for k, v in (upd_zm or {}).items():
            if k == "conv_dlf" and v is not None:
                upd[k] = spread(v.reshape(-1, nH), ph_H, pf_f, dp_f, G, w)
            else:
                upd[k] = per_level(v)
        return t_f, upd

    replay._wants_forcing = True
    return replay
