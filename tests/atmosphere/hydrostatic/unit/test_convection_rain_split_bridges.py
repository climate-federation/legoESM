"""#929 in-updraft rain-split CONSERVATION through the 3 standalone convection bridges.

Bechtold (and Tiedtke) split their detrained condensate into anvil cloud
(``dq_c_conv_dt``) and in-updraft rain (``dq_r_conv_dt``) by ``precip_efficiency``.
Before the #929 bridge fix, ``make_convection_physics`` forwarded ONLY
``dq_c_conv_dt`` and DROPPED ``dq_r_conv_dt`` -- so at the 0.7 default, 70 % of
the convective condensate vanished from the water budget.  The fix routes
``dq_r_conv_dt`` to a ``q_r`` rain tracer when the state has one (microphysics
sediments it -- matching the unified pipeline's precip routing), else folds it
back into the ``q_c`` tendency, so the TOTAL convective condensate the bridge
routes to tracer sinks is INVARIANT to ``precip_efficiency`` (a pure
re-partition of a shared positive quantity).

Each test runs a firing Bechtold column through one bridge at
``precip_efficiency`` 0.0 (legacy) and 0.7 (default), and asserts:

* NON-VACUOUS: the total condensate routed at PE=0 is strictly > 0 (the state
  actually fires convection -- otherwise the invariance check is trivially
  satisfied by zero).
* CONSERVATION: the total condensate routed at PE=0.7 EQUALS that at PE=0 to
  machine precision -- the split re-partitions water, never creating or
  destroying it.  (Before the fix, PE=0.7 routed only 30 % -> a 0.7 relative
  gap, orders of magnitude outside the tolerance.)

The routed condensate is summed over the whole grid: Bechtold's
``dq_c_conv_dt``/``dq_r_conv_dt`` are non-negative SOURCES, so the sum is a
strictly-positive linear reduction under which the re-partition invariance holds
exactly.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis_3d
from legoesm.grids.vertical import (
    compute_terrain_metric,
    create_height_coordinate,
    create_sigma_coordinate,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.atmosphere.dynamics.gcm.spectral_pe import isothermal_rest_state_spectral
from legoesm.atmosphere.physics.convection.config import (
    BechtoldConfig,
    ConvectionConfig,
)
from legoesm.atmosphere.physics.convection.integration import (
    make_convection_physics,
)
from legoesm.thermo import saturation_mixing_ratio

# --- Conditionally-unstable firing fixture (the canonical spectral-PE
# convection profile: warm moist surface under a ~6.5 K/km troposphere, so a
# lifted moist parcel finds positive CAPE and Bechtold detrains condensate). ---
_T_SURFACE_K = 300.0        # fixture surface temperature [K]
_T_SIGMA_EXPONENT = 0.19    # fixture lapse exponent: T(sigma) = T_s * sigma^e
_T_FLOOR_K = 200.0          # stratospheric temperature floor [K]
_SIGMA_FLOOR = 0.05         # clip so sigma^e does not blow up at the model top
_RH_BL = 0.95               # near-saturated boundary layer
_RH_FREE = 0.5              # drier free troposphere
_RH_SPLIT_SIGMA = 0.7       # boundary-layer / free-troposphere split in sigma

_DT_S = 300.0               # fixture physics timestep [s]
_N_CUBE = 8                 # cubed-sphere face resolution
_N_LEV = 10                 # vertical levels
_GAUSSIAN_NMAX = 21         # spectral truncation
_NH_DOMAIN_TOP_M = 30000.0  # non-hydrostatic height-coordinate top [m]
_RTOL = 1e-11               # x64 machine-precision invariance tolerance
_DTYPE = jnp.float64


@pytest.fixture(autouse=True)
def _fp64_policy():
    """Run in fp64 so the re-partition invariance is testable at ~1e-11.

    The default legoESM policy is fp32 (independent of ``JAX_ENABLE_X64``); the
    state builders honour ``get_policy().storage``, so pin fp64 for the state
    construction, then restore the prior policy."""
    prev = get_policy()
    prev_x64 = jax.config.jax_enable_x64
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)
        # set_policy restores the policy OBJECT but NOT the global
        # jax_enable_x64 flag it flipped to run fp64, so x64 would leak into
        # later fp32 tests in the same process (codex #956/#929).  Restore the
        # flag explicitly.
        jax.config.update("jax_enable_x64", prev_x64)


def _firing_T_qv(sigma_full, horiz_shape):
    """Conditionally-unstable ``T`` [K] and near-saturated ``q_v`` [kg/kg] on a
    firing column, broadcast to ``horiz_shape + (nlev,)`` (grid-space)."""
    nlev = int(sigma_full.shape[0])
    ndim_h = len(horiz_shape)
    lead = (1,) * ndim_h
    full_shape = tuple(horiz_shape) + (nlev,)

    t_col = jnp.maximum(
        _T_SURFACE_K
        * jnp.power(jnp.clip(sigma_full, _SIGMA_FLOOR, None), _T_SIGMA_EXPONENT),
        _T_FLOOR_K,
    ).astype(_DTYPE)
    t_grid = jnp.broadcast_to(t_col.reshape(lead + (nlev,)), full_shape)

    p_full = jnp.broadcast_to(
        (sigma_full * constants.p_ref).reshape(lead + (nlev,)).astype(_DTYPE),
        full_shape,
    )
    q_sat = saturation_mixing_ratio(t_grid, p_full)
    rh = jnp.where(sigma_full > _RH_SPLIT_SIGMA, _RH_BL, _RH_FREE).astype(_DTYPE)
    q_v_grid = jnp.broadcast_to(rh.reshape(lead + (nlev,)), full_shape) * q_sat
    return t_grid, q_v_grid


def _bechtold_bridge(pe, model_type, dt=_DT_S, inplume=True):
    # ``inplume`` toggles ``use_ifs_inplume_precip`` (production default True):
    # the IFS in-plume precipitation generates rain INDEPENDENTLY of
    # ``precip_efficiency``, so a PE=0 firing column still emits ``dq_r`` under
    # the default.  The legacy-premise bridge tests below ("PE=0 => no rain")
    # therefore pass ``inplume=False`` to isolate the ``precip_efficiency``
    # split from the in-plume source (#1322).
    cfg = ConvectionConfig(
        scheme="bechtold",
        bechtold=BechtoldConfig(precip_efficiency=pe,
                                use_ifs_inplume_precip=inplume),
    )
    return make_convection_physics(cfg, model_type=model_type, dt=dt)


def _total_routed_condensate(field_stack):
    """Grid-total of the routed convective condensate sources (>= 0)."""
    return float(jnp.sum(field_stack))


# ---------------------------------------------------------------------------
# Hydrostatic bridge (tracers = name-keyed dict; q_c always emitted)
# ---------------------------------------------------------------------------

def _run_hydrostatic(pe, with_qr, inplume=True):
    grid = create_cubed_sphere(_N_CUBE)
    sigma = create_sigma_coordinate(_N_LEV)
    state = held_suarez_init(grid, sigma)
    horiz = state.p_s.data.shape  # (6, n, n)
    t_grid, q_v_grid = _firing_T_qv(sigma.sigma_full, horiz)
    dtype = state.T.data.dtype
    dims_3d = state.T.dims
    zeros = jnp.zeros(t_grid.shape, dtype=dtype)
    tracers = {
        "q_v": Field(data=q_v_grid.astype(dtype), name="q_v",
                     dims=dims_3d, units="kg/kg"),
        "q_c": Field(data=zeros, name="q_c", dims=dims_3d, units="kg/kg"),
    }
    if with_qr:
        tracers["q_r"] = Field(data=zeros, name="q_r",
                               dims=dims_3d, units="kg/kg")
    state = state._replace(
        T=state.T.replace(data=t_grid.astype(dtype)), tracers=tracers,
    )
    tend, _ = _bechtold_bridge(pe, "hydrostatic", inplume=inplume)(
        state, grid, sigma)
    return tend.tracer_tendencies


def _hydro_total(tt):
    total = tt["q_c"].data
    if "q_r" in tt:
        total = total + tt["q_r"].data
    return _total_routed_condensate(total)


def test_hydrostatic_bridge_conserves_condensate_folded():
    """No ``q_r`` tracer -> the split is folded back into ``q_c``; the total
    convective condensate routed is invariant to precip_efficiency."""
    s0 = _hydro_total(_run_hydrostatic(0.0, with_qr=False))
    s7 = _hydro_total(_run_hydrostatic(0.7, with_qr=False))
    assert s0 > 0.0, "hydrostatic state did not fire Bechtold (vacuous test)"
    assert s7 == pytest.approx(s0, rel=_RTOL), (
        f"rain split lost/created water through the hydrostatic bridge: "
        f"PE=0 routed {s0:.6e}, PE=0.7 routed {s7:.6e}"
    )


def test_hydrostatic_bridge_conserves_condensate_qr_tracer():
    """With a ``q_r`` tracer present -> the rain is routed to ``q_r``; the total
    (q_c + q_r) still equals the PE=0 condensate."""
    s0 = _hydro_total(_run_hydrostatic(0.0, with_qr=True))
    s7 = _hydro_total(_run_hydrostatic(0.7, with_qr=True))
    assert s0 > 0.0, "hydrostatic state did not fire Bechtold (vacuous test)"
    assert s7 == pytest.approx(s0, rel=_RTOL)


def test_hydrostatic_bridge_routes_rain_to_qr_only_when_split_on():
    """PE=0 (in-plume OFF) emits no ``q_r`` tendency (legacy, dq_r None); PE=0.7
    with a ``q_r`` tracer routes a strictly-positive rain tendency there (not
    folded).  ``inplume=False`` isolates the precip_efficiency split: with the
    production default ``use_ifs_inplume_precip=True`` a PE=0 column DOES rain
    (see ``test_inplume_default_rains_at_pe0``) — #1322."""
    tt0 = _run_hydrostatic(0.0, with_qr=True, inplume=False)
    assert "q_r" not in tt0, "PE=0 (in-plume off) must not emit a rain tendency"
    tt7 = _run_hydrostatic(0.7, with_qr=True)
    assert "q_r" in tt7, "PE=0.7 must route rain to the q_r tracer"
    assert _total_routed_condensate(tt7["q_r"].data) > 0.0


def test_inplume_default_rains_at_pe0():
    """Companion to the legacy-premise tests (#1322): with the PRODUCTION
    default ``use_ifs_inplume_precip=True``, the IFS in-plume precipitation
    generates rain INDEPENDENTLY of ``precip_efficiency``, so a PE=0 firing
    column DOES emit a strictly-positive ``dq_r`` — the invariant the three
    ``inplume=False`` bridge tests deliberately switch off.  Pins the default
    so a future flip back to ``precip_efficiency``-gated rain is caught."""
    tt0 = _run_hydrostatic(0.0, with_qr=True, inplume=True)
    assert "q_r" in tt0, (
        "in-plume default (use_ifs_inplume_precip=True) must emit convective "
        "rain even at precip_efficiency=0")
    assert _total_routed_condensate(tt0["q_r"].data) > 0.0


# ---------------------------------------------------------------------------
# Non-hydrostatic bridge (tracers = array; slot 0=q_v, 1=q_c, 2=q_r)
# ---------------------------------------------------------------------------

def _build_nonhydrostatic_firing_state(n_tracers):
    """Conditionally-unstable firing non-hydro state with ``n_tracers`` tracer
    slots (slot 0=q_v; slot 1=q_c and slot 2=q_r when present)."""
    n, nlev = _N_CUBE, _N_LEV
    grid = create_cubed_sphere(n)
    hc = create_height_coordinate(nlev, _NH_DOMAIN_TOP_M)
    terrain_metric = compute_terrain_metric(jnp.zeros((6, n, n), dtype=_DTYPE), hc)

    # Impose the SAME conditionally-unstable firing T profile as the
    # hydrostatic/spectral fixtures, via theta_prime: T_target(p) =
    # T_s * (p/p0)^e, theta_target = T_target / exner_ref, theta' =
    # theta_target - theta_ref (rho'=0 keeps rho = rho_ref, so the diagnosed p
    # is close to the reference p).  A near-saturated q_v(T_target) then gives a
    # lifted parcel positive CAPE and Bechtold detrains condensate.
    p_over_p0 = (hc.exner_ref ** (1.0 / constants.kappa)).astype(_DTYPE)   # (nlev,)
    t_target = jnp.maximum(
        _T_SURFACE_K * p_over_p0 ** _T_SIGMA_EXPONENT, _T_FLOOR_K,
    ).astype(_DTYPE)                                                       # (nlev,)
    p_target = (constants.p_ref * p_over_p0).astype(_DTYPE)                # (nlev,)
    theta_col = (t_target / hc.exner_ref.astype(_DTYPE) - hc.theta_ref).astype(_DTYPE)
    theta_p = jnp.broadcast_to(theta_col[None, None, None, :], (6, n, n, nlev))

    q_sat = saturation_mixing_ratio(t_target, p_target)                   # (nlev,)
    rh = jnp.where(p_over_p0 > _RH_SPLIT_SIGMA, _RH_BL, _RH_FREE).astype(_DTYPE)
    q_v_grid = jnp.broadcast_to((rh * q_sat)[None, None, None, :], (6, n, n, nlev))

    tracers_arr = jnp.zeros((6, n, n, nlev, n_tracers), dtype=_DTYPE)
    tracers_arr = tracers_arr.at[..., 0].set(q_v_grid)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_tr = ("face", "x", "y", "level", "tracer")
    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev), dtype=_DTYPE), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev), dtype=_DTYPE), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1), dtype=_DTYPE), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=theta_p, name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=_DTYPE),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n), dtype=_DTYPE), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=tracers_arr, name="tracers", dims=dims_tr,
                      units="kg/kg"),
    )
    return state, grid, hc, terrain_metric


def _run_nonhydrostatic(pe, n_tracers):
    state, grid, hc, terrain_metric = _build_nonhydrostatic_firing_state(n_tracers)
    tend, _ = _bechtold_bridge(pe, "nonhydrostatic")(
        state, grid, hc, terrain_metric,
    )
    dtr = tend.dtracers_dt.data  # (6, n, n, nlev, n_tracers)
    total = dtr[..., 1]  # q_c slot
    if n_tracers > 2:
        total = total + dtr[..., 2]  # q_r slot
    return _total_routed_condensate(total)


def test_nonhydrostatic_bridge_conserves_condensate_folded():
    """n_tracers=2 (no q_r slot) -> rain folded into q_c; total invariant."""
    s0 = _run_nonhydrostatic(0.0, n_tracers=2)
    s7 = _run_nonhydrostatic(0.7, n_tracers=2)
    assert s0 > 0.0, "non-hydrostatic state did not fire Bechtold (vacuous)"
    assert s7 == pytest.approx(s0, rel=_RTOL), (
        f"rain split lost/created water through the non-hydrostatic bridge "
        f"(fold): PE=0 {s0:.6e}, PE=0.7 {s7:.6e}"
    )


def test_nonhydrostatic_bridge_conserves_condensate_qr_slot():
    """n_tracers=3 (q_rain slot 2) -> rain routed to slot 2; total invariant."""
    s0 = _run_nonhydrostatic(0.0, n_tracers=3)
    s7 = _run_nonhydrostatic(0.7, n_tracers=3)
    assert s0 > 0.0, "non-hydrostatic state did not fire Bechtold (vacuous)"
    assert s7 == pytest.approx(s0, rel=_RTOL)


# ---------------------------------------------------------------------------
# Spectral-PE bridge (tracers = name-keyed dict; q_c emitted iff "q_c" present)
# ---------------------------------------------------------------------------

def _build_spectral_firing_state(include_qc, include_qr):
    """Conditionally-unstable firing spectral state; the ``q_c`` / ``q_r``
    tracers are included per the flags (``q_v`` is always present)."""
    grid = create_gaussian_grid(n_max=_GAUSSIAN_NMAX)
    sigma = create_sigma_coordinate(_N_LEV)
    rest = isothermal_rest_state_spectral(grid, sigma, perturbation_amplitude=0.0)
    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels

    t_grid, q_v_grid = _firing_T_qv(sigma.sigma_full, (n_lat, n_lon))
    t_hat = sh_analysis_3d(grid, t_grid)
    dims = ("lat", "lon", "level")
    zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=_DTYPE)
    tracers = {"q_v": Field(data=q_v_grid, name="q_v", dims=dims, units="kg/kg")}
    if include_qc:
        tracers["q_c"] = Field(data=zeros, name="q_c", dims=dims, units="kg/kg")
    if include_qr:
        tracers["q_r"] = Field(data=zeros, name="q_r", dims=dims, units="kg/kg")
    state = rest._replace(
        T_hat=rest.T_hat.replace(data=t_hat), tracers=tracers,
    )
    return state, grid, sigma


def _run_spectral(pe, with_qr):
    state, grid, sigma = _build_spectral_firing_state(
        include_qc=True, include_qr=with_qr)
    tend, _ = _bechtold_bridge(pe, "spectral_pe")(state, grid, sigma)
    return tend.tracers


def _spectral_total(tt):
    total = tt["q_c"].data
    if "q_r" in tt and tt["q_r"] is not None:
        total = total + tt["q_r"].data
    return _total_routed_condensate(total)


def test_spectral_bridge_conserves_condensate_folded():
    """No ``q_r`` tracer -> rain folded into ``q_c``; total invariant."""
    s0 = _spectral_total(_run_spectral(0.0, with_qr=False))
    s7 = _spectral_total(_run_spectral(0.7, with_qr=False))
    assert s0 > 0.0, "spectral state did not fire Bechtold (vacuous test)"
    assert s7 == pytest.approx(s0, rel=_RTOL), (
        f"rain split lost/created water through the spectral bridge (fold): "
        f"PE=0 {s0:.6e}, PE=0.7 {s7:.6e}"
    )


def test_spectral_bridge_conserves_condensate_qr_tracer():
    """With a ``q_r`` tracer present -> rain routed to ``q_r``; total invariant."""
    s0 = _spectral_total(_run_spectral(0.0, with_qr=True))
    s7 = _spectral_total(_run_spectral(0.7, with_qr=True))
    assert s0 > 0.0, "spectral state did not fire Bechtold (vacuous test)"
    assert s7 == pytest.approx(s0, rel=_RTOL)


# ---------------------------------------------------------------------------
# #929: a rain-splitting scheme (bechtold/tiedtke, pe>0) needs a condensate
# tracer to hold the diverted rain — a condensate-less state is an UNSUPPORTED
# config and the bridge must raise LOUDLY (no silent water drop; bechtold's dq_v
# is NOT -(dq_c+dq_r) pointwise, so dropping dq_r leaks column water).
# ---------------------------------------------------------------------------

def test_nonhydrostatic_bridge_raises_without_condensate_tracer():
    """pe=0.7 emits dq_r but a vapor-only non-hydro state (n_tracers=1) has no
    q_c/q_r slot to receive it -> loud ValueError."""
    state, grid, hc, tm = _build_nonhydrostatic_firing_state(n_tracers=1)
    with pytest.raises(ValueError, match="condensate"):
        _bechtold_bridge(0.7, "nonhydrostatic")(state, grid, hc, tm)


def test_nonhydrostatic_bridge_no_raise_when_split_off():
    """Non-vacuity control: the guard is dq_r-GATED, not n_tracers-gated.  At
    pe=0.0 Bechtold emits dq_r=None, so the SAME vapor-only state (n_tracers=1)
    does NOT raise (byte-identical to the pre-#929 sbm/dca/kuo path)."""
    state, grid, hc, tm = _build_nonhydrostatic_firing_state(n_tracers=1)
    tend, _ = _bechtold_bridge(0.0, "nonhydrostatic", inplume=False)(
        state, grid, hc, tm)
    assert tend is not None  # reached — no raise


def test_spectral_bridge_raises_without_condensate_tracer():
    """pe=0.7 emits dq_r but a q_v-only spectral state has neither a q_c nor a
    q_r tracer to receive the convective rain split -> loud ValueError."""
    state, grid, sigma = _build_spectral_firing_state(
        include_qc=False, include_qr=False)
    with pytest.raises(ValueError):
        _bechtold_bridge(0.7, "spectral_pe")(state, grid, sigma)


def test_spectral_bridge_raises_when_tracers_is_none():
    """#929 / codex round-3: the spectral routing+raise is nested under
    ``state.tracers is not None``.  A tracer-less spectral state
    (``state.tracers is None`` -- a VALID spectral-PE state) would SKIP the
    whole block and silently drop the split.  With pe=0.7 Bechtold emits a
    (non-None) dq_r source, so even ``tracers is None`` must raise LOUDLY,
    consistent with the q_v-only case above."""
    state, grid, sigma = _build_spectral_firing_state(
        include_qc=False, include_qr=False)
    state = state._replace(tracers=None)
    with pytest.raises(ValueError, match="tracers"):
        _bechtold_bridge(0.7, "spectral_pe")(state, grid, sigma)


def test_spectral_bridge_tracers_none_no_raise_when_split_off():
    """Non-vacuity control: the ``tracers is None`` guard is dq_r-GATED, not
    tracers-gated.  At pe=0.0 Bechtold emits dq_r=None, so a tracer-less state
    does NOT raise (byte-identical to the pre-#929 drop-the-tendency path)."""
    state, grid, sigma = _build_spectral_firing_state(
        include_qc=False, include_qr=False)
    state = state._replace(tracers=None)
    tend, _ = _bechtold_bridge(0.0, "spectral_pe", inplume=False)(
        state, grid, sigma)
    assert tend is not None  # reached — no raise
