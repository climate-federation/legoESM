"""Salinity(-pressure)-dependent seawater freezing point (``eos.freezing_point``).

MED-1.  The freeze checks that cap SST / trigger ice formation historically used
a single FIXED constant ``constants.T_freeze_ocean`` (271.35 K, ~-1.8 C).  Real
seawater freezes along a LIQUIDUS that decreases with salinity (and, weakly,
pressure): at S = 35 PSU, p = 0 the true value is ~-1.92 C.  ``freezing_point``
exposes three schemes; this suite exercises the real function directly.

Checks:
  * ``constant`` scheme is byte-identical to ``constants.T_freeze_ocean`` (scalar
    + array), so the default wiring in the ocean/slab freeze paths is inert;
  * ``linear_S`` and ``unesco`` give 0 C at S=0 (pure water) and the expected
    depressions at S=35 (unesco ~-1.92 C, NOT -1.8);
  * SIGN: T_f is monotone DECREASING in salinity, and pressure LOWERS it;
  * the function is differentiable (grad finite, incl. at S=0), jit-safe, and
    vmap-safe, and floors negative-S overshoots without NaN;
  * unknown scheme raises ``ValueError`` (dispatch hardening);
  * the consumer configs default to ``scheme="constant"`` (byte-identical opt-in);
  * CONSUMER WIRING (MED-1 follow-up; codex "unwired consumers"): the scheme
    actually reaches every freeze check — the lat-lon slab/two-layer clamps
    (``simple_ocean``), the MPAS slab/two-layer clamps (``simple_ocean_mpas``),
    the lat-lon ``_apply_freeze_floor`` (per-cell threshold from LOCAL surface
    salinity), the OMIP under-ice relaxation (``under_ice_freeze_relax``
    per-cell liquidus target), and the flux-feedback ice-mask threshold — with
    (i) thresholds varying with local S under a liquidus scheme, (ii) Q_freeze
    responding consistently, and (iii) ``"constant"`` byte-identical to the
    pre-MED-1 behaviour.  The slabs carry no prognostic salinity, so their
    liquidus is the uniform ``constants.S_ocean_ref`` evaluation via the single
    shared owner ``eos.slab_freeze_point_K``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    VALID_FREEZE_SCHEMES,
    FreezingPointConfig,
    freezing_point,
)

# Reference depression in degC (temperature difference => same value in K).
_T0_K = constants.T_freeze  # pure-water freezing point (0 degC reference) [K]


@pytest.fixture(autouse=True)
def _fp64_policy():
    """Run the whole module in fp64.  The liquidus asserts here are at the
    1e-12 (f64) level, but the global default ``PrecisionPolicy`` is fp32 and
    the EOS / ``rest_state_latlon_cgrid_ocean`` builders honour it — so under
    fp32 the true-zero S=0 liquidus reads as ~-6e-6 noise and the seawater
    depressions lose ~6 digits.  ``tests/ocean/unit`` is in no CI tier, so main
    never exercised this and the drift went unseen (issue #955)."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    prev = get_policy()
    prev_x64 = jax.config.jax_enable_x64
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)
        # set_policy restores the PrecisionPolicy object but NOT the global
        # jax_enable_x64 flag it flipped on — that would leak x64 into
        # later-imported test modules in a shared session (codex #956).
        jax.config.update("jax_enable_x64", prev_x64)


# ---------------------------------------------------------------------------
# constant scheme — byte-identical to the historical fixed value
# ---------------------------------------------------------------------------

def test_constant_scheme_byte_identical_scalar():
    """``constant`` returns exactly ``constants.T_freeze_ocean`` (any S/p)."""
    for S in (0.0, 35.0, 40.0):
        tf = freezing_point(S, 0.0, scheme="constant")
        assert float(tf) == constants.T_freeze_ocean  # exact, not approx
    # Salinity/pressure are ignored by the constant scheme.
    assert float(freezing_point(35.0, 1.0e7, scheme="constant")) == constants.T_freeze_ocean


def test_constant_scheme_byte_identical_array():
    """Array input -> per-cell field of the fixed constant, exactly."""
    S = jnp.array([0.0, 30.0, 34.7, 40.0])
    tf = freezing_point(S, scheme="constant")
    assert tf.shape == S.shape
    assert np.array_equal(np.asarray(tf), np.full(S.shape, constants.T_freeze_ocean))


def test_default_scheme_is_constant():
    """The default (no ``scheme=``) is the byte-identical constant."""
    assert float(freezing_point(35.0)) == constants.T_freeze_ocean
    assert FreezingPointConfig().scheme == "constant"


# ---------------------------------------------------------------------------
# pure-water endpoint (S=0 -> 0 degC) for the liquidus schemes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["linear_S", "unesco"])
def test_pure_water_freezes_at_zero_C(scheme):
    """S=0, p=0 -> T0 = constants.T_freeze (0 degC) for the liquidus schemes."""
    tf = float(freezing_point(0.0, 0.0, scheme=scheme))
    assert tf == pytest.approx(_T0_K, abs=1e-9)
    assert (tf - _T0_K) == pytest.approx(0.0, abs=1e-9)  # 0 degC


# ---------------------------------------------------------------------------
# the headline value: S=35 -> ~-1.92 C (unesco), NOT -1.8
# ---------------------------------------------------------------------------

def test_unesco_at_S35_is_minus_1p92_C():
    """UNESCO/Millero at S=35, p=0 is ~-1.922 C (271.23 K), not -1.8."""
    tf_K = float(freezing_point(35.0, 0.0, scheme="unesco"))
    tf_C = tf_K - _T0_K
    assert tf_C == pytest.approx(-1.9223, abs=5e-3)
    assert tf_C == pytest.approx(-1.92, abs=1e-2)      # spec target
    # Strictly colder than the old fixed -1.8 C constant.
    assert tf_K < constants.T_freeze_ocean
    # Distinct from the fixed constant by ~0.12 C.
    assert (constants.T_freeze_ocean - tf_K) == pytest.approx(0.12, abs=0.02)


def test_linear_S_slope():
    """linear_S: T_f = T0 - 0.0575*S; at S=35 -> -2.0125 C."""
    tf_C = float(freezing_point(35.0, 0.0, scheme="linear_S")) - _T0_K
    assert tf_C == pytest.approx(-0.0575 * 35.0, abs=1e-9)


# ---------------------------------------------------------------------------
# SIGN gate: monotone decreasing in S; pressure lowers T_f
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["linear_S", "unesco"])
def test_monotone_decreasing_in_salinity(scheme):
    """T_f decreases as S increases (more saline water freezes colder)."""
    S = jnp.array([0.0, 10.0, 20.0, 30.0, 35.0, 40.0])
    tf = np.asarray(freezing_point(S, 0.0, scheme=scheme))
    diffs = np.diff(tf)
    assert np.all(diffs < 0.0), f"{scheme} not strictly decreasing: {tf}"


def test_pressure_lowers_freezing_point_unesco():
    """Deeper (higher p) water freezes colder for the pressure-aware scheme."""
    tf_surface = float(freezing_point(35.0, 0.0, scheme="unesco"))
    tf_deep = float(freezing_point(35.0, 1.0e7, scheme="unesco"))  # 1000 dbar
    assert tf_deep < tf_surface
    # -7.53e-4 degC/dbar * 1000 dbar = -0.753 degC lowering.
    assert (tf_deep - tf_surface) == pytest.approx(-0.753, abs=1e-6)


def test_linear_S_is_pressure_independent():
    """linear_S ignores pressure (surface liquidus, per MOM6 linear form)."""
    a = float(freezing_point(35.0, 0.0, scheme="linear_S"))
    b = float(freezing_point(35.0, 1.0e7, scheme="linear_S"))
    assert a == b


# ---------------------------------------------------------------------------
# differentiability / jit / vmap / negative-S guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["linear_S", "unesco"])
def test_grad_finite_and_negative(scheme):
    """d T_f / d S is finite everywhere (incl. S=0) and negative for S>0.

    At exactly S=0 the ``jnp.maximum(S, 0)`` guard's subgradient is tie-broken by
    JAX, so only FINITENESS is asserted there (the S**1.5 term is grad-safe at 0,
    unlike S*sqrt(S)); strict negativity is checked at strictly-positive S.
    """
    g = jax.grad(lambda s: freezing_point(s, 0.0, scheme=scheme))
    assert np.isfinite(float(g(0.0))), f"{scheme} grad not finite at S=0"
    for S in (1e-6, 5.0, 35.0):
        dg = float(g(S))
        assert np.isfinite(dg), f"{scheme} grad not finite at S={S}"
        assert dg < 0.0, f"{scheme} grad not negative at S={S}: {dg}"
    # linear_S slope is exactly the coefficient.
    if scheme == "linear_S":
        assert float(g(35.0)) == pytest.approx(-0.0575, abs=1e-12)


def test_jit_safe():
    """freezing_point compiles under jit with the scheme static."""
    f = jax.jit(freezing_point, static_argnames=("scheme",))
    tf = f(35.0, 0.0, scheme="unesco")
    assert float(tf) == pytest.approx(
        float(freezing_point(35.0, 0.0, scheme="unesco")), abs=1e-12
    )


def test_vmap_safe():
    """vmap over a salinity vector matches the direct broadcast."""
    S = jnp.array([30.0, 33.0, 35.0, 37.0])
    direct = freezing_point(S, 0.0, scheme="unesco")
    vmapped = jax.vmap(lambda s: freezing_point(s, 0.0, scheme="unesco"))(S)
    assert np.allclose(np.asarray(direct), np.asarray(vmapped), atol=1e-12)


@pytest.mark.parametrize("scheme", ["linear_S", "unesco"])
def test_negative_salinity_guarded(scheme):
    """A negative-S overshoot is floored at 0 (no NaN, returns T0)."""
    tf = float(freezing_point(-5.0, 0.0, scheme=scheme))
    assert np.isfinite(tf)
    assert tf == pytest.approx(_T0_K, abs=1e-9)  # clamped S=0 -> pure water


# ---------------------------------------------------------------------------
# dispatch hardening + config schema
# ---------------------------------------------------------------------------

def test_unknown_scheme_raises():
    """A typo'd scheme raises ValueError, never silently defaults."""
    with pytest.raises(ValueError, match="Unknown freezing-point scheme"):
        freezing_point(35.0, 0.0, scheme="linearS")  # typo
    with pytest.raises(ValueError):
        freezing_point(35.0, 0.0, scheme="millero")


def test_valid_freeze_schemes_set():
    assert VALID_FREEZE_SCHEMES == frozenset({"constant", "linear_S", "unesco"})
    # Every scheme in the set is actually dispatchable.
    for scheme in VALID_FREEZE_SCHEMES:
        assert np.isfinite(float(freezing_point(35.0, 0.0, scheme=scheme)))


def test_consumer_configs_default_constant():
    """The wired consumer configs default to the byte-identical constant."""
    from legoesm.ocean.simple_ocean import SimpleOceanConfig
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.mpas_config import MPASOceanConfig, MPASSimpleOceanConfig
    from legoesm.ocean.physics.surface_forcing.config import FluxFeedbackConfig

    assert SimpleOceanConfig().freezing.scheme == "constant"
    assert LatLonCGridOceanConfig().freezing.scheme == "constant"
    assert MPASOceanConfig().freezing.scheme == "constant"
    assert MPASSimpleOceanConfig().freezing.scheme == "constant"
    assert FluxFeedbackConfig().freezing.scheme == "constant"


# ---------------------------------------------------------------------------
# consumer wiring (MED-1 follow-up; codex "unwired consumers" defect)
# ---------------------------------------------------------------------------

def _cooling_forcing(shape):
    """``AtmToSurface`` driving a strong surface heat LOSS (freeze-clamp regime)."""
    from legoesm.core.coupling_fields import AtmToSurface
    z = jnp.zeros(shape)
    return AtmToSurface(
        sw_down=z,
        lw_down=jnp.full(shape, 150.0),
        precip_total=z, precip_snow=z,
        T_lowest=jnp.full(shape, 240.0),   # cold air -> strong sensible loss
        q_lowest=z,                        # dry air  -> strong latent loss
        u_lowest=jnp.full(shape, 15.0),    # windy    -> large exchange
        v_lowest=z,
        p_lowest=jnp.full(shape, 95000.0),
        p_surface=jnp.full(shape, 1.0e5),
        rho_lowest=jnp.full(shape, 1.25),
        cos_zenith=z,
        co2_ppmv=jnp.array(400.0),
        has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )


def test_slab_freeze_point_single_owner():
    """Both slab modules bind THE shared ``eos.slab_freeze_point_K`` (no dup)."""
    from legoesm.ocean import simple_ocean, simple_ocean_mpas
    from legoesm.ocean.eos import slab_freeze_point_K

    assert simple_ocean.slab_freeze_point_K is slab_freeze_point_K
    assert simple_ocean_mpas.slab_freeze_point_K is slab_freeze_point_K
    # "constant": the caller's T_freeze passes through EXACTLY (byte-identical).
    assert slab_freeze_point_K(constants.T_freeze_ocean, "constant") \
        == constants.T_freeze_ocean
    assert slab_freeze_point_K(123.456, "constant") == 123.456
    # Liquidus: evaluated at the reference salinity (slabs carry no S).
    for scheme in ("linear_S", "unesco"):
        assert float(slab_freeze_point_K(constants.T_freeze_ocean, scheme)) \
            == float(freezing_point(constants.S_ocean_ref, 0.0, scheme=scheme))
    # Typos can never silently fall back to the constant.
    with pytest.raises(ValueError, match="Unknown freezing-point scheme"):
        slab_freeze_point_K(constants.T_freeze_ocean, "milero")


@pytest.mark.parametrize("mode", ["slab", "two_layer"])
def test_latlon_slab_clamp_and_q_freeze_track_scheme(mode):
    """simple_ocean clamps: unesco shifts the floor AND Q_freeze; constant is
    byte-identical to the pre-MED-1 ``maximum(T_trial, cfg.T_freeze)`` clamp."""
    from legoesm.ocean.simple_ocean import (
        SimpleOceanConfig, init_slab_state, make_ocean,
    )
    shape, dt = (4,), 86400.0
    forcing = _cooling_forcing(shape)
    st = init_slab_state(shape, T_sfc_init=271.5)     # just above freezing
    cfg_c = SimpleOceanConfig(mode=mode)              # default: constant
    cfg_u = cfg_c._replace(freezing=FreezingPointConfig(scheme="unesco"))
    cfg_off = cfg_c._replace(T_freeze=100.0)          # clamp never fires

    s_c, _, _, _ = make_ocean(cfg_c)(st, forcing, dt)
    s_u, _, _, _ = make_ocean(cfg_u)(st, forcing, dt)
    s_off, _, _, _ = make_ocean(cfg_off)(st, forcing, dt)
    T_trial = np.asarray(s_off.T_sfc.data)            # unclamped trial SST

    tf_u = float(freezing_point(constants.S_ocean_ref, 0.0, scheme="unesco"))
    # The cooling genuinely drives the trial below BOTH floors (clamp exercised).
    assert np.all(T_trial < tf_u)
    assert tf_u < cfg_c.T_freeze                      # unesco floor is COLDER

    # (iii) constant path byte-identical to the pre-MED-1 clamp + Q_freeze.
    C_mix = cfg_c.rho_ocean * cfg_c.c_ocean * cfg_c.h_mix
    assert np.array_equal(np.asarray(s_c.T_sfc.data),
                          np.maximum(T_trial, cfg_c.T_freeze))
    # fp64 exposes the C_mix*max(..)/dt reassociation fp32 rounding masked; the
    # single-max T_sfc clamp above stays byte-identical.  Match the file's own
    # Q_freeze tolerance (the dq allclose rtol=1e-12 below).  Issue #955.
    assert np.allclose(np.asarray(s_c.Q_freeze.data),
                       C_mix * np.maximum(cfg_c.T_freeze - T_trial, 0.0) / dt,
                       rtol=1e-12)
    # (i) the scheme shifts the floor actually applied ...
    assert np.array_equal(np.asarray(s_u.T_sfc.data),
                          np.maximum(T_trial, tf_u))
    assert np.all(np.asarray(s_u.T_sfc.data) < np.asarray(s_c.T_sfc.data))
    # (ii) ... and Q_freeze responds consistently (same floor in clamp + energy).
    assert np.allclose(np.asarray(s_u.Q_freeze.data),
                       C_mix * np.maximum(tf_u - T_trial, 0.0) / dt,
                       rtol=1e-12)
    dq = np.asarray(s_c.Q_freeze.data) - np.asarray(s_u.Q_freeze.data)
    assert np.allclose(dq, C_mix * (cfg_c.T_freeze - tf_u) / dt, rtol=1e-12)


@pytest.mark.parametrize("mode", ["slab", "two_layer"])
def test_mpas_slab_clamp_and_q_freeze_track_scheme(mode):
    """simple_ocean_mpas clamps (codex defect 1b): same contract as lat-lon —
    the NEW ``MPASSimpleOceanConfig.freezing`` reaches BOTH the clamp and
    Q_freeze; the default stays byte-identical to the fixed ``config.T_freeze``."""
    from legoesm.ocean.mpas_config import MPASSimpleOceanConfig
    from legoesm.ocean.simple_ocean_mpas import (
        init_mpas_slab_state, make_mpas_ocean,
    )
    n, dt = 8, 86400.0
    forcing = _cooling_forcing((n,))
    st = init_mpas_slab_state(n, T_sfc_init=271.5)
    cfg_c = MPASSimpleOceanConfig(mode=mode)
    cfg_u = cfg_c._replace(freezing=FreezingPointConfig(scheme="unesco"))
    cfg_off = cfg_c._replace(T_freeze=100.0)

    s_c, _, _, _ = make_mpas_ocean(cfg_c)(st, forcing, dt)
    s_u, _, _, _ = make_mpas_ocean(cfg_u)(st, forcing, dt)
    s_off, _, _, _ = make_mpas_ocean(cfg_off)(st, forcing, dt)
    T_trial = np.asarray(s_off.T_sfc.data)

    tf_u = float(freezing_point(constants.S_ocean_ref, 0.0, scheme="unesco"))
    assert np.all(T_trial < tf_u)

    C_mix = cfg_c.rho_ocean * cfg_c.c_ocean * cfg_c.h_mix
    # (iii) byte-identical constant default (pre-MED-1 behaviour).
    assert np.array_equal(np.asarray(s_c.T_sfc.data),
                          np.maximum(T_trial, cfg_c.T_freeze))
    # fp64 exposes the C_mix*max(..)/dt reassociation fp32 rounding masked; the
    # single-max T_sfc clamp above stays byte-identical.  Match the file's own
    # Q_freeze tolerance (the dq allclose rtol=1e-12 below).  Issue #955.
    assert np.allclose(np.asarray(s_c.Q_freeze.data),
                       C_mix * np.maximum(cfg_c.T_freeze - T_trial, 0.0) / dt,
                       rtol=1e-12)
    # (i)+(ii) scheme shifts the clamp AND the Q_freeze energy booking together.
    assert np.array_equal(np.asarray(s_u.T_sfc.data),
                          np.maximum(T_trial, tf_u))
    assert np.allclose(np.asarray(s_u.Q_freeze.data),
                       C_mix * np.maximum(tf_u - T_trial, 0.0) / dt,
                       rtol=1e-12)


def _latlon_freeze_model(**cfg_kw):
    """Small lat-lon C-grid PE model with the freeze floor ON (mirrors
    tests/ocean/unit/test_freeze_floor.py)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(8, 16)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, n_barotropic_substeps=8,
        enable_runtime_checks=False, freeze_floor=True, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


@pytest.mark.parametrize("scheme", ["linear_S", "unesco"])
def test_latlon_freeze_floor_tracks_local_salinity(scheme):
    """lat-lon ``_apply_freeze_floor``: with a liquidus scheme the SURFACE floor
    is per-cell from the LOCAL surface salinity (fresher floors warmer)."""
    state, model = _latlon_freeze_model(
        freezing=FreezingPointConfig(scheme=scheme))
    T = np.array(state.T.data)
    S = np.array(state.S.data)
    T[..., 0] = -5.0        # every surface cell super-cooled below ALL floors
    T[..., 1:] = 3.0
    S[..., 0] = 35.0        # saline background ...
    S[2, 3, 0] = 0.0        # ... a fresh cell (river mouth) ...
    S[4, 5, 0] = 20.0       # ... and a brackish shelf cell
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)),
                           S=state.S.replace(data=jnp.asarray(S)))

    out = model._apply_freeze_floor(state)
    To = np.asarray(out.T.data)

    def tf_c(s):
        return float(freezing_point(s, 0.0, scheme=scheme)) - float(constants.T_freeze)

    # (i) the floor VARIES with the local S — each cell at ITS OWN liquidus.
    assert To[2, 3, 0] == pytest.approx(tf_c(0.0), abs=1e-12)    # 0 degC
    assert To[4, 5, 0] == pytest.approx(tf_c(20.0), abs=1e-12)
    assert To[0, 0, 0] == pytest.approx(tf_c(35.0), abs=1e-12)
    assert To[2, 3, 0] > To[4, 5, 0] > To[0, 0, 0]
    # Deep levels untouched (surface-only cap).
    assert np.array_equal(To[..., 1:], T[..., 1:])


def test_latlon_freeze_floor_constant_byte_identical():
    """Default ("constant") floors at the scalar ``freeze_floor_temp_c`` exactly
    — byte-identical to the pre-MED-1 clamp expression."""
    state, model = _latlon_freeze_model()   # default FreezingPointConfig()
    T = np.array(state.T.data)
    S = np.array(state.S.data)
    T[..., 0] = -5.0
    S[..., 0] = 35.0
    S[2, 3, 0] = 0.0        # salinity must NOT matter on the constant path
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)),
                           S=state.S.replace(data=jnp.asarray(S)))
    out = model._apply_freeze_floor(state)
    expected = T.copy()
    expected[..., 0] = np.maximum(T[..., 0], model.config.freeze_floor_temp_c)
    assert np.array_equal(np.asarray(out.T.data), expected)


def test_under_ice_relax_constant_path_byte_identical():
    """omip2 ``under_ice_freeze_relax`` (codex defect 1a): the default constant
    path reproduces the historical fixed-scalar update exactly."""
    from legoesm.ocean.coupler.omip2_applicator import under_ice_freeze_relax
    T = np.array([-0.5, 1.0, 3.0])
    sic = np.array([1.0, 0.5, 0.0])
    day = 86400.0
    legacy = under_ice_freeze_relax(T, sic, day, tau_ice_days=20.0)
    tfc = float(constants.T_freeze_ocean) - float(constants.T_freeze)
    # Explicit scalar target == default target (same constant).
    explicit = under_ice_freeze_relax(T, sic, day, tau_ice_days=20.0,
                                      T_freeze_C=tfc)
    assert np.array_equal(legacy, explicit)
    # Hand-computed historical float-scalar update.
    alpha = min(day / (20.0 * 86400.0), 1.0) * sic
    assert np.array_equal(legacy, T + alpha * (tfc - T))
    # sic=0 open water untouched.
    assert legacy[2] == T[2]


def test_under_ice_relax_per_cell_liquidus():
    """Liquidus scheme: the relaxation target is PER-CELL from the local
    surface salinity (fresher shelf water relaxes toward a warmer target)."""
    from legoesm.ocean.coupler.omip2_applicator import under_ice_freeze_relax
    S = np.array([0.0, 20.0, 35.0])
    T = np.full(3, -0.5)
    sic = np.ones(3)
    dt_huge = 1.0e12          # alpha clips to 1 -> lands exactly on the target
    out = under_ice_freeze_relax(T, sic, dt_huge, S_top=S,
                                 freeze_scheme="unesco")
    expected = (np.asarray(freezing_point(S, 0.0, scheme="unesco"))
                - float(constants.T_freeze))
    assert np.allclose(out, expected, atol=1e-12)
    # (i) target varies with LOCAL S, monotone (fresher -> warmer target).
    assert out[0] > out[1] > out[2]
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[2] == pytest.approx(-1.9223, abs=5e-3)
    # A per-cell T_freeze_C ARRAY is accepted directly (the float() defect).
    out2 = under_ice_freeze_relax(np.full(2, -3.0), np.ones(2), dt_huge,
                                  T_freeze_C=np.array([-1.0, -2.0]))
    assert np.allclose(out2, [-1.0, -2.0], atol=1e-12)


def test_under_ice_relax_guards():
    """Ambiguous / incomplete liquidus requests fail loud (dispatch hardening)."""
    from legoesm.ocean.coupler.omip2_applicator import under_ice_freeze_relax
    T, sic = np.zeros(2), np.ones(2)
    with pytest.raises(ValueError, match="not both"):
        under_ice_freeze_relax(T, sic, 3600.0, T_freeze_C=-1.8,
                               S_top=np.ones(2), freeze_scheme="unesco")
    with pytest.raises(ValueError, match="S_top"):
        under_ice_freeze_relax(T, sic, 3600.0, freeze_scheme="unesco")
    with pytest.raises(ValueError, match="Unknown freezing-point scheme"):
        under_ice_freeze_relax(T, sic, 3600.0, freeze_scheme="millero")


def test_flux_feedback_ice_threshold_tracks_local_salinity():
    """flux_feedback ice mask (codex defect 1c): a liquidus scheme makes the
    threshold the LOCAL freezing point; the constant default is byte-identical."""
    from legoesm.ocean.physics.surface_forcing.config import FluxFeedbackConfig
    from legoesm.ocean.physics.surface_forcing.flux_feedback import (
        flux_feedback_surface_forcing,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    # T_surf = -1.85 degC sits BETWEEN the fixed -1.8 threshold and the S=35
    # unesco liquidus (~-1.92): iced under "constant", NOT iced at S=35 under
    # "unesco" (its own freezing point is colder than -1.85).
    T = jnp.full((3, 2), -1.85)
    S = jnp.stack([jnp.full(2, 0.0), jnp.full(2, 20.0), jnp.full(2, 35.0)])
    dz_0 = jnp.ones(3)
    sf = OceanSurfaceForcing(q_prescribed=jnp.full(3, -50.0))   # cooling

    cfg_c = FluxFeedbackConfig()
    out_c = flux_feedback_surface_forcing(T, S, dz_0, sf, cfg_c)
    # (iii) constant: every cell is below -1.8 under cooling -> ALL zeroed;
    # and the explicit constant scheme is byte-identical to the default cfg.
    assert np.array_equal(np.asarray(out_c.dT_dt), np.zeros((3, 2)))
    assert np.array_equal(np.asarray(out_c.Q_net), np.zeros(3))
    out_c2 = flux_feedback_surface_forcing(
        T, S, dz_0, sf,
        cfg_c._replace(freezing=FreezingPointConfig(scheme="constant")))
    for a, b in zip(out_c, out_c2):
        assert np.array_equal(np.asarray(a), np.asarray(b))

    # (i) unesco: per-cell threshold — fresh/brackish cells (liquidus 0 /
    # ~-1.08 degC, warmer than -1.85) are iced; the S=35 cell is NOT.
    cfg_u = cfg_c._replace(freezing=FreezingPointConfig(scheme="unesco"))
    out_u = flux_feedback_surface_forcing(T, S, dz_0, sf, cfg_u)
    dT = np.asarray(out_u.dT_dt)[:, 0]
    qn = np.asarray(out_u.Q_net)
    assert dT[0] == 0.0 and qn[0] == 0.0
    assert dT[1] == 0.0 and qn[1] == 0.0
    # (ii) the un-iced saline cell keeps the FULL cooling tendency.
    expected = -50.0 / (cfg_u.rho_0 * cfg_u.c_sw * 1.0)
    assert dT[2] == pytest.approx(expected, rel=1e-12)
    assert qn[2] == pytest.approx(-50.0, rel=1e-12)


def test_run_omip_core2_freeze_scheme_wiring():
    """``--freeze-scheme`` drift-check (pattern: test_omip_prognostic_ice):
    the production OMIP runner sources its choices from eos, threads the
    scheme into the model configs of all three wired builders AND the
    under-ice relaxation, and fails loud when the scheme has no consumer."""
    import inspect
    from scripts.run import run_omip_core2 as R

    # #939 refactored the argparse setup out of main() into _build_arg_parser();
    # the freeze-scheme wiring (choices + fail-loud guards live in the parser,
    # the model-config threading lives in main()) now spans BOTH, so the
    # drift-check inspects both sources.
    src = inspect.getsource(R.main) + "\n" + inspect.getsource(R._build_arg_parser)
    # Choices come from the single eos source of truth (no copied literal set).
    assert "sorted(VALID_FREEZE_SCHEMES)" in src
    # Fail-loud guards: no consumer / the unwired cube builder.
    assert "has no consumer without" in src
    assert "--freeze-scheme is not wired for --grid cubed_sphere" in src
    # Model-config threading: tripole + latlon_bathy + mpas call sites.
    assert src.count("freezing=_freezing_ovr") == 3
    # The under-ice relaxation receives the LOCAL surface salinity + scheme.
    assert "freeze_scheme=args.freeze_scheme" in src
    # And the builders actually accept the override parameter.
    for b in (R.build_tripole, R.build_latlon_bathy, R.build_mpas_ocean):
        assert "freezing" in inspect.signature(b).parameters, b.__name__
