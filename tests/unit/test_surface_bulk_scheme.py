"""Surface bulk-flux scheme (``--surface-bulk-scheme``) wiring.

The experiment-level ``ExperimentConfig.surface_bulk_scheme`` selects the
atmosphere surface-layer bulk-flux algorithm: ``"constant"`` (neutral transfer
coefficients, no convective gustiness — the legacy default) vs the
stability-dependent MOST schemes ``"coare3"`` / ``"large_yeager"`` (which add the
free-convection velocity scale w*, the fix for anemic evaporation over a calm,
convectively-unstable warm ocean).

``physics_pipeline._resolve_turbulence`` must propagate the selection into the
turbulence scheme's ``SurfaceLayerConfig``; the default ``"constant"`` must leave
the resolved config untouched (byte-identical legacy behaviour).  The matching
coupler-side ocean-tile scheme (``CouplerConfig.bulk_scheme``) is wired by
``run_coupled`` for interface energy consistency.
"""

from __future__ import annotations

import pytest

from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import _resolve_turbulence


@pytest.mark.parametrize("scheme", ["coare3", "large_yeager"])
def test_resolve_turbulence_propagates_surface_bulk_scheme(scheme: str) -> None:
    cfg = ExperimentConfig(
        turbulence="holtslag_boville", surface_bulk_scheme=scheme,
    )
    _fn, turb_config = _resolve_turbulence(cfg)
    assert turb_config is not None
    assert turb_config.surface.bulk_scheme == scheme


def test_resolve_turbulence_threads_gustiness_zi() -> None:
    # surface_gustiness_zi must reach the turbulence scheme's SurfaceLayerConfig.
    cfg = ExperimentConfig(
        turbulence="holtslag_boville", surface_bulk_scheme="coare3",
        surface_gustiness_zi=600.0,
    )
    _fn, turb_config = _resolve_turbulence(cfg)
    assert turb_config.surface.gustiness_w_zi == 600.0
    # default None => SurfaceLayerConfig default None (scheme-native: the
    # kernel resolves coare3 -> 600 m / AeroBulk parity, others -> off)
    cfg0 = ExperimentConfig(turbulence="holtslag_boville")
    _fn0, tc0 = _resolve_turbulence(cfg0)
    assert tc0.surface.gustiness_w_zi is None


def test_resolve_turbulence_default_constant_unchanged() -> None:
    # Default surface_bulk_scheme="constant" => resolved config untouched.
    cfg = ExperimentConfig(turbulence="holtslag_boville")
    _fn, turb_config = _resolve_turbulence(cfg)
    assert turb_config.surface.bulk_scheme == "constant"


def test_resolve_turbulence_none_scheme_is_noop() -> None:
    # turbulence='none' => (None, None) regardless of surface_bulk_scheme;
    # the override guard must tolerate a None turb_config.
    cfg = ExperimentConfig(turbulence="none", surface_bulk_scheme="coare3")
    fn, turb_config = _resolve_turbulence(cfg)
    assert fn is None and turb_config is None


def test_coupler_config_accepts_matching_scheme() -> None:
    # run_coupled keeps the coupler ocean-tile scheme consistent with the
    # atmosphere; the field must exist and round-trip.
    from legoesm.coupler.config import CouplerConfig

    ccfg = CouplerConfig(bulk_scheme="coare3")
    assert ccfg.bulk_scheme == "coare3"


# ---- slab/two-layer ocean heat-budget side (interface energy consistency) ----

def _ocean_forcing(T_lowest: float, q_lowest: float, u: float = 2.0):
    import jax.numpy as jnp
    from legoesm.core.coupling_fields import AtmToSurface

    shape = (6, 4, 4)
    z = jnp.zeros(shape)
    return AtmToSurface(
        sw_down=jnp.full(shape, 200.0), lw_down=jnp.full(shape, 350.0),
        precip_total=z, precip_snow=z,
        T_lowest=jnp.full(shape, T_lowest), q_lowest=jnp.full(shape, q_lowest),
        u_lowest=jnp.full(shape, u), v_lowest=z,
        p_lowest=jnp.full(shape, 1.0e5), p_surface=jnp.full(shape, 1.0e5),
        rho_lowest=jnp.full(shape, 1.15), cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.array(400.0), has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(0.0),
    )


def test_ocean_coare3_enhances_unstable_latent_flux() -> None:
    """Over a warm ocean under cold, calm air (deeply unstable), the MOST scheme's
    convective gustiness (w*) must give a LARGER latent flux than the constant
    neutral scheme — the evaporation-deficit fix this option targets."""
    import jax.numpy as jnp
    from legoesm.ocean.simple_ocean import (
        SimpleOceanConfig, _ocean_turbulent_fluxes,
    )
    from legoesm.thermo import saturation_mixing_ratio

    T_sfc = jnp.full((6, 4, 4), 300.0)                 # warm ocean
    # cold, calm air (T_air well below SST => strongly unstable surface layer)
    fc = _ocean_forcing(T_lowest=288.0, q_lowest=8e-3, u=2.0)  # const-ok: test air temp [K]
    q_sfc = saturation_mixing_ratio(T_sfc, fc.p_surface)
    _sh_c, lh_const = _ocean_turbulent_fluxes(
        T_sfc, q_sfc, fc, SimpleOceanConfig(bulk_scheme="constant"))
    _sh_m, lh_coare = _ocean_turbulent_fluxes(
        T_sfc, q_sfc, fc, SimpleOceanConfig(bulk_scheme="coare3"))
    assert jnp.all(jnp.isfinite(lh_coare))
    assert float(lh_coare.mean()) > float(lh_const.mean())


def test_coupler_ocean_tile_gustiness_raises_latent_flux() -> None:
    """The coupler ocean tile (``ocean_tile_response``) drives the 3D-ocean
    q_net.  Over a calm warm ocean under unstable air, enabling the convective
    gustiness BL depth (``CouplerConfig.gustiness_w_zi``) must raise the tile
    latent heat flux vs gustiness_w_zi=0 — keeping the air-sea interface
    energy-consistent with the atmosphere surface layer (the 3D-ocean
    cold-collapse fix; cmip_air_sea_decoupling)."""
    import jax.numpy as jnp
    from legoesm.coupler.config import CouplerConfig
    from legoesm.coupler.coupler import ocean_tile_response

    fc = _ocean_forcing(T_lowest=288.0, q_lowest=8e-3, u=2.0)  # const-ok: test air temp [K]
    sst = jnp.full((6, 4, 4), 300.0)                  # warm calm ocean
    u_o = v_o = jnp.zeros((6, 4, 4))
    base = CouplerConfig(bulk_scheme="coare3", gustiness_w_zi=0.0)
    gust = CouplerConfig(bulk_scheme="coare3", gustiness_w_zi=600.0)
    r_off = ocean_tile_response(fc, sst, u_o, v_o, base)
    r_on = ocean_tile_response(fc, sst, u_o, v_o, gust)
    assert jnp.all(jnp.isfinite(r_on.lhflx))
    assert float(r_on.lhflx.mean()) > float(r_off.lhflx.mean())


@pytest.mark.parametrize("scheme", ["coare3", "large_yeager"])
def test_compute_most_fluxes_float32_carry_stable(scheme: str) -> None:
    """Regression: MOST under float32 inputs (the atmosphere coupled path) must
    keep its fori_loop carry dtype-stable.  Before the fix a float64 physical
    constant promoted a carry leaf float32->float64 mid-loop, raising
    'scan body ... carry input and carry output must have equal types'.  Wrapped
    in jit so the fori_loop equal-types invariant is actually enforced."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.bulk_flux import compute_most_fluxes

    f32 = lambda v: jnp.full((16,), v, dtype=jnp.float32)  # noqa: E731
    fn = jax.jit(lambda **kw: compute_most_fluxes(**kw, scheme=scheme))
    tau_x, tau_y, sh, lh, ust = fn(
        u_rel=f32(4.0), v_rel=f32(-2.0), T_atm=f32(288.0), q_atm=f32(8e-3),
        T_sfc=f32(300.0), q_sfc=f32(2.0e-2), rho=f32(1.15),
    )
    for arr in (tau_x, tau_y, sh, lh, ust):
        assert jnp.all(jnp.isfinite(arr))
    assert float(lh.mean()) > 0.0  # evaporation upward over a warm ocean


def test_scheme_native_gustiness_defaults() -> None:
    """gustiness_w_zi=None resolves scheme-natively (AeroBulk parity):
    coare3 -> built-in 600 m gustiness; large_yeager -> off."""
    import jax.numpy as jnp
    from legoesm.core.bulk_flux import compute_most_fluxes
    args = dict(
        u_rel=jnp.array([0.5]), v_rel=jnp.array([0.0]),
        T_atm=jnp.array([298.0]), q_atm=jnp.array([0.012]),
        T_sfc=jnp.array([302.0]), q_sfc=jnp.array([0.025]),
        rho=jnp.array([1.15]),
    )
    for scheme, zi_native in (("coare3", 600.0), ("large_yeager", 0.0)):
        default = compute_most_fluxes(**args, scheme=scheme)
        explicit = compute_most_fluxes(**args, scheme=scheme,
                                       gustiness_w_zi=zi_native)
        for a, b in zip(default, explicit):
            assert jnp.allclose(a, b), scheme
    # coare3 default therefore beats coare3 with gustiness forced off at calm
    lh_default = compute_most_fluxes(**args, scheme="coare3")[3]
    lh_off = compute_most_fluxes(**args, scheme="coare3",
                                 gustiness_w_zi=0.0)[3]
    assert float(lh_default[0]) > float(lh_off[0])


def test_convective_gustiness_raises_calm_unstable_flux() -> None:
    """COARE convective gustiness must boost the latent flux over a calm but
    convectively-unstable warm ocean; the coare3 default is scheme-native ON
    (600 m, AeroBulk parity), explicit 0.0 disables."""
    import jax.numpy as jnp
    from legoesm.core.bulk_flux import compute_most_fluxes

    f = lambda v: jnp.full((8,), v)  # noqa: E731
    args = dict(u_rel=f(0.5), v_rel=f(0.0), T_atm=f(290.0), q_atm=f(0.010),
                T_sfc=f(302.0), q_sfc=f(0.026), rho=f(1.1), scheme="coare3")
    _tx, _ty, _sh0, lh_off, _u0 = compute_most_fluxes(**args, gustiness_w_zi=0.0)
    _tx, _ty, _sh1, lh_on, _u1 = compute_most_fluxes(**args, gustiness_w_zi=600.0)
    assert jnp.all(jnp.isfinite(lh_on))
    assert float(lh_on.mean()) > float(lh_off.mean())   # gustiness => more evap
    # default param (None) == scheme-native ON for coare3 (AeroBulk zi0=600)
    _tx, _ty, _shd, lh_def, _ud = compute_most_fluxes(**args)
    assert jnp.allclose(lh_def, lh_on)


def test_compute_most_fluxes_2m_diagnostic() -> None:
    """The opt-in 2m air-temperature diagnostic must sit between the lowest model
    level and the (warmer) surface, and the default (off) must keep the legacy
    5-tuple return (byte-identical for all existing callers)."""
    import jax.numpy as jnp
    from legoesm.core.bulk_flux import compute_most_fluxes

    f = lambda v: jnp.full((8,), v)  # noqa: E731
    args = dict(u_rel=f(4.0), v_rel=f(0.0), T_atm=f(289.0), q_atm=f(0.010),
                T_sfc=f(301.0), q_sfc=f(0.025), rho=f(1.15), scheme="coare3")
    out5 = compute_most_fluxes(**args)
    assert len(out5) == 5                       # default return unchanged
    out6 = compute_most_fluxes(**args, return_2m=True)
    assert len(out6) == 6
    T2 = out6[5]
    # 2 m air over a warm ocean: between the lowest level (289) and SST (301),
    # and WARMER than the lowest level (the diagnostic the raw "tas" proxy misses)
    assert jnp.all(T2 > 289.0) and jnp.all(T2 < 301.0)


def test_ocean_unknown_bulk_scheme_raises() -> None:
    # Dispatch hardening (CLAUDE.md): an unknown scheme must raise, not silently
    # fall through to a default.
    import jax.numpy as jnp
    from legoesm.ocean.simple_ocean import (
        SimpleOceanConfig, _ocean_turbulent_fluxes,
    )

    T_sfc = jnp.full((6, 4, 4), 300.0)
    fc = _ocean_forcing(T_lowest=288.0, q_lowest=8e-3)  # const-ok: test air temp [K]
    with pytest.raises(ValueError, match="bulk_scheme"):
        _ocean_turbulent_fluxes(
            T_sfc, jnp.full_like(T_sfc, 0.02), fc,
            SimpleOceanConfig(bulk_scheme="bogus"))
