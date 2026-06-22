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
    # default None => unchanged (0.0)
    cfg0 = ExperimentConfig(turbulence="holtslag_boville")
    _fn0, tc0 = _resolve_turbulence(cfg0)
    assert tc0.surface.gustiness_w_zi == 0.0


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


def test_convective_gustiness_raises_calm_unstable_flux() -> None:
    """COARE convective gustiness (opt-in) must boost the latent flux over a calm
    but convectively-unstable warm ocean, and the default (off) must be
    byte-identical (OMIP/forward paths unchanged)."""
    import jax.numpy as jnp
    from legoesm.core.bulk_flux import compute_most_fluxes

    f = lambda v: jnp.full((8,), v)  # noqa: E731
    args = dict(u_rel=f(0.5), v_rel=f(0.0), T_atm=f(290.0), q_atm=f(0.010),
                T_sfc=f(302.0), q_sfc=f(0.026), rho=f(1.1), scheme="coare3")
    _tx, _ty, _sh0, lh_off, _u0 = compute_most_fluxes(**args, gustiness_w_zi=0.0)
    _tx, _ty, _sh1, lh_on, _u1 = compute_most_fluxes(**args, gustiness_w_zi=600.0)
    assert jnp.all(jnp.isfinite(lh_on))
    assert float(lh_on.mean()) > float(lh_off.mean())   # gustiness => more evap
    # default param == off (byte-identical)
    _tx, _ty, _shd, lh_def, _ud = compute_most_fluxes(**args)
    assert jnp.allclose(lh_def, lh_off)


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
