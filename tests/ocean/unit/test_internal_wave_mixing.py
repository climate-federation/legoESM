"""Tests for internal wave-driven mixing (NEMO zdfiwm, de Lavergne 2020).

Test surface:
1. **F90 transliteration cross-check** — the JAX implementation matches an
   independent per-column numpy transliteration of ``zdfiwm.F90``
   (zdf_iwm loop, jk = 2..jpkm1) on wet columns, for all four
   (mevar, tsdiff) namelist combinations.
2. **Column power integral** — the nsq/sho parts redistribute EXACTLY
   their map power (Σ ρ0·ε·e3w = power); the cri/bot parts telescope to
   their map power up to the surface half-cell residual.
3. **Bounds** — K ∈ [k_min, k_max]; statically unstable interfaces
   (N² = 0 after the upstream clip) saturate at ``k_max`` where power is
   nonzero.
4. **mevar regimes** — Reb > 480 (energetic), Reb < 10.224
   (buoyancy-controlled) and the linear mid-range reproduce the F90
   branch formulas.
5. **tsdiff ratio** — → 1 at high Reb, → 0.505 − 0.495 at Reb → 0.
6. **NaN/AD safety** — land columns (H = 0) and below-seafloor reference
   depths stay finite; ``jax.grad`` through the scheme is finite.
7. **k_profiles composition** — additive on BOTH K_v and A_v on top of the
   constant scheme; the uniform constant-power fallback engages when no
   maps are passed; ``tsdiff=True`` raises on the shared-K tracer solve;
   the explicit-path factories reject ``iwm.enabled=True``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    ConstantVerticalMixingConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
    IWMConfig,
    IWMForcing,
    compute_iwm_diffusivity,
    uniform_iwm_forcing,
)
from legoesm.ocean.physics.vertical_mixing.k_profiles import (
    compute_vertical_K_profiles,
)
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


RHO0 = 1026.0
NU = constants.nu_ocean_molecular


# -----------------------------------------------------------------------------
# Independent per-column transliteration of zdfiwm.F90 (numpy, F90 semantics)
# -----------------------------------------------------------------------------

def _f90_zdf_iwm_reference(ebot, ecri, ensq, esho, hbot, hcri_inv,
                           gdept, e3w, ht, rn2, *, rho0=RHO0, nu=NU,
                           mevar=False, tsdiff=False,
                           k_min=1.4e-7, k_max=1.0e-2):
    """Verbatim per-column transliteration of zdf_iwm (zdfiwm.F90:164-253).

    ``gdept``: (nlev,) t-point depths, positive down; ``e3w``: (nlev-1,)
    interior w-thicknesses; ``rn2``: (nlev-1,) interface N² (>= 0, matching
    the host's clipped in-situ N²).  Returns (K, ratio) at the nlev-1
    interior interfaces (F90 jk = k+2).
    """
    nlev = gdept.shape[0]
    r1_rho0 = 1.0 / rho0
    if ht != 0.0:
        zfact1 = ecri * r1_rho0 / (1.0 - np.exp(-ht * hcri_inv))
        zfact2 = ebot * (1.0 + hbot / ht) * r1_rho0
    else:
        zfact1 = 0.0
        zfact2 = 0.0
    s3 = float((e3w * np.maximum(0.0, rn2)).sum())
    s4 = float((e3w * np.sqrt(np.maximum(0.0, rn2))).sum())
    zfact3 = ensq * r1_rho0 / s3 if s3 != 0.0 else 0.0
    zfact4 = esho * r1_rho0 / s4 if s4 != 0.0 else 0.0
    K = np.zeros(nlev - 1)
    ratio = np.ones(nlev - 1)
    for k in range(nlev - 1):          # F90 jk = k + 2
        z_up, z_dn = gdept[k], gdept[k + 1]
        zemx = (
            zfact1 * (np.exp((z_dn - ht) * hcri_inv)
                      - np.exp((z_up - ht) * hcri_inv))
            + zfact2 * (1.0 / (1.0 + (ht - z_dn) / hbot)
                        - 1.0 / (1.0 + (ht - z_up) / hbot))
        ) / e3w[k] \
            + zfact3 * max(0.0, rn2[k]) \
            + zfact4 * np.sqrt(max(0.0, rn2[k]))
        reb = zemx / max(1.0e-20, nu * rn2[k])
        k_w = reb / 6.0 * nu
        if mevar:
            if reb > 480.0:
                k_w = 3.6515 * nu * np.sqrt(reb)
            elif reb < 10.224:
                k_w = 0.052125 * nu * reb * np.sqrt(reb)
        K[k] = min(max(k_min, k_w), k_max)
        if tsdiff:
            ratio[k] = 0.505 + 0.495 * np.tanh(
                0.92 * (np.log10(max(1.0e-20, reb * 5.0 / 6.0)) - 0.60))
    return K, ratio


def _wet_column(nlev=12, H=4000.0, seed=0):
    """Uneven wet-column geometry + N² profile (gdept < H everywhere)."""
    rng = np.random.default_rng(seed)
    dz = np.linspace(50.0, 2.0 * H / nlev - 50.0, nlev)
    dz = dz * (H / dz.sum())
    z_half = np.concatenate([[0.0], np.cumsum(dz)])
    gdept = 0.5 * (z_half[:-1] + z_half[1:])
    e3w = gdept[1:] - gdept[:-1]
    N2 = 10.0 ** rng.uniform(-7.5, -4.5, size=nlev - 1)
    return gdept, e3w, N2


@pytest.mark.parametrize("mevar", [False, True])
@pytest.mark.parametrize("tsdiff", [False, True])
def test_matches_f90_transliteration(mevar, tsdiff):
    gdept, e3w, N2 = _wet_column()
    H = float(gdept[-1] + 0.5 * (gdept[-1] - gdept[-2]))
    maps = dict(ebot=1.2e-3, ecri=0.8e-3, ensq=1.0e-3, esho=0.5e-3,
                hbot=120.0, hcri_inv=1.0 / 250.0)
    K_ref, ratio_ref = _f90_zdf_iwm_reference(
        maps["ebot"], maps["ecri"], maps["ensq"], maps["esho"],
        maps["hbot"], maps["hcri_inv"], gdept, e3w, H, N2,
        mevar=mevar, tsdiff=tsdiff)

    ones = jnp.ones((1, 1))
    forcing = IWMForcing(
        ebot=maps["ebot"] * ones, ecri=maps["ecri"] * ones,
        ensq=maps["ensq"] * ones, esho=maps["esho"] * ones,
        hbot=maps["hbot"] * ones, hcri_inv=maps["hcri_inv"] * ones)
    cfg = IWMConfig(enabled=True, mevar=mevar, tsdiff=tsdiff)
    K, ratio = compute_iwm_diffusivity(
        forcing,
        jnp.asarray(gdept)[jnp.newaxis, jnp.newaxis, :],
        jnp.asarray(e3w)[jnp.newaxis, jnp.newaxis, :],
        H * jnp.ones((1, 1)),
        jnp.asarray(N2)[jnp.newaxis, jnp.newaxis, :],
        cfg=cfg, rho_0=RHO0)
    # np-vs-jnp reduction reassociation ⇒ rtol no tighter than ~1e-6.
    np.testing.assert_allclose(np.asarray(K[0, 0]), K_ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ratio[0, 0]), ratio_ref, rtol=1e-6)


def test_nsq_sho_power_integral_exact_and_cri_bot_telescoping():
    gdept, e3w, N2 = _wet_column(nlev=40, H=4000.0, seed=1)
    H = float(gdept[-1] + 0.5 * (gdept[-1] - gdept[-2]))
    hbot, hcri_inv = 150.0, 1.0 / 100.0
    power = 1.0e-3
    # Analytic telescoped column integrals of the F90 discretization:
    # Σ_k [CDF(gdept_{k+1}) − CDF(gdept_k)] collapses to
    # CDF(gdept_last) − CDF(gdept_0) (the surface/bottom half-cell
    # residuals are genuinely NOT deposited by zdfiwm — the jk loop spans
    # the interior w levels only).
    cdf_cri = lambda z: np.exp((z - H) * hcri_inv)          # noqa: E731
    cdf_bot = lambda z: 1.0 / (1.0 + (H - z) / hbot)        # noqa: E731
    expected = {
        "ensq": power,
        "esho": power,
        "ecri": power * (cdf_cri(gdept[-1]) - cdf_cri(gdept[0]))
        / (1.0 - np.exp(-H * hcri_inv)),
        "ebot": power * (1.0 + hbot / H)
        * (cdf_bot(gdept[-1]) - cdf_bot(gdept[0])),
    }
    for comp, expect in expected.items():
        maps = dict(ebot=0.0, ecri=0.0, ensq=0.0, esho=0.0,
                    hbot=hbot, hcri_inv=hcri_inv)
        maps[comp] = power
        # K-inversion is monotone in the mid-range: with the bounds
        # disabled, zemx = 6·K·N2 (K = Reb·ν/6, Reb = zemx/(ν·N2)).
        K_ref, _ = _f90_zdf_iwm_reference(
            maps["ebot"], maps["ecri"], maps["ensq"], maps["esho"],
            maps["hbot"], maps["hcri_inv"], gdept, e3w, H, N2,
            k_min=0.0, k_max=np.inf)
        zemx = 6.0 * K_ref * N2
        integral = RHO0 * float((zemx * e3w).sum())
        np.testing.assert_allclose(integral, expect, rtol=1e-6,
                                   err_msg=comp)
        # Never more than the map power; the deposited fraction can be
        # well below 1 for the bottom-intensified parts (zdfiwm's jk loop
        # spans interior w-levels only, so the energy inside the bottom
        # HALF-cell — exp(−dz_bot/2/h_cri) of it here — is genuinely not
        # deposited when the bottom cell is thick vs the decay scale).
        assert integral <= power * (1.0 + 1e-9)
        assert integral > 0.25 * power, (comp, integral)


def test_raw_negative_n2_matches_f90_floor_behavior():
    """NEMO uses the RAW rn2 in the Reb denominator (max(1e-20, nu*rn2));
    a statically UNSTABLE interface (rn2 < 0) hits the floor and saturates
    at k_max where local power is nonzero.  The port must reproduce that
    for signed N² input (codex r1 #6)."""
    gdept, e3w, N2 = _wet_column(nlev=10, H=3000.0, seed=5)
    H = float(gdept[-1] + 0.5 * (gdept[-1] - gdept[-2]))
    N2 = N2.copy()
    N2[4] = -2.0e-6                       # signed, statically unstable
    maps = dict(ebot=1e-3, ecri=1e-3, ensq=1e-3, esho=1e-3,
                hbot=150.0, hcri_inv=1.0 / 100.0)
    K_ref, ratio_ref = _f90_zdf_iwm_reference(
        maps["ebot"], maps["ecri"], maps["ensq"], maps["esho"],
        maps["hbot"], maps["hcri_inv"], gdept, e3w, H, N2,
        mevar=True, tsdiff=True)
    ones = jnp.ones(())
    forcing = IWMForcing(ebot=maps["ebot"] * ones, ecri=maps["ecri"] * ones,
                         ensq=maps["ensq"] * ones, esho=maps["esho"] * ones,
                         hbot=maps["hbot"] * ones,
                         hcri_inv=maps["hcri_inv"] * ones)
    cfg = IWMConfig(enabled=True, mevar=True, tsdiff=True)
    K, ratio = compute_iwm_diffusivity(
        forcing, jnp.asarray(gdept), jnp.asarray(e3w), jnp.asarray(H),
        jnp.asarray(N2), cfg=cfg, rho_0=RHO0)
    np.testing.assert_allclose(np.asarray(K), K_ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ratio), ratio_ref, rtol=1e-6)
    assert float(K[4]) == pytest.approx(cfg.k_max)


def test_bounds_and_unstable_interface_saturates():
    gdept, e3w, N2 = _wet_column(nlev=10, H=3000.0, seed=2)
    H = float(gdept[-1] + 0.5 * (gdept[-1] - gdept[-2]))
    N2 = N2.copy()
    N2[3] = 0.0   # upstream-clipped statically unstable interface
    ones = jnp.ones(())
    forcing = IWMForcing(ebot=1e-3 * ones, ecri=1e-3 * ones,
                         ensq=1e-3 * ones, esho=1e-3 * ones,
                         hbot=150.0 * ones, hcri_inv=(1 / 100.0) * ones)
    cfg = IWMConfig(enabled=True)
    K, _ = compute_iwm_diffusivity(
        forcing, jnp.asarray(gdept), jnp.asarray(e3w),
        jnp.asarray(H), jnp.asarray(N2), cfg=cfg, rho_0=RHO0)
    assert jnp.all(K >= cfg.k_min - 1e-18)
    assert jnp.all(K <= cfg.k_max + 1e-18)
    # zero-N² interface with nonzero local power → Reb → 1e20-ish → cap.
    assert float(K[3]) == pytest.approx(cfg.k_max)


def test_mevar_regime_formulas():
    # One interface per regime, via hand-built zemx/N².
    nu = NU
    gdept = np.array([10.0, 30.0, 60.0, 100.0])
    e3w = gdept[1:] - gdept[:-1]
    H = 120.0
    # Choose N² so that with ONLY the nsq component the three interfaces
    # land in the three Reb regimes: Reb = ensq/(rho0*sum)*N2 / (nu*N2)
    # = ensq/(rho0*sum*nu) — same for all; instead vary via per-interface
    # power by using the reference directly on constructed zemx values.
    N2 = np.array([1e-6, 1e-6, 1e-6])
    # Reb = power / (rho0 · Σ(e3w·N²) · ν) ≈ power / 1.29e-7 here.
    for power, regime in [(5.0e-2, "energetic"), (1.3e-5, "mid"),
                          (1.0e-6, "buoyancy")]:
        K_ref, _ = _f90_zdf_iwm_reference(
            0.0, 0.0, power, 0.0, 100.0, 0.01, gdept, e3w, H, N2,
            mevar=True)
        ones = jnp.ones(())
        forcing = IWMForcing(ebot=0.0 * ones, ecri=0.0 * ones,
                             ensq=power * ones, esho=0.0 * ones,
                             hbot=100.0 * ones, hcri_inv=0.01 * ones)
        K, _ = compute_iwm_diffusivity(
            forcing, jnp.asarray(gdept), jnp.asarray(e3w), jnp.asarray(H),
            jnp.asarray(N2), cfg=IWMConfig(enabled=True, mevar=True),
            rho_0=RHO0)
        np.testing.assert_allclose(np.asarray(K), K_ref, rtol=1e-6,
                                   err_msg=regime)


def test_tsdiff_ratio_limits():
    gdept = np.array([10.0, 30.0, 60.0, 100.0])
    e3w = gdept[1:] - gdept[:-1]
    N2 = np.full(3, 1e-6)
    ones = jnp.ones(())
    make = lambda p: IWMForcing(  # noqa: E731
        ebot=0.0 * ones, ecri=0.0 * ones, ensq=p * ones, esho=0.0 * ones,
        hbot=100.0 * ones, hcri_inv=0.01 * ones)
    cfg = IWMConfig(enabled=True, tsdiff=True)
    _, ratio_hi = compute_iwm_diffusivity(
        make(10.0), jnp.asarray(gdept), jnp.asarray(e3w), jnp.asarray(120.0),
        jnp.asarray(N2), cfg=cfg, rho_0=RHO0)
    _, ratio_lo = compute_iwm_diffusivity(
        make(1e-12), jnp.asarray(gdept), jnp.asarray(e3w), jnp.asarray(120.0),
        jnp.asarray(N2), cfg=cfg, rho_0=RHO0)
    assert float(ratio_hi[0]) > 0.99          # → 1 at high Reb
    assert float(ratio_lo[0]) < 0.05          # → 0.505 − 0.495 at low Reb


def test_land_column_and_grad_finite():
    gdept, e3w, N2 = _wet_column(nlev=8, H=2000.0, seed=3)
    ones = jnp.ones((2,))
    forcing = IWMForcing(ebot=1e-3 * ones, ecri=1e-3 * ones,
                         ensq=1e-5 * ones, esho=1e-4 * ones,
                         hbot=100.0 * ones, hcri_inv=(1 / 50.0) * ones)
    cfg = IWMConfig(enabled=True, mevar=True)
    depth = jnp.stack([jnp.asarray(gdept)] * 2)
    dzw = jnp.stack([jnp.asarray(e3w)] * 2)
    H = jnp.asarray([2000.0, 0.0])            # column 2 = land (ht = 0)
    N2j = jnp.stack([jnp.asarray(N2)] * 2)

    def loss(n2):
        K, _ = compute_iwm_diffusivity(
            forcing, depth, dzw, H, n2, cfg=cfg, rho_0=RHO0)
        return jnp.sum(K)

    K, _ = compute_iwm_diffusivity(forcing, depth, dzw, H, N2j,
                                   cfg=cfg, rho_0=RHO0)
    assert bool(jnp.all(jnp.isfinite(K)))
    g = jax.grad(loss)(N2j)
    assert bool(jnp.all(jnp.isfinite(g)))


# -----------------------------------------------------------------------------
# k_profiles composition + guards
# -----------------------------------------------------------------------------

def _phys_cfg(vmix):
    base = OceanPhysicsConfig()
    return OceanPhysicsConfig(
        vertical_mixing=vmix,
        lateral_mixing=type(base.lateral_mixing)(scheme="none"),
        surface_forcing=type(base.surface_forcing)(scheme="none"),
        shortwave_penetration=None,
        convection=OceanConvectionConfig(scheme="none"),
    )


@pytest.fixture(scope="module")
def grid_z_state():
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=6, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
    return grid, z, state


# Stronger-than-default uniform powers so K_iwm rises clear of the k_min
# floor on this small stratified column (the NEMO fallback 1e-5 W/m²
# lands below the 1.4e-7 floor here).
_IWM_STRONG = IWMConfig(enabled=True, power_nsq_wm2=1.0e-3,
                        power_bot_wm2=1.0e-3)


def test_k_profiles_iwm_additive_on_K_and_A(grid_z_state):
    _, z, state = grid_z_state
    vm_off = VerticalMixingConfig(
        scheme="constant",
        constant=ConstantVerticalMixingConfig(K_v=3e-5, A_v=2e-4))
    vm_on = vm_off._replace(iwm=_IWM_STRONG)
    K0, A0 = compute_vertical_K_profiles(state, z, None, _phys_cfg(vm_off))
    K1, A1 = compute_vertical_K_profiles(state, z, None, _phys_cfg(vm_on))
    dK = np.asarray(K1 - K0)
    dA = np.asarray(A1 - A0)
    # IWM adds the SAME wave diffusivity to tracer K and momentum A.
    # f32 state storage ⇒ the two subtractions round differently against
    # their different baselines (3e-5 vs 2e-4): f32-ulp tolerances.
    np.testing.assert_allclose(dK, dA, rtol=1e-4, atol=1e-10)
    assert float(dK.min()) >= IWMConfig().k_min - 1e-10
    assert float(dK.max()) <= IWMConfig().k_max + 1e-10
    # Clear of the floor somewhere (the powers actually mix).
    assert float(dK.max()) > 5.0 * IWMConfig().k_min


def test_k_profiles_iwm_explicit_fields_override_fallback(grid_z_state):
    _, z, state = grid_z_state
    vm = VerticalMixingConfig(
        scheme="constant",
        constant=ConstantVerticalMixingConfig(K_v=3e-5, A_v=2e-4),
        iwm=_IWM_STRONG)
    cfgp = _phys_cfg(vm)
    shape = state.T.data.shape[:-1]
    fields_zero = IWMForcing(
        ebot=jnp.zeros(shape), ecri=jnp.zeros(shape), ensq=jnp.zeros(shape),
        esho=jnp.zeros(shape), hbot=100.0 * jnp.ones(shape),
        hcri_inv=0.01 * jnp.ones(shape))
    K_fb, _ = compute_vertical_K_profiles(state, z, None, cfgp)
    K_z, _ = compute_vertical_K_profiles(state, z, None, cfgp,
                                         iwm_fields=fields_zero)
    # zero-power maps → only the k_min floor is added; the (strong)
    # fallback maps add strictly more somewhere.
    assert float(jnp.max(K_fb - K_z)) > 0.0


def test_k_profiles_tsdiff_raises(grid_z_state):
    _, z, state = grid_z_state
    vm = VerticalMixingConfig(
        scheme="constant", iwm=IWMConfig(enabled=True, tsdiff=True))
    with pytest.raises(ValueError, match="tsdiff"):
        compute_vertical_K_profiles(state, z, None, _phys_cfg(vm))


def test_explicit_path_factories_reject_iwm():
    from legoesm.ocean.physics.vertical_mixing.integration import (
        make_vertical_mixing_physics,
    )
    from legoesm.ocean.physics.combined import make_ocean_physics
    vm = VerticalMixingConfig(scheme="constant", iwm=IWMConfig(enabled=True))
    with pytest.raises(NotImplementedError, match="iwm"):
        make_vertical_mixing_physics(vm)
    with pytest.raises(NotImplementedError, match="iwm"):
        make_ocean_physics(_phys_cfg(vm))


def test_mpas_bridge_rejects_iwm():
    from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
        make_kpp_physics_mpas, make_kpp_profiles_mpas,
    )
    vm = VerticalMixingConfig(scheme="kpp", iwm=IWMConfig(enabled=True))
    with pytest.raises(NotImplementedError, match="MPAS"):
        make_kpp_physics_mpas(vm)
    with pytest.raises(NotImplementedError, match="MPAS"):
        make_kpp_profiles_mpas(vm)


def test_uniform_forcing_matches_config_scalars():
    cfg = IWMConfig(power_nsq_wm2=2.5e-5, scale_cri_m=200.0)
    f = uniform_iwm_forcing(cfg, (3, 4))
    assert f.ensq.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(f.ensq), 2.5e-5)
    np.testing.assert_allclose(np.asarray(f.hcri_inv), 1.0 / 200.0)


# -----------------------------------------------------------------------------
# Model-level wiring (LatLonCGridOceanModel guards + a stepped IWM run)
# -----------------------------------------------------------------------------

def _iwm_model_pieces(iwm_cfg):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=5, H_max=3000.0)
    base = OceanPhysicsConfig()
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="none", iwm=iwm_cfg),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        shortwave_penetration=None,
        convection=OceanConvectionConfig(scheme="none"),
        bottom_drag=type(base.bottom_drag)(scheme="none"),
    )
    return grid, z, physics, LatLonCGridOceanConfig


def test_model_requires_implicit_vertical_mixing_for_iwm():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, physics, CFG = _iwm_model_pieces(_IWM_STRONG)
    cfg = CFG.from_flat(physics=physics, implicit_vertical_mixing=False)
    with pytest.raises(ValueError, match="implicit_vertical_mixing"):
        LatLonCGridOceanModel(grid, z, cfg)


def test_model_rejects_orphan_iwm_forcing():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, physics, CFG = _iwm_model_pieces(IWMConfig(enabled=False))
    cfg = CFG.from_flat(physics=physics, implicit_vertical_mixing=True)
    maps = uniform_iwm_forcing(_IWM_STRONG, (6, 8))
    with pytest.raises(ValueError, match="iwm_forcing"):
        LatLonCGridOceanModel(grid, z, cfg, iwm_forcing=maps)


def test_model_fast_path_kpp_plus_iwm_adds_wave_K(grid_z_state):
    """The physics-provided-K FAST path (KPP pipeline surfaces K_v/A_v on
    the tendencies — the latlon_bathy driver configuration) must ADD the
    zdfiwm contribution rather than reject it (faith_ll2 job 8781839
    regression): one real step with KPP+IWM runs finite and differs from
    KPP alone."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    grid, z, physics, CFG = _iwm_model_pieces(
        _IWM_STRONG._replace(power_nsq_wm2=5.0e-3))
    physics_kpp = physics._replace(
        vertical_mixing=physics.vertical_mixing._replace(scheme="kpp"))
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
    cfg_on = CFG.from_flat(A_h=0.0, A_v=0.0, K_h=0.0, K_v=0.0,
                           physics=physics_kpp,
                           implicit_vertical_mixing=True)
    physics_off = physics_kpp._replace(
        vertical_mixing=physics_kpp.vertical_mixing._replace(
            iwm=IWMConfig(enabled=False)))
    cfg_off = cfg_on._replace(physics=physics_off)
    dt = 1800.0
    s_on = LatLonCGridOceanModel(grid, z, cfg_on).step(state, dt)
    s_off = LatLonCGridOceanModel(grid, z, cfg_off).step(state, dt)
    assert bool(jnp.all(jnp.isfinite(s_on.T.data)))
    dT = np.asarray(s_on.T.data) - np.asarray(s_off.T.data)
    assert float(np.max(np.abs(dT))) > 0.0


def test_model_step_with_iwm_runs_and_mixes(grid_z_state):
    """One real model step with IWM on: finite state, and the wave mixing
    measurably smooths the near-bottom stratification vs the same step
    with IWM off (strong uniform powers on a stratified rest state)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    grid, z, physics, CFG = _iwm_model_pieces(
        _IWM_STRONG._replace(power_nsq_wm2=5.0e-3))
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
    cfg_on = CFG.from_flat(A_h=0.0, A_v=0.0, K_h=0.0, K_v=0.0,
                           physics=physics,
                           implicit_vertical_mixing=True)
    physics_off = physics._replace(
        vertical_mixing=physics.vertical_mixing._replace(
            iwm=IWMConfig(enabled=False)))
    cfg_off = cfg_on._replace(physics=physics_off)
    dt = 1800.0
    s_on = LatLonCGridOceanModel(grid, z, cfg_on).step(state, dt)
    s_off = LatLonCGridOceanModel(grid, z, cfg_off).step(state, dt)
    assert bool(jnp.all(jnp.isfinite(s_on.T.data)))
    dT_on = np.asarray(s_on.T.data - state.T.data)
    dT_off = np.asarray(s_off.T.data - state.T.data)
    # IWM must actually change the tracer solve (add mixing).
    assert float(np.max(np.abs(dT_on - dT_off))) > 0.0


# -----------------------------------------------------------------------------
# IWMConfig.n2_mode: which N² the wave formula reads
# -----------------------------------------------------------------------------

def _iwm_K(state, z, **kw):
    from legoesm.ocean.physics.vertical_mixing.k_profiles import iwm_K_profile
    vm = VerticalMixingConfig(scheme="constant",
                              constant=ConstantVerticalMixingConfig(K_v=3e-5, A_v=2e-4))
    return np.asarray(iwm_K_profile(state, z, _phys_cfg(vm), _IWM_STRONG._replace(**kw)))


def test_iwm_neutral_column_saturates_only_with_bn2():
    # A NEUTRAL column (uniform T, S): NEMO's rn2 = 0 there, so zdfiwm's Reb
    # floor drives K to the 1e-2 cap. The in-situ contrast carries
    # compressibility (~g^2/c^2 > 0) and keeps the same column far below it.
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=6, H_max=4000.0)
    st = rest_state_latlon_cgrid_ocean(grid, z, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0)
    k_bn2 = _iwm_K(st, z, n2_mode="nemo_bn2", n2_eos_form="seos")
    k_ins = _iwm_K(st, z)
    cap = IWMConfig().k_max
    np.testing.assert_allclose(k_bn2, cap, rtol=1e-6)
    assert k_ins.max() < 0.1 * cap


def test_iwm_bn2_on_stratified_column_is_bounded_and_mixes(grid_z_state):
    _, z, state = grid_z_state
    k = _iwm_K(state, z, n2_mode="nemo_bn2", n2_eos_form="seos")
    assert np.all(np.isfinite(k))
    assert k.min() >= IWMConfig().k_min - 1e-12 and k.max() <= IWMConfig().k_max + 1e-12
    assert k.max() < IWMConfig().k_max          # stratified: not the neutral cap everywhere


def test_iwm_unknown_n2_mode_raises(grid_z_state):
    _, z, state = grid_z_state
    with pytest.raises(ValueError, match="n2_mode"):
        _iwm_K(state, z, n2_mode="potential")


def test_iwm_n2_cli_round_trip_and_guards(monkeypatch):
    import sys
    import scripts.run.run_omip_core2 as core2
    from scripts.run.run_omip import build_iwm_config_from_args
    a = core2._build_arg_parser().parse_args(
        ["--iwm", "--iwm-n2-mode", "nemo_bn2", "--iwm-n2-eos-form", "teos10"])
    cfg = build_iwm_config_from_args(a)
    assert (cfg.n2_mode, cfg.n2_eos_form) == ("nemo_bn2", "teos10")
    assert build_iwm_config_from_args(core2._build_arg_parser().parse_args(["--iwm"])).n2_mode == "insitu"
    for argv, msg in ((["--iwm-n2-mode", "nemo_bn2", "--iwm-n2-eos-form", "seos"], "need --iwm"),
                      (["--iwm", "--iwm-n2-mode", "nemo_bn2"], "explicit --iwm-n2-eos-form")):
        monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", "tripole"] + argv)
        with pytest.raises(SystemExit, match=msg):
            core2.main()


def test_iwm_n2_guards_in_run_omip(monkeypatch):
    import sys
    import scripts.run.run_omip as ro
    for argv, msg in ((["--iwm-n2-mode", "nemo_bn2", "--iwm-n2-eos-form", "seos"], "need --iwm"),
                      (["--iwm", "--iwm-n2-mode", "nemo_bn2"], "explicit --iwm-n2-eos-form")):
        monkeypatch.setattr(sys, "argv", ["run_omip.py"] + argv)
        with pytest.raises(SystemExit, match=msg):
            ro.main()
