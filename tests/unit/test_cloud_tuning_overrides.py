"""Cloud-tuning CLI overrides (``--rh-crit`` / ``--q-c-diagnostic`` /
``--conv-cloud-max``) thread from ExperimentConfig through
``build_physics_pipeline`` into the hot-loop ``CloudConfig``.

These are the SW/LW knob for the coare3 moisture-driven planetary-albedo
overshoot (raise rh_crit / lower q_c_diagnostic => optically thinner / less
stratiform cloud).  Default ``None`` must leave the resolved pipeline at the
CloudConfig defaults (byte-identical to the validated path).
"""

from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.clouds.config import (
    CloudConfig,
    build_cloud_config,
)
from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import make_hybrid_levels

NLEV = 10


def _config(**over):
    microphysics = over.pop("microphysics", "none")
    return ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        radiation="none", convection="none", turbulence="none",
        cloud_scheme="sundqvist", microphysics=microphysics,
        **over,
    )


def test_cloudconfig_accepts_all_overrides() -> None:
    # The override target fields must exist on CloudConfig and round-trip.
    cc = CloudConfig(scheme="sundqvist", rh_crit=0.82,
                     q_c_diagnostic=3.0e-4, conv_cloud_max=0.18)
    assert (cc.rh_crit, cc.q_c_diagnostic, cc.conv_cloud_max) == (0.82, 3.0e-4, 0.18)


def test_build_cloud_config_defaults_are_byte_identical() -> None:
    # All-None overrides => exactly the CloudConfig defaults (the shared
    # helper the pipeline + clt diagnostic both use must not drift, #689).
    assert build_cloud_config("sundqvist") == CloudConfig(scheme="sundqvist")


def test_build_cloud_config_applies_overrides() -> None:
    cc = build_cloud_config(
        "xu_randall", convective_cloud=True, rh_crit=0.82,
        q_c_diagnostic=3.0e-4, conv_cloud_max=0.18,
    )
    assert cc.scheme == "xu_randall"
    assert cc.convective_cloud is True
    assert (cc.rh_crit, cc.q_c_diagnostic, cc.conv_cloud_max) == (0.82, 3.0e-4, 0.18)


def test_build_cloud_config_matches_pipeline_wiring() -> None:
    # The diagnostic path (build_cloud_config from ExperimentConfig fields)
    # must reproduce what build_physics_pipeline feeds compute_cloud_properties.
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(NLEV)
    cfg = _config(cloud_rh_crit=0.82, cloud_q_c_diagnostic=3.0e-4,
                  cloud_conv_cloud_max=0.18)
    pipe = build_physics_pipeline(grid, sigma, cfg)
    diag = build_cloud_config(
        cfg.cloud_scheme,
        convective_cloud=False,  # conv_precip unavailable in the diagnostic
        rh_crit=pipe._cloud_rh_crit,
        q_c_diagnostic=pipe._cloud_q_c_diagnostic,
        conv_cloud_max=pipe._cloud_conv_cloud_max,
    )
    assert diag.scheme == "sundqvist"
    assert (diag.rh_crit, diag.q_c_diagnostic, diag.conv_cloud_max) == (
        0.82, 3.0e-4, 0.18)


def test_pipeline_threads_cloud_overrides() -> None:
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(NLEV)
    pipe = build_physics_pipeline(
        grid, sigma,
        _config(cloud_rh_crit=0.82, cloud_q_c_diagnostic=3.0e-4,
                cloud_conv_cloud_max=0.18),
    )
    assert pipe._cloud_rh_crit == 0.82
    assert pipe._cloud_q_c_diagnostic == 3.0e-4
    assert pipe._cloud_conv_cloud_max == 0.18


def test_pipeline_default_overrides_none() -> None:
    # Default config => None => pipeline leaves CloudConfig at its defaults.
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(NLEV)
    pipe = build_physics_pipeline(grid, sigma, _config())
    assert pipe._cloud_rh_crit is None
    assert pipe._cloud_q_c_diagnostic is None
    assert pipe._cloud_conv_cloud_max is None


def test_cloudconfig_inhomogeneity_roundtrips() -> None:
    cc = CloudConfig(scheme="sundqvist", cloud_inhomogeneity_factor=0.7)
    assert cc.cloud_inhomogeneity_factor == 0.7
    # default is homogeneous (no change)
    assert CloudConfig(scheme="sundqvist").cloud_inhomogeneity_factor == 1.0


def test_build_cloud_config_applies_inhomogeneity() -> None:
    cc = build_cloud_config("sundqvist", cloud_inhomogeneity_factor=0.6)
    assert cc.cloud_inhomogeneity_factor == 0.6
    # all-None => default 1.0 (byte-identical)
    assert build_cloud_config("sundqvist").cloud_inhomogeneity_factor == 1.0


def test_pipeline_threads_inhomogeneity() -> None:
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(NLEV)
    pipe = build_physics_pipeline(
        grid, sigma, _config(cloud_inhomogeneity_factor=0.7))
    assert pipe._cloud_inhomogeneity_factor == 0.7
    assert build_physics_pipeline(
        grid, sigma, _config())._cloud_inhomogeneity_factor is None


def test_inhomogeneity_scales_radiative_water_path() -> None:
    """chi linearly scales the LWP/IWP fed to radiation (Cahalan plane-parallel
    correction): chi=0.5 halves the path vs the homogeneous chi=1.0."""
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        compute_cloud_properties)
    from legoesm import constants

    ncol, nlev = 2, NLEV
    p_s = 1.0e5
    sh = jnp.linspace(0.0, 1.0, nlev + 1)
    sf = 0.5 * (sh[:-1] + sh[1:])
    p_full = jnp.broadcast_to((sf * p_s)[None, :], (ncol, nlev))
    dp = jnp.broadcast_to(((sh[1:] - sh[:-1]) * p_s)[None, :], (ncol, nlev))
    T = jnp.full((ncol, nlev), 280.0)
    q_v = jnp.full((ncol, nlev), 8e-3)
    q_cloud = jnp.full((ncol, nlev), 2e-4)     # explicit prognostic cloud water
    q_ice = jnp.zeros((ncol, nlev))

    def _lwp(chi):
        cfg = CloudConfig(scheme="sundqvist", cloud_inhomogeneity_factor=chi)
        out = compute_cloud_properties(T=T, p_full=p_full, q_v=q_v, dp=dp,
                                       config=cfg, q_cloud=q_cloud, q_ice=q_ice)
        return jnp.asarray(out.lwp)

    lwp_full = _lwp(1.0)
    lwp_half = _lwp(0.5)
    assert float(jnp.max(lwp_full)) > 0.0
    # chi=0.5 halves the radiative water path, everywhere
    import numpy as np
    np.testing.assert_allclose(np.asarray(lwp_half), 0.5 * np.asarray(lwp_full),
                               rtol=1e-6, atol=1e-20)


def test_pipeline_threads_nc_default_and_conv_cloud_coeff() -> None:
    """The two levers wired 2026-08-02 for the calibration campaign: both had a
    ``__param_spec__`` entry but no flat ExperimentConfig scalar, so no member
    could vary them.  ``None`` must still leave the pipeline at the defaults."""
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(NLEV)
    pipe = build_physics_pipeline(
        grid, sigma,
        _config(cloud_Nc_default=4.0e7, cloud_conv_cloud_coeff=0.12))
    assert pipe._cloud_Nc_default == 4.0e7
    assert pipe._cloud_conv_cloud_coeff == 0.12
    pipe_def = build_physics_pipeline(grid, sigma, _config())
    assert pipe_def._cloud_Nc_default is None
    assert pipe_def._cloud_conv_cloud_coeff is None


def test_nc_default_sets_the_specified_droplet_psd_radius() -> None:
    """``Nc_default`` is the SW/albedo lever, not a dead field: under a
    specified-Nc double-moment run (the droplet-number tracer left at 0, i.e.
    morrison with predict_Nc=False) it is what the M2005 gamma PSD sees, and the
    Morrison liquid effective radius scales as ``N_c^(-1/3)`` at fixed water.

    Sign convention: MORE droplets at the same condensate => SMALLER droplets =>
    a BRIGHTER cloud.  So raising Nc_default must LOWER r_eff_liq.  The marine
    value (4e7) must therefore give a LARGER radius than the continental default
    (1e8) applied globally today."""
    import jax.numpy as jnp
    import numpy as np
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        compute_cloud_properties)

    ncol, nlev = 2, NLEV
    p_s = 1.0e5
    sh = jnp.linspace(0.0, 1.0, nlev + 1)
    sf = 0.5 * (sh[:-1] + sh[1:])
    p_full = jnp.broadcast_to((sf * p_s)[None, :], (ncol, nlev))
    dp = jnp.broadcast_to(((sh[1:] - sh[:-1]) * p_s)[None, :], (ncol, nlev))
    T = jnp.full((ncol, nlev), 285.0)          # warm => liquid PSD branch
    q_v = jnp.full((ncol, nlev), 8e-3)
    q_cloud = jnp.full((ncol, nlev), 5e-4)
    q_ice = jnp.zeros((ncol, nlev))
    # Specified-Nc: the prognostic droplet-number slot stays at 0, which is the
    # `n_cloud <= 1.0` branch that falls back to config.Nc_default.
    n_cloud = jnp.zeros((ncol, nlev))
    # Overcast, so the specified IN-CLOUD Nc pairs with an in-cloud q_c without
    # the cf^(-1/3) reconstruction; a broken layer drives the PSD into its
    # LAMMIN clip, which would mask the Nc response behind a constant.
    # ``clubb_cf_override_p_min_pa`` must be zeroed or the override is applied
    # only below 700 hPa and the levels above stay at the RH fraction (~0 here).
    cf_one = jnp.ones((ncol, nlev))

    def _r_eff(nc0):
        out = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="sundqvist", Nc_default=nc0,
                               clubb_cf_override_p_min_pa=0.0,
                               clubb_cf_override_ramp_pa=0.0),
            q_cloud=q_cloud, q_ice=q_ice, n_cloud=n_cloud,
            cloud_fraction_override=cf_one)
        return np.asarray(out.r_eff_liq)

    r_marine = _r_eff(4.0e7)
    r_continental = _r_eff(1.0e8)   # the current global default
    assert np.all(r_marine > r_continental), (
        "raising the specified droplet number must SHRINK the droplets")
    # Nc^(-1/3) scaling, up to the Martin pgam shape drift over the range.
    ratio = float(np.mean(r_marine / r_continental))
    assert ratio == pytest.approx((1.0e8 / 4.0e7) ** (1.0 / 3.0), rel=0.15)


def test_conv_cloud_coeff_scales_convective_cover() -> None:
    """``conv_cloud_coeff`` is the Slingo cloud-amount SLOPE and must move the
    cover linearly BELOW the ``conv_cloud_max`` cap — the reason the cap alone
    is not a usable lever: at the production defaults (coeff 0.04, cap 0.15) the
    cap does not bind until ``ln(1+P/P0) > 3.75``, i.e. ~42x the P0 reference
    precip rate, so a member that can only move the cap changes nothing over
    most of the tropics."""
    import jax.numpy as jnp
    import numpy as np
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        convective_cloud_fraction)

    ncol, nlev = 2, NLEV
    p_s = 1.0e5
    sh = jnp.linspace(0.0, 1.0, nlev + 1)
    sf = 0.5 * (sh[:-1] + sh[1:])
    p_full = jnp.broadcast_to((sf * p_s)[None, :], (ncol, nlev))
    base = CloudConfig(scheme="sundqvist", convective_cloud=True)
    # ~3 mm/day: ln(1 + P/P0) ~ 1.4 => cf_conv ~ 0.056 at coeff 0.04, well
    # under the 0.15 cap, so the response to the slope is un-clipped.
    conv_precip = jnp.full((ncol,), 3.0 * 1.1574e-5)

    cf_1x = np.asarray(convective_cloud_fraction(conv_precip, p_full, base))
    cf_2x = np.asarray(convective_cloud_fraction(
        conv_precip, p_full, base._replace(conv_cloud_coeff=0.08)))
    assert float(cf_1x.max()) > 0.0, "the deck must carry some cover"
    assert float(cf_2x.max()) < base.conv_cloud_max, "must stay under the cap"
    np.testing.assert_allclose(cf_2x, 2.0 * cf_1x, rtol=1e-6, atol=1e-12)
    # The cap is inert here: doubling it changes nothing at this precip rate.
    cf_cap2x = np.asarray(convective_cloud_fraction(
        conv_precip, p_full, base._replace(conv_cloud_max=0.30)))
    np.testing.assert_allclose(cf_cap2x, cf_1x, rtol=1e-12, atol=1e-14)


def test_run_coupled_exposes_and_forwards_the_two_new_cloud_flags() -> None:
    """run_coupled is the second affected driver (repo rule: a new user-tunable
    field gets a CLI flag in EVERY affected run script).

    It builds its ``ExperimentConfig`` inline in ``main()`` rather than in a
    testable ``build_config_from_args``, so the round-trip is checked in two
    parts: the parser produces the dest (default None => byte-identical), and
    ``main`` — the symbol that actually runs, not a delegating wrapper — passes
    the dest through as the matching keyword."""
    import inspect

    from scripts.run.run_coupled import build_parser, main

    args = build_parser().parse_args(
        ["--cloud-nc-default", "4e7", "--cloud-conv-cloud-coeff", "0.12"])
    assert args.cloud_Nc_default == pytest.approx(4.0e7)
    assert args.cloud_conv_cloud_coeff == pytest.approx(0.12)
    defaults = build_parser().parse_args([])
    assert defaults.cloud_Nc_default is None
    assert defaults.cloud_conv_cloud_coeff is None
    # ExperimentConfig must accept what the parser produces.
    ExperimentConfig(cloud_Nc_default=args.cloud_Nc_default,
                     cloud_conv_cloud_coeff=args.cloud_conv_cloud_coeff
                     ).validate_strict()
    src = inspect.getsource(main)
    assert "cloud_Nc_default=args.cloud_Nc_default" in src
    assert "cloud_conv_cloud_coeff=args.cloud_conv_cloud_coeff" in src


def test_pipeline_threads_subgrid_autoconv() -> None:
    # --subgrid-autoconv (#613) must reach the hot-loop MorrisonConfig.
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(NLEV)
    pipe = build_physics_pipeline(
        grid, sigma,
        _config(microphysics="morrison", subgrid_autoconversion=True),
    )
    assert pipe.micro_config.subgrid_autoconversion is True


def test_subgrid_autoconv_rejects_non_morrison() -> None:
    # Fail loudly (not a silent no-op) when the flag is set on a scheme
    # that has no sub-grid warm-rain closure.
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(NLEV)
    with pytest.raises(ValueError, match="subgrid_autoconversion"):
        build_physics_pipeline(
            grid, sigma,
            _config(microphysics="kessler", subgrid_autoconversion=True),
        )
