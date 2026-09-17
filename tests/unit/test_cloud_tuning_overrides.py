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


def test_build_cloud_config_accepts_every_kwarg_the_pipeline_passes():
    """The builder and its only caller must agree on the argument list.

    ``physics_pipeline.compute_radiation_core`` calls ``build_cloud_config``
    with a fixed keyword list, unconditionally.  From 2026-08-10 (8e76f1fc5)
    until this test was written, that list contained ``conv_cloud_coeff`` and
    the builder's signature did not, so EVERY run that reached the radiation
    branch with a cloud scheme died with

        TypeError: build_cloud_config() got an unexpected keyword argument
        'conv_cloud_coeff'

    Nothing caught it because the builder's own tests only pass the arguments
    the builder declares — which is exactly the set that cannot detect a
    missing one.  This test reads the caller's keyword list out of the source
    and asserts the builder accepts all of it, so the next parameter added on
    one side and forgotten on the other goes red here instead of in a GPU job.
    """
    import ast
    import inspect
    from pathlib import Path

    from legoesm.atmosphere.physics.clouds.config import build_cloud_config

    pipeline = Path(inspect.getsourcefile(
        __import__("legoesm.driver.physics_pipeline", fromlist=["x"])))
    tree = ast.parse(pipeline.read_text())
    called = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "build_cloud_config"):
            called |= {kw.arg for kw in node.keywords if kw.arg is not None}
    assert called, "no build_cloud_config call found — has it been renamed?"

    accepted = set(inspect.signature(build_cloud_config).parameters)
    missing = sorted(called - accepted)
    assert not missing, (
        f"physics_pipeline passes {missing} to build_cloud_config, which does "
        f"not accept them; every radiation step with clouds would raise "
        f"TypeError. Builder accepts: {sorted(accepted)}")


def test_conv_cloud_coeff_override_actually_reaches_the_config():
    """NON-VACUITY: the passthrough must change the value, not just be accepted."""
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    base = build_cloud_config("sundqvist")
    tuned = build_cloud_config("sundqvist", conv_cloud_coeff=0.02)
    assert base.conv_cloud_coeff != 0.02
    assert tuned.conv_cloud_coeff == 0.02
