"""Substep schedule, no-op path and checkpoint-level equivalence."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.wave.config import (
    WaveConfig,
    WaveGSEConfig,
    WavePhysicsConfig,
    WaveTimeConfig,
)
from legoesm.wave.grid import prepare_env, wave_cell_area
from legoesm.wave.physics.integrator import (
    model_advance,
    prepare_integration_schedule,
    simple_stot_integrator,
)


def test_schedule_single_step_when_all_components_hourly():
    schedule = prepare_integration_schedule(
        3600, 3600, 3600, 3600, 3600, 3600, jnp.float32, True)
    assert schedule["num_steps"] == 1
    assert bool(schedule["should_integrate_x"][0])
    assert bool(schedule["should_integrate_stot"][0])


def test_simultaneous_spatial_events_count_once():
    schedule = prepare_integration_schedule(
        3600, 1800, 1800, 0, 0, 0, jnp.float32, True)
    np.testing.assert_array_equal(
        np.asarray(schedule["yfirst_per_step"]), [True, False])
    resumed = prepare_integration_schedule(
        3600, 1800, 1800, 0, 0, 0, jnp.float32, True, spatial_phase=1)
    np.testing.assert_array_equal(
        np.asarray(resumed["yfirst_per_step"]), [False, True])


def test_schedule_mask_for_a_component_coarser_than_the_substep():
    schedule = prepare_integration_schedule(
        3600, 3600, 0, 1800, 0, 3600, jnp.float32, True, dt_gse=3600)
    assert schedule["num_steps"] == 2
    np.testing.assert_array_equal(
        np.asarray(schedule["should_integrate_x"]), [False, True])
    np.testing.assert_array_equal(
        np.asarray(schedule["should_integrate_th"]), [True, True])
    np.testing.assert_array_equal(
        np.asarray(schedule["should_integrate_stot"]), [False, True])
    np.testing.assert_array_equal(
        np.asarray(schedule["should_integrate_gse"]), [False, True])


def test_schedule_gcd_substeps():
    schedule = prepare_integration_schedule(
        3600, 3600, 3600, 1800, 0, 3600, jnp.float32, True)
    assert schedule["num_steps"] == 2
    assert list(np.asarray(schedule["should_integrate_th"])) == [True, True]
    assert list(np.asarray(schedule["should_integrate_k"])) == [False, False]


def test_schedule_zero_when_everything_off():
    schedule = prepare_integration_schedule(
        3600, 0, 0, 0, 0, 0, jnp.float32, True)
    assert schedule["num_steps"] == 0


def _switched_off_config(tiny_config: WaveConfig) -> WaveConfig:
    return tiny_config._replace(
        physics=WavePhysicsConfig(
            fachfa=0.0, facvx=0.0, facvy=0.0, facth=0.0, fdg=0.0),
        gse=WaveGSEConfig(enabled=False))


def test_nonfinite_wavenumber_is_not_silently_zeroed(tiny_config, tiny_state):
    from legoesm.wave.physics.integrator import _sanitize_physics_fields

    env = prepare_env(tiny_config, tiny_state.depth)
    bad = env._replace(wavenum=env.wavenum.at[0, 2, 2, 0, 0].set(jnp.nan))
    out = _sanitize_physics_fields(bad, jnp.ones(tiny_state.mask.shape, dtype=bool))
    assert bool(jnp.isnan(out.wavenum[0, 2, 2, 0, 0]))
    # Land columns are still zeroed.
    out_land = _sanitize_physics_fields(
        env, jnp.zeros(tiny_state.mask.shape, dtype=bool))
    assert np.all(np.asarray(out_land.wavenum) == 0.0)


def test_all_switches_off_leaves_interior_unchanged(tiny_config, tiny_state):
    config = _switched_off_config(tiny_config)
    env = prepare_env(config, tiny_state.depth)
    stot = jnp.zeros_like(tiny_state.spectra)
    out = model_advance(
        tiny_state.spectra, stot, tiny_state.depth[..., None, None],
        env, config, tiny_state.mask, jnp.zeros_like(tiny_state.spectra),
    )
    assert np.allclose(
        np.asarray(out[:, 1:-1, 1:-1]), np.asarray(tiny_state.spectra[:, 1:-1, 1:-1]),
        atol=1e-6)


def test_simple_stot_integrator_is_si(tiny_config, tiny_state):
    spectra = jnp.ones_like(tiny_state.spectra) * 0.02
    stot = jnp.ones_like(spectra) * 1e-4
    mask = jnp.ones(tiny_state.mask.shape, dtype=bool)
    out = simple_stot_integrator(spectra, stot, 3600.0, mask)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(spectra + stot * 3600.0), rtol=1e-6)
    land = jnp.zeros(tiny_state.mask.shape, dtype=bool)
    np.testing.assert_array_equal(
        np.asarray(simple_stot_integrator(spectra, stot, 3600.0, land)),
        np.asarray(spectra))


def test_source_runs_after_the_spatial_phase(tiny_config, tiny_state):
    """A non-commuting source pins the documented spatial-then-source order."""
    config = tiny_config._replace(
        time=WaveTimeConfig(
            dt_global_s=3600, dt_x_s=3600, dt_y_s=0, dt_th_s=0,
            dt_k_s=0, dt_stot_s=3600, dt_gse_s=0),
        gse=WaveGSEConfig(enabled=False))
    env = prepare_env(config, tiny_state.depth)
    mask = jnp.ones(tiny_state.mask.shape, dtype=bool)
    boundary = jnp.zeros_like(tiny_state.spectra)

    def constant_tendency(spectra, **kwargs):
        del kwargs
        return jnp.ones_like(spectra)

    def identity_source(spectra, stot, dt, mask):
        del stot, dt, mask
        return spectra

    def doubling_source(spectra, stot, dt, mask):
        del stot, dt, mask
        return 2.0 * spectra

    kwargs = dict(
        depth=tiny_state.depth[..., None, None], env=env, config=config,
        mask=mask, boundary_condition=boundary,
        x_tendency_fn=constant_tendency, y_tendency_fn=constant_tendency,
    )
    once = model_advance(
        tiny_state.spectra, jnp.zeros_like(tiny_state.spectra),
        stot_advance=identity_source, **kwargs)
    doubled = model_advance(
        tiny_state.spectra, jnp.zeros_like(tiny_state.spectra),
        stot_advance=doubling_source, **kwargs)
    np.testing.assert_allclose(
        np.asarray(doubled), 2.0 * np.asarray(once), rtol=1e-6)


def test_gse_runs_through_model_advance(tiny_config, tiny_state):
    from legoesm.wave.physics.gse import GSESpatialAveraging

    config = tiny_config._replace(
        time=WaveTimeConfig(
            dt_global_s=3600, dt_x_s=0, dt_y_s=0, dt_th_s=0,
            dt_k_s=0, dt_stot_s=0, dt_gse_s=3600),
        gse=WaveGSEConfig(enabled=True, alpha_s=1.0, alpha_n=1.0,
                          periodic_x=True),
    )
    env = prepare_env(config, tiny_state.depth)
    mask = jnp.ones(tiny_state.mask.shape, dtype=bool)
    boundary = jnp.zeros_like(tiny_state.spectra)
    spectra = tiny_state.spectra.at[0, 2, 2].set(0.5)
    gse = GSESpatialAveraging(config.gse)
    with_gse = model_advance(
        spectra, jnp.zeros_like(spectra), tiny_state.depth[..., None, None],
        env, config, mask, boundary, gse_alleviation=gse)
    without = model_advance(
        spectra, jnp.zeros_like(spectra), tiny_state.depth[..., None, None],
        env, config, mask, boundary, gse_alleviation=None)
    assert not np.allclose(np.asarray(with_gse), np.asarray(without), atol=1e-7)


def test_source_runs_once_in_a_two_substep_scan(tiny_config, tiny_state):
    """The scan must index the schedule, not always use step zero."""
    config = tiny_config._replace(
        time=WaveTimeConfig(
            dt_global_s=3600, dt_x_s=1800, dt_y_s=0, dt_th_s=0,
            dt_k_s=0, dt_stot_s=3600, dt_gse_s=0),
        gse=WaveGSEConfig(enabled=False))
    env = prepare_env(config, tiny_state.depth)
    mask = jnp.ones(tiny_state.mask.shape, dtype=bool)

    def add_one(spectra, stot, dt, mask):
        del stot, dt, mask
        return spectra + 1.0

    out = model_advance(
        tiny_state.spectra, jnp.zeros_like(tiny_state.spectra),
        tiny_state.depth[..., None, None], env, config, mask,
        jnp.zeros_like(tiny_state.spectra), stot_advance=add_one,
        x_tendency_fn=lambda spectra, **kwargs: jnp.zeros_like(spectra),
    )
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(tiny_state.spectra) + 1.0, rtol=1e-6)


def test_gse_runs_after_the_spatial_phase(tiny_config, tiny_state):
    """A nonlinear spatial tendency distinguishes spatial-then-GSE order."""
    from legoesm.wave.physics.gse import GSESpatialAveraging

    def square_tendency(spectra, **kwargs):
        del kwargs
        return 0.01 * spectra ** 2

    def make_config(dt_gse):
        return tiny_config._replace(
            time=WaveTimeConfig(
                dt_global_s=3600, dt_x_s=3600, dt_y_s=0, dt_th_s=0,
                dt_k_s=0, dt_stot_s=0, dt_gse_s=dt_gse),
            gse=WaveGSEConfig(enabled=dt_gse > 0, alpha_s=1.0,
                              alpha_n=1.0, periodic_x=True),
        )

    both_config = make_config(3600)
    env = prepare_env(both_config, tiny_state.depth)
    mask = jnp.ones(tiny_state.mask.shape, dtype=bool)
    boundary = jnp.zeros_like(tiny_state.spectra)
    spectra = tiny_state.spectra.at[0, 2, 2].set(0.5)
    gse = GSESpatialAveraging(both_config.gse)
    both = model_advance(
        spectra, jnp.zeros_like(spectra), tiny_state.depth[..., None, None],
        env, both_config, mask, boundary,
        x_tendency_fn=square_tendency, y_tendency_fn=square_tendency,
        gse_alleviation=gse)
    spatial_config = make_config(0)
    spatial_env = prepare_env(spatial_config, tiny_state.depth)
    spatial = model_advance(
        spectra, jnp.zeros_like(spectra), tiny_state.depth[..., None, None],
        spatial_env, spatial_config, mask, boundary,
        x_tendency_fn=square_tendency, y_tendency_fn=square_tendency)
    # The in-scan GSE phase projects to non-negative after averaging.
    expected = jnp.maximum(gse(spatial, env, mask, 3600.0), 0.0)
    np.testing.assert_allclose(
        np.asarray(both), np.asarray(expected), rtol=1e-5, atol=1e-8)


def test_gse_runs_before_a_nonlinear_source(tiny_config, tiny_state):
    """A nonlinear source distinguishes GSE-then-source from source-then-GSE."""
    from legoesm.wave.physics.gse import GSESpatialAveraging

    config = tiny_config._replace(
        time=WaveTimeConfig(
            dt_global_s=3600, dt_x_s=0, dt_y_s=0, dt_th_s=0,
            dt_k_s=0, dt_stot_s=3600, dt_gse_s=3600),
        gse=WaveGSEConfig(enabled=True, alpha_s=1.0, alpha_n=1.0,
                          periodic_x=True),
    )
    env = prepare_env(config, tiny_state.depth)
    mask = jnp.ones(tiny_state.mask.shape, dtype=bool)
    boundary = jnp.zeros_like(tiny_state.spectra)
    spectra = tiny_state.spectra.at[0, 2, 2].set(0.5)
    gse = GSESpatialAveraging(config.gse)

    def identity_source(s, stot, dt, mask):
        del stot, dt, mask
        return s

    def squaring_source(s, stot, dt, mask):
        del stot, dt, mask
        return s ** 2

    kwargs = dict(
        depth=tiny_state.depth[..., None, None], env=env, config=config,
        mask=mask, boundary_condition=boundary, gse_alleviation=gse,
    )
    gse_only = model_advance(
        spectra, jnp.zeros_like(spectra), stot_advance=identity_source, **kwargs)
    squared = model_advance(
        spectra, jnp.zeros_like(spectra), stot_advance=squaring_source, **kwargs)
    np.testing.assert_allclose(
        np.asarray(squared), np.asarray(gse_only) ** 2, rtol=1e-5, atol=1e-8)


def test_audit_clipping_reports_mass(tiny_config, tiny_state):
    def sink(spectra, stot, dt, mask):
        del stot, dt, mask
        return spectra - 1.0

    config = tiny_config._replace(
        time=WaveTimeConfig(
            dt_global_s=3600, dt_x_s=0, dt_y_s=0, dt_th_s=0,
            dt_k_s=0, dt_stot_s=3600, dt_gse_s=0),
        gse=WaveGSEConfig(enabled=False))
    env = prepare_env(config, tiny_state.depth)
    mask = jnp.ones(tiny_state.mask.shape, dtype=bool)
    out, clipped = model_advance(
        tiny_state.spectra, jnp.zeros_like(tiny_state.spectra),
        tiny_state.depth[..., None, None], env, config, mask,
        jnp.zeros_like(tiny_state.spectra), stot_advance=sink,
        audit_clipping=True)
    assert np.all(np.asarray(out) >= 0.0)
    negative = jnp.where(
        mask[..., None, None], jnp.maximum(1.0 - tiny_state.spectra, 0.0), 0.0)
    weight = wave_cell_area(env) * env.quadrature_df * env.dth
    expected = jnp.sum(negative * weight, axis=(1, 2, 3, 4))
    np.testing.assert_allclose(
        np.asarray(clipped), np.asarray(expected), rtol=1e-6)


def test_checkpoint_levels_agree(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    stot = jnp.zeros_like(tiny_state.spectra)
    outputs = []
    for level in ("none", "per_step", "per_phase"):
        config = tiny_config._replace(
            physics=tiny_config.physics._replace(checkpoint_level=level))
        outputs.append(model_advance(
            tiny_state.spectra, stot, tiny_state.depth[..., None, None],
            env, config, tiny_state.mask, jnp.zeros_like(tiny_state.spectra),
        ))
    for other in outputs[1:]:
        assert np.allclose(np.asarray(outputs[0]), np.asarray(other), atol=1e-6)
