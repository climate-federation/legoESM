"""Unit tests for the joint ML physics parameterization package."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.ml_parameterization import (
    apply_predicted_sundqvist_rain_survival_fraction,
)
from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.microphysics.sundqvist import (
    diagnose_sundqvist_process_rates,
    sundqvist_microphysics,
)
from legoesm.ml.physics import (
    PhysicsColumnBatch,
    PhysicsModelConfig,
    PhysicsParameterizationModel,
    PhysicsTeacherDataset,
    PhysicsTrainingConfig,
    load_physics_checkpoint,
    load_physics_stats,
    pack_physics_parameterization_features,
    pack_physics_parameterization_targets,
    save_physics_checkpoint,
    save_physics_stats,
    train_physics_parameterization,
    unpack_physics_parameterization_targets,
)
from legoesm.ml.physics.io import PhysicsNormalizationBundle
from legoesm.ml.normalization import NormalizationStats


def test_feature_vector_shape():
    nlev = 4
    feature = pack_physics_parameterization_features(
        T=jnp.ones((nlev,)),
        u=jnp.ones((nlev,)),
        v=jnp.ones((nlev,)),
        q_v=jnp.ones((nlev,)) * 1e-3,
        q_c=jnp.zeros((nlev,)),
        q_r=jnp.zeros((nlev,)),
        p_full=jnp.linspace(2e4, 1e5, nlev),
        z_full=jnp.linspace(100.0, 10000.0, nlev),
        p_s=jnp.asarray(1e5),
        T_sfc=jnp.asarray(300.0),
        q_sfc=jnp.asarray(0.01),
        lat=jnp.asarray(0.4),
        cape=jnp.asarray(500.0),
        M_c=jnp.asarray(1e-3),
        dt=jnp.asarray(300.0),
    )
    assert feature.shape == (6 * nlev + 8,)


def test_kessler_feature_vector_shape():
    nlev = 4
    feature = pack_physics_parameterization_features(
        T=jnp.ones((nlev,)),
        u=jnp.ones((nlev,)),
        v=jnp.ones((nlev,)),
        q_v=jnp.ones((nlev,)) * 1e-3,
        q_c=jnp.ones((nlev,)) * 2e-4,
        q_r=jnp.ones((nlev,)) * 1e-4,
        p_full=jnp.linspace(2e4, 1e5, nlev),
        z_full=jnp.linspace(100.0, 10000.0, nlev),
        p_s=jnp.asarray(1e5),
        T_sfc=jnp.asarray(300.0),
        q_sfc=jnp.asarray(0.01),
        lat=jnp.asarray(0.4),
        cape=jnp.asarray(500.0),
        M_c=jnp.asarray(1e-3),
        dt=jnp.asarray(300.0),
        microphysics_scheme="kessler",
    )
    assert feature.shape == (8 * nlev + 8,)


def test_sundqvist_feature_vector_shape():
    nlev = 4
    feature = pack_physics_parameterization_features(
        T=jnp.ones((nlev,)),
        u=jnp.ones((nlev,)),
        v=jnp.ones((nlev,)),
        q_v=jnp.ones((nlev,)) * 1e-3,
        q_c=jnp.ones((nlev,)) * 2e-4,
        q_r=jnp.ones((nlev,)) * 1e-4,
        p_full=jnp.linspace(2e4, 1e5, nlev),
        z_full=jnp.linspace(100.0, 10000.0, nlev),
        p_s=jnp.asarray(1e5),
        T_sfc=jnp.asarray(300.0),
        q_sfc=jnp.asarray(0.01),
        lat=jnp.asarray(0.4),
        cape=jnp.asarray(500.0),
        M_c=jnp.asarray(1e-3),
        dt=jnp.asarray(300.0),
        microphysics_scheme="sundqvist",
    )
    assert feature.shape == (7 * nlev + 8,)


def test_target_vector_shape_and_unpack():
    nlev = 4
    packed = pack_physics_parameterization_targets(
        Km=jnp.ones((2, nlev)),
        Kh=jnp.ones((2, nlev)) * 2.0,
        M_eq=jnp.ones((2,)) * 3.0,
    )
    assert packed.shape == (2, 2 * nlev + 1)
    unpacked = unpack_physics_parameterization_targets(packed, nlev)
    assert unpacked["Km"].shape == (2, nlev)
    assert unpacked["Kh"].shape == (2, nlev)
    assert unpacked["M_eq"].shape == (2,)


def test_kessler_target_vector_shape_and_unpack():
    nlev = 4
    packed = pack_physics_parameterization_targets(
        Km=jnp.ones((2, nlev)),
        Kh=jnp.ones((2, nlev)) * 2.0,
        M_eq=jnp.ones((2,)) * 3.0,
        dq_v_dt_micro=jnp.ones((2, nlev)) * -1.0e-6,
        dq_c_dt_micro=jnp.ones((2, nlev)) * 5.0e-7,
        dq_r_dt_micro=jnp.ones((2, nlev)) * 5.0e-7,
        precip_micro=jnp.ones((2,)) * 1.0e-4,
        microphysics_scheme="kessler",
    )
    assert packed.shape == (2, 5 * nlev + 2)
    unpacked = unpack_physics_parameterization_targets(
        packed,
        nlev,
        microphysics_scheme="kessler",
    )
    assert unpacked["dq_v_dt_micro"].shape == (2, nlev)
    assert unpacked["dq_c_dt_micro"].shape == (2, nlev)
    assert unpacked["dq_r_dt_micro"].shape == (2, nlev)
    assert unpacked["precip_micro"].shape == (2,)


def test_sundqvist_target_vector_shape_and_unpack():
    nlev = 4
    packed = pack_physics_parameterization_targets(
        Km=jnp.ones((2, nlev)),
        Kh=jnp.ones((2, nlev)) * 2.0,
        M_eq=jnp.ones((2,)) * 3.0,
        rain_survival_fraction=jnp.ones((2,)) * 0.8,
        microphysics_scheme="sundqvist",
    )
    assert packed.shape == (2, 2 * nlev + 2)
    unpacked = unpack_physics_parameterization_targets(
        packed,
        nlev,
        microphysics_scheme="sundqvist",
    )
    assert unpacked["rain_survival_fraction"].shape == (2,)


def test_checkpoint_and_stats_roundtrip():
    nlev = 4
    model = PhysicsParameterizationModel(
        nlev=nlev,
        hidden_dim=16,
        n_layers=2,
        key=jax.random.PRNGKey(0),
    )
    stats = PhysicsNormalizationBundle(
        input_stats=NormalizationStats(
            mean=jnp.zeros((6 * nlev + 8,)),
            std=jnp.ones((6 * nlev + 8,)),
        ),
        output_stats=NormalizationStats(
            mean=jnp.zeros((2 * nlev + 1,)),
            std=jnp.ones((2 * nlev + 1,)),
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "physics.eqx"
        stats_path = Path(tmpdir) / "physics_stats.npz"
        save_physics_checkpoint(model, ckpt)
        save_physics_stats(stats, stats_path)
        restored_model = load_physics_checkpoint(model, ckpt)
        restored_stats = load_physics_stats(stats_path)

    x = jnp.ones((6 * nlev + 8,))
    assert jnp.allclose(model(x), restored_model(x))
    assert jnp.allclose(stats.input_stats.mean, restored_stats.input_stats.mean)
    assert jnp.allclose(stats.output_stats.std, restored_stats.output_stats.std)


def test_kessler_checkpoint_and_stats_roundtrip():
    nlev = 4
    model = PhysicsParameterizationModel(
        nlev=nlev,
        hidden_dim=16,
        n_layers=2,
        microphysics_scheme="kessler",
        key=jax.random.PRNGKey(0),
    )
    stats = PhysicsNormalizationBundle(
        input_stats=NormalizationStats(
            mean=jnp.zeros((8 * nlev + 8,)),
            std=jnp.ones((8 * nlev + 8,)),
        ),
        output_stats=NormalizationStats(
            mean=jnp.zeros((5 * nlev + 2,)),
            std=jnp.ones((5 * nlev + 2,)),
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "physics.eqx"
        stats_path = Path(tmpdir) / "physics_stats.npz"
        save_physics_checkpoint(model, ckpt)
        save_physics_stats(stats, stats_path)
        restored_model = load_physics_checkpoint(model, ckpt)
        restored_stats = load_physics_stats(stats_path)

    x = jnp.ones((8 * nlev + 8,))
    assert jnp.allclose(model(x), restored_model(x))
    assert jnp.allclose(stats.input_stats.mean, restored_stats.input_stats.mean)
    assert jnp.allclose(stats.output_stats.std, restored_stats.output_stats.std)


def test_sundqvist_checkpoint_and_stats_roundtrip():
    nlev = 4
    model = PhysicsParameterizationModel(
        nlev=nlev,
        hidden_dim=16,
        n_layers=2,
        microphysics_scheme="sundqvist",
        key=jax.random.PRNGKey(0),
    )
    stats = PhysicsNormalizationBundle(
        input_stats=NormalizationStats(
            mean=jnp.zeros((7 * nlev + 8,)),
            std=jnp.ones((7 * nlev + 8,)),
        ),
        output_stats=NormalizationStats(
            mean=jnp.zeros((2 * nlev + 2,)),
            std=jnp.ones((2 * nlev + 2,)),
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "physics.eqx"
        stats_path = Path(tmpdir) / "physics_stats.npz"
        save_physics_checkpoint(model, ckpt)
        save_physics_stats(stats, stats_path)
        restored_model = load_physics_checkpoint(model, ckpt)
        restored_stats = load_physics_stats(stats_path)

    x = jnp.ones((7 * nlev + 8,))
    assert jnp.allclose(model(x), restored_model(x))
    assert jnp.allclose(stats.input_stats.mean, restored_stats.input_stats.mean)
    assert jnp.allclose(stats.output_stats.std, restored_stats.output_stats.std)


def test_sundqvist_training_pipeline_evaluates_microphysics_targets():
    n_samples = 12
    nlev = 4
    p_full = jnp.tile(jnp.linspace(2.0e4, 1.0e5, nlev), (n_samples, 1))
    p_half = jnp.tile(jnp.linspace(1.0e4, 1.05e5, nlev + 1), (n_samples, 1))
    z_full = jnp.tile(jnp.linspace(500.0, 9500.0, nlev), (n_samples, 1))
    z_half = jnp.tile(jnp.linspace(0.0, 10000.0, nlev + 1), (n_samples, 1))
    columns = PhysicsColumnBatch(
        T=jnp.tile(jnp.linspace(250.0, 295.0, nlev), (n_samples, 1)),
        u=jnp.ones((n_samples, nlev)) * 8.0,
        v=jnp.ones((n_samples, nlev)) * -4.0,
        q_v=jnp.ones((n_samples, nlev)) * 8.0e-3,
        q_c=jnp.ones((n_samples, nlev)) * 2.0e-4,
        q_r=jnp.ones((n_samples, nlev)) * 1.0e-4,
        p_full=p_full,
        p_half=p_half,
        z_full=z_full,
        z_half=z_half,
        p_s=jnp.ones((n_samples,)) * 1.0e5,
        T_sfc=jnp.ones((n_samples,)) * 300.0,
        q_sfc=jnp.ones((n_samples,)) * 1.2e-2,
        lat=jnp.linspace(-0.6, 0.6, n_samples),
        cape=jnp.ones((n_samples,)) * 250.0,
        M_c=jnp.ones((n_samples,)) * 2.0e-3,
        dt=jnp.ones((n_samples,)) * 150.0,
        day=jnp.linspace(0.0, 1.0, n_samples),
    )
    dataset = PhysicsTeacherDataset(
        columns=columns,
        Km=jnp.ones((n_samples, nlev)) * 10.0,
        Kh=jnp.ones((n_samples, nlev)) * 12.0,
        M_eq=jnp.ones((n_samples,)) * 3.0e-3,
        rain_survival_fraction=jnp.ones((n_samples,)) * 0.75,
        dq_v_dt_micro=jnp.ones((n_samples, nlev)) * -2.0e-7,
        dq_c_dt_micro=jnp.ones((n_samples, nlev)) * 1.5e-7,
        dq_r_dt_micro=jnp.ones((n_samples, nlev)) * 5.0e-8,
        precip_micro=jnp.ones((n_samples,)) * 8.0e-5,
        sample_days=(0.0, 1.0),
        microphysics_scheme="sundqvist",
    )

    result = train_physics_parameterization(
        dataset,
        model_config=PhysicsModelConfig(hidden_dim=8, n_layers=1, seed=0),
        training_config=PhysicsTrainingConfig(
            batch_size=4,
            epochs=1,
            patience=1,
            train_fraction=0.5,
            val_fraction=0.25,
            seed=0,
        ),
    )

    assert result.metrics.rmse_rain_survival_fraction >= 0.0
    assert result.metrics.rmse_dq_v_dt_micro == 0.0
    assert result.metrics.rmse_dq_c_dt_micro == 0.0
    assert result.metrics.rmse_dq_r_dt_micro == 0.0
    assert result.metrics.rmse_precip_micro == 0.0


def test_sundqvist_rain_survival_rebuild_matches_physical_scheme():
    T = jnp.array([[282.0, 276.0]])
    q_v = jnp.array([[8.5e-3, 5.5e-3]])
    q_c = jnp.array([[2.0e-4, 1.0e-4]])
    q_r = jnp.zeros_like(q_c)
    p_full = jnp.array([[8.5e4, 6.5e4]])
    p_half = jnp.array([[9.5e4, 7.5e4, 5.5e4]])
    rho = jnp.array([[1.05, 0.82]])
    dz = jnp.array([[1200.0, 1500.0]])
    hydrometeors = HydrometeorState(
        q_c=q_c,
        q_r=q_r,
        q_i=jnp.zeros_like(q_c),
        q_s=jnp.zeros_like(q_c),
        q_g=jnp.zeros_like(q_c),
        N_c=jnp.zeros_like(q_c),
        N_r=jnp.zeros_like(q_c),
        N_i=jnp.zeros_like(q_c),
    )
    config = SundqvistConfig()
    physical = sundqvist_microphysics(
        T=T,
        q_v=q_v,
        hydrometeors=hydrometeors,
        p_full=p_full,
        p_half=p_half,
        rho=rho,
        dz=dz,
        dt=150.0,
        config=config,
    )
    rates = diagnose_sundqvist_process_rates(
        T=T,
        q_v=q_v,
        hydrometeors=hydrometeors,
        p_full=p_full,
        p_half=p_half,
        rho=rho,
        dz=dz,
        dt=150.0,
        config=config,
    )
    generated_flux = jnp.sum(rates.autoconversion * rho * dz, axis=1)
    rain_survival_fraction = jnp.where(
        generated_flux > 1.0e-12,
        rates.precipitation / generated_flux,
        1.0,
    )
    rebuilt = apply_predicted_sundqvist_rain_survival_fraction(
        rain_survival_fraction,
        T=T,
        q_v=q_v,
        hydrometeors=hydrometeors,
        p_full=p_full,
        p_half=p_half,
        rho=rho,
        dz=dz,
        dt=150.0,
        config=config,
    )
    assert jnp.allclose(
        rebuilt.dT_dt,
        physical.dT_dt,
        rtol=1.0e-6,
        atol=1.0e-12,
    )
    assert jnp.allclose(
        rebuilt.dq_v_dt,
        physical.dq_v_dt,
        rtol=1.0e-6,
        atol=1.0e-12,
    )
    assert jnp.allclose(rebuilt.dq_c_dt, physical.dq_c_dt, rtol=1.0e-6, atol=1.0e-12)
    assert jnp.allclose(rebuilt.dq_r_dt, physical.dq_r_dt, rtol=1.0e-6, atol=1.0e-12)
    assert jnp.allclose(
        rebuilt.precipitation,
        physical.precipitation,
        rtol=1.0e-6,
        atol=1.0e-12,
    )


def test_sundqvist_rain_survival_rebuild_drains_incoming_qr():
    """The ML rebuild path must apply the same diagnostic-rain semantics
    as the physical scheme.

    When q_r > 0 on input (e.g., from a prior Kessler step or warm-start),
    both the leaf scheme and the ML rebuild must:
      (a) drain q_r to ~0 in one step (``dq_r_dt = -q_r/dt``),
      (b) include the drained mass in ``precipitation``.

    Pre-fix the ML path emitted ``dq_r_dt = autoconv - evap`` (no drain),
    so an ML model trained against a teacher snapshot that exhibited
    this bug would learn to leak rain mass.
    """
    T = jnp.array([[282.0, 276.0]])
    q_v = jnp.array([[5.0e-3, 3.0e-3]])  # subsaturated to suppress autoconv
    q_c = jnp.zeros((1, 2))
    q_r = jnp.array([[3.0e-4, 2.0e-4]])  # nonzero rain on input
    p_full = jnp.array([[8.5e4, 6.5e4]])
    p_half = jnp.array([[9.5e4, 7.5e4, 5.5e4]])
    rho = jnp.array([[1.05, 0.82]])
    dz = jnp.array([[1200.0, 1500.0]])
    hydrometeors = HydrometeorState(
        q_c=q_c, q_r=q_r,
        q_i=jnp.zeros_like(q_c),
        q_s=jnp.zeros_like(q_c),
        q_g=jnp.zeros_like(q_c),
        N_c=jnp.zeros_like(q_c),
        N_r=jnp.zeros_like(q_c),
        N_i=jnp.zeros_like(q_c),
    )
    config = SundqvistConfig()
    dt = 150.0

    physical = sundqvist_microphysics(
        T=T, q_v=q_v, hydrometeors=hydrometeors,
        p_full=p_full, p_half=p_half, rho=rho, dz=dz, dt=dt, config=config,
    )
    # Use survival-fraction = 1 (no rain re-evaporation) for the rebuild;
    # the q_r-drain channel is independent.
    rebuilt = apply_predicted_sundqvist_rain_survival_fraction(
        jnp.ones((1,)),
        T=T, q_v=q_v, hydrometeors=hydrometeors,
        p_full=p_full, p_half=p_half, rho=rho, dz=dz, dt=dt, config=config,
    )
    # q_r drains in both paths.
    q_r_after_phys = q_r + physical.dq_r_dt * dt
    q_r_after_ml = q_r + rebuilt.dq_r_dt * dt
    assert float(jnp.max(jnp.abs(q_r_after_phys))) < 1e-12
    assert float(jnp.max(jnp.abs(q_r_after_ml))) < 1e-12
    # Drain flux contributes to precipitation in both paths.
    dp = p_half[:, 1:] - p_half[:, :-1]
    expected_drain = float(
        jnp.sum(q_r * dp, axis=1)[0] / (9.80616 * dt)
    )
    assert float(physical.precipitation[0]) >= expected_drain - 1e-12
    assert float(rebuilt.precipitation[0]) >= expected_drain - 1e-12
