"""ExperimentConfig.turbulence_override → the corrected scheme config reaches physics.

The injection point for the LES-informed correction loop (iter 35): a refined /
per-column ``clubb_lite`` config (e.g. a corrected ``C_K`` field) supplied via
``ExperimentConfig.turbulence_override`` must flow through the driver's
physics-config construction into the turbulence kernel — WITHOUT a new driver
signature — while ``None`` (default) stays byte-identical to the scheme-string
default.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.turbulence.config import (
    CLUBBLiteConfig,
    TurbulenceConfig,
)
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
)
from legoesm.driver.physics_pipeline import (
    build_physics_pipeline,
    turbulence_config_for,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate


def _config(turbulence="clubb_lite", override=None):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence=turbulence,
        turbulence_override=override,
    )


def test_turbulence_config_for_default_vs_override():
    base = _config()
    # No override → the default scheme config (C_K = 0.4).
    default_tc = turbulence_config_for(base)
    assert default_tc.scheme == "clubb_lite"
    assert float(default_tc.clubb_lite.C_K) == float(CLUBBLiteConfig().C_K)

    # Override → returned verbatim (carries the refined C_K).
    over = TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=1.1))
    cfg = _config(override=over)
    assert turbulence_config_for(cfg) is over
    assert float(turbulence_config_for(cfg).clubb_lite.C_K) == 1.1


def test_override_none_is_default_byte_identical():
    """turbulence_override=None ⇒ exactly the scheme-string default config."""
    a = turbulence_config_for(_config(override=None))
    b = TurbulenceConfig(scheme="clubb_lite")
    assert a == b


def test_louis_scalars_reach_the_kernel_config():
    """The 2026-08-01 calibration-campaign finding: louis_l_mix_max /
    louis_c_louis (& friends) were documented in driver/config.py as targeting
    LouisConfig, threaded between config objects, and injected by the ML
    tuning path — but turbulence_config_for, the single source every dycore's
    production kernel consumes, silently dropped them. The two scalars ranked
    2nd and 5th in the C48 sensitivity sweep tuned NOTHING. Each documented
    scalar must reach its LouisConfig leaf."""
    cfg = _config(turbulence="louis")._replace(
        louis_l_mix_max=250.0, louis_Ri_crit=0.4, louis_b_louis=4.0,
        louis_c_louis=12.0, louis_d_louis=6.0)
    tc = turbulence_config_for(cfg)
    assert tc.scheme == "louis"
    assert float(tc.louis.l_mix_max) == 250.0
    assert float(tc.louis.Ri_crit) == 0.4
    assert float(tc.louis.b_louis) == 4.0
    assert float(tc.louis.c_louis) == 12.0
    assert float(tc.louis.d_louis) == 6.0


def test_louis_defaults_stay_byte_identical():
    """All-default louis scalars ⇒ no _replace: the identity contract that
    every pre-fix run is bit-reproducible must survive the threading."""
    a = turbulence_config_for(_config(turbulence="louis"))
    assert a == TurbulenceConfig(scheme="louis")


def test_louis_scalars_do_not_touch_other_schemes():
    """A non-louis scheme with (inapplicable) louis scalars set is unchanged —
    the threading is scoped to the active scheme, not sprayed."""
    cfg = _config(turbulence="clubb_lite")._replace(louis_l_mix_max=250.0)
    a = turbulence_config_for(cfg)
    assert a == TurbulenceConfig(scheme="clubb_lite")


def test_override_reaches_built_fv_pipeline():
    """The override's C_K reaches the REAL FV physics pipeline's turbulence
    config (the kernel that clubb_lite_turbulence reads C_K from)."""
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)

    default_pipe = build_physics_pipeline(grid, sigma, _config())
    assert float(default_pipe.turbulence_config.C_K) == float(CLUBBLiteConfig().C_K)

    over = TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=1.15))
    over_pipe = build_physics_pipeline(grid, sigma, _config(override=over))
    assert float(over_pipe.turbulence_config.C_K) == 1.15


def test_override_accepts_per_column_ck_array():
    """A PER-COLUMN C_K array (the correction-loop feedback field) flows through
    to the kernel config unchanged (iter-24 clubb_lite broadcasts it per column)."""
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    ncol = 8 * 16
    ck_field = jnp.linspace(0.3, 0.9, ncol)
    over = TurbulenceConfig(
        scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=ck_field))
    pipe = build_physics_pipeline(grid, sigma, _config(override=over))
    np.testing.assert_allclose(
        np.asarray(pipe.turbulence_config.C_K), np.asarray(ck_field))


def test_validate_strict_rejects_scheme_mismatch():
    """An override whose scheme differs from `turbulence` is a config error."""
    bad = _config(turbulence="clubb_lite",
                  override=TurbulenceConfig(scheme="smagorinsky"))
    with pytest.raises(ValueError, match="turbulence_override.scheme"):
        bad.validate_strict()


def test_validate_strict_accepts_matching_override():
    ok = _config(turbulence="clubb_lite",
                 override=TurbulenceConfig(scheme="clubb_lite",
                                           clubb_lite=CLUBBLiteConfig(C_K=0.7)))
    ok.validate_strict()  # no raise


def test_validate_strict_accepts_no_override():
    _config(override=None).validate_strict()  # no raise


def test_deploy_output_reaches_fv_pipeline_with_per_column_ck():
    """Clause-6 END-TO-END (the literal 'update the parameters in the AMIP/CMIP
    simulation'): a SAVED campaign output JSON → ``corrected_turbulence_override`` → the
    FV physics pipeline's turbulence-kernel config carries the PER-COLUMN corrected C_K.
    The deploy→override and override→pipeline halves are each tested; THIS locks the
    whole deploy-artifact → model-input chain as ONE flow (a regression in either the
    deploy loader or the pipeline injection that breaks the hand-off fails here)."""
    import json

    from legoesm.training.deploy_correction import corrected_turbulence_override

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    ck_field = np.linspace(0.35, 0.85, 8 * 16)                # per-column corrected C_K
    output = json.loads(json.dumps({"C_K": ck_field.tolist()}))  # the real on-disk --out shape
    over = corrected_turbulence_override(output)             # the DEPLOY artifact → override
    pipe = build_physics_pipeline(grid, sigma, _config(override=over))
    np.testing.assert_allclose(np.asarray(pipe.turbulence_config.C_K), ck_field)


def test_multi_deploy_output_reaches_fv_pipeline_all_three_coefficients():
    """The SIMULTANEOUS multi-coefficient deploy reaches the kernel for ALL THREE
    coefficients — not just C_K. A multi campaign `--out` ('fields' keyed by
    promotion_key) → corrected_turbulence_override → the FV pipeline's turbulence config
    carries the per-column C_K AND Pr_t AND C_eps (the single test above only locked C_K;
    a pipeline that dropped Pr_t/C_eps would silently deploy a partial correction)."""
    import json

    from legoesm.training.deploy_correction import corrected_turbulence_override

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    ncol = 8 * 16
    ck = np.linspace(0.30, 0.90, ncol)
    prt = np.linspace(0.50, 1.50, ncol)
    ceps = np.linspace(0.10, 0.60, ncol)
    output = json.loads(json.dumps({
        "coefficients": ["C_K", "Pr_t", "C_eps"],
        "fields": {"clubb_lite_C_K": ck.tolist(), "clubb_lite_Pr_t": prt.tolist(),
                   "clubb_lite_C_eps": ceps.tolist()}}))
    over = corrected_turbulence_override(output)
    tc = build_physics_pipeline(grid, sigma, _config(override=over)).turbulence_config
    np.testing.assert_allclose(np.asarray(tc.C_K), ck)
    np.testing.assert_allclose(np.asarray(tc.Pr_t), prt)
    np.testing.assert_allclose(np.asarray(tc.C_eps), ceps)


def test_on_disk_deploy_artifact_reaches_fv_pipeline(tmp_path):
    """The REAL on-disk deploy path: the campaign writes ``--out`` to a JSON FILE, and
    deploying reads THAT FILE via ``corrected_turbulence_override(path)`` and injects it at
    RUNTIME — because ``turbulence_override`` is intentionally NOT serialized into a config
    (``config.py:995``, a runtime-only injection), so the campaign-output JSON is the
    persistent deploy artifact, NOT a serialized deployed config. The file's per-column C_K
    reaches the model's turbulence kernel."""
    import json

    from legoesm.training.deploy_correction import corrected_turbulence_override

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    ck_field = np.linspace(0.35, 0.85, 8 * 16)
    out_file = tmp_path / "corrected_clubb.json"
    out_file.write_text(json.dumps({"C_K": ck_field.tolist()}))   # the campaign --out, on disk
    over = corrected_turbulence_override(str(out_file))           # deploy reads the FILE path
    pipe = build_physics_pipeline(grid, sigma, _config(override=over))
    np.testing.assert_allclose(np.asarray(pipe.turbulence_config.C_K), ck_field)


def test_validate_strict_rejects_non_turbulenceconfig_override():
    """A non-TurbulenceConfig override (e.g. a bare string) is rejected — it would
    otherwise crash later inside get_turbulence_fn (Codex iter-35)."""
    bad = _config(turbulence="clubb_lite", override="not-a-config")
    with pytest.raises(ValueError, match="must be a TurbulenceConfig"):
        bad.validate_strict()


def test_override_is_runtime_only_not_serialized():
    """The override (possibly a per-column JAX array) is NOT persisted: both
    serialisers drop it to None, so a config round-trip cannot silently corrupt
    it — the override is re-applied in memory after load (Codex iter-35)."""
    from legoesm.driver.config import (
        config_to_dict,
        experiment_config_from_dict,
        experiment_config_to_dict,
    )

    over = TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=1.3))
    cfg = _config(override=over)

    d = experiment_config_to_dict(cfg)
    assert d["turbulence_override"] is None
    restored = experiment_config_from_dict(d)
    assert restored.turbulence_override is None
    assert restored.turbulence == "clubb_lite"          # base config survives
    assert config_to_dict(cfg)["turbulence_override"] is None


def _tiny_coupled_clubb(c_k_override=None, days=0.5, diag_days=0.5):
    from legoesm.driver.config import OutputConfig
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    override = None
    if c_k_override is not None:
        override = TurbulenceConfig(
            scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=c_k_override))
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=diag_days),
        radiation="gray", turbulence="clubb_lite", days=days,
        turbulence_override=override,
    )
    ocean_grid = create_latlon_grid(n_lat=8, n_lon=16)
    driver = CoupledESMDriver(atm, PRESETS["aquaplanet"](), ocean_grid=ocean_grid)
    driver.setup()
    driver.run()
    return driver


@pytest.mark.slow
def test_override_changes_real_simulation_output():
    """The injected C_K actually CHANGES the simulation: two real coupled runs
    with different clubb_lite C_K diverge (the done-criterion's mechanism —
    updating the parameter updates the model state)."""
    low = _tiny_coupled_clubb(c_k_override=0.2)
    high = _tiny_coupled_clubb(c_k_override=1.2)
    t_low = np.asarray(low.state.T.data)
    t_high = np.asarray(high.state.T.data)
    assert np.all(np.isfinite(t_low)) and np.all(np.isfinite(t_high))
    # Different turbulent diffusivity ⇒ a different temperature field.
    assert not np.allclose(t_low, t_high)


@pytest.mark.slow
def test_per_column_ck_array_runs_real_forward_step():
    """A PER-COLUMN C_K array (the correction-loop feedback field) flows through
    a REAL forward pass (clubb_lite broadcasts it per column) — finite output, no
    shape error — not just pipeline construction (Codex iter-35)."""
    ncol = 8 * 16
    ck_field = jnp.linspace(0.2, 1.0, ncol)
    driver = _tiny_coupled_clubb(c_k_override=ck_field)
    t = np.asarray(driver.state.T.data)
    assert t.shape == (8, 16, 5)
    assert np.all(np.isfinite(t))
