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
