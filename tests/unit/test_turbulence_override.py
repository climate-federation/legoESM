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


# ---------------------------------------------------------------------------
# surface_stability_scheme: plumbing AND flux CONSUMPTION on the latlon+HB lane.
#
# The plumbing-only coverage above (turbulence_config_for returns the right
# CONFIG object) did not catch the 2026-08 inert-knob bug: the injected
# ``SurfaceLayerConfig.stability_scheme`` reached ``compute_most_fluxes`` but
# the coare3 branch selected ``psi_m_coare``/``psi_h_coare`` unconditionally
# (bulk_flux.py), so two AMIP runs differing only in
# ``surface_stability_scheme`` were BIT-IDENTICAL.  This test closes the gap
# at the CONSUMPTION level: the resolver the driver uses
# (``turbulence_config_for``) -> the kernel the latlon compiled lane runs
# (``holtslag_boville_turbulence``, dispatched by ``get_turbulence_fn`` and
# called with ``config=self.turbulence_config`` in
# ``PhysicsPipeline.physics_step_no_rad``) -> different surface fluxes.
# ---------------------------------------------------------------------------
def _hb_lane_config(stability_scheme):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="holtslag_boville",
        surface_bulk_scheme="coare3", surface_gustiness_zi=300.0,
        surface_stability_scheme=stability_scheme,
    )


def _hb_shflx(stability_scheme):
    """Sensible flux from the EXACT lane chain: resolver -> dispatched HB
    kernel -> its internal compute_surface_fluxes, on a strongly stable
    synthetic column (+80 K surface inversion, 10 m/s wind)."""
    from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
        holtslag_boville_turbulence,
    )
    from legoesm.atmosphere.physics.turbulence.integration import (
        get_turbulence_fn,
    )

    tc = turbulence_config_for(_hb_lane_config(stability_scheme))
    name, turb_fn, sub = get_turbulence_fn(tc)
    # Name the symbol that runs (not a delegating wrapper): the latlon lane's
    # turbulence kernel IS holtslag_boville_turbulence.
    assert name == "holtslag_boville"
    assert turb_fn is holtslag_boville_turbulence
    # Plumbing (the pre-existing guarantee): the injection reached the
    # sub-config the kernel will read.
    assert sub.surface.bulk_scheme == "coare3"
    assert sub.surface.stability_scheme == stability_scheme
    assert float(sub.surface.gustiness_w_zi) == 300.0

    ncol, nlev = 2, 4
    ones = jnp.ones((ncol, nlev))
    z_full = jnp.broadcast_to(jnp.asarray([3000.0, 2000.0, 1000.0, 100.0]),
                              (ncol, nlev))
    z_half = jnp.broadcast_to(
        jnp.asarray([3500.0, 2500.0, 1500.0, 500.0, 0.0]), (ncol, nlev + 1))
    T = jnp.broadcast_to(jnp.asarray([270.0, 280.0, 290.0, 300.0]),  # noqa: N806 — canonical temperature symbol
                         (ncol, nlev))
    p_full = jnp.broadcast_to(
        jnp.asarray([70000.0, 80000.0, 90000.0, 99000.0]), (ncol, nlev))
    p_half = jnp.broadcast_to(
        jnp.asarray([65000.0, 75000.0, 85000.0, 95000.0, 100000.0]),
        (ncol, nlev + 1))
    from legoesm import constants
    rho = p_full / (constants.R_d * T)
    out = turb_fn(
        u=10.0 * ones, v=0.0 * ones, T=T, q_v=0.002 * ones,
        p_full=p_full, p_half=p_half, z_full=z_full, z_half=z_half,
        T_sfc=jnp.full((ncol,), 220.0), q_sfc=jnp.full((ncol,), 3e-4),
        rho=rho, dt=600.0, config=sub,
    )
    return np.asarray(out.shflx)


def test_surface_stability_scheme_reaches_hb_flux_consumption():
    """The knob changes the FLUX the lane kernel produces — not just the
    config object.  grachev2007_sheba (a genuinely different SBL tail on
    coare3) must move the stable sensible flux by > 5%; BH91 must be a
    nonzero change (COARE's native stable branch is BH91 with rounded
    constants, so that pair is rounding-level by physics)."""
    sh_dyer = _hb_shflx("dyer1974")
    sh_gr = _hb_shflx("grachev2007_sheba")
    sh_bh = _hb_shflx("beljaars_holtslag1991")
    assert np.all(np.isfinite(sh_dyer)) and np.all(np.isfinite(sh_gr))
    assert np.all(sh_dyer != sh_bh), (
        "surface_stability_scheme is inert through the HB lane kernel "
        "(the 2026-08 bit-identical A/B bug)")
    assert np.all(np.abs(sh_gr - sh_dyer) > 0.05 * np.abs(sh_dyer))
