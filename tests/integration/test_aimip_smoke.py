"""Smoke tests for the AIMIP intercomparison pipeline.

Covers:

1. :class:`legoesm.training.aimip_params.AIMIPClassicalParams` — defaults,
   sigmoid-constrained ``as_dict``, ``to_*_config`` builders preserve
   each scheme's published defaults.

2. ``make_aimip_classical_spectral_physics`` — builds a callable
   physics_fn without import-time errors and exposes Tiedtke /
   Louis / Surface / McFarlane via ``combined.make_physics`` with
   ``model_type='spectral_pe'``.

3. End-to-end differentiability: a synthetic loss over
   ``params.as_dict()`` returns finite gradients on every
   ``AIMIP_CLASSICAL_CONSTRAINTS`` entry (validates that the Equinox
   pytree is traceable through ``eqx.filter_value_and_grad``).

4. ``ExperimentConfig.aimip_variant`` field round-trips through
   ``experiment_config_to_dict`` / ``experiment_config_from_dict``
   and rejects unknown variants in ``validate_strict``.

5. ``ml.training.TrainingConfig(optimizer='muon')`` builds a valid
   optimizer via ``create_optimizer``.

The full ERA5-driven smoke run lives in ``scripts/run_aimip.py
--smoke`` and is exercised by users; it is omitted here to keep
CI cost bounded (network IO + spectral transforms).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pytest

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------
# Section 1: AIMIPClassicalParams
# ----------------------------------------------------------------------

def test_aimip_classical_params_defaults():
    from legoesm.training.aimip_params import (
        AIMIP_CLASSICAL_CONSTRAINTS,
        AIMIPClassicalParams,
    )
    params = AIMIPClassicalParams.from_defaults()
    assert len(params.raw_values) == len(AIMIP_CLASSICAL_CONSTRAINTS)
    # Constraint names match raw_values keys exactly.
    names_raw = set(params.raw_values.keys())
    names_constraints = {c.name for c in AIMIP_CLASSICAL_CONSTRAINTS}
    assert names_raw == names_constraints


def test_aimip_classical_params_as_dict_within_bounds():
    from legoesm.training.aimip_params import AIMIPClassicalParams
    params = AIMIPClassicalParams.from_defaults()
    d = params.as_dict()
    for c in params.constraints:
        v = float(d[c.name])
        assert math.isfinite(v)
        # Sigmoid output is strictly within [lo, hi].
        assert c.min_val <= v <= c.max_val, (
            f"{c.name}={v} outside [{c.min_val}, {c.max_val}]"
        )


def test_aimip_classical_params_to_tiedtke_preserves_defaults():
    """``to_tiedtke_config`` at defaults equals the canonical TiedtkeConfig."""
    from legoesm.atmosphere.physics.convection.config import TiedtkeConfig
    from legoesm.training.aimip_params import AIMIPClassicalParams
    params = AIMIPClassicalParams.from_defaults()
    cfg = params.to_tiedtke_config()
    canon = TiedtkeConfig()
    for field in (
        "tau_M_u_relax", "tau_MC_proxy", "cape_threshold",
        "precip_efficiency", "downdraft_alpha", "downdraft_RH_min",
    ):
        assert math.isclose(
            float(getattr(cfg, field)),
            float(getattr(canon, field)),
            rel_tol=1e-3, abs_tol=1e-3,
        ), f"{field}: cfg={getattr(cfg, field)} canon={getattr(canon, field)}"


def test_aimip_classical_params_to_louis_preserves_defaults():
    from legoesm.atmosphere.physics.turbulence.config import LouisConfig
    from legoesm.training.aimip_params import AIMIPClassicalParams
    params = AIMIPClassicalParams.from_defaults()
    cfg = params.to_louis_config()
    canon = LouisConfig()
    for field in ("l_mix_max", "Ck", "Ri_crit", "b_louis", "c_louis"):
        assert math.isclose(
            float(getattr(cfg, field)),
            float(getattr(canon, field)),
            rel_tol=1e-3, abs_tol=1e-3,
        )
    # Surface sub-config also threaded.
    assert math.isclose(
        float(cfg.surface.Cd_neutral),
        float(canon.surface.Cd_neutral),
        rel_tol=1e-3,
    )


def test_aimip_classical_params_to_mcfarlane_preserves_defaults():
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        McFarlaneConfig,
    )
    from legoesm.training.aimip_params import AIMIPClassicalParams
    params = AIMIPClassicalParams.from_defaults()
    cfg = params.to_mcfarlane_config()
    canon = McFarlaneConfig()
    for field in (
        "h_topo", "G_0", "efficiency", "min_wind", "envelope_scale",
    ):
        assert math.isclose(
            float(getattr(cfg, field)),
            float(getattr(canon, field)),
            rel_tol=1e-3, abs_tol=1e-3,
        )


def test_aimip_classical_params_to_cloud_scheme_xu_randall():
    from legoesm.training.aimip_params import AIMIPClassicalParams
    params = AIMIPClassicalParams.from_defaults()
    cfg = params.to_cloud_config()
    assert cfg.scheme == "xu_randall"


# ----------------------------------------------------------------------
# Section 2: physics_fn builder + differentiability
# ----------------------------------------------------------------------

def test_make_aimip_classical_spectral_physics_callable():
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        make_aimip_classical_spectral_physics,
    )
    params = AIMIPClassicalParams.from_defaults()
    grid = create_gaussian_grid(21, dealiasing="quadratic")
    fn = make_aimip_classical_spectral_physics(params, grid, dt=1800.0)
    assert callable(fn)


def test_aimip_classical_params_gradient_through_synthetic_loss():
    """eqx.filter_value_and_grad returns finite gradients on every knob."""
    from legoesm.training.aimip_params import AIMIPClassicalParams

    params = AIMIPClassicalParams.from_defaults()

    def synthetic_loss(p):
        d = p.as_dict()
        # Sum constrained values normalized by their bound widths so
        # every knob contributes a finite, well-scaled gradient.
        total = jnp.array(0.0, dtype=jnp.float64)
        for c in p.constraints:
            total = total + d[c.name] / (c.max_val - c.min_val)
        return total

    loss, grads = eqx.filter_value_and_grad(synthetic_loss)(params)
    assert math.isfinite(float(loss))
    # Every raw-value leaf has a non-zero finite gradient.
    for name, g in grads.raw_values.items():
        gv = float(g)
        assert math.isfinite(gv), f"NaN/Inf gradient on {name}: {gv}"
        assert abs(gv) > 0.0, f"Zero gradient on {name}"


# ----------------------------------------------------------------------
# Section 3: ExperimentConfig.aimip_variant validation + round-trip
# ----------------------------------------------------------------------

def test_experiment_config_aimip_variant_default_empty():
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig()
    assert cfg.aimip_variant == ""
    cfg.validate_strict()  # default config (no aimip) must still validate


def test_experiment_config_aimip_variant_classical_accepts_correct_schemes():
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig(
        aimip_variant="classical",
        convection="tiedtke",
        turbulence="louis",
        gravity_wave_drag="mcfarlane",
        cloud_scheme="xu_randall",
    )
    cfg.validate_strict()  # should not raise


def test_experiment_config_aimip_variant_rejects_unknown():
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig(aimip_variant="bogus")
    with pytest.raises(ValueError, match="aimip_variant"):
        cfg.validate_strict()


def test_experiment_config_aimip_variant_classical_requires_tiedtke():
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig(
        aimip_variant="classical",
        convection="sbm",          # wrong: must be tiedtke
        turbulence="louis",
        gravity_wave_drag="mcfarlane",
        cloud_scheme="xu_randall",
    )
    with pytest.raises(ValueError, match="tiedtke"):
        cfg.validate_strict()


@pytest.mark.parametrize("variant", ["column_nn", "sfno_physics", "sfno_full"])
def test_experiment_config_aimip_variant_non_classical_round_trip(variant):
    """The column_nn / sfno_* variants have no prerequisite scheme
    constraints; ``validate_strict`` must accept them out of the box."""
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig(aimip_variant=variant)
    cfg.validate_strict()
    assert cfg.aimip_variant == variant


# ----------------------------------------------------------------------
# Section 4: Muon optimizer dispatch
# ----------------------------------------------------------------------

def test_create_optimizer_muon_returns_gradient_transformation():
    from legoesm.ml.training import TrainingConfig, create_optimizer
    cfg = TrainingConfig(
        lr=1e-3, warmup_steps=2, total_steps=10, optimizer="muon",
    )
    opt = create_optimizer(cfg)
    # Smoke-init on a dummy param tree.
    params = {"w": jnp.ones((4, 4))}
    state = opt.init(params)
    assert state is not None


def test_create_optimizer_rejects_unknown():
    from legoesm.ml.training import TrainingConfig, create_optimizer
    with pytest.raises(ValueError, match="optimizer"):
        create_optimizer(TrainingConfig(optimizer="not_an_optimizer"))


# ----------------------------------------------------------------------
# Section 5: run_aimip dispatch smoke
# ----------------------------------------------------------------------

def test_run_aimip_module_importable_and_lists_variants():
    """``scripts/run_aimip.py`` imports cleanly and exposes the variant set."""
    import importlib.util
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_aimip_run", repo_root / "scripts" / "run_aimip.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module._VALID_VARIANTS) >= {
        "classical", "column_nn", "sfno_physics",
    }
