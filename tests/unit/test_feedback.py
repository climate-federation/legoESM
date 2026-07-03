"""Unit tests for :mod:`legoesm.training.feedback`.

Stage 7 application: the hardened strategy dispatch for building the feedback
field and the bridge that splices a per-column field into a scheme *Config
(traced-in-loss).  Checks dispatch raises, both strategies, shape validation,
unknown-field raise, and differentiability through the whole chain.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.feedback import (
    apply_column_parameter_field,
    build_parameter_field,
)


class _ToyConfig(NamedTuple):
    """Minimal scheme-like config: a scalar default that the feedback promotes
    to a per-column array leaf.  ``entrainment_coeff`` is declared column-
    promoted via a ``shape``-keyed ``__param_spec__``; ``other`` is not."""

    entrainment_coeff: float = 0.1
    other: float = 2.0


# Canonical nested __param_spec__ layout (keyed by class -> "params" -> field),
# matching real legoESM scheme configs.
_ToyConfig.__param_spec__ = {
    "_ToyConfig": {
        "scheme_key": "toy",
        "params": {
            "entrainment_coeff": {"shape": "n_col", "units": "1", "tunable_tier": 2},
            "other": {"shape": None},
        },
    }
}
_PROMOTED = frozenset({"entrainment_coeff"})


class _FlatSpecConfig(NamedTuple):
    """Config carrying a FLAT class-attached spec ({field: {...}})."""

    k: float = 0.3


_FlatSpecConfig.__param_spec__ = {"k": {"shape": "n_col"}}


def test_build_static_field():
    field = build_parameter_field(
        "static", grid_shape=(2, 2),
        flat_indices=jnp.array([0, 3]), values=jnp.array([1.0, 2.0]),
        background=-1.0,
    )
    assert field.shape == (2, 2)
    f = np.asarray(field).reshape(-1)
    assert f[0] == pytest.approx(1.0)
    assert f[3] == pytest.approx(2.0)
    assert f[1] == pytest.approx(-1.0)


def test_build_environment_field():
    grid_env = jnp.array([[300.0], [250.0], [300.0], [250.0]])
    field = build_parameter_field(
        "environment", grid_shape=(2, 2),
        grid_env=grid_env,
        sample_env=jnp.array([[300.0]]), sample_values=jnp.array([0.5]),
        length_scales=jnp.array([10.0]), background=-9.0,
    )
    assert field.shape == (2, 2)
    f = np.asarray(field).reshape(-1)
    # Columns at SST 300 recover the sample; SST 250 (5σ) -> background.
    assert f[0] == pytest.approx(0.5, rel=1e-9)
    assert f[2] == pytest.approx(0.5, rel=1e-9)
    assert f[1] == pytest.approx(-9.0)


def test_build_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown feedback strategy"):
        build_parameter_field("bogus", grid_shape=(2, 2))


def test_build_static_missing_inputs_raises():
    with pytest.raises(ValueError, match="flat_indices and values"):
        build_parameter_field("static", grid_shape=(2, 2))


def test_build_environment_missing_inputs_raises():
    with pytest.raises(ValueError, match="requires"):
        build_parameter_field(
            "environment", grid_shape=(2, 2),
            grid_env=jnp.zeros((4, 1)),  # other inputs missing
        )


def test_build_environment_grid_mismatch_raises():
    with pytest.raises(ValueError, match="column vector"):
        build_parameter_field(
            "environment", grid_shape=(2, 2),
            grid_env=jnp.zeros((3, 1)),  # 3 rows != 4 columns
            sample_env=jnp.zeros((1, 1)), sample_values=jnp.array([1.0]),
            length_scales=jnp.array([1.0]),
        )


def test_build_static_rejects_environment_inputs():
    with pytest.raises(ValueError, match="environment-only input"):
        build_parameter_field(
            "static", grid_shape=(2, 2),
            flat_indices=jnp.array([0]), values=jnp.array([1.0]),
            grid_env=jnp.zeros((4, 1)),  # leaked from the other strategy
        )


def test_build_environment_rejects_static_inputs():
    with pytest.raises(ValueError, match="static-only input"):
        build_parameter_field(
            "environment", grid_shape=(2, 2),
            grid_env=jnp.zeros((4, 1)), sample_env=jnp.zeros((1, 1)),
            sample_values=jnp.array([1.0]), length_scales=jnp.array([1.0]),
            flat_indices=jnp.array([0]),  # leaked from the other strategy
        )


def test_apply_rejects_scalar_field():
    cfg = _ToyConfig()
    with pytest.raises(ValueError, match="scalar/0-d"):
        apply_column_parameter_field(cfg, "entrainment_coeff", 1.0)
    with pytest.raises(ValueError, match="scalar/0-d"):
        apply_column_parameter_field(cfg, "entrainment_coeff", jnp.asarray(1.0))


def test_build_environment_non_1d_output_raises(monkeypatch):
    """A non-1-D kernel output (right size, wrong rank) must be rejected."""
    import legoesm.training.feedback as fb

    monkeypatch.setattr(
        fb, "environment_kernel_field",
        lambda *a, **k: jnp.zeros((4, 1)),  # size 4 but rank 2
    )
    with pytest.raises(ValueError, match="column vector"):
        build_parameter_field(
            "environment", grid_shape=(2, 2),
            grid_env=jnp.zeros((4, 1)), sample_env=jnp.zeros((1, 1)),
            sample_values=jnp.array([1.0]), length_scales=jnp.array([1.0]),
        )


def test_apply_splices_per_column_field():
    cfg = _ToyConfig()
    field = jnp.array([[0.2, 0.3], [0.4, 0.5]])
    new_cfg = apply_column_parameter_field(
        cfg, "entrainment_coeff", field, promoted_fields=_PROMOTED)
    assert new_cfg.entrainment_coeff.shape == (4,)
    np.testing.assert_allclose(
        np.asarray(new_cfg.entrainment_coeff), [0.2, 0.3, 0.4, 0.5])
    # Production scalar default is unchanged on the original.
    assert cfg.entrainment_coeff == 0.1
    assert new_cfg.other == 2.0


def test_apply_authorizes_via_nested_param_spec_shape_key():
    """Without an allowlist, the nested __param_spec__ shape key authorizes."""
    cfg = _ToyConfig()
    new_cfg = apply_column_parameter_field(
        cfg, "entrainment_coeff", jnp.zeros(4))  # spec shape='n_col'
    assert new_cfg.entrainment_coeff.shape == (4,)


def test_apply_authorizes_via_flat_param_spec_shape_key():
    cfg = _FlatSpecConfig()
    new_cfg = apply_column_parameter_field(cfg, "k", jnp.zeros(4))
    assert new_cfg.k.shape == (4,)


def test_apply_rejects_unpromoted_field():
    cfg = _ToyConfig()
    # 'other' has shape=None in the spec and is not in the allowlist.
    with pytest.raises(ValueError, match="not declared column-promoted"):
        apply_column_parameter_field(cfg, "other", jnp.zeros(4))
    with pytest.raises(ValueError, match="not in promoted_fields"):
        apply_column_parameter_field(
            cfg, "other", jnp.zeros(4), promoted_fields=_PROMOTED)


def test_apply_unknown_field_raises():
    cfg = _ToyConfig()
    # Authorize the (unknown) name so we reach apply_param_overrides' check.
    with pytest.raises(ValueError, match="no field"):
        apply_column_parameter_field(
            cfg, "not_a_field", jnp.zeros(4),
            promoted_fields=frozenset({"not_a_field"}))


def test_apply_expected_ncol_mismatch_raises():
    cfg = _ToyConfig()
    with pytest.raises(ValueError, match="!= expected_ncol"):
        apply_column_parameter_field(
            cfg, "entrainment_coeff", jnp.zeros(3), expected_ncol=4,
            promoted_fields=_PROMOTED)


def test_feedback_chain_differentiable():
    """End-to-end: diagnosed values -> field -> config leaf, under jax.grad."""
    cfg = _ToyConfig()

    def loss(values):
        field = build_parameter_field(
            "static", grid_shape=(2, 2),
            flat_indices=jnp.array([0, 1, 2, 3]), values=values,
        )
        new_cfg = apply_column_parameter_field(cfg, "entrainment_coeff", field)
        return jnp.sum(new_cfg.entrainment_coeff ** 2)

    g = jax.grad(loss)(jnp.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(np.asarray(g), [2.0, 4.0, 6.0, 8.0], rtol=1e-12)
