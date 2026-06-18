"""Cross-resolution deploy via the environment kernel (iter 69).

The per-column array deploy is GRID-LOCKED (the iter-58 guard rejects a different
grid). The environment kernel is GRID-AGNOSTIC: the campaign's diagnosed
(environment, coefficient) samples evaluate on ANY grid's environment by
environmental similarity — so a cheap low-res campaign deploys on the expensive
high-res run. These tests cover build → serialize → deserialize → apply on a
DIFFERENT-ncol grid, the coverage diagnostic, and the hardening guards.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.training.column_manifest import ColumnEnvironment
from legoesm.training.deploy_correction import (
    apply_env_kernel_override,
    build_env_kernel,
    env_kernel_from_dict,
    env_kernel_to_dict,
)
from legoesm.training.feedback_assembly import reduce_column_diagnosis


class _Diag(NamedTuple):
    C_K: jax.Array
    valid: jax.Array


class _Rec(NamedTuple):
    environment: ColumnEnvironment


_LENGTH_SCALES = [2.0, 1200.0, 8.0]  # [SST_K, CAPE_J/kg, shear_m/s] ~ the sample std
#   (like column_environment_grid's std-based scales — so the kernel covers the range)
# 3 diagnosed columns spanning an environment range.
_ENVS = [(298.0, 100.0, 5.0), (300.0, 1500.0, 12.0), (302.0, 2800.0, 20.0)]
_CKS = [0.30, 0.45, 0.62]


def _kernel():
    recs = [_Rec(ColumnEnvironment(*e)) for e in _ENVS]
    diags = [_Diag(C_K=jnp.array([c, c]), valid=jnp.array([True, True]))
             for c in _CKS]
    return build_env_kernel(recs, diags, "clubb_coefficient",
                            length_scales=_LENGTH_SCALES, field="C_K")


def test_build_matches_reduce_column_diagnosis():
    # The kernel samples must EQUAL the shared reducer's output (no divergence).
    diags = [_Diag(C_K=jnp.array([c, c]), valid=jnp.array([True, True]))
             for c in _CKS]
    expected = [float(reduce_column_diagnosis(d, "clubb_coefficient")[0])
                for d in diags]
    k = _kernel()
    np.testing.assert_allclose(np.asarray(k.sample_values), expected)
    assert bool(np.all(np.asarray(k.valid)))
    np.testing.assert_allclose(np.asarray(k.env_lo), np.min(_ENVS, axis=0))
    np.testing.assert_allclose(np.asarray(k.env_hi), np.max(_ENVS, axis=0))


def test_apply_on_different_ncol_grid():
    # Deploy on a DIFFERENT-resolution grid (5 columns vs 3 samples) — grid-agnostic.
    k = _kernel()
    new_env = jnp.asarray([
        (298.5, 200.0, 6.0),    # near sample 0 → ~0.30
        (301.0, 2000.0, 15.0),  # between 1 and 2
        (302.0, 2800.0, 20.0),  # == sample 2 → ~0.62
        (299.0, 800.0, 9.0),
        (300.5, 1600.0, 13.0),
    ])
    override, coverage = apply_env_kernel_override(k, new_env)
    assert isinstance(override, TurbulenceConfig)
    ck = np.asarray(override.clubb_lite.C_K)
    assert ck.shape == (5,)                          # any ncol, not the campaign's 3
    assert np.all(np.isfinite(ck))
    assert np.all((ck >= 0.30 - 1e-6) & (ck <= 0.62 + 1e-6))  # within sample range
    assert coverage["n_columns"] == 5
    assert coverage["fraction_covered"] == 1.0       # all near a sample
    assert coverage["fraction_in_hull"] == 1.0


def test_apply_out_of_hull_falls_back_and_warns():
    # A grid whose climate lies OUTSIDE the sampled hull → background fallback +
    # low coverage (the domain-shift warning Codex asked for).
    k = _kernel()
    far = jnp.asarray([(250.0, 5e4, 200.0), (255.0, 6e4, 250.0)])  # absurd env
    override, coverage = apply_env_kernel_override(k, far)
    ck = np.asarray(override.clubb_lite.C_K)
    np.testing.assert_allclose(ck, 0.0)              # == background (no neighbor)
    assert coverage["fraction_covered"] == 0.0
    assert coverage["fraction_in_hull"] == 0.0


def test_json_round_trip(tmp_path):
    import json
    k = _kernel()
    p = tmp_path / "kernel.json"
    p.write_text(json.dumps(env_kernel_to_dict(k)))
    k2 = env_kernel_from_dict(json.loads(p.read_text()))
    new_env = jnp.asarray([(300.0, 1500.0, 12.0), (298.5, 150.0, 5.5)])
    o1, _ = apply_env_kernel_override(k, new_env)
    o2, _ = apply_env_kernel_override(k2, new_env)
    np.testing.assert_allclose(
        np.asarray(o1.clubb_lite.C_K), np.asarray(o2.clubb_lite.C_K))


def test_deployed_override_passes_validate_strict():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    k = _kernel()
    override, _ = apply_env_kernel_override(
        k, jnp.asarray([(300.0, 1500.0, 12.0)] * 6))
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite", turbulence_override=override)
    cfg.validate_strict()


# --------------------------------------------------------------------------- #
# Hardening guards.
# --------------------------------------------------------------------------- #
def test_build_rejects_bad_field():
    recs = [_Rec(ColumnEnvironment(*_ENVS[0]))]
    diags = [_Diag(C_K=jnp.array([0.3]), valid=jnp.array([True]))]
    with pytest.raises(ValueError, match="must be one of"):
        build_env_kernel(recs, diags, "clubb_coefficient",
                         length_scales=_LENGTH_SCALES, field="bogus")


def test_build_rejects_misaligned_or_empty_or_allinvalid():
    rec = _Rec(ColumnEnvironment(*_ENVS[0]))
    diag = _Diag(C_K=jnp.array([0.3]), valid=jnp.array([True]))
    with pytest.raises(ValueError, match="must align"):
        build_env_kernel([rec, rec], [diag], "clubb_coefficient",
                         length_scales=_LENGTH_SCALES)
    with pytest.raises(ValueError, match="at least one"):
        build_env_kernel([], [], "clubb_coefficient", length_scales=_LENGTH_SCALES)
    invalid = _Diag(C_K=jnp.array([0.3]), valid=jnp.array([False]))
    with pytest.raises(ValueError, match="no VALID"):
        build_env_kernel([rec], [invalid], "clubb_coefficient",
                         length_scales=_LENGTH_SCALES)


def test_build_rejects_bad_length_scales():
    recs = [_Rec(ColumnEnvironment(*_ENVS[0]))]
    diags = [_Diag(C_K=jnp.array([0.3]), valid=jnp.array([True]))]
    with pytest.raises(ValueError, match="length_scales"):
        build_env_kernel(recs, diags, "clubb_coefficient", length_scales=[1.0, 2.0])


def test_from_dict_rejects_mislabelled_artifact():
    # A wrong/foreign artifact tag fails loudly; a missing tag is tolerated
    # (pre-tag iter-69 kernels still deserialize).
    k = _kernel()
    d = env_kernel_to_dict(k)
    assert d["artifact"] == "raw_environment_kernel"
    bad = dict(d, artifact="accepted_campaign_field")
    with pytest.raises(ValueError, match="raw_environment_kernel"):
        env_kernel_from_dict(bad)
    no_tag = {key: v for key, v in d.items() if key != "artifact"}
    env_kernel_from_dict(no_tag)   # back-compat: missing tag OK


def test_apply_rejects_wrong_grid_env_shape():
    k = _kernel()
    with pytest.raises(ValueError, match=r"must be \(ncol"):
        apply_env_kernel_override(k, jnp.asarray([0.3, 0.4, 0.5]))   # 1-D
    with pytest.raises(ValueError, match=r"must be \(ncol"):
        apply_env_kernel_override(k, jnp.zeros((4, 2)))              # wrong npred
