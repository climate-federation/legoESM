"""Deploy a saved correction-campaign output as a production turbulence override.

Covers ``legoesm.training.deploy_correction`` — the loader that closes the
"campaign output JSON → production AMIP/CMIP run" loop (iter 57): it must turn
both campaign output shapes (single ``--diagnosis-method`` and multi
``--coefficients``) into a per-column ``TurbulenceConfig`` that ``ExperimentConfig.
validate_strict`` accepts, and raise LOUDLY on a malformed / non-CLUBB output.
"""

from __future__ import annotations

import json

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
from legoesm.training.deploy_correction import (
    assert_deploy_compatible,
    corrected_clubb_config,
    corrected_turbulence_override,
    grid_fingerprint,
)


class _FakeGrid:
    """Duck-typed grid for the deploy guard: 2-D lat/lon column field + ncol.

    ``make_adapter`` reads ``grid_n_columns`` + ``grid_lat.shape``; the
    fingerprint reads flattened ``grid_lat``/``grid_lon`` — so this exercises the
    coordinate hash without standing up a full grid object.
    """

    def __init__(self, lat, lon):
        self.grid_lat = jnp.asarray(lat)
        self.grid_lon = jnp.asarray(lon)

    @property
    def grid_n_columns(self):
        return int(self.grid_lat.reshape(-1).size)


def _grid_8x16(lon_shift=0.0, dtype=jnp.float64):
    # Radian-range coordinates (the model convention) so the float32/float64
    # fingerprint is dtype-invariant (abs error < 5e-7 < the 1e-6 round step).
    lat = jnp.linspace(-1.5, 1.5, 128, dtype=dtype).reshape(8, 16)
    lon = (jnp.linspace(0.0, 6.0, 128, dtype=dtype) + lon_shift).reshape(8, 16)
    return _FakeGrid(lat, lon)


def _single_output(field="C_K", vals=(0.5, 0.6, 0.7)):
    # Mirrors the single-coefficient campaign JSON (top-level field array + extras).
    return {field: list(vals), "biases": [1.0, 0.5], "accepted": [True],
            "step_fractions": [1.0]}


def _multi_output(fields=None):
    fields = fields or {"clubb_lite_C_K": [0.5, 0.6],
                        "clubb_lite_Pr_t": [0.4, 0.5],
                        "clubb_lite_C_eps": [0.1, 0.2]}
    return {"coefficients": ["C_K", "Pr_t", "C_eps"], "fields": fields,
            "biases": [1.0, 0.5]}


# --------------------------------------------------------------------------- #
# corrected_clubb_config — both shapes
# --------------------------------------------------------------------------- #
def test_single_output_builds_per_column_field():
    cfg = corrected_clubb_config(_single_output("Pr_t", (0.3, 0.4, 0.5)))
    assert isinstance(cfg, CLUBBLiteConfig)
    np.testing.assert_allclose(np.asarray(cfg.Pr_t), [0.3, 0.4, 0.5])
    # The untouched fields keep their scalar defaults (byte-identical to prod).
    assert float(cfg.C_K) == float(CLUBBLiteConfig().C_K)
    assert float(cfg.C_eps) == float(CLUBBLiteConfig().C_eps)


def test_multi_output_builds_all_three_fields():
    cfg = corrected_clubb_config(_multi_output())
    np.testing.assert_allclose(np.asarray(cfg.C_K), [0.5, 0.6])
    np.testing.assert_allclose(np.asarray(cfg.Pr_t), [0.4, 0.5])
    np.testing.assert_allclose(np.asarray(cfg.C_eps), [0.1, 0.2])


def test_multi_output_partial_fields():
    cfg = corrected_clubb_config(_multi_output({"clubb_lite_C_K": [0.5, 0.6]}))
    np.testing.assert_allclose(np.asarray(cfg.C_K), [0.5, 0.6])
    assert float(cfg.Pr_t) == float(CLUBBLiteConfig().Pr_t)


# --------------------------------------------------------------------------- #
# corrected_turbulence_override — dict / path / file
# --------------------------------------------------------------------------- #
def test_override_from_dict():
    over = corrected_turbulence_override(_multi_output())
    assert isinstance(over, TurbulenceConfig)
    assert over.scheme == "clubb_lite"
    np.testing.assert_allclose(np.asarray(over.clubb_lite.C_K), [0.5, 0.6])


def test_override_from_path_and_file(tmp_path):
    p = tmp_path / "campaign_out.json"
    p.write_text(json.dumps(_single_output("C_eps", (0.1, 0.2))))
    over_path = corrected_turbulence_override(str(p))
    np.testing.assert_allclose(np.asarray(over_path.clubb_lite.C_eps), [0.1, 0.2])
    with open(p) as f:
        over_file = corrected_turbulence_override(f)
    np.testing.assert_allclose(np.asarray(over_file.clubb_lite.C_eps), [0.1, 0.2])


# --------------------------------------------------------------------------- #
# The whole point: the loaded override is accepted by validate_strict
# --------------------------------------------------------------------------- #
def test_loaded_override_passes_validate_strict():
    over = corrected_turbulence_override(_multi_output())
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite", turbulence_override=over,
    )
    cfg.validate_strict()  # must not raise — override is a clubb_lite TurbulenceConfig


def test_scheme_mismatch_rejected_by_validate_strict():
    # Deploying a clubb_lite override onto a non-clubb base must fail loudly.
    over = corrected_turbulence_override(_multi_output())
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="none", turbulence_override=over,
    )
    with pytest.raises((ValueError, AssertionError)):
        cfg.validate_strict()


# --------------------------------------------------------------------------- #
# Dispatch hardening — malformed / non-CLUBB output raises
# --------------------------------------------------------------------------- #
def test_unknown_promotion_key_raises():
    with pytest.raises(ValueError, match="unsupported promotion_key"):
        corrected_clubb_config({"fields": {"gray_tau_equator": [1.0, 2.0]}})


def test_empty_output_raises():
    # No "fields" + no C_K/Pr_t/C_eps → single branch, zero recognized fields.
    with pytest.raises(ValueError, match="exactly one"):
        corrected_clubb_config({"biases": [1.0]})


def test_single_output_with_two_fields_raises():
    with pytest.raises(ValueError, match="exactly one"):
        corrected_clubb_config({"C_K": [0.5], "Pr_t": [0.4]})


def test_empty_fields_dict_raises():
    with pytest.raises(ValueError, match="no corrected CLUBB"):
        corrected_clubb_config({"fields": {}})


def test_non_dict_toplevel_raises():
    with pytest.raises(ValueError, match="must be a JSON object"):
        corrected_clubb_config([0.5, 0.6])


def test_non_dict_fields_raises():
    with pytest.raises(ValueError, match="'fields' must be a dict"):
        corrected_clubb_config({"fields": None})


def test_empty_array_raises():
    with pytest.raises(ValueError, match="is empty"):
        corrected_clubb_config({"C_K": []})


def test_non_finite_array_raises():
    with pytest.raises(ValueError, match="non-finite"):
        corrected_clubb_config({"C_K": [0.5, float("nan")]})


def test_rank2_array_raises():
    with pytest.raises(ValueError, match="1-D per-column"):
        corrected_clubb_config({"C_K": [[0.5, 0.6]]})


def test_arrays_are_jax_for_differentiable_deploy():
    over = corrected_turbulence_override(_multi_output())
    assert isinstance(over.clubb_lite.C_K, jnp.ndarray)


# --------------------------------------------------------------------------- #
# Grid-compatibility guard (iter 58): per-column coefficients must belong on the
# DEPLOY grid, or they silently land on the wrong cells.
# --------------------------------------------------------------------------- #
def test_grid_fingerprint_deterministic():
    # Same grid construction → identical fingerprint (stable hash, not salted
    # Python hash) — the campaign→deploy normal case (same config/dtype).
    fp = grid_fingerprint(_grid_8x16(dtype=jnp.float64))
    fpb = grid_fingerprint(_grid_8x16(dtype=jnp.float64))
    assert fp == fpb
    assert fp["ncol"] == 128 and fp["shape_2d"] == [8, 16]
    assert len(fp["coord_sha256"]) == 64  # sha256 hexdigest


def test_grid_fingerprint_distinguishes_same_shape_different_coords():
    a = grid_fingerprint(_grid_8x16())
    b = grid_fingerprint(_grid_8x16(lon_shift=1.0))  # same ncol+shape, diff coords
    assert a["ncol"] == b["ncol"] and a["shape_2d"] == b["shape_2d"]
    assert a["coord_sha256"] != b["coord_sha256"]


def _provenanced_output(grid, ncol=128):
    return {"C_K": list(np.linspace(0.3, 0.8, ncol)),
            "grid": grid_fingerprint(grid)}


def test_compatible_grid_passes():
    grid = _grid_8x16()
    assert_deploy_compatible(_provenanced_output(grid), grid)  # no raise
    over = corrected_turbulence_override(_provenanced_output(grid), grid=grid)
    assert over.scheme == "clubb_lite"


def test_length_mismatch_raises():
    out = _provenanced_output(_grid_8x16(), ncol=128)
    smaller = _FakeGrid(jnp.zeros((8, 8)), jnp.zeros((8, 8)))  # 64 columns
    with pytest.raises(ValueError, match="spans 128 columns but the target grid"):
        assert_deploy_compatible(out, smaller)


def test_same_ncol_different_grid_raises():
    # THE silent-corruption case: same ncol + shape, different coordinates.
    out = _provenanced_output(_grid_8x16())
    other = _grid_8x16(lon_shift=1.0)
    with pytest.raises(ValueError, match="coord_sha256"):
        assert_deploy_compatible(out, other)


def test_missing_provenance_is_explicit_error():
    grid = _grid_8x16()
    out = {"C_K": list(np.linspace(0.3, 0.8, 128))}  # no "grid" block
    with pytest.raises(ValueError, match="lacks a 'grid'"):
        assert_deploy_compatible(out, grid)
    # ...but length still matches, so the escape hatch deploys.
    assert_deploy_compatible(out, grid, allow_unverified_grid=True)


def test_null_coord_hash_is_not_a_bypass():
    # A present-but-null coord_sha256 must NOT silently skip the identity check
    # (Codex Q3): treat it as missing provenance, not a verified match.
    grid = _grid_8x16()
    out = {"C_K": list(np.linspace(0.3, 0.8, 128)),
           "grid": {"ncol": 128, "shape_2d": [8, 16], "coord_sha256": None}}
    with pytest.raises(ValueError, match="coord_sha256 is null"):
        assert_deploy_compatible(out, grid)
    assert_deploy_compatible(out, grid, allow_unverified_grid=True)  # escape hatch


@pytest.mark.parametrize("block", [None, "bad", {"coord_sha256": ""}])
def test_malformed_grid_block_is_unverified_not_crash(block):
    # A null / non-dict / empty-hash grid block must route to the explicit
    # unverified-provenance ValueError, never an AttributeError (Codex follow-up).
    grid = _grid_8x16()
    out = {"C_K": list(np.linspace(0.3, 0.8, 128)), "grid": block}
    with pytest.raises(ValueError, match="cannot be verified"):
        assert_deploy_compatible(out, grid)
    assert_deploy_compatible(out, grid, allow_unverified_grid=True)


def test_allow_unverified_still_enforces_length():
    grid = _grid_8x16()
    out = {"C_K": list(np.linspace(0.3, 0.8, 64))}  # wrong length, no provenance
    with pytest.raises(ValueError, match="spans 64 columns"):
        assert_deploy_compatible(out, grid, allow_unverified_grid=True)


def test_override_without_grid_skips_check():
    # Backward compat: grid=None deploys with no grid verification.
    over = corrected_turbulence_override({"C_K": [0.5, 0.6]})
    assert over.scheme == "clubb_lite"


def test_real_latlon_grid_roundtrip():
    import jax.numpy as _jnp
    from legoesm.grids.latlon import create_latlon_grid
    grid = create_latlon_grid(8, 16, dtype=_jnp.float64)
    out = _provenanced_output(grid, ncol=int(grid.grid_n_columns))
    assert_deploy_compatible(out, grid)  # deploys onto its own grid
    other = create_latlon_grid(16, 8, dtype=_jnp.float64)  # 128 cols, diff shape
    with pytest.raises(ValueError):
        assert_deploy_compatible(out, other)


def test_integer_json_coerced_to_float_for_autodiff():
    # A hand-edited integer-valued coefficient must NOT enter as a non-diff int.
    cfg = corrected_clubb_config({"C_K": [1, 2]})
    assert jnp.issubdtype(cfg.C_K.dtype, jnp.floating)
    np.testing.assert_allclose(np.asarray(cfg.C_K), [1.0, 2.0])
