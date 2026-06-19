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


def test_covered_but_out_of_hull_diagnostics_diverge():
    """``fraction_covered`` and ``fraction_in_hull`` are INDEPENDENT diagnostics and
    must diverge in the soft domain-shift boundary zone.

    The two existing tests only hit the degenerate cases where they coincide (both
    1.0 all-in-hull, both 0.0 far-away), so a regression that made ``in_hull`` track
    coverage (e.g. deriving the hull from the length-scales instead of the sample
    bounding box) would pass them.  Here one deploy column sits JUST below the SST
    hull (297 K < the 298 K sample minimum) but only 0.5 SST-length-scales from
    sample 0, so it is COVERED (has a kernel neighbor) yet OUT of hull; a second
    column equals sample 1 exactly (covered AND in hull).  So ``fraction_covered``
    (1.0) must exceed ``fraction_in_hull`` (0.5) — the boundary-zone warning the
    in-hull diagnostic exists to surface (a low fraction_in_hull at full coverage
    means the deploy grid is extrapolating just past the sampled envelope).
    """
    k = _kernel()
    # col 0: covered (≈ sample 0) but SST below the hull min ⇒ out of hull.
    # col 1: identical to sample 1 ⇒ covered AND in hull.
    env = jnp.asarray([(297.0, 100.0, 5.0), (300.0, 1500.0, 12.0)])
    override, coverage = apply_env_kernel_override(k, env)
    assert coverage["fraction_covered"] == 1.0       # BOTH have a kernel neighbor
    assert coverage["fraction_in_hull"] == 0.5       # only the sample-1 column
    assert coverage["fraction_covered"] > coverage["fraction_in_hull"]  # diverge
    # The covered-but-out-of-hull column still gets a BOUNDED blend (NW kernel never
    # extrapolates beyond the sampled values) — not background, not a wild value.
    ck = np.asarray(override.clubb_lite.C_K)
    assert np.all(np.isfinite(ck))
    assert np.all((ck >= 0.30 - 1e-6) & (ck <= 0.62 + 1e-6))  # within sample range
    assert ck[0] != pytest.approx(k.background)      # NOT the background fallback


def test_apply_min_fraction_covered_fails_loud_on_no_op_deploy():
    """Opt-in fail-loud: an (near-)all-background deploy (grid outside the sampled
    environments ⇒ NO-OP correction) RAISES when ``min_fraction_covered`` is set,
    so a misconfigured cross-grid deploy is caught before a multi-day run."""
    k = _kernel()
    far = jnp.asarray([(250.0, 5e4, 200.0), (255.0, 6e4, 250.0)])  # coverage 0.0
    # Default (None) preserves the original behaviour: NO raise even at 0 coverage.
    _override, cov = apply_env_kernel_override(k, far)
    assert cov["fraction_covered"] == 0.0
    # Opt-in guard fires on the no-op deploy.
    with pytest.raises(ValueError, match="below the required min_fraction_covered"):
        apply_env_kernel_override(k, far, min_fraction_covered=0.5)
    # A well-covered deploy passes the same guard (coverage 1.0 >= 0.5).
    good = jnp.asarray([(300.0, 1500.0, 12.0), (298.5, 150.0, 5.5)])
    override, cov_ok = apply_env_kernel_override(k, good, min_fraction_covered=0.5)
    assert cov_ok["fraction_covered"] >= 0.5
    assert np.all(np.isfinite(np.asarray(override.clubb_lite.C_K)))
    # An out-of-range threshold itself fails loud.
    with pytest.raises(ValueError, match=r"min_fraction_covered must be in \[0, 1\]"):
        apply_env_kernel_override(k, good, min_fraction_covered=1.5)


def test_apply_rejects_negative_min_total_weight():
    """A NEGATIVE min_total_weight silently disables the background fallback (every
    column reads ``total >= a_negative_floor`` = True ⇒ a 'neighbor', none falls back),
    the opposite of the intended protection — so it fails loud at the call (iter 265)."""
    k = _kernel()
    good = jnp.asarray([(300.0, 1500.0, 12.0), (298.5, 150.0, 5.5)])
    with pytest.raises(ValueError, match=r"min_total_weight must be >= 0"):
        apply_env_kernel_override(k, good, min_total_weight=-0.1)
    # 0.0 (any nonzero similarity counts) is accepted — only NEGATIVE is rejected.
    override, _ = apply_env_kernel_override(k, good, min_total_weight=0.0)
    assert np.all(np.isfinite(np.asarray(override.clubb_lite.C_K)))


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


def test_from_dict_ignores_extra_averaging_provenance_block():
    """The env-kernel write site stamps an iter-269 ``averaging`` provenance block
    (snapshot vs climatology the kernel was trained against) alongside the kernel keys;
    ``env_kernel_from_dict`` reads only the kernel keys, so the extra block is INERT on
    reload — the kernel reconstructs bit-identically. Locks that the write-site stamp
    cannot break the cross-resolution deploy round-trip."""
    import json

    k = _kernel()
    d = env_kernel_to_dict(k)
    d["averaging"] = {"era5_time_idx": 12, "era5_n_times": 30}     # the write-site stamp
    k2 = env_kernel_from_dict(json.loads(json.dumps(d)))
    np.testing.assert_array_equal(
        np.asarray(k2.length_scales), np.asarray(k.length_scales))
    new_env = jnp.asarray([(300.0, 1500.0, 12.0), (298.5, 150.0, 5.5)])
    o1, _ = apply_env_kernel_override(k, new_env)
    o2, _ = apply_env_kernel_override(k2, new_env)
    np.testing.assert_allclose(
        np.asarray(o1.clubb_lite.C_K), np.asarray(o2.clubb_lite.C_K))   # inert extra block


def test_json_round_trip_preserves_every_field_exactly():
    """The env-kernel JSON (the cross-grid deploy persistence artifact) must
    round-trip ALL fields bit-exactly — the env-kernel analog of the iter-214
    checkpoint-field lock.

    ``test_json_round_trip`` above checks only behavioural equivalence via the
    deployed ``C_K``, which does NOT depend on ``env_lo``/``env_hi`` (they feed only
    the ``fraction_in_hull`` domain-shift diagnostic).  So a corruption / mis-order
    of the hull bounds — or a flatten of the ``(nsamp, 3)`` ``sample_env`` — would
    pass that test yet silently break the deploy's domain-shift warning on the
    production grid.  This asserts each field is preserved exactly, including the
    2-D ``sample_env`` shape.
    """
    import json

    k = _kernel()
    k2 = env_kernel_from_dict(json.loads(json.dumps(env_kernel_to_dict(k))))
    # The 2-D sample_env keeps its (nsamp, 3) shape (not flattened/transposed) …
    assert np.asarray(k2.sample_env).shape == np.asarray(k.sample_env).shape
    np.testing.assert_allclose(
        np.asarray(k2.sample_env), np.asarray(k.sample_env), rtol=1e-12)
    # … and every other field round-trips exactly — env_lo/env_hi are the ones the
    # behavioural C_K test cannot reach.
    np.testing.assert_allclose(
        np.asarray(k2.sample_values), np.asarray(k.sample_values), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(k2.length_scales), np.asarray(k.length_scales), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(k2.env_lo), np.asarray(k.env_lo), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(k2.env_hi), np.asarray(k.env_hi), rtol=1e-12)
    np.testing.assert_array_equal(
        np.asarray(k2.valid, dtype=bool), np.asarray(k.valid, dtype=bool))
    assert k2.field == k.field
    assert float(k2.background) == pytest.approx(float(k.background))


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
def test_build_rejects_non_finite_environment_tag():
    """A worst column whose environment tag is non-finite (e.g. a NaN SST over land)
    must FAIL LOUD — not silently build a kernel whose ``env_lo``/``env_hi`` = min/max
    propagate the NaN AND whose `<out>.env_kernel.json` carries a non-standard
    ``NaN``/``Infinity`` token (unparseable by the cross-grid deploy reader). Consistent
    with the iter-179 clustering/feedback fail-loud convention (iter 247)."""
    recs = [_Rec(ColumnEnvironment(sst_K=float("nan"), cape_J_kg=1500.0,
                                   bulk_shear_m_s=10.0)),
            _Rec(ColumnEnvironment(*_ENVS[1]))]
    diags = [_Diag(C_K=jnp.array([0.4, 0.4]), valid=jnp.array([True, True]))
             for _ in recs]
    with pytest.raises(ValueError, match="non-finite"):
        build_env_kernel(recs, diags, "clubb_coefficient",
                         length_scales=_LENGTH_SCALES, field="C_K")


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


def test_build_rejects_nonpositive_length_scales():
    """A length_scale is a similarity-distance DIVISOR; it must be strictly positive.
    The evaluator floors any scale at 1e-30 (so it never divides by zero — no NaN), but
    that means a ZERO or NEGATIVE scale is silently floored to ~0, collapsing that
    predictor to an exact-match-only kernel ⇒ a silent near-all-background no-op deploy.
    Reject it at construction (the production scales come from the per-predictor std)."""
    recs = [_Rec(ColumnEnvironment(*_ENVS[0]))]
    diags = [_Diag(C_K=jnp.array([0.3]), valid=jnp.array([True]))]
    for bad in ([2.0, 0.0, 8.0], [2.0, -1200.0, 8.0]):   # one zero / one negative scale
        with pytest.raises(ValueError, match="strictly POSITIVE"):
            build_env_kernel(recs, diags, "clubb_coefficient", length_scales=bad,
                             field="C_K")


def test_kernel_field_nonpositive_scale_is_exact_match_only_not_nan():
    """Lock the ACTUAL degenerate behavior the construction guard prevents: the
    lower-level ``environment_kernel_field`` FLOORS a non-positive scale at 1e-30, so a
    ZERO (and, identically, a NEGATIVE) scale yields a FINITE exact-match-only field (a
    grid column matching a sample keeps its value; a non-matching one → background) —
    NOT a NaN.  This is why the iter-257 guard fails loud on a silent no-op, not a crash.
    """
    from legoesm.training.parameter_field import environment_kernel_field

    grid = jnp.array([[298.0], [305.0]])        # col 0 matches sample 0; col 1 matches none
    senv = jnp.array([[298.0], [300.0]])
    svals = jnp.array([0.3, 0.45])
    valid = jnp.array([True, True])
    zero = environment_kernel_field(grid, senv, svals, length_scales=jnp.array([0.0]),
                                    background=99.0, valid=valid)
    neg = environment_kernel_field(grid, senv, svals, length_scales=jnp.array([-5.0]),
                                   background=99.0, valid=valid)
    assert bool(jnp.all(jnp.isfinite(zero)))                  # floored to 1e-30 — no NaN
    assert float(zero[0]) == pytest.approx(0.3)               # exact match keeps its value
    assert float(zero[1]) == pytest.approx(99.0)              # no match → background no-op
    np.testing.assert_allclose(np.asarray(neg), np.asarray(zero))  # negative ≡ zero (floored)


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


def test_from_dict_rejects_non_dict_and_missing_required_keys():
    """A truncated/corrupted kernel artifact fails loud with a CLEAR message (iter 304):
    a non-dict top-level, and a dict missing any required key, raise a named ValueError
    instead of a bare KeyError/AttributeError — parallel to the iter-106 ERA5-loader
    required-key hardening. 'artifact' stays optional (pre-tag back-compat)."""
    d = env_kernel_to_dict(_kernel())
    with pytest.raises(ValueError, match="must be a JSON object"):
        env_kernel_from_dict([1, 2, 3])                          # a JSON list, not object
    for key in ("sample_env", "sample_values", "valid", "length_scales", "field",
                "background", "env_lo", "env_hi"):
        truncated = {k: v for k, v in d.items() if k != key}
        with pytest.raises(ValueError, match=rf"missing required key.*{key}"):
            env_kernel_from_dict(truncated)


def test_from_dict_rejects_mismatched_hull_bound_length():
    """A corrupted/hand-edited kernel JSON with a wrong-length ``env_lo``/``env_hi`` is
    rejected on deserialize (iter 284) — the validator's per-field length loop guards
    all 5 fields but only ``length_scales`` was tested.  ``env_lo``/``env_hi`` are the
    HULL bounds (the deploy trust region, iter 217); a length-1 ``env_lo`` would silently
    BROADCAST and corrupt the in-hull domain-shift check, so it must fail LOUD."""
    d = env_kernel_to_dict(_kernel())                            # 3-predictor kernel
    for field in ("env_lo", "env_hi"):
        bad = dict(d)
        bad[field] = [1.0, 2.0]                                  # length 2, not npred=3
        with pytest.raises(ValueError, match=rf"env kernel '{field}' length"):
            env_kernel_from_dict(bad)


def test_apply_rejects_wrong_grid_env_shape():
    k = _kernel()
    with pytest.raises(ValueError, match=r"must be \(ncol"):
        apply_env_kernel_override(k, jnp.asarray([0.3, 0.4, 0.5]))   # 1-D
    with pytest.raises(ValueError, match=r"must be \(ncol"):
        apply_env_kernel_override(k, jnp.zeros((4, 2)))              # wrong npred
