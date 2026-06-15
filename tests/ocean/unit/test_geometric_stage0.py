"""Shared GEOMETRIC Stage-0 twin machinery (truth + ETKI recovery).

Covers the pure pieces — observation-vector assembly, the streaming mean/std
accumulator, exact-IC ``restore_state`` roundtrip, and the parameter-error
diagnostic — without the expensive forward integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
from legoesm.ocean.fidelity import geometric_stage0 as g0


@pytest.fixture(autouse=True)
def _fp64():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def test_accumulator_mean_std_matches_numpy():
    acc = g0.Accumulator(("T", "S", "psi", "eke"))
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(7):
        s = {"T": rng.normal(size=(3, 4, 2)), "S": rng.normal(size=(3, 4, 2)),
             "psi": rng.normal(size=(4, 5)), "eke": rng.normal(size=(3, 4))}
        acc.add(s)
        samples.append(s)
    out = acc.finalize()
    for k in ("T", "S", "psi", "eke"):
        stack = np.stack([s[k] for s in samples])
        assert np.allclose(out[f"mean_{k}"], stack.mean(0))
        assert np.allclose(out[f"std_{k}"], stack.std(0), atol=1e-10)


def _synthetic_truth_obs(rng):
    land = np.ones((4, 5)); land[0, 0] = 0.0          # one dry cell
    obs = {}
    for f, shp in (("T", (4, 5, 3)), ("S", (4, 5, 3)),
                   ("psi", (5, 6)), ("eke", (4, 5))):
        obs[f"mean_{f}"] = rng.normal(size=shp)
        obs[f"std_{f}"] = np.abs(rng.normal(size=shp))
    return obs, land


def test_observation_target_self_consistent():
    rng = np.random.default_rng(1)
    obs, land = _synthetic_truth_obs(rng)
    y, r_diag, packer = g0.build_observation_target(obs, land)
    # Packing the truth itself reproduces y exactly → zero self-misfit.
    g = packer(obs)
    assert np.array_equal(g, y)
    assert float(np.sum((y - g) ** 2 / r_diag)) == 0.0
    # r_diag strictly positive + finite (no division blow-up in ETKI).
    assert np.all(np.isfinite(r_diag)) and np.all(r_diag > 0)
    # Dry T/S/eke cells are excluded (psi uses all cells); count the masked total.
    # Each field contributes a mean AND a std map (×2). T and S are 3-level.
    n_wet = int((land > 0.5).sum())          # 19 of 20
    expect = (2 * (n_wet * 3)    # T (mean+std, 3 levels)
              + 2 * (n_wet * 3)  # S (mean+std, 3 levels)
              + 2 * n_wet        # eke (mean+std, 2-D)
              + 2 * (5 * 6))     # psi (mean+std, all vorticity-grid cells)
    assert y.shape[0] == expect


def test_observation_target_perturbed_has_positive_misfit():
    rng = np.random.default_rng(2)
    obs, land = _synthetic_truth_obs(rng)
    y, r_diag, packer = g0.build_observation_target(obs, land)
    perturbed = {k: v + 0.5 for k, v in obs.items()}
    g = packer(perturbed)
    assert float(np.sum((y - g) ** 2 / r_diag)) > 0.0


def test_relative_param_error_zero_at_truth():
    from legoesm.ocean.physics.lateral_mixing.eke import GeometricConfig
    from legoesm.training.trainable_ocean_params import (
        GEOMETRIC_TRAINABLE, TrainableOceanParams,
    )
    true_geom = GeometricConfig()
    # Ensemble all sitting exactly at the true (default) params.
    raw = TrainableOceanParams.from_defaults(GEOMETRIC_TRAINABLE).raw
    theta = np.broadcast_to(np.asarray(raw), (5, raw.shape[0]))
    errs = g0.relative_param_error(theta, true_geom, GEOMETRIC_TRAINABLE)
    for name, (val, true, rel) in errs.items():
        assert rel < 1e-6, f"{name}: rel={rel}"


def test_restore_state_roundtrip():
    """restore_state reconstructs the prognostic + carry fields from a snapshot."""
    import jax.numpy as jnp
    from legoesm.ocean.physics.lateral_mixing.eke import GeometricConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    recipe = g0.build_geometric_recipe(GeometricConfig())
    cfg = recipe.model_config
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    seeded = g0.seed_freerun_carries(model, recipe.initial_state, cfg, recipe.grid)

    # Build a "snapshot" dict with perturbed values to prove restore overwrites.
    snap = {}
    for f in seeded._fields:
        obj = getattr(seeded, f)
        if obj is None:
            continue
        if hasattr(obj, "data"):
            snap[f] = np.asarray(obj.data) + 1.0      # perturb every Field
        elif isinstance(obj, jnp.ndarray):
            snap[f] = np.asarray(obj) + 2.0           # perturb every carry

    restored = g0.restore_state(recipe.initial_state, snap,
                                model=model, cfg=cfg, grid=recipe.grid)
    # Field leaves and plain-array carries both take the snapshot values.
    assert np.allclose(np.asarray(restored.T.data), snap["T"])
    assert np.allclose(np.asarray(restored.eke.data), snap["eke"])
    assert restored.eke.units == "m^3/s^2"            # metadata preserved
    assert np.allclose(np.asarray(restored.psi), snap["psi"])
    assert np.allclose(np.asarray(restored.dpsin), snap["dpsin"])
