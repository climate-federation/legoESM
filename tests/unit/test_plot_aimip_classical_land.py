"""The AIMIP plot rollouts hand the classical physics a grid carrying the ERA5
land fraction (user 2026-10-28, Q33: every AIMIP run).  The physics reads the
grid the rollout passes it, so attaching to a copy the rollout never sees is
the same as not attaching."""
from __future__ import annotations

import importlib.util
import types
from collections import namedtuple
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_Spec = namedtuple("_Spec", "pe_config dt n_levels n_train_days windows rollout_hours")


class _Stop(Exception):
    pass


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_land_probe", _ROOT / "scripts" / "plot" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("script", ["plot_aimip_time_series", "plot_aimip_bias_maps"])
def test_classical_rollout_grid_carries_the_land_fraction(monkeypatch, script):
    import legoesm.atmosphere.dynamics.gcm.spectral_pe as spe
    import legoesm.training.aimip_params as ap
    import legoesm.training.aimip_spatial as sp
    import legoesm.training.neural_gcm_spectral as ngs
    from legoesm.grids.gaussian import create_gaussian_grid

    grid = create_gaussian_grid(8, dealiasing="quadratic")
    land = jnp.asarray(np.random.default_rng(0).random((grid.n_lat, grid.n_lon)))
    seen = []

    def fake_rollout(state, physics_fn, grid_, *a, **k):
        seen.append(grid_)
        raise _Stop

    monkeypatch.setattr(sp, "era5_land_fraction", lambda g, *a, **k: land)
    monkeypatch.setattr(ap, "make_aimip_classical_spectral_physics",
                        lambda *a, **k: (lambda *a, **k: None))
    monkeypatch.setattr(ngs, "spectral_rollout", fake_rollout)
    monkeypatch.setattr(ngs, "load_training_data",
                        lambda *a, **k: ([object()], [object()], [0]))
    monkeypatch.setattr(spe, "spectral_pe_to_grid",
                        lambda *a, **k: {"T": np.zeros((grid.n_lat, grid.n_lon, 4))})

    pe = types.SimpleNamespace(sponge_tau=0.0, spectral_filter_strength=0.0)
    spec_cfg = _Spec(pe, 1800.0, 4, 1, (), 24)
    mod = _load(script)
    with pytest.raises(_Stop):
        if script == "plot_aimip_time_series":
            mod._run_variant_series("classical", None, grid, None, spec_cfg, {},
                                    object(), 1, np.ones(grid.n_lat))
        else:
            mod._compute_bias_for_variant("classical", None, grid, None, spec_cfg,
                                          {}, [2000], [], 1, "unused")
    np.testing.assert_array_equal(np.asarray(seen[0].land_frac),
                                  np.asarray(land).reshape(-1))
