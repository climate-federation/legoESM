"""Unit tests for the ORCA2 K0 twin-training script's pure pieces (no decks needed)."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64, reason="needs JAX_ENABLE_X64=1")

_P = (Path(__file__).resolve().parents[3]
      / "scripts/validate/ocean_fidelity/orca2_l4/orca2_k0_twin_training.py")
_spec = importlib.util.spec_from_file_location("twin", _P)
twin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(twin)


def test_perturbation_scales_and_reflects_inside_bounds():
    vals = {"a": 1.0, "b": 9.0}
    out = twin.perturbed(vals, {"a": (0.5, 2.0), "b": (1.0, 10.0)}, 1.25)
    assert out["a"] == 1.25 and out["b"] == pytest.approx(9.0 / 1.25)
    with pytest.raises(ValueError, match="cannot be perturbed"):
        twin.perturbed({"c": 1.0}, {"c": (0.9, 1.1)}, 1.25)
    # x1.25 lands within the collector's edge margin of hi: reflected instead
    out = twin.perturbed({"d": 8.0}, {"d": (0.0, 10.0 + 0.005)}, 1.25)
    assert out["d"] == pytest.approx(8.0 / 1.25)


class Tke(NamedTuple):
    c_k: float = 0.1
    c_eps: float = 0.7


class Treg(NamedTuple):
    aei0: float = 3000.0


class Gm(NamedTuple):
    kappa_redi_diffusive_velocity: float = 0.018
    treguier: Treg = Treg()


class VM(NamedTuple):
    tke: Tke = Tke()


class Phys(NamedTuple):
    vertical_mixing: VM = VM()


class Cfg(NamedTuple):
    physics: Phys = Phys()
    gm_redi: Gm = Gm()


def test_with_params_reaches_both_subconfigs_and_leaves_the_model_alone():
    model = SimpleNamespace(config=Cfg())
    m = twin.with_params(model, {twin.TKE: {"c_k": 0.2},
                                 twin.GM: {"kappa_redi_diffusive_velocity": 0.02},
                                 twin.TREG: {"aei0": 2000.0}})
    assert m.config.physics.vertical_mixing.tke.c_k == 0.2
    assert m.config.physics.vertical_mixing.tke.c_eps == 0.7
    assert m.config.gm_redi.kappa_redi_diffusive_velocity == 0.02
    assert m.config.gm_redi.treguier.aei0 == 2000.0
    assert model.config == Cfg() and m is not model
    with pytest.raises(ValueError, match="no injection path"):
        twin.with_params(model, {"ocean.vm.iwm": {"x": 1.0}})


def test_misfit_is_zero_at_truth_and_ignores_deep_and_dry_cells():
    z = SimpleNamespace(is_active=np.ones((2, 2, 3)), t_depth_ref=np.array([10.0, 500.0, 2000.0]),
                        z_full_ref=None, h_partial=np.ones((2, 2, 3)))
    with pytest.raises(ValueError, match="t_depth_ref"):
        twin.misfit(SimpleNamespace(**{**vars(z), "t_depth_ref": None}), np.ones((2, 2)), 1000.0)
    z.is_active[0, 0, :] = 0.0
    loss = twin.misfit(z, np.ones((2, 2)), depth_m=1000.0)
    rng = np.random.default_rng(0)

    def state(T, S, eta):
        return SimpleNamespace(T=SimpleNamespace(data=jnp.asarray(T)),
                               S=SimpleNamespace(data=jnp.asarray(S)),
                               eta=SimpleNamespace(data=jnp.asarray(eta)))

    T, S, eta = rng.normal(size=(2, 2, 3)), rng.normal(size=(2, 2, 3)), rng.normal(size=(2, 2))
    truth = state(T, S, eta)
    assert float(loss(truth, truth)) == 0.0
    T2 = T.copy(); T2[..., 2] += 5.0; T2[0, 0, :] += 5.0          # deep level and dry column
    assert float(loss(state(T2, S, eta), truth)) == pytest.approx(0.0, abs=1e-15)
    T3 = T.copy(); T3[1, 1, 0] += 1.0
    assert float(loss(state(T3, S, eta), truth)) > 0.0
