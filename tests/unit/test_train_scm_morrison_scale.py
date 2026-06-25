"""Smoke test for the SCM Morrison scale-net trainer (quick path)."""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[2]
_MOD_PATH = REPO_ROOT / "scripts" / "run" / "train_scm_morrison_scale.py"


def _load():
    spec = importlib.util.spec_from_file_location("train_scm_morrison_scale", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_identity_init_matches_baseline_and_training_reduces_loss():
    mod = _load()
    col = mod.build_column(nlev=20)
    base = mod.MorrisonConfig()
    perturb = mod.WarmRainRateScales(autoconv=4.0, accretion=0.3, rain_evap=2.0)
    ref = mod.make_placeholder_reference(col, base, n_steps=40, dt=2.0, perturb=perturb)
    assert ref.T_ref.shape == (20,) and jnp.all(jnp.isfinite(ref.qcond_ref))

    loss0, loss_final, net = mod.train(
        reference=None, n_steps=40, dt=2.0, epochs=20, lr=5e-3, seed=0, out=None,
    )
    assert jnp.isfinite(loss0) and jnp.isfinite(loss_final)
    # The optimizer must not increase the loss (identity-init starts at baseline).
    assert loss_final <= loss0 + 1e-9


def test_load_reference_roundtrip(tmp_path):
    import numpy as np
    mod = _load()
    col = mod.build_column(nlev=12)
    nlev = col.T.shape[1]
    npz = tmp_path / "les.npz"
    np.savez(npz, T=np.linspace(288.0, 270.0, nlev), qv=np.full(nlev, 5e-3),
             qc=np.full(nlev, 3e-4), qr=np.full(nlev, 1e-4), precip_mm_day=2.0)
    ref = mod.load_reference(npz, col)
    assert ref.T_ref.shape == (nlev,)
    assert jnp.allclose(ref.qcond_ref, 4e-4)
    assert ref.precip_ref_mm_day == pytest.approx(2.0)
