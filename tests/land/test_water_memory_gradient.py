"""Tests for the water-memory gradient harness
(``scripts/experiment/water_memory_gradient.py``).

Validates the three feasibility properties on a tiny synthetic time-varying case,
without external data:
  A the per-cell memory gradient is finite through the two-leaf canopy Newton solve
    and matches a finite-difference check (the diagonal-Jacobian trick);
  B a NaN-poisoned cell stays confined to its own gradient entry (columns
    independent) -> good cells' gradients remain finite;
  C jax.checkpoint on the scan body is EXACT (same gradient as no-checkpoint).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# Reverse-mode AD through a multi-step canopy-Newton scan is COMPILE-bound (~4 min),
# so this is opt-in: it runs under `-m slow`, out of the default fast lane.
pytestmark = pytest.mark.slow

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "experiment" / "water_memory_gradient.py")
_spec = importlib.util.spec_from_file_location("water_memory_gradient", _SCRIPT)
wm = importlib.util.module_from_spec(_spec)
sys.modules["water_memory_gradient"] = wm
_spec.loader.exec_module(wm)

_NCOL, _NSTEPS, _DT = 6, 60, 3600.0


def _case():
    cfg = wm._SYNTH_CFG
    F, doy_seq, late_flags, lat = wm.synthetic_window(_NCOL, _NSTEPS, dt=_DT, seed=1)
    # DRY start (near wilting 0.15) so the column is water-limited -> the memory
    # gradient is genuinely non-zero (a wet, unstressed column gives d GPP/d theta = 0,
    # which would make the test vacuous).
    st0 = wm.init_multilayer_land_state(_NCOL, cfg, T_init=296.0, theta_init=0.18,
                                        TgC_init=23.0)
    return cfg, F, doy_seq, late_flags, lat, st0


def test_memory_gradient_finite_and_matches_fd():
    """A: finite per-cell gradient + AD == FD (diagonal trick)."""
    cfg, F, doy_seq, late_flags, lat, st0 = _case()
    good = jnp.ones(_NCOL, bool)
    M, late = wm.water_memory_map(st0, F, doy_seq, late_flags, cfg, lat=lat, dt=_DT,
                                  good_mask=good, use_checkpoint=True)
    M = np.asarray(M)
    assert np.all(np.isfinite(M)), "memory gradient must be finite over all cells"
    # a drying, zero-precip, water-limited window: extra spring water -> more late GPP,
    # so the memory is POSITIVE and non-trivial (guards against a vacuous all-zero pass).
    assert np.all(M >= -1e-6), f"memory should be non-negative in a drying column, got {M.min()}"
    assert M.max() > 1e-3, f"expected a non-zero memory signal, got max {M.max():.3e}"

    # finite-difference check on a couple of cells (diagonal: perturb cell k only)
    def S(dtheta):
        theta_p = st0.theta_soil + dtheta[:, None]
        st = st0._replace(theta_soil=theta_p,
                          psi_soil=wm.psi_from_theta(theta_p, cfg.hydraulics))
        def body(carry, xs):
            st, acc = carry
            Fi, doy_i, latef = xs
            ns, gpp = wm._step_and_gpp(st, Fi, doy_i, cfg, lat, _DT, None)
            return (ns, acc + jnp.where(latef, gpp, 0.0)), None
        (_s, lg), _ = jax.lax.scan(body, (st, jnp.zeros(_NCOL)), (F, doy_seq, late_flags))
        return float(jnp.sum(lg) * _DT)

    eps = 1e-4
    for k in (0, _NCOL - 1):
        fd = (S(jnp.zeros(_NCOL).at[k].set(eps))
              - S(jnp.zeros(_NCOL).at[k].set(-eps))) / (2 * eps)
        rel = abs(M[k] - fd) / (abs(fd) + 1e-30)
        assert rel < 0.15, f"cell {k}: AD {M[k]:.3e} vs FD {fd:.3e} (rel {rel:.2e})"


def test_checkpoint_is_exact():
    """C: jax.checkpoint gives the SAME gradient as no-checkpoint."""
    cfg, F, doy_seq, late_flags, lat, st0 = _case()
    good = jnp.ones(_NCOL, bool)
    M_ck, _ = wm.water_memory_map(st0, F, doy_seq, late_flags, cfg, lat=lat, dt=_DT,
                                  good_mask=good, use_checkpoint=True)
    M_no, _ = wm.water_memory_map(st0, F, doy_seq, late_flags, cfg, lat=lat, dt=_DT,
                                  good_mask=good, use_checkpoint=False)
    assert np.max(np.abs(np.asarray(M_ck) - np.asarray(M_no))) < 1e-8


def test_nan_cell_is_confined():
    """B: a NaN-poisoned cell leaves the other cells' gradients finite (masked out)."""
    cfg, F, doy_seq, late_flags, lat, st0 = _case()
    F_bad = F._replace(sw_down=F.sw_down.at[:, 0].set(jnp.nan))   # poison cell 0
    good = jnp.ones(_NCOL, bool).at[0].set(False)
    M, _ = wm.water_memory_map(st0, F_bad, doy_seq, late_flags, cfg, lat=lat, dt=_DT,
                               good_mask=good, use_checkpoint=True)
    M = np.asarray(M)
    assert np.all(np.isfinite(M[1:])), "good cells must stay finite when cell 0 is NaN"


def test_monthly_kernel_sums_to_full_window():
    """The temporal memory kernel summed over ALL months equals the full-window
    integrated map: Σ_m ∂(month-m GPP_i)/∂θ_i = ∂(∫_window GPP_i)/∂θ_i. Validates the
    month-binning + stacking + sign of monthly_memory_kernel (the synthetic window's
    doy straddles Jun/Jul, so >=2 months are spanned -> a non-trivial check)."""
    cfg, F, doy_seq, late_flags, lat, st0 = _case()
    good = jnp.ones(_NCOL, bool)
    months, K = wm.monthly_memory_kernel(st0, F, doy_seq, cfg, lat=lat, dt=_DT,
                                         good_mask=good, use_checkpoint=True)
    assert len(months) >= 2, f"window should span >=2 months, got {months}"
    K_sum = np.nansum(np.asarray(K), axis=0)
    nsteps = int(np.asarray(F.sw_down).shape[0])
    M_full, _ = wm.water_memory_map(st0, F, doy_seq, jnp.ones(nsteps, bool), cfg,
                                    lat=lat, dt=_DT, good_mask=good, use_checkpoint=True)
    assert np.allclose(K_sum, np.asarray(M_full), atol=1e-6, rtol=1e-5), (
        f"kernel sum {K_sum} != full-window map {np.asarray(M_full)}")


def test_spin_forward_is_nograd_and_finite():
    """The no-grad spin-to-perturb-point runs and returns a finite state."""
    cfg, F, doy_seq, late_flags, lat, st0 = _case()
    st1 = wm.spin_forward(st0, F, doy_seq, cfg, lat=lat, dt=_DT)
    assert np.all(np.isfinite(np.asarray(st1.theta_soil)))
    assert np.all(np.isfinite(np.asarray(st1.T_soil)))
