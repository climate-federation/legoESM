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


def test_percell_injection_at_step0_equals_initial_perturbation():
    """water_memory_map_percell with onset=0 and late_lag=0 (inject dtheta at step 0,
    count all GPP) must equal the initial-state-perturbation map (water_memory_map with
    all-True late_flags): injecting at step 0 IS perturbing the initial state. Validates
    the per-cell injection + psi-recompute + per-cell late-window machinery."""
    cfg, F, doy_seq, late_flags, lat, st0 = _case()
    good = jnp.ones(_NCOL, bool)
    nsteps = int(np.asarray(F.sw_down).shape[0])
    M_init, _ = wm.water_memory_map(st0, F, doy_seq, jnp.ones(nsteps, bool), cfg,
                                    lat=lat, dt=_DT, good_mask=good, use_checkpoint=True)
    M_pc, _ = wm.water_memory_map_percell(
        st0, F, doy_seq, cfg, lat=lat, dt=_DT,
        onset_step=jnp.zeros(_NCOL, jnp.int32), late_lo_steps=0,
        good_mask=good, use_checkpoint=True)
    assert np.allclose(np.asarray(M_pc), np.asarray(M_init), atol=1e-6, rtol=1e-5), (
        f"per-cell(onset=0,lag=0) {np.asarray(M_pc)} != initial-pert {np.asarray(M_init)}")


def test_growing_season_kernel_bins_sum_to_map():
    """The growing-season kernel bins (time-since-onset, non-overlapping, covering the
    window) must SUM to the per-cell map with late_lo=0 (all post-onset GPP): the bins
    partition [onset, end). Validates the since-onset bin construction."""
    cfg, F, doy_seq, late_flags, lat, st0 = _case()
    good = jnp.ones(_NCOL, bool)
    onset = jnp.zeros(_NCOL, jnp.int32)                    # onset at step 0 (simplest)
    nsteps = int(np.asarray(F.sw_down).shape[0])
    bin_steps = nsteps // 2 + 1                            # 2 bins covering the window
    K = wm.growing_season_kernel(st0, F, doy_seq, cfg, lat=lat, dt=_DT, onset_step=onset,
                                 bin_steps=bin_steps, n_bins=2, good_mask=good,
                                 use_checkpoint=True)
    K_sum = np.nansum(np.asarray(K), axis=0)
    M_all, _ = wm.water_memory_map_percell(st0, F, doy_seq, cfg, lat=lat, dt=_DT,
                                           onset_step=onset, late_lo_steps=0,
                                           good_mask=good, use_checkpoint=True)
    assert np.allclose(K_sum, np.asarray(M_all), atol=1e-6, rtol=1e-5), (
        f"bin-sum {K_sum} != full post-onset map {np.asarray(M_all)}")


def test_forward_window_diag_runs_and_detects_onset():
    """_forward_window_diag scans without a carry-dtype error and returns sensible
    per-cell diagnostics. The synthetic forcing is warm + snow-free, so onset = step 0
    for all cells. (This exercises the scan whose int32/int64 onset-carry mismatch broke
    the growing-season run -- a run_real-only path the other tests don't cover.)"""
    cfg, F, doy_seq, late_flags, lat, st0 = _case()
    ok, msnow, mtsoil, onset = wm._forward_window_diag(
        st0, F, doy_seq, cfg, lat, _DT, None, min_onset_doy=float(np.asarray(doy_seq)[0]))
    onset = np.asarray(onset); nsteps = int(np.asarray(F.sw_down).shape[0])
    assert np.all(np.asarray(ok)), "warm synthetic cells should stay finite"
    assert np.all((onset >= 0) & (onset <= nsteps)), f"onset out of range: {onset}"
    assert np.all(onset == 0), f"warm snow-free cells should onset at step 0, got {onset}"
    assert np.all(np.asarray(msnow) < 1.0)   # no snow in the warm synthetic case


def test_forward_window_diag_lai_onset_gates_on_greenup():
    """With an lai_fn that stays LEAFLESS (LAI=0) for the first `g` steps then greens up,
    onset must land exactly at `g` -- not at step 0 (snow-free from the start). This is
    the LAI-based green-up onset: it keeps the leafless LAI~0 canopy-Newton singularity
    (TODO-1) out of the differentiated boreal/arctic gradient path, instead of firing the
    moment the snow clears. Without lai_fn the warm synthetic onsets at 0 (covered above)."""
    cfg, F, doy_seq, late_flags, lat, st0 = _case()
    g = 20                                                     # green-up step
    ncol = _NCOL

    def lai_fn(theta_top, doy):
        # leafless (0) before green-up doy, leafed-out (2) after -- keyed off doy so the
        # scan sees a step-varying LAI without threading a step counter into lai_fn.
        return jnp.where(doy >= float(np.asarray(doy_seq)[g]), 2.0, 0.0) * jnp.ones(ncol)

    ok, msnow, mtsoil, onset = wm._forward_window_diag(
        st0, F, doy_seq, cfg, lat, _DT, None,
        min_onset_doy=float(np.asarray(doy_seq)[0]), lai_fn=lai_fn, lai_thresh=0.5)
    onset = np.asarray(onset)
    assert np.all(onset == g), f"LAI-gated onset should fire at green-up step {g}, got {onset}"


def test_spin_forward_is_nograd_and_finite():
    """The no-grad spin-to-perturb-point runs and returns a finite state."""
    cfg, F, doy_seq, late_flags, lat, st0 = _case()
    st1 = wm.spin_forward(st0, F, doy_seq, cfg, lat=lat, dt=_DT)
    assert np.all(np.isfinite(np.asarray(st1.theta_soil)))
    assert np.all(np.isfinite(np.asarray(st1.T_soil)))
