"""OSSE / twin demonstration: the correction loop *reduces a known bias*.

The real-model capstone (``test_correction_loop_real_rerun``) proves the loop
CLOSES — the LES-informed per-column coefficient genuinely reaches the re-run's
turbulence kernel — but deliberately does NOT assert the SIGN of the bias change,
because the real model's true coefficient sensitivity needs real ERA5 at HPC
scale to pin down.

This test isolates and PROVES the other half: the loop LOGIC (detect worst column
→ diagnose → inject per-column → re-run → measure bias → monotonic ``improved``
flag) actually LOWERS the bias when handed a correct diagnosis.  It does so with a
fully controllable, deterministic synthetic ``run_fn`` whose only response is a
known monotonic ``T = T0 + alpha·C_K`` — so a "perfect" diagnosis (the truth
coefficient) MUST drive the re-run onto the reference and zero the bias.

Decomposition (each leg locked independently):
  * the per-column injection reaches the REAL kernel ........ capstone (real model)
  * the LES diagnoses the physically correct coefficient .... diagnosis locks (iter 353-355)
  * GIVEN a correct diagnosis, the loop REDUCES the bias .... THIS test (synthetic, sign asserted)
Only the empirical COMPOSITION at real-ERA5 HPC scale remains out of reach here.

The synthetic ``run_fn`` exercises the REAL loop machinery end-to-end:
``make_compare_fn`` (worst-column ranking + area-weighted bias) →
``run_correction_iteration`` (diagnose → ``assemble_feedback_field`` →
``apply_feedback_to_scheme`` → re-run → ``bias_improvement``).  Nothing about the
correction code is mocked; only the physics ``T(C_K)`` is a pinned linear law.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.training.compare_reanalysis import ColumnState  # noqa: E402
from legoesm.training.correction_loop import (  # noqa: E402
    make_compare_fn,
    run_correction_iteration,
)

_NLAT, _NLON, _NLEV = 4, 8, 3
_N = _NLAT * _NLON
_WROW, _WCOL = _NLAT // 2, _NLON // 2
_WFLAT = _WROW * _NLON + _WCOL          # the one perturbed (worst) column, row-major

_C_K_DEFAULT = float(CLUBBLiteConfig().C_K)   # 0.4 — the biased baseline / background
_C_K_OBS = 0.9                                # the "truth" coefficient (within bounds)
_ALPHA_K_PER_CK = 12.0                        # T sensitivity [K per unit C_K]

# Fixed, C_K-independent background fields (surface-last profiles); only T responds
# to C_K, so the bias is driven entirely by the turbulence coefficient.
_T0 = jnp.broadcast_to(jnp.linspace(285.0, 235.0, _NLEV), (_NLAT, _NLON, _NLEV))
_Q0 = jnp.full((_NLAT, _NLON, _NLEV), 5e-3)
_U0 = jnp.full((_NLAT, _NLON, _NLEV), 8.0)
_V0 = jnp.zeros((_NLAT, _NLON, _NLEV))
_PS0 = jnp.full((_NLAT, _NLON), 1.0e5)
_SST0 = jnp.full((_NLAT, _NLON), 290.0)


def _synth_run_fn(config):
    """Deterministic synthetic model: ``T = T0 + alpha·C_K`` (per column).

    Reads ``config.C_K`` (scalar baseline OR per-column ``(_N,)`` after injection),
    broadcasts it to the grid, and applies the pinned linear T response.  All other
    fields are C_K-independent, so correcting a column's C_K onto the truth value
    drives that column's T exactly onto the reference."""
    ck = jnp.asarray(config.C_K)
    ck_field = jnp.broadcast_to(ck, (_N,)).reshape(_NLAT, _NLON)
    T = _T0 + _ALPHA_K_PER_CK * ck_field[:, :, None]  # noqa: N806 — temperature
    return ColumnState(T=T, q_v=_Q0, u=_U0, v=_V0, p_s=_PS0, sst_K=_SST0)


class _Eddy:
    """Mock LES diagnosis returning a constant eddy coefficient (= injected C_K)."""

    def __init__(self, k):
        self.K = jnp.array([k, k])
        self.valid = jnp.array([True, True])


@pytest.mark.filterwarnings("error::FutureWarning")
def test_correction_loop_reduces_known_bias_osse():
    """A perfect diagnosis drives the re-run onto the reference ⇒ the loop's own
    ``bias.improved`` flag is True, the global bias strictly drops (to ~0), and the
    targeted worst column improves.  This is the synthetic proof of the
    done-criterion verb 'updating these parameters improve the biases'."""
    sigma = create_sigma_coordinate(_NLEV)
    sigma_full = jnp.asarray(sigma.sigma_full)
    sigma_half = jnp.asarray(sigma.sigma_half)
    grid = create_latlon_grid(_NLAT, _NLON, dtype=jnp.float64)
    rad2deg = 180.0 / np.pi
    lat_deg = jnp.asarray(np.asarray(grid.grid_lat) * rad2deg)
    lon_deg = jnp.asarray(np.asarray(grid.grid_lon) * rad2deg)

    # Reference ("observations"): the default coefficient everywhere EXCEPT the one
    # perturbed worst column, which carries the truth C_K.  Built through the SAME
    # synthetic run_fn so the loop can reproduce it exactly with the right injection.
    ck_ref = jnp.full((_N,), _C_K_DEFAULT).at[_WFLAT].set(_C_K_OBS)
    reference = _synth_run_fn(CLUBBLiteConfig()._replace(C_K=ck_ref))

    # Sanity: only the perturbed column differs from the uniform-default baseline.
    baseline_state = _synth_run_fn(CLUBBLiteConfig())
    dT = np.asarray(  # noqa: N806 — delta-temperature per column
        jnp.abs(reference.T - baseline_state.T)).reshape(_N, _NLEV).max(axis=1)
    assert dT[_WFLAT] == pytest.approx(_ALPHA_K_PER_CK * (_C_K_OBS - _C_K_DEFAULT))
    assert float(np.delete(dT, _WFLAT).max()) == 0.0  # bias is pinned to one column

    compare_fn = make_compare_fn(
        reference=reference, sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=lat_deg, lon_deg=lon_deg, area_weights=jnp.ones((_NLAT, _NLON)),
        n_worst=1, run_amip_fn=_synth_run_fn,
    )

    def diagnose_fn(record, model_ctx):
        return _Eddy(_C_K_OBS)   # the perfect (truth) diagnosis

    result = run_correction_iteration(
        CLUBBLiteConfig(),
        compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        promotion_key="clubb_lite_C_K", grid_shape=(_NLAT, _NLON),
        background=_C_K_DEFAULT,
    )

    # The loop targeted the perturbed column and injected the truth coefficient
    # there (background elsewhere) — so the updated C_K field equals the reference's.
    assert result.n_diagnosed == 1 and result.n_corrected == 1
    ck = np.asarray(result.updated_config.C_K)
    assert ck.shape == (_N,)
    assert ck[_WFLAT] == pytest.approx(_C_K_OBS)
    assert np.allclose(np.delete(ck, _WFLAT), _C_K_DEFAULT)
    np.testing.assert_allclose(ck, np.asarray(ck_ref))

    # THE POINT: updating the parameter LOWERED the bias.  The re-run reproduces the
    # reference (same C_K field ⇒ same deterministic T), so the global bias drops to
    # ~0, the loop's monotonic `improved` flag is True, and the worst column improves.
    assert bool(result.bias.improved) is True
    assert float(result.bias.updated_bias) < float(result.bias.baseline_bias)
    assert float(result.bias.baseline_bias) > 0.0
    assert float(result.bias.updated_bias) == pytest.approx(0.0, abs=1e-9)
    assert float(result.worst_column_change) > 0.0   # >0 ⇒ worst column improved


@pytest.mark.filterwarnings("error::FutureWarning")
def test_correction_loop_osse_is_non_vacuous():
    """Synthetic-violation self-test: a NO-OP diagnosis (returns the background
    coefficient, so the injected column is unchanged) must NOT register an
    improvement.  Proves the sister test's ``improved is True`` is a real signal,
    not vacuously satisfied by the harness — the loop only reports improvement when
    the parameter update genuinely moves the model toward the reference."""
    sigma = create_sigma_coordinate(_NLEV)
    grid = create_latlon_grid(_NLAT, _NLON, dtype=jnp.float64)
    rad2deg = 180.0 / np.pi
    ck_ref = jnp.full((_N,), _C_K_DEFAULT).at[_WFLAT].set(_C_K_OBS)
    reference = _synth_run_fn(CLUBBLiteConfig()._replace(C_K=ck_ref))

    compare_fn = make_compare_fn(
        reference=reference, sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=jnp.asarray(np.asarray(grid.grid_lat) * rad2deg),
        lon_deg=jnp.asarray(np.asarray(grid.grid_lon) * rad2deg),
        area_weights=jnp.ones((_NLAT, _NLON)), n_worst=1, run_amip_fn=_synth_run_fn,
    )

    def noop_diagnose_fn(record, model_ctx):
        return _Eddy(_C_K_DEFAULT)   # the background — no correction at all

    result = run_correction_iteration(
        CLUBBLiteConfig(),
        compare_fn=compare_fn, diagnose_fn=noop_diagnose_fn,
        promotion_key="clubb_lite_C_K", grid_shape=(_NLAT, _NLON),
        background=_C_K_DEFAULT,
    )

    # The injected column kept the background ⇒ the re-run reproduces the biased
    # baseline ⇒ no improvement (strict ``updated < baseline`` is False) and the
    # worst column is unchanged.  The bias is non-zero throughout (a real bias the
    # correct diagnosis WOULD have removed — cf. the sister test).
    assert bool(result.bias.improved) is False
    assert float(result.bias.updated_bias) == pytest.approx(
        float(result.bias.baseline_bias))
    assert float(result.bias.baseline_bias) > 0.0
    assert float(result.worst_column_change) == pytest.approx(0.0, abs=1e-12)
