"""Blowup must not masquerade as a completed run (chain-harness masking bugs).

Two independent defects let a numerically-blown-up AMIP run report success and
poison a SLURM ``afterok`` chain (observed on the lat-lon 3-yr chain: it blew up
at day 515 but wrote ``checkpoint_day_1095.npz`` and exited 0, so the next link
restarted "already done"):

1. ``run_amip.py`` DISCARDED ``driver.run()``'s status string.  A BLOWUP leaves
   the state finite-but-unphysical (e.g. T in [153, 400] K), so the NaN/Inf
   sweep passed and the run exited 0.  Fix: honour the status via
   ``status_to_exit_code``.

2. ``ModelDriver._finalize_run`` wrote the final checkpoint labelled with the
   TARGET day regardless of ``run_status``.  Fix: gate that write on
   ``run_status == "COMPLETED"`` (matching the spectral / MPAS paths).

These tests lock both contracts.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

from legoesm.driver.run_status import status_to_exit_code
from legoesm.driver.model_driver import ModelDriver


# --- bug 1: the status -> exit-code contract run_amip now honours -----------

def test_completed_status_is_exit_zero():
    assert status_to_exit_code("COMPLETED") == 0


@pytest.mark.parametrize(
    "status",
    [
        "BLOWUP at day 515: temperature out of physical bounds (min=153.3K, max=400.2K)",
        "BLOWUP at day 5: non-finite winds",
        "BLOWUP at day 5: max wind 999.9 m/s",
        "",              # unexpected/empty — defensive
        "weird",         # unexpected — defensive
    ],
)
def test_non_completed_status_is_exit_one(status):
    """Anything that is not exactly COMPLETED must fail the run (exit 1) so an
    ``afterok`` chain stops instead of restarting from garbage."""
    assert status_to_exit_code(status) == 1


# --- bug 2: _finalize_run only checkpoints a CLEAN run ----------------------

def _make_fake_driver(tmp_path):
    """Minimal stand-in exposing exactly the attributes ``_finalize_run``
    touches, so we can drive it without building a full model."""
    fake = MagicMock()
    fake.state.u.data = jnp.zeros(4)          # real array for block_until_ready
    fake._mpi_rank = None                     # root
    fake._output_dir = tmp_path
    fake.diagnostics.print_summary.return_value = "summary"
    # A bare MagicMock would auto-create a TRUTHY ``_checkpoint_callback``
    # attribute, silently rerouting every checkpoint assertion below.  A
    # standalone (uncoupled) run has it set to None by ``run()``.
    fake._checkpoint_callback = None
    return fake


def test_blowup_writes_no_final_checkpoint(tmp_path):
    """A BLOWUP status must NOT write the target-day final checkpoint — the
    garbage-state-labelled-as-day-1095 masking bug."""
    fake = _make_fake_driver(tmp_path)
    status = ModelDriver._finalize_run(
        fake,
        run_status="BLOWUP at day 515: temperature out of physical bounds",
        t_jit=0.0, t_start=0.0, n_steps_total=630720,
        START_DAY=730.0, N_DAYS=365.0, checkpoint_interval=17280,
    )
    fake.save_checkpoint.assert_not_called()
    assert status.startswith("BLOWUP")            # status propagated intact
    fake._restore_halo_backend.assert_called_once()  # lifecycle still runs


def test_completed_writes_the_final_checkpoint(tmp_path):
    """A CLEAN run still writes its final checkpoint at the target day, so the
    fix does not regress normal chaining."""
    fake = _make_fake_driver(tmp_path)
    ModelDriver._finalize_run(
        fake,
        run_status="COMPLETED",
        t_jit=0.0, t_start=0.0, n_steps_total=17280,
        START_DAY=0.0, N_DAYS=10.0, checkpoint_interval=17280,
    )
    fake.save_checkpoint.assert_called_once_with(17280, 10.0)  # START_DAY + N_DAYS


def test_disabled_checkpointing_writes_nothing_even_when_completed(tmp_path):
    """checkpoint_interval == 0 disables checkpoints regardless of status."""
    fake = _make_fake_driver(tmp_path)
    ModelDriver._finalize_run(
        fake, run_status="COMPLETED",
        t_jit=0.0, t_start=0.0, n_steps_total=100,
        START_DAY=0.0, N_DAYS=1.0, checkpoint_interval=0,
    )
    fake.save_checkpoint.assert_not_called()


def test_coupled_final_checkpoint_routes_through_callback(tmp_path):
    """A COUPLED run's final checkpoint must go through the coupled
    ``checkpoint_callback`` (full atm + ocean + surface state), mirroring the
    periodic path — ``self.save_checkpoint`` alone would write an
    atmosphere-only file that ``run_coupled --resume`` picks up while the
    coupled ``coupled_day_*.npz`` is missing (stale-ocean resume)."""
    fake = _make_fake_driver(tmp_path)
    coupled_ckpt = MagicMock()
    fake._checkpoint_callback = coupled_ckpt
    ModelDriver._finalize_run(
        fake, run_status="COMPLETED",
        t_jit=0.0, t_start=0.0, n_steps_total=17280,
        START_DAY=0.0, N_DAYS=10.0, checkpoint_interval=17280,
    )
    coupled_ckpt.assert_called_once_with(17280, 10.0)
    fake.save_checkpoint.assert_not_called()


def test_coupled_blowup_writes_no_final_checkpoint_either(tmp_path):
    """The COMPLETED gate applies to the coupled callback path too."""
    fake = _make_fake_driver(tmp_path)
    coupled_ckpt = MagicMock()
    fake._checkpoint_callback = coupled_ckpt
    ModelDriver._finalize_run(
        fake, run_status="BLOWUP at day 515: temperature out of physical bounds",
        t_jit=0.0, t_start=0.0, n_steps_total=630720,
        START_DAY=730.0, N_DAYS=365.0, checkpoint_interval=17280,
    )
    coupled_ckpt.assert_not_called()
    fake.save_checkpoint.assert_not_called()


# --- run_amip post-run verdict: both signals must be clean ------------------

def test_resolve_run_exit_contract():
    """``run_amip`` exits 0 only when the driver status is COMPLETED AND the
    final-state NaN/Inf sweep is clean — either signal alone fails the run
    (locks the helper ``main()`` and the profile path route through)."""
    from scripts.run.run_amip import _resolve_run_exit

    assert _resolve_run_exit("COMPLETED", True) == 0
    assert _resolve_run_exit("COMPLETED", False) == 1          # finite sweep trips
    assert _resolve_run_exit("BLOWUP at day 515", True) == 1   # status trips
    assert _resolve_run_exit("BLOWUP at day 515", False) == 1
    assert _resolve_run_exit("", True) == 1                    # defensive


def test_sync_finite_verdict_is_identity_without_mpi():
    """Single-rank / SPMD drivers pass the verdict through untouched (no
    mpi4py import, no reduction)."""
    from scripts.run.run_amip import _sync_finite_verdict_across_ranks

    fake = MagicMock()
    fake._mpi_rank = None
    fake._device_config = None
    assert _sync_finite_verdict_across_ranks(fake, True, None) == (True, None)
    assert _sync_finite_verdict_across_ranks(fake, False, "T") == (False, "T")
    # SPMD multi-controller: rank set, but device config is NOT mpi4jax-style
    # distributed — verdict already process-identical, passes through.
    fake._mpi_rank = 1
    fake._device_config = MagicMock(is_distributed=False)
    assert _sync_finite_verdict_across_ranks(fake, False, "u") == (False, "u")
