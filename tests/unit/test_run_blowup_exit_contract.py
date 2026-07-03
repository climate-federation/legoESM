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
