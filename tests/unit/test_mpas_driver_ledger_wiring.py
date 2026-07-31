"""Driver wiring of the MPAS budget ledger (#1311) — tripwire tests.

The functional behaviour is pinned at the two layers below the driver:
``test_mpas_physics_ledger`` (per-scheme rows close against the applied
physics) and ``test_mpas_step_ledger`` (full step ledger closes per column).
A full ``_run_mpas`` e2e needs forcing assets and is not a unit test, so the
driver layer gets SOURCE tripwires — each names the symbol that RUNS
(``ModelDriver._run_mpas``, the method the production chain calls) and each
fails if its forward/read/write is deleted (verified by construction: the
asserted strings exist only in the added wiring).
"""
import inspect

import pytest


@pytest.fixture(scope="module")
def run_mpas_src():
    from legoesm.driver.model_driver import ModelDriver

    return inspect.getsource(ModelDriver._run_mpas)


def test_physics_factory_receives_the_ledger_flag(run_mpas_src):
    """Both make_physics calls (full + held-radiation norad variant) must
    forward budget_ledger — a missing forward on either variant silently
    zeroes the ledger on those steps (the #1385 silent-drop defect class)."""
    assert run_mpas_src.count("budget_ledger=_budget_ledger_on") >= 2, (
        "make_physics no longer receives budget_ledger on both the full and "
        "norad variants — the flag would be silently inert (#1311/#1385 "
        "defect class)")


def test_step_ledger_side_channel_is_read(run_mpas_src):
    assert '_step_ledger' in run_mpas_src, (
        "_run_mpas no longer reads model._step_ledger — accumulation is dead")


def test_percolumn_file_is_written_with_its_own_name(run_mpas_src):
    """The MPAS per-column schema must never share the FV global-row
    filename (budget_ledger.npz) — a consumer reading one schema must not
    silently get the other."""
    assert "budget_ledger_columns.npz" in run_mpas_src
    assert 'np.savez' in run_mpas_src


def test_mpi_refusal_is_narrowed_not_deleted(run_mpas_src):
    """Serial-only for now: the multi-rank guard must still be present (the
    per-rank gather is unwired), but must be conditioned on world size —
    not the old unconditional refusal."""
    assert "serial-only" in run_mpas_src
    assert '_mpi_world_size", 1) or 1) > 1' in run_mpas_src, (
        "the MPI guard is no longer keyed on world size — either the serial "
        "path refuses again (regression to the pre-port refusal) or the "
        "MPI path silently produces a rank-local ledger")
    # -np 1 cell-partition MPI takes _mpi_step, which bypasses model.step's
    # eager side-channel — without this clause the flag is SILENTLY inert
    # there (the exact #1311 failure mode).
    assert '_voronoi_layout", None) is not None' in run_mpas_src
