"""Contract tests for the fast complete-real-ERA5-loop validator's CLI.

The validator's ``main()`` is ``# pragma: no cover`` (it builds a coupled model, ingests a
LOCAL ERA5 archive, and runs the whole correction loop — minutes of compute needing data not
present in CI), so the unit-testable surface is the argument-parser CONTRACT + the
``_run_coupled`` config wiring.  ``tests/run/test_cli_imports_resolve.py`` separately guards
that every ``from ... import`` in the module resolves and that it runs as a bare script.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.experiment.check_real_era5_full_loop import _build_arg_parser


def test_required_flags_enforced():
    """The local-archive dir + date are REQUIRED — omitting them is a parse error, not a
    silent default that would later crash deep in the ingest."""
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--local-era5-dir", "/x"])  # date still missing


def test_defaults_are_small_and_fast():
    """The defaults pick a TINY config (the validator is a fast composition check, not the
    HPC science run) — small resolution/levels and a 2-column LES budget."""
    args = _build_arg_parser().parse_args(
        ["--local-era5-dir", "/archive", "--local-era5-date", "20170901"])
    assert args.local_era5_dir == "/archive"
    assert args.local_era5_date == "20170901"
    assert args.resolution == 4
    assert args.nlev == 10
    assert args.era5_n_times == 2
    assert args.n_worst == 2


def test_overrides_are_typed_ints():
    """The numeric knobs round-trip as ints (not strings) so the model config + budgets are
    built from the right types."""
    args = _build_arg_parser().parse_args([
        "--local-era5-dir", "/a", "--local-era5-date", "20200101",
        "--resolution", "8", "--nlev", "16", "--era5-n-times", "4", "--n-worst", "3"])
    assert (args.resolution, args.nlev, args.era5_n_times, args.n_worst) == (8, 16, 4, 3)
    assert all(isinstance(v, int) for v in
               (args.resolution, args.nlev, args.era5_n_times, args.n_worst))


def test_run_coupled_builds_clubb_lite_config():
    """``_run_coupled`` must wire C_K through a clubb_lite turbulence override on a coupled
    (CMIP) latlon config — without running it (the ``.setup()/.run()`` are the heavy part).

    We monkeypatch the driver to capture the config the validator hands it, proving the
    plumbing (grid type, turbulence scheme, C_K) is correct."""
    import scripts.experiment.check_real_era5_full_loop as mod

    captured = {}

    class _FakeDriver:
        def __init__(self, atm, ocean, *, ocean_grid):
            captured["atm"] = atm
            captured["ocean_grid"] = ocean_grid

        def setup(self):
            captured["setup"] = True

        def run(self):
            captured["run"] = True

    import legoesm.driver.coupled_esm_driver as drv_mod

    orig = drv_mod.CoupledESMDriver
    drv_mod.CoupledESMDriver = _FakeDriver
    try:
        mod._run_coupled(0.7, resolution=4, nlev=10)
    finally:
        drv_mod.CoupledESMDriver = orig

    atm = captured["atm"]
    assert captured["ocean_grid"] is None          # same-grid identity coupling
    assert captured.get("setup") and captured.get("run")
    assert atm.grid.grid_type == "latlon"
    assert atm.turbulence == "clubb_lite"
    assert atm.turbulence_override.scheme == "clubb_lite"
    assert atm.turbulence_override.clubb_lite.C_K == pytest.approx(0.7)


def test_run_coupled_accepts_per_column_ck_field():
    """The correction loop injects a per-column ``(n_columns,)`` C_K field for the re-run;
    ``_run_coupled`` must pass it THROUGH to clubb_lite (which broadcasts it), NOT ``float()``
    it (iter 507: ``float(1-D array)`` crashed, so the per-column re-run was never exercised).
    Monkeypatch the driver so this checks the array plumbing without running the model."""
    import jax.numpy as jnp

    import scripts.experiment.check_real_era5_full_loop as mod

    captured = {}

    class _FakeDriver:
        def __init__(self, atm, ocean, *, ocean_grid):
            captured["atm"] = atm

        def setup(self):
            pass

        def run(self):
            pass

    import legoesm.driver.coupled_esm_driver as drv_mod

    field = jnp.asarray([0.2, 0.3, 0.4, 0.5])
    orig = drv_mod.CoupledESMDriver
    drv_mod.CoupledESMDriver = _FakeDriver
    try:
        mod._run_coupled(field, resolution=4, nlev=10)   # MUST NOT raise on the 1-D field
    finally:
        drv_mod.CoupledESMDriver = orig

    injected = captured["atm"].turbulence_override.clubb_lite.C_K
    # the per-column field reaches the config UN-collapsed (not floated to a scalar)
    assert jnp.ndim(injected) == 1
    np.testing.assert_allclose(np.asarray(injected), np.asarray(field))
