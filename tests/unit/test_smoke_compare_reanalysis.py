"""Unit test for ``scripts/experiment/smoke_compare_reanalysis.py`` — the operator
end-to-end SYNTHETIC preflight (config -> synthetic ERA5 -> campaign --dry-run).

Confirms the turnkey-chain smoke actually drives the three real entry points and
returns success (exit 0) on the synthetic path, and writes the intermediate
artifacts — so an operator can trust a green smoke before launching the multi-day
real-ERA5 run.
"""

from __future__ import annotations

from pathlib import Path


def test_run_smoke_validates_the_turnkey_chain(tmp_path):
    from scripts.experiment.smoke_compare_reanalysis import run_smoke

    rc = run_smoke(str(tmp_path), resolution=8, nlev=5, era5_nlat=8, era5_nlon=16)
    assert rc == 0                                          # the whole preamble validated
    # The two generated inputs the campaign dry-run consumed actually exist.
    assert (tmp_path / "amip_clubb_lite.json").is_file()
    assert (tmp_path / "synthetic_era5.zarr").exists()


def test_main_with_explicit_workdir_returns_zero(tmp_path, capsys):
    from scripts.experiment.smoke_compare_reanalysis import main

    rc = main(["--workdir", str(tmp_path), "--resolution", "8", "--nlev", "5"])
    assert rc == 0
    assert "PASS" in capsys.readouterr().out                # the green operator message


def test_run_smoke_uses_operator_provided_era5_store(tmp_path):
    """``era5_zarr=<path>`` validates the operator's OWN ERA5 ingest: the provided
    store is used directly (NOT regenerated) and the dry-run loads/regrids it.

    A synthetic store stands in for a 'real' one — the point is the operator-provided
    PATH branch (so a store-specific ingest problem surfaces in the preflight, not the
    multi-day job)."""
    from scripts.data.make_synthetic_era5 import main as era5_main
    from scripts.experiment.smoke_compare_reanalysis import run_smoke

    provided = tmp_path / "operator_era5.zarr"
    assert era5_main([str(provided), "--nlat", "10", "--nlon", "20", "--ntime", "1"]) == 0

    work = tmp_path / "work"
    rc = run_smoke(str(work), era5_zarr=str(provided), resolution=8, nlev=5)
    assert rc == 0
    # The provided store was used directly — the smoke did NOT regenerate synthetic.
    assert not (work / "synthetic_era5.zarr").exists()
    assert provided.exists()


def test_run_smoke_raises_loudly_if_config_generation_fails(tmp_path, monkeypatch):
    """A failure in an EARLY stage (config gen) must abort with a clear error rather
    than silently dry-running against a missing/garbage config."""
    import pytest

    import scripts.experiment.smoke_compare_reanalysis as mod

    # Force the config generator to report failure (non-zero exit).
    monkeypatch.setattr(
        "scripts.experiment.write_amip_clubb_lite_config.main", lambda argv: 1)
    with pytest.raises(RuntimeError, match="config generation failed"):
        run = mod.run_smoke
        run(str(Path(tmp_path)))
