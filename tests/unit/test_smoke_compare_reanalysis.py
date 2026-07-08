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


def test_run_smoke_validates_realistic_ocean_only_surface_flux_flags(tmp_path):
    """REGRESSION (iter 484): forwarding the realistic --ocean-only/--surface-flux flags to the
    dry-run must validate end-to-end on a STRUCTURED (lat-lon) grid. --ocean-only built a FLAT
    (n_columns,) valid_mask that assert_per_column_fields_match_grid REJECTED as not
    broadcastable to the (nlat,nlon) grid — silently broken for EVERY structured grid (only
    flat MPAS worked) until this full-harness smoke surfaced it. (On a flat synthetic config
    --ocean-only is a safe no-op the dry-run still validates the wiring of.)"""
    from scripts.experiment.smoke_compare_reanalysis import run_smoke

    rc = run_smoke(str(tmp_path), resolution=8, nlev=5, era5_nlat=8, era5_nlon=16,
                   ocean_only=True, surface_flux=True)
    assert rc == 0


def test_run_smoke_ocean_only_with_a_real_land_mask_excludes_land(tmp_path, capsys):
    """REGRESSION (iter 486/490): the COMPLETE realistic AMIP combo — a real (synthetic) land
    mask + --ocean-only + --surface-flux TOGETHER (iter 490). --ocean-only must rank a STRICT
    SUBSET of columns (land EXCLUDED), validating the full harness on a NON-trivial mask (not
    the flat-config all-ocean no-op), while --surface-flux (the SST-driven LES BC, valid over
    the ocean columns ocean-only keeps) rides along — the exact flag combo the recommended
    realistic run uses, all three at once."""
    from scripts.data.make_synthetic_land_mask import main as mask_main
    from scripts.experiment.smoke_compare_reanalysis import run_smoke

    mask = tmp_path / "land.nc"
    assert mask_main([str(mask)]) == 0
    rc = run_smoke(str(tmp_path), resolution=8, nlev=5, era5_nlat=8, era5_nlon=16,
                   ocean_only=True, surface_flux=True, land_mask_path=str(mask))
    assert rc == 0
    out = capsys.readouterr().out
    import re
    m = re.search(r"ranking the (\d+) ocean columns .* of (\d+) total", out)
    assert m is not None, out
    n_ocean, n_total = int(m.group(1)), int(m.group(2))
    assert 0 < n_ocean < n_total                          # land columns were EXCLUDED


def test_run_smoke_ocean_only_on_a_cubed_sphere_grid(tmp_path):
    """REGRESSION (iter 485): --ocean-only must validate end-to-end on a CUBED-SPHERE grid
    too. The iter-484 flat-mask bug was grid-shape-specific — the fix reshapes the mask to the
    grid shape, which for cubed-sphere is the 3-D (6,n,n), a DIFFERENT shape than lat-lon's
    (nlat,nlon). This smokes the full harness (ERA5 regrid -> compare -> rank -> ocean mask)
    on (6,4,4), confirming the fix generalizes beyond lat-lon."""
    from scripts.experiment.smoke_compare_reanalysis import run_smoke

    rc = run_smoke(str(tmp_path), grid_type="cubed_sphere", resolution=4, nlev=5,
                   era5_nlat=8, era5_nlon=16, ocean_only=True, surface_flux=True)
    assert rc == 0


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


def test_run_smoke_cmip_mode_validates_the_coupled_driver(tmp_path):
    """The CMIP path must also preflight — both done-criterion modes (AMIP + CMIP),
    not just the AMIP default.

    ``mode='cmip'`` threads ``--mode cmip`` to the campaign, which builds a
    ``CoupledESMDriver`` (interactive ocean) via the ``--coupled-preset aquaplanet``
    default during the dry-run construction.  A regression in the coupled-driver
    preflight (or a removed coupled-preset default) would otherwise slip past the
    AMIP-only tests.
    """
    from scripts.experiment.smoke_compare_reanalysis import run_smoke

    rc = run_smoke(str(tmp_path), mode="cmip", resolution=8, nlev=5,
                   era5_nlat=8, era5_nlon=16)
    assert rc == 0                                          # coupled-driver dry-run validated


def test_run_smoke_propagates_a_dry_run_failure(tmp_path, monkeypatch, capsys):
    """A campaign dry-run FAILURE (bad config/paths/scheme) must PROPAGATE: run_smoke
    returns the non-zero code and ``main`` prints FAIL — never a false PASS that would
    let the operator launch a broken multi-day job."""
    from scripts.experiment.smoke_compare_reanalysis import main, run_smoke

    # The config + ERA5 generation succeed; force only the campaign dry-run to fail.
    monkeypatch.setattr("scripts.run.run_correction_campaign.main", lambda argv: 3)

    rc = run_smoke(str(tmp_path / "w1"))
    assert rc == 3                                          # the dry-run failure propagated
    rc2 = main(["--workdir", str(tmp_path / "w2")])
    assert rc2 == 3
    assert "FAIL" in capsys.readouterr().out               # FAIL, not a false PASS


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
