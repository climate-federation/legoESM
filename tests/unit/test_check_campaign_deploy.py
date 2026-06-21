"""Unit test for ``scripts/experiment/check_campaign_deploy.py`` — the deploy-check
preflight (a campaign output + base config -> a grid-verified, validate_strict-sound
deployed config, reported without running the model).
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest


def _base_config(tmp_path, resolution=8, nlev=5):
    from scripts.experiment.write_amip_clubb_lite_config import main as cfg_main

    cfg = tmp_path / "base.json"
    assert cfg_main([str(cfg), "--resolution", str(resolution), "--nlev", str(nlev)]) == 0
    return str(cfg)


def _campaign_output(tmp_path, base_cfg, grid, *, name="out.json", field="C_K",
                     lo=0.3, hi=0.6, grid_for_provenance=None, averaging=None, health=None):
    from scripts.run.run_correction_campaign import _grid_provenance

    ncol = int(np.prod(grid.grid_shape_2d))
    out = tmp_path / name
    payload = {field: list(np.linspace(lo, hi, ncol)),
               "grid": _grid_provenance(base_cfg, grid_for_provenance or grid),
               "biases": [1.0, 0.5], "accepted": [True]}
    if averaging is not None:
        payload["averaging"] = averaging
    if health is not None:
        payload["health"] = health
    with open(out, "w") as f:
        json.dump(payload, f)
    return str(out)


def test_check_deploy_surfaces_the_campaign_health_verdict(tmp_path, capsys):
    """The deploy-check surfaces the campaign HEALTH verdict (iter 312): the output JSON is
    written even on a non-zero campaign exit (only the exit status gates), so a deployer
    reading a saved output — without having seen the campaign's exit code — must know whether
    it actually IMPROVED. A non-'improved' verdict ('stalled') prints a WARNING; 'improved'
    shows the status with no warning; absent health → no health line (pre-312 back-compat)."""
    from scripts.experiment.check_campaign_deploy import check_deploy, main
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)

    stalled = _campaign_output(
        tmp_path, base_cfg, grid, name="stalled.json",
        health={"status": "stalled", "message": "Bias reduced only 0.1%."})
    assert check_deploy(cfg, stalled)["health"]["status"] == "stalled"
    main(["--base-config", cfg, "--campaign-output", stalled])
    msg = capsys.readouterr().out
    assert "campaign health: stalled" in msg and "NOT" in msg and "improved" in msg

    good = _campaign_output(
        tmp_path, base_cfg, grid, name="good.json",
        health={"status": "improved", "message": "Bias reduced 12%."})
    main(["--base-config", cfg, "--campaign-output", good])
    out = capsys.readouterr().out
    assert "campaign health: improved" in out and "WARNING" not in out

    none_out = _campaign_output(tmp_path, base_cfg, grid, name="nohealth.json")
    assert check_deploy(cfg, none_out)["health"] is None
    main(["--base-config", cfg, "--campaign-output", none_out])
    assert "campaign health" not in capsys.readouterr().out


def test_check_deploy_validates_and_reports_stats(tmp_path):
    from scripts.experiment.check_campaign_deploy import check_deploy
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    out = _campaign_output(tmp_path, base_cfg, grid, lo=0.3, hi=0.6)

    stats = check_deploy(cfg, out)
    assert stats["n_columns"] == 128                     # 8 x 16
    assert stats["grid_shape"] == (8, 16)
    assert set(stats["corrected"]) == {"C_K"}            # only C_K was corrected
    assert stats["corrected"]["C_K"]["min"] == pytest.approx(0.3)
    assert stats["corrected"]["C_K"]["max"] == pytest.approx(0.6)


def test_check_deploy_flags_an_out_of_bounds_coefficient(tmp_path, capsys):
    """A deployed coefficient OUTSIDE its calibratable (param-spec) range is FLAGGED (iter
    309): the campaign clips to bounds by default, so an out-of-range deploy means
    --allow-unphysical-coeff was used OR the output is corrupted, and the production physics
    uses the value RAW (no deploy-time clip) — so check_deploy reports in_bounds=False and
    main() prints a WARNING. Non-vacuous: an in-bounds deploy reports in_bounds=True + no
    warning (C_K calibratable range is (0.1, 1.2))."""
    from scripts.experiment.check_campaign_deploy import check_deploy, main
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)

    # An absurd C_K (≈100) is far outside the (0.1, 1.2) calibratable range.
    bad = _campaign_output(tmp_path, base_cfg, grid, name="oob.json", lo=100.0, hi=100.0)
    ck = check_deploy(cfg, bad)["corrected"]["C_K"]
    assert ck.get("bounds") is not None and ck["in_bounds"] is False
    main(["--base-config", cfg, "--campaign-output", bad])
    msg = capsys.readouterr().out
    assert "OUTSIDE the calibratable range" in msg and "C_K" in msg

    # An in-bounds deploy: in_bounds True + no out-of-range warning (the contrast).
    ok = _campaign_output(tmp_path, base_cfg, grid, name="ok.json", lo=0.3, hi=0.6)
    assert check_deploy(cfg, ok)["corrected"]["C_K"]["in_bounds"] is True
    main(["--base-config", cfg, "--campaign-output", ok])
    assert "OUTSIDE" not in capsys.readouterr().out


def test_check_deploy_surfaces_averaging_provenance_for_the_deployer(tmp_path, capsys):
    """The deploy-check surfaces the SOURCE climate (iter 277) so the deployer — possibly
    a different person weeks later — confirms a snapshot-trained vs climatology-trained
    correction before deploying. Absent → no source line (back-compat); snapshot → a
    confirm-intent note."""
    from scripts.experiment.check_campaign_deploy import check_deploy, main
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)

    # absent averaging (a pre-267 output) → stats key is None, no source line printed.
    out0 = _campaign_output(tmp_path, base_cfg, grid, name="no_av.json")
    assert check_deploy(cfg, out0)["averaging"] is None
    main(["--base-config", cfg, "--campaign-output", out0])
    assert "source:" not in capsys.readouterr().out

    # climatology → the source line names it; snapshot → adds the confirm-intent note.
    clim = _campaign_output(tmp_path, base_cfg, grid, name="clim.json",
                            averaging={"era5_n_times": 30, "era5_time_idx": 12})
    assert check_deploy(cfg, clim)["averaging"] == {"era5_n_times": 30, "era5_time_idx": 12}
    main(["--base-config", cfg, "--campaign-output", clim])
    o = capsys.readouterr().out
    assert "30-time ERA5 climatology @ idx 12" in o and "confirm this matches" not in o

    snap = _campaign_output(tmp_path, base_cfg, grid, name="snap.json",
                            averaging={"era5_n_times": 1, "era5_time_idx": 5})
    main(["--base-config", cfg, "--campaign-output", snap])
    o = capsys.readouterr().out
    assert "SINGLE ERA5 snapshot @ idx 5" in o and "confirm this matches your deploy" in o

    # the MODEL-side averaging window (iter 313) is surfaced when present, so the deployer
    # confirms BOTH windows are comparable (the iter-267 alignment is two-sided).
    with_model = _campaign_output(
        tmp_path, base_cfg, grid, name="model_av.json",
        averaging={"era5_n_times": 30, "era5_time_idx": 0,
                   "model_days": 200, "model_diag_days": 5, "model_n_samples": 40})
    main(["--base-config", cfg, "--campaign-output", with_model])
    o = capsys.readouterr().out
    assert "model climatology window: 200 days" in o and "40 samples" in o


def test_main_returns_zero_and_prints_ok(tmp_path, capsys):
    from scripts.experiment.check_campaign_deploy import main
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    out = _campaign_output(tmp_path, base_cfg, grid)
    rc = main(["--base-config", cfg, "--campaign-output", out])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "OK" in captured and "C_K" in captured


def test_build_deployed_config_carries_the_override_and_validates(tmp_path):
    """``build_deployed_config`` returns a runnable deployed config: the base config
    with ``turbulence_override`` set to the per-column correction (the canonical way
    to USE a campaign output, since the override does not serialize)."""
    from scripts.experiment.check_campaign_deploy import build_deployed_config
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    out = _campaign_output(tmp_path, base_cfg, grid, lo=0.31, hi=0.59)

    deployed, dgrid = build_deployed_config(cfg, out)
    # The deployed config carries the per-column override (a real GCM driver would
    # then inject it) — base had a scalar default; deployed has the (128,) field.
    assert deployed.turbulence_override is not None
    ck = np.asarray(deployed.turbulence_override.clubb_lite.C_K)
    assert ck.shape == (128,)
    assert ck.min() == pytest.approx(0.31) and ck.max() == pytest.approx(0.59)
    assert tuple(dgrid.grid_shape_2d) == (8, 16)


def test_build_env_kernel_deployed_override_cross_grid(tmp_path):
    """The CROSS-GRID deploy (iter 475): a campaign's env-kernel + a TARGET base config + the
    target model state → a deployed config whose turbulence_override carries the per-target-column
    C_K (via environmental similarity) + a coverage diagnostic. The grid-agnostic analog of
    build_deployed_config (which is same-grid only)."""
    from typing import NamedTuple

    from legoesm.driver.config import config_to_dict
    from legoesm.training.column_manifest import ColumnEnvironment
    from legoesm.training.compare_reanalysis import ColumnState
    from legoesm.training.deploy_correction import build_env_kernel, env_kernel_to_dict

    from scripts.experiment.check_campaign_deploy import (
        build_env_kernel_deployed_override,
    )
    from scripts.experiment.write_amip_clubb_lite_config import (
        build_amip_clubb_lite_config,
    )

    class _Diag(NamedTuple):
        C_K: object
        valid: object

    class _Rec(NamedTuple):
        environment: ColumnEnvironment

    # a kernel over an SST/CAPE/shear range
    envs = [(298.0, 100.0, 5.0), (300.0, 1500.0, 12.0), (302.0, 2800.0, 20.0)]
    recs = [_Rec(ColumnEnvironment(*e)) for e in envs]
    diags = [_Diag(C_K=jnp.array([c, c]), valid=jnp.array([True, True]))
             for c in (0.30, 0.45, 0.62)]
    kernel = build_env_kernel(recs, diags, "clubb_coefficient",
                              length_scales=[2.0, 1200.0, 8.0], field="C_K")
    kpath = tmp_path / "out.json.env_kernel.json"
    kpath.write_text(json.dumps(env_kernel_to_dict(kernel)))

    # a TARGET base config (latlon 8 -> 8x16) + a synthetic target model state
    cpath = tmp_path / "target.json"
    cpath.write_text(json.dumps(config_to_dict(
        build_amip_clubb_lite_config(resolution=8, nlev=5))))
    nlat, nlon, nlev = 8, 16, 5
    state = ColumnState(
        T=jnp.full((nlat, nlon, nlev), 290.0), q_v=jnp.full((nlat, nlon, nlev), 8e-3),
        u=jnp.full((nlat, nlon, nlev), 10.0), v=jnp.zeros((nlat, nlon, nlev)),
        p_s=jnp.full((nlat, nlon), 1.0e5), sst_K=jnp.full((nlat, nlon), 300.0))

    deployed, grid, coverage = build_env_kernel_deployed_override(
        str(kpath), str(cpath), state)
    assert deployed.turbulence_override is not None
    ck = np.asarray(deployed.turbulence_override.clubb_lite.C_K)
    assert ck.shape == (nlat * nlon,) and bool(np.all(np.isfinite(ck)))   # per target column
    assert {"fraction_covered", "fraction_in_hull", "n_columns"} <= set(coverage)
    assert tuple(grid.grid_shape_2d) == (8, 16)
    assert coverage["sst_from_model"] is True            # the target state carried a real SST

    # A target state with NO SST: the env-kernel's DOMINANT predictor (SST) would be
    # fabricated from air temperature, so require_sst fails loud (a production deploy must not
    # silently use an approximate-SST environment); the default (warn) proceeds but FLAGS it.
    state_no_sst = state._replace(sst_K=None)
    with pytest.raises(ValueError, match="require_sst=True"):
        build_env_kernel_deployed_override(str(kpath), str(cpath), state_no_sst,
                                           require_sst=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # column_environment_grid SST warning
        _, _, cov2 = build_env_kernel_deployed_override(str(kpath), str(cpath), state_no_sst)
    assert cov2["sst_from_model"] is False


def test_check_deploy_surfaces_the_cross_grid_env_kernel_sidecar(tmp_path, capsys):
    """check_deploy reports a `<output>.env_kernel.json` sidecar (iter 474) so a deployer learns
    the cross-grid env-kernel option exists; absent → None; a JSON sharing the name but NOT an
    env-kernel (wrong artifact tag) is ignored."""
    from scripts.experiment.check_campaign_deploy import check_deploy, main
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    out = _campaign_output(tmp_path, base_cfg, grid, name="env.json")

    assert check_deploy(cfg, out)["env_kernel"] is None             # no sidecar → None

    kernel_path = out + ".env_kernel.json"
    with open(kernel_path, "w") as f:                               # a real env-kernel artifact
        json.dump({"artifact": "raw_environment_kernel", "field": "clubb_lite_C_K",
                   "predictor_names": ["sst_K", "cape_J_kg", "bulk_shear_m_s"],
                   "sample_env": [[298.0, 100.0, 5.0], [300.0, 1500.0, 12.0],
                                  [302.0, 2800.0, 20.0]],
                   "sample_values": [0.4, 0.5, 0.45], "valid": [True, True, True],
                   "length_scales": [2.0, 1200.0, 8.0], "background": 0.4,
                   "env_lo": [298.0, 100.0, 5.0], "env_hi": [302.0, 2800.0, 20.0]}, f)
    ek = check_deploy(cfg, out)["env_kernel"]
    assert ek is not None and ek["field"] == "clubb_lite_C_K" and ek["n_samples"] == 3
    # the training-env hull is surfaced (iter 476) so the operator can judge cross-grid transfer
    assert ek["hull"]["sst_K"] == [298.0, 302.0]
    assert ek["hull"]["cape_J_kg"] == [100.0, 2800.0]
    main(["--base-config", cfg, "--campaign-output", out])
    msg = capsys.readouterr().out
    assert "cross-grid" in msg and "trained-env hull" in msg and "sst_K" in msg

    with open(kernel_path, "w") as f:                               # wrong artifact → ignored
        json.dump({"artifact": "something_else", "field": "x"}, f)
    assert check_deploy(cfg, out)["env_kernel"] is None


def test_build_deployed_config_preserves_runtime_injections_from_effective_config(tmp_path):
    """The iter-464 effective-config sidecar is DEPLOYABLE end-to-end: build_deployed_config on
    an effective config (carrying the runtime injections dataset=custom + forcing_path +
    insolation_start_doy) applies the C_K override AND preserves those fields — so a same-window
    deploy reproduces the SST boundary + insolation the C_K was calibrated on (iter 468)."""
    from legoesm.driver.config import config_to_dict

    from scripts.experiment.check_campaign_deploy import build_deployed_config
    from scripts.experiment.write_amip_clubb_lite_config import (
        build_amip_clubb_lite_config,
    )
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    # an EFFECTIVE config = base clubb_lite + the runtime injections (iter 449/450 + the forcing)
    eff = build_amip_clubb_lite_config(resolution=8, nlev=5)._replace(
        dataset="custom", forcing_path="/archive/era5_amip.nc", insolation_start_doy=244.0)
    eff_path = tmp_path / "corrected.effective_config.json"
    with open(eff_path, "w") as f:
        json.dump(config_to_dict(eff), f)

    base_cfg, grid, _ = load_base_config_and_grid(str(eff_path))
    out = _campaign_output(tmp_path, base_cfg, grid, lo=0.31, hi=0.59)
    deployed, _dgrid = build_deployed_config(str(eff_path), out)

    assert deployed.turbulence_override is not None         # the C_K correction applied
    assert deployed.dataset == "custom"                     # --amip-forcing injection preserved
    assert deployed.forcing_path == "/archive/era5_amip.nc"
    assert deployed.insolation_start_doy == 244.0           # --align-insolation injection preserved


def test_allow_unverified_grid_is_the_escape_hatch_but_keeps_the_length_check(tmp_path):
    """``allow_unverified_grid`` is the escape hatch for a campaign output that PREDATES
    grid provenance (no 'grid' block): refused by default (no provenance to verify),
    deployed on the array-LENGTH check alone when the flag is set — but a WRONG-LENGTH
    field still fails even then."""
    from scripts.experiment.check_campaign_deploy import check_deploy

    cfg = _base_config(tmp_path)                         # base 8x16 = 128 cols
    nogrid = tmp_path / "nogrid.json"                    # NO grid-provenance block
    nogrid.write_text(json.dumps({"C_K": list(np.linspace(0.3, 0.6, 128))}))

    with pytest.raises((ValueError, SystemExit)):        # default: provenance required
        check_deploy(cfg, str(nogrid))
    stats = check_deploy(cfg, str(nogrid), allow_unverified_grid=True)   # escape hatch
    assert stats["n_columns"] == 128

    bad_len = tmp_path / "badlen.json"
    bad_len.write_text(json.dumps({"C_K": list(np.linspace(0.3, 0.6, 64))}))  # 64 != 128
    with pytest.raises((ValueError, SystemExit)):        # length check survives the bypass
        check_deploy(cfg, str(bad_len), allow_unverified_grid=True)


def test_deployed_config_drives_a_real_driver_with_the_per_column_correction(tmp_path):
    """The runbook's step-5 deploy snippet must COMPOSE end-to-end: build_deployed_config
    → make_base_driver_builder → a REAL ModelDriver whose turbulence config carries the
    per-column LES-corrected C_K.

    This is what makes the operator's deployed run actually USE the correction (not the
    scalar default).  The AMIP-path analog of the iter-37 CMIP capstone, but through the
    OPERATOR's documented ``build_deployed_config`` + ``make_base_driver_builder``
    composition — no existing test covers that exact path (iter 205-207 wired
    ``corrected_turbulence_override`` → ``build_physics_pipeline`` directly).
    """
    from scripts.experiment.check_campaign_deploy import build_deployed_config
    from scripts.run.run_correction_campaign import (
        load_base_config_and_grid,
        make_base_driver_builder,
    )

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    ck_lo, ck_hi = 0.31, 0.59
    out = _campaign_output(tmp_path, base_cfg, grid, lo=ck_lo, hi=ck_hi)

    deployed, _ = build_deployed_config(cfg, out)
    build_driver, _ = make_base_driver_builder("amip")
    driver = build_driver(deployed)                      # constructs + setup()s the real driver
    drv_ck = np.asarray(driver.physics.turbulence_config.C_K)
    # The per-column correction reached the RUNNING driver's turbulence kernel config.
    assert drv_ck.shape == (128,)
    assert drv_ck.min() == pytest.approx(ck_lo) and drv_ck.max() == pytest.approx(ck_hi)


def test_cmip_deployed_config_drives_the_coupled_driver_with_the_correction(tmp_path):
    """The CMIP deploy path through the operator's documented composition: the runbook's
    ``make_base_driver_builder("cmip", coupled_preset=PRESETS[...]())`` must build a
    CoupledESMDriver carrying the per-column LES-corrected C_K.

    Both done-criterion modes' deploy paths are thus locked (AMIP in the sibling test).
    NOTE the preset must be a RESOLVED OBJECT (``PRESETS["aquaplanet"]()``), not the name
    string — the runbook calls this out, and this test guards it.
    """
    from legoesm.driver.coupled_config import PRESETS

    from scripts.experiment.check_campaign_deploy import build_deployed_config
    from scripts.run.run_correction_campaign import (
        load_base_config_and_grid,
        make_base_driver_builder,
    )

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    ck_lo, ck_hi = 0.32, 0.58
    out = _campaign_output(tmp_path, base_cfg, grid, lo=ck_lo, hi=ck_hi)

    deployed, _ = build_deployed_config(cfg, out)
    build_driver, _ = make_base_driver_builder(
        "cmip", coupled_preset=PRESETS["aquaplanet"]())
    driver = build_driver(deployed)                      # constructs + setup()s the coupled driver
    drv_ck = np.asarray(driver._atm.physics.turbulence_config.C_K)  # coupled-driver attribute
    assert drv_ck.shape == (128,)
    assert drv_ck.min() == pytest.approx(ck_lo) and drv_ck.max() == pytest.approx(ck_hi)


def test_check_deploy_refuses_a_different_grid_of_the_same_column_count(tmp_path):
    """The grid fingerprint must refuse a correction learned on a DIFFERENT grid even
    when the column count matches (8x16 vs 16x8 are both 128) — else the per-column
    coefficients would silently land on the wrong cells (iter 58)."""
    from legoesm.grids.latlon import create_latlon_grid

    from scripts.experiment.check_campaign_deploy import check_deploy
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)                         # base grid is 8 x 16 (128 cols)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    wrong_grid = create_latlon_grid(16, 8, dtype=jnp.float64)   # 16 x 8 — also 128 cols
    out = _campaign_output(tmp_path, base_cfg, grid,
                           grid_for_provenance=wrong_grid)      # provenance from the WRONG grid
    with pytest.raises((ValueError, SystemExit)):
        check_deploy(cfg, out)
