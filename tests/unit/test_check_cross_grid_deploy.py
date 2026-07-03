"""The cross-grid env-kernel deploy-check CLI (check_cross_grid_deploy.py).

Locks the CLI glue: parse -> target grid/sigma from the base config -> restart ColumnState
(SST from --sst-npz) -> build_env_kernel_deployed_override -> report coverage. The restart
LOADING (the only part needing a real checkpoint file) is monkeypatched; the kernel
evaluation + deploy validation + SST fail-loud run FOR REAL on a synthetic target state.
"""
from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest


def _write_kernel_and_config(tmp_path):
    """A real env-kernel JSON + a real target base config (latlon 8 -> 8x16, nlev 5)."""
    from typing import NamedTuple

    from legoesm.driver.config import config_to_dict
    from legoesm.training.column_manifest import ColumnEnvironment
    from legoesm.training.deploy_correction import build_env_kernel, env_kernel_to_dict

    from scripts.experiment.write_amip_clubb_lite_config import (
        build_amip_clubb_lite_config,
    )

    class _Diag(NamedTuple):
        C_K: object
        valid: object

    class _Rec(NamedTuple):
        environment: ColumnEnvironment

    envs = [(298.0, 100.0, 5.0), (300.0, 1500.0, 12.0), (302.0, 2800.0, 20.0)]
    recs = [_Rec(ColumnEnvironment(*e)) for e in envs]
    diags = [_Diag(C_K=jnp.array([c, c]), valid=jnp.array([True, True]))
             for c in (0.30, 0.45, 0.62)]
    kernel = build_env_kernel(recs, diags, "clubb_coefficient",
                              length_scales=[2.0, 1200.0, 8.0], field="C_K")
    kpath = tmp_path / "out.json.env_kernel.json"
    kpath.write_text(json.dumps(env_kernel_to_dict(kernel)))
    cpath = tmp_path / "target.json"
    cpath.write_text(json.dumps(config_to_dict(
        build_amip_clubb_lite_config(resolution=8, nlev=5))))
    return str(kpath), str(cpath)


def _patch_restart_loader(monkeypatch):
    """Make load_model_from_restart return a synthetic 8x16xnlev target ColumnState that
    HONORS the passed sst_K (so the --sst-npz / require_sst paths are exercised for real)."""
    from legoesm.training.compare_reanalysis import ColumnState

    import scripts.validate.compare_amip_era5 as cae

    def _fake(restart_path, grid, sigma, nlev, *, sst_K=None, **kw):
        nlat, nlon = 8, 16
        return ColumnState(
            T=jnp.full((nlat, nlon, nlev), 290.0), q_v=jnp.full((nlat, nlon, nlev), 8e-3),
            u=jnp.full((nlat, nlon, nlev), 10.0), v=jnp.zeros((nlat, nlon, nlev)),
            p_s=jnp.full((nlat, nlon), 1.0e5), sst_K=sst_K)

    monkeypatch.setattr(cae, "load_model_from_restart", _fake)


def test_parser_flag_contract():
    from scripts.experiment.check_cross_grid_deploy import _build_arg_parser

    ns = _build_arg_parser().parse_args(
        ["--kernel", "k.json", "--target-base-config", "c.json",
         "--target-restart", "r.npz", "--sst-npz", "s.npz",
         "--min-fraction-covered", "0.7", "--allow-approximate-sst"])
    assert ns.kernel == "k.json" and ns.sst_npz == "s.npz"
    assert ns.min_fraction_covered == 0.7 and ns.allow_approximate_sst is True


def test_cross_grid_deploy_ok_with_sst(tmp_path, monkeypatch, capsys):
    """Happy path: a real SST (--sst-npz) → the kernel deploys on the target → exit 0,
    coverage with sst_from_model=True reported."""
    from scripts.experiment.check_cross_grid_deploy import main

    kpath, cpath = _write_kernel_and_config(tmp_path)
    _patch_restart_loader(monkeypatch)
    rpath = tmp_path / "target_restart.npz"
    rpath.write_text("")                                  # only needs to EXIST (loader patched)
    sst = tmp_path / "sst.npz"
    np.savez(sst, sst_K=np.full((8, 16), 300.0))

    rc = main(["--kernel", kpath, "--target-base-config", cpath,
               "--target-restart", str(rpath), "--sst-npz", str(sst)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "env-kernel deploys onto the target grid (8, 16)" in out
    assert "sst_from_model=True" in out and "apply_env_kernel_override" in out


def test_cross_grid_deploy_fails_loud_without_sst(tmp_path, monkeypatch):
    """No --sst-npz and no --allow-approximate-sst → require_sst fail-loud (the env-kernel's
    dominant predictor would be a fabricated SST — untrustworthy for a production deploy)."""
    from scripts.experiment.check_cross_grid_deploy import main

    kpath, cpath = _write_kernel_and_config(tmp_path)
    _patch_restart_loader(monkeypatch)
    rpath = tmp_path / "target_restart.npz"
    rpath.write_text("")
    with pytest.raises(ValueError, match="require_sst=True"):
        main(["--kernel", kpath, "--target-base-config", cpath,
              "--target-restart", str(rpath)])


def test_cross_grid_deploy_fail_fast_on_missing_path(tmp_path):
    """A typo'd path fails fast (SystemExit) BEFORE the heavy grid/restart machinery."""
    from scripts.experiment.check_cross_grid_deploy import main

    with pytest.raises(SystemExit, match="does not exist"):
        main(["--kernel", str(tmp_path / "nope.json"),
              "--target-base-config", str(tmp_path / "nope2.json"),
              "--target-restart", str(tmp_path / "nope3.npz")])
