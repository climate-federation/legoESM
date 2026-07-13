"""Smoke tests for ``scripts/run/run_fluxnet_offline.py``.

Verifies the runner module imports cleanly (sys.path bootstrap + netCDF4
shim ordering + CHATS7 writer reuse), that ``_parse_args`` returns a
namespace with the expected fields, and that the FLUXNET forcing loader
integrates with the site metadata.

We do NOT run ``compute_clm_ml_canopy_fluxes`` here -- per-step wall time
is ~20-25 s so a full step would time-out the smoke suite.

Skipped when either:
  * the US-MMS slim CSV is not present in the checkout, or
  * the ``multilayer_canopy`` package (from ``clm-ml-jax/src``) is not
    importable (minimal CI images).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


# tests/land/unit/<file>.py -> parents[3] = repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_RUNNER_PATH = _REPO_ROOT / "scripts" / "run" / "run_fluxnet_offline.py"
_US_MMS_CSV = (
    _REPO_ROOT / "data" / "fluxnet" / "US-MMS"
    / "AMF_US-MMS_FLUXNET_FLUXMET_HR_1999-2023_v1.3_r1_slim.csv"
)


def _multilayer_canopy_available() -> bool:
    """Check ``multilayer_canopy`` (from ``clm-ml-jax/src``) is importable.

    We add ``clm-ml-jax/src`` to ``sys.path`` explicitly to match the
    runner's own bootstrap -- ``multilayer_canopy`` is not installed as
    a proper package.
    """
    clm_src = _REPO_ROOT / "clm-ml-jax" / "src"
    if not clm_src.exists():
        return False
    if str(clm_src) not in sys.path:
        sys.path.insert(0, str(clm_src))
    return importlib.util.find_spec("multilayer_canopy") is not None


_SKIP_REASON = None
if not _US_MMS_CSV.exists():
    _SKIP_REASON = f"US-MMS slim FLUXNET CSV not present at {_US_MMS_CSV}"
elif not _multilayer_canopy_available():
    _SKIP_REASON = "multilayer_canopy package not importable (clm-ml-jax/src missing)"

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


def _load_runner():
    """Import the runner module from its file path.

    The runner does its own sys.path bootstrap + netCDF4 shim install at
    import time; a successful ``exec_module`` is proof both worked.
    """
    spec = importlib.util.spec_from_file_location(
        "_run_fluxnet_offline_under_test", _RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_imports_cleanly() -> None:
    """The module loads: sys.path bootstrap + shim ordering are correct."""
    mod = _load_runner()
    # Sanity: the CHATS7 writers were re-exported by the module.
    assert callable(mod._write_flux_row)
    assert callable(mod._write_fsun_row)
    assert callable(mod._write_aux_row)
    # Sanity: the FLUXNET loader is bound.
    assert callable(mod.load_fluxnet_forcing)
    assert callable(mod.build_atm2sfc_at)


def test_parse_args_us_mms_defaults() -> None:
    """``_parse_args`` returns the expected namespace fields for US-MMS."""
    mod = _load_runner()
    args = mod._parse_args([
        "--site", "US-MMS",
        "--output-dir", "results/us_mms_smoke",
    ])
    assert args.site == "US-MMS"
    assert args.output_dir == pathlib.Path("results/us_mms_smoke")
    # Defaults from the CLI spec.
    assert args.start_date is None
    assert args.end_date is None
    assert args.spinup_days == 0
    assert args.met_type == 3
    assert args.num_ml_steps == 30
    assert args.runge_kutta_type == 41
    assert args.allow_snow is False
    assert args.csv_path is None


def test_parse_args_fi_hyy_with_overrides() -> None:
    """CLI overrides propagate (window + met/RK/spinup + allow-snow)."""
    mod = _load_runner()
    args = mod._parse_args([
        "--site", "FI-Hyy",
        "--output-dir", "results/fi_hyy_smoke",
        "--start-date", "2010-06-01",
        "--end-date", "2010-06-30",
        "--spinup-days", "45",
        "--met-type", "1",
        "--num-ml-steps", "60",
        "--runge-kutta-type", "20",
        "--allow-snow",
    ])
    assert args.site == "FI-Hyy"
    assert args.start_date == "2010-06-01"
    assert args.end_date == "2010-06-30"
    assert args.spinup_days == 45
    assert args.met_type == 1
    assert args.num_ml_steps == 60
    assert args.runge_kutta_type == 20
    assert args.allow_snow is True


def test_parse_args_rejects_unknown_site() -> None:
    """The CLI must reject a site code that isn't in the ``SITES`` dict."""
    mod = _load_runner()
    with pytest.raises(SystemExit):
        mod._parse_args([
            "--site", "US-BOGUS",
            "--output-dir", "results/bogus",
        ])


def test_default_csv_for_us_mms_resolves() -> None:
    """The default-CSV resolver finds the in-tree slim US-MMS file."""
    mod = _load_runner()
    csv_path = mod._default_csv_for_site("US-MMS")
    assert csv_path.exists()
    assert csv_path.name.endswith("_slim.csv")
    assert csv_path.parent.name == "US-MMS"


def test_forcing_loader_integrates_with_site_metadata() -> None:
    """The runner's loader path returns a FluxnetForcing wired to SITES."""
    mod = _load_runner()
    csv_path = mod._default_csv_for_site("US-MMS")
    forcing = mod.load_fluxnet_forcing(csv_path, "US-MMS")
    # Site metadata is the shared SITES entry (identity check).
    assert forcing.site is mod.SITES["US-MMS"]
    # PFT code is CLM's broadleaf_deciduous_temperate_tree.
    assert forcing.site.pft_clm == 7
    # Slim CSV is hourly (dt = 3600 s).
    assert forcing.dt_s == 3600.0
    # At least some rows should be valid.
    assert forcing.valid.any()


def test_snow_filter_only_affects_fi_hyy() -> None:
    """FI-Hyy default drops Nov-Apr; other sites left alone regardless."""
    import numpy as np

    mod = _load_runner()
    # Build a tiny synthetic mask/forcing-like object to hit the branch
    # without loading a whole CSV.  The filter only touches
    # ``forcing.site.code`` and ``forcing.time.month`` so a duck-typed
    # SimpleNamespace suffices.
    from types import SimpleNamespace

    import pandas as pd

    time = pd.DatetimeIndex(["2020-01-15", "2020-06-15", "2020-11-15"])
    mask = np.ones(3, dtype=bool)

    # US-MMS: no snow filter applied.
    us_mms = SimpleNamespace(site=mod.SITES["US-MMS"], time=time)
    out_mask, dropped = mod._apply_snow_filter(us_mms, mask.copy(), allow_snow=False)
    assert dropped == 0
    assert out_mask.tolist() == [True, True, True]

    # FI-Hyy: drops Jan + Nov, keeps Jun.
    fi_hyy = SimpleNamespace(site=mod.SITES["FI-Hyy"], time=time)
    out_mask, dropped = mod._apply_snow_filter(fi_hyy, mask.copy(), allow_snow=False)
    assert dropped == 2
    assert out_mask.tolist() == [False, True, False]

    # FI-Hyy with --allow-snow: no drops.
    out_mask, dropped = mod._apply_snow_filter(fi_hyy, mask.copy(), allow_snow=True)
    assert dropped == 0
    assert out_mask.tolist() == [True, True, True]
