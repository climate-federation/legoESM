"""CLI coverage for the real LMIP entrypoint."""

from __future__ import annotations

from scripts.run.run_lmip import _parse_args, build_config_from_args


def test_issue484_new_lmip_flags_flow_to_config():
    args = _parse_args([
        "--lat", "45.5",
        "--max-wallclock-seconds", "5400",
        "--restart-buffer-seconds", "450",
        "--seed", "91",
        "--cd-land", "0.004",
        "--ch-land", "0.005",
        "--z0-land", "0.08",
        "--beta-min", "0.2",
        "--carbon-scheme", "differland",
        "--no-snow-albedo-feedback",
    ])
    cfg = build_config_from_args(args)

    assert cfg.max_wallclock_seconds == 5400
    assert cfg.restart_buffer_seconds == 450
    assert cfg.seed == 91
    assert cfg.land.Cd_land == 0.004
    assert cfg.land.Ch_land == 0.005
    assert cfg.land.z0_land == 0.08
    assert cfg.land.beta_min == 0.2
    assert cfg.land.carbon.scheme == "differland"
    assert cfg.land.snow_albedo_feedback is False
