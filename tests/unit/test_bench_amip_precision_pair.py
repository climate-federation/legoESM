"""bench_amip_precision_pair: argument parsing, input-path resolution, log
parsing and the JSON summary (no GPU, no model run)."""
import importlib.util
import json
from pathlib import Path

_PY = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "bench_amip_precision_pair.py"
spec = importlib.util.spec_from_file_location("bench_amip_precision_pair", _PY)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

_LOG = """\
land drag: 0 land column-steps on the neutral seed ... and 6720 reusing the last valid land drag (latest solve held or non-finite) up to step 432; run totals 20709 / 6720
  Day    1.0: T=[185.3,307.4]K mean=251.1K  p_s=985.1hPa  |u|_max=118.9m/s  (0.0010 sim-days/s, 966s)
land drag: 0 ... and 3952 reusing the last valid land drag (latest solve held or non-finite) up to step 864; run totals 20709 / 10672
  Day    2.0: T=[185.6,307.6]K mean=251.0K  (0.0015 sim-days/s, 1341s)
  Day    3.0: T=[182.3,307.5]K mean=250.9K  (0.0017 sim-days/s, 1716s)
"""


def test_days_below_two_refused():
    import pytest
    with pytest.raises(SystemExit):
        B.build_parser().parse_args(["--output", "x", "--days", "1"])


def test_arm_command_env_and_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    args = B.build_parser().parse_args(["--output", str(tmp_path), "--gpu", "1",
                                        "--land-ic", "/x/soil.npz"])
    for name, deck, x64 in B.ARMS:
        cmd, env = B.arm_command(name, deck, x64, ["--ic-path", "/ic"], args)
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
        assert env.get("JAX_ENABLE_X64") == ("1" if x64 else None)
        assert cmd[cmd.index("--config") + 1] == f"config/amip/{deck}"
        assert cmd[cmd.index("--output") + 1] == str(tmp_path / name)
        assert cmd[cmd.index("--land-ic") + 1] == "/x/soil.npz"
        assert "--params" not in cmd


def test_cpu_only_platform_refused(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    with pytest.raises(SystemExit, match="non-GPU"):
        B.main(["--output", str(tmp_path / "o")])


def test_arm_ok_needs_completed_exit0_and_rate():
    good = {"status": "COMPLETED", "exit_code": 0, "s_per_sim_day": 375.0}
    assert B.arm_ok(good)
    assert not B.arm_ok({**good, "exit_code": 1})
    assert not B.arm_ok({**good, "status": "BLOWUP at day 1"})
    assert not B.arm_ok({**good, "s_per_sim_day": None})


def test_parser_defaults_and_arms():
    a = B.build_parser().parse_args(["--output", "x"])
    assert (a.days, a.arms, a.land_ic, a.params) == (3, ["fp64", "fp32"], None, None)
    a = B.build_parser().parse_args(["--output", "x", "--days", "2", "--arms", "fp32"])
    assert (a.days, a.arms) == (2, ["fp32"])


def test_decks_exist_and_x64_only_on_fp64():
    for name, deck, x64 in B.ARMS:
        assert (B.REPO / "config" / "amip" / deck).is_file()
        assert x64 == (name == "fp64")


def test_path_flags_follow_the_environment(tmp_path, monkeypatch):
    for var in set(B._PATH_FLAGS.values()):
        monkeypatch.setenv(var, str(tmp_path / var))
    flags = B.path_flags()
    # every input flag of amip_production.sh is known to the checker
    named = {f for f in flags if f.startswith("--")} - {"--sst-offset", "--volcanic-aerosol-lw"}
    assert named == set(B._PATH_FLAGS)
    assert flags[flags.index("--ic-path") + 1] == str(tmp_path / "ERA5_IC")
    monkeypatch.setenv("ERA5_IC", "rel/ic.zarr")       # relative -> absolute
    monkeypatch.chdir(tmp_path)
    got = B.path_flags()
    assert got[got.index("--ic-path") + 1] == str(tmp_path / "rel" / "ic.zarr")
    miss = dict(B.missing_inputs(flags))
    assert set(miss) == set(B._PATH_FLAGS.values())
    (tmp_path / "ERA5_IC").mkdir()
    assert "ERA5_IC" not in dict(B.missing_inputs(flags))


def test_parse_log_and_rate():
    days, held = B.parse_log(_LOG)
    assert days == {1.0: 966, 2.0: 1341, 3.0: 1716}
    assert held == 10672                 # the run's total, not a sum of totals
    assert B.parse_log("no land")[1] is None
    assert B.s_per_day(days) == (1716 - 966) / 2
    assert B.s_per_day({1.0: 966}) is None


def test_run_status(tmp_path):
    assert B.run_status(tmp_path / "none.txt") == "NO STATUS (run died)"
    (tmp_path / "r.txt").write_text("legoESM AMIP run\nStatus: BLOWUP at day 1\n")
    assert B.run_status(tmp_path / "r.txt") == "BLOWUP at day 1"


def test_write_summary(tmp_path):
    args = B.build_parser().parse_args(["--output", str(tmp_path)])
    env = {"git_sha": "abc", "gpu": "RTX", "jax": "0.0"}
    arms = {"fp32": {"s_per_sim_day": 375.0, "day_end_wall_s": {1.0: 966}}}
    B.write_summary(tmp_path / "s.json", args, env, arms)
    got = json.loads((tmp_path / "s.json").read_text())
    assert got["git_sha"] == "abc" and got["days"] == 3
    assert got["arms"]["fp32"]["s_per_sim_day"] == 375.0
    assert set(B.environment()) >= {"git_sha", "gpu", "jax", "jaxlib", "cuda_plugin",
                                    "cuda_runtime", "driver_cuda"}
    assert B._driver_cuda("| Driver Version: 570.1   CUDA Version: 12.8 |") == "12.8"
