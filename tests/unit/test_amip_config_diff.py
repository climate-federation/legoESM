"""Direct test of scripts/validate/amip_config_diff.py, run-vs-run mode."""
import importlib.util
import json
import pathlib

_P = pathlib.Path(__file__).resolve().parents[2] / "scripts/validate/amip_config_diff.py"
_spec = importlib.util.spec_from_file_location("amip_config_diff", _P)
acd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acd)


def _run(tmp_path, name, cfg):
    d = tmp_path / name
    d.mkdir()
    (d / "experiment_config.json").write_text(json.dumps(cfg))
    return str(d)


def test_two_runs_differing_only_as_expected_pass(tmp_path, capsys):
    a = _run(tmp_path, "ctl", {"land_snow_scheme": "bulk", "output_dir": "x",
                               "land": {"freeze_thaw": True}})
    b = _run(tmp_path, "arm", {"land_snow_scheme": "layered", "output_dir": "y",
                               "land": {"freeze_thaw": True}})
    assert acd.main(["--against", a, "--run", b, "--expect", "land_snow_scheme"]) == 0
    assert "land_snow_scheme" in capsys.readouterr().out


def test_an_unexpected_or_one_sided_field_fails(tmp_path):
    a = _run(tmp_path, "ctl", {"land_snow_scheme": "bulk", "land": {"freeze_thaw": True}})
    b = _run(tmp_path, "arm", {"land_snow_scheme": "layered", "land": {"freeze_thaw": False}})
    assert acd.main(["--against", a, "--run", b, "--expect", "land_snow_scheme"]) == 1
    c = _run(tmp_path, "arm2", {"land_snow_scheme": "layered",
                                "land": {"freeze_thaw": True}, "land_snow_insulation": False})
    assert acd.main(["--against", a, "--run", c, "--expect", "land_snow_scheme"]) == 1
    assert acd.main(["--against", a, "--run", c,
                     "--expect", "land_snow_scheme,land_snow_insulation"]) == 0


def test_run_vs_run_compares_input_paths_but_not_output_dirs(tmp_path):
    a = _run(tmp_path, "ctl", {"land_snow_scheme": "bulk", "output_dir": "x",
                               "land_ic_path": "/ic/a.npz"})
    b = _run(tmp_path, "arm", {"land_snow_scheme": "layered", "output_dir": "y",
                               "land_ic_path": "/ic/b.npz"})
    assert acd.main(["--against", a, "--run", b, "--expect", "land_snow_scheme"]) == 1
