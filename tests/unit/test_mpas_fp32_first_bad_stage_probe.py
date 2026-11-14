"""The first-bad-stage probe reports min/max/non-finite counts eagerly and
from traced code, without hiding NaNs."""
import csv
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp


def _load(tmp_path, monkeypatch):
    monkeypatch.setenv("PROBE_CSV", str(tmp_path / "p.csv"))
    monkeypatch.setenv("PROBE_ARM", "t")
    path = (Path(__file__).resolve().parents[2] / "scripts" / "validate"
            / "mpas_fp32_first_bad_stage.py")
    spec = importlib.util.spec_from_file_location("_fbs_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_reports_nan_eager_and_traced(tmp_path, monkeypatch):
    P = _load(tmp_path, monkeypatch)
    x = jnp.array([1.0, jnp.nan, -3.0], jnp.float32)
    P._report("eager", {"x": x})

    @jax.jit
    def f(y):
        P._report("traced", {"y": y * 2.0})
        return y
    f(x)
    jax.effects_barrier()
    P._FH.close()
    rows = list(csv.DictReader(open(tmp_path / "p.csv")))
    by = {r["stage"]: r for r in rows}
    assert by["eager"]["n_nonfinite"] == "1" and by["traced"]["n_nonfinite"] == "1"
    assert by["eager"]["dtype"] == "float32"
    assert by["traced"]["leaf"] == "['y']"
