"""--heat-ledger: CLI round trip, refusal off the wired lane, the driver-side
accumulator, and the restart-fingerprint exclusion.

The model-side ledger (bitwise neutrality, telescoping, known answer) is
tested in tests/ocean/unit/test_heat_ledger.py.
"""
from __future__ import annotations

import os
import pathlib
import sys
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def test_flag_round_trip():
    p = _core2()._build_arg_parser()
    assert p.parse_args([]).heat_ledger is False
    assert p.parse_args(["--heat-ledger"]).heat_ledger is True


@pytest.mark.parametrize("extra", [
    ["--grid", "mpas"],
    ["--grid", "tripole", "--n-gpus", "2"],
    ["--grid", "tripole", "--scan-block", "4"],
])
def test_refused_off_the_wired_lane(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv",
                        ["run_omip_core2.py", "--heat-ledger", *extra])
    with pytest.raises(ValueError, match="--heat-ledger is wired"):
        _core2().main()


def test_excluded_from_restart_fingerprint():
    src = RUNNER.read_text()
    i = src.index("_RESTART_FP_EXCLUDE = frozenset({")
    j = src.index("})", i)
    assert '"heat_ledger"' in src[i:j]


def _state(T, eta):
    f = lambda a: SimpleNamespace(data=jnp.asarray(a))  # noqa: E731
    return SimpleNamespace(T=f(T), eta=f(eta), H_bathy=f(np.full(eta.shape, 1000.0)))


def test_accumulator_known_answer():
    """Synthetic steps with known content changes: every stage, the controls
    and the applied q_net come back in J, summed over the window."""
    from legoesm.ocean.vertical import create_ocean_z_star
    core2 = _core2()
    ny, nx, nz = 3, 6, 2
    zc = create_ocean_z_star(n_levels=nz, H_max=1000.0)
    area = np.full((ny, nx), 1.0e6)
    lm = np.ones((ny, nx))
    acc = core2._HeatLedgerAccumulator(
        area=area, land_mask=lm, z_coord=zc, min_water_column_m=0.5,
        ew_cyclic_overlap=True, rho_0=1000.0, c_sw=4000.0)
    vol = 1000.0 * area[:, 1:-1].sum()          # counted columns, full depth
    eta = np.zeros((ny, nx))
    s0 = _state(np.full((ny, nx, nz), 1.0), eta)
    s1 = _state(np.full((ny, nx, nz), 2.0), eta)   # model step: +1 K
    s2 = _state(np.full((ny, nx, nz), 2.5), eta)   # post-step:  +0.5 K
    for _ in range(2):
        acc.begin_step(s0, jnp.full((ny, nx), 10.0), 100.0)
        acc.model_step(s1, {"entry": acc.content(s0),
                            "step_a": 0.25 * vol, "step_b": 0.75 * vol,
                            "split_x": 123.0, "end": acc.content(s1)})
        acc.mark("post_geothermal", s2)
        acc.end_step()
    out = acc.drain()
    k = 1000.0 * 4000.0
    assert int(out["ledger_n_steps"]) == 2
    assert float(out["ledger_model_step_J"]) == pytest.approx(2 * k * vol)
    assert float(out["ledger_step_a_J"]) == pytest.approx(2 * 0.25 * k * vol)
    assert float(out["ledger_post_geothermal_J"]) == pytest.approx(2 * 0.5 * k * vol)
    assert abs(float(out["ledger_model_step_unattributed_J"])) <= 1e-9 * k * vol
    assert abs(float(out["ledger_model_entry_mismatch_J"])) <= 1e-9 * k * vol
    # step 2 began at s0 after step 1 ended at s2: -1.5 K between steps
    assert float(out["ledger_between_steps_J"]) == pytest.approx(-1.5 * k * vol)
    assert float(out["ledger_applied_qnet_J"]) == pytest.approx(
        2 * 100.0 * 10.0 * area[:, 1:-1].sum())
    assert list(out["ledger_stages"]) == [
        "applied_qnet", "model_step", "step_a", "step_b", "split_x",
        "model_step_unattributed", "model_entry_mismatch", "post_geothermal",
        "between_steps"]
    assert acc.drain() == {}


def test_accumulator_refuses_fp32(monkeypatch):
    core2 = _core2()
    monkeypatch.setattr(core2, "jax", SimpleNamespace(
        config=SimpleNamespace(jax_enable_x64=False), jit=jax.jit))
    with pytest.raises(SystemExit, match="fp64"):
        core2._HeatLedgerAccumulator(
            area=np.ones((2, 2)), land_mask=np.ones((2, 2)), z_coord=None,
            min_water_column_m=0.5, ew_cyclic_overlap=False, rho_0=1.0,
            c_sw=1.0)


def test_step_and_post_step_marks_are_wired():
    src = RUNNER.read_text()
    assert src.count("model.step_with_heat_ledger(") == 1
    for stage in ("post_sss_restore", "post_ice_freeze_relax",
                  "post_geothermal", "post_isf", "post_bbl", "post_woa_nudge"):
        assert src.count(f'_hl_acc.mark("{stage}", state)') == 1, stage
    assert src.count("heat_ledger=_hl_acc") == 2
