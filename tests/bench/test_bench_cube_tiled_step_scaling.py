"""Direct tests for the sub-face tiled cube step scaling lane (item 5).

The END-TO-END lane needs >=24 devices and a tiled-stage compile that is
cluster-scale (the tiled module's own note: not laptop-benchable; the
pre-existing adapter parity gate itself runs ~10 min on a laptop CPU) — so
these tests lock the fast layers: CLI rejections, the anti-fake HLO census
logic, and the parity-tolerance source; the end-to-end smoke is device-count
gated for cluster CI.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_BENCH = (Path(__file__).resolve().parents[2]
          / "scripts" / "bench" / "bench_cube_tiled_step_scaling.py")
_spec = importlib.util.spec_from_file_location("bench_cube_tiled", _BENCH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_rejects_bad_args(monkeypatch):
    # kt=1 is the face-only lane — refuse (fake >6-device claim otherwise).
    monkeypatch.setattr(sys, "argv", ["bench", "--kt", "1"])
    with pytest.raises(SystemExit):
        mod.main()
    # resolution must tile evenly.
    monkeypatch.setattr(sys, "argv", ["bench", "--kt", "2",
                                      "--resolution", "9"])
    with pytest.raises(SystemExit):
        mod.main()
    # timing window sanity.
    monkeypatch.setattr(sys, "argv", ["bench", "--kt", "2",
                                      "--resolution", "8",
                                      "--steps", "2", "--warmup", "2"])
    with pytest.raises(SystemExit):
        mod.main()


def test_rejects_insufficient_devices(monkeypatch):
    # kt=2 needs 24 devices; the default test process has far fewer ->
    # the device guard must fire BEFORE any model build.
    import jax

    if len(jax.devices()) >= 24:
        pytest.skip("host exposes >=24 devices; guard not reachable")
    monkeypatch.setattr(sys, "argv", ["bench", "--kt", "2",
                                      "--resolution", "8", "--steps", "2",
                                      "--warmup", "1"])
    with pytest.raises(SystemExit, match="need 24 devices"):
        mod.main()


def test_collective_census_helper():
    hlo = "\n".join([
        "%x = collective-permute(...)",
        "%y = collective-permute-start(...)",
        "%z = collective-permute-done(...)",  # not counted (done)
        "%w = add(...)",
    ])
    assert mod._count_collective_permutes(hlo) == 2


def test_parity_tolerances_match_adapter_gate():
    # The lane's tolerances must stay the adapter gate's f32-honest bounds
    # (a silent loosening here would let a stage regression pass the lane).
    assert mod.TILED_PARITY_ATOL == {"u": 2e-5, "v": 2e-5,
                                     "T": 1e-4, "p_s": 0.06}
