"""Single-rank smoke tests for the CRM and spectral-LES MPI scaling benches.

Each bench runs end-to-end at np=1 on a tiny grid and appends ONE JSON line
with finite timings, a finite final state and provenance. Multi-rank runs
need an MPI launcher and are covered by the Derecho ladder job.
"""
from __future__ import annotations

import json

import pytest

from tests.atmosphere.nonhydrostatic.integration._bench_smoke_helpers import (
    REPO_ROOT, fail_on_nonzero, run_bench,
)

CRM = REPO_ROOT / "scripts" / "bench" / "bench_plane_crm_dd_scaling.py"
LES = REPO_ROOT / "scripts" / "bench" / "bench_spectral_les_dd_scaling.py"


def _one_record(path):
    lines = path.read_text().splitlines()
    assert len(lines) == 1, lines
    return json.loads(lines[0])


def _check_common(rec, mode):
    assert rec["mode"] == mode and rec["n_ranks"] == 1
    assert rec["finite"] is True
    assert len(rec["ms_per_step_windows"]) == rec["windows"] == 2
    assert all(t > 0.0 for t in rec["ms_per_step_windows"])
    assert rec["ms_per_step_min"] == min(rec["ms_per_step_windows"])
    assert rec["provenance"]["git_sha"]


@pytest.mark.parametrize("mode", ["strong", "weak"])
def test_crm_bench_single_rank(tmp_path, mode):
    out = tmp_path / "crm.jsonl"
    result = run_bench(CRM, [
        "--mode", mode, "--nx", "8", "--ny", "8", "--nlev", "8",
        "--warmup-steps", "1", "--time-steps", "2", "--windows", "2",
        "--semi-implicit", "--theta-noise-amp", "0.1", "--time-physics", "--physics-calls", "1",
        "--radiation", "gray", "--output", str(out),
    ], timeout_s=900)
    fail_on_nonzero(result, "bench_plane_crm_dd_scaling.py",
                    extra_kwargs=f" --mode {mode}")
    rec = _one_record(out)
    _check_common(rec, mode)
    assert (rec["ny_global"], rec["nx_global"], rec["nlev"]) == (8, 8, 8)
    # production defaults resolved from the driver, not the bench
    assert rec["dycore_config"]["substep_horizontal_acoustic"] is True
    assert rec["dycore_config"]["sgs_vertical_diffusion"] is True
    phys = rec["physics"]
    assert rec["max_abs_dtheta_prime"] > 0.0  # the dynamics actually ran
    assert phys["non_rad_calls_per_step"] == 3  # SSP-RK3 slow-tendency calls
    assert phys["non_rad_ms"] > 0.0 and phys["rad_ms"] > 0.0
    assert phys["radiation_interval"] > 1  # driver default gates radiation
    assert phys["amortised_ms_per_step"] == pytest.approx(
        3 * phys["non_rad_ms"] + phys["rad_ms"] / phys["radiation_interval"])


@pytest.mark.parametrize("mode", ["strong", "weak"])
def test_les_bench_single_rank(tmp_path, mode):
    out = tmp_path / "les.jsonl"
    result = run_bench(LES, [
        "--mode", mode, "--nx", "16", "--ny", "16", "--nz", "16",
        "--warmup-steps", "2", "--time-steps", "2", "--windows", "2",
        "--output", str(out),
    ], timeout_s=900)
    fail_on_nonzero(result, "bench_spectral_les_dd_scaling.py",
                    extra_kwargs=f" --mode {mode}")
    rec = _one_record(out)
    _check_common(rec, mode)
    assert rec["max_abs_dtheta"] > 0.0  # the dynamics actually ran
    assert rec["les_config"]["dealias"] is True
    assert rec["les_config"]["time_scheme"] == "rk3"
    assert rec["les_config"]["smagorinsky_dynamic"] is True
