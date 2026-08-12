"""Unit tests for the #1516 GPU-binding validator's verdict logic.

``evaluate_binding`` is the pure gate the batch validator exits on; these
tests pin its semantics against the placements MEASURED on Levante:

* the FIXED shape (job 26829180 arms A/C): each rank's PID on exactly one,
  different physical GPU — pinned PASSes, control calls it UNEXPECTED;
* the DEFECT shapes: both ranks colliding on one GPU (jobs 26806063 /
  26815351) and each rank spread over BOTH GPUs with compute defaulting to
  device 0 (job 26829180 arm B; also the pre-reorder job 26829100 where a
  post-mpi4py pin was silently ignored) — pinned FAILs, control is
  red-as-expected.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_gpu_binding_np2",
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts" / "validate" / "check_gpu_binding_np2.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
evaluate_binding = _MOD.evaluate_binding

_A = "GPU-aaaa"
_B = "GPU-bbbb"


def _recs(*uuid_lists):
    return [{"rank": str(i), "physical_uuids": list(u)}
            for i, u in enumerate(uuid_lists)]


def test_distinct_single_gpus_pass_pinned():
    ok, msg = evaluate_binding(_recs([_A], [_B]), "pinned")
    assert ok and "PASS" in msg


def test_shared_gpu_fails_pinned():
    # Jobs 26806063 / 26815351: both ranks computed on one physical GPU.
    ok, msg = evaluate_binding(_recs([_A], [_A]), "pinned")
    assert not ok and "FAIL" in msg


def test_rank_spread_over_both_gpus_fails_pinned():
    # Job 26829100/26829180 arm B: n_local=2 per rank, contexts on BOTH
    # devices, compute defaulting to device 0 — the collision signature.
    ok, msg = evaluate_binding(_recs([_A, _B], [_A, _B]), "pinned")
    assert not ok and "FAIL" in msg


def test_control_arm_requires_the_collision():
    # Unpinned control: collision = red-as-expected (exit ok) ...
    ok, msg = evaluate_binding(_recs([_A, _B], [_A, _B]), "unpinned")
    assert ok and "RED AS EXPECTED" in msg
    ok, msg = evaluate_binding(_recs([_A], [_A]), "unpinned")
    assert ok
    # ... and a distinct placement WITHOUT the pin means the control
    # proves nothing -> exit nonzero.
    ok, msg = evaluate_binding(_recs([_A], [_B]), "unpinned")
    assert not ok and "UNEXPECTED" in msg


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        evaluate_binding(_recs([_A], [_B]), "bogus")
