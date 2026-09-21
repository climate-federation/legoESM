"""Direct tests for the shared scaling-benchmark metadata helper.

Locks the roadmap item-9 contract: every scaling record is self-describing,
and the GPU-direct / precision-knob fields make a host-staged or f32-ablation
run falsifiable from the record alone.
"""

from __future__ import annotations

import os

import importlib.util
from pathlib import Path

import pytest

_MOD = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "metadata.py"
_spec = importlib.util.spec_from_file_location("bench_metadata", _MOD)
md = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(md)


def _record(**over):
    kw = dict(
        grid="latlon",
        component="ocean",
        resolution=100,
        n_levels=75,
        precision="float64",
        n_ranks=8,
    )
    kw.update(over)
    return md.scaling_metadata(**kw)


def test_record_carries_all_required_keys():
    rec = _record()
    for k in md.REQUIRED_KEYS:
        assert k in rec, f"missing required key {k!r}"
        assert rec[k] not in (None, ""), f"empty required key {k!r}"
    # validate() must accept a well-formed record.
    assert md.validate_scaling_metadata(rec) == []


def test_record_carries_roadmap_item9_fields():
    rec = _record(
        n_gpus=8,
        decomposition="2d_pencil",
        solver_variant="single_reduce_pcg",
        solver_residual=3.2e-11,
        conservation_drift=1e-14,
        cells_per_rank=4096,
        scaling_kind="strong",
    )
    for k in (
        "grid", "component", "resolution", "n_levels", "precision",
        "precision_knobs", "backend", "n_ranks", "n_gpus", "device_count",
        "devices_per_rank", "decomposition", "solver_variant",
        "solver_residual", "cells_per_rank", "scaling_kind",
        "gpu_direct_requested", "gpu_direct_active", "host_staged_halo",
        "schema_version", "timestamp_utc", "cpu_affinity",
    ):
        assert k in rec, f"item-9 field {k!r} absent"
    if hasattr(os, "sched_getaffinity"):
        assert rec["cpu_affinity"] == len(os.sched_getaffinity(0)) > 0
    else:
        assert rec["cpu_affinity"] is None
    assert rec["solver_variant"] == "single_reduce_pcg"
    assert rec["solver_residual"] == pytest.approx(3.2e-11)
    assert rec["scaling_kind"] == "strong"


def test_precision_knobs_snapshot_env(monkeypatch):
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    monkeypatch.setenv("LEGOESM_BAROCLINIC_F32", "1")
    monkeypatch.delenv("LEGOESM_VMIX_F32_SOLVE", raising=False)
    knobs = md.precision_knobs()
    assert knobs["JAX_ENABLE_X64"] == "1"
    assert knobs["LEGOESM_BAROCLINIC_F32"] == "1"
    assert knobs["LEGOESM_VMIX_F32_SOLVE"] == "0"  # default off, still recorded
    # And the full record carries the snapshot.
    assert _record()["precision_knobs"]["LEGOESM_BAROCLINIC_F32"] == "1"


def test_gpu_direct_active_requires_gpu_and_toggle_and_cuda(monkeypatch):
    monkeypatch.setenv("MPI4JAX_USE_CUDA_MPI", "1")
    monkeypatch.setattr(md, "mpi4jax_cuda_support", lambda: True)
    g = md.gpu_direct_mode(backend="gpu")
    assert g["gpu_direct_requested"] is True
    assert g["gpu_direct_active"] is True
    assert g["host_staged_halo"] is False


def test_gpu_run_without_cuda_mpi_is_flagged_host_staged(monkeypatch):
    # GPU backend + toggle set but mpi4jax lacks CUDA support => host-staged
    # halo (a scaling bug that must be visible in the record, not hidden).
    monkeypatch.setenv("MPI4JAX_USE_CUDA_MPI", "1")
    monkeypatch.setattr(md, "mpi4jax_cuda_support", lambda: False)
    g = md.gpu_direct_mode(backend="gpu")
    assert g["gpu_direct_active"] is False
    assert g["host_staged_halo"] is True


def test_gpu_run_without_toggle_is_host_staged(monkeypatch):
    monkeypatch.delenv("MPI4JAX_USE_CUDA_MPI", raising=False)
    monkeypatch.setattr(md, "mpi4jax_cuda_support", lambda: True)
    g = md.gpu_direct_mode(backend="gpu")
    assert g["gpu_direct_requested"] is False
    assert g["gpu_direct_active"] is False
    assert g["host_staged_halo"] is True


def test_cpu_backend_never_host_staged(monkeypatch):
    # On CPU the GPU-direct toggle is moot; neither flag should fire.
    monkeypatch.setenv("MPI4JAX_USE_CUDA_MPI", "1")
    monkeypatch.setattr(md, "mpi4jax_cuda_support", lambda: False)
    g = md.gpu_direct_mode(backend="cpu")
    assert g["gpu_direct_active"] is False
    assert g["host_staged_halo"] is False


def test_validate_raises_on_missing_required_key():
    rec = _record()
    del rec["grid"]
    with pytest.raises(ValueError, match="not self-describing"):
        md.validate_scaling_metadata(rec)
    # Non-strict returns the list instead of raising.
    assert md.validate_scaling_metadata(rec, strict=False) == ["grid"]


def test_validate_flags_empty_string_required_key():
    rec = _record(component="")
    with pytest.raises(ValueError):
        md.validate_scaling_metadata(rec)


def test_validate_catches_missing_comparability_field():
    # A record missing a COMPARABILITY field (not just 'grid') must fail —
    # precision_knobs / gpu_direct_active / device counts are what make a row
    # falsifiable, so their absence is a hygiene failure.
    for k in ("precision_knobs", "gpu_direct_active", "device_count",
              "process_count", "n_gpus", "host_staged_halo"):
        rec = _record()
        del rec[k]
        assert k in md.validate_scaling_metadata(rec, strict=False), (
            f"validator missed missing comparability key {k!r}")
        with pytest.raises(ValueError):
            md.validate_scaling_metadata(rec)


def test_validate_flags_empty_precision_knobs():
    # An EMPTY precision_knobs dict hides an f32/TF32 ablation => must fail,
    # even though it is not None/"".
    rec = _record()
    rec["precision_knobs"] = {}
    assert "precision_knobs" in md.validate_scaling_metadata(rec, strict=False)
    with pytest.raises(ValueError):
        md.validate_scaling_metadata(rec)


def test_validate_allows_scalar_zero_and_false():
    # 0 / False are legitimate values for required keys (CPU run: n_gpus=0,
    # gpu_direct_active=False) and must NOT be flagged as empty.
    rec = _record(n_gpus=0)
    rec["gpu_direct_active"] = False
    rec["host_staged_halo"] = False
    assert md.validate_scaling_metadata(rec) == []


def test_validate_flags_absent_present_key():
    rec = _record()
    del rec["solver_variant"]
    problems = md.validate_scaling_metadata(rec, strict=False)
    assert any("solver_variant" in p for p in problems)


def test_n_ranks_defaults_to_process_count():
    # Omitting n_ranks (single-process SPMD driver) must record the true
    # process count, NOT a device count — the codex-flagged SPMD mislabel.
    rec = md.scaling_metadata(
        grid="cubed_sphere", component="atmosphere", resolution=48,
        n_levels=30, precision="float32", n_gpus=4)
    assert rec["n_ranks"] == rec["process_count"]
    # Explicit rank count (route-A) is honored verbatim.
    rec2 = _record(n_ranks=16)
    assert rec2["n_ranks"] == 16


def test_annotate_incomplete_flags_without_raising(recwarn):
    good = _record()
    assert md.annotate_incomplete(good) is good
    assert "_incomplete" not in good  # a complete record is untouched
    bad = _record()
    del bad["backend"]
    out = md.annotate_incomplete(bad)
    assert out["_incomplete"] == ["backend"]  # flagged, not raised
    assert any(issubclass(w.category, RuntimeWarning) for w in recwarn.list)


def test_partition_metrics_and_extra_passthrough():
    rec = _record(
        partition_metrics={"edge_cut": 1234, "owned_halo_ratio": 0.12,
                           "cells_per_rank_min": 4000,
                           "cells_per_rank_max": 4200, "message_count": 6},
        extra={"note": "metis"},
    )
    assert rec["partition_metrics"]["edge_cut"] == 1234
    assert rec["extra"]["note"] == "metis"


def test_decomposition_none_is_legal_single_device():
    rec = _record(decomposition="none", n_ranks=1)
    assert md.validate_scaling_metadata(rec) == []  # "none" != empty


def test_devices_per_rank_defaults_from_jax_counts():
    # With no override it is derived (or None on a JAX-less env); if present it
    # must be a positive int.
    rec = _record()
    dpr = rec["devices_per_rank"]
    assert dpr is None or (isinstance(dpr, int) and dpr >= 1)
    # Explicit override is honored verbatim.
    assert _record(devices_per_rank=4)["devices_per_rank"] == 4


# --------------------------------------------------------------------------
# schema v2: transport / virtual_cpu_devices / launcher (anti-fake-scaling)
# --------------------------------------------------------------------------

def test_schema_v2_required_keys_present():
    rec = _record()
    for k in ("transport", "virtual_cpu_devices", "launcher"):
        assert k in rec, f"v2 key {k!r} absent"
    assert rec["schema_version"] >= 2
    assert md.validate_scaling_metadata(rec) == []


def test_transport_explicit_wins_and_unknown_raises():
    assert _record(transport="nccl")["transport"] == "nccl"
    with pytest.raises(ValueError, match="unknown transport"):
        _record(transport="carrier-pigeon")


def test_transport_route_a_inferred_from_rank_excess():
    # mpirun -np 8 route-A: each rank is a single-process JAX
    # (process_count()==1) but the driver records n_ranks=8 — the world JAX
    # cannot see is exactly the mpi4jax signature.
    t = md.resolve_transport(
        None, n_ranks=8, process_count=1, device_count=1, backend="cpu")
    assert t == "mpi4jax"


def test_transport_route_b_nccl_on_gpu_gloo_on_cpu():
    kw = dict(n_ranks=4, process_count=4, device_count=4)
    assert md.resolve_transport(None, backend="gpu", **kw) == "nccl"
    assert md.resolve_transport(None, backend="cpu", **kw) == "gloo"


def test_transport_single_process_spmd_and_serial():
    assert md.resolve_transport(
        None, n_ranks=1, process_count=1, device_count=4,
        backend="cpu") == "xla-local"
    assert md.resolve_transport(
        None, n_ranks=1, process_count=1, device_count=1,
        backend="cpu") == "none"


def test_virtual_cpu_devices_detected_from_xla_flags(monkeypatch):
    monkeypatch.setenv(
        "XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    assert md.detect_virtual_cpu_devices(backend="cpu") is True
    # A GPU backend with the flag set is NOT a virtual-CPU proxy.
    assert md.detect_virtual_cpu_devices(backend="gpu") is False
    # count=1 is the serial default, not forced parallelism.
    monkeypatch.setenv(
        "XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    assert md.detect_virtual_cpu_devices(backend="cpu") is False
    monkeypatch.delenv("XLA_FLAGS")
    assert md.detect_virtual_cpu_devices(backend="cpu") is False


def test_launcher_detection_priority(monkeypatch):
    for var in ("SLURM_JOB_ID", "PALS_RANKID", "PALS_NODEID", "PBS_JOBID",
                "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        monkeypatch.delenv(var, raising=False)
    assert md.detect_launcher() == "none"
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert md.detect_launcher() == "openmpi"
    # PALS jobs also carry PBS_JOBID — PALS must win over plain PBS.
    monkeypatch.setenv("PBS_JOBID", "12345.desched1")
    assert md.detect_launcher() == "pbs"
    monkeypatch.setenv("PALS_RANKID", "0")
    assert md.detect_launcher() == "pbs-pals"
    # SLURM outranks all (SLURM steps may export OMPI vars via plugins).
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    assert md.detect_launcher() == "slurm"


def test_host_staged_semantics_gated_on_mpi4jax_transport(monkeypatch):
    # A route-B NCCL (or intra-process xla-local) GPU row has NO mpi4jax halo
    # to host-stage: neither gpu_direct_active nor host_staged_halo may fire
    # (codex: an NCCL row must not look like a broken host-staged MPI row).
    monkeypatch.delenv("MPI4JAX_USE_CUDA_MPI", raising=False)
    monkeypatch.setattr(md, "mpi4jax_cuda_support", lambda: False)
    for t in ("nccl", "gloo", "xla-local", "none"):
        g = md.gpu_direct_mode(backend="gpu", transport=t)
        assert g["gpu_direct_active"] is False, t
        assert g["host_staged_halo"] is False, t
    # Default transport (mpi4jax) keeps the fail-loud host-staged semantics.
    g = md.gpu_direct_mode(backend="gpu")
    assert g["host_staged_halo"] is True


def test_validate_catches_missing_v2_keys():
    for k in ("transport", "virtual_cpu_devices", "launcher"):
        rec = _record()
        del rec[k]
        assert k in md.validate_scaling_metadata(rec, strict=False)
        with pytest.raises(ValueError):
            md.validate_scaling_metadata(rec)


# ---------------------------------------------------------------------------
# tidy_throughput_fields: flat SYPD/throughput metrics for SPMD bench lanes
# ---------------------------------------------------------------------------

def test_tidy_throughput_fields_canonical_formulas():
    # 600 s of model time per 50 ms wall step: sypd = (600/0.05)/(365.25*
    # 86400)*86400 = 12000/365.25; mcells = 1e6 cells / 0.05 s / 1e6.
    out = md.tidy_throughput_fields(
        dt_seconds=600.0, time_per_step_ms=50.0, total_cells=1_000_000)
    assert out["sypd"] == pytest.approx(12000.0 / 365.25)
    assert out["mcells_per_s"] == pytest.approx(20.0)
    assert out["dt_seconds"] == 600.0
    assert out["time_per_step_ms"] == 50.0
    assert out["total_cells"] == 1_000_000


def test_tidy_throughput_fields_matches_run_cpu_mpi_formula():
    # Parity with the canonical run_cpu_mpi_scaling.py computation so SPMD
    # rows and MPI rows are directly comparable on one plot.
    dt_used, time_per_step, total_cells = 390.0, 0.123, 6 * 48 * 48 * 26
    expect_sypd = (dt_used / time_per_step) / (365.25 * 86400) * 86400.0
    expect_mcells = (total_cells / time_per_step) / 1e6
    out = md.tidy_throughput_fields(
        dt_seconds=dt_used, time_per_step_ms=time_per_step * 1e3,
        total_cells=total_cells)
    assert out["sypd"] == pytest.approx(expect_sypd, rel=1e-12)
    assert out["mcells_per_s"] == pytest.approx(expect_mcells, rel=1e-12)


def test_tidy_throughput_fields_zero_time_is_flagged_not_inf():
    out = md.tidy_throughput_fields(
        dt_seconds=600.0, time_per_step_ms=0.0, total_cells=10)
    assert out["sypd"] == 0.0
    assert out["mcells_per_s"] == 0.0


def test_tidy_throughput_fields_none_time_emits_honest_nulls():
    """A run with NO per-step time (the zero-length block_steps=0 parity
    path) must propagate nulls — a throughput is never fabricated."""
    out = md.tidy_throughput_fields(
        dt_seconds=600.0, time_per_step_ms=None, total_cells=10)
    assert out["time_per_step_ms"] is None
    assert out["sypd"] is None
    assert out["mcells_per_s"] is None
    assert out["dt_seconds"] == 600.0
    assert out["total_cells"] == 10


def test_count_collective_permutes_matches_hyphen_and_underscore():
    """Canonical CP census (#1113): counts StableHLO underscore + optimized
    hyphen spellings, and drops the async ``-done`` companion so one logical
    exchange counts once."""
    hlo = "\n".join([
        "  %a = collective-permute(%x)",          # optimized HLO
        "  %b = collective_permute(%y)",          # StableHLO
        "  %c = collective-permute-done(%a)",     # async companion -> excluded
        "  %d = collective_permute_done(%b)",     # async companion -> excluded
        "  %e = all-gather(%z)",                  # different collective
    ])
    assert md.count_collective_permutes(hlo) == 2
    assert md.count_collective_permutes("no collectives here") == 0


def _cpu_compile():
    """Pin the HLO-probe compiles to a CPU device.

    The probes read ``compile().as_text()``, and some backends return None for
    it (the experimental Apple ``mps`` plugin does) — which would make the
    "collective-free fn -> all-zero census" assertion vacuously unreachable on
    a dev laptop.  CPU is also the reproducible-by-construction reference the
    census docstring names (the DEFAULT schedule, no GPU collective-combining),
    so this keeps the assertion meaningful on every machine.  Production probes
    deliberately do NOT pin: cluster jobs must census the REAL on-device
    executable.
    """
    import jax
    return jax.default_device(jax.devices("cpu")[0])


def test_hlo_collective_permutes_lowers_counts_and_is_error_safe():
    """The best-effort probe lowers a fn and counts its collective-permutes: a
    fn with none -> 0; an unlowerable fn -> None (never raises). Real ppermute
    counting is exercised by the MPAS/cube bench gates and
    count_collective_permutes' synthetic HLO test above."""
    import jax.numpy as jnp
    with _cpu_compile():
        assert md.hlo_collective_permutes(lambda x: x + 1, jnp.arange(4.0)) == 0

        def _boom(x):
            raise RuntimeError("unlowerable")
        assert md.hlo_collective_permutes(_boom, jnp.arange(4.0)) is None


def test_count_collectives_full_census_all_families():
    """Full census counts every collective family with the same op-call-form
    discipline: async ``-start`` once (``-done`` excluded), StableHLO
    underscore + optimized hyphen, and a config-header flag echo that merely
    CONTAINS an op name never inflates the count."""
    hlo = "\n".join([
        # config-header echo of XLA_FLAGS -> must NOT match (no op-call paren)
        "  // xla_gpu_collective_permute_combine_threshold_bytes=33554432",
        "  %a = collective-permute(%x)",             # permute (optimized)
        "  %b = collective_permute(%y)",             # permute (StableHLO)
        "  %c = collective-permute-done(%a)",        # async companion -> drop
        "  %r1 = all-reduce(%p)",                    # reduction (hyphen)
        "  %r2 = all_reduce_start(%q)",              # async reduction -> count once
        "  %r3 = all-reduce-done(%r2)",              # async companion -> drop
        "  %g = all-gather(%z)",                     # all-gather
        "  %a2a = all-to-all(%w)",                   # all-to-all
        "  %rs = reduce-scatter(%v)",                # reduce-scatter
    ])
    c = md.count_collectives(hlo)
    assert c["collective_permute"] == 2
    assert c["all_reduce"] == 2          # hyphen op + async-start (done dropped)
    assert c["all_gather"] == 1
    assert c["all_to_all"] == 1
    assert c["reduce_scatter"] == 1
    assert c["total"] == 7
    # permute family stays bit-identical to the canonical scalar helper
    assert c["collective_permute"] == md.count_collective_permutes(hlo)

    empty = md.count_collectives("no collectives here")
    assert empty["total"] == 0
    assert set(empty) == {"collective_permute", "all_reduce", "all_gather",
                          "all_to_all", "reduce_scatter", "total"}


def test_census_does_not_false_drop_ops_with_done_in_metadata():
    """Regression (codex): the ``-done`` async COMPANION is excluded by the
    regex structurally (``op-done(`` never matches ``op(?:[_-]start)?\\(``), so
    a line that merely CONTAINS the substring "done" elsewhere — an XLA
    metadata op_name, a ``%done_*`` SSA name — must STILL be counted.  A blunt
    ``"done" not in line`` filter would false-drop these to zero."""
    hlo = "\n".join([
        '  %r = all-reduce(%p), metadata={op_name="jit(step)/done_stage/psum"}',
        '  %done_mass = f32[] collective-permute(%q)',
        '  %g = all-gather(%z), metadata={op_name="reduce_done/x"}',
    ])
    c = md.count_collectives(hlo)
    assert c["all_reduce"] == 1        # NOT dropped despite "done" in metadata
    assert c["collective_permute"] == 1  # NOT dropped despite %done_ SSA name
    assert c["all_gather"] == 1
    # canonical permute helper is fixed by the same shared counter
    assert md.count_collective_permutes(hlo) == 1


def test_hlo_collective_census_lowers_and_is_error_safe():
    """Best-effort full-census probe: a collective-free fn -> all-zero dict;
    an unlowerable fn -> None (never raises)."""
    import jax.numpy as jnp
    with _cpu_compile():
        census = md.hlo_collective_census(lambda x: x + 1, jnp.arange(4.0))
        assert census is not None and census["total"] == 0

        def _boom(x):
            raise RuntimeError("unlowerable")
        assert md.hlo_collective_census(_boom, jnp.arange(4.0)) is None
