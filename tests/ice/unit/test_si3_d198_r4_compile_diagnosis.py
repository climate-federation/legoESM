"""Unit and control tests for the DECISION 198 round 4 O17 reverse-compile diagnosis probe."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest
from legoesm.core.precision import PrecisionPolicy, set_policy

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import (
    DEFAULT_MASTER,
    RUNG_SPECS,
    _compile_with_optional_xla_log,
    _count_hlo_text,
    _preconfigure_xla_logging,
    diagnose_slices,
    parse_xla_pass_log,
)


@pytest.fixture(autouse=True)
def _fp64_libm_policy() -> None:
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))


def test_d198_r4_probe_prefix_1_2_metrics() -> None:
    """Run the committed diagnostic probe on prefixes k=1,2 and verify all stamped metrics."""
    if not DEFAULT_MASTER.exists():
        pytest.skip(f"master directory unavailable: {DEFAULT_MASTER}")

    report = diagnose_slices(DEFAULT_MASTER, prefixes=(1, 2), execute=True)
    assert report["status"] == "PASS"
    assert report["mode"] == "diagnostic"
    assert len(report["rung_registry"]) == len(RUNG_SPECS) == 14

    prov = report["provenance"]
    assert isinstance(prov["git_sha"], str) and len(prov["git_sha"]) >= 10
    assert prov["default_backend"] == "cpu"
    assert prov["jax_enable_x64"] is True
    assert isinstance(prov["argv"], list)
    assert isinstance(prov["env"], dict)

    results = report["results"]
    assert len(results) == 2
    r1, r2 = results[0], results[1]

    assert r1["label"] == "prefix_1_1_o7_blk1_strength"
    assert r1["k_start"] == 1 and r1["k_end"] == 1
    assert r1["rungs"] == ["o7_blk1_strength"]
    assert r1["fwd_jaxpr_top_eqns"] == 670
    assert r1["fwd_jaxpr_rec_eqns"] == 1345
    assert r1["vg_jaxpr_top_eqns"] == 1753
    assert r1["vg_jaxpr_rec_eqns"] == 3646

    assert r2["label"] == "prefix_1_2_o5_aevp"
    assert r2["k_start"] == 1 and r2["k_end"] == 2
    assert r2["rungs"] == ["o7_blk1_strength", "o5_aevp"]
    assert r2["fwd_jaxpr_top_eqns"] == 2178
    assert r2["fwd_jaxpr_rec_eqns"] == 11431
    assert r2["vg_jaxpr_top_eqns"] == 6569
    assert r2["vg_jaxpr_rec_eqns"] == 33122

    for r in (r1, r2):
        assert r["lower_seconds"] > 0.0
        assert r["lower_total_seconds"] >= r["lower_seconds"]
        assert r["compile_seconds"] > 0.0
        assert r["execution_seconds"] is not None and r["execution_seconds"] > 0.0
        assert r["objective"] is not None
        assert r["peak_rss_kib"] > 0
        assert r["hlo_pre_opt_instructions"] > 0
        assert r["hlo_post_opt_instructions"] > 0

    assert r2["hlo_pre_opt_instructions"] > r1["hlo_pre_opt_instructions"]
    assert r2["hlo_post_opt_instructions"] > r1["hlo_post_opt_instructions"]
    assert r2["compile_seconds"] > r1["compile_seconds"]


def test_d198_r4_controls_tmp_guard_and_parsers(tmp_path: Path) -> None:
    """Verify planted violations fire on /tmp paths, invalid slice bounds, and XLA/HLO parsers."""
    with pytest.raises(ValueError, match="must not be under /tmp"):
        _preconfigure_xla_logging(["--xla-pass-log", "/tmp/forbidden_xla.log"])

    with pytest.raises(ValueError, match="must not be under /tmp"):
        _compile_with_optional_xla_log(None, Path("/tmp/forbidden_xla.log"))

    if DEFAULT_MASTER.exists():
        with pytest.raises(ValueError, match="out of range"):
            diagnose_slices(DEFAULT_MASTER, prefixes=(0,))
        with pytest.raises(ValueError, match="out of range"):
            diagnose_slices(DEFAULT_MASTER, prefixes=(15,))
        with pytest.raises(ValueError, match="out of range"):
            diagnose_slices(DEFAULT_MASTER, slices=((5, 3),))

    hlo_sample = """HloModule test_mod

%sub_comp (p0: f64[]) -> f64[] {
  %p0 = f64[] parameter(0)
  ROOT %neg = f64[] negate(%p0)
}

ENTRY %main (a: f64[]) -> f64[] {
  %a = f64[] parameter(0)
  %b = f64[] call(%a), to_apply=%sub_comp
  ROOT %c = f64[] add(%a, %b)
}
"""
    counts_base = _count_hlo_text(hlo_sample)
    assert counts_base == {
        "instructions_total": 5,
        "instructions_entry": 3,
        "computations": 2,
    }
    # Planted extra instruction in ENTRY must increase both total and entry counts.
    hlo_planted = hlo_sample.replace(
        "  ROOT %c = f64[] add(%a, %b)",
        "  %d = f64[] multiply(%a, %b)\n  ROOT %c = f64[] add(%d, %b)",
    )
    counts_planted = _count_hlo_text(hlo_planted)
    assert counts_planted["instructions_total"] == counts_base["instructions_total"] + 1
    assert counts_planted["instructions_entry"] == counts_base["instructions_entry"] + 1

    vlog_lines = [
        "I1103 08:00:00.000000 100 hlo_pass_pipeline.cc:296] Running HLO pass pipeline on module jit_objective: simplification",
        "I1103 08:00:01.000000 100 hlo_pass_pipeline.cc:174]   HLO pass fusion",
        "I1103 08:00:11.000000 100 hlo_pass_pipeline.cc:169] HLO pass: fusion time: 10.0 s (10000000 us)",
        "I1103 08:00:11.000000 100 hlo_pass_pipeline.cc:174]   HLO pass copy-insertion",
        "I1103 08:00:15.000000 100 hlo_pass_pipeline.cc:169] HLO pass: copy-insertion time: 4.0 s (4000000 us)",
        "I1103 08:00:15.000000 100 hlo_pass_pipeline.cc:169] HLO pass: simplification time: 14.0 s (14000000 us)",
        "I1103 08:00:15.000000 100 cpu_compiler.cc:2102] Compiling: jit_objective",
        "I1103 08:00:15.500000 100 buffer_assignment.cc:100] Assigning buffers to module jit_objective",
        "I1103 08:00:16.000000 100 buffer_assignment.cc:120] Number of buffers to assign: 12345",
        "I1103 08:00:17.500000 100 buffer_assignment.cc:140] total allocation: 8.50MiB",
        "I1103 08:00:17.500000 100 buffer_assignment.cc:150] Buffer assignment done.",
        "I1103 08:00:19.000000 100 cpu_compiler.cc:1740] Compile LLVM module with 4 parts",
        "I1103 08:00:22.000000 101 ir_compiler.cc:200] Compiled LLVM module: __compute_module_fusion_1 time: 2.5 s (2500000 us)",
        "I1103 08:00:23.000000 100 cpu_compiler.cc:1780] Compilation finished",
        "I1103 08:00:23.000000 100 cpu_compiler.cc:2099] Compiled module: jit_objective for CPU using JIT time: 8.0 s (8000000 us)",
    ]
    log_file = tmp_path / "sample_xla.log"
    log_file.write_text("\n".join(vlog_lines) + "\n")
    parsed = parse_xla_pass_log(log_file)
    assert parsed["module"] == "jit_objective"
    assert parsed["hlo_passes_wall_s"] == 15.0
    assert parsed["hlo_passes_sum_s"] == 14.0
    assert parsed["buffer_assignment_s"] == 2.0
    assert parsed["buffers_to_assign"] == 12345
    assert parsed["total_allocation"] == "8.50MiB"
    assert parsed["llvm_ir_emit_s"] == 1.5
    assert parsed["llvm_codegen_wall_s"] == 4.0
    assert parsed["backend_jit_total_s"] == 8.0
    assert parsed["top_hlo_passes"][0] == {
        "pass": "fusion",
        "total_s": 10.0,
        "max_s": 10.0,
        "calls": 1,
    }

    # Planted perturbation in fusion duration must change the parsed top pass time.
    planted_lines = [
        line.replace("(10000000 us)", "(95000000 us)") for line in vlog_lines
    ]
    log_file.write_text("\n".join(planted_lines) + "\n")
    parsed_planted = parse_xla_pass_log(log_file)
    assert parsed_planted["top_hlo_passes"][0]["total_s"] == 95.0
    assert parsed_planted["top_hlo_passes"][0]["total_s"] != parsed["top_hlo_passes"][0]["total_s"]
