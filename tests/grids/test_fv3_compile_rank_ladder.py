"""Direct unit tests for the FV3 compile rank ladder and launcher wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "validate" / "fv3_native" / "compile_rank_ladder.py"
_spec = importlib.util.spec_from_file_location("compile_rank_ladder", _SCRIPT)
ladder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ladder)
_GATE_SCRIPT = (_ROOT / "scripts" / "validate" / "fv3_native"
                / "tiled_m6_model_gate.py")
_gate_spec = importlib.util.spec_from_file_location("tiled_m6_model_gate",
                                                    _GATE_SCRIPT)
gate = importlib.util.module_from_spec(_gate_spec)
_gate_spec.loader.exec_module(gate)


@pytest.mark.parametrize(
    ("devices", "expected"),
    [(1, (1, "collapsed")),
     (6, (1, "faces-6x1x1")),
     (24, (2, "windows-6x2x2"))],
)
def test_partition_shapes_cover_the_preregistered_ladder(devices, expected):
    assert ladder.partition_for_devices(devices) == expected


def test_unregistered_partition_shape_is_refused():
    with pytest.raises(ValueError, match="must be one of"):
        ladder.partition_for_devices(12)


def test_lower_and_compile_are_timed_separately():
    calls = []

    class Compiled:
        pass

    class Lowered:
        def compile(self):
            calls.append("compile")
            return Compiled()

    class Jitted:
        def lower(self, *operands):
            calls.append(("lower", operands))
            return Lowered()

    ticks = iter((10.0, 12.5, 20.0, 27.0))
    events = []
    lower_s, compile_s, compiled = ladder.measure_lower_compile(
        Jitted(), ("state", 900.0), clock=lambda: next(ticks),
        report=lambda *event: events.append(event),
    )

    assert lower_s == 2.5
    assert compile_s == 7.0
    assert isinstance(compiled, Compiled)
    assert calls == [("lower", ("state", 900.0)), "compile"]
    assert events == [
        ("lower", "begin", None),
        ("lower", "complete", 2.5),
        ("compile", "begin", None),
        ("compile", "complete", 7.0),
    ]


def test_hlo_instruction_count_counts_assignments_not_headers():
    hlo = """HloModule jit_step

ENTRY %main.4 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %one = f32[] constant(1)
  ROOT %out = f32[] add(%x, %one)
}
"""
    assert ladder.count_hlo_instructions(hlo) == 3


def test_fake_device_flag_is_replaced_not_duplicated():
    got = ladder._replace_fake_device_flag(
        "--xla_cpu_max_isa=AVX --xla_force_host_platform_device_count=6", 24)
    assert got == "--xla_cpu_max_isa=AVX --xla_force_host_platform_device_count=24"


def test_cpu_driver_launches_the_full_fake_device_ladder(monkeypatch):
    calls = []

    def run(cmd, *, env, check):
        calls.append((cmd, env, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ladder.subprocess, "run", run)
    assert ladder.main(["--backend", "cpu"]) == 0
    assert [call[0][call[0].index("--devices") + 1] for call in calls] == [
        "1", "6", "24"]
    assert [call[1]["XLA_FLAGS"].split("=")[-1] for call in calls] == [
        "1", "6", "24"]
    assert all(call[1]["JAX_PLATFORMS"] == "cpu" for call in calls)
    assert all(call[1]["JAX_LOG_COMPILES"] == "1" for call in calls)


@pytest.mark.parametrize("name", [
    "tiled_m6_model_gate_mpi.sbatch",
    "tiled_m6_model_gate_gpu.sbatch",
])
def test_cpu_and_gpu_launchers_enable_compile_logs(name):
    launcher = _ROOT / "scripts" / "cluster" / "fv3_native" / name
    assert "JAX_LOG_COMPILES=1" in launcher.read_text()


def test_gpu_launcher_does_not_claim_cache_state_in_prose():
    launcher = (_ROOT / "scripts" / "cluster" / "fv3_native"
                / "tiled_m6_model_gate_gpu.sbatch")
    comments = "\n".join(
        line for line in launcher.read_text().splitlines()
        if line.lstrip().startswith("#") and not line.startswith("#!"))
    assert "cache" not in comments.lower()


def test_python_frame_sampler_is_not_present_in_gate():
    source = _GATE_SCRIPT.read_text()
    assert "faulthandler" not in source
    assert "SIGUSR1" not in source


def test_gate_marks_each_compile_and_first_dispatch_boundary():
    source = _GATE_SCRIPT.read_text()
    for phase, event in (
        ("lower", "begin"), ("lower", "complete"),
        ("compile", "begin"), ("compile", "complete"),
        ("first_dispatch", "begin"), ("first_dispatch", "submitted"),
        ("first_dispatch", "complete"),
    ):
        assert f'"{phase}", "{event}"' in source


def test_runtime_banner_prints_resolved_settings(monkeypatch, capsys):
    class Device:
        platform = "cpu"

        def __str__(self):
            return "cpu:0"

    class Config:
        jax_compilation_cache_dir = None

    class Jax:
        __version__ = "test-jax"
        config = Config()

        @staticmethod
        def local_devices():
            return [Device()]

        @staticmethod
        def process_index():
            return 7

    monkeypatch.setattr(gate.os, "sched_getaffinity", lambda pid: {2, 3})
    monkeypatch.setenv("XLA_FLAGS", "--test-flag")
    gate._runtime_banner(Jax())
    line = capsys.readouterr().out
    for expected in (
        "rank=7", "compile_cache=none", "XLA_FLAGS='--test-flag'",
        "jax=test-jax", "jaxlib=", "nccl=n/a", "visible_devices=['cpu:0']",
        "cpu_affinity=[2, 3]",
    ):
        assert expected in line


def test_gate_nq_flag_defaults_to_one_and_reaches_every_ic_site():
    """--nq (2026-09-13): default 1 keeps every existing reference npz
    valid; and EVERY dcmip16_initial_state call in the gate -- the flat
    reference, the in-process reference and the window arm -- must carry
    n_tracers=args.nq, or one arm would silently run a different tracer
    count from the one it is scored against."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(gate))
    defaults = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None)
                == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults[node.args[0].value] = kw.value.value
    assert defaults["--nq"] == 1
    sites = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "attr", None) == "dcmip16_initial_state"]
    assert len(sites) >= 3, "expected three IC construction sites"
    assert defaults.get("--terminator") in (None, False)
    for call in sites:
        kws = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert kws.get("n_tracers") == "args.nq", ast.unparse(call)
        assert kws.get("terminator") == "args.terminator", ast.unparse(call)


def test_gate_main_does_not_shadow_the_time_module():
    """Job 9751309: a function-scope `import time` inside main's flat-ref
    branch made `time` a LOCAL of the whole function, so the phase
    instrumentation's `time.perf_counter()` on every OTHER path raised
    UnboundLocalError right after the IC check -- the gate was broken for
    every non-flat-ref run and its source-inspection test could not see
    it. Python decides locals statically, so this is checkable without
    running the gate: `time` must not be a local variable of main."""
    assert "time" not in gate.main.__code__.co_varnames


def test_tracer_window_change_reports_only_window_motion_per_tracer():
    """The gate refuses a tracer whose compute window did not change.
    Halo-only motion must read as ZERO (a halo fill is not transport), and
    every ['q'] leaf is reported, sorted, while non-tracer leaves are not."""
    import numpy as np
    n, ng, km = 4, 2, 3
    m = n + 2 * ng
    cs = slice(ng, ng + n)
    q0 = np.zeros((6, m, m, km))
    q1 = np.ones((6, m, m, km))
    moved0 = q0.copy(); moved0[:, cs, cs, :] = 1e-9         # window motion
    halo1 = q1.copy(); halo1[:, 0, :, :] += 5.0             # halo only
    flat = {"['q'][1]": halo1, "['q'][0]": moved0,
            "['state']['pt']": np.zeros((6, m, m, km))}
    prev = {"['q'][1]": q1, "['q'][0]": q0,
            "['state']['pt']": np.ones((6, m, m, km))}
    got = gate.tracer_window_change(flat, prev, n, ng)
    assert [p for p, _ in got] == ["['q'][0]", "['q'][1]"]
    assert got[0][1] == 1e-9
    assert got[1][1] == 0.0


def test_ref_npz_without_distributed_is_refused_before_any_work():
    """codex 2026-09-13: an in-process run built its own reference and
    silently ignored --ref-npz, a no-op dressed as a gate."""
    with pytest.raises(SystemExit) as e:
        gate.main(["--ref-npz", "does-not-need-to-exist.npz"])
    assert e.value.code == 2


def test_gpu_peak_ceiling_is_refused_outside_rounding_and_wired_in_the_launcher():
    """User call 2026-09-15: GPU rows pass within a peak-relative ceiling;
    the flag refuses anything that is not rounding-level, and the GPU
    launcher passes it while the CPU launcher does not."""
    from pathlib import Path
    with pytest.raises(SystemExit) as e:
        gate.main(["--distributed", "--ref-npz", "x.npz", "--gpu-max-rel-peak", "1e-3"])
    assert e.value.code == 2
    root = Path(__file__).resolve().parents[2] / "scripts" / "cluster" / "fv3_native"
    gpu = (root / "tiled_m6_model_gate_gpu.sbatch").read_text()
    cpu = (root / "tiled_m6_model_gate.sbatch").read_text()
    assert "--gpu-max-rel-peak" in gpu and 'GPU_MAX_REL_PEAK:-1e-13' in gpu
    assert "--gpu-max-rel-peak" not in cpu


def test_timed_steps_are_checked_for_divergence_and_agree_across_ranks():
    """codex 2026-09-16: the scored steps refuse a non-finite state, the
    timed ones did not -- a deck that blew up after the last scored step
    still printed a p50. The check must also be agreed across ranks before
    any rank leaves the loop, or they desynchronise on the next collective."""
    import ast
    import inspect
    src = inspect.getsource(gate)
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    # the WINDOW arm's timed loop: `for it in range(args.timing)`, not the
    # flat-reference loop `range(args.steps + args.timing)`
    loops = [n for n in ast.walk(fn)
             if isinstance(n, ast.For)
             and ast.unparse(n.iter).strip() == "range(args.timing)"]
    assert loops, "no window timed-step loop found"
    body = ast.unparse(loops[0])
    assert "isfinite" in body, "timed steps are not checked for divergence"
    assert "process_allgather" in body, "the check is not agreed across ranks"
    assert "break" in body and "timing_ok" in body
    # and the report is skipped when it was refused
    assert "if timing_ok:" in src
