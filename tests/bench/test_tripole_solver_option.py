"""Tripole solver option on the scaling ladder and the bench's measured-table advisory."""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench_ocean_latlon_spmd_scaling_opt",
        ROOT / "scripts/bench/bench_ocean_latlon_spmd_scaling.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()
GRID = dict(tripole=True, mode="strong", n_lat=3072, n_lon=4352, nlev=75,
            A_h_from_mesh=True)
JAC = dict(variant="standard", precond="jacobi", iters=200)
ZL = dict(variant="single_reduce", precond="zonal_line", iters=80)


def advise(n, prec, solver, **over):
    return bench.tripole_solver_advisory(**{**GRID, **over}, n_devices=n,
                                         precision=prec, **solver)


@pytest.mark.parametrize("n,prec", [(32, "float32"), (64, "float32"),
                                    (128, "float32"), (128, "float64")])
def test_suggests_zonal_line_where_measured_faster(n, prec):
    msg = advise(n, prec, JAC)
    assert msg and "zline80sr" in msg and "less step time" in msg


@pytest.mark.parametrize("n,prec", [(8, "float32"), (16, "float32"),
                                    (8, "float64"), (64, "float64")])
def test_silent_where_jacobi_not_slower(n, prec):
    assert advise(n, prec, JAC) is None


@pytest.mark.parametrize("n,prec", [(8, "float64"), (16, "float64"),
                                    (32, "float64"), (8, "float32")])
def test_suggests_jacobi_where_zonal_line_slower(n, prec):
    msg = advise(n, prec, ZL)
    assert msg and "jacobi200" in msg and "less step time" in msg


def test_silent_where_zonal_line_already_best():
    assert advise(128, "float32", ZL) is None
    assert advise(16, "float32", ZL) is None


@pytest.mark.parametrize("over", [dict(), dict(n_lat=1536), dict(nlev=40),
                                  dict(mode="weak"), dict(A_h_from_mesh=False),
                                  dict(backend="cpu")])
def test_unmeasured_outside_table(over):
    n = 96 if not over else 128
    assert "unmeasured" in advise(n, "float32", JAC, **over)


def test_other_solvers_get_no_advisory():
    assert advise(128, "float32", dict(variant="standard", precond="jacobi",
                                       iters=60)) is None
    assert bench.tripole_solver_advisory(**{**GRID, "tripole": False},
                                         n_devices=128, precision="float32",
                                         **JAC) is None


def _lane_args(value):
    text = (ROOT / "scripts/cluster/scaling_levante/nature_ladder.sbatch").read_text()
    block = re.search(r'case "\$\{TRIPOLE_SOLVER:-jacobi200\}" in.*?esac', text, re.S)
    assert block, "tripole solver switch missing from the ladder lane"
    env = "unset TRIPOLE_SOLVER; " if value is None else f"export TRIPOLE_SOLVER={value}; "
    script = f'args=(); {env}f() {{ {block.group(0)}\n }}; f && printf "%s\\n" "${{args[@]}}"'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_lane_default_is_jacobi200():
    out = _lane_args(None)
    assert out.returncode == 0
    assert out.stdout.split() == ["--tripole", "--nlev", "75", "--A-h-from-mesh",
                                  "--pcg-fixed-iters", "200"]


def test_lane_zline80sr_selects_the_measured_combination():
    a = _lane_args("zline80sr").stdout.split()
    assert a[a.index("--pcg-fixed-iters") + 1] == "80"
    assert a[a.index("--pcg-precond") + 1] == "zonal_line"
    assert a[a.index("--pcg-variant") + 1] == "single_reduce"


def test_lane_tags_non_default_solver():
    text = (ROOT / "scripts/cluster/scaling_levante/nature_ladder.sbatch").read_text()
    assert '${TRIPOLE_SOLVER:-jacobi200}' in text.split('local tag=')[1].split('\n')[1]


def test_lane_rejects_unknown_solver():
    out = _lane_args("cheb")
    assert out.returncode != 0 and "must be jacobi200 or zline80sr" in out.stderr
