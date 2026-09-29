"""Guard the parity/conservation gates of ``bench_mpas_spmd_scaling``.

The Voronoi sibling of tests/bench/test_bench_ocean_latlon_spmd_gates.py —
the gates are the fail-fast correctness armor the Derecho/Levante
icosahedral GPU jobs run before any timed ladder:

* wiring tripwire — the gate flags/constants exist (a rename would only
  break at runtime on a cluster);
* a direct shard->gather round-trip of ``gather_voronoi_state_spmd`` (the
  new leaf symbol) over 2 virtual devices;
* an end-to-end single-process 2-virtual-device CPU run with BOTH gates
  armed exits 0, writes the JSONL record, and prints per-field parity lines;
* the smoke-window cap refuses long runs (the re-association floor grows
  with steps).

Multi-controller federation of the same bench:
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_BENCH = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "bench" / "bench_mpas_spmd_scaling.py"
)


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench_mpas_spmd_scaling", _BENCH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_gate_symbols_and_flags_exist():
    mod = _load_bench()
    assert set(mod.MPAS_PARITY_TOLS) == {"float32", "float64"}
    for per_field in mod.MPAS_PARITY_TOLS.values():
        assert set(per_field) == {"u", "T", "p_s"}
        for rtol, atol in per_field.values():
            assert rtol > 0 and atol > 0
    assert mod.MPAS_PARITY_MAX_STEPS >= 4
    assert set(mod.MASS_RTOL_DEFAULTS) == {"float32", "float64"}
    # Parser accepts the gate flags (argparse would SystemExit on unknowns).
    src = Path(_BENCH).read_text()
    for flag in ("--parity-gate", "--check-conservation", "--mass-rtol",
                 "--multicontroller", "--coordinator", "--partition-method",
                 "--reorder-for", "--lloyd"):
        assert flag in src


def test_lloyd_flag_reaches_the_mesh_builder(monkeypatch):
    """``--lloyd 0`` must select the synthetic scaling mesh, not lloyd=50.

    Non-vacuous by construction: the sentinel records the kwarg
    ``create_voronoi_mesh`` actually receives, so dropping the plumbing
    (the state before this flag existed) makes the assertion fail rather
    than silently building/loading the production SCVT cache key.
    """
    import legoesm.grids.voronoi as voronoi

    mod = _load_bench()
    seen = {}

    def _spy(subdivision_level, **kwargs):
        seen["level"] = subdivision_level
        seen.update(kwargs)
        raise RuntimeError("stop-after-mesh-request")

    monkeypatch.setattr(voronoi, "create_voronoi_mesh", _spy)
    with pytest.raises(RuntimeError, match="stop-after-mesh-request"):
        mod.build_model_and_state(4, 4, 1, 1, "sfc", dt=600.0, lloyd_iterations=0)
    assert seen["level"] == 4
    assert seen["lloyd_iterations"] == 0



def test_del4_coeff_is_the_shared_law_capped_by_dt():
    sys.path.insert(0, str(_BENCH.parent))
    from legoesm import constants
    mod = _load_bench()
    assert mod.DEL4_S_MAX == 6e-4   # user-approved margin, 2026-09-27
    assert mod.hyperdiff_coeff(7, "icosahedral") == 1.0e16 / 16.0 ** 3   # shared law

    def s_num(level, dt):
        dx = constants.R_earth * (4 * 3.141592653589793 / (10 * 4 ** level + 2)) ** 0.5
        return mod.del4_coeff(level, dt) * dt / dx ** 4

    # On the ladder's own timesteps the law binds (s_num stays under the cap)...
    for level, dt in ((4, 600.0), (7, 30.0), (8, 5.0)):
        assert mod.del4_coeff(level, dt) == mod.hyperdiff_coeff(level, "icosahedral")
        assert s_num(level, dt) < mod.DEL4_S_MAX
    # ...and a long step hits the cap, which then scales as 1/dt.
    assert s_num(4, 6000.0) == pytest.approx(mod.DEL4_S_MAX, rel=1e-12)
    assert mod.del4_coeff(4, 3000.0) == pytest.approx(2 * mod.del4_coeff(4, 6000.0))


def test_builder_passes_the_capped_coefficient(monkeypatch):
    """The spy records what ``build_model_and_state`` hands the model config,
    so reverting to a literal (the fixed 1e16 that sent s7 non-finite by step
    4) fails here."""
    import legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas as pe
    import legoesm.grids.voronoi as voronoi

    mod = _load_bench()
    real = voronoi.create_voronoi_mesh
    monkeypatch.setattr(voronoi, "create_voronoi_mesh",
                        lambda subdivision_level, **kw: real(2, lloyd_iterations=0))
    seen = {}

    def _spy(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop-at-config")

    monkeypatch.setattr(pe, "MPASPrimitiveEquationConfig", _spy)
    with pytest.raises(RuntimeError, match="stop-at-config"):
        mod.build_model_and_state(7, 4, 1, 1, "sfc", dt=30.0, lloyd_iterations=0)
    want = mod.del4_coeff(7, 30.0)
    assert seen["nu_del4"] == seen["nu_del4_ps"] == want == 1.0e16 / 16.0 ** 3
    # ...and a long step at level 4 where the CAP binds: an uncapped builder
    # would hand the model the bare law (1e16) here.
    seen.clear()
    with pytest.raises(RuntimeError, match="stop-at-config"):
        mod.build_model_and_state(4, 4, 1, 1, "sfc", dt=6000.0, lloyd_iterations=0)
    assert (seen["nu_del4"] == seen["nu_del4_ps"] == mod.del4_coeff(4, 6000.0)
            < mod.hyperdiff_coeff(4, "icosahedral"))




def test_gather_voronoi_state_spmd_round_trip():
    """Direct exercise of the new gather: shard -> gather == original."""
    if len(__import__("jax").devices()) < 2:
        pytest.skip("needs 2 devices "
                    "(set XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    import jax.numpy as jnp
    import numpy as np
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.mesh import create_voronoi_device_mesh, shard_pytree
    from legoesm.parallel.sharded_dynamics import gather_voronoi_state_spmd
    from legoesm.parallel.voronoi_partition import (
        reorder_voronoi_for_sharding,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    mesh = reorder_voronoi_for_sharding(
        create_voronoi_mesh(subdivision_level=3), 2, method="sfc")
    state = baroclinic_wave_init_mpas(
        mesh, create_sigma_coordinate(4), perturbed=True)
    dev = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
        n_devices=2)
    sharded = shard_pytree(state, dev)
    back = gather_voronoi_state_spmd(sharded, dev)
    for name in ("u", "T", "p_s", "phis"):
        np.testing.assert_array_equal(
            np.asarray(getattr(back, name).data),
            np.asarray(getattr(state, name).data),
            err_msg=f"{name}: shard->gather round-trip not identity")
    # Single-device config passes through unchanged.
    dev1 = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
        n_devices=1)
    assert gather_voronoi_state_spmd(state, dev1) is state
    assert isinstance(jnp.asarray(0.0), object)  # keep jnp import honest


@pytest.mark.timeout(600)
def test_single_process_two_virtual_devices_with_gates(tmp_path):
    out = tmp_path / "mpas_spmd.jsonl"
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    proc = subprocess.run(
        [
            sys.executable, str(_BENCH),
            "--n-devices", "2",
            "--subdivision", "3", "--nlev", "4",
            "--steps", "4", "--warmup", "1",
            "--partition-method", "sfc",
            "--parity-gate", "--check-conservation",
            "--out", str(out),
        ],
        env=env, capture_output=True, text=True, timeout=570,
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}"
    )
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert rec["component"] == "mpas_atm"
    assert rec["n_devices"] == 2
    assert rec["finite_ok"] is True and rec["valid"] is True
    assert "parity" in proc.stdout and "MISMATCH" not in proc.stdout
    # #1113 ask 2: the ppermute round count is now recorded per row.
    assert "hlo_collective_permutes" in rec
    assert rec["hlo_collective_permutes"] is None or isinstance(
        rec["hlo_collective_permutes"], int)


@pytest.mark.timeout(600)
def test_single_device_reference_leg_with_reorder_for(tmp_path):
    """The ladder's nd=1 reference: mesh partitioned/ghost-padded for the
    ladder target (--reorder-for 4 pads 642 -> 644 cells) but run on ONE
    device — the codex-r1 stale-model/dev-config hazard regression guard."""
    out = tmp_path / "ref.jsonl"
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    proc = subprocess.run(
        [
            sys.executable, str(_BENCH),
            "--n-devices", "1", "--reorder-for", "4",
            "--subdivision", "3", "--nlev", "4",
            "--steps", "2", "--warmup", "0",
            "--partition-method", "sfc",
            "--out", str(out),
        ],
        env=env, capture_output=True, text=True, timeout=570,
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}"
    )
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert rec["n_devices"] == 1
    assert rec["n_cells"] == 644  # 10*4^3+2 = 642, ghost-padded to %4 == 0


def test_parity_gate_refuses_long_windows(tmp_path):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    proc = subprocess.run(
        [
            sys.executable, str(_BENCH),
            "--n-devices", "2", "--subdivision", "3", "--nlev", "4",
            "--steps", "30", "--warmup", "2",
            "--parity-gate", "--out", str(tmp_path / "x.jsonl"),
        ],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "smoke gate" in (proc.stdout + proc.stderr)
