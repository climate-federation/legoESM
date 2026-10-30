"""Column sharding of the WB training step: CLI, placement, radiation constraint.

The multi-device checks run in a subprocess with four forced CPU devices,
because the device count is fixed at JAX's first import.
"""
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "scripts" / "run" / "train_weatherbench_scale.py"
_spec = importlib.util.spec_from_file_location("train_wb_scale_col", _ENTRY)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_cli_column_devices():
    assert mod.build_scale_config_from_args([]).column_devices == 0
    assert mod.build_scale_config_from_args(
        ["--column-devices", "16"]).column_devices == 16
    with pytest.raises(SystemExit):
        mod.build_scale_config_from_args(["--column-devices", "-1"])


def _run4(body: str) -> str:
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4",
               JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    code = textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("m", {str(_ENTRY)!r})
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    """) + textwrap.dedent(body)
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def test_placement_and_replicated_gradient():
    out = _run4("""
        import jax, jax.numpy as jnp, numpy as np
        from jax.sharding import PartitionSpec as P
        from legoesm.parallel.column_shard import create_column_mesh
        mesh = create_column_mesh(4)
        n_lat, n_lon = 8, 4
        place_sample, place_rep, in_trace_rep = mod._column_placers(
            mesh, n_lat, n_lat * n_lon)
        s = place_sample({"lat": np.ones((n_lat, 3)), "col": np.ones((n_lat * n_lon,)),
                          "spec": np.ones((5, n_lat)), "t": 2.0})
        print("LAT", s["lat"].sharding.spec == P("col", None))
        print("COL", s["col"].sharding.spec == P("col"))
        print("SPEC", s["spec"].sharding.is_fully_replicated)
        print("SCALAR", isinstance(s["t"], jax.Array))
        w = place_rep(np.arange(3.0))
        print("W", w.sharding.is_fully_replicated)

        # A per-latitude parameter: without the in-trace constraint its
        # gradient inherits the samples' latitude split.
        wl = place_rep(np.ones(n_lat))

        def loss(w, x, rep):
            if rep:
                w = in_trace_rep(w)
            return jnp.sum(x["lat"][:, 0] * w ** 2)
        g = jax.jit(jax.grad(loss), static_argnums=2)(wl, s, True)
        g0 = jax.jit(jax.grad(loss), static_argnums=2)(wl, s, False)
        print("G", g.sharding.is_fully_replicated,
              bool(np.allclose(np.asarray(g), 2.0)))
        print("G0", g0.sharding.is_fully_replicated)
    """)
    for tag in ("LAT True", "COL True", "SPEC True", "SCALAR True", "W True",
                "G True True", "G0 False"):
        assert tag in out, out


def test_radiation_column_constraint():
    out = _run4("""
        import jax, jax.numpy as jnp
        from legoesm.parallel.column_shard import create_column_mesh
        from legoesm.atmosphere.physics.radiation import integration as ri
        mesh = create_column_mesh(4)
        cons = ri._column_constraint(mesh, 8)
        f = jax.jit(lambda a, b: (cons(a) * 2.0, cons(b)))
        a, b = f(jnp.ones((8, 3)), jnp.ones((5,)))
        print("A", a.sharding.spec[0] == "col")
        print("B", b.sharding.is_fully_replicated)
        try:
            ri._column_constraint(mesh, 6)
            print("RAISE False")
        except ValueError:
            print("RAISE True")
    """)
    for tag in ("A True", "B True", "RAISE True"):
        assert tag in out, out


def test_loop_and_probe_use_the_given_placement():
    import numpy as np

    from legoesm.training.data_parallel import mpi_data_parallel_training_loop
    from legoesm.training.inert_params import measure_leaf_reachability
    import jax.numpy as jnp
    import optax

    seen = []

    def place(s):
        seen.append(True)
        return jnp.asarray(s)

    params = {"w": jnp.asarray(1.0)}
    opt = optax.sgd(0.1)
    mpi_data_parallel_training_loop(
        lambda p, s: jnp.sum(p["w"] * s), params, opt.init(params), opt,
        [np.ones(2), np.ones(2)], 1, 1, place=place)
    assert len(seen) == 2
    seen.clear()
    import jax
    measure_leaf_reachability(
        jax.jit(jax.value_and_grad(lambda a, s: jnp.sum(a["w"] * s))),
        params, [np.ones(2)], n_probe=1, device_put=place)
    assert len(seen) == 1


def test_distributed_init_precedes_every_backend_import():
    """jax.distributed.initialize must run before legoesm/equinox are imported
    (either can initialise XLA; job 27736774 died exactly so)."""
    import ast

    tree = ast.parse(_ENTRY.read_text())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    init_line = next(n.lineno for n in ast.walk(main)
                     if isinstance(n, ast.Call)
                     and getattr(n.func, "id", None) == "_init_distributed")
    heavy = [n.lineno for n in ast.walk(main)
             if isinstance(n, (ast.Import, ast.ImportFrom))
             and any(a.name.split(".")[0] in ("equinox", "legoesm", "optax")
                     for a in n.names)
             or (isinstance(n, ast.ImportFrom)
                 and (n.module or "").split(".")[0] in ("legoesm", "equinox"))]
    # main() -> the training package import (initialises XLA) -> _main().
    assert heavy and init_line < min(heavy)


def test_sample_fingerprint_sees_structure_order_and_content():
    import numpy as np
    x = np.array([1.0, 2.0, 3.0, 4.0])
    a = (x, {"y": np.zeros(2)})
    crc, m = mod._sample_fingerprint([a])
    # a permutation keeps the sum but moves the position-weighted sum
    _, mp = mod._sample_fingerprint([(x[::-1].copy(), {"y": np.zeros(2)})])
    assert m[0, 0] == mp[0, 0] and m[0, 1] != mp[0, 1]
    # a different dtype or shape changes the structure checksum
    assert crc != mod._sample_fingerprint([(x.astype(np.float32),
                                            {"y": np.zeros(2)})])[0]
    assert crc != mod._sample_fingerprint([(x, {"y": np.zeros(3)})])[0]
    assert crc == mod._sample_fingerprint([(x.copy(), {"y": np.zeros(2)})])[0]
    # single process: nothing to compare against
    assert mod._assert_samples_agree([a], "samples", 1e-9) == 0.0


def test_physical_params_runs():
    """_physical_params used numpy without importing it (NameError at the
    first checkpoint of every run)."""
    from types import SimpleNamespace

    import numpy as np
    p = SimpleNamespace(classical=SimpleNamespace(
        as_dict=lambda: {"a": np.asarray(2.0), "b": np.ones((2, 1))}))
    assert mod._physical_params(p) == {"classical.a": 2.0,
                                       "classical.b": [1.0, 1.0]}
