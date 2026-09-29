"""Measure how the WeatherBench training arm's memory responds to sharding.

The classical T63/L32 arm runs out of memory compiling its gradient.  This
answers one question: how much of that memory goes away when the horizontal
columns are split across devices, and where the split has to be applied for
it to happen.

Compiles only, by default -- no ERA5, no execution -- so the question can be
asked of a configuration in minutes.  Reports XLA's scratch arena for the
value-and-grad program, the collectives sharding inserted, and a census of
buffer widths.

Two arms, and the difference between them is the finding:

  * default: place the carry on a column mesh and let the compiler propagate
    the split.  Worth 4-14%.
  * ``SHARD_RAD=1``: additionally constrain the radiation solver's own column
    axis.  Worth close to the device count.

Input placement alone fails because the largest arrays in the arena belong to
the radiation solver and carry the full column count on an INNER axis, after
the sub-column expansion; the split never propagates into them, so they stay
full width at every device count.  Constraining the solver boundary is what
reaches them.

WHAT THIS DOES NOT ESTABLISH, per adversarial review of the first results:

  * The compiler's arena estimate is not a measured runtime peak, and nothing
    here validates one against the other.
  * The carry is a test fixture, not a real sample.  Branches a real field
    would exercise may be folded away, so every number is a LOWER bound.
  * Constraining the solver's outputs as well as its inputs grants the
    compiler freedom that a real implementation constraining only the entry
    point may not reproduce.  Treat the constraint set here as the spec any
    implementation has to match.

Environment:
  ``DECK``       deck to measure (arena depends on the deck far more than on
                 resolution: two T63/L32 decks differ by 10x)
  ``SHARD_RAD``  ``1`` constrains the radiation solver's column axis
  ``RUN``        ``1`` also executes once and prints loss and gradient norm
  ``LEGOESM_DIST`` ``1`` joins a multi-process job before touching a device
  ``HLO_OUT``    write the optimized HLO here

Usage: wb_shard_arena.py <n_max> <nlev> <n_devices> [steps]
"""
import os
import re
import sys
from collections import Counter
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from legoesm.training.losses import combined_loss
from legoesm.training.scale_build import build_mode_components

sys.path.insert(0, ".")
from tests.unit.test_spectral_carry_tracer_set import _carry  # noqa: E402

_COLLECTIVE_RE = re.compile(
    r"=\s*[^\s=]+\s+((?:all-gather|all-reduce|all-to-all|collective-permute|"
    r"reduce-scatter|collective-broadcast)(?:-start|-done)?)\(")


def census_collectives(hlo_text: str) -> Counter:
    """Count communication operations in optimized HLO, async spellings included.

    XLA emits ``all-reduce-start`` / ``all-reduce-done`` for overlapped
    collectives, so a census that knows only the synchronous names reports
    "none" on a program full of communication.
    """
    return Counter(_COLLECTIVE_RE.findall(hlo_text))


def census_leading_widths(hlo_text: str) -> Counter:
    """Count how often each LEADING array width appears in optimized HLO.

    A heuristic for "did anything get partitioned at all".  It counts textual
    shape occurrences rather than allocated buffers, and it is blind to a
    column count that sits on an inner axis -- which is precisely how the
    radiation arrays stayed full width while this census read as mostly
    partitioned.  Anything quantitative belongs to the buffer-assignment
    report instead.
    """
    lead = Counter()
    for shape in re.findall(r"[a-z]\d*\[([0-9,]+)\]", hlo_text):
        dims = [int(d) for d in shape.split(",") if d]
        if dims:
            lead[dims[0]] += 1
    return lead


def main() -> None:
    n_max = int(sys.argv[1])
    nlev = int(sys.argv[2])
    ndev = int(sys.argv[3])
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    gib = 1024.0 ** 3

    if os.environ.get("LEGOESM_DIST") == "1":
        # Multi-node: every process joins before any device is touched,
        # after which jax.devices() spans the whole job.
        jax.distributed.initialize()
        print(f"DIST process {jax.process_index()} of {jax.process_count()}")

    devices = jax.devices()
    if len(devices) < ndev:
        raise SystemExit(
            f"asked for {ndev} devices, jax sees {len(devices)}: {devices}"
        )
    devices = devices[:ndev]
    mesh = Mesh(np.asarray(devices), ("col",))
    print(f"BACKEND {jax.default_backend()} DEVICES {[str(d) for d in devices]}")

    # --- optional: force the partition INSIDE the radiation solver ---------
    # Input sharding alone leaves the RRTMGP working arrays at full column
    # width (they carry the column count on a non-leading axis after the
    # sub-column expansion, so the partition does not propagate into them).
    # SHARD_RAD=1 constrains the solver's own column-leading arguments and
    # results, which is what a real implementation would do.
    if os.environ.get("SHARD_RAD") == "1":
        from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp as _rr

        _solver = _rr.RRTMGP
        _orig_solve = _solver.solve_columns

        def _constrain(x):
            if not hasattr(x, "shape") or getattr(x, "ndim", 0) == 0:
                return x
            if x.shape and x.shape[0] % ndev == 0 and x.shape[0] >= ndev:
                spec = P("col", *((None,) * (x.ndim - 1)))
                return jax.lax.with_sharding_constraint(
                    x, NamedSharding(mesh, spec))
            return x

        def _sharded_solve(self, *a, **kw):
            a = tuple(_constrain(x) for x in a)
            kw = {k: _constrain(v) for k, v in kw.items()}
            out = _orig_solve(self, *a, **kw)
            return jax.tree.map(_constrain, out)

        _solver.solve_columns = _sharded_solve
        print("SHARD_RAD active: radiation column axis constrained")


    deck = os.environ.get("DECK", "config/wb/campaign/spectral_t63_bechtold_clubb.yaml")
    print(f"DECK {deck}")
    yml = yaml.safe_load(open(deck))
    yml["spectral"] = dict(yml["spectral"], n_max=n_max)
    yml["nlev"] = nlev
    cfg = SimpleNamespace(mode="physics", training_core="spectral", smoke=False,
                          multi_step_hours=(6,), n_days=1)
    _m, grid, sigma, params, make_run_seg, loss_config, dt = build_mode_components(cfg, yml)
    make_run_seg(params)  # warm the RRTMGP table cache outside any trace
    sigma_full = jnp.asarray(sigma.sigma_full)
    arr, static = eqx.partition(params, eqx.is_inexact_array)

    n_lat, n_lon = int(grid.n_lat), int(grid.n_lon)
    ncol = n_lat * n_lon
    if n_lat % ndev or ncol % ndev:
        raise SystemExit(
            f"n_lat={n_lat} / ncol={ncol} not divisible by {ndev} devices; "
            "pick a truncation whose Gaussian latitude count divides the mesh"
        )
    print(f"GRID n_lat={n_lat} n_lon={n_lon} ncol={ncol} nlev={nlev}")

    ic = _carry(n_lat, n_lon, nlev=nlev, extras=True)
    target = _carry(n_lat, n_lon, nlev=nlev, extras=True)
    forcing = {
        "T_sfc": jnp.full((ncol,), 288.0),
        "sic": jnp.zeros((ncol,)),
        "day_of_year": jnp.asarray(244.0),
        "seconds_of_day": jnp.asarray(21600.0),
        # physics mode reads the land mask off the forcing, not the grid
        "land_frac": jnp.zeros((n_lat, n_lon)),
    }


    def _spec(x):
        """Shard the leading latitude / column axis, replicate everything else."""
        x = jnp.asarray(x)
        if x.ndim == 0:
            return NamedSharding(mesh, P())
        lead = x.shape[0]
        if lead in (n_lat, ncol) and lead % ndev == 0:
            return NamedSharding(mesh, P("col", *((None,) * (x.ndim - 1))))
        return NamedSharding(mesh, P(*((None,) * x.ndim)))


    def _place(tree):
        return jax.tree.map(
            lambda x: jax.device_put(jnp.asarray(x), _spec(x)), tree
        )


    ic = _place(ic)
    target = _place(target)
    forcing = _place(forcing)
    arr = jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P(*((None,) * x.ndim)))),
        arr,
    )


    def loss_fn(a, ic_, tg_, fc_):
        pred = make_run_seg(eqx.combine(a, static)).raw(ic_, steps, fc_)
        return combined_loss(pred, tg_, sigma_full, grid=grid, config=loss_config)


    span_h = steps * float(dt) / 3600.0
    print(f"ROLLOUT steps={steps} dt={float(dt):.1f}s span={span_h:.3f}h")
    if abs(span_h - 6.0) > 1e-6:
        raise SystemExit(
            f"steps x dt = {span_h:.3f} h, not the 6 h lead this deck trains "
            "on; pass the matching step count or the arena is measured over "
            "the wrong rollout")

    vg = eqx.filter_jit(jax.value_and_grad(loss_fn))
    lowered = vg.lower(arr, ic, target, forcing)
    compiled = lowered.compile()
    inner = getattr(compiled, "compiled", compiled)
    m = inner.memory_analysis()

    print(f"MEM n_max={n_max} nlev={nlev} ndev={ndev} steps={steps}")
    for name in ("temp_size_in_bytes", "argument_size_in_bytes",
                 "output_size_in_bytes", "alias_size_in_bytes"):
        v = getattr(m, name, None)
        if v is not None:
            print(f"  {name:28s} {v / gib:10.4f} gib")
    total = (getattr(m, "temp_size_in_bytes", 0)
             + getattr(m, "argument_size_in_bytes", 0)
             + getattr(m, "output_size_in_bytes", 0)
             - getattr(m, "alias_size_in_bytes", 0))
    print(f"  {'device_total':28s} {total / gib:10.4f} gib")

    # --- did the physics actually get partitioned? -------------------------
    txt = inner.as_text()
    print(f"HLO_CHARS {len(txt)}")

    coll = census_collectives(txt)
    print("COLLECTIVES " + (", ".join(f"{k}={v}" for k, v in sorted(coll.items()))
                            or "none"))

    lead = census_leading_widths(txt)
    want_full = {n_lat, ncol, ncol * 8}
    want_shard = {n_lat // ndev, ncol // ndev, (ncol * 8) // ndev}
    full_ct = sum(v for k, v in lead.items() if k in want_full)
    shard_ct = sum(v for k, v in lead.items() if k in want_shard)
    print(f"WIDTH full={full_ct} sharded={shard_ct} "
          f"(full widths {sorted(want_full)}, sharded {sorted(want_shard)})")
    print("TOP_WIDTHS " + ", ".join(f"{k}:{v}" for k, v in lead.most_common(12)))

    out = os.environ.get("HLO_OUT")
    if out:
        with open(out, "w") as fh:
            fh.write(txt)
        print(f"HLO_WRITTEN {out}")

    if os.environ.get("RUN") == "1":
        # Numerical control: a sharding constraint must not change the answer.
        # Compile-only cannot see that, so execute once and print the loss and
        # the gradient norm at full precision for a device-count comparison.
        val, grad = vg(arr, ic, target, forcing)
        leaves = [x for x in jax.tree.leaves(grad) if hasattr(x, "shape")]
        gnorm = float(jnp.sqrt(sum(jnp.vdot(x, x).real for x in leaves)))
        print(f"RUN loss={float(val):.17g} gradnorm={gnorm:.17g}")
        if not (np.isfinite(float(val)) and np.isfinite(gnorm)):
            raise SystemExit(
                "RUN produced a non-finite loss or gradient, so it cannot compare "
                "one device count against another: nan == nan proves nothing. "
                "Seed the carry from a real sample before trusting this control.")


if __name__ == "__main__":
    main()
