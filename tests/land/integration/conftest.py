"""Per-test JAX cache reset for the land integration tests.

These integration tests each compile a heavy per-column graph (the two-leaf
canopy Newton ``while_loop`` solver, the soil Richards/thermal solvers).
Accumulating ~a dozen *distinct* compiled XLA executables in one long-lived
pytest process trips a jaxlib>=0.9 CPU instability — a SIGSEGV in the batched
``while_loop`` lowering (``_while_loop_batching_rule``) — that does **not**
reproduce when a test runs in isolation and is **not** a memory problem (it
crashes at ~3 GB RSS).  It is unrelated to the physics, which is correct.

Clearing JAX's compilation/tracing caches after every test keeps the live
executable set bounded so the instability is never reached.  This is a
test-harness mitigation only; production time-stepping runs the same solvers
inside a single jitted ``lax.scan`` and is unaffected.
"""

import jax
import pytest


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()
