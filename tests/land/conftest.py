"""Per-test JAX cache reset for the land tests.

Every compiled XLA executable keeps its code mapped in the process, and the
land tests compile many distinct per-column graphs (the canopy Newton solve,
the soil Richards/thermal solvers, eager op-by-op calls).  One long pytest
process climbs about 1600 memory maps per canopy test until it reaches the
kernel's ``vm.max_map_count`` (65530 on Derecho); LLVM then fails
``allocateMappedMemory`` ("Cannot allocate memory", "Failed to materialize
symbols") and the whole process aborts, at ~5 GB RSS on a 230 GB node.  With
the caches cleared after each test the count stays near 2000.

Test-harness mitigation only; production steps the same solvers inside one
jitted scan and holds a bounded set of executables.
"""

import jax
import pytest


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()
