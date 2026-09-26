"""Contract tests for the ppermute latency/bandwidth microbenchmark.

Runs on CPU virtual devices — the FIT logic and the refusals are what these
gate, not the hardware constants.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

# The script under test lives in scripts/bench, which is not a package and is
# not on the path pytest builds from `testpaths`; without this line the module
# fails to COLLECT (ModuleNotFoundError) and the whole file silently
# contributes nothing.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))

from bench_ppermute_microbench import (  # noqa: E402
    COLLECTIVE_CHOICES, _HLO_TOKEN, _build_mpi, _ring, fit_latency_bandwidth,
    mpi_expected_source, sweep_elems, transport_of, verify_mpi,
)


def test_fit_recovers_known_latency_and_bandwidth():
    """t = 12 us + bytes / 25 GB/s must round-trip through the fit."""
    lat_us, bw_gbs = 12.0, 25.0
    sizes = np.array([1 << k for k in range(18, 24)], dtype=float)
    times = lat_us + sizes / (bw_gbs * 1e9) * 1e6      # bytes -> us
    fit_lat, fit_bw = fit_latency_bandwidth(sizes, times)
    assert fit_lat == pytest.approx(lat_us, rel=1e-6, abs=1e-6)
    assert fit_bw == pytest.approx(bw_gbs, rel=1e-6)


def test_fit_is_not_fooled_by_a_pure_latency_curve():
    """A flat curve (all latency) must yield an enormous, not negative, BW."""
    sizes = np.array([1 << k for k in range(18, 24)], dtype=float)
    times = np.full_like(sizes, 30.0)
    _, fit_bw = fit_latency_bandwidth(sizes, times)
    assert fit_bw > 1e3 or np.isnan(fit_bw), fit_bw


def test_single_device_is_refused():
    """A latency/bandwidth fit from one device would be meaningless."""
    import subprocess
    import sys
    from pathlib import Path

    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "bench" / "bench_ppermute_microbench.py")
    out = subprocess.run(
        [sys.executable, str(script), "--n-devices", "1"],
        capture_output=True, text=True, timeout=300,
        env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin",
             "PYTHONPATH": ":".join(sys.path)},
    )
    assert out.returncode != 0
    assert "needs >=2 devices" in (out.stderr + out.stdout)


def test_every_collective_choice_has_a_census_token_and_a_transport():
    """A choice with no HLO token would make the 'program contains the
    primitive' gate raise a KeyError instead of refusing; one with no
    transport would write a receipt that cannot be attributed to a lane."""
    assert len(COLLECTIVE_CHOICES) == 4
    for c in COLLECTIVE_CHOICES:
        assert c in _HLO_TOKEN, c
        assert transport_of(c)
    labels = {transport_of(c) for c in COLLECTIVE_CHOICES}
    assert labels == {"mpi4jax", "xla-cpu/local"}, labels
    assert all(transport_of(c) == "mpi4jax" for c in COLLECTIVE_CHOICES
               if c.startswith("mpi_"))


def test_importing_the_bench_does_not_initialise_a_backend():
    """Under --multicontroller jax.distributed.initialize() must run before
    anything touches XLA; a module-level backend lookup broke every gloo arm
    at startup (codex round 2). Import in a fresh interpreter and check no
    backend exists afterwards."""
    import subprocess
    import sys as _sys
    code = ("import bench_ppermute_microbench, jax; "
            "from jax._src import xla_bridge as xb; "
            "print('BACKENDS', sorted(xb._backends.keys()))")
    r = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=120,
                       env={**__import__("os").environ, "JAX_PLATFORMS": "cpu",
                            "PYTHONPATH": ":".join(_sys.path)})
    assert r.returncode == 0, r.stderr[-800:]
    assert "BACKENDS []" in r.stdout, r.stdout


def test_transport_label_follows_the_active_client(monkeypatch):
    """A hardcoded 'gloo' would attribute an MPI-collectives or GPU run to
    the wrong lane, and a label read from the option alone attributes a
    single-process virtual-device run to a transport JAX never built (codex
    round 3). The label follows the resolved option only when a second
    process exists."""
    import jax
    assert jax.process_count() == 1
    assert transport_of("ppermute") == "xla-cpu/local"
    monkeypatch.setattr(jax, "process_count", lambda *a, **k: 2)
    assert transport_of("ppermute") == "xla-cpu/gloo"
    jax.config.update("jax_cpu_collectives_implementation", "mpi")
    try:
        assert transport_of("ppermute") == "xla-cpu/mpi"
    finally:
        jax.config.update("jax_cpu_collectives_implementation", "gloo")
    assert transport_of("mpi_sendrecv") == "mpi4jax"


def test_identity_ring_is_refused_on_the_mpi_arm_too():
    """--ring-stride equal to the rank count made every MPI rank exchange
    with itself and the known-answer check still passed (a rank does receive
    its own index from itself). The gloo arm refused it; the MPI arm must."""
    class FakeComm:
        def Get_rank(self):
            return 0

    # Both MPI entry points refuse BEFORE touching the MPI stack, so this
    # runs without an MPI runtime and a bypass of _ring in either one fails.
    with pytest.raises(ValueError, match="multiple of the device count"):
        _build_mpi(FakeComm(), 4, 8, stride=4)
    with pytest.raises(ValueError, match="multiple of the device count"):
        verify_mpi(FakeComm(), 4, 4, "mpi_sendrecv")


def test_sweep_cap_keeps_the_large_end_and_refuses_a_useless_cap():
    full = sweep_elems(None, 4)
    assert full[0] == 64 and full[-1] == 1 << 22
    capped = sweep_elems(4096, 4)                 # 4 MiB cap, float32
    assert capped[-1] * 4 == 4096 * 1024
    assert capped[-1] < full[-1]
    assert capped[:8] == full[:8]                 # the small end is untouched
    with pytest.raises(SystemExit, match="smallest usable cap is 512"):
        sweep_elems(256, 4)   # one size >= 256 KiB: the slope has one point
    assert sweep_elems(512, 4)[-1] * 4 == 512 * 1024   # two points: allowed


def test_too_few_repetitions_are_refused_up_front():
    """With n_reps below the floor a fast transport's per-op cost sits inside
    the timer noise and the fold guard misreports a working loop as folded
    (seen on the on-node MPI arm at n_reps=3)."""
    import subprocess
    import sys as _sys
    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "bench" / "bench_ppermute_microbench.py")
    r = subprocess.run([_sys.executable, str(script), "--n-reps", "3"],
                       capture_output=True, text=True, timeout=120,
                       env={**__import__("os").environ, "JAX_PLATFORMS": "cpu",
                            "PYTHONPATH": ":".join(_sys.path)})
    assert r.returncode != 0
    assert "too few to resolve" in (r.stdout + r.stderr)


def test_mpi_expected_source_inverts_the_ring():
    """The MPI arm's known-answer check must expect exactly what the gloo
    arm's ring sends: for every (src, dst) pair in _ring, dst expects src."""
    for n, stride in ((4, 1), (8, 3), (12, 5)):
        for src, dst in _ring(n, stride):
            assert mpi_expected_source(dst, n, stride) == src
