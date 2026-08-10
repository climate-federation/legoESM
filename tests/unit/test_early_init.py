"""Serial-path unit tests for ``legoesm.parallel.early_init``.

The multi-node branch needs a real MPI launch (covered by ``tests/distributed``);
here we pin the *serial / single-task* contract, which is what every non-MPI
import of an entry-point script hits: ``maybe_init_jax_distributed()`` must be a
cheap no-op (no mpi4py / jax import, no ``jax.distributed.initialize``) and
return ``False`` when the environment reports a single task.
"""

from __future__ import annotations

import legoesm.parallel.early_init as early_init

_TASK_ENV_VARS = ("SLURM_NTASKS", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE")


def _clear_task_env(monkeypatch):
    for var in _TASK_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_no_env_is_serial_noop(monkeypatch):
    # No launcher env at all -> treated as a single task -> no-op.
    _clear_task_env(monkeypatch)
    assert early_init.maybe_init_jax_distributed() is False


def test_ntasks_one_is_serial_noop(monkeypatch):
    _clear_task_env(monkeypatch)
    monkeypatch.setenv("SLURM_NTASKS", "1")
    assert early_init.maybe_init_jax_distributed() is False


def test_returns_before_importing_jax(monkeypatch):
    # The serial path must not even import jax: force an ImportError if it tries.
    _clear_task_env(monkeypatch)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "1")
    import builtins

    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name in ("jax", "mpi4py") or name.startswith(("jax.", "mpi4py.")):
            raise AssertionError(f"serial path must not import {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    assert early_init.maybe_init_jax_distributed() is False


def test_already_initialized_is_noop_without_jax_probe(monkeypatch):
    # #693: idempotency must NOT probe jax.process_count() (that inits the XLA
    # backend, breaking jax.distributed.initialize). When _INITIALIZED is set,
    # return False without importing jax/mpi4py even under a multi-task env.
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setattr(early_init, "_INITIALIZED", True)
    import builtins

    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name in ("jax", "mpi4py") or name.startswith(("jax.", "mpi4py.")):
            raise AssertionError(f"idempotent path must not import {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    assert early_init.maybe_init_jax_distributed() is False


# ---------------------------------------------------------------------------
# _pin_local_rank_gpu_visibility (#1516): a single-node multi-rank launch
# must bind each local rank to a DISTINCT GPU — the multi-node path does this
# inside jax.distributed.initialize(local_device_ids=...), which the
# single-node early return skips; unpinned, every rank boots default GPU 0
# and the job silently runs on one device.  Decision logic tested with NO
# GPUs: physical-device discovery is monkeypatched, and binding is asserted
# from the env JAX will consume — never from a device listing (nvidia-smi
# ignores CUDA_VISIBLE_DEVICES, the #1516 vacuous-guard trap).
# ---------------------------------------------------------------------------

_PIN_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "JAX_PLATFORM_NAME",
    "LEGOESM_NO_LOCAL_GPU_PIN",
    "PALS_LOCAL_RANKID", "OMPI_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID",
    "SLURM_STEP_NUM_TASKS", "SLURM_NTASKS", "PMI_SIZE",
    "OMPI_COMM_WORLD_SIZE",
)


def _clear_pin_env(monkeypatch, n_physical_gpus=2, nvml="derive"):
    """nvml="derive" -> NVML sees n_physical_gpus with synthetic UUIDs;
    otherwise pass an explicit (count, uuids) pair, e.g. (-1, None) for
    "NVML unavailable"."""
    import os

    for var in _PIN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    if nvml == "derive":
        nvml = (n_physical_gpus,
                [f"GPU-fake-{i}" for i in range(n_physical_gpus)])
    monkeypatch.setattr(early_init, "_nvml_accessible_gpus", lambda: nvml)
    monkeypatch.setattr(
        early_init, "_nvidia_gpu_count", lambda: n_physical_gpus)
    return os


def test_pin_unset_cvd_pins_nvml_uuid(monkeypatch):
    # The defect scenario: no shim, no CVD, 2 ranks sharing a 2-GPU node.
    # Pin by NVML UUID — under a cgroup the accessible set need not start
    # at physical index 0 (Levante job 26815351: job GPUs were 0 and 2).
    os = _clear_pin_env(monkeypatch, n_physical_gpus=2)
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    assert early_init._pin_local_rank_gpu_visibility() is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-fake-1"


def test_pin_unset_cvd_no_nvml_falls_back_to_index(monkeypatch):
    # Workstation case: no CVD, no NVML — index into the /proc device range.
    os = _clear_pin_env(monkeypatch, n_physical_gpus=2, nvml=(-1, None))
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    assert early_init._pin_local_rank_gpu_visibility() is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_pin_multi_device_cvd_indexes_visible_list(monkeypatch):
    # SLURM exports the full allocation list to every task: index by the
    # guarded SLURM_LOCALID within it.
    os = _clear_pin_env(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_LOCALID", "2")
    assert early_init._pin_local_rank_gpu_visibility() is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_pin_preserves_uuid_style_entries(monkeypatch):
    os = _clear_pin_env(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa,GPU-bbb")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    assert early_init._pin_local_rank_gpu_visibility() is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-bbb"


def test_pin_respects_real_task_isolation(monkeypatch):
    # gpus-per-task / gpu-bind=single:1 / #693 shim: the task's cgroup
    # admits exactly ONE device (NVML count 1) — its CVD single entry maps
    # to a DIFFERENT physical GPU per rank (measured, Levante job
    # 26815351).  The external decision wins.
    os = _clear_pin_env(monkeypatch, nvml=(1, ["GPU-fake-0"]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    assert early_init._pin_local_rank_gpu_visibility() is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_pin_levante_identical_single_entry_repins(monkeypatch):
    # THE measured Levante collision (jobs 26806063 + 26815351): srun
    # --gpu-bind=none hands EVERY task CVD='0' while the job cgroup admits
    # both GPUs (NVML count 2) — both ranks then compute on one physical
    # device.  A single-entry CVD contradicted by NVML is the collision
    # itself, not a shim: repin THIS rank to the NVML UUID at its
    # node-local rank.
    os = _clear_pin_env(monkeypatch, n_physical_gpus=2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    assert early_init._pin_local_rank_gpu_visibility() is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-fake-1"


def test_pin_single_entry_without_nvml_trusts_shim(monkeypatch):
    # Single-entry CVD but NVML unavailable: cannot cross-check, so the
    # possible external shim must be trusted (repinning blind could break
    # a correct #693 launch).
    os = _clear_pin_env(monkeypatch, nvml=(-1, None))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    assert early_init._pin_local_rank_gpu_visibility() is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_pin_respects_disabled_cvd(monkeypatch):
    os = _clear_pin_env(monkeypatch)
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    for disabled in ("", "-1", "NoDevFiles"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", disabled)
        assert early_init._pin_local_rank_gpu_visibility() is False
        assert os.environ["CUDA_VISIBLE_DEVICES"] == disabled


def test_pin_noop_on_cpu_only_host(monkeypatch):
    os = _clear_pin_env(monkeypatch, n_physical_gpus=0)
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    assert early_init._pin_local_rank_gpu_visibility() is False
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_pin_noop_on_explicit_cpu_platform(monkeypatch):
    # CPU-backend MPI test runs on GPU nodes must keep working.
    os = _clear_pin_env(monkeypatch)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    assert early_init._pin_local_rank_gpu_visibility() is False
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_pin_escape_hatch(monkeypatch):
    os = _clear_pin_env(monkeypatch)
    monkeypatch.setenv("LEGOESM_NO_LOCAL_GPU_PIN", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    assert early_init._pin_local_rank_gpu_visibility() is False
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_pin_no_local_rank_raises(monkeypatch):
    # GPUs present, multi-rank, but no way to bind -> refuse loudly:
    # a silent 1-GPU run in an N-GPU costume is worse than a hard failure.
    import pytest

    _clear_pin_env(monkeypatch)
    with pytest.raises(RuntimeError, match="no launcher node-local rank"):
        early_init._pin_local_rank_gpu_visibility()


def test_pin_more_ranks_than_gpus_raises(monkeypatch):
    import pytest

    _clear_pin_env(monkeypatch, n_physical_gpus=2)
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "2")
    with pytest.raises(RuntimeError, match="more local ranks than GPUs"):
        early_init._pin_local_rank_gpu_visibility()


def _fake_single_node_mpi(monkeypatch, size=2, rank=1):
    """Stub mpi4py so maybe_init_jax_distributed sees a single-node
    multi-rank world without a real launcher."""
    import sys
    import types

    comm = types.SimpleNamespace(
        Get_rank=lambda: rank,
        Get_size=lambda: size,
        allgather=lambda x: [x] * size,  # every rank on THIS host
    )
    fake = types.ModuleType("mpi4py")
    fake.MPI = types.SimpleNamespace(COMM_WORLD=comm)
    monkeypatch.setitem(sys.modules, "mpi4py", fake)


def test_maybe_init_single_node_multirank_pins_gpu(monkeypatch):
    # THE WIRING GATE for #1516, asserted against the symbol that runs
    # (maybe_init_jax_distributed itself, run_amip's import-time call):
    # the single-node early return must apply the per-rank binding it
    # used to skip.  Removing the fix turns this red.
    os = _clear_pin_env(monkeypatch, n_physical_gpus=2)
    monkeypatch.setattr(early_init, "_INITIALIZED", False)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    _fake_single_node_mpi(monkeypatch, size=2, rank=1)
    # Still returns False (jax.distributed NOT initialized) ...
    assert early_init.maybe_init_jax_distributed() is False
    # ... but this rank is now bound to its own GPU (by NVML UUID).
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-fake-1"


def test_maybe_init_pins_before_mpi_import(monkeypatch):
    # ORDERING GATE (job 26829100): a CUDA-aware MPI stack (Open MPI/UCX)
    # initialises the CUDA driver during MPI_Init, and the driver snapshots
    # CUDA_VISIBLE_DEVICES at first initialisation — a pin applied after
    # `from mpi4py import MPI` is silently IGNORED (measured: post-pin
    # CVD='0'/'1' per rank, yet every rank enumerated BOTH GPUs and
    # computed on device 0).  Prove the order by making the mpi4py import
    # explode: the pin must ALREADY have been applied.
    import sys

    import pytest

    os = _clear_pin_env(monkeypatch, n_physical_gpus=2)
    monkeypatch.setattr(early_init, "_INITIALIZED", False)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    monkeypatch.setitem(sys.modules, "mpi4py", None)  # import -> ImportError
    with pytest.raises(ImportError):
        early_init.maybe_init_jax_distributed()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-fake-1"


def test_maybe_init_single_rank_never_pins(monkeypatch):
    # A single process may legitimately drive multiple GPUs (documented
    # eff=0.5 hazard): the serial path must stay pin-free.
    os = _clear_pin_env(monkeypatch, n_physical_gpus=2)
    monkeypatch.setattr(early_init, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    assert early_init.maybe_init_jax_distributed() is False
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


# ---------------------------------------------------------------------------
# resolve_coordinator_port: env override > job-id-derived > legacy default.
# ---------------------------------------------------------------------------

_PORT_ENV_VARS = ("LEGOESM_COORDINATOR_PORT", "SLURM_JOB_ID", "PBS_JOBID")


def _clear_port_env(monkeypatch):
    for var in _PORT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_port_env_override_wins(monkeypatch):
    _clear_port_env(monkeypatch)
    monkeypatch.setenv("LEGOESM_COORDINATOR_PORT", "23456")
    monkeypatch.setenv("SLURM_JOB_ID", "8888")  # must lose to the override
    assert early_init.resolve_coordinator_port() == 23456


def test_port_no_jobid_falls_back_to_legacy(monkeypatch):
    _clear_port_env(monkeypatch)
    assert early_init.resolve_coordinator_port() == 1234
    assert early_init.resolve_coordinator_port(default=4321) == 4321


def test_port_jobid_derivation_deterministic_and_in_range(monkeypatch):
    _clear_port_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "8523462")
    p1 = early_init.resolve_coordinator_port()
    p2 = early_init.resolve_coordinator_port()
    # Every rank of one job must compute the SAME port with no communication.
    assert p1 == p2
    assert 20000 <= p1 < 60000


def test_port_differs_across_jobs(monkeypatch):
    # Two jobs sharing a node must not collide on the coordinator port
    # (the historical hardcoded 1234 EADDRINUSE failure mode).
    _clear_port_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "1000001")
    p_a = early_init.resolve_coordinator_port()
    monkeypatch.setenv("SLURM_JOB_ID", "1000002")
    p_b = early_init.resolve_coordinator_port()
    assert p_a != p_b


def test_port_pbs_jobid_honored(monkeypatch):
    _clear_port_env(monkeypatch)
    monkeypatch.setenv("PBS_JOBID", "1234567.desched1")
    p = early_init.resolve_coordinator_port()
    assert 20000 <= p < 60000


# ---------------------------------------------------------------------------
# init_jax_distributed_with_fallback: transport chosen by ENVIRONMENT.
# ---------------------------------------------------------------------------

_LAUNCHER_ENV_VARS = (
    "SLURM_JOB_ID", "OMPI_COMM_WORLD_SIZE", "PALS_RANKID", "PMI_RANK",
    # World-size vars too: a test suite RUNNING INSIDE a SLURM job inherits
    # SLURM_NTASKS/SLURM_STEP_NUM_TASKS from the host job; leaving them set
    # makes the fake-jax process_count and the guard's launcher_world_size
    # disagree (host SLURM_NTASKS=1 vs the test's OMPI=4) and the
    # silent-fallback guard fires on a correctly-federated fake (found
    # running the suite under sbatch on Ginsburg).
    "SLURM_NTASKS", "SLURM_STEP_NUM_TASKS", "PMI_SIZE",
    "OMPI_COMM_WORLD_RANK", "SLURM_PROCID", "PALS_LOCAL_RANKID",
    "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",
)


class _FakeDistributed:
    def __init__(self, fail_bare_with: str | None = None):
        self.calls = []
        self._fail_bare_with = fail_bare_with

    def is_initialized(self):
        # The helper's cross-path idempotency probe (never initialized in
        # these unit tests — the federation itself is faked).
        return False

    def initialize(self, *args, **kwargs):
        self.calls.append(kwargs)
        if not kwargs and self._fail_bare_with is not None:
            raise RuntimeError(self._fail_bare_with)


def _with_fake_jax(monkeypatch, fake, process_count: int | None = None):
    import os
    import sys
    import types

    # The post-init silent-fallback guard compares jax.process_count()
    # against the launcher-declared world size; the fake federation
    # matches the declared size by default so guarded paths pass.
    if process_count is None:
        # SAME precedence as early_init.launcher_world_size (step size, then
        # the MPI launcher's world, allocation-wide SLURM_NTASKS last) so the
        # fake federation always matches what the guard will declare.
        for var in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE",
                    "PMI_SIZE", "SLURM_NTASKS"):
            v = os.environ.get(var)
            if v and v.isdigit():
                process_count = int(v)
                break
        else:
            process_count = 1
    fake_jax = types.SimpleNamespace(
        distributed=fake, process_count=lambda: process_count)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)


def _clear_launcher_env(monkeypatch):
    for var in _LAUNCHER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Shared idempotency flag with maybe_init_jax_distributed.
    monkeypatch.setattr(early_init, "_INITIALIZED", False)


def test_fallback_pals_only_uses_mpi4py_bootstrap(monkeypatch):
    # PALS/PMI launcher, no SLURM/OMPI -> mpi4py bootstrap directly (bare
    # initialize has no PALS auto-detection), with the #693 one-device
    # binding. mpi4py is only find_spec'd; skip if entirely absent.
    import importlib.util

    if importlib.util.find_spec("mpi4py") is None:
        import pytest

        pytest.skip("mpi4py not installed")
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("PALS_RANKID", "0")
    fake = _FakeDistributed()
    _with_fake_jax(monkeypatch, fake)
    early_init.init_jax_distributed_with_fallback()
    assert fake.calls == [
        {"cluster_detection_method": "mpi4py", "local_device_ids": [0]},
    ]
    assert early_init._INITIALIZED is True


def test_fallback_slurm_uses_bare_initialize(monkeypatch):
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    fake = _FakeDistributed()
    _with_fake_jax(monkeypatch, fake)
    early_init.init_jax_distributed_with_fallback()
    assert fake.calls == [{}]
    assert early_init._INITIALIZED is True


def test_fallback_real_failure_reraises(monkeypatch):
    # A genuine initialize() failure under an auto-detectable launcher must
    # NOT be masked by a second attempt — even when the message mentions
    # cluster/coordinator words.
    import pytest

    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    fake = _FakeDistributed(
        fail_bare_with="failed to connect to cluster coordinator",
    )
    _with_fake_jax(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="cluster coordinator"):
        early_init.init_jax_distributed_with_fallback()
    assert fake.calls == [{}]  # exactly one attempt, no silent fallback
    assert early_init._INITIALIZED is False


def test_fallback_already_initialized_is_noop(monkeypatch):
    _clear_launcher_env(monkeypatch)
    fake = _FakeDistributed(fail_bare_with="already initialized")
    _with_fake_jax(monkeypatch, fake)
    early_init.init_jax_distributed_with_fallback()  # no raise
    assert fake.calls == [{}]
    assert early_init._INITIALIZED is True


def test_fallback_respects_shared_initialized_flag(monkeypatch):
    _clear_launcher_env(monkeypatch)
    monkeypatch.setattr(early_init, "_INITIALIZED", True)
    fake = _FakeDistributed()
    _with_fake_jax(monkeypatch, fake)
    early_init.init_jax_distributed_with_fallback()
    assert fake.calls == []  # idempotent no-op, no jax.distributed touch


# ---------------------------------------------------------------------------
# init_multicontroller_distributed: the shared --multicontroller entry point
# (ocean/atm SPMD benches + the run_omip route-B driver).
# ---------------------------------------------------------------------------

def test_multicontroller_coordinator_uses_launcher_env(monkeypatch):
    # Explicit coordinator -> read rank/size from the OMPI launcher env and
    # call initialize() directly (the mpiexec / self-spawn path).
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    fake = _FakeDistributed()
    _with_fake_jax(monkeypatch, fake)
    early_init.init_multicontroller_distributed("host:5000")
    assert fake.calls == [
        {"coordinator_address": "host:5000",
         "num_processes": 4, "process_id": 2,
         # Per-rank device binding applied on the explicit-coordinator
         # path too (clean env -> default [0]).
         "local_device_ids": [0]},
    ]
    assert early_init._INITIALIZED is True


def test_multicontroller_coordinator_missing_env_raises(monkeypatch):
    # A coordinator without a launcher rank env is a HARD error — never a
    # silent single-process fallback (that would run N un-federated copies
    # clobbering each other's output).
    import pytest

    _clear_launcher_env(monkeypatch)
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    monkeypatch.delenv("PMI_SIZE", raising=False)
    fake = _FakeDistributed()
    _with_fake_jax(monkeypatch, fake)
    with pytest.raises(SystemExit, match="no launcher rank env"):
        early_init.init_multicontroller_distributed("host:5000")
    assert fake.calls == []  # never initialized on a bad env


def test_multicontroller_no_coordinator_delegates_to_fallback(monkeypatch):
    # No coordinator -> environment-routed fallback (here SLURM -> bare init).
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    fake = _FakeDistributed()
    _with_fake_jax(monkeypatch, fake)
    early_init.init_multicontroller_distributed(None)
    assert fake.calls == [{}]  # bare auto-detect via init_..._with_fallback
    assert early_init._INITIALIZED is True
