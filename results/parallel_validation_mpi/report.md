# Parallel Validation Report

- Timestamp (UTC): 2026-03-02T12:09:42.909351+00:00
- Overall pass: True

## Host

- Backend: cpu
- Local devices: 1 (cpu:0)
- Device counts: cpu=1, gpu=0, tpu=0
- JAX process count: 1
- mpirun: /opt/homebrew/bin/mpirun
- mpiexec: /opt/homebrew/bin/mpiexec
- mpi4py: True
- mpi4jax: True

## Unit Tests

- Status: pass
- Return code: 0
- Elapsed (s): 1.61
- Log: `results/parallel_validation_mpi/logs/unit_test_parallel.log`

## MPI Validation

- Status: pass
- Launcher: /opt/homebrew/bin/mpirun
- Launcher args: `--oversubscribe`
- MCA overrides: btl=self,tcp
- Env overrides: MPI4JAX_NO_WARN_JAX_VERSION=1
- np=2: pass (rc=0, 5.14s) log=`results/parallel_validation_mpi/logs/mpi_np2.log`
- np=3: pass (rc=0, 5.82s) log=`results/parallel_validation_mpi/logs/mpi_np3.log`
- np=6: pass (rc=0, 7.91s) log=`results/parallel_validation_mpi/logs/mpi_np6.log`

## Scaling Validation

- Status: pass
- Thresholds: compile<= 30.0s, strong_eff>= 0.120, weak_step_growth<= 2.500, weak_per_dev_tput_ratio>= 0.350

### Backend: CPU

- Status: pass
- Requested devices: [1, 2, 3, 6]; available: 1
- Case `strong`: pass
  n=1 grid=256: pass, compile=0.086s, steady=89.641ms/step, tput=4.39 Mcells/s, per_device=4.39, per_rank=4.39, speedup=1.00x, eff=1.00, step_growth=1.00, per_dev_ratio=1.00 (log=`results/parallel_validation_mpi/logs/scaling_cpu_strong_n1.log`)
  n=2 grid=256: pass, compile=0.145s, steady=138.438ms/step, tput=2.84 Mcells/s, per_device=1.42, per_rank=2.84, speedup=0.65x, eff=0.32, step_growth=1.54, per_dev_ratio=0.32 (log=`results/parallel_validation_mpi/logs/scaling_cpu_strong_n2.log`)
  n=3 grid=256: pass, compile=0.130s, steady=130.350ms/step, tput=3.02 Mcells/s, per_device=1.01, per_rank=3.02, speedup=0.69x, eff=0.23, step_growth=1.45, per_dev_ratio=0.23 (log=`results/parallel_validation_mpi/logs/scaling_cpu_strong_n3.log`)
  n=6 grid=256: pass, compile=0.095s, steady=98.240ms/step, tput=4.00 Mcells/s, per_device=0.67, per_rank=4.00, speedup=0.91x, eff=0.15, step_growth=1.10, per_dev_ratio=0.15 (log=`results/parallel_validation_mpi/logs/scaling_cpu_strong_n6.log`)
- Case `weak`: pass
  n=1 grid=256: pass, compile=0.096s, steady=91.435ms/step, tput=4.30 Mcells/s, per_device=4.30, per_rank=4.30, speedup=1.00x, eff=1.00, step_growth=1.00, per_dev_ratio=1.00 (log=`results/parallel_validation_mpi/logs/scaling_cpu_weak_n1.log`)
  n=2 grid=362: pass, compile=0.147s, steady=164.853ms/step, tput=4.77 Mcells/s, per_device=2.38, per_rank=4.77, speedup=0.55x, eff=0.28, step_growth=1.80, per_dev_ratio=0.55 (log=`results/parallel_validation_mpi/logs/scaling_cpu_weak_n2.log`)
  n=3 grid=443: pass, compile=0.180s, steady=177.098ms/step, tput=6.65 Mcells/s, per_device=2.22, per_rank=6.65, speedup=0.52x, eff=0.17, step_growth=1.94, per_dev_ratio=0.52 (log=`results/parallel_validation_mpi/logs/scaling_cpu_weak_n3.log`)
  n=6 grid=627: pass, compile=0.110s, steady=111.957ms/step, tput=21.07 Mcells/s, per_device=3.51, per_rank=21.07, speedup=0.82x, eff=0.14, step_growth=1.22, per_dev_ratio=0.82 (log=`results/parallel_validation_mpi/logs/scaling_cpu_weak_n6.log`)

### Backend: GPU

- Status: skipped
- Reason: No local GPU devices detected.
