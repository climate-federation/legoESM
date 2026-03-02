# Parallel Validation Report

- Timestamp (UTC): 2026-03-02T21:20:55.160635+00:00
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
- Elapsed (s): 3.23
- Log: `results/parallel_validation_atm_mpi_dev2/logs/unit_test_parallel.log`

## MPI Validation

- Status: skipped
- Launcher: /opt/homebrew/bin/mpirun
- Reason: N/A
- Missing: N/A
- np=2: skipped (rc=213, 0.10s) log=`results/parallel_validation_atm_mpi_dev2/logs/mpi_np2.log` reason=MPI launcher/socket access is restricted in this environment.

### MPI Scaling

- Status: skipped
- Workload: `atmosphere_sw`
- Model dt: 300.0s
- Thresholds: compile<= 30.0s, strong_eff>= 0.120, weak_step_growth<= 2.500, weak_per_rank_tput_ratio>= 0.350
- Case `strong`: skipped
  np=2 grid=8: skipped, reason=MPI launcher/socket access is restricted in this environment. (log=`results/parallel_validation_atm_mpi_dev2/logs/mpi_scaling_strong_np2.log`)
- Case `weak`: skipped
  np=2 grid=8: skipped, reason=MPI launcher/socket access is restricted in this environment. (log=`results/parallel_validation_atm_mpi_dev2/logs/mpi_scaling_weak_np2.log`)

## Scaling Validation

- Status: pass
- Workload: `atmosphere_sw`
- Model dt: 300.0s
- Thresholds: compile<= 30.0s, strong_eff>= 0.120, weak_step_growth<= 2.500, weak_per_dev_tput_ratio>= 0.350

### Backend: CPU

- Status: pass
- Requested devices: [1]; available: 1
- Case `strong`: pass
  n=1 grid=8: pass, compile=2.271s, steady=1385.404ms/step, tput=0.00 Mcells/s, per_device=0.00, per_rank=0.00, speedup=1.00x, eff=1.00, step_growth=1.00, per_dev_ratio=1.00 (log=`results/parallel_validation_atm_mpi_dev2/logs/scaling_cpu_strong_n1.log`)
- Case `weak`: pass
  n=1 grid=8: pass, compile=2.350s, steady=1329.918ms/step, tput=0.00 Mcells/s, per_device=0.00, per_rank=0.00, speedup=1.00x, eff=1.00, step_growth=1.00, per_dev_ratio=1.00 (log=`results/parallel_validation_atm_mpi_dev2/logs/scaling_cpu_weak_n1.log`)
