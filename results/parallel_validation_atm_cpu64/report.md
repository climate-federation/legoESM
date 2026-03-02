# Parallel Validation Report

- Timestamp (UTC): 2026-03-02T21:25:55.276507+00:00
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
- Elapsed (s): 3.36
- Log: `results/parallel_validation_atm_cpu64/logs/unit_test_parallel.log`

## MPI Validation

- Status: skipped
- Reason: Missing MPI prerequisites.
- Missing: MPI launcher (mpirun/mpiexec)

### MPI Scaling

- Status: skipped
- Reason: Missing MPI prerequisites.

## Scaling Validation

- Status: pass
- Workload: `atmosphere_sw`
- Model dt: 300.0s
- Thresholds: compile<= 30.0s, strong_eff>= 0.120, weak_step_growth<= 2.500, weak_per_dev_tput_ratio>= 0.350

### Backend: CPU

- Status: pass
- Requested devices: [1, 2]; available: 1
- Case `strong`: pass
  n=1 grid=64: pass, compile=2.233s, steady=544.251ms/step, tput=0.05 Mcells/s, per_device=0.05, per_rank=0.05, speedup=1.00x, eff=1.00, step_growth=1.00, per_dev_ratio=1.00 (log=`results/parallel_validation_atm_cpu64/logs/scaling_cpu_strong_n1.log`)
  n=2 grid=64: pass, compile=3.713s, steady=1016.682ms/step, tput=0.02 Mcells/s, per_device=0.01, per_rank=0.02, speedup=0.54x, eff=0.27, step_growth=1.87, per_dev_ratio=0.27 (log=`results/parallel_validation_atm_cpu64/logs/scaling_cpu_strong_n2.log`)
- Case `weak`: pass
  n=1 grid=64: pass, compile=2.287s, steady=507.802ms/step, tput=0.05 Mcells/s, per_device=0.05, per_rank=0.05, speedup=1.00x, eff=1.00, step_growth=1.00, per_dev_ratio=1.00 (log=`results/parallel_validation_atm_cpu64/logs/scaling_cpu_weak_n1.log`)
  n=2 grid=91: pass, compile=3.532s, steady=1019.445ms/step, tput=0.05 Mcells/s, per_device=0.02, per_rank=0.05, speedup=0.50x, eff=0.25, step_growth=2.01, per_dev_ratio=0.50 (log=`results/parallel_validation_atm_cpu64/logs/scaling_cpu_weak_n2.log`)
