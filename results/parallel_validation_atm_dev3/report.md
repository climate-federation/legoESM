# Parallel Validation Report

- Timestamp (UTC): 2026-03-02T21:22:00.626117+00:00
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
- Elapsed (s): 3.49
- Log: `results/parallel_validation_atm_dev3/logs/unit_test_parallel.log`

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
- Requested devices: [1]; available: 1
- Case `strong`: pass
  n=1 grid=8: pass, compile=2.231s, steady=1442.045ms/step, tput=0.00 Mcells/s, per_device=0.00, per_rank=0.00, speedup=1.00x, eff=1.00, step_growth=1.00, per_dev_ratio=1.00 (log=`results/parallel_validation_atm_dev3/logs/scaling_cpu_strong_n1.log`)
- Case `weak`: pass
  n=1 grid=8: pass, compile=2.065s, steady=1555.635ms/step, tput=0.00 Mcells/s, per_device=0.00, per_rank=0.00, speedup=1.00x, eff=1.00, step_growth=1.00, per_dev_ratio=1.00 (log=`results/parallel_validation_atm_dev3/logs/scaling_cpu_weak_n1.log`)
