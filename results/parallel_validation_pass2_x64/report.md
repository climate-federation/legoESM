# Parallel Validation Report

- Timestamp (UTC): 2026-03-02T10:08:33.289023+00:00
- Overall pass: True

## Host

- Backend: cpu
- Local devices: 1 (cpu:0)
- JAX process count: 1
- mpirun: not found
- mpiexec: not found
- mpi4py: False
- mpi4jax: False

## Unit Tests

- Status: pass
- Return code: 0
- Elapsed (s): 1.72
- Log: `results/parallel_validation_pass2_x64/logs/unit_test_parallel.log`

## MPI Validation

- Status: skipped
- Reason: Missing MPI prerequisites.
- Missing: MPI launcher (mpirun/mpiexec), mpi4py, mpi4jax

## Scaling Validation

- Status: pass
- n=1: pass, 138.371 ms/iter, 2.84 Mcells/s, speedup=1.00x, eff=1.00 (log=`results/parallel_validation_pass2_x64/logs/scaling_n1.log`)
- n=2: pass, 291.786 ms/iter, 1.35 Mcells/s, speedup=0.47x, eff=0.24 (log=`results/parallel_validation_pass2_x64/logs/scaling_n2.log`)
- n=3: pass, 313.834 ms/iter, 1.25 Mcells/s, speedup=0.44x, eff=0.15 (log=`results/parallel_validation_pass2_x64/logs/scaling_n3.log`)
- n=6: pass, 281.538 ms/iter, 1.40 Mcells/s, speedup=0.49x, eff=0.08 (log=`results/parallel_validation_pass2_x64/logs/scaling_n6.log`)
