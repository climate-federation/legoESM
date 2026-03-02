# Parallel Validation Report

- Timestamp (UTC): 2026-03-02T10:07:41.490617+00:00
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
- Elapsed (s): 1.92
- Log: `results/parallel_validation_pass1/logs/unit_test_parallel.log`

## MPI Validation

- Status: skipped
- Reason: Missing MPI prerequisites.
- Missing: MPI launcher (mpirun/mpiexec), mpi4py, mpi4jax

## Scaling Validation

- Status: pass
- n=1: pass, 185.424 ms/iter, 2.12 Mcells/s, speedup=1.00x, eff=1.00 (log=`results/parallel_validation_pass1/logs/scaling_n1.log`)
- n=2: pass, 284.751 ms/iter, 1.38 Mcells/s, speedup=0.65x, eff=0.33 (log=`results/parallel_validation_pass1/logs/scaling_n2.log`)
- n=3: pass, 263.776 ms/iter, 1.49 Mcells/s, speedup=0.70x, eff=0.23 (log=`results/parallel_validation_pass1/logs/scaling_n3.log`)
- n=6: pass, 193.327 ms/iter, 2.03 Mcells/s, speedup=0.96x, eff=0.16 (log=`results/parallel_validation_pass1/logs/scaling_n6.log`)
