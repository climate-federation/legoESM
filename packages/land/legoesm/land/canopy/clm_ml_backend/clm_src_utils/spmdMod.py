"""
JAX/Python translation of the CLM SPMD initialization module.

In the standalone multilayer canopy configuration there is only a
single processor, so ``masterproc`` is always ``True``.

Original Fortran module: spmdMod
"""

# masterproc: proc-0 logical for printing messages — Fortran line 10
# logical, parameter :: masterproc = .true.
masterproc: bool = True
