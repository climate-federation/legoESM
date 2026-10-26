"""Resolution-scaled del4 hyperdiffusion for the scaling benchmarks.

One law for every grid type: nu ~ dx^4, anchored per grid.  Shared by
run_levante_gpu_scaling.py and bench_mpas_spmd_scaling.py so the two
harnesses cannot carry different coefficients for the same mesh.
"""


def hyperdiff_coeff(n_grid: int, grid_type: str = "cubed-sphere") -> float:
    """del4 coefficient [m^4/s] for a resolution index.

    ``n_grid`` is the subdivision level (icosahedral), truncation T (spectral),
    or cells per edge / latitude count (cubed-sphere, latlon).
    Icosahedral: 1e16 at subdivision <= 4, 16x weaker per finer level (dx
    halves per level). A fixed 1e16 sent s7 non-finite within 4 steps at 30 s.
    """
    if grid_type == "icosahedral":
        return 1.0e16 / 16.0 ** max(0, n_grid - 4)
    if grid_type == "spectral":
        ref_n, ref_coeff = 42, 2.5e16
    elif grid_type == "latlon":
        ref_n, ref_coeff = 64, 5e16
    elif grid_type == "cubed-sphere":
        ref_n, ref_coeff = 48, 5e16
    else:
        raise ValueError(f"hyperdiff_coeff: unknown grid_type {grid_type!r}")
    return ref_coeff * (ref_n / n_grid) ** 4
