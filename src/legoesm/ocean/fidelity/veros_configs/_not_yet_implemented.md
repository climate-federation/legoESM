# Veros per-case adapters — implementation status

| Case                | Status      | Notes |
|---------------------|-------------|-------|
| acc_channel         | implemented | thin adapter of `veros.setups.acc.acc.ACCSetup` |
| global_overturning  | implemented | thin adapter of `veros.setups.global_4deg.global_4deg.GlobalFourDegreeSetup`; downloads ~50 MB on first run |
| lock_exchange       | implemented | Petersen 2015 Fig. 5 geometry: channel, no rotation, dense block release |
| overflow            | implemented | Petersen 2015 Fig. 7: shelf/slope/abyss dense overflow |
| eady_uniform        | implemented | re-entrant zonal channel, uniform N², linear shear, Gaussian jet envelope, single-wavenumber T perturbation |
| dino                | implemented | Kamm et al. 2025 GMD basin + Drake-passage channel, Bryan-Cox T*/S* restoring + cubic-Hermite wind stress; Jerlov SW penetration intentionally omitted on the Veros side |

When a new case lands:

1. Add `<case>.py` in this directory with `make_setup(**kwargs) ->
   veros.VerosSetup`.
2. Register it in `AVAILABLE_CASES` in `__init__.py`.
3. Add a corresponding entry to `tests/ocean/fidelity/test_veros_runner.py`
   with `pytest.importorskip("veros")` + a `@pytest.mark.slow` smoke test.
