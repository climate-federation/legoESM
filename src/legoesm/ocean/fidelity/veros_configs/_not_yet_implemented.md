# Veros per-case adapters — implementation status

| Case                | Status      | Notes |
|---------------------|-------------|-------|
| acc_channel         | implemented | thin adapter of `veros.setups.acc.acc.ACCSetup` |
| global_overturning  | implemented | thin adapter of `veros.setups.global_4deg.global_4deg.GlobalFourDegreeSetup`; downloads ~50 MB on first run |
| lock_exchange       | TODO        | needs a custom `VerosSetup` (Petersen 2015 Fig. 5 geometry: channel, no rotation, dense block release) |
| overflow            | TODO        | extends `lock_exchange` with a sloped bottom (Petersen 2015 Fig. 7) |
| eady_uniform        | TODO        | custom `VerosSetup` (re-entrant zonal channel, uniform N², linear shear) |
| dino                | TODO        | custom `VerosSetup` (basin + Drake-like channel, Bryan-Cox restoring); biggest of the four |

The four TODO cases each need ~80–150 LOC of bespoke Veros setup code that
mirrors the matching legoESM experiment's geometry, forcing, and initial
condition. Tracked as task #12 in the ocean fidelity plan; see
`/Users/pierregentine/.claude/plans/define-a-plan-to-abundant-narwhal.md`.

When each TODO lands:

1. Add `<case>.py` in this directory with `make_setup(**kwargs) ->
   veros.VerosSetup`.
2. Register it in `AVAILABLE_CASES` in `__init__.py`.
3. Add a corresponding entry to `tests/ocean/fidelity/test_veros_runner.py`
   with `pytest.importorskip("veros")` + a `@pytest.mark.slow` smoke test.
