# Scripts Organization

Canonical category runners are now organized by domain under `scripts/`:

- Atmosphere:
  - `scripts/atmosphere/run_shallow_water_tests.py`
  - `scripts/atmosphere/run_hydrostatic_tests.py`
  - `scripts/atmosphere/run_nonhydrostatic_tests.py`
  - `scripts/atmosphere/run_rce_tests.py`
  - `scripts/atmosphere/run_aquaplanet_tests.py`
- Ocean:
  - `scripts/ocean/run_ocean_category_tests.py`

The atmosphere matrix driver remains:

- `scripts/run_atmosphere_test_matrix.py`

It now advertises category entry points via:

```bash
.venv/bin/python scripts/run_atmosphere_test_matrix.py --list-category-scripts
```

Notes:

- Existing legacy scripts in `scripts/` are kept for backward compatibility.
- New category runners are the preferred entry points for routine test execution.

