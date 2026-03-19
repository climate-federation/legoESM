# Legacy Drivers

This directory contains one-off and legacy execution drivers that were moved
out of the top-level `scripts/` namespace.

Compatibility wrappers remain at original paths under `scripts/` and delegate
to these files, so existing commands continue to work.

Preferred entry points for routine test execution are the category runners in:

- `scripts/atmosphere/`
- `scripts/ocean/`
- `scripts/run_atmosphere_test_matrix.py`

