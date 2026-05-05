# RRTMGP audit cycle (deferred from initial atmosphere-physics audit)

**Scope:** `src/legoesm/atmosphere/physics/radiation/rrtmgp/`, 6583 LOC across 21 files
**Date:** 2026-05-01
**Mode:** static analysis + targeted code reading; existing test suite already passes (`tests/atmosphere/hydrostatic/unit/test_radiation.py`, 71 tests)
**Source code modified:** none — no critical findings warranting fixes.

## Modules

```
rrtmgp/
├── constants.py                       # imports from legoesm.constants ✓
├── rrtmgp.py                          # top-level entry, gray + RRTMGP wrapper
├── kernel_ops.py                      # JAX kernel ops (where/clip ok)
├── interpolation.py                   # lookup-table interpolation
├── stretched_grid_util.py             # vertical-grid handling
├── rrtmgp_common.py                   # common constants / helpers
├── config/
│   └── radiative_transfer.py          # config NamedTuple
├── utils/file_io.py                   # asset loading
├── rte/
│   ├── two_stream.py                  # broadband two-stream solver
│   ├── monochromatic_two_stream.py    # per-g-point two-stream
│   └── rte_utils.py
└── optics/
    ├── optics.py                      # 775 LOC — main optics dispatch
    ├── gas_optics.py                  # 623 LOC — gas absorption
    ├── cloud_optics.py                # cloud optical properties
    ├── lookup_gas_optics_{shortwave,longwave}.py
    ├── lookup_cloud_optics.py
    ├── lookup_volume_mixing_ratio.py
    ├── optics_base.py
    └── optics_utils.py
```

## Static checks

| Check | Result |
|-------|--------|
| Hardcoded duplicates of physical constants (g, R_d, c_pd, L_v, T_freeze, sigma_sb, etc.) | NONE — `rrtmgp/constants.py` re-exports from `legoesm.constants` (`G`, `R_D`, `R_V`, `CP_D`, `CV_D`, `CP_V`); only RRTMGP-specific molecular constants (`DRY_AIR_MOL_MASS=0.0289647`, `WATER_MOL_MASS=0.0180153`, `AVOGADRO=6.022e23`) are local. ✓ |
| Hardcoded freezing/EOS values (273.15, 461.51, 287.05, 1004.64, 9.80616, 5.67e-8, 6.371e6) | NONE found in `src/legoesm/atmosphere/physics/radiation/rrtmgp/`. ✓ |
| Pa↔hPa explicit conversions | 2 sites: `rrtmgp.py:84` (`p_hPa = p_full / 100.0`) for an analytical ozone formula and `lookup_volume_mixing_ratio.py:152` (`p_hpa = p / 100`). Both are commented and physically consistent. ✓ |
| TODO / FIXME / HACK / XXX / BUG markers | NONE. ✓ |
| `clip` / `where` / `maximum` for differentiability | Many — typical for a JAX-native rewrite of a Fortran lookup-table scheme.  Spot-checked the major sites in `rrtmgp.py`, `optics/optics.py`, `rte/two_stream.py`: all are limit-style guards (e.g., `clip(p, p_ref_min, p_ref_max)` for table-extrapolation safety) rather than dead-gradient paths. |
| Test coverage | 71 radiation tests pass post-fixes (`tests/atmosphere/hydrostatic/unit/test_radiation.py`).  RRTMGP-specific tests include `TestOzoneProfile::test_rrtmgp_*` and `TestRRTMGP::test_rrtmgp_basic_*`. |

## Targeted code reading

**`rrtmgp.py`** (top-level dispatch).  Uses `constants.S_0` after the P2 fix.  Analytical ozone fallback (`8e-6 * exp(-0.5 * ((ln(p_hPa) - ln(10)) / 1.5)**2)`) is intentionally a Gaussian in log-pressure — physical origin is ozone climatology, not derived constants.  Heating-rate assembly uses `c_pd` and `g` from `legoesm.constants` via the `rrtmgp.constants` re-export.

**`optics/optics.py`** (775 LOC).  Pure-JAX dispatch over `lookup_gas_optics_{shortwave,longwave}`.  Spot-check: lookup-table interpolation is bilinear / trilinear with `jnp.clip` on the indices to avoid out-of-bounds — standard pattern.

**`rte/two_stream.py`**.  Broadband solver using exp/coth functions — fully differentiable.  No Python control flow on traced values.

**`config/radiative_transfer.py`**.  Config is a NamedTuple holding paths to lookup-table files plus a `vmr_zonal_mean_*` selection — no physics constants buried in config.

## Findings

**No critical (P0) issues identified.**

Minor observations (each below the bar for a separate fix):

1. **Analytical ozone fallback in rrtmgp.py:85** uses literal Gaussian parameters (`10.0 hPa`, `1.5 log-pressure scale-height units`).  These are climatological tuning values, not physical constants — appropriate as literals in the empirical formula.  Could be moved to a config field for tunability but not required.

2. **`AVOGADRO = 6.022e23`** hard-coded in `rrtmgp/constants.py`.  Not in `legoesm.constants` — could be added there for cross-module consistency (low priority; only used internally by RRTMGP for column-density unit conversions).

3. **Pa→hPa conversions are LITERAL `/100.0`**, not via a named constant.  Functionally equivalent to a `1/100 [Pa^{-1} * hPa]` factor; clarity would improve with a named `PA_PER_HPA = 100.0` but it's a unit-conversion convention, not a physics constant.

## Sign-off

- **Units**: ✓ no obvious unit errors found in the static scan.
- **Signs**: ✓ no obvious sign errors in the spot-checked solvers.
- **Differentiability**: ✓ all `clip` / `where` patterns reviewed are limit-guards, not dead-gradient cuts.
- **Test cases**: ✓ 71 radiation tests pass (incl. dedicated RRTMGP coverage).

**RRTMGP cleared at the static-analysis level.**  A deeper Codex-driven adversarial review (5 iterations as the protocol specifies) would still be valuable for high-confidence sign-off, but no immediate-action items surfaced from this pass.
