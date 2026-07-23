## Findings

- **MEDIUM — CONFIRMED:** Enabling either knob on the default `topography="flat"` configuration is a silent no-op. Flat setup creates an all-zero `f_land`, not `None`; `_run_mpas` only rejects a missing mask, then passes the zero mask into both corrections. The log reports the feature as active even though it changes no cell. `packages/coupler/legoesm/driver/config.py:771`, `packages/coupler/legoesm/driver/model_driver.py:887`, `packages/coupler/legoesm/driver/model_driver.py:5835`, `packages/tools/legoesm/forcing/surface_utils.py:80`, `packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py:200`.

- **MEDIUM — CONFIRMED:** `mpas_land_lapse_K_per_km` is accepted but has no effect for the valid `radiation="none"` configuration. Construction of `forcing["T_sfc"]`—the sole lapse application—is gated on radiation being enabled. Turbulence then falls back to the atmospheric lowest-level temperature, without a lapse-adjusted anchor. `packages/coupler/legoesm/driver/model_driver.py:5952`, `packages/coupler/legoesm/driver/model_driver.py:5966`, `packages/coupler/legoesm/driver/model_driver.py:5978`, `packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:586`.

- **LOW — CONFIRMED:** `mpas_land_beta` is accepted but inert when `turbulence="none"`: the only beta consumer is inside the MPAS turbulence closure, while combined physics omits that closure entirely for a disabled turbulence scheme. `packages/atmosphere/legoesm/atmosphere/physics/combined.py:386`, `packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:602`. This is physically unsurprising—there is no surface latent-flux scheme to modify—but contradicts the change’s fail-loud/no-silent-no-op intent.

- **LOW — CONFIRMED (fail-closed):** Validation identifies the MPAS lane from `dycore.discretization`, while execution dispatches from `grid.grid_type`; there is no cross-field invariant in `validate_strict`. `packages/coupler/legoesm/driver/config.py:1708`, `packages/coupler/legoesm/driver/model_driver.py:5245`. The component factory rejects unsupported grid/discretization tuples before `_run_mpas`, so this is a validation/diagnostic inconsistency rather than a reachable wrong-lane execution. `packages/coupler/legoesm/driver/component_factory.py:75`, `packages/coupler/legoesm/driver/component_factory.py:282`.

- **LOW — PLAUSIBLE:** Sea ice and land fraction are independently sourced; lapse is applied after the SIC blend with no overlap/partition check. Therefore a cell with both nonzero SIC and land fraction cools an already ice-blended ocean anchor on its land fraction. `packages/tools/legoesm/forcing/surface_utils.py:37`, `packages/coupler/legoesm/driver/model_driver.py:5975`. This is only a physical-policy concern: if SIC is a cell-area fraction and fractions are properly partitioned, the linear blend is defensible; the code does not establish that invariant.

## No finding in the remaining requested categories

- Formula signs and units are correct: lapse converts K/km to K/m and subtracts `f_land × lapse × max(z,0)`; beta gives the intended effective humidity gradient. `packages/coupler/legoesm/driver/model_driver.py:5832`, `packages/tools/legoesm/forcing/surface_utils.py:80`, `packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py:200`.

- No JIT/retrace issue found. `f_land` and Python-float `land_beta` are factory closures; the gate is a Python conditional, and the only JAX conversion/cast is inside the enabled trace. `packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:351`, `packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:602`.

- Applying lapse to radiation is consistent with the stated effective-surface-anchor model. In the MPAS combined path, the relevant consumers are radiation and turbulence; both intentionally receive `forcing["T_sfc"]`. `packages/coupler/legoesm/driver/model_driver.py:5978`, `packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:586`.

- Full and no-radiation physics are built with identical `f_land`/beta arguments, and both receive the same daily forcing. `packages/coupler/legoesm/driver/model_driver.py:5852`, `packages/coupler/legoesm/driver/model_driver.py:5875`, `packages/coupler/legoesm/driver/model_driver.py:6449`.

- No production MPAS caller besides the driver was found; direct alternative callers are tests and retain default behavior. No new restart state is introduced; the anchor is deterministically recomputed from checkpointed/static fields and the config contract.

The new tests miss all confirmed no-op configurations above: flat zero-land driver setup, `radiation="none"` with lapse enabled, and `turbulence="none"` with beta enabled.
