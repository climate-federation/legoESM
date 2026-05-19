# Bryan 1987 THC spinup -- results skeleton

Acceptance template for the Phase F production Bryan-style THC
spinup. Smoke verified locally; production needs cluster compute.

## Setup

* Driver: ``scripts/ocean_long_runs/run_bryan_thc.py``
* Domain: hemispheric basin 0-60 E, 0-70 N, 4 km flat bottom
* Forcing: CORE-II Normal-Year Forcing (perpetual climatology)
* Bulk fluxes: Large & Yeager 2009
* Resolution: 24x24 (smoke) / 36x36 or 60x60 (production)
* Years: 500-1000 (long spinup)
* Diagnostic cadence: every 10 model years (configurable)
* Restart cadence: at every diagnostic emit

## Acceptance bars (Bryan 1986/1987)

| metric | acceptance | rationale |
|---|---|---|
| MOC strengthens to equilibrium within 500-1000 yr | -- | Bryan 1987 |
| Final NADW analogue transport | 15-20 Sv | Bryan 1987 hemispheric model |
| Heat-content drift after 100 yr | < 1 W/m^2 | spurious-energy budget |
| RPE drift | < 0.5 mW/m^2 | Petersen 2015 reference |
| Multiple-equilibrium stability | single-cell, stable | Bryan 1986 (multiple equilibria for |Sa| > 0.5) |

## Cluster fill-in table (placeholder)

| year | MOC [Sv] | RPE flux [mW/m^2] | heat-content drift [W/m^2] |
|---|---|---|---|
|  10 | ... | ... | ... |
| 100 | ... | ... | ... |
| 250 | ... | ... | ... |
| 500 | ... | ... | ... |
| 1000 | ... | ... | ... |

## Smoke confirmation (local)

```
==> Building Bryan hemispheric basin @ 24x24
   RPE_0 = -3.7683e+24  |  vol_0 = 1.855e+17
==> Year 1/1
Wall time: 0.7s
```

Confirms the driver:
* builds the regional basin grid,
* steps the dycore through a model day,
* emits RPE / energy / tracer diagnostics + restart at the
  configured cadence,
* writes the summary JSON ready for cluster fill-in.

## Known open items

Same as the OMIP-2 driver: the bulk-flux coupling to the state's
surface boundary fields is the remaining piece before a production
run will produce a faithful THC spinup. The dycore + diagnostics +
restart harness are in place.
