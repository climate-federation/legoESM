**D1: A** — Independent snow store keeps the switch orthogonal to the (production-off) liquid interception path; requiring liquid-on would couple two opt-in features needlessly and block the production config.

**D2: A** — fcansno has no other consumer and elai/esai come from masking's exposed-area params; allowing masking-off would silently produce a store with zero radiative effect.

**D3: A** — Buried-stem physics demands exposed area; plain LAI+SAI would intercept snow on buried vegetation, a known CLM5 artifact you're porting away from.

**D4: A** — Project rules mandate conservation; CLM's hvap-only charge loses L_f. Recycling (L_s−L_v)E between latent and sensible conserves column energy exactly.

**D5: A** — Canopy snow is atmospherically held, not soil-supplied; frozen soil throttling it would double-count a limitation that doesn't apply, and D13 needs an uncapped supply to draw the store.

**D6: A** — Frost is a small, distinct flux (deposition vs interception); deferring keeps the diff surface minimal and honestly recorded rather than entangling a second port in this one.

**D7: A** — Albedo must see the same state the radiation absorbed (mirrors the existing masking hand-off); CLM's mid-step fcansno is an ordering artifact of its single-pass structure.

**D8: A** — 0.5 K / 0.05 keeps the smoothed transitions inside physical gradient ranges without smearing the 270.15 K unloading ramp into the freezing transition; 0.1 K / 0.01 approaches the hard switches the config explicitly warns against.

**D9: A** — Bulk-only matches the existing masking restriction and the production snow bucket; layered-pack interaction is unscoped and the interpolation of intercepted snow into layers is undefined CLM behavior anyway.

**D10: A** — Empty cold-start is mass-neutral and matches how a fresh opt-in feature must onboard existing restarts; refusing breaks every existing run.

**D11: A** — Consistent with the established `canopy_snow_masking` / `land_canopy_snow_masking` / `--land-canopy-snow-masking` pattern; no reason to invent a second convention.

**D12: A** — These are structural CLM5 coefficients, not calibrated tunables; exposing them invites drift from the reference the oracle pins at 1e-12.

**D13: A** — Leaf-temperature phase split with CLM's store order is the faithful port and physically right (warm leaves shouldn't sublimate first); all-from-snow contradicts the warm-branch liquid-first order you're porting.

**D14: A** — With gfortran-compiled verbatim reference code, any discrepancy above roundoff is a porting bug; 1e-9 tolerance would mask it.

**D15: A** — The ground pack must only see snow that actually reaches it; forcing snowfall double-counts intercepted mass and breaks the mass ledger.

Unanimous A.
